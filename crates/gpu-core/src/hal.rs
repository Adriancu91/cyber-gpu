//! Hardware access traits. The kernel implements them over ECAM and
//! `map_mmio`; host tests implement them with software models (crate
//! `gpu-sim`, every type there is explicitly a SIMULATION).
//!
//! The traits are deliberately narrow: 32-bit accesses only, because that is
//! what `cyber-universe` `pci::Device` offers (`read32`/`write32`) and what
//! NVIDIA BAR0 registers use.

use crate::error::{GpuError, GpuResult};

/// PCI configuration space of one function (4 KiB with ECAM).
pub trait ConfigSpace {
    /// Read the aligned dword at `offset` (`offset & 3` must be 0).
    fn read32(&mut self, offset: u16) -> u32;
    /// Write the aligned dword at `offset`.
    fn write32(&mut self, offset: u16, value: u32);

    fn read16(&mut self, offset: u16) -> u16 {
        let v = self.read32(offset & !3);
        (v >> ((offset & 2) * 8)) as u16
    }

    fn read8(&mut self, offset: u16) -> u8 {
        let v = self.read32(offset & !3);
        (v >> ((offset & 3) * 8)) as u8
    }
}

/// A mapped MMIO window (one BAR, or part of it).
pub trait Mmio {
    /// Length of the mapped window in bytes.
    fn len(&self) -> u64;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Raw volatile 32-bit read. Callers go through [`checked_read32`].
    fn read32_raw(&mut self, offset: u64) -> u32;
    /// Raw volatile 32-bit write. Callers go through [`checked_write32`].
    fn write32_raw(&mut self, offset: u64, value: u32);
}

fn check(len: u64, offset: u64) -> GpuResult<()> {
    if offset & 3 != 0 {
        return Err(GpuError::InvalidArgument("acces MMIO nealiniat"));
    }
    match offset.checked_add(4) {
        Some(end) if end <= len => Ok(()),
        _ => Err(GpuError::OutOfBounds {
            offset,
            len: 4,
            limit: len,
        }),
    }
}

/// Bounds- and alignment-checked read.
pub fn checked_read32<M: Mmio + ?Sized>(m: &mut M, offset: u64) -> GpuResult<u32> {
    check(m.len(), offset)?;
    Ok(m.read32_raw(offset))
}

/// Bounds- and alignment-checked write.
pub fn checked_write32<M: Mmio + ?Sized>(m: &mut M, offset: u64, value: u32) -> GpuResult<()> {
    check(m.len(), offset)?;
    m.write32_raw(offset, value);
    Ok(())
}

/// Monotonic time source, used for every bounded wait.
pub trait Clock {
    fn now_us(&mut self) -> u64;
    fn delay_us(&mut self, us: u32);
}

/// Poll `cond` until it returns `Some`, or fail with `Timeout` after
/// `timeout_us`. Never spins without a bound.
pub fn poll_until<C: Clock + ?Sized, T>(
    clock: &mut C,
    timeout_us: u64,
    step_us: u32,
    mut cond: impl FnMut() -> Option<T>,
) -> GpuResult<T> {
    let start = clock.now_us();
    loop {
        if let Some(v) = cond() {
            return Ok(v);
        }
        let waited = clock.now_us().saturating_sub(start);
        if waited >= timeout_us {
            return Err(GpuError::Timeout { waited_us: waited });
        }
        clock.delay_us(step_us.max(1));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Win([u32; 4]);
    impl Mmio for Win {
        fn len(&self) -> u64 {
            16
        }
        fn read32_raw(&mut self, o: u64) -> u32 {
            self.0[(o / 4) as usize]
        }
        fn write32_raw(&mut self, o: u64, v: u32) {
            self.0[(o / 4) as usize] = v;
        }
    }

    struct FakeClock(u64);
    impl Clock for FakeClock {
        fn now_us(&mut self) -> u64 {
            self.0
        }
        fn delay_us(&mut self, us: u32) {
            self.0 += us as u64;
        }
    }

    #[test]
    fn mmio_bounds_and_alignment() {
        let mut w = Win([1, 2, 3, 4]);
        assert_eq!(checked_read32(&mut w, 12), Ok(4));
        assert!(matches!(
            checked_read32(&mut w, 16),
            Err(GpuError::OutOfBounds { .. })
        ));
        assert!(matches!(
            checked_read32(&mut w, 2),
            Err(GpuError::InvalidArgument(_))
        ));
        assert!(matches!(
            checked_read32(&mut w, u64::MAX - 3),
            Err(GpuError::OutOfBounds { .. })
        ));
        checked_write32(&mut w, 4, 9).unwrap();
        assert_eq!(w.0[1], 9);
    }

    #[test]
    fn poll_until_times_out() {
        let mut c = FakeClock(0);
        let r: GpuResult<()> = poll_until(&mut c, 100, 10, || None);
        assert_eq!(r, Err(GpuError::Timeout { waited_us: 100 }));
        let mut n = 0;
        let r = poll_until(&mut c, 100, 10, || {
            n += 1;
            (n == 3).then_some(n)
        });
        assert_eq!(r, Ok(3));
    }

    #[test]
    fn narrow_config_reads() {
        struct Cfg;
        impl ConfigSpace for Cfg {
            fn read32(&mut self, _o: u16) -> u32 {
                0x2482_10de
            }
            fn write32(&mut self, _o: u16, _v: u32) {}
        }
        assert_eq!(Cfg.read16(0), 0x10de);
        assert_eq!(Cfg.read16(2), 0x2482);
        assert_eq!(Cfg.read8(3), 0x24);
    }
}
