//! Base Address Registers.
//!
//! Decoding is read-only. Sizing follows the PCI 3.0 procedure (§6.2.5.1):
//! write all ones, read back the mask, restore. Because writing a BAR moves
//! the device's decode window, sizing is done with memory and I/O decode
//! disabled in the Command register, and every register is restored and read
//! back afterwards. On the primary GPU the firmware framebuffer may live in a
//! BAR: the caller must guarantee nothing touches the device (interrupts off,
//! single CPU) between the first write and the final restore.

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::ConfigSpace;

use crate::header::{self, CMD_IO, CMD_MEMORY, REG_BAR0, REG_COMMAND};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BarKind {
    Mem32,
    Mem64,
    Io,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bar {
    /// BAR slot (0..=5). A 64-bit BAR occupies `index` and `index + 1`.
    pub index: u8,
    pub kind: BarKind,
    pub prefetchable: bool,
    /// Bus address programmed by firmware (low type bits masked).
    pub base: u64,
    /// Size in bytes, only known after [`size_bars`].
    pub size: Option<u64>,
}

impl Bar {
    pub fn end(&self) -> Option<u64> {
        self.size.and_then(|s| self.base.checked_add(s))
    }
}

fn bar_reg(i: u8) -> u16 {
    REG_BAR0 + 4 * i as u16
}

/// Decode the six BARs of a type-0 header without writing anything.
/// The upper half of a 64-bit BAR is reported as `None`.
pub fn decode_bars<C: ConfigSpace + ?Sized>(cfg: &mut C) -> [Option<Bar>; 6] {
    let mut out = [None; 6];
    if header::header_type(cfg) != 0 {
        return out;
    }
    let mut i = 0u8;
    while i < 6 {
        let lo = cfg.read32(bar_reg(i));
        if lo & 1 == 1 {
            out[i as usize] = Some(Bar {
                index: i,
                kind: BarKind::Io,
                prefetchable: false,
                base: (lo & !0x3) as u64,
                size: None,
            });
            i += 1;
            continue;
        }
        let prefetchable = lo & 0x8 != 0;
        let is64 = (lo >> 1) & 0x3 == 0x2;
        if is64 && i < 5 {
            let hi = cfg.read32(bar_reg(i + 1));
            out[i as usize] = Some(Bar {
                index: i,
                kind: BarKind::Mem64,
                prefetchable,
                base: ((hi as u64) << 32) | (lo & !0xf) as u64,
                size: None,
            });
            i += 2;
        } else {
            out[i as usize] = Some(Bar {
                index: i,
                kind: BarKind::Mem32,
                prefetchable,
                base: (lo & !0xf) as u64,
                size: None,
            });
            i += 1;
        }
    }
    // An unimplemented BAR reads as 0; without sizing we cannot tell it from
    // a BAR at address 0, so keep only BARs with a non-zero base here.
    for b in out.iter_mut() {
        if matches!(b, Some(bar) if bar.base == 0) {
            *b = None;
        }
    }
    out
}

/// Explicit acknowledgement that the caller accepts a configuration-space
/// write sequence with decode disabled. Constructing one is a visible
/// decision at the call site (and in the log).
pub struct SizingPermit {
    pub reason: &'static str,
}

impl SizingPermit {
    pub fn new(reason: &'static str) -> Self {
        SizingPermit { reason }
    }
}

fn logged_write<C: ConfigSpace + ?Sized>(cfg: &mut C, j: &mut Journal, off: u16, v: u32) -> u32 {
    let before = cfg.read32(off);
    cfg.write32(off, v);
    let after = cfg.read32(off);
    j.push(Event::ConfigWrite {
        offset: off,
        before,
        written: v,
        after,
    });
    after
}

/// Size from a read-back mask of a memory BAR (64-bit combined).
pub fn mem_size_from_mask(mask: u64, is64: bool) -> Option<u64> {
    let m = mask & !0xf;
    let m = if is64 { m } else { m | 0xffff_ffff_0000_0000 };
    if m == 0xffff_ffff_0000_0000 && !is64 || m == 0 {
        return None; // BAR not implemented
    }
    Some((!m).wrapping_add(1))
}

/// Size from a read-back mask of an I/O BAR.
pub fn io_size_from_mask(mask: u32) -> Option<u64> {
    let mut m = mask & !0x3;
    if m == 0 {
        return None;
    }
    if m & 0xffff_0000 == 0 {
        m |= 0xffff_0000; // 16-bit I/O decoder
    }
    Some((!m).wrapping_add(1) as u64)
}

/// Determine BAR sizes. Restores the original BAR values and the Command
/// register, verifies them by read-back, and records every write in `j`.
/// On a verification failure the error names the register; the device must
/// then be treated as Faulted.
pub fn size_bars<C: ConfigSpace + ?Sized>(
    cfg: &mut C,
    _permit: &SizingPermit,
    j: &mut Journal,
) -> GpuResult<[Option<Bar>; 6]> {
    if header::header_type(cfg) != 0 {
        return Err(GpuError::InvalidArgument("nu e un antet de tip 0"));
    }
    j.push(Event::Note("dimensionare BAR: decodare oprita"));
    let cmd_dword = cfg.read32(REG_COMMAND);
    let cmd = cmd_dword as u16;
    // Clearing decode bits only: the status half of the dword is RW1C, write 0 there.
    logged_write(cfg, j, REG_COMMAND, (cmd & !(CMD_MEMORY | CMD_IO)) as u32);

    let mut out = [None; 6];
    let mut original = [0u32; 6];
    for (i, o) in original.iter_mut().enumerate() {
        *o = cfg.read32(bar_reg(i as u8));
    }

    let mut i = 0u8;
    while i < 6 {
        let lo = original[i as usize];
        let io = lo & 1 == 1;
        let is64 = !io && (lo >> 1) & 0x3 == 0x2 && i < 5;
        let mask_lo = logged_write(cfg, j, bar_reg(i), 0xffff_ffff);
        let mut mask_hi = 0;
        if is64 {
            mask_hi = logged_write(cfg, j, bar_reg(i + 1), 0xffff_ffff);
        }
        let size = if io {
            io_size_from_mask(mask_lo)
        } else {
            mem_size_from_mask(((mask_hi as u64) << 32) | mask_lo as u64, is64)
        };
        if let Some(size) = size {
            let base = if io {
                (lo & !0x3) as u64
            } else if is64 {
                ((original[i as usize + 1] as u64) << 32) | (lo & !0xf) as u64
            } else {
                (lo & !0xf) as u64
            };
            out[i as usize] = Some(Bar {
                index: i,
                kind: if io {
                    BarKind::Io
                } else if is64 {
                    BarKind::Mem64
                } else {
                    BarKind::Mem32
                },
                prefetchable: !io && lo & 0x8 != 0,
                base,
                size: Some(size),
            });
        }
        i += if is64 { 2 } else { 1 };
    }

    // Restore BARs first, then the Command register, then verify everything.
    for (i, &v) in original.iter().enumerate() {
        logged_write(cfg, j, bar_reg(i as u8), v);
    }
    logged_write(cfg, j, REG_COMMAND, cmd as u32);
    for (i, &v) in original.iter().enumerate() {
        let now = cfg.read32(bar_reg(i as u8));
        if now != v {
            j.push(Event::Step {
                name: "restaurare BAR",
                ok: false,
            });
            return Err(GpuError::VerifyFailed {
                offset: bar_reg(i as u8) as u32,
                expected: v,
                actual: now,
            });
        }
    }
    let now = cfg.read16(REG_COMMAND);
    if now != cmd {
        j.push(Event::Step {
            name: "restaurare COMMAND",
            ok: false,
        });
        return Err(GpuError::VerifyFailed {
            offset: REG_COMMAND as u32,
            expected: cmd as u32,
            actual: now as u32,
        });
    }
    j.push(Event::Step {
        name: "dimensionare BAR",
        ok: true,
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_sim::pci::SimConfig;

    #[test]
    fn mask_math() {
        // 16 MiB 32-bit BAR
        assert_eq!(mem_size_from_mask(0xff00_0000, false), Some(16 << 20));
        // 256 MiB 64-bit BAR
        assert_eq!(
            mem_size_from_mask(0xffff_ffff_f000_000c, true),
            Some(256 << 20)
        );
        // 8 GiB 64-bit BAR
        assert_eq!(
            mem_size_from_mask(0xffff_fffe_0000_000c, true),
            Some(8 << 30)
        );
        assert_eq!(mem_size_from_mask(0, false), None);
        assert_eq!(io_size_from_mask(0xffff_ff81), Some(128));
        assert_eq!(io_size_from_mask(0x0000_ff81), Some(128));
        assert_eq!(io_size_from_mask(0), None);
    }

    #[test]
    fn decode_is_read_only() {
        let mut cfg = SimConfig::ga104_rtx3070ti();
        let before = cfg.writes;
        let bars = decode_bars(&mut cfg);
        assert_eq!(cfg.writes, before);
        let b0 = bars[0].unwrap();
        assert_eq!(
            (b0.kind, b0.base, b0.prefetchable),
            (BarKind::Mem32, 0xf600_0000, false)
        );
        let b1 = bars[1].unwrap();
        assert_eq!((b1.kind, b1.prefetchable), (BarKind::Mem64, true));
        assert!(bars[2].is_none(), "upper half of BAR1");
        assert_eq!(bars[3].unwrap().kind, BarKind::Mem64);
        assert_eq!(bars[5].unwrap().kind, BarKind::Io);
    }

    #[test]
    fn sizing_restores_everything() {
        let mut cfg = SimConfig::ga104_rtx3070ti();
        let snapshot = cfg.snapshot();
        let mut j = Journal::new(256);
        let bars = size_bars(&mut cfg, &SizingPermit::new("test"), &mut j).unwrap();
        assert_eq!(bars[0].unwrap().size, Some(16 << 20));
        assert_eq!(bars[1].unwrap().size, Some(256 << 20));
        assert_eq!(bars[3].unwrap().size, Some(32 << 20));
        assert_eq!(bars[5].unwrap().size, Some(128));
        assert_eq!(
            cfg.snapshot(),
            snapshot,
            "config space identical after sizing"
        );
        assert!(j.writes() > 0);
        assert!(
            !cfg.decode_was_enabled_during_bar_write,
            "BARs written only with decode off"
        );
    }

    #[test]
    fn sizing_detects_failed_restore() {
        let mut cfg = SimConfig::ga104_rtx3070ti();
        cfg.fault_sticky_bar = Some(0);
        let mut j = Journal::new(256);
        let r = size_bars(&mut cfg, &SizingPermit::new("test"), &mut j);
        assert!(matches!(
            r,
            Err(GpuError::VerifyFailed { offset: 0x10, .. })
        ));
    }

    /// SIMULARE: sizes expected on the workstation (K0 inventory).
    #[test]
    fn sizing_simulated_workstation_matches_inventory() {
        use gpu_sim::workstation as ws;
        let mut cfg = SimConfig::workstation_rtx3070ti();
        let before = cfg.snapshot();
        let mut j = Journal::new(64);
        let b = size_bars(&mut cfg, &SizingPermit::new("test"), &mut j).unwrap();
        assert_eq!(b[0].unwrap().size, Some(ws::BAR0_SIZE));
        assert_eq!(b[1].unwrap().size, Some(ws::BAR1_SIZE));
        assert_eq!(b[3].unwrap().size, Some(ws::BAR3_SIZE));
        assert_eq!(b[5].unwrap().size, Some(ws::IO_SIZE));
        assert_eq!(b[1].unwrap().base, ws::BAR1_BASE);
        assert_eq!(cfg.snapshot(), before);
    }
}
