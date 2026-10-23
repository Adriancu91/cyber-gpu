//! BAR0 register allow-list.
//!
//! Policy (docs/RISKS.md, prompt §4.3): only registers listed here may be
//! read, and none may be written. The read-only accessor has no write method,
//! so a write to BAR0 cannot be expressed through this crate. New registers
//! are added one per change (D-013), each with its source.

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Mmio, checked_read32};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reg {
    pub offset: u64,
    pub name: &'static str,
    pub source: &'static str,
}

pub const NV_PMC_BOOT_0: Reg = Reg {
    offset: 0x0000_0000,
    name: "NV_PMC_BOOT_0",
    source: "open-gpu-doc manuals/ampere/ga100/dev_boot.ref.txt (MIT); nouveau device/base.c",
};

pub const NV_PMC_BOOT_42: Reg = Reg {
    offset: 0x0000_0a00,
    name: "NV_PMC_BOOT_42",
    source: "open-gpu-doc manuals/ampere/ga100/dev_boot.ref.txt (MIT); nova-core regs.rs",
};

/// Bit 0 (READ_PROTECTION_LEVEL0) is set once FWSEC has lowered the
/// protection of scratch group 05, i.e. once the next register may be read.
pub const NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK: Reg = Reg {
    offset: 0x0011_8128,
    name: "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK",
    source: "nouveau devinit/tu102.c tu102_devinit_wait (MIT); nova-core gfw.rs (procedure only)",
};

/// Bits 7:0: GFW boot progress, 0xff = completed.
pub const NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT: Reg = Reg {
    offset: 0x0011_8234,
    name: "NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT",
    source: "nouveau devinit/tu102.c tu102_devinit_wait (MIT); nova-core gfw.rs (procedure only)",
};

pub const READ_ALLOWED: &[Reg] = &[
    NV_PMC_BOOT_0,
    NV_PMC_BOOT_42,
    NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK,
    NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT,
];

/// Smallest BAR0 window for level 0 (BOOT_0/BOOT_42): one 4 KiB page.
pub const PROBE_WINDOW: u64 = 0x1000;

/// The VBIOS mirror in BAR0 (PROM). Base from the nova-core VBIOS
/// documentation ("ROM_OFFSET", BAR0 + 0x300000) and nouveau
/// `bios/shadowprom.c`; the 1 MiB length is our bound (DEDUS), reads past it
/// are refused.
pub const PROM_BASE: u64 = 0x0030_0000;
pub const PROM_SIZE: u64 = 0x0010_0000;

/// BAR0 window needed by I2 and I3 (GFW registers + PROM): 4 MiB.
pub const INIT_WINDOW: u64 = PROM_BASE + PROM_SIZE;

// Level 0 only maps the probe window: BOOT_0/BOOT_42 must fit in it.
const _: () = assert!(NV_PMC_BOOT_0.offset + 4 <= PROBE_WINDOW);
const _: () = assert!(NV_PMC_BOOT_42.offset + 4 <= PROBE_WINDOW);

/// NVIDIA returns 0xbadfxxxx for reads of registers the CPU may not access
/// (priv errors). Treated as a hardware fault, never as data.
pub fn is_priv_error(v: u32) -> bool {
    v >> 16 == 0xbadf
}

/// Read-only view of BAR0 restricted to [`READ_ALLOWED`].
pub struct ReadOnlyBar0<'a, M: Mmio + ?Sized> {
    mmio: &'a mut M,
    journal: &'a mut Journal,
}

impl<'a, M: Mmio + ?Sized> ReadOnlyBar0<'a, M> {
    pub fn new(mmio: &'a mut M, journal: &'a mut Journal) -> Self {
        ReadOnlyBar0 { mmio, journal }
    }

    pub fn read(&mut self, reg: Reg) -> GpuResult<u32> {
        let v = self.read_quiet(reg)?;
        self.journal.push(Event::MmioRead {
            bar: 0,
            offset: reg.offset,
            value: v,
        });
        Ok(v)
    }

    /// Same checks as [`Self::read`], without a journal entry (for polling
    /// loops, which journal the first and last values themselves).
    pub fn read_quiet(&mut self, reg: Reg) -> GpuResult<u32> {
        if !READ_ALLOWED.iter().any(|r| r.offset == reg.offset) {
            return Err(GpuError::NotPermitted(
                "registru BAR0 care nu e pe lista permisa",
            ));
        }
        checked_read32(self.mmio, reg.offset)
    }

    /// Read the dword at `rom_offset` inside the PROM window. Not journaled
    /// one by one (a VBIOS scan is thousands of reads); callers count them.
    pub fn read_prom(&mut self, rom_offset: u64) -> GpuResult<u32> {
        if rom_offset & 3 != 0 || rom_offset >= PROM_SIZE {
            return Err(GpuError::OutOfBounds {
                offset: rom_offset,
                len: 4,
                limit: PROM_SIZE,
            });
        }
        checked_read32(self.mmio, PROM_BASE + rom_offset)
    }

    pub fn journal(&mut self) -> &mut Journal {
        self.journal
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_sim::mmio::SimMmio;

    #[test]
    fn allow_list_enforced() {
        let mut m = SimMmio::ga104_bar0();
        let mut j = Journal::new(16);
        let mut bar = ReadOnlyBar0::new(&mut m, &mut j);
        assert_eq!(bar.read(NV_PMC_BOOT_0), Ok(0x1740_00a1));
        let rogue = Reg {
            offset: 0x0000_0200,
            name: "NV_PMC_ENABLE",
            source: "-",
        };
        assert!(matches!(bar.read(rogue), Err(GpuError::NotPermitted(_))));
        assert_eq!(m.reads, [0], "the refused register was never touched");
        assert!(m.writes.is_empty());
        assert_eq!(j.len(), 1);
    }

    #[test]
    fn window_covers_allow_list() {
        assert!(READ_ALLOWED.iter().all(|r| r.offset + 4 <= INIT_WINDOW));
    }

    #[test]
    fn prom_reads_are_bounded() {
        let mut m = SimMmio::new(INIT_WINDOW);
        m.set(PROM_BASE, 0xaa55);
        let mut j = Journal::new(4);
        let mut bar = ReadOnlyBar0::new(&mut m, &mut j);
        assert_eq!(bar.read_prom(0), Ok(0xaa55));
        assert!(bar.read_prom(PROM_SIZE).is_err());
        assert!(bar.read_prom(2).is_err());
        assert_eq!(m.reads, [PROM_BASE]);
        assert!(m.writes.is_empty());
        assert!(is_priv_error(0xbadf_1100) && !is_priv_error(0x1740_00a1));
    }
}
