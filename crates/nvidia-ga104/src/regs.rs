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

pub const READ_ALLOWED: &[Reg] = &[NV_PMC_BOOT_0, NV_PMC_BOOT_42];

/// Smallest BAR0 window that covers every allowed register: one 4 KiB page.
pub const PROBE_WINDOW: u64 = 0x1000;

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
        if !READ_ALLOWED.iter().any(|r| r.offset == reg.offset) {
            return Err(GpuError::NotPermitted(
                "registru BAR0 care nu e pe lista permisa",
            ));
        }
        let v = checked_read32(self.mmio, reg.offset)?;
        self.journal.push(Event::MmioRead {
            bar: 0,
            offset: reg.offset,
            value: v,
        });
        Ok(v)
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
        assert!(READ_ALLOWED.iter().all(|r| r.offset + 4 <= PROBE_WINDOW));
    }
}
