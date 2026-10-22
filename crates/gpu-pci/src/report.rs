//! Reproducible, read-only device report (the `gpu identifica` level 0 output).

use alloc::vec::Vec;
use core::fmt;

use gpu_core::error::GpuResult;
use gpu_core::hal::ConfigSpace;
use gpu_core::ids::{PciAddress, PciId};

use crate::bar::{self, Bar, BarKind};
use crate::caps::{self, PcieLink, ResizableBar};
use crate::header;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceReport {
    pub address: Option<PciAddress>,
    pub id: PciId,
    pub command: u16,
    pub bars: [Option<Bar>; 6],
    pub link: Option<PcieLink>,
    pub power_state: Option<u8>,
    pub resizable: Vec<ResizableBar>,
    pub has_msi: bool,
    pub has_msix: bool,
}

impl DeviceReport {
    /// Build the report without writing to the device.
    pub fn read<C: ConfigSpace + ?Sized>(
        cfg: &mut C,
        address: Option<PciAddress>,
    ) -> GpuResult<Self> {
        let id = header::read_identity(cfg)?;
        let caps = caps::list(cfg);
        Ok(DeviceReport {
            address,
            id,
            command: header::command(cfg),
            bars: bar::decode_bars(cfg),
            link: caps::pcie_link(cfg),
            power_state: caps::power_state(cfg),
            resizable: caps::resizable_bars(cfg),
            has_msi: caps.iter().any(|c| c.id == caps::CAP_MSI),
            has_msix: caps.iter().any(|c| c.id == caps::CAP_MSIX),
        })
    }

    /// Fill in sizes obtained from [`bar::size_bars`].
    pub fn with_sizes(mut self, sized: &[Option<Bar>; 6]) -> Self {
        for (slot, s) in self.bars.iter_mut().zip(sized.iter()) {
            if let (Some(b), Some(s)) = (slot.as_mut(), s) {
                b.size = s.size;
            }
        }
        self
    }

    /// Size of a BAR as reported by Resizable BAR (read-only alternative to sizing).
    pub fn rebar_size(&self, index: u8) -> Option<u64> {
        self.resizable
            .iter()
            .find(|r| r.bar_index == index)
            .map(|r| r.current_bytes())
    }
}

fn mib(b: u64) -> u64 {
    b >> 20
}

impl fmt::Display for DeviceReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(a) = self.address {
            write!(f, "{a} ")?;
        }
        writeln!(f, "{}", self.id)?;
        writeln!(
            f,
            "  command {:#06x} (mem {} bm {})",
            self.command,
            self.command & header::CMD_MEMORY != 0,
            self.command & header::CMD_BUS_MASTER != 0
        )?;
        for b in self.bars.iter().flatten() {
            let kind = match b.kind {
                BarKind::Mem32 => "mem32",
                BarKind::Mem64 => "mem64",
                BarKind::Io => "io",
            };
            write!(
                f,
                "  BAR{} {kind}{} @ {:#x}",
                b.index,
                if b.prefetchable { " pref" } else { "" },
                b.base
            )?;
            match (b.size, self.rebar_size(b.index)) {
                (Some(s), _) if s >= 1 << 20 => writeln!(f, " marime {} MiB", mib(s))?,
                (Some(s), _) => writeln!(f, " marime {s} B")?,
                (None, Some(s)) => writeln!(f, " marime {} MiB (din ReBAR)", mib(s))?,
                (None, None) => writeln!(f, " marime necunoscuta (fara dimensionare)")?,
            }
        }
        if let Some(l) = self.link {
            writeln!(
                f,
                "  PCIe x{} {} (max x{} {})",
                l.width,
                PcieLink::speed_name(l.speed),
                l.max_width,
                PcieLink::speed_name(l.max_speed)
            )?;
        }
        if let Some(p) = self.power_state {
            writeln!(f, "  stare energie D{p}")?;
        }
        for r in &self.resizable {
            writeln!(
                f,
                "  ReBAR BAR{}: curent {} MiB, maxim {} MiB",
                r.bar_index,
                mib(r.current_bytes()),
                r.max_bytes().map(mib).unwrap_or(0)
            )?;
        }
        writeln!(f, "  MSI {} MSI-X {}", self.has_msi, self.has_msix)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use gpu_sim::pci::SimConfig;

    #[test]
    fn report_is_reproducible_and_read_only() {
        let mut cfg = SimConfig::ga104_rtx3070ti();
        let a = DeviceReport::read(&mut cfg, None).unwrap().to_string();
        let b = DeviceReport::read(&mut cfg, None).unwrap().to_string();
        assert_eq!(a, b);
        assert_eq!(cfg.writes, 0);
        assert!(a.contains("10de:2482"));
        assert!(a.contains("BAR1 mem64 pref"));
        assert!(a.contains("(din ReBAR)"));
        assert!(a.contains("PCIe x16 16GT/s"));
    }

    #[test]
    fn absent_device() {
        let mut cfg = SimConfig::absent();
        assert_eq!(
            DeviceReport::read(&mut cfg, None),
            Err(gpu_core::GpuError::NotPresent)
        );
    }
}
