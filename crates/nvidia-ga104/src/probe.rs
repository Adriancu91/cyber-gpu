//! Level-0 identification. Reads only: configuration header, BAR decode,
//! capability lists, and two BAR0 registers. Zero writes, which the tests
//! assert through the journal and the simulated devices.

use core::fmt;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{ConfigSpace, Mmio};
use gpu_core::ids::{CLASS_DISPLAY, PciAddress};
use gpu_pci::{BarKind, DeviceReport};

use crate::boot::{Boot0, Boot42, CHIP_GA104, chip_name};
use crate::ids;
use crate::regs::{NV_PMC_BOOT_0, NV_PMC_BOOT_42, PROBE_WINDOW, ReadOnlyBar0};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub report: DeviceReport,
    pub model: &'static str,
    /// Bus address of BAR0 (32-bit, non-prefetchable on GA104 boards).
    pub bar0: u64,
}

/// Stage 1: configuration space only.
pub fn probe_config<C: ConfigSpace + ?Sized>(
    cfg: &mut C,
    addr: Option<PciAddress>,
) -> GpuResult<Candidate> {
    let report = DeviceReport::read(cfg, addr)?;
    let id = report.id;
    let model = ids::lookup(id.vendor, id.device).ok_or(GpuError::Unsupported {
        vendor: id.vendor,
        device: id.device,
    })?;
    if id.class != CLASS_DISPLAY {
        return Err(GpuError::Unsupported {
            vendor: id.vendor,
            device: id.device,
        });
    }
    let bar0 = match report.bars[0] {
        Some(b) if b.kind == BarKind::Mem32 && !b.prefetchable => b.base,
        _ => {
            return Err(GpuError::Blocked(
                "BAR0 lipsa sau neasteptat (nu e mem32 neprefetchable)",
            ));
        }
    };
    if report.power_state.is_some_and(|d| d != 0) {
        return Err(GpuError::Blocked(
            "placa nu e in D0; nivelul 0 nu schimba starea de energie",
        ));
    }
    Ok(Candidate {
        report,
        model: model.name,
        bar0,
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChipIdentity {
    pub boot0: Boot0,
    pub boot42: Boot42,
    /// BOOT_0 and BOOT_42 agree on the chip id.
    pub consistent: bool,
}

/// Stage 2: read BOOT_0 and BOOT_42 through the read-only accessor.
/// `bar0` must map at least [`PROBE_WINDOW`] bytes, uncached.
pub fn probe_mmio<M: Mmio + ?Sized>(
    bar0: &mut M,
    journal: &mut Journal,
) -> GpuResult<ChipIdentity> {
    if bar0.len() < PROBE_WINDOW {
        return Err(GpuError::InvalidArgument("fereastra BAR0 prea mica"));
    }
    let raw0 = ReadOnlyBar0::new(bar0, journal).read(NV_PMC_BOOT_0)?;
    if raw0 == 0xffff_ffff {
        journal.push(Event::Note(
            "BOOT_0 = 0xffffffff: placa nu raspunde (D3cold / legatura cazuta)",
        ));
        return Err(GpuError::NotResponding);
    }
    let raw42 = ReadOnlyBar0::new(bar0, journal).read(NV_PMC_BOOT_42)?;
    let boot0 = Boot0::decode(raw0);
    let boot42 = Boot42::decode(raw42);
    if boot0.chip_id() != CHIP_GA104 {
        return Err(GpuError::Blocked("BOOT_0 nu indica GA104 (0x174)"));
    }
    Ok(ChipIdentity {
        boot0,
        boot42,
        consistent: boot42.chip_id == boot0.chip_id(),
    })
}

/// Full level-0 result.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Level0 {
    pub candidate: Candidate,
    pub chip: ChipIdentity,
}

impl fmt::Display for Level0 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "gpu nivel 0: {} ({})",
            self.candidate.model,
            chip_name(self.chip.boot0.chip_id())
        )?;
        write!(f, "{}", self.candidate.report)?;
        writeln!(
            f,
            "  NV_PMC_BOOT_0  = {:#010x} arch {:#x} impl {:#x} rev {:x}.{:x}",
            self.chip.boot0.raw,
            self.chip.boot0.architecture,
            self.chip.boot0.implementation,
            self.chip.boot0.major_revision,
            self.chip.boot0.minor_revision
        )?;
        writeln!(
            f,
            "  NV_PMC_BOOT_42 = {:#010x} chip {:#x} bit29 {} consecvent {}",
            self.chip.boot42.raw,
            self.chip.boot42.chip_id,
            self.chip.boot42.bit29,
            self.chip.consistent
        )
    }
}

/// Convenience: both stages, recording a step event for each.
pub fn level0<C: ConfigSpace + ?Sized, M: Mmio + ?Sized>(
    cfg: &mut C,
    addr: Option<PciAddress>,
    bar0: &mut M,
    journal: &mut Journal,
) -> GpuResult<Level0> {
    let candidate = probe_config(cfg, addr);
    journal.push(Event::Step {
        name: "nivel0: configuratie",
        ok: candidate.is_ok(),
    });
    let candidate = candidate?;
    let chip = probe_mmio(bar0, journal);
    journal.push(Event::Step {
        name: "nivel0: BOOT_0/BOOT_42",
        ok: chip.is_ok(),
    });
    Ok(Level0 {
        candidate,
        chip: chip?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;
    use gpu_sim::mmio::SimMmio;
    use gpu_sim::pci::SimConfig;

    fn sim_bar0() -> SimMmio {
        let mut m = SimMmio::ga104_bar0();
        m.set(0xa00, (0x174 << 20) | (0xa << 16) | (0x1 << 12));
        m
    }

    #[test]
    fn level0_on_simulated_board_is_write_free() {
        let mut cfg = SimConfig::ga104_rtx3070ti();
        let mut bar0 = sim_bar0();
        let mut j = Journal::new(64);
        let l0 = level0(&mut cfg, None, &mut bar0, &mut j).unwrap();
        assert_eq!(l0.candidate.model, "GeForce RTX 3070 Ti");
        assert_eq!(l0.candidate.bar0, 0xf600_0000);
        assert!(l0.chip.consistent);
        assert_eq!(cfg.writes, 0);
        assert!(bar0.writes.is_empty());
        assert_eq!(bar0.reads, [0x0, 0xa00]);
        assert_eq!(j.writes(), 0);
        let text = l0.to_string();
        assert!(text.contains("GA104"));
        assert!(text.contains("NV_PMC_BOOT_0  = 0x174000a1"));
    }

    #[test]
    fn rejects_wrong_devices() {
        let mut cfg = SimConfig::ga104_rtx3070ti();
        cfg.set_device_id(0x2204); // GA102
        assert!(matches!(
            probe_config(&mut cfg, None),
            Err(GpuError::Unsupported { .. })
        ));
        let mut cfg = SimConfig::absent();
        assert_eq!(probe_config(&mut cfg, None), Err(GpuError::NotPresent));
        let mut cfg = SimConfig::ga104_rtx3070ti();
        cfg.set_power_state(3);
        assert!(matches!(
            probe_config(&mut cfg, None),
            Err(GpuError::Blocked(_))
        ));
    }

    #[test]
    fn mmio_failures() {
        let mut j = Journal::new(16);
        let mut dead = sim_bar0();
        dead.absent = true;
        assert_eq!(probe_mmio(&mut dead, &mut j), Err(GpuError::NotResponding));
        assert_eq!(dead.reads, [0], "stops after the first all-ones read");
        let mut other = SimMmio::new(16 << 20);
        other.set(0, 0x1720_00a1); // GA102
        assert!(matches!(
            probe_mmio(&mut other, &mut j),
            Err(GpuError::Blocked(_))
        ));
        let mut tiny = SimMmio::new(0x100);
        assert!(matches!(
            probe_mmio(&mut tiny, &mut j),
            Err(GpuError::InvalidArgument(_))
        ));
        let mut inconsistent = SimMmio::ga104_bar0();
        inconsistent.set(0xa00, 0x1720_0000);
        assert!(!probe_mmio(&mut inconsistent, &mut j).unwrap().consistent);
    }
}
