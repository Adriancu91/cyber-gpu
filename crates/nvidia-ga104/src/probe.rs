//! Level-0 identification. Reads only: configuration header, BAR decode,
//! capability lists, and two BAR0 registers. Zero writes, which the tests
//! assert through the journal and the simulated devices.

use core::fmt;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{ConfigSpace, Mmio};
use gpu_core::ids::{CLASS_DISPLAY, PciAddress};
use gpu_pci::header::CMD_MEMORY;
use gpu_pci::{BarKind, DeviceReport};

use crate::boot::{Boot0, Boot42, chip_name};
use crate::ids::{self, Family};
use crate::regs::{NV_PMC_BOOT_0, NV_PMC_BOOT_42, PROBE_WINDOW, ReadOnlyBar0};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub report: DeviceReport,
    pub model: &'static str,
    /// Bus address of BAR0 (32-bit, non-prefetchable on GA104 boards).
    pub bar0: u64,
}

impl Candidate {
    /// Whether the function decodes its memory BARs (Command bit 1). On the
    /// ASUS laptop the firmware leaves it OFF (observed 12 Nov 2026): BAR0
    /// does not answer until the minimal wake-up (C-116, `gpu_pci::mem_space`).
    pub fn memory_space_enabled(&self) -> bool {
        self.report.command & CMD_MEMORY != 0
    }
}

/// Why BAR0 must not be read yet (C-116): the same text in the journal and
/// in the error, so the OS can show it.
pub const MEMORY_SPACE_OFF: &str = "Memory Space oprit de firmware: BAR0 nu raspunde; registrele nu se citesc (trezirea minima C-116 cere DA)";

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
    check_candidate(report, model.name)
}

/// Stage 1 for GA104 **or** GA106 (C-116). Reads exactly what
/// [`probe_config`] reads; only the id table is wider.
pub fn probe_config_ampere<C: ConfigSpace + ?Sized>(
    cfg: &mut C,
    addr: Option<PciAddress>,
) -> GpuResult<(Candidate, Family)> {
    let report = DeviceReport::read(cfg, addr)?;
    let id = report.id;
    let (family, model) =
        ids::lookup_ampere(id.vendor, id.device).ok_or(GpuError::Unsupported {
            vendor: id.vendor,
            device: id.device,
        })?;
    Ok((check_candidate(report, model.name)?, family))
}

fn check_candidate(report: DeviceReport, model: &'static str) -> GpuResult<Candidate> {
    let id = report.id;
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
        model,
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
    probe_mmio_family(bar0, journal, Family::Ga104)
}

/// Stage 2 for the family chosen by the PCI id: BOOT_0 must report that
/// family's chip id. Same two reads as [`probe_mmio`].
pub fn probe_mmio_family<M: Mmio + ?Sized>(
    bar0: &mut M,
    journal: &mut Journal,
    family: Family,
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
    if boot0.chip_id() != family.chip_id() {
        return Err(GpuError::Blocked(match family {
            Family::Ga104 => "BOOT_0 nu indica GA104 (0x174)",
            Family::Ga106 => "BOOT_0 nu indica GA106 (0x176)",
        }));
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

/// Level 0 for GA104 or GA106 (C-116). Read-only, like [`level0`]; the
/// family says whether later (writing) steps may run: see
/// [`Family::writes_allowed`]. [`level0`] stays GA104-only and remains the
/// gate of the writing chain. One more guard than [`level0`]: when Memory
/// Space is off in Command (the laptop as the firmware leaves it), BAR0 is
/// **not read** — the error names the minimal wake-up (C-116 step 2).
pub fn level0_ampere<C: ConfigSpace + ?Sized, M: Mmio + ?Sized>(
    cfg: &mut C,
    addr: Option<PciAddress>,
    bar0: &mut M,
    journal: &mut Journal,
) -> GpuResult<(Level0, Family)> {
    let candidate = probe_config_ampere(cfg, addr);
    journal.push(Event::Step {
        name: "nivel0: configuratie",
        ok: candidate.is_ok(),
    });
    let (candidate, family) = candidate?;
    if family == Family::Ga106 {
        journal.push(Event::Note(
            "GA106 (C-116): doar citire; lantul cu scrieri ramane doar pentru GA104",
        ));
    }
    if !candidate.memory_space_enabled() {
        journal.push(Event::Note(MEMORY_SPACE_OFF));
        journal.push(Event::Step {
            name: "nivel0: BOOT_0/BOOT_42",
            ok: false,
        });
        return Err(GpuError::Blocked(MEMORY_SPACE_OFF));
    }
    let chip = probe_mmio_family(bar0, journal, family);
    journal.push(Event::Step {
        name: "nivel0: BOOT_0/BOOT_42",
        ok: chip.is_ok(),
    });
    Ok((
        Level0 {
            candidate,
            chip: chip?,
        },
        family,
    ))
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

    /// SIMULARE built from the K0 inventory of the real workstation.
    #[test]
    fn level0_on_simulated_workstation_config() {
        use gpu_sim::workstation as ws;
        for rebar in [true, false] {
            let mut cfg = SimConfig::workstation_rtx3070ti();
            if !rebar {
                cfg = cfg.without_rebar();
            }
            let addr = PciAddress {
                segment: 0,
                bus: ws::BUS,
                dev: 0,
                func: 0,
            };
            let mut bar0 = sim_bar0();
            let mut j = Journal::new(64);
            let l0 = level0(&mut cfg, Some(addr), &mut bar0, &mut j).unwrap();
            let id = l0.candidate.report.id;
            assert_eq!((id.vendor, id.device), (ws::VENDOR, ws::DEVICE));
            assert_eq!(
                (id.subsys_vendor, id.subsys_device),
                (ws::SUBSYS_VENDOR, ws::SUBSYS_DEVICE)
            );
            assert_eq!(id.revision, ws::REVISION);
            assert_eq!(l0.candidate.bar0, ws::BAR0_BASE);
            let bars = &l0.candidate.report.bars;
            assert_eq!(bars[1].unwrap().base, ws::BAR1_BASE);
            assert_eq!(bars[3].unwrap().base, ws::BAR3_BASE);
            assert_eq!(bars[5].unwrap().base, ws::IO_BASE);
            // Everything below 4 GiB, as Windows reported.
            assert!(bars.iter().flatten().all(|b| b.base < 1 << 32));
            let rebar1 = l0.candidate.report.rebar_size(1);
            assert_eq!(rebar1, rebar.then_some(ws::BAR1_SIZE));
            assert_eq!(cfg.writes, 0);
            assert!(bar0.writes.is_empty());
            let text = l0.to_string();
            assert!(text.contains("0000:b3:00.0"));
            assert!(text.contains("subsys 1043:8813"));
            assert!(text.contains("@ 0xf9000000"));
        }
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

    // C-116. All SIMULARE: the laptop profile is built from what `gpu identifica`
    // showed on 12 Nov 2026 (ids, Command 0, BAR0 base) plus public facts
    // (chip id 0x176); no register has been read from the laptop yet.
    #[test]
    fn ga106_laptop_as_firmware_leaves_it_is_refused_before_any_bar0_read() {
        let mut cfg = SimConfig::ga106_laptop_asus();
        let mut bar0 = SimMmio::ga106_bar0();
        let mut j = Journal::new(64);
        let r = level0_ampere(&mut cfg, None, &mut bar0, &mut j);
        assert_eq!(r, Err(GpuError::Blocked(MEMORY_SPACE_OFF)));
        assert!(
            bar0.reads.is_empty(),
            "BAR0 untouched while Memory Space is off"
        );
        assert_eq!(cfg.writes, 0);
        assert_eq!(j.writes(), 0);
        assert!(
            j.events()
                .any(|e| matches!(e, Event::Note(n) if n.contains("Memory Space oprit")))
        );
        // probe_config_ampere alone still identifies the laptop (reads only).
        let (c, fam) = probe_config_ampere(&mut cfg, None).unwrap();
        assert_eq!(fam, Family::Ga106);
        assert_eq!(c.model, "GeForce RTX 3060 Laptop");
        assert_eq!(c.bar0, 0x8500_0000);
        assert!(!c.memory_space_enabled());
        assert_eq!(cfg.writes, 0);
        // The GA104-only gate still refuses the laptop by id.
        assert!(matches!(
            level0(&mut cfg, None, &mut bar0, &mut Journal::new(64)),
            Err(GpuError::Unsupported { .. })
        ));
    }

    #[test]
    fn ga106_laptop_with_memory_space_on_identifies_read_only() {
        let mut cfg = SimConfig::ga106_laptop_asus();
        cfg.write_raw(0x04, 0x0010_0002); // as after the minimal wake-up (model)
        let mut bar0 = SimMmio::ga106_bar0();
        let mut j = Journal::new(64);
        let (l0, fam) = level0_ampere(&mut cfg, None, &mut bar0, &mut j).unwrap();
        assert_eq!(fam, Family::Ga106);
        assert!(!fam.writes_allowed());
        assert_eq!(chip_name(l0.chip.boot0.chip_id()), "GA106");
        assert!(l0.chip.consistent);
        assert_eq!(l0.candidate.model, "GeForce RTX 3060 Laptop");
        assert!(l0.candidate.memory_space_enabled());
        assert_eq!(bar0.reads, [0x0, 0xa00]);
        assert_eq!(j.writes(), 0);
        assert_eq!(cfg.writes, 0);
        assert!(bar0.writes.is_empty());
        assert!(l0.to_string().contains("GA106"));
    }

    #[test]
    fn ampere_level0_still_accepts_the_workstation() {
        let mut cfg = SimConfig::ga104_rtx3070ti();
        let mut bar0 = sim_bar0();
        let mut j = Journal::new(64);
        let (l0, fam) = level0_ampere(&mut cfg, None, &mut bar0, &mut j).unwrap();
        assert_eq!(fam, Family::Ga104);
        assert!(l0.candidate.memory_space_enabled());
        assert_eq!(
            l0,
            level0(&mut cfg, None, &mut bar0, &mut Journal::new(64)).unwrap()
        );
    }

    #[test]
    fn ga106_pci_id_with_other_chip_or_asleep_is_refused() {
        // A "3060 Laptop" PCI id but BOOT_0 says GA104: refuse, do not guess.
        let mut cfg = SimConfig::ga106_laptop_asus();
        cfg.write_raw(0x04, 0x0010_0002);
        let mut bar0 = sim_bar0();
        let mut j = Journal::new(64);
        assert_eq!(
            level0_ampere(&mut cfg, None, &mut bar0, &mut j),
            Err(GpuError::Blocked("BOOT_0 nu indica GA106 (0x176)"))
        );
        // Laptop dGPU powered down (D3): refused before any BAR0 read.
        let mut cfg = SimConfig::ga106_laptop_asus();
        cfg.write_raw(0x04, 0x0010_0002);
        cfg.set_power_state(3);
        let mut bar0 = SimMmio::ga106_bar0();
        assert!(matches!(
            level0_ampere(&mut cfg, None, &mut bar0, &mut Journal::new(64)),
            Err(GpuError::Blocked(_))
        ));
        assert!(bar0.reads.is_empty());
    }
}
