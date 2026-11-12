//! C-54 (2): the read-only tour for the automatic loop.
//!
//! Adrian's decision of 1 Nov 19:45 ("da bucla automata"): Cyber Universe
//! may run tests on the workstation without him present, **only tests that
//! read**. This module is the GPU part of that list: every read below has
//! already been done on the real board with 0 writes (28 Oct – 1 Nov), or
//! is a register read the I6 pre-check needs (marked `pe_placa: false`).
//!
//! [`run_tour`] runs the reads through the same read-only code paths the OS
//! already uses (`probe`, `gfw`, `vbios`, `fwsec`, `i6_check`) and returns
//! [`Fact`]s — `key = value` pairs in ASCII that the OS prints as
//! `fapt <key> = <value>` lines. [`crate::board_compare`] reads those lines
//! (and the older report lines) back and compares them with the same tour
//! run on the `gpu-sim` model. A tour that wrote anything is an error, not
//! a result.

use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, ConfigSpace, Mmio};
use gpu_core::ids::PciAddress;
use gpu_pci::bar::BarKind;

use crate::boot::chip_name;
use crate::firmware::GSP_570_144;
use crate::regs::INIT_WINDOW;

/// One read of the tour, as documentation and as the OS allow-list source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SafeRead {
    pub id: &'static str,
    pub what: &'static str,
    /// Already read on the real board with 0 writes.
    pub pe_placa: bool,
}

/// The tour, in order. Nothing here writes: config space reads, BAR0 reads
/// of allow-listed registers, PROM reads (VBIOS, never stored).
pub const SAFE_READS: &[SafeRead] = &[
    SafeRead {
        id: "pci",
        what: "configuratia PCI: id, subsistem, clasa, Command, BAR-uri, stare D, legatura PCIe, Resizable BAR",
        pe_placa: true,
    },
    SafeRead {
        id: "boot",
        what: "NV_PMC_BOOT_0 (0x0) si NV_PMC_BOOT_42 (0xa00): identitatea cipului",
        pe_placa: true,
    },
    SafeRead {
        id: "gfw",
        what: "PRIV_LEVEL_MASK (0x118128) si GFW_BOOT (0x118234): firmware-ul placii a terminat pornirea",
        pe_placa: true,
    },
    SafeRead {
        id: "vbios",
        what: "VBIOS prin PROM (0x300000): imaginile, tabela BIT (doar pozitii si marimi in raport)",
        pe_placa: true,
    },
    SafeRead {
        id: "fwsec",
        what: "tabela PMU si descriptorul FWSEC_PROD din VBIOS (doar pozitii si marimi; ucode-ul nu se citeste)",
        pe_placa: true,
    },
    SafeRead {
        id: "i6",
        what: "pregatirea I6: siguranta ucode (0x8241e0), marimea VRAM (0x1183a4), WPR2 (0x1fa828/0x1fa824)",
        pe_placa: false,
    },
];

/// Candidate reads that are NOT in the tour, with the reason.
pub const NOT_YET: &[(&str, &str)] = &[(
    "temperatura",
    "registrul de temperatura GA104 nu apare in antetele NVIDIA MIT 570.144; doar in nouveau (de verificat sursa) — intra in tur doar dupa ce sursa e confirmata si e citit o data cu Adrian de fata",
)];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fact {
    pub key: String,
    pub value: String,
}

impl Fact {
    pub fn new(key: impl Into<String>, value: impl Into<String>) -> Self {
        Fact {
            key: key.into(),
            value: value.into(),
        }
    }
}

impl fmt::Display for Fact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fapt {} = {}", self.key, self.value)
    }
}

/// Key of a BAR0 register fact, e.g. `bar0.0x118128`.
pub fn bar0_key(offset: u64) -> String {
    format!("bar0.{offset:#08x}")
}

/// Key/value helpers shared with [`crate::board_compare`], so both sides
/// format the same thing the same way.
pub(crate) fn bar_value(base: u64, kind: BarKind, prefetch: bool) -> String {
    match kind {
        BarKind::Io => format!("{base:#x} io"),
        BarKind::Mem32 => format!("{base:#x} mem{}", if prefetch { " prefetch" } else { "" }),
        BarKind::Mem64 => format!(
            "{base:#x} mem 64b{}",
            if prefetch { " prefetch" } else { "" }
        ),
    }
}

pub(crate) fn image_value(offset: u32, code: u8, size: u32, vendor: u16, device: u16) -> String {
    format!("{offset:#07x} tip {code:#04x} {size} octeti {vendor:04x}:{device:04x}")
}

pub(crate) fn desc_value(v: u8, at: u32, hdr: u32, stored: u32, imem: u32, dmem: u32) -> String {
    format!("v{v} la {at:#x} antet {hdr:#x} ucode {stored:#x} imem {imem:#x} dmem {dmem:#x}")
}

pub(crate) fn sig_value(count: u8, mask: u16, engines: u16, ucode_id: u8) -> String {
    format!("{count} masca {mask:#06x} motoare {engines:#06x} ucode_id {ucode_id}")
}

pub(crate) fn yes(b: bool) -> &'static str {
    if b { "da" } else { "nu" }
}

/// What a tour found.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tour {
    pub facts: Vec<Fact>,
    /// Read accesses journaled (config + BAR0); PROM reads are counted apart.
    pub reads: usize,
}

impl Tour {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.facts
            .iter()
            .rev()
            .find(|f| f.key == key)
            .map(|f| f.value.as_str())
    }
}

/// Runs the tour. `bar0` is the mapped BAR0 window (16 MiB for the I6
/// part; smaller windows skip the parts that need more, with a fact saying
/// so). Stops only if the board does not answer the identification.
/// Returns `Err` if any write was journaled — which the code paths used
/// here cannot do; the check is there so a future change cannot slip one in.
pub fn run_tour<C, M, K>(
    cfg: &mut C,
    addr: Option<PciAddress>,
    bar0: &mut M,
    clock: &mut K,
    journal: &mut Journal,
) -> GpuResult<Tour>
where
    C: ConfigSpace + ?Sized,
    M: Mmio + ?Sized,
    K: Clock + ?Sized,
{
    tour(cfg, addr, bar0, clock, journal, false)
}

/// C-116: the same tour, accepting GA104 **or** GA106 (the laptop). Same
/// reads, same 0-write check. For GA106 two facts are added:
/// `chip.familie = GA106` and `chip.scrieri = interzise ...`. For GA104
/// the result is identical to [`run_tour`] plus `chip.familie = GA104`.
/// With Memory Space off (the laptop as the firmware leaves it) it stops
/// before any BAR0 read, with [`crate::probe::MEMORY_SPACE_OFF`]: the
/// caller wakes the function first (`gpu_pci::mem_space`, behind a permit).
pub fn run_tour_ampere<C, M, K>(
    cfg: &mut C,
    addr: Option<PciAddress>,
    bar0: &mut M,
    clock: &mut K,
    journal: &mut Journal,
) -> GpuResult<Tour>
where
    C: ConfigSpace + ?Sized,
    M: Mmio + ?Sized,
    K: Clock + ?Sized,
{
    tour(cfg, addr, bar0, clock, journal, true)
}

/// What the wake-up did around a tour (C-116).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wake {
    /// Memory Space was already on: the tour ran as [`run_tour_ampere`], no write.
    NotNeeded,
    /// Memory Space was turned on for the tour (one Command write) and the
    /// old Command value was put back after it (one more write).
    EnabledAndRestored,
}

/// Fact key written by [`run_tour_awake`].
pub const FACT_WAKE: &str = "pci.trezire";

/// C-116 step 2, the "minimal wake-up" around the read tour, for a GPU the
/// firmware left with Memory Space OFF (Adrian's laptop). Reads only the
/// configuration space first; if the function already decodes memory the
/// tour runs as [`run_tour_ampere`] with no write. Otherwise, behind the
/// permit, it turns Memory Space on (`gpu_pci::mem_space`: one Command
/// write, read back), runs the tour, and **always** puts Command back —
/// also when the tour fails — so the board is left as the firmware had
/// it. The tour's own 0-write check still holds between the two writes.
/// The fact `pci.trezire` says which case happened. Nothing here has run
/// on the laptop; SIMULATION only.
pub fn run_tour_awake<C, M, K>(
    cfg: &mut C,
    addr: Option<PciAddress>,
    bar0: &mut M,
    clock: &mut K,
    journal: &mut Journal,
    permit: &gpu_pci::mem_space::MemorySpacePermit,
) -> GpuResult<(Tour, Wake)>
where
    C: ConfigSpace + ?Sized,
    M: Mmio + ?Sized,
    K: Clock + ?Sized,
{
    let (candidate, _family) = crate::probe::probe_config_ampere(cfg, addr)?;
    if candidate.memory_space_enabled() {
        let mut t = run_tour_ampere(cfg, addr, bar0, clock, journal)?;
        t.facts.push(Fact::new(
            FACT_WAKE,
            "nu a fost nevoie (Memory Space era pornit)",
        ));
        return Ok((t, Wake::NotNeeded));
    }
    journal.push(Event::Note(
        "C-116 trezire minima: Memory Space oprit de firmware; il pornesc pentru tur si il repun dupa",
    ));
    let old = gpu_pci::mem_space::enable_memory_space(cfg, permit, journal)?;
    let result = run_tour_ampere(cfg, addr, bar0, clock, journal);
    let restored = gpu_pci::mem_space::restore_memory_space(cfg, permit, journal, old);
    let mut t = result?;
    restored?;
    t.facts.push(Fact::new(
        FACT_WAKE,
        "Memory Space pornit de noi pentru tur (C-116, o scriere in Command), repus la loc dupa",
    ));
    Ok((t, Wake::EnabledAndRestored))
}

fn tour<C, M, K>(
    cfg: &mut C,
    addr: Option<PciAddress>,
    bar0: &mut M,
    clock: &mut K,
    journal: &mut Journal,
    ampere: bool,
) -> GpuResult<Tour>
where
    C: ConfigSpace + ?Sized,
    M: Mmio + ?Sized,
    K: Clock + ?Sized,
{
    let writes_before = journal.writes();
    let len_before = journal.len();
    let mut facts = Vec::new();
    journal.push(Event::Note(
        "tur de citire (C-54): incepe, 0 scrieri permise",
    ));

    // pci + boot
    let l0 = if ampere {
        let (l0, family) = crate::probe::level0_ampere(cfg, addr, bar0, journal)?;
        facts.push(Fact::new("chip.familie", family.name()));
        if !family.writes_allowed() {
            facts.push(Fact::new(
                "chip.scrieri",
                "interzise (C-116: doar citire pana la DA-ul lui Adrian)",
            ));
        }
        l0
    } else {
        crate::probe::level0(cfg, addr, bar0, journal)?
    };
    let r = &l0.candidate.report;
    let id = r.id;
    facts.push(Fact::new(
        "pci.id",
        format!("{:04x}:{:04x}", id.vendor, id.device),
    ));
    facts.push(Fact::new(
        "pci.subsistem",
        format!("{:04x}:{:04x}", id.subsys_vendor, id.subsys_device),
    ));
    facts.push(Fact::new(
        "pci.clasa",
        format!("{:02x}{:02x}{:02x}", id.class, id.subclass, id.prog_if),
    ));
    facts.push(Fact::new("pci.command", format!("{:#06x}", r.command)));
    facts.push(Fact::new("pci.bus_master", yes(r.command & 0x4 != 0)));
    if let Some(d) = r.power_state {
        facts.push(Fact::new("pci.stare", format!("D{d}")));
    }
    for b in r.bars.iter().flatten() {
        facts.push(Fact::new(
            format!("pci.bar{}", b.index),
            bar_value(b.base, b.kind, b.prefetchable),
        ));
    }
    for rb in &r.resizable {
        facts.push(Fact::new(
            format!("pci.rebar.bar{}", rb.bar_index),
            format!("{} MiB", rb.current_bytes() >> 20),
        ));
    }
    if let Some(l) = r.link {
        facts.push(Fact::new(
            "pci.legatura",
            format!(
                "gen{} x{} (max gen{} x{})",
                l.speed, l.width, l.max_speed, l.max_width
            ),
        ));
    }
    facts.push(Fact::new(
        bar0_key(0),
        format!("{:#010x}", l0.chip.boot0.raw),
    ));
    facts.push(Fact::new(
        bar0_key(0xa00),
        format!("{:#010x}", l0.chip.boot42.raw),
    ));
    facts.push(Fact::new("chip.nume", chip_name(l0.chip.boot0.chip_id())));
    facts.push(Fact::new("chip.consecvent", yes(l0.chip.consistent)));

    // gfw
    if bar0.len() < INIT_WINDOW {
        facts.push(Fact::new("gfw", "sarit: fereastra BAR0 sub 4 MiB"));
    } else {
        match crate::gfw::wait_gfw_boot(bar0, journal, clock, crate::gfw::GFW_TIMEOUT_US) {
            Ok(s) => {
                facts.push(Fact::new("gfw.progres", format!("{:#04x}", s.progress)));
                facts.push(Fact::new("gfw.runde", s.polls.to_string()));
                facts.push(Fact::new(
                    bar0_key(0x118128),
                    format!("{:#010x}", s.priv_level_mask),
                ));
            }
            Err(e) => facts.push(Fact::new("gfw", format!("eroare: {e}"))),
        }
    }

    // vbios + fwsec
    let mut desc = None;
    if bar0.len() >= INIT_WINDOW {
        const BUDGET_US: u64 = 2_000_000;
        match crate::vbios::read_layout(bar0, journal, clock, BUDGET_US) {
            Ok(lay) => {
                facts.push(Fact::new("vbios.ifr", yes(lay.ifr_version.is_some())));
                facts.push(Fact::new("vbios.imagini", lay.images.len().to_string()));
                for (n, i) in lay.images.iter().enumerate() {
                    facts.push(Fact::new(
                        format!("vbios.imagine{n}"),
                        image_value(i.offset, i.code_type, i.size, i.vendor, i.device),
                    ));
                }
                if let Some((at, toks)) = &lay.bit {
                    facts.push(Fact::new("vbios.bit", format!("{at:#x} {}", toks.len())));
                    let t: Vec<String> = toks
                        .iter()
                        .map(|t| format!("{:#04x}v{}", t.id, t.version))
                        .collect();
                    facts.push(Fact::new("vbios.bit.jetoane", t.join(" ")));
                }
                match crate::fwsec::locate(bar0, journal, clock, BUDGET_US, &lay) {
                    Ok(loc) => {
                        facts.push(Fact::new("fwsec.pmu", format!("{:#x}", loc.pmu_table)));
                        let a: Vec<String> =
                            loc.pmu_apps.iter().map(|a| format!("{a:#04x}")).collect();
                        facts.push(Fact::new("fwsec.aplicatii", a.join(" ")));
                        let d = &loc.desc;
                        facts.push(Fact::new(
                            "fwsec.descriptor",
                            desc_value(
                                d.version(),
                                loc.desc_prom,
                                d.header_size(),
                                d.stored_size,
                                d.imem_load_size,
                                d.dmem_load_size,
                            ),
                        ));
                        facts.push(Fact::new(
                            "fwsec.semnaturi",
                            sig_value(
                                d.signature_count,
                                d.signature_versions,
                                d.engine_id_mask,
                                d.ucode_id,
                            ),
                        ));
                        desc = Some(loc.desc);
                    }
                    Err(e) => facts.push(Fact::new("fwsec", format!("eroare: {e}"))),
                }
            }
            Err(e) => facts.push(Fact::new("vbios", format!("eroare: {e}"))),
        }
    } else {
        facts.push(Fact::new("vbios", "sarit: fereastra BAR0 sub 4 MiB"));
    }

    // i6 pre-check (read-only)
    match desc {
        None => facts.push(Fact::new("i6", "sarit: fara descriptorul FWSEC")),
        Some(_) if bar0.len() < 16 << 20 => {
            facts.push(Fact::new("i6", "sarit: fereastra BAR0 sub 16 MiB"))
        }
        Some(d) => {
            let bl = GSP_570_144[2].size;
            let elf = GSP_570_144[3].size;
            match crate::i6_check::i6_check(bar0, journal, &d, bl, elf) {
                Ok(c) => {
                    facts.push(Fact::new(
                        "i6.siguranta",
                        format!(
                            "{:#x} versiunea {} semnatura {}",
                            c.signature.fuse_offset, c.signature.fuse_version, c.signature.index
                        ),
                    ));
                    facts.push(Fact::new("i6.vram", format!("{} MiB", c.fb_size >> 20)));
                    facts.push(Fact::new(
                        "i6.wpr2",
                        match c.wpr2 {
                            None => String::from("coborat"),
                            Some((s, e)) => format!("RIDICAT {s:#x}..{e:#x}"),
                        },
                    ));
                    facts.push(Fact::new("i6.trece", yes(c.passes())));
                    facts.push(Fact::new("i6.wpr2.etapa", c.wpr2_stage()));
                }
                Err(e) => facts.push(Fact::new("i6", format!("eroare: {e}"))),
            }
        }
    }

    // Every register value read in this tour (BAR0 and config), as facts:
    // the comparison sees the raw values too, not only our decoding.
    let mut reads = 0;
    for e in journal.events().skip(len_before) {
        match *e {
            Event::MmioRead {
                bar: 0,
                offset,
                value,
            } => {
                reads += 1;
                if !(crate::regs::PROM_BASE..crate::regs::PROM_BASE + crate::regs::PROM_SIZE)
                    .contains(&offset)
                {
                    facts.push(Fact::new(bar0_key(offset), format!("{value:#010x}")));
                }
            }
            Event::MmioRead { .. } | Event::ConfigRead { .. } => reads += 1,
            _ => {}
        }
    }

    if journal.writes() != writes_before {
        return Err(GpuError::NotPermitted(
            "turul de citire a jurnalizat o scriere: rezultat respins",
        ));
    }
    journal.push(Event::Note("tur de citire (C-54): gata, 0 scrieri"));
    Ok(Tour { facts, reads })
}

#[cfg(test)]
mod tests {
    //! SIMULARE: the tour on the observed-workstation model (gpu-sim).
    use super::*;
    use gpu_sim::clock::SimClock;
    use gpu_sim::observed;

    #[test]
    fn tour_on_observed_model_reads_only_and_matches_the_board() {
        let mut cfg = observed::observed_config();
        let mut bar0 = observed::observed_bar0();
        let mut clock = SimClock::default();
        let mut j = Journal::new(4096);
        let t = run_tour(&mut cfg, None, &mut bar0, &mut clock, &mut j).unwrap();
        assert_eq!(cfg.writes, 0);
        assert!(bar0.writes.is_empty(), "the tour wrote BAR0");
        assert_eq!(j.writes(), 0);
        // What the board showed (hub rezultate/, 28-29 Oct).
        for (k, v) in [
            ("pci.id", "10de:2482"),
            ("pci.subsistem", "1043:8813"),
            ("pci.command", "0x0003"),
            ("pci.bus_master", "nu"),
            ("pci.stare", "D0"),
            ("pci.bar0", "0xf9000000 mem"),
            ("pci.bar1", "0xe0000000 mem 64b prefetch"),
            ("pci.bar3", "0xf0000000 mem 64b prefetch"),
            ("pci.bar5", "0xf000 io"),
            ("pci.rebar.bar0", "16 MiB"),
            ("pci.rebar.bar1", "256 MiB"),
            ("pci.rebar.bar3", "32 MiB"),
            ("bar0.0x000000", "0xb74000a1"),
            ("bar0.0x000a00", "0x174a1000"),
            ("bar0.0x118128", "0x00008b8f"),
            ("chip.nume", "GA104"),
            ("chip.consecvent", "da"),
            ("gfw.progres", "0xff"),
            ("gfw.runde", "1"),
            ("vbios.ifr", "nu"),
            ("vbios.imagini", "4"),
            ("vbios.imagine0", "0x00000 tip 0x00 65024 octeti 10de:2482"),
            ("vbios.imagine1", "0x0fe00 tip 0x03 83968 octeti 10de:2482"),
            ("vbios.imagine2", "0x24600 tip 0xe0 22016 octeti 10de:2200"),
            ("vbios.imagine3", "0x29c00 tip 0xe0 386048 octeti 10de:2200"),
            ("vbios.bit", "0x1b0 17"),
            ("fwsec.pmu", "0x87ecb"),
            (
                "fwsec.aplicatii",
                "0x01 0x00 0x00 0x00 0x00 0x07 0x08 0x00 0x45 0x85 0x49 0x89 0x00 0x00 0x00 0x00",
            ),
            (
                "fwsec.descriptor",
                "v3 la 0x41208 antet 0x4ac ucode 0xea00 imem 0xe200 dmem 0x800",
            ),
            (
                "fwsec.semnaturi",
                "3 masca 0x0007 motoare 0x0400 ucode_id 9",
            ),
        ] {
            assert_eq!(t.get(k), Some(v), "{k}");
        }
        // The I6 registers are still SIM (0): the fuse read gives no
        // usable version, so the pre-check reports an error — honestly.
        assert!(t.get("i6").is_some_and(|v| v.starts_with("eroare")));
        assert!(
            t.facts
                .iter()
                .all(|f| f.value.is_ascii() && f.key.is_ascii())
        );
    }

    #[test]
    fn tour_with_small_window_skips_instead_of_guessing() {
        let mut cfg = observed::observed_config();
        let mut bar0 = gpu_sim::mmio::SimMmio::new(0x1000);
        bar0.set(0, observed::BOOT_0);
        bar0.set(0xa00, observed::BOOT_42);
        let mut j = Journal::new(64);
        let t = run_tour(&mut cfg, None, &mut bar0, &mut SimClock::default(), &mut j).unwrap();
        assert_eq!(t.get("gfw"), Some("sarit: fereastra BAR0 sub 4 MiB"));
        assert_eq!(t.get("vbios"), Some("sarit: fereastra BAR0 sub 4 MiB"));
        assert!(t.get("i6").unwrap().starts_with("sarit"));
    }

    #[test]
    fn tour_stops_when_the_board_does_not_answer() {
        let mut cfg = observed::observed_config();
        let mut bar0 = observed::observed_bar0();
        bar0.absent = true;
        let mut j = Journal::new(64);
        assert!(run_tour(&mut cfg, None, &mut bar0, &mut SimClock::default(), &mut j).is_err());
    }

    #[test]
    fn safe_reads_list_is_honest() {
        // Only the I6 reads are not yet proven on the board.
        let unproven: Vec<_> = SAFE_READS
            .iter()
            .filter(|r| !r.pe_placa)
            .map(|r| r.id)
            .collect();
        assert_eq!(unproven, ["i6"]);
        assert!(NOT_YET.iter().any(|(id, _)| *id == "temperatura"));
    }

    /// C-116, SIMULARE: the read tour on a GA106 laptop model built from the
    /// workstation model with only the ids changed (VBIOS, fuses, VRAM size
    /// are the workstation's SIM values, not the laptop's).
    #[test]
    fn ampere_tour_on_ga106_laptop_model_reads_only() {
        let mut cfg = observed::observed_config();
        cfg.set_device_id(0x2520);
        let mut bar0 = observed::observed_bar0();
        bar0.set(0, 0xb760_00a1);
        bar0.set(0xa00, 0x176a_1000);
        let mut j = Journal::new(4096);
        let t =
            run_tour_ampere(&mut cfg, None, &mut bar0, &mut SimClock::default(), &mut j).unwrap();
        assert_eq!(t.get("chip.nume"), Some("GA106"));
        assert_eq!(t.get("chip.familie"), Some("GA106"));
        assert!(t.get("chip.scrieri").unwrap().starts_with("interzise"));
        assert_eq!(t.get("pci.id"), Some("10de:2520"));
        assert_eq!(j.writes(), 0);
        assert_eq!(cfg.writes, 0);
        assert!(bar0.writes.is_empty());
        // The GA104 tour refuses the laptop instead of guessing.
        let mut cfg2 = observed::observed_config();
        cfg2.set_device_id(0x2520);
        let mut j2 = Journal::new(4096);
        assert!(
            run_tour(
                &mut cfg2,
                None,
                &mut bar0,
                &mut SimClock::default(),
                &mut j2
            )
            .is_err()
        );
        // The laptop as the firmware leaves it (Memory Space off): the tour
        // stops before the first BAR0 read.
        let mut cfg3 = gpu_sim::pci::SimConfig::ga106_laptop_asus();
        let mut bar3 = gpu_sim::mmio::SimMmio::ga106_bar0();
        let mut j3 = Journal::new(4096);
        assert_eq!(
            run_tour_ampere(
                &mut cfg3,
                None,
                &mut bar3,
                &mut SimClock::default(),
                &mut j3
            ),
            Err(GpuError::Blocked(crate::probe::MEMORY_SPACE_OFF))
        );
        assert!(bar3.reads.is_empty());
        assert_eq!(cfg3.writes, 0);
    }

    #[test]
    fn ampere_tour_on_workstation_adds_only_the_family() {
        let mut j = Journal::new(4096);
        let a = run_tour(
            &mut observed::observed_config(),
            None,
            &mut observed::observed_bar0(),
            &mut SimClock::default(),
            &mut j,
        )
        .unwrap();
        let mut j = Journal::new(4096);
        let b = run_tour_ampere(
            &mut observed::observed_config(),
            None,
            &mut observed::observed_bar0(),
            &mut SimClock::default(),
            &mut j,
        )
        .unwrap();
        assert_eq!(b.get("chip.familie"), Some("GA104"));
        assert!(b.get("chip.scrieri").is_none());
        let rest: Vec<_> = b.facts.iter().filter(|f| f.key != "chip.familie").collect();
        assert_eq!(rest.len(), a.facts.len());
        assert!(rest.iter().zip(a.facts.iter()).all(|(x, y)| *x == y));
    }

    /// C-116 step 2, SIMULARE: the laptop as the firmware leaves it
    /// (Command 0). The wake-up writes Command once, the tour reads with 0
    /// writes, Command goes back to 0: two configuration writes in all,
    /// no GPU register written.
    #[test]
    fn awake_tour_on_laptop_model_two_config_writes_board_left_as_found() {
        use gpu_pci::mem_space::MemorySpacePermit;
        let permit = MemorySpacePermit::new("test SIMULARE C-116");
        let mut cfg = observed::observed_config();
        cfg.set_device_id(0x2520);
        cfg.write_raw(0x04, 0x0010_0000); // Memory Space off, as observed on the laptop
        let mut bar0 = observed::observed_bar0();
        bar0.set(0, 0xb760_00a1);
        bar0.set(0xa00, 0x176a_1000);
        let mut j = Journal::new(4096);
        let (t, wake) = run_tour_awake(
            &mut cfg,
            None,
            &mut bar0,
            &mut SimClock::default(),
            &mut j,
            &permit,
        )
        .unwrap();
        assert_eq!(wake, Wake::EnabledAndRestored);
        assert_eq!(t.get("chip.familie"), Some("GA106"));
        assert_eq!(t.get("chip.nume"), Some("GA106"));
        assert!(
            t.get(FACT_WAKE)
                .unwrap()
                .starts_with("Memory Space pornit de noi")
        );
        assert_eq!(cfg.writes, 2, "enable + restore");
        assert_eq!(j.writes(), 2);
        assert!(bar0.writes.is_empty(), "no GPU register written");
        assert_eq!(
            cfg.read_raw(0x04) & 0xffff,
            0x0000,
            "Command as the firmware left it"
        );
        // The restore is also journaled as a step.
        assert!(j.events().any(
            |e| matches!(e, Event::Step { name, ok: true } if *name == gpu_pci::mem_space::STEP_RESTORED)
        ));
        // Memory Space already on (the workstation): no write at all.
        let mut j = Journal::new(4096);
        let mut cfg2 = observed::observed_config();
        let (t2, wake2) = run_tour_awake(
            &mut cfg2,
            None,
            &mut observed::observed_bar0(),
            &mut SimClock::default(),
            &mut j,
            &permit,
        )
        .unwrap();
        assert_eq!(wake2, Wake::NotNeeded);
        assert_eq!(t2.get("chip.familie"), Some("GA104"));
        assert_eq!(cfg2.writes, 0);
        assert_eq!(j.writes(), 0);
    }

    /// C-116, SIMULARE: when the tour fails after the wake-up (the board does
    /// not answer BOOT_0), Command is still put back.
    #[test]
    fn awake_tour_restores_command_even_when_the_tour_fails() {
        use gpu_pci::mem_space::MemorySpacePermit;
        let permit = MemorySpacePermit::new("test SIMULARE C-116");
        let mut cfg = gpu_sim::pci::SimConfig::ga106_laptop_asus();
        let mut bar0 = gpu_sim::mmio::SimMmio::new(16 << 20);
        bar0.set(0, 0xffff_ffff); // BOOT_0 all ones: not responding
        let mut j = Journal::new(4096);
        let r = run_tour_awake(
            &mut cfg,
            None,
            &mut bar0,
            &mut SimClock::default(),
            &mut j,
            &permit,
        );
        assert!(matches!(r, Err(GpuError::NotResponding)));
        assert_eq!(cfg.writes, 2, "enable + restore");
        assert_eq!(cfg.read_raw(0x04) & 0xffff, 0x0000);
        // A non-NVIDIA / unknown device: refused before any write.
        let mut cfg = gpu_sim::pci::SimConfig::ga106_laptop_asus();
        cfg.set_device_id(0x25a0); // GA107: not in our tables
        let mut j = Journal::new(64);
        assert!(matches!(
            run_tour_awake(
                &mut cfg,
                None,
                &mut bar0,
                &mut SimClock::default(),
                &mut j,
                &permit
            ),
            Err(GpuError::Unsupported { .. })
        ));
        assert_eq!(cfg.writes, 0);
    }
}
