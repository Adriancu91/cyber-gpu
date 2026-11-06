//! C-54 (1): the window steps "in a chain" — I6 → I7 → … one after the
//! other, with a safe stop at the first failure and a complete journal.
//!
//! The driver owns the order and the safety rules; the OS owns how each
//! step is done (BAR mapping, DMA buffer, stick files) and passes it as a
//! closure. Rules, all tested in SIMULARE:
//! 1. steps run in [`WINDOW_CHAIN`] order; the first failure stops the
//!    chain: no later normal step runs;
//! 2. **cleanup steps** (Bus Master off) still run after a failure, once any
//!    writing step was attempted — the board is left as it was found;
//! 3. without `writes_allowed` (the automatic loop, C-53: no Adrian present)
//!    the chain stops *before* the first writing step, which never runs;
//! 4. a step that is `Blocked` in [`crate::init::GA104_PLAN`] (I7 today)
//!    stops the chain honestly — it is never "tried";
//! 5. a read-only step that journals a write is a failure;
//! 6. every step is journaled (`Event::Step`), with a note where it stopped.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::GpuResult;

use crate::init::{GA104_PLAN, StepStatus};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainStep {
    pub id: &'static str,
    pub what: &'static str,
    /// The step writes to the board (config space or BAR0).
    pub writes_hardware: bool,
    /// Runs even after a failure (if a writing step was attempted).
    pub cleanup: bool,
    /// The step of the init plan it belongs to (its status is checked).
    pub plan_step: &'static str,
}

/// The window chain (plan `docs/FEREASTRA_C-49.md`, then what follows).
pub const WINDOW_CHAIN: &[ChainStep] = &[
    ChainStep {
        id: "pregatire-i6",
        what: "doar citire: semnatura FWSEC, VRAM, WPR2 coborat, regiunea FRTS",
        writes_hardware: false,
        cleanup: false,
        plan_step: "I6",
    },
    ChainStep {
        id: "bus-master-pornit",
        what: "1 scriere in Command (bitul 2), recitita",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I5",
    },
    ChainStep {
        id: "i6-fwsec-frts",
        what: "FWSEC-FRTS pe falcon-ul GSP (18 registre in fereastra GSP); dovada: falcon oprit fara eroare, WPR2 unde am cerut",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I6",
    },
    ChainStep {
        id: "i7-booter-gsp",
        what: "booter_load pe SEC2 si pornirea GSP RISC-V",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I7",
    },
    ChainStep {
        id: "bus-master-oprit",
        what: "Bus Master oprit la loc (1 scriere, recitita)",
        writes_hardware: true,
        cleanup: true,
        plan_step: "I5",
    },
    ChainStep {
        id: "tur-dupa",
        what: "turul de citire dupa pasi (C-54), ca dovada a starii in care ramane placa",
        writes_hardware: false,
        cleanup: true,
        plan_step: "I0",
    },
];

/// C-71: the whole window, I6 to the first computation, for `gpu lant` at a
/// window with Adrian (run with [`run_chain_ready`] and
/// [`READY_FOR_WINDOW`]). Every step after I6 is written and tested **only in
/// SIMULARE**; on the board each one is a first try, and the chain stops at
/// the first failure with Bus Master put back. [`WINDOW_CHAIN`] stays as it
/// was (it stops before I7).
pub const WINDOW_CHAIN_FULL: &[ChainStep] = &[
    WINDOW_CHAIN[0],
    WINDOW_CHAIN[1],
    WINDOW_CHAIN[2],
    ChainStep {
        id: "i6-wpr2-citit",
        what: "doar citire: WPR2 dupa I6 (read_frts_after_i6), forma capatului",
        writes_hardware: false,
        cleanup: false,
        plan_step: "I6",
    },
    WINDOW_CHAIN[3],
    ChainStep {
        id: "i7-citit",
        what: "doar citire: cutiile SEC2 si RISC-V GSP dupa I7 (read_after_i7)",
        writes_hardware: false,
        cleanup: false,
        plan_step: "I7",
    },
    ChainStep {
        id: "i8-gsp-init",
        what: "SetSystemInfo + SetRegistry, soneria GSP, secventiatorul cerut de GSP-RM (doar ferestrele GSP/SEC2), asteptarea GSP_INIT_DONE",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I8",
    },
    ChainStep {
        id: "i8e-golden",
        what: "golden image: client, dispozitiv, canal, contextele GR promovate (mesaje RPC + soneria dupa fiecare)",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I8",
    },
    ChainStep {
        id: "i9-canal-lucru",
        what: "clientul de lucru pe tabelele noastre de pagini (scrise in VRAM), grupul de canale, canalul de lucru",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I9",
    },
    ChainStep {
        id: "i10-add64",
        what: "primul calcul: add64 (64 de adunari f32, NAK), rezultatul comparat bit cu bit cu procesorul",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I10",
    },
    ChainStep {
        id: "i10-cost-lansare",
        what: "masuratoarea 1 (un singur program): 16 lansari add64 cu marca de timp dupa fiecare",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I10",
    },
    WINDOW_CHAIN[4],
    WINDOW_CHAIN[5],
];

/// C-83: the window of 9 nov — [`WINDOW_CHAIN_FULL`] with I11 (large copy
/// RAM ↔ VRAM), I12 (Q8_0 matrix × vector), I13 (one Kronos layer) and I14
/// (whole #513 tokens) after I10, before the cleanup.
/// Written and tested **only in SIMULARE**; run with [`run_chain_ready`] and
/// [`READY_FOR_WINDOW_C83`]. [`WINDOW_CHAIN_FULL`] stays as it was.
pub const WINDOW_CHAIN_C83: &[ChainStep] = &[
    WINDOW_CHAIN_FULL[0],
    WINDOW_CHAIN_FULL[1],
    WINDOW_CHAIN_FULL[2],
    WINDOW_CHAIN_FULL[3],
    WINDOW_CHAIN_FULL[4],
    WINDOW_CHAIN_FULL[5],
    WINDOW_CHAIN_FULL[6],
    WINDOW_CHAIN_FULL[7],
    WINDOW_CHAIN_FULL[8],
    WINDOW_CHAIN_FULL[9],
    WINDOW_CHAIN_FULL[10],
    ChainStep {
        id: "i11-copiere",
        what: "copiere RAM -> VRAM -> RAM prin canalul de lucru (copy_b128), verificata bit cu bit, viteza din 3 marci de timp",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I11",
    },
    ChainStep {
        id: "i12-q8-matvec",
        what: "y = W*x, W Q8_0 3072x768 (copiere sus, q8_matvec, copiere jos), comparat cu procesorul",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I12",
    },
    ChainStep {
        id: "i13-strat",
        what: "un strat Kronos sintetic 768/12 capete pe 4 pozitii (16 lansari fiecare), comparat cu procesorul",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I13",
    },
    ChainStep {
        id: "i14-simbol",
        what: "#513 de pe stick: textele de control, 5 simboluri greedy pe placa (195 de lansari fiecare), comparate cu procesorul",
        writes_hardware: true,
        cleanup: false,
        plan_step: "I14",
    },
    WINDOW_CHAIN_FULL[11],
    WINDOW_CHAIN_FULL[12],
];

/// C-83: [`READY_FOR_WINDOW`] plus I11–I14.
pub const READY_FOR_WINDOW_C83: &[&str] = &[
    "I5", "I6", "I7", "I8", "I9", "I10", "I11", "I12", "I13", "I14",
];

/// C-71: plan steps that are `Blocked` only because they write to the board
/// (or need the step before on the board) and may run in the full window
/// chain when Adrian allows writes. Everything else blocked still stops.
pub const READY_FOR_WINDOW: &[&str] = &["I5", "I6", "I7", "I8", "I9", "I10"];

/// Steps whose plan status may be `Blocked` and still run in the chain:
/// I5/I6 are blocked in the plan only because they write ("doar cu Adrian
/// de fata"); `writes_allowed` is exactly that permission. Anything else
/// blocked (I7+: code or prerequisites missing) stops the chain.
const BLOCKED_ONLY_FOR_WRITES: &[&str] = &["I5", "I6"];

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    Passed,
    Failed(String),
    /// Not run, with the reason.
    Skipped(&'static str),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainReport {
    pub steps: Vec<(&'static str, Outcome)>,
    /// The step where the chain stopped, and why (None = all passed).
    pub stopped_at: Option<(&'static str, String)>,
    /// Writes journaled during the whole chain.
    pub writes: usize,
}

impl ChainReport {
    pub fn all_passed(&self) -> bool {
        self.stopped_at.is_none()
    }
    pub fn outcome(&self, id: &str) -> Option<&Outcome> {
        self.steps.iter().find(|(s, _)| *s == id).map(|(_, o)| o)
    }
}

impl fmt::Display for ChainReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (id, o) in &self.steps {
            match o {
                Outcome::Passed => writeln!(f, "  {id}: TRECE")?,
                Outcome::Failed(e) => writeln!(f, "  {id}: ESUAT: {e}")?,
                Outcome::Skipped(w) => writeln!(f, "  {id}: sarit ({w})")?,
            }
        }
        match &self.stopped_at {
            None => writeln!(f, "lant: toti pasii au trecut; {} scrieri", self.writes),
            Some((id, why)) => writeln!(
                f,
                "lant: oprit la {id}: {why}; {} scrieri; pasii de curatare au rulat",
                self.writes
            ),
        }
    }
}

fn plan_status(plan_step: &str) -> Option<StepStatus> {
    GA104_PLAN
        .iter()
        .find(|s| s.id == plan_step)
        .map(|s| s.status)
}

/// Runs `steps` in order with the rules of the module doc. `run` does one
/// step (the OS's implementation); it gets the journal to record accesses.
pub fn run_chain(
    steps: &'static [ChainStep],
    writes_allowed: bool,
    journal: &mut Journal,
    run: impl FnMut(&ChainStep, &mut Journal) -> GpuResult<()>,
) -> ChainReport {
    run_chain_ready(steps, BLOCKED_ONLY_FOR_WRITES, writes_allowed, journal, run)
}

/// C-71: [`run_chain`] with the list of plan steps that may run although
/// blocked in the plan (`ready`, e.g. [`READY_FOR_WINDOW`]). Same rules.
pub fn run_chain_ready(
    steps: &'static [ChainStep],
    ready: &[&str],
    writes_allowed: bool,
    journal: &mut Journal,
    mut run: impl FnMut(&ChainStep, &mut Journal) -> GpuResult<()>,
) -> ChainReport {
    let start_writes = journal.writes();
    let mut out = Vec::new();
    let mut stopped: Option<(&'static str, String)> = None;
    let mut wrote_something = false;
    journal.push(Event::Note(if writes_allowed {
        "lant (C-54): incepe, scrieri PERMISE (fereastra cu Adrian)"
    } else {
        "lant (C-54): incepe, doar citire (fara scrieri)"
    }));
    for s in steps {
        if stopped.is_some() {
            if s.cleanup && (wrote_something || !s.writes_hardware) {
                // fall through: cleanup runs
            } else {
                out.push((s.id, Outcome::Skipped("lantul s-a oprit inainte")));
                continue;
            }
        } else {
            // Gates before a normal step.
            let gate = if s.writes_hardware && !writes_allowed {
                Some(String::from(
                    "refuzat: pas care scrie, iar modul e doar citire (fara Adrian)",
                ))
            } else {
                match plan_status(s.plan_step) {
                    None => Some(format!("pas necunoscut in plan: {}", s.plan_step)),
                    Some(StepStatus::Blocked(why)) if !ready.contains(&s.plan_step) => {
                        Some(format!("blocat in plan ({}): {why}", s.plan_step))
                    }
                    Some(StepStatus::NotImplemented) => {
                        Some(format!("neimplementat ({})", s.plan_step))
                    }
                    _ => None,
                }
            };
            if let Some(why) = gate {
                journal.push(Event::Step {
                    name: s.id,
                    ok: false,
                });
                out.push((s.id, Outcome::Skipped("oprire inainte de pas")));
                stopped = Some((s.id, why));
                continue;
            }
        }
        let before = journal.writes();
        let r = run(s, journal);
        let wrote = journal.writes() != before;
        if s.writes_hardware {
            wrote_something = true;
        }
        let r = match r {
            Ok(()) if wrote && !s.writes_hardware => {
                Err(String::from("pas doar-citire care a jurnalizat o scriere"))
            }
            Ok(()) => Ok(()),
            Err(e) => Err(format!("{e}")),
        };
        journal.push(Event::Step {
            name: s.id,
            ok: r.is_ok(),
        });
        match r {
            Ok(()) => out.push((s.id, Outcome::Passed)),
            Err(e) => {
                if stopped.is_none() {
                    stopped = Some((s.id, e.clone()));
                }
                out.push((s.id, Outcome::Failed(e)));
            }
        }
    }
    if stopped.is_some() {
        journal.push(Event::Note("lant (C-54): oprit; vezi pasul cu ok=false"));
    }
    ChainReport {
        steps: out,
        stopped_at: stopped,
        writes: journal.writes() - start_writes,
    }
}

#[cfg(test)]
mod tests {
    //! SIMULARE: the steps are closures over the gpu-sim board.
    use super::*;
    use gpu_core::error::GpuError;
    use gpu_core::hal::ConfigSpace;
    use gpu_pci::bus_master::{BusMasterPermit, restore_command, set_bus_master};
    use gpu_sim::observed;

    fn ids(r: &ChainReport, want: fn(&Outcome) -> bool) -> Vec<&'static str> {
        r.steps
            .iter()
            .filter(|(_, o)| want(o))
            .map(|(s, _)| *s)
            .collect()
    }

    /// The OS side, simulated: Bus Master for real on the sim config space,
    /// I6 as `i6`, everything else passes without touching anything.
    fn window(
        writes_allowed: bool,
        i6: GpuResult<()>,
    ) -> (ChainReport, gpu_sim::pci::SimConfig, Journal) {
        let mut cfg = observed::observed_config();
        let mut j = Journal::new(256);
        let permit = BusMasterPermit::new("test C-54 (SIMULARE)");
        let mut old = None;
        let r = run_chain(WINDOW_CHAIN, writes_allowed, &mut j, |s, j| match s.id {
            "bus-master-pornit" => {
                old = Some(set_bus_master(&mut cfg, &permit, j, true)?);
                Ok(())
            }
            "bus-master-oprit" => match old {
                Some(o) => restore_command(&mut cfg, &permit, j, o),
                None => Ok(()),
            },
            "i6-fwsec-frts" => i6,
            _ => Ok(()),
        });
        (r, cfg, j)
    }

    #[test]
    fn i7_is_blocked_so_the_chain_stops_there_and_cleans_up() {
        let (r, mut cfg, _) = window(true, Ok(()));
        assert_eq!(
            ids(&r, |o| *o == Outcome::Passed),
            [
                "pregatire-i6",
                "bus-master-pornit",
                "i6-fwsec-frts",
                "bus-master-oprit",
                "tur-dupa"
            ]
        );
        let (at, why) = r.stopped_at.clone().unwrap();
        assert_eq!(at, "i7-booter-gsp");
        assert!(why.starts_with("blocat in plan (I7)"), "{why}");
        // Bus Master on, then off: 2 writes, Command back to 0x0003.
        assert_eq!(r.writes, 2);
        assert_eq!(cfg.read32(0x04) as u16, observed::COMMAND);
    }

    #[test]
    fn first_failure_stops_and_bus_master_is_put_back() {
        let (r, mut cfg, j) = window(true, Err(GpuError::Timeout { waited_us: 1000 }));
        assert_eq!(r.stopped_at.as_ref().unwrap().0, "i6-fwsec-frts");
        assert!(matches!(
            r.outcome("i6-fwsec-frts"),
            Some(Outcome::Failed(_))
        ));
        assert_eq!(
            r.outcome("i7-booter-gsp"),
            Some(&Outcome::Skipped("lantul s-a oprit inainte"))
        );
        assert_eq!(r.outcome("bus-master-oprit"), Some(&Outcome::Passed));
        assert_eq!(r.outcome("tur-dupa"), Some(&Outcome::Passed));
        assert_eq!(cfg.read32(0x04) as u16, observed::COMMAND);
        assert!(j.events().any(|e| *e
            == Event::Step {
                name: "i6-fwsec-frts",
                ok: false
            }));
    }

    #[test]
    fn read_only_mode_never_runs_a_writing_step() {
        let mut ran = Vec::new();
        let mut j = Journal::new(64);
        let r = run_chain(WINDOW_CHAIN, false, &mut j, |s, _| {
            ran.push(s.id);
            Ok(())
        });
        assert!(ran.iter().all(|id| {
            !WINDOW_CHAIN
                .iter()
                .find(|s| s.id == *id)
                .unwrap()
                .writes_hardware
        }));
        assert_eq!(ran, ["pregatire-i6", "tur-dupa"]);
        assert_eq!(r.stopped_at.as_ref().unwrap().0, "bus-master-pornit");
        assert_eq!(r.writes, 0);
    }

    #[test]
    fn failure_before_any_write_skips_the_write_cleanup() {
        let mut ran = Vec::new();
        let mut j = Journal::new(64);
        let r = run_chain(WINDOW_CHAIN, true, &mut j, |s, _| {
            ran.push(s.id);
            if s.id == "pregatire-i6" {
                Err(GpuError::Blocked("WPR2 ridicat: trebuie pornire rece"))
            } else {
                Ok(())
            }
        });
        assert_eq!(ran, ["pregatire-i6", "tur-dupa"]);
        assert_eq!(
            r.outcome("bus-master-oprit"),
            Some(&Outcome::Skipped("lantul s-a oprit inainte"))
        );
        assert!(format!("{r}").contains("oprit la pregatire-i6: blocat: WPR2 ridicat"));
    }

    #[test]
    fn a_read_only_step_that_writes_is_a_failure() {
        let mut cfg = observed::observed_config();
        let mut j = Journal::new(64);
        let permit = BusMasterPermit::new("test: greseala plantata");
        let r = run_chain(WINDOW_CHAIN, true, &mut j, |s, j| {
            if s.id == "pregatire-i6" {
                // Planted bug: the read-only step turns Bus Master on.
                set_bus_master(&mut cfg, &permit, j, true).map(|_| ())
            } else {
                Ok(())
            }
        });
        assert_eq!(r.stopped_at.as_ref().unwrap().0, "pregatire-i6");
        assert!(
            matches!(r.outcome("pregatire-i6"), Some(Outcome::Failed(e)) if e.contains("doar-citire"))
        );
        assert_eq!(
            r.outcome("bus-master-pornit"),
            Some(&Outcome::Skipped("lantul s-a oprit inainte"))
        );
    }

    #[test]
    fn chain_steps_exist_in_the_plan_and_cleanups_come_last() {
        for s in WINDOW_CHAIN {
            assert!(plan_status(s.plan_step).is_some(), "{}", s.id);
        }
        let first_cleanup = WINDOW_CHAIN.iter().position(|s| s.cleanup).unwrap();
        assert!(WINDOW_CHAIN[first_cleanup..].iter().all(|s| s.cleanup));
    }
    /// C-71: the full chain's gates — read-only mode stops before Bus
    /// Master; the old ready list stops it at I7 (blocked in the plan);
    /// only with READY_FOR_WINDOW does it go further. SIMULARE.
    #[test]
    fn full_chain_gates() {
        let ids: Vec<_> = WINDOW_CHAIN_FULL.iter().map(|s| s.id).collect();
        assert_eq!(ids.first(), Some(&"pregatire-i6"));
        assert_eq!(&ids[ids.len() - 2..], ["bus-master-oprit", "tur-dupa"]);
        let mut ran = Vec::new();
        let mut j = Journal::new(64);
        let r = run_chain_ready(
            WINDOW_CHAIN_FULL,
            READY_FOR_WINDOW,
            false,
            &mut j,
            |s, _| {
                ran.push(s.id);
                Ok(())
            },
        );
        assert_eq!(r.stopped_at.unwrap().0, "bus-master-pornit");
        assert_eq!(ran, ["pregatire-i6", "tur-dupa"]);
        let mut j = Journal::new(64);
        let r = run_chain(WINDOW_CHAIN_FULL, true, &mut j, |_, _| Ok(()));
        assert_eq!(r.stopped_at.unwrap().0, "i7-booter-gsp");
        let mut j = Journal::new(64);
        let r = run_chain_ready(WINDOW_CHAIN_FULL, READY_FOR_WINDOW, true, &mut j, |_, _| {
            Ok(())
        });
        assert!(r.all_passed());
    }

    /// C-83: the new chain is the old one with I11 and I12 between I10 and
    /// the cleanup; without writes allowed it stops before the first writer;
    /// a failure in I11 skips I12 and still runs the cleanup.
    #[test]
    fn c83_chain_adds_i11_i12_before_cleanup() {
        let ids: Vec<_> = WINDOW_CHAIN_C83.iter().map(|s| s.id).collect();
        let full: Vec<_> = WINDOW_CHAIN_FULL.iter().map(|s| s.id).collect();
        assert_eq!(ids[..11], full[..11]);
        assert_eq!(
            ids[11..15],
            ["i11-copiere", "i12-q8-matvec", "i13-strat", "i14-simbol"]
        );
        assert_eq!(ids[15..], full[11..]);
        assert!(full[10] == "i10-cost-lansare" && full[11] == "bus-master-oprit");
        let mut j = Journal::new(256);
        let r = run_chain_ready(
            WINDOW_CHAIN_C83,
            READY_FOR_WINDOW_C83,
            true,
            &mut j,
            |s, _| {
                if s.id == "i11-copiere" {
                    Err(GpuError::InvalidArgument("i11: inregistrarea 1 lipseste"))
                } else {
                    Ok(())
                }
            },
        );
        assert_eq!(r.stopped_at.as_ref().map(|s| s.0), Some("i11-copiere"));
        assert!(matches!(
            r.outcome("i12-q8-matvec"),
            Some(Outcome::Skipped(_))
        ));
        assert_eq!(r.outcome("bus-master-oprit"), Some(&Outcome::Passed));
        // The old list of ready steps stops at I11 (blocked in the plan).
        let mut j = Journal::new(256);
        let r = run_chain_ready(WINDOW_CHAIN_C83, READY_FOR_WINDOW, true, &mut j, |_, _| {
            Ok(())
        });
        assert_eq!(r.stopped_at.as_ref().map(|s| s.0), Some("i11-copiere"));
        // All pass with the C-83 list.
        let mut j = Journal::new(256);
        let r = run_chain_ready(
            WINDOW_CHAIN_C83,
            READY_FOR_WINDOW_C83,
            true,
            &mut j,
            |_, _| Ok(()),
        );
        assert!(r.all_passed(), "{r}");
    }
}
