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
                    Some(StepStatus::Blocked(why))
                        if !BLOCKED_ONLY_FOR_WRITES.contains(&s.plan_step) =>
                    {
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
}
