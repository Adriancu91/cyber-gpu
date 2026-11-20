//! C-151 point 3 — the reset from inside (tool 2), our part: **FLR at the start
//! of the chain, only when WPR2 is up** (variant 1 of
//! `docs/C-142_K513_VITEZA_FLR.md` §4; the note for the OS:
//! `docs/C-151_RESET_FLR.md`).
//!
//! The order (RM 570.144 `kbifDoFunctionLevelReset_TU102` + tinygrad
//! `nvdev.py`, both MIT, read as specifications):
//! 1. WPR2 read (`NV_PFB_PRI_MMU_WPR2_ADDR_HI`); 0 → nothing to do, no write;
//! 2. the configuration saved, Bus Master off, FLR, 100 ms, configuration
//!    written back ([`gpu_pci::flr`]);
//! 3. GFW boot awaited (our I2, reads only): after an FLR the VBIOS runs its
//!    initialisation again, as after power-on (RM waits for it after FLR);
//! 4. **WPR2 read again: must be 0.** Only then may the chain (I3 … I14) run.
//!
//! Every outcome is a [`FlrVerdict`] with a journal row (`kf`) and what it
//! means for the OS ([`FlrVerdict::meaning`]). There is no second attempt in
//! here: if WPR2 is still up, the OS stops cleanly (cold start, as today).
//!
//! Tested only in SIMULARE (`gpu_sim::pci::SimConfig` + `gpu_sim::fake_gsp`).
//! That FLR lowers WPR2 on GA104 is DEDUCED (~80 %), never seen on the board.

use alloc::format;
use alloc::string::String;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::GpuError;
use gpu_core::hal::{Clock, ConfigSpace, Mmio};
use gpu_pci::flr::{self, FlrPermit};

use crate::gfw;
use crate::gsp::HardwareWritePermit;
use crate::gsp::falcon::{GuardedBar0, Wpr2, read_wpr2};

/// The board as the reset sees it: BAR0 and the configuration space of the
/// GPU function, reached one after the other (never both at once).
pub trait FlrBoard {
    type Bar0: Mmio + ?Sized;
    type Cfg: ConfigSpace + ?Sized;
    fn bar0(&mut self) -> &mut Self::Bar0;
    fn cfg(&mut self) -> &mut Self::Cfg;
}

/// What happened, one value per outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlrVerdict {
    /// WPR2 was already 0: nothing written, the chain runs as today.
    NuEraNevoie,
    /// FLR done, GFW boot completed, WPR2 reads 0: the chain may run.
    Wpr2Coborat { ready_after_us: u64, gfw_us: u64 },
    /// FLR done, GFW completed, but WPR2 is still up: stop, cold start.
    Wpr2IncaRidicat,
    /// The board does not announce FLR: nothing written.
    FaraFlr,
    /// After the FLR the function never answered again (all ones).
    PlacaNuRaspunde,
    /// The configuration written back differs from the saved one.
    ConfigNerescrisa,
    /// GFW boot did not complete after the FLR.
    GfwNeterminat,
    /// Any other error (named in the journal).
    Eroare,
}

impl FlrVerdict {
    /// Short code for the `kf` row.
    pub fn code(&self) -> &'static str {
        match self {
            FlrVerdict::NuEraNevoie => "nu-era-nevoie",
            FlrVerdict::Wpr2Coborat { .. } => "wpr2-coborat",
            FlrVerdict::Wpr2IncaRidicat => "wpr2-inca-ridicat",
            FlrVerdict::FaraFlr => "fara-flr",
            FlrVerdict::PlacaNuRaspunde => "placa-nu-raspunde",
            FlrVerdict::ConfigNerescrisa => "config-nerescrisa",
            FlrVerdict::GfwNeterminat => "gfw-neterminat",
            FlrVerdict::Eroare => "eroare",
        }
    }

    /// May the chain (I3 …) run now, in the same power-on?
    pub fn chain_may_run(&self) -> bool {
        matches!(
            self,
            FlrVerdict::NuEraNevoie | FlrVerdict::Wpr2Coborat { .. }
        )
    }

    /// What the result means, for the cycle report (Romanian).
    pub fn meaning(&self) -> &'static str {
        match self {
            FlrVerdict::NuEraNevoie => "WPR2 era deja 0: niciun reset, lantul merge ca azi",
            FlrVerdict::Wpr2Coborat { .. } => {
                "FLR a coborat WPR2: lantul poate rula din nou in aceeasi pornire (varianta 1 merge)"
            }
            FlrVerdict::Wpr2IncaRidicat => {
                "FLR nu coboara WPR2 pe placa asta: oprire curata, pornire rece; urmeaza varianta 2 (Booter Unload cu FWSEC-SB)"
            }
            FlrVerdict::FaraFlr => "placa nu anunta FLR: nimic scris; pornire rece ca azi",
            FlrVerdict::PlacaNuRaspunde => {
                "placa nu mai raspunde dupa FLR: oprire curata, repornire in Windows, cablul daca trebuie"
            }
            FlrVerdict::ConfigNerescrisa => {
                "configuratia PCI nu s-a rescris: placa nu se mai foloseste in pornirea asta"
            }
            FlrVerdict::GfwNeterminat => {
                "VBIOS nu si-a terminat pornirea dupa FLR: oprire curata, pornire rece"
            }
            FlrVerdict::Eroare => "eroare neasteptata (vezi jurnalul): oprire curata",
        }
    }
}

/// The verdict and the two WPR2 readings, for the `kf` row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlrReport {
    pub wpr2_before: Option<Wpr2>,
    pub wpr2_after: Option<Wpr2>,
    pub verdict: FlrVerdict,
}

fn wpr2_text(w: Option<Wpr2>) -> String {
    match w {
        None => String::from("0"),
        Some(w) => format!("{:#x}..{:#x}", w.start, w.end),
    }
}

impl FlrReport {
    /// `kf <code> wpr2_inainte=<..> wpr2_dupa=<..> gata_us=<n> gfw_us=<n>`.
    pub fn row(&self) -> String {
        let (r, g) = match self.verdict {
            FlrVerdict::Wpr2Coborat {
                ready_after_us,
                gfw_us,
            } => (ready_after_us, gfw_us),
            _ => (0, 0),
        };
        format!(
            "kf {} wpr2_inainte={} wpr2_dupa={} gata_us={} gfw_us={}",
            self.verdict.code(),
            wpr2_text(self.wpr2_before),
            wpr2_text(self.wpr2_after),
            r,
            g
        )
    }
}

/// Reader for a `kf` row: `(code, chain may run)`; `None` if not a `kf` row
/// or the code is unknown.
pub fn parse_row(line: &str) -> Option<(&str, bool)> {
    let rest = line.trim().strip_prefix("kf ")?;
    let code = rest.split_whitespace().next()?;
    let may = match code {
        "nu-era-nevoie" | "wpr2-coborat" => true,
        "wpr2-inca-ridicat" | "fara-flr" | "placa-nu-raspunde" | "config-nerescrisa"
        | "gfw-neterminat" | "eroare" => false,
        _ => return None,
    };
    Some((code, may))
}

fn read_w<B: FlrBoard + ?Sized>(
    b: &mut B,
    j: &mut Journal,
    hw: Option<&HardwareWritePermit>,
) -> Result<Option<Wpr2>, GpuError> {
    let mut g = GuardedBar0::new(b.bar0(), j, hw)?;
    read_wpr2(&mut g)
}

/// Variant 1 of C-142 §4: FLR only when WPR2 is up, then GFW boot, then WPR2
/// read again. `hw` is the chain's BAR0 permit (needed on the real board,
/// used here only for reads); `permit` is the OS's explicit FLR permit.
pub fn reset_if_wpr2_up<B: FlrBoard + ?Sized, K: Clock + ?Sized>(
    board: &mut B,
    clock: &mut K,
    permit: &FlrPermit,
    hw: Option<&HardwareWritePermit>,
    j: &mut Journal,
) -> FlrReport {
    let mut rep = FlrReport {
        wpr2_before: None,
        wpr2_after: None,
        verdict: FlrVerdict::Eroare,
    };
    match read_w(board, j, hw) {
        Ok(None) => {
            j.push(Event::Note("C-151: WPR2 = 0, nu e nevoie de FLR"));
            rep.verdict = FlrVerdict::NuEraNevoie;
            return rep;
        }
        Ok(Some(w)) => rep.wpr2_before = Some(w),
        Err(_) => return rep,
    }
    j.push(Event::Note(permit.reason));
    let saved = match flr::save_config(board.cfg()) {
        Ok(s) => s,
        Err(_) => return rep,
    };
    let done = match flr::function_level_reset(board.cfg(), clock, permit, j, &saved) {
        Ok(d) => d,
        Err(GpuError::Blocked(m)) if m.contains("nu anunta FLR") => {
            rep.verdict = FlrVerdict::FaraFlr;
            return rep;
        }
        Err(GpuError::NotResponding) => {
            rep.verdict = FlrVerdict::PlacaNuRaspunde;
            return rep;
        }
        Err(GpuError::Blocked(m)) if m.contains("configuratia") => {
            rep.verdict = FlrVerdict::ConfigNerescrisa;
            return rep;
        }
        Err(_) => return rep,
    };
    let gfw_us = match gfw::wait_gfw_boot(board.bar0(), j, clock, gfw::GFW_TIMEOUT_US) {
        Ok(s) => s.waited_us,
        Err(_) => {
            rep.verdict = FlrVerdict::GfwNeterminat;
            return rep;
        }
    };
    match read_w(board, j, hw) {
        Ok(None) => {
            j.push(Event::Step {
                name: "C-151 FLR: WPR2 coborat",
                ok: true,
            });
            rep.verdict = FlrVerdict::Wpr2Coborat {
                ready_after_us: done.ready_after_us,
                gfw_us,
            };
        }
        Ok(Some(w)) => {
            rep.wpr2_after = Some(w);
            j.push(Event::Step {
                name: "C-151 FLR: WPR2 inca ridicat",
                ok: false,
            });
            rep.verdict = FlrVerdict::Wpr2IncaRidicat;
        }
        Err(_) => {}
    }
    rep
}

#[cfg(test)]
mod tests {
    //! T138 (C-151 point 3), SIMULARE: WPR2 up → FLR → WPR2 down → the
    //! chain (I6 + I7) runs a second time in the same "power-on".
    use super::*;
    use crate::gsp::falcon::tests_support::booted;
    use crate::gsp::falcon::{Engine, booter_load, fwsec_frts};
    use gpu_core::error::GpuResult;
    use gpu_sim::clock::SimClock;
    use gpu_sim::fake_gsp::{FakeGsp, SimFault, SimUcode};
    use gpu_sim::pci::SimConfig;

    /// The test's board: when the configuration space has seen an FLR the
    /// BAR0 model is told before it is next read (SIM coupling).
    struct Board {
        cfg: SimConfig,
        gsp: FakeGsp,
        seen: u32,
    }
    impl FlrBoard for Board {
        type Bar0 = FakeGsp;
        type Cfg = SimConfig;
        fn bar0(&mut self) -> &mut FakeGsp {
            while self.seen < self.cfg.flr_count {
                self.seen += 1;
                self.gsp.sim_flr();
            }
            &mut self.gsp
        }
        fn cfg(&mut self) -> &mut SimConfig {
            &mut self.cfg
        }
    }

    fn board(gsp: FakeGsp) -> Board {
        Board {
            cfg: SimConfig::workstation_rtx3070ti().with_flr(),
            gsp,
            seen: 0,
        }
    }

    fn permit() -> FlrPermit {
        FlrPermit::new("C-151 test: FLR in SIMULARE")
    }

    fn chain_again(g: &mut FakeGsp, j: &mut Journal, c: &mut SimClock) -> GpuResult<()> {
        let frts = g.frts;
        let load = |b: &mut GuardedBar0<'_, FakeGsp>, e: Engine| {
            let u = match e {
                Engine::Gsp => SimUcode::FwsecFrts,
                Engine::Sec2 => SimUcode::BooterLoad,
            };
            b.model().sim_load(e.base(), u);
            Ok(())
        };
        let mut b = GuardedBar0::new(g, j, None)?;
        fwsec_frts(&mut b, c, load, frts)?;
        booter_load(&mut b, c, load, 0x1_0000_0000)?;
        Ok(())
    }

    #[test]
    fn t138a_flr_lowers_wpr2_and_the_chain_runs_twice() {
        let (g, mut j, mut c) = booted();
        assert!(g.riscv_active());
        let mut b = board(g);
        let saved = flr::save_config(b.cfg()).unwrap();
        let rep = reset_if_wpr2_up(&mut b, &mut c, &permit(), None, &mut j);
        assert!(rep.wpr2_before.is_some());
        let FlrVerdict::Wpr2Coborat {
            ready_after_us,
            gfw_us,
        } = rep.verdict
        else {
            panic!("{rep:?}")
        };
        assert!(ready_after_us >= 100_000, "{ready_after_us}");
        assert!(gfw_us > 0);
        assert!(rep.verdict.chain_may_run());
        assert_eq!(b.cfg.flr_count, 1);
        assert_eq!(b.gsp.flr_count, 1);
        // configuration back exactly; BARs never written with decode on
        assert_eq!(flr::save_config(b.cfg()).unwrap(), saved);
        assert!(!b.cfg.decode_was_enabled_during_bar_write);
        // the row and its reader
        let row = rep.row();
        assert!(row.starts_with("kf wpr2-coborat wpr2_inainte=0x"), "{row}");
        assert!(row.contains("wpr2_dupa=0 "), "{row}");
        assert_eq!(parse_row(&row), Some(("wpr2-coborat", true)));
        // no WPR2 register was ever written by us
        assert!(
            !b.gsp
                .writes
                .iter()
                .any(|&(o, _)| o == 0x1fa824 || o == 0x1fa828)
        );
        // the chain a second time, same power-on: I6 + I7 pass again
        chain_again(&mut b.gsp, &mut j, &mut c).unwrap();
        assert!(b.gsp.riscv_active());
        let rows: alloc::vec::Vec<String> = j.events().map(|e| format!("{e}")).collect();
        for want in [
            "C-151 FLR: Bus Master oprit",
            "C-151 FLR declansat",
            "C-151 FLR: configuratia rescrisa",
            "C-151 FLR: WPR2 coborat",
        ] {
            assert!(rows.iter().any(|r| r.contains(want)), "{want}");
        }
    }

    #[test]
    fn t138b_wpr2_down_means_no_write_at_all() {
        let mut b = board(FakeGsp::new());
        let mut j = Journal::new(64);
        let mut c = SimClock::default();
        let rep = reset_if_wpr2_up(&mut b, &mut c, &permit(), None, &mut j);
        assert_eq!(rep.verdict, FlrVerdict::NuEraNevoie);
        assert!(rep.verdict.chain_may_run());
        assert_eq!(b.cfg.writes, 0);
        assert_eq!(b.cfg.flr_count, 0);
        assert!(b.gsp.writes.is_empty());
        assert_eq!(parse_row(&rep.row()), Some(("nu-era-nevoie", true)));
    }

    #[test]
    fn t138c_wpr2_survives_flr_is_a_clean_stop() {
        let (mut g, mut j, mut c) = booted();
        g.fault = SimFault::Wpr2SurvivesFlr;
        let mut b = board(g);
        let rep = reset_if_wpr2_up(&mut b, &mut c, &permit(), None, &mut j);
        assert_eq!(rep.verdict, FlrVerdict::Wpr2IncaRidicat);
        assert!(!rep.verdict.chain_may_run());
        assert!(rep.wpr2_after.is_some());
        assert!(rep.verdict.meaning().contains("varianta 2"));
        // exactly one FLR: no second attempt
        assert_eq!(b.cfg.flr_count, 1);
        assert_eq!(parse_row(&rep.row()), Some(("wpr2-inca-ridicat", false)));
    }

    #[test]
    fn t138d_every_other_failure_is_named() {
        // no FLR announced: nothing written
        let (g, mut j, mut c) = booted();
        let mut b = board(g);
        b.cfg = SimConfig::workstation_rtx3070ti();
        let rep = reset_if_wpr2_up(&mut b, &mut c, &permit(), None, &mut j);
        assert_eq!(rep.verdict, FlrVerdict::FaraFlr);
        assert_eq!(b.cfg.writes, 0);
        // the function never answers again
        let (g, mut j, mut c) = booted();
        let mut b = board(g);
        b.cfg.flr_never_ready = true;
        let rep = reset_if_wpr2_up(&mut b, &mut c, &permit(), None, &mut j);
        assert_eq!(rep.verdict, FlrVerdict::PlacaNuRaspunde);
        // a BAR that refuses the restore
        let (g, mut j, mut c) = booted();
        let mut b = board(g);
        b.cfg.fault_sticky_bar = Some(0);
        let rep = reset_if_wpr2_up(&mut b, &mut c, &permit(), None, &mut j);
        assert_eq!(rep.verdict, FlrVerdict::ConfigNerescrisa);
        // the VBIOS never finishes after the FLR
        let (mut g, mut j, mut c) = booted();
        g.fault = SimFault::GfwNeverAfterFlr;
        let mut b = board(g);
        let rep = reset_if_wpr2_up(&mut b, &mut c, &permit(), None, &mut j);
        assert_eq!(rep.verdict, FlrVerdict::GfwNeterminat);
        assert!(!rep.verdict.chain_may_run());
        // Transactions Pending stuck: journaled, the reset goes on (RM, Linux)
        let (g, mut j, mut c) = booted();
        let mut b = board(g);
        b.cfg.pending_transactions = true;
        let rep = reset_if_wpr2_up(&mut b, &mut c, &permit(), None, &mut j);
        assert!(matches!(rep.verdict, FlrVerdict::Wpr2Coborat { .. }));
        assert!(
            j.events()
                .any(|e| format!("{e}").contains("tranzactii in asteptare"))
        );
        assert_eq!(parse_row("kf ceva"), None);
        assert_eq!(parse_row("kg 1"), None);
    }
}
