//! C-87 step 3 — recovery without restarting the computer, when the board stops answering in the middle of work
//! (a channel that does not finish, GSP-RM silent).
//!
//! The danger after a hang is not the board, it is the computer's RAM: pages of host RAM are mapped for the GPU
//! (I11, the logits copy) and a channel still alive could write into them later, after the OS gave them to someone
//! else. So the rule here: **those pages go back to the OS only when it is proven that nothing on the board can
//! write there any more**, otherwise they stay set aside (quarantine) until the next start. The computer keeps
//! running (Kronos continues on the processor); only the board waits.
//!
//! The steps, in this order, each bounded by the caller's own timeout, every one tried even if one before failed:
//! 1. no more submissions (the caller has already stopped: `kronos_gen::Stop::Eroare`);
//! 2. ask GSP-RM whether it answers (a read-only control, e.g. `user_space::query_gr_info`);
//! 3. if it answers: take the channel group off the runlist (`user_space::stop_compute_group`, an RM control — **no
//!    register is written by us**);
//! 4. read `GPGet`/`GPPut` from USERD once more (only a read);
//! 5. **Bus Master off, always** (the same 1 write + re-read as the chain's cleanup step): with it off the board
//!    cannot reach the computer's RAM at all;
//! 6. the verdict ([`Recovered`]) and the journal lines.
//!
//! Tested only in SIMULARE (fake operations + `FakeRm`). Nothing here touches a board by itself: the OS gives the
//! operations ([`RecoveryOps`]) and runs this only inside the chain, behind Adrian's `DA`.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_core::error::GpuResult;

/// What the OS does on the board (or the simulation), each one bounded in time by the OS.
pub trait RecoveryOps {
    /// A read-only RM control; `Ok` = GSP-RM answered.
    fn gsp_ping(&mut self) -> GpuResult<()>;
    /// `stop_compute_group` (RM control). Only called when the ping answered.
    fn stop_group(&mut self) -> GpuResult<()>;
    /// `(GPGet, GPPut)` read from USERD (no write).
    fn ring_pointers(&mut self) -> GpuResult<(u32, u32)>;
    /// Bus Master off and re-read; `Ok(true)` = the re-read shows it off.
    fn bus_master_off(&mut self) -> GpuResult<bool>;
}

/// What can be done with the host-RAM pages the GPU had mapped.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RamPages {
    /// Nothing on the board can write them any more: the OS may reuse them.
    Eliberate,
    /// Not proven: they stay set aside until the computer restarts (the OS must never reuse them).
    Carantina,
}

/// May a new chain run on the board without restarting the computer?
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NextChain {
    /// GSP-RM answered and the group was taken off: a new chain (with a new `DA`) may start.
    Permis,
    /// The board did not answer as needed: the GPU stays off until the next cold start; the computer keeps working.
    DoarDupaRepornire,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Recovered {
    pub gsp_answers: bool,
    pub group_stopped: bool,
    /// `(GPGet, GPPut)` if they could be read.
    pub ring: Option<(u32, u32)>,
    pub bus_master_off: bool,
    pub ram: RamPages,
    pub next: NextChain,
    /// One line per step, for the journal (stick + report).
    pub lines: Vec<String>,
}

impl Recovered {
    /// The last line of the journal: what Adrian needs to know.
    pub fn summary(&self) -> String {
        let ram = match self.ram {
            RamPages::Eliberate => "paginile RAM ale placii eliberate",
            RamPages::Carantina => "paginile RAM ale placii puse deoparte pana la repornire",
        };
        let next = match self.next {
            NextChain::Permis => "un lant nou se poate porni (cu DA)",
            NextChain::DoarDupaRepornire => {
                "placa ramane oprita pana la urmatoarea pornire; calculatorul merge mai departe"
            }
        };
        format!(
            "recuperare: Bus Master {}; {ram}; {next}",
            if self.bus_master_off {
                "oprit"
            } else {
                "NECONFIRMAT oprit"
            }
        )
    }
}

/// Runs the recovery steps in order (every one tried, whatever failed before) and decides.
pub fn recover<O: RecoveryOps + ?Sized>(ops: &mut O) -> Recovered {
    let mut lines = Vec::new();
    let gsp_answers = match ops.gsp_ping() {
        Ok(()) => {
            lines.push(String::from("recuperare 1: GSP-RM raspunde"));
            true
        }
        Err(e) => {
            lines.push(format!(
                "recuperare 1: GSP-RM NU raspunde ({e}); grupul nu poate fi oprit prin RM"
            ));
            false
        }
    };
    let group_stopped = gsp_answers
        && match ops.stop_group() {
            Ok(()) => {
                lines.push(String::from(
                    "recuperare 2: grupul de canale scos de pe lista de rulare (control RM, niciun registru scris de noi)",
                ));
                true
            }
            Err(e) => {
                lines.push(format!("recuperare 2: oprirea grupului refuzata: {e}"));
                false
            }
        };
    let ring = match ops.ring_pointers() {
        Ok((get, put)) => {
            lines.push(format!(
                "recuperare 3: GPGet {get}, GPPut {put}{}",
                if get == put {
                    " (inelul golit)"
                } else {
                    " (intrari neluate de placa)"
                }
            ));
            Some((get, put))
        }
        Err(e) => {
            lines.push(format!("recuperare 3: USERD necitit: {e}"));
            None
        }
    };
    let bus_master_off = match ops.bus_master_off() {
        Ok(true) => {
            lines.push(String::from(
                "recuperare 4: Bus Master oprit (recitit): placa nu mai ajunge la RAM",
            ));
            true
        }
        Ok(false) => {
            lines.push(String::from(
                "recuperare 4: Bus Master NU apare oprit la recitire",
            ));
            false
        }
        Err(e) => {
            lines.push(format!("recuperare 4: Bus Master: {e}"));
            false
        }
    };
    // RAM: free only if nothing can write it now AND nothing can later — the group off the runlist (so a later
    // Bus Master on, for a new chain, does not wake the old channel) and Bus Master off now.
    let ram = if group_stopped && bus_master_off {
        RamPages::Eliberate
    } else {
        RamPages::Carantina
    };
    let next = if group_stopped && bus_master_off {
        NextChain::Permis
    } else {
        NextChain::DoarDupaRepornire
    };
    let mut r = Recovered {
        gsp_answers,
        group_stopped,
        ring,
        bus_master_off,
        ram,
        next,
        lines,
    };
    let s = r.summary();
    r.lines.push(s);
    r
}

/// Why a bounded wait ended without the last record: tells a stuck channel from a slow one, for the journal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hang {
    /// The board took the entry (GPGet moved) but the work did not finish: a launch hangs or is very slow.
    LucruNeterminat { records_seen: usize, of: usize },
    /// The board did not even take the entry (GPGet still behind GPPut): the channel is not running.
    CanalOprit,
}

/// `records_seen` = records with the right payload, in order; `get`/`put` from USERD after the wait.
pub fn classify(get: u32, put: u32, records_seen: usize, of: usize) -> Hang {
    if get != put && records_seen == 0 {
        Hang::CanalOprit
    } else {
        Hang::LucruNeterminat { records_seen, of }
    }
}

impl Hang {
    pub fn line(&self) -> String {
        match self {
            Hang::LucruNeterminat { records_seen, of } => format!(
                "placa a luat comanda, dar s-a oprit dupa {records_seen} din {of} inregistrari (o lansare nu s-a terminat)"
            ),
            Hang::CanalOprit => String::from(
                "placa nu a luat comanda (GPGet in urma lui GPPut): canalul nu ruleaza",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::error::GpuError;

    #[derive(Default)]
    struct Fake {
        silent: bool,
        refuse_stop: bool,
        userd_broken: bool,
        bm_stays_on: bool,
        bm_error: bool,
        calls: Vec<&'static str>,
    }

    impl RecoveryOps for Fake {
        fn gsp_ping(&mut self) -> GpuResult<()> {
            self.calls.push("ping");
            if self.silent {
                Err(GpuError::Timeout { waited_us: 500_000 })
            } else {
                Ok(())
            }
        }
        fn stop_group(&mut self) -> GpuResult<()> {
            self.calls.push("stop");
            if self.refuse_stop {
                Err(GpuError::HardwareFault { code: 0x1f })
            } else {
                Ok(())
            }
        }
        fn ring_pointers(&mut self) -> GpuResult<(u32, u32)> {
            self.calls.push("ring");
            if self.userd_broken {
                Err(GpuError::NotResponding)
            } else {
                Ok((3, 4))
            }
        }
        fn bus_master_off(&mut self) -> GpuResult<bool> {
            self.calls.push("bm");
            if self.bm_error {
                Err(GpuError::NotResponding)
            } else {
                Ok(!self.bm_stays_on)
            }
        }
    }

    #[test]
    fn a_stuck_channel_with_gsp_alive_is_recovered() {
        let mut f = Fake::default();
        let r = recover(&mut f);
        assert_eq!(f.calls, ["ping", "stop", "ring", "bm"]);
        assert_eq!((r.ram, r.next), (RamPages::Eliberate, NextChain::Permis));
        assert_eq!(r.ring, Some((3, 4)));
        assert_eq!(r.lines.len(), 5);
        assert!(r.lines[2].contains("neluate"));
        assert!(r.summary().contains("eliberate"));
    }

    #[test]
    fn gsp_silent_quarantines_the_ram_and_still_turns_bus_master_off() {
        let mut f = Fake {
            silent: true,
            ..Fake::default()
        };
        let r = recover(&mut f);
        // No RM call after a silent ping; Bus Master off is still done.
        assert_eq!(f.calls, ["ping", "ring", "bm"]);
        assert!(r.bus_master_off && !r.group_stopped);
        assert_eq!(
            (r.ram, r.next),
            (RamPages::Carantina, NextChain::DoarDupaRepornire)
        );
        assert!(r.summary().contains("deoparte"));
    }

    #[test]
    fn every_step_is_tried_whatever_failed_before() {
        let mut f = Fake {
            refuse_stop: true,
            userd_broken: true,
            ..Fake::default()
        };
        let r = recover(&mut f);
        assert_eq!(f.calls, ["ping", "stop", "ring", "bm"]);
        assert_eq!(r.ring, None);
        assert_eq!(r.ram, RamPages::Carantina);
        for bad in [
            Fake {
                bm_stays_on: true,
                ..Fake::default()
            },
            Fake {
                bm_error: true,
                ..Fake::default()
            },
        ] {
            let mut f = bad;
            let r = recover(&mut f);
            assert!(!r.bus_master_off);
            assert_eq!(
                r.ram,
                RamPages::Carantina,
                "fara Bus Master oprit, RAM-ul nu se elibereaza"
            );
            assert_eq!(r.next, NextChain::DoarDupaRepornire);
            assert!(r.summary().contains("NECONFIRMAT"));
        }
    }

    #[test]
    fn hangs_are_told_apart() {
        assert_eq!(classify(2, 3, 0, 196), Hang::CanalOprit);
        assert_eq!(
            classify(3, 3, 40, 196),
            Hang::LucruNeterminat {
                records_seen: 40,
                of: 196
            }
        );
        assert!(classify(3, 3, 0, 196).line().contains("0 din 196"));
        assert!(Hang::CanalOprit.line().contains("nu ruleaza"));
        // How far the board got: records 0..=2 with their payload, record 3 still zero.
        let mut rec = alloc::vec![0u8; 16 * 5];
        for k in 0..3u32 {
            rec[16 * k as usize..16 * k as usize + 4].copy_from_slice(&(0xa00 + k).to_le_bytes());
        }
        assert_eq!(crate::transfer::records_seen(&rec, 4, 0xa00), 3);
        assert_eq!(crate::transfer::records_seen(&rec, 4, 0xb00), 0);
        assert_eq!(crate::transfer::records_seen(&rec[..16], 4, 0xa00), 1);
        assert_eq!(
            classify(5, 5, 3, 5),
            Hang::LucruNeterminat {
                records_seen: 3,
                of: 5
            }
        );
    }
}
