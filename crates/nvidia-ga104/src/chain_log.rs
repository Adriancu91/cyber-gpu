//! C-71: checking the stick journal of a whole `gpu lant` window
//! ([`crate::chain::WINDOW_CHAIN_FULL`]: I6 → I7 → I8 → golden → I9 → I10 →
//! launch cost → Bus Master back → tour), step by step.
//!
//! The chain journals every step as `pas <id>: ok|ESEC` *after* the step's
//! own accesses ([`crate::chain::run_chain_ready`]), so the journal splits
//! into one segment per step: the accesses since the previous `pas` line.
//! Each segment is checked with the rule of its step:
//! - read-only steps: no write at all (BAR0 or config space);
//! - `bus-master-pornit` / `bus-master-oprit`: exactly one config write to
//!   Command (0x04), bit 2 set / the value from before put back;
//! - `i6-fwsec-frts`: [`crate::i6_log::check_i6`]; `i6-wpr2-citit`: the WPR2
//!   read-back with the driver's rule; `i7-booter-gsp`: [`crate::i6_log::check_i7`];
//!   `i7-citit`: GSP RISC-V active;
//! - `i8-gsp-init`: writes only in the GSP/SEC2 windows (the same windows as
//!   `GuardedBar0`), the first one the GSP doorbell (`NV_PGSP_QUEUE_HEAD(0)`
//!   = 0, after SetSystemInfo + SetRegistry), then the sequencer's writes;
//! - `i8e-golden`, `i9-canal-lucru`: RM calls are messages in memory, so the
//!   only register writes allowed are the GSP doorbell (value 0, at least one)
//!   and the four VRAM-window control registers
//!   ([`crate::vram_window::CONTROL_REGISTERS`]); with the number of RM calls
//!   the OS printed, the doorbells must match it exactly (one per message);
//! - `i10-*`: the VRAM-window registers and the work-channel doorbell
//!   (`0xbb0090`, from the MIT headers, not yet observed; counted separately, see
//!   `docs/IMPLEMENTATION_STATUS.md`), nothing else.
//!
//! - across the whole chain: every move of the VRAM window
//!   (`NV_PBUS_BAR0_WINDOW`) reads back as written, and by the end of the
//!   journal the window is back to the value it had before its first move
//!   (board state put back, like Bus Master).
//!
//! Verdict: [`ChainVerdict::Passed`] only if every step of the chain is in
//! the journal, in order, `ok`, and no rule is broken; a chain that stopped
//! at a step and then only ran its cleanup steps (Bus Master put back) is
//! [`ChainVerdict::StoppedCleanly`]; anything else is to be analysed.
//!
//! Pure text processing: this module never touches a board.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_core::diag::Event;

use crate::chain::{ChainStep, WINDOW_CHAIN_FULL};
use crate::i6_log::{self, I6Summary};

/// `NV_PGSP_QUEUE_HEAD(0)`: the GSP doorbell (570.144 `dev_gsp.h`, MIT).
pub const GSP_DOORBELL: u64 = 0x11_0c00;
/// The work-channel doorbell, `NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET` +
/// `NV_VIRTUAL_FUNCTION_DOORBELL` (0x30090): first DEDUS from
/// nouveau/tinygrad, now from the 570.144 MIT headers ([`crate::push::DOORBELL`],
/// docs/GA104_INITIALIZATION.md). Not yet observed on the board.
pub const WORK_DOORBELL: u64 = 0xbb_0090;

const _: () = assert!(WORK_DOORBELL == crate::push::DOORBELL);
/// `NV_PBUS_BAR0_WINDOW`: the VRAM window register moved by
/// [`crate::vram_window::VramWindow`] (570.144 `gm107/dev_bus.h`, MIT).
pub const VRAM_WINDOW: u64 = nvidia_rm570::regs::gm107_dev_bus::NV_PBUS_BAR0_WINDOW as u64;
/// PCI Command register (config space offset).
const COMMAND: u16 = 0x04;

/// Parses a step line written by the chain (`pas <id>: ok` / `pas <id>: ESEC`),
/// possibly after a log prefix. Returns the id and whether it passed.
pub fn parse_step(line: &str) -> Option<(&str, bool)> {
    let rest = &line[line.find("pas ")? + 4..];
    let (id, res) = rest.split_once(": ")?;
    if id.is_empty() || id.contains(char::is_whitespace) {
        return None;
    }
    match res.trim_end() {
        "ok" => Some((id, true)),
        "ESEC" => Some((id, false)),
        _ => None,
    }
}

/// What the journal says about one step of the chain.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StepLog {
    pub id: String,
    /// The chain's own result for the step (`ok` / `ESEC`).
    pub ok: bool,
    /// Register accesses in the step's segment.
    pub accesses: usize,
    pub bar0_writes: usize,
    pub config_writes: usize,
    /// GSP doorbell writes (`0x110c00` = 0).
    pub gsp_doorbells: u32,
    /// Work-channel doorbell writes (`0xbb0090`), I10 only.
    pub work_doorbells: u32,
    /// The first rule the segment breaks (None = conform).
    pub problem: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ChainVerdict {
    /// Every step present, in order, `ok`, all rules kept.
    Passed,
    /// Stopped at this step (`ESEC`), only cleanup steps after it, all rules
    /// kept, Bus Master put back if it was turned on.
    StoppedCleanly(String),
    /// Anything else (the reason).
    ToAnalyse(String),
}

/// A whole `gpu lant` stick journal, checked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChainJournal {
    pub steps: Vec<StepLog>,
    /// Problems not tied to one step (order, unknown ids, trailing accesses).
    pub problems: Vec<String>,
    /// I6 / I7 load order, when those steps are in the journal.
    pub i6: Option<I6Summary>,
    pub i7: Option<I6Summary>,
    /// C-75: I7 in the board's order (GSP reset into RISC-V + LibOS
    /// mailboxes before Booter, GSP-RM version after), when the journal has
    /// that form ([`i6_log::check_i7_libos`]); `i7` then holds its Booter part.
    pub i7_libos: Option<i6_log::I7LibosSummary>,
    /// GSP doorbells in golden + I9 (one per RM call on a correct run).
    pub rm_doorbells: u32,
}

impl ChainJournal {
    pub fn step(&self, id: &str) -> Option<&StepLog> {
        self.steps.iter().find(|s| s.id == id)
    }

    pub fn verdict(&self) -> ChainVerdict {
        if let Some(p) = self.problems.first() {
            return ChainVerdict::ToAnalyse(p.clone());
        }
        if let Some(s) = self.steps.iter().find(|s| s.problem.is_some()) {
            return ChainVerdict::ToAnalyse(format!(
                "{}: {}",
                s.id,
                s.problem.as_deref().unwrap_or("")
            ));
        }
        let failed: Vec<&StepLog> = self.steps.iter().filter(|s| !s.ok).collect();
        if failed.is_empty() {
            return if self.steps.len() == WINDOW_CHAIN_FULL.len() {
                ChainVerdict::Passed
            } else {
                ChainVerdict::ToAnalyse(format!(
                    "lipsesc pasi: {} din {} in jurnal",
                    self.steps.len(),
                    WINDOW_CHAIN_FULL.len()
                ))
            };
        }
        let first = failed[0];
        let at = self
            .steps
            .iter()
            .position(|s| s.id == first.id)
            .unwrap_or(0);
        for s in &self.steps[at + 1..] {
            if !chain_step(&s.id).is_some_and(|c| c.cleanup) {
                return ChainVerdict::ToAnalyse(format!(
                    "pasul {} a rulat dupa oprirea de la {}",
                    s.id, first.id
                ));
            }
            if !s.ok {
                return ChainVerdict::ToAnalyse(format!("pasul de curatare {} a esuat", s.id));
            }
        }
        let bm_on = self.step("bus-master-pornit").is_some_and(|s| s.ok);
        if bm_on && !self.step("bus-master-oprit").is_some_and(|s| s.ok) {
            return ChainVerdict::ToAnalyse(String::from(
                "Bus Master a fost pornit si nu a fost pus la loc",
            ));
        }
        ChainVerdict::StoppedCleanly(first.id.clone())
    }
}

fn chain_step(id: &str) -> Option<&'static ChainStep> {
    WINDOW_CHAIN_FULL.iter().find(|s| s.id == id)
}

/// Checks a whole `gpu lant` stick journal. `frts` is the FRTS region the OS
/// printed (I6), `wpr_meta_addr` the `GspFwWprMeta` address (I7 mailboxes),
/// `rm_calls` the number of RM calls the OS printed for golden + I9, if known
/// (then the doorbells must match it exactly). Pure text processing.
pub fn check_chain_journal(
    text: &str,
    frts: (u64, u64),
    wpr_meta_addr: u64,
    rm_calls: Option<u32>,
) -> ChainJournal {
    let mut out = ChainJournal {
        steps: Vec::new(),
        problems: Vec::new(),
        i6: None,
        i7: None,
        i7_libos: None,
        rm_doorbells: 0,
    };
    let mut seg: Vec<Event> = Vec::new();
    // Everything up to the current segment (for the WPR2 read-back).
    let mut all: Vec<Event> = Vec::new();
    let mut next = 0usize; // index in WINDOW_CHAIN_FULL
    let mut command_before: Option<u32> = None;
    // The VRAM window register: (value before the first move, last value written).
    let mut window: Option<(u32, u32)> = None;
    for line in text.lines() {
        if let Some((id, ok)) = parse_step(line) {
            let Some(pos) = WINDOW_CHAIN_FULL[next..].iter().position(|s| s.id == id) else {
                out.problems.push(if chain_step(id).is_some() {
                    format!("pasul {id} in alta ordine sau de doua ori")
                } else {
                    format!("pas necunoscut in lant: {id}")
                });
                seg.clear();
                continue;
            };
            let step = &WINDOW_CHAIN_FULL[next + pos];
            next += pos + 1;
            let log = check_step(
                step,
                ok,
                &seg,
                &all,
                frts,
                wpr_meta_addr,
                &mut command_before,
                &mut out,
            );
            out.steps.push(log);
            seg.clear();
            continue;
        }
        if let Some(e) = i6_log::parse_line(line) {
            if let Event::MmioWrite {
                bar: 0,
                offset: VRAM_WINDOW,
                before,
                written,
                after,
            } = e
            {
                window.get_or_insert((before, written)).1 = written;
                if after != written {
                    out.problems.push(format!(
                        "fereastra VRAM ({VRAM_WINDOW:#x}) nu s-a mutat: scris {written:#010x}, recitit {after:#010x}"
                    ));
                }
            }
            seg.push(e);
            all.push(e);
        }
    }
    // The VRAM window is board state, like Bus Master: whatever moved it
    // (page tables in golden / I9 / I10) must put it back before the end.
    if let Some((original, last)) = window
        && last != original
    {
        out.problems.push(format!(
            "fereastra VRAM ({VRAM_WINDOW:#x}) lasata la {last:#010x}, inainte era {original:#010x}"
        ));
    }
    if seg
        .iter()
        .any(|e| matches!(e, Event::MmioWrite { .. } | Event::ConfigWrite { .. }))
    {
        out.problems
            .push(String::from("scrieri dupa ultimul pas din jurnal"));
    }
    if let Some(n) = rm_calls
        && out.rm_doorbells != n
    {
        out.problems.push(format!(
            "sonerii GSP la golden + I9: {}, apeluri RM: {n} (trebuie una pe mesaj)",
            out.rm_doorbells
        ));
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn check_step(
    step: &ChainStep,
    ok: bool,
    seg: &[Event],
    all: &[Event],
    frts: (u64, u64),
    wpr_meta_addr: u64,
    command_before: &mut Option<u32>,
    out: &mut ChainJournal,
) -> StepLog {
    let mut bar0 = Vec::new();
    let mut cfg = Vec::new();
    let mut other_bar = false;
    for e in seg {
        match *e {
            Event::MmioWrite {
                bar: 0,
                offset,
                written,
                ..
            } => bar0.push((offset, written)),
            Event::MmioWrite { .. } => other_bar = true,
            Event::ConfigWrite {
                offset,
                before,
                written,
                after,
            } => cfg.push((offset, before, written, after)),
            _ => {}
        }
    }
    let gsp_doorbells = bar0
        .iter()
        .filter(|&&(o, v)| o == GSP_DOORBELL && v == 0)
        .count() as u32;
    let work_doorbells = bar0.iter().filter(|&&(o, _)| o == WORK_DOORBELL).count() as u32;
    let mut log = StepLog {
        id: String::from(step.id),
        ok,
        accesses: seg.len(),
        bar0_writes: bar0.len(),
        config_writes: cfg.len(),
        gsp_doorbells,
        work_doorbells,
        problem: None,
    };
    let wrote = !bar0.is_empty() || !cfg.is_empty() || other_bar;
    let problem: Option<String> = if !step.writes_hardware {
        if wrote {
            Some(String::from("pas doar-citire cu scrieri in jurnal"))
        } else if ok && step.id == "i6-wpr2-citit" {
            match i6_log::wpr2_after_i6_full(all, frts) {
                Some((_, _, Ok(_))) => None,
                Some((lo, hi, Err(e))) => {
                    Some(format!("WPR2 dupa I6: LO {lo:#010x} HI {hi:#010x}: {e}"))
                }
                None => Some(String::from("WPR2 dupa I6: nicio citire HI nenula")),
            }
        } else if ok && step.id == "i7-citit" {
            let cpuctl = seg.iter().rev().find_map(|e| match *e {
                Event::MmioRead {
                    bar: 0,
                    offset: i6_log::GSP_RISCV_CPUCTL,
                    value,
                } => Some(value),
                _ => None,
            });
            match cpuctl {
                Some(v) if v & (1 << 7) != 0 => None,
                Some(v) => Some(format!("GSP RISC-V nu e activ (CPUCTL {v:#010x})")),
                None => Some(String::from("lipseste citirea RISC-V CPUCTL")),
            }
        } else {
            None
        }
    } else if other_bar {
        Some(String::from("scriere in alt BAR decat BAR0"))
    } else {
        match step.id {
            "bus-master-pornit" | "bus-master-oprit" => {
                bus_master(step.id, ok, &bar0, &cfg, command_before)
            }
            "i6-fwsec-frts" | "i7-booter-gsp" if !cfg.is_empty() => {
                Some(String::from("scriere in configuratie in mijlocul pasului"))
            }
            "i6-fwsec-frts" if ok => match i6_log::check_i6(seg) {
                Ok(s) => {
                    out.i6 = Some(s);
                    None
                }
                Err(e) => Some(load_error("I6", e)),
            },
            "i7-booter-gsp"
                if ok
                    && bar0
                        .first()
                        .is_some_and(|&(o, _)| (0x11_0000..=0x11_3fff).contains(&o)) =>
            {
                match i6_log::check_i7_libos(seg, wpr_meta_addr, None) {
                    Ok(s) => {
                        out.i7 = Some(s.booter.clone());
                        out.i7_libos = Some(s);
                        None
                    }
                    Err(e) => Some(load_error("I7", e)),
                }
            }
            "i7-booter-gsp" if ok => match i6_log::check_i7(seg, wpr_meta_addr) {
                Ok(s) => {
                    out.i7 = Some(s);
                    None
                }
                Err(e) => Some(load_error("I7", e)),
            },
            _ if !cfg.is_empty() => Some(String::from("scriere in configuratie dupa Bus Master")),
            "i8-gsp-init" => i8(ok, &bar0),
            "i8e-golden" | "i9-canal-lucru" => {
                out.rm_doorbells += gsp_doorbells;
                rm_calls(ok, &bar0, gsp_doorbells)
            }
            "i10-add64" | "i10-cost-lansare" => bar0
                .iter()
                .find(|&&(o, _)| o != WORK_DOORBELL && !is_vram_window(o))
                .map(|&(o, _)| format!("scriere neasteptata in I10 la {o:#x}")),
            _ => None,
        }
    };
    log.problem = problem;
    log
}

fn load_error(what: &str, e: i6_log::I6LogError) -> String {
    format!(
        "{what}: scrierea #{} ({:#x}): {}",
        e.write_index, e.offset, e.what
    )
}

fn in_gsp_sec2(o: u64) -> bool {
    crate::gsp::falcon::WRITE_RANGES
        .iter()
        .any(|&(lo, hi)| (u64::from(lo)..=u64::from(hi)).contains(&o))
}

fn is_vram_window(o: u64) -> bool {
    crate::vram_window::CONTROL_REGISTERS
        .iter()
        .any(|&r| u64::from(r) == o)
}

fn bus_master(
    id: &str,
    ok: bool,
    bar0: &[(u64, u32)],
    cfg: &[(u16, u32, u32, u32)],
    command_before: &mut Option<u32>,
) -> Option<String> {
    if !bar0.is_empty() {
        return Some(String::from("scriere BAR0 la Bus Master"));
    }
    // A failed step that wrote nothing, or the restore when Bus Master was
    // never turned on (the OS has nothing to put back): nothing to check.
    if cfg.is_empty() && (!ok || (id == "bus-master-oprit" && command_before.is_none())) {
        return None;
    }
    let [(off, before, written, after)] = cfg else {
        return Some(format!(
            "{} scrieri in configuratie (asteptat 1)",
            cfg.len()
        ));
    };
    if *off != COMMAND {
        return Some(format!(
            "scriere in configuratie la {off:#x}, nu la Command"
        ));
    }
    if (after & 0xffff) != (written & 0xffff) {
        return Some(String::from("Command recitit diferit de ce s-a scris"));
    }
    if id == "bus-master-pornit" {
        *command_before = Some(*before);
        if written & 0b100 == 0 {
            return Some(String::from("Bus Master (bitul 2) nu e pornit"));
        }
        None
    } else {
        match command_before {
            Some(b) if (*b & 0xffff) == (written & 0xffff) => None,
            Some(b) => Some(format!(
                "Command pus {:#06x}, inainte era {:#06x}",
                written & 0xffff,
                *b & 0xffff
            )),
            None => Some(String::from("Bus Master oprit fara sa fi fost pornit")),
        }
    }
}

fn i8(ok: bool, bar0: &[(u64, u32)]) -> Option<String> {
    if let Some(&(o, _)) = bar0.iter().find(|&&(o, _)| !in_gsp_sec2(o)) {
        return Some(format!("scriere in afara ferestrelor GSP/SEC2 la {o:#x}"));
    }
    match bar0.first() {
        Some(&(GSP_DOORBELL, 0)) => None,
        Some(&(o, v)) => Some(format!(
            "prima scriere I8 trebuie soneria GSP, e {o:#x} = {v:#x}"
        )),
        None if ok => Some(String::from("I8 fara sonerie GSP")),
        None => None,
    }
}

fn rm_calls(ok: bool, bar0: &[(u64, u32)], bells: u32) -> Option<String> {
    if let Some(&(o, v)) = bar0
        .iter()
        .find(|&&(o, v)| !(o == GSP_DOORBELL && v == 0) && !is_vram_window(o))
    {
        return Some(format!(
            "scriere care nu e sonerie si nici fereastra VRAM: {o:#x} = {v:#x}"
        ));
    }
    if ok && bells == 0 {
        return Some(String::from("niciun apel RM sunat (0 sonerii GSP)"));
    }
    None
}

/// The window chain's step ids, in order (for tools that print a table).
pub fn step_ids() -> Vec<&'static str> {
    WINDOW_CHAIN_FULL.iter().map(|s| s.id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn step_lines_parse_with_and_without_prefix() {
        assert_eq!(
            parse_step("pas i8-gsp-init: ok"),
            Some(("i8-gsp-init", true))
        );
        assert_eq!(
            parse_step("[ 12.5] INFO  gpu: pas i8e-golden: ESEC"),
            Some(("i8e-golden", false))
        );
        assert_eq!(parse_step("pas i8e-golden: poate"), None);
        assert_eq!(parse_step("bar0 w 0x110c00: 0x0 <- 0x0 => 0x0"), None);
        assert_eq!(parse_step("nota: lant (C-54): incepe"), None);
    }

    #[test]
    fn an_empty_journal_is_to_analyse() {
        let r = check_chain_journal("", (0, 1), 0, None);
        assert!(matches!(r.verdict(), ChainVerdict::ToAnalyse(_)));
        assert_eq!(step_ids().len(), WINDOW_CHAIN_FULL.len());
    }
}
