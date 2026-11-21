//! C-157 point 3 — "the diet, part two": what is left of the driver's time per token after the first diet and P3
//! (os-328, #513, the glued list: `kt median driver (us)` = **759 µs**), and helpers that take most of it away
//! **without sending anything new to the board** (the same programs, launches, pushbuffer bytes and register values).
//!
//! Where the 759 µs go (DEDUS; `docs/C-157_P3_DIETA_DOI.md`: the journal's own clock + `cyber-universe` `gpu_i14.rs`
//! / `gpu_i11.rs` read + the host bench `examples/dieta_doi.rs`):
//! - the 89 timestamp records read word by word through the VRAM window after every token: 356 reads, ~166 µs;
//! - the wait for the last record polls every **100 µs** (`run_records_inner`): ~50 µs late on average;
//! - the logits' FNV-1a fingerprint (64 KiB, byte by byte, inside the token's clock): ~100–150 µs;
//! - the argmax on floats in a **soft-float** kernel: 85 µs measured (16 384 emulated compares);
//! - the rest (building the launches, the shadows, the pushbuffer, the ring): ~150 µs.
//!
//! What this module gives the OS (all by addition, all off unless the OS turns them on):
//! - [`RecordPlan`] + [`ends_span`]: read **two** records (the first and the last, 8 words) on most tokens, all of
//!   them only on the positions that feed the `kp` profile. The board's span and busy time of a token are the same
//!   number either way (the durations of [`crate::transfer::analyze_sequence`] add up to last − first).
//! - [`WaitPlan`] + [`poll_planned`]: sleep through most of the board's known time, then poll every few µs.
//! - `gpu_compute::model::argmax_bits`: the argmax on the logits' bits, integer compares only (same answer).
//! - The fingerprint stays the same function; the OS computes it **after** the token's clock stops (it is a test
//!   aid, not part of generating). Nothing to add here for that.
//! - [`Diet2Stats`]: `kt` rows to see the effect on the board, and their reader [`parse_diet2`].
//!
//! Tested only in SIMULARE (`tests/sim_c157_diet2.rs`, T147). Nothing here is proven on the RTX 3070 Ti.

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::Clock;

use crate::launch_cost::RECORD;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

// ------------------------------------------------------------------------------------------------ the records

/// Which positions read all the timestamp records of their submission (for the `kp` profile by kind) and which read
/// only the first and the last (the token's board span). `full_every` = 1 reads all of them always (as today).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordPlan {
    pub full_every: usize,
}

impl RecordPlan {
    /// Every 8th position (and positions 0 and 1) feeds the profile: 4 of 32 positions are still a median.
    pub const FULL_EVERY: usize = 8;
    /// Today's behaviour: every record of every token.
    pub const ALL: RecordPlan = RecordPlan { full_every: 1 };
    pub const DIET: RecordPlan = RecordPlan {
        full_every: Self::FULL_EVERY,
    };

    /// `true` when position `pos` reads all its records.
    pub fn reads_all(&self, pos: usize) -> bool {
        self.full_every <= 1 || pos < 2 || pos.is_multiple_of(self.full_every)
    }

    /// The 32-bit words read through the VRAM window for a submission of `n` launches (`n + 1` records).
    pub fn words(&self, n: usize, pos: usize) -> usize {
        let per = RECORD as usize / 4;
        if self.reads_all(pos) {
            per * (n + 1)
        } else {
            2 * per
        }
    }
}

fn payload(r: &[u8]) -> GpuResult<u32> {
    let b: [u8; 4] = r
        .get(..4)
        .and_then(|s| s.try_into().ok())
        .ok_or(bad("inregistrare scurta"))?;
    Ok(u32::from_le_bytes(b))
}

fn time_ns(r: &[u8]) -> GpuResult<u64> {
    let b: [u8; 8] = r
        .get(8..16)
        .and_then(|s| s.try_into().ok())
        .ok_or(bad("inregistrare scurta"))?;
    Ok(u64::from_le_bytes(b))
}

/// The board's span of a submission of `n` launches (ns, record `n` − record 0) from only its first and last record
/// (16 bytes each, as [`crate::transfer::analyze_sequence`] reads them). Checks what can be checked with two
/// records: record 0 carries `first_payload`, record `n` carries `first_payload + n`, the clock did not go back.
/// The channel runs the launches in order, so the last payload means all of them finished; which launch took how
/// long is NOT known (that is what the full read on [`RecordPlan::reads_all`] positions is for).
pub fn ends_span(first: &[u8], last: &[u8], n: usize, first_payload: u32) -> GpuResult<u64> {
    if payload(first)? != first_payload {
        return Err(bad(
            "secventa: inregistrarea 0 lipseste (canalul nu a pornit)",
        ));
    }
    if payload(last)? != first_payload.wrapping_add(n as u32) {
        return Err(bad(
            "secventa: ultima inregistrare nu are valoarea asteptata",
        ));
    }
    let (t0, t1) = (time_ns(first)?, time_ns(last)?);
    t1.checked_sub(t0)
        .ok_or(bad("secventa: ceasul placii merge inapoi"))
}

// ------------------------------------------------------------------------------------------------ the wait

/// How the driver waits for the last record. Today (`gpu_i11.rs` `run_records_inner`): one read every 100 µs, so a
/// token ends on average ~50 µs after the board finished. The plan: sleep through the board's shortest recent time
/// minus a margin, then read every [`Self::fine_us`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WaitPlan {
    /// The read step after the first sleep (µs).
    pub fine_us: u32,
    /// How much earlier than the shortest recent board time the reads start (µs).
    pub margin_us: u32,
    /// The longest single delay while sleeping (the abort key is asked between them).
    pub coarse_us: u32,
    recent: Vec<u64>,
}

impl WaitPlan {
    pub const FINE_US: u32 = 5;
    pub const MARGIN_US: u32 = 100;
    pub const COARSE_US: u32 = 100;
    /// Board times remembered (µs).
    pub const RECENT: usize = 4;

    pub fn new() -> Self {
        WaitPlan {
            fine_us: Self::FINE_US,
            margin_us: Self::MARGIN_US,
            coarse_us: Self::COARSE_US,
            recent: Vec::new(),
        }
    }

    /// The board took `board_us` for the last token (its own clock, `ends_span` / 1000).
    pub fn observe(&mut self, board_us: u64) {
        if self.recent.len() == Self::RECENT {
            self.recent.remove(0);
        }
        self.recent.push(board_us);
    }

    /// After an error, or a submission that is not a token of the same list: no sleep until new times are seen.
    pub fn forget(&mut self) {
        self.recent.clear();
    }

    /// The first sleep (µs): the shortest recent board time minus the margin; 0 with nothing seen.
    pub fn first_sleep_us(&self) -> u64 {
        self.recent
            .iter()
            .min()
            .map_or(0, |m| m.saturating_sub(u64::from(self.margin_us)))
    }
}

impl Default for WaitPlan {
    fn default() -> Self {
        Self::new()
    }
}

/// What one planned wait did.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WaitStats {
    /// Reads of the condition (each one a VRAM-window read on the board).
    pub polls: u32,
    /// µs from the start of the wait to the read that saw the value.
    pub waited_us: u64,
}

/// [`gpu_core::hal::poll_until`] with a [`WaitPlan`]: one read at once, then sleep [`WaitPlan::first_sleep_us`] (in
/// delays of at most `coarse_us`, asking the abort key between them), then a read every `fine_us`, up to
/// `timeout_us` in all. Never spins without a bound; a timeout or abort is an error, as in `poll_until`.
pub fn poll_planned<C: Clock + ?Sized, T>(
    clock: &mut C,
    timeout_us: u64,
    plan: &WaitPlan,
    mut cond: impl FnMut() -> Option<T>,
) -> GpuResult<(T, WaitStats)> {
    let start = clock.now_us();
    let mut st = WaitStats::default();
    let sleep = plan.first_sleep_us().min(timeout_us);
    let mut asleep = true;
    loop {
        st.polls += 1;
        if let Some(v) = cond() {
            st.waited_us = clock.now_us().saturating_sub(start);
            return Ok((v, st));
        }
        if asleep {
            // No reads while sleeping: delays of at most `coarse_us`, the abort key asked between them.
            loop {
                let waited = clock.now_us().saturating_sub(start);
                if waited >= sleep {
                    break;
                }
                if clock.abort_requested() {
                    return Err(GpuError::Aborted { waited_us: waited });
                }
                let left = (sleep - waited).min(u64::from(plan.coarse_us.max(1)));
                clock.delay_us(u32::try_from(left).unwrap_or(1));
            }
            asleep = false;
            continue;
        }
        let waited = clock.now_us().saturating_sub(start);
        if waited >= timeout_us {
            return Err(GpuError::Timeout { waited_us: waited });
        }
        if clock.abort_requested() {
            return Err(GpuError::Aborted { waited_us: waited });
        }
        clock.delay_us(plan.fine_us.max(1));
    }
}

// ------------------------------------------------------------------------------------------------ the rows

/// Per-token figures of the second diet, for `kt` rows (the OS pushes one entry per token).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Diet2Stats {
    /// Words of records read per token.
    pub record_words: Vec<u32>,
    /// Reads of the wait per token.
    pub polls: Vec<u32>,
}

fn median(v: &[u32]) -> u32 {
    if v.is_empty() {
        return 0;
    }
    let mut s = v.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

impl Diet2Stats {
    pub fn add(&mut self, record_words: usize, polls: u32) {
        self.record_words
            .push(u32::try_from(record_words).unwrap_or(u32::MAX));
        self.polls.push(polls);
    }

    /// `kt …` rows; `on` = whether the second diet was on.
    pub fn rows(&self, on: bool) -> Vec<(&'static str, u32)> {
        alloc::vec![
            ("kt dieta 2 (0/1)", u32::from(on)),
            (
                "kt cuvinte de inregistrari citite pe simbol (median)",
                median(&self.record_words)
            ),
            (
                "kt citiri ale asteptarii pe simbol (median)",
                median(&self.polls)
            ),
        ]
    }

    pub fn journal(&self, on: bool, j: &mut Journal) {
        for (what, value) in self.rows(on) {
            j.push(Event::Value { what, value });
        }
    }
}

/// The rows of [`Diet2Stats::rows`] read back from a stick journal (`None` = missing; the last value wins).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParsedDiet2 {
    pub on: Option<u32>,
    pub record_words: Option<u32>,
    pub polls: Option<u32>,
}

pub fn parse_diet2(text: &str) -> ParsedDiet2 {
    let mut p = ParsedDiet2::default();
    for line in text.lines() {
        let Some(rest) = line.split("gpu: valoare kt ").nth(1) else {
            continue;
        };
        let Some((name, val)) = rest.rsplit_once(" = 0x") else {
            continue;
        };
        let Ok(v) = u32::from_str_radix(val.trim(), 16) else {
            continue;
        };
        let slot = match name {
            "dieta 2 (0/1)" => &mut p.on,
            "cuvinte de inregistrari citite pe simbol (median)" => &mut p.record_words,
            "citiri ale asteptarii pe simbol (median)" => &mut p.polls,
            _ => continue,
        };
        *slot = Some(v);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(payload: u32, t: u64) -> [u8; 16] {
        let mut r = [0u8; 16];
        r[..4].copy_from_slice(&payload.to_le_bytes());
        r[8..].copy_from_slice(&t.to_le_bytes());
        r
    }

    #[test]
    fn record_plan() {
        let p = RecordPlan::DIET;
        assert!(p.reads_all(0) && p.reads_all(1) && p.reads_all(8) && p.reads_all(16));
        assert!(!p.reads_all(2) && !p.reads_all(9));
        assert_eq!(p.words(88, 3), 8);
        assert_eq!(p.words(88, 8), 4 * 89);
        assert!((0..40).all(|pos| RecordPlan::ALL.reads_all(pos)));
    }

    #[test]
    fn ends() {
        assert_eq!(
            ends_span(&rec(0x100, 5_000), &rec(0x100 + 88, 2_426_000), 88, 0x100),
            Ok(2_421_000)
        );
        assert!(ends_span(&rec(0xff, 0), &rec(0x100 + 88, 1), 88, 0x100).is_err());
        assert!(ends_span(&rec(0x100, 0), &rec(0x100 + 87, 1), 88, 0x100).is_err());
        assert!(ends_span(&rec(0x100, 9), &rec(0x100 + 88, 1), 88, 0x100).is_err());
        assert!(ends_span(&[0; 8], &rec(0x100 + 88, 1), 88, 0x100).is_err());
    }

    struct Fake {
        now: u64,
        delays: Vec<u32>,
        abort_at: Option<u64>,
    }

    impl Clock for Fake {
        fn now_us(&mut self) -> u64 {
            self.now
        }
        fn delay_us(&mut self, us: u32) {
            self.delays.push(us);
            self.now += u64::from(us);
        }
        fn abort_requested(&mut self) -> bool {
            self.abort_at.is_some_and(|a| self.now >= a)
        }
    }

    fn fake() -> Fake {
        Fake {
            now: 1_000,
            delays: Vec::new(),
            abort_at: None,
        }
    }

    #[test]
    fn plan_sleeps_then_polls_finely() {
        // A clock that the condition can read: the condition sees the fake clock's time through a shared cell.
        let t = core::cell::Cell::new(0u64);
        struct Shared<'a>(&'a core::cell::Cell<u64>, Vec<u32>);
        impl Clock for Shared<'_> {
            fn now_us(&mut self) -> u64 {
                self.0.get()
            }
            fn delay_us(&mut self, us: u32) {
                self.1.push(us);
                self.0.set(self.0.get() + u64::from(us));
            }
        }
        // Board done 2 421 µs after the doorbell; recent board times 2 430, 2 421, 2 440.
        let mut plan = WaitPlan::new();
        [2_430, 2_421, 2_440]
            .into_iter()
            .for_each(|b| plan.observe(b));
        assert_eq!(plan.first_sleep_us(), 2_321);
        let mut c = Shared(&t, Vec::new());
        let (_, st) = poll_planned(&mut c, 1_000_000, &plan, || {
            (t.get() >= 2_421).then_some(())
        })
        .unwrap();
        assert!(
            st.waited_us >= 2_421 && st.waited_us < 2_421 + WaitPlan::FINE_US as u64,
            "{st:?}"
        );
        // 1 read at once + 21 fine reads (2 321 .. 2 421 by 5) — far fewer than late reads every 100 µs.
        assert!(st.polls <= 23, "{st:?}");
        assert!(c.1.iter().all(|&d| d <= WaitPlan::COARSE_US));
        // Today's wait (poll_until, 100 µs): ends at 2 500, 79 µs late.
        t.set(0);
        let mut c = Shared(&t, Vec::new());
        gpu_core::hal::poll_until(&mut c, 1_000_000, 100, || (t.get() >= 2_421).then_some(()))
            .unwrap();
        assert_eq!(t.get(), 2_500);
        // Nothing seen yet: fine reads from the start (never a sleep past the board).
        t.set(0);
        let mut c = Shared(&t, Vec::new());
        let (_, st) = poll_planned(&mut c, 1_000_000, &WaitPlan::new(), || {
            (t.get() >= 40).then_some(())
        })
        .unwrap();
        assert_eq!(st.waited_us, 40);
        // A board faster than the sleep: late by at most the sleep (bounded by the margin when times are steady).
        t.set(0);
        let mut c = Shared(&t, Vec::new());
        let (_, st) = poll_planned(&mut c, 1_000_000, &plan, || {
            (t.get() >= 2_000).then_some(())
        })
        .unwrap();
        assert_eq!(st.waited_us, 2_321);
        // Timeout and abort stay errors.
        t.set(0);
        let mut c = Shared(&t, Vec::new());
        assert!(matches!(
            poll_planned(&mut c, 3_000, &plan, || None::<()>),
            Err(GpuError::Timeout { .. })
        ));
        let mut f = fake();
        f.abort_at = Some(1_500);
        assert!(matches!(
            poll_planned(&mut f, 1_000_000, &plan, || None::<()>),
            Err(GpuError::Aborted { .. })
        ));
        // forget(): no sleep.
        plan.forget();
        assert_eq!(plan.first_sleep_us(), 0);
    }

    #[test]
    fn rows_and_parse() {
        let mut s = Diet2Stats::default();
        s.add(8, 22);
        s.add(356, 24);
        s.add(8, 21);
        let mut j = Journal::new(64);
        s.journal(true, &mut j);
        let text: alloc::string::String = j
            .events()
            .map(|e| alloc::format!("[  1.000000] gpu: {e}\n"))
            .collect();
        let p = parse_diet2(&text);
        assert_eq!(
            p,
            ParsedDiet2 {
                on: Some(1),
                record_words: Some(8),
                polls: Some(22)
            }
        );
    }
}
