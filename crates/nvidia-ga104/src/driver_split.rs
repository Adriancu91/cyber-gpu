//! C-147 point 7 — where the driver's time of one token goes, read from the timestamps of a stick journal.
//!
//! The OS prints every journal row with its boot clock (`[  125.776468] gpu: …`). During `kronos --gpu` one token is
//! always the same rows: the VRAM window moved (`bar0 w 0x001700`) for the words written since the last token
//! (launch slots, the zeroed timestamp records, the pushbuffer, the ring), the TLB invalidation, the doorbell
//! (`bar0 w 0xbb0090`), then the window moved again to wait for and read the records. The rows of the window moves
//! are the only marks we have, so [`split_by_bell`] cuts the text at every doorbell and adds the time between two
//! consecutive rows to the window that was open at the first of them (its base, the value written in `0x1700`), or
//! to "no window" when the window was back at its original value (the CPU between two windows: reading the logits,
//! the argmax, building the next token's launches, the journal).
//!
//! What it does NOT know: what the words in each window are (the base only says where in VRAM they are; the
//! meaning is ours, DEDUS, written in `docs/C-147_PUNCTUL7_ESTIMARE.md`), nor how much of the time in the window
//! of the records is the board computing. Host analysis only; nothing here touches a board.

use alloc::vec::Vec;

/// `NV_PBUS_BAR0_WINDOW` (the VRAM window register) as the journal prints it.
pub const WINDOW_ROW: &str = "bar0 w 0x001700:";
/// The work channel's doorbell as the journal prints it.
pub const BELL_ROW: &str = "bar0 w 0xbb0090:";

/// The boot clock of a journal row, in µs (`[  125.776468] …` → 125 776 468). `None` for a row without it.
pub fn row_time_us(line: &str) -> Option<u64> {
    let rest = line.trim_start().strip_prefix('[')?;
    let (t, _) = rest.split_once(']')?;
    let (s, frac) = t.trim().split_once('.')?;
    let s: u64 = s.parse().ok()?;
    if frac.is_empty() || frac.len() > 9 || !frac.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut us: u64 = 0;
    for i in 0..6 {
        us = us * 10 + frac.as_bytes().get(i).map_or(0, |b| u64::from(b - b'0'));
    }
    s.checked_mul(1_000_000)?.checked_add(us)
}

/// The value written by a `bar0 w` row (`… OLD <- NEW => GOT` → NEW) and the value it replaced (OLD).
fn written(line: &str, row: &str) -> Option<(u32, u32)> {
    let rest = line.split(row).nth(1)?;
    let (old, rest) = rest.split_once("<-")?;
    let new = rest.split("=>").next()?;
    let hex = |s: &str| u32::from_str_radix(s.trim().strip_prefix("0x")?, 16).ok();
    Some((hex(old)?, hex(new)?))
}

/// The time of one doorbell-to-doorbell interval, split by the window open.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BellSplit {
    /// The clock of the doorbell that starts the interval (µs).
    pub start_us: u64,
    /// Until the next doorbell (µs).
    pub total_us: u64,
    /// (window base written in `0x1700`, µs with it open), in the order first seen.
    pub windows: Vec<(u32, u64)>,
    /// µs with the window at its original value (no VRAM access through it).
    pub outside_us: u64,
    /// Window moves in the interval.
    pub moves: u32,
}

impl BellSplit {
    /// µs with the window at `base` (0 if never opened there).
    pub fn window_us(&self, base: u32) -> u64 {
        self.windows.iter().find(|w| w.0 == base).map_or(0, |w| w.1)
    }

    fn add(&mut self, open: Option<u32>, dt: u64) {
        match open {
            None => self.outside_us += dt,
            Some(b) => match self.windows.iter_mut().find(|w| w.0 == b) {
                Some(w) => w.1 += dt,
                None => self.windows.push((b, dt)),
            },
        }
    }
}

/// Splits the journal text at every doorbell. The window's original value is the OLD of the first window row of
/// the text. Only complete intervals (a doorbell at both ends) are returned; rows without a clock are skipped.
pub fn split_by_bell(text: &str) -> Vec<BellSplit> {
    let mut out = Vec::new();
    let mut home: Option<u32> = None;
    let mut open: Option<u32> = None;
    let mut cur: Option<BellSplit> = None;
    let mut last_t: Option<u64> = None;
    for line in text.lines() {
        let Some(t) = row_time_us(line) else { continue };
        if let (Some(c), Some(lt)) = (cur.as_mut(), last_t) {
            c.add(open, t.saturating_sub(lt));
        }
        last_t = Some(t);
        if let Some((old, new)) = written(line, WINDOW_ROW) {
            let h = *home.get_or_insert(old);
            open = (new != h).then_some(new);
            if let Some(c) = cur.as_mut() {
                c.moves += 1;
            }
        } else if line.contains(BELL_ROW) {
            if let Some(mut c) = cur.take() {
                c.total_us = t.saturating_sub(c.start_us);
                out.push(c);
            }
            cur = Some(BellSplit {
                start_us: t,
                ..BellSplit::default()
            });
        }
    }
    out
}

/// The median of the steady intervals: those whose total is at most `1.5 ×` the median total (the first, cold
/// tokens and the ones with a journal flush to the stick are left out and counted).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SteadySplit {
    /// Intervals in the text / kept as steady.
    pub intervals: usize,
    pub steady: usize,
    pub total_us: u64,
    pub outside_us: u64,
    /// (base, median µs over the steady intervals; 0 where the window was not opened).
    pub windows: Vec<(u32, u64)>,
    /// The slow intervals left out (µs each, in order).
    pub slow: Vec<u64>,
}

fn median(mut v: Vec<u64>) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// See [`SteadySplit`]. `splits` from [`split_by_bell`]; only the intervals of one run should be given (the caller
/// picks them, e.g. the ones with the same number of window moves as most of the run).
pub fn steady(splits: &[BellSplit]) -> SteadySplit {
    let mid = median(splits.iter().map(|s| s.total_us).collect());
    let (keep, slow): (Vec<&BellSplit>, Vec<&BellSplit>) =
        splits.iter().partition(|s| s.total_us * 2 <= mid * 3);
    let mut bases: Vec<u32> = Vec::new();
    for s in &keep {
        for w in &s.windows {
            if !bases.contains(&w.0) {
                bases.push(w.0);
            }
        }
    }
    SteadySplit {
        intervals: splits.len(),
        steady: keep.len(),
        total_us: median(keep.iter().map(|s| s.total_us).collect()),
        outside_us: median(keep.iter().map(|s| s.outside_us).collect()),
        windows: bases
            .iter()
            .map(|&b| (b, median(keep.iter().map(|s| s.window_us(b)).collect())))
            .collect(),
        slow: slow.iter().map(|s| s.total_us).collect(),
    }
}

/// The intervals of the token loop: the longest run of consecutive intervals with the same number of window moves
/// (the generation repeats the same rows; the chain before it does not).
pub fn token_loop(splits: &[BellSplit]) -> &[BellSplit] {
    let (mut best, mut at, mut i) = (0usize, 0usize, 0usize);
    while i < splits.len() {
        let mut j = i + 1;
        while j < splits.len() && splits[j].moves == splits[i].moves {
            j += 1;
        }
        if j - i > best {
            (best, at) = (j - i, i);
        }
        i = j;
    }
    &splits[at..at + best]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Rows copied from the os-322 stick journal (hub `rezultate/os-automat/2026-11-20/gpu-jurnal-017.txt`, lines
    // 3731–3761): two tokens of `kronos --gpu`.
    const OS322: &str = "\
[  125.753495] gpu: bar0 w 0xbb0090: 0x00000000 <- 0x00000001 => 0x00000000
[  125.773164] gpu: nota: C-55 gpu lant: fereastra cu Adrian (P6-42)
[  125.773169] gpu: bar0 w 0x001700: 0x0001fff0 <- 0x00001b2f => 0x00001b2f
[  125.773575] gpu: bar0 w 0x001700: 0x00001b2f <- 0x0000101f => 0x0000101f
[  125.773583] gpu: bar0 w 0x001700: 0x0000101f <- 0x0001fff0 => 0x0001fff0
[  125.774378] gpu: nota: C-55 gpu lant: fereastra cu Adrian (P6-42)
[  125.774381] gpu: bar0 w 0x001700: 0x0001fff0 <- 0x00001b2d => 0x00001b2d
[  125.774911] gpu: bar0 w 0x001700: 0x00001b2d <- 0x000010e6 => 0x000010e6
[  125.776428] gpu: bar0 w 0x001700: 0x000010e6 <- 0x0000101f => 0x0000101f
[  125.776438] gpu: bar0 w 0x001700: 0x0000101f <- 0x0001fff0 => 0x0001fff0
[  125.776443] gpu: nota: C-55 gpu lant: fereastra cu Adrian (P6-42)
[  125.776447] gpu: bar0 w 0xb830a0: 0x00100000 <- 0x00100000 => 0x00100000
[  125.776453] gpu: bar0 w 0xb830a4: 0x00000000 <- 0x00000000 => 0x00000000
[  125.776458] gpu: bar0 w 0xb830b0: 0x00000001 <- 0x80000001 => 0x80000001
[  125.776465] gpu: nota: soneria canalului de lucru (I10): scriere fara recitire
[  125.776468] gpu: bar0 w 0xbb0090: 0x00000000 <- 0x00000001 => 0x00000000
[  125.776472] gpu: nota: C-55 gpu lant: fereastra cu Adrian (P6-42)
[  125.776476] gpu: bar0 w 0x001700: 0x0001fff0 <- 0x00001b2f => 0x00001b2f
[  125.779381] gpu: bar0 w 0x001700: 0x00001b2f <- 0x0000101f => 0x0000101f
[  125.779388] gpu: bar0 w 0x001700: 0x0000101f <- 0x0001fff0 => 0x0001fff0
[  125.780456] gpu: nota: C-55 gpu lant: fereastra cu Adrian (P6-42)
[  125.780459] gpu: bar0 w 0x001700: 0x0001fff0 <- 0x00001b2d => 0x00001b2d
[  125.780988] gpu: bar0 w 0x001700: 0x00001b2d <- 0x000010e6 => 0x000010e6
[  125.782510] gpu: bar0 w 0x001700: 0x000010e6 <- 0x0000101f => 0x0000101f
[  125.782519] gpu: bar0 w 0x001700: 0x0000101f <- 0x0001fff0 => 0x0001fff0
[  125.782524] gpu: nota: C-55 gpu lant: fereastra cu Adrian (P6-42)
[  125.782528] gpu: bar0 w 0xb830a0: 0x00100000 <- 0x00100000 => 0x00100000
[  125.782534] gpu: bar0 w 0xb830a4: 0x00000000 <- 0x00000000 => 0x00000000
[  125.782539] gpu: bar0 w 0xb830b0: 0x00000001 <- 0x80000001 => 0x80000001
[  125.782545] gpu: nota: soneria canalului de lucru (I10): scriere fara recitire
[  125.782548] gpu: bar0 w 0xbb0090: 0x00000000 <- 0x00000001 => 0x00000000
";

    #[test]
    fn clock_of_a_row() {
        assert_eq!(row_time_us("[  125.776468] gpu: x"), Some(125_776_468));
        assert_eq!(row_time_us("[    0.5] gpu: x"), Some(500_000));
        assert_eq!(row_time_us("gpu: x"), None);
        assert_eq!(row_time_us("[ 1.2x] gpu"), None);
    }

    #[test]
    fn two_tokens_of_os322() {
        let s = split_by_bell(OS322);
        assert_eq!(s.len(), 2);
        let t = &s[1];
        assert_eq!(t.total_us, 125_782_548 - 125_776_468);
        // waiting for the records (and reading them): 2 905 µs; the CPU between: 1 068 + the small gaps.
        assert_eq!(t.window_us(0x1b2f), 125_779_381 - 125_776_476);
        assert_eq!(t.window_us(0x10e6), 125_782_510 - 125_780_988);
        assert_eq!(t.window_us(0x1b2d), 125_780_988 - 125_780_459);
        assert_eq!(t.moves, 7);
        let sum: u64 = t.windows.iter().map(|w| w.1).sum::<u64>() + t.outside_us;
        assert_eq!(sum, t.total_us, "every µs counted once");
        assert!(
            t.outside_us > 1_000 && t.outside_us < 1_200,
            "{}",
            t.outside_us
        );
    }

    #[test]
    fn steady_leaves_out_the_slow_ones() {
        let mk = |total| BellSplit {
            total_us: total,
            outside_us: total / 4,
            windows: alloc::vec![(7, total / 2)],
            ..BellSplit::default()
        };
        let v = [mk(6_000), mk(22_000), mk(5_800), mk(56_000), mk(5_900)];
        let s = steady(&v);
        assert_eq!((s.intervals, s.steady), (5, 3));
        assert_eq!(s.total_us, 5_900);
        assert_eq!(s.windows, alloc::vec![(7, 2_950)]);
        assert_eq!(s.slow, alloc::vec![22_000, 56_000]);
    }

    #[test]
    fn token_loop_is_the_longest_same_shape_run() {
        let mk = |moves| BellSplit {
            moves,
            ..BellSplit::default()
        };
        let v = [mk(40), mk(3), mk(7), mk(7), mk(7), mk(75)];
        assert_eq!(token_loop(&v).len(), 3);
        assert!(token_loop(&[]).is_empty());
    }
}
