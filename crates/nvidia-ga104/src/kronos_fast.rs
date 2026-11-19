//! C-145 — the driver's pauses in `kronos --gpu`, and three ways to shorten them without touching what is sent to
//! the board.
//!
//! The first automatic cycle (`os-307`, 19 Nov 2026, gpu-jurnal-011) measured on the RTX 3070 Ti: the launches of one
//! token take a median 22.9 ms (board clock), the whole token as the OS runs it 294.8 ms (OS clock, without the CPU
//! check), and a position with the CPU check ~4.16 s. Read from the journal's own timestamps
//! (`docs/C-145_PAUZELE_DRIVERULUI.md`): every journal row is written **and flushed** on the stick before the next
//! access (~16.4 ms per row, 15 rows per token ≈ 246 ms), the launch area (196 × 512 B) is rewritten word by word
//! through the VRAM window with a read-back of every word (~40 ms), and the reference CPU engine (1 core) checks every
//! position for ~3.87 s **between** two tokens.
//!
//! What this module gives the OS (all by addition, all off unless the OS turns them on):
//! - [`gen_routine`]: the routine accesses of a generation (the VRAM window, the TLB invalidate, the work-channel
//!   doorbell, and the notes that come with them) for `gpu_core::diag::Journal::set_routine` — they still reach the
//!   journal, in order, with their values; only the flush of the stick waits (at most [`GEN_MAX_PENDING`] rows, and
//!   the OS flushes after every token). Steps, values (`kg`/`kv`/`kt`), reads and every other write go out at once.
//! - [`WriteShadow`]: the last bytes written into a host-only area (the launch slots), so only the words that changed
//!   since the previous token are written (and read back) again. Never for areas the GPU writes.
//! - [`TimeSplit`]: the time of every token split into board / driver / journal / screen / CPU check, with `kt …`
//!   journal rows and their reader, and [`FIGURES`]: what each of the existing speed figures includes.
//! - `kronos_gen::Generation::run_check_after`: the CPU check after the board, not between tokens.
//!
//! Nothing here changes a pushbuffer, a launch, a register value or the order of the accesses: the board receives
//! exactly what it received at `os-307`. Tested only in SIMULARE (`tests/sim_c145.rs`); the speed on the board is an
//! estimate until the automatic cycle measures it.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

use gpu_core::diag::{Event, Journal, Routine};
use gpu_core::error::{GpuError, GpuResult};

use crate::chain_log::{VRAM_WINDOW, WORK_DOORBELL};
use crate::vram_window::CONTROL_REGISTERS;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

// ------------------------------------------------------------------------------------------------ routine rows

/// BAR0 writes a generation repeats for every token: the four registers `vram_window` may write (the window and the
/// three TLB-invalidate registers) and the work-channel doorbell.
pub const GEN_WRITES: [u64; 5] = [
    CONTROL_REGISTERS[0] as u64,
    CONTROL_REGISTERS[1] as u64,
    CONTROL_REGISTERS[2] as u64,
    CONTROL_REGISTERS[3] as u64,
    WORK_DOORBELL,
];

const _: () = assert!(GEN_WRITES[0] == VRAM_WINDOW);

/// Most routine rows held before the stick is flushed anyway (one token is ~15 rows; the upload ~24 per piece).
/// After a hang, at most this many rows are missing from the stick (they are in the memory journal).
pub const GEN_MAX_PENDING: u32 = 64;

/// The routine of a generation; `notes` = the notes the OS repeats with them (the permit's reason, the doorbell's
/// note), compared by text.
pub fn gen_routine(notes: &'static [&'static str]) -> Routine {
    Routine {
        writes: &GEN_WRITES,
        notes,
        max_pending: GEN_MAX_PENDING,
    }
}

// ------------------------------------------------------------------------------------------------ write shadow

/// What was last written into a host-only area of `len` bytes at VRAM `base` (word by word, and which words are
/// known). [`Self::changes`] gives the runs of words that differ (or were never written); the caller writes them,
/// reads them back as before, and only then [`Self::remember`]s them. After any error: [`Self::forget`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WriteShadow {
    base: u64,
    words: Vec<u32>,
    known: Vec<bool>,
}

impl WriteShadow {
    /// Largest area a shadow follows (the launch area of a 24-layer model is ~200 KiB).
    pub const MAX_LEN: u64 = 4 << 20;

    pub fn new(base: u64, len: u64) -> GpuResult<Self> {
        if !base.is_multiple_of(4) || !len.is_multiple_of(4) || len == 0 || len > Self::MAX_LEN {
            return Err(bad("umbra: zona nealiniata, goala sau prea mare"));
        }
        let n = (len / 4) as usize;
        Ok(WriteShadow {
            base,
            words: alloc::vec![0; n],
            known: alloc::vec![false; n],
        })
    }

    fn index(&self, pa: u64, len: usize) -> GpuResult<usize> {
        if !pa.is_multiple_of(4) || !len.is_multiple_of(4) || pa < self.base {
            return Err(bad("umbra: adresa sau marime nealiniata"));
        }
        let i = ((pa - self.base) / 4) as usize;
        if i + len / 4 > self.words.len() {
            return Err(bad("umbra: scriere in afara zonei"));
        }
        Ok(i)
    }

    /// The runs of `new` (written at `pa`) that differ from what is known: `(pa of the run, byte range in new)`,
    /// in order, adjacent words merged.
    pub fn changes(&self, pa: u64, new: &[u8]) -> GpuResult<Vec<(u64, Range<usize>)>> {
        let i0 = self.index(pa, new.len())?;
        let mut out: Vec<(u64, Range<usize>)> = Vec::new();
        for (k, c) in new.as_chunks::<4>().0.iter().enumerate() {
            let w = u32::from_le_bytes(*c);
            if self.known[i0 + k] && self.words[i0 + k] == w {
                continue;
            }
            let b = 4 * k;
            match out.last_mut() {
                Some((_, r)) if r.end == b => r.end = b + 4,
                _ => out.push((pa + b as u64, b..b + 4)),
            }
        }
        Ok(out)
    }

    /// `new` is now at `pa` (written and read back).
    pub fn remember(&mut self, pa: u64, new: &[u8]) -> GpuResult<()> {
        let i0 = self.index(pa, new.len())?;
        for (k, c) in new.as_chunks::<4>().0.iter().enumerate() {
            self.words[i0 + k] = u32::from_le_bytes(*c);
            self.known[i0 + k] = true;
        }
        Ok(())
    }

    /// Nothing is known any more (after an error, or when something else may have written the area).
    pub fn forget(&mut self) {
        self.known.iter_mut().for_each(|k| *k = false);
    }

    pub fn known_words(&self) -> usize {
        self.known.iter().filter(|&&k| k).count()
    }
}

// ------------------------------------------------------------------------------------------------ time split

/// One token's time, split. `host_us` is what the OS already measures (P6-95: from the first launch written to the
/// logits read, without the CPU check); `board_us` the board's span of the token's records; `journal_us` the time
/// spent inside the stick journal during the token; `screen_us` the time printing during the token; `check_us` the
/// CPU check of that position (outside `host_us`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TokenSplit {
    pub pos: usize,
    pub host_us: u64,
    pub board_us: u64,
    pub journal_us: u64,
    pub screen_us: u64,
    pub check_us: u64,
}

impl TokenSplit {
    /// The rest of `host_us`: the driver's own work (writing the launches through the window, the doorbell, waiting
    /// for the records, reading the logits).
    pub fn driver_us(&self) -> u64 {
        self.host_us
            .saturating_sub(self.board_us)
            .saturating_sub(self.journal_us)
            .saturating_sub(self.screen_us)
    }

    /// What a user waits for this position: the token and its CPU check.
    pub fn total_us(&self) -> u64 {
        self.host_us + self.check_us
    }
}

/// Medians of the split (µs) and the rates they give (× 100).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SplitSummary {
    pub tokens: usize,
    pub board_us: u64,
    pub driver_us: u64,
    pub journal_us: u64,
    pub screen_us: u64,
    pub check_us: u64,
    pub host_us: u64,
    pub total_us: u64,
    /// Tokens per second × 100 without the check (median `host_us`) and with it (median `total_us`).
    pub per_s_x100: u64,
    pub per_s_check_x100: u64,
}

/// The split of every token of a generation.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TimeSplit {
    pub tokens: Vec<TokenSplit>,
}

fn median(mut v: Vec<u64>) -> u64 {
    v.sort_unstable();
    v.get(v.len() / 2).copied().unwrap_or(0)
}

fn rate_x100(us: u64) -> u64 {
    100_000_000u64.checked_div(us).unwrap_or(0)
}

impl TimeSplit {
    pub fn new() -> Self {
        Self::default()
    }

    /// Refuses a part larger than the token's host time (the clocks would be mixed up).
    pub fn add(&mut self, t: TokenSplit) -> GpuResult<()> {
        if t.board_us.max(t.journal_us).max(t.screen_us) > t.host_us {
            return Err(bad("timpul pe simbol: o parte mai mare decat intregul"));
        }
        self.tokens.push(t);
        Ok(())
    }

    pub fn summary(&self) -> SplitSummary {
        let m = |f: fn(&TokenSplit) -> u64| median(self.tokens.iter().map(f).collect());
        let host = m(|t| t.host_us);
        let total = m(|t| t.total_us());
        SplitSummary {
            tokens: self.tokens.len(),
            board_us: m(|t| t.board_us),
            driver_us: m(|t| t.driver_us()),
            journal_us: m(|t| t.journal_us),
            screen_us: m(|t| t.screen_us),
            check_us: m(|t| t.check_us),
            host_us: host,
            total_us: total,
            per_s_x100: rate_x100(host),
            per_s_check_x100: rate_x100(total),
        }
    }

    /// `kt …` journal rows (values, µs, saturated to 32 bits).
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        let s = self.summary();
        let c = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        alloc::vec![
            ("kt simboluri", c(s.tokens as u64)),
            ("kt median placa (us)", c(s.board_us)),
            ("kt median driver (us)", c(s.driver_us)),
            ("kt median jurnal (us)", c(s.journal_us)),
            ("kt median ecran (us)", c(s.screen_us)),
            ("kt median verificare procesor (us)", c(s.check_us)),
            ("kt median simbol fara verificare (us)", c(s.host_us)),
            ("kt median simbol cu verificare (us)", c(s.total_us)),
            (
                "kt simboluri pe secunda x100 (fara verificare)",
                c(s.per_s_x100)
            ),
            (
                "kt simboluri pe secunda x100 (cu verificare)",
                c(s.per_s_check_x100)
            ),
        ]
    }

    pub fn journal(&self, j: &mut Journal) {
        for (what, value) in self.rows() {
            j.push(Event::Value { what, value });
        }
    }

    /// Lines for the report (`label`: "SIMULARE" or "placa").
    pub fn lines(&self, label: &str) -> Vec<String> {
        let s = self.summary();
        let r = |x: u64| format!("{}.{:02}", x / 100, x % 100);
        alloc::vec![
            format!(
                "timpul pe simbol ({label}, median, {} simboluri): placa {} us, driver {} us, jurnal {} us, ecran {} us = {} us ({} simboluri/s)",
                s.tokens,
                s.board_us,
                s.driver_us,
                s.journal_us,
                s.screen_us,
                s.host_us,
                r(s.per_s_x100)
            ),
            format!(
                "verificarea pe procesor ({label}): median {} us pe pozitie; cu ea {} us pe simbol ({} simboluri/s)",
                s.check_us,
                s.total_us,
                r(s.per_s_check_x100)
            ),
        ]
    }
}

/// The `kt` rows read back from a stick journal (`None` = row missing).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParsedSplit {
    pub tokens: Option<u32>,
    pub board_us: Option<u32>,
    pub driver_us: Option<u32>,
    pub journal_us: Option<u32>,
    pub screen_us: Option<u32>,
    pub check_us: Option<u32>,
    pub host_us: Option<u32>,
    pub total_us: Option<u32>,
    pub per_s_x100: Option<u32>,
    pub per_s_check_x100: Option<u32>,
}

/// Reads the `kt` rows (`gpu: valoare kt … = 0x…`) of a stick journal; the last value of each wins.
pub fn parse_split(text: &str) -> ParsedSplit {
    let mut p = ParsedSplit::default();
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
            "simboluri" => &mut p.tokens,
            "median placa (us)" => &mut p.board_us,
            "median driver (us)" => &mut p.driver_us,
            "median jurnal (us)" => &mut p.journal_us,
            "median ecran (us)" => &mut p.screen_us,
            "median verificare procesor (us)" => &mut p.check_us,
            "median simbol fara verificare (us)" => &mut p.host_us,
            "median simbol cu verificare (us)" => &mut p.total_us,
            "simboluri pe secunda x100 (fara verificare)" => &mut p.per_s_x100,
            "simboluri pe secunda x100 (cu verificare)" => &mut p.per_s_check_x100,
            _ => continue,
        };
        *slot = Some(v);
    }
    p
}

/// What each speed figure of `os-307` includes (C-145 point 1), for the report and the docs.
pub const FIGURES: [(&str, &str); 3] = [
    (
        "43.93 simboluri/s doar lansarile",
        "ceasul placii: suma duratelor lansarilor unui simbol (median 22,9 ms); fara driver, jurnal, ecran sau procesor",
    ),
    (
        "3.10 simboluri/s cu driverul, jurnalul si ecranul",
        "ceasul OS-ului de la prima lansare scrisa pana la logit-urile citite (median 294,8 ms); FARA verificarea pe procesor",
    ),
    (
        "0.24 simboluri/s (ceasul placii, cu pauzele driverului)",
        "ceasul placii de la prima inregistrare a primei pozitii la ultima a ultimei, impartit la pozitii: CU verificarea pe procesor (~3,87 s pe pozitie, motorul de referinta pe 1 nucleu) intre pozitii",
    ),
];

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::diag::Flush;

    #[test]
    fn routine_is_the_window_the_invalidate_and_the_doorbell() {
        assert_eq!(GEN_WRITES, [0x1700, 0xb830a0, 0xb830a4, 0xb830b0, 0xbb0090]);
        const NOTES: [&str; 1] = ["permis"];
        let r = gen_routine(&NOTES);
        let w = |off| Event::MmioWrite {
            bar: 0,
            offset: off,
            before: 0,
            written: 0,
            after: 0,
        };
        for off in GEN_WRITES {
            assert!(r.covers(&w(off)));
        }
        // Every other register write, every step and every value stays immediate.
        for off in [0x110c00, 0x1103c0, 0x8241e0, 0x1704] {
            assert!(!r.covers(&w(off)));
        }
        assert!(!r.covers(&Event::Step {
            name: "i14-simbol",
            ok: true
        }));
        assert!(!r.covers(&Event::Value {
            what: "kg simboluri cerute",
            value: 16
        }));
        assert!(!r.covers(&Event::ConfigWrite {
            offset: 4,
            before: 3,
            written: 7,
            after: 7
        }));
        assert!(r.covers(&Event::Note("permis")));
        assert!(!r.covers(&Event::Note("alta nota")));
        let _ = Flush::Now;
    }

    #[test]
    fn shadow_gives_only_the_changed_words() {
        let mut s = WriteShadow::new(0x1000, 64).unwrap();
        let a: Vec<u8> = (0..32u8).collect();
        // Nothing known: one run, all of it.
        assert_eq!(s.changes(0x1000, &a).unwrap(), [(0x1000, 0..32)]);
        s.remember(0x1000, &a).unwrap();
        assert!(s.changes(0x1000, &a).unwrap().is_empty());
        let mut b = a.clone();
        b[5] ^= 1; // word 1
        b[8] ^= 1; // word 2 (adjacent: merged)
        b[28] ^= 1; // word 7
        assert_eq!(
            s.changes(0x1000, &b).unwrap(),
            [(0x1004, 4..12), (0x101c, 28..32)]
        );
        // A part never written is unknown.
        assert_eq!(s.changes(0x1020, &[0; 8]).unwrap(), [(0x1020, 0..8)]);
        s.forget();
        assert_eq!(s.known_words(), 0);
        assert_eq!(s.changes(0x1000, &a).unwrap(), [(0x1000, 0..32)]);
        // Outside, unaligned: refused.
        assert!(s.changes(0x1040, &[0; 4]).is_err());
        assert!(s.changes(0x1002, &[0; 4]).is_err());
        assert!(s.changes(0x0ffc, &[0; 4]).is_err());
        assert!(WriteShadow::new(0, 6).is_err());
        assert!(WriteShadow::new(0, WriteShadow::MAX_LEN + 4).is_err());
    }

    #[test]
    fn split_of_os_307_and_its_rows() {
        // os-307's figures: 294.8 ms per token, 22.9 ms on the board, ~15 rows × 16.4 ms in the journal, ~3.87 s check.
        let mut t = TimeSplit::new();
        for pos in 0..18 {
            t.add(TokenSplit {
                pos,
                host_us: 294_806,
                board_us: 22_871,
                journal_us: 246_000,
                screen_us: 0,
                check_us: 3_870_293,
            })
            .unwrap();
        }
        let s = t.summary();
        assert_eq!(s.driver_us, 294_806 - 22_871 - 246_000);
        assert_eq!(s.per_s_x100, 339); // 3.39 without the check (the medians: one token's time)
        assert_eq!(s.per_s_check_x100, 24); // 0.24 with it: the "0.24" of os-307
        let mut j = Journal::new(64);
        t.journal(&mut j);
        let text: String = j
            .events()
            .map(|e| format!("[  209.000000] gpu: {}\n", e))
            .collect();
        let p = parse_split(&text);
        assert_eq!(p.tokens, Some(18));
        assert_eq!(p.journal_us, Some(246_000));
        assert_eq!(p.check_us, Some(3_870_293));
        assert_eq!(p.per_s_check_x100, Some(24));
        assert_eq!(p.driver_us, Some(25_935));
        assert!(t.lines("placa")[0].contains("jurnal 246000 us"));
        // A part larger than the whole is refused.
        assert!(
            t.add(TokenSplit {
                pos: 18,
                host_us: 10,
                board_us: 11,
                ..Default::default()
            })
            .is_err()
        );
    }
}
