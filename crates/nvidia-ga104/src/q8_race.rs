//! C-147 step 3 (stage 4) — the **race** between the Q8_0 products, to be run
//! on the board by the OS: for each of Kronos #513's matrix shapes ("lanes"),
//! each program ("entrant") is launched `n` times with a timestamp after every
//! launch (the method of `launch_cost`, C-69: the board's own clock, no CPU in
//! the loop), then its y is judged. The rows `kr …` go into the stick journal;
//! [`parse_journal`] reads them back; [`Standing`] says which program wins
//! each lane.
//!
//! The rule, written before any run: a lane changes program only if the new
//! one **passes** its judge on the board and its median is lower by at least
//! [`MIN_GAIN_PERMILLE`] ‰; otherwise the program of today stays. A tie (as
//! in SIMULARE, where every launch costs the same) keeps `q8_matvec`.
//!
//! Heats run in the order A B B A per lane (drift of the clocks or the
//! temperature during the race weighs on both alike).
//!
//! Not entered: the f32 matrix-matrix programs of C-20 E (`matmul_naive` …
//! `matmul_vec4`). For one token they would need the weights dequantized to
//! f32 (4× the bytes read, ~443 MB per token instead of ~111) and a 1-column
//! right-hand side; they belong to stage 6 (several answers at once), where
//! the right-hand side has many columns.
//!
//! Tested **only in SIMULARE**: the simulated clock gives every launch the
//! same time, so only the method is checked, never a winner.

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use crate::launch_cost::{self, LaunchCost};
use crate::push::PushBuf;
use crate::q8_matvec::{self as mv, Verdict};
use crate::q8_matvec32 as w32;

/// A lane wins a new program only when it is at least this much faster (‰ of the old median).
pub const MIN_GAIN_PERMILLE: u64 = 50;
/// Launches per heat (each lane: 4 heats, so 2 × this per entrant).
pub const RACE_LAUNCHES: u32 = 16;

/// The programs in the race.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entrant {
    /// C-83 I12: one block per thread (today's program).
    Q8Matvec,
    /// C-147 step 3: a quarter block per thread, 128-bit x loads.
    Q8Matvec32,
}

/// Both entrants, today's first.
pub const ENTRANTS: [Entrant; 2] = [Entrant::Q8Matvec, Entrant::Q8Matvec32];

impl Entrant {
    pub fn code(self) -> &'static [u8] {
        match self {
            Entrant::Q8Matvec => mv::Q8_MATVEC,
            Entrant::Q8Matvec32 => w32::Q8_MATVEC32,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Entrant::Q8Matvec => "q8_matvec",
            Entrant::Q8Matvec32 => "q8_matvec32",
        }
    }

    /// The constant buffer (same layout for both; `q8_matvec32` also checks x is 16-aligned).
    pub fn cbuf(
        self,
        w: u64,
        x: u64,
        y: u64,
        rows: usize,
        cols: usize,
    ) -> GpuResult<[u8; mv::CBUF_SIZE as usize]> {
        match self {
            Entrant::Q8Matvec => mv::q8_cbuf(w, x, y, rows, cols),
            Entrant::Q8Matvec32 => w32::cbuf(w, x, y, rows, cols),
        }
    }

    pub fn qmd(
        self,
        program_va: u64,
        cbuf_va: u64,
        rows: usize,
    ) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
        match self {
            Entrant::Q8Matvec => mv::q8_qmd(program_va, cbuf_va, rows),
            Entrant::Q8Matvec32 => w32::qmd(program_va, cbuf_va, rows),
        }
    }

    /// The program's own judge (bit for bit with its oracle, else the margin).
    pub fn judge(
        self,
        w: &[u8],
        rows: usize,
        cols: usize,
        x: &[f32],
        got: &[f32],
    ) -> GpuResult<Verdict> {
        match self {
            Entrant::Q8Matvec => mv::judge(w, rows, cols, x, got),
            Entrant::Q8Matvec32 => w32::judge(w, rows, cols, x, got),
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// One matrix shape of the race and how many times a token uses it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lane {
    pub name: &'static str,
    pub rows: usize,
    pub cols: usize,
    pub per_token: u32,
}

/// Kronos #513 (E 768, 12 layers, vocab 16 384): QKV, out, fc, proj per layer, the logits once.
pub const K513_LANES: [Lane; 5] = [
    Lane {
        name: "qkv",
        rows: 2304,
        cols: 768,
        per_token: 12,
    },
    Lane {
        name: "out",
        rows: 768,
        cols: 768,
        per_token: 12,
    },
    Lane {
        name: "fc",
        rows: 3072,
        cols: 768,
        per_token: 12,
    },
    Lane {
        name: "proj",
        rows: 768,
        cols: 3072,
        per_token: 12,
    },
    Lane {
        name: "logits",
        rows: 16_384,
        cols: 768,
        per_token: 1,
    },
];

/// One heat: `lane` (index into the lanes given), `entrant`, `n` launches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Heat {
    pub lane: usize,
    pub entrant: Entrant,
    pub n: u32,
}

/// The heats, in order: per lane A B B A.
pub fn heats(lanes: usize, n: u32) -> Vec<Heat> {
    let (a, b) = (Entrant::Q8Matvec, Entrant::Q8Matvec32);
    (0..lanes)
        .flat_map(|lane| [a, b, b, a].map(|entrant| Heat { lane, entrant, n }))
        .collect()
}

/// The pushbuffer of one heat: `launch_cost::encode` (start marker, then `n` × launch + timestamp record).
pub fn encode_heat(
    p: &mut PushBuf<'_>,
    qmd_va: u64,
    records_va: u64,
    heat: &Heat,
    first_payload: u32,
) -> GpuResult<()> {
    launch_cost::encode(p, qmd_va, records_va, heat.n, first_payload)
}

/// What one heat gave.
#[derive(Clone, Debug, PartialEq)]
pub struct HeatResult {
    pub heat: Heat,
    pub cost: LaunchCost,
    /// y after the heat, judged by the entrant's own judge.
    pub passes: bool,
}

/// The race so far, lane by lane.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Standing {
    /// Per lane, per entrant: every launch time (ns) and whether every heat passed.
    pub times: Vec<[(Vec<u64>, bool); 2]>,
}

fn median(v: &[u64]) -> Option<u64> {
    let mut s = v.to_vec();
    s.sort_unstable();
    s.get(s.len() / 2).copied()
}

/// Who wins a lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Winner {
    /// Today's program stays (`why`: tie / not faster enough / the new one failed / no data).
    Stays(&'static str),
    /// The new program is faster by `gain_permille` ‰ and passed.
    New { gain_permille: u64 },
}

impl Standing {
    pub fn add(&mut self, lanes: usize, r: &HeatResult) {
        if self.times.len() < lanes {
            self.times
                .resize_with(lanes, || [(Vec::new(), true), (Vec::new(), true)]);
        }
        let e = &mut self.times[r.heat.lane][r.heat.entrant.index()];
        e.0.extend_from_slice(&r.cost.each_ns);
        e.1 &= r.passes;
    }

    /// The median launch (ns) of `entrant` on `lane`.
    pub fn median(&self, lane: usize, entrant: Entrant) -> Option<u64> {
        median(&self.times.get(lane)?[entrant.index()].0)
    }

    /// The rule of the module doc.
    pub fn winner(&self, lane: usize) -> Winner {
        let Some(t) = self.times.get(lane) else {
            return Winner::Stays("fara date");
        };
        let (Some(old), Some(new)) = (median(&t[0].0), median(&t[1].0)) else {
            return Winner::Stays("fara date");
        };
        if !t[1].1 {
            return Winner::Stays("programul nou nu trece proba");
        }
        if !t[0].1 {
            return Winner::Stays("programul de azi nu trece proba (de analizat)");
        }
        if new >= old {
            return Winner::Stays("nu e mai rapid");
        }
        let gain = (old - new) * 1000 / old.max(1);
        if gain < MIN_GAIN_PERMILLE {
            return Winner::Stays("castig sub prag");
        }
        Winner::New {
            gain_permille: gain,
        }
    }

    /// Estimated time of the products of one token (µs) with today's program and with the winners
    /// (Σ median × uses per token). A paper sum from the race, not a token measured.
    pub fn token_us(&self, lanes: &[Lane]) -> Option<(u64, u64)> {
        let mut today = 0u64;
        let mut best = 0u64;
        for (i, l) in lanes.iter().enumerate() {
            let old = self.median(i, Entrant::Q8Matvec)?;
            let pick = match self.winner(i) {
                Winner::New { .. } => self.median(i, Entrant::Q8Matvec32)?,
                Winner::Stays(_) => old,
            };
            today += old * u64::from(l.per_token);
            best += pick * u64::from(l.per_token);
        }
        Some((today / 1000, best / 1000))
    }
}

/// The journal rows: per lane and entrant `kr <lane> <program> mediana (ns)`, `… minim (ns)`, `… trece`; per
/// lane `kr <lane> castiga` (0 = q8_matvec, 1 = q8_matvec32) and `kr <lane> castig (promile)`; then
/// `kr simbol azi (us)` / `kr simbol cu castigatorii (us)`. The names come from [`ROW_NAMES`] (5 lanes max).
pub fn rows(s: &Standing, lanes: &[Lane]) -> GpuResult<Vec<(&'static str, u32)>> {
    if lanes.len() > ROW_NAMES.len() {
        return Err(GpuError::InvalidArgument("cursa: prea multe benzi"));
    }
    let clip = |v: u64| v.min(u32::MAX as u64 - 1) as u32;
    let mut out = Vec::new();
    out.push((RACE_LANES, lanes.len() as u32));
    for (i, n) in ROW_NAMES.iter().enumerate().take(lanes.len()) {
        for e in ENTRANTS {
            let k = e.index();
            let t = s.times.get(i).map(|t| &t[k]);
            let min = t.and_then(|t| t.0.iter().min().copied());
            out.push((n.median[k], s.median(i, e).map_or(u32::MAX, clip)));
            out.push((n.min[k], min.map_or(u32::MAX, clip)));
            out.push((n.passes[k], t.map_or(u32::MAX, |t| t.1 as u32)));
        }
        let (who, gain) = match s.winner(i) {
            Winner::New { gain_permille } => (1, clip(gain_permille)),
            Winner::Stays(_) => (0, 0),
        };
        out.push((n.winner, who));
        out.push((n.gain, gain));
    }
    if let Some((today, best)) = s.token_us(lanes) {
        out.push((TOKEN_TODAY, clip(today)));
        out.push((TOKEN_BEST, clip(best)));
    }
    Ok(out)
}

/// Writes [`rows`] into the journal.
pub fn journal(s: &Standing, lanes: &[Lane], j: &mut Journal) -> GpuResult<()> {
    for (what, value) in rows(s, lanes)? {
        j.push(Event::Value { what, value });
    }
    Ok(())
}

/// The names of one lane's rows.
pub struct LaneRows {
    pub median: [&'static str; 2],
    pub min: [&'static str; 2],
    pub passes: [&'static str; 2],
    pub winner: &'static str,
    pub gain: &'static str,
}

macro_rules! lane_rows {
    ($l:literal) => {
        LaneRows {
            median: [
                concat!("kr ", $l, " q8_matvec mediana (ns)"),
                concat!("kr ", $l, " q8_matvec32 mediana (ns)"),
            ],
            min: [
                concat!("kr ", $l, " q8_matvec minim (ns)"),
                concat!("kr ", $l, " q8_matvec32 minim (ns)"),
            ],
            passes: [
                concat!("kr ", $l, " q8_matvec trece"),
                concat!("kr ", $l, " q8_matvec32 trece"),
            ],
            winner: concat!("kr ", $l, " castiga"),
            gain: concat!("kr ", $l, " castig (promile)"),
        }
    };
}

/// Row names, in the order of [`K513_LANES`].
pub const ROW_NAMES: [LaneRows; 5] = [
    lane_rows!("qkv"),
    lane_rows!("out"),
    lane_rows!("fc"),
    lane_rows!("proj"),
    lane_rows!("logits"),
];
pub const RACE_LANES: &str = "kr benzi";
pub const TOKEN_TODAY: &str = "kr simbol azi (us)";
pub const TOKEN_BEST: &str = "kr simbol cu castigatorii (us)";

/// The race as read back from a stick journal (the last `kr benzi` row starts it again).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedRace {
    pub lanes: Option<u32>,
    /// Per lane: medians, minimums, passes (per entrant), winner, gain.
    pub lane: [ParsedLane; 5],
    pub token_today_us: Option<u32>,
    pub token_best_us: Option<u32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParsedLane {
    pub median: [Option<u32>; 2],
    pub min: [Option<u32>; 2],
    pub passes: [Option<bool>; 2],
    pub winner: Option<u32>,
    pub gain: Option<u32>,
}

/// Reads the `kr` rows of a stick journal (`… valoare <name> = 0x<hex>`).
pub fn parse_journal(text: &str) -> ParsedRace {
    let mut out = ParsedRace::default();
    for (name, v) in text.lines().filter_map(crate::kronos_perf::value_row) {
        let some = (v != u32::MAX).then_some(v);
        if name == RACE_LANES {
            out = ParsedRace {
                lanes: some,
                ..ParsedRace::default()
            };
            continue;
        }
        match name {
            TOKEN_TODAY => out.token_today_us = some,
            TOKEN_BEST => out.token_best_us = some,
            _ => {}
        }
        for (i, n) in ROW_NAMES.iter().enumerate() {
            let l = &mut out.lane[i];
            for k in 0..2 {
                if name == n.median[k] {
                    l.median[k] = some;
                } else if name == n.min[k] {
                    l.min[k] = some;
                } else if name == n.passes[k] {
                    l.passes[k] = some.map(|v| v == 1);
                }
            }
            if name == n.winner {
                l.winner = some;
            } else if name == n.gain {
                l.gain = some;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn result(lane: usize, entrant: Entrant, each: &[u64], passes: bool) -> HeatResult {
        HeatResult {
            heat: Heat {
                lane,
                entrant,
                n: each.len() as u32,
            },
            cost: LaunchCost {
                launches: each.len() as u32,
                each_ns: each.to_vec(),
                min_ns: *each.iter().min().unwrap(),
                median_ns: 0,
                max_ns: 0,
                total_ns: each.iter().sum(),
            },
            passes,
        }
    }

    #[test]
    fn heats_are_abba() {
        let h = heats(2, 8);
        assert_eq!(h.len(), 8);
        let e: Vec<Entrant> = h.iter().take(4).map(|h| h.entrant).collect();
        assert_eq!(
            e,
            [
                Entrant::Q8Matvec,
                Entrant::Q8Matvec32,
                Entrant::Q8Matvec32,
                Entrant::Q8Matvec
            ]
        );
        assert!(h[4..].iter().all(|h| h.lane == 1 && h.n == 8));
    }

    #[test]
    fn the_rule() {
        let mut s = Standing::default();
        // lane 0: new 20 % faster, passes → New(200).
        s.add(3, &result(0, Entrant::Q8Matvec, &[100, 100, 101], true));
        s.add(3, &result(0, Entrant::Q8Matvec32, &[80, 80, 81], true));
        // lane 1: new 3 % faster → stays (under 50 ‰).
        s.add(3, &result(1, Entrant::Q8Matvec, &[100, 100], true));
        s.add(3, &result(1, Entrant::Q8Matvec32, &[97, 97], true));
        // lane 2: new much faster but one heat failed its judge → stays.
        s.add(3, &result(2, Entrant::Q8Matvec, &[100], true));
        s.add(3, &result(2, Entrant::Q8Matvec32, &[10], true));
        s.add(3, &result(2, Entrant::Q8Matvec32, &[10], false));
        assert_eq!(s.winner(0), Winner::New { gain_permille: 200 });
        assert_eq!(s.winner(1), Winner::Stays("castig sub prag"));
        assert_eq!(s.winner(2), Winner::Stays("programul nou nu trece proba"));
        assert_eq!(s.winner(7), Winner::Stays("fara date"));
        // A tie keeps today's program.
        let mut t = Standing::default();
        t.add(1, &result(0, Entrant::Q8Matvec, &[5, 5], true));
        t.add(1, &result(0, Entrant::Q8Matvec32, &[5, 5], true));
        assert_eq!(t.winner(0), Winner::Stays("nu e mai rapid"));
    }

    #[test]
    fn rows_round_trip_and_token_sum() {
        let lanes = &K513_LANES[..2];
        let mut s = Standing::default();
        s.add(2, &result(0, Entrant::Q8Matvec, &[100_000; 3], true));
        s.add(2, &result(0, Entrant::Q8Matvec32, &[60_000; 3], true));
        s.add(2, &result(1, Entrant::Q8Matvec, &[50_000; 3], true));
        s.add(2, &result(1, Entrant::Q8Matvec32, &[49_000; 3], true));
        // today: 12 × 100 + 12 × 50 = 1800 µs; with the winners: 12 × 60 + 12 × 50 = 1320 µs.
        assert_eq!(s.token_us(lanes), Some((1800, 1320)));
        let mut j = Journal::new(256);
        journal(&s, lanes, &mut j).unwrap();
        assert_eq!(j.events().count(), rows(&s, lanes).unwrap().len());
        // As the OS writes value rows on the stick.
        let text: alloc::string::String = rows(&s, lanes)
            .unwrap()
            .iter()
            .map(|(n, v)| alloc::format!("[1.0] gpu: valoare {n} = {v:#010x}\n"))
            .collect();
        let p = parse_journal(&text);
        assert_eq!(p.lanes, Some(2));
        assert_eq!(p.lane[0].median, [Some(100_000), Some(60_000)]);
        assert_eq!(p.lane[0].passes, [Some(true), Some(true)]);
        assert_eq!((p.lane[0].winner, p.lane[0].gain), (Some(1), Some(400)));
        assert_eq!((p.lane[1].winner, p.lane[1].gain), (Some(0), Some(0)));
        assert_eq!(p.lane[2], ParsedLane::default());
        assert_eq!(
            (p.token_today_us, p.token_best_us),
            (Some(1800), Some(1320))
        );
        // A missing entrant is written as "no value" and read back as None.
        let mut half = Standing::default();
        half.add(1, &result(0, Entrant::Q8Matvec, &[7], true));
        let r = rows(&half, &K513_LANES[..1]).unwrap();
        assert!(r.contains(&(ROW_NAMES[0].median[1], u32::MAX)));
        assert!(rows(&half, &[K513_LANES[0]; 6]).is_err());
    }

    #[test]
    fn lanes_are_513() {
        let l = &K513_LANES;
        assert_eq!(l.iter().map(|l| l.per_token).sum::<u32>(), 49);
        assert_eq!(l[4].rows, crate::kronos_speed::K513_VOCAB);
        assert!(l.iter().all(|l| l.cols == 768 || l.cols == 3072));
        for (l, n) in l.iter().zip(&ROW_NAMES) {
            assert!(n.median[0].contains(l.name));
        }
    }
}
