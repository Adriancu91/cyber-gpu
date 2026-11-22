//! C-160 point 6 — **the first training "backward" program on the real board**: everything the OS needs to send
//! [`crate::colsum`] (`colsum.comp`, Mesa NAK 25.0.7: the bias gradient Σ dy and the LayerNorm dγ = Σ dy·x̂,
//! dβ = Σ dy) a few times, outside the generation path, and to judge what comes back.
//!
//! - [`SETS`]: the fixed probe inputs (dy, x and the forward statistics μ, rstd), made here from fixed seeds, so the
//!   OS and the tests use the same bytes ([`ProbeSet::buffer`], [`ProbeSet::inputs_fnv`]).
//! - [`ProbeSet::buffer`]: one [`BUF_BYTES`] buffer of host RAM per set (the OS may reuse one): dy, x, μ, rstd, then
//!   the two outputs s0, s1, each followed by [`GUARD`] words; outputs and guards are filled with NaN ([`FILL`]),
//!   so an element the board did not write cannot pass and a write past the columns is seen.
//! - [`launch_bytes`]: the constant buffer (0x100 bytes) and the QMD for one set.
//! - [`judge_set`] → [`SetResult`]: the margin written in T155 before any measurement
//!   (`gpu_compute::train::colsum_margin`, against the f64 oracle `train::layernorm_back_rows`), how far the worst
//!   element is (‰ of its margin), the guards, and how many elements are **bit for bit** the CPU model
//!   `gpu_compute::layer::colsum_gpu` — the program has no MUFU, no shared memory, no atomics, so the model follows
//!   the code NAK made operation by operation: the prediction is "all of them".
//! - [`Probe`]: the run (all sets), its `kt colsum …` journal rows ([`Probe::rows`]), report lines and
//!   [`parse_colsum`] to read the rows back from a stick journal.
//!
//! Tested **only in SIMULARE** (T166, `tests/sim_c160_p6_colsum.rs`); nothing here has run on a board.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::layer::{self, LnStats};
use gpu_compute::train;
use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use crate::colsum::{self as cs, Mode};
use crate::qmd::ComputeQmd;

/// The host buffer of one set (64 KiB, 16 pages; the OS maps it like the copy program's host RAM).
pub const BUF_BYTES: usize = 64 << 10;
/// Words after each output, filled with [`FILL`], that must stay untouched (the last block's threads past `c`).
pub const GUARD: usize = 256;
/// The word outputs and guards are filled with before a launch (a quiet NaN).
pub const FILL: u32 = 0x7fc0_0000;
/// Bytes of constant buffer 0 sent (the program reads 0x64; the QMD's size field is 0x100, as in T155).
pub const CBUF_SIZE: usize = 0x100;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// One probe set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeSet {
    pub name: &'static str,
    pub mode: Mode,
    pub rows: u32,
    pub c: u32,
    pub seed: u64,
    /// Added to every x (a large mean makes the recomputed x̂ = (x − μ)·rstd sensitive to μ's rounding).
    pub offset: f32,
}

/// The probe: 5 launches of the same program, each with its own buffer contents.
pub const SETS: [ProbeSet; 5] = [
    ProbeSet {
        name: "deplasare-64x64",
        mode: Mode::Bias,
        rows: 64,
        c: 64,
        seed: 1,
        offset: 0.0,
    },
    ProbeSet {
        name: "ln-64x64",
        mode: Mode::LayerNorm,
        rows: 64,
        c: 64,
        seed: 2,
        offset: 0.0,
    },
    ProbeSet {
        name: "ln-3x257",
        mode: Mode::LayerNorm,
        rows: 3,
        c: 257,
        seed: 3,
        offset: 0.0,
    },
    ProbeSet {
        name: "ln-8x768",
        mode: Mode::LayerNorm,
        rows: 8,
        c: 768,
        seed: 4,
        offset: 0.0,
    },
    ProbeSet {
        name: "ln-8x768-media100",
        mode: Mode::LayerNorm,
        rows: 8,
        c: 768,
        seed: 5,
        offset: 100.0,
    },
];

/// Byte offsets inside the buffer of one set (every region aligned to 256 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub dy: usize,
    pub x: usize,
    pub mean: usize,
    pub rstd: usize,
    /// s0: `c` outputs + [`GUARD`] words.
    pub s0: usize,
    /// s1: `c` outputs + [`GUARD`] words (untouched in [`Mode::Bias`]).
    pub s1: usize,
    /// End of s1's guard: the board writes nothing in `[end, BUF_BYTES)`.
    pub end: usize,
}

fn up256(v: usize) -> usize {
    v.div_ceil(256) * 256
}

/// The inputs of one set.
#[derive(Clone, Debug, PartialEq)]
pub struct Inputs {
    pub dy: Vec<f32>,
    pub x: Vec<f32>,
    pub mean: Vec<f32>,
    pub rstd: Vec<f32>,
}

impl ProbeSet {
    pub fn layout(&self) -> Layout {
        let (r, c) = (self.rows as usize, self.c as usize);
        let dy = 0;
        let x = up256(dy + 4 * r * c);
        let mean = up256(x + 4 * r * c);
        let rstd = up256(mean + 4 * r);
        let s0 = up256(rstd + 4 * r);
        let s1 = up256(s0 + 4 * (c + GUARD));
        let end = s1 + 4 * (c + GUARD);
        Layout {
            dy,
            x,
            mean,
            rstd,
            s0,
            s1,
            end,
        }
    }

    /// dy in [-0.01, 0.01), x = offset + [-1.5, 1.5) (xorshift, integer steps only), μ and rstd as the forward
    /// pass keeps them (`train::layernorm_stats`, also written for the bias set, where the program ignores them).
    pub fn inputs(&self) -> Inputs {
        let (r, c) = (self.rows as usize, self.c as usize);
        let mut s = self.seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let mut x = Vec::with_capacity(r * c);
        let mut dy = Vec::with_capacity(r * c);
        for _ in 0..r * c {
            x.push(self.offset + next() * 1.5);
            dy.push(next() * 0.01);
        }
        let (mean, rstd) = train::layernorm_stats(&x, r, c).expect("probe shapes are valid");
        Inputs { dy, x, mean, rstd }
    }

    /// The host buffer for this set (module doc).
    pub fn buffer(&self) -> Vec<u8> {
        let l = self.layout();
        let i = self.inputs();
        let mut b = alloc::vec![0u8; BUF_BYTES];
        let mut put = |at: usize, v: &[f32]| {
            for (k, f) in v.iter().enumerate() {
                b[at + 4 * k..at + 4 * k + 4].copy_from_slice(&f.to_bits().to_le_bytes());
            }
        };
        put(l.dy, &i.dy);
        put(l.x, &i.x);
        put(l.mean, &i.mean);
        put(l.rstd, &i.rstd);
        for k in 0..2 * (self.c as usize + GUARD) {
            let at = if k < self.c as usize + GUARD {
                l.s0 + 4 * k
            } else {
                l.s1 + 4 * (k - self.c as usize - GUARD)
            };
            b[at..at + 4].copy_from_slice(&FILL.to_le_bytes());
        }
        b
    }

    /// FNV-1a (32 bit) of the input bytes of [`Self::buffer`] (everything before s0).
    pub fn inputs_fnv(&self) -> u32 {
        fnv(&self.buffer()[..self.layout().s0])
    }

    /// The program's arguments for a buffer at `buf_va`.
    pub fn args(&self, buf_va: u64) -> cs::Args {
        let l = self.layout();
        cs::Args {
            dy: buf_va + l.dy as u64,
            x: buf_va + l.x as u64,
            mean: buf_va + l.mean as u64,
            rstd: buf_va + l.rstd as u64,
            s0: buf_va + l.s0 as u64,
            s1: buf_va + l.s1 as u64,
            rows: self.rows,
            c: self.c,
            mode: self.mode,
        }
    }

    /// The bytes the OS reads back after the launch: `[s0, end)` of [`Self::layout`].
    pub fn out_range(&self) -> (usize, usize) {
        let l = self.layout();
        (l.s0, l.end - l.s0)
    }

    /// What the CPU model `layer::colsum_gpu` says the board writes, as the bytes of [`Self::out_range`].
    pub fn model_out(&self) -> Vec<u8> {
        let i = self.inputs();
        let (r, c) = (self.rows as usize, self.c as usize);
        let ln = (self.mode == Mode::LayerNorm).then_some(LnStats {
            x: &i.x,
            mean: &i.mean,
            rstd: &i.rstd,
        });
        let (s0, s1) = layer::colsum_gpu(&i.dy, r, c, ln);
        let mut b = self.buffer();
        let l = self.layout();
        for (k, v) in s0.iter().enumerate() {
            b[l.s0 + 4 * k..l.s0 + 4 * k + 4].copy_from_slice(&v.to_bits().to_le_bytes());
        }
        for (k, v) in s1.iter().enumerate() {
            b[l.s1 + 4 * k..l.s1 + 4 * k + 4].copy_from_slice(&v.to_bits().to_le_bytes());
        }
        b[l.s0..l.end].to_vec()
    }
}

/// FNV-1a, 32 bit.
pub fn fnv(bytes: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in bytes {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// The constant buffer (0x100 bytes: the program's 0x64, then zero) and the QMD (⌈c/256⌉ blocks of 256 threads,
/// 19 registers, no shared memory) for set `set` whose buffer is at `buf_va` (GPU VA of the host buffer, aligned to
/// 4 KiB). `prog_va`: where [`cs::COLSUM`] was loaded (aligned to 256); `cbuf_va`: where the cbuf bytes go
/// (aligned to 256).
pub fn launch_bytes(
    set: usize,
    prog_va: u64,
    cbuf_va: u64,
    buf_va: u64,
) -> GpuResult<([u8; CBUF_SIZE], [u8; nvidia_rm570::QMD_SIZE])> {
    let s = SETS.get(set).ok_or(bad("colsum proba: set necunoscut"))?;
    if !buf_va.is_multiple_of(4096) || !cbuf_va.is_multiple_of(256) {
        return Err(bad(
            "colsum proba: bufferul nealiniat la 4 KiB sau cbuf nealiniat la 256",
        ));
    }
    let a = s.args(buf_va);
    let mut cb = [0u8; CBUF_SIZE];
    cb[..cs::CBUF_BYTES].copy_from_slice(&a.cbuf().ok_or(bad("colsum proba: argumente"))?);
    let q = ComputeQmd {
        program_va: prog_va,
        registers: cs::REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: CBUF_SIZE as u32,
        grid: a.grid(),
        block: cs::BLOCK,
        release: None,
    }
    .to_bytes()?;
    Ok((cb, q))
}

/// The verdict of one set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every element within its written margin, guards untouched.
    InMargine,
    /// Some element outside its margin (written, finite).
    NuTrece,
    /// Wrong length, an output not written / not finite, a guard word changed, or s1 written in the bias mode.
    Invalid,
}

/// What one set gave.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetResult {
    /// Index in [`SETS`].
    pub set: usize,
    pub verdict: Verdict,
    /// Worst element, in ‰ of its margin (`None` when invalid).
    pub worst_permille: Option<u32>,
    /// Elements bit for bit equal to the CPU model `layer::colsum_gpu`.
    pub exact: u32,
    /// Elements judged: `c` (bias) or `2c` (dγ and dβ).
    pub elements: u32,
    /// All guard words still [`FILL`] (and s1 untouched in the bias mode).
    pub guards_ok: bool,
    /// FNV-1a of the bytes read back.
    pub out_fnv: u32,
    /// The launch time on the board clock, if the OS measured it.
    pub ns: Option<u64>,
}

impl SetResult {
    pub fn passes(&self) -> bool {
        self.verdict == Verdict::InMargine
    }
    /// 1 = within the margin, 0 = outside, 2 = invalid.
    pub fn code(&self) -> u32 {
        match self.verdict {
            Verdict::InMargine => 1,
            Verdict::NuTrece => 0,
            Verdict::Invalid => 2,
        }
    }
}

fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Judges the bytes the board wrote for set `set` (the [`ProbeSet::out_range`] of its buffer).
pub fn judge_set(set: usize, out: &[u8], ns: Option<u64>) -> GpuResult<SetResult> {
    let s = SETS.get(set).ok_or(bad("colsum proba: set necunoscut"))?;
    let (r, c) = (s.rows as usize, s.c as usize);
    let ln = s.mode == Mode::LayerNorm;
    let elements = if ln { 2 * c } else { c } as u32;
    let mut res = SetResult {
        set,
        verdict: Verdict::Invalid,
        worst_permille: None,
        exact: 0,
        elements,
        guards_ok: false,
        out_fnv: fnv(out),
        ns,
    };
    let (_, len) = s.out_range();
    if out.len() != len {
        return Ok(res);
    }
    let l = s.layout();
    let got = words(out);
    let s0 = &got[..c + GUARD];
    let s1 = &got[(l.s1 - l.s0) / 4..];
    let mut guards_ok = s0[c..].iter().all(|&w| w == FILL) && s1[c..].iter().all(|&w| w == FILL);
    if !ln {
        guards_ok &= s1.iter().all(|&w| w == FILL);
    }
    // The gap between s0's guard and s1 (alignment) is not written either.
    let gap = &got[c + GUARD..(l.s1 - l.s0) / 4];
    let gap_ok = gap.iter().all(|&w| w == 0);
    res.guards_ok = guards_ok && gap_ok;

    let want = words(&s.model_out());
    res.exact = (0..c).filter(|&k| s0[k] == want[k]).count() as u32
        + if ln {
            let o = (l.s1 - l.s0) / 4;
            (0..c).filter(|&k| s1[k] == want[o + k]).count() as u32
        } else {
            0
        };

    let finite = |v: &[u32]| v[..c].iter().all(|&w| f32::from_bits(w).is_finite());
    if !res.guards_ok || !finite(s0) || (ln && !finite(s1)) {
        return Ok(res);
    }
    let i = s.inputs();
    let ones = alloc::vec![1.0f32; c];
    let (_, or_g, or_b) = train::layernorm_back_rows(&i.x, &i.dy, &ones, r, c)?;
    let (bg, bb) = train::colsum_margin(&i.x, &i.dy, r, c)?;
    let mut worst = 0.0f64;
    let mut inside = true;
    let mut check = |got: &[u32], want: &[f32], m: &[f64]| {
        for k in 0..c {
            let d = (f32::from_bits(got[k]) as f64 - want[k] as f64).abs();
            inside &= d <= m[k];
            worst = worst.max(d / m[k]);
        }
    };
    if ln {
        check(s0, &or_g, &bg);
        check(s1, &or_b, &bb);
    } else {
        check(s0, &or_b, &bb);
    }
    let pm = worst * 1000.0;
    res.worst_permille = Some(if pm >= u32::MAX as f64 {
        u32::MAX
    } else {
        let t = pm as u32;
        if (t as f64) < pm { t + 1 } else { t }
    });
    res.verdict = if inside {
        Verdict::InMargine
    } else {
        Verdict::NuTrece
    };
    Ok(res)
}

/// The whole probe (one [`SetResult`] per set, in order).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Probe {
    pub sets: Vec<SetResult>,
}

impl Probe {
    pub fn add(&mut self, r: SetResult) {
        self.sets.push(r);
    }
    /// All [`SETS`] judged, all within the margin.
    pub fn passes(&self) -> bool {
        self.sets.len() == SETS.len() && self.sets.iter().all(SetResult::passes)
    }
    pub fn in_margin(&self) -> usize {
        self.sets.iter().filter(|r| r.passes()).count()
    }
    pub fn exact(&self) -> u32 {
        self.sets.iter().map(|r| r.exact).sum()
    }
    pub fn elements(&self) -> u32 {
        self.sets.iter().map(|r| r.elements).sum()
    }
    /// The worst ‰ over the sets (`u32::MAX` if any set had none).
    pub fn worst_permille(&self) -> u32 {
        self.sets
            .iter()
            .map(|r| r.worst_permille.unwrap_or(u32::MAX))
            .max()
            .unwrap_or(0)
    }
    /// Median launch time (ns) over the sets that have one.
    pub fn median_ns(&self) -> Option<u64> {
        let mut v: Vec<u64> = self.sets.iter().filter_map(|r| r.ns).collect();
        if v.is_empty() {
            return None;
        }
        v.sort_unstable();
        Some(v[v.len() / 2])
    }

    /// The `kt colsum …` rows (values in the stick journal, read back by [`parse_colsum`]).
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        let c = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        let mut out = alloc::vec![
            ("kt colsum seturi", self.sets.len() as u32),
            ("kt colsum seturi in margine", self.in_margin() as u32),
            ("kt colsum elemente", self.elements()),
            ("kt colsum elemente exacte (model)", self.exact()),
            (
                "kt colsum abaterea maxima (promile din margine)",
                self.worst_permille()
            ),
        ];
        if let Some(ns) = self.median_ns() {
            out.push(("kt colsum timp median (ns)", c(ns)));
        }
        for r in &self.sets {
            out.extend([
                ("kt colsum set: index", r.set as u32),
                ("kt colsum set: intrari (fnv)", SETS[r.set].inputs_fnv()),
                ("kt colsum set: verdict", r.code()),
                ("kt colsum set: exacte", r.exact),
                (
                    "kt colsum set: promile",
                    r.worst_permille.unwrap_or(u32::MAX),
                ),
                ("kt colsum set: garzi", u32::from(r.guards_ok)),
                ("kt colsum set: iesire (fnv)", r.out_fnv),
            ]);
            if let Some(ns) = r.ns {
                out.push(("kt colsum set: timp (ns)", c(ns)));
            }
        }
        out
    }

    pub fn journal(&self, j: &mut Journal) {
        for (what, value) in self.rows() {
            j.push(Event::Value { what, value });
        }
    }

    /// Report lines (`label`: "SIMULARE" or "placa").
    pub fn lines(&self, label: &str) -> Vec<String> {
        let mut out = alloc::vec![format!(
            "colsum ({label}): {} / {} seturi in margine, {} / {} elemente exacte fata de model, abaterea maxima {} promile din margine",
            self.in_margin(),
            self.sets.len(),
            self.exact(),
            self.elements(),
            self.worst_permille()
        )];
        for r in &self.sets {
            let t = r.ns.map(|n| format!(", {n} ns")).unwrap_or_default();
            let w = r
                .worst_permille
                .map(|p| format!("{p}"))
                .unwrap_or_else(|| String::from("-"));
            out.push(format!(
                "colsum set {} ({label}): {}, exacte {}/{}, {} promile, garzi {}{}",
                SETS[r.set].name,
                ["NU TRECE", "in margine", "INVALID"][r.code() as usize],
                r.exact,
                r.elements,
                w,
                if r.guards_ok { "neatinse" } else { "ATINSE" },
                t
            ));
        }
        out
    }
}

/// What [`parse_colsum`] reads back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedColsum {
    pub sets: Option<u32>,
    pub in_margin: Option<u32>,
    pub elements: Option<u32>,
    pub exact: Option<u32>,
    pub worst_permille: Option<u32>,
    pub median_ns: Option<u32>,
    /// Per set, in order: (index, inputs fnv, verdict, exact, ‰, guards, output fnv, ns or `u32::MAX`).
    pub per_set: Vec<[u32; 8]>,
}

/// Reads the rows of [`Probe::rows`] (`gpu: valoare kt colsum … = 0x…`). A new `kt colsum seturi` row starts over
/// (the last run in a journal is kept).
pub fn parse_colsum(text: &str) -> ParsedColsum {
    let mut p = ParsedColsum::default();
    for line in text.lines() {
        let Some(rest) = line.split("valoare kt colsum ").nth(1) else {
            continue;
        };
        let Some((name, val)) = rest.rsplit_once(" = 0x") else {
            continue;
        };
        let Ok(v) = u32::from_str_radix(val.trim(), 16) else {
            continue;
        };
        let last = |p: &mut ParsedColsum, k: usize| {
            if let Some(s) = p.per_set.last_mut() {
                s[k] = v;
            }
        };
        match name {
            "seturi" => {
                p = ParsedColsum {
                    sets: Some(v),
                    ..ParsedColsum::default()
                }
            }
            "seturi in margine" => p.in_margin = Some(v),
            "elemente" => p.elements = Some(v),
            "elemente exacte (model)" => p.exact = Some(v),
            "abaterea maxima (promile din margine)" => p.worst_permille = Some(v),
            "timp median (ns)" => p.median_ns = Some(v),
            "set: index" => p.per_set.push([v, 0, 0, 0, 0, 0, 0, u32::MAX]),
            "set: intrari (fnv)" => last(&mut p, 1),
            "set: verdict" => last(&mut p, 2),
            "set: exacte" => last(&mut p, 3),
            "set: promile" => last(&mut p, 4),
            "set: garzi" => last(&mut p, 5),
            "set: iesire (fnv)" => last(&mut p, 6),
            "set: timp (ns)" => last(&mut p, 7),
            _ => {}
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_fit_and_are_aligned() {
        for s in &SETS {
            let l = s.layout();
            assert!(l.end <= BUF_BYTES, "{}: {}", s.name, l.end);
            for o in [l.dy, l.x, l.mean, l.rstd, l.s0, l.s1] {
                assert_eq!(o % 256, 0);
            }
            assert_eq!(s.buffer().len(), BUF_BYTES);
        }
    }

    #[test]
    fn model_output_passes_and_unwritten_is_invalid() {
        for (i, s) in SETS.iter().enumerate() {
            let r = judge_set(i, &s.model_out(), None).unwrap();
            assert!(r.passes(), "{}: {r:?}", s.name);
            assert_eq!(r.exact, r.elements);
            let (o, n) = s.out_range();
            let r = judge_set(i, &s.buffer()[o..o + n], None).unwrap();
            assert_eq!(r.verdict, Verdict::Invalid);
        }
        assert!(judge_set(9, &[], None).is_err());
        assert_eq!(judge_set(0, &[0; 8], None).unwrap().code(), 2);
    }

    #[test]
    fn sets_differ() {
        let f: Vec<u32> = SETS.iter().map(ProbeSet::inputs_fnv).collect();
        for i in 0..f.len() {
            for j in i + 1..f.len() {
                assert_ne!(f[i], f[j]);
            }
        }
    }

    #[test]
    fn launch_bytes_checks_alignment() {
        let (cb, _) = launch_bytes(3, 0x1000, 0x2000, 0x10_000).unwrap();
        let at = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
        let l = SETS[3].layout();
        assert_eq!(at(0x28), 0x10_000);
        assert_eq!(at(0x48), 0x10_000 + l.s0 as u64);
        assert!(cb[cs::CBUF_BYTES..].iter().all(|&b| b == 0));
        assert!(launch_bytes(3, 0x1000, 0x2000, 0x10_200).is_err());
        assert!(launch_bytes(3, 0x1000, 0x2010, 0x10_000).is_err());
        assert!(launch_bytes(7, 0x1000, 0x2000, 0x10_000).is_err());
    }
}
