//! C-168 point 4, step 3 (`docs/C-168_ATTN_FWD_TC_PLAN.md` §10, predictions H4 / H5 written before any board run):
//! **the probe `--atentie-fwd-tc`** — [`crate::attn_fwd_tc`] alone on the board, beside [`crate::attn_fwd_tile3`] on
//! the same inputs, outside any training step.
//!
//! - [`SETS`]: four small **aligned** shapes (d = 64, T a multiple of 64 — the only shapes `attn_fwd_tc` takes; none of
//!   the five small sets of [`crate::attn_tile_probe::SETS`] is aligned) read back whole, and the timing set
//!   `atentie-513-b4` (the same as the attention probe's, read back in bands).
//! - The buffer of a set is [`ProbeSet::layout`] (only qkv, P and att are used). Before **each** of the two programs the
//!   OS fills [`fwd_ranges`] (the P rows and att rows read back + their guards) with [`FILL`].
//! - [`launch_bytes`]: cbuf 0 (0x800 bytes, the arguments at 0x700) and the QMD of `attn_fwd_tc` **at the 64 KiB shared
//!   step** ([`STEP_BYTES`]: 2 blocks per SM; at its own 32 KiB step only 1 would fit). `attn_fwd_tile3` is launched
//!   as in the attention probe: [`crate::attn_tile_probe::launch_bytes3`] with `Prog3(Prog::Fwd)`.
//! - [`judge`]: the two read-backs compared — O (att) and P (j ≤ i) as relative norms against `attn_fwd_tile3`'s
//!   ([`O_LIMIT`], [`P_LIMIT`]), exact 0 above the diagonal inside the diagonal tile, the tiles above the diagonal tile
//!   **not written** (still [`FILL`]), the guards untouched; on the small sets also the words equal to the CPU model
//!   `gpu_compute::attn_fwd_tc::fwd_tile` (counted, not required: the board's `MUFU.EX2` / `RCP` are approximate).
//! - [`Probe`]: report lines ([`Probe::lines`]) and [`parse`] for the lines the OS sends back.
//!
//! Tested **only in SIMULARE** (T301, `tests/sim_c168_attn_fwd_tc_probe.rs`); nothing here has run on a board.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::attn_fwd_tc as m;
use gpu_core::error::{GpuError, GpuResult};

use crate::attn_fwd_tc as tc;
pub use crate::attn_tile_probe::{FILL, GUARD, ProbeSet};
use crate::qmd::ComputeQmd;

/// The SM shared-memory step of every `attn_fwd_tc` launch (as in the training step): 2 blocks per SM.
pub const STEP_BYTES: u32 = 64 << 10;
/// O (att) against `attn_fwd_tile3`: ‖tc − tile3‖ / ‖tile3‖ at most this (H3 / H4).
pub const O_LIMIT: f64 = 1.0e-3;
/// P (j ≤ i) against `attn_fwd_tile3`: at most this (H3 / H4).
pub const P_LIMIT: f64 = 2.0e-3;
/// Index of the timing set in [`SETS`].
pub const TIMING_SET: usize = 4;
/// `attn_fwd_tile3` on `atentie-513-b4` measured by the OS (os-364 … os-375, median of 5): ~1,00 ms.
pub const TILE3_513_B4_NS: u64 = 1_001_000;
/// H5 for this probe (DEDUS, §10): `attn_fwd_tc` on `atentie-513-b4` at the 64 KiB step, ns (low, high).
pub const PREDICTED_513_B4_NS: (u64, u64) = (367_000, 620_000);

const fn small(
    name: &'static str,
    batch: u32,
    t: u32,
    heads: u32,
    seed: u64,
    amp: f32,
) -> ProbeSet {
    ProbeSet {
        name,
        batch,
        t,
        c: 64 * heads,
        heads,
        seed,
        amp,
        bands: None,
        repeats: 1,
    }
}

/// The probe: four small aligned shapes, then the timing set (the attention probe's `atentie-513-b4`).
pub const SETS: [ProbeSet; 5] = [
    // one tile: only the diagonal tile, one block
    small("tc-t64", 1, 64, 1, 21, 2.0),
    // three row tiles: the middle block does one tile (the pair loop's odd case); two texts
    small("tc-t192-b2", 2, 192, 1, 22, 2.0),
    // four row tiles, two heads: two full pairs per head
    small("tc-t256-h2", 1, 256, 2, 23, 2.0),
    // five row tiles, a sharper softmax (qkv in [−4, 4))
    small("tc-t320-ascutit", 1, 320, 1, 24, 4.0),
    crate::attn_tile_probe::SETS[crate::attn_tile_probe::TIMING_SET],
];

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// May `attn_fwd_tc` run on this set (its [`tc::fits`] for a buffer at a 4 KiB address)?
pub fn fits(set: &ProbeSet) -> bool {
    set.shape_ok() && tc::fits(&set.fwd_args(0x10_0000))
}

/// The byte ranges `(offset, len)` the OS fills with [`FILL`] before each program and reads back after it, in the
/// order [`judge`] expects them concatenated: the P rows (b, h, i) read back, the P guard, the att rows (b, i), the
/// att guard. Adjacent ranges are merged.
pub fn fwd_ranges(set: &ProbeSet) -> Vec<(usize, usize)> {
    let (t, c, h) = (set.t as usize, set.c as usize, set.heads as usize);
    let l = set.layout();
    let (texts, rows) = (set.texts(), set.rows());
    let mut v: Vec<(usize, usize)> = Vec::new();
    let mut push = |o: usize, n: usize| match v.last_mut() {
        Some(last) if last.0 + last.1 == o => last.1 += n,
        _ => v.push((o, n)),
    };
    for &b in &texts {
        for hh in 0..h {
            for &i in &rows {
                push(l.p + 4 * ((b * h + hh) * t + i) * t, 4 * t);
            }
        }
    }
    push(l.p_guard, 4 * GUARD);
    for &b in &texts {
        for &i in &rows {
            push(l.att + 4 * (b * t + i) * c, 4 * c);
        }
    }
    push(l.att_guard, 4 * GUARD);
    v
}

/// Bytes of one read-back ([`fwd_ranges`] in total).
pub fn fwd_read_len(set: &ProbeSet) -> usize {
    fwd_ranges(set).iter().map(|r| r.1).sum()
}

/// Constant buffer 0 ([`tc::CBUF_BYTES`]) and the QMD (at [`STEP_BYTES`]) of `attn_fwd_tc` on `set`, whose buffer is at
/// `buf_va` (4 KiB aligned); `prog_va` and `cbuf_va` aligned to 256. An error for a set the program does not take.
pub fn launch_bytes(
    set: &ProbeSet,
    prog_va: u64,
    cbuf_va: u64,
    buf_va: u64,
) -> GpuResult<([u8; tc::CBUF_BYTES], [u8; nvidia_rm570::QMD_SIZE])> {
    if !buf_va.is_multiple_of(4096) || !cbuf_va.is_multiple_of(256) || !prog_va.is_multiple_of(256)
    {
        return Err(bad(
            "atentie-fwd-tc: bufferul nealiniat la 4 KiB sau programul / cbuf nealiniate la 256",
        ));
    }
    if !set.shape_ok() {
        return Err(bad("atentie-fwd-tc: forma setului"));
    }
    let a = set.fwd_args(buf_va);
    let cb = tc::cbuf(&a).ok_or(bad("atentie-fwd-tc: setul nu e aliniat (fits)"))?;
    let q: ComputeQmd = tc::qmd(&a, prog_va, cbuf_va, None).ok_or(bad("atentie-fwd-tc: QMD"))?;
    Ok((cb, q.to_bytes_with_sm_config(&tc::SHARED, STEP_BYTES)?))
}

/// What one set gave.
#[derive(Clone, Debug, PartialEq)]
pub struct SetResult {
    pub name: &'static str,
    /// ‖O_tc − O_tile3‖ / ‖O_tile3‖ (`None`: a non-finite value, a zero norm or a bad length).
    pub o_rel: Option<f64>,
    /// The same for P on j ≤ i.
    pub p_rel: Option<f64>,
    /// The largest |P_tc − P_tile3| on j ≤ i.
    pub p_worst_abs: f64,
    /// Inside the diagonal tile, j > i: exact +0 in `attn_fwd_tc`'s P.
    pub zeros_ok: bool,
    /// The tiles above the diagonal tile still [`FILL`] in `attn_fwd_tc`'s P (nothing written there).
    pub unwritten_ok: bool,
    /// Both programs' P and att guards still [`FILL`].
    pub guards_ok: bool,
    /// Small sets: (words equal to the CPU model, words compared) — P on j ≤ i and att. `None` on the banded set.
    pub exact_model: Option<(u32, u32)>,
    /// Launch times (ns) of `attn_fwd_tile3` and `attn_fwd_tc`, if the OS measured them.
    pub ns_tile3: Vec<u64>,
    pub ns_tc: Vec<u64>,
}

fn median(v: &[u64]) -> Option<u64> {
    if v.is_empty() {
        return None;
    }
    let mut v = v.to_vec();
    v.sort_unstable();
    Some(v[v.len() / 2])
}

impl SetResult {
    /// H4 on this set: both norms within their limits, the zeros, nothing written above, the guards untouched.
    pub fn passes(&self) -> bool {
        self.o_rel.is_some_and(|x| x <= O_LIMIT)
            && self.p_rel.is_some_and(|x| x <= P_LIMIT)
            && self.zeros_ok
            && self.unwritten_ok
            && self.guards_ok
    }
    pub fn median_tile3(&self) -> Option<u64> {
        median(&self.ns_tile3)
    }
    pub fn median_tc(&self) -> Option<u64> {
        median(&self.ns_tc)
    }
}

fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

fn sqrt64(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    // Newton from a bit-level first guess; 6 steps are far past f64 precision for the norms used here.
    let mut r = f64::from_bits((x.to_bits() >> 1) + (1023u64 << 51));
    for _ in 0..6 {
        r = 0.5 * (r + x / r);
    }
    r
}

#[derive(Default)]
struct Norm {
    num: f64,
    den: f64,
    finite: bool,
}

impl Norm {
    fn new() -> Self {
        Norm {
            num: 0.0,
            den: 0.0,
            finite: true,
        }
    }
    fn add(&mut self, got: f32, want: f32) -> f64 {
        if !got.is_finite() || !want.is_finite() {
            self.finite = false;
            return f64::INFINITY;
        }
        let d = got as f64 - want as f64;
        self.num += d * d;
        self.den += want as f64 * want as f64;
        d.abs()
    }
    fn rel(&self) -> Option<f64> {
        (self.finite && self.den > 0.0).then(|| sqrt64(self.num / self.den))
    }
}

/// Judges set `set` from the read-backs of `attn_fwd_tile3` (`tile3`) and `attn_fwd_tc` (`tc_out`), each the bytes of
/// [`fwd_ranges`] concatenated, and the launch times. An error for a wrong length or a set `attn_fwd_tc` does not take.
pub fn judge(
    set: &ProbeSet,
    tile3: &[u8],
    tc_out: &[u8],
    ns_tile3: &[u64],
    ns_tc: &[u64],
) -> GpuResult<SetResult> {
    if !fits(set) {
        return Err(bad("atentie-fwd-tc: setul nu e aliniat (fits)"));
    }
    let len = fwd_read_len(set);
    if tile3.len() != len || tc_out.len() != len {
        return Err(bad("atentie-fwd-tc: lungimea citita"));
    }
    let (r, g) = (words(tile3), words(tc_out));
    let (t, c, h) = (set.t as usize, set.c as usize, set.heads as usize);
    let (texts, rows) = (set.texts(), set.rows());
    let np = texts.len() * h * rows.len() * t;
    let na = texts.len() * rows.len() * c;
    let f = f32::from_bits;
    let (mut on, mut pn) = (Norm::new(), Norm::new());
    let mut p_worst_abs = 0.0f64;
    let (mut zeros_ok, mut unwritten_ok) = (true, true);
    let mut k = 0;
    for _ in &texts {
        for _ in 0..h {
            for &i in &rows {
                let tile_end = (i / 64 + 1) * 64;
                for j in 0..t {
                    let (rw, gw) = (r[k + j], g[k + j]);
                    if j <= i {
                        p_worst_abs = p_worst_abs.max(pn.add(f(gw), f(rw)));
                    } else if j < tile_end {
                        zeros_ok &= gw == 0;
                    } else {
                        unwritten_ok &= gw == FILL;
                    }
                }
                k += t;
            }
        }
    }
    let guard_ok = |w: &[u32]| w.iter().all(|&x| x == FILL);
    let mut guards_ok = guard_ok(&r[np..np + GUARD]) && guard_ok(&g[np..np + GUARD]);
    let a0 = np + GUARD;
    for idx in 0..na {
        on.add(f(g[a0 + idx]), f(r[a0 + idx]));
    }
    guards_ok &= guard_ok(&r[a0 + na..]) && guard_ok(&g[a0 + na..]);
    let exact_model = if set.bands.is_none() {
        exact_against_model(set, &g)
    } else {
        None
    };
    Ok(SetResult {
        name: set.name,
        o_rel: on.rel(),
        p_rel: pn.rel(),
        p_worst_abs,
        zeros_ok,
        unwritten_ok,
        guards_ok,
        exact_model,
        ns_tile3: ns_tile3.to_vec(),
        ns_tc: ns_tc.to_vec(),
    })
}

/// Words of `attn_fwd_tc`'s read-back (P on j ≤ i, att) bit for bit equal to the CPU model, of those compared. Small
/// (whole) sets only.
fn exact_against_model(set: &ProbeSet, g: &[u32]) -> Option<(u32, u32)> {
    let s = set.shape();
    let (nq, _, _) = s.sizes()?;
    let qkv: Vec<f32> = (0..nq).map(|i| set.qkv_at(i)).collect();
    let (p, att) = m::fwd_model(&s, &qkv, set.scale(), f32::from_bits(FILL))?;
    let (t, np) = (s.t, s.batch * s.heads * s.t * s.t);
    let (mut same, mut all) = (0u32, 0u32);
    for (idx, &w) in g[..np].iter().enumerate() {
        let (i, j) = ((idx / t) % t, idx % t);
        if j <= i {
            all += 1;
            same += (p[idx].to_bits() == w) as u32;
        }
    }
    let a0 = np + GUARD;
    for (idx, &w) in g[a0..a0 + att.len()].iter().enumerate() {
        all += 1;
        same += (att[idx].to_bits() == w) as u32;
    }
    Some((same, all))
}

/// The whole probe.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Probe {
    pub sets: Vec<SetResult>,
}

fn sci(x: Option<f64>) -> String {
    match x {
        None => String::from("INVALID"),
        Some(0.0) => String::from("0"),
        Some(v) => {
            let mut e = 0i32;
            let mut m = v;
            while m >= 10.0 {
                m /= 10.0;
                e += 1;
            }
            while m < 1.0 {
                m *= 10.0;
                e -= 1;
            }
            let tenths = (m * 10.0 + 0.5) as u32;
            let (a, b) = if tenths >= 100 {
                (1, 0)
            } else {
                (tenths / 10, tenths % 10)
            };
            let e = if tenths >= 100 { e + 1 } else { e };
            format!("{a},{b}e{e}")
        }
    }
}

fn ms3(ns: u64) -> String {
    let us = (ns + 500) / 1000;
    format!("{},{:03} ms", us / 1000, us % 1000)
}

impl Probe {
    pub fn add(&mut self, r: SetResult) {
        self.sets.push(r);
    }
    /// H4: every set judged and passing.
    pub fn passes(&self) -> bool {
        !self.sets.is_empty() && self.sets.iter().all(SetResult::passes)
    }
    pub fn in_limits(&self) -> usize {
        self.sets.iter().filter(|s| s.passes()).count()
    }
    /// Report lines (`label` = "placa" or "SIMULARE"); one per set, then the verdict and the time against H5.
    pub fn lines(&self, label: &str) -> Vec<String> {
        let mut out = Vec::new();
        for s in &self.sets {
            let exact = match s.exact_model {
                Some((a, b)) => format!(", exacte fata de model {a}/{b}"),
                None => String::new(),
            };
            let time = match (s.median_tile3(), s.median_tc()) {
                (Some(a), Some(b)) if b > 0 => {
                    format!(", tile3 {} ns, tc {} ns, x{}", a, b, ratio2(a, b))
                }
                _ => String::new(),
            };
            out.push(format!(
                "atentie-fwd-tc set {} ({label}): O {} P {} (cel mai rau P {}), zerouri {}, nescrise {}, garzi {}{exact}{time}: {}",
                s.name,
                sci(s.o_rel),
                sci(s.p_rel),
                sci(Some(s.p_worst_abs)),
                if s.zeros_ok { "da" } else { "NU" },
                if s.unwritten_ok { "da" } else { "NU" },
                if s.guards_ok { "neatinse" } else { "ATINSE" },
                if s.passes() { "TRECE" } else { "NU TRECE" },
            ));
        }
        out.push(format!(
            "atentie-fwd-tc ({label}): {} / {} seturi in margine (O <= 1e-3, P <= 2e-3): {}",
            self.in_limits(),
            self.sets.len(),
            if self.passes() { "TRECE" } else { "NU TRECE" }
        ));
        if let Some(s) = self.sets.iter().find(|s| s.name == SETS[TIMING_SET].name)
            && let Some(tcns) = s.median_tc()
        {
            let (lo, hi) = PREDICTED_513_B4_NS;
            let verdict = if tcns < lo {
                "sub fereastra"
            } else if tcns <= hi {
                "in fereastra"
            } else {
                "peste fereastra"
            };
            out.push(format!(
                "atentie-fwd-tc timp ({label}): atentie-513-b4 attn_fwd_tc mediana {} (prezis {} - {}): {verdict}",
                ms3(tcns),
                ms3(lo),
                ms3(hi)
            ));
        }
        out
    }
}

fn ratio2(a: u64, b: u64) -> String {
    let h = (a * 100 + b / 2) / b;
    format!("{},{:02}", h / 100, h % 100)
}

/// One set as read from a report line: (name, O, P, passes, tile3 ns, tc ns).
#[derive(Clone, Debug, PartialEq)]
pub struct Parsed {
    pub name: String,
    pub o_rel: Option<f64>,
    pub p_rel: Option<f64>,
    pub passes: bool,
    pub ns_tile3: Option<u64>,
    pub ns_tc: Option<u64>,
}

fn parse_sci(w: &str) -> Option<f64> {
    if w == "0" {
        return Some(0.0);
    }
    let (m, e) = w.split_once('e')?;
    let (a, b) = m.split_once(',')?;
    let mant = a.parse::<u32>().ok()? as f64 + b.parse::<u32>().ok()? as f64 / 10.0;
    let e: i32 = e.parse().ok()?;
    let mut v = mant;
    for _ in 0..e.unsigned_abs() {
        if e > 0 { v *= 10.0 } else { v /= 10.0 }
    }
    Some(v)
}

/// The `atentie-fwd-tc set …` lines of a report (any prefix before them is ignored).
pub fn parse(text: &str) -> Vec<Parsed> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.split("atentie-fwd-tc set ").nth(1) else {
            continue;
        };
        let name = rest.split(' ').next().unwrap_or("");
        let num = |key: &str| {
            rest.split(key)
                .nth(1)
                .and_then(|x| x.split(' ').next())
                .and_then(|w| w.parse::<u64>().ok())
        };
        let sci_after = |key: &str| {
            rest.split(key)
                .nth(1)
                .and_then(|x| x.split(' ').next())
                .and_then(parse_sci)
        };
        out.push(Parsed {
            name: String::from(name),
            o_rel: sci_after(": O "),
            p_rel: sci_after(" P "),
            passes: rest.trim_end().ends_with(": TRECE"),
            ns_tile3: num("tile3 "),
            ns_tc: num(", tc "),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_fit_and_others_do_not() {
        for s in &SETS {
            assert!(fits(s), "{}", s.name);
        }
        let n = crate::attn_tile_probe::SETS
            .iter()
            .filter(|s| fits(s))
            .count();
        assert_eq!(
            n, 1,
            "only atentie-513-b4 of the attention probe is aligned"
        );
    }

    #[test]
    fn sci_and_parse_round_trip() {
        assert_eq!(sci(Some(2.73e-4)), "2,7e-4");
        assert_eq!(sci(Some(9.96e-4)), "1,0e-3");
        assert_eq!(parse_sci("2,7e-4").map(|v| (v * 1e5).round()), Some(27.0));
        assert_eq!(ratio2(1_000_000, 400_000), "2,50");
    }
}
