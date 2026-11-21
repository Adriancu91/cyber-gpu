//! C-160 point 1 — **P8, the first tensor-core computation on the real board** (approved by Adrian, 21 Nov 2026
//! 16:18): everything the OS needs to send [`crate::mma16`] (`mma16_f16`, Mesa NAK 26.3.1) a few times, outside
//! the generation path, and to judge what comes back.
//!
//! - [`SETS`]: the fixed probe inputs (A, B binary16; C f32), made here from fixed seeds, so the OS and the tests
//!   use the same bytes ([`ProbeSet::inputs`], [`ProbeSet::page`], [`ProbeSet::inputs_fnv`]).
//! - [`ProbeSet::page`]: one 4 KiB page of host RAM per set: A at [`A_OFF`], B at [`B_OFF`], C at [`C_OFF`], D at
//!   [`D_OFF`] filled with NaN ([`D_FILL`]), so an element the board did not write cannot pass.
//! - [`launch_bytes`]: the constant buffer (0x800 bytes, addresses at 0x700) and the QMD for one set.
//! - [`judge_set`] → [`SetResult`]: the written tolerance (`gpu_compute::mma::judge`), how far the worst element
//!   is (‰ of its tolerance), and which candidate adder models (`gpu_compute::mma::ADD_MODELS`) the board
//!   matches bit for bit — that last one answers "how many terms does GA104 add at once, and how does it round",
//!   which no NVIDIA document says.
//! - [`Probe`]: the run (all sets), its `kt mma16 …` journal rows ([`Probe::rows`]), report lines and
//!   [`parse_mma16`] to read the rows back from a stick journal.
//!
//! Tested **only in SIMULARE** (T150, `tests/sim_c160_p8.rs`); nothing here has run on a board. The verdict
//! (tolerance) was written in C-147 point 5, before any run; this module does not change it.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::mma::{self, ADD_MODELS, MmaVerdict};
use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use crate::mma16;

/// One page of host RAM per set.
pub const PAGE_BYTES: usize = 4096;
/// A (16×16 binary16, row-major, 512 B).
pub const A_OFF: u64 = 0x000;
/// B (16×16 binary16, row-major, 512 B).
pub const B_OFF: u64 = 0x200;
/// C (16×16 f32, row-major, 1 KiB).
pub const C_OFF: u64 = 0x400;
/// D (16×16 f32, row-major, 1 KiB): what the board writes.
pub const D_OFF: u64 = 0x800;
/// Bytes of D.
pub const D_BYTES: usize = 1024;
/// The word D is filled with before a launch (a quiet NaN): an unwritten element is never "within tolerance".
pub const D_FILL: u32 = 0x7fc0_0000;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// How the inputs of one set are made.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// A = identity, B from the seed in [-2, 2), C = 0: D must be B exactly (every model agrees). Catches a
    /// transposed or shifted layout before any arithmetic question.
    Identity(u64),
    /// A, B, C from the seed, all in [-2, 2) (C rounded to binary16 first, then widened).
    Uniform(u64),
    /// A, B from the seed with exponents spread over 2⁻¹⁴..2¹ (subnormals included), C = 3 × such a value: sums
    /// with very different sizes, where the adder's block and rounding show.
    Scales(u64),
}

/// One probe set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeSet {
    pub name: &'static str,
    pub kind: Kind,
}

/// The probe: 6 launches of the same program, each with its own page.
pub const SETS: [ProbeSet; 6] = [
    ProbeSet {
        name: "identitate",
        kind: Kind::Identity(11),
    },
    ProbeSet {
        name: "uniform-1",
        kind: Kind::Uniform(1),
    },
    ProbeSet {
        name: "uniform-2",
        kind: Kind::Uniform(2),
    },
    ProbeSet {
        name: "uniform-3",
        kind: Kind::Uniform(3),
    },
    ProbeSet {
        name: "scari-1",
        kind: Kind::Scales(21),
    },
    ProbeSet {
        name: "scari-2",
        kind: Kind::Scales(22),
    },
];

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// [-2, 2) rounded to binary16.
    fn uniform(&mut self) -> u16 {
        mma::f32_to_f16(self.next() as f32 / (1u64 << 31) as f32 * 2.0 - 2.0)
    }
    /// Any sign, biased exponent 0..=16 (2⁻¹⁴ and subnormals up to just under 4), any mantissa.
    fn scaled(&mut self) -> u16 {
        let r = self.next();
        ((r & 1) as u16) << 15 | (((r >> 1) % 17) as u16) << 10 | ((r >> 8) & 0x3ff) as u16
    }
}

/// The inputs of one set.
#[derive(Clone, Debug, PartialEq)]
pub struct Inputs {
    pub a: [u16; 256],
    pub b: [u16; 256],
    pub c: [f32; 256],
}

impl ProbeSet {
    pub fn inputs(&self) -> Inputs {
        let mut x = Inputs {
            a: [0; 256],
            b: [0; 256],
            c: [0.0; 256],
        };
        match self.kind {
            Kind::Identity(seed) => {
                let mut g = Lcg(seed);
                for i in 0..16 {
                    x.a[i * 16 + i] = 0x3c00;
                }
                x.b.iter_mut().for_each(|v| *v = g.uniform());
            }
            Kind::Uniform(seed) => {
                let mut g = Lcg(seed);
                x.a.iter_mut().for_each(|v| *v = g.uniform());
                x.b.iter_mut().for_each(|v| *v = g.uniform());
                x.c.iter_mut()
                    .for_each(|v| *v = mma::f16_to_f32(g.uniform()));
            }
            Kind::Scales(seed) => {
                let mut g = Lcg(seed);
                x.a.iter_mut().for_each(|v| *v = g.scaled());
                x.b.iter_mut().for_each(|v| *v = g.scaled());
                x.c.iter_mut()
                    .for_each(|v| *v = mma::f16_to_f32(g.scaled()) * 3.0);
            }
        }
        x
    }

    /// The host page for this set (module doc): A, B, C, then D = [`D_FILL`]; the rest zero.
    pub fn page(&self) -> [u8; PAGE_BYTES] {
        let x = self.inputs();
        let mut p = [0u8; PAGE_BYTES];
        for (i, h) in x.a.iter().enumerate() {
            let at = A_OFF as usize + 2 * i;
            p[at..at + 2].copy_from_slice(&h.to_le_bytes());
        }
        for (i, h) in x.b.iter().enumerate() {
            let at = B_OFF as usize + 2 * i;
            p[at..at + 2].copy_from_slice(&h.to_le_bytes());
        }
        for (i, f) in x.c.iter().enumerate() {
            let at = C_OFF as usize + 4 * i;
            p[at..at + 4].copy_from_slice(&f.to_bits().to_le_bytes());
        }
        for i in 0..256 {
            let at = D_OFF as usize + 4 * i;
            p[at..at + 4].copy_from_slice(&D_FILL.to_le_bytes());
        }
        p
    }

    /// FNV-1a (32 bit) of the A, B, C bytes of [`Self::page`] (the first 2 KiB): written in the journal, so a run
    /// is known to have used exactly these inputs.
    pub fn inputs_fnv(&self) -> u32 {
        let p = self.page();
        let mut h: u32 = 0x811c_9dc5;
        for &b in &p[..D_OFF as usize] {
            h ^= b as u32;
            h = h.wrapping_mul(0x0100_0193);
        }
        h
    }
}

/// The constant buffer (0x800 B: zero, then the addresses of A, B, C, D at 0x700..0x720) and the QMD (one block of
/// 32 threads, 32 registers, no shared memory, cbuf size 0x800) for one set whose page is at `page_va` (GPU VA of
/// the host page, aligned to 4 KiB). `prog_va`: where [`mma16::MMA16_F16`] was loaded; `cbuf_va`: where the cbuf
/// bytes go (aligned to 256).
pub fn launch_bytes(
    prog_va: u64,
    cbuf_va: u64,
    page_va: u64,
) -> GpuResult<(
    [u8; mma16::CBUF_SIZE as usize],
    [u8; nvidia_rm570::QMD_SIZE],
)> {
    if !page_va.is_multiple_of(PAGE_BYTES as u64) || !cbuf_va.is_multiple_of(256) {
        return Err(bad(
            "mma16 proba: pagina nealiniata la 4 KiB sau cbuf nealiniat la 256",
        ));
    }
    let cb = mma16::cbuf(
        page_va + A_OFF,
        page_va + B_OFF,
        page_va + C_OFF,
        page_va + D_OFF,
    )?;
    let q = mma16::qmd(prog_va, cbuf_va)?;
    Ok((cb, q))
}

/// What one set gave.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SetResult {
    /// Index in [`SETS`].
    pub set: usize,
    pub verdict: MmaVerdict,
    /// Worst element, in ‰ of its tolerance (`None`: D not finite / not written).
    pub worst_permille: Option<u32>,
    /// `gpu_compute::mma::matching_models` (bit i = `ADD_MODELS[i]` matches bit for bit).
    pub models: u32,
    /// The launch time on the board clock, if the OS measured it (records before / after).
    pub ns: Option<u64>,
}

impl SetResult {
    pub fn passes(&self) -> bool {
        matches!(self.verdict, MmaVerdict::InToleranta { .. })
    }
    pub fn exact(&self) -> usize {
        match self.verdict {
            MmaVerdict::InToleranta { exact } => exact,
            _ => 0,
        }
    }
    /// 1 = within tolerance, 0 = outside, 2 = invalid (NaN / not written / wrong length).
    pub fn code(&self) -> u32 {
        match self.verdict {
            MmaVerdict::InToleranta { .. } => 1,
            MmaVerdict::NuTrece { .. } => 0,
            MmaVerdict::Invalid => 2,
        }
    }
}

/// D (1 KiB read back from [`D_OFF`]) as 256 f32; `None` for a wrong length.
pub fn d_from_bytes(d: &[u8]) -> Option<[f32; 256]> {
    if d.len() != D_BYTES {
        return None;
    }
    let mut out = [0.0f32; 256];
    for (o, w) in out.iter_mut().zip(d.as_chunks::<4>().0) {
        *o = f32::from_bits(u32::from_le_bytes(*w));
    }
    Some(out)
}

/// Judges the D bytes the board wrote for set `set` (index in [`SETS`]).
pub fn judge_set(set: usize, d: &[u8], ns: Option<u64>) -> GpuResult<SetResult> {
    let s = SETS.get(set).ok_or(bad("mma16 proba: set necunoscut"))?;
    let x = s.inputs();
    let Some(got) = d_from_bytes(d) else {
        return Ok(SetResult {
            set,
            verdict: MmaVerdict::Invalid,
            worst_permille: None,
            models: 0,
            ns,
        });
    };
    Ok(SetResult {
        set,
        verdict: mma::judge(&x.a, &x.b, &x.c, &got),
        worst_permille: mma::worst_permille(&x.a, &x.b, &x.c, &got),
        models: mma::matching_models(&x.a, &x.b, &x.c, &got),
        ns,
    })
}

/// Names of the models in a mask, joined with ", " (`niciunul` for 0).
pub fn model_names(mask: u32) -> String {
    let v: Vec<&str> = ADD_MODELS
        .iter()
        .enumerate()
        .filter(|(i, _)| mask & (1 << i) != 0)
        .map(|(_, m)| m.name())
        .collect();
    if v.is_empty() {
        String::from("niciunul")
    } else {
        v.join(", ")
    }
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
    /// All [`SETS`] judged, all within tolerance.
    pub fn passes(&self) -> bool {
        self.sets.len() == SETS.len() && self.sets.iter().all(SetResult::passes)
    }
    pub fn in_tolerance(&self) -> usize {
        self.sets.iter().filter(|r| r.passes()).count()
    }
    pub fn exact(&self) -> usize {
        self.sets.iter().map(SetResult::exact).sum()
    }
    /// The worst ‰ over the sets (`u32::MAX` if any set had none: D not finite).
    pub fn worst_permille(&self) -> u32 {
        self.sets
            .iter()
            .map(|r| r.worst_permille.unwrap_or(u32::MAX))
            .max()
            .unwrap_or(0)
    }
    /// The models that matched on **every** set (AND of the masks; the identity set matches all of them).
    pub fn models(&self) -> u32 {
        if self.sets.is_empty() {
            return 0;
        }
        self.sets.iter().fold(u32::MAX, |m, r| m & r.models) & ((1 << ADD_MODELS.len()) - 1)
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

    /// The `kt mma16 …` rows (values in the stick journal, read back by [`parse_mma16`]).
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        let c = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        let mut out = alloc::vec![
            ("kt mma16 seturi", self.sets.len() as u32),
            ("kt mma16 seturi in toleranta", self.in_tolerance() as u32),
            ("kt mma16 elemente exacte", self.exact() as u32),
            (
                "kt mma16 abaterea maxima (promile din toleranta)",
                self.worst_permille()
            ),
            ("kt mma16 modele potrivite (masca)", self.models()),
        ];
        if let Some(ns) = self.median_ns() {
            out.push(("kt mma16 timp median (ns)", c(ns)));
        }
        for r in &self.sets {
            out.extend([
                ("kt mma16 set: index", r.set as u32),
                ("kt mma16 set: intrari (fnv)", SETS[r.set].inputs_fnv()),
                ("kt mma16 set: verdict", r.code()),
                ("kt mma16 set: exacte", r.exact() as u32),
                (
                    "kt mma16 set: promile",
                    r.worst_permille.unwrap_or(u32::MAX),
                ),
                ("kt mma16 set: modele (masca)", r.models),
            ]);
            if let Some(ns) = r.ns {
                out.push(("kt mma16 set: timp (ns)", c(ns)));
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
            "mma16 ({label}): {} / {} seturi in toleranta, {} / {} elemente exacte, abaterea maxima {} promile din toleranta; modele potrivite pe toate seturile: {}",
            self.in_tolerance(),
            self.sets.len(),
            self.exact(),
            256 * self.sets.len(),
            self.worst_permille(),
            model_names(self.models())
        )];
        for r in &self.sets {
            let t = r.ns.map(|n| format!(", {n} ns")).unwrap_or_default();
            let w = r
                .worst_permille
                .map(|p| format!("{p}"))
                .unwrap_or_else(|| String::from("-"));
            out.push(format!(
                "mma16 set {} ({label}): {} exacte {}/256, {} promile, modele: {}{}",
                SETS[r.set].name,
                ["NU TRECE", "in toleranta", "INVALID"][r.code() as usize],
                r.exact(),
                w,
                model_names(r.models),
                t
            ));
        }
        out
    }
}

/// What [`parse_mma16`] reads back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedMma16 {
    pub sets: Option<u32>,
    pub in_tolerance: Option<u32>,
    pub exact: Option<u32>,
    pub worst_permille: Option<u32>,
    pub models: Option<u32>,
    pub median_ns: Option<u32>,
    /// Per set, in order: (index, fnv, verdict, exact, ‰, models, ns or `u32::MAX`).
    pub per_set: Vec<[u32; 7]>,
}

/// Reads the rows of [`Probe::rows`] (`gpu: valoare kt mma16 … = 0x…`). A new `kt mma16 seturi` row starts over
/// (the last run in a journal is kept).
pub fn parse_mma16(text: &str) -> ParsedMma16 {
    let mut p = ParsedMma16::default();
    for line in text.lines() {
        let Some(rest) = line.split("valoare kt mma16 ").nth(1) else {
            continue;
        };
        let Some((name, val)) = rest.rsplit_once(" = 0x") else {
            continue;
        };
        let Ok(v) = u32::from_str_radix(val.trim(), 16) else {
            continue;
        };
        let last = |p: &mut ParsedMma16, k: usize| {
            if let Some(s) = p.per_set.last_mut() {
                s[k] = v;
            }
        };
        match name {
            "seturi" => {
                p = ParsedMma16 {
                    sets: Some(v),
                    ..ParsedMma16::default()
                }
            }
            "seturi in toleranta" => p.in_tolerance = Some(v),
            "elemente exacte" => p.exact = Some(v),
            "abaterea maxima (promile din toleranta)" => p.worst_permille = Some(v),
            "modele potrivite (masca)" => p.models = Some(v),
            "timp median (ns)" => p.median_ns = Some(v),
            "set: index" => p.per_set.push([v, 0, 0, 0, 0, 0, u32::MAX]),
            "set: intrari (fnv)" => last(&mut p, 1),
            "set: verdict" => last(&mut p, 2),
            "set: exacte" => last(&mut p, 3),
            "set: promile" => last(&mut p, 4),
            "set: modele (masca)" => last(&mut p, 5),
            "set: timp (ns)" => last(&mut p, 6),
            _ => {}
        }
    }
    p
}

/// C-157 point 2 (after os-335): what the real board (RTX 3070 Ti, GA104) wrote in os-335, 21 Nov 2026 22:21,
/// per set in [`SETS`] order — MEASURED, copied from `gpu-jurnal-034.txt` (`kt mma16 set: exacte` / `promile`).
pub const OS335_EXACT: [u32; 6] = [256, 80, 63, 71, 89, 83];
/// See [`OS335_EXACT`]: the worst element per set, in ‰ of its tolerance.
pub const OS335_PERMILLE: [u32; 6] = [0, 108, 72, 110, 148, 124];

/// FNV-1a (32 bit) of the D bytes a launch wrote (1 KiB): one row per set lets a later reader check any adder
/// model bit for bit against the board without the 1 KiB itself in the journal.
pub fn d_fnv(d: &[u8]) -> u32 {
    let mut h: u32 = 0x811c_9dc5;
    for &b in d {
        h ^= b as u32;
        h = h.wrapping_mul(0x0100_0193);
    }
    h
}

/// D of set `set` under [`gpu_compute::mma::GA104_OS335`], as the 1 KiB the board would write.
pub fn ga104_fit_bytes(set: usize) -> GpuResult<[u8; D_BYTES]> {
    let s = SETS.get(set).ok_or(bad("mma16 proba: set necunoscut"))?;
    let x = s.inputs();
    let d = mma::mma16_hw(&x.a, &x.b, &x.c, mma::GA104_OS335)
        .ok_or(bad("mma16 proba: modelul GA104 a refuzat intrarile"))?;
    let mut out = [0u8; D_BYTES];
    for (o, v) in out.as_chunks_mut::<4>().0.iter_mut().zip(&d) {
        *o = v.to_bits().to_le_bytes();
    }
    Ok(out)
}

/// The two extra rows for one set (C-157 point 2, optional for the OS: call it next to [`judge_set`] with the
/// same D bytes): `kt mma16 set: iesire (fnv)` = [`d_fnv`] of D, and `kt mma16 set: model ga104` = 1 when D is
/// bit for bit [`ga104_fit_bytes`], 0 when not, 2 for a wrong length.
pub fn fit_rows(set: usize, d: &[u8]) -> GpuResult<[(&'static str, u32); 2]> {
    let want = ga104_fit_bytes(set)?;
    let m = if d.len() != D_BYTES {
        2
    } else {
        u32::from(d == want)
    };
    Ok([
        ("kt mma16 set: iesire (fnv)", d_fnv(d)),
        ("kt mma16 set: model ga104", m),
    ])
}

/// Pushes [`fit_rows`] into the journal.
pub fn journal_fit(j: &mut Journal, set: usize, d: &[u8]) -> GpuResult<()> {
    for (what, value) in fit_rows(set, d)? {
        j.push(Event::Value { what, value });
    }
    Ok(())
}

/// Reads the [`fit_rows`] back from a stick journal, in order: (D fnv, model ga104; `u32::MAX` if its row is
/// missing). Every pair in the text is returned (the OS may write them before or after [`Probe::journal`]); the
/// last [`SETS`]`.len()` pairs are the last run.
pub fn parse_mma16_fit(text: &str) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.split("valoare kt mma16 ").nth(1) else {
            continue;
        };
        let Some((name, val)) = rest.rsplit_once(" = 0x") else {
            continue;
        };
        let Ok(v) = u32::from_str_radix(val.trim(), 16) else {
            continue;
        };
        match name {
            "set: iesire (fnv)" => out.push((v, u32::MAX)),
            "set: model ga104" => {
                if let Some(last) = out.last_mut() {
                    last.1 = v;
                }
            }
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_set_gives_b_and_matches_every_model() {
        let s = SETS[0];
        let x = s.inputs();
        let d = mma::mma16_exact(&x.a, &x.b, &x.c).unwrap();
        for (got, b) in d.iter().zip(&x.b) {
            assert_eq!(*got, mma::f16_to_f32(*b));
        }
        assert_eq!(
            mma::matching_models(&x.a, &x.b, &x.c, &d),
            (1 << ADD_MODELS.len()) - 1
        );
    }

    #[test]
    fn page_layout_and_fill() {
        let p = SETS[1].page();
        let x = SETS[1].inputs();
        assert_eq!(u16::from_le_bytes([p[0], p[1]]), x.a[0]);
        assert_eq!(
            u16::from_le_bytes([p[B_OFF as usize + 2], p[B_OFF as usize + 3]]),
            x.b[1]
        );
        let c = &p[C_OFF as usize..C_OFF as usize + 4];
        assert_eq!(
            f32::from_bits(u32::from_le_bytes(c.try_into().unwrap())),
            x.c[0]
        );
        let d = d_from_bytes(&p[D_OFF as usize..D_OFF as usize + D_BYTES]).unwrap();
        assert!(d.iter().all(|v| v.to_bits() == D_FILL));
        assert!(p[D_OFF as usize + D_BYTES..].iter().all(|&b| b == 0));
    }

    #[test]
    fn sets_differ_and_fnv_is_stable() {
        let f: Vec<u32> = SETS.iter().map(ProbeSet::inputs_fnv).collect();
        for i in 0..f.len() {
            for j in i + 1..f.len() {
                assert_ne!(f[i], f[j]);
            }
        }
        assert_eq!(f[0], SETS[0].inputs_fnv());
    }

    #[test]
    fn launch_bytes_checks_alignment() {
        let (cb, _) = launch_bytes(0x1000, 0x2000, 0x10_000).unwrap();
        let at = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
        assert_eq!(at(0x700), 0x10_000);
        assert_eq!(at(0x718), 0x10_800);
        assert!(launch_bytes(0x1000, 0x2000, 0x10_200).is_err());
        assert!(launch_bytes(0x1000, 0x2010, 0x10_000).is_err());
    }

    #[test]
    fn unwritten_d_is_invalid() {
        let p = SETS[2].page();
        let r = judge_set(2, &p[D_OFF as usize..D_OFF as usize + D_BYTES], None).unwrap();
        assert_eq!(r.verdict, MmaVerdict::Invalid);
        assert_eq!(r.code(), 2);
        assert!(judge_set(9, &[], None).is_err());
        assert_eq!(judge_set(1, &[0; 8], None).unwrap().code(), 2);
    }
}
