//! C-147 point 5 (tensor cores, T0–T1, preparation only): the oracle for the
//! first cooperative-matrix program, `tools/nak/mma16_f16.comp`:
//!
//! `D[16,16] = A[16,16] · B[16,16] + C[16,16]`, A and B in IEEE 16-bit floats
//! (binary16), C and D in f32, all row-major.
//!
//! Three things live here, kept apart on purpose:
//! - [`f16_to_f32`] / [`f32_to_f16`]: exact widening and round-to-nearest-even
//!   narrowing (no_std, bit operations only);
//! - [`mma16_exact`]: the **reference** — every product of two binary16 values
//!   is exact in f64, the 17 terms of an element are added in f64 and rounded
//!   once to f32. This is "the right answer", not what any board computes;
//! - [`mma16_block8_trunc`]: a **hypothesis** of how an Ampere tensor core
//!   adds (published measurements on V100/T4/A100: products exact, inner
//!   products of a few terms added at once, the result truncated toward zero).
//!   `gpu-sim` uses it so the model is not suspiciously exact; it is never
//!   evidence of what the RTX 3070 Ti does.
//!
//! The verdict ([`judge`], written before any run): every element must be
//! within [`tolerance`] = `TOL_TERMS · 2⁻²³ · (|c| + Σ|a·b|)` of the reference.
//! Bit-for-bit equality is **not** asked for: the order and rounding of the
//! hardware adder are not documented by NVIDIA.

/// Rows / columns / depth of the one shape the program uses.
pub const M: usize = 16;
pub const N: usize = 16;
pub const K: usize = 16;
/// Terms per element in the tolerance: the K products plus C.
pub const TOL_TERMS: u32 = K as u32 + 1;
/// Terms the [`mma16_block8_trunc`] hypothesis adds in one step.
pub const HYPOTHESIS_BLOCK: usize = 8;

/// binary16 → f32, exact for every value (subnormals, ±0, ±inf, NaN kept as NaN).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = match (exp, man) {
        (0, 0) => sign,
        (0, m) => {
            // Subnormal: m · 2⁻²⁴, exact in f32.
            let v = m as f32 * f32::from_bits((127 - 24) << 23);
            v.to_bits() | sign
        }
        (0x1f, 0) => sign | 0x7f80_0000,
        (0x1f, m) => sign | 0x7fc0_0000 | (m << 13),
        (e, m) => sign | ((e + 127 - 15) << 23) | (m << 13),
    };
    f32::from_bits(bits)
}

/// f32 → binary16, round to nearest, ties to even; overflow → ±inf; NaN → a quiet NaN.
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign
            | 0x7c00
            | if man != 0 {
                0x200 | (man >> 13) as u16
            } else {
                0
            };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign; // below half of the smallest subnormal (2⁻²⁵)
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let half = 1u32 << (shift - 1);
        let rem = m & ((1u32 << shift) - 1);
        let mut q = m >> shift;
        if rem > half || (rem == half && q & 1 == 1) {
            q += 1;
        }
        return sign | q as u16;
    }
    let mut q = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && q & 1 == 1) {
        q += 1;
    }
    sign | q as u16
}

fn check(a: &[u16], b: &[u16], c: &[f32]) -> bool {
    a.len() == M * K && b.len() == K * N && c.len() == M * N
}

/// The reference D = A·B + C (module doc). `None` if a slice has the wrong length.
pub fn mma16_exact(a: &[u16], b: &[u16], c: &[f32]) -> Option<[f32; M * N]> {
    if !check(a, b, c) {
        return None;
    }
    let mut d = [0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut s = c[i * N + j] as f64;
            for p in 0..K {
                s += f16_to_f32(a[i * K + p]) as f64 * f16_to_f32(b[p * N + j]) as f64;
            }
            d[i * N + j] = s as f32;
        }
    }
    Some(d)
}

/// `x` (f64) to f32 rounded toward zero.
fn to_f32_trunc(x: f64) -> f32 {
    let r = x as f32;
    if r.is_finite() && r.abs() as f64 > x.abs() {
        // One step toward zero (r ≠ 0 here, since |r| > |x| ≥ 0).
        f32::from_bits(r.to_bits() - 1)
    } else {
        r
    }
}

/// The hypothesis (module doc): per element, the accumulator starts at C; each
/// block of [`HYPOTHESIS_BLOCK`] exact products is added to it at once and the
/// sum truncated toward zero to f32. SIMULARE only.
pub fn mma16_block8_trunc(a: &[u16], b: &[u16], c: &[f32]) -> Option<[f32; M * N]> {
    if !check(a, b, c) {
        return None;
    }
    let mut d = [0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut acc = c[i * N + j];
            for blk in (0..K).step_by(HYPOTHESIS_BLOCK) {
                let mut s = acc as f64;
                for p in blk..blk + HYPOTHESIS_BLOCK {
                    s += f16_to_f32(a[i * K + p]) as f64 * f16_to_f32(b[p * N + j]) as f64;
                }
                acc = to_f32_trunc(s);
            }
            d[i * N + j] = acc;
        }
    }
    Some(d)
}

/// C-160 point 1 (P8): how one add step of a candidate model rounds to f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rounding {
    /// Round to nearest, ties to even (what `as f32` does).
    Nearest,
    /// Toward zero (the published V100/T4/A100 behaviour).
    TowardZero,
}

/// C-160 point 1 (P8): one candidate model of the tensor-core adder: the accumulator starts at C, every block of
/// `block` exact products is added to it at once, and the sum is rounded to f32 with `rounding`. The board's D is
/// compared bit for bit with each of [`ADD_MODELS`]: the ones that match on every probe set say how GA104 adds
/// (MEASURED only once a board has answered; in SIMULARE the model is [`mma16_block8_trunc`] by construction).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AddModel {
    pub block: usize,
    pub rounding: Rounding,
}

impl AddModel {
    /// Short name for a report line, e.g. `bloc8-taiere`, `bloc16-rotunjire` (= the reference rounded once).
    pub fn name(self) -> &'static str {
        match (self.block, self.rounding) {
            (1, Rounding::Nearest) => "bloc1-rotunjire",
            (2, Rounding::Nearest) => "bloc2-rotunjire",
            (4, Rounding::Nearest) => "bloc4-rotunjire",
            (8, Rounding::Nearest) => "bloc8-rotunjire",
            (16, Rounding::Nearest) => "bloc16-rotunjire",
            (1, Rounding::TowardZero) => "bloc1-taiere",
            (2, Rounding::TowardZero) => "bloc2-taiere",
            (4, Rounding::TowardZero) => "bloc4-taiere",
            (8, Rounding::TowardZero) => "bloc8-taiere",
            (16, Rounding::TowardZero) => "bloc16-taiere",
            _ => "alt-model",
        }
    }
}

/// The 10 candidate models, in the bit order of [`matching_models`]'s mask (bit i = `ADD_MODELS[i]`).
pub const ADD_MODELS: [AddModel; 10] = {
    const fn m(block: usize, rounding: Rounding) -> AddModel {
        AddModel { block, rounding }
    }
    [
        m(1, Rounding::Nearest),
        m(2, Rounding::Nearest),
        m(4, Rounding::Nearest),
        m(8, Rounding::Nearest),
        m(16, Rounding::Nearest),
        m(1, Rounding::TowardZero),
        m(2, Rounding::TowardZero),
        m(4, Rounding::TowardZero),
        m(8, Rounding::TowardZero),
        m(16, Rounding::TowardZero),
    ]
};

/// D under one [`AddModel`]. `None` for wrong lengths or a block that does not divide [`K`].
/// `mma16_blocked(.., AddModel { block: 8, rounding: TowardZero })` = [`mma16_block8_trunc`].
pub fn mma16_blocked(a: &[u16], b: &[u16], c: &[f32], model: AddModel) -> Option<[f32; M * N]> {
    if !check(a, b, c) || model.block == 0 || !K.is_multiple_of(model.block) {
        return None;
    }
    let mut d = [0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut acc = c[i * N + j];
            for blk in (0..K).step_by(model.block) {
                let mut s = acc as f64;
                for p in blk..blk + model.block {
                    s += f16_to_f32(a[i * K + p]) as f64 * f16_to_f32(b[p * N + j]) as f64;
                }
                acc = match model.rounding {
                    Rounding::Nearest => s as f32,
                    Rounding::TowardZero => to_f32_trunc(s),
                };
            }
            d[i * N + j] = acc;
        }
    }
    Some(d)
}

/// Bit mask of the [`ADD_MODELS`] whose D equals `got` bit for bit on every element (bit i = `ADD_MODELS[i]`).
/// 0 = no candidate fits exactly (the board may still be within [`tolerance`]); 0 also for wrong lengths.
pub fn matching_models(a: &[u16], b: &[u16], c: &[f32], got: &[f32]) -> u32 {
    if got.len() != M * N {
        return 0;
    }
    let mut mask = 0;
    for (i, m) in ADD_MODELS.iter().enumerate() {
        if let Some(d) = mma16_blocked(a, b, c, *m)
            && d.iter().zip(got).all(|(x, y)| x.to_bits() == y.to_bits())
        {
            mask |= 1 << i;
        }
    }
    mask
}

/// C-157 point 2 (after os-335): a finer, hardware-shaped model of the tensor-core adder, closer to how a
/// fixed-point block adder works than [`AddModel`]:
///
/// - per element the accumulator starts at C; every block of `block` products is added to it in one step;
/// - in a step, the reference exponent `E` is the largest **nominal** exponent of the non-zero terms: for a
///   product `a·b` it is `exp(a) + exp(b)` (binary16 exponents, subnormals = −14, so the product significand is in
///   [0, 4) with two integer bits), for the accumulator its real exponent;
/// - every term is cut toward zero to a multiple of `2^(E − 23 − extra_bits)` (alignment with truncation), the
///   cut terms are added exactly, and the sum is rounded toward zero to f32.
///
/// [`GA104_OS335`] is the one member of this family that reproduces every figure the real board wrote in os-335
/// (exact counts and worst ‰ of all 6 sets). That is a match on the **fingerprint**, not yet bit for bit: the
/// journal rows of [`mma16_hw`] in `nvidia_ga104::mma16_probe::fit_rows` let the next board run confirm it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HwAdder {
    /// Products added per step (must divide [`K`]).
    pub block: usize,
    /// Bits kept below the 24-bit f32 significand when aligning (0 = none).
    pub extra_bits: i32,
}

/// The model that fits the real RTX 3070 Ti (GA104) in os-335, 21 Nov 2026 (MEASURED fingerprint, see
/// [`HwAdder`]): 8 products + the accumulator per step, one extra alignment bit, truncation everywhere.
pub const GA104_OS335: HwAdder = HwAdder {
    block: 8,
    extra_bits: 1,
};

/// Unbiased exponent of a binary16 for the nominal product exponent (subnormals and zero count as −14).
fn h_exp(h: u16) -> i32 {
    let f = ((h >> 10) & 0x1f) as i32;
    if f == 0 { -14 } else { f - 15 }
}

/// Unbiased exponent of a finite, non-zero, normal f64.
fn f64_exp(x: f64) -> i32 {
    ((x.to_bits() >> 52) & 0x7ff) as i32 - 1023
}

/// 2^e as f64 (e within the normal f64 range).
fn pow2(e: i32) -> f64 {
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// D under one [`HwAdder`]. `None` for wrong lengths, a block that does not divide [`K`], `extra_bits` outside
/// 0..=8, or a non-finite input.
pub fn mma16_hw(a: &[u16], b: &[u16], c: &[f32], model: HwAdder) -> Option<[f32; M * N]> {
    if !check(a, b, c)
        || model.block == 0
        || model.block > K
        || !K.is_multiple_of(model.block)
        || !(0..=8).contains(&model.extra_bits)
        || c.iter().any(|v| !v.is_finite())
        || a.iter().chain(b).any(|h| (h >> 10) & 0x1f == 0x1f)
    {
        return None;
    }
    let mut d = [0.0f32; M * N];
    let mut terms = [0.0f64; K + 1];
    for i in 0..M {
        for j in 0..N {
            let mut acc = c[i * N + j];
            for blk in (0..K).step_by(model.block) {
                let n = model.block + 1;
                terms[0] = acc as f64;
                let mut e_max: Option<i32> = (acc != 0.0).then(|| f64_exp(acc as f64));
                for (t, p) in (blk..blk + model.block).enumerate() {
                    let (x, y) = (a[i * K + p], b[p * N + j]);
                    let v = f16_to_f32(x) as f64 * f16_to_f32(y) as f64;
                    terms[t + 1] = v;
                    if v != 0.0 {
                        let e = h_exp(x) + h_exp(y);
                        e_max = Some(e_max.map_or(e, |m| m.max(e)));
                    }
                }
                let Some(e) = e_max else {
                    acc = 0.0;
                    continue;
                };
                // Every term is below 2^(e+2) (product significand < 4, accumulator ≤ its own exponent), so a
                // cut term has at most 2 + 23 + 8 = 33 bits above q and a sum of at most 17 terms fits in 38:
                // the additions below are exact in f64.
                let q = pow2(e - 23 - model.extra_bits);
                let mut s = 0.0f64;
                for t in &terms[..n] {
                    // Toward zero: `as i64` truncates; |t/q| < 2^34 here.
                    s += (t / q) as i64 as f64 * q;
                }
                acc = to_f32_trunc(s);
            }
            d[i * N + j] = acc;
        }
    }
    Some(d)
}

/// The worst element's distance from the reference, in thousandths of its [`tolerance`] (1000 = exactly at the
/// limit; above 1000 = outside). `None` for wrong lengths or a non-finite input or result. A zero tolerance with a
/// non-zero distance counts as `u32::MAX`.
pub fn worst_permille(a: &[u16], b: &[u16], c: &[f32], got: &[f32]) -> Option<u32> {
    let r = mma16_exact(a, b, c)?;
    if got.len() != M * N || r.iter().chain(got).any(|v| !v.is_finite()) {
        return None;
    }
    let mut worst = 0u32;
    for idx in 0..M * N {
        let dist = (got[idx] as f64 - r[idx] as f64).abs();
        let tol = tolerance(a, b, c, idx / N, idx % N);
        let p = if dist == 0.0 {
            0
        } else if tol == 0.0 {
            u32::MAX
        } else {
            // Rounded up by hand (no `f64::ceil` in no_std).
            let x = dist / tol * 1000.0;
            if x >= u32::MAX as f64 {
                u32::MAX
            } else {
                let q = x as u32;
                if (q as f64) < x { q + 1 } else { q }
            }
        };
        worst = worst.max(p);
    }
    Some(worst)
}

/// The allowed distance from the reference for element (i, j): `TOL_TERMS · 2⁻²³ · (|c| + Σ|a·b|)`.
pub fn tolerance(a: &[u16], b: &[u16], c: &[f32], i: usize, j: usize) -> f64 {
    let mut s = (c[i * N + j] as f64).abs();
    for p in 0..K {
        s += (f16_to_f32(a[i * K + p]) as f64 * f16_to_f32(b[p * N + j]) as f64).abs();
    }
    TOL_TERMS as f64 * s / (1u64 << 23) as f64
}

/// What [`judge`] says about a board's (or the model's) D.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MmaVerdict {
    /// Every element within tolerance; `exact` = how many equal the reference bit for bit.
    InToleranta { exact: usize },
    /// The first element outside tolerance (row-major index), with the values.
    NuTrece {
        index: usize,
        got: f32,
        reference: f32,
        tol: f64,
    },
    /// Wrong slice lengths, or a non-finite input or result.
    Invalid,
}

/// Judges `got` (16×16, row-major) against [`mma16_exact`] with [`tolerance`].
pub fn judge(a: &[u16], b: &[u16], c: &[f32], got: &[f32]) -> MmaVerdict {
    let Some(r) = mma16_exact(a, b, c) else {
        return MmaVerdict::Invalid;
    };
    if got.len() != M * N || r.iter().chain(got).any(|v| !v.is_finite()) {
        return MmaVerdict::Invalid;
    }
    let mut exact = 0;
    for idx in 0..M * N {
        let tol = tolerance(a, b, c, idx / N, idx % N);
        if (got[idx] as f64 - r[idx] as f64).abs() > tol {
            return MmaVerdict::NuTrece {
                index: idx,
                got: got[idx],
                reference: r[idx],
                tol,
            };
        }
        if got[idx].to_bits() == r[idx].to_bits() {
            exact += 1;
        }
    }
    MmaVerdict::InToleranta { exact }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
        /// A finite binary16 in about [-4, 4] (exponents 0..=16 biased, any mantissa, any sign).
        fn h(&mut self) -> u16 {
            let r = self.next();
            ((r & 1) as u16) << 15 | (((r >> 1) % 17) as u16) << 10 | ((r >> 8) & 0x3ff) as u16
        }
    }

    fn inputs(
        seed: u64,
    ) -> (
        alloc::vec::Vec<u16>,
        alloc::vec::Vec<u16>,
        alloc::vec::Vec<f32>,
    ) {
        let mut g = Lcg(seed);
        let a = (0..M * K).map(|_| g.h()).collect();
        let b = (0..K * N).map(|_| g.h()).collect();
        let c = (0..M * N).map(|_| f16_to_f32(g.h()) * 3.0).collect();
        (a, b, c)
    }

    #[test]
    fn every_finite_f16_round_trips() {
        for h in 0..=u16::MAX {
            let f = f16_to_f32(h);
            if f.is_nan() {
                assert!(f16_to_f32(f32_to_f16(f)).is_nan());
                continue;
            }
            assert_eq!(f32_to_f16(f), h, "h={h:#06x}");
        }
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0x0001), 2.0f32.powi(-24));
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }

    #[test]
    fn narrowing_rounds_to_nearest_even() {
        // 1 + 2^-11 is halfway between 1 and 1 + 2^-10: ties to even -> 1.
        assert_eq!(f32_to_f16(1.0 + 2.0f32.powi(-11)), 0x3c00);
        // 1 + 3·2^-11 is halfway between 1 + 2^-10 (odd) and 1 + 2^-9: -> 1 + 2^-9.
        assert_eq!(f32_to_f16(1.0 + 3.0 * 2.0f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_f16(65520.0), 0x7c00); // rounds up to inf
        assert_eq!(f32_to_f16(65519.0), 0x7bff);
        assert_eq!(f32_to_f16(2.0f32.powi(-25)), 0x0000); // tie to even (0)
        assert_eq!(f32_to_f16(3.0 * 2.0f32.powi(-25)), 0x0002); // 1.5 units -> 2
        assert_eq!(f32_to_f16(-0.0), 0x8000);
    }

    #[test]
    fn identity_times_b_plus_zero_is_b() {
        let mut a = [0u16; M * K];
        for i in 0..M {
            a[i * K + i] = 0x3c00;
        }
        let (_, b, _) = inputs(7);
        let c = [0.0f32; M * N];
        let d = mma16_exact(&a, &b, &c).unwrap();
        for i in 0..M * N {
            assert_eq!(d[i], f16_to_f32(b[i]));
        }
        assert_eq!(
            judge(&a, &b, &c, &d),
            MmaVerdict::InToleranta { exact: M * N }
        );
    }

    #[test]
    fn the_hypothesis_stays_in_tolerance_and_is_not_always_exact() {
        let mut differs = 0;
        for seed in 1..=20 {
            let (a, b, c) = inputs(seed);
            let h = mma16_block8_trunc(&a, &b, &c).unwrap();
            match judge(&a, &b, &c, &h) {
                MmaVerdict::InToleranta { exact } => differs += M * N - exact,
                v => panic!("seed {seed}: {v:?}"),
            }
        }
        assert!(differs > 0, "truncation should show in the last bits");
    }

    #[test]
    fn blocked_models_match_the_older_functions() {
        for seed in 1..=5 {
            let (a, b, c) = inputs(seed);
            let t8 = AddModel {
                block: 8,
                rounding: Rounding::TowardZero,
            };
            assert_eq!(
                mma16_blocked(&a, &b, &c, t8),
                mma16_block8_trunc(&a, &b, &c)
            );
            let r16 = AddModel {
                block: 16,
                rounding: Rounding::Nearest,
            };
            assert_eq!(mma16_blocked(&a, &b, &c, r16), mma16_exact(&a, &b, &c));
            let h = mma16_block8_trunc(&a, &b, &c).unwrap();
            assert_ne!(matching_models(&a, &b, &c, &h) & (1 << 8), 0);
            assert!(worst_permille(&a, &b, &c, &h).unwrap() <= 1000);
        }
        let (a, b, c) = inputs(1);
        let bad = AddModel {
            block: 3,
            rounding: Rounding::Nearest,
        };
        assert_eq!(mma16_blocked(&a, &b, &c, bad), None);
        assert_eq!(bad.name(), "alt-model");
        assert_eq!(ADD_MODELS[8].name(), "bloc8-taiere");
    }

    #[test]
    fn hw_adder_stays_in_tolerance_and_refuses_bad_shapes() {
        for seed in 1..=20 {
            let (a, b, c) = inputs(seed);
            let d = mma16_hw(&a, &b, &c, GA104_OS335).unwrap();
            assert!(matches!(
                judge(&a, &b, &c, &d),
                MmaVerdict::InToleranta { .. }
            ));
        }
        let (a, b, c) = inputs(1);
        let bad = |block, extra_bits| mma16_hw(&a, &b, &c, HwAdder { block, extra_bits }).is_none();
        assert!(bad(0, 1) && bad(3, 1) && bad(32, 1) && bad(8, -1) && bad(8, 9));
        let mut inf = a.clone();
        inf[5] = 0x7c00;
        assert_eq!(mma16_hw(&inf, &b, &c, GA104_OS335), None);
        // Identity · B + 0 = B for every member of the family.
        let mut id = [0u16; M * K];
        for i in 0..M {
            id[i * K + i] = 0x3c00;
        }
        let z = [0.0f32; M * N];
        let d = mma16_hw(&id, &b, &z, GA104_OS335).unwrap();
        for (x, y) in d.iter().zip(&b) {
            assert_eq!(*x, f16_to_f32(*y));
        }
    }

    #[test]
    fn worst_permille_sees_a_planted_error() {
        let (a, b, c) = inputs(4);
        let mut r = mma16_exact(&a, &b, &c).unwrap();
        assert_eq!(worst_permille(&a, &b, &c, &r), Some(0));
        r[100] *= 1.01;
        assert!(worst_permille(&a, &b, &c, &r).unwrap() > 1000);
        assert_eq!(worst_permille(&a, &b, &c, &r[..3]), None);
        assert_eq!(matching_models(&a, &b, &c, &r), 0);
    }

    #[test]
    fn planted_errors_are_caught() {
        let (a, b, c) = inputs(3);
        let r = mma16_exact(&a, &b, &c).unwrap();
        // One element 1 % off.
        let mut bad = r;
        bad[37] *= 1.01;
        assert!(matches!(
            judge(&a, &b, &c, &bad),
            MmaVerdict::NuTrece { index: 37, .. }
        ));
        // B read transposed (a classic layout mistake).
        let bt: alloc::vec::Vec<u16> = (0..K * N).map(|i| b[(i % N) * N + i / N]).collect();
        let wrong = mma16_exact(&a, &bt, &c).unwrap();
        assert!(matches!(
            judge(&a, &b, &c, &wrong),
            MmaVerdict::NuTrece { .. }
        ));
        // C forgotten.
        let zero = [0.0f32; M * N];
        let no_c = mma16_exact(&a, &b, &zero).unwrap();
        assert!(matches!(
            judge(&a, &b, &c, &no_c),
            MmaVerdict::NuTrece { .. }
        ));
        // Wrong length, NaN.
        assert_eq!(judge(&a, &b, &c, &r[..10]), MmaVerdict::Invalid);
        let mut nan = r;
        nan[0] = f32::NAN;
        assert_eq!(judge(&a, &b, &c, &nan), MmaVerdict::Invalid);
    }
}
