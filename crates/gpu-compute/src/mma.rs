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
