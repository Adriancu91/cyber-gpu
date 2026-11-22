//! C-160 point 7, part two — the verdict and the CPU model for `tools/nak/mm_tc.comp`, the training matmul on the
//! tensor cores: `C = op(A)·op(B) (+ bias)` with the same interface as `mm_train` (A, B, C, bias in f32; flags
//! [`TA`], [`TB`], [`BIAS`]), but every element of op(A) and op(B) rounded to binary16 before the products, and the
//! products added in f32 by the tensor cores (f16 × f16 → f32).
//!
//! Kept apart on purpose (like [`crate::mma`]):
//! - [`reference`]: **the right answer for the rounded inputs** — the products of the f16 values are exact in f64,
//!   summed in f64, plus the bias. Not what any board computes.
//! - [`margin`]: how far a correct tensor-core sum may be from [`reference`], written **before** any run:
//!   `(K + 3)·2⁻²³·(Σ|â·b̂| + |bias|) + 2⁻¹²⁶` (â, b̂ = the f16-rounded values). Derivation in
//!   `docs/C-160_P7_MM_TC.md` §3: the adder model fitted on the real board in os-335 ([`crate::mma::GA104_OS335`],
//!   8 products per step, truncation) stays under `(K/8)·5,5·2⁻²³·Σ`; round-to-nearest under `K·2⁻²⁴·Σ`; the
//!   margin covers both with room.
//! - [`rounding_margin`]: what rounding the inputs to f16 may cost against the **f32 training oracle** (the unrounded
//!   inputs): `Σ(|a|·|b|·(2⁻¹⁰ + 2⁻²²) + 2⁻²⁵·(|a| + |b|) + 2⁻⁵⁰)` on top of [`margin`]. This is the price of the
//!   speed, not a fault of the program; it is reported apart so nobody mistakes it for "the same numbers".
//! - [`model`]: the **hypothesis** of what the board computes (the kernel's order: 64×64 blocks, K in slices of 32,
//!   16 at a time, each 16-deep step = [`crate::mma::mma16_hw`] with [`crate::mma::GA104_OS335`]; then one f32 add of
//!   the bias with subnormals flushed). `gpu-sim` uses it; it is never evidence of what the RTX 3070 Ti does.
//!
//! Inputs outside the binary16 range (|x| > 65 504) become ±inf in the program: [`reference`] and [`model`] refuse
//! them (`None`) instead of pretending. Tiny values (|x| < 2⁻²⁴ ≈ 6·10⁻⁸) become 0 — gradients that small need
//! scaling before `mm_tc` (the "loss scaling" of mixed-precision training; the next step of point 7).

use alloc::vec::Vec;

use crate::math::add_f32_ftz;
use crate::mma::{self, GA104_OS335, HwAdder, f16_to_f32, f32_to_f16};

/// A is stored transposed (`[K, M]`).
pub const TA: u32 = 1;
/// B is stored transposed (`[N, K]`, PyTorch's `weight [out, in]`).
pub const TB: u32 = 2;
/// Add `bias[col]` after the sum.
pub const BIAS: u32 = 4;
/// The C block of one workgroup (rows = columns).
pub const BLOCK_MN: usize = 64;
/// The K slice copied to shared memory at a time.
pub const BLOCK_K: usize = 32;
/// The largest finite binary16.
pub const F16_MAX: f32 = 65504.0;

/// One product `C[m, n] = op(A)[m, k]·op(B)[k, n] (+ bias)`, as the program sees it.
#[derive(Clone, Copy, Debug)]
pub struct Product<'a> {
    pub a: &'a [f32],
    pub b: &'a [f32],
    /// Read only with [`BIAS`].
    pub bias: Option<&'a [f32]>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub flags: u32,
}

impl Product<'_> {
    /// Lengths, flags and the binary16 range checked; `false` = the program would compute garbage.
    pub fn valid(&self) -> bool {
        let finite16 = |v: &f32| v.is_finite() && v.abs() <= F16_MAX;
        self.m > 0
            && self.n > 0
            && self.k > 0
            && self.flags & !(TA | TB | BIAS) == 0
            && self.a.len() >= self.m * self.k
            && self.b.len() >= self.k * self.n
            && (self.flags & BIAS == 0 || self.bias.is_some_and(|b| b.len() >= self.n))
            && self.a[..self.m * self.k].iter().all(finite16)
            && self.b[..self.k * self.n].iter().all(finite16)
            && self.bias_slice().iter().all(|v| v.is_finite())
    }

    /// op(A)[r, p] as stored (f32).
    pub fn op_a(&self, r: usize, p: usize) -> f32 {
        if self.flags & TA != 0 {
            self.a[p * self.m + r]
        } else {
            self.a[r * self.k + p]
        }
    }

    /// op(B)[p, c] as stored (f32).
    pub fn op_b(&self, p: usize, c: usize) -> f32 {
        if self.flags & TB != 0 {
            self.b[c * self.k + p]
        } else {
            self.b[p * self.n + c]
        }
    }

    fn bias_slice(&self) -> &[f32] {
        match (self.flags & BIAS != 0, self.bias) {
            (true, Some(b)) => &b[..self.n.min(b.len())],
            _ => &[],
        }
    }

    fn bias_at(&self, c: usize) -> f32 {
        if self.flags & BIAS != 0 {
            self.bias.map_or(0.0, |b| b[c])
        } else {
            0.0
        }
    }
}

/// `x` rounded to binary16 and back (round to nearest, ties to even — the `F2F.F16.F32` of the program).
pub fn round_f16(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

/// The reference on the f16-rounded inputs, in f64 (row-major `[m, n]`). `None` if [`Product::valid`] fails.
pub fn reference(p: &Product<'_>) -> Option<Vec<f64>> {
    if !p.valid() {
        return None;
    }
    let mut out = Vec::with_capacity(p.m * p.n);
    for r in 0..p.m {
        for c in 0..p.n {
            let mut s = p.bias_at(c) as f64;
            for q in 0..p.k {
                s += round_f16(p.op_a(r, q)) as f64 * round_f16(p.op_b(q, c)) as f64;
            }
            out.push(s);
        }
    }
    Some(out)
}

/// The written margin of element (r, c) against [`reference`] (module doc).
pub fn margin(p: &Product<'_>, r: usize, c: usize) -> f64 {
    let mut s = (p.bias_at(c) as f64).abs();
    for q in 0..p.k {
        s += (round_f16(p.op_a(r, q)) as f64 * round_f16(p.op_b(q, c)) as f64).abs();
    }
    (p.k as f64 + 3.0) * s / (1u64 << 23) as f64 + f64::from_bits((1023 - 126) << 52) // + 2⁻¹²⁶: the FTZ step
}

/// What rounding the inputs to f16 may add against the unrounded (f32) product, element (r, c) (module doc).
pub fn rounding_margin(p: &Product<'_>, r: usize, c: usize) -> f64 {
    let rel = 1.0 / 1024.0 + 1.0 / (1u64 << 22) as f64;
    let abs = 1.0 / (1u64 << 25) as f64;
    let mut s = 0.0f64;
    for q in 0..p.k {
        let (a, b) = (p.op_a(r, q).abs() as f64, p.op_b(q, c).abs() as f64);
        s += a * b * rel + abs * (a + b) + abs * abs;
    }
    s
}

/// The hypothesis of what the board computes (module doc), with the given tensor-core adder.
/// `None` if [`Product::valid`] fails.
pub fn model_with(p: &Product<'_>, adder: HwAdder) -> Option<Vec<f32>> {
    const T: usize = 16;
    if !p.valid() {
        return None;
    }
    let kpad = p.k.div_ceil(BLOCK_K) * BLOCK_K;
    let mut out = alloc::vec![0.0f32; p.m * p.n];
    let mut ha = alloc::vec![0u16; T * T];
    let mut hb = alloc::vec![0u16; T * T];
    // Each 16×16 tile of C is one accumulator of one subgroup; tiles past m / n hold zeros and are not stored.
    for r0 in (0..p.m).step_by(T) {
        for c0 in (0..p.n).step_by(T) {
            let mut acc = [0.0f32; T * T];
            for k0 in (0..kpad).step_by(T) {
                for i in 0..T {
                    for q in 0..T {
                        let (r, kq) = (r0 + i, k0 + q);
                        ha[i * T + q] = if r < p.m && kq < p.k {
                            f32_to_f16(p.op_a(r, kq))
                        } else {
                            0
                        };
                    }
                }
                for q in 0..T {
                    for j in 0..T {
                        let (kq, c) = (k0 + q, c0 + j);
                        hb[q * T + j] = if kq < p.k && c < p.n {
                            f32_to_f16(p.op_b(kq, c))
                        } else {
                            0
                        };
                    }
                }
                acc = mma::mma16_hw(&ha, &hb, &acc, adder)?;
            }
            for i in 0..T {
                for j in 0..T {
                    let (r, c) = (r0 + i, c0 + j);
                    if r < p.m && c < p.n {
                        let v = acc[i * T + j];
                        out[r * p.n + c] = if p.flags & BIAS != 0 {
                            add_f32_ftz(v, p.bias_at(c))
                        } else {
                            v
                        };
                    }
                }
            }
        }
    }
    Some(out)
}

/// [`model_with`] the adder fitted on the real board in os-335 ([`GA104_OS335`]). SIMULARE only.
pub fn model(p: &Product<'_>) -> Option<Vec<f32>> {
    model_with(p, GA104_OS335)
}

/// What [`judge`] says.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Verdict {
    /// Every element within [`margin`] of [`reference`]: the worst one in ‰ of its margin, and how many equal the
    /// reference rounded to f32 bit for bit.
    InMargine { worst_permille: u32, exact: usize },
    /// The first element (row-major index) outside its margin.
    NuTrece {
        index: usize,
        got: f32,
        reference: f64,
        margin: f64,
    },
    /// Invalid product, wrong length, or a non-finite result.
    Invalid,
}

fn permille(dist: f64, tol: f64) -> u32 {
    if dist == 0.0 {
        0
    } else if tol == 0.0 {
        u32::MAX
    } else {
        let x = dist / tol * 1000.0;
        if x >= u32::MAX as f64 {
            u32::MAX
        } else {
            let q = x as u32;
            if (q as f64) < x { q + 1 } else { q }
        }
    }
}

/// Judges `got` (`[m, n]`, row-major) against [`reference`] with [`margin`].
pub fn judge(p: &Product<'_>, got: &[f32]) -> Verdict {
    let Some(r) = reference(p) else {
        return Verdict::Invalid;
    };
    if got.len() != p.m * p.n || got.iter().any(|v| !v.is_finite()) {
        return Verdict::Invalid;
    }
    let (mut worst, mut exact) = (0u32, 0usize);
    for (idx, (&g, &want)) in got.iter().zip(&r).enumerate() {
        let tol = margin(p, idx / p.n, idx % p.n);
        let dist = (g as f64 - want).abs();
        if dist > tol {
            return Verdict::NuTrece {
                index: idx,
                got: g,
                reference: want,
                margin: tol,
            };
        }
        worst = worst.max(permille(dist, tol));
        if g.to_bits() == (want as f32).to_bits() {
            exact += 1;
        }
    }
    Verdict::InMargine {
        worst_permille: worst,
        exact,
    }
}

/// The worst element of `got` against an **f32 oracle** `want` (the unrounded product, e.g. `train::linear`), in ‰
/// of [`margin`] + [`rounding_margin`]. `None` for wrong lengths, an invalid product or a non-finite value.
pub fn worst_vs_f32(p: &Product<'_>, got: &[f32], want: &[f32]) -> Option<u32> {
    if !p.valid() || got.len() != p.m * p.n || want.len() != p.m * p.n {
        return None;
    }
    let mut worst = 0u32;
    for (idx, (&g, &w)) in got.iter().zip(want).enumerate() {
        if !g.is_finite() || !w.is_finite() {
            return None;
        }
        let (r, c) = (idx / p.n, idx % p.n);
        // The f32 oracle itself sits within K·2⁻²⁴·Σ|a·b| of the exact sum (Higham); counted in with margin().
        let tol = margin(p, r, c) + rounding_margin(p, r, c);
        worst = worst.max(permille((g as f64 - w as f64).abs(), tol));
    }
    Some(worst)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        }
    }

    #[test]
    fn model_is_in_margin_for_every_flag_combination() {
        let mut g = rng(7);
        for &(m, n, k) in &[
            (1, 1, 1),
            (16, 16, 16),
            (17, 33, 40),
            (70, 65, 31),
            (64, 64, 96),
        ] {
            let a: Vec<f32> = (0..m * k).map(|_| g()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| g()).collect();
            let bias: Vec<f32> = (0..n).map(|_| g()).collect();
            for flags in 0..8 {
                let p = Product {
                    a: &a,
                    b: &b,
                    bias: Some(&bias),
                    m,
                    n,
                    k,
                    flags,
                };
                let got = model(&p).unwrap();
                match judge(&p, &got) {
                    Verdict::InMargine { worst_permille, .. } => assert!(worst_permille <= 1000),
                    v => panic!("{m}x{n}x{k} flags {flags}: {v:?}"),
                }
            }
        }
    }

    #[test]
    fn unrounded_inputs_are_caught() {
        // A "program" that forgets the f16 rounding (exact f32 products) must fail the verdict.
        let mut g = rng(11);
        let (m, n, k) = (16, 16, 64);
        let a: Vec<f32> = (0..m * k).map(|_| g()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| g()).collect();
        let p = Product {
            a: &a,
            b: &b,
            bias: None,
            m,
            n,
            k,
            flags: 0,
        };
        let f32_sum: Vec<f32> = (0..m * n)
            .map(|i| {
                (0..k)
                    .map(|q| p.op_a(i / n, q) as f64 * p.op_b(q, i % n) as f64)
                    .sum::<f64>() as f32
            })
            .collect();
        assert!(matches!(judge(&p, &f32_sum), Verdict::NuTrece { .. }));
        // ... while against the f32 oracle the model stays inside margin + rounding margin.
        assert!(worst_vs_f32(&p, &model(&p).unwrap(), &f32_sum).unwrap() <= 1000);
    }

    #[test]
    fn out_of_range_and_bad_flags_are_refused() {
        let a = [70000.0f32];
        let b = [1.0f32];
        let mut p = Product {
            a: &a,
            b: &b,
            bias: None,
            m: 1,
            n: 1,
            k: 1,
            flags: 0,
        };
        assert!(reference(&p).is_none() && model(&p).is_none());
        let a = [1.0f32];
        p.a = &a;
        assert!(model(&p).is_some());
        p.flags = 8;
        assert!(model(&p).is_none());
        p.flags = BIAS; // bias flag without a bias
        assert!(model(&p).is_none());
    }

    #[test]
    fn rounding_to_f16_is_nearest_even() {
        assert_eq!(round_f16(1.0 + 1.0 / 2048.0), 1.0); // tie -> even
        assert_eq!(round_f16(1.0 + 3.0 / 2048.0), 1.0 + 2.0 / 1024.0);
        assert_eq!(round_f16(1.0e-8), 0.0); // below half the smallest subnormal
    }
}
