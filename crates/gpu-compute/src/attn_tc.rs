//! C-157 (`docs/C-157_ATTN_BACK_TC.md`, plan and predictions B1–B6 written before this code): the CPU model and the
//! verdict for `attn_back_tile_tc`, mode 1 of the attention backward (dQ = dS·K, dK = dSᵀ·Q, dV = Pᵀ·dO) on the tensor
//! cores — every element of dS, P, Q, K, dO rounded to binary16 before the products, the products added in f32
//! (f16 × f16 → f32), with the same arguments and the same causal mask as [`crate::layer::attn_back_col_gpu`].
//!
//! Kept apart on purpose (as in [`crate::mm_tc`]):
//! - [`model`]: the **hypothesis** of what the program computes, in its order: per head and per 64-row block,
//!   the 64 × 64 tiles below (and on) the diagonal in ascending order, each in 4 steps of 16
//!   ([`crate::mma::mma16_hw`] with [`crate::mma::GA104_OS335`]); on the diagonal tile the steps that hold only
//!   zeros for a 16-row piece are skipped. Never evidence of what the RTX 3070 Ti does (SIMULARE).
//! - [`head_products`]: the same three sums written as whole `mm_tc` products of one head (K = T, the mask as
//!   zeros), so [`crate::mm_tc::reference`] / [`crate::mm_tc::margin`] / [`crate::mm_tc::judge`] judge them with the
//!   margin written before os-335.
//!
//! The mixed #513 step already carries the error scale (2¹⁴) in dO and dS, so no scale is applied here.

use alloc::vec::Vec;

use crate::layer::AttnShape;
use crate::mm_tc::{self, F16_MAX, Product, TA};
use crate::mma::{self, GA104_OS335, HwAdder, f32_to_f16};

/// The tile edge of the program (rows of a block, columns of a head, K per tile).
pub const TILE: usize = 64;
/// One tensor-core step.
const STEP: usize = 16;

/// Which of the three sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sum {
    /// dQ[t, e] = Σ_{j ≤ t} dS[t, j]·K[j, e]
    Dq,
    /// dK[t, e] = Σ_{i ≥ t} dS[i, t]·Q[i, e]
    Dk,
    /// dV[t, e] = Σ_{i ≥ t} P[i, t]·dO[i, e]
    Dv,
}

/// The three sums in the order the program does them (and their column block in dqkv: 0, C, 2C).
pub const SUMS: [Sum; 3] = [Sum::Dq, Sum::Dk, Sum::Dv];

fn h16(x: f32) -> Option<u16> {
    (x.is_finite() && x.abs() <= F16_MAX).then(|| f32_to_f16(x))
}

/// What the model may change (to check that planted mistakes are caught; [`Variant::PROGRAM`] is the plan).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    /// Skip the all-zero steps of the diagonal tile (the plan: yes).
    pub skip_zero_steps: bool,
    /// MISTAKE: read dS[t, i] instead of dS[i, t] in dK (row order instead of the transposed load).
    pub dk_untransposed: bool,
    /// MISTAKE: P instead of dS in dK.
    pub dk_uses_p: bool,
    /// MISTAKE: no causal mask on the diagonal tile (reads above the diagonal).
    pub no_diag_mask: bool,
    /// MISTAKE: the last step of 16 of every tile forgotten.
    pub drop_last_step: bool,
}

impl Variant {
    /// The planned program.
    pub const PROGRAM: Variant = Variant {
        skip_zero_steps: true,
        dk_untransposed: false,
        dk_uses_p: false,
        no_diag_mask: false,
        drop_last_step: false,
    };
}

/// [`model`] with a chosen adder and variant. Returns dqkv `[B·T, 3C]`: the dQ, dK, dV columns of every head are
/// written, nothing else (zeros). `None` for a bad shape, d > 64, wrong slice lengths, or a value the program would
/// read that is not a finite binary16 (it would become ±inf / NaN on the board).
#[allow(clippy::too_many_arguments)]
pub fn model_with(
    s: &AttnShape,
    qkv: &[f32],
    p: &[f32],
    dout: &[f32],
    ds: &[f32],
    adder: HwAdder,
    v: Variant,
) -> Option<Vec<f32>> {
    let (nq, no, np) = s.sizes()?;
    let (t, c, d) = (s.t, s.c, s.d());
    if d == 0 || d > TILE || qkv.len() != nq || dout.len() != no || p.len() != np || ds.len() != np
    {
        return None;
    }
    let ntiles = t.div_ceil(TILE);
    let mut out = alloc::vec![0.0f32; nq];
    let mut ha = [0u16; STEP * STEP];
    let mut hb = [0u16; STEP * STEP];
    let steps = if v.drop_last_step {
        TILE / STEP - 1
    } else {
        TILE / STEP
    };
    for b in 0..s.batch {
        for h in 0..s.heads {
            let base = (b * s.heads + h) * t * t;
            for bx in 0..ntiles {
                let t0 = bx * TILE;
                for (si, sum) in SUMS.iter().enumerate() {
                    // the 16 accumulators of the block: rows r0 (block-local), columns c0
                    for r0 in (0..TILE).step_by(STEP) {
                        for c0 in (0..TILE).step_by(STEP) {
                            let mut acc = [0.0f32; STEP * STEP];
                            let tiles: Vec<usize> = match sum {
                                Sum::Dq => (0..=bx).collect(),
                                _ => (bx..ntiles).collect(),
                            };
                            for &kt in &tiles {
                                for st in 0..steps {
                                    let kk = st * STEP;
                                    if v.skip_zero_steps && kt == bx {
                                        let zero = match sum {
                                            Sum::Dq => kk >= r0 + STEP,
                                            _ => kk + STEP <= r0,
                                        };
                                        if zero {
                                            continue;
                                        }
                                    }
                                    for i in 0..STEP {
                                        let tp = t0 + r0 + i;
                                        for q in 0..STEP {
                                            let kq = kt * TILE + kk + q;
                                            ha[i * STEP + q] = match sum {
                                                Sum::Dq => {
                                                    if tp < t
                                                        && kq < t
                                                        && (kq <= tp || v.no_diag_mask)
                                                    {
                                                        h16(ds[base + tp * t + kq])?
                                                    } else {
                                                        0
                                                    }
                                                }
                                                Sum::Dk | Sum::Dv => {
                                                    if kq < t
                                                        && tp < t
                                                        && (tp <= kq || v.no_diag_mask)
                                                    {
                                                        let src = if *sum == Sum::Dv || v.dk_uses_p
                                                        {
                                                            p
                                                        } else {
                                                            ds
                                                        };
                                                        let at = if *sum == Sum::Dk
                                                            && v.dk_untransposed
                                                        {
                                                            tp * t + kq
                                                        } else {
                                                            kq * t + tp
                                                        };
                                                        h16(src[base + at])?
                                                    } else {
                                                        0
                                                    }
                                                }
                                            };
                                        }
                                    }
                                    for q in 0..STEP {
                                        let kq = kt * TILE + kk + q;
                                        for j in 0..STEP {
                                            let e = c0 + j;
                                            hb[q * STEP + j] = if kq < t && e < d {
                                                let x = match sum {
                                                    Sum::Dq => {
                                                        qkv[(b * t + kq) * 3 * c + c + h * d + e]
                                                    }
                                                    Sum::Dk => {
                                                        qkv[(b * t + kq) * 3 * c + h * d + e]
                                                    }
                                                    Sum::Dv => dout[(b * t + kq) * c + h * d + e],
                                                };
                                                h16(x)?
                                            } else {
                                                0
                                            };
                                        }
                                    }
                                    acc = mma::mma16_hw(&ha, &hb, &acc, adder)?;
                                }
                            }
                            for i in 0..STEP {
                                let tp = t0 + r0 + i;
                                for j in 0..STEP {
                                    let e = c0 + j;
                                    if tp < t && e < d {
                                        out[(b * t + tp) * 3 * c + si * c + h * d + e] =
                                            acc[i * STEP + j];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Some(out)
}

/// The planned program with the adder fitted on the real board in os-335. SIMULARE only.
pub fn model(s: &AttnShape, qkv: &[f32], p: &[f32], dout: &[f32], ds: &[f32]) -> Option<Vec<f32>> {
    model_with(s, qkv, p, dout, ds, GA104_OS335, Variant::PROGRAM)
}

/// The three sums of head (b, h) as whole `mm_tc` products (owned storage; see [`HeadProducts::product`]).
pub struct HeadProducts {
    t: usize,
    d: usize,
    /// dS with zeros above the diagonal, [T, T] (A of dQ; A of dK stored transposed).
    ds_masked: Vec<f32>,
    /// P with zeros above the diagonal, [T, T] (A of dV, stored transposed).
    p_masked: Vec<f32>,
    k: Vec<f32>,
    q: Vec<f32>,
    dout: Vec<f32>,
}

impl HeadProducts {
    /// The product of one sum: `[T, d]` = op(A)[T, T]·B[T, d].
    pub fn product(&self, sum: Sum) -> Product<'_> {
        let (a, b, flags) = match sum {
            Sum::Dq => (&self.ds_masked, &self.k, 0),
            Sum::Dk => (&self.ds_masked, &self.q, TA),
            Sum::Dv => (&self.p_masked, &self.dout, TA),
        };
        Product {
            a,
            b,
            bias: None,
            m: self.t,
            n: self.d,
            k: self.t,
            flags,
        }
    }
}

/// Builds [`HeadProducts`] for head (b, h). `None` for a bad shape or wrong lengths.
#[allow(clippy::too_many_arguments)]
pub fn head_products(
    s: &AttnShape,
    qkv: &[f32],
    p: &[f32],
    dout: &[f32],
    ds: &[f32],
    b: usize,
    h: usize,
) -> Option<HeadProducts> {
    let (nq, no, np) = s.sizes()?;
    let (t, c, d) = (s.t, s.c, s.d());
    if b >= s.batch
        || h >= s.heads
        || qkv.len() != nq
        || dout.len() != no
        || p.len() != np
        || ds.len() != np
    {
        return None;
    }
    let base = (b * s.heads + h) * t * t;
    let mask = |m: &[f32]| -> Vec<f32> {
        (0..t * t)
            .map(|x| if x % t <= x / t { m[base + x] } else { 0.0 })
            .collect()
    };
    let col = |off: usize, stride: usize, src: &[f32]| -> Vec<f32> {
        let mut v = Vec::with_capacity(t * d);
        for i in 0..t {
            v.extend_from_slice(&src[(b * t + i) * stride + off + h * d..][..d]);
        }
        v
    };
    Some(HeadProducts {
        t,
        d,
        ds_masked: mask(ds),
        p_masked: mask(p),
        k: col(c, 3 * c, qkv),
        q: col(0, 3 * c, qkv),
        dout: col(0, c, dout),
    })
}

/// The values of one sum of head (b, h) taken out of a dqkv `[B·T, 3C]`, row-major `[T, d]`.
pub fn head_values(s: &AttnShape, dqkv: &[f32], sum: Sum, b: usize, h: usize) -> Vec<f32> {
    let (t, c, d) = (s.t, s.c, s.d());
    let si = SUMS.iter().position(|x| *x == sum).unwrap_or(0);
    let mut v = Vec::with_capacity(t * d);
    for i in 0..t {
        v.extend_from_slice(&dqkv[(b * t + i) * 3 * c + si * c + h * d..][..d]);
    }
    v
}

/// Judges the dqkv of the program (or of a model) on every head and sum with [`mm_tc::judge`]; returns the worst
/// verdict (the first `NuTrece` / `Invalid` found, else the largest ‰ and the total of exact elements).
pub fn judge(
    s: &AttnShape,
    qkv: &[f32],
    p: &[f32],
    dout: &[f32],
    ds: &[f32],
    got: &[f32],
) -> mm_tc::Verdict {
    let mut worst = 0u32;
    let mut exact = 0usize;
    for b in 0..s.batch {
        for h in 0..s.heads {
            let Some(hp) = head_products(s, qkv, p, dout, ds, b, h) else {
                return mm_tc::Verdict::Invalid;
            };
            for sum in SUMS {
                match mm_tc::judge(&hp.product(sum), &head_values(s, got, sum, b, h)) {
                    mm_tc::Verdict::InMargine {
                        worst_permille,
                        exact: e,
                    } => {
                        worst = worst.max(worst_permille);
                        exact += e;
                    }
                    bad => return bad,
                }
            }
        }
    }
    mm_tc::Verdict::InMargine {
        worst_permille: worst,
        exact,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_wide_heads_and_wrong_lengths() {
        let s = AttnShape {
            batch: 1,
            t: 8,
            c: 128,
            heads: 1,
        };
        let (nq, no, np) = s.sizes().unwrap();
        let z = |n| alloc::vec![0.0f32; n];
        assert!(model(&s, &z(nq), &z(np), &z(no), &z(np)).is_none()); // d = 128 > 64
        let s = AttnShape { c: 64, ..s };
        let (nq, no, np) = s.sizes().unwrap();
        assert!(model(&s, &z(nq), &z(np), &z(no), &z(np - 1)).is_none());
        assert!(model(&s, &z(nq), &z(np), &z(no), &z(np)).is_some());
    }
}
