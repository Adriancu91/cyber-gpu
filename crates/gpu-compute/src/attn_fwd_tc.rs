//! C-168 point 4, step 2 (`docs/C-168_ATTN_FWD_TC_PLAN.md` §3 and §8, plan and predictions H1–H7 written before the
//! code): the CPU model of `tools/nak/attn_fwd_tc.comp` — the causal attention **forward on the tensor cores**, two
//! passes over the key tiles, P written once. What the program computes, in its order (the NAK 26.3.1 listing
//! `tools/nak/out2/attn_fwd_tc.sm86.txt`: every f32 operation `fadd.ftz` / `fmul.ftz` / `fmnmx.ftz`, no `ffma`):
//!
//! - Q, K, V and P rounded to binary16 (round to nearest even, with subnormals: `F2FP.PACK_AB`); the products
//!   S = Q·Kᵀ and O += P·V in steps of 16 by [`crate::mma::mma16_hw`] with the adder fitted on the board in os-335
//!   ([`crate::mma::GA104_OS335`], the `mm_tc` hypothesis). An element's sum does not depend on how many columns an
//!   instruction holds, so the program's 16 × 8 accumulators are modelled as 16 × 16 ones.
//! - Pass 1, per row r and per quarter q (the 16 columns 4q + 16m + i of every key tile, k = 4m + i in order):
//!   x = s·scale (masked to −f32::MAX above the diagonal), the tile maximum from −f32::MAX, mn = max(m, tile max),
//!   l = l·ex2((m − mn)·log2e), then l = l + (live ? ex2((x_k − mn)·log2e) : 0) for k = 0 … 15; m = mn.
//! - The 4 quarters joined in the order q = 0, 1, 2, 3: M = max, L = Σ l_q·ex2((m_q − M)·log2e) from 0, inv = 1 / L.
//! - Pass 2: the same S (the same code, the same bits), P = ex2((x − M)·log2e)·inv where live, exact 0 above the
//!   diagonal, written in f32 for every key tile kt ≤ rt and in f16 as the A operand of O += P·V; on the diagonal tile
//!   the steps of 16 past the last row of a 16-row piece are skipped (they hold only zeros), as in the program.
//!
//! `MUFU.EX2` and `MUFU.RCP` are modelled correctly rounded ([`crate::layer::ex2_ftz`], `ftz(1 / L)`); the board's
//! are approximate (~2 ulp), so the verdict on the board is a margin against `attn_fwd_tile3`, not bits.
//! **SIMULARE only:** nothing here is evidence of what the RTX 3070 Ti does.

use alloc::vec::Vec;

use crate::layer::{AttnShape, ex2_ftz};
use crate::math::{add_f32_ftz, ftz};
use crate::mm_tc::F16_MAX;
use crate::mma::{self, GA104_OS335, f16_to_f32, f32_to_f16};

/// The tile edge (rows of a row tile, keys of a key tile, the head width).
pub const TILE: usize = 64;
/// One tensor-core step.
const STEP: usize = 16;
/// log2(e) as the listing encodes it (`0x3fb8aa3b`).
pub const LOG2E: u32 = 0x3fb8_aa3b;
/// The program's "minus infinity" (`NEG` = −3.402823466e38 = `0xff7fffff`).
pub const NEG: f32 = -f32::MAX;

/// What the model may change (planted mistakes, to show the comparisons are not blind; [`FwdVariant::PROGRAM`] is
/// the program).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FwdVariant {
    /// MISTAKE: the 4 quarters of a row joined in the order 3, 2, 1, 0.
    pub join_reversed: bool,
    /// MISTAKE: no causal mask in pass 2 (P above the diagonal of the diagonal tile not zeroed).
    pub no_mask_pass2: bool,
    /// MISTAKE: P not rounded to f16 before P·V (the f32 value's f16 neighbour below, truncation).
    pub p_truncated: bool,
}

impl FwdVariant {
    /// The program.
    pub const PROGRAM: FwdVariant = FwdVariant {
        join_reversed: false,
        no_mask_pass2: false,
        p_truncated: false,
    };
}

/// What one row tile of one head gives: `p` = 64 rows × (rt + 1)·64 columns (the keys of tiles 0 … rt, row-major)
/// as written into the P buffer, `att` = 64 rows × 64 (the head's columns of att).
#[derive(Clone, Debug, PartialEq)]
pub struct FwdTile {
    pub p: Vec<f32>,
    pub att: Vec<f32>,
}

fn mul_ftz(a: f32, b: f32) -> f32 {
    ftz(ftz(a) * ftz(b))
}

/// `MUFU.EX2`(`fmul.ftz`(`fadd.ftz`(a, −b), log2e)).
fn ex2_diff(a: f32, b: f32) -> f32 {
    ex2_ftz(mul_ftz(add_f32_ftz(a, -b), f32::from_bits(LOG2E)))
}

/// `fmnmx.ftz` (max): both inputs flushed.
fn max_ftz(a: f32, b: f32) -> f32 {
    ftz(a).max(ftz(b))
}

fn h16(x: f32) -> Option<u16> {
    (x.is_finite() && x.abs() <= F16_MAX).then(|| f32_to_f16(x))
}

/// The f16 bits of a 64 × 64 tile of qkv: rows `row0 ..`, columns `col0 ..` (row-major). `None` if a value is not a
/// finite binary16 (on the board it would become inf / NaN in the products).
fn tile16(s: &AttnShape, qkv: &[f32], row0: usize, col0: usize) -> Option<Vec<u16>> {
    let ld = 3 * s.c;
    let mut out = alloc::vec![0u16; TILE * TILE];
    for r in 0..TILE {
        for e in 0..TILE {
            out[r * TILE + e] = h16(qkv[(row0 + r) * ld + col0 + e])?;
        }
    }
    Some(out)
}

/// S = Q·Kᵀ of one key tile (f32, 64 × 64), in the program's steps; on the diagonal tile the 16 × 32 pieces a warp
/// skips (columns wc ≥ wr + 16, all above the diagonal) stay 0.
fn s_tile(q16: &[u16], k16: &[u16], diag: bool) -> Option<Vec<f32>> {
    let mut s = alloc::vec![0.0f32; TILE * TILE];
    let mut ha = [0u16; STEP * STEP];
    let mut hb = [0u16; STEP * STEP];
    for r0 in (0..TILE).step_by(STEP) {
        for c0 in (0..TILE).step_by(STEP) {
            let wc = c0 / 32 * 32;
            if diag && wc >= r0 + STEP {
                continue;
            }
            let mut acc = [0.0f32; STEP * STEP];
            for ks in (0..TILE).step_by(STEP) {
                for i in 0..STEP {
                    for q in 0..STEP {
                        ha[i * STEP + q] = q16[(r0 + i) * TILE + ks + q];
                        hb[q * STEP + i] = k16[(c0 + i) * TILE + ks + q];
                    }
                }
                acc = mma::mma16_hw(&ha, &hb, &acc, GA104_OS335)?;
            }
            for i in 0..STEP {
                s[(r0 + i) * TILE + c0..(r0 + i) * TILE + c0 + STEP]
                    .copy_from_slice(&acc[i * STEP..(i + 1) * STEP]);
            }
        }
    }
    Some(s)
}

/// Column of element k (0 … 15) of quarter q: 4q + 16(k / 4) + k % 4.
fn col_of(q: usize, k: usize) -> usize {
    4 * q + 16 * (k >> 2) + (k & 3)
}

/// Row tile `rt` of head (b, h): [`FwdTile`]. `None` for a shape the program does not take (d ≠ 64, T not a multiple
/// of 64, rt / b / h out of range, a wrong `qkv` length) or a Q / K / V value that is not a finite binary16.
pub fn fwd_tile(
    s: &AttnShape,
    qkv: &[f32],
    scale: f32,
    (b, h, rt): (usize, usize, usize),
) -> Option<FwdTile> {
    fwd_tile_with(s, qkv, scale, (b, h, rt), FwdVariant::PROGRAM)
}

/// [`fwd_tile`] with a chosen variant.
pub fn fwd_tile_with(
    s: &AttnShape,
    qkv: &[f32],
    scale: f32,
    (b, h, rt): (usize, usize, usize),
    v: FwdVariant,
) -> Option<FwdTile> {
    let (nq, _, _) = s.sizes()?;
    let (t, c) = (s.t, s.c);
    if s.d() != TILE
        || !t.is_multiple_of(TILE)
        || qkv.len() != nq
        || b >= s.batch
        || h >= s.heads
        || rt >= t / TILE
    {
        return None;
    }
    let brow = b * t;
    let i0 = rt * TILE;
    let q16 = tile16(s, qkv, brow + i0, h * TILE)?;
    // ---- pass 1: per row r and quarter q, the running maximum and sum
    let mut mrun = [[NEG; 4]; TILE];
    let mut lrun = [[0.0f32; 4]; TILE];
    let mut s_tiles: Vec<Vec<f32>> = Vec::with_capacity(rt + 1);
    for kt in 0..=rt {
        let k16 = tile16(s, qkv, brow + kt * TILE, c + h * TILE)?;
        let diag = kt == rt;
        let st = s_tile(&q16, &k16, diag)?;
        for r in 0..TILE {
            let lim = if diag { r } else { TILE - 1 };
            for q in 0..4 {
                let mut xs = [0.0f32; 16];
                let mut tmax = NEG;
                for (k, xk) in xs.iter_mut().enumerate() {
                    let col = col_of(q, k);
                    *xk = if col <= lim {
                        mul_ftz(st[r * TILE + col], scale)
                    } else {
                        NEG
                    };
                    tmax = max_ftz(tmax, *xk);
                }
                let mn = max_ftz(mrun[r][q], tmax);
                let mut l = mul_ftz(lrun[r][q], ex2_diff(mrun[r][q], mn));
                for (k, &xk) in xs.iter().enumerate() {
                    let ek = ex2_diff(xk, mn);
                    l = add_f32_ftz(l, if col_of(q, k) <= lim { ek } else { 0.0 });
                }
                lrun[r][q] = l;
                mrun[r][q] = mn;
            }
        }
        s_tiles.push(st);
    }
    // ---- the 4 quarters of each row joined in a fixed order
    let mut big_m = [0.0f32; TILE];
    let mut inv = [0.0f32; TILE];
    for r in 0..TILE {
        let mq = mrun[r];
        let m = max_ftz(max_ftz(mq[0], mq[1]), max_ftz(mq[2], mq[3]));
        let mut l = 0.0f32;
        for k in 0..4 {
            let k = if v.join_reversed { 3 - k } else { k };
            l = add_f32_ftz(l, mul_ftz(lrun[r][k], ex2_diff(mq[k], m)));
        }
        big_m[r] = m;
        inv[r] = ftz(1.0 / l);
    }
    // ---- pass 2: P (f32 out, f16 into O += P·V)
    let wcols = (rt + 1) * TILE;
    let mut p = alloc::vec![0.0f32; TILE * wcols];
    let mut acco = alloc::vec![[0.0f32; STEP * STEP]; (TILE / STEP) * (TILE / STEP)];
    let mut ha = [0u16; STEP * STEP];
    let mut hb = [0u16; STEP * STEP];
    for (kt, st) in s_tiles.iter().enumerate() {
        let diag = kt == rt;
        let mut p16 = alloc::vec![0u16; TILE * TILE];
        for r in 0..TILE {
            let lim = if diag && !v.no_mask_pass2 {
                r
            } else {
                TILE - 1
            };
            for col in 0..TILE {
                let x = mul_ftz(st[r * TILE + col], scale);
                let pe = mul_ftz(ex2_diff(x, big_m[r]), inv[r]);
                let pv = if col <= lim { pe } else { 0.0 };
                p[r * wcols + kt * TILE + col] = pv;
                let hv = f32_to_f16(pv);
                p16[r * TILE + col] = if v.p_truncated && f16_to_f32(hv) > pv && hv > 0 {
                    hv - 1
                } else {
                    hv
                };
            }
        }
        let v16 = tile16(s, qkv, brow + kt * TILE, 2 * c + h * TILE)?;
        for r0 in (0..TILE).step_by(STEP) {
            let k1 = if diag { r0 + STEP } else { TILE };
            for c0 in (0..TILE).step_by(STEP) {
                let acc = &mut acco[(r0 / STEP) * (TILE / STEP) + c0 / STEP];
                for kk in (0..k1).step_by(STEP) {
                    for i in 0..STEP {
                        for q in 0..STEP {
                            ha[i * STEP + q] = p16[(r0 + i) * TILE + kk + q];
                            hb[q * STEP + i] = v16[(kk + q) * TILE + c0 + i];
                        }
                    }
                    *acc = mma::mma16_hw(&ha, &hb, acc, GA104_OS335)?;
                }
            }
        }
    }
    let mut att = alloc::vec![0.0f32; TILE * TILE];
    for r0 in (0..TILE).step_by(STEP) {
        for c0 in (0..TILE).step_by(STEP) {
            let acc = &acco[(r0 / STEP) * (TILE / STEP) + c0 / STEP];
            for i in 0..STEP {
                att[(r0 + i) * TILE + c0..(r0 + i) * TILE + c0 + STEP]
                    .copy_from_slice(&acc[i * STEP..(i + 1) * STEP]);
            }
        }
    }
    Some(FwdTile { p, att })
}

/// The whole launch on the CPU (every b, h and row tile): (P [B, H, T, T], att [B·T, C]). The tiles of P **above**
/// the diagonal tile are not written by the program; here they are left as `fill` (pass NaN to check nobody reads
/// them, 0 to compare with `attn_fwd_tile3`, which writes 0 there). `None` as [`fwd_tile`].
pub fn fwd_model(
    s: &AttnShape,
    qkv: &[f32],
    scale: f32,
    fill: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    fwd_model_with(s, qkv, scale, fill, FwdVariant::PROGRAM)
}

/// [`fwd_model`] with a chosen variant.
pub fn fwd_model_with(
    s: &AttnShape,
    qkv: &[f32],
    scale: f32,
    fill: f32,
    v: FwdVariant,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let (_, no, np) = s.sizes()?;
    let (t, c) = (s.t, s.c);
    if s.d() != TILE || !t.is_multiple_of(TILE) {
        return None;
    }
    let mut p = alloc::vec![fill; np];
    let mut att = alloc::vec![0.0f32; no];
    for b in 0..s.batch {
        for h in 0..s.heads {
            for rt in 0..t / TILE {
                let tile = fwd_tile_with(s, qkv, scale, (b, h, rt), v)?;
                let wcols = (rt + 1) * TILE;
                for r in 0..TILE {
                    let i = rt * TILE + r;
                    let at = ((b * s.heads + h) * t + i) * t;
                    p[at..at + wcols].copy_from_slice(&tile.p[r * wcols..(r + 1) * wcols]);
                    let o = (b * t + i) * c + h * TILE;
                    att[o..o + TILE].copy_from_slice(&tile.att[r * TILE..(r + 1) * TILE]);
                }
            }
        }
    }
    Some((p, att))
}

/// ‖got − want‖ / ‖want‖ (f64), over the indices where `keep` is true; `None` for different lengths, a non-finite
/// value kept, or a zero norm.
pub fn rel_norm(got: &[f32], want: &[f32], keep: impl Fn(usize) -> bool) -> Option<f64> {
    if got.len() != want.len() {
        return None;
    }
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if !keep(i) {
            continue;
        }
        if !g.is_finite() || !w.is_finite() {
            return None;
        }
        let d = g as f64 - w as f64;
        num += d * d;
        den += w as f64 * w as f64;
    }
    (den > 0.0).then(|| crate::train::sqrt64(num / den))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rng(seed: u64) -> impl FnMut() -> f32 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        }
    }

    fn shape(batch: usize, t: usize, heads: usize) -> AttnShape {
        AttnShape {
            batch,
            t,
            c: 64 * heads,
            heads,
        }
    }

    #[test]
    fn refuses_shapes_the_program_does_not_take() {
        let s = AttnShape {
            batch: 1,
            t: 64,
            c: 96,
            heads: 3,
        };
        let qkv = alloc::vec![0.0f32; s.sizes().unwrap().0];
        assert!(fwd_model(&s, &qkv, 0.125, 0.0).is_none());
        let s = shape(1, 96, 1);
        let qkv = alloc::vec![0.0f32; s.sizes().unwrap().0];
        assert!(fwd_model(&s, &qkv, 0.125, 0.0).is_none());
        let s = shape(1, 64, 1);
        let mut qkv = alloc::vec![0.0f32; s.sizes().unwrap().0];
        assert!(fwd_tile(&s, &qkv, 0.125, (0, 0, 1)).is_none());
        qkv[3] = 1.0e5; // not a finite binary16
        assert!(fwd_model(&s, &qkv, 0.125, 0.0).is_none());
    }

    /// Rows of P sum to ~1, P is exactly 0 above the diagonal, and row 0 of every head is (1, 0, …) with att = V row 0
    /// rounded to f16.
    #[test]
    fn rows_are_a_softmax_and_row_zero_copies_v() {
        let s = shape(2, 128, 2);
        let mut r = rng(5);
        let qkv: Vec<f32> = (0..s.sizes().unwrap().0).map(|_| r()).collect();
        let (p, att) = fwd_model(&s, &qkv, 0.125, f32::NAN).unwrap();
        for bh in 0..4 {
            for i in 0..128 {
                let row = &p[(bh * 128 + i) * 128..(bh * 128 + i + 1) * 128];
                let sum: f64 = row[..=i].iter().map(|&x| x as f64).sum();
                assert!((sum - 1.0).abs() < 1e-5, "{bh} {i} {sum}");
                let tile_end = (i / 64 + 1) * 64;
                assert!(row[i + 1..tile_end].iter().all(|&x| x == 0.0));
                assert!(row[tile_end..].iter().all(|x| x.is_nan()));
            }
        }
        for b in 0..2 {
            for h in 0..2 {
                for e in 0..64 {
                    let v = qkv[(b * 128) * 384 + 256 + h * 64 + e];
                    assert_eq!(att[(b * 128) * 128 + h * 64 + e], f16_to_f32(f32_to_f16(v)));
                }
            }
        }
    }

    #[test]
    fn rel_norm_basics() {
        assert_eq!(rel_norm(&[1.0, 2.0], &[1.0, 2.0], |_| true), Some(0.0));
        let e = rel_norm(&[3.0, 4.0], &[0.0, 5.0], |_| true).unwrap();
        assert!((e - 0.4f64.sqrt()).abs() < 1e-12);
        assert_eq!(
            rel_norm(&[f32::NAN, 1.0], &[0.0, 1.0], |i| i == 1),
            Some(0.0)
        );
        assert!(rel_norm(&[1.0], &[0.0], |_| true).is_none());
    }
}
