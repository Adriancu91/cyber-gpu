//! C-83 I13 — one layer of Kronos (`kronos_gpt`, hub `integrare/kronos/format_greutati.md`) for ONE new position,
//! with the key/value caches of the positions before it — what a decoding step does per layer:
//!
//! ```text
//! h = LayerNorm(x; ln1)            qkv = Wqkv·h + bqkv          (Q8_0, rows q | k | v)
//! k, v → caches[pos]               a = attention(q, caches[0..=pos]) per head, scale 1/√D
//! x = x + (Wout·a + bout)          h = LayerNorm(x; ln2)
//! f = GELU_erf(Wfc·h + bfc)        x = x + (Wproj·f + bproj)
//! ```
//!
//! Two versions:
//! - [`step_cpu`]: the CPU engine's plain order (sequential sums, `1/√(var+eps)`, the softmax of
//!   `reference::softmax_rows`), the reference the GPU is compared with;
//! - the `*_gpu` functions: what each of our GPU programs (`tools/nak/{vadd,layernorm,attention,gelu}.comp`)
//!   computes, step by step in its order. They are the **simulation models** of those programs. They are not
//!   bit-exact oracles: the programs use the GPU's approximate `MUFU.RSQ` / `MUFU.EX2` / `MUFU.RCP` and NAK fuses
//!   some operations, so the layer is judged with [`first_outside`] and the tolerance [`TOL_ABS`] / [`TOL_REL`],
//!   fixed here before any run on the board.
//!
//! Nothing here runs on a GPU.

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::math::{self, add_f32_ftz, fma_f32_ftz, ftz};
use crate::q8;

/// Tolerance of a whole layer, per element: |GPU − CPU| ≤ `TOL_ABS + TOL_REL·|CPU|`. Why this size: the
/// approximate units are good to ~2⁻²² relative; LayerNorm multiplies errors by 1/σ (~10 here), the matrix
/// products add 768–3072 terms; that gives ~1e-5 at worst, and 1e-4 leaves a factor 10. For the logits the
/// hub's MODEL.md says deviations of ~1e-3 keep the greedy tokens; a layer at 1e-4 is well inside that.
pub const TOL_ABS: f32 = 1e-4;
pub const TOL_REL: f32 = 1e-4;
/// Most positions the attention program holds (its shared array).
pub const MAX_POS: usize = 512;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The weights of one layer (`straturi.i.`), Q8_0 matrices as KGR1 bytes.
#[derive(Clone, Copy)]
pub struct LayerWeights<'a> {
    pub ln1_g: &'a [f32],
    pub ln1_b: &'a [f32],
    /// [3E, E]
    pub wqkv: &'a [u8],
    pub bqkv: &'a [f32],
    /// [E, E]
    pub wout: &'a [u8],
    pub bout: &'a [f32],
    pub ln2_g: &'a [f32],
    pub ln2_b: &'a [f32],
    /// [4E, E]
    pub wfc: &'a [u8],
    pub bfc: &'a [f32],
    /// [E, 4E]
    pub wproj: &'a [u8],
    pub bproj: &'a [f32],
}

/// The shape: `e` = n_emb, `heads` (D = e / heads), `eps` = ln_eps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    pub e: usize,
    pub heads: usize,
    pub eps: f32,
}

impl Shape {
    pub fn d(&self) -> usize {
        self.e / self.heads
    }

    pub fn check(&self, w: &LayerWeights<'_>) -> GpuResult<()> {
        let e = self.e;
        if e == 0
            || self.heads == 0
            || !e.is_multiple_of(self.heads)
            || !e.is_multiple_of(q8::BLOCK)
        {
            return Err(bad("strat: forma invalida"));
        }
        let ok = [w.ln1_g, w.ln1_b, w.bout, w.ln2_g, w.ln2_b, w.bproj]
            .iter()
            .all(|v| v.len() == e)
            && w.bqkv.len() == 3 * e
            && w.bfc.len() == 4 * e
            && w.wqkv.len() == q8::matrix_bytes(3 * e, e)?
            && w.wout.len() == q8::matrix_bytes(e, e)?
            && w.wfc.len() == q8::matrix_bytes(4 * e, e)?
            && w.wproj.len() == q8::matrix_bytes(e, 4 * e)?;
        if !ok {
            return Err(bad("strat: marimi de greutati nepotrivite"));
        }
        Ok(())
    }
}

fn check_caches(s: &Shape, kc: &[f32], vc: &[f32], pos: usize) -> GpuResult<()> {
    if pos >= MAX_POS || kc.len() < (pos + 1) * s.e || vc.len() < (pos + 1) * s.e {
        return Err(bad("strat: pozitie peste cache"));
    }
    Ok(())
}

// ---------------------------------------------------------------- CPU engine order

pub(crate) fn layernorm_cpu(x: &[f32], g: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / math::sqrt(var + eps);
    x.iter()
        .zip(g)
        .zip(b)
        .map(|((v, g), b)| (v - mean) * inv * g + b)
        .collect()
}

fn matvec_bias_cpu(
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    b: &[f32],
) -> GpuResult<Vec<f32>> {
    let mut y = alloc::vec![0.0f32; rows];
    q8::matvec_cpu(w, rows, cols, x, &mut y)?;
    Ok(y.iter().zip(b).map(|(y, b)| y + b).collect())
}

/// One position through the layer in the CPU engine's order: `x` (E values) is replaced by the layer's output; the
/// new key and value are written into `kc`/`vc` at `pos` (`[position][E]`).
pub fn step_cpu(
    s: &Shape,
    w: &LayerWeights<'_>,
    x: &mut [f32],
    kc: &mut [f32],
    vc: &mut [f32],
    pos: usize,
) -> GpuResult<()> {
    s.check(w)?;
    check_caches(s, kc, vc, pos)?;
    let (e, d) = (s.e, s.d());
    if x.len() != e {
        return Err(bad("strat: x are alta marime"));
    }
    let h = layernorm_cpu(x, w.ln1_g, w.ln1_b, s.eps);
    let qkv = matvec_bias_cpu(w.wqkv, 3 * e, e, &h, w.bqkv)?;
    kc[pos * e..(pos + 1) * e].copy_from_slice(&qkv[e..2 * e]);
    vc[pos * e..(pos + 1) * e].copy_from_slice(&qkv[2 * e..]);
    let scale = 1.0 / math::sqrt(d as f32);
    let mut a = alloc::vec![0.0f32; e];
    let mut sc = alloc::vec![0.0f32; pos + 1];
    for hd in 0..s.heads {
        let q = &qkv[hd * d..(hd + 1) * d];
        for (t, v) in sc.iter_mut().enumerate() {
            let k = &kc[t * e + hd * d..t * e + (hd + 1) * d];
            *v = q.iter().zip(k).fold(0.0f32, |a, (q, k)| a + q * k) * scale;
        }
        crate::reference::softmax_rows(&mut sc, 1, pos + 1, false)?;
        for c in 0..d {
            a[hd * d + c] = (0..=pos).fold(0.0f32, |acc, t| acc + sc[t] * vc[t * e + hd * d + c]);
        }
    }
    let o = matvec_bias_cpu(w.wout, e, e, &a, w.bout)?;
    for (x, o) in x.iter_mut().zip(&o) {
        *x += o;
    }
    let h2 = layernorm_cpu(x, w.ln2_g, w.ln2_b, s.eps);
    let mut f = matvec_bias_cpu(w.wfc, 4 * e, e, &h2, w.bfc)?;
    crate::reference::gelu_erf(&mut f);
    let p = matvec_bias_cpu(w.wproj, e, 4 * e, &f, w.bproj)?;
    for (x, p) in x.iter_mut().zip(&p) {
        *x += p;
    }
    Ok(())
}

// ---------------------------------------------------------------- the GPU programs, as models

/// `vadd.comp`: c = a + b, `FADD.FTZ` (bit-exact model).
pub fn vadd_gpu(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(&a, &b)| add_f32_ftz(a, b)).collect()
}

pub(crate) fn mul_ftz(a: f32, b: f32) -> f32 {
    ftz(ftz(a) * ftz(b))
}

/// 32 partial sums, the fixed tree of our programs (stride 16..1, `FADD.FTZ`).
fn tree_sum(mut p: [f32; 32]) -> f32 {
    let mut st = 16;
    while st > 0 {
        for t in 0..st {
            p[t] = add_f32_ftz(p[t], p[t + st]);
        }
        st /= 2;
    }
    p[0]
}

/// `layernorm.comp`: thread t sums x[t], x[t+32], …; tree; mean = sum·inv_n; the same for (x − mean)²; then
/// `inversesqrt` (modelled as 1/√ correctly rounded — the board's `MUFU.RSQ` is approximate).
pub fn layernorm_gpu(x: &[f32], g: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len();
    let inv_n = 1.0 / n as f32;
    let mut p = [0.0f32; 32];
    for (t, pt) in p.iter_mut().enumerate() {
        *pt = x
            .iter()
            .skip(t)
            .step_by(32)
            .fold(0.0, |s, &v| add_f32_ftz(s, v));
    }
    let mean = mul_ftz(tree_sum(p), inv_n);
    for (t, pt) in p.iter_mut().enumerate() {
        *pt = x.iter().skip(t).step_by(32).fold(0.0, |s, &v| {
            let dd = add_f32_ftz(v, -mean);
            fma_f32_ftz(dd, dd, s)
        });
    }
    let inv = 1.0 / math::sqrt(fma_f32_ftz(tree_sum(p), inv_n, eps));
    x.iter()
        .zip(g)
        .zip(b)
        .map(|((&v, &g), &b)| fma_f32_ftz(mul_ftz(add_f32_ftz(v, -mean), inv), g, b))
        .collect()
}

/// The largest vector `layernorm256.comp` takes (256 threads × 4 values in registers).
pub const LN256_MAX_N: usize = 1024;

/// The block sum of `layernorm256.comp`: per warp a butterfly (lane ^ 16, 8, 4, 2, 1 — every lane ends with the same
/// bits, since a + b = b + a), then the 8 warp sums added in order w = 0..7.
fn block_sum256(p: &[f32; 256]) -> f32 {
    let mut warps = [0.0f32; 8];
    for (w, out) in warps.iter_mut().enumerate() {
        let mut v = [0.0f32; 32];
        v.copy_from_slice(&p[32 * w..32 * w + 32]);
        for m in [16usize, 8, 4, 2, 1] {
            let before = v;
            for (l, vl) in v.iter_mut().enumerate() {
                *vl = add_f32_ftz(before[l], before[l ^ m]);
            }
        }
        *out = v[0];
    }
    let mut r = warps[0];
    for &w in &warps[1..] {
        r = add_f32_ftz(r, w);
    }
    r
}

/// `layernorm256.comp` (C-155 point 2), in the order of the code NAK 25.0.7 made of it (`tools/nak/out/
/// layernorm256.sm86.txt`): 256 threads; thread t holds x[t], x[t+256], x[t+512], x[t+768]; its partial sum starts
/// with its first value as read (`sel`, no flush) and adds the rest with `fadd.ftz`; [`block_sum256`]. NAK never
/// rounds the mean: every deviation is one `ffma.ftz` d = (−sum)·inv_n + x. The variance: d₀² (`fmul.ftz`), then an
/// FMA chain, [`block_sum256`], `ffma.ftz` (var·inv_n + eps), `MUFU.RSQ` (modelled as 1/√ correctly rounded — the
/// board's unit is approximate, so the result is compared in a margin); y = (d·inv)·g + b (`fmul`, `ffma`).
/// `n` must be 1 ..= [`LN256_MAX_N`]. Another summation order than [`layernorm_gpu`]: the two differ in the last
/// bits; both are within the layer tolerance of [`step_cpu`].
pub fn layernorm_gpu256(x: &[f32], g: &[f32], b: &[f32], eps: f32) -> GpuResult<Vec<f32>> {
    let n = x.len();
    if n == 0 || n > LN256_MAX_N || g.len() != n || b.len() != n {
        return Err(GpuError::InvalidArgument(
            "layernorm256: n intre 1 si 1024, g si b de aceeasi lungime",
        ));
    }
    let inv_n = 1.0 / n as f32;
    let mut p = [0.0f32; 256];
    for (t, pt) in p.iter_mut().enumerate() {
        let mut it = x.iter().skip(t).step_by(256);
        *pt = match it.next() {
            Some(&first) => it.fold(first, |s, &v| add_f32_ftz(s, v)),
            None => 0.0,
        };
    }
    let neg_sum = ftz(-block_sum256(&p));
    let dev = |v: f32| fma_f32_ftz(neg_sum, inv_n, v);
    for (t, pt) in p.iter_mut().enumerate() {
        let mut it = x.iter().skip(t).step_by(256);
        *pt = match it.next() {
            Some(&first) => {
                let d0 = dev(first);
                it.fold(mul_ftz(d0, d0), |q, &v| {
                    let d = dev(v);
                    fma_f32_ftz(d, d, q)
                })
            }
            None => 0.0,
        };
    }
    let inv = 1.0 / math::sqrt(fma_f32_ftz(block_sum256(&p), inv_n, eps));
    Ok(x.iter()
        .zip(g)
        .zip(b)
        .map(|((&v, &g), &b)| fma_f32_ftz(mul_ftz(dev(v), inv), g, b))
        .collect())
}

/// The block maximum of `xent.comp`: per warp a butterfly of `max` (lane ^ 16, 8, 4, 2, 1), then the 8 warp maxima
/// in order. A maximum does not depend on the order (no NaN in the logits), so this is `max` over the 256 values.
fn block_max256(p: &[f32; 256]) -> f32 {
    p.iter().fold(-f32::MAX, |m, &v| m.max(v))
}

/// The row of the board's loss program, `xent.comp` (C-160 point 4, program 7), for one row of `v = z.len()`
/// logits: m = max (thread t over z[t], z[t+256], … from −f32::MAX, then [`block_max256`]); s = Σ e^(z−m) (thread t
/// adds its values in order from 0.0 with `fadd.ftz`, then [`block_sum256`]); inv = 1/s; d[k] = (e^(z[k]−m)·inv −
/// [k = tgt])·scale; loss = log₂s·ln 2 + m − z[tgt] (one `ffma.ftz`, then `fadd.ftz`; NaN when `tgt ≥ v`, as the
/// program writes it, and then no one-hot). In the order of the code NAK 25.0.7 made (`tools/nak/out/xent.sm86.txt`).
/// `exp`, `ln` and `1/s` are modelled correctly rounded (`math::exp`, f64 ln, f32 division): the board's
/// `MUFU.EX2` / `MUFU.LG2` / `MUFU.RCP` are approximate, so board and model are compared with the f64 training
/// oracle `train::softmax_xent` within a written margin, not bit for bit. Returns `(d, loss)`.
pub fn xent_gpu256(z: &[f32], tgt: u32, scale: f32) -> (Vec<f32>, f32) {
    let v = z.len();
    let e = |x: f32, m: f32| ftz(math::exp(add_f32_ftz(x, -m)));
    let mut p = [-f32::MAX; 256];
    for (t, pt) in p.iter_mut().enumerate() {
        *pt = z
            .iter()
            .skip(t)
            .step_by(256)
            .fold(-f32::MAX, |m, &x| m.max(x));
    }
    let m = block_max256(&p);
    let mut q = [0.0f32; 256];
    for (t, qt) in q.iter_mut().enumerate() {
        *qt = z
            .iter()
            .skip(t)
            .step_by(256)
            .fold(0.0, |s, &x| add_f32_ftz(s, e(x, m)));
    }
    let s = block_sum256(&q);
    let inv = ftz(1.0 / s);
    let tgt = tgt as usize;
    let d = z
        .iter()
        .enumerate()
        .map(|(k, &x)| {
            let p = mul_ftz(e(x, m), inv);
            let p = if k == tgt { add_f32_ftz(p, -1.0) } else { p };
            mul_ftz(p, scale)
        })
        .collect();
    let loss = if tgt < v {
        // NAK: lg2 = MUFU.LG2(s); ffma.ftz(lg2, ln 2 as f32, m); fadd.ftz(−z[tgt]).
        let lg2 = (crate::reference::ln(s as f64) / core::f64::consts::LN_2) as f32;
        add_f32_ftz(fma_f32_ftz(lg2, f32::from_bits(0x3f31_7218), m), -z[tgt])
    } else {
        f32::NAN
    };
    (d, loss)
}

/// The row of the board's LayerNorm-backward program, `ln_back.comp` (C-160 point 4, program 4, `dx` half), for one
/// row of `c = x.len()` numbers with the row's `mu` and `rs` (= 1/√(var + eps)) kept by the forward pass and
/// `inv_c` = 1/c as f32. In the order of the code NAK 25.0.7 made (`tools/nak/out/ln_back.sm86.txt`):
/// x̂ = `fmul.ftz`(`fadd.ftz`(x, −μ), rs); g = `fmul.ftz`(dy, γ); thread t over k = t, t+256, … from 0.0:
/// s1 = `fadd.ftz`(s1, g), s2 = `ffma.ftz`(g, x̂, s2); both through [`block_sum256`]; m1 = `fmul.ftz`(S1, inv_c),
/// m2 = `fmul.ftz`(S2, inv_c); in the last pass NAK fused g − m1 into one fma (g is not rounded there):
/// dx = `fmul.ftz`(rs, `ffma.ftz`(−x̂, m2, `ffma.ftz`(dy, γ, −m1))). Only add / mul / fma, no `MUFU`: this is meant to
/// be the board's result bit for bit (not proven on the board).
pub fn ln_back_gpu256(
    x: &[f32],
    dy: &[f32],
    gamma: &[f32],
    mu: f32,
    rs: f32,
    inv_c: f32,
) -> Vec<f32> {
    let c = x.len();
    let hat = |k: usize| mul_ftz(add_f32_ftz(x[k], -mu), rs);
    let grad = |k: usize| mul_ftz(dy[k], gamma[k]);
    let mut p1 = [0.0f32; 256];
    let mut p2 = [0.0f32; 256];
    for t in 0..256 {
        let (mut s1, mut s2) = (0.0f32, 0.0f32);
        for k in (t..c).step_by(256) {
            let g = grad(k);
            s1 = add_f32_ftz(s1, g);
            s2 = fma_f32_ftz(g, hat(k), s2);
        }
        p1[t] = s1;
        p2[t] = s2;
    }
    let m1 = mul_ftz(block_sum256(&p1), inv_c);
    let m2 = mul_ftz(block_sum256(&p2), inv_c);
    (0..c)
        .map(|k| {
            mul_ftz(
                rs,
                fma_f32_ftz(-hat(k), m2, fma_f32_ftz(dy[k], gamma[k], -m1)),
            )
        })
        .collect()
}

/// The row of the board's training LayerNorm-forward program, `ln_fwd.comp` (C-160 point 4, program 3), for one row
/// of `c = x.len()` numbers with `inv_c` = 1/c as f32 and ε. In the order of the code NAK 25.0.7 made
/// (`tools/nak/out/ln_fwd.sm86.txt`): thread t over k = t, t+256, … from 0.0: s = `fadd.ftz`(s, x); [`block_sum256`];
/// μ = `fmul.ftz`(S, inv_c) (`precise`: rounded once, the μ written is the μ subtracted); d = `fadd.ftz`(x, −μ);
/// q = `ffma.ftz`(d, d, q) from 0.0; [`block_sum256`]; rs = `MUFU.RSQ`(`ffma.ftz`(Q, inv_c, ε)) — modelled as 1/√
/// correctly rounded (the board's unit is approximate, so board and model are compared with the f64 oracle in a
/// written margin, not bit for bit); y = `ffma.ftz`(`fmul.ftz`(d, rs), γ, β). Returns `(y, μ, rs)`.
pub fn ln_fwd_gpu256(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    inv_c: f32,
    eps: f32,
) -> (Vec<f32>, f32, f32) {
    let c = x.len();
    let mut p = [0.0f32; 256];
    for (t, pt) in p.iter_mut().enumerate() {
        *pt = (t..c)
            .step_by(256)
            .fold(0.0f32, |s, k| add_f32_ftz(s, x[k]));
    }
    let mu = mul_ftz(block_sum256(&p), inv_c);
    let dev = |k: usize| add_f32_ftz(x[k], -mu);
    for (t, pt) in p.iter_mut().enumerate() {
        *pt = (t..c).step_by(256).fold(0.0f32, |q, k| {
            let d = dev(k);
            fma_f32_ftz(d, d, q)
        });
    }
    let rs = ftz(1.0 / math::sqrt(fma_f32_ftz(block_sum256(&p), inv_c, eps)));
    let y = (0..c)
        .map(|k| fma_f32_ftz(mul_ftz(dev(k), rs), gamma[k], beta[k]))
        .collect();
    (y, mu, rs)
}

/// The forward statistics the LayerNorm mode of [`colsum_gpu`] recomputes x̂ from: `x` `[rows, c]`, μ and rstd
/// `[rows]`.
#[derive(Clone, Copy, Debug)]
pub struct LnStats<'a> {
    pub x: &'a [f32],
    pub mean: &'a [f32],
    pub rstd: &'a [f32],
}

/// The board's column-sum program, `colsum.comp` (C-160 point 4, program 2 + dγ/dβ of program 4), over `rows × c`:
/// each column's thread adds its rows in order r = 0, 1, … from 0.0. Without `ln`: s0 = Σ dy (`fadd.ftz`), s1 empty.
/// With `ln`: x̂ = `fmul.ftz`(`fadd.ftz`(x, −μ_r), rstd_r), s0 = `ffma.ftz`(dy, x̂, s0) (dγ), s1 = `fadd.ftz`(s1, dy)
/// (dβ). In the order of the code NAK 25.0.7 made (`tools/nak/out/colsum.sm86.txt`); only add / mul / fma, meant to
/// be the board's bits (not proven on the board). Returns `(s0, s1)`.
pub fn colsum_gpu(
    dy: &[f32],
    rows: usize,
    c: usize,
    ln: Option<LnStats<'_>>,
) -> (Vec<f32>, Vec<f32>) {
    match ln {
        None => {
            let s0 = (0..c)
                .map(|k| (0..rows).fold(0.0f32, |s, r| add_f32_ftz(s, dy[r * c + k])))
                .collect();
            (s0, Vec::new())
        }
        Some(st) => {
            let mut s0 = alloc::vec![0.0f32; c];
            let mut s1 = alloc::vec![0.0f32; c];
            for k in 0..c {
                let (mut sg, mut sb) = (0.0f32, 0.0f32);
                for r in 0..rows {
                    let d = dy[r * c + k];
                    let h = mul_ftz(add_f32_ftz(st.x[r * c + k], -st.mean[r]), st.rstd[r]);
                    sg = fma_f32_ftz(d, h, sg);
                    sb = add_f32_ftz(sb, d);
                }
                s0[k] = sg;
                s1[k] = sb;
            }
            (s0, s1)
        }
    }
}

/// The scalar arguments of `attention.comp` (cbuf0 0x48..0x58).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttnArgs {
    pub n_pos: usize,
    pub e: usize,
    pub d: usize,
    pub scale: f32,
}

/// `attention.comp` for head `hd`: scores with an FMA chain, max tree, `exp` (modelled with `math::exp`; the board's
/// `MUFU.EX2` is approximate), sum tree, `1/sum`, then Σ p·v as an FMA chain, times 1/sum.
pub fn attention_head_gpu(q: &[f32], kc: &[f32], vc: &[f32], a: &AttnArgs, hd: usize) -> Vec<f32> {
    let AttnArgs { n_pos, e, d, scale } = *a;
    let base = hd * d;
    let mut sc = alloc::vec![0.0f32; n_pos];
    let mut pm = [-f32::MAX; 32];
    for s in 0..n_pos {
        let dot = (0..d).fold(0.0f32, |a, c| {
            fma_f32_ftz(q[base + c], kc[s * e + base + c], a)
        });
        sc[s] = mul_ftz(dot, scale);
        pm[s % 32] = pm[s % 32].max(sc[s]);
    }
    let mut st = 16;
    while st > 0 {
        for t in 0..st {
            pm[t] = pm[t].max(pm[t + st]);
        }
        st /= 2;
    }
    let m = pm[0];
    let mut ps = [0.0f32; 32];
    for s in 0..n_pos {
        sc[s] = ftz(math::exp(add_f32_ftz(sc[s], -m)));
        ps[s % 32] = add_f32_ftz(ps[s % 32], sc[s]);
    }
    let inv = 1.0 / tree_sum(ps);
    (0..d)
        .map(|c| {
            let acc = (0..n_pos).fold(0.0f32, |a, s| fma_f32_ftz(sc[s], vc[s * e + base + c], a));
            mul_ftz(acc, inv)
        })
        .collect()
}

/// `gelu.comp`: 0.5·x·(1 + erf(x/√2)), erf by A&S 7.1.26 in f32 (exp modelled with `math::exp`).
pub fn gelu_gpu(x: f32) -> f32 {
    let z = x * core::f32::consts::FRAC_1_SQRT_2;
    let a = z.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * a);
    let p = t
        * (0.254_829_6
            + t * (-0.284_496_74 + t * (1.421_413_8 + t * (-1.453_152 + t * 1.061_405_4))));
    let e = 1.0 - p * math::exp(-a * a);
    let erf = if z < 0.0 { -e } else { e };
    0.5 * x * (1.0 + erf)
}

/// 2^w as `MUFU.EX2.FTZ` would give it if it were exact: the input and the result flushed, the power correctly rounded
/// (f64 range reduction + Taylor). The board's unit is approximate (~2 ulp), so programs using it are compared with
/// an oracle in a margin, not bit for bit.
fn ex2_ftz(w: f32) -> f32 {
    let w = ftz(w);
    if w.is_nan() {
        return w;
    }
    if w >= 128.0 {
        return f32::INFINITY;
    }
    if w < -151.0 {
        return 0.0;
    }
    let wd = w as f64;
    let k = if wd >= 0.0 {
        (wd + 0.5) as i64
    } else {
        (wd - 0.5) as i64
    };
    let r = (wd - k as f64) * core::f64::consts::LN_2; // |r| ≤ ln2/2, w − k exact in f64
    let (mut term, mut sum) = (1.0f64, 1.0f64);
    for i in 1..=14 {
        term *= r / i as f64;
        sum += term;
    }
    let p = f64::from_bits(((1023 + k) as u64) << 52); // k in [-151, 127]: a normal f64
    ftz((sum * p) as f32)
}

/// The f32 constants of `gelu_tr.comp` as NAK 25.0.7 encoded them (`tools/nak/out/gelu_tr.sm86.txt`).
mod gelu_k {
    pub const RSQRT2: u32 = 0x3f35_04f3; // 1/√2
    pub const A: u32 = 0x3ea7_ba05; // 0.3275911
    pub const C5: u32 = 0x3f87_dc22; // 1.061405429
    pub const C4: u32 = 0xbfba_00e3; // −1.453152027
    pub const C3: u32 = 0x3fb5_f0e3; // 1.421413741
    pub const C2: u32 = 0xbe91_a98e; // −0.284496736
    pub const C1: u32 = 0x3e82_7906; // 0.254829592
    pub const LOG2E: u32 = 0x3fb8_aa3b; // log2(e): exp(x) = ex2(x·log2 e)
    pub const RSQRT2PI: u32 = 0x3ecc_422a; // 1/√(2π)
}

/// One element of the board's training GELU program, `gelu_tr.comp` (C-160 point 4, program 5), in the order of the
/// code NAK 25.0.7 made (`tools/nak/out/gelu_tr.sm86.txt`): z = `fmul.ftz`(u, 1/√2); a = |z|;
/// t = `MUFU.RCP`(`ffma.ftz`(a, 0.3275911, 1)); the A&S 7.1.26 polynomial by `ffma.ftz` (Horner from c5), p = `fmul.ftz`(t, ·);
/// ex = `MUFU.EX2`(−`fmul.ftz`(`fmul.ftz`(a, log2 e), a)) = e^(−a²); e = `ffma.ftz`(−p, ex, 1); erf = ±e by the sign of z;
/// Φ = `fmul.ftz`(`fadd.ftz`(erf, 1), 0.5). `g = None` is mode 0 (forward): `fmul.ftz`(u, Φ). `Some(g)` is mode 1
/// (backward): `fmul.ftz`(g, `ffma.ftz`(u, `fmul.ftz`(ex, 1/√(2π)), Φ)). `MUFU.RCP` / `MUFU.EX2` are modelled correctly
/// rounded (the board's are approximate): board and model are judged against the f64 training oracle
/// (`train::gelu_rows`, `train::gelu_back_rows`) in a written margin.
pub fn gelu_tr_gpu(u: f32, g: Option<f32>) -> f32 {
    use gelu_k::*;
    let k = f32::from_bits;
    let z = mul_ftz(u, k(RSQRT2));
    let a = ftz(z.abs());
    let t = ftz(1.0 / fma_f32_ftz(a, k(A), 1.0));
    let mut q = fma_f32_ftz(t, k(C5), k(C4));
    q = fma_f32_ftz(t, q, k(C3));
    q = fma_f32_ftz(t, q, k(C2));
    q = fma_f32_ftz(t, q, k(C1));
    let p = mul_ftz(t, q);
    let ex = ex2_ftz(-mul_ftz(mul_ftz(a, k(LOG2E)), a));
    let e = fma_f32_ftz(-p, ex, 1.0);
    let erf = if z < 0.0 { -e } else { e };
    let cdf = mul_ftz(add_f32_ftz(erf, 1.0), 0.5);
    match g {
        None => mul_ftz(u, cdf),
        Some(g) => mul_ftz(g, fma_f32_ftz(u, mul_ftz(ex, k(RSQRT2PI)), cdf)),
    }
}

/// The shape of one attention backward (C-160 point 4, program 6, `attn_back.comp`): B texts of T positions, C = H·d.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnShape {
    pub batch: usize,
    pub t: usize,
    pub c: usize,
    pub heads: usize,
}

impl AttnShape {
    /// Head width d = C / H.
    pub fn d(&self) -> usize {
        self.c / self.heads
    }

    /// Elements of qkv / dqkv ([B·T, 3C]), of dO ([B·T, C]) and of P / dS ([B, H, T, T]); `None` for a bad shape.
    pub fn sizes(&self) -> Option<(usize, usize, usize)> {
        if self.batch == 0
            || self.t == 0
            || self.heads == 0
            || self.c == 0
            || !self.c.is_multiple_of(self.heads)
        {
            return None;
        }
        let rows = self.batch.checked_mul(self.t)?;
        let qkv = rows.checked_mul(self.c)?.checked_mul(3)?;
        let p = self
            .batch
            .checked_mul(self.heads)?
            .checked_mul(self.t)?
            .checked_mul(self.t)?;
        Some((qkv, rows * self.c, p))
    }
}

/// Mode 0 of the board's attention backward (`attn_back.comp`), one thread = one row (b, h, i), in the order of the
/// code NAK 25.0.7 made (`tools/nak/out/attn_back.sm86.txt`): dP_ij = `ffma.ftz` chain over e ascending of
/// dO_ie·V_je (from 0); dot = `ffma.ftz` chain over j ≤ i ascending of P_ij·dP_ij (from 0); dS_ij =
/// `fmul.ftz`(`fmul.ftz`(P_ij, `fadd.ftz`(dP_ij, −dot)), scale) for j ≤ i and 0 for j > i (the causal mask: P above
/// the diagonal is never read). Returns the T values of the row of dS. No MUFU: the model is the program, bit for bit.
#[allow(clippy::too_many_arguments)]
pub fn attn_back_row_gpu(
    s: &AttnShape,
    qkv: &[f32],
    p: &[f32],
    dout: &[f32],
    scale: f32,
    b: usize,
    h: usize,
    i: usize,
) -> Vec<f32> {
    let (t, c, d) = (s.t, s.c, s.d());
    let prow = ((b * s.heads + h) * t + i) * t;
    let orow = (b * t + i) * c + h * d;
    let mut row = alloc::vec![0.0f32; t];
    for (j, rj) in row.iter_mut().enumerate().take(i + 1) {
        let vrow = (b * t + j) * 3 * c + 2 * c + h * d;
        let mut acc = 0.0f32;
        for e in 0..d {
            acc = fma_f32_ftz(dout[orow + e], qkv[vrow + e], acc);
        }
        *rj = acc;
    }
    let mut dot = 0.0f32;
    for j in 0..=i {
        dot = fma_f32_ftz(p[prow + j], row[j], dot);
    }
    for j in 0..=i {
        let x = add_f32_ftz(row[j], -dot);
        row[j] = mul_ftz(mul_ftz(p[prow + j], x), scale);
    }
    row
}

/// Mode 1 of `attn_back.comp`, one (b, h, position tp, e), in the program's order: dQ = `ffma.ftz` chain over
/// j = 0 … tp of dS_tp,j·K_je; dK = chain over i = tp … T−1 of dS_i,tp·Q_ie; dV = chain over the same i of
/// P_i,tp·dO_ie (all from 0). Returns (dQ, dK, dV) for column h·d + e of row b·T + tp.
#[allow(clippy::too_many_arguments)]
pub fn attn_back_col_gpu(
    s: &AttnShape,
    qkv: &[f32],
    p: &[f32],
    dout: &[f32],
    ds: &[f32],
    b: usize,
    h: usize,
    tp: usize,
    e: usize,
) -> (f32, f32, f32) {
    let (t, c, d) = (s.t, s.c, s.d());
    let base = (b * s.heads + h) * t * t;
    let col = h * d + e;
    let mut dq = 0.0f32;
    for j in 0..=tp {
        dq = fma_f32_ftz(
            ds[base + tp * t + j],
            qkv[(b * t + j) * 3 * c + c + col],
            dq,
        );
    }
    let (mut dk, mut dv) = (0.0f32, 0.0f32);
    for i in tp..t {
        dk = fma_f32_ftz(ds[base + i * t + tp], qkv[(b * t + i) * 3 * c + col], dk);
        dv = fma_f32_ftz(p[base + i * t + tp], dout[(b * t + i) * c + col], dv);
    }
    (dq, dk, dv)
}

/// Both launches of the board's attention backward on the CPU (mode 0 over every row, then mode 1 over every
/// element): returns (dS [B, H, T, T], dqkv [B·T, 3C]). `None` for a bad shape or wrong slice lengths.
pub fn attn_back_gpu(
    s: &AttnShape,
    qkv: &[f32],
    p: &[f32],
    dout: &[f32],
    scale: f32,
) -> Option<(Vec<f32>, Vec<f32>)> {
    let (nq, no, np) = s.sizes()?;
    if qkv.len() != nq || dout.len() != no || p.len() != np {
        return None;
    }
    let mut ds = alloc::vec![0.0f32; np];
    for b in 0..s.batch {
        for h in 0..s.heads {
            for i in 0..s.t {
                let at = ((b * s.heads + h) * s.t + i) * s.t;
                ds[at..at + s.t]
                    .copy_from_slice(&attn_back_row_gpu(s, qkv, p, dout, scale, b, h, i));
            }
        }
    }
    let mut dqkv = alloc::vec![0.0f32; nq];
    let c = s.c;
    for b in 0..s.batch {
        for h in 0..s.heads {
            for tp in 0..s.t {
                let r = b * s.t + tp;
                for e in 0..s.d() {
                    let (dq, dk, dv) = attn_back_col_gpu(s, qkv, p, dout, &ds, b, h, tp, e);
                    let col = h * s.d() + e;
                    dqkv[r * 3 * c + col] = dq;
                    dqkv[r * 3 * c + c + col] = dk;
                    dqkv[r * 3 * c + 2 * c + col] = dv;
                }
            }
        }
    }
    Some((ds, dqkv))
}

/// log2(e) as f32, the constant NAK 25.0.7 encoded for `exp()` in `attn_fwd.comp` (`tools/nak/out/attn_fwd.sm86.txt`).
pub const ATTN_FWD_LOG2E: u32 = 0x3fb8_aa3b;

/// One row (b, h, i) of the board's training attention forward, `attn_fwd.comp` (C-160 point 4, program 6, forward
/// half), in the order of the code NAK 25.0.7 made (`tools/nak/out/attn_fwd.sm86.txt`): s_j = `fmul.ftz`(`ffma.ftz`
/// chain over e ascending of Q_ie·K_je from 0, scale) for j ≤ i; m = max from −f32::MAX; x_j = `MUFU.EX2`(`fmul.ftz`(
/// `fadd.ftz`(s_j, −m), log2 e)); sum = `fadd.ftz` chain from 0, j ascending; inv = `MUFU.RCP`(sum); P_ij =
/// `fmul.ftz`(x_j, inv) for j ≤ i and +0 for j > i (the causal mask); att_ie = `ffma.ftz` chain over j ascending of
/// P_ij·V_je from 0. Returns (row i of P [T values], att row [d values]). `MUFU.EX2` / `MUFU.RCP` are modelled correctly
/// rounded (the board's are approximate): board and model are judged against the f64 training oracle
/// (`train::attention_rows`) in a written margin, not bit for bit.
pub fn attn_fwd_row_gpu(
    s: &AttnShape,
    qkv: &[f32],
    scale: f32,
    b: usize,
    h: usize,
    i: usize,
) -> (Vec<f32>, Vec<f32>) {
    let (t, c, d) = (s.t, s.c, s.d());
    let qrow = (b * t + i) * 3 * c + h * d;
    let mut row = alloc::vec![0.0f32; t];
    let mut m = -f32::MAX;
    for (j, rj) in row.iter_mut().enumerate().take(i + 1) {
        let krow = (b * t + j) * 3 * c + c + h * d;
        let mut acc = 0.0f32;
        for e in 0..d {
            acc = fma_f32_ftz(qkv[qrow + e], qkv[krow + e], acc);
        }
        *rj = mul_ftz(acc, scale);
        m = m.max(*rj);
    }
    let mut sum = 0.0f32;
    for rj in row.iter_mut().take(i + 1) {
        *rj = ex2_ftz(mul_ftz(
            add_f32_ftz(*rj, -m),
            f32::from_bits(ATTN_FWD_LOG2E),
        ));
        sum = add_f32_ftz(sum, *rj);
    }
    let inv = ftz(1.0 / sum);
    for rj in row.iter_mut().take(i + 1) {
        *rj = mul_ftz(*rj, inv);
    }
    let att = (0..d)
        .map(|e| {
            let mut acc = 0.0f32;
            for (j, &pj) in row.iter().enumerate().take(i + 1) {
                acc = fma_f32_ftz(pj, qkv[(b * t + j) * 3 * c + 2 * c + h * d + e], acc);
            }
            acc
        })
        .collect();
    (row, att)
}

/// The whole launch of the board's training attention forward on the CPU (every (b, h, i)): returns
/// (P [B, H, T, T], att [B·T, C]). `None` for a bad shape or a wrong `qkv` length.
pub fn attn_fwd_gpu(s: &AttnShape, qkv: &[f32], scale: f32) -> Option<(Vec<f32>, Vec<f32>)> {
    let (nq, no, np) = s.sizes()?;
    if qkv.len() != nq {
        return None;
    }
    let mut p = alloc::vec![0.0f32; np];
    let mut att = alloc::vec![0.0f32; no];
    let d = s.d();
    for b in 0..s.batch {
        for h in 0..s.heads {
            for i in 0..s.t {
                let (row, a) = attn_fwd_row_gpu(s, qkv, scale, b, h, i);
                let at = ((b * s.heads + h) * s.t + i) * s.t;
                p[at..at + s.t].copy_from_slice(&row);
                let o = (b * s.t + i) * s.c + h * d;
                att[o..o + d].copy_from_slice(&a);
            }
        }
    }
    Some((p, att))
}

/// C-139: every vector one position leaves in the activation scratch of a layer, computed with the **program
/// models** above (`layernorm_gpu`, `q8::matvec_gpu_order`, `vadd_gpu`, `attention_head_gpu`, `gelu_gpu`) in the order
/// of the 16 launches — what each launch should have written, to find the first one that did not. In SIMULARE the
/// model's memory equals this bit for bit; on a board only inside the layer tolerance.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerTrace {
    /// The layer's input (x before launch 0).
    pub x_in: Vec<f32>,
    /// Launch 0: LayerNorm 1.
    pub h: Vec<f32>,
    /// Launch 1: Wqkv·h (3E).
    pub qkv: Vec<f32>,
    /// Launch 2: + bqkv (3E: q | k | v). Launches 3 and 4 copy k and v into the caches.
    pub qkvb: Vec<f32>,
    /// Launch 5: the attention.
    pub att: Vec<f32>,
    /// Launches 6, 7: Wout·att, + bout.
    pub o: Vec<f32>,
    pub ob: Vec<f32>,
    /// Launch 8: x + ob.
    pub x1: Vec<f32>,
    /// Launch 9: LayerNorm 2.
    pub h2: Vec<f32>,
    /// Launches 10, 11: Wfc·h2, + bfc (4E).
    pub f: Vec<f32>,
    pub fb: Vec<f32>,
    /// Launch 12: GELU (4E).
    pub g: Vec<f32>,
    /// Launches 13, 14: Wproj·g, + bproj.
    pub p: Vec<f32>,
    pub pb: Vec<f32>,
    /// Launch 15: x1 + pb, the layer's output (written over x).
    pub x_out: Vec<f32>,
}

/// The attention launch for every head, on explicit caches (`kc`/`vc` = `[position][E]`, rows `0..n_pos`).
pub fn attention_gpu(s: &Shape, q: &[f32], kc: &[f32], vc: &[f32], n_pos: usize) -> Vec<f32> {
    let (e, d) = (s.e, s.d());
    let a = AttnArgs {
        n_pos,
        e,
        d,
        scale: 1.0 / math::sqrt(d as f32),
    };
    (0..s.heads)
        .flat_map(|hd| attention_head_gpu(q, kc, vc, &a, hd))
        .collect()
}

/// C-139: [`LayerTrace`] of position `pos` for input `x`. `kc`/`vc` hold the rows of the positions **before** `pos`
/// (rows from `pos` on are not read; nothing is written).
pub fn trace_gpu(
    s: &Shape,
    w: &LayerWeights<'_>,
    x: &[f32],
    kc: &[f32],
    vc: &[f32],
    pos: usize,
) -> GpuResult<LayerTrace> {
    s.check(w)?;
    let e = s.e;
    if x.len() != e || pos >= MAX_POS || kc.len() < pos * e || vc.len() < pos * e {
        return Err(bad("strat: x de alta marime sau pozitie peste cache"));
    }
    let mv = |m: &[u8], rows: usize, cols: usize, v: &[f32]| -> GpuResult<Vec<f32>> {
        let mut y = alloc::vec![0.0f32; rows];
        q8::matvec_gpu_order(m, rows, cols, v, &mut y)?;
        Ok(y)
    };
    let h = layernorm_gpu(x, w.ln1_g, w.ln1_b, s.eps);
    let qkv = mv(w.wqkv, 3 * e, e, &h)?;
    let qkvb = vadd_gpu(&qkv, w.bqkv);
    let mut k: Vec<f32> = kc[..pos * e].to_vec();
    k.extend_from_slice(&qkvb[e..2 * e]);
    let mut v: Vec<f32> = vc[..pos * e].to_vec();
    v.extend_from_slice(&qkvb[2 * e..]);
    let att = attention_gpu(s, &qkvb, &k, &v, pos + 1);
    let o = mv(w.wout, e, e, &att)?;
    let ob = vadd_gpu(&o, w.bout);
    let x1 = vadd_gpu(x, &ob);
    let h2 = layernorm_gpu(&x1, w.ln2_g, w.ln2_b, s.eps);
    let f = mv(w.wfc, 4 * e, e, &h2)?;
    let fb = vadd_gpu(&f, w.bfc);
    let g: Vec<f32> = fb.iter().map(|&v| gelu_gpu(v)).collect();
    let p = mv(w.wproj, e, 4 * e, &g)?;
    let pb = vadd_gpu(&p, w.bproj);
    let x_out = vadd_gpu(&x1, &pb);
    Ok(LayerTrace {
        x_in: x.to_vec(),
        h,
        qkv,
        qkvb,
        att,
        o,
        ob,
        x1,
        h2,
        f,
        fb,
        g,
        p,
        pb,
        x_out,
    })
}

/// The first element outside the layer tolerance: `(index, gpu, cpu)`; lengths that differ or a value that is not
/// finite never pass.
pub fn first_outside(got: &[f32], cpu: &[f32]) -> Option<(usize, f32, f32)> {
    (0..got.len().max(cpu.len())).find_map(|i| {
        let g = got.get(i).copied().unwrap_or(f32::NAN);
        let c = cpu.get(i).copied().unwrap_or(f32::NAN);
        let ok = g.is_finite() && c.is_finite() && (g - c).abs() <= TOL_ABS + TOL_REL * c.abs();
        (!ok).then_some((i, g, c))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_models_match_the_cpu_engine_closely() {
        let x: Vec<f32> = (0..768)
            .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
            .collect();
        let g: Vec<f32> = (0..768).map(|i| 1.0 + (i % 7) as f32 / 10.0).collect();
        let b: Vec<f32> = (0..768).map(|i| (i % 5) as f32 / 100.0).collect();
        let a = layernorm_cpu(&x, &g, &b, 1e-5);
        let c = layernorm_gpu(&x, &g, &b, 1e-5);
        assert_eq!(first_outside(&c, &a), None);
        for v in [-6.0f32, -1.5, -0.1, 0.0, 0.3, 2.0, 5.0] {
            let mut r = [v];
            crate::reference::gelu_erf(&mut r);
            assert!((gelu_gpu(v) - r[0]).abs() <= 1e-6, "{v}");
        }
        assert_eq!(
            vadd_gpu(&[1.0, f32::MIN_POSITIVE / 2.0], &[2.0, 0.0]),
            [3.0, 0.0]
        );
        assert!(first_outside(&[1.0], &[1.0, 2.0]).is_some());
        assert!(first_outside(&[1.001], &[1.0]).is_some());
        assert!(first_outside(&[1.000_05], &[1.0]).is_none());
    }

    /// C-155 point 2: the new LayerNorm's model is within the layer tolerance of the CPU engine for every length the
    /// models use (64 … 1024, incl. lengths that are not multiples of 256), close to the old program's model, and
    /// refuses what the program cannot take.
    #[test]
    fn layernorm256_model_matches_the_cpu_engine() {
        for n in [64usize, 100, 255, 256, 257, 512, 768, 1000, 1024] {
            let x: Vec<f32> = (0..n)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0 + 3.0)
                .collect();
            let g: Vec<f32> = (0..n).map(|i| 1.0 + (i % 7) as f32 / 10.0).collect();
            let b: Vec<f32> = (0..n).map(|i| (i % 5) as f32 / 100.0).collect();
            let a = layernorm_cpu(&x, &g, &b, 1e-5);
            let c = layernorm_gpu256(&x, &g, &b, 1e-5).unwrap();
            assert_eq!(first_outside(&c, &a), None, "n = {n}");
            let old = layernorm_gpu(&x, &g, &b, 1e-5);
            let d = c
                .iter()
                .zip(&old)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max);
            assert!(d < 1e-5, "n = {n}: {d}");
        }
        // One warp's butterfly: 32 ones in lane order give exactly 32 per warp, 256 for the block.
        let mut p = [1.0f32; 256];
        assert_eq!(block_sum256(&p), 256.0);
        p[0] = f32::MIN_POSITIVE / 2.0;
        p[1] = 0.0;
        assert_eq!(block_sum256(&p), 254.0);
        assert!(layernorm_gpu256(&[], &[], &[], 1e-5).is_err());
        let big = alloc::vec![0.0f32; 1025];
        assert!(layernorm_gpu256(&big, &big, &big, 1e-5).is_err());
        assert!(layernorm_gpu256(&[1.0, 2.0], &[1.0], &[0.0, 0.0], 1e-5).is_err());
    }

    /// C-139: the trace of the program models ends where the CPU engine ends (inside the layer tolerance), position
    /// after position, and reads only the cache rows before its position.
    #[test]
    fn the_trace_follows_the_cpu_engine() {
        let e = 64;
        let s = Shape {
            e,
            heads: 4,
            eps: 1e-5,
        };
        let mut seed = 7u64;
        let mut r = move || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let mut mat = |rows: usize, cols: usize| {
            let rb = cols / q8::BLOCK * q8::BLOCK_BYTES;
            let mut w = alloc::vec![0u8; rows * rb];
            for i in 0..rows {
                let row: Vec<f32> = (0..cols).map(|_| r() * 0.1).collect();
                q8::quantize_row(&row, &mut w[i * rb..(i + 1) * rb]).unwrap();
            }
            w
        };
        let (wqkv, wout, wfc, wproj) = (mat(3 * e, e), mat(e, e), mat(4 * e, e), mat(e, 4 * e));
        let vec = |n: usize, mid: f32| -> Vec<f32> {
            (0..n)
                .map(|i| mid + ((i * 37 % 19) as f32 - 9.0) / 300.0)
                .collect()
        };
        let (g1, b1, g2, b2) = (vec(e, 1.0), vec(e, 0.0), vec(e, 1.0), vec(e, 0.0));
        let (bqkv, bout, bfc, bproj) = (vec(3 * e, 0.0), vec(e, 0.0), vec(4 * e, 0.0), vec(e, 0.0));
        let w = LayerWeights {
            ln1_g: &g1,
            ln1_b: &b1,
            wqkv: &wqkv,
            bqkv: &bqkv,
            wout: &wout,
            bout: &bout,
            ln2_g: &g2,
            ln2_b: &b2,
            wfc: &wfc,
            bfc: &bfc,
            wproj: &wproj,
            bproj: &bproj,
        };
        let (mut kc, mut vc) = (
            alloc::vec![0.0f32; MAX_POS * e],
            alloc::vec![0.0f32; MAX_POS * e],
        );
        for pos in 0..3 {
            let x: Vec<f32> = (0..e)
                .map(|i| ((i * 13 + pos * 7) % 23) as f32 / 11.0 - 1.0)
                .collect();
            // Rows from `pos` on are poisoned: the trace must not read them.
            let (mut kp, mut vp) = (kc.clone(), vc.clone());
            kp[pos * e..].fill(f32::NAN);
            vp[pos * e..].fill(f32::NAN);
            let t = trace_gpu(&s, &w, &x, &kp, &vp, pos).unwrap();
            let mut cpu = x.clone();
            step_cpu(&s, &w, &mut cpu, &mut kc, &mut vc, pos).unwrap();
            assert_eq!(first_outside(&t.x_out, &cpu), None, "pozitia {pos}");
            assert_eq!(
                first_outside(&t.qkvb[e..2 * e], &kc[pos * e..(pos + 1) * e]),
                None
            );
            assert_eq!(t.x_in, x);
            assert_eq!((t.qkv.len(), t.f.len(), t.g.len()), (3 * e, 4 * e, 4 * e));
            assert_eq!(t.x_out, vadd_gpu(&t.x1, &t.pb));
        }
        assert!(trace_gpu(&s, &w, &[0.0; 3], &kc, &vc, 0).is_err());
        assert!(trace_gpu(&s, &w, &alloc::vec![0.0; e], &kc[..e], &vc, 2).is_err());
    }
}
