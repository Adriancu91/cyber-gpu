//! C-160 point 3 — the CPU oracle of one whole **training step** of Kronos (`kronos_gpt`, the architecture of the hub's
//! `integrare/kronos/PAS_DE_AUR.md`): forward, mean cross-entropy, backward through every tensor, gradient clipping
//! and the two optimizer recipes of the "golden step" (`sgd`, `adamw_513`).
//!
//! It is the **checking tool** for the training programs of the board (C-160 point 4): every GPU program is compared
//! with the matching part of this file, and this file is compared with PyTorch on the golden step (the integration
//! test `tests/c160_p3_golden_step.rs`). It is not a CPU training path for Kronos (hub rule of 13 nov).
//!
//! Precision: tensors are f32 (as in PyTorch); every sum (dot products, means, variances, softmax, weight
//! gradients) is accumulated in f64 and rounded once, so the oracle sits as close as possible to the exact value and
//! the comparison with PyTorch (f32, other summation order) is judged with the tolerances written in the hub's
//! `PAS_DE_AUR.md` ([`TOL_LOSS_REL`] …). The optimizer updates follow PyTorch 2.14's single-tensor formulas step by
//! step in f32, so the weights after a step round the way PyTorch's do.
//!
//! Sources (formulas only, our own code): the hub's `PAS_DE_AUR.md` (network, loss, optimizers), the PyTorch
//! documentation of `LayerNorm`, `GELU`, `MultiheadAttention`, `SGD`, `AdamW`, `clip_grad_norm_` (BSD), and the usual
//! backward formulas (as in `docs/C-160_MOTOR_STUDIU.md`). Nothing here runs on a GPU.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

/// Loss, relative (hub `PAS_DE_AUR.md`, "Toleranțe recomandate").
pub const TOL_LOSS_REL: f64 = 1e-5;
/// Step-1 gradient, absolute per element.
pub const TOL_GRAD_ABS: f32 = 1e-5;
/// Step-1 gradient, ‖Δ‖ / ‖g‖ per tensor.
pub const TOL_GRAD_REL_NORM: f64 = 1e-3;
/// SGD weights after steps 1 and 10, absolute.
pub const TOL_SGD_W_ABS: f32 = 1e-6;
/// AdamW weights after steps 1 and 10, absolute (= 2 · lr).
pub const TOL_ADAMW_W_ABS: f32 = 1e-4;

/// LayerNorm epsilon of `kronos_gpt`.
pub const LN_EPS: f64 = 1e-5;

fn need(cond: bool, what: &'static str) -> GpuResult<()> {
    if cond {
        Ok(())
    } else {
        Err(GpuError::InvalidArgument(what))
    }
}

/// The shape of a `kronos_gpt` network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arch {
    pub vocab: usize,
    pub ctx: usize,
    pub n_emb: usize,
    pub heads: usize,
    pub layers: usize,
}

/// The small model of the golden step (`pas_de_aur_mic/reteta.json`).
pub const MIC: Arch = Arch {
    vocab: 256,
    ctx: 32,
    n_emb: 64,
    heads: 4,
    layers: 2,
};

/// #513 (`pas_de_aur_513/reteta.json`).
pub const M513: Arch = Arch {
    vocab: 16_384,
    ctx: 512,
    n_emb: 768,
    heads: 12,
    layers: 12,
};

/// One parameter tensor: PyTorch name and shape (row-major).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorSpec {
    pub name: String,
    pub shape: Vec<usize>,
}

impl TensorSpec {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Tensors per layer, in `named_parameters()` order.
pub const PER_LAYER: usize = 12;
const LAYER_NAMES: [&str; PER_LAYER] = [
    "ln1.weight",
    "ln1.bias",
    "atentie.in_proj_weight",
    "atentie.in_proj_bias",
    "atentie.out_proj.weight",
    "atentie.out_proj.bias",
    "ln2.weight",
    "ln2.bias",
    "mlp.0.weight",
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
];

// Offsets inside a layer's block.
const LN1_W: usize = 0;
const LN1_B: usize = 1;
const IN_W: usize = 2;
const IN_B: usize = 3;
const OUT_W: usize = 4;
const OUT_B: usize = 5;
const LN2_W: usize = 6;
const LN2_B: usize = 7;
const FC_W: usize = 8;
const FC_B: usize = 9;
const PROJ_W: usize = 10;
const PROJ_B: usize = 11;

impl Arch {
    fn check(&self) -> GpuResult<()> {
        need(
            self.vocab > 0
                && self.ctx > 0
                && self.layers > 0
                && self.heads > 0
                && self.n_emb > 0
                && self.n_emb.is_multiple_of(self.heads),
            "train: arhitectura",
        )
    }

    /// Head width `d = n_emb / heads`.
    pub fn head_dim(&self) -> usize {
        self.n_emb / self.heads
    }

    /// Number of parameter tensors (the tied head has no tensor of its own).
    pub fn tensor_count(&self) -> usize {
        2 + PER_LAYER * self.layers + 2
    }

    /// Index of tensor `off` of layer `l` ([`PER_LAYER`] per layer).
    fn at(&self, l: usize, off: usize) -> usize {
        2 + PER_LAYER * l + off
    }

    fn final_ln(&self) -> (usize, usize) {
        let b = 2 + PER_LAYER * self.layers;
        (b, b + 1)
    }

    /// The tensors in PyTorch's `named_parameters()` order (the hub's table "Ordinea tensorilor").
    pub fn tensors(&self) -> Vec<TensorSpec> {
        let (c, v, t) = (self.n_emb, self.vocab, self.ctx);
        let mut out = Vec::with_capacity(self.tensor_count());
        let mut push = |name: String, shape: Vec<usize>| out.push(TensorSpec { name, shape });
        push("emb_token.weight".into(), vec![v, c]);
        push("emb_pozitie.weight".into(), vec![t, c]);
        for l in 0..self.layers {
            let shapes: [Vec<usize>; PER_LAYER] = [
                vec![c],
                vec![c],
                vec![3 * c, c],
                vec![3 * c],
                vec![c, c],
                vec![c],
                vec![c],
                vec![c],
                vec![4 * c, c],
                vec![4 * c],
                vec![c, 4 * c],
                vec![c],
            ];
            for (name, shape) in LAYER_NAMES.iter().zip(shapes) {
                push(alloc::format!("straturi.{l}.{name}"), shape);
            }
        }
        push("ln.weight".into(), vec![c]);
        push("ln.bias".into(), vec![c]);
        out
    }

    /// Checks that `params` has every tensor with the right number of elements.
    pub fn check_params(&self, params: &[Vec<f32>]) -> GpuResult<()> {
        self.check()?;
        let specs = self.tensors();
        need(params.len() == specs.len(), "train: numarul tensorilor")?;
        for (p, s) in params.iter().zip(&specs) {
            need(p.len() == s.elements(), "train: marimea unui tensor")?;
        }
        Ok(())
    }
}

/// A batch: `inputs[b·len + t]`, `targets[b·len + t]` = the token after `inputs[b·len + t]`.
#[derive(Clone, Copy, Debug)]
pub struct Batch<'a> {
    pub inputs: &'a [u32],
    pub targets: &'a [u32],
    pub batch: usize,
    pub len: usize,
}

// ---------------------------------------------------------------- f64 helpers (no_std: no libm)

fn sqrt64(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return if x == 0.0 { 0.0 } else { f64::NAN };
    }
    let mut y = f64::from_bits((x.to_bits() >> 1) + (1023u64 << 51));
    for _ in 0..8 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// e^x in f64, range reduction + Taylor (relative error ~1e-15 on [-700, 700]).
fn exp64(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if x > 709.0 {
        return f64::INFINITY;
    }
    if x < -745.0 {
        return 0.0;
    }
    const LN2: f64 = core::f64::consts::LN_2;
    let kf = x / LN2;
    let k = if kf >= 0.0 {
        (kf + 0.5) as i64
    } else {
        (kf - 0.5) as i64
    };
    let r = x - k as f64 * LN2;
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for i in 1..=18 {
        term *= r / i as f64;
        sum += term;
    }
    let (k1, k2) = (k / 2, k - k / 2);
    let p1 = f64::from_bits(((1023 + k1) as u64) << 52);
    let p2 = f64::from_bits(((1023 + k2) as u64) << 52);
    sum * p1 * p2
}

/// erf in f64: `2/√π · e^(−x²) · Σ 2ⁿ x^(2n+1) / (1·3·…·(2n+1))` (all terms positive, no cancellation);
/// ±1 for |x| ≥ 6 (1 − erf(6) < 3e-17).
fn erf64(x: f64) -> f64 {
    let a = if x < 0.0 { -x } else { x };
    if a >= 6.0 {
        return if x < 0.0 { -1.0 } else { 1.0 };
    }
    let x2 = a * a;
    let mut term = a;
    let mut sum = a;
    let mut n = 0.0f64;
    loop {
        n += 1.0;
        term *= 2.0 * x2 / (2.0 * n + 1.0);
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    let r = 2.0 / sqrt64(core::f64::consts::PI) * exp64(-x2) * sum;
    if x < 0.0 { -r } else { r }
}

fn ln64(x: f64) -> f64 {
    crate::reference::ln(x)
}

/// Exact GELU `0.5·u·(1 + erf(u/√2))` and its derivative `Φ(u) + u·φ(u)`.
fn gelu_and_grad(u: f64) -> (f64, f64) {
    let cdf = 0.5 * (1.0 + erf64(u * core::f64::consts::FRAC_1_SQRT_2));
    let pdf = exp64(-0.5 * u * u) / sqrt64(2.0 * core::f64::consts::PI);
    (u * cdf, cdf + u * pdf)
}

// ---------------------------------------------------------------- building blocks

/// y[r, o] = Σ_k x[r, k]·w[o, k] + b[o]  (PyTorch `Linear`: weight [out, in]).
fn linear(
    x: &[f32],
    rows: usize,
    k_in: usize,
    w: &[f32],
    out: usize,
    b: Option<&[f32]>,
) -> Vec<f32> {
    let mut y = vec![0.0f32; rows * out];
    for r in 0..rows {
        let xr = &x[r * k_in..(r + 1) * k_in];
        for o in 0..out {
            let wo = &w[o * k_in..(o + 1) * k_in];
            let mut acc = 0.0f64;
            for k in 0..k_in {
                acc += xr[k] as f64 * wo[k] as f64;
            }
            if let Some(b) = b {
                acc += b[o] as f64;
            }
            y[r * out + o] = acc as f32;
        }
    }
    y
}

/// Backward of [`linear`]: returns dx; writes dW (and db) — the sums over rows in f64.
#[allow(clippy::too_many_arguments)]
fn linear_back(
    dy: &[f32],
    x: &[f32],
    rows: usize,
    k_in: usize,
    w: &[f32],
    out: usize,
    dw: &mut [f32],
    db: Option<&mut [f32]>,
) -> Vec<f32> {
    let mut acc_w = vec![0.0f64; out * k_in];
    for r in 0..rows {
        let xr = &x[r * k_in..(r + 1) * k_in];
        for o in 0..out {
            let g = dy[r * out + o] as f64;
            if g == 0.0 {
                continue;
            }
            let row = &mut acc_w[o * k_in..(o + 1) * k_in];
            for k in 0..k_in {
                row[k] += g * xr[k] as f64;
            }
        }
    }
    for (d, a) in dw.iter_mut().zip(&acc_w) {
        *d = *a as f32;
    }
    if let Some(db) = db {
        for (o, d) in db.iter_mut().enumerate() {
            let mut s = 0.0f64;
            for r in 0..rows {
                s += dy[r * out + o] as f64;
            }
            *d = s as f32;
        }
    }
    let mut dx = vec![0.0f32; rows * k_in];
    let mut acc = vec![0.0f64; k_in];
    for r in 0..rows {
        acc.iter_mut().for_each(|a| *a = 0.0);
        for o in 0..out {
            let g = dy[r * out + o] as f64;
            if g == 0.0 {
                continue;
            }
            let wo = &w[o * k_in..(o + 1) * k_in];
            for k in 0..k_in {
                acc[k] += g * wo[k] as f64;
            }
        }
        for k in 0..k_in {
            dx[r * k_in + k] = acc[k] as f32;
        }
    }
    dx
}

/// What LayerNorm keeps for the backward pass.
struct LnCache {
    xhat: Vec<f32>,
    rstd: Vec<f64>,
}

fn layernorm(x: &[f32], rows: usize, c: usize, g: &[f32], b: &[f32]) -> (Vec<f32>, LnCache) {
    let mut y = vec![0.0f32; rows * c];
    let mut xhat = vec![0.0f32; rows * c];
    let mut rstd = vec![0.0f64; rows];
    for r in 0..rows {
        let row = &x[r * c..(r + 1) * c];
        let mean = row.iter().map(|&v| v as f64).sum::<f64>() / c as f64;
        let var = row
            .iter()
            .map(|&v| (v as f64 - mean) * (v as f64 - mean))
            .sum::<f64>()
            / c as f64;
        let rs = 1.0 / sqrt64(var + LN_EPS);
        rstd[r] = rs;
        for k in 0..c {
            let h = (row[k] as f64 - mean) * rs;
            xhat[r * c + k] = h as f32;
            y[r * c + k] = (h * g[k] as f64 + b[k] as f64) as f32;
        }
    }
    (y, LnCache { xhat, rstd })
}

/// LayerNorm backward: returns dx, writes dγ and dβ.
fn layernorm_back(
    dy: &[f32],
    cache: &LnCache,
    rows: usize,
    c: usize,
    g: &[f32],
    dg: &mut [f32],
    db: &mut [f32],
) -> Vec<f32> {
    let mut acc_g = vec![0.0f64; c];
    let mut acc_b = vec![0.0f64; c];
    let mut dx = vec![0.0f32; rows * c];
    let mut dxhat = vec![0.0f64; c];
    for r in 0..rows {
        let mut m1 = 0.0f64;
        let mut m2 = 0.0f64;
        for k in 0..c {
            let d = dy[r * c + k] as f64;
            let h = cache.xhat[r * c + k] as f64;
            acc_g[k] += d * h;
            acc_b[k] += d;
            dxhat[k] = d * g[k] as f64;
            m1 += dxhat[k];
            m2 += dxhat[k] * h;
        }
        m1 /= c as f64;
        m2 /= c as f64;
        let rs = cache.rstd[r];
        for k in 0..c {
            let h = cache.xhat[r * c + k] as f64;
            dx[r * c + k] = (rs * (dxhat[k] - m1 - h * m2)) as f32;
        }
    }
    for k in 0..c {
        dg[k] = acc_g[k] as f32;
        db[k] = acc_b[k] as f32;
    }
    dx
}

/// What one layer keeps for the backward pass.
struct LayerCache {
    ln1: LnCache,
    h1: Vec<f32>,
    qkv: Vec<f32>,
    /// Attention probabilities [b, head, i, j] (0 above the diagonal).
    p: Vec<f64>,
    att: Vec<f32>,
    ln2: LnCache,
    h2: Vec<f32>,
    u: Vec<f32>,
    act: Vec<f32>,
}

/// Causal multi-head attention core on `qkv` [B·T, 3C] (rows Q | K | V, head h = columns h·d … (h+1)·d − 1).
fn attention(a: &Arch, qkv: &[f32], bsz: usize, t: usize) -> (Vec<f32>, Vec<f64>) {
    let (c, nh, d) = (a.n_emb, a.heads, a.head_dim());
    let scale = 1.0 / sqrt64(d as f64);
    let mut att = vec![0.0f32; bsz * t * c];
    let mut p = vec![0.0f64; bsz * nh * t * t];
    let mut s = vec![0.0f64; t];
    for b in 0..bsz {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for i in 0..t {
                let qi = &qkv[(b * t + i) * 3 * c + h * d..][..d];
                let mut max = f64::NEG_INFINITY;
                for (j, sj) in s.iter_mut().enumerate().take(i + 1) {
                    let kj = &qkv[(b * t + j) * 3 * c + c + h * d..][..d];
                    let mut dot = 0.0f64;
                    for e in 0..d {
                        dot += qi[e] as f64 * kj[e] as f64;
                    }
                    *sj = dot * scale;
                    if *sj > max {
                        max = *sj;
                    }
                }
                let mut sum = 0.0f64;
                for sj in s.iter_mut().take(i + 1) {
                    *sj = exp64(*sj - max);
                    sum += *sj;
                }
                for j in 0..=i {
                    p[base + i * t + j] = s[j] / sum;
                }
                for e in 0..d {
                    let mut acc = 0.0f64;
                    for j in 0..=i {
                        acc += p[base + i * t + j]
                            * qkv[(b * t + j) * 3 * c + 2 * c + h * d + e] as f64;
                    }
                    att[(b * t + i) * c + h * d + e] = acc as f32;
                }
            }
        }
    }
    (att, p)
}

/// Backward of [`attention`]: from d(att) [B·T, C] to d(qkv) [B·T, 3C].
fn attention_back(
    a: &Arch,
    datt: &[f32],
    qkv: &[f32],
    p: &[f64],
    bsz: usize,
    t: usize,
) -> Vec<f32> {
    let (c, nh, d) = (a.n_emb, a.heads, a.head_dim());
    let scale = 1.0 / sqrt64(d as f64);
    let mut dq = vec![0.0f64; bsz * t * c];
    let mut dk = vec![0.0f64; bsz * t * c];
    let mut dv = vec![0.0f64; bsz * t * c];
    let mut dp = vec![0.0f64; t];
    for b in 0..bsz {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for i in 0..t {
                let da = &datt[(b * t + i) * c + h * d..][..d];
                // dP_ij = Σ_e dA_ie · V_je ; dV_je += P_ij · dA_ie
                for j in 0..=i {
                    let vj = &qkv[(b * t + j) * 3 * c + 2 * c + h * d..][..d];
                    let pij = p[base + i * t + j];
                    let mut acc = 0.0f64;
                    for e in 0..d {
                        acc += da[e] as f64 * vj[e] as f64;
                        dv[(b * t + j) * c + h * d + e] += pij * da[e] as f64;
                    }
                    dp[j] = acc;
                }
                // dS_ij = P_ij (dP_ij − Σ_j' P_ij' dP_ij'), then through the scale.
                let mut dot = 0.0f64;
                for j in 0..=i {
                    dot += p[base + i * t + j] * dp[j];
                }
                let qi = &qkv[(b * t + i) * 3 * c + h * d..][..d];
                for j in 0..=i {
                    let ds = p[base + i * t + j] * (dp[j] - dot) * scale;
                    if ds == 0.0 {
                        continue;
                    }
                    let kj = &qkv[(b * t + j) * 3 * c + c + h * d..][..d];
                    for e in 0..d {
                        dq[(b * t + i) * c + h * d + e] += ds * kj[e] as f64;
                        dk[(b * t + j) * c + h * d + e] += ds * qi[e] as f64;
                    }
                }
            }
        }
    }
    let mut dqkv = vec![0.0f32; bsz * t * 3 * c];
    for r in 0..bsz * t {
        for k in 0..c {
            dqkv[r * 3 * c + k] = dq[r * c + k] as f32;
            dqkv[r * 3 * c + c + k] = dk[r * c + k] as f32;
            dqkv[r * 3 * c + 2 * c + k] = dv[r * c + k] as f32;
        }
    }
    dqkv
}

// ---------------------------------------------------------------- the step

/// Mean cross-entropy of the batch and the gradient of every tensor (before clipping), in `params` order.
/// `emb_token`'s gradient is the sum of its two uses (input embedding + tied output head).
pub fn loss_and_grad(
    a: &Arch,
    params: &[Vec<f32>],
    batch: &Batch,
) -> GpuResult<(f32, Vec<Vec<f32>>)> {
    a.check_params(params)?;
    let (bsz, t) = (batch.batch, batch.len);
    need(bsz > 0 && t > 0 && t <= a.ctx, "train: forma lotului")?;
    need(
        batch.inputs.len() == bsz * t && batch.targets.len() == bsz * t,
        "train: marimea lotului",
    )?;
    need(
        batch
            .inputs
            .iter()
            .chain(batch.targets)
            .all(|&s| (s as usize) < a.vocab),
        "train: simbol in afara vocabularului",
    )?;
    let (c, v) = (a.n_emb, a.vocab);
    let n = bsz * t;
    let f = 4 * c;
    let emb = &params[0];
    let pos = &params[1];

    // Forward.
    let mut x = vec![0.0f32; n * c];
    for b in 0..bsz {
        for i in 0..t {
            let r = b * t + i;
            let tok = batch.inputs[r] as usize;
            for k in 0..c {
                x[r * c + k] = emb[tok * c + k] + pos[i * c + k];
            }
        }
    }
    let mut caches: Vec<LayerCache> = Vec::with_capacity(a.layers);
    for l in 0..a.layers {
        let w = |off: usize| &params[a.at(l, off)][..];
        let (h1, ln1) = layernorm(&x, n, c, w(LN1_W), w(LN1_B));
        let qkv = linear(&h1, n, c, w(IN_W), 3 * c, Some(w(IN_B)));
        let (att, p) = attention(a, &qkv, bsz, t);
        let y = linear(&att, n, c, w(OUT_W), c, Some(w(OUT_B)));
        for (xi, yi) in x.iter_mut().zip(&y) {
            *xi += *yi;
        }
        let (h2, ln2) = layernorm(&x, n, c, w(LN2_W), w(LN2_B));
        let u = linear(&h2, n, c, w(FC_W), f, Some(w(FC_B)));
        let act: Vec<f32> = u
            .iter()
            .map(|&ui| gelu_and_grad(ui as f64).0 as f32)
            .collect();
        let z = linear(&act, n, f, w(PROJ_W), c, Some(w(PROJ_B)));
        for (xi, zi) in x.iter_mut().zip(&z) {
            *xi += *zi;
        }
        caches.push(LayerCache {
            ln1,
            h1,
            qkv,
            p,
            att,
            ln2,
            h2,
            u,
            act,
        });
    }
    let (lnf_w, lnf_b) = a.final_ln();
    let (hf, lnf) = layernorm(&x, n, c, &params[lnf_w], &params[lnf_b]);
    let logits = linear(&hf, n, c, emb, v, None);

    // Loss and d(logits) = (softmax − onehot) / N.
    let mut total = 0.0f64;
    let mut dlogits = vec![0.0f32; n * v];
    let mut e = vec![0.0f64; v];
    for r in 0..n {
        let row = &logits[r * v..(r + 1) * v];
        let max = row.iter().fold(f32::NEG_INFINITY, |m, &z| m.max(z)) as f64;
        let mut sum = 0.0f64;
        for (ek, &z) in e.iter_mut().zip(row) {
            *ek = exp64(z as f64 - max);
            sum += *ek;
        }
        let tgt = batch.targets[r] as usize;
        total += ln64(sum) + max - row[tgt] as f64;
        for k in 0..v {
            let s = e[k] / sum - if k == tgt { 1.0 } else { 0.0 };
            dlogits[r * v + k] = (s / n as f64) as f32;
        }
    }
    let loss = (total / n as f64) as f32;

    // Backward.
    let mut grads: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0f32; p.len()]).collect();
    let mut d_emb_head = vec![0.0f32; v * c];
    let dhf = linear_back(&dlogits, &hf, n, c, emb, v, &mut d_emb_head, None);
    let mut dx = {
        let (gw, gb) = grads.split_at_mut(lnf_b);
        layernorm_back(&dhf, &lnf, n, c, &params[lnf_w], &mut gw[lnf_w], &mut gb[0])
    };
    for l in (0..a.layers).rev() {
        let cache = &caches[l];
        let w = |off: usize| &params[a.at(l, off)][..];
        // x = x_mid + proj(gelu(fc(LN2(x_mid))))
        let mut d_proj_w = vec![0.0f32; c * f];
        let mut d_proj_b = vec![0.0f32; c];
        let dact = linear_back(
            &dx,
            &cache.act,
            n,
            f,
            w(PROJ_W),
            c,
            &mut d_proj_w,
            Some(&mut d_proj_b),
        );
        let du: Vec<f32> = dact
            .iter()
            .zip(&cache.u)
            .map(|(&g, &ui)| (g as f64 * gelu_and_grad(ui as f64).1) as f32)
            .collect();
        let mut d_fc_w = vec![0.0f32; f * c];
        let mut d_fc_b = vec![0.0f32; f];
        let dh2 = linear_back(
            &du,
            &cache.h2,
            n,
            c,
            w(FC_W),
            f,
            &mut d_fc_w,
            Some(&mut d_fc_b),
        );
        let mut d_ln2_w = vec![0.0f32; c];
        let mut d_ln2_b = vec![0.0f32; c];
        let dxm = layernorm_back(&dh2, &cache.ln2, n, c, w(LN2_W), &mut d_ln2_w, &mut d_ln2_b);
        for (a_, b_) in dx.iter_mut().zip(&dxm) {
            *a_ += *b_;
        }
        // x_mid = x_in + out(attn(in(LN1(x_in))))
        let mut d_out_w = vec![0.0f32; c * c];
        let mut d_out_b = vec![0.0f32; c];
        let datt = linear_back(
            &dx,
            &cache.att,
            n,
            c,
            w(OUT_W),
            c,
            &mut d_out_w,
            Some(&mut d_out_b),
        );
        let dqkv = attention_back(a, &datt, &cache.qkv, &cache.p, bsz, t);
        let mut d_in_w = vec![0.0f32; 3 * c * c];
        let mut d_in_b = vec![0.0f32; 3 * c];
        let dh1 = linear_back(
            &dqkv,
            &cache.h1,
            n,
            c,
            w(IN_W),
            3 * c,
            &mut d_in_w,
            Some(&mut d_in_b),
        );
        let mut d_ln1_w = vec![0.0f32; c];
        let mut d_ln1_b = vec![0.0f32; c];
        let dxi = layernorm_back(&dh1, &cache.ln1, n, c, w(LN1_W), &mut d_ln1_w, &mut d_ln1_b);
        for (a_, b_) in dx.iter_mut().zip(&dxi) {
            *a_ += *b_;
        }
        for (off, g) in [
            (LN1_W, d_ln1_w),
            (LN1_B, d_ln1_b),
            (IN_W, d_in_w),
            (IN_B, d_in_b),
            (OUT_W, d_out_w),
            (OUT_B, d_out_b),
            (LN2_W, d_ln2_w),
            (LN2_B, d_ln2_b),
            (FC_W, d_fc_w),
            (FC_B, d_fc_b),
            (PROJ_W, d_proj_w),
            (PROJ_B, d_proj_b),
        ] {
            grads[a.at(l, off)] = g;
        }
    }
    // Embeddings: scatter-add of dx (f64 sums), then the tied head's part.
    let mut acc_tok = vec![0.0f64; v * c];
    let mut acc_pos = vec![0.0f64; a.ctx * c];
    for b in 0..bsz {
        for i in 0..t {
            let r = b * t + i;
            let tok = batch.inputs[r] as usize;
            for k in 0..c {
                acc_tok[tok * c + k] += dx[r * c + k] as f64;
                acc_pos[i * c + k] += dx[r * c + k] as f64;
            }
        }
    }
    for (k, g) in grads[0].iter_mut().enumerate() {
        *g = acc_tok[k] as f32 + d_emb_head[k];
    }
    for (k, g) in grads[1].iter_mut().enumerate() {
        *g = acc_pos[k] as f32;
    }
    Ok((loss, grads))
}

/// Total L2 norm of all gradients ("norm of norms", as `clip_grad_norm_`).
pub fn grad_norm(grads: &[Vec<f32>]) -> f64 {
    let s: f64 = grads.iter().flatten().map(|&g| g as f64 * g as f64).sum();
    sqrt64(s)
}

/// `clip_grad_norm_(max_norm)`: `g ← g · min(1, max_norm / (N + 1e-6))`; returns N.
pub fn clip_grad_norm(grads: &mut [Vec<f32>], max_norm: f32) -> f32 {
    let total = grad_norm(grads) as f32;
    let coef = max_norm / (total + 1e-6);
    if coef < 1.0 {
        for g in grads.iter_mut().flatten() {
            *g *= coef;
        }
    }
    total
}

/// SGD without momentum or decay: `w ← w + (−lr)·g`, in f32 (`param.add_(grad, alpha = −lr)`).
pub fn sgd_step(params: &mut [Vec<f32>], grads: &[Vec<f32>], lr: f32) -> GpuResult<()> {
    need(params.len() == grads.len(), "sgd: numarul tensorilor")?;
    for (p, g) in params.iter_mut().zip(grads) {
        need(p.len() == g.len(), "sgd: marimea unui tensor")?;
        for (w, &gi) in p.iter_mut().zip(g) {
            *w += -lr * gi;
        }
    }
    Ok(())
}

/// AdamW as PyTorch 2.14 applies it (single-tensor path, `amsgrad = False`, one parameter group).
#[derive(Clone, Debug)]
pub struct AdamW {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
    /// Steps done so far.
    pub t: u32,
    pub m: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl AdamW {
    /// The `adamw_513` recipe: lr 5e-5, betas (0.9, 0.999), eps 1e-8, weight decay 0.1.
    pub fn recipe_513(params: &[Vec<f32>]) -> Self {
        Self::new(params, 5e-5, 0.9, 0.999, 1e-8, 0.1)
    }

    pub fn new(
        params: &[Vec<f32>],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
    ) -> Self {
        let zeros = || {
            params
                .iter()
                .map(|p| vec![0.0f32; p.len()])
                .collect::<Vec<_>>()
        };
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            t: 0,
            m: zeros(),
            v: zeros(),
        }
    }

    /// One step, per element, in PyTorch's order and f32 rounding:
    /// `w ·= 1 − lr·wd`; `m = lerp(m, g, 1 − β1)`; `v = v·β2 + (1 − β2)·g²`;
    /// `w −= (lr / (1 − β1ᵗ)) · m / (√v / √(1 − β2ᵗ) + eps)`.
    pub fn step(&mut self, params: &mut [Vec<f32>], grads: &[Vec<f32>]) -> GpuResult<()> {
        need(
            params.len() == grads.len() && params.len() == self.m.len(),
            "adamw: numarul tensorilor",
        )?;
        self.t += 1;
        let t = self.t as i32;
        let decay = (1.0 - self.lr * self.weight_decay) as f32;
        let w1 = (1.0 - self.beta1) as f32;
        let b2 = self.beta2 as f32;
        let one_minus_b2 = (1.0 - self.beta2) as f32;
        let mut pow1 = 1.0f64;
        let mut pow2 = 1.0f64;
        for _ in 0..t {
            pow1 *= self.beta1;
            pow2 *= self.beta2;
        }
        let bc1 = 1.0 - pow1;
        let bc2_sqrt = sqrt64(1.0 - pow2) as f32;
        let step_size = (self.lr / bc1) as f32;
        let eps = self.eps as f32;
        for (k, (p, g)) in params.iter_mut().zip(grads).enumerate() {
            need(
                p.len() == g.len() && p.len() == self.m[k].len(),
                "adamw: marimea unui tensor",
            )?;
            let (m, v) = (&mut self.m[k], &mut self.v[k]);
            for i in 0..p.len() {
                let gi = g[i];
                p[i] *= decay;
                // torch.lerp with weight < 0.5: self + weight·(end − self)
                m[i] += w1 * (gi - m[i]);
                v[i] = v[i] * b2 + one_minus_b2 * gi * gi;
                let denom = crate::math::sqrt(v[i]) / bc2_sqrt + eps;
                p[i] += -step_size * (m[i] / denom);
            }
        }
        Ok(())
    }
}

/// Which recipe of the golden step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Recipe {
    /// `sgd`: lr 0.01, no clipping.
    Sgd,
    /// `adamw_513`: clipping at 1.0, then [`AdamW::recipe_513`].
    AdamW513,
}

/// What a run of the golden step produces, to compare with `reteta.json` and the `.f32` files.
pub struct GoldenRun {
    /// Loss of step k (on the weights before update k), k = 1 … steps.
    pub losses: Vec<f32>,
    /// Gradient norm of step k, before clipping.
    pub grad_norms: Vec<f32>,
    /// Gradient of step 1, before clipping.
    pub grad_step1: Vec<Vec<f32>>,
    pub after_step1: Vec<Vec<f32>>,
    pub after_last: Vec<Vec<f32>>,
}

/// The golden step: `steps` times zero-grad → forward → backward → (clip) → optimizer, on the same batch.
pub fn golden_run(
    a: &Arch,
    initial: &[Vec<f32>],
    batch: &Batch,
    recipe: Recipe,
    steps: usize,
) -> GpuResult<GoldenRun> {
    need(steps > 0, "train: pasi")?;
    let mut params: Vec<Vec<f32>> = initial.to_vec();
    let mut adam = AdamW::recipe_513(&params);
    let mut run = GoldenRun {
        losses: Vec::with_capacity(steps),
        grad_norms: Vec::with_capacity(steps),
        grad_step1: Vec::new(),
        after_step1: Vec::new(),
        after_last: Vec::new(),
    };
    for k in 0..steps {
        let (loss, mut grads) = loss_and_grad(a, &params, batch)?;
        run.losses.push(loss);
        run.grad_norms.push(grad_norm(&grads) as f32);
        if k == 0 {
            run.grad_step1 = grads.clone();
        }
        match recipe {
            Recipe::Sgd => sgd_step(&mut params, &grads, 0.01)?,
            Recipe::AdamW513 => {
                clip_grad_norm(&mut grads, 1.0);
                adam.step(&mut params, &grads)?;
            }
        }
        if k == 0 {
            run.after_step1 = params.clone();
        }
    }
    run.after_last = params;
    Ok(run)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers_match_std() {
        for &x in &[-30.0, -3.2, -0.5, 0.0, 1e-3, 0.7, 2.5, 50.0] {
            let r = exp64(x);
            assert!(((r - f64::exp(x)) / f64::exp(x)).abs() < 1e-14, "exp {x}");
        }
        for &x in &[0.0, 1e-8, 0.3, 2.0, 1234.5] {
            assert!((sqrt64(x) - f64::sqrt(x)).abs() <= 1e-15 * f64::sqrt(x).max(1.0));
        }
        // erf reference values (Abramowitz & Stegun table 7.1).
        for &(x, want) in &[
            (0.5, 0.520_499_877_813_046_5),
            (1.0, 0.842_700_792_949_714_9),
            (2.0, 0.995_322_265_018_952_7),
            (3.5, 0.999_999_256_901_627_7),
        ] {
            assert!(
                (erf64(x) - want).abs() < 1e-14,
                "erf {x}: {}",
                erf64(x) - want
            );
            assert!((erf64(-x) + want).abs() < 1e-14);
        }
    }

    #[test]
    fn layout_of_the_small_model() {
        let specs = MIC.tensors();
        assert_eq!(specs.len(), 28);
        assert_eq!(specs[4].name, "straturi.0.atentie.in_proj_weight");
        assert_eq!(specs[4].shape, vec![192, 64]);
        assert_eq!(specs[27].name, "ln.bias");
        assert_eq!(specs.iter().map(|s| s.elements()).sum::<usize>(), 118_528);
        assert_eq!(M513.tensors().len(), 148);
        assert_eq!(
            M513.tensors().iter().map(|s| s.elements()).sum::<usize>(),
            98_032_128
        );
    }

    /// Tiny deterministic model for the finite-difference check (independent of PyTorch).
    fn tiny() -> (Arch, Vec<Vec<f32>>, Vec<u32>, Vec<u32>) {
        let a = Arch {
            vocab: 11,
            ctx: 5,
            n_emb: 8,
            heads: 2,
            layers: 2,
        };
        let mut s = 0x1234_5678u32;
        let mut rnd = || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s as f32 / u32::MAX as f32) - 0.5
        };
        let params = a
            .tensors()
            .iter()
            .map(|t| {
                let ones = t.name.ends_with("ln1.weight")
                    || t.name.ends_with("ln2.weight")
                    || t.name == "ln.weight";
                (0..t.elements())
                    .map(|_| if ones { 1.0 + 0.3 * rnd() } else { 0.8 * rnd() })
                    .collect()
            })
            .collect();
        let inputs = vec![3, 1, 4, 1, 5, 9, 2, 6, 5, 3];
        let targets = vec![1, 4, 1, 5, 9, 2, 6, 5, 3, 5];
        (a, params, inputs, targets)
    }

    #[test]
    fn gradient_matches_finite_differences() {
        let (a, params, inputs, targets) = tiny();
        let batch = Batch {
            inputs: &inputs,
            targets: &targets,
            batch: 2,
            len: 5,
        };
        let (_, grads) = loss_and_grad(&a, &params, &batch).unwrap();
        // Central differences in f32 need a large h; check every tensor at a few elements.
        let h = 2e-3f32;
        let mut worst = 0.0f64;
        for (ti, p) in params.iter().enumerate() {
            for &ei in &[0usize, p.len() / 2, p.len() - 1] {
                let mut plus = params.clone();
                plus[ti][ei] += h;
                let mut minus = params.clone();
                minus[ti][ei] -= h;
                let lp = loss_and_grad(&a, &plus, &batch).unwrap().0 as f64;
                let lm = loss_and_grad(&a, &minus, &batch).unwrap().0 as f64;
                let fd = (lp - lm) / (2.0 * h as f64);
                let g = grads[ti][ei] as f64;
                let err = (fd - g).abs() / (1e-2 + g.abs());
                worst = worst.max(err);
                assert!(err < 2e-2, "tensor {ti} elem {ei}: fd {fd} grad {g}");
            }
        }
        assert!(worst < 2e-2);
    }

    #[test]
    fn bad_inputs_are_errors() {
        let (a, params, inputs, mut targets) = tiny();
        targets[0] = 11; // outside the vocabulary
        let batch = Batch {
            inputs: &inputs,
            targets: &targets,
            batch: 2,
            len: 5,
        };
        assert!(loss_and_grad(&a, &params, &batch).is_err());
        let short = &params[..27];
        let ok = Batch {
            inputs: &inputs,
            targets: &inputs,
            batch: 2,
            len: 5,
        };
        assert!(loss_and_grad(&a, short, &ok).is_err());
        let long = Batch {
            inputs: &inputs,
            targets: &inputs,
            batch: 1,
            len: 10, // > ctx
        };
        assert!(loss_and_grad(&a, &params, &long).is_err());
    }
}
