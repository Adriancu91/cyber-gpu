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

fn layernorm_cpu(x: &[f32], g: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
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

fn mul_ftz(a: f32, b: f32) -> f32 {
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
}
