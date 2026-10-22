//! Reference operators in f32 (row-major, contiguous). Clarity over speed:
//! this is the oracle that GPU kernels are diffed against, and the first
//! execution path for Kronos W2 inference on the CPU backend.
//!
//! Shapes are passed explicitly and checked; a mismatch is an error, never a
//! partial computation.

use alloc::vec;
use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::math;

fn need(cond: bool, what: &'static str) -> GpuResult<()> {
    if cond {
        Ok(())
    } else {
        Err(GpuError::InvalidArgument(what))
    }
}

/// out = a + b (elementwise)
pub fn add(a: &[f32], b: &[f32], out: &mut [f32]) -> GpuResult<()> {
    need(
        a.len() == b.len() && a.len() == out.len(),
        "add: lungimi diferite",
    )?;
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
    Ok(())
}

/// x[rows, cols] += bias[cols]
pub fn bias_add(x: &mut [f32], bias: &[f32], rows: usize, cols: usize) -> GpuResult<()> {
    need(x.len() == rows * cols && bias.len() == cols, "bias: forme")?;
    for r in 0..rows {
        for c in 0..cols {
            x[r * cols + c] += bias[c];
        }
    }
    Ok(())
}

/// out[m, n] = a[m, k] * b[k, n]
pub fn matmul(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> GpuResult<()> {
    need(
        a.len() == m * k && b.len() == k * n && out.len() == m * n,
        "matmul: forme",
    )?;
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Ok(())
}

/// Per-row layer normalization with affine gamma/beta (biased variance, as PyTorch).
pub fn layernorm(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    out: &mut [f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> GpuResult<()> {
    need(
        x.len() == rows * cols && out.len() == x.len(),
        "layernorm: forme",
    )?;
    need(
        gamma.len() == cols && beta.len() == cols && cols > 0,
        "layernorm: gamma/beta",
    )?;
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv = 1.0 / math::sqrt(var + eps);
        for c in 0..cols {
            out[r * cols + c] = (row[c] - mean) * inv * gamma[c] + beta[c];
        }
    }
    Ok(())
}

/// GELU, tanh approximation (GPT-2 / `approximate="tanh"`).
pub fn gelu_tanh(x: &mut [f32]) {
    const K: f32 = 0.797_884_6; // sqrt(2/pi)
    for v in x.iter_mut() {
        let u = *v;
        *v = 0.5 * u * (1.0 + math::tanh(K * (u + 0.044715 * u * u * u)));
    }
}

/// GELU, exact erf form (PyTorch default `approximate="none"`).
pub fn gelu_erf(x: &mut [f32]) {
    for v in x.iter_mut() {
        let u = *v;
        *v = 0.5 * u * (1.0 + math::erf(u * core::f32::consts::FRAC_1_SQRT_2));
    }
}

/// Numerically stable softmax over each row of x[rows, cols]. When
/// `causal` is set, row r only sees columns 0..=r (others become 0).
pub fn softmax_rows(x: &mut [f32], rows: usize, cols: usize, causal: bool) -> GpuResult<()> {
    need(x.len() == rows * cols && cols > 0, "softmax: forme")?;
    need(!causal || rows <= cols, "softmax cauzal: rows > cols")?;
    for r in 0..rows {
        let row = &mut x[r * cols..(r + 1) * cols];
        let visible = if causal { r + 1 } else { cols };
        let max = row[..visible]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for v in row[..visible].iter_mut() {
            *v = math::exp(*v - max);
            sum += *v;
        }
        for v in row[..visible].iter_mut() {
            *v /= sum;
        }
        for v in row[visible..].iter_mut() {
            *v = 0.0;
        }
    }
    Ok(())
}

/// out[t, d] = table[ids[t], d]; an id outside the vocabulary is an error.
pub fn embedding(
    ids: &[u32],
    table: &[f32],
    vocab: usize,
    dim: usize,
    out: &mut [f32],
) -> GpuResult<()> {
    need(
        table.len() == vocab * dim && out.len() == ids.len() * dim,
        "embedding: forme",
    )?;
    for (t, &id) in ids.iter().enumerate() {
        let id = id as usize;
        if id >= vocab {
            return Err(GpuError::OutOfBounds {
                offset: id as u64,
                len: 1,
                limit: vocab as u64,
            });
        }
        out[t * dim..(t + 1) * dim].copy_from_slice(&table[id * dim..(id + 1) * dim]);
    }
    Ok(())
}

/// Multi-head causal self-attention core: q, k, v and out are [T, H*D]
/// (heads interleaved as in a fused QKV projection split). For each head h:
/// out_h = softmax_causal(q_h k_h^T / sqrt(D)) v_h.
pub fn causal_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    t: usize,
    heads: usize,
    d: usize,
) -> GpuResult<()> {
    let w = heads * d;
    need(
        q.len() == t * w && k.len() == t * w && v.len() == t * w && out.len() == t * w,
        "atentie: forme",
    )?;
    need(d > 0 && heads > 0, "atentie: dimensiuni")?;
    let scale = 1.0 / math::sqrt(d as f32);
    let mut scores = vec![0.0f32; t * t];
    for h in 0..heads {
        for i in 0..t {
            for j in 0..t {
                let mut s = 0.0;
                for c in 0..d {
                    s += q[i * w + h * d + c] * k[j * w + h * d + c];
                }
                scores[i * t + j] = s * scale;
            }
        }
        softmax_rows(&mut scores, t, t, true)?;
        for i in 0..t {
            for c in 0..d {
                let mut acc = 0.0;
                for j in 0..=i {
                    acc += scores[i * t + j] * v[j * w + h * d + c];
                }
                out[i * w + h * d + c] = acc;
            }
        }
    }
    Ok(())
}

/// Mean cross-entropy of logits[rows, vocab] against targets (used by exams
/// and, later, training checks).
pub fn cross_entropy(logits: &[f32], targets: &[u32], vocab: usize) -> GpuResult<f32> {
    need(
        logits.len() == targets.len() * vocab && vocab > 0 && !targets.is_empty(),
        "cross-entropy: forme",
    )?;
    let mut total = 0.0f64;
    let mut row: Vec<f32> = vec![0.0; vocab];
    for (r, &tgt) in targets.iter().enumerate() {
        if tgt as usize >= vocab {
            return Err(GpuError::OutOfBounds {
                offset: tgt as u64,
                len: 1,
                limit: vocab as u64,
            });
        }
        row.copy_from_slice(&logits[r * vocab..(r + 1) * vocab]);
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row.iter().map(|&x| math::exp(x - max) as f64).sum();
        // log(sum) via ln(sum) = ln(m * 2^e); small helper below.
        total += ln(sum) + max as f64 - row[tgt as usize] as f64;
    }
    Ok((total / targets.len() as f64) as f32)
}

/// Natural log in f64 for positive finite x (used only by cross_entropy).
fn ln(x: f64) -> f64 {
    debug_assert!(x > 0.0 && x.is_finite());
    let bits = x.to_bits();
    let e = ((bits >> 52) & 0x7ff) as i64 - 1023;
    let m = f64::from_bits((bits & !(0x7ffu64 << 52)) | (1023u64 << 52)); // [1, 2)
    // ln(m) = 2 atanh((m-1)/(m+1)), series converges fast for m in [1,2).
    let z = (m - 1.0) / (m + 1.0);
    let z2 = z * z;
    let mut term = z;
    let mut sum = 0.0;
    let mut n = 1.0;
    for _ in 0..30 {
        sum += term / n;
        term *= z2;
        n += 2.0;
    }
    2.0 * sum + e as f64 * core::f64::consts::LN_2
}

#[cfg(test)]
mod tests {
    //! Differential tests: the expected values come from an independent
    //! f64 implementation written with `std` math, or from hand computation.
    use super::*;

    fn close(a: &[f32], b: &[f64], tol: f64) {
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (*x as f64 - y).abs() <= tol * (1.0 + y.abs()),
                "index {i}: {x} vs {y}"
            );
        }
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn rand_vec(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n).map(|_| lcg(seed)).collect()
    }

    #[test]
    fn small_hand_cases() {
        let mut o = [0.0; 4];
        matmul(&[1., 2., 3., 4.], &[5., 6., 7., 8.], &mut o, 2, 2, 2).unwrap();
        assert_eq!(o, [19., 22., 43., 50.]);
        add(&[1., 2.], &[3., 4.], &mut o[..2]).unwrap();
        assert_eq!(o[..2], [4., 6.]);
        let mut x = [1., 1., 1., 1.];
        bias_add(&mut x, &[1., 2.], 2, 2).unwrap();
        assert_eq!(x, [2., 3., 2., 3.]);
        assert!(matmul(&[1.], &[1.], &mut o, 2, 2, 2).is_err());
    }

    #[test]
    fn matmul_vs_f64() {
        let mut s = 1;
        let (m, k, n) = (17, 33, 9);
        let a = rand_vec(m * k, &mut s);
        let b = rand_vec(k * n, &mut s);
        let mut o = vec![0.0; m * n];
        matmul(&a, &b, &mut o, m, k, n).unwrap();
        let mut r = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                r[i * n + j] = (0..k)
                    .map(|p| a[i * k + p] as f64 * b[p * n + j] as f64)
                    .sum();
            }
        }
        close(&o, &r, 1e-5);
    }

    #[test]
    fn layernorm_vs_f64() {
        let mut s = 2;
        let (rows, cols) = (5, 768);
        let x = rand_vec(rows * cols, &mut s);
        let g = rand_vec(cols, &mut s);
        let bta = rand_vec(cols, &mut s);
        let mut o = vec![0.0; rows * cols];
        layernorm(&x, &g, &bta, &mut o, rows, cols, 1e-5).unwrap();
        let mut r = vec![0.0f64; rows * cols];
        for i in 0..rows {
            let row: Vec<f64> = x[i * cols..(i + 1) * cols]
                .iter()
                .map(|&v| v as f64)
                .collect();
            let mean = row.iter().sum::<f64>() / cols as f64;
            let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / cols as f64;
            for c in 0..cols {
                r[i * cols + c] =
                    (row[c] - mean) / (var + 1e-5).sqrt() * g[c] as f64 + bta[c] as f64;
            }
        }
        close(&o, &r, 1e-4);
    }

    #[test]
    fn gelu_vs_f64() {
        let mut s = 3;
        let x = rand_vec(1000, &mut s)
            .iter()
            .map(|v| v * 6.0)
            .collect::<Vec<_>>();
        let mut a = x.clone();
        gelu_tanh(&mut a);
        let r: Vec<f64> = x
            .iter()
            .map(|&u| {
                let u = u as f64;
                0.5 * u
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * (u + 0.044715 * u.powi(3))).tanh())
            })
            .collect();
        close(&a, &r, 1e-5);
        let mut e = x.clone();
        gelu_erf(&mut e);
        // erf form and tanh form agree within ~1e-3 (known property of the approximation).
        for (p, q) in e.iter().zip(&a) {
            assert!((p - q).abs() < 2e-3);
        }
    }

    #[test]
    fn softmax_properties() {
        let mut s = 4;
        let (rows, cols) = (6, 6);
        let mut x = rand_vec(rows * cols, &mut s)
            .iter()
            .map(|v| v * 30.0)
            .collect::<Vec<_>>();
        let orig = x.clone();
        softmax_rows(&mut x, rows, cols, true).unwrap();
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
            assert!(row[r + 1..].iter().all(|&v| v == 0.0), "causal mask");
            let o = &orig[r * cols..(r + 1) * cols];
            let max = o[..=r]
                .iter()
                .cloned()
                .fold(f64::NEG_INFINITY as f32, f32::max) as f64;
            let den: f64 = o[..=r].iter().map(|&v| (v as f64 - max).exp()).sum();
            for c in 0..=r {
                assert!((row[c] as f64 - (o[c] as f64 - max).exp() / den).abs() < 1e-6);
            }
        }
        // Large logits do not overflow.
        let mut big = [1000.0f32, 1000.0];
        softmax_rows(&mut big, 1, 2, false).unwrap();
        assert_eq!(big, [0.5, 0.5]);
    }

    #[test]
    fn embedding_bounds() {
        let table = [0., 1., 10., 11., 20., 21.];
        let mut o = [0.0; 4];
        embedding(&[2, 0], &table, 3, 2, &mut o).unwrap();
        assert_eq!(o, [20., 21., 0., 1.]);
        assert!(matches!(
            embedding(&[3, 0], &table, 3, 2, &mut o),
            Err(GpuError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn attention_vs_f64() {
        let mut s = 5;
        let (t, h, d) = (7, 3, 8);
        let w = h * d;
        let q = rand_vec(t * w, &mut s);
        let k = rand_vec(t * w, &mut s);
        let v = rand_vec(t * w, &mut s);
        let mut o = vec![0.0; t * w];
        causal_attention(&q, &k, &v, &mut o, t, h, d).unwrap();
        let mut r = vec![0.0f64; t * w];
        for hh in 0..h {
            for i in 0..t {
                let sc: Vec<f64> = (0..=i)
                    .map(|j| {
                        (0..d)
                            .map(|c| q[i * w + hh * d + c] as f64 * k[j * w + hh * d + c] as f64)
                            .sum::<f64>()
                            / (d as f64).sqrt()
                    })
                    .collect();
                let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let den: f64 = sc.iter().map(|x| (x - m).exp()).sum();
                for c in 0..d {
                    r[i * w + hh * d + c] = (0..=i)
                        .map(|j| (sc[j] - m).exp() / den * v[j * w + hh * d + c] as f64)
                        .sum();
                }
            }
        }
        close(&o, &r, 1e-5);
        // Row 0 only attends to itself: out = v[0].
        assert_eq!(o[..w], v[..w]);
    }

    #[test]
    fn cross_entropy_vs_f64() {
        let logits = [2.0f32, 1.0, 0.1, 0.5, 2.5, -1.0];
        let ce = cross_entropy(&logits, &[0, 1], 3).unwrap();
        let row = |r: &[f32], t: usize| {
            let den: f64 = r.iter().map(|&x| (x as f64).exp()).sum();
            -((r[t] as f64).exp() / den).ln()
        };
        let want = (row(&logits[..3], 0) + row(&logits[3..], 1)) / 2.0;
        assert!((ce as f64 - want).abs() < 1e-6);
        assert!((ln(10.0) - 10f64.ln()).abs() < 1e-12);
        assert!(cross_entropy(&logits, &[0, 3], 3).is_err());
    }
}
