//! C-83 I14 — one whole token of Kronos (`kronos_gpt`, KGR1 Q8_0): embedding (token row + position row, both Q8_0),
//! every layer ([`crate::layer`]), the final LayerNorm, the logits with the tied head (`emb_token`, Q8_0) — the CPU
//! engine's order ([`step_cpu`]) and the model of our embedding program ([`embed_gpu`]).
//!
//! The verdict for a token, fixed before any run on the board: the greedy token (argmax, the lowest index on a
//! tie) must be the CPU's, and every logit within `LOGIT_ABS + LOGIT_REL·|logit|` (1e-3: the hub's MODEL.md says
//! deviations of that order keep #477's greedy tokens; the smallest top-1/top-2 gap it lists is 0.016).
//!
//! Nothing here runs on a GPU.

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::layer::{self, LayerWeights, Shape};
use crate::math::{fma_f32_ftz, ftz};
use crate::q8;

pub const LOGIT_ABS: f32 = 1e-3;
pub const LOGIT_REL: f32 = 1e-3;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// A model's weights (the order of KGR1's tensors).
pub struct ModelWeights<'a> {
    pub shape: Shape,
    pub vocab: usize,
    pub context: usize,
    /// [vocab, E] Q8_0 (also the tied head).
    pub emb_tok: &'a [u8],
    /// [context, E] Q8_0.
    pub emb_pos: &'a [u8],
    pub layers: Vec<LayerWeights<'a>>,
    pub ln_g: &'a [f32],
    pub ln_b: &'a [f32],
}

impl ModelWeights<'_> {
    pub fn check(&self) -> GpuResult<()> {
        let e = self.shape.e;
        if self.emb_tok.len() != q8::matrix_bytes(self.vocab, e)?
            || self.emb_pos.len() != q8::matrix_bytes(self.context, e)?
            || self.ln_g.len() != e
            || self.ln_b.len() != e
            || self.layers.is_empty()
            || self.context > layer::MAX_POS
        {
            return Err(bad("model: marimi nepotrivite"));
        }
        self.layers.iter().try_for_each(|l| self.shape.check(l))
    }

    /// Bytes from the start of a Q8_0 table to row `r`.
    pub fn row_offset(&self, r: usize) -> usize {
        r * self.shape.e / q8::BLOCK * q8::BLOCK_BYTES
    }
}

fn dequant(row: &[u8], i: usize) -> (f32, f32) {
    let b = &row[i / q8::BLOCK * q8::BLOCK_BYTES..];
    (
        f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        b[4 + i % q8::BLOCK] as i8 as f32,
    )
}

/// The caches of every layer (`[layer][position][E]`, keys and values).
pub struct Caches {
    pub k: Vec<Vec<f32>>,
    pub v: Vec<Vec<f32>>,
}

impl Caches {
    pub fn new(layers: usize, e: usize) -> Caches {
        Caches {
            k: (0..layers)
                .map(|_| alloc::vec![0.0; layer::MAX_POS * e])
                .collect(),
            v: (0..layers)
                .map(|_| alloc::vec![0.0; layer::MAX_POS * e])
                .collect(),
        }
    }
}

/// One token at `pos` in the CPU engine's order; returns the logits (vocab values).
pub fn step_cpu(
    m: &ModelWeights<'_>,
    c: &mut Caches,
    token: usize,
    pos: usize,
) -> GpuResult<Vec<f32>> {
    m.check()?;
    if token >= m.vocab || pos >= m.context {
        return Err(bad("model: simbol sau pozitie in afara tabelelor"));
    }
    let e = m.shape.e;
    let (tr, pr) = (
        &m.emb_tok[m.row_offset(token)..],
        &m.emb_pos[m.row_offset(pos)..],
    );
    let mut x: Vec<f32> = (0..e)
        .map(|i| {
            let ((st, qt), (sp, qp)) = (dequant(tr, i), dequant(pr, i));
            st * qt + sp * qp
        })
        .collect();
    for (i, l) in m.layers.iter().enumerate() {
        layer::step_cpu(&m.shape, l, &mut x, &mut c.k[i], &mut c.v[i], pos)?;
    }
    let h = layer::layernorm_cpu(&x, m.ln_g, m.ln_b, m.shape.eps);
    let mut logits = alloc::vec![0.0f32; m.vocab];
    q8::matvec_cpu(m.emb_tok, m.vocab, e, &h, &mut logits)?;
    Ok(logits)
}

/// `embed.comp` as NAK compiled it: `FFMA.FTZ(scale_tok, q_tok, FMUL.FTZ(scale_pos, q_pos))` (`embed.sm86.txt`).
pub fn embed_gpu(tok_row: &[u8], pos_row: &[u8], n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let ((st, qt), (sp, qp)) = (dequant(tok_row, i), dequant(pos_row, i));
            ftz(fma_f32_ftz(st, qt, layer::mul_ftz(sp, qp)))
        })
        .collect()
}

/// The greedy token: the largest logit, the lowest index on a tie; `None` if any logit is not finite.
pub fn argmax(logits: &[f32]) -> Option<usize> {
    if logits.is_empty() || logits.iter().any(|v| !v.is_finite()) {
        return None;
    }
    Some(
        logits
            .iter()
            .enumerate()
            .fold(0, |b, (i, &v)| if v > logits[b] { i } else { b }),
    )
}

/// The first logit outside the token tolerance: `(index, gpu, cpu)`.
pub fn first_logit_outside(got: &[f32], cpu: &[f32]) -> Option<(usize, f32, f32)> {
    (0..got.len().max(cpu.len())).find_map(|i| {
        let g = got.get(i).copied().unwrap_or(f32::NAN);
        let c = cpu.get(i).copied().unwrap_or(f32::NAN);
        let ok = g.is_finite() && c.is_finite() && (g - c).abs() <= LOGIT_ABS + LOGIT_REL * c.abs();
        (!ok).then_some((i, g, c))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_and_tolerance() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]), Some(1));
        assert_eq!(argmax(&[1.0, f32::NAN]), None);
        assert_eq!(argmax(&[]), None);
        assert!(first_logit_outside(&[10.0], &[10.01]).is_none());
        assert!(first_logit_outside(&[10.0], &[10.02]).is_some());
        assert!(first_logit_outside(&[1.0], &[1.0, 2.0]).is_some());
    }

    #[test]
    fn embed_model_is_close_to_the_cpu_sum() {
        let mut row = [0u8; 72];
        let xs: Vec<f32> = (0..64).map(|i| (i as f32 - 30.0) / 40.0).collect();
        q8::quantize_row(&xs, &mut row).unwrap();
        let g = embed_gpu(&row, &row, 64);
        for (i, v) in g.iter().enumerate() {
            let (s, q) = dequant(&row, i);
            assert!((v - 2.0 * s * q).abs() <= 1e-7, "{i}");
        }
    }
}
