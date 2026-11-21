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

/// C-139: what every launch of one token should leave in memory, with the program models ([`layer::trace_gpu`],
/// [`embed_gpu`], `q8::matvec_gpu_order`): the embedding, every layer's vectors, the final LayerNorm and the logits.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenTrace {
    /// The embedding launch (x before layer 0).
    pub x0: Vec<f32>,
    pub layers: Vec<layer::LayerTrace>,
    pub h_final: Vec<f32>,
    pub logits: Vec<f32>,
}

/// C-139: [`TokenTrace`] of `token` at `pos`. `prev` holds the caches of the positions before `pos` (e.g. the CPU
/// engine's after it ran them); nothing is written.
pub fn trace_gpu(
    m: &ModelWeights<'_>,
    prev: &Caches,
    token: usize,
    pos: usize,
) -> GpuResult<TokenTrace> {
    m.check()?;
    if token >= m.vocab || pos >= m.context {
        return Err(bad("model: simbol sau pozitie in afara tabelelor"));
    }
    if prev.k.len() != m.layers.len() || prev.v.len() != m.layers.len() {
        return Err(bad("model: cache-uri pentru alt numar de straturi"));
    }
    let e = m.shape.e;
    let x0 = embed_gpu(
        &m.emb_tok[m.row_offset(token)..],
        &m.emb_pos[m.row_offset(pos)..],
        e,
    );
    let mut layers = Vec::with_capacity(m.layers.len());
    let mut x = x0.clone();
    for (i, l) in m.layers.iter().enumerate() {
        let t = layer::trace_gpu(&m.shape, l, &x, &prev.k[i], &prev.v[i], pos)?;
        x = t.x_out.clone();
        layers.push(t);
    }
    let h_final = layer::layernorm_gpu(&x, m.ln_g, m.ln_b, m.shape.eps);
    let mut logits = alloc::vec![0.0f32; m.vocab];
    q8::matvec_gpu_order(m.emb_tok, m.vocab, e, &h_final, &mut logits)?;
    Ok(TokenTrace {
        x0,
        layers,
        h_final,
        logits,
    })
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

/// C-155 point 3: "no token" in the board's argmax result (any logit not finite, or no logit).
pub const ARGMAX_NONE: u32 = u32::MAX;

/// C-155 point 3: the 4 words the argmax program (`tools/nak/argmax.comp`) writes for `logits`: the index of the
/// largest logit (the lowest on a tie; [`ARGMAX_NONE`] if any logit is NaN/infinite or there is none — so word 0 is
/// exactly [`argmax`]), the bits of the largest finite logit (−0.0 written as +0.0; −∞ if none), how many logits are
/// not finite (by their exponent bits), and n. Compared exactly, subnormal numbers included (the program compares
/// integer keys of the bits, not floats with FTZ). Picking a maximum does no arithmetic, so this is the program's result for any split of the
/// elements between threads.
pub fn argmax_result(logits: &[f32]) -> [u32; 4] {
    let mut best = f32::NEG_INFINITY;
    let mut at = ARGMAX_NONE;
    let mut nonfinite = 0u32;
    for (i, &v) in logits.iter().enumerate() {
        if v.to_bits() & 0x7f80_0000 == 0x7f80_0000 {
            nonfinite += 1;
        } else if v > best {
            best = v;
            at = i as u32;
        }
    }
    let token = if nonfinite == 0 { at } else { ARGMAX_NONE };
    let bits = if best.to_bits() == 0x8000_0000 {
        0
    } else {
        best.to_bits()
    };
    [token, bits, nonfinite, logits.len() as u32]
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
        // C-155 point 3: word 0 of the board's result is `argmax`, on ties, -0.0 = +0.0, NaN, infinities.
        for v in [
            &[1.0, 3.0, 3.0, 2.0][..],
            &[-0.0, 0.0],
            &[0.0, -0.0, 0.0],
            &[f32::MIN, f32::MIN],
            &[2.0, f32::NAN, 5.0],
            &[f32::INFINITY, 1.0],
            &[f32::NEG_INFINITY],
            &[],
        ] {
            let r = argmax_result(v);
            assert_eq!(r[0], argmax(v).map_or(ARGMAX_NONE, |i| i as u32), "{v:?}");
            assert_eq!(r[3] as usize, v.len());
        }
        assert_eq!(
            argmax_result(&[2.0, f32::NAN, 5.0]),
            [ARGMAX_NONE, 5.0f32.to_bits(), 1, 3]
        );
        assert_eq!(
            argmax_result(&[]),
            [ARGMAX_NONE, f32::NEG_INFINITY.to_bits(), 0, 0]
        );
        assert_eq!(argmax_result(&[-0.0, 0.0]), [0, 0, 0, 2]);
        // Subnormals are ordered exactly (no FTZ): the smallest positive one beats 0.
        let tiny = f32::from_bits(1);
        assert_eq!(argmax_result(&[0.0, tiny, -tiny])[0], 1);
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
