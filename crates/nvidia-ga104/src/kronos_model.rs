//! C-83 I14 — one whole token of Kronos on the work channel: the embedding (`embed.comp`: token row + position row,
//! both Q8_0), every layer (I13, [`crate::kronos_layer::token_launches`]), the final LayerNorm and the logits with
//! the tied head (`q8_matvec` over `emb_token`). `16·layers + 3` launches per token (195 for #513's 12 layers); all
//! layers share one activation scratch (the launches run one after the other), each layer has its own weights and
//! caches. The greedy token is read on the CPU from the logits (64 KiB for a vocabulary of 16 384).
//!
//! The verdict ([`judge`], fixed before any run): the greedy token must be the CPU engine's and every logit within
//! `gpu_compute::model::{LOGIT_ABS, LOGIT_REL}`.
//!
//! Tested **only in SIMULARE** on a small synthetic model; #513 itself is read by the OS from the stick at the
//! window. Nothing here touches a board.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::layer::{LayerWeights, Shape};
use gpu_compute::model::{self, ModelWeights};
use gpu_compute::q8;
use gpu_core::error::{GpuError, GpuResult};

use crate::kronos_layer::{self as kl, Launch, LayerVas, OwnedLayer, Prog};

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// GPU VAs of a whole model. `layers[i]` all use the same scratch (same `x`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModelVas {
    pub emb_tok: u64,
    pub emb_pos: u64,
    pub ln_g: u64,
    pub ln_b: u64,
    pub layers: Vec<LayerVas>,
    /// The final LayerNorm's output (E values) and the logits (vocab values).
    pub h_final: u64,
    pub logits: u64,
}

impl ModelVas {
    pub fn check(&self) -> GpuResult<()> {
        let Some(first) = self.layers.first() else {
            return Err(bad("model: fara straturi"));
        };
        if self.layers.iter().any(|l| l.x != first.x) {
            return Err(bad("model: straturile trebuie sa imparta acelasi x"));
        }
        Ok(())
    }
}

/// Launches per token for `layers` layers.
pub const fn launches_per_token(layers: usize) -> usize {
    kl::LAUNCHES_PER_TOKEN * layers + 3
}

/// Bytes of the one pushbuffer stretch a token needs
/// ([`crate::transfer::encode_sequence`] of [`launches_per_token`]): 7 832 for
/// #513's 12 layers. The pushbuffer area must hold at least this; twice this
/// lets the next token be written while the previous one is still in flight.
pub const fn push_bytes_per_token(layers: usize) -> usize {
    4 * crate::transfer::sequence_words(launches_per_token(layers))
}

/// The launches of one token at `pos` (`token` < vocab, `pos` < context); the logits land at `v.logits`.
pub fn token_launches(
    s: &Shape,
    vocab: usize,
    context: usize,
    v: &ModelVas,
    token: usize,
    pos: usize,
) -> GpuResult<Vec<Launch>> {
    v.check()?;
    if token >= vocab || pos >= context || context > gpu_compute::layer::MAX_POS {
        return Err(bad("model: simbol sau pozitie in afara tabelelor"));
    }
    let e = s.e;
    let row = (e / q8::BLOCK * q8::BLOCK_BYTES) as u64;
    let x = v.layers[0].x;
    let mut out = Vec::with_capacity(launches_per_token(v.layers.len()));
    out.push(Launch {
        prog: Prog::Embed,
        cbuf: kl::cbuf(
            &[
                v.emb_tok + token as u64 * row,
                v.emb_pos + pos as u64 * row,
                x,
            ],
            &[e as u32],
        ),
        grid: kl::blocks(e, 256),
    });
    for l in &v.layers {
        out.extend(kl::token_launches(s, l, pos)?);
    }
    out.push(Launch {
        prog: Prog::LayerNorm,
        cbuf: kl::cbuf(
            &[x, v.ln_g, v.ln_b, v.h_final],
            &[e as u32, (1.0 / e as f32).to_bits(), s.eps.to_bits()],
        ),
        grid: 1,
    });
    let mut c = [0u8; 256];
    c.copy_from_slice(&crate::q8_matvec::q8_cbuf(
        v.emb_tok, v.h_final, v.logits, vocab, e,
    )?);
    out.push(Launch {
        prog: Prog::Q8Matvec,
        cbuf: c,
        grid: vocab as u32,
    });
    Ok(out)
}

/// A whole model owned (the synthetic test model; the OS builds the same from a KGR1 file).
pub struct OwnedModel {
    pub shape: Shape,
    pub vocab: usize,
    pub context: usize,
    pub emb_tok: Vec<u8>,
    pub emb_pos: Vec<u8>,
    pub layers: Vec<OwnedLayer>,
    pub ln_g: Vec<f32>,
    pub ln_b: Vec<f32>,
}

impl OwnedModel {
    pub fn weights(&self) -> ModelWeights<'_> {
        ModelWeights {
            shape: self.shape,
            vocab: self.vocab,
            context: self.context,
            emb_tok: &self.emb_tok,
            emb_pos: &self.emb_pos,
            layers: self
                .layers
                .iter()
                .map(|l| l.weights())
                .collect::<Vec<LayerWeights<'_>>>(),
            ln_g: &self.ln_g,
            ln_b: &self.ln_b,
        }
    }
}

/// A synthetic model (no file): layers from [`kl::test_layer`], embeddings quantized from the I11 pattern.
pub fn test_model(
    s: Shape,
    layers: usize,
    vocab: usize,
    context: usize,
    seed: u32,
) -> GpuResult<OwnedModel> {
    let e = s.e;
    let emb = |rows: usize, k: u32| {
        crate::q8_matvec::test_matrix(rows, e, seed.wrapping_mul(97).wrapping_add(k)).map(|m| m.0)
    };
    let l0 = kl::test_layer(e, seed)?;
    Ok(OwnedModel {
        shape: s,
        vocab,
        context,
        emb_tok: emb(vocab, 1)?,
        emb_pos: emb(context, 2)?,
        layers: (0..layers)
            .map(|i| kl::test_layer(e, seed.wrapping_add(1000 + i as u32)))
            .collect::<GpuResult<Vec<_>>>()?,
        ln_g: l0.ln1_g.clone(),
        ln_b: l0.ln2_b.clone(),
    })
}

/// What the board's logits say.
#[derive(Clone, Debug, PartialEq)]
pub enum TokenVerdict {
    Trece {
        token: usize,
        max_abs: f32,
    },
    /// The greedy token differs from the CPU's.
    SimbolDiferit {
        gpu: Option<usize>,
        cpu: Option<usize>,
    },
    /// The same token, but a logit outside the tolerance.
    LogitInAfara {
        index: usize,
        got: f32,
        cpu: f32,
    },
}

impl TokenVerdict {
    pub fn passes(&self) -> bool {
        matches!(self, TokenVerdict::Trece { .. })
    }

    pub fn line(&self, pos: usize) -> String {
        match self {
            TokenVerdict::Trece { token, max_abs } => format!(
                "i14: pozitia {pos}: TRECE - simbolul {token} ca procesorul, abaterea maxima a logit-urilor {max_abs:e}"
            ),
            TokenVerdict::SimbolDiferit { gpu, cpu } => {
                format!(
                    "i14: pozitia {pos}: NU TRECE - simbolul placii {gpu:?}, al procesorului {cpu:?}"
                )
            }
            TokenVerdict::LogitInAfara { index, got, cpu } => format!(
                "i14: pozitia {pos}: NU TRECE - acelasi simbol, dar logit-ul {index}: placa {got:e}, procesor {cpu:e}"
            ),
        }
    }
}

pub fn judge(got: &[f32], cpu: &[f32]) -> TokenVerdict {
    let (g, c) = (model::argmax(got), model::argmax(cpu));
    if g.is_none() || g != c || got.len() != cpu.len() {
        return TokenVerdict::SimbolDiferit { gpu: g, cpu: c };
    }
    if let Some((index, got, cpu)) = model::first_logit_outside(got, cpu) {
        return TokenVerdict::LogitInAfara { index, got, cpu };
    }
    TokenVerdict::Trece {
        token: g.unwrap_or(0),
        max_abs: got
            .iter()
            .zip(cpu)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_list_for_a_token() {
        let s = Shape {
            e: 64,
            heads: 2,
            eps: 1e-5,
        };
        let b = crate::mmu::VA_BASE;
        let w = kl::WeightVas {
            ln1_g: b,
            ln1_b: b,
            wqkv: b,
            bqkv: b,
            wout: b,
            bout: b,
            ln2_g: b,
            ln2_b: b,
            wfc: b,
            bfc: b,
            wproj: b,
            bproj: b,
        };
        let l = |k: u64| {
            LayerVas::with_scratch(
                64,
                b + 0x10_0000,
                b + k * 0x10_0000,
                b + k * 0x10_0000 + 0x8_0000,
                w,
            )
            .unwrap()
        };
        let v = ModelVas {
            emb_tok: b + 0x100_0000,
            emb_pos: b + 0x200_0000,
            ln_g: b,
            ln_b: b,
            layers: alloc::vec![l(2), l(3)],
            h_final: b + 0x300_0000,
            logits: b + 0x300_1000,
        };
        let x = token_launches(&s, 256, 16, &v, 7, 3).unwrap();
        assert_eq!(x.len(), launches_per_token(2));
        assert_eq!(x[0].prog, Prog::Embed);
        assert_eq!(
            u64::from_le_bytes(x[0].cbuf[0x28..0x30].try_into().unwrap()),
            v.emb_tok + 7 * 72
        );
        assert_eq!(
            u64::from_le_bytes(x[0].cbuf[0x30..0x38].try_into().unwrap()),
            v.emb_pos + 3 * 72
        );
        assert_eq!(x.last().unwrap().grid, 256);
        assert!(token_launches(&s, 256, 16, &v, 256, 0).is_err());
        assert!(token_launches(&s, 256, 16, &v, 0, 16).is_err());
        let mut bad = v.clone();
        bad.layers[1].x += 4;
        assert!(token_launches(&s, 256, 16, &bad, 0, 0).is_err());
        assert_eq!(launches_per_token(12), 195);
    }

    #[test]
    fn token_verdicts() {
        assert!(judge(&[0.0, 2.0, 1.0], &[0.0, 2.0005, 1.0]).passes());
        assert!(matches!(
            judge(&[2.0, 1.0], &[1.0, 2.0]),
            TokenVerdict::SimbolDiferit { .. }
        ));
        assert!(matches!(
            judge(&[0.0, 2.0, 1.1], &[0.0, 2.0, 1.0]),
            TokenVerdict::LogitInAfara { index: 2, .. }
        ));
        assert!(!judge(&[f32::NAN, 1.0], &[0.0, 1.0]).passes());
    }
}
