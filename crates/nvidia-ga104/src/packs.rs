//! C-157 point 8 — the compute packs and the choice by model card (Adrian's decisions of 20 Nov 19:41 / 19:45, hub
//! `DECIZII.md`; design: `docs/C-157_P8_FISA_MODELULUI.md`).
//!
//! One base (board start, GSP, memory, the work channel, safety) and **compute packs** per model kind: the programs and
//! the prepared list of launches. [`select`] reads a [`ModelCard`] and returns the first pack that takes it, with its
//! proof: the forms for which "identical to the CPU" passed on the real board, or [`Proof::Needed`] when the form was
//! never proved there (the OS then runs that proof before any work). No pack → a [`Refusal`] with every reason, never a
//! wrong computation.
//!
//! The pack's conditions are the ones the programs already check one by one (`kronos_layer::token_launches`,
//! `kronos_model`, the OS's KGR1 reader); here they are gathered so a card is judged before anything is uploaded.
//! Pure data and checks: nothing here touches a board.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::card::{Arch, ModelCard, Quant};
use gpu_compute::layer::MAX_POS;
use gpu_compute::q8;
use gpu_core::error::GpuResult;

use crate::kronos_diag::TokenList;
use crate::kronos_layer::Prog;
use crate::kronos_speed::ModelLimits;

/// A form proved "identical to the CPU" on the real board with a pack: the card's numbers and the run that shows it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoardProof {
    /// (layers, heads, n_emb, context, vocabulary), as [`ModelCard::form`].
    pub form: (usize, usize, usize, usize, usize),
    /// The model and the run (hub `rezultate/`).
    pub run: &'static str,
}

/// What a pack accepts (the conditions of its programs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Needs {
    pub layers_max: usize,
    /// n_emb must be a multiple of this (the Q8_0 block, 32).
    pub e_multiple: usize,
    /// Values per head at most (the generation attention program).
    pub head_dim_max: usize,
    /// Positions at most (the KV caches of the programs).
    pub context_max: usize,
    pub tied_head: bool,
}

/// A compute pack: fixed data, no code per model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pack {
    pub name: &'static str,
    pub version: u32,
    pub needs: Needs,
    /// The list of launches it runs on the board.
    pub list: TokenList,
    pub proofs: &'static [BoardProof],
}

/// The generation pack of Kronos GPT, Q8_0 weights: `lipita-ln-32` (the default on the board since os-346).
pub const KRONOS_GPT_Q8_GEN: Pack = Pack {
    name: "kronos-gpt-q8-generare",
    version: 1,
    needs: Needs {
        layers_max: 64,
        e_multiple: q8::BLOCK,
        head_dim_max: 128,
        context_max: MAX_POS,
        tied_head: true,
    },
    list: TokenList::FusedLn32,
    proofs: &[
        BoardProof {
            form: (12, 12, 768, 512, 16_384),
            run: "#513, os-346 (22 nov 2026): 16 simboluri identice cu procesorul",
        },
        BoardProof {
            form: (24, 12, 768, 512, 16_384),
            run: "#651, os-355 (22 nov 2026): 16 simboluri identice cu procesorul",
        },
    ],
};

/// Every pack the driver has, in the order [`select`] tries them.
pub const PACKS: &[Pack] = &[KRONOS_GPT_Q8_GEN];

/// One reason a pack does not take a card.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mismatch {
    Arch(String),
    Quant(String),
    Layers { have: usize, max: usize },
    Vocab,
    EMultiple { e: usize, multiple: usize },
    Heads { e: usize, heads: usize },
    HeadDim { have: usize, max: usize },
    Context { have: usize, max: usize },
    TiedHead,
}

impl Mismatch {
    pub fn line(&self) -> String {
        match self {
            Mismatch::Arch(a) => format!("arhitectura `{a}` (stiu doar `kronos_gpt`)"),
            Mismatch::Quant(q) => format!("greutati `{q}` (pachetul cere `q8_0`)"),
            Mismatch::Layers { have, max } => format!("{have} straturi (intre 1 si {max})"),
            Mismatch::Vocab => "vocabular 0".into(),
            Mismatch::EMultiple { e, multiple } => format!("n_emb {e} nu e multiplu de {multiple}"),
            Mismatch::Heads { e, heads } => format!("{heads} capete nu impart n_emb {e}"),
            Mismatch::HeadDim { have, max } => format!("capete de {have} valori (cel mult {max})"),
            Mismatch::Context { have, max } => format!("context {have} (cel mult {max})"),
            Mismatch::TiedHead => "capul nu e legat de emb_token (pachetul il cere legat)".into(),
        }
    }
}

fn quant_name(q: &Quant) -> String {
    match q {
        Quant::F32 => "f32".into(),
        Quant::Q8_0 => "q8_0".into(),
        Quant::Other(s) => s.clone(),
    }
}

impl Pack {
    /// Every reason this pack does not take `c` (empty = it takes it).
    pub fn mismatches(&self, c: &ModelCard) -> Vec<Mismatch> {
        let n = &self.needs;
        let mut m = Vec::new();
        if let Arch::Other(a) = &c.arch {
            m.push(Mismatch::Arch(a.clone()));
        }
        if c.quant != Quant::Q8_0 {
            m.push(Mismatch::Quant(quant_name(&c.quant)));
        }
        if c.layers == 0 || c.layers > n.layers_max {
            m.push(Mismatch::Layers {
                have: c.layers,
                max: n.layers_max,
            });
        }
        if c.vocab == 0 {
            m.push(Mismatch::Vocab);
        }
        if c.e == 0 || !c.e.is_multiple_of(n.e_multiple) {
            m.push(Mismatch::EMultiple {
                e: c.e,
                multiple: n.e_multiple,
            });
        }
        match c.head_dim() {
            None => m.push(Mismatch::Heads {
                e: c.e,
                heads: c.heads,
            }),
            Some(d) if d > n.head_dim_max => m.push(Mismatch::HeadDim {
                have: d,
                max: n.head_dim_max,
            }),
            Some(_) => {}
        }
        if c.context == 0 || c.context > n.context_max {
            m.push(Mismatch::Context {
                have: c.context,
                max: n.context_max,
            });
        }
        if n.tied_head && !c.tied_head {
            m.push(Mismatch::TiedHead);
        }
        m
    }

    /// The board proof for the card's form, if any.
    pub fn proof_for(&self, c: &ModelCard) -> Option<&'static BoardProof> {
        self.proofs.iter().find(|p| p.form == c.form())
    }
}

/// How sure the choice is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Proof {
    /// This form passed "identical to the CPU" on the real board with this pack.
    Board(&'static BoardProof),
    /// The pack takes the card but the form was never proved on the board: run the proof first.
    Needed,
}

/// The pack chosen for a card.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub pack: &'static Pack,
    pub proof: Proof,
    pub card: ModelCard,
}

impl Choice {
    /// The work channel's limits from the card's numbers and the pack's list (`piece` = the largest upload piece).
    pub fn limits(&self, piece: u64) -> GpuResult<ModelLimits> {
        self.limits_with(piece, &[])
    }

    /// C-157 point 8 step 2: [`Self::limits`] with one more launch per program of `extra` after the list (the OS
    /// appends `Prog::Argmax` when the token is chosen on the board).
    pub fn limits_with(&self, piece: u64, extra: &[Prog]) -> GpuResult<ModelLimits> {
        ModelLimits::of_card_lists(&self.card, piece, &[self.pack.list], extra)
    }

    /// One line for the OS console, in Romanian.
    pub fn line(&self) -> String {
        let p = self.pack;
        match self.proof {
            Proof::Board(b) => format!(
                "pachet {} v{}: dovedit pe placa ({})",
                p.name, p.version, b.run
            ),
            Proof::Needed => format!(
                "pachet {} v{}: forma noua, PROBA NECESARA (identic cu procesorul) inainte de lucru",
                p.name, p.version
            ),
        }
    }
}

/// No pack takes the card: every pack's reasons.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Refusal {
    pub reasons: Vec<(&'static str, Vec<Mismatch>)>,
}

impl Refusal {
    /// The lines for the OS console, in Romanian (the first says the work is refused).
    pub fn lines(&self) -> Vec<String> {
        let mut out = alloc::vec![String::from(
            "REFUZ: niciun pachet de calcul nu se potriveste fisei modelului; nimic nu ruleaza pe placa"
        )];
        for (name, ms) in &self.reasons {
            for m in ms {
                out.push(format!("  {name}: {}", m.line()));
            }
        }
        out
    }
}

/// Picks the pack for `card` among `packs` (the first that takes it).
pub fn select_from(packs: &'static [Pack], card: &ModelCard) -> Result<Choice, Refusal> {
    let mut reasons = Vec::new();
    for p in packs {
        let ms = p.mismatches(card);
        if ms.is_empty() {
            let proof = match p.proof_for(card) {
                Some(b) => Proof::Board(b),
                None => Proof::Needed,
            };
            return Ok(Choice {
                pack: p,
                proof,
                card: card.clone(),
            });
        }
        reasons.push((p.name, ms));
    }
    Err(Refusal { reasons })
}

/// Picks the pack for `card` among [`PACKS`].
pub fn select(card: &ModelCard) -> Result<Choice, Refusal> {
    select_from(PACKS, card)
}
