//! C-155 point 1 (T142), SIMULARE: P3 — the glued list — ready to be bound by the OS. The OS sends every token of
//! a text through `kronos_diag::send_token` (the plain list of 16 launches per layer). Here the same text runs as
//! the OS runs it — prompt, then greedy tokens, each token followed by the copy of the logits down into RAM — through
//! `kronos_diag::send_token_list(TokenList::Fused, …)`, in the way the OS uses (`EachLaunch`) and in `PerLayer`,
//! with the driver's diet on (pushbuffer through a shadow, records not zeroed, C-154). What must hold:
//! - the RAM logits of the glued list are the plain list's **bit for bit**, at every position;
//! - the CPU oracle (`gpu_compute::model::step_cpu`) judges every position `Trece` (same greedy token, every logit
//!   within `LOGIT_ABS + LOGIT_REL·|cpu|`), and the generated tokens are the oracle's;
//! - a planted wrong program (no GELU in the glued tail) is caught by the oracle.
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about the board.

mod common;

use common::sys_write;
use common::token_sim::TokenSim;
use gpu_compute::layer::Shape;
use gpu_compute::model::{self, Caches};
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenList, Way};
use nvidia_ga104::kronos_fast::WriteShadow;
use nvidia_ga104::kronos_layer::{self as kl, Prog};
use nvidia_ga104::kronos_model as km;

const PROMPT: [usize; 3] = [5, 200, 17];

fn model() -> km::OwnedModel {
    km::test_model(
        Shape {
            e: 64,
            heads: 4,
            eps: 1e-5,
        },
        3,
        256,
        16,
        11,
    )
    .unwrap()
}

/// One token as the OS runs it for the texts: the list, the copy of the logits down into RAM (filled with 0xEE
/// before), sent in `way`; the RAM logits.
fn token(s: &mut TokenSim<'_>, list: TokenList, tok: usize, pos: usize, way: Way) -> Vec<f32> {
    let (m, vas) = (s.m, s.vas.clone());
    let bytes = (4 * m.vocab as u64).next_multiple_of(16);
    let down = kl::copy_launch(vas.logits, s.host.va, bytes).unwrap();
    let host = s.host.clone();
    sys_write(&mut s.r.sys, &host, &vec![0xee; bytes as usize]);
    kd::send_token_list(s, list, &m.weights(), &vas, tok, pos, way, &[down]).unwrap();
    s.ram_logits()
}

/// The prompt then greedy tokens up to the context: (token fed, RAM logits) per position.
fn text(s: &mut TokenSim<'_>, list: TokenList, way: Way) -> Vec<(usize, Vec<f32>)> {
    let ctx = s.m.context;
    let mut out = Vec::new();
    let mut tok = PROMPT[0];
    for pos in 0..ctx {
        let logits = token(s, list, tok, pos, way);
        out.push((tok, logits.clone()));
        tok = if pos + 1 < PROMPT.len() {
            PROMPT[pos + 1]
        } else {
            model::argmax(&logits).unwrap()
        };
    }
    out
}

fn diet(s: &mut TokenSim<'_>) {
    s.push_shadow = Some(WriteShadow::new(s.r.push.pa, s.r.push.size).unwrap());
    s.zero_records = false;
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn glued_text_matches_plain_and_oracle() {
    let m = model();
    for mode in [SeqMode::EachLaunch, SeqMode::PerLayer] {
        let way = Way::of(mode);
        let mut plain = TokenSim::new(&m, 0);
        let mut glued = TokenSim::new(&m, 0);
        diet(&mut glued);
        let a = text(&mut plain, TokenList::Plain, way);
        let b = text(&mut glued, TokenList::Fused, way);
        // The oracle on the CPU, fed the same tokens.
        let w = m.weights();
        let mut c = Caches::new(m.layers.len(), m.shape.e);
        let mut max_abs = 0.0f32;
        for (pos, ((ta, la), (tb, lb))) in a.iter().zip(&b).enumerate() {
            assert_eq!(ta, tb, "{mode:?}: simbolul dat la pozitia {pos}");
            assert_eq!(bits(la), bits(lb), "{mode:?}: logit-urile la pozitia {pos}");
            let cpu = model::step_cpu(&w, &mut c, *tb, pos).unwrap();
            let v = km::judge(lb, &cpu);
            assert!(v.passes(), "{mode:?}: {}", v.line(pos));
            if let km::TokenVerdict::Trece { max_abs: d, .. } = v {
                max_abs = max_abs.max(d);
            }
        }
        // Submissions per token: one (EachLaunch) or one per layer + the tail (PerLayer).
        let per_token = match mode {
            SeqMode::PerLayer => m.layers.len() + 1,
            _ => 1,
        };
        assert_eq!(glued.submissions.len(), m.context * per_token);
        let n = TokenList::Fused.launches_per_token(m.layers.len()) + 1;
        assert_eq!(glued.submissions.iter().sum::<usize>(), m.context * n);
        // The diet held: no record zeroed, and after the first two tokens only a few words a token.
        assert!(glued.driver_words.iter().all(|x| x.2 == 0));
        eprintln!(
            "{mode:?}: {} pozitii, {} lansari pe simbol (simpla {}), abaterea maxima fata de procesor {max_abs:e}",
            m.context,
            n,
            TokenList::Plain.launches_per_token(m.layers.len()) + 1
        );
    }
}

/// The ranges the glued list is cut into per layer: the embedding rides with layer 0, 7 launches a layer, the tail
/// (final LayerNorm, logits, copy) last; the plain list's are unchanged.
#[test]
fn per_layer_cuts_and_push_bytes() {
    let f = kd::submissions_of(TokenList::Fused, SeqMode::PerLayer, 2, 18);
    assert_eq!(f, vec![0..8, 8..15, 15..18]);
    assert_eq!(
        kd::submissions_of(TokenList::Plain, SeqMode::PerLayer, 2, 36),
        kd::submissions(SeqMode::PerLayer, 2, 36)
    );
    assert_eq!(
        Way::of(SeqMode::PerLayer)
            .with_driver_embedding()
            .submissions_of(TokenList::Fused, 2, 18),
        vec![1..8, 8..15, 15..18]
    );
    // #513 (12 layers) with the copy of the logits: 88 glued launches against 196.
    let (nf, np) = (
        TokenList::Fused.launches_per_token(12) + 1,
        TokenList::Plain.launches_per_token(12) + 1,
    );
    assert_eq!((nf, np), (88, 196));
    let (bf, bp) = (
        kd::push_bytes_of(TokenList::Fused, SeqMode::EachLaunch, 12, nf),
        kd::push_bytes(SeqMode::EachLaunch, 12, np),
    );
    assert_eq!(bp, 9440);
    assert!(bf < bp, "{bf} >= {bp}");
    assert_eq!(bf, 4 * kd::words(SeqMode::EachLaunch, nf));
    eprintln!("#513: pushbuffer pe simbol (EachLaunch) {bf} B lipit fata de {bp} B");
    assert_eq!(TokenList::Fused.programs().len(), 8);
    assert!(TokenList::Fused.programs().contains(&Prog::Q8Fused));
    assert!(!TokenList::Plain.programs().contains(&Prog::Q8Fused));
}

/// A glued program that runs but writes nothing (a launch the board would skip) — planted in the GPU model.
fn fused_writes_nothing(_: &mut gpu_sim::cmdproc::Launch<'_>) -> gpu_core::error::GpuResult<()> {
    Ok(())
}

/// A wrong glued program (one that writes nothing) gives a text the oracle refuses.
#[test]
fn planted_wrong_fused_program_is_caught() {
    let m = model();
    let mut s = TokenSim::new(&m, 0);
    for p in s.models.iter_mut() {
        if p.0 == Prog::Q8Fused {
            p.1 = fused_writes_nothing;
        }
    }
    let w = m.weights();
    let mut c = Caches::new(m.layers.len(), m.shape.e);
    let mut caught = false;
    for (pos, tok) in PROMPT.into_iter().enumerate() {
        let got = token(&mut s, TokenList::Fused, tok, pos, Way::REPAIR);
        let cpu = model::step_cpu(&w, &mut c, tok, pos).unwrap();
        caught |= !km::judge(&got, &cpu).passes();
    }
    assert!(caught, "programul gresit a trecut de oracol");
}
