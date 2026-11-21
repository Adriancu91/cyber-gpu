//! C-155 point 2 (T143), SIMULARE: the new LayerNorm (`layernorm256`, 256 threads, one read of `x`) in the glued
//! list (`TokenList::FusedLn`). The same text as T142 — prompt, then greedy tokens, each followed by the copy of the
//! logits down into RAM — with the driver's diet on, in `EachLaunch` and `PerLayer`. What must hold:
//! - every LayerNorm of the token (2 per layer + the final one) is on the new program; nothing else changes;
//! - the CPU oracle (`gpu_compute::model::step_cpu`) judges every position `Trece` and the tokens are the same as
//!   the glued list with the old LayerNorm (the logits differ only in the last bits: another summation order);
//! - the QMD declares what the program needs (256 threads, 32 bytes shared, one barrier) — the model refuses less;
//! - a planted wrong `layernorm256` (writes nothing) is caught by the oracle.
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
use nvidia_ga104::layernorm256 as ln;

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

fn token(s: &mut TokenSim<'_>, list: TokenList, tok: usize, pos: usize, way: Way) -> Vec<f32> {
    let (m, vas) = (s.m, s.vas.clone());
    let bytes = (4 * m.vocab as u64).next_multiple_of(16);
    let down = kl::copy_launch(vas.logits, s.host.va, bytes).unwrap();
    let host = s.host.clone();
    sys_write(&mut s.r.sys, &host, &vec![0xee; bytes as usize]);
    kd::send_token_list(s, list, &m.weights(), &vas, tok, pos, way, &[down]).unwrap();
    s.ram_logits()
}

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

#[test]
fn t143a_new_layernorm_text_matches_the_oracle() {
    let m = model();
    for mode in [SeqMode::EachLaunch, SeqMode::PerLayer] {
        let way = Way::of(mode);
        let mut old = TokenSim::new(&m, 0);
        let mut new = TokenSim::new(&m, 0);
        diet(&mut old);
        diet(&mut new);
        let a = text(&mut old, TokenList::Fused, way);
        let b = text(&mut new, TokenList::FusedLn, way);
        let w = m.weights();
        let mut c = Caches::new(m.layers.len(), m.shape.e);
        let (mut max_cpu, mut max_old) = (0.0f32, 0.0f32);
        for (pos, ((ta, la), (tb, lb))) in a.iter().zip(&b).enumerate() {
            assert_eq!(ta, tb, "{mode:?}: simbolul dat la pozitia {pos}");
            assert_eq!(
                model::argmax(la),
                model::argmax(lb),
                "{mode:?}: simbolul ales la pozitia {pos}"
            );
            let cpu = model::step_cpu(&w, &mut c, *tb, pos).unwrap();
            let v = km::judge(lb, &cpu);
            assert!(v.passes(), "{mode:?}: {}", v.line(pos));
            if let km::TokenVerdict::Trece { max_abs: d, .. } = v {
                max_cpu = max_cpu.max(d);
            }
            for (x, y) in la.iter().zip(lb) {
                max_old = max_old.max((x - y).abs());
            }
        }
        assert!(
            max_old < 1e-4,
            "{mode:?}: fata de LayerNorm vechi {max_old:e}"
        );
        let n = TokenList::FusedLn.launches_per_token(m.layers.len()) + 1;
        assert_eq!(new.submissions.iter().sum::<usize>(), m.context * n);
        assert!(new.driver_words.iter().all(|x| x.2 == 0));
        eprintln!(
            "{mode:?}: {} pozitii, {n} lansari pe simbol; abaterea maxima fata de procesor {max_cpu:e}, fata de LayerNorm vechi {max_old:e}",
            m.context
        );
    }
}

#[test]
fn t143b_every_layernorm_is_on_the_new_program() {
    let m = model();
    let s = TokenSim::new(&m, 0);
    let w = m.weights();
    let old = TokenList::Fused.launches(&w, &s.vas, 5, 2).unwrap();
    let new = TokenList::FusedLn.launches(&w, &s.vas, 5, 2).unwrap();
    assert_eq!(old.len(), new.len());
    let count = |v: &[kl::Launch], p: Prog| v.iter().filter(|l| l.prog == p).count();
    assert_eq!(count(&new, Prog::LayerNorm256), 2 * m.layers.len() + 1);
    assert_eq!(count(&new, Prog::LayerNorm), 0);
    assert_eq!(count(&old, Prog::LayerNorm), 2 * m.layers.len() + 1);
    for (a, b) in old.iter().zip(&new) {
        // Only the program changes: the same cbuf bytes and grid.
        assert_eq!((a.cbuf, a.grid), (b.cbuf, b.grid));
        if a.prog != Prog::LayerNorm {
            assert_eq!(a.prog, b.prog);
        }
    }
    // Wider than the program takes: the list stays as it was.
    assert_eq!(ln::widen(old.clone(), 1025), old);
    assert_eq!(ln::widen(old.clone(), 0), old);
    // #513 with the copy of the logits: still 88 launches.
    assert_eq!(TokenList::FusedLn.launches_per_token(12) + 1, 88);
    assert_eq!(TokenList::FusedLn.name(), "lipita-ln");
    assert_eq!(TokenList::FusedLn.programs().len(), 9);
    assert!(TokenList::FusedLn.programs().contains(&Prog::LayerNorm256));
    assert!(!TokenList::Fused.programs().contains(&Prog::LayerNorm256));
    // The QMD: 256 threads, 32 bytes shared, one barrier.
    assert_eq!(Prog::LayerNorm256.block(), 256);
    let sh = Prog::LayerNorm256.shared().unwrap();
    assert_eq!((sh.bytes, sh.barriers), (32, 1));
    assert_eq!(Prog::LayerNorm256.registers(), ln::REGISTERS);
    let q = new
        .iter()
        .find(|l| l.prog == Prog::LayerNorm256)
        .unwrap()
        .qmd(0x1_0000, 0x2_0000)
        .unwrap();
    assert_ne!(
        q,
        old.iter()
            .find(|l| l.prog == Prog::LayerNorm)
            .unwrap()
            .qmd(0x1_0000, 0x2_0000)
            .unwrap()
    );
}

#[test]
fn t143c_program_bytes_are_the_nak_dump() {
    let sha = nvidia_ga104::sha256::digest(ln::LAYERNORM256);
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, ln::LAYERNORM256_SHA256);
    assert_eq!(ln::LAYERNORM256.len() % 16, 0);
    let txt = include_str!("../../../tools/nak/out/layernorm256.sm86.txt");
    assert!(
        txt.contains("bar.sync") || txt.contains("bar"),
        "fara bariera"
    );
    assert!(txt.contains("shfl"), "fara shfl");
}

fn ln_writes_nothing(_: &mut gpu_sim::cmdproc::Launch<'_>) -> gpu_core::error::GpuResult<()> {
    Ok(())
}

#[test]
fn t143d_planted_wrong_layernorm_is_caught() {
    let m = model();
    let mut s = TokenSim::new(&m, 0);
    for p in s.models.iter_mut() {
        if p.0 == Prog::LayerNorm256 {
            p.1 = ln_writes_nothing;
        }
    }
    let w = m.weights();
    let mut c = Caches::new(m.layers.len(), m.shape.e);
    let mut caught = false;
    for (pos, tok) in PROMPT.into_iter().enumerate() {
        let got = token(&mut s, TokenList::FusedLn, tok, pos, Way::REPAIR);
        let cpu = model::step_cpu(&w, &mut c, tok, pos).unwrap();
        caught |= !km::judge(&got, &cpu).passes();
    }
    assert!(caught, "programul gresit a trecut de oracol");
}

/// #513's width (768 = 3 values per thread): one layer, a few positions; the oracle passes and the tokens are the
/// old LayerNorm's.
#[test]
fn t143e_width_768() {
    let m = km::test_model(
        Shape {
            e: 768,
            heads: 12,
            eps: 1e-5,
        },
        1,
        256,
        4,
        5,
    )
    .unwrap();
    let way = Way::of(SeqMode::EachLaunch);
    let mut old = TokenSim::new(&m, 0);
    let mut new = TokenSim::new(&m, 0);
    let w = m.weights();
    let mut c = Caches::new(m.layers.len(), m.shape.e);
    for (pos, tok) in [5usize, 200, 17, 99].into_iter().enumerate() {
        let a = token(&mut old, TokenList::Fused, tok, pos, way);
        let b = token(&mut new, TokenList::FusedLn, tok, pos, way);
        assert_eq!(model::argmax(&a), model::argmax(&b), "pozitia {pos}");
        let cpu = model::step_cpu(&w, &mut c, tok, pos).unwrap();
        let v = km::judge(&b, &cpu);
        assert!(v.passes(), "{}", v.line(pos));
    }
}
