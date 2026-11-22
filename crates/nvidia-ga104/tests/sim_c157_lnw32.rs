//! C-157 point 2 (T162), SIMULARE: after the race of the products on the board (the automatic cycle of 22 nov 05:06:
//! 5 kinds passed on quarters, +373 per mille on the token), the list Cyber-GPU picks — `TokenList::FusedLn32`: the
//! new LayerNorm (`layernorm256`, on the board since os-331) and the glued products on `q8_fused32` / the logits on
//! `q8_matvec32`, together. What must hold:
//! - the list is exactly `FusedLn` with the products swapped (= `Fused32` with the LayerNorms swapped): the same 88
//!   launches at #513, the same cbuf bytes and grids, only programs change;
//! - the CPU oracle judges every position `Trece`; the tokens are those of `FusedLn` and of `Fused32`; the logits
//!   differ from each of them only in the last bits (< 1e-4);
//! - the programs to upload are `PROGRAMS_FUSED32` + `LayerNorm256`.
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about the board, and
//! nothing here says which list is faster (every simulated launch costs the same).

mod common;

use common::sys_write;
use common::token_sim::TokenSim;
use gpu_compute::layer::Shape;
use gpu_compute::model::{self, Caches};
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenList, Way};
use nvidia_ga104::kronos_fast::WriteShadow;
use nvidia_ga104::kronos_layer::{self as kl, Prog};
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::q8_fused32 as f32x;

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

/// Largest |a - b| over two texts, and how many logits differ in their bits.
fn apart(a: &[(usize, Vec<f32>)], b: &[(usize, Vec<f32>)]) -> (f32, usize) {
    let (mut max, mut differ) = (0.0f32, 0usize);
    for ((_, la), (_, lb)) in a.iter().zip(b) {
        for (x, y) in la.iter().zip(lb) {
            max = max.max((x - y).abs());
            differ += (x.to_bits() != y.to_bits()) as usize;
        }
    }
    (max, differ)
}

#[test]
fn t162a_ln32_text_matches_the_oracle_and_both_halves() {
    let m = model();
    for mode in [SeqMode::EachLaunch, SeqMode::PerLayer] {
        let way = Way::of(mode);
        let mut runs = Vec::new();
        for list in [TokenList::FusedLn, TokenList::Fused32, TokenList::FusedLn32] {
            let mut s = TokenSim::new(&m, 0);
            diet(&mut s);
            let t = text(&mut s, list, way);
            if list == TokenList::FusedLn32 {
                let n = list.launches_per_token(m.layers.len()) + 1;
                assert_eq!(s.submissions.iter().sum::<usize>(), m.context * n);
                assert!(s.driver_words.iter().all(|x| x.2 == 0));
            }
            runs.push(t);
        }
        let (ln, q, both) = (&runs[0], &runs[1], &runs[2]);
        let w = m.weights();
        let mut c = Caches::new(m.layers.len(), m.shape.e);
        let mut max_cpu = 0.0f32;
        for (pos, (tb, lb)) in both.iter().enumerate() {
            for other in [ln, q] {
                assert_eq!(other[pos].0, *tb, "{mode:?}: simbolul dat la pozitia {pos}");
                assert_eq!(
                    model::argmax(&other[pos].1),
                    model::argmax(lb),
                    "{mode:?}: simbolul ales la pozitia {pos}"
                );
            }
            let cpu = model::step_cpu(&w, &mut c, *tb, pos).unwrap();
            let v = km::judge(lb, &cpu);
            assert!(v.passes(), "{mode:?}: {}", v.line(pos));
            if let km::TokenVerdict::Trece { max_abs: d, .. } = v {
                max_cpu = max_cpu.max(d);
            }
        }
        let (d_ln, n_ln) = apart(ln, both);
        let (d_q, n_q) = apart(q, both);
        assert!(d_ln < 1e-4, "{mode:?}: fata de lipita-ln {d_ln:e}");
        assert!(d_q < 1e-4, "{mode:?}: fata de lipita-32 {d_q:e}");
        // Both swaps must have run: the bits differ from each half alone.
        assert!(
            n_ln > 0,
            "{mode:?}: identic cu lipita-ln: sferturile nu au rulat?"
        );
        assert!(
            n_q > 0,
            "{mode:?}: identic cu lipita-32: LayerNorm nou nu a rulat?"
        );
        eprintln!(
            "{mode:?}: {} pozitii; abaterea maxima fata de procesor {max_cpu:e}, fata de lipita-ln {d_ln:e} ({n_ln} diferite), fata de lipita-32 {d_q:e} ({n_q} diferite)",
            m.context
        );
    }
}

#[test]
fn t162b_list_is_both_swaps_and_nothing_else() {
    let m = model();
    let s = TokenSim::new(&m, 0);
    let w = m.weights();
    for pos in [0usize, 2, 7] {
        let base = TokenList::Fused.launches(&w, &s.vas, 5, pos).unwrap();
        let ln = TokenList::FusedLn.launches(&w, &s.vas, 5, pos).unwrap();
        let q = TokenList::Fused32.launches(&w, &s.vas, 5, pos).unwrap();
        let both = TokenList::FusedLn32.launches(&w, &s.vas, 5, pos).unwrap();
        assert_eq!(both, f32x::narrow(ln.clone()), "= lipita-ln cu sferturile");
        assert_eq!(base.len(), both.len());
        for (i, b) in both.iter().enumerate() {
            assert_eq!(
                (base[i].cbuf, base[i].grid),
                (b.cbuf, b.grid),
                "lansarea {i}"
            );
            let want = if ln[i].prog != base[i].prog {
                ln[i].prog
            } else {
                q[i].prog
            };
            assert_eq!(b.prog, want, "lansarea {i}");
        }
        let count = |p: Prog| both.iter().filter(|l| l.prog == p).count();
        assert_eq!(count(Prog::Q8Fused32), 4 * m.layers.len());
        assert_eq!(count(Prog::Q8Matvec32), 1);
        assert_eq!(
            count(Prog::Q8Fused) + count(Prog::Q8Matvec) + count(Prog::LayerNorm),
            0
        );
        assert_eq!(count(Prog::LayerNorm256), 2 * m.layers.len() + 1);
        for l in &both {
            assert!(TokenList::FusedLn32.programs().contains(&l.prog));
        }
    }
    // #513 with the copy of the logits: still 88 launches.
    assert_eq!(TokenList::FusedLn32.launches_per_token(12) + 1, 88);
    assert_eq!(
        TokenList::FusedLn32.per_layer(),
        TokenList::Fused.per_layer()
    );
    assert_eq!(TokenList::FusedLn32.name(), "lipita-ln-32");
    let p = TokenList::FusedLn32.programs();
    assert_eq!(p, &kl::PROGRAMS_FUSED_LN32);
    assert_eq!(&p[..10], &kl::PROGRAMS_FUSED32);
    assert_eq!(p[10], Prog::LayerNorm256);
    for q in kl::PROGRAMS_FUSED_LN {
        assert!(p.contains(&q));
    }
    // token.comp has no such operation.
    let list = TokenList::FusedLn32.launches(&w, &s.vas, 5, 0).unwrap();
    assert!(nvidia_ga104::kronos_token::work_list(&list).is_err());
}

/// #513's width (768) and one layer: the oracle passes and the tokens are those of `lipita-ln` (the list on the board
/// today, os-336).
#[test]
fn t162c_width_768() {
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
        let a = token(&mut old, TokenList::FusedLn, tok, pos, way);
        let b = token(&mut new, TokenList::FusedLn32, tok, pos, way);
        assert_eq!(model::argmax(&a), model::argmax(&b), "pozitia {pos}");
        let cpu = model::step_cpu(&w, &mut c, tok, pos).unwrap();
        let v = km::judge(&b, &cpu);
        assert!(v.passes(), "{}", v.line(pos));
    }
    let list = TokenList::FusedLn32.launches(&w, &new.vas, 5, 0).unwrap();
    assert_eq!(
        f32x::count(&list),
        4 + 1,
        "toate produsele (si logit-urile) pe sferturi"
    );
}
