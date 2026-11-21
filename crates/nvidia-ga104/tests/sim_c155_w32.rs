//! C-155 point 4 (T146), SIMULARE: the race of the products moved into the token list — `q8_fused32` (`q8_fused`
//! with the quarter-block work split of `q8_matvec32`) in `TokenList::Fused32`. The same text as T142/T143 — prompt,
//! then greedy tokens, each followed by the copy of the logits down into RAM — with the driver's diet on, in
//! `EachLaunch` and `PerLayer`. What must hold:
//! - every glued product of the token (4 per layer) is on `q8_fused32` and the logits on `q8_matvec32`, with the cbuf
//!   bytes unchanged; nothing else changes;
//! - the CPU oracle (`gpu_compute::model::step_cpu`) judges every position `Trece` and the tokens are those of the
//!   glued list (the logits differ only in the last bits: another summation order);
//! - an x that is not 16-aligned is refused by the model (the 128-bit loads) and `narrow` never puts it there;
//! - a planted wrong `q8_fused32` (the summation order of `q8_fused`, or nothing written) is caught.
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about the board, and
//! nothing here says which program is faster (every simulated launch costs the same).

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

#[test]
fn t146a_fused32_text_matches_the_oracle() {
    let m = model();
    for mode in [SeqMode::EachLaunch, SeqMode::PerLayer] {
        let way = Way::of(mode);
        let mut old = TokenSim::new(&m, 0);
        let mut new = TokenSim::new(&m, 0);
        diet(&mut old);
        diet(&mut new);
        let a = text(&mut old, TokenList::Fused, way);
        let b = text(&mut new, TokenList::Fused32, way);
        let w = m.weights();
        let mut c = Caches::new(m.layers.len(), m.shape.e);
        let (mut max_cpu, mut max_old, mut differ) = (0.0f32, 0.0f32, 0usize);
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
                differ += (x.to_bits() != y.to_bits()) as usize;
            }
        }
        assert!(max_old < 1e-4, "{mode:?}: fata de lista lipita {max_old:e}");
        // Another summation order: some logits must differ in the last bits (otherwise the new program did not run).
        assert!(
            differ > 0,
            "{mode:?}: identic cu q8_fused: programul nou nu a rulat?"
        );
        let n = TokenList::Fused32.launches_per_token(m.layers.len()) + 1;
        assert_eq!(new.submissions.iter().sum::<usize>(), m.context * n);
        assert!(new.driver_words.iter().all(|x| x.2 == 0));
        eprintln!(
            "{mode:?}: {} pozitii, {n} lansari pe simbol; abaterea maxima fata de procesor {max_cpu:e}, fata de lista lipita {max_old:e} ({differ} logit-uri diferite in ultimii biti)",
            m.context
        );
    }
}

#[test]
fn t146b_every_glued_product_is_on_the_new_program() {
    let m = model();
    let s = TokenSim::new(&m, 0);
    let w = m.weights();
    let old = TokenList::Fused.launches(&w, &s.vas, 5, 2).unwrap();
    let new = TokenList::Fused32.launches(&w, &s.vas, 5, 2).unwrap();
    assert_eq!(old.len(), new.len());
    let count = |v: &[kl::Launch], p: Prog| v.iter().filter(|l| l.prog == p).count();
    let fused = count(&old, Prog::Q8Fused);
    assert_eq!(fused, 4 * m.layers.len());
    assert_eq!(count(&old, Prog::Q8Matvec), 1, "logit-urile");
    assert_eq!(count(&new, Prog::Q8Fused32), fused);
    assert_eq!(count(&new, Prog::Q8Matvec32), 1);
    assert_eq!(f32x::count(&new), fused + 1);
    assert_eq!(count(&new, Prog::Q8Fused) + count(&new, Prog::Q8Matvec), 0);
    for (a, b) in old.iter().zip(&new) {
        // Only the program changes: the same cbuf bytes and grid.
        assert_eq!((a.cbuf, a.grid), (b.cbuf, b.grid));
        if !matches!(a.prog, Prog::Q8Fused | Prog::Q8Matvec) {
            assert_eq!(a.prog, b.prog);
        }
    }
    // #513 with the copy of the logits: still 88 launches.
    assert_eq!(TokenList::Fused32.launches_per_token(12) + 1, 88);
    assert_eq!(TokenList::Fused32.name(), "lipita-32");
    assert_eq!(TokenList::Fused32.programs(), &kl::PROGRAMS_FUSED32);
    assert!(!TokenList::Fused.programs().contains(&Prog::Q8Fused32));
    // The QMD: 32 threads, 128 bytes shared, no barrier; another program address and register count than q8_fused.
    assert_eq!(Prog::Q8Fused32.block(), 32);
    let sh = Prog::Q8Fused32.shared().unwrap();
    assert_eq!((sh.bytes, sh.barriers), (128, 0));
    assert_eq!(Prog::Q8Fused32.registers(), f32x::REGISTERS);
    // token.comp has no such operation.
    assert!(nvidia_ga104::kronos_token::work_list(&new).is_err());
}

#[test]
fn t146c_program_bytes_are_the_nak_dump() {
    let sha = nvidia_ga104::sha256::digest(f32x::Q8_FUSED32);
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, f32x::Q8_FUSED32_SHA256);
    assert_eq!(f32x::Q8_FUSED32.len() % 16, 0);
    assert_ne!(f32x::Q8_FUSED32, nvidia_ga104::q8_fused::Q8_FUSED);
    let txt = include_str!("../../../tools/nak/out/q8_fused32.sm86.txt");
    // The 128-bit loads of x and no hardware barrier (one warp).
    assert!(txt.contains(".b128"), "fara citiri de 128 de biti");
    assert!(!txt.contains("bar.sync"), "bariera neasteptata");
}

#[test]
fn t146d_unaligned_x_is_refused_and_never_chosen() {
    let m = model();
    let s = TokenSim::new(&m, 0);
    let w = m.weights();
    let mut list = TokenList::Fused.launches(&w, &s.vas, 5, 2).unwrap();
    let at = nvidia_ga104::q8_fused::CBUF_X;
    let i = list.iter().position(|l| l.prog == Prog::Q8Fused).unwrap();
    let x = u64::from_le_bytes(list[i].cbuf[at..at + 8].try_into().unwrap());
    list[i].cbuf[at..at + 8].copy_from_slice(&(x + 4).to_le_bytes());
    let out = f32x::narrow(list);
    assert_eq!(out[i].prog, Prog::Q8Fused, "x nealiniat dat la q8_fused32");
}

#[test]
fn t146d2_model_refuses_unaligned_x() {
    // A hand-written launch with x + 4 on the work channel: the model refuses it (no y written, the token fails).
    let m = model();
    let mut s = TokenSim::new(&m, 0);
    for p in s.models.iter_mut() {
        if p.0 == Prog::Q8Fused32 {
            p.1 = shifted_x;
        }
    }
    let (vas, wts) = (s.vas.clone(), m.weights());
    let bytes = (4 * m.vocab as u64).next_multiple_of(16);
    let down = kl::copy_launch(vas.logits, s.host.va, bytes).unwrap();
    let r = kd::send_token_list(
        &mut s,
        TokenList::Fused32,
        &wts,
        &vas,
        5,
        0,
        Way::REPAIR,
        &[down],
    );
    let refused = r.is_err() || s.ram_logits().iter().any(|v| !v.is_finite());
    assert!(refused, "x nealiniat la 16 a trecut");
}

/// The real model, but with x moved by 4 bytes in the cbuf it reads.
fn shifted_x(l: &mut gpu_sim::cmdproc::Launch<'_>) -> gpu_core::error::GpuResult<()> {
    let x = l.mem.read_u64(l.cbuf0 + 0x30)? + 4;
    l.mem.write_u32(l.cbuf0 + 0x30, x as u32);
    l.mem.write_u32(l.cbuf0 + 0x34, (x >> 32) as u32);
    gpu_sim::cmdproc::q8_fused32_on_cpu(l)
}

/// The program of yesterday under the new name: `q8_fused`'s order. In SIMULARE that is still within the oracle's
/// margin, so the check that catches it is "not bit for bit what the new program gives".
fn old_order(l: &mut gpu_sim::cmdproc::Launch<'_>) -> gpu_core::error::GpuResult<()> {
    gpu_sim::cmdproc::q8_fused_on_cpu(l)
}

fn writes_nothing(_: &mut gpu_sim::cmdproc::Launch<'_>) -> gpu_core::error::GpuResult<()> {
    Ok(())
}

#[test]
fn t146e_planted_wrong_programs_are_caught() {
    let m = model();
    let way = Way::of(SeqMode::EachLaunch);
    let mut good = TokenSim::new(&m, 0);
    let good_text = text(&mut good, TokenList::Fused32, way);
    // (1) the old order under the new name: not the new program's bits.
    let mut s = TokenSim::new(&m, 0);
    for p in s.models.iter_mut() {
        if p.0 == Prog::Q8Fused32 {
            p.1 = old_order;
        }
    }
    let t = text(&mut s, TokenList::Fused32, way);
    let same = good_text.iter().zip(&t).all(|(a, b)| {
        a.1.iter()
            .zip(&b.1)
            .all(|(x, y)| x.to_bits() == y.to_bits())
    });
    assert!(!same, "ordinea veche nu se deosebeste de programul nou");
    // (2) nothing written: the oracle catches it.
    let mut s = TokenSim::new(&m, 0);
    for p in s.models.iter_mut() {
        if p.0 == Prog::Q8Fused32 {
            p.1 = writes_nothing;
        }
    }
    let w = m.weights();
    let mut c = Caches::new(m.layers.len(), m.shape.e);
    let mut caught = false;
    for (pos, tok) in PROMPT.into_iter().enumerate() {
        let got = token(&mut s, TokenList::Fused32, tok, pos, Way::REPAIR);
        let cpu = model::step_cpu(&w, &mut c, tok, pos).unwrap();
        caught |= !km::judge(&got, &cpu).passes();
    }
    assert!(caught, "programul gresit a trecut de oracol");
}

/// #513's width (768: 96 quarters per row, exactly 3 per thread; proj: 3072 inputs, 12 per thread): one layer, a
/// few positions; the oracle passes and the tokens are those of the glued list.
#[test]
fn t146f_width_768() {
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
        let b = token(&mut new, TokenList::Fused32, tok, pos, way);
        assert_eq!(model::argmax(&a), model::argmax(&b), "pozitia {pos}");
        let cpu = model::step_cpu(&w, &mut c, tok, pos).unwrap();
        let v = km::judge(&b, &cpu);
        assert!(v.passes(), "{}", v.line(pos));
    }
    let list = TokenList::Fused32.launches(&w, &new.vas, 5, 0).unwrap();
    assert_eq!(
        f32x::count(&list),
        4 + 1,
        "toate produsele (si logit-urile) pe sferturi"
    );
}
