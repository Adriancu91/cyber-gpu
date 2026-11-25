//! C-157 (session 239, T248), SIMULARE: `token32.comp` — "one program per token" for today's generation list
//! (`lipita-ln-32` + the argmax on the board + `embed_dev`, `docs/C-157_TOKEN32.md`, predictions written before the
//! code). What must hold:
//! - P1: one token of today's list + its argmax in ONE launch = the 25 separate launches (11 at E 768, 1 layer): the same logits bit for bit
//!   and the same 4 argmax words, at Kronos's width (E 768, 12 heads) too, for G = 1, 7, 48 workgroups;
//! - P2: the whole text (prompt [5, 200, 17] + 12 generated = 14 positions, the token of each generated position read
//!   on the board) in ONE launch = one position per doorbell: the same tokens, the same argmax words at every position,
//!   the same K/V caches bit for bit;
//! - P3: `token.comp` (the old program) still refuses this list; an unknown operation (13) stops the model; the
//!   constant buffer without the workgroup count (0x0c) is refused.
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about the board — not the
//! grid barrier, not that the 32-thread bodies of `layernorm256` / `argmax` give the bits of the 256-thread programs
//! (DEDUS from the source and the listing).

mod common;

use common::sys_read;
use common::token_sim::TokenSim;
use gpu_compute::layer::Shape;
use nvidia_ga104::argmax as am;
use nvidia_ga104::kronos_batch::{self as kb, Input};
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenIo, TokenList, Way};
use nvidia_ga104::kronos_gen::{GenPlan, Generation, Stop, TopOutput};
use nvidia_ga104::kronos_layer::{Launch, Prog};
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::kronos_token as kt;

const PROMPT: [usize; 3] = [5, 200, 17];
const N: usize = 12;

fn small() -> km::OwnedModel {
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

fn wide() -> km::OwnedModel {
    km::test_model(
        Shape {
            e: 768,
            heads: 12,
            eps: 1e-5,
        },
        1,
        512,
        8,
        9,
    )
    .unwrap()
}

/// Today's list of one token (`lipita-ln-32`) + the argmax into the rig's host slot.
fn today(s: &TokenSim<'_>, tok: usize, pos: usize) -> Vec<Launch> {
    let m = s.m;
    let mut l =
        km::token_launches_fused_ln32(&m.shape, m.vocab, m.context, &s.vas, tok, pos).unwrap();
    l.push(am::launch(s.vas.logits, s.argmax_va(), m.vocab).unwrap());
    l
}

fn logits(s: &mut TokenSim<'_>) -> Vec<u32> {
    let (va, n) = (s.vas.logits, s.m.vocab);
    s.read(&[(va, n)]).unwrap().remove(0)
}

/// P1: tokens one after the other, the separate way and the one-launch way on two rigs; logits and argmax words.
fn same_tokens(m: &km::OwnedModel, toks: &[usize], groups: u32) {
    let mut sep = TokenSim::new(m, 0);
    let mut one = TokenSim::new(m, 0);
    for (pos, &tok) in toks.iter().enumerate() {
        let l = today(&sep, tok, pos);
        assert_eq!(l.len(), km::launches_per_token_fused(m.layers.len()) + 1);
        assert!(l.iter().any(|x| x.prog == Prog::LayerNorm256));
        assert!(l.iter().any(|x| x.prog == Prog::Q8Fused32));
        assert!(l.iter().any(|x| x.prog == Prog::Q8Matvec32));
        assert_eq!(l.last().unwrap().prog, Prog::Argmax);
        let vas = sep.vas.clone();
        kd::send_token_list(
            &mut sep,
            TokenList::FusedLn32,
            &m.weights(),
            &vas,
            tok,
            pos,
            Way::of(SeqMode::EachLaunch),
            &l[l.len() - 1..],
        )
        .unwrap();
        let counter = one.submit_token32(&l, groups, false, None).unwrap();
        assert_eq!(counter, (l.len() as u32 - 1) * groups, "contorul barierei");
        assert_eq!(
            logits(&mut sep),
            logits(&mut one),
            "pozitia {pos}: logit-urile difera"
        );
        let (a, b) = (sep.ram_argmax(), one.ram_argmax());
        assert_eq!(a.words, b.words, "pozitia {pos}: cuvintele argmax difera");
        assert!(a.words[0] < m.vocab as u32);
    }
    assert_eq!(one.submissions, vec![1; toks.len()], "o lansare pe simbol");
}

#[test]
fn t248a_one_token_of_todays_list_in_one_launch() {
    let m = small();
    for g in [1, 7, 48] {
        same_tokens(&m, &[5, 200, 17, 3, 99], g);
    }
}

#[test]
fn t248b_the_same_at_kronos_width() {
    let m = wide();
    let n = km::launches_per_token_fused(1) + 1;
    assert_eq!(n, 11);
    same_tokens(&m, &[42, 7, 300], 48);
}

/// One position per doorbell (today's way): the tokens, the argmax words, the caches.
fn caches(s: &mut TokenSim<'_>) -> Vec<u8> {
    let maps: Vec<_> = s.caches.iter().flat_map(|(k, v)| [*k, *v]).collect();
    maps.iter()
        .flat_map(|c| common::get_bytes(s.r.mem.writer_mut(), c.pa, c.size))
        .collect()
}

#[test]
fn t248c_the_whole_text_in_one_launch() {
    let m = small();
    let positions = PROMPT.len() + N - 1;
    let per = kb::launches_per_position(m.layers.len());
    // Today: one position per doorbell.
    let mut sep = TokenSim::new(&m, 0);
    let mut want_seq = PROMPT.to_vec();
    let mut want_words = Vec::new();
    for pos in 0..positions {
        let l = today(&sep, want_seq[pos], pos);
        assert_eq!(l.len(), per);
        let vas = sep.vas.clone();
        kd::send_token_list(
            &mut sep,
            TokenList::FusedLn32,
            &m.weights(),
            &vas,
            want_seq[pos],
            pos,
            Way::of(SeqMode::EachLaunch),
            &l[l.len() - 1..],
        )
        .unwrap();
        let r = sep.ram_argmax();
        want_words.push(r.words);
        if pos + 1 >= want_seq.len() {
            want_seq.push(r.words[0] as usize);
        }
    }
    assert_eq!(want_seq.len(), PROMPT.len() + N);
    // One launch: the whole text, the generated tokens read on the board from the slot of the position before.
    let mut s = TokenSim::with_batch(&m, 0, 0x20000, per * positions).unwrap();
    let rm = s.results.clone().unwrap();
    let mut g = Generation::new(GenPlan::new(&PROMPT, N, m.vocab, m.context).unwrap());
    let (p0, inputs) = g.next_batch(usize::MAX).unwrap();
    assert_eq!((p0, inputs.len()), (0, positions));
    assert_eq!(
        inputs.iter().filter(|i| matches!(i, Input::Board)).count(),
        positions - PROMPT.len()
    );
    let launches = kb::batch_launches(
        kb::TokenList::FusedLn32,
        &m.shape,
        m.vocab,
        m.context,
        &s.vas,
        rm.va,
        p0,
        &inputs,
    )
    .unwrap();
    assert_eq!(launches.len(), per * positions);
    assert_eq!((per, launches.len()), (25, 350));
    assert_eq!(
        launches.iter().filter(|l| l.prog == Prog::EmbedDev).count(),
        N - 1
    );
    let counter = s.submit_token32(&launches, 48, false, None).unwrap();
    assert_eq!(counter, (launches.len() as u32 - 1) * 48);
    assert_eq!(s.submissions, vec![1], "tot textul intr-o singura lansare");
    let buf = sys_read(&s.r.sys, &rm, rm.size);
    let res = kb::read_results(&buf, 0, positions).unwrap();
    let words: Vec<[u32; 4]> = res.iter().map(|r| r.words).collect();
    assert_eq!(
        words, want_words,
        "aceleasi 4 cuvinte argmax la fiecare pozitie"
    );
    let outs: Vec<TopOutput> = res
        .into_iter()
        .map(|result| TopOutput {
            result,
            start_ns: 0,
            end_ns: 0,
            busy_ns: 0,
        })
        .collect();
    g.accept_batch(Ok(outs));
    assert_eq!(g.stop, Some(Stop::Gata));
    assert_eq!(g.seq, want_seq, "aceleasi simboluri");
    assert!(caches(&mut s) == caches(&mut sep), "aceleasi cache-uri K/V");
}

#[test]
fn t248d_refusals() {
    let m = small();
    let mut s = TokenSim::new(&m, 0);
    let l = today(&s, 5, 0);
    // The old program has none of today's programs.
    assert!(kt::work_list(&l).is_err());
    assert!(kt::work_list32(&l).is_ok());
    assert!(kt::work_list32(&[]).is_err());
    let mut odd = l[0].clone();
    odd.cbuf[0x20] = 1;
    assert!(kt::work_list32(&[odd]).is_err());
    assert!(
        kt::work_list32(&[Launch {
            grid: 0,
            ..l[0].clone()
        }])
        .is_err()
    );
    // Every program has its operation number in the new list.
    let w = kt::work_list32(&l).unwrap();
    for (i, x) in l.iter().enumerate() {
        let at = kt::ENTRY * i + kt::ENTRY_OP;
        assert_eq!(
            u32::from_le_bytes(w[at..at + 4].try_into().unwrap()),
            kt::op_number(x.prog)
        );
    }
    // Without the workgroup count at 0x0c: refused.
    assert!(s.submit_token32(&l, 4, true, None).is_err());
    // An unknown operation (13) in entry 3: the model stops.
    let mut s = TokenSim::new(&m, 0);
    assert!(s.submit_token32(&l, 4, false, Some((3, 13))).is_err());
    // A good launch on a fresh rig still passes.
    let mut s = TokenSim::new(&m, 0);
    assert!(s.submit_token32(&l, 4, false, None).is_ok());
}
