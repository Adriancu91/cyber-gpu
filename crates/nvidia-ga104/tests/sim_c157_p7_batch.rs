//! C-157 point 7 (T180), SIMULARE: the chosen token stays on the board (`embed_dev.comp`, `nvidia_ga104::kronos_batch`):
//! several positions on ONE doorbell, the token of each generated position read on the board from the argmax slot of
//! the position before. What must hold (`docs/C-157_P7_SIMBOL_PE_PLACA.md`, predictions written before the code):
//! - the whole text (prompt [5, 200, 17] + 12 tokens = 14 positions) in one submission gives the same tokens, the same
//!   4 argmax words at every position and the same K/V caches bit for bit as one position per doorbell (today's way);
//!   the same with batches of 4; with the stale-read model of the board on (invalidation before every launch);
//! - sent with one cache invalidation per submission (`Once`) under the stale-read model it makes other tokens without
//!   an error: the batch must go with `EachLaunch`;
//! - a planted `embed_dev` reading the wrong slot is caught; an argmax index outside the vocabulary makes `embed_dev`
//!   write NaN (nothing read from the table) and the generation stops there with an error; "no token" stops it with
//!   `LogitiNefiniti` — never a made-up token.
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about the board.

mod common;

use std::sync::atomic::{AtomicU64, Ordering};

use common::token_sim::TokenSim;
use common::{get_bytes, sys_read};
use gpu_compute::layer::Shape;
use gpu_compute::model::ARGMAX_NONE;
use gpu_core::error::GpuResult;
use nvidia_ga104::argmax::{self as am, ArgmaxResult};
use nvidia_ga104::kronos_batch::{self as kb, Input};
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenIo, TokenList, Way};
use nvidia_ga104::kronos_gen::{GenPlan, Generation, Stop, TopOutput, TopRunner};
use nvidia_ga104::kronos_layer::Prog;
use nvidia_ga104::kronos_model as km;

const PROMPT: [usize; 3] = [5, 200, 17];
const N: usize = 12;

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

/// Today's way: one position per doorbell, `lipita-ln-32` + the argmax, the 16 bytes read after each.
struct OneByOne<'s, 'a> {
    s: &'s mut TokenSim<'a>,
    words: Vec<[u32; 4]>,
}

impl TopRunner for OneByOne<'_, '_> {
    fn run_top(&mut self, tok: usize, pos: usize) -> GpuResult<TopOutput> {
        let (m, vas) = (self.s.m, self.s.vas.clone());
        let extra = [am::launch(vas.logits, self.s.argmax_va(), m.vocab)?];
        kd::send_token_list(
            self.s,
            TokenList::FusedLn32,
            &m.weights(),
            &vas,
            tok,
            pos,
            Way::of(SeqMode::EachLaunch),
            &extra,
        )?;
        let r = self.s.ram_argmax();
        self.words.push(r.words);
        let (start_ns, end_ns, busy_ns) = *self.s.times.last().unwrap();
        Ok(TopOutput {
            result: r,
            start_ns,
            end_ns,
            busy_ns,
        })
    }
}

/// What a run leaves: the generation, the argmax words by position, the caches' bytes, the doorbells.
struct Run {
    g: Generation,
    words: Vec<[u32; 4]>,
    caches: Vec<u8>,
    doorbells: usize,
    embed_dev: usize,
}

fn caches(s: &mut TokenSim<'_>) -> Vec<u8> {
    let maps: Vec<_> = s.caches.iter().flat_map(|(k, v)| [*k, *v]).collect();
    maps.iter()
        .flat_map(|c| get_bytes(s.r.mem.writer_mut(), c.pa, c.size))
        .collect()
}

fn one_by_one(m: &km::OwnedModel) -> Run {
    let mut s = TokenSim::new(m, 0);
    let plan = GenPlan::new(&PROMPT, N, m.vocab, m.context).unwrap();
    let mut g = Generation::new(plan);
    let mut r = OneByOne {
        s: &mut s,
        words: Vec::new(),
    };
    let stop = g.run_top(&mut r).clone();
    assert_eq!(stop, Stop::Gata);
    let words = r.words;
    let doorbells = s.submissions.len();
    Run {
        g,
        words,
        caches: caches(&mut s),
        doorbells,
        embed_dev: 0,
    }
}

/// The batch way: `next_batch(max)` → `batch_launches` → ONE submission → the 16-byte slots read → `accept_batch`.
fn batched(s: &mut TokenSim<'_>, max: usize) -> Run {
    batched_in(s, max, SeqMode::EachLaunch)
}

fn batched_in(s: &mut TokenSim<'_>, max: usize, mode: SeqMode) -> Run {
    let m = s.m;
    let plan = GenPlan::new(&PROMPT, N, m.vocab, m.context).unwrap();
    let mut g = Generation::new(plan);
    let rm = s.results.clone().unwrap();
    let per = kb::launches_per_position(m.layers.len());
    let mut embed_dev = 0;
    while let Some((p0, inputs)) = g.next_batch(max) {
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
        assert_eq!(launches.len(), per * inputs.len());
        embed_dev += launches.iter().filter(|l| l.prog == Prog::EmbedDev).count();
        let outs = s.submit(&launches, 0, mode).and_then(|()| {
            let buf = sys_read(&s.r.sys, &rm, rm.size);
            let res = kb::read_results(&buf, p0, inputs.len())?;
            let st = s.last_stamps.clone();
            let d = s.durations.last().unwrap().clone();
            Ok(res
                .into_iter()
                .enumerate()
                .map(|(j, result)| TopOutput {
                    result,
                    start_ns: st[j * per],
                    end_ns: st[(j + 1) * per],
                    busy_ns: d[j * per..(j + 1) * per].iter().sum(),
                })
                .collect::<Vec<_>>())
        });
        g.accept_batch(outs);
    }
    let buf = sys_read(&s.r.sys, &rm, rm.size);
    let positions = g.plan.positions();
    let words = kb::read_results(&buf, 0, positions)
        .unwrap()
        .iter()
        .map(|r| r.words)
        .collect();
    Run {
        words,
        caches: caches(s),
        doorbells: s.submissions.len(),
        embed_dev,
        g,
    }
}

fn slots_for(m: &km::OwnedModel, max: usize) -> usize {
    kb::launches_per_position(m.layers.len()) * max
}

#[test]
fn t180a_whole_text_on_one_doorbell_same_tokens_words_caches() {
    let m = model();
    let want = one_by_one(&m);
    let positions = PROMPT.len() + N - 1;
    assert_eq!(want.doorbells, positions);
    assert_eq!(want.g.generated().len(), N);
    let mut s = TokenSim::with_batch(&m, 0, 0x20000, slots_for(&m, positions)).unwrap();
    let got = batched(&mut s, usize::MAX);
    assert_eq!(got.g.stop, Some(Stop::Gata));
    assert_eq!(got.g.seq, want.g.seq, "aceleasi simboluri");
    assert_eq!(
        got.words, want.words,
        "aceleasi 4 cuvinte argmax la fiecare pozitie"
    );
    assert!(
        got.caches == want.caches,
        "aceleasi cache-uri K/V bit cu bit"
    );
    assert_eq!(got.doorbells, 1, "o singura sonerie");
    // Every position after the prompt took its token from the board.
    assert_eq!(got.embed_dev, positions - PROMPT.len());
}

#[test]
fn t180b_batches_of_four_and_the_stale_read_model() {
    let m = model();
    let want = one_by_one(&m);
    let positions = PROMPT.len() + N - 1;
    for (max, stale) in [(4usize, 0u64), (4, 128), (usize::MAX, 128), (1, 0)] {
        let slots = slots_for(&m, max.min(positions));
        let mut s = TokenSim::with_batch(&m, stale, 0x20000, slots).unwrap();
        let got = batched(&mut s, max);
        assert_eq!(got.g.seq, want.g.seq, "max {max}, stale {stale}");
        assert_eq!(got.words, want.words, "max {max}, stale {stale}");
        assert!(got.caches == want.caches, "max {max}, stale {stale}");
        assert_eq!(
            got.doorbells,
            positions.div_ceil(max.min(positions)),
            "max {max}"
        );
    }
}

#[test]
fn t180c_next_batch_inputs() {
    let m = model();
    let plan = GenPlan::new(&PROMPT, N, m.vocab, m.context).unwrap();
    let g = Generation::new(plan.clone());
    let (p0, inp) = g.next_batch(5).unwrap();
    assert_eq!(p0, 0);
    assert_eq!(
        inp,
        vec![
            Input::Cpu(5),
            Input::Cpu(200),
            Input::Cpu(17),
            Input::Board,
            Input::Board
        ]
    );
    assert!(g.next_batch(0).is_none());
    assert_eq!(g.next_batch(usize::MAX).unwrap().1.len(), plan.positions());
    // A Board token at position 0 is refused; so is a batch past the context.
    let s = TokenSim::with_batch(&m, 0, 0x20000, 64).unwrap();
    let rm = s.results.clone().unwrap();
    let list = |p0: usize, inp: &[Input]| {
        kb::batch_launches(
            kb::TokenList::FusedLn,
            &m.shape,
            m.vocab,
            m.context,
            &s.vas,
            rm.va,
            p0,
            inp,
        )
    };
    assert!(list(0, &[Input::Board]).is_err());
    assert!(list(15, &[Input::Cpu(1), Input::Board]).is_err());
    assert!(list(0, &[]).is_err());
    // The list of a Cpu position = today's list + its argmax; a Board position differs only in launch 0.
    let a = list(4, &[Input::Cpu(7)]).unwrap();
    let today = km::token_launches_fused_ln(&m.shape, m.vocab, m.context, &s.vas, 7, 4).unwrap();
    assert_eq!(a.len(), today.len() + 1);
    assert!(
        a[..today.len()]
            .iter()
            .zip(&today)
            .all(|(x, y)| x.prog == y.prog && x.cbuf == y.cbuf && x.grid == y.grid)
    );
    assert_eq!(a.last().unwrap().prog, Prog::Argmax);
    let b = list(4, &[Input::Board]).unwrap();
    assert_eq!(b[0].prog, Prog::EmbedDev);
    assert!(
        b[1..]
            .iter()
            .zip(&a[1..])
            .all(|(x, y)| x.prog == y.prog && x.cbuf == y.cbuf)
    );
    assert_eq!(
        u64::from_le_bytes(
            b[0].cbuf[kb::CBUF_RESULT..kb::CBUF_RESULT + 8]
                .try_into()
                .unwrap()
        ),
        rm.va + 16 * 3
    );
}

// ------------------------------------------------------------------------------------------------ planted errors

fn embed_dev_wrong_slot(l: &mut gpu_sim::cmdproc::Launch<'_>) -> GpuResult<()> {
    // Reads the slot of its own position (not yet written: the 0xEE fill) instead of the one before.
    let r = l.mem.read_u64(l.cbuf0 + 0x30)? + 16;
    l.mem.write_u32(l.cbuf0 + 0x30, r as u32);
    l.mem.write_u32(l.cbuf0 + 0x34, (r >> 32) as u32);
    gpu_sim::cmdproc::embed_dev_on_cpu(l)
}

/// The slot address the planted argmax spoils, and what it writes there.
static TARGET: AtomicU64 = AtomicU64::new(0);
static MODE: AtomicU64 = AtomicU64::new(0);

fn argmax_spoiled(l: &mut gpu_sim::cmdproc::Launch<'_>) -> GpuResult<()> {
    gpu_sim::cmdproc::argmax_on_cpu(l)?;
    let res = l.mem.read_u64(l.cbuf0 + 0x30)?;
    if res == TARGET.load(Ordering::SeqCst) {
        match MODE.load(Ordering::SeqCst) {
            // An index past the vocabulary (256), no non-finite count: not a result of this launch.
            0 => l.mem.write_u32(res, 256 + 3),
            // "No token": the index ARGMAX_NONE with one non-finite logit.
            _ => {
                l.mem.write_u32(res, ARGMAX_NONE);
                l.mem.write_u32(res + 8, 1);
            }
        }
    }
    Ok(())
}

fn plant(s: &mut TokenSim<'_>, p: Prog, f: gpu_sim::cmdproc::ProgramFn) {
    for x in s.models.iter_mut() {
        if x.0 == p {
            x.1 = f;
        }
    }
}

#[test]
fn t180d_planted_wrong_slot_is_caught() {
    let m = model();
    let want = one_by_one(&m);
    let mut s = TokenSim::with_batch(&m, 0, 0x20000, slots_for(&m, 14)).unwrap();
    plant(&mut s, Prog::EmbedDev, embed_dev_wrong_slot);
    let got = batched(&mut s, usize::MAX);
    assert_ne!(got.g.seq, want.g.seq);
    // The fill 0xEEEEEEEE is not a token: embed_dev wrote NaN, the first generated position's argmax says no token.
    assert_eq!(got.g.stop, Some(Stop::LogitiNefiniti { pos: PROMPT.len() }));
    assert_eq!(got.g.seq, want.g.seq[..PROMPT.len() + 1].to_vec());
}

#[test]
fn t180e_bad_index_poisons_and_stops_no_guessed_token() {
    let m = model();
    let want = one_by_one(&m);
    for (mode, at) in [(0u64, 6usize), (1, 6), (0, PROMPT.len() - 1)] {
        let mut s = TokenSim::with_batch(&m, 0, 0x20000, slots_for(&m, 14)).unwrap();
        let rm = s.results.clone().unwrap();
        TARGET.store(rm.va + 16 * at as u64, Ordering::SeqCst);
        MODE.store(mode, Ordering::SeqCst);
        plant(&mut s, Prog::Argmax, argmax_spoiled);
        let got = batched(&mut s, usize::MAX);
        TARGET.store(0, Ordering::SeqCst);
        match mode {
            0 => assert!(
                matches!(got.g.stop, Some(Stop::Eroare { pos, .. }) if pos == at),
                "{:?}",
                got.g.stop
            ),
            _ => assert_eq!(got.g.stop, Some(Stop::LogitiNefiniti { pos: at })),
        }
        // Everything up to the spoiled slot is today's; nothing after it was kept.
        assert_eq!(
            got.g.seq,
            want.g.seq[..at + 1].to_vec(),
            "mode {mode}, at {at}"
        );
        // The board ran on: every later position saw NaN (embed_dev read nothing from the table) → "no token",
        // all logits not finite.
        for p in at + 1..PROMPT.len() + N - 1 {
            let r = ArgmaxResult::from_words(got.words[p]);
            assert_eq!(r.words[0], ARGMAX_NONE, "pozitia {p}");
            assert_eq!(r.nonfinite(), m.vocab as u32, "pozitia {p}");
            assert_eq!(r.token(m.vocab).unwrap(), None);
        }
    }
}

#[test]
fn t180f_one_invalidation_per_submission_is_not_enough() {
    // With the board's stale-read model on, the batch sent with ONE cache invalidation at its start (`Once`) reads old
    // bytes (the argmax slots, the caches) and makes OTHER tokens without any error — the reason the batch must go
    // with an invalidation before every launch (`EachLaunch`, the way that passed I14), as t180b shows.
    let m = model();
    let want = one_by_one(&m);
    let mut s = TokenSim::with_batch(&m, 128, 0x20000, slots_for(&m, 14)).unwrap();
    let got = batched_in(&mut s, usize::MAX, SeqMode::Once);
    assert!(s.stale_bytes() > 0);
    assert_ne!(got.g.seq, want.g.seq);
    // Without the stale model the same batch is right: the difference is the stale reads alone.
    let mut s = TokenSim::with_batch(&m, 0, 0x20000, slots_for(&m, 14)).unwrap();
    assert_eq!(
        batched_in(&mut s, usize::MAX, SeqMode::Once).g.seq,
        want.g.seq
    );
    let mut s = TokenSim::with_batch(&m, 128, 0x20000, slots_for(&m, 14)).unwrap();
    let _ = batched(&mut s, usize::MAX);
    assert_eq!(s.stale_bytes(), 0);
}
