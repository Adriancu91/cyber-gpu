//! C-157 point 7, second part (T194), SIMULARE: the next batch written while the board works on the one before —
//! two zones of launch slots + records, at most two batches at the board (`kronos_batch::run_pipelined`,
//! `TwoZones`, `Generation::batch_ahead`). What must hold (`docs/C-157_P7_LOT_URMATOR.md`, predictions written
//! before the code):
//! - prompt [5, 200, 17] + 12 tokens, batches of 1 / 4 / 5, with and without the stale-read model: the same tokens,
//!   the same 4 argmax words at every position and the same K/V caches bit for bit as one position per doorbell;
//!   ⌈14 / batch⌉ doorbells; every batch but the first written while the one before was at the board;
//! - an EOS in the middle of a batch: the same sequence as one position per doorbell, the batch after it sent,
//!   waited for and dropped, no error;
//! - planted: both batches at the board in ONE zone → caught; a bad argmax index at position 6 → the same stop as
//!   T180e, the batch after it waited for.
//!
//! The board model runs the batches sent only when the driver waits (the worst case for a zone written too early).
//! The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about the board.

mod common;

use std::sync::atomic::{AtomicU64, Ordering};

use common::token_sim::TokenSim;
use common::{get_bytes, gpu_runs, put_bytes, span, sys_read};
use gpu_compute::layer::Shape;
use gpu_core::error::{GpuError, GpuResult};
use gpu_sim::cmdproc::{Needs, ProgramFn};
use nvidia_ga104::argmax as am;
use nvidia_ga104::gsp::golden::Mapping;
use nvidia_ga104::kronos_batch::{self as kb, BatchIo, Input, TwoZones};
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenList, Way};
use nvidia_ga104::kronos_gen::{GenPlan, Generation, Stop, TopOutput, TopRunner};
use nvidia_ga104::kronos_layer::Prog;
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::transfer;

const PROMPT: [usize; 3] = [5, 200, 17];
const N: usize = 12;
const POSITIONS: usize = PROMPT.len() + N - 1;

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

/// Today's way (as T180): one position per doorbell, the 16 bytes read after each.
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

fn caches(s: &mut TokenSim<'_>) -> Vec<u8> {
    let maps: Vec<_> = s.caches.iter().flat_map(|(k, v)| [*k, *v]).collect();
    maps.iter()
        .flat_map(|c| get_bytes(s.r.mem.writer_mut(), c.pa, c.size))
        .collect()
}

struct Want {
    g: Generation,
    words: Vec<[u32; 4]>,
    caches: Vec<u8>,
}

fn one_by_one(m: &km::OwnedModel, eos: Option<usize>) -> Want {
    let mut s = TokenSim::new(m, 0);
    let mut plan = GenPlan::new(&PROMPT, N, m.vocab, m.context).unwrap();
    plan.eos = eos;
    let mut g = Generation::new(plan);
    let mut r = OneByOne {
        s: &mut s,
        words: Vec::new(),
    };
    g.run_top(&mut r);
    let words = r.words;
    Want {
        g,
        words,
        caches: caches(&mut s),
    }
}

/// One batch sent and not yet read back.
#[derive(Clone, Copy)]
struct Pending {
    first: u32,
    launches: usize,
    fence: u64,
    done: u64,
    executed: bool,
}

/// [`BatchIo`] on the simulated channel: `send` writes the batch's cbufs + QMDs into the zone's launch slots, its
/// records into the zone's records, the pushbuffer, and rings — the board model does NOT run; `wait` runs the board
/// over everything sent (if the zone's batch has not run yet), reads the zone's records and the argmax slots, retires.
struct PipeSim<'s, 'a> {
    s: &'s mut TokenSim<'a>,
    per_zone: usize,
    mode: SeqMode,
    payload: u32,
    pending: [Option<Pending>; 2],
    /// Planted: every batch into zone 0.
    one_zone: bool,
    /// Sends made while another batch was sent and not yet run by the board model.
    sent_while_unrun: usize,
    board_runs: usize,
    embed_dev: usize,
}

impl<'s, 'a> PipeSim<'s, 'a> {
    fn new(s: &'s mut TokenSim<'a>, per_zone: usize, mode: SeqMode) -> Self {
        assert!(512 * 2 * per_zone as u64 <= s.launches.size);
        assert!(16 * 2 * (per_zone as u64 + 1) <= s.records.size);
        PipeSim {
            s,
            per_zone,
            mode,
            payload: 0x10_0000,
            pending: [None, None],
            one_zone: false,
            sent_while_unrun: 0,
            board_runs: 0,
            embed_dev: 0,
        }
    }

    fn zone(&self, z: usize) -> usize {
        if self.one_zone { 0 } else { z }
    }

    fn records_of(&self, z: usize) -> (u64, u64) {
        let off = 16 * (self.per_zone as u64 + 1) * z as u64;
        (self.s.records.va + off, self.s.records.pa + off)
    }

    fn board_runs_everything_sent(&mut self) -> GpuResult<()> {
        let s = &mut *self.s;
        let programs: Vec<(&[u8], ProgramFn, Needs)> = s
            .models
            .iter()
            .map(|&(p, f, needs)| (p.code(), f, needs))
            .collect();
        let mut maps: Vec<Mapping> = vec![s.blob, s.launches, s.records];
        maps.extend(s.caches.iter().flat_map(|(k, c)| [*k, *c]));
        maps.extend(s.progs.iter().map(|(_, m)| *m));
        let rm = s.results.clone().unwrap();
        let mut loads: Vec<(u64, u64)> = maps.iter().map(span).collect();
        loads.push((s.host.va, s.host.size));
        loads.push((rm.va, rm.size));
        let mut stores = vec![
            span(&s.blob),
            span(&s.records),
            (s.host.va, s.host.size),
            (rm.va, rm.size),
        ];
        stores.extend(s.caches.iter().flat_map(|(k, c)| [span(k), span(c)]));
        gpu_runs(&mut s.r, &programs, &loads, &stores)?;
        for p in self.pending.iter_mut().flatten() {
            p.executed = true;
        }
        self.board_runs += 1;
        Ok(())
    }
}

impl BatchIo for PipeSim<'_, '_> {
    fn send(&mut self, zone: usize, first_pos: usize, inputs: &[Input]) -> GpuResult<()> {
        let z = self.zone(zone);
        let m = self.s.m;
        let rm = self.s.results.clone().unwrap();
        let launches = kb::batch_launches(
            kb::TokenList::FusedLn32,
            &m.shape,
            m.vocab,
            m.context,
            &self.s.vas,
            rm.va,
            first_pos,
            inputs,
        )?;
        let n = launches.len();
        if n > self.per_zone {
            return Err(GpuError::InvalidArgument("SIM: lotul nu incape in zona"));
        }
        self.embed_dev += launches.iter().filter(|l| l.prog == Prog::EmbedDev).count();
        let va = |p: Prog| self.s.progs.iter().find(|(q, _)| *q == p).unwrap().1.va;
        let mut qmds = Vec::with_capacity(n);
        for (i, l) in launches.iter().enumerate() {
            let off = 512 * (z * self.per_zone + i) as u64;
            let (cb_va, cb_pa) = (self.s.launches.va + off, self.s.launches.pa + off);
            let qmd = l.qmd(va(l.prog), cb_va)?;
            let w = self.s.r.mem.writer_mut();
            put_bytes(w, cb_pa, &l.cbuf);
            put_bytes(w, cb_pa + 256, &qmd);
            qmds.push(cb_va + 256);
        }
        let (rec_va, rec_pa) = self.records_of(z);
        put_bytes(self.s.r.mem.writer_mut(), rec_pa, &vec![0; 16 * (n + 1)]);
        let first = self.payload;
        self.payload += 0x100;
        let mut words = vec![0u32; kd::words(self.mode, n)];
        let mut pb = PushBuf::new(&mut words);
        kd::encode(self.mode, &mut pb, &qmds, rec_va, first)?;
        let fence = rec_va + 16 * n as u64;
        let done = (first + n as u32) as u64;
        self.s
            .r
            .wq
            .submit_fenced(self.s.r.mem.writer_mut(), &words, fence, done)?;
        if self.pending.iter().flatten().any(|p| !p.executed) {
            self.sent_while_unrun += 1;
        }
        self.pending[z] = Some(Pending {
            first,
            launches: n,
            fence,
            done,
            executed: false,
        });
        self.s.submissions.push(n);
        Ok(())
    }

    fn wait(&mut self, zone: usize, first_pos: usize, n: usize) -> GpuResult<Vec<TopOutput>> {
        let z = self.zone(zone);
        let p =
            self.pending[z].ok_or(GpuError::InvalidArgument("SIM: niciun lot trimis in zona"))?;
        if !p.executed {
            self.board_runs_everything_sent()?;
        }
        let (_, rec_pa) = self.records_of(z);
        let rec = get_bytes(
            self.s.r.mem.writer_mut(),
            rec_pa,
            16 * (p.launches as u64 + 1),
        );
        let each = transfer::analyze_sequence(&rec, p.launches, p.first)?;
        let ts = |k: usize| u64::from_le_bytes(rec[16 * k + 8..16 * k + 16].try_into().unwrap());
        let rm = self.s.results.clone().unwrap();
        let buf = sys_read(&self.s.r.sys, &rm, rm.size);
        let res = kb::read_results(&buf, first_pos, n)?;
        self.s.r.wq.retire(p.fence, p.done)?;
        self.pending[z] = None;
        let per = p.launches / n;
        Ok(res
            .into_iter()
            .enumerate()
            .map(|(j, result)| TopOutput {
                result,
                start_ns: ts(j * per),
                end_ns: ts((j + 1) * per),
                busy_ns: each[j * per..(j + 1) * per].iter().sum(),
            })
            .collect())
    }
}

struct Got {
    g: Generation,
    run: kb::PipelineRun,
    words: Vec<[u32; 4]>,
    caches: Vec<u8>,
    sent_while_unrun: usize,
    board_runs: usize,
    embed_dev: usize,
}

fn rig(m: &km::OwnedModel, stale: u64, max: usize) -> (TokenSim<'_>, usize) {
    let per_zone = kb::launches_per_position(m.layers.len()) * max.min(POSITIONS);
    (
        TokenSim::with_batch(m, stale, 0x40000, 2 * per_zone + 1).unwrap(),
        per_zone,
    )
}

fn pipelined(
    s: &mut TokenSim<'_>,
    per_zone: usize,
    max: usize,
    eos: Option<usize>,
    one_zone: bool,
) -> Got {
    let m = s.m;
    let mut plan = GenPlan::new(&PROMPT, N, m.vocab, m.context).unwrap();
    plan.eos = eos;
    let mut g = Generation::new(plan);
    let per = kb::launches_per_position(m.layers.len());
    let mut zones = TwoZones::new(per_zone).unwrap();
    let mut io = PipeSim::new(s, per_zone, SeqMode::EachLaunch);
    io.one_zone = one_zone;
    let run = kb::run_pipelined(&mut g, &mut io, &mut zones, per, max);
    let (sent_while_unrun, board_runs, embed_dev) =
        (io.sent_while_unrun, io.board_runs, io.embed_dev);
    if !one_zone {
        assert_eq!(zones.in_flight(), 0, "nicio zona ramasa ocupata");
    }
    let rm = s.results.clone().unwrap();
    let buf = sys_read(&s.r.sys, &rm, rm.size);
    let words = kb::read_results(&buf, 0, POSITIONS)
        .unwrap()
        .iter()
        .map(|r| r.words)
        .collect();
    Got {
        g,
        run,
        words,
        caches: caches(s),
        sent_while_unrun,
        board_runs,
        embed_dev,
    }
}

#[test]
fn t194a_two_zones_same_tokens_words_caches() {
    let m = model();
    let want = one_by_one(&m, None);
    assert_eq!(want.g.stop, Some(Stop::Gata));
    for (max, stale) in [(4usize, 0u64), (4, 128), (5, 0), (1, 0), (1, 128)] {
        let (mut s, per_zone) = rig(&m, stale, max);
        let got = pipelined(&mut s, per_zone, max, None, false);
        let batches = POSITIONS.div_ceil(max);
        assert_eq!(got.run.stop, Stop::Gata, "max {max}, stale {stale}");
        assert_eq!(got.g.seq, want.g.seq, "max {max}, stale {stale}");
        assert_eq!(got.words, want.words, "max {max}, stale {stale}");
        assert!(got.caches == want.caches, "max {max}, stale {stale}");
        assert_eq!(s.stale_bytes(), 0, "max {max}, stale {stale}");
        // ⌈14 / batch⌉ doorbells; every batch but the first written while the one before was at the board.
        assert_eq!(got.run.sent, batches);
        assert_eq!(s.submissions.len(), batches);
        assert_eq!(got.run.overlapped, batches - 1);
        // The board model runs everything sent when the driver waits, so in it only every second send finds the batch
        // before still not run at all (in the driver's view all `batches - 1` were written while one was at the board).
        assert_eq!(got.sent_while_unrun, batches / 2, "max {max}");
        assert_eq!(got.run.max_in_flight, 2.min(batches));
        assert_eq!(got.run.drained_batches, 0);
        assert!(got.run.drain_error.is_none());
        // The board model ran once per pair of batches (the worst case: nothing ran before the driver waited).
        assert!(
            got.board_runs <= batches.div_ceil(2) + 1,
            "{}",
            got.board_runs
        );
        assert_eq!(got.embed_dev, POSITIONS - PROMPT.len());
    }
}

#[test]
fn t194b_batch_ahead_is_next_batch_without_the_results() {
    let m = model();
    let plan = GenPlan::new(&PROMPT, N, m.vocab, m.context).unwrap();
    let g = Generation::new(plan.clone());
    assert_eq!(g.batch_ahead(0, 4), g.next_batch(4));
    assert_eq!(
        g.batch_ahead(2, 3).unwrap(),
        (2, vec![Input::Cpu(17), Input::Board, Input::Board])
    );
    assert_eq!(g.batch_ahead(12, 4).unwrap().1.len(), POSITIONS - 12);
    assert!(g.batch_ahead(POSITIONS, 4).is_none());
    assert!(g.batch_ahead(0, 0).is_none());
    assert!(g.batch_ahead(1, usize::MAX).is_some());
    let mut stopped = Generation::new(plan);
    stopped.stop = Some(Stop::Gata);
    assert!(stopped.batch_ahead(0, 4).is_none());
}

#[test]
fn t194c_eos_mid_batch_drains_the_batch_after() {
    let m = model();
    let free = one_by_one(&m, None);
    // The small test model generates [17, 17, 75, 75, ...]: EOS = 75, first made at position 4 — with batches of 3
    // (0..3, 3..6, 6..9, ...) that is the middle of the second batch, and the third one is already at the board.
    const B: usize = 3;
    let made = free.g.generated();
    let i = made.iter().position(|&t| !PROMPT.contains(&t)).unwrap();
    let eos = made[i];
    let pos = PROMPT.len() - 1 + i;
    assert!(
        !pos.is_multiple_of(B) && pos / B + 1 < POSITIONS.div_ceil(B),
        "EOS {eos} la pozitia {pos}"
    );
    let want = one_by_one(&m, Some(eos));
    assert_eq!(want.g.stop, Some(Stop::Sfarsit));
    assert_eq!(want.g.generated().len(), i + 1);
    eprintln!("EOS {eos} la pozitia {pos}, generate {made:?}");
    let (mut s, per_zone) = rig(&m, 0, B);
    let got = pipelined(&mut s, per_zone, B, Some(eos), false);
    assert_eq!(got.g.seq, want.g.seq);
    assert_eq!(got.g.stop, want.g.stop);
    assert_eq!(got.run.stop, want.g.stop.clone().unwrap());
    // The batch after the one with the EOS was already at the board: waited for, dropped, no error.
    assert_eq!(got.run.drained_batches, 1);
    assert_eq!(got.run.drained_positions, B);
    assert!(got.run.drain_error.is_none());
    assert_eq!(got.run.sent, pos / B + 2);
    assert_eq!(got.run.sent, s.submissions.len());
}

#[test]
fn t194d_planted_one_zone_for_two_batches_is_caught() {
    let m = model();
    let want = one_by_one(&m, None);
    let (mut s, per_zone) = rig(&m, 0, 4);
    let got = pipelined(&mut s, per_zone, 4, None, true);
    // Batch 0 ran the launches of batch 1 (its slots rewritten before the board read them): its own argmax slots
    // were never written → not a token → error stop; the tokens and caches are not today's.
    assert_ne!(got.g.seq, want.g.seq);
    assert!(
        matches!(got.g.stop, Some(Stop::Eroare { pos: 0, .. })),
        "{:?}",
        got.g.stop
    );
    assert!(got.caches != want.caches);
    // The second wait found nothing in its zone (batch 0's wait retired both): reported, not hidden.
    assert!(got.run.drain_error.is_some());
}

// ------------------------------------------------------------------------------------------------ planted argmax

static TARGET: AtomicU64 = AtomicU64::new(0);

fn argmax_spoiled(l: &mut gpu_sim::cmdproc::Launch<'_>) -> GpuResult<()> {
    gpu_sim::cmdproc::argmax_on_cpu(l)?;
    let res = l.mem.read_u64(l.cbuf0 + 0x30)?;
    if res == TARGET.load(Ordering::SeqCst) {
        // An index past the vocabulary (256): not a result of this launch.
        l.mem.write_u32(res, 256 + 3);
    }
    Ok(())
}

#[test]
fn t194e_bad_index_stops_as_t180e_and_drains() {
    let m = model();
    let want = one_by_one(&m, None);
    let at = 6usize;
    let (mut s, per_zone) = rig(&m, 0, 4);
    let rm = s.results.clone().unwrap();
    TARGET.store(rm.va + 16 * at as u64, Ordering::SeqCst);
    for x in s.models.iter_mut() {
        if x.0 == Prog::Argmax {
            x.1 = argmax_spoiled;
        }
    }
    let got = pipelined(&mut s, per_zone, 4, None, false);
    TARGET.store(0, Ordering::SeqCst);
    assert!(
        matches!(got.g.stop, Some(Stop::Eroare { pos, .. }) if pos == at),
        "{:?}",
        got.g.stop
    );
    assert_eq!(got.g.seq, want.g.seq[..at + 1].to_vec());
    // Batches 4..8 (with the bad slot) and 8..12 (sent before the stop was known): the second one drained.
    assert_eq!(got.run.drained_batches, 1);
    assert_eq!(got.run.drained_positions, 4);
    assert_eq!(got.run.sent, 3);
    // Every position after the bad slot saw NaN on the board: "no token".
    for p in at + 1..12 {
        let r = am::ArgmaxResult::from_words(got.words[p]);
        assert_eq!(r.token(m.vocab).unwrap(), None, "pozitia {p}");
    }
}

/// A `BatchIo` that refuses the send of batch `bad` (writes nothing) and passes the rest to the simulator.
struct RefuseSend<'p, 's, 'a> {
    inner: &'p mut PipeSim<'s, 'a>,
    k: usize,
    bad: usize,
}

impl BatchIo for RefuseSend<'_, '_, '_> {
    fn send(&mut self, zone: usize, first_pos: usize, inputs: &[Input]) -> GpuResult<()> {
        let k = self.k;
        self.k += 1;
        if k == self.bad {
            return Err(GpuError::InvalidArgument("plantat: trimitere refuzata"));
        }
        self.inner.send(zone, first_pos, inputs)
    }
    fn wait(&mut self, zone: usize, first_pos: usize, n: usize) -> GpuResult<Vec<TopOutput>> {
        self.inner.wait(zone, first_pos, n)
    }
}

#[test]
fn t194f_refused_send_stops_at_its_batch_after_the_one_before() {
    let m = model();
    let want = one_by_one(&m, None);
    let (mut s, per_zone) = rig(&m, 0, 4);
    let plan = GenPlan::new(&PROMPT, N, m.vocab, m.context).unwrap();
    let mut g = Generation::new(plan);
    let per = kb::launches_per_position(m.layers.len());
    let mut zones = TwoZones::new(per_zone).unwrap();
    let mut sim = PipeSim::new(&mut s, per_zone, SeqMode::EachLaunch);
    let mut io = RefuseSend {
        inner: &mut sim,
        k: 0,
        bad: 2,
    };
    let run = kb::run_pipelined(&mut g, &mut io, &mut zones, per, 4);
    // Batches 0 (0..4) and 1 (4..8) read normally; batch 2 (8..12) refused → error at position 8, nothing guessed.
    assert!(
        matches!(run.stop, Stop::Eroare { pos: 8, .. }),
        "{:?}",
        run.stop
    );
    assert_eq!(g.seq, want.g.seq[..9].to_vec());
    assert_eq!(run.sent, 2);
    assert_eq!(zones.in_flight(), 0);
    // A zone too small for the batch is refused the same way, before anything is written.
    let (mut s, _) = rig(&m, 0, 4);
    let plan = GenPlan::new(&PROMPT, N, m.vocab, m.context).unwrap();
    let mut g = Generation::new(plan);
    let mut zones = TwoZones::new(per * 4 - 1).unwrap();
    let mut io = PipeSim::new(&mut s, per * 4, SeqMode::EachLaunch);
    let run = kb::run_pipelined(&mut g, &mut io, &mut zones, per, 4);
    assert!(
        matches!(run.stop, Stop::Eroare { pos: 0, .. }),
        "{:?}",
        run.stop
    );
    assert_eq!(run.sent, 0);
    assert_eq!(s.submissions.len(), 0);
}
