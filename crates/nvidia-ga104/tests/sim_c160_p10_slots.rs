//! Order 20:11 step 5 — T204, SIMULARE: **the next batch prepared in parallel with the training step** (two batch
//! slots: `Plan::with_batch_slots`, `slot`, `slot_uploads`). On the shape of the OS's `antrenare mini` (vocabulary
//! 16 384, context 32, E 64, 4 heads, 2 layers, batch 2 × 32, AdamW lr 1e-3, clipping 1.0), 10 steps on 10 different
//! batches:
//! - the resident path step by step (T202): write the batch, run the step, read the losses;
//! - the "parallel" path: the next batch written **in the middle** of the current step (its launches submitted in two
//!   halves, the write between them) and the losses of a step read only **after** the next step has started — with two
//!   slots (the board and the CPU never touch the same slot), and, as a planted mistake, with one slot.
//!
//! The simulated board runs every submission at once, so "in parallel" is modelled by the order of the CPU's writes
//! and reads between the halves. Predictions written before the code: `docs/C-160_P10_LOT_IN_PARALEL.md` (L1–L3).
//! Nothing here runs on the RTX 3070 Ti.

use std::sync::OnceLock;

use gpu_compute::train::{self, AdamW, Arch, Batch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS, LN_FWD_NEEDS,
    MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, XENT_NEEDS, attn_back_on_cpu, attn_fwd_on_cpu,
    colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu,
    ln_back_res_on_cpu, ln_fwd_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu, opt_step_on_cpu,
    vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{self as ts, BatchVas, PROGRAMS, Plan, Prog};
use nvidia_ga104::transfer;
use nvidia_rm570::regs::clc7c0 as c;

const MINI: Arch = Arch {
    vocab: 16_384,
    ctx: 32,
    n_emb: 64,
    heads: 4,
    layers: 2,
};
const B: usize = 2;
const T: usize = 32;
const STEPS: usize = 10;
const LR: f64 = 1e-3;
const INIT_HALF_WIDTH: f64 = 0.034_641_016;
/// The OS's tolerance against the oracle (os-380).
const TOL: f64 = 1e-3;

const VA: u64 = nvidia_ga104::mmu::VA_BASE;
const ZONE: u64 = VA + 0x20_0000;
const REC: u64 = VA + 0x30_0000;
const PUSH: u64 = VA + 0x32_0000;
const PROGS: u64 = VA + 0x40_0000;
const HEAP: u64 = VA + 0x100_0000;

fn prog_va(p: Prog) -> u64 {
    PROGS + 0x1_0000 * p.index() as u64
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn initial() -> Vec<Vec<f32>> {
    let mut g = Rng(0x00c1_9e10_2026_1124);
    MINI.tensors()
        .iter()
        .map(|t| {
            let n = t.elements();
            let leaf = t.name.rsplit('.').nth(1).unwrap_or("");
            if t.name.ends_with("bias") {
                vec![0.0f32; n]
            } else if leaf.starts_with("ln") {
                vec![1.0f32; n]
            } else {
                (0..n)
                    .map(|_| ((g.unit() * 2.0 - 1.0) * INIT_HALF_WIDTH) as f32)
                    .collect()
            }
        })
        .collect()
}

/// `STEPS` different batches (two rows of 33 symbols each). Batch 3 repeats symbols on purpose (only 8 distinct), so
/// the token segment count changes a lot between steps.
fn batches() -> &'static Vec<(Vec<u32>, Vec<u32>)> {
    static D: OnceLock<Vec<(Vec<u32>, Vec<u32>)>> = OnceLock::new();
    D.get_or_init(|| {
        let mut g = Rng(0x5eed_0000_0000_0204);
        (0..STEPS)
            .map(|k| {
                let (mut inp, mut tgt) = (Vec::new(), Vec::new());
                for _ in 0..B {
                    let row: Vec<u32> = (0..=T)
                        .map(|_| {
                            let r = g.next();
                            if k == 3 {
                                (r % 8) as u32 * 997
                            } else {
                                (r % MINI.vocab as u64) as u32
                            }
                        })
                        .collect();
                    inp.extend_from_slice(&row[..T]);
                    tgt.extend_from_slice(&row[1..]);
                }
                (inp, tgt)
            })
            .collect()
    })
}

fn board() -> CommandProcessor {
    ts::check_programs().expect("SHA-256 programe");
    let mut cp = CommandProcessor::new();
    for p in PROGRAMS {
        let code = p.code();
        match p {
            Prog::MmTrain => cp.register_program_with_needs(code, mm_train_on_cpu, MM_TRAIN_NEEDS),
            Prog::Xent => cp.register_program_with_needs(code, xent_on_cpu, XENT_NEEDS),
            Prog::LnFwd => cp.register_program_with_needs(code, ln_fwd_on_cpu, LN_FWD_NEEDS),
            Prog::LnBack => cp.register_program_with_needs(code, ln_back_on_cpu, LN_BACK_NEEDS),
            Prog::Colsum => cp.register_program(code, colsum_on_cpu),
            Prog::GeluTr => cp.register_program(code, gelu_tr_on_cpu),
            Prog::AttnFwd => cp.register_program(code, attn_fwd_on_cpu),
            Prog::AttnBack => cp.register_program(code, attn_back_on_cpu),
            Prog::EmbFwd => cp.register_program(code, emb_fwd_on_cpu),
            Prog::EmbBack => cp.register_program(code, emb_back_on_cpu),
            Prog::Gnorm => cp.register_program_with_needs(code, gnorm_on_cpu, GNORM_NEEDS),
            Prog::OptStep => cp.register_program(code, opt_step_on_cpu),
            Prog::Vadd => cp.register_program(code, vadd_on_cpu),
            other => panic!("program neasteptat in PROGRAMS: {other:?}"),
        };
        cp.mem.write(prog_va(p), code);
    }
    // The two fused programs (order 20:11 step 3), for the plans with both fusions.
    for p in [Prog::LnBackRes, Prog::MmTrainGelu] {
        let code = p.code();
        match p {
            Prog::LnBackRes => {
                cp.register_program_with_needs(code, ln_back_res_on_cpu, LN_BACK_RES_NEEDS)
            }
            _ => cp.register_program_with_needs(code, mm_train_gelu_on_cpu, MM_TRAIN_GELU_NEEDS),
        };
        cp.mem.write(prog_va(p), code);
    }
    cp
}

fn submit(cp: &mut CommandProcessor, first: &mut u32, launches: &[ts::Launch]) {
    let (zone, qmds) = ts::pack(launches, prog_va, ZONE).unwrap();
    assert!(zone.len() as u64 <= REC - ZONE, "zona de control prea mare");
    cp.mem.write(ZONE, &zone);
    let n = launches.len();
    cp.mem.write(REC, &vec![0u8; 16 * (n + 1)]);
    let mut w = vec![0u32; 2 + transfer::sequence_words_fresh(n)];
    let mut pb = PushBuf::new(&mut w);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    transfer::encode_sequence_fresh(&mut pb, &qmds, REC, *first).unwrap();
    let nw = pb.len();
    let bytes: Vec<u8> = w[..nw].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, nw as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
    *first = first.wrapping_add(n as u32 + 1);
}

#[derive(Clone, Debug, PartialEq)]
struct Run {
    losses: Vec<f32>,
    norms: Vec<f32>,
    weights: Vec<Vec<f32>>,
    /// Bytes the CPU wrote / read around the steps (after the first full upload, for the resident path).
    up: u64,
    down: u64,
}

fn put(cp: &mut CommandProcessor, set: Vec<(u64, Vec<u8>)>, up: &mut u64) {
    for (va, bytes) in set {
        *up += bytes.len() as u64;
        cp.mem.write(va, &bytes);
    }
}

fn read_back(cp: &CommandProcessor, plan: &Plan, down: &mut u64) -> (f32, f32) {
    let [(lr, ln), (nr, nn)] = plan.step_reads();
    *down += ln + nn;
    let rows = cp.mem.read_f32s(lr, (ln / 4) as usize).unwrap();
    let norm = cp.mem.read_f32s(nr, (nn / 4) as usize).unwrap()[0];
    (ts::Outcome::mean_loss(&rows), norm)
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    /// `Plan::new` + `resident`.
    Plain,
    /// `Plan::new` + both fusions + `resident` + `with_recompute` (order 20:11 steps 3–4).
    FusedRecompute,
}

fn base_plan(kind: Kind) -> Plan {
    let x = &batches()[0].0;
    let p = Plan::new(MINI, B, T, x, HEAP).unwrap();
    match kind {
        Kind::Plain => p.resident().unwrap(),
        Kind::FusedRecompute => p
            .with_fused_back()
            .with_fused_gelu()
            .resident()
            .unwrap()
            .with_recompute()
            .unwrap(),
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Path {
    /// T202: write the batch, run the step, read its losses — one slot, nothing in parallel.
    StepByStep,
    /// The parallel order with two slots.
    Parallel,
    /// Planted: the parallel order with one slot.
    ParallelOneSlot,
}

fn run(kind: Kind, path: Path) -> Run {
    let mut cp = board();
    let params = initial();
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let data = batches();
    let one = base_plan(kind);
    let two = one.clone().with_batch_slots().unwrap();
    // The plan of step k (its slot, its batch).
    let plan_of = |k: usize| -> Plan {
        let p = match path {
            Path::Parallel => two.slot(k % 2).unwrap(),
            _ => one.clone(),
        };
        p.rebatch(&data[k].0).unwrap()
    };
    let p0 = plan_of(0);
    let mut first = 0x7000u32;
    let mut once = 0u64;
    put(
        &mut cp,
        p0.uploads(&params, &data[0].0, &data[0].1).unwrap(),
        &mut once,
    );
    if path == Path::Parallel {
        put(&mut cp, two.slot(1).unwrap().slot_uploads(), &mut once);
    }
    let (mut up, mut down) = (0u64, 0u64);
    let (mut losses, mut norms) = (Vec::new(), Vec::new());
    for k in 0..STEPS {
        let plan = plan_of(k);
        let list = plan.step(Recipe::AdamW513, &opt, k as u32 + 1).unwrap();
        match path {
            Path::StepByStep => {
                if k > 0 {
                    put(
                        &mut cp,
                        plan.batch_uploads(&data[k].0, &data[k].1).unwrap(),
                        &mut up,
                    );
                }
                submit(&mut cp, &mut first, &list);
                let (l, n) = read_back(&cp, &plan, &mut down);
                losses.push(l);
                norms.push(n);
            }
            Path::Parallel | Path::ParallelOneSlot => {
                let half = list.len() / 2;
                submit(&mut cp, &mut first, &list[..half]);
                // The CPU while the board runs step k: the losses of step k − 1 (its slot is free now), then the
                // batch of step k + 1 into that same slot.
                if k > 0 {
                    let (l, n) = read_back(&cp, &plan_of(k - 1), &mut down);
                    losses.push(l);
                    norms.push(n);
                }
                if k + 1 < STEPS {
                    let next = plan_of(k + 1);
                    put(
                        &mut cp,
                        next.batch_uploads(&data[k + 1].0, &data[k + 1].1).unwrap(),
                        &mut up,
                    );
                }
                submit(&mut cp, &mut first, &list[half..]);
            }
        }
    }
    if path != Path::StepByStep {
        let (l, n) = read_back(&cp, &plan_of(STEPS - 1), &mut down);
        losses.push(l);
        norms.push(n);
    }
    let weights = one
        .unflat(&cp.mem.read_f32s(one.w, one.n_params).unwrap())
        .unwrap();
    Run {
        losses,
        norms,
        weights,
        up,
        down,
    }
}

type Cache = std::sync::Mutex<Vec<((Kind, Path), &'static Run)>>;

fn cached(kind: Kind, path: Path) -> &'static Run {
    static R: OnceLock<Cache> = OnceLock::new();
    let m = R.get_or_init(Default::default);
    if let Some((_, r)) = m.lock().unwrap().iter().find(|(k, _)| *k == (kind, path)) {
        return r;
    }
    let r: &'static Run = Box::leak(Box::new(run(kind, path)));
    m.lock().unwrap().push(((kind, path), r));
    r
}

fn bits(r: &Run) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    (
        r.losses.iter().map(|x| x.to_bits()).collect(),
        r.norms.iter().map(|x| x.to_bits()).collect(),
        r.weights.iter().flatten().map(|x| x.to_bits()).collect(),
    )
}

/// Replace every batch-slot-1 address (and the `res` sub-addresses) in `cb` by its slot-0 twin.
fn to_slot0(cb: &[u8; 0x100], s0: &BatchVas, s1: &BatchVas) -> [u8; 0x100] {
    let mut pairs = vec![
        (s1.tok, s0.tok),
        (s1.tgt, s0.tgt),
        (s1.seg_tok.off, s0.seg_tok.off),
        (s1.seg_tok.dst, s0.seg_tok.dst),
        (s1.seg_tok.rows, s0.seg_tok.rows),
        (s1.loss_rows, s0.loss_rows),
    ];
    for d in [0u64, 4, 8, 12, 16, 20] {
        pairs.push((s1.res + d, s0.res + d));
    }
    let mut out = *cb;
    for at in 0..=(0x100 - 8) {
        let w = u64::from_le_bytes(out[at..at + 8].try_into().unwrap());
        if let Some(&(_, to)) = pairs.iter().find(|(from, _)| *from == w) {
            out[at..at + 8].copy_from_slice(&to.to_le_bytes());
        }
    }
    out
}

fn oracle() -> &'static (Vec<f32>, Vec<Vec<f32>>) {
    static O: OnceLock<(Vec<f32>, Vec<Vec<f32>>)> = OnceLock::new();
    O.get_or_init(|| {
        let mut p = initial();
        let mut adam = AdamW::new(&p, LR, 0.9, 0.999, 1e-8, 0.1);
        let mut losses = Vec::new();
        for (x, y) in batches() {
            let b = Batch {
                inputs: x,
                targets: y,
                batch: B,
                len: T,
            };
            let (loss, mut g) = train::loss_and_grad(&MINI, &p, &b).unwrap();
            losses.push(loss);
            train::clip_grad_norm(&mut g, 1.0);
            adam.step(&mut p, &g).unwrap();
        }
        (losses, p)
    })
}

fn rel_dev(a: &[Vec<f32>], b: &[Vec<f32>]) -> f64 {
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
        d += (*x as f64 - *y as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(1e-30)).sqrt()
}

/// L1: slot 0 = the resident plan's addresses; slot 1 after the old end; +3 840 bytes; the launch lists of the two
/// slots differ only in the batch addresses.
#[test]
fn t204_l1_two_slots_layout() {
    for kind in [Kind::Plain, Kind::FusedRecompute] {
        let one = base_plan(kind);
        let two = one.clone().with_batch_slots().unwrap();
        assert_eq!(one.batch_slots(), 1);
        assert_eq!(two.batch_slots(), 2);
        assert_eq!(two.end - one.end, 3_840, "{kind:?}: arena +3 840 B");
        let s0 = two.slot(0).unwrap();
        let s1 = two.slot(1).unwrap();
        assert_eq!(
            s0.batch_area(),
            one.batch_area(),
            "{kind:?}: locul 0 = planul rezident"
        );
        assert_eq!(s0.w, one.w);
        assert_eq!(s0.logits, one.logits);
        assert_eq!(s0.xs, one.xs);
        let a1 = s1.batch_area();
        for va in [
            a1.tok,
            a1.tgt,
            a1.seg_tok.off,
            a1.seg_tok.dst,
            a1.seg_tok.rows,
            a1.loss_rows,
            a1.res,
        ] {
            assert!(
                va >= one.end && va < two.end,
                "{kind:?}: locul 1 dupa capatul vechi"
            );
        }
        // Again on a slot plan: unchanged; with_recompute keeps the slots.
        assert_eq!(two.clone().with_batch_slots().unwrap().end, two.end);
        assert_eq!(two.clone().with_recompute().unwrap().batch_slots(), 2);
        let opt = AdamW::new(&initial(), LR, 0.9, 0.999, 1e-8, 0.1);
        for (k, (x, _)) in batches().iter().enumerate().take(3) {
            let l_one = one
                .rebatch(x)
                .unwrap()
                .step(Recipe::AdamW513, &opt, k as u32 + 1)
                .unwrap();
            let l0 = s0
                .rebatch(x)
                .unwrap()
                .step(Recipe::AdamW513, &opt, k as u32 + 1)
                .unwrap();
            let l1 = s1
                .rebatch(x)
                .unwrap()
                .step(Recipe::AdamW513, &opt, k as u32 + 1)
                .unwrap();
            assert_eq!(l0, l_one, "{kind:?}: locul 0 = aceleasi lansari");
            assert_eq!(l1.len(), l0.len());
            let mut differ = 0;
            for (a, b) in l1.iter().zip(&l0) {
                assert_eq!(
                    (a.prog, a.grid, a.tag, a.layer),
                    (b.prog, b.grid, b.tag, b.layer)
                );
                differ += usize::from(a.cbuf != b.cbuf);
                assert_eq!(
                    to_slot0(&a.cbuf, &s0.batch_area(), &a1),
                    b.cbuf,
                    "{kind:?} {}: doar adresele lotului difera",
                    a.tag
                );
            }
            assert!(
                differ >= 3,
                "{kind:?}: emb, xent, emb_back, norma folosesc locul"
            );
        }
    }
}

/// L2: the parallel path with two slots = the step-by-step path, bit for bit (plain and fused + recompute).
#[test]
fn t204_l2_parallel_same_bits() {
    for kind in [Kind::Plain, Kind::FusedRecompute] {
        let a = cached(kind, Path::StepByStep);
        let b = cached(kind, Path::Parallel);
        assert_eq!(a.losses.len(), STEPS);
        assert_eq!(b.losses.len(), STEPS);
        assert_eq!(
            bits(a),
            bits(b),
            "{kind:?}: in paralel = pas cu pas, bit cu bit"
        );
        assert_eq!(b.down, a.down, "{kind:?}: aceleasi citiri");
        assert!(a.losses.iter().all(|l| l.is_finite() && *l > 0.0));
        let (ol, ow) = oracle();
        let dl = a
            .losses
            .iter()
            .zip(ol)
            .map(|(x, y)| ((*x as f64 - *y as f64) / *y as f64).abs())
            .fold(0.0, f64::max);
        let dw = rel_dev(&b.weights, ow);
        eprintln!(
            "T204 L2 {kind:?}: pierderi fata de oracol {dl:.2e}, greutati {dw:.2e}, sus {} B, jos {} B",
            b.up, b.down
        );
        assert!(dl < TOL && dw < TOL, "{kind:?}: oracolul");
    }
    // The plain step-by-step path is T202's path (same plan, same order) — it learns: loss finite, weights moved.
    let a = cached(Kind::Plain, Path::StepByStep);
    let moved = a
        .weights
        .iter()
        .flatten()
        .zip(initial().iter().flatten())
        .filter(|(x, y)| x != y)
        .count();
    assert!(moved > 100_000, "greutatile s-au miscat: {moved}");
}

/// L3: one slot in the parallel order → other numbers; the refusals.
#[test]
fn t204_l3_planted_one_slot() {
    for kind in [Kind::Plain, Kind::FusedRecompute] {
        let a = cached(kind, Path::StepByStep);
        let bad = cached(kind, Path::ParallelOneSlot);
        let diff_l = a
            .losses
            .iter()
            .zip(&bad.losses)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        let diff_w = a
            .weights
            .iter()
            .flatten()
            .zip(bad.weights.iter().flatten())
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();
        eprintln!(
            "T204 L3 {kind:?}: un singur loc: {diff_l}/{STEPS} pierderi, {diff_w} greutati alti biti"
        );
        assert!(
            diff_l >= STEPS - 1,
            "{kind:?}: un singur loc -> alte pierderi ({diff_l})"
        );
        assert!(
            diff_w > 1_000,
            "{kind:?}: un singur loc -> alte greutati ({diff_w})"
        );
    }
    let one = base_plan(Kind::Plain);
    assert!(one.slot(0).is_err(), "planul cu un loc refuza slot()");
    let two = one.clone().with_batch_slots().unwrap();
    assert!(two.slot(2).is_err(), "locul 2 nu exista");
    let x = &batches()[0].0;
    let not_resident = Plan::new(MINI, B, T, x, HEAP).unwrap();
    assert!(
        not_resident.with_batch_slots().is_err(),
        "doua locuri cer un plan rezident"
    );
}
