//! C-157 point 2 — T206, SIMULARE: the second row of cuts for the whole training step (after os-386). The four tools
//! the OS already uses in generation (`kronos --gpu`, diets 1 and 2), applied to the resident training step of
//! `antrenare mini` (vocabulary 16 384, context 32, E 64, 4 heads, 2 layers, batch 2 × 32, AdamW lr 1e-3), 10 steps
//! on 10 different batches, on `gpu_sim::cmdproc`:
//! - A: the pushbuffer written through a [`WriteShadow`] (only the words that changed since the last step);
//! - B: the timestamp records zeroed once, not every step;
//! - C: only the first and the last record read, all of them on the profile steps (`RecordPlan::DIET`).
//!
//! Predictions written before the code: `docs/C-157_P2_TAIETURI_PAS.md` (K1–K5). Nothing here runs on the RTX 3070 Ti.

use std::sync::OnceLock;

use gpu_compute::train::{AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_FWD_NEEDS, MM_TRAIN_NEEDS, XENT_NEEDS,
    attn_back_on_cpu, attn_fwd_on_cpu, colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu,
    gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_fwd_on_cpu, mm_train_on_cpu, opt_step_on_cpu,
    vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::kronos_diet2::{RecordPlan, ends_span};
use nvidia_ga104::kronos_fast::WriteShadow;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{self as ts, PROGRAMS, Plan, Prog};
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

const VA: u64 = nvidia_ga104::mmu::VA_BASE;
const ZONE: u64 = VA + 0x20_0000;
const REC: u64 = VA + 0x30_0000;
/// The compute class is bound once, in its own stretch (in the OS: the channel's setup); the step's pushbuffer
/// follows at `PUSH`, as `WorkQueue::submit_fenced` places it once everything before is retired.
const SETUP: u64 = VA + 0x31_f000;
const PUSH: u64 = VA + 0x32_0000;
const PUSH_BYTES: u64 = 0x1_0000;
const PROGS: u64 = VA + 0x40_0000;
const HEAP: u64 = VA + 0x100_0000;
/// Payloads move by more than one step's records every submission (the OS: `TOKEN_FIRST + STRIDE * subs`).
const FIRST: u32 = 0x7000;
const STRIDE: u32 = 0x100;

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
}

fn initial() -> Vec<Vec<f32>> {
    let mut g = Rng(0x5eed_0000_0000_0206);
    MINI.tensors()
        .iter()
        .map(|t| {
            (0..t.elements())
                .map(|_| ((g.next() >> 11) as f64 / (1u64 << 53) as f64 * 0.07 - 0.035) as f32)
                .collect()
        })
        .collect()
}

fn batches() -> &'static Vec<(Vec<u32>, Vec<u32>)> {
    static D: OnceLock<Vec<(Vec<u32>, Vec<u32>)>> = OnceLock::new();
    D.get_or_init(|| {
        let mut g = Rng(0x5eed_0000_0000_2060);
        (0..STEPS)
            .map(|_| {
                let (mut inp, mut tgt) = (Vec::new(), Vec::new());
                for _ in 0..B {
                    let row: Vec<u32> = (0..=T)
                        .map(|_| (g.next() % MINI.vocab as u64) as u32)
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
    // The model's clock: a fixed cost per launch and per method, so the records carry growing times (SIMULARE).
    cp.sim_launch_ns = 3_000;
    cp.sim_method_ns = 10;
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
    let mut w = [0u32; 2];
    let mut pb = PushBuf::new(&mut w);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(SETUP, &bytes);
    let e = push::gpfifo_entry(SETUP, 2, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
    cp
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Way {
    /// `train_mini_resident` in os-386: records zeroed and the whole pushbuffer written every step, all records read.
    Today,
    /// Cuts A + B + C.
    Cuts,
}

/// What the driver did on one step (32-bit words).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct StepWords {
    push_written: usize,
    push_skipped: usize,
    rec_zeroed: usize,
    rec_read: usize,
    batch: usize,
    results_read: usize,
    /// The board's span of the step (ns): from all the records or from the two ends.
    span_ns: u64,
    /// The span from all the records too (the simulation can always look; the driver cannot after cut C).
    span_all_ns: u64,
}

#[derive(Clone, Debug, PartialEq)]
struct Run {
    losses: Vec<f32>,
    norms: Vec<f32>,
    weights: Vec<Vec<f32>>,
    steps: Vec<StepWords>,
    /// The pushbuffer area after the last step.
    push_area: Vec<u8>,
}

/// Writes the step's pushbuffer words at `PUSH`, whole or through the shadow (only the changed runs); returns
/// (written, skipped).
fn write_push(
    cp: &mut CommandProcessor,
    shadow: Option<&mut WriteShadow>,
    words: &[u32],
) -> (usize, usize) {
    let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
    match shadow {
        None => {
            cp.mem.write(PUSH, &bytes);
            (words.len(), 0)
        }
        Some(sh) => {
            let runs = sh.changes(PUSH, &bytes).unwrap();
            let mut n = 0;
            for (pa, r) in runs {
                n += r.len() / 4;
                cp.mem.write(pa, &bytes[r]);
            }
            sh.remember(PUSH, &bytes).unwrap();
            (n, words.len() - n)
        }
    }
}

fn run(way: Way) -> Run {
    let mut cp = board();
    let params = initial();
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let data = batches();
    let plan0 = Plan::new(MINI, B, T, &data[0].0, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    for (va, bytes) in plan0.uploads(&params, &data[0].0, &data[0].1).unwrap() {
        cp.mem.write(va, &bytes);
    }
    let mut shadow = (way == Way::Cuts).then(|| WriteShadow::new(PUSH, PUSH_BYTES).unwrap());
    let (mut losses, mut norms, mut steps) = (Vec::new(), Vec::new(), Vec::new());
    for (k, (x, y)) in data.iter().enumerate() {
        let mut sw = StepWords::default();
        let plan = plan0.rebatch(x).unwrap();
        if k > 0 {
            for (va, bytes) in plan.batch_uploads(x, y).unwrap() {
                sw.batch += bytes.len() / 4;
                cp.mem.write(va, &bytes);
            }
        }
        let list = plan.step(Recipe::AdamW513, &opt, k as u32 + 1).unwrap();
        let (zone, qmds) = ts::pack(&list, prog_va, ZONE).unwrap();
        assert!(zone.len() as u64 <= REC - ZONE);
        cp.mem.write(ZONE, &zone);
        let n = list.len();
        if way == Way::Today || k == 0 {
            cp.mem.write(REC, &vec![0u8; 16 * (n + 1)]);
            sw.rec_zeroed = 4 * (n + 1);
        }
        let first = FIRST + STRIDE * k as u32;
        let mut w = vec![0u32; transfer::sequence_words_fresh(n)];
        let mut pb = PushBuf::new(&mut w);
        transfer::encode_sequence_fresh(&mut pb, &qmds, REC, first).unwrap();
        let nw = pb.len();
        assert_eq!(nw, w.len());
        (sw.push_written, sw.push_skipped) = write_push(&mut cp, shadow.as_mut(), &w);
        let e = push::gpfifo_entry(PUSH, nw as u32, false).unwrap();
        cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
        // The OS's wait: the last record carries the step's last payload.
        let last = first + n as u32;
        assert_eq!(cp.mem.read_u32(REC + 16 * n as u64).unwrap(), last);
        let mut recs = vec![0u8; 16 * (n + 1)];
        cp.mem.read(REC, &mut recs).unwrap();
        sw.span_all_ns = transfer::analyze_sequence(&recs, n, first)
            .unwrap()
            .iter()
            .sum();
        let all = way == Way::Today || RecordPlan::DIET.reads_all(k);
        if all {
            sw.rec_read = 4 * (n + 1);
            sw.span_ns = sw.span_all_ns;
        } else {
            sw.rec_read = 8;
            sw.span_ns = ends_span(&recs[..16], &recs[16 * n..16 * (n + 1)], n, first).unwrap();
        }
        let [(lr, ln), (nr, nn)] = plan.step_reads();
        sw.results_read = ((ln + nn) / 4) as usize;
        let rows = cp.mem.read_f32s(lr, (ln / 4) as usize).unwrap();
        losses.push(ts::Outcome::mean_loss(&rows));
        norms.push(cp.mem.read_f32s(nr, (nn / 4) as usize).unwrap()[0]);
        steps.push(sw);
    }
    let weights = plan0
        .unflat(&cp.mem.read_f32s(plan0.w, plan0.n_params).unwrap())
        .unwrap();
    let mut push_area = vec![0u8; 4 * transfer::sequence_words_fresh(steps_launches())];
    cp.mem.read(PUSH, &mut push_area).unwrap();
    Run {
        losses,
        norms,
        weights,
        steps,
        push_area,
    }
}

fn steps_launches() -> usize {
    let plan0 = Plan::new(MINI, B, T, &batches()[0].0, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    plan0.launches_per_step(Recipe::AdamW513)
}

fn today() -> &'static Run {
    static R: OnceLock<Run> = OnceLock::new();
    R.get_or_init(|| run(Way::Today))
}

fn cuts() -> &'static Run {
    static R: OnceLock<Run> = OnceLock::new();
    R.get_or_init(|| run(Way::Cuts))
}

#[test]
fn t206_k1_same_numbers_bit_for_bit() {
    let (a, b) = (today(), cuts());
    assert_eq!(a.losses.len(), STEPS);
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&a.losses), bits(&b.losses), "pierderile");
    assert_eq!(bits(&a.norms), bits(&b.norms), "normele");
    for (ta, tb) in a.weights.iter().zip(&b.weights) {
        assert_eq!(bits(ta), bits(tb), "greutatile");
    }
    assert!(a.losses.iter().all(|l| l.is_finite()));
    assert_eq!(a.push_area, b.push_area, "pushbuffer-ul din memorie");
    println!(
        "T206 K1 (SIMULARE): 10 pasi, pierderi {:.4} .. {:.4}; cu taieturi = azi bit cu bit",
        a.losses[0],
        a.losses[STEPS - 1]
    );
}

#[test]
fn t206_k2_k3_words_per_step() {
    let n = steps_launches();
    assert_eq!(n, 75);
    let full = transfer::sequence_words_fresh(n);
    assert_eq!(full, 908);
    let (a, b) = (today(), cuts());
    for (k, (sa, sb)) in a.steps.iter().zip(&b.steps).enumerate() {
        // Today: everything, every step.
        assert_eq!(
            (sa.push_written, sa.push_skipped),
            (full, 0),
            "azi, pasul {k}"
        );
        assert_eq!(sa.rec_zeroed, 4 * (n + 1));
        assert_eq!(sa.rec_read, 4 * (n + 1));
        // K2: the shadow writes everything once, then only the records' payloads.
        if k == 0 {
            assert_eq!(sb.push_written, full);
        } else {
            assert_eq!(
                sb.push_written,
                n + 1,
                "pasul {k}: doar valorile inregistrarilor"
            );
            assert_eq!(sb.push_skipped, full - (n + 1));
        }
        // K3: zeroed once; two records except on the profile steps.
        assert_eq!(sb.rec_zeroed, if k == 0 { 4 * (n + 1) } else { 0 });
        let all = matches!(k, 0 | 1 | 8);
        assert_eq!(sb.rec_read, if all { 4 * (n + 1) } else { 8 }, "pasul {k}");
        // The same board span either way.
        assert_eq!(sb.span_ns, sb.span_all_ns, "pasul {k}");
        assert!(sb.span_ns > 0);
        assert_eq!((sa.batch, sa.results_read), (sb.batch, sb.results_read));
    }
    // K5: per step from step 2 on (the worst batch), at 0.66 µs per word written (and read back) and 0.5 per read.
    let worst =
        |r: &Run, f: fn(&StepWords) -> (usize, usize)| r.steps[2..].iter().map(f).max().unwrap();
    let io = |s: &StepWords| {
        (
            s.push_written + s.rec_zeroed + s.batch,
            s.rec_read + s.results_read,
        )
    };
    let (wa, ra) = worst(a, io);
    let (wb, rb) = (
        b.steps[2..].iter().map(|s| io(s).0).max().unwrap(),
        b.steps[2..]
            .iter()
            .filter(|s| s.rec_read == 8)
            .map(|s| io(s).1)
            .max()
            .unwrap(),
    );
    let us = |w: usize, r: usize| w as f64 * 0.66 + r as f64 * 0.5;
    let batch = b.steps[2..].iter().map(|s| s.batch).max().unwrap();
    println!("T206 K2/K3/K5 (SIMULARE + aritmetica), {n} lansari pe pas:");
    println!(
        "  azi:         scrise {wa:4} (pushbuffer {full}, inregistrari {}, lotul {batch}), citite {ra:3} -> ~{:.0} us",
        4 * (n + 1),
        us(wa, ra)
    );
    println!(
        "  cu taieturi: scrise {wb:4} (pushbuffer {}, inregistrari 0, lotul {batch}), citite {rb:3} -> ~{:.0} us",
        n + 1,
        us(wb, rb)
    );
    println!("  cu lotul in paralel (P10): ~{:.0} us", us(wb - batch, rb));
    assert!((1500..=1600).contains(&wa), "{wa}");
    assert!((390..=410).contains(&wb), "{wb}");
    assert_eq!(rb, 8 + 65);
    assert!((250.0..=350.0).contains(&us(wb, rb)));
    assert!(us(wa, ra) - us(wb, rb) > 800.0);
}

/// K4: without zeroing, a step that stopped half way is still caught — the last record keeps the previous step's
/// payload, so the wait never ends, the two-ends span refuses it and `records_seen` says how far the board got.
#[test]
fn t206_k4_step_stopped_half_way_is_caught() {
    let mut cp = board();
    let params = initial();
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let data = batches();
    let plan0 = Plan::new(MINI, B, T, &data[0].0, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    for (va, bytes) in plan0.uploads(&params, &data[0].0, &data[0].1).unwrap() {
        cp.mem.write(va, &bytes);
    }
    let mut shadow = WriteShadow::new(PUSH, PUSH_BYTES).unwrap();
    const STOP_AFTER: usize = 40;
    for (k, (x, y)) in data.iter().take(3).enumerate() {
        let plan = plan0.rebatch(x).unwrap();
        if k > 0 {
            for (va, bytes) in plan.batch_uploads(x, y).unwrap() {
                cp.mem.write(va, &bytes);
            }
        }
        let list = plan.step(Recipe::AdamW513, &opt, k as u32 + 1).unwrap();
        let (zone, qmds) = ts::pack(&list, prog_va, ZONE).unwrap();
        cp.mem.write(ZONE, &zone);
        let n = list.len();
        if k == 0 {
            cp.mem.write(REC, &vec![0u8; 16 * (n + 1)]);
        }
        let first = FIRST + STRIDE * k as u32;
        let mut w = vec![0u32; transfer::sequence_words_fresh(n)];
        let mut pb = PushBuf::new(&mut w);
        transfer::encode_sequence_fresh(&mut pb, &qmds, REC, first).unwrap();
        write_push(&mut cp, Some(&mut shadow), &w);
        // Step 2 stops after launch STOP_AFTER: the board only gets the words up to its record.
        let nw = if k == 2 {
            transfer::sequence_words_fresh(STOP_AFTER)
        } else {
            w.len()
        };
        let e = push::gpfifo_entry(PUSH, nw as u32, false).unwrap();
        cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
        let last = first + n as u32;
        let mut recs = vec![0u8; 16 * (n + 1)];
        cp.mem.read(REC, &mut recs).unwrap();
        let got = u32::from_le_bytes(recs[16 * n..16 * n + 4].try_into().unwrap());
        if k < 2 {
            assert_eq!(got, last);
            assert!(ends_span(&recs[..16], &recs[16 * n..], n, first).is_ok());
        } else {
            assert_ne!(got, last, "asteptarea nu se termina");
            assert_eq!(
                got,
                FIRST + STRIDE + n as u32,
                "ultima inregistrare = a pasului de dinainte"
            );
            assert!(ends_span(&recs[..16], &recs[16 * n..], n, first).is_err());
            assert!(transfer::analyze_sequence(&recs, n, first).is_err());
            assert_eq!(transfer::records_seen(&recs, n, first), STOP_AFTER + 1);
            println!(
                "T206 K4 (SIMULARE): pasul oprit dupa lansarea {STOP_AFTER}: ultima inregistrare {got:#x} (asteptat {last:#x}), prins; {} inregistrari ale pasului vazute",
                STOP_AFTER + 1
            );
        }
    }
}
