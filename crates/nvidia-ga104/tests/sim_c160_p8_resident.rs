//! C-160 P8 — T202, SIMULARE: the **resident arena** (`Plan::resident`, `rebatch`, `batch_uploads`, `step_reads`),
//! the library side of "the training arena in VRAM" (command centre's note of 23 Nov 23:36, step 2). On the shape of
//! the OS's `antrenare mini` (vocabulary 16 384, context 32, E 64, 4 heads, 2 layers, batch 2 × 32, AdamW lr 1e-3,
//! clipping 1.0), 10 steps on 10 **different** batches:
//! - the resident path: one full upload, then per step only the batch (~1.3 KiB) up and the losses + norm (260 B)
//!   down, the weights read once at the end;
//! - the path of os-380 (`gpu_i14.rs`, `train_mini_probe`): a new plan every step, everything uploaded, the weights
//!   and m, v read back every step;
//! - the CPU oracle (`gpu_compute::train`).
//!
//! Predictions written before the code: `docs/C-160_P8_ARENA_VRAM.md` (R1–R3). Nothing here runs on the RTX 3070 Ti.

use std::sync::OnceLock;

use gpu_compute::train::{self, AdamW, Arch, Batch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_FWD_NEEDS, MM_TRAIN_NEEDS, XENT_NEEDS,
    attn_back_on_cpu, attn_fwd_on_cpu, colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu,
    gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_fwd_on_cpu, mm_train_on_cpu, opt_step_on_cpu,
    vadd_on_cpu, xent_on_cpu,
};
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
        let mut g = Rng(0x5eed_0000_0000_0202);
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

/// The path of os-380: a new plan every step, everything up, weights and m, v down.
fn today() -> &'static Run {
    static R: OnceLock<Run> = OnceLock::new();
    R.get_or_init(|| {
        let mut cp = board();
        let mut params = initial();
        let mut adam = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
        let (mut first, mut up, mut down) = (0x7000u32, 0u64, 0u64);
        let (mut losses, mut norms) = (Vec::new(), Vec::new());
        for (x, y) in batches() {
            let plan = Plan::new(MINI, B, T, x, HEAP).unwrap();
            put(&mut cp, plan.uploads(&params, x, y).unwrap(), &mut up);
            let flat = |s: &[Vec<f32>]| {
                s.iter()
                    .flatten()
                    .flat_map(|v| v.to_le_bytes())
                    .collect::<Vec<u8>>()
            };
            put(
                &mut cp,
                vec![(plan.m, flat(&adam.m)), (plan.v, flat(&adam.v))],
                &mut up,
            );
            let t = adam.t + 1;
            submit(
                &mut cp,
                &mut first,
                &plan.step(Recipe::AdamW513, &adam, t).unwrap(),
            );
            let (loss, norm) = read_back(&cp, &plan, &mut down);
            losses.push(loss);
            norms.push(norm);
            let rd = |va: u64| {
                plan.unflat(&cp.mem.read_f32s(va, plan.n_params).unwrap())
                    .unwrap()
            };
            params = rd(plan.w);
            adam.m = rd(plan.m);
            adam.v = rd(plan.v);
            adam.t = t;
            down += 3 * 4 * plan.n_params as u64;
        }
        Run {
            losses,
            norms,
            weights: params,
            up,
            down,
        }
    })
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Fault {
    None,
    /// Planted: the batch uploads forgotten after step 1 (the launches of the new plan, the old batch in memory).
    NoBatchUploads,
}

/// The resident path: one plan, one full upload; per step only the batch up and losses + norm down.
fn resident(fault: Fault) -> Run {
    let mut cp = board();
    let params = initial();
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let data = batches();
    let plan0 = Plan::new(MINI, B, T, &data[0].0, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    let mut first = 0x7000u32;
    let mut once = 0u64;
    put(
        &mut cp,
        plan0.uploads(&params, &data[0].0, &data[0].1).unwrap(),
        &mut once,
    );
    assert_eq!(
        once,
        plan0.traffic().once_up,
        "urcarea de la pornire = traffic().once_up"
    );
    let (mut up, mut down) = (0u64, 0u64);
    let (mut losses, mut norms) = (Vec::new(), Vec::new());
    for (k, (x, y)) in data.iter().enumerate() {
        let plan = plan0.rebatch(x).unwrap();
        if k > 0 && fault != Fault::NoBatchUploads {
            put(&mut cp, plan.batch_uploads(x, y).unwrap(), &mut up);
        }
        submit(
            &mut cp,
            &mut first,
            &plan.step(Recipe::AdamW513, &opt, k as u32 + 1).unwrap(),
        );
        let (loss, norm) = read_back(&cp, &plan, &mut down);
        losses.push(loss);
        norms.push(norm);
    }
    let weights = plan0
        .unflat(&cp.mem.read_f32s(plan0.w, plan0.n_params).unwrap())
        .unwrap();
    Run {
        losses,
        norms,
        weights,
        up,
        down,
    }
}

fn resident_ok() -> &'static Run {
    static R: OnceLock<Run> = OnceLock::new();
    R.get_or_init(|| resident(Fault::None))
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

/// Every address of the plan (not the token segment count).
fn addresses(p: &Plan) -> Vec<u64> {
    let mut a = vec![
        p.base,
        p.w,
        p.g,
        p.m,
        p.v,
        p.tok,
        p.tgt,
        p.seg_tok.off,
        p.seg_tok.dst,
        p.seg_tok.rows,
        p.seg_pos.off,
        p.seg_pos.dst,
        p.seg_pos.rows,
        p.lnf.x,
        p.lnf.mean,
        p.lnf.rstd,
        p.hf,
        p.logits,
        p.dlogits,
        p.loss_rows,
        p.dx[0],
        p.dx[1],
        p.dtmp,
        p.dwide,
        p.dwide2,
        p.dqkv,
        p.ds,
        p.part,
        p.res,
        p.end,
    ];
    a.extend(&p.xs);
    for l in &p.layers {
        a.extend([
            l.ln1.x, l.ln1.mean, l.ln1.rstd, l.h1, l.qkv, l.p, l.att, l.ln2.x, l.ln2.mean,
            l.ln2.rstd, l.h2, l.u, l.act,
        ]);
    }
    a
}

#[test]
fn t202_r1_resident_addresses_do_not_depend_on_the_batch() {
    let data = batches();
    let plan0 = Plan::new(MINI, B, T, &data[0].0, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    assert!(plan0.is_resident());
    let mut worst = 0u64;
    let mut counts = Vec::new();
    for (x, _) in data {
        let p = plan0.rebatch(x).unwrap();
        assert_eq!(addresses(&p), addresses(&plan0), "o adresa depinde de lot");
        counts.push(p.seg_tok.count);
        worst = worst.max(Plan::new(MINI, B, T, x, HEAP).unwrap().bytes());
    }
    // The worst batch: 64 distinct symbols.
    let all: Vec<u32> = (0..(B * T) as u32).map(|i| i * 211).collect();
    let p = plan0.rebatch(&all).unwrap();
    assert_eq!(p.seg_tok.count, B * T);
    assert_eq!(addresses(&p), addresses(&plan0));
    let extra = plan0.bytes() - worst;
    println!(
        "T202 R1 (SIMULARE): arena rezidenta {} octeti ({:.1} MiB), cea mai mare arena pe lot {} (+{} octeti); simboluri distincte pe lot: {:?}",
        plan0.bytes(),
        plan0.bytes() as f64 / (1u64 << 20) as f64,
        worst,
        extra,
        counts
    );
    assert!(extra <= 1024, "arena rezidenta prea mare: +{extra}");
    assert!(
        counts.contains(&8),
        "lotul 3 trebuia sa aiba 8 simboluri distincte"
    );
}

#[test]
fn t202_r1_resident_keeps_programs_fusions_and_launch_count() {
    let x = &batches()[0].0;
    for p in [
        Plan::new(MINI, B, T, x, HEAP).unwrap(),
        Plan::new_tiles3(MINI, B, T, x, HEAP)
            .unwrap()
            .with_fused_back()
            .with_fused_gelu(),
        Plan::new_p_only_tiles(MINI, B, T, x, HEAP).unwrap(),
        Plan::new_tiles3_mv_ram(MINI, B, T, x, HEAP).unwrap(),
    ] {
        let r = p.clone().resident().unwrap();
        assert_eq!(r.programs(), p.programs());
        assert_eq!(r.fused_programs(), p.fused_programs());
        assert_eq!(r.p_relaunches(), p.p_relaunches());
        assert_eq!(r.mv_in_ram(), p.mv_in_ram());
        assert_eq!(
            r.launches_per_step(Recipe::Sgd),
            p.launches_per_step(Recipe::Sgd)
        );
        // Same batch → the same uploads, at the resident addresses.
        let y = &batches()[0].1;
        let params = initial();
        let a: Vec<Vec<u8>> = p
            .uploads(&params, x, y)
            .unwrap()
            .into_iter()
            .map(|u| u.1)
            .collect();
        let b: Vec<Vec<u8>> = r
            .uploads(&params, x, y)
            .unwrap()
            .into_iter()
            .map(|u| u.1)
            .collect();
        assert_eq!(a, b);
        // A resident plan stays as it is.
        assert_eq!(r.clone().resident().unwrap(), r);
    }
}

#[test]
fn t202_r2_resident_is_bit_for_bit_the_os380_path_and_the_oracle() {
    let (r, o) = (resident_ok(), today());
    let tr = Plan::new(MINI, B, T, &batches()[0].0, HEAP)
        .unwrap()
        .resident()
        .unwrap()
        .traffic();
    println!(
        "T202 R2 (SIMULARE): pierderi rezident {:?}\n  drumul os-380 {:?}",
        r.losses, o.losses
    );
    println!(
        "T202 R2 (SIMULARE): octeti pe legatura dupa pornire, {} pasi: rezident {} sus / {} jos; drumul os-380 {} sus / {} jos; traffic(): o data {} sus, pe pas {} sus / {} jos, la sfarsit {} jos; azi pe pas {} sus / {} jos",
        STEPS,
        r.up,
        r.down,
        o.up,
        o.down,
        tr.once_up,
        tr.step_up,
        tr.step_down,
        tr.final_down,
        tr.today_step_up,
        tr.today_step_down
    );
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    assert_eq!(bits(&r.losses), bits(&o.losses), "pierderile difera");
    assert_eq!(bits(&r.norms), bits(&o.norms), "normele difera");
    assert!(r.weights == o.weights, "greutatile difera");
    // The oracle (os-380's verdict).
    let (ol, ow) = oracle();
    let worst_loss = r
        .losses
        .iter()
        .zip(ol)
        .map(|(a, b)| ((*a as f64 - *b as f64) / (*b as f64).abs()).abs())
        .fold(0.0f64, f64::max);
    let dw = rel_dev(&r.weights, ow);
    println!("T202 R2 (SIMULARE): fata de oracol: pierderea {worst_loss:.1e}, greutatile {dw:.1e}");
    assert!(worst_loss <= TOL && dw <= TOL);
    // The bytes: after the first upload, per step only the batch up and losses + norm down.
    assert!(r.up <= (STEPS as u64 - 1) * tr.step_up);
    assert_eq!(r.down, STEPS as u64 * tr.step_down);
    assert!(tr.step_up < 2048 && tr.step_down == 4 * (B * T) as u64 + 4);
    assert!(o.up > 1000 * r.up && o.down > 1000 * r.down);
}

#[test]
fn t202_r3_forgotten_batch_uploads_are_caught() {
    let bad = resident(Fault::NoBatchUploads);
    let good = resident_ok();
    assert_eq!(bad.losses[0].to_bits(), good.losses[0].to_bits());
    let differ = bad
        .losses
        .iter()
        .zip(&good.losses)
        .skip(1)
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    println!(
        "T202 R3 (SIMULARE): fara urcarile lotului: {differ} din {} pierderi diferite",
        STEPS - 1
    );
    assert_eq!(differ, STEPS - 1);
    assert!(bad.weights != good.weights);
}

#[test]
fn t202_r3_refusals() {
    let (x, y) = &batches()[0];
    let plain = Plan::new(MINI, B, T, x, HEAP).unwrap();
    assert!(!plain.is_resident());
    assert!(plain.rebatch(x).is_err(), "lot nou pe un plan nerezident");
    assert!(plain.batch_uploads(x, y).is_err());
    let r = plain.resident().unwrap();
    let (x2, y2) = &batches()[1];
    assert!(
        r.batch_uploads(x2, y2).is_err(),
        "alt lot decat al planului"
    );
    assert!(r.rebatch(x2).unwrap().batch_uploads(x2, y2).is_ok());
    let mut out = x2.clone();
    out[5] = MINI.vocab as u32;
    assert!(r.rebatch(&out).is_err(), "simbol in afara vocabularului");
    assert!(r.rebatch(&x2[1..]).is_err(), "alta forma");
    let mut bad_t = y2.clone();
    bad_t[0] = MINI.vocab as u32;
    assert!(r.rebatch(x2).unwrap().batch_uploads(x2, &bad_t).is_err());
}
