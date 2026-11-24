//! C-157 point 2 after os-392 — T210: the per-launch profile of the training step (`nvidia_ga104::train_profile`).
//! Predictions written before the code: `docs/C-157_P2_PROFIL_PAS.md` (Q1–Q3; Q4 is for the board).
//!
//! - Q1/Q2 on host arithmetic with synthetic times (distinct per launch, so a wrong grouping shows);
//! - Q1 end to end in SIMULARE: `antrenare mini` (vocabulary 16 384, context 32, E 64, 4 heads, 2 layers, batch 2 × 32,
//!   AdamW), 3 steps with the batch through RAM (`batch_ram`), unfused and fused, on `gpu_sim::cmdproc` — the records
//!   the model writes go through `StepProfile::add_records`. The model's clock is a fixed cost per launch and per
//!   method: its numbers test the counting and say nothing about the RTX 3070 Ti.
//! - Q3: the fused list changes only the fused tags.

use gpu_compute::train::{AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS, LN_FWD_NEEDS,
    MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, XENT_NEEDS, attn_back_on_cpu, attn_fwd_on_cpu,
    colsum_on_cpu, copy_b128_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu,
    ln_back_on_cpu, ln_back_res_on_cpu, ln_fwd_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu,
    opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::batch_ram as br;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_profile::{COPY_DOWN, COPY_UP, StepProfile};
use nvidia_ga104::train_step::{self as ts, PROGRAMS, Plan, Prog};
use nvidia_ga104::transfer::{self, COPY_B128};
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

const VA: u64 = nvidia_ga104::mmu::VA_BASE;
const ZONE: u64 = VA + 0x20_0000;
const REC: u64 = VA + 0x30_0000;
const SETUP: u64 = VA + 0x31_f000;
const PUSH: u64 = VA + 0x32_0000;
const PROGS: u64 = VA + 0x40_0000;
const STAGE: u64 = VA + 0x38_0000;
const COPY: u64 = VA + 0x3f_0000;
const HEAP: u64 = VA + 0x100_0000;
const FIRST: u32 = 0x7000;
const STRIDE: u32 = 0x100;

fn prog_va(p: Prog) -> u64 {
    PROGS + 0x1_0000 * p.index() as u64
}

fn batch(seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mut s = seed;
    let mut sym = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 33) % MINI.vocab as u64) as u32
    };
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for _ in 0..B {
        let row: Vec<u32> = (0..=T).map(|_| sym()).collect();
        x.extend_from_slice(&row[..T]);
        y.extend_from_slice(&row[1..]);
    }
    (x, y)
}

fn params() -> Vec<Vec<f32>> {
    MINI.tensors()
        .iter()
        .enumerate()
        .map(|(i, t)| {
            (0..t.elements())
                .map(|k| (((k * 31 + i * 7) % 97) as f32 - 48.0) * 4e-4)
                .collect()
        })
        .collect()
}

fn plan(fused: bool) -> Plan {
    let p = Plan::new(MINI, B, T, &batch(1).0, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    if fused {
        p.with_fused_back().with_fused_gelu()
    } else {
        p
    }
}

fn list(fused: bool) -> Vec<ts::Launch> {
    let p = params();
    let opt = AdamW::new(&p, 1e-3, 0.9, 0.999, 1e-8, 0.1);
    plan(fused).step(Recipe::AdamW513, &opt, 1).unwrap()
}

// ------------------------------------------------------------------------------------------------ Q1/Q2 host

#[test]
fn t210_q1_groups_add_up_on_synthetic_times() {
    let l = list(false);
    assert_eq!(l.len(), 75);
    let mut p = StepProfile::from_launches(&l).unwrap();
    // Three steps, every launch a different time; the middle step is the median of every group (k = 1).
    for k in 0..3u64 {
        let ns: Vec<u64> = (0..l.len() as u64)
            .map(|i| 1_000 + 37 * i + 5_000 * k)
            .collect();
        p.add_step(&ns).unwrap();
    }
    let total: u64 = (0..75u64).map(|i| 1_000 + 37 * i + 5_000).sum();
    assert_eq!(p.total_ns(), total);
    for groups in [p.by_tag(), p.by_prog(), p.by_stage()] {
        assert_eq!(groups.iter().map(|g| g.median_ns).sum::<u64>(), total);
        assert_eq!(groups.iter().map(|g| g.launches).sum::<u32>(), 75);
        assert!(groups.windows(2).all(|w| w[0].median_ns >= w[1].median_ns));
    }
    // The tag of a layer launch adds both layers: `inapoi:proj-dX` is launch 28 (layer 1) and 49 (layer 0).
    let pdx = p
        .by_tag()
        .into_iter()
        .find(|g| g.name == "inapoi:proj-dX")
        .unwrap();
    assert_eq!(pdx.launches, 2);
    assert_eq!(
        pdx.median_ns,
        (1_000 + 37 * 28 + 5_000) + (1_000 + 37 * 49 + 5_000)
    );
    // Stages: 23 forward launches (embedding, 2 × 10, ln-final, logit), 1 loss, 48 backward, 3 optimizer.
    let st = p.by_stage();
    let n = |s: &str| st.iter().find(|g| g.name == s).unwrap().launches;
    assert_eq!(
        (n("inainte"), n("eroarea"), n("inapoi"), n("opt")),
        (23, 1, 48, 3)
    );
    let lines = p.lines(5);
    println!("{}", lines.join("\n"));
    assert!(lines[0].contains("75 lansari"));
}

#[test]
fn t210_q2_refusals() {
    let l = list(false);
    let mut p = StepProfile::from_launches(&l).unwrap();
    assert!(p.add_step(&[1; 74]).is_err());
    // Records: one missing, then the clock going back.
    let n = l.len();
    let mut rec = vec![0u8; 16 * (n + 1)];
    for k in 0..=n {
        rec[16 * k..16 * k + 4].copy_from_slice(&(FIRST + k as u32).to_le_bytes());
        rec[16 * k + 8..16 * k + 16].copy_from_slice(&(1_000 * k as u64).to_le_bytes());
    }
    p.add_records(&rec, FIRST).unwrap();
    let mut missing = rec.clone();
    missing[16 * 40..16 * 40 + 4].copy_from_slice(&0u32.to_le_bytes());
    assert!(p.add_records(&missing, FIRST).is_err());
    let mut back = rec.clone();
    back[16 * 41 + 8..16 * 41 + 16].copy_from_slice(&5u64.to_le_bytes());
    assert!(p.add_records(&back, FIRST).is_err());
    assert_eq!(p.steps(), 1, "only the good step is kept");
    assert!(StepProfile::from_launches(&[]).is_err());
    assert!(p.clone().with_copies(5, 2).is_err(), "copies after steps");
    assert!(StepProfile::from_launches(&l).unwrap().lines(3).is_empty());
}

#[test]
fn t210_q3_fused_changes_only_fused_tags() {
    let tags = |l: &[ts::Launch]| {
        let p = StepProfile::from_launches(l).unwrap();
        let mut p = p;
        p.add_step(&vec![1; l.len()]).unwrap();
        let mut v: Vec<(&str, u32)> = p.by_tag().iter().map(|g| (g.name, g.launches)).collect();
        v.sort();
        v
    };
    let (a, b) = (tags(&list(false)), tags(&list(true)));
    let gone: Vec<_> = a.iter().filter(|x| !b.contains(x)).collect();
    let new: Vec<_> = b.iter().filter(|x| !a.contains(x)).collect();
    println!("dispar: {gone:?}\napar: {new:?}");
    assert_eq!(
        gone,
        [
            &("inapoi:gelu", 2),
            &("inapoi:ln1-dx", 2),
            &("inapoi:ln2-dx", 2),
            &("inapoi:proj-dX", 2),
            &("inapoi:rezidual-ln1", 2),
            &("inapoi:rezidual-ln2", 2),
        ]
    );
    assert_eq!(new.len(), 3);
    assert!(new.iter().all(|(t, n)| *n == 2 && t.contains('+')));
}

// ------------------------------------------------------------------------------------------------ Q1 in SIMULARE

fn board() -> CommandProcessor {
    ts::check_programs().expect("SHA-256 programe");
    let mut cp = CommandProcessor::new();
    cp.sim_launch_ns = 3_000;
    cp.sim_method_ns = 10;
    for p in PROGRAMS
        .iter()
        .copied()
        .chain([Prog::LnBackRes, Prog::MmTrainGelu])
    {
        let code = p.code();
        match p {
            Prog::MmTrain => cp.register_program_with_needs(code, mm_train_on_cpu, MM_TRAIN_NEEDS),
            Prog::MmTrainGelu => {
                cp.register_program_with_needs(code, mm_train_gelu_on_cpu, MM_TRAIN_GELU_NEEDS)
            }
            Prog::Xent => cp.register_program_with_needs(code, xent_on_cpu, XENT_NEEDS),
            Prog::LnFwd => cp.register_program_with_needs(code, ln_fwd_on_cpu, LN_FWD_NEEDS),
            Prog::LnBack => cp.register_program_with_needs(code, ln_back_on_cpu, LN_BACK_NEEDS),
            Prog::LnBackRes => {
                cp.register_program_with_needs(code, ln_back_res_on_cpu, LN_BACK_RES_NEEDS)
            }
            Prog::Colsum => cp.register_program(code, colsum_on_cpu),
            Prog::GeluTr => cp.register_program(code, gelu_tr_on_cpu),
            Prog::AttnFwd => cp.register_program(code, attn_fwd_on_cpu),
            Prog::AttnBack => cp.register_program(code, attn_back_on_cpu),
            Prog::EmbFwd => cp.register_program(code, emb_fwd_on_cpu),
            Prog::EmbBack => cp.register_program(code, emb_back_on_cpu),
            Prog::Gnorm => cp.register_program_with_needs(code, gnorm_on_cpu, GNORM_NEEDS),
            Prog::OptStep => cp.register_program(code, opt_step_on_cpu),
            Prog::Vadd => cp.register_program(code, vadd_on_cpu),
            other => panic!("program neasteptat: {other:?}"),
        };
        cp.mem.write(prog_va(p), code);
    }
    cp.register_program(COPY_B128, copy_b128_on_cpu);
    cp.mem.write(COPY, COPY_B128);
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

/// 3 steps through RAM on the model; returns the profile and the board time of every step (from the records).
fn run_sim(fused: bool) -> (StepProfile, Vec<u64>) {
    let mut cp = board();
    let p0 = params();
    let opt = AdamW::new(&p0, 1e-3, 0.9, 0.999, 1e-8, 0.1);
    let plan0 = plan(fused);
    let (x0, y0) = batch(1);
    cp.mem.write(HEAP, &vec![0u8; plan0.bytes() as usize]);
    for (va, bytes) in plan0.uploads(&p0, &x0, &y0).unwrap() {
        cp.mem.write(va, &bytes);
    }
    let mut prof: Option<StepProfile> = None;
    let mut totals = Vec::new();
    for k in 0..3usize {
        let (x, y) = batch(1 + k as u64);
        let plan = plan0.rebatch(&x).unwrap();
        let l = plan.step(Recipe::AdamW513, &opt, k as u32 + 1).unwrap();
        let v = br::via_ram(&plan, &x, &y, STAGE).unwrap();
        cp.mem.write(STAGE, &v.staging);
        let (zone, qmds) = br::pack_via_ram(&l, prog_va, COPY, ZONE, &v).unwrap();
        cp.mem.write(ZONE, &zone);
        let n = qmds.len();
        let p = prof.get_or_insert_with(|| {
            StepProfile::from_launches(&l)
                .unwrap()
                .with_copies(v.copies_in.len(), v.copies_out.len())
                .unwrap()
        });
        assert_eq!(p.launches(), n);
        assert_eq!(n, l.len() + br::extra_launches(&v));
        cp.mem.write(REC, &vec![0u8; 16 * (n + 1)]);
        let first = FIRST + STRIDE * k as u32;
        let mut w = vec![0u32; transfer::sequence_words_fresh(n)];
        let mut pb = PushBuf::new(&mut w);
        transfer::encode_sequence_fresh(&mut pb, &qmds, REC, first).unwrap();
        let nw = pb.len();
        let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
        cp.mem.write(PUSH, &bytes);
        let e = push::gpfifo_entry(PUSH, nw as u32, false).unwrap();
        cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
        let mut rec = vec![0u8; 16 * (n + 1)];
        cp.mem.read(REC, &mut rec).unwrap();
        let t = |i: usize| u64::from_le_bytes(rec[16 * i + 8..16 * i + 16].try_into().unwrap());
        totals.push(t(n) - t(0));
        p.add_records(&rec, first).unwrap();
    }
    (prof.unwrap(), totals)
}

#[test]
fn t210_q1_simulare_records_to_profile() {
    for fused in [false, true] {
        let (p, totals) = run_sim(fused);
        let n = if fused { 69 + 7 } else { 75 + 7 };
        assert_eq!(p.launches(), n);
        assert_eq!(p.steps(), 3);
        let mut t = totals.clone();
        t.sort_unstable();
        assert_eq!(p.total_ns(), t[1], "SIMULARE: total = first record → last");
        for groups in [p.by_tag(), p.by_prog(), p.by_stage()] {
            assert_eq!(groups.iter().map(|g| g.launches).sum::<u32>() as usize, n);
        }
        let st = p.by_stage();
        let lot = st.iter().find(|g| g.name == "lot").unwrap();
        assert_eq!(lot.launches, 7);
        let tags = p.by_tag();
        assert_eq!(tags.iter().find(|g| g.name == COPY_UP).unwrap().launches, 5);
        assert_eq!(
            tags.iter().find(|g| g.name == COPY_DOWN).unwrap().launches,
            2
        );
        // SIMULARE clock: every launch ≥ the model's fixed 3 000 ns.
        assert!(p.total_ns() >= 3_000 * n as u64);
        println!("SIMULARE (fused = {fused}):\n{}", p.lines(6).join("\n"));
    }
}
