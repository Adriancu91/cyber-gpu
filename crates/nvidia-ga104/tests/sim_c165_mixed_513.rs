//! C-165 / the centre's priority 2 — T223, SIMULARE: the **mixed** training step (every matmul on `mm_tc`, f16
//! inputs, f32 sums, loss scale S) on **#513's d(logits)**: Kronos's vocabulary (16 384) and 1 024 symbols per batch.
//! (a) #513's real list (12 layers × 768, batch 2 × 512) is built and packed, not run (too large for the simulator);
//! (b) the `antrenare mini` model with a 32 × 32 batch (the same V · N as #513) runs 2 steps against the CPU oracle at
//! S = 1 024 (today's `LOSS_SCALE`) and S = 2¹⁴ (`LOSS_SCALE_513`). Plan and predictions written before the code
//! (`docs/C-165_PAS_MIXT_513.md`, commit `d221fd3`).
//!
//! Nothing here runs on the RTX 3070 Ti; the tensor-core adder is the model fitted on os-335 (P8).

use std::sync::OnceLock;

use gpu_compute::mm_tc::round_f16;
use gpu_compute::train::{self, AdamW, Arch, Batch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS, LN_FWD_NEEDS,
    MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, Needs, XENT_NEEDS,
    attn_back_on_cpu, attn_back_tile_on_cpu, attn_back_tile0_on_cpu, attn_fwd_on_cpu,
    attn_fwd_tile_on_cpu, attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, emb_back_on_cpu,
    emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu,
    ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu,
    opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, LOSS_SCALE, LOSS_SCALE_513, MixedLaunch, PROGRAMS_ALL_FUSED2, Plan, Prog,
};
use nvidia_ga104::transfer;
use nvidia_rm570::regs::clc7c0 as c;

/// `antrenare mini` (cyber-universe `kernel/src/shell/antrenare_mini.rs`), with the default vocabulary.
const MINI: Arch = Arch {
    vocab: 16_384,
    ctx: 32,
    n_emb: 64,
    heads: 4,
    layers: 2,
};
/// 32 × 32 = 1 024 symbols per batch, as #513's 2 × 512.
const B: usize = 32;
const T: usize = 32;
const STEPS: u32 = 2;
const LR: f64 = 1e-3;
const INIT_HALF_WIDTH: f64 = 0.034_641_016;

const VA: u64 = nvidia_ga104::mmu::VA_BASE;
const ZONE: u64 = VA + 0x20_0000;
const REC: u64 = VA + 0x30_0000;
const PUSH: u64 = VA + 0x32_0000;
const PROGS: u64 = VA + 0x40_0000;
const TC: u64 = PROGS + 0x1_0000 * 23;
const TCG: u64 = PROGS + 0x1_0000 * 24;
const HEAP: u64 = VA + 0x100_0000;

fn prog_va(p: Prog) -> u64 {
    PROGS + 0x1_0000 * p.index() as u64
}

fn needs(s: nvidia_ga104::qmd::SharedMemory) -> Needs {
    Needs {
        shared_bytes: s.bytes,
        barriers: u32::from(s.barriers),
    }
}

/// xorshift64* — our own sequence.
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
    let mut g = Rng(0x00c1_65a0_2026_1125);
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

fn batch() -> (Vec<u32>, Vec<u32>) {
    let mut g = Rng(0x5eed_0000_0000_0223);
    let (mut inp, mut tgt) = (Vec::new(), Vec::new());
    for _ in 0..B {
        let row: Vec<u32> = (0..=T)
            .map(|_| (g.next() % MINI.vocab as u64) as u32)
            .collect();
        inp.extend_from_slice(&row[..T]);
        tgt.extend_from_slice(&row[1..]);
    }
    (inp, tgt)
}

/// The rows of `emb_token` whose symbol is neither an input nor a target.
fn rare_rows(inputs: &[u32], targets: &[u32]) -> Vec<usize> {
    let mut used = vec![false; MINI.vocab];
    for &s in inputs.iter().chain(targets) {
        used[s as usize] = true;
    }
    (0..MINI.vocab).filter(|&v| !used[v]).collect()
}

struct Run {
    losses: Vec<f64>,
    grad1: Vec<Vec<f32>>,
    after_last: Vec<Vec<f32>>,
    /// Step 1's d(logits) as the board left them (S times the true ones): the largest |x| and the share of nonzero
    /// values below f16's smallest normal number (2⁻¹⁴) — what `mm_tc` reads in f16. (0, 0) for the oracle.
    dlogits: (f32, f64),
}

fn oracle() -> &'static Run {
    static O: OnceLock<Run> = OnceLock::new();
    O.get_or_init(|| {
        let (inputs, targets) = batch();
        let b = Batch {
            inputs: &inputs,
            targets: &targets,
            batch: B,
            len: T,
        };
        let mut p = initial();
        let mut adam = AdamW::new(&p, LR, 0.9, 0.999, 1e-8, 0.1);
        let mut run = Run {
            losses: Vec::new(),
            grad1: Vec::new(),
            after_last: Vec::new(),
            dlogits: (0.0, 0.0),
        };
        for k in 0..STEPS {
            let (loss, mut g) = train::loss_and_grad(&MINI, &p, &b).unwrap();
            run.losses.push(loss as f64);
            if k == 0 {
                run.grad1 = g.clone();
            }
            train::clip_grad_norm(&mut g, 1.0);
            adam.step(&mut p, &g).unwrap();
        }
        run.after_last = p;
        run
    })
}

fn board() -> CommandProcessor {
    ts::check_programs().expect("SHA-256 programe");
    let mut cp = CommandProcessor::new();
    for p in PROGRAMS_ALL_FUSED2 {
        let code = p.code();
        let sh = || needs(p.shared().unwrap());
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
            Prog::AttnP => cp.register_program(code, attn_p_on_cpu),
            Prog::AttnFwdTile2 | Prog::AttnFwdTile3 => {
                cp.register_program_with_needs(code, attn_fwd_tile_on_cpu, sh())
            }
            Prog::AttnBackTile0_2 | Prog::AttnBackTile0_3 => {
                cp.register_program_with_needs(code, attn_back_tile0_on_cpu, sh())
            }
            Prog::AttnBackTile2 | Prog::AttnBackTile3 => {
                cp.register_program_with_needs(code, attn_back_tile_on_cpu, sh())
            }
            Prog::AttnPTile => cp.register_program_with_needs(code, attn_p_tile_on_cpu, sh()),
            Prog::LnBackRes => {
                cp.register_program_with_needs(code, ln_back_res_on_cpu, LN_BACK_RES_NEEDS)
            }
            Prog::MmTrainGelu => {
                cp.register_program_with_needs(code, mm_train_gelu_on_cpu, MM_TRAIN_GELU_NEEDS)
            }
            Prog::MmTrainTile
            | Prog::ColsumPart
            | Prog::MmTv128x64
            | Prog::MmTv64x128
            | Prog::MmTv128x128
            | Prog::MmTv64x64K8
            | Prog::MmTrainTileGelu => unreachable!("nu e in PROGRAMS_ALL_FUSED2"),
        };
        cp.mem.write(prog_va(p), code);
    }
    cp.register_program_with_needs(MM_TC, mm_tc_on_cpu, MM_TC_NEEDS);
    cp.mem.write(TC, MM_TC);
    cp.register_program_with_needs(MM_TC_GELU, mm_tc_gelu_on_cpu, MM_TC_GELU_NEEDS);
    cp.mem.write(TCG, MM_TC_GELU);
    cp
}

fn submit(
    cp: &mut CommandProcessor,
    first: &mut u32,
    launches: &[MixedLaunch],
) -> Result<(), String> {
    let (zone, qmds) =
        ts::pack_mixed_gelu(launches, prog_va, TC, TCG, ZONE).map_err(|e| format!("{e:?}"))?;
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
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
        .map_err(|err| format!("{err:?}"))?;
    *first = first.wrapping_add(n as u32 + 1);
    Ok(())
}

/// `STEPS` mixed steps at scale `s` on the simulated board, `_3` attention (as #513 on the board today).
fn on_board(s: f32) -> Run {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = Plan::new_tiles3(MINI, B, T, &inputs, HEAP).unwrap();
    let mut cp = board();
    for (va, bytes) in plan.uploads(&params, &inputs, &targets).unwrap() {
        cp.mem.write(va, &bytes);
    }
    for (va, bytes) in plan.mixed_uploads(s).unwrap() {
        cp.mem.write(va, &bytes);
    }
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let mut first = 0x7000u32;
    let mut run = Run {
        losses: Vec::new(),
        grad1: Vec::new(),
        after_last: Vec::new(),
        dlogits: (0.0, 0.0),
    };
    for step in 1..=STEPS {
        let list = plan.step_mixed(Recipe::AdamW513, &opt, step, s).unwrap();
        submit(&mut cp, &mut first, &list).unwrap_or_else(|e| panic!("pasul {step}: {e}"));
        let rows = cp.mem.read_f32s(plan.loss_rows, B * T).unwrap();
        run.losses.push(ts::Outcome::mean_loss(&rows) as f64);
        let r = cp.mem.read_f32s(plan.res, 1).unwrap();
        assert!(r[0].is_finite(), "pasul {step}: norma nefinita");
        if step == 1 {
            let inv = 1.0 / s;
            let g = cp.mem.read_f32s(plan.g, plan.n_params).unwrap();
            assert!(g.iter().all(|x| x.is_finite()), "gradient nefinit");
            let g: Vec<f32> = g.iter().map(|x| x * inv).collect();
            run.grad1 = plan.unflat(&g).unwrap();
            let d = cp.mem.read_f32s(plan.dlogits, B * T * MINI.vocab).unwrap();
            let max = d.iter().fold(0.0f32, |m, x| m.max(x.abs()));
            let nz = d.iter().filter(|x| **x != 0.0).count().max(1);
            let sub = d
                .iter()
                .filter(|x| **x != 0.0 && x.abs() < 6.103_515_6e-5)
                .count();
            run.dlogits = (max, sub as f64 / nz as f64);
        }
    }
    let w = cp.mem.read_f32s(plan.w, plan.n_params).unwrap();
    assert!(w.iter().all(|x| x.is_finite()), "greutati nefinite");
    run.after_last = plan.unflat(&w).unwrap();
    run
}

fn mixed(s: f32) -> &'static Run {
    static LOW: OnceLock<Run> = OnceLock::new();
    static HIGH: OnceLock<Run> = OnceLock::new();
    if s == LOSS_SCALE {
        LOW.get_or_init(|| on_board(s))
    } else if s == LOSS_SCALE_513 {
        HIGH.get_or_init(|| on_board(s))
    } else {
        unreachable!()
    }
}

/// ‖a − b‖ / ‖b‖ (f64 sums).
fn rel(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0.0f64, 0.0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64 - *y as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(1e-300)).sqrt()
}

fn rows_of(t: &[f32], rows: &[usize]) -> Vec<f32> {
    let c = MINI.n_emb;
    rows.iter()
        .flat_map(|&r| t[r * c..(r + 1) * c].iter().copied())
        .collect()
}

struct Figures {
    loss: f64,
    grad: (f64, String),
    emb_grad: f64,
    rare_grad: f64,
}

fn figures(run: &Run) -> Figures {
    let o = oracle();
    let (inputs, targets) = batch();
    let rare = rare_rows(&inputs, &targets);
    let names: Vec<String> = MINI.tensors().iter().map(|t| t.name.clone()).collect();
    let loss = run
        .losses
        .iter()
        .zip(&o.losses)
        .map(|(a, b)| ((a - b) / b).abs())
        .fold(0.0, f64::max);
    let mut grad = (0.0, String::new());
    for (i, n) in names.iter().enumerate() {
        if n.ends_with("in_proj_bias") {
            continue;
        }
        let g = rel(&run.grad1[i], &o.grad1[i]);
        if g > grad.0 {
            grad = (g, n.clone());
        }
    }
    Figures {
        loss,
        grad,
        emb_grad: rel(&run.grad1[0], &o.grad1[0]),
        rare_grad: rel(&rows_of(&run.grad1[0], &rare), &rows_of(&o.grad1[0], &rare)),
    }
}

fn show(label: &str, r: &Run, f: &Figures) {
    println!(
        "kt c223 {label} SIMULARE: eroarea {:.2e}; gradientul pas 1 cel mai rau {:.2e} ({}); emb_token {:.2e}, randurile rare {:.2e}; d(logits)*S max {:.3}, sub 2^-14 {:.1} %",
        f.loss,
        f.grad.0,
        f.grad.1,
        f.emb_grad,
        f.rare_grad,
        r.dlogits.0,
        100.0 * r.dlogits.1
    );
}

// ------------------------------------------------------------------ the tests

/// Prediction 5: the rule gives today's scale for `antrenare mini` and 2¹⁴ for #513.
#[test]
fn t223_loss_scale_rule() {
    assert_eq!(ts::loss_scale_for(16_384, 64), LOSS_SCALE);
    assert_eq!(ts::loss_scale_for(16_384, 1_024), LOSS_SCALE_513);
    assert_eq!(ts::loss_scale_for(16_384, 2 * 512), 16_384.0);
    assert_eq!(ts::loss_scale_for(256, 64), 16.0);
    assert_eq!(ts::loss_scale_for(1, 1), 1.0);
    assert_eq!(ts::loss_scale_for(0, 0), 1.0);
    assert_eq!(ts::loss_scale_for(1 << 20, 1 << 20), 16_777_216.0);
    // Every value the rule gives is accepted by the mixed step.
    let plan = Plan::new(MINI, 2, 32, &[1u32; 64], HEAP).unwrap();
    for (v, n) in [(16_384, 64), (16_384, 1_024), (1 << 20, 1 << 20), (1, 1)] {
        assert!(plan.mixed_uploads(ts::loss_scale_for(v, n)).is_ok());
    }
}

/// Prediction 1: #513's real list (not run) — the two plans the OS runs today, mixed at S = 2¹⁴.
#[test]
fn t223_list_513_mixed() {
    let m = gpu_compute::train::M513;
    let inputs: Vec<u32> = (0..2 * 512u32).map(|i| (i * 7919) % 16_384).collect();
    let base = VA + (1 << 30);
    let plain = Plan::new_tiles3(m, 2, 512, &inputs, base)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split();
    let glued = Plan::new_tiles3(m, 2, 512, &inputs, base)
        .unwrap()
        .with_fused_back()
        .with_fused_gelu()
        .with_mm_tile_gelu()
        .with_colsum_split();
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    for (label, plan, gelu) in [
        ("placi3+mm-placi+colsum", &plain, 0usize),
        ("lipit-placi", &glued, 12),
    ] {
        let f32_list = plan.step(Recipe::AdamW513, &opt, 1).unwrap();
        let list = plan
            .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
            .unwrap();
        let tc = list.iter().filter(|l| l.name() == "mm_tc").count();
        let tcg = list.iter().filter(|l| l.name() == "mm_tc_gelu").count();
        let left_f32_mm = list
            .iter()
            .filter(|l| matches!(l, MixedLaunch::Std(x) if x.prog == Prog::MmTrain && x.tag != ts::TAG_UNSCALE))
            .count();
        let (zone, qmds) = ts::pack_mixed_gelu(&list, prog_va, TC, TCG, ZONE).unwrap();
        println!(
            "kt c223 lista #513 {label} SIMULARE: {} lansari F32, {} mixt ({tc} mm_tc + {tcg} mm_tc_gelu), zona {:.0} KiB, {} QMD",
            f32_list.len(),
            list.len(),
            zone.len() as f64 / 1024.0,
            qmds.len()
        );
        assert_eq!(list.len(), f32_list.len() + 1);
        assert_eq!(list.len(), plan.launches_per_step_mixed(Recipe::AdamW513));
        assert_eq!(tc + tcg, 147);
        assert_eq!(tcg, gelu);
        assert_eq!(left_f32_mm, 0, "o inmultire a modelului a ramas F32");
        assert_eq!(qmds.len(), list.len());
        assert!((300 * 1024..=600 * 1024).contains(&zone.len()));
    }
    // The scale is what the plan's batch needs.
    assert_eq!(
        ts::loss_scale_for(m.vocab, plain.batch * plain.len),
        LOSS_SCALE_513
    );
}

/// Prediction 2 and 4: S = 2¹⁴ on 1 024 symbols per batch, within the written tolerances, no overflow.
#[test]
fn t223_mixed_2_14_on_1024_symbols_within_tolerance() {
    let r = mixed(LOSS_SCALE_513);
    let f = figures(r);
    show("mixt S=2^14 lot 32x32", r, &f);
    assert!(f.loss <= ts::MIX_TOL_LOSS_REL, "eroarea {:e}", f.loss);
    assert!(
        f.grad.0 <= ts::MIX_TOL_GRAD_REL_NORM,
        "gradientul {:e} ({})",
        f.grad.0,
        f.grad.1
    );
    assert!(f.rare_grad <= 1e-3, "randurile rare {:e}", f.rare_grad);
    // Prediction 4: |S · d(logits)| ≤ S / N = 16 (plus rounding), far below f16's 65 504.
    assert!(r.dlogits.0 <= 16.1, "d(logits)*S max {}", r.dlogits.0);
    assert!(round_f16(r.dlogits.0).is_finite());
}

/// Prediction 3 (FAILED, `docs/C-165_PAS_MIXT_513.md` §"Rezultate"): today's S = 1 024 on the same batch puts ~half of
/// d(logits) below f16's smallest normal number, yet the rare rows' gradient is as good as at S = 2¹⁴ (the f16 rounding
/// of the hidden states dominates; the subnormal d(logits) errors average out over 1 024 symbols). Asserted here: what
/// was measured, with margin.
#[test]
fn t223_mixed_1024_on_1024_symbols_subnormal_but_same_error() {
    let low = mixed(LOSS_SCALE);
    let high = mixed(LOSS_SCALE_513);
    let fl = figures(low);
    let fh = figures(high);
    show("mixt S=1024 lot 32x32", low, &fl);
    let ratio = fl.rare_grad / fh.rare_grad;
    println!(
        "kt c223 scara: randurile rare S=1024 / S=2^14 = {ratio:.3}; d(logits) sub 2^-14: {:.1} % fata de {:.1} %",
        100.0 * low.dlogits.1,
        100.0 * high.dlogits.1
    );
    assert!(fl.loss <= ts::MIX_TOL_LOSS_REL, "eroarea {:e}", fl.loss);
    assert!(
        fl.grad.0 <= ts::MIX_TOL_GRAD_REL_NORM,
        "gradientul {:e}",
        fl.grad.0
    );
    assert!(
        fl.rare_grad <= 1e-3,
        "randurile rare S=1024 {:e}",
        fl.rare_grad
    );
    assert!((0.8..=1.25).contains(&ratio), "raportul {ratio}");
    // The share of subnormal inputs: ~half at S = 1 024, none at S = 2¹⁴.
    assert!(low.dlogits.1 >= 0.4, "S=1024 sub 2^-14 {}", low.dlogits.1);
    assert!(
        high.dlogits.1 <= 1e-3,
        "S=2^14 sub 2^-14 {}",
        high.dlogits.1
    );
    assert!((low.dlogits.0 - 1.0).abs() <= 0.01);
}
