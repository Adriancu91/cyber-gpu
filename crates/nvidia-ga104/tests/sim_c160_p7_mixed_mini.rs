//! Order of 23 Nov 20:11, step 2 — T199, SIMULARE: the **mixed** training step (every matmul on `mm_tc`, f16 inputs,
//! f32 sums, loss scale S) on the shape of the OS's `antrenare mini`: Kronos's vocabulary (16 384), context 32, E 64,
//! 4 heads, 2 layers, batch 2 × 32, AdamW lr 1e-3 / 0.9 / 0.999 / 1e-8 / decay 0.1, clipping at 1.0. Against the CPU
//! oracle (`gpu_compute::train`) from the same start; predictions written before the first run
//! (`docs/C-160_P7_PAS_MIXT_MINI.md`, commit `265557b`).
//!
//! Nothing here runs on the RTX 3070 Ti; the tensor-core adder is the model fitted on os-335 (P8).

use std::sync::OnceLock;

use gpu_compute::train::{self, AdamW, Arch, Batch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS, LN_FWD_NEEDS, MM_TC_NEEDS,
    MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, Needs, XENT_NEEDS, attn_back_on_cpu,
    attn_back_tile_on_cpu, attn_back_tile0_on_cpu, attn_fwd_on_cpu, attn_fwd_tile_on_cpu,
    attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu,
    gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu, ln_fwd_on_cpu, mm_tc_on_cpu,
    mm_train_gelu_on_cpu, mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, LOSS_SCALE, MixedLaunch, PROGRAMS_ALL_FUSED2, Plan, Prog,
};
use nvidia_ga104::transfer;
use nvidia_rm570::regs::clc7c0 as c;

/// `antrenare mini` (cyber-universe `kernel/src/shell/antrenare_mini.rs`, `arch()`), with the default vocabulary.
const MINI: Arch = Arch {
    vocab: 16_384,
    ctx: 32,
    n_emb: 64,
    heads: 4,
    layers: 2,
};
const B: usize = 2;
const T: usize = 32;
const STEPS: u32 = 5;
/// `antrenare mini`'s defaults: `AdamW::new(&params, lr, 0.9, 0.999, 1e-8, 0.1)`, lr 1e-3.
const LR: f64 = 1e-3;
/// The OS's start: weights uniform in ±0.0346 (standard deviation 0.02), biases 0, LayerNorm weights 1.
const INIT_HALF_WIDTH: f64 = 0.034_641_016;

const VA: u64 = nvidia_ga104::mmu::VA_BASE;
const ZONE: u64 = VA + 0x20_0000;
const REC: u64 = VA + 0x30_0000;
const PUSH: u64 = VA + 0x32_0000;
const PROGS: u64 = VA + 0x40_0000;
/// `mm_tc` after the 23 programs of `PROGRAMS_ALL_FUSED2` (T200: `ln_back_res` took slot 21, T201: `mm_train_gelu`
/// slot 22).
const TC: u64 = PROGS + 0x1_0000 * 23;
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

// ------------------------------------------------------------------ the start: weights, batch

/// xorshift64* — our own sequence (the OS uses MT19937; the law of the start values is the same).
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
    let mut g = Rng(0x00c1_9e10_2026_1123);
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

/// Two rows of 33 pseudo-random symbols: inputs = the first 32, targets = the next 32 (Kronos's data is not in the repo).
fn batch() -> (Vec<u32>, Vec<u32>) {
    let mut g = Rng(0x5eed_0000_0000_0199);
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

/// The rows of `emb_token` whose symbol is neither an input nor a target: their gradient comes only from the tied
/// head's small d(logits). Refuses a list that names a batch symbol.
fn rare_rows(
    inputs: &[u32],
    targets: &[u32],
    pick: Option<&[usize]>,
) -> Result<Vec<usize>, String> {
    let mut used = vec![false; MINI.vocab];
    for &s in inputs.iter().chain(targets) {
        used[s as usize] = true;
    }
    match pick {
        None => Ok((0..MINI.vocab).filter(|&v| !used[v]).collect()),
        Some(p) => match p.iter().find(|&&v| used[v]) {
            Some(v) => Err(format!("simbolul {v} e in lot, nu e rar")),
            None => Ok(p.to_vec()),
        },
    }
}

// ------------------------------------------------------------------ the oracle and the board

struct Run {
    losses: Vec<f64>,
    /// Step-1 gradient, divided by S on the board.
    grad1: Vec<Vec<f32>>,
    after_last: Vec<Vec<f32>>,
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layout {
    Simple,
    Tiles3,
    /// T200: [`Plan::new_tiles3`] + [`Plan::with_fused_back`].
    Tiles3Fused,
    /// T201: [`Plan::new_tiles3`] + both fusions (F32 only: the mixed step refuses `with_fused_gelu`).
    Tiles3Both,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    F32,
    Mixed(f32),
    /// Mixed, but the norm compared with 1.0 (the F32 list's `gnorm` launches): clipping ~S times too strong.
    MixedNormNotScaled(f32),
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
            // T200: `ln_back_res` (`Plan::with_fused_back`).
            Prog::LnBackRes => {
                cp.register_program_with_needs(code, ln_back_res_on_cpu, LN_BACK_RES_NEEDS)
            }
            // T201: `mm_train_gelu` (`Plan::with_fused_gelu`).
            Prog::MmTrainGelu => {
                cp.register_program_with_needs(code, mm_train_gelu_on_cpu, MM_TRAIN_GELU_NEEDS)
            }
        };
        cp.mem.write(prog_va(p), code);
    }
    cp.register_program_with_needs(MM_TC, mm_tc_on_cpu, MM_TC_NEEDS);
    cp.mem.write(TC, MM_TC);
    cp
}

/// One step = one submission (the same path as T169).
fn submit(
    cp: &mut CommandProcessor,
    first: &mut u32,
    launches: &[MixedLaunch],
) -> Result<(), String> {
    let (zone, qmds) = ts::pack_mixed(launches, prog_va, TC, ZONE).map_err(|e| format!("{e:?}"))?;
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

/// `STEPS` steps on the simulated board; returns the run and how many launches of step 1 were on `mm_tc`.
fn on_board(layout: Layout, mode: Mode) -> (Run, usize) {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = match layout {
        Layout::Simple => Plan::new(MINI, B, T, &inputs, HEAP),
        Layout::Tiles3 => Plan::new_tiles3(MINI, B, T, &inputs, HEAP),
        Layout::Tiles3Fused => {
            Plan::new_tiles3(MINI, B, T, &inputs, HEAP).map(Plan::with_fused_back)
        }
        Layout::Tiles3Both => Plan::new_tiles3(MINI, B, T, &inputs, HEAP)
            .map(|p| p.with_fused_back().with_fused_gelu()),
    }
    .unwrap();
    let mut cp = board();
    for (va, bytes) in plan.uploads(&params, &inputs, &targets).unwrap() {
        cp.mem.write(va, &bytes);
    }
    let s = match mode {
        Mode::F32 => 1.0,
        Mode::Mixed(s) | Mode::MixedNormNotScaled(s) => s,
    };
    if mode != Mode::F32 {
        for (va, bytes) in plan.mixed_uploads(s).unwrap() {
            cp.mem.write(va, &bytes);
        }
    }
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let mut first = 0x7000u32;
    let mut run = Run {
        losses: Vec::new(),
        grad1: Vec::new(),
        after_last: Vec::new(),
    };
    let mut tc = 0;
    for step in 1..=STEPS {
        let list: Vec<MixedLaunch> = match mode {
            Mode::F32 => plan
                .step(Recipe::AdamW513, &opt, step)
                .unwrap()
                .into_iter()
                .map(MixedLaunch::Std)
                .collect(),
            Mode::Mixed(_) => plan.step_mixed(Recipe::AdamW513, &opt, step, s).unwrap(),
            Mode::MixedNormNotScaled(_) => {
                let plain = plan.step(Recipe::AdamW513, &opt, step).unwrap();
                let mut l = plan.step_mixed(Recipe::AdamW513, &opt, step, s).unwrap();
                for x in l.iter_mut() {
                    if x.tag().starts_with("opt:norma") {
                        let p = plain.iter().find(|p| p.tag == x.tag()).unwrap();
                        *x = MixedLaunch::Std(p.clone());
                    }
                }
                l
            }
        };
        if step == 1 {
            tc = list.iter().filter(|l| l.name() == "mm_tc").count();
        }
        submit(&mut cp, &mut first, &list).unwrap_or_else(|e| panic!("pasul {step}: {e}"));
        let rows = cp.mem.read_f32s(plan.loss_rows, B * T).unwrap();
        run.losses.push(ts::Outcome::mean_loss(&rows) as f64);
        let r = cp.mem.read_f32s(plan.res, 1).unwrap();
        assert!(r[0].is_finite(), "pasul {step}: norma nefinita");
        if step == 1 {
            let inv = 1.0 / s;
            let g = cp.mem.read_f32s(plan.g, plan.n_params).unwrap();
            let g: Vec<f32> = g.iter().map(|x| x * inv).collect();
            run.grad1 = plan.unflat(&g).unwrap();
        }
    }
    run.after_last = plan
        .unflat(&cp.mem.read_f32s(plan.w, plan.n_params).unwrap())
        .unwrap();
    (run, tc)
}

// ------------------------------------------------------------------ the measures

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

/// The movement of the weights: (w − w₀) − (w_ref − w₀), relative to (w_ref − w₀).
fn moved(w: &[f32], w_ref: &[f32], w0: &[f32]) -> f64 {
    let a: Vec<f32> = w.iter().zip(w0).map(|(x, z)| x - z).collect();
    let b: Vec<f32> = w_ref.iter().zip(w0).map(|(x, z)| x - z).collect();
    rel(&a, &b)
}

struct Figures {
    loss: f64,
    /// Worst tensor outside `in_proj_bias`, and its name.
    grad: (f64, String),
    emb_grad: f64,
    rare_grad: f64,
    move_worst: (f64, String),
    rare_move: f64,
}

fn figures(run: &Run) -> Figures {
    let o = oracle();
    let (inputs, targets) = batch();
    let rare = rare_rows(&inputs, &targets, None).unwrap();
    let w0 = initial();
    let names: Vec<String> = MINI.tensors().iter().map(|t| t.name.clone()).collect();
    let loss = run
        .losses
        .iter()
        .zip(&o.losses)
        .map(|(a, b)| ((a - b) / b).abs())
        .fold(0.0, f64::max);
    let mut grad = (0.0, String::new());
    let mut move_worst = (0.0, String::new());
    for (i, n) in names.iter().enumerate() {
        if n.ends_with("in_proj_bias") {
            continue;
        }
        let g = rel(&run.grad1[i], &o.grad1[i]);
        if g > grad.0 {
            grad = (g, n.clone());
        }
        let m = moved(&run.after_last[i], &o.after_last[i], &w0[i]);
        if m > move_worst.0 {
            move_worst = (m, n.clone());
        }
    }
    Figures {
        loss,
        grad,
        emb_grad: rel(&run.grad1[0], &o.grad1[0]),
        rare_grad: rel(&rows_of(&run.grad1[0], &rare), &rows_of(&o.grad1[0], &rare)),
        move_worst,
        rare_move: moved(
            &rows_of(&run.after_last[0], &rare),
            &rows_of(&o.after_last[0], &rare),
            &rows_of(&w0[0], &rare),
        ),
    }
}

fn show(label: &str, f: &Figures) {
    println!(
        "kt c199 {label} SIMULARE: eroarea {:.2e}; gradientul pas 1 cel mai rau {:.2e} ({}); emb_token {:.2e}, randurile rare {:.2e}; miscarea greutatilor cel mai rau {:.2e} ({}), randurile rare {:.2e}",
        f.loss,
        f.grad.0,
        f.grad.1,
        f.emb_grad,
        f.rare_grad,
        f.move_worst.0,
        f.move_worst.1,
        f.rare_move
    );
}

fn simple(mode: Mode) -> &'static (Run, usize) {
    static F: OnceLock<(Run, usize)> = OnceLock::new();
    static M: OnceLock<(Run, usize)> = OnceLock::new();
    static ONE: OnceLock<(Run, usize)> = OnceLock::new();
    match mode {
        Mode::F32 => F.get_or_init(|| on_board(Layout::Simple, mode)),
        Mode::Mixed(s) if s == LOSS_SCALE => M.get_or_init(|| on_board(Layout::Simple, mode)),
        Mode::Mixed(1.0) => ONE.get_or_init(|| on_board(Layout::Simple, mode)),
        _ => unreachable!(),
    }
}

// ------------------------------------------------------------------ the tests

#[test]
fn t199_start_is_near_uniform_and_the_rare_rows_are_most_of_the_vocabulary() {
    let o = oracle();
    let (inputs, targets) = batch();
    let rare = rare_rows(&inputs, &targets, None).unwrap();
    println!(
        "kt c199 start: eroarea {:.4} (ln 16384 = {:.4}), randuri rare {} din {}",
        o.losses[0],
        (MINI.vocab as f64).ln(),
        rare.len(),
        MINI.vocab
    );
    assert!((o.losses[0] - (MINI.vocab as f64).ln()).abs() < 0.1);
    assert!(rare.len() >= MINI.vocab - 2 * B * T);
    // Planted (b): a batch symbol named as rare is refused.
    assert!(rare_rows(&inputs, &targets, Some(&[rare[0], targets[3] as usize])).is_err());
    assert!(rare_rows(&inputs, &targets, Some(&[rare[0], rare[1]])).is_ok());
    // The oracle's loss goes down over the 5 steps (AdamW on one batch).
    assert!(o.losses[STEPS as usize - 1] < o.losses[0]);
}

#[test]
fn t199_f32_witness_matches_the_oracle() {
    let (run, tc) = simple(Mode::F32);
    let f = figures(run);
    show("F32 (martor)", &f);
    assert_eq!(*tc, 0);
    assert!(f.loss <= 1e-5, "eroarea {:e}", f.loss);
    assert!(f.grad.0 <= 1e-3, "gradientul {:e} ({})", f.grad.0, f.grad.1);
}

#[test]
fn t199_mixed_1024_within_the_written_tolerance() {
    let (run, tc) = simple(Mode::Mixed(LOSS_SCALE));
    let f = figures(run);
    show("mixt S=1024", &f);
    println!("kt c199 mixt: {tc} lansari pe mm_tc la pas");
    assert!(*tc > 0);
    assert!(f.loss <= 1e-3, "eroarea {:e}", f.loss);
    assert!(f.grad.0 <= 5e-3, "gradientul {:e} ({})", f.grad.0, f.grad.1);
    assert!(f.rare_grad <= 1e-3, "randurile rare {:e}", f.rare_grad);
    assert!(
        f.move_worst.0 <= 5e-2,
        "miscarea {:e} ({})",
        f.move_worst.0,
        f.move_worst.1
    );
}

#[test]
fn t199_loss_scaling_matters_on_kronos_vocabulary() {
    let one = figures(&simple(Mode::Mixed(1.0)).0);
    let sc = figures(&simple(Mode::Mixed(LOSS_SCALE)).0);
    show("mixt S=1", &one);
    println!(
        "kt c199 scalarea: randurile rare gradient S=1 / S=1024 = {:.1}; emb_token intreg {:.2}; miscarea randurilor rare {:.2}",
        one.rare_grad / sc.rare_grad,
        one.emb_grad / sc.emb_grad,
        one.rare_move / sc.rare_move
    );
    // Predictions 3 and 4 (§ predictions).
    assert!(
        (5e-3..=5e-2).contains(&one.rare_grad),
        "S=1 rare {:e}",
        one.rare_grad
    );
    assert!(one.rare_grad >= 10.0 * sc.rare_grad);
    assert!(one.emb_grad / sc.emb_grad < 3.0);
    assert!(one.rare_move / sc.rare_move <= 10.0);
}

#[test]
fn t199_tiles3_mixed_same_figures_as_simple() {
    let (run, tc) = on_board(Layout::Tiles3, Mode::Mixed(LOSS_SCALE));
    let t = figures(&run);
    let s = figures(&simple(Mode::Mixed(LOSS_SCALE)).0);
    show("mixt S=1024 atentia _3", &t);
    assert_eq!(*tc_ref(), tc);
    for (a, b, what) in [
        (t.loss, s.loss, "eroarea"),
        (t.grad.0, s.grad.0, "gradientul"),
        (t.rare_grad, s.rare_grad, "randurile rare"),
        (t.move_worst.0, s.move_worst.0, "miscarea"),
    ] {
        assert!(
            a <= 2.0 * b.max(1e-7),
            "{what}: _3 {a:e} fata de simplu {b:e}"
        );
    }
}

fn tc_ref() -> &'static usize {
    &simple(Mode::Mixed(LOSS_SCALE)).1
}

#[test]
fn t199_planted_norm_not_scaled_is_caught() {
    let (run, _) = on_board(Layout::Simple, Mode::MixedNormNotScaled(LOSS_SCALE));
    let f = figures(&run);
    show("mixt gresit (norma fata de 1,0)", &f);
    assert!(
        f.move_worst.0 > 5e-2,
        "greseala nu a fost prinsa: {:e}",
        f.move_worst.0
    );
}

/// T200 prediction 1 on `antrenare mini`'s shape (vocabulary 16 384): the mixed step (`mm_tc`, S = 1 024) and the
/// F32 step with `ln_back_res` give the same bits as without it, 5 AdamW steps, 4 launches fewer (SIMULARE).
#[test]
fn t200_mixed_mini_fused_is_bit_for_bit() {
    let bits = |r: &Run| {
        (
            r.losses.iter().map(|x| x.to_bits()).collect::<Vec<u64>>(),
            r.grad1
                .iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<u32>>(),
            r.after_last
                .iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<u32>>(),
        )
    };
    for mode in [Mode::Mixed(LOSS_SCALE), Mode::F32] {
        let (old, tc_old) = on_board(Layout::Tiles3, mode);
        let (new, tc_new) = on_board(Layout::Tiles3Fused, mode);
        assert_eq!(tc_old, tc_new);
        let same = bits(&old) == bits(&new);
        println!(
            "kt c200 antrenare mini {mode:?}: ln_back_res = nelipit bit cu bit {} (5 pasi AdamW, SIMULARE)",
            if same { "DA" } else { "NU" }
        );
        assert!(same, "{mode:?}: lipit != nelipit");
    }
    let (inputs, _) = batch();
    let p = Plan::new_tiles3(MINI, B, T, &inputs, HEAP).unwrap();
    let opt = AdamW::new(&initial(), LR, 0.9, 0.999, 1e-8, 0.1);
    let fused = p.clone().with_fused_back();
    assert_eq!(
        p.step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE)
            .unwrap()
            .len(),
        fused
            .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE)
            .unwrap()
            .len()
            + 2 * MINI.layers
    );
}

/// T201 prediction G1 on `antrenare mini`'s shape: the F32 step with both fusions (`ln_back_res` + `mm_train_gelu`)
/// gives the same bits as without them, 5 AdamW steps (SIMULARE); the mixed step refuses `with_fused_gelu`.
#[test]
fn t201_mini_f32_both_fusions_bit_for_bit() {
    let bits = |r: &Run| {
        (
            r.losses.iter().map(|x| x.to_bits()).collect::<Vec<u64>>(),
            r.grad1
                .iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<u32>>(),
            r.after_last
                .iter()
                .flatten()
                .map(|x| x.to_bits())
                .collect::<Vec<u32>>(),
        )
    };
    let (old, _) = on_board(Layout::Tiles3, Mode::F32);
    let (new, _) = on_board(Layout::Tiles3Both, Mode::F32);
    let same = bits(&old) == bits(&new);
    println!(
        "kt c201 antrenare mini F32: ln_back_res + mm_train_gelu = nelipit bit cu bit {} (5 pasi AdamW, SIMULARE)",
        if same { "DA" } else { "NU" }
    );
    assert!(same, "F32: lipit != nelipit");
    let (inputs, _) = batch();
    let p = Plan::new_tiles3(MINI, B, T, &inputs, HEAP).unwrap();
    let opt = AdamW::new(&initial(), LR, 0.9, 0.999, 1e-8, 0.1);
    assert!(
        p.with_fused_gelu()
            .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE)
            .is_err()
    );
}
