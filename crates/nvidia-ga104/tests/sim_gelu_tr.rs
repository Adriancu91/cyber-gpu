//! C-160 point 4 — T158, SIMULARE: training program 5 (`tools/nak/gelu_tr.comp`, compiled by Mesa NAK to
//! `tools/nak/out/gelu_tr.sm86.sass`): GELU forward (mode 0, `u·Φ(u)`) and backward (mode 1, `g·(Φ(u) + u·φ(u))`),
//! one element per thread, blocks of 256, launched through our pushbuffer + QMD in the command processor model. The
//! "program" runs on the CPU (`gelu_tr_on_cpu` → `gpu_compute::layer::gelu_tr_gpu`, in the order of the code NAK made;
//! `MUFU.RCP` / `MUFU.EX2` modelled correctly rounded). The outputs are judged against the f64 training oracle
//! (`gpu_compute::train::gelu_rows` / `gelu_back_rows`, the very lines `loss_and_grad` uses for the MLP, which gave
//! PyTorch's numbers at the small golden step, T151) within a margin written BEFORE the measurement (`bound_*`
//! below), with room for the board's approximate RCP and EX2. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::train;
use gpu_sim::cmdproc::{CommandProcessor, Event, gelu_tr_on_cpu, ln_back_on_cpu};
use nvidia_ga104::gelu_tr::{self as gt, GELU_TR, Mode};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const CBUF_BACK: u64 = 0x21_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const QMD_BACK: u64 = 0x31_1000;
const PUSH: u64 = 0x32_0000;
const PUSH_BACK: u64 = 0x32_1000;
const UU: u64 = 0x100_0000;
const G: u64 = 0x400_0000;
const Y: u64 = 0x800_0000;
const DU: u64 = 0xc00_0000;
const SENTINEL: f32 = -12345.0;
/// 2⁻²⁴: one rounding of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// Room for each of the board's `MUFU.RCP` and `MUFU.EX2` (relative): 2⁻²², written before the measurement. The
/// model rounds correctly (≤ u); the approximate units are documented at ~1–2 ulp.
const E_MUFU: f64 = 4.0 * U;
/// Abramowitz & Stegun 7.1.26: |erf − formula| ≤ 1.5e-7 with exact arithmetic.
const E_AS: f64 = 1.5e-7;
/// 2⁻¹²⁶: the smallest normal f32; the program flushes anything smaller to 0 (an absolute slack per FTZ step).
const FTZ: f64 = 1.175_494_350_822_287_5e-38;

// A&S 7.1.26 coefficients (c1..c5) and p.
const AS_P: f64 = 0.327_591_1;
const AS_C: [f64; 5] = [
    0.254_829_592,
    -0.284_496_736,
    1.421_413_741,
    -1.453_152_027,
    1.061_405_429,
];

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

// ------------------------------------------------------------------ the margin, written before the measurement
//
// With z = u/√2, a = |z|, t = 1/(1 + p·a), P(t) = Σ cᵢ tⁱ, ex = e^(−a²), Φ = ½(1 + erf):
//   t: the rounding of z, of the ffma, the constant p and the RCP → relative 4u + E_MUFU;
//   P: Horner in ffma + the final fmul + the f32 coefficients → |P'(t)|·t·Δt + 12u·Σ|cᵢ|tⁱ;
//   ex: w = a·log2e·a has relative error ≤ 5u, so e^(−a²) is off by 5u·a² relative, + the EX2 (u + E_MUFU);
//   erf: A&S + ΔP·ex + P·ex·Δex + u·|e| (the ffma) + the shift of the argument erf'(z)·u·a;
//   Φ: ½(Δerf + u·|1 + erf|) (the fadd; ·½ is exact);
//   forward y = u·Φ: |u|·ΔΦ + u·|y| + FTZ;
//   backward d = Φ + u·φ, φ = ex·(1/√2π): Δφ = φ·(Δex + 2u); Δd = ΔΦ + |u|·Δφ + u·|d|; y = g·d: |g|·Δd + u·|y| + FTZ.

struct Parts {
    dcdf: f64,
    pdf: f64,
    dpdf: f64,
}

fn parts(u: f32, e_mufu: f64) -> Parts {
    let u = u as f64;
    let a = (u * core::f64::consts::FRAC_1_SQRT_2).abs();
    let t = 1.0 / (1.0 + AS_P * a);
    let (mut pabs, mut dp) = (0.0f64, 0.0f64);
    for (i, ci) in AS_C.iter().enumerate() {
        pabs += ci.abs() * t.powi(i as i32 + 1);
        dp += (i as f64 + 1.0) * ci * t.powi(i as i32);
    }
    let p: f64 = AS_C
        .iter()
        .enumerate()
        .map(|(i, ci)| ci * t.powi(i as i32 + 1))
        .sum();
    let ex = (-a * a).exp();
    let dt = 4.0 * U + e_mufu;
    let d_p = dp.abs() * t * dt + 12.0 * U * pabs;
    let dex = e_mufu + U + 5.0 * U * a * a;
    let e = 1.0 - p * ex;
    let erf = if u < 0.0 { -e } else { e };
    let derf = E_AS
        + d_p * ex
        + p.abs() * ex * dex
        + U * e.abs()
        + 2.0 / core::f64::consts::PI.sqrt() * ex * U * a;
    let dcdf = 0.5 * (derf + U * (1.0 + erf).abs()) + FTZ;
    let pdf = ex / (2.0 * core::f64::consts::PI).sqrt();
    Parts {
        dcdf,
        pdf,
        dpdf: pdf * (dex + 2.0 * U) + FTZ,
    }
}

fn bound_fwd(u: f32, y_true: f32, e_mufu: f64) -> f64 {
    let q = parts(u, e_mufu);
    (u as f64).abs() * q.dcdf + U * (y_true as f64).abs() + 2.0 * FTZ
}

fn bound_back(u: f32, g: f32, y_true: f32, e_mufu: f64) -> f64 {
    let q = parts(u, e_mufu);
    let cdf = 0.5 * (1.0 + as_erf(u as f64 * core::f64::consts::FRAC_1_SQRT_2));
    let d = cdf + u as f64 * q.pdf;
    let dd = q.dcdf + (u as f64).abs() * q.dpdf + U * d.abs();
    (g as f64).abs() * dd + U * (y_true as f64).abs() + 2.0 * FTZ
}

/// erf in f64 by A&S, used only for the size of d in the margin.
fn as_erf(x: f64) -> f64 {
    // A&S is enough for a magnitude (|Δ| ≤ 1.5e-7 on a term multiplied by u).
    let a = x.abs();
    let t = 1.0 / (1.0 + AS_P * a);
    let p: f64 = AS_C
        .iter()
        .enumerate()
        .map(|(i, ci)| ci * t.powi(i as i32 + 1))
        .sum();
    let e = 1.0 - p * (-a * a).exp();
    if x < 0.0 { -e } else { e }
}

// ------------------------------------------------------------------ launching

fn cp_with(run: gpu_sim::cmdproc::ProgramFn) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program(GELU_TR, run);
    cp.mem.write(PROG, GELU_TR);
    cp
}

fn args(n: usize, mode: Mode, y: u64) -> gt::Args {
    gt::Args {
        u: UU,
        g: G,
        y,
        n: n as u32,
        mode,
    }
}

struct At {
    cbuf: u64,
    qmd: u64,
    push: u64,
    sem: u32,
}

const FIRST: At = At {
    cbuf: CBUF,
    qmd: QMD,
    push: PUSH,
    sem: 1,
};
const SECOND: At = At {
    cbuf: CBUF_BACK,
    qmd: QMD_BACK,
    push: PUSH_BACK,
    sem: 2,
};

fn write_cbuf(cp: &mut CommandProcessor, at: &At, a: &gt::Args) {
    let mut cb = vec![0u8; 0x100];
    cb[..gt::CBUF_BYTES].copy_from_slice(&a.cbuf().unwrap());
    cp.mem.write(at.cbuf, &cb);
}

fn launch_at(
    cp: &mut CommandProcessor,
    at: &At,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<(), gpu_core::error::GpuError> {
    let q = ComputeQmd {
        program_va: PROG,
        registers: gt::REGISTERS,
        cbuf0_va: at.cbuf,
        cbuf0_size: 0x100,
        grid,
        block,
        release: Some((SEM, at.sem)),
    };
    cp.mem.write(at.qmd, &q.to_bytes().unwrap());
    let mut w = [0u32; 16];
    let mut p = PushBuf::new(&mut w);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(at.qmd >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = p.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(at.push, &bytes);
    let e = push::gpfifo_entry(at.push, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

fn load(cp: &mut CommandProcessor, u: &[f32], g: &[f32]) {
    cp.mem.write_f32s(UU, u);
    cp.mem.write_f32s(G, g);
    cp.mem.write_f32s(Y, &vec![SENTINEL; u.len() + 300]);
    cp.mem.write_f32s(DU, &vec![SENTINEL; u.len() + 300]);
    cp.mem.write_u32(SEM, 0);
}

/// One launch of `mode` over all of `u`, with `grid` blocks; returns y (and checks nothing past n was written).
fn run_grid(
    run: gpu_sim::cmdproc::ProgramFn,
    u: &[f32],
    g: &[f32],
    mode: Mode,
    blocks: u32,
) -> Vec<f32> {
    let mut cp = cp_with(run);
    load(&mut cp, u, g);
    let a = args(u.len(), mode, Y);
    write_cbuf(&mut cp, &FIRST, &a);
    launch_at(&mut cp, &FIRST, [blocks, 1, 1], gt::BLOCK).unwrap();
    assert!(matches!(cp.events[1], Event::Launch { program: 0, .. }));
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let y = cp.mem.read_f32s(Y, u.len() + 300).unwrap();
    assert!(y[u.len()..].iter().all(|&v| v == SENTINEL), "wrote after n");
    y[..u.len()].to_vec()
}

fn run(u: &[f32], g: &[f32], mode: Mode) -> Vec<f32> {
    run_grid(gelu_tr_on_cpu, u, g, mode, args(u.len(), mode, Y).grid()[0])
}

/// The largest share of the margin used (1.0 = at the edge); `Err` at the first element outside.
fn judge(u: &[f32], g: &[f32], mode: Mode, got: &[f32]) -> Result<f64, String> {
    judge_with(u, g, mode, got, E_MUFU)
}

/// [`judge`] with another room for the MUFU units (0 = the model's own margin, no room for the board).
fn judge_with(u: &[f32], g: &[f32], mode: Mode, got: &[f32], e_mufu: f64) -> Result<f64, String> {
    let want = match mode {
        Mode::Forward => train::gelu_rows(u),
        Mode::Backward => train::gelu_back_rows(g, u).unwrap(),
    };
    let mut worst = 0.0f64;
    for i in 0..u.len() {
        let b = match mode {
            Mode::Forward => bound_fwd(u[i], want[i], e_mufu),
            Mode::Backward => bound_back(u[i], g[i], want[i], e_mufu),
        };
        let d = (got[i] as f64 - want[i] as f64).abs();
        if d.is_nan() || d > b {
            return Err(format!(
                "{mode:?} u[{i}] = {}: got {} want {} |Δ| {d:e} > {b:e}",
                u[i], got[i], want[i]
            ));
        }
        worst = worst.max(d / b);
    }
    Ok(worst)
}

fn sweep(lo: f32, hi: f32, n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| lo + (hi - lo) * i as f32 / (n - 1) as f32)
        .collect()
}

fn specials() -> Vec<f32> {
    let mut v = vec![
        0.0, -0.0, 1e-30, -1e-30, 1e-40, -1e-40, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, -3.0, 5.5,
        -5.5, 9.3, -9.3, 13.0, -13.0, 14.0, -14.0, 20.0, -20.0, 50.0, -50.0, 1e4, -1e4,
    ];
    v.extend([
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        0.327_591_1,
        -2.5e-3,
        7.0e-4,
    ]);
    v
}

// ------------------------------------------------------------------ tests

#[test]
fn t158_binary_is_the_recorded_build() {
    let hex: String = nvidia_ga104::sha256::digest(GELU_TR)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, gt::GELU_TR_SHA256);
    assert_eq!(GELU_TR.len(), 57 * 16, "57 instructions of 16 bytes");
    let txt = include_str!("../../../tools/nak/out/gelu_tr.sm86.txt");
    assert_eq!(
        txt.matches("mufu.rcp").count(),
        1,
        "one MUFU.RCP (the t of A&S)"
    );
    assert_eq!(
        txt.matches("mufu.exp2").count(),
        1,
        "one MUFU.EX2 shared by Φ and φ"
    );
    assert!(!txt.contains("mufu.rsq") && !txt.contains("bar.") && !txt.contains("shfl"));
    assert!(txt.contains("c[0x0][0x44]"), "the mode is read from 0x44");
    assert!(txt.contains("Num GPRs: 19"));
    let ok = args(1000, Mode::Backward, Y);
    assert!(ok.cbuf().is_some());
    assert!(gt::Args { n: 0, ..ok }.cbuf().is_none());
    assert!(gt::Args { g: 0, ..ok }.cbuf().is_none(), "backward needs g");
    assert!(gt::Args { y: 0, ..ok }.cbuf().is_none());
    assert!(gt::Args { u: 0, ..ok }.cbuf().is_none());
    let fwd = gt::Args {
        g: 0,
        mode: Mode::Forward,
        ..ok
    };
    assert!(fwd.cbuf().is_some(), "forward never reads g");
    let cb = ok.cbuf().unwrap();
    assert_eq!(u64::from_le_bytes(cb[0x28..0x30].try_into().unwrap()), UU);
    assert_eq!(u64::from_le_bytes(cb[0x30..0x38].try_into().unwrap()), G);
    assert_eq!(u64::from_le_bytes(cb[0x38..0x40].try_into().unwrap()), Y);
    assert_eq!(u32::from_le_bytes(cb[0x40..0x44].try_into().unwrap()), 1000);
    assert_eq!(u32::from_le_bytes(cb[0x44..0x48].try_into().unwrap()), 1);
    assert_eq!(
        u32::from_le_bytes(fwd.cbuf().unwrap()[0x44..0x48].try_into().unwrap()),
        0
    );
    assert_eq!(ok.grid(), [4, 1, 1]);
    assert_eq!(gt::Args { n: 1024, ..ok }.grid(), [4, 1, 1]);
    assert_eq!(gt::Args { n: 1025, ..ok }.grid(), [5, 1, 1]);
}

#[test]
fn t158_forward_and_backward_are_within_the_written_margin() {
    let mut r = rng(0x9e1);
    let mut cases: Vec<(&str, Vec<f32>)> = vec![
        ("sweep -12..12, 10 007", sweep(-12.0, 12.0, 10_007)),
        ("sweep -1..1 (near 0), 3 001", sweep(-1.0, 1.0, 3_001)),
        ("special values", specials()),
    ];
    let normal: Vec<f32> = (0..4096).map(|_| 2.0 * (r() + r() + r())).collect();
    cases.push(("fc-like, |u| <= 6, 4 096", normal));
    let mut worst_all = (0.0f64, 0.0f64);
    for (what, u) in &cases {
        let g: Vec<f32> = (0..u.len()).map(|_| r() * 1e-2).collect();
        let y = run(u, &g, Mode::Forward);
        let wf = judge(u, &g, Mode::Forward, &y).unwrap_or_else(|e| panic!("{what}: {e}"));
        let du = run(u, &g, Mode::Backward);
        let wb = judge(u, &g, Mode::Backward, &du).unwrap_or_else(|e| panic!("{what}: {e}"));
        // The model (correctly rounded RCP / EX2) fits even with no room for the board's units, so a board whose
        // units are within E_MUFU of exact stays inside the full margin (the MUFU terms are added on top).
        let mf = judge_with(u, &g, Mode::Forward, &y, 0.0)
            .unwrap_or_else(|e| panic!("{what}, no room: {e}"));
        let mb = judge_with(u, &g, Mode::Backward, &du, 0.0)
            .unwrap_or_else(|e| panic!("{what}, no room: {e}"));
        println!(
            "T158 {what}: forward {:.1} %, backward {:.1} % of the margin (without the board's room: {:.1} %, {:.1} %)",
            100.0 * wf,
            100.0 * wb,
            100.0 * mf,
            100.0 * mb
        );
        worst_all = (worst_all.0.max(wf), worst_all.1.max(wb));
        // The launched model is the per-element model, bit for bit.
        for i in 0..u.len() {
            assert_eq!(
                y[i].to_bits(),
                gpu_compute::layer::gelu_tr_gpu(u[i], None).to_bits()
            );
            assert_eq!(
                du[i].to_bits(),
                gpu_compute::layer::gelu_tr_gpu(u[i], Some(g[i])).to_bits()
            );
        }
    }
    println!(
        "T158 worst: forward {:.1} %, backward {:.1} % of the margin",
        100.0 * worst_all.0,
        100.0 * worst_all.1
    );
}

#[test]
fn t158_absolute_errors_are_small() {
    // The size of what the margin allows, for the note: on |u| ≤ 12, g = 1.
    let u = sweep(-12.0, 12.0, 20_001);
    let g = vec![1.0f32; u.len()];
    let y = run(&u, &g, Mode::Forward);
    let du = run(&u, &g, Mode::Backward);
    let wf = train::gelu_rows(&u);
    let wb = train::gelu_back_rows(&g, &u).unwrap();
    let mf = (0..u.len())
        .map(|i| (y[i] - wf[i]).abs())
        .fold(0.0f32, f32::max);
    let mb = (0..u.len())
        .map(|i| (du[i] - wb[i]).abs())
        .fold(0.0f32, f32::max);
    println!("T158 max |Δ| on [-12, 12]: forward {mf:e}, backward (g = 1) {mb:e}");
    assert!(mf < 2e-6 && mb < 2e-7, "{mf:e} {mb:e}");
}

#[test]
fn t158_runs_are_the_same_bits_and_forward_ignores_g() {
    let mut r = rng(7);
    let u: Vec<f32> = (0..777).map(|_| 5.0 * r()).collect();
    let g1: Vec<f32> = (0..777).map(|_| r()).collect();
    let g2: Vec<f32> = (0..777).map(|_| 1e3 * r()).collect();
    let a = run(&u, &g1, Mode::Forward);
    let b = run(&u, &g2, Mode::Forward);
    assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    let c1 = run(&u, &g1, Mode::Backward);
    let c2 = run(&u, &g1, Mode::Backward);
    assert!(c1.iter().zip(&c2).all(|(x, y)| x.to_bits() == y.to_bits()));
    assert!(c1.iter().zip(&a).any(|(x, y)| x != y), "the modes differ");
}

#[test]
fn t158_a_short_grid_leaves_the_tail_untouched() {
    let u = sweep(-3.0, 3.0, 600);
    let g = vec![0.5f32; 600];
    let mut cp = cp_with(gelu_tr_on_cpu);
    load(&mut cp, &u, &g);
    write_cbuf(&mut cp, &FIRST, &args(600, Mode::Backward, Y));
    launch_at(&mut cp, &FIRST, [1, 1, 1], gt::BLOCK).unwrap();
    let y = cp.mem.read_f32s(Y, 600).unwrap();
    assert!(y[..256].iter().all(|&v| v != SENTINEL));
    assert!(
        y[256..].iter().all(|&v| v == SENTINEL),
        "one block = 256 elements"
    );
}

#[test]
fn t158_refuses_a_wrong_block_or_grid() {
    let u = sweep(-3.0, 3.0, 256);
    let g = vec![1.0f32; 256];
    let mut cp = cp_with(gelu_tr_on_cpu);
    load(&mut cp, &u, &g);
    write_cbuf(&mut cp, &FIRST, &args(256, Mode::Forward, Y));
    assert!(launch_at(&mut cp, &FIRST, [1, 1, 1], [128, 1, 1]).is_err());
    assert!(launch_at(&mut cp, &FIRST, [1, 2, 1], gt::BLOCK).is_err());
    let y = cp.mem.read_f32s(Y, 256).unwrap();
    assert!(
        y.iter().all(|&v| v == SENTINEL),
        "refused launches wrote nothing"
    );
}

#[test]
fn t158_forward_then_backward_chain() {
    // Two launches in one command processor, as in the training step: mode 0 writes the activation (kept for proj's
    // backward), mode 1 reads the same u and the gradient of the activation and writes du.
    let mut r = rng(0xc4a1);
    let n = 64 * 256 + 17; // the small model's MLP: rows × 4C, not a multiple of 256
    let u: Vec<f32> = (0..n).map(|_| 3.0 * (r() + r())).collect();
    let g: Vec<f32> = (0..n).map(|_| 1e-3 * r()).collect();
    let mut cp = cp_with(gelu_tr_on_cpu);
    load(&mut cp, &u, &g);
    let fwd = args(n, Mode::Forward, Y);
    let back = args(n, Mode::Backward, DU);
    write_cbuf(&mut cp, &FIRST, &fwd);
    write_cbuf(&mut cp, &SECOND, &back);
    launch_at(&mut cp, &FIRST, fwd.grid(), gt::BLOCK).unwrap();
    launch_at(&mut cp, &SECOND, back.grid(), gt::BLOCK).unwrap();
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 2);
    let act = cp.mem.read_f32s(Y, n).unwrap();
    let du = cp.mem.read_f32s(DU, n + 300).unwrap();
    assert!(du[n..].iter().all(|&v| v == SENTINEL));
    let wf = judge(&u, &g, Mode::Forward, &act).unwrap();
    let wb = judge(&u, &g, Mode::Backward, &du[..n]).unwrap();
    println!(
        "T158 chain: activation {:.1} %, du {:.1} % of the margin",
        100.0 * wf,
        100.0 * wb
    );
}

#[test]
fn t158_planted_mistakes_are_caught() {
    let u = sweep(-6.0, 6.0, 2_001);
    let g: Vec<f32> = (0..u.len()).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
    let fwd = |f: fn(f32) -> f32| -> Vec<f32> { u.iter().map(|&x| f(x)).collect() };
    let back = |f: fn(f32, f32) -> f32| -> Vec<f32> {
        u.iter().zip(&g).map(|(&x, &gi)| f(x, gi)).collect()
    };
    fn cdf(x: f32) -> f32 {
        0.5 * (1.0 + gpu_compute::math::erf(x * core::f32::consts::FRAC_1_SQRT_2))
    }
    fn pdf(x: f32) -> f32 {
        gpu_compute::math::exp(-0.5 * x * x) * 0.398_942_3
    }
    let mistakes: Vec<(&str, Mode, Vec<f32>)> = vec![
        // (1) The derivative without the u·φ term.
        ("du without u*phi", Mode::Backward, back(|x, g| g * cdf(x))),
        // (2) φ without 1/√(2π).
        (
            "phi without 1/sqrt(2pi)",
            Mode::Backward,
            back(|x, g| g * (cdf(x) + x * pdf(x) / 0.398_942_3)),
        ),
        // (3) The tanh approximation of GELU (PyTorch's approximate='tanh'), not Kronos's exact erf.
        (
            "tanh GELU",
            Mode::Forward,
            fwd(|x| {
                0.5 * x * (1.0 + gpu_compute::math::tanh(0.797_884_6 * (x + 0.044_715 * x * x * x)))
            }),
        ),
        // (4) The modes swapped.
        (
            "backward in forward's place",
            Mode::Forward,
            back(|x, g| gpu_compute::layer::gelu_tr_gpu(x, Some(g))),
        ),
        (
            "forward in backward's place",
            Mode::Backward,
            fwd(|x| gpu_compute::layer::gelu_tr_gpu(x, None)),
        ),
        // (5) Φ(u) instead of u·Φ(u).
        ("Phi instead of u*Phi", Mode::Forward, fwd(cdf)),
        // (6) The sign of erf lost for negative u.
        (
            "erf without its sign",
            Mode::Forward,
            fwd(|x| {
                0.5 * x * (1.0 + gpu_compute::math::erf(x.abs() * core::f32::consts::FRAC_1_SQRT_2))
            }),
        ),
    ];
    for (what, mode, got) in &mistakes {
        assert!(
            judge(&u, &g, *mode, got).is_err(),
            "planted mistake not caught: {what}"
        );
    }
    // (7) Another program's model registered for this code (ln_back: other block contract) refuses or writes garbage.
    let mut cp = cp_with(ln_back_on_cpu);
    load(&mut cp, &u, &g);
    write_cbuf(&mut cp, &FIRST, &args(u.len(), Mode::Forward, Y));
    let caught = match launch_at(
        &mut cp,
        &FIRST,
        args(u.len(), Mode::Forward, Y).grid(),
        gt::BLOCK,
    ) {
        Err(_) => true,
        Ok(()) => {
            let y = cp.mem.read_f32s(Y, u.len()).unwrap();
            judge(&u, &g, Mode::Forward, &y).is_err()
        }
    };
    assert!(caught, "another program's model was not caught");
}
