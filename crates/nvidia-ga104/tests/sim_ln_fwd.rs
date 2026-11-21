//! C-160 point 4 — T157, SIMULARE: training program 3 (`tools/nak/ln_fwd.comp`, compiled by Mesa NAK to
//! `tools/nak/out/ln_fwd.sm86.sass`): LayerNorm forward over rows of C numbers that keeps μ and rstd for the backward
//! pass, one block of 256 threads per row, launched through our pushbuffer + QMD in the command processor model. The
//! "program" runs on the CPU (`ln_fwd_on_cpu` → `gpu_compute::layer::ln_fwd_gpu256`, in the order of the code NAK
//! made; `MUFU.RSQ` modelled correctly rounded). y, μ and rstd are judged against the f64 training oracle
//! (`gpu_compute::train::layernorm_rows` + `layernorm_stats`, the very calculation `loss_and_grad` does, which gave
//! PyTorch's numbers at the small golden step, T151) within a margin written BEFORE the measurement (`bounds`
//! below), with room for the board's approximate RSQ. Then the chain forward → backward: `ln_back` fed with the μ and
//! rstd this program wrote. Nothing here runs on the RTX 3070 Ti.

use std::path::{Path, PathBuf};

use gpu_compute::train::{self, MIC};
use gpu_sim::cmdproc::{
    CommandProcessor, Event, LN_BACK_NEEDS, LN_FWD_NEEDS, ln_back_on_cpu, ln_fwd_on_cpu,
    xent_on_cpu,
};
use nvidia_ga104::ln_back::{self as lb, LN_BACK};
use nvidia_ga104::ln_fwd::{self as lf, LN_FWD};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const PROG_BACK: u64 = 0x18_0000;
const CBUF: u64 = 0x20_0000;
const CBUF_BACK: u64 = 0x21_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const QMD_BACK: u64 = 0x31_1000;
const PUSH: u64 = 0x32_0000;
const PUSH_BACK: u64 = 0x32_1000;
const GAMMA: u64 = 0x40_0000;
const BETA: u64 = 0x44_0000;
const MEAN: u64 = 0x48_0000;
const RSTD: u64 = 0x4c_0000;
const X: u64 = 0x100_0000;
const Y: u64 = 0x400_0000;
const DY: u64 = 0x800_0000;
const DX: u64 = 0xc00_0000;
const EPS: f32 = 1e-5;
const SENTINEL: f32 = -12345.0;
/// 2⁻²⁴: one rounding of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// Room for the board's `MUFU.RSQ` (relative): 2⁻²², written before the measurement. The model rounds correctly
/// (≤ u); NVIDIA documents ~2 ulp (2⁻²² relative) for the approximate reciprocal square root.
const E_RSQ: f64 = 4.0 * U;
/// 2⁻¹²⁶: the smallest normal f32; the program flushes anything smaller to 0.
const FTZ: f64 = 1.175_494_350_822_287_5e-38;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn cp_with(run: gpu_sim::cmdproc::ProgramFn) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(LN_FWD, run, LN_FWD_NEEDS);
    cp.mem.write(PROG, LN_FWD);
    cp
}

fn args(rows: usize, cc: usize) -> lf::Args {
    lf::Args {
        x: X,
        gamma: GAMMA,
        beta: BETA,
        y: Y,
        mean: MEAN,
        rstd: RSTD,
        rows: rows as u32,
        c: cc as u32,
        eps: EPS,
    }
}

struct Case {
    x: Vec<f32>,
    gamma: Vec<f32>,
    beta: Vec<f32>,
    rows: usize,
    c: usize,
}

fn load(cp: &mut CommandProcessor, k: &Case) {
    let mut cb = vec![0u8; 0x100];
    cb[..lf::CBUF_BYTES].copy_from_slice(&args(k.rows, k.c).cbuf().unwrap());
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(X, &k.x);
    cp.mem.write_f32s(GAMMA, &k.gamma);
    cp.mem.write_f32s(BETA, &k.beta);
    cp.mem.write_f32s(Y, &vec![SENTINEL; k.rows * k.c + 16]);
    cp.mem.write_f32s(MEAN, &vec![SENTINEL; k.rows + 4]);
    cp.mem.write_f32s(RSTD, &vec![SENTINEL; k.rows + 4]);
    cp.mem.write_u32(SEM, 0);
}

struct Launch {
    prog: u64,
    registers: u32,
    cbuf: u64,
    qmd: u64,
    push: u64,
    sem: u32,
}

const FWD: Launch = Launch {
    prog: PROG,
    registers: lf::REGISTERS,
    cbuf: CBUF,
    qmd: QMD,
    push: PUSH,
    sem: 1,
};

fn launch_at(
    cp: &mut CommandProcessor,
    at: &Launch,
    grid: [u32; 3],
    block: [u32; 3],
    shared: SharedMemory,
) -> Result<(), gpu_core::error::GpuError> {
    let q = ComputeQmd {
        program_va: at.prog,
        registers: at.registers,
        cbuf0_va: at.cbuf,
        cbuf0_size: 0x100,
        grid,
        block,
        release: Some((SEM, at.sem)),
    };
    cp.mem
        .write(at.qmd, &q.to_bytes_with_shared(&shared).unwrap());
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

fn launch(
    cp: &mut CommandProcessor,
    grid: [u32; 3],
    block: [u32; 3],
    shared: SharedMemory,
) -> Result<(), gpu_core::error::GpuError> {
    launch_at(cp, &FWD, grid, block, shared)
}

struct Out {
    y: Vec<f32>,
    mean: Vec<f32>,
    rstd: Vec<f32>,
}

/// Runs the program on `grid` rows; checks nothing was written after y, μ, rstd.
fn run_grid(k: &Case, rows_launched: usize) -> Out {
    let mut cp = cp_with(ln_fwd_on_cpu);
    load(&mut cp, k);
    let grid = args(rows_launched, k.c).grid();
    assert_eq!(grid, [rows_launched as u32, 1, 1]);
    launch(&mut cp, grid, lf::BLOCK, lf::SHARED).unwrap();
    assert!(matches!(cp.events[1], Event::Launch { program: 0, .. }));
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let y = cp.mem.read_f32s(Y, k.rows * k.c + 16).unwrap();
    let mean = cp.mem.read_f32s(MEAN, k.rows + 4).unwrap();
    let rstd = cp.mem.read_f32s(RSTD, k.rows + 4).unwrap();
    assert!(
        y[k.rows * k.c..].iter().all(|&v| v == SENTINEL),
        "wrote after y"
    );
    assert!(mean[k.rows..].iter().all(|&v| v == SENTINEL));
    assert!(rstd[k.rows..].iter().all(|&v| v == SENTINEL));
    Out {
        y: y[..k.rows * k.c].to_vec(),
        mean: mean[..k.rows].to_vec(),
        rstd: rstd[..k.rows].to_vec(),
    }
}

fn run(k: &Case) -> Out {
    run_grid(k, k.rows)
}

// ------------------------------------------------------------------ the margin, written before the measurement
//
// For one row (u = 2⁻²⁴; μ, var, rs = 1/√(var + ε), dₖ = xₖ − μ, hₖ = dₖ·rs, yₖ = hₖ·γₖ + βₖ exact in f64),
// P = ⌈c/256⌉ + 12 roundings on the path of a sum (the thread's adds, 5 butterfly, 7 warp adds):
//   S = Σx:  (P + 1)·u·Σ|x|;  μ̃ = S̃·inv_c, inv_c and the product rounded → Δμ = ΔS/c + 2u·|μ|;
//   d̃ₖ = x − μ̃ rounded → |Δdₖ| ≤ Δμ + u·|dₖ|;
//   Q = Σd̃² (fma chain): Σd = 0, so μ̃'s shift enters only squared → ΔQ = (P + 3)·u·Σd² + c·(Δμ + u·max|d|)²;
//   v = Q̃·inv_c + ε (one fma, inv_c rounded) → Δv = ΔQ/c + u·var + u·(var + ε);
//   rs̃ = RSQ(ṽ) → Δrs = rs·(½·Δv/(var + ε) + E_RSQ) (·1.01 for the second order);
//   h̃ = d̃·rs̃ rounded → Δh = |Δd|·rs + |d|·Δrs + u·|h|;  ỹ = fma(h̃, γ, β) → Δy = |γ|·Δh + u·|y|
//   + the oracle's own rounding to f32 (u·|y|) + FTZ ((|γ| + 2)·2⁻¹²⁶).
// μ written vs the oracle's μ rounded to f32: Δμ + u·|μ|;  rstd written: Δrs + u·rs.

struct Bounds {
    y: Vec<f64>,
    mean: f64,
    rstd: f64,
}

fn bounds(x: &[f32], gamma: &[f32], beta: &[f32]) -> Bounds {
    let c = x.len();
    let cf = c as f64;
    let eps = EPS as f64;
    let mu = x.iter().map(|&v| v as f64).sum::<f64>() / cf;
    let d: Vec<f64> = x.iter().map(|&v| v as f64 - mu).collect();
    let sd2: f64 = d.iter().map(|v| v * v).sum();
    let var = sd2 / cf;
    let rs = 1.0 / (var + eps).sqrt();
    let p = c.div_ceil(256) as f64 + 12.0;
    let ax: f64 = x.iter().map(|&v| (v as f64).abs()).sum();
    let dmu = (p + 1.0) * U * ax / cf + 2.0 * U * mu.abs();
    let dmax = d.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let dq = (p + 3.0) * U * sd2 + cf * (dmu + U * dmax).powi(2);
    let dv = dq / cf + U * var + U * (var + eps);
    let drs = 1.01 * rs * (0.5 * dv / (var + eps) + E_RSQ);
    let y = (0..c)
        .map(|k| {
            let h = d[k] * rs;
            let yk = h * gamma[k] as f64 + beta[k] as f64;
            let dd = dmu + U * d[k].abs();
            let dh = dd * rs + d[k].abs() * drs + U * h.abs();
            (gamma[k] as f64).abs() * dh
                + 2.0 * U * yk.abs()
                + ((gamma[k] as f64).abs() + 2.0) * FTZ
        })
        .collect();
    Bounds {
        y,
        mean: dmu + U * mu.abs(),
        rstd: drs + U * rs,
    }
}

#[derive(Default, Debug)]
struct Worst {
    y: f64,
    y_ratio: f64,
    mean_ratio: f64,
    rstd_ratio: f64,
}

fn check(k: &Case, o: &Out) -> Result<Worst, String> {
    let oy = train::layernorm_rows(&k.x, &k.gamma, &k.beta, k.rows, k.c).unwrap();
    let (om, or) = train::layernorm_stats(&k.x, k.rows, k.c).unwrap();
    let mut w = Worst::default();
    for r in 0..k.rows {
        let s = r * k.c..(r + 1) * k.c;
        let b = bounds(&k.x[s.clone()], &k.gamma, &k.beta);
        for (j, i) in s.enumerate() {
            let diff = (o.y[i] as f64 - oy[i] as f64).abs();
            if diff > b.y[j] {
                return Err(format!("y[{r},{j}]: |Δ| {diff:e} > {:e}", b.y[j]));
            }
            w.y = w.y.max(diff);
            w.y_ratio = w.y_ratio.max(diff / b.y[j]);
        }
        let dm = (o.mean[r] as f64 - om[r] as f64).abs();
        if dm > b.mean {
            return Err(format!("mean[{r}]: |Δ| {dm:e} > {:e}", b.mean));
        }
        let dr = (o.rstd[r] as f64 - or[r] as f64).abs();
        if dr > b.rstd {
            return Err(format!("rstd[{r}]: |Δ| {dr:e} > {:e}", b.rstd));
        }
        w.mean_ratio = w.mean_ratio.max(dm / b.mean);
        w.rstd_ratio = w.rstd_ratio.max(dr / b.rstd);
    }
    Ok(w)
}

fn judge(what: &str, k: &Case, o: &Out) -> Worst {
    let w = check(k, o).unwrap_or_else(|e| panic!("{what}: {e}"));
    eprintln!(
        "T157 {what}: y max |Δ| = {:.3e} ({:.4} of the margin); μ {:.4}, rstd {:.4} of their margins",
        w.y, w.y_ratio, w.mean_ratio, w.rstd_ratio
    );
    w
}

fn synthetic(seed: u64, rows: usize, cc: usize, offset: f32, spread: f32) -> Case {
    let mut r = rng(seed);
    Case {
        x: (0..rows * cc).map(|_| offset + r() * spread).collect(),
        gamma: (0..cc).map(|_| 1.0 + r() * 0.2).collect(),
        beta: (0..cc).map(|_| r() * 0.1).collect(),
        rows,
        c: cc,
    }
}

#[test]
fn t157_binary_is_the_recorded_build() {
    let hex: String = nvidia_ga104::sha256::digest(LN_FWD)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, lf::LN_FWD_SHA256);
    assert_eq!(LN_FWD.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out/ln_fwd.sm86.txt");
    assert!(txt.contains("bar"), "no barrier in the dump");
    assert!(txt.contains("shfl"), "warp butterflies expected");
    assert_eq!(
        txt.matches("mufu.rsq").count(),
        1,
        "exactly one MUFU.RSQ (the inversesqrt of the row)"
    );
    assert!(
        !txt.contains("mufu.rcp") && !txt.contains("mufu.ex2") && !txt.contains("mufu.lg2"),
        "no other MUFU expected"
    );
    let ok = args(4, 768);
    assert!(ok.cbuf().is_some());
    assert!(lf::Args { rows: 0, ..ok }.cbuf().is_none());
    assert!(lf::Args { c: 0, ..ok }.cbuf().is_none());
    assert!(lf::Args { rstd: 0, ..ok }.cbuf().is_none());
    assert!(lf::Args { eps: 0.0, ..ok }.cbuf().is_none());
    assert!(
        lf::Args {
            eps: f32::NAN,
            ..ok
        }
        .cbuf()
        .is_none()
    );
    let cb = ok.cbuf().unwrap();
    assert_eq!(u64::from_le_bytes(cb[0x28..0x30].try_into().unwrap()), X);
    assert_eq!(u64::from_le_bytes(cb[0x50..0x58].try_into().unwrap()), RSTD);
    assert_eq!(u32::from_le_bytes(cb[0x58..0x5c].try_into().unwrap()), 768);
    assert_eq!(
        f32::from_le_bytes(cb[0x5c..0x60].try_into().unwrap()),
        1.0 / 768.0
    );
    assert_eq!(f32::from_le_bytes(cb[0x60..0x64].try_into().unwrap()), EPS);
}

#[test]
fn t157_synthetic_rows_are_within_the_written_margin() {
    for (i, (rows, cc, offset, spread)) in [
        (3usize, 1usize, 0.0f32, 1.0f32), // var = 0: rstd = 1/√ε
        (4, 64, 0.0, 1.0),                // the small model's width
        (3, 255, 0.5, 2.0),
        (3, 256, 0.0, 1.0),
        (3, 257, -1.0, 3.0),
        (4, 768, 0.0, 1.0), // #513's width
        (2, 1000, 0.0, 0.05),
        (2, 1024, 0.0, 1.0),
        (2, 2048, 0.0, 1.0),  // more than layernorm256 can hold
        (2, 3072, 0.0, 4.0),  // an MLP-wide row (4·768)
        (3, 768, 100.0, 1.0), // a large mean: μ rounded to f32 counts
        (2, 768, 0.0, 0.003), // var ≈ ε: ε counts
    ]
    .into_iter()
    .enumerate()
    {
        let k = synthetic(0x1f_0000 + i as u64, rows, cc, offset, spread);
        let o = run(&k);
        judge(
            &format!("rows={rows} c={cc} offset={offset} spread={spread}"),
            &k,
            &o,
        );
        // The model is what NAK's order gives, row by row (the launch adds nothing).
        for r in 0..rows {
            let s = r * cc..(r + 1) * cc;
            let (y, m, rs) = gpu_compute::layer::ln_fwd_gpu256(
                &k.x[s.clone()],
                &k.gamma,
                &k.beta,
                1.0 / cc as f32,
                EPS,
            );
            assert!(
                y.iter()
                    .zip(&o.y[s])
                    .all(|(a, b)| a.to_bits() == b.to_bits())
            );
            assert_eq!(m.to_bits(), o.mean[r].to_bits());
            assert_eq!(rs.to_bits(), o.rstd[r].to_bits());
        }
    }
}

#[test]
fn t157_runs_are_the_same_bits() {
    let k = synthetic(77, 5, 768, 0.0, 1.0);
    let a = run(&k);
    let b = run(&k);
    assert!(
        a.y.iter()
            .zip(&b.y)
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
    assert!(
        a.mean
            .iter()
            .zip(&b.mean)
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
    assert!(
        a.rstd
            .iter()
            .zip(&b.rstd)
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );
}

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../gpu-compute/tests/date/pas_de_aur_mic")
}

fn read_f32(p: &Path, n: usize) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(b.len(), 4 * n, "{}", p.display());
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

#[test]
fn t157_the_small_golden_model_layernorms() {
    // Every LayerNorm (γ, β) of the small golden model after 10 steps of `sgd` (PyTorch's weights: not ones and
    // zeros), on the golden batch's shape (64 rows × 64).
    let mut seen = 0;
    for t in MIC.tensors() {
        if !(t.name.contains("ln") && t.name.ends_with("weight")) || t.elements() != MIC.n_emb {
            continue;
        }
        let at = dir().join("sgd").join("dupa_pas10");
        let gamma = read_f32(&at.join(format!("{}.f32", t.name)), t.elements());
        let bias = t.name.replace("weight", "bias");
        let beta = read_f32(&at.join(format!("{bias}.f32")), t.elements());
        let mut k = synthetic(0x70 + seen, 64, MIC.n_emb, 0.0, 1.5);
        k.gamma = gamma;
        k.beta = beta;
        let o = run(&k);
        judge(&t.name, &k, &o);
        seen += 1;
    }
    assert!(
        seen >= 3,
        "the small model has LN1, LN2 per layer and the final LN (found {seen})"
    );
}

/// The chain the training step will use: `ln_fwd` writes μ and rstd, `ln_back` reads them (two launches, one
/// command processor). dx is judged against `train::layernorm_back_rows` with a tolerance written before the
/// measurement: per row, max |Δdx| ≤ 2⁻¹⁶ · max |dx| (the ln_back margin is ~1e-7 relative; the RSQ room and μ
/// rounded add at most a few u on x̂ and rs).
#[test]
fn t157_forward_then_backward_chain() {
    for (i, (rows, cc, offset)) in [
        (4usize, 64usize, 0.0f32),
        (3, 768, 0.0),
        (2, 768, 50.0),
        (2, 2048, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        let k = synthetic(0x2c_0000 + i as u64, rows, cc, offset, 1.0);
        let mut r = rng(0x5eed + i as u64);
        let dy: Vec<f32> = (0..rows * cc).map(|_| r() * 0.01).collect();
        let mut cp = cp_with(ln_fwd_on_cpu);
        cp.register_program_with_needs(LN_BACK, ln_back_on_cpu, LN_BACK_NEEDS);
        cp.mem.write(PROG_BACK, LN_BACK);
        load(&mut cp, &k);
        cp.mem.write_f32s(DY, &dy);
        cp.mem.write_f32s(DX, &vec![SENTINEL; rows * cc]);
        let back = lb::Args {
            x: X,
            dy: DY,
            gamma: GAMMA,
            mean: MEAN,
            rstd: RSTD,
            dx: DX,
            rows: rows as u32,
            c: cc as u32,
        };
        let mut cb = vec![0u8; 0x100];
        cb[..lb::CBUF_BYTES].copy_from_slice(&back.cbuf().unwrap());
        cp.mem.write(CBUF_BACK, &cb);
        launch(&mut cp, args(rows, cc).grid(), lf::BLOCK, lf::SHARED).unwrap();
        let at = Launch {
            prog: PROG_BACK,
            registers: lb::REGISTERS,
            cbuf: CBUF_BACK,
            qmd: QMD_BACK,
            push: PUSH_BACK,
            sem: 2,
        };
        launch_at(&mut cp, &at, back.grid(), lb::BLOCK, lb::SHARED).unwrap();
        assert_eq!(cp.mem.read_u32(SEM).unwrap(), 2);
        let dx = cp.mem.read_f32s(DX, rows * cc).unwrap();
        let (odx, _, _) = train::layernorm_back_rows(&k.x, &dy, &k.gamma, rows, cc).unwrap();
        let mut worst = 0.0f64;
        for rr in 0..rows {
            let s = rr * cc..(rr + 1) * cc;
            let big = odx[s.clone()]
                .iter()
                .fold(0.0f64, |m, &v| m.max((v as f64).abs()));
            let d = s
                .map(|j| (dx[j] as f64 - odx[j] as f64).abs())
                .fold(0.0f64, f64::max);
            assert!(
                d <= big / 65536.0,
                "chain c={cc} row {rr}: max |Δdx| {d:e} > 2^-16 · {big:e}"
            );
            worst = worst.max(d / big);
        }
        eprintln!(
            "T157 chain rows={rows} c={cc} offset={offset}: max |Δdx| / max |dx| = {worst:.3e}"
        );
    }
}

#[test]
fn t157_a_short_grid_leaves_the_other_rows_untouched() {
    let k = synthetic(9, 4, 300, 0.0, 1.0);
    let o = run_grid(&k, 2);
    assert!(o.y[..2 * 300].iter().all(|&v| v != SENTINEL));
    assert!(o.y[2 * 300..].iter().all(|&v| v == SENTINEL));
    assert!(o.mean[2..].iter().all(|&v| v == SENTINEL));
    assert!(o.rstd[2..].iter().all(|&v| v == SENTINEL));
}

#[test]
fn t157_refuses_a_wrong_block_or_a_qmd_without_shared_memory() {
    let k = synthetic(5, 1, 256, 0.0, 1.0);
    let mut cp = cp_with(ln_fwd_on_cpu);
    load(&mut cp, &k);
    assert!(launch(&mut cp, [1, 1, 1], [32, 1, 1], lf::SHARED).is_err());
    let few = SharedMemory {
        bytes: 0,
        ..lf::SHARED
    };
    assert!(launch(&mut cp, [1, 1, 1], lf::BLOCK, few).is_err());
    let no_bar = SharedMemory {
        barriers: 0,
        ..lf::SHARED
    };
    assert!(launch(&mut cp, [1, 1, 1], lf::BLOCK, no_bar).is_err());
    let y = cp.mem.read_f32s(Y, 256).unwrap();
    assert!(
        y.iter().all(|&v| v == SENTINEL),
        "refused launches wrote nothing"
    );
    assert_eq!(cp.mem.read_f32s(MEAN, 1).unwrap()[0], SENTINEL);
}

#[test]
fn t157_planted_mistakes_are_caught() {
    let k = synthetic(0xbad, 2, 256, 0.3, 1.0);
    // (1) Another program's model registered for this code (xent: the same block, other arguments).
    let mut cp = cp_with(xent_on_cpu);
    load(&mut cp, &k);
    let caught = match launch(&mut cp, [2, 1, 1], lf::BLOCK, lf::SHARED) {
        Err(_) => true,
        Ok(()) => {
            let o = Out {
                y: cp.mem.read_f32s(Y, 512).unwrap(),
                mean: cp.mem.read_f32s(MEAN, 2).unwrap(),
                rstd: cp.mem.read_f32s(RSTD, 2).unwrap(),
            };
            check(&k, &o).is_err()
        }
    };
    assert!(caught);
    let good = run(&k);
    // (2) The statistics of the other row.
    let swapped = Out {
        y: good.y.clone(),
        mean: vec![good.mean[1], good.mean[0]],
        rstd: vec![good.rstd[1], good.rstd[0]],
    };
    assert!(check(&k, &swapped).is_err(), "μ / rstd of the wrong row");
    // (3) The unbiased variance (÷ (c − 1) instead of ÷ c: PyTorch's LayerNorm uses ÷ c).
    let mut unbiased = Out {
        y: Vec::new(),
        mean: good.mean.clone(),
        rstd: Vec::new(),
    };
    for r in 0..2 {
        let row = &k.x[r * 256..(r + 1) * 256];
        let mu = row.iter().map(|&v| v as f64).sum::<f64>() / 256.0;
        let var = row.iter().map(|&v| (v as f64 - mu).powi(2)).sum::<f64>() / 255.0;
        let rs = 1.0 / (var + EPS as f64).sqrt();
        unbiased.rstd.push(rs as f32);
        unbiased.y.extend(
            row.iter()
                .zip(&k.gamma)
                .zip(&k.beta)
                .map(|((&x, &g), &b)| ((x as f64 - mu) * rs * g as f64 + b as f64) as f32),
        );
    }
    assert!(
        check(&k, &unbiased).is_err(),
        "÷ (c − 1) must fall outside the margin"
    );
    // (4) ε forgotten, on a row whose variance is close to ε.
    let small = synthetic(0x5a11, 1, 768, 0.0, 0.003);
    let gs = run(&small);
    let row = &small.x;
    let mu = row.iter().map(|&v| v as f64).sum::<f64>() / 768.0;
    let var = row.iter().map(|&v| (v as f64 - mu).powi(2)).sum::<f64>() / 768.0;
    let no_eps = Out {
        y: gs.y.clone(),
        mean: gs.mean.clone(),
        rstd: vec![(1.0 / var.sqrt()) as f32],
    };
    assert!(check(&small, &no_eps).is_err(), "rstd without ε");
    // (5) β forgotten.
    let no_beta = Out {
        y: good
            .y
            .iter()
            .enumerate()
            .map(|(i, &v)| v - k.beta[i % 256])
            .collect(),
        mean: good.mean.clone(),
        rstd: good.rstd.clone(),
    };
    assert!(check(&k, &no_beta).is_err(), "y without β");
}
