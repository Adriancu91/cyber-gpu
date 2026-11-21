//! C-160 point 4 — T154, SIMULARE: training program 4, `dx` half (`tools/nak/ln_back.comp`, compiled by Mesa NAK to
//! `tools/nak/out/ln_back.sm86.sass`): LayerNorm backward over rows of C numbers, one block of 256 threads per row,
//! launched through our pushbuffer + QMD in the command processor model. The "program" runs on the CPU
//! (`ln_back_on_cpu` → `gpu_compute::layer::ln_back_gpu256`, in the order of the code NAK made). The results are
//! judged against the f64 training oracle `gpu_compute::train::layernorm_back_rows` (the very calculation
//! `loss_and_grad` does for LN1 / LN2 / the final LN, which gave PyTorch's numbers at the small golden step, T151)
//! within a margin written BEFORE the measurement (`bound_dx` below). Rows: C = 1 … 2048 (64 = the small model,
//! 768 = #513), rows with a large mean, γ of the small golden model. Nothing here runs on the RTX 3070 Ti.

use std::path::{Path, PathBuf};

use gpu_compute::train::{self, MIC};
use gpu_sim::cmdproc::{CommandProcessor, Event, LN_BACK_NEEDS, ln_back_on_cpu, xent_on_cpu};
use nvidia_ga104::ln_back::{self as lb, LN_BACK};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
const GAMMA: u64 = 0x40_0000;
const MEAN: u64 = 0x48_0000;
const RSTD: u64 = 0x4c_0000;
const X: u64 = 0x100_0000;
const DY: u64 = 0x400_0000;
const DX: u64 = 0x800_0000;
const SENTINEL: f32 = -12345.0;
/// 2⁻²⁴: one rounding of f32.
const U: f64 = 1.0 / 16_777_216.0;
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
    cp.register_program_with_needs(LN_BACK, run, LN_BACK_NEEDS);
    cp.mem.write(PROG, LN_BACK);
    cp
}

fn args(rows: usize, cc: usize) -> lb::Args {
    lb::Args {
        x: X,
        dy: DY,
        gamma: GAMMA,
        mean: MEAN,
        rstd: RSTD,
        dx: DX,
        rows: rows as u32,
        c: cc as u32,
    }
}

struct Case {
    x: Vec<f32>,
    dy: Vec<f32>,
    gamma: Vec<f32>,
    rows: usize,
    c: usize,
}

fn load(cp: &mut CommandProcessor, k: &Case, mean: &[f32], rstd: &[f32]) {
    let mut cb = vec![0u8; 0x100];
    cb[..lb::CBUF_BYTES].copy_from_slice(&args(k.rows, k.c).cbuf().unwrap());
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(X, &k.x);
    cp.mem.write_f32s(DY, &k.dy);
    cp.mem.write_f32s(GAMMA, &k.gamma);
    cp.mem.write_f32s(MEAN, mean);
    cp.mem.write_f32s(RSTD, rstd);
    cp.mem.write_f32s(DX, &vec![SENTINEL; k.rows * k.c + 16]);
    cp.mem.write_u32(SEM, 0);
}

fn launch(
    cp: &mut CommandProcessor,
    grid: [u32; 3],
    block: [u32; 3],
    shared: SharedMemory,
) -> Result<(), gpu_core::error::GpuError> {
    let q = ComputeQmd {
        program_va: PROG,
        registers: lb::REGISTERS,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid,
        block,
        release: Some((SEM, 1)),
    };
    cp.mem.write(QMD, &q.to_bytes_with_shared(&shared).unwrap());
    let mut w = [0u32; 16];
    let mut p = PushBuf::new(&mut w);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = p.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

/// Runs the program with the given per-row statistics; returns dx and checks nothing was written after it.
fn run_with(k: &Case, mean: &[f32], rstd: &[f32]) -> Vec<f32> {
    let mut cp = cp_with(ln_back_on_cpu);
    load(&mut cp, k, mean, rstd);
    let grid = args(k.rows, k.c).grid();
    assert_eq!(grid, [k.rows as u32, 1, 1]);
    launch(&mut cp, grid, lb::BLOCK, lb::SHARED).unwrap();
    assert!(matches!(cp.events[1], Event::Launch { program: 0, .. }));
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let dx = cp.mem.read_f32s(DX, k.rows * k.c + 16).unwrap();
    assert!(
        dx[k.rows * k.c..].iter().all(|&v| v == SENTINEL),
        "wrote after dx"
    );
    dx[..k.rows * k.c].to_vec()
}

/// Runs the program with μ and rstd as the forward pass keeps them (`train::layernorm_stats`).
fn run(k: &Case) -> Vec<f32> {
    let (mean, rstd) = train::layernorm_stats(&k.x, k.rows, k.c).unwrap();
    run_with(k, &mean, &rstd)
}

// ------------------------------------------------------------------ the margin, written before the measurement
//
// For one row (u = 2⁻²⁴; μ, rs = 1/√(var+eps), ĥₖ, gₖ = dyₖ·γₖ, S1 = Σg, S2 = Σ g·ĥ, m1 = S1/c, m2 = S2/c,
// tₖ = gₖ − m1 − ĥₖ·m2, dxₖ = rs·tₖ all exact, in f64), P = ⌈c/256⌉ + 12 roundings on the path of a sum (the thread's
// adds, 5 butterfly, 7 warp adds):
//   ĥₖ on the board: μ and rs rounded to f32 (u·|μ| in x − μ, u in rs), the subtraction and the product (2u), and the
//     oracle's own ĥ kept in f32 (u)                              → eĥₖ = rs·u·|μ| + 5u·|ĥₖ|;
//   gₖ: one rounding                                               → u·|gₖ|;
//   S1: (P + 1)·u·Σ|g|;  S2: Σ|gₖ|·eĥₖ + (P + 2)·u·Σ|gₖ·ĥₖ| (the fma chain);
//   m1, m2: the sum's error / c, plus 1/c rounded and the product → + 2u·|m|;
//   tₖ: Δg + Δm1 + |m2|·eĥ + |ĥ|·Δm2 + 2 roundings (2u·(|g − m1| + |t|));
//   dxₖ = rs·t: rs·Δt + rs rounded and the product (2u·|dx|) + the oracle's rounding to f32 (u·|dx|)
//     + FTZ: everything under 2⁻¹²⁶ is flushed (rs + 2)·2⁻¹²⁶ absolute.

fn bound_dx(x: &[f32], dy: &[f32], gamma: &[f32]) -> Vec<f64> {
    let c = x.len();
    let cf = c as f64;
    let mu = x.iter().map(|&v| v as f64).sum::<f64>() / cf;
    let var = x
        .iter()
        .map(|&v| (v as f64 - mu) * (v as f64 - mu))
        .sum::<f64>()
        / cf;
    let rs = 1.0 / (var + train::LN_EPS).sqrt();
    let h: Vec<f64> = x.iter().map(|&v| (v as f64 - mu) * rs).collect();
    let g: Vec<f64> = dy
        .iter()
        .zip(gamma)
        .map(|(&a, &b)| a as f64 * b as f64)
        .collect();
    let eh: Vec<f64> = h
        .iter()
        .map(|&hk| rs * U * mu.abs() + 5.0 * U * hk.abs())
        .collect();
    let p = c.div_ceil(256) as f64 + 12.0;
    let a1: f64 = g.iter().map(|v| v.abs()).sum();
    let a2: f64 = g.iter().zip(&h).map(|(a, b)| (a * b).abs()).sum();
    let s1: f64 = g.iter().sum();
    let s2: f64 = g.iter().zip(&h).map(|(a, b)| a * b).sum();
    let (m1, m2) = (s1 / cf, s2 / cf);
    let ds1 = (p + 1.0) * U * a1;
    let ds2: f64 = g.iter().zip(&eh).map(|(a, e)| a.abs() * e).sum::<f64>() + (p + 2.0) * U * a2;
    let dm1 = ds1 / cf + 2.0 * U * m1.abs();
    let dm2 = ds2 / cf + 2.0 * U * m2.abs();
    (0..c)
        .map(|k| {
            let t = g[k] - m1 - h[k] * m2;
            let dt = U * g[k].abs()
                + dm1
                + m2.abs() * eh[k]
                + h[k].abs() * dm2
                + 2.0 * U * ((g[k] - m1).abs() + t.abs());
            let dx = rs * t;
            rs * dt + 3.0 * U * dx.abs() + (rs + 2.0) * FTZ
        })
        .collect()
}

struct Worst {
    abs: f64,
    ratio: f64,
}

/// Judges the program's dx against `train::layernorm_back_rows`; panics outside the margin.
fn judge(what: &str, k: &Case, dx: &[f32]) -> Worst {
    let (or_dx, _, _) = train::layernorm_back_rows(&k.x, &k.dy, &k.gamma, k.rows, k.c).unwrap();
    let mut w = Worst {
        abs: 0.0,
        ratio: 0.0,
    };
    for r in 0..k.rows {
        let s = r * k.c..(r + 1) * k.c;
        let b = bound_dx(&k.x[s.clone()], &k.dy[s.clone()], &k.gamma);
        for (j, i) in s.enumerate() {
            let diff = (dx[i] as f64 - or_dx[i] as f64).abs();
            assert!(
                diff <= b[j],
                "{what} dx[{r},{j}]: |Δ| {diff:e} > {:e}",
                b[j]
            );
            w.abs = w.abs.max(diff);
            w.ratio = w.ratio.max(diff / b[j]);
        }
    }
    eprintln!(
        "T154 {what}: dx max |Δ| = {:.3e} ({:.4} of the margin)",
        w.abs, w.ratio
    );
    w
}

fn synthetic(seed: u64, rows: usize, cc: usize, offset: f32, spread: f32) -> Case {
    let mut r = rng(seed);
    Case {
        x: (0..rows * cc).map(|_| offset + r() * spread).collect(),
        dy: (0..rows * cc).map(|_| r() * 0.01).collect(),
        gamma: (0..cc).map(|_| 1.0 + r() * 0.2).collect(),
        rows,
        c: cc,
    }
}

#[test]
fn t154_binary_is_the_recorded_build() {
    let hex: String = nvidia_ga104::sha256::digest(LN_BACK)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, lb::LN_BACK_SHA256);
    assert_eq!(LN_BACK.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out/ln_back.sm86.txt");
    assert!(txt.contains("bar"), "no barrier in the dump");
    assert!(txt.contains("shfl"), "warp butterflies expected");
    assert!(
        !txt.contains("mufu"),
        "no MUFU expected: only add / mul / fma"
    );
    let ok = args(4, 768);
    assert!(ok.cbuf().is_some());
    assert!(lb::Args { rows: 0, ..ok }.cbuf().is_none());
    assert!(lb::Args { c: 0, ..ok }.cbuf().is_none());
    assert!(lb::Args { rstd: 0, ..ok }.cbuf().is_none());
    let cb = ok.cbuf().unwrap();
    assert_eq!(u32::from_le_bytes(cb[0x58..0x5c].try_into().unwrap()), 768);
    assert_eq!(
        f32::from_le_bytes(cb[0x5c..0x60].try_into().unwrap()),
        1.0 / 768.0
    );
    assert_eq!(u64::from_le_bytes(cb[0x50..0x58].try_into().unwrap()), DX);
}

#[test]
fn t154_synthetic_rows_are_within_the_written_margin() {
    for (i, (rows, cc, offset, spread)) in [
        (3usize, 1usize, 0.0f32, 1.0f32),
        (4, 64, 0.0, 1.0),
        (3, 255, 0.5, 2.0),
        (3, 256, 0.0, 1.0),
        (3, 257, -1.0, 3.0),
        (4, 768, 0.0, 1.0), // #513's width
        (2, 1000, 0.0, 0.05),
        (2, 2048, 0.0, 1.0),
        (3, 768, 100.0, 1.0), // a large mean: μ rounded to f32 counts
    ]
    .into_iter()
    .enumerate()
    {
        let k = synthetic(0x1b_0000 + i as u64, rows, cc, offset, spread);
        let dx = run(&k);
        judge(
            &format!("rows={rows} c={cc} offset={offset} spread={spread}"),
            &k,
            &dx,
        );
        // LayerNorm's backward is orthogonal to 1 and to x̂: Σ dx ≈ 0 on every row (c > 1).
        if cc > 1 {
            for r in 0..rows {
                let s: f64 = dx[r * cc..(r + 1) * cc].iter().map(|&v| v as f64).sum();
                let a: f64 = dx[r * cc..(r + 1) * cc]
                    .iter()
                    .map(|&v| (v as f64).abs())
                    .sum();
                assert!(
                    s.abs() <= 1e-4 * a + 1e-12,
                    "c={cc} row {r}: Σdx = {s:e} (Σ|dx| {a:e})"
                );
            }
        }
    }
}

#[test]
fn t154_runs_are_the_same_bits() {
    let k = synthetic(77, 5, 768, 0.0, 1.0);
    let a = run(&k);
    let b = run(&k);
    assert!(a.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
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
fn t154_the_small_golden_model_gammas() {
    // Every LayerNorm γ of the small golden model (after 10 steps of `sgd`, PyTorch's weights, so not all ones), on the golden batch's
    // shape (64 rows × 64) with x and dy of realistic size.
    let mut seen = 0;
    for t in MIC.tensors() {
        if !(t.name.contains("ln") && t.name.ends_with("weight")) || t.elements() != MIC.n_emb {
            continue;
        }
        let p = dir()
            .join("sgd")
            .join("dupa_pas10")
            .join(format!("{}.f32", t.name));
        let gamma = read_f32(&p, t.elements());
        let mut k = synthetic(0x60 + seen, 64, MIC.n_emb, 0.0, 1.5);
        k.gamma = gamma;
        let dx = run(&k);
        judge(&t.name, &k, &dx);
        seen += 1;
    }
    assert!(
        seen >= 3,
        "the small model has LN1, LN2 per layer and the final LN (found {seen})"
    );
}

#[test]
fn t154_refuses_a_wrong_block_or_a_qmd_without_shared_memory() {
    let k = synthetic(5, 1, 256, 0.0, 1.0);
    let (mean, rstd) = train::layernorm_stats(&k.x, 1, 256).unwrap();
    let mut cp = cp_with(ln_back_on_cpu);
    load(&mut cp, &k, &mean, &rstd);
    assert!(launch(&mut cp, [1, 1, 1], [32, 1, 1], lb::SHARED).is_err());
    let few = SharedMemory {
        bytes: 0,
        ..lb::SHARED
    };
    assert!(launch(&mut cp, [1, 1, 1], lb::BLOCK, few).is_err());
    let no_bar = SharedMemory {
        barriers: 0,
        ..lb::SHARED
    };
    assert!(launch(&mut cp, [1, 1, 1], lb::BLOCK, no_bar).is_err());
    assert!(
        cp.mem
            .read_f32s(DX, 256)
            .unwrap()
            .iter()
            .all(|&x| x == SENTINEL),
        "refused launches wrote nothing"
    );
}

fn outside(k: &Case, dx: &[f32]) -> bool {
    let (or_dx, _, _) = train::layernorm_back_rows(&k.x, &k.dy, &k.gamma, k.rows, k.c).unwrap();
    (0..k.rows).any(|r| {
        let s = r * k.c..(r + 1) * k.c;
        let b = bound_dx(&k.x[s.clone()], &k.dy[s.clone()], &k.gamma);
        s.enumerate()
            .any(|(j, i)| (dx[i] as f64 - or_dx[i] as f64).abs() > b[j])
    })
}

#[test]
fn t154_planted_mistakes_are_caught() {
    let k = synthetic(0xbad, 2, 256, 0.3, 1.0);
    // (1) Another program's model registered for this code (xent: the same block, other arguments).
    let (mean, rstd) = train::layernorm_stats(&k.x, 2, 256).unwrap();
    let mut cp = cp_with(xent_on_cpu);
    load(&mut cp, &k, &mean, &rstd);
    let caught = match launch(&mut cp, [2, 1, 1], lb::BLOCK, lb::SHARED) {
        Err(_) => true,
        Ok(()) => outside(&k, &cp.mem.read_f32s(DX, 512).unwrap()),
    };
    assert!(caught);
    // (2) The statistics of the other row (a classic indexing bug).
    let swapped = run_with(&k, &[mean[1], mean[0]], &[rstd[1], rstd[0]]);
    assert!(
        outside(&k, &swapped),
        "μ / rstd of the wrong row must fall outside the margin"
    );
    // (3) Forgetting the x̂·m2 term (the backward of the variance).
    let mut no_m2 = Vec::new();
    for r in 0..2 {
        let s = r * 256..(r + 1) * 256;
        let (mu, rs) = (mean[r], rstd[r]);
        let g: Vec<f32> = k.dy[s.clone()]
            .iter()
            .zip(&k.gamma)
            .map(|(a, b)| a * b)
            .collect();
        let m1 = g.iter().sum::<f32>() / 256.0;
        no_m2.extend(g.iter().map(|&gk| rs * (gk - m1)));
        let _ = mu;
    }
    assert!(
        outside(&k, &no_m2),
        "dx without the variance term must fall outside the margin"
    );
    // (4) rstd forgotten in the final product (dx = t instead of rs·t).
    let mut no_rs = run(&k);
    for r in 0..2 {
        for v in &mut no_rs[r * 256..(r + 1) * 256] {
            *v /= rstd[r];
        }
    }
    assert!(outside(&k, &no_rs));
}
