//! C-160 point 4 — T153, SIMULARE: training program 7 (`tools/nak/xent.comp`, compiled by Mesa NAK to
//! `tools/nak/out/xent.sm86.sass`): softmax + cross-entropy and its gradient, one block of 256 threads per row,
//! launched through our pushbuffer + QMD in the command processor model. The "program" runs on the CPU
//! (`xent_on_cpu` → `gpu_compute::layer::xent_gpu256`, in the kernel's order of the sums). `exp`, `ln` and `1/s` are
//! approximate `MUFU` units on the board, so nothing here is bit for bit: the program's results are judged against
//! the f64 training oracle `gpu_compute::train::softmax_xent` (the part of `loss_and_grad` that gave the same numbers
//! as PyTorch at the small golden step, T151) within a margin written BEFORE the measurement (`bound_d`, `bound_loss`
//! below). The rows: synthetic (v = 1, 255, 256, 257, 1000, 16 384 = #513's vocabulary, large spreads) and the real
//! logits of the small golden model for its golden batch. Nothing here runs on the RTX 3070 Ti.

use std::path::{Path, PathBuf};

use gpu_compute::train::{self, Batch, MIC};
use gpu_sim::cmdproc::{CommandProcessor, Event, XENT_NEEDS, layernorm256_on_cpu, xent_on_cpu};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_ga104::xent::{self as xe, XENT};
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
const TGT: u64 = 0x40_0000;
const LOSS: u64 = 0x48_0000;
const Z: u64 = 0x100_0000;
const D: u64 = 0x800_0000;
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
    cp.register_program_with_needs(XENT, run, XENT_NEEDS);
    cp.mem.write(PROG, XENT);
    cp
}

fn args(rows: usize, v: usize) -> xe::Args {
    xe::Args {
        logits: Z,
        targets: TGT,
        dlogits: D,
        loss: LOSS,
        rows: rows as u32,
        v: v as u32,
        scale: 1.0 / rows as f32,
    }
}

fn load(cp: &mut CommandProcessor, z: &[f32], targets: &[u32], v: usize) {
    let rows = targets.len();
    let mut cb = vec![0u8; 0x100];
    cb[..xe::CBUF_BYTES].copy_from_slice(&args(rows, v).cbuf().unwrap());
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(Z, z);
    let tb: Vec<u8> = targets.iter().flat_map(|t| t.to_le_bytes()).collect();
    cp.mem.write(TGT, &tb);
    cp.mem.write_f32s(D, &vec![SENTINEL; rows * v + 16]);
    cp.mem.write_f32s(LOSS, &vec![SENTINEL; rows + 16]);
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
        registers: xe::REGISTERS,
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

/// Runs the program on `rows = targets.len()` rows; returns (dlogits, row losses) and checks nothing was written
/// after them.
fn run(z: &[f32], targets: &[u32], v: usize) -> (Vec<f32>, Vec<f32>) {
    let rows = targets.len();
    let mut cp = cp_with(xent_on_cpu);
    load(&mut cp, z, targets, v);
    let grid = args(rows, v).grid();
    assert_eq!(grid, [rows as u32, 1, 1]);
    launch(&mut cp, grid, xe::BLOCK, xe::SHARED).unwrap();
    assert!(matches!(cp.events[1], Event::Launch { program: 0, .. }));
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let d = cp.mem.read_f32s(D, rows * v + 16).unwrap();
    let l = cp.mem.read_f32s(LOSS, rows + 16).unwrap();
    assert!(
        d[rows * v..].iter().all(|&x| x == SENTINEL),
        "wrote after dlogits"
    );
    assert!(l[rows..].iter().all(|&x| x == SENTINEL), "wrote after loss");
    (d[..rows * v].to_vec(), l[..rows].to_vec())
}

// ------------------------------------------------------------------ the margin, written before the measurement
//
// For one row with maximum m, aₖ = |zₖ − m|, A = maxₖ aₖ, the board's error sources (u = 2⁻²⁴):
//   e^(zₖ−m): the subtraction (u·aₖ in the argument), the ·log₂e of GLSL's exp (u·aₖ), MUFU.EX2 (≤ 2⁻²¹ = 8u, a
//             generous figure for its ~2 ulp)                         → δeₖ ≤ 2u·aₖ + 8u (relative);
//   s = Σ e:  ⌈v/256⌉ − 1 adds per thread + 5 butterfly + 7 warp adds  → δs ≤ (⌈v/256⌉ + 12)·u + 2u·A + 8u;
//   1/s:      MUFU.RCP (≤ 2⁻²² = 4u) + 1 rounding                       → 5u;
//   pₖ = e·inv: 1 rounding                                              → δpₖ ≤ δeₖ + δs + 6u;
//   dₖ = (pₖ − yₖ)·scale: 2 roundings on |pₖ − yₖ|, the oracle's own rounding to f32 (1 on |dₖ|):
//     |Δdₖ| ≤ scale·(pₖ·δpₖ + 3u·|pₖ − yₖ|) + 2⁻¹⁴⁹.
//   FTZ (added after the first run: the first margin forgot it): the program flushes every result under 2⁻¹²⁶ to 0
//     (`mufu.exp2`, `fmul.ftz`), the f64 oracle keeps subnormal d → + (scale + 1)·2⁻¹²⁶ absolute on dₖ.
//   loss = ln s + m − z_t: ln of a relative error δs (δs absolute), MUFU.LG2 (≤ 2⁻²¹ absolute, also generous),
//     3 roundings: |Δloss| ≤ δs + 2⁻²¹ + 3u·(|ln s| + |m| + |z_t|).
// pₖ, yₖ, ln s are taken from the f64 oracle.

fn ceil256(v: usize) -> f64 {
    v.div_ceil(256) as f64
}

fn delta_s(v: usize, big_a: f64) -> f64 {
    (ceil256(v) + 12.0) * U + 2.0 * U * big_a + 8.0 * U
}

/// Margin of dₖ for every k of one row.
fn bound_d(z: &[f32], tgt: usize, scale: f64) -> Vec<f64> {
    let m = z.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let big_a = z.iter().map(|&x| (x as f64 - m).abs()).fold(0.0, f64::max);
    let s: f64 = z.iter().map(|&x| (x as f64 - m).exp()).sum();
    let ds = delta_s(z.len(), big_a);
    z.iter()
        .enumerate()
        .map(|(k, &x)| {
            let a = (x as f64 - m).abs();
            let p = (x as f64 - m).exp() / s;
            let y = if k == tgt { 1.0 } else { 0.0 };
            let dp = 2.0 * U * a + 8.0 * U + ds + 6.0 * U;
            scale * (p * dp + 3.0 * U * (p - y).abs()) + (scale + 1.0) * FTZ
        })
        .collect()
}

fn bound_loss(z: &[f32], tgt: usize) -> f64 {
    let m = z.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b)) as f64;
    let big_a = z.iter().map(|&x| (x as f64 - m).abs()).fold(0.0, f64::max);
    let s: f64 = z.iter().map(|&x| (x as f64 - m).exp()).sum();
    delta_s(z.len(), big_a) + 8.0 * U + 3.0 * U * (s.ln().abs() + m.abs() + (z[tgt] as f64).abs())
}

struct Worst {
    d_abs: f64,
    d_ratio: f64,
    loss_abs: f64,
    loss_ratio: f64,
}

/// Judges the program's rows against `train::softmax_xent`; panics outside the margin.
fn judge(what: &str, z: &[f32], targets: &[u32], v: usize, d: &[f32], loss: &[f32]) -> Worst {
    let rows = targets.len();
    let (or_loss, or_d) = train::softmax_xent(z, targets, v).unwrap();
    let scale = 1.0 / rows as f64;
    let mut w = Worst {
        d_abs: 0.0,
        d_ratio: 0.0,
        loss_abs: 0.0,
        loss_ratio: 0.0,
    };
    for r in 0..rows {
        let zr = &z[r * v..(r + 1) * v];
        let t = targets[r] as usize;
        let bd = bound_d(zr, t, scale);
        for k in 0..v {
            let diff = (d[r * v + k] as f64 - or_d[r * v + k] as f64).abs();
            assert!(
                diff <= bd[k],
                "{what} d[{r},{k}]: |Δ| {diff:e} > {:e}",
                bd[k]
            );
            w.d_abs = w.d_abs.max(diff);
            w.d_ratio = w.d_ratio.max(diff / bd[k]);
        }
        let bl = bound_loss(zr, t);
        let diff = (loss[r] as f64 - or_loss[r]).abs();
        assert!(diff <= bl, "{what} loss[{r}]: |Δ| {diff:e} > {bl:e}");
        w.loss_abs = w.loss_abs.max(diff);
        w.loss_ratio = w.loss_ratio.max(diff / bl);
    }
    eprintln!(
        "T153 {what}: d max |Δ| = {:.3e} ({:.4} of the margin), loss max |Δ| = {:.3e} ({:.4} of the margin)",
        w.d_abs, w.d_ratio, w.loss_abs, w.loss_ratio
    );
    w
}

#[test]
fn t153_binary_is_the_recorded_build() {
    let hex: String = nvidia_ga104::sha256::digest(XENT)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, xe::XENT_SHA256);
    assert_eq!(XENT.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out/xent.sm86.txt");
    assert!(txt.contains("bar"), "no barrier in the dump");
    assert!(txt.contains("mufu.exp2"), "exp() expected as MUFU.EX2");
    assert!(txt.contains("mufu.log2"), "log() expected as MUFU.LG2");
    assert!(txt.contains("mufu.rcp"), "1/s expected as MUFU.RCP");
    assert!(txt.contains("shfl"), "warp butterflies expected");
    let ok = args(4, 256);
    assert!(ok.cbuf().is_some());
    assert!(xe::Args { rows: 0, ..ok }.cbuf().is_none());
    assert!(xe::Args { v: 0, ..ok }.cbuf().is_none());
    assert!(xe::Args { loss: 0, ..ok }.cbuf().is_none());
    assert!(
        xe::Args {
            scale: f32::NAN,
            ..ok
        }
        .cbuf()
        .is_none()
    );
    let cb = ok.cbuf().unwrap();
    assert_eq!(u32::from_le_bytes(cb[0x48..0x4c].try_into().unwrap()), 256);
    assert_eq!(f32::from_le_bytes(cb[0x4c..0x50].try_into().unwrap()), 0.25);
}

#[test]
fn t153_synthetic_rows_are_within_the_written_margin() {
    let mut r = rng(0x7e57_c0de);
    for (v, rows, spread) in [
        (1usize, 3usize, 1.0f32),
        (255, 5, 4.0),
        (256, 4, 4.0),
        (257, 4, 8.0),
        (1000, 6, 20.0),
        (16_384, 2, 12.0),
        (300, 3, 200.0), // most e^(z−m) underflow to 0: only a few terms count
    ] {
        let z: Vec<f32> = (0..rows * v).map(|_| r() * spread).collect();
        let targets: Vec<u32> = (0..rows).map(|i| ((i * 7919 + 13) % v) as u32).collect();
        let (d, loss) = run(&z, &targets, v);
        judge(
            &format!("v={v} rows={rows} spread={spread}"),
            &z,
            &targets,
            v,
            &d,
            &loss,
        );
        // Every row's gradient sums to ~0 (softmax sums to 1, minus the one-hot).
        for row in 0..rows {
            let sum: f64 = d[row * v..(row + 1) * v].iter().map(|&x| x as f64).sum();
            assert!(sum.abs() < 1e-5, "v={v} row {row}: Σd = {sum:e}");
        }
    }
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

fn read_i32(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::try_from(i32::from_le_bytes(*c)).unwrap())
        .collect()
}

#[test]
fn t153_the_small_golden_model_loss_and_gradient() {
    let params: Vec<Vec<f32>> = MIC
        .tensors()
        .iter()
        .map(|t| {
            read_f32(
                &dir().join("initiale").join(format!("{}.f32", t.name)),
                t.elements(),
            )
        })
        .collect();
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: 2,
        len: 32,
    };
    let v = MIC.vocab;
    let z = train::logits(&MIC, &params, &batch).unwrap();
    assert_eq!(z.len(), 64 * v);
    let (d, loss) = run(&z, &targets, v);
    judge("pas_de_aur_mic", &z, &targets, v, &d, &loss);
    // The mean of the rows against the oracle's step-1 loss (the number PyTorch gave, T151).
    let (want, _) = train::loss_and_grad(&MIC, &params, &batch).unwrap();
    let mean = loss.iter().map(|&x| x as f64).sum::<f64>() / loss.len() as f64;
    let margin: f64 = (0..64)
        .map(|r| bound_loss(&z[r * v..(r + 1) * v], targets[r] as usize))
        .sum::<f64>()
        / 64.0
        + 64.0 * U * mean.abs();
    eprintln!(
        "T153 mean loss: {mean:.9} vs loss_and_grad {want:.9}, |Δ| = {:.3e} (margin {margin:.3e})",
        (mean - want as f64).abs()
    );
    assert!((mean - want as f64).abs() <= margin);
}

#[test]
fn t153_a_target_outside_the_row_is_visible_and_reads_nothing_outside() {
    let v = 300usize;
    let mut r = rng(3);
    let z: Vec<f32> = (0..2 * v).map(|_| r()).collect();
    let (d, loss) = run(&z, &[5, v as u32], v);
    assert!(loss[0].is_finite());
    assert_eq!(loss[1].to_bits(), 0x7fc0_0000, "NaN loss for a bad target");
    // No one-hot was subtracted on row 1: every d is ≥ 0 (pure softmax · scale).
    assert!(d[v..].iter().all(|&x| x >= 0.0));
    assert!(d[..v].iter().any(|&x| x < 0.0), "row 0 has its one-hot");
}

#[test]
fn t153_refuses_a_wrong_block_or_a_qmd_without_shared_memory() {
    let v = 256usize;
    let z = vec![0.5f32; v];
    let mut cp = cp_with(xent_on_cpu);
    load(&mut cp, &z, &[1], v);
    assert!(launch(&mut cp, [1, 1, 1], [32, 1, 1], xe::SHARED).is_err());
    let few = SharedMemory {
        bytes: 0,
        ..xe::SHARED
    };
    assert!(launch(&mut cp, [1, 1, 1], xe::BLOCK, few).is_err());
    let no_bar = SharedMemory {
        barriers: 0,
        ..xe::SHARED
    };
    assert!(launch(&mut cp, [1, 1, 1], xe::BLOCK, no_bar).is_err());
    assert!(
        cp.mem
            .read_f32s(D, v)
            .unwrap()
            .iter()
            .all(|&x| x == SENTINEL),
        "refused launches wrote nothing"
    );
    assert_eq!(cp.mem.read_f32s(LOSS, 1).unwrap()[0], SENTINEL);
}

#[test]
fn t153_planted_mistakes_are_caught() {
    // (1) Another program's model registered for this code (layernorm256: the same block, other arguments).
    let v = 256usize;
    let mut r = rng(0xbad);
    let z: Vec<f32> = (0..v).map(|_| r() * 3.0).collect();
    let mut cp = cp_with(layernorm256_on_cpu);
    load(&mut cp, &z, &[17], v);
    let caught = match launch(&mut cp, [1, 1, 1], xe::BLOCK, xe::SHARED) {
        Err(_) => true,
        Ok(()) => {
            let d = cp.mem.read_f32s(D, v).unwrap();
            let (_, or_d) = train::softmax_xent(&z, &[17], v).unwrap();
            d.iter().zip(&or_d).any(|(a, b)| (a - b).abs() > 1e-4)
        }
    };
    assert!(caught);
    // (2) The gradient without the one-hot (a classic bug) falls outside the margin.
    let (mut d, _) = run(&z, &[17], v);
    d[17] += 1.0;
    let (_, or_d) = train::softmax_xent(&z, &[17], v).unwrap();
    let bd = bound_d(&z, 17, 1.0);
    assert!(
        (0..v).any(|k| (d[k] as f64 - or_d[k] as f64).abs() > bd[k]),
        "a gradient without the one-hot must fall outside the margin"
    );
    // (3) The softmax without subtracting the maximum overflows for large logits: the oracle stays finite, a naive
    // e^z would not. Our rows with z ≈ 100 stay finite and inside the margin.
    let big: Vec<f32> = (0..v).map(|k| 90.0 + (k % 7) as f32).collect();
    let (d, loss) = run(&big, &[3], v);
    assert!(d.iter().all(|x| x.is_finite()) && loss[0].is_finite());
    judge("z ≈ 100", &big, &[3], v, &d, &loss);
}
