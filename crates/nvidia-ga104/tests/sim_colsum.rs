//! C-160 point 4 — T155, SIMULARE: training program 2 + the dγ/dβ half of program 4 (`tools/nak/colsum.comp`,
//! compiled by Mesa NAK to `tools/nak/out/colsum.sm86.sass`): column sums over the rows of a gradient, one thread per
//! column, launched through our pushbuffer + QMD in the command processor model. The "program" runs on the CPU
//! (`colsum_on_cpu` → `gpu_compute::layer::colsum_gpu`, in the order of the code NAK made). Judged against the f64
//! training oracle `gpu_compute::train::layernorm_back_rows` (dγ, dβ — the calculation `loss_and_grad` does, which
//! gave PyTorch's numbers at the small golden step, T151; dβ = Σ dy is also the bias gradient) within a margin
//! written BEFORE the measurement (`bound_*` below). Nothing here runs on the RTX 3070 Ti.

use gpu_compute::train;
use gpu_sim::cmdproc::{CommandProcessor, Event, LN_BACK_NEEDS, colsum_on_cpu, ln_back_on_cpu};
use nvidia_ga104::colsum::{self as cs, COLSUM, Mode};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
const MEAN: u64 = 0x40_0000;
const RSTD: u64 = 0x48_0000;
const S0: u64 = 0x50_0000;
const S1: u64 = 0x58_0000;
const X: u64 = 0x100_0000;
const DY: u64 = 0x800_0000;
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

struct Case {
    x: Vec<f32>,
    dy: Vec<f32>,
    rows: usize,
    c: usize,
}

fn synthetic(seed: u64, rows: usize, cc: usize, offset: f32) -> Case {
    let mut r = rng(seed);
    Case {
        x: (0..rows * cc).map(|_| offset + r() * 1.5).collect(),
        dy: (0..rows * cc).map(|_| r() * 0.01).collect(),
        rows,
        c: cc,
    }
}

fn args(k: &Case, mode: Mode) -> cs::Args {
    cs::Args {
        dy: DY,
        x: X,
        mean: MEAN,
        rstd: RSTD,
        s0: S0,
        s1: S1,
        rows: k.rows as u32,
        c: k.c as u32,
        mode,
    }
}

fn cp_with(run: gpu_sim::cmdproc::ProgramFn) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program(COLSUM, run);
    cp.mem.write(PROG, COLSUM);
    cp
}

fn load(cp: &mut CommandProcessor, k: &Case, mode: Mode) {
    let (mean, rstd) = train::layernorm_stats(&k.x, k.rows, k.c).unwrap();
    let mut cb = vec![0u8; 0x100];
    cb[..cs::CBUF_BYTES].copy_from_slice(&args(k, mode).cbuf().unwrap());
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(X, &k.x);
    cp.mem.write_f32s(DY, &k.dy);
    cp.mem.write_f32s(MEAN, &mean);
    cp.mem.write_f32s(RSTD, &rstd);
    cp.mem.write_f32s(S0, &vec![SENTINEL; k.c + 16]);
    cp.mem.write_f32s(S1, &vec![SENTINEL; k.c + 16]);
    cp.mem.write_u32(SEM, 0);
}

fn launch(
    cp: &mut CommandProcessor,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<(), gpu_core::error::GpuError> {
    let q = ComputeQmd {
        program_va: PROG,
        registers: cs::REGISTERS,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid,
        block,
        release: Some((SEM, 1)),
    };
    cp.mem.write(QMD, &q.to_bytes().unwrap());
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

/// Runs the program; returns (s0, s1) and checks nothing was written after them (s1 untouched in bias mode).
fn run(k: &Case, mode: Mode) -> (Vec<f32>, Vec<f32>) {
    let mut cp = cp_with(colsum_on_cpu);
    load(&mut cp, k, mode);
    let grid = args(k, mode).grid();
    assert_eq!(grid, [k.c.div_ceil(256) as u32, 1, 1]);
    launch(&mut cp, grid, cs::BLOCK).unwrap();
    assert!(matches!(cp.events[1], Event::Launch { program: 0, .. }));
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let s0 = cp.mem.read_f32s(S0, k.c + 16).unwrap();
    let s1 = cp.mem.read_f32s(S1, k.c + 16).unwrap();
    assert!(s0[k.c..].iter().all(|&v| v == SENTINEL), "wrote after s0");
    assert!(s1[k.c..].iter().all(|&v| v == SENTINEL), "wrote after s1");
    if mode == Mode::Bias {
        assert!(s1.iter().all(|&v| v == SENTINEL), "bias mode wrote s1");
    }
    (s0[..k.c].to_vec(), s1[..k.c].to_vec())
}

// ------------------------------------------------------------------ the margin, written before the measurement
//
// Column k, `rows` = R terms added in order from 0.0 (R roundings on the path; u = 2⁻²⁴):
//   dβ = Σ dy:  R·u·Σ|dy| + the oracle's rounding to f32 u·|dβ| + FTZ (R + 1)·2⁻¹²⁶;
//   dγ = Σ dy·x̂ with x̂ recomputed on the board (as in T154: eĥ = rs·u·|μ| + 5u·|ĥ|, which also covers the oracle's
//   x̂ kept in f32): Σ|dy|·eĥ + R·u·Σ|dy·ĥ| (one rounding per fma) + u·|dγ| + FTZ (R + 1)·2⁻¹²⁶.
// μ, rs, ĥ exact in f64.

fn bounds(k: &Case) -> (Vec<f64>, Vec<f64>) {
    let (rows, cc) = (k.rows, k.c);
    let cf = cc as f64;
    let mut stats = Vec::with_capacity(rows);
    for r in 0..rows {
        let row = &k.x[r * cc..(r + 1) * cc];
        let mu = row.iter().map(|&v| v as f64).sum::<f64>() / cf;
        let var = row.iter().map(|&v| (v as f64 - mu).powi(2)).sum::<f64>() / cf;
        stats.push((mu, 1.0 / (var + train::LN_EPS).sqrt()));
    }
    let rf = rows as f64;
    let mut bg = vec![0.0; cc];
    let mut bb = vec![0.0; cc];
    for j in 0..cc {
        let (mut a_d, mut a_dh, mut e, mut sg, mut sb) = (0.0, 0.0, 0.0, 0.0, 0.0);
        for (r, &(mu, rs)) in stats.iter().enumerate() {
            let d = k.dy[r * cc + j] as f64;
            let h = (k.x[r * cc + j] as f64 - mu) * rs;
            a_d += d.abs();
            a_dh += (d * h).abs();
            e += d.abs() * (rs * U * mu.abs() + 5.0 * U * h.abs());
            sg += d * h;
            sb += d;
        }
        bb[j] = rf * U * a_d + U * sb.abs() + (rf + 1.0) * FTZ;
        bg[j] = e + rf * U * a_dh + U * sg.abs() + (rf + 1.0) * FTZ;
    }
    (bg, bb)
}

/// Judges (dγ, dβ) — or the bias sum alone — against `train::layernorm_back_rows`; returns the worst fractions of the
/// margin; panics outside it.
fn judge(what: &str, k: &Case, s0: &[f32], s1: Option<&[f32]>) -> (f64, f64) {
    let gamma = vec![1.0f32; k.c];
    let (_, or_g, or_b) = train::layernorm_back_rows(&k.x, &k.dy, &gamma, k.rows, k.c).unwrap();
    let (bg, bb) = bounds(k);
    let mut worst = (0.0f64, 0.0f64);
    let check = |name: &str, got: &[f32], want: &[f32], b: &[f64]| -> f64 {
        let mut w = 0.0f64;
        for j in 0..k.c {
            let diff = (got[j] as f64 - want[j] as f64).abs();
            assert!(
                diff <= b[j],
                "{what} {name}[{j}]: |Δ| {diff:e} > {:e}",
                b[j]
            );
            w = w.max(diff / b[j]);
        }
        w
    };
    match s1 {
        None => worst.1 = check("bias", s0, &or_b, &bb),
        Some(s1) => {
            worst.0 = check("dγ", s0, &or_g, &bg);
            worst.1 = check("dβ", s1, &or_b, &bb);
        }
    }
    eprintln!(
        "T155 {what}: dγ {:.4} / dβ (or bias) {:.4} of the margin",
        worst.0, worst.1
    );
    worst
}

#[test]
fn t155_binary_is_the_recorded_build() {
    let hex: String = nvidia_ga104::sha256::digest(COLSUM)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, cs::COLSUM_SHA256);
    assert_eq!(COLSUM.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out/colsum.sm86.txt");
    assert!(
        !txt.contains("mufu"),
        "no MUFU expected: only add / mul / fma"
    );
    assert!(!txt.contains("bar."), "no barrier expected");
    let k = synthetic(1, 4, 768, 0.0);
    let ok = args(&k, Mode::LayerNorm);
    assert!(ok.cbuf().is_some());
    assert!(cs::Args { rows: 0, ..ok }.cbuf().is_none());
    assert!(cs::Args { c: 0, ..ok }.cbuf().is_none());
    assert!(cs::Args { s1: 0, ..ok }.cbuf().is_none());
    let bias = cs::Args {
        x: 0,
        mean: 0,
        rstd: 0,
        s1: 0,
        mode: Mode::Bias,
        ..ok
    };
    assert!(bias.cbuf().is_some(), "bias mode needs only dy and s0");
    assert!(cs::Args { s0: 0, ..bias }.cbuf().is_none());
    let cb = ok.cbuf().unwrap();
    assert_eq!(u32::from_le_bytes(cb[0x58..0x5c].try_into().unwrap()), 4);
    assert_eq!(u32::from_le_bytes(cb[0x5c..0x60].try_into().unwrap()), 768);
    assert_eq!(u32::from_le_bytes(cb[0x60..0x64].try_into().unwrap()), 1);
    assert_eq!(
        u32::from_le_bytes(bias.cbuf().unwrap()[0x60..0x64].try_into().unwrap()),
        0
    );
    assert_eq!(ok.grid(), [3, 1, 1]);
}

#[test]
fn t155_column_sums_are_within_the_written_margin() {
    for (i, (rows, cc, offset)) in [
        (1usize, 1usize, 0.0f32),
        (64, 64, 0.0),  // the small golden batch: 2 × 32 rows of 64
        (64, 256, 0.0), // the small model's MLP bias (4·64)
        (3, 257, 0.5),  // a partial last block
        (300, 100, 0.0),
        (512, 768, 0.0),  // #513's width, one text of 512
        (64, 768, 100.0), // a large mean
        (2048, 64, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        let k = synthetic(0x5c_0000 + i as u64, rows, cc, offset);
        let (g, b) = run(&k, Mode::LayerNorm);
        judge(
            &format!("LN rows={rows} c={cc} offset={offset}"),
            &k,
            &g,
            Some(&b),
        );
        let (bias, _) = run(&k, Mode::Bias);
        judge(&format!("bias rows={rows} c={cc}"), &k, &bias, None);
        // The bias sum and dβ are the same additions in the same order: the same bits.
        assert!(bias.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
    }
}

#[test]
fn t155_a_short_grid_leaves_the_last_columns_visible() {
    let k = synthetic(9, 8, 600, 0.0);
    let mut cp = cp_with(colsum_on_cpu);
    load(&mut cp, &k, Mode::LayerNorm);
    launch(&mut cp, [2, 1, 1], cs::BLOCK).unwrap(); // 512 threads for 600 columns: a caller bug
    let s0 = cp.mem.read_f32s(S0, 600).unwrap();
    assert!(s0[..512].iter().all(|&v| v != SENTINEL));
    assert!(
        s0[512..].iter().all(|&v| v == SENTINEL),
        "unreached columns stay visible"
    );
}

#[test]
fn t155_planted_mistakes_are_caught() {
    let k = synthetic(0xbad, 64, 256, 0.3);
    let (bg, bb) = bounds(&k);
    let gamma = vec![1.0f32; k.c];
    let (_, or_g, or_b) = train::layernorm_back_rows(&k.x, &k.dy, &gamma, k.rows, k.c).unwrap();
    let out = |got: &[f32], want: &[f32], b: &[f64]| {
        (0..k.c).any(|j| (got[j] as f64 - want[j] as f64).abs() > b[j])
    };
    // (1) Another program's model registered for this code (ln_back: the same block, other arguments).
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(COLSUM, ln_back_on_cpu, LN_BACK_NEEDS);
    cp.mem.write(PROG, COLSUM);
    load(&mut cp, &k, Mode::LayerNorm);
    let caught = match launch(&mut cp, [1, 1, 1], cs::BLOCK) {
        Err(_) => true,
        Ok(()) => {
            let s0 = cp.mem.read_f32s(S0, k.c).unwrap();
            out(&s0, &or_g, &bg)
        }
    };
    assert!(caught);
    // (2) dγ without x̂ (Σ dy instead of Σ dy·x̂).
    let (_, b) = run(&k, Mode::LayerNorm);
    assert!(
        out(&b, &or_g, &bg),
        "dβ in place of dγ must fall outside the margin"
    );
    // (3) One row left out of the sum (a loop bound off by one).
    let short = Case {
        x: k.x[..63 * 256].to_vec(),
        dy: k.dy[..63 * 256].to_vec(),
        rows: 63,
        c: 256,
    };
    let (_, b63) = run(&short, Mode::LayerNorm);
    assert!(
        out(&b63, &or_b, &bb),
        "a missing row must fall outside the margin"
    );
}
