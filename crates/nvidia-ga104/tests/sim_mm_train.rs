//! C-160 point 4 — T152, SIMULARE: training program 1 (`tools/nak/mm_train.comp`, compiled by Mesa NAK to
//! `tools/nak/out/mm_train.sm86.sass`): `C = op(A)·op(B) (+ bias)`, the three products of a `Linear` layer
//! (forward `Y = X·Wᵀ + b`, `dX = dY·W`, `dW = dYᵀ·X`), launched through our pushbuffer + QMD in the command
//! processor model. The "program" runs on the CPU (`mm_train_on_cpu`, following the kernel's tiles); it is checked
//! (1) bit for bit against `reference::matmul_t_fma_ftz` (what the SASS computes: an `ffma.ftz` chain + `fadd.ftz`)
//! and (2) against the training oracle `gpu_compute::train::linear` / `linear_back` (f64 sums, C-160 point 3) on the
//! small golden model's real layer-0 `in_proj` weights, with a written error bound. Nothing here runs on the
//! RTX 3070 Ti.

use std::path::Path;

use gpu_compute::train;
use gpu_sim::cmdproc::{
    CommandProcessor, Event, MM_TRAIN_BIAS, MM_TRAIN_NEEDS, MM_TRAIN_TA, MM_TRAIN_TB,
    matmul_smem_on_cpu, mm_train_on_cpu,
};
use nvidia_ga104::mm_train::{self as mt, MM_TRAIN as MM};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const A: u64 = 0x40_0000;
const B: u64 = 0x50_0000;
const BIAS: u64 = 0x5f_0000;
const C: u64 = 0x60_0000;
const QMD: u64 = 0x70_0000;
const PUSH: u64 = 0x71_0000;
const SENTINEL: f32 = -12345.0;
const SHARED: SharedMemory = mt::SHARED;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

struct Job<'a> {
    a: &'a [f32],
    b: &'a [f32],
    bias: Option<&'a [f32]>,
    m: usize,
    k: usize,
    n: usize,
    flags: u32,
}

fn cp_with(run: gpu_sim::cmdproc::ProgramFn) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(MM, run, MM_TRAIN_NEEDS);
    cp.mem.write(PROG, MM);
    cp
}

fn load(cp: &mut CommandProcessor, j: &Job<'_>) {
    let args = mt::Args {
        a: A,
        b: B,
        c: C,
        bias: BIAS,
        m: j.m as u32,
        n: j.n as u32,
        k: j.k as u32,
        flags: j.flags,
    };
    let mut cb = vec![0u8; 0x100];
    match args.cbuf() {
        Some(bytes) => cb[..mt::CBUF_BYTES].copy_from_slice(&bytes),
        // Unknown flags: Args refuses them; write them raw so the model's own refusal is tested too.
        None => cb[0x54..0x58].copy_from_slice(&j.flags.to_le_bytes()),
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(A, j.a);
    cp.mem.write_f32s(B, j.b);
    if let Some(bias) = j.bias {
        cp.mem.write_f32s(BIAS, bias);
    }
    cp.mem.write_f32s(C, &vec![SENTINEL; j.m * j.n + 16]);
    cp.mem.write_u32(SEM, 0);
}

fn launch(
    cp: &mut CommandProcessor,
    grid: [u32; 3],
    shared: SharedMemory,
) -> Result<(), gpu_core::error::GpuError> {
    let q = ComputeQmd {
        program_va: PROG,
        registers: mt::REGISTERS,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid,
        block: mt::BLOCK,
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

/// Runs one job through the model; returns C and checks that nothing was written after it.
fn run(cp: &mut CommandProcessor, j: &Job<'_>) -> Vec<f32> {
    load(cp, j);
    let grid = [j.n.div_ceil(16) as u32, j.m.div_ceil(16) as u32, 1];
    assert_eq!(
        grid,
        mt::Args {
            a: A,
            b: B,
            c: C,
            bias: 0,
            m: j.m as u32,
            n: j.n as u32,
            k: j.k as u32,
            flags: 0
        }
        .grid()
    );
    launch(cp, grid, SHARED).unwrap();
    assert!(matches!(cp.events[1], Event::Launch { program: 0, .. }));
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let got = cp.mem.read_f32s(C, j.m * j.n + 16).unwrap();
    assert!(
        got[j.m * j.n..].iter().all(|&x| x == SENTINEL),
        "wrote after C"
    );
    got[..j.m * j.n].to_vec()
}

fn want(j: &Job<'_>) -> Vec<f32> {
    let mut out = vec![0.0; j.m * j.n];
    gpu_compute::reference::matmul_t_fma_ftz(
        j.a,
        j.b,
        j.bias,
        &mut out,
        j.m,
        j.k,
        j.n,
        j.flags & MM_TRAIN_TA != 0,
        j.flags & MM_TRAIN_TB != 0,
    )
    .unwrap();
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn t152_binary_is_the_recorded_build() {
    let hex: String = nvidia_ga104::sha256::digest(MM)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, mt::MM_TRAIN_SHA256);
    assert_eq!(MM.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out/mm_train.sm86.txt");
    assert!(txt.contains("bar"), "no barrier in the dump");
    assert!(
        txt.contains("ffma.ftz") && txt.contains("fadd.ftz"),
        "FMA chain + bias add expected"
    );
    // Args refuses what the program would misread.
    let ok = mt::Args {
        a: A,
        b: B,
        c: C,
        bias: 0,
        m: 2,
        n: 3,
        k: 4,
        flags: mt::TA | mt::TB,
    };
    assert!(ok.cbuf().is_some());
    assert!(mt::Args { flags: 8, ..ok }.cbuf().is_none());
    assert!(
        mt::Args {
            flags: mt::BIAS,
            ..ok
        }
        .cbuf()
        .is_none()
    );
    assert!(mt::Args { k: 0, ..ok }.cbuf().is_none());
    assert_eq!(
        (mt::TA, mt::TB, mt::BIAS),
        (MM_TRAIN_TA, MM_TRAIN_TB, MM_TRAIN_BIAS)
    );
}

#[test]
fn t152_all_forms_match_the_bit_exact_oracle_including_partial_tiles() {
    let mut r = rng(0x2545_f491_4f6c_dd1d);
    for (m, k, n) in [
        (37usize, 37usize, 21usize),
        (16, 16, 16),
        (5, 1, 3),
        (33, 48, 17),
    ] {
        let mut a: Vec<f32> = (0..m * k).map(|_| r()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
        let bias: Vec<f32> = (0..n).map(|_| r()).collect();
        a[0] = f32::from_bits(0x0000_1234); // subnormal input: FTZ flushes it
        for flags in [
            0,
            MM_TRAIN_TA,
            MM_TRAIN_TB,
            MM_TRAIN_TA | MM_TRAIN_TB,
            MM_TRAIN_BIAS,
            MM_TRAIN_TB | MM_TRAIN_BIAS,
        ] {
            let j = Job {
                a: &a,
                b: &b,
                bias: (flags & MM_TRAIN_BIAS != 0).then_some(&bias[..]),
                m,
                k,
                n,
                flags,
            };
            let mut cp = cp_with(mm_train_on_cpu);
            assert_eq!(
                bits(&run(&mut cp, &j)),
                bits(&want(&j)),
                "{m}x{k}x{n} f{flags}"
            );
        }
    }
}

#[test]
fn t152_transposed_reads_really_differ_and_negative_zero_survives() {
    // The same stored A read with and without TA must give different C (square, not symmetric).
    let mut r = rng(7);
    let (m, k, n) = (19usize, 19usize, 19usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    let plain = Job {
        a: &a,
        b: &b,
        bias: None,
        m,
        k,
        n,
        flags: 0,
    };
    let ta = Job {
        flags: MM_TRAIN_TA,
        ..plain
    };
    let tb = Job {
        flags: MM_TRAIN_TB,
        ..plain
    };
    let (c0, c1, c2) = (
        run(&mut cp_with(mm_train_on_cpu), &plain),
        run(&mut cp_with(mm_train_on_cpu), &ta),
        run(&mut cp_with(mm_train_on_cpu), &tb),
    );
    assert_ne!(bits(&c0), bits(&c1));
    assert_ne!(bits(&c0), bits(&c2));
    // Every product is -2^-140 (flushed to -0.0); K = 19 = 16 + 3: a zero-padded last tile would give +0.0.
    let tiny = f32::from_bits(0x1c80_0000);
    let (m, k, n) = (3usize, 19usize, 2usize);
    let a = vec![-tiny; m * k];
    let b = vec![tiny; k * n];
    for flags in [0, MM_TRAIN_TA, MM_TRAIN_TB] {
        let j = Job {
            a: &a,
            b: &b,
            bias: None,
            m,
            k,
            n,
            flags,
        };
        let got = run(&mut cp_with(mm_train_on_cpu), &j);
        assert!(got.iter().all(|x| x.to_bits() == 0x8000_0000), "f{flags}");
    }
}

#[test]
fn t152_refuses_unknown_flags_and_a_qmd_without_shared_memory() {
    let a = vec![1.0f32; 16 * 16];
    let j = Job {
        a: &a,
        b: &a,
        bias: None,
        m: 16,
        k: 16,
        n: 16,
        flags: 8,
    };
    let mut cp = cp_with(mm_train_on_cpu);
    load(&mut cp, &j);
    assert!(launch(&mut cp, [1, 1, 1], SHARED).is_err());
    let j = Job { flags: 0, ..j };
    load(&mut cp, &j);
    let few = SharedMemory {
        bytes: 1024,
        ..SHARED
    };
    assert!(launch(&mut cp, [1, 1, 1], few).is_err());
    let no_bar = SharedMemory {
        barriers: 0,
        ..SHARED
    };
    assert!(launch(&mut cp, [1, 1, 1], no_bar).is_err());
    let got = cp.mem.read_f32s(C, 16 * 16).unwrap();
    assert!(
        got.iter().all(|&x| x == SENTINEL),
        "refused launches wrote nothing"
    );
}

#[test]
fn t152_a_model_that_ignores_the_flags_is_caught() {
    // Planted mistake: the step-3 model (no transposes, no bias, m/n/k at other offsets) registered for this code.
    let mut r = rng(11);
    let (m, k, n) = (16usize, 16usize, 16usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    let j = Job {
        a: &a,
        b: &b,
        bias: None,
        m,
        k,
        n,
        flags: MM_TRAIN_TB,
    };
    let mut cp = cp_with(matmul_smem_on_cpu);
    load(&mut cp, &j);
    let caught = match launch(&mut cp, [1, 1, 1], SHARED) {
        Err(_) => true,
        Ok(()) => bits(&cp.mem.read_f32s(C, m * n).unwrap()) != bits(&want(&j)),
    };
    assert!(caught);
}

// ------------------------------------------------------------------ against the training oracle (C-160 point 3)

fn golden(name: &str, n: usize) -> Vec<f32> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../gpu-compute/tests/date/pas_de_aur_mic/initiale")
        .join(name);
    let b = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(b.len(), 4 * n);
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Worst |got − oracle| measured against the written bound `(K + 2)·2⁻²⁴·Σ|aᵢbᵢ| (+ |bias|)` of an f32 FMA chain
/// (Higham, γ_K) and the worst relative error; panics if any element is outside the bound.
fn judge(what: &str, got: &[f32], oracle: &[f32], scale: &[f64], k: usize) -> (f64, f64) {
    let u = (k as f64 + 2.0) * 2f64.powi(-24);
    let (mut worst_abs, mut worst_ratio) = (0f64, 0f64);
    for i in 0..got.len() {
        let d = (got[i] as f64 - oracle[i] as f64).abs();
        let bound = u * scale[i] + 1e-30;
        assert!(d <= bound, "{what}[{i}]: |d| {d:e} > bound {bound:e}");
        worst_abs = worst_abs.max(d);
        worst_ratio = worst_ratio.max(d / bound);
    }
    eprintln!("T152 {what}: max |d| = {worst_abs:.3e}, max |d| / bound = {worst_ratio:.4}");
    (worst_abs, worst_ratio)
}

#[test]
fn t152_linear_forward_and_backward_match_the_training_oracle_on_the_small_model() {
    // Layer 0 `in_proj` of the small golden model: W [3C, C] = [192, 64], b [192]; rows = 2 texts × 32 positions.
    let (c_in, c_out, rows) = (64usize, 192usize, 64usize);
    let w = golden("straturi.0.atentie.in_proj_weight.f32", c_out * c_in);
    let bias = golden("straturi.0.atentie.in_proj_bias.f32", c_out);
    let mut r = rng(0x51ce);
    let x: Vec<f32> = (0..rows * c_in).map(|_| r()).collect();
    let dy: Vec<f32> = (0..rows * c_out).map(|_| r() * 0.01).collect();

    // Forward: Y[rows, 192] = X · Wᵀ + b  →  A = X, B = W stored [N, K], flags TB | BIAS.
    let fwd = Job {
        a: &x,
        b: &w,
        bias: Some(&bias),
        m: rows,
        k: c_in,
        n: c_out,
        flags: MM_TRAIN_TB | MM_TRAIN_BIAS,
    };
    let y = run(&mut cp_with(mm_train_on_cpu), &fwd);
    assert_eq!(bits(&y), bits(&want(&fwd)));
    let y_or = train::linear(&x, rows, c_in, &w, c_out, Some(&bias));
    let s: Vec<f64> = (0..rows * c_out)
        .map(|i| {
            let (rr, o) = (i / c_out, i % c_out);
            (0..c_in)
                .map(|p| (x[rr * c_in + p] as f64 * w[o * c_in + p] as f64).abs())
                .sum::<f64>()
                + (bias[o] as f64).abs()
        })
        .collect();
    judge("Y = X·Wᵀ + b", &y, &y_or, &s, c_in);

    // The oracle's backward (dX and dW at once).
    let mut dw_or = vec![0.0f32; c_out * c_in];
    let dx_or = train::linear_back(&dy, &x, rows, c_in, &w, c_out, &mut dw_or, None);

    // dX[rows, 64] = dY · W  →  A = dY [M, K = 192], B = W [K, N], flags 0.
    let bx = Job {
        a: &dy,
        b: &w,
        bias: None,
        m: rows,
        k: c_out,
        n: c_in,
        flags: 0,
    };
    let dx = run(&mut cp_with(mm_train_on_cpu), &bx);
    assert_eq!(bits(&dx), bits(&want(&bx)));
    let s: Vec<f64> = (0..rows * c_in)
        .map(|i| {
            let (rr, kk) = (i / c_in, i % c_in);
            (0..c_out)
                .map(|o| (dy[rr * c_out + o] as f64 * w[o * c_in + kk] as f64).abs())
                .sum()
        })
        .collect();
    judge("dX = dY·W", &dx, &dx_or, &s, c_out);

    // dW[192, 64] = dYᵀ · X  →  A = dY stored [K = rows, M = 192], B = X [K, N], flags TA.
    let bw = Job {
        a: &dy,
        b: &x,
        bias: None,
        m: c_out,
        k: rows,
        n: c_in,
        flags: MM_TRAIN_TA,
    };
    let dw = run(&mut cp_with(mm_train_on_cpu), &bw);
    assert_eq!(bits(&dw), bits(&want(&bw)));
    let s: Vec<f64> = (0..c_out * c_in)
        .map(|i| {
            let (o, kk) = (i / c_in, i % c_in);
            (0..rows)
                .map(|rr| (dy[rr * c_out + o] as f64 * x[rr * c_in + kk] as f64).abs())
                .sum()
        })
        .collect();
    judge("dW = dYᵀ·X", &dw, &dw_or, &s, rows);
}

#[test]
fn t152_the_oracle_comparison_catches_a_wrong_form() {
    // Planted mistake: dW computed without TA (reads dY as if it were [M, K]) must fail the bound somewhere.
    let (c_in, c_out, rows) = (64usize, 192usize, 64usize);
    let w = golden("straturi.0.atentie.in_proj_weight.f32", c_out * c_in);
    let mut r = rng(0x51ce);
    let x: Vec<f32> = (0..rows * c_in).map(|_| r()).collect();
    let dy: Vec<f32> = (0..rows * c_out).map(|_| r() * 0.01).collect();
    let mut dw_or = vec![0.0f32; c_out * c_in];
    train::linear_back(&dy, &x, rows, c_in, &w, c_out, &mut dw_or, None);
    let wrong = Job {
        a: &dy,
        b: &x,
        bias: None,
        m: c_out,
        k: rows,
        n: c_in,
        flags: 0,
    };
    let dw = run(&mut cp_with(mm_train_on_cpu), &wrong);
    let worst = dw
        .iter()
        .zip(&dw_or)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(
        worst > 1e-3,
        "a wrong form must be far from the oracle (got {worst:e})"
    );
}
