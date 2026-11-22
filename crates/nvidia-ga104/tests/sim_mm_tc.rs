//! C-160 point 7, part two — T168, SIMULARE: the training matmul on the tensor cores (`tools/nak/mm_tc.comp`,
//! second NAK tool, Mesa 26.3.1 → `tools/nak/out2/mm_tc.sm86.sass`): `C = op(A)·op(B) (+ bias)` with op(A), op(B)
//! rounded to f16 and added in f32 by the tensor cores, launched through our pushbuffer + QMD in the command
//! processor model. The "program" runs on the CPU (`mm_tc_on_cpu` = `gpu_compute::mm_tc::model`, the tensor-core
//! adder fitted on os-335 — a HYPOTHESIS). Checked (1) against the reference on the rounded inputs, in the margin
//! written in `gpu_compute::mm_tc` before any run, (2) against the f32 training oracle `train::linear` /
//! `linear_back` on the small golden model's real layer-0 `in_proj` weights, in margin + the written price of the
//! f16 rounding. Nothing here runs on the RTX 3070 Ti.

use std::path::Path;

use gpu_compute::mm_tc::{self as oracle, Product, Verdict};
use gpu_compute::train;
use gpu_sim::cmdproc::{CommandProcessor, Event, MM_TC_NEEDS, mm_tc_on_cpu, mm_train_on_cpu};
use nvidia_ga104::mm_tc::{self as tc, MM_TC};
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

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

fn cp_with(run: gpu_sim::cmdproc::ProgramFn) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(MM_TC, run, MM_TC_NEEDS);
    cp.mem.write(PROG, MM_TC);
    cp
}

fn args(p: &Product<'_>) -> tc::Args {
    tc::Args {
        a: A,
        b: B,
        c: C,
        bias: if p.flags & tc::BIAS != 0 { BIAS } else { 0 },
        m: p.m as u32,
        n: p.n as u32,
        k: p.k as u32,
        flags: p.flags,
    }
}

fn load(cp: &mut CommandProcessor, p: &Product<'_>) {
    let cb = match args(p).cbuf() {
        Some(bytes) => bytes.to_vec(),
        // Refused by Args: write the raw values so the model's own refusal is tested too.
        None => {
            let mut cb = vec![0u8; tc::CBUF_BYTES];
            let at = tc::PUSH_BASE;
            for (i, va) in [A, B, C, BIAS].into_iter().enumerate() {
                cb[at + 8 * i..at + 8 * i + 8].copy_from_slice(&va.to_le_bytes());
            }
            for (i, v) in [p.m as u32, p.n as u32, p.k as u32, p.flags]
                .into_iter()
                .enumerate()
            {
                cb[at + 0x20 + 4 * i..at + 0x24 + 4 * i].copy_from_slice(&v.to_le_bytes());
            }
            cb
        }
    };
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(A, &p.a[..p.m * p.k]);
    cp.mem.write_f32s(B, &p.b[..p.k * p.n]);
    if let Some(bias) = p.bias {
        cp.mem.write_f32s(BIAS, bias);
    }
    cp.mem.write_f32s(C, &vec![SENTINEL; p.m * p.n + 16]);
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
        registers: tc::REGISTERS,
        cbuf0_va: CBUF,
        cbuf0_size: tc::CBUF_BYTES as u32,
        grid,
        block,
        release: Some((SEM, 1)),
    };
    cp.mem.write(QMD, &q.to_bytes_with_shared(&shared).unwrap());
    let mut w = [0u32; 16];
    let mut pb = PushBuf::new(&mut w);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    pb.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    pb.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = pb.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

/// Runs one product through the model; returns C and checks that nothing was written after it.
fn run(cp: &mut CommandProcessor, p: &Product<'_>) -> Vec<f32> {
    load(cp, p);
    launch(cp, args(p).grid(), tc::BLOCK, tc::SHARED).unwrap();
    assert!(matches!(cp.events[1], Event::Launch { program: 0, .. }));
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let got = cp.mem.read_f32s(C, p.m * p.n + 16).unwrap();
    assert!(
        got[p.m * p.n..].iter().all(|&x| x == SENTINEL),
        "wrote after C"
    );
    got[..p.m * p.n].to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn in_margin(what: &str, p: &Product<'_>, got: &[f32]) -> (u32, usize) {
    match oracle::judge(p, got) {
        Verdict::InMargine {
            worst_permille,
            exact,
        } => {
            eprintln!(
                "T168 {what}: in margine, cel mai rau {worst_permille} ‰, {exact}/{} exacte",
                got.len()
            );
            (worst_permille, exact)
        }
        v => panic!("{what}: {v:?}"),
    }
}

#[test]
fn t168_binary_is_the_recorded_build_and_uses_the_tensor_cores() {
    let hex: String = nvidia_ga104::sha256::digest(MM_TC)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, tc::MM_TC_SHA256);
    assert_eq!(MM_TC.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out2/mm_tc.sm86.txt");
    assert!(
        txt.contains("hmma"),
        "no tensor-core instruction in the dump"
    );
    assert!(txt.contains("bar"), "no barrier in the dump");
    assert!(txt.contains("c[0x0][0x700]"), "push constants not at 0x700");
    assert_eq!(
        (tc::TA, tc::TB, tc::BIAS),
        (oracle::TA, oracle::TB, oracle::BIAS)
    );
    assert_eq!(tc::SHARED.bytes, MM_TC_NEEDS.shared_bytes);
}

#[test]
fn t168_all_forms_in_margin_including_partial_blocks() {
    let mut r = rng(0x2545_f491_4f6c_dd1d);
    for (m, k, n) in [
        (70usize, 33usize, 65usize),
        (64, 32, 64),
        (5, 1, 3),
        (130, 96, 17),
        (16, 200, 16),
    ] {
        let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
        let bias: Vec<f32> = (0..n).map(|_| r()).collect();
        for flags in 0..8u32 {
            let p = Product {
                a: &a,
                b: &b,
                bias: (flags & tc::BIAS != 0).then_some(&bias[..]),
                m,
                n,
                k,
                flags,
            };
            let got = run(&mut cp_with(mm_tc_on_cpu), &p);
            // The simulated board computes exactly the hypothesis (the model is the stand-in for the program) ...
            assert_eq!(bits(&got), bits(&oracle::model(&p).unwrap()));
            // ... and the hypothesis is inside the margin written before.
            let (w, _) = in_margin(&format!("{m}x{k}x{n} f{flags}"), &p, &got);
            assert!(w <= 1000);
        }
    }
}

#[test]
fn t168_transposes_really_differ() {
    let mut r = rng(7);
    let (m, k, n) = (19usize, 19usize, 19usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    let mk = |flags| Product {
        a: &a,
        b: &b,
        bias: None,
        m,
        n,
        k,
        flags,
    };
    let c0 = run(&mut cp_with(mm_tc_on_cpu), &mk(0));
    let c1 = run(&mut cp_with(mm_tc_on_cpu), &mk(tc::TA));
    let c2 = run(&mut cp_with(mm_tc_on_cpu), &mk(tc::TB));
    assert_ne!(bits(&c0), bits(&c1));
    assert_ne!(bits(&c0), bits(&c2));
    // A result judged against the wrong transpose is caught.
    assert!(matches!(
        oracle::judge(&mk(tc::TA), &c0),
        Verdict::NuTrece { .. }
    ));
}

#[test]
fn t168_refusals_write_nothing() {
    let a = vec![1.0f32; 80 * 64];
    let base = Product {
        a: &a,
        b: &a,
        bias: None,
        m: 64,
        n: 64,
        k: 64,
        flags: 0,
    };
    let empty = |cp: &CommandProcessor| {
        cp.mem
            .read_f32s(C, 64 * 64)
            .unwrap()
            .iter()
            .all(|&x| x == SENTINEL)
    };
    // Unknown flag bits.
    let mut cp = cp_with(mm_tc_on_cpu);
    load(&mut cp, &Product { flags: 8, ..base });
    assert!(launch(&mut cp, [1, 1, 1], tc::BLOCK, tc::SHARED).is_err());
    assert!(empty(&cp));
    // Too little shared memory, no barrier, the wrong block shape, a grid that does not cover C.
    for (grid, block, shared) in [
        (
            [1, 1, 1],
            tc::BLOCK,
            SharedMemory {
                bytes: 8 * 1024,
                ..tc::SHARED
            },
        ),
        (
            [1, 1, 1],
            tc::BLOCK,
            SharedMemory {
                barriers: 0,
                ..tc::SHARED
            },
        ),
        ([1, 1, 1], [16, 16, 1], tc::SHARED),
        ([1, 1, 1], tc::BLOCK, tc::SHARED),
    ] {
        let p = if grid == [1, 1, 1] && block == tc::BLOCK && shared == tc::SHARED {
            Product { m: 65, ..base } // needs 2 blocks in y
        } else {
            base
        };
        let mut cp = cp_with(mm_tc_on_cpu);
        load(&mut cp, &p);
        assert!(launch(&mut cp, grid, block, shared).is_err());
        assert!(empty(&cp));
    }
    // An input outside binary16 (would be ±inf on the board): refused, nothing written.
    let mut big = a.clone();
    big[5] = 1.0e5;
    let mut cp = cp_with(mm_tc_on_cpu);
    load(&mut cp, &Product { a: &big, ..base });
    assert!(launch(&mut cp, [1, 1, 1], tc::BLOCK, tc::SHARED).is_err());
    assert!(empty(&cp));
}

#[test]
fn t168_planted_mistakes_are_caught() {
    let mut r = rng(11);
    let (m, k, n) = (64usize, 96usize, 64usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    let p = Product {
        a: &a,
        b: &b,
        bias: None,
        m,
        n,
        k,
        flags: tc::TB,
    };
    // (1) The f32 program (no rounding to f16): exact-ish f32 sums are OUTSIDE the f16 margin — the verdict tells
    //     the two programs apart, so a board that silently ran F32 code would not pass as "tensor cores".
    let f32_like: Vec<f32> = (0..m * n)
        .map(|i| {
            (0..k)
                .map(|q| p.op_a(i / n, q) as f64 * p.op_b(q, i % n) as f64)
                .sum::<f64>() as f32
        })
        .collect();
    assert!(matches!(
        oracle::judge(&p, &f32_like),
        Verdict::NuTrece { .. }
    ));
    // (2) The mm_train model registered for this binary (push constants at 0x28, blocks of 16×16): refused.
    let mut cp = cp_with(mm_train_on_cpu);
    load(&mut cp, &p);
    assert!(launch(&mut cp, args(&p).grid(), tc::BLOCK, tc::SHARED).is_err());
    // (3) One element moved by 2 units in the last place at a large value: caught or within margin, never silent:
    //     move the largest element by 1 % instead — far outside.
    let mut got = run(&mut cp_with(mm_tc_on_cpu), &p);
    let i = (0..got.len())
        .max_by(|&x, &y| got[x].abs().total_cmp(&got[y].abs()))
        .unwrap();
    got[i] *= 1.01;
    assert!(matches!(
        oracle::judge(&p, &got),
        Verdict::NuTrece { index, .. } if index == i
    ));
}

// ------------------------------------------------------------------ against the f32 training oracle

fn golden(name: &str, n: usize) -> Vec<f32> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../gpu-compute/tests/date/pas_de_aur_mic/initiale")
        .join(name);
    let b = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(b.len(), 4 * n);
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

/// Max |got − f32 oracle| relative to max |oracle|, for the report.
fn rel(got: &[f32], want: &[f32]) -> f64 {
    let d = got
        .iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - *w as f64).abs())
        .fold(0.0, f64::max);
    let s = want.iter().map(|w| (*w as f64).abs()).fold(0.0, f64::max);
    d / s
}

#[test]
fn t168_linear_forward_and_backward_against_the_f32_training_oracle() {
    // Layer 0 `in_proj` of the small golden model: W [192, 64], b [192]; rows = 2 texts × 32 positions (like T152).
    let (c_in, c_out, rows) = (64usize, 192usize, 64usize);
    let w = golden("straturi.0.atentie.in_proj_weight.f32", c_out * c_in);
    let bias = golden("straturi.0.atentie.in_proj_bias.f32", c_out);
    let mut r = rng(0x51ce);
    let x: Vec<f32> = (0..rows * c_in).map(|_| r()).collect();
    let dy: Vec<f32> = (0..rows * c_out).map(|_| r() * 0.01).collect();

    let check = |what: &str, p: Product<'_>, want: &[f32]| {
        let got = run(&mut cp_with(mm_tc_on_cpu), &p);
        in_margin(what, &p, &got);
        let w = oracle::worst_vs_f32(&p, &got, want).unwrap();
        let e = rel(&got, want);
        eprintln!(
            "T168 {what}: fata de oracolul f32 {w} ‰ din marja + pretul f16; abaterea relativa {e:.2e}"
        );
        assert!(w <= 1000, "{what}: {w} ‰");
        // The f16 rounding must be visible (else the program would not be the tensor-core one), but small.
        assert!(e > 1e-6 && e < 2e-3, "{what}: {e:e}");
    };

    // Forward: Y = X·Wᵀ + b.
    let y_or = train::linear(&x, rows, c_in, &w, c_out, Some(&bias));
    check(
        "Y = X·Wt + b",
        Product {
            a: &x,
            b: &w,
            bias: Some(&bias),
            m: rows,
            n: c_out,
            k: c_in,
            flags: tc::TB | tc::BIAS,
        },
        &y_or,
    );
    let mut dw_or = vec![0.0f32; c_out * c_in];
    let dx_or = train::linear_back(&dy, &x, rows, c_in, &w, c_out, &mut dw_or, None);
    // dX = dY·W.
    check(
        "dX = dY·W",
        Product {
            a: &dy,
            b: &w,
            bias: None,
            m: rows,
            n: c_in,
            k: c_out,
            flags: 0,
        },
        &dx_or,
    );
    // dW = dYᵀ·X.
    check(
        "dW = dYt·X",
        Product {
            a: &dy,
            b: &x,
            bias: None,
            m: c_out,
            n: c_in,
            k: rows,
            flags: tc::TA,
        },
        &dw_or,
    );
}

#[test]
fn t168_tiny_gradients_vanish_without_scaling() {
    // The honest limit of f16 (written in the note): values under 2⁻²⁵ become 0, so a gradient of 1e-8 is lost
    // unless it is scaled first (loss scaling — the next step of point 7). Scaled by 2¹⁶ it survives.
    let (m, k, n) = (16usize, 16usize, 16usize);
    let dy = vec![1.0e-8f32; m * k];
    let x = vec![1.0f32; k * n];
    let p = Product {
        a: &dy,
        b: &x,
        bias: None,
        m,
        n,
        k,
        flags: 0,
    };
    let got = run(&mut cp_with(mm_tc_on_cpu), &p);
    assert!(got.iter().all(|&v| v == 0.0));
    let scaled: Vec<f32> = dy.iter().map(|v| v * 65536.0).collect();
    let p = Product { a: &scaled, ..p };
    let got = run(&mut cp_with(mm_tc_on_cpu), &p);
    let back = got[0] as f64 / 65536.0;
    assert!((back - 16.0e-8).abs() / 16.0e-8 < 2e-3, "{back:e}");
}
