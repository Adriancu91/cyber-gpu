//! C-160 point 7 — T170, SIMULARE: the column sum split over rows (`docs/C-160_P7_COLSUM_IMPARTIT.md`): pass 1
//! `colsum_part` (new NAK binary) into partials, pass 2 the existing `colsum` over them, launched through our
//! pushbuffer + QMD in the command processor model (`colsum_part_on_cpu`, `colsum_on_cpu`). Judged against the f64
//! oracle `train::layernorm_back_rows` within `train::colsum_split_margin`, written before the code (commit `5d582a5`).
//! Nothing here runs on the RTX 3070 Ti.

use gpu_compute::{layer, train};
use gpu_sim::cmdproc::{CommandProcessor, colsum_on_cpu, colsum_part_on_cpu};
use nvidia_ga104::colsum::{self as cs, COLSUM, Mode};
use nvidia_ga104::colsum_part::{self as cp_, COLSUM_PART, Launch, Plan};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::clc7c0 as c;

const PROG_SUM: u64 = 0x10_0000;
const PROG_PART: u64 = 0x18_0000;
const CBUF: u64 = 0x20_0000; // three cbufs, 0x100 apart
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
const MEAN: u64 = 0x40_0000;
const RSTD: u64 = 0x48_0000;
const S0: u64 = 0x50_0000;
const S1: u64 = 0x58_0000;
const X: u64 = 0x100_0000;
const P0: u64 = 0x400_0000;
const P1: u64 = 0x600_0000;
const DY: u64 = 0x800_0000;
const SENTINEL: f32 = -12345.0;

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

fn plan(k: &Case, mode: Mode, chunk: usize) -> Plan {
    Plan {
        part: cp_::Args {
            dy: DY,
            x: X,
            mean: MEAN,
            rstd: RSTD,
            p0: P0,
            p1: P1,
            rows: k.rows as u32,
            c: k.c as u32,
            mode,
            chunk: chunk as u32,
        },
        s0: S0,
        s1: S1,
    }
}

fn machine(k: &Case) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program(COLSUM, colsum_on_cpu);
    cp.register_program(COLSUM_PART, colsum_part_on_cpu);
    cp.mem.write(PROG_SUM, COLSUM);
    cp.mem.write(PROG_PART, COLSUM_PART);
    let (mean, rstd) = train::layernorm_stats(&k.x, k.rows, k.c).unwrap();
    cp.mem.write_f32s(X, &k.x);
    cp.mem.write_f32s(DY, &k.dy);
    cp.mem.write_f32s(MEAN, &mean);
    cp.mem.write_f32s(RSTD, &rstd);
    cp.mem.write_f32s(S0, &vec![SENTINEL; k.c + 16]);
    cp.mem.write_f32s(S1, &vec![SENTINEL; k.c + 16]);
    cp.mem.write_u32(SEM, 0);
    cp
}

/// One QMD + pushbuffer submission; the semaphore must reach `n`.
fn submit(cp: &mut CommandProcessor, prog: u64, regs: u32, cbuf: u64, grid: [u32; 3], n: u32) {
    let q = ComputeQmd {
        program_va: prog,
        registers: regs,
        cbuf0_va: cbuf,
        cbuf0_size: 0x100,
        grid,
        block: cs::BLOCK,
        release: Some((SEM, n)),
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
    let len = p.len();
    let bytes: Vec<u8> = w[..len].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, len as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), n);
}

/// Runs the plan's launches in order; returns (s0, s1) and checks nothing was written after them.
fn run(k: &Case, mode: Mode, chunk: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cp = machine(k);
    let (ls, n) = plan(k, mode, chunk).launches().unwrap();
    for (i, l) in ls[..n].iter().enumerate() {
        let cb_va = CBUF + 0x100 * i as u64;
        let mut cb = vec![0u8; 0x100];
        match l {
            Launch::Part(a) => {
                cb[..cp_::CBUF_BYTES].copy_from_slice(&a.cbuf().unwrap());
                cp.mem.write(cb_va, &cb);
                submit(
                    &mut cp,
                    PROG_PART,
                    cp_::REGISTERS,
                    cb_va,
                    a.grid(),
                    i as u32 + 1,
                );
            }
            Launch::Sum(a) => {
                cb[..cs::CBUF_BYTES].copy_from_slice(&a.cbuf().unwrap());
                cp.mem.write(cb_va, &cb);
                submit(
                    &mut cp,
                    PROG_SUM,
                    cs::REGISTERS,
                    cb_va,
                    a.grid(),
                    i as u32 + 1,
                );
            }
        }
    }
    let s0 = cp.mem.read_f32s(S0, k.c + 16).unwrap();
    let s1 = cp.mem.read_f32s(S1, k.c + 16).unwrap();
    assert!(s0[k.c..].iter().all(|&v| v == SENTINEL), "wrote after s0");
    assert!(s1[k.c..].iter().all(|&v| v == SENTINEL), "wrote after s1");
    if mode == Mode::Bias {
        assert!(s1.iter().all(|&v| v == SENTINEL), "bias mode wrote s1");
    }
    (s0[..k.c].to_vec(), s1[..k.c].to_vec())
}

/// Worst fractions of the written margin (dγ, dβ-or-bias); panics outside it.
fn judge(what: &str, k: &Case, chunk: usize, s0: &[f32], s1: Option<&[f32]>) -> (f64, f64) {
    let gamma = vec![1.0f32; k.c];
    let (_, or_g, or_b) = train::layernorm_back_rows(&k.x, &k.dy, &gamma, k.rows, k.c).unwrap();
    let (bg, bb) = train::colsum_split_margin(&k.x, &k.dy, k.rows, k.c, chunk).unwrap();
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
    let worst = match s1 {
        None => (0.0, check("bias", s0, &or_b, &bb)),
        Some(s1) => (check("dγ", s0, &or_g, &bg), check("dβ", s1, &or_b, &bb)),
    };
    eprintln!(
        "T170 {what}: dγ {:.4} / dβ (or bias) {:.4} of the margin",
        worst.0, worst.1
    );
    worst
}

/// Largest |s − oracle| over both outputs, in absolute terms.
fn max_err(k: &Case, s0: &[f32], s1: &[f32]) -> f64 {
    let gamma = vec![1.0f32; k.c];
    let (_, or_g, or_b) = train::layernorm_back_rows(&k.x, &k.dy, &gamma, k.rows, k.c).unwrap();
    (0..k.c)
        .map(|j| ((s0[j] as f64 - or_g[j] as f64).abs()).max((s1[j] as f64 - or_b[j] as f64).abs()))
        .fold(0.0, f64::max)
}

#[test]
fn t170_binary_and_arguments() {
    let hex: String = nvidia_ga104::sha256::digest(COLSUM_PART)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, cp_::COLSUM_PART_SHA256);
    assert_eq!(COLSUM_PART.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out/colsum_part.sm86.txt");
    assert!(!txt.contains("mufu"), "no MUFU expected");
    assert!(!txt.contains("bar."), "no barrier expected");
    assert!(!txt.contains("atom"), "no atomics expected");
    let k = synthetic(1, 100, 768, 0.0);
    let p = plan(&k, Mode::LayerNorm, 7);
    assert_eq!(p.part.groups(), 15);
    assert_eq!(p.part.grid(), [3, 15, 1]);
    let cb = p.part.cbuf().unwrap();
    assert_eq!(u32::from_le_bytes(cb[0x58..0x5c].try_into().unwrap()), 100);
    assert_eq!(u32::from_le_bytes(cb[0x64..0x68].try_into().unwrap()), 7);
    assert!(cp_::Args { chunk: 0, ..p.part }.cbuf().is_none());
    assert!(cp_::Args { p1: 0, ..p.part }.cbuf().is_none());
    let (ls, n) = p.launches().unwrap();
    assert_eq!(n, 3);
    assert!(
        matches!(ls[1], Launch::Sum(a) if a.rows == 15 && a.dy == P0 && a.s0 == S0 && a.mode == Mode::Bias)
    );
    assert!(matches!(ls[2], Launch::Sum(a) if a.dy == P1 && a.s0 == S1));
    let bias = Plan {
        part: cp_::Args {
            mode: Mode::Bias,
            x: 0,
            mean: 0,
            rstd: 0,
            p1: 0,
            ..p.part
        },
        s1: 0,
        ..p
    };
    assert_eq!(bias.launches().unwrap().1, 2);
    assert!(Plan { s0: 0, ..p }.launches().is_none());
    assert!(Plan { s1: 0, ..p }.launches().is_none());
    assert_eq!(cp_::default_chunk(6144), 80);
    assert_eq!(cp_::default_chunk(64), 8);
    assert_eq!(cp_::default_chunk(3), 3);
    assert_eq!(cp_::default_chunk(0), 1);
}

#[test]
fn t170_within_the_written_margin() {
    let mut worst = 0.0f64;
    for (i, (rows, cc, offset)) in [
        (1usize, 1usize, 0.0f32),
        (64, 64, 0.0),
        (3, 257, 0.5),
        (300, 100, 0.0),
        (512, 768, 0.0),
        (64, 768, 100.0),
        (2048, 64, 0.0),
    ]
    .into_iter()
    .enumerate()
    {
        let k = synthetic(0x170_0000 + i as u64, rows, cc, offset);
        for chunk in [1usize, 7, 64, rows, rows + 5] {
            let (g, b) = run(&k, Mode::LayerNorm, chunk);
            let w = judge(
                &format!("LN {rows}x{cc} off={offset} chunk={chunk}"),
                &k,
                chunk,
                &g,
                Some(&b),
            );
            let (bias, _) = run(&k, Mode::Bias, chunk);
            let wb = judge(
                &format!("bias {rows}x{cc} chunk={chunk}"),
                &k,
                chunk,
                &bias,
                None,
            );
            // The bias sum and dβ: the same additions in the same order.
            assert!(bias.iter().zip(&b).all(|(p, q)| p.to_bits() == q.to_bits()));
            // The board's arithmetic is the CPU model, bit for bit.
            let st = train::layernorm_stats(&k.x, rows, cc).unwrap();
            let ln = layer::LnStats {
                x: &k.x,
                mean: &st.0,
                rstd: &st.1,
            };
            let (mg, mb) = layer::colsum_split_gpu(&k.dy, rows, cc, Some(ln), chunk);
            assert!(g.iter().zip(&mg).all(|(p, q)| p.to_bits() == q.to_bits()));
            assert!(b.iter().zip(&mb).all(|(p, q)| p.to_bits() == q.to_bits()));
            worst = worst.max(w.0).max(w.1).max(wb.1);
        }
    }
    eprintln!("T170 worst: {worst:.4} of the margin");
    assert!(worst <= 0.5, "prediction: at most 50 % of the margin");
}

#[test]
fn t170_one_chunk_is_colsum_bit_for_bit() {
    for (rows, cc) in [(1usize, 5usize), (64, 64), (3, 257), (8, 768)] {
        let k = synthetic(0x1700 + rows as u64, rows, cc, 0.2);
        let st = train::layernorm_stats(&k.x, rows, cc).unwrap();
        let ln = layer::LnStats {
            x: &k.x,
            mean: &st.0,
            rstd: &st.1,
        };
        let (cg, cb) = layer::colsum_gpu(&k.dy, rows, cc, Some(ln));
        for chunk in [rows, rows + 1, 10_000] {
            let (g, b) = run(&k, Mode::LayerNorm, chunk);
            assert!(
                g.iter().zip(&cg).all(|(p, q)| p.to_bits() == q.to_bits()),
                "{rows}x{cc} chunk={chunk}"
            );
            assert!(b.iter().zip(&cb).all(|(p, q)| p.to_bits() == q.to_bits()));
        }
    }
}

#[test]
fn t170_error_not_larger_than_colsum() {
    // Prediction 3 (written before): shorter sums are usually no worse than colsum's single run of R rows.
    for (seed, rows, cc, chunk) in [(31u64, 8usize, 768usize, 3usize), (32, 6144, 768, 64)] {
        let k = synthetic(seed, rows, cc, 0.0);
        let st = train::layernorm_stats(&k.x, rows, cc).unwrap();
        let ln = layer::LnStats {
            x: &k.x,
            mean: &st.0,
            rstd: &st.1,
        };
        let (cg, cb) = layer::colsum_gpu(&k.dy, rows, cc, Some(ln));
        let (g, b) = run(&k, Mode::LayerNorm, chunk);
        let (e_split, e_one) = (max_err(&k, &g, &b), max_err(&k, &cg, &cb));
        eprintln!("T170 {rows}x{cc} chunk={chunk}: split {e_split:e}, colsum {e_one:e}");
        assert!(e_split <= e_one, "split {e_split:e} > colsum {e_one:e}");
    }
}

#[test]
fn t170_planted_mistakes_are_caught() {
    let k = synthetic(0xbad1, 100, 256, 0.3);
    let chunk = 7;
    let gamma = vec![1.0f32; k.c];
    let (_, or_g, or_b) = train::layernorm_back_rows(&k.x, &k.dy, &gamma, k.rows, k.c).unwrap();
    let (bg, bb) = train::colsum_split_margin(&k.x, &k.dy, k.rows, k.c, chunk).unwrap();
    let out = |got: &[f32], want: &[f32], b: &[f64]| {
        (0..k.c).any(|j| (got[j] as f64 - want[j] as f64).abs() > b[j])
    };
    // (1) A chunk skipped: pass 2 over one row fewer of partials.
    let mut cp = machine(&k);
    let p = plan(&k, Mode::LayerNorm, chunk);
    let (ls, _) = p.launches().unwrap();
    let mut cb = vec![0u8; 0x100];
    let Launch::Part(a) = ls[0] else { panic!() };
    cb[..cp_::CBUF_BYTES].copy_from_slice(&a.cbuf().unwrap());
    cp.mem.write(CBUF, &cb);
    submit(&mut cp, PROG_PART, cp_::REGISTERS, CBUF, a.grid(), 1);
    let Launch::Sum(s) = ls[2] else { panic!() };
    let skipped = cs::Args {
        rows: s.rows - 1,
        ..s
    };
    let mut cb = vec![0u8; 0x100];
    cb[..cs::CBUF_BYTES].copy_from_slice(&skipped.cbuf().unwrap());
    cp.mem.write(CBUF + 0x100, &cb);
    submit(
        &mut cp,
        PROG_SUM,
        cs::REGISTERS,
        CBUF + 0x100,
        skipped.grid(),
        2,
    );
    let s1 = cp.mem.read_f32s(S1, k.c).unwrap();
    assert!(
        out(&s1, &or_b, &bb),
        "a skipped chunk must fall outside the margin"
    );
    // (2) The last chunk truncated: pass 1 told 98 rows instead of 100 (100 is not a multiple of 7).
    let short = Case {
        x: k.x[..98 * 256].to_vec(),
        dy: k.dy[..98 * 256].to_vec(),
        rows: 98,
        c: 256,
    };
    let (_, b98) = run(&short, Mode::LayerNorm, chunk);
    assert!(
        out(&b98, &or_b, &bb),
        "a truncated last chunk must fall outside the margin"
    );
    // (3) p0 and p1 swapped in pass 2: dβ where dγ belongs.
    let (_, b) = run(&k, Mode::LayerNorm, chunk);
    assert!(
        out(&b, &or_g, &bg),
        "dβ in place of dγ must fall outside the margin"
    );
}
