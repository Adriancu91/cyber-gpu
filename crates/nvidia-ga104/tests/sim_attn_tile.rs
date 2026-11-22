//! C-160 point 7 — T173, SIMULARE: the training attention forward "on tiles" (`tools/nak/attn_fwd_tile.comp`, Mesa NAK
//! 25.0.7; `docs/C-160_P7_ATENTIA_PE_PLACI.md`). The plan and the predictions were written and pushed BEFORE this code
//! (commit `abd3307`):
//!   1. rows i ≤ 15: P and att bit for bit those of `attn_fwd` (each of the 16 partial sums holds at most one x);
//!   2. rows i ≥ 16: |ΔP| ≤ 2⁻²⁰·P, |Δatt| ≤ 2⁻²⁰·Σ|P·V| against `attn_fwd`; against the f64 oracle within T160's
//!      margin, the worst ≤ 20 % of it;
//!   3. the chain attn_fwd_tile → attn_back against the f64 P: ≤ 2⁻¹⁴;
//!   4. registers ≤ 64, no atomics;
//!   5. planted mistakes caught (a grid one block short, a wrong block, d > 64, the old sum in place of the 16 partials).
//!
//! What the simulation can and cannot show: the "program" runs on the CPU (`attn_fwd_tile_on_cpu` →
//! `gpu_compute::layer::attn_fwd_tile_row_gpu`, written from the GLSL source in the order of every sum), so here we
//! check the model against the f64 oracle and against `attn_fwd`, the driver side (grid, QMD with 37 376 bytes of
//! shared memory, cbuf, which rows get written) and, from the NAK listing, registers / barriers / MUFU / no atomics.
//! That the board computes what the model says is a prediction for the board. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::layer::{AttnShape, attn_back_gpu, attn_fwd_gpu, attn_fwd_tile_gpu};
use gpu_compute::train;
use gpu_sim::cmdproc::{CommandProcessor, attn_fwd_on_cpu, attn_fwd_tile_on_cpu};
use nvidia_ga104::attn_back as ab;
use nvidia_ga104::attn_fwd::{self as af, ATTN_FWD};
use nvidia_ga104::attn_fwd_tile::{self as at, ATTN_FWD_TILE};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
const QKV: u64 = 0x100_0000;
const P: u64 = 0x400_0000;
const ATT: u64 = 0x800_0000;
const SENTINEL: f32 = -12345.0;
/// 2⁻²⁴: one rounding of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// 2⁻¹²⁶: the program flushes anything smaller to 0 (an absolute slack per FTZ step).
const FTZ: f64 = 1.175_494_350_822_287_5e-38;
/// The oracle sums in f64: room for its own error, relative to the sum of |terms|.
const F64: f64 = 1.0 / (1u64 << 40) as f64;
/// Room for the board's approximate `MUFU.EX2` / `MUFU.RCP` (relative, each), as in T160.
const MUFU: f64 = 1.0 / (1u64 << 22) as f64;
/// Prediction 2: the tile program against attn_fwd, relative.
const NEAR: f64 = 1.0 / (1u64 << 20) as f64;

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
    s: AttnShape,
    qkv: Vec<f32>,
}

impl Case {
    fn new(seed: u64, s: AttnShape, amp: f32) -> Case {
        let (nq, _, _) = s.sizes().unwrap();
        let mut r = rng(seed);
        Case {
            s,
            qkv: (0..nq).map(|_| r() * amp).collect(),
        }
    }

    fn scale(&self) -> f32 {
        ab::Args::scale_for(self.s.d() as u32)
    }
}

// The margin of T160 (copied unchanged, written before T160's measurement). It holds for the 16 partial sums too: the
// bound of a sum's rounding is the depth of its additions times u, and the tree's depth ⌈(i+1)/16⌉ + 15 is ≤ i + 1
// for every i ≥ 15 (and for i ≤ 15 the sum is attn_fwd's chain).
// ------------------------------------------------------------------ the margin, written before the measurement
//
// Inputs are exact f32 values. For row (b, h, i), with the exact (f64) values s*_j = (Σ_e Q_ie·K_je)/√d, P*_j and
// a*_j = s*_j − max s*:
//   s_j  (ffma chain of d terms, then ·scale32):  Δs_j = s·(1.01·d·u·Σ|Q·K| + 2d·FTZ) + 2.01·u·|s*_j| + F64·s·Σ|Q·K|
//        (the fmul, and scale32 against 1/√d, one u each)
//   x_j = EX2((s_j − m)·log2e32): the fadd, the fmul and log2e32 give 3.01·u·|a_j| in the exponent (natural units),
//        the unit itself ≤ MUFU: ε_j = Δs_j + 3.01·u·|a*_j| + 3.01·u·Δs_j + MUFU (relative). m cancels (softmax is
//        invariant to the shift), so it brings no error of its own.
//   Σx (fadd chain of i+1 positive terms):  ε_Σ = Σ_j P*_j·ε_j + 1.01·(i+1)·u
//   P_j = x_j·RCP(Σx):  ΔP_j = 1.05·P*_j·(ε_j + ε_Σ + 2.01·u + MUFU) + (i+3)·FTZ
//   att_e (ffma chain of i+1 terms):  Σ_j ΔP_j·|V_je| + 1.01·(i+1)·u·Σ_j (P*_j + ΔP_j)·|V_je| + 2(i+1)·FTZ
//        + the oracle's rounding to f32 (1.01·u·|want|) and its f64 sums (F64·Σ|terms|).
// 1.05 covers the second-order products. Above the diagonal P must be exactly +0.

struct Margins {
    dp: Vec<f64>,
    datt: Vec<f64>,
}

fn margins(k: &Case, p_star: &[f64]) -> Margins {
    let s = &k.s;
    let (t, c, d, nh) = (s.t, s.c, s.d(), s.heads);
    let scale = 1.0 / (d as f64).sqrt();
    let (_, no, np) = s.sizes().unwrap();
    let mut dp = vec![0.0f64; np];
    let mut datt = vec![0.0f64; no];
    let q = |b: usize, i: usize, part: usize, col: usize| {
        k.qkv[(b * t + i) * 3 * c + part * c + col] as f64
    };
    for b in 0..s.batch {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for i in 0..t {
                let mut sx = vec![0.0f64; i + 1];
                let mut ds = vec![0.0f64; i + 1];
                for j in 0..=i {
                    let (mut acc, mut abs) = (0.0f64, 0.0f64);
                    for e in 0..d {
                        let x = q(b, i, 0, h * d + e) * q(b, j, 1, h * d + e);
                        acc += x;
                        abs += x.abs();
                    }
                    sx[j] = acc * scale;
                    ds[j] = scale * (1.01 * d as f64 * U * abs + 2.0 * d as f64 * FTZ)
                        + 2.01 * U * sx[j].abs()
                        + F64 * scale * abs;
                }
                let m = sx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let eps: Vec<f64> = (0..=i)
                    .map(|j| ds[j] + 3.01 * U * (sx[j] - m).abs() + 3.01 * U * ds[j] + MUFU)
                    .collect();
                let pr = |j: usize| p_star[base + i * t + j];
                let e_sum: f64 =
                    (0..=i).map(|j| pr(j) * eps[j]).sum::<f64>() + 1.01 * (i + 1) as f64 * U;
                for j in 0..=i {
                    dp[base + i * t + j] =
                        1.05 * pr(j) * (eps[j] + e_sum + 2.01 * U + MUFU) + (i + 3) as f64 * FTZ;
                }
                for e in 0..d {
                    let col = h * d + e;
                    let (mut carried, mut abs) = (0.0f64, 0.0f64);
                    for j in 0..=i {
                        let v = q(b, j, 2, col).abs();
                        carried += dp[base + i * t + j] * v;
                        abs += (pr(j) + dp[base + i * t + j]) * v;
                    }
                    let n = (i + 1) as f64;
                    datt[(b * t + i) * c + col] =
                        carried + 1.01 * n * U * abs + F64 * abs + 2.0 * n * FTZ;
                }
            }
        }
    }
    Margins { dp, datt }
}

#[derive(Debug, Default)]
struct Verdict {
    share_p: f64,
    share_att: f64,
}

/// Judges P (j ≤ i in the margin, j > i exactly +0) and att against the f64 oracle `train::attention_rows`.
fn judge(k: &Case, p: &[f32], att: &[f32]) -> Result<Verdict, String> {
    let s = &k.s;
    let (want_att, p_star) = train::attention_rows(s.c, s.heads, &k.qkv, s.batch, s.t).unwrap();
    let mg = margins(k, &p_star);
    let mut v = Verdict::default();
    let t = s.t;
    for (at, (&got, (&w, &dw))) in p.iter().zip(p_star.iter().zip(&mg.dp)).enumerate() {
        let (i, j) = ((at / t) % t, at % t);
        if j > i {
            if got.to_bits() != 0 {
                return Err(format!("P[{at}] above the diagonal = {got}, not +0"));
            }
            continue;
        }
        let err = (got as f64 - w).abs();
        if err.is_nan() || err > dw {
            return Err(format!(
                "P[{at}] = {got}, want {w}, |err| {err:e} > margin {dw:e}"
            ));
        }
        v.share_p = v.share_p.max(err / dw);
    }
    for (at, ((&got, &w), &m)) in att.iter().zip(&want_att).zip(&mg.datt).enumerate() {
        let b = m + 1.01 * U * (w as f64).abs();
        let err = (got as f64 - w as f64).abs();
        if err.is_nan() || err > b {
            return Err(format!(
                "att[{at}] = {got}, want {w}, |err| {err:e} > margin {b:e}"
            ));
        }
        v.share_att = v.share_att.max(err / b);
    }
    Ok(v)
}

// ------------------------------------------------------------------ launching

fn args(k: &Case) -> af::Args {
    af::Args {
        qkv: QKV,
        p: P,
        att: ATT,
        batch: k.s.batch as u32,
        t: k.s.t as u32,
        c: k.s.c as u32,
        heads: k.s.heads as u32,
        scale: k.scale(),
    }
}

/// Which program a launch uses.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Prog {
    Rows32,
    Tile,
}

/// One launch through pushbuffer + QMD (with the tile program's shared memory in the QMD).
fn launch(
    k: &Case,
    prog: Prog,
    grid: [u32; 3],
    block: [u32; 3],
    cbuf: &[u8],
) -> Result<(Vec<f32>, Vec<f32>), gpu_core::error::GpuError> {
    let (_, no, np) = k.s.sizes().unwrap();
    let mut cp = CommandProcessor::new();
    let (code, regs, shared): (&[u8], u32, Option<SharedMemory>) = match prog {
        Prog::Rows32 => {
            cp.register_program(ATTN_FWD, attn_fwd_on_cpu);
            (ATTN_FWD, af::REGISTERS, None)
        }
        Prog::Tile => {
            cp.register_program(ATTN_FWD_TILE, attn_fwd_tile_on_cpu);
            (ATTN_FWD_TILE, at::REGISTERS, Some(at::SHARED))
        }
    };
    cp.mem.write(PROG, code);
    cp.mem.write_f32s(QKV, &k.qkv);
    cp.mem.write_f32s(P, &vec![SENTINEL; np + 64]);
    cp.mem.write_f32s(ATT, &vec![SENTINEL; no + 64]);
    cp.mem.write_u32(SEM, 0);
    let mut cb = vec![0u8; 0x100];
    cb[..cbuf.len()].copy_from_slice(cbuf);
    cp.mem.write(CBUF, &cb);
    let q = ComputeQmd {
        program_va: PROG,
        registers: regs,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid,
        block,
        release: Some((SEM, 1)),
    };
    let qb = match shared {
        Some(s) => q.to_bytes_with_shared(&s).unwrap(),
        None => q.to_bytes().unwrap(),
    };
    cp.mem.write(QMD, &qb);
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
    cp.run_gpfifo(&[(e.entry0, e.entry1)])?;
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let pv = cp.mem.read_f32s(P, np + 64).unwrap();
    let av = cp.mem.read_f32s(ATT, no + 64).unwrap();
    assert!(pv[np..].iter().all(|&v| v == SENTINEL), "wrote after P");
    assert!(av[no..].iter().all(|&v| v == SENTINEL), "wrote after att");
    Ok((pv[..np].to_vec(), av[..no].to_vec()))
}

fn run(k: &Case, prog: Prog) -> (Vec<f32>, Vec<f32>) {
    let a = args(k);
    let cb = a.cbuf().unwrap();
    match prog {
        Prog::Rows32 => launch(k, prog, a.grid(), af::BLOCK, &cb).unwrap(),
        Prog::Tile => launch(k, prog, at::grid(&a), at::BLOCK, &cb).unwrap(),
    }
}

fn shape(batch: usize, t: usize, c: usize, heads: usize) -> AttnShape {
    AttnShape { batch, t, c, heads }
}

/// T160's cases + T across the 64-row tiles and #513's head width (d 64, 12 heads).
fn cases() -> Vec<(&'static str, Case)> {
    vec![
        (
            "small model B2 T32 C64 H4",
            Case::new(1, shape(2, 32, 64, 4), 1.5),
        ),
        ("T70 C96 H3 (d 32)", Case::new(2, shape(1, 70, 96, 3), 1.0)),
        ("T33 C80 H2 (d 40)", Case::new(3, shape(2, 33, 80, 2), 1.0)),
        ("T5 C8 H1", Case::new(4, shape(3, 5, 8, 1), 2.0)),
        ("T17 C3 H3 (d 1)", Case::new(5, shape(1, 17, 3, 3), 2.0)),
        ("sharp softmax", Case::new(6, shape(1, 40, 64, 4), 6.0)),
        (
            "very sharp (FTZ range)",
            Case::new(7, shape(1, 24, 16, 1), 30.0),
        ),
        ("T130 C32 H2", Case::new(8, shape(1, 130, 32, 2), 1.0)),
        (
            "#513 head width, T 200 (d 64)",
            Case::new(9, shape(1, 200, 768, 12), 1.0),
        ),
        (
            "T 64 exactly, T 65",
            Case::new(10, shape(2, 65, 128, 2), 1.0),
        ),
    ]
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn hex(code: &[u8]) -> String {
    nvidia_ga104::sha256::digest(code)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// Prediction 4 (checked when compiling).
const _: () = assert!(at::REGISTERS <= 64);
const _: () = assert!(at::SHARED.bytes <= at::SHARED.max_bytes);

// ------------------------------------------------------------------ the tests

#[test]
fn t173_binary_and_listing() {
    assert_eq!(hex(ATTN_FWD_TILE), at::ATTN_FWD_TILE_SHA256);
    let txt = include_str!("../../../tools/nak/out/attn_fwd_tile.sm86.txt");
    assert!(txt.contains(&format!("Num GPRs: {}", at::REGISTERS)));
    let ops: Vec<&str> = txt
        .lines()
        .filter_map(|l| {
            l.split(" = ")
                .nth(1)
                .and_then(|r| r.split_whitespace().next())
        })
        .collect();
    let count = |p: &str| ops.iter().filter(|o| o.starts_with(p)).count();
    // No atomics anywhere (every sum is a fixed chain).
    assert_eq!(count("atom") + count("red."), 0, "atomics in the listing");
    // The two MUFU units of attn_fwd: one EX2 (exp) and one RCP (1/sum).
    let mufu: Vec<&&str> = ops.iter().filter(|o| o.starts_with("mufu")).collect();
    assert!(mufu.iter().any(|o| o.contains("exp2")), "{mufu:?}");
    assert!(mufu.iter().any(|o| o.contains("rcp")), "{mufu:?}");
    // The QMD's shared size: 37 376 bytes, within NVK's 48 KiB.
    assert_eq!(at::SHARED.bytes, 37_376);
    println!(
        "T173 listing: {} registers, {} ops, mufu {:?}",
        at::REGISTERS,
        ops.len(),
        mufu
    );
}

/// Predictions 1 and 2, and the f64 oracle within T160's margin.
#[test]
fn t173_tile_against_attn_fwd_and_the_oracle() {
    for (name, k) in cases() {
        let (p, att) = run(&k, Prog::Tile);
        let (p32, a32) = run(&k, Prog::Rows32);
        // The launch is the model, bit for bit, twice.
        let (mp, ma) = attn_fwd_tile_gpu(&k.s, &k.qkv, k.scale()).unwrap();
        assert_eq!(bits(&p), bits(&mp), "{name}: P = the model");
        assert_eq!(bits(&att), bits(&ma), "{name}: att = the model");
        let (t, c, d, nh) = (k.s.t, k.s.c, k.s.d(), k.s.heads);
        let mut same_rows = 0usize;
        let mut rows = 0usize;
        let mut worst_p = 0.0f64;
        let mut worst_a = 0.0f64;
        for b in 0..k.s.batch {
            for h in 0..nh {
                for i in 0..t {
                    rows += 1;
                    let at = ((b * nh + h) * t + i) * t;
                    let (pr, pr32) = (&p[at..at + t], &p32[at..at + t]);
                    let o = (b * t + i) * c + h * d;
                    let (ar, ar32) = (&att[o..o + d], &a32[o..o + d]);
                    let same = bits(pr) == bits(pr32) && bits(ar) == bits(ar32);
                    if i <= 15 {
                        assert!(
                            same,
                            "{name}: row (b {b}, h {h}, i {i}) differs from attn_fwd"
                        );
                    }
                    same_rows += same as usize;
                    for j in 0..t {
                        let (x, y) = (pr[j] as f64, pr32[j] as f64);
                        if j > i {
                            assert_eq!(pr[j].to_bits(), 0);
                            continue;
                        }
                        let r = if y == 0.0 {
                            (x - y).abs()
                        } else {
                            (x - y).abs() / y
                        };
                        worst_p = worst_p.max(r / NEAR);
                        assert!(
                            (x - y).abs() <= NEAR * y.abs(),
                            "{name}: P[{i},{j}] {x:e} vs {y:e}"
                        );
                    }
                    for e in 0..d {
                        let mag: f64 = (0..=i)
                            .map(|j| {
                                (pr[j] as f64
                                    * k.qkv[(b * t + j) * 3 * c + 2 * c + h * d + e] as f64)
                                    .abs()
                            })
                            .sum();
                        let diff = (ar[e] as f64 - ar32[e] as f64).abs();
                        if mag > 0.0 {
                            worst_a = worst_a.max(diff / (NEAR * mag));
                        }
                        assert!(diff <= NEAR * mag, "{name}: att[{i},{e}] off by {diff:e}");
                    }
                }
            }
        }
        let v = judge(&k, &p, &att).unwrap_or_else(|e| panic!("{name}: {e}"));
        println!(
            "T173 {name}: oracle P {:.1} % / att {:.1} % of the margin; vs attn_fwd: {same_rows}/{rows} rows identical, \
             P {:.1} % / att {:.1} % of 2^-20",
            100.0 * v.share_p,
            100.0 * v.share_att,
            100.0 * worst_p,
            100.0 * worst_a
        );
        assert!(
            v.share_p <= 0.20 && v.share_att <= 0.20,
            "{name}: prediction 2 (≤ 20 %)"
        );
    }
}

/// Prediction 3: the chain of the training step, attn_fwd_tile's P into attn_back, against the f64 P — on T160's
/// chain cases (its first 6) plus the longer ones. "very sharp (FTZ range)" is left out as in T160: there the f32 P
/// itself is flushed to 0 where the f64 P is not, for attn_fwd too (its rows are attn_fwd's bit for bit, prediction 1).
#[test]
fn t173_chain_tile_then_backward() {
    for (name, k) in cases()
        .into_iter()
        .filter(|(n, _)| !n.starts_with("very sharp"))
    {
        let s = &k.s;
        let (p, _) = run(&k, Prog::Tile);
        let (_, no, _) = s.sizes().unwrap();
        let mut r = rng(77);
        let dout: Vec<f32> = (0..no).map(|_| r()).collect();
        let (_, dqkv) = attn_back_gpu(s, &k.qkv, &p, &dout, k.scale()).unwrap();
        let (_, p64) = train::attention_rows(s.c, s.heads, &k.qkv, s.batch, s.t).unwrap();
        let want =
            train::attention_back_rows(s.c, s.heads, &dout, &k.qkv, &p64, s.batch, s.t).unwrap();
        let c = s.c;
        for part in 0..3 {
            let pick = |v: &[f32]| -> Vec<f64> {
                v.chunks(3 * c)
                    .flat_map(|r| r[part * c..(part + 1) * c].iter().map(|&x| x as f64))
                    .collect()
            };
            let (g, w) = (pick(&dqkv), pick(&want));
            let big = w.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let worst = g
                .iter()
                .zip(&w)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                worst <= big / 16384.0,
                "{name}: part {part} off by {worst:e} (largest {big:e})"
            );
        }
    }
}

/// Prediction 5: planted mistakes are caught.
#[test]
fn t173_planted_mistakes_are_caught() {
    let k = Case::new(21, shape(2, 130, 64, 2), 1.0);
    let a = args(&k);
    let cb = a.cbuf().unwrap();
    let g = at::grid(&a);
    assert_eq!(g, [3, 2, 2]);
    // A grid one block short: rows 128, 129 stay sentinels.
    let (p, att) = launch(&k, Prog::Tile, [g[0] - 1, g[1], g[2]], at::BLOCK, &cb).unwrap();
    let left = p.iter().chain(&att).filter(|&&x| x == SENTINEL).count();
    assert_eq!(
        left,
        2 * 2 * 2 * 130 + 2 * 2 * 64,
        "rows 128, 129 of every (b, h)"
    );
    // The full grid writes everything.
    let (p, att) = launch(&k, Prog::Tile, g, at::BLOCK, &cb).unwrap();
    assert!(!p.iter().chain(&att).any(|&x| x == SENTINEL));
    // A wrong block, a grid past ⌈T/64⌉ or past H: refused.
    assert!(launch(&k, Prog::Tile, g, [128, 1, 1], &cb).is_err());
    assert!(launch(&k, Prog::Tile, g, [16, 16, 1], &cb).is_err());
    assert!(launch(&k, Prog::Tile, [g[0] + 1, g[1], g[2]], at::BLOCK, &cb).is_err());
    assert!(launch(&k, Prog::Tile, [g[0], g[1] + 1, g[2]], at::BLOCK, &cb).is_err());
    // d > 64: refused by the program's shape check and by the simulator.
    let k65 = Case::new(22, shape(1, 8, 130, 2), 1.0);
    let a65 = args(&k65);
    assert!(!at::shape_ok(&a65));
    assert!(at::shape_ok(&a));
    assert!(attn_fwd_tile_gpu(&k65.s, &k65.qkv, k65.scale()).is_none());
    assert!(
        launch(
            &k65,
            Prog::Tile,
            at::grid(&a65),
            at::BLOCK,
            &a65.cbuf().unwrap()
        )
        .is_err()
    );
    // The old sum (attn_fwd's single chain) in place of the 16 partials: differs on a long shape.
    let long = Case::new(23, shape(1, 200, 768, 12), 1.0);
    let (pt, _) = run(&long, Prog::Tile);
    let (old, _) = attn_fwd_gpu(&long.s, &long.qkv, long.scale()).unwrap();
    assert_ne!(
        bits(&pt),
        bits(&old),
        "the model would not tell the two sums apart"
    );
    // The tile result changed in one place (a planted wrong P) leaves the oracle's margin.
    let mut bad = pt.clone();
    let t = long.s.t;
    bad[150 * t + 3] *= 1.0 + 1.0 / 4096.0;
    let (_, att_t) = run(&long, Prog::Tile);
    assert!(judge(&long, &bad, &att_t).is_err());
}
