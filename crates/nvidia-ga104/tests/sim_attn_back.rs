//! C-160 point 4 — T159, SIMULARE: training program 6, backward half (`tools/nak/attn_back.comp`, compiled by Mesa NAK
//! to `tools/nak/out/attn_back.sm86.sass`): the causal multi-head attention backward in two launches — mode 0 writes
//! the rows of dS (dP, the row dot, P·(dP − dot)·scale, 0 above the diagonal), mode 1 writes dQ | dK | dV into dqkv —
//! launched through our pushbuffer + QMD in the command processor model. The "program" runs on the CPU
//! (`attn_back_on_cpu` → `gpu_compute::layer::attn_back_row_gpu` / `attn_back_col_gpu`, in the order of the code NAK
//! made; no MUFU, so the model is the code). The outputs are judged against the f64 training oracle
//! (`gpu_compute::train::attention_back_rows`, the very lines `loss_and_grad` uses, which gave PyTorch's numbers at the
//! small golden step, T151) within a margin written BEFORE the measurement (`margins` below). Nothing here runs on the
//! RTX 3070 Ti.

use gpu_compute::layer::{AttnShape, attn_back_gpu};
use gpu_compute::train;
use gpu_sim::cmdproc::{CommandProcessor, Event, attn_back_on_cpu, gelu_tr_on_cpu};
use nvidia_ga104::attn_back::{self as ab, ATTN_BACK, Mode};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const CBUF2: u64 = 0x21_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const QMD2: u64 = 0x31_1000;
const PUSH: u64 = 0x32_0000;
const PUSH2: u64 = 0x32_1000;
const QKV: u64 = 0x100_0000;
const P: u64 = 0x400_0000;
const DOUT: u64 = 0x800_0000;
const DS: u64 = 0xc00_0000;
const DQKV: u64 = 0x1000_0000;
const SENTINEL: f32 = -12345.0;
/// 2⁻²⁴: one rounding of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// 2⁻¹²⁶: the program flushes anything smaller to 0 (an absolute slack per FTZ step).
const FTZ: f64 = 1.175_494_350_822_287_5e-38;
/// The oracle sums in f64: room for its own error, relative to the sum of |terms|.
const F64: f64 = 1.0 / (1u64 << 40) as f64;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// One case: the shape and the inputs the board would hold (P already rounded to f32), plus the f64 P of the forward.
struct Case {
    s: AttnShape,
    qkv: Vec<f32>,
    p32: Vec<f32>,
    p64: Vec<f64>,
    dout: Vec<f32>,
}

impl Case {
    fn new(seed: u64, s: AttnShape, qkv_amp: f32, dout_amp: f32) -> Case {
        let (nq, no, _) = s.sizes().unwrap();
        let mut r = rng(seed);
        let qkv: Vec<f32> = (0..nq).map(|_| r() * qkv_amp).collect();
        let dout: Vec<f32> = (0..no).map(|_| r() * dout_amp).collect();
        let (_, p64) = train::attention_rows(s.c, s.heads, &qkv, s.batch, s.t).unwrap();
        let p32: Vec<f32> = p64.iter().map(|&x| x as f32).collect();
        Case {
            s,
            qkv,
            p32,
            p64,
            dout,
        }
    }

    fn scale(&self) -> f32 {
        ab::Args::scale_for(self.s.d() as u32)
    }
}

// ------------------------------------------------------------------ the margin, written before the measurement
//
// Inputs are exact f32 values (the oracle reads the same P, as f64). An f32 `ffma` chain of n terms from 0 is off by at
// most 1.01·n·u·Σ|terms| (+ the error carried in its operands), each FTZ step by 2⁻¹²⁶:
//   dP_ij  (n = d):      ΔdP  = 1.01·d·u·Σ_e|dO_ie·V_je| + 2d·FTZ
//   dot_i  (n = i+1):    Δdot = Σ_j P_ij·ΔdP_ij + 1.01·(i+1)·u·Σ_j P_ij·|dP_ij| + 2(i+1)·FTZ
//   dS_ij = (P·(dP − dot))·scale32: the fadd, the two fmul and scale32 vs 1/√d each ≤ u:
//                        ΔdS  = s·P·(ΔdP + Δdot) + 4.1·u·s·P·(|dP − dot| + ΔdP + Δdot) + 3·FTZ
//   dQ_te  (n = t+1):    Σ_j ΔdS_tj·|K_je| + 1.01·(t+1)·u·Σ_j (|dS_tj| + ΔdS_tj)·|K_je| + 2(t+1)·FTZ
//   dK_te  (n = T−t):    Σ_i ΔdS_it·|Q_ie| + 1.01·n·u·Σ_i (|dS_it| + ΔdS_it)·|Q_ie| + 2n·FTZ
//   dV_te  (n = T−t):    1.01·n·u·Σ_i P_it·|dO_ie| + 2n·FTZ
// plus, for every output, the oracle's own rounding to f32 (1.01·u·|want|) and its f64 sums (2⁻⁴⁰·Σ|terms|).

struct Margins {
    /// The exact dS (f64) and its margin, [B, H, T, T].
    ds: Vec<f64>,
    dds: Vec<f64>,
    /// The margin of every dqkv element, [B·T, 3C].
    dqkv: Vec<f64>,
}

fn margins(k: &Case) -> Margins {
    let s = &k.s;
    let (t, c, d, nh) = (s.t, s.c, s.d(), s.heads);
    let scale = 1.0 / (d as f64).sqrt();
    let (nq, _, np) = s.sizes().unwrap();
    let mut ds = vec![0.0f64; np];
    let mut dds = vec![0.0f64; np];
    let mut m = vec![0.0f64; nq];
    let q = |b: usize, i: usize, part: usize, col: usize| {
        k.qkv[(b * t + i) * 3 * c + part * c + col] as f64
    };
    let o = |b: usize, i: usize, col: usize| k.dout[(b * t + i) * c + col] as f64;
    for b in 0..s.batch {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for i in 0..t {
                let (mut dp, mut ddp) = (vec![0.0f64; i + 1], vec![0.0f64; i + 1]);
                for j in 0..=i {
                    let (mut acc, mut abs) = (0.0f64, 0.0f64);
                    for e in 0..d {
                        let x = o(b, i, h * d + e) * q(b, j, 2, h * d + e);
                        acc += x;
                        abs += x.abs();
                    }
                    dp[j] = acc;
                    ddp[j] = 1.01 * d as f64 * U * abs + F64 * abs + 2.0 * d as f64 * FTZ;
                }
                let pr = |j: usize| k.p32[base + i * t + j] as f64;
                let (mut dot, mut ddot, mut abs) = (0.0f64, 0.0f64, 0.0f64);
                for j in 0..=i {
                    dot += pr(j) * dp[j];
                    ddot += pr(j) * ddp[j];
                    abs += (pr(j) * dp[j]).abs();
                }
                ddot += 1.01 * (i + 1) as f64 * U * abs + F64 * abs + 2.0 * (i + 1) as f64 * FTZ;
                for j in 0..=i {
                    let x = dp[j] - dot;
                    ds[base + i * t + j] = pr(j) * x * scale;
                    dds[base + i * t + j] = scale * pr(j) * (ddp[j] + ddot)
                        + 4.1 * U * scale * pr(j) * (x.abs() + ddp[j] + ddot)
                        + 3.0 * FTZ;
                }
            }
        }
    }
    for b in 0..s.batch {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for tp in 0..t {
                let r = b * t + tp;
                for e in 0..d {
                    let col = h * d + e;
                    let (mut carried, mut abs) = (0.0f64, 0.0f64);
                    for j in 0..=tp {
                        let at = base + tp * t + j;
                        carried += dds[at] * q(b, j, 1, col).abs();
                        abs += (ds[at].abs() + dds[at]) * q(b, j, 1, col).abs();
                    }
                    let n = (tp + 1) as f64;
                    m[r * 3 * c + col] = carried + 1.01 * n * U * abs + F64 * abs + 2.0 * n * FTZ;
                    let n = (t - tp) as f64;
                    let (mut carried, mut abs_k, mut abs_v) = (0.0f64, 0.0f64, 0.0f64);
                    for i in tp..t {
                        let at = base + i * t + tp;
                        carried += dds[at] * q(b, i, 0, col).abs();
                        abs_k += (ds[at].abs() + dds[at]) * q(b, i, 0, col).abs();
                        abs_v += k.p32[at] as f64 * o(b, i, col).abs();
                    }
                    m[r * 3 * c + c + col] =
                        carried + 1.01 * n * U * abs_k + F64 * abs_k + 2.0 * n * FTZ;
                    m[r * 3 * c + 2 * c + col] = 1.01 * n * U * abs_v + F64 * abs_v + 2.0 * n * FTZ;
                }
            }
        }
    }
    Margins { ds, dds, dqkv: m }
}

/// What one judgement found: the largest share of the margin used (1.0 = at the edge) and the largest |Δ|.
#[derive(Debug, Default)]
struct Verdict {
    share_ds: f64,
    share_dqkv: f64,
    max_abs: f64,
}

/// Judges dS (where it is defined: j ≤ i must be in the margin, j > i must be exactly 0) and dqkv against the f64
/// oracle `train::attention_back_rows` fed the same inputs. `Err` at the first value outside.
fn judge(k: &Case, ds: &[f32], dqkv: &[f32]) -> Result<Verdict, String> {
    let s = &k.s;
    let mg = margins(k);
    let p_same: Vec<f64> = k.p32.iter().map(|&x| x as f64).collect();
    let want =
        train::attention_back_rows(s.c, s.heads, &k.dout, &k.qkv, &p_same, s.batch, s.t).unwrap();
    let mut v = Verdict::default();
    let t = s.t;
    for (at, (&got, (&w, &dw))) in ds.iter().zip(mg.ds.iter().zip(&mg.dds)).enumerate() {
        let (i, j) = ((at / t) % t, at % t);
        if j > i {
            if got.to_bits() != 0 {
                return Err(format!("dS[{at}] above the diagonal = {got}, not +0"));
            }
            continue;
        }
        let err = (got as f64 - w).abs();
        if err.is_nan() || err > dw {
            return Err(format!(
                "dS[{at}] = {got}, want {w}, |err| {err:e} > margin {dw:e}"
            ));
        }
        v.share_ds = v.share_ds.max(err / dw);
    }
    for (at, ((&got, &w), &m)) in dqkv.iter().zip(&want).zip(&mg.dqkv).enumerate() {
        let b = m + 1.01 * U * (w as f64).abs();
        let err = (got as f64 - w as f64).abs();
        if err.is_nan() || err > b {
            return Err(format!(
                "dqkv[{at}] = {got}, want {w}, |err| {err:e} > margin {b:e}"
            ));
        }
        v.share_dqkv = v.share_dqkv.max(err / b);
        v.max_abs = v.max_abs.max(err);
    }
    Ok(v)
}

// ------------------------------------------------------------------ launching

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
    cbuf: CBUF2,
    qmd: QMD2,
    push: PUSH2,
    sem: 2,
};

fn cp_with(run: gpu_sim::cmdproc::ProgramFn) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program(ATTN_BACK, run);
    cp.mem.write(PROG, ATTN_BACK);
    cp
}

fn args(k: &Case, mode: Mode) -> ab::Args {
    ab::Args {
        qkv: QKV,
        p: P,
        dout: DOUT,
        ds: DS,
        dqkv: DQKV,
        batch: k.s.batch as u32,
        t: k.s.t as u32,
        c: k.s.c as u32,
        heads: k.s.heads as u32,
        scale: k.scale(),
        mode,
    }
}

fn launch_at(
    cp: &mut CommandProcessor,
    at: &At,
    a: &ab::Args,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<(), gpu_core::error::GpuError> {
    let mut cb = vec![0u8; 0x100];
    cb[..ab::CBUF_BYTES].copy_from_slice(&a.cbuf().unwrap());
    cp.mem.write(at.cbuf, &cb);
    let q = ComputeQmd {
        program_va: PROG,
        registers: ab::REGISTERS,
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

fn load(cp: &mut CommandProcessor, k: &Case, p: &[f32]) {
    let (nq, _, np) = k.s.sizes().unwrap();
    cp.mem.write_f32s(QKV, &k.qkv);
    cp.mem.write_f32s(P, p);
    cp.mem.write_f32s(DOUT, &k.dout);
    cp.mem.write_f32s(DS, &vec![SENTINEL; np + 64]);
    cp.mem.write_f32s(DQKV, &vec![SENTINEL; nq + 64]);
    cp.mem.write_u32(SEM, 0);
}

/// Both launches (rows, then columns) with P as given; returns (dS, dqkv) and checks nothing past them was written.
fn run_with_p(k: &Case, p: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (nq, _, np) = k.s.sizes().unwrap();
    let mut cp = cp_with(attn_back_on_cpu);
    load(&mut cp, k, p);
    let rows = args(k, Mode::Rows);
    launch_at(&mut cp, &FIRST, &rows, rows.grid(), ab::BLOCK).unwrap();
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let cols = args(k, Mode::Columns);
    launch_at(&mut cp, &SECOND, &cols, cols.grid(), ab::BLOCK).unwrap();
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 2);
    let launches = cp
        .events
        .iter()
        .filter(|e| matches!(e, Event::Launch { program: 0, .. }))
        .count();
    assert_eq!(launches, 2);
    let ds = cp.mem.read_f32s(DS, np + 64).unwrap();
    let dqkv = cp.mem.read_f32s(DQKV, nq + 64).unwrap();
    assert!(ds[np..].iter().all(|&v| v == SENTINEL), "wrote after dS");
    assert!(
        dqkv[nq..].iter().all(|&v| v == SENTINEL),
        "wrote after dqkv"
    );
    (ds[..np].to_vec(), dqkv[..nq].to_vec())
}

fn run(k: &Case) -> (Vec<f32>, Vec<f32>) {
    run_with_p(k, &k.p32)
}

fn shape(batch: usize, t: usize, c: usize, heads: usize) -> AttnShape {
    AttnShape { batch, t, c, heads }
}

/// The cases of T159: the small golden model's shape, T across the 32-thread block of mode 0, d > 32 (mode 1 loops
/// over e twice), one head, d = 1, sharp softmax (P near one-hot: dP − dot cancels), tiny gradients (as dlogits/(B·T)).
fn cases() -> Vec<(&'static str, Case)> {
    vec![
        (
            "small model B2 T32 C64 H4",
            Case::new(1, shape(2, 32, 64, 4), 1.5, 1.0),
        ),
        (
            "T70 C96 H3 (d 32)",
            Case::new(2, shape(1, 70, 96, 3), 1.0, 1.0),
        ),
        (
            "T33 C80 H2 (d 40)",
            Case::new(3, shape(2, 33, 80, 2), 1.0, 1.0),
        ),
        ("T5 C8 H1", Case::new(4, shape(3, 5, 8, 1), 2.0, 1.0)),
        (
            "T17 C3 H3 (d 1)",
            Case::new(5, shape(1, 17, 3, 3), 2.0, 1.0),
        ),
        ("sharp softmax", Case::new(6, shape(1, 40, 64, 4), 6.0, 1.0)),
        (
            "tiny gradients",
            Case::new(7, shape(2, 32, 64, 4), 1.5, 1e-4),
        ),
    ]
}

// ------------------------------------------------------------------ the tests

#[test]
fn t159_binary_is_the_recorded_build() {
    let hex: String = nvidia_ga104::sha256::digest(ATTN_BACK)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, ab::ATTN_BACK_SHA256);
    assert_eq!(ATTN_BACK.len(), 319 * 16, "319 instructions of 16 bytes");
    let txt = include_str!("../../../tools/nak/out/attn_back.sm86.txt");
    // The model's arithmetic, op for op: five ffma chains (dP, dot, dQ, dK, dV), −dot, dP − dot, ·P, ·scale.
    assert_eq!(txt.matches("= ffma.ftz").count(), 5, "five ffma chains");
    assert_eq!(txt.matches("= fmul.ftz").count(), 2, "P·x and ·scale");
    assert_eq!(txt.matches("= fadd.ftz").count(), 2, "−dot and dP − dot");
    assert!(
        !txt.contains("mufu")
            && !txt.contains("bar.")
            && !txt.contains("shfl")
            && !txt.contains("atom"),
        "no MUFU, no barrier, no shuffle, no atomics"
    );
    assert!(
        txt.contains("ldc.b32 c[0x0][+0x60]"),
        "the mode is read from 0x60"
    );
    assert!(txt.contains("c[0x0][0x64]"), "the scale is read from 0x64");
    assert!(txt.contains(&format!("Num GPRs: {}", ab::REGISTERS)));
    let k = Case::new(9, shape(2, 33, 64, 4), 1.0, 1.0);
    let ok = args(&k, Mode::Columns);
    let cb = ok.cbuf().unwrap();
    let rd64 = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
    let rd32 = |o: usize| u32::from_le_bytes(cb[o..o + 4].try_into().unwrap());
    assert_eq!(
        [rd64(0x28), rd64(0x30), rd64(0x38), rd64(0x40), rd64(0x48)],
        [QKV, P, DOUT, DS, DQKV]
    );
    assert_eq!(
        [rd32(0x50), rd32(0x54), rd32(0x58), rd32(0x5c), rd32(0x60)],
        [33, 64, 4, 16, 1]
    );
    assert_eq!(f32::from_bits(rd32(0x64)), 0.25);
    assert_eq!(rd64(0x0), 0);
    assert_eq!(rd32(0x8), 0);
    let rows = args(&k, Mode::Rows);
    assert_eq!(
        u32::from_le_bytes(rows.cbuf().unwrap()[0x60..0x64].try_into().unwrap()),
        0
    );
    assert_eq!(rows.grid(), [2, 4, 2]);
    assert_eq!(ok.grid(), [4, 33, 2]);
    for bad in [
        ab::Args { qkv: 0, ..ok },
        ab::Args { p: 0, ..ok },
        ab::Args { dout: 0, ..ok },
        ab::Args { ds: 0, ..ok },
        ab::Args { dqkv: 0, ..ok },
        ab::Args { t: 0, ..ok },
        ab::Args { batch: 0, ..ok },
        ab::Args { heads: 0, ..ok },
        ab::Args { c: 66, ..ok },
    ] {
        assert!(bad.cbuf().is_none(), "{bad:?}");
    }
    assert_eq!(ab::Args::scale_for(64), 0.125);
    assert_eq!(ab::Args::scale_for(48), (1.0 / 48f64.sqrt()) as f32);
}

#[test]
fn t159_within_the_written_margin() {
    for (what, k) in cases() {
        let (ds, dqkv) = run(&k);
        let v = judge(&k, &ds, &dqkv).unwrap_or_else(|e| panic!("{what}: {e}"));
        println!(
            "T159 {what}: dS {:.1} % of the margin, dqkv {:.1} %, max |Δ| {:.2e}",
            100.0 * v.share_ds,
            100.0 * v.share_dqkv,
            v.max_abs
        );
        assert!(v.share_ds < 1.0 && v.share_dqkv < 1.0);
    }
}

#[test]
fn t159_launched_is_the_model_and_runs_are_the_same_bits() {
    let k = Case::new(11, shape(2, 32, 64, 4), 1.5, 1.0);
    let (ds, dqkv) = run(&k);
    let (ds2, dqkv2) = run(&k);
    let (mds, mdqkv) = attn_back_gpu(&k.s, &k.qkv, &k.p32, &k.dout, k.scale()).unwrap();
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(same(&ds, &ds2) && same(&dqkv, &dqkv2), "two runs differ");
    assert!(
        same(&ds, &mds) && same(&dqkv, &mdqkv),
        "the launch is not the model"
    );
}

/// The causal mask is the loop bounds: P above the diagonal is never read. Garbage there (NaN, huge) changes no bit.
#[test]
fn t159_the_mask_never_reads_above_the_diagonal() {
    let k = Case::new(12, shape(2, 37, 48, 3), 1.5, 1.0);
    let (ds, dqkv) = run(&k);
    let t = k.s.t;
    let mut p = k.p32.clone();
    for (at, x) in p.iter_mut().enumerate() {
        if at % t > (at / t) % t {
            *x = if at % 2 == 0 { f32::NAN } else { 1e30 };
        }
    }
    let (ds2, dqkv2) = run_with_p(&k, &p);
    let same = |a: &[f32], b: &[f32]| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
    assert!(same(&ds, &ds2) && same(&dqkv, &dqkv2));
}

/// The chain the training step will use: P from the forward in f64, rounded to f32 for the board. dqkv is judged
/// against `train::attention_back_rows` with the f64 P (exactly `loss_and_grad`), with a tolerance written before the
/// measurement: per tensor part (dQ, dK, dV), max |Δ| ≤ 2⁻¹⁶ · max |want| (rounding P adds ≤ u relative per term).
#[test]
fn t159_against_the_training_steps_own_p() {
    for (what, k) in cases() {
        let (_, dqkv) = run(&k);
        let s = &k.s;
        let want = train::attention_back_rows(s.c, s.heads, &k.dout, &k.qkv, &k.p64, s.batch, s.t)
            .unwrap();
        for part in 0..3 {
            let pick = |v: &[f32]| -> Vec<f32> {
                v.chunks(3 * s.c)
                    .flat_map(|r| r[part * s.c..(part + 1) * s.c].to_vec())
                    .collect()
            };
            let (g, w) = (pick(&dqkv), pick(&want));
            let scale = w.iter().fold(0.0f32, |m, &x| m.max(x.abs())) as f64;
            let err = g
                .iter()
                .zip(&w)
                .fold(0.0f64, |m, (&a, &b)| m.max((a as f64 - b as f64).abs()));
            assert!(
                err <= scale / 65_536.0,
                "{what}, part {part}: max |Δ| {err:e} > 2^-16 · {scale:e}"
            );
        }
    }
}

/// A short mode-0 grid (fewer heads, fewer batches) leaves the other rows of dS untouched; grids past H / T and a
/// wrong block are refused (the program does not guard them), with nothing written.
#[test]
fn t159_short_grids_and_refusals() {
    let k = Case::new(13, shape(2, 40, 64, 4), 1.0, 1.0);
    let (_, _, np) = k.s.sizes().unwrap();
    let (full_ds, _) = run(&k);
    let mut cp = cp_with(attn_back_on_cpu);
    load(&mut cp, &k, &k.p32);
    let rows = args(&k, Mode::Rows);
    launch_at(&mut cp, &FIRST, &rows, [2, 3, 1], ab::BLOCK).unwrap();
    let ds = cp.mem.read_f32s(DS, np).unwrap();
    let t = k.s.t;
    for (at, (&got, &full)) in ds.iter().zip(&full_ds).enumerate() {
        let (b, h) = (at / (4 * t * t), (at / (t * t)) % 4);
        if b == 0 && h < 3 {
            assert_eq!(got.to_bits(), full.to_bits(), "dS[{at}]");
        } else {
            assert_eq!(got, SENTINEL, "dS[{at}] written by a short grid");
        }
    }
    // Only the first 32 rows of each (b, h) when the grid's x covers 32 positions.
    let mut cp = cp_with(attn_back_on_cpu);
    load(&mut cp, &k, &k.p32);
    launch_at(&mut cp, &FIRST, &rows, [1, 4, 2], ab::BLOCK).unwrap();
    let ds = cp.mem.read_f32s(DS, np).unwrap();
    for (at, &got) in ds.iter().enumerate() {
        let i = (at / t) % t;
        assert_eq!(i >= 32, got == SENTINEL, "dS[{at}]");
    }
    let cols = args(&k, Mode::Columns);
    for (a, grid, block) in [
        (rows, [3, 4, 2], ab::BLOCK),
        (rows, [2, 5, 2], ab::BLOCK),
        (cols, [5, 40, 2], ab::BLOCK),
        (cols, [4, 41, 2], ab::BLOCK),
        (rows, rows.grid(), [64, 1, 1]),
        (cols, cols.grid(), [32, 2, 1]),
    ] {
        let mut cp = cp_with(attn_back_on_cpu);
        load(&mut cp, &k, &k.p32);
        assert!(
            launch_at(&mut cp, &FIRST, &a, grid, block).is_err(),
            "{grid:?} {block:?}"
        );
        assert!(
            cp.mem
                .read_f32s(DS, np)
                .unwrap()
                .iter()
                .all(|&v| v == SENTINEL)
        );
    }
}

#[test]
fn t159_planted_mistakes_are_caught() {
    let k = Case::new(21, shape(2, 24, 64, 4), 1.5, 1.0);
    let s = k.s;
    let (t, c, d, nh) = (s.t, s.c, s.d(), s.heads);
    let (ds, dqkv) = run(&k);
    // Mistakes in dS (then dqkv recomputed from it with the right mode 1, so only the dS mistake shows).
    let cols_from = |ds: &[f32]| -> Vec<f32> {
        let mut out = vec![0.0f32; dqkv.len()];
        for b in 0..s.batch {
            for h in 0..nh {
                for tp in 0..t {
                    for e in 0..d {
                        let (q, kk, v) = gpu_compute::layer::attn_back_col_gpu(
                            &s, &k.qkv, &k.p32, &k.dout, ds, b, h, tp, e,
                        );
                        let r = b * t + tp;
                        out[r * 3 * c + h * d + e] = q;
                        out[r * 3 * c + c + h * d + e] = kk;
                        out[r * 3 * c + 2 * c + h * d + e] = v;
                    }
                }
            }
        }
        out
    };
    let ds_map = |f: &dyn Fn(usize, f32) -> f32| -> Vec<f32> {
        ds.iter().enumerate().map(|(at, &x)| f(at, x)).collect()
    };
    let scale = k.scale();
    // dS without the row dot (dS = P·dP·scale): recover dP·P·scale = dS + P·dot·scale.
    let (_, dqkv_ok) = attn_back_gpu(&s, &k.qkv, &k.p32, &k.dout, scale).unwrap();
    assert!(judge(&k, &ds, &dqkv_ok).is_ok());
    let mut mistakes: Vec<(&str, Vec<f32>, Vec<f32>)> = Vec::new();
    // (1) the scale forgotten
    let a = ds_map(&|_, x| x / scale);
    mistakes.push(("dS without the scale", cols_from(&a), a));
    // (2) the scale applied twice
    let a = ds_map(&|_, x| x * scale);
    mistakes.push(("dS with the scale twice", cols_from(&a), a));
    // (3) dS transposed (dS_ji in place of dS_ij)
    let a = ds_map(&|at, _| {
        let (blk, i, j) = (at / (t * t), (at / t) % t, at % t);
        ds[blk * t * t + j * t + i]
    });
    mistakes.push(("dS transposed", cols_from(&a), a));
    // (4) the sign of dS flipped
    let a = ds_map(&|_, x| -x);
    mistakes.push(("dS with the wrong sign", cols_from(&a), a));
    // (5) dS without the row dot: P·dP·scale, recomputed from the inputs.
    let a: Vec<f32> = {
        let mut out = vec![0.0f32; ds.len()];
        for b in 0..s.batch {
            for h in 0..nh {
                for i in 0..t {
                    for j in 0..=i {
                        let mut acc = 0.0f32;
                        for e in 0..d {
                            acc = k.dout[(b * t + i) * c + h * d + e]
                                .mul_add(k.qkv[(b * t + j) * 3 * c + 2 * c + h * d + e], acc);
                        }
                        let at = ((b * nh + h) * t + i) * t + j;
                        out[at] = k.p32[at] * acc * scale;
                    }
                }
            }
        }
        out
    };
    mistakes.push(("dS without the row dot", cols_from(&a), a));
    // Mistakes in mode 1 (dS right).
    let swap = |x: &[f32], p1: usize, p2: usize| -> Vec<f32> {
        let mut out = x.to_vec();
        for r in 0..s.batch * t {
            for k2 in 0..c {
                out.swap(r * 3 * c + p1 * c + k2, r * 3 * c + p2 * c + k2);
            }
        }
        out
    };
    // (6) dQ and dK swapped in dqkv
    mistakes.push(("dQ and dK swapped", swap(&dqkv, 0, 1), ds.clone()));
    // (7) dK and dV swapped
    mistakes.push(("dK and dV swapped", swap(&dqkv, 1, 2), ds.clone()));
    // (8) dV = Σ P·dO over the wrong triangle (i ≤ t instead of i ≥ t)
    let mut a = dqkv.clone();
    for b in 0..s.batch {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for tp in 0..t {
                for e in 0..d {
                    let col = h * d + e;
                    let mut acc = 0.0f32;
                    for i in 0..=tp {
                        acc = k.p32[base + tp * t + i].mul_add(k.dout[(b * t + i) * c + col], acc);
                    }
                    a[(b * t + tp) * 3 * c + 2 * c + col] = acc;
                }
            }
        }
    }
    mistakes.push(("dV over the wrong triangle", a, ds.clone()));
    // (9) dK with P in place of dS
    let mut a = dqkv.clone();
    for b in 0..s.batch {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for tp in 0..t {
                for e in 0..d {
                    let col = h * d + e;
                    let mut acc = 0.0f32;
                    for i in tp..t {
                        acc =
                            k.p32[base + i * t + tp].mul_add(k.qkv[(b * t + i) * 3 * c + col], acc);
                    }
                    a[(b * t + tp) * 3 * c + c + col] = acc;
                }
            }
        }
    }
    mistakes.push(("dK with P in place of dS", a, ds.clone()));
    // (10) heads mixed: dqkv of head h written in the columns of head (h+1) mod H
    let mut a = dqkv.clone();
    for r in 0..s.batch * t {
        for part in 0..3 {
            for h in 0..nh {
                for e in 0..d {
                    a[r * 3 * c + part * c + ((h + 1) % nh) * d + e] =
                        dqkv[r * 3 * c + part * c + h * d + e];
                }
            }
        }
    }
    mistakes.push(("heads shifted", a, ds.clone()));
    for (what, dq, dsm) in &mistakes {
        assert!(
            judge(&k, dsm, dq).is_err(),
            "planted mistake not caught: {what}"
        );
    }
    // (11) another program's model registered for this code (gelu_tr: other block contract) refuses or writes garbage.
    let mut cp = cp_with(gelu_tr_on_cpu);
    load(&mut cp, &k, &k.p32);
    let rows = args(&k, Mode::Rows);
    let caught = match launch_at(&mut cp, &FIRST, &rows, rows.grid(), ab::BLOCK) {
        Err(_) => true,
        Ok(()) => {
            let (_, _, np) = s.sizes().unwrap();
            let got = cp.mem.read_f32s(DS, np).unwrap();
            judge(&k, &got, &dqkv).is_err()
        }
    };
    assert!(caught, "another program's model was not caught");
}
