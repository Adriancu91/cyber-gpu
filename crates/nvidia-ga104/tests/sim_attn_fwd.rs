//! C-160 point 4 — T160, SIMULARE: training program 6, forward half (`tools/nak/attn_fwd.comp`, compiled by Mesa NAK
//! to `tools/nak/out/attn_fwd.sm86.sass`): the causal multi-head attention over all T positions of B texts, writing
//! the softmax probabilities P [B, H, T, T] (+0 above the diagonal, kept for `attn_back`) and att [B·T, C] — launched
//! through our pushbuffer + QMD in the command processor model. The "program" runs on the CPU (`attn_fwd_on_cpu` →
//! `gpu_compute::layer::attn_fwd_row_gpu`, in the order of the code NAK made; `MUFU.EX2` / `MUFU.RCP` modelled
//! correctly rounded). P and att are judged against the f64 training oracle (`gpu_compute::train::attention_rows`, the
//! very lines `loss_and_grad` uses, which gave PyTorch's numbers at the small golden step, T151) within a margin
//! written BEFORE the measurement (`margins` below), with room for the board's approximate MUFU units. Nothing here
//! runs on the RTX 3070 Ti.

use gpu_compute::layer::{AttnShape, attn_back_gpu, attn_fwd_gpu};
use gpu_compute::train;
use gpu_sim::cmdproc::{CommandProcessor, Event, attn_back_on_cpu, attn_fwd_on_cpu};
use nvidia_ga104::attn_back::{self as ab, ATTN_BACK};
use nvidia_ga104::attn_fwd::{self as af, ATTN_FWD};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const PROG2: u64 = 0x18_0000;
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
/// Room for the board's approximate `MUFU.EX2` / `MUFU.RCP` (relative, each), as for `MUFU.RSQ` in T157.
const MUFU: f64 = 1.0 / (1u64 << 22) as f64;

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
    max_abs: f64,
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
        v.max_abs = v.max_abs.max(err);
    }
    Ok(v)
}

// ------------------------------------------------------------------ launching

fn cp_with(run: gpu_sim::cmdproc::ProgramFn) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program(ATTN_FWD, run);
    cp.mem.write(PROG, ATTN_FWD);
    cp
}

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

#[allow(clippy::too_many_arguments)]
fn launch_raw(
    cp: &mut CommandProcessor,
    prog: u64,
    regs: u32,
    cb_bytes: &[u8],
    grid: [u32; 3],
    block: [u32; 3],
    sem: u32,
) -> Result<(), gpu_core::error::GpuError> {
    let mut cb = vec![0u8; 0x100];
    cb[..cb_bytes.len()].copy_from_slice(cb_bytes);
    let cbuf = CBUF + 0x1000 * sem as u64;
    cp.mem.write(cbuf, &cb);
    let q = ComputeQmd {
        program_va: prog,
        registers: regs,
        cbuf0_va: cbuf,
        cbuf0_size: 0x100,
        grid,
        block,
        release: Some((SEM, sem)),
    };
    let qmd = QMD + 0x1000 * sem as u64;
    cp.mem.write(qmd, &q.to_bytes().unwrap());
    let mut w = [0u32; 16];
    let mut p = PushBuf::new(&mut w);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(qmd >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = p.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    let push_at = PUSH + 0x1000 * sem as u64;
    cp.mem.write(push_at, &bytes);
    let e = push::gpfifo_entry(push_at, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

fn launch(
    cp: &mut CommandProcessor,
    a: &af::Args,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<(), gpu_core::error::GpuError> {
    launch_raw(cp, PROG, af::REGISTERS, &a.cbuf().unwrap(), grid, block, 1)
}

fn load(cp: &mut CommandProcessor, k: &Case) {
    let (_, no, np) = k.s.sizes().unwrap();
    cp.mem.write_f32s(QKV, &k.qkv);
    cp.mem.write_f32s(P, &vec![SENTINEL; np + 64]);
    cp.mem.write_f32s(ATT, &vec![SENTINEL; no + 64]);
    cp.mem.write_u32(SEM, 0);
}

/// One launch with the full grid; returns (P, att) and checks nothing past them was written.
fn run(k: &Case) -> (Vec<f32>, Vec<f32>) {
    let (_, no, np) = k.s.sizes().unwrap();
    let mut cp = cp_with(attn_fwd_on_cpu);
    load(&mut cp, k);
    let a = args(k);
    launch(&mut cp, &a, a.grid(), af::BLOCK).unwrap();
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let launches = cp
        .events
        .iter()
        .filter(|e| matches!(e, Event::Launch { program: 0, .. }))
        .count();
    assert_eq!(launches, 1);
    let p = cp.mem.read_f32s(P, np + 64).unwrap();
    let att = cp.mem.read_f32s(ATT, no + 64).unwrap();
    assert!(p[np..].iter().all(|&v| v == SENTINEL), "wrote after P");
    assert!(att[no..].iter().all(|&v| v == SENTINEL), "wrote after att");
    (p[..np].to_vec(), att[..no].to_vec())
}

fn shape(batch: usize, t: usize, c: usize, heads: usize) -> AttnShape {
    AttnShape { batch, t, c, heads }
}

/// The cases of T160: the small golden model's shape, T across the 32-thread block, d = 40, one head, d = 1, a sharp
/// softmax (P near one-hot, exponents down to the FTZ range), a flat one (all s equal), large T.
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
    ]
}

// ------------------------------------------------------------------ the tests

#[test]
fn t160_binary_is_the_recorded_build() {
    let hex: String = nvidia_ga104::sha256::digest(ATTN_FWD)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, af::ATTN_FWD_SHA256);
    let txt = include_str!("../../../tools/nak/out/attn_fwd.sm86.txt");
    // The model's arithmetic: two ffma chains (Q·K, P·V), one MUFU.EX2 and one MUFU.RCP per row element / row.
    assert_eq!(ATTN_FWD.len(), 221 * 16, "221 instructions of 16 bytes");
    assert_eq!(
        txt.matches("= ffma.ftz").count(),
        2,
        "two ffma chains (Q·K, P·V)"
    );
    assert_eq!(txt.matches("= mufu.exp2").count(), 1, "one EX2 (exp)");
    assert!(
        txt.contains("fmul.ftz r12 0x3fb8aa3b"),
        "exp(x) = ex2(x·log2 e), log2 e as the model's constant"
    );
    assert_eq!(gpu_compute::layer::ATTN_FWD_LOG2E, 0x3fb8_aa3b);
    assert_eq!(
        txt.matches("= mufu.rcp").count(),
        1,
        "one RCP (1/sum), no Newton step"
    );
    assert!(
        !txt.contains("bar.") && !txt.contains("shfl") && !txt.contains("atom"),
        "no barrier, no shuffle, no atomics"
    );
    assert!(txt.contains("c[0x0][0x50]"), "the scale is read from 0x50");
    assert!(txt.contains(&format!("Num GPRs: {}", af::REGISTERS)));
    let k = Case::new(9, shape(2, 33, 64, 4), 1.0);
    let cb = args(&k).cbuf().unwrap();
    let rd64 = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
    let rd32 = |o: usize| u32::from_le_bytes(cb[o..o + 4].try_into().unwrap());
    assert_eq!([rd64(0x28), rd64(0x30), rd64(0x38)], [QKV, P, ATT]);
    assert_eq!(
        [rd32(0x40), rd32(0x44), rd32(0x48), rd32(0x4c)],
        [33, 64, 4, 16]
    );
    assert_eq!(f32::from_le_bytes(cb[0x50..0x54].try_into().unwrap()), 0.25);
    assert!(cb[..0x28].iter().all(|&b| b == 0), "workgroup base left 0");
    assert_eq!(args(&k).grid(), [2, 4, 2]);
}

#[test]
fn t160_forward_in_the_written_margin() {
    for (name, k) in cases() {
        let (p, att) = run(&k);
        let v = judge(&k, &p, &att).unwrap_or_else(|e| panic!("{name}: {e}"));
        println!(
            "T160 {name}: P {:.1} % of the margin, att {:.1} %, max |Δ| att {:.2e}",
            100.0 * v.share_p,
            100.0 * v.share_att,
            v.max_abs
        );
        // The launch is the model, bit for bit, and two runs give the same bits.
        let (mp, ma) = attn_fwd_gpu(&k.s, &k.qkv, k.scale()).unwrap();
        assert_eq!(bits(&p), bits(&mp), "{name}: P = the model");
        assert_eq!(bits(&att), bits(&ma), "{name}: att = the model");
        let (p2, a2) = run(&k);
        assert_eq!(
            (bits(&p), bits(&att)),
            (bits(&p2), bits(&a2)),
            "{name}: same bits twice"
        );
        // Every row of P sums to 1 within the margin of its terms.
        let t = k.s.t;
        for (r, row) in p.chunks(t).enumerate() {
            let sum: f64 = row.iter().map(|&x| x as f64).sum();
            assert!(
                (sum - 1.0).abs() < 4e-6 * t as f64,
                "{name}: row {r} sums to {sum}"
            );
        }
    }
}

/// (P, att) of one launch.
type Out = (Vec<f32>, Vec<f32>);

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// The chain of the training step: attn_fwd's P (f32, from the board's program) straight into attn_back, against
/// `attention_back_rows` with the f64 P of the forward (exactly `loss_and_grad`): each part (dQ, dK, dV) within 2⁻¹⁴ of
/// its largest value.
#[test]
fn t160_chain_forward_then_backward() {
    for (name, k) in cases().into_iter().take(6) {
        let s = &k.s;
        let (p, _) = run(&k);
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

/// The P of this program is the P attn_back reads: two launches on one command processor (forward, then the backward's
/// row mode) give the same dS as attn_back's model fed this P.
#[test]
fn t160_two_programs_one_processor() {
    let k = Case::new(11, shape(2, 33, 64, 4), 1.2);
    let s = &k.s;
    let (_, no, np) = s.sizes().unwrap();
    let mut cp = cp_with(attn_fwd_on_cpu);
    cp.register_program(ATTN_BACK, attn_back_on_cpu);
    cp.mem.write(PROG2, ATTN_BACK);
    load(&mut cp, &k);
    let a = args(&k);
    launch(&mut cp, &a, a.grid(), af::BLOCK).unwrap();
    let mut r = rng(5);
    let dout: Vec<f32> = (0..no).map(|_| r()).collect();
    const DOUT: u64 = 0xc00_0000;
    const DS: u64 = 0x1000_0000;
    const DQKV: u64 = 0x1400_0000;
    cp.mem.write_f32s(DOUT, &dout);
    let b = ab::Args {
        qkv: QKV,
        p: P,
        dout: DOUT,
        ds: DS,
        dqkv: DQKV,
        batch: s.batch as u32,
        t: s.t as u32,
        c: s.c as u32,
        heads: s.heads as u32,
        scale: k.scale(),
        mode: ab::Mode::Rows,
    };
    launch_raw(
        &mut cp,
        PROG2,
        ab::REGISTERS,
        &b.cbuf().unwrap(),
        b.grid(),
        ab::BLOCK,
        2,
    )
    .unwrap();
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 2);
    let p = cp.mem.read_f32s(P, np).unwrap();
    let ds = cp.mem.read_f32s(DS, np).unwrap();
    let (want_ds, _) = attn_back_gpu(s, &k.qkv, &p, &dout, k.scale()).unwrap();
    assert_eq!(bits(&ds), bits(&want_ds));
}

#[test]
fn t160_mask_short_grid_and_refusals() {
    let k = Case::new(12, shape(2, 70, 64, 4), 1.0);
    let (_, no, np) = k.s.sizes().unwrap();
    let t = k.s.t;
    // A short grid (x = 1 of 3, one head, one text) writes only those rows; the rest keeps the sentinel.
    let mut cp = cp_with(attn_fwd_on_cpu);
    load(&mut cp, &k);
    let a = args(&k);
    launch(&mut cp, &a, [1, 1, 1], af::BLOCK).unwrap();
    let p = cp.mem.read_f32s(P, np).unwrap();
    let att = cp.mem.read_f32s(ATT, no).unwrap();
    let (mp, _) = attn_fwd_gpu(&k.s, &k.qkv, k.scale()).unwrap();
    for (at, &x) in p.iter().enumerate() {
        let (bh, i) = (at / (t * t), (at / t) % t);
        if bh == 0 && i < 32 {
            assert_eq!(
                x.to_bits(),
                mp[at].to_bits(),
                "P[{at}] written as the model"
            );
        } else {
            assert_eq!(x, SENTINEL, "P[{at}] outside the short grid touched");
        }
    }
    for (at, &x) in att.iter().enumerate() {
        let (row, col) = (at / k.s.c, at % k.s.c);
        let touched = row < 32 && col < k.s.d();
        assert_eq!(x == SENTINEL, !touched, "att[{at}]");
    }
    // Refused, nothing written: a block of 64, a grid past H or past ⌈T/32⌉, a bad shape in the arguments.
    for (grid, block) in [
        ([3, 4, 2], [64, 1, 1]),
        ([3, 5, 2], af::BLOCK),
        ([4, 4, 2], af::BLOCK),
    ] {
        let mut cp = cp_with(attn_fwd_on_cpu);
        load(&mut cp, &k);
        assert!(
            launch(&mut cp, &a, grid, block).is_err(),
            "{grid:?} {block:?}"
        );
        assert!(
            cp.mem
                .read_f32s(P, np)
                .unwrap()
                .iter()
                .all(|&v| v == SENTINEL)
        );
        assert!(
            cp.mem
                .read_f32s(ATT, no)
                .unwrap()
                .iter()
                .all(|&v| v == SENTINEL)
        );
    }
    let mut bad = a;
    bad.heads = 3; // 64 is not a multiple of 3
    assert!(bad.cbuf().is_none());
    let mut bad = a;
    bad.p = 0;
    assert!(bad.cbuf().is_none());
    // Garbage above the diagonal of a previous P is overwritten with +0 (the mask is the loop bounds).
    let mut cp = cp_with(attn_fwd_on_cpu);
    load(&mut cp, &k);
    cp.mem.write_f32s(P, &vec![f32::NAN; np]);
    launch(&mut cp, &a, a.grid(), af::BLOCK).unwrap();
    let p = cp.mem.read_f32s(P, np).unwrap();
    assert_eq!(bits(&p), bits(&mp));
}

/// Mistakes planted in a copy of the model must fall outside the margin (the margin is not so wide it lets them pass).
#[test]
fn t160_planted_mistakes_are_caught() {
    let k = Case::new(1, shape(2, 32, 64, 4), 1.5);
    let s = k.s;
    let (t, c, d) = (s.t, s.c, s.d());
    let good = attn_fwd_gpu(&s, &k.qkv, k.scale()).unwrap();
    assert!(judge(&k, &good.0, &good.1).is_ok());
    let swap = |f: &dyn Fn(usize, usize) -> (usize, usize)| -> Vec<f32> {
        // qkv with columns moved: f(part, col) -> (part, col) to read from.
        let mut q = k.qkv.clone();
        for r in 0..s.batch * t {
            for part in 0..3 {
                for col in 0..c {
                    let (pp, cc) = f(part, col);
                    q[r * 3 * c + part * c + col] = k.qkv[r * 3 * c + pp * c + cc];
                }
            }
        }
        q
    };
    let mut wrong: Vec<(&str, Out)> = vec![
        (
            "scale twice",
            attn_fwd_gpu(&s, &k.qkv, k.scale() * k.scale()).unwrap(),
        ),
        ("no scale", attn_fwd_gpu(&s, &k.qkv, 1.0).unwrap()),
        (
            "Q and K swapped",
            attn_fwd_gpu(&s, &swap(&|p, col| ([1, 0, 2][p], col)), k.scale()).unwrap(),
        ),
        (
            "V read from K",
            attn_fwd_gpu(&s, &swap(&|p, col| ([0, 1, 1][p], col)), k.scale()).unwrap(),
        ),
        ("heads moved", {
            let (p, a) = attn_fwd_gpu(&s, &swap(&|p, col| (p, (col + d) % c)), k.scale()).unwrap();
            (p, a)
        }),
    ];
    // P transposed (the mask on the wrong side), att left right.
    let mut pt = good.0.clone();
    for bh in 0..s.batch * s.heads {
        for i in 0..t {
            for j in 0..t {
                pt[bh * t * t + i * t + j] = good.0[bh * t * t + j * t + i];
            }
        }
    }
    wrong.push(("P transposed", (pt, good.1.clone())));
    // One position more (j ≤ i + 1): the mask off by one. Q·K of row i with row i+1 leaks into the softmax.
    let mut leak = good.clone();
    for bh in 0..s.batch * s.heads {
        let row = bh * t * t + 3 * t;
        leak.0[row + 4] = leak.0[row + 3] * 0.5;
    }
    wrong.push(("mask off by one", leak));
    // P not normalised (no 1/sum).
    let mut raw = good.clone();
    for x in raw.0.iter_mut() {
        *x *= 1.1;
    }
    wrong.push(("1/sum missing", raw));
    for (name, (p, a)) in &wrong {
        assert!(
            judge(&k, p, a).is_err(),
            "planted mistake not caught: {name}"
        );
    }
    assert_eq!(wrong.len(), 8);
}
