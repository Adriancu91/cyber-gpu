//! C-160 point 4 — T161, SIMULARE: training program 8, the embeddings: `tools/nak/emb_fwd.comp` (forward,
//! `x = emb[tok] + pos[i]`, row b·T + i) and `tools/nak/emb_back.comp` (backward: the gradients of `emb_token` and
//! `emb_pozitie` by segments built on the CPU, without atomics), compiled by Mesa NAK to `tools/nak/out/emb_*.sm86.sass`
//! and launched through our pushbuffer + QMD in the command processor model. The "programs" run on the CPU
//! (`emb_fwd_on_cpu` / `emb_back_on_cpu` → `gpu_compute::layer::{emb_fwd_gpu, emb_back_gpu}`, in the order of the code
//! NAK made). The forward is one `fadd.ftz` per element, so it must equal the training oracle bit for bit
//! (`gpu_compute::train::embed_rows`); the backward is a chain of f32 additions, judged against the f64 sums of
//! `gpu_compute::train::embed_back_rows` (the very lines `loss_and_grad` uses, which gave PyTorch's gradient at the small
//! golden step, T151) within a margin written BEFORE the measurement (`bound` below), and against PyTorch's own
//! gradient of step 1 (`pas_de_aur_mic/sgd/grad_pas1`). Nothing here runs on the RTX 3070 Ti.

use std::path::{Path, PathBuf};

use gpu_compute::layer::{EmbSegments, emb_back_gpu, emb_fwd_gpu};
use gpu_compute::train::{self, Arch, Batch};
use gpu_sim::cmdproc::{CommandProcessor, Event, emb_back_on_cpu, emb_fwd_on_cpu};
use nvidia_ga104::emb_back::{self as eb, EMB_BACK, Mode};
use nvidia_ga104::emb_fwd::{self as ef, EMB_FWD};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const PROG2: u64 = 0x18_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x38_0000;
const TOK: u64 = 0x100_0000;
const EMB: u64 = 0x200_0000;
const POS: u64 = 0x400_0000;
const X: u64 = 0x500_0000;
const OFF: u64 = 0x700_0000;
const DST: u64 = 0x720_0000;
const ROWS: u64 = 0x740_0000;
const G: u64 = 0x800_0000;
const SENTINEL: f32 = -12345.0;
/// 2⁻²⁴: one rounding of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// 2⁻¹²⁶: the smallest normal f32; the program flushes anything smaller to 0 (an absolute slack per FTZ step).
const FTZ: f64 = 1.175_494_350_822_287_5e-38;

/// The small golden model (hub `integrare/kronos/pas_de_aur_mic/reteta.json`).
const MIC: Arch = Arch {
    vocab: 256,
    layers: 2,
    n_emb: 64,
    heads: 4,
    ctx: 32,
};

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

// ------------------------------------------------------------------ the margin, written before the measurement
//
// Element (row d, k) of a table gradient, with the m rows x_1..x_m of dx that used it (in the segment's order):
// the program's chain a = 0, a = fl(a + x_j) has |a − Σx| ≤ 1.01·(m − 1)·u·Σ|x_j| (the first addition is exact),
// + m·FTZ for the flushes; the oracle rounds its f64 sum once (u·|want|); in the add mode both add the head's h once
// more (2.01·u·|want| for the two roundings) + one FTZ.
fn bound(m: usize, sum_abs: f64, want: f32, add: bool) -> f64 {
    let w = (want as f64).abs();
    let mut b =
        1.01 * m.saturating_sub(1) as f64 * U * sum_abs + 2.01 * U * w + (m + 1) as f64 * FTZ;
    if add {
        b += 2.01 * U * w + FTZ;
    }
    b
}

/// The largest share of the margin used, over the token table (add mode on `head`) and the position table (write
/// mode, rows ≥ T must be 0); `Err` at the first element outside.
struct Want<'a> {
    dx: &'a [f32],
    tok: &'a [u32],
    t: usize,
    c: usize,
    vocab: usize,
    ctx: usize,
    head: &'a [f32],
}

fn judge(w: &Want, got_tok: &[f32], got_pos: &[f32]) -> Result<(f64, f64), String> {
    let (want_tok, want_pos) =
        train::embed_back_rows(w.dx, w.tok, w.t, w.c, w.vocab, w.ctx, Some(w.head)).unwrap();
    let c = w.c;
    let mut abs_tok = vec![(0usize, 0.0f64); w.vocab * c];
    let mut abs_pos = vec![(0usize, 0.0f64); w.ctx * c];
    for (r, &id) in w.tok.iter().enumerate() {
        for k in 0..c {
            let x = (w.dx[r * c + k] as f64).abs();
            let e = &mut abs_tok[id as usize * c + k];
            e.0 += 1;
            e.1 += x;
            let e = &mut abs_pos[(r % w.t) * c + k];
            e.0 += 1;
            e.1 += x;
        }
    }
    let mut worst = (0.0f64, 0.0f64);
    for (name, got, want, abs, add) in [
        ("emb_token", got_tok, &want_tok, &abs_tok, true),
        ("emb_pozitie", got_pos, &want_pos, &abs_pos, false),
    ] {
        assert_eq!(got.len(), want.len(), "{name}: length");
        for i in 0..want.len() {
            let b = bound(abs[i].0, abs[i].1, want[i], add);
            let d = (got[i] as f64 - want[i] as f64).abs();
            if d.is_nan() || d > b {
                return Err(format!(
                    "{name}[{i}]: got {} want {} |Δ| {d:e} > {b:e}",
                    got[i], want[i]
                ));
            }
            let share = if b > 0.0 { d / b } else { 0.0 };
            if add {
                worst.0 = worst.0.max(share);
            } else {
                worst.1 = worst.1.max(share);
            }
        }
    }
    Ok(worst)
}

// ------------------------------------------------------------------ launching

fn cp_both() -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program(EMB_FWD, emb_fwd_on_cpu);
    cp.register_program(EMB_BACK, emb_back_on_cpu);
    cp.mem.write(PROG, EMB_FWD);
    cp.mem.write(PROG2, EMB_BACK);
    cp.mem.write_u32(SEM, 0);
    cp
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

fn fwd_args(n: usize, c: usize, t: usize) -> ef::Args {
    ef::Args {
        tok: TOK,
        emb: EMB,
        pos: POS,
        x: X,
        batch: (n / t.max(1)) as u32,
        c: c as u32,
        t: t as u32,
    }
}

fn write_ids(cp: &mut CommandProcessor, at: u64, v: &[u32]) {
    for (i, &x) in v.iter().enumerate() {
        cp.mem.write_u32(at + 4 * i as u64, x);
    }
}

/// The forward launch over all rows; returns x and checks nothing past it was written.
fn run_fwd(
    cp: &mut CommandProcessor,
    emb: &[f32],
    pos: &[f32],
    tok: &[u32],
    t: usize,
    c: usize,
    sem: u32,
) -> Vec<f32> {
    write_ids(cp, TOK, tok);
    cp.mem.write_f32s(EMB, emb);
    cp.mem.write_f32s(POS, pos);
    let n = tok.len() * c;
    cp.mem.write_f32s(X, &vec![SENTINEL; n + 64]);
    let a = fwd_args(tok.len(), c, t);
    launch_raw(
        cp,
        PROG,
        ef::REGISTERS,
        &a.cbuf().unwrap(),
        a.grid(),
        ef::BLOCK,
        sem,
    )
    .unwrap();
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), sem);
    let x = cp.mem.read_f32s(X, n + 64).unwrap();
    assert!(x[n..].iter().all(|&v| v == SENTINEL), "wrote after x");
    x[..n].to_vec()
}

fn back_args(seg: &EmbSegments, c: usize, mode: Mode, g: u64) -> eb::Args {
    eb::Args {
        dx: X,
        off: OFF,
        dst: DST,
        rows: ROWS,
        g,
        segments: seg.len() as u32,
        c: c as u32,
        mode,
    }
}

/// One backward launch: dx at X, the segments' lists, g preloaded with `g0`; returns g.
fn run_back(
    cp: &mut CommandProcessor,
    dx: &[f32],
    seg: &EmbSegments,
    c: usize,
    mode: Mode,
    g0: &[f32],
    sem: u32,
) -> Vec<f32> {
    assert!(seg.check(dx.len() / c, g0.len() / c), "the driver's check");
    cp.mem.write_f32s(X, dx);
    write_ids(cp, OFF, &seg.off);
    write_ids(cp, DST, &seg.dst);
    write_ids(cp, ROWS, &seg.rows);
    cp.mem.write_f32s(G, g0);
    cp.mem.write_f32s(G + 4 * g0.len() as u64, &[SENTINEL; 64]);
    let a = back_args(seg, c, mode, G);
    launch_raw(
        cp,
        PROG2,
        eb::REGISTERS,
        &a.cbuf().unwrap(),
        a.grid(),
        eb::BLOCK,
        sem,
    )
    .unwrap();
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), sem);
    let g = cp.mem.read_f32s(G, g0.len() + 64).unwrap();
    assert!(
        g[g0.len()..].iter().all(|&v| v == SENTINEL),
        "wrote after g"
    );
    g[..g0.len()].to_vec()
}

/// Both tables' gradients through the board's path: the token table = the head's dW + the add launch, the position
/// table = zeroed by the driver + the write launch.
fn board_back(
    dx: &[f32],
    tok: &[u32],
    t: usize,
    c: usize,
    vocab: usize,
    ctx: usize,
    head: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let mut cp = cp_both();
    let st = EmbSegments::tokens(tok, vocab).unwrap();
    let g_tok = run_back(&mut cp, dx, &st, c, Mode::Add, head, 1);
    let sp = EmbSegments::positions(tok.len() / t, t).unwrap();
    let g_pos = run_back(&mut cp, dx, &sp, c, Mode::Write, &vec![0.0; ctx * c], 2);
    let launches = cp
        .events
        .iter()
        .filter(|e| matches!(e, Event::Launch { .. }))
        .count();
    assert_eq!(launches, 2);
    (g_tok, g_pos)
}

// ------------------------------------------------------------------ the golden step's data

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

fn golden() -> (Vec<Vec<f32>>, Vec<u32>, Vec<u32>) {
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
    (
        params,
        read_i32(&dir().join("intrari.i32")),
        read_i32(&dir().join("tinte.i32")),
    )
}

/// A random case: ids drawn from `vocab` (a small vocabulary repeats tokens), dx of size ~`scale`.
struct Case {
    tok: Vec<u32>,
    t: usize,
    c: usize,
    vocab: usize,
    ctx: usize,
    emb: Vec<f32>,
    pos: Vec<f32>,
    dx: Vec<f32>,
    head: Vec<f32>,
}

impl Case {
    fn new(
        seed: u64,
        batch: usize,
        t: usize,
        c: usize,
        vocab: usize,
        ctx: usize,
        scale: f32,
    ) -> Self {
        let mut r = rng(seed);
        let n = batch * t;
        let tok = (0..n)
            .map(|_| (((r() + 1.0) * 0.5 * vocab as f32) as usize).min(vocab - 1) as u32)
            .collect();
        Self {
            tok,
            t,
            c,
            vocab,
            ctx,
            emb: (0..vocab * c).map(|_| r() * 0.05).collect(),
            pos: (0..ctx * c).map(|_| r() * 0.02).collect(),
            dx: (0..n * c).map(|_| r() * scale).collect(),
            head: (0..vocab * c).map(|_| r() * scale * 0.1).collect(),
        }
    }

    fn want(&self) -> Want<'_> {
        Want {
            dx: &self.dx,
            tok: &self.tok,
            t: self.t,
            c: self.c,
            vocab: self.vocab,
            ctx: self.ctx,
            head: &self.head,
        }
    }
}

/// The cases of T161: the golden model's shape, C across the 64-thread block, C = 1, one text, a tiny vocabulary
/// (every token used ~100 times), a vocabulary larger than the batch (most tokens unused), ctx > T (position rows
/// never written), large and tiny gradients.
fn cases() -> Vec<(&'static str, Case)> {
    vec![
        (
            "forma modelului mic: B 2, T 32, C 64, V 256",
            Case::new(1, 2, 32, 64, 256, 32, 1e-3),
        ),
        (
            "C 70 (trece de blocul de 64), B 3, T 5",
            Case::new(2, 3, 5, 70, 40, 8, 0.1),
        ),
        ("C 1, B 4, T 9", Case::new(3, 4, 9, 1, 11, 9, 1.0)),
        (
            "un text: B 1, T 17, C 130",
            Case::new(4, 1, 17, 130, 50, 20, 0.5),
        ),
        (
            "vocabular 3: fiecare simbol ~100 de randuri",
            Case::new(5, 10, 30, 48, 3, 30, 1.0),
        ),
        (
            "vocabular 5000 > lotul",
            Case::new(6, 2, 24, 32, 5000, 64, 1e-2),
        ),
        ("gradienti mari (1e6)", Case::new(7, 4, 16, 64, 64, 16, 1e6)),
        (
            "gradienti mici (1e-30, aproape de FTZ)",
            Case::new(8, 4, 16, 64, 64, 16, 1e-30),
        ),
    ]
}

// ------------------------------------------------------------------ tests

#[test]
fn t161_binaries_are_the_recorded_builds() {
    let hex = |b: &[u8]| -> String {
        nvidia_ga104::sha256::digest(b)
            .iter()
            .map(|x| format!("{x:02x}"))
            .collect()
    };
    assert_eq!(hex(EMB_FWD), ef::EMB_FWD_SHA256);
    assert_eq!(hex(EMB_BACK), eb::EMB_BACK_SHA256);
    let fwd = include_str!("../../../tools/nak/out/emb_fwd.sm86.txt");
    let back = include_str!("../../../tools/nak/out/emb_back.sm86.txt");
    assert_eq!(
        EMB_FWD.len(),
        FWD_INSTR * 16,
        "emb_fwd instructions of 16 bytes"
    );
    assert_eq!(
        EMB_BACK.len(),
        BACK_INSTR * 16,
        "emb_back instructions of 16 bytes"
    );
    // The model's arithmetic: one fadd per element in the forward; the backward's chain + the add mode's one fadd.
    assert_eq!(
        fwd.matches("= fadd.ftz").count(),
        1,
        "one fadd.ftz (emb + pos)"
    );
    assert!(
        !fwd.contains("ffma") && !fwd.contains("fmul") && !fwd.contains("mufu"),
        "nothing else on floats, no division (the row is b·T + i from the grid)"
    );
    assert!(fwd.contains("s2r sr[0x27]"), "the text b = the block's z");
    assert_eq!(
        back.matches("= fadd.ftz").count(),
        BACK_FADD,
        "the chain and the add mode"
    );
    assert!(
        !back.contains("ffma") && !back.contains("fmul"),
        "nothing else on floats"
    );
    for t in [fwd, back] {
        assert!(
            !t.contains("bar.") && !t.contains("shfl") && !t.contains("atom"),
            "no barrier, no shuffle, no atomics"
        );
    }
    assert!(!back.contains("mufu"), "no approximate unit");
    assert!(fwd.contains(&format!("Num GPRs: {}", ef::REGISTERS)));
    assert!(back.contains(&format!("Num GPRs: {}", eb::REGISTERS)));
    assert!(
        fwd.contains("c[0x0][0x48]") && fwd.contains("c[0x0][0x4c]"),
        "C at 0x48, T at 0x4c"
    );
    assert!(
        back.contains("c[0x0][0x50]") && back.contains("c[0x0][0x54]"),
        "C at 0x50, the mode at 0x54"
    );
    // The constant buffers we write.
    let a = fwd_args(64, 64, 32);
    let cb = a.cbuf().unwrap();
    let rd64 = |cb: &[u8], o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
    let rd32 = |cb: &[u8], o: usize| u32::from_le_bytes(cb[o..o + 4].try_into().unwrap());
    assert_eq!(
        [0x28, 0x30, 0x38, 0x40].map(|o| rd64(&cb, o)),
        [TOK, EMB, POS, X]
    );
    assert_eq!([rd32(&cb, 0x48), rd32(&cb, 0x4c)], [64, 32]);
    assert!(cb[..0x28].iter().all(|&b| b == 0), "workgroup base left 0");
    assert_eq!(a.grid(), [1, 32, 2]);
    assert_eq!(fwd_args(6, 130, 3).grid(), [3, 3, 2]);
    let seg = EmbSegments::positions(2, 32).unwrap();
    let b = back_args(&seg, 70, Mode::Add, G);
    let cb = b.cbuf().unwrap();
    assert_eq!(
        [0x28, 0x30, 0x38, 0x40, 0x48].map(|o| rd64(&cb, o)),
        [X, OFF, DST, ROWS, G]
    );
    assert_eq!([rd32(&cb, 0x50), rd32(&cb, 0x54)], [70, 1]);
    assert_eq!(
        rd32(&back_args(&seg, 70, Mode::Write, G).cbuf().unwrap(), 0x54),
        0
    );
    assert!(cb[..0x28].iter().all(|&b| b == 0), "workgroup base left 0");
    assert_eq!(b.grid(), [2, 32, 1]);
}

#[test]
fn t161_forward_is_the_oracle_bit_for_bit() {
    // The real tables and batch of the small golden step.
    let (params, inputs, _) = golden();
    let mut cp = cp_both();
    let x = run_fwd(&mut cp, &params[0], &params[1], &inputs, 32, 64, 1);
    let want = train::embed_rows(&params[0], &params[1], &inputs, 32, 64).unwrap();
    assert_eq!(bits(&x), bits(&want), "golden step");
    assert_eq!(
        bits(&x),
        bits(&emb_fwd_gpu(&params[0], &params[1], &inputs, 32, 64).unwrap())
    );
    for (name, k) in cases() {
        let mut cp = cp_both();
        let x = run_fwd(&mut cp, &k.emb, &k.pos, &k.tok, k.t, k.c, 1);
        let want = train::embed_rows(&k.emb, &k.pos, &k.tok, k.t, k.c).unwrap();
        assert_eq!(bits(&x), bits(&want), "{name}");
    }
    // FTZ: a subnormal sum is flushed by the board (the oracle keeps it): the only difference allowed, ≤ 2⁻¹²⁶.
    let tiny = f32::MIN_POSITIVE / 4.0;
    let x = emb_fwd_gpu(&[tiny, 1.0], &[tiny, 0.5], &[0], 1, 2).unwrap();
    assert_eq!(x, vec![0.0, 1.5]);
    let o = train::embed_rows(&[tiny, 1.0], &[tiny, 0.5], &[0], 1, 2).unwrap();
    assert!((o[0] - x[0]).abs() as f64 <= FTZ);
}

#[test]
fn t161_backward_matches_pytorch_at_the_golden_step() {
    let (params, inputs, targets) = golden();
    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: 2,
        len: 32,
    };
    let (dx, head) = train::embed_grad_inputs(&MIC, &params, &batch).unwrap();
    let (g_tok, g_pos) = board_back(&dx, &inputs, 32, 64, 256, 32, &head);
    // Against the oracle's own lines, in the written margin.
    let w = Want {
        dx: &dx,
        tok: &inputs,
        t: 32,
        c: 64,
        vocab: 256,
        ctx: 32,
        head: &head,
    };
    let (st, sp) = judge(&w, &g_tok, &g_pos).unwrap();
    // Against PyTorch's gradient of step 1 (T151's tolerance for gradients, 1e-6 absolute).
    let pt_tok = read_f32(&dir().join("sgd/grad_pas1/emb_token.weight.f32"), 256 * 64);
    let pt_pos = read_f32(&dir().join("sgd/grad_pas1/emb_pozitie.weight.f32"), 32 * 64);
    let worst = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let (dt, dp) = (worst(&g_tok, &pt_tok), worst(&g_pos, &pt_pos));
    println!(
        "kt c161 pas_de_aur_mic emb_token {:.1}% emb_pozitie {:.1}% din margine; fata de PyTorch max|Δ| {dt:.2e} / {dp:.2e}",
        100.0 * st,
        100.0 * sp
    );
    assert!(dt <= 1e-6 && dp <= 1e-6, "PyTorch: {dt:e} {dp:e}");
    // The same gradients as the whole oracle step (loss_and_grad), within the margin.
    let (_, g) = train::loss_and_grad(&MIC, &params, &batch).unwrap();
    let (st2, sp2) = judge(&w, &g[0], &g[1]).unwrap();
    assert!(
        st2 <= 1e-9 && sp2 <= 1e-9,
        "loss_and_grad = embed_back_rows"
    );
}

#[test]
fn t161_backward_in_the_written_margin() {
    for (name, k) in cases() {
        let (g_tok, g_pos) = board_back(&k.dx, &k.tok, k.t, k.c, k.vocab, k.ctx, &k.head);
        let (st, sp) = judge(&k.want(), &g_tok, &g_pos).unwrap_or_else(|e| panic!("{name}: {e}"));
        println!(
            "kt c161 {name}: emb_token {:.1}% emb_pozitie {:.1}% din margine",
            100.0 * st,
            100.0 * sp
        );
        // The launch is the model, bit for bit.
        let mut want_tok = k.head.clone();
        let st_seg = EmbSegments::tokens(&k.tok, k.vocab).unwrap();
        emb_back_gpu(&k.dx, k.c, &st_seg, &mut want_tok, true).unwrap();
        assert_eq!(bits(&g_tok), bits(&want_tok), "{name}: model");
        // Tokens not in the batch keep the head's dW untouched; positions ≥ T stay 0.
        for id in 0..k.vocab {
            if !k.tok.contains(&(id as u32)) {
                assert_eq!(
                    bits(&g_tok[id * k.c..(id + 1) * k.c]),
                    bits(&k.head[id * k.c..(id + 1) * k.c]),
                    "{name}: unused token {id}"
                );
            }
        }
        assert!(
            g_pos[k.t * k.c..].iter().all(|&v| v.to_bits() == 0),
            "{name}: positions ≥ T"
        );
    }
}

#[test]
fn t161_segments_runs_and_refusals() {
    // The driver's lists.
    let s = EmbSegments::tokens(&[5, 2, 5, 0, 2, 5], 6).unwrap();
    assert_eq!(s.dst, vec![0, 2, 5]);
    assert_eq!(s.off, vec![0, 1, 3, 6]);
    assert_eq!(s.rows, vec![3, 1, 4, 0, 2, 5]);
    assert!(s.check(6, 6));
    assert!(!s.check(5, 6), "a row past dx");
    assert!(!s.check(6, 5), "a destination past g");
    let p = EmbSegments::positions(3, 2).unwrap();
    assert_eq!(
        (p.dst.clone(), p.off.clone(), p.rows.clone()),
        (vec![0, 1], vec![0, 3, 6], vec![0, 2, 4, 1, 3, 5])
    );
    assert!(
        EmbSegments::tokens(&[1, 6], 6).is_none(),
        "id past the vocabulary"
    );
    assert!(EmbSegments::tokens(&[], 6).is_none());
    assert!(EmbSegments::positions(0, 4).is_none());
    let mut dup = s.clone();
    dup.dst = vec![0, 2, 2];
    assert!(
        !dup.check(6, 6),
        "repeated destination (two threads on one element)"
    );
    let mut bad = s.clone();
    bad.off = vec![0, 3, 1, 6];
    assert!(!bad.check(6, 6), "offsets going back");
    bad.off = vec![0, 1, 3, 5];
    assert!(!bad.check(6, 6), "offsets not ending at rows.len()");
    bad.off = vec![1, 1, 3, 6];
    assert!(!bad.check(6, 6), "offsets not from 0");
    let mut g = vec![1.0f32; 12];
    assert!(emb_back_gpu(&[0.0; 12], 2, &dup, &mut g, true).is_none());
    assert!(g.iter().all(|&v| v == 1.0), "refused: g unchanged");
    // The forward refuses an id past the table, a short position table, an empty shape.
    assert!(emb_fwd_gpu(&[0.0; 6], &[0.0; 4], &[3], 2, 2).is_none());
    assert!(emb_fwd_gpu(&[0.0; 6], &[0.0; 2], &[1, 0], 2, 2).is_none());
    assert!(emb_fwd_gpu(&[0.0; 6], &[0.0; 4], &[], 2, 2).is_none());
    assert!(train::embed_rows(&[0.0; 6], &[0.0; 4], &[3, 0], 2, 2).is_err());
    assert!(
        train::embed_back_rows(&[0.0; 4], &[1, 0], 2, 2, 2, 1, None).is_err(),
        "T > ctx"
    );
    // Constant buffers refused: empty, T or B past the grid, a missing address.
    assert!(fwd_args(0, 4, 1).cbuf().is_none());
    assert!(fwd_args(70_000, 4, 70_000).cbuf().is_none());
    assert!(fwd_args(70_000, 4, 1).cbuf().is_none());
    assert!(
        ef::Args {
            x: 0,
            ..fwd_args(4, 4, 2)
        }
        .cbuf()
        .is_none()
    );
    let seg = EmbSegments::positions(2, 4).unwrap();
    assert!(back_args(&seg, 0, Mode::Add, G).cbuf().is_none());
    assert!(back_args(&seg, 4, Mode::Add, 0).cbuf().is_none());
    assert!(
        eb::Args {
            segments: 0,
            ..back_args(&seg, 4, Mode::Add, G)
        }
        .cbuf()
        .is_none()
    );

    // Two runs: the same bits (no atomics). Write ignores what g held; add uses it.
    let k = Case::new(21, 3, 20, 70, 7, 24, 0.3);
    let a = board_back(&k.dx, &k.tok, k.t, k.c, k.vocab, k.ctx, &k.head);
    let b = board_back(&k.dx, &k.tok, k.t, k.c, k.vocab, k.ctx, &k.head);
    assert_eq!((bits(&a.0), bits(&a.1)), (bits(&b.0), bits(&b.1)));
    let sp = EmbSegments::positions(3, 20).unwrap();
    let mut cp = cp_both();
    let junk = vec![f32::NAN; 24 * 70];
    let w = run_back(&mut cp, &k.dx, &sp, 70, Mode::Write, &junk, 1);
    assert_eq!(
        bits(&w[..20 * 70]),
        bits(&a.1[..20 * 70]),
        "write: old g (NaN) ignored"
    );
    assert!(
        w[20 * 70..].iter().all(|v| v.is_nan()),
        "rows in no segment untouched"
    );

    // A short grid: x = 1 of 2 (C = 70 → only k < 64), y = 2 of the segments → only those elements.
    let mut cp = cp_both();
    cp.mem.write_f32s(X, &k.dx);
    write_ids(&mut cp, OFF, &sp.off);
    write_ids(&mut cp, DST, &sp.dst);
    write_ids(&mut cp, ROWS, &sp.rows);
    cp.mem.write_f32s(G, &vec![SENTINEL; 24 * 70]);
    let ar = back_args(&sp, 70, Mode::Write, G);
    launch_raw(
        &mut cp,
        PROG2,
        eb::REGISTERS,
        &ar.cbuf().unwrap(),
        [1, 2, 1],
        eb::BLOCK,
        1,
    )
    .unwrap();
    let g = cp.mem.read_f32s(G, 24 * 70).unwrap();
    for (i, v) in g.iter().enumerate() {
        let (row, col) = (i / 70, i % 70);
        if row < 2 && col < 64 {
            assert_eq!(v.to_bits(), a.1[i].to_bits(), "short grid ({row}, {col})");
        } else {
            assert_eq!(*v, SENTINEL, "short grid wrote ({row}, {col})");
        }
    }
    // The same for the forward: x = 1 of 2, y = 3 positions, z = 1 text.
    let mut cp = cp_both();
    write_ids(&mut cp, TOK, &k.tok);
    cp.mem.write_f32s(EMB, &k.emb);
    cp.mem.write_f32s(POS, &k.pos);
    cp.mem.write_f32s(X, &vec![SENTINEL; 60 * 70]);
    let af_ = fwd_args(60, 70, 20);
    launch_raw(
        &mut cp,
        PROG,
        ef::REGISTERS,
        &af_.cbuf().unwrap(),
        [1, 3, 1],
        ef::BLOCK,
        1,
    )
    .unwrap();
    let x = cp.mem.read_f32s(X, 60 * 70).unwrap();
    let want = train::embed_rows(&k.emb, &k.pos, &k.tok, 20, 70).unwrap();
    for (i, v) in x.iter().enumerate() {
        let (row, col) = (i / 70, i % 70);
        if row < 3 && col < 64 {
            assert_eq!(v.to_bits(), want[i].to_bits());
        } else {
            assert_eq!(*v, SENTINEL);
        }
    }
    // A wrong block or a workgroup base ≠ 0: not modelled, refused (nothing written).
    let mut cp = cp_both();
    cp.mem.write_f32s(X, &[SENTINEL; 8]);
    assert!(
        launch_raw(
            &mut cp,
            PROG,
            ef::REGISTERS,
            &fwd_args(2, 4, 2).cbuf().unwrap(),
            [1, 2, 1],
            [32, 1, 1],
            1
        )
        .is_err()
    );
    assert!(
        launch_raw(
            &mut cp,
            PROG,
            ef::REGISTERS,
            &{
                let mut cb = fwd_args(2, 4, 2).cbuf().unwrap();
                cb[0x8] = 1;
                cb
            },
            [1, 2, 1],
            ef::BLOCK,
            1
        )
        .is_err()
    );
    assert!(
        cp.mem
            .read_f32s(X, 8)
            .unwrap()
            .iter()
            .all(|&v| v == SENTINEL)
    );
}

#[test]
fn t161_two_programs_one_processor() {
    // Forward then backward on the same command processor (the step's first and last launches of the embeddings):
    // the forward's x is exactly the oracle's, the backward's gradients are the model's.
    let k = Case::new(31, 2, 32, 64, 256, 32, 1e-3);
    let mut cp = cp_both();
    let x = run_fwd(&mut cp, &k.emb, &k.pos, &k.tok, k.t, k.c, 1);
    assert_eq!(
        bits(&x),
        bits(&train::embed_rows(&k.emb, &k.pos, &k.tok, k.t, k.c).unwrap())
    );
    let st = EmbSegments::tokens(&k.tok, k.vocab).unwrap();
    let g = run_back(&mut cp, &k.dx, &st, k.c, Mode::Add, &k.head, 2);
    let mut want = k.head.clone();
    emb_back_gpu(&k.dx, k.c, &st, &mut want, true).unwrap();
    assert_eq!(bits(&g), bits(&want));
    assert!(matches!(
        cp.events
            .iter()
            .filter(|e| matches!(e, Event::Launch { .. }))
            .count(),
        2
    ));
}

#[test]
fn t161_planted_mistakes_are_caught() {
    let k = Case::new(41, 3, 16, 40, 12, 20, 0.5);
    let c = k.c;
    let n = k.tok.len();
    let (good_tok, good_pos) = board_back(&k.dx, &k.tok, k.t, c, k.vocab, k.ctx, &k.head);
    assert!(judge(&k.want(), &good_tok, &good_pos).is_ok());
    let sum_rows = |rows: &mut dyn Iterator<Item = usize>, k_: usize| -> f32 {
        let mut a = 0.0f32;
        for r in rows {
            a += k.dx[r * c + k_];
        }
        a
    };
    let mut mistakes: Vec<(&str, Vec<f32>, Vec<f32>)> = Vec::new();
    // (1) the head forgotten (write instead of add on the token table).
    let st = EmbSegments::tokens(&k.tok, k.vocab).unwrap();
    let mut m = k.head.clone();
    emb_back_gpu(&k.dx, c, &st, &mut m, false).unwrap();
    mistakes.push(("capul uitat", m, good_pos.clone()));
    // (2) the head added twice.
    let mut m = k.head.clone();
    emb_back_gpu(&k.dx, c, &st, &mut m, true).unwrap();
    for (mi, hi) in m.iter_mut().zip(&k.head) {
        *mi += *hi;
    }
    mistakes.push(("capul de doua ori", m, good_pos.clone()));
    // (3) positions summed over only the first text.
    let mut m = vec![0.0f32; k.ctx * c];
    for i in 0..k.t {
        for k_ in 0..c {
            m[i * c + k_] = k.dx[i * c + k_];
        }
    }
    mistakes.push(("pozitiile doar din primul text", good_tok.clone(), m));
    // (4) position r instead of i (rows past T land on later positions).
    let mut m = vec![0.0f32; k.ctx * c];
    for r in 0..n.min(k.ctx) {
        for k_ in 0..c {
            m[r * c + k_] = k.dx[r * c + k_];
        }
    }
    mistakes.push(("pozitia r in loc de i", good_tok.clone(), m));
    // (5) the token one row off (id + 1).
    let mut m = k.head.clone();
    for (r, &id) in k.tok.iter().enumerate() {
        let d = (id as usize + 1) % k.vocab;
        for k_ in 0..c {
            m[d * c + k_] += k.dx[r * c + k_];
        }
    }
    mistakes.push(("simbolul deplasat cu unu", m, good_pos.clone()));
    // (6) the segment's last row dropped.
    let mut m = k.head.clone();
    for s in 0..st.len() {
        let rows = &st.rows[st.off[s] as usize..st.off[s + 1] as usize - 1];
        for k_ in 0..c {
            m[st.dst[s] as usize * c + k_] += sum_rows(&mut rows.iter().map(|&r| r as usize), k_);
        }
    }
    mistakes.push(("ultimul rand din segment pierdut", m, good_pos.clone()));
    // (7) dx read transposed inside the row (column C−1−k).
    let mut m = vec![0.0f32; k.ctx * c];
    for i in 0..k.t {
        for k_ in 0..c {
            m[i * c + k_] = sum_rows(&mut (0..n / k.t).map(|b| b * k.t + i), c - 1 - k_);
        }
    }
    mistakes.push(("coloana inversata", good_tok.clone(), m));
    // (8) the position gradient scaled by 1/B (a mean instead of a sum).
    let b = (n / k.t) as f32;
    let m: Vec<f32> = good_pos.iter().map(|v| v / b).collect();
    mistakes.push(("media in loc de suma", good_tok.clone(), m));
    for (name, t, p) in &mistakes {
        assert!(
            judge(&k.want(), t, p).is_err(),
            "planted mistake not caught: {name}"
        );
    }
    // The forward: the position r instead of i must differ from the oracle.
    let want = train::embed_rows(&k.emb, &k.pos, &k.tok, k.t, c).unwrap();
    let mut bad = want.clone();
    let r = k.t + 1;
    for k_ in 0..c {
        bad[r * c + k_] = k.emb[k.tok[r] as usize * c + k_] + k.pos[r * c + k_];
    }
    assert_ne!(bits(&bad), bits(&want), "pozitia r in loc de i (inainte)");
    println!("kt c161 greseli plantate prinse: {} + 1", mistakes.len());
}

const FWD_INSTR: usize = 42;
const BACK_INSTR: usize = 80;
/// The chain's `fadd.ftz` (from `a = 0`, NAK's `mov rZ`) and the add mode's one.
const BACK_FADD: usize = 2;
