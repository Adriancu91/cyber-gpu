//! C-160 point 7 — T172, SIMULARE: the attention programs with blocks of 128 threads (`tools/nak/attn_fwd128.comp`,
//! `tools/nak/attn_back128.comp`, Mesa NAK 25.0.7; `docs/C-160_P7_FIR_LIMITAT.md`). The plan and the predictions were
//! written and pushed BEFORE this code (commit `b656c68`):
//!   1. on every shape, the programs of 128 give **the same bits** as those of 32 (P, att, dS, dqkv);
//!   2. registers ≤ 32 (48 warps per SM fit);
//!   3. a grid too small (rows or heads left out) is caught by the sentinels; a wrong block is refused.
//!
//! What the simulation can and cannot show: the "programs" run on the CPU (`attn_fwd128_on_cpu`,
//! `attn_back128_on_cpu` → the same `gpu_compute::layer` rows as the programs of 32), so here we check the driver side
//! (the grid, the QMD with blocks of 128, the cbuf, which rows / heads get written) and, from the NAK listings, that
//! the arithmetic of every thread is the same op sequence as in the programs of 32. That the board gives the same bits
//! is a prediction for the board, not a measurement. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::layer::AttnShape;
use gpu_core::error::GpuError;
use gpu_sim::cmdproc::{
    CommandProcessor, attn_back_on_cpu, attn_back128_on_cpu, attn_fwd_on_cpu, attn_fwd128_on_cpu,
};
use nvidia_ga104::attn_back::{self as ab, ATTN_BACK};
use nvidia_ga104::attn_back128::{self as ab128, ATTN_BACK128};
use nvidia_ga104::attn_fwd::{self as af, ATTN_FWD};
use nvidia_ga104::attn_fwd128::{self as af128, ATTN_FWD128};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::clc7c0 as c;

const FWD: u64 = 0x10_0000;
const BACK: u64 = 0x18_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
const QKV: u64 = 0x100_0000;
const P: u64 = 0x400_0000;
const ATT: u64 = 0x800_0000;
const DOUT: u64 = 0xc00_0000;
const DS: u64 = 0x1000_0000;
const DQKV: u64 = 0x1400_0000;
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

fn shape(batch: usize, t: usize, c: usize, heads: usize) -> AttnShape {
    AttnShape { batch, t, c, heads }
}

/// Which programs one run uses: blocks of 32 (the originals) or of 128.
#[derive(Clone, Copy, PartialEq)]
enum Width {
    W32,
    W128,
}

/// Grids of one run (forward, backward rows, backward columns); the tests change them to plant mistakes.
struct Grids {
    fwd: [u32; 3],
    rows: [u32; 3],
    cols: [u32; 3],
}

struct Run {
    p: Vec<f32>,
    att: Vec<f32>,
    ds: Vec<f32>,
    dqkv: Vec<f32>,
}

fn processor() -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program(ATTN_FWD, attn_fwd_on_cpu);
    cp.register_program(ATTN_BACK, attn_back_on_cpu);
    cp.register_program(ATTN_FWD128, attn_fwd128_on_cpu);
    cp.register_program(ATTN_BACK128, attn_back128_on_cpu);
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
) -> Result<(), GpuError> {
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

fn fwd_args(s: &AttnShape) -> af::Args {
    af::Args {
        qkv: QKV,
        p: P,
        att: ATT,
        batch: s.batch as u32,
        t: s.t as u32,
        c: s.c as u32,
        heads: s.heads as u32,
        scale: ab::Args::scale_for(s.d() as u32),
    }
}

fn back_args(s: &AttnShape, mode: ab::Mode) -> ab::Args {
    ab::Args {
        qkv: QKV,
        p: P,
        dout: DOUT,
        ds: DS,
        dqkv: DQKV,
        batch: s.batch as u32,
        t: s.t as u32,
        c: s.c as u32,
        heads: s.heads as u32,
        scale: ab::Args::scale_for(s.d() as u32),
        mode,
    }
}

fn grids(s: &AttnShape, w: Width) -> Grids {
    let f = fwd_args(s);
    let r = back_args(s, ab::Mode::Rows);
    let k = back_args(s, ab::Mode::Columns);
    match w {
        Width::W32 => Grids {
            fwd: f.grid(),
            rows: r.grid(),
            cols: k.grid(),
        },
        Width::W128 => Grids {
            fwd: af128::grid(&f),
            rows: ab128::grid(&r),
            cols: ab128::grid(&k),
        },
    }
}

/// Forward, backward rows, backward columns on one command processor, all buffers filled with the sentinel first
/// (plus 64 values after each, which must stay untouched).
fn run(s: &AttnShape, seed: u64, w: Width, g: &Grids, block: [u32; 3]) -> Result<Run, GpuError> {
    let (nq, no, np) = s.sizes().unwrap();
    let mut r = rng(seed);
    let qkv: Vec<f32> = (0..nq).map(|_| r() * 1.3).collect();
    let dout: Vec<f32> = (0..no).map(|_| r()).collect();
    let mut cp = processor();
    let (fwd_code, back_code, fwd_regs, back_regs) = match w {
        Width::W32 => (ATTN_FWD, ATTN_BACK, af::REGISTERS, ab::REGISTERS),
        Width::W128 => (
            ATTN_FWD128,
            ATTN_BACK128,
            af128::REGISTERS,
            ab128::REGISTERS,
        ),
    };
    cp.mem.write(FWD, fwd_code);
    cp.mem.write(BACK, back_code);
    cp.mem.write_f32s(QKV, &qkv);
    cp.mem.write_f32s(DOUT, &dout);
    for (at, n) in [(P, np), (ATT, no), (DS, np), (DQKV, nq)] {
        cp.mem.write_f32s(at, &vec![SENTINEL; n + 64]);
    }
    cp.mem.write_u32(SEM, 0);
    let f = fwd_args(s);
    launch_raw(&mut cp, FWD, fwd_regs, &f.cbuf().unwrap(), g.fwd, block, 1)?;
    let rr = back_args(s, ab::Mode::Rows);
    launch_raw(
        &mut cp,
        BACK,
        back_regs,
        &rr.cbuf().unwrap(),
        g.rows,
        block,
        2,
    )?;
    let kk = back_args(s, ab::Mode::Columns);
    launch_raw(
        &mut cp,
        BACK,
        back_regs,
        &kk.cbuf().unwrap(),
        g.cols,
        block,
        3,
    )?;
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 3);
    let mut out = Vec::new();
    for (at, n) in [(P, np), (ATT, no), (DS, np), (DQKV, nq)] {
        let v = cp.mem.read_f32s(at, n + 64).unwrap();
        assert!(v[n..].iter().all(|&x| x == SENTINEL), "wrote past a buffer");
        out.push(v[..n].to_vec());
    }
    let dqkv = out.pop().unwrap();
    let ds = out.pop().unwrap();
    let att = out.pop().unwrap();
    let p = out.pop().unwrap();
    Ok(Run { p, att, ds, dqkv })
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn sentinels(r: &Run) -> usize {
    [&r.p, &r.att, &r.ds, &r.dqkv]
        .iter()
        .map(|v| v.iter().filter(|&&x| x == SENTINEL).count())
        .sum()
}

/// T across the 32- and 128-thread blocks, H not a multiple of 4 (a block of mode 1 with idle warps), one head, the
/// small golden model (H 4, d 16) and #513's head width (d 64) with 12 heads.
fn shapes() -> Vec<(&'static str, AttnShape)> {
    vec![
        ("small golden model", shape(2, 32, 64, 4)),
        ("T 33", shape(1, 33, 32, 2)),
        ("T 129, H 5", shape(2, 129, 40, 5)),
        ("one head, T 200", shape(1, 200, 16, 1)),
        ("H 12, d 8, T 70", shape(3, 70, 96, 12)),
        ("#513 head width, T 140", shape(1, 140, 768, 12)),
        ("T 1", shape(2, 1, 64, 4)),
    ]
}

// ------------------------------------------------------------------ the tests

fn hex(code: &[u8]) -> String {
    nvidia_ga104::sha256::digest(code)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The float arithmetic of a NAK listing, op name by op name, in listing order.
fn float_ops(txt: &str) -> Vec<String> {
    txt.lines()
        .filter_map(|l| {
            let rhs = l.split(" = ").nth(1)?;
            let op = rhs.split_whitespace().next()?;
            let name = op.split('.').next().unwrap_or(op);
            matches!(
                name,
                "ffma" | "fadd" | "fmul" | "fmnmx" | "mufu" | "fsetp" | "fset"
            )
            .then(|| op.to_string())
        })
        .collect()
}

// Prediction 2 (checked when compiling).
const _: () = assert!(af128::REGISTERS <= 32 && ab128::REGISTERS <= 32);

#[test]
fn t172_binaries_and_listings() {
    assert_eq!(hex(ATTN_FWD128), af128::ATTN_FWD128_SHA256);
    assert_eq!(hex(ATTN_BACK128), ab128::ATTN_BACK128_SHA256);
    // Prediction 2: the registers still allow 48 warps per SM (64 K registers / (48 · 32 threads) = 42 → ≤ 40 after
    // NAK's rounding to 8; we predicted ≤ 32).
    let f32t = include_str!("../../../tools/nak/out/attn_fwd.sm86.txt");
    let f128t = include_str!("../../../tools/nak/out/attn_fwd128.sm86.txt");
    let b32t = include_str!("../../../tools/nak/out/attn_back.sm86.txt");
    let b128t = include_str!("../../../tools/nak/out/attn_back128.sm86.txt");
    assert!(f128t.contains(&format!("Num GPRs: {}", af128::REGISTERS)));
    assert!(b128t.contains(&format!("Num GPRs: {}", ab128::REGISTERS)));
    // Prediction 1, from the code: every thread does the same float operations, in the same order.
    assert!(!float_ops(f32t).is_empty());
    assert_eq!(
        float_ops(f128t),
        float_ops(f32t),
        "attn_fwd: the float ops changed"
    );
    assert_eq!(
        float_ops(b128t),
        float_ops(b32t),
        "attn_back: the float ops changed"
    );
    for t in [f128t, b128t] {
        assert!(!t.contains("bar.") && !t.contains("atom") && !t.contains("red."));
    }
}

#[test]
fn t172_same_bits_as_blocks_of_32() {
    for (seed, (name, s)) in shapes().into_iter().enumerate() {
        let a = run(
            &s,
            seed as u64 + 1,
            Width::W32,
            &grids(&s, Width::W32),
            af::BLOCK,
        )
        .unwrap();
        let b = run(
            &s,
            seed as u64 + 1,
            Width::W128,
            &grids(&s, Width::W128),
            af128::BLOCK,
        )
        .unwrap();
        assert_eq!(
            sentinels(&a),
            0,
            "{name}: the programs of 32 left elements unwritten"
        );
        assert_eq!(
            sentinels(&b),
            0,
            "{name}: the programs of 128 left elements unwritten"
        );
        assert_eq!(bits(&b.p), bits(&a.p), "{name}: P");
        assert_eq!(bits(&b.att), bits(&a.att), "{name}: att");
        assert_eq!(bits(&b.ds), bits(&a.ds), "{name}: dS");
        assert_eq!(bits(&b.dqkv), bits(&a.dqkv), "{name}: dqkv");
    }
}

#[test]
fn t172_grids() {
    let s = shape(2, 129, 40, 5);
    let f = fwd_args(&s);
    assert_eq!(af128::grid(&f), [2, 5, 2]);
    assert_eq!(ab128::grid(&back_args(&s, ab::Mode::Rows)), [2, 5, 2]);
    assert_eq!(ab128::grid(&back_args(&s, ab::Mode::Columns)), [2, 129, 2]);
    // #513: 12 heads → 3 blocks of 4 warps; T 512 → 4 blocks of 128 rows.
    let m = shape(12, 512, 768, 12);
    assert_eq!(af128::grid(&fwd_args(&m)), [4, 12, 12]);
    assert_eq!(ab128::grid(&back_args(&m, ab::Mode::Columns)), [3, 512, 12]);
}

#[test]
fn t172_planted_mistakes_caught() {
    let s = shape(2, 129, 40, 5);
    let good = grids(&s, Width::W128);
    let want = run(&s, 9, Width::W128, &good, af128::BLOCK).unwrap();
    assert_eq!(sentinels(&want), 0);
    // (a) rows of the forward left out: ⌈T/256⌉ instead of ⌈T/128⌉.
    let g = Grids {
        fwd: [1, 5, 2],
        ..grids(&s, Width::W128)
    };
    let r = run(&s, 9, Width::W128, &g, af128::BLOCK).unwrap();
    assert!(
        sentinels(&r) > 0 || bits(&r.p) != bits(&want.p),
        "short forward grid not caught"
    );
    // (b) rows of dS left out.
    let g = Grids {
        rows: [1, 5, 2],
        ..grids(&s, Width::W128)
    };
    let r = run(&s, 9, Width::W128, &g, af128::BLOCK).unwrap();
    assert!(sentinels(&r) > 0, "short row grid not caught");
    // (c) mode 1 with ⌊H/4⌋ blocks: head 4 never written.
    let g = Grids {
        cols: [1, 129, 2],
        ..grids(&s, Width::W128)
    };
    let r = run(&s, 9, Width::W128, &g, af128::BLOCK).unwrap();
    assert!(sentinels(&r) > 0, "head left out not caught");
    // (d) the programs of 128 launched with blocks of 32, and the old grids (too big for 128: not modelled) refused.
    assert!(run(&s, 9, Width::W128, &good, af::BLOCK).is_err());
    assert!(run(&s, 9, Width::W128, &grids(&s, Width::W32), af128::BLOCK).is_err());
    // (e) the programs of 32 with blocks of 128 refused too (nothing changed for them).
    assert!(run(&s, 9, Width::W32, &grids(&s, Width::W32), af128::BLOCK).is_err());
}
