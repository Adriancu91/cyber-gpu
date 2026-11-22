//! C-160 point 7 — T175, SIMULARE: mode 0 (the rows of dS) of the training attention backward "on tiles"
//! (`tools/nak/attn_back_tile0.comp`, Mesa NAK 25.0.7; `docs/C-160_P7_ATENTIA_INAPOI_MOD0_PE_PLACI.md`). The plan and
//! the predictions were written and pushed BEFORE this code (commit `80968c3`):
//!   1. dS bit for bit that of `attn_back` mode 0 on every shape with d ≤ 64, the sign of zeros included; so the chain
//!      `attn_back_tile0` → `attn_back_tile` is `attn_back_gpu` bit for bit;
//!   2. NaN / 1e30 above the diagonal of P change no bit; the zeros above the diagonal are +0;
//!   3. registers ≤ 80 (3 blocks of 256 per SM), 32 768 bytes shared, no atomics, no MUFU, no local memory;
//!   4. planted mistakes caught (grid one block short, wrong block, d > 64, grid past H, mode 1 writes nothing, the e
//!      loop run to 64 instead of d, the dot chain summed per tile).
//!
//! What the simulation can and cannot show: the simulator's stand-in (`attn_back_tile0_on_cpu`) computes every row with
//! `attn_back_row_gpu`, so on its own it would only repeat the claim. The claim is checked here by a separate, literal
//! CPU emulation of the GLSL source (`emulate` below: blocks, 256 threads, the two shared tiles with the program's
//! skewed addresses, the barriers as phases, the same loops and bounds, dS used as scratch exactly as the program
//! does), compared bit for bit with `attn_back`'s mode 0; the driver side (grid, QMD with 32 KiB of shared memory,
//! cbuf, which rows get written) goes through the command processor. That the board computes what the GLSL says is a
//! prediction for the board. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::layer::{AttnShape, attn_back_col_gpu, attn_back_gpu, attn_fwd_gpu};
use gpu_compute::math::{add_f32_ftz, fma_f32_ftz};
use gpu_compute::train;
use gpu_sim::cmdproc::{CommandProcessor, attn_back_on_cpu, attn_back_tile0_on_cpu};
use nvidia_ga104::attn_back::{self as ab, ATTN_BACK, Mode};
use nvidia_ga104::attn_back_tile0::{self as b0, ATTN_BACK_TILE0};
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
const DOUT: u64 = 0x800_0000;
const DS: u64 = 0xc00_0000;
const DQKV: u64 = 0x1000_0000;
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

/// One backward case: qkv, the forward P (from the board's `attn_fwd` model), dO.
struct Case {
    s: AttnShape,
    qkv: Vec<f32>,
    p: Vec<f32>,
    dout: Vec<f32>,
}

fn shape(batch: usize, t: usize, c: usize, heads: usize) -> AttnShape {
    AttnShape { batch, t, c, heads }
}

fn scale(s: &AttnShape) -> f32 {
    ab::Args::scale_for(s.d() as u32)
}

impl Case {
    fn new(seed: u64, s: AttnShape, amp: f32, gscale: f32) -> Case {
        let (nq, no, _) = s.sizes().unwrap();
        let mut r = rng(seed);
        let qkv: Vec<f32> = (0..nq).map(|_| r() * amp).collect();
        let dout: Vec<f32> = (0..no).map(|_| r() * gscale).collect();
        let (p, _) = attn_fwd_gpu(&s, &qkv, scale(&s)).unwrap();
        Case { s, qkv, p, dout }
    }

    /// `attn_back` (the model of the old program): (dS of mode 0, dqkv of mode 1), the reference of prediction 1.
    fn old(&self) -> (Vec<f32>, Vec<f32>) {
        attn_back_gpu(&self.s, &self.qkv, &self.p, &self.dout, scale(&self.s)).unwrap()
    }
}

/// T174's cases (d ≤ 64): T across the 64-row tiles and #513's head width.
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
        ("T5 C8 H1 B3", Case::new(4, shape(3, 5, 8, 1), 2.0, 1.0)),
        (
            "T17 C3 H3 (d 1)",
            Case::new(5, shape(1, 17, 3, 3), 2.0, 1.0),
        ),
        ("sharp softmax", Case::new(6, shape(1, 40, 64, 4), 6.0, 1.0)),
        (
            "small gradients",
            Case::new(7, shape(2, 32, 64, 4), 1.5, 1e-4),
        ),
        ("T129 C32 H2", Case::new(8, shape(1, 129, 32, 2), 1.0, 1.0)),
        (
            "#513 head width, T 200 (d 64)",
            Case::new(9, shape(1, 200, 768, 12), 1.0, 1.0),
        ),
        (
            "T 64 exactly",
            Case::new(10, shape(2, 64, 128, 2), 1.0, 1.0),
        ),
    ]
}

// ------------------------------------------------------------------ the GLSL, emulated literally

/// What the emulation runs: the program, or a planted mistake.
#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Program,
    /// The e loop run to 64 instead of d (extra ffma(0, 0, acc) terms: −0 becomes +0).
    LoopTo64,
    /// The dot chain summed per tile from 0, then added to the running dot.
    DotPerTile,
}

/// The program's skewed shared address of (row, column).
fn sk(row: usize, col: usize) -> usize {
    row * 64 + ((col + row) & 63)
}

/// `attn_back_tile0.comp`, mode 0, block by block and thread by thread: what all 256 threads write into a tile happens
/// before any thread reads it (the barriers), the arithmetic is the source's, in its loop order, and dS (global) is
/// used as the program uses it: dP written in pass 1, read back by the same thread in pass 2. `grid_x` lets a test
/// launch a grid one block short.
fn emulate(k: &Case, grid_x: usize, v: Variant, ds: &mut [f32]) {
    let s = &k.s;
    let (t, c, d, nh) = (s.t, s.c, s.d(), s.heads);
    let row3 = 3 * c;
    let ntiles = t.div_ceil(64);
    let sc = scale(s);
    for b in 0..s.batch {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for bx in 0..grid_x {
                let i0 = bx * 64;
                let mut sa = vec![0.0f32; 64 * 64];
                let mut sb = vec![0.0f32; 64 * 64];
                let mut dot = [0.0f32; 64];
                for jt in 0..=bx {
                    // loads of the dO and V tiles
                    for lid in 0..256 {
                        for n in 0..16 {
                            let idx = lid + 256 * n;
                            let (r, x) = (idx >> 6, idx & 63);
                            let i = i0 + r;
                            let jr = jt * 64 + r;
                            sa[idx] = if i < t && x < d {
                                k.dout[(b * t + i) * c + h * d + x]
                            } else {
                                0.0
                            };
                            sb[sk(x, r)] = if jr < t && x < d {
                                k.qkv[(b * t + jr) * row3 + 2 * c + h * d + x]
                            } else {
                                0.0
                            };
                        }
                    }
                    // 4 x 4 dP per thread; written into dS for j <= i < T
                    let elen = if v == Variant::LoopTo64 { 64 } else { d };
                    let mut acc = vec![[0.0f32; 16]; 256];
                    for (lid, a_th) in acc.iter_mut().enumerate() {
                        let (tx, ty) = (lid & 15, lid >> 4);
                        for e in 0..elen {
                            let ob: [f32; 4] = core::array::from_fn(|a| sa[(ty + 16 * a) * 64 + e]);
                            let vb: [f32; 4] = core::array::from_fn(|kk| sb[sk(e, tx + 16 * kk)]);
                            for a in 0..4 {
                                for kk in 0..4 {
                                    a_th[a * 4 + kk] = fma_f32_ftz(ob[a], vb[kk], a_th[a * 4 + kk]);
                                }
                            }
                        }
                        for a in 0..4 {
                            let i = i0 + ty + 16 * a;
                            for kk in 0..4 {
                                let j = jt * 64 + tx + 16 * kk;
                                if i < t && j <= i {
                                    ds[base + i * t + j] = a_th[a * 4 + kk];
                                }
                            }
                        }
                    }
                    // the dP tile and the P tile into shared
                    for (lid, a_th) in acc.iter().enumerate() {
                        let (tx, ty) = (lid & 15, lid >> 4);
                        for a in 0..4 {
                            for kk in 0..4 {
                                sa[sk(ty + 16 * a, tx + 16 * kk)] = a_th[a * 4 + kk];
                            }
                        }
                    }
                    for lid in 0..256 {
                        for n in 0..16 {
                            let idx = lid + 256 * n;
                            let (r, x) = (idx >> 6, idx & 63);
                            let i = i0 + r;
                            let j = jt * 64 + x;
                            sb[sk(r, x)] = if i < t && j <= i {
                                k.p[base + i * t + j]
                            } else {
                                0.0
                            };
                        }
                    }
                    // threads 0..63: the dot chain of their row
                    for (lid, dl) in dot.iter_mut().enumerate() {
                        let i = i0 + lid;
                        let mut part = 0.0f32;
                        for jj in 0..64 {
                            let j = jt * 64 + jj;
                            if i < t && j <= i {
                                let (pv, dp) = (sb[sk(lid, jj)], sa[sk(lid, jj)]);
                                if v == Variant::DotPerTile {
                                    part = fma_f32_ftz(pv, dp, part);
                                } else {
                                    *dl = fma_f32_ftz(pv, dp, *dl);
                                }
                            }
                        }
                        if v == Variant::DotPerTile {
                            *dl = add_f32_ftz(*dl, part);
                        }
                    }
                }
                // pass 2: dot through shared, every thread rescales its own values
                sa[..64].copy_from_slice(&dot);
                for lid in 0..256 {
                    let (tx, ty) = (lid & 15, lid >> 4);
                    let dr: [f32; 4] = core::array::from_fn(|a| sa[ty + 16 * a]);
                    for jt in 0..ntiles {
                        for (a, &dra) in dr.iter().enumerate() {
                            let i = i0 + ty + 16 * a;
                            for kk in 0..4 {
                                let j = jt * 64 + tx + 16 * kk;
                                if i < t && j < t {
                                    let mut y = 0.0f32;
                                    if j <= i {
                                        let x = add_f32_ftz(ds[base + i * t + j], -dra);
                                        y = mul_ftz(mul_ftz(k.p[base + i * t + j], x), sc);
                                    }
                                    ds[base + i * t + j] = y;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

/// `fmul.ftz`: the product rounded once, subnormal results flushed to a zero of their sign (as the model's).
fn mul_ftz(a: f32, b: f32) -> f32 {
    fma_f32_ftz(a, b, -0.0)
}

fn emulated(k: &Case, v: Variant) -> Vec<f32> {
    let (_, _, np) = k.s.sizes().unwrap();
    let mut out = vec![SENTINEL; np];
    emulate(k, k.s.t.div_ceil(64), v, &mut out);
    out
}

// ------------------------------------------------------------------ launching (pushbuffer + QMD)

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
        scale: scale(&k.s),
        mode,
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Prog {
    /// `attn_back` (blocks of 32), mode 0.
    Old,
    /// `attn_back_tile0`.
    Tile0,
}

/// One launch; returns dS (sentinels where nothing was written). dqkv must stay untouched.
fn launch(
    k: &Case,
    prog: Prog,
    grid: [u32; 3],
    block: [u32; 3],
    cbuf: &[u8],
) -> Result<Vec<f32>, gpu_core::error::GpuError> {
    let (nq, _, np) = k.s.sizes().unwrap();
    let mut cp = CommandProcessor::new();
    let (code, regs, shared): (&[u8], u32, Option<SharedMemory>) = match prog {
        Prog::Old => {
            cp.register_program(ATTN_BACK, attn_back_on_cpu);
            (ATTN_BACK, ab::REGISTERS, None)
        }
        Prog::Tile0 => {
            cp.register_program(ATTN_BACK_TILE0, attn_back_tile0_on_cpu);
            (ATTN_BACK_TILE0, b0::REGISTERS, Some(b0::SHARED))
        }
    };
    cp.mem.write(PROG, code);
    cp.mem.write_f32s(QKV, &k.qkv);
    cp.mem.write_f32s(P, &k.p);
    cp.mem.write_f32s(DOUT, &k.dout);
    cp.mem.write_f32s(DS, &vec![SENTINEL; np + 64]);
    cp.mem.write_f32s(DQKV, &vec![SENTINEL; nq]);
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
    // dqkv is mode 1's output: never written by mode 0.
    assert!(
        cp.mem
            .read_f32s(DQKV, nq)
            .unwrap()
            .iter()
            .all(|&v| v == SENTINEL),
        "mode 0 wrote dqkv"
    );
    let out = cp.mem.read_f32s(DS, np + 64).unwrap();
    assert!(out[np..].iter().all(|&v| v == SENTINEL), "wrote after dS");
    Ok(out[..np].to_vec())
}

fn run(k: &Case, prog: Prog) -> Vec<f32> {
    let a = args(k, Mode::Rows);
    let cb = a.cbuf().unwrap();
    match prog {
        Prog::Old => launch(k, prog, a.grid(), ab::BLOCK, &cb).unwrap(),
        Prog::Tile0 => launch(k, prog, b0::grid(&a), b0::BLOCK, &cb).unwrap(),
    }
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

// Prediction 3: ≤ 80 registers keep 3 blocks of 256 per SM (64 K registers, allocated by 8), like the 32 KiB shared.
const _: () = assert!(b0::REGISTERS.div_ceil(8) * 8 * 256 * 3 <= 65_536);
const _: () = assert!(b0::SHARED.bytes <= b0::SHARED.max_bytes);
const _: () = assert!(b0::REGISTERS <= 80);

// ------------------------------------------------------------------ the tests

#[test]
fn t175_binary_and_listing() {
    assert_eq!(hex(ATTN_BACK_TILE0), b0::ATTN_BACK_TILE0_SHA256);
    let txt = include_str!("../../../tools/nak/out/attn_back_tile0.sm86.txt");
    assert!(txt.contains(&format!("Num GPRs: {}", b0::REGISTERS)));
    let ops: Vec<&str> = txt
        .lines()
        .filter_map(|l| {
            l.split(" = ")
                .nth(1)
                .and_then(|r| r.split_whitespace().next())
        })
        .collect();
    let count = |p: &str| ops.iter().filter(|o| o.starts_with(p)).count();
    assert_eq!(count("atom") + count("red."), 0, "atomics in the listing");
    assert_eq!(count("mufu"), 0, "MUFU in the listing");
    let local = txt
        .lines()
        .filter(|l| l.contains(".local") || l.contains("ldl") || l.contains("stl"))
        .count();
    assert_eq!(local, 0, "local memory in the listing");
    // 16 ffma in the dP loop (one per result, the e loop not unrolled) + 1 in the dot chain.
    assert_eq!(count("ffma"), 17);
    assert_eq!(b0::SHARED.bytes, 32_768);
    println!(
        "T175 listing: {} registers, {} ops, {} ffma",
        b0::REGISTERS,
        ops.len(),
        count("ffma")
    );
}

/// Prediction 1: the GLSL, emulated literally, is `attn_back` mode 0 bit for bit; the simulator's launch too; and the
/// emulated dS fed to mode 1 gives `attn_back`'s dqkv bit for bit.
#[test]
fn t175_tile0_is_attn_back_mode0_bit_for_bit() {
    for (name, k) in cases() {
        let (old_ds, old_dqkv) = k.old();
        let em = emulated(&k, Variant::Program);
        assert_eq!(
            bits(&em),
            bits(&old_ds),
            "{name}: the emulated GLSL ≠ attn_back"
        );
        let launched_old = run(&k, Prog::Old);
        assert_eq!(
            bits(&launched_old),
            bits(&old_ds),
            "{name}: old launch = model"
        );
        assert_eq!(
            bits(&run(&k, Prog::Tile0)),
            bits(&old_ds),
            "{name}: tile0 launch ≠ attn_back"
        );
        // the chain into mode 1 (whose tile program is T174's)
        let s = &k.s;
        let (nq, _, _) = s.sizes().unwrap();
        let mut dqkv = vec![0.0f32; nq];
        for b in 0..s.batch {
            for h in 0..s.heads {
                for tp in 0..s.t {
                    for e in 0..s.d() {
                        let (q, kk, v) =
                            attn_back_col_gpu(s, &k.qkv, &k.p, &k.dout, &em, b, h, tp, e);
                        let r = (b * s.t + tp) * 3 * s.c + h * s.d() + e;
                        dqkv[r] = q;
                        dqkv[r + s.c] = kk;
                        dqkv[r + 2 * s.c] = v;
                    }
                }
            }
        }
        assert_eq!(bits(&dqkv), bits(&old_dqkv), "{name}: the chain's dqkv");
        println!(
            "T175 {name}: {} values of dS, bit for bit = attn_back mode 0",
            old_ds.len()
        );
    }
}

/// Prediction 1, against the f64 training oracle with the f64 P (T159's check): dS in the emulation, then mode 1 →
/// every part of dqkv within 2⁻¹⁶ of its largest value.
#[test]
fn t175_chain_against_the_f64_oracle() {
    for (name, k) in cases().into_iter().filter(|(n, _)| !n.starts_with("sharp")) {
        let s = &k.s;
        let (_, p64) = train::attention_rows(s.c, s.heads, &k.qkv, s.batch, s.t).unwrap();
        let p32: Vec<f32> = p64.iter().map(|&x| x as f32).collect();
        let kk = Case {
            s: *s,
            qkv: k.qkv.clone(),
            p: p32,
            dout: k.dout.clone(),
        };
        let ds = emulated(&kk, Variant::Program);
        let (_, got) = attn_back_gpu(s, &kk.qkv, &kk.p, &kk.dout, scale(s)).unwrap();
        // the dqkv above used attn_back's own mode 0; check it saw the same dS as the emulation
        let (ds_old, _) = kk.old();
        assert_eq!(bits(&ds), bits(&ds_old), "{name}");
        let want =
            train::attention_back_rows(s.c, s.heads, &k.dout, &k.qkv, &p64, s.batch, s.t).unwrap();
        let c = s.c;
        for part in 0..3 {
            let pick = |v: &[f32]| -> Vec<f64> {
                v.chunks(3 * c)
                    .flat_map(|r| r[part * c..(part + 1) * c].iter().map(|&x| x as f64))
                    .collect()
            };
            let (g, w) = (pick(&got), pick(&want));
            let big = w.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let worst = g
                .iter()
                .zip(&w)
                .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
            assert!(
                worst <= big / 65536.0,
                "{name}: part {part} off by {worst:e} (largest {big:e})"
            );
        }
    }
}

/// Prediction 2: NaN and 1e30 above the diagonal of P change no bit; the zeros there are +0.
#[test]
fn t175_mask_is_the_loop_bounds() {
    let mut k = Case::new(31, shape(2, 130, 64, 2), 1.0, 1.0);
    let clean = emulated(&k, Variant::Program);
    let t = k.s.t;
    for bh in 0..k.s.batch * k.s.heads {
        for i in 0..t {
            for j in i + 1..t {
                let at = (bh * t + i) * t + j;
                k.p[at] = if (i + j) % 2 == 0 { f32::NAN } else { 1e30 };
                assert_eq!(clean[at].to_bits(), 0, "+0 above the diagonal");
            }
        }
    }
    assert_eq!(
        bits(&emulated(&k, Variant::Program)),
        bits(&clean),
        "emulated GLSL"
    );
    assert_eq!(bits(&run(&k, Prog::Tile0)), bits(&clean), "launch");
}

/// Prediction 4: planted mistakes are caught.
#[test]
fn t175_planted_mistakes_are_caught() {
    let k = Case::new(21, shape(2, 130, 64, 2), 1.0, 1.0);
    let a = args(&k, Mode::Rows);
    let cb = a.cbuf().unwrap();
    let g = b0::grid(&a);
    assert_eq!(g, [3, 2, 2]);
    // A grid one block short: rows 128, 129 of every (b, h) stay sentinels (130 values each).
    let out = launch(&k, Prog::Tile0, [g[0] - 1, g[1], g[2]], b0::BLOCK, &cb).unwrap();
    assert_eq!(
        out.iter().filter(|&&x| x == SENTINEL).count(),
        2 * 2 * 2 * 130
    );
    let (_, _, np) = k.s.sizes().unwrap();
    let mut em = vec![SENTINEL; np];
    emulate(&k, 2, Variant::Program, &mut em);
    assert_eq!(bits(&em), bits(&out), "the emulation, one block short");
    // The full grid writes everything.
    assert!(!run(&k, Prog::Tile0).contains(&SENTINEL));
    // A wrong block, a grid past ⌈T/64⌉ or past H: refused.
    assert!(launch(&k, Prog::Tile0, g, [128, 1, 1], &cb).is_err());
    assert!(launch(&k, Prog::Tile0, g, [16, 16, 1], &cb).is_err());
    assert!(launch(&k, Prog::Tile0, [g[0] + 1, g[1], g[2]], b0::BLOCK, &cb).is_err());
    assert!(launch(&k, Prog::Tile0, [g[0], g[1] + 1, g[2]], b0::BLOCK, &cb).is_err());
    // Mode 1 in the cbuf: the program exits, nothing written; the driver side refuses it.
    let a1 = args(&k, Mode::Columns);
    assert!(!b0::shape_ok(&a1));
    assert!(b0::shape_ok(&a));
    let out1 = launch(&k, Prog::Tile0, g, b0::BLOCK, &a1.cbuf().unwrap()).unwrap();
    assert!(
        out1.iter().all(|&x| x == SENTINEL),
        "mode 1 wrote something"
    );
    // d > 64: refused by the shape check and by the simulator.
    let k65 = Case::new(22, shape(1, 8, 130, 2), 1.0, 1.0);
    let a65 = args(&k65, Mode::Rows);
    assert!(!b0::shape_ok(&a65));
    assert!(
        launch(
            &k65,
            Prog::Tile0,
            b0::grid(&a65),
            b0::BLOCK,
            &a65.cbuf().unwrap()
        )
        .is_err()
    );
    // The dot chain summed per tile: other bits (T 130 crosses three tiles).
    assert_ne!(
        bits(&emulated(&k, Variant::DotPerTile)),
        bits(&k.old().0),
        "the per-tile dot went unnoticed"
    );
}

/// Prediction 4, the e loop: run to 64 instead of d, it adds ffma(0, 0, acc) terms that turn a −0 into +0. A built row
/// (d 32) whose products are all negative and subnormal (flushed to −0 by FTZ) gives dP = −0, dot = +0, dS = −0 there;
/// the program keeps those bits, the planted loop does not.
#[test]
fn t175_the_e_loop_stops_at_d() {
    let mut k = Case::new(41, shape(1, 70, 96, 3), 1.0, 1.0);
    let (t, c, d) = (k.s.t, k.s.c, k.s.d());
    let i = 66; // in the second row tile
    for e in 0..d {
        k.dout[i * c + e] = 1e-37; // head 0
    }
    for j in 0..t {
        for e in 0..d {
            k.qkv[j * 3 * c + 2 * c + e] = -0.05; // V of head 0: every product −5e-39, subnormal
        }
    }
    let (old_ds, _) = k.old();
    let row = &old_ds[i * t..i * t + t];
    assert!(
        row[..=i].iter().all(|&x| x.to_bits() == 0x8000_0000),
        "the built row is −0 in attn_back"
    );
    let em = emulated(&k, Variant::Program);
    assert_eq!(bits(&em), bits(&old_ds), "the program keeps the −0");
    assert_eq!(bits(&run(&k, Prog::Tile0)), bits(&old_ds), "launch");
    let bad = emulated(&k, Variant::LoopTo64);
    assert_ne!(bits(&bad), bits(&old_ds), "the loop to 64 went unnoticed");
    assert!(bad[i * t..i * t + i + 1].iter().all(|&x| x.to_bits() == 0));
}
