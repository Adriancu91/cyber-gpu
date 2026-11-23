//! C-157 point 2 after os-358 — T182, SIMULARE: `attn_back_tile0_2` (`tools/nak/attn_back_tile0_2.comp`, Mesa NAK
//! 25.0.7; `docs/C-157_P2_ATTN_TILE2.md`, section `attn_back_tile0_2`). The plan and the predictions were written and
//! pushed BEFORE this code (`133ce7c`):
//!   1. the same operations: no `mufu`, no atomics, no local memory, 1 barrier;
//!   2. registers ≤ 80 (3 blocks of 256 per SM), 33 024 shared bytes in the QMD;
//!   3. loops: the dP loop ≤ 28 instructions per 16 `ffma` and no `bssy`; the dot loop below the diagonal without `bssy`;
//!   4. dS = `attn_back_tile0`'s bit for bit on T175's 10 shapes; planted mistakes caught (old shared size, a grid one
//!      block short, mode 1, d > 64).
//!
//! What the simulation can and cannot show: the simulator's stand-in computes every row with `attn_back_row_gpu` (the
//! model of `attn_back_tile0` too), so a launch only checks the driver side (QMD with the new shared size, grid, block,
//! cbuf, which rows get written). The claim "same bits" is checked by a separate, literal CPU emulation of the new GLSL
//! (`emulate` below: the 64-wide dO tile, the 65-wide V and P tiles, the skewed dP tile, the loops unrolled by 4 with
//! the rest of d, the dot chain without the guard below the diagonal), compared bit for bit with `attn_back` mode 0.
//! That the board computes what the GLSL says is a prediction for the board, as is the time. Nothing here runs on the
//! RTX 3070 Ti.

use gpu_compute::layer::{AttnShape, attn_back_gpu, attn_fwd_gpu};
use gpu_compute::math::{add_f32_ftz, fma_f32_ftz};
use gpu_sim::cmdproc::{CommandProcessor, Needs, attn_back_tile0_on_cpu};
use nvidia_ga104::attn_back::{self as ab, Mode};
use nvidia_ga104::attn_back_tile0::{self as b0, ATTN_BACK_TILE0};
use nvidia_ga104::attn_back_tile0_2::{self as b2, ATTN_BACK_TILE0_2};
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

// ------------------------------------------------------------------ the new GLSL, emulated literally

/// What the emulation runs: the program, or a planted mistake.
#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Program,
    /// The guard dropped on the diagonal tile too (where it is NOT always true): P above the diagonal summed.
    NoGuardOnDiagonal,
    /// The rest loop of d skipped (only d & !3 terms).
    NoRest,
}

/// The skewed shared address of the dP tile (row, column) in `sa`.
fn sk(row: usize, col: usize) -> usize {
    row * 64 + ((col + row) & 63)
}

/// `attn_back_tile0_2.comp`, mode 0, block by block and thread by thread, with the barriers as phases.
fn emulate(k: &Case, grid_x: usize, v: Variant, ds: &mut [f32]) {
    let s = &k.s;
    let (t, c, d, nh) = (s.t, s.c, s.d(), s.heads);
    let row3 = 3 * c;
    let ntiles = t.div_ceil(64);
    let sc = scale(s);
    let d4 = d & !3;
    for b in 0..s.batch {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for bx in 0..grid_x {
                let i0 = bx * 64;
                let mut sa = vec![0.0f32; 64 * 64];
                let mut sb = vec![0.0f32; 64 * 65];
                let mut dot = [0.0f32; 64];
                for jt in 0..=bx {
                    // F3: every thread's 16 + 16 loads, then the stores (the order inside a thread does not change
                    // what lands in shared: every address is written once)
                    for lid in 0..256 {
                        let mut lo = [0.0f32; 16];
                        let mut lv = [0.0f32; 16];
                        for n in 0..16 {
                            let idx = lid + 256 * n;
                            let (r, x) = (idx >> 6, idx & 63);
                            let i = i0 + r;
                            let jr = jt * 64 + r;
                            if i < t && x < d {
                                lo[n] = k.dout[(b * t + i) * c + h * d + x];
                            }
                            if jr < t && x < d {
                                lv[n] = k.qkv[(b * t + jr) * row3 + 2 * c + h * d + x];
                            }
                        }
                        for n in 0..16 {
                            let idx = lid + 256 * n;
                            sa[idx] = lo[n];
                            sb[(idx & 63) * 65 + (idx >> 6)] = lv[n];
                        }
                    }
                    let mut acc = vec![[0.0f32; 16]; 256];
                    for (lid, a_th) in acc.iter_mut().enumerate() {
                        let (tx, ty) = (lid & 15, lid >> 4);
                        let step = |e: usize, a_th: &mut [f32; 16]| {
                            let ob: [f32; 4] = core::array::from_fn(|a| sa[(ty + 16 * a) * 64 + e]);
                            let vb: [f32; 4] = core::array::from_fn(|a| sb[e * 65 + tx + 16 * a]);
                            for a in 0..4 {
                                for kk in 0..4 {
                                    a_th[a * 4 + kk] = fma_f32_ftz(ob[a], vb[kk], a_th[a * 4 + kk]);
                                }
                            }
                        };
                        let mut e0 = 0;
                        while e0 < d4 {
                            let (ob0, vb0) = (ty * 64 + e0, e0 * 65 + tx);
                            for u in 0..4 {
                                let ob: [f32; 4] =
                                    core::array::from_fn(|a| sa[ob0 + (1024 * a + u)]);
                                let vb: [f32; 4] =
                                    core::array::from_fn(|a| sb[vb0 + (65 * u + 16 * a)]);
                                for a in 0..4 {
                                    for kk in 0..4 {
                                        a_th[a * 4 + kk] =
                                            fma_f32_ftz(ob[a], vb[kk], a_th[a * 4 + kk]);
                                    }
                                }
                            }
                            e0 += 4;
                        }
                        if v != Variant::NoRest {
                            for e in d4..d {
                                step(e, a_th);
                            }
                        }
                        for a in 0..4 {
                            let i = i0 + ty + 16 * a;
                            for kk in 0..4 {
                                let j = jt * 64 + tx + 16 * kk;
                                if i < t && (jt < bx || j <= i) {
                                    ds[base + i * t + j] = a_th[a * 4 + kk];
                                }
                            }
                        }
                    }
                    for (lid, a_th) in acc.iter().enumerate() {
                        let (tx, ty) = (lid & 15, lid >> 4);
                        for a in 0..4 {
                            for kk in 0..4 {
                                sa[sk(ty + 16 * a, tx + 16 * kk)] = a_th[a * 4 + kk];
                            }
                        }
                    }
                    for lid in 0..256 {
                        let mut lp = [0.0f32; 16];
                        for (n, l) in lp.iter_mut().enumerate() {
                            let idx = lid + 256 * n;
                            let (r, x) = (idx >> 6, idx & 63);
                            let i = i0 + r;
                            let j = jt * 64 + x;
                            if i < t && j <= i {
                                *l = k.p[base + i * t + j];
                            }
                        }
                        for (n, &l) in lp.iter().enumerate() {
                            let idx = lid + 256 * n;
                            sb[(idx >> 6) * 65 + (idx & 63)] = l;
                        }
                    }
                    for (lid, dl) in dot.iter_mut().enumerate() {
                        let pb = lid * 65;
                        let i = i0 + lid;
                        if jt < bx {
                            for jj in 0..64 {
                                *dl = fma_f32_ftz(sb[pb + jj], sa[sk(lid, jj)], *dl);
                            }
                        } else {
                            for jj in 0..64 {
                                let j = jt * 64 + jj;
                                if v == Variant::NoGuardOnDiagonal || (i < t && j <= i) {
                                    // the planted variant also sums above the diagonal (the tile holds P = 0
                                    // there; a P read without the load guard would be any value: 1 here)
                                    let pv = if v == Variant::NoGuardOnDiagonal
                                        && i < t
                                        && j < t
                                        && j > i
                                    {
                                        1.0
                                    } else {
                                        sb[pb + jj]
                                    };
                                    *dl = fma_f32_ftz(pv, sa[sk(lid, jj)], *dl);
                                }
                            }
                        }
                    }
                }
                // pass 2 (unchanged from attn_back_tile0)
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
    Tile0,
    Tile02,
}

/// One launch; `qmd_shared` is what the QMD declares, the simulator demands the program's own shared size and barrier.
/// Returns dS (sentinels where nothing was written); dqkv must stay untouched.
fn launch(
    k: &Case,
    prog: Prog,
    grid: [u32; 3],
    block: [u32; 3],
    qmd_shared: SharedMemory,
    cbuf: &[u8],
) -> Result<Vec<f32>, gpu_core::error::GpuError> {
    let (nq, _, np) = k.s.sizes().unwrap();
    let mut cp = CommandProcessor::new();
    let (code, regs, own) = match prog {
        Prog::Tile0 => (ATTN_BACK_TILE0, b0::REGISTERS, b0::SHARED),
        Prog::Tile02 => (ATTN_BACK_TILE0_2, b2::REGISTERS, b2::SHARED),
    };
    cp.register_program_with_needs(
        code,
        attn_back_tile0_on_cpu,
        Needs {
            shared_bytes: own.bytes,
            barriers: u32::from(own.barriers),
        },
    );
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
    cp.mem
        .write(QMD, &q.to_bytes_with_shared(&qmd_shared).unwrap());
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
        Prog::Tile0 => launch(k, prog, b0::grid(&a), b0::BLOCK, b0::SHARED, &cb).unwrap(),
        Prog::Tile02 => launch(k, prog, b2::grid(&a), b2::BLOCK, b2::SHARED, &cb).unwrap(),
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

// Prediction 2: ≤ 80 registers and 33 024 shared bytes keep 3 blocks of 256 per SM.
const _: () = assert!(b2::REGISTERS.div_ceil(8) * 8 * 256 * 3 <= 65_536);
const _: () = assert!(b2::REGISTERS <= 80);
const _: () = assert!(b2::SHARED.bytes == 33_024);
const _: () = assert!(3 * (b2::SHARED.bytes + 1024) <= 102_400);
const _: () = assert!(b2::SHARED.bytes <= b2::SHARED.max_bytes);

// ------------------------------------------------------------------ the tests

/// Predictions 1 and 2, read from the NAK listing.
#[test]
fn t182_binary_and_listing() {
    assert_eq!(hex(ATTN_BACK_TILE0_2), b2::ATTN_BACK_TILE0_2_SHA256);
    let txt = include_str!("../../../tools/nak/out/attn_back_tile0_2.sm86.txt");
    assert!(txt.contains(&format!("Num GPRs: {}", b2::REGISTERS)));
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
    println!(
        "T182 listing: {} registers, {} ops, {} ffma",
        b2::REGISTERS,
        ops.len(),
        count("ffma")
    );
}

/// Prediction 4: the new GLSL, emulated literally, is `attn_back` mode 0 (= `attn_back_tile0`) bit for bit; the
/// launches of both tile programs too.
#[test]
fn t182_tile0_2_is_tile0_bit_for_bit() {
    for (name, k) in cases() {
        let (old_ds, _) = k.old();
        assert_eq!(
            bits(&emulated(&k, Variant::Program)),
            bits(&old_ds),
            "{name}: the emulated new GLSL ≠ attn_back mode 0"
        );
        let t0 = run(&k, Prog::Tile0);
        assert_eq!(bits(&t0), bits(&old_ds), "{name}: tile0 launch");
        assert_eq!(
            bits(&run(&k, Prog::Tile02)),
            bits(&t0),
            "{name}: tile0_2 launch ≠ tile0"
        );
        println!(
            "T182 {name}: {} values of dS, bit for bit = attn_back_tile0",
            old_ds.len()
        );
    }
}

/// NaN / 1e30 above the diagonal of P change no bit (the loads stay guarded on the diagonal tile).
#[test]
fn t182_mask_is_the_loop_bounds() {
    let mut k = Case::new(31, shape(2, 130, 64, 2), 1.0, 1.0);
    let clean = emulated(&k, Variant::Program);
    let t = k.s.t;
    for bh in 0..k.s.batch * k.s.heads {
        for i in 0..t {
            for j in i + 1..t {
                k.p[(bh * t + i) * t + j] = if (i + j) % 2 == 0 { f32::NAN } else { 1e30 };
            }
        }
    }
    assert_eq!(
        bits(&emulated(&k, Variant::Program)),
        bits(&clean),
        "emulated GLSL"
    );
    assert_eq!(bits(&run(&k, Prog::Tile02)), bits(&clean), "launch");
}

/// Prediction 4: planted mistakes are caught.
#[test]
fn t182_planted_mistakes_are_caught() {
    let k = Case::new(21, shape(2, 130, 64, 2), 1.0, 1.0);
    let a = args(&k, Mode::Rows);
    let cb = a.cbuf().unwrap();
    let g = b2::grid(&a);
    assert_eq!(g, [3, 2, 2]);
    // The QMD with attn_back_tile0's shared size (256 bytes short): refused.
    assert!(launch(&k, Prog::Tile02, g, b2::BLOCK, b0::SHARED, &cb).is_err());
    assert!(launch(&k, Prog::Tile02, g, b2::BLOCK, b2::SHARED, &cb).is_ok());
    // A grid one block short: rows 128, 129 of every (b, h) stay sentinels (130 values each).
    let out = launch(
        &k,
        Prog::Tile02,
        [g[0] - 1, g[1], g[2]],
        b2::BLOCK,
        b2::SHARED,
        &cb,
    )
    .unwrap();
    assert_eq!(
        out.iter().filter(|&&x| x == SENTINEL).count(),
        2 * 2 * 2 * 130
    );
    let (_, _, np) = k.s.sizes().unwrap();
    let mut em = vec![SENTINEL; np];
    emulate(&k, 2, Variant::Program, &mut em);
    assert_eq!(bits(&em), bits(&out), "the emulation, one block short");
    // A wrong block or a grid past ⌈T/64⌉: refused.
    assert!(launch(&k, Prog::Tile02, g, [128, 1, 1], b2::SHARED, &cb).is_err());
    assert!(
        launch(
            &k,
            Prog::Tile02,
            [g[0] + 1, g[1], g[2]],
            b2::BLOCK,
            b2::SHARED,
            &cb
        )
        .is_err()
    );
    // Mode 1: refused by the shape check, the program writes nothing.
    let a1 = args(&k, Mode::Columns);
    assert!(!b2::shape_ok(&a1) && b2::shape_ok(&a));
    let out1 = launch(
        &k,
        Prog::Tile02,
        g,
        b2::BLOCK,
        b2::SHARED,
        &a1.cbuf().unwrap(),
    )
    .unwrap();
    assert!(
        out1.iter().all(|&x| x == SENTINEL),
        "mode 1 wrote something"
    );
    // d > 64: refused.
    let k65 = Case::new(22, shape(1, 8, 130, 2), 1.0, 1.0);
    let a65 = args(&k65, Mode::Rows);
    assert!(!b2::shape_ok(&a65));
    assert!(
        launch(
            &k65,
            Prog::Tile02,
            b2::grid(&a65),
            b2::BLOCK,
            b2::SHARED,
            &a65.cbuf().unwrap()
        )
        .is_err()
    );
    // The emulation's own mistakes: the guard dropped on the diagonal tile, the rest of d skipped.
    let old = k.old().0;
    assert_ne!(
        bits(&emulated(&k, Variant::NoGuardOnDiagonal)),
        bits(&old),
        "diagonal guard"
    );
    let k1 = Case::new(23, shape(1, 17, 3, 3), 2.0, 1.0); // d 1: everything is the rest loop
    let old1 = k1.old().0;
    assert_ne!(
        bits(&emulated(&k1, Variant::NoRest)),
        bits(&old1),
        "rest of d"
    );
}
