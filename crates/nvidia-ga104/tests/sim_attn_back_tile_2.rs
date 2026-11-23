//! C-157 point 2 after os-358 — T183, SIMULARE: `attn_back_tile_2` (`tools/nak/attn_back_tile_2.comp`, Mesa NAK 25.0.7;
//! `docs/C-157_P2_ATTN_TILE2.md`, section `attn_back_tile_2`). The plan and the predictions were written and pushed
//! BEFORE this code (`f145d37`):
//!   1. the same operations: no `mufu`, no atomics, no local memory, the same barriers as `attn_back_tile`;
//!   2. registers ≤ 80 (3 blocks of 256 per SM), 33 024 shared bytes in the QMD;
//!   3. loops: dQ below the diagonal and dK / dV right of it ≤ 30 instructions per 16 `ffma`, no `bssy`;
//!   4. dqkv = `attn_back` mode 1 (= `attn_back_tile`) bit for bit on T174's 10 shapes; planted mistakes caught (old
//!      shared size, a grid one block short, mode 0, d > 64, the `i < T` guard dropped on the last tile: −0 → +0).
//!
//! What the simulation can and cannot show: the simulator's stand-in computes every column with `attn_back_col_gpu`
//! (the model of `attn_back_tile` too), so a launch only checks the driver side. The claim "same bits" is checked by a
//! separate, literal CPU emulation of the new GLSL (`emulate` below: the 65-wide dS tile of pass dQ, the unguarded
//! tiles, the loops unrolled by 4), compared bit for bit with `attn_back` mode 1. That the board computes what the GLSL
//! says is a prediction for the board, as is the time. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::layer::{AttnShape, attn_back_gpu, attn_fwd_gpu};
use gpu_compute::math::fma_f32_ftz;
use gpu_sim::cmdproc::{CommandProcessor, Needs, attn_back_tile_on_cpu};
use nvidia_ga104::attn_back::{self as ab, Mode};
use nvidia_ga104::attn_back_tile::{self as bt, ATTN_BACK_TILE};
use nvidia_ga104::attn_back_tile_2::{self as b2, ATTN_BACK_TILE_2};
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

/// One backward case: qkv, the forward P (from the board's `attn_fwd` model), dO, and dS from `attn_back` mode 0.
struct Case {
    s: AttnShape,
    qkv: Vec<f32>,
    p: Vec<f32>,
    dout: Vec<f32>,
    ds: Vec<f32>,
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
        let (ds, _) = attn_back_gpu(&s, &qkv, &p, &dout, scale(&s)).unwrap();
        Case {
            s,
            qkv,
            p,
            dout,
            ds,
        }
    }

    /// `attn_back` mode 1 (the model of the old program), the reference of prediction 1.
    fn old_dqkv(&self) -> Vec<f32> {
        attn_back_gpu(&self.s, &self.qkv, &self.p, &self.dout, scale(&self.s))
            .unwrap()
            .1
    }
}

/// T159's cases (d ≤ 64) + T across the 64-position tiles and #513's head width.
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
    /// dK / dV: every tile right of the diagonal unguarded, the last (partial) one too.
    LastTileUnguarded,
}

fn fma16(a_th: &mut [f32; 16], a: usize, sv: f32, kb: &[f32; 4]) {
    for kk in 0..4 {
        a_th[a * 4 + kk] = fma_f32_ftz(sv, kb[kk], a_th[a * 4 + kk]);
    }
}

/// `attn_back_tile_2.comp`, mode 1, block by block and thread by thread, with the barriers as phases. Writes into
/// `dqkv` only what the program writes.
fn emulate(k: &Case, grid_x: usize, v: Variant, dqkv: &mut [f32]) {
    let s = &k.s;
    let (t, c, d, nh) = (s.t, s.c, s.d(), s.heads);
    let row3 = 3 * c;
    let ntiles = t.div_ceil(64);
    for b in 0..s.batch {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for bx in 0..grid_x {
                let t0 = bx * 64;
                let mut sa = vec![0.0f32; 64 * 65];
                let mut sb = vec![0.0f32; 64 * 64];
                // pass dQ
                let mut acc = vec![[0.0f32; 16]; 256];
                for jt in 0..=bx {
                    for lid in 0..256 {
                        for n in 0..16 {
                            let idx = lid + 256 * n;
                            let (r, x) = (idx >> 6, idx & 63);
                            let tp = t0 + r;
                            let j = jt * 64 + x;
                            sa[r * 65 + x] = if tp < t && j <= tp {
                                k.ds[base + tp * t + j]
                            } else {
                                0.0
                            };
                            let jr = jt * 64 + r;
                            sb[idx] = if jr < t && x < d {
                                k.qkv[(b * t + jr) * row3 + c + h * d + x]
                            } else {
                                0.0
                            };
                        }
                    }
                    for (lid, a_th) in acc.iter_mut().enumerate() {
                        let (tx, ty) = (lid & 15, lid >> 4);
                        if jt < bx {
                            for jj0 in (0..64).step_by(4) {
                                let (sb0, kb0) = (ty * 65 + jj0, jj0 * 64 + tx);
                                for u in 0..4 {
                                    let kb: [f32; 4] =
                                        core::array::from_fn(|kk| sb[kb0 + (64 * u + 16 * kk)]);
                                    for a in 0..4 {
                                        fma16(a_th, a, sa[sb0 + (1040 * a + u)], &kb);
                                    }
                                }
                            }
                        } else {
                            for jj in 0..64 {
                                let j = jt * 64 + jj;
                                let kb: [f32; 4] =
                                    core::array::from_fn(|kk| sb[jj * 64 + tx + 16 * kk]);
                                for a in 0..4 {
                                    let r = ty + 16 * a;
                                    if j <= t0 + r {
                                        fma16(a_th, a, sa[r * 65 + jj], &kb);
                                    }
                                }
                            }
                        }
                    }
                }
                store(&acc, b, h, t0, t, c, d, 0, dqkv);
                // passes dK (0) and dV (1)
                for pass in 0..2 {
                    let mut acc = vec![[0.0f32; 16]; 256];
                    for it in bx..ntiles {
                        for lid in 0..256 {
                            for n in 0..16 {
                                let idx = lid + 256 * n;
                                let (r, x) = (idx >> 6, idx & 63);
                                let i = it * 64 + r;
                                let tp = t0 + x;
                                let (mut sv, mut bv) = (0.0f32, 0.0f32);
                                if i < t {
                                    if tp <= i {
                                        sv = if pass == 0 {
                                            k.ds[base + i * t + tp]
                                        } else {
                                            k.p[base + i * t + tp]
                                        };
                                    }
                                    if x < d {
                                        bv = if pass == 0 {
                                            k.qkv[(b * t + i) * row3 + h * d + x]
                                        } else {
                                            k.dout[(b * t + i) * c + h * d + x]
                                        };
                                    }
                                }
                                sa[idx] = sv;
                                sb[idx] = bv;
                            }
                        }
                        let free =
                            it > bx && (it * 64 + 64 <= t || v == Variant::LastTileUnguarded);
                        for (lid, a_th) in acc.iter_mut().enumerate() {
                            let (tx, ty) = (lid & 15, lid >> 4);
                            if free {
                                for ii0 in (0..64).step_by(4) {
                                    let (bb0, sa0) = (ii0 * 64 + tx, ii0 * 64 + ty);
                                    for u in 0..4 {
                                        let bb: [f32; 4] =
                                            core::array::from_fn(|kk| sb[bb0 + (64 * u + 16 * kk)]);
                                        for a in 0..4 {
                                            fma16(a_th, a, sa[sa0 + (64 * u + 16 * a)], &bb);
                                        }
                                    }
                                }
                            } else {
                                for ii in 0..64 {
                                    let i = it * 64 + ii;
                                    let bb: [f32; 4] =
                                        core::array::from_fn(|kk| sb[ii * 64 + tx + 16 * kk]);
                                    for a in 0..4 {
                                        let r = ty + 16 * a;
                                        if i < t && t0 + r <= i {
                                            fma16(a_th, a, sa[ii * 64 + r], &bb);
                                        }
                                    }
                                }
                            }
                        }
                    }
                    store(&acc, b, h, t0, t, c, d, pass + 1, dqkv);
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn store(
    acc: &[[f32; 16]],
    b: usize,
    h: usize,
    t0: usize,
    t: usize,
    c: usize,
    d: usize,
    part: usize,
    dqkv: &mut [f32],
) {
    for (lid, a_th) in acc.iter().enumerate() {
        let (tx, ty) = (lid & 15, lid >> 4);
        for a in 0..4 {
            let tp = t0 + ty + 16 * a;
            for kk in 0..4 {
                let e = tx + 16 * kk;
                if tp < t && e < d {
                    dqkv[(b * t + tp) * 3 * c + part * c + h * d + e] = a_th[a * 4 + kk];
                }
            }
        }
    }
}

fn emulated(k: &Case, v: Variant) -> Vec<f32> {
    let (nq, _, _) = k.s.sizes().unwrap();
    let mut out = vec![SENTINEL; nq];
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
    Tile,
    Tile2,
}

/// One launch; `qmd_shared` is what the QMD declares, the simulator demands the program's own shared size and barrier.
/// Returns dqkv (sentinels where nothing was written); dS must stay untouched.
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
        Prog::Tile => (ATTN_BACK_TILE, bt::REGISTERS, bt::SHARED),
        Prog::Tile2 => (ATTN_BACK_TILE_2, b2::REGISTERS, b2::SHARED),
    };
    cp.register_program_with_needs(
        code,
        attn_back_tile_on_cpu,
        Needs {
            shared_bytes: own.bytes,
            barriers: u32::from(own.barriers),
        },
    );
    cp.mem.write(PROG, code);
    cp.mem.write_f32s(QKV, &k.qkv);
    cp.mem.write_f32s(P, &k.p);
    cp.mem.write_f32s(DOUT, &k.dout);
    cp.mem.write_f32s(DS, &k.ds);
    cp.mem.write_f32s(DQKV, &vec![SENTINEL; nq + 64]);
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
    assert_eq!(
        bits(&cp.mem.read_f32s(DS, np).unwrap()),
        bits(&k.ds),
        "mode 1 changed dS"
    );
    let out = cp.mem.read_f32s(DQKV, nq + 64).unwrap();
    assert!(out[nq..].iter().all(|&v| v == SENTINEL), "wrote after dqkv");
    Ok(out[..nq].to_vec())
}

fn run(k: &Case, prog: Prog) -> Vec<f32> {
    let a = args(k, Mode::Columns);
    let cb = a.cbuf().unwrap();
    match prog {
        Prog::Tile => launch(k, prog, bt::grid(&a), bt::BLOCK, bt::SHARED, &cb).unwrap(),
        Prog::Tile2 => launch(k, prog, b2::grid(&a), b2::BLOCK, b2::SHARED, &cb).unwrap(),
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

/// Predictions 1 and 2, read from the NAK listings.
#[test]
fn t183_binary_and_listing() {
    assert_eq!(hex(ATTN_BACK_TILE_2), b2::ATTN_BACK_TILE_2_SHA256);
    let txt = include_str!("../../../tools/nak/out/attn_back_tile_2.sm86.txt");
    let old = include_str!("../../../tools/nak/out/attn_back_tile.sm86.txt");
    assert!(txt.contains(&format!("Num GPRs: {}", b2::REGISTERS)));
    let ops = |t: &'static str| -> Vec<&'static str> {
        t.lines()
            .filter_map(|l| {
                l.split(" = ")
                    .nth(1)
                    .and_then(|r| r.split_whitespace().next())
            })
            .collect()
    };
    let (new_ops, old_ops) = (ops(txt), ops(old));
    let count = |o: &[&str], p: &str| o.iter().filter(|x| x.starts_with(p)).count();
    assert_eq!(
        count(&new_ops, "atom") + count(&new_ops, "red."),
        0,
        "atomics"
    );
    assert_eq!(count(&new_ops, "mufu"), 0, "MUFU");
    let local = txt
        .lines()
        .filter(|l| l.contains(".local") || l.contains("ldl") || l.contains("stl"))
        .count();
    assert_eq!(local, 0, "local memory in the listing");
    let bars = |t: &str| t.lines().filter(|l| l.contains("bar.sync")).count();
    assert_eq!(bars(txt), bars(old), "barriers");
    println!(
        "T183 listing: {} registers, {} ops, {} ffma (old {} / {})",
        b2::REGISTERS,
        new_ops.len(),
        count(&new_ops, "ffma"),
        bt::REGISTERS,
        count(&old_ops, "ffma")
    );
}

/// Prediction 4: the new GLSL, emulated literally, is `attn_back` mode 1 (= `attn_back_tile`) bit for bit; the launches
/// of both tile programs too.
#[test]
fn t183_tile_2_is_tile_bit_for_bit() {
    for (name, k) in cases() {
        let old = k.old_dqkv();
        assert_eq!(
            bits(&emulated(&k, Variant::Program)),
            bits(&old),
            "{name}: the emulated new GLSL ≠ attn_back mode 1"
        );
        let t1 = run(&k, Prog::Tile);
        assert_eq!(bits(&t1), bits(&old), "{name}: tile launch");
        assert_eq!(
            bits(&run(&k, Prog::Tile2)),
            bits(&t1),
            "{name}: tile_2 launch ≠ tile"
        );
        println!(
            "T183 {name}: {} values of dqkv, bit for bit = attn_back_tile",
            old.len()
        );
    }
}

/// NaN / 1e30 above the diagonal of P and dS change no bit.
#[test]
fn t183_mask_is_the_loop_bounds() {
    let mut k = Case::new(31, shape(2, 130, 64, 2), 1.0, 1.0);
    let clean = emulated(&k, Variant::Program);
    let t = k.s.t;
    for bh in 0..k.s.batch * k.s.heads {
        for i in 0..t {
            for j in i + 1..t {
                let at = (bh * t + i) * t + j;
                k.p[at] = if (i + j) % 2 == 0 { f32::NAN } else { 1e30 };
                k.ds[at] = if (i + j) % 2 == 1 { f32::NAN } else { -1e30 };
            }
        }
    }
    assert_eq!(
        bits(&emulated(&k, Variant::Program)),
        bits(&clean),
        "emulated GLSL"
    );
    assert_eq!(bits(&run(&k, Prog::Tile2)), bits(&clean), "launch");
}

/// Prediction 4: planted mistakes are caught.
#[test]
fn t183_planted_mistakes_are_caught() {
    let k = Case::new(21, shape(2, 130, 64, 2), 1.0, 1.0);
    let a = args(&k, Mode::Columns);
    let cb = a.cbuf().unwrap();
    let g = b2::grid(&a);
    assert_eq!(g, [3, 2, 2]);
    // The QMD with attn_back_tile's shared size (256 bytes short): refused.
    assert!(launch(&k, Prog::Tile2, g, b2::BLOCK, bt::SHARED, &cb).is_err());
    assert!(launch(&k, Prog::Tile2, g, b2::BLOCK, b2::SHARED, &cb).is_ok());
    // A grid one block short: positions 128, 129 of every (b, h) stay sentinels (3 × d values each).
    let out = launch(
        &k,
        Prog::Tile2,
        [g[0] - 1, g[1], g[2]],
        b2::BLOCK,
        b2::SHARED,
        &cb,
    )
    .unwrap();
    assert_eq!(
        out.iter().filter(|&&x| x == SENTINEL).count(),
        2 * 2 * 2 * 3 * 32
    );
    let (nq, _, _) = k.s.sizes().unwrap();
    let mut em = vec![SENTINEL; nq];
    emulate(&k, 2, Variant::Program, &mut em);
    assert_eq!(bits(&em), bits(&out), "the emulation, one block short");
    // A wrong block or a grid past ⌈T/64⌉: refused.
    assert!(launch(&k, Prog::Tile2, g, [128, 1, 1], b2::SHARED, &cb).is_err());
    assert!(
        launch(
            &k,
            Prog::Tile2,
            [g[0] + 1, g[1], g[2]],
            b2::BLOCK,
            b2::SHARED,
            &cb
        )
        .is_err()
    );
    // Mode 0: refused by the shape check, the program writes nothing.
    let a0 = args(&k, Mode::Rows);
    assert!(!b2::shape_ok(&a0) && b2::shape_ok(&a));
    let out0 = launch(
        &k,
        Prog::Tile2,
        g,
        b2::BLOCK,
        b2::SHARED,
        &a0.cbuf().unwrap(),
    )
    .unwrap();
    assert!(
        out0.iter().all(|&x| x == SENTINEL),
        "mode 0 wrote something"
    );
    // d > 64: refused.
    let k65 = Case::new(22, shape(1, 8, 130, 2), 1.0, 1.0);
    let a65 = args(&k65, Mode::Columns);
    assert!(!b2::shape_ok(&a65));
    assert!(
        launch(
            &k65,
            Prog::Tile2,
            b2::grid(&a65),
            b2::BLOCK,
            b2::SHARED,
            &a65.cbuf().unwrap()
        )
        .is_err()
    );
}

/// Prediction 4, the last tile keeps `i < T`: a built head (T 70, d 32) whose dK products are all negative and subnormal
/// (flushed to −0) gives dK = −0; unguarded, the rows i ≥ T of the last tile add ffma(0, 0, −0) = +0.
#[test]
fn t183_the_last_tile_keeps_its_guard() {
    let mut k = Case::new(41, shape(1, 70, 96, 3), 1.0, 1.0);
    let (t, c, d) = (k.s.t, k.s.c, k.s.d());
    for i in 0..t {
        for j in 0..=i {
            k.ds[i * t + j] = 1e-37; // head 0
        }
        for e in 0..d {
            k.qkv[i * 3 * c + e] = -0.05; // Q of head 0
        }
    }
    let old = k.old_dqkv_from_ds();
    let dk0 = |v: &[f32]| -> Vec<u32> {
        (0..t)
            .flat_map(|tp| (0..d).map(move |e| (tp, e)))
            .map(|(tp, e)| v[tp * 3 * c + c + e].to_bits())
            .collect()
    };
    assert!(
        dk0(&old).iter().all(|&x| x == 0x8000_0000),
        "the built dK is −0 in attn_back"
    );
    assert_eq!(
        bits(&emulated(&k, Variant::Program)),
        bits(&old),
        "the program keeps the −0"
    );
    assert_eq!(bits(&run(&k, Prog::Tile2)), bits(&old), "launch");
    let bad = emulated(&k, Variant::LastTileUnguarded);
    assert_ne!(
        bits(&bad),
        bits(&old),
        "the unguarded last tile went unnoticed"
    );
    assert!(
        dk0(&bad)[..64 * d].iter().all(|&x| x == 0),
        "block 0's dK became +0"
    );
}

impl Case {
    /// mode 1 of the model on this case's own dS (which a test may have edited).
    fn old_dqkv_from_ds(&self) -> Vec<f32> {
        let s = &self.s;
        let (nq, _, _) = s.sizes().unwrap();
        let mut out = vec![0.0f32; nq];
        for b in 0..s.batch {
            for h in 0..s.heads {
                for tp in 0..s.t {
                    for e in 0..s.d() {
                        let (q, kk, v) = gpu_compute::layer::attn_back_col_gpu(
                            s, &self.qkv, &self.p, &self.dout, &self.ds, b, h, tp, e,
                        );
                        let r = (b * s.t + tp) * 3 * s.c + h * s.d() + e;
                        out[r] = q;
                        out[r + s.c] = kk;
                        out[r + 2 * s.c] = v;
                    }
                }
            }
        }
        out
    }
}
