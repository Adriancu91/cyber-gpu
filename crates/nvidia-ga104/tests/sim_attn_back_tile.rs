//! C-160 point 7 — T174, SIMULARE: mode 1 (columns) of the training attention backward "on tiles"
//! (`tools/nak/attn_back_tile.comp`, Mesa NAK 25.0.7; `docs/C-160_P7_ATENTIA_INAPOI_PE_PLACI.md`). The plan and the
//! predictions were written and pushed BEFORE this code (commit `3021024`):
//!   1. dqkv bit for bit that of `attn_back` mode 1 on every shape with d ≤ 64 (every output is the same ffma chain);
//!   2. NaN / 1e30 above the diagonal of P and dS change no bit;
//!   3. registers ≤ 64 (MISSED: 73, see below), 32 768 bytes shared, no atomics, no MUFU, no local memory;
//!   4. planted mistakes caught (grid one block short, wrong block, d > 64, grid past H, mode 0 writes nothing, a
//!      wrong dK bound).
//!
//! What the simulation can and cannot show: the simulator's stand-in (`attn_back_tile_on_cpu`) computes every element
//! with `attn_back_col_gpu`, so on its own it would only repeat the claim. The claim is checked here by a separate,
//! literal CPU emulation of the GLSL source (`emulate` below: blocks, 256 threads, the two shared tiles, the barriers
//! as phases, the same loops and bounds), compared bit for bit with `attn_back`'s mode 1; the driver side (grid, QMD
//! with 32 KiB of shared memory, cbuf, which positions get written) goes through the command processor. That the board
//! computes what the GLSL says is a prediction for the board. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::layer::{AttnShape, attn_back_gpu, attn_fwd_gpu};
use gpu_compute::math::fma_f32_ftz;
use gpu_compute::train;
use gpu_sim::cmdproc::{CommandProcessor, attn_back_on_cpu, attn_back_tile_on_cpu};
use nvidia_ga104::attn_back::{self as ab, ATTN_BACK, Mode};
use nvidia_ga104::attn_back_tile::{self as bt, ATTN_BACK_TILE};
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

// ------------------------------------------------------------------ the GLSL, emulated literally

/// Which dK bound the emulation uses: the program's (t ≤ i) or a planted mistake (t < i).
#[derive(Clone, Copy, PartialEq)]
enum Bound {
    Program,
    Planted,
}

/// `attn_back_tile.comp`, mode 1, block by block and thread by thread: the loads of a tile by all 256 threads happen
/// before any thread reads it (the barriers), the arithmetic is the source's, in its loop order. Writes into `dqkv`
/// only what the program writes. `grid_x` lets a test launch a grid one block short.
fn emulate(k: &Case, grid_x: usize, bound: Bound, dqkv: &mut [f32]) {
    let s = &k.s;
    let (t, c, d, nh) = (s.t, s.c, s.d(), s.heads);
    let row3 = 3 * c;
    let ntiles = t.div_ceil(64);
    for b in 0..s.batch {
        for h in 0..nh {
            let base = (b * nh + h) * t * t;
            for bx in 0..grid_x {
                let t0 = bx * 64;
                let mut sa = vec![0.0f32; 64 * 64];
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
                            sa[idx] = if tp < t && j <= tp {
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
                        for jj in 0..64 {
                            let j = jt * 64 + jj;
                            let kb: [f32; 4] =
                                core::array::from_fn(|kk| sb[jj * 64 + tx + 16 * kk]);
                            for a in 0..4 {
                                let r = ty + 16 * a;
                                if j <= t0 + r {
                                    let sv = sa[r * 64 + jj];
                                    for kk in 0..4 {
                                        a_th[a * 4 + kk] =
                                            fma_f32_ftz(sv, kb[kk], a_th[a * 4 + kk]);
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
                        for (lid, a_th) in acc.iter_mut().enumerate() {
                            let (tx, ty) = (lid & 15, lid >> 4);
                            for ii in 0..64 {
                                let i = it * 64 + ii;
                                let bb: [f32; 4] =
                                    core::array::from_fn(|kk| sb[ii * 64 + tx + 16 * kk]);
                                for a in 0..4 {
                                    let r = ty + 16 * a;
                                    let inside = match bound {
                                        Bound::Program => t0 + r <= i,
                                        Bound::Planted => pass == 1 || t0 + r < i,
                                    };
                                    if i < t && inside {
                                        let sv = sa[ii * 64 + r];
                                        for kk in 0..4 {
                                            a_th[a * 4 + kk] =
                                                fma_f32_ftz(sv, bb[kk], a_th[a * 4 + kk]);
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

fn emulated(k: &Case) -> Vec<f32> {
    let (nq, _, _) = k.s.sizes().unwrap();
    let mut out = vec![SENTINEL; nq];
    emulate(k, k.s.t.div_ceil(64), Bound::Program, &mut out);
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
    /// `attn_back` (blocks of 32), mode 1.
    Old,
    /// `attn_back_tile`.
    Tile,
}

/// One launch with dS / P given (mode 0 already done); returns dqkv (sentinels where nothing was written).
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
        Prog::Tile => {
            cp.register_program(ATTN_BACK_TILE, attn_back_tile_on_cpu);
            (ATTN_BACK_TILE, bt::REGISTERS, Some(bt::SHARED))
        }
    };
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
    // dS and P are inputs: never written by mode 1.
    assert_eq!(bits(&cp.mem.read_f32s(DS, np).unwrap()), bits(&k.ds));
    let out = cp.mem.read_f32s(DQKV, nq + 64).unwrap();
    assert!(out[nq..].iter().all(|&v| v == SENTINEL), "wrote after dqkv");
    Ok(out[..nq].to_vec())
}

fn run(k: &Case, prog: Prog) -> Vec<f32> {
    let a = args(k, Mode::Columns);
    let cb = a.cbuf().unwrap();
    match prog {
        Prog::Old => launch(k, prog, a.grid(), ab::BLOCK, &cb).unwrap(),
        Prog::Tile => launch(k, prog, bt::grid(&a), bt::BLOCK, &cb).unwrap(),
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

// Prediction 3 said ≤ 64 registers: MISSED, NAK gave 73 (the note). Harmless for the occupancy (DEDUCED): 73 → 80 per
// thread → 3 blocks of 256 per SM by registers, and the 32 KiB of shared memory allow 3 blocks too (SM 8.6), so the
// same 24 warps per SM as with ≤ 64. Checked here: the registers keep 3 blocks per SM.
const _: () = assert!(bt::REGISTERS.div_ceil(8) * 8 * 256 * 3 <= 65_536);
const _: () = assert!(bt::SHARED.bytes <= bt::SHARED.max_bytes);

// ------------------------------------------------------------------ the tests

#[test]
fn t174_binary_and_listing() {
    assert_eq!(hex(ATTN_BACK_TILE), bt::ATTN_BACK_TILE_SHA256);
    let txt = include_str!("../../../tools/nak/out/attn_back_tile.sm86.txt");
    assert!(txt.contains(&format!("Num GPRs: {}", bt::REGISTERS)));
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
    // 16 ffma in the dQ loop + 16 in the dK / dV loop (one per result, loops not unrolled).
    assert_eq!(count("ffma"), 32);
    assert_eq!(bt::SHARED.bytes, 32_768);
    println!(
        "T174 listing: {} registers, {} ops, {} ffma",
        bt::REGISTERS,
        ops.len(),
        count("ffma")
    );
}

/// Prediction 1: the GLSL, emulated literally, is `attn_back` mode 1 bit for bit; the simulator's launch too.
#[test]
fn t174_tile_is_attn_back_mode1_bit_for_bit() {
    for (name, k) in cases() {
        let old = k.old_dqkv();
        let em = emulated(&k);
        assert_eq!(
            bits(&em),
            bits(&old),
            "{name}: the emulated GLSL ≠ attn_back"
        );
        let tile = run(&k, Prog::Tile);
        let launched_old = run(&k, Prog::Old);
        assert_eq!(
            bits(&launched_old),
            bits(&old),
            "{name}: old launch = model"
        );
        assert_eq!(bits(&tile), bits(&old), "{name}: tile launch ≠ attn_back");
        println!(
            "T174 {name}: {} values of dqkv, bit for bit = attn_back mode 1",
            old.len()
        );
    }
}

/// Prediction 1, the chain: attn_back mode 0 → attn_back_tile against the f64 training oracle with the f64 P
/// (T159's check): every part within 2⁻¹⁶ of its largest value.
#[test]
fn t174_chain_against_the_f64_oracle() {
    for (name, k) in cases().into_iter().filter(|(n, _)| !n.starts_with("sharp")) {
        let s = &k.s;
        // P in f64 from the oracle, rounded to f32 for the board (as T159).
        let (_, p64) = train::attention_rows(s.c, s.heads, &k.qkv, s.batch, s.t).unwrap();
        let p32: Vec<f32> = p64.iter().map(|&x| x as f32).collect();
        let (ds, _) = attn_back_gpu(s, &k.qkv, &p32, &k.dout, scale(s)).unwrap();
        let kk = Case {
            s: *s,
            qkv: k.qkv.clone(),
            p: p32,
            dout: k.dout.clone(),
            ds,
        };
        let got = emulated(&kk);
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

/// Prediction 2: NaN and 1e30 above the diagonal of P and dS change no bit.
#[test]
fn t174_mask_is_the_loop_bounds() {
    let mut k = Case::new(31, shape(2, 130, 64, 2), 1.0, 1.0);
    let clean = emulated(&k);
    let t = k.s.t;
    for bh in 0..k.s.batch * k.s.heads {
        for i in 0..t {
            for j in i + 1..t {
                let at = (bh * t + i) * t + j;
                k.p[at] = if (i + j) % 2 == 0 { f32::NAN } else { 1e30 };
                k.ds[at] = if (i + j) % 2 == 0 { 1e30 } else { f32::NAN };
            }
        }
    }
    assert_eq!(bits(&emulated(&k)), bits(&clean), "emulated GLSL");
    assert_eq!(bits(&run(&k, Prog::Tile)), bits(&clean), "launch");
}

/// Prediction 4: planted mistakes are caught.
#[test]
fn t174_planted_mistakes_are_caught() {
    let k = Case::new(21, shape(2, 130, 64, 2), 1.0, 1.0);
    let a = args(&k, Mode::Columns);
    let cb = a.cbuf().unwrap();
    let g = bt::grid(&a);
    assert_eq!(g, [3, 2, 2]);
    // A grid one block short: positions 128, 129 of every (b, h) stay sentinels (3 parts × 32 columns each).
    let out = launch(&k, Prog::Tile, [g[0] - 1, g[1], g[2]], bt::BLOCK, &cb).unwrap();
    assert_eq!(
        out.iter().filter(|&&x| x == SENTINEL).count(),
        2 * 2 * 2 * 3 * 32
    );
    let (nq, _, _) = k.s.sizes().unwrap();
    let mut em = vec![SENTINEL; nq];
    emulate(&k, 2, Bound::Program, &mut em);
    assert_eq!(bits(&em), bits(&out), "the emulation, one block short");
    // The full grid writes everything.
    assert!(!run(&k, Prog::Tile).contains(&SENTINEL));
    // A wrong block, a grid past ⌈T/64⌉ or past H: refused.
    assert!(launch(&k, Prog::Tile, g, [128, 1, 1], &cb).is_err());
    assert!(launch(&k, Prog::Tile, g, [16, 16, 1], &cb).is_err());
    assert!(launch(&k, Prog::Tile, [g[0] + 1, g[1], g[2]], bt::BLOCK, &cb).is_err());
    assert!(launch(&k, Prog::Tile, [g[0], g[1] + 1, g[2]], bt::BLOCK, &cb).is_err());
    // Mode 0 in the cbuf: the program exits, nothing written; the driver side refuses it.
    let a0 = args(&k, Mode::Rows);
    assert!(!bt::shape_ok(&a0));
    assert!(bt::shape_ok(&a));
    let out0 = launch(&k, Prog::Tile, g, bt::BLOCK, &a0.cbuf().unwrap()).unwrap();
    assert!(
        out0.iter().all(|&x| x == SENTINEL),
        "mode 0 wrote something"
    );
    // d > 64: refused by the shape check and by the simulator.
    let k65 = Case::new(22, shape(1, 8, 130, 2), 1.0, 1.0);
    let a65 = args(&k65, Mode::Columns);
    assert!(!bt::shape_ok(&a65));
    assert!(
        launch(
            &k65,
            Prog::Tile,
            bt::grid(&a65),
            bt::BLOCK,
            &a65.cbuf().unwrap()
        )
        .is_err()
    );
    // A wrong dK bound (t < i instead of t ≤ i) gives other bits.
    let mut bad = vec![SENTINEL; nq];
    emulate(&k, 3, Bound::Planted, &mut bad);
    assert_ne!(
        bits(&bad),
        bits(&k.old_dqkv()),
        "the planted bound went unnoticed"
    );
}
