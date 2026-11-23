//! C-157 point 2 after os-364 — T190, SIMULARE: `attn_fwd_tile3` (`tools/nak/attn_fwd_tile3.comp`, Mesa NAK 25.0.7;
//! `docs/C-157_P2_ATTN_TILE3.md`). The plan and the predictions were written and pushed BEFORE this code (`3bc3308`):
//!   1. the same operations: the same MUFU as `attn_fwd_tile2` (16 `ex2` + 1 `rcp`), the same number of `ffma`, no
//!      atomics, no local memory;
//!   2. registers ≤ 128 (2 blocks of 256 per SM, as `_2`), 37 888 shared bytes in the QMD;
//!   3. loops: ≤ 4 memory latencies in phase A's tile loop and in phase B, ≤ 8 in phase C's outer loop (checked with
//!      `tools/buget-antrenare/nak_bucle.py`, written in the note);
//!   4. P and att = `attn_fwd_tile_gpu` (= `attn_fwd_tile2`) bit for bit on T181's 10 shapes; garbage (NaN, 1e30) in P
//!      before the launch changes no bit; planted mistakes caught.
//!
//! As in T181: the simulator's stand-in computes every row with the CPU model, so a launch only checks the driver
//! side; the claim "same bits" is checked by a literal CPU emulation of the new GLSL (`emulate`: clamped loads, then
//! selects, block by block with the barriers as phases; `exp` = `ex2(x · log2e)` as the model writes NAK's lowering).
//! That the board computes what the GLSL says is a prediction for the board, as is the time. Nothing here runs on the
//! RTX 3070 Ti.

// The emulation indexes per-thread arrays by the thread id `lid`, as the GLSL does (clarity over iterators).
#![allow(clippy::needless_range_loop)]

use gpu_compute::layer::{ATTN_FWD_LOG2E, AttnShape, attn_fwd_tile_gpu, ex2_ftz};
use gpu_compute::math::{add_f32_ftz, fma_f32_ftz, ftz};
use gpu_sim::cmdproc::{CommandProcessor, Needs, attn_fwd_tile_on_cpu};
use nvidia_ga104::attn_back as ab;
use nvidia_ga104::attn_fwd as af;
use nvidia_ga104::attn_fwd_tile as at;
use nvidia_ga104::attn_fwd_tile2::{self as at2, ATTN_FWD_TILE2};
use nvidia_ga104::attn_fwd_tile3::{self as at3, ATTN_FWD_TILE3};
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
    /// The CPU model (P, att): the reference of prediction 4.
    fn model(&self) -> (Vec<f32>, Vec<f32>) {
        attn_fwd_tile_gpu(&self.s, &self.qkv, self.scale()).unwrap()
    }
}

fn shape(batch: usize, t: usize, c: usize, heads: usize) -> AttnShape {
    AttnShape { batch, t, c, heads }
}

/// T173 / T181's 10 shapes (the same seeds and amplitudes).
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

// ------------------------------------------------------------------ the new GLSL, emulated literally

/// What the emulation runs: the program, or a planted mistake.
#[derive(Clone, Copy, PartialEq)]
enum Variant {
    Program,
    /// Phase B adds every exp to the row sum (no select on `i < T && j <= i`).
    NoSelectInSum,
    /// Phase C on the diagonal tile keeps `v · inv` where `j > i` (no select).
    NoSelectOnDiagonal,
    /// The Q / K / V tiles keep the value read at the clamped address where the condition is false (no select).
    NoSelectOnTiles,
}

fn mul_ftz(a: f32, b: f32) -> f32 {
    ftz(ftz(a) * ftz(b))
}

/// GLSL `exp(x)` as the model writes NAK's lowering: `ex2(x · log2e)`.
fn exp_nak(x: f32) -> f32 {
    ex2_ftz(mul_ftz(x, f32::from_bits(ATTN_FWD_LOG2E)))
}

/// `attn_fwd_tile3.comp`, block by block and thread by thread, with the barriers as phases. Every global load reads
/// the clamped address the GLSL computes (an out-of-range index would panic here), then selects. `p` holds what the
/// P buffer holds before the launch (garbage allowed); blocks with `bx >= grid_x` are not run.
fn emulate(k: &Case, grid_x: usize, v: Variant, p: &mut [f32], att: &mut [f32]) {
    let (t, cc, heads, d) = (k.s.t, k.s.c, k.s.heads, k.s.d());
    let row3 = 3 * cc;
    let scale = k.scale();
    let qkv = &k.qkv;
    let ntiles = t.div_ceil(64);
    let tile_sel = |cond: bool, x: f32| {
        if cond || v == Variant::NoSelectOnTiles {
            x
        } else {
            0.0
        }
    };
    for b in 0..k.s.batch {
        for h in 0..heads {
            let pb = (b * heads + h) * t;
            for bx in 0..grid_x {
                let i0 = bx * 64;
                let mut qs = vec![0.0f32; 64 * 65];
                let mut ks = vec![0.0f32; 64 * 65];
                // Load the tile at column offset `off` of qkv, rows `base + r`, into `dst` (16 loads, then selects).
                let load_tile = |dst: &mut [f32], base: usize, off: usize| {
                    for lid in 0..256 {
                        let mut ld = [0.0f32; 16];
                        for (n, l) in ld.iter_mut().enumerate() {
                            let idx = lid + 256 * n;
                            let (r, e) = (idx >> 6, idx & 63);
                            let row = (base + r).min(t - 1);
                            *l = qkv[(b * t + row) * row3 + off + h * d + e.min(d - 1)];
                        }
                        for (n, &l) in ld.iter().enumerate() {
                            let idx = lid + 256 * n;
                            let (r, e) = (idx >> 6, idx & 63);
                            dst[r * 65 + e] = tile_sel(base + r < t && e < d, l);
                        }
                    }
                };
                // phase A
                load_tile(&mut qs, i0, 0);
                let mut mx = vec![[-f32::MAX; 4]; 256];
                for jt in 0..=bx {
                    load_tile(&mut ks, jt * 64, cc);
                    for lid in 0..256 {
                        let (tx, ty) = (lid & 15, lid >> 4);
                        let mut acc = [0.0f32; 16];
                        for e in 0..d {
                            for a in 0..4 {
                                for kk in 0..4 {
                                    acc[a * 4 + kk] = fma_f32_ftz(
                                        qs[(ty + 16 * a) * 65 + e],
                                        ks[(tx + 16 * kk) * 65 + e],
                                        acc[a * 4 + kk],
                                    );
                                }
                            }
                        }
                        for a in 0..4 {
                            let i = i0 + ty + 16 * a;
                            for kk in 0..4 {
                                let j = jt * 64 + tx + 16 * kk;
                                if i < t && j <= i {
                                    let s = mul_ftz(acc[a * 4 + kk], scale);
                                    p[(pb + i) * t + j] = s;
                                    mx[lid][a] = mx[lid][a].max(s);
                                }
                            }
                        }
                    }
                }
                let mut rm = [0.0f32; 64];
                for (r, m) in rm.iter_mut().enumerate() {
                    let mut x = -f32::MAX;
                    for tx in 0..16 {
                        let lid = (r % 16) * 16 + tx;
                        x = x.max(mx[lid][r / 16]);
                    }
                    *m = x;
                }
                // phase B
                let mut ps = vec![[0.0f32; 4]; 256];
                for jt in 0..=bx {
                    for lid in 0..256 {
                        let (tx, ty) = (lid & 15, lid >> 4);
                        let mut lp = [0.0f32; 16];
                        for kk in 0..4 {
                            let j = jt * 64 + tx + 16 * kk;
                            for a in 0..4 {
                                let ic = (i0 + ty + 16 * a).min(t - 1);
                                lp[kk * 4 + a] = p[(pb + ic) * t + j.min(ic)];
                            }
                        }
                        for kk in 0..4 {
                            let j = jt * 64 + tx + 16 * kk;
                            for a in 0..4 {
                                let i = i0 + ty + 16 * a;
                                let live = i < t && j <= i;
                                let x = exp_nak(add_f32_ftz(lp[kk * 4 + a], -rm[ty + 16 * a]));
                                if live || v == Variant::NoSelectInSum {
                                    ps[lid][a] = add_f32_ftz(ps[lid][a], x);
                                }
                                if live {
                                    p[(pb + i) * t + j] = x;
                                }
                            }
                        }
                    }
                }
                let mut rinv = [0.0f32; 64];
                for (r, ri) in rinv.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for tx in 0..16 {
                        sum = add_f32_ftz(sum, ps[(r % 16) * 16 + tx][r / 16]);
                    }
                    *ri = ftz(1.0 / sum);
                }
                // phase C
                let mut acc2 = vec![[0.0f32; 16]; 256];
                for jt in 0..ntiles {
                    if jt <= bx {
                        for lid in 0..256 {
                            let (tx, ty) = (lid & 15, lid >> 4);
                            let mut lp = [0.0f32; 16];
                            for a in 0..4 {
                                let ic = (i0 + ty + 16 * a).min(t - 1);
                                for kk in 0..4 {
                                    let j = jt * 64 + tx + 16 * kk;
                                    let jc = if jt < bx { j } else { j.min(ic) };
                                    lp[a * 4 + kk] = p[(pb + ic) * t + jc];
                                }
                            }
                            for a in 0..4 {
                                let r = ty + 16 * a;
                                let i = i0 + r;
                                for kk in 0..4 {
                                    let cl = tx + 16 * kk;
                                    let j = jt * 64 + cl;
                                    let x = mul_ftz(lp[a * 4 + kk], rinv[r]);
                                    let keep = (i < t && j <= i)
                                        || (v == Variant::NoSelectOnDiagonal && jt == bx && i < t);
                                    let pv = if keep { x } else { 0.0 };
                                    if i < t && j < t {
                                        p[(pb + i) * t + j] = pv;
                                    }
                                    qs[r * 65 + cl] = pv;
                                }
                            }
                        }
                        load_tile(&mut ks, jt * 64, 2 * cc);
                        for lid in 0..256 {
                            let (tx, ty) = (lid & 15, lid >> 4);
                            for jj in 0..64 {
                                for a in 0..4 {
                                    let r = ty + 16 * a;
                                    if jt < bx || jt * 64 + jj <= i0 + r {
                                        for kk in 0..4 {
                                            acc2[lid][a * 4 + kk] = fma_f32_ftz(
                                                qs[r * 65 + jj],
                                                ks[jj * 65 + tx + 16 * kk],
                                                acc2[lid][a * 4 + kk],
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        for lid in 0..256 {
                            let (tx, ty) = (lid & 15, lid >> 4);
                            for a in 0..4 {
                                let i = i0 + ty + 16 * a;
                                for kk in 0..4 {
                                    let j = jt * 64 + tx + 16 * kk;
                                    if i < t && j < t {
                                        p[(pb + i) * t + j] = 0.0;
                                    }
                                }
                            }
                        }
                    }
                }
                for (lid, acc) in acc2.iter().enumerate() {
                    let (tx, ty) = (lid & 15, lid >> 4);
                    for a in 0..4 {
                        let i = i0 + ty + 16 * a;
                        for kk in 0..4 {
                            let e = tx + 16 * kk;
                            if i < t && e < d {
                                att[(b * t + i) * cc + h * d + e] = acc[a * 4 + kk];
                            }
                        }
                    }
                }
            }
        }
    }
}

/// The emulation over the full grid, P starting as `p0` (one value everywhere).
fn emulated_from(k: &Case, v: Variant, p0: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let (_, no, _) = k.s.sizes().unwrap();
    let mut p = p0.to_vec();
    let mut att = vec![SENTINEL; no];
    emulate(k, k.s.t.div_ceil(64), v, &mut p, &mut att);
    (p, att)
}

fn emulated(k: &Case, v: Variant) -> (Vec<f32>, Vec<f32>) {
    let (_, _, np) = k.s.sizes().unwrap();
    emulated_from(k, v, &vec![SENTINEL; np])
}

// ------------------------------------------------------------------ launching (pushbuffer + QMD)

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

#[derive(Clone, Copy, PartialEq, Debug)]
enum Prog {
    Tile2,
    Tile3,
}

/// One launch through pushbuffer + QMD; `qmd_shared` is what the QMD declares, the simulator demands the program's own
/// shared size and barrier (as the board would need them).
fn launch(
    k: &Case,
    prog: Prog,
    grid: [u32; 3],
    block: [u32; 3],
    qmd_shared: SharedMemory,
) -> Result<(Vec<f32>, Vec<f32>), gpu_core::error::GpuError> {
    let (_, no, np) = k.s.sizes().unwrap();
    let mut cp = CommandProcessor::new();
    let (code, regs, own) = match prog {
        Prog::Tile2 => (ATTN_FWD_TILE2, at2::REGISTERS, at2::SHARED),
        Prog::Tile3 => (ATTN_FWD_TILE3, at3::REGISTERS, at3::SHARED),
    };
    cp.register_program_with_needs(
        code,
        attn_fwd_tile_on_cpu,
        Needs {
            shared_bytes: own.bytes,
            barriers: u32::from(own.barriers),
        },
    );
    cp.mem.write(PROG, code);
    cp.mem.write_f32s(QKV, &k.qkv);
    cp.mem.write_f32s(P, &vec![SENTINEL; np + 64]);
    cp.mem.write_f32s(ATT, &vec![SENTINEL; no + 64]);
    cp.mem.write_u32(SEM, 0);
    let cbuf = args(k).cbuf().unwrap();
    let mut cb = vec![0u8; 0x100];
    cb[..cbuf.len()].copy_from_slice(&cbuf);
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
    let pv = cp.mem.read_f32s(P, np + 64).unwrap();
    let av = cp.mem.read_f32s(ATT, no + 64).unwrap();
    assert!(pv[np..].iter().all(|&v| v == SENTINEL), "wrote after P");
    assert!(av[no..].iter().all(|&v| v == SENTINEL), "wrote after att");
    Ok((pv[..np].to_vec(), av[..no].to_vec()))
}

fn run(k: &Case, prog: Prog) -> (Vec<f32>, Vec<f32>) {
    let a = args(k);
    match prog {
        Prog::Tile2 => launch(k, prog, at2::grid(&a), at2::BLOCK, at2::SHARED).unwrap(),
        Prog::Tile3 => launch(k, prog, at3::grid(&a), at3::BLOCK, at3::SHARED).unwrap(),
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

const LISTING: &str = include_str!("../../../tools/nak/out/attn_fwd_tile3.sm86.txt");
const LISTING_2: &str = include_str!("../../../tools/nak/out/attn_fwd_tile2.sm86.txt");

/// The opcode of every instruction line (`r = op ...` or `op ...`, predicate dropped).
fn ops(txt: &str) -> Vec<&str> {
    txt.lines()
        .filter_map(|l| l.split_once("//").map(|(ins, _)| ins))
        .filter_map(|ins| {
            let ins = ins.split_once(" = ").map_or(ins, |(_, r)| r).trim();
            let ins = if ins.starts_with('@') {
                ins.split_once(char::is_whitespace)
                    .map_or("", |(_, r)| r)
                    .trim()
            } else {
                ins
            };
            ins.split_whitespace().next()
        })
        .collect()
}

// Prediction 2 (checked when compiling): ≤ 128 registers keep 2 blocks of 256 per SM, as `_2`.
const _: () = assert!(at3::REGISTERS > 0 && at3::REGISTERS <= 128);
const _: () = assert!(at3::REGISTERS.div_ceil(8) * 8 * 256 * 2 <= 65_536);
const _: () = assert!(at3::SHARED.bytes == 37_888 && at3::SHARED.bytes == at2::SHARED.bytes);
const _: () = assert!(at3::SHARED.bytes <= at3::SHARED.max_bytes);

// ------------------------------------------------------------------ the tests

/// Predictions 1 and 2, from the NAK listing.
#[test]
fn t190_binary_and_listing() {
    assert_eq!(hex(ATTN_FWD_TILE3), at3::ATTN_FWD_TILE3_SHA256);
    assert!(LISTING.contains(&format!("Num GPRs: {}", at3::REGISTERS)));
    let (new, old) = (ops(LISTING), ops(LISTING_2));
    let count = |v: &[&str], p: &str| v.iter().filter(|o| o.starts_with(p)).count();
    assert_eq!(count(&new, "atom") + count(&new, "red."), 0, "atomics");
    let mufu = |v: &[&str]| {
        let mut m: Vec<String> = v
            .iter()
            .filter(|o| o.starts_with("mufu"))
            .map(|o| o.to_string())
            .collect();
        m.sort();
        m
    };
    assert_eq!(mufu(&new), mufu(&old), "MUFU differs from attn_fwd_tile2");
    assert_eq!(count(&new, "ffma"), count(&old, "ffma"), "ffma count");
    assert!(
        !LISTING.contains("ld.local") && !LISTING.contains("st.local"),
        "local memory"
    );
    assert_eq!(count(&new, "bar.sync"), count(&old, "bar.sync"), "barriers");
    println!(
        "T190 listing: {} registers (_2 {}), {} ops (_2 {}), {} ffma, mufu {:?}, sel {} (_2 {}), imnmx {} (_2 {})",
        at3::REGISTERS,
        at2::REGISTERS,
        new.len(),
        old.len(),
        count(&new, "ffma"),
        mufu(&new),
        count(&new, "sel"),
        count(&old, "sel"),
        count(&new, "imnmx"),
        count(&old, "imnmx"),
    );
}

/// Prediction 4: the new GLSL, emulated literally, is the model (= `attn_fwd_tile2`) bit for bit; the launches too.
#[test]
fn t190_tile3_is_tile2_bit_for_bit() {
    for (name, k) in cases() {
        let (mp, ma) = k.model();
        let (ep, ea) = emulated(&k, Variant::Program);
        assert_eq!(bits(&ep), bits(&mp), "{name}: emulated P ≠ the model");
        assert_eq!(bits(&ea), bits(&ma), "{name}: emulated att ≠ the model");
        let (p2, a2) = run(&k, Prog::Tile2);
        let (p3, a3) = run(&k, Prog::Tile3);
        assert_eq!(bits(&p3), bits(&p2), "{name}: P of the _3 launch ≠ _2");
        assert_eq!(bits(&a3), bits(&a2), "{name}: att of the _3 launch ≠ _2");
        assert_eq!(bits(&p3), bits(&mp), "{name}: launch P = the model");
        println!(
            "T190 {name}: {} values of P + {} of att, bit for bit = attn_fwd_tile2",
            mp.len(),
            ma.len()
        );
    }
}

/// Garbage in P before the launch (an old step's values: NaN, 1e30, −1e30) changes no bit: the clamped loads that can
/// reach it are thrown away by the selects.
#[test]
fn t190_garbage_in_p_changes_nothing() {
    for k in [
        Case::new(31, shape(2, 130, 64, 2), 1.0),
        Case::new(32, shape(1, 65, 40, 1), 2.0),
    ] {
        let (_, _, np) = k.s.sizes().unwrap();
        let clean = emulated(&k, Variant::Program);
        for g in [f32::NAN, 1e30, -1e30, f32::INFINITY] {
            let got = emulated_from(&k, Variant::Program, &vec![g; np]);
            assert_eq!(bits(&got.0), bits(&clean.0), "P with garbage {g}");
            assert_eq!(bits(&got.1), bits(&clean.1), "att with garbage {g}");
        }
    }
}

/// Prediction 4: planted mistakes are caught.
#[test]
fn t190_planted_mistakes_are_caught() {
    let k = Case::new(21, shape(2, 130, 64, 2), 1.0);
    let a = args(&k);
    let g = at3::grid(&a);
    assert_eq!(g, [3, 2, 2]);
    // The QMD with attn_fwd_tile's shared size (512 bytes short): refused.
    assert!(launch(&k, Prog::Tile3, g, at3::BLOCK, at::SHARED).is_err());
    assert!(launch(&k, Prog::Tile3, g, at3::BLOCK, at3::SHARED).is_ok());
    // A grid one block short: rows 128, 129 stay sentinels; the emulation with 2 blocks agrees.
    let (p, att) = launch(
        &k,
        Prog::Tile3,
        [g[0] - 1, g[1], g[2]],
        at3::BLOCK,
        at3::SHARED,
    )
    .unwrap();
    let left = p.iter().chain(&att).filter(|&&x| x == SENTINEL).count();
    assert_eq!(left, 2 * 2 * 2 * 130 + 2 * 2 * 64);
    let (_, no, np) = k.s.sizes().unwrap();
    let (mut ep, mut ea) = (vec![SENTINEL; np], vec![SENTINEL; no]);
    emulate(&k, 2, Variant::Program, &mut ep, &mut ea);
    assert_eq!(bits(&ep), bits(&p), "the emulation, one block short (P)");
    assert_eq!(
        bits(&ea),
        bits(&att),
        "the emulation, one block short (att)"
    );
    // A wrong block or a grid past ⌈T/64⌉: refused.
    assert!(launch(&k, Prog::Tile3, g, [128, 1, 1], at3::SHARED).is_err());
    assert!(
        launch(
            &k,
            Prog::Tile3,
            [g[0] + 1, g[1], g[2]],
            at3::BLOCK,
            at3::SHARED
        )
        .is_err()
    );
    // d > 64: refused.
    let k65 = Case::new(22, shape(1, 8, 130, 2), 1.0);
    let a65 = args(&k65);
    assert!(!at3::shape_ok(&a65));
    assert!(launch(&k65, Prog::Tile3, at3::grid(&a65), at3::BLOCK, at3::SHARED).is_err());
    // The emulation's own mistakes (the selects of the plan dropped one at a time).
    let model = k.model();
    for (v, what) in [
        (Variant::NoSelectInSum, "select in phase B's sum"),
        (Variant::NoSelectOnDiagonal, "select on phase C's diagonal"),
    ] {
        let got = emulated(&k, v);
        assert!(
            bits(&got.0) != bits(&model.0) || bits(&got.1) != bits(&model.1),
            "{what}: not caught"
        );
    }
    // The tile selects: whether the entries they guard are ever used is written in the note (printed here).
    let got = emulated(&k, Variant::NoSelectOnTiles);
    let same = bits(&got.0) == bits(&model.0) && bits(&got.1) == bits(&model.1);
    println!("T190 select on the Q / K / V tiles dropped: same bits = {same}");
}
