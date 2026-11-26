//! T261 (C-157, `docs/C-157_ATTN_BACK0_TC_V4.md`, session 251), SIMULARE: `attn_back_tile0_tc` with 128-bit global
//! loads and stores (`tools/nak/attn_back_tile0_tc_v4.comp`, second NAK tool, Mesa 26.3.1).
//! Checked (predictions U4–U6, written before the code): the GLSL reads of both programs written out in Rust thread by
//! thread put the SAME f32 values in the same halves of the f16 dO / V tiles and in the same words of the P tile, for
//! every aligned shape, block and tile (with NaN above the diagonal of P); every vec4 is aligned and wholly inside its
//! row; at most 3 numbers above the diagonal per vector. The dP store and pass 2 written out the same way leave the
//! SAME final dS bits (from any dP tile and any row dot), and v4 writes only its block's rows, j < T. Through our
//! pushbuffer + QMD in the command-processor model both programs give the same bits, and unaligned shapes are refused
//! by `fits` and by the model (no write). Nothing here runs on the RTX 3070 Ti.

use gpu_sim::cmdproc::{
    ATTN_BACK_TILE0_TC_NEEDS, ATTN_BACK_TILE0_TC_V4_NEEDS, CommandProcessor, Needs, ProgramFn,
    attn_back_tile0_tc_on_cpu, attn_back_tile0_tc_v4_on_cpu,
};
use nvidia_ga104::attn_back::{Args, Mode};
use nvidia_ga104::attn_back_tile0_tc::{self as tc, ATTN_BACK_TILE0_TC};
use nvidia_ga104::attn_back_tile0_tc_v4::{self as v4, ATTN_BACK_TILE0_TC_V4};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

#[derive(Clone, Copy, Debug)]
struct Shape {
    b: usize,
    t: usize,
    c: usize,
    h: usize,
}

impl Shape {
    fn d(&self) -> usize {
        self.c / self.h
    }
    fn nq(&self) -> usize {
        self.b * self.t * 3 * self.c
    }
    fn no(&self) -> usize {
        self.b * self.t * self.c
    }
    fn np(&self) -> usize {
        self.b * self.h * self.t * self.t
    }
}

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

struct Inputs {
    qkv: Vec<f32>,
    p: Vec<f32>,
    dout: Vec<f32>,
}

/// Finite f16-range inputs; P above the diagonal = NaN (neither program may let it into the result).
fn inputs(s: &Shape, seed: u64) -> Inputs {
    let mut r = rng(seed);
    let qkv = (0..s.nq()).map(|_| r() / 2.0).collect();
    let dout = (0..s.no()).map(|_| r() / 16.0).collect();
    let mut p = vec![0f32; s.np()];
    for bh in 0..s.b * s.h {
        for i in 0..s.t {
            for j in 0..s.t {
                p[(bh * s.t + i) * s.t + j] = if j > i { f32::NAN } else { r().abs() / 2.0 };
            }
        }
    }
    Inputs { qkv, p, dout }
}

/// Aligned shapes (T, d multiples of 4), with partial tiles (T = 100, 132), several heads and samples.
const SMALL: [Shape; 6] = [
    Shape {
        b: 2,
        t: 16,
        c: 32,
        h: 2,
    },
    Shape {
        b: 1,
        t: 64,
        c: 64,
        h: 1,
    },
    Shape {
        b: 2,
        t: 100,
        c: 96,
        h: 2,
    },
    Shape {
        b: 1,
        t: 132,
        c: 40,
        h: 1,
    },
    Shape {
        b: 1,
        t: 200,
        c: 192,
        h: 3,
    },
    Shape {
        b: 2,
        t: 128,
        c: 128,
        h: 2,
    },
];

fn all_blocks(s: &Shape) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for b in 0..s.b {
        for h in 0..s.h {
            for bx in 0..s.t.div_ceil(64) {
                v.push((bx, h, b));
            }
        }
    }
    v
}

/// A planted mistake, to show the comparisons can fail.
#[derive(Clone, Copy, PartialEq)]
enum Plant {
    None,
    /// the two f16 of a dO word in the wrong order
    SwappedHalves,
    /// a P vector kept only when all 4 numbers are at or below the diagonal
    DiagonalByVector,
    /// pass 2 on the diagonal tile without the per-element select (the vector stored whole)
    NoDiagonalSelect,
}

/// The dO / V halves (index = 2 · word + half, f32 bits before the f16 rounding) and the P tile words, for tile `jt`
/// of block (bx, h, b).
struct Tiles {
    ov: Vec<u32>,
    p: Vec<u32>,
}

fn sk(row: usize, col: usize) -> usize {
    row * 64 + ((col + row) & 63)
}

/// `attn_back_tile0_tc.comp`'s reads, as written there (clamped scalar loads, select at the store).
fn tc_tiles(s: &Shape, x_: &Inputs, (bx, h, b): (usize, usize, usize), jt: usize) -> Tiles {
    let (t, c, d) = (s.t, s.c, s.d());
    let i0 = bx * 64;
    let base = (b * s.h + h) * t * t;
    let mut ov = vec![u32::MAX; 8192];
    for q in 0..2048 {
        let r = q >> 5;
        let x0 = (q & 31) * 2;
        let i = (i0 + r).min(t - 1);
        let jr = (jt * 64 + r).min(t - 1);
        for u in 0..2 {
            let x = (x0 + u).min(d - 1);
            let lo = x_.dout[(b * t + i) * c + h * d + x];
            let lv = x_.qkv[(b * t + jr) * 3 * c + 2 * c + h * d + x];
            let xu = x0 + u;
            ov[2 * q + u] = if i0 + r < t && xu < d {
                lo.to_bits()
            } else {
                0
            };
            ov[4096 + 2 * q + u] = if jt * 64 + r < t && xu < d {
                lv.to_bits()
            } else {
                0
            };
        }
    }
    let mut p = vec![u32::MAX; 4096];
    for idx in 0..4096 {
        let i = i0 + (idx >> 6);
        let j = jt * 64 + (idx & 63);
        let ic = i.min(t - 1);
        let lp = x_.p[base + ic * t + j.min(ic)];
        p[sk(idx >> 6, idx & 63)] = if i < t && j <= i { lp.to_bits() } else { 0 };
    }
    Tiles { ov, p }
}

/// Counters of what v4's reads touched besides the values.
#[derive(Default)]
struct Reads {
    vectors: usize,
    above_diag_max: usize,
}

fn vec4(buf: &[f32], at: usize) -> [f32; 4] {
    assert_eq!(at % 4, 0, "vec4 misaligned at {at}");
    assert!(at + 3 < buf.len(), "vec4 at {at} outside ({})", buf.len());
    [buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]
}

/// `attn_back_tile0_tc_v4.comp`'s loads + shared stores, as written there (index × in = vector 0 when nothing kept).
fn v4_tiles(
    s: &Shape,
    x_: &Inputs,
    (bx, h, b): (usize, usize, usize),
    jt: usize,
    rd: &mut Reads,
    plant: Plant,
) -> Tiles {
    let (t, c, d) = (s.t, s.c, s.d());
    let i0 = bx * 64;
    let base = (b * s.h + h) * t * t;
    let mut ov = vec![u32::MAX; 8192];
    let mut p = vec![u32::MAX; 4096];
    for lid in 0..256 {
        let tx = lid & 15;
        let ty = lid >> 4;
        let x4 = tx * 4;
        for sv in 0..4 {
            let r = ty + 16 * sv;
            let i = i0 + r;
            let jr = jt * 64 + r;
            let ino = i < t && x4 < d;
            let inv = jr < t && x4 < d;
            let ao = ((b * t + i) * c + h * d + x4) * usize::from(ino);
            let av = ((b * t + jr) * 3 * c + 2 * c + h * d + x4) * usize::from(inv);
            if ino {
                assert!(x4 + 3 < d, "dO vec4 leaves its head");
            }
            let o = vec4(&x_.dout, ao);
            let v = vec4(&x_.qkv, av);
            rd.vectors += 2;
            let q = r * 32 + x4 / 2;
            for m in 0..4 {
                let mm = if plant == Plant::SwappedHalves && m < 2 {
                    1 - m
                } else {
                    m
                };
                ov[2 * q + mm] = if ino { o[m].to_bits() } else { 0 };
                ov[4096 + 2 * q + m] = if inv { v[m].to_bits() } else { 0 };
            }
            // the P tile
            let j = jt * 64 + x4;
            let inp = i < t && j <= i;
            if inp {
                assert!(j + 3 < t, "P vec4 leaves its row");
                rd.above_diag_max = rd.above_diag_max.max((j + 3).saturating_sub(i));
            }
            let pv = vec4(&x_.p, (base + i * t + j) * usize::from(inp));
            rd.vectors += 1;
            for m in 0..4 {
                let kept = if plant == Plant::DiagonalByVector {
                    inp && j + 3 <= i
                } else {
                    inp && j + m <= i
                };
                p[sk(r, x4 + m)] = if kept { pv[m].to_bits() } else { 0 };
            }
        }
    }
    Tiles { ov, p }
}

fn compare_reads(
    s: &Shape,
    x_: &Inputs,
    blocks: &[(usize, usize, usize)],
    plant: Plant,
) -> (bool, Reads) {
    let mut rd = Reads::default();
    let mut same = true;
    for &blk in blocks {
        for jt in 0..=blk.0 {
            let a = tc_tiles(s, x_, blk, jt);
            let b = v4_tiles(s, x_, blk, jt, &mut rd, plant);
            assert!(
                a.ov.iter().chain(&a.p).all(|&w| w != u32::MAX),
                "tc left a hole"
            );
            assert!(
                b.ov.iter().chain(&b.p).all(|&w| w != u32::MAX),
                "v4 left a hole"
            );
            same &= a.ov == b.ov && a.p == b.p;
        }
    }
    (same, rd)
}

/// U4: the same values in shared memory on every aligned shape, block and tile; vectors aligned and in their rows.
#[test]
fn u4_v4_reads_put_tc_values_in_shared_memory() {
    let mut above = 0;
    for (n, s) in SMALL.iter().enumerate() {
        let x_ = inputs(s, 10 + n as u64);
        let (same, rd) = compare_reads(s, &x_, &all_blocks(s), Plant::None);
        assert!(same, "{s:?}");
        assert!(rd.vectors > 0);
        assert!(rd.above_diag_max <= 3);
        above = above.max(rd.above_diag_max);
    }
    assert_eq!(above, 3);
    // the planted mistakes are caught
    let s = SMALL[2];
    let x_ = inputs(&s, 99);
    for plant in [Plant::SwappedHalves, Plant::DiagonalByVector] {
        assert!(!compare_reads(&s, &x_, &all_blocks(&s), plant).0);
    }
}

/// U4 on #513's attention shape (T 512, C 768, H 12): heads 0 and 11 of the first sample, every block.
#[test]
fn u4_v4_reads_at_513() {
    let s = Shape {
        b: 1,
        t: 512,
        c: 768,
        h: 12,
    };
    let x_ = inputs(&s, 513);
    let blocks: Vec<_> = all_blocks(&s)
        .into_iter()
        .filter(|&(_, h, _)| h == 0 || h == 11)
        .collect();
    let (same, rd) = compare_reads(&s, &x_, &blocks, Plant::None);
    assert!(same);
    assert_eq!(rd.above_diag_max, 3);
}

// ---- the dP store and pass 2, written out ----

/// A deterministic dP value for (block, row i, column j) — any values do: both programs store the same tile.
fn dp(i: usize, j: usize, salt: usize) -> f32 {
    let mut x = (i as u64 * 1_000_003 + j as u64 * 7_919 + salt as u64 * 104_729) | 1;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// The row dot of row i (any value; read from cs by both programs).
fn dr(i: usize, salt: usize) -> f32 {
    dp(i, 100_000, salt) / 4.0
}

/// `attn_back_tile0_tc.comp`'s dP stores (per tile) then pass 2, for block (bx, h, b), on `ds`.
#[allow(clippy::needless_range_loop)] // written as the GLSL indexes it
fn tc_passes(
    s: &Shape,
    x_: &Inputs,
    (bx, h, b): (usize, usize, usize),
    scale: f32,
    ds: &mut [f32],
) {
    let t = s.t;
    let i0 = bx * 64;
    let base = (b * s.h + h) * t * t;
    let ntiles = t.div_ceil(64);
    for jt in 0..=bx {
        for lid in 0..256 {
            let (tx, ty) = (lid & 15, lid >> 4);
            for a in 0..4 {
                let i = i0 + ty + 16 * a;
                for k in 0..4 {
                    let j = jt * 64 + tx + 16 * k;
                    if i < t && (jt < bx || j <= i) {
                        ds[base + i * t + j] = dp(i, j, b * s.h + h);
                    }
                }
            }
        }
    }
    for lid in 0..256 {
        let (tx, ty) = (lid & 15, lid >> 4);
        for jt in 0..bx {
            for a in 0..4 {
                let ic = (i0 + ty + 16 * a).min(t - 1);
                let i = i0 + ty + 16 * a;
                for k in 0..4 {
                    let o = base + ic * t + jt * 64 + tx + 16 * k;
                    let y = (x_.p[o] * (ds[o] - dr(i, b * s.h + h))) * scale;
                    if i < t {
                        ds[base + i * t + jt * 64 + tx + 16 * k] = y;
                    }
                }
            }
        }
        let jt = bx;
        let mut v = [[0f32; 4]; 4];
        for a in 0..4 {
            let ic = (i0 + ty + 16 * a).min(t - 1);
            let i = i0 + ty + 16 * a;
            for k in 0..4 {
                let o = base + ic * t + (jt * 64 + tx + 16 * k).min(ic);
                v[a][k] = (x_.p[o] * (ds[o] - dr(i, b * s.h + h))) * scale;
            }
        }
        for a in 0..4 {
            let i = i0 + ty + 16 * a;
            for k in 0..4 {
                let j = jt * 64 + tx + 16 * k;
                if i < t && j < t {
                    ds[base + i * t + j] = if j <= i { v[a][k] } else { 0.0 };
                }
            }
        }
        for jt in bx + 1..ntiles {
            for a in 0..4 {
                let i = i0 + ty + 16 * a;
                for k in 0..4 {
                    let j = jt * 64 + tx + 16 * k;
                    if i < t && j < t {
                        ds[base + i * t + j] = 0.0;
                    }
                }
            }
        }
    }
}

/// `attn_back_tile0_tc_v4.comp`'s dP vector stores then pass 2 on vectors. Panics on a misaligned vector or a vector
/// past its row; records every word written (must be the block's rows, j < T).
fn v4_passes(
    s: &Shape,
    x_: &Inputs,
    (bx, h, b): (usize, usize, usize),
    scale: f32,
    ds: &mut [f32],
    plant: Plant,
) {
    let t = s.t;
    let i0 = bx * 64;
    let base = (b * s.h + h) * t * t;
    let ntiles = t.div_ceil(64);
    let salt = b * s.h + h;
    let put = |ds: &mut [f32], at: usize, v: [f32; 4]| {
        assert_eq!(at % 4, 0, "store misaligned");
        let (row, col) = ((at - base) / t, (at - base) % t);
        assert!(
            (i0..(i0 + 64).min(t)).contains(&row) && col + 3 < t,
            "write outside the block's rows / past T: ({row}, {col})"
        );
        ds[at..at + 4].copy_from_slice(&v);
    };
    for jt in 0..=bx {
        for lid in 0..256 {
            let (tx, ty) = (lid & 15, lid >> 4);
            let x4 = tx * 4;
            for sv in 0..4 {
                let i = i0 + ty + 16 * sv;
                let j = jt * 64 + x4;
                if i < t && j < t {
                    put(
                        ds,
                        base + i * t + j,
                        core::array::from_fn(|m| dp(i, j + m, salt)),
                    );
                }
            }
        }
    }
    for lid in 0..256 {
        let (tx, ty) = (lid & 15, lid >> 4);
        let x4 = tx * 4;
        for jt in 0..bx {
            for a in 0..4 {
                let i = i0 + ty + 16 * a;
                let ic = i.min(t - 1);
                let o = base + ic * t + jt * 64 + x4;
                let (l, p) = (vec4(ds, o), vec4(&x_.p, o));
                let y: [f32; 4] = core::array::from_fn(|m| (p[m] * (l[m] - dr(i, salt))) * scale);
                if i < t {
                    put(ds, base + i * t + jt * 64 + x4, y);
                }
            }
        }
        let j = bx * 64 + x4;
        for a in 0..4 {
            let i = i0 + ty + 16 * a;
            let ic = i.min(t - 1);
            let o = (base + ic * t + j) * usize::from(j < t);
            let (l, p) = (vec4(ds, o), vec4(&x_.p, o));
            let v: [f32; 4] = core::array::from_fn(|m| (p[m] * (l[m] - dr(i, salt))) * scale);
            if i < t && j < t {
                let out = if plant == Plant::NoDiagonalSelect {
                    v
                } else {
                    core::array::from_fn(|m| if j + m <= i { v[m] } else { 0.0 })
                };
                put(ds, base + i * t + j, out);
            }
        }
        for jt in bx + 1..ntiles {
            let j = jt * 64 + x4;
            for a in 0..4 {
                let i = i0 + ty + 16 * a;
                if i < t && j < t {
                    put(ds, base + i * t + j, [0.0; 4]);
                }
            }
        }
    }
}

fn compare_passes(s: &Shape, x_: &Inputs, plant: Plant) -> bool {
    const POISON: f32 = -3.5;
    let scale = Args::scale_for(s.d() as u32);
    let mut a = vec![POISON; s.np()];
    let mut b = vec![POISON; s.np()];
    for blk in all_blocks(s) {
        tc_passes(s, x_, blk, scale, &mut a);
        v4_passes(s, x_, blk, scale, &mut b, plant);
    }
    assert!(a.iter().all(|&x| x != POISON), "tc left a word of dS");
    a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// U5: the same final dS bits (NaN above the diagonal of P), only the block's rows and j < T written.
#[test]
fn u5_v4_passes_give_tc_bits() {
    for (n, s) in SMALL.iter().enumerate() {
        let x_ = inputs(s, 40 + n as u64);
        assert!(compare_passes(s, &x_, Plant::None), "{s:?}");
    }
    let s = SMALL[2];
    assert!(!compare_passes(&s, &inputs(&s, 3), Plant::NoDiagonalSelect));
}

// ---- through the command-processor model (pushbuffer + QMD), as the OS would launch ----

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x70_0000;
const PUSH: u64 = 0x71_0000;
const QKV: u64 = 0x100_0000;
const P: u64 = 0x200_0000;
const DOUT: u64 = 0x300_0000;
const DS: u64 = 0x400_0000;
const DQKV: u64 = 0x500_0000;
const SENTINEL: f32 = -12345.0;

struct Prog {
    code: &'static [u8],
    run: ProgramFn,
    needs: Needs,
    registers: u32,
    shared: SharedMemory,
}

const TC: Prog = Prog {
    code: ATTN_BACK_TILE0_TC,
    run: attn_back_tile0_tc_on_cpu,
    needs: ATTN_BACK_TILE0_TC_NEEDS,
    registers: tc::REGISTERS,
    shared: tc::SHARED,
};
const V4: Prog = Prog {
    code: ATTN_BACK_TILE0_TC_V4,
    run: attn_back_tile0_tc_v4_on_cpu,
    needs: ATTN_BACK_TILE0_TC_V4_NEEDS,
    registers: v4::REGISTERS,
    shared: v4::SHARED,
};

fn args(s: &Shape, shift_ds: u64) -> Args {
    Args {
        qkv: QKV,
        p: P,
        dout: DOUT,
        ds: DS + shift_ds,
        dqkv: DQKV,
        batch: s.b as u32,
        t: s.t as u32,
        c: s.c as u32,
        heads: s.h as u32,
        scale: Args::scale_for((s.c / s.h) as u32),
        mode: Mode::Rows,
    }
}

/// Inputs for the model run: P = a real softmax (finite, 0 above the diagonal), V / dO in f16 range.
fn model_inputs(s: &Shape, seed: u64) -> Inputs {
    let mut r = rng(seed);
    let qkv: Vec<f32> = (0..s.nq()).map(|_| r() / 2.0).collect();
    let dout = (0..s.no()).map(|_| r() / 16.0).collect();
    let shp = gpu_compute::layer::AttnShape {
        batch: s.b,
        t: s.t,
        c: s.c,
        heads: s.h,
    };
    let (p, _) =
        gpu_compute::layer::attn_fwd_gpu(&shp, &qkv, Args::scale_for(s.d() as u32)).unwrap();
    Inputs { qkv, p, dout }
}

/// One launch of `prog`; the cbuf is `attn_back_tile0_tc`'s (the same bytes for both), so an unaligned shape reaches
/// the model.
fn run(
    prog: &Prog,
    s: &Shape,
    shift_ds: u64,
    x_: &Inputs,
) -> Result<Vec<u32>, gpu_core::error::GpuError> {
    let a = args(s, shift_ds);
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(prog.code, prog.run, prog.needs);
    cp.mem.write(PROG, prog.code);
    cp.mem.write(CBUF, &tc::cbuf(&a).unwrap());
    cp.mem.write_f32s(QKV, &x_.qkv);
    cp.mem.write_f32s(P, &x_.p);
    cp.mem.write_f32s(DOUT, &x_.dout);
    cp.mem.write_f32s(a.ds, &vec![SENTINEL; s.np() + 16]);
    cp.mem.write_u32(SEM, 0);
    let q = ComputeQmd {
        program_va: PROG,
        registers: prog.registers,
        cbuf0_va: CBUF,
        cbuf0_size: tc::CBUF_BYTES as u32,
        grid: tc::grid(&a),
        block: tc::BLOCK,
        release: Some((SEM, 1)),
    };
    cp.mem
        .write(QMD, &q.to_bytes_with_shared(&prog.shared).unwrap());
    let mut w = [0u32; 16];
    let mut pb = PushBuf::new(&mut w);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    pb.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    pb.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = pb.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    let res = cp.run_gpfifo(&[(e.entry0, e.entry1)]);
    let got = cp.mem.read_f32s(a.ds, s.np() + 16).unwrap();
    assert!(
        got[s.np()..].iter().all(|&x| x == SENTINEL),
        "wrote after dS"
    );
    if res.is_err() {
        assert!(got.iter().all(|&x| x == SENTINEL), "refused but wrote");
    }
    res?;
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    Ok(got[..s.np()].iter().map(|x| x.to_bits()).collect())
}

/// U6: the same bits as `attn_back_tile0_tc` through the command processor on aligned shapes.
#[test]
fn u6_v4_gives_tc_bits_through_the_command_processor() {
    for (i, s) in SMALL.iter().enumerate() {
        let x_ = model_inputs(s, 200 + i as u64);
        assert!(v4::fits(&args(s, 0)), "{s:?}");
        let old = run(&TC, s, 0, &x_).unwrap();
        let new = run(&V4, s, 0, &x_).unwrap();
        assert!(
            old.iter().all(|&b| b != SENTINEL.to_bits()),
            "{s:?}: dS not written"
        );
        assert_eq!(old, new, "{s:?}");
    }
}

/// U6: unaligned shapes are refused by `fits` / `cbuf` and by the model, which writes nothing.
#[test]
fn u6_unaligned_shapes_are_refused() {
    let cases = [
        (
            Shape {
                b: 1,
                t: 130,
                c: 64,
                h: 1,
            },
            0u64,
        ),
        (
            Shape {
                b: 1,
                t: 16,
                c: 12,
                h: 2,
            },
            0,
        ),
        (
            Shape {
                b: 1,
                t: 16,
                c: 32,
                h: 2,
            },
            4,
        ),
    ];
    for (s, shift) in cases {
        let a = args(&s, shift);
        assert!(tc::shape_ok(&a));
        assert!(!v4::fits(&a), "{s:?} +{shift}");
        assert!(v4::cbuf(&a).is_none());
        let x_ = model_inputs(&s, 7);
        assert!(run(&TC, &s, shift, &x_).is_ok());
        assert!(
            run(&V4, &s, shift, &x_).is_err(),
            "{s:?} +{shift}: the model guessed"
        );
    }
    let s = SMALL[0];
    for which in 0..4 {
        let mut a = args(&s, 0);
        match which {
            0 => a.qkv += 4,
            1 => a.p += 8,
            2 => a.dout += 12,
            _ => a.ds += 4,
        }
        assert!(!v4::fits(&a));
    }
}

/// Which sets of the OS's attention probe v4 may take (mode 0): all but `t130-d64` (T = 130).
#[test]
fn probe_sets_that_fit() {
    use nvidia_ga104::attn_tile_probe::SETS;
    let fit: Vec<&str> = SETS
        .iter()
        .filter(|s| v4::fits(&s.back_args(0x10_0000, Mode::Rows)))
        .map(|s| s.name)
        .collect();
    assert_eq!(
        fit,
        [
            "mic-b2-t16-d16",
            "t64-d40",
            "t200-h3",
            "ascutit",
            "atentie-513-b4"
        ]
    );
}
