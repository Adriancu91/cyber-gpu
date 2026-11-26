//! T258 (C-157 after os-457, `docs/C-157_ATTN_BACK_TC16_V4.md`, session 248), SIMULARE: `attn_back_tile_tc16` with
//! 128-bit global loads (`tools/nak/attn_back_tile_tc16_v4.comp`) and the same with the next tile read ahead
//! (`attn_back_tile_tc16_v4pf.comp`; the same loads, issued earlier), second NAK tool, Mesa 26.3.1.
//! Checked (predictions V4–V5, written before the code): the GLSL reads of both programs written out in Rust thread
//! by thread put the SAME f32 values in the same halves of shared memory for every aligned shape, block, pass and tile
//! (with NaN above the diagonal of P / dS and past the heads); every vec4 v4 reads is aligned and wholly inside its
//! buffer and row, also the read-ahead after the last tile; at most 3 numbers above the diagonal per vector; through
//! our pushbuffer + QMD in the command-processor model both programs give the same bits, and unaligned shapes are
//! refused by `fits` and by the model (no write). Nothing here runs on the RTX 3070 Ti.

use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    CommandProcessor, Needs, ProgramFn, attn_back_tile_tc16_on_cpu, attn_back_tile_tc16_v4_on_cpu,
    attn_back_tile_tc16_v4pf_on_cpu,
};
use nvidia_ga104::attn_back::{Args, Mode};
use nvidia_ga104::attn_back_tile_tc16::{self as tc16, ATTN_BACK_TILE_TC16};
use nvidia_ga104::attn_back_tile_tc16_v4::{self as v4, ATTN_BACK_TILE_TC16_V4};
use nvidia_ga104::attn_back_tile_tc16_v4pf::{self as v4pf, ATTN_BACK_TILE_TC16_V4PF};
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
    ds: Vec<f32>,
}

/// Finite f16-range inputs; P / dS above the diagonal = NaN (neither program may let them into a product).
fn inputs(s: &Shape, seed: u64) -> Inputs {
    let mut r = rng(seed);
    let qkv = (0..s.nq()).map(|_| r()).collect();
    let dout = (0..s.no()).map(|_| r()).collect();
    let mut p = vec![0f32; s.np()];
    let mut ds = vec![0f32; s.np()];
    for bh in 0..s.b * s.h {
        for i in 0..s.t {
            for j in 0..s.t {
                let at = (bh * s.t + i) * s.t + j;
                if j > i {
                    p[at] = f32::NAN;
                    ds[at] = f32::NAN;
                } else {
                    p[at] = r().abs() / 2.0;
                    ds[at] = r() / 8.0;
                }
            }
        }
    }
    Inputs { qkv, p, dout, ds }
}

/// The halves of sa / sb (index = 2 · word + half) as f32 bits, for tile `kt` of `pass` of block (bx, h, b).
type Tiles = (Vec<u32>, Vec<u32>);

/// `attn_back_tile_tc16.comp`'s reads, as written there (clamped scalar loads, select at the store).
fn tc16_tile(
    s: &Shape,
    x_: &Inputs,
    (bx, h, b): (usize, usize, usize),
    pass: usize,
    kt: usize,
) -> Tiles {
    let (t, c, d) = (s.t, s.c, s.d());
    let (srcs, srcb) = if pass == 2 {
        (&x_.p, &x_.dout)
    } else {
        (&x_.ds, &x_.qkv)
    };
    let strb = if pass == 2 { c } else { 3 * c };
    let offb = if pass == 0 { c } else { 0 };
    let t0 = bx * 64;
    let base = (b * s.h + h) * t * t;
    let mut sa = vec![u32::MAX; 4096];
    let mut sb = vec![u32::MAX; 4096];
    for q in 0..2048 {
        let r = q >> 5;
        let x0 = (q & 31) * 2;
        for m in 0..2 {
            let x = x0 + m;
            let ls = if pass == 0 {
                let tc = (t0 + r).min(t - 1);
                srcs[base + tc * t + (kt * 64 + x).min(tc)]
            } else {
                let ic = (kt * 64 + r).min(t - 1);
                srcs[base + ic * t + (t0 + x).min(ic)]
            };
            let kr = (kt * 64 + r).min(t - 1);
            let lb = srcb[(b * t + kr) * strb + offb + h * d + x.min(d - 1)];
            let keep = if pass == 0 {
                t0 + r < t && kt * 64 + x <= t0 + r
            } else {
                kt * 64 + r < t && t0 + x <= kt * 64 + r
            };
            sa[2 * q + m] = if keep { ls.to_bits() } else { 0 };
            let keepb = kt * 64 + r < t && x < d;
            sb[2 * q + m] = if keepb { lb.to_bits() } else { 0 };
        }
    }
    (sa, sb)
}

/// What v4's reads touched besides the values: vectors read, above-diagonal numbers read.
#[derive(Default)]
struct Reads {
    vectors: usize,
    above_diag_max: usize,
}

/// A planted mistake, to show the comparison can fail: `SwappedHalves` = the two f16 of a word in the wrong order;
/// `DiagonalByVector` = a vec4 kept only when all 4 numbers are at or below the diagonal.
#[derive(Clone, Copy, PartialEq)]
enum Plant {
    None,
    SwappedHalves,
    DiagonalByVector,
}

/// `attn_back_tile_tc16_v4.comp`'s `load_tile` + store, as written there. Panics on a vec4 that is misaligned or
/// not wholly inside its buffer (index ×0 = vector 0 when nothing in it is kept).
fn v4_tile(
    s: &Shape,
    x_: &Inputs,
    (bx, h, b): (usize, usize, usize),
    pass: usize,
    kt: usize,
    rd: &mut Reads,
    plant: Plant,
) -> Tiles {
    let (t, c, d) = (s.t, s.c, s.d());
    let (srcs, srcb) = if pass == 2 {
        (&x_.p, &x_.dout)
    } else {
        (&x_.ds, &x_.qkv)
    };
    let strb = if pass == 2 { c } else { 3 * c };
    let offb = if pass == 0 { c } else { 0 };
    let t0 = bx * 64;
    let base = (b * s.h + h) * t * t;
    let mut sa = vec![u32::MAX; 4096];
    let mut sb = vec![u32::MAX; 4096];
    let vec4 = |buf: &[f32], at: usize| -> [f32; 4] {
        assert!(at + 3 < buf.len(), "vec4 at {at} outside ({})", buf.len());
        [buf[at], buf[at + 1], buf[at + 2], buf[at + 3]]
    };
    for lid in 0..256 {
        for sv in 0..4 {
            let e = lid + 256 * sv;
            let r = e >> 4;
            let x = (e & 15) * 4;
            let (row, col) = if pass == 0 {
                (t0 + r, kt * 64 + x)
            } else {
                (kt * 64 + r, t0 + x)
            };
            let ina = if plant == Plant::DiagonalByVector {
                row < t && col + 3 <= row
            } else {
                row < t && col <= row
            };
            let at_a = base + row * t + col;
            if ina {
                assert_eq!(at_a % 4, 0, "sa vec4 misaligned");
                assert!(col + 3 < t, "sa vec4 leaves its row");
                rd.above_diag_max = rd.above_diag_max.max((col + 3).saturating_sub(row).min(3));
            }
            let ra = vec4(srcs, if ina { at_a } else { 0 });
            let kr = kt * 64 + r;
            let inb = kr < t && x < d;
            let at_b = (b * t + kr) * strb + offb + h * d + x;
            if inb {
                assert_eq!(at_b % 4, 0, "sb vec4 misaligned");
                assert!(x + 3 < d, "sb vec4 leaves its head");
            }
            let rb = vec4(srcb, if inb { at_b } else { 0 });
            rd.vectors += 2;
            let q = r * 32 + (x >> 1);
            for m in 0..4 {
                let keep = ina
                    && if pass == 0 {
                        t0 + r < t && kt * 64 + x + m <= t0 + r
                    } else {
                        kt * 64 + r < t && t0 + x + m <= kt * 64 + r
                    };
                let keepb = inb && kr < t && x + m < d;
                let half = if plant == Plant::SwappedHalves {
                    2 * (q + m / 2) + 1 - m % 2
                } else {
                    2 * (q + m / 2) + m % 2
                };
                sa[half] = if keep { ra[m].to_bits() } else { 0 };
                sb[half] = if keepb { rb[m].to_bits() } else { 0 };
            }
        }
    }
    (sa, sb)
}

fn compare_reads(s: &Shape, blocks: &[(usize, usize, usize)], seed: u64) -> Reads {
    let x_ = inputs(s, seed);
    let ntiles = s.t.div_ceil(64);
    let mut rd = Reads::default();
    for &blk in blocks {
        let bx = blk.0;
        for pass in 0..3 {
            let (first, last) = if pass == 0 { (0, bx) } else { (bx, ntiles - 1) };
            for kt in first..=last {
                let old = tc16_tile(s, &x_, blk, pass, kt);
                let new = v4_tile(s, &x_, blk, pass, kt, &mut rd, Plant::None);
                assert!(old.0.iter().chain(&old.1).all(|&w| w != u32::MAX));
                assert_eq!(old, new, "{s:?} block {blk:?} pass {pass} tile {kt}");
                for w in old.0.iter().chain(&old.1) {
                    assert!(!f32::from_bits(*w).is_nan(), "a NaN got into a tile");
                }
            }
            // the read-ahead after the last tile: only vectors inside the buffers (values unused)
            let _ = v4_tile(s, &x_, blk, pass, last + 1, &mut rd, Plant::None);
        }
    }
    rd
}

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
        c: 120,
        h: 3,
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
        c: 64,
        h: 1,
    },
    Shape {
        b: 1,
        t: 8,
        c: 8,
        h: 2,
    },
    Shape {
        b: 1,
        t: 68,
        c: 48,
        h: 4,
    },
];

/// V4: the same values in shared memory as tc16 on aligned shapes, every vec4 aligned and inside.
#[test]
fn v4_reads_put_tc16_values_in_shared_memory() {
    for (i, s) in SMALL.iter().enumerate() {
        let rd = compare_reads(s, &all_blocks(s), 100 + i as u64);
        assert!(rd.vectors > 0);
        assert!(rd.above_diag_max <= 3);
    }
}

/// The comparison is not blind: each planted mistake changes some tile of the smallest shape.
#[test]
fn planted_mistakes_are_caught() {
    let s = SMALL[0];
    let x_ = inputs(&s, 9);
    for plant in [Plant::SwappedHalves, Plant::DiagonalByVector] {
        let mut differs = false;
        for pass in 0..3 {
            let old = tc16_tile(&s, &x_, (0, 1, 1), pass, 0);
            let new = v4_tile(&s, &x_, (0, 1, 1), pass, 0, &mut Reads::default(), plant);
            differs |= old != new;
        }
        assert!(differs);
    }
}

/// V4 on #513's attention shape (T 512, C 768, H 12): the first and last head of the first sample, every block.
#[test]
fn v4_reads_put_tc16_values_in_shared_memory_at_513() {
    let s = Shape {
        b: 1,
        t: 512,
        c: 768,
        h: 12,
    };
    let blocks: Vec<_> = all_blocks(&s)
        .into_iter()
        .filter(|&(_, h, _)| h == 0 || h == 11)
        .collect();
    let rd = compare_reads(&s, &blocks, 513);
    assert_eq!(rd.above_diag_max, 3);
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

const TC16: Prog = Prog {
    code: ATTN_BACK_TILE_TC16,
    run: attn_back_tile_tc16_on_cpu,
    needs: ATTN_BACK_TILE_TC16_NEEDS,
    registers: tc16::REGISTERS,
    shared: tc16::SHARED,
};
const V4: Prog = Prog {
    code: ATTN_BACK_TILE_TC16_V4,
    run: attn_back_tile_tc16_v4_on_cpu,
    needs: ATTN_BACK_TILE_TC16_V4_NEEDS,
    registers: v4::REGISTERS,
    shared: v4::SHARED,
};
const V4PF: Prog = Prog {
    code: ATTN_BACK_TILE_TC16_V4PF,
    run: attn_back_tile_tc16_v4pf_on_cpu,
    needs: ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    registers: v4pf::REGISTERS,
    shared: v4pf::SHARED,
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
        mode: Mode::Columns,
    }
}

/// One launch of `prog`; the cbuf is tc16's (the same bytes for both), so an unaligned shape reaches the model.
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
    cp.mem.write(CBUF, &tc16::cbuf(&a).unwrap());
    cp.mem.write_f32s(QKV, &x_.qkv);
    cp.mem.write_f32s(P, &x_.p);
    cp.mem.write_f32s(DOUT, &x_.dout);
    cp.mem.write_f32s(a.ds, &x_.ds);
    cp.mem.write_f32s(DQKV, &vec![SENTINEL; s.nq() + 16]);
    cp.mem.write_u32(SEM, 0);
    let q = ComputeQmd {
        program_va: PROG,
        registers: prog.registers,
        cbuf0_va: CBUF,
        cbuf0_size: tc16::CBUF_BYTES as u32,
        grid: tc16::grid(&a),
        block: tc16::BLOCK,
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
    let got = cp.mem.read_f32s(DQKV, s.nq() + 16).unwrap();
    assert!(
        got[s.nq()..].iter().all(|&x| x == SENTINEL),
        "wrote after dqkv"
    );
    if res.is_err() {
        assert!(got.iter().all(|&x| x == SENTINEL), "refused but wrote");
    }
    res?;
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    Ok(got[..s.nq()].iter().map(|x| x.to_bits()).collect())
}

/// V5: the same bits as tc16 through the command processor on aligned shapes.
#[test]
fn v4_gives_tc16_bits_through_the_command_processor() {
    for (i, s) in SMALL.iter().enumerate() {
        let x_ = inputs(s, 200 + i as u64);
        assert!(v4::fits(&args(s, 0)), "{s:?}");
        let old = run(&TC16, s, 0, &x_).unwrap();
        let new = run(&V4, s, 0, &x_).unwrap();
        let newpf = run(&V4PF, s, 0, &x_).unwrap();
        assert!(
            old.iter().all(|&b| b != SENTINEL.to_bits()),
            "{s:?}: dqkv not written"
        );
        assert_eq!(old, new, "{s:?}");
        assert_eq!(old, newpf, "{s:?} (pf)");
    }
}

/// V5: unaligned shapes are refused by `fits` / `cbuf` and by the model, which writes nothing.
#[test]
fn unaligned_shapes_are_refused() {
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
        assert!(tc16::shape_ok(&a));
        assert!(!v4::fits(&a), "{s:?} +{shift}");
        assert!(v4::cbuf(&a).is_none());
        let x_ = inputs(&s, 7);
        assert!(run(&TC16, &s, shift, &x_).is_ok());
        for prog in [&V4, &V4PF] {
            let got = run(prog, &s, shift, &x_);
            assert!(got.is_err(), "{s:?} +{shift}: the model guessed");
        }
        assert!(!v4pf::fits(&a));
    }
    // the remaining addresses, one by one
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

/// Which sets of the OS's `--atentie-tc` probe v4 / v4pf may take (buffer at a page): all but `t130-d64` (T = 130).
#[test]
fn probe_sets_that_fit() {
    use nvidia_ga104::attn_tile_probe::SETS;
    let fit: Vec<&str> = SETS
        .iter()
        .filter(|s| v4::fits(&s.back_args(0x10_0000, Mode::Columns)))
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
    for s in SETS {
        let a = s.back_args(0x10_0000, Mode::Columns);
        assert_eq!(v4::fits(&a), v4pf::fits(&a));
    }
}
