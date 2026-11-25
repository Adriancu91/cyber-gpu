//! T236 (C-157 p.2, `docs/C-157_P2_CITIRI_128.md`, session 230), SIMULARE: `mm_tc_pf` with 128-bit global loads
//! (`tools/nak/mm_tc_v4.comp` → `tools/nak/out2/mm_tc_v4.sm86.sass`, second NAK tool, Mesa 26.3.1).
//! Checked: the binary is the recorded build, uses the tensor cores like `mm_tc`, spills nothing and loads 128 bits at
//! a time; the GLSL reads written out in Rust thread by thread (4 + 4 vec4 per thread per slice, index ×0 outside,
//! zero chosen at the shared store, 4 lanes stored along the row or down the column) put the SAME values in shared
//! memory as `mm_tc`'s branchy read for every aligned product, block, slice and flag, and every vec4 is aligned and
//! wholly inside its buffer; unaligned products are refused by `fits` and by the model (no write); through our
//! pushbuffer + QMD in the command-processor model both programs give the same bits.
//! Nothing here runs on the RTX 3070 Ti.

use gpu_compute::mm_tc::Product;
use gpu_sim::cmdproc::{
    CommandProcessor, MM_TC_NEEDS, MM_TC_V4_NEEDS, ProgramFn, mm_tc_on_cpu, mm_tc_v4_on_cpu,
};
use nvidia_ga104::mm_tc::{self as tc, MM_TC};
use nvidia_ga104::mm_tc_v4::{self as v4, MM_TC_V4};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const A: u64 = 0x100_0000;
const B: u64 = 0x200_0000;
const BIAS: u64 = 0x2f0_0000;
const C: u64 = 0x300_0000;
const QMD: u64 = 0x70_0000;
const PUSH: u64 = 0x71_0000;
const SENTINEL: f32 = -12345.0;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

struct Prog {
    code: &'static [u8],
    run: ProgramFn,
    needs: gpu_sim::cmdproc::Needs,
    registers: u32,
    shared: SharedMemory,
}

const MM_TC_P: Prog = Prog {
    code: MM_TC,
    run: mm_tc_on_cpu,
    needs: MM_TC_NEEDS,
    registers: tc::REGISTERS,
    shared: tc::SHARED,
};
const MM_TC_V4_P: Prog = Prog {
    code: MM_TC_V4,
    run: mm_tc_v4_on_cpu,
    needs: MM_TC_V4_NEEDS,
    registers: v4::REGISTERS,
    shared: v4::SHARED,
};

fn args(p: &Product<'_>) -> tc::Args {
    tc::Args {
        a: A,
        b: B,
        c: C,
        bias: if p.flags & tc::BIAS != 0 { BIAS } else { 0 },
        m: p.m as u32,
        n: p.n as u32,
        k: p.k as u32,
        flags: p.flags,
    }
}

/// One launch of `prog` on `p` with `grid` / `block`; `Ok(C)` or the processor's refusal. Checks nothing after C.
fn run_with(
    prog: &Prog,
    p: &Product<'_>,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<Vec<u32>, gpu_core::error::GpuError> {
    run_at(prog, p, grid, block, 0, 0)
}

/// [`run_with`] with A and B placed `da` / `db` bytes after their usual (16-byte aligned) addresses.
fn run_at(
    prog: &Prog,
    p: &Product<'_>,
    grid: [u32; 3],
    block: [u32; 3],
    da: u64,
    db: u64,
) -> Result<Vec<u32>, gpu_core::error::GpuError> {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(prog.code, prog.run, prog.needs);
    cp.mem.write(PROG, prog.code);
    let mut ar = args(p);
    ar.a += da;
    ar.b += db;
    cp.mem.write(CBUF, &ar.cbuf().unwrap());
    cp.mem.write_f32s(A + da, &p.a[..p.m * p.k]);
    cp.mem.write_f32s(B + db, &p.b[..p.k * p.n]);
    if let Some(bias) = p.bias {
        cp.mem.write_f32s(BIAS, bias);
    }
    cp.mem.write_f32s(C, &vec![SENTINEL; p.m * p.n + 16]);
    cp.mem.write_u32(SEM, 0);
    let q = ComputeQmd {
        program_va: PROG,
        registers: prog.registers,
        cbuf0_va: CBUF,
        cbuf0_size: tc::CBUF_BYTES as u32,
        grid,
        block,
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
    cp.run_gpfifo(&[(e.entry0, e.entry1)])?;
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let got = cp.mem.read_f32s(C, p.m * p.n + 16).unwrap();
    assert!(
        got[p.m * p.n..].iter().all(|&x| x == SENTINEL),
        "wrote after C"
    );
    Ok(got[..p.m * p.n].iter().map(|x| x.to_bits()).collect())
}

/// `mm_tc` and `mm_tc_v4` on the same product, the same launch: the same bits.
fn same_bits(what: &str, p: &Product<'_>) {
    let a = args(p);
    assert!(v4::fits(&a), "{what}: not aligned");
    let old = run_with(&MM_TC_P, p, a.grid(), tc::BLOCK).unwrap();
    let new = run_with(&MM_TC_V4_P, p, v4::grid(&a), v4::BLOCK).unwrap();
    assert!(
        old.iter().all(|&b| b != SENTINEL.to_bits()),
        "{what}: C not written"
    );
    assert_eq!(old, new, "{what}: mm_tc_v4 != mm_tc");
}

/// What one thread stores into the f16 slices for the K-slice at `t` of block (`bx`, `by`): the f32 value of every
/// element of `as_` (64×32) and `bs_` (32×64), as bits. `Branch` = `mm_tc` (`if (inside) x = load(at)`); `V4` =
/// `mm_tc_v4` (4 + 4 vec4 per thread, index ×0 outside, zero at the store); `V4WrongLanes` = a planted mistake (the
/// 4 lanes of a transposed vector stored along the row instead of down the column).
#[derive(Clone, Copy, PartialEq)]
enum Read {
    Branch,
    V4,
    V4WrongLanes,
}

fn slice(p: &Product<'_>, bx: usize, by: usize, t: usize, how: Read) -> (Vec<u32>, Vec<u32>) {
    const BM: usize = 64;
    const BN: usize = 64;
    const BK: usize = 32;
    const TH: usize = 128;
    let (ta, tb) = (p.flags & tc::TA != 0, p.flags & tc::TB != 0);
    let (row0, col0) = (by * BM, bx * BN);
    let mut a_s = vec![u32::MAX; BM * BK];
    let mut b_s = vec![u32::MAX; BK * BN];
    if how == Read::Branch {
        for e in 0..BM * BK {
            let (r, q) = if ta {
                (e % BM, e / BM)
            } else {
                (e / BK, e % BK)
            };
            let (gr, gp) = (row0 + r, t + q);
            let at = if ta { gp * p.m + gr } else { gr * p.k + gp };
            let x = if gr < p.m && gp < p.k { p.a[at] } else { 0.0 };
            a_s[r * BK + q] = x.to_bits();
        }
        for e in 0..BK * BN {
            let (q, cl) = if tb {
                (e % BK, e / BK)
            } else {
                (e / BN, e % BN)
            };
            let (gp, gc) = (t + q, col0 + cl);
            let at = if tb { gc * p.k + gp } else { gp * p.n + gc };
            let x = if gp < p.k && gc < p.n { p.b[at] } else { 0.0 };
            b_s[q * BN + cl] = x.to_bits();
        }
        return (a_s, b_s);
    }
    // The vec4 load of the GLSL: index at/4 times the inside bit; the whole vector must be aligned and in the buffer.
    let load4 = |buf: &[f32], at: usize, inside: bool, what: &str| -> [f32; 4] {
        let i = if inside { at / 4 } else { 0 };
        assert_eq!(at % 4, 0, "{what}: vector not aligned");
        assert!(4 * i + 4 <= buf.len(), "{what}: vector reads outside");
        [buf[4 * i], buf[4 * i + 1], buf[4 * i + 2], buf[4 * i + 3]]
    };
    for tid in 0..TH {
        for s in 0..(BM * BK) / (4 * TH) {
            let e = tid + s * TH;
            let (r, q) = if ta {
                (4 * (e % 16), e / 16)
            } else {
                (e / 8, 4 * (e % 8))
            };
            let (gr, gp) = (row0 + r, t + q);
            let inside = gr < p.m && gp < p.k;
            if inside {
                // The claim fits() rests on: a vector inside its first element is inside for all four.
                let last = if ta { (gr + 3, gp) } else { (gr, gp + 3) };
                assert!(last.0 < p.m && last.1 < p.k, "A vector straddles the edge");
            }
            let at = if ta { gp * p.m + gr } else { gr * p.k + gp };
            let v = load4(&p.a[..p.m * p.k], at, inside, "A");
            let step = if ta && how != Read::V4WrongLanes {
                BK
            } else {
                1
            };
            for (i, x) in v.iter().enumerate() {
                let x = if inside { *x } else { 0.0 };
                let idx = r * BK + q + i * step;
                assert_eq!(a_s[idx], u32::MAX, "two writers in as_");
                a_s[idx] = x.to_bits();
            }
        }
        for s in 0..(BK * BN) / (4 * TH) {
            let e = tid + s * TH;
            let (q, cl) = if tb {
                (4 * (e % 8), e / 8)
            } else {
                (e / 16, 4 * (e % 16))
            };
            let (gp, gc) = (t + q, col0 + cl);
            let inside = gp < p.k && gc < p.n;
            if inside {
                let last = if tb { (gp + 3, gc) } else { (gp, gc + 3) };
                assert!(last.0 < p.k && last.1 < p.n, "B vector straddles the edge");
            }
            let at = if tb { gc * p.k + gp } else { gp * p.n + gc };
            let v = load4(&p.b[..p.k * p.n], at, inside, "B");
            let step = if tb { BN } else { 1 };
            for (i, x) in v.iter().enumerate() {
                let x = if inside { *x } else { 0.0 };
                let idx = q * BN + cl + i * step;
                assert_eq!(b_s[idx], u32::MAX, "two writers in bs_");
                b_s[idx] = x.to_bits();
            }
        }
    }
    assert!(
        a_s.iter().chain(&b_s).all(|&x| x != u32::MAX),
        "a hole in the slice"
    );
    (a_s, b_s)
}

/// Aligned shapes (m, k, n) for flags `f`: the row lengths fits() looks at are multiples of 4; the others are not
/// (edges inside a tile).
fn aligned_shapes(f: u32) -> Vec<(usize, usize, usize)> {
    let mut v = Vec::new();
    for (m, k, n) in [
        (64usize, 32usize, 64usize),
        (68, 36, 60),
        (4, 4, 4),
        (132, 96, 72),
        (8, 200, 132),
        (100, 64, 4),
        (61, 44, 57),
        (63, 33, 65),
    ] {
        let a = tc::Args {
            a: A,
            b: B,
            c: C,
            bias: 0,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            flags: f,
        };
        if v4::fits(&a) {
            v.push((m, k, n));
        }
    }
    v
}

#[test]
fn t236a_binary_is_the_recorded_build_and_loads_128_bits() {
    let hex: String = nvidia_ga104::sha256::digest(MM_TC_V4)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, v4::MM_TC_V4_SHA256);
    assert_eq!(MM_TC_V4.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out2/mm_tc_v4.sm86.txt");
    let old = include_str!("../../../tools/nak/out2/mm_tc.sm86.txt");
    let hmma = |t: &str| t.lines().filter(|l| l.contains("hmma")).count();
    assert_eq!(hmma(txt), hmma(old), "the same tensor-core chain");
    assert!(txt.contains("bar"), "no barrier in the dump");
    assert!(
        txt.contains("c[0x0][+0x700]"),
        "push constants not at 0x700"
    );
    assert!(
        !txt.contains("ldl") && !txt.contains("stl"),
        "register spills"
    );
    let loads: Vec<&str> = txt.lines().filter(|l| l.contains("ld.global")).collect();
    let wide = loads.iter().filter(|l| l.contains(".b128")).count();
    eprintln!("T236a: {} ld.global, {wide} of 128 bits", loads.len());
    assert!(
        wide >= 16,
        "P1: the 4 + 4 vectors of a slice, twice (before the loop and in it)"
    );
    assert!(txt.contains(&format!("Num GPRs: {}", v4::REGISTERS)));
    const { assert!(v4::BLOCK[0] * v4::REGISTERS <= 65_536) };
    assert_eq!(v4::SHARED.bytes, MM_TC_V4_NEEDS.shared_bytes);
    assert_eq!(v4::SHARED.bytes, tc::SHARED.bytes);
}

#[test]
fn t236b_the_128_bit_reads_put_the_same_values_in_shared_memory() {
    let mut r = rng(0x0423_0230_c157_0128);
    let mut checked = 0usize;
    for flags in 0..4u32 {
        for (m, k, n) in aligned_shapes(flags) {
            let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
            let p = Product {
                a: &a,
                b: &b,
                bias: None,
                m,
                n,
                k,
                flags,
            };
            for by in 0..m.div_ceil(64) {
                for bx in 0..n.div_ceil(64) {
                    // Every slice the loop sees, plus the read after the last one (all outside, never used).
                    for i in 0..=k.div_ceil(32) {
                        let new = slice(&p, bx, by, 32 * i, Read::V4);
                        if i < k.div_ceil(32) {
                            let old = slice(&p, bx, by, 32 * i, Read::Branch);
                            assert!(
                                new == old,
                                "{m}x{k}x{n} flags {flags} block {bx},{by} slice {i}"
                            );
                            checked += 1;
                        } else {
                            assert!(new.0.iter().chain(&new.1).all(|&x| x == 0));
                        }
                    }
                }
            }
        }
    }
    eprintln!("T236b: {checked} slices identical");
    assert!(checked > 100);
}

#[test]
fn t236c_planted_mistake_and_unaligned_shapes_are_caught() {
    let mut r = rng(0x0002_360c);
    let (m, k, n) = (64usize, 32usize, 64usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    let p = Product {
        a: &a,
        b: &b,
        bias: None,
        m,
        n,
        k,
        flags: tc::TA,
    };
    let ok = std::panic::catch_unwind(|| slice(&p, 0, 0, 0, Read::V4WrongLanes));
    assert!(
        ok.is_err() || ok.unwrap() != slice(&p, 0, 0, 0, Read::Branch),
        "the wrong lane order must not pass"
    );
    // A shape fits() refuses really would read a straddling or unaligned vector: the transcription catches it.
    let (m, k, n) = (64usize, 33usize, 64usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    let p = Product {
        a: &a,
        b: &b,
        bias: None,
        m,
        n,
        k,
        flags: 0,
    };
    assert!(!v4::fits(&args(&p)));
    assert!(std::panic::catch_unwind(|| slice(&p, 0, 0, 0, Read::V4)).is_err());
}

#[test]
fn t236d_same_bits_as_mm_tc_through_the_command_processor() {
    let mut r = rng(0x9e37_79b9_7f4a_7c16);
    let mut runs = 0;
    for flags in 0..8u32 {
        for (m, k, n) in aligned_shapes(flags & 3) {
            let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
            let bias: Vec<f32> = (0..n).map(|_| r()).collect();
            let p = Product {
                a: &a,
                b: &b,
                bias: (flags & tc::BIAS != 0).then_some(&bias[..]),
                m,
                n,
                k,
                flags,
            };
            same_bits(&format!("{m}x{k}x{n} flags {flags}"), &p);
            runs += 1;
        }
    }
    eprintln!("T236d: {runs} products, same bits");
    assert!(runs >= 30);
}

#[test]
fn t236e_model_refuses_unaligned_products_wrong_block_and_short_grid() {
    let mut r = rng(7);
    let (m, k, n) = (200usize, 32usize, 300usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    let p = Product {
        a: &a,
        b: &b,
        bias: None,
        m,
        n,
        k,
        flags: 0,
    };
    let g = v4::grid(&args(&p));
    assert_eq!(g, [5, 4, 1]);
    assert!(
        run_with(&MM_TC_V4_P, &p, g, [256, 1, 1]).is_err(),
        "256 threads"
    );
    assert!(
        run_with(&MM_TC_V4_P, &p, [4, 4, 1], v4::BLOCK).is_err(),
        "short grid"
    );
    assert!(run_with(&MM_TC_V4_P, &p, g, v4::BLOCK).is_ok());
    // A or B 4 / 8 bytes off a 16-byte boundary: refused by the model and by fits(); mm_tc takes them.
    for (da, db) in [(4u64, 0u64), (0, 8), (12, 12)] {
        assert!(
            run_at(&MM_TC_V4_P, &p, g, v4::BLOCK, da, db).is_err(),
            "+{da}/+{db}"
        );
        assert!(run_at(&MM_TC_P, &p, g, tc::BLOCK, da, db).is_ok());
    }
    // Row lengths not a multiple of 4: refused, whatever the flags.
    let (m, k, n) = (66usize, 34usize, 62usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    for flags in 0..4u32 {
        let p = Product {
            a: &a,
            b: &b,
            bias: None,
            m,
            n,
            k,
            flags,
        };
        assert!(!v4::fits(&args(&p)));
        let g = v4::grid(&args(&p));
        assert!(
            run_with(&MM_TC_V4_P, &p, g, v4::BLOCK).is_err(),
            "flags {flags}"
        );
        assert!(run_with(&MM_TC_P, &p, g, tc::BLOCK).is_ok());
    }
}
