//! T235 (C-157 p.2 after os-423, `docs/C-157_P2_CITIRI_IN_AVANS.md`), SIMULARE: `mm_tc` with the global loads issued
//! ahead (`tools/nak/mm_tc_pf.comp` → `tools/nak/out2/mm_tc_pf.sm86.sass`, second NAK tool, Mesa 26.3.1).
//! Checked: the binary is the recorded build, uses the tensor cores like `mm_tc`, spills nothing, and its loads are in
//! flight together (the T234 rule); the GLSL change itself — the branch-free read (index clamped to element 0, value
//! replaced by 0 afterwards) and the next slice read one step ahead — written out in Rust thread by thread puts the
//! SAME values in shared memory as `mm_tc`'s branchy read, for every block, slice, thread and flag, and never reads
//! outside A or B; both programs through our pushbuffer + QMD in the command-processor model give the same bits (the
//! CPU stand-ins share `gpu_compute::mm_tc::model`, matched by `mm_tc` on the board in os-423).
//! Nothing here runs on the RTX 3070 Ti.

use gpu_compute::mm_tc::Product;
use gpu_sim::cmdproc::{
    CommandProcessor, MM_TC_NEEDS, MM_TC_PF_NEEDS, ProgramFn, mm_tc_on_cpu, mm_tc_pf_on_cpu,
};
use nvidia_ga104::mm_tc::{self as tc, MM_TC};
use nvidia_ga104::mm_tc_pf::{self as pf, MM_TC_PF};
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
const MM_TC_PF_P: Prog = Prog {
    code: MM_TC_PF,
    run: mm_tc_pf_on_cpu,
    needs: MM_TC_PF_NEEDS,
    registers: pf::REGISTERS,
    shared: pf::SHARED,
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
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(prog.code, prog.run, prog.needs);
    cp.mem.write(PROG, prog.code);
    cp.mem.write(CBUF, &args(p).cbuf().unwrap());
    cp.mem.write_f32s(A, &p.a[..p.m * p.k]);
    cp.mem.write_f32s(B, &p.b[..p.k * p.n]);
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

/// `mm_tc` and `mm_tc_pf` on the same product, the same launch: the same bits.
fn same_bits(what: &str, p: &Product<'_>) {
    let a = args(p);
    let old = run_with(&MM_TC_P, p, a.grid(), tc::BLOCK).unwrap();
    let new = run_with(&MM_TC_PF_P, p, pf::grid(&a), pf::BLOCK).unwrap();
    assert!(
        old.iter().all(|&b| b != SENTINEL.to_bits()),
        "{what}: C not written"
    );
    assert_eq!(old, new, "{what}: mm_tc_pf != mm_tc");
}

/// Rust transcription of the per-thread reads of both GLSL programs for one K-slice starting at `t` of block
/// (`bx`, `by`): returns, per element of the f16 slices `as_` (64×32) and `bs_` (32×64), the f32 value the thread
/// rounds to f16 and stores (see [`Read`]); the clamped load index is checked against the buffer size.
#[derive(Clone, Copy, PartialEq)]
enum Read {
    /// `mm_tc`: `if (inside) x = load(at)`.
    Branch,
    /// `mm_tc_pf`: `v = load(inside ? at : 0); x = inside ? v : 0`.
    Ahead,
    /// Planted mistake: the clamped load without the zero chosen afterwards.
    AheadNoSelect,
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
    for tid in 0..TH {
        for s in 0..(BM * BK) / TH {
            let e = tid + s * TH;
            let (r, q) = if ta {
                (e % BM, e / BM)
            } else {
                (e / BK, e % BK)
            };
            let (gr, gp) = (row0 + r, t + q);
            let inside = gr < p.m && gp < p.k;
            let at = if ta { gp * p.m + gr } else { gr * p.k + gp };
            let x = if how == Read::Branch {
                if inside { p.a[at] } else { 0.0 }
            } else {
                let idx = if inside { at } else { 0 };
                assert!(idx < p.m * p.k, "read outside A");
                let v = p.a[idx];
                if inside || how == Read::AheadNoSelect {
                    v
                } else {
                    0.0
                }
            };
            assert_eq!(a_s[r * BK + q], u32::MAX, "two writers");
            a_s[r * BK + q] = x.to_bits();
        }
        for s in 0..(BK * BN) / TH {
            let e = tid + s * TH;
            let (q, cl) = if tb {
                (e % BK, e / BK)
            } else {
                (e / BN, e % BN)
            };
            let (gp, gc) = (t + q, col0 + cl);
            let inside = gp < p.k && gc < p.n;
            let at = if tb { gc * p.k + gp } else { gp * p.n + gc };
            let x = if how == Read::Branch {
                if inside { p.b[at] } else { 0.0 }
            } else {
                let idx = if inside { at } else { 0 };
                assert!(idx < p.k * p.n, "read outside B");
                let v = p.b[idx];
                if inside || how == Read::AheadNoSelect {
                    v
                } else {
                    0.0
                }
            };
            assert_eq!(b_s[q * BN + cl], u32::MAX, "two writers");
            b_s[q * BN + cl] = x.to_bits();
        }
    }
    assert!(
        a_s.iter().chain(&b_s).all(|&x| x != u32::MAX),
        "a hole in the slice"
    );
    (a_s, b_s)
}

/// `mm_tc_pf`'s loop order in Rust: slice 0 read before the loop, slice t+32 read (into the same registers) after the
/// barrier of slice t, only while t + 32 < k. Returns the slices in the order the tensor cores see them.
fn slices_ahead(p: &Product<'_>, bx: usize, by: usize) -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut regs = slice(p, bx, by, 0, Read::Ahead);
    let mut seen = Vec::new();
    let mut t = 0;
    while t < p.k {
        seen.push(regs.clone()); // stored to shared memory, barrier
        if t + 32 < p.k {
            regs = slice(p, bx, by, t + 32, Read::Ahead);
        }
        t += 32;
    }
    seen
}

#[test]
fn t235a_binary_is_the_recorded_build_and_uses_the_tensor_cores() {
    let hex: String = nvidia_ga104::sha256::digest(MM_TC_PF)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, pf::MM_TC_PF_SHA256);
    assert_eq!(MM_TC_PF.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out2/mm_tc_pf.sm86.txt");
    let old = include_str!("../../../tools/nak/out2/mm_tc.sm86.txt");
    let hmma = |t: &str| t.lines().filter(|l| l.contains("hmma")).count();
    eprintln!("T235: hmma mm_tc_pf {} / mm_tc {}", hmma(txt), hmma(old));
    assert_eq!(hmma(txt), hmma(old), "the same tensor-core chain");
    assert!(txt.contains("bar"), "no barrier in the dump");
    // The pointers to A and B are read once into uniform registers (`ldc c[0x0][+0x700]`), not per load.
    assert!(
        txt.contains("c[0x0][+0x700]"),
        "push constants not at 0x700"
    );
    assert!(
        !txt.contains("ldl") && !txt.contains("stl"),
        "register spills"
    );
    // P1 predicted 80-112 registers: MISSED, NAK 26.3.1 gives 248 (the dump's "Num GPRs"); 128 threads x 248 still
    // fit 2 blocks per SM by registers, as many as the shared memory allows at the 64 KiB step.
    assert!(txt.contains("Num GPRs: 248"));
    const { assert!(pf::REGISTERS == 248 && 2 * pf::BLOCK[0] * pf::REGISTERS <= 65_536) };
    assert_eq!(pf::SHARED.bytes, MM_TC_PF_NEEDS.shared_bytes);
    assert_eq!(pf::SHARED.bytes, tc::SHARED.bytes);
}

#[test]
fn t235b_the_reads_ahead_put_the_same_values_in_shared_memory() {
    let mut r = rng(0x0423_0229_c157_0002);
    let mut checked = 0usize;
    for (m, k, n) in [
        (64usize, 32usize, 64usize),
        (65, 33, 63),
        (5, 1, 3),
        (130, 96, 70),
        (1, 200, 129),
        (100, 64, 1),
    ] {
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
            for by in 0..m.div_ceil(64) {
                for bx in 0..n.div_ceil(64) {
                    let ahead = slices_ahead(&p, bx, by);
                    assert_eq!(ahead.len(), k.div_ceil(32));
                    for (i, s) in ahead.iter().enumerate() {
                        let old = slice(&p, bx, by, 32 * i, Read::Branch);
                        assert_eq!(
                            *s, old,
                            "{m}x{k}x{n} flags {flags} block {bx},{by} slice {i}"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    eprintln!("T235b: {checked} slices identical");
    assert!(checked > 100);
}

#[test]
fn t235c_a_planted_mistake_in_the_transcription_is_caught() {
    // The clamped load without the zero chosen afterwards puts A[0] / B[0] where `mm_tc` puts 0: caught on an edge
    // block, harmless (identical) on a full inner block.
    let mut r = rng(0x0002_350c);
    let (m, k, n) = (65usize, 33usize, 63usize);
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
        let edge = slice(&p, 0, 1, 32, Read::Branch);
        assert_ne!(
            edge,
            slice(&p, 0, 1, 32, Read::AheadNoSelect),
            "flags {flags}"
        );
        assert_eq!(edge, slice(&p, 0, 1, 32, Read::Ahead), "flags {flags}");
    }
    let (m, k, n) = (64usize, 32usize, 64usize);
    let p = Product {
        a: &a[..m * k],
        b: &b[..k * n],
        bias: None,
        m,
        n,
        k,
        flags: 0,
    };
    assert_eq!(
        slice(&p, 0, 0, 0, Read::Branch),
        slice(&p, 0, 0, 0, Read::AheadNoSelect)
    );
}

#[test]
fn t235d_same_bits_as_mm_tc_through_the_command_processor() {
    let mut r = rng(0x9e37_79b9_7f4a_7c15);
    for (m, k, n) in [
        (64usize, 32usize, 64usize),
        (130, 33, 129),
        (5, 1, 3),
        (200, 96, 70),
        (64, 200, 257),
    ] {
        let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
        let bias: Vec<f32> = (0..n).map(|_| r()).collect();
        for flags in 0..8u32 {
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
        }
    }
}

#[test]
fn t235e_model_refuses_wrong_block_and_short_grid() {
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
    let g = pf::grid(&args(&p));
    assert_eq!(g, [5, 4, 1]);
    assert!(
        run_with(&MM_TC_PF_P, &p, g, [256, 1, 1]).is_err(),
        "256 threads"
    );
    assert!(
        run_with(&MM_TC_PF_P, &p, [4, 4, 1], pf::BLOCK).is_err(),
        "short grid"
    );
    assert!(run_with(&MM_TC_PF_P, &p, g, pf::BLOCK).is_ok());
}
