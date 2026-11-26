//! T269 (C-157 p.2, `docs/C-157_MM_TC_V5_PLAN.md`, session 258), SIMULARE: `mm_tc_v4` with a 12 KiB block
//! (`tools/nak/mm_tc_v5.comp` → `tools/nak/out2/mm_tc_v5.sm86.sass`, second NAK tool, Mesa 26.3.1).
//! Checked: the binary is the recorded build (same tensor-core chain and 128-bit loads as `mm_tc_v4`, no spills to
//! memory, the registers and shared size the occupancy plan needs); the K loop of the GLSL is `mm_tc_v4`'s, character
//! for character; the new ending, written out in Rust lane by lane (each subgroup's four 16×16 pieces through its own
//! scratch), writes every element of C exactly once, from the right accumulator, and nothing outside C, and touches
//! only its own scratch; through our pushbuffer + QMD in the command-processor model `mm_tc_v5` gives the same bits as
//! `mm_tc` and refuses what `mm_tc_v4` refuses. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::mm_tc::Product;
use gpu_sim::cmdproc::{
    CommandProcessor, MM_TC_NEEDS, MM_TC_V5_NEEDS, ProgramFn, mm_tc_on_cpu, mm_tc_v5_on_cpu,
};
use nvidia_ga104::mm_tc::{self as tc, MM_TC};
use nvidia_ga104::mm_tc_v4::{self as v4};
use nvidia_ga104::mm_tc_v5::{self as v5, MM_TC_V5};
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
const MM_TC_V5_P: Prog = Prog {
    code: MM_TC_V5,
    run: mm_tc_v5_on_cpu,
    needs: MM_TC_V5_NEEDS,
    registers: v5::REGISTERS,
    shared: v5::SHARED,
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

/// One launch of `prog` on `p`, A and B placed `da` / `db` bytes after their usual addresses; `Ok(C)` or the refusal.
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

fn aligned_shapes(f: u32) -> Vec<(usize, usize, usize)> {
    [
        (64usize, 32usize, 64usize),
        (68, 36, 60),
        (4, 4, 4),
        (132, 96, 72),
        (8, 200, 132),
        (100, 64, 4),
        (61, 44, 57),
        (63, 33, 65),
    ]
    .into_iter()
    .filter(|&(m, k, n)| {
        v5::fits(&tc::Args {
            a: A,
            b: B,
            c: C,
            bias: 0,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            flags: f,
        })
    })
    .collect()
}

#[test]
fn t269a_binary_is_the_recorded_build_with_the_planned_shape() {
    let hex: String = nvidia_ga104::sha256::digest(MM_TC_V5)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, v5::MM_TC_V5_SHA256);
    assert_eq!(MM_TC_V5.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out2/mm_tc_v5.sm86.txt");
    let old = include_str!("../../../tools/nak/out2/mm_tc_v4.sm86.txt");
    let count = |t: &str, what: &str| t.lines().filter(|l| l.contains(what)).count();
    // V2: the same tensor-core chain, the same 128-bit loads, the same f32 -> f16 conversions
    assert_eq!(count(txt, "hmma"), count(old, "hmma"), "hmma");
    assert_eq!(count(txt, ".b128"), count(old, ".b128"), "128-bit loads");
    assert_eq!(count(txt, "= f2f"), count(old, "= f2f"), "f2f");
    assert_eq!(
        count(txt, "bar.sync"),
        count(old, "bar.sync") - 1,
        "V3: no block barrier before the ending"
    );
    assert!(
        txt.contains("c[0x0][+0x700]"),
        "push constants not at 0x700"
    );
    // V1: registers and shared memory as planned, nothing spilled to memory
    assert!(txt.contains("Spills to mem: 0") && txt.contains("SLM size: 0"));
    assert!(txt.contains(&format!("Num GPRs: {}", v5::REGISTERS)));
    const { assert!(v5::REGISTERS <= 128, "V1: more than 128 registers") };
    assert!(txt.contains(&format!("Shared size: {}", v5::SHARED.bytes)));
    assert_eq!(v5::SHARED.bytes, MM_TC_V5_NEEDS.shared_bytes);
    assert_eq!(v5::SHARED.bytes, 12_288);
    assert_eq!(v5::BLOCK, v4::BLOCK);
}

/// The GLSL between `void main()` and the store of the accumulators.
fn k_loop(src: &str) -> &str {
    let a = src.find("void main()").expect("main");
    let b = src[a..].find("    for (uint i = 0u; i < 2u; i++) {\n        for (uint j = 0u; j < 2u; j++) {\n            coopMatStore");
    let b2 = src[a..]
        .find("    // Each warp stores its 2x2 accumulators")
        .or(b)
        .expect("end of the K loop");
    &src[a..a + b2]
}

#[test]
fn t269b_the_k_loop_is_mm_tc_v4s() {
    let v4s = include_str!("../../../tools/nak/mm_tc_v4.comp");
    let v5s = include_str!("../../../tools/nak/mm_tc_v5.comp");
    assert_eq!(k_loop(v5s), k_loop(v4s));
    assert!(k_loop(v5s).contains("coopMatMulAdd"));
    // and load_slice, the slice declarations and the defines are the same text too
    let decl = |s: &str| -> String {
        let a = s.find("#define BM").unwrap();
        let b = s.find("void main()").unwrap();
        s[a..b]
            .lines()
            .filter(|l| !l.starts_with("shared float cs_") && !l.starts_with("shared float st_"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(decl(v5s), decl(v4s));
    assert!(v5s.contains("shared float st_[4u * 256u];"));
    assert!(!v5s.contains("cs_["), "the 64x64 f32 block is gone");
}

/// The ending of `mm_tc_v5.comp` written out: subgroup `w`, lane, accumulator (i, j), round s → (scratch index,
/// row, column) of C, for block (bx, by), only where the GLSL writes (inside C).
fn ending(
    m: usize,
    n: usize,
    bx: usize,
    by: usize,
    wrong: bool,
) -> Vec<(usize, usize, usize, usize, usize, usize)> {
    let mut out = Vec::new();
    for tid in 0..128usize {
        let w = tid / 32;
        let (wr, wc) = ((w / 2) * 32, (w % 2) * 32);
        let lane = tid % 32;
        let sb = w * 256;
        for i in 0..2 {
            for j in 0..2 {
                for s in 0..8 {
                    let e = lane + s * 32;
                    // the planted mistake: the scratch read column-major (lanes 16 apart along a column)
                    let (er, ec) = if wrong {
                        (e % 16, e / 16)
                    } else {
                        (e / 16, e % 16)
                    };
                    let gr = by * 64 + wr + 16 * i + er;
                    let gc = bx * 64 + wc + 16 * j + ec;
                    if gr < m && gc < n {
                        // (subgroup, scratch index, accumulator row in the 64x64 block, column, gr, gc): the GLSL reads
                        // st_[sb + e] and writes it to (gr, gc)
                        out.push((
                            w,
                            sb + e,
                            wr + 16 * i + e / 16,
                            wc + 16 * j + e % 16,
                            gr,
                            gc,
                        ));
                    }
                }
            }
        }
    }
    out
}

/// Does the ending (or the planted mistake) write every element of C exactly once, from the scratch slot where
/// `coopMatStore` (row-major, stride 16) put that element's accumulator value, of its own subgroup?
fn ending_is_right(wrong: bool) -> bool {
    for (m, n) in [(64usize, 64usize), (61, 57), (132, 72), (4, 4), (100, 130)] {
        let mut seen = vec![0u32; m * n];
        for by in 0..m.div_ceil(64) {
            for bx in 0..n.div_ceil(64) {
                for (w, idx, r, cl, gr, gc) in ending(m, n, bx, by, wrong) {
                    if !(w * 256..w * 256 + 256).contains(&idx) || idx >= 4 * 256 {
                        return false; // foreign or outside scratch
                    }
                    // idx holds the block element (r, cl), stored by subgroup w (rows (w/2)*32.., columns (w%2)*32..)
                    if (r / 32) * 2 + cl / 32 != w || idx - w * 256 != (r % 16) * 16 + cl % 16 {
                        return false;
                    }
                    if (gr, gc) != (by * 64 + r, bx * 64 + cl) {
                        return false; // written to another element of C
                    }
                    seen[gr * n + gc] += 1;
                }
            }
        }
        if !seen.iter().all(|&x| x == 1) {
            return false;
        }
    }
    true
}

#[test]
fn t269c_the_ending_writes_every_element_once_from_its_own_scratch() {
    assert!(ending_is_right(false));
    assert!(!ending_is_right(true), "the planted mistake must be caught");
}

#[test]
fn t269d_same_bits_as_mm_tc_through_the_command_processor() {
    let mut r = rng(0x0258_c157_0005);
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
            let ar = args(&p);
            let old = run_at(&MM_TC_P, &p, ar.grid(), tc::BLOCK, 0, 0).unwrap();
            let new = run_at(&MM_TC_V5_P, &p, v5::grid(&ar), v5::BLOCK, 0, 0).unwrap();
            assert!(
                old.iter().all(|&x| x != SENTINEL.to_bits()),
                "C not written"
            );
            assert_eq!(old, new, "{m}x{k}x{n} flags {flags}: mm_tc_v5 != mm_tc");
            runs += 1;
        }
    }
    eprintln!("T269d: {runs} products, same bits");
    assert!(runs >= 30);
}

#[test]
fn t269e_model_refuses_what_mm_tc_v4_refuses() {
    let mut r = rng(9);
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
    let g = v5::grid(&args(&p));
    assert!(
        run_at(&MM_TC_V5_P, &p, g, [256, 1, 1], 0, 0).is_err(),
        "256 threads"
    );
    assert!(
        run_at(&MM_TC_V5_P, &p, [4, 4, 1], v5::BLOCK, 0, 0).is_err(),
        "short grid"
    );
    assert!(run_at(&MM_TC_V5_P, &p, g, v5::BLOCK, 0, 0).is_ok());
    for (da, db) in [(4u64, 0u64), (0, 8), (12, 12)] {
        assert!(
            run_at(&MM_TC_V5_P, &p, g, v5::BLOCK, da, db).is_err(),
            "+{da}/+{db}"
        );
    }
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
        assert!(!v5::fits(&args(&p)));
        assert!(run_at(&MM_TC_V5_P, &p, v5::grid(&args(&p)), v5::BLOCK, 0, 0).is_err());
    }
}
