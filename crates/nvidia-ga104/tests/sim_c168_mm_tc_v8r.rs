//! T303 (C-168 p.3 follow-up, `docs/C-168_V8R.md` §4–§5, session 287), SIMULARE: the module `nvidia_ga104::mm_tc_v8r`
//! (v8 with the fragment reads of a slice reordered, `tools/nak/out2/mm_tc_v8r.sm86.sass`, Mesa NAK 26.3.1). Checked:
//! - the binary is the recorded build; the module's QMD numbers are the listing's (120 registers, 24 576 B shared, no
//!   spill, push constants at 0x700) and give 2 blocks per SM at 64 KiB; the K loop has v8's reads (16 `hmma`, 4 `ldsm.x4`,
//!   4 `ldsm.x2`, no plain shared load, one barrier) and 84 register copies (v8: 148);
//! - through our pushbuffer + QMD (with v8r's OWN registers) in the command-processor model: the same bits as `mm_tc`
//!   and as `mm_tc_v8` on the edge sets (TA / TB / BIAS, partial blocks), v8 and v8r in one command processor;
//! - refusals: unaligned A, a 128-thread block (v7 / v8n), a grid too small for C — nothing half computed.
//!
//! Nothing here runs on the RTX 3070 Ti: the bits on the board and the speed are predictions (V1–V4), not results.

use gpu_compute::mm_tc::Product;
use gpu_sim::cmdproc::{
    CommandProcessor, MM_TC_NEEDS, MM_TC_V8_NEEDS, MM_TC_V8R_NEEDS, ProgramFn, mm_tc_on_cpu,
    mm_tc_v8_on_cpu, mm_tc_v8r_on_cpu,
};
use nvidia_ga104::mm_tc::{self as tc, MM_TC};
use nvidia_ga104::mm_tc_v8::{self as v8, MM_TC_V8};
use nvidia_ga104::mm_tc_v8r::{self as v8r, MM_TC_V8R};
use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const LISTING_V8R: &str = include_str!("../../../tools/nak/out2/mm_tc_v8r.sm86.txt");

fn count(t: &str, what: &str) -> usize {
    t.lines().filter(|l| l.contains(what)).count()
}

/// The lines of the K loop: from the block holding the first `ldsm` to the block barrier that ends the slice.
fn k_loop(listing: &str) -> Vec<&str> {
    let lines: Vec<&str> = listing.lines().collect();
    let first = lines.iter().position(|l| l.contains("ldsm")).unwrap();
    let start = (0..first)
        .rev()
        .find(|&i| lines[i].starts_with("block"))
        .unwrap();
    let end = (first..lines.len())
        .find(|&i| lines[i].contains("bar.sync"))
        .unwrap();
    lines[start..=end].to_vec()
}

#[test]
fn t303a_module_numbers_are_the_listing() {
    let hex: String = nvidia_ga104::sha256::digest(MM_TC_V8R)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, v8r::MM_TC_V8R_SHA256);
    assert!(LISTING_V8R.contains("Spills to mem: 0") && LISTING_V8R.contains("SLM size: 0"));
    assert!(LISTING_V8R.contains(&format!("Num GPRs: {}", v8r::REGISTERS)));
    assert!(LISTING_V8R.contains(&format!("Shared size: {}", v8r::SHARED.bytes)));
    assert!(LISTING_V8R.contains("c[0x0][+0x700]"), "push constants");
    assert_eq!(v8r::REGISTERS, 120);
    assert_eq!(v8r::SHARED.bytes, MM_TC_V8R_NEEDS.shared_bytes);
    assert_eq!(MM_TC_V8R_NEEDS.barriers, MM_TC_V8_NEEDS.barriers);
    assert_eq!(v8r::BLOCK, [256, 1, 1]);
    assert_eq!(v8r::CBUF_BYTES, tc::CBUF_BYTES);
    let k = Kernel {
        name: "mm_tc_v8r",
        threads: v8r::BLOCK[0],
        registers: v8r::REGISTERS,
        shared_bytes: v8r::SHARED.bytes,
    };
    for r in [Reserve::None, Reserve::OneKib] {
        assert_eq!(
            blocks_per_sm(&SM86, &k, 64 << 10, r).unwrap().blocks,
            v8r::BLOCKS_PER_SM
        );
    }
    let txt = k_loop(LISTING_V8R).join("\n");
    assert_eq!(count(&txt, "hmma.m16n8k16.f32"), 16);
    assert_eq!(count(&txt, "ldsm.16.m8n8.x4"), 4);
    assert_eq!(count(&txt, "ldsm.16.m8n8.x2"), 4);
    assert_eq!(count(&txt, "ld.shared"), 0);
    assert_eq!(count(&txt, "bar.sync"), 1);
    assert_eq!(
        count(&txt, " = mov "),
        84,
        "register copies in the loop (v8: 148)"
    );
    assert_eq!(count(LISTING_V8R, "hmma"), 16);
    assert_eq!(count(LISTING_V8R, "movm"), 0);
    assert_eq!(count(LISTING_V8R, "ldsm"), 8);
}

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

fn run_at(
    prog: &Prog,
    p: &Product<'_>,
    grid: [u32; 3],
    block: [u32; 3],
    da: u64,
) -> Result<Vec<u32>, gpu_core::error::GpuError> {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(prog.code, prog.run, prog.needs);
    cp.mem.write(PROG, prog.code);
    let mut ar = args(p);
    ar.a += da;
    cp.mem.write(CBUF, &ar.cbuf().unwrap());
    cp.mem.write_f32s(A + da, &p.a[..p.m * p.k]);
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

#[test]
fn t303b_same_bits_as_mm_tc_and_v8_and_the_same_refusals() {
    let base = Prog {
        code: MM_TC,
        run: mm_tc_on_cpu,
        needs: MM_TC_NEEDS,
        registers: tc::REGISTERS,
        shared: tc::SHARED,
    };
    let p8 = Prog {
        code: MM_TC_V8,
        run: mm_tc_v8_on_cpu,
        needs: MM_TC_V8_NEEDS,
        registers: v8::REGISTERS,
        shared: v8::SHARED,
    };
    let p8r = Prog {
        code: MM_TC_V8R,
        run: mm_tc_v8r_on_cpu,
        needs: MM_TC_V8R_NEEDS,
        registers: v8r::REGISTERS,
        shared: v8r::SHARED,
    };
    let mut r = rng(0x168_0303);
    let mut sets = 0;
    for &(m, k, n) in &[
        (128usize, 16usize, 128usize),
        (132, 36, 68),
        (4, 4, 4),
        (260, 48, 196),
        (8, 200, 132),
        (256, 64, 384),
    ] {
        for flags in 0..8u32 {
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
            assert!(v8r::fits(&ar));
            assert_eq!(v8r::grid(&ar), v8::grid(&ar));
            let want = run_at(&base, &p, ar.grid(), tc::BLOCK, 0).unwrap();
            let g8 = run_at(&p8, &p, v8::grid(&ar), v8::BLOCK, 0).unwrap();
            let g8r = run_at(&p8r, &p, v8r::grid(&ar), v8r::BLOCK, 0).unwrap();
            assert_eq!(g8r, want, "v8r {m}x{k}x{n} flags {flags}");
            assert_eq!(g8r, g8, "v8r = v8 {m}x{k}x{n} flags {flags}");
            // refusals: A not 16-byte aligned, a 128-thread block
            assert!(run_at(&p8r, &p, v8r::grid(&ar), v8r::BLOCK, 4).is_err());
            assert!(run_at(&p8r, &p, v8r::grid(&ar), [128, 1, 1], 0).is_err());
            sets += 1;
        }
    }
    // a grid too small for C is refused, not half computed
    let a = vec![0.5f32; 256 * 16];
    let b = vec![0.25f32; 16 * 256];
    let p = Product {
        a: &a,
        b: &b,
        bias: None,
        m: 256,
        n: 256,
        k: 16,
        flags: 0,
    };
    assert!(run_at(&p8r, &p, [2, 1, 1], v8r::BLOCK, 0).is_err());
    let mut ar = args(&p);
    ar.k = 18;
    assert!(!v8r::fits(&ar));
    assert_eq!(sets, 48);
}

/// v8 and v8r loaded in ONE command processor (as the OS probe will have them), one after the other on the same
/// product into two outputs: each launch is dispatched to its own program (by code), the two outputs are equal.
#[test]
fn t303c_v8_and_v8r_side_by_side() {
    const PROG2: u64 = 0x18_0000;
    const C2: u64 = 0x380_0000;
    let (m, k, n) = (384usize, 96usize, 256usize);
    let mut r = rng(0x168_0304);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(MM_TC_V8, mm_tc_v8_on_cpu, MM_TC_V8_NEEDS);
    cp.register_program_with_needs(MM_TC_V8R, mm_tc_v8r_on_cpu, MM_TC_V8R_NEEDS);
    cp.mem.write(PROG, MM_TC_V8);
    cp.mem.write(PROG2, MM_TC_V8R);
    cp.mem.write_f32s(A, &a);
    cp.mem.write_f32s(B, &b);
    let mut words = Vec::new();
    for (i, (prog, regs, out)) in [(PROG, v8::REGISTERS, C), (PROG2, v8r::REGISTERS, C2)]
        .into_iter()
        .enumerate()
    {
        let ar = tc::Args {
            a: A,
            b: B,
            c: out,
            bias: 0,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            flags: 0,
        };
        let cb = CBUF + 0x1000 * i as u64;
        cp.mem.write(cb, &ar.cbuf().unwrap());
        cp.mem.write_f32s(out, &vec![SENTINEL; m * n]);
        let q = ComputeQmd {
            program_va: prog,
            registers: regs,
            cbuf0_va: cb,
            cbuf0_size: tc::CBUF_BYTES as u32,
            grid: v8::grid(&ar),
            block: v8::BLOCK,
            release: Some((SEM + 16 * i as u64, 1)),
        };
        let qa = QMD + 0x100 * i as u64;
        cp.mem
            .write(qa, &q.to_bytes_with_shared(&v8::SHARED).unwrap());
        words.push(qa);
    }
    let mut w = [0u32; 16];
    let mut pb = PushBuf::new(&mut w);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    for qa in words {
        pb.inc(1, c::NVC7C0_SEND_PCAS_A, &[(qa >> 8) as u32])
            .unwrap();
        pb.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
            .unwrap();
    }
    let len = pb.len();
    let bytes: Vec<u8> = w[..len].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, len as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    assert_eq!(cp.mem.read_u32(SEM + 16).unwrap(), 1);
    let g8 = cp.mem.read_f32s(C, m * n).unwrap();
    let g8r = cp.mem.read_f32s(C2, m * n).unwrap();
    assert!(g8.iter().all(|&x| x != SENTINEL));
    let b8: Vec<u32> = g8.iter().map(|x| x.to_bits()).collect();
    let b8r: Vec<u32> = g8r.iter().map(|x| x.to_bits()).collect();
    assert_eq!(b8, b8r);
}
