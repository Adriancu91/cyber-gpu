//! T226 (C-165 after os-416, `docs/C-165_CURSA_MM.md` §„`mm_tc128`”), SIMULARE: `mm_tc` on 128×128 tiles
//! (`tools/nak/mm_tc128.comp` → `tools/nak/out2/mm_tc128.sm86.sass`, second NAK tool, Mesa 26.3.1). Both programs run
//! through our pushbuffer + QMD in the command-processor model; the "programs" run on the CPU
//! (`mm_tc_on_cpu` / `mm_tc128_on_cpu`, the same tensor-core adder HYPOTHESIS fitted on os-335). Checked: the binary
//! is the recorded build and uses the tensor cores; the same bits as `mm_tc` on edge shapes (all flags) and on the 15
//! matmul shapes of step #513 (rows and columns cut to keep the CPU time short, K whole); the model refuses the wrong
//! block and a grid that does not cover C. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::mm_tc::Product;
use gpu_compute::train::{self, AdamW, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, MM_TC_NEEDS, MM_TC128_NEEDS, ProgramFn, mm_tc_on_cpu, mm_tc128_on_cpu,
};
use nvidia_ga104::mm_tc::{self as tc, MM_TC};
use nvidia_ga104::mm_tc128::{self as tc128, MM_TC128};
use nvidia_ga104::mm_train_variants as mv;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_ga104::train_step::Plan;
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
const HEAP: u64 = nvidia_ga104::mmu::VA_BASE + 0x100_0000;

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
const MM_TC128_P: Prog = Prog {
    code: MM_TC128,
    run: mm_tc128_on_cpu,
    needs: MM_TC128_NEEDS,
    registers: tc128::REGISTERS,
    shared: tc128::SHARED,
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

/// `mm_tc` (64×64 tiles) and `mm_tc128` on the same product: the same bits.
fn same_bits(what: &str, p: &Product<'_>) {
    let a = args(p);
    let old = run_with(&MM_TC_P, p, a.grid(), tc::BLOCK).unwrap();
    let new = run_with(&MM_TC128_P, p, tc128::grid(&a), tc128::BLOCK).unwrap();
    assert!(
        old.iter().all(|&b| b != SENTINEL.to_bits()),
        "{what}: C not written"
    );
    assert_eq!(old, new, "{what}: mm_tc128 != mm_tc");
}

#[test]
fn t226_binary_is_the_recorded_build_and_uses_the_tensor_cores() {
    let hex: String = nvidia_ga104::sha256::digest(MM_TC128)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, tc128::MM_TC128_SHA256);
    assert_eq!(MM_TC128.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out2/mm_tc128.sm86.txt");
    // Prediction 2 (ratio): 2 × 4 accumulators per subgroup vs 2 × 2 -> twice the HMMA of mm_tc (each 16×16×16
    // coopMatMulAdd is two HMMA.16816, so 32 per K-slice of 32, not 16 as written in the prediction).
    let hmma = |t: &str| t.lines().filter(|l| l.contains("hmma")).count();
    let old = include_str!("../../../tools/nak/out2/mm_tc.sm86.txt");
    eprintln!("T226: hmma mm_tc128 {} / mm_tc {}", hmma(txt), hmma(old));
    assert_eq!(
        hmma(txt),
        2 * hmma(old),
        "twice the accumulators per subgroup"
    );
    assert!(txt.contains("bar"), "no barrier in the dump");
    assert!(txt.contains("c[0x0][0x700]"), "push constants not at 0x700");
    assert!(
        !txt.contains("ldl") && !txt.contains("stl"),
        "register spills"
    );
    const { assert!(tc128::REGISTERS <= 128, "prediction 1") };
    assert_eq!(tc128::SHARED.bytes, MM_TC128_NEEDS.shared_bytes);
    assert_eq!(tc128::SHARED.bytes, 24 * 1024);
}

#[test]
fn t226_same_bits_as_mm_tc_on_edges_all_flags() {
    let mut r = rng(0x9e37_79b9_7f4a_7c15);
    for (m, k, n) in [
        (128usize, 32usize, 128usize),
        (130, 33, 129),
        (5, 1, 3),
        (200, 96, 70),
        (64, 200, 257),
        (257, 48, 16),
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

/// The 15 matmul shapes of step #513 (`Plan::new_tiles3` + `with_mm_tile`, the shapes of the os-416 race), M and N
/// cut to at most 160 / 288 (one whole 128 tile + an edge) so the CPU model stays fast; K whole (the full chain).
#[test]
fn t226_same_bits_as_mm_tc_on_the_513_shapes() {
    let (bsz, tl) = (2usize, 512usize);
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let plan = Plan::new_tiles3(train::M513, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile()
        .resident()
        .unwrap();
    let params: Vec<Vec<f32>> = train::M513
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    let forms = mv::forms(&plan.step(Recipe::AdamW513, &opt, 1).unwrap());
    assert_eq!(forms.len(), 15);
    assert_eq!(forms.iter().map(|f| f.count).sum::<u32>(), 147);
    let mut r = rng(0x5151_0513_0416_0226);
    for f in &forms {
        // Full-size launch shape: the grid the OS would send.
        let full = tc::Args {
            a: A,
            b: B,
            c: C,
            bias: if f.flags & tc::BIAS != 0 { BIAS } else { 0 },
            m: f.m,
            n: f.n,
            k: f.k,
            flags: f.flags,
        };
        assert!(full.cbuf().is_some(), "{f:?}: mm_tc refuses the shape");
        let g = tc128::grid(&full);
        assert_eq!(
            (g[0], g[1]),
            (f.n.div_ceil(128), f.m.div_ceil(128)),
            "{f:?}"
        );
        let (m, n, k) = (
            (f.m as usize).min(160),
            (f.n as usize).min(288),
            f.k as usize,
        );
        let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
        let bias: Vec<f32> = (0..n).map(|_| r()).collect();
        let p = Product {
            a: &a,
            b: &b,
            bias: (f.flags & tc::BIAS != 0).then_some(&bias[..]),
            m,
            n,
            k,
            flags: f.flags,
        };
        same_bits(&format!("#513 {f:?} cut to {m}x{k}x{n}"), &p);
    }
}

#[test]
fn t226_model_refuses_wrong_block_and_short_grid() {
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
    let g = tc128::grid(&args(&p));
    assert_eq!(g, [3, 2, 1]);
    assert!(
        run_with(&MM_TC128_P, &p, g, [128, 1, 1]).is_err(),
        "128 threads"
    );
    assert!(
        run_with(&MM_TC128_P, &p, [2, 2, 1], tc128::BLOCK).is_err(),
        "short grid"
    );
    // mm_tc's grid (64×64 tiles) covers C too, but mm_tc refuses 256 threads.
    assert!(run_with(&MM_TC_P, &p, args(&p).grid(), tc128::BLOCK).is_err());
}
