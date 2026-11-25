//! C-165 — T233, SIMULARE: the register diet of `mm_tv_128x128` (`docs/C-165_DIETA_128.md`). The program
//! `tools/nak/mm_tv_128x128t512.comp` (the same body `mm_train_tv.glsl`, 128 × 128 of C per block, 512 threads with
//! 8 × 4 results each) compiled by Mesa NAK 25.0.7, launched through our pushbuffer + QMD in the command processor
//! model; the "program" runs on the CPU (`mm_tv_128x128t512_on_cpu`). Checked bit for bit against
//! `reference::matmul_t_fma_ftz` and against `mm_tv_128x128` (the variant the #513 step uses since os-418). Nothing
//! here runs on the RTX 3070 Ti; the speed is for the race on the board.

use gpu_compute::train::{self, AdamW, Recipe};
use gpu_sim::cmdproc::{
    self as cpm, CommandProcessor, Event, MM_TRAIN_BIAS, MM_TRAIN_TA, MM_TRAIN_TB, Needs, ProgramFn,
};
use nvidia_ga104::mm_train::{self as mt};
use nvidia_ga104::mm_train_variants::{
    self as mv, MM_TV_128X128T512_VARIANT as DIET, VARIANTS, Variant,
};
use nvidia_ga104::occupancy::{self as occ, Reserve, SM86};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_ga104::train_step::Plan;
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const A: u64 = 0x40_0000;
const B: u64 = 0x80_0000;
const BIAS: u64 = 0xbf_0000;
const C: u64 = 0xc0_0000;
const QMD: u64 = 0x100_0000;
const PUSH: u64 = 0x101_0000;
const SENTINEL: f32 = -12345.0;
/// `mm_tv_128x128` in [`VARIANTS`].
const BIG: usize = 3;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

struct Job<'a> {
    a: &'a [f32],
    b: &'a [f32],
    bias: Option<&'a [f32]>,
    m: usize,
    k: usize,
    n: usize,
    flags: u32,
}

fn args(j: &Job<'_>) -> mt::Args {
    mt::Args {
        a: A,
        b: B,
        c: C,
        bias: BIAS,
        m: j.m as u32,
        n: j.n as u32,
        k: j.k as u32,
        flags: j.flags,
    }
}

fn model(v: &Variant) -> (ProgramFn, Needs) {
    if v.name == DIET.name {
        (
            cpm::mm_tv_128x128t512_on_cpu,
            cpm::MM_TV_128X128T512.needs(),
        )
    } else {
        assert_eq!(v.name, "mm_tv_128x128");
        (cpm::mm_tv_128x128_on_cpu, cpm::MM_TV_128X128.needs())
    }
}

fn cp_for(v: &Variant) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    let (f, needs) = model(v);
    cp.register_program_with_needs(v.code, f, needs);
    cp.mem.write(PROG, v.code);
    cp
}

fn load(cp: &mut CommandProcessor, j: &Job<'_>) {
    let mut cb = vec![0u8; 0x100];
    match args(j).cbuf() {
        Some(bytes) => cb[..mt::CBUF_BYTES].copy_from_slice(&bytes),
        None => cb[0x54..0x58].copy_from_slice(&j.flags.to_le_bytes()),
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(A, j.a);
    cp.mem.write_f32s(B, j.b);
    if let Some(bias) = j.bias {
        cp.mem.write_f32s(BIAS, bias);
    }
    cp.mem.write_f32s(C, &vec![SENTINEL; j.m * j.n + 16]);
    cp.mem.write_u32(SEM, 0);
}

fn launch(
    cp: &mut CommandProcessor,
    registers: u32,
    block: [u32; 3],
    grid: [u32; 3],
    shared: SharedMemory,
) -> Result<(), gpu_core::error::GpuError> {
    let q = ComputeQmd {
        program_va: PROG,
        registers,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid,
        block,
        release: Some((SEM, 1)),
    };
    cp.mem.write(QMD, &q.to_bytes_with_shared(&shared).unwrap());
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
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

fn run(v: &Variant, j: &Job<'_>) -> Vec<f32> {
    let mut cp = cp_for(v);
    load(&mut cp, j);
    launch(
        &mut cp,
        v.registers,
        v.block(),
        v.grid(&args(j)),
        v.shared(),
    )
    .unwrap();
    assert!(matches!(cp.events[1], Event::Launch { program: 0, .. }));
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let got = cp.mem.read_f32s(C, j.m * j.n + 16).unwrap();
    assert!(
        got[j.m * j.n..].iter().all(|&x| x == SENTINEL),
        "wrote after C"
    );
    got[..j.m * j.n].to_vec()
}

fn want(j: &Job<'_>) -> Vec<f32> {
    let mut out = vec![0.0; j.m * j.n];
    gpu_compute::reference::matmul_t_fma_ftz(
        j.a,
        j.b,
        j.bias,
        &mut out,
        j.m,
        j.k,
        j.n,
        j.flags & MM_TRAIN_TA != 0,
        j.flags & MM_TRAIN_TB != 0,
    )
    .unwrap();
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

const ALL_FLAGS: [u32; 8] = [
    0,
    MM_TRAIN_TA,
    MM_TRAIN_TB,
    MM_TRAIN_TA | MM_TRAIN_TB,
    MM_TRAIN_BIAS,
    MM_TRAIN_TA | MM_TRAIN_BIAS,
    MM_TRAIN_TB | MM_TRAIN_BIAS,
    MM_TRAIN_TA | MM_TRAIN_TB | MM_TRAIN_BIAS,
];

#[test]
fn t233a_binary_is_the_recorded_build_and_the_table_matches_the_model() {
    let hex: String = nvidia_ga104::sha256::digest(DIET.code)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, DIET.sha256);
    assert_eq!(DIET.code.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out/mm_tv_128x128t512.sm86.txt");
    assert!(txt.contains("bar"), "no barrier");
    assert!(txt.contains("SLM size: 0\n"), "local memory used");
    assert!(txt.contains(&format!("Num GPRs: {}\n", DIET.registers)));
    const { assert!(DIET.registers <= 80, "the diet: at most 80 registers") };
    let ffma = txt.matches("ffma.ftz").count() as u32;
    let fadd = txt.matches("fadd.ftz").count() as u32;
    assert!(ffma >= DIET.tm * DIET.tn);
    assert_eq!(fadd, DIET.tm * DIET.tn, "one bias add per result");
    assert_eq!(DIET.block(), [512, 1, 1]);
    let g = cpm::MM_TV_128X128T512;
    assert_eq!(
        (g.bm, g.bn, g.bk, g.tm, g.tn),
        (DIET.bm, DIET.bn, DIET.bk, DIET.tm, DIET.tn)
    );
    assert_eq!(g.threads(), 512);
    assert_eq!(DIET.shared().bytes, g.needs().shared_bytes);
    assert_eq!(
        DIET.shared().bytes,
        VARIANTS[BIG].shared().bytes,
        "same 8 KiB"
    );
    // Same pieces of C as mm_tv_128x128: the same grid on every shape.
    let a = mt::Args {
        a: 0,
        b: 0,
        c: 0,
        bias: 0,
        m: 1024,
        n: 2304,
        k: 768,
        flags: 0,
    };
    assert_eq!(DIET.grid(&a), VARIANTS[BIG].grid(&a));
    // The race tables are unchanged (5 entries, the diet is outside them).
    assert_eq!(VARIANTS.len(), 5);
    assert!(VARIANTS.iter().all(|v| v.name != DIET.name));
    println!(
        "kt t233 {}: {} octeti, {} registre, bloc {}, partajata {} B, {ffma} ffma, {fadd} fadd, sha {}",
        DIET.name,
        DIET.code.len(),
        DIET.registers,
        DIET.block()[0],
        DIET.shared().bytes,
        &DIET.sha256[..16]
    );
}

#[test]
fn t233b_edges_all_flags_and_negative_zero_bit_for_bit() {
    let mut r = rng(0x233b);
    for (m, k, n) in [
        (128usize, 8usize, 128usize),
        (129, 17, 65),
        (65, 9, 130),
        (37, 37, 21),
        (5, 1, 3),
        (1, 40, 1),
        (257, 33, 140),
    ] {
        let mut a: Vec<f32> = (0..m * k).map(|_| r()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
        let bias: Vec<f32> = (0..n).map(|_| r()).collect();
        a[0] = f32::from_bits(0x0000_1234);
        for flags in ALL_FLAGS {
            let j = Job {
                a: &a,
                b: &b,
                bias: (flags & MM_TRAIN_BIAS != 0).then_some(&bias[..]),
                m,
                k,
                n,
                flags,
            };
            let oracle = bits(&want(&j));
            assert_eq!(bits(&run(&DIET, &j)), oracle, "diet {m}x{k}x{n} f{flags}");
            assert_eq!(
                bits(&run(&VARIANTS[BIG], &j)),
                oracle,
                "128x128 {m}x{k}x{n} f{flags}"
            );
        }
    }
    // -0.0 survives, no zero padding accumulated (K = 19 = 8·2 + 3).
    let tiny = f32::from_bits(0x1c80_0000);
    let (m, k, n) = (3usize, 19usize, 2usize);
    let a = vec![-tiny; m * k];
    let b = vec![tiny; k * n];
    for flags in [0, MM_TRAIN_TA, MM_TRAIN_TB, MM_TRAIN_TA | MM_TRAIN_TB] {
        let j = Job {
            a: &a,
            b: &b,
            bias: None,
            m,
            k,
            n,
            flags,
        };
        assert!(
            run(&DIET, &j).iter().all(|x| x.to_bits() == 0x8000_0000),
            "f{flags}"
        );
    }
}

#[test]
fn t233c_the_513_forms_cut_down_match_mm_tv_128x128() {
    let (bsz, tl) = (2usize, 512usize);
    let ar = train::M513;
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let plan = Plan::new_tiles3(
        ar,
        bsz,
        tl,
        &inputs,
        nvidia_ga104::mmu::VA_BASE + 0x100_0000,
    )
    .unwrap()
    .resident()
    .unwrap();
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    drop(params);
    let forms = mv::forms(&plan.step(Recipe::AdamW513, &opt, 1).unwrap());
    assert_eq!(forms.len(), 15);
    assert_eq!(forms.iter().map(|f| f.count).sum::<u32>(), 147);
    let mut r = rng(0x233c);
    for f in &forms {
        assert_eq!(f.m % DIET.bm, 0);
        assert_eq!(f.n % DIET.bn, 0);
        let (m, n, k) = (
            (f.m / 8) as usize + 3,
            (f.n / 64) as usize + 5,
            (f.k / 64) as usize + 7,
        );
        let x: Vec<f32> = (0..m * k).map(|_| r()).collect();
        let y: Vec<f32> = (0..k * n).map(|_| r()).collect();
        let bias: Vec<f32> = (0..n).map(|_| r()).collect();
        let j = Job {
            a: &x,
            b: &y,
            bias: (f.flags & MM_TRAIN_BIAS != 0).then_some(&bias[..]),
            m,
            k,
            n,
            flags: f.flags,
        };
        assert_eq!(
            bits(&run(&DIET, &j)),
            bits(&run(&VARIANTS[BIG], &j)),
            "{f:?}"
        );
    }
}

#[test]
fn t233d_occupancy_two_blocks_of_sixteen_warps_from_32_kib() {
    let diet = occ::diet_128_kernel();
    let big = occ::variant_kernel(BIG);
    assert_eq!(
        (diet.threads, diet.registers, diet.shared_bytes),
        (512, DIET.registers, 8192)
    );
    for kib in [8u32, 16, 32, 64] {
        for res in [Reserve::None, Reserve::OneKib] {
            let d = occ::blocks_per_sm(&SM86, &diet, kib << 10, res).unwrap();
            let b = occ::blocks_per_sm(&SM86, &big, kib << 10, res).unwrap();
            println!(
                "kt t233 ocupare {kib} KiB {res:?}: dieta {} blocuri = {} grupuri ({:?}); 128x128 {} blocuri = {} grupuri ({:?})",
                d.blocks,
                d.blocks * 16,
                d.bound,
                b.blocks,
                b.blocks * 8,
                b.bound
            );
            if kib >= 32 {
                assert_eq!(d.blocks, 2, "P2: two blocks at {kib} KiB");
                assert_eq!(b.blocks, 2, "128x128 stays at 2 (registers)");
                assert_eq!(d.blocks * 16, 2 * b.blocks * 8, "twice the warps");
            }
        }
    }
    // At today's step of mm_tv_128x128 (one block's shared memory) both have one block.
    let d8 = occ::blocks_per_sm(&SM86, &diet, 8 << 10, Reserve::None).unwrap();
    assert_eq!(d8.blocks, 1);
}

#[test]
fn t233e_wrong_block_or_shared_memory_refused_nothing_written() {
    let a = vec![1.0f32; 128 * 128];
    let j = Job {
        a: &a,
        b: &a,
        bias: None,
        m: 128,
        k: 128,
        n: 128,
        flags: 0,
    };
    let mut cp = cp_for(&DIET);
    load(&mut cp, &j);
    // mm_tv_128x128's block (256 threads) on this program.
    assert!(
        launch(
            &mut cp,
            DIET.registers,
            VARIANTS[BIG].block(),
            [1, 1, 1],
            DIET.shared()
        )
        .is_err()
    );
    let small = SharedMemory {
        bytes: DIET.shared().bytes - 256,
        ..DIET.shared()
    };
    assert!(launch(&mut cp, DIET.registers, DIET.block(), [1, 1, 1], small).is_err());
    let no_bar = SharedMemory {
        barriers: 0,
        ..DIET.shared()
    };
    assert!(launch(&mut cp, DIET.registers, DIET.block(), [1, 1, 1], no_bar).is_err());
    let got = cp.mem.read_f32s(C, 128 * 128).unwrap();
    assert!(got.iter().all(|&x| x == SENTINEL));
}
