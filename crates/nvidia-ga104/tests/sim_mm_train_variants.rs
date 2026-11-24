//! C-165 — T217, SIMULARE: the race variants of `mm_train_tile` (`tools/nak/mm_tv_*.comp` over
//! `tools/nak/mm_train_tv.glsl`, compiled by Mesa NAK to `tools/nak/out/mm_tv_*.sm86.sass`), launched through our
//! pushbuffer + QMD in the command processor model. The "programs" run on the CPU (`mm_train_tv_on_cpu`, following
//! each variant's tiles). Checked bit for bit (1) against `reference::matmul_t_fma_ftz`, (2) against `mm_train` and
//! (3) against `mm_train_tile` (the race's witness) on all 8 flag combinations and partial tiles; plus the shapes of
//! the #513 step the race runs on. Nothing here runs on the RTX 3070 Ti.

#![allow(clippy::needless_range_loop)] // the variant index also picks the CPU model (`model(i)`)

use gpu_compute::train::{self, AdamW, Recipe};
use gpu_sim::cmdproc::{
    self as cpm, CommandProcessor, Event, MM_TRAIN_BIAS, MM_TRAIN_NEEDS, MM_TRAIN_TA, MM_TRAIN_TB,
    MM_TRAIN_TILE_NEEDS, MmTileGeometry, Needs, ProgramFn, mm_train_on_cpu, mm_train_tile_on_cpu,
};
use nvidia_ga104::mm_train::{self as mt, MM_TRAIN};
use nvidia_ga104::mm_train_variants::{self as mv, VARIANTS, Variant};
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

/// The CPU model and the simulator's geometry for each entry of [`VARIANTS`] (same order).
fn model(i: usize) -> (ProgramFn, Needs, Option<MmTileGeometry>) {
    match i {
        0 => (mm_train_tile_on_cpu, MM_TRAIN_TILE_NEEDS, None),
        1 => (
            cpm::mm_tv_128x64_on_cpu,
            cpm::MM_TV_128X64.needs(),
            Some(cpm::MM_TV_128X64),
        ),
        2 => (
            cpm::mm_tv_64x128_on_cpu,
            cpm::MM_TV_64X128.needs(),
            Some(cpm::MM_TV_64X128),
        ),
        3 => (
            cpm::mm_tv_128x128_on_cpu,
            cpm::MM_TV_128X128.needs(),
            Some(cpm::MM_TV_128X128),
        ),
        4 => (
            cpm::mm_tv_64x64k8_on_cpu,
            cpm::MM_TV_64X64K8.needs(),
            Some(cpm::MM_TV_64X64K8),
        ),
        _ => unreachable!(),
    }
}

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

#[derive(Clone, Copy)]
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

/// `None` = `mm_train` (16×16 blocks), `Some(i)` = `VARIANTS[i]`.
fn cp_for(p: Option<usize>) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    match p {
        Some(i) => {
            let (f, needs, _) = model(i);
            cp.register_program_with_needs(VARIANTS[i].code, f, needs);
            cp.mem.write(PROG, VARIANTS[i].code);
        }
        None => {
            cp.register_program_with_needs(MM_TRAIN, mm_train_on_cpu, MM_TRAIN_NEEDS);
            cp.mem.write(PROG, MM_TRAIN);
        }
    }
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

fn launch_variant(
    cp: &mut CommandProcessor,
    v: &Variant,
    grid: [u32; 3],
) -> Result<(), gpu_core::error::GpuError> {
    launch(cp, v.registers, v.block(), grid, v.shared())
}

/// Runs one job through the model; returns C and checks that nothing was written after it.
fn run(p: Option<usize>, j: &Job<'_>) -> Vec<f32> {
    let mut cp = cp_for(p);
    load(&mut cp, j);
    match p {
        Some(i) => launch_variant(&mut cp, &VARIANTS[i], VARIANTS[i].grid(&args(j))).unwrap(),
        None => launch(
            &mut cp,
            mt::REGISTERS,
            mt::BLOCK,
            args(j).grid(),
            mt::SHARED,
        )
        .unwrap(),
    }
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

fn dump(name: &str) -> &'static str {
    match name {
        "mm_train_tile" => include_str!("../../../tools/nak/out/mm_train_tile.sm86.txt"),
        "mm_tv_128x64" => include_str!("../../../tools/nak/out/mm_tv_128x64.sm86.txt"),
        "mm_tv_64x128" => include_str!("../../../tools/nak/out/mm_tv_64x128.sm86.txt"),
        "mm_tv_128x128" => include_str!("../../../tools/nak/out/mm_tv_128x128.sm86.txt"),
        "mm_tv_64x64k8" => include_str!("../../../tools/nak/out/mm_tv_64x64k8.sm86.txt"),
        _ => unreachable!(),
    }
}

#[test]
fn t217_binaries_are_the_recorded_builds_and_the_table_matches_the_models() {
    let mut names = std::collections::BTreeSet::new();
    for (i, v) in VARIANTS.iter().enumerate() {
        assert!(names.insert(v.name), "duplicate {}", v.name);
        let hex: String = nvidia_ga104::sha256::digest(v.code)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hex, v.sha256, "{}", v.name);
        assert_eq!(v.code.len() % 16, 0, "whole 16-byte instructions");
        let txt = dump(v.name);
        assert!(txt.contains("bar"), "{}: no barrier", v.name);
        assert!(
            txt.contains("ffma.ftz") && txt.contains("fadd.ftz"),
            "{}",
            v.name
        );
        assert!(
            txt.contains("SLM size: 0\n"),
            "{}: local memory used",
            v.name
        );
        assert!(
            txt.contains(&format!("Num GPRs: {}\n", v.registers)),
            "{}",
            v.name
        );
        // The chain: bk·tm·tn FMAs per slice (unrolled), tm·tn bias adds.
        let ffma = txt.matches("ffma.ftz").count() as u32;
        let fadd = txt.matches("fadd.ftz").count() as u32;
        assert!(ffma >= v.tm * v.tn, "{}: {ffma} ffma", v.name);
        assert_eq!(fadd, v.tm * v.tn, "{}: {fadd} fadd", v.name);
        assert_eq!(v.block(), [256, 1, 1], "{}", v.name);
        assert!(v.shared().bytes <= v.shared().max_bytes);
        let (_, needs, geo) = model(i);
        assert_eq!(v.shared().bytes, needs.shared_bytes, "{}", v.name);
        if let Some(g) = geo {
            assert_eq!(
                (g.bm, g.bn, g.bk, g.tm, g.tn),
                (v.bm, v.bn, v.bk, v.tm, v.tn),
                "{}",
                v.name
            );
            assert_eq!(g.threads(), v.block()[0]);
        }
        println!(
            "kt t217 {}: {} octeti, {} registre, bloc {}, partajata {} B, {ffma} ffma, {fadd} fadd, sha {}",
            v.name,
            v.code.len(),
            v.registers,
            v.block()[0],
            v.shared().bytes,
            &v.sha256[..16]
        );
    }
    assert_eq!(VARIANTS[0].name, "mm_train_tile", "the witness stays first");
}

#[test]
fn t217_every_variant_matches_the_oracle_mm_train_and_mm_train_tile_bit_for_bit() {
    let mut r = rng(0x9e37_79b9_7f4a_7c15);
    for (m, k, n) in [
        (128usize, 16usize, 128usize),
        (129, 17, 65),
        (65, 9, 130),
        (37, 37, 21),
        (5, 1, 3),
        (1, 40, 1),
        (200, 33, 140),
    ] {
        let mut a: Vec<f32> = (0..m * k).map(|_| r()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
        let bias: Vec<f32> = (0..n).map(|_| r()).collect();
        a[0] = f32::from_bits(0x0000_1234); // subnormal input: FTZ flushes it
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
            let plain = bits(&run(None, &j));
            assert_eq!(plain, oracle, "mm_train {m}x{k}x{n} f{flags}");
            for i in 0..VARIANTS.len() {
                assert_eq!(
                    bits(&run(Some(i), &j)),
                    oracle,
                    "{} {m}x{k}x{n} f{flags}",
                    VARIANTS[i].name
                );
            }
        }
    }
}

#[test]
fn t217_negative_zero_survives_and_no_zero_padding_is_accumulated() {
    // Every product is -2^-140 (flushed to -0.0); K = 19 = 16 + 3 = 8·2 + 3: a zero-padded last slice gives +0.0.
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
        for i in 0..VARIANTS.len() {
            let got = run(Some(i), &j);
            assert!(
                got.iter().all(|x| x.to_bits() == 0x8000_0000),
                "{} f{flags}",
                VARIANTS[i].name
            );
        }
    }
}

#[test]
fn t217_refuses_unknown_flags_too_little_shared_memory_and_wrong_blocks() {
    let a = vec![1.0f32; 128 * 128];
    let j = Job {
        a: &a,
        b: &a,
        bias: None,
        m: 128,
        k: 128,
        n: 128,
        flags: 8,
    };
    for i in 1..VARIANTS.len() {
        let v = &VARIANTS[i];
        let mut cp = cp_for(Some(i));
        load(&mut cp, &j);
        assert!(
            launch_variant(&mut cp, v, [1, 1, 1]).is_err(),
            "{} flag 8",
            v.name
        );
        let j0 = Job { flags: 0, ..j };
        load(&mut cp, &j0);
        let small = SharedMemory {
            bytes: v.shared().bytes - 256,
            ..v.shared()
        };
        assert!(
            launch(&mut cp, v.registers, v.block(), [1, 1, 1], small).is_err(),
            "{} shared too small",
            v.name
        );
        let no_bar = SharedMemory {
            barriers: 0,
            ..v.shared()
        };
        assert!(
            launch(&mut cp, v.registers, v.block(), [1, 1, 1], no_bar).is_err(),
            "{} no barrier",
            v.name
        );
        assert!(
            launch(&mut cp, v.registers, mt::BLOCK, [1, 1, 1], v.shared()).is_err(),
            "{} 16x16 blocks",
            v.name
        );
        let got = cp.mem.read_f32s(C, 128 * 128).unwrap();
        assert!(
            got.iter().all(|&x| x == SENTINEL),
            "{}: refused launches wrote nothing",
            v.name
        );
    }
}

/// Planted mistake: the race must never pair a program with another variant's grid. A grid made for bigger pieces
/// leaves part of C unwritten — the comparison with the oracle catches it; a grid for smaller pieces only adds idle
/// blocks (same bits).
#[test]
fn t217_another_variants_grid_is_caught_when_too_small_and_harmless_when_too_big() {
    let mut r = rng(7);
    let (m, k, n) = (256usize, 24usize, 256usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    let j = Job {
        a: &a,
        b: &b,
        bias: None,
        m,
        k,
        n,
        flags: 0,
    };
    let oracle = bits(&want(&j));
    let (mut caught, mut harmless) = (0, 0);
    for (i, v) in VARIANTS.iter().enumerate() {
        for w in &VARIANTS {
            let g = w.grid(&args(&j));
            let mut cp = cp_for(Some(i));
            load(&mut cp, &j);
            launch_variant(&mut cp, v, g).unwrap();
            let got = bits(&cp.mem.read_f32s(C, m * n).unwrap());
            let covers = g[0] * v.bn >= n as u32 && g[1] * v.bm >= m as u32;
            if covers {
                assert_eq!(got, oracle, "{} with {}'s grid", v.name, w.name);
                harmless += 1;
            } else {
                assert_ne!(
                    got, oracle,
                    "{} with {}'s grid must be caught",
                    v.name, w.name
                );
                assert!(got.contains(&SENTINEL.to_bits()));
                caught += 1;
            }
        }
    }
    assert!(caught > 0 && harmless > 0, "{caught} {harmless}");
}

/// The shapes the race runs on: the `mm_train` products of the #513 step (os-407: `Plan::new_tiles3`, resident,
/// 2 × 512). Prediction 3: every side divides by 128 (no partial tile in the race). Bits on cut-down copies of each
/// shape (the same flags, sides ÷ 8) for every variant.
#[test]
fn t217_the_513_forms_divide_by_every_tile_and_cut_down_copies_match() {
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
    let launches = plan.step(Recipe::AdamW513, &opt, 1).unwrap();
    let forms = mv::forms(&launches);
    let total: u32 = forms.iter().map(|f| f.count).sum();
    assert_eq!(total, 147, "the 147 mm_train launches of #513");
    // The same list on the tiled plan (the programs differ, the shapes do not).
    let tiled = Plan::new_tiles3(
        ar,
        bsz,
        tl,
        &inputs,
        nvidia_ga104::mmu::VA_BASE + 0x100_0000,
    )
    .unwrap()
    .with_mm_tile()
    .resident()
    .unwrap();
    assert_eq!(
        mv::forms(&tiled.step(Recipe::AdamW513, &opt, 1).unwrap()),
        forms
    );
    let mut r = rng(0x513);
    let mut flops_all = 0u64;
    for f in &forms {
        let a = mt::Args {
            a: 0,
            b: 0,
            c: 0,
            bias: 0,
            m: f.m,
            n: f.n,
            k: f.k,
            flags: f.flags,
        };
        flops_all += Variant::flops(&a) * f.count as u64;
        let blocks: Vec<String> = VARIANTS
            .iter()
            .map(|v| {
                assert_eq!(f.m % v.bm, 0, "{f:?} {}", v.name);
                assert_eq!(f.n % v.bn, 0, "{f:?} {}", v.name);
                let g = v.grid(&a);
                format!("{}={}", &v.name[3..], g[0] * g[1])
            })
            .collect();
        println!(
            "kt t217 forma #513 {}x{}x{} f{} x{}: {:.2} GFLOP; blocuri {}",
            f.m,
            f.n,
            f.k,
            f.flags,
            f.count,
            Variant::flops(&a) as f64 / 1e9,
            blocks.join(" ")
        );
        // Cut-down copy, bits checked for every variant.
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
        let oracle = bits(&want(&j));
        for i in 0..VARIANTS.len() {
            assert_eq!(
                bits(&run(Some(i), &j)),
                oracle,
                "{} {f:?}",
                VARIANTS[i].name
            );
        }
    }
    println!(
        "kt t217 #513: {} forme, 147 lansari, {:.1} GFLOP pe pas in mm_train",
        forms.len(),
        flops_all as f64 / 1e9
    );
}
