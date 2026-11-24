//! C-157 p.2x — T214, SIMULARE: `mm_train` on 64×64 tiles (`tools/nak/mm_train_tile.comp`, compiled by Mesa NAK to
//! `tools/nak/out/mm_train_tile.sm86.sass`), launched through our pushbuffer + QMD in the command processor model.
//! The "program" runs on the CPU (`mm_train_tile_on_cpu`, following the kernel's tiles, 4×4 per thread). Checked
//! bit for bit (1) against `reference::matmul_t_fma_ftz` and (2) against `mm_train` (`mm_train_on_cpu`, the program
//! already used in the step) on all 8 flag combinations, partial tiles and the shapes of the small golden step.
//! Nothing here runs on the RTX 3070 Ti.

use gpu_sim::cmdproc::{
    CommandProcessor, Event, MM_TRAIN_BIAS, MM_TRAIN_NEEDS, MM_TRAIN_TA, MM_TRAIN_TB,
    MM_TRAIN_TILE_NEEDS, mm_train_on_cpu, mm_train_tile_on_cpu,
};
use nvidia_ga104::mm_train::{self as mt, MM_TRAIN};
use nvidia_ga104::mm_train_tile::{self as mtt, MM_TRAIN_TILE};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
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

/// Which program: the tiled one (under test) or `mm_train` (the reference already in the step).
#[derive(Clone, Copy, PartialEq)]
enum Prog {
    Tile,
    Plain,
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

fn cp_for(p: Prog) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    match p {
        Prog::Tile => {
            cp.register_program_with_needs(
                MM_TRAIN_TILE,
                mm_train_tile_on_cpu,
                MM_TRAIN_TILE_NEEDS,
            );
            cp.mem.write(PROG, MM_TRAIN_TILE);
        }
        Prog::Plain => {
            cp.register_program_with_needs(MM_TRAIN, mm_train_on_cpu, MM_TRAIN_NEEDS);
            cp.mem.write(PROG, MM_TRAIN);
        }
    }
    cp
}

fn load(cp: &mut CommandProcessor, j: &Job<'_>) {
    let mut cb = vec![0u8; 0x100];
    match args(j).cbuf() {
        Some(bytes) => cb[..mtt::CBUF_BYTES].copy_from_slice(&bytes),
        // Refused by Args: write the flags raw so the model's own refusal is tested too.
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
    p: Prog,
    grid: [u32; 3],
    shared: SharedMemory,
) -> Result<(), gpu_core::error::GpuError> {
    let (registers, block) = match p {
        Prog::Tile => (mtt::REGISTERS, mtt::BLOCK),
        Prog::Plain => (mt::REGISTERS, mt::BLOCK),
    };
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

/// Runs one job through the model; returns C and checks that nothing was written after it.
fn run(p: Prog, j: &Job<'_>) -> Vec<f32> {
    let mut cp = cp_for(p);
    load(&mut cp, j);
    let (grid, shared) = match p {
        Prog::Tile => (mtt::grid(&args(j)), mtt::SHARED),
        Prog::Plain => (args(j).grid(), mt::SHARED),
    };
    launch(&mut cp, p, grid, shared).unwrap();
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
fn t214_binary_is_the_recorded_build() {
    let hex: String = nvidia_ga104::sha256::digest(MM_TRAIN_TILE)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, mtt::MM_TRAIN_TILE_SHA256);
    assert_eq!(MM_TRAIN_TILE.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out/mm_train_tile.sm86.txt");
    assert!(txt.contains("bar"), "no barrier in the dump");
    assert!(
        txt.contains("ffma.ftz") && txt.contains("fadd.ftz"),
        "FMA chain + bias add expected"
    );
    // The 16 sums stay in registers: no per-thread local memory (our QMD does not set it up).
    assert!(txt.contains("SLM size: 0\n"), "local memory used");
    assert!(txt.contains(&format!("Num GPRs: {}\n", mtt::REGISTERS)));
    assert_eq!(mtt::SHARED.bytes, MM_TRAIN_TILE_NEEDS.shared_bytes);
    assert_eq!(mtt::CBUF_BYTES, mt::CBUF_BYTES);
    let g = mtt::grid(&mt::Args {
        a: A,
        b: B,
        c: C,
        bias: 0,
        m: 6144,
        n: 2304,
        k: 768,
        flags: mt::TB,
    });
    assert_eq!(g, [36, 96, 1], "qkv of #513: 36 x 96 blocks of 64x64");
}

#[test]
fn t214_all_flags_match_the_oracle_and_mm_train_bit_for_bit_with_partial_tiles() {
    let mut r = rng(0x9e37_79b9_7f4a_7c15);
    for (m, k, n) in [
        (64usize, 16usize, 64usize),
        (65, 17, 129),
        (37, 37, 21),
        (5, 1, 3),
        (1, 40, 1),
        (130, 33, 70),
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
            let tile = bits(&run(Prog::Tile, &j));
            assert_eq!(tile, bits(&want(&j)), "oracle {m}x{k}x{n} f{flags}");
            assert_eq!(
                tile,
                bits(&run(Prog::Plain, &j)),
                "mm_train {m}x{k}x{n} f{flags}"
            );
        }
    }
}

#[test]
fn t214_the_three_products_of_a_linear_layer_on_golden_step_shapes() {
    // Shapes of the small golden step's layers (rows = B*T = 64, widths 64 / 256) and a cut of #513's (768 wide).
    let mut r = rng(0x0bad_cafe);
    for (rows, inp, out) in [(64usize, 64usize, 192usize), (64, 256, 64), (96, 768, 80)] {
        let x: Vec<f32> = (0..rows * inp).map(|_| r() * 0.5).collect();
        let w: Vec<f32> = (0..out * inp).map(|_| r() * 0.05).collect();
        let bias: Vec<f32> = (0..out).map(|_| r() * 0.1).collect();
        let dy: Vec<f32> = (0..rows * out).map(|_| r() * 0.01).collect();
        let jobs = [
            // forward Y = X·Wᵀ + b
            Job {
                a: &x,
                b: &w,
                bias: Some(&bias),
                m: rows,
                k: inp,
                n: out,
                flags: MM_TRAIN_TB | MM_TRAIN_BIAS,
            },
            // dX = dY·W
            Job {
                a: &dy,
                b: &w,
                bias: None,
                m: rows,
                k: out,
                n: inp,
                flags: 0,
            },
            // dW = dYᵀ·X
            Job {
                a: &dy,
                b: &x,
                bias: None,
                m: out,
                k: rows,
                n: inp,
                flags: MM_TRAIN_TA,
            },
        ];
        for j in &jobs {
            let tile = bits(&run(Prog::Tile, j));
            assert_eq!(
                tile,
                bits(&run(Prog::Plain, j)),
                "{rows}x{inp}x{out} f{}",
                j.flags
            );
            assert_eq!(tile, bits(&want(j)));
        }
    }
}

#[test]
fn t214_negative_zero_survives_and_no_zero_padding_is_accumulated() {
    // Every product is -2^-140 (flushed to -0.0); K = 19 = 16 + 3: a zero-padded last tile would give +0.0.
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
        let got = run(Prog::Tile, &j);
        assert!(got.iter().all(|x| x.to_bits() == 0x8000_0000), "f{flags}");
    }
}

#[test]
fn t214_refuses_unknown_flags_wrong_blocks_and_too_little_shared_memory() {
    let a = vec![1.0f32; 64 * 64];
    let j = Job {
        a: &a,
        b: &a,
        bias: None,
        m: 64,
        k: 64,
        n: 64,
        flags: 8,
    };
    let mut cp = cp_for(Prog::Tile);
    load(&mut cp, &j);
    assert!(
        launch(&mut cp, Prog::Tile, [1, 1, 1], mtt::SHARED).is_err(),
        "flag 8"
    );
    let j = Job { flags: 0, ..j };
    load(&mut cp, &j);
    // mm_train's 2 KiB are not enough for this program.
    assert!(
        launch(&mut cp, Prog::Tile, [1, 1, 1], mt::SHARED).is_err(),
        "2 KiB"
    );
    let no_bar = SharedMemory {
        barriers: 0,
        ..mtt::SHARED
    };
    assert!(
        launch(&mut cp, Prog::Tile, [1, 1, 1], no_bar).is_err(),
        "no barrier"
    );
    // mm_train's 16×16 blocks are refused by the tiled model.
    let mut cp2 = CommandProcessor::new();
    cp2.register_program_with_needs(MM_TRAIN_TILE, mm_train_tile_on_cpu, MM_TRAIN_TILE_NEEDS);
    cp2.mem.write(PROG, MM_TRAIN_TILE);
    load(&mut cp2, &j);
    assert!(
        launch(&mut cp2, Prog::Plain, [4, 4, 1], mtt::SHARED).is_err(),
        "16x16 blocks"
    );
    for cpx in [&cp, &cp2] {
        let got = cpx.mem.read_f32s(C, 64 * 64).unwrap();
        assert!(
            got.iter().all(|&x| x == SENTINEL),
            "refused launches wrote nothing"
        );
    }
}

#[test]
fn t214_a_grid_too_small_leaves_the_rest_of_c_untouched() {
    // Planted mistake: mm_train's grid (16×16 pieces) with the tiled program covers 64×64 pieces per block —
    // here the opposite: a grid of 1×1 for a 65×65 C must leave everything past row / column 63 unwritten.
    let mut r = rng(3);
    let (m, k, n) = (65usize, 8usize, 65usize);
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
    let mut cp = cp_for(Prog::Tile);
    load(&mut cp, &j);
    launch(&mut cp, Prog::Tile, [1, 1, 1], mtt::SHARED).unwrap();
    let got = cp.mem.read_f32s(C, m * n).unwrap();
    let full = want(&j);
    for row in 0..m {
        for col in 0..n {
            let v = got[row * n + col];
            if row < 64 && col < 64 {
                assert_eq!(v.to_bits(), full[row * n + col].to_bits());
            } else {
                assert_eq!(v, SENTINEL, "({row}, {col})");
            }
        }
    }
}
