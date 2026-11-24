//! C-165 follow-up — T220, SIMULARE: `mm_train_gelu` on `mm_train_tile`'s 64×64 tiles
//! (`tools/nak/mm_train_tile_gelu.comp`, compiled by Mesa NAK to `tools/nak/out/mm_train_tile_gelu.sm86.sass`),
//! launched through our pushbuffer + QMD in the command processor model. The "program" runs on the CPU
//! (`mm_train_tile_gelu_on_cpu`). Checked bit for bit (1) against the oracle `gelu_tr_gpu_back` over
//! `reference::matmul_t_fma_ftz` and (2) against `mm_train_gelu` (`mm_train_gelu_on_cpu`, the program the step uses
//! today) on the 4 allowed flag combinations, partial tiles and the "proj-dX" shapes of the small golden step and of
//! #513. Nothing here runs on the RTX 3070 Ti.

use gpu_sim::cmdproc::{
    CommandProcessor, Event, MM_TRAIN_BIAS, MM_TRAIN_GELU_NEEDS, MM_TRAIN_TA, MM_TRAIN_TB,
    MM_TRAIN_TILE_GELU_NEEDS, mm_train_gelu_on_cpu, mm_train_tile_gelu_on_cpu,
    mm_train_tile_on_cpu,
};
use nvidia_ga104::mm_train_gelu::{self as mg, MM_TRAIN_GELU};
use nvidia_ga104::mm_train_tile as mtt;
use nvidia_ga104::mm_train_tile_gelu::{self as mttg, MM_TRAIN_TILE_GELU};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_ga104::train_step::{self as ts, Prog as TsProg};
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const A: u64 = 0x400_0000;
const B: u64 = 0x800_0000;
const C: u64 = 0x1000_0000;
const U: u64 = 0x1800_0000;
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

/// Which program runs the launch.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Which {
    /// `mm_train_tile_gelu` (under test).
    TileGelu,
    /// `mm_train_gelu` (the reference already in the step).
    Gelu,
    /// Planted mistake: `mm_train_tile_gelu`'s bytes run by `mm_train_tile`'s model (the epilogue forgotten).
    TileNoGelu,
}

struct Job {
    a: Vec<f32>,
    b: Vec<f32>,
    u: Vec<f32>,
    m: usize,
    k: usize,
    n: usize,
    flags: u32,
}

fn job(m: usize, k: usize, n: usize, flags: u32, seed: u64) -> Job {
    let mut r = rng(seed);
    Job {
        a: (0..m * k).map(|_| r()).collect(),
        b: (0..k * n).map(|_| r()).collect(),
        u: (0..m * n).map(|_| r() * 2.0).collect(),
        m,
        k,
        n,
        flags,
    }
}

fn args(j: &Job) -> mg::Args {
    mg::Args {
        a: A,
        b: B,
        c: C,
        u: U,
        m: j.m as u32,
        n: j.n as u32,
        k: j.k as u32,
        flags: j.flags,
    }
}

fn run(j: &Job, w: Which, grid: Option<[u32; 3]>) -> Result<Vec<f32>, String> {
    let mut cp = CommandProcessor::new();
    let (code, block, regs, shared): (&[u8], [u32; 3], u32, SharedMemory) = match w {
        Which::TileGelu | Which::TileNoGelu => (
            MM_TRAIN_TILE_GELU,
            mttg::BLOCK,
            mttg::REGISTERS,
            mttg::SHARED,
        ),
        Which::Gelu => (MM_TRAIN_GELU, mg::BLOCK, mg::REGISTERS, mg::SHARED),
    };
    let _ = match w {
        Which::TileGelu => cp.register_program_with_needs(
            code,
            mm_train_tile_gelu_on_cpu,
            MM_TRAIN_TILE_GELU_NEEDS,
        ),
        Which::TileNoGelu => {
            cp.register_program_with_needs(code, mm_train_tile_on_cpu, MM_TRAIN_TILE_GELU_NEEDS)
        }
        Which::Gelu => {
            cp.register_program_with_needs(code, mm_train_gelu_on_cpu, MM_TRAIN_GELU_NEEDS)
        }
    };
    cp.mem.write(PROG, code);
    let mut cb = vec![0u8; 0x100];
    match args(j).cbuf() {
        Some(bytes) => cb[..mttg::CBUF_BYTES].copy_from_slice(&bytes),
        None => {
            // Refused by Args: write the cbuf raw so the model's own refusal is tested too.
            cb[0x28..0x30].copy_from_slice(&A.to_le_bytes());
            cb[0x30..0x38].copy_from_slice(&B.to_le_bytes());
            cb[0x38..0x40].copy_from_slice(&C.to_le_bytes());
            cb[0x40..0x48].copy_from_slice(&U.to_le_bytes());
            cb[0x48..0x4c].copy_from_slice(&(j.m as u32).to_le_bytes());
            cb[0x4c..0x50].copy_from_slice(&(j.n as u32).to_le_bytes());
            cb[0x50..0x54].copy_from_slice(&(j.k as u32).to_le_bytes());
            cb[0x54..0x58].copy_from_slice(&j.flags.to_le_bytes());
        }
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(A, &j.a);
    cp.mem.write_f32s(B, &j.b);
    cp.mem.write_f32s(U, &j.u);
    cp.mem.write_f32s(C, &vec![SENTINEL; j.m * j.n + 16]);
    cp.mem.write_u32(SEM, 0);
    let a = args(j);
    let grid = grid.unwrap_or(match w {
        Which::Gelu => a.grid(),
        _ => mttg::grid(&a),
    });
    let q = ComputeQmd {
        program_va: PROG,
        registers: regs,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid,
        block,
        release: Some((SEM, 1)),
    };
    cp.mem.write(QMD, &q.to_bytes_with_shared(&shared).unwrap());
    let mut words = [0u32; 16];
    let mut pb = PushBuf::new(&mut words);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    pb.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    pb.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = pb.len();
    let bytes: Vec<u8> = words[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
        .map_err(|e| format!("{e:?}"))?;
    if !cp
        .events
        .iter()
        .any(|e| matches!(e, Event::Launch { program: 0, .. }))
    {
        return Err("nicio lansare".into());
    }
    if cp.mem.read_u32(SEM).unwrap() != 1 {
        return Err("semaforul nu a fost scris".into());
    }
    let out = cp.mem.read_f32s(C, j.m * j.n + 16).unwrap();
    assert!(
        out[j.m * j.n..].iter().all(|&v| v == SENTINEL),
        "{w:?}: scris dupa C"
    );
    Ok(out[..j.m * j.n].to_vec())
}

/// The oracle: `gelu_tr_gpu_back`(u, the `ffma.ftz` chain).
fn oracle(j: &Job) -> Vec<f32> {
    let mut acc = vec![0.0f32; j.m * j.n];
    gpu_compute::reference::matmul_t_fma_ftz(
        &j.a,
        &j.b,
        None,
        &mut acc,
        j.m,
        j.k,
        j.n,
        j.flags & MM_TRAIN_TA != 0,
        j.flags & MM_TRAIN_TB != 0,
    )
    .unwrap();
    acc.iter()
        .zip(&j.u)
        .map(|(&g, &u)| gpu_compute::layer::gelu_tr_gpu_back(u, g))
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Prediction 1: on every allowed flag combination and on shapes with partial tiles in M, N and K (and K < 16, K not a
/// multiple of 16), `mm_train_tile_gelu` = the oracle = `mm_train_gelu`, bit for bit.
#[test]
fn t220_tile_gelu_is_mm_train_gelu_bit_for_bit() {
    let shapes = [
        (64, 48, 128),
        (65, 17, 63),
        (1, 5, 1),
        (130, 33, 70),
        (64, 768, 192),
        (100, 200, 129),
    ];
    let mut cases = 0;
    for (si, &(m, k, n)) in shapes.iter().enumerate() {
        for flags in [0, MM_TRAIN_TA, MM_TRAIN_TB, MM_TRAIN_TA | MM_TRAIN_TB] {
            let j = job(m, k, n, flags, 220 + si as u64 * 8 + flags as u64);
            let o = bits(&oracle(&j));
            let t = bits(&run(&j, Which::TileGelu, None).unwrap());
            let g = bits(&run(&j, Which::Gelu, None).unwrap());
            assert_eq!(t, o, "tile_gelu != oracol: {m}x{k}x{n} flags {flags}");
            assert_eq!(g, o, "mm_train_gelu != oracol: {m}x{k}x{n} flags {flags}");
            cases += 1;
        }
    }
    println!(
        "kt t220 mm_train_tile_gelu = oracolul = mm_train_gelu bit cu bit: {cases} cazuri (SIMULARE)"
    );
}

/// Prediction 1 on the real shapes: "proj-dX" of the small golden step (64 rows × 512 × 128) and of #513
/// (1 024 × 3 072 × 768, one block row of it to keep the test short: the other rows are the same code).
#[test]
fn t220_tile_gelu_on_the_proj_dx_shapes() {
    for (m, k, n, name) in [
        (64, 128, 512, "MIC"),
        (64, 768, 3072, "#513 (64 de randuri)"),
    ] {
        let j = job(m, k, n, 0, 2200 + n as u64);
        let o = bits(&oracle(&j));
        let t = bits(&run(&j, Which::TileGelu, None).unwrap());
        assert_eq!(t, o, "{name}");
        println!("kt t220 proj-dX {name} {m}x{k}x{n}: = oracolul (SIMULARE)");
    }
}

/// Prediction 2: the mistakes are caught — the epilogue forgotten (`mm_train_tile`'s model on these bytes), a grid
/// one block row short (cells left at the sentinel), `mm_train_gelu`'s 16×16 grid with blocks of 256 (harmless: extra
/// blocks write nothing — the same bits), the BIAS flag (refused by `Args` and by the model), C overlapping u
/// (refused).
#[test]
fn t220_planted_mistakes() {
    let j = job(130, 40, 70, 0, 2201);
    let o = bits(&oracle(&j));
    let no_gelu = bits(&run(&j, Which::TileNoGelu, None).unwrap());
    assert_ne!(no_gelu, o, "epilogul uitat neprins");
    let g = mttg::grid(&args(&j));
    let short = bits(&run(&j, Which::TileGelu, Some([g[0], g[1] - 1, 1])).unwrap());
    assert_ne!(short, o, "grila scurta neprinsa");
    let big = bits(&run(&j, Which::TileGelu, Some(args(&j).grid())).unwrap());
    assert_eq!(
        big, o,
        "grila lui mm_train_gelu trebuia sa dea aceiasi biti"
    );
    let jb = job(16, 16, 16, MM_TRAIN_BIAS, 2202);
    assert!(args(&jb).cbuf().is_none());
    assert!(run(&jb, Which::TileGelu, None).is_err(), "BIAS acceptat");
    let mut ov = args(&j);
    ov.u = ov.c;
    assert!(ov.cbuf().is_none(), "C = u acceptat");
    println!(
        "kt t220 greseli: epilog uitat prins, grila scurta prinsa, grila 16x16 inofensiva, BIAS refuzat, C = u refuzat (SIMULARE)"
    );
}

/// The program's bytes, checksum and launch numbers are the ones the step uses.
#[test]
fn t220_program_table() {
    ts::check_programs().unwrap();
    let p = TsProg::MmTrainTileGelu;
    assert_eq!(p.code(), MM_TRAIN_TILE_GELU);
    assert_eq!(p.sha256(), mttg::MM_TRAIN_TILE_GELU_SHA256);
    assert_eq!(p.block(), mtt::BLOCK);
    assert_eq!(p.shared(), Some(mtt::SHARED));
    assert_eq!(p.name(), "mm_train_tile_gelu");
    assert_eq!(p.index(), 29);
    assert_eq!(
        ts::PROGRAMS_ALL_TILE_GELU[..29],
        ts::PROGRAMS_ALL_CHOICE[..]
    );
    assert_eq!(p.registers(), mttg::REGISTERS);
    assert_eq!(mttg::REGISTERS, nvidia_ga104::mm_train_tile::REGISTERS);
    assert_eq!(MM_TRAIN_TILE_GELU_NEEDS.shared_bytes, mtt::SHARED.bytes);
}
