//! C-20 E (step 1), SIMULARE: the naive matmul compiled by Mesa NAK
//! (`tools/nak/out/matmul_naive.sm86.sass`, 816 bytes, 51 instructions,
//! 19 registers) launched through our pushbuffer + QMD in the command
//! processor model. The "program" runs on the CPU (`matmul_naive_on_cpu`,
//! oracle `reference::matmul_fma_ftz`); nothing here runs on the RTX 3070 Ti.

use gpu_sim::cmdproc::{
    CommandProcessor, Event, MATMUL_1D_NEEDS, MATMUL_2D_NEEDS, MATMUL_SMEM_NEEDS,
    MATMUL_VEC4_NEEDS, matmul_1d_on_cpu, matmul_2d_on_cpu, matmul_coalesced_on_cpu,
    matmul_naive_on_cpu, matmul_smem_on_cpu, matmul_vec4_on_cpu,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, MAX_SHARED_BYTES, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const MATMUL: &[u8] = include_bytes!("../../../tools/nak/out/matmul_naive.sm86.sass");
const COALESCED: &[u8] = include_bytes!("../../../tools/nak/out/matmul_coalesced.sm86.sass");
const SMEM: &[u8] = include_bytes!("../../../tools/nak/out/matmul_smem.sm86.sass");
const ONE_D: &[u8] = include_bytes!("../../../tools/nak/out/matmul_1d.sm86.sass");
const TWO_D: &[u8] = include_bytes!("../../../tools/nak/out/matmul_2d.sm86.sass");
const VEC4: &[u8] = include_bytes!("../../../tools/nak/out/matmul_vec4.sm86.sass");
const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const A: u64 = 0x40_0000;
const B: u64 = 0x48_0000;
const C: u64 = 0x50_0000;
const QMD: u64 = 0x58_0000;
const PUSH: u64 = 0x60_0000;
const SENTINEL: f32 = -12345.0;

fn data(m: usize, k: usize, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut s = 0x2545_f491_4f6c_dd1du64;
    let mut r = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    };
    let mut a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    // A few subnormal inputs: FTZ must flush them.
    a[3] = f32::from_bits(0x0000_1234);
    a[m * k - 1] = -f32::from_bits(0x0040_0000);
    (a, b)
}

fn setup(m: u32, k: u32, n: u32) -> (CommandProcessor, Vec<f32>, Vec<f32>) {
    let mut cp = CommandProcessor::new();
    cp.register_program(MATMUL, matmul_naive_on_cpu);
    let (a, b) = data(m as usize, k as usize, n as usize);
    cp.mem.write(PROG, MATMUL);
    let mut cb = vec![0u8; 0x100];
    for (off, va) in [(0x28, A), (0x30, B), (0x38, C)] {
        cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
    }
    for (off, v) in [(0x40, m), (0x44, n), (0x48, k)] {
        cb[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(A, &a);
    cp.mem.write_f32s(B, &b);
    // C plus 16 sentinel words after it.
    cp.mem.write_f32s(C, &vec![SENTINEL; (m * n) as usize + 16]);
    cp.mem.write_u32(SEM, 0);
    (cp, a, b)
}

fn launch(cp: &mut CommandProcessor, grid: [u32; 3]) -> Result<(), gpu_core::error::GpuError> {
    launch_shared(cp, grid, None)
}

fn launch_shared(
    cp: &mut CommandProcessor,
    grid: [u32; 3],
    shared: Option<SharedMemory>,
) -> Result<(), gpu_core::error::GpuError> {
    launch_with(cp, grid, [16, 16, 1], 19, shared)
}

fn launch_with(
    cp: &mut CommandProcessor,
    grid: [u32; 3],
    block: [u32; 3],
    registers: u32,
    shared: Option<SharedMemory>,
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
    let qmd = match shared {
        Some(s) => q.to_bytes_with_shared(&s).unwrap(),
        None => q.to_bytes().unwrap(),
    };
    cp.mem.write(QMD, &qmd);
    let mut w = [0u32; 16];
    let mut p = PushBuf::new(&mut w);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = p.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn naive_matmul_runs_in_the_model_and_matches_the_ffma_ftz_oracle() {
    let (m, k, n) = (37usize, 19usize, 21usize);
    let (mut cp, a, b) = setup(m as u32, k as u32, n as u32);
    launch(&mut cp, [3, 2, 1]).unwrap(); // 48 x 32 threads cover 37 x 21
    let got = cp.mem.read_f32s(C, m * n + 16).unwrap();
    let mut want = vec![0.0; m * n];
    gpu_compute::reference::matmul_fma_ftz(&a, &b, &mut want, m, k, n).unwrap();
    assert_eq!(bits(&got[..m * n]), bits(&want));
    assert!(
        got[m * n..].iter().all(|&x| x == SENTINEL),
        "nothing written past C"
    );
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    assert!(matches!(cp.events[1], Event::Launch { program: 0, .. }));
    // The CPU backend's oracle (two roundings, no FTZ) is a different
    // function: comparing a GPU kernel with it would report false errors.
    let mut cpu = vec![0.0; m * n];
    gpu_compute::reference::matmul(&a, &b, &mut cpu, m, k, n).unwrap();
    let differ = (0..m * n)
        .filter(|&i| cpu[i].to_bits() != want[i].to_bits())
        .count();
    assert!(differ > 0);
    for i in 0..m * n {
        assert!((cpu[i] - want[i]).abs() <= 1e-5 * want[i].abs().max(1.0));
    }
}

#[test]
fn a_grid_that_is_too_small_leaves_c_partly_unwritten() {
    let (m, k, n) = (37usize, 19usize, 21usize);
    let (mut cp, ..) = setup(m as u32, k as u32, n as u32);
    launch(&mut cp, [2, 2, 1]).unwrap(); // only 32 rows
    let got = cp.mem.read_f32s(C, m * n).unwrap();
    assert!(got[..32 * n].iter().all(|&x| x != SENTINEL));
    assert!(got[32 * n..].iter().all(|&x| x == SENTINEL));
}

#[test]
fn wrong_block_shape_or_changed_program_is_refused() {
    let (mut cp, ..) = setup(16, 16, 16);
    let mut other = MATMUL.to_vec();
    other[200] ^= 1;
    cp.mem.write(PROG, &other);
    assert!(launch(&mut cp, [1, 1, 1]).is_err());
    let (mut cp, ..) = setup(16, 16, 16);
    let mut cb = vec![0u8; 8];
    cb[0] = 1; // non-zero workgroup base: not modelled
    cp.mem.write(CBUF, &cb);
    assert!(launch(&mut cp, [1, 1, 1]).is_err());
}

#[test]
fn coalesced_step_gives_the_same_bits_with_the_grid_swapped() {
    let (m, k, n) = (37usize, 19usize, 21usize);
    let (mut cp, a, b) = setup(m as u32, k as u32, n as u32);
    cp.register_program(COALESCED, matmul_coalesced_on_cpu);
    cp.mem.write(PROG, COALESCED);
    launch(&mut cp, [2, 3, 1]).unwrap(); // x covers 32 >= 21 columns, y 48 >= 37 rows
    assert!(matches!(cp.events[1], Event::Launch { program: 1, .. }));
    let got = cp.mem.read_f32s(C, m * n).unwrap();
    let mut want = vec![0.0; m * n];
    gpu_compute::reference::matmul_fma_ftz(&a, &b, &mut want, m, k, n).unwrap();
    assert_eq!(bits(&got), bits(&want));
    assert_ne!(COALESCED, MATMUL);
}

/// Step 3 (session 8): shared-memory tiles. NAK: 218 instructions, 3488 B,
/// 19 registers, 2 KiB shared, 1 barrier, `ffma.ftz` (tools/nak/README.md).
fn smem_cp(m: usize, k: usize, n: usize) -> (CommandProcessor, Vec<f32>, Vec<f32>) {
    let (mut cp, a, b) = setup(m as u32, k as u32, n as u32);
    cp.register_program_with_needs(SMEM, matmul_smem_on_cpu, MATMUL_SMEM_NEEDS);
    cp.mem.write(PROG, SMEM);
    (cp, a, b)
}

const SMEM_QMD: SharedMemory = SharedMemory {
    bytes: 2048,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};

#[test]
fn smem_step_matches_the_ffma_ftz_oracle_including_partial_tiles() {
    // K = 37: two full tiles + a partial one of 5; M, N not multiples of 16.
    for (m, k, n) in [
        (37usize, 37usize, 21usize),
        (16, 16, 16),
        (5, 1, 3),
        (33, 48, 17),
    ] {
        let (mut cp, a, b) = smem_cp(m, k, n);
        let grid = [n.div_ceil(16) as u32, m.div_ceil(16) as u32, 1];
        launch_shared(&mut cp, grid, Some(SMEM_QMD)).unwrap();
        assert!(matches!(cp.events[1], Event::Launch { program: 1, .. }));
        let got = cp.mem.read_f32s(C, m * n + 16).unwrap();
        let mut want = vec![0.0; m * n];
        gpu_compute::reference::matmul_fma_ftz(&a, &b, &mut want, m, k, n).unwrap();
        assert_eq!(bits(&got[..m * n]), bits(&want), "{m}x{k}x{n}");
        assert!(got[m * n..].iter().all(|&x| x == SENTINEL), "wrote after C");
        assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    }
}

#[test]
fn smem_step_keeps_negative_zero_like_the_oracle() {
    // Every product is -2^-140: subnormal, flushed by FFMA.FTZ to -0.0, and
    // -0.0 + (-0.0) stays -0.0. If the last, partial tile (K = 19 = 16 + 3)
    // were padded with zeros and accumulated, -0.0 + 0*0 would give +0.0.
    let (m, k, n) = (3usize, 19usize, 2usize);
    let (mut cp, ..) = smem_cp(m, k, n);
    let tiny = f32::from_bits(0x1c80_0000); // 2^-70, a normal f32
    let a = vec![-tiny; m * k];
    let b = vec![tiny; k * n];
    cp.mem.write_f32s(A, &a);
    cp.mem.write_f32s(B, &b);
    launch_shared(&mut cp, [1, 1, 1], Some(SMEM_QMD)).unwrap();
    let got = cp.mem.read_f32s(C, m * n).unwrap();
    let mut want = vec![0.0; m * n];
    gpu_compute::reference::matmul_fma_ftz(&a, &b, &mut want, m, k, n).unwrap();
    assert!(
        want.iter().all(|x| x.to_bits() == 0x8000_0000),
        "the case must give -0.0"
    );
    assert_eq!(bits(&got), bits(&want));
}

#[test]
fn smem_step_refuses_a_qmd_without_shared_memory_or_barrier() {
    let (mut cp, ..) = smem_cp(16, 16, 16);
    assert!(launch(&mut cp, [1, 1, 1]).is_err()); // plain QMD: 0 shared bytes
    let few = SharedMemory {
        bytes: 1024,
        ..SMEM_QMD
    };
    assert!(launch_shared(&mut cp, [1, 1, 1], Some(few)).is_err());
    let no_bar = SharedMemory {
        barriers: 0,
        ..SMEM_QMD
    };
    assert!(launch_shared(&mut cp, [1, 1, 1], Some(no_bar)).is_err());
    let got = cp.mem.read_f32s(C, 16 * 16).unwrap();
    assert!(
        got.iter().all(|&x| x == SENTINEL),
        "refused launches wrote nothing"
    );
    assert_ne!(SMEM, COALESCED);
}

/// Step 4 (session 9): 1D blocktiling, 8 results per thread. NAK: 478
/// instructions, 7648 B, 24 registers, 4 KiB shared, 1 barrier, `ffma.ftz`.
fn one_d_cp(m: usize, k: usize, n: usize) -> (CommandProcessor, Vec<f32>, Vec<f32>) {
    let (mut cp, a, b) = setup(m as u32, k as u32, n as u32);
    cp.register_program_with_needs(ONE_D, matmul_1d_on_cpu, MATMUL_1D_NEEDS);
    cp.mem.write(PROG, ONE_D);
    (cp, a, b)
}

const ONE_D_QMD: SharedMemory = SharedMemory {
    bytes: 4096,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};

#[test]
fn one_d_step_matches_the_ffma_ftz_oracle_including_partial_blocks() {
    // K = 37 = 4 full tiles of 8 + 5; M, N not multiples of 64; a 2x2 grid.
    for (m, k, n) in [
        (37usize, 37usize, 21usize),
        (64, 8, 64),
        (5, 1, 3),
        (130, 19, 70),
    ] {
        let (mut cp, a, b) = one_d_cp(m, k, n);
        let grid = [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1];
        launch_with(&mut cp, grid, [512, 1, 1], 24, Some(ONE_D_QMD)).unwrap();
        assert!(matches!(cp.events[1], Event::Launch { program: 1, .. }));
        let got = cp.mem.read_f32s(C, m * n + 16).unwrap();
        let mut want = vec![0.0; m * n];
        gpu_compute::reference::matmul_fma_ftz(&a, &b, &mut want, m, k, n).unwrap();
        assert_eq!(bits(&got[..m * n]), bits(&want), "{m}x{k}x{n}");
        assert!(got[m * n..].iter().all(|&x| x == SENTINEL), "wrote after C");
        assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    }
}

#[test]
fn one_d_step_keeps_negative_zero_and_refuses_wrong_launches() {
    // As in step 3: K = 11 = 8 + 3, every product flushed to -0.0; padding
    // the last tile with zeros and accumulating them would give +0.0.
    let (m, k, n) = (9usize, 11usize, 2usize);
    let (mut cp, ..) = one_d_cp(m, k, n);
    let tiny = f32::from_bits(0x1c80_0000);
    let (a, b) = (vec![-tiny; m * k], vec![tiny; k * n]);
    cp.mem.write_f32s(A, &a);
    cp.mem.write_f32s(B, &b);
    launch_with(&mut cp, [1, 1, 1], [512, 1, 1], 24, Some(ONE_D_QMD)).unwrap();
    let got = cp.mem.read_f32s(C, m * n).unwrap();
    assert!(got.iter().all(|x| x.to_bits() == 0x8000_0000));
    // Step 3's 2 KiB of shared memory, a 16x16 block, no barrier: refused.
    let (mut cp, ..) = one_d_cp(64, 8, 64);
    assert!(launch_with(&mut cp, [1, 1, 1], [512, 1, 1], 24, Some(SMEM_QMD)).is_err());
    assert!(launch_with(&mut cp, [1, 1, 1], [16, 16, 1], 24, Some(ONE_D_QMD)).is_err());
    let no_bar = SharedMemory {
        barriers: 0,
        ..ONE_D_QMD
    };
    assert!(launch_with(&mut cp, [1, 1, 1], [512, 1, 1], 24, Some(no_bar)).is_err());
    let got = cp.mem.read_f32s(C, 64 * 64).unwrap();
    assert!(
        got.iter().all(|&x| x == SENTINEL),
        "refused launches wrote nothing"
    );
    assert_ne!(ONE_D, SMEM);
}

/// Step 5 (session 10): 2D blocktiling, 8x8 results per thread, 64 threads.
/// NAK: 1953 instructions, 31248 B, 94 registers, 4 KiB shared, 1 barrier,
/// 64 `ffma.ftz` (inner loops unrolled; no local memory).
fn two_d_cp(m: usize, k: usize, n: usize) -> (CommandProcessor, Vec<f32>, Vec<f32>) {
    let (mut cp, a, b) = setup(m as u32, k as u32, n as u32);
    cp.register_program_with_needs(TWO_D, matmul_2d_on_cpu, MATMUL_2D_NEEDS);
    cp.mem.write(PROG, TWO_D);
    (cp, a, b)
}

#[test]
fn two_d_step_matches_the_ffma_ftz_oracle_including_partial_blocks() {
    for (m, k, n) in [
        (37usize, 37usize, 21usize),
        (64, 8, 64),
        (5, 1, 3),
        (130, 19, 70),
        (71, 64, 129),
    ] {
        let (mut cp, a, b) = two_d_cp(m, k, n);
        let grid = [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1];
        launch_with(&mut cp, grid, [64, 1, 1], 94, Some(ONE_D_QMD)).unwrap();
        assert!(matches!(cp.events[1], Event::Launch { program: 1, .. }));
        let got = cp.mem.read_f32s(C, m * n + 16).unwrap();
        let mut want = vec![0.0; m * n];
        gpu_compute::reference::matmul_fma_ftz(&a, &b, &mut want, m, k, n).unwrap();
        assert_eq!(bits(&got[..m * n]), bits(&want), "{m}x{k}x{n}");
        assert!(got[m * n..].iter().all(|&x| x == SENTINEL), "wrote after C");
        assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    }
}

#[test]
fn two_d_step_keeps_negative_zero_and_refuses_wrong_launches() {
    let (m, k, n) = (9usize, 11usize, 2usize);
    let (mut cp, ..) = two_d_cp(m, k, n);
    let tiny = f32::from_bits(0x1c80_0000);
    cp.mem.write_f32s(A, &vec![-tiny; m * k]);
    cp.mem.write_f32s(B, &vec![tiny; k * n]);
    launch_with(&mut cp, [1, 1, 1], [64, 1, 1], 94, Some(ONE_D_QMD)).unwrap();
    let got = cp.mem.read_f32s(C, m * n).unwrap();
    assert!(got.iter().all(|x| x.to_bits() == 0x8000_0000));
    // Step 4's block shape, step 3's shared memory, no barrier: all refused.
    let (mut cp, ..) = two_d_cp(64, 8, 64);
    assert!(launch_with(&mut cp, [1, 1, 1], [512, 1, 1], 94, Some(ONE_D_QMD)).is_err());
    assert!(launch_with(&mut cp, [1, 1, 1], [64, 1, 1], 94, Some(SMEM_QMD)).is_err());
    let no_bar = SharedMemory {
        barriers: 0,
        ..ONE_D_QMD
    };
    assert!(launch_with(&mut cp, [1, 1, 1], [64, 1, 1], 94, Some(no_bar)).is_err());
    let got = cp.mem.read_f32s(C, 64 * 64).unwrap();
    assert!(
        got.iter().all(|&x| x == SENTINEL),
        "refused launches wrote nothing"
    );
    assert_ne!(TWO_D, ONE_D);
}

/// A planted mistake in the step 5 model (the 8x8 cell of a thread taken
/// transposed) must be caught by the oracle comparison.
#[test]
fn two_d_wrong_cell_layout_is_caught() {
    fn transposed(l: &mut gpu_sim::cmdproc::Launch<'_>) -> Result<(), gpu_core::error::GpuError> {
        matmul_2d_on_cpu(l)?;
        // Swap two cells of C as a wrong thread->cell mapping would.
        let c = l.mem.read_u64(l.cbuf0 + 0x38)?;
        let n = l.mem.read_u32(l.cbuf0 + 0x44)? as u64;
        let (x, y) = (l.mem.read_u32(c + 4)?, l.mem.read_u32(c + 4 * n)?);
        l.mem.write_u32(c + 4, y);
        l.mem.write_u32(c + 4 * n, x);
        Ok(())
    }
    let (m, k, n) = (16usize, 9usize, 16usize);
    let (mut cp, a, b) = setup(m as u32, k as u32, n as u32);
    cp.register_program_with_needs(TWO_D, transposed, MATMUL_2D_NEEDS);
    cp.mem.write(PROG, TWO_D);
    launch_with(&mut cp, [1, 1, 1], [64, 1, 1], 94, Some(ONE_D_QMD)).unwrap();
    let got = cp.mem.read_f32s(C, m * n).unwrap();
    let mut want = vec![0.0; m * n];
    gpu_compute::reference::matmul_fma_ftz(&a, &b, &mut want, m, k, n).unwrap();
    assert_ne!(bits(&got), bits(&want));
}

/// Step 6 (session 24): step 5 with 128-bit loads/stores and the A tile
/// transposed in shared memory. NAK: 821 instructions, 13136 B, 90 registers,
/// 4 KiB shared, 1 barrier, 64 `ffma.ftz`, 4 + 4 128-bit loads, 16 128-bit stores.
const VEC4_REGS: u32 = 90;
fn vec4_cp(m: usize, k: usize, n: usize) -> (CommandProcessor, Vec<f32>, Vec<f32>) {
    let (mut cp, a, b) = setup(m as u32, k as u32, n as u32);
    cp.register_program_with_needs(VEC4, matmul_vec4_on_cpu, MATMUL_VEC4_NEEDS);
    cp.mem.write(PROG, VEC4);
    (cp, a, b)
}

fn want(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut w = vec![0.0; m * n];
    gpu_compute::reference::matmul_fma_ftz(a, b, &mut w, m, k, n).unwrap();
    w
}

#[test]
fn vec4_step_matches_the_ffma_ftz_oracle_including_partial_blocks() {
    for (m, k, n) in [
        (37usize, 36usize, 20usize),
        (64, 8, 64),
        (5, 4, 4),
        (130, 20, 72),
        (71, 64, 132),
        (1, 12, 68),
    ] {
        let (mut cp, a, b) = vec4_cp(m, k, n);
        let grid = [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1];
        launch_with(&mut cp, grid, [64, 1, 1], VEC4_REGS, Some(ONE_D_QMD)).unwrap();
        assert!(matches!(cp.events[1], Event::Launch { program: 1, .. }));
        let got = cp.mem.read_f32s(C, m * n + 16).unwrap();
        assert_eq!(
            bits(&got[..m * n]),
            bits(&want(&a, &b, m, k, n)),
            "{m}x{k}x{n}"
        );
        assert!(got[m * n..].iter().all(|&x| x == SENTINEL), "wrote after C");
        assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    }
}

#[test]
fn vec4_step_gives_the_same_bits_as_step_5() {
    let (m, k, n) = (70usize, 28usize, 68usize);
    let grid = [n.div_ceil(64) as u32, m.div_ceil(64) as u32, 1];
    let (mut cp6, ..) = vec4_cp(m, k, n);
    launch_with(&mut cp6, grid, [64, 1, 1], VEC4_REGS, Some(ONE_D_QMD)).unwrap();
    let (mut cp5, ..) = two_d_cp(m, k, n);
    launch_with(&mut cp5, grid, [64, 1, 1], 94, Some(ONE_D_QMD)).unwrap();
    assert_eq!(
        bits(&cp6.mem.read_f32s(C, m * n).unwrap()),
        bits(&cp5.mem.read_f32s(C, m * n).unwrap())
    );
}

#[test]
fn vec4_step_keeps_negative_zero_and_refuses_what_it_cannot_run() {
    let (m, k, n) = (9usize, 12usize, 4usize);
    let (mut cp, ..) = vec4_cp(m, k, n);
    let tiny = f32::from_bits(0x1c80_0000);
    cp.mem.write_f32s(A, &vec![-tiny; m * k]);
    cp.mem.write_f32s(B, &vec![tiny; k * n]);
    launch_with(&mut cp, [1, 1, 1], [64, 1, 1], VEC4_REGS, Some(ONE_D_QMD)).unwrap();
    let got = cp.mem.read_f32s(C, m * n).unwrap();
    assert!(got.iter().all(|x| x.to_bits() == 0x8000_0000));
    // K or N not a multiple of 4: refused, nothing written.
    for (k, n) in [(10usize, 64usize), (8, 66)] {
        let (mut cp, ..) = vec4_cp(64, k, n);
        assert!(launch_with(&mut cp, [2, 1, 1], [64, 1, 1], VEC4_REGS, Some(ONE_D_QMD)).is_err());
        assert!(
            cp.mem
                .read_f32s(C, 64 * n)
                .unwrap()
                .iter()
                .all(|&x| x == SENTINEL)
        );
    }
    // C not aligned to 16 bytes: refused.
    let (mut cp, ..) = vec4_cp(64, 8, 64);
    cp.mem.write(CBUF + 0x38, &(C + 4).to_le_bytes());
    assert!(launch_with(&mut cp, [1, 1, 1], [64, 1, 1], VEC4_REGS, Some(ONE_D_QMD)).is_err());
    // Step 4's block shape, step 3's shared memory, no barrier: refused.
    let (mut cp, ..) = vec4_cp(64, 8, 64);
    assert!(launch_with(&mut cp, [1, 1, 1], [512, 1, 1], VEC4_REGS, Some(ONE_D_QMD)).is_err());
    assert!(launch_with(&mut cp, [1, 1, 1], [64, 1, 1], VEC4_REGS, Some(SMEM_QMD)).is_err());
    let no_bar = SharedMemory {
        barriers: 0,
        ..ONE_D_QMD
    };
    assert!(launch_with(&mut cp, [1, 1, 1], [64, 1, 1], VEC4_REGS, Some(no_bar)).is_err());
    let got = cp.mem.read_f32s(C, 64 * 64 + 16).unwrap();
    assert!(
        got.iter().all(|&x| x == SENTINEL),
        "refused launches wrote nothing"
    );
    assert_ne!(VEC4, TWO_D);
}

/// Planted mistakes in the step 6 model (a vec4 of C written one vector to
/// the right; the last vector of a row of B dropped) must be caught.
#[test]
fn vec4_planted_mistakes_are_caught() {
    fn shifted(l: &mut gpu_sim::cmdproc::Launch<'_>) -> Result<(), gpu_core::error::GpuError> {
        matmul_vec4_on_cpu(l)?;
        let c = l.mem.read_u64(l.cbuf0 + 0x38)?;
        let v = l.mem.read_f32s(c, 4)?;
        l.mem.write_f32s(c + 16, &v);
        Ok(())
    }
    fn dropped(l: &mut gpu_sim::cmdproc::Launch<'_>) -> Result<(), gpu_core::error::GpuError> {
        let b = l.mem.read_u64(l.cbuf0 + 0x30)?;
        let n = l.mem.read_u32(l.cbuf0 + 0x44)? as u64;
        l.mem.write_f32s(b + 4 * (n - 4), &[0.0; 4]);
        matmul_vec4_on_cpu(l)
    }
    for f in [shifted as gpu_sim::cmdproc::ProgramFn, dropped] {
        let (m, k, n) = (16usize, 12usize, 16usize);
        let (mut cp, a, b) = setup(m as u32, k as u32, n as u32);
        cp.register_program_with_needs(VEC4, f, MATMUL_VEC4_NEEDS);
        cp.mem.write(PROG, VEC4);
        launch_with(&mut cp, [1, 1, 1], [64, 1, 1], VEC4_REGS, Some(ONE_D_QMD)).unwrap();
        let got = cp.mem.read_f32s(C, m * n).unwrap();
        assert_ne!(bits(&got), bits(&want(&a, &b, m, k, n)));
    }
}
