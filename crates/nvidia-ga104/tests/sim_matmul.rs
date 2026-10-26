//! C-20 E (step 1), SIMULARE: the naive matmul compiled by Mesa NAK
//! (`tools/nak/out/matmul_naive.sm86.sass`, 816 bytes, 51 instructions,
//! 19 registers) launched through our pushbuffer + QMD in the command
//! processor model. The "program" runs on the CPU (`matmul_naive_on_cpu`,
//! oracle `reference::matmul_fma_ftz`); nothing here runs on the RTX 3070 Ti.

use gpu_sim::cmdproc::{CommandProcessor, Event, matmul_naive_on_cpu};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::clc7c0 as c;

const MATMUL: &[u8] = include_bytes!("../../../tools/nak/out/matmul_naive.sm86.sass");
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
    let q = ComputeQmd {
        program_va: PROG,
        registers: 19,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid,
        block: [16, 16, 1],
        release: Some((SEM, 1)),
    };
    cp.mem.write(QMD, &q.to_bytes().unwrap());
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
