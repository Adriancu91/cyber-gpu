//! C-69, SIMULARE: measurement 1 of the one-program target (the cost of one
//! launch) run end to end on the command processor model: start marker, N
//! `add64` launches, a timestamp release after each. The model's clock is
//! the test's own (`sim_launch_ns`), so the numbers prove only that the
//! encoding, the records and their reading agree — nothing about the board.

use gpu_sim::cmdproc::{CommandProcessor, add64_on_cpu};
use nvidia_ga104::launch_cost::{self, RECORD};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::clc7c0 as c;

const ADD64: &[u8] = include_bytes!("../../../tools/nak/out/add64.sm86.sass");
const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const REC: u64 = 0x30_0000;
const A: u64 = 0x40_0000;
const B: u64 = 0x40_1000;
const C: u64 = 0x40_2000;
const QMD: u64 = 0x50_0000;
const PUSH: u64 = 0x60_0000;

fn board(launch_ns: u64, method_ns: u64) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program(ADD64, add64_on_cpu);
    cp.mem.write(PROG, ADD64);
    let mut cb = vec![0u8; 0x100];
    for (off, va) in [(0x28, A), (0x30, B), (0x38, C)] {
        cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(A, &[1.0; 64]);
    cp.mem.write_f32s(B, &[2.0; 64]);
    cp.mem.write_f32s(C, &[0.0; 64]);
    let q = ComputeQmd {
        program_va: PROG,
        registers: 19,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid: [1, 1, 1],
        block: [64, 1, 1],
        release: None,
    };
    cp.mem.write(QMD, &q.to_bytes().unwrap());
    cp.sim_launch_ns = launch_ns;
    cp.sim_method_ns = method_ns;
    cp.gpu_time_ns = 1_000_000;
    cp
}

fn run(cp: &mut CommandProcessor, n: u32, first: u32) -> Result<(), gpu_core::error::GpuError> {
    let mut words = vec![0u32; 16 + 16 * n as usize];
    let mut p = PushBuf::new(&mut words);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    launch_cost::encode(&mut p, QMD, REC, n, first).unwrap();
    let len = p.len();
    let bytes: Vec<u8> = words[..len].iter().flat_map(|w| w.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, len as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

fn read_records(cp: &CommandProcessor, n: u32) -> Vec<u8> {
    let mut b = vec![0u8; ((n as u64 + 1) * RECORD) as usize];
    cp.mem.read(REC, &mut b).unwrap();
    b
}

/// T75: 16 launches, simulated cost 4000 ns per launch + 10 ns per method.
#[test]
fn launch_cost_is_read_from_gpu_timestamps_simulated() {
    let mut cp = board(4000, 10);
    cp.mem.write(REC, &vec![0u8; 17 * RECORD as usize]);
    run(&mut cp, 16, 0x500).unwrap();
    assert_eq!(cp.timestamps.len(), 17);
    let cost = launch_cost::analyze(&read_records(&cp, 16), 16, 0x500).unwrap();
    // Per launch: 2 launch methods + 5 semaphore methods = 7 x 10 ns + 4000.
    assert_eq!(cost.each_ns, vec![4070; 16]);
    assert_eq!(
        (cost.min_ns, cost.median_ns, cost.max_ns),
        (4070, 4070, 4070)
    );
    assert_eq!(cost.total_ns, 16 * 4070);
    // The work itself was done 16 times on the same output (1 + 2).
    assert_eq!(cp.mem.read_f32s(C, 64).unwrap(), vec![3.0; 64]);
}

/// T75: a launch the model refuses stops the stream: the records after it
/// keep their old value and the reading says so, instead of a number.
#[test]
fn a_launch_that_did_not_finish_gives_no_number() {
    let mut cp = board(4000, 0);
    cp.mem.write(REC, &vec![0u8; 5 * RECORD as usize]);
    // Program bytes changed: the model refuses the first launch.
    let mut other = ADD64.to_vec();
    other[100] ^= 1;
    cp.mem.write(PROG, &other);
    assert!(run(&mut cp, 4, 0x500).is_err());
    assert_eq!(cp.timestamps.len(), 1, "only the start marker");
    assert!(launch_cost::analyze(&read_records(&cp, 4), 4, 0x500).is_err());
}
