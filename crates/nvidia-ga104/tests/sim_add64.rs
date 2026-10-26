//! C-20 D (part 1), SIMULARE: the whole first-launch path — pushbuffer and
//! GPFIFO from `push.rs`, QMD from `qmd.rs`, the program compiled by Mesa NAK
//! (`tools/nak/out/add64.sm86.sass`, 240 bytes) — executed by the command
//! processor model in `gpu-sim::cmdproc`, which reads every field with the
//! numbers generated from NVIDIA's headers. The "program" runs on the CPU.
//! Nothing here touches or proves anything about the real board.

use gpu_sim::cmdproc::{CommandProcessor, Event, add64_on_cpu};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

const ADD64: &[u8] = include_bytes!("../../../tools/nak/out/add64.sm86.sass");
const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const A: u64 = 0x40_0000;
const B: u64 = 0x40_1000;
const C: u64 = 0x40_2000;
const QMD: u64 = 0x50_0000;
const PUSH: u64 = 0x60_0000;

fn setup() -> (CommandProcessor, Vec<f32>, Vec<f32>) {
    let mut cp = CommandProcessor::new();
    cp.register_program(ADD64, add64_on_cpu);
    let a: Vec<f32> = (0..64).map(|i| i as f32 * 0.25 - 3.0).collect();
    let b: Vec<f32> = (0..64).map(|i| (i * i) as f32 / 7.0).collect();
    cp.mem.write(PROG, ADD64);
    let mut cb = vec![0u8; 0x100];
    for (off, va) in [(0x28, A), (0x30, B), (0x38, C)] {
        cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(A, &a);
    cp.mem.write_f32s(B, &b);
    cp.mem.write_f32s(C, &[f32::NAN; 64]);
    cp.mem.write_u32(SEM, 0);
    (cp, a, b)
}

fn qmd(release: Option<(u64, u32)>) -> ComputeQmd {
    ComputeQmd {
        program_va: PROG,
        registers: 19,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid: [1, 1, 1],
        block: [64, 1, 1],
        release,
    }
}

/// SET_OBJECT, SEND_PCAS_A, SEND_SIGNALING_PCAS2_B, then a host semaphore
/// release with value 7 — the order of NAK `hw_runner.rs` / tinygrad.
fn pushbuffer(words: &mut [u32]) -> usize {
    let mut p = PushBuf::new(words);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let exec = h::NVC56F_SEM_EXECUTE_OPERATION_RELEASE;
    p.inc(0, h::NVC56F_SEM_ADDR_LO, &[SEM as u32 + 8, 0, 7, 0, exec])
        .unwrap();
    p.len()
}

fn submit(cp: &mut CommandProcessor) -> Result<(), gpu_core::error::GpuError> {
    let mut words = [0u32; 64];
    let n = pushbuffer(&mut words);
    let bytes: Vec<u8> = words[..n].iter().flat_map(|w| w.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

#[test]
fn add64_runs_in_the_model_and_matches_the_reference() {
    let (mut cp, a, b) = setup();
    cp.mem.write(QMD, &qmd(Some((SEM, 1))).to_bytes().unwrap());
    submit(&mut cp).unwrap();

    let mut want = vec![0.0; 64];
    gpu_compute::reference::add_ftz(&a, &b, &mut want).unwrap();
    let got = cp.mem.read_f32s(C, 64).unwrap();
    assert_eq!(
        got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        want.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1, "release0 from the QMD");
    assert_eq!(
        cp.mem.read_u32(SEM + 8).unwrap(),
        7,
        "host semaphore after the launch"
    );
    assert_eq!(
        cp.events,
        vec![
            Event::SetObject {
                subch: 1,
                class: 0xc7c0
            },
            Event::Launch {
                qmd: QMD,
                program: 0
            },
            Event::Release {
                addr: SEM,
                payload: 1
            },
            Event::Release {
                addr: SEM + 8,
                payload: 7
            },
        ]
    );
}

#[test]
fn bad_launches_are_refused() {
    // Program bytes changed by one byte: the model does not recognise it.
    let (mut cp, ..) = setup();
    let mut other = ADD64.to_vec();
    other[100] ^= 1;
    cp.mem.write(PROG, &other);
    cp.mem.write(QMD, &qmd(None).to_bytes().unwrap());
    assert!(submit(&mut cp).is_err());

    // QMD never written: reading unmapped GPU memory faults.
    let (mut cp, ..) = setup();
    assert!(submit(&mut cp).is_err());

    // A semaphore that never arrives: acquire >= 5 on a zero word.
    let (mut cp, ..) = setup();
    let mut w = [0u32; 8];
    let mut p = PushBuf::new(&mut w);
    let acq = h::NVC56F_SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ;
    p.inc(0, h::NVC56F_SEM_ADDR_LO, &[SEM as u32, 0, 5, 0, acq])
        .unwrap();
    let n = p.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    assert!(cp.run_gpfifo(&[(e.entry0, e.entry1)]).is_err());
}
