//! C-147 point 5 (T142), SIMULARE: the first tensor-core program, `mma16_f16` (second NAK tool, Mesa 26.3.1),
//! launched through the command-processor model the way the driver would send it (SET_OBJECT, SEND_PCAS_A,
//! SEND_SIGNALING_PCAS2_B, host semaphore), with the QMD and the constant buffer from `nvidia_ga104::mma16`.
//! D is judged by `gpu_compute::mma::judge` (reference in f64 + written tolerance). The model's arithmetic
//! (`mma16_block8_trunc`) is a hypothesis; nothing here proves anything about the RTX 3070 Ti.

use gpu_compute::mma::{self, MmaVerdict};
use gpu_sim::cmdproc::{
    CommandProcessor, Event, MMA16_F16_NEEDS, MMA16_PUSH_BASE, mma16_f16_on_cpu,
};
use nvidia_ga104::mma16::{self, MMA16_F16};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const A: u64 = 0x40_0000;
const B: u64 = 0x40_1000;
const C: u64 = 0x40_2000;
const D: u64 = 0x40_3000;
const QMD: u64 = 0x50_0000;
const PUSH: u64 = 0x60_0000;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// A value in [-2, 2), rounded to binary16 by the driver-side conversion.
    fn h(&mut self) -> u16 {
        mma::f32_to_f16(self.next() as f32 / (1u64 << 31) as f32 * 2.0 - 2.0)
    }
}

fn halves(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Case {
    a: Vec<u16>,
    b: Vec<u16>,
    c: Vec<f32>,
}

fn case(seed: u64) -> Case {
    let mut g = Lcg(seed);
    Case {
        a: (0..mma::M * mma::K).map(|_| g.h()).collect(),
        b: (0..mma::K * mma::N).map(|_| g.h()).collect(),
        c: (0..mma::M * mma::N)
            .map(|_| mma::f16_to_f32(g.h()))
            .collect(),
    }
}

fn setup(k: &Case, program: &[u8]) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(MMA16_F16, mma16_f16_on_cpu, MMA16_F16_NEEDS);
    cp.mem.write(PROG, program);
    cp.mem.write(CBUF, &mma16::cbuf(A, B, C, D).unwrap());
    cp.mem.write(A, &halves(&k.a));
    cp.mem.write(B, &halves(&k.b));
    cp.mem.write_f32s(C, &k.c);
    // D starts as NaN, so an element nobody wrote cannot pass.
    cp.mem.write_f32s(D, &[f32::NAN; 256]);
    cp.mem.write_u32(SEM, 0);
    cp
}

fn submit(cp: &mut CommandProcessor) -> Result<(), gpu_core::error::GpuError> {
    let mut words = [0u32; 64];
    let mut p = PushBuf::new(&mut words);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let exec = h::NVC56F_SEM_EXECUTE_OPERATION_RELEASE;
    p.inc(0, h::NVC56F_SEM_ADDR_LO, &[SEM as u32 + 8, 0, 9, 0, exec])
        .unwrap();
    let n = p.len();
    let bytes: Vec<u8> = words[..n].iter().flat_map(|w| w.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

#[test]
fn t142a_push_base_is_the_same_in_driver_and_model() {
    assert_eq!(mma16::PUSH_BASE as u64, MMA16_PUSH_BASE);
}

#[test]
fn t142b_mma16_runs_in_the_model_within_tolerance() {
    let mut not_exact = 0;
    for seed in 1..=8 {
        let k = case(seed);
        let mut cp = setup(&k, MMA16_F16);
        cp.mem.write(QMD, &mma16::qmd(PROG, CBUF).unwrap());
        submit(&mut cp).unwrap();
        let got = cp.mem.read_f32s(D, 256).unwrap();
        match mma::judge(&k.a, &k.b, &k.c, &got) {
            MmaVerdict::InToleranta { exact } => not_exact += 256 - exact,
            v => panic!("seed {seed}: {v:?}"),
        }
        assert_eq!(
            cp.mem.read_u32(SEM + 8).unwrap(),
            9,
            "host semaphore after the launch"
        );
        assert!(cp.events.contains(&Event::Launch {
            qmd: QMD,
            program: 0
        }));
    }
    // The hypothesis truncates: some elements must differ from the f64 reference, or the
    // tolerance would never be exercised.
    assert!(not_exact > 0);
}

#[test]
fn t142c_wrong_launches_are_refused_and_d_stays_untouched() {
    let k = case(5);
    // One byte of the program changed: the model does not recognise it.
    let mut other = MMA16_F16.to_vec();
    other[200] ^= 1;
    let mut cp = setup(&k, &other);
    cp.mem.write(QMD, &mma16::qmd(PROG, CBUF).unwrap());
    assert!(submit(&mut cp).is_err());
    // Addresses written at 0x28 (the first tool's layout) instead of 0x700: the program would read zeros.
    let mut cp = setup(&k, MMA16_F16);
    let mut cb = vec![0u8; mma16::CBUF_SIZE as usize];
    for (i, va) in [A, B, C, D].into_iter().enumerate() {
        cb[0x28 + 8 * i..0x30 + 8 * i].copy_from_slice(&va.to_le_bytes());
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write(QMD, &mma16::qmd(PROG, CBUF).unwrap());
    assert!(submit(&mut cp).is_err());
    assert!(cp.mem.read_f32s(D, 256).unwrap().iter().all(|v| v.is_nan()));
    // Two warps instead of one.
    let mut cp = setup(&k, MMA16_F16);
    let mut q = nvidia_ga104::qmd::ComputeQmd {
        program_va: PROG,
        registers: mma16::MMA16_F16_REGISTERS,
        cbuf0_va: CBUF,
        cbuf0_size: mma16::CBUF_SIZE,
        grid: [1, 1, 1],
        block: [64, 1, 1],
        release: None,
    };
    cp.mem.write(QMD, &q.to_bytes().unwrap());
    assert!(submit(&mut cp).is_err());
    // A grid of two blocks.
    q.block = [32, 1, 1];
    q.grid = [2, 1, 1];
    let mut cp = setup(&k, MMA16_F16);
    cp.mem.write(QMD, &q.to_bytes().unwrap());
    assert!(submit(&mut cp).is_err());
    assert!(cp.mem.read_f32s(D, 256).unwrap().iter().all(|v| v.is_nan()));
}
