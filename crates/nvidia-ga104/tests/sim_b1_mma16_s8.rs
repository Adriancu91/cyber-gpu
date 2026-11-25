//! B1 (T229), SIMULARE: the int8 tensor-core probe `mma16_s8` (second NAK tool, Mesa 26.3.1; two
//! `imma.m16n8k32.i8.i8`) launched through the command-processor model the way the driver would send it
//! (SET_OBJECT, SEND_PCAS_A, SEND_SIGNALING_PCAS2_B, host semaphore), with the QMD and the constant buffer
//! from `nvidia_ga104::mma16`. Integer sums cannot depend on the adder's order, so D is judged **bit for bit**
//! against `gpu_compute::mma_s8::mma16_s8_ref`. Nothing here proves anything about the RTX 3070 Ti.

use gpu_compute::mma_s8::{self as s8, mma16_s8_ref, mma16_s8_wrap};
use gpu_sim::cmdproc::{
    CommandProcessor, Event, MMA16_F16_NEEDS, MMA16_PUSH_BASE, MMA16_S8_NEEDS, mma16_f16_on_cpu,
    mma16_s8_on_cpu,
};
use nvidia_ga104::mma16::{self, MMA16_F16, MMA16_S8};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

const PROG: u64 = 0x10_0000;
const PROG_F16: u64 = 0x11_0000;
const CBUF: u64 = 0x20_0000;
const CBUF_F16: u64 = 0x21_0000;
const SEM: u64 = 0x30_0000;
const A: u64 = 0x40_0000;
const B: u64 = 0x40_1000;
const C: u64 = 0x40_2000;
const D: u64 = 0x40_3000;
const D_F16: u64 = 0x40_4000;
const QMD: u64 = 0x50_0000;
const QMD_F16: u64 = 0x50_1000;
const PUSH: u64 = 0x60_0000;
/// What D holds before a launch, so an element nobody wrote cannot pass.
const UNWRITTEN: i32 = 0x7fff_fff7;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

struct Case {
    a: Vec<i8>,
    b: Vec<i8>,
    c: Vec<i32>,
}

fn random(seed: u64) -> Case {
    let mut g = Lcg(seed);
    Case {
        a: (0..s8::M * s8::K).map(|_| g.next() as u8 as i8).collect(),
        b: (0..s8::K * s8::N).map(|_| g.next() as u8 as i8).collect(),
        c: (0..s8::M * s8::N)
            .map(|_| (g.next() % (1 << 21)) as i32 - (1 << 20))
            .collect(),
    }
}

fn i8s(v: &[i8]) -> Vec<u8> {
    v.iter().map(|x| *x as u8).collect()
}

fn i32s(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn read_d(cp: &CommandProcessor, at: u64) -> Vec<i32> {
    let mut raw = vec![0u8; 4 * s8::M * s8::N];
    cp.mem.read(at, &mut raw).unwrap();
    raw.chunks(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect()
}

fn setup(k: &Case, program: &[u8]) -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(MMA16_S8, mma16_s8_on_cpu, MMA16_S8_NEEDS);
    cp.mem.write(PROG, program);
    cp.mem.write(CBUF, &mma16::cbuf_s8(A, B, C, D).unwrap());
    cp.mem.write(A, &i8s(&k.a));
    cp.mem.write(B, &i8s(&k.b));
    cp.mem.write(C, &i32s(&k.c));
    cp.mem.write(D, &i32s(&[UNWRITTEN; 256]));
    cp.mem.write_u32(SEM, 0);
    cp
}

fn submit(cp: &mut CommandProcessor, qmds: &[u64]) -> Result<(), gpu_core::error::GpuError> {
    let mut words = [0u32; 96];
    let mut p = PushBuf::new(&mut words);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    for q in qmds {
        p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(q >> 8) as u32]).unwrap();
        let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
        p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
            .unwrap();
    }
    let exec = h::NVC56F_SEM_EXECUTE_OPERATION_RELEASE;
    p.inc(0, h::NVC56F_SEM_ADDR_LO, &[SEM as u32 + 8, 0, 9, 0, exec])
        .unwrap();
    let n = p.len();
    let bytes: Vec<u8> = words[..n].iter().flat_map(|w| w.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

fn run(k: &Case) -> Vec<i32> {
    let mut cp = setup(k, MMA16_S8);
    cp.mem.write(QMD, &mma16::qmd_s8(PROG, CBUF).unwrap());
    submit(&mut cp, &[QMD]).unwrap();
    assert_eq!(cp.mem.read_u32(SEM + 8).unwrap(), 9, "host semaphore");
    assert!(cp.events.contains(&Event::Launch {
        qmd: QMD,
        program: 0
    }));
    read_d(&cp, D)
}

#[test]
fn t229a_program_qmd_and_push_base() {
    assert_eq!(MMA16_S8.len(), 2_368);
    assert_eq!(mma16::MMA16_S8_REGISTERS, 32);
    assert_eq!(mma16::PUSH_BASE as u64, MMA16_PUSH_BASE);
    // The s8 QMD differs from the f16 one only where the program address does (same block, registers, cbuf).
    let qs = mma16::qmd_s8(PROG, CBUF).unwrap();
    let qf = mma16::qmd(PROG, CBUF).unwrap();
    assert_eq!(qs, qf);
}

#[test]
fn t229b_random_sets_are_bit_exact() {
    for seed in 1..=8 {
        let k = random(seed);
        let want = mma16_s8_ref(&k.a, &k.b, &k.c).expect("random sets fit in i32");
        let got = run(&k);
        assert_eq!(got, want.to_vec(), "seed {seed}");
        assert!(!got.contains(&UNWRITTEN));
    }
}

#[test]
fn t229c_edges() {
    let full = |a: i8, b: i8, c: i32| Case {
        a: vec![a; s8::M * s8::K],
        b: vec![b; s8::K * s8::N],
        c: vec![c; s8::M * s8::N],
    };
    let k = full(-128, -128, -3);
    assert!(run(&k).iter().all(|v| *v == 524_288 - 3));
    let k = full(-128, 127, 11);
    assert!(run(&k).iter().all(|v| *v == -32 * 16_256 + 11));
    // A = [I16 | 0]: D = the first 16 rows of B + C (catches a row/column mix-up of A or B).
    let mut k = random(99);
    k.a = (0..s8::M * s8::K)
        .map(|i| i8::from(i / s8::K == i % s8::K))
        .collect();
    let got = run(&k);
    for i in 0..s8::M {
        for j in 0..s8::N {
            assert_eq!(
                got[i * s8::N + j],
                k.b[i * s8::N + j] as i32 + k.c[i * s8::N + j]
            );
        }
    }
    assert_eq!(got, mma16_s8_ref(&k.a, &k.b, &k.c).unwrap().to_vec());
}

#[test]
fn t229d_overflow_is_refused_by_the_oracle_and_wraps_in_the_model() {
    let k = Case {
        a: vec![1; s8::M * s8::K],
        b: vec![1; s8::K * s8::N],
        c: vec![i32::MAX; s8::M * s8::N],
    };
    assert_eq!(mma16_s8_ref(&k.a, &k.b, &k.c), None);
    let got = run(&k);
    assert!(got.iter().all(|v| *v == i32::MAX.wrapping_add(32)));
    assert_eq!(got, mma16_s8_wrap(&k.a, &k.b, &k.c).unwrap().to_vec());
}

#[test]
fn t229e_wrong_launches_are_refused_and_d_stays_untouched() {
    let k = random(5);
    let untouched = |cp: &CommandProcessor| read_d(cp, D).iter().all(|v| *v == UNWRITTEN);
    // One byte of the program changed.
    let mut other = MMA16_S8.to_vec();
    other[300] ^= 1;
    let mut cp = setup(&k, &other);
    cp.mem.write(QMD, &mma16::qmd_s8(PROG, CBUF).unwrap());
    assert!(submit(&mut cp, &[QMD]).is_err());
    assert!(untouched(&cp));
    // Addresses at 0x28 (the first tool's layout) instead of 0x700.
    let mut cp = setup(&k, MMA16_S8);
    let mut cb = vec![0u8; mma16::CBUF_SIZE as usize];
    for (i, va) in [A, B, C, D].into_iter().enumerate() {
        cb[0x28 + 8 * i..0x30 + 8 * i].copy_from_slice(&va.to_le_bytes());
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write(QMD, &mma16::qmd_s8(PROG, CBUF).unwrap());
    assert!(submit(&mut cp, &[QMD]).is_err());
    assert!(untouched(&cp));
    // B misaligned by 8 (written by hand: cbuf_s8 refuses it).
    let mut cp = setup(&k, MMA16_S8);
    let mut cb = vec![0u8; mma16::CBUF_SIZE as usize];
    for (i, va) in [A, B + 8, C, D].into_iter().enumerate() {
        cb[0x700 + 8 * i..0x708 + 8 * i].copy_from_slice(&va.to_le_bytes());
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write(QMD, &mma16::qmd_s8(PROG, CBUF).unwrap());
    assert!(submit(&mut cp, &[QMD]).is_err());
    assert!(untouched(&cp));
    // Two warps instead of one.
    let mut cp = setup(&k, MMA16_S8);
    let q = nvidia_ga104::qmd::ComputeQmd {
        program_va: PROG,
        registers: mma16::MMA16_S8_REGISTERS,
        cbuf0_va: CBUF,
        cbuf0_size: mma16::CBUF_SIZE,
        grid: [1, 1, 1],
        block: [64, 1, 1],
        release: None,
    };
    cp.mem.write(QMD, &q.to_bytes().unwrap());
    assert!(submit(&mut cp, &[QMD]).is_err());
    assert!(untouched(&cp));
}

#[test]
fn t229f_f16_and_s8_in_one_processor_each_run_their_own_program() {
    let k = random(7);
    let mut cp = setup(&k, MMA16_S8);
    cp.register_program_with_needs(MMA16_F16, mma16_f16_on_cpu, MMA16_F16_NEEDS);
    cp.mem.write(PROG_F16, MMA16_F16);
    // The f16 program reads A and B as 256 halves each (A's first 512 bytes, B's 512 bytes); C as f32 zeros.
    let zeros = vec![0u8; 4 * 256];
    let c_f16 = 0x40_5000;
    cp.mem.write(c_f16, &zeros);
    cp.mem
        .write(CBUF_F16, &mma16::cbuf(A, B, c_f16, D_F16).unwrap());
    cp.mem.write(D_F16, &i32s(&[UNWRITTEN; 256]));
    cp.mem.write(QMD, &mma16::qmd_s8(PROG, CBUF).unwrap());
    cp.mem
        .write(QMD_F16, &mma16::qmd(PROG_F16, CBUF_F16).unwrap());
    // Random i8 bytes read as halves can be inf/NaN; the f16 model may then refuse. Use small bytes for B
    // in this test only, so both programs run: halves with high byte in 0..=0x3b are finite.
    let mut k2 = random(7);
    for (i, v) in k2.b.iter_mut().enumerate() {
        if i % 2 == 1 {
            *v &= 0x1b;
        }
    }
    for (i, v) in k2.a.iter_mut().take(512).enumerate() {
        if i % 2 == 1 {
            *v &= 0x1b;
        }
    }
    cp.mem.write(A, &i8s(&k2.a));
    cp.mem.write(B, &i8s(&k2.b));
    submit(&mut cp, &[QMD, QMD_F16]).unwrap();
    assert!(cp.events.contains(&Event::Launch {
        qmd: QMD,
        program: 0
    }));
    assert!(cp.events.contains(&Event::Launch {
        qmd: QMD_F16,
        program: 1
    }));
    assert_eq!(
        read_d(&cp, D),
        mma16_s8_ref(&k2.a, &k2.b, &k.c).unwrap().to_vec()
    );
    // The f16 program wrote f32 values over D_F16 (not the s8 sentinel any more).
    assert!(read_d(&cp, D_F16).iter().any(|v| *v != UNWRITTEN));
}
