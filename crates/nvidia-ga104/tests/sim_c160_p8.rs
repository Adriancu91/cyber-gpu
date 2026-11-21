//! C-160 point 1 (T150), SIMULARE: the P8 probe as the OS would send it — for every set of
//! `nvidia_ga104::mma16_probe::SETS`, one host page (A, B, C, D = NaN), the cbuf + QMD from `launch_bytes`, one
//! launch through the command-processor model, D read back and judged. The model's arithmetic
//! (`mma16_block8_trunc`) is a hypothesis; nothing here proves anything about the RTX 3070 Ti.

use gpu_compute::mma::{self, ADD_MODELS};
use gpu_core::diag::Journal;
use gpu_sim::cmdproc::{CommandProcessor, MMA16_F16_NEEDS, mma16_f16_on_cpu};
use nvidia_ga104::mma16::MMA16_F16;
use nvidia_ga104::mma16_probe::{self as mp, D_BYTES, D_OFF, PAGE_BYTES, Probe, SETS, parse_mma16};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const QMD: u64 = 0x20_0800;
const SEM: u64 = 0x30_0000;
const PAGE: u64 = 0x40_0000;
const PUSH: u64 = 0x60_0000;

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

/// One set through the model; `page` lets a test plant a mistake in the bytes; `launch = false` skips the launch.
fn run_set(i: usize, page: [u8; PAGE_BYTES], launch: bool) -> Vec<u8> {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(MMA16_F16, mma16_f16_on_cpu, MMA16_F16_NEEDS);
    cp.mem.write(PROG, MMA16_F16);
    let (cb, q) = mp::launch_bytes(PROG, CBUF, PAGE).unwrap();
    cp.mem.write(CBUF, &cb);
    cp.mem.write(QMD, &q);
    cp.mem.write(PAGE, &page);
    cp.mem.write_u32(SEM, 0);
    if launch {
        submit(&mut cp).unwrap_or_else(|e| panic!("set {i}: {e:?}"));
        assert_eq!(cp.mem.read_u32(SEM + 8).unwrap(), 9);
    }
    let mut d = vec![0u8; D_BYTES];
    cp.mem.read(PAGE + D_OFF, &mut d).unwrap();
    d
}

#[test]
fn t150a_the_probe_passes_and_names_the_models_adder() {
    let mut p = Probe::default();
    for (i, s) in SETS.iter().enumerate() {
        let d = run_set(i, s.page(), true);
        p.add(mp::judge_set(i, &d, Some(1000 + i as u64)).unwrap());
    }
    assert!(p.passes(), "{:#?}", p.lines("SIMULARE"));
    assert!(p.worst_permille() <= 1000);
    // The model is block-8 truncation by construction, and the sets are chosen so that only that one fits.
    assert_eq!(p.models(), 1 << 8, "{}", mp::model_names(p.models()));
    assert_eq!(mp::model_names(p.models()), "bloc8-taiere");
    // The identity set: every element exact.
    assert_eq!(p.sets[0].exact(), 256);
    for l in p.lines("SIMULARE") {
        println!("{l}");
    }
}

#[test]
fn t150b_the_sets_tell_every_candidate_model_apart() {
    // For every pair of candidate adders, at least one set gives a different D: whatever the board matches,
    // the mask names at most one model.
    for (i, m1) in ADD_MODELS.iter().enumerate() {
        for m2 in &ADD_MODELS[i + 1..] {
            let differs = SETS.iter().any(|s| {
                let x = s.inputs();
                let d1 = mma::mma16_blocked(&x.a, &x.b, &x.c, *m1).unwrap();
                let d2 = mma::mma16_blocked(&x.a, &x.b, &x.c, *m2).unwrap();
                d1.iter().zip(&d2).any(|(a, b)| a.to_bits() != b.to_bits())
            });
            assert!(differs, "{} = {} on every set", m1.name(), m2.name());
        }
    }
    // And every candidate model stays within the written tolerance on every set: the tolerance does not
    // decide which model the board uses, only whether it computes the product.
    for s in &SETS {
        let x = s.inputs();
        for m in &ADD_MODELS {
            let d = mma::mma16_blocked(&x.a, &x.b, &x.c, *m).unwrap();
            assert!(
                mma::worst_permille(&x.a, &x.b, &x.c, &d).unwrap() <= 1000,
                "{} {}",
                s.name,
                m.name()
            );
        }
    }
}

#[test]
fn t150c_rows_read_back_from_a_journal() {
    let mut p = Probe::default();
    for (i, s) in SETS.iter().enumerate() {
        let d = run_set(i, s.page(), true);
        p.add(mp::judge_set(i, &d, Some(2000 + i as u64)).unwrap());
    }
    let mut j = Journal::new(256);
    p.journal(&mut j);
    assert_eq!(j.len(), p.rows().len());
    let text: String = p
        .rows()
        .iter()
        .map(|(w, v)| format!("gpu: valoare {w} = {v:#x}\n"))
        .collect();
    let r = parse_mma16(&text);
    assert_eq!(r.sets, Some(6));
    assert_eq!(r.in_tolerance, Some(6));
    assert_eq!(r.models, Some(1 << 8));
    assert_eq!(r.exact, Some(p.exact() as u32));
    assert_eq!(r.median_ns, Some(2003));
    assert_eq!(r.per_set.len(), 6);
    assert_eq!(r.per_set[0][1], SETS[0].inputs_fnv());
    assert_eq!(r.per_set[0][3], 256);
    assert_eq!(r.per_set[5][6], 2005);
    // A second run in the same journal replaces the first.
    let twice = format!("{text}{text}");
    assert_eq!(parse_mma16(&twice), r);
}

#[test]
fn t150d_planted_mistakes_are_caught() {
    // Not launched: D stays NaN -> invalid, never "within tolerance".
    let d = run_set(1, SETS[1].page(), false);
    let r = mp::judge_set(1, &d, None).unwrap();
    assert_eq!(r.code(), 2);
    // B written transposed into the page: the board would compute another product -> outside tolerance.
    let mut page = SETS[1].page();
    let x = SETS[1].inputs();
    for i in 0..256 {
        let t = x.b[(i % 16) * 16 + i / 16];
        let at = mp::B_OFF as usize + 2 * i;
        page[at..at + 2].copy_from_slice(&t.to_le_bytes());
    }
    let d = run_set(1, page, true);
    let r = mp::judge_set(1, &d, None).unwrap();
    assert_eq!(r.code(), 0);
    // The D of another set: caught as well.
    let d = run_set(2, SETS[2].page(), true);
    assert_eq!(mp::judge_set(3, &d, None).unwrap().code(), 0);
    // One set missing: the probe does not pass.
    let mut p = Probe::default();
    for (i, s) in SETS.iter().enumerate().skip(1) {
        p.add(mp::judge_set(i, &run_set(i, s.page(), true), None).unwrap());
    }
    assert!(!p.passes());
}
