//! T245 (C-157, `docs/C-157_PROBA_BARIERA.md`): the grid-barrier probe in SIMULARE — the driver's constant buffer
//! and QMD (`nvidia_ga104::grid_barrier`) run through the program's model (`gpu_sim::cmdproc::grid_barrier_on_cpu`),
//! and the verdict on what came back. SIMULARE: says nothing about the barrier on a real SM.

use gpu_sim::cmdproc::{GpuMem, Launch, grid_barrier_on_cpu};
use nvidia_ga104::grid_barrier::{self as gb, Verdict};
use nvidia_ga104::gsp::work_queue::GrShape;

const GA104: GrShape = GrShape {
    gpcs: 6,
    tpc_per_gpc: 4,
    sm_per_tpc: 2,
    max_warps_per_sm: 48,
};

const CBUF: u64 = 0x10_0000;
const SYNC: u64 = 0x20_0000;
const DATA: u64 = 0x30_0000;
const OUT: u64 = 0x40_0000;

fn setup(groups: u32, rounds: u32) -> GpuMem {
    let mut mem = GpuMem::default();
    mem.write(CBUF, &gb::cbuf(SYNC, DATA, OUT, rounds, groups).unwrap());
    let (s, d, o) = gb::buffer_bytes(groups);
    mem.write(SYNC, &vec![0; s as usize]);
    mem.write(DATA, &vec![0; d as usize]);
    mem.write(OUT, &vec![0xff; o as usize]);
    mem
}

fn run(mem: &mut GpuMem, groups: u32) -> gpu_core::error::GpuResult<()> {
    grid_barrier_on_cpu(&mut Launch {
        mem,
        cbuf0: CBUF,
        grid: [groups, 1, 1],
        block: [gb::BLOCK, 1, 1],
    })
}

fn results(mem: &GpuMem, groups: u32) -> (u32, Vec<u32>) {
    let out = (0..groups as u64)
        .map(|g| mem.read_u32(OUT + 4 * g).unwrap())
        .collect();
    (mem.read_u32(SYNC).unwrap(), out)
}

#[test]
fn t245_holds_at_48_and_96_groups() {
    for per_sm in [1, 2] {
        let g = gb::groups_for(&GA104, per_sm).unwrap();
        assert_eq!(g, 48 * per_sm);
        for rounds in [gb::ROUNDS_SHORT, gb::ROUNDS_LONG] {
            let mut mem = setup(g, rounds);
            run(&mut mem, g).unwrap();
            let (counter, out) = results(&mem, g);
            assert_eq!(counter, rounds * g);
            assert_eq!(gb::judge(g, rounds, counter, &out).unwrap(), Verdict::Holds);
            // the last round's half holds `rounds` in every word
            let half = (rounds as u64 & 1) * g as u64;
            for w in 0..g as u64 {
                assert_eq!(mem.read_u32(DATA + 4 * (half + w)).unwrap(), rounds);
            }
        }
    }
}

#[test]
fn t245_refusals_and_plants() {
    // more than 2 per SM, or none: refused before anything is built
    assert!(gb::groups_for(&GA104, 3).is_err());
    assert!(gb::groups_for(&GA104, 0).is_err());
    // counter not cleared before the launch: the model refuses (a real board would end at the wrong count)
    let mut mem = setup(48, 5);
    mem.write_u32(SYNC, 7);
    assert!(run(&mut mem, 48).is_err());
    // wrong block shape / a 2D grid: refused
    let mut mem = setup(48, 5);
    assert!(
        grid_barrier_on_cpu(&mut Launch {
            mem: &mut mem,
            cbuf0: CBUF,
            grid: [48, 1, 1],
            block: [64, 1, 1],
        })
        .is_err()
    );
    // data buffer of G words instead of 2·G: the model faults on the missing half (pages are 4 KiB in the model, so
    // G = 1024 makes the second half start on a page never written)
    let big = 1024;
    let mut small = GpuMem::default();
    small.write(CBUF, &gb::cbuf(SYNC, DATA, OUT, 3, big as u32).unwrap());
    small.write(SYNC, &[0; 4]);
    small.write(DATA, &vec![0; 4 * big]);
    small.write(OUT, &vec![0; 4 * big]);
    assert!(run(&mut small, big as u32).is_err());
    let mut mem = setup(48, 3);
    run(&mut mem, 48).unwrap();
    // planted results the board could return: a skipped barrier, stale words
    let (c, mut out) = results(&mem, 48);
    assert_eq!(gb::judge(48, 3, c, &out).unwrap(), Verdict::Holds);
    assert!(matches!(
        gb::judge(48, 3, c - 1, &out).unwrap(),
        Verdict::BadCounter {
            got: 143,
            want: 144
        }
    ));
    out[5] = 2;
    out[40] = 1;
    assert_eq!(
        gb::judge(48, 3, c, &out).unwrap(),
        Verdict::StaleWords {
            groups: 2,
            rounds: 3
        }
    );
}

#[test]
fn t245_qmd_and_cost() {
    let q = gb::qmd(0x1000, 16, CBUF, 96).unwrap();
    assert_eq!(q.len(), nvidia_rm570::QMD_SIZE);
    assert!(gb::qmd(0x1000, 16, CBUF, 0).is_err());
    // B5 arithmetic (DEDUS): 194 barriers of #513's token list at 1.5 µs = 291 µs, about the launch cost today (≤ 336)
    let one = gb::barrier_ns(20_000, 20_000 + 1_000 * 1_500).unwrap();
    assert_eq!(one, 1_500);
    assert_eq!(194 * one / 1_000, 291);
}

#[test]
fn t245_program_and_group_count() {
    // COMPILAT (Mesa NAK 25.0.7, session 237), NERULAT: 123 instructions of 16 bytes, 19 registers
    assert_eq!(gb::PROGRAM.len(), 1_968);
    assert!(gb::PROGRAM.len().is_multiple_of(16));
    assert_eq!(gb::REGISTERS, 19);
    // the program reads G from 0x0c; a constant buffer with 0 there is refused by the model (on the board: no wait
    // at the barrier and a division by 0 for the neighbour)
    let mut mem = setup(48, 3);
    mem.write(CBUF + gb::CBUF_GROUP_COUNT as u64, &[0; 4]);
    assert!(run(&mut mem, 48).is_err());
    // and a count that differs from the grid as well
    let mut mem = setup(48, 3);
    assert!(run(&mut mem, 96).is_err());
}

/// T259 (C-157 after os-459, `docs/C-157_OS459_BARIERA.md`): the board's numbers (MĂSURAT by the OS, 26 Nov 11:01)
/// through the driver's own arithmetic, and the wide run (G = 384 / 768) in SIMULARE with the SM count GR0 reported.
#[test]
fn t259_os459_numbers_and_wide_run() {
    // os-459, MĂSURAT: R = 1 and R = 1 001 at G = 48 and 96
    let b48 = gb::barrier_ns(3_968, 1_130_624).unwrap();
    let b96 = gb::barrier_ns(3_456, 1_146_592).unwrap();
    assert_eq!((b48, b96), (1_126, 1_143));
    // B3 (1–3 µs) and B4 (dearer at 96, under ×2) held
    assert!((1_000..=3_000).contains(&b48));
    assert!(b96 > b48 && b96 < 2 * b48);
    // the counters the OS printed are the ones `judge` wants
    for (g, c) in [(48, 48_048), (96, 96_096)] {
        assert_eq!(
            gb::judge(g, gb::ROUNDS_LONG, c, &vec![0; g as usize]).unwrap(),
            Verdict::Holds
        );
    }
    // §3, DEDUS: 88 launches per token today vs 87 barriers in one program
    let launches = 88 * nvidia_ga104::persistent_study::GEN_GAP_NS;
    assert_eq!(launches / 1_000, 340);
    assert_eq!(87 * b48 / 1_000, 97);
    // W6, DEDUS: at 2 µs the 87 barriers cost about half of today's launch starts
    assert_eq!(87 * 2_000 / 1_000, 174);

    // the wide run: G from the 48 SMs GR0 reported (os-459), never from `GrShape` (design maximum, refused)
    assert_eq!(gb::groups_for_wide_sms(48, 8).unwrap(), 384);
    assert_eq!(gb::groups_for_wide_sms(48, 16).unwrap(), 768);
    assert!(gb::groups_for_wide(&GA104, 8).is_err());
    for per_sm in [0, 1, 2, 4, 15, 17] {
        assert!(gb::groups_for_wide_sms(48, per_sm).is_err());
    }
    // SIMULARE: the same verdict path holds at both G (says nothing about 16 blocks per SM on the board)
    for g in [384, 768] {
        for rounds in [gb::ROUNDS_SHORT, gb::ROUNDS_LONG] {
            let mut mem = setup(g, rounds);
            run(&mut mem, g).unwrap();
            let (counter, out) = results(&mem, g);
            assert_eq!(counter, rounds * g);
            assert_eq!(gb::judge(g, rounds, counter, &out).unwrap(), Verdict::Holds);
        }
        assert_eq!(
            gb::qmd(0x1000, gb::REGISTERS, CBUF, g).unwrap().len(),
            nvidia_rm570::QMD_SIZE
        );
    }
}
