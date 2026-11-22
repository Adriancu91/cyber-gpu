//! C-160 point 7 (T176), SIMULARE: the probe of the training attention "on tiles" as the OS would send it to the
//! board — for every small set of `nvidia_ga104::attn_tile_probe::SETS`, one buffer (qkv, dO; the outputs and their
//! guards = NaN), the cbuf + QMD of each program from `launch_bytes`, the chain `attn_fwd_tile` → `attn_back_tile0` →
//! `attn_back_tile` through the command-processor model, the bytes of `read_ranges` read back and judged with the
//! margins written before the code (`docs/C-160_P7_ATENTIA_PROBA.md`, `608f377`). The "programs" run on the CPU
//! (`gpu_sim::cmdproc::attn_*_tile*_on_cpu` = the CPU models, whose equality with the GLSL is T173–T175); nothing here
//! proves anything about the RTX 3070 Ti. The timing set is too large for the stand-ins: its banded judgement is checked
//! on the model's output, and the banding itself on a medium set run whole.

use gpu_core::diag::Journal;
use gpu_sim::cmdproc::{
    CommandProcessor, attn_back_tile_on_cpu, attn_back_tile0_on_cpu, attn_fwd_tile_on_cpu,
};
use nvidia_ga104::attn_tile_probe::{
    self as ap, FILL, GUARD, PROGS, Probe, ProbeSet, Prog, SETS, TIMING_SET, Verdict,
    parse_attn_tile,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

const PROG: [u64; 3] = [0x10_0000, 0x14_0000, 0x18_0000];
const CBUF: u64 = 0x20_0000;
const QMD: u64 = 0x21_0000;
const SEM: u64 = 0x30_0000;
const PUSH: u64 = 0x31_0000;
const BUF: u64 = 0x40_0000;

fn cp() -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program(Prog::Fwd.binary(), attn_fwd_tile_on_cpu);
    cp.register_program(Prog::Rows.binary(), attn_back_tile0_on_cpu);
    cp.register_program(Prog::Cols.binary(), attn_back_tile_on_cpu);
    for (k, p) in PROGS.iter().enumerate() {
        cp.mem.write(PROG[k], p.binary());
    }
    cp
}

/// One launch of program `k` (its cbuf + QMD already written at slot k), then a semaphore release of `9 + k`.
fn submit(cp: &mut CommandProcessor, k: usize) -> Result<(), gpu_core::error::GpuError> {
    let qmd = QMD + 0x1000 * k as u64;
    let mut words = [0u32; 64];
    let mut p = PushBuf::new(&mut words);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(qmd >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let exec = h::NVC56F_SEM_EXECUTE_OPERATION_RELEASE;
    p.inc(
        0,
        h::NVC56F_SEM_ADDR_LO,
        &[SEM as u32 + 8, 0, 9 + k as u32, 0, exec],
    )
    .unwrap();
    let n = p.len();
    let bytes: Vec<u8> = words[..n].iter().flat_map(|w| w.to_le_bytes()).collect();
    let at = PUSH + 0x1000 * k as u64;
    cp.mem.write(at, &bytes);
    let e = push::gpfifo_entry(at, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
}

/// A mistake planted in what is sent or in memory between launches.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Plant {
    None,
    /// Every element j ≤ i of P row 0 of head 0, text 0, row T − 1 multiplied by (1 + 2⁻¹⁴) after the forward launch.
    PRowScaled,
    /// One dS element (row T − 1, j = 0) moved by one unit in the last place after both backward launches.
    DsUlp,
    /// One word after dS written (a write past the end).
    GuardHit,
    /// The mode-1 program never launched (dqkv stays NaN).
    NoCols,
    /// The forward grid one block short in x (the last rows of P / att stay NaN).
    FwdGridShort,
}

fn run(s: &ProbeSet, plant: Plant) -> Vec<u8> {
    let mut cp = cp();
    cp.mem.write(BUF, &s.buffer());
    cp.mem.write_u32(SEM + 8, 0);
    for (k, prog) in PROGS.iter().enumerate() {
        let (cb, mut q) =
            ap::launch_bytes(s, *prog, PROG[k], CBUF + 0x1000 * k as u64, BUF).unwrap();
        if plant == Plant::FwdGridShort && *prog == Prog::Fwd {
            // Rebuild the QMD with a grid one block short in x.
            let a = s.fwd_args(BUF);
            let mut g = nvidia_ga104::attn_fwd_tile::grid(&a);
            g[0] -= 1;
            q = nvidia_ga104::qmd::ComputeQmd {
                program_va: PROG[k],
                registers: prog.registers(),
                cbuf0_va: CBUF + 0x1000 * k as u64,
                cbuf0_size: ap::CBUF_SIZE as u32,
                grid: g,
                block: [256, 1, 1],
                release: None,
            }
            .to_bytes_with_shared(&prog.shared())
            .unwrap();
        }
        cp.mem.write(CBUF + 0x1000 * k as u64, &cb);
        cp.mem.write(QMD + 0x1000 * k as u64, &q);
    }
    let l = s.layout();
    let (t, hh) = (s.t as usize, s.heads as usize);
    for (k, prog) in PROGS.iter().enumerate() {
        if plant == Plant::NoCols && *prog == Prog::Cols {
            continue;
        }
        submit(&mut cp, k).unwrap_or_else(|e| panic!("{} {}: {e:?}", s.name, prog.name()));
        assert_eq!(cp.mem.read_u32(SEM + 8).unwrap(), 9 + k as u32);
        if plant == Plant::PRowScaled && *prog == Prog::Fwd {
            let i = t - 1;
            let at = BUF + l.p as u64 + 4 * (i * t) as u64;
            let mut row = cp.mem.read_f32s(at, i + 1).unwrap();
            for v in &mut row {
                *v *= 1.0 + 1.0 / 16384.0;
            }
            cp.mem.write_f32s(at, &row);
        }
    }
    let _ = hh;
    match plant {
        Plant::DsUlp => {
            let at = BUF + l.ds as u64 + 4 * ((t - 1) * t) as u64;
            let w = cp.mem.read_u32(at).unwrap();
            cp.mem.write_u32(at, w + 1);
        }
        Plant::GuardHit => cp.mem.write_u32(BUF + l.ds_guard as u64 + 40, 0),
        _ => {}
    }
    let mut all = vec![0u8; l.bytes];
    cp.mem.read(BUF, &mut all).unwrap();
    assert_eq!(
        &all[..l.p],
        &s.buffer()[..l.p],
        "{}: inputs changed",
        s.name
    );
    s.read_ranges()
        .iter()
        .flat_map(|&(o, n)| all[o..o + n].to_vec())
        .collect()
}

#[test]
fn t176a_the_small_sets_pass_and_the_backward_is_bit_for_bit() {
    let mut p = Probe::default();
    for (i, s) in SETS[..TIMING_SET].iter().enumerate() {
        let out = run(s, Plant::None);
        let ns = [7000 + i as u64];
        p.add(ap::judge_set(i, &out, [&ns, &ns, &ns]).unwrap());
    }
    for l in p.lines("SIMULARE") {
        println!("{l}");
    }
    assert!(p.passes(), "{:#?}", p.lines("SIMULARE"));
    assert!(p.timing_skipped());
    // The stand-ins ARE the models: all exact here (prediction 2 for the backward; on the board the forward is the
    // 20-80 % prediction because of the MUFU units).
    for (k, (ex, el, w)) in p.per_prog().iter().enumerate() {
        assert_eq!(ex, el, "{}", PROGS[k].name());
        assert_eq!(*w, 0);
    }
    // Elements: P and dS count j ≤ i only.
    let tri = |s: &ProbeSet| (s.batch * s.heads) as usize * (s.t as usize * (s.t as usize + 1) / 2);
    let pp = p.per_prog();
    let want_fwd: usize = SETS[..TIMING_SET]
        .iter()
        .map(|s| tri(s) + (s.batch * s.t * s.c) as usize)
        .sum();
    assert_eq!(pp[0].1 as usize, want_fwd);
    assert_eq!(
        pp[1].1 as usize,
        SETS[..TIMING_SET].iter().map(tri).sum::<usize>()
    );
    assert!(p.sets.iter().all(|r| r.progs.iter().all(|q| q.guards_ok)));
}

#[test]
fn t176b_planted_mistakes_are_caught_and_each_program_is_judged_alone() {
    let s = &SETS[1]; // t130-d64: two tiles, the last one partial
    // P scaled by 2⁻¹⁴ after the forward: the forward is outside its MUFU margin, but the backward programs, judged on
    // the board's own P, stay bit for bit — the isolation the plan asks for.
    let r = ap::judge_set(1, &run(s, Plant::PRowScaled), [&[], &[], &[]]).unwrap();
    assert_eq!(r.progs[0].verdict, Verdict::NuTrece, "{r:?}");
    assert!(r.progs[1].passes() && r.progs[1].exact == r.progs[1].elements);
    assert!(r.progs[2].passes() && r.progs[2].exact == r.progs[2].elements);
    // One unit in the last place of dS: still in the safety margin, but no longer all exact.
    let r = ap::judge_set(1, &run(s, Plant::DsUlp), [&[], &[], &[]]).unwrap();
    assert!(r.progs[1].passes(), "{r:?}");
    assert_eq!(r.progs[1].exact + 1, r.progs[1].elements);
    assert!(r.progs[1].worst_permille.unwrap() > 0);
    // A guard word written: invalid.
    let r = ap::judge_set(1, &run(s, Plant::GuardHit), [&[], &[], &[]]).unwrap();
    assert_eq!(r.progs[1].verdict, Verdict::Invalid);
    assert!(!r.progs[1].guards_ok && r.progs[0].guards_ok && r.progs[2].guards_ok);
    // Mode 1 never launched: dqkv NaN → invalid; the other two pass.
    let r = ap::judge_set(1, &run(s, Plant::NoCols), [&[], &[], &[]]).unwrap();
    assert_eq!(r.progs[2].verdict, Verdict::Invalid);
    assert!(r.progs[0].passes() && r.progs[1].passes());
    // The forward grid one block short: rows 64..129 of P / att are NaN → invalid.
    let r = ap::judge_set(1, &run(s, Plant::FwdGridShort), [&[], &[], &[]]).unwrap();
    assert_eq!(r.progs[0].verdict, Verdict::Invalid);
    // A set missing from the probe: the probe does not pass.
    let mut p = Probe::default();
    for i in [0, 1, 2, 4] {
        p.add(ap::judge_set(i, &SETS[i].model_out(), [&[], &[], &[]]).unwrap());
    }
    assert!(!p.passes());
}

#[test]
fn t176c_bands_are_what_the_whole_launch_wrote() {
    // A medium banded set run whole through the stand-ins; the banded read must be exactly the model's banded output
    // and pass — so judging the timing set on bands judges what the whole launches wrote there.
    static TEXTS: [usize; 2] = [0, 2];
    static ROWS: [usize; 3] = [0, 40, 95];
    let s = ProbeSet {
        name: "benzi-b3-t96",
        batch: 3,
        t: 96,
        c: 128,
        heads: 2,
        seed: 77,
        amp: 2.0,
        bands: Some((&TEXTS, &ROWS)),
        repeats: 1,
    };
    let whole = ProbeSet { bands: None, ..s };
    assert!(s.shape_ok() && whole.shape_ok());
    let mut cp = cp();
    cp.mem.write(BUF, &s.buffer());
    for (k, prog) in PROGS.iter().enumerate() {
        let (cb, q) = ap::launch_bytes(&s, *prog, PROG[k], CBUF + 0x1000 * k as u64, BUF).unwrap();
        cp.mem.write(CBUF + 0x1000 * k as u64, &cb);
        cp.mem.write(QMD + 0x1000 * k as u64, &q);
        submit(&mut cp, k).unwrap();
    }
    let l = s.layout();
    let mut all = vec![0u8; l.bytes];
    cp.mem.read(BUF, &mut all).unwrap();
    let out: Vec<u8> = s
        .read_ranges()
        .iter()
        .flat_map(|&(o, n)| all[o..o + n].to_vec())
        .collect();
    assert_eq!(
        out,
        s.model_out(),
        "banded read = the model's banded output"
    );
    let r = ap::judge(&s, usize::MAX, &out, [&[], &[], &[]]).unwrap();
    assert!(r.passes(), "{r:?}");
    assert!(r.progs.iter().all(|p| p.exact == p.elements));
    // The bands were all that was filled: the rest of P was written by the launch over zeros, not over NaN.
    let p_full = &all[l.p..l.p_guard];
    assert!(
        p_full
            .as_chunks::<4>()
            .0
            .iter()
            .all(|w| u32::from_le_bytes(*w) != FILL)
    );
}

#[test]
fn t176d_the_timing_set_judges_its_bands_and_the_rows_round_trip() {
    let s = &SETS[TIMING_SET];
    assert_eq!(s.repeats, 5);
    let ns_f = [530_000, 520_000, 540_000, 600_000, 510_000];
    let ns_0 = [490_000; 5];
    let ns_1 = [600_000, 610_000, 590_000, 605_000, 595_000];
    let out = s.model_out();
    assert_eq!(out.len(), s.read_len());
    let r = ap::judge_set(TIMING_SET, &out, [&ns_f, &ns_0, &ns_1]).unwrap();
    assert!(r.passes(), "{r:?}");
    let mut p = Probe::default();
    for (i, si) in SETS[..TIMING_SET].iter().enumerate() {
        p.add(ap::judge_set(i, &si.model_out(), [&[], &[], &[]]).unwrap());
    }
    p.add(r);
    assert!(p.passes() && !p.timing_skipped());
    let lines = p.lines("SIMULARE");
    for l in &lines {
        println!("{l}");
    }
    assert!(
        lines
            .iter()
            .any(|l| l.contains("mediana 530000 ns pe 5 lansari"))
    );
    // The journal rows and their reader.
    let mut j = Journal::new(4096);
    p.journal(&mut j);
    let text: String = p
        .rows()
        .iter()
        .map(|(w, v)| format!("gpu: valoare {w} = 0x{v:x}\n"))
        .collect();
    let q = parse_attn_tile(&text);
    assert_eq!(q.sets, Some(6));
    assert_eq!(q.in_margin, Some(6));
    assert_eq!(q.timing_skipped, Some(0));
    assert_eq!(q.timing_ns, [Some(530_000), Some(490_000), Some(600_000)]);
    let pp = p.per_prog();
    for (got, want) in q.per_prog.iter().zip(pp) {
        assert_eq!(*got, [Some(want.0), Some(want.1), Some(want.2)]);
    }
    assert_eq!(q.per_set.len(), 6);
    assert_eq!(q.per_set[5][0], TIMING_SET as u32);
    assert_eq!(q.per_set[5][1], s.inputs_fnv());
    assert_eq!(q.per_set[5][2], 1); // fwd verdict
    assert_eq!(q.per_set[5][7], 1); // back0 verdict
    assert_eq!(q.per_set[5][12], 1); // back1 verdict
    // A second run in the same journal starts over.
    let twice = format!("{text}{text}");
    assert_eq!(parse_attn_tile(&twice), q);
    // GUARD words per output, four outputs.
    assert_eq!(GUARD * 4 * 4, 4096);
}
