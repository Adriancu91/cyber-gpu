//! C-157 point 2 after os-375 (T197), SIMULARE: the probe "old beside new" with the **fourth** generation of the tiled
//! training attention — `attn_fwd_tile3` → `attn_back_tile0_3` → `attn_back_tile_4` (only the last program is new) —
//! on every small set of `nvidia_ga104::attn_tile_probe::SETS`, after the first and the third generation on the same
//! buffer refilled, all through the command-processor model with the cbuf + QMD of `launch_bytes` / `launch_bytes3` /
//! `launch_bytes4`; the fourth read-back judged with the old margins and compared word for word with the first (plan
//! and predictions written before the code: `docs/C-157_P2_ATTN_TILE4.md`, session 189, commit `50e8e1f`).
//!
//! The "programs" run on the CPU: every generation is registered with the **same** CPU models (the equality of
//! `attn_back_tile_4`'s GLSL with its model is T196), so here the chains are equal by construction — this test checks
//! the path (QMD, shared memory, buffers, comparison, rows), not the board. Nothing here proves anything about the
//! RTX 3070 Ti.

use gpu_core::diag::Journal;
use gpu_sim::cmdproc::{
    CommandProcessor, Needs, ProgramFn, attn_back_tile_on_cpu, attn_back_tile0_on_cpu,
    attn_fwd_tile_on_cpu,
};
use nvidia_ga104::attn_tile_probe::{
    self as ap, OS375_NS, PROGS, PROGS2, PROGS3, PROGS4, Probe, Probe2, Probe3, Probe4, ProbeSet,
    Prog, Prog4, SETS, TIMING_SET, parse_attn_tile, parse_attn_tile2, parse_attn_tile3,
    parse_attn_tile4,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

/// Program addresses: slot 3 · i + k for generation index i (0: gen 1, 1: gen 3, 2: gen 4) and role k.
const PROG: [u64; 9] = [
    0x10_0000, 0x14_0000, 0x18_0000, 0x1c_0000, 0x50_0000, 0x54_0000, 0x58_0000, 0x5c_0000,
    0x40_0000,
];
const CBUF: u64 = 0x20_0000;
const QMD: u64 = 0x21_0000;
const SEM: u64 = 0x30_0000;
const PUSH: u64 = 0x31_0000;
const BUF: u64 = 0x60_0000;

const RUNS: [ProgramFn; 3] = [
    attn_fwd_tile_on_cpu,
    attn_back_tile0_on_cpu,
    attn_back_tile_on_cpu,
];

fn cp() -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    for k in 0..3 {
        let progs = [
            (PROGS[k].binary(), PROGS[k].shared(), true),
            (PROGS3[k].binary(), PROGS3[k].shared(), true),
            // fwd / back0 of the fourth generation are the third's binaries (registered once).
            (PROGS4[k].binary(), PROGS4[k].shared(), PROGS4[k].is_new()),
        ];
        for (i, (code, sh, register)) in progs.into_iter().enumerate() {
            if register {
                cp.register_program_with_needs(
                    code,
                    RUNS[k],
                    Needs {
                        shared_bytes: sh.bytes,
                        barriers: u32::from(sh.barriers),
                    },
                );
            }
            cp.mem.write(PROG[3 * i + k], code);
        }
    }
    cp
}

/// One launch of slot `k` (its cbuf + QMD already written), then a semaphore release of `9 + k`.
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

#[derive(Clone, Copy, PartialEq, Debug)]
enum Plant {
    None,
    /// `attn_back_tile_4`'s QMD declares the first generation's shared memory (32 768 < 33 024).
    Back4OldShared,
    /// One dK word of the fourth chain moved by one unit in the last place after its launches.
    Dqkv4Ulp,
}

/// Runs generation `g` (1, 3 or 4) on `s` in `cp`, on the buffer refilled; returns the read-back.
fn chain(
    cp: &mut CommandProcessor,
    s: &ProbeSet,
    g: usize,
    plant: Plant,
) -> Result<Vec<u8>, gpu_core::error::GpuError> {
    cp.mem.write(BUF, &s.buffer());
    cp.mem.write_u32(SEM + 8, 0);
    let i = match g {
        1 => 0,
        3 => 1,
        _ => 2,
    };
    for k in 0..3 {
        let slot = 3 * i + k;
        let cbuf = CBUF + 0x1000 * slot as u64;
        let (cb, mut q) = match g {
            1 => ap::launch_bytes(s, PROGS[k], PROG[slot], cbuf, BUF).unwrap(),
            3 => ap::launch_bytes3(s, PROGS3[k], PROG[slot], cbuf, BUF).unwrap(),
            _ => ap::launch_bytes4(s, PROGS4[k], PROG[slot], cbuf, BUF).unwrap(),
        };
        if g == 4 && k == 2 && plant == Plant::Back4OldShared {
            let a = s.back_args(BUF, nvidia_ga104::attn_back::Mode::Columns);
            q = nvidia_ga104::qmd::ComputeQmd {
                program_va: PROG[slot],
                registers: PROGS4[2].registers(),
                cbuf0_va: cbuf,
                cbuf0_size: ap::CBUF_SIZE as u32,
                grid: nvidia_ga104::attn_back_tile_4::grid(&a),
                block: [256, 1, 1],
                release: None,
            }
            .to_bytes_with_shared(&Prog::Cols.shared())
            .unwrap();
        }
        cp.mem.write(cbuf, &cb);
        cp.mem.write(QMD + 0x1000 * slot as u64, &q);
        submit(cp, slot)?;
        assert_eq!(cp.mem.read_u32(SEM + 8).unwrap(), 9 + slot as u32);
    }
    let l = s.layout();
    if g == 4 && plant == Plant::Dqkv4Ulp {
        let at = BUF + l.dqkv as u64 + 4 * 7;
        let w = cp.mem.read_u32(at).unwrap();
        cp.mem.write_u32(at, w + 1);
    }
    let mut all = vec![0u8; l.bytes];
    cp.mem.read(BUF, &mut all).unwrap();
    assert_eq!(
        &all[..l.p],
        &s.buffer()[..l.p],
        "{}: inputs changed",
        s.name
    );
    Ok(s.read_ranges()
        .iter()
        .flat_map(|&(o, n)| all[o..o + n].to_vec())
        .collect())
}

#[test]
fn t197a_the_fourth_programs_are_the_ones_the_notes_name() {
    for (k, n) in PROGS4.iter().enumerate() {
        assert_eq!(n.old(), PROGS[k]);
        assert_eq!(n.prev(), PROGS3[k]);
        let hex: String = nvidia_ga104::sha256::digest(n.binary())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hex, n.sha256(), "{}", n.name());
        assert_ne!(n.binary(), PROGS[k].binary());
        assert_ne!(n.binary(), PROGS2[k].binary());
        // Only the last program is new; fwd / back0 are the third generation's files.
        assert_eq!(n.binary() == PROGS3[k].binary(), !n.is_new());
    }
    assert_eq!(PROGS4.map(Prog4::is_new), [false, false, true]);
    assert_eq!(PROGS4.map(Prog4::registers), [120, 68, 80]);
    assert_eq!(PROGS4.map(|p| p.shared().bytes), [37_888, 33_024, 33_024]);
    assert_eq!(
        PROGS4.map(Prog4::name),
        ["attn_fwd_tile3", "attn_back_tile0_3", "attn_back_tile_4"]
    );
    assert!(PROGS4[2].sha256().starts_with("74f2cb81"));
    // The first three lists are unchanged (the OS reads them).
    assert_eq!(PROGS.map(Prog::registers), [55, 50, 73]);
    assert_eq!(PROGS2.map(|p| p.registers()), [117, 52, 79]);
    assert_eq!(PROGS3.map(|p| p.registers()), [120, 68, 80]);
    assert!(PROGS3[2].sha256().starts_with("67c8d9a5"));
    // The same cbuf as the old programs on every set; fwd / back0: the very QMD of the third generation; back1: the
    // same QMD bytes too (same registers, shared memory, grid) — only the program behind `prog_va` differs.
    for s in &SETS {
        for k in 0..3 {
            let (cb_o, q_o) = ap::launch_bytes(s, PROGS[k], 0x1000, 0x2000, 0x10_000).unwrap();
            let (cb_3, q_3) = ap::launch_bytes3(s, PROGS3[k], 0x1000, 0x2000, 0x10_000).unwrap();
            let (cb_4, q_4) = ap::launch_bytes4(s, PROGS4[k], 0x1000, 0x2000, 0x10_000).unwrap();
            assert_eq!(cb_o, cb_4, "{} {}", s.name, PROGS4[k].name());
            assert_eq!(cb_3, cb_4);
            assert_ne!(q_o, q_4, "{} {}", s.name, PROGS4[k].name());
            assert_eq!(q_3, q_4, "{} {}", s.name, PROGS4[k].name());
        }
    }
    assert!(ap::launch_bytes4(&SETS[0], PROGS4[2], 0x1000, 0x2000, 0x10_200).is_err());
}

#[test]
fn t197b_the_fourth_generation_on_the_small_sets() {
    let mut cp = cp();
    let (mut old, mut p3, mut p4) = (Probe::default(), Probe3::default(), Probe4::default());
    for (i, s) in SETS[..TIMING_SET].iter().enumerate() {
        let o = chain(&mut cp, s, 1, Plant::None).unwrap();
        let n3 = chain(&mut cp, s, 3, Plant::None).unwrap();
        let n4 = chain(&mut cp, s, 4, Plant::None).unwrap();
        old.add(ap::judge_set(i, &o, [&[], &[], &[]]).unwrap());
        p3.add(ap::judge_pair_set(i, &o, &n3, [&[], &[], &[]]).unwrap());
        let r = ap::judge_pair_set(i, &o, &n4, [&[], &[], &[]]).unwrap();
        assert!(r.identical(), "{}: {:?}", s.name, r.same);
        assert_eq!(n3, n4, "{}", s.name);
        p4.add(r);
    }
    for l in p4.lines("SIMULARE", Some(&p3)) {
        println!("{l}");
    }
    assert!(old.passes() && p3.passes() && p4.passes() && p4.identical());
    assert!(p4.timing_skipped());
    assert_eq!(p4.identical_sets(), TIMING_SET);
    assert_eq!(p4.in_margin(), TIMING_SET);
    assert_eq!(p4.per_prog(), old.per_prog());
    assert_eq!(p4.same_per_prog(), p3.same_per_prog());
    for (k, (ex, el, w)) in p4.per_prog().iter().enumerate() {
        assert_eq!(ex, el, "{}", PROGS4[k].name());
        assert_eq!(*w, 0);
    }
}

#[test]
fn t197c_planted_mistakes_are_caught() {
    let s = &SETS[1]; // t130-d64: two tiles, the last one partial
    let mut cp = cp();
    // attn_back_tile_4 with the first generation's shared memory in its QMD: refused at that launch.
    assert!(chain(&mut cp, s, 4, Plant::Back4OldShared).is_err());
    // One ulp of dqkv in the fourth chain: still in the margin, but no longer identical (exactly one word, back1).
    let o = chain(&mut cp, s, 1, Plant::None).unwrap();
    let n = chain(&mut cp, s, 4, Plant::Dqkv4Ulp).unwrap();
    let r = ap::judge_pair_set(1, &o, &n, [&[], &[], &[]]).unwrap();
    assert!(r.new.passes(), "{r:?}");
    assert!(!r.identical());
    assert_eq!(r.same[2].0 + 1, r.same[2].1);
    assert_eq!(r.same[0].0, r.same[0].1);
    assert_eq!(r.same[1].0, r.same[1].1);
    let mut p = Probe4::default();
    p.add(r);
    assert_eq!(p.identical_sets(), 0);
    assert!(!p.identical());
    // A short read-back on the new side: invalid and nothing counted identical.
    let r = ap::judge_pair_set(1, &o, &n[..n.len() - 4], [&[], &[], &[]]).unwrap();
    assert!(!r.new.passes());
    assert!(r.same.iter().all(|&(a, b)| a == 0 && b > 0));
    // A set missing: the probe does not pass.
    let mut p = Probe4::default();
    for i in [0, 1, 3, 4] {
        let m = SETS[i].model_out();
        p.add(ap::judge_pair_set(i, &m, &m, [&[], &[], &[]]).unwrap());
    }
    assert!(!p.passes() && p.identical());
    assert!(!Probe4::default().identical());
}

#[test]
fn t197d_the_timing_set_rows_and_the_ratio_to_the_third_generation() {
    let s = &SETS[TIMING_SET];
    let out = s.model_out();
    let (t3_f, t3_0, t3_1) = ([1_000_000u64; 5], [830_000u64; 5], [1_597_000u64; 5]);
    let (t4_f, t4_0, t4_1) = (
        [1_010_000u64, 990_000, 1_000_000, 1_005_000, 995_000],
        [830_000u64; 5],
        [1_280_000u64; 5],
    );
    let mut old = Probe::default();
    let mut p2 = Probe2::default();
    let mut p3 = Probe3::default();
    let mut p4 = Probe4::default();
    for (i, si) in SETS[..TIMING_SET].iter().enumerate() {
        let m = si.model_out();
        old.add(ap::judge_set(i, &m, [&[], &[], &[]]).unwrap());
        p2.add(ap::judge_pair_set(i, &m, &m, [&[], &[], &[]]).unwrap());
        p3.add(ap::judge_pair_set(i, &m, &m, [&[], &[], &[]]).unwrap());
        p4.add(ap::judge_pair_set(i, &m, &m, [&[], &[], &[]]).unwrap());
    }
    old.add(
        ap::judge_set(
            TIMING_SET,
            &out,
            [&[2_490_000u64; 5], &[1_810_000; 5], &[3_990_000; 5]],
        )
        .unwrap(),
    );
    p2.add(
        ap::judge_pair_set(
            TIMING_SET,
            &out,
            &out,
            [&[1_530_000u64; 5], &[1_540_000; 5], &[2_250_000; 5]],
        )
        .unwrap(),
    );
    p3.add(ap::judge_pair_set(TIMING_SET, &out, &out, [&t3_f, &t3_0, &t3_1]).unwrap());
    p4.add(ap::judge_pair_set(TIMING_SET, &out, &out, [&t4_f, &t4_0, &t4_1]).unwrap());
    assert!(p4.passes() && p4.identical() && !p4.timing_skipped());
    let lines = p4.lines("SIMULARE", Some(&p3));
    for l in &lines {
        println!("{l}");
    }
    assert!(lines[0].starts_with(
        "atentie4 vechi langa nou (SIMULARE): 6 / 6 seturi in margine, 6 / 6 seturi identice"
    ));
    assert!(lines.iter().any(|l| {
        l.contains(
            "attn_back_tile_4 (SIMULARE): mediana 1280000 ns pe 5 lansari (estimat ~1,2-1,35 ms)",
        ) && l.contains("attn_back_tile_3 1597000 ns (aceeasi rulare), de 1,24 ori mai rapid")
    }));
    assert!(
        lines
            .iter()
            .any(|l| l.contains("attn_fwd_tile3 (SIMULARE): mediana 1000000")
                && l.contains("attn_fwd_tile3 1000000 ns (aceeasi rulare), de 1,00 ori"))
    );
    assert!(
        lines
            .iter()
            .any(|l| l.contains("cuvinte identice cu attn_back_tile")
                && l.starts_with("atentie4 attn_back_tile_4"))
    );
    // Without the third probe: the os-375 medians.
    let alone = p4.lines("SIMULARE", None);
    assert!(
        alone
            .iter()
            .any(|l| l.contains(&format!("attn_back_tile_3 {} ns (os-375)", OS375_NS[2])))
    );
    // The rows, their reader, and none of the four readers sees the others' rows.
    let mut j = Journal::new(4096);
    p4.journal(&mut j);
    let text = |rows: Vec<(&'static str, u32)>| -> String {
        rows.iter()
            .map(|(w, v)| format!("gpu: valoare {w} = 0x{v:x}\n"))
            .collect()
    };
    let (text1, text2, text3, text4) = (
        text(old.rows()),
        text(p2.rows()),
        text(p3.rows()),
        text(p4.rows()),
    );
    let all = format!("{text1}{text2}{text3}{text4}");
    assert_eq!(parse_attn_tile(&all), parse_attn_tile(&text1));
    assert_eq!(parse_attn_tile2(&all), parse_attn_tile2(&text2));
    assert_eq!(parse_attn_tile3(&all), parse_attn_tile3(&text3));
    let q = parse_attn_tile4(&all);
    assert_eq!(q, parse_attn_tile4(&text4));
    assert_eq!(
        parse_attn_tile4(&format!("{text1}{text2}{text3}")),
        Default::default()
    );
    assert_eq!(q.sets, Some(6));
    assert_eq!(q.in_margin, Some(6));
    assert_eq!(q.identical_sets, Some(6));
    assert_eq!(q.timing_skipped, Some(0));
    assert_eq!(
        q.timing_ns,
        [Some(1_000_000), Some(830_000), Some(1_280_000)]
    );
    let (pp, sp) = (p4.per_prog(), p4.same_per_prog());
    for k in 0..3 {
        assert_eq!(
            q.per_prog[k],
            [
                Some(pp[k].0),
                Some(pp[k].1),
                Some(pp[k].2),
                Some(sp[k].0),
                Some(sp[k].1)
            ]
        );
    }
    assert_eq!(q.per_set.len(), 6);
    let last = q.per_set[5];
    assert_eq!(last[0], TIMING_SET as u32);
    assert_eq!(last[1], s.inputs_fnv());
    assert_eq!([last[2], last[8], last[14]], [1, 1, 1]);
    // The same values as the third probe's rows, under the other prefix.
    let r3 = p3.rows();
    let r4 = p4.rows();
    assert_eq!(r3.len(), r4.len());
    for ((w3, v3), (w4, v4)) in r3.iter().zip(&r4) {
        assert_eq!(w3.replacen("atentie3", "atentie4", 1), *w4);
        if !w3.contains("timp") {
            assert_eq!(v3, v4, "{w3}");
        }
    }
    // A second run in the same journal starts over.
    assert_eq!(parse_attn_tile4(&format!("{text4}{text4}")), q);
}
