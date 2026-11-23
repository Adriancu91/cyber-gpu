//! C-157 point 2 after os-364 (T191), SIMULARE: the probe "old beside new" with the **third** generation of the tiled
//! training attention — on every small set of `nvidia_ga104::attn_tile_probe::SETS`, the first-generation chain, the
//! second (`attn_*_2`) and then the third (`attn_fwd_tile3` → `attn_back_tile0_3` → `attn_back_tile_3`), each on the
//! same buffer refilled, all through the command-processor model with the cbuf + QMD of `launch_bytes` /
//! `launch_bytes2` / `launch_bytes3`; the third read-back judged with the old margins and compared word for word with
//! the first (plan and predictions written before the code: `docs/C-157_P2_ATTN_TILE3.md`, session 182, commit
//! `b6e3b5b`).
//!
//! The "programs" run on the CPU: all three generations are registered with the **same** CPU models (whose equality
//! with each GLSL is T173–T175, T181–T183 and T188–T190), so here the chains are equal by construction — this test
//! checks the path (QMD, shared memory, buffers, comparison, rows), not the board. Nothing here proves anything about
//! the RTX 3070 Ti.

use gpu_core::diag::Journal;
use gpu_sim::cmdproc::{
    CommandProcessor, Needs, ProgramFn, attn_back_tile_on_cpu, attn_back_tile0_on_cpu,
    attn_fwd_tile_on_cpu,
};
use nvidia_ga104::attn_tile_probe::{
    self as ap, OS364_NS, PROGS, PROGS2, PROGS3, Probe, Probe2, Probe3, ProbeSet, Prog, Prog3,
    SETS, TIMING_SET, parse_attn_tile, parse_attn_tile2, parse_attn_tile3,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

/// Program addresses: generation g (1–3) of role k at slot 3 · (g − 1) + k.
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
            (PROGS[k].binary(), PROGS[k].shared()),
            (PROGS2[k].binary(), PROGS2[k].shared()),
            (PROGS3[k].binary(), PROGS3[k].shared()),
        ];
        for (g, (code, sh)) in progs.into_iter().enumerate() {
            cp.register_program_with_needs(
                code,
                RUNS[k],
                Needs {
                    shared_bytes: sh.bytes,
                    barriers: u32::from(sh.barriers),
                },
            );
            cp.mem.write(PROG[3 * g + k], code);
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
    /// The third forward's QMD declares the first generation's shared memory (37 376 < 37 888).
    Fwd3OldShared,
    /// One dqkv word of the third chain moved by one unit in the last place after its launches.
    Dqkv3Ulp,
}

/// Runs generation `g` (1–3) on `s` in `cp`, on the buffer refilled; returns the read-back.
fn chain(
    cp: &mut CommandProcessor,
    s: &ProbeSet,
    g: usize,
    plant: Plant,
) -> Result<Vec<u8>, gpu_core::error::GpuError> {
    cp.mem.write(BUF, &s.buffer());
    cp.mem.write_u32(SEM + 8, 0);
    for k in 0..3 {
        let slot = 3 * (g - 1) + k;
        let cbuf = CBUF + 0x1000 * slot as u64;
        let (cb, mut q) = match g {
            1 => ap::launch_bytes(s, PROGS[k], PROG[slot], cbuf, BUF).unwrap(),
            2 => ap::launch_bytes2(s, PROGS2[k], PROG[slot], cbuf, BUF).unwrap(),
            _ => ap::launch_bytes3(s, PROGS3[k], PROG[slot], cbuf, BUF).unwrap(),
        };
        if g == 3 && k == 0 && plant == Plant::Fwd3OldShared {
            let a = s.fwd_args(BUF);
            q = nvidia_ga104::qmd::ComputeQmd {
                program_va: PROG[slot],
                registers: PROGS3[0].registers(),
                cbuf0_va: cbuf,
                cbuf0_size: ap::CBUF_SIZE as u32,
                grid: nvidia_ga104::attn_fwd_tile3::grid(&a),
                block: [256, 1, 1],
                release: None,
            }
            .to_bytes_with_shared(&Prog::Fwd.shared())
            .unwrap();
        }
        cp.mem.write(cbuf, &cb);
        cp.mem.write(QMD + 0x1000 * slot as u64, &q);
        submit(cp, slot)?;
        assert_eq!(cp.mem.read_u32(SEM + 8).unwrap(), 9 + slot as u32);
    }
    let l = s.layout();
    if g == 3 && plant == Plant::Dqkv3Ulp {
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
fn t191a_the_third_programs_are_the_ones_the_notes_name() {
    for (k, n) in PROGS3.iter().enumerate() {
        assert_eq!(n.old(), PROGS[k]);
        assert_eq!(n.prev(), PROGS2[k]);
        let hex: String = nvidia_ga104::sha256::digest(n.binary())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hex, n.sha256(), "{}", n.name());
        assert_ne!(n.binary(), PROGS[k].binary());
        assert_ne!(n.binary(), PROGS2[k].binary());
    }
    assert_eq!(PROGS3.map(Prog3::registers), [120, 68, 80]);
    assert_eq!(PROGS3.map(|p| p.shared().bytes), [37_888, 33_024, 33_024]);
    assert_eq!(
        PROGS3.map(Prog3::name),
        ["attn_fwd_tile3", "attn_back_tile0_3", "attn_back_tile_3"]
    );
    assert!(PROGS3[0].sha256().starts_with("4127e5fa"));
    assert!(PROGS3[1].sha256().starts_with("d8f4ec40"));
    assert!(PROGS3[2].sha256().starts_with("67c8d9a5"));
    // The first two lists are unchanged (the OS reads them).
    assert_eq!(PROGS.map(Prog::registers), [55, 50, 73]);
    assert_eq!(PROGS2.map(|p| p.registers()), [117, 52, 79]);
    // The same cbuf as the old programs on every set; the QMD differs from the second generation's only by the
    // registers (same shared memory, same grid): a QMD with `_2`'s registers would not be the one sent.
    for s in &SETS {
        for k in 0..3 {
            let (cb_o, q_o) = ap::launch_bytes(s, PROGS[k], 0x1000, 0x2000, 0x10_000).unwrap();
            let (cb_2, q_2) = ap::launch_bytes2(s, PROGS2[k], 0x1000, 0x2000, 0x10_000).unwrap();
            let (cb_3, q_3) = ap::launch_bytes3(s, PROGS3[k], 0x1000, 0x2000, 0x10_000).unwrap();
            assert_eq!(cb_o, cb_3, "{} {}", s.name, PROGS3[k].name());
            assert_eq!(cb_2, cb_3);
            assert_ne!(q_o, q_3, "{} {}", s.name, PROGS3[k].name());
            assert_ne!(q_2, q_3, "{} {}", s.name, PROGS3[k].name());
        }
    }
    assert!(ap::launch_bytes3(&SETS[0], PROGS3[0], 0x1000, 0x2000, 0x10_200).is_err());
}

#[test]
fn t191b_three_generations_on_the_small_sets() {
    let mut cp = cp();
    let (mut old, mut p2, mut p3) = (Probe::default(), Probe2::default(), Probe3::default());
    for (i, s) in SETS[..TIMING_SET].iter().enumerate() {
        let o = chain(&mut cp, s, 1, Plant::None).unwrap();
        let n2 = chain(&mut cp, s, 2, Plant::None).unwrap();
        let n3 = chain(&mut cp, s, 3, Plant::None).unwrap();
        old.add(ap::judge_set(i, &o, [&[], &[], &[]]).unwrap());
        p2.add(ap::judge_pair_set(i, &o, &n2, [&[], &[], &[]]).unwrap());
        let r = ap::judge_pair_set(i, &o, &n3, [&[], &[], &[]]).unwrap();
        assert!(r.identical(), "{}: {:?}", s.name, r.same);
        assert_eq!(n2, n3, "{}", s.name);
        p3.add(r);
    }
    for l in p3.lines("SIMULARE", Some(&p2)) {
        println!("{l}");
    }
    assert!(old.passes() && p2.passes() && p3.passes() && p3.identical());
    assert!(p3.timing_skipped());
    assert_eq!(p3.identical_sets(), TIMING_SET);
    assert_eq!(p3.in_margin(), TIMING_SET);
    assert_eq!(p3.per_prog(), old.per_prog());
    assert_eq!(p3.same_per_prog(), p2.same_per_prog());
    for (k, (ex, el, w)) in p3.per_prog().iter().enumerate() {
        assert_eq!(ex, el, "{}", PROGS3[k].name());
        assert_eq!(*w, 0);
    }
}

#[test]
fn t191c_planted_mistakes_are_caught() {
    let s = &SETS[1]; // t130-d64: two tiles, the last one partial
    let mut cp = cp();
    // The third forward with the first generation's shared memory in its QMD: refused, nothing launched.
    assert!(chain(&mut cp, s, 3, Plant::Fwd3OldShared).is_err());
    // One ulp of dqkv in the third chain: still in the margin, but no longer identical (exactly one word, back1).
    let o = chain(&mut cp, s, 1, Plant::None).unwrap();
    let n = chain(&mut cp, s, 3, Plant::Dqkv3Ulp).unwrap();
    let r = ap::judge_pair_set(1, &o, &n, [&[], &[], &[]]).unwrap();
    assert!(r.new.passes(), "{r:?}");
    assert!(!r.identical());
    assert_eq!(r.same[2].0 + 1, r.same[2].1);
    assert_eq!(r.same[0].0, r.same[0].1);
    assert_eq!(r.same[1].0, r.same[1].1);
    let mut p = Probe3::default();
    p.add(r);
    assert_eq!(p.identical_sets(), 0);
    assert!(!p.identical());
    // A short read-back on the new side: invalid and nothing counted identical.
    let r = ap::judge_pair_set(1, &o, &n[..n.len() - 4], [&[], &[], &[]]).unwrap();
    assert!(!r.new.passes());
    assert!(r.same.iter().all(|&(a, b)| a == 0 && b > 0));
    // A set missing: the probe does not pass.
    let mut p = Probe3::default();
    for i in [0, 1, 3, 4] {
        let m = SETS[i].model_out();
        p.add(ap::judge_pair_set(i, &m, &m, [&[], &[], &[]]).unwrap());
    }
    assert!(!p.passes() && p.identical());
    assert!(!Probe3::default().identical());
}

#[test]
fn t191d_the_timing_set_rows_and_the_ratio_to_the_second_generation() {
    let s = &SETS[TIMING_SET];
    let out = s.model_out();
    let (t2_f, t2_0, t2_1) = ([1_530_000u64; 5], [1_540_000u64; 5], [2_250_000u64; 5]);
    let (t3_f, t3_0, t3_1) = (
        [810_000u64, 800_000, 820_000, 805_000, 815_000],
        [550_000u64; 5],
        [1_600_000u64; 5],
    );
    let mut old = Probe::default();
    let mut p2 = Probe2::default();
    let mut p3 = Probe3::default();
    for (i, si) in SETS[..TIMING_SET].iter().enumerate() {
        let m = si.model_out();
        old.add(ap::judge_set(i, &m, [&[], &[], &[]]).unwrap());
        p2.add(ap::judge_pair_set(i, &m, &m, [&[], &[], &[]]).unwrap());
        p3.add(ap::judge_pair_set(i, &m, &m, [&[], &[], &[]]).unwrap());
    }
    old.add(
        ap::judge_set(
            TIMING_SET,
            &out,
            [&[2_490_000u64; 5], &[1_810_000; 5], &[3_990_000; 5]],
        )
        .unwrap(),
    );
    p2.add(ap::judge_pair_set(TIMING_SET, &out, &out, [&t2_f, &t2_0, &t2_1]).unwrap());
    p3.add(ap::judge_pair_set(TIMING_SET, &out, &out, [&t3_f, &t3_0, &t3_1]).unwrap());
    assert!(p3.passes() && p3.identical() && !p3.timing_skipped());
    let lines = p3.lines("SIMULARE", Some(&p2));
    for l in &lines {
        println!("{l}");
    }
    assert!(lines[0].starts_with(
        "atentie3 vechi langa nou (SIMULARE): 6 / 6 seturi in margine, 6 / 6 seturi identice"
    ));
    assert!(lines.iter().any(|l| {
        l.contains(
            "attn_fwd_tile3 (SIMULARE): mediana 810000 ns pe 5 lansari (estimat ~0,7-1,1 ms)",
        ) && l.contains("attn_fwd_tile2 1530000 ns (aceeasi rulare), de 1,88 ori mai rapid")
    }));
    assert!(lines.iter().any(|l| l.contains("attn_back_tile0_3")
        && l.contains("attn_back_tile0_2 1540000 ns")
        && l.contains("de 2,80 ori")));
    assert!(
        lines
            .iter()
            .any(|l| l.contains("cuvinte identice cu attn_back_tile")
                && l.starts_with("atentie3 attn_back_tile_3"))
    );
    // Without the second probe: the os-364 medians.
    let alone = p3.lines("SIMULARE", None);
    assert!(
        alone
            .iter()
            .any(|l| l.contains(&format!("attn_back_tile_2 {} ns (os-364)", OS364_NS[2])))
    );
    // The rows, their reader, and none of the three readers sees the others' rows.
    let mut j = Journal::new(4096);
    p3.journal(&mut j);
    let text = |rows: Vec<(&'static str, u32)>| -> String {
        rows.iter()
            .map(|(w, v)| format!("gpu: valoare {w} = 0x{v:x}\n"))
            .collect()
    };
    let (text1, text2, text3) = (text(old.rows()), text(p2.rows()), text(p3.rows()));
    let all = format!("{text1}{text2}{text3}");
    assert_eq!(parse_attn_tile(&all), parse_attn_tile(&text1));
    assert_eq!(parse_attn_tile2(&all), parse_attn_tile2(&text2));
    let q = parse_attn_tile3(&all);
    assert_eq!(q, parse_attn_tile3(&text3));
    assert_eq!(
        parse_attn_tile3(&format!("{text1}{text2}")),
        Default::default()
    );
    assert_eq!(q.sets, Some(6));
    assert_eq!(q.in_margin, Some(6));
    assert_eq!(q.identical_sets, Some(6));
    assert_eq!(q.timing_skipped, Some(0));
    assert_eq!(q.timing_ns, [Some(810_000), Some(550_000), Some(1_600_000)]);
    let (pp, sp) = (p3.per_prog(), p3.same_per_prog());
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
    // The same values as the second probe's rows, under the other prefix.
    let r2 = p2.rows();
    let r3 = p3.rows();
    assert_eq!(r2.len(), r3.len());
    for ((w2, v2), (w3, v3)) in r2.iter().zip(&r3) {
        assert_eq!(w2.replacen("atentie2", "atentie3", 1), *w3);
        if !w2.contains("timp") {
            assert_eq!(v2, v3, "{w2}");
        }
    }
    // A second run in the same journal starts over.
    assert_eq!(parse_attn_tile3(&format!("{text3}{text3}")), q);
}
