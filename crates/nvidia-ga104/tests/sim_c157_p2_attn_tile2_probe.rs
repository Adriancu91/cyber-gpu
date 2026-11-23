//! C-157 point 2 (T184), SIMULARE: the probe "old beside new" for the tiled training attention — on every small set of
//! `nvidia_ga104::attn_tile_probe::SETS`, the first-generation chain (`attn_fwd_tile` → `attn_back_tile0` →
//! `attn_back_tile`) and then, on the same buffer refilled, the second-generation chain (`attn_fwd_tile2` →
//! `attn_back_tile0_2` → `attn_back_tile_2`), both through the command-processor model with the cbuf + QMD of
//! `launch_bytes` / `launch_bytes2`; the new read-back judged with the old margins and compared word for word with the
//! old (plan and predictions written before the code: `docs/C-157_P2_ATTN_TILE2.md`, commit `b6be099`).
//!
//! The "programs" run on the CPU: old and new are registered with the **same** CPU models (whose equality with each
//! GLSL is T173–T175 and T181–T183), so here the two chains are equal by construction — this test checks the path
//! (QMD, shared memory, buffers, comparison, rows), not the board. Nothing here proves anything about the RTX 3070 Ti.

use gpu_core::diag::Journal;
use gpu_sim::cmdproc::{
    CommandProcessor, Needs, ProgramFn, attn_back_tile_on_cpu, attn_back_tile0_on_cpu,
    attn_fwd_tile_on_cpu,
};
use nvidia_ga104::attn_tile_probe::{
    self as ap, OS358_NS, PROGS, PROGS2, Probe, Probe2, ProbeSet, Prog, Prog2, SETS, TIMING_SET,
    parse_attn_tile, parse_attn_tile2,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

/// Program addresses: old at slots 0–2, new at slots 3–5.
const PROG: [u64; 6] = [
    0x10_0000, 0x14_0000, 0x18_0000, 0x1c_0000, 0x50_0000, 0x54_0000,
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
        let (o, n) = (PROGS[k], PROGS2[k]);
        for (code, sh, slot) in [(o.binary(), o.shared(), k), (n.binary(), n.shared(), 3 + k)] {
            cp.register_program_with_needs(
                code,
                RUNS[k],
                Needs {
                    shared_bytes: sh.bytes,
                    barriers: u32::from(sh.barriers),
                },
            );
            cp.mem.write(PROG[slot], code);
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
    /// The new forward's QMD declares the old shared memory (37 376 < 37 888).
    NewFwdOldShared,
    /// One dS word of the new chain moved by one unit in the last place after its launches.
    NewDsUlp,
}

/// Runs one chain (`second`: the new one) on `s` in `cp`, on the buffer refilled; returns the read-back.
fn chain(
    cp: &mut CommandProcessor,
    s: &ProbeSet,
    second: bool,
    plant: Plant,
) -> Result<Vec<u8>, gpu_core::error::GpuError> {
    cp.mem.write(BUF, &s.buffer());
    cp.mem.write_u32(SEM + 8, 0);
    for k in 0..3 {
        let slot = if second { 3 + k } else { k };
        let cbuf = CBUF + 0x1000 * slot as u64;
        let (cb, mut q) = if second {
            ap::launch_bytes2(s, PROGS2[k], PROG[slot], cbuf, BUF).unwrap()
        } else {
            ap::launch_bytes(s, PROGS[k], PROG[slot], cbuf, BUF).unwrap()
        };
        if second && k == 0 && plant == Plant::NewFwdOldShared {
            let a = s.fwd_args(BUF);
            q = nvidia_ga104::qmd::ComputeQmd {
                program_va: PROG[slot],
                registers: PROGS2[0].registers(),
                cbuf0_va: cbuf,
                cbuf0_size: ap::CBUF_SIZE as u32,
                grid: nvidia_ga104::attn_fwd_tile2::grid(&a),
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
    if second && plant == Plant::NewDsUlp {
        let t = s.t as usize;
        let at = BUF + l.ds as u64 + 4 * ((t - 1) * t) as u64;
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
fn t184a_the_new_programs_are_the_ones_the_notes_name() {
    for (k, n) in PROGS2.iter().enumerate() {
        assert_eq!(n.old(), PROGS[k]);
        let hex: String = nvidia_ga104::sha256::digest(n.binary())
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hex, n.sha256(), "{}", n.name());
        assert_ne!(n.binary(), PROGS[k].binary());
    }
    assert_eq!(PROGS2.map(Prog2::registers), [117, 52, 79]);
    assert_eq!(PROGS2.map(|p| p.shared().bytes), [37_888, 33_024, 33_024]);
    assert_eq!(
        PROGS2.map(Prog2::name),
        ["attn_fwd_tile2", "attn_back_tile0_2", "attn_back_tile_2"]
    );
    // The old list is unchanged (the OS reads it).
    assert_eq!(
        PROGS.map(Prog::name),
        ["attn_fwd_tile", "attn_back_tile0", "attn_back_tile"]
    );
    assert_eq!(PROGS.map(Prog::registers), [55, 50, 73]);
    // The same cbuf as the old program on every set; the QMDs differ (registers, shared memory).
    for s in &SETS {
        for k in 0..3 {
            let (cb_o, q_o) = ap::launch_bytes(s, PROGS[k], 0x1000, 0x2000, 0x10_000).unwrap();
            let (cb_n, q_n) = ap::launch_bytes2(s, PROGS2[k], 0x1000, 0x2000, 0x10_000).unwrap();
            assert_eq!(cb_o, cb_n, "{} {}", s.name, PROGS2[k].name());
            assert_ne!(q_o, q_n, "{} {}", s.name, PROGS2[k].name());
        }
    }
    assert!(ap::launch_bytes2(&SETS[0], PROGS2[0], 0x1000, 0x2000, 0x10_200).is_err());
}

#[test]
fn t184b_old_beside_new_on_the_small_sets() {
    let mut cp = cp();
    let (mut old, mut new) = (Probe::default(), Probe2::default());
    for (i, s) in SETS[..TIMING_SET].iter().enumerate() {
        let o = chain(&mut cp, s, false, Plant::None).unwrap();
        let n = chain(&mut cp, s, true, Plant::None).unwrap();
        old.add(ap::judge_set(i, &o, [&[], &[], &[]]).unwrap());
        let ns = [7000 + i as u64];
        let r = ap::judge_pair_set(i, &o, &n, [&ns, &ns, &ns]).unwrap();
        assert!(r.identical(), "{}: {:?}", s.name, r.same);
        new.add(r);
    }
    for l in new.lines("SIMULARE", Some(&old)) {
        println!("{l}");
    }
    assert!(old.passes() && new.passes() && new.identical());
    assert!(new.timing_skipped());
    assert_eq!(new.identical_sets(), TIMING_SET);
    // Same model on both sides: everything exact, and the new chain's per-program counts = the old ones.
    assert_eq!(new.per_prog(), old.per_prog());
    for (k, (ex, el, w)) in new.per_prog().iter().enumerate() {
        assert_eq!(ex, el, "{}", PROGS2[k].name());
        assert_eq!(*w, 0);
    }
    // Words compared: P + att, dS, dqkv of every small set (above-diagonal zeros included).
    let sp = new.same_per_prog();
    let want_ds: usize = SETS[..TIMING_SET]
        .iter()
        .map(|s| (s.batch * s.heads * s.t * s.t) as usize)
        .sum();
    assert_eq!(sp[1], (want_ds as u32, want_ds as u32));
    let want_dq: usize = SETS[..TIMING_SET]
        .iter()
        .map(|s| (s.batch * s.t * 3 * s.c) as usize)
        .sum();
    assert_eq!(sp[2].1 as usize, want_dq);
}

#[test]
fn t184c_planted_mistakes_are_caught() {
    let s = &SETS[1]; // t130-d64: two tiles, the last one partial
    let mut cp = cp();
    // The new forward with the old shared memory in its QMD: refused, nothing launched.
    assert!(chain(&mut cp, s, true, Plant::NewFwdOldShared).is_err());
    // One ulp of dS in the new chain: still in the margin, but no longer identical to the old (exactly one word).
    let o = chain(&mut cp, s, false, Plant::None).unwrap();
    let n = chain(&mut cp, s, true, Plant::NewDsUlp).unwrap();
    let r = ap::judge_pair_set(1, &o, &n, [&[], &[], &[]]).unwrap();
    assert!(r.new.passes(), "{r:?}");
    assert!(!r.identical());
    assert_eq!(r.same[1].0 + 1, r.same[1].1);
    assert_eq!(r.same[0].0, r.same[0].1);
    assert_eq!(r.same[2].0, r.same[2].1);
    // A short read-back on the new side: invalid and nothing counted identical.
    let r = ap::judge_pair_set(1, &o, &n[..n.len() - 4], [&[], &[], &[]]).unwrap();
    assert!(!r.new.passes());
    assert!(r.same.iter().all(|&(a, b)| a == 0 && b > 0));
    // A set missing: the probe does not pass.
    let mut p = Probe2::default();
    for i in [0, 1, 2, 4] {
        let m = SETS[i].model_out();
        p.add(ap::judge_pair_set(i, &m, &m, [&[], &[], &[]]).unwrap());
    }
    assert!(!p.passes() && p.identical());
    assert!(!Probe2::default().identical());
}

#[test]
fn t184d_the_timing_set_rows_and_the_ratio_to_the_old_times() {
    let s = &SETS[TIMING_SET];
    let out = s.model_out();
    let (o_f, o_0, o_1) = ([2_490_000u64; 5], [1_810_000u64; 5], [3_990_000u64; 5]);
    let (n_f, n_0, n_1) = (
        [900_000u64, 910_000, 890_000, 905_000, 895_000],
        [800_000u64; 5],
        [1_500_000u64; 5],
    );
    let mut old = Probe::default();
    let mut new = Probe2::default();
    for (i, si) in SETS[..TIMING_SET].iter().enumerate() {
        let m = si.model_out();
        old.add(ap::judge_set(i, &m, [&[], &[], &[]]).unwrap());
        new.add(ap::judge_pair_set(i, &m, &m, [&[], &[], &[]]).unwrap());
    }
    old.add(ap::judge_set(TIMING_SET, &out, [&o_f, &o_0, &o_1]).unwrap());
    new.add(ap::judge_pair_set(TIMING_SET, &out, &out, [&n_f, &n_0, &n_1]).unwrap());
    assert!(new.passes() && new.identical() && !new.timing_skipped());
    let lines = new.lines("SIMULARE", Some(&old));
    for l in &lines {
        println!("{l}");
    }
    assert!(lines.iter().any(|l| {
        l.contains("attn_fwd_tile2 (SIMULARE): mediana 900000 ns pe 5 lansari")
            && l.contains("vechiul 2490000 ns (aceeasi rulare), de 2,76 ori mai rapid")
    }));
    assert!(
        lines
            .iter()
            .any(|l| l.contains("attn_back_tile_2") && l.contains("de 2,66 ori"))
    );
    // Without the old probe: the os-358 medians.
    let alone = new.lines("SIMULARE", None);
    assert!(
        alone
            .iter()
            .any(|l| l.contains(&format!("vechiul {} ns (os-358)", OS358_NS[1])))
    );
    // The rows, their reader, and the two readers do not see each other's rows.
    let mut j = Journal::new(4096);
    new.journal(&mut j);
    let text2: String = new
        .rows()
        .iter()
        .map(|(w, v)| format!("gpu: valoare {w} = 0x{v:x}\n"))
        .collect();
    let text1: String = old
        .rows()
        .iter()
        .map(|(w, v)| format!("gpu: valoare {w} = 0x{v:x}\n"))
        .collect();
    let both = format!("{text1}{text2}");
    assert_eq!(parse_attn_tile(&both), parse_attn_tile(&text1));
    let q = parse_attn_tile2(&both);
    assert_eq!(q, parse_attn_tile2(&text2));
    assert_eq!(parse_attn_tile2(&text1), Default::default());
    assert_eq!(q.sets, Some(6));
    assert_eq!(q.in_margin, Some(6));
    assert_eq!(q.identical_sets, Some(6));
    assert_eq!(q.timing_skipped, Some(0));
    assert_eq!(q.timing_ns, [Some(900_000), Some(800_000), Some(1_500_000)]);
    let (pp, sp) = (new.per_prog(), new.same_per_prog());
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
    assert_eq!([last[2], last[8], last[14]], [1, 1, 1]); // verdicts
    assert_eq!(last[7], new.sets[5].same[0].0); // fwd identical words
    assert_eq!(last[19], new.sets[5].same[2].0); // back1 identical words
    // A second run in the same journal starts over.
    assert_eq!(parse_attn_tile2(&format!("{text2}{text2}")), q);
}
