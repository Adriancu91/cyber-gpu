//! T301 (C-168 point 4, step 3, `docs/C-168_ATTN_FWD_TC_PLAN.md` §10), SIMULARE: the probe `--atentie-fwd-tc`
//! (`nvidia_ga104::attn_fwd_tc_probe`) — `attn_fwd_tc` beside `attn_fwd_tile3` on the same inputs, as the OS will run
//! it: one buffer per set, the read-back ranges filled with NaN before each program, the launches through our
//! pushbuffer + QMD (attn_fwd_tc at the 64 KiB step), the two read-backs judged with `attn_fwd_tc_probe::judge`.
//! The stand-ins of the simulator are the CPU models, so what this checks is the probe's plumbing and its verdict,
//! and the size of the gap the board will be judged on (H4). Nothing here runs on the RTX 3070 Ti.

// The read-back is built per head and row, indexing the per-head tiles (clarity over iterators).
#![allow(clippy::needless_range_loop)]

use gpu_compute::attn_fwd_tc as m;
use gpu_compute::layer::attn_fwd_tile_row_gpu;
use gpu_sim::cmdproc::{
    ATTN_FWD_TC_NEEDS, CommandProcessor, attn_fwd_tc_on_cpu, attn_fwd_tile_on_cpu,
};
use nvidia_ga104::attn_fwd_tc::{self as tc, ATTN_FWD_TC};
use nvidia_ga104::attn_fwd_tc_probe::{self as pr, FILL, GUARD, ProbeSet, SETS};
use nvidia_ga104::attn_fwd_tile3::ATTN_FWD_TILE3;
use nvidia_ga104::attn_tile_probe::{self as ap, Prog, Prog3};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_rm570::regs::clc7c0 as c;

const BUF: u64 = 0x1000_0000;
const PROG_TC: u64 = 0x10_0000;
const PROG_T3: u64 = 0x20_0000;
const CBUF: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;

fn submit(cp: &mut CommandProcessor) {
    let mut w = [0u32; 16];
    let mut pb = PushBuf::new(&mut w);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    pb.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    pb.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = pb.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
}

fn fill(cp: &mut CommandProcessor, set: &ProbeSet) {
    for (o, n) in pr::fwd_ranges(set) {
        let words: Vec<u8> = (0..n / 4).flat_map(|_| FILL.to_le_bytes()).collect();
        cp.mem.write(BUF + o as u64, &words);
    }
}

fn read_back(cp: &CommandProcessor, set: &ProbeSet) -> Vec<u8> {
    let mut out = Vec::new();
    for (o, n) in pr::fwd_ranges(set) {
        let mut b = vec![0u8; n];
        cp.mem.read(BUF + o as u64, &mut b).unwrap();
        out.extend_from_slice(&b);
    }
    out
}

/// The probe on one small set, as the OS will do it: (tile3 read-back, tc read-back).
fn run_set(set: &ProbeSet) -> (Vec<u8>, Vec<u8>) {
    let mut cp = CommandProcessor::new();
    cp.register_program(ATTN_FWD_TILE3, attn_fwd_tile_on_cpu);
    cp.register_program_with_needs(ATTN_FWD_TC, attn_fwd_tc_on_cpu, ATTN_FWD_TC_NEEDS);
    cp.mem.write(PROG_T3, ATTN_FWD_TILE3);
    cp.mem.write(PROG_TC, ATTN_FWD_TC);
    cp.mem.write(BUF, &set.buffer());
    // attn_fwd_tile3 (as the attention probe launches it)
    fill(&mut cp, set);
    let (cb, q) = ap::launch_bytes3(set, Prog3(Prog::Fwd), PROG_T3, CBUF, BUF).unwrap();
    cp.mem.write(CBUF, &cb);
    cp.mem.write(QMD, &q);
    submit(&mut cp);
    let t3 = read_back(&cp, set);
    // attn_fwd_tc on the same inputs, the ranges filled again
    fill(&mut cp, set);
    let (cb, q) = pr::launch_bytes(set, PROG_TC, CBUF, BUF).unwrap();
    cp.mem.write(CBUF, &cb);
    cp.mem.write(QMD, &q);
    submit(&mut cp);
    (t3, read_back(&cp, set))
}

/// H4 in SIMULARE on the four small aligned sets: both programs through the command processor, judged; every set
/// passes, `attn_fwd_tc`'s words are the CPU model's (the stand-in is the model), the gap to tile3 is printed.
#[test]
fn t301_small_sets_end_to_end() {
    let mut probe = pr::Probe::default();
    for set in &SETS[..pr::TIMING_SET] {
        let (t3, tcb) = run_set(set);
        let r = pr::judge(set, &t3, &tcb, &[1_000], &[400]).unwrap();
        eprintln!(
            "T301 {}: O {:?} P {:?} worst {:e} exact {:?}",
            set.name, r.o_rel, r.p_rel, r.p_worst_abs, r.exact_model
        );
        assert!(r.passes(), "{}: {r:?}", set.name);
        let (same, all) = r.exact_model.unwrap();
        assert_eq!(same, all, "{}: stand-in = model", set.name);
        assert!(r.o_rel.unwrap() > 0.0, "{}: f16 vs F32 differ", set.name);
        probe.add(r);
    }
    assert!(probe.passes());
    let lines = probe.lines("SIMULARE");
    assert_eq!(lines.len(), pr::TIMING_SET + 1);
    assert!(
        lines
            .last()
            .unwrap()
            .ends_with("4 / 4 seturi in margine (O <= 1e-3, P <= 2e-3): TRECE")
    );
}

/// The read-back of the banded timing set, built from the CPU models (the simulator skips the big set, as the OS does).
fn banded(set: &ProbeSet) -> (Vec<u8>, Vec<u8>) {
    let s = set.shape();
    let (nq, _, _) = s.sizes().unwrap();
    let qkv: Vec<f32> = (0..nq).map(|i| set.qkv_at(i)).collect();
    let (t, c, h) = (s.t, s.c, s.heads);
    let mut t3: Vec<f32> = Vec::new();
    let mut tcw: Vec<u32> = Vec::new();
    let mut t3a: Vec<f32> = Vec::new();
    let mut tca: Vec<f32> = Vec::new();
    for &b in &set.texts() {
        let tiles: Vec<Vec<m::FwdTile>> = (0..h)
            .map(|hh| {
                (0..t / 64)
                    .map(|rt| m::fwd_tile(&s, &qkv, set.scale(), (b, hh, rt)).unwrap())
                    .collect()
            })
            .collect();
        for hh in 0..h {
            for &i in &set.rows() {
                let (row, _) = attn_fwd_tile_row_gpu(&s, &qkv, set.scale(), b, hh, i);
                t3.extend_from_slice(&row);
                let (rt, r) = (i / 64, i % 64);
                let wc = (rt + 1) * 64;
                let tile = &tiles[hh][rt];
                tcw.extend(tile.p[r * wc..(r + 1) * wc].iter().map(|x| x.to_bits()));
                tcw.extend(std::iter::repeat_n(FILL, t - wc));
            }
        }
        for &i in &set.rows() {
            for hh in 0..h {
                let (_, a) = attn_fwd_tile_row_gpu(&s, &qkv, set.scale(), b, hh, i);
                t3a.extend_from_slice(&a);
                let (rt, r) = (i / 64, i % 64);
                tca.extend_from_slice(&tiles[hh][rt].att[r * 64..(r + 1) * 64]);
            }
        }
    }
    assert_eq!(t3a.len(), set.texts().len() * set.rows().len() * c);
    let mut a = Vec::new();
    let mut g = Vec::new();
    let guard: Vec<u32> = vec![FILL; GUARD];
    for x in &t3 {
        a.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    for x in &tcw {
        g.extend_from_slice(&x.to_le_bytes());
    }
    for v in [&mut a, &mut g] {
        for w in &guard {
            v.extend_from_slice(&w.to_le_bytes());
        }
    }
    for x in &t3a {
        a.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    for x in &tca {
        g.extend_from_slice(&x.to_bits().to_le_bytes());
    }
    for v in [&mut a, &mut g] {
        for w in &guard {
            v.extend_from_slice(&w.to_le_bytes());
        }
    }
    (a, g)
}

/// H4 in SIMULARE on `atentie-513-b4` (bands: texts 0 and 3, rows 0 / 300 / 511, all 12 heads), and the time verdict
/// line against H5.
#[test]
fn t301_timing_set_by_the_models() {
    let set = &SETS[pr::TIMING_SET];
    assert_eq!(set.name, "atentie-513-b4");
    let (t3, tcb) = banded(set);
    assert_eq!(t3.len(), pr::fwd_read_len(set));
    let r = pr::judge(set, &t3, &tcb, &[1_001_000; 5], &[500_000; 5]).unwrap();
    eprintln!(
        "T301 {}: O {:?} P {:?} worst {:e}",
        set.name, r.o_rel, r.p_rel, r.p_worst_abs
    );
    assert!(r.passes(), "{r:?}");
    assert_eq!(r.exact_model, None);
    let mut probe = pr::Probe::default();
    probe.add(r);
    let lines = probe.lines("SIMULARE");
    assert!(
        lines[0].contains("tile3 1001000 ns, tc 500000 ns, x2,00"),
        "{}",
        lines[0]
    );
    assert!(
        lines[2].ends_with("(prezis 0,367 ms - 0,620 ms): in fereastra"),
        "{}",
        lines[2]
    );
}

fn set_word(b: &mut [u8], idx: usize, w: u32) {
    b[4 * idx..4 * idx + 4].copy_from_slice(&w.to_le_bytes());
}

fn word(b: &[u8], idx: usize) -> u32 {
    u32::from_le_bytes(b[4 * idx..4 * idx + 4].try_into().unwrap())
}

/// The verdict is not blind: each planted mistake in `attn_fwd_tc`'s read-back fails the set, for the right reason.
#[test]
fn t301_planted_mistakes_are_caught() {
    let set = &SETS[1]; // tc-t192-b2: T 192, B 2, H 1
    let (t3, good) = run_set(set);
    let t = 192usize;
    let np = 2 * t * t;
    let judge = |g: &[u8]| pr::judge(set, &t3, g, &[], &[]).unwrap();
    assert!(judge(&good).passes());
    // (a) a word written above the diagonal tile (row 10, column 100)
    let mut g = good.clone();
    set_word(&mut g, 10 * t + 100, 0);
    let r = judge(&g);
    assert!(!r.passes() && !r.unwritten_ok && r.zeros_ok && r.guards_ok);
    // (b) not an exact 0 above the diagonal inside the diagonal tile (row 70, column 100)
    let mut g = good.clone();
    set_word(&mut g, 70 * t + 100, 1.0e-30f32.to_bits());
    let r = judge(&g);
    assert!(!r.passes() && !r.zeros_ok && r.unwritten_ok);
    // (c) a P guard word and an att guard word
    for at in [np, g.len() / 4 - 1] {
        let mut g = good.clone();
        set_word(&mut g, at, 0);
        let r = judge(&g);
        assert!(!r.passes() && !r.guards_ok, "{at}");
    }
    // (d) one row of P 10 % too large (row 20 of text 1): P over its limit, O untouched
    let mut g = good.clone();
    for j in 0..=20 {
        let k = t * t + 20 * t + j;
        let v = f32::from_bits(word(&g, k)) * 1.1;
        set_word(&mut g, k, v.to_bits());
    }
    let r = judge(&g);
    assert!(!r.passes() && r.p_rel.unwrap() > pr::P_LIMIT && r.o_rel.unwrap() <= pr::O_LIMIT);
    // (e) att rows of text 1 shifted by one position: O over its limit
    let mut g = good.clone();
    let a0 = np + GUARD;
    for i in 1..t {
        for e in 0..64 {
            let w = word(&good, a0 + (t + i - 1) * 64 + e);
            set_word(&mut g, a0 + (t + i) * 64 + e, w);
        }
    }
    let r = judge(&g);
    assert!(!r.passes() && r.o_rel.unwrap() > pr::O_LIMIT);
    // (f) a NaN left in P below the diagonal (a block that did not write): invalid norm
    let mut g = good.clone();
    set_word(&mut g, 191 * t + 5, FILL);
    let r = judge(&g);
    assert!(!r.passes() && r.p_rel.is_none());
    // (g) wrong lengths and a set attn_fwd_tc does not take
    assert!(pr::judge(set, &t3, &good[4..], &[], &[]).is_err());
    let unaligned = &ap::SETS[4]; // ascutit: T 96
    assert!(!pr::fits(unaligned));
    assert!(pr::judge(unaligned, &[], &[], &[], &[]).is_err());
    assert!(pr::launch_bytes(unaligned, PROG_TC, CBUF, BUF).is_err());
    assert!(pr::launch_bytes(set, PROG_TC, CBUF, BUF + 0x100).is_err());
}

/// The launch bytes: cbuf = `attn_fwd_tc::cbuf`, QMD = `attn_fwd_tc::qmd` at the 64 KiB step — not the one-block
/// step (32 KiB, one block per SM), which the plain shared encoding would pick.
#[test]
fn t301_launch_bytes_at_the_64k_step() {
    let set = &SETS[pr::TIMING_SET];
    let (cb, q) = pr::launch_bytes(set, PROG_TC, CBUF, BUF).unwrap();
    let a = set.fwd_args(BUF);
    assert_eq!(cb, tc::cbuf(&a).unwrap());
    let base = tc::qmd(&a, PROG_TC, CBUF, None).unwrap();
    assert_eq!(
        q,
        base.to_bytes_with_sm_config(&tc::SHARED, 64 << 10).unwrap()
    );
    assert_ne!(q, base.to_bytes_with_shared(&tc::SHARED).unwrap());
    assert_eq!(tc::grid(&a), [4, 12, 4]);
}

/// The report lines read back by `parse` (what we will read from the OS report).
#[test]
fn t301_lines_parse_back() {
    let mut probe = pr::Probe::default();
    for set in &SETS[..2] {
        let (t3, tcb) = run_set(set);
        probe.add(
            pr::judge(
                set,
                &t3,
                &tcb,
                &[900_000, 1_000_000, 950_000],
                &[400_000, 410_000, 405_000],
            )
            .unwrap(),
        );
    }
    let mut bad = probe.sets[1].clone();
    bad.name = "rau";
    bad.unwritten_ok = false;
    probe.add(bad);
    let text: String = probe
        .lines("placa")
        .iter()
        .map(|l| format!("kronos: lant: gpu0: kronos --gpu: {l}\n"))
        .collect();
    let got = pr::parse(&text);
    assert_eq!(got.len(), 3);
    assert_eq!(got[0].name, "tc-t64");
    assert!(got[0].passes && got[1].passes && !got[2].passes);
    assert_eq!(got[0].ns_tile3, Some(950_000));
    assert_eq!(got[0].ns_tc, Some(405_000));
    let o = probe.sets[0].o_rel.unwrap();
    let back = got[0].o_rel.unwrap();
    assert!((back - o).abs() <= 0.06 * o, "{back} {o}");
    assert!(text.contains("2 / 3 seturi in margine"));
}
