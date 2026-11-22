//! C-160 point 6 (T166), SIMULARE: the first training "backward" program as the OS would send it to the board —
//! for every set of `nvidia_ga104::colsum_probe::SETS`, one host buffer (dy, x, μ, rstd; outputs and guards = NaN),
//! the cbuf + QMD from `launch_bytes`, one launch of `colsum` through the command-processor model, the outputs read
//! back and judged against the f64 oracle within the margin written in T155. The "program" runs on the CPU
//! (`gpu_sim::cmdproc::colsum_on_cpu`); nothing here proves anything about the RTX 3070 Ti.

use gpu_core::diag::Journal;
use gpu_sim::cmdproc::{CommandProcessor, colsum_on_cpu};
use nvidia_ga104::colsum::{self as cs, COLSUM};
use nvidia_ga104::colsum_probe::{
    self as cp_, BUF_BYTES, FILL, Probe, SETS, Verdict, parse_colsum,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const QMD: u64 = 0x20_0800;
const SEM: u64 = 0x30_0000;
const BUF: u64 = 0x40_0000;
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

/// A mistake planted in what is sent.
#[derive(Clone, Copy, PartialEq)]
enum Plant {
    None,
    /// No launch at all.
    NoLaunch,
    /// One block fewer than ⌈c/256⌉ (the last columns stay NaN).
    ShortGrid,
    /// One row fewer in the cbuf.
    RowMissing,
    /// The other mode in the cbuf (dγ/dβ ↔ bias).
    OtherMode,
    /// μ and rstd of row 0 swapped with row 1 in the buffer.
    StatsSwapped,
}

/// One set through the model; returns the bytes of `out_range`.
fn run_set(i: usize, plant: Plant) -> Vec<u8> {
    let s = SETS[i];
    let mut cp = CommandProcessor::new();
    cp.register_program(COLSUM, colsum_on_cpu);
    cp.mem.write(PROG, COLSUM);
    let (mut cb, q) = cp_::launch_bytes(i, PROG, CBUF, BUF).unwrap();
    let mut buf = s.buffer();
    assert_eq!(buf.len(), BUF_BYTES);
    let mut q = q;
    match plant {
        Plant::RowMissing => cb[0x58..0x5c].copy_from_slice(&(s.rows - 1).to_le_bytes()),
        Plant::OtherMode => {
            let m = u32::from_le_bytes(cb[0x60..0x64].try_into().unwrap());
            cb[0x60..0x64].copy_from_slice(&(1 - m).to_le_bytes());
        }
        Plant::ShortGrid => {
            let mut a = s.args(BUF);
            a.c = s.c; // same cbuf, fewer blocks
            let grid = [a.grid()[0] - 1, 1, 1];
            q = nvidia_ga104::qmd::ComputeQmd {
                program_va: PROG,
                registers: cs::REGISTERS,
                cbuf0_va: CBUF,
                cbuf0_size: cp_::CBUF_SIZE as u32,
                grid,
                block: cs::BLOCK,
                release: None,
            }
            .to_bytes()
            .unwrap();
        }
        Plant::StatsSwapped => {
            let l = s.layout();
            for base in [l.mean, l.rstd] {
                let (a, b) = (base, base + 4);
                let t: [u8; 4] = buf[a..a + 4].try_into().unwrap();
                buf.copy_within(b..b + 4, a);
                buf[b..b + 4].copy_from_slice(&t);
            }
        }
        _ => {}
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write(QMD, &q);
    cp.mem.write(BUF, &buf);
    cp.mem.write_u32(SEM, 0);
    if plant != Plant::NoLaunch {
        submit(&mut cp).unwrap_or_else(|e| panic!("set {i}: {e:?}"));
        assert_eq!(cp.mem.read_u32(SEM + 8).unwrap(), 9);
    }
    // Nothing written outside [s0, end) of the buffer.
    let mut all = vec![0u8; BUF_BYTES];
    cp.mem.read(BUF, &mut all).unwrap();
    let (o, n) = s.out_range();
    if plant != Plant::StatsSwapped {
        assert_eq!(&all[..o], &s.buffer()[..o], "set {i}: inputs changed");
    }
    assert!(
        all[o + n..].iter().all(|&b| b == 0),
        "set {i}: wrote past the end"
    );
    all[o..o + n].to_vec()
}

#[test]
fn t166a_the_probe_passes_bit_for_bit_with_the_model() {
    let mut p = Probe::default();
    for i in 0..SETS.len() {
        let out = run_set(i, Plant::None);
        p.add(cp_::judge_set(i, &out, Some(5000 + i as u64)).unwrap());
    }
    for l in p.lines("SIMULARE") {
        println!("{l}");
    }
    assert!(p.passes(), "{:#?}", p.lines("SIMULARE"));
    // No MUFU, no shared memory, no atomics: the model is the code; the prediction for the board is "all exact".
    assert_eq!(p.exact(), p.elements());
    assert_eq!(p.elements(), 64 + 2 * (64 + 257 + 768 + 768));
    assert!(p.worst_permille() <= 1000);
    assert!(p.sets.iter().all(|r| r.guards_ok));
}

#[test]
fn t166b_planted_mistakes_are_caught() {
    let cases = [
        (Plant::NoLaunch, None),
        (Plant::ShortGrid, None),
        (Plant::RowMissing, Some(Verdict::NuTrece)),
        (Plant::OtherMode, None),
        (Plant::StatsSwapped, Some(Verdict::NuTrece)),
    ];
    let mut caught = 0;
    for (plant, want) in cases {
        for (i, set) in SETS.iter().enumerate() {
            // A short grid only exists with 2+ blocks; swapped stats only matter in the LayerNorm mode.
            if plant == Plant::ShortGrid && set.c <= 256 {
                continue;
            }
            if plant == Plant::StatsSwapped && set.mode == cs::Mode::Bias {
                continue;
            }
            let out = run_set(i, plant);
            let r = cp_::judge_set(i, &out, None).unwrap();
            assert!(!r.passes(), "set {i}: plant not caught: {r:?}");
            if let Some(v) = want {
                assert_eq!(r.verdict, v, "set {i}");
            }
            caught += 1;
        }
    }
    assert!(caught >= 20, "{caught}");
}

#[test]
fn t166c_a_write_into_a_guard_or_one_ulp_too_far() {
    // A guard word written: invalid even when every column is right.
    let s = SETS[2];
    let mut out = s.model_out();
    let g = 4 * s.c as usize;
    out[g..g + 4].copy_from_slice(&0.0f32.to_bits().to_le_bytes());
    let r = cp_::judge_set(2, &out, None).unwrap();
    assert_eq!(r.verdict, Verdict::Invalid);
    assert!(!r.guards_ok);
    // One element moved far past its margin: outside.
    let mut out = s.model_out();
    let v = f32::from_le_bytes(out[0..4].try_into().unwrap());
    out[0..4].copy_from_slice(&(v + 1e-3).to_bits().to_le_bytes());
    let r = cp_::judge_set(2, &out, None).unwrap();
    assert_eq!(r.verdict, Verdict::NuTrece);
    assert_eq!(r.exact, r.elements - 1);
    // One ulp off: still inside the margin, but no longer bit for bit — the probe reports it as a count.
    let mut out = s.model_out();
    let w = u32::from_le_bytes(out[4..8].try_into().unwrap());
    out[4..8].copy_from_slice(&(w + 1).to_le_bytes());
    let r = cp_::judge_set(2, &out, None).unwrap();
    assert!(r.passes());
    assert_eq!(r.exact, r.elements - 1);
    // NaN where the board wrote nothing.
    let mut out = s.model_out();
    out[8..12].copy_from_slice(&FILL.to_le_bytes());
    assert_eq!(
        cp_::judge_set(2, &out, None).unwrap().verdict,
        Verdict::Invalid
    );
}

#[test]
fn t166d_rows_read_back_from_a_journal() {
    let mut p = Probe::default();
    for i in 0..SETS.len() {
        let out = run_set(i, Plant::None);
        p.add(cp_::judge_set(i, &out, Some(7000 + i as u64)).unwrap());
    }
    let mut j = Journal::new(256);
    p.journal(&mut j);
    assert_eq!(j.len(), p.rows().len());
    let text: String = p
        .rows()
        .iter()
        .map(|(w, v)| format!("gpu: valoare {w} = 0x{v:08x}\n"))
        .collect();
    let back = parse_colsum(&text);
    assert_eq!(back.sets, Some(SETS.len() as u32));
    assert_eq!(back.in_margin, Some(SETS.len() as u32));
    assert_eq!(back.exact, Some(p.exact()));
    assert_eq!(back.median_ns, Some(7002));
    assert_eq!(back.per_set.len(), SETS.len());
    for (i, row) in back.per_set.iter().enumerate() {
        assert_eq!(row[0], i as u32);
        assert_eq!(row[1], SETS[i].inputs_fnv());
        assert_eq!(row[2], 1);
        assert_eq!(row[5], 1);
        assert_eq!(row[6], cp_::fnv(&SETS[i].model_out()));
        assert_eq!(row[7], 7000 + i as u32);
    }
    // The expected fingerprints for the note (the board must give these if it is bit for bit the model).
    for (i, s) in SETS.iter().enumerate() {
        println!(
            "SIMULARE colsum set {} {}: intrari 0x{:08x}, iesire (model) 0x{:08x}",
            i,
            s.name,
            s.inputs_fnv(),
            cp_::fnv(&s.model_out())
        );
    }
}
