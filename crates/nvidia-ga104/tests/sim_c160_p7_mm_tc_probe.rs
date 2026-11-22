//! C-160 point 7, part four (T171), SIMULARE: the `mm_tc` probe as the OS would send it to the board — for every
//! small set of `nvidia_ga104::mm_tc_probe::SETS`, one buffer (A, B, bias; C and the guard = NaN), the cbuf + QMD
//! (with the 24 KiB of shared memory) from `launch_bytes`, one launch through the command-processor model, the bytes
//! of `read_ranges` read back and judged in the margin written in T168. The timing set `qkv-513` is too large for the
//! CPU stand-in: its banded judgement is checked on the model's output, and the banding itself on a medium product
//! run whole. The "program" runs on the CPU (`gpu_sim::cmdproc::mm_tc_on_cpu` = the os-335 adder HYPOTHESIS);
//! nothing here proves anything about the RTX 3070 Ti.

use std::time::Instant;

use gpu_core::diag::Journal;
use gpu_sim::cmdproc::{CommandProcessor, MM_TC_NEEDS, mm_tc_on_cpu};
use nvidia_ga104::mm_tc::{self as tc, MM_TC};
use nvidia_ga104::mm_tc_probe::{
    self as tp, FILL, GUARD, Probe, ProbeSet, QKV_SET, SETS, Scale, Verdict, parse_mm_tc,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const QMD: u64 = 0x20_1000;
const SEM: u64 = 0x30_0000;
const PUSH: u64 = 0x31_0000;
const BUF: u64 = 0x40_0000;

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
#[derive(Clone, Copy, PartialEq, Debug)]
enum Plant {
    None,
    /// No launch at all.
    NoLaunch,
    /// TB dropped from the flags in the cbuf (B read the other way round).
    FlagLost,
    /// One K fewer in the cbuf.
    KShort,
    /// 16 rows fewer in the cbuf (the last rows of C stay NaN).
    MShort,
    /// Two rows of the stored A swapped in the buffer.
    RowsSwapped,
}

fn cp() -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(MM_TC, mm_tc_on_cpu, MM_TC_NEEDS);
    cp.mem.write(PROG, MM_TC);
    cp
}

/// Loads set `s` (index `i` for `launch_bytes`, or a custom set with its own cbuf + QMD).
fn load(cp: &mut CommandProcessor, s: &ProbeSet, i: Option<usize>, plant: Plant) {
    let (mut cb, q) = match i {
        Some(i) => {
            let (cb, q) = tp::launch_bytes(i, PROG, CBUF, BUF).unwrap();
            (cb.to_vec(), q.to_vec())
        }
        None => {
            let a = s.args(BUF);
            let q = ComputeQmd {
                program_va: PROG,
                registers: tc::REGISTERS,
                cbuf0_va: CBUF,
                cbuf0_size: tc::CBUF_BYTES as u32,
                grid: a.grid(),
                block: tc::BLOCK,
                release: None,
            }
            .to_bytes_with_shared(&tc::SHARED)
            .unwrap();
            (a.cbuf().unwrap().to_vec(), q.to_vec())
        }
    };
    let set_u32 =
        |cb: &mut Vec<u8>, at: usize, v: u32| cb[at..at + 4].copy_from_slice(&v.to_le_bytes());
    let get_u32 = |cb: &Vec<u8>, at: usize| u32::from_le_bytes(cb[at..at + 4].try_into().unwrap());
    let mut buf = s.buffer();
    match plant {
        Plant::FlagLost => {
            let f = get_u32(&cb, 0x72c);
            set_u32(&mut cb, 0x72c, f & !tc::TB);
        }
        Plant::KShort => set_u32(&mut cb, 0x728, s.k - 1),
        Plant::MShort => set_u32(&mut cb, 0x720, s.m - 16),
        Plant::RowsSwapped => {
            // Stored rows 0 and 1 of A (whatever TA says), K or M words long.
            let w = if s.flags & tc::TA != 0 { s.m } else { s.k } as usize * 4;
            let (r0, r1) = (s.layout().a, s.layout().a + w);
            let t = buf[r0..r0 + w].to_vec();
            buf.copy_within(r1..r1 + w, r0);
            buf[r1..r1 + w].copy_from_slice(&t);
        }
        _ => {}
    }
    cp.mem.write(CBUF, &cb);
    cp.mem.write(QMD, &q);
    cp.mem.write(BUF, &buf);
    cp.mem.write_u32(SEM, 0);
}

/// Reads back the set's `read_ranges`, checking that the inputs and the bytes after the guard were not touched.
fn read_back(cp: &CommandProcessor, s: &ProbeSet, inputs_intact: bool) -> Vec<u8> {
    let l = s.layout();
    let mut all = vec![0u8; l.bytes];
    cp.mem.read(BUF, &mut all).unwrap();
    if inputs_intact {
        assert_eq!(
            &all[..l.c],
            &s.buffer()[..l.c],
            "{}: inputs changed",
            s.name
        );
    }
    assert!(
        all[l.end..].iter().all(|&b| b == 0),
        "{}: wrote past the end",
        s.name
    );
    s.read_ranges()
        .iter()
        .flat_map(|&(o, n)| all[o..o + n].to_vec())
        .collect()
}

fn run_set(i: usize, plant: Plant) -> Vec<u8> {
    let s = SETS[i];
    let mut cp = cp();
    load(&mut cp, &s, Some(i), plant);
    if plant != Plant::NoLaunch {
        submit(&mut cp).unwrap_or_else(|e| panic!("set {i}: {e:?}"));
        assert_eq!(cp.mem.read_u32(SEM + 8).unwrap(), 9);
    }
    read_back(&cp, &s, plant != Plant::RowsSwapped)
}

#[test]
fn t171a_the_small_sets_pass_bit_for_bit_with_the_model() {
    let mut p = Probe::default();
    for i in 0..QKV_SET {
        let out = run_set(i, Plant::None);
        p.add(tp::judge_set(i, &out, &[6000 + i as u64]).unwrap());
    }
    for l in p.lines("SIMULARE") {
        println!("{l}");
    }
    assert!(p.passes(), "{:#?}", p.lines("SIMULARE"));
    assert!(p.qkv_skipped());
    // The CPU stand-in IS the model: all exact here. On the board this is the ~50 % prediction.
    assert_eq!(p.exact_model(), p.elements());
    assert_eq!(
        p.elements(),
        16 * 16 + 70 * 130 + 64 * 64 + 100 * 72 + 32 * 48
    );
    // Prediction 3 (written before): ≤ 100 ‰; with the model itself it must sit well under.
    assert!(p.worst_permille() <= 100, "{}", p.worst_permille());
    assert!(p.sets.iter().all(|r| r.guards_ok));
}

#[test]
fn t171b_planted_mistakes_are_caught() {
    let cases = [
        (Plant::NoLaunch, Some(Verdict::Invalid)),
        (Plant::FlagLost, Some(Verdict::NuTrece)),
        (Plant::KShort, Some(Verdict::NuTrece)),
        (Plant::MShort, Some(Verdict::Invalid)),
        (Plant::RowsSwapped, Some(Verdict::NuTrece)),
    ];
    let mut caught = 0;
    for (plant, want) in cases {
        for (i, s) in SETS[..QKV_SET].iter().enumerate() {
            if plant == Plant::FlagLost
                && (s.flags & tc::TB == 0 || s.n as usize * s.k as usize == 0)
            {
                continue;
            }
            if plant == Plant::MShort && s.m <= 16 {
                continue;
            }
            let out = run_set(i, plant);
            let r = tp::judge_set(i, &out, &[]).unwrap();
            assert!(!r.passes(), "{plant:?} set {}: not caught: {r:?}", s.name);
            if let Some(v) = want {
                assert_eq!(r.verdict, v, "{plant:?} set {}", s.name);
            }
            caught += 1;
        }
    }
    assert!(caught >= 18, "{caught}");
}

#[test]
fn t171c_a_qmd_without_the_shared_memory_is_refused_not_guessed() {
    // The model of the command processor refuses a launch whose QMD does not declare the 24 KiB + barrier — the board
    // would compute garbage or hang; the probe's own QMD declares them (t171a).
    let s = SETS[1];
    let mut cp = cp();
    load(&mut cp, &s, Some(1), Plant::None);
    let a = s.args(BUF);
    let q = ComputeQmd {
        program_va: PROG,
        registers: tc::REGISTERS,
        cbuf0_va: CBUF,
        cbuf0_size: tc::CBUF_BYTES as u32,
        grid: a.grid(),
        block: tc::BLOCK,
        release: None,
    }
    .to_bytes()
    .unwrap();
    cp.mem.write(QMD, &q);
    assert!(submit(&mut cp).is_err());
}

#[test]
fn t171d_bands_are_the_same_tiles_as_the_whole_product() {
    // A medium product run whole through the stand-in; the banded read (rows 0..16, 64..80, 144..160) must be exactly
    // the model's banded output — so judging qkv-513 on three bands judges what the whole launch wrote there.
    static BANDS: [usize; 3] = [0, 64, 144];
    let s = ProbeSet {
        name: "160x130x40-benzi",
        m: 160,
        n: 130,
        k: 40,
        flags: tc::TB | tc::BIAS,
        seed: 99,
        scale: Scale::Qkv,
        bands: Some(&BANDS),
        repeats: 1,
    };
    let whole = ProbeSet { bands: None, ..s };
    let mut cp = cp();
    load(&mut cp, &whole, None, Plant::None);
    submit(&mut cp).unwrap();
    let all = read_back(&cp, &whole, true);
    let n = s.n as usize;
    let banded: Vec<u8> = BANDS
        .iter()
        .flat_map(|&r| all[4 * r * n..4 * (r + 16) * n].to_vec())
        .chain(all[4 * 160 * n..].iter().copied())
        .collect();
    assert_eq!(banded.len(), s.read_len());
    assert_eq!(banded, s.model_out());
}

#[test]
fn t171e_qkv_513_judged_on_its_bands() {
    let s = SETS[QKV_SET];
    assert_eq!(s.args(BUF).grid(), [36, 96, 1]);
    assert_eq!(s.args(BUF).flop(), 21_743_271_936);
    let t = Instant::now();
    let model = s.model_out();
    eprintln!(
        "T171 qkv-513: modelul pe 3 benzi in {:?} (CPU, nu placa)",
        t.elapsed()
    );
    // Prediction 4 (DEDUS): 1,3–2,7 ms. Here only the arithmetic of the rows is checked, with made-up times.
    let ns = [2_100_000, 2_000_000, 1_900_000, 2_050_000, 1_950_000];
    let r = tp::judge_set(QKV_SET, &model, &ns).unwrap();
    assert!(r.passes(), "{r:?}");
    assert_eq!(r.elements, 48 * 2304);
    assert_eq!(r.exact_model, r.elements);
    assert_eq!(r.median_ns(), Some(2_000_000));
    assert_eq!(r.gflops(), Some(10_871)); // 21,74 GFLOP / 2 ms ≈ 10,9 TFLOP/s
    assert!(r.worst_permille.unwrap() <= 100);
    // One band shifted by a row (the board's rows in the wrong place): outside.
    let n = 4 * 2304;
    let mut moved = model.clone();
    moved.copy_within(n..16 * n, 0);
    assert_eq!(
        tp::judge_set(QKV_SET, &moved, &ns).unwrap().verdict,
        Verdict::NuTrece
    );
    // The guard written: invalid.
    let mut g = model.clone();
    let at = 48 * n;
    g[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
    assert_eq!(
        tp::judge_set(QKV_SET, &g, &ns).unwrap().verdict,
        Verdict::Invalid
    );
    // The fingerprint for the note.
    println!(
        "SIMULARE mm_tc set {QKV_SET} qkv-513: intrari 0x{:08x}, iesire (model) 0x{:08x}",
        s.inputs_fnv(),
        tp::fnv(&model)
    );
}

#[test]
fn t171f_rows_read_back_from_a_journal() {
    let mut p = Probe::default();
    for i in 0..QKV_SET {
        p.add(tp::judge_set(i, &run_set(i, Plant::None), &[7000 + i as u64]).unwrap());
    }
    let q = SETS[QKV_SET];
    p.add(tp::judge_set(QKV_SET, &q.model_out(), &[1_500_000, 1_400_000, 1_600_000]).unwrap());
    let mut j = Journal::new(256);
    p.journal(&mut j);
    assert_eq!(j.len(), p.rows().len());
    let text: String = p
        .rows()
        .iter()
        .map(|(w, v)| format!("gpu: valoare {w} = 0x{v:08x}\n"))
        .collect();
    let back = parse_mm_tc(&text);
    assert_eq!(back.sets, Some(SETS.len() as u32));
    assert_eq!(back.in_margin, Some(SETS.len() as u32));
    assert_eq!(back.exact_model, Some(p.exact_model()));
    assert_eq!(back.qkv_skipped, Some(0));
    assert_eq!(back.qkv_median_ns, Some(1_500_000));
    assert_eq!(back.qkv_gflops, Some(14_495));
    assert_eq!(back.per_set.len(), SETS.len());
    for (i, row) in back.per_set.iter().enumerate() {
        assert_eq!(row[0], i as u32);
        assert_eq!(row[1], SETS[i].inputs_fnv());
        assert_eq!(row[2], 1);
        assert_eq!(row[5], 1);
    }
    for l in p.lines("SIMULARE") {
        println!("{l}");
    }
    // The expected fingerprints for the note (the board must give these if it is bit for bit the model).
    for (i, s) in SETS[..QKV_SET].iter().enumerate() {
        println!(
            "SIMULARE mm_tc set {} {}: intrari 0x{:08x}, iesire (model) 0x{:08x}",
            i,
            s.name,
            s.inputs_fnv(),
            tp::fnv(&s.model_out())
        );
        assert_eq!(back.per_set[i][6], tp::fnv(&s.model_out()));
    }
    // Without qkv: the row says so and the small sets still decide.
    let mut small = Probe::default();
    for r in p.sets.iter().take(QKV_SET) {
        small.add(r.clone());
    }
    assert!(small.passes() && small.qkv_skipped());
    assert!(small.lines("SIMULARE")[1].contains("omis"));
    let _ = (FILL, GUARD);
}
