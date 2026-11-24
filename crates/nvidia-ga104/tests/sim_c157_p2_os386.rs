//! C-157 point 2 — T205, SIMULARE + arithmetic: why the whole training step of os-386 took ~370 ms while the board took
//! 3.65 ms. Predictions written before the code: `docs/C-157_P2_OS386_PASUL_INTREG.md` (J1–J3).
//!
//! - J1: the 22 journal rows every step of `train_mini_resident` leaves on the stick (os-386, `gpu-jurnal-069.txt`),
//!   replayed through `gpu_core::diag::Journal` with a counting live destination: without a routine stretch every row
//!   is `Flush::Now` (one stick write each); with `kronos_fast::gen_routine` + `flush_live()` per step none is; with
//!   the routine but without `flush_live()` the 64-row cap forces a write every ~3 steps.
//! - J2: the VRAM words the driver writes (and reads back) and reads per step on the resident plan of `antrenare mini`,
//!   plus the host CPU time of `rebatch` + `step` + `pack` (printed only: the host is not the workstation).
//!
//! Nothing here runs on the RTX 3070 Ti.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

use gpu_compute::train::{AdamW, Arch, Recipe};
use gpu_core::diag::{Event, Flush, Journal, LiveSink};
use nvidia_ga104::kronos_fast::{GEN_MAX_PENDING, gen_routine};
use nvidia_ga104::train_step::{self as ts, Plan};
use nvidia_ga104::transfer;

/// The OS's notes repeated with the routine writes (`gpu_i14.rs`, `GEN_NOTES`), by text.
const NOTES: [&str; 2] = [
    "C-55 gpu lant: fereastra cu Adrian (P6-42)",
    "soneria canalului de lucru (I10): scriere fara recitire",
];
const WINDOW: u64 = 0x1700;
const DOORBELL: u64 = 0xbb_0090;
const TLB: [u64; 3] = [0xb8_30a0, 0xb8_30a4, 0xb8_30b0];

fn w(offset: u64) -> Event {
    Event::MmioWrite {
        bar: 0,
        offset,
        before: 0,
        written: 1,
        after: 1,
    }
}

/// One step's rows, in the order of os-386 (`gpu-jurnal-069.txt`, e.g. 145.993362 … 146.340727).
fn step_rows() -> Vec<Event> {
    let note = Event::Note(NOTES[0]);
    let mut v = Vec::new();
    // the flush of the collected writes: the window moved 5 times
    v.push(note);
    v.extend((0..5).map(|_| w(WINDOW)));
    // the TLB invalidate
    v.push(note);
    v.extend(TLB.iter().map(|&o| w(o)));
    // the doorbell
    v.push(Event::Note(NOTES[1]));
    v.push(w(DOORBELL));
    // the wait for the last record
    v.push(note);
    v.extend((0..3).map(|_| w(WINDOW)));
    // the per-row losses, then the norm (one window each)
    for _ in 0..2 {
        v.push(note);
        v.extend((0..2).map(|_| w(WINDOW)));
    }
    v
}

static NOW: AtomicU32 = AtomicU32::new(0);
static LATER: AtomicU32 = AtomicU32::new(0);
static FLUSHES: AtomicU32 = AtomicU32::new(0);

fn line(_: &Event, f: Flush) {
    match f {
        Flush::Now => NOW.fetch_add(1, Ordering::SeqCst),
        Flush::Later => LATER.fetch_add(1, Ordering::SeqCst),
    };
}
fn flush() {
    FLUSHES.fetch_add(1, Ordering::SeqCst);
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Way {
    /// os-386: no routine stretch.
    Today,
    /// The routine of a generation + `flush_live()` once per step (as once per token in `kronos --gpu`).
    RoutinePerStep,
    /// The routine without `flush_live()`.
    RoutineNoFlush,
}

/// (rows handed `Now`, rows held, `flush` calls) for 10 steps.
fn replay(way: Way) -> (u32, u32, u32) {
    NOW.store(0, Ordering::SeqCst);
    LATER.store(0, Ordering::SeqCst);
    FLUSHES.store(0, Ordering::SeqCst);
    let mut j = Journal::with_live_sink(4096, LiveSink { line, flush });
    if way != Way::Today {
        j.set_routine(Some(gen_routine(&NOTES)));
    }
    for _ in 0..10 {
        for e in step_rows() {
            j.push(e);
        }
        if way == Way::RoutinePerStep {
            j.flush_live();
        }
    }
    let r = (
        NOW.load(Ordering::SeqCst),
        LATER.load(Ordering::SeqCst),
        FLUSHES.load(Ordering::SeqCst),
    );
    j.set_routine(None);
    r
}

#[test]
fn t205_j1_journal_rows_per_step() {
    assert_eq!(step_rows().len(), 22, "os-386: 22 de randuri pe pas");
    assert_eq!(GEN_MAX_PENDING, 64);
    // All three in one test: the counters are process-wide.
    let today = replay(Way::Today);
    let per_step = replay(Way::RoutinePerStep);
    let no_flush = replay(Way::RoutineNoFlush);
    println!("T205 J1 (SIMULARE): 10 pasi x 22 randuri");
    println!(
        "  azi (os-386, fara rutina):      Now {:3}, tinute {:3}, goliri {}",
        today.0, today.1, today.2
    );
    println!(
        "  rutina + flush_live pe pas:      Now {:3}, tinute {:3}, goliri {}",
        per_step.0, per_step.1, per_step.2
    );
    println!(
        "  rutina fara flush_live:          Now {:3}, tinute {:3}, goliri {}",
        no_flush.0, no_flush.1, no_flush.2
    );
    assert_eq!(today, (220, 0, 0));
    assert_eq!(per_step, (0, 220, 10));
    // Rows 65, 130 and 195 go out `Now` (each taking the 64 held before it to the stick).
    assert_eq!(no_flush, (3, 217, 0));
    // os-386 measured ~16.5 ms per `Now` row (stick write + serial + screen): 22 rows ≈ 363 ms per step.
    let ms = today.0 as f64 / 10.0 * 16.5;
    assert!((330.0..=400.0).contains(&ms), "{ms}");
}

const MINI: Arch = Arch {
    vocab: 16_384,
    ctx: 32,
    n_emb: 64,
    heads: 4,
    layers: 2,
};
const B: usize = 2;
const T: usize = 32;
const HEAP: u64 = nvidia_ga104::mmu::VA_BASE + 0x100_0000;
const ZONE: u64 = nvidia_ga104::mmu::VA_BASE + 0x20_0000;

struct Lcg(u64);
impl Lcg {
    fn sym(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) % MINI.vocab as u64) as u32
    }
}

#[test]
fn t205_j2_driver_words_per_step() {
    let mut g = Lcg(0x0386);
    let batches: Vec<(Vec<u32>, Vec<u32>)> = (0..10)
        .map(|_| {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for _ in 0..B {
                let row: Vec<u32> = (0..=T).map(|_| g.sym()).collect();
                x.extend_from_slice(&row[..T]);
                y.extend_from_slice(&row[1..]);
            }
            (x, y)
        })
        .collect();
    let params: Vec<Vec<f32>> = MINI
        .tensors()
        .iter()
        .map(|t| vec![0.01f32; t.elements()])
        .collect();
    let adam = AdamW::new(&params, 1e-3, 0.9, 0.999, 1e-8, 0.1);
    let plan0 = Plan::new(MINI, B, T, &batches[0].0, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    let prog_va =
        |p: ts::Prog| nvidia_ga104::mmu::VA_BASE + 0x40_0000 + 0x1_0000 * p.index() as u64;
    let (mut cpu_us, mut writes, mut reads) = (Vec::new(), Vec::new(), Vec::new());
    for (k, (x, y)) in batches.iter().enumerate() {
        let t0 = Instant::now();
        let plan = plan0.rebatch(x).unwrap();
        let up: usize = if k > 0 {
            plan.batch_uploads(x, y)
                .unwrap()
                .iter()
                .map(|(_, b)| b.len())
                .sum()
        } else {
            0
        };
        let list = plan.step(Recipe::AdamW513, &adam, k as u32 + 1).unwrap();
        let (_zone, qmds) = ts::pack(&list, prog_va, ZONE).unwrap();
        cpu_us.push(t0.elapsed().as_micros() as u64);
        let n = list.len();
        assert_eq!(qmds.len(), n);
        assert_eq!(n, plan0.launches_per_step(Recipe::AdamW513));
        let [(_, lr_n), (_, res_n)] = plan.step_reads();
        // Written (and read back by the OS's flush): the records zeroed, the batch, the pushbuffer.
        let wr = 4 * (n + 1) + up / 4 + transfer::sequence_words_fresh(n);
        // Read: every record (`run_records`, all = true), the per-row losses, the norm area.
        let rd = 4 * (n + 1) + (lr_n / 4) as usize + (res_n / 4) as usize;
        if k > 0 {
            writes.push(wr);
            reads.push(rd);
        }
    }
    let n = plan0.launches_per_step(Recipe::AdamW513);
    // The batch's segment lists change length with the batch: the worst step counts.
    let (wr, rd) = (*writes.iter().max().unwrap(), *reads.iter().max().unwrap());
    assert!(reads.iter().all(|&x| x == rd), "citirile nu depind de lot");
    println!(
        "  scrise pe pas intre {} si {}",
        writes.iter().min().unwrap(),
        wr
    );
    let mut c = cpu_us.clone();
    c.sort_unstable();
    let driver_us = wr as f64 * 0.66 + rd as f64 * 0.5;
    println!("T205 J2 (aritmetica, planul rezident al lui antrenare mini): {n} lansari pe pas");
    println!(
        "  cuvinte VRAM scrise + recitite pe pas: {wr} (pushbuffer {}, inregistrari {}, lotul {})",
        transfer::sequence_words_fresh(n),
        4 * (n + 1),
        wr - transfer::sequence_words_fresh(n) - 4 * (n + 1)
    );
    println!("  cuvinte VRAM citite pe pas: {rd}");
    println!("  la 0,66 us / 0,5 us pe cuvant: ~{:.0} us", driver_us);
    println!(
        "  procesorul gazdei (rebatch + step + pack), median: {} us (doar orientativ)",
        c[c.len() / 2]
    );
    assert_eq!(n, 75);
    assert!((1400..=1700).contains(&wr), "{wr}");
    assert!((340..=420).contains(&rd), "{rd}");
    assert!((900.0..=1400.0).contains(&driver_us), "{driver_us}");
}
