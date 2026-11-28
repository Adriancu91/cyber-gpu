//! T308 (CALCUL on the board's MEASURED schedule, predictions written BEFORE the next row): which `--mm-tc-dxdw`
//! rows with v8 / v8n / v8r end inside the 1 800 s max-speed window. Nothing here runs on a board.
//!
//! Why: os-505 (28 nov 2026, `ciclu-auto-20261128-1051.txt`, `gpu-jurnal-200.txt`) ran the row
//! `--tablou-viteza --viteza-1800 --mm-tc --mm-tc-v6 --mm-tc-v7 --mm-tc-v8 --mm-tc-v8r --mm-tc-64k --mm-tc-dxdw`:
//! 5 companions (v6, v7, v8, v8n, v8r), each plain and `@64k`. The chain ran 2 429 s; the OS guard flagged 5 dW sets
//! outside the window (v8@64k, v8n, v8n@64k, v8r, v8r@64k) — exactly the dW numbers the step needs. The command
//! centre asked (note of 11:10) for the row split, or dW first. The OS row rules (`ciclu.rs`, P6-263) need v6 before
//! v7, v7 before v8, v8 before v8r, so the companions cannot simply be dropped.
//!
//! The model: the group starts below are the first doorbell of each group of launches in `gpu-jurnal-200.txt`
//! (a group = a gap > 3 s before it), in seconds since the accepted request (`ks cerere viteza maxima = 0` at
//! 123.552 s). Per set: one head group (mm_tc + mm_tc@64k), then 10 companion groups at a fixed slot (the CPU
//! judging the program just run: qkv 39.3 s, dX 46.3 s, dW 113.8 s). Removing a companion run removes its slot.

use nvidia_ga104::gsp::perf::BoostPlan;

/// os-505, seconds since the accepted request (MEASURED, `gpu-jurnal-200.txt`).
const HEAD: [f64; 3] = [89.8, 552.2, 1_100.4];
const QKV: [f64; 10] = [
    139.7, 178.9, 218.2, 257.1, 296.4, 335.6, 375.7, 414.6, 454.1, 493.1,
];
const DX: [f64; 10] = [
    614.4, 660.4, 707.0, 753.1, 799.4, 846.1, 892.4, 938.4, 985.0, 1_031.0,
];
const DW: [f64; 10] = [
    1_231.7, 1_345.4, 1_459.3, 1_572.9, 1_686.2, 1_799.9, 1_914.5, 2_027.9, 2_141.6, 2_255.9,
];
/// The longest group run per set (last doorbell − first): qkv 6.1, dX 5.8, dW 6.5 s.
const RUN: [f64; 3] = [6.1, 5.8, 6.5];
/// Companion order in the OS chain (each plain, then `@64k`).
const NAMES: [&str; 10] = [
    "v6", "v6@64k", "v7", "v7@64k", "v8", "v8@64k", "v8n", "v8n@64k", "v8r", "v8r@64k",
];

fn sets() -> [&'static [f64; 10]; 3] {
    [&QKV, &DX, &DW]
}

/// (head → first companion, companion slot, last companion → next head) per set, from os-505.
fn shape(i: usize) -> (f64, f64, f64) {
    let s = sets()[i];
    let first = s[0] - HEAD[i];
    let slot = (s[9] - s[0]) / 9.0;
    let next = if i < 2 { HEAD[i + 1] - s[9] } else { 0.0 };
    (first, slot, next)
}

/// Starts of the kept companion runs per set, in a given set order, for a mask of kept runs (index into `NAMES`).
fn schedule(order: [usize; 3], keep: &[usize]) -> [Vec<(usize, f64)>; 3] {
    let mut out: [Vec<(usize, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut t = HEAD[0];
    for (k, &i) in order.iter().enumerate() {
        let (first, slot, _) = shape(i);
        let mut last = t;
        let mut at = t + first;
        for &c in keep {
            out[i].push((c, at));
            last = at;
            at += slot;
        }
        if k < 2 {
            // The gap to the next head: the measured one after this set; after dW (only when reordered) the
            // longest measured one (dX → dW, 69.4 s).
            let gap = if i < 2 { shape(i).2 } else { shape(1).2 };
            t = last + gap;
        }
    }
    out
}

fn inside(start: f64, run: f64) -> bool {
    BoostPlan {
        enabled: true,
        duration_s: 1_800,
        ..BoostPlan::default()
    }
    .covers_ms((start * 1000.0) as u64, (run * 1000.0) as u64)
}

fn outside(s: &[Vec<(usize, f64)>; 3]) -> Vec<(usize, &'static str)> {
    let mut v = Vec::new();
    for (i, set) in s.iter().enumerate() {
        for &(c, a) in set {
            if !inside(a, RUN[i]) {
                v.push((i, NAMES[c]));
            }
        }
    }
    v
}

const ALL: [usize; 10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
/// Only the `@64k` runs of the companions (the step runs everything at the 64 KiB step).
const ONLY_64K: [usize; 5] = [1, 3, 5, 7, 9];

/// The model rebuilds os-505 itself: every companion start within 1 s, and exactly the 5 dW sets the OS flagged.
#[test]
fn t308_model_rebuilds_os505() {
    let s = schedule([0, 1, 2], &ALL);
    for (i, want) in sets().iter().enumerate() {
        for ((_, got), w) in s[i].iter().zip(want.iter()) {
            assert!((got - w).abs() < 1.5, "{i} {got} {w}");
        }
    }
    let out = outside(&s);
    assert_eq!(
        out,
        vec![
            (2, "v8@64k"),
            (2, "v8n"),
            (2, "v8n@64k"),
            (2, "v8r"),
            (2, "v8r@64k")
        ]
    );
    println!(
        "kt c308 CALCUL: modelul reface os-505 (5 seturi dW in afara, exact cele raportate de OS)"
    );
}

/// Proposal A (one row, one new OS flag, e.g. `--mm-tc-doar-64k`): the companions only at `@64k` (mm_tc plain and
/// mm_tc@64k stay as the reference). Prediction (DEDUS): all 15 companion sets inside, the last dW ends ~1 270 s
/// after the request (~530 s of margin); the chain ~1 440 s instead of 2 429 s.
#[test]
fn t308_only_64k_fits_with_margin() {
    let s = schedule([0, 1, 2], &ONLY_64K);
    assert!(outside(&s).is_empty());
    let end = s[2].last().unwrap().1 + RUN[2];
    assert!((1_230.0..1_310.0).contains(&end), "{end}");
    println!(
        "kt c308 DEDUS: insotitorii doar la 64k: toate 15 seturi in fereastra, ultimul dW se termina la ~{end:.0} s (fereastra 1 800 s)"
    );
}

/// Proposal B (no new program order, a new OS flag `--mm-tc-dw-intai`: dW first, then qkv, then dX): dW all inside,
/// but the late dX runs fall out — worse for the step (dX of v8r is measured now, dW is not).
#[test]
fn t308_dw_first_loses_dx() {
    let s = schedule([2, 0, 1], &ALL);
    let out = outside(&s);
    assert!(out.iter().all(|&(i, _)| i == 1), "{out:?}");
    assert!(!out.is_empty());
    assert!(s[2].iter().all(|&(_, a)| inside(a, RUN[2])));
    println!(
        "kt c308 DEDUS: dW intai: dW toate in fereastra, dar {} seturi dX ies afara",
        out.len()
    );
}

/// Dropping v6 alone (would need an OS rule change: v7 without v6) is not enough: v8r@64k on dW still falls out.
#[test]
fn t308_without_v6_still_misses() {
    let s = schedule([0, 1, 2], &[2, 3, 4, 5, 6, 7, 8, 9]);
    let out = outside(&s);
    assert!(out.contains(&(2, "v8r@64k")), "{out:?}");
    println!("kt c308 DEDUS: fara v6: tot {} seturi dW afara", out.len());
}

/// Proposal C (two rows): the existing row without `--mm-tc-dxdw` keeps qkv; a second row with only the dW set
/// (a new OS flag, e.g. `--mm-tc-doar-dw`) has 10 companion runs at the dW slot from the start: inside by ~550 s.
#[test]
fn t308_dw_alone_fits() {
    let (first, slot, _) = shape(2);
    let last = HEAD[0] + first + 9.0 * slot;
    assert!(inside(last, RUN[2]));
    assert!(last + RUN[2] < 1_300.0, "{last}");
}

/// The os-505 numbers against the predictions V1–V4 (`docs/C-168_V8R.md` §3, written before the board).
#[test]
fn t308_v8r_against_v1_v4() {
    // MEASURED (os-505, ns, @64k): qkv-513, qkv-513-dx.
    const V8R_QKV: f64 = 0.981;
    const V8R_DX: f64 = 0.933;
    const V8_QKV: f64 = 1.065;
    const V8_DX: f64 = 1.037;
    const V8N_QKV: f64 = 1.015;
    const V8N_DX: f64 = 0.948;
    // V2 0.85-1.01, V3 0.83-0.99: inside.
    const { assert!(V8R_QKV >= 0.85 && V8R_QKV <= 1.01) };
    const { assert!(V8R_DX >= 0.83 && V8R_DX <= 0.99) };
    // The reading rule: ≥ ×1.05 over v8 on both → the register copies were v8's brake.
    const { assert!(V8_QKV / V8R_QKV >= 1.05 && V8_DX / V8R_DX >= 1.05) };
    // Over v8n only ×1.035 / ×1.016: small.
    let q = V8N_QKV / V8R_QKV;
    let d = V8N_DX / V8R_DX;
    assert!(
        (1.02..1.05).contains(&q) && (1.0..1.03).contains(&d),
        "{q} {d}"
    );
    println!(
        "kt c308 MASURAT os-505: V1 7/7 aceiasi biti, V2 {V8R_QKV} ms, V3 {V8R_DX} ms (in benzi); V4 dW in afara ferestrei (nemasurat); v8r/v8n qkv x{q:.3}, dX x{d:.3}"
    );
}
