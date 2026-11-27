//! T283 (CALCUL on the board's MEASURED numbers): the `mm_tc` probe with `--mm-tc-dxdw` (os-485, the automatic cycle
//! of 27 nov 2026 09:52, `ciclu-auto-20261127-0952.txt`, `gpu-jurnal-171.txt`). Nothing here runs on a board; the
//! numbers and the journal's timestamps are copied from the OS report.
//!
//! What this test establishes: the chain ran 1 611 s, the max-speed request (600 s) was accepted at 114.3 s of the
//! journal, and the board was in P8 (210 MHz core, 405 MHz memory) at the end. Every timing whose launches came after
//! 714.3 s is not a time at full speed. The bits (same as `mm_tc`) are valid for every set.

use nvidia_ga104::gsp::perf::BoostPlan;

/// The same FLOP for the three shapes: 2 * 6 144 * 2 304 * 768 (forward TB, dX no flag, dW TA).
const FLOP: f64 = 2.0 * 6144.0 * 2304.0 * 768.0;

/// The order in which the OS runs the programs on each set (`gpu_i14.rs`): `mm_tc`, then the companions in the
/// order of the probe's lines 106-113.
const NAMES: [&str; 8] = [
    "mm_tc",
    "mm_tc@64k",
    "v4",
    "v4@64k",
    "v5",
    "v5@64k",
    "v6",
    "v6@64k",
];

/// Median ns of 5 launches, as printed by the OS (MEASURED), in `NAMES` order.
const QKV: [f64; 8] = [
    21_142_976.0,
    10_921_728.0,
    4_740_736.0,
    2_624_928.0,
    4_747_072.0,
    1_869_760.0,
    3_636_064.0,
    1_260_064.0,
];
const DX: [f64; 8] = [
    21_005_056.0,
    10_833_344.0,
    4_307_680.0,
    2_522_816.0,
    4_276_576.0,
    1_619_840.0,
    38_680_608.0,
    13_221_920.0,
];
const DW: [f64; 8] = [
    21_268_000.0,
    61_798_688.0,
    46_663_360.0,
    30_368_896.0,
    46_227_328.0,
    25_236_992.0,
    39_288_320.0,
    17_753_632.0,
];

/// Journal seconds (`gpu-jurnal-171.txt`): the accepted request (`ks cerere viteza maxima = 0`), the end reading.
const REQUEST_S: f64 = 114.308;
const END_S: f64 = 1_697.601;

/// First and last doorbell (`w 0xbb0090`) of the 5 launches' group of each program, in `NAMES` order (journal
/// seconds). The groups: 465 doorbells; per set `mm_tc` = uploads (2 per 8 MiB piece) + 4 fills + 5 launches + 4
/// reads, then 13 per companion (4 + 5 + 4). For `mm_tc` the 5 launches are given; for a companion its 13.
const QKV_AT: [(f64, f64); 8] = [
    (188.0, 189.8),
    (194.7, 199.2),
    (233.1, 238.4),
    (271.9, 276.5),
    (310.3, 315.1),
    (348.6, 353.1),
    (386.7, 391.5),
    (425.3, 429.8),
];
const DX_AT: [(f64, f64); 8] = [
    (492.2, 493.9),
    (495.3, 499.9),
    (540.9, 546.3),
    (586.7, 591.3),
    (631.7, 636.6),
    (677.6, 682.2),
    (722.9, 727.5),
    (768.0, 772.6),
];
const DW_AT: [(f64, f64); 8] = [
    (848.3, 849.7),
    (851.8, 856.0),
    (963.7, 968.6),
    (1_076.3, 1_081.0),
    (1_188.4, 1_193.0),
    (1_301.0, 1_305.7),
    (1_413.4, 1_419.2),
    (1_527.2, 1_531.8),
];

// Checked at compile time (the numbers are constants): the chain was longer than the boost; the slow numbers sit
// exactly after the cut (v6 dX x9.0 the time of v5 dX, the model says <= x1); mm_tc@64k on dW x5.7 its dX time.
const _: () = assert!(END_S - REQUEST_S > 600.0);
const _: () = assert!(DX[6] / DX[4] > 8.5 && DX[7] / DX[5] > 8.0);
const _: () = assert!(DW[1] / DX[1] > 5.0);

fn plan() -> BoostPlan {
    BoostPlan {
        enabled: true,
        ..BoostPlan::default()
    }
}

fn covered(at: (f64, f64)) -> bool {
    let since = ((at.0 - REQUEST_S) * 1000.0) as u64;
    let run = ((at.1 - at.0) * 1000.0) as u64;
    plan().covers_ms(since, run)
}

fn tflops(ns: f64) -> f64 {
    FLOP / (ns * 1e-9) / 1e12
}

/// The journal's hex values agree with the report and the end state is P8.
#[test]
fn t283_numbers_and_end_state() {
    // gpu-jurnal-171.txt: `kt mm_tc qkv median ns = 0x01429dc0`, the dX / dW `mm_tc` set times.
    assert_eq!(0x0142_9dc0 as f64, QKV[0]);
    assert_eq!(0x0140_8300 as f64, DX[0]);
    assert_eq!(0x0144_8620 as f64, DW[0]);
    // End: `kd ceasul nucleului (MHz) = 0xd2`, memory `0x195`, `stare de performanta = 8` (P8); after the request
    // (115.4 s): 0x618 = 1 560 MHz, 0x251d = 9 501 MHz, P0.
    assert_eq!((0xd2, 0x195, 0x618, 0x251d), (210, 405, 1560, 9501));
    // The chain: 1 611 s, more than the boost's 600 s; the older runs today (<= 440 s) all ended in P0.
    // qkv-513 repeats os-483 within 0.4 % (mm_tc 21.148, v5@64k 1.869, v6@64k 1.263 ms).
    for (now, then) in [
        (QKV[0], 21_148_288.0),
        (QKV[5], 1_868_544.0),
        (QKV[7], 1_263_456.0),
    ] {
        assert!((now / then - 1.0).abs() < 0.004, "{now} {then}");
    }
}

/// Which timings were measured inside the 600 s window: all of qkv-513; dX up to v5@64k; nothing of dW.
#[test]
fn t283_which_timings_are_at_full_speed() {
    let expiry = REQUEST_S + 600.0;
    assert!((714.0..714.5).contains(&expiry));
    assert!(QKV_AT.iter().all(|&a| covered(a)));
    let dx: Vec<bool> = DX_AT.iter().map(|&a| covered(a)).collect();
    assert_eq!(dx, [true, true, true, true, true, true, false, false]);
    assert!(DW_AT.iter().all(|&a| !covered(a)));
    // The margin on both sides of the cut: v5@64k dX ended 32 s before, v6 dX started 8.6 s after.
    assert!(expiry - DX_AT[5].1 > 30.0 && DX_AT[6].0 - expiry > 8.0);
    // The slow numbers sit exactly after the cut: v6 dX x9.0 the time of v5 dX (the model says <= x1).
    // Core clock P0 / P8 ~ 1 785 / 210 = 8.5: the size of the jump (DEDUS, not a proof).
    let clocks = 1_785.0 / 210.0;
    assert!((8.4..8.6).contains(&clocks));
    // mm_tc on dW (848 s) came right after 10 pieces of upload kept the board busy; mm_tc@64k 3 s later is x5.7 its
    // dX time. Neither is a time at full speed (NECUNOSCUT how high the clock was).
    let names: Vec<&str> = NAMES
        .iter()
        .zip(DX_AT.iter())
        .filter(|&(_, &a)| !covered(a))
        .map(|(n, _)| *n)
        .collect();
    println!(
        "kt c283 CALCUL: cererea la {REQUEST_S} s, expirata la {expiry:.1} s; dX in afara: {names:?}; dW: toate 8 in afara"
    );
}

/// The valid numbers against the fitted line of T282 (T = a + b * path, from v5 and v6 on TB): dX v5 is the first
/// point on a third path (1 024), not used for the fit.
#[test]
fn t283_valid_dx_against_the_line() {
    // T282: the line through (1 408, 2 108) and (576, 2 108 * v6@64k / v5@64k of os-483).
    let slice_v6 = 2_108.0 * 1_263_456.0 / 1_868_544.0;
    let b = (2_108.0 - slice_v6) / (1_408.0 - 576.0);
    let a = 2_108.0 - b * 1_408.0;
    let t = |path: f64| (a + b * path).max(1_024.0);
    // v5@64k, forward (TB, path 1 408) against dX (no flag, path 1 024): the same FLOP, the same 6 full waves of
    // 192 blocks (dX 1 152 blocks, qkv 3 456 = 18 waves).
    let measured = QKV[5] / DX[5];
    let line = t(1_408.0) / t(1_024.0);
    assert!((1.15..1.16).contains(&measured), "{measured}");
    assert!(
        (measured / line - 1.0).abs() < 0.03,
        "line x{line}, board x{measured}"
    );
    assert!((tflops(DX[5]) - 13.42).abs() < 0.02, "{}", tflops(DX[5]));
    // v4@64k (2 blocks per SM) gains only x1.04 on dX: with fewer blocks the shared-memory path weighs less.
    assert!((1.03..1.05).contains(&(QKV[3] / DX[3])));
    // mm_tc (1 block per SM, latency-bound): the same time on the three paths within 1 %.
    assert!((QKV[0] / DX[0] - 1.0).abs() < 0.01);
    println!(
        "kt c283 CALCUL: v5@64k dX {:.3} ms = {:.2} TFLOP/s; inainte/dX placa x{measured:.3}, dreapta x{line:.3}",
        DX[5] / 1e6,
        tflops(DX[5])
    );
}

/// DEDUS, written before the rerun: what the same probe should show inside the window. The bands V9 / V10 written in
/// session 264 stay the ones to judge.
#[test]
fn t283_dedus_for_the_rerun() {
    let slice_v6 = 2_108.0 * 1_263_456.0 / 1_868_544.0;
    let b = (2_108.0 - slice_v6) / (1_408.0 - 576.0);
    let a = 2_108.0 - b * 1_408.0;
    let t = |path: f64| (a + b * path).max(1_024.0);
    // Anchored on today's valid v5@64k dX (1.620 ms) and os-483's v6@64k TB.
    let v6_dx = DX[5] * t(832.0) / t(1_024.0);
    let v5_dw = DX[5] * t(1_920.0) / t(1_024.0);
    let v6_dw = QKV[7] * t(1_088.0) / t(576.0);
    // dW: 432 blocks = 2.25 waves of 192 -> up to x(3 / 2.25) more if a short wave costs a full one.
    let tail = 3.0 / 2.25;
    assert!((1.45e6..1.50e6).contains(&v6_dx), "{v6_dx}");
    assert!((2.2e6..2.3e6).contains(&v5_dw), "{v5_dw}");
    assert!((1.60e6..1.66e6).contains(&v6_dw), "{v6_dw}");
    assert!((v6_dw * tail - 2.18e6).abs() < 0.03e6);
    println!(
        "kt c283 DEDUS: in fereastra: dX v6@64k ~{:.2} ms (v6/v5 x{:.2}); dW v5@64k {:.2}-{:.2} ms, v6@64k {:.2}-{:.2} ms (v6/v5 x{:.2})",
        v6_dx / 1e6,
        DX[5] / v6_dx,
        v5_dw / 1e6,
        v5_dw * tail / 1e6,
        v6_dw / 1e6,
        v6_dw * tail / 1e6,
        v5_dw / v6_dw
    );
}
