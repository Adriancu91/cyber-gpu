//! T287 (CALCUL on the board's MEASURED schedule, prediction written BEFORE the result): which `--mm-tc-dxdw` rows
//! end inside the max-speed window. Nothing here runs on a board.
//!
//! Why: the P2zzl row (`--mm-tc --mm-tc-v5 --mm-tc-v6 --mm-tc-64k --mm-tc-dxdw`) was refused by the OS at 13:28 on
//! 27 nov 2026 (`ciclu-auto-20261127-1328-refuzat.txt`): `--mm-tc-v5` needs `--mm-tc-v4` right before it (P6-242),
//! our mistake. The command centre restarted at 14:08 with the row tested in `ciclu.rs`, which has the same 8
//! programs as os-485. This test rebuilds the os-485 schedule (journal seconds of `gpu-jurnal-171.txt`, copied in
//! T283) and places any subset of the programs on it, to say before the result what will be inside the window.
//!
//! The model (DEDUS from os-485, one run): in each set the slot after `mm_tc@64k` and after every companion is the
//! same length (the CPU judging the program just run: qkv ~38 s, dX ~45 s, dW ~112 s); the gap from the last
//! program of a set to the next set's `mm_tc` is the measured one; the run of a group of 5 launches is the measured
//! one. Removing a companion removes its two slots (the program and its `@64k`).

use nvidia_ga104::gsp::perf::BoostPlan;

/// Journal seconds of the accepted request in os-485 (`ks cerere viteza maxima = 0`).
const REQUEST_S: f64 = 114.308;

/// os-485 group starts (first doorbell of the 5 launches' group), in the OS order
/// `mm_tc, mm_tc@64k, v4, v4@64k, v5, v5@64k, v6, v6@64k` (T283, MEASURED).
const QKV: [f64; 8] = [188.0, 194.7, 233.1, 271.9, 310.3, 348.6, 386.7, 425.3];
const DX: [f64; 8] = [492.2, 495.3, 540.9, 586.7, 631.7, 677.6, 722.9, 768.0];
const DW: [f64; 8] = [
    848.3, 851.8, 963.7, 1_076.3, 1_188.4, 1_301.0, 1_413.4, 1_527.2,
];
/// The longest group run of a set (last doorbell − first, T283): qkv 5.3, dX 5.4, dW 5.8 s.
const RUN: [f64; 3] = [5.3, 5.4, 5.8];

/// The mean slot after `mm_tc@64k` (6 slots up to the last program) and the gap from the last program's start to
/// the next set's `mm_tc` (dW: none, the chain ends).
fn slots(set: &[f64; 8]) -> (f64, f64) {
    ((set[7] - set[1]) / 6.0, 0.0)
}

/// Start (journal s) of each kept program in each set, and the chain's last end. `companions` = how many companion
/// programs beside `mm_tc` (each one runs twice: plain and `@64k`).
fn schedule(companions: usize) -> ([Vec<f64>; 3], f64) {
    let sets = [&QKV, &DX, &DW];
    let gaps = [DX[0] - QKV[7], DW[0] - DX[7]];
    let mut out: [Vec<f64>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut t = QKV[0];
    let mut end = 0.0;
    for (i, set) in sets.iter().enumerate() {
        let (slot, _) = slots(set);
        let first_gap = set[1] - set[0];
        out[i].push(t);
        t += first_gap;
        out[i].push(t);
        for _ in 0..(2 * companions) {
            t += slot;
            out[i].push(t);
        }
        // `out[i]` holds mm_tc, mm_tc@64k and 2 * companions more; `t` is the last program's start.
        end = t + RUN[i];
        if i < 2 {
            t += gaps[i];
        }
    }
    (out, end)
}

fn plan(duration_s: u32) -> BoostPlan {
    BoostPlan {
        enabled: true,
        duration_s,
        ..BoostPlan::default()
    }
}

fn inside(duration_s: u32, start: f64, run: f64) -> bool {
    plan(duration_s).covers_ms(((start - REQUEST_S) * 1000.0) as u64, (run * 1000.0) as u64)
}

/// The model rebuilds os-485 itself (3 companions = v4, v5, v6): every start within 2 s.
#[test]
fn t287_model_rebuilds_os485() {
    let (s, end) = schedule(3);
    for (got, want) in s.iter().zip([&QKV, &DX, &DW]) {
        assert_eq!(got.len(), 8);
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 2.0, "{g} {w}");
        }
    }
    assert!((end - (1_527.2 + 5.8)).abs() < 2.0, "{end}");
}

/// Prediction for the 14:08 restart (`--tablou-viteza --mm-tc --mm-tc-v4 --mm-tc-v5 --mm-tc-v6 --mm-tc-64k
/// --mm-tc-dxdw`, the same 8 programs as os-485, the request 600 s): the same cut. Inside: all of qkv and dX up to
/// v5@64k; outside: dX v6, v6@64k and all 8 of dW → the os-489 guard should flag 10 timings (DEDUS).
#[test]
fn t287_restart_1408_same_cut_as_os485() {
    let (s, _) = schedule(3);
    let outside: Vec<usize> = (0..3)
        .map(|i| s[i].iter().filter(|&&a| !inside(600, a, RUN[i])).count())
        .collect();
    assert_eq!(outside, vec![0, 2, 8]);
    println!(
        "kt c287 DEDUS: randul repornit 14:08 (8 programe) — in afara ferestrei de 600 s: qkv 0, dX 2 (v6, v6@64k), dW 8"
    );
}

/// The short row (P6-251 is on `main` of the OS, not yet in a Release: `--mm-tc --mm-tc-v6 --mm-tc-64k
/// --mm-tc-dxdw`, 4 programs): still too long for 600 s — dW v6@64k, the number we need, ends ~30 s after.
#[test]
fn t287_short_row_still_misses_600() {
    let (s, end) = schedule(1);
    let dw_last = *s[2].last().unwrap();
    assert!((735.0..755.0).contains(&end), "{end}");
    assert!(!inside(600, dw_last, RUN[2]));
    // dW v6 (plain) is inside, by ~80 s; only its @64k misses.
    assert!(inside(600, s[2][2], RUN[2]));
    let over = end - (REQUEST_S + 600.0);
    assert!((20.0..40.0).contains(&over), "{over}");
    println!(
        "kt c287 DEDUS: randul scurt (mm_tc + v6) se termina la ~{end:.0} s; fereastra 600 s expira la 714,3 s; dW v6@64k iese cu ~{over:.0} s"
    );
}

/// With the request held 1 800 s (P2zzm, only with Adrian's "da" in DECIZII.md) every row with up to 3 companions
/// fits (the os-485 / 14:08 row included); 4 companions (e.g. v4, v5, v6, v7) would not (~1 920 s).
#[test]
fn t287_1800_fits_up_to_three_companions() {
    for companions in 1..=3 {
        let (s, end) = schedule(companions);
        assert!(
            inside(1_800, *s[2].last().unwrap(), RUN[2]),
            "{companions} {end}"
        );
    }
    let (s4, end4) = schedule(4);
    assert!(!inside(1_800, *s4[2].last().unwrap(), RUN[2]), "{end4}");
    // The P2zzm row (`--mm-tc --mm-tc-v6 --mm-tc-v7 ...`, 2 companions): ~1 140 s, ~770 s of margin.
    let (_, end) = schedule(2);
    assert!((1_120.0..1_160.0).contains(&end), "{end}");
    println!(
        "kt c287 DEDUS: cu 1 800 s randul P2zzm (v6, v7) se termina la ~{end:.0} s (expira la 1 914 s); 4 insotitori ~{end4:.0} s nu incap"
    );
    // And no row with at least one companion fits 600 s.
    for companions in 1..=4 {
        let (s, _) = schedule(companions);
        assert!(!inside(600, *s[2].last().unwrap(), RUN[2]));
    }
}

/// The result of the 14:08 restart (the cycle of 14:42, `ciclu-auto-20261127-1442.txt`, MEASURED by the OS, lines
/// 117-127 and 203): the os-489 guard flagged exactly the 10 sets predicted above, the chain ran 1 597 s after the
/// accepted request. The inside numbers repeat os-485 (v5@64k dX 1.617 ms against 1.620 ms).
#[test]
fn t287_os_1442_confirms_the_prediction() {
    // The guard's rows, seconds since the accepted request (start of the 5 launches).
    const FLAGGED_DX: [f64; 2] = [617.3, 662.9];
    const FLAGGED_DW: [f64; 8] = [
        741.7, 746.9, 860.2, 973.1, 1_086.3, 1_199.5, 1_312.9, 1_426.2,
    ];
    const FLAGGED_TOTAL: usize = 10;
    const CHAIN_AFTER_REQUEST_S: f64 = 1_597.0;
    assert_eq!(FLAGGED_DX.len() + FLAGGED_DW.len(), FLAGGED_TOTAL);
    let (s, end) = schedule(3);
    // The model's starts for the same programs (since the request) are 8-14 s EARLIER than the board's, every one
    // (the same direction: the short row would end even later than ~747 s, still outside).
    for (got, want) in s[1][6..].iter().zip(FLAGGED_DX.iter()) {
        let d = (got - REQUEST_S) - want;
        assert!((-15.0..0.0).contains(&d), "{got} {want}");
    }
    for (got, want) in s[2].iter().zip(FLAGGED_DW.iter()) {
        let d = (got - REQUEST_S) - want;
        assert!((-15.0..0.0).contains(&d), "{got} {want}");
    }
    assert!(((end - REQUEST_S) - FLAGGED_DW[7]).abs() < 15.0);
    const { assert!(CHAIN_AFTER_REQUEST_S > 600.0) };
    // Inside the window, v5@64k on dX repeats os-485 within 0.3 %.
    const { assert!((1_616_704.0_f64 / 1_619_840.0 - 1.0).abs() < 0.003) };
    println!(
        "kt c287 MASURAT 14:42: 10 seturi in afara ferestrei, exact cele prezise (dX v6, v6@64k; dW toate 8)"
    );
}
