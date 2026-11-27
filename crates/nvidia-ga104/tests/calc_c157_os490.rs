//! T289 (CALCUL on the board's MEASURED numbers): os-490, the `mm_tc` probe with `mm_tc_v6` / `mm_tc_v7` on the
//! forward, dX and dW sets of qkv-513, run with the 1 800 s boost request approved by Adrian (cycle 15:20,
//! `ciclu-auto-20261127-1520.txt`). Checked against the bands written before (W6–W8 in `docs/C-157_CITIRI_CONFLICTE.md`
//! §3, the window estimate in P2zzm, T283's estimates). Nothing here runs on a board; the numbers are copied from the
//! OS report. Also the DEDUS step estimate P2 of `docs/C-157_OS490_V7_IN_PAS.md` §3.

/// 2·m·n·k of each qkv-513 set (the same work: 6 144 × 768 × 2 304 in some order).
const FLOP: f64 = 2.0 * 6144.0 * 768.0 * 2304.0;

/// Medians of 5 launches (ns) at the 64 KiB step: (forward `TB | BIAS`, dX no flag, dW `TA`).
const V6: [f64; 3] = [1_256_544.0, 1_380_128.0, 1_962_880.0];
const V7: [f64; 3] = [1_194_752.0, 1_147_488.0, 1_464_352.0];
const TC: [f64; 3] = [21_137_760.0, 21_014_464.0, 21_242_048.0];

#[test]
fn t289_window_and_bands() {
    // The chain ran 1 190 s after the accepted request (1 800 s): 0 sets outside; P2zzm estimated ~1 150 s.
    let (ran, held, est) = (1_190.0f64, 1_800.0f64, 1_150.0f64);
    assert!(ran < held);
    assert!((ran - est).abs() / est < 0.05);
    let r: Vec<f64> = (0..3).map(|i| V6[i] / V7[i]).collect();
    // W6 dX ×1,05–1,40; W7 dW ×1,10–1,60; W8 order dW > dX > forward.
    assert!((1.05..=1.40).contains(&r[1]), "W6 {}", r[1]);
    assert!((1.10..=1.60).contains(&r[2]), "W7 {}", r[2]);
    assert!(r[2] > r[1] && r[1] > r[0], "W8");
    // The threshold of §4: W7 above ×1,10 → the shared-memory path was the brake on dW too.
    assert!(r[2] > 1.10);
    // T283: dX v6@64k ~1,48 ms (came out 7 % faster), dW v6@64k 1,63–2,18 ms (inside).
    assert!(((V6[1] / 1e6) - 1.48).abs() / 1.48 < 0.08);
    assert!((1.63..=2.18).contains(&(V6[2] / 1e6)));
    // v7@64k is the fastest on all three sets; ×14,5–18,3 over plain mm_tc.
    let over_tc: Vec<f64> = (0..3).map(|i| TC[i] / V7[i]).collect();
    assert!(over_tc.iter().all(|x| (14.0..19.0).contains(x)));
    let tf: Vec<f64> = V7.iter().map(|ns| FLOP / ns / 1e3).collect();
    // dW still ×1,28 slower than dX at the same work.
    let dw_dx = V7[2] / V7[1];
    assert!((1.25..1.30).contains(&dw_dx));
    println!(
        "kt c289 os-490 CALCUL: fereastra {ran} / {held} s (estimat {est}); v7/v6 inainte x{:.3}, dX x{:.3} (W6), dW x{:.3} (W7); v7@64k {:.2} / {:.2} / {:.2} TFLOP/s; dW / dX v7 x{dw_dx:.3}",
        r[0], r[1], r[2], tf[0], tf[1], tf[2]
    );
}

/// V9 (os-485): dX v6 / v5 ×1,00–1,10 — v5@64k dX 1,620 ms was in os-485's window, v6@64k dX now 1,380 ms: ×1,17 (two
/// different runs; outside the band, on the fast side).
#[test]
fn t289_v9_across_runs() {
    let v5_dx = 1_620_000.0f64;
    let x = v5_dx / V6[1];
    assert!(x > 1.10 && x < 1.20, "{x}");
    println!(
        "kt c289 V9 CALCUL: dX v6/v5 x{x:.3} (banda 1,00-1,10: NECONFIRMATA, mai repede; doua rulari)"
    );
}

/// P2 (DEDUS): the #513 step at lot 6 with v7 in place of v5. os-487: board 200,77 ms, `mm_tc_v5` 75,4 %.
#[test]
fn t289_p2_step_estimate() {
    let board = 200.77f64;
    let mm5 = board * 0.754;
    let rest = board - mm5;
    // qkv shape: v5@64k forward 1,869 (os-483), dX 1,620 (os-485), dW 2,28–3,05 (T283, never measured at speed).
    let v7_sum: f64 = V7.iter().sum::<f64>() / 1e6;
    let lo = (1.869 + 1.620 + 2.28) / v7_sum;
    let hi = (1.869 + 1.620 + 3.05) / v7_sum;
    assert!((1.50..1.53).contains(&lo) && (1.70..1.73).contains(&hi));
    // Band widened at the bottom to ×1,35 (other shapes in the step).
    let (fast, slow) = (rest + mm5 / 1.72, rest + mm5 / 1.35);
    assert!((136.0..139.0).contains(&fast) && (160.0..163.0).contains(&slow));
    let sps = |ms: f64| 3072.0 / (ms / 1e3);
    println!(
        "kt c289 P2 DEDUS: forma qkv v5/v7 x{lo:.2}-{hi:.2}; pasul lot 6 placa {fast:.0}-{slow:.0} ms = {:.0}-{:.0} simb/s",
        sps(slow),
        sps(fast)
    );
}
