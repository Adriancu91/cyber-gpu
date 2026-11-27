//! T282 (CALCUL on the board's MEASURED numbers): the `mm_tc` probe with `mm_tc_v6` (os-483, the automatic cycle of
//! 27 nov 2026 09:02, `ciclu-auto-20261127-0902.txt`, `gpu-jurnal-169.txt`). Nothing here runs on a board; the numbers
//! are copied from the OS report and judged against the bands written before the run (V5-V7 in
//! `docs/C-157_MM_TC_V6_PLAN.md`, V7b in `docs/C-157_MM_TC_V6_IN_PAS.md`).

/// qkv-513: m 6 144, n 2 304, k 768 (the `TB` case of the #513 step's forward products).
const QKV_FLOP: f64 = 2.0 * 6144.0 * 2304.0 * 768.0;

/// Median ns of the 5 qkv-513 launches, as printed by the OS (MEASURED).
const MM_TC: f64 = 21_148_288.0;
const V4: f64 = 4_740_896.0;
const V4_64K: f64 = 2_630_496.0;
const V5: f64 = 4_744_416.0;
const V5_64K: f64 = 1_868_544.0;
const V6: f64 = 3_640_000.0;
const V6_64K: f64 = 1_263_456.0;

/// Block-slice clocks of `mm_tc_v5@64k` at os-476 (T275) and the shared-memory path per block and slice (T279).
const SLICE_V5: f64 = 2_108.0;
const PATH_V5_TB: f64 = 1_408.0;
const PATH_V6_TB: f64 = 576.0;

fn tflops(ns: f64) -> f64 {
    QKV_FLOP / (ns * 1e-9) / 1e12
}

/// The numbers are the OS's own (the journal's hex values agree) and the old programs repeat os-476.
#[test]
fn t282_numbers_repeat_os476() {
    // gpu-jurnal-169.txt: `kt mm_tc_v6@64k qkv median ns = 0x00134760`, `kt mm_tc qkv median ns = 0x0142b280`.
    assert_eq!(0x0013_4760 as f64, V6_64K);
    assert_eq!(0x0142_b280 as f64, MM_TC);
    // os-476: v5@64k 1,868 ms, v4@64k 2,629 ms -> today within 0.1 %: same conditions (DEDUS).
    assert!((V5_64K / 1.868e6 - 1.0).abs() < 0.001);
    assert!((V4_64K / 2.629e6 - 1.0).abs() < 0.001);
    // v4 and v5 without 64 KiB: the same time (1-2 blocks per SM, occupancy is not the difference there).
    assert!((V5 / V4 - 1.0).abs() < 0.001);
    assert!((tflops(V6_64K) - 17.20).abs() < 0.02, "{}", tflops(V6_64K));
    println!(
        "kt c282 CALCUL: qkv-513 mm_tc {:.2}, v4@64k {:.2}, v5@64k {:.2}, v6 {:.2}, v6@64k {:.2} TFLOP/s",
        tflops(MM_TC),
        tflops(V4_64K),
        tflops(V5_64K),
        tflops(V6),
        tflops(V6_64K)
    );
}

/// The verdicts against the bands written before the run.
#[test]
fn t282_against_the_written_bands() {
    let ms = V6_64K / 1e6;
    // V7 (plan, written on the wrong label: "no transpose"): 1.70-1.87 ms -> missed, faster.
    assert!(ms < 1.70, "V7: sub banda (mai repede)");
    // V7b (written after the label was corrected to TB): 1.44-1.78 ms (x1.05-1.30) -> missed, faster.
    let gain = V5_64K / V6_64K;
    assert!(ms < 1.44 && gain > 1.30, "V7b: sub banda, x{gain}");
    assert!((1.47..1.49).contains(&gain), "{gain}");
    // Without 64 KiB the gain is smaller: x1.30 (1-2 blocks per SM) against x1.48 (4 blocks per SM).
    let gain_low = V5 / V6;
    assert!((1.29..1.31).contains(&gain_low), "{gain_low}");
    assert!(gain > gain_low + 0.15);
    // v6 against the first tensor program (mm_tc, 24 KiB, 64 registers): x16.7.
    assert!((16.6..16.8).contains(&(MM_TC / V6_64K)));
}

/// The two margin models of `docs/C-157_CITIRI_CONFLICTE.md` §4 against the board, and the line through the two
/// measured points (v5, v6), used below only as DEDUS for what is still to run.
#[test]
fn t282_margin_models_and_fitted_line() {
    let slice_v6 = SLICE_V5 * V6_64K / V5_64K;
    assert!((1_420.0..1_430.0).contains(&slice_v6), "{slice_v6}");
    // "overlapped": T = max(2 108, path) -> no gain; "serial": T = max(1 024, 700 + path) -> 1 276.
    let overlapped = SLICE_V5.max(PATH_V6_TB);
    let serial = 1_024.0f64.max(700.0 + PATH_V6_TB);
    assert!(slice_v6 < overlapped && slice_v6 > serial);
    // Share of the serial model's gain that showed up on the board.
    let share = (SLICE_V5 - slice_v6) / (SLICE_V5 - serial);
    assert!((0.80..0.84).contains(&share), "{share}");
    // The line through (1 408, 2 108) and (576, slice_v6): T = a + b * path.
    let b = (SLICE_V5 - slice_v6) / (PATH_V5_TB - PATH_V6_TB);
    let a = SLICE_V5 - b * PATH_V5_TB;
    assert!(
        (0.81..0.83).contains(&b) && (945.0..960.0).contains(&a),
        "a {a} b {b}"
    );
    // The tensor floor (1 024) sits below the intercept: the rest (~950) is not the shared-memory path.
    assert!(a < 1_024.0 && a > 900.0);
    println!(
        "kt c282 CALCUL: felia v6@64k {slice_v6:.0} tacti (v5 2108; suprapus 2108, in serie {serial:.0}); {:.0} % din castigul modelului in serie; dreapta T = {a:.0} + {b:.3} x cale",
        share * 100.0
    );
}

/// DEDUS from the fitted line (written before os-485 and before the step with v6): the dX / dW sets, `mm_tc_v7` on
/// qkv-513, and the #513 step with v6 at lot 2 and lot 6. The bands written earlier (V8, V9, V10, W5) stay as written.
#[test]
fn t282_dedus_next_runs() {
    let slice_v6 = SLICE_V5 * V6_64K / V5_64K;
    let b = (SLICE_V5 - slice_v6) / (PATH_V5_TB - PATH_V6_TB);
    let a = SLICE_V5 - b * PATH_V5_TB;
    let t = |path: f64| (a + b * path).max(1_024.0);
    // Paths per block and slice (T275 + T279): no flag (dX) v5 1 024 / v6 832; TA (dW) v5 1 920 / v6 1 088; v7 TB 256.
    let dx = t(1_024.0) / t(832.0);
    let dw = t(1_920.0) / t(1_088.0);
    let dw_over_dx_v5 = t(1_920.0) / t(1_024.0);
    assert!((1.09..1.11).contains(&dx), "dX {dx}");
    assert!((1.36..1.38).contains(&dw), "dW {dw}");
    assert!(dw_over_dx_v5 > 1.15, "V10 b: {dw_over_dx_v5}");
    let v7 = slice_v6 / t(256.0);
    let v7_ms = V6_64K / 1e6 / v7;
    assert!((1.21..1.24).contains(&v7), "v7/v6 {v7}");
    // The step: three equal thirds of FLOP (forward TB, dX, dW).
    let step_v5 = SLICE_V5 + t(1_024.0) + t(1_920.0);
    let step_v6 = slice_v6 + t(832.0) + t(1_088.0);
    let mm_gain = step_v5 / step_v6;
    assert!((1.30..1.32).contains(&mm_gain), "{mm_gain}");
    // The model's v5 step average against os-481 (11.71 TFLOP/s in the step, 11.63 in the probe): within 3 %.
    let avg_v5 = step_v5 / 3.0;
    assert!((avg_v5 / SLICE_V5 - 11.63 / 11.71 * 1.0).abs() < 0.03);
    // os-481 MEASURED: lot 6 board 203.061 ms, of which mm_tc_v5 153.451; lot 2 board 80.159, mm_tc_v5 56.153.
    let board = |total: f64, mm: f64, g: f64| total - mm + mm / g;
    let sps6 = |ms: f64| 6.0 * 512.0 / (ms / 1e3);
    let l6 = board(203.061, 153.451, mm_gain);
    let (l6_lo, l6_hi) = (board(203.061, 153.451, 1.40), board(203.061, 153.451, 1.20));
    let l2 = board(80.159, 56.153, mm_gain);
    assert!((165.0..168.5).contains(&l6), "{l6}");
    assert!((18_200.0..18_700.0).contains(&sps6(l6)), "{}", sps6(l6));
    assert!((158.0..160.5).contains(&l6_lo) && (176.5..178.5).contains(&l6_hi));
    assert!((66.0..68.0).contains(&l2), "{l2}");
    println!(
        "kt c282 DEDUS: dX v6/v5 x{dx:.2}, dW v6/v5 x{dw:.2} (v5 dW/dX x{dw_over_dx_v5:.2}); v7/v6 qkv-513 x{v7:.2} = {v7_ms:.2} ms; pasul cu v6: inmultirile x{mm_gain:.2} -> lot 6 placa {l6:.1} ms ({l6_lo:.0}-{l6_hi:.0}) = {:.0} simb/s, lot 2 {l2:.1} ms",
        sps6(l6)
    );
}
