//! T281 (CALCUL on the board's MEASURED numbers): os-481, the #513 step with `mm_tc_v5` (note C-157_P2zzf), lot 2
//! (`ciclu-auto-20261127-0718.txt`) and lot 6 with two slots (`ciclu-auto-20261127-0749.txt`, `gpu-jurnal-165.txt`).
//! Nothing here runs on a board; the numbers are copied from the OS reports and checked against the bands written
//! before the run (P2zzf: lot 2 78-84 ms, lot 6 202-222 ms).

/// Lot 2: the board time against the band, and the `mm_tc_v5` gain over os-473 (`mm_tc_v4`) once the clock is
/// taken out (os-481 lot 2 ran at 1 815 MHz, os-473 at 1 935 MHz).
#[test]
fn t281_lot2_band_and_clock() {
    let (board, mm_v5, clk) = (80_159.0f64, 56_153.0f64, 1_815.0f64);
    let (board_473, mm_v4, clk_473) = (97_076.0f64, 73_921.0f64, 1_935.0f64);
    assert!(
        (78_000.0..=84_000.0).contains(&board),
        "P2zzf lot 2: in banda scrisa inainte"
    );
    // attn_fwd_tile3 (unchanged program): 6 218 us at 1 935 MHz, 6 613 us at 1 815 MHz -> the clock explains it.
    let attn = 6_613.0f64 / 6_218.0;
    let clock = clk_473 / clk;
    assert!((attn - clock).abs() < 0.01, "attn {attn} clock {clock}");
    // mm_tc_v5 at the same clock as os-473: the step gain = the probe gain (os-476: x1.407).
    let raw = mm_v4 / mm_v5;
    let same_clock = mm_v4 / (mm_v5 / clock);
    assert!((1.30..1.33).contains(&raw), "{raw}");
    assert!((1.38..1.43).contains(&same_clock), "{same_clock}");
    // The whole board at 1 935 MHz (DEDUS: everything scales with the clock, as attn_fwd_tile3 did).
    let board_1935 = board / clock;
    assert!((74_000.0..76_000.0).contains(&board_1935), "{board_1935}");
    // FLOP of the 147 products at lot 2 = 1.797 TFLOP / 3 (os-473 note: lot 6 = 1.797 TFLOP).
    let tflops = 1.797e12 / 3.0 / (mm_v5 * 1e-6) / 1e12;
    assert!((10.5..10.8).contains(&tflops), "{tflops}");
    println!(
        "kt c281 lot 2 CALCUL: placa {:.2} ms (banda 78-84), mm_tc_v5 x{raw:.3} fata de os-473 (x{same_clock:.3} la acelasi ceas), {tflops:.2} TFLOP/s; la 1935 MHz placa ~{:.1} ms (os-473 {:.2})",
        board / 1e3,
        board_1935 / 1e3,
        board_473 / 1e3
    );
}

/// Lot 6 with two slots: the board is in the band, but the whole step is held by the dashboard written to the
/// stick after every step (11 values, ~23-24 ms each, one stick write per value; MEASURED in `gpu-jurnal-165.txt`).
#[test]
fn t281_lot6_board_and_dashboard() {
    let (board, mm_v5, full) = (203_061.0f64, 153_451.0f64, 274_281.0f64);
    assert!(
        (202_000.0..=222_000.0).contains(&board),
        "P2zzf lot 6: in banda scrisa inainte"
    );
    let tflops = 1.797e12 / (mm_v5 * 1e-6) / 1e12;
    assert!(
        (11.6..11.8).contains(&tflops),
        "{tflops} (proba qkv-513 os-476: 11,6)"
    );
    // The dashboard: from the first value (temperature) to the last ("sectiuni ocupate"), 14 readings in the journal.
    let mut dash = [
        235.9, 233.9, 249.4, 266.3, 236.8, 235.9, 544.2, 242.3, 264.3, 237.3, 238.9, 238.1, 230.9,
        230.7,
    ];
    dash.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let dash_med = (dash[6] + dash[7]) / 2.0;
    assert!((236.0..240.0).contains(&dash_med), "{dash_med}");
    // + ~28 ms from the last value to the next doorbell (MEASURED, every step) = the CPU's time between two steps.
    let cpu = dash_med + 28.0;
    assert!(
        cpu > board / 1e3,
        "the dashboard is longer than the board's step: it is the critical path"
    );
    // With two slots the step = max(board, CPU between steps) + a little; step 1 (no dashboard before it): +7.0 ms.
    let model = (board / 1e3).max(cpu) + 7.0;
    assert!(
        (model - full / 1e3).abs() < 6.0,
        "model {model} measured {}",
        full / 1e3
    );
    // Without a stick write per value (or with the dashboard every N steps): the step = the board + ~7 ms (DEDUS).
    let fixed = board / 1e3 + 7.0;
    let sps = |ms: f64| 6.0 * 512.0 / (ms / 1e3);
    assert!((14_500.0..14_800.0).contains(&sps(fixed)), "{}", sps(fixed));
    println!(
        "kt c281 lot 6 CALCUL: placa {:.2} ms (banda 202-222), mm_tc_v5 {tflops:.2} TFLOP/s; tabloul pe stick {dash_med:.1} + 28 ms intre pasi > placa -> pasul {:.1} ms (model {model:.1}); fara el ~{fixed:.0} ms = {:.0} simb/s (azi {:.0})",
        board / 1e3,
        full / 1e3,
        sps(fixed),
        sps(full / 1e3)
    );
}
