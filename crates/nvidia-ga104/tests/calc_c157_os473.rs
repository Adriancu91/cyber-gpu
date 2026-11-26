//! C-157 — T273, CALCUL on the host (no program simulated, nothing run on the board): the comparison of os-473
//! (`docs/C-157_OS473.md`) — the fast mixed step #513 with dS on `attn_back_tile0_tc_v4` (`--atentie-ds`, P2zu).
//!
//! The prediction (`docs/C-157_OS470.md` §3) was written for lot 6; the board ran lot 2. The rule behind it — the
//! v4 / F32 ratio of the probe (×2.16) carries over to the step, because all dS programs hold 1 block per SM — is
//! applied here to lot 2 AFTER the run (marked so in the note). Measured numbers are copied from the board reports:
//! `rezultate/os-automat/2026-11-27/ciclu-auto-20261127-0033.txt` (os-473) and
//! `rezultate/os-automat/2026-11-26/ciclu-auto-20261126-1958.txt` (the same step at lot 2 without `--atentie-ds`).
//! Everything computed from them is DEDUS.

/// os-470 probe (`atentie-513-b4`, median ms per launch).
const PROBE_F32_MS: f64 = 0.82;
const PROBE_V4_MS: f64 = 0.38;

/// Lot 2, 26 Nov 19:58 (without `--atentie-ds`), per step, µs.
const BASE_DS_F32_US: f64 = 5192.0;
const BASE_BOARD_US: f64 = 99_822.0;
const BASE_STEP_US: f64 = 111_451.0;
const BASE_MM_US: f64 = 73_884.0;
const BASE_TC16_US: f64 = 2450.0;

/// Lot 2, os-473 (27 Nov 00:33, with `--atentie-ds`), per step, µs.
const NEW_DS_V4_US: f64 = 2427.0;
const NEW_BOARD_US: f64 = 97_076.0;
const NEW_STEP_US: f64 = 108_322.0;
const NEW_MM_US: f64 = 73_921.0;
const NEW_TC16_US: f64 = 2442.0;

const TOKENS_LOT2: f64 = 2.0 * 512.0;

#[test]
fn ds_in_the_step_follows_the_probe_ratio() {
    let probe_ratio = PROBE_F32_MS / PROBE_V4_MS;
    assert!((probe_ratio - 2.158).abs() < 0.001);
    // The rule applied to lot 2 (after the run): 5.19 ms / 2.16 = 2.41 ms.
    let predicted = BASE_DS_F32_US / probe_ratio;
    assert!((predicted - 2406.0).abs() < 1.0);
    // Measured 2.43 ms: +0.9 % over the rule.
    let off = NEW_DS_V4_US / predicted - 1.0;
    assert!(off > 0.0 && off < 0.01, "{off}");
    // In the step the ratio is ×2.14 (probe ×2.16).
    let step_ratio = BASE_DS_F32_US / NEW_DS_V4_US;
    assert!((step_ratio - 2.139).abs() < 0.001);
}

#[test]
fn the_board_gain_is_all_ds() {
    let ds_gain = BASE_DS_F32_US - NEW_DS_V4_US; // 2 765 µs
    let board_gain = BASE_BOARD_US - NEW_BOARD_US; // 2 746 µs
    assert_eq!(ds_gain, 2765.0);
    assert_eq!(board_gain, 2746.0);
    // The rest of the board time moved by +19 µs (0.02 %): the other programs are unchanged.
    assert!(((NEW_BOARD_US - NEW_DS_V4_US) - (BASE_BOARD_US - BASE_DS_F32_US)).abs() < 25.0);
    assert!((NEW_MM_US - BASE_MM_US).abs() < 50.0);
    assert!((NEW_TC16_US - BASE_TC16_US).abs() < 10.0);
}

#[test]
fn whole_step_and_tokens_per_second() {
    let step_gain = BASE_STEP_US - NEW_STEP_US; // 3 129 µs
    assert_eq!(step_gain, 3129.0);
    let tps_base = TOKENS_LOT2 / (BASE_STEP_US * 1e-6);
    let tps_new = TOKENS_LOT2 / (NEW_STEP_US * 1e-6);
    assert!((tps_base - 9188.0).abs() < 1.0 && (tps_new - 9453.0).abs() < 1.0);
    // +2.9 % tokens per second; dS is now 2.5 % of the board time, mm_tc_v4 76 %.
    assert!((tps_new / tps_base - 1.0 - 0.0289).abs() < 0.001);
    assert!((NEW_DS_V4_US / NEW_BOARD_US - 0.025).abs() < 0.001);
    assert!((NEW_MM_US / NEW_BOARD_US - 0.761).abs() < 0.001);
}

#[test]
fn lot6_prediction_restated_with_the_lot2_check() {
    // The lot 6 prediction (os-470 §3) stays untested; with the +0.9 % seen at lot 2 it would land at
    // 13.4 / 2.16 × 1.009 = 6.26 ms dS (band 5.9–6.4 ms), step ~266 ms (band 264–268 ms).
    let ds6 = 13.4 / (PROBE_F32_MS / PROBE_V4_MS)
        * (NEW_DS_V4_US / (BASE_DS_F32_US / (PROBE_F32_MS / PROBE_V4_MS)));
    assert!(ds6 > 5.9 && ds6 < 6.4, "{ds6}");
    let step6 = 273.09 - (13.4 - ds6);
    assert!(step6 > 264.0 && step6 < 268.0, "{step6}");
}
