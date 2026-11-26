//! C-157 — T270, CALCUL on the host (no program simulated, nothing run on the board): the comparison of os-470
//! (`docs/C-157_OS470.md`) — os-467 resumed with the repaired dS programs `attn_back_tile0_tc` / `_tc_v4` (P2zx).
//!
//! 1. What we wrote before the board: E7 (`docs/C-157_ATTN_BACK0_TC.md`, for tc: 5.2 ms → 1.8–3.2 ms per step, i.e.
//!    ×1.63–2.89 over the F32 `attn_back_tile0_3`), Z5 (`docs/C-157_OS467_ATTN0_TC_BLOCAJ.md` §4b: tc at most ~5 % above
//!    E7), P2zt (v4 ×1.2–1.8 over tc, not bound under ×1.15) and `docs/C-157_OS469.md` §4 (dS gain ×1.2–2 per launch).
//! 2. The probe holds 1 block per SM for all three dS programs, at its own step and at 64 KiB: the ratios carry over to
//!    the fast mixed step (lot 6) unchanged.
//! 3. The prediction for os-472 (P2zu bound, `--atentie-ds`), written here before that run: the dS share of the step.
//!
//! The measured numbers are copied from the board report (`rezultate/os-automat/2026-11-26/ciclu-auto-20261126-2254.txt`
//! = os-470) and from os-468 (`-2003.txt`, the step at lot 6); everything derived from them here is DEDUS.

use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
use nvidia_ga104::qmd::{SharedMemory, sm_config_shared_units};
use nvidia_ga104::{
    attn_back_tile0_3 as ds_f32, attn_back_tile0_tc as ds_tc, attn_back_tile0_tc_v4 as ds_v4,
};

const KIB64: u32 = 64 * 1024;

/// os-470, `atentie-513-b4`, median of 5 launches (ms).
const F32_MS: f64 = 0.82;
const TC_MS: f64 = 0.56;
const V4_MS: f64 = 0.38;
/// os-470, the four aligned small sets in total (ns): v4 and tc.
const SMALL_V4_NS: f64 = 253_888.0;
const SMALL_TC_NS: f64 = 300_448.0;

/// os-468 (lot 6, in the fast step): `attn_back_tile0_3` per step (12 launches) and the whole step on the board.
const STEP_LOT6_DS_F32_MS: f64 = 13.4;
const STEP_LOT6_MS: f64 = 273.09;
const STEP_LOT6_TOKENS: f64 = 6.0 * 512.0;
/// os-469: the batch scale probe (b4) → step (lot 6) per launch, for the attention programs on the special circuits.
const BATCH_SCALE_OS469: f64 = 1.32;

fn one_block_step(s: &SharedMemory) -> u32 {
    let units = sm_config_shared_units(s.bytes.next_multiple_of(0x100));
    (units - 1) * 4 * 1024
}

fn blocks(
    name: &'static str,
    threads: u32,
    registers: u32,
    s: &SharedMemory,
    step: u32,
    r: Reserve,
) -> u32 {
    let k = Kernel {
        name,
        threads,
        registers,
        shared_bytes: s.bytes,
    };
    blocks_per_sm(&SM86, &k, step, r)
        .expect("runs on SM86")
        .blocks
}

#[test]
fn t270_written_predictions_against_os470() {
    // E7 for tc: ×1.63–2.89 over F32. Measured ×1.46 → below the band (NECONFIRMATĂ, by ~10 %).
    let (e7_lo, e7_hi) = (5.2 / 3.2, 5.2 / 1.8);
    let tc_gain = F32_MS / TC_MS;
    assert!(
        tc_gain < e7_lo && tc_gain > 0.85 * e7_lo,
        "tc ×{tc_gain} vs {e7_lo}..{e7_hi}"
    );
    // Z5: tc at most ~5 % above the slow end of E7. Measured: ~11 % above.
    let over = TC_MS / (F32_MS / e7_lo) - 1.0;
    assert!(over > 0.05 && over < 0.15, "{over}");
    // v4 over tc: band ×1.2–1.8 (P2zt). Measured ×1.47 → CONFIRMATĂ; above the ×1.15 bind threshold.
    let v4_tc = TC_MS / V4_MS;
    assert!(v4_tc > 1.2 && v4_tc < 1.8, "{v4_tc}");
    // v4 over F32: ×2.16 — inside the E7 band written for tc, and just above the ×1.2–2 of `C-157_OS469.md` §4.
    let v4_f32 = F32_MS / V4_MS;
    assert!(v4_f32 > e7_lo && v4_f32 < e7_hi && v4_f32 > 2.0, "{v4_f32}");
    // Small aligned sets: the gain of v4 shrinks to ×1.18 where few blocks run (less to hide).
    let small = SMALL_TC_NS / SMALL_V4_NS;
    assert!(small > 1.1 && small < v4_tc, "{small}");
}

#[test]
fn t270_ds_programs_hold_one_block_in_probe_and_step() {
    for (name, regs, block, s) in [
        (
            "ds_f32",
            ds_f32::REGISTERS,
            ds_f32::BLOCK[0],
            &ds_f32::SHARED,
        ),
        ("ds_tc", ds_tc::REGISTERS, ds_tc::BLOCK[0], &ds_tc::SHARED),
        ("ds_v4", ds_v4::REGISTERS, ds_v4::BLOCK[0], &ds_v4::SHARED),
    ] {
        let own = one_block_step(s);
        for step in [own, KIB64] {
            for r in [Reserve::None, Reserve::OneKib] {
                assert_eq!(blocks(name, block, regs, s, step, r), 1, "{name} at {step}");
            }
        }
    }
}

#[test]
fn t270_prediction_for_os472_written_before_the_run() {
    // F32 dS per launch, step (lot 6) over probe (b4): ×1.36 — the same batch scale as os-469 (×1.32) within 4 %.
    let f32_scale = STEP_LOT6_DS_F32_MS / 12.0 / F32_MS;
    assert!(
        (f32_scale / BATCH_SCALE_OS469 - 1.0).abs() < 0.04,
        "{f32_scale}"
    );
    // v4 in the step: 0.38 ms × (1.30..1.40) × 12 = 5.9–6.4 ms instead of 13.4 → 7.0–7.5 ms less.
    let (lo, hi) = (V4_MS * 1.30 * 12.0, V4_MS * 1.40 * 12.0);
    assert!(lo > 5.8 && hi < 6.5, "{lo} {hi}");
    let (save_lo, save_hi) = (STEP_LOT6_DS_F32_MS - hi, STEP_LOT6_DS_F32_MS - lo);
    assert!(save_lo > 6.9 && save_hi < 7.6, "{save_lo} {save_hi}");
    // The step: 273.09 → ~265.6–266.2 ms before the board's drift (±0.5 % = ±1.4 ms seen in os-468) → 264–268 ms.
    let (step_lo, step_hi) = (STEP_LOT6_MS - save_hi - 1.4, STEP_LOT6_MS - save_lo + 1.4);
    assert!(step_lo > 264.0 && step_hi < 268.0, "{step_lo} {step_hi}");
    let tok = |ms: f64| STEP_LOT6_TOKENS / (ms / 1000.0);
    assert!(tok(step_hi) > 11_450.0 && tok(step_lo) < 11_650.0);
    // dS was 4.9 % of the step; with v4 it becomes ~2.3 %. `mm_tc_v4` (216.6 ms) stays ~81 %.
    let share = (lo + hi) / 2.0 / (STEP_LOT6_MS - (save_lo + save_hi) / 2.0);
    assert!(share > 0.02 && share < 0.026, "{share}");
}
