//! C-157 — T267, CALCUL on the host (no program simulated, nothing run on the board): the comparison of os-469
//! (`docs/C-157_OS469.md`) — the attention probe `atentie-tc` run a second time at the 64 KiB shared-memory step, the
//! step the fast mixed #513 step uses (P2zw point 3).
//!
//! 1. What we wrote before the board (P2zw / `docs/C-157_OS462_OS465.md`): tc16 in the step (lot 2) 0.209 ms per launch,
//!    v4 ~0.12–0.19 ms, v4pf possibly slower than tc16. Scaled to the probe's batch (`atentie-513-b4`) with the F32
//!    batch factor (1.59 / 0.764), these are predictions for the probe at 64 KiB.
//! 2. The probe at 64 KiB against the step: the v4 / tc16 ratio of the probe against os-468 (lot 6, in the step).
//! 3. The occupancy reading: `attn_back_tile_tc` (32 KiB) gains ×1.67 at 64 KiB — possible only with 2 blocks per SM,
//!    i.e. without the 1 KiB reserve per block (R0, already seen in os-423).
//! 4. The dS programs (`attn_back_tile0_tc`, `_3`) hold 1 block per SM at 64 KiB under either reading: the 64 KiB row
//!    would not change their time, so os-467 resumed is comparable to the step as it is.
//!
//! The measured numbers are copied from the board reports (`rezultate/os-automat/2026-11-26/ciclu-auto-20261126-2029.txt`
//! = os-469, `-2003.txt` = os-468, `-1553.txt` = os-462 in the hub); everything derived from them here is DEDUS.

use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
use nvidia_ga104::qmd::{SharedMemory, sm_config_shared_units};
use nvidia_ga104::{
    attn_back_tile_3 as f32prog, attn_back_tile_tc as tc, attn_back_tile_tc16 as tc16,
    attn_back_tile_tc16_v4 as v4, attn_back_tile_tc16_v4pf as v4pf, attn_back_tile0_3 as ds_f32,
    attn_back_tile0_tc as ds_tc,
};

const KIB64: u32 = 64 * 1024;

/// os-469 (`atentie-513-b4`, median of 5 launches, ms): at the probe's own step and at 64 KiB.
const TC_OWN: f64 = 1.16;
const TC_64K: f64 = 0.69;
const TC16_OWN: f64 = 0.87;
const TC16_64K: f64 = 0.41;
const V4_OWN: f64 = 0.52;
const V4_64K: f64 = 0.34;
const V4PF_64K: f64 = 0.45;

/// os-462 (lot 2, in the fast step): `attn_back_tile_tc16` 2.51 ms over 12 launches.
const TC16_STEP_LOT2: f64 = 2.51 / 12.0;
/// The F32 batch factor b4 / lot 2 (os-465 probe 1.59 ms, os-456 step 0.764 ms per launch): batch only, 1 block both.
const BATCH_B4_OVER_LOT2: f64 = 1.59 / 0.764;
/// os-468 (lot 6, in the step): attention backward per step, µs — tc16 (os-465 lot 6) and v4.
const STEP_LOT6_TC16_US: f64 = 6475.0;
const STEP_LOT6_V4_US: f64 = 5393.0;

/// The step (bytes) `ComputeQmd::encode_with_shared` asks for: the smallest step holding one block.
fn one_block_step(s: &SharedMemory) -> u32 {
    let units = sm_config_shared_units(s.bytes.next_multiple_of(0x100));
    (units - 1) * 4 * 1024
}

fn kernel(name: &'static str, threads: u32, registers: u32, s: &SharedMemory) -> Kernel {
    Kernel {
        name,
        threads,
        registers,
        shared_bytes: s.bytes,
    }
}

fn blocks(k: &Kernel, step: u32, r: Reserve) -> u32 {
    blocks_per_sm(&SM86, k, step, r)
        .expect("runs on SM86")
        .blocks
}

#[test]
fn t267_predictions_from_the_step_hold_for_the_probe_at_64k() {
    // tc16: the step time scaled by the batch.
    let tc16_pred = TC16_STEP_LOT2 * BATCH_B4_OVER_LOT2;
    assert!((tc16_pred - 0.435).abs() < 0.005, "{tc16_pred}");
    let err = TC16_64K / tc16_pred - 1.0;
    assert!(err.abs() < 0.10, "tc16 at 64 KiB off by {err}");
    // v4: P2zw band 0.12–0.19 ms in the step (lot 2) → the probe band.
    let (lo, hi) = (0.12 * BATCH_B4_OVER_LOT2, 0.19 * BATCH_B4_OVER_LOT2);
    assert!(lo < V4_64K && V4_64K < hi, "v4 {V4_64K} outside {lo}..{hi}");
    // v4pf "may come out slower than tc16 in the step": it did.
    const { assert!(V4PF_64K > TC16_64K) };
    // The gain of v4 over tc16 shrinks at 64 KiB (more blocks hide the waiting): ×1.67 → ×1.21.
    let own = TC16_OWN / V4_OWN;
    let at64 = TC16_64K / V4_64K;
    assert!(own > 1.6 && at64 < 1.25 && at64 > 1.15, "{own} {at64}");
}

#[test]
fn t267_probe_at_64k_matches_the_step_ratio() {
    let probe = TC16_64K / V4_64K;
    let step = STEP_LOT6_TC16_US / STEP_LOT6_V4_US;
    assert!(
        (probe / step - 1.0).abs() < 0.02,
        "probe {probe} step {step}"
    );
    // Per launch in the step (lot 6) against the probe (b4): both programs scale the same way with the batch.
    let tc16_scale = STEP_LOT6_TC16_US / 12.0 / 1000.0 / TC16_64K;
    let v4_scale = STEP_LOT6_V4_US / 12.0 / 1000.0 / V4_64K;
    assert!((tc16_scale / v4_scale - 1.0).abs() < 0.02);
    assert!(tc16_scale > 1.25 && tc16_scale < 1.40, "{tc16_scale}");
}

#[test]
fn t267_tc_gain_at_64k_needs_two_blocks_so_no_reserve() {
    let k_tc = kernel("tc", tc::BLOCK[0], tc::REGISTERS, &tc::SHARED);
    assert_eq!(tc::SHARED.bytes, 32 * 1024);
    // At its own step: one block.
    let own = one_block_step(&tc::SHARED);
    assert_eq!(own, 32 * 1024);
    assert_eq!(blocks(&k_tc, own, Reserve::None), 1);
    // At 64 KiB: 2 blocks without the reserve, still 1 with it.
    assert_eq!(blocks(&k_tc, KIB64, Reserve::None), 2);
    assert_eq!(blocks(&k_tc, KIB64, Reserve::OneKib), 1);
    // Measured ×1.67: far from the ~×1.0 a program with the same block count shows (the F32 `_3` at 1 block in both
    // places, and the small sets below) → only R0 explains it.
    let gain = TC_OWN / TC_64K;
    assert!(gain > 1.5, "{gain}");
    // tc16 / v4 / v4pf at 64 KiB under R0: 4 / 3 / 2 blocks (registers bind v4 and v4pf).
    let k16 = kernel("tc16", tc16::BLOCK[0], tc16::REGISTERS, &tc16::SHARED);
    let k4 = kernel("v4", v4::BLOCK[0], v4::REGISTERS, &v4::SHARED);
    let k4pf = kernel("v4pf", v4pf::BLOCK[0], v4pf::REGISTERS, &v4pf::SHARED);
    assert_eq!(blocks(&k16, KIB64, Reserve::None), 4);
    assert_eq!(blocks(&k4, KIB64, Reserve::None), 3);
    assert_eq!(blocks(&k4pf, KIB64, Reserve::None), 2);
}

#[test]
fn t267_small_sets_do_not_move_and_ds_holds_one_block() {
    // os-469, the four small sets (ns): (today, at 64 KiB) for tc and tc16 — few blocks per SM in all, so the step
    // changes nothing beyond the noise of the board (≤ 5 %).
    let pairs: [(f64, f64); 10] = [
        (29536.0, 29632.0),
        (25024.0, 25248.0),
        (122752.0, 121024.0),
        (93568.0, 94464.0),
        (60288.0, 61152.0),
        (42848.0, 42496.0),
        (219456.0, 221120.0),
        (198976.0, 207104.0),
        (79936.0, 80256.0),
        (58400.0, 58496.0),
    ];
    for (a, b) in pairs {
        assert!((b / a - 1.0).abs() < 0.05, "{a} {b}");
    }
    // The dS programs: 1 block per SM at 64 KiB under both readings → the 64 KiB row would not change them.
    for (name, regs, block, s) in [
        ("ds_tc", ds_tc::REGISTERS, ds_tc::BLOCK[0], &ds_tc::SHARED),
        (
            "ds_f32",
            ds_f32::REGISTERS,
            ds_f32::BLOCK[0],
            &ds_f32::SHARED,
        ),
        (
            "f32",
            f32prog::REGISTERS,
            f32prog::BLOCK[0],
            &f32prog::SHARED,
        ),
    ] {
        let k = kernel(name, block, regs, s);
        assert_eq!(blocks(&k, KIB64, Reserve::None), 1, "{name}");
        assert_eq!(blocks(&k, KIB64, Reserve::OneKib), 1, "{name}");
    }
}
