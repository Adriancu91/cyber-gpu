//! C-157 — T264, CALCUL on the host (no program simulated, nothing run on the board): the comparison of os-462 and
//! os-465 (`docs/C-157_OS462_OS465.md`) checked against our own constants.
//!
//! 1. Why `attn_back_tile_tc16` gained twice what we predicted inside the fast mixed step (os-462: −6.7 ms, predicted
//!    −2.2…−4.0): the probe `atentie-tc` launches it at the shared-memory step of ONE block (16 KiB → 1 block per SM),
//!    the fast step at the 64 KiB step (3–4 blocks per SM). `attn_back_tile_3` (F32, 33 024 B) is at 1 block per SM
//!    in both places, so its time scales with the batch only.
//! 2. What the occupancy says for v4 / v4pf in the step (os-465 measured them at 1 block per SM).
//! 3. The wide barrier probe (os-465) against W1–W6 (`docs/C-157_OS459_BARIERA.md` §4) and the `token32` arithmetic.
//!
//! The measured numbers are copied from the board reports (`rezultate/os-automat/2026-11-26/ciclu-auto-20261126-1553.txt`,
//! `-1633.txt`, `-1636.txt` in the hub); everything derived from them here is DEDUS.

use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
use nvidia_ga104::qmd::{SharedMemory, sm_config_shared_units};
use nvidia_ga104::{
    attn_back_tile_3 as f32prog, attn_back_tile_tc16 as tc16, attn_back_tile_tc16_v4 as v4,
    attn_back_tile_tc16_v4pf as v4pf, grid_barrier,
};

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
fn t264_probe_runs_tc16_at_one_block_the_step_at_three_or_four() {
    let k_tc16 = kernel("tc16", tc16::BLOCK[0], tc16::REGISTERS, &tc16::SHARED);
    let k_f32 = kernel(
        "_3",
        f32prog::BLOCK[0],
        f32prog::REGISTERS,
        &f32prog::SHARED,
    );

    // The probe (`to_bytes_with_shared`): tc16 = 16 384 B → the 16 KiB step; `_3` = 33 024 B → the 64 KiB step.
    assert_eq!(one_block_step(&tc16::SHARED), 16 * 1024);
    assert_eq!(one_block_step(&f32prog::SHARED), 64 * 1024);
    // At its own step tc16 holds exactly one block (none at all with the 1 KiB reserve: the board ran it, so at least
    // one; the arithmetic cannot say more).
    assert_eq!(blocks(&k_tc16, 16 * 1024, Reserve::None), 1);
    // The fast step (`--mixt-513-rapid`: every launch at 64 KiB): 3 (R1) or 4 (R0) blocks.
    assert_eq!(blocks(&k_tc16, 64 * 1024, Reserve::OneKib), 3);
    assert_eq!(blocks(&k_tc16, 64 * 1024, Reserve::None), 4);
    // `_3` is at one block per SM in both places.
    assert_eq!(blocks(&k_f32, 64 * 1024, Reserve::None), 1);
    assert_eq!(blocks(&k_f32, 64 * 1024, Reserve::OneKib), 1);
}

#[test]
fn t264_measured_ratios_fit_the_occupancy_reading() {
    // MĂSURAT (os-465 probe, `atentie-513-b4` = batch 4, median of 5 per launch).
    let probe_f32_b4_ms = 1.59_f64;
    let probe_tc16_b4_ms = 0.87_f64;
    // MĂSURAT (os-462, fast mixed step #513, batch 2, profile of 10 steps, 12 launches each).
    let step_f32_b2_ms = 9.172_f64 / 12.0; // os-456, `attn_back_tile_3` 9 172 µs
    let step_tc16_b2_ms = 2.511_f64 / 12.0; // os-462, `attn_back_tile_tc16` 2 511 µs

    // F32: same occupancy in probe and step → the time halves with the batch (×2 ± 5 %).
    let f32_ratio = probe_f32_b4_ms / step_f32_b2_ms;
    assert!((1.9..=2.1).contains(&f32_ratio), "F32 b4/b2 = {f32_ratio}");

    // tc16: ×4.2 for half the work → ×2.1 per equal work, between 1 (no gain from more blocks) and 3 (perfect hiding
    // with the 3 blocks of the 1 KiB reading). This is the factor our prediction (P2zp) left out.
    let tc16_ratio = probe_tc16_b4_ms / step_tc16_b2_ms;
    let per_work = tc16_ratio / 2.0;
    assert!(
        (4.0..=4.3).contains(&tc16_ratio),
        "tc16 b4/b2 = {tc16_ratio}"
    );
    assert!(
        per_work > 1.5 && per_work <= 3.0,
        "tc16 gain from occupancy = {per_work}"
    );

    // The gain in the step: 9.17 − 2.51 = 6.66 ms; the predicted band was 2.2–4.0 ms (missed, too low).
    let gain = 9.172 - 2.511;
    assert!(gain > 4.0 && (6.6..6.7).contains(&gain));
}

#[test]
fn t264_v4_in_the_step_v4pf_loses_blocks() {
    let k_tc16 = kernel("tc16", tc16::BLOCK[0], tc16::REGISTERS, &tc16::SHARED);
    let k_v4 = kernel("v4", v4::BLOCK[0], v4::REGISTERS, &v4::SHARED);
    let k_v4pf = kernel("v4pf", v4pf::BLOCK[0], v4pf::REGISTERS, &v4pf::SHARED);
    // os-465 timed all three at the 16 KiB step = 1 block per SM (same shared memory as tc16).
    assert_eq!(one_block_step(&v4::SHARED), 16 * 1024);
    assert_eq!(one_block_step(&v4pf::SHARED), 16 * 1024);
    // In the fast step (64 KiB): v4 keeps 3 (registers: 72 → 3 blocks of 256 threads), v4pf drops to 2 (104 registers),
    // tc16 has 3–4. So the step can only shrink v4's ×1.67 lead, and v4pf may lose to tc16 there.
    for r in [Reserve::None, Reserve::OneKib] {
        assert_eq!(blocks(&k_v4, 64 * 1024, r), 3);
        assert_eq!(blocks(&k_v4pf, 64 * 1024, r), 2);
        assert!(blocks(&k_tc16, 64 * 1024, r) >= blocks(&k_v4, 64 * 1024, r));
    }
}

#[test]
fn t264_wide_barrier_against_w1_w6() {
    // The G the OS used (os-465).
    assert_eq!(grid_barrier::groups_for_wide_sms(48, 8).unwrap(), 384);
    assert_eq!(grid_barrier::groups_for_wide_sms(48, 16).unwrap(), 768);
    // MĂSURAT: one barrier (R = 1 001 minus R = 1, per round), ns.
    let (b48, b96, b384, b768) = (1128.0_f64, 1146.0, 2025.0, 3114.0);
    // W3 1.2–2.5 µs at 384; W4 1.5–4.0 µs at 768 and above 384.
    assert!((1200.0..=2500.0).contains(&b384));
    assert!((1500.0..=4000.0).contains(&b768) && b768 > b384);
    assert!(b48 < b96 && b96 < b384);
    // W5: R = 1 (launch + one barrier) under 15 µs: 4.448 / 5.824 µs measured.
    let r1_ns = [4448.0_f64, 5824.0];
    assert!(r1_ns.iter().all(|&t| t < 15_000.0));
    // W6: 87 barriers per token against 88 launch starts of 3.87 µs (os-346).
    let starts = 88.0 * 3870.0;
    let at384 = 87.0 * b384;
    let at768 = 87.0 * b768;
    assert!(at384 < starts && at768 < starts);
    // The ceiling of the gain, per token: ~165 µs at G = 384, ~70 µs at G = 768 → G = 384 is the one to try.
    assert!((starts - at384 - 164_500.0).abs() < 1_000.0);
    assert!((starts - at768 - 69_640.0).abs() < 1_000.0);
}
