//! C-157 — T268, CALCUL on the host (no program simulated, nothing run on the board): the plan and the predictions for
//! `mm_tc_v5` (`mm_tc_v4` with the 16 KiB output block replaced by a 1 KiB-per-warp staging area, so 4 blocks fit on an
//! SM at the 64 KiB step instead of 2), written in `docs/C-157_MM_TC_V5_PLAN.md` BEFORE any code or compilation.
//!
//! Measured inputs (MĂSURAT, copied from the board reports the OS sent to the hub): os-468 (`mm_tc_v4` in the #513
//! step, lot 6: 216 644 µs, 147 launches), os-443 (`mm_tc_v4` probe qkv-513 at 1 and 2 blocks per SM), os-423 (the
//! occupancy probe of `mm_train_tile` / `mm_tv_64x64k8`). Everything computed from them here is DEDUS.

use nvidia_ga104::occupancy::{Bound, Kernel, Reserve, SM86, blocks_per_sm};
use nvidia_ga104::wave_model::{OS423_WAVES, OS443_TC_V4, SMS};

const E: u64 = 768;
const V: u64 = 16_384;
const LAYERS: u64 = 12;
/// Lot 6 × 512 tokens.
const TOKENS: u64 = 6 * 512;
/// `mm_tc_v4` in one #513 step, os-468 (median of 10 steps).
const OS468_MM_US: u64 = 216_644;
const OS468_STEP_US: u64 = 273_090;

/// The 147 products of one #513 step as (M, N, K, launches): per layer qkv / proj / fc / out forward, their dX and dW,
/// then the head (logits forward, dX, dW).
fn step_products() -> Vec<(u64, u64, u64, u64)> {
    let t = TOKENS;
    let mut v = Vec::new();
    // (rows of W = input width, cols of W = output width)
    for (i, o) in [(E, 3 * E), (E, E), (E, 4 * E), (4 * E, E)] {
        v.push((t, o, i, LAYERS)); // forward: X (t × i) · W (i × o)
        v.push((t, i, o, LAYERS)); // dX = dY · Wᵀ
        v.push((i, o, t, LAYERS)); // dW = Xᵀ · dY
    }
    v.push((t, V, E, 1));
    v.push((t, E, V, 1));
    v.push((E, V, t, 1));
    v
}

fn flops(p: &[(u64, u64, u64, u64)]) -> u64 {
    p.iter().map(|&(m, n, k, c)| 2 * m * n * k * c).sum()
}

#[test]
fn t268_the_step_runs_at_the_probe_speed() {
    let p = step_products();
    assert_eq!(p.iter().map(|x| x.3).sum::<u64>(), 147);
    let f = flops(&p);
    assert_eq!(f, 1_797_443_813_376);
    // step: 8.30 TFLOP/s; probe qkv-513 (6144 × 2304 × 768 at 2 blocks per SM): 8.29 — the same speed.
    let step_gflops = f / OS468_MM_US / 1_000; // GFLOP/s
    assert_eq!(step_gflops, 8_296);
    let probe_flop = 2 * 6_144 * 2_304 * 768u64;
    let probe_ns = OS443_TC_V4[1].1;
    assert_eq!(probe_flop / probe_ns, 8_287); // FLOP per ns = GFLOP/s: 8.29 TFLOP/s
    // share of the step: 79.3 %
    assert_eq!(OS468_MM_US * 1_000 / OS468_STEP_US, 793);
}

#[test]
fn t268_far_from_the_tensor_peak() {
    // GA104 tensor cores, f16 inputs with f32 sums (half rate on GeForce Ampere): 48 SMs × 4 × 64 FMA × 2 per clock,
    // at the 1 950 MHz the board held in os-414 → 47.9 TFLOP/s. `mm_tc_v4` uses 17 % of it.
    let peak_gflops = SMS as u64 * 4 * 64 * 2 * 1_950 / 1_000;
    assert_eq!(peak_gflops, 47_923);
    assert_eq!(8_296 * 1_000 / peak_gflops, 173);
}

#[test]
fn t268_occupancy_v4_two_blocks_v5_four() {
    let v4 = Kernel {
        name: "mm_tc_v4",
        threads: 128,
        registers: 128,
        shared_bytes: 24_576, // as_ 4 KiB + bs_ 4 KiB + cs_ 16 KiB
    };
    let v5 = Kernel {
        name: "mm_tc_v5",
        threads: 128,
        registers: 128,
        shared_bytes: 12_288, // as_ 4 KiB + bs_ 4 KiB + 4 warps × 16 × 16 f32 staging
    };
    for r in [Reserve::None, Reserve::OneKib] {
        let a = blocks_per_sm(&SM86, &v4, 64 << 10, r).unwrap();
        assert_eq!((a.blocks, a.bound), (2, Bound::Shared));
        // 4 under both readings: registers stop it at 4 (R0: 5 would fit in shared memory; R1: a tie at 4)
        assert_eq!(blocks_per_sm(&SM86, &v5, 64 << 10, r).unwrap().blocks, 4);
        // the 32 KiB step (today's other step) would give 2 with v5
        assert_eq!(blocks_per_sm(&SM86, &v5, 32 << 10, r).unwrap().blocks, 2);
    }
    // if NAK gives more than 128 registers, v5 drops to 3 blocks (prediction branch V1b)
    let fat = Kernel {
        registers: 136,
        ..v5
    };
    assert_eq!(
        blocks_per_sm(&SM86, &fat, 64 << 10, Reserve::None)
            .unwrap()
            .blocks,
        3
    );
}

/// Per-block speed-up from `b1` to `b2` blocks per SM, in thousandths, from (blocks, ns per wave) pairs.
fn per_block_gain(b1: u64, w1: u64, b2: u64, w2: u64) -> u64 {
    (w1 * b2 * 1_000) / (w2 * b1)
}

#[test]
fn t268_what_the_board_did_when_blocks_per_sm_doubled() {
    // os-443, mm_tc_v4: 1 → 2 blocks per SM: ×1.806
    let w1 = OS443_TC_V4[0].1 / OS443_TC_V4[0].2;
    let w2 = OS443_TC_V4[1].1 / OS443_TC_V4[1].2;
    assert_eq!(per_block_gain(1, w1, 2, w2), 1_805);
    // os-423, mm_train_tile: 1 → 2 ×1.810 (the same as mm_tc_v4), 2 → 4 ×1.431
    let tile = OS423_WAVES
        .iter()
        .find(|w| w.variant == 0)
        .unwrap()
        .by_blocks;
    assert_eq!(per_block_gain(1, tile[0].1, 2, tile[1].1), 1_809);
    assert_eq!(per_block_gain(2, tile[1].1, 4, tile[2].1), 1_431);
    // os-423, mm_tv_64x64k8: 2 → 4 ×1.587
    let tv = OS423_WAVES
        .iter()
        .find(|w| w.variant == 4)
        .unwrap()
        .by_blocks;
    assert_eq!(per_block_gain(2, tv[0].1, 4, tv[1].1), 1_587);
}

/// Full waves + the partial one of a launch of `g` blocks at `b` per SM, as (full, rest-blocks).
fn waves(g: u64, b: u64) -> (u64, u64) {
    let slots = SMS as u64 * b;
    (g / slots, g % slots)
}

#[test]
fn t268_the_step_shapes_have_almost_no_tail() {
    let p = step_products();
    let tiles = |m: u64, n: u64| m.div_ceil(64) * n.div_ceil(64);
    // weight of the products whose last wave is partial at 4 blocks per SM, in thousandths of the step's FLOP
    let total = flops(&p);
    let tail: u64 = p
        .iter()
        .filter(|&&(m, n, _, _)| waves(tiles(m, n), 4).1 != 0)
        .map(|&(m, n, k, c)| 2 * m * n * k * c)
        .sum();
    assert_eq!(tail * 1_000 / total, 96); // only dW of qkv (432 blocks) and of proj (144): 9.6 % of the FLOP
    assert_eq!(waves(432, 4), (2, 48));
    assert_eq!(waves(144, 4), (0, 144));
    assert_eq!(waves(432, 2), (4, 48));
}

#[test]
fn t268_the_predicted_bands() {
    // V7: probe qkv-513 at 4 blocks per SM, per-block gain ×1.35–1.65 (central ×1.45, between the two measured 2 → 4)
    let probe_ns = OS443_TC_V4[1].1;
    let (lo, mid, hi) = (
        probe_ns * 1_000 / 1_650,
        probe_ns * 1_000 / 1_450,
        probe_ns * 1_000 / 1_350,
    );
    assert_eq!((lo / 1_000, mid / 1_000, hi / 1_000), (1_590, 1_809, 1_943)); // µs
    let probe_flop = 2 * 6_144 * 2_304 * 768u64;
    assert_eq!(
        (probe_flop / hi, probe_flop / mid, probe_flop / lo),
        (11_187, 12_016, 13_674) // GFLOP/s
    );
    // V8: the #513 step, lot 6 (the rest of the step, 56.4 ms, unchanged)
    let rest = OS468_STEP_US - OS468_MM_US;
    assert_eq!(rest, 56_446);
    let mm = |g: u64| OS468_MM_US * 1_000 / g;
    assert_eq!(
        (mm(1_650), mm(1_450), mm(1_350)),
        (131_299, 149_409, 160_477)
    );
    let step = |g: u64| rest + mm(g);
    assert_eq!(
        (step(1_650), step(1_450), step(1_350)),
        (187_745, 205_855, 216_923)
    );
    // tokens per second: 3 072 per step
    let tps = |g: u64| TOKENS * 1_000_000 / step(g);
    assert_eq!(
        (tps(1_350), tps(1_450), tps(1_650)),
        (14_161, 14_923, 16_362)
    );
    // the threshold written before: under ×1.2 on the probe → occupancy is not the brake
    assert_eq!(probe_ns * 1_000 / 1_200 / 1_000, 2_186);
}
