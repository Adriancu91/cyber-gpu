//! C-165 (`docs/C-165_VALURI.md`): the "waves" model of a `mm_train` product on the GA104 — pure arithmetic on board
//! times the OS measured (os-423); nothing is sent, nothing is simulated.
//!
//! A launch of `g` blocks with `b` blocks per SM runs in ⌈g / 48b⌉ waves on the 48 SMs; a full wave lasts about as
//! long as one block of that variant at that occupancy (measured, [`OS423_WAVES`]); the last, partial wave is charged
//! at the occupancy it really has (⌈rest / 48⌉ blocks per SM). Time grows linearly with K. Everything this module
//! returns is **DEDUS** — a prediction for the race, never a measurement.

use gpu_core::error::GpuResult;

use crate::mm_train_variants::{Form, MmChoice, VARIANTS};
use crate::occupancy::{self, Reserve, SM86};

/// SMs of the RTX 3070 Ti (GA104, 48 enabled; public NVIDIA spec sheet).
pub const SMS: u32 = 48;

/// The K at which [`OS423_WAVES`] was measured (the qkv shape 1024 × 2304 × 768).
pub const CAL_K: u64 = 768;

/// One step's matmul time on the board / the same launches timed alone in the race (os-418, check V3: 0.936 for
/// `mm_tv_128x128`), in thousandths.
pub const STEP_OVER_RACE_PERMILLE: u64 = 936;

/// Time of one full wave at K = [`CAL_K`], per blocks-per-SM, for one variant of [`VARIANTS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaveTimes {
    /// Index in [`VARIANTS`].
    pub variant: usize,
    /// (blocks per SM, ns per wave), increasing blocks; `0` = unused slot.
    pub by_blocks: [(u32, u64); 4],
}

/// From the occupancy probe os-423 (qkv shape, race clock): launch time / waves. `mm_train_tile`: 576 blocks,
/// 2 840.0 / 1 569.3 / 1 096.6 / 1 040.6 µs at 1 / 2 / 4 / 6 blocks per SM = 12 / 6 / 3 / 2 waves. `mm_tv_64x64k8`:
/// 1 769.1 / 1 114.4 / 1 002.4 µs at 2 / 4 / 6 = 6 / 3 / 2 waves. `mm_tv_128x128`: 144 blocks, 1 890.2 / 1 275.5 µs at
/// 1 / 2 = 3 / 2 waves.
pub const OS423_WAVES: [WaveTimes; 3] = [
    WaveTimes {
        variant: 0,
        by_blocks: [(1, 236_667), (2, 261_550), (4, 365_533), (6, 520_300)],
    },
    WaveTimes {
        variant: 4,
        by_blocks: [(2, 294_850), (4, 371_467), (6, 501_200), (0, 0)],
    },
    WaveTimes {
        variant: 3,
        by_blocks: [(1, 630_067), (2, 637_750), (0, 0), (0, 0)],
    },
];

fn table(variant: usize) -> Option<&'static WaveTimes> {
    OS423_WAVES.iter().find(|w| w.variant == variant)
}

/// ns of one wave of `variant` with `b` blocks per SM, at K = [`CAL_K`] and the race clock. Between two measured
/// occupancies: linear. Below the first one (only `mm_tv_64x64k8` at b = 1): the first one × the witness's
/// 1-block / 2-block ratio (DEDUS). Above the last one: `None`. `None` for a variant without times.
pub fn wave_ns(variant: usize, b: u32) -> Option<u64> {
    let t = table(variant)?;
    let pts: alloc::vec::Vec<(u32, u64)> =
        t.by_blocks.iter().copied().filter(|p| p.0 != 0).collect();
    if b == 0 {
        return None;
    }
    if let Some(p) = pts.iter().find(|p| p.0 == b) {
        return Some(p.1);
    }
    let first = pts[0];
    if b < first.0 {
        let w = &OS423_WAVES[0].by_blocks;
        // witness: b=1 / b=2
        return Some(first.1 * w[0].1 / w[1].1);
    }
    let hi = pts.iter().find(|p| p.0 > b)?;
    let lo = pts.iter().rev().find(|p| p.0 < b)?;
    let span = (hi.0 - lo.0) as u64;
    let up = (b - lo.0) as u64;
    Some(lo.1 + (hi.1 - lo.1) * up / span)
}

/// Blocks of one product of shape (M, N) on `variant` (one block per bm × bn piece of C).
pub fn blocks(variant: usize, m: u32, n: u32) -> u32 {
    let v = &VARIANTS[variant];
    m.div_ceil(v.bm) * n.div_ceil(v.bn)
}

/// Blocks per SM of `variant` at the shared-memory step `step_bytes` (R0: os-423 P5 showed no 1 KiB reserve).
pub fn blocks_per_sm(variant: usize, step_bytes: u32) -> Option<u32> {
    let f = occupancy::blocks_per_sm(
        &SM86,
        &occupancy::variant_kernel(variant),
        step_bytes,
        Reserve::None,
    )?;
    (f.blocks > 0).then_some(f.blocks)
}

/// Predicted ns of one product (M, N, K) on `variant` with `b` blocks per SM, at the race clock (DEDUS).
pub fn launch_ns(variant: usize, m: u32, n: u32, k: u32, b: u32) -> Option<u64> {
    let g = blocks(variant, m, n);
    let slots = SMS * b;
    let (full, rest) = (g / slots, g % slots);
    let mut ns = full as u64 * wave_ns(variant, b)?;
    if rest != 0 {
        ns += wave_ns(variant, rest.div_ceil(SMS))?;
    }
    Some(ns * k as u64 / CAL_K)
}

/// Predicted ns of one product on `variant` at the step `step_bytes` (its blocks per SM there).
pub fn launch_ns_at(variant: usize, m: u32, n: u32, k: u32, step_bytes: u32) -> Option<u64> {
    launch_ns(variant, m, n, k, blocks_per_sm(variant, step_bytes)?)
}

/// The variants the model can judge (those with [`OS423_WAVES`]), in [`VARIANTS`] order.
pub fn modelled() -> impl Iterator<Item = usize> {
    (0..VARIANTS.len()).filter(|&i| table(i).is_some())
}

/// The model's pick at `step_bytes`: for every form, the modelled variant with the smallest predicted time (ties: the
/// lower index). Returns the choice and Σ predicted ns × launches at the **race** clock (× [`STEP_OVER_RACE_PERMILLE`]
/// for the step). A prediction for the race to confirm — never a substitute for it.
pub fn predicted_choice(forms: &[Form], step_bytes: u32) -> GpuResult<(MmChoice, u64)> {
    let mut c = MmChoice::new();
    let mut total = 0u64;
    for f in forms {
        let best = modelled()
            .filter_map(|v| launch_ns_at(v, f.m, f.n, f.k, step_bytes).map(|t| (t, v)))
            .min();
        if let Some((t, v)) = best {
            c.set(f.m, f.n, f.k, f.flags, v)?;
            total += t * f.count as u64;
        }
    }
    Ok((c, total))
}

/// C-157 session 241 (`docs/C-157_VALURI_V4.md`): rows and columns of C per `mm_tc_v4` block (64 × 64, as `mm_tc`).
pub const TC_V4_TILE: u32 = 64;

/// The `mm_tc_v4` calibration from os-443: qkv-513 (**6 144 × 2 304 × 768** = 3 456 blocks), per blocks per SM:
/// (blocks per SM, launch ns, waves) = 4 737 216 ns / 72 waves at 1 block per SM (step 32 KiB), 2 623 680 ns / 36
/// waves at 2 (step 64 KiB; three 24 KiB blocks never fit). Board clock of the probe (MĂSURAT); everything derived
/// from it is DEDUS.
pub const OS443_TC_V4: [(u32, u64, u64); 2] = [(1, 4_737_216, 72), (2, 2_623_680, 36)];

/// The K of the `mm_tc_v4` calibration (qkv-513).
pub const TC_V4_CAL_K: u64 = 768;

fn tc_v4_cal(b: u32) -> Option<(u64, u64)> {
    OS443_TC_V4.iter().find(|p| p.0 == b).map(|p| (p.1, p.2))
}

/// Blocks of one `mm_tc_v4` product C (M × N).
pub fn tc_v4_blocks(m: u32, n: u32) -> u32 {
    m.div_ceil(TC_V4_TILE) * n.div_ceil(TC_V4_TILE)
}

/// ns of one `mm_tc_v4` wave with `b` blocks per SM (1 or 2), at K = [`TC_V4_CAL_K`], rounded down; `None` otherwise.
pub fn tc_v4_wave_ns(b: u32) -> Option<u64> {
    tc_v4_cal(b).map(|(t, w)| t / w)
}

/// Predicted ns of one `mm_tc_v4` product (M, N, K) at `b` blocks per SM (DEDUS): full waves at `b`, the last partial
/// wave at the occupancy it really has, linear in K — the same rule as [`launch_ns`] (exact fractions, one rounding).
pub fn tc_v4_launch_ns(m: u32, n: u32, k: u32, b: u32) -> Option<u64> {
    let g = tc_v4_blocks(m, n);
    let slots = SMS * b;
    let (full, rest) = (g / slots, g % slots);
    let (tb, wb) = tc_v4_cal(b)?;
    // ns × wb × wr, so both fractions stay exact
    let (num, den) = if rest != 0 {
        let (tr, wr) = tc_v4_cal(rest.div_ceil(SMS))?;
        (
            (full as u128 * tb as u128 * wr as u128) + (tr as u128 * wb as u128),
            wb as u128 * wr as u128,
        )
    } else {
        (full as u128 * tb as u128, wb as u128)
    };
    Some((num * k as u128 / (den * TC_V4_CAL_K as u128)) as u64)
}

/// Busy "slot-ns" of one `mm_tc_v4` product at `b` blocks per SM: every block costs one wave's time at its K.
fn tc_v4_busy(m: u32, n: u32, k: u32, b: u32) -> Option<u128> {
    let (t, w) = tc_v4_cal(b)?;
    Some(t as u128 * tc_v4_blocks(m, n) as u128 * k as u128 / (w as u128 * TC_V4_CAL_K as u128))
}

/// The idle share, in thousandths, of one `mm_tc_v4` launch's slots (1 − busy / (launch time × slots)).
pub fn tc_v4_idle_permille(m: u32, n: u32, k: u32, b: u32) -> Option<u64> {
    let offered = tc_v4_launch_ns(m, n, k, b)? as u128 * (SMS * b) as u128;
    let busy = tc_v4_busy(m, n, k, b)?;
    Some((1_000 - (busy * 1_000 / offered.max(1)).min(1_000)) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measured_points_and_interpolation() {
        assert_eq!(wave_ns(0, 6), Some(520_300));
        assert_eq!(wave_ns(0, 3), Some((261_550 + 365_533) / 2));
        assert_eq!(wave_ns(3, 3), None);
        assert_eq!(wave_ns(1, 1), None);
        // 64x64k8 at 1 block: 294 850 × 236 667 / 261 550
        assert_eq!(wave_ns(4, 1), Some(294_850 * 236_667 / 261_550));
    }

    #[test]
    fn qkv_reproduces_the_probe() {
        // 1024 × 2304 × 768 on 128x128 at 2 blocks: 144 blocks = 1 wave of 96 + 48 left (1 block per SM).
        assert_eq!(launch_ns(3, 1024, 2304, 768, 2), Some(637_750 + 630_067));
        assert_eq!(launch_ns(0, 1024, 2304, 768, 6), Some(2 * 520_300));
    }

    #[test]
    fn tc_v4_reproduces_os443() {
        assert_eq!(tc_v4_blocks(6144, 2304), 3456);
        assert_eq!(tc_v4_launch_ns(6144, 2304, 768, 2), Some(2_623_680));
        assert_eq!(tc_v4_launch_ns(6144, 2304, 768, 1), Some(4_737_216));
        assert_eq!(tc_v4_wave_ns(2), Some(72_880));
        // 768 × 768: 144 blocks = 1 wave of 96 + 48 at 1 per SM (4 737 216 / 72 = 65 794.67)
        assert_eq!(tc_v4_launch_ns(768, 768, 768, 2), Some(72_880 + 65_794));
        assert_eq!(tc_v4_idle_permille(1024, 768, 768, 2), Some(0));
        assert_eq!(tc_v4_wave_ns(3), None);
    }
}
