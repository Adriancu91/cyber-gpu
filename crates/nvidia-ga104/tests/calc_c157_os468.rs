//! C-157 — T265, CALCUL on the host (no program simulated, nothing run on the board): the comparison of os-468
//! (`attn_back_tile_tc16_v4` bound in the fast mixed step #513, lot 6) with os-465 lot 6 (`attn_back_tile_tc16`, same
//! line otherwise), written in `docs/C-157_OS468.md`.
//!
//! The measured numbers are copied from the board reports in the hub
//! (`rezultate/os-automat/2026-11-26/ciclu-auto-20261126-1748.txt` = os-465 lot 6,
//! `ciclu-auto-20261126-2003.txt` = os-468); everything derived from them here is DEDUS.

/// Per-program board time of one step (median of 10 steps, microseconds, 12 launches each where noted).
struct Profile {
    step_us: u32,
    attn_back_us: u32,
    mm_tc_v4_us: u32,
    attn_fwd_tile3_us: u32,
    attn_back_tile0_3_us: u32,
    losses_e4: [u32; 10],
    norms_e4: [u32; 10],
}

/// os-465 lot 6: `attn_back_tile_tc16` on the 12 attention launches.
const OS465_LOT6: Profile = Profile {
    step_us: 272_810,
    attn_back_us: 6_475,
    mm_tc_v4_us: 215_379,
    attn_fwd_tile3_us: 15_902,
    attn_back_tile0_3_us: 13_348,
    losses_e4: [
        98716, 98592, 98780, 98980, 99305, 99805, 99746, 99675, 100021, 99387,
    ],
    norms_e4: [18486, 6506, 5725, 6570, 5450, 5405, 5349, 5389, 5369, 5257],
};

/// os-468: `attn_back_tile_tc16_v4` on the 12 (aligned) attention launches, tc16 on none.
const OS468: Profile = Profile {
    step_us: 273_090,
    attn_back_us: 5_393,
    mm_tc_v4_us: 216_644,
    attn_fwd_tile3_us: 15_989,
    attn_back_tile0_3_us: 13_414,
    losses_e4: [
        98716, 98592, 98780, 98980, 99305, 99805, 99746, 99675, 100021, 99387,
    ],
    norms_e4: [18486, 6506, 5725, 6570, 5450, 5405, 5349, 5389, 5369, 5257],
};

/// Steps 2..10 of each run (step 1 carries the warm-up), board microseconds.
const OS465_STEPS: [u32; 9] = [
    272_810, 272_763, 272_752, 272_705, 272_680, 272_825, 272_856, 272_965, 272_489,
];
const OS468_STEPS: [u32; 9] = [
    273_024, 273_112, 273_227, 273_101, 273_040, 273_090, 273_083, 272_969, 272_904,
];

fn per_mille(new: u32, old: u32) -> i64 {
    (i64::from(new) - i64::from(old)) * 1000 / i64::from(old)
}

#[test]
fn t265_same_numbers_as_tc16() {
    // P2zv / W3: v4 does the same arithmetic as tc16 → the printed losses and norms are the same, all 10 steps.
    assert_eq!(OS468.losses_e4, OS465_LOT6.losses_e4);
    assert_eq!(OS468.norms_e4, OS465_LOT6.norms_e4);
}

#[test]
fn t265_attention_gain_inside_the_p2zw_band() {
    let gain = OS465_LOT6.attn_back_us - OS468.attn_back_us;
    assert_eq!(gain, 1_082);
    // Per launch: tc16 539.6 µs → v4 449.4 µs = ×1.20.
    let ratio_e3 = OS465_LOT6.attn_back_us * 1000 / OS468.attn_back_us;
    assert_eq!(ratio_e3, 1_200);
    // P2zw (lot 2): v4 ~0.12–0.19 ms against tc16 0.209 → ×1.10…×1.74 per launch. Carried to lot 6 (tc16 measured
    // 6 475 µs per step): 12 × 539.6 × (1 − 1/r) = 0.59…2.76 ms. The measured 1.08 ms sits inside, in the lower half.
    let low = OS465_LOT6.attn_back_us - OS465_LOT6.attn_back_us * 100 / 110;
    let high = OS465_LOT6.attn_back_us - OS465_LOT6.attn_back_us * 100 / 174;
    assert!((580..=600).contains(&low) && (2_740..=2_770).contains(&high));
    assert!(low < gain && gain < high);
    // Far below the isolated probe (os-465, 1 block per SM: ×1.67): the step already runs tc16 at 3–4 blocks per SM.
    assert!(ratio_e3 < 1_670);
}

#[test]
fn t265_the_step_did_not_get_faster_because_everything_else_drifted() {
    // The whole board step: +0.28 ms, although the attention took 1.08 ms off.
    assert_eq!(OS468.step_us - OS465_LOT6.step_us, 280);
    let rest_old = OS465_LOT6.step_us - OS465_LOT6.attn_back_us;
    let rest_new = OS468.step_us - OS468.attn_back_us;
    assert_eq!(rest_new - rest_old, 1_362);
    assert_eq!(per_mille(rest_new, rest_old), 5); // +0.5 %
    // The same programs, the same work: the three big compute-bound ones are all +0.5…0.6 %.
    for (new, old) in [
        (OS468.mm_tc_v4_us, OS465_LOT6.mm_tc_v4_us),
        (OS468.attn_fwd_tile3_us, OS465_LOT6.attn_fwd_tile3_us),
        (OS468.attn_back_tile0_3_us, OS465_LOT6.attn_back_tile0_3_us),
    ] {
        assert!((4..=5).contains(&per_mille(new, old)));
    }
    // Inside each run steps 2..10 stay within 0.5 ms; the medians of steps 2..10 differ by 0.32 ms, the same size as
    // that spread: the whole-step difference alone is noise, the per-program pattern above is what says "drift".
    let spread = |s: &[u32; 9]| s.iter().max().unwrap() - s.iter().min().unwrap();
    assert!(spread(&OS465_STEPS) <= 500 && spread(&OS468_STEPS) <= 500);
    let median = |s: &[u32; 9]| {
        let mut v = *s;
        v.sort_unstable();
        v[4]
    };
    assert_eq!(median(&OS468_STEPS) - median(&OS465_STEPS), 320);
    // Without the drift (rest at the os-465 level) the step would have been 271.73 ms = 11 305 tokens/s (lot 6 × 512).
    let step_wo_drift = rest_old + OS468.attn_back_us;
    assert_eq!(step_wo_drift, 271_728);
    assert_eq!(6 * 512 * 1_000_000 / u64::from(step_wo_drift), 11_305);
    // What is left to win in the attention backward: 2.0 % of the step; the step is 79 % `mm_tc_v4`.
    assert_eq!(OS468.attn_back_us * 1000 / OS468.step_us, 19);
    assert_eq!(OS468.mm_tc_v4_us * 1000 / OS468.step_us, 793);
}
