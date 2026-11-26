//! C-157 — T266, CALCUL on the host (nothing simulated, nothing run on the board): the time predictions for `token32`
//! at G = 384 and G = 768 (`docs/C-157_TOKEN32_PREDICTII.md`), written BEFORE any board run of `token32`.
//!
//! Inputs, all MĂSURAT on the board by the OS: the per-kind time of one token of today's list (`lipita-ln-32`, 88
//! launches; hub `rezultate/os-automat/2026-11-26/ciclu-auto-20261126-2003.txt`), the cost of one launch start
//! (3.87 µs, os-459) and of one grid barrier (os-465: 2.025 µs at G = 384, 3.114 µs at G = 768). Everything derived is
//! DEDUS; the multiplier `m` (how much slower the matrix-vector products and the attention get with fewer warps per
//! SM) is a guess with a stated band, not a measurement.

use nvidia_ga104::kronos_token::{token32_groups_for_sms, token32_step_for};

/// (kind, µs per token, launches per token) — today's list, median of 18 tokens.
const TODAY: [(&str, f64, u32); 11] = [
    ("atentie", 250.0, 12),
    ("fc", 224.0, 12),
    ("qkv", 175.0, 12),
    ("proj", 143.0, 12),
    ("logit", 87.0, 1),
    ("out", 73.0, 12),
    ("ln1", 48.0, 12),
    ("ln2", 46.0, 12),
    ("copiere", 7.0, 1),
    ("embed", 5.0, 1),
    ("ln_final", 4.0, 1),
];
const START_US: f64 = 3.87;
const BARRIER_384_US: f64 = 2.025;
const BARRIER_768_US: f64 = 3.114;
const SMS: u32 = 48;

fn work(kinds: &[&str]) -> f64 {
    TODAY
        .iter()
        .filter(|(k, _, _)| kinds.contains(k))
        .map(|&(_, t, n)| t - f64::from(n) * START_US)
        .sum()
}

/// Board µs per token for `token32`: one start, 87 barriers, the warp-bound work × m, the rest as today.
fn token32_us(barrier: f64, m: f64) -> f64 {
    let bound = work(&["atentie", "fc", "qkv", "proj", "logit", "out"]);
    let rest = work(&["ln1", "ln2", "copiere", "embed", "ln_final"]);
    START_US + 87.0 * barrier + m * bound + rest
}

#[test]
fn t266_today_is_one_third_starts() {
    let launches: u32 = TODAY.iter().map(|t| t.2).sum();
    assert_eq!(launches, 88);
    let total: f64 = TODAY.iter().map(|t| t.1).sum();
    assert_eq!(total, 1062.0); // the report: ~1 066 µs per token
    let starts = f64::from(launches) * START_US;
    assert!((340.0..341.0).contains(&starts));
    // The three layer norms are pure start cost (layernorm256 works ~0.01 µs per launch beyond its start).
    let ln = work(&["ln1", "ln2", "ln_final"]);
    assert!(ln < 2.0);
    // The work that depends on how many warps run at once: products + attention = 716 µs.
    let bound = work(&["atentie", "fc", "qkv", "proj", "logit", "out"]);
    assert!((715.0..717.0).contains(&bound));
}

#[test]
fn t266_groups_and_steps() {
    // G = 384 → 8 per SM at the 32 KiB step; G = 768 → 16 per SM at the 64 KiB step (today's launches of the
    // products also run 16 one-warp blocks per SM: same warps per SM as G = 768).
    assert_eq!(token32_groups_for_sms(SMS, 8).unwrap(), 384);
    assert_eq!(token32_groups_for_sms(SMS, 16).unwrap(), 768);
    assert_eq!(token32_step_for(384, SMS).unwrap(), 32 * 1024);
    assert_eq!(token32_step_for(768, SMS).unwrap(), 64 * 1024);
}

#[test]
fn t266_predictions() {
    let today = 1062.0;
    // G = 384: half today's warps per SM. m = 1.0 (bandwidth-bound, warps do not matter) … 2.0 (latency-bound, one
    // row per warp at a time, as Little's law says for ~100 B in flight per warp today).
    let g384_best = token32_us(BARRIER_384_US, 1.0);
    let g384_worst = token32_us(BARRIER_384_US, 2.0);
    assert!((900.0..905.0).contains(&g384_best));
    assert!((1615.0..1620.0).contains(&g384_worst));
    // Break-even: m ≈ 1.22. Above it G = 384 is slower than today.
    let even = (today - token32_us(BARRIER_384_US, 0.0))
        / work(&["atentie", "fc", "qkv", "proj", "logit", "out"]);
    assert!((1.21..1.23).contains(&even));
    // G = 768: today's warps per SM; m = 0.9 (fewer tails between launches) … 1.15 (the strong loads of the
    // activations, the K/V caches through `.strong.sys`).
    let g768_best = token32_us(BARRIER_768_US, 0.9);
    let g768_worst = token32_us(BARRIER_768_US, 1.15);
    assert!((920.0..926.0).contains(&g768_best));
    assert!((1100.0..1106.0).contains(&g768_worst));
    // The central guesses: G = 384 at m = 1.5 → ~1.26 ms (slower); G = 768 at m = 1.0 → ~0.995 ms (−6 %).
    assert!((1255.0..1265.0).contains(&token32_us(BARRIER_384_US, 1.5)));
    assert!((990.0..1000.0).contains(&token32_us(BARRIER_768_US, 1.0)));
}
