//! C-168 p.3 G7 — T304 C, CALCUL on the board numbers of os-503 (28 Nov 2026, the #513 step at lot 6 with the
//! products picked by shape on v8n / v8 / v7, `pack_mixed_fast_attn8` + `RATES_OS500`) against os-496 (the same step
//! all on v7) and against the prediction of T300 (`docs/C-168_OS500_V8.md` §5: ~133 ms). Note:
//! `docs/C-168_OS503_G7.md`.
//!
//! The numbers are copied from the hub (`rezultate/os-automat/2026-11-28/ciclu-auto-20261128-0608.txt`, the profile
//! of the step: median of 10 steps read whole; os-496: `.../2026-11-27/ciclu-auto-20261127-1954.txt`). Only the 8
//! longest labels are printed by the OS, so a label missing from a list is only bounded by the shortest one shown.
//! Nothing here runs on a board; the last part (the step with the guard) is DEDUS.

use nvidia_ga104::mm_tc_v8 as t8;

const SM: u32 = 48;

/// (rows per block, columns per block, blocks per SM) of v7 / v8n at 64 KiB.
const V7: (u32, u32, u32) = (64, 64, 4);
const V8N: (u32, u32, u32) = (128, 64, 3);

/// Fill of the waves: blocks / (waves · slots).
fn eff(m: u32, n: u32, t: (u32, u32, u32)) -> f64 {
    let blocks = (m.div_ceil(t.0) * n.div_ceil(t.1)) as f64;
    let slots = (SM * t.2) as f64;
    blocks / ((blocks / slots).ceil() * slots)
}

/// The 147 products of the step at lot 6 (as T294 / T300): (label, m, n, k, launches).
fn step_shapes() -> Vec<(&'static str, u32, u32, u32, u32)> {
    let (t, e, v) = (3072u32, 768u32, 16_384u32);
    vec![
        ("qkv", t, 3 * e, e, 12),
        ("out", t, e, e, 12),
        ("fc", t, 4 * e, e, 12),
        ("proj", t, e, 4 * e, 12),
        ("logit", t, v, e, 1),
        ("qkv-dX", t, e, 3 * e, 12),
        ("out-dX", t, e, e, 12),
        ("fc-dX", t, e, 4 * e, 12),
        ("proj-dX", t, 4 * e, e, 12),
        ("cap-dX", t, e, v, 1),
        ("qkv-dW", e, 3 * e, t, 12),
        ("out-dW", e, e, t, 12),
        ("fc-dW", e, 4 * e, t, 12),
        ("proj-dW", 4 * e, e, t, 12),
        ("cap-dW", e, v, t, 1),
    ]
}

/// os-503, lot 6: the step's board time (median, µs), the programs and the 7 matmul labels among the 8 longest.
const OS503_STEP_US: f64 = 140_673.0;
const OS503_V8N_US: f64 = 88_296.0;
const OS503_V7_US: f64 = 2_829.0;
const OS503_LABELS: [(&str, f64); 7] = [
    ("cap-dX", 12_781.0),
    ("fc-dW", 7_774.0),
    ("proj-dW", 7_757.0),
    ("fc", 7_690.0),
    ("qkv-dW", 7_540.0),
    ("proj-dX", 7_408.0),
    ("fc-dX", 6_969.0),
];
/// os-496, lot 6 (all on v7): mm_tc_v7 total and the labels among the 8 longest; the shortest of them (qkv-dW,
/// 7 868 µs) bounds every label not shown — the head's dX among them.
const OS496_V7_US: f64 = 98_099.0;
const OS496_LABELS: [(&str, f64); 7] = [
    ("proj-dW", 9_806.0),
    ("proj-dX", 9_625.0),
    ("fc", 9_451.0),
    ("fc-dW", 9_285.0),
    ("fc-dX", 8_772.0),
    ("proj", 8_372.0),
    ("qkv-dW", 7_868.0),
];
/// T300's prediction for the step with v8n (DEDUS, `docs/C-168_OS500_V8.md` §5).
const T300_STEP_MS: f64 = 133.0;
const T300_MM_MS: f64 = 83.7;

fn flop(m: u32, n: u32, k: u32, c: u32) -> f64 {
    2.0 * m as f64 * n as f64 * k as f64 * c as f64
}

/// v7's rate per unit of wave fill, calibrated on os-496's 98,1 ms (as T300).
fn r7() -> f64 {
    step_shapes()
        .iter()
        .map(|&(_, m, n, k, c)| flop(m, n, k, c) / eff(m, n, V7))
        .sum::<f64>()
        / (OS496_V7_US * 1e-6)
}

/// The wave model's time (µs) of one label on v8n at the measured rate (118 %).
fn model_v8n_us(label: &str) -> f64 {
    let &(_, m, n, k, c) = step_shapes().iter().find(|s| s.0 == label).unwrap();
    let f8n = t8::RATES_OS500.v8n_pct as f64 / 100.0;
    flop(m, n, k, c) / (f8n * r7() * eff(m, n, V8N)) * 1e6
}

fn model_v7_us(label: &str) -> f64 {
    let &(_, m, n, k, c) = step_shapes().iter().find(|s| s.0 == label).unwrap();
    flop(m, n, k, c) / (r7() * eff(m, n, V7)) * 1e6
}

/// C1: where the 7,7 ms between the prediction (~133) and the board (140,7) went. Every 12-launch label shown is at or
/// under the model (−11 … −3 %); the head's dX alone is ×3,7 the model (12,78 ms = 6,0 TFLOP/s against ~22).
#[test]
fn t304_c1_where_the_miss_is() {
    let mm = OS503_V8N_US + OS503_V7_US;
    assert!((mm - 91_125.0).abs() < 1.0);
    let miss_mm = mm * 1e-3 - T300_MM_MS;
    assert!((7.0..8.0).contains(&miss_mm), "{miss_mm}");
    let mut rest = 0.0;
    for &(label, us) in &OS503_LABELS {
        let model = model_v8n_us(label);
        let r = us / model;
        if label == "cap-dX" {
            assert!(r > 3.5, "{label} {r}");
            continue;
        }
        assert!((0.85..1.0).contains(&r), "{label} {r}");
        rest += us - model;
    }
    let cap = OS503_LABELS[0].1;
    let cap_model = model_v8n_us("cap-dX");
    let excess = (cap - cap_model) * 1e-3;
    // The head's dX alone explains more than the whole miss; the six labels shown gained 1,9 ms over the model.
    assert!(excess > miss_mm);
    assert!(rest < 0.0);
    let (_, m, n, k, _) = step_shapes()[9];
    let tflops = flop(m, n, k, 1) / (cap * 1e-6) / 1e12;
    assert!((5.9..6.2).contains(&tflops));
    println!(
        "kt c304 C1 CALCUL: inmultirile {:.1} ms (prezis {T300_MM_MS}, +{miss_mm:.1}); dX-ul capului {:.2} ms pe v8n = {tflops:.1} TFLOP/s (modelul {:.2} ms, x{:.1}) -> +{excess:.1} ms; celelalte 6 etichete aratate {:+.1} ms fata de model",
        mm * 1e-3,
        cap * 1e-3,
        cap_model * 1e-3,
        cap / cap_model,
        rest * 1e-3
    );
}

/// C2: v8n against v7 in the step itself (os-496 → os-503, the same labels): ×1,19 … ×1,30 on the shapes with full
/// waves (better than the 118 % of the probe), ×1,04 on qkv-dW (both last waves ¾ full); the head's dX the opposite.
#[test]
fn t304_c2_v8n_against_v7_in_the_step() {
    let mut full = Vec::new();
    for &(label, us503) in &OS503_LABELS {
        let Some(&(_, us496)) = OS496_LABELS.iter().find(|x| x.0 == label) else {
            continue;
        };
        let x = us496 / us503;
        if label == "qkv-dW" {
            assert!((1.0..1.1).contains(&x), "{x}");
        } else {
            assert!((1.18..1.31).contains(&x), "{label} {x}");
            full.push(x);
        }
    }
    assert_eq!(full.len(), 5);
    // The head's dX on v7 (os-496) was not among the 8 longest → below 7 868 µs; on v8n it is 12 781.
    let bound = OS496_LABELS.iter().map(|x| x.1).fold(f64::MAX, f64::min);
    let x_cap = bound / OS503_LABELS[0].1;
    assert!(x_cap < 0.62);
    println!(
        "kt c304 C2 CALCUL: v8n fata de v7 in pas: x{:.2} .. x{:.2} pe formele cu valuri pline (proba: 1,18), x{:.2} pe qkv-dW; dX-ul capului: v7 sub {:.2} ms (os-496), v8n {:.2} ms -> v8n de cel putin x{:.2} mai lent acolo",
        full.iter().cloned().fold(f64::MAX, f64::min),
        full.iter().cloned().fold(0.0, f64::max),
        OS496_LABELS[6].1 / OS503_LABELS[4].1,
        bound * 1e-3,
        OS503_LABELS[0].1 * 1e-3,
        1.0 / x_cap
    );
}

/// C3 (DEDUS, the prediction for the step with the guard, written before the board): the head's dX back on v7 at
/// between the model (4,09 ms) and os-496's bound (7,87) → the step 132,0 – 135,8 ms = 22 630 – 23 280 simb/s
/// (os-503: 140,7 / 21 837). Same bits as os-503 (T304 S3).
#[test]
fn t304_c3_step_with_the_guard() {
    let lo_cap = model_v7_us("cap-dX");
    let hi_cap = OS496_LABELS.iter().map(|x| x.1).fold(f64::MAX, f64::min);
    assert!((4_000.0..4_200.0).contains(&lo_cap), "{lo_cap}");
    let cap = OS503_LABELS[0].1;
    let lo = (OS503_STEP_US - cap + lo_cap) * 1e-3;
    let hi = (OS503_STEP_US - cap + hi_cap) * 1e-3;
    assert!(lo > 131.5 && hi < 136.0, "{lo} {hi}");
    assert!(hi < OS503_STEP_US * 1e-3 - 4.5);
    let sps = |ms: f64| 3072.0 / (ms * 1e-3);
    // Still above T300's ~133 at the top of the band, and short of Windows (~39 300) by ×1,7.
    assert!(sps(lo) < 23_300.0 && sps(hi) > 22_600.0);
    assert!(lo < T300_STEP_MS);
    println!(
        "kt c304 C3 DEDUS: cu garda (dX-ul capului pe v7, {:.2} .. {:.2} ms): pasul lot 6 {lo:.1} .. {hi:.1} ms = {:.0} .. {:.0} simb/s (os-503: {:.1} ms / 21837); aceiasi biti",
        lo_cap * 1e-3,
        hi_cap * 1e-3,
        sps(hi),
        sps(lo),
        OS503_STEP_US * 1e-3
    );
}
