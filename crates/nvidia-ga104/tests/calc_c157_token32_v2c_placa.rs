//! T291 (CALCUL on the board's MEASURED numbers): `token32_v2c` on the board (os-495, cycle 18:06,
//! `ciclu-auto-20261127-1806.txt`, rows 83–140), judged against the predictions C1–C8 written before the code in
//! `docs/C-157_TOKEN32_CEAS.md` §3. Nothing here runs on a board; the numbers are copied from the OS report (the `kc`
//! rows = the clock area of the LAST token, already converted to ns by `token_clock::ClockSummary::rows`).
//! The predictions D1–D6 for the next program are DEDUS (`docs/C-157_TOKEN32_V2C_PLACA.md` §4).

/// The same chain, G = 768 (median board µs per token): token32_v2, token32_v2c, the ordinary list.
const V2_US: f64 = 1666.0;
const V2C_US: f64 = 1676.0;
const LIST_US: f64 = 1067.0;

/// `kc` summary rows (last token).
const BOARD_LAST_NS: f64 = 1_741_600.0;
const WINDOWS_NS: f64 = 1_738_379.0;
const COPY_NS: f64 = 3_221.0;
const BARRIER_NS: f64 = 171_939.0;
const MHZ: f64 = 1951.0;
const POLLS_X10: f64 = 265.0;
const OPS: f64 = 87.0;
const GROUPS: f64 = 768.0;

/// Per kind: (name, ops, window, slowest work, mean work) in ns, summed over the token; then the ordinary list's
/// median µs per token for the same kind (os-495 "timpul pe fel de lansare", launch starts included).
const KINDS: [(&str, f64, f64, f64, f64, f64); 10] = [
    ("embed", 1.0, 9_610.0, 7_763.0, 778.0, 5.0),
    ("ln1", 12.0, 242_190.0, 222_716.0, 4_061.0, 48.0),
    ("qkv", 12.0, 201_181.0, 171_350.0, 156_539.0, 175.0),
    ("atentie", 12.0, 423_505.0, 403_314.0, 9_699.0, 250.0),
    ("out", 12.0, 95_470.0, 63_132.0, 56_056.0, 73.0),
    ("ln2", 12.0, 237_721.0, 218_142.0, 4_101.0, 46.0),
    ("fc", 12.0, 255_892.0, 234_437.0, 204_250.0, 224.0),
    ("proj", 12.0, 162_930.0, 138_952.0, 116_226.0, 143.0),
    ("ln_final", 1.0, 20_018.0, 18_398.0, 340.0, 4.0),
    ("logit", 1.0, 89_857.0, 88_229.0, 79_262.0, 87.0),
];

fn kind(name: &str) -> (f64, f64, f64, f64, f64) {
    let k = KINDS.iter().find(|k| k.0 == name).unwrap();
    (k.1, k.2, k.3, k.4, k.5)
}

#[test]
fn t291_area_is_consistent() {
    // The windows are the sum of the 87 windows; slowest work + barrier = windows; + the copy = the board's last token.
    let w: f64 = KINDS.iter().map(|k| k.2).sum();
    assert!((w - WINDOWS_NS).abs() < 20.0, "{w}");
    let work: f64 = KINDS.iter().map(|k| k.3).sum();
    assert!((work + BARRIER_NS - WINDOWS_NS).abs() < 20.0);
    assert!((WINDOWS_NS + COPY_NS - BOARD_LAST_NS).abs() < 1.0);
    assert_eq!(KINDS.iter().map(|k| k.1).sum::<f64>(), OPS);
}

#[test]
fn t291_c1_to_c8() {
    // C1: TRECE, 16 identical tokens, fingerprint 0x6c8501f0, counter 18 / 18 (copied from the report; no number here).
    // C2: the clock costs ≤ +3 %.
    let c2 = V2C_US / V2_US - 1.0;
    assert!(c2 <= 0.03 && c2 > 0.0, "C2 {c2}");
    // C3: 1,4–2,0 GHz.
    assert!((1400.0..=2000.0).contains(&MHZ));
    // C4: the clean barrier 0,14–0,40 ms; per barrier.
    assert!((140_000.0..=400_000.0).contains(&BARRIER_NS));
    let per_barrier = BARRIER_NS / (OPS - 1.0);
    assert!((1_900.0..2_100.0).contains(&per_barrier), "{per_barrier}");
    // C5: the products' slowest work ≥ 0,40 ms over the list without starts (~0,514 ms) — NOT confirmed.
    let mm: f64 = ["qkv", "out", "fc", "proj", "logit"]
        .iter()
        .map(|n| kind(n).2)
        .sum();
    let over = mm - 514_000.0;
    assert!(over < 400_000.0, "C5 {over}");
    assert!((180_000.0..185_000.0).contains(&over));
    // ... and against the same list in the same run (starts included) the products are EQUAL (±1 %).
    let mm_list: f64 = ["qkv", "out", "fc", "proj", "logit"]
        .iter()
        .map(|n| kind(n).4)
        .sum::<f64>()
        * 1000.0;
    assert!((mm / mm_list - 1.0).abs() < 0.01, "{}", mm / mm_list);
    // C6: the 25 layernorm256 = 0,05–0,20 ms — NOT confirmed: ~0,46 ms.
    let ln: f64 = ["ln1", "ln2", "ln_final"].iter().map(|n| kind(n).2).sum();
    assert!(ln > 200_000.0, "C6 {ln}");
    assert!((455_000.0..465_000.0).contains(&ln));
    // C7: polls per workgroup and barrier ≥ 20.
    const { assert!(POLLS_X10 / 10.0 >= 20.0) };
    // C8: the 12 attentions' window ≤ ×1,3 the list without starts (~0,20 ms) — NOT confirmed: ×2,1.
    let att = kind("atentie").1;
    assert!(att / 200_000.0 > 1.3);
    assert!((2.10..2.13).contains(&(att / 200_000.0)));
    println!(
        "kt c291 os-495 CALCUL: C2 +{:.1} %; C3 {MHZ} MHz; C4 bariera {:.3} ms ({:.2} us pe bariera); C5 inmultirile {:.3} ms (+{:.3} peste lista fara porniri; = lista cu porniri x{:.3}); C6 layernorm {:.3} ms; C7 {:.1} citiri; C8 atentia x{:.2}",
        c2 * 100.0,
        BARRIER_NS / 1e6,
        per_barrier / 1e3,
        mm / 1e6,
        over / 1e6,
        mm / mm_list,
        ln / 1e6,
        POLLS_X10 / 10.0,
        att / 200_000.0
    );
}

/// Where the ~0,67 ms over the list go (last token vs the list's median per kind, starts included).
#[test]
fn t291_where_the_rest_is() {
    let mut extra = Vec::new();
    for k in KINDS {
        extra.push((k.0, k.3 / 1000.0 - k.5));
    }
    let get = |n: &str| extra.iter().find(|e| e.0 == n).unwrap().1;
    let ln = get("ln1") + get("ln2") + get("ln_final");
    let att = get("atentie");
    let mm = get("qkv") + get("out") + get("fc") + get("proj") + get("logit");
    let bar = BARRIER_NS / 1000.0;
    // copy: 3,2 µs here vs 7 in the list; embed +2,8.
    let rest = get("embed") + (COPY_NS / 1000.0 - 7.0);
    let total = ln + att + mm + bar + rest;
    // The list's per-kind medians add up to 1 062 µs, its token median is 1 067: 5 µs of rounding between them.
    assert!(
        (total - (BOARD_LAST_NS / 1000.0 - LIST_US)).abs() < 6.0,
        "{total}"
    );
    assert!((355.0..365.0).contains(&ln), "{ln}");
    assert!((150.0..156.0).contains(&att), "{att}");
    assert!(mm.abs() < 10.0, "{mm}");
    // One workgroup works on layernorm (mean ≈ max / G + the empty loop of the others) and 12 on attention.
    let (_, _, ln1_max, ln1_mean, _) = kind("ln1");
    let empty_per_op = (ln1_mean - ln1_max / GROUPS) / 12.0;
    assert!((250.0..350.0).contains(&empty_per_op), "{empty_per_op}");
    let (_, _, a_max, a_mean, _) = kind("atentie");
    let working = (a_mean - 12.0 * empty_per_op) / a_max * GROUPS;
    assert!((10.0..16.0).contains(&working), "{working}");
    // The imbalance of the products (slowest − mean), summed: ~84 µs.
    let imb: f64 = ["qkv", "out", "fc", "proj", "logit"]
        .iter()
        .map(|n| kind(n).2 - kind(n).3)
        .sum::<f64>()
        / 1000.0;
    assert!((80.0..88.0).contains(&imb), "{imb}");
    println!(
        "kt c291 restul CALCUL: layernorm +{ln:.0} us, atentia +{att:.0} us, bariera {bar:.0} us, inmultirile {mm:+.0} us, altele {rest:+.0} us = +{total:.0} us; grupuri care lucreaza la atentie ~{working:.0}; dezechilibrul inmultirilor {imb:.0} us"
    );
}

/// D1–D3 (DEDUS): the board per token if layernorm and attention cost what they cost in the list.
#[test]
fn t291_next_bands() {
    let ln = ["ln1", "ln2", "ln_final"]
        .iter()
        .map(|n| kind(n).2)
        .sum::<f64>()
        / 1000.0;
    let att = kind("atentie").2 / 1000.0;
    let board = BOARD_LAST_NS / 1000.0;
    // D1: layernorm folded into the next product (25 operations and 25 barriers fewer): − ln + a recomputation in every
    // group of 0,5–1,5× the list's layernorm (98 µs), − 25 barriers of ~2 µs.
    let per_barrier = BARRIER_NS / 1000.0 / (OPS - 1.0);
    let d1_fast = board - ln + 0.5 * 98.0 - 25.0 * per_barrier;
    let d1_slow = board - ln + 1.5 * 98.0 - 25.0 * per_barrier;
    assert!((1_280.0..1_300.0).contains(&d1_fast) && (1_375.0..1_395.0).contains(&d1_slow));
    // D2: attention on 64 groups per head (768): its work falls to 1,0–1,5× the list's 0,20 ms
    // (its 12 barriers stay).
    let barriers12 = 12.0 * per_barrier;
    let d2_fast = d1_fast - att + 200.0 + barriers12;
    let d2_slow = d1_slow - att + 300.0 + barriers12;
    assert!((1_095.0..1_110.0).contains(&d2_fast) && (1_290.0..1_310.0).contains(&d2_slow));
    // With the driver part of v2 (~0,61 ms): tokens per second.
    let sps = |us: f64| 1e6 / (us + 610.0);
    assert!(sps(d2_slow) > 1e6 / 2066.0);
    println!(
        "kt c291 D DEDUS: D1 {d1_fast:.0}-{d1_slow:.0} us; D1+D2 {d2_fast:.0}-{d2_slow:.0} us = {:.0}-{:.0} simb/s cu driverul (lista 484)",
        sps(d2_slow),
        sps(d2_fast)
    );
}
