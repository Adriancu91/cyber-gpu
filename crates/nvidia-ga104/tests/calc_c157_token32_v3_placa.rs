//! T296 (CALCUL on the board's MEASURED numbers): `token32_v3` on the board (os-498, cycle 27 nov 22:11,
//! hub `rezultate/os-automat/2026-11-27/ciclu-auto-20261127-2211.txt`, rows 83–106), judged against the predictions L1–L7
//! written before the code in `docs/C-157_TOKEN32_V3_LN.md` §2. Nothing here runs on a board; the numbers are copied
//! from the OS report. The split of the remainder (§2 of `docs/C-157_TOKEN32_V3_PLACA.md`) is DEDUS: it reuses the
//! per-kind work measured on `token32_v2c` in os-495 (T291), scaled from that run's last token to v2's median.

/// The same chain (os-498): median board µs per token, then with the driver.
const LIST_US: f64 = 1067.0;
const LIST_DRV_US: f64 = 2030.0;
const V2_384: (f64, f64) = (1936.0, 2577.0);
const V2_768: (f64, f64) = (1667.0, 2281.0);
const V3_384: (f64, f64) = (1682.0, 2314.0);
const V3_768: (f64, f64) = (1453.0, 2091.0);

/// os-495 (T291): `token32_v2c` last token, slowest work of the 25 LayerNorm (ln1 + ln2 + ln_final) and the barrier
/// over 86 barriers, in ns; the board's last token.
const V2C_LN_NS: f64 = 222_716.0 + 218_142.0 + 18_398.0;
const V2C_BARRIER_NS: f64 = 171_939.0;
const V2C_LAST_NS: f64 = 1_741_600.0;
const FUSED: f64 = 25.0;

#[test]
fn t296_l1_l5_l6() {
    // L1: TRECE at G = 384 and 768, 16 identical tokens, fingerprint 0x6c8501f0 (= the ordinary list), counter 18 / 18
    // (copied; no number). L6: words written per token, median of tokens 2–18 = 38 (the first 4 032), like v2.
    let (first, rest, list_words) = (4032.0_f64, 38.0_f64, 4032.0_f64);
    assert!(rest < list_words / 20.0);
    assert!(first <= 5632.0);
    // L5: G = 768 still faster than G = 384.
    assert!(V3_768.0 < V3_384.0);
    // v2 in the same chain repeats os-495 / 10:01 (1 666 / 1 668 at 768).
    assert!((V2_768.0 - 1666.0).abs() <= 3.0);
}

#[test]
fn t296_l3_l4_not_confirmed() {
    // L3: the board at G = 768 1,18–1,38 ms — NOT confirmed: 1,453 ms, 73 µs over the top.
    assert!(V3_768.0 > 1380.0);
    assert!((70.0..80.0).contains(&(V3_768.0 - 1380.0)));
    // v3 / v2 at 768 and 384: −214 µs (×1,147) and −254 µs (×1,151).
    assert_eq!(V2_768.0 - V3_768.0, 214.0);
    assert_eq!(V2_384.0 - V3_384.0, 254.0);
    // L4: ~505–560 tokens/s with the driver — NOT confirmed: 478.
    let tps = 1.0e6 / V3_768.1;
    assert!(tps < 505.0 && (477.0..479.0).contains(&tps), "{tps}");
    // The driver part stayed ~0,6 ms (v2 614, v3 638 µs).
    let drv_v2 = V2_768.1 - V2_768.0;
    let drv_v3 = V3_768.1 - V3_768.0;
    assert_eq!((drv_v2, drv_v3), (614.0, 638.0));
    // With the driver v3 is 3 % slower than the ordinary list (2 091 vs 2 030); on the board alone ×1,36.
    assert!((V3_768.1 / LIST_DRV_US - 1.03).abs() < 0.005);
    assert!((V3_768.0 / LIST_US - 1.362).abs() < 0.002);
}

#[test]
fn t296_the_fused_start_costs_about_11_us() {
    // DEDUS: what v3 removes from v2 = the 25 LayerNorm launches' work + 25 barriers, both measured on v2c (os-495) and
    // scaled from v2c's last token to v2's median in this chain.
    let scale = V2_768.0 * 1000.0 / V2C_LAST_NS;
    let removed_us = (V2C_LN_NS + V2C_BARRIER_NS * FUSED / 86.0) * scale / 1000.0;
    assert!((485.0..495.0).contains(&removed_us), "{removed_us}");
    // With nothing added v3 would be ~1,18 ms (= the bottom of L3) ...
    let floor = V2_768.0 - removed_us;
    assert!((1172.0..1182.0).contains(&floor), "{floor}");
    // ... so the 25 fused starts cost ~275–285 µs = ~11 µs each, predicted 1,5–4 µs (×3–7 over).
    let added = V3_768.0 - floor;
    let each = added / FUSED;
    assert!((271.0..281.0).contains(&added), "{added}");
    assert!((10.8..11.3).contains(&each), "{each}");
    assert!(each > 4.0 * 2.5);
}
