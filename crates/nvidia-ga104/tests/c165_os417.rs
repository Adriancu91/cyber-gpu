//! T227 (C-165, `docs/C-165_CURSA_MM.md` §„os-417”): the board probe of os-417 — `mm_tv_64x64k8` alone on the qkv
//! shape, mapped last but with `qmd::program_map_bytes` (rows 214–222 of `ciclu-auto-20261125-0348.txt`, copied
//! unchanged into `tests/rapoarte/os417-proba64k8.txt`) — read with `parse_race`, then put together with the os-416
//! race. On the processor: the board times are the board's (measured by the OS); nothing here runs on the board.

use nvidia_ga104::mm_train_variants::{
    self as mv, MmChoice, RACE_CLEARED_OS417, RACE_HOLD, RACE_WINNER_513, VARIANTS, parse_race,
};
use nvidia_ga104::qmd;

const OS417: &str = include_str!("rapoarte/os417-proba64k8.txt");
const OS416: &str = include_str!("rapoarte/os416-cursa.txt");
const PROFILE_417: &str = include_str!("rapoarte/os417-profil.txt");
/// What the OS printed in os-417: "pasul ar scadea cu ~12854 us" (DEDUS, from the isolated times).
const OS_SAVED_US: u64 = 12_854;
/// The plate's step time (median of 10) with the same list (`--placi3 --mm-placi --colsum-impartit`): os-414, os-417.
const PLATE_414_US: u64 = 463_594;
const PLATE_417_US: u64 = 463_732;
const QKV: (u32, u32, u32, u32) = (1024, 2304, 768, 6);
const K8: usize = 4;

/// Predictions 1–2: two timed rows, both "=", qkv → `mm_tv_64x64k8`, the OS's saving.
#[test]
fn t227_the_real_os417_probe() {
    let p = parse_race(OS417);
    assert_eq!(p.rows.len(), 2, "timed rows");
    assert_eq!((p.untimed, p.unknown), (0, 0));
    assert!(p.rows.iter().all(|r| r.same_bits), "both = the witness");
    assert_eq!(p.counts.len(), 1);
    assert_eq!(p.counts[0].count, 12);
    assert_eq!(VARIANTS[K8].name, "mm_tv_64x64k8");

    let pick = p.choice().unwrap();
    let (m, n, k, f) = QKV;
    assert_eq!(pick.of(m, n, k, f), K8);
    let saved_us = p.saved_ns(&pick) / 1000;
    assert!(
        saved_us.abs_diff(OS_SAVED_US) <= 2,
        "ours {saved_us} us, the OS {OS_SAVED_US} us"
    );
    let med = |v: usize| p.rows.iter().find(|r| r.variant == v).unwrap().median_ns;
    println!(
        "kt t227 os-417: 2 randuri, ambele =; qkv: martor {} us, 64x64k8 {} us ({} promile din martor); economia {saved_us} us (OS: {OS_SAVED_US})",
        med(0) / 1000,
        med(K8) / 1000,
        med(K8) * 1000 / med(0)
    );
}

/// Prediction 3: os-416 + os-417 together — qkv on `64x64k8`, the other 14 shapes on `128x128`.
#[test]
fn t227_os416_and_os417_together() {
    let a = parse_race(OS416);
    let b = parse_race(OS417);
    // Same race clock: the witness on qkv within 1 ‰ in both runs.
    let w = |p: &mv::ParsedRace| {
        p.rows
            .iter()
            .find(|r| (r.m, r.n, r.k, r.flags, r.variant) == (QKV.0, QKV.1, QKV.2, QKV.3, 0))
            .unwrap()
            .median_ns
    };
    assert!(
        w(&a).abs_diff(w(&b)) * 1000 <= w(&a),
        "the witness moved by more than 1 ‰"
    );

    let mut rows = a.rows.clone();
    rows.extend(b.rows.iter().copied());
    let pick = MmChoice::from_race(&rows).unwrap();
    let mut on_k8 = 0;
    for f in &a.counts {
        let v = pick.of(f.m, f.n, f.k, f.flags);
        if (f.m, f.n, f.k, f.flags) == QKV {
            assert_eq!(v, K8, "{f:?}");
            on_k8 += 1;
        } else {
            assert_eq!(v, RACE_WINNER_513, "{f:?}");
        }
    }
    assert_eq!(on_k8, 1);
    assert_eq!(pick.entries().count(), 15);
    assert_eq!(pick.programs().len(), 2, "mm_tv_128x128 + mm_tv_64x64k8");

    // The saving of the merged pick over the os-416 pick, on the step's counts (isolated times, DEDUS).
    let only_416 = a.choice().unwrap();
    let saved = |c: &MmChoice| {
        let mut s = 0u64;
        for f in &a.counts {
            let v = c.of(f.m, f.n, f.k, f.flags);
            let t = |v: usize| {
                rows.iter()
                    .filter(|r| (r.m, r.n, r.k, r.flags, r.variant) == (f.m, f.n, f.k, f.flags, v))
                    .map(|r| r.median_ns)
                    .min()
                    .unwrap()
            };
            s += (t(0) - t(v)) * u64::from(f.count);
        }
        s / 1000
    };
    let (s416, s_both) = (saved(&only_416), saved(&pick));
    assert!(s_both > s416);
    println!(
        "kt t227 os-416 + os-417: qkv pe 64x64k8, 14 forme pe 128x128; economia {s_both} us fata de {s416} us cu os-416 singur (+{} us; DEDUS)",
        s_both - s416
    );
}

/// Prediction 4: the os-417 step = the os-414 step (same list) within 1 ‰; the products are still ~93 % of the plate.
#[test]
fn t227_the_step_repeats() {
    assert!(PLATE_414_US.abs_diff(PLATE_417_US) * 1000 <= PLATE_414_US);
    let line = PROFILE_417
        .lines()
        .find(|l| l.contains("profilul pasului"))
        .unwrap();
    assert!(line.contains("463732 us, 483 lansari"), "{line}");
    let mm = PROFILE_417
        .lines()
        .find(|l| l.contains("  mm_train_tile "))
        .unwrap();
    assert!(
        mm.contains("431180 us") && mm.contains("147 lansari"),
        "{mm}"
    );
    println!(
        "kt t227 pasul os-417 {PLATE_417_US} us, os-414 {PLATE_414_US} us ({} promile); inmultirile 431180 us = {} promile din placa",
        PLATE_414_US.abs_diff(PLATE_417_US) * 1000 / PLATE_414_US,
        431_180 * 1000 / PLATE_417_US
    );
}

/// Prediction 5 + the list: the guard page is there for `64x64k8`; the full race list has all 5, `RACE_HOLD` is unchanged.
#[test]
fn t227_guard_page_and_race_lists() {
    let len = VARIANTS[K8].code.len() as u64;
    assert_eq!(len, 16_368);
    assert_eq!(
        qmd::program_page_slack(len),
        16,
        "os-413: 16 B after the code"
    );
    assert!(
        qmd::program_map_bytes(len) - len >= 4096,
        "os-417: a page after the code"
    );

    assert_eq!(RACE_HOLD, [4], "unchanged (additions only)");
    assert_eq!(RACE_CLEARED_OS417, [4]);
    assert_eq!(mv::race_list().count(), 4);
    let full: Vec<usize> = mv::race_list_full().map(|(i, _)| i).collect();
    assert_eq!(full, vec![0, 1, 2, 3, 4]);
}
