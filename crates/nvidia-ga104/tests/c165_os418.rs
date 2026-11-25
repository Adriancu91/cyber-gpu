//! T228 (C-165, `docs/C-165_CURSA_MM.md` §„os-418”): the #513 step of os-418 — the os-414 list plus the pick of the
//! os-416 race (`--mm-alegere`: all 147 products on `mm_tv_128x128`) — rows 158–218 of `ciclu-auto-20261125-0433.txt`,
//! copied unchanged into `tests/rapoarte/os418-pas.txt`, compared with os-414 / os-417 and with the race. On the
//! processor: the board times are the board's (measured by the OS); nothing here runs on the board.

use nvidia_ga104::mm_train_variants::{RACE_WINNER_513, VARIANTS, parse_race};
use nvidia_ga104::train_profile::{ParsedProfile, parse_profiles};

const OS418: &str = include_str!("rapoarte/os418-pas.txt");
const OS417_PROFILE: &str = include_str!("rapoarte/os417-profil.txt");
const OS414_PROFILE: &str = include_str!("rapoarte/os414-profil.txt");
const OS416_RACE: &str = include_str!("rapoarte/os416-cursa.txt");

/// The whole step (median of 10, us): os-414 (same list without the pick) and os-418 — from the `banc` rows.
const STEP_414_US: u64 = 474_272;
const PLATE_414_US: u64 = 463_594;
/// What session 219 predicted for os-418 (§„os-416”, P6): ~341 ms, interval 325–345 ms.
const PREDICTED_US: u64 = 341_000;

fn the_513_profile(text: &str) -> ParsedProfile {
    parse_profiles(text)
        .into_iter()
        .find(|p| p.context.contains("antrenare #513"))
        .expect("the #513 profile")
}

/// `| ms pe pas 338.394 |` → 338 394 us; `| placa ms pe pas 327.581 |` → 327 581 us.
fn banc_us(text: &str, key: &str) -> u64 {
    let line = text
        .lines()
        .find(|l| l.contains("banc antrenare:"))
        .unwrap();
    let field = line
        .split(" | ")
        .find(|f| f.starts_with(key))
        .unwrap_or_else(|| panic!("{key} in {line}"));
    let ms = field[key.len()..].trim();
    let (a, b) = ms.split_once('.').unwrap();
    a.parse::<u64>().unwrap() * 1000 + b.parse::<u64>().unwrap()
}

/// The losses and norms of the 10 steps, as printed.
fn banc_numbers(text: &str) -> (String, String) {
    let line = text
        .lines()
        .find(|l| l.contains("banc antrenare:"))
        .unwrap();
    let get = |k: &str| {
        line.split(" | ")
            .find(|f| f.starts_with(k))
            .unwrap()
            .trim()
            .to_string()
    };
    (get("pierderea "), get("norma "))
}

/// Check 1: one profile, the same launches as os-414/417, every product on `mm_tv_128x128`.
#[test]
fn t228_the_profile() {
    let p = the_513_profile(OS418);
    assert_eq!((p.steps, p.total_us, p.launches), (10, 327_581, 483));
    let mm = p.program(VARIANTS[RACE_WINNER_513].name).unwrap();
    assert_eq!((mm.us, mm.launches), (294_946, 147));
    assert!(p.program("mm_train_tile").is_none() && p.program("mm_train").is_none());
    assert!(p.programs_us().abs_diff(p.total_us) <= p.programs.len() as u64);
    assert_eq!(banc_us(OS418, "placa ms pe pas "), p.total_us);
    println!(
        "kt t228 os-418: placa {} us, {} lansari; mm_tv_128x128 {} us = {} promile din placa",
        p.total_us,
        p.launches,
        mm.us,
        mm.us * 1000 / p.total_us
    );
}

/// Check 2: the same numbers as os-414, digit for digit.
#[test]
fn t228_same_numbers_as_os414() {
    let (l418, n418) = banc_numbers(OS418);
    let (l414, n414) = banc_numbers(OS414_PROFILE);
    assert_eq!(l418, l414);
    assert_eq!(n418, n414);
    assert!(OS418.contains("TRECE (pasul #513 cu arena in VRAM = oracolul"));
}

/// Checks 3–4: the race ratio carries into the step; nothing else moved.
#[test]
fn t228_race_ratio_holds_in_the_step() {
    let a = the_513_profile(OS417_PROFILE);
    let b = the_513_profile(OS418);
    let mm_a = a.program("mm_train_tile").unwrap().us;
    let mm_b = b.program(VARIANTS[RACE_WINNER_513].name).unwrap().us;

    // The race (isolated, 1 785 MHz): Σ 128x128 × launches / Σ witness × launches.
    let race = parse_race(OS416_RACE);
    let sum = |v: usize| -> u64 {
        race.counts
            .iter()
            .map(|f| {
                let r = race
                    .rows
                    .iter()
                    .find(|r| (r.m, r.n, r.k, r.flags, r.variant) == (f.m, f.n, f.k, f.flags, v))
                    .unwrap();
                r.median_ns * u64::from(f.count)
            })
            .sum()
    };
    let (w, c) = (sum(0), sum(RACE_WINNER_513));
    let race_pm = c * 1000 / w;
    let step_pm = mm_b * 1000 / mm_a;
    assert_eq!(race_pm, 690);
    assert!(
        race_pm.abs_diff(step_pm) <= 10,
        "race {race_pm} ‰, step {step_pm} ‰"
    );

    let rest_a = a.total_us - mm_a;
    let rest_b = b.total_us - mm_b;
    assert!(rest_a.abs_diff(rest_b) <= 500, "{rest_a} vs {rest_b}");

    let out_414 = STEP_414_US - PLATE_414_US;
    let out_418 = banc_us(OS418, "ms pe pas ") - b.total_us;
    assert!(out_414.abs_diff(out_418) <= 1000, "{out_414} vs {out_418}");

    // Step time / isolated time of the same program: the witness and 128x128 alike (the clock, not the program).
    let iso_w = mm_a * 1000 / (w / 1000);
    let iso_c = mm_b * 1000 / (c / 1000);
    assert!(iso_w.abs_diff(iso_c) <= 15, "{iso_w} vs {iso_c}");
    println!(
        "kt t228 raportul cursei {race_pm} promile, in pas {step_pm}; restul placii {rest_a} -> {rest_b} us; in afara placii {out_414} -> {out_418} us; pas/izolat: martor {iso_w}, 128x128 {iso_c} promile"
    );
}

/// Check 5: the step against the prediction of session 219.
#[test]
fn t228_the_prediction() {
    let step = banc_us(OS418, "ms pe pas ");
    assert_eq!(step, 338_394);
    assert!((325_000..=345_000).contains(&step));
    assert!(step.abs_diff(PREDICTED_US) * 100 <= PREDICTED_US * 2);
    println!(
        "kt t228 pasul {step} us fata de ~{PREDICTED_US} prezis ({} promile); fata de os-414 {} promile",
        step as i64 * 1000 / PREDICTED_US as i64 - 1000,
        step * 1000 / STEP_414_US
    );
}
