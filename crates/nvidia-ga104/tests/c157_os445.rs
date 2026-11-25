//! T247 (C-157 p.2, `docs/C-157_P2_OS445_PAS_MIXT.md`): os-445 — the fast mixed #513 step (`--mixt-513
//! --mixt-513-rapid`, 147 launches on `mm_tc_v4` at 64 KiB). The board's rows are copied unchanged into
//! `tests/rapoarte/os445-mixt-rapid.txt` (MĂSURAT by the OS, automatic cycle 22:02); this test reads the 10 (loss,
//! norm) pairs back and compares them with the two CPU oracles: `ORACLE_513_SYNTH` (F32) and `ORACLE_513_MIXED_SYNTH`
//! (the same step with the matmul inputs rounded to f16, session 238). Nothing runs on (or simulates) the board here.

use gpu_compute::train::{ORACLE_513_MIXED_SYNTH, ORACLE_513_SYNTH};

const OS445: &str = include_str!("rapoarte/os445-mixt-rapid.txt");

/// `… pasul k: pierderea L, norma gradientului N; …` → [(L, N); 10].
fn board() -> Vec<(f64, f64)> {
    OS445
        .lines()
        .filter_map(|l| {
            let r = l.split(": pasul ").nth(1)?;
            let l_ = r
                .split("pierderea ")
                .nth(1)?
                .split(',')
                .next()?
                .parse()
                .ok()?;
            let n_ = r
                .split("norma gradientului ")
                .nth(1)?
                .split(';')
                .next()?
                .parse()
                .ok()?;
            Some((l_, n_))
        })
        .collect()
}

fn worst(b: &[(f64, f64)], o: &[(f32, f32)], steps: core::ops::Range<usize>) -> (f64, f64) {
    steps.fold((0.0, 0.0), |(wl, wn), k| {
        let dl = ((b[k].0 - o[k].0 as f64) / o[k].0 as f64).abs();
        let dn = ((b[k].1 - o[k].1 as f64) / o[k].1 as f64).abs();
        (wl.max(dl), wn.max(dn))
    })
}

#[test]
fn t247_os445_rows_are_read() {
    let b = board();
    assert_eq!(b.len(), 10);
    assert_eq!(b[0], (9.8493, 3.2731));
    assert_eq!(b[9], (10.0178, 1.0812));
}

#[test]
fn t247_the_os_verdict_is_reproduced() {
    // The OS's line: against the F32 oracle, steps 1–9: loss 7.2e-3, norm 1.3e-2 → DIFERA.
    let (l, n) = worst(&board(), &ORACLE_513_SYNTH, 0..9);
    assert!((7.1e-3..7.3e-3).contains(&l), "{l:e}");
    assert!((1.2e-2..1.3e-2).contains(&n), "{n:e}");
}

#[test]
fn t247_steps_1_to_6_are_the_mixed_oracle() {
    let b = board();
    // M3, part one: steps 1–6 within the mixed step's tolerance (loss 1e-3, norm 2e-3) of the mixed oracle — and closer
    // to it than to the F32 oracle.
    let (lm, nm) = worst(&b, &ORACLE_513_MIXED_SYNTH, 0..6);
    let (lf, nf) = worst(&b, &ORACLE_513_SYNTH, 0..6);
    assert!(lm <= 1e-3 && nm <= 2e-3, "{lm:e} {nm:e}");
    assert!(
        lm <= 3e-5 && nm <= 2.5e-4,
        "MĂSURAT: 2,0e-5 / 2,2e-4; acum {lm:e} {nm:e}"
    );
    assert!(
        nm < nf && lm <= lf,
        "mixt {lm:e}/{nm:e} fata de F32 {lf:e}/{nf:e}"
    );
    // Steps 1–3 are the same 4 decimals.
    for k in 0..3 {
        assert_eq!(
            (b[k].0 as f32, b[k].1 as f32),
            ORACLE_513_MIXED_SYNTH[k],
            "pasul {}",
            k + 1
        );
    }
}

#[test]
fn t247_from_step_7_the_run_drifts_even_from_the_mixed_oracle() {
    // M3, part two (the prediction "steps 1–9 under 1e-3" MISSED): steps 7–9 leave the mixed oracle too — but stay
    // closer to it than to the F32 oracle in the loss.
    let b = board();
    let (lm, nm) = worst(&b, &ORACLE_513_MIXED_SYNTH, 6..9);
    let (lf, nf) = worst(&b, &ORACLE_513_SYNTH, 6..9);
    assert!(nm > 2e-3, "{nm:e}");
    assert!(
        lm < lf && nm < nf,
        "mixt {lm:e}/{nm:e} fata de F32 {lf:e}/{nf:e}"
    );
}

#[test]
fn t247_the_two_oracles_part_where_the_board_parts() {
    // M2: the mixed oracle against the F32 oracle — steps 1–5 under 1e-3, then apart (step 7 norm ≥ 1e-2).
    let o: Vec<(f64, f64)> = ORACLE_513_MIXED_SYNTH
        .iter()
        .map(|&(l, n)| (l as f64, n as f64))
        .collect();
    let (l, n) = worst(&o, &ORACLE_513_SYNTH, 0..5);
    assert!(l <= 1e-3 && n <= 1e-3, "{l:e} {n:e}");
    let (_, n7) = worst(&o, &ORACLE_513_SYNTH, 6..7);
    assert!(n7 >= 1e-2, "{n7:e}");
}
