//! T293 (CALCUL on the board's MEASURED numbers): os-496, the #513 step with `mm_tc_v7` in it (`--pas-v7`, note
//! C-157_P2zzr), two rows in one boot: lot 2 with the mixed oracle (cycle 19:48, `ciclu-auto-20261127-1948.txt`) and
//! lot 6 with two slots and the dashboard held (cycle 19:54, `ciclu-auto-20261127-1954.txt`). Nothing here runs on a
//! board; the numbers are copied from the OS reports and checked against what was written before the run (P2zzr:
//! P1 the 147 picks on v7 and the same numbers as the oracle; P2 board 137-162 ms at lot 6). It also writes down the
//! step profile that is the "before" measurement of C-168 (points 3-5).

const TOKENS6: f64 = 6.0 * 512.0;

fn sps(us: f64, tokens: f64) -> f64 {
    tokens / (us * 1e-6)
}

/// The 147 matmuls of #513 at lot 6 (T268): 1.797 TFLOP per step.
const MM_TFLOP_LOT6: f64 = 1.797;

/// P1: lot 2, the picks and the oracle.
#[test]
fn t293_p1_lot2_picks_and_oracle() {
    let picks_v7 = 147u32;
    let (picks_v6, picks_v5) = (0u32, 0u32);
    assert_eq!((picks_v7, picks_v6, picks_v5), (147, 0, 0), "P1: alegerile");
    // "ACELEASI NUMERE": max deviation of the loss 9.9e-8, of the norm 6.3e-7, of the weights 8.8e-6 (tolerance 1e-3).
    let (dl, dn, dw) = (9.9e-8f64, 6.3e-7f64, 8.8e-6f64);
    assert!(dl < 1e-3 && dn < 2e-3 && dw < 1e-3);
    // Lot 2: board 57.127 ms, mm_tc_v7 33.928 ms (59.3 %) for a third of the lot-6 work.
    let (board2, mm2) = (57_127.0f64, 33_928.0f64);
    let tflops2 = MM_TFLOP_LOT6 / 3.0 / (mm2 * 1e-6);
    assert!((17.0..18.5).contains(&tflops2), "{tflops2}");
    println!(
        "kt c293 P1 CONFIRMATA: lot 2 alegerile v7 147 / v6 0 / v5 0, oracolul mixt ACELEASI NUMERE (pierderea {dl:.1e}, norma {dn:.1e}); placa {:.2} ms, mm_tc_v7 {:.2} ms = {tflops2:.1} TFLOP/s",
        board2 / 1e3,
        mm2 / 1e3
    );
}

/// P2: lot 6, the board against the band 137-162 ms; the matmuls x1.54 against os-487 (v5), the rest unchanged.
#[test]
fn t293_p2_lot6_board_in_band() {
    let (board, full) = (147_810.0f64, 147_722.0f64);
    assert!(
        (137_000.0..=162_000.0).contains(&board),
        "P2: banda 137-162 ms"
    );
    // os-487 (v5): board 200.952 ms, mm_tc_v5 75.4 % of it.
    let board487 = 200_952.0f64;
    let mm487 = board487 * 0.754;
    let mm496 = 98_099.0f64;
    let speedup = mm487 / mm496;
    assert!(
        (1.35..=1.72).contains(&speedup),
        "P2 socoteala: inmultirile x1,35-1,72 ({speedup})"
    );
    // The rest of the step was predicted unchanged (~49.4 ms): 49.7 ms (+0.6 %).
    let rest = board - mm496;
    let rest487 = board487 - mm487;
    assert!(
        ((rest - rest487) / rest487).abs() < 0.02,
        "{rest} {rest487}"
    );
    let tflops = MM_TFLOP_LOT6 / (mm496 * 1e-6);
    assert!((18.0..18.6).contains(&tflops), "{tflops}");
    // Whole step = board (two slots); the record x1.36 over 15 301 (os-487).
    assert!(((full - board) / board).abs() < 0.001);
    let s = sps(full, TOKENS6);
    assert!((20_700.0..20_900.0).contains(&s), "{s}");
    let gain = s / 15_301.0;
    assert!((1.34..1.38).contains(&gain));
    println!(
        "kt c293 P2 CONFIRMATA: lot 6 placa {:.2} ms (banda 137-162, mijloc ~145); pasul intreg {:.2} ms = {s:.0} simb/s (x{gain:.2} fata de os-487); mm_tc_v7 {:.2} ms = {tflops:.2} TFLOP/s, x{speedup:.3} fata de v5 in pas; restul {:.1} ms (os-487 {:.1})",
        board / 1e3,
        full / 1e3,
        mm496 / 1e3,
        rest / 1e3,
        rest487 / 1e3
    );
}

/// The "before" profile of C-168 (lot 6, 147.81 ms): where the time sits per program family, and how far the matmuls
/// are from the board's peak. The peak: 48 SM x 4 tensor units x 64 f16 FMA (f32 sums) x 2 FLOP x clock.
#[test]
fn t293_profile_before_c168() {
    let board = 147_810.0f64;
    let progs: [(&str, f64); 17] = [
        ("mm_tc_v7", 98_099.0),
        ("attn_fwd_tile3", 16_149.0),
        ("attn_back_tile0_tc_v4", 6_400.0),
        ("attn_back_tile_tc16_v4", 5_466.0),
        ("opt_step", 4_788.0),
        ("gelu_tr", 4_259.0),
        ("colsum_part", 3_367.0),
        ("vadd", 2_468.0),
        ("ln_back", 1_636.0),
        ("xent", 1_453.0),
        ("ln_fwd", 1_429.0),
        ("colsum", 1_361.0),
        ("gnorm", 735.0),
        ("emb_back", 121.0),
        ("emb_fwd", 62.0),
        ("copy_b128", 35.0),
        ("mm_train", 5.0),
    ];
    let sum: f64 = progs.iter().map(|p| p.1).sum();
    // The per-program medians add up to the board within 0.5 %.
    assert!(((sum - board) / board).abs() < 0.005, "{sum}");
    let small: f64 = progs[4..].iter().map(|p| p.1).sum();
    assert!((21_000.0..22_500.0).contains(&small), "{small}");
    // Peak at the measured median clock 1 935 MHz. Public GA104 sheet: dense f16 with f32 sums = 43.5 TFLOP/s at the
    // 1 770 MHz boost on 48 SM -> 512 FLOP per SM per clock (4 tensor units x 64 FMA x 2).
    let peak = 48.0 * 4.0 * 64.0 * 2.0 * 1.935e9 / 1e12;
    assert!((47.0..48.0).contains(&peak), "{peak}");
    let tflops = MM_TFLOP_LOT6 / (98_099.0 * 1e-6);
    let frac = tflops / peak;
    assert!((0.37..0.40).contains(&frac), "{frac}");
    // C-168 p.3 target 28 TFLOP/s -> the same matmuls in 64.2 ms, the step 113.9 ms = 26 960 simb/s (DEDUS).
    let mm_at_28 = MM_TFLOP_LOT6 / 28.0 * 1e6;
    let step_at_28 = board - 98_099.0 + mm_at_28;
    println!(
        "kt c293 profil CALCUL: mm_tc_v7 {:.1} ms ({:.1} %), atentia inainte {:.1} ms, atentia inapoi {:.1} ms, marunte {:.1} ms; varful {peak:.1} TFLOP/s la 1 935 MHz, v7 in pas {tflops:.1} = {:.0} %; la 28 TFLOP/s: inmultirile {:.1} ms, pasul {:.1} ms = {:.0} simb/s (DEDUS)",
        98.099,
        98_099.0 / board * 100.0,
        16.149,
        (6_400.0 + 5_466.0) / 1e3,
        small / 1e3,
        frac * 100.0,
        mm_at_28 / 1e3,
        step_at_28 / 1e3,
        sps(step_at_28, TOKENS6)
    );
}
