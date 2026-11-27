//! C-157 — T285, CALCUL on the host (nothing simulated on a GPU model, nothing run on the board): the comparison of the
//! board run of `token32_v2` (hub `rezultate/2026-11-27-workstation-token32-v2/`, cycle log
//! `rezultate/os-automat/2026-11-27/ciclu-auto-20261127-1001.txt`, lines 54 and 66–97) with the predictions V1–V5 written
//! BEFORE the run in `docs/C-157_OS471.md` §3. Note: `docs/C-157_TOKEN32_V2_PLACA.md`.
//!
//! Inputs MĂSURAT on the board by the OS (the speed request accepted, P0 at the end: the chain ran inside the window):
//! today's list 1 067 µs per token (board), 2 083 µs with the driver; `token32` (repeat of os-471) 3 857 / 8 449 µs at
//! G = 384 and 4 109 / 8 700 µs at G = 768; `token32_v2` 1 969 / 2 578 µs at G = 384 and 1 668 / 2 281 µs at G = 768;
//! the list words written per token: 5 632 for `token32`, "348" for `token32_v2` (the OS prints total / tokens over the
//! 18 tokens, the first one whole). Everything below is DEDUS from those numbers.

const LIST_BOARD: f64 = 1067.0;
const LIST_HOST: f64 = 2083.0;
/// (G, board µs, with the driver µs)
const T32: [(u32, f64, f64); 2] = [(384, 3857.0, 8449.0), (768, 4109.0, 8700.0)];
const V2: [(u32, f64, f64); 2] = [(384, 1969.0, 2578.0), (768, 1668.0, 2281.0)];
/// T266 / T271's model inputs: one launch start, the barrier (os-465), the warp-bound work of today's list.
const START_US: f64 = 3.87;
const BARRIER: [f64; 2] = [2.025, 3.114];
const BOUND_US: f64 = 716.4;
/// os-471's x-traffic factor (cause 1, `docs/C-157_OS471.md` §2): x read from L2 by every row, ×4.55 the bytes.
const X_TRAFFIC: f64 = 4.55;
const WHOLE_WORDS: f64 = 5632.0;
const TOKENS: f64 = 18.0;

fn m(board: f64, barrier: f64) -> f64 {
    (board - START_US - 87.0 * barrier - 2.0) / BOUND_US
}

#[test]
fn t285_v3_v4_v5_against_the_board() {
    let (b384, b768) = (V2[0].1, V2[1].1);
    // V3: G = 384 0.9–1.75 ms → 1.969 ms, ~12 % past the slow end: NOT confirmed.
    assert!(b384 > 1750.0 && (1.12..1.13).contains(&(b384 / 1750.0)));
    // V4: G = 768 0.95–2.1 ms → 1.668 ms: confirmed. The order flipped against `token32`: 768 is now the faster one.
    assert!((950.0..=2100.0).contains(&b768));
    assert!(b768 < b384 - 250.0);
    assert!(
        T32[1].1 > T32[0].1,
        "token32 (v1): 768 was the slower one, again today"
    );
    // V5: the driver part 0.9–1.2 ms → ~0.61 ms at both G: NOT confirmed, better than predicted (one launch instead
    // of 88, so the pushbuffer and the records shrink too — the prediction only removed the whole-list rewrite).
    for (_, board, host) in V2 {
        let driver = host - board;
        assert!((605.0..615.0).contains(&driver), "driver part {driver}");
    }
    assert!(
        (1010.0..1020.0).contains(&(LIST_HOST - LIST_BOARD)),
        "today's list: ~1.02 ms"
    );
    // Against the plain list: on the board ×1.85 / ×1.56 slower, with the driver ×1.24 / ×1.10 slower.
    assert!((1.84..1.86).contains(&(b384 / LIST_BOARD)));
    assert!((1.56..1.57).contains(&(b768 / LIST_BOARD)));
    assert!((1.09..1.10).contains(&(V2[1].2 / LIST_HOST)));
    // Against `token32` in the same chain: ×1.96 (384) and ×2.46 (768) faster on the board.
    assert!((1.95..1.97).contains(&(T32[0].1 / b384)));
    assert!((2.46..2.47).contains(&(T32[1].1 / b768)));
}

#[test]
fn t285_cause_1_holds_at_768_and_the_rest_stays() {
    // The work multiplier m of T266's model (board = start + 87 barriers + m · 716 µs + 2 µs).
    let m32: Vec<f64> = T32.iter().zip(BARRIER).map(|(t, b)| m(t.1, b)).collect();
    let mv2: Vec<f64> = V2.iter().zip(BARRIER).map(|(t, b)| m(t.1, b)).collect();
    assert!((5.12..5.14).contains(&m32[0]) && (5.34..5.36).contains(&m32[1]));
    assert!((2.48..2.51).contains(&mv2[0]) && (1.93..1.95).contains(&mv2[1]));
    // Cause 1 alone (x through L1: the ×4.55 traffic becomes ×1) predicts m − 3.55.
    let pred: Vec<f64> = m32.iter().map(|x| x - (X_TRAFFIC - 1.0)).collect();
    let pred_us: Vec<f64> = pred
        .iter()
        .zip(BARRIER)
        .map(|(x, b)| START_US + 87.0 * b + 2.0 + x * BOUND_US)
        .collect();
    // G = 768: predicted ~1.57 ms, measured 1.668 ms — within ~6 %: cause 1 explains the drop there.
    assert!((1560.0..1575.0).contains(&pred_us[1]));
    assert!((V2[1].1 / pred_us[1] - 1.0).abs() < 0.07);
    // G = 384: predicted ~1.31 ms, measured 1.969 ms (+0.65 ms): at 8 warps per SM the work no longer hides its waits.
    assert!((1305.0..1320.0).contains(&pred_us[0]));
    assert!((640.0..670.0).contains(&(V2[0].1 - pred_us[0])));
    // The rest os-471 could not explain (m − 4.55 at 768, ~0.8) is still there with v2 (m − 1, ~0.94): L1 did not touch it.
    let rest32 = m32[1] - X_TRAFFIC;
    let restv2 = mv2[1] - 1.0;
    assert!((0.79..0.81).contains(&rest32) && (0.93..0.95).contains(&restv2));
    assert!(
        (restv2 * BOUND_US - 673.0).abs() < 5.0,
        "~0.67 ms of the 1.668 ms is the unexplained rest"
    );
}

#[test]
fn t285_the_words_and_the_prize() {
    // "348 words per token" = total / 18 (integer division) with the first token written whole (5 632 words):
    // the 17 tokens after it wrote 37.2–38.2 words each — T271's ≤ 48 (38 between two tokens) holds on the board.
    let lo = (348.0 * TOKENS - WHOLE_WORDS) / (TOKENS - 1.0);
    let hi = (349.0 * TOKENS - 1.0 - WHOLE_WORDS) / (TOKENS - 1.0);
    assert!((37.1..37.3).contains(&lo) && (38.1..38.3).contains(&hi));
    // At ~0.63 µs per BAR word (T271) that is ~24 µs per token, against 3.55 ms for the whole list.
    assert!((23.0..25.0).contains(&(38.0 * 0.63)));
    // Break-even with today's list (with the driver) at G = 768: board ≤ 2 083 − 613 = 1 470 µs, i.e. m ≤ ~1.67.
    let driver = V2[1].2 - V2[1].1;
    let board_even = LIST_HOST - driver;
    let m_even = m(board_even, BARRIER[1]);
    assert!((1465.0..1475.0).contains(&board_even) && (1.66..1.68).contains(&m_even));
    // The prize if the rest goes (m = 1): ~0.99 ms board + 0.61 ms driver = ~1.61 ms = ~620 tokens/s (today 480, ×1.29).
    let best = START_US + 87.0 * BARRIER[1] + 2.0 + BOUND_US + driver;
    assert!((1600.0..1612.0).contains(&best));
    let tps = 1e6 / best;
    assert!((620.0..626.0).contains(&tps));
    assert!((1.29..1.31).contains(&(LIST_HOST / best)));
    // Today v2 at 768 with the driver: ~438 tokens/s (−9 % against the list).
    assert!((438.0..439.0).contains(&(1e6 / V2[1].2)));
}

#[test]
fn t285_the_blocks_per_sm_ceiling() {
    // Why G = 768 cannot simply grow (DEDUS from the GA104 limits, sm_86): blocks of 32 threads = 1 warp per block,
    // at most 16 resident blocks per SM → 16 warps per SM is the ceiling for `token32` as it is (G = 48 × 16 = 768).
    // With 92 registers a warp takes 96 × 32 = 3 072 of the SM's 65 536 → 21 warps would fit by registers alone.
    let regs_per_warp = 96.0 * 32.0;
    assert!((21.0..22.0).contains(&(65536.0 / regs_per_warp)));
    assert_eq!(48 * 16, 768);
    // More warps per SM need blocks of 64 threads (a program change, not a QMD change): left for a later step.
}
