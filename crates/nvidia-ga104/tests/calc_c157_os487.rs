//! T286 (CALCUL on the board's MEASURED numbers): os-487, the #513 step at lot 6 with two slots, `mm_tc_v5` and the
//! dashboard values held in the journal (note C-157_P2zzj, `Journal::set_hold_values`), cycle 12:24
//! (`ciclu-auto-20261127-1224.txt`, `gpu-jurnal-177.txt`). Compared with os-481 lot 6 (cycle 07:49, dashboard written
//! value by value) and the record run without the dashboard (cycle 08:24). Nothing here runs on a board; the numbers
//! are copied from the OS reports and checked against what was written before the run (P2zzj: whole step
//! ~205-215 ms, board ~203 ms, the same losses, "outside the board" under ~12 ms).

const TOKENS: f64 = 6.0 * 512.0;

fn sps(us: f64) -> f64 {
    TOKENS / (us * 1e-6)
}

/// The losses and norms of the 10 steps are the same digits as os-481 lot 6 (same program, same synthetic batches).
#[test]
fn t286_same_losses_as_os481() {
    let os487 = [
        9.8716, 9.8592, 9.8780, 9.8980, 9.9305, 9.9805, 9.9746, 9.9675, 10.0021, 9.9387,
    ];
    let os481 = [
        9.8716, 9.8592, 9.8780, 9.8980, 9.9305, 9.9805, 9.9746, 9.9675, 10.0021, 9.9387,
    ];
    let n487 = [
        1.8486, 0.6506, 0.5725, 0.6570, 0.5450, 0.5405, 0.5349, 0.5389, 0.5369, 0.5257,
    ];
    let n481 = [
        1.8486, 0.6506, 0.5725, 0.6570, 0.5450, 0.5405, 0.5349, 0.5389, 0.5369, 0.5257,
    ];
    assert_eq!(os487, os481, "P2zzj: aceleasi pierderi");
    assert_eq!(n487, n481, "P2zzj: aceleasi norme");
    // The "NU TRECE" of the report is the synthetic-batch rule "last below first" (known since os-465 / os-468):
    // random tokens cannot be learned, so the loss drifts up from ~ln 16384. Step 1 within 2 % of ln 16384.
    let ln_v = (16_384.0f64).ln();
    assert!((os487[0] - ln_v).abs() / ln_v < 0.02);
    assert!(
        os487[9] > os487[0],
        "criteriul pe loturi sintetice: NU TRECE, asteptat"
    );
    println!(
        "kt c286 pierderi CALCUL: aceleasi 10 pierderi si norme ca os-481 lot 6; pasul 1 {:.4} (ln V {ln_v:.4}); ultima > prima = NU TRECE asteptat pe loturi sintetice",
        os487[0]
    );
}

/// The whole step against the band written before (P2zzj: 205-215 ms): it came out below it, equal to the board.
#[test]
fn t286_whole_step_equals_board() {
    let (board, full) = (200_952.0f64, 200_768.0f64);
    let full_481 = 274_281.0f64;
    let (board_rec, full_rec) = (200_400.0f64, 200_300.0f64); // cycle 08:24, no dashboard (REZULTAT os-481 record)
    // P2zzj band for the whole step: missed, on the good side.
    assert!(full < 205_000.0, "sub banda 205-215 ms (in bine)");
    // "outside the board" = 0 us in the report; the whole step equals the board within 0.1 %.
    assert!(((full - board) / board).abs() < 0.001);
    // The +7 ms of the P2zzj model was the first step only (step 1: 225 504 - 218 109 = 7.4 ms; os-481: 7.0 ms).
    let step1_extra = 225_504.0 - 218_109.0;
    assert!((7_000.0..8_000.0).contains(&step1_extra));
    // The dashboard held in the journal costs nothing against the record without dashboard (0.25 %).
    let vs_record = (full - full_rec) / full_rec;
    assert!(vs_record.abs() < 0.005, "{vs_record}");
    // 71 ms of the 07:49 step are gone.
    let gone = full_481 - full;
    assert!((72_000.0..75_000.0).contains(&gone), "{gone}");
    assert!((15_250.0..15_350.0).contains(&sps(full)), "{}", sps(full));
    println!(
        "kt c286 pas CALCUL: pasul intreg {:.2} ms = placa {:.2} ms (banda P2zzj 205-215: sub ea, in bine) = {:.0} simb/s; fata de 07:49 -{:.1} ms ({:.0} simb/s); fata de recordul fara tablou {:+.2} % (placa {:.1} / {:.1}); +7 ms doar la pasul 1",
        full / 1e3,
        board / 1e3,
        sps(full),
        gone / 1e3,
        sps(full_481),
        vs_record * 100.0,
        board / 1e3,
        board_rec / 1e3
    );
}

/// The board itself: 1 % shorter than os-481 lot 6; the compute-bound programs follow the clock, the memory-bound
/// ones do not move. `mm_tc_v5` in the step: 11.85 TFLOP/s.
#[test]
fn t286_board_and_programs() {
    let (board, board_481) = (200_952.0f64, 203_061.0f64);
    let (mm, mm_481) = (151_614.0f64, 153_451.0f64);
    let (attn, attn_481) = (15_926.0f64, 16_102.0f64);
    let (vadd, vadd_481) = (2_476.0f64, 2_476.0f64);
    let (colp, colp_481) = (3_383.0f64, 3_390.0f64);
    assert!((-0.012..-0.009).contains(&(board / board_481 - 1.0)));
    // Compute-bound: -1.1 / -1.2 %; the median clock went 1935 -> 1965 MHz (+1.55 %), but its minimum was 1785.
    assert!((-0.013..-0.010).contains(&(mm / mm_481 - 1.0)));
    assert!((-0.013..-0.009).contains(&(attn / attn_481 - 1.0)));
    // Memory-bound: unchanged (memory clock 9501 MHz both times).
    assert_eq!(vadd, vadd_481);
    assert!((colp / colp_481 - 1.0).abs() < 0.003);
    let tflops = 1.797e12 / (mm * 1e-6) / 1e12;
    assert!((11.8..11.9).contains(&tflops), "{tflops}");
    // Matmuls are still 75.4 % of the board: the next gain at lot 6 is again in mm_tc (v6 / v7 in the step).
    assert!((0.75..0.76).contains(&(mm / board)));
    println!(
        "kt c286 placa CALCUL: {:.2} ms ({:+.2} % fata de os-481); mm_tc_v5 {tflops:.2} TFLOP/s, {:.1} % din placa; attn_fwd_tile3 {:+.2} %; vadd / colsum_part neschimbate (memorie)",
        board / 1e3,
        (board / board_481 - 1.0) * 100.0,
        mm / board * 100.0,
        (attn / attn_481 - 1.0) * 100.0
    );
}
