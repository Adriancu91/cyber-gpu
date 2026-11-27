//! T284 (CALCUL on the board's MEASURED numbers): the `mm_tc` probe with `--mm-tc-v7 --mm-tc-dxdw` (os-486, the
//! automatic cycle of 27 nov 2026 11:09, `ciclu-auto-20261127-1109.txt`, `gpu-jurnal-175.txt`). Nothing here runs on
//! a board; the numbers and the journal's timestamps are copied from the OS report.
//!
//! What this test establishes: `mm_tc_v7` gives the same bits as `mm_tc` on every aligned set (W4). The max-speed
//! request (600 s) was accepted at 114.9 s and the chain ran ~2 100 s: the forward set `qkv-513` is inside the
//! window for all 10 programs, dX only up to v4@64k, dW not at all. So W5 (forward, v7 / v6) is judged; W6, W7, W8
//! are not. And the rerun that P2zzl asked for (fewer programs, same 600 s) would still leave dW outside.

use nvidia_ga104::gsp::perf::BoostPlan;

/// 2 * 6 144 * 2 304 * 768, the same for the three shapes.
const FLOP: f64 = 2.0 * 6144.0 * 2304.0 * 768.0;

const NAMES: [&str; 10] = [
    "mm_tc",
    "mm_tc@64k",
    "v4",
    "v4@64k",
    "v5",
    "v5@64k",
    "v6",
    "v6@64k",
    "v7",
    "v7@64k",
];

/// Median ns of 5 launches, as printed by the OS (MEASURED), in `NAMES` order.
const QKV: [f64; 10] = [
    21_153_792.0,
    10_927_936.0,
    4_737_856.0,
    2_621_952.0,
    4_722_144.0,
    1_870_048.0,
    3_635_552.0,
    1_257_664.0,
    3_404_000.0,
    1_196_480.0,
];
const DX: [f64; 10] = [
    21_009_664.0,
    10_844_032.0,
    4_312_032.0,
    2_530_848.0,
    42_118_592.0,
    15_585_056.0,
    38_652_192.0,
    12_934_208.0,
    36_129_824.0,
    11_450_752.0,
];
const DW: [f64; 10] = [
    21_265_088.0,
    105_637_120.0,
    46_836_128.0,
    30_459_872.0,
    46_392_320.0,
    24_164_960.0,
    39_344_736.0,
    17_984_832.0,
    36_365_664.0,
    14_101_920.0,
];

/// Journal seconds: the accepted request (`ks cerere viteza maxima = 0`), the clear at the end (P8 before it).
const REQUEST_S: f64 = 114.907;
const END_S: f64 = 2_102.954;

/// First and last doorbell (`w 0xbb0090`) of each program's group of 13 (4 fills + 5 launches + 4 reads), journal
/// seconds; `mm_tc` and `mm_tc@64k` sit in one group with the set's uploads (gaps < 2 s), given as one span.
const QKV_AT: [(f64, f64); 9] = [
    (163.5, 205.6),
    (239.2, 243.8),
    (277.4, 282.4),
    (316.3, 320.9),
    (354.6, 359.5),
    (393.1, 398.5),
    (432.5, 437.4),
    (471.0, 475.7),
    (509.3, 513.9),
];
const DX_AT: [(f64, f64); 9] = [
    (567.6, 585.1),
    (625.6, 631.0),
    (671.6, 676.5),
    (717.6, 722.2),
    (763.0, 767.7),
    (808.2, 814.0),
    (854.5, 859.4),
    (899.9, 904.5),
    (945.4, 950.3),
];
const DW_AT: [(f64, f64); 9] = [
    (1_013.4, 1_032.7),
    (1_141.5, 1_146.1),
    (1_254.0, 1_258.6),
    (1_366.9, 1_372.3),
    (1_480.2, 1_485.1),
    (1_592.9, 1_597.6),
    (1_705.0, 1_709.7),
    (1_817.8, 1_822.4),
    (1_930.3, 1_936.0),
];

const _: () = assert!(END_S - REQUEST_S > 600.0);

fn plan(duration_s: u32) -> BoostPlan {
    BoostPlan {
        enabled: true,
        duration_s,
        ..BoostPlan::default()
    }
}

fn covered(duration_s: u32, at: (f64, f64)) -> bool {
    let since = ((at.0 - REQUEST_S) * 1000.0) as u64;
    let run = ((at.1 - at.0) * 1000.0) as u64;
    plan(duration_s).covers_ms(since, run)
}

fn tflops(ns: f64) -> f64 {
    FLOP / (ns * 1e-9) / 1e12
}

/// The journal's hex values agree with the report; qkv-513 repeats os-485 / os-483 within 0.5 %.
#[test]
fn t284_numbers_and_repeat() {
    // gpu-jurnal-175.txt: `kt mm_tc qkv timp median (ns) = 0x0142c800`, `kt mm_tc_v7 qkv median ns = 0x0033f0e0`.
    assert_eq!(0x0142_c800 as f64, QKV[0]);
    assert_eq!(0x0033_f0e0 as f64, QKV[8]);
    // After the request: core 0x6db = 1 755 MHz, P0; at the end 0xd2 = 210 MHz, P8; board 34.7-36.5 C.
    assert_eq!((0x6db, 0xd2), (1755, 210));
    // os-485 (same programs, same sets, inside its window): mm_tc 21.143, v5@64k 1.870, v6@64k 1.260 ms;
    // dX v4@64k 2.523, v4 4.308 ms.
    for (now, then) in [
        (QKV[0], 21_142_976.0),
        (QKV[5], 1_869_760.0),
        (QKV[7], 1_260_064.0),
        (DX[3], 2_522_816.0),
        (DX[2], 4_307_680.0),
    ] {
        assert!((now / then - 1.0).abs() < 0.005, "{now} {then}");
    }
}

/// Inside the 600 s window: all of qkv-513; dX up to v4@64k (v5 starts 2.7 s after the end); nothing of dW.
#[test]
fn t284_which_timings_are_at_full_speed() {
    let expiry = REQUEST_S + 600.0;
    assert!((714.8..715.0).contains(&expiry));
    assert!(QKV_AT.iter().all(|&a| covered(600, a)));
    let dx: Vec<bool> = DX_AT.iter().map(|&a| covered(600, a)).collect();
    assert_eq!(
        dx,
        [true, true, true, false, false, false, false, false, false]
    );
    assert!(DW_AT.iter().all(|&a| !covered(600, a)));
    // The jump sits exactly at the cut: v4@64k dX is at speed (2.531 ms, os-485 2.523), v5 2.7 s after the cut is
    // x9.9 its os-485 time inside the window (4.277 ms).
    assert!(DX_AT[3].0 - expiry < 3.0);
    const { assert!(DX[4] / 4_276_576.0 > 9.0) };
    // Even a 1 800 s request would have left the last program (v7@64k dW, 1 930 s) out.
    assert!(!covered(1800, DW_AT[8]));
    assert!(covered(2400, DW_AT[8]));
    println!(
        "kt c284 CALCUL: cererea la {REQUEST_S} s, expirata la {expiry:.1} s; qkv: toate 10 in fereastra; dX: pana la v4@64k; dW: niciunul"
    );
}

/// W4 + W5: same bits (from the report), forward v7 / v6 inside the window.
#[test]
fn t284_w5_forward_v7_over_v6() {
    let w5_64k = QKV[7] / QKV[9];
    let w5_16k = QKV[6] / QKV[8];
    assert!((1.00..1.20).contains(&w5_64k), "{w5_64k}");
    assert!((1.04..1.06).contains(&w5_64k));
    assert!((1.06..1.08).contains(&w5_16k));
    assert!((tflops(QKV[9]) - 18.17).abs() < 0.02);
    // The T282 line (slice = 953 + 0.82 * shared-memory path, fitted on v5 / v6 TB) would give v7 (TB path 256,
    // T279) x1.23 over v6 (576). The board: x1.05. Below 576 the path stops being the brake: the slice that is
    // left (DEDUS, in the line's units) ~1 356 against the 1 024 floor of the circuits.
    let slice_v6 = 2_108.0 * 1_263_456.0 / 1_868_544.0;
    let b = (2_108.0 - slice_v6) / (1_408.0 - 576.0);
    let a = 2_108.0 - b * 1_408.0;
    let t = |path: f64| (a + b * path).max(1_024.0);
    let line = t(576.0) / t(256.0);
    assert!((1.20..1.25).contains(&line), "{line}");
    let slice_v7 = slice_v6 / w5_64k;
    assert!((1_340.0..1_370.0).contains(&slice_v7), "{slice_v7}");
    // What a cut of 320 passes bought: ~69 units, a slope of ~0.22 instead of 0.82.
    let slope = (slice_v6 - slice_v7) / (576.0 - 256.0);
    assert!((0.18..0.26).contains(&slope), "{slope}");
    println!(
        "kt c284 CALCUL: W5 v7/v6 qkv-513 @64k x{w5_64k:.3} (banda 1,00-1,20), fara 64k x{w5_16k:.3}; v7@64k {:.3} ms = {:.2} TFLOP/s; dreapta T282 x{line:.2}; panta sub 576: {slope:.2}",
        QKV[9] / 1e6,
        tflops(QKV[9])
    );
}

/// The rerun asked in P2zzl (session 269: `--mm-tc --mm-tc-v5 --mm-tc-v6 --mm-tc-64k --mm-tc-dxdw`) has 4 companions
/// after `mm_tc` / `mm_tc@64k` on each set. With today's spacing (judging each program against the model on the
/// CPU: qkv 38.5 s, dX 45.7 s, dW 112.6 s per companion) it would still put every dW companion after 714.9 s.
#[test]
fn t284_the_rerun_does_not_fit_600_s() {
    let step = |at: &[(f64, f64); 9]| (at[8].0 - at[1].0) / 7.0;
    let (sq, sx, sw) = (step(&QKV_AT), step(&DX_AT), step(&DW_AT));
    assert!(
        (38.0..39.0).contains(&sq) && (45.0..46.5).contains(&sx) && (112.0..113.0).contains(&sw)
    );
    // Today's offsets: set start -> first companion, last companion -> next set's uploads.
    let (q_first, x_first, w_first) = (
        QKV_AT[1].0 - QKV_AT[0].0,
        DX_AT[1].0 - DX_AT[0].0,
        DW_AT[1].0 - DW_AT[0].0,
    );
    let (q_to_x, x_to_w) = (DX_AT[0].0 - QKV_AT[8].0, DW_AT[0].0 - DX_AT[8].0);
    let companions = 4.0;
    let q_last = QKV_AT[0].0 + q_first + (companions - 1.0) * sq;
    let x_start = q_last + q_to_x;
    let x_last = x_start + x_first + (companions - 1.0) * sx;
    let w_start = x_last + x_to_w;
    let w_first_at = w_start + w_first;
    let w_last = w_first_at + (companions - 1.0) * sw + 6.0;
    let expiry = REQUEST_S + 600.0;
    assert!(x_last + 6.0 < expiry, "dX fits: {x_last}");
    assert!(w_first_at > expiry, "dW does not: {w_first_at}");
    // With 1 800 s it fits with > 10 min to spare; 3 600 s is NVIDIA's limit (already checked by encode_boost).
    assert!(w_last - REQUEST_S < 1_800.0 - 600.0, "{w_last}");
    println!(
        "kt c284 DEDUS: reluarea P2zzl: dX gata la ~{:.0} s, primul dW la ~{w_first_at:.0} s > {expiry:.0} s; cu 1 800 s gata la ~{w_last:.0} s",
        x_last + 6.0
    );
}

/// Outside the window the numbers say nothing about speed: v6 / v7 on dX and dW were both measured slowed down.
/// Their ratios are written down (MEASURED) but not judged against W6 / W7.
#[test]
fn t284_outside_the_window_not_judged() {
    let dx = DX[7] / DX[9];
    let dw = DW[7] / DW[9];
    // Even slowed, v7 is not slower than v6 on either set (the same bits, fewer bank conflicts).
    assert!(dx > 1.0 && dw > 1.0);
    // v5@64k dX: 15.585 ms now against 1.620 ms in os-485's window: x9.6, the size of the P0 / P8 core clocks.
    assert!((9.0..10.5).contains(&(DX[5] / 1_619_840.0)));
    println!(
        "kt c284 NEJUDECAT: dX v6/v7 @64k x{dx:.2}, dW x{dw:.2} (placa incetinita, in afara ferestrei)"
    );
    let _ = NAMES;
}
