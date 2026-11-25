//! T243 (C-157 p.2, `docs/C-157_P2_OS436.md` §5): os-438 — `mm_tc_v4` next to `mm_tc` / `mm_tc_pf` on the board. The
//! board's rows are copied unchanged into `tests/rapoarte/os438-mm-tc-v4.txt` (MĂSURAT by the OS); this test reads
//! them back and states which of the predictions Q1–Q4 (written before the run, `625ebfe`) held. Nothing runs on (or
//! simulates) the board here.

const OS438: &str = include_str!("rapoarte/os438-mm-tc-v4.txt");

/// `… mm_tc_v4 set <name> (placa): … mm_tc <a> ns, mm_tc_v4 <b> ns = x…` → (name, a, b).
fn rows(prog: &str) -> Vec<(String, u64, u64)> {
    let head = format!("{prog} set ");
    OS438
        .lines()
        .filter_map(|l| {
            let r = l.split(&head).nth(1)?;
            let (name, rest) = r.split_once(" (placa)")?;
            let a = rest
                .split("mm_tc ")
                .nth(1)?
                .split(" ns")
                .next()?
                .parse()
                .ok()?;
            let b = rest
                .split(&format!("{prog} "))
                .last()?
                .split(" ns")
                .next()?
                .parse()
                .ok()?;
            Some((name.to_string(), a, b))
        })
        .collect()
}

fn ns(prog: &str, set: &str) -> u64 {
    rows(prog).into_iter().find(|r| r.0 == set).unwrap().2
}

#[test]
fn t243_os438_against_q1_q4() {
    // Q1 (same bits on every aligned set): HELD — 5 / 5, the unaligned one skipped as designed.
    assert!(OS438.contains(
        "proba mm_tc_v4: TRECE: 5 / 5 seturi cu aceiasi biti ca mm_tc, 1 sarite (nealiniate)"
    ));
    assert_eq!(rows("mm_tc_v4").len(), 5);
    assert_eq!(rows("mm_tc_pf").len(), 6);
    // Q2 (qkv-513 ~5.7–6.4 ms, 3.4–3.8 TFLOP/s, x1.2–1.4 over mm_tc_pf): MISSED, faster — 4.73 ms, 4.59 TFLOP/s, x1.66.
    let (pf, v4) = (ns("mm_tc_pf", "qkv-513"), ns("mm_tc_v4", "qkv-513"));
    assert_eq!((pf, v4), (7_860_256, 4_731_264));
    let flop = 2.0 * 6144.0 * 2304.0 * 768.0;
    let tflops = flop / v4 as f64 / 1e3;
    assert!((4.55..4.65).contains(&tflops));
    let x = pf * 1000 / v4;
    assert!(x > 1_400, "Q2 missed upwards: x{}", x as f64 / 1000.0);
    // Q3 ("under the x1.5 threshold"): MISSED — x1.66, above.
    assert!(x >= 1_500);
    // Q4 (small sets x1.0–1.3 over mm_tc_pf): MISSED upwards on the smallest — x1.31 / x1.38 / x1.45.
    for (set, lo, hi) in [
        ("16x16x16", 1_300, 1_320),
        ("64x64x96-ta", 1_370, 1_390),
        ("32x48x64-limite-f16", 1_440, 1_460),
    ] {
        let r = ns("mm_tc_pf", set) * 1000 / ns("mm_tc_v4", set);
        assert!((lo..=hi).contains(&r), "{set}: x{r}");
    }
    // the only longer small set stays inside the band: 100x72x200 x1.17
    let r = ns("mm_tc_pf", "100x72x200-ta-tb") * 1000 / ns("mm_tc_v4", "100x72x200-ta-tb");
    assert!((1_160..=1_180).contains(&r));
    // per "slice" (DEDUS, mm_tc's 12.2 µs per slice scaled by time): v4 ≈ 2.7 µs, the prediction said 3.3–3.7
    let per_slice_ns = 12_200 * v4 / 21_132_096;
    assert!((2_700..=2_760).contains(&per_slice_ns), "{per_slice_ns}");
}
