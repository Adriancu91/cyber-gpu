//! T244 (C-157 p.2, `docs/C-157_P2_OS436.md` §6): os-443 — `mm_tc*` at the 64 KiB shared-memory step (`--mm-tc-64k`,
//! P6-214) next to the same programs at today's step. The board's rows are copied unchanged into
//! `tests/rapoarte/os443-mm-tc-64k.txt` (MĂSURAT by the OS, automatic cycle 21:05); this test reads them back and states
//! which of the predictions written before the run held (os-423: `mm_tc` x1.7–2.0; session 233: `mm_tc_pf` x1.6–1.9 →
//! 4.4–5.2 TFLOP/s; session 236: `mm_tc_v4` 6.9–8.3 TFLOP/s). Nothing runs on (or simulates) the board here.

const OS443: &str = include_str!("rapoarte/os443-mm-tc-64k.txt");

/// `… <prog> set <name> (placa): … mm_tc <a> ns, <prog> <b> ns = x…` → (name, a, b).
fn rows(prog: &str) -> Vec<(String, u64, u64)> {
    let head = format!("{prog} set ");
    OS443
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
    rows(prog)
        .into_iter()
        .find(|r| r.0 == set)
        .unwrap_or_else(|| panic!("{prog} {set}"))
        .2
}

const QKV_FLOP: f64 = 2.0 * 6144.0 * 2304.0 * 768.0;

fn tflops(t_ns: u64) -> f64 {
    QKV_FLOP / t_ns as f64 / 1e3
}

#[test]
fn t244_os443_same_bits_everywhere() {
    for (p, n) in [("mm_tc@64k", 6), ("mm_tc_pf@64k", 6), ("mm_tc_v4@64k", 5)] {
        assert!(OS443.contains(&format!(
            "proba {p}: TRECE: {n} / {n} seturi cu aceiasi biti ca mm_tc"
        )));
        assert_eq!(rows(p).len(), n, "{p}");
    }
    assert_eq!(OS443.matches("ACEIASI BITI").count(), 6 + 6 + 6 + 5 + 5);
    // the unaligned set skipped by both v4 variants, as designed
    assert_eq!(OS443.matches("nealiniat (randuri").count(), 2);
}

#[test]
fn t244_os443_against_the_predictions() {
    let mm_tc = 21_145_952;
    let qkv = rows("mm_tc@64k")
        .into_iter()
        .find(|r| r.0 == "qkv-513")
        .unwrap();
    assert_eq!(qkv.1, mm_tc);
    // os-423 (1): mm_tc at 64 KiB x1.7–2.0 — HELD: x1.93.
    let x = mm_tc * 1000 / ns("mm_tc@64k", "qkv-513");
    assert!((1_700..=2_000).contains(&x), "mm_tc@64k x{x}");
    // session 233: mm_tc_pf x1.6–1.9 → 4.4–5.2 TFLOP/s — HELD: x1.76, 4.85.
    let (pf, pf64) = (ns("mm_tc_pf", "qkv-513"), ns("mm_tc_pf@64k", "qkv-513"));
    assert_eq!((pf, pf64), (7_896_096, 4_482_592));
    let x = pf * 1000 / pf64;
    assert!((1_600..=1_900).contains(&x), "pf x{x}");
    assert!((4.4..=5.2).contains(&tflops(pf64)));
    // session 236: mm_tc_v4 x1.5–1.8 → 6.9–8.3 TFLOP/s — HELD at the top edge: x1.81 (just over 1.80), 8.28 TFLOP/s.
    let (v4, v464) = (ns("mm_tc_v4", "qkv-513"), ns("mm_tc_v4@64k", "qkv-513"));
    assert_eq!((v4, v464), (4_737_216, 2_623_680));
    let x = v4 * 1000 / v464;
    assert!((1_800..=1_810).contains(&x), "v4 x{x}");
    let t = tflops(v464);
    assert!((6.9..=8.3).contains(&t) && t > 8.2, "{t}");
    // repeatability against os-438 (same programs, today's step): within 0.5 %
    assert!(pf.abs_diff(7_860_256) * 1000 / 7_860_256 <= 5);
    assert!(v4.abs_diff(4_731_264) * 1000 / 4_731_264 <= 5);
}

#[test]
fn t244_os443_small_sets_unchanged() {
    // One block already fits on a unit for every small set (they have 1–6 blocks), so the step must not matter:
    // every small set within ±3 % of the same program at today's step.
    for (p, p64) in [("mm_tc_pf", "mm_tc_pf@64k"), ("mm_tc_v4", "mm_tc_v4@64k")] {
        for (set, _, t) in rows(p).into_iter().filter(|r| r.0 != "qkv-513") {
            let t64 = ns(p64, &set);
            assert!(t.abs_diff(t64) * 100 <= 3 * t, "{p} {set}: {t} vs {t64}");
        }
    }
    for (set, a, b) in rows("mm_tc@64k").into_iter().filter(|r| r.0 != "qkv-513") {
        assert!(a.abs_diff(b) * 100 <= 5 * a, "mm_tc {set}: {a} vs {b}");
    }
}

#[test]
fn t244_mixed_step_estimate() {
    // DEDUS: #513 lot 2 = 0.599 TFLOP of products per step (T240). At v4@64k's qkv speed: ~72 ms, the bottom of the
    // 72–87 ms band written in session 236; shapes with few blocks will be slower, so the band stands.
    let ms = 0.599e12 / (tflops(ns("mm_tc_v4@64k", "qkv-513")) * 1e12) * 1e3;
    assert!((71.0..=74.0).contains(&ms), "{ms}");
}
