//! C-157 p.2 (session 264) — T278, SIMULARE: the two new timing sets of the `mm_tc` probe, `qkv-513-dx` (no flag)
//! and `qkv-513-dw` (`TA`), outside `mm_tc_probe::SETS` (which stays as it is), reached through `mm_tc_probe::set`
//! (indices 6 and 7). Prediction L5 written before this test (`docs/C-157_MM_TC_V6_IN_PAS.md` §3). Like `qkv-513`
//! (T171e) they are too large for the CPU stand-in of the board: the banded judgement is checked on the CPU model's
//! output; that `mm_tc_v6` gives `mm_tc`'s bits on these flags is T276e. Nothing here runs on the RTX 3070 Ti.

use gpu_core::diag::Journal;
use nvidia_ga104::mm_tc::{self as tc, TA};
use nvidia_ga104::mm_tc_probe::{
    self as tp, Probe, QKV_DW_SET, QKV_DX_SET, QKV_SET, QKV_T_SETS, SETS, Verdict, parse_mm_tc,
};
use nvidia_ga104::{mm_tc_v5, mm_tc_v6};

const BUF: u64 = 0x4000_0000;

#[test]
fn t278a_indices_and_the_old_sets_unchanged() {
    for (i, s) in SETS.iter().enumerate() {
        assert_eq!(tp::set(i), Some(s));
    }
    assert_eq!(tp::set(QKV_DX_SET), Some(&QKV_T_SETS[0]));
    assert_eq!(tp::set(QKV_DW_SET), Some(&QKV_T_SETS[1]));
    assert_eq!(tp::set(8), None);
    assert_eq!((QKV_DX_SET, QKV_DW_SET), (SETS.len(), SETS.len() + 1));
    // The forward qkv and its two backward products: the same three sizes, the same work.
    let q = SETS[QKV_SET];
    let (dx, dw) = (QKV_T_SETS[0], QKV_T_SETS[1]);
    assert_eq!((dx.m, dx.n, dx.k, dx.flags), (q.m, q.k, q.n, 0));
    assert_eq!((dw.m, dw.n, dw.k, dw.flags), (q.n, q.k, q.m, TA));
    for s in [q, dx, dw] {
        let a = s.args(BUF);
        assert_eq!(a.flop(), 21_743_271_936);
        assert!(mm_tc_v5::fits(&a) && mm_tc_v6::fits(&a), "{}", s.name);
        assert!(s.layout().bytes < 90 << 20, "{}", s.name);
        assert!(
            s.bands
                .unwrap()
                .iter()
                .all(|&r| r % 16 == 0 && r + 16 <= s.m as usize)
        );
    }
    assert_eq!(dx.args(BUF).grid(), [12, 96, 1]);
    assert_eq!(dw.args(BUF).grid(), [12, 36, 1]);
    // launch_bytes takes the new indices (before: refused) and builds mm_tc's QMD for them.
    for i in [QKV_DX_SET, QKV_DW_SET] {
        let (cb, qmd) = tp::launch_bytes(i, 0x10_0000, 0x20_0000, BUF).unwrap();
        assert_eq!(cb, tp::set(i).unwrap().args(BUF).cbuf().unwrap());
        assert_eq!(qmd.len(), nvidia_rm570::QMD_SIZE);
    }
    assert!(tp::launch_bytes(8, 0x10_0000, 0x20_0000, BUF).is_err());
    assert!(tp::judge_set(8, &[], &[]).is_err());
    println!(
        "kt c278 a SIMULARE: seturile 0-5 neschimbate; 6 = {} ({}x{}x{}, fara steag), 7 = {} ({}x{}x{}, TA), cate 21,74 GFLOP, ambele aliniate pentru v5 / v6",
        dx.name, dx.m, dx.n, dx.k, dw.name, dw.m, dw.n, dw.k
    );
}

#[test]
fn t278b_judged_on_their_bands() {
    let ns = [2_100_000, 2_000_000, 1_900_000, 2_050_000, 1_950_000];
    for i in [QKV_DX_SET, QKV_DW_SET] {
        let s = *tp::set(i).unwrap();
        let t = std::time::Instant::now();
        let model = s.model_out();
        eprintln!(
            "T278 {}: modelul pe 3 benzi in {:?} (CPU)",
            s.name,
            t.elapsed()
        );
        let r = tp::judge_set(i, &model, &ns).unwrap();
        assert!(r.passes(), "{} {r:?}", s.name);
        assert_eq!(r.elements, 48 * s.n);
        assert_eq!(r.exact_model, r.elements);
        assert_eq!(r.gflops(), Some(10_871));
        // One band shifted by a row: outside.
        let row = 4 * s.n as usize;
        let mut moved = model.clone();
        moved.copy_within(row..16 * row, 0);
        assert_eq!(
            tp::judge_set(i, &moved, &ns).unwrap().verdict,
            Verdict::NuTrece,
            "{}",
            s.name
        );
        // The guard written: invalid.
        let mut g = model.clone();
        let at = 48 * row;
        g[at..at + 4].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(tp::judge_set(i, &g, &ns).unwrap().verdict, Verdict::Invalid);
        println!(
            "kt c278 b SIMULARE: {} intrari 0x{:08x}, iesire (model) 0x{:08x}, {} promile din margine",
            s.name,
            s.inputs_fnv(),
            tp::fnv(&model),
            r.worst_permille.unwrap()
        );
    }
}

#[test]
fn t278c_rows_and_lines_carry_the_new_sets() {
    let mut p = Probe::default();
    for (i, ns) in [(QKV_DX_SET, 1_700_000u64), (QKV_DW_SET, 2_400_000)] {
        let s = tp::set(i).unwrap();
        p.add(tp::judge_set(i, &s.model_out(), &[ns]).unwrap());
    }
    let mut j = Journal::new(256);
    p.journal(&mut j);
    let text: String = p
        .rows()
        .iter()
        .map(|(w, v)| format!("gpu: valoare {w} = 0x{v:08x}\n"))
        .collect();
    let back = parse_mm_tc(&text);
    assert_eq!(back.per_set.len(), 2);
    for (row, i) in back.per_set.iter().zip([QKV_DX_SET, QKV_DW_SET]) {
        assert_eq!(row[0], i as u32);
        assert_eq!(row[1], tp::set(i).unwrap().inputs_fnv());
        assert_eq!(row[2], 1);
        assert_eq!(row[5], 1);
    }
    assert_eq!(back.per_set[0][7], 1_700_000);
    assert_eq!(back.per_set[1][7], 2_400_000);
    let lines = p.lines("SIMULARE");
    assert!(lines.iter().any(|l| l.contains("qkv-513-dx")));
    assert!(lines.iter().any(|l| l.contains("qkv-513-dw")));
    // The timing products go through the same QMD shape as qkv-513 (mm_tc's block, 24 KiB).
    assert_eq!(tc::BLOCK, mm_tc_v6::BLOCK);
    for l in &lines {
        println!("{l}");
    }
}
