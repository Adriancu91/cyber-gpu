//! T224 (C-157 point 2, `docs/C-157_P2_COMPARATII.md` §„os-414 + os-415”): the step-profile reader `parse_profiles`,
//! on the processor, on the **real** profile rows of two board runs (copied from the OS's automatic reports in
//! `Cyber-hub/rezultate/os-automat/2026-11-25/`, `ciclu-auto-20261125-0018.txt` = os-414 and `-0058.txt` = os-415).
//! Nothing here runs on (or simulates) the board: it is the board's own text in, rows and differences out.

use nvidia_ga104::train_profile::{ParsedProfile, parse_profiles};

const OS414: &str = include_str!("rapoarte/os414-profil.txt");
const OS415: &str = include_str!("rapoarte/os415-profil.txt");

const COLSUM: &[&str] = &["colsum", "colsum_part"];
const MATMUL: &[&str] = &["mm_train_tile", "mm_train_tile_gelu"];
const SMALL_FUSED: &[&str] = &["gelu_tr", "vadd", "ln_back", "ln_back_res"];

fn k513(text: &str) -> ParsedProfile {
    let all = parse_profiles(text);
    assert_eq!(all.len(), 2, "mini + #513");
    assert!(all[0].context.contains("antrenare mini"));
    assert!(all[1].context.contains("antrenare #513"));
    all[1].clone()
}

#[test]
fn t224_p1_two_profiles_and_headers() {
    let a = k513(OS414);
    let b = k513(OS415);
    assert_eq!((a.steps, a.total_us, a.launches), (10, 463_594, 483));
    assert_eq!((b.steps, b.total_us, b.launches), (10, 485_577, 349));
    // The mini step (same list both nights) is the same within noise.
    let ma = &parse_profiles(OS414)[0];
    let mb = &parse_profiles(OS415)[0];
    assert_eq!(ma.launches, mb.launches);
    assert!(ma.total_us.abs_diff(mb.total_us) < 50);
}

#[test]
fn t224_p2_rows_add_up() {
    for p in [k513(OS414), k513(OS415)] {
        let rows = p.programs.len() as u64;
        let sum = p.programs_us();
        assert!(
            sum <= p.total_us && p.total_us - sum <= rows,
            "{} vs {}",
            sum,
            p.total_us
        );
        let launches: u32 = p.programs.iter().map(|g| g.launches).sum();
        assert_eq!(launches, p.launches);
        let stages: u32 = p.stages.iter().map(|g| g.launches).sum();
        assert_eq!(stages, p.launches);
        assert_eq!(p.tags.len(), 8);
    }
}

#[test]
fn t224_p3_fusing_is_a_wash_without_colsum() {
    let a = k513(OS414);
    let b = k513(OS415);
    // os-415 ran without the split colsum: that alone is the whole +22 ms.
    assert_eq!(a.programs_sum(COLSUM), 848 + 1_540);
    assert_eq!(b.programs_sum(COLSUM), 24_332);
    let d = b.total_without(COLSUM) as i64 - a.total_without(COLSUM) as i64;
    assert_eq!(d, 39);
    assert!(
        d.unsigned_abs() * 10_000 < a.total_us,
        "under 0.1 per mille of the step"
    );
    // With the split colsum the fused list would land where os-414 did (DEDUS): ~463.6 ms.
    assert_eq!(b.total_without(COLSUM) + a.programs_sum(COLSUM), 463_633);
}

#[test]
fn t224_p4_matmul_up_small_programs_down() {
    let a = k513(OS414);
    let b = k513(OS415);
    let mm = b.programs_sum(MATMUL) as i64 - a.programs_sum(MATMUL) as i64;
    let small = b.programs_sum(SMALL_FUSED) as i64 - a.programs_sum(SMALL_FUSED) as i64;
    assert_eq!(mm, 1_190);
    assert_eq!(small, -1_160);
    // The plain tile program did not change (135 launches): the whole matmul rise is the 12 fused launches.
    let d = a.diff_programs(&b);
    let tile = d.iter().find(|x| x.name == "mm_train_tile").unwrap();
    assert_eq!((tile.before_launches, tile.after_launches), (147, 135));
    let gelu = d.iter().find(|x| x.name == "mm_train_tile_gelu").unwrap();
    assert_eq!((gelu.before_us, gelu.after_launches), (0, 12));
    // 36 launches fewer: gelu_tr 24 -> 12, vadd 48 -> 24 (the ln_back rows only change name).
    let launches = |p: &ParsedProfile| p.programs.iter().map(|g| g.launches).sum::<u32>();
    assert_eq!(launches(&a) - launches(&b), 483 - 349);
}

#[test]
fn t224_p5_proj_dx_tag() {
    let a = k513(OS414);
    let b = k513(OS415);
    let before = a.tag("inapoi:proj-dX").unwrap();
    let after = b.tag("inapoi:proj-dX+gelu").unwrap();
    assert_eq!((before.launches, after.launches), (12, 12));
    let d = after.us as i64 - before.us as i64;
    assert_eq!(d, 1_256);
    assert_eq!(d / 12, 104); // ~105 us per launch
    assert_eq!(d * 1000 / before.us as i64, 29); // +3.0 %
}

#[test]
fn t224_refuses_broken_text() {
    assert!(parse_profiles("").is_empty());
    // A header that does not parse gives no profile; rows without a header are ignored.
    let bad = "x: profilul pasului (placa, median pe zece pasi cititi intregi): 1 us, 1 lansari\n\
               x:   mm 5 us ( 1.0 %, 1 lansari)\n";
    assert!(parse_profiles(bad).is_empty());
    // A broken row ends the profile; nothing after it is taken.
    let cut = "x: profilul pasului (placa, median pe 3 pasi cititi intregi): 10 us, 2 lansari\n\
               x:  pe programe:\n\
               x:   mm 6 us ( 60.0 %, 1 lansari)\n\
               x:   gelu ?? us\n\
               x:   ln 4 us ( 40.0 %, 1 lansari)\n";
    let p = parse_profiles(cut);
    assert_eq!(p.len(), 1);
    assert_eq!(p[0].programs.len(), 1);
    assert_eq!(p[0].total_without(&["mm"]), 4);
}
