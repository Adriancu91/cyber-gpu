//! C-157 after os-457 — T257, SIMULARE on the processor: the figures the OS wrote from the board for the five small
//! sets of `--atentie-tc` (26 nov 2026, 09:43), recomputed with our model (`attn_tc::model`, the adder fitted in
//! os-335) and the OS's own formula (`rel_ppm` in `gpu_i14`: relative L2 norm against the F32 dQ/dK/dV, in
//! millionths). Predictions R1–R3 written before this code (`docs/C-157_P2_OS457_ATTN_TC.md` §4, commit `19358ab`).
//! The board's P is not the model's P bit for bit (MUFU exp in the forward), so R1/R2 ask for the printed digits.
//! Nothing here runs on the RTX 3070 Ti.

use gpu_compute::attn_tc::{self, Variant};
use gpu_compute::layer::{attn_back_gpu, attn_fwd_tile_gpu};
use gpu_compute::mm_tc::Verdict;
use gpu_compute::mma::GA104_OS335;
use nvidia_ga104::attn_tile_probe::SETS;

/// What os-457 printed for each small set: (name, worst ‰ with `judge` or `None` = out of margin, rel in ppm).
const BOARD: [(&str, Option<u32>, u32); 5] = [
    ("mic-b2-t16-d16", Some(86), 277),
    ("t130-d64", Some(46), 287),
    ("t64-d40", Some(45), 276),
    ("t200-h3", Some(40), 276),
    ("ascutit", None, 302),
];

/// The OS's `rel_ppm` (`gpu_i14`, P6-227), re-written here: sqrt(sum (a-b)^2 / sum b^2) in millionths, truncated.
fn rel_ppm(a: &[f32], b: &[f32]) -> Option<u32> {
    if a.len() != b.len() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return None;
    }
    let (mut num, mut den) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        let d = f64::from(*x) - f64::from(*y);
        num += d * d;
        den += f64::from(*y) * f64::from(*y);
    }
    if den == 0.0 {
        return None;
    }
    let r2 = ((num / den) * 1e12) as u64;
    Some(u32::try_from(r2.isqrt()).unwrap_or(u32::MAX))
}

struct Row {
    name: &'static str,
    old: Verdict,
    new: Verdict,
    rel: u32,
}

fn rows() -> Vec<Row> {
    let mut out = Vec::new();
    for (name, _, _) in BOARD {
        let set = *SETS.iter().find(|s| s.name == name).unwrap();
        let s = set.shape();
        let (nq, no, _) = s.sizes().unwrap();
        let qkv: Vec<f32> = (0..nq).map(|i| set.qkv_at(i)).collect();
        let dout: Vec<f32> = (0..no).map(|i| set.dout_at(i)).collect();
        let (p, _) = attn_fwd_tile_gpu(&s, &qkv, set.scale()).unwrap();
        let (ds, f32dq) = attn_back_gpu(&s, &qkv, &p, &dout, set.scale()).unwrap();
        let got =
            attn_tc::model_with(&s, &qkv, &p, &dout, &ds, GA104_OS335, Variant::PROGRAM).unwrap();
        out.push(Row {
            name,
            old: attn_tc::judge(&s, &qkv, &p, &dout, &ds, &got),
            new: attn_tc::judge_nominal(&s, &qkv, &p, &dout, &ds, &got),
            rel: rel_ppm(&got, &f32dq).unwrap(),
        });
    }
    out
}

#[test]
fn t257_r1_r2_r3_small_sets_model_against_the_board() {
    let rows = rows();
    let (mut rel_same, mut worst_same) = (0, 0);
    for (r, (name, board_worst, board_rel)) in rows.iter().zip(BOARD) {
        assert_eq!(r.name, name);
        println!(
            "{name}: SIMULARE rel {},{:02}e-4 (placa {},{:02}e-4) | vechea {:?} (placa {:?}) | noua {:?}",
            r.rel / 100,
            r.rel % 100,
            board_rel / 100,
            board_rel % 100,
            r.old,
            board_worst,
            r.new
        );
        // R1: the printed digits are rel / 100 and rel % 100 → the same ppm; at most ±2 ppm anywhere
        if r.rel == board_rel {
            rel_same += 1;
        }
        assert!(
            r.rel.abs_diff(board_rel) <= 2,
            "R1 {name}: {} vs {board_rel}",
            r.rel
        );
        // R2
        match (board_worst, r.old) {
            (Some(w), Verdict::InMargine { worst_permille, .. }) => {
                if worst_permille == w {
                    worst_same += 1;
                }
                assert!(
                    worst_permille.abs_diff(w) <= 3,
                    "R2 {name}: {worst_permille} vs {w}"
                );
            }
            (None, Verdict::NuTrece { index, got, .. }) => {
                assert_eq!(index, 394, "R2 {name}");
                assert_eq!(got, 3.0900992e-7, "R2 {name}");
            }
            (b, v) => panic!("R2 {name}: board {b:?}, model {v:?}"),
        }
        // R3
        assert!(
            matches!(r.new, Verdict::InMargine { .. }),
            "R3 {name}: {:?}",
            r.new
        );
    }
    println!("R1: {rel_same} / 5 aceleasi cifre; R2: {worst_same} / 4 acelasi promilaj");
    assert!(rel_same >= 4, "R1: {rel_same} / 5");
    assert!(worst_same >= 3, "R2: {worst_same} / 4");
}
