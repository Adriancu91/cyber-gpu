//! C-157 — T256, SIMULARE on the processor: the `ascutit` set of the attention probe leaves the `mm_tc` margin with our
//! own model (`attn_tc::model`), as the OS found in P6-227. Plan, cause and predictions S1–S7 written before this code
//! (`docs/C-157_ATTN_TC_SUBNORMALE.md`, commit `e874196`). The new margin (`mm_tc::margin_nominal`) is written AFTER the
//! miss: it is a correction of the margin, not a prediction, and is reported beside the old one.
//! Nothing here runs on the RTX 3070 Ti: the tensor-core sums are `mma::mma16_hw` with the adder fitted in os-335.

use gpu_compute::attn_tc::{self, SUMS, Variant, head_products, head_values};
use gpu_compute::layer::{attn_back_gpu, attn_fwd_tile_gpu};
use gpu_compute::mm_tc::{self, Product, Verdict};
use gpu_compute::mma::GA104_OS335;
use nvidia_ga104::attn_tile_probe::{ProbeSet, SETS};

struct Case {
    set: ProbeSet,
    qkv: Vec<f32>,
    p: Vec<f32>,
    dout: Vec<f32>,
    ds: Vec<f32>,
}

/// The inputs the probe writes, P of the tiled forward and dS of mode 0 (the CPU models the OS's simulator runs);
/// `dout_scale` multiplies dO (1 = the probe as it is).
fn case(set: &ProbeSet, dout_scale: f32) -> Case {
    let s = set.shape();
    let (nq, no, _) = s.sizes().unwrap();
    let qkv: Vec<f32> = (0..nq).map(|i| set.qkv_at(i)).collect();
    let dout: Vec<f32> = (0..no).map(|i| set.dout_at(i) * dout_scale).collect();
    let (p, _) = attn_fwd_tile_gpu(&s, &qkv, set.scale()).unwrap();
    let (ds, _) = attn_back_gpu(&s, &qkv, &p, &dout, set.scale()).unwrap();
    Case {
        set: *set,
        qkv,
        p,
        dout,
        ds,
    }
}

fn small_sets() -> Vec<ProbeSet> {
    SETS.iter().filter(|s| s.bands.is_none()).copied().collect()
}

fn ascutit() -> ProbeSet {
    *SETS.iter().find(|s| s.name == "ascutit").unwrap()
}

fn model(k: &Case, v: Variant) -> Option<Vec<f32>> {
    let s = k.set.shape();
    attn_tc::model_with(&s, &k.qkv, &k.p, &k.dout, &k.ds, GA104_OS335, v)
}

fn judge_old(k: &Case, got: &[f32]) -> Verdict {
    attn_tc::judge(&k.set.shape(), &k.qkv, &k.p, &k.dout, &k.ds, got)
}

fn judge_new(k: &Case, got: &[f32]) -> Verdict {
    attn_tc::judge_nominal(&k.set.shape(), &k.qkv, &k.p, &k.dout, &k.ds, got)
}

#[test]
fn t256a_s1_s2_reproduces_the_os_miss_and_its_cause() {
    let k = case(&ascutit(), 1.0);
    let got = model(&k, Variant::PROGRAM).unwrap();
    let v = judge_old(&k, &got);
    println!("S1 ascutit, marginea veche: {v:?}");
    let Verdict::NuTrece {
        index,
        got: g,
        reference,
        margin,
    } = v
    else {
        panic!("S1: expected NuTrece, got {v:?}");
    };
    // the OS's figures (P6-227): element 394 of the head, 3.0900992e-7 vs 3.0901583e-7, margin 3.6e-12
    assert_eq!(index, 394);
    assert_eq!(g, 3.0900992e-7);
    assert!((reference - 3.0901583e-7).abs() < 1e-13, "{reference}");
    assert!((margin - 3.6e-12).abs() < 0.1e-12, "{margin}");
    // S2: which sum, and whether its row / column reads a non-zero f16 subnormal
    let s = k.set.shape();
    let hp = head_products(&s, &k.qkv, &k.p, &k.dout, &k.ds, 0, 0).unwrap();
    let mut found = false;
    for sum in SUMS {
        let pr = hp.product(sum);
        if let Verdict::NuTrece { index: i, .. } =
            mm_tc::judge(&pr, &head_values(&s, &got, sum, 0, 0))
            && i == index
        {
            let (r, c) = (i / pr.n, i % pr.n);
            let sub = mm_tc::reads_subnormal(&pr, r, c);
            println!(
                "S2 {sum:?}: element ({r}, {c}) citeste o valoare f16 subnormala: {sub}; marginea noua {:.3e}",
                mm_tc::margin_nominal(&pr, r, c)
            );
            assert!(sub, "S2");
            found = true;
            break;
        }
    }
    assert!(found, "S2: the failing sum not found");
}

#[test]
fn t256b_s3_s4_new_margin_on_the_probe_sets() {
    for set in small_sets() {
        let k = case(&set, 1.0);
        let got = model(&k, Variant::PROGRAM).unwrap();
        let (old, new) = (judge_old(&k, &got), judge_new(&k, &got));
        println!("S3/S4 {}: vechea {old:?} | noua {new:?}", set.name);
        match new {
            Verdict::InMargine { worst_permille, .. } => assert!(worst_permille <= 1000),
            bad => panic!("S3 {}: {bad:?}", set.name),
        }
        if set.name != "ascutit" {
            assert!(matches!(old, Verdict::InMargine { .. }), "{}", set.name);
        }
    }
}

#[test]
fn t256c_s5_planted_mistakes_still_caught() {
    let k = case(&ascutit(), 1.0);
    let mistakes = [
        (
            "dK netranspus",
            Variant {
                dk_untransposed: true,
                ..Variant::PROGRAM
            },
        ),
        (
            "P in loc de dS",
            Variant {
                dk_uses_p: true,
                ..Variant::PROGRAM
            },
        ),
        (
            "ultimul pas uitat",
            Variant {
                drop_last_step: true,
                ..Variant::PROGRAM
            },
        ),
    ];
    for (name, v) in mistakes {
        let got = model(&k, v).unwrap();
        let new = judge_new(&k, &got);
        println!("S5 {name}: {new:?}");
        assert!(
            matches!(new, Verdict::NuTrece { .. } | Verdict::Invalid),
            "S5 {name}: not caught ({new:?})"
        );
    }
    // "no mask": with zeros above the diagonal (what the probe's buffers hold) it computes the same numbers, so it is
    // only visible with junk there — 0.5 above the diagonal of P and dS, the judgement still on the masked products.
    let none = Variant {
        no_diag_mask: true,
        ..Variant::PROGRAM
    };
    let same = model(&k, none).unwrap();
    println!(
        "S5 fara masca, zerouri deasupra diagonalei: {:?}",
        judge_new(&k, &same)
    );
    let t = k.set.t as usize;
    let mut junk = Case {
        set: k.set,
        qkv: k.qkv.clone(),
        p: k.p.clone(),
        dout: k.dout.clone(),
        ds: k.ds.clone(),
    };
    for x in 0..junk.p.len() {
        if x % t > (x / t) % t {
            junk.p[x] = 0.5;
            junk.ds[x] = 0.5;
        }
    }
    let got = model(&junk, none).unwrap();
    let new = judge_new(&junk, &got);
    println!("S5 fara masca, 0,5 deasupra diagonalei: {new:?}");
    assert!(
        matches!(new, Verdict::NuTrece { .. }),
        "S5 no mask: {new:?}"
    );
    let right = model(&junk, Variant::PROGRAM).unwrap();
    assert!(matches!(
        judge_new(&junk, &right),
        Verdict::InMargine { .. }
    ));
}

#[test]
fn t256d_s6_scale_in_the_probe() {
    // Predicted "not enough" (P itself subnormal in dV); it turned out enough: with dO x 2^14 the old margin holds.
    let k = case(&ascutit(), 16384.0);
    let got = model(&k, Variant::PROGRAM).unwrap();
    let (old, new) = (judge_old(&k, &got), judge_new(&k, &got));
    println!("S6 ascutit cu dO x 2^14: vechea {old:?} | noua {new:?}");
    assert!(matches!(old, Verdict::InMargine { .. }), "S6 old: {old:?}");
    assert!(matches!(new, Verdict::InMargine { .. }), "S6 new: {new:?}");
}

#[test]
fn t256e_s7_same_margin_without_subnormals() {
    // normal operands only: 1 + i/64 style values, signs mixed
    let (m, n, kk) = (32usize, 24usize, 40usize);
    let a: Vec<f32> = (0..m * kk)
        .map(|i| (1.0 + (i % 97) as f32 / 64.0) * if i % 3 == 0 { -1.0 } else { 0.5 })
        .collect();
    let b: Vec<f32> = (0..kk * n)
        .map(|i| (0.25 + (i % 53) as f32 / 128.0) * if i % 5 == 0 { -2.0 } else { 1.0 })
        .collect();
    let p = Product {
        a: &a,
        b: &b,
        bias: None,
        m,
        n,
        k: kk,
        flags: 0,
    };
    for r in 0..m {
        for c in 0..n {
            assert!(!mm_tc::reads_subnormal(&p, r, c));
            assert_eq!(
                mm_tc::margin(&p, r, c).to_bits(),
                mm_tc::margin_nominal(&p, r, c).to_bits(),
                "({r}, {c})"
            );
        }
    }
    let got = mm_tc::model(&p).unwrap();
    assert_eq!(mm_tc::judge(&p, &got), mm_tc::judge_nominal(&p, &got));
    // one subnormal operand: the nominal margin is wider, never narrower
    let mut a2 = a.clone();
    a2[0] = 1.0e-6;
    let p2 = Product { a: &a2, ..p };
    assert!(mm_tc::reads_subnormal(&p2, 0, 0));
    assert!(mm_tc::margin_nominal(&p2, 0, 0) > mm_tc::margin(&p2, 0, 0));
    println!(
        "S7: {} elemente, aceeasi margine bit cu bit fara subnormale",
        m * n
    );
}

#[test]
fn t256f_dp_of_mode0_on_the_probe_sets() {
    // `attn_back_tile0_tc` (session 245, not yet handed over): dP = dO·Vᵀ reads only qkv and dO (normal values here),
    // so both margins should agree on every small set.
    use gpu_compute::attn_tc::{Back0Variant, back0_dp_with};
    for set in small_sets() {
        let k = case(&set, 1.0);
        let s = set.shape();
        let dp = back0_dp_with(&s, &k.qkv, &k.dout, GA104_OS335, Back0Variant::PROGRAM).unwrap();
        let old = attn_tc::judge_dp(&s, &k.qkv, &k.dout, &dp);
        let new = attn_tc::judge_dp_nominal(&s, &k.qkv, &k.dout, &dp);
        println!("dP {}: vechea {old:?} | noua {new:?}", set.name);
        assert!(matches!(old, Verdict::InMargine { .. }), "{}", set.name);
        assert_eq!(old, new, "{}", set.name);
    }
}
