//! C-157 — T252, SIMULARE on the processor: the model and the verdict of `attn_back_tile_tc` (dQ, dK, dV on the
//! tensor cores, `docs/C-157_ATTN_BACK_TC.md`; plan and predictions B1–B6 pushed BEFORE this code, `be02aaa`).
//! Nothing here runs on the RTX 3070 Ti: the tensor-core sums are `mma::mma16_hw` with the adder fitted in os-335.

use gpu_compute::attn_tc::{self, SUMS, Sum, Variant, head_products, head_values};
use gpu_compute::layer::{AttnShape, attn_back_gpu, attn_fwd_gpu};
use gpu_compute::mm_tc::{self, Verdict};
use gpu_compute::mma::GA104_OS335;

/// The error scale of the mixed #513 step (already in dO and dS there).
const S: f32 = 16384.0;

struct Case {
    s: AttnShape,
    qkv: Vec<f32>,
    p: Vec<f32>,
    dout: Vec<f32>,
    ds: Vec<f32>,
    /// dqkv of today's F32 program (`attn_back` mode 1 model)
    f32_dqkv: Vec<f32>,
}

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// qkv ~ U(−1.7, 1.7) (sd ≈ 1), P = the board's forward softmax, dO ~ U(−1, 1)·1e-5·S, dS = mode 0 of the board.
fn case(batch: usize, t: usize, c: usize, heads: usize, seed: u64) -> Case {
    let s = AttnShape { batch, t, c, heads };
    let (nq, no, _) = s.sizes().unwrap();
    let scale = 1.0 / (s.d() as f32).sqrt();
    let mut r = lcg(seed);
    let qkv: Vec<f32> = (0..nq).map(|_| 1.7 * r()).collect();
    let dout: Vec<f32> = (0..no).map(|_| 1e-5 * S * r()).collect();
    let (p, _) = attn_fwd_gpu(&s, &qkv, scale).unwrap();
    let (ds, f32_dqkv) = attn_back_gpu(&s, &qkv, &p, &dout, scale).unwrap();
    Case {
        s,
        qkv,
        p,
        dout,
        ds,
        f32_dqkv,
    }
}

fn shapes() -> Vec<Case> {
    vec![
        case(1, 64, 64, 1, 1),   // one tile
        case(1, 100, 64, 1, 2),  // T not a multiple of 64
        case(2, 130, 96, 2, 3),  // B > 1, H > 1, d = 48
        case(1, 192, 40, 1, 4),  // d = 40 (not a multiple of 16)
        case(1, 17, 24, 3, 5),   // T < 64, d = 8
        case(2, 128, 128, 2, 6), // d = 64, two heads, two rows
    ]
}

fn run(k: &Case, v: Variant) -> Option<Vec<f32>> {
    attn_tc::model_with(&k.s, &k.qkv, &k.p, &k.dout, &k.ds, GA104_OS335, v)
}

fn rel(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        num += (*g as f64 - *w as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    (num / den).sqrt()
}

#[test]
fn t252a_b1_same_bits_as_whole_products() {
    for k in shapes() {
        let got = attn_tc::model(&k.s, &k.qkv, &k.p, &k.dout, &k.ds).unwrap();
        let noskip = run(
            &k,
            Variant {
                skip_zero_steps: false,
                ..Variant::PROGRAM
            },
        )
        .unwrap();
        let mut n = 0usize;
        for b in 0..k.s.batch {
            for h in 0..k.s.heads {
                let hp = head_products(&k.s, &k.qkv, &k.p, &k.dout, &k.ds, b, h).unwrap();
                for sum in SUMS {
                    let whole = mm_tc::model(&hp.product(sum)).unwrap();
                    let mine = head_values(&k.s, &got, sum, b, h);
                    let ns = head_values(&k.s, &noskip, sum, b, h);
                    for (i, w) in whole.iter().enumerate() {
                        assert_eq!(
                            mine[i].to_bits(),
                            w.to_bits(),
                            "{:?} {sum:?} b{b} h{h} el {i}",
                            k.s
                        );
                        assert_eq!(ns[i].to_bits(), w.to_bits(), "noskip {:?} {sum:?}", k.s);
                        n += 1;
                    }
                }
            }
        }
        println!(
            "B1 {:?}: {n} elemente, aceiasi biti ca mm_tc::model (si fara pasii sariti)",
            k.s
        );
    }
}

#[test]
fn t252b_b2_b3_in_margin_and_close_to_f32() {
    let mut all = shapes();
    all.push(case(1, 512, 64, 1, 513)); // one #513 head, mixed-step magnitudes
    for k in &all {
        let got = attn_tc::model(&k.s, &k.qkv, &k.p, &k.dout, &k.ds).unwrap();
        let v = attn_tc::judge(&k.s, &k.qkv, &k.p, &k.dout, &k.ds, &got);
        let Verdict::InMargine {
            worst_permille,
            exact,
        } = v
        else {
            panic!("{:?}: {v:?}", k.s);
        };
        assert!(worst_permille < 300, "{:?}: {worst_permille} ‰", k.s);
        let mut errs = [0.0f64; 3];
        for (si, sum) in SUMS.iter().enumerate() {
            let mut g = Vec::new();
            let mut w = Vec::new();
            for b in 0..k.s.batch {
                for h in 0..k.s.heads {
                    g.extend(head_values(&k.s, &got, *sum, b, h));
                    w.extend(head_values(&k.s, &k.f32_dqkv, *sum, b, h));
                }
            }
            errs[si] = rel(&g, &w);
        }
        println!(
            "B2/B3 {:?}: InMargine cel mai rau {worst_permille} ‰, {exact} exacte; fata de F32: dQ {:.2e} dK {:.2e} dV {:.2e}",
            k.s, errs[0], errs[1], errs[2]
        );
        if k.s.t == 512 {
            for e in errs {
                assert!((2e-4..=1e-3).contains(&e), "B3 #513: {e:e}");
            }
        }
    }
}

#[test]
fn t252c_b4_planted_mistakes_caught() {
    let k = case(2, 130, 96, 2, 7);
    let good = attn_tc::model(&k.s, &k.qkv, &k.p, &k.dout, &k.ds).unwrap();
    // NaN above the diagonal of dS and P: the program never reads it, no bit changes.
    let t = k.s.t;
    let mut nan = Case {
        s: k.s,
        qkv: k.qkv.clone(),
        p: k.p.clone(),
        dout: k.dout.clone(),
        ds: k.ds.clone(),
        f32_dqkv: Vec::new(),
    };
    for (x, (pv, dv)) in nan.p.iter_mut().zip(nan.ds.iter_mut()).enumerate() {
        if x % t > (x / t) % t {
            *pv = f32::NAN;
            *dv = f32::NAN;
        }
    }
    let with_nan = attn_tc::model(&nan.s, &nan.qkv, &nan.p, &nan.dout, &nan.ds).unwrap();
    assert!(
        good.iter()
            .zip(&with_nan)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    // The mask forgotten: with NaN above the diagonal the model refuses (the board would make NaN).
    assert!(
        run(
            &nan,
            Variant {
                no_diag_mask: true,
                ..Variant::PROGRAM
            }
        )
        .is_none()
    );
    let wrong = [
        (
            "dK fara transpusa",
            Variant {
                dk_untransposed: true,
                ..Variant::PROGRAM
            },
        ),
        (
            "P in loc de dS la dK",
            Variant {
                dk_uses_p: true,
                ..Variant::PROGRAM
            },
        ),
        (
            "ultimul pas de 16 lipsa",
            Variant {
                drop_last_step: true,
                ..Variant::PROGRAM
            },
        ),
    ];
    for (name, v) in wrong {
        let got = run(&k, v).unwrap();
        let verdict = attn_tc::judge(&k.s, &k.qkv, &k.p, &k.dout, &k.ds, &got);
        assert!(
            matches!(verdict, Verdict::NuTrece { .. }),
            "{name}: {verdict:?}"
        );
        println!("B4 {name}: prins ({verdict:?})");
    }
    // dK alone is touched by the dK mistakes: dQ and dV keep their bits.
    let got = run(
        &k,
        Variant {
            dk_uses_p: true,
            ..Variant::PROGRAM
        },
    )
    .unwrap();
    for sum in [Sum::Dq, Sum::Dv] {
        let a = head_values(&k.s, &good, sum, 1, 1);
        let b = head_values(&k.s, &got, sum, 1, 1);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}
