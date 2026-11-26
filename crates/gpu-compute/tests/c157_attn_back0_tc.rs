//! C-157 — T254, SIMULARE on the processor: the model and the verdict of `attn_back_tile0_tc` (mode 0 of the
//! attention backward with dP = dO·Vᵀ on the tensor cores, dot and dS in F32; `docs/C-157_ATTN_BACK0_TC.md`, plan and
//! predictions E1–E7 pushed BEFORE this code, `57b4079`). Nothing here runs on the RTX 3070 Ti: the tensor-core sums are
//! `mma::mma16_hw` with the adder fitted in os-335.

use gpu_compute::attn_tc::{
    self, Back0Variant, SUMS, Sum, back0_dp_with, back0_tail, head_block, head_dp, head_values,
};
use gpu_compute::layer::{AttnShape, attn_back_gpu, attn_fwd_gpu};
use gpu_compute::math::fma_f32_ftz;
use gpu_compute::mm_tc::{self, Verdict};
use gpu_compute::mma::GA104_OS335;

/// The error scale of the mixed #513 step (already in dO there).
const S: f32 = 16384.0;

struct Case {
    s: AttnShape,
    scale: f32,
    qkv: Vec<f32>,
    p: Vec<f32>,
    dout: Vec<f32>,
    /// dS and dqkv of today's F32 programs
    ds: Vec<f32>,
    dqkv: Vec<f32>,
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

/// As T252: qkv ~ U(−1.7, 1.7), P = the board's forward softmax, dO ~ U(−1, 1)·1e-5·`dscale`.
fn case_scaled(batch: usize, t: usize, c: usize, heads: usize, seed: u64, dscale: f32) -> Case {
    let s = AttnShape { batch, t, c, heads };
    let (nq, no, _) = s.sizes().unwrap();
    let scale = 1.0 / (s.d() as f32).sqrt();
    let mut r = lcg(seed);
    let qkv: Vec<f32> = (0..nq).map(|_| 1.7 * r()).collect();
    let dout: Vec<f32> = (0..no).map(|_| 1e-5 * dscale * r()).collect();
    let (p, _) = attn_fwd_gpu(&s, &qkv, scale).unwrap();
    let (ds, dqkv) = attn_back_gpu(&s, &qkv, &p, &dout, scale).unwrap();
    Case {
        s,
        scale,
        qkv,
        p,
        dout,
        ds,
        dqkv,
    }
}

fn case(batch: usize, t: usize, c: usize, heads: usize, seed: u64) -> Case {
    case_scaled(batch, t, c, heads, seed, S)
}

fn shapes() -> Vec<Case> {
    vec![
        case(1, 64, 64, 1, 11),   // one tile
        case(1, 100, 64, 1, 12),  // T not a multiple of 64
        case(2, 130, 96, 2, 13),  // B > 1, H > 1, d = 48
        case(1, 192, 40, 1, 14),  // d = 40 (not a multiple of 16)
        case(1, 17, 24, 3, 15),   // T < 64, d = 8
        case(2, 128, 128, 2, 16), // d = 64, two heads, two rows
    ]
}

fn rel(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for (g, w) in got.iter().zip(want) {
        num += (*g as f64 - *w as f64).powi(2);
        den += (*w as f64).powi(2);
    }
    (num / den).sqrt()
}

/// Today's F32 dP (the `ffma.ftz` chain over e of `attn_back_row_gpu`), on and below the diagonal; 0 above.
fn dp_f32(k: &Case) -> Vec<f32> {
    let (t, c, d) = (k.s.t, k.s.c, k.s.d());
    let mut dp = vec![0.0f32; k.p.len()];
    for b in 0..k.s.batch {
        for h in 0..k.s.heads {
            for i in 0..t {
                for j in 0..=i {
                    let mut acc = 0.0f32;
                    for e in 0..d {
                        acc = fma_f32_ftz(
                            k.dout[(b * t + i) * c + h * d + e],
                            k.qkv[(b * t + j) * 3 * c + 2 * c + h * d + e],
                            acc,
                        );
                    }
                    dp[((b * k.s.heads + h) * t + i) * t + j] = acc;
                }
            }
        }
    }
    dp
}

/// Only the elements on and below the diagonal of a `[B, H, T, T]` buffer.
fn lower(s: &AttnShape, m: &[f32]) -> Vec<f32> {
    let t = s.t;
    m.iter()
        .enumerate()
        .filter(|(x, _)| x % t <= (x / t) % t)
        .map(|(_, v)| *v)
        .collect()
}

#[test]
fn t254a_e1_same_bits_where_they_must_be() {
    for k in shapes() {
        // (a) dP in the program's order = the whole mm_tc product of the head, bit for bit.
        let dp = back0_dp_with(&k.s, &k.qkv, &k.dout, GA104_OS335, Back0Variant::PROGRAM).unwrap();
        let mut n = 0usize;
        for b in 0..k.s.batch {
            for h in 0..k.s.heads {
                let hd = head_dp(&k.s, &k.qkv, &k.dout, b, h).unwrap();
                let whole = mm_tc::model(&hd.product()).unwrap();
                let mine = head_block(&k.s, &dp, b, h);
                let t = k.s.t;
                for (i, w) in whole.iter().enumerate() {
                    // only the tiles the program computes (on and below the diagonal)
                    if (i % t) / 64 > (i / t) / 64 {
                        assert_eq!(mine[i], 0.0);
                        continue;
                    }
                    assert_eq!(mine[i].to_bits(), w.to_bits(), "{:?} b{b} h{h} el {i}", k.s);
                    n += 1;
                }
            }
        }
        // (b) the F32 tail fed with today's F32 dP = today's dS, bit for bit.
        let tail = back0_tail(&k.s, &k.p, &dp_f32(&k), k.scale, Back0Variant::PROGRAM);
        assert!(
            tail.iter()
                .zip(&k.ds)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "E1b {:?}",
            k.s
        );
        println!(
            "E1 {:?}: dP {n} elemente = mm_tc::model bit cu bit; coada F32 cu dP-ul de azi = dS-ul de azi ({} elemente)",
            k.s,
            k.ds.len()
        );
    }
}

#[test]
fn t254b_e2_e3_e4_margin_and_distance_to_f32() {
    let mut all = shapes();
    all.push(case(1, 512, 64, 1, 513)); // one #513 head, mixed-step magnitudes
    for k in &all {
        let dp = back0_dp_with(&k.s, &k.qkv, &k.dout, GA104_OS335, Back0Variant::PROGRAM).unwrap();
        let v = attn_tc::judge_dp(&k.s, &k.qkv, &k.dout, &dp);
        let Verdict::InMargine {
            worst_permille,
            exact,
        } = v
        else {
            panic!("E2 {:?}: {v:?}", k.s);
        };
        assert!(worst_permille < 300, "E2 {:?}: {worst_permille} ‰", k.s);
        let ds = attn_tc::back0_model(&k.s, &k.qkv, &k.p, &k.dout, k.scale).unwrap();
        let e_dp = rel(&lower(&k.s, &dp), &lower(&k.s, &dp_f32(k)));
        let e_ds = rel(&ds, &k.ds);
        // E4: both backward programs on the tensor cores.
        let dqkv_tc = attn_tc::model(&k.s, &k.qkv, &k.p, &k.dout, &ds).unwrap();
        let dqkv_b3 = attn_tc::model(&k.s, &k.qkv, &k.p, &k.dout, &k.ds).unwrap();
        let mut e4 = [0.0f64; 3];
        for (si, sum) in SUMS.iter().enumerate() {
            let (mut g, mut w) = (Vec::new(), Vec::new());
            for b in 0..k.s.batch {
                for h in 0..k.s.heads {
                    g.extend(head_values(&k.s, &dqkv_tc, *sum, b, h));
                    w.extend(head_values(&k.s, &k.dqkv, *sum, b, h));
                }
            }
            e4[si] = rel(&g, &w);
        }
        let dv_same = (0..k.s.batch).all(|b| {
            (0..k.s.heads).all(|h| {
                head_values(&k.s, &dqkv_tc, Sum::Dv, b, h)
                    .iter()
                    .zip(&head_values(&k.s, &dqkv_b3, Sum::Dv, b, h))
                    .all(|(x, y)| x.to_bits() == y.to_bits())
            })
        });
        assert!(dv_same, "E4 dV {:?}", k.s);
        println!(
            "E2-E4 {:?}: dP InMargine cel mai rau {worst_permille} ‰ ({exact} exacte); fata de F32: dP {e_dp:.2e}, dS {e_ds:.2e}; \
             ambele inapoi pe tc: dQ {:.2e} dK {:.2e} dV {:.2e} (dV = B3 bit cu bit)",
            k.s, e4[0], e4[1], e4[2]
        );
        if k.s.t == 512 {
            assert!((2e-4..=1e-3).contains(&e_dp), "E3 dP #513: {e_dp:e}");
            assert!((3e-4..=3e-3).contains(&e_ds), "E3 dS #513: {e_ds:e}");
            for e in &e4[..2] {
                assert!((3e-4..=3e-3).contains(e), "E4 #513: {e:e}");
            }
        }
    }
}

#[test]
fn t254c_e5_planted_mistakes_caught() {
    // (a) NaN above the diagonal of P: never read, no bit changes.
    let k = case(2, 130, 96, 2, 17);
    let good = attn_tc::back0_model(&k.s, &k.qkv, &k.p, &k.dout, k.scale).unwrap();
    let t = k.s.t;
    let mut pn = k.p.clone();
    for (x, v) in pn.iter_mut().enumerate() {
        if x % t > (x / t) % t {
            *v = f32::NAN;
        }
    }
    let with_nan = attn_tc::back0_model(&k.s, &k.qkv, &pn, &k.dout, k.scale).unwrap();
    assert!(
        good.iter()
            .zip(&with_nan)
            .all(|(a, b)| a.to_bits() == b.to_bits())
    );
    println!("E5a NaN peste diagonala in P: aceiasi biti");
    // (c) the mask forgotten in the dot chain: P above the diagonal is read (NaN there) → dS differs.
    let bad = attn_tc::back0_model_with(
        &k.s,
        &k.qkv,
        &pn,
        &k.dout,
        k.scale,
        GA104_OS335,
        Back0Variant {
            dot_no_mask: true,
            ..Back0Variant::PROGRAM
        },
    )
    .unwrap();
    let e = rel(&bad, &good);
    assert!(e.is_nan() || e > 1e-2, "E5c: {e:e}"); // NaN or > 1 %
    println!("E5c masca uitata in dot: dS diferit ({e:e})");
    // (b) V without the transpose, d = 64 = T: dP out of the margin.
    let k = case(1, 64, 64, 1, 18);
    let dp = back0_dp_with(
        &k.s,
        &k.qkv,
        &k.dout,
        GA104_OS335,
        Back0Variant {
            v_untransposed: true,
            ..Back0Variant::PROGRAM
        },
    )
    .unwrap();
    let v = attn_tc::judge_dp(&k.s, &k.qkv, &k.dout, &dp);
    assert!(matches!(v, Verdict::NuTrece { .. }), "E5b: {v:?}");
    println!("E5b V fara transpusa: prins ({v:?})");
    // (d) dO without the 2^14 scale: dP in f16 subnormals, dS ≥ 1e-2 off.
    let k = case_scaled(1, 512, 64, 1, 513, 1.0);
    let ds = attn_tc::back0_model(&k.s, &k.qkv, &k.p, &k.dout, k.scale).unwrap();
    let e = rel(&ds, &k.ds);
    // Predicted ≥ 1e-2: WRONG (2.95e-3 came out; dO ~ 1e-5 still keeps ~7 bits as an f16 subnormal). What holds:
    // without the scale dS is ~10× worse than with it (3.0e-4), so the scale stays required. Kept as measured.
    let ks = case(1, 512, 64, 1, 513);
    let scaled = rel(
        &attn_tc::back0_model(&ks.s, &ks.qkv, &ks.p, &ks.dout, ks.scale).unwrap(),
        &ks.ds,
    );
    assert!(e >= 5.0 * scaled, "E5d: {e:e} vs {scaled:e}");
    println!("E5d dO fara scara: dS {e:.2e} fata de F32 (cu scara {scaled:.2e})");
}
