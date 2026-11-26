//! C-157 — T271, CALCUL on the host (nothing simulated on a GPU model, nothing run on the board): the comparison of
//! os-471 (`token32` on the board, `docs/C-157_OS471.md`) with the predictions T1–T6 of `docs/C-157_TOKEN32_PREDICTII.md`
//! (T266), and the two causes found afterwards in the program listing and in the OS's code.
//!
//! Inputs MĂSURAT on the board by the OS (hub `rezultate/os-automat/2026-11-26/ciclu-auto-20261126-2321.txt`, lines
//! 66–72 and 82–97): `token32` 3 859 µs per token at G = 384, 4 339 µs at G = 768; with the driver 8 435 / 8 915 µs;
//! today's list in the same chain 1 069 µs (board) and 2 079 µs (with the driver). Everything below is DEDUS.

use gpu_compute::layer::Shape;
use nvidia_ga104::argmax as am;
use nvidia_ga104::kronos_layer::{LayerVas, WeightVas};
use nvidia_ga104::kronos_model::{self as km, ModelVas};
use nvidia_ga104::kronos_token as kt;

const LIST_BOARD: f64 = 1069.0;
const LIST_HOST: f64 = 2079.0;
const T32: [(u32, f64, f64); 2] = [(384, 3859.0, 8435.0), (768, 4339.0, 8915.0)];
/// T266's inputs: one launch start, the barriers (os-465), the warp-bound work of today's list (products + attention).
const START_US: f64 = 3.87;
const BARRIER: [(u32, f64); 2] = [(384, 2.025), (768, 3.114)];
const BOUND_US: f64 = 716.4;

#[test]
fn t271_t1_t2_confirmed_t3_t5_refuted() {
    // T3: 0.90–1.62 ms at G = 384; T4: 0.92–1.10 ms at G = 768; T5: G = 768 ≤ G = 384 − 0.1 ms.
    let (g384, g768) = (T32[0].1, T32[1].1);
    assert!(
        g384 > 1620.0 && g384 / 1620.0 > 2.3,
        "T3: outside the band, ~2.4x its slow end"
    );
    assert!(
        g768 > 1100.0 && g768 / 1100.0 > 3.9,
        "T4: outside the band, ~3.9x its slow end"
    );
    assert!(
        g768 > g384 + 400.0,
        "T5 refuted: G = 768 is ~0.48 ms SLOWER"
    );
    // Against today's list: ×3.61 and ×4.06 slower on the board.
    assert!((3.60..3.62).contains(&(g384 / LIST_BOARD)));
    assert!((4.05..4.07).contains(&(g768 / LIST_BOARD)));
}

#[test]
fn t271_the_work_multiplier() {
    // T266's model: board = one start + 87 barriers + m · 716 µs + ~2 µs. Solved for m with the measured times.
    for ((g, board, _), (gb, barrier)) in T32.iter().zip(BARRIER) {
        assert_eq!(*g, gb);
        let m = (board - START_US - 87.0 * barrier - 2.0) / BOUND_US;
        match g {
            384 => assert!((5.13..5.15).contains(&m), "m at 384 = {m}"),
            _ => assert!((5.67..5.69).contains(&m), "m at 768 = {m}"),
        }
    }
    // T266's band was m = 1.0–2.0 (G = 384) and 0.9–1.15 (G = 768): the work got ~5x longer, not 1–2x.
}

/// Kronos #513's shapes: per layer qkv 2304 rows, proj 768, fc 3072 (x = 768 floats) and out 768 rows (x = 3072
/// floats); the logits 16 384 rows of x = 768. Weights q8_0: 36 bytes per 32 values.
const LAYERS: f64 = 12.0;
fn x_bytes_per_token() -> f64 {
    let per_layer = (2304.0 + 768.0 + 3072.0) * 768.0 * 4.0 + 768.0 * 3072.0 * 4.0;
    LAYERS * per_layer + 16384.0 * 768.0 * 4.0
}
fn weight_bytes_per_token() -> f64 {
    let per_layer = (2304.0 + 768.0 + 3072.0) * 768.0 + 768.0 * 3072.0;
    (LAYERS * per_layer + 16384.0 * 768.0) / 32.0 * 36.0
}

#[test]
fn t271_cause_1_the_activations_bypass_l1() {
    // `token32` reads x through `coherent` buffers → `ld.global.strong.sys` (listing: 262 × 32 bit, 12 × 128 bit),
    // which skip the SM's L1; the separate `q8_matvec32` / `q8_fused32` read x `readonly` → `ld.global.constant`
    // (L1-cached: one SM reads x from L2 once, its 16 warps hit L1). Every ROW of `token32` reads its whole x from L2.
    let x = x_bytes_per_token();
    let w = weight_bytes_per_token();
    assert!((389.0e6..391.0e6).contains(&x), "x read per token: {x}");
    // The weights read per token: 109.7 MB (the OS uploads 110 786 560 bytes: + embeddings, biases, LayerNorms).
    assert!((109.0e6..110.5e6).contains(&w), "weights per token: {w}");
    // Bytes per token through L2 ×4.55 (x + weights against weights alone). If the work time follows the bytes as
    // today (latency-bound, similar bytes in flight per warp), m ≈ 4.55 — measured 5.1 (G = 384) / 5.7 (G = 768).
    let ratio = (x + w) / w;
    assert!((4.5..4.6).contains(&ratio), "{ratio}");
    let predicted_384 = START_US + 87.0 * BARRIER[0].1 + ratio * BOUND_US + 2.0;
    assert!((3420.0..3460.0).contains(&predicted_384), "{predicted_384}");
    // 89 % of the measured 3 859 µs: the cause explains most of it; the rest (~0.4 ms at 384, ~0.7 ms at 768) is NOT
    // explained (candidates: the attention's scalar strong loads of q and K/V, the polling of the barrier counter by
    // 384 / 768 warps on one L2 line while the late groups still load).
    assert!(predicted_384 / T32[0].1 > 0.88);
}

fn vas(layers: usize, e: usize) -> ModelVas {
    let mut next = 0x1_0000_0000u64;
    let mut take = |n: u64| {
        let a = next;
        next += n.div_ceil(256) * 256;
        a
    };
    let scratch = take(nvidia_ga104::kronos_layer::scratch_bytes(e));
    let layers = (0..layers)
        .map(|_| {
            let kc = take(nvidia_ga104::kronos_layer::cache_bytes(e));
            let vc = take(nvidia_ga104::kronos_layer::cache_bytes(e));
            let mut w = || take(4 * 4 * e as u64 * e as u64);
            let wv = WeightVas {
                ln1_g: w(),
                ln1_b: w(),
                wqkv: w(),
                bqkv: w(),
                wout: w(),
                bout: w(),
                ln2_g: w(),
                ln2_b: w(),
                wfc: w(),
                bfc: w(),
                wproj: w(),
                bproj: w(),
            };
            LayerVas::with_scratch(e, scratch, kc, vc, wv).unwrap()
        })
        .collect();
    ModelVas {
        emb_tok: take(16384 * 4 * e as u64),
        emb_pos: take(512 * 4 * e as u64),
        ln_g: take(4 * e as u64),
        ln_b: take(4 * e as u64),
        layers,
        h_final: take(4 * e as u64),
        logits: take(4 * 16384),
    }
}

#[test]
fn t271_cause_2_the_driver_rewrites_the_whole_list() {
    // The OS (`gpu_i14.rs`, `submit32`) writes `work_list32` whole before every token: 88 × 256 B through the BAR.
    let s = Shape {
        e: 768,
        heads: 12,
        eps: 1e-5,
    };
    let v = vas(12, 768);
    let slot = 0x2_0000_0000u64;
    let list = |tok: usize, pos: usize| {
        let mut l = km::token_launches_fused_ln32(&s, 16384, 512, &v, tok, pos).unwrap();
        l.push(am::launch(v.logits, slot, 16384).unwrap());
        kt::work_list32(&l).unwrap()
    };
    let (a, b, c) = (list(5, 3), list(200, 4), list(200, 5));
    assert_eq!(a.len(), 88 * kt::ENTRY);
    let whole = a.len() / 4;
    assert_eq!(whole, 5632);
    // Driver part per token, MĂSURAT: list 2 079 − 1 069 = 1 010 µs; token32 8 435 − 3 859 = 4 576 µs (+3 566 µs).
    let extra = (T32[0].2 - T32[0].1) - (LIST_HOST - LIST_BOARD);
    assert!((3560.0..3570.0).contains(&extra));
    // ≈ 0.63 µs per word of the list: the OS's word-by-word BAR writes (os-324: ~0.6–0.8 µs per uncached word).
    assert!((0.62..0.64).contains(&(extra / whole as f64)));
    // What really changes from one token to the next (new token and new position): a handful of words.
    for (p, n) in [(&a, &b), (&b, &c)] {
        let ch = kt::work_list_changes(p, n).unwrap();
        assert!(
            !ch.is_empty() && ch.len() <= 48,
            "changed words: {}",
            ch.len()
        );
        let mut patched = p.clone();
        for &(at, w) in &ch {
            patched[at..at + 4].copy_from_slice(&w.to_le_bytes());
        }
        assert_eq!(&patched, n, "the changes rebuild the next list exactly");
    }
    let n_ab = kt::work_list_changes(&a, &b).unwrap().len();
    eprintln!("T271: changed words between consecutive tokens: {n_ab} of {whole}");
    // Refusals: other lengths.
    assert!(kt::work_list_changes(&a, &a[..a.len() - 4]).is_err());
    assert!(kt::work_list_changes(&[], &[]).is_err());
    assert!(kt::work_list_changes(&a, &a).unwrap().is_empty());
}
