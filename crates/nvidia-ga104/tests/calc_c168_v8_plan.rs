//! T294 (CALCUL, written with the plan, BEFORE any v8 code): C-168 point 3, the "v8" matmul. Counts, for the 147
//! matmuls of the #513 step at lot 6 (os-496: `mm_tc_v7` 98.1 ms), how many blocks each tile size makes and how full
//! the last wave is, with 48 SM and the blocks per SM that the 64 KiB step allows:
//!   v7     64 x 64,  128 threads, 14 336 B shared                 -> 4 blocks per SM (192 slots)
//!   v8    128 x 128, 256 threads, 2 x 12 288 B (two K=16 stages)  -> 2 blocks per SM (96 slots)
//!   v8n   128 x 64,  128 threads, 2 x 9 216 B                     -> 3 blocks per SM (144 slots)
//! Nothing here runs on a board and nothing here is a v8 measurement. The per-block speed of v8 is an ASSUMPTION
//! (band f = 1.25-1.60 x v7's per-SM rate at full waves), written down so that the board can prove it wrong.

const T: u32 = 3072; // tokens at lot 6 (6 x 512)
const E: u32 = 768;
const V: u32 = 16_384;
const SM: u32 = 48;

#[derive(Clone, Copy)]
struct Mm {
    label: &'static str,
    m: u32,
    n: u32,
    k: u32,
    count: u32,
}

fn shapes() -> Vec<Mm> {
    let l = 12;
    vec![
        Mm {
            label: "inainte:qkv",
            m: T,
            n: 3 * E,
            k: E,
            count: l,
        },
        Mm {
            label: "inainte:out",
            m: T,
            n: E,
            k: E,
            count: l,
        },
        Mm {
            label: "inainte:fc",
            m: T,
            n: 4 * E,
            k: E,
            count: l,
        },
        Mm {
            label: "inainte:proj",
            m: T,
            n: E,
            k: 4 * E,
            count: l,
        },
        Mm {
            label: "inainte:logit",
            m: T,
            n: V,
            k: E,
            count: 1,
        },
        Mm {
            label: "inapoi:qkv-dX",
            m: T,
            n: E,
            k: 3 * E,
            count: l,
        },
        Mm {
            label: "inapoi:out-dX",
            m: T,
            n: E,
            k: E,
            count: l,
        },
        Mm {
            label: "inapoi:fc-dX",
            m: T,
            n: E,
            k: 4 * E,
            count: l,
        },
        Mm {
            label: "inapoi:proj-dX",
            m: T,
            n: 4 * E,
            k: E,
            count: l,
        },
        Mm {
            label: "inapoi:cap-dX",
            m: T,
            n: E,
            k: V,
            count: 1,
        },
        Mm {
            label: "inapoi:qkv-dW",
            m: E,
            n: 3 * E,
            k: T,
            count: l,
        },
        Mm {
            label: "inapoi:out-dW",
            m: E,
            n: E,
            k: T,
            count: l,
        },
        Mm {
            label: "inapoi:fc-dW",
            m: E,
            n: 4 * E,
            k: T,
            count: l,
        },
        Mm {
            label: "inapoi:proj-dW",
            m: 4 * E,
            n: E,
            k: T,
            count: l,
        },
        Mm {
            label: "inapoi:cap-dW",
            m: E,
            n: V,
            k: T,
            count: 1,
        },
    ]
}

fn flop(s: &Mm) -> f64 {
    2.0 * s.m as f64 * s.n as f64 * s.k as f64 * s.count as f64
}

/// (rows, cols of C per block, blocks per SM at 64 KiB)
const V7: (u32, u32, u32) = (64, 64, 4);
const V8: (u32, u32, u32) = (128, 128, 2);
const V8N: (u32, u32, u32) = (128, 64, 3);

fn blocks(s: &Mm, t: (u32, u32, u32)) -> u32 {
    s.m.div_ceil(t.0) * s.n.div_ceil(t.1)
}

/// Useful share of the slots: blocks / (waves * slots).
fn eff(s: &Mm, t: (u32, u32, u32)) -> f64 {
    let b = blocks(s, t);
    let slots = SM * t.2;
    b as f64 / (b.div_ceil(slots) * slots) as f64
}

/// Shared bytes per block (A and B slices in f16, rows padded by 8 halves) and blocks per SM at the 64 KiB step with
/// the 1 KiB reserve per block (the rule measured on the board, os-423 / os-469).
#[test]
fn t294_shared_and_blocks_per_sm() {
    let stage = |bm: u32, bn: u32, bk: u32| (bm + bn) * (bk + 8) * 2;
    // v7: one stage at BK 32 = 10 240 B, plus its own 4 KiB output area = 14 336 B (as compiled).
    let v7 = stage(64, 64, 32);
    assert_eq!(v7 + 4 * 1024, 14_336);
    let v8 = 2 * stage(128, 128, 16);
    let v8n = 2 * stage(128, 64, 16);
    assert_eq!(v8, 24_576);
    assert_eq!(v8n, 18_432);
    // The output staging (one 16x16 f32 per warp: 8 KiB for 8 warps, 4 KiB for 4) reuses the stage buffers after the
    // last slice, so it adds nothing.
    assert!(8 * 1024 <= v8 && 4 * 1024 <= v8n);
    let per_sm = |bytes: u32| (64 * 1024) / (bytes + 1024);
    assert_eq!(per_sm(14_336), V7.2);
    assert_eq!(per_sm(v8), V8.2);
    assert_eq!(per_sm(v8n), V8N.2);
    // Registers: 65 536 per SM. v8 needs <= 128 per thread for 2 blocks (256 threads), v8n <= 170 for 3 (128 threads).
    assert_eq!(65_536 / (256 * 128), 2);
    assert_eq!(65_536 / (128 * 168), 3);
    // Conflict-free ldsm on rows of 48 B (BK 16 + 8 halves): the 8 row starts of one 8x8 matrix fall on 8 different
    // 16-byte groups of the 128-byte bank line.
    let groups: Vec<u32> = (0..8).map(|r| (r * 48 % 128) / 16).collect();
    let mut g = groups.clone();
    g.sort_unstable();
    g.dedup();
    assert_eq!(g.len(), 8, "{groups:?}");
    println!(
        "kt c294 memoria CALCUL: v8 {v8} B (2 trepte K=16) -> {} blocuri pe SM; v8n {v8n} B -> {}; v7 14 336 B -> {}; randuri de 48 B fara conflict la ldsm",
        per_sm(v8),
        per_sm(v8n),
        per_sm(14_336)
    );
}

/// The 147 matmuls = 1.797 TFLOP (T268), and the wave picture per shape.
#[test]
fn t294_waves_per_shape() {
    let sh = shapes();
    let n: u32 = sh.iter().map(|s| s.count).sum();
    assert_eq!(n, 147);
    let total: f64 = sh.iter().map(flop).sum();
    assert!((total / 1e12 - 1.797).abs() < 0.001, "{total}");
    for s in &sh {
        println!(
            "kt c294 forma {:<16} {:>5}x{:<5} K {:>5} x{:>2}: blocuri v7 {:>4} ({:>3.0} %), v8 {:>4} ({:>3.0} %), v8n {:>4} ({:>3.0} %)",
            s.label,
            s.m,
            s.n,
            s.k,
            s.count,
            blocks(s, V7),
            eff(s, V7) * 100.0,
            blocks(s, V8),
            eff(s, V8) * 100.0,
            blocks(s, V8N),
            eff(s, V8N) * 100.0
        );
    }
    // The shapes with N = 768 (or M = 768 x N = 768) are where the big tile loses: out-dW has 36 blocks of 128 x 128
    // for 96 slots (37.5 %), qkv-dX / fc-dX / out / proj have 144 (75 %).
    let out_dw = sh.iter().find(|s| s.label == "inapoi:out-dW").unwrap();
    assert_eq!(blocks(out_dw, V8), 36);
    assert!((eff(out_dw, V8) - 0.375).abs() < 1e-9);
    // v7 already wastes a quarter on out-dW (144 blocks on 192 slots).
    assert!((eff(out_dw, V7) - 0.75).abs() < 1e-9);
    let qkv = sh.iter().find(|s| s.label == "inainte:qkv").unwrap();
    assert_eq!(blocks(qkv, V8), 432);
    assert!((eff(qkv, V8) - 0.9).abs() < 1e-9);
}

/// The step prediction band (DEDUS): v7's per-SM rate R7 at full waves is calibrated so that the 147 matmuls take the
/// measured 98.1 ms with v7's wave losses; v8 = f * R7, v8n = 0.9 * f * R7 (same warp tile, half the reuse of B per
/// block); each shape takes the fastest of the three. f = 1.25 / 1.60.
#[test]
fn t294_step_band() {
    let sh = shapes();
    let measured = 98.099e-3;
    let r7 = sh.iter().map(|s| flop(s) / eff(s, V7)).sum::<f64>() / measured;
    assert!((18.5e12..21.0e12).contains(&r7), "{r7}");
    let step = |f: f64| -> (f64, u32, u32, u32) {
        let (mut t, mut p7, mut p8, mut p8n) = (0.0, 0u32, 0u32, 0u32);
        for s in &sh {
            let c7 = flop(s) / (r7 * eff(s, V7));
            let c8 = flop(s) / (f * r7 * eff(s, V8));
            let c8n = flop(s) / (0.9 * f * r7 * eff(s, V8N));
            let best = c7.min(c8).min(c8n);
            if best == c7 {
                p7 += s.count;
            } else if best == c8 {
                p8 += s.count;
            } else {
                p8n += s.count;
            }
            t += best;
        }
        (t, p7, p8, p8n)
    };
    let (lo, a7, a8, a8n) = step(1.60);
    let (hi, b7, b8, b8n) = step(1.25);
    assert!(lo < hi && hi < measured);
    let board = 147.810e-3;
    let rest = board - measured;
    let sps = |mm: f64| 3072.0 / (rest + mm);
    // Only v8 at 128 x 128 everywhere (no picker): what the waves cost.
    let only_v8 = |f: f64| {
        sh.iter()
            .map(|s| flop(s) / (f * r7 * eff(s, V8)))
            .sum::<f64>()
    };
    println!(
        "kt c294 banda DEDUS: R7 {:.1} TFLOP/s la valuri pline; f 1,60: inmultirile {:.1} ms (v7 {a7} / v8 {a8} / v8n {a8n}), pasul {:.1} ms = {:.0} simb/s; f 1,25: {:.1} ms (v7 {b7} / v8 {b8} / v8n {b8n}), pasul {:.1} ms = {:.0} simb/s; doar v8 peste tot: {:.1} / {:.1} ms",
        r7 / 1e12,
        lo * 1e3,
        (rest + lo) * 1e3,
        sps(lo),
        hi * 1e3,
        (rest + hi) * 1e3,
        sps(hi),
        only_v8(1.60) * 1e3,
        only_v8(1.25) * 1e3
    );
    // The band written in the plan (C-168_V8_PLAN.md §4, prediction G5) must be this one.
    assert!((60e-3..70e-3).contains(&lo), "{lo}");
    assert!((75e-3..86e-3).contains(&hi), "{hi}");
}

/// A side result of the wave count (DEDUS): the "dW x1.28 slower than dX at the same work" of os-490 (qkv-513 shape:
/// dX 1.147 ms, dW 1.464 ms on v7@64k) is mostly the last wave. dX (3072 x 768) = 576 blocks of 64 x 64 = 3 full
/// waves of 192; dW (768 x 2304) = 432 blocks = 2.25 waves -> 3 waves paid for 2.25 of work: x1.33 predicted.
#[test]
fn t294_dw_slower_than_dx_is_the_last_wave() {
    let dx = Mm {
        label: "qkv-513-dx",
        m: T,
        n: E,
        k: 3 * E,
        count: 1,
    };
    let dw = Mm {
        label: "qkv-513-dw",
        m: E,
        n: 3 * E,
        k: T,
        count: 1,
    };
    assert_eq!(flop(&dx), flop(&dw));
    let predicted = eff(&dx, V7) / eff(&dw, V7);
    let measured = 1.464 / 1.147;
    assert!((predicted - 4.0 / 3.0).abs() < 1e-9);
    // The waves explain the ratio within 5 %; the rest (dW reads A transposed: TA) is small.
    assert!(
        (predicted / measured - 1.0).abs() < 0.05,
        "{predicted} {measured}"
    );
    // With v8n (128 x 64, 144 slots) dW is 216 blocks = 1.5 waves (75 %): the same loss. The cure for dW is a tile
    // whose block count for 768 x 2304 is a multiple of the slots (not chosen in this plan).
    assert!((eff(&dw, V8N) - 0.75).abs() < 1e-9);
    println!(
        "kt c294 dW CALCUL+DEDUS: dX {} blocuri (3 valuri pline), dW {} blocuri (2,25 valuri) -> x{predicted:.3} prezis din valuri, masurat x{measured:.3} (os-490)",
        blocks(&dx, V7),
        blocks(&dw, V7)
    );
}
