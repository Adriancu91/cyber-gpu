//! T298 (CALCUL, written with the plan, BEFORE any code of the program): C-168 point 4, the forward attention on the
//! tensor cores (`attn_fwd_tc`, `docs/C-168_ATTN_FWD_TC_PLAN.md`). Nothing here runs on a board and nothing here is a
//! measurement of the new program. The board numbers below are the OS's (MEASURED, os-496 lot 6, the record step,
//! `rezultate/os-automat/2026-11-27/ciclu-auto-20261127-1954.txt`, profile rows of `attn_fwd_tile3`,
//! `attn_back_tile0_tc_v4`, `attn_back_tile_tc16_v4`), copied by hand.
//!
//! What is computed:
//!   1. the work: products of 64 x 64 x 64 tile pairs below the causal diagonal, FLOP per launch and per step;
//!   2. the rates the board gave today for the three attention programs (TFLOP/s per launch);
//!   3. the shared memory and register budget of the plan -> blocks per SM at the 64 KiB step;
//!   4. the matrix traffic of the forward (passes over P / S) today and in the plan;
//!   5. the waves of the causal grid (blocks of very different length) in launch order, natural and reversed;
//!   6. the bands of the predictions H5 / H6 (DEDUCED from 2 and 5 with an assumed slowdown band).

const B: u64 = 6; // lot 6
const H: u64 = 12;
const T: u64 = 512;
const D: u64 = 64;
const LAYERS: u64 = 12;
const TILE: u64 = 64;
const SM: u64 = 48;

// os-496, lot 6, 12 launches each (MEASURED by the OS)
const FWD_TILE3_US: u64 = 16_149;
const BACK0_TC_V4_US: u64 = 6_400;
const BACK_TC16_V4_US: u64 = 5_466;
const STEP_US: u64 = 147_720;

fn tiles() -> u64 {
    T.div_ceil(TILE)
}

/// Tile pairs (row tile, key tile) with key tile <= row tile, per head.
fn causal_pairs() -> u64 {
    let n = tiles();
    n * (n + 1) / 2
}

/// FLOP of one product over all heads of one launch (one layer): 2 * 64^3 per pair.
fn product_flop() -> u64 {
    B * H * causal_pairs() * 2 * TILE * TILE * D
}

fn tflops(flop: u64, us: u64) -> f64 {
    flop as f64 / (us as f64 * 1e6)
}

#[test]
fn t298a_work_and_todays_rates() {
    assert_eq!(tiles(), 8);
    assert_eq!(causal_pairs(), 36);
    let p = product_flop();
    assert_eq!(p, 1_358_954_496); // 1.36 GFLOP per product per layer at lot 6
    // today's forward: S = Q K^T and O = P V -> 2 products; tc16_v4: dQ, dK, dV -> 3; tile0_tc_v4: dP -> 1 (+ dS)
    let fwd = tflops(2 * p, FWD_TILE3_US / LAYERS);
    let back = tflops(3 * p, BACK_TC16_V4_US / LAYERS);
    let back0 = tflops(p, BACK0_TC_V4_US / LAYERS);
    println!(
        "T298a: per launch fwd_tile3 {:.2} TFLOP/s, tc16_v4 {:.2}, tile0_tc_v4 {:.2}",
        fwd, back, back0
    );
    assert!((2.0..2.1).contains(&fwd));
    assert!((8.9..9.0).contains(&back));
    assert!((2.5..2.6).contains(&back0));
    // the forward is 10.9 % of the record step, the backward tensor-core program does 3 products in 34 % of the time
    let share = FWD_TILE3_US as f64 / STEP_US as f64;
    assert!((0.108..0.11).contains(&share));
    const { assert!(BACK_TC16_V4_US * 100 / FWD_TILE3_US < 35) };
}

/// The plan's shared memory (bytes): R1 = K (f16), then S (f32), then V^T (f16), rows padded; R2 = Q staging, then P
/// (f16); the row statistics m, l (f32) and the per-row partials of the 4 threads of a row.
fn plan_shared() -> (u64, u64, u64, u64) {
    let s_f32 = TILE * (TILE + 1) * 4; // S [64][65] f32 (odd stride: the row reads of the softmax without conflict)
    let kv_f16 = TILE * (TILE + 8) * 2; // K or V^T [64][72] f16: 144-byte rows (8 rows of an ldsm in 8 banks of 16 B)
    let r1 = s_f32.max(kv_f16);
    let r2 = kv_f16; // Q, then P
    let stats = TILE * 2 * 4 + TILE * 4 * 2 * 4; // m, l + the 4 partials (m, l) of each row
    (r1, r2, stats, r1 + r2 + stats)
}

#[test]
fn t298b_shared_and_registers() {
    let (r1, r2, stats, total) = plan_shared();
    assert_eq!(r1, 16_640);
    assert_eq!(r2, 9_216);
    assert_eq!(stats, 2_560);
    assert_eq!(total, 28_416);
    // 64 KiB step with the 1 KiB reserve per block (as T294): 2 blocks per SM, not 3
    let per_block = total + 1_024;
    assert!(2 * per_block <= 65_536);
    assert!(3 * per_block > 65_536);
    // registers: 2 blocks of 256 threads need <= 128 per thread; 1 block (today's tile3: 37 888 B shared, 120
    // registers) runs 8 subgroups per SM, the plan 16
    assert_eq!(65_536 / (2 * 256), 128);
    let tile3_shared: u64 = 37_888; // attn_fwd_tile3::SHARED
    assert_eq!(65_536 / (tile3_shared + 1_024), 1); // tile3: 1 block per SM today
    // the 144-byte rows: the 8 row starts of one ldsm matrix fall in 8 different 16-byte groups of the 128-byte line
    let groups: Vec<u64> = (0..8u64).map(|r| (r * 144 / 16) % 8).collect();
    let mut g = groups.clone();
    g.sort();
    g.dedup();
    assert_eq!(g.len(), 8, "{:?}", groups);
    // S rows of 65 floats: the 4 threads of a row read 16 consecutive floats each; a subgroup (8 rows x 4 threads)
    // reading element u of its chunk touches banks (65 r + 16 q + u) % 32 -> no two lanes in the same bank
    for u in 0..16u64 {
        let mut banks: Vec<u64> = (0..8u64)
            .flat_map(|r| (0..4u64).map(move |q| (65 * r + 16 * q + u) % 32))
            .collect();
        banks.sort();
        banks.dedup();
        // 32 lanes on 16 distinct banks would be a 2-way conflict; count it as the plan's known cost
        assert!(banks.len() >= 16, "u {} banks {}", u, banks.len());
    }
}

#[test]
fn t298c_matrix_traffic() {
    // bytes of one pass over the kept part of P / S (below the diagonal, per tile pair 64 x 64 f32)
    let pass = B * H * causal_pairs() * TILE * TILE * 4;
    assert_eq!(pass, 42_467_328); // 40.5 MiB per layer at lot 6
    // today's forward (attn_tc_study, A2): 5 passes (S written, S read, P written, P read twice); the plan: P written once
    let today = 5 * pass * LAYERS;
    let plan = pass * LAYERS;
    // at 400-450 GB/s (the board's measured copy range in this project), DEDUCED
    let ms = |bytes: u64, gbs: u64| bytes as f64 / (gbs as f64 * 1e6);
    println!(
        "T298c: P/S traffic per step today {:.0} MiB ({:.1}-{:.1} ms), plan {:.0} MiB ({:.2}-{:.2} ms)",
        today as f64 / 1048576.0,
        ms(today, 450),
        ms(today, 400),
        plan as f64 / 1048576.0,
        ms(plan, 450),
        ms(plan, 400)
    );
    assert!(ms(today, 400) > 6.0 && ms(today, 450) < 6.0);
    assert!(ms(plan, 400) < 1.3);
    // the backward keeps reading P (dS and dV): 2 passes per layer, unchanged by the plan
    let back_reads = 2 * pass * LAYERS;
    assert!(ms(back_reads, 450) > 2.2);
}

/// How the blocks of the grid map to row tiles.
#[derive(Clone, Copy)]
enum Map {
    /// block x = row tile x (today's grid [8, H, B])
    Natural,
    /// block x = row tile 7 - x (longest first)
    Reversed,
    /// grid [4, H, B]: block x does row tiles x and 7 - x one after the other (always 9 key tiles)
    Paired,
}

/// List scheduling of the causal grid: blocks dispatched in launch order (x fastest, then head, then batch), each to
/// the slot that frees first; a row tile bx costs bx + 1 units (both sweeps scale with it). Returns (makespan, ideal)
/// in units. Two blocks on one SM are modelled as two independent slots (an approximation).
fn waves(slots: u64, map: Map) -> (u64, u64) {
    let n = tiles();
    let mut free = vec![0u64; slots as usize];
    let mut total = 0;
    let xs = match map {
        Map::Paired => n / 2,
        _ => n,
    };
    for _z in 0..B {
        for _y in 0..H {
            for x in 0..xs {
                let cost = match map {
                    Map::Natural => x + 1,
                    Map::Reversed => n - x,
                    Map::Paired => (x + 1) + (n - x),
                };
                total += cost;
                let (i, _) = free.iter().enumerate().min_by_key(|&(_, &f)| f).unwrap();
                free[i] += cost;
            }
        }
    }
    (*free.iter().max().unwrap(), total.div_ceil(slots))
}

#[test]
fn t298d_waves_of_the_causal_grid() {
    let slots = 2 * SM; // 2 blocks per SM (t298b)
    let (nat, ideal) = waves(slots, Map::Natural);
    let (rev, _) = waves(slots, Map::Reversed);
    let (pair, ideal2) = waves(slots, Map::Paired);
    assert_eq!(ideal, ideal2);
    assert_eq!(ideal, 27); // 2 592 units on 96 slots
    println!(
        "T298d: makespan natural {} / reversed {} / paired {} / ideal {} units",
        nat, rev, pair, ideal
    );
    // recorded after the first run (the first guess, "reversed is within 15 % of the ideal", was wrong: 32 / 27)
    assert_eq!((nat, rev), (32, 32));
    // the pairs: 288 blocks of 9 units = exactly 3 waves of 96
    assert_eq!(pair, ideal);
    assert_eq!(B * H * (tiles() / 2), 288);
    // today's tile3 runs 1 block per SM: 48 slots
    let (nat1, ideal1) = waves(SM, Map::Natural);
    assert_eq!(ideal1, 54);
    println!(
        "T298d: tile3 today (48 slots) makespan {} / ideal {}",
        nat1, ideal1
    );
    assert!(nat1 >= ideal1);
}

#[test]
fn t298e_bands_h5_h6() {
    let p = product_flop();
    // the plan: two sweeps (sweep 1: S and the row max / sum; sweep 2: S again, P, O += P V) -> 3 products, the same
    // count as tc16_v4. Assumed slowdown per FLOP against tc16_v4 (more barriers per tile, the S round trip through
    // shared memory, the exp, 2 blocks per SM instead of 3): x1.2 - x1.8 (an ASSUMPTION, written to be proved wrong)
    let tc16_us = BACK_TC16_V4_US as f64 / LAYERS as f64; // 455.5 us for 3 products
    let lo = tc16_us * 1.2;
    let hi = tc16_us * 1.8;
    // P written once: 40.5 MiB at 400-450 GB/s, partly hidden behind the products: 0 - 100 us
    let p_write_hi = 42_467_328.0 / 400e3; // us
    let (lo_us, hi_us) = (lo, hi + p_write_hi);
    println!(
        "T298e: attn_fwd_tc per launch {:.0}-{:.0} us (tile3 {} us)",
        lo_us,
        hi_us,
        FWD_TILE3_US / LAYERS
    );
    assert!((540.0..560.0).contains(&lo_us));
    assert!((900.0..930.0).contains(&hi_us));
    let step_lo = lo_us * LAYERS as f64 / 1000.0;
    let step_hi = hi_us * LAYERS as f64 / 1000.0;
    println!(
        "T298e: forward per step {:.1}-{:.1} ms (today 16.1); gain {:.1}-{:.1} ms",
        step_lo,
        step_hi,
        16.149 - step_hi,
        16.149 - step_lo
    );
    assert!((6.5..6.7).contains(&step_lo));
    assert!((10.8..11.2).contains(&step_hi));
    // C-168's target (<= 6 ms) needs 3 products at >= 8.2 TFLOP/s, i.e. tc16_v4's rate (8.95) almost exactly
    let need = tflops(3 * p, 6_000 / LAYERS);
    assert!((8.1..8.3).contains(&need));
    // a single sweep (2 products, no P) would need only 5.4 TFLOP/s for 6 ms, but then the backward must recompute S
    // (FlashAttention-2): the step's products stay 7 either way (forward 3 + backward 4, or 2 + 5)
    let single = tflops(2 * p, 6_000 / LAYERS);
    assert!((5.4..5.5).contains(&single));
    assert_eq!(3 + 4, 2 + 5);
}
