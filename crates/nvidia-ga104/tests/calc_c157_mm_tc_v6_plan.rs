//! C-157 — T275, CALCUL on the host (no program simulated, nothing run on the board): the plan and the predictions for
//! `mm_tc_v6` (`mm_tc_v5` with the f16 slices written to shared memory four values at a time, one 64-bit store, and
//! the transposed operands kept column-major so those stores stay contiguous), written in
//! `docs/C-157_MM_TC_V6_PLAN.md` BEFORE any shader code or compilation.
//!
//! What is computed exactly here: the shared-memory wavefronts of the slice stores of `mm_tc_v5` (read off
//! `tools/nak/mm_tc_v5.comp`: four 16-bit stores per vector) and of the planned `mm_tc_v6` (one 64-bit store per
//! vector), under the public CUDA bank model for compute capability 8.6 (32 banks of 4 bytes; a 64-bit access is
//! served per half-warp). Measured inputs (MĂSURAT, board reports the OS sent to the hub): os-476 (`mm_tc_v5` qkv-513
//! at the 64 KiB step: 1 868 192 ns), os-473 (#513 lot 2 step: 97,08 ms on the board), os-468 (lot 6: `mm_tc_v4`
//! 216,6 ms of 273,1). Everything derived from them is DEDUS.

use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
use nvidia_ga104::wave_model::SMS;

const BM: u32 = 64;
const BN: u32 = 64;
const BK: u32 = 32;
const THREADS: u32 = 128;
/// vec4 loads per thread per slice, for A and for B (`NV` in the shader).
const NV: u32 = (BM * BK) / (4 * THREADS);

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    A,
    B,
}

/// (row-or-k index, k-or-col index) of the first element of vector `s` of thread `tid`, exactly as `load_slice`
/// in `mm_tc_v5.comp`: for A (r, p), for B (p, c).
fn vector_origin(op: Op, trans: bool, e: u32) -> (u32, u32) {
    match (op, trans) {
        (Op::A, false) => (e / 8, 4 * (e % 8)),
        (Op::A, true) => (4 * (e % 16), e / 16),
        (Op::B, false) => (e / 16, 4 * (e % 16)),
        (Op::B, true) => (4 * (e % 8), e / 8),
    }
}

/// Wavefronts one warp-wide shared store needs: lanes are grouped (all 32 for ≤ 4 bytes, half-warps for 8 bytes);
/// in each group, the bank with the most distinct 4-byte words sets the count. Two lanes writing the two halves of
/// one word share that word (no conflict).
fn wavefronts(bytes: &[u32; 32], width: u32) -> u32 {
    // (first lane, lanes) of each group
    let groups: &[(usize, usize)] = if width <= 4 {
        &[(0, 32)]
    } else {
        &[(0, 16), (16, 16)]
    };
    let mut total = 0;
    for &(first, n) in groups {
        let mut words: [Vec<u32>; 32] = Default::default();
        for &addr in &bytes[first..first + n] {
            for w in 0..width.div_ceil(4) {
                let word = addr / 4 + w;
                let b = &mut words[(word % 32) as usize];
                if !b.contains(&word) {
                    b.push(word);
                }
            }
        }
        total += words.iter().map(|w| w.len() as u32).max().unwrap();
    }
    total
}

/// `mm_tc_v5`: per slice, per warp, the wavefronts of the 16 stores of one operand (four `float16_t` per vector,
/// row-major `as_[r][p]` stride BK, `bs_[p][c]` stride BN, step 1 along the vector or BK / BN when transposed).
fn v5_store_wavefronts(op: Op, trans: bool, warp: u32) -> (u32, u32) {
    let (mut stores, mut wf) = (0, 0);
    for s in 0..NV {
        for q in 0..4 {
            let mut bytes = [0u32; 32];
            for (lane, b) in bytes.iter_mut().enumerate() {
                let e = warp * 32 + lane as u32 + s * THREADS;
                let (x, y) = vector_origin(op, trans, e);
                let (base, step) = match op {
                    Op::A => (x * BK + y, if trans { BK } else { 1 }),
                    Op::B => (x * BN + y, if trans { BN } else { 1 }),
                };
                *b = 2 * (base + q * step);
            }
            stores += 1;
            wf += wavefronts(&bytes, 2);
        }
    }
    (stores, wf)
}

/// The planned `mm_tc_v6`: one 64-bit store per vector. Not transposed: the same row-major arrays as v5 (the four
/// values are already contiguous). Transposed: op(A) kept column-major `[p][r]` (stride BM), op(B) column-major
/// `[c][p]` (stride BK), so the four values of a vector are contiguous there too.
fn v6_store_wavefronts(op: Op, trans: bool, warp: u32) -> (u32, u32) {
    let (mut stores, mut wf) = (0, 0);
    for s in 0..NV {
        let mut bytes = [0u32; 32];
        for (lane, b) in bytes.iter_mut().enumerate() {
            let e = warp * 32 + lane as u32 + s * THREADS;
            let (x, y) = vector_origin(op, trans, e);
            let base = match (op, trans) {
                (Op::A, false) => x * BK + y,
                (Op::A, true) => y * BM + x,
                (Op::B, false) => x * BN + y,
                (Op::B, true) => y * BK + x,
            };
            assert_eq!(base % 4, 0, "a 64-bit store must be 8-byte aligned");
            *b = 2 * base;
        }
        stores += 1;
        wf += wavefronts(&bytes, 8);
    }
    (stores, wf)
}

/// Every element of the 64 × 32 (A) or 32 × 64 (B) slice is written exactly once by the v6 stores.
fn v6_covers_once(op: Op, trans: bool) -> bool {
    let mut hit = vec![0u8; (BM * BK) as usize];
    for e in 0..NV * THREADS {
        let (x, y) = vector_origin(op, trans, e);
        let base = match (op, trans) {
            (Op::A, false) => x * BK + y,
            (Op::A, true) => y * BM + x,
            (Op::B, false) => x * BN + y,
            (Op::B, true) => y * BK + x,
        };
        for q in 0..4 {
            hit[(base + q) as usize] += 1;
        }
    }
    hit.iter().all(|&h| h == 1)
}

/// Per warp per slice, (A, B) wavefronts for one flag combination (summed over the 4 warps, then / 4).
fn per_warp(f: fn(Op, bool, u32) -> (u32, u32), ta: bool, tb: bool) -> (u32, u32) {
    let a: u32 = (0..4).map(|w| f(Op::A, ta, w).1).sum::<u32>() / 4;
    let b: u32 = (0..4).map(|w| f(Op::B, tb, w).1).sum::<u32>() / 4;
    (a, b)
}

#[test]
fn t275_v5_stores_conflict_on_the_transposed_operands() {
    // 16 stores per operand per warp per slice (4 vectors × 4 halves), as in the NAK listing (32 u16 stores)
    assert_eq!(v5_store_wavefronts(Op::A, false, 0).0, 16);
    // forward (no flag): 2-way on both operands
    assert_eq!(per_warp(v5_store_wavefronts, false, false), (32, 32));
    // dX = dY · Wᵀ (TB): B 8-way
    assert_eq!(per_warp(v5_store_wavefronts, false, true), (32, 128));
    // dW = Xᵀ · dY (TA): A 16-way (four lanes' rows 64 bytes apart × 4 → the same bank)
    assert_eq!(per_warp(v5_store_wavefronts, true, false), (256, 32));
    assert_eq!(per_warp(v5_store_wavefronts, true, true), (256, 128));
}

#[test]
fn t275_v6_stores_are_contiguous_in_every_case() {
    for ta in [false, true] {
        for tb in [false, true] {
            assert_eq!(v6_store_wavefronts(Op::A, ta, 0).0, 4); // 4 stores of 64 bits per operand
            assert_eq!(per_warp(v6_store_wavefronts, ta, tb), (8, 8)); // 2 half-warps × 4, no conflict
        }
        assert!(v6_covers_once(Op::A, ta));
        assert!(v6_covers_once(Op::B, ta));
    }
}

#[test]
fn t275_store_cycles_per_block_slice() {
    // one wavefront per SM clock; 4 warps per block
    let cyc = |f: fn(Op, bool, u32) -> (u32, u32), ta, tb| {
        let (a, b) = per_warp(f, ta, tb);
        4 * (a + b)
    };
    let v5 = [
        cyc(v5_store_wavefronts, false, false),
        cyc(v5_store_wavefronts, false, true),
        cyc(v5_store_wavefronts, true, false),
    ];
    assert_eq!(v5, [256, 640, 1_152]); // forward, dX, dW
    let v6 = cyc(v6_store_wavefronts, true, false);
    assert_eq!(v6, 64);
    // the #513 step: forward, dX and dW have the same FLOP (2·t·i·o each) → each a third of the matmul time
    let mean_v5 = v5.iter().sum::<u32>() / 3;
    assert_eq!(mean_v5, 682);

    // What one block-slice costs today, from os-476 (qkv-513, forward, v5 at 64 KiB, 4 blocks per SM):
    // 2·64·64·32 FLOP per block-slice; 11.63 TFLOP/s over 48 SMs at 1 950 MHz → SM clocks per block-slice.
    let flop_bs = 2 * 64 * 64 * 32u64;
    let probe_flop = 2 * 6_144 * 2_304 * 768u64;
    let probe_ns = 1_868_192u64;
    let clocks_per_bs = SMS as u64 * 1_950 * probe_ns / 1_000 * flop_bs / probe_flop;
    assert_eq!(clocks_per_bs, 2_108);
    // share of the store pipe in that time: forward 12 %, and if dX / dW ran at the same speed: 30 % / 55 %
    let share = |c: u32| c as u64 * 1_000 / clocks_per_bs;
    assert_eq!((share(256), share(640), share(1_152)), (121, 303, 546));
    // forward ceiling of the gain if the stores were fully on the critical path: 2108 / (2108 − 192) = ×1.100
    assert_eq!(clocks_per_bs * 1_000 / (clocks_per_bs - 192), 1_100);
}

#[test]
fn t275_occupancy_unchanged_unless_registers_grow() {
    // same 12 KiB as v5 (as_ 4 KiB + bs_ 4 KiB as f16vec4 arrays, the same bytes; st_ 4 KiB)
    let v6 = Kernel {
        name: "mm_tc_v6",
        threads: 128,
        registers: 128,
        shared_bytes: 12_288,
    };
    for r in [Reserve::None, Reserve::OneKib] {
        assert_eq!(blocks_per_sm(&SM86, &v6, 64 << 10, r).unwrap().blocks, 4);
    }
    // v5 sits exactly at 128 registers: one more register step and v6 falls to 3 blocks (V1b → 4 fixed-flag binaries)
    let fat = Kernel {
        registers: 136,
        ..v6
    };
    assert_eq!(
        blocks_per_sm(&SM86, &fat, 64 << 10, Reserve::None)
            .unwrap()
            .blocks,
        3
    );
}

#[test]
fn t275_the_predicted_bands() {
    // V7: forward probe qkv-513 at 64 KiB: 1.868 ms → 1.70–1.87 ms (×1.00–1.10; the ceiling above is ×1.100)
    let p = 1_868_192u64;
    assert_eq!((p * 1_000 / 1_100 / 1_000, p / 1_000), (1_698, 1_868));
    // V8: the #513 step against the same step with v5 (not measured yet): matmuls ×1.03–1.20 (central ×1.10);
    // the step keeps its other part. Lot 2: matmuls ≈ 69 % of the board time (os-473); lot 6: ≈ 79 % (os-468).
    let ratio = |share: u64, g: u64| (1_000 - share) + share * 1_000 / g; // board(v6) / board(v5), thousandths
    assert_eq!(
        (ratio(690, 1_200), ratio(690, 1_100), ratio(690, 1_030)),
        (885, 937, 979)
    );
    assert_eq!(
        (ratio(793, 1_200), ratio(793, 1_100), ratio(793, 1_030)),
        (867, 927, 976)
    );
    // threshold written before: matmuls under ×1.03 in the step → the stores are not on the critical path
}
