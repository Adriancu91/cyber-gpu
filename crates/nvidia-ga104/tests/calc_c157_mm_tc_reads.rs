//! C-157 — T279, CALCUL on the host (no program simulated, nothing run on the board): the shared-memory **read**
//! conflicts of the `coopMatLoad`s in the slice loop of `mm_tc_v5` / `mm_tc_v6`, on the three flag cases of the #513
//! step (forward = `TB`, dX = no flag, dW = `TA`; correction of session 264), and the plan for padded slice rows
//! (`docs/C-157_CITIRI_CONFLICTE.md`), written BEFORE any shader code.
//!
//! The per-lane addresses are read off the NAK listing `tools/nak/out2/mm_tc_v6.sm86.txt` (the loop blocks
//! L361..L369; `r26` = `s2r sr[0x0]`, the lane id): the test checks that the listing still holds exactly these
//! loads, then counts wavefronts under the public CUDA bank model for compute capability 8.6 (32 banks of 4 bytes;
//! a 32-bit `lds` is one warp-wide request; an `ldsm.16.m8n8.x4` is served one 8 × 8 matrix at a time, the 8 row
//! addresses of 16 bytes from lanes 8m..8m+7). `mm_tc_v5` has the same reads on every flag (its listing: the same
//! lane formulas, `r36` / `r5`), equal to v6 without flags.

use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};

const LISTING_V6: &str = include_str!("../../../tools/nak/out2/mm_tc_v6.sm86.txt");
const LISTING_V5: &str = include_str!("../../../tools/nak/out2/mm_tc_v5.sm86.txt");

const BM: u32 = 64;
const BN: u32 = 64;
const BK: u32 = 32;
const THREADS: u32 = 128;
const NV: u32 = (BM * BK) / (4 * THREADS);
/// Shared layout of v5 / v6: `as4_` at 0, `bs4_` at 0x1000 (4 KiB each), `st_` at 0x2000.
const B_BASE: u32 = 0x1000;

/// Distinct 4-byte words per bank, max over banks, for one request (a set of byte addresses each `width` bytes).
fn wavefronts(addrs: &[u32], width: u32) -> u32 {
    let mut words: [Vec<u32>; 32] = Default::default();
    for &a in addrs {
        for w in 0..width / 4 {
            let word = a / 4 + w;
            let b = &mut words[(word % 32) as usize];
            if !b.contains(&word) {
                b.push(word);
            }
        }
    }
    words.iter().map(|w| w.len() as u32).max().unwrap()
}

/// A 32-bit `lds` of one warp: one request of 32 lanes.
fn lds_wf(addr: impl Fn(u32) -> u32) -> u32 {
    let a: Vec<u32> = (0..32).map(addr).collect();
    wavefronts(&a, 4)
}

/// An `ldsm.16.m8n8.x4`: four 8 × 8 matrices, lanes 8m..8m+7 give the 16-byte rows of matrix m.
fn ldsm_wf(addr: impl Fn(u32) -> u32) -> u32 {
    (0..4)
        .map(|m| {
            let a: Vec<u32> = (8 * m..8 * m + 8).map(&addr).collect();
            wavefronts(&a, 16)
        })
        .sum()
}

/// Lane formulas of the listing, with the slice row stride `s` in bytes (v6: the strides of its four layouts).
/// `lds` path (the operand needs a transpose in registers, `movm`): 8 rows of 4 words.
fn lds_lane(s: u32) -> impl Fn(u32) -> u32 {
    move |l| s * (l >> 2) + 4 * (l & 3)
}
/// op(A) row-major (`[r][p]`, no `TA`) through `ldsm`: rows `l & 15`, second 16 bytes for lanes 16..31.
fn ldsm_a_lane(s: u32) -> impl Fn(u32) -> u32 {
    move |l| s * (l & 15) + (l & 16)
}
/// op(B) column-major (`[c][p]`, `TB`) through `ldsm`: rows `(l & 7) | ((l & 16) >> 1)`, 16 bytes more for lane bit 8.
fn ldsm_b_lane(s: u32) -> impl Fn(u32) -> u32 {
    move |l| s * ((l & 7) | ((l & 16) >> 1)) + 2 * (l & 8)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Path {
    Lds,
    Ldsm,
}

/// Which load each operand takes on a flag case: op(A) uses `ldsm` unless transposed; op(B) uses `ldsm` only
/// when transposed (column-major is the shape the tensor cores want for B).
fn paths(ta: bool, tb: bool) -> (Path, Path) {
    (
        if ta { Path::Lds } else { Path::Ldsm },
        if tb { Path::Ldsm } else { Path::Lds },
    )
}

/// Row strides in bytes: (A row-major, A column-major, B row-major, B column-major).
#[derive(Clone, Copy)]
struct Strides {
    a_row: u32,
    a_col: u32,
    b_row: u32,
    b_col: u32,
}
/// v5 / v6 as compiled: BK = 32 halves (64 B), BM = BN = 64 halves (128 B).
const DENSE: Strides = Strides {
    a_row: 64,
    a_col: 128,
    b_row: 128,
    b_col: 64,
};
/// The plan (v7): 8 halves (16 bytes) of padding per row.
const PADDED: Strides = Strides {
    a_row: 80,
    a_col: 144,
    b_row: 144,
    b_col: 80,
};

/// Read wavefronts of one warp in one K = 32 slice: (op(A), op(B)). Per K = 16 step: two fragments per operand;
/// on the `lds` path one fragment pair is 8 loads (4 per 16 × 16 fragment), on the `ldsm` path one `x4` per fragment.
fn reads_per_warp(st: Strides, ta: bool, tb: bool) -> (u32, u32) {
    let (pa, pb) = paths(ta, tb);
    let a = match pa {
        Path::Ldsm => 2 * ldsm_wf(ldsm_a_lane(st.a_row)),
        Path::Lds => 8 * lds_wf(lds_lane(st.a_col)),
    };
    let b = match pb {
        Path::Ldsm => 2 * ldsm_wf(ldsm_b_lane(st.b_col)),
        Path::Lds => 8 * lds_wf(lds_lane(st.b_row)),
    };
    (2 * a, 2 * b) // two K = 16 steps per slice
}

fn reads_per_block(st: Strides, ta: bool, tb: bool) -> u32 {
    let (a, b) = reads_per_warp(st, ta, tb);
    4 * (a + b)
}

/// The loop loads of a listing: (true = ldsm, byte offset), in order, without the ending's `st_` reads (`.x4`).
fn loop_loads(listing: &str) -> Vec<(bool, u32)> {
    let mut v = Vec::new();
    for line in listing.lines() {
        let is_ldsm = line.contains("ldsm.16.m8n8.x4 [");
        let is_lds = line.contains("ld.shared.strong.cta.b32 [") && !line.contains(".x4+");
        if !(is_ldsm || is_lds) {
            continue;
        }
        let inside = &line[line.find('[').unwrap() + 1..line.find(']').unwrap()];
        let off = inside
            .split('+')
            .find_map(|p| p.strip_prefix("0x"))
            .map(|h| u32::from_str_radix(h, 16).unwrap())
            .unwrap_or(0);
        v.push((is_ldsm, off));
    }
    v
}

#[test]
fn t279_the_listing_holds_these_loads() {
    let v6 = loop_loads(LISTING_V6);
    let lds: Vec<u32> = v6.iter().filter(|x| !x.0).map(|x| x.1).collect();
    let ldsm: Vec<u32> = v6.iter().filter(|x| x.0).map(|x| x.1).collect();
    // op(A) TA (kk 0, kk 16), op(B) no TB (kk 0, kk 16): 8 + 8 + 8 + 8 = 32 lds
    let a_t = [0, 0x10, 0x400, 0x410, 0x20, 0x30, 0x420, 0x430];
    let b_n = [
        0x1000, 0x1400, 0x1010, 0x1410, 0x1020, 0x1420, 0x1030, 0x1430,
    ];
    let mut want: Vec<u32> = Vec::new();
    want.extend(a_t);
    want.extend(b_n);
    want.extend(a_t.iter().map(|o| o + 0x800));
    want.extend(b_n.iter().map(|o| o + 0x800));
    assert_eq!(lds, want);
    // op(A) no TA, op(B) TB: one ldsm per fragment
    assert_eq!(
        ldsm,
        [0, 0x400, 0x1000, 0x1400, 0x20, 0x420, 0x1020, 0x1420]
    );
    // the lane formulas, as NAK wrote them (shift / mask constants of the address arithmetic)
    for s in [
        "r26       = s2r sr[0x0]",
        "r36       = shf.l.w.i32 r36 rZ 0x7", // (lane >> 2) << 7 : lds rows 128 B
        "r36       = lop3.LUT[0xc0] r26 0xf rZ", // lane & 15 : ldsm A rows
        "r36       = shf.l.w.i32 r36 rZ 0x6", // ... << 6 : 64 B
        "r36       = lop3.LUT[0xea] r26 0x7 r36", // (lane & 7) | ((lane & 16) >> 1) : ldsm B rows
        "r38       = shf.l.w.i32 r38 rZ 0x1", // (lane & 8) << 1 : ldsm B second 16 B
    ] {
        assert!(LISTING_V6.contains(s), "listing changed: {s}");
    }
    // the same row offsets mod the two operands, block-uniform bases; every lds / ldsm row 16-byte aligned
    for l in 0..32 {
        assert_eq!(ldsm_a_lane(64)(l) % 16, 0);
        assert_eq!(ldsm_b_lane(64)(l) % 16, 0);
        assert_eq!(lds_lane(128)(l) % 4, 0);
    }
    // v5: one path for every flag — 16 lds (op(B) row-major) + 4 ldsm (op(A) row-major), as v6 without flags
    let v5 = loop_loads(LISTING_V5);
    let v5_lds: Vec<u32> = v5.iter().filter(|x| !x.0).map(|x| x.1).collect();
    let v5_ldsm: Vec<u32> = v5.iter().filter(|x| x.0).map(|x| x.1).collect();
    assert_eq!(v5_lds.len(), 16);
    assert!(v5_lds.iter().all(|o| (B_BASE..2 * B_BASE).contains(o)));
    assert_eq!(v5_ldsm, [0, 0x400, 0x20, 0x420]);
}

#[test]
fn t279_dense_reads_conflict() {
    // one request each
    assert_eq!(lds_wf(lds_lane(128)), 8); // 8 rows 128 B apart: the same 4 banks, 8-way
    assert_eq!(ldsm_wf(ldsm_a_lane(64)), 16); // 64 B rows: 2 of 8 16-byte bank groups, 4-way per matrix
    assert_eq!(ldsm_wf(ldsm_b_lane(64)), 16);
    // without conflict an lds is 1, an ldsm.x4 is 4
    // per warp per slice (op(A), op(B))
    assert_eq!(reads_per_warp(DENSE, false, false), (64, 128)); // dX of #513 (no flag) = v5 on every flag
    assert_eq!(reads_per_warp(DENSE, false, true), (64, 64)); // forward (TB)
    assert_eq!(reads_per_warp(DENSE, true, false), (128, 128)); // dW (TA)
    assert_eq!(reads_per_warp(DENSE, true, true), (128, 64));
    // per block (4 warps) per slice
    let rb = |ta, tb| reads_per_block(DENSE, ta, tb);
    assert_eq!(
        (rb(false, true), rb(false, false), rb(true, false)),
        (512, 768, 1_024)
    );
}

#[test]
fn t279_reads_plus_writes_per_block_slice() {
    // writes (T275, flag order none / TB / TA): v5 256 / 640 / 1 152, v6 64 on every flag
    let v5_w = [256u32, 640, 1_152];
    let v6_w = 64u32;
    let v5_r = reads_per_block(DENSE, false, false); // v5 reads the same on every flag
    assert_eq!(v5_r, 768);
    let v6_r = [
        reads_per_block(DENSE, false, false),
        reads_per_block(DENSE, false, true),
        reads_per_block(DENSE, true, false),
    ];
    let v5: Vec<u32> = v5_w.iter().map(|w| w + v5_r).collect();
    let v6: Vec<u32> = v6_r.iter().map(|r| r + v6_w).collect();
    assert_eq!(v5, [1_024, 1_408, 1_920]); // dX, forward (= the qkv-513 probe), dW
    assert_eq!(v6, [832, 576, 1_088]);
    // against the 2 108 SM clocks of one block-slice at the os-476 speed (T275) and the tensor floor:
    // 4 warps × 2 × 2 × 2 kk = 64 hmma.m16n8k16 (2 048 FMA each) at 128 FP16·FP32 FMA per clock per SM → 1 024
    let tensor = 64 * 2_048 / 128;
    assert_eq!(tensor, 1_024);
    // v5 qkv-513 (TB): the shared pipe 1 408 of 2 108 clocks (67 %), above the tensor floor
    assert_eq!(1_408 * 1_000 / 2_108, 667);
    // v6 on the dW (TA) case: the reads (1 024) now outweigh the writes (64) — the next cut is at the reads
    assert!(v6_r[2] > 15 * v6_w);
}

/// Store wavefronts of the v6 64-bit stores (half-warp groups) with row strides in halves, as in T275.
fn store_wf(op_a: bool, trans: bool, stride_halves: u32) -> u32 {
    let origin = |e: u32| -> (u32, u32) {
        match (op_a, trans) {
            (true, false) => (e / 8, 4 * (e % 8)),
            (true, true) => (4 * (e % 16), e / 16),
            (false, false) => (e / 16, 4 * (e % 16)),
            (false, true) => (4 * (e % 8), e / 8),
        }
    };
    let mut wf = 0;
    for warp in 0..4 {
        for s in 0..NV {
            for half in 0..2 {
                let a: Vec<u32> = (0..16)
                    .map(|l| {
                        let e = warp * 32 + half * 16 + l + s * THREADS;
                        let (x, y) = origin(e);
                        // A: (r, p); B: (p, c). Contiguous index along the vector, the other one × stride.
                        let base = match (op_a, trans) {
                            (true, false) => x * stride_halves + y,
                            (true, true) => y * stride_halves + x,
                            (false, false) => x * stride_halves + y,
                            (false, true) => y * stride_halves + x,
                        };
                        assert_eq!(base % 4, 0, "64-bit store must be 8-byte aligned");
                        2 * base
                    })
                    .collect();
                wf += wavefronts(&a, 8);
            }
        }
    }
    wf
}

/// Every element of the slice lands at a distinct address inside the array (padding columns never written).
fn covers_once(op_a: bool, trans: bool, stride_halves: u32, array_halves: u32) -> bool {
    let mut hit = vec![0u8; array_halves as usize];
    for e in 0..NV * THREADS {
        let (x, y) = match (op_a, trans) {
            (true, false) => (e / 8, 4 * (e % 8)),
            (true, true) => (4 * (e % 16), e / 16),
            (false, false) => (e / 16, 4 * (e % 16)),
            (false, true) => (4 * (e % 8), e / 8),
        };
        let base = match (op_a, trans) {
            (true, false) | (false, false) => x * stride_halves + y,
            (true, true) | (false, true) => y * stride_halves + x,
        };
        for q in 0..4 {
            let i = (base + q) as usize;
            if i >= hit.len() {
                return false;
            }
            hit[i] += 1;
        }
    }
    hit.iter().all(|&h| h <= 1) && hit.iter().filter(|&&h| h == 1).count() == (BM * BK) as usize
}

#[test]
fn t279_padded_rows_remove_the_read_conflicts() {
    // one request each, the padded strides
    assert_eq!(lds_wf(lds_lane(144)), 1);
    assert_eq!(ldsm_wf(ldsm_a_lane(80)), 4);
    assert_eq!(ldsm_wf(ldsm_b_lane(80)), 4);
    for (ta, tb) in [(false, false), (false, true), (true, false), (true, true)] {
        assert_eq!(reads_per_warp(PADDED, ta, tb), (16, 16), "ta {ta} tb {tb}");
        assert_eq!(reads_per_block(PADDED, ta, tb), 128);
    }
    // the 64-bit stores with the padded strides: the 128-byte layouts stay conflict-free (32 per operand per
    // block); the 64-byte layouts (A row-major, B column-major) go 2-way — a half-warp writes two 64-byte rows and
    // at 80 bytes apart the second one wraps onto the first one's banks. Only S ≡ 64 (mod 128) keeps those
    // stores free, and that is exactly the stride that makes ldsm 4-way: the trade is +32 store wavefronts for
    // −192 read wavefronts per operand per block.
    for (op_a, trans, stride, wf) in [
        (true, false, 40, 64),
        (true, true, 72, 32),
        (false, false, 72, 32),
        (false, true, 40, 64),
    ] {
        assert_eq!(store_wf(op_a, trans, stride), wf, "{op_a} {trans}");
    }
    // ... and the dense ones as T275 found (v6: 8 per operand per warp)
    assert_eq!(store_wf(true, true, 64), 32);
    assert_eq!(store_wf(false, true, 32), 32);
    // other strides tried on paper: 16 bytes is the smallest padding that frees both shapes
    assert_eq!(lds_wf(lds_lane(136)), 2); // 4 bytes of padding: 2-way left
    assert_eq!(ldsm_wf(ldsm_a_lane(72)), 8); // 8 bytes: ldsm rows must stay 16-byte aligned anyway
}

#[test]
fn t279_padded_layout_fits() {
    // array sizes: the larger of the two layouts per operand (halves)
    let a = (BM * PADDED.a_row / 2).max(BK * PADDED.a_col / 2);
    let b = (BK * PADDED.b_row / 2).max(BN * PADDED.b_col / 2);
    assert_eq!((a, b), (2_560, 2_560));
    for (op_a, trans, stride, arr) in [
        (true, false, 40, a),
        (true, true, 72, a),
        (false, false, 72, b),
        (false, true, 40, b),
    ] {
        assert!(covers_once(op_a, trans, stride, arr), "{op_a} {trans}");
    }
    // coopMatLoad strides in f16vec4 units, every one a multiple of 16 bytes (the os-467 rule)
    for s in [PADDED.a_row, PADDED.a_col, PADDED.b_row, PADDED.b_col] {
        assert_eq!(s % 16, 0);
    }
    assert_eq!(
        [
            PADDED.a_row / 8,
            PADDED.a_col / 8,
            PADDED.b_row / 8,
            PADDED.b_col / 8
        ],
        [10, 18, 18, 10]
    );
    // shared memory: 5 120 + 5 120 + 4 096 (st_) = 14 336 B; still 4 blocks per SM at the 64 KiB step
    let shared = 2 * (a + b) + 4 * 4 * 256;
    assert_eq!(shared, 14_336);
    let v7 = Kernel {
        name: "mm_tc_v7 (plan)",
        threads: 128,
        registers: 128,
        shared_bytes: shared,
    };
    for r in [Reserve::None, Reserve::OneKib] {
        assert_eq!(blocks_per_sm(&SM86, &v7, 64 << 10, r).unwrap().blocks, 4);
    }
    // at the 32 KiB step it would drop to 2 (v6: 2 as well) — the probe and the step run at 64 KiB
    assert_eq!(
        blocks_per_sm(&SM86, &v7, 32 << 10, Reserve::None)
            .unwrap()
            .blocks,
        2
    );
}

#[test]
fn t279_the_predicted_bands() {
    // shared pipe per block-slice (reads + the 64-bit stores of both operands), flag order none / TB / TA
    let v6 = [832u32, 576, 1_088];
    let v7 = [
        reads_per_block(PADDED, false, false)
            + store_wf(true, false, 40)
            + store_wf(false, false, 72),
        reads_per_block(PADDED, false, true)
            + store_wf(true, false, 40)
            + store_wf(false, true, 40),
        reads_per_block(PADDED, true, false)
            + store_wf(true, true, 72)
            + store_wf(false, false, 72),
    ];
    assert_eq!(v7, [224, 256, 192]);
    // two bounding models of a block-slice (SM clocks), calibrated on os-476 (v5, TB, 2 108 clocks, pipe 1 408):
    //   overlap  T = max(2 108, pipe)            → no gain from any pipe cut;
    //   additive T = max(1 024, 700 + pipe)      → the shared pipe in series with the rest, the tensor floor below.
    let add = |pipe: u32| (700 + pipe).max(1_024);
    assert_eq!(add(1_408), 2_108);
    assert!(v7.iter().all(|&p| add(p) == 1_024)); // v7 would sit on the tensor floor in this model
    // the ceiling of v7 against v6 on each case (additive), thousandths
    let ceil: Vec<u32> = v6.iter().map(|&p| add(p) * 1_000 / 1_024).collect();
    assert_eq!(ceil, [1_496, 1_246, 1_746]);
    // and against v5 (pipe 1 024 / 1 408 / 1 920): ×1.69 / ×2.06 / ×2.55 at most
    let v5 = [1_024u32, 1_408, 1_920];
    let ceil5: Vec<u32> = v5.iter().map(|&p| add(p) * 1_000 / 1_024).collect();
    assert_eq!(ceil5, [1_683, 2_058, 2_558]);
    // written bands (doc §4): v7 / v6 on the probe sets — TB (qkv-513) ×1.00–1.20, none (dx) ×1.05–1.40,
    // TA (dw) ×1.10–1.60; all inside [1.000, ceiling]
    for (lo, hi, c) in [
        (1_000, 1_200, ceil[1]),
        (1_050, 1_400, ceil[0]),
        (1_100, 1_600, ceil[2]),
    ] {
        assert!(lo >= 1_000 && hi <= c);
    }
}
