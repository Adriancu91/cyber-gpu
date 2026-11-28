//! C-168 point 3 (`docs/C-168_V8_PLAN.md`, session 279): the training matmul on the tensor cores, **version 8** —
//! two programs from one source (`tools/nak/mm_tc_v8.comp`, compiled by the **second** NAK tool, Mesa 26.3.1, SM 8.6):
//!
//! - [`MM_TC_V8`]: 128×128 blocks of C, 256 threads (8 warps of 64×32), 24 576 B of shared memory → 2 blocks per SM;
//! - [`MM_TC_V8N`]: 128×64 blocks, 128 threads (4 warps), 18 432 B → 3 blocks per SM (for the forms where 128×128
//!   would leave the last wave of blocks a quarter empty, T294).
//!
//! In shared memory op(A) is always kept by rows and op(B) by columns (the turn done in registers), so every fragment
//! is read with `ldsm`; two stages, one barrier per K slice of 16; 16×8 accumulators (one `hmma.m16n8k16` each).
//! **Same job, same constant buffer ([`tc::Args::cbuf`], push constants at 0x700), same alignment rule ([`fits`] =
//! [`crate::mm_tc_v4::fits`]) and the same bits as `mm_tc`** (the same f16 values through the same `hmma` chain, K in
//! steps of 16 in order, one f32 bias add). The CPU models are `gpu_sim::cmdproc::mm_tc_v8_on_cpu` /
//! `mm_tc_v8n_on_cpu`. Tested **only in SIMULARE** (T295); nothing here has run on a board.

use crate::mm_tc as tc;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out2/mm_tc_v8.sm86.sass` (Mesa NAK 26.3.1, SM 8.6): 128×128 blocks.
pub const MM_TC_V8: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v8.sm86.sass");
/// SHA-256 of [`MM_TC_V8`] (`sh tools/nak/build2.sh mm_tc_v8`; checked by a test, written in the note).
pub const MM_TC_V8_SHA256: &str =
    "1b910a5392cb5f9a67ee3ad340ef7dea94e5a56630ac77bcf0903d8b0f8c4b29";
/// `tools/nak/out2/mm_tc_v8n.sm86.sass` (the same source with `-DV8N`): 128×64 blocks.
pub const MM_TC_V8N: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v8n.sm86.sass");
/// SHA-256 of [`MM_TC_V8N`] (`sh tools/nak/build2.sh mm_tc_v8n`).
pub const MM_TC_V8N_SHA256: &str =
    "9a0702bccfe2d82e699a99ee22096fb1a8b8844a130b559aae136f9c7ef20aae";

/// Registers per thread of [`MM_TC_V8`] (NAK dump `mm_tc_v8.sm86.txt`).
pub const REGISTERS: u32 = 128;
/// Registers per thread of [`MM_TC_V8N`] (NAK dump `mm_tc_v8n.sm86.txt`).
pub const REGISTERS_N: u32 = 144;
/// Threads per block of [`MM_TC_V8`].
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Threads per block of [`MM_TC_V8N`].
pub const BLOCK_N: [u32; 3] = [128, 1, 1];
/// Rows of C per block (both programs).
pub const BLOCK_M: u32 = 128;
/// Columns of C per block of [`MM_TC_V8`].
pub const BLOCK_NCOLS: u32 = 128;
/// Columns of C per block of [`MM_TC_V8N`].
pub const BLOCK_NCOLS_N: u32 = 64;
/// Shared memory of [`MM_TC_V8`]: two stages of (128 + 128) rows × 48 B (16 f16 + 8 of padding); the per-warp output
/// scratch reuses them. One barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 2 * (128 + 128) * 48,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Shared memory of [`MM_TC_V8N`]: two stages of (128 + 64) rows × 48 B. One barrier.
pub const SHARED_N: SharedMemory = SharedMemory {
    bytes: 2 * (128 + 64) * 48,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Blocks per SM of [`MM_TC_V8`] at 64 KiB of shared memory (registers: 2 × 256 × 128 = 65 536).
pub const BLOCKS_PER_SM: u32 = 2;
/// Blocks per SM of [`MM_TC_V8N`] at 64 KiB (registers: 3 × 128 × 144 = 55 296; shared 3 × (18 432 + 1 024)).
pub const BLOCKS_PER_SM_N: u32 = 3;
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// May `mm_tc_v8` / `mm_tc_v8n` run this product? Exactly [`crate::mm_tc_v4::fits`] (the same 128-bit loads).
/// `false` → launch `mm_tc_pf` instead.
pub fn fits(args: &tc::Args) -> bool {
    crate::mm_tc_v4::fits(args)
}

/// The grid of [`MM_TC_V8`]: one 256-thread block per 128×128 piece of C.
pub fn grid(args: &tc::Args) -> [u32; 3] {
    [args.n.div_ceil(BLOCK_NCOLS), args.m.div_ceil(BLOCK_M), 1]
}

/// The grid of [`MM_TC_V8N`]: one 128-thread block per 128×64 piece of C (64 columns).
pub fn grid_n(args: &tc::Args) -> [u32; 3] {
    [args.n.div_ceil(BLOCK_NCOLS_N), args.m.div_ceil(BLOCK_M), 1]
}

/// C-168 p.3 G7 (session 280): which program one aligned product takes in the step — `mm_tc_v7` (64×64 blocks, 4 per
/// SM at 64 KiB), [`MM_TC_V8`] or [`MM_TC_V8N`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    V7,
    V8,
    V8n,
}

/// Blocks per SM of `mm_tc_v7` at 64 KiB (os-486 / os-490, T294) — the third candidate of [`pick`].
pub const BLOCKS_PER_SM_V7: u32 = 4;

/// The rates [`pick`] uses until the board measures them: per SM at full waves, in percent of `mm_tc_v7`'s rate.
/// **DEDUS** (`docs/C-168_V8_PLAN.md` §4, the middle of the band f = 1,25–1,60; v8n = 0,9 · f). To be replaced by the
/// rates of the probe `--mm-tc-v8` (G5 / G6) before the step uses the picker on the board.
pub const RATES_PLAN: Rates = Rates {
    v8_pct: 140,
    v8n_pct: 126,
};

/// The rates **measured on the board** (os-500, 28 Nov 2026, probe `--mm-tc-v8` at 64 KiB, medians of 5 launches):
/// per SM at full waves, in percent of `mm_tc_v7`'s, on the two probe shapes where all three programs fill their waves
/// exactly (qkv-513 and qkv-513-dx): v8 112,7 / 110,6 %, v8n 118,4 / 123,4 %. Taken as the lower of the two, rounded
/// down (T300, `docs/C-168_OS500_V8.md`). The `pick` with these rates puts no product on v8 at lot 6 (v8n is faster
/// per SM and never fills its waves worse on the step's shapes).
pub const RATES_OS500: Rates = Rates {
    v8_pct: 110,
    v8n_pct: 118,
};

/// Rates per SM at full waves, in percent of `mm_tc_v7`'s (`mm_tc_v7` = 100). Both non-zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rates {
    pub v8_pct: u32,
    pub v8n_pct: u32,
}

/// The pick by waves (`docs/C-168_V8_PLAN.md` §3, the model of T294): a program's time on a product ∝
/// waves · slots / (blocks · rate), with slots = `sms` · blocks per SM — the work is the same for all three, the last
/// wave half empty is paid in full. The cheapest wins; on a tie `mm_tc_v7` (measured on the board), then v8, then v8n.
/// `with_v7 = false` (v7 not loaded) leaves only v8 / v8n. Integers only (cross-multiplied), so the pick is exact.
/// The caller checks [`fits`] first; `sms` and the rates must be non-zero.
pub fn pick(args: &tc::Args, sms: u32, rates: Rates, with_v7: bool) -> Pick {
    // (pick, rows per block, columns per block, blocks per SM, rate)
    let all = [
        (Pick::V7, 64, 64, BLOCKS_PER_SM_V7, 100),
        (Pick::V8, BLOCK_M, BLOCK_NCOLS, BLOCKS_PER_SM, rates.v8_pct),
        (
            Pick::V8n,
            BLOCK_M,
            BLOCK_NCOLS_N,
            BLOCKS_PER_SM_N,
            rates.v8n_pct,
        ),
    ];
    // cost = (waves · slots) / (blocks · rate) as a fraction (numerator, denominator).
    let cost = |&(_, bm, bn, per_sm, rate): &(Pick, u32, u32, u32, u32)| -> (u128, u128) {
        let blocks = u128::from(args.m.div_ceil(bm)) * u128::from(args.n.div_ceil(bn));
        let slots = u128::from(sms) * u128::from(per_sm);
        let waves = blocks.div_ceil(slots.max(1));
        (waves * slots, blocks.max(1) * u128::from(rate.max(1)))
    };
    let mut best: Option<(Pick, (u128, u128))> = None;
    for c in all.iter().filter(|c| with_v7 || c.0 != Pick::V7) {
        let k = cost(c);
        best = match best {
            Some((p, b)) if b.0 * k.1 <= k.0 * b.1 => Some((p, b)),
            _ => Some((c.0, k)),
        };
    }
    best.map_or(Pick::V8, |b| b.0)
}

/// C-168 p.3 after os-503 (session 288, `docs/C-168_OS503_G7.md`): the depth of the sum (`k`) from which a product
/// goes back to `mm_tc_v7` in [`pick_long_k`]. In the #513 step only the head's dX has such a sum (k = 16 384, the
/// vocabulary); every other product has k ≤ 3 072. On the board (os-503, lot 6) that one launch took **12,78 ms on
/// v8n** (6,0 TFLOP/s; the wave model: ~3,5 ms), while on v7 in os-496 it was below 7,87 ms (outside the 8 longest
/// labels). Why v8n is slow there is not known (DEDUS candidates in the note); the guard only avoids it.
pub const LONG_K_OS503: u32 = 8192;

/// [`pick`], except that a product with `k ≥ long_k` takes `mm_tc_v7` when it is loaded (`with_v7`). `long_k = 0`
/// = no guard = [`pick`] exactly. The bits do not change (v7, v8 and v8n give `mm_tc`'s bits, os-500 / T297 N3).
pub fn pick_long_k(args: &tc::Args, sms: u32, rates: Rates, with_v7: bool, long_k: u32) -> Pick {
    if with_v7 && long_k != 0 && args.k >= long_k {
        Pick::V7
    } else {
        pick(args, sms, rates, with_v7)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
    use crate::sha256::digest as sha256;

    fn hex(d: &[u8]) -> alloc::string::String {
        d.iter().map(|b| alloc::format!("{b:02x}")).collect()
    }

    #[test]
    fn programs_are_the_compiled_ones() {
        assert_eq!(MM_TC_V8.len() % 16, 0);
        assert_eq!(MM_TC_V8N.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_TC_V8)), MM_TC_V8_SHA256);
        assert_eq!(hex(&sha256(MM_TC_V8N)), MM_TC_V8N_SHA256);
    }

    #[test]
    fn blocks_per_sm_at_64_kib() {
        for (threads, regs, shared, want) in [
            (BLOCK[0], REGISTERS, SHARED.bytes, BLOCKS_PER_SM),
            (BLOCK_N[0], REGISTERS_N, SHARED_N.bytes, BLOCKS_PER_SM_N),
        ] {
            let k = Kernel {
                name: "mm_tc_v8",
                threads,
                registers: regs,
                shared_bytes: shared,
            };
            for r in [Reserve::None, Reserve::OneKib] {
                assert_eq!(blocks_per_sm(&SM86, &k, 64 << 10, r).unwrap().blocks, want);
            }
        }
    }

    #[test]
    fn grids_cover_c() {
        let a = tc::Args {
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            bias: 0,
            m: 6144,
            n: 2304,
            k: 768,
            flags: tc::TB,
        };
        assert_eq!(grid(&a), [18, 48, 1]);
        assert_eq!(grid_n(&a), [36, 48, 1]);
        assert!(fits(&a));
        assert!(!fits(&tc::Args { k: 766, ..a }));
    }

    /// G7 (DEDUS): the pick by waves on the lot-6 forms of step #513 (48 SM) at the plan's rates (f = 1,4) gives the
    /// split written in `docs/C-168_V8_PLAN.md` §4: out-dW (768×768) on v7, qkv (exact tie, v8 first) / 3072×3072 /
    /// the logits on v8, every form with 768 on v8n.
    #[test]
    fn pick_by_waves_lot6() {
        let a = |m, n| tc::Args {
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            bias: 0,
            m,
            n,
            k: 768,
            flags: 0,
        };
        let p = |m, n| pick(&a(m, n), 48, RATES_PLAN, true);
        assert_eq!(p(3072, 2304), Pick::V8);
        assert_eq!(p(3072, 3072), Pick::V8);
        assert_eq!(p(3072, 16_384), Pick::V8);
        assert_eq!(p(3072, 768), Pick::V8n);
        assert_eq!(p(768, 3072), Pick::V8n);
        assert_eq!(p(768, 2304), Pick::V8n);
        assert_eq!(p(768, 768), Pick::V7);
        // Without v7 loaded: v8 / v8n only.
        assert_eq!(pick(&a(768, 768), 48, RATES_PLAN, false), Pick::V8n);
        // At v7's rate (no gain per SM) v7 keeps every form where its waves are no worse.
        let same = Rates {
            v8_pct: 100,
            v8n_pct: 100,
        };
        assert_eq!(pick(&a(3072, 768), 48, same, true), Pick::V7);
        assert_eq!(pick(&a(3072, 3072), 48, same, true), Pick::V7);
    }
}
