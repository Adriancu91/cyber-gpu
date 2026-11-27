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
}
