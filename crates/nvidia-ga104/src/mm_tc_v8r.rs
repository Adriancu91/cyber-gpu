//! C-168 point 3 follow-up (`docs/C-168_V8R.md`, sessions 286–287): **`mm_tc_v8r`** = [`crate::mm_tc_v8::MM_TC_V8`] with
//! the fragment reads of a K slice reordered (the 4 B fragments first, then one A fragment at a time with its 4 `hmma`).
//! Source `tools/nak/mm_tc_v8r.comp`, compiled by the second NAK tool (Mesa 26.3.1, SM 8.6).
//!
//! Everything but the order of the reads inside a slice is v8's: 128×128 blocks of C, 256 threads, 24 576 B of shared
//! memory, one barrier, the same constant buffer ([`tc::Args::cbuf`], push constants at 0x700), the same alignment
//! rule ([`fits`]) and the same grid. Every accumulator still gets one `hmma` per slice on the same values in the same
//! order along K, so **the bits are v8's (= `mm_tc`'s)**. NAK gives it 120 registers instead of 128 (still 2 blocks per
//! SM at 64 KiB) and drops the swap through a scratch register that chained v8's 16 `hmma` (T302, counted on the
//! listing). The CPU model is `gpu_sim::cmdproc::mm_tc_v8r_on_cpu`. Tested **only in SIMULARE** (T303); nothing here
//! has run on a board — the speed is a prediction (V1–V4 in the note).

use crate::mm_tc as tc;
use crate::mm_tc_v8 as v8;
use crate::qmd::SharedMemory;

/// `tools/nak/out2/mm_tc_v8r.sm86.sass` (Mesa NAK 26.3.1, SM 8.6): 128×128 blocks, reads reordered.
pub const MM_TC_V8R: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v8r.sm86.sass");
/// SHA-256 of [`MM_TC_V8R`] (`sh tools/nak/build2.sh mm_tc_v8r`; identical in two builds; checked by a test).
pub const MM_TC_V8R_SHA256: &str =
    "6e28d531fc104ec1b37edb0f2303337c81eef66a4a82b5ad02861add18274bf4";

/// Registers per thread (NAK dump `mm_tc_v8r.sm86.txt`: `Num GPRs: 120`; v8 has 128). **Use this one in the QMD.**
pub const REGISTERS: u32 = 120;
/// Threads per block: as v8.
pub const BLOCK: [u32; 3] = v8::BLOCK;
/// Rows of C per block.
pub const BLOCK_M: u32 = v8::BLOCK_M;
/// Columns of C per block.
pub const BLOCK_NCOLS: u32 = v8::BLOCK_NCOLS;
/// Shared memory: as v8 (two stages of (128 + 128) rows × 48 B, one barrier).
pub const SHARED: SharedMemory = v8::SHARED;
/// Blocks per SM at 64 KiB of shared memory: 2 (registers 2 × 256 × 120 = 61 440; shared 2 × (24 576 + 1 024)).
pub const BLOCKS_PER_SM: u32 = 2;
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// May `mm_tc_v8r` run this product? Exactly [`crate::mm_tc_v8::fits`]. `false` → launch `mm_tc_pf` instead.
pub fn fits(args: &tc::Args) -> bool {
    v8::fits(args)
}

/// The grid: one 256-thread block per 128×128 piece of C (= [`crate::mm_tc_v8::grid`]).
pub fn grid(args: &tc::Args) -> [u32; 3] {
    v8::grid(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
    use crate::sha256::digest as sha256;

    #[test]
    fn program_is_the_compiled_one() {
        assert_eq!(MM_TC_V8R.len() % 16, 0);
        let hex: alloc::string::String = sha256(MM_TC_V8R)
            .iter()
            .map(|b| alloc::format!("{b:02x}"))
            .collect();
        assert_eq!(hex, MM_TC_V8R_SHA256);
        assert_ne!(MM_TC_V8R, v8::MM_TC_V8);
    }

    #[test]
    fn two_blocks_per_sm_at_64_kib() {
        let k = Kernel {
            name: "mm_tc_v8r",
            threads: BLOCK[0],
            registers: REGISTERS,
            shared_bytes: SHARED.bytes,
        };
        for r in [Reserve::None, Reserve::OneKib] {
            assert_eq!(
                blocks_per_sm(&SM86, &k, 64 << 10, r).unwrap().blocks,
                BLOCKS_PER_SM
            );
        }
        assert_eq!((REGISTERS, v8::REGISTERS), (120, 128));
    }

    #[test]
    fn grid_and_fits_are_v8s() {
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
        assert!(fits(&a));
        assert!(!fits(&tc::Args { k: 766, ..a }));
    }
}
