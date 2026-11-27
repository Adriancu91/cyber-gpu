//! C-157 p.2 after session 265 (`docs/C-157_CITIRI_CONFLICTE.md`): [`crate::mm_tc_v6`] with **16 bytes of padding
//! on every shared slice row** (`tools/nak/mm_tc_v7.comp`, compiled by the **second** NAK tool, Mesa 26.3.1, for
//! SM 8.6).
//!
//! In `mm_tc_v6` the slice rows are 64 or 128 bytes, so every `coopMatLoad` read conflicts (a 32-bit `lds` 8-way, an
//! `ldsm` matrix 4-way; counted in T279). Here the rows are 80 / 144 bytes: the reads are conflict-free on every flag,
//! the 64-bit stores into the 80-byte layouts 2-way. Global loads, the branch on `flags`, the conversions, the
//! `coopMatMulAdd` chain and the ending are `mm_tc_v6`'s.
//!
//! **Same job, same constant buffer ([`tc::Args::cbuf`]), same launch (128 threads, 64×64 tiles,
//! [`tc::Args::grid`]), same alignment rule ([`fits`] = [`crate::mm_tc_v4::fits`]) and the same bits** as
//! `mm_tc_v6`; the program and the shared memory (14 336 B instead of 12 288, still 4 blocks per SM at 64 KiB) differ.
//! The CPU model is `gpu_sim::cmdproc::mm_tc_v7_on_cpu`. Tested **only in SIMULARE** (T280).

use crate::mm_tc as tc;
use crate::mm_tc_v4 as v4;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out2/mm_tc_v7.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const MM_TC_V7: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v7.sm86.sass");
/// SHA-256 of [`MM_TC_V7`] (`sh tools/nak/build2.sh mm_tc_v7`; checked by a test, written in the note).
pub const MM_TC_V7_SHA256: &str =
    "953ed6b85fb0022183ef0e5b4b3262cd3f3742c43dcea2397ed9e2d285e1cf6a";
/// Registers per thread (NAK dump `mm_tc_v7.sm86.txt`).
pub const REGISTERS: u32 = 128;
/// Threads per block: 128 × 1 × 1, as `mm_tc`.
pub const BLOCK: [u32; 3] = tc::BLOCK;
/// Rows and columns of C per block, as `mm_tc`.
pub const BLOCK_MN: u32 = tc::BLOCK_MN;
/// Shared memory: the padded f16 slices of op(A) and op(B) (640 `f16vec4` each, the larger of the two layouts:
/// 64 rows of 80 B or 32 rows of 144 B) + one 16×16 f32 scratch per subgroup (4): 14 336 B, one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 640 * 8 + 640 * 8 + 4 * 16 * 16 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// May `mm_tc_v7` run this product? Exactly [`crate::mm_tc_v4::fits`] (the same 128-bit loads). `false` → launch
/// `mm_tc_pf` instead.
pub fn fits(args: &tc::Args) -> bool {
    v4::fits(args)
}

/// The grid for one product: `mm_tc`'s (one 128-thread block per 64×64 piece of C).
pub fn grid(args: &tc::Args) -> [u32; 3] {
    args.grid()
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
    fn program_is_the_compiled_one() {
        assert_eq!(MM_TC_V7.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_TC_V7)), MM_TC_V7_SHA256);
    }

    #[test]
    fn four_blocks_per_sm_at_64_kib() {
        assert_eq!(SHARED.bytes, 14_336);
        let k = Kernel {
            name: "mm_tc_v7",
            threads: BLOCK[0],
            registers: REGISTERS,
            shared_bytes: SHARED.bytes,
        };
        for r in [Reserve::None, Reserve::OneKib] {
            assert_eq!(blocks_per_sm(&SM86, &k, 64 << 10, r).unwrap().blocks, 4);
        }
    }
}
