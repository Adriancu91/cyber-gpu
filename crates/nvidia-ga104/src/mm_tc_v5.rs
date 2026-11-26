//! C-157 p.2 after os-468 (`docs/C-157_MM_TC_V5_PLAN.md`, session 258): [`crate::mm_tc_v4`] with a **12 KiB** block
//! (`tools/nak/mm_tc_v5.comp`, compiled by the **second** NAK tool, Mesa 26.3.1, for SM 8.6).
//!
//! `mm_tc_v4` keeps the finished 64×64 f32 block in shared memory (16 of its 24 KiB) only to write it out; at the
//! 64 KiB step 2 blocks fit on an SM. Here every subgroup writes its four 16×16 accumulators one at a time through its
//! own 16×16 f32 scratch (`mm_tc128`'s ending, bit for bit on the board in os-423): 4 + 4 + 4 KiB → 4 blocks per SM.
//! The K loop is `mm_tc_v4`'s, unchanged.
//!
//! **Same job, same constant buffer ([`tc::Args::cbuf`]), same launch (128 threads, 64×64 tiles,
//! [`tc::Args::grid`]), same alignment rule ([`fits`] = [`crate::mm_tc_v4::fits`]) and the same bits** as `mm_tc_v4`;
//! only the QMD's shared memory ([`SHARED`]) differs. The CPU model is `gpu_sim::cmdproc::mm_tc_v5_on_cpu`. Tested
//! **only in SIMULARE** (T269).

use crate::mm_tc as tc;
use crate::mm_tc_v4 as v4;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out2/mm_tc_v5.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const MM_TC_V5: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v5.sm86.sass");
/// SHA-256 of [`MM_TC_V5`] (`sh tools/nak/build2.sh mm_tc_v5`; checked by a test, written in the note).
pub const MM_TC_V5_SHA256: &str =
    "f34f4f934135b39d4eccb59ef969688864d4bf78938a32eaf23eca594aee23d6";
/// Registers per thread (NAK dump `mm_tc_v5.sm86.txt`).
pub const REGISTERS: u32 = 128;
/// Threads per block: 128 × 1 × 1, as `mm_tc`.
pub const BLOCK: [u32; 3] = tc::BLOCK;
/// Rows and columns of C per block, as `mm_tc`.
pub const BLOCK_MN: u32 = tc::BLOCK_MN;
/// Shared memory: the f16 slices of op(A) (64×32) and op(B) (32×64) + one 16×16 f32 scratch per subgroup (4):
/// 12 KiB, one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: (64 * 32 + 32 * 64) * 2 + 4 * 16 * 16 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// May `mm_tc_v5` run this product? Exactly [`crate::mm_tc_v4::fits`] (the same 128-bit loads). `false` → launch
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
        assert_eq!(MM_TC_V5.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_TC_V5)), MM_TC_V5_SHA256);
    }

    #[test]
    fn four_blocks_per_sm_at_64_kib() {
        assert_eq!(SHARED.bytes, 12_288);
        let k = Kernel {
            name: "mm_tc_v5",
            threads: BLOCK[0],
            registers: REGISTERS,
            shared_bytes: SHARED.bytes,
        };
        for r in [Reserve::None, Reserve::OneKib] {
            assert_eq!(blocks_per_sm(&SM86, &k, 64 << 10, r).unwrap().blocks, 4);
        }
    }
}
