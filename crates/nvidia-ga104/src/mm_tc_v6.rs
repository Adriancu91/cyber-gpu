//! C-157 p.2 after os-476 (`docs/C-157_MM_TC_V6_PLAN.md`, session 263): [`crate::mm_tc_v5`] with the f16 slices
//! written to shared memory **four values per 64-bit store** (`tools/nak/mm_tc_v6.comp`, compiled by the **second**
//! NAK tool, Mesa 26.3.1, for SM 8.6).
//!
//! In `mm_tc_v5` every thread writes 32 separate 16-bit values per slice; on the transposed operands (dX, dW: two
//! thirds of the #513 step) they hit one bank 8 and 16 times (counted in T275). Here a vector is one store; op(A) is
//! kept column-major when A is transposed and op(B) column-major when B is, and `coopMatLoad` reads them
//! `ColumnMajor`, so the tensor cores get the same matrices. Global loads, slices, the `coopMatMulAdd` chain and the
//! ending are `mm_tc_v5`'s.
//!
//! **Same job, same constant buffer ([`tc::Args::cbuf`]), same launch (128 threads, 64×64 tiles,
//! [`tc::Args::grid`]), same alignment rule ([`fits`] = [`crate::mm_tc_v4::fits`]), the same 12 KiB and the same
//! bits** as `mm_tc_v5`; only the program differs. The CPU model is `gpu_sim::cmdproc::mm_tc_v6_on_cpu`. Tested
//! **only in SIMULARE** (T276).

use crate::mm_tc as tc;
use crate::mm_tc_v4 as v4;
use crate::mm_tc_v5 as v5;
use crate::qmd::SharedMemory;

/// `tools/nak/out2/mm_tc_v6.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const MM_TC_V6: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v6.sm86.sass");
/// SHA-256 of [`MM_TC_V6`] (`sh tools/nak/build2.sh mm_tc_v6`; checked by a test, written in the note).
pub const MM_TC_V6_SHA256: &str =
    "8c761fa50321ee891f510a6b70ffac88778312be6b399747411cbfa637df980f";
/// Registers per thread (NAK dump `mm_tc_v6.sm86.txt`).
pub const REGISTERS: u32 = 128;
/// Threads per block: 128 × 1 × 1, as `mm_tc`.
pub const BLOCK: [u32; 3] = tc::BLOCK;
/// Rows and columns of C per block, as `mm_tc`.
pub const BLOCK_MN: u32 = tc::BLOCK_MN;
/// Shared memory: exactly [`crate::mm_tc_v5::SHARED`] (the same f16 slices, held as `f16vec4`, and the same scratch):
/// 12 KiB, one barrier.
pub const SHARED: SharedMemory = v5::SHARED;
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// May `mm_tc_v6` run this product? Exactly [`crate::mm_tc_v4::fits`] (the same 128-bit loads). `false` → launch
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
        assert_eq!(MM_TC_V6.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_TC_V6)), MM_TC_V6_SHA256);
    }

    #[test]
    fn four_blocks_per_sm_at_64_kib() {
        assert_eq!(SHARED.bytes, 12_288);
        let k = Kernel {
            name: "mm_tc_v6",
            threads: BLOCK[0],
            registers: REGISTERS,
            shared_bytes: SHARED.bytes,
        };
        for r in [Reserve::None, Reserve::OneKib] {
            assert_eq!(blocks_per_sm(&SM86, &k, 64 << 10, r).unwrap().blocks, 4);
        }
    }
}
