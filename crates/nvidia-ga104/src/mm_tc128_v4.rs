//! C-157 p.2 (`docs/C-157_P2_MM_TC128_V4.md`, session 231): [`crate::mm_tc128`] with the cure of
//! [`crate::mm_tc_v4`] — the next K-slice read ahead with **128-bit** global loads (`tools/nak/mm_tc128_v4.comp`,
//! compiled by the **second** NAK tool, Mesa 26.3.1, for SM 8.6).
//!
//! **Only for aligned products** ([`fits`], the same rule as [`crate::mm_tc_v4::fits`]): a misaligned 128-bit load is a
//! fault on the board, so the host launches `mm_tc128` when [`fits`] says no.
//!
//! **Same job, same constant buffer ([`tc::Args::cbuf`], push constants at 0x700), same launch (256 threads, 128×128
//! tiles, [`grid`]), same shared memory and the same arithmetic** as `mm_tc128` (and so the same bits as `mm_tc`). The
//! CPU model is `gpu_sim::cmdproc::mm_tc128_v4_on_cpu` (refuses unaligned products, otherwise `mm_tc128`'s). Tested
//! **only in SIMULARE** (T237).

use crate::mm_tc as tc;
use crate::mm_tc128 as t128;
use crate::qmd::SharedMemory;

/// `tools/nak/out2/mm_tc128_v4.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const MM_TC128_V4: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc128_v4.sm86.sass");
/// SHA-256 of [`MM_TC128_V4`] (`sh tools/nak/build2.sh mm_tc128_v4`; checked by a test, written in the note).
pub const MM_TC128_V4_SHA256: &str =
    "8bd6361b65a2975832135fc81d2db402b76fbb3736dce45c60ae817e93352d93";
/// Registers per thread (NAK dump `mm_tc128_v4.sm86.txt`).
pub const REGISTERS: u32 = 168;
/// Threads per block: 256 × 1 × 1, as `mm_tc128`.
pub const BLOCK: [u32; 3] = t128::BLOCK;
/// Rows and columns of C per block, as `mm_tc128`.
pub const BLOCK_MN: u32 = t128::BLOCK_MN;
/// Shared memory: exactly `mm_tc128`'s (24 KiB, one barrier).
pub const SHARED: SharedMemory = t128::SHARED;
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// May `mm_tc128_v4` run this product? The same rule as [`crate::mm_tc_v4::fits`]: the stored row length of A and of
/// B a multiple of 4, the addresses of A and B multiples of 16 bytes. `false` → launch `mm_tc128` instead.
pub fn fits(args: &tc::Args) -> bool {
    crate::mm_tc_v4::fits(args)
}

/// The grid for one product: `mm_tc128`'s (one 256-thread block per 128×128 piece of C).
pub fn grid(args: &tc::Args) -> [u32; 3] {
    t128::grid(args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::digest as sha256;

    fn hex(d: &[u8]) -> alloc::string::String {
        d.iter().map(|b| alloc::format!("{b:02x}")).collect()
    }

    #[test]
    fn program_is_the_compiled_one() {
        assert_eq!(MM_TC128_V4.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_TC128_V4)), MM_TC128_V4_SHA256);
    }

    #[test]
    fn launch_and_alignment_are_mm_tc128_and_mm_tc_v4() {
        let a = tc::Args {
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            bias: 0,
            m: 1024,
            n: 2304,
            k: 768,
            flags: tc::TB,
        };
        assert_eq!(grid(&a), [18, 8, 1]);
        assert!(fits(&a));
        assert!(!fits(&tc::Args { k: 766, ..a }));
        assert!(!fits(&tc::Args { a: 0x1004, ..a }));
        assert_eq!(SHARED.bytes, t128::SHARED.bytes);
        const { assert!(BLOCK[0] * REGISTERS <= 65_536) };
    }
}
