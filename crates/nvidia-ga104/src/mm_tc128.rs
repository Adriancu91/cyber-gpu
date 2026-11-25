//! C-165 after os-416 (`docs/C-165_CURSA_MM.md` §„`mm_tc128`”, session 220): [`crate::mm_tc`] on 128×128 tiles
//! (`tools/nak/mm_tc128.comp`, compiled by the **second** NAK tool, Mesa 26.3.1, for SM 8.6) — the tile that won the
//! f32 race on all 15 shapes of step #513, applied to the tensor-core matmul.
//!
//! **Same job, same constant buffer ([`tc::Args::cbuf`], push constants at 0x700) and the same arithmetic** as
//! `mm_tc`: op(A), op(B) rounded to f16 when copied to shared memory, every element of C summed in f32 by the tensor
//! cores from 0 with K in steps of 16 in order, then one f32 add of `bias[col]`. Only the launch changes: blocks of
//! 256 threads (8 subgroups), one 128×128 piece of C per block, 24 KiB of shared memory (the f16 slices + one 16×16
//! f32 scratch per subgroup). The CPU model is `gpu_sim::cmdproc::mm_tc128_on_cpu` (= `mm_tc`'s, with the bigger
//! tile). Tested **only in SIMULARE** (T226); it goes to the board only in a race next to `mm_tc`, after `mm_tc` has
//! passed once alone (os-352 / os-384).

use crate::mm_tc as tc;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out2/mm_tc128.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const MM_TC128: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc128.sm86.sass");
/// SHA-256 of [`MM_TC128`] (`sh tools/nak/build2.sh mm_tc128`; checked by a test, written in the note).
pub const MM_TC128_SHA256: &str =
    "df5041bd4feccb0dcb2c5cff7bfb278548eefaaa180eafec9917a008e4268f79";
/// Registers per thread (NAK dump `mm_tc128.sm86.txt`).
pub const REGISTERS: u32 = 128;
/// Threads per block: 256 × 1 × 1 (8 subgroups of 32).
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Rows and columns of C per block.
pub const BLOCK_MN: u32 = 128;
/// Shared memory: the f16 slices of op(A) (128×32) and op(B) (32×128) + 8 scratches of 16×16 f32; one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: (128 * 32 + 32 * 128) * 2 + 8 * 16 * 16 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// The grid for one product (the same [`tc::Args`] as `mm_tc`): one 256-thread block per 128×128 piece of C.
pub fn grid(args: &tc::Args) -> [u32; 3] {
    [args.n.div_ceil(BLOCK_MN), args.m.div_ceil(BLOCK_MN), 1]
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
        assert_eq!(MM_TC128.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_TC128)), MM_TC128_SHA256);
    }

    #[test]
    fn launch_shape() {
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
        assert_eq!(grid(&tc::Args { m: 129, n: 1, ..a }), [1, 2, 1]);
        assert_eq!(SHARED.bytes, tc::SHARED.bytes);
        const { assert!(SHARED.bytes <= MAX_SHARED_BYTES) };
        const { assert!(BLOCK[0] * REGISTERS <= 65_536) };
    }
}
