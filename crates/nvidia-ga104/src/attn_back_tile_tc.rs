//! C-157 (`docs/C-157_ATTN_BACK_TC.md`, plan and predictions B1–B6 pushed before this code, `be02aaa`):
//! [`crate::attn_back_tile_3`]'s job (mode 1: dQ = dS·K, dK = dSᵀ·Q, dV = Pᵀ·dO) on the **tensor cores**
//! (`tools/nak/attn_back_tile_tc.comp`, compiled by the second NAK tool, Mesa 26.3.1, SM 8.6): every element of
//! dS, P, Q, K, dO rounded to binary16 in shared memory, the products added in f32 (`hmma.m16n8k16.f32`).
//!
//! **Same arguments and constant buffer** ([`crate::attn_back::Args::cbuf`] with [`crate::attn_back::Mode::Columns`]),
//! **same grid, block and result places** as `attn_back_tile_3`; only the program, its registers and its shared size
//! differ. The numbers are NOT the same bits as the F32 program: the CPU model is `gpu_compute::attn_tc::model`, the
//! verdict `gpu_compute::attn_tc::judge` (the `mm_tc` margin, per head). The mixed #513 step already carries the error
//! scale 2¹⁴ in dO / dS. Tested **only in SIMULARE** (T252). Nothing here has run on the board.

use crate::attn_back::Args;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out2/attn_back_tile_tc.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const ATTN_BACK_TILE_TC: &[u8] =
    include_bytes!("../../../tools/nak/out2/attn_back_tile_tc.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE_TC`] (`sh tools/nak/build2.sh attn_back_tile_tc`; checked by a test, in the note).
pub const ATTN_BACK_TILE_TC_SHA256: &str =
    "527d8dd1dc5000c26541561e8c60f59f4b5dab0d034a4d5ee593ae415aed5c65";
/// Registers per thread (NAK dump `attn_back_tile_tc.sm86.txt`: `Num GPRs: 48`).
pub const REGISTERS: u32 = 48;
/// Threads per block: 256, as `attn_back_tile_3`.
pub const BLOCK: [u32; 3] = crate::attn_back_tile::BLOCK;
/// The largest head width.
pub const MAX_D: u32 = crate::attn_back_tile::MAX_D;
/// Shared memory: two f16 tiles 64 × 64 (16 KiB) and the f32 result tile 64 × 64 (16 KiB) = 32 768 bytes
/// (the dump's `Shared size`); one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 2 * 64 * 64 * 2 + 64 * 64 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads (the same as `attn_back`).
pub const CBUF_BYTES: usize = crate::attn_back::CBUF_BYTES;

/// The grid: [⌈T/64⌉, H, B] (as `attn_back_tile_3`).
pub fn grid(a: &Args) -> [u32; 3] {
    crate::attn_back_tile::grid(a)
}

/// Whether the program takes this launch (as `attn_back_tile_3`: mode 1, d = C / H ≤ 64).
pub fn shape_ok(a: &Args) -> bool {
    crate::attn_back_tile::shape_ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::digest as sha256;

    const DUMP: &str = include_str!("../../../tools/nak/out2/attn_back_tile_tc.sm86.txt");

    fn hex(b: &[u8]) -> alloc::string::String {
        use core::fmt::Write;
        let mut s = alloc::string::String::new();
        for x in b {
            let _ = write!(s, "{x:02x}");
        }
        s
    }

    #[test]
    fn bytes_match_the_written_checksum() {
        assert_eq!(hex(&sha256(ATTN_BACK_TILE_TC)), ATTN_BACK_TILE_TC_SHA256);
        assert_eq!(ATTN_BACK_TILE_TC.len() % 16, 0);
    }

    /// B5 (docs/C-157_ATTN_BACK_TC.md): 0 local memory, ≤ 64 registers, tensor-core products, no `ffma` at all.
    #[test]
    fn dump_says_what_the_note_says() {
        assert!(DUMP.contains(&alloc::format!("Num GPRs: {REGISTERS}\n")));
        assert!(DUMP.contains("SLM size: 0\n"));
        assert!(DUMP.contains(&alloc::format!("Shared size: {}\n", SHARED.bytes)));
        assert!(DUMP.contains("Spills to mem: 0\n"));
        assert_eq!(DUMP.matches("hmma.m16n8k16.f32").count(), 4);
        assert_eq!(DUMP.matches("= ffma").count(), 0);
        const { assert!(REGISTERS <= 64) };
    }
}
