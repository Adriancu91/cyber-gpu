//! C-157 (`docs/C-157_ATTN_BACK_TC.md` §6, session 244): [`crate::attn_back_tile_tc`] in **16 KiB** of shared memory
//! instead of 32 (`tools/nak/attn_back_tile_tc16.comp`, Mesa NAK 26.3.1, SM 8.6): the f16 tiles sa / sb packed two
//! per 32-bit word and the f32 result tile laid over both after a barrier. **The same arithmetic** (same tiles, same
//! f16 rounding — RNE, `f2fp.pack_ab` — same tensor-core order, same guards), so the CPU model is the same
//! `gpu_compute::attn_tc::model` and the board test is "the same words as `attn_back_tile_tc`". The same arguments,
//! cbuf ([`crate::attn_back_tile_tc::cbuf`], arguments at 0x700), grid and block. Why: at the 64 KiB step of the fast
//! mixed step 3–4 blocks fit on an SM instead of 1–2 (DEDUS, `occupancy`). Tested **only in SIMULARE**.

use crate::attn_back::Args;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out2/attn_back_tile_tc16.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const ATTN_BACK_TILE_TC16: &[u8] =
    include_bytes!("../../../tools/nak/out2/attn_back_tile_tc16.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE_TC16`] (`sh tools/nak/build2.sh attn_back_tile_tc16`; checked by a test, in the note).
pub const ATTN_BACK_TILE_TC16_SHA256: &str =
    "8cbbab7ed17988f5c6030fbb3ffcee7163f24a47a670a9ae4ae91b7be8ee5286";
/// Registers per thread (NAK dump `attn_back_tile_tc16.sm86.txt`: `Num GPRs: 48`).
pub const REGISTERS: u32 = 48;
/// Threads per block: 256, as `attn_back_tile_tc`.
pub const BLOCK: [u32; 3] = crate::attn_back_tile_tc::BLOCK;
/// Shared memory: one array of 4 096 words (16 384 bytes, the dump's `Shared size`); one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 64 * 64 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Size of constant buffer 0 (as `attn_back_tile_tc`: arguments at 0x700).
pub const CBUF_BYTES: usize = crate::attn_back_tile_tc::CBUF_BYTES;

/// Constant buffer 0: [`crate::attn_back_tile_tc::cbuf`].
pub fn cbuf(a: &Args) -> Option<[u8; CBUF_BYTES]> {
    crate::attn_back_tile_tc::cbuf(a)
}

/// The grid: [⌈T/64⌉, H, B].
pub fn grid(a: &Args) -> [u32; 3] {
    crate::attn_back_tile_tc::grid(a)
}

/// Whether the program takes this launch (mode 1, d = C / H ≤ 64).
pub fn shape_ok(a: &Args) -> bool {
    crate::attn_back_tile_tc::shape_ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::digest as sha256;

    const DUMP: &str = include_str!("../../../tools/nak/out2/attn_back_tile_tc16.sm86.txt");
    const DUMP32: &str = include_str!("../../../tools/nak/out2/attn_back_tile_tc.sm86.txt");

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
        assert_eq!(
            hex(&sha256(ATTN_BACK_TILE_TC16)),
            ATTN_BACK_TILE_TC16_SHA256
        );
        assert_eq!(ATTN_BACK_TILE_TC16.len() % 16, 0);
        assert_ne!(
            ATTN_BACK_TILE_TC16,
            crate::attn_back_tile_tc::ATTN_BACK_TILE_TC
        );
    }

    /// D2 (docs/C-157_ATTN_BACK_TC.md §6): 16 KiB, ≤ 64 registers, 0 local memory, the same tensor-core and
    /// shared-matrix instructions as the 32 KiB program, RNE packing, no `ffma`; the arguments read at 0x700.
    #[test]
    fn dump_says_what_the_note_says() {
        assert!(DUMP.contains(&alloc::format!("Num GPRs: {REGISTERS}\n")));
        assert!(DUMP.contains("SLM size: 0\n"));
        assert!(DUMP.contains("Shared size: 16384\n"));
        assert!(DUMP.contains("Spills to mem: 0\n"));
        for op in ["hmma.m16n8k16.f32", "ldsm.16.m8n8.x4", "movm.16.m8n8.trans"] {
            assert_eq!(DUMP.matches(op).count(), DUMP32.matches(op).count(), "{op}");
        }
        assert_eq!(DUMP.matches("= ffma").count(), 0);
        // `f2fp.pack_ab` without a rounding suffix = round to nearest even (NAK prints any other mode).
        assert_eq!(DUMP.matches("= f2fp.pack_ab r").count(), 2);
        for part in DUMP.split("c[0x0][").skip(1) {
            let off = part.trim_start_matches('+').split(']').next().unwrap();
            let v = u32::from_str_radix(off.trim_start_matches("0x"), 16).unwrap();
            assert!(v <= 0x8 || (0x700..0x740).contains(&v), "citire la {v:#x}");
        }
        const { assert!(REGISTERS <= 64) };
    }
}
