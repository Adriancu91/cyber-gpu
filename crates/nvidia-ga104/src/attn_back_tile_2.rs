//! C-157 point 2 after os-358 (`docs/C-157_P2_ATTN_TILE2.md`, section `attn_back_tile_2`, plan written before this
//! code) — [`crate::attn_back_tile`] (mode 1: dQ, dK, dV) with **the same arithmetic in the same order** and shorter
//! loops (`tools/nak/attn_back_tile_2.comp`, Mesa NAK 25.0.7, SM 8.6): the dS tile of pass dQ with rows of 65 floats,
//! the tiles strictly below / right of the diagonal without guards, the global loads of a tile issued a few at a time
//! before their shared stores, the inner loops unrolled by 4. The same arguments, constant buffer
//! ([`crate::attn_back::Args::cbuf`] with [`crate::attn_back::Mode::Columns`]), grid, block and CPU model
//! (`gpu_compute::layer::attn_back_col_gpu`) as `attn_back_tile`; only the shared size differs. That the board gives
//! `attn_back_tile`'s bits is a **prediction**; tested **only in SIMULARE** (T183). Nothing here has run on the board.

use crate::attn_back::Args;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/attn_back_tile_2.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_BACK_TILE_2: &[u8] =
    include_bytes!("../../../tools/nak/out/attn_back_tile_2.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE_2`] (checked by a test; written in the note).
pub const ATTN_BACK_TILE_2_SHA256: &str =
    "dc50f7d4e6fc3ac5b35c2f7ad81d71e7d1aad2d2cfb8a55a2ceddefa7967468b";
/// Registers per thread (NAK dump `attn_back_tile_2.sm86.txt`).
pub const REGISTERS: u32 = 79;
/// Threads per block: 256, as `attn_back_tile`.
pub const BLOCK: [u32; 3] = crate::attn_back_tile::BLOCK;
/// The largest head width.
pub const MAX_D: u32 = crate::attn_back_tile::MAX_D;
/// Shared memory: the dS / P tile with room for rows of 65 (64 × 65) and the K / Q / dO tile 64 × 64 f32 = 33 024 bytes
/// (256 more than `attn_back_tile`); one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: (64 * 65 + 64 * 64) * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads (the same as `attn_back`).
pub const CBUF_BYTES: usize = crate::attn_back::CBUF_BYTES;

/// The grid: [⌈T/64⌉, H, B] (as `attn_back_tile`).
pub fn grid(a: &Args) -> [u32; 3] {
    crate::attn_back_tile::grid(a)
}

/// Whether the program takes this launch (as `attn_back_tile`: mode 1, d = C / H ≤ 64).
pub fn shape_ok(a: &Args) -> bool {
    crate::attn_back_tile::shape_ok(a)
}
