//! C-157 point 2 after os-358 (`docs/C-157_P2_ATTN_TILE2.md`, plan written before this code) — [`crate::attn_fwd_tile`]
//! with **the same arithmetic in the same order** and shorter loops (`tools/nak/attn_fwd_tile2.comp`, Mesa NAK 25.0.7,
//! SM 8.6): shared tiles with rows of 65 floats (no bank conflicts), the tiles below the diagonal without the `j <= i`
//! guard, the 16 global loads of a tile issued before its shared stores, the inner loops unrolled by 4. The same
//! arguments, constant buffer ([`crate::attn_fwd::Args::cbuf`]), grid, block and CPU model
//! (`gpu_compute::layer::attn_fwd_tile_gpu`) as `attn_fwd_tile`; only the shared size differs. That the board gives
//! `attn_fwd_tile`'s bits is a **prediction**; tested **only in SIMULARE** (T181). Nothing here has run on the board.

use crate::attn_fwd::Args;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/attn_fwd_tile2.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_FWD_TILE2: &[u8] = include_bytes!("../../../tools/nak/out/attn_fwd_tile2.sm86.sass");
/// SHA-256 of [`ATTN_FWD_TILE2`] (checked by a test; written in the note).
pub const ATTN_FWD_TILE2_SHA256: &str =
    "2e8cda57ead70edf8b72b7960f34acbe2f3e32fd4ed617eec53c5922c8a3dc08";
/// Registers per thread (NAK dump `attn_fwd_tile2.sm86.txt`).
pub const REGISTERS: u32 = 117;
/// Threads per block: 256, as `attn_fwd_tile`.
pub const BLOCK: [u32; 3] = crate::attn_fwd_tile::BLOCK;
/// Rows of one block.
pub const ROWS_PER_BLOCK: u32 = crate::attn_fwd_tile::ROWS_PER_BLOCK;
/// The largest head width.
pub const MAX_D: u32 = crate::attn_fwd_tile::MAX_D;
/// Shared memory: the two tiles with rows of 65 (2 × 64 × 65 f32), 64 × 16 per-thread row values, the row maxima and
/// the row 1/sum (64 f32 each) = 37 888 bytes (512 more than `attn_fwd_tile`); one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: (2 * 64 * 65 + 64 * 16 + 64 + 64) * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads (the same as `attn_fwd`).
pub const CBUF_BYTES: usize = crate::attn_fwd::CBUF_BYTES;

/// The grid: [⌈T/64⌉, H, B] (as `attn_fwd_tile`).
pub fn grid(a: &Args) -> [u32; 3] {
    crate::attn_fwd_tile::grid(a)
}

/// Whether the program takes this shape (as `attn_fwd_tile`: d = C / H ≤ 64).
pub fn shape_ok(a: &Args) -> bool {
    crate::attn_fwd_tile::shape_ok(a)
}
