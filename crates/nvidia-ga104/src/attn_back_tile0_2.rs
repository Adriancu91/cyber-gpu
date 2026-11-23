//! C-157 point 2 after os-358 (`docs/C-157_P2_ATTN_TILE2.md`, section `attn_back_tile0_2`, plan written before this
//! code) — [`crate::attn_back_tile0`] with **the same arithmetic in the same order** and shorter loops
//! (`tools/nak/attn_back_tile0_2.comp`, Mesa NAK 25.0.7, SM 8.6): the V and P tiles with rows of 65 floats, the dot
//! chain below the diagonal without the `j <= i` guard, the global loads of a tile issued before its shared stores, the
//! dP loop and the dot loop unrolled by 4. The same arguments, constant buffer ([`crate::attn_back::Args::cbuf`] with
//! [`crate::attn_back::Mode::Rows`]), grid, block and CPU model (`gpu_compute::layer::attn_back_row_gpu`) as
//! `attn_back_tile0`; only the shared size differs. That the board gives `attn_back_tile0`'s bits is a **prediction**;
//! tested **only in SIMULARE** (T182). Nothing here has run on the board.

use crate::attn_back::Args;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/attn_back_tile0_2.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_BACK_TILE0_2: &[u8] =
    include_bytes!("../../../tools/nak/out/attn_back_tile0_2.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE0_2`] (checked by a test; written in the note).
pub const ATTN_BACK_TILE0_2_SHA256: &str =
    "9a8f97603cbcc7a89bec9430ed0d34ba940dc03077841064d85cb9962f4a00d2";
/// Registers per thread (NAK dump `attn_back_tile0_2.sm86.txt`).
pub const REGISTERS: u32 = 52;
/// Threads per block: 256, as `attn_back_tile0`.
pub const BLOCK: [u32; 3] = crate::attn_back_tile0::BLOCK;
/// Rows of one block.
pub const ROWS_PER_BLOCK: u32 = crate::attn_back_tile0::ROWS_PER_BLOCK;
/// The largest head width.
pub const MAX_D: u32 = crate::attn_back_tile0::MAX_D;
/// Shared memory: the dO / dP tile 64 × 64 and the V / P tile with rows of 65 (64 × 65) f32 = 33 024 bytes (256 more
/// than `attn_back_tile0`); one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: (64 * 64 + 64 * 65) * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads (the same as `attn_back`).
pub const CBUF_BYTES: usize = crate::attn_back::CBUF_BYTES;

/// The grid: [⌈T/64⌉, H, B] (as `attn_back_tile0`).
pub fn grid(a: &Args) -> [u32; 3] {
    crate::attn_back_tile0::grid(a)
}

/// Whether the program takes this launch (as `attn_back_tile0`: mode 0, d = C / H ≤ 64).
pub fn shape_ok(a: &Args) -> bool {
    crate::attn_back_tile0::shape_ok(a)
}
