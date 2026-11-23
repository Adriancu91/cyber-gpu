//! C-157 point 2 after the 15:57 cycle (`docs/C-157_P2_ATTN_TILE4.md`, plan written before this code) —
//! [`crate::attn_back_tile_3`] (mode 1: dQ, dK, dV) with one change: in passes dK / dV the **full diagonal tile** also
//! takes the unguarded inner loop (`tools/nak/attn_back_tile_4.comp`, Mesa NAK 25.0.7, SM 8.6). The tile load already
//! holds +0 above the diagonal and the masked steps come first, while the accumulator is still +0, so for finite
//! Q / dO the same bits are predicted. The same arguments, constant buffer ([`crate::attn_back::Args::cbuf`] with
//! [`crate::attn_back::Mode::Columns`]), grid, block, shared size and CPU model (`gpu_compute::layer::attn_back_col_gpu`)
//! as `attn_back_tile_3`. Tested **only in SIMULARE** (T196). Nothing here has run on the board.

use crate::attn_back::Args;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/attn_back_tile_4.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_BACK_TILE_4: &[u8] =
    include_bytes!("../../../tools/nak/out/attn_back_tile_4.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE_4`] (checked by a test; written in the note).
pub const ATTN_BACK_TILE_4_SHA256: &str =
    "74f2cb81c2d7d69e9311b3fac4ff3ccd690da97f27e8875f516e1827c19eac35";
/// Registers per thread (NAK dump `attn_back_tile_4.sm86.txt`).
pub const REGISTERS: u32 = 80;
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
