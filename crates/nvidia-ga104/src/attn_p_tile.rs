//! C-160 point 7 (`docs/C-160_P7_ATTN_P_TILE.md`, plan written before this code) — `attn_p_tile`: [`crate::attn_fwd_tile2`]
//! **without P·V**, only the softmax probabilities P [B, H, T, T] (`tools/nak/attn_p_tile.comp`, Mesa NAK 25.0.7, SM 8.6).
//! Phases A and B are `attn_fwd_tile2`'s literally; phase C only writes P (p · 1/sum for j ≤ i, +0 above the diagonal).
//! The same arguments and constant buffer ([`Args`] = [`crate::attn_fwd::Args`]; `att` is never read or written), grid,
//! blocks of 256 and shared size as `attn_fwd_tile2`; CPU model `gpu_compute::layer::attn_p_tile_row_gpu` (the P part of
//! `attn_fwd_tile_row_gpu`). That the board gives `attn_fwd_tile`'s P bits is a **prediction**; tested **only in
//! SIMULARE** (T186). Nothing here has run on the board, and it is not in `train_step` (no new `Prog` variant).

use crate::qmd::SharedMemory;

/// `tools/nak/out/attn_p_tile.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_P_TILE: &[u8] = include_bytes!("../../../tools/nak/out/attn_p_tile.sm86.sass");
/// SHA-256 of [`ATTN_P_TILE`] (checked by a test; written in the note).
pub const ATTN_P_TILE_SHA256: &str =
    "3da85d6b8ec647c0641b2e9b4f8d92fd29a05cd7296b716b28eaedc8dbc96b71";
/// Registers per thread (NAK dump `attn_p_tile.sm86.txt`).
pub const REGISTERS: u32 = 116;
/// Threads per block: 256, as `attn_fwd_tile2`.
pub const BLOCK: [u32; 3] = crate::attn_fwd_tile2::BLOCK;
/// Rows of one block.
pub const ROWS_PER_BLOCK: u32 = crate::attn_fwd_tile2::ROWS_PER_BLOCK;
/// The largest head width.
pub const MAX_D: u32 = crate::attn_fwd_tile2::MAX_D;
/// Shared memory: as `attn_fwd_tile2` (phase A still loads the Q / K tiles with rows of 65) = 37 888 bytes; one barrier.
pub const SHARED: SharedMemory = crate::attn_fwd_tile2::SHARED;
/// Bytes of constant buffer 0 the program reads (the same as `attn_fwd`).
pub const CBUF_BYTES: usize = crate::attn_fwd::CBUF_BYTES;

/// One launch: the arguments of `attn_fwd` (`att` must still be a nonzero address; the program does not touch it).
pub type Args = crate::attn_fwd::Args;

/// The grid: [⌈T/64⌉, H, B] (as `attn_fwd_tile2`).
pub fn grid(a: &Args) -> [u32; 3] {
    crate::attn_fwd_tile2::grid(a)
}

/// Whether the program takes this shape (as `attn_fwd_tile2`: d = C / H ≤ 64).
pub fn shape_ok(a: &Args) -> bool {
    crate::attn_fwd_tile2::shape_ok(a)
}
