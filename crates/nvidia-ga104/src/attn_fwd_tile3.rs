//! C-157 point 2 after os-364 (`docs/C-157_P2_ATTN_TILE3.md`, section `attn_fwd_tile3`, plan written before this
//! code) — [`crate::attn_fwd_tile2`] with **the same arithmetic in the same order** and branch-free global loads
//! (`tools/nak/attn_fwd_tile3.comp`, Mesa NAK 25.0.7, SM 8.6): every global load at a clamped address, always issued,
//! the loads of a group first and the selects 0 / value after them (the Q / K / V tiles, P in phase B and in phase C);
//! the stores keep their guards. The same arguments, constant buffer ([`crate::attn_fwd::Args::cbuf`]), grid, block,
//! shared size and CPU model (`gpu_compute::layer::attn_fwd_tile_gpu`) as `attn_fwd_tile2`. That the board gives
//! `attn_fwd_tile`'s bits is a **prediction**; tested **only in SIMULARE** (T190). Nothing here has run on the board.

use crate::attn_fwd::Args;
use crate::qmd::SharedMemory;

/// `tools/nak/out/attn_fwd_tile3.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_FWD_TILE3: &[u8] = include_bytes!("../../../tools/nak/out/attn_fwd_tile3.sm86.sass");
/// SHA-256 of [`ATTN_FWD_TILE3`] (checked by a test; written in the note).
pub const ATTN_FWD_TILE3_SHA256: &str =
    "4127e5fa49b984894ad038cde8a56d5edfaa5bce38b653ad91d3fc63e59642bc";
/// Registers per thread (NAK dump `attn_fwd_tile3.sm86.txt`).
pub const REGISTERS: u32 = 120;
/// Threads per block: 256, as `attn_fwd_tile`.
pub const BLOCK: [u32; 3] = crate::attn_fwd_tile::BLOCK;
/// Rows of one block.
pub const ROWS_PER_BLOCK: u32 = crate::attn_fwd_tile::ROWS_PER_BLOCK;
/// The largest head width.
pub const MAX_D: u32 = crate::attn_fwd_tile::MAX_D;
/// Shared memory: the same as `attn_fwd_tile2` (37 888 bytes, one barrier).
pub const SHARED: SharedMemory = crate::attn_fwd_tile2::SHARED;
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
