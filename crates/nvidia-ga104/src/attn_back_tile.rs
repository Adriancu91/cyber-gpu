//! C-160 point 7 (`docs/C-160_P7_ATENTIA_INAPOI_PE_PLACI.md`) — mode 1 (columns) of the training attention backward
//! "on tiles" (`tools/nak/attn_back_tile.comp`, Mesa NAK 25.0.7, SM 8.6). Blocks of 256 threads own 64 positions
//! (b, h, t0 … t0 + 63) and all d ≤ 64 columns; every thread computes 4 × 4 results of dQ, then of dK, then of dV from
//! 64 × 64 tiles in shared memory. The same constant buffer as [`crate::attn_back`] ([`crate::attn_back::Args::cbuf`]
//! with [`crate::attn_back::Mode::Columns`]); any other mode makes the program exit writing nothing, so the rows of dS
//! (mode 0) stay [`crate::attn_back`]'s, launched first. Every output is the very `ffma` chain of `attn_back` mode 1
//! (same terms, order and bounds): CPU model `gpu_compute::layer::attn_back_col_gpu`, bit for bit. Needs d ≤ 64.
//! Tested **only in SIMULARE** (T174); nothing here has run on the board.

use crate::attn_back::{Args, Mode};
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/attn_back_tile.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_BACK_TILE: &[u8] = include_bytes!("../../../tools/nak/out/attn_back_tile.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE`] (checked by a test; written in the note).
pub const ATTN_BACK_TILE_SHA256: &str =
    "1da80d3f0902e4e3693da5e179600c39929f6c8f92a58d9ad519ba729a8d7e20";
/// Registers per thread (NAK dump `attn_back_tile.sm86.txt`).
pub const REGISTERS: u32 = 73;
/// Threads per block: 256 (16 × 16 in the code; one dimension in the launch).
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Positions of one block.
pub const POSITIONS_PER_BLOCK: u32 = 64;
/// The largest head width (the shared tiles are 64 wide).
pub const MAX_D: u32 = 64;
/// Shared memory: two 64 × 64 f32 tiles = 32 768 bytes; barriers (`bar.sync`).
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 2 * 64 * 64 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads (the same as `attn_back`).
pub const CBUF_BYTES: usize = crate::attn_back::CBUF_BYTES;

/// The grid: [⌈T/64⌉, H, B].
pub fn grid(a: &Args) -> [u32; 3] {
    [a.t.div_ceil(POSITIONS_PER_BLOCK), a.heads, a.batch]
}

/// Whether the program takes this launch: mode 1 ([`Mode::Columns`]), d = C / H ≤ 64, C divisible by H, nothing empty.
pub fn shape_ok(a: &Args) -> bool {
    a.mode == Mode::Columns
        && a.heads != 0
        && a.t != 0
        && a.c != 0
        && a.c.is_multiple_of(a.heads)
        && a.c / a.heads <= MAX_D
}
