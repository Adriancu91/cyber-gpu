//! C-160 point 7 (`docs/C-160_P7_ATENTIA_INAPOI_MOD0_PE_PLACI.md`) — mode 0 (the rows of dS) of the training attention
//! backward "on tiles" (`tools/nak/attn_back_tile0.comp`, Mesa NAK 25.0.7, SM 8.6). Blocks of 256 threads own 64 rows
//! (b, h, i0 … i0 + 63); every thread computes 4 × 4 values of dP per 64 × 64 column tile from tiles in shared memory,
//! one thread per row carries the chain dot = Σ P·dP, then every thread rescales its own values into dS. The same
//! constant buffer as [`crate::attn_back`] ([`crate::attn_back::Args::cbuf`] with [`crate::attn_back::Mode::Rows`]);
//! any other mode makes the program exit writing nothing. Launched before [`crate::attn_back_tile`] (mode 1). Every
//! value is the one `attn_back` mode 0 computes (same chains, order and operations): CPU model
//! `gpu_compute::layer::attn_back_row_gpu`, bit for bit. Needs d ≤ 64.
//! Tested **only in SIMULARE** (T175); nothing here has run on the board.

use crate::attn_back::{Args, Mode};
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/attn_back_tile0.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_BACK_TILE0: &[u8] =
    include_bytes!("../../../tools/nak/out/attn_back_tile0.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE0`] (checked by a test; written in the note).
pub const ATTN_BACK_TILE0_SHA256: &str =
    "7107ecc6eceddd05b0e7c0d4360d5fe01b79e5ffa2bbf306afafe2ab06530a9a";
/// Registers per thread (NAK dump `attn_back_tile0.sm86.txt`).
pub const REGISTERS: u32 = 50;
/// Threads per block: 256 (16 × 16 in the code; one dimension in the launch).
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Rows of one block.
pub const ROWS_PER_BLOCK: u32 = 64;
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
    [a.t.div_ceil(ROWS_PER_BLOCK), a.heads, a.batch]
}

/// Whether the program takes this launch: mode 0 ([`Mode::Rows`]), d = C / H ≤ 64, C divisible by H, nothing empty.
pub fn shape_ok(a: &Args) -> bool {
    a.mode == Mode::Rows
        && a.heads != 0
        && a.t != 0
        && a.c != 0
        && a.c.is_multiple_of(a.heads)
        && a.c / a.heads <= MAX_D
}
