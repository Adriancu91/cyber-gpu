//! C-160 point 7 (`docs/C-160_P7_ATENTIA_PE_PLACI.md`) — the training attention forward "on tiles"
//! (`tools/nak/attn_fwd_tile.comp`, Mesa NAK 25.0.7, SM 8.6). Blocks of 256 threads own 64 rows (b, h, i0 … i0 + 63);
//! every thread computes 4 × 4 results from Q / K / P / V tiles in shared memory (the reuse a thread-per-row program
//! cannot have). The same arguments and outputs as [`crate::attn_fwd`] (the same constant buffer:
//! [`crate::attn_fwd::Args::cbuf`]): P [B, H, T, T] (+0 above the diagonal) and att [B·T, C]. s, m, x and the order
//! of att are `attn_fwd`'s; only the row sum is 16 partial chains (CPU model:
//! `gpu_compute::layer::attn_fwd_tile_gpu`). Needs d ≤ 64. Tested **only in SIMULARE** (T173); nothing here has run
//! on the board.

use crate::attn_fwd::Args;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/attn_fwd_tile.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_FWD_TILE: &[u8] = include_bytes!("../../../tools/nak/out/attn_fwd_tile.sm86.sass");
/// SHA-256 of [`ATTN_FWD_TILE`] (checked by a test; written in the note).
pub const ATTN_FWD_TILE_SHA256: &str =
    "4ef8dca31f2cdd9c4e672ae126aafc013e47f0443b8cf0f9ae584cb07eebe632";
/// Registers per thread (NAK dump `attn_fwd_tile.sm86.txt`).
pub const REGISTERS: u32 = 55;
/// Threads per block: 256 (16 × 16 in the code; one dimension in the launch).
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Rows of one block.
pub const ROWS_PER_BLOCK: u32 = 64;
/// The largest head width (the shared tiles are 64 wide).
pub const MAX_D: u32 = 64;
/// Shared memory: the Q / P tile and the K / V tile (2 × 64 × 64 f32), 64 × 16 per-thread row values, the row maxima
/// and the row 1/sum (64 f32 each) = 37 376 bytes; barriers (`bar.sync`).
pub const SHARED: SharedMemory = SharedMemory {
    bytes: (2 * 64 * 64 + 64 * 16 + 64 + 64) * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads (the same as `attn_fwd`).
pub const CBUF_BYTES: usize = crate::attn_fwd::CBUF_BYTES;

/// The grid: [⌈T/64⌉, H, B].
pub fn grid(a: &Args) -> [u32; 3] {
    [a.t.div_ceil(ROWS_PER_BLOCK), a.heads, a.batch]
}

/// Whether the program takes this shape (d = C / H ≤ 64, C divisible by H, nothing empty).
pub fn shape_ok(a: &Args) -> bool {
    a.heads != 0 && a.t != 0 && a.c != 0 && a.c.is_multiple_of(a.heads) && a.c / a.heads <= MAX_D
}
