//! C-160 point 7 (`docs/C-160_P7_FIR_LIMITAT.md`) — [`crate::attn_back`] with blocks of 128 threads instead of 32
//! (`tools/nak/attn_back128.comp`, Mesa NAK 25.0.7, SM 8.6). GA104 keeps at most 16 blocks per SM, so blocks of 32 used
//! only 16 of the 48 warp slots. The code of every thread is the code of `attn_back` (the same sums in the same order,
//! the same CPU model `gpu_compute::layer::attn_back_gpu`); only the grid changes:
//!
//! - mode 0 (rows): [⌈T/128⌉, H, B], thread i = the global x index, as before;
//! - mode 1 (columns): [⌈H/4⌉, T, B], warp w of block x takes head 4x + w (lanes loop over e as before); a warp with
//!   head ≥ H leaves at once.
//!
//! The constant buffer is laid out as for `attn_back` (the same push constants), so [`crate::attn_back::Args::cbuf`]
//! serves both. Tested **only in SIMULARE** (T172); nothing here has run on the board.

use crate::attn_back::{Args, Mode};

/// `tools/nak/out/attn_back128.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_BACK128: &[u8] = include_bytes!("../../../tools/nak/out/attn_back128.sm86.sass");
/// SHA-256 of [`ATTN_BACK128`] (checked by a test; written in the note).
pub const ATTN_BACK128_SHA256: &str =
    "5c611f12b1d99caaa5910575e68c006a29bd7f13d4473eda5c7386e010cbb45f";
/// Registers per thread (NAK dump `attn_back128.sm86.txt`).
pub const REGISTERS: u32 = 25;
/// Threads per block: 128.
pub const BLOCK: [u32; 3] = [128, 1, 1];
/// Bytes of constant buffer 0 the program reads (the same as `attn_back`).
pub const CBUF_BYTES: usize = crate::attn_back::CBUF_BYTES;

/// Heads per block in mode 1 (one warp each).
pub const HEADS_PER_BLOCK: u32 = 4;

/// The grid for blocks of 128: [⌈T/128⌉, H, B] for [`Mode::Rows`], [⌈H/4⌉, T, B] for [`Mode::Columns`].
pub fn grid(a: &Args) -> [u32; 3] {
    match a.mode {
        Mode::Rows => [a.t.div_ceil(128), a.heads, a.batch],
        Mode::Columns => [a.heads.div_ceil(HEADS_PER_BLOCK), a.t, a.batch],
    }
}
