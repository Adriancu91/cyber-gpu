//! C-157 point 2 after os-364 (`docs/C-157_P2_ATTN_TILE3.md`, section `attn_back_tile0_3`, plan written before this
//! code, `c266ba2`) — [`crate::attn_back_tile0_2`] with **the same arithmetic in the same order** and branch-free
//! global loads (`tools/nak/attn_back_tile0_3.comp`, Mesa NAK 25.0.7, SM 8.6): the dO, V and P tile loads and the
//! loads of pass 2 run always at clamped addresses, then a select 0 / value; above the diagonal pass 2 stores 0.0
//! with no load. The same arguments, constant buffer ([`crate::attn_back::Args::cbuf`] with
//! [`crate::attn_back::Mode::Rows`]), grid, block, shared size and CPU model (`gpu_compute::layer::attn_back_row_gpu`)
//! as `attn_back_tile0_2`. That the board gives `attn_back_tile0`'s bits is a **prediction**; tested **only in
//! SIMULARE** (T188). Nothing here has run on the board.

use crate::attn_back::Args;
use crate::qmd::SharedMemory;

/// `tools/nak/out/attn_back_tile0_3.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_BACK_TILE0_3: &[u8] =
    include_bytes!("../../../tools/nak/out/attn_back_tile0_3.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE0_3`] (checked by a test; written in the note).
pub const ATTN_BACK_TILE0_3_SHA256: &str =
    "d8f4ec408707d6a12ae32b153962a3a55742adb05bc1ac21f1fddaa7c83b611f";
/// Registers per thread (NAK dump `attn_back_tile0_3.sm86.txt`).
pub const REGISTERS: u32 = 68;
/// Threads per block: 256, as `attn_back_tile0`.
pub const BLOCK: [u32; 3] = crate::attn_back_tile0::BLOCK;
/// Rows of one block.
pub const ROWS_PER_BLOCK: u32 = crate::attn_back_tile0::ROWS_PER_BLOCK;
/// The largest head width.
pub const MAX_D: u32 = crate::attn_back_tile0::MAX_D;
/// Shared memory: the same tiles as `attn_back_tile0_2` (33 024 bytes); one barrier.
pub const SHARED: SharedMemory = crate::attn_back_tile0_2::SHARED;
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
