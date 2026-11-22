//! C-160 point 7 (`docs/C-160_P7_FIR_LIMITAT.md`) — [`crate::attn_fwd`] with blocks of 128 threads instead of 32
//! (`tools/nak/attn_fwd128.comp`, Mesa NAK 25.0.7, SM 8.6). GA104 keeps at most 16 blocks per SM, so blocks of 32 used
//! only 16 of the 48 warp slots. The code of every thread is the code of `attn_fwd` (the same sums in the same order,
//! the same CPU model `gpu_compute::layer::attn_fwd_gpu`); only the grid changes: [⌈T/128⌉, H, B]. The constant buffer
//! is laid out as for `attn_fwd` (the same push constants), so [`crate::attn_fwd::Args::cbuf`] serves both.
//! Tested **only in SIMULARE** (T172); nothing here has run on the board.

use crate::attn_fwd::Args;

/// `tools/nak/out/attn_fwd128.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_FWD128: &[u8] = include_bytes!("../../../tools/nak/out/attn_fwd128.sm86.sass");
/// SHA-256 of [`ATTN_FWD128`] (checked by a test; written in the note).
pub const ATTN_FWD128_SHA256: &str =
    "2fa2e8ffbc96bf2edb03b6422dba4dcca433dbd394e5d40d8b8a1a19ba7fb78b";
/// Registers per thread (NAK dump `attn_fwd128.sm86.txt`).
pub const REGISTERS: u32 = 25;
/// Threads per block: 128.
pub const BLOCK: [u32; 3] = [128, 1, 1];
/// Bytes of constant buffer 0 the program reads (the same as `attn_fwd`).
pub const CBUF_BYTES: usize = crate::attn_fwd::CBUF_BYTES;

/// The grid for blocks of 128: [⌈T/128⌉, H, B].
pub fn grid(a: &Args) -> [u32; 3] {
    [a.t.div_ceil(128), a.heads, a.batch]
}
