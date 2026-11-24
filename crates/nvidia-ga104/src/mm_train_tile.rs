//! C-157 p.2x (`docs/C-157_P2x_MM_TRAIN_TILE.md`): [`crate::mm_train`] on 64×64 tiles, 4×4 results per thread
//! (`tools/nak/mm_train_tile.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6). Same job, same constant buffer
//! ([`mm::Args::cbuf`]) and **the same bits** as `mm_train`: every element is the `ffma.ftz` chain p = 0..K from +0.0,
//! then one `fadd.ftz` of the bias (oracle `gpu_compute::reference::matmul_t_fma_ftz`). What changes is the
//! launch: blocks of 256 threads, a 64×64 piece of C per block, 8 KiB of shared memory. The CPU model is
//! `gpu_sim::cmdproc::mm_train_tile_on_cpu`. Tested **only in SIMULARE** (T214); the speed on the board is DEDUS.

use crate::mm_train as mm;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/mm_train_tile.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const MM_TRAIN_TILE: &[u8] = include_bytes!("../../../tools/nak/out/mm_train_tile.sm86.sass");
/// SHA-256 of [`MM_TRAIN_TILE`] (checked by a test; written in the note).
pub const MM_TRAIN_TILE_SHA256: &str =
    "f8acce2165870280d06271c6b0a60ca96ebc418b23fc7abb8ff496e08bbf094d";
/// Registers per thread (NAK dump `mm_train_tile.sm86.txt`).
pub const REGISTERS: u32 = 34;
/// Threads per block: 256 in x (thread `tid` owns rows `tid / 16 + 16i`, columns `tid % 16 + 16j` of its 64×64 piece).
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Rows and columns of C per block.
pub const TILE: u32 = 64;
/// Shared memory: the 16×64 tiles of op(A) and op(B) (8 KiB); one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 2 * 16 * 64 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads: as `mm_train`.
pub const CBUF_BYTES: usize = mm::CBUF_BYTES;

/// The grid for one product of [`mm::Args`]: one block per 64×64 piece of C.
pub fn grid(args: &mm::Args) -> [u32; 3] {
    [args.n.div_ceil(TILE), args.m.div_ceil(TILE), 1]
}
