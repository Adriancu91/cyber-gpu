//! C-165 follow-up (`docs/C-165_MM_TILE_GELU.md`): [`crate::mm_train_tile`] with GELU's backward fused into the store
//! (`tools/nak/mm_train_tile_gelu.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6). The loads, the 64×64 pieces, the
//! 4×4 per thread and the order of the fused multiply-adds are `mm_train_tile`'s; each finished sum `acc` is written
//! as `acc · (Φ(u) + u·φ(u))` with exactly the epilogue of [`crate::mm_train_gelu`] (u read at `[row, col]`).
//! **The same bits as `mm_train_gelu`** (T220, SIMULARE).
//!
//! In a plan with both [`Plan::with_mm_tile`](crate::train_step::Plan::with_mm_tile) and
//! [`Plan::with_fused_gelu`](crate::train_step::Plan::with_fused_gelu) it lets "inapoi:proj-dX+gelu" leave
//! `mm_train_gelu`'s 16×16 tiles too ([`Plan::with_mm_tile_gelu`](crate::train_step::Plan::with_mm_tile_gelu)).
//! Same constant buffer as `mm_train_gelu` ([`mg::Args::cbuf`]); only the launch changes (blocks of 256, one 64×64
//! piece of C per block). The CPU model is `gpu_sim::cmdproc::mm_train_tile_gelu_on_cpu`. Speed on the board: NOT
//! TESTED.

use crate::mm_train_gelu as mg;
use crate::mm_train_tile as mtt;
use crate::qmd::SharedMemory;

/// `tools/nak/out/mm_train_tile_gelu.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const MM_TRAIN_TILE_GELU: &[u8] =
    include_bytes!("../../../tools/nak/out/mm_train_tile_gelu.sm86.sass");
/// SHA-256 of [`MM_TRAIN_TILE_GELU`] (checked by a test; written in the note).
pub const MM_TRAIN_TILE_GELU_SHA256: &str =
    "0c4f96bad3e3dbce87d5c309ba7e2c3d426003ff2097e86dc782ce472ccb11af";
/// Registers per thread (NAK dump `mm_train_tile_gelu.sm86.txt`).
pub const REGISTERS: u32 = 34;
/// Threads per block: as `mm_train_tile`, 256 in x.
pub const BLOCK: [u32; 3] = mtt::BLOCK;
/// Shared memory: as `mm_train_tile` (the two 16×64 tiles, 8 KiB; one barrier).
pub const SHARED: SharedMemory = mtt::SHARED;
/// Bytes of constant buffer 0 the program reads: as `mm_train_gelu`.
pub const CBUF_BYTES: usize = mg::CBUF_BYTES;

/// The grid for one product of [`mg::Args`]: one block per 64×64 piece of C.
pub fn grid(args: &mg::Args) -> [u32; 3] {
    [args.n.div_ceil(mtt::TILE), args.m.div_ceil(mtt::TILE), 1]
}
