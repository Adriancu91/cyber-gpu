//! C-155 point 2 — the new LayerNorm (`tools/nak/layernorm256.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6).
//!
//! On the board (os-322, P0) the old `layernorm` took ~21.8 µs per launch for 768 values: one warp of 32 threads,
//! three passes over `x` in global memory, each a chain of dependent loads (DEDUS: waiting, not computing). The new
//! program: one block of **256 threads**, `x`, `g` and `b` read **once** into registers (all loads issued before the
//! first use), the sums by a warp butterfly (`shfl.bfly`) and 8 warp sums through 32 bytes of shared memory with one
//! barrier. Same arguments, same cbuf bytes as `layernorm` — only the program and the block change.
//!
//! The result differs from the old program in the last bits (another summation order); the oracle that mirrors the
//! new order is `gpu_compute::layer::layernorm_gpu256`, and the proof on the board is the usual one (the same tokens
//! as the processor, the logits within `LOGIT_ABS`/`LOGIT_REL`). Tested **only in SIMULARE**.

use alloc::vec::Vec;

use crate::kronos_layer::{Launch, Prog};
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/layernorm256.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const LAYERNORM256: &[u8] = include_bytes!("../../../tools/nak/out/layernorm256.sm86.sass");
/// SHA-256 of [`LAYERNORM256`] (checked by a test; written in the note for the OS).
pub const LAYERNORM256_SHA256: &str =
    "a407dd68507f51d30a8f8ecd0853f8a7d95d9d4aa23a6cdd5644b6fe5b38334d";
/// Registers per thread (NAK dump `layernorm256.sm86.txt`).
pub const REGISTERS: u32 = 25;
/// Threads per block (`local_size_x`).
pub const BLOCK: u32 = 256;
/// The largest vector the program takes (256 threads × 4 values in registers).
pub const MAX_N: usize = gpu_compute::layer::LN256_MAX_N;
/// Shared memory: the 8 warp sums (32 bytes); one hardware barrier (`bar.sync` in the dump).
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 32,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};

/// The same launches with every `layernorm` replaced by `layernorm256` (the cbuf stays byte for byte), when the
/// vector fits ([`MAX_N`]); otherwise the list is returned unchanged.
pub fn widen(mut launches: Vec<Launch>, e: usize) -> Vec<Launch> {
    if e == 0 || e > MAX_N {
        return launches;
    }
    for l in launches.iter_mut() {
        if l.prog == Prog::LayerNorm {
            l.prog = Prog::LayerNorm256;
        }
    }
    launches
}
