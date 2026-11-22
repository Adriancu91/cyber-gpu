//! C-160 point 4 — training program 9: the total gradient norm and the clipping factor (`tools/nak/gnorm.comp`,
//! compiled by Mesa NAK 25.0.7 for SM 8.6), F32:
//!
//! `N = √(Σ g²)` over ALL gradient tensors (kept one after the other in one buffer), `c = min(1, max_norm / (N + 1e-6))`
//! — PyTorch's `clip_grad_norm_`. No atomics: block b sums `x[b·chunk .. (b+1)·chunk)` (thread t takes every 256th
//! element from b·chunk + t, then the block sum of `ln_fwd`), so the same bits come out on every run. Mode
//! [`Mode::Partial`] writes one partial sum of squares per block; [`Mode::Finish`] (one block) adds the partial sums
//! and writes `res[0] = N`, `res[1] = c`; [`Mode::Whole`] (one block) does both for a buffer of at most one chunk. The
//! √ and the division of the finish are approximate units on the board, so the CPU model
//! (`gpu_compute::layer::{gnorm_block_gpu, gnorm_finish_gpu, grad_norm_gpu}`) rounds them correctly and both are
//! judged against the f64 oracle (`gpu_compute::train::grad_norm`) in a written margin. Tested **only in SIMULARE**
//! (T163).

use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/gnorm.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const GNORM: &[u8] = include_bytes!("../../../tools/nak/out/gnorm.sm86.sass");
/// SHA-256 of [`GNORM`] (checked by a test; written in the note).
pub const GNORM_SHA256: &str = "253c8e00f63e65c78283b4296cd9963e04146facd8231090d933d5d37322a594";
/// Registers per thread (NAK dump `gnorm.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 256.
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Bytes of constant buffer 0 the program reads (push constants end at 0x50).
pub const CBUF_BYTES: usize = 0x50;
/// Shared memory and barriers for the QMD: the 8 warp sums of the block sum, one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 8 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};

/// What one launch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Squares → `part[b]`, one per block (mode 1).
    Partial,
    /// Plain sum of `x` (the partial sums), one block → `res` (mode 2).
    Finish,
    /// Squares → `res`, one block (mode 3).
    Whole,
}

impl Mode {
    fn bits(self) -> u32 {
        match self {
            Mode::Partial => 1,
            Mode::Finish => 2,
            Mode::Whole => 3,
        }
    }
}

/// One launch: GPU virtual addresses, the shape and the mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[n]` f32: all gradients ([`Mode::Partial`], [`Mode::Whole`]) or the partial sums ([`Mode::Finish`]).
    pub x: u64,
    /// `[blocks]` f32, written in [`Mode::Partial`] (may be 0 otherwise).
    pub part: u64,
    /// `[2]` f32, written in [`Mode::Finish`] / [`Mode::Whole`]: N, then the clipping factor (may be 0 otherwise).
    pub res: u64,
    pub n: u32,
    /// Elements per block: a multiple of 256 (in [`Mode::Finish`] / [`Mode::Whole`]: at least n).
    pub chunk: u32,
    pub mode: Mode,
    /// `max_norm` of `clip_grad_norm_` (1.0 in `adamw_513`).
    pub max_norm: f32,
}

impl Args {
    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0 (left 0), x / part /
    /// res at 0x28 / 0x30 / 0x38, n 0x40, chunk 0x44, mode 0x48, max_norm 0x4c. `None` for an empty launch, a chunk that
    /// is not a positive multiple of 256, a single-block mode whose chunk does not cover n, or a missing address.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        let single = self.mode != Mode::Partial;
        if self.n == 0
            || self.chunk == 0
            || !self.chunk.is_multiple_of(256)
            || (single && self.chunk < self.n)
            || self.x == 0
            || (single && self.res == 0)
            || (!single && self.part == 0)
        {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (i, va) in [self.x, self.part, self.res].iter().enumerate() {
            let off = 0x28 + 8 * i;
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        cb[0x40..0x44].copy_from_slice(&self.n.to_le_bytes());
        cb[0x44..0x48].copy_from_slice(&self.chunk.to_le_bytes());
        cb[0x48..0x4c].copy_from_slice(&self.mode.bits().to_le_bytes());
        cb[0x4c..0x50].copy_from_slice(&self.max_norm.to_le_bytes());
        Some(cb)
    }

    /// The grid: ⌈n / chunk⌉ blocks in [`Mode::Partial`], one block otherwise.
    pub fn grid(&self) -> [u32; 3] {
        match self.mode {
            Mode::Partial => [self.n.div_ceil(self.chunk.max(1)), 1, 1],
            _ => [1, 1, 1],
        }
    }
}
