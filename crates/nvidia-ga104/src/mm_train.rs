//! C-160 point 4 — training program 1: `C = op(A)·op(B) (+ bias)` in f32 (`tools/nak/mm_train.comp`, compiled by
//! Mesa NAK 25.0.7 for SM 8.6). One program for the three products of a `Linear` layer of Kronos:
//!
//! | product | A | B | flags |
//! |---|---|---|---|
//! | forward `Y = X·Wᵀ + b` | `X` `[rows, in]` | `W` stored `[out, in]` | [`TB`] \| [`BIAS`] |
//! | `dX = dY·W` | `dY` `[rows, out]` | `W` `[out, in]` | 0 |
//! | `dW = dYᵀ·X` | `dY` stored `[rows, out]` | `X` `[rows, in]` | [`TA`] |
//!
//! Tiles of 16×16 in shared memory (the layout of matmul step 3), one `ffma.ftz` chain per element over p = 0..K
//! from +0.0, then one `fadd.ftz` of the bias: the bit-exact oracle is
//! `gpu_compute::reference::matmul_t_fma_ftz`; the training oracle (f64 sums) is `gpu_compute::train::linear` /
//! `linear_back`. Correctness first; speed is C-160 point 7. Tested **only in SIMULARE** (T152).

use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/mm_train.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const MM_TRAIN: &[u8] = include_bytes!("../../../tools/nak/out/mm_train.sm86.sass");
/// SHA-256 of [`MM_TRAIN`] (checked by a test; written in the note).
pub const MM_TRAIN_SHA256: &str =
    "69d54ef2334cca67d66fff11cbca838cfd366e33f12a5f9340196c16063fa350";
/// Registers per thread (NAK dump `mm_train.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 16 × 16 (thread x → column of C, y → row).
pub const BLOCK: [u32; 3] = [16, 16, 1];
/// Shared memory: the two 16×16 f32 tiles; one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 2 * 16 * 16 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// A is stored transposed (`[K, M]`).
pub const TA: u32 = 1;
/// B is stored transposed (`[N, K]`, PyTorch's `weight [out, in]`).
pub const TB: u32 = 2;
/// Add `bias[col]` after the sum.
pub const BIAS: u32 = 4;
/// Bytes of constant buffer 0 the program reads (push constants end at 0x58).
pub const CBUF_BYTES: usize = 0x58;

/// One product: GPU virtual addresses, the shape of `C = op(A)[m, k]·op(B)[k, n]` and the flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Args {
    pub a: u64,
    pub b: u64,
    pub c: u64,
    /// Read only with [`BIAS`]; 0 otherwise.
    pub bias: u64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub flags: u32,
}

impl Args {
    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4 (left 0),
    /// a/b/c/bias at 0x28/0x30/0x38/0x40, m/n/k/flags at 0x48/0x4c/0x50/0x54. `None` for an empty product,
    /// unknown flag bits, or a bias flag without a bias address.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return None;
        }
        if self.flags & !(TA | TB | BIAS) != 0 || (self.flags & BIAS != 0 && self.bias == 0) {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (off, va) in [
            (0x28, self.a),
            (0x30, self.b),
            (0x38, self.c),
            (0x40, self.bias),
        ] {
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        for (off, v) in [
            (0x48, self.m),
            (0x4c, self.n),
            (0x50, self.k),
            (0x54, self.flags),
        ] {
            cb[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        Some(cb)
    }

    /// The grid: one 16×16 block per 16×16 tile of C.
    pub fn grid(&self) -> [u32; 3] {
        [self.n.div_ceil(16), self.m.div_ceil(16), 1]
    }
}
