//! C-160 point 4 — training program 7: the loss of the training step, softmax + cross-entropy with its gradient
//! (`tools/nak/xent.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6). One block of 256 threads per row of `v` logits:
//!
//! - `loss[r] = ln Σₖ e^(z[k]−m) + m − z[target[r]]` (m = the row's maximum); the mean over rows is the caller's;
//! - `dlogits[r, k] = (softmax(z)[k] − [k = target[r]])·scale`, with `scale = 1/rows` for the mean loss.
//!
//! `exp`, `ln` and the reciprocal are the GPU's approximate `MUFU` units: the CPU model
//! (`gpu_compute::layer::xent_gpu256`) and the board are judged against the f64 training oracle
//! `gpu_compute::train::softmax_xent` within a margin written before the measurement. A target outside `0..v` makes
//! the row's loss NaN (visible) instead of reading outside the row. Tested **only in SIMULARE** (T153).

use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/xent.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const XENT: &[u8] = include_bytes!("../../../tools/nak/out/xent.sm86.sass");
/// SHA-256 of [`XENT`] (checked by a test; written in the note).
pub const XENT_SHA256: &str = "44cc3536ec1dbd0a1cd2440472eaa58b1d8124578dd9defe0edc4e6f420e339d";
/// Registers per thread (NAK dump `xent.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 256 (8 warps); one block per row.
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Shared memory: the 8 warp results (`parts`); one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 8 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads (push constants end at 0x50).
pub const CBUF_BYTES: usize = 0x50;

/// One launch: GPU virtual addresses, the number of rows and logits per row, the gradient scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[rows, v]` f32.
    pub logits: u64,
    /// `[rows]` u32, each `< v`.
    pub targets: u64,
    /// `[rows, v]` f32, written.
    pub dlogits: u64,
    /// `[rows]` f32, written.
    pub loss: u64,
    pub rows: u32,
    pub v: u32,
    /// `1/rows` for the gradient of the mean loss.
    pub scale: f32,
}

impl Args {
    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4 (left 0),
    /// logits/targets/dlogits/loss at 0x28/0x30/0x38/0x40, v at 0x48, scale at 0x4c. `None` for an empty launch,
    /// a missing address or a scale that is not a finite number.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.rows == 0 || self.v == 0 || !self.scale.is_finite() {
            return None;
        }
        if [self.logits, self.targets, self.dlogits, self.loss].contains(&0) {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (off, va) in [
            (0x28, self.logits),
            (0x30, self.targets),
            (0x38, self.dlogits),
            (0x40, self.loss),
        ] {
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        cb[0x48..0x4c].copy_from_slice(&self.v.to_le_bytes());
        cb[0x4c..0x50].copy_from_slice(&self.scale.to_bits().to_le_bytes());
        Some(cb)
    }

    /// The grid: one block per row.
    pub fn grid(&self) -> [u32; 3] {
        [self.rows, 1, 1]
    }
}
