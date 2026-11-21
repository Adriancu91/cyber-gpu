//! C-160 point 4 — training program 3: LayerNorm forward over rows that keeps the row statistics
//! (`tools/nak/ln_fwd.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6). One block of 256 threads per row of `c`
//! numbers (any `c`, two passes over the row in memory):
//!
//! `μ = Σx / c`, `var = Σ(x − μ)² / c`, `rstd = 1/√(var + eps)`, `y = (x − μ)·rstd·γ + β`; `mean[r] = μ`,
//! `rstd[r] = rstd` for the backward programs (`ln_back`, `colsum`).
//!
//! One `MUFU.RSQ` per row (approximate on the board), so the CPU model (`gpu_compute::layer::ln_fwd_gpu256`) takes
//! 1/√ correctly rounded and both are judged against the f64 training oracle (`gpu_compute::train::layernorm_rows`,
//! `layernorm_stats`) within a margin written before the measurement. Tested **only in SIMULARE** (T157).

use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/ln_fwd.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const LN_FWD: &[u8] = include_bytes!("../../../tools/nak/out/ln_fwd.sm86.sass");
/// SHA-256 of [`LN_FWD`] (checked by a test; written in the note).
pub const LN_FWD_SHA256: &str = "895fc7988e387cd8ff279695996f5a78e3f326874fa8013aac5efea31e9f2204";
/// Registers per thread (NAK dump `ln_fwd.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 256 (8 warps); one block per row.
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Shared memory: the 8 warp results (`parts[8]`), reused by both sums; one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 8 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads (push constants end at 0x64).
pub const CBUF_BYTES: usize = 0x64;

/// One launch: GPU virtual addresses, the number of rows, numbers per row and ε.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[rows, c]` f32: the input.
    pub x: u64,
    /// `[c]` f32: γ.
    pub gamma: u64,
    /// `[c]` f32: β.
    pub beta: u64,
    /// `[rows, c]` f32, written: the output.
    pub y: u64,
    /// `[rows]` f32, written: μ of every row.
    pub mean: u64,
    /// `[rows]` f32, written: 1/√(var + eps) of every row.
    pub rstd: u64,
    pub rows: u32,
    pub c: u32,
    /// ε (Kronos: 1e-5).
    pub eps: f32,
}

impl Args {
    /// `1/c` as the program receives it (f32, correctly rounded).
    pub fn inv_c(&self) -> f32 {
        1.0 / self.c as f32
    }

    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4 (left 0),
    /// x/γ/β/y/μ/rstd at 0x28/0x30/0x38/0x40/0x48/0x50, c at 0x58, 1/c at 0x5c, ε at 0x60. `None` for an empty
    /// launch, a missing address or an ε that is not a positive finite number.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.rows == 0 || self.c == 0 || !(self.eps > 0.0 && self.eps.is_finite()) {
            return None;
        }
        let vas = [self.x, self.gamma, self.beta, self.y, self.mean, self.rstd];
        if vas.contains(&0) {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (i, va) in vas.iter().enumerate() {
            let off = 0x28 + 8 * i;
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        cb[0x58..0x5c].copy_from_slice(&self.c.to_le_bytes());
        cb[0x5c..0x60].copy_from_slice(&self.inv_c().to_bits().to_le_bytes());
        cb[0x60..0x64].copy_from_slice(&self.eps.to_bits().to_le_bytes());
        Some(cb)
    }

    /// The grid: one block per row.
    pub fn grid(&self) -> [u32; 3] {
        [self.rows, 1, 1]
    }
}
