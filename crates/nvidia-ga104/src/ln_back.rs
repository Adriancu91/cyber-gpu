//! C-160 point 4 — training program 4, first half: LayerNorm backward for `dx` (`tools/nak/ln_back.comp`, compiled
//! by Mesa NAK 25.0.7 for SM 8.6). One block of 256 threads per row of `c` numbers, with the row's μ and
//! `rstd = 1/√(var + eps)` kept by the forward pass:
//!
//! `x̂ = (x − μ)·rstd`, `g = dy·γ`, `m1 = Σg / c`, `m2 = Σ g·x̂ / c`, `dx = rstd·((g − m1) − x̂·m2)`.
//!
//! dγ = Σ_r dy·x̂ and dβ = Σ_r dy are column sums over the rows, left to the column-sum program (#2): no atomics, the
//! same result at every run. No `MUFU` unit and no division (the caller gives `1/c`), so the CPU model
//! (`gpu_compute::layer::ln_back_gpu256`) follows the code NAK made; both are judged against the f64 training oracle
//! `gpu_compute::train::layernorm_back_rows` within a margin written before the measurement. Tested **only in
//! SIMULARE** (T154).

use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/ln_back.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const LN_BACK: &[u8] = include_bytes!("../../../tools/nak/out/ln_back.sm86.sass");
/// SHA-256 of [`LN_BACK`] (checked by a test; written in the note).
pub const LN_BACK_SHA256: &str = "369afd8dd343b9a508d481998b75de899d417d33c991c11351ec21db85cd7063";
/// Registers per thread (NAK dump `ln_back.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 256 (8 warps); one block per row.
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Shared memory: the 8 warp results of both sums (`parts[16]`); one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 16 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads (push constants end at 0x60).
pub const CBUF_BYTES: usize = 0x60;

/// One launch: GPU virtual addresses, the number of rows and numbers per row.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[rows, c]` f32: the forward input of the LayerNorm.
    pub x: u64,
    /// `[rows, c]` f32: the gradient of its output.
    pub dy: u64,
    /// `[c]` f32: γ.
    pub gamma: u64,
    /// `[rows]` f32: μ of every row (forward pass).
    pub mean: u64,
    /// `[rows]` f32: 1/√(var + eps) of every row (forward pass).
    pub rstd: u64,
    /// `[rows, c]` f32, written.
    pub dx: u64,
    pub rows: u32,
    pub c: u32,
}

impl Args {
    /// `1/c` as the program receives it (f32, correctly rounded).
    pub fn inv_c(&self) -> f32 {
        1.0 / self.c as f32
    }

    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4 (left 0),
    /// x/dy/γ/μ/rstd/dx at 0x28/0x30/0x38/0x40/0x48/0x50, c at 0x58, 1/c at 0x5c. `None` for an empty launch or a
    /// missing address.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.rows == 0 || self.c == 0 {
            return None;
        }
        let vas = [self.x, self.dy, self.gamma, self.mean, self.rstd, self.dx];
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
        Some(cb)
    }

    /// The grid: one block per row.
    pub fn grid(&self) -> [u32; 3] {
        [self.rows, 1, 1]
    }
}
