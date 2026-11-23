//! Order 20:11, step 3 (`docs/C-160_P7_LIPIRE_INAPOI.md`): `ln_back` with the residual addition fused in
//! (`tools/nak/ln_back_res.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6). The same program as [`crate::ln_back`],
//! plus the address `dres` of the gradient arriving on the residual path:
//!
//! `dx = dres + rstd·((g − m1) − x̂·m2)`, the product rounded first, then one `fadd.ftz` (no FFMA).
//!
//! In the step it replaces `ln_back` + `vadd` ("inapoi:rezidual-ln2" / "inapoi:rezidual-ln1"): one launch and two
//! passes over `[rows, c]` fewer each. The CPU model is `gpu_compute::layer::ln_back_res_gpu256`. Tested **only in
//! SIMULARE** (T200).

use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/ln_back_res.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const LN_BACK_RES: &[u8] = include_bytes!("../../../tools/nak/out/ln_back_res.sm86.sass");
/// SHA-256 of [`LN_BACK_RES`] (checked by a test; written in the note).
pub const LN_BACK_RES_SHA256: &str =
    "e2d0961efd8e01fd8d30f4ed48b85ada8d4fc588a1b8047725fa90913c1ab3ee";
/// Registers per thread (NAK dump `ln_back_res.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 256 (8 warps); one block per row.
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Shared memory: as `ln_back` (`parts[16]`); one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 16 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0 the program reads (push constants end at 0x68).
pub const CBUF_BYTES: usize = 0x68;

/// One launch: [`crate::ln_back::Args`] plus `dres`.
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
    /// `[rows, c]` f32, written: `dres + ln_back(…)`.
    pub dx: u64,
    /// `[rows, c]` f32: the residual path's gradient, added. Must not be `dx` or `dy`.
    pub dres: u64,
    pub rows: u32,
    pub c: u32,
}

impl Args {
    /// `1/c` as the program receives it (f32, correctly rounded).
    pub fn inv_c(&self) -> f32 {
        1.0 / self.c as f32
    }

    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4 (left 0),
    /// x/dy/γ/μ/rstd/dx/dres at 0x28/0x30/0x38/0x40/0x48/0x50/0x58, c at 0x60, 1/c at 0x64. `None` for an empty
    /// launch, a missing address, or `dx` equal to `dres` or `dy` (the rows are read and written by different
    /// threads' passes: no aliasing).
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.rows == 0 || self.c == 0 {
            return None;
        }
        let vas = [
            self.x, self.dy, self.gamma, self.mean, self.rstd, self.dx, self.dres,
        ];
        if vas.contains(&0) || self.dx == self.dres || self.dx == self.dy {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (i, va) in vas.iter().enumerate() {
            let off = 0x28 + 8 * i;
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        cb[0x60..0x64].copy_from_slice(&self.c.to_le_bytes());
        cb[0x64..0x68].copy_from_slice(&self.inv_c().to_bits().to_le_bytes());
        Some(cb)
    }

    /// The grid: one block per row.
    pub fn grid(&self) -> [u32; 3] {
        [self.rows, 1, 1]
    }
}
