//! C-160 point 4 — training program 10: the optimizer step over all parameters (`tools/nak/opt_step.comp`, compiled
//! by Mesa NAK 25.0.7 for SM 8.6), F32, one element per thread, blocks of 256, `i < n` (w, g, m, v: one buffer each,
//! the tensors one after the other):
//!
//! `g ← g·c` (c = the clipping factor `gnorm` wrote, or 1.0 written by the driver); [`Mode::Sgd`]: `w ← w + (−lr)·g`;
//! [`Mode::AdamW`] (PyTorch 2.14 order): `w ← w·(1 − lr·wd)`, `m ← m + (1−β1)·(g − m)`, `v ← v·β2 + ((1−β2)·g)·g`,
//! `w ← w + (−lr/(1−β1ᵗ))·(m / (√v / √(1−β2ᵗ) + eps))`. Every operation `precise` (no fused multiply-add), so SGD and
//! the m, v updates are PyTorch's f32 operations bit for bit (with FTZ); the √ and the two divisions of AdamW are
//! approximate units on the board, modelled correctly rounded (`gpu_compute::layer::opt_step_gpu`) and judged in a
//! written margin against the oracle (`gpu_compute::train::{sgd_step, AdamW::step}`). Tested **only in SIMULARE**
//! (T164).

/// `tools/nak/out/opt_step.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const OPT_STEP: &[u8] = include_bytes!("../../../tools/nak/out/opt_step.sm86.sass");
/// SHA-256 of [`OPT_STEP`] (checked by a test; written in the note).
pub const OPT_STEP_SHA256: &str =
    "8e15a8b6410bef4b91bee949b332805ee261aa6a9a06d2a7b668f42c240b3b12";
/// Registers per thread (NAK dump `opt_step.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 256.
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Bytes of constant buffer 0 the program reads (push constants end at 0x78).
pub const CBUF_BYTES: usize = 0x78;

/// Which optimizer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// The `sgd` recipe (m, v are not touched).
    Sgd,
    /// The `adamw_513` recipe.
    AdamW,
}

/// One launch: GPU virtual addresses, the number of elements, the mode and the step's f32 scalars
/// (`gpu_compute::layer::OptScalars`, the same fields).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[n]` f32: the gradients (not written).
    pub g: u64,
    /// `[1]` f32: the clipping factor.
    pub coef: u64,
    /// `[n]` f32, read and written: the weights.
    pub w: u64,
    /// `[n]` f32, read and written in [`Mode::AdamW`]: the first moment (may be 0 for SGD).
    pub m: u64,
    /// `[n]` f32, read and written in [`Mode::AdamW`]: the second moment (may be 0 for SGD).
    pub v: u64,
    pub n: u32,
    pub mode: Mode,
    /// −lr, then 1 − lr·wd, 1 − β1, β2, 1 − β2, √(1 − β2ᵗ), −lr/(1 − β1ᵗ), eps.
    pub scalars: [f32; 8],
}

impl Args {
    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0 (left 0), g / coef /
    /// w / m / v at 0x28 / 0x30 / 0x38 / 0x40 / 0x48, n 0x50, mode 0x54, the 8 scalars from 0x58. `None` for an empty
    /// launch or a missing address the mode needs.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        let adam = self.mode == Mode::AdamW;
        if self.n == 0
            || [self.g, self.coef, self.w].contains(&0)
            || (adam && (self.m == 0 || self.v == 0))
        {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (i, va) in [self.g, self.coef, self.w, self.m, self.v]
            .iter()
            .enumerate()
        {
            let off = 0x28 + 8 * i;
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        cb[0x50..0x54].copy_from_slice(&self.n.to_le_bytes());
        cb[0x54..0x58].copy_from_slice(&(adam as u32).to_le_bytes());
        for (i, s) in self.scalars.iter().enumerate() {
            let off = 0x58 + 4 * i;
            cb[off..off + 4].copy_from_slice(&s.to_le_bytes());
        }
        Some(cb)
    }

    /// The grid: ⌈n / 256⌉ blocks.
    pub fn grid(&self) -> [u32; 3] {
        [self.n.div_ceil(256), 1, 1]
    }
}
