//! C-160 point 4 — training program 2 and the second half of program 4: column sums over the rows of a `[rows, c]`
//! gradient (`tools/nak/colsum.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6):
//!
//! - [`Mode::Bias`]: `s0[k] = Σ_r dy[r, k]` — the gradient of a bias;
//! - [`Mode::LayerNorm`]: `s0[k] = Σ_r dy·x̂` (dγ) and `s1[k] = Σ_r dy` (dβ), with `x̂ = (x − μ_r)·rstd_r` recomputed
//!   from the forward pass's statistics (the same operations as `ln_back`).
//!
//! One thread per column (grid x = ⌈c/256⌉ blocks of 256); every thread adds its rows in order from 0.0: no atomics,
//! no shared memory, the same bits at every run. First version, correct rather than fast (few blocks for c = 768).
//! Only add / mul / fma: the CPU model `gpu_compute::layer::colsum_gpu` follows the code NAK made. Judged against
//! the f64 training oracle (`gpu_compute::train::layernorm_back_rows` for dγ / dβ) within a written margin.
//! Tested **only in SIMULARE** (T155).

/// `tools/nak/out/colsum.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const COLSUM: &[u8] = include_bytes!("../../../tools/nak/out/colsum.sm86.sass");
/// SHA-256 of [`COLSUM`] (checked by a test; written in the note).
pub const COLSUM_SHA256: &str = "7a0fa2bde1b464cff50af588371bb09adf0f03399e880a07d75021431a47fb10";
/// Registers per thread (NAK dump `colsum.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 256, one column each.
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Bytes of constant buffer 0 the program reads (push constants end at 0x64).
pub const CBUF_BYTES: usize = 0x64;

/// What the sums are for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `s0 = Σ dy` (a bias); `x`, `mean`, `rstd`, `s1` unused (may be 0).
    Bias,
    /// `s0 = Σ dy·x̂` (dγ), `s1 = Σ dy` (dβ).
    LayerNorm,
}

/// One launch: GPU virtual addresses and the shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[rows, c]` f32.
    pub dy: u64,
    /// `[rows, c]` f32, the forward input of the LayerNorm ([`Mode::LayerNorm`]).
    pub x: u64,
    /// `[rows]` f32 μ ([`Mode::LayerNorm`]).
    pub mean: u64,
    /// `[rows]` f32 1/√(var + eps) ([`Mode::LayerNorm`]).
    pub rstd: u64,
    /// `[c]` f32, written.
    pub s0: u64,
    /// `[c]` f32, written ([`Mode::LayerNorm`]).
    pub s1: u64,
    pub rows: u32,
    pub c: u32,
    pub mode: Mode,
}

impl Args {
    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4 (left 0),
    /// dy/x/μ/rstd/s0/s1 at 0x28/0x30/0x38/0x40/0x48/0x50, rows at 0x58, c at 0x5c, the mode at 0x60 (0 = bias,
    /// 1 = LayerNorm). `None` for an empty launch or an address the mode needs missing.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.rows == 0 || self.c == 0 || self.dy == 0 || self.s0 == 0 {
            return None;
        }
        let ln = self.mode == Mode::LayerNorm;
        if ln && [self.x, self.mean, self.rstd, self.s1].contains(&0) {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        let vas = [self.dy, self.x, self.mean, self.rstd, self.s0, self.s1];
        for (i, va) in vas.iter().enumerate() {
            let off = 0x28 + 8 * i;
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        cb[0x58..0x5c].copy_from_slice(&self.rows.to_le_bytes());
        cb[0x5c..0x60].copy_from_slice(&self.c.to_le_bytes());
        cb[0x60..0x64].copy_from_slice(&u32::from(ln).to_le_bytes());
        Some(cb)
    }

    /// The grid: ⌈c/256⌉ blocks.
    pub fn grid(&self) -> [u32; 3] {
        [self.c.div_ceil(256), 1, 1]
    }
}
