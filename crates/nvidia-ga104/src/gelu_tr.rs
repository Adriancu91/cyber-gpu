//! C-160 point 4 — training program 5: GELU forward and backward, element by element (`tools/nak/gelu_tr.comp`,
//! compiled by Mesa NAK 25.0.7 for SM 8.6). Blocks of 256 threads, one element per thread, `i < n`:
//!
//! mode 0 (forward): `y = u·Φ(u)`; mode 1 (backward): `y = g·(Φ(u) + u·φ(u))` — `du` from `dact`.
//! Φ from erf by Abramowitz & Stegun 7.1.26 (as the generation's `gelu`), φ from the same exponential: one `MUFU.RCP`
//! and one `MUFU.EX2` per element (approximate on the board), so the CPU model (`gpu_compute::layer::gelu_tr_gpu`)
//! takes them correctly rounded and both are judged against the f64 training oracle (`gpu_compute::train::gelu_rows`,
//! `gelu_back_rows`) within a margin written before the measurement. Mode 0 never reads `g`. No shared memory, no
//! barrier. Tested **only in SIMULARE** (T158).

/// `tools/nak/out/gelu_tr.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const GELU_TR: &[u8] = include_bytes!("../../../tools/nak/out/gelu_tr.sm86.sass");
/// SHA-256 of [`GELU_TR`] (checked by a test; written in the note).
pub const GELU_TR_SHA256: &str = "52dd5f9a81fd88a864d77ebbeab5a3cc3ab537b89362f1dd36424e21c4ae54f1";
/// Registers per thread (NAK dump `gelu_tr.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 256.
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Bytes of constant buffer 0 the program reads (push constants end at 0x48).
pub const CBUF_BYTES: usize = 0x48;

/// Which direction one launch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `y = gelu(u)` (the MLP activation).
    Forward,
    /// `y = g·gelu'(u)` (the gradient of the MLP's fc output).
    Backward,
}

/// One launch: GPU virtual addresses, the number of elements and the mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[n]` f32: the input of GELU (fc's output, kept from the forward pass).
    pub u: u64,
    /// `[n]` f32: the gradient of GELU's output (read only in [`Mode::Backward`]; may be 0 in [`Mode::Forward`]).
    pub g: u64,
    /// `[n]` f32, written: gelu(u) or du.
    pub y: u64,
    pub n: u32,
    pub mode: Mode,
}

impl Args {
    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4 (left 0),
    /// u/g/y at 0x28/0x30/0x38, n at 0x40, the mode at 0x44 (0 forward, 1 backward). `None` for an empty launch or a
    /// missing address the mode needs.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        let back = self.mode == Mode::Backward;
        if self.n == 0 || self.u == 0 || self.y == 0 || (back && self.g == 0) {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (i, va) in [self.u, self.g, self.y].iter().enumerate() {
            let off = 0x28 + 8 * i;
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        cb[0x40..0x44].copy_from_slice(&self.n.to_le_bytes());
        cb[0x44..0x48].copy_from_slice(&(back as u32).to_le_bytes());
        Some(cb)
    }

    /// The grid: ⌈n / 256⌉ blocks.
    pub fn grid(&self) -> [u32; 3] {
        [self.n.div_ceil(256), 1, 1]
    }
}
