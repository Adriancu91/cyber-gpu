//! C-160 point 4 — training program 8, backward half: the gradients of the two embedding tables **without atomics**
//! (`tools/nak/emb_back.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6), F32.
//!
//! The driver groups the rows of dx [N, C] into S segments on the CPU (`gpu_compute::layer::EmbSegments`): segment s
//! sums the rows `rows[off[s]..off[s+1]]` and puts the sum in row `dst[s]` of g:
//! - `emb_pozitie`: one segment per position i < T (rows b·T + i), [`Mode::Write`] — g's rows ≥ T are left to the
//!   driver (zeroed);
//! - `emb_token`: one segment per **distinct** input token (rows ascending), [`Mode::Add`] onto g, which already holds
//!   the tied output head's dW (the order of `loss_and_grad`: the scatter sum first, then + the head).
//!
//! Blocks of 64 threads along k, grid [⌈C/64⌉, S, 1]. Each sum is one thread's chain of `fadd.ftz` from 0 in the
//! segment's order: no atomics, no shared memory, the same bits on every run; the destinations are distinct, so no two
//! threads write one element. The CPU model is `gpu_compute::layer::emb_back_gpu` (bit for bit); the oracle is the
//! training step's f64 sums (`gpu_compute::train::embed_back_rows`), judged in a margin written before the measurement.
//! The program trusts the lists: the driver checks them (`EmbSegments::check`) before the launch. Tested **only in
//! SIMULARE** (T161).

/// `tools/nak/out/emb_back.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const EMB_BACK: &[u8] = include_bytes!("../../../tools/nak/out/emb_back.sm86.sass");
/// SHA-256 of [`EMB_BACK`] (checked by a test; written in the note).
pub const EMB_BACK_SHA256: &str =
    "fca7b007aebb0b8fbfbdbb209f6d4a7162e579212132b47a85da0524a12af0be";
/// Registers per thread (NAK dump `emb_back.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 64.
pub const BLOCK: [u32; 3] = [64, 1, 1];
/// Bytes of constant buffer 0 the program reads (push constants end at 0x58).
pub const CBUF_BYTES: usize = 0x58;

/// What one launch does with row `dst[s]` of g.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// `g = Σ dx` (the position table).
    Write,
    /// `g = g + Σ dx` (the token table, on top of the tied head's dW).
    Add,
}

/// One launch: GPU virtual addresses, the number of segments S and the width C.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[N, C]` f32: the gradient of the residual stream entering layer 0.
    pub dx: u64,
    /// `[S + 1]` u32: segment offsets into `rows`.
    pub off: u64,
    /// `[S]` u32: the destination row of each segment (distinct).
    pub dst: u64,
    /// u32: the rows of dx, grouped by segment.
    pub rows: u64,
    /// `[*, C]` f32, read (in [`Mode::Add`]) and written: the gradient of the table.
    pub g: u64,
    /// S, the number of segments (the grid's y).
    pub segments: u32,
    pub c: u32,
    pub mode: Mode,
}

impl Args {
    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4 (left 0),
    /// dx/off/dst/rows/g at 0x28/0x30/0x38/0x40/0x48, C at 0x50, the mode at 0x54 (0 write, 1 add). `None` for an
    /// empty launch, S past the grid's limit or a missing address.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        let vas = [self.dx, self.off, self.dst, self.rows, self.g];
        if self.c == 0 || self.segments == 0 || self.segments > 65_535 || vas.contains(&0) {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (i, va) in vas.iter().enumerate() {
            let off = 0x28 + 8 * i;
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        cb[0x50..0x54].copy_from_slice(&self.c.to_le_bytes());
        cb[0x54..0x58].copy_from_slice(&((self.mode == Mode::Add) as u32).to_le_bytes());
        Some(cb)
    }

    /// The grid: [⌈C/64⌉, S, 1].
    pub fn grid(&self) -> [u32; 3] {
        [self.c.div_ceil(64), self.segments, 1]
    }
}
