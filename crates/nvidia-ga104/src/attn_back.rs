//! C-160 point 4 — training program 6, backward half: causal multi-head attention backward (`tools/nak/attn_back.comp`,
//! compiled by Mesa NAK 25.0.7 for SM 8.6). Blocks of 32 threads, two modes, two launches in order:
//!
//! - mode 0 (rows), grid [⌈T/32⌉, H, B]: one thread per (b, h, i) writes row i of dS [B, H, T, T]:
//!   dP_ij = Σ_e dO_ie·V_je, dot = Σ_j P_ij·dP_ij, dS_ij = P_ij·(dP_ij − dot)·scale for j ≤ i, 0 for j > i;
//! - mode 1 (columns), grid [H, T, B]: 32 threads loop over e and write dQ | dK | dV into dqkv [B·T, 3C]:
//!   dQ_te = Σ_{j≤t} dS_tj·K_je, dK_te = Σ_{i≥t} dS_it·Q_ie, dV_te = Σ_{i≥t} P_it·dO_ie.
//!
//! The causal mask is the loop bounds: P above the diagonal is never read. Every sum is one thread's `ffma` chain in a
//! fixed order (no atomics, no shared memory, no barrier, no MUFU), so the CPU model
//! (`gpu_compute::layer::attn_back_gpu`) is the program bit for bit, and it is judged against the f64 training oracle
//! (`gpu_compute::train::attention_back_rows`) within a margin written before the measurement. Tested **only in
//! SIMULARE** (T159).

/// `tools/nak/out/attn_back.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_BACK: &[u8] = include_bytes!("../../../tools/nak/out/attn_back.sm86.sass");
/// SHA-256 of [`ATTN_BACK`] (checked by a test; written in the note).
pub const ATTN_BACK_SHA256: &str =
    "7092074774576fbe9d213b0d3adb4f02d6fa013771cde8315526879fb0079f54";
/// Registers per thread (NAK dump `attn_back.sm86.txt`).
pub const REGISTERS: u32 = 25;
/// Threads per block: 32.
pub const BLOCK: [u32; 3] = [32, 1, 1];
/// Bytes of constant buffer 0 the program reads (push constants end at 0x68).
pub const CBUF_BYTES: usize = 0x68;

/// Which half of the backward one launch computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Mode 0: the rows of dS (and dP on the way).
    Rows,
    /// Mode 1: dQ | dK | dV from dS and P (after a [`Mode::Rows`] launch).
    Columns,
}

/// One launch: GPU virtual addresses, the shape, the scale 1/√d and the mode.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[B·T, 3C]` f32: Q | K | V (kept from the forward pass).
    pub qkv: u64,
    /// `[B, H, T, T]` f32: the softmax probabilities (only j ≤ i read).
    pub p: u64,
    /// `[B·T, C]` f32: the gradient of the attention output.
    pub dout: u64,
    /// `[B, H, T, T]` f32: dS, written by [`Mode::Rows`], read by [`Mode::Columns`].
    pub ds: u64,
    /// `[B·T, 3C]` f32, written by [`Mode::Columns`]: dQ | dK | dV.
    pub dqkv: u64,
    pub batch: u32,
    pub t: u32,
    pub c: u32,
    pub heads: u32,
    /// 1/√d as f32 (the oracle uses it in f64).
    pub scale: f32,
    pub mode: Mode,
}

impl Args {
    /// The scale the training step uses for head width `d`: 1/√d computed in f64 (Newton, `no_std`), then rounded to
    /// f32. 0 for d = 0.
    pub fn scale_for(d: u32) -> f32 {
        if d == 0 {
            return 0.0;
        }
        let x = d as f64;
        let mut r = x.max(1.0);
        for _ in 0..64 {
            let next = 0.5 * (r + x / r);
            if next >= r {
                break;
            }
            r = next;
        }
        (1.0 / r) as f32
    }

    fn shape_ok(&self) -> bool {
        self.batch > 0
            && self.t > 0
            && self.heads > 0
            && self.c > 0
            && self.c.is_multiple_of(self.heads)
            && self.batch <= 65_535
            && self.t <= 65_535
            && self.heads <= 65_535
    }

    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4/0x8 (left 0),
    /// qkv/p/dout/ds/dqkv at 0x28/0x30/0x38/0x40/0x48, T at 0x50, C at 0x54, H at 0x58, d at 0x5c, the mode at 0x60
    /// (0 rows, 1 columns), the scale at 0x64. `None` for a bad shape or a missing address.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if !self.shape_ok() || [self.qkv, self.p, self.dout, self.ds, self.dqkv].contains(&0) {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (i, va) in [self.qkv, self.p, self.dout, self.ds, self.dqkv]
            .iter()
            .enumerate()
        {
            let off = 0x28 + 8 * i;
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        let d = self.c / self.heads;
        let mode = (self.mode == Mode::Columns) as u32;
        for (off, v) in [
            (0x50, self.t),
            (0x54, self.c),
            (0x58, self.heads),
            (0x5c, d),
            (0x60, mode),
        ] {
            cb[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        cb[0x64..0x68].copy_from_slice(&self.scale.to_le_bytes());
        Some(cb)
    }

    /// The grid: [⌈T/32⌉, H, B] for [`Mode::Rows`], [H, T, B] for [`Mode::Columns`].
    pub fn grid(&self) -> [u32; 3] {
        match self.mode {
            Mode::Rows => [self.t.div_ceil(32), self.heads, self.batch],
            Mode::Columns => [self.heads, self.t, self.batch],
        }
    }
}
