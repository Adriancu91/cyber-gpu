//! C-160 point 4 — training program 6, forward half: causal multi-head attention over all T positions of B texts
//! (`tools/nak/attn_fwd.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6), keeping the softmax probabilities P
//! [B, H, T, T] for the backward program ([`crate::attn_back`]). Blocks of 32 threads, grid [⌈T/32⌉, H, B], one thread
//! per (b, h, i):
//!
//! s_j = (Σ_e Q_ie·K_je)·scale, m = max_j s_j, x_j = exp(s_j − m), P_ij = x_j·(1/Σ x) for j ≤ i and 0 for j > i,
//! att_ie = Σ_j P_ij·V_je.
//!
//! The causal mask is the loop bounds. Every sum is one thread's chain in a fixed order (no atomics, no shared memory,
//! no barrier); `exp` and `1/Σ` are the approximate `MUFU.EX2` / `MUFU.RCP`, so the CPU model
//! (`gpu_compute::layer::attn_fwd_gpu`) follows every operation and both are judged against the f64 training oracle
//! (`gpu_compute::train::attention_rows`) within a margin written before the measurement. Tested **only in SIMULARE**
//! (T160).

/// `tools/nak/out/attn_fwd.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_FWD: &[u8] = include_bytes!("../../../tools/nak/out/attn_fwd.sm86.sass");
/// SHA-256 of [`ATTN_FWD`] (checked by a test; written in the note).
pub const ATTN_FWD_SHA256: &str =
    "fe97ab49c5b04303d0e2a2e2a9d7bc9c22d8bfbaaf2e8cbfa3e41104dbaa5596";
/// Registers per thread (NAK dump `attn_fwd.sm86.txt`).
pub const REGISTERS: u32 = 25;
/// Threads per block: 32.
pub const BLOCK: [u32; 3] = [32, 1, 1];
/// Bytes of constant buffer 0 the program reads (push constants end at 0x54).
pub const CBUF_BYTES: usize = 0x54;

/// One launch: GPU virtual addresses, the shape and the scale 1/√d.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[B·T, 3C]` f32: Q | K | V (the output of `in_proj`).
    pub qkv: u64,
    /// `[B, H, T, T]` f32, written: the softmax probabilities (+0 above the diagonal), kept for `attn_back`.
    pub p: u64,
    /// `[B·T, C]` f32, written: the attention output, heads side by side.
    pub att: u64,
    pub batch: u32,
    pub t: u32,
    pub c: u32,
    pub heads: u32,
    /// 1/√d as f32 ([`crate::attn_back::Args::scale_for`]).
    pub scale: f32,
}

impl Args {
    fn shape_ok(&self) -> bool {
        self.batch > 0
            && self.t > 0
            && self.heads > 0
            && self.c > 0
            && self.c.is_multiple_of(self.heads)
            && self.batch <= 65_535
            && self.heads <= 65_535
            && self.t <= 65_535 * 32
    }

    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4/0x8 (left 0),
    /// qkv/p/att at 0x28/0x30/0x38, T at 0x40, C at 0x44, H at 0x48, d at 0x4c, the scale at 0x50. `None` for a bad
    /// shape or a missing address.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if !self.shape_ok() || [self.qkv, self.p, self.att].contains(&0) {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (i, va) in [self.qkv, self.p, self.att].iter().enumerate() {
            let off = 0x28 + 8 * i;
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        let d = self.c / self.heads;
        for (off, v) in [
            (0x40, self.t),
            (0x44, self.c),
            (0x48, self.heads),
            (0x4c, d),
        ] {
            cb[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        cb[0x50..0x54].copy_from_slice(&self.scale.to_le_bytes());
        Some(cb)
    }

    /// The grid: [⌈T/32⌉, H, B].
    pub fn grid(&self) -> [u32; 3] {
        [self.t.div_ceil(32), self.heads, self.batch]
    }
}
