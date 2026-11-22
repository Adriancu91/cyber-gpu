//! C-160 point 4 — training program 8, forward half: the input embeddings of the batch (`tools/nak/emb_fwd.comp`,
//! compiled by Mesa NAK 25.0.7 for SM 8.6), F32:
//!
//! `x[r, k] = emb[tok[r], k] + pos[r mod T, k]` for the N = B·T rows, k < C.
//!
//! Blocks of 64 threads along k, grid [⌈C/64⌉, T, B] (position i = the block's y, text b = its z, row r = b·T + i: no
//! division, so no approximate unit — a first build with `r mod T` used NAK's RCP-based division, which is exact only
//! for an RCP at most 1 ulp above the true value, and was dropped). One `fadd.ftz` per element, so the CPU
//! model (`gpu_compute::layer::emb_fwd_gpu`) is the program bit for bit and the oracle is the f32 sum of the training
//! step (`gpu_compute::train::embed_rows`). The program trusts the token ids: [`Args::cbuf`] cannot see them, so the
//! driver checks them against the vocabulary before the launch (`emb_fwd_gpu` returns `None` on the same rule).
//! Tested **only in SIMULARE** (T161).

/// `tools/nak/out/emb_fwd.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const EMB_FWD: &[u8] = include_bytes!("../../../tools/nak/out/emb_fwd.sm86.sass");
/// SHA-256 of [`EMB_FWD`] (checked by a test; written in the note).
pub const EMB_FWD_SHA256: &str = "247e0e42871932822aa201cc6ee8e254ae9e657bec4ce6e30bbd3b691cd5bebb";
/// Registers per thread (NAK dump `emb_fwd.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: 64.
pub const BLOCK: [u32; 3] = [64, 1, 1];
/// Bytes of constant buffer 0 the program reads (push constants end at 0x50).
pub const CBUF_BYTES: usize = 0x50;

/// One launch: GPU virtual addresses and the shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[N]` u32: the input token ids (each < the vocabulary, checked by the driver).
    pub tok: u64,
    /// `[vocab, C]` f32: `emb_token.weight`.
    pub emb: u64,
    /// `[≥ T, C]` f32: `emb_pozitie.weight`.
    pub pos: u64,
    /// `[N, C]` f32, written: the residual stream entering layer 0.
    pub x: u64,
    /// B, the number of texts (the grid's z).
    pub batch: u32,
    pub c: u32,
    /// T, the length of each text (row r = b·T + i uses position i).
    pub t: u32,
}

impl Args {
    /// Constant buffer 0 as NVK lays it out for this program (NAK dump): the workgroup base at 0x0/0x4/0x8 (left 0),
    /// tok/emb/pos/x at 0x28/0x30/0x38/0x40, C at 0x48, T at 0x4c. B is only the grid's z. `None` for an empty shape,
    /// T or B past the grid's limit, or a missing address.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.c == 0
            || self.t == 0
            || self.batch == 0
            || self.t > 65_535
            || self.batch > 65_535
            || [self.tok, self.emb, self.pos, self.x].contains(&0)
        {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (i, va) in [self.tok, self.emb, self.pos, self.x].iter().enumerate() {
            let off = 0x28 + 8 * i;
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        cb[0x48..0x4c].copy_from_slice(&self.c.to_le_bytes());
        cb[0x4c..0x50].copy_from_slice(&self.t.to_le_bytes());
        Some(cb)
    }

    /// The grid: [⌈C/64⌉, T, B].
    pub fn grid(&self) -> [u32; 3] {
        [self.c.div_ceil(64), self.t, self.batch]
    }
}
