//! C-160 point 7 — `attn_p`: the causal attention's softmax probabilities P [B, H, T, T] only, without P·V
//! (`tools/nak/attn_p.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6; `docs/C-160_P7_ATTN_P.md`). It is
//! [`crate::attn_fwd`] without its last loop: the same constant buffer ([`Args`] = [`crate::attn_fwd::Args`]; `att` is
//! read by nobody and nothing is written there), the same blocks of 32 and grid [⌈T/32⌉, H, B], and every operation that
//! leads to P is `attn_fwd`'s, in the same order — so it rebuilds a layer's P bit for bit when the training step does
//! not keep it (`train_step::Plan::new_p_recalc`). Tested **only in SIMULARE** (T178); in the step
//! since session 175 as `train_step::Prog::AttnP`, used only by `train_step::Plan::new_p_only` (T185).

/// `tools/nak/out/attn_p.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ATTN_P: &[u8] = include_bytes!("../../../tools/nak/out/attn_p.sm86.sass");
/// SHA-256 of [`ATTN_P`] (checked by a test; written in the note).
pub const ATTN_P_SHA256: &str = "786ae9959c3b5321ce04200f8c9e4fd47e8312091199669200e41e10148d386e";
/// Registers per thread (NAK dump `attn_p.sm86.txt`).
pub const REGISTERS: u32 = 22;
/// Threads per block: 32 (as `attn_fwd`).
pub const BLOCK: [u32; 3] = crate::attn_fwd::BLOCK;
/// Bytes of constant buffer 0 (the push constants of `attn_fwd`, unchanged).
pub const CBUF_BYTES: usize = crate::attn_fwd::CBUF_BYTES;

/// One launch: the arguments of `attn_fwd` (`att` must still be a nonzero address; the program does not touch it).
pub type Args = crate::attn_fwd::Args;
