//! C-155 point 4 — the race of the products, inside the token list: `q8_fused` with the work split of
//! `q8_matvec32` (`tools/nak/q8_fused32.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6).
//!
//! On the board (os-328, the glued list at P0) the products take 60 % of the board's time per token: fc 486,
//! proj 406, qkv 381, logits 186 µs. Measured against the bytes they read, they stream at ~60–75 GB/s, about 12 % of
//! the 608 GB/s of the board, while a plain VRAM copy reached 282 GB/s in the same test (DEDUS from the `kp`/`kc` rows).
//! `q8_fused` gives one 36-byte block to a thread: at 768 inputs only 24 of the 32 threads work, and the 32 threads
//! of a step read 32 words 36 bytes apart. `q8_fused32` gives a quarter block (8 columns) to a thread: all 32 threads
//! work, a step reads 8 whole consecutive blocks (288 bytes, no gaps), x comes in two 16-byte loads per quarter. The
//! tail (bias, residual, GELU, K/V copy) is `q8_fused`'s, the push constants are `q8_fused`'s byte for byte: the
//! launcher swaps only the program ([`narrow`]).
//!
//! The result differs from `q8_fused` in the last bits (the summation order of `q8_matvec32`); the oracle is
//! `gpu_compute::q8::row_gpu_order_w32` followed by `q8_fused`'s tail (bit for bit in SIMULARE). On the board the
//! proof is the usual one: the same tokens as the processor, the logits within the margin. Which list is faster is
//! only known on the board (in SIMULARE every launch costs the same). Tested **only in SIMULARE**.

use alloc::vec::Vec;

use crate::kronos_layer::{Launch, Prog};
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/q8_fused32.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const Q8_FUSED32: &[u8] = include_bytes!("../../../tools/nak/out/q8_fused32.sm86.sass");
/// SHA-256 of [`Q8_FUSED32`] (checked by a test; written in the note for the OS).
pub const Q8_FUSED32_SHA256: &str =
    "f26faedd659916b8d0508831329319b754213c7637a6ba40d42f8ab02344bcb2";
/// Registers per thread (NAK dump `q8_fused32.sm86.txt`).
pub const REGISTERS: u32 = 22;
/// Threads per row (`local_size_x`), as `q8_fused`.
pub const BLOCK: u32 = 32;
/// `x` must be aligned to this (the 128-bit loads).
pub const X_ALIGN: u64 = 16;
/// Shared memory: the 32 partial sums, as `q8_fused`; no `BAR.SYNC` in the dump (one warp).
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 128,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 0,
};

/// The quarter-block program for a launch of `q8_fused` or `q8_matvec` whose `x` (cbuf 0x30 in both) is aligned to
/// [`X_ALIGN`]: `q8_fused32` or `q8_matvec32` (the same cbuf bytes); `None` otherwise.
pub fn quarter(l: &Launch) -> Option<Prog> {
    let at = crate::q8_fused::CBUF_X;
    let x = u64::from_le_bytes(l.cbuf[at..at + 8].try_into().unwrap_or([1; 8]));
    if x == 0 || !x.is_multiple_of(X_ALIGN) {
        return None;
    }
    match l.prog {
        Prog::Q8Fused => Some(Prog::Q8Fused32),
        Prog::Q8Matvec => Some(Prog::Q8Matvec32),
        _ => None,
    }
}

/// The same launches with every product whose `x` is 16-aligned on its quarter-block program (`q8_fused` →
/// `q8_fused32`, `q8_matvec` → `q8_matvec32`; the cbuf stays byte for byte); a launch whose `x` is not aligned keeps
/// its program (never an unaligned 128-bit load).
pub fn narrow(mut launches: Vec<Launch>) -> Vec<Launch> {
    for l in launches.iter_mut() {
        if let Some(p) = quarter(l) {
            l.prog = p;
        }
    }
    launches
}

/// How many launches of `launches` run on a quarter-block program (for the journal row `kt lansari pe sferturi`).
pub fn count(launches: &[Launch]) -> usize {
    launches
        .iter()
        .filter(|l| matches!(l.prog, Prog::Q8Fused32 | Prog::Q8Matvec32))
        .count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn launch(x: u64) -> Launch {
        let mut cbuf = [0u8; 256];
        cbuf[crate::q8_fused::CBUF_X..crate::q8_fused::CBUF_X + 8]
            .copy_from_slice(&x.to_le_bytes());
        Launch {
            prog: Prog::Q8Fused,
            cbuf,
            grid: 4,
        }
    }

    #[test]
    fn narrow_swaps_only_aligned_fused_launches() {
        let mut other = launch(0x1000);
        other.prog = Prog::LayerNorm;
        let mut mv = launch(0x2000);
        mv.prog = Prog::Q8Matvec;
        let out = narrow(alloc::vec![
            launch(0x1000),
            launch(0x1004),
            other.clone(),
            launch(0),
            mv
        ]);
        assert_eq!(out[0].prog, Prog::Q8Fused32);
        assert_eq!(out[0].cbuf, launch(0x1000).cbuf);
        assert_eq!(out[1].prog, Prog::Q8Fused);
        assert_eq!(out[2], other);
        assert_eq!(out[3].prog, Prog::Q8Fused);
        assert_eq!(out[4].prog, Prog::Q8Matvec32);
        assert_eq!(count(&out), 2);
    }
}
