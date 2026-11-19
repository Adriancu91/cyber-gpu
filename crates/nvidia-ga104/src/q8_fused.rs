//! C-147 step 2 (etapa 3a) — `q8_matvec` with the small operations after it glued into the same launch
//! (`tools/nak/q8_fused.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6): the bias, the residual add, GELU, and for
//! QKV the copy of the new key and value into the caches. One layer goes from 16 launches to 7 and #513's token
//! from 196 to 88 (with the copy of the logits down).
//!
//! Every value goes through the same operations, in the same order, as the separate launches did (the dot product
//! exactly as `q8_matvec`, then `bias + `, `res + `, GELU), so the expected result is the old chain's, bit for bit
//! in SIMULARE. On the board the proof is the usual one: the same tokens as the processor.
//!
//! Tested **only in SIMULARE**; nothing here touches a board.

use gpu_compute::q8;
use gpu_core::error::{GpuError, GpuResult};

use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/q8_fused.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const Q8_FUSED: &[u8] = include_bytes!("../../../tools/nak/out/q8_fused.sm86.sass");
/// Registers per thread (NAK dump `q8_fused.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per row (`local_size_x`), as `q8_matvec`.
pub const BLOCK: u32 = 32;
/// Shared memory: the 32 partial sums, as `q8_matvec`; no `BAR.SYNC` in the dump (one warp).
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 128,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 0,
};

/// The push constants in cbuf 0: seven 64-bit GPU VAs, then four 32-bit words.
pub const CBUF_W: usize = 0x28;
pub const CBUF_X: usize = 0x30;
pub const CBUF_BIAS: usize = 0x38;
pub const CBUF_RES: usize = 0x40;
pub const CBUF_Y: usize = 0x48;
pub const CBUF_K: usize = 0x50;
pub const CBUF_V: usize = 0x58;
pub const CBUF_ROWS: usize = 0x60;
pub const CBUF_BLOCKS: usize = 0x64;
pub const CBUF_FLAGS: usize = 0x68;
pub const CBUF_SPLIT: usize = 0x6c;
/// Bytes of push constants (`tools/nak/build.sh kernel q8_fused 72`).
pub const PUSH_BYTES: usize = 0x70 - 0x28;

/// `y = res + v` after the bias.
pub const FLAG_RES: u32 = 1;
/// `y = gelu(v)` (after the bias and, if set, the residual).
pub const FLAG_GELU: u32 = 2;
/// Rows `[split, 2·split)` also to `k`, rows `[2·split, 3·split)` to `v`.
pub const FLAG_KV: u32 = 4;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// What one fused launch does after `W·x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tail {
    /// `y = W·x + bias`.
    Bias,
    /// `y = res + (W·x + bias)` (the attention output and the MLP projection).
    BiasRes { res: u64 },
    /// `y = gelu(W·x + bias)` (the MLP's first matrix).
    BiasGelu,
    /// `y = W·x + bias`, and the key and value thirds also to `k` and `v` (QKV; `split` = n_emb).
    BiasKv { k: u64, v: u64, split: usize },
}

impl Tail {
    pub fn flags(self) -> u32 {
        match self {
            Tail::Bias => 0,
            Tail::BiasRes { .. } => FLAG_RES,
            Tail::BiasGelu => FLAG_GELU,
            Tail::BiasKv { .. } => FLAG_KV,
        }
    }
}

/// The constant buffer for `y[rows] = tail(W[rows × cols] · x[cols] + bias)`. Unused pointers are 0 (the program
/// does not touch them).
pub fn cbuf(
    w: u64,
    x: u64,
    bias: u64,
    y: u64,
    rows: usize,
    cols: usize,
    tail: Tail,
) -> GpuResult<[u8; 256]> {
    q8::matrix_bytes(rows, cols)?;
    if rows == 0 || rows > i32::MAX as usize || cols / q8::BLOCK > u32::MAX as usize {
        return Err(bad("q8_fused: numar de randuri invalid"));
    }
    let (res, k, v, split) = match tail {
        Tail::BiasRes { res } => (res, 0, 0, 0),
        Tail::BiasKv { k, v, split } => {
            if split == 0 || 3 * split != rows {
                return Err(bad("q8_fused: QKV cere randuri = 3 x n_emb"));
            }
            (0, k, v, split)
        }
        _ => (0, 0, 0, 0),
    };
    if !(w | x | bias | y | res | k | v).is_multiple_of(4) {
        return Err(bad("q8_fused: adrese nealiniate la 4"));
    }
    if w == 0 || x == 0 || bias == 0 || y == 0 {
        return Err(bad("q8_fused: adresa lipsa"));
    }
    if matches!(tail, Tail::BiasRes { .. }) && (res == 0 || res == y) {
        return Err(bad("q8_fused: rezidualul lipsa sau acelasi cu iesirea"));
    }
    if matches!(tail, Tail::BiasKv { .. }) && (k == 0 || v == 0) {
        return Err(bad("q8_fused: cache-urile lipsa"));
    }
    let mut cb = [0u8; 256];
    for (at, va) in [
        (CBUF_W, w),
        (CBUF_X, x),
        (CBUF_BIAS, bias),
        (CBUF_RES, res),
        (CBUF_Y, y),
        (CBUF_K, k),
        (CBUF_V, v),
    ] {
        cb[at..at + 8].copy_from_slice(&va.to_le_bytes());
    }
    for (at, word) in [
        (CBUF_ROWS, rows as u32),
        (CBUF_BLOCKS, (cols / q8::BLOCK) as u32),
        (CBUF_FLAGS, tail.flags()),
        (CBUF_SPLIT, split as u32),
    ] {
        cb[at..at + 4].copy_from_slice(&word.to_le_bytes());
    }
    Ok(cb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbuf_layout_and_refusals() {
        let c = cbuf(
            0x1000,
            0x2000,
            0x3000,
            0x4000,
            96,
            64,
            Tail::BiasKv {
                k: 0x5000,
                v: 0x6000,
                split: 32,
            },
        )
        .unwrap();
        let u64_at = |o: usize| u64::from_le_bytes(c[o..o + 8].try_into().unwrap());
        let u32_at = |o: usize| u32::from_le_bytes(c[o..o + 4].try_into().unwrap());
        assert_eq!(u64_at(CBUF_W), 0x1000);
        assert_eq!(u64_at(CBUF_RES), 0);
        assert_eq!(u64_at(CBUF_V), 0x6000);
        assert_eq!(u32_at(CBUF_ROWS), 96);
        assert_eq!(u32_at(CBUF_BLOCKS), 2);
        assert_eq!(u32_at(CBUF_FLAGS), FLAG_KV);
        assert_eq!(u32_at(CBUF_SPLIT), 32);
        assert!(c[..CBUF_W].iter().all(|&b| b == 0));
        assert_eq!(PUSH_BYTES, 72);
        // Refusals: wrong QKV split, residual = output, misaligned, missing caches.
        let kv = |split| Tail::BiasKv {
            k: 0x5000,
            v: 0x6000,
            split,
        };
        assert!(cbuf(0x1000, 0x2000, 0x3000, 0x4000, 96, 64, kv(31)).is_err());
        let r = Tail::BiasRes { res: 0x4000 };
        assert!(cbuf(0x1000, 0x2000, 0x3000, 0x4000, 32, 64, r).is_err());
        assert!(cbuf(0x1002, 0x2000, 0x3000, 0x4000, 32, 64, Tail::Bias).is_err());
        let nok = Tail::BiasKv {
            k: 0,
            v: 0x6000,
            split: 32,
        };
        assert!(cbuf(0x1000, 0x2000, 0x3000, 0x4000, 96, 64, nok).is_err());
        assert!(cbuf(0x1000, 0x2000, 0x3000, 0x4000, 32, 60, Tail::Bias).is_err());
    }
}
