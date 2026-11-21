//! C-155 point 3 — the argmax on the board (`tools/nak/argmax.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6).
//!
//! Today every token ends with the copy of the logits down into RAM (`kronos_layer::copy_launch`, 64 KiB for #513's
//! 16 384 values), the OS reads them (os-324: median 47 µs) and picks the greedy token on the processor (85 µs). With
//! this program the board picks it: one block of 256 threads reads the logits in VRAM and writes **one 16-byte
//! result** into RAM ([`ArgmaxResult`]); the OS reads 16 bytes. Nothing is computed differently: choosing a maximum
//! does no arithmetic, so the result is exactly `gpu_compute::model::argmax` of the same logits (the oracle,
//! word for word: `gpu_compute::model::argmax_result`).
//!
//! The proof on the board (for the OS's note): in the same token, the copy of the logits **and** this launch; the
//! board's index must be the processor's argmax of the copied logits at every position (and the 4 words exactly
//! `argmax_result` of them). After that, the fast test can drop the copy. Tested **only in SIMULARE**.

use crate::kronos_layer::{Launch, Prog, cbuf};
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};
use gpu_compute::model::{self, ARGMAX_NONE};
use gpu_core::error::{GpuError, GpuResult};

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// `tools/nak/out/argmax.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const ARGMAX: &[u8] = include_bytes!("../../../tools/nak/out/argmax.sm86.sass");
/// SHA-256 of [`ARGMAX`] (checked by a test; written in the note for the OS).
pub const ARGMAX_SHA256: &str = "4d67cd3495e67cbb63a598e2ad4298925bed51d9660b65094c05586f0cb80572";
/// Registers per thread (NAK dump `argmax.sm86.txt`: 493 instructions, 7 888 bytes).
pub const REGISTERS: u32 = 25;
/// Threads per block (`local_size_x`); one block.
pub const BLOCK: u32 = 256;
/// Shared memory: the 8 warp picks (value, index, count: 96 bytes); one hardware barrier (`bar.sync` in the dump).
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 96,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// The result the program writes: 4 words.
pub const RESULT_BYTES: u64 = 16;
/// Where the arguments sit in the cbuf (NVK 25.0.7 push constants from 0x28): logits, result, n.
pub const CBUF_LOGITS: usize = 0x28;
pub const CBUF_RESULT: usize = 0x30;
pub const CBUF_N: usize = 0x38;

/// One launch of the argmax over `n` f32 logits at `logits` (VRAM, 4-aligned), the 16-byte result at `result`
/// (16-aligned; host RAM for the OS). Appended after a token's list, like the copy of the logits.
pub fn launch(logits: u64, result: u64, n: usize) -> GpuResult<Launch> {
    if n == 0 || n > u32::MAX as usize - 1 {
        return Err(bad("argmax: n in afara 1..2^32-1"));
    }
    if !logits.is_multiple_of(4) || !result.is_multiple_of(16) {
        return Err(bad(
            "argmax: logit-uri nealiniate la 4 sau rezultat nealiniat la 16",
        ));
    }
    Ok(Launch {
        prog: Prog::Argmax,
        cbuf: cbuf(&[logits, result], &[n as u32]),
        grid: 1,
    })
}

/// The 16 bytes the program writes, read back by the driver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArgmaxResult {
    pub words: [u32; 4],
}

impl ArgmaxResult {
    pub fn from_words(words: [u32; 4]) -> Self {
        ArgmaxResult { words }
    }

    /// From the 16 bytes read (little-endian words); fewer bytes = an error.
    pub fn from_bytes(b: &[u8]) -> GpuResult<Self> {
        if b.len() < RESULT_BYTES as usize {
            return Err(bad("argmax: rezultat mai scurt de 16 octeti"));
        }
        let w = |i: usize| u32::from_le_bytes([b[4 * i], b[4 * i + 1], b[4 * i + 2], b[4 * i + 3]]);
        Ok(ArgmaxResult::from_words([w(0), w(1), w(2), w(3)]))
    }

    /// The largest finite logit.
    pub fn top(&self) -> f32 {
        f32::from_bits(self.words[1])
    }

    /// Logits that were NaN or infinite.
    pub fn nonfinite(&self) -> u32 {
        self.words[2]
    }

    /// The token for a vocabulary of `n`: `Ok(Some(i))`, `Ok(None)` when some logit was not finite (the generation
    /// stops as with the logits: `Stop::LogitiNefiniti`), or an error when the 16 bytes are not a result of this
    /// launch — the RAM fill left there (the program did not write), another `n`, an index past `n`, or a
    /// "no token" without a reason. Never a guessed token.
    pub fn token(&self, n: usize) -> GpuResult<Option<usize>> {
        let [i, _, bad_count, got_n] = self.words;
        if got_n as usize != n {
            return Err(bad(
                "argmax: rezultat nescris sau pentru alt n (umplutura ramasa?)",
            ));
        }
        match (i, bad_count) {
            (ARGMAX_NONE, 0) => Err(bad("argmax: niciun simbol fara logit-uri nefinite")),
            (ARGMAX_NONE, _) => Ok(None),
            (i, 0) if (i as usize) < n => Ok(Some(i as usize)),
            _ => Err(bad(
                "argmax: index in afara vocabularului sau simbol cu logit-uri nefinite",
            )),
        }
    }

    /// True if these are exactly the words the program writes for `logits` (the check when the copy of the logits
    /// rides in the same token).
    pub fn agrees(&self, logits: &[f32]) -> bool {
        self.words == model::argmax_result(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_cbuf_and_checks() {
        let l = launch(0x1_0000_0004, 0x2_0000_0010, 16384).unwrap();
        assert_eq!(l.prog, Prog::Argmax);
        assert_eq!(l.grid, 1);
        assert_eq!(
            &l.cbuf[CBUF_LOGITS..CBUF_LOGITS + 8],
            &0x1_0000_0004u64.to_le_bytes()
        );
        assert_eq!(
            &l.cbuf[CBUF_RESULT..CBUF_RESULT + 8],
            &0x2_0000_0010u64.to_le_bytes()
        );
        assert_eq!(&l.cbuf[CBUF_N..CBUF_N + 4], &16384u32.to_le_bytes());
        assert!(l.cbuf[..CBUF_LOGITS].iter().all(|&b| b == 0));
        assert!(l.cbuf[CBUF_N + 4..].iter().all(|&b| b == 0));
        assert!(launch(0x1000, 0x2008, 4).is_err());
        assert!(launch(0x1002, 0x2000, 4).is_err());
        assert!(launch(0x1000, 0x2000, 0).is_err());
    }

    #[test]
    fn result_reading() {
        let v = [1.0f32, 7.0, 7.0, -3.0];
        let r = ArgmaxResult::from_words(model::argmax_result(&v));
        assert_eq!(r.token(4).unwrap(), Some(1));
        assert!(r.agrees(&v));
        assert_eq!(r.top(), 7.0);
        assert!(!r.agrees(&[1.0, 7.0, 7.5, -3.0]));
        // Another n, the RAM fill, an index past n, "none" without a reason: errors, never a token.
        assert!(r.token(5).is_err());
        assert!(ArgmaxResult::from_words([0xeeee_eeee; 4]).token(4).is_err());
        assert!(ArgmaxResult::from_words([4, 0, 0, 4]).token(4).is_err());
        assert!(
            ArgmaxResult::from_words([ARGMAX_NONE, 0, 0, 4])
                .token(4)
                .is_err()
        );
        assert!(ArgmaxResult::from_words([1, 0, 2, 4]).token(4).is_err());
        let nan = ArgmaxResult::from_words(model::argmax_result(&[1.0, f32::NAN]));
        assert_eq!(nan.token(2).unwrap(), None);
        assert_eq!(nan.nonfinite(), 1);
        let b: Vec<u8> = r.words.iter().flat_map(|w| w.to_le_bytes()).collect();
        assert_eq!(ArgmaxResult::from_bytes(&b).unwrap(), r);
        assert!(ArgmaxResult::from_bytes(&b[..15]).is_err());
    }
}
