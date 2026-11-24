//! C-157 p.2 (`docs/C-160_P7_LIPIRE_GELU_TC.md`, session 199): [`crate::mm_tc`] with GELU's backward fused into the
//! store (`tools/nak/mm_tc_gelu.comp`, compiled by the **second** NAK tool, Mesa 26.3.1, for SM 8.6). Everything up
//! to the final store is `mm_tc`'s (same f16 slices, same `HMMA`, same 64×64 f32 tile); each finished sum `v` is
//! written as `v · (Φ(u) + u·φ(u))` with exactly `gelu_tr` mode 1's operations (u read at `[row, col]`).
//!
//! In the **mixed** training step it replaces "inapoi:proj-dX" (`mm_tc`) + "inapoi:gelu" (`gelu_tr` mode 1): one
//! launch and two passes over `[rows, 4C]` fewer per layer. The CPU model is `gpu_sim::cmdproc::mm_tc_gelu_on_cpu`.
//! Tested **only in SIMULARE** (T207); nothing here has run on a board.

use crate::mm_tc as tc;
use crate::qmd::SharedMemory;

/// `tools/nak/out2/mm_tc_gelu.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const MM_TC_GELU: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_gelu.sm86.sass");
/// SHA-256 of [`MM_TC_GELU`] (checked by a test; written in the note).
pub const MM_TC_GELU_SHA256: &str =
    "03de80cfffdce1805e7dd343ae8fafa4853c32c6dcf635697efc9ba1e62557d8";
/// Registers per thread (NAK dump `mm_tc_gelu.sm86.txt`).
pub const REGISTERS: u32 = 64;
/// Threads per block: as `mm_tc`, 128 × 1 × 1.
pub const BLOCK: [u32; 3] = tc::BLOCK;
/// Shared memory: as `mm_tc` (24 KiB, one barrier).
pub const SHARED: SharedMemory = tc::SHARED;
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// One product `C = GELU'(u) ⊙ (op(A)·op(B))` with f16 inputs; `C`, `u` are `[m, n]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Args {
    pub a: u64,
    pub b: u64,
    pub c: u64,
    /// `[m, n]` f32: the input of the forward GELU (the fc output). Must not overlap `c`.
    pub u: u64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// [`tc::TA`] / [`tc::TB`] only (no bias).
    pub flags: u32,
}

impl Args {
    /// Constant buffer 0: `mm_tc`'s layout with `u` at the bias slot (0x718). `None` for an empty product, a flag
    /// other than TA / TB, a missing or unaligned address, or `c == u`.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.flags & !(tc::TA | tc::TB) != 0 || self.u == 0 || self.c == self.u {
            return None;
        }
        tc::Args {
            a: self.a,
            b: self.b,
            c: self.c,
            bias: self.u,
            m: self.m,
            n: self.n,
            k: self.k,
            flags: self.flags,
        }
        .cbuf()
    }

    /// The grid: as `mm_tc`, one 128-thread block per 64×64 block of C.
    pub fn grid(&self) -> [u32; 3] {
        [
            self.n.div_ceil(tc::BLOCK_MN),
            self.m.div_ceil(tc::BLOCK_MN),
            1,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::digest as sha256;

    fn hex(d: &[u8]) -> alloc::string::String {
        d.iter().map(|b| alloc::format!("{b:02x}")).collect()
    }

    #[test]
    fn program_is_the_compiled_one() {
        assert_eq!(MM_TC_GELU.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_TC_GELU)), MM_TC_GELU_SHA256);
        assert_ne!(MM_TC_GELU, tc::MM_TC);
    }

    #[test]
    fn cbuf_is_mm_tc_with_u_at_the_bias_slot_and_refusals() {
        let ok = Args {
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            u: 0x4000,
            m: 64,
            n: 256,
            k: 64,
            flags: tc::TB,
        };
        let cb = ok.cbuf().unwrap();
        let same = tc::Args {
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            bias: 0x4000,
            m: 64,
            n: 256,
            k: 64,
            flags: tc::TB,
        }
        .cbuf()
        .unwrap();
        assert_eq!(cb, same);
        assert_eq!(ok.grid(), [4, 1, 1]);
        assert!(
            Args {
                flags: tc::BIAS,
                ..ok
            }
            .cbuf()
            .is_none()
        );
        assert!(Args { u: 0, ..ok }.cbuf().is_none());
        assert!(Args { u: ok.c, ..ok }.cbuf().is_none());
        assert!(Args { k: 0, ..ok }.cbuf().is_none());
    }
}
