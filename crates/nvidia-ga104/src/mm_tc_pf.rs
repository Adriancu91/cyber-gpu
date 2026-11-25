//! C-157 p.2 after os-423 (`docs/C-157_P2_CITIRI_IN_AVANS.md`, session 229): [`crate::mm_tc`] with the global loads
//! issued **ahead** (`tools/nak/mm_tc_pf.comp`, compiled by the **second** NAK tool, Mesa 26.3.1, for SM 8.6).
//!
//! os-423 ran `mm_tc` on the board right bit for bit but at 1.03 TFLOP/s: every global load was waited on before the
//! next one (one branch per load, T234). Here every thread reads its 16 + 16 values of a K-slice into registers with
//! no branch (index clamped to element 0 outside the matrix, value replaced by 0 after the load), and the next slice
//! is read while the tensor cores work on the current one.
//!
//! **Same job, same constant buffer ([`tc::Args::cbuf`], push constants at 0x700), same launch (128 threads, 64×64
//! tiles, [`tc::Args::grid`]), same shared memory and the same arithmetic** as `mm_tc`. The CPU model is
//! `gpu_sim::cmdproc::mm_tc_pf_on_cpu` (= `mm_tc`'s). Tested **only in SIMULARE** (T235); on the board it goes next
//! to `mm_tc` (same products, same bits expected, only the time may differ).

use crate::mm_tc as tc;
use crate::qmd::SharedMemory;

/// `tools/nak/out2/mm_tc_pf.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const MM_TC_PF: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_pf.sm86.sass");
/// SHA-256 of [`MM_TC_PF`] (`sh tools/nak/build2.sh mm_tc_pf`; checked by a test, written in the note).
pub const MM_TC_PF_SHA256: &str =
    "48defc5de143bd55c980521cb769844ba6d3986be85564cd8e331c4bd7bfa048";
/// Registers per thread (NAK dump `mm_tc_pf.sm86.txt`).
pub const REGISTERS: u32 = 248;
/// Threads per block: 128 × 1 × 1, as `mm_tc`.
pub const BLOCK: [u32; 3] = tc::BLOCK;
/// Rows and columns of C per block, as `mm_tc`.
pub const BLOCK_MN: u32 = tc::BLOCK_MN;
/// Shared memory: exactly `mm_tc`'s (24 KiB, one barrier).
pub const SHARED: SharedMemory = tc::SHARED;
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// The grid for one product: `mm_tc`'s (one 128-thread block per 64×64 piece of C).
pub fn grid(args: &tc::Args) -> [u32; 3] {
    args.grid()
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
        assert_eq!(MM_TC_PF.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_TC_PF)), MM_TC_PF_SHA256);
    }

    #[test]
    fn launch_shape_is_mm_tc() {
        let a = tc::Args {
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            bias: 0,
            m: 1024,
            n: 2304,
            k: 768,
            flags: tc::TB,
        };
        assert_eq!(grid(&a), a.grid());
        assert_eq!(grid(&a), [36, 16, 1]);
        assert_eq!(SHARED.bytes, 24 * 1024);
        const { assert!(BLOCK[0] * REGISTERS <= 65_536) };
    }
}
