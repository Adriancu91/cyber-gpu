//! C-157 after os-457 (`docs/C-157_ATTN_BACK_TC16_V4.md`, session 248): [`crate::attn_back_tile_tc16_v4`] (128-bit
//! global loads) **plus the next tile read ahead** of the tensor-core products (`tools/nak/attn_back_tile_tc16_v4pf.comp`,
//! Mesa NAK 26.3.1, SM 8.6), as `mm_tc_v4` does. The cost: 104 registers (2 blocks per SM at the 64 KiB step instead
//! of 3). Which of v4 / v4pf is faster only the board can say.
//!
//! **Only for aligned shapes** ([`fits`]): a misaligned 128-bit load is a fault on the board, so the host launches
//! `attn_back_tile_tc16` when [`fits`] says no. **The same arithmetic, the same values in the same words of shared
//! memory, the same constant buffer (arguments at 0x700), grid and block** as tc16, so the CPU model is the same
//! `gpu_compute::attn_tc::model` (`gpu_sim::cmdproc::attn_back_tile_tc16_v4_on_cpu`, which refuses unaligned shapes)
//! and the board test is "the same words as `attn_back_tile_tc16`". Tested **only in SIMULARE** (T258).

use crate::attn_back::Args;
use crate::attn_back_tile_tc16_v4 as v4;
use crate::qmd::SharedMemory;

/// `tools/nak/out2/attn_back_tile_tc16_v4pf.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const ATTN_BACK_TILE_TC16_V4PF: &[u8] =
    include_bytes!("../../../tools/nak/out2/attn_back_tile_tc16_v4pf.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE_TC16_V4PF`] (`sh tools/nak/build2.sh attn_back_tile_tc16_v4pf`; checked by a test).
pub const ATTN_BACK_TILE_TC16_V4PF_SHA256: &str =
    "3ba898a96d35e63438a784cbe11bc0f7b136682cc8a283872f4fbe922255f853";
/// Registers per thread (NAK dump `attn_back_tile_tc16_v4pf.sm86.txt`: `Num GPRs: 104`; 2 blocks per SM).
pub const REGISTERS: u32 = 104;
/// 128-bit global loads in the program (NAK dump: 8 before the tile loop, 8 in it; no other global load).
pub const LOADS_128: usize = 16;
/// Threads per block: 256, as tc16.
pub const BLOCK: [u32; 3] = v4::BLOCK;
/// Shared memory: exactly tc16's (16 KiB, one barrier).
pub const SHARED: SharedMemory = v4::SHARED;
/// Size of constant buffer 0 (as tc16: arguments at 0x700).
pub const CBUF_BYTES: usize = v4::CBUF_BYTES;

/// The alignment rule of [`crate::attn_back_tile_tc16_v4::fits`] (the same 128-bit loads).
pub fn fits(a: &Args) -> bool {
    v4::fits(a)
}

/// [`crate::attn_back_tile_tc16_v4::fits_cbuf`].
pub fn fits_cbuf(cb: &[u8]) -> bool {
    v4::fits_cbuf(cb)
}

/// Constant buffer 0: [`crate::attn_back_tile_tc16_v4::cbuf`] (only when [`fits`]).
pub fn cbuf(a: &Args) -> Option<[u8; CBUF_BYTES]> {
    v4::cbuf(a)
}

/// The grid: [⌈T/64⌉, H, B], as tc16.
pub fn grid(a: &Args) -> [u32; 3] {
    v4::grid(a)
}

/// tc16's shape rule; alignment is [`fits`].
pub fn shape_ok(a: &Args) -> bool {
    v4::shape_ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attn_back::Mode;
    use crate::sha256::digest as sha256;

    const DUMP: &str = include_str!("../../../tools/nak/out2/attn_back_tile_tc16_v4pf.sm86.txt");
    const DUMP16: &str = include_str!("../../../tools/nak/out2/attn_back_tile_tc16.sm86.txt");

    fn hex(b: &[u8]) -> alloc::string::String {
        use core::fmt::Write;
        let mut s = alloc::string::String::new();
        for x in b {
            let _ = write!(s, "{x:02x}");
        }
        s
    }

    fn a513() -> Args {
        Args {
            qkv: 0x1000,
            p: 0x2000,
            dout: 0x3000,
            ds: 0x4000,
            dqkv: 0x5004,
            batch: 2,
            t: 512,
            c: 768,
            heads: 12,
            scale: Args::scale_for(64),
            mode: Mode::Columns,
        }
    }

    #[test]
    fn bytes_match_the_written_checksum() {
        assert_eq!(
            hex(&sha256(ATTN_BACK_TILE_TC16_V4PF)),
            ATTN_BACK_TILE_TC16_V4PF_SHA256
        );
        assert_eq!(ATTN_BACK_TILE_TC16_V4PF.len() % 16, 0);
        assert_ne!(
            ATTN_BACK_TILE_TC16_V4PF,
            crate::attn_back_tile_tc16::ATTN_BACK_TILE_TC16
        );
        assert_ne!(ATTN_BACK_TILE_TC16_V4PF, v4::ATTN_BACK_TILE_TC16_V4);
    }

    /// V1–V2 (docs/C-157_ATTN_BACK_TC16_V4.md): 128-bit global loads, no scalar global f32 loads, 16 KiB, 0 local
    /// memory, 0 spills, the same tensor-core and shared-matrix instructions as tc16, RNE packing, no `ffma`; the
    /// arguments read at 0x700.
    #[test]
    fn dump_says_what_the_note_says() {
        assert!(DUMP.contains(&alloc::format!("Num GPRs: {REGISTERS}\n")));
        assert!(DUMP.contains("SLM size: 0\n"));
        assert!(DUMP.contains("Shared size: 16384\n"));
        assert!(DUMP.contains("Spills to mem: 0\n"));
        for op in ["hmma.m16n8k16.f32", "ldsm.16.m8n8.x4", "movm.16.m8n8.trans"] {
            assert_eq!(DUMP.matches(op).count(), DUMP16.matches(op).count(), "{op}");
        }
        assert_eq!(DUMP.matches("= ffma").count(), 0);
        assert_eq!(
            DUMP.matches("= ld.global.a64.constant.b128").count(),
            LOADS_128
        );
        assert_eq!(
            DUMP.matches("= ld.global").count(),
            LOADS_128,
            "a global load that is not 128 bits"
        );
        for part in DUMP.split("c[0x0][").skip(1) {
            let off = part.trim_start_matches('+').split(']').next().unwrap();
            let v = u32::from_str_radix(off.trim_start_matches("0x"), 16).unwrap();
            assert!(v <= 0x8 || (0x700..0x740).contains(&v), "citire la {v:#x}");
        }
    }

    #[test]
    fn fits_the_513_shape_and_refuses_the_rest() {
        let a = a513();
        assert!(fits(&a));
        let cb = cbuf(&a).unwrap();
        assert_eq!(cb, crate::attn_back_tile_tc::cbuf(&a).unwrap());
        assert!(fits_cbuf(&cb));
        assert_eq!(grid(&a), [8, 12, 2]);
        for f in [
            |a: &mut Args| a.t = 130,
            |a: &mut Args| a.qkv += 4,
            |a: &mut Args| a.p += 8,
            |a: &mut Args| a.dout += 12,
            |a: &mut Args| a.ds += 4,
            |a: &mut Args| {
                a.c = 24;
                a.heads = 4
            },
        ] {
            let mut b = a513();
            f(&mut b);
            assert!(!fits(&b));
            assert!(cbuf(&b).is_none());
            if let Some(cb) = crate::attn_back_tile_tc::cbuf(&b) {
                assert!(!fits_cbuf(&cb));
            }
        }
        assert!(!fits_cbuf(&[0u8; 0x700]));
    }
}
