//! C-157 after os-457 (`docs/C-157_ATTN_BACK_TC16_V4.md`, session 248): [`crate::attn_back_tile_tc16`] with
//! **128-bit** global loads (`tools/nak/attn_back_tile_tc16_v4.comp`, Mesa NAK 26.3.1, SM 8.6). os-457 measured tc16
//! waiting on its loads most of the time (32 scalar loads per thread per tile, at most 2 in flight in the listing);
//! here 4 + 4 `vec4` loads per thread per tile, 5 in flight. The variant that also reads the next tile ahead of the
//! products is [`crate::attn_back_tile_tc16_v4pf`] (104 registers instead of 72).
//!
//! **Only for aligned shapes** ([`fits`]): a misaligned 128-bit load is a fault on the board, so the host launches
//! `attn_back_tile_tc16` when [`fits`] says no. **The same arithmetic, the same values in the same words of shared
//! memory, the same constant buffer (arguments at 0x700), grid and block** as tc16, so the CPU model is the same
//! `gpu_compute::attn_tc::model` (`gpu_sim::cmdproc::attn_back_tile_tc16_v4_on_cpu`, which refuses unaligned shapes)
//! and the board test is "the same words as `attn_back_tile_tc16`". Tested **only in SIMULARE** (T258).

use crate::attn_back::Args;
use crate::qmd::SharedMemory;

/// `tools/nak/out2/attn_back_tile_tc16_v4.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const ATTN_BACK_TILE_TC16_V4: &[u8] =
    include_bytes!("../../../tools/nak/out2/attn_back_tile_tc16_v4.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE_TC16_V4`] (`sh tools/nak/build2.sh attn_back_tile_tc16_v4`; checked by a test).
pub const ATTN_BACK_TILE_TC16_V4_SHA256: &str =
    "b19c845184c4110618a7864b00c2c087f8df9e86c9d50e6b4662bb8d45e95a86";
/// Registers per thread (NAK dump `attn_back_tile_tc16_v4.sm86.txt`: `Num GPRs: 72`; 3 blocks per SM at the
/// 64 KiB step, as tc16 — the shared memory is the limit).
pub const REGISTERS: u32 = 72;
/// 128-bit global loads in the program (NAK dump: 4 + 4 in the tile loop; no other global load).
pub const LOADS_128: usize = 8;
/// Threads per block: 256, as tc16.
pub const BLOCK: [u32; 3] = crate::attn_back_tile_tc16::BLOCK;
/// Shared memory: exactly tc16's (16 KiB, one barrier).
pub const SHARED: SharedMemory = crate::attn_back_tile_tc16::SHARED;
/// Size of constant buffer 0 (as tc16: arguments at 0x700).
pub const CBUF_BYTES: usize = crate::attn_back_tile_tc16::CBUF_BYTES;

/// May `attn_back_tile_tc16_v4` run this launch? tc16's shape rule ([`shape_ok`]) plus: T and d = C / H multiples of
/// 4 and the addresses of qkv, p, dout and dS multiples of 16 bytes — then every 128-bit load is aligned and lies
/// wholly inside its row. `false` → launch `attn_back_tile_tc16` instead.
pub fn fits(a: &Args) -> bool {
    shape_ok(a)
        && a.t.is_multiple_of(4)
        && (a.c / a.heads).is_multiple_of(4)
        && [a.qkv, a.p, a.dout, a.ds]
            .iter()
            .all(|v| v.is_multiple_of(16))
}

/// The same rule read from a constant buffer laid out for the tc programs (arguments at 0x700, as
/// [`crate::attn_back_tile_tc::cbuf`] writes them): `false` if `cb` is too short or its mode is not 1.
pub fn fits_cbuf(cb: &[u8]) -> bool {
    const B: usize = crate::attn_back_tile_tc::PUSH_BASE;
    if cb.len() < B + 0x3c {
        return false;
    }
    let u64_at = |o: usize| u64::from_le_bytes(cb[B + o..B + o + 8].try_into().unwrap_or([1; 8]));
    let u32_at = |o: usize| u32::from_le_bytes(cb[B + o..B + o + 4].try_into().unwrap_or([1; 4]));
    u32_at(0x38) == 1
        && [0x00, 0x08, 0x10, 0x18]
            .iter()
            .all(|&o| u64_at(o).is_multiple_of(16))
        && u32_at(0x28).is_multiple_of(4)
        && u32_at(0x34).is_multiple_of(4)
        && u32_at(0x28) != 0
        && u32_at(0x34) != 0
}

/// Constant buffer 0: [`crate::attn_back_tile_tc::cbuf`], only when [`fits`] (otherwise `None`).
pub fn cbuf(a: &Args) -> Option<[u8; CBUF_BYTES]> {
    if !fits(a) {
        return None;
    }
    crate::attn_back_tile_tc::cbuf(a)
}

/// The grid: [⌈T/64⌉, H, B], as tc16.
pub fn grid(a: &Args) -> [u32; 3] {
    crate::attn_back_tile_tc16::grid(a)
}

/// tc16's shape rule (mode 1, d = C / H ≤ 64); alignment is [`fits`].
pub fn shape_ok(a: &Args) -> bool {
    crate::attn_back_tile_tc16::shape_ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attn_back::Mode;
    use crate::sha256::digest as sha256;

    const DUMP: &str = include_str!("../../../tools/nak/out2/attn_back_tile_tc16_v4.sm86.txt");
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
            hex(&sha256(ATTN_BACK_TILE_TC16_V4)),
            ATTN_BACK_TILE_TC16_V4_SHA256
        );
        assert_eq!(ATTN_BACK_TILE_TC16_V4.len() % 16, 0);
        assert_ne!(
            ATTN_BACK_TILE_TC16_V4,
            crate::attn_back_tile_tc16::ATTN_BACK_TILE_TC16
        );
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
