//! C-157 (`docs/C-157_ATTN_BACK_TC.md`, plan and predictions B1–B6 pushed before this code, `be02aaa`):
//! [`crate::attn_back_tile_3`]'s job (mode 1: dQ = dS·K, dK = dSᵀ·Q, dV = Pᵀ·dO) on the **tensor cores**
//! (`tools/nak/attn_back_tile_tc.comp`, compiled by the second NAK tool, Mesa 26.3.1, SM 8.6): every element of
//! dS, P, Q, K, dO rounded to binary16 in shared memory, the products added in f32 (`hmma.m16n8k16.f32`).
//!
//! **Same arguments** ([`crate::attn_back::Args`] with [`crate::attn_back::Mode::Columns`]), **same grid, block and
//! result places** as `attn_back_tile_3`; the program, its registers, its shared size **and the place of the arguments
//! in cbuf 0** differ (0x700, as every second-tool program — [`cbuf`]; session 244 found this in the dump). The numbers are NOT the same bits as the F32 program: the CPU model is `gpu_compute::attn_tc::model`, the
//! verdict `gpu_compute::attn_tc::judge` (the `mm_tc` margin, per head). The mixed #513 step already carries the error
//! scale 2¹⁴ in dO / dS. Tested **only in SIMULARE** (T252). Nothing here has run on the board.

use crate::attn_back::Args;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out2/attn_back_tile_tc.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const ATTN_BACK_TILE_TC: &[u8] =
    include_bytes!("../../../tools/nak/out2/attn_back_tile_tc.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE_TC`] (`sh tools/nak/build2.sh attn_back_tile_tc`; checked by a test, in the note).
pub const ATTN_BACK_TILE_TC_SHA256: &str =
    "527d8dd1dc5000c26541561e8c60f59f4b5dab0d034a4d5ee593ae415aed5c65";
/// Registers per thread (NAK dump `attn_back_tile_tc.sm86.txt`: `Num GPRs: 48`).
pub const REGISTERS: u32 = 48;
/// Threads per block: 256, as `attn_back_tile_3`.
pub const BLOCK: [u32; 3] = crate::attn_back_tile::BLOCK;
/// The largest head width.
pub const MAX_D: u32 = crate::attn_back_tile::MAX_D;
/// Shared memory: two f16 tiles 64 × 64 (16 KiB) and the f32 result tile 64 × 64 (16 KiB) = 32 768 bytes
/// (the dump's `Shared size`); one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 2 * 64 * 64 * 2 + 64 * 64 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Where NVK 26.3.1 (the second NAK tool) puts the push constants in cbuf 0 — the dump reads `c[0x0][+0x700]` …
/// `c[0x0][+0x738]` (session 244). **Not** `attn_back`'s 0x28: the arguments are the same, their place is not.
pub const PUSH_BASE: usize = 0x700;
/// Size of constant buffer 0 (covers 0x700 + the 0x40 bytes of arguments, multiple of 256), as `mm_tc`.
/// Session 244: was wrongly `attn_back::CBUF_BYTES` (0x68) in session 243; nothing outside this repo used it.
pub const CBUF_BYTES: usize = 0x800;

/// Constant buffer 0 for this program: zero up to 0x700 (the workgroup base at 0x0 / 0x4 / 0x8 left 0), then the
/// bytes 0x28..0x68 of [`crate::attn_back::Args::cbuf`] (qkv, p, dout, ds, dqkv, T, C, H, d, mode, scale) at
/// 0x700..0x740. `None` if `attn_back` refuses the arguments, the mode is not [`crate::attn_back::Mode::Columns`] or
/// the shape does not fit ([`shape_ok`]).
pub fn cbuf(a: &Args) -> Option<[u8; CBUF_BYTES]> {
    if a.mode != crate::attn_back::Mode::Columns || !shape_ok(a) {
        return None;
    }
    cbuf_from_f32(&a.cbuf()?)
}

/// The same move for a cbuf already laid out for the F32 programs (`attn_back` / `attn_back_tile_3`, arguments at
/// 0x28): bytes 0x28..0x68 go to 0x700..0x740. `None` if `cb` is shorter than 0x68 bytes, its workgroup base
/// (0x0..0xc) is not zero, or its mode (0x60) is not 1.
pub fn cbuf_from_f32(cb: &[u8]) -> Option<[u8; CBUF_BYTES]> {
    const ARGS: usize = 0x28;
    const END: usize = crate::attn_back::CBUF_BYTES;
    if cb.len() < END || cb[..0xc].iter().any(|&b| b != 0) || cb[0x60..0x64] != 1u32.to_le_bytes() {
        return None;
    }
    let mut out = [0u8; CBUF_BYTES];
    out[PUSH_BASE..PUSH_BASE + (END - ARGS)].copy_from_slice(&cb[ARGS..END]);
    Some(out)
}

/// The grid: [⌈T/64⌉, H, B] (as `attn_back_tile_3`).
pub fn grid(a: &Args) -> [u32; 3] {
    crate::attn_back_tile::grid(a)
}

/// Whether the program takes this launch (as `attn_back_tile_3`: mode 1, d = C / H ≤ 64).
pub fn shape_ok(a: &Args) -> bool {
    crate::attn_back_tile::shape_ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::digest as sha256;

    const DUMP: &str = include_str!("../../../tools/nak/out2/attn_back_tile_tc.sm86.txt");

    fn hex(b: &[u8]) -> alloc::string::String {
        use core::fmt::Write;
        let mut s = alloc::string::String::new();
        for x in b {
            let _ = write!(s, "{x:02x}");
        }
        s
    }

    #[test]
    fn bytes_match_the_written_checksum() {
        assert_eq!(hex(&sha256(ATTN_BACK_TILE_TC)), ATTN_BACK_TILE_TC_SHA256);
        assert_eq!(ATTN_BACK_TILE_TC.len() % 16, 0);
    }

    /// B5 (docs/C-157_ATTN_BACK_TC.md): 0 local memory, ≤ 64 registers, tensor-core products, no `ffma` at all.
    #[test]
    fn dump_says_what_the_note_says() {
        assert!(DUMP.contains(&alloc::format!("Num GPRs: {REGISTERS}\n")));
        assert!(DUMP.contains("SLM size: 0\n"));
        assert!(DUMP.contains(&alloc::format!("Shared size: {}\n", SHARED.bytes)));
        assert!(DUMP.contains("Spills to mem: 0\n"));
        assert_eq!(DUMP.matches("hmma.m16n8k16.f32").count(), 4);
        assert_eq!(DUMP.matches("= ffma").count(), 0);
        const { assert!(REGISTERS <= 64) };
    }

    /// Session 244: every cbuf-0 word the program reads is the workgroup base (0x0 / 0x4 / 0x8) or an argument at
    /// 0x700..0x740 — never `attn_back`'s 0x28..0x68 — and [`cbuf`] puts T, C, H, d, mode where the dump reads them.
    #[test]
    fn cbuf_is_where_the_dump_reads() {
        let mut seen = alloc::vec::Vec::new();
        for part in DUMP.split("c[0x0][").skip(1) {
            let off = part.trim_start_matches('+').split(']').next().unwrap();
            let v = u32::from_str_radix(off.trim_start_matches("0x"), 16).unwrap() as usize;
            assert!(
                v <= 0x8 || (PUSH_BASE..PUSH_BASE + 0x40).contains(&v),
                "citire la {v:#x}"
            );
            seen.push(v);
        }
        for off in [
            0x700, 0x708, 0x710, 0x718, 0x720, 0x728, 0x72c, 0x730, 0x734, 0x738,
        ] {
            assert!(seen.contains(&off), "{off:#x} necitit");
        }
        let a = Args {
            qkv: 0x1000,
            p: 0x2000,
            dout: 0x3000,
            ds: 0x4000,
            dqkv: 0x5000,
            batch: 2,
            t: 512,
            c: 768,
            heads: 12,
            scale: Args::scale_for(64),
            mode: crate::attn_back::Mode::Columns,
        };
        let cb = cbuf(&a).unwrap();
        let u32_at = |o: usize| u32::from_le_bytes(cb[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
        assert_eq!(
            [
                u64_at(0x700),
                u64_at(0x708),
                u64_at(0x710),
                u64_at(0x718),
                u64_at(0x720)
            ],
            [0x1000, 0x2000, 0x3000, 0x4000, 0x5000]
        );
        assert_eq!(
            [
                u32_at(0x728),
                u32_at(0x72c),
                u32_at(0x730),
                u32_at(0x734),
                u32_at(0x738)
            ],
            [512, 768, 12, 64, 1]
        );
        assert!(cb[..PUSH_BASE].iter().all(|&b| b == 0));
        let rows = Args {
            mode: crate::attn_back::Mode::Rows,
            ..a
        };
        assert!(cbuf(&rows).is_none());
        assert!(cbuf(&Args { c: 12 * 128, ..a }).is_none());
    }
}
