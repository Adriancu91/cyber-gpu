//! C-157 (`docs/C-157_ATTN_BACK0_TC.md`, plan and predictions E1–E7 pushed before this code, `57b4079`):
//! [`crate::attn_back_tile0_3`]'s job (mode 0: dP = dO·Vᵀ, dot = Σ P·dP, dS = P·(dP − dot)·scale) with **only the
//! dP product on the tensor cores** (`tools/nak/attn_back_tile0_tc.comp`, compiled by the second NAK tool, Mesa 26.3.1,
//! SM 8.6): every element of dO and V rounded to binary16 in shared memory, the products added in f32
//! (`hmma.m16n8k16.f32`); the dot chain (`ffma.ftz`) and pass 2 are `attn_back_tile0_3`'s, in the same order.
//!
//! **Same arguments** ([`crate::attn_back::Args`] with [`crate::attn_back::Mode::Rows`]), **same grid, block, shared
//! size and result places** as `attn_back_tile0_3`; the arguments sit at **0x700** in cbuf 0, as every second-tool
//! program ([`cbuf`]). The numbers are NOT the same bits as the F32 program: the CPU model is
//! `gpu_compute::attn_tc::back0_model`, the verdict on dP `gpu_compute::attn_tc::judge_dp`. The mixed #513 step
//! already carries the error scale 2¹⁴ in dO. Tested **only in SIMULARE** (T254). Nothing here has run on the board.

use crate::attn_back::Args;
use crate::qmd::SharedMemory;

/// `tools/nak/out2/attn_back_tile0_tc.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const ATTN_BACK_TILE0_TC: &[u8] =
    include_bytes!("../../../tools/nak/out2/attn_back_tile0_tc.sm86.sass");
/// SHA-256 of [`ATTN_BACK_TILE0_TC`] (`sh tools/nak/build2.sh attn_back_tile0_tc`; checked by a test, in the note).
pub const ATTN_BACK_TILE0_TC_SHA256: &str =
    "95fc392e6de3cf3bc2f5f93a33487e2ebefdedf38951f35ab1a9168540c1ef8d";
/// Registers per thread (NAK dump `attn_back_tile0_tc.sm86.txt`: `Num GPRs: 56`; `attn_back_tile0_3` has 68).
pub const REGISTERS: u32 = 56;
/// Threads per block: 256, as `attn_back_tile0_3`.
pub const BLOCK: [u32; 3] = crate::attn_back_tile0::BLOCK;
/// The largest head width.
pub const MAX_D: u32 = crate::attn_back_tile0::MAX_D;
/// Shared memory: the f16 dO / V tiles (16 KiB, later the f32 P tile) and the f32 dP tile 64 × 65 = 33 024 bytes,
/// the same as `attn_back_tile0_3` (the dump's `Shared size`); one barrier.
pub const SHARED: SharedMemory = crate::attn_back_tile0_3::SHARED;
/// Where the second NAK tool puts the push constants in cbuf 0 (the dump reads `c[0x0][0x700]` … `c[0x0][0x73c]`).
pub const PUSH_BASE: usize = crate::attn_back_tile_tc::PUSH_BASE;
/// Size of constant buffer 0 (as `attn_back_tile_tc`).
pub const CBUF_BYTES: usize = crate::attn_back_tile_tc::CBUF_BYTES;

/// Constant buffer 0 for this program: zero up to 0x700, then the bytes 0x28..0x68 of
/// [`crate::attn_back::Args::cbuf`] at 0x700..0x740. `None` if `attn_back` refuses the arguments or the launch does
/// not fit ([`shape_ok`]: mode 0, d ≤ 64).
pub fn cbuf(a: &Args) -> Option<[u8; CBUF_BYTES]> {
    if !shape_ok(a) {
        return None;
    }
    cbuf_from_f32(&a.cbuf()?)
}

/// The same move for a cbuf laid out for the F32 programs (`attn_back_tile0_3`, arguments at 0x28): bytes 0x28..0x68
/// go to 0x700..0x740. `None` if `cb` is shorter than 0x68 bytes, its workgroup base (0x0..0xc) is not zero, or its
/// mode (0x60) is not 0.
pub fn cbuf_from_f32(cb: &[u8]) -> Option<[u8; CBUF_BYTES]> {
    const ARGS: usize = 0x28;
    const END: usize = crate::attn_back::CBUF_BYTES;
    if cb.len() < END || cb[..0xc].iter().any(|&b| b != 0) || cb[0x60..0x64] != 0u32.to_le_bytes() {
        return None;
    }
    let mut out = [0u8; CBUF_BYTES];
    out[PUSH_BASE..PUSH_BASE + (END - ARGS)].copy_from_slice(&cb[ARGS..END]);
    Some(out)
}

/// The grid: [⌈T/64⌉, H, B] (as `attn_back_tile0_3`).
pub fn grid(a: &Args) -> [u32; 3] {
    crate::attn_back_tile0_3::grid(a)
}

/// Whether the program takes this launch (as `attn_back_tile0_3`: mode 0, d = C / H ≤ 64).
pub fn shape_ok(a: &Args) -> bool {
    crate::attn_back_tile0_3::shape_ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::digest as sha256;

    const DUMP: &str = include_str!("../../../tools/nak/out2/attn_back_tile0_tc.sm86.txt");

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
        assert_eq!(hex(&sha256(ATTN_BACK_TILE0_TC)), ATTN_BACK_TILE0_TC_SHA256);
        assert_eq!(ATTN_BACK_TILE0_TC.len() % 16, 0);
    }

    /// E6 (docs/C-157_ATTN_BACK0_TC.md): 0 local memory, ≤ 64 registers, tensor-core products (4 steps × 2
    /// accumulators × 2 `hmma`, unrolled), `ffma` only in the dot chain (8, all into one register), the shared size
    /// of `attn_back_tile0_3`.
    #[test]
    fn dump_says_what_the_note_says() {
        assert!(DUMP.contains(&alloc::format!("Num GPRs: {REGISTERS}\n")));
        assert!(DUMP.contains("SLM size: 0\n"));
        assert!(DUMP.contains(&alloc::format!("Shared size: {}\n", SHARED.bytes)));
        assert!(DUMP.contains("Spills to mem: 0\n"));
        assert_eq!(DUMP.matches("hmma.m16n8k16.f32").count(), 16);
        let ffma: alloc::vec::Vec<&str> = DUMP.lines().filter(|l| l.contains("= ffma")).collect();
        assert_eq!(ffma.len(), 8);
        let dst = |l: &str| l.split_whitespace().next().unwrap_or("").to_owned();
        assert!(
            ffma.iter()
                .all(|l| dst(l) == dst(ffma[0]) && l.contains("ffma.ftz"))
        );
        const { assert!(REGISTERS <= 64) };
    }

    /// Every cbuf-0 word the program reads is the workgroup base (0x0 / 0x4 / 0x8) or an argument at 0x700..0x740,
    /// and [`cbuf`] puts the pointers, T, C, H, d, mode and scale where the dump reads them.
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
        // qkv, p, dout, ds, T, C, H, d, mode, scale (dqkv at 0x720 is not read: mode 0 does not write it)
        for off in [
            0x700, 0x708, 0x710, 0x718, 0x728, 0x72c, 0x730, 0x734, 0x738, 0x73c,
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
            mode: crate::attn_back::Mode::Rows,
        };
        let cb = cbuf(&a).unwrap();
        let u32_at = |o: usize| u32::from_le_bytes(cb[o..o + 4].try_into().unwrap());
        let u64_at = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
        assert_eq!(
            [u64_at(0x700), u64_at(0x708), u64_at(0x710), u64_at(0x718)],
            [0x1000, 0x2000, 0x3000, 0x4000]
        );
        assert_eq!(
            [
                u32_at(0x728),
                u32_at(0x72c),
                u32_at(0x730),
                u32_at(0x734),
                u32_at(0x738)
            ],
            [512, 768, 12, 64, 0]
        );
        assert_eq!(u32_at(0x73c), Args::scale_for(64).to_bits());
        assert!(cb[..PUSH_BASE].iter().all(|&b| b == 0));
        let cols = Args {
            mode: crate::attn_back::Mode::Columns,
            ..a
        };
        assert!(cbuf(&cols).is_none());
        assert!(cbuf(&Args { c: 12 * 128, ..a }).is_none());
        assert_eq!(grid(&a), [8, 12, 2]);
    }
}
