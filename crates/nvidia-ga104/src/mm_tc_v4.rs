//! C-157 p.2 after os-423 (`docs/C-157_P2_CITIRI_128.md`, session 230): [`crate::mm_tc_pf`] with **128-bit** global
//! loads (`tools/nak/mm_tc_v4.comp`, compiled by the **second** NAK tool, Mesa 26.3.1, for SM 8.6).
//!
//! Session 229 found that NAK keeps at most ~5 global loads in flight; `mm_tc_pf` issues 32 per K-slice. Here every
//! load brings 4 neighbouring floats: 4 + 4 loads per slice. **Only for aligned products** ([`fits`]): a misaligned
//! 128-bit load is a fault on the board, so the host launches `mm_tc_pf` when [`fits`] says no.
//!
//! **Same job, same constant buffer ([`tc::Args::cbuf`], push constants at 0x700), same launch (128 threads, 64×64
//! tiles, [`tc::Args::grid`]), same shared memory and the same arithmetic** as `mm_tc`. The CPU model is
//! `gpu_sim::cmdproc::mm_tc_v4_on_cpu` (refuses unaligned products, otherwise `mm_tc`'s). Tested **only in SIMULARE**
//! (T236).

use crate::mm_tc as tc;
use crate::qmd::SharedMemory;

/// `tools/nak/out2/mm_tc_v4.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const MM_TC_V4: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v4.sm86.sass");
/// SHA-256 of [`MM_TC_V4`] (`sh tools/nak/build2.sh mm_tc_v4`; checked by a test, written in the note).
pub const MM_TC_V4_SHA256: &str =
    "488ca341fd5bf400cbaf41eebe65998c5f7ae167b8c5443828ef451e1967acf1";
/// Registers per thread (NAK dump `mm_tc_v4.sm86.txt`).
pub const REGISTERS: u32 = 128;
/// Threads per block: 128 × 1 × 1, as `mm_tc`.
pub const BLOCK: [u32; 3] = tc::BLOCK;
/// Rows and columns of C per block, as `mm_tc`.
pub const BLOCK_MN: u32 = tc::BLOCK_MN;
/// Shared memory: exactly `mm_tc`'s (24 KiB, one barrier).
pub const SHARED: SharedMemory = tc::SHARED;
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// May `mm_tc_v4` run this product? The stored row length of A (`k`, or `m` with [`tc::TA`]) and of B (`n`, or `k`
/// with [`tc::TB`]) must be a multiple of 4 and the addresses of A and B multiples of 16 bytes: then every 128-bit
/// load is aligned and lies wholly inside or wholly outside its matrix. `false` → launch `mm_tc_pf` instead.
pub fn fits(args: &tc::Args) -> bool {
    let a_row = if args.flags & tc::TA != 0 {
        args.m
    } else {
        args.k
    };
    let b_row = if args.flags & tc::TB != 0 {
        args.k
    } else {
        args.n
    };
    a_row.is_multiple_of(4)
        && b_row.is_multiple_of(4)
        && args.a.is_multiple_of(16)
        && args.b.is_multiple_of(16)
}

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

    fn args(m: u32, n: u32, k: u32, flags: u32) -> tc::Args {
        tc::Args {
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            bias: 0,
            m,
            n,
            k,
            flags,
        }
    }

    #[test]
    fn program_is_the_compiled_one() {
        assert_eq!(MM_TC_V4.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_TC_V4)), MM_TC_V4_SHA256);
    }

    #[test]
    fn fits_the_513_shapes_and_refuses_the_rest() {
        // #513: qkv 1024x2304x768 (B^T), the MLP 1024x3072x768 / 1024x768x3072, the backward ones with TA.
        for (m, n, k, f) in [
            (1024, 2304, 768, tc::TB),
            (1024, 3072, 768, 0),
            (1024, 768, 3072, tc::TB),
            (768, 2304, 1024, tc::TA),
        ] {
            assert!(fits(&args(m, n, k, f)), "{m}x{n}x{k} flags {f}");
        }
        assert!(!fits(&args(64, 64, 33, 0)), "k not a multiple of 4");
        assert!(
            fits(&args(64, 64, 33, tc::TA)),
            "k may be odd with TA and no TB"
        );
        assert!(
            !fits(&args(64, 64, 33, tc::TA | tc::TB)),
            "k not a multiple of 4 with TB"
        );
        assert!(
            !fits(&args(65, 64, 32, tc::TA)),
            "m not a multiple of 4 with TA"
        );
        assert!(!fits(&args(64, 63, 32, 0)), "n not a multiple of 4");
        let mut a = args(64, 64, 32, 0);
        a.a += 4;
        assert!(!fits(&a), "A not 16-byte aligned");
        let mut b = args(64, 64, 32, 0);
        b.b += 8;
        assert!(!fits(&b), "B not 16-byte aligned");
        const { assert!(BLOCK[0] * REGISTERS <= 65_536) };
    }
}
