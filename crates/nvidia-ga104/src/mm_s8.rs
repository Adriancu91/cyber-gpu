//! B1 step 3 (`docs/B1_MM_S8.md`, session 225): the int8 matmul on the tensor cores, on 128×128 tiles
//! (`tools/nak/mm_s8.comp`, compiled by the **second** NAK tool, Mesa 26.3.1, for SM 8.6; `imma.m16n8k32.i8.i8`).
//!
//! `C[m, n] = op(A)[m, k] · op(B)[k, n] (+ bias[n])` with A, B signed 8-bit integers and C, bias signed 32-bit
//! integers. **The same constant buffer as `mm_tc`** ([`tc::Args`], push constants at 0x700, flags TA / TB / BIAS —
//! TB = B stored [n, k], the Q8_0 weight layout) and **the same launch as `mm_tc128`** (blocks of 256 threads, one
//! 128×128 piece of C per block), 16 KiB of shared memory. Integer products are exact and i32 addition modulo 2³²
//! does not depend on the order, so the verdict is **bit for bit** against `gpu_compute::mma_s8::mm_s8_ref` while
//! the result fits in i32. The CPU model is `gpu_sim::cmdproc::mm_s8_on_cpu`. Tested **only in SIMULARE** (T231);
//! it goes to the board only after the `mma16_s8` probe has passed there.

use crate::mm_tc as tc;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out2/mm_s8.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const MM_S8: &[u8] = include_bytes!("../../../tools/nak/out2/mm_s8.sm86.sass");
/// SHA-256 of [`MM_S8`] (`sh tools/nak/build2.sh mm_s8`; checked by a test, written in the note).
pub const MM_S8_SHA256: &str = "76ce12d512ebf563657b2d4b16356f07405d42995c75743a503eec296dca92c6";
/// Registers per thread (NAK dump `mm_s8.sm86.txt`).
pub const REGISTERS: u32 = 104;
/// Threads per block: 256 × 1 × 1 (8 subgroups of 32).
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Rows and columns of C per block.
pub const BLOCK_MN: u32 = 128;
/// Shared memory: the i8 slices of op(A) (128×32) and op(B) (32×128) + 8 scratches of 16×16 i32; one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 128 * 32 + 32 * 128 + 8 * 16 * 16 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Bytes of constant buffer 0: as `mm_tc` (push constants at 0x700).
pub const CBUF_BYTES: usize = tc::CBUF_BYTES;

/// Constant buffer 0 for one product: exactly [`tc::Args::cbuf`] (the same layout; `a`, `b` point at i8 data,
/// `c`, `bias` at i32 data). `None` for an empty product, unknown flags, BIAS without an address, or an address
/// not aligned to 4.
pub fn cbuf(args: &tc::Args) -> Option<[u8; CBUF_BYTES]> {
    args.cbuf()
}

/// The grid for one product: one 256-thread block per 128×128 piece of C.
pub fn grid(args: &tc::Args) -> [u32; 3] {
    [args.n.div_ceil(BLOCK_MN), args.m.div_ceil(BLOCK_MN), 1]
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
        assert_eq!(MM_S8.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_S8)), MM_S8_SHA256);
    }

    #[test]
    fn launch_shape() {
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
        assert_eq!(grid(&a), [18, 8, 1]);
        assert_eq!(grid(&tc::Args { m: 129, n: 1, ..a }), [1, 2, 1]);
        assert_eq!(cbuf(&a), a.cbuf());
        const { assert!(SHARED.bytes <= MAX_SHARED_BYTES) };
        const { assert!(BLOCK[0] * REGISTERS <= 65_536) };
    }
}
