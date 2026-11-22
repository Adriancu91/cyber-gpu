//! C-160 point 7, part two — the training matmul on the tensor cores: `C = op(A)·op(B) (+ bias)`
//! (`tools/nak/mm_tc.comp`, compiled by the **second** NAK tool, Mesa 26.3.1, for SM 8.6). The same interface as
//! [`crate::mm_train`] (A, B, C, bias in f32; flags [`TA`], [`TB`], [`BIAS`]), so it can take its place in a training
//! step list, but:
//! - op(A) and op(B) are **rounded to binary16** when copied to shared memory and the products are added in f32 by
//!   the tensor cores (`HMMA.16816.F32`): the answer is **not** the same numbers as `mm_train` / PyTorch-F32; the
//!   verdict is `gpu_compute::mm_tc::judge` (reference on the rounded inputs, written margin) and, against the f32
//!   training oracle, `gpu_compute::mm_tc::worst_vs_f32` (margin + the written price of the rounding);
//! - the push constants are at **cbuf 0 + 0x700** (NVK 26.3.1, as for [`crate::mma16`]), not 0x28;
//! - blocks of 128 threads (4 subgroups), each computing a 64×64 block of C; 24 KiB of shared memory, one barrier.
//!
//! Inputs must fit binary16 (|x| ≤ 65 504) and values under ~6·10⁻⁸ become 0: gradients need scaling before this
//! program (the next step of point 7). Tested **only in SIMULARE** (T168, `gpu_sim::cmdproc::mm_tc_on_cpu`, whose
//! tensor-core arithmetic is a hypothesis fitted on os-335); nothing here has run on a board.

use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out2/mm_tc.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const MM_TC: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc.sm86.sass");
/// SHA-256 of [`MM_TC`] (`sh tools/nak/build2.sh mm_tc`, identical on two builds).
pub const MM_TC_SHA256: &str = "f60cd944ab67ae17ddc0d61931085791dc25f76397de82070fe0b0f1c3821ee3";
/// Registers per thread (NAK dump `mm_tc.sm86.txt`).
pub const REGISTERS: u32 = 64;
/// Threads per block: 128 × 1 × 1 (4 subgroups of 32).
pub const BLOCK: [u32; 3] = [128, 1, 1];
/// Rows and columns of C per block.
pub const BLOCK_MN: u32 = 64;
/// Shared memory: the f16 slices of op(A) (64×32) and op(B) (32×64) + the 64×64 f32 block; one barrier.
pub const SHARED: SharedMemory = SharedMemory {
    bytes: (64 * 32 + 32 * 64) * 2 + 64 * 64 * 4,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// A is stored transposed (`[K, M]`).
pub const TA: u32 = 1;
/// B is stored transposed (`[N, K]`, PyTorch's `weight [out, in]`).
pub const TB: u32 = 2;
/// Add `bias[col]` after the sum.
pub const BIAS: u32 = 4;
/// Where NVK 26.3.1 puts the push constants in cbuf 0 (dump: `c[0x0][0x700]` …).
pub const PUSH_BASE: usize = 0x700;
/// Size of constant buffer 0 (covers 0x700 + the 0x30 bytes of arguments, multiple of 256).
pub const CBUF_BYTES: usize = 0x800;

/// One product: GPU virtual addresses, the shape of `C = op(A)[m, k]·op(B)[k, n]` and the flags.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Args {
    pub a: u64,
    pub b: u64,
    pub c: u64,
    /// Read only with [`BIAS`]; 0 otherwise.
    pub bias: u64,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub flags: u32,
}

impl Args {
    /// Constant buffer 0 as NVK 26.3.1 lays it out for this program (NAK dump): zero up to 0x700, then a/b/c/bias at
    /// 0x700/0x708/0x710/0x718 and m/n/k/flags at 0x720/0x724/0x728/0x72c. `None` for an empty product, unknown flag
    /// bits, a bias flag without a bias address, or an address not aligned to 4.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.m == 0 || self.n == 0 || self.k == 0 {
            return None;
        }
        if self.flags & !(TA | TB | BIAS) != 0 || (self.flags & BIAS != 0 && self.bias == 0) {
            return None;
        }
        if [self.a, self.b, self.c, self.bias]
            .iter()
            .any(|v| !v.is_multiple_of(4))
            || self.a == 0
            || self.b == 0
            || self.c == 0
        {
            return None;
        }
        let mut cb = [0u8; CBUF_BYTES];
        for (i, va) in [self.a, self.b, self.c, self.bias].into_iter().enumerate() {
            let at = PUSH_BASE + 8 * i;
            cb[at..at + 8].copy_from_slice(&va.to_le_bytes());
        }
        for (i, v) in [self.m, self.n, self.k, self.flags].into_iter().enumerate() {
            let at = PUSH_BASE + 0x20 + 4 * i;
            cb[at..at + 4].copy_from_slice(&v.to_le_bytes());
        }
        Some(cb)
    }

    /// The grid: one 128-thread block per 64×64 block of C.
    pub fn grid(&self) -> [u32; 3] {
        [self.n.div_ceil(BLOCK_MN), self.m.div_ceil(BLOCK_MN), 1]
    }

    /// Multiply-adds of the product (2·m·n·k operations): for the speed rows (DEDUS until measured).
    pub fn flop(&self) -> u64 {
        2 * self.m as u64 * self.n as u64 * self.k as u64
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
        assert_eq!(MM_TC.len() % 16, 0);
        assert_eq!(hex(&sha256(MM_TC)), MM_TC_SHA256);
    }

    #[test]
    fn cbuf_layout_and_refusals() {
        let ok = Args {
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            bias: 0x4000,
            m: 70,
            n: 130,
            k: 33,
            flags: TB | BIAS,
        };
        let cb = ok.cbuf().unwrap();
        assert!(cb[..PUSH_BASE].iter().all(|&b| b == 0));
        let u64at = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
        let u32at = |o: usize| u32::from_le_bytes(cb[o..o + 4].try_into().unwrap());
        assert_eq!(
            [u64at(0x700), u64at(0x708), u64at(0x710), u64at(0x718)],
            [0x1000, 0x2000, 0x3000, 0x4000]
        );
        assert_eq!(
            [u32at(0x720), u32at(0x724), u32at(0x728), u32at(0x72c)],
            [70, 130, 33, TB | BIAS]
        );
        assert_eq!(ok.grid(), [3, 2, 1]);
        assert!(Args { flags: 8, ..ok }.cbuf().is_none());
        assert!(Args { bias: 0, ..ok }.cbuf().is_none());
        assert!(Args { k: 0, ..ok }.cbuf().is_none());
        assert!(Args { a: 0x1002, ..ok }.cbuf().is_none());
        assert_eq!(ok.flop(), 2 * 70 * 130 * 33);
    }
}
