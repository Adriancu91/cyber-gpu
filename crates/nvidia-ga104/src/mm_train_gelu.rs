//! Order 20:11, step 3, second part (`docs/C-160_P7_LIPIRE_GELU.md`): `mm_train` with GELU's backward fused into the
//! store (`tools/nak/mm_train_gelu.comp`, compiled by Mesa NAK 25.0.7 for SM 8.6). The loop, tiles and order of the
//! fused multiply-adds are [`crate::mm_train`]'s; each finished sum `acc` is written as
//! `acc · (Φ(u) + u·φ(u))` with exactly `gelu_tr` mode 1's operations (u read at `[row, col]`).
//!
//! In the step it replaces "inapoi:proj-dX" (`mm_train`) + "inapoi:gelu" (`gelu_tr` mode 1): one launch and two
//! passes over `[rows, 4C]` fewer per layer. The CPU model is `gpu_sim::cmdproc::mm_train_gelu_on_cpu`. Tested
//! **only in SIMULARE** (T201).

use crate::mm_train as mm;
use crate::qmd::SharedMemory;

/// `tools/nak/out/mm_train_gelu.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const MM_TRAIN_GELU: &[u8] = include_bytes!("../../../tools/nak/out/mm_train_gelu.sm86.sass");
/// SHA-256 of [`MM_TRAIN_GELU`] (checked by a test; written in the note).
pub const MM_TRAIN_GELU_SHA256: &str =
    "ee7e3145e1c26dea908f376119ad8f6fef284d392a1428ee8eba5d9b7ad0b053";
/// Registers per thread (NAK dump `mm_train_gelu.sm86.txt`).
pub const REGISTERS: u32 = 19;
/// Threads per block: as `mm_train`, 16 × 16.
pub const BLOCK: [u32; 3] = mm::BLOCK;
/// Shared memory: as `mm_train` (two 16×16 f32 tiles, one barrier).
pub const SHARED: SharedMemory = mm::SHARED;
/// Bytes of constant buffer 0 the program reads: as `mm_train` (push constants end at 0x58).
pub const CBUF_BYTES: usize = mm::CBUF_BYTES;

/// One product `C = GELU'(u) ⊙ (op(A)·op(B))`; `C`, `u` are `[m, n]`.
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
    /// [`mm::TA`] / [`mm::TB`] only (no bias).
    pub flags: u32,
}

impl Args {
    /// Constant buffer 0: `mm_train`'s layout with `u` at the bias slot (0x40). `None` for an empty product, a flag
    /// other than TA / TB, a missing address, or `c == u`.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.flags & !(mm::TA | mm::TB) != 0 {
            return None;
        }
        if [self.a, self.b, self.c, self.u].contains(&0) || self.c == self.u {
            return None;
        }
        mm::Args {
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

    /// The grid: as `mm_train`, one 16×16 block per 16×16 tile of C.
    pub fn grid(&self) -> [u32; 3] {
        [self.n.div_ceil(16), self.m.div_ceil(16), 1]
    }
}
