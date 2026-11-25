//! C-147 point 5 (tensor cores, T0–T1) — **preparation only, never sent to a
//! board without Adrian's separate "da"**: the first program that uses the
//! SM 8.6 tensor cores, `tools/nak/mma16_f16.comp`, compiled by the **second**
//! NAK tool (Mesa 26.3.1, `tools/nak/build2.sh`).
//!
//! `D[16,16] = A[16,16] · B[16,16] + C[16,16]` (A, B binary16; C, D f32;
//! row-major), one warp of 32 threads. NAK emits it as two
//! `hmma.m16n8k16.f32` (one per 8-column half), with `movm.16.m8n8.trans` to
//! put B in the order the instruction wants (`mma16_f16.sm86.txt`).
//!
//! Two differences against every program of the first tool, both read in the
//! listing and checked by the tests:
//! - the push constants are at **cbuf 0 + 0x700** (NVK 26.3.1), not 0x28
//!   (NVK 25.0.7) — [`cbuf`] writes them there, the constant buffer is 0x800 B;
//! - 32 registers per thread.
//!
//! The verdict is `gpu_compute::mma::judge` (reference in f64, tolerance
//! `17 · 2⁻²³ · (|c| + Σ|a·b|)`), written before any run: the order in which
//! the tensor core adds is not documented, so bit-for-bit is not asked for.
//!
//! Tested **only in SIMULARE** (`gpu_sim::cmdproc::mma16_f16_on_cpu`, whose
//! arithmetic is a hypothesis); nothing here has run on a board.

use gpu_core::error::{GpuError, GpuResult};

use crate::qmd::ComputeQmd;

/// `tools/nak/out2/mma16_f16.sm86.sass` (Mesa NAK 26.3.1, SM 8.6; 78 instructions, 1 248 bytes).
pub const MMA16_F16: &[u8] = include_bytes!("../../../tools/nak/out2/mma16_f16.sm86.sass");
/// SHA-256 of [`MMA16_F16`] (`sh tools/nak/build2.sh mma16_f16`, identical on two builds).
pub const MMA16_F16_SHA256: &str =
    "f8c0db68b654372d94ed96c767c7775c32909298a1b77076ddda81a66afa1ef9";
/// SHA-256 of [`MMA16_S8`], `tools/nak/out2/mma16_s8.sm86.sass` — the int8 probe (two
/// `imma.m16n8k32.i8.i8`). B1 (sesiunea 223): modelled in SIMULARE (`gpu_sim::cmdproc::mma16_s8_on_cpu`,
/// oracle `gpu_compute::mma_s8`); never sent to a board.
pub const MMA16_S8_SHA256: &str =
    "7f6bc59aad0871f1164902ad0f19aad5f89d388003f6d3afd5a4551e9e0c84af";
/// B1: `tools/nak/out2/mma16_s8.sm86.sass` (Mesa NAK 26.3.1, SM 8.6; 148 instructions, 2 368 bytes):
/// `D[16,16] = A[16,32] · B[32,16] + C[16,16]`, A/B i8, C/D i32, row-major; one warp.
pub const MMA16_S8: &[u8] = include_bytes!("../../../tools/nak/out2/mma16_s8.sm86.sass");
/// Registers per thread of [`MMA16_S8`] (dump: `Num GPRs: 32`, `Shared size: 0`).
pub const MMA16_S8_REGISTERS: u32 = 32;
/// Registers per thread (dump: `Num GPRs: 32`).
pub const MMA16_F16_REGISTERS: u32 = 32;
/// Threads in the one block (`local_size_x`): one warp.
pub const MMA16_F16_BLOCK: u32 = 32;
/// Where NVK 26.3.1 puts the push constants in cbuf 0 (dump: `ldc c[0x0][+0x700]` … `+0x71c`).
pub const PUSH_BASE: usize = 0x700;
/// Size of the constant buffer (covers 0x700 + 4 addresses, multiple of 16).
pub const CBUF_SIZE: u32 = 0x800;
/// Alignment of a, b, c, d (`buffer_reference_align = 16`; the program reads C with 64-bit loads).
pub const ALIGN: u64 = 16;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The constant buffer: zero, then the addresses of A, B, C, D at 0x700, 0x708, 0x710, 0x718.
pub fn cbuf(a: u64, b: u64, c: u64, d: u64) -> GpuResult<[u8; CBUF_SIZE as usize]> {
    if [a, b, c, d]
        .iter()
        .any(|v| *v == 0 || !v.is_multiple_of(ALIGN))
    {
        return Err(bad("mma16_f16: a, b, c, d nenule si aliniate la 16"));
    }
    let mut out = [0u8; CBUF_SIZE as usize];
    for (i, va) in [a, b, c, d].into_iter().enumerate() {
        let at = PUSH_BASE + 8 * i;
        out[at..at + 8].copy_from_slice(&va.to_le_bytes());
    }
    Ok(out)
}

/// The QMD bytes: one block of 32×1×1, 32 registers, no shared memory.
pub fn qmd(program_va: u64, cbuf_va: u64) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    ComputeQmd {
        program_va,
        registers: MMA16_F16_REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: CBUF_SIZE,
        grid: [1, 1, 1],
        block: [MMA16_F16_BLOCK, 1, 1],
        release: None,
    }
    .to_bytes()
}

/// B1: the constant buffer of [`MMA16_S8`] — the same layout as [`cbuf`] (NVK 26.3.1 push constants at
/// 0x700: a, b, c, d; the dump reads A with 4-byte loads, B by byte, C with 8-byte loads).
pub fn cbuf_s8(a: u64, b: u64, c: u64, d: u64) -> GpuResult<[u8; CBUF_SIZE as usize]> {
    cbuf(a, b, c, d).map_err(|_| bad("mma16_s8: a, b, c, d nenule si aliniate la 16"))
}

/// B1: the QMD of [`MMA16_S8`]: one block of 32×1×1, 32 registers, no shared memory.
pub fn qmd_s8(program_va: u64, cbuf_va: u64) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    ComputeQmd {
        program_va,
        registers: MMA16_S8_REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: CBUF_SIZE,
        grid: [1, 1, 1],
        block: [MMA16_F16_BLOCK, 1, 1],
        release: None,
    }
    .to_bytes()
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
        assert_eq!(MMA16_F16.len(), 78 * 16);
        assert_eq!(hex(&sha256(MMA16_F16)), MMA16_F16_SHA256);
    }

    #[test]
    fn cbuf_puts_the_addresses_at_0x700() {
        let cb = cbuf(0x1000, 0x2000, 0x3000, 0x4000).unwrap();
        assert!(cb[..PUSH_BASE].iter().all(|&b| b == 0));
        let at = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
        assert_eq!(
            [at(0x700), at(0x708), at(0x710), at(0x718)],
            [0x1000, 0x2000, 0x3000, 0x4000]
        );
        assert!(cbuf(0x1008, 0x2000, 0x3000, 0x4000).is_err());
        assert!(cbuf(0, 0x2000, 0x3000, 0x4000).is_err());
    }

    #[test]
    fn s8_program_is_the_compiled_one() {
        assert_eq!(MMA16_S8.len(), 148 * 16);
        assert_eq!(hex(&sha256(MMA16_S8)), MMA16_S8_SHA256);
        assert!(cbuf_s8(0x1004, 0x2000, 0x3000, 0x4000).is_err());
    }
}
