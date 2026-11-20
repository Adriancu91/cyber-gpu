//! C-147 step 3 (stage 4) — `q8_matvec` with all 32 threads at work:
//! y = W·x with W in Q8_0 (KGR1), the program `tools/nak/q8_matvec32.comp`
//! compiled by Mesa NAK 25.0.7 on the host (SM 8.6).
//!
//! What changes against `q8_matvec` (C-83 I12), nothing else:
//! - the unit of work is a quarter block (8 columns), so at 768 inputs all 32
//!   threads of a row work (3 units each) instead of 24 (one block each);
//! - in one step the 32 threads read 8 whole consecutive blocks (288 bytes
//!   without gaps), instead of 32 words 36 bytes apart;
//! - x is read with 128-bit loads (`ld.global.constant.b128`, two per unit),
//!   so x must be **16-byte aligned** (the OS's blob parts and activation
//!   vectors already are); the 36-byte blocks are not 16-aligned, so the
//!   numbers stay 32-bit loads (a 128-bit read of W would need another layout
//!   of the weights — not done, KGR1 stays as Kronos writes it).
//!
//! Same push constants (cbuf 0x28 … 0x44), same launch shape (grid x = rows,
//! 32×1×1), same 128 bytes of shared memory and no `BAR.SYNC`: a launch slot of
//! `q8_matvec` becomes one of `q8_matvec32` by changing the program address
//! and the register count in the QMD.
//!
//! The result differs from `q8_matvec`'s in the last bits (another summation
//! order). The verdict ([`judge`], written before any run): **identic** = bit
//! for bit with `gpu_compute::q8::matvec_gpu_order_w32` (the order NAK emits,
//! read in `q8_matvec32.sm86.txt`); **in marja** = within the same margin of
//! the CPU engine as I12; otherwise **nu trece**.
//!
//! Tested **only in SIMULARE**; nothing here has run on a board.

use gpu_compute::q8;
use gpu_core::error::{GpuError, GpuResult};

use crate::q8_matvec::{self as mv, Verdict};
use crate::qmd::{ComputeQmd, MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out/q8_matvec32.sm86.sass` (Mesa NAK 25.0.7, SM 8.6; 199 instructions,
/// SHA-256 `86ae584b…`).
pub const Q8_MATVEC32: &[u8] = include_bytes!("../../../tools/nak/out/q8_matvec32.sm86.sass");
/// SHA-256 of [`Q8_MATVEC32`] (`sh tools/nak/build.sh q8_matvec32`).
pub const Q8_MATVEC32_SHA256: &str =
    "86ae584bd079454150a585968c2ea63ef2b7dadec31054d21a6f59d6afc7c303";
/// Registers per thread (NAK dump `q8_matvec32.sm86.txt`: 22 GPRs).
pub const Q8_MATVEC32_REGISTERS: u32 = 22;
/// Threads per row (`local_size_x`), as `q8_matvec`.
pub const Q8_MATVEC32_BLOCK: u32 = 32;
/// Shared memory: the 32 partial sums, as `q8_matvec`; no `BAR.SYNC` in the dump.
pub const Q8_MATVEC32_SHARED: SharedMemory = SharedMemory {
    bytes: 128,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 0,
};
/// Alignment `q8_matvec32` needs for x (vec4 loads).
pub const X_ALIGN: u64 = 16;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The constant buffer: the one of `q8_matvec` (same offsets), plus the check that x is 16-aligned.
pub fn cbuf(
    w: u64,
    x: u64,
    y: u64,
    rows: usize,
    cols: usize,
) -> GpuResult<[u8; mv::CBUF_SIZE as usize]> {
    if !x.is_multiple_of(X_ALIGN) {
        return Err(bad("q8_matvec32: x trebuie aliniat la 16"));
    }
    mv::q8_cbuf(w, x, y, rows, cols)
}

/// The QMD bytes: grid x = rows, 32×1×1, 22 registers, 128 B shared.
pub fn qmd(program_va: u64, cbuf_va: u64, rows: usize) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    if rows == 0 || rows > i32::MAX as usize {
        return Err(bad("Q8_0: numar de randuri invalid"));
    }
    ComputeQmd {
        program_va,
        registers: Q8_MATVEC32_REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: mv::CBUF_SIZE,
        grid: [rows as u32, 1, 1],
        block: [Q8_MATVEC32_BLOCK, 1, 1],
        release: None,
    }
    .to_bytes_with_shared(&Q8_MATVEC32_SHARED)
}

/// Judges the board's `got` for W·x computed by `q8_matvec32` (module doc).
pub fn judge(w: &[u8], rows: usize, cols: usize, x: &[f32], got: &[f32]) -> GpuResult<Verdict> {
    let mut oracle = alloc::vec![0.0f32; rows];
    let mut cpu = alloc::vec![0.0f32; rows];
    q8::matvec_gpu_order_w32(w, rows, cols, x, &mut oracle)?;
    q8::matvec_cpu(w, rows, cols, x, &mut cpu)?;
    let m = q8::margin(w, rows, cols, x)?;
    if let Some((row, got, cpu, margin)) = q8::first_outside_margin(got, &cpu, &m) {
        return Ok(Verdict::NuTrece {
            row,
            got,
            cpu,
            margin,
        });
    }
    let mut diff = (0..rows).filter(|&r| got[r].to_bits() != oracle[r].to_bits());
    Ok(match diff.next() {
        None => Verdict::Identic,
        Some(first) => Verdict::InMarja {
            differing: 1 + diff.count(),
            first,
        },
    })
}

/// What one row of W costs in global reads, per program, for `cols` inputs
/// (C-147 step 3, counted from the two listings — a paper figure, not a
/// measurement): `(load instructions per warp, bytes the warp asks for)`.
/// `q8_matvec`: per block 9 word loads per thread (scale + 8 words) + 8 for x;
/// `q8_matvec32`: per unit 3 word loads (scale + 2 words) + 2 × 128-bit for x.
pub fn reads_per_row(cols: usize) -> ((usize, usize), (usize, usize)) {
    let blocks = cols / q8::BLOCK;
    let old_steps = blocks.div_ceil(q8::GPU_THREADS);
    let new_steps = (blocks * q8::W32_QUARTERS).div_ceil(q8::GPU_THREADS);
    let bytes = blocks * q8::BLOCK_BYTES + blocks * q8::BLOCK * 4;
    (
        (old_steps * (9 + 32), bytes),
        (new_steps * (3 + 2), bytes + blocks * 3 * 4),
    )
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
        assert_eq!(Q8_MATVEC32.len(), 199 * 16);
        assert_eq!(hex(&sha256(Q8_MATVEC32)), Q8_MATVEC32_SHA256);
        assert_ne!(Q8_MATVEC32, mv::Q8_MATVEC);
    }

    #[test]
    fn cbuf_like_q8_matvec_but_x_16_aligned() {
        let a = cbuf(0x1000, 0x2000, 0x3004, 3072, 768).unwrap();
        assert_eq!(a, mv::q8_cbuf(0x1000, 0x2000, 0x3004, 3072, 768).unwrap());
        assert!(cbuf(0x1000, 0x2004, 0x3000, 1, 32).is_err());
        assert!(cbuf(0x1002, 0x2000, 0x3000, 1, 32).is_err());
        let q = qmd(0x10_0000, 0x10_1000, 3072).unwrap();
        let old = mv::q8_qmd(0x10_0000, 0x10_1000, 3072).unwrap();
        // Only the register count differs from q8_matvec's QMD.
        assert_ne!(q, old);
        let same = ComputeQmd {
            program_va: 0x10_0000,
            registers: mv::Q8_REGISTERS,
            cbuf0_va: 0x10_1000,
            cbuf0_size: mv::CBUF_SIZE,
            grid: [3072, 1, 1],
            block: [32, 1, 1],
            release: None,
        }
        .to_bytes_with_shared(&Q8_MATVEC32_SHARED)
        .unwrap();
        assert_eq!(same, old);
        assert!(qmd(0, 0, 0).is_err());
    }

    #[test]
    fn verdicts() {
        let (w, x) = mv::test_matrix(40, 768, 9).unwrap();
        let mut y = alloc::vec![0.0f32; 40];
        q8::matvec_gpu_order_w32(&w, 40, 768, &x, &mut y).unwrap();
        assert_eq!(judge(&w, 40, 768, &x, &y).unwrap(), Verdict::Identic);
        // q8_matvec's result: in the margin, and not the new program's bits.
        let mut old = alloc::vec![0.0f32; 40];
        q8::matvec_gpu_order(&w, 40, 768, &x, &mut old).unwrap();
        assert!(matches!(
            judge(&w, 40, 768, &x, &old).unwrap(),
            Verdict::InMarja { .. }
        ));
        let mut wrong = y.clone();
        wrong[3] = -wrong[3] + 1.0;
        assert!(!judge(&w, 40, 768, &x, &wrong).unwrap().passes());
        assert!(!judge(&w, 40, 768, &x, &y[..39]).unwrap().passes());
    }

    #[test]
    fn paper_reads() {
        // 768 inputs: 24 blocks. q8_matvec: 1 step × 41 loads; q8_matvec32: 3 steps × 5 loads.
        let ((ol, ob), (nl, nb)) = reads_per_row(768);
        assert_eq!((ol, nl), (41, 15));
        assert_eq!(ob, 24 * 36 + 768 * 4);
        assert_eq!(nb, ob + 24 * 12);
        let ((ol, _), (nl, _)) = reads_per_row(3072);
        assert_eq!((ol, nl), (3 * 41, 12 * 5));
    }
}
