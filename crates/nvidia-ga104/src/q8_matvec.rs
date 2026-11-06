//! C-83 I12 — Kronos's basic operation on the board: y = W·x with W in Q8_0
//! (KGR1: per row, blocks of 36 bytes = f32 scale + 32 × i8), the program
//! `tools/nak/q8_matvec.comp` compiled by Mesa NAK on the host (DECIZII
//! 26 oct). One workgroup of 32 threads per row (one warp; NAK needed no
//! hardware barrier), 128 bytes of shared memory for the 32 partial sums.
//!
//! The verdict ([`judge`]) is written before any run:
//! - **identic**: every row equal bit for bit to
//!   `gpu_compute::q8::matvec_gpu_order` (the program's own order and
//!   operations, read in the NAK dump) — what we expect;
//! - **in marja**: not identical, but every row within
//!   `gpu_compute::q8::margin` of the CPU engine's order — passes, and the
//!   difference from the oracle is reported for analysis (e.g. the board
//!   flushing subnormals differently);
//! - otherwise **nu trece**, naming the first row.
//!
//! The test input for the window ([`test_matrix`]) is synthetic and
//! deterministic (no file needed): a 3072 × 768 matrix (the shape of
//! Kronos's `mlp.0` at n_emb 768), quantized with the exporter's rule.
//! Tested **only in SIMULARE**; nothing here touches a board.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::q8;
use gpu_core::error::{GpuError, GpuResult};

use crate::qmd::{ComputeQmd, MAX_SHARED_BYTES, SharedMemory};
use crate::transfer::pattern_word;

/// `tools/nak/out/q8_matvec.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const Q8_MATVEC: &[u8] = include_bytes!("../../../tools/nak/out/q8_matvec.sm86.sass");
/// Registers per thread (NAK dump `q8_matvec.sm86.txt`: 19 GPRs).
pub const Q8_REGISTERS: u32 = 19;
/// Threads per row (`local_size_x`).
pub const Q8_BLOCK: u32 = 32;
/// Shared memory: 32 f32 partial sums; no `BAR.SYNC` in the dump.
pub const Q8_SHARED: SharedMemory = SharedMemory {
    bytes: 128,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 0,
};
/// Push constants in cbuf 0 (NVK, as for the other programs): W, x, y
/// (64-bit GPU VAs), rows, blocks per row (32-bit).
pub const CBUF_W: usize = 0x28;
pub const CBUF_X: usize = 0x30;
pub const CBUF_Y: usize = 0x38;
pub const CBUF_ROWS: usize = 0x40;
pub const CBUF_BLOCKS: usize = 0x44;
pub const CBUF_SIZE: u32 = 0x100;
/// The window's test shape: Kronos `mlp.0` at n_emb 768.
pub const TEST_ROWS: usize = 3072;
pub const TEST_COLS: usize = 768;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The constant buffer for y[rows] = W[rows × cols] · x[cols].
pub fn q8_cbuf(
    w: u64,
    x: u64,
    y: u64,
    rows: usize,
    cols: usize,
) -> GpuResult<[u8; CBUF_SIZE as usize]> {
    q8::matrix_bytes(rows, cols)?;
    if rows == 0 || rows > i32::MAX as usize || cols / q8::BLOCK > u32::MAX as usize {
        return Err(bad("Q8_0: numar de randuri invalid"));
    }
    if !(w | x | y).is_multiple_of(4) {
        return Err(bad("Q8_0: adrese nealiniate la 4"));
    }
    let mut cb = [0u8; CBUF_SIZE as usize];
    cb[CBUF_W..CBUF_W + 8].copy_from_slice(&w.to_le_bytes());
    cb[CBUF_X..CBUF_X + 8].copy_from_slice(&x.to_le_bytes());
    cb[CBUF_Y..CBUF_Y + 8].copy_from_slice(&y.to_le_bytes());
    cb[CBUF_ROWS..CBUF_ROWS + 4].copy_from_slice(&(rows as u32).to_le_bytes());
    cb[CBUF_BLOCKS..CBUF_BLOCKS + 4].copy_from_slice(&((cols / q8::BLOCK) as u32).to_le_bytes());
    Ok(cb)
}

/// The QMD bytes (with the shared memory of [`Q8_SHARED`]): grid x = rows.
pub fn q8_qmd(
    program_va: u64,
    cbuf_va: u64,
    rows: usize,
) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    if rows == 0 || rows > i32::MAX as usize {
        return Err(bad("Q8_0: numar de randuri invalid"));
    }
    ComputeQmd {
        program_va,
        registers: Q8_REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: CBUF_SIZE,
        grid: [rows as u32, 1, 1],
        block: [Q8_BLOCK, 1, 1],
        release: None,
    }
    .to_bytes_with_shared(&Q8_SHARED)
}

/// The window's input, from the I11 pattern (no file): W (`rows × cols`,
/// values in about ±0.05, quantized as the exporter does) and x (about ±1).
pub fn test_matrix(rows: usize, cols: usize, seed: u32) -> GpuResult<(Vec<u8>, Vec<f32>)> {
    let rb = q8::matrix_bytes(1, cols)?;
    let mut w = alloc::vec![0u8; q8::matrix_bytes(rows, cols)?];
    let f = |i: u64| (pattern_word(seed, i) >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0;
    let mut row = alloc::vec![0.0f32; cols];
    for r in 0..rows {
        for (c, v) in row.iter_mut().enumerate() {
            *v = f((r * cols + c) as u64) * 0.05;
        }
        q8::quantize_row(&row, &mut w[r * rb..(r + 1) * rb])?;
    }
    let base = (rows * cols) as u64;
    let x = (0..cols as u64).map(|c| f(base + c)).collect();
    Ok((w, x))
}

/// What the board's y says.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// Every row identical bit for bit with the program's oracle.
    Identic,
    /// Not identical (`differing` rows, first at `first`), but all within
    /// the margin of the CPU engine: passes, to be analysed.
    InMarja { differing: usize, first: usize },
    /// Row `row` outside the margin (or y has the wrong length).
    NuTrece {
        row: usize,
        got: f32,
        cpu: f32,
        margin: f64,
    },
}

impl Verdict {
    pub fn passes(&self) -> bool {
        !matches!(self, Verdict::NuTrece { .. })
    }

    /// One journal line.
    pub fn line(&self) -> String {
        match self {
            Verdict::Identic => String::from(
                "i12: TRECE - identic bit cu bit cu oracolul programului (ordinea NAK)",
            ),
            Verdict::InMarja { differing, first } => format!(
                "i12: TRECE in marja - {differing} randuri difera de oracol (primul {first}), toate in marja fata de procesor; de analizat"
            ),
            Verdict::NuTrece {
                row,
                got,
                cpu,
                margin,
            } => {
                format!(
                    "i12: NU TRECE - randul {row}: placa {got:e}, procesor {cpu:e}, marja {margin:e}"
                )
            }
        }
    }
}

/// Judges the board's `got` for W·x (module doc). `got` shorter or longer
/// than `rows` never passes.
pub fn judge(w: &[u8], rows: usize, cols: usize, x: &[f32], got: &[f32]) -> GpuResult<Verdict> {
    let mut oracle = alloc::vec![0.0f32; rows];
    let mut cpu = alloc::vec![0.0f32; rows];
    q8::matvec_gpu_order(w, rows, cols, x, &mut oracle)?;
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
    let diff: Vec<usize> = (0..rows)
        .filter(|&r| got[r].to_bits() != oracle[r].to_bits())
        .collect();
    Ok(match diff.first() {
        None => Verdict::Identic,
        Some(&first) => Verdict::InMarja {
            differing: diff.len(),
            first,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbuf_and_qmd() {
        let cb = q8_cbuf(0x1000, 0x2000, 0x3004, 3072, 768).unwrap();
        assert_eq!(
            u64::from_le_bytes(cb[0x28..0x30].try_into().unwrap()),
            0x1000
        );
        assert_eq!(
            u64::from_le_bytes(cb[0x38..0x40].try_into().unwrap()),
            0x3004
        );
        assert_eq!(u32::from_le_bytes(cb[0x40..0x44].try_into().unwrap()), 3072);
        assert_eq!(u32::from_le_bytes(cb[0x44..0x48].try_into().unwrap()), 24);
        assert!(cb[..8].iter().all(|&b| b == 0), "baza grupurilor = 0");
        assert!(q8_cbuf(0x1002, 0x2000, 0x3000, 1, 32).is_err());
        assert!(q8_cbuf(0x1000, 0x2000, 0x3000, 1, 48).is_err());
        assert!(q8_cbuf(0x1000, 0x2000, 0x3000, 0, 32).is_err());
        q8_qmd(0x10_0000, 0x10_1000, 3072).unwrap();
        assert!(q8_qmd(0x10_0000, 0x10_1000, 0).is_err());
    }

    #[test]
    fn test_matrix_is_deterministic_and_in_range() {
        let (w, x) = test_matrix(4, 64, 1).unwrap();
        assert_eq!((w.len(), x.len()), (4 * 2 * 36, 64));
        assert_eq!(test_matrix(4, 64, 1).unwrap().0, w);
        assert_ne!(test_matrix(4, 64, 2).unwrap().0, w);
        assert!(x.iter().all(|v| v.abs() <= 1.0));
        let s = f32::from_le_bytes(w[..4].try_into().unwrap());
        assert!(s > 0.0 && s <= 0.05 / 127.0 * 1.0001);
    }

    #[test]
    fn verdicts() {
        let (w, x) = test_matrix(40, 96, 5).unwrap();
        let mut y = alloc::vec![0.0f32; 40];
        q8::matvec_gpu_order(&w, 40, 96, &x, &mut y).unwrap();
        assert_eq!(judge(&w, 40, 96, &x, &y).unwrap(), Verdict::Identic);
        let mut cpu = alloc::vec![0.0f32; 40];
        q8::matvec_cpu(&w, 40, 96, &x, &mut cpu).unwrap();
        // The CPU engine's order: passes in the margin (rows that differ are named).
        match judge(&w, 40, 96, &x, &cpu).unwrap() {
            Verdict::Identic | Verdict::InMarja { .. } => {}
            v => panic!("{v:?}"),
        }
        let mut one_ulp = y.clone();
        one_ulp[7] = f32::from_bits(one_ulp[7].to_bits() + 1);
        assert_eq!(
            judge(&w, 40, 96, &x, &one_ulp).unwrap(),
            Verdict::InMarja {
                differing: 1,
                first: 7
            }
        );
        let mut wrong = y.clone();
        wrong[9] = 0.0;
        let v = judge(&w, 40, 96, &x, &wrong).unwrap();
        assert!(!v.passes() && v.line().contains("randul 9"), "{v:?}");
        assert!(!judge(&w, 40, 96, &x, &y[..39]).unwrap().passes());
    }
}
