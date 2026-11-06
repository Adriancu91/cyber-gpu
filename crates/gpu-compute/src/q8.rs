//! C-83 I12 — Kronos's Q8_0 weights (KGR1, hub `integrare/kronos/format_greutati.md`):
//! every row of a matrix with `cols` columns (a multiple of 32) is `cols / 32`
//! blocks of 36 bytes = an f32 scale (little-endian) + 32 signed bytes; the
//! value is `scale × number`.
//!
//! Three things the GPU step needs on the CPU side:
//! - [`matvec_cpu`]: y = W·x in the order of the OS's scalar engine
//!   (`cyber-universe/user/src/kgr.rs`, `Mat::dot` without SIMD: per block
//!   `part += q × x` with a separate multiply and add, then `acc += scale × part`);
//! - [`matvec_gpu_order`]: the same product in exactly the order and with
//!   exactly the operations of our GPU program `tools/nak/q8_matvec.comp`
//!   as NAK compiled it (FMA with flush-to-zero, see `q8_matvec.sm86.txt`):
//!   the oracle the GPU result must match **bit for bit**;
//! - [`margin`]: the tolerance, written before any run, between the GPU and
//!   the CPU engine (different summation orders): both are within
//!   `k·u·S` of the exact sum, so they differ by at most `2·k·u·S`
//!   (`u` = 2⁻²⁴, `k` = the longest chain of roundings, `S` = Σ|scale·q·x|),
//!   plus a flush-to-zero allowance.
//!
//! Nothing here runs on a GPU.

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::math::{add_f32_ftz, fma_f32_ftz};

/// Columns per block.
pub const BLOCK: usize = 32;
/// Bytes per block: f32 scale + 32 × i8.
pub const BLOCK_BYTES: usize = 36;
/// Threads per row in the GPU program (one workgroup per row).
pub const GPU_THREADS: usize = 32;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// Bytes of a `rows × cols` Q8_0 matrix.
pub fn matrix_bytes(rows: usize, cols: usize) -> GpuResult<usize> {
    if cols == 0 || !cols.is_multiple_of(BLOCK) {
        return Err(bad("Q8_0: coloanele trebuie sa fie multiplu de 32"));
    }
    rows.checked_mul(cols / BLOCK)
        .and_then(|b| b.checked_mul(BLOCK_BYTES))
        .ok_or(bad("Q8_0: matrice prea mare"))
}

fn check(w: &[u8], rows: usize, cols: usize, x: &[f32], y: &[f32]) -> GpuResult<()> {
    if w.len() != matrix_bytes(rows, cols)? || x.len() != cols || y.len() != rows {
        return Err(bad("Q8_0: marimi nepotrivite"));
    }
    Ok(())
}

/// One block: (scale, the 32 numbers).
fn block(b: &[u8]) -> (f32, impl Iterator<Item = f32> + '_) {
    let s = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    (s, b[4..BLOCK_BYTES].iter().map(|&q| q as i8 as f32))
}

/// Quantizes one row as Kronos's exporter does: per block `scale = max|x| / 127`,
/// `number = round(x / scale)` (half away from zero), an all-zero block → scale 0.
pub fn quantize_row(x: &[f32], out: &mut [u8]) -> GpuResult<()> {
    if !x.len().is_multiple_of(BLOCK) || out.len() != x.len() / BLOCK * BLOCK_BYTES {
        return Err(bad("Q8_0: marimi nepotrivite la cuantizare"));
    }
    for (xs, o) in x
        .as_chunks::<BLOCK>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<BLOCK_BYTES>().0.iter_mut())
    {
        let m = xs.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let s = m / 127.0;
        o[..4].copy_from_slice(&s.to_le_bytes());
        for (q, &v) in o[4..].iter_mut().zip(xs) {
            let r = if s == 0.0 { 0.0 } else { v / s };
            // round half away from zero, clamped to i8
            let n = if r >= 0.0 {
                (r + 0.5) as i32
            } else {
                (r - 0.5) as i32
            };
            *q = n.clamp(-127, 127) as i8 as u8;
        }
    }
    Ok(())
}

/// y = W·x in the OS's scalar order (two roundings per term).
pub fn matvec_cpu(w: &[u8], rows: usize, cols: usize, x: &[f32], y: &mut [f32]) -> GpuResult<()> {
    check(w, rows, cols, x, y)?;
    let rb = cols / BLOCK * BLOCK_BYTES;
    for (r, out) in y.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (b, xs) in w[r * rb..(r + 1) * rb]
            .as_chunks::<BLOCK_BYTES>()
            .0
            .iter()
            .zip(x.as_chunks::<BLOCK>().0.iter())
        {
            let (s, qs) = block(b);
            let part = qs.zip(xs).fold(0.0f32, |a, (q, &x)| a + q * x);
            acc += s * part;
        }
        *out = acc;
    }
    Ok(())
}

/// `FMUL.FTZ`: inputs flushed, one rounding, result flushed.
fn mul_f32_ftz(a: f32, b: f32) -> f32 {
    crate::math::ftz(crate::math::ftz(a) * crate::math::ftz(b))
}

/// One row in the GPU program's order: thread `t` takes blocks t, t+32, …;
/// per block `s = q₀·x₀` then `s = fma(q, x, s)` over the other 31, then
/// `part = fma(scale, s, part)` (all `FFMA.FTZ`); then the 32 parts are added
/// in the fixed tree (stride 16, 8, 4, 2, 1) with `FADD.FTZ`.
pub fn row_gpu_order(row: &[u8], x: &[f32]) -> f32 {
    let mut parts = [0.0f32; GPU_THREADS];
    for (t, part) in parts.iter_mut().enumerate() {
        for (b, blk) in row
            .as_chunks::<BLOCK_BYTES>()
            .0
            .iter()
            .enumerate()
            .skip(t)
            .step_by(GPU_THREADS)
        {
            let (scale, qs) = block(blk);
            let mut terms = qs.zip(&x[b * BLOCK..(b + 1) * BLOCK]);
            // NAK folded `0 + q0·x0` into `FMUL.FTZ` (q8_matvec.sm86.txt, l. 71),
            // then 31 `FFMA.FTZ` in column order.
            let first = terms.next().map_or(0.0, |(q, &x)| mul_f32_ftz(q, x));
            let s = terms.fold(first, |s, (q, &x)| fma_f32_ftz(q, x, s));
            *part = fma_f32_ftz(scale, s, *part);
        }
    }
    let mut stride = GPU_THREADS / 2;
    while stride > 0 {
        for t in 0..stride {
            parts[t] = add_f32_ftz(parts[t], parts[t + stride]);
        }
        stride /= 2;
    }
    parts[0]
}

/// y = W·x exactly as the GPU program computes it (the bit-for-bit oracle).
pub fn matvec_gpu_order(
    w: &[u8],
    rows: usize,
    cols: usize,
    x: &[f32],
    y: &mut [f32],
) -> GpuResult<()> {
    check(w, rows, cols, x, y)?;
    let rb = cols / BLOCK * BLOCK_BYTES;
    for (r, out) in y.iter_mut().enumerate() {
        *out = row_gpu_order(&w[r * rb..(r + 1) * rb], x);
    }
    Ok(())
}

/// The allowed |GPU − CPU| for each row, fixed before any run (module doc):
/// `2·k·2⁻²⁴·S·1.01 + k·2⁻¹²⁵` with `k = 64 + 2·blocks + 8` roundings at
/// most in either order and `S = Σ|scale·q·x|` computed in f64.
pub fn margin(w: &[u8], rows: usize, cols: usize, x: &[f32]) -> GpuResult<Vec<f64>> {
    let y = alloc::vec![0.0f32; rows];
    check(w, rows, cols, x, &y)?;
    let blocks = cols / BLOCK;
    let rb = blocks * BLOCK_BYTES;
    let k = (64 + 2 * blocks + 8) as f64;
    let u = 1.0 / (1u64 << 24) as f64;
    let tiny = k * f64::from_bits(((1023 - 125) as u64) << 52); // 2^-125
    Ok((0..rows)
        .map(|r| {
            let s: f64 = w[r * rb..(r + 1) * rb]
                .as_chunks::<BLOCK_BYTES>()
                .0
                .iter()
                .zip(x.as_chunks::<BLOCK>().0.iter())
                .map(|(b, xs)| {
                    let (sc, qs) = block(b);
                    qs.zip(xs)
                        .map(|(q, &x)| (sc as f64 * q as f64 * x as f64).abs())
                        .sum::<f64>()
                })
                .sum();
            2.0 * k * u * s * 1.01 + tiny
        })
        .collect())
}

/// The first row where `got` and `cpu` differ by more than its margin (or
/// either is not finite): `(row, got, cpu, margin)`. `None` = all within.
pub fn first_outside_margin(
    got: &[f32],
    cpu: &[f32],
    margins: &[f64],
) -> Option<(usize, f32, f32, f64)> {
    (0..got.len().max(cpu.len()).max(margins.len())).find_map(|r| {
        // A missing value (lengths differ) counts as NaN: never "within".
        let g = got.get(r).copied().unwrap_or(f32::NAN);
        let c = cpu.get(r).copied().unwrap_or(f32::NAN);
        let m = margins.get(r).copied().unwrap_or(-1.0);
        let ok = g.is_finite() && c.is_finite() && ((g as f64) - (c as f64)).abs() <= m;
        (!ok).then_some((r, g, c, m))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator (tests only).
    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn matrix(rows: usize, cols: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
        let mut s = seed;
        let mut w = alloc::vec![0u8; matrix_bytes(rows, cols).unwrap()];
        let rb = cols / BLOCK * BLOCK_BYTES;
        for r in 0..rows {
            let row: Vec<f32> = (0..cols).map(|_| lcg(&mut s) * 0.05).collect();
            quantize_row(&row, &mut w[r * rb..(r + 1) * rb]).unwrap();
        }
        let x = (0..cols).map(|_| lcg(&mut s)).collect();
        (w, x)
    }

    #[test]
    fn quantize_follows_the_exporter_rule() {
        let mut x = [0.0f32; 32];
        x[0] = 1.27;
        x[1] = -0.635;
        x[2] = 0.004_99; // 0.499 of a step → 0
        x[3] = 0.005_01; // just over half a step → 1
        let mut o = [0u8; 36];
        quantize_row(&x, &mut o).unwrap();
        assert_eq!(
            f32::from_le_bytes(o[..4].try_into().unwrap()),
            1.27f32 / 127.0
        );
        assert_eq!(&o[4..8], &[127, (-64i8) as u8, 0, 1]);
        quantize_row(&[0.0; 32], &mut o).unwrap();
        assert!(o.iter().all(|&b| b == 0));
        assert!(quantize_row(&[0.0; 31], &mut o).is_err());
    }

    /// The two orders agree within the margin, and the GPU order really is a
    /// different computation (some rows differ in the last bits).
    #[test]
    fn gpu_order_within_margin_of_the_cpu_engine() {
        for (rows, cols) in [(1, 32), (7, 96), (64, 768), (16, 3072), (5, 1056)] {
            let (w, x) = matrix(rows, cols, (rows * cols) as u64);
            let (mut a, mut b) = (alloc::vec![0.0; rows], alloc::vec![0.0; rows]);
            matvec_cpu(&w, rows, cols, &x, &mut a).unwrap();
            matvec_gpu_order(&w, rows, cols, &x, &mut b).unwrap();
            let m = margin(&w, rows, cols, &x).unwrap();
            assert_eq!(first_outside_margin(&b, &a, &m), None, "{rows}x{cols}");
            // The exact value (f64) is within half the margin of each.
            let rb = cols / BLOCK * BLOCK_BYTES;
            for r in 0..rows {
                let exact: f64 = w[r * rb..(r + 1) * rb]
                    .as_chunks::<BLOCK_BYTES>()
                    .0
                    .iter()
                    .zip(x.as_chunks::<BLOCK>().0.iter())
                    .map(|(bl, xs)| {
                        let (s, qs) = block(bl);
                        s as f64 * qs.zip(xs).map(|(q, &x)| q as f64 * x as f64).sum::<f64>()
                    })
                    .sum();
                assert!((a[r] as f64 - exact).abs() <= m[r] / 2.0);
                assert!((b[r] as f64 - exact).abs() <= m[r] / 2.0);
            }
        }
        let (w, x) = matrix(64, 768, 1);
        let (mut a, mut b) = (alloc::vec![0.0f32; 64], alloc::vec![0.0f32; 64]);
        matvec_cpu(&w, 64, 768, &x, &mut a).unwrap();
        matvec_gpu_order(&w, 64, 768, &x, &mut b).unwrap();
        assert!(a.iter().zip(&b).any(|(p, q)| p.to_bits() != q.to_bits()));
    }

    #[test]
    fn margin_catches_a_wrong_result() {
        let (w, x) = matrix(8, 256, 3);
        let mut y = alloc::vec![0.0f32; 8];
        matvec_cpu(&w, 8, 256, &x, &mut y).unwrap();
        let m = margin(&w, 8, 256, &x).unwrap();
        let mut bad = y.clone();
        bad[5] += 1e-3; // a wrong block or a lost term is far bigger than the margin
        assert_eq!(first_outside_margin(&bad, &y, &m).map(|f| f.0), Some(5));
        let mut nan = y.clone();
        nan[2] = f32::NAN;
        assert_eq!(first_outside_margin(&nan, &y, &m).map(|f| f.0), Some(2));
        assert!(
            first_outside_margin(&y[..7], &y, &m).is_some(),
            "rand lipsa"
        );
        // A one-column swap inside a block (wrong unpacking) is caught too.
        let mut w2 = w.clone();
        w2.swap(4, 5);
        let mut z = alloc::vec![0.0f32; 8];
        matvec_cpu(&w2, 8, 256, &x, &mut z).unwrap();
        assert!(first_outside_margin(&z, &y, &m).is_some());
    }

    #[test]
    fn shapes_are_checked() {
        let mut y = [0.0f32; 2];
        assert!(matvec_cpu(&[0; 72], 2, 32, &[0.0; 32], &mut y).is_ok());
        assert!(matvec_cpu(&[0; 72], 2, 31, &[0.0; 31], &mut y).is_err());
        assert!(matvec_gpu_order(&[0; 71], 2, 32, &[0.0; 32], &mut y).is_err());
        assert!(matvec_gpu_order(&[0; 72], 2, 32, &[0.0; 31], &mut y).is_err());
        assert!(matrix_bytes(usize::MAX, 64).is_err());
    }
}
