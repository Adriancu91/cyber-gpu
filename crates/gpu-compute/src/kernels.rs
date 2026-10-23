//! Faster CPU kernels, checked against [`crate::reference`].
//!
//! - bf16 <-> f32 slice conversion (Kronos stores weights in bf16);
//! - [`matmul_blocked`]: matmul over panels of B packed into a small f32
//!   buffer (cache blocking, unit-stride inner loop the compiler can
//!   vectorize when the target has SIMD);
//! - [`matmul_bf16w`]: the same with B (the weights) stored as bf16 and
//!   decoded panel by panel; activations and accumulation stay f32.
//!
//! Design rule: every output element accumulates its k products in the
//! same order (p = 0, 1, …, k-1) and with the same operations as
//! `reference::matmul`, so the results are **bit-identical** to the oracle.
//! Tests check exactly that, plus an f64 error bound. No `unsafe`, `no_std`,
//! nothing GA104-specific.

use alloc::vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::tensor::{bf16_to_f32, f32_to_bf16};

/// Panel sizes: a KC x NC panel of B is 64 KiB of f32 (fits in L2).
pub const KC: usize = 256;
pub const NC: usize = 64;

fn need(cond: bool, what: &'static str) -> GpuResult<()> {
    if cond {
        Ok(())
    } else {
        Err(GpuError::InvalidArgument(what))
    }
}

pub fn bf16_to_f32_slice(src: &[u16], dst: &mut [f32]) -> GpuResult<()> {
    need(src.len() == dst.len(), "bf16->f32: lungimi diferite")?;
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = bf16_to_f32(s);
    }
    Ok(())
}

pub fn f32_to_bf16_slice(src: &[f32], dst: &mut [u16]) -> GpuResult<()> {
    need(src.len() == dst.len(), "f32->bf16: lungimi diferite")?;
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = f32_to_bf16(s);
    }
    Ok(())
}

/// Shared driver: `pack(p0, kc, j0, nc, panel)` fills `panel[kc * nc]` with
/// B[p0..p0+kc, j0..j0+nc] as f32, row-major.
fn matmul_panels(
    a: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    mut pack: impl FnMut(usize, usize, usize, usize, &mut [f32]),
) {
    out.fill(0.0);
    let mut panel = vec![0.0f32; KC.min(k.max(1)) * NC.min(n.max(1))];
    let mut j0 = 0;
    while j0 < n {
        let nc = NC.min(n - j0);
        let mut p0 = 0;
        while p0 < k {
            let kc = KC.min(k - p0);
            let panel = &mut panel[..kc * nc];
            pack(p0, kc, j0, nc, panel);
            for i in 0..m {
                let arow = &a[i * k + p0..i * k + p0 + kc];
                let orow = &mut out[i * n + j0..i * n + j0 + nc];
                for (p, &av) in arow.iter().enumerate() {
                    let brow = &panel[p * nc..(p + 1) * nc];
                    for (o, &bv) in orow.iter_mut().zip(brow) {
                        *o += av * bv;
                    }
                }
            }
            p0 += kc;
        }
        j0 += nc;
    }
}

/// out[m, n] = a[m, k] * b[k, n], all f32. Bit-identical to `reference::matmul`.
pub fn matmul_blocked(
    a: &[f32],
    b: &[f32],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> GpuResult<()> {
    need(
        a.len() == m * k && b.len() == k * n && out.len() == m * n,
        "matmul: forme",
    )?;
    matmul_panels(a, out, m, k, n, |p0, kc, j0, nc, panel| {
        for p in 0..kc {
            let src = &b[(p0 + p) * n + j0..(p0 + p) * n + j0 + nc];
            panel[p * nc..(p + 1) * nc].copy_from_slice(src);
        }
    });
    Ok(())
}

/// out[m, n] = a[m, k] (f32) * w[k, n] (bf16 bits). Equal, bit for bit, to
/// `reference::matmul(a, decode(w))`.
pub fn matmul_bf16w(
    a: &[f32],
    w: &[u16],
    out: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
) -> GpuResult<()> {
    need(
        a.len() == m * k && w.len() == k * n && out.len() == m * n,
        "matmul bf16: forme",
    )?;
    matmul_panels(a, out, m, k, n, |p0, kc, j0, nc, panel| {
        for p in 0..kc {
            let src = &w[(p0 + p) * n + j0..(p0 + p) * n + j0 + nc];
            for (d, &s) in panel[p * nc..(p + 1) * nc].iter_mut().zip(src) {
                *d = bf16_to_f32(s);
            }
        }
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    //! Differential tests. Oracles: `reference::matmul` (bit-exact) and an
    //! independent f64 computation with a proven error bound; for bf16, an
    //! f64 nearest-even rounding written from the definition.
    use super::*;
    use crate::reference;
    use alloc::vec::Vec;

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 11
    }

    fn rand_vec(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| (lcg(seed) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32)
            .collect()
    }

    /// |computed - exact| <= gamma_k * sum |a||b|, gamma_k = k u / (1 - k u),
    /// u = 2^-24 (standard bound for recursive summation, Higham §3.1).
    fn check_bound(o: &[f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize) {
        let u = 2f64.powi(-24);
        let g = k as f64 * u / (1.0 - k as f64 * u);
        for i in 0..m {
            for j in 0..n {
                let (mut exact, mut absum) = (0.0f64, 0.0f64);
                for p in 0..k {
                    let t = a[i * k + p] as f64 * b[p * n + j] as f64;
                    exact += t;
                    absum += t.abs();
                }
                let err = (o[i * n + j] as f64 - exact).abs();
                assert!(err <= g * absum + 1e-30, "({i},{j}) err {err}");
            }
        }
    }

    const SHAPES: &[(usize, usize, usize)] = &[
        (1, 1, 1),
        (3, 5, 7),
        (17, 33, 9),
        (2, 300, 65),  // crosses KC and NC
        (5, 513, 130), // several panels in both directions, ragged edges
        (4, 0, 3),     // k = 0: output is all zeros
        (64, 257, 64),
    ];

    #[test]
    fn blocked_is_bit_identical_to_reference() {
        let mut s = 7;
        for &(m, k, n) in SHAPES {
            let a = rand_vec(m * k, &mut s);
            let b = rand_vec(k * n, &mut s);
            let (mut want, mut got) = (vec![1.0; m * n], vec![9.0; m * n]);
            reference::matmul(&a, &b, &mut want, m, k, n).unwrap();
            matmul_blocked(&a, &b, &mut got, m, k, n).unwrap();
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&got), bits(&want), "shape {m}x{k}x{n}");
            check_bound(&got, &a, &b, m, k, n);
        }
    }

    #[test]
    fn bf16_weights_match_decoded_reference() {
        let mut s = 11;
        for &(m, k, n) in SHAPES {
            let a = rand_vec(m * k, &mut s);
            let wf = rand_vec(k * n, &mut s);
            let mut w = vec![0u16; k * n];
            f32_to_bf16_slice(&wf, &mut w).unwrap();
            let mut wd = vec![0.0; k * n];
            bf16_to_f32_slice(&w, &mut wd).unwrap();
            let (mut want, mut got) = (vec![0.0; m * n], vec![0.0; m * n]);
            reference::matmul(&a, &wd, &mut want, m, k, n).unwrap();
            matmul_bf16w(&a, &w, &mut got, m, k, n).unwrap();
            assert_eq!(got, want, "shape {m}x{k}x{n}");
            check_bound(&got, &a, &wd, m, k, n);
            // Quantization effect vs the f32 weights stays within bf16 precision.
            let mut full = vec![0.0; m * n];
            reference::matmul(&a, &wf, &mut full, m, k, n).unwrap();
            for i in 0..m * n {
                let mag: f64 = (0..k)
                    .map(|p| (a[(i / n) * k + p] * wf[p * n + i % n]).abs() as f64)
                    .sum();
                assert!(((got[i] - full[i]) as f64).abs() <= mag / 256.0 + 1e-6);
            }
        }
    }

    #[test]
    fn shape_errors_are_refused() {
        let mut o = [0.0; 4];
        assert!(matmul_blocked(&[1.0; 4], &[1.0; 3], &mut o, 2, 2, 2).is_err());
        assert!(matmul_bf16w(&[1.0; 4], &[0; 4], &mut o[..3], 2, 2, 2).is_err());
        assert!(bf16_to_f32_slice(&[0; 2], &mut o).is_err());
        assert!(f32_to_bf16_slice(&o, &mut [0u16; 3]).is_err());
    }

    /// Every one of the 65 536 bf16 values survives bf16 -> f32 -> bf16.
    #[test]
    fn bf16_roundtrip_exhaustive() {
        for h in 0..=u16::MAX {
            let f = bf16_to_f32(h);
            let back = f32_to_bf16(f);
            if f.is_nan() {
                assert!(bf16_to_f32(back).is_nan(), "{h:#06x}");
                assert_eq!(back & 0x8000, h & 0x8000, "sign kept");
            } else {
                assert_eq!(back, h, "{h:#06x}");
            }
        }
    }

    /// Independent oracle: nearest bf16 by f64 distance, ties to even;
    /// overflow follows IEEE (treat +inf as 2^128 for the distance).
    fn oracle_bf16(x: f32) -> u16 {
        let bits = x.to_bits();
        let sign = (bits >> 16) as u16 & 0x8000;
        let mag = bits & 0x7fff_ffff;
        let lo = (mag >> 16) as u16; // truncation toward zero
        let hi = lo + 1;
        let val = |h: u16| -> f64 {
            if h == 0x7f80 {
                2f64.powi(128)
            } else {
                f32::from_bits((h as u32) << 16) as f64
            }
        };
        let xv = f32::from_bits(mag) as f64;
        let (dl, dh) = (xv - val(lo), val(hi) - xv);
        let pick = if dl < dh {
            lo
        } else if dh < dl {
            hi
        } else if lo & 1 == 0 {
            lo
        } else {
            hi
        };
        sign | pick
    }

    #[test]
    fn f32_to_bf16_matches_oracle() {
        let mut s = 3;
        let mut special: Vec<u32> = vec![
            0x0000_0000,
            0x8000_0000,
            0x0000_0001, // smallest subnormal
            0x0000_8000, // subnormal exactly halfway
            0x0001_8000, // halfway, odd -> rounds up
            0x7f7f_ffff, // f32::MAX -> +inf under RNE
            0x7f7f_7fff, // just below the overflow halfway point
            0x7f80_0000, // +inf
            0xff80_0000, // -inf
            0x3f80_8000, // 1 + 2^-8, tie -> even (1.0)
            0x3f81_8000, // tie -> even (up)
        ];
        special.extend((0..200_000).map(|_| lcg(&mut s) as u32));
        for b in special {
            let x = f32::from_bits(b);
            if x.is_nan() {
                continue;
            }
            assert_eq!(f32_to_bf16(x), oracle_bf16(x), "{b:#010x}");
        }
    }

    /// Host timing, informative only (`cargo test --release -- --ignored`).
    #[test]
    #[ignore]
    fn timing_reference_vs_blocked() {
        extern crate std;
        use std::time::Instant;
        let (m, k, n) = (64, 768, 768);
        let mut s = 5;
        let a = rand_vec(m * k, &mut s);
        let b = rand_vec(k * n, &mut s);
        let mut w = vec![0u16; k * n];
        f32_to_bf16_slice(&b, &mut w).unwrap();
        let mut o = vec![0.0; m * n];
        let t = Instant::now();
        reference::matmul(&a, &b, &mut o, m, k, n).unwrap();
        let tr = t.elapsed();
        let t = Instant::now();
        matmul_blocked(&a, &b, &mut o, m, k, n).unwrap();
        let tb = t.elapsed();
        let t = Instant::now();
        matmul_bf16w(&a, &w, &mut o, m, k, n).unwrap();
        let tw = t.elapsed();
        let gflop = 2.0 * (m * k * n) as f64 / 1e9;
        std::println!(
            "{m}x{k}x{n}: referinta {tr:?} ({:.2} GFLOP/s), blocuri {tb:?} ({:.2}), bf16 {tw:?} ({:.2})",
            gflop / tr.as_secs_f64(),
            gflop / tb.as_secs_f64(),
            gflop / tw.as_secs_f64()
        );
    }
}
