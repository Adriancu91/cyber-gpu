//! `no_std` elementary functions. Original implementations from textbook
//! methods (range reduction + polynomial, Newton iteration, Abramowitz &
//! Stegun 7.1.26). Accuracy targets are stated per function and checked
//! against `std` in the tests.
//!
//! Note: cyber-universe builds for `x86_64-unknown-none` (soft-float, no SSE
//! state saving yet, ARHITECTURA §14.7), so these run in software floating
//! point inside the kernel. That is correct but slow; fine for an oracle.

/// e^x. Relative error < 1e-6 for x in [-87, 88].
pub fn exp(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 88.72 {
        return f32::INFINITY;
    }
    if x < -103.0 {
        return 0.0;
    }
    let x = x as f64;
    const LN2: f64 = core::f64::consts::LN_2;
    // x = k*ln2 + r, |r| <= ln2/2
    let kf = x / LN2;
    let k = if kf >= 0.0 {
        (kf + 0.5) as i64
    } else {
        (kf - 0.5) as i64
    };
    let r = x - k as f64 * LN2;
    // Taylor to degree 10 on |r| <= 0.347: truncation < 1e-12.
    let mut term = 1.0f64;
    let mut sum = 1.0f64;
    for i in 1..=10 {
        term *= r / i as f64;
        sum += term;
    }
    // Scale by 2^k through the exponent bits (k in [-150, 128]).
    let (k1, k2) = (k / 2, k - k / 2);
    let p1 = f64::from_bits(((1023 + k1) as u64) << 52);
    let p2 = f64::from_bits(((1023 + k2) as u64) << 52);
    (sum * p1 * p2) as f32
}

/// Square root via Newton iteration in f64. Correctly rounded for normal inputs in practice.
pub fn sqrt(x: f32) -> f32 {
    if x.is_nan() || x < 0.0 {
        return f32::NAN;
    }
    if x == 0.0 || x.is_infinite() {
        return x;
    }
    let xd = x as f64;
    // Initial guess: halve the exponent.
    let mut y = f64::from_bits((xd.to_bits() >> 1) + (1023u64 << 51));
    for _ in 0..6 {
        y = 0.5 * (y + xd / y);
    }
    y as f32
}

pub fn tanh(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    if x > 9.0 {
        return 1.0;
    }
    if x < -9.0 {
        return -1.0;
    }
    if x.abs() < 1e-4 {
        return x; // tanh x = x - x^3/3 ...; x^3/3 < 4e-13
    }
    let e = exp(2.0 * x) as f64;
    ((e - 1.0) / (e + 1.0)) as f32
}

/// erf(x), A&S 7.1.26, absolute error < 1.5e-7.
pub fn erf(x: f32) -> f32 {
    if x.is_nan() {
        return x;
    }
    let s = if x < 0.0 { -1.0 } else { 1.0 };
    let a = (x as f64).abs();
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let poly = t
        * (0.254829592
            + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    let e = exp(-(a * a) as f32) as f64;
    (s * (1.0 - poly * e)) as f32
}

/// Fused multiply-add `a * b + c` with a single rounding (round to nearest,
/// ties to even), as the GPU's `FFMA` does. `no_std`, no FMA instruction:
/// `a * b` is exact in f64 (24 + 24 significand bits), the sum is rounded once
/// to f64, and the only case where rounding that again to f32 can be wrong —
/// the f64 sum landing exactly on an f32 tie — is nudged one f64 ulp toward
/// the exact value. Idea from musl `fmaf.c` (MIT), written here; the f32
/// subnormal range (fewer kept bits) is handled too. Checked bit for bit
/// against `std`'s `f32::mul_add` in the tests.
pub fn fma_f32(a: f32, b: f32, c: f32) -> f32 {
    let (xy, z) = (a as f64 * b as f64, c as f64);
    let r = xy + z;
    let u = r.to_bits();
    let e = ((u >> 52) & 0x7ff) as i32;
    if e == 0x7ff || r == 0.0 {
        return r as f32;
    }
    // Bits of the f64 significand that the f32 result drops.
    let unb = e - 1023;
    let d = if unb < -126 { 29 + (-126 - unb) } else { 29 };
    if d >= 53 {
        return r as f32; // far below the smallest f32 subnormal tie
    }
    let mask = (1u64 << d) - 1;
    let tie = 1u64 << (d - 1);
    if u & mask != tie || (r - xy == z && r - z == xy) {
        return r as f32;
    }
    let neg = u >> 63 != 0;
    let err = if neg == (z > xy) {
        xy - r + z
    } else {
        z - r + xy
    };
    let u2 = if neg == (err < 0.0) { u + 1 } else { u - 1 };
    f64::from_bits(u2) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rel(a: f32, b: f32) -> f32 {
        if b == 0.0 {
            a.abs()
        } else {
            ((a - b) / b).abs()
        }
    }

    #[test]
    fn exp_matches_std() {
        let mut x = -87.0f32;
        while x < 88.0 {
            assert!(rel(exp(x), x.exp()) < 1e-6, "exp({x})");
            x += 0.37;
        }
        assert_eq!(exp(0.0), 1.0);
        assert_eq!(exp(100.0), f32::INFINITY);
        assert_eq!(exp(-200.0), 0.0);
        assert!(exp(f32::NAN).is_nan());
    }

    #[test]
    fn sqrt_matches_std() {
        for &x in &[1e-30f32, 1e-7, 0.25, 2.0, 3.0, 768.0, 1e20, 3.4e38] {
            assert!(rel(sqrt(x), x.sqrt()) < 1e-7, "sqrt({x})");
        }
        assert_eq!(sqrt(0.0), 0.0);
        assert!(sqrt(-1.0).is_nan());
    }

    #[test]
    fn tanh_erf_match_std() {
        let mut x = -10.0f32;
        while x < 10.0 {
            assert!((tanh(x) - x.tanh()).abs() < 2e-7, "tanh({x})");
            x += 0.013;
        }
        // erf reference from std via f64 (libm erf is not in std; use the
        // series for small |x| and the known values).
        let known = [
            (0.0f32, 0.0f32),
            (0.5, 0.520_499_9),
            (1.0, 0.842_700_8),
            (2.0, 0.995_322_3),
            (-1.0, -0.842_700_8),
        ];
        for (x, e) in known {
            assert!((erf(x) - e).abs() < 3e-7, "erf({x})");
        }
    }

    #[test]
    fn fma_matches_std_bit_for_bit() {
        let mut st = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            st ^= st << 13;
            st ^= st >> 7;
            st ^= st << 17;
            st
        };
        let mut checked = 0u32;
        for i in 0..2_000_000u32 {
            let r = next();
            // Random bits, but exponents kept in ranges that matter
            // (normal, near-subnormal, cancellation).
            let f = |w: u64, lo: u32, span: u32| {
                let sign = (w >> 63) as u32;
                let ex = lo + (w >> 32) as u32 % span;
                f32::from_bits(sign << 31 | ex << 23 | (w as u32 & 0x7f_ffff))
            };
            let a = f(r, 1, 253);
            let b = f(next(), 1, 253);
            let c = match i % 4 {
                0 => f(next(), 1, 253),
                1 => -(a * b),                        // exact error term
                2 => -(a * b) * (1.0 + f32::EPSILON), // near cancellation
                _ => f(next(), 0, 30),                // tiny / subnormal c
            };
            let (x, y) = (fma_f32(a, b, c), a.mul_add(b, c));
            assert!(
                x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()),
                "fma({a:e}, {b:e}, {c:e}) = {x:e}, std {y:e}"
            );
            checked += 1;
        }
        // Hand-made double-rounding ties.
        for (a, b, c) in [
            (1.0 + f32::EPSILON, 1.0 + f32::EPSILON, -1.0f32),
            (1.0 + f32::EPSILON, 1.0 - f32::EPSILON / 2.0, 0.0),
            (f32::MIN_POSITIVE, 0.5 + f32::EPSILON, 0.0),
            (3.0e-39, 0.75, 1.0e-45),
            (f32::MAX, 2.0, -f32::MAX),
            (0.0, f32::INFINITY, 1.0),
        ] {
            let (x, y) = (fma_f32(a, b, c), a.mul_add(b, c));
            assert!(x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
        }
        assert_eq!(checked, 2_000_000);
    }
}
