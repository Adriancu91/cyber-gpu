//! B1 (int8 IMMA, SIMULARE only): the oracle for `tools/nak/mma16_s8.comp`:
//!
//! `D[16,16] = A[16,32] · B[32,16] + C[16,16]`, A and B signed 8-bit integers,
//! C and D signed 32-bit integers, all row-major.
//!
//! Integer products are exact and integer addition modulo 2³² is associative, so
//! unlike the f16 program the order in which the tensor core adds cannot change
//! the bits: the verdict is **bit-for-bit** equality.
//! - [`mma16_s8_ref`]: the reference, in i64; `None` when an element does not fit
//!   in i32 (inputs that must never be sent to a board until measured);
//! - [`mma16_s8_wrap`]: the model used by `gpu-sim` — the same sums modulo 2³².
//!   On overflow this is a **hypothesis** (the instruction NAK emits has no
//!   `.sat`), never evidence of what the RTX 3070 Ti does.

/// Rows of A, C, D.
pub const M: usize = 16;
/// Columns of B, C, D.
pub const N: usize = 16;
/// Depth (columns of A, rows of B).
pub const K: usize = 32;

fn sizes_ok(a: &[i8], b: &[i8], c: &[i32]) -> bool {
    a.len() == M * K && b.len() == K * N && c.len() == M * N
}

/// The reference: every element in i64; `None` on wrong lengths or if any element leaves i32.
pub fn mma16_s8_ref(a: &[i8], b: &[i8], c: &[i32]) -> Option<[i32; M * N]> {
    if !sizes_ok(a, b, c) {
        return None;
    }
    let mut d = [0i32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut s = c[i * N + j] as i64;
            for k in 0..K {
                s += a[i * K + k] as i64 * b[k * N + j] as i64;
            }
            d[i * N + j] = i32::try_from(s).ok()?;
        }
    }
    Some(d)
}

/// The model: the same sums modulo 2³² (equal to [`mma16_s8_ref`] whenever that is `Some`).
pub fn mma16_s8_wrap(a: &[i8], b: &[i8], c: &[i32]) -> Option<[i32; M * N]> {
    if !sizes_ok(a, b, c) {
        return None;
    }
    let mut d = [0i32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut s = c[i * N + j];
            for k in 0..K {
                s = s.wrapping_add(a[i * K + k] as i32 * b[k * N + j] as i32);
            }
            d[i * N + j] = s;
        }
    }
    Some(d)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extremes_fit_and_match() {
        let a = [-128i8; M * K];
        let b = [-128i8; K * N];
        let c = [7i32; M * N];
        let r = mma16_s8_ref(&a, &b, &c).unwrap();
        assert!(r.iter().all(|v| *v == 32 * 16_384 + 7));
        assert_eq!(Some(r), mma16_s8_wrap(&a, &b, &c));
    }

    #[test]
    fn overflow_is_none_in_ref_and_wraps_in_model() {
        let a = [1i8; M * K];
        let b = [1i8; K * N];
        let c = [i32::MAX; M * N];
        assert_eq!(mma16_s8_ref(&a, &b, &c), None);
        let w = mma16_s8_wrap(&a, &b, &c).unwrap();
        assert!(w.iter().all(|v| *v == i32::MAX.wrapping_add(32)));
    }

    #[test]
    fn wrong_lengths_are_refused() {
        assert_eq!(mma16_s8_ref(&[0; 10], &[0; K * N], &[0; M * N]), None);
        assert_eq!(mma16_s8_wrap(&[0; M * K], &[0; 3], &[0; M * N]), None);
    }
}
