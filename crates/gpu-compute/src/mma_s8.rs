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

/// B1 step 3 (`docs/B1_MM_S8.md`): one product of `tools/nak/mm_s8.comp`,
/// `C[m, n] = op(A)[m, k] · op(B)[k, n] (+ bias[n])` in integers. The flags are
/// `mm_tc`'s ([`crate::mm_tc::TA`]: A stored [k, m]; [`crate::mm_tc::TB`]: B stored
/// [n, k]; [`crate::mm_tc::BIAS`]).
#[derive(Clone, Copy, Debug)]
pub struct ProductS8<'a> {
    pub a: &'a [i8],
    pub b: &'a [i8],
    /// Read only with [`crate::mm_tc::BIAS`].
    pub bias: Option<&'a [i32]>,
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub flags: u32,
}

impl ProductS8<'_> {
    fn valid(&self) -> bool {
        use crate::mm_tc::{BIAS, TA, TB};
        self.m > 0
            && self.n > 0
            && self.k > 0
            && self.flags & !(TA | TB | BIAS) == 0
            && self.a.len() >= self.m * self.k
            && self.b.len() >= self.k * self.n
            && (self.flags & BIAS == 0 || self.bias.is_some_and(|b| b.len() >= self.n))
    }

    fn at_a(&self, r: usize, p: usize) -> i64 {
        if self.flags & crate::mm_tc::TA != 0 {
            self.a[p * self.m + r] as i64
        } else {
            self.a[r * self.k + p] as i64
        }
    }

    fn at_b(&self, p: usize, c: usize) -> i64 {
        if self.flags & crate::mm_tc::TB != 0 {
            self.b[c * self.k + p] as i64
        } else {
            self.b[p * self.n + c] as i64
        }
    }

    fn sums(&self) -> Option<alloc::vec::Vec<i64>> {
        if !self.valid() {
            return None;
        }
        let bias = self.bias.filter(|_| self.flags & crate::mm_tc::BIAS != 0);
        let mut out = alloc::vec![0i64; self.m * self.n];
        for r in 0..self.m {
            for c in 0..self.n {
                let mut s = 0i64;
                for p in 0..self.k {
                    s += self.at_a(r, p) * self.at_b(p, c);
                }
                if let Some(b) = bias {
                    s += b[c] as i64;
                }
                out[r * self.n + c] = s;
            }
        }
        Some(out)
    }
}

/// The reference of `mm_s8`: every element in i64; `None` on an invalid product or
/// if any element leaves i32 (inputs never to be sent to a board until measured).
pub fn mm_s8_ref(p: &ProductS8<'_>) -> Option<alloc::vec::Vec<i32>> {
    p.sums()?
        .into_iter()
        .map(|s| i32::try_from(s).ok())
        .collect()
}

/// The model used by `gpu-sim`: the same sums modulo 2³² (equal to [`mm_s8_ref`]
/// whenever that is `Some`; on overflow a HYPOTHESIS, the instruction has no `.sat`).
pub fn mm_s8_wrap(p: &ProductS8<'_>) -> Option<alloc::vec::Vec<i32>> {
    Some(p.sums()?.into_iter().map(|s| s as i32).collect())
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

    #[test]
    fn mm_s8_on_the_probe_shape_is_the_probe() {
        let a: alloc::vec::Vec<i8> = (0..M * K).map(|i| (i * 37 % 251) as u8 as i8).collect();
        let b: alloc::vec::Vec<i8> = (0..K * N).map(|i| (i * 91 % 253) as u8 as i8).collect();
        let c = [0i32; M * N];
        let p = ProductS8 {
            a: &a,
            b: &b,
            bias: None,
            m: M,
            n: N,
            k: K,
            flags: 0,
        };
        assert_eq!(
            mm_s8_ref(&p).unwrap(),
            mma16_s8_ref(&a, &b, &c).unwrap().to_vec()
        );
        // TB: the same B stored [n, k] gives the same C.
        let bt: alloc::vec::Vec<i8> = (0..N * K).map(|i| b[(i % K) * N + i / K]).collect();
        let q = ProductS8 {
            b: &bt,
            flags: crate::mm_tc::TB,
            ..p
        };
        assert_eq!(mm_s8_ref(&q), mm_s8_ref(&p));
    }

    #[test]
    fn mm_s8_overflow_and_refusals() {
        let a = [1i8; 4];
        let b = [1i8; 4];
        let bias = [i32::MAX; 2];
        let p = ProductS8 {
            a: &a,
            b: &b,
            bias: Some(&bias),
            m: 2,
            n: 2,
            k: 2,
            flags: crate::mm_tc::BIAS,
        };
        assert_eq!(mm_s8_ref(&p), None);
        assert_eq!(
            mm_s8_wrap(&p).unwrap(),
            alloc::vec![i32::MAX.wrapping_add(2); 4]
        );
        assert_eq!(mm_s8_ref(&ProductS8 { flags: 8, ..p }), None);
        assert_eq!(mm_s8_ref(&ProductS8 { bias: None, ..p }), None);
        assert_eq!(mm_s8_ref(&ProductS8 { k: 0, ..p }), None);
    }
}
