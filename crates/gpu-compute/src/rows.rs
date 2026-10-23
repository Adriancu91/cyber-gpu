//! Row-range versions of the operators whose rows cannot be split by plain
//! slicing, so that a caller can spread one operator over several CPU cores.
//!
//! Most operators already split by slicing: matmul (`a`, `out` rows),
//! `layernorm`, `bias_add`, `gelu_*`, `embedding` (a slice of `ids`). Three do
//! not, because a row depends on its absolute index or on a global reduction:
//!
//! - [`softmax_rows_range`]: with `causal`, row r sees columns `0..=r`, so a
//!   slice must know its first absolute row;
//! - [`causal_attention_rows`]: output row i reads rows `0..=i` of k and v;
//! - [`cross_entropy_terms`] + [`cross_entropy_mean`]: the loss is a mean, so
//!   each part produces per-row terms and one thread sums them in row order.
//!
//! [`split_rows`] gives a deterministic, balanced partition.
//!
//! Contract: computing every part (in any order, on any core) and joining
//! them gives a result **bit-identical** to the matching function in
//! [`crate::reference`] — the same operations in the same order per row.
//! Tests check that for many partitions. There is no global or shared
//! state: every function only touches the slices it is given, so parts can
//! run concurrently on disjoint output slices. `no_std`, no `unsafe`.

use alloc::vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::math;
use crate::reference::ln;

fn need(cond: bool, what: &'static str) -> GpuResult<()> {
    if cond {
        Ok(())
    } else {
        Err(GpuError::InvalidArgument(what))
    }
}

/// Rows `[r0, r1)` of part `part` when `rows` rows are split into `parts`
/// parts: sizes differ by at most one, the first `rows % parts` parts get
/// the extra row, and the parts cover `0..rows` in order with no overlap.
pub fn split_rows(rows: usize, parts: usize, part: usize) -> GpuResult<(usize, usize)> {
    need(parts > 0 && part < parts, "split_rows: part >= parts")?;
    let (base, extra) = (rows / parts, rows % parts);
    let r0 = part * base + part.min(extra);
    let r1 = r0 + base + usize::from(part < extra);
    Ok((r0, r1))
}

/// Softmax over the rows `row0 .. row0 + n` of a `[_, cols]` matrix, given
/// as `x_rows = x[row0 * cols .. (row0 + n) * cols]`. With `causal`, absolute
/// row r sees columns `0..=r` (needs `row0 + n <= cols`). Bit-identical to
/// [`crate::reference::softmax_rows`] on the same rows.
pub fn softmax_rows_range(
    x_rows: &mut [f32],
    row0: usize,
    cols: usize,
    causal: bool,
) -> GpuResult<()> {
    need(
        cols > 0 && x_rows.len().is_multiple_of(cols),
        "softmax felie: forme",
    )?;
    let n = x_rows.len() / cols;
    need(!causal || row0 + n <= cols, "softmax cauzal: rows > cols")?;
    for (i, row) in x_rows.chunks_exact_mut(cols).enumerate() {
        let visible = if causal { row0 + i + 1 } else { cols };
        softmax_prefix(row, visible);
    }
    Ok(())
}

/// Same steps as the loop body of `reference::softmax_rows`.
fn softmax_prefix(row: &mut [f32], visible: usize) {
    let max = row[..visible]
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in row[..visible].iter_mut() {
        *v = math::exp(*v - max);
        sum += *v;
    }
    for v in row[..visible].iter_mut() {
        *v /= sum;
    }
    for v in row[visible..].iter_mut() {
        *v = 0.0;
    }
}

/// Output rows `r0 .. r0 + n` of [`crate::reference::causal_attention`]:
/// `q`, `k`, `v` are the whole `[t, heads * d]` inputs, `out_rows` is
/// `out[r0 * w .. (r0 + n) * w]` with `w = heads * d`. Bit-identical to the
/// reference on those rows. Scratch: one row of `t` scores per call.
#[allow(clippy::too_many_arguments)]
pub fn causal_attention_rows(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out_rows: &mut [f32],
    t: usize,
    heads: usize,
    d: usize,
    r0: usize,
) -> GpuResult<()> {
    let w = heads * d;
    need(d > 0 && heads > 0, "atentie: dimensiuni")?;
    need(
        q.len() == t * w && k.len() == t * w && v.len() == t * w,
        "atentie: forme",
    )?;
    need(out_rows.len().is_multiple_of(w), "atentie felie: forme")?;
    let n = out_rows.len() / w;
    need(r0 + n <= t, "atentie felie: randuri peste t")?;
    let scale = 1.0 / math::sqrt(d as f32);
    let mut scores = vec![0.0f32; t];
    for h in 0..heads {
        for (li, orow) in out_rows.chunks_exact_mut(w).enumerate() {
            let i = r0 + li;
            // Scores for j > i are masked to 0 by the causal softmax, so only
            // 0..=i are computed; each one with the reference's operations.
            for (j, sc) in scores[..=i].iter_mut().enumerate() {
                let mut s = 0.0;
                for c in 0..d {
                    s += q[i * w + h * d + c] * k[j * w + h * d + c];
                }
                *sc = s * scale;
            }
            softmax_prefix(&mut scores[..=i], i + 1);
            for c in 0..d {
                let mut acc = 0.0;
                for (j, &p) in scores[..=i].iter().enumerate() {
                    acc += p * v[j * w + h * d + c];
                }
                orow[h * d + c] = acc;
            }
        }
    }
    Ok(())
}

/// Per-row cross-entropy terms (f64) for a slice of rows:
/// `terms[r] = ln(sum exp(x - max)) + max - x[target]`. Join the parts'
/// terms in row order and pass them to [`cross_entropy_mean`].
pub fn cross_entropy_terms(
    logits_rows: &[f32],
    targets_rows: &[u32],
    vocab: usize,
    terms: &mut [f64],
) -> GpuResult<()> {
    need(
        vocab > 0
            && logits_rows.len() == targets_rows.len() * vocab
            && terms.len() == targets_rows.len(),
        "cross-entropy felie: forme",
    )?;
    for ((row, &tgt), term) in logits_rows
        .chunks_exact(vocab)
        .zip(targets_rows)
        .zip(terms.iter_mut())
    {
        if tgt as usize >= vocab {
            return Err(GpuError::OutOfBounds {
                offset: tgt as u64,
                len: 1,
                limit: vocab as u64,
            });
        }
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f64 = row.iter().map(|&x| math::exp(x - max) as f64).sum();
        *term = ln(sum) + max as f64 - row[tgt as usize] as f64;
    }
    Ok(())
}

/// Mean of all rows' terms, summed in row order on one thread: equal, bit
/// for bit, to [`crate::reference::cross_entropy`].
pub fn cross_entropy_mean(terms: &[f64]) -> GpuResult<f32> {
    need(!terms.is_empty(), "cross-entropy: niciun rand")?;
    let mut total = 0.0f64;
    for &t in terms {
        total += t;
    }
    Ok((total / terms.len() as f64) as f32)
}

#[cfg(test)]
mod tests {
    //! Differential tests: every partition must rebuild the reference result
    //! bit for bit. Parts are computed in reverse order on purpose (no
    //! hidden dependence between parts).
    use super::*;
    use crate::reference;
    use alloc::vec::Vec;

    fn rand_vec(n: usize, seed: &mut u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0
            })
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    const PARTS: &[usize] = &[1, 2, 3, 4, 7, 16, 40];

    #[test]
    fn split_covers_all_rows_once_and_balanced() {
        for rows in [0usize, 1, 5, 16, 17, 100, 1023] {
            for &parts in PARTS {
                let mut next = 0;
                for p in 0..parts {
                    let (r0, r1) = split_rows(rows, parts, p).unwrap();
                    assert_eq!(r0, next, "rows {rows} parts {parts} part {p}");
                    let len = r1 - r0;
                    assert!(len == rows / parts || len == rows / parts + 1);
                    next = r1;
                }
                assert_eq!(next, rows);
            }
        }
        assert!(split_rows(4, 0, 0).is_err());
        assert!(split_rows(4, 2, 2).is_err());
    }

    #[test]
    fn softmax_parts_match_reference() {
        let mut s = 11;
        for &(rows, cols, causal) in &[(9usize, 9usize, true), (5, 12, true), (13, 7, false)] {
            let x = rand_vec(rows * cols, &mut s);
            let mut want = x.clone();
            reference::softmax_rows(&mut want, rows, cols, causal).unwrap();
            for &parts in PARTS {
                let mut got = x.clone();
                for p in (0..parts).rev() {
                    let (r0, r1) = split_rows(rows, parts, p).unwrap();
                    softmax_rows_range(&mut got[r0 * cols..r1 * cols], r0, cols, causal).unwrap();
                }
                assert_eq!(bits(&got), bits(&want), "{rows}x{cols} parts {parts}");
            }
        }
        let mut x = [0.0f32; 6];
        assert!(softmax_rows_range(&mut x, 2, 3, true).is_err()); // rows 2..4 > cols 3
        assert!(softmax_rows_range(&mut x[..5], 0, 3, false).is_err());
    }

    #[test]
    fn attention_parts_match_reference() {
        let mut s = 23;
        for &(t, heads, d) in &[(1usize, 1usize, 1usize), (8, 2, 4), (19, 3, 5), (32, 4, 16)] {
            let w = heads * d;
            let q = rand_vec(t * w, &mut s);
            let k = rand_vec(t * w, &mut s);
            let v = rand_vec(t * w, &mut s);
            let mut want = vec![0.0; t * w];
            reference::causal_attention(&q, &k, &v, &mut want, t, heads, d).unwrap();
            for &parts in PARTS {
                let mut got = vec![7.0; t * w];
                for p in (0..parts).rev() {
                    let (r0, r1) = split_rows(t, parts, p).unwrap();
                    causal_attention_rows(&q, &k, &v, &mut got[r0 * w..r1 * w], t, heads, d, r0)
                        .unwrap();
                }
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "t {t} h {heads} d {d} parts {parts}"
                );
            }
        }
        let z = [0.0f32; 8];
        let mut o = [0.0f32; 4];
        assert!(causal_attention_rows(&z, &z, &z, &mut o, 4, 1, 2, 3).is_err()); // rows 3..5 > t
        assert!(causal_attention_rows(&z, &z, &z, &mut o, 4, 1, 0, 0).is_err());
    }

    #[test]
    fn cross_entropy_parts_match_reference() {
        let mut s = 31;
        for &(rows, vocab) in &[(1usize, 2usize), (7, 11), (33, 50)] {
            let logits = rand_vec(rows * vocab, &mut s);
            let targets: Vec<u32> = (0..rows).map(|r| ((r * 7 + 3) % vocab) as u32).collect();
            let want = reference::cross_entropy(&logits, &targets, vocab).unwrap();
            for &parts in PARTS {
                let mut terms = vec![0.0f64; rows];
                for p in (0..parts).rev() {
                    let (r0, r1) = split_rows(rows, parts, p).unwrap();
                    cross_entropy_terms(
                        &logits[r0 * vocab..r1 * vocab],
                        &targets[r0..r1],
                        vocab,
                        &mut terms[r0..r1],
                    )
                    .unwrap();
                }
                let got = cross_entropy_mean(&terms).unwrap();
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "{rows}x{vocab} parts {parts}"
                );
            }
        }
        let mut t1 = [0.0f64; 1];
        assert!(cross_entropy_terms(&[0.0, 1.0], &[2], 2, &mut t1).is_err()); // target >= vocab
        assert!(cross_entropy_mean(&[]).is_err());
    }
}
