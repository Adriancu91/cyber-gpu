//! C-157 (os-467, `docs/C-157_OS467_ATTN0_TC_BLOCAJ.md`): two checks on our GPU programs that need no board, for the
//! faults `gpu-sim` cannot see (it runs a CPU stand-in for each program, never the program's instructions).
//!
//! 1. [`coopmat_shared_strides`] — on the GLSL source: every `coopMatLoad` / `coopMatStore` into a `shared` array must
//!    use a row stride that is a multiple of 16 bytes. NAK turns these into 64-bit (and wider) shared accesses, one
//!    row per group of lanes; with a stride of 65 `float`s (260 bytes, os-467) half of the 64-bit stores fall on
//!    `4 mod 8` addresses — a misaligned shared address, which stops the warp on the board and the block never ends.
//!    The 16-byte rule is the one CUDA's public documentation gives for `wmma::load_matrix_sync` /
//!    `store_matrix_sync` (`ldm` a multiple of 16 bytes); idea only, nothing copied.
//! 2. [`barriers_outside_uniform`] — on a NAK listing (`NAK IR after calc_instr_deps`): every `bar.sync` must sit in a
//!    uniform block (`block.u`), i.e. a block every thread of the workgroup reaches together. A `bar.sync` in a
//!    non-uniform block is a barrier some threads may never reach (the block hangs).
//!
//! Both are conservative text checks over our own sources and NAK's printed listing; a stride we cannot evaluate is
//! reported, never assumed good.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

/// One cooperative-matrix access into shared memory whose row stride is not a multiple of 16 bytes (or unknown).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StrideFinding {
    /// 1-based line of the call in the source.
    pub line: usize,
    /// `coopMatLoad` or `coopMatStore`.
    pub call: &'static str,
    /// The shared array.
    pub array: String,
    /// The stride in bytes, `None` when the expression could not be evaluated.
    pub stride_bytes: Option<u64>,
}

/// Bytes of one element of a GLSL scalar / vector type used in `shared` declarations (`None`: unknown type).
fn type_bytes(t: &str) -> Option<u64> {
    Some(match t {
        "float" | "uint" | "int" => 4,
        "float16_t" | "uint16_t" | "int16_t" => 2,
        "int8_t" | "uint8_t" => 1,
        "vec2" | "uvec2" | "ivec2" | "f16vec4" => 8,
        "vec4" | "uvec4" | "ivec4" => 16,
        "f16vec2" => 4,
        _ => return None,
    })
}

/// Strip a `//` comment.
fn code(line: &str) -> &str {
    line.split("//").next().unwrap_or("")
}

/// A literal (`65u`, `64`, `0x40u`) or a `const uint NAME = …u;` / `#define NAME …u` already seen.
fn eval(expr: &str, consts: &[(String, u64)]) -> Option<u64> {
    let e = expr.trim();
    let lit = e.trim_end_matches('u').trim_end_matches('U');
    if let Some(h) = lit.strip_prefix("0x") {
        return u64::from_str_radix(h, 16).ok();
    }
    if let Ok(v) = lit.parse::<u64>() {
        return Some(v);
    }
    consts.iter().find(|(n, _)| n == e).map(|(_, v)| *v)
}

/// Split the argument list of a call at top-level commas.
fn split_args(s: &str) -> Vec<String> {
    let (mut out, mut cur, mut depth) = (Vec::new(), String::new(), 0i32);
    for ch in s.chars() {
        match ch {
            '(' | '[' => {
                depth += 1;
                cur.push(ch);
            }
            ')' | ']' => {
                depth -= 1;
                cur.push(ch);
            }
            ',' if depth == 0 => out.push(core::mem::take(&mut cur).trim().to_string()),
            _ => cur.push(ch),
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Check 1: every `coopMatLoad` / `coopMatStore` on a `shared` array of `src` (GLSL) with a row stride that is not a
/// multiple of 16 bytes, or that cannot be evaluated. Empty = the program passes.
pub fn coopmat_shared_strides(src: &str) -> Vec<StrideFinding> {
    let mut shared: Vec<(String, u64)> = Vec::new();
    let mut consts: Vec<(String, u64)> = Vec::new();
    for l in src.lines() {
        let c = code(l).trim();
        if let Some(rest) = c.strip_prefix("shared ") {
            let mut it = rest.split_whitespace();
            if let (Some(t), Some(n)) = (it.next(), it.next())
                && let Some(b) = type_bytes(t)
            {
                let name = n.split('[').next().unwrap_or("").trim_end_matches(';');
                shared.push((name.to_string(), b));
            }
        }
        if let Some(rest) = c.strip_prefix("#define ") {
            let mut it = rest.split_whitespace();
            if let (Some(n), Some(v), None) = (it.next(), it.next(), it.next())
                && let Some(v) = eval(v, &consts)
            {
                consts.push((n.to_string(), v));
            }
        }
        if let Some(rest) = c.strip_prefix("const uint ")
            && let Some((n, v)) = rest.split_once('=')
            && let Some(v) = eval(v.trim().trim_end_matches(';'), &consts)
        {
            consts.push((n.trim().to_string(), v));
        }
    }
    let lines: Vec<&str> = src.lines().map(code).collect();
    let mut out = Vec::new();
    for (i, l) in lines.iter().enumerate() {
        for call in ["coopMatLoad", "coopMatStore"] {
            let mut from = 0;
            while let Some(p) = l[from..].find(call) {
                let start = from + p + call.len();
                from = start;
                if !l[start..].trim_start().starts_with('(') {
                    continue;
                }
                // Gather the text up to the matching ')' (calls may span lines).
                let mut text = String::new();
                let mut depth = 0i32;
                'collect: for (k, part) in lines[i..].iter().enumerate() {
                    let s = if k == 0 { &l[start..] } else { part };
                    for ch in s.chars() {
                        match ch {
                            '(' => {
                                depth += 1;
                                if depth == 1 {
                                    continue;
                                }
                            }
                            ')' => {
                                depth -= 1;
                                if depth == 0 {
                                    break 'collect;
                                }
                            }
                            _ => {}
                        }
                        text.push(ch);
                    }
                    text.push(' ');
                }
                let args = split_args(&text);
                if args.len() < 4 {
                    continue;
                }
                let array = args[1].trim().to_string();
                let Some(&(_, eb)) = shared.iter().find(|(n, _)| *n == array) else {
                    continue; // not shared memory (a buffer): not this check
                };
                let stride_bytes = eval(&args[3], &consts).map(|s| s * eb);
                if stride_bytes.is_none_or(|b| !b.is_multiple_of(16)) {
                    out.push(StrideFinding {
                        line: i + 1,
                        call,
                        array,
                        stride_bytes,
                    });
                }
            }
        }
    }
    out
}

/// Check 2: the 1-based lines of a NAK listing where `bar.sync` sits in a block not marked uniform (`block.u`).
/// Empty = every barrier is reached by the whole workgroup together. A listing without any `block` header is
/// reported as line 0 (nothing could be checked).
pub fn barriers_outside_uniform(listing: &str) -> Vec<usize> {
    let mut uniform: Option<bool> = None;
    let mut out = Vec::new();
    for (i, l) in listing.lines().enumerate() {
        let t = l.trim_start();
        if t.starts_with("block") {
            uniform = Some(t.starts_with("block.u "));
        } else if code(t).contains("bar.sync") && uniform != Some(true) {
            out.push(i + 1);
        }
    }
    if uniform.is_none() {
        out.push(0);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stride_65_floats_is_refused_66_passes() {
        let bad = "shared float cs[64 * 65];\ncoopMatStore(acc[j], cs, wr * 65u + wc, 65u,\n gl_CooperativeMatrixLayoutRowMajor);\n";
        let f = coopmat_shared_strides(bad);
        assert_eq!(f.len(), 1);
        assert_eq!(
            (f[0].line, f[0].call, f[0].stride_bytes),
            (2, "coopMatStore", Some(260))
        );
        let good = bad.replace("65", "66");
        // 66 · 4 = 264 bytes: a multiple of 8 but not of 16 → still refused by the 16-byte rule.
        assert_eq!(coopmat_shared_strides(&good)[0].stride_bytes, Some(264));
        let good = bad.replace("65", "68");
        assert!(coopmat_shared_strides(&good).is_empty());
    }

    #[test]
    fn consts_buffers_and_unknown_strides() {
        let s = "const uint BN = 64u;\nshared float cs_[64 * BN];\ncoopMatStore(a, cs_, 0u, BN, L);\n\
                 coopMatLoad(m, args.a.v, 0, 16, L);\ncoopMatStore(a, cs_, 0u, n + 1u, L);\n";
        let f = coopmat_shared_strides(s);
        assert_eq!(f.len(), 1);
        assert_eq!((f[0].line, f[0].stride_bytes), (5, None));
    }

    #[test]
    fn barrier_in_non_uniform_block_is_found() {
        let ok = "block.u 0 L0 [] -> {\n  bar.sync // delay=6\n} -> [1]\n";
        assert!(barriers_outside_uniform(ok).is_empty());
        let bad = "block.u 0 L0 [] -> {\n} -> [1, 2]\nblock 1 L1 [0] -> {\n  bar.sync // delay=6\n} -> [2]\n";
        assert_eq!(barriers_outside_uniform(bad), [4]);
        assert_eq!(barriers_outside_uniform("bar.sync\n"), [1, 0]);
    }
}
