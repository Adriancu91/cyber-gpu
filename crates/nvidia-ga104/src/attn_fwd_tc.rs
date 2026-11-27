//! C-168 point 4, step 1 (`docs/C-168_ATTN_FWD_TC_PLAN.md`, plan + predictions H1–H7 written before the code,
//! session 281; this program: session 282) — causal attention **forward on the tensor cores**
//! (`tools/nak/attn_fwd_tc.comp`, compiled by the **second** NAK tool, Mesa 26.3.1, SM 8.6).
//!
//! Two passes over the key tiles, P written once: pass 1 computes S = Q·Kᵀ and the row maximum / sum; pass 2 computes S
//! again, P (f32 into `p`, f16 into shared memory) and O += P·V. Grid in **pairs** of row tiles (block x does rows x
//! and ⌈T/64⌉ − 1 − x, so every block has the same number of key tiles). The same job and arguments as
//! [`crate::attn_fwd_tile3`] (qkv, p, att, T, C, H, d, scale — [`crate::attn_fwd::Args`]), but the push constants sit
//! at **0x700** (NVK 26.3.1), so the constant buffer is [`cbuf`], not `attn_fwd::Args::cbuf`.
//!
//! **What `p` holds afterwards:** every tile at or below the diagonal tile (exact 0 above the diagonal inside the
//! diagonal tile, as `attn_fwd_tile3`); the tiles **above** the diagonal tile are **not written** (`attn_fwd_tile3`
//! wrote 0 there). The backward programs never use those numbers.
//!
//! **Only aligned shapes** ([`fits`]): d = 64, T a multiple of 64, qkv / p / att multiples of 16 bytes; otherwise the
//! host keeps `attn_fwd_tile3`. Nothing here has run on a board; the CPU model and the SIMULARE test are step 2.

use crate::attn_fwd::Args;
use crate::qmd::{ComputeQmd, MAX_SHARED_BYTES, SharedMemory};

/// `tools/nak/out2/attn_fwd_tc.sm86.sass` (Mesa NAK 26.3.1, SM 8.6).
pub const ATTN_FWD_TC: &[u8] = include_bytes!("../../../tools/nak/out2/attn_fwd_tc.sm86.sass");
/// SHA-256 of [`ATTN_FWD_TC`] (`sh tools/nak/build2.sh attn_fwd_tc`; checked by a test, written in the note).
pub const ATTN_FWD_TC_SHA256: &str =
    "ce2b2f22ee58c6e50091f532e7520b4493bf04555a2e2b2f8acfc1ae7e929257";
/// Registers per thread (NAK dump `attn_fwd_tc.sm86.txt`).
pub const REGISTERS: u32 = 120;
/// Threads per block: 256 = 8 warps.
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Rows of one row tile (a block does two of them, a pair).
pub const ROWS_PER_TILE: u32 = 64;
/// The only head width this program takes.
pub const D: u32 = 64;
/// Shared memory: three areas of 64 rows × 144 B (72 f16 — `ldsm` without bank conflicts) = 27 648 bytes; the f32
/// S / O tile (rows of 68 floats) is kept over the first two. One barrier (`bar.sync 0`).
pub const SHARED: SharedMemory = SharedMemory {
    bytes: 3 * 64 * 144,
    max_bytes: MAX_SHARED_BYTES,
    barriers: 1,
};
/// Blocks per SM at the 64 KiB step (shared: 2 × (27 648 + 1 024) ≤ 65 536; registers: 2 × 256 × [`REGISTERS`]
/// ≤ 65 536 needs [`REGISTERS`] ≤ 128).
pub const BLOCKS_PER_SM: u32 = 2;
/// Where NVK 26.3.1 puts the push constants in cbuf 0 (the dump reads `c[0x0][+0x700]` … `+0x72c`).
pub const PUSH_BASE: usize = 0x700;
/// Bytes of the arguments: qkv, p, att (3 × 8), T, C, H, d (4 × 4), scale (4) = 44, in `attn_fwd::Args::cbuf`'s
/// order (there at 0x28..0x54).
pub const PUSH_BYTES: usize = 0x2c;
/// Size of constant buffer 0 (covers 0x700 + the arguments, a multiple of 256), as the other tc programs.
pub const CBUF_BYTES: usize = 0x800;

/// May `attn_fwd_tc` run this launch? `attn_fwd_tile3`'s shape rule plus: d = C / H = 64, T a multiple of 64 and
/// the addresses of qkv, p, att multiples of 16 bytes (every 128-bit load and store aligned, every tile whole).
/// `false` → launch `attn_fwd_tile3`.
pub fn fits(a: &Args) -> bool {
    crate::attn_fwd_tile3::shape_ok(a)
        && a.c / a.heads == D
        && a.t.is_multiple_of(ROWS_PER_TILE)
        && a.batch > 0
        && a.batch <= 65_535
        && a.heads <= 65_535
        && [a.qkv, a.p, a.att]
            .iter()
            .all(|&v| v != 0 && v.is_multiple_of(16))
        && a.cbuf().is_some()
}

/// Constant buffer 0: zero up to 0x700, then bytes 0x28..0x54 of [`Args::cbuf`] at 0x700..0x72c. `None` unless
/// [`fits`].
pub fn cbuf(a: &Args) -> Option<[u8; CBUF_BYTES]> {
    if !fits(a) {
        return None;
    }
    cbuf_from_f32(&a.cbuf()?)
}

/// The same move for a cbuf already laid out for `attn_fwd_tile3` (arguments at 0x28): `None` if it is shorter than
/// [`crate::attn_fwd::CBUF_BYTES`], its workgroup base (0x0..0xc) is not zero, or the shape it holds does not fit
/// (d ≠ 64, T not a multiple of 64, an address not a multiple of 16).
pub fn cbuf_from_f32(cb: &[u8]) -> Option<[u8; CBUF_BYTES]> {
    const ARGS: usize = 0x28;
    const END: usize = crate::attn_fwd::CBUF_BYTES;
    if cb.len() < END || cb[..0xc].iter().any(|&b| b != 0) {
        return None;
    }
    let mut out = [0u8; CBUF_BYTES];
    out[PUSH_BASE..PUSH_BASE + PUSH_BYTES].copy_from_slice(&cb[ARGS..END]);
    if fits_cbuf(&out) { Some(out) } else { None }
}

/// [`fits`] read from a constant buffer laid out by [`cbuf`]: the three addresses non-zero multiples of 16, T a
/// non-zero multiple of 64, d = 64 and C = 64·H. `false` if `cb` is too short.
pub fn fits_cbuf(cb: &[u8]) -> bool {
    const B: usize = PUSH_BASE;
    if cb.len() < B + PUSH_BYTES {
        return false;
    }
    let u64_at = |o: usize| u64::from_le_bytes(cb[B + o..B + o + 8].try_into().unwrap_or([1; 8]));
    let u32_at = |o: usize| u32::from_le_bytes(cb[B + o..B + o + 4].try_into().unwrap_or([1; 4]));
    let (t, c, h, d) = (u32_at(0x18), u32_at(0x1c), u32_at(0x20), u32_at(0x24));
    [0x00, 0x08, 0x10]
        .iter()
        .all(|&o| u64_at(o) != 0 && u64_at(o).is_multiple_of(16))
        && t != 0
        && t.is_multiple_of(ROWS_PER_TILE)
        && d == D
        && h != 0
        && c == D * h
}

/// The grid: [⌈⌈T/64⌉ / 2⌉, H, B] — block x does the row tiles x and ⌈T/64⌉ − 1 − x (one of them if they coincide).
pub fn grid(a: &Args) -> [u32; 3] {
    [a.t.div_ceil(ROWS_PER_TILE).div_ceil(2), a.heads, a.batch]
}

/// The row tiles block `bx` does, in order, for `nt` = T / 64 row tiles (the shader's pair loop).
pub fn row_tiles(bx: u32, nt: u32) -> ([u32; 2], usize) {
    let second = nt - 1 - bx;
    if second > bx {
        ([bx, second], 2)
    } else {
        ([bx, bx], 1)
    }
}

/// The QMD inputs of one launch (program, registers, cbuf, [`grid`], [`BLOCK`]); encode it with
/// [`ComputeQmd::encode_with_sm_config`] at the 64 KiB step and [`SHARED`]. `None` unless [`fits`].
pub fn qmd(
    a: &Args,
    program_va: u64,
    cbuf0_va: u64,
    release: Option<(u64, u32)>,
) -> Option<ComputeQmd> {
    if !fits(a) {
        return None;
    }
    Some(ComputeQmd {
        program_va,
        registers: REGISTERS,
        cbuf0_va,
        cbuf0_size: CBUF_BYTES as u32,
        grid: grid(a),
        block: BLOCK,
        release,
    })
}

// H1 in the constants: 2 blocks per SM at the 64 KiB step (shared with the 1 KiB reserve, registers).
const _: () = assert!(REGISTERS <= 128);
const _: () = assert!(2 * (SHARED.bytes + 1024) <= 64 * 1024);
const _: () = assert!(BLOCKS_PER_SM * BLOCK[0] * REGISTERS <= 65_536);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sha256::digest as sha256;

    const DUMP: &str = include_str!("../../../tools/nak/out2/attn_fwd_tc.sm86.txt");

    fn hex(b: &[u8]) -> alloc::string::String {
        use core::fmt::Write;
        let mut s = alloc::string::String::new();
        for x in b {
            let _ = write!(s, "{x:02x}");
        }
        s
    }

    fn a513() -> Args {
        Args {
            qkv: 0x10_0000,
            p: 0x20_0000,
            att: 0x30_0000,
            batch: 6,
            t: 512,
            c: 768,
            heads: 12,
            scale: 0.125,
        }
    }

    #[test]
    fn bytes_match_the_written_checksum() {
        assert_eq!(hex(&sha256(ATTN_FWD_TC)), ATTN_FWD_TC_SHA256);
        assert_eq!(ATTN_FWD_TC.len() % 16, 0);
    }

    /// H1 (the plan's prediction): ≤ 128 registers (2 blocks per SM), 0 local memory, no spills; the shared size of
    /// [`SHARED`]; the arguments read only at 0x700..0x72c.
    #[test]
    fn h1_registers_shared_local() {
        assert!(DUMP.contains(&alloc::format!("Num GPRs: {REGISTERS}\n")));
        assert!(DUMP.contains("SLM size: 0\n"));
        assert!(DUMP.contains("Spills to mem: 0\n"));
        assert!(DUMP.contains(&alloc::format!("Shared size: {}\n", SHARED.bytes)));
        for part in DUMP.split("c[0x0][").skip(1) {
            let off = part.trim_start_matches('+').split(']').next().unwrap();
            let v = u32::from_str_radix(off.trim_start_matches("0x"), 16).unwrap();
            assert!(
                v <= 0x8 || (0x700..0x700 + PUSH_BYTES as u32).contains(&v),
                "citire la {v:#x}"
            );
        }
    }

    /// H2 (listing, session 282): every operand read from shared memory is an `ldsm` (Q, K, P by rows / columns, Vᵀ
    /// by columns; no `movm`), the only plain shared loads are the 12 × 128-bit reads of the f32 S / O tile; all
    /// global loads / stores 128 bits; 16 `hmma` in pass 1, 16 + 16 in pass 2 (+ the 16-steps of P unrolled, 4 × 4);
    /// 13 barriers in the program (2 at a row tile's start, 3 per key tile in pass 1, 1 between the passes, 5 per key
    /// tile in pass 2, 2 at the end); `exp2` 16 + 1 per tile in pass 1, 16 in pass 2, 4 at the join; one `rcp`;
    /// 8 shuffles (the join of the 4 partials); no `ffma` (`precise`).
    #[test]
    fn h2_listing() {
        let n = |op: &str| DUMP.matches(op).count();
        assert_eq!(n("= ld.global"), 20);
        assert_eq!(
            n("= ld.global.a64.constant.b128"),
            20,
            "a global load that is not 128 bits"
        );
        assert_eq!(n("st.global"), 8);
        assert_eq!(
            n("st.global.a64.strong.gpu.b128"),
            8,
            "a global store that is not 128 bits"
        );
        assert_eq!(n("= ld.shared"), 12);
        assert_eq!(n("= ld.shared.strong.cta.b128"), 12);
        assert_eq!(n("ldsm.16.m8n8.x4"), 8);
        assert_eq!(n("ldsm.16.m8n8.x2"), 48);
        assert_eq!(
            n("movm"),
            0,
            "a transposed shared read: not every operand went through ldsm"
        );
        assert_eq!(n("hmma.m16n8k16.f32"), 48);
        assert_eq!(n("hmma"), 48);
        assert_eq!(n("bar.sync"), 13);
        assert_eq!(n("mufu.exp2"), 37);
        assert_eq!(n("mufu.rcp"), 1);
        assert_eq!(n("= shfl"), 8);
        assert_eq!(n("= ffma"), 0);
        assert_eq!(ATTN_FWD_TC.len(), 19_952);
    }

    #[test]
    fn fits_the_513_shape_and_refuses_the_rest() {
        let a = a513();
        assert!(fits(&a));
        let cb = cbuf(&a).unwrap();
        assert!(cb[..PUSH_BASE].iter().all(|&b| b == 0));
        assert_eq!(
            &cb[PUSH_BASE..PUSH_BASE + PUSH_BYTES],
            &a.cbuf().unwrap()[0x28..0x54]
        );
        assert!(cb[PUSH_BASE + PUSH_BYTES..].iter().all(|&b| b == 0));
        assert!(fits_cbuf(&cb));
        assert_eq!(cbuf_from_f32(&a.cbuf().unwrap()), Some(cb));
        assert_eq!(grid(&a), [4, 12, 6]);
        let q = qmd(&a, 0x1000, 0x2000, None).unwrap();
        assert_eq!(
            (q.grid, q.block, q.registers),
            ([4, 12, 6], BLOCK, REGISTERS)
        );
        for f in [
            |a: &mut Args| a.t = 500,
            |a: &mut Args| a.t = 96,
            |a: &mut Args| a.qkv += 4,
            |a: &mut Args| a.p += 8,
            |a: &mut Args| a.att += 12,
            |a: &mut Args| a.att = 0,
            |a: &mut Args| {
                a.c = 384;
                a.heads = 12
            },
            |a: &mut Args| a.heads = 0,
        ] {
            let mut b = a513();
            f(&mut b);
            assert!(!fits(&b));
            assert!(cbuf(&b).is_none());
            assert!(qmd(&b, 0x1000, 0x2000, None).is_none());
            if let Some(cb) = b.cbuf() {
                assert!(cbuf_from_f32(&cb).is_none());
            }
        }
        assert!(!fits_cbuf(&[0u8; 0x700]));
    }

    /// The pairs (T298d): every block does nt + 1 key tiles (the middle one of an odd nt does (nt + 1) / 2); every
    /// row tile exactly once.
    #[test]
    fn pairs_cover_every_row_tile_once() {
        for nt in 1..=16u32 {
            let mut seen = alloc::vec![0u32; nt as usize];
            for bx in 0..nt.div_ceil(2) {
                let (r, n) = row_tiles(bx, nt);
                let keys: u32 = r[..n].iter().map(|&x| x + 1).sum();
                if n == 2 {
                    assert_eq!(keys, nt + 1);
                } else {
                    assert_eq!(2 * bx + 1, nt);
                }
                for &x in &r[..n] {
                    seen[x as usize] += 1;
                }
            }
            assert!(seen.iter().all(|&s| s == 1), "nt = {nt}");
        }
    }
}
