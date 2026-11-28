//! C-168 p.5 P5d (`docs/C-168_P5_LIPIRE_PLAN.md` §12, session 294): the **forward residual** fused into the store of the
//! fast tensor-core programs — four programs, each the same source as its plain program with `-DRES`
//! (`tools/nak/mm_tc_v7.comp`, `tools/nak/mm_tc_v8.comp`, `tools/nak/mm_tc_v8r.comp`; second NAK tool, Mesa 26.3.1,
//! SM 8.6):
//!
//! - [`MM_TC_V7_RES`]: `mm_tc_v7` (64×64 blocks of 128 threads, 14 336 B shared) + the residual add;
//! - [`MM_TC_V8_RES`]: `mm_tc_v8` (128×128, 256 threads, 24 576 B) + the residual add;
//! - [`MM_TC_V8N_RES`]: `mm_tc_v8n` (128×64, 128 threads, 18 432 B) + the residual add;
//! - [`MM_TC_V8R_RES`]: `mm_tc_v8r` (v8's block, grid and shared memory, fewer register copies) + the residual add.
//!
//! Everything up to the value the plain program writes is unchanged (op(A)·op(B) (+ bias), the same bits as `mm_tc`);
//! then each element adds `res[row, col]` with one f32 add (`vadd`'s `fadd.ftz`) and writes that to C. In the training
//! step this replaces "out" / "proj" (`mm_tc*`, writing `tmp`) + the forward `vadd` (`x' = x + tmp`): one launch and
//! two n×C passes over memory fewer per residual. The constant buffer is `mm_tc`'s plus `res` at 0x730
//! ([`Args::cbuf`]); the block, grid and shared memory are the plain program's; only aligned products
//! ([`crate::mm_tc_v8::fits`] / [`crate::mm_tc_v7::fits`]). The CPU models are `gpu_sim::cmdproc::mm_tc_v7_res_on_cpu`
//! / `mm_tc_v8_res_on_cpu` / `mm_tc_v8n_res_on_cpu` / `mm_tc_v8r_res_on_cpu`. Tested **only in SIMULARE** (T311);
//! nothing here has run on a board.

use crate::mm_tc as tc;

/// `tools/nak/out2/mm_tc_v7_res.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v7_res`).
pub const MM_TC_V7_RES: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v7_res.sm86.sass");
/// SHA-256 of [`MM_TC_V7_RES`].
pub const MM_TC_V7_RES_SHA256: &str =
    "b1bf01ef80c7aec52124a778791d0ba1f7099fb47cde63266af052c4e96b7c62";
/// `tools/nak/out2/mm_tc_v8_res.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8_res`).
pub const MM_TC_V8_RES: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v8_res.sm86.sass");
/// SHA-256 of [`MM_TC_V8_RES`].
pub const MM_TC_V8_RES_SHA256: &str =
    "4346e09afccb2dd49b34e5a0f730f4605d842e4858075fd08e8f1aea7da702ca";
/// `tools/nak/out2/mm_tc_v8n_res.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8n_res`).
pub const MM_TC_V8N_RES: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v8n_res.sm86.sass");
/// SHA-256 of [`MM_TC_V8N_RES`].
pub const MM_TC_V8N_RES_SHA256: &str =
    "088ead60c2815d2b4dfe7d79e18f97c3f7ef1af3911f2d87c580ffebccb946ab";
/// `tools/nak/out2/mm_tc_v8r_res.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8r_res`).
pub const MM_TC_V8R_RES: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v8r_res.sm86.sass");
/// SHA-256 of [`MM_TC_V8R_RES`].
pub const MM_TC_V8R_RES_SHA256: &str =
    "650480bb8f223515c898519f2651136ef855f8c79951afa22f3a4e7140303a43";

/// Registers per thread of [`MM_TC_V7_RES`] (NAK dump `mm_tc_v7_res.sm86.txt`; `mm_tc_v7`: 128).
pub const REGISTERS_V7: u32 = 128;
/// Registers per thread of [`MM_TC_V8_RES`] (NAK dump `mm_tc_v8_res.sm86.txt`; `mm_tc_v8`: 128).
pub const REGISTERS_V8: u32 = 128;
/// Registers per thread of [`MM_TC_V8N_RES`] (NAK dump `mm_tc_v8n_res.sm86.txt`; `mm_tc_v8n`: 144).
pub const REGISTERS_V8N: u32 = 144;
/// Registers per thread of [`MM_TC_V8R_RES`] (NAK dump `mm_tc_v8r_res.sm86.txt`; `mm_tc_v8r`: 120).
pub const REGISTERS_V8R: u32 = 120;

/// Where `res` sits in constant buffer 0 (right after `mm_tc`'s 0x30 bytes of arguments; the place of `act` in
/// [`crate::mm_tc_fast_gelu_fwd`]).
pub const RES_AT: usize = tc::PUSH_BASE + 0x30;

/// One product `C = (op(A)·op(B) (+ bias)) + res`; C and `res` are `[m, n]` f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Args {
    /// The product exactly as the plain fast program takes it.
    pub tc: tc::Args,
    /// `[m, n]` f32: the residual added to every element. Must not overlap A, B, C or the bias.
    pub res: u64,
}

impl Args {
    /// Constant buffer 0: `mm_tc`'s ([`tc::Args::cbuf`]) with `res` at 0x730. `None` when `mm_tc` refuses the product,
    /// or `res` is 0, not a multiple of 4, or overlaps A, B, C or the bias.
    pub fn cbuf(&self) -> Option<[u8; tc::CBUF_BYTES]> {
        let mut cb = self.tc.cbuf()?;
        let t = &self.tc;
        let (m, n, k) = (u64::from(t.m), u64::from(t.n), u64::from(t.k));
        let len = 4 * m * n;
        let over = |va: u64, l: u64| va < self.res + len && self.res < va + l;
        if self.res == 0
            || !self.res.is_multiple_of(4)
            || over(t.c, len)
            || over(t.a, 4 * m * k)
            || over(t.b, 4 * k * n)
            || (t.flags & tc::BIAS != 0 && over(t.bias, 4 * n))
        {
            return None;
        }
        cb[RES_AT..RES_AT + 8].copy_from_slice(&self.res.to_le_bytes());
        Some(cb)
    }
}

/// Checks the four programs against their SHA-256 (as `train_step::check_programs` does for the others).
pub fn check() -> Result<(), &'static str> {
    for (code, sha, name) in [
        (MM_TC_V7_RES, MM_TC_V7_RES_SHA256, "mm_tc_v7_res"),
        (MM_TC_V8_RES, MM_TC_V8_RES_SHA256, "mm_tc_v8_res"),
        (MM_TC_V8N_RES, MM_TC_V8N_RES_SHA256, "mm_tc_v8n_res"),
        (MM_TC_V8R_RES, MM_TC_V8R_RES_SHA256, "mm_tc_v8r_res"),
    ] {
        if code.is_empty() || code.len() % 16 != 0 || hex(&crate::sha256::digest(code)) != sha {
            return Err(name);
        }
    }
    Ok(())
}

fn hex(d: &[u8]) -> alloc::string::String {
    d.iter().map(|b| alloc::format!("{b:02x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn programs_are_the_compiled_ones() {
        check().unwrap();
        assert_ne!(MM_TC_V7_RES, crate::mm_tc_v7::MM_TC_V7);
        assert_ne!(MM_TC_V8_RES, crate::mm_tc_v8::MM_TC_V8);
        assert_ne!(MM_TC_V8N_RES, crate::mm_tc_v8::MM_TC_V8N);
        assert_ne!(MM_TC_V8R_RES, crate::mm_tc_v8r::MM_TC_V8R);
        assert_ne!(MM_TC_V8_RES, MM_TC_V8R_RES);
        assert_ne!(MM_TC_V8_RES, crate::mm_tc_fast_gelu_fwd::MM_TC_V8_GELUF);
    }

    fn args() -> Args {
        Args {
            tc: tc::Args {
                a: 0x10_0000,
                b: 0x20_0000,
                c: 0x30_0000,
                bias: 0x40_0000,
                m: 256,
                n: 512,
                k: 64,
                flags: tc::TB | tc::BIAS,
            },
            res: 0x50_0000,
        }
    }

    #[test]
    fn cbuf_is_mm_tc_plus_res() {
        let g = args();
        let cb = g.cbuf().unwrap();
        let plain = g.tc.cbuf().unwrap();
        assert_eq!(cb[..RES_AT], plain[..RES_AT]);
        assert_eq!(cb[RES_AT..RES_AT + 8], 0x50_0000u64.to_le_bytes());
        assert!(cb[RES_AT + 8..].iter().all(|&b| b == 0));
        assert_eq!(RES_AT, 0x730);
    }

    #[test]
    fn refusals() {
        for res in [
            0u64,
            0x50_0002,
            0x30_0000,
            0x30_0000 + 4 * 256 * 512 - 4,
            0x10_0000,
            0x20_0000,
            0x40_0000,
        ] {
            assert_eq!(Args { res, ..args() }.cbuf(), None, "res {res:#x}");
        }
        assert!(
            Args {
                res: 0x40_0000 + 4 * 512,
                ..args()
            }
            .cbuf()
            .is_some()
        );
        let mut g = args();
        g.tc.m = 0;
        assert_eq!(g.cbuf(), None);
    }

    #[test]
    fn registers_keep_the_blocks_per_sm() {
        // M2 (docs/C-168_P5_LIPIRE_PLAN.md §12): v7 4, v8 / v8r 2, v8n 3 blocks per SM at 65 536 registers.
        assert_eq!(65_536 / (128 * REGISTERS_V7), 4);
        assert_eq!(
            65_536 / (256 * REGISTERS_V8),
            crate::mm_tc_v8::BLOCKS_PER_SM
        );
        assert_eq!(
            65_536 / (256 * REGISTERS_V8R),
            crate::mm_tc_v8::BLOCKS_PER_SM
        );
        assert_eq!(
            65_536 / (128 * REGISTERS_V8N),
            crate::mm_tc_v8::BLOCKS_PER_SM_N
        );
    }
}
