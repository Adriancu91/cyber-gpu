//! C-168 p.5 P5c (`docs/C-168_P5_LIPIRE_PLAN.md` §9, session 293): GELU's **forward** fused into the store of the fast
//! tensor-core programs — three programs, each the same source as its plain program with `-DGELU_FWD`
//! (`tools/nak/mm_tc_v7.comp`, `tools/nak/mm_tc_v8.comp`; second NAK tool, Mesa 26.3.1, SM 8.6):
//!
//! - [`MM_TC_V7_GELUF`]: `mm_tc_v7` (64×64 blocks of 128 threads, 14 336 B shared) + the `act` store;
//! - [`MM_TC_V8_GELUF`]: `mm_tc_v8` (128×128, 256 threads, 24 576 B) + the `act` store;
//! - [`MM_TC_V8N_GELUF`]: `mm_tc_v8n` (128×64, 128 threads, 18 432 B) + the `act` store.
//!
//! Everything up to and including the write of C is the plain program's (C = u = op(A)·op(B) (+ bias), the same bits as
//! `mm_tc`); then each element also writes `act = u·Φ(u)` with exactly `gelu_tr.comp` mode 0's operations on the value
//! just written. `u` stays in C because GELU's backward reads it (P5b). In the training step this replaces "fc"
//! (`mm_tc*`) + the forward `gelu_tr`: one launch and one read of `u` fewer per layer. The constant buffer is `mm_tc`'s
//! plus `act` at 0x730 ([`Args::cbuf`]); the block, grid and shared memory are the plain program's; only aligned
//! products ([`crate::mm_tc_v8::fits`] / [`crate::mm_tc_v7::fits`]). The CPU models are
//! `gpu_sim::cmdproc::mm_tc_v7_geluf_on_cpu` / `mm_tc_v8_geluf_on_cpu` / `mm_tc_v8n_geluf_on_cpu`. Tested **only in
//! SIMULARE** (T310); nothing here has run on a board.

use crate::mm_tc as tc;

/// `tools/nak/out2/mm_tc_v7_geluf.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v7_geluf`).
pub const MM_TC_V7_GELUF: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v7_geluf.sm86.sass");
/// SHA-256 of [`MM_TC_V7_GELUF`].
pub const MM_TC_V7_GELUF_SHA256: &str =
    "8e869b7ff19cd77c142e9edba57123c55281c78a3bd984bb83eedb44ce9c3fe1";
/// `tools/nak/out2/mm_tc_v8_geluf.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8_geluf`).
pub const MM_TC_V8_GELUF: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v8_geluf.sm86.sass");
/// SHA-256 of [`MM_TC_V8_GELUF`].
pub const MM_TC_V8_GELUF_SHA256: &str =
    "82aceab9f48d727083f7dff91f7f271663e83dc8887630ea3b49561c5dd4bfbf";
/// `tools/nak/out2/mm_tc_v8n_geluf.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8n_geluf`).
pub const MM_TC_V8N_GELUF: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v8n_geluf.sm86.sass");
/// SHA-256 of [`MM_TC_V8N_GELUF`].
pub const MM_TC_V8N_GELUF_SHA256: &str =
    "8e8890a4276f248fd7c36925706558955310b51a006a0716f4dc98ffbbea5a07";

/// Registers per thread of [`MM_TC_V7_GELUF`] (NAK dump `mm_tc_v7_geluf.sm86.txt`; `mm_tc_v7`: 128).
pub const REGISTERS_V7: u32 = 128;
/// Registers per thread of [`MM_TC_V8_GELUF`] (NAK dump `mm_tc_v8_geluf.sm86.txt`; `mm_tc_v8`: 128).
pub const REGISTERS_V8: u32 = 128;
/// Registers per thread of [`MM_TC_V8N_GELUF`] (NAK dump `mm_tc_v8n_geluf.sm86.txt`; `mm_tc_v8n`: 144).
pub const REGISTERS_V8N: u32 = 144;

/// Where `act` sits in constant buffer 0 (right after `mm_tc`'s 0x30 bytes of arguments).
pub const ACT_AT: usize = tc::PUSH_BASE + 0x30;

/// One product `u = op(A)·op(B) (+ bias)` written to C, plus `act = GELU(u)`; C and `act` are `[m, n]` f32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Args {
    /// The product exactly as the plain fast program takes it (C = u).
    pub tc: tc::Args,
    /// `[m, n]` f32: GELU(u). Must not overlap A, B, C or the bias.
    pub act: u64,
}

impl Args {
    /// Constant buffer 0: `mm_tc`'s ([`tc::Args::cbuf`]) with `act` at 0x730. `None` when `mm_tc` refuses the product,
    /// or `act` is 0, not a multiple of 4, or overlaps A, B, C or the bias.
    pub fn cbuf(&self) -> Option<[u8; tc::CBUF_BYTES]> {
        let mut cb = self.tc.cbuf()?;
        let t = &self.tc;
        let (m, n, k) = (u64::from(t.m), u64::from(t.n), u64::from(t.k));
        let len = 4 * m * n;
        let over = |va: u64, l: u64| va < self.act + len && self.act < va + l;
        if self.act == 0
            || !self.act.is_multiple_of(4)
            || over(t.c, len)
            || over(t.a, 4 * m * k)
            || over(t.b, 4 * k * n)
            || (t.flags & tc::BIAS != 0 && over(t.bias, 4 * n))
        {
            return None;
        }
        cb[ACT_AT..ACT_AT + 8].copy_from_slice(&self.act.to_le_bytes());
        Some(cb)
    }
}

/// Checks the three programs against their SHA-256 (as `train_step::check_programs` does for the others).
pub fn check() -> Result<(), &'static str> {
    for (code, sha, name) in [
        (MM_TC_V7_GELUF, MM_TC_V7_GELUF_SHA256, "mm_tc_v7_geluf"),
        (MM_TC_V8_GELUF, MM_TC_V8_GELUF_SHA256, "mm_tc_v8_geluf"),
        (MM_TC_V8N_GELUF, MM_TC_V8N_GELUF_SHA256, "mm_tc_v8n_geluf"),
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
        assert_ne!(MM_TC_V7_GELUF, crate::mm_tc_v7::MM_TC_V7);
        assert_ne!(MM_TC_V8_GELUF, crate::mm_tc_v8::MM_TC_V8);
        assert_ne!(MM_TC_V8N_GELUF, crate::mm_tc_v8::MM_TC_V8N);
        assert_ne!(MM_TC_V8_GELUF, MM_TC_V8N_GELUF);
        assert_ne!(MM_TC_V8_GELUF, crate::mm_tc_fast_gelu::MM_TC_V8_GELU);
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
            act: 0x50_0000,
        }
    }

    #[test]
    fn cbuf_is_mm_tc_plus_act() {
        let g = args();
        let cb = g.cbuf().unwrap();
        let plain = g.tc.cbuf().unwrap();
        assert_eq!(cb[..ACT_AT], plain[..ACT_AT]);
        assert_eq!(cb[ACT_AT..ACT_AT + 8], 0x50_0000u64.to_le_bytes());
        assert!(cb[ACT_AT + 8..].iter().all(|&b| b == 0));
        assert_eq!(ACT_AT, 0x730);
    }

    #[test]
    fn refusals() {
        for act in [
            0u64,
            0x50_0002,
            0x30_0000,
            0x30_0000 + 4 * 256 * 512 - 4,
            0x10_0000,
            0x20_0000,
        ] {
            assert_eq!(Args { act, ..args() }.cbuf(), None, "act {act:#x}");
        }
        // the bias is n floats only: just after it is fine
        assert!(
            Args {
                act: 0x40_0000 + 4 * 512,
                ..args()
            }
            .cbuf()
            .is_some()
        );
        assert_eq!(
            Args {
                act: 0x40_0000,
                ..args()
            }
            .cbuf(),
            None
        );
        // mm_tc's own refusals carry over
        let mut g = args();
        g.tc.m = 0;
        assert_eq!(g.cbuf(), None);
    }

    #[test]
    fn registers_keep_the_blocks_per_sm() {
        // L2 (docs/C-168_P5_LIPIRE_PLAN.md §9): v7 4, v8 2, v8n 3 blocks per SM at 65 536 registers.
        assert_eq!(65_536 / (128 * REGISTERS_V7), 4);
        assert_eq!(
            65_536 / (256 * REGISTERS_V8),
            crate::mm_tc_v8::BLOCKS_PER_SM
        );
        assert_eq!(
            65_536 / (128 * REGISTERS_V8N),
            crate::mm_tc_v8::BLOCKS_PER_SM_N
        );
    }
}
