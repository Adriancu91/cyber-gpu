//! C-168 p.5 P5b (`docs/C-168_P5_LIPIRE_PLAN.md` §7, session 291): GELU's backward fused into the store of the **fast**
//! tensor-core programs — three programs, each the same source as its non-GELU program with `-DGELU`
//! (`tools/nak/mm_tc_v7.comp`, `tools/nak/mm_tc_v8.comp`; second NAK tool, Mesa 26.3.1, SM 8.6):
//!
//! - [`MM_TC_V7_GELU`]: `mm_tc_v7` (64×64 blocks of 128 threads, 14 336 B shared) with `mm_tc_gelu`'s store;
//! - [`MM_TC_V8_GELU`]: `mm_tc_v8` (128×128, 256 threads, 24 576 B) with `mm_tc_gelu`'s store;
//! - [`MM_TC_V8N_GELU`]: `mm_tc_v8n` (128×64, 128 threads, 18 432 B) with `mm_tc_gelu`'s store.
//!
//! Everything up to the final store is the non-GELU program's (the same f16 values through the same `hmma` chain as
//! `mm_tc`); each finished sum `v` is written as `v · (Φ(u) + u·φ(u))` with exactly `mm_tc_gelu.comp`'s (= `gelu_tr`
//! mode 1's) operations, `u` read at `[row, col]` from the bias slot. The constant buffer is
//! [`crate::mm_tc_gelu::Args::cbuf`]; the block, grid and shared memory are the non-GELU program's; only aligned
//! products ([`crate::mm_tc_v8::fits`]). The CPU models are `gpu_sim::cmdproc::mm_tc_v7_gelu_on_cpu` /
//! `mm_tc_v8_gelu_on_cpu` / `mm_tc_v8n_gelu_on_cpu`. Tested **only in SIMULARE** (T307); nothing here has run on a
//! board.

use crate::mm_tc as tc;
use crate::mm_tc_gelu as tg;

/// `tools/nak/out2/mm_tc_v7_gelu.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v7_gelu`).
pub const MM_TC_V7_GELU: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v7_gelu.sm86.sass");
/// SHA-256 of [`MM_TC_V7_GELU`].
pub const MM_TC_V7_GELU_SHA256: &str =
    "ed8082045e2255d96dbcca4f42bb44bb4e199a91e57e1dc03e95d57c756a10c2";
/// `tools/nak/out2/mm_tc_v8_gelu.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8_gelu`).
pub const MM_TC_V8_GELU: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v8_gelu.sm86.sass");
/// SHA-256 of [`MM_TC_V8_GELU`].
pub const MM_TC_V8_GELU_SHA256: &str =
    "d0d061ce7711fdf9f1a9e56ea52162fc84752f23f53987e78907be694af05417";
/// `tools/nak/out2/mm_tc_v8n_gelu.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8n_gelu`).
pub const MM_TC_V8N_GELU: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v8n_gelu.sm86.sass");
/// SHA-256 of [`MM_TC_V8N_GELU`].
pub const MM_TC_V8N_GELU_SHA256: &str =
    "81e6e4ab50145e08dfa04627a0fc46afb17ba331087f547ddb7bd3c3fa259714";

/// Registers per thread of [`MM_TC_V7_GELU`] (NAK dump `mm_tc_v7_gelu.sm86.txt`; `mm_tc_v7`: 128).
pub const REGISTERS_V7: u32 = 128;
/// Registers per thread of [`MM_TC_V8_GELU`] (NAK dump `mm_tc_v8_gelu.sm86.txt`; `mm_tc_v8`: 128).
pub const REGISTERS_V8: u32 = 128;
/// Registers per thread of [`MM_TC_V8N_GELU`] (NAK dump `mm_tc_v8n_gelu.sm86.txt`; `mm_tc_v8n`: 144).
pub const REGISTERS_V8N: u32 = 144;

/// The `mm_tc` arguments of the same product (u in the bias slot, flags TA / TB): what the pick, [`crate::mm_tc_v8::fits`]
/// and the grids take.
pub fn tc_args(args: &tg::Args) -> tc::Args {
    tc::Args {
        a: args.a,
        b: args.b,
        c: args.c,
        bias: args.u,
        m: args.m,
        n: args.n,
        k: args.k,
        flags: args.flags,
    }
}

/// Checks the three programs against their SHA-256 (as `train_step::check_programs` does for the others).
pub fn check() -> Result<(), &'static str> {
    for (code, sha, name) in [
        (MM_TC_V7_GELU, MM_TC_V7_GELU_SHA256, "mm_tc_v7_gelu"),
        (MM_TC_V8_GELU, MM_TC_V8_GELU_SHA256, "mm_tc_v8_gelu"),
        (MM_TC_V8N_GELU, MM_TC_V8N_GELU_SHA256, "mm_tc_v8n_gelu"),
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
        assert_ne!(MM_TC_V7_GELU, crate::mm_tc_v7::MM_TC_V7);
        assert_ne!(MM_TC_V8_GELU, crate::mm_tc_v8::MM_TC_V8);
        assert_ne!(MM_TC_V8N_GELU, crate::mm_tc_v8::MM_TC_V8N);
        assert_ne!(MM_TC_V8_GELU, MM_TC_V8N_GELU);
    }

    #[test]
    fn tc_args_is_the_same_product() {
        let g = tg::Args {
            a: 0x1000,
            b: 0x2000,
            c: 0x3000,
            u: 0x4000,
            m: 256,
            n: 512,
            k: 64,
            flags: tc::TB,
        };
        let t = tc_args(&g);
        assert_eq!(t.bias, g.u);
        assert_eq!(t.cbuf().unwrap(), g.cbuf().unwrap());
        assert!(crate::mm_tc_v8::fits(&t));
    }

    #[test]
    fn registers_keep_the_blocks_per_sm() {
        // J2 (docs/C-168_P5_LIPIRE_PLAN.md §7): the same registers as the non-GELU programs, so the same blocks per SM
        // at 65 536 registers: v7 4, v8 2, v8n 3.
        assert_eq!(REGISTERS_V7, crate::mm_tc_v7::REGISTERS);
        assert_eq!(REGISTERS_V8, crate::mm_tc_v8::REGISTERS);
        assert_eq!(REGISTERS_V8N, crate::mm_tc_v8::REGISTERS_N);
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
