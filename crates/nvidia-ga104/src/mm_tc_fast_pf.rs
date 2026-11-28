//! C-168 p.5 P5e (`docs/C-168_P5_LIPIRE_PLAN.md` §15, session 295): the `_gelu` (P5b) and `_res` (P5d) programs with
//! the **tail reads issued one fragment ahead** — the same sources with `-DPF` (`tools/nak/mm_tc_v7.comp`,
//! `tools/nak/mm_tc_v8.comp`, `tools/nak/mm_tc_v8r.comp`; second NAK tool, Mesa 26.3.1, SM 8.6).
//!
//! On the board (os-512, §14) the `_gelu` tail read `u` one element at a time: each of a thread's 64 reads waited for
//! the memory alone (~190 µs more per launch at batch 6 than the arithmetic predicted). In the `_pf` programs the reads
//! of the next 16×8 (v8 / v8n / v8r) or 16×16 (v7) fragment are issued before the current one is worked, at clamped
//! (always valid) addresses; the arithmetic on every element is letter for letter the program's without `-DPF`, on the
//! same value (`tools/nak/compara_pf.py`: the same floating-point operations in the same order, 8 / 16 reads waiting at
//! once instead of 1). The registers, block, grid, shared memory and constant buffer are the ones of the program each
//! replaces, so a `_pf` program goes into the same slot of [`crate::train_step::GeluFast`] /
//! [`crate::train_step::ResFast`] with nothing else changed. Tested **only in SIMULARE** (T312); nothing here has run
//! on a board.

/// `tools/nak/out2/mm_tc_v7_gelu_pf.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v7_gelu_pf`): `mm_tc_v7_gelu` with `-DPF`.
pub const MM_TC_V7_GELU_PF: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v7_gelu_pf.sm86.sass");
/// SHA-256 of [`MM_TC_V7_GELU_PF`].
pub const MM_TC_V7_GELU_PF_SHA256: &str =
    "4c5a3e96d27f4133c9eafb0118d3e4d798cca73b1633cd1d7af3aac01613e713";
/// `tools/nak/out2/mm_tc_v8_gelu_pf.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8_gelu_pf`): `mm_tc_v8_gelu` with `-DPF`.
pub const MM_TC_V8_GELU_PF: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v8_gelu_pf.sm86.sass");
/// SHA-256 of [`MM_TC_V8_GELU_PF`].
pub const MM_TC_V8_GELU_PF_SHA256: &str =
    "7f8a0e9478e2708653bed3ec6d7ce637b62260295422a023df735594f0ac5023";
/// `tools/nak/out2/mm_tc_v8n_gelu_pf.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8n_gelu_pf`): `mm_tc_v8n_gelu` with `-DPF`.
pub const MM_TC_V8N_GELU_PF: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v8n_gelu_pf.sm86.sass");
/// SHA-256 of [`MM_TC_V8N_GELU_PF`].
pub const MM_TC_V8N_GELU_PF_SHA256: &str =
    "f90955d272158714aae0ba269b048a44cc9c0f2441514b2bfb2717dbd83417a2";
/// `tools/nak/out2/mm_tc_v7_res_pf.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v7_res_pf`): `mm_tc_v7_res` with `-DPF`.
pub const MM_TC_V7_RES_PF: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v7_res_pf.sm86.sass");
/// SHA-256 of [`MM_TC_V7_RES_PF`].
pub const MM_TC_V7_RES_PF_SHA256: &str =
    "f4948df06a49d301061346e3b26c3e1113980bd3bee070180ef6177764d3fc0b";
/// `tools/nak/out2/mm_tc_v8_res_pf.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8_res_pf`): `mm_tc_v8_res` with `-DPF`.
pub const MM_TC_V8_RES_PF: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v8_res_pf.sm86.sass");
/// SHA-256 of [`MM_TC_V8_RES_PF`].
pub const MM_TC_V8_RES_PF_SHA256: &str =
    "e8adabc871bc54c6342acf879081c5511c99459f761ac33690a27e6ade3048b5";
/// `tools/nak/out2/mm_tc_v8n_res_pf.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8n_res_pf`): `mm_tc_v8n_res` with `-DPF`.
pub const MM_TC_V8N_RES_PF: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v8n_res_pf.sm86.sass");
/// SHA-256 of [`MM_TC_V8N_RES_PF`].
pub const MM_TC_V8N_RES_PF_SHA256: &str =
    "5c32a4e31bc262863448ed2372d475612b2cd2ec79afbf33fe82709a523cb7c6";
/// `tools/nak/out2/mm_tc_v8r_res_pf.sm86.sass` (`sh tools/nak/build2.sh mm_tc_v8r_res_pf`): `mm_tc_v8r_res` with `-DPF`.
pub const MM_TC_V8R_RES_PF: &[u8] =
    include_bytes!("../../../tools/nak/out2/mm_tc_v8r_res_pf.sm86.sass");
/// SHA-256 of [`MM_TC_V8R_RES_PF`].
pub const MM_TC_V8R_RES_PF_SHA256: &str =
    "5ed7ad98fcea454b83d1239f1ebe0e28569f7f7179437d938390c717357a20bd";

/// Registers per thread of [`MM_TC_V7_GELU_PF`] (NAK dump `mm_tc_v7_gelu_pf.sm86.txt`; `mm_tc_v7_gelu`: 128).
pub const REGISTERS_V7_GELU: u32 = 128;
/// Registers per thread of [`MM_TC_V8_GELU_PF`] (NAK dump `mm_tc_v8_gelu_pf.sm86.txt`; `mm_tc_v8_gelu`: 128).
pub const REGISTERS_V8_GELU: u32 = 128;
/// Registers per thread of [`MM_TC_V8N_GELU_PF`] (NAK dump `mm_tc_v8n_gelu_pf.sm86.txt`; `mm_tc_v8n_gelu`: 144).
pub const REGISTERS_V8N_GELU: u32 = 144;
/// Registers per thread of [`MM_TC_V7_RES_PF`] (NAK dump `mm_tc_v7_res_pf.sm86.txt`; `mm_tc_v7_res`: 128).
pub const REGISTERS_V7_RES: u32 = 128;
/// Registers per thread of [`MM_TC_V8_RES_PF`] (NAK dump `mm_tc_v8_res_pf.sm86.txt`; `mm_tc_v8_res`: 128).
pub const REGISTERS_V8_RES: u32 = 128;
/// Registers per thread of [`MM_TC_V8N_RES_PF`] (NAK dump `mm_tc_v8n_res_pf.sm86.txt`; `mm_tc_v8n_res`: 144).
pub const REGISTERS_V8N_RES: u32 = 144;
/// Registers per thread of [`MM_TC_V8R_RES_PF`] (NAK dump `mm_tc_v8r_res_pf.sm86.txt`; `mm_tc_v8r_res`: 120).
pub const REGISTERS_V8R_RES: u32 = 120;

/// Checks the seven programs against their SHA-256 (as `train_step::check_programs` does for the others).
pub fn check() -> Result<(), &'static str> {
    for (code, sha, name) in [
        (
            MM_TC_V7_GELU_PF,
            MM_TC_V7_GELU_PF_SHA256,
            "mm_tc_v7_gelu_pf",
        ),
        (
            MM_TC_V8_GELU_PF,
            MM_TC_V8_GELU_PF_SHA256,
            "mm_tc_v8_gelu_pf",
        ),
        (
            MM_TC_V8N_GELU_PF,
            MM_TC_V8N_GELU_PF_SHA256,
            "mm_tc_v8n_gelu_pf",
        ),
        (MM_TC_V7_RES_PF, MM_TC_V7_RES_PF_SHA256, "mm_tc_v7_res_pf"),
        (MM_TC_V8_RES_PF, MM_TC_V8_RES_PF_SHA256, "mm_tc_v8_res_pf"),
        (
            MM_TC_V8N_RES_PF,
            MM_TC_V8N_RES_PF_SHA256,
            "mm_tc_v8n_res_pf",
        ),
        (
            MM_TC_V8R_RES_PF,
            MM_TC_V8R_RES_PF_SHA256,
            "mm_tc_v8r_res_pf",
        ),
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
    fn programs_are_the_compiled_ones_and_new() {
        check().unwrap();
        assert_ne!(MM_TC_V7_GELU_PF, crate::mm_tc_fast_gelu::MM_TC_V7_GELU);
        assert_ne!(MM_TC_V8_GELU_PF, crate::mm_tc_fast_gelu::MM_TC_V8_GELU);
        assert_ne!(MM_TC_V8N_GELU_PF, crate::mm_tc_fast_gelu::MM_TC_V8N_GELU);
        assert_ne!(MM_TC_V7_RES_PF, crate::mm_tc_fast_res::MM_TC_V7_RES);
        assert_ne!(MM_TC_V8_RES_PF, crate::mm_tc_fast_res::MM_TC_V8_RES);
        assert_ne!(MM_TC_V8N_RES_PF, crate::mm_tc_fast_res::MM_TC_V8N_RES);
        assert_ne!(MM_TC_V8R_RES_PF, crate::mm_tc_fast_res::MM_TC_V8R_RES);
    }

    #[test]
    fn registers_are_the_replaced_programs() {
        // N2 (docs/C-168_P5_LIPIRE_PLAN.md §15): the registers of the program each one replaces, so the QMDs the packer
        // writes for the `_gelu` / `_res` slot stay right.
        assert_eq!(REGISTERS_V7_GELU, crate::mm_tc_fast_gelu::REGISTERS_V7);
        assert_eq!(REGISTERS_V8_GELU, crate::mm_tc_fast_gelu::REGISTERS_V8);
        assert_eq!(REGISTERS_V8N_GELU, crate::mm_tc_fast_gelu::REGISTERS_V8N);
        assert_eq!(REGISTERS_V7_RES, crate::mm_tc_fast_res::REGISTERS_V7);
        assert_eq!(REGISTERS_V8_RES, crate::mm_tc_fast_res::REGISTERS_V8);
        assert_eq!(REGISTERS_V8N_RES, crate::mm_tc_fast_res::REGISTERS_V8N);
        assert_eq!(REGISTERS_V8R_RES, crate::mm_tc_fast_res::REGISTERS_V8R);
    }
}
