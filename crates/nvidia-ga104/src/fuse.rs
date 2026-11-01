//! Ucode fuse versions (read-only) — which signature the boot ROM accepts.
//!
//! A signed HS program (FWSEC from the VBIOS, Booter from the firmware files)
//! carries several signatures, one per "ucode version". The board's fuses say
//! which version it has burnt in; the driver must patch the matching
//! signature into the image before loading it, otherwise the falcon boot ROM
//! refuses the program. Before this module the fuse register was NECUNOSCUT
//! for GA10x, so no signature could be chosen for the real board.
//!
//! Sources (read as specification, code written by us):
//! - open-gpu-kernel-modules@570.144 (MIT),
//!   `src/common/inc/swref/published/ampere/ga100/dev_fuse.h`:
//!   `NV_FUSE_OPT_FPF_SEC2_UCODE1_VERSION` = 0x00824140,
//!   `NV_FUSE_OPT_FPF_GSP_UCODE1_VERSION` = 0x008241C0, field `DATA` 15:0;
//! - `src/nvidia/src/kernel/gpu/gsp/arch/ampere/kernel_gsp_ga100.c`
//!   (`kgspReadUcodeFuseVersion_GA100`) and
//!   `src/nvidia/src/kernel/gpu/sec2/arch/ampere/kernel_sec2_ga100.c`
//!   (`ksec2ReadUcodeFuseVersion_GA100`): register = `UCODE1 + 4 * (id - 1)`
//!   for ucode ids 1..=16; version = index of the highest set bit + 1, or 0
//!   when the register reads 0;
//! - `src/nvidia/generated/g_kernel_gsp_nvoc.c` / `g_kernel_sec2_nvoc.c`:
//!   every chip that is not Turing (GA102/GA104 included) uses the GA100
//!   variant above.
//!
//! The use of the version (bit `1 << version` of the descriptor's signature
//! mask, index = set bits below it) is in [`crate::fwsec::signature_index`]
//! (`kernel_gsp_frts_tu102.c`), and for Booter in
//! [`crate::gsp::hs_load::booter_signature_index`].
//!
//! The registers lie at ~8 MiB, outside the 4 MiB window I2/I3 need: reading
//! them needs the whole 16 MiB BAR0 mapped ([`FUSE_WINDOW`]). One journaled
//! read per call; nothing is written.

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Mmio, checked_read32};

use crate::fwsec::{FalconUcodeDescV3, signature_index};
use crate::regs::is_priv_error;

/// `NV_FUSE_OPT_FPF_SEC2_UCODE1_VERSION` (ga100 `dev_fuse.h`, MIT).
pub const NV_FUSE_OPT_FPF_SEC2_UCODE1_VERSION: u64 = 0x0082_4140;
/// `NV_FUSE_OPT_FPF_GSP_UCODE1_VERSION` (ga100 `dev_fuse.h`, MIT).
pub const NV_FUSE_OPT_FPF_GSP_UCODE1_VERSION: u64 = 0x0082_41c0;
/// RM reads ids 1..=16 (`index < 16`).
pub const MAX_UCODE_ID: u8 = 16;
/// BAR0 mapping needed to reach the fuse block: the whole BAR0 (16 MiB).
pub const FUSE_WINDOW: u64 = 0x0100_0000;

// Both blocks fit in BAR0 and lie outside the 4 MiB I2/I3 window.
const _: () = assert!(NV_FUSE_OPT_FPF_GSP_UCODE1_VERSION + 4 * MAX_UCODE_ID as u64 <= FUSE_WINDOW);
const _: () = assert!(NV_FUSE_OPT_FPF_SEC2_UCODE1_VERSION >= crate::regs::INIT_WINDOW);

/// Which falcon the program runs on (FWSEC: GSP; Booter: SEC2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FuseEngine {
    Gsp,
    Sec2,
}

/// BAR0 offset of the fuse for `ucode_id` (1-based) on `engine`.
pub fn ucode_fuse_offset(engine: FuseEngine, ucode_id: u8) -> GpuResult<u64> {
    if ucode_id == 0 || ucode_id > MAX_UCODE_ID {
        return Err(GpuError::InvalidArgument(
            "siguranta: ucode_id trebuie sa fie intre 1 si 16",
        ));
    }
    let base = match engine {
        FuseEngine::Gsp => NV_FUSE_OPT_FPF_GSP_UCODE1_VERSION,
        FuseEngine::Sec2 => NV_FUSE_OPT_FPF_SEC2_UCODE1_VERSION,
    };
    Ok(base + 4 * u64::from(ucode_id - 1))
}

/// Raw register value → version. Only `DATA` (15:0) is defined; anything in
/// 31:16, a priv error (0xbadf....) or an absent device is refused instead
/// of being turned into a version.
pub fn decode_ucode_fuse(raw: u32) -> GpuResult<u8> {
    if raw == 0xffff_ffff {
        return Err(GpuError::NotResponding);
    }
    if is_priv_error(raw) {
        return Err(GpuError::HardwareFault { code: raw });
    }
    if raw >> 16 != 0 {
        return Err(GpuError::Blocked(
            "siguranta: biti nedocumentati (31:16) nenuli; nu aleg semnatura",
        ));
    }
    if raw == 0 {
        return Ok(0);
    }
    Ok((32 - raw.leading_zeros()) as u8)
}

/// Reads the fuse version of `ucode_id` on `engine`: one read, journaled.
pub fn read_ucode_fuse_version<M: Mmio + ?Sized>(
    bar0: &mut M,
    journal: &mut Journal,
    engine: FuseEngine,
    ucode_id: u8,
) -> GpuResult<u8> {
    let off = ucode_fuse_offset(engine, ucode_id)?;
    if bar0.len() < FUSE_WINDOW {
        return Err(GpuError::InvalidArgument(
            "fereastra BAR0 prea mica pentru sigurante (trebuie 16 MiB)",
        ));
    }
    let raw = checked_read32(bar0, off)?;
    journal.push(Event::MmioRead {
        bar: 0,
        offset: off,
        value: raw,
    });
    decode_ucode_fuse(raw)
}

/// What I6 will patch into FWSEC on this board.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SignatureChoice {
    /// BAR0 offset of the fuse that was read.
    pub fuse_offset: u64,
    pub fuse_version: u8,
    /// Index into the descriptor's signatures (0-based).
    pub index: usize,
}

/// FWSEC (runs on GSP): reads its fuse and picks the signature, read-only.
/// `Blocked` if the VBIOS has no signature for the board's version.
pub fn fwsec_signature_choice<M: Mmio + ?Sized>(
    bar0: &mut M,
    journal: &mut Journal,
    desc: &FalconUcodeDescV3,
) -> GpuResult<SignatureChoice> {
    let fuse_offset = ucode_fuse_offset(FuseEngine::Gsp, desc.ucode_id)?;
    let fuse_version = read_ucode_fuse_version(bar0, journal, FuseEngine::Gsp, desc.ucode_id)?;
    let index = signature_index(desc.signature_versions, fuse_version)?;
    if index >= usize::from(desc.signature_count) {
        return Err(GpuError::Blocked(
            "FWSEC: indexul semnaturii depaseste numarul de semnaturi din VBIOS",
        ));
    }
    journal.push(Event::Step {
        name: "I6 pregatire: semnatura FWSEC aleasa (doar citire)",
        ok: true,
    });
    Ok(SignatureChoice {
        fuse_offset,
        fuse_version,
        index,
    })
}

/// Booter (runs on SEC2): the SEC2 fuse of the ucode id from the file's
/// `meta_data`, read-only; feed it to
/// [`crate::gsp::hs_load::booter_image`].
pub fn booter_fuse_version<M: Mmio + ?Sized>(
    bar0: &mut M,
    journal: &mut Journal,
    meta: &crate::gsp::hs_load::BooterMeta,
) -> GpuResult<u8> {
    let id = u8::try_from(meta.ucode_id)
        .map_err(|_| GpuError::InvalidArgument("booter: ucode_id prea mare"))?;
    read_ucode_fuse_version(bar0, journal, FuseEngine::Sec2, id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_sim::mmio::SimMmio;

    /// The descriptor fields `gpu fwsec` showed on the real board (29 oct):
    /// ucode_id 9, 3 signatures, mask 0x0007. Other fields: SIM.
    fn board_desc() -> FalconUcodeDescV3 {
        FalconUcodeDescV3 {
            hdr: 0,
            stored_size: 0xea00,
            pkc_data_offset: 0,
            interface_offset: 0,
            imem_phys_base: 0,
            imem_load_size: 0xe200,
            imem_virt_base: 0,
            dmem_phys_base: 0,
            dmem_load_size: 0x800,
            engine_id_mask: 0,
            ucode_id: 9,
            signature_count: 3,
            signature_versions: 0x0007,
        }
    }

    #[test]
    fn offsets_match_rm() {
        assert_eq!(ucode_fuse_offset(FuseEngine::Gsp, 1).unwrap(), 0x8241c0);
        assert_eq!(ucode_fuse_offset(FuseEngine::Sec2, 1).unwrap(), 0x824140);
        // FWSEC on this board: ucode_id 9 on GSP.
        assert_eq!(ucode_fuse_offset(FuseEngine::Gsp, 9).unwrap(), 0x8241e0);
        assert_eq!(ucode_fuse_offset(FuseEngine::Gsp, 16).unwrap(), 0x8241fc);
        assert!(ucode_fuse_offset(FuseEngine::Gsp, 0).is_err());
        assert!(ucode_fuse_offset(FuseEngine::Sec2, 17).is_err());
    }

    #[test]
    fn decode_is_highest_bit_plus_one() {
        assert_eq!(decode_ucode_fuse(0).unwrap(), 0);
        assert_eq!(decode_ucode_fuse(1).unwrap(), 1);
        assert_eq!(decode_ucode_fuse(0b11).unwrap(), 2);
        assert_eq!(decode_ucode_fuse(0b100).unwrap(), 3);
        assert_eq!(decode_ucode_fuse(0x8000).unwrap(), 16);
        assert!(matches!(
            decode_ucode_fuse(0x1_0000),
            Err(GpuError::Blocked(_))
        ));
        assert!(matches!(
            decode_ucode_fuse(0xbadf_1100),
            Err(GpuError::HardwareFault { .. })
        ));
        assert!(matches!(
            decode_ucode_fuse(0xffff_ffff),
            Err(GpuError::NotResponding)
        ));
    }

    #[test]
    fn signature_choice_on_simulated_board() {
        // SIMULARE: fuse values are ours, not read from the board.
        for (raw, version, index) in [(0u32, 0u8, 0usize), (1, 1, 1), (0b11, 2, 2)] {
            let mut m = SimMmio::ga104_bar0();
            m.set(0x8241e0, raw);
            let mut j = Journal::new(8);
            let c = fwsec_signature_choice(&mut m, &mut j, &board_desc()).unwrap();
            assert_eq!(
                c,
                SignatureChoice {
                    fuse_offset: 0x8241e0,
                    fuse_version: version,
                    index
                }
            );
            assert_eq!(m.reads, [0x8241e0], "exactly one read");
            assert!(m.writes.is_empty(), "nothing written");
        }
    }

    #[test]
    fn unsupported_or_bad_fuse_is_refused_without_writes() {
        // Version 3 (raw 0b100): the VBIOS has signatures only for 0..=2.
        let mut m = SimMmio::ga104_bar0();
        m.set(0x8241e0, 0b100);
        let mut j = Journal::new(8);
        assert!(matches!(
            fwsec_signature_choice(&mut m, &mut j, &board_desc()),
            Err(GpuError::Blocked(_))
        ));
        assert!(m.writes.is_empty());
        // A descriptor whose mask claims more signatures than it carries.
        let mut d = board_desc();
        d.signature_count = 2;
        let mut m = SimMmio::ga104_bar0();
        m.set(0x8241e0, 0b11);
        assert!(matches!(
            fwsec_signature_choice(&mut m, &mut j, &d),
            Err(GpuError::Blocked(_))
        ));
        // Priv error from the fuse block.
        let mut m = SimMmio::ga104_bar0();
        m.set(0x8241e0, 0xbadf_5040);
        assert!(matches!(
            fwsec_signature_choice(&mut m, &mut j, &board_desc()),
            Err(GpuError::HardwareFault { .. })
        ));
        assert!(m.writes.is_empty());
    }

    #[test]
    fn booter_reads_the_sec2_fuse() {
        // SIMULARE: ucode_id 2 (as in the synthetic Booter of hs_load tests).
        let meta = crate::gsp::hs_load::BooterMeta {
            fuse_ver: 0,
            engine_id: 4,
            ucode_id: 2,
        };
        let mut m = SimMmio::ga104_bar0();
        m.set(0x824144, 1);
        let mut j = Journal::new(8);
        let v = booter_fuse_version(&mut m, &mut j, &meta).unwrap();
        assert_eq!(v, 1);
        assert_eq!(m.reads, [0x824144]);
        // 2 signatures (as in booter_load-570.144.bin), version 1 -> index 0.
        assert_eq!(
            crate::gsp::hs_load::booter_signature_index(2, u32::from(v)).unwrap(),
            0
        );
        let bad = crate::gsp::hs_load::BooterMeta {
            ucode_id: 300,
            ..meta
        };
        assert!(booter_fuse_version(&mut m, &mut j, &bad).is_err());
        assert_eq!(m.reads.len(), 1, "refused before reading");
    }

    #[test]
    fn small_window_is_refused_before_any_read() {
        let mut m = SimMmio::new(crate::regs::INIT_WINDOW);
        let mut j = Journal::new(8);
        assert!(matches!(
            read_ucode_fuse_version(&mut m, &mut j, FuseEngine::Gsp, 9),
            Err(GpuError::InvalidArgument(_))
        ));
        assert!(m.reads.is_empty() && m.writes.is_empty());
    }
}
