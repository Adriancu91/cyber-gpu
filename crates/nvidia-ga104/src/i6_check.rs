//! I6 pre-check — everything FWSEC-FRTS needs, read from the board without
//! writing anything (plan `docs/FEREASTRA_C-49.md`, proposed OS command
//! `gpu pregatire-i6`).
//!
//! Reads, each journaled once:
//! - the FWSEC ucode fuse ([`crate::fuse`]) → which of the VBIOS signatures;
//! - `NV_PGC6_AON_SECURE_SCRATCH_GROUP_42` (0x1183a4) → VRAM size in MiB
//!   ([`crate::fb_layout::fb_size_from_scratch42`]);
//! - `NV_PFB_PRI_MMU_WPR2_ADDR_HI` (0x1fa828, and `_LO` if HI ≠ 0) → whether
//!   some driver already ran FWSEC since power-on (then I6 must not run:
//!   a cold start is needed).
//!
//! Then computes the FRTS region (1 MiB under the VGA workspace) where I6
//! will ask FWSEC to put WPR2. The VGA workspace register is not read (which
//! boards need it is DEDUS, see `fb_layout`); the default is used.

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Mmio, checked_read32};

use crate::fb_layout::{FbLayout, FbParams, fb_size_from_scratch42};
use crate::fuse::{SignatureChoice, fwsec_signature_choice};
use crate::fwsec::FalconUcodeDescV3;
use crate::regs::is_priv_error;
use nvidia_rm570::regs::tu102_dev_fb as fb;

/// `NV_PGC6_AON_SECURE_SCRATCH_GROUP_42`: VRAM size in MiB on GA102+
/// (nova-core, read for the procedure; nouveau `ga102` fb code, MIT).
pub const NV_PGC6_AON_SECURE_SCRATCH_GROUP_42: u64 = 0x0011_83a4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I6Check {
    pub signature: SignatureChoice,
    pub scratch42: u32,
    pub fb_size: u64,
    /// WPR2 as the MMU reports it; `Some` = already up (I6 must not run).
    pub wpr2: Option<(u64, u64)>,
    /// Where I6 will ask FWSEC to put WPR2 (start, end).
    pub frts: (u64, u64),
}

impl I6Check {
    /// I6 may be attempted: WPR2 is down (cold start). The signature and
    /// the FRTS region exist, or this value would not have been built.
    pub fn passes(&self) -> bool {
        self.wpr2.is_none()
    }
}

fn read<M: Mmio + ?Sized>(bar0: &mut M, journal: &mut Journal, off: u64) -> GpuResult<u32> {
    let v = checked_read32(bar0, off)?;
    journal.push(Event::MmioRead {
        bar: 0,
        offset: off,
        value: v,
    });
    if v == 0xffff_ffff {
        return Err(GpuError::NotResponding);
    }
    if is_priv_error(v) {
        return Err(GpuError::HardwareFault { code: v });
    }
    Ok(v)
}

/// Read-only. `bootloader_size` / `gsp_elf_size` are the sizes of the
/// verified 570.144 files (I4); they only move regions below FRTS.
pub fn i6_check<M: Mmio + ?Sized>(
    bar0: &mut M,
    journal: &mut Journal,
    desc: &FalconUcodeDescV3,
    bootloader_size: u64,
    gsp_elf_size: u64,
) -> GpuResult<I6Check> {
    // The fuse read comes first and refuses a window under 16 MiB before
    // touching the board.
    let signature = fwsec_signature_choice(bar0, journal, desc)?;
    let scratch42 = read(bar0, journal, NV_PGC6_AON_SECURE_SCRATCH_GROUP_42)?;
    let fb_size = fb_size_from_scratch42(scratch42);
    let hi = read(bar0, journal, u64::from(fb::NV_PFB_PRI_MMU_WPR2_ADDR_HI))?;
    let wpr2 = if hi == 0 {
        None
    } else {
        let lo = read(bar0, journal, u64::from(fb::NV_PFB_PRI_MMU_WPR2_ADDR_LO))?;
        let w = crate::gsp::falcon::decode_wpr2(lo, hi);
        journal.push(Event::Note(
            "I6: WPR2 deja ridicat (FWSEC a rulat dupa pornire): trebuie pornire rece",
        ));
        Some((w.start, w.end))
    };
    let layout = FbLayout::compute(&FbParams {
        fb_size,
        vga_workspace_reg: None,
        bootloader_size,
        gsp_elf_size,
    })?;
    let out = I6Check {
        signature,
        scratch42,
        fb_size,
        wpr2,
        frts: (layout.frts.start, layout.frts.end),
    };
    journal.push(Event::Step {
        name: "I6 pregatire (doar citire)",
        ok: out.passes(),
    });
    Ok(out)
}

#[cfg(test)]
mod tests {
    //! SIMULARE: register values are ours (8 GiB as on the workstation, K0).
    use super::*;
    use gpu_sim::mmio::SimMmio;

    fn desc() -> FalconUcodeDescV3 {
        // ucode_id / count / mask as `gpu fwsec` showed on the board (29 oct).
        FalconUcodeDescV3 {
            ucode_id: 9,
            signature_count: 3,
            signature_versions: 0x0007,
            ..Default::default()
        }
    }

    fn board() -> SimMmio {
        let mut m = SimMmio::ga104_bar0();
        m.set(0x8241e0, 0b1); // fuse version 1 (SIM)
        m.set(0x1183a4, 8192); // 8 GiB
        m
    }

    #[test]
    fn cold_board_passes_with_frts_under_vga() {
        let mut m = board();
        let mut j = Journal::new(16);
        let c = i6_check(&mut m, &mut j, &desc(), 0x6000, 63 << 20).unwrap();
        assert!(c.passes());
        assert_eq!((c.signature.fuse_version, c.signature.index), (1, 1));
        assert_eq!(c.fb_size, 8 << 30);
        assert_eq!(c.frts, (0x1_ffe0_0000, 0x1_fff0_0000));
        assert_eq!(
            m.reads,
            [0x8241e0, 0x1183a4, 0x1fa828],
            "3 reads, LO not needed"
        );
        assert!(m.writes.is_empty(), "nothing written");
    }

    #[test]
    fn wpr2_already_up_does_not_pass() {
        let mut m = board();
        // WPR2 0x1_f5f0_0000..0x1_fff0_0000 (VAL = addr >> 12, in bits 31:4).
        m.set(0x1fa824, ((0x1_f5f0_0000u64 >> 12) as u32) << 4);
        m.set(0x1fa828, ((0x1_fff0_0000u64 >> 12) as u32) << 4);
        let mut j = Journal::new(16);
        let c = i6_check(&mut m, &mut j, &desc(), 0x6000, 63 << 20).unwrap();
        assert!(!c.passes());
        assert_eq!(c.wpr2, Some((0x1_f5f0_0000, 0x1_fff0_0000)));
        assert!(m.writes.is_empty());
    }

    #[test]
    fn bad_reads_stop_the_check() {
        // Priv error on the VRAM size register.
        let mut m = board();
        m.set(0x1183a4, 0xbadf_1201);
        let mut j = Journal::new(16);
        assert!(matches!(
            i6_check(&mut m, &mut j, &desc(), 0x6000, 63 << 20),
            Err(GpuError::HardwareFault { .. })
        ));
        // Implausible VRAM size (16 MiB): the layout refuses it.
        let mut m = board();
        m.set(0x1183a4, 16);
        assert!(i6_check(&mut m, &mut j, &desc(), 0x6000, 63 << 20).is_err());
        // Board gone.
        let mut m = board();
        m.set(0x1fa828, 0xffff_ffff);
        assert!(matches!(
            i6_check(&mut m, &mut j, &desc(), 0x6000, 63 << 20),
            Err(GpuError::NotResponding)
        ));
        // 4 MiB window: refused before any read.
        let mut m = SimMmio::new(crate::regs::INIT_WINDOW);
        assert!(i6_check(&mut m, &mut j, &desc(), 0x6000, 63 << 20).is_err());
        assert!(m.reads.is_empty());
    }
}
