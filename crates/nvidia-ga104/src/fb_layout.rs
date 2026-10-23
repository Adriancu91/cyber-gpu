//! I5 preparation — where GSP-RM's regions go in VRAM (pure arithmetic).
//!
//! Nothing here touches the board. The inputs that come from the board (VRAM
//! size, the VGA workspace register) and from the firmware files (bootloader
//! and GSP ELF sizes) are parameters; on hardware they are NECUNOSCUT until
//! I4/I5 are allowed.
//!
//! The layout is computed top-down, as described in the nova-core patches
//! "compute layout of the FRTS region" (v2 19/21, LKML May 2025) and "compute
//! layout of more framebuffer regions required for GSP" (v9 01/15, LKML Nov
//! 2025) — descriptions read, no code copied (docs/PROVENANCE.md):
//!
//! ```text
//! fb end  ┬ VGA workspace (default: last 1 MiB)
//!         ├ FRTS, 1 MiB, below the VGA start aligned down to 128 KiB ┐
//!         ├ boot (bootloader), aligned down to 4 KiB                 │
//!         ├ ELF (GSP-RM image), aligned down to 64 KiB               │ WPR2
//!         ├ WPR2 heap (LIBOS3 size), aligned to 1 MiB                │
//!         ├ GspFwWprMeta, WPR2 start aligned down to 1 MiB           ┘
//!         ├ heap (non-WPR), 1 MiB
//! 0       ┴ free for us
//! ```
//!
//! The values here are the *offsets* GspFwWprMeta will carry. Its byte
//! encoding is version-specific firmware ABI (r570) and waits for I8.

use gpu_core::error::{GpuError, GpuResult};

pub const KIB: u64 = 1 << 10;
pub const MIB: u64 = 1 << 20;
pub const GIB: u64 = 1 << 30;

pub const VGA_WORKSPACE_DEFAULT: u64 = MIB;
pub const VGA_WORKSPACE_SMALL: u64 = 128 * KIB;
pub const FRTS_SIZE: u64 = MIB;
pub const FRTS_ALIGN: u64 = 128 * KIB;
pub const BOOT_ALIGN: u64 = 4 * KIB;
pub const ELF_ALIGN: u64 = 64 * KIB;
pub const HEAP_ALIGN: u64 = MIB;
pub const NON_WPR_HEAP_SIZE: u64 = MIB;

/// LIBOS3 (GA102 and later) GSP heap parameters.
pub const LIBOS3_CARVEOUT: u64 = 22 * MIB;
pub const BASE_RM_SIZE: u64 = 8 * MIB;
pub const CLIENT_ALLOC_SIZE: u64 = 96 * MIB;
pub const OVERHEAD_PER_GIB: u64 = 96 * KIB;
pub const LIBOS3_HEAP_MIN: u64 = 88 * MIB;
pub const LIBOS3_HEAP_MAX: u64 = 280 * MIB;

/// Bit 3 of NV_PDISP_VGA_WORKSPACE_BASE (0x625f04): the address is valid;
/// bits 31:8 hold the address in 64 KiB units.
pub const VGA_WORKSPACE_VALID: u32 = 1 << 3;

fn align_down(v: u64, a: u64) -> u64 {
    v & !(a - 1)
}
fn align_up(v: u64, a: u64) -> u64 {
    (v + a - 1) & !(a - 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub start: u64,
    pub end: u64,
}

impl Region {
    pub fn len(&self) -> u64 {
        self.end - self.start
    }
    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
    fn contains(&self, o: &Region) -> bool {
        self.start <= o.start && o.end <= self.end
    }
}

/// VRAM size from NV_PGC6_AON_SECURE_SCRATCH_GROUP_42 (0x1183a4) on GA102+:
/// the value is in MiB.
pub fn fb_size_from_scratch42(v: u32) -> u64 {
    (v as u64) << 20
}

/// The GSP-RM heap inside WPR2 on LIBOS3 chips: carveout + base RM + one
/// client + 96 KiB per started GiB of VRAM (each part 1 MiB-aligned), then
/// kept inside [88 MiB, 280 MiB).
pub fn gsp_heap_size_libos3(fb_size: u64) -> u64 {
    let gib = fb_size.div_ceil(GIB);
    let sum = LIBOS3_CARVEOUT
        + BASE_RM_SIZE
        + align_up(CLIENT_ALLOC_SIZE, MIB)
        + align_up(gib * OVERHEAD_PER_GIB, MIB);
    sum.clamp(LIBOS3_HEAP_MIN, LIBOS3_HEAP_MAX - 1)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FbParams {
    pub fb_size: u64,
    /// Raw NV_PDISP_VGA_WORKSPACE_BASE, if it was read. Which boards read it
    /// is not settled in our sources (the summaries of the nova-core patch
    /// disagree on display enabled vs disabled) — DEDUS; `None` = default.
    pub vga_workspace_reg: Option<u32>,
    pub bootloader_size: u64,
    pub gsp_elf_size: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FbLayout {
    pub fb: Region,
    pub vga_workspace: Region,
    pub frts: Region,
    pub boot: Region,
    pub elf: Region,
    pub wpr2_heap: Region,
    pub wpr2: Region,
    pub heap: Region,
}

fn below(end: u64, size: u64, align: u64) -> GpuResult<Region> {
    let start = end.checked_sub(size).ok_or(GpuError::InvalidArgument(
        "FB: memoria placii prea mica pentru regiunile GSP",
    ))?;
    Ok(Region {
        start: align_down(start, align),
        end,
    })
}

fn vga_start(fb_end: u64, reg: Option<u32>) -> GpuResult<u64> {
    let default = fb_end - VGA_WORKSPACE_DEFAULT;
    match reg {
        Some(v) if v & VGA_WORKSPACE_VALID != 0 => {
            let addr = ((v >> 8) as u64) << 16;
            if addr >= fb_end {
                Err(GpuError::InvalidArgument(
                    "FB: zona VGA raportata dupa sfarsitul memoriei",
                ))
            } else if addr < default {
                Ok(fb_end - VGA_WORKSPACE_SMALL)
            } else {
                Ok(addr)
            }
        }
        _ => Ok(default),
    }
}

impl FbLayout {
    pub fn compute(p: &FbParams) -> GpuResult<Self> {
        if p.fb_size < 256 * MIB || !p.fb_size.is_multiple_of(MIB) {
            return Err(GpuError::InvalidArgument(
                "FB: marimea memoriei placii nu e plauzibila",
            ));
        }
        if p.bootloader_size == 0 || p.gsp_elf_size == 0 {
            return Err(GpuError::InvalidArgument(
                "FB: marimile firmware-ului lipsesc (vin de la I4)",
            ));
        }
        let fb = Region {
            start: 0,
            end: p.fb_size,
        };
        let vga_workspace = Region {
            start: vga_start(fb.end, p.vga_workspace_reg)?,
            end: fb.end,
        };
        let frts_start = align_down(vga_workspace.start, FRTS_ALIGN) - FRTS_SIZE;
        let frts = Region {
            start: frts_start,
            end: frts_start + FRTS_SIZE,
        };
        let boot = below(frts.start, p.bootloader_size, BOOT_ALIGN)?;
        let elf = below(boot.start, p.gsp_elf_size, ELF_ALIGN)?;
        let heap_size = gsp_heap_size_libos3(p.fb_size);
        let wpr2_heap = Region {
            start: align_down(
                elf.start
                    .checked_sub(heap_size)
                    .ok_or(GpuError::InvalidArgument(
                        "FB: memoria placii prea mica pentru heap-ul GSP",
                    ))?,
                HEAP_ALIGN,
            ),
            end: align_down(elf.start, HEAP_ALIGN),
        };
        // WPR2 starts below GspFwWprMeta. The heap start is 1 MiB-aligned, so
        // for any structure size in 1..=1 MiB the result is one MiB lower.
        let wpr2 = Region {
            start: align_down(wpr2_heap.start - 1, MIB),
            end: frts.end,
        };
        let heap = below(wpr2.start, NON_WPR_HEAP_SIZE, MIB)?;
        let l = FbLayout {
            fb,
            vga_workspace,
            frts,
            boot,
            elf,
            wpr2_heap,
            wpr2,
            heap,
        };
        l.check()?;
        Ok(l)
    }

    /// Invariants: every region inside VRAM, top-down order with no overlap,
    /// WPR2 holds FRTS, boot, ELF and the heap.
    fn check(&self) -> GpuResult<()> {
        let order = [
            self.heap,
            self.wpr2_heap,
            self.elf,
            self.boot,
            self.frts,
            self.vga_workspace,
        ];
        let ok = order.iter().all(|r| !r.is_empty() && self.fb.contains(r))
            && order.windows(2).all(|w| w[0].end <= w[1].start)
            && self.heap.end <= self.wpr2.start
            && [self.wpr2_heap, self.elf, self.boot, self.frts]
                .iter()
                .all(|r| self.wpr2.contains(r));
        if ok {
            Ok(())
        } else {
            Err(GpuError::InvalidArgument(
                "FB: regiunile GSP se suprapun sau ies din memorie",
            ))
        }
    }

    /// VRAM below the non-WPR heap: what is left for our buffers.
    pub fn free_below(&self) -> Region {
        Region {
            start: 0,
            end: self.heap.start,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Pure arithmetic. The firmware sizes are SIM (the real ones come from
    //! the files at I4); the 8 GiB VRAM is the RTX 3070 Ti's (FAPT, K0).
    use super::*;

    fn rtx3070ti() -> FbParams {
        FbParams {
            fb_size: 8 * GIB,
            vga_workspace_reg: None,
            bootloader_size: 0x5000, // SIM
            gsp_elf_size: 30 * MIB,  // SIM
        }
    }

    #[test]
    fn heap_size_for_8_gib() {
        // 22 + 8 + 96 + align_up(8 * 96 KiB) = 127 MiB.
        assert_eq!(gsp_heap_size_libos3(8 * GIB), 127 * MIB);
        assert_eq!(gsp_heap_size_libos3(8 * GIB + 1), 127 * MIB);
        assert_eq!(gsp_heap_size_libos3(48 * GIB), 131 * MIB);
        assert_eq!(gsp_heap_size_libos3(4096 * GIB), LIBOS3_HEAP_MAX - 1);
    }

    #[test]
    fn layout_of_the_workstation_board() {
        let l = FbLayout::compute(&rtx3070ti()).unwrap();
        let r = |s: u64, e: u64| Region { start: s, end: e };
        assert_eq!(l.vga_workspace, r(0x1_fff0_0000, 0x2_0000_0000));
        assert_eq!(l.frts, r(0x1_ffe0_0000, 0x1_fff0_0000));
        assert_eq!(l.boot, r(0x1_ffdf_b000, 0x1_ffe0_0000));
        assert_eq!(l.elf, r(0x1_fdff_0000, 0x1_ffdf_b000));
        assert_eq!(l.wpr2_heap, r(0x1_f600_0000, 0x1_fdf0_0000));
        assert_eq!(l.wpr2, r(0x1_f5f0_0000, 0x1_fff0_0000));
        assert_eq!(l.heap, r(0x1_f5e0_0000, 0x1_f5f0_0000));
        assert_eq!(l.wpr2_heap.len(), 127 * MIB);
        // 162 MiB at the top are GSP's; the rest of the 8 GiB is ours.
        assert_eq!(l.fb.end - l.free_below().end, 162 * MIB);
    }

    #[test]
    fn vga_register_rules() {
        let mut p = rtx3070ti();
        let reg = |addr: u64| (((addr >> 16) as u32) << 8) | VGA_WORKSPACE_VALID;
        p.vga_workspace_reg = Some(reg(0x1_fff8_0000));
        assert_eq!(
            FbLayout::compute(&p).unwrap().vga_workspace.start,
            0x1_fff8_0000
        );
        // Below the default: the small 128 KiB workspace at the very end.
        p.vga_workspace_reg = Some(reg(0x1000_0000));
        assert_eq!(
            FbLayout::compute(&p).unwrap().vga_workspace.start,
            0x2_0000_0000 - 128 * KIB
        );
        // Not valid: default.
        p.vga_workspace_reg = Some(reg(0x1_fff8_0000) & !VGA_WORKSPACE_VALID);
        assert_eq!(
            FbLayout::compute(&p).unwrap().vga_workspace.start,
            0x1_fff0_0000
        );
        // Past the end of VRAM: refused, not guessed.
        p.vga_workspace_reg = Some(reg(0x2_0000_0000));
        assert!(FbLayout::compute(&p).is_err());
    }

    #[test]
    fn implausible_inputs_are_refused() {
        let mut p = rtx3070ti();
        p.fb_size = 128 * MIB;
        assert!(FbLayout::compute(&p).is_err());
        let mut p = rtx3070ti();
        p.fb_size = 8 * GIB + 4096;
        assert!(FbLayout::compute(&p).is_err());
        let mut p = rtx3070ti();
        p.gsp_elf_size = 0;
        assert!(FbLayout::compute(&p).is_err());
        // An ELF larger than the VRAM cannot be placed.
        let mut p = rtx3070ti();
        p.fb_size = 256 * MIB;
        p.gsp_elf_size = 200 * MIB;
        assert!(FbLayout::compute(&p).is_err());
    }

    #[test]
    fn invariants_hold_for_many_sizes() {
        for gib in [1u64, 2, 4, 6, 8, 10, 12, 16, 24, 48] {
            for elf_mib in [1u64, 20, 30, 40, 64] {
                for boot in [0x1000u64, 0x5000, 0x1_2345] {
                    let p = FbParams {
                        fb_size: gib * GIB,
                        vga_workspace_reg: None,
                        bootloader_size: boot,
                        gsp_elf_size: elf_mib * MIB + 0x123,
                    };
                    let l = FbLayout::compute(&p).unwrap();
                    assert_eq!(l.frts.len(), FRTS_SIZE);
                    assert!(l.boot.len() >= boot && l.elf.len() >= p.gsp_elf_size);
                    assert_eq!(l.boot.start % BOOT_ALIGN, 0);
                    assert_eq!(l.elf.start % ELF_ALIGN, 0);
                    assert_eq!(l.wpr2.start % MIB, 0);
                    assert_eq!(l.heap.len(), NON_WPR_HEAP_SIZE);
                }
            }
        }
    }

    #[test]
    fn scratch42_is_mib() {
        assert_eq!(fb_size_from_scratch42(8192), 8 * GIB);
    }
}
