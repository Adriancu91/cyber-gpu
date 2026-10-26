//! `GspFwWprMeta` (r570) — the 256-byte description Booter reads at I7 to
//! find GSP-RM in system memory and to lay out WPR2 in VRAM.
//!
//! Layout: open-gpu-kernel-modules@570.144
//! `src/nvidia/arch/nvalloc/common/inc/gsp/gsp_fw_wpr_meta.h` (MIT), field by
//! field; the offsets below are computed from that header (natural alignment,
//! 256 bytes total, as its comment says). Which FB region goes in which field
//! is DEDUS from nouveau `gsp/r535.c` (MIT, read, not copied). The VRAM
//! offsets come from [`crate::fb_layout`].

use gpu_core::error::{GpuError, GpuResult};

use super::put_u64;
use crate::fb_layout::{FRTS_ALIGN, FbLayout};

pub const SIZE: usize = 256;
pub const MAGIC: u64 = 0xdc3a_ae21_371a_60b3;
pub const REVISION: u64 = 1;
/// Value Booter writes into `verified` once it accepted the image.
pub const VERIFIED: u64 = 0xa0a0_a0a0_a0a0_a0a0;

/// Byte offsets of the fields we fill (from the 570.144 header).
pub mod off {
    pub const MAGIC: usize = 0;
    pub const REVISION: usize = 8;
    pub const SYSMEM_ADDR_OF_RADIX3_ELF: usize = 16;
    pub const SIZE_OF_RADIX3_ELF: usize = 24;
    pub const SYSMEM_ADDR_OF_BOOTLOADER: usize = 32;
    pub const SIZE_OF_BOOTLOADER: usize = 40;
    pub const BOOTLOADER_CODE_OFFSET: usize = 48;
    pub const BOOTLOADER_DATA_OFFSET: usize = 56;
    pub const BOOTLOADER_MANIFEST_OFFSET: usize = 64;
    pub const SYSMEM_ADDR_OF_SIGNATURE: usize = 72;
    pub const SIZE_OF_SIGNATURE: usize = 80;
    pub const GSP_FW_RSVD_START: usize = 88;
    pub const NON_WPR_HEAP_OFFSET: usize = 96;
    pub const NON_WPR_HEAP_SIZE: usize = 104;
    pub const GSP_FW_WPR_START: usize = 112;
    pub const GSP_FW_HEAP_OFFSET: usize = 120;
    pub const GSP_FW_HEAP_SIZE: usize = 128;
    pub const GSP_FW_OFFSET: usize = 136;
    pub const BOOT_BIN_OFFSET: usize = 144;
    pub const FRTS_OFFSET: usize = 152;
    pub const FRTS_SIZE: usize = 160;
    pub const GSP_FW_WPR_END: usize = 168;
    pub const FB_SIZE: usize = 176;
    pub const VGA_WORKSPACE_OFFSET: usize = 184;
    pub const VGA_WORKSPACE_SIZE: usize = 192;
    pub const BOOT_COUNT: usize = 200;
    /// 32-byte union (partition RPC / crash-report queue), left zero.
    pub const UNION_PARTITION: usize = 208;
    pub const HEAP_VF_PARTITION_COUNT: usize = 240;
    pub const FLAGS: usize = 241;
    pub const PMU_RESERVED_SIZE: usize = 244;
    pub const VERIFIED: usize = 248;
}

/// Where things are in *system* memory (DMA addresses the GPU will use) and
/// the bootloader offsets. On hardware these come from I5 (buffers) and from
/// the bootloader file's header ([`crate::fw_hdr::RiscvBootloader`]: code =
/// `monitorCodeOffset`, data = `monitorDataOffset`, manifest =
/// `manifestOffset`; parsed only on synthetic files so far, SIMULARE).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SysmemInputs {
    /// DMA address of the radix3 level-0 page describing `.fwimage`.
    pub radix3_elf_addr: u64,
    /// Size of the `.fwimage` section in bytes.
    pub radix3_elf_size: u64,
    pub bootloader_addr: u64,
    pub bootloader_size: u64,
    pub bootloader_code_offset: u64,
    pub bootloader_data_offset: u64,
    pub bootloader_manifest_offset: u64,
    /// DMA address and size of the `.fwsignature_ga10x` copy.
    pub signature_addr: u64,
    pub signature_size: u64,
}

fn dma_ok(addr: u64, size: u64) -> bool {
    // Page-aligned, non-null, and inside the OS DMA window (16 MiB – 4 GiB,
    // VT-d off on the workstation: the GPU sees physical addresses).
    addr != 0
        && addr.is_multiple_of(4096)
        && addr >= 16 << 20
        && addr.checked_add(size).is_some_and(|e| e <= 1 << 32)
}

/// Builds the 256 bytes. Refuses inputs that cannot be right (zero sizes,
/// unaligned or out-of-window DMA addresses, a layout whose WPR end does not
/// match the VGA rule) instead of producing a structure GSP would reject.
pub fn encode(fb: &FbLayout, s: &SysmemInputs) -> GpuResult<[u8; SIZE]> {
    if s.radix3_elf_size == 0 || s.bootloader_size == 0 || s.signature_size == 0 {
        return Err(GpuError::InvalidArgument(
            "WPR meta: marimi zero (ELF, bootloader sau semnatura)",
        ));
    }
    if !dma_ok(s.radix3_elf_addr, 4096)
        || !dma_ok(s.bootloader_addr, s.bootloader_size)
        || !dma_ok(s.signature_addr, s.signature_size)
    {
        return Err(GpuError::InvalidArgument(
            "WPR meta: adresa DMA nealiniata sau in afara ferestrei 16 MiB - 4 GiB",
        ));
    }
    let wpr_end = fb.frts.end;
    if wpr_end != fb.vga_workspace.start & !(FRTS_ALIGN - 1) {
        return Err(GpuError::InvalidArgument(
            "WPR meta: sfarsitul WPR2 nu e zona VGA aliniata la 128 KiB",
        ));
    }
    let mut b = [0u8; SIZE];
    let fields: [(usize, u64); 26] = [
        (off::MAGIC, MAGIC),
        (off::REVISION, REVISION),
        (off::SYSMEM_ADDR_OF_RADIX3_ELF, s.radix3_elf_addr),
        (off::SIZE_OF_RADIX3_ELF, s.radix3_elf_size),
        (off::SYSMEM_ADDR_OF_BOOTLOADER, s.bootloader_addr),
        (off::SIZE_OF_BOOTLOADER, s.bootloader_size),
        (off::BOOTLOADER_CODE_OFFSET, s.bootloader_code_offset),
        (off::BOOTLOADER_DATA_OFFSET, s.bootloader_data_offset),
        (
            off::BOOTLOADER_MANIFEST_OFFSET,
            s.bootloader_manifest_offset,
        ),
        (off::SYSMEM_ADDR_OF_SIGNATURE, s.signature_addr),
        (off::SIZE_OF_SIGNATURE, s.signature_size),
        (off::GSP_FW_RSVD_START, fb.heap.start),
        (off::NON_WPR_HEAP_OFFSET, fb.heap.start),
        (off::NON_WPR_HEAP_SIZE, fb.heap.len()),
        (off::GSP_FW_WPR_START, fb.wpr2.start),
        (off::GSP_FW_HEAP_OFFSET, fb.wpr2_heap.start),
        (off::GSP_FW_HEAP_SIZE, fb.wpr2_heap.len()),
        (off::GSP_FW_OFFSET, fb.elf.start),
        (off::BOOT_BIN_OFFSET, fb.boot.start),
        (off::FRTS_OFFSET, fb.frts.start),
        (off::FRTS_SIZE, fb.frts.len()),
        (off::GSP_FW_WPR_END, wpr_end),
        (off::FB_SIZE, fb.fb.len()),
        (off::VGA_WORKSPACE_OFFSET, fb.vga_workspace.start),
        (off::VGA_WORKSPACE_SIZE, fb.vga_workspace.len()),
        (off::BOOT_COUNT, 0),
    ];
    for (o, v) in fields {
        put_u64(&mut b, o, v);
    }
    // Union, VF partition count, flags, padding, pmuReservedSize, verified: 0.
    Ok(b)
}

/// Whether Booter marked the structure as verified (read back from VRAM at I7
/// on hardware; in SIMULARE, from our own buffer).
pub fn is_verified(b: &[u8; SIZE]) -> bool {
    super::get_u64(b, off::VERIFIED) == VERIFIED
}

#[cfg(test)]
mod tests {
    use super::super::get_u64;
    use super::*;
    use crate::fb_layout::{FbParams, GIB};

    fn layout() -> FbLayout {
        FbLayout::compute(&FbParams {
            fb_size: 8 * GIB,
            vga_workspace_reg: None,
            bootloader_size: 24_684,
            gsp_elf_size: 63_571_696,
        })
        .unwrap()
    }

    fn inputs() -> SysmemInputs {
        SysmemInputs {
            radix3_elf_addr: 0x0200_0000,
            radix3_elf_size: 63_000_000,
            bootloader_addr: 0x0210_0000,
            bootloader_size: 24_684,
            bootloader_code_offset: 0x100,
            bootloader_data_offset: 0x4000,
            bootloader_manifest_offset: 0x5000,
            signature_addr: 0x0220_0000,
            signature_size: 4096,
        }
    }

    #[test]
    fn offsets_follow_the_570_144_header() {
        // 26 u64 fields from 0, then a 32-byte union, two u8, 2 bytes of
        // padding, one u32 and the final u64: 256 bytes, as the header says.
        assert_eq!(off::BOOT_COUNT, 25 * 8);
        assert_eq!(off::UNION_PARTITION + 32, off::HEAP_VF_PARTITION_COUNT);
        assert_eq!(off::FLAGS + 1 + 2, off::PMU_RESERVED_SIZE);
        assert_eq!(off::PMU_RESERVED_SIZE + 4, off::VERIFIED);
        assert_eq!(off::VERIFIED + 8, SIZE);
    }

    #[test]
    fn encodes_the_workstation_layout() {
        let fb = layout();
        let b = encode(&fb, &inputs()).unwrap();
        assert_eq!(get_u64(&b, off::MAGIC), MAGIC);
        assert_eq!(get_u64(&b, off::REVISION), 1);
        assert_eq!(get_u64(&b, off::GSP_FW_WPR_START), fb.wpr2.start);
        assert_eq!(get_u64(&b, off::GSP_FW_WPR_END), 0x1_fff0_0000);
        assert_eq!(get_u64(&b, off::FRTS_SIZE), 1 << 20);
        assert_eq!(get_u64(&b, off::GSP_FW_HEAP_SIZE), 127 << 20);
        assert_eq!(get_u64(&b, off::FB_SIZE), 8 * GIB);
        assert_eq!(get_u64(&b, off::SIZE_OF_SIGNATURE), 4096);
        assert!(!is_verified(&b));
        assert!(b[off::UNION_PARTITION..].iter().all(|&x| x == 0));
    }

    #[test]
    fn refuses_impossible_inputs() {
        let fb = layout();
        let mut s = inputs();
        s.signature_size = 0;
        assert!(encode(&fb, &s).is_err());
        let mut s = inputs();
        s.bootloader_addr = 0x0210_0010; // unaligned
        assert!(encode(&fb, &s).is_err());
        let mut s = inputs();
        s.radix3_elf_addr = 0x1000; // below the 16 MiB DMA window
        assert!(encode(&fb, &s).is_err());
        let mut s = inputs();
        s.signature_addr = 0xffff_f000;
        s.signature_size = 0x2000; // crosses 4 GiB
        assert!(encode(&fb, &s).is_err());
    }
}
