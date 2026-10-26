//! Cross-check (C-20 B): offsets and sizes written by hand in this crate in
//! session 5 must equal the ones generated from NVIDIA's 570.144 headers
//! (crate `nvidia-rm570`, layout computed by clang via bindgen).
//! Pure arithmetic: no device, no simulation, nothing written anywhere.

use core::mem::{offset_of, size_of};
use nvidia_ga104::gsp::{libos, msgq, wpr_meta};
use nvidia_rm570::regs::*;
use nvidia_rm570::structs::{GSP_ARGUMENTS_CACHED, GspFwWprMeta, msgqTxHeader};

#[test]
fn wpr_meta_matches_the_header() {
    use wpr_meta::off;
    type M = GspFwWprMeta;
    assert_eq!(wpr_meta::SIZE, size_of::<M>());
    assert_eq!(wpr_meta::MAGIC, nvidia_rm570::GSP_FW_WPR_META_MAGIC);
    assert_eq!(
        wpr_meta::REVISION,
        nvidia_rm570::structs::GSP_FW_WPR_META_REVISION as u64
    );
    assert_eq!(wpr_meta::VERIFIED, nvidia_rm570::GSP_FW_WPR_META_VERIFIED);
    let pairs = [
        (off::MAGIC, offset_of!(M, magic)),
        (off::REVISION, offset_of!(M, revision)),
        (
            off::SYSMEM_ADDR_OF_RADIX3_ELF,
            offset_of!(M, sysmemAddrOfRadix3Elf),
        ),
        (off::SIZE_OF_RADIX3_ELF, offset_of!(M, sizeOfRadix3Elf)),
        (
            off::SYSMEM_ADDR_OF_BOOTLOADER,
            offset_of!(M, sysmemAddrOfBootloader),
        ),
        (off::SIZE_OF_BOOTLOADER, offset_of!(M, sizeOfBootloader)),
        (
            off::BOOTLOADER_CODE_OFFSET,
            offset_of!(M, bootloaderCodeOffset),
        ),
        (
            off::BOOTLOADER_DATA_OFFSET,
            offset_of!(M, bootloaderDataOffset),
        ),
        (
            off::BOOTLOADER_MANIFEST_OFFSET,
            offset_of!(M, bootloaderManifestOffset),
        ),
        (
            off::SYSMEM_ADDR_OF_SIGNATURE,
            offset_of!(M, __bindgen_anon_1),
        ),
        (off::GSP_FW_RSVD_START, offset_of!(M, gspFwRsvdStart)),
        (off::NON_WPR_HEAP_OFFSET, offset_of!(M, nonWprHeapOffset)),
        (off::NON_WPR_HEAP_SIZE, offset_of!(M, nonWprHeapSize)),
        (off::GSP_FW_WPR_START, offset_of!(M, gspFwWprStart)),
        (off::GSP_FW_HEAP_OFFSET, offset_of!(M, gspFwHeapOffset)),
        (off::GSP_FW_HEAP_SIZE, offset_of!(M, gspFwHeapSize)),
        (off::GSP_FW_OFFSET, offset_of!(M, gspFwOffset)),
        (off::BOOT_BIN_OFFSET, offset_of!(M, bootBinOffset)),
        (off::FRTS_OFFSET, offset_of!(M, frtsOffset)),
        (off::FRTS_SIZE, offset_of!(M, frtsSize)),
        (off::GSP_FW_WPR_END, offset_of!(M, gspFwWprEnd)),
        (off::FB_SIZE, offset_of!(M, fbSize)),
        (off::VGA_WORKSPACE_OFFSET, offset_of!(M, vgaWorkspaceOffset)),
        (off::VGA_WORKSPACE_SIZE, offset_of!(M, vgaWorkspaceSize)),
        (off::BOOT_COUNT, offset_of!(M, bootCount)),
        (off::UNION_PARTITION, offset_of!(M, __bindgen_anon_2)),
        (
            off::HEAP_VF_PARTITION_COUNT,
            offset_of!(M, gspFwHeapVfPartitionCount),
        ),
        (off::FLAGS, offset_of!(M, flags)),
        (off::PMU_RESERVED_SIZE, offset_of!(M, pmuReservedSize)),
        (off::VERIFIED, offset_of!(M, verified)),
    ];
    for (i, (ours, theirs)) in pairs.iter().enumerate() {
        assert_eq!(ours, theirs, "field #{i}");
    }
    // The signature pair sits inside the first union: addr then size.
    assert_eq!(off::SIZE_OF_SIGNATURE, off::SYSMEM_ADDR_OF_SIGNATURE + 8);
}

#[test]
fn queue_header_matches_the_header() {
    use msgq::qoff;
    type T = msgqTxHeader;
    assert_eq!(qoff::VERSION, offset_of!(T, version));
    assert_eq!(qoff::SIZE, offset_of!(T, size));
    assert_eq!(qoff::MSG_SIZE, offset_of!(T, msgSize));
    assert_eq!(qoff::MSG_COUNT, offset_of!(T, msgCount));
    assert_eq!(qoff::WRITE_PTR, offset_of!(T, writePtr));
    assert_eq!(qoff::FLAGS, offset_of!(T, flags));
    assert_eq!(qoff::RX_HDR_OFF, offset_of!(T, rxHdrOff));
    assert_eq!(qoff::ENTRY_OFF, offset_of!(T, entryOff));
    assert_eq!(qoff::RX_HDR, size_of::<T>());
    assert_eq!(msgq::MSGQ_VERSION, nvidia_rm570::structs::MSGQ_VERSION);
    assert_eq!(
        msgq::MSGQ_FLAGS_SWAP_RX,
        nvidia_rm570::structs::MSGQ_FLAGS_SWAP_RX
    );
}

#[test]
fn gsp_arguments_size_matches_the_header() {
    assert_eq!(libos::GSP_ARGS_SIZE, size_of::<GSP_ARGUMENTS_CACHED>());
}

/// The two GFW_BOOT registers of I2 were taken from nouveau in session 2; the
/// NVIDIA header now confirms them (study, lesson 4).
#[test]
fn gfw_registers_match_the_header() {
    use nvidia_ga104::regs;
    let g = ga102_dev_gc6_island::NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK;
    assert_eq!(
        regs::NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK.offset,
        g as u64
    );
    let p = ga102_dev_gc6_island_addendum::NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT;
    assert_eq!(
        regs::NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT.offset,
        p as u64
    );
    let f = ga102_dev_gc6_island_addendum::NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT_PROGRESS;
    assert_eq!(f.mask(), nvidia_ga104::gfw::GFW_COMPLETED as u32);
}

/// `push.rs` (session 1, read by hand from clc56f.h) against the generated
/// values from the same header.
#[test]
fn pushbuffer_encoding_matches_the_header() {
    use nvidia_ga104::push::{self, SecOp};
    use nvidia_rm570::regs::clc56f as h;
    assert_eq!(push::AMPERE_CHANNEL_GPFIFO_A, h::AMPERE_CHANNEL_GPFIFO_A);
    assert_eq!(
        push::AMPERE_COMPUTE_B,
        nvidia_rm570::regs::clc7c0::AMPERE_COMPUTE_B
    );
    assert_eq!(SecOp::IncMethod as u32, h::NVC56F_DMA_SEC_OP_INC_METHOD);
    assert_eq!(
        SecOp::NonIncMethod as u32,
        h::NVC56F_DMA_SEC_OP_NON_INC_METHOD
    );
    assert_eq!(
        SecOp::ImmdDataMethod as u32,
        h::NVC56F_DMA_SEC_OP_IMMD_DATA_METHOD
    );
    assert_eq!(SecOp::OneInc as u32, h::NVC56F_DMA_SEC_OP_ONE_INC);

    let w = push::method_header(SecOp::IncMethod, 5, 0x6c, 3).unwrap();
    assert_eq!(h::NVC56F_DMA_SEC_OP.get(w), 1);
    assert_eq!(h::NVC56F_DMA_METHOD_COUNT.get(w), 3);
    assert_eq!(h::NVC56F_DMA_METHOD_SUBCHANNEL.get(w), 5);
    assert_eq!(h::NVC56F_DMA_METHOD_ADDRESS.get(w), 0x6c >> 2);
    let i = push::immediate(2, 0x100, 0x123).unwrap();
    assert_eq!(h::NVC56F_DMA_IMMD_DATA.get(i), 0x123);

    let e = push::gpfifo_entry(0x12_3456_7890, 40, true).unwrap();
    assert_eq!(h::NVC56F_GP_ENTRY0_GET.get(e.entry0) << 2, 0x3456_7890);
    assert_eq!(h::NVC56F_GP_ENTRY1_GET_HI.get(e.entry1), 0x12);
    assert_eq!(h::NVC56F_GP_ENTRY1_LENGTH.get(e.entry1), 40);
    assert_eq!(h::NVC56F_GP_ENTRY1_SYNC.get(e.entry1), 1);
    // Study lesson 14: tinygrad sets LEVEL=1 (SUBROUTINE); ours is 0 (MAIN).
    // Kept as is until I9; recorded in docs/COMPATIBILITATE_FIRMWARE.md.
    assert_eq!(h::NVC56F_GP_ENTRY1_LEVEL.get(e.entry1), 0);
}
