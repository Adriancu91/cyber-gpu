//! LibOS boot arguments and `GSP_ARGUMENTS_CACHED` (r570).
//!
//! GSP-RM runs on LibOS; at I7 the driver hands GSP (through a mailbox
//! register pair) the DMA address of a page of `LibosMemoryRegionInitArgument`
//! entries. The entries name the log buffers and the `RMARGS` buffer, which
//! holds `GSP_ARGUMENTS_CACHED` — and that one tells GSP-RM where the message
//! queues are ([`super::msgq`]).
//!
//! Layouts: open-gpu-kernel-modules@570.144 `libos_init_args.h` and
//! `gsp_init_args.h` (MIT). The 8-character ids and the four standard entries
//! (LOGINIT, LOGINTR, LOGRM, RMARGS) are DEDUS from nouveau `gsp/r535.c`
//! (MIT, read, not copied). Not confirmed on the board.

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use super::msgq::QueueLayout;
use super::{put_u32, put_u64};

/// Size of one `LibosMemoryRegionInitArgument` (3×u64 + 2×u8, padded to 8).
pub const REGION_ARG_SIZE: usize = 32;
/// Size of `GSP_ARGUMENTS_CACHED` on x86-64 (see the offsets test).
pub const GSP_ARGS_SIZE: usize = 72;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RegionKind {
    None = 0,
    Contiguous = 1,
    Radix3 = 2,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum RegionLoc {
    None = 0,
    Sysmem = 1,
    Fb = 2,
}

/// LibOS packs up to 8 ASCII characters big-endian into a u64 ("LOGINIT").
pub fn id8(name: &str) -> u64 {
    name.bytes()
        .take(8)
        .fold(0u64, |id, c| (id << 8) | c as u64)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegionArg {
    pub name: &'static str,
    pub pa: u64,
    pub size: u64,
    pub kind: RegionKind,
    pub loc: RegionLoc,
}

/// Encodes the argument page (one 4 KiB page, zero-filled after the entries).
pub fn encode_region_args(args: &[RegionArg]) -> GpuResult<Vec<u8>> {
    if args.is_empty() || args.len() * REGION_ARG_SIZE > 4096 {
        return Err(GpuError::InvalidArgument("LibOS: numar de regiuni invalid"));
    }
    let mut page = alloc::vec![0u8; 4096];
    for (i, a) in args.iter().enumerate() {
        if a.pa == 0 || a.size == 0 || !a.pa.is_multiple_of(4096) {
            return Err(GpuError::InvalidArgument(
                "LibOS: regiune cu adresa nula/nealiniata sau marime zero",
            ));
        }
        let o = i * REGION_ARG_SIZE;
        put_u64(&mut page, o, id8(a.name));
        put_u64(&mut page, o + 8, a.pa);
        put_u64(&mut page, o + 16, a.size);
        page[o + 24] = a.kind as u8;
        page[o + 25] = a.loc as u8;
    }
    Ok(page)
}

/// The four regions nouveau hands to GSP-RM (DEDUS): three log buffers and
/// the RM arguments, all contiguous in system memory.
pub fn standard_regions(loginit: u64, logintr: u64, logrm: u64, rmargs: u64) -> [RegionArg; 4] {
    let c = |name, pa, size| RegionArg {
        name,
        pa,
        size,
        kind: RegionKind::Contiguous,
        loc: RegionLoc::Sysmem,
    };
    [
        c("LOGINIT", loginit, 0x10000),
        c("LOGINTR", logintr, 0x10000),
        c("LOGRM", logrm, 0x10000),
        c("RMARGS", rmargs, 0x1000),
    ]
}

/// Byte offset of `bDmemStack` in `GSP_ARGUMENTS_CACHED` (cross-checked with
/// bindgen in `tests/rm570_crosscheck.rs`).
pub const B_DMEM_STACK_OFF: usize = 48;

/// `GSP_ARGUMENTS_CACHED`: the message-queue arguments, suspend/resume
/// arguments (zero at first boot), GPU instance 0, **DMEM stack**, no profiler.
///
/// `bDmemStack = 1` (session 87, after the `os-280` journal): every driver
/// known to boot GSP-RM 570 on this chip family sets it — NVIDIA's RM
/// (`kgspPopulateGspRmInitArgs`: `RmGspStackPlacement` default 0 → DMEM
/// stack), nouveau r570 (`r570_gsp_set_rmargs`: `args->bDmemStack = 1`),
/// tinygrad (`ip.py`: `GSP_ARGUMENTS_CACHED(bDmemStack=True, …)`). Until
/// session 86 we sent 0 ("LibOS task stack", the non-default placement that
/// RM only takes when a registry key asks for it). DEDUS, not proven: that
/// this is why GSP-RM 570.144 answered `GSP_INIT_DONE` with
/// `NV_ERR_OPERATING_SYSTEM` (0x59) on the board; checked only at the next
/// `gpu lant`. Nothing else in the structure changes.
pub fn encode_gsp_args(q: &QueueLayout, shared_mem_pa: u64) -> GpuResult<[u8; GSP_ARGS_SIZE]> {
    encode_gsp_args_with(q, shared_mem_pa, true)
}

/// [`encode_gsp_args`] with `bDmemStack` chosen (C-108 variants; `false` =
/// what we sent up to `os-280`, 0 = "LibOS task stack").
pub fn encode_gsp_args_with(
    q: &QueueLayout,
    shared_mem_pa: u64,
    dmem_stack: bool,
) -> GpuResult<[u8; GSP_ARGS_SIZE]> {
    if shared_mem_pa == 0 || !shared_mem_pa.is_multiple_of(4096) {
        return Err(GpuError::InvalidArgument(
            "GSP args: adresa memoriei partajate nula sau nealiniata",
        ));
    }
    let mut b = [0u8; GSP_ARGS_SIZE];
    // MESSAGE_QUEUE_INIT_ARGUMENTS
    put_u64(&mut b, 0, shared_mem_pa);
    put_u32(&mut b, 8, q.page_table_entries as u32);
    put_u64(&mut b, 16, q.cmd_queue_offset as u64);
    put_u64(&mut b, 24, q.stat_queue_offset as u64);
    // GSP_SR_INIT_ARGUMENTS at 32 (oldLevel, flags, bInPMTransition): 0
    // gpuInstance at 44: 0; profilerArgs at 56/64: 0.
    b[B_DMEM_STACK_OFF] = dmem_stack as u8;
    Ok(b)
}

#[cfg(test)]
mod tests {
    use super::super::{get_u32, get_u64};
    use super::*;

    #[test]
    fn id8_packs_big_endian_ascii() {
        assert_eq!(id8("LOGINIT"), 0x004c_4f47_494e_4954);
        assert_eq!(id8("RMARGS"), 0x0000_524d_4152_4753);
        assert_eq!(id8("ABCDEFGHIJ"), id8("ABCDEFGH"));
    }

    #[test]
    fn region_page_and_refusals() {
        let r = standard_regions(0x0100_0000, 0x0101_0000, 0x0102_0000, 0x0103_0000);
        let p = encode_region_args(&r).unwrap();
        assert_eq!(p.len(), 4096);
        assert_eq!(get_u64(&p, 0), id8("LOGINIT"));
        assert_eq!(get_u64(&p, 3 * 32 + 8), 0x0103_0000);
        assert_eq!(get_u64(&p, 3 * 32 + 16), 0x1000);
        assert_eq!((p[3 * 32 + 24], p[3 * 32 + 25]), (1, 1));
        assert!(p[4 * 32..].iter().all(|&b| b == 0));
        let mut bad = r;
        bad[1].pa = 0x0101_0010;
        assert!(encode_region_args(&bad).is_err());
        assert!(encode_region_args(&[]).is_err());
    }

    #[test]
    fn gsp_args_offsets() {
        // MESSAGE_QUEUE_INIT_ARGUMENTS = u64, u32(+4 pad), NvLength, NvLength = 32;
        // GSP_SR_INIT_ARGUMENTS = u32, u32, NvBool (+3 pad) = 12 -> 44;
        // gpuInstance u32 -> 48; bDmemStack NvBool (+7 pad) -> 56;
        // profilerArgs { u64 pa; u64 size } -> 72.
        let q = QueueLayout::standard();
        let b = encode_gsp_args(&q, 0x0300_0000).unwrap();
        assert_eq!(get_u64(&b, 0), 0x0300_0000);
        assert_eq!(get_u32(&b, 8), 129);
        assert_eq!(get_u64(&b, 16), 0x1000);
        assert_eq!(get_u64(&b, 24), 0x41000);
        // Session 87: bDmemStack = 1 (as RM, nouveau r570, tinygrad); every
        // other byte after the queue arguments stays zero.
        assert_eq!(B_DMEM_STACK_OFF, 48);
        assert_eq!(b[B_DMEM_STACK_OFF], 1);
        for (i, &x) in b.iter().enumerate().skip(32) {
            if i != B_DMEM_STACK_OFF {
                assert_eq!(x, 0, "byte {i}");
            }
        }
        assert!(encode_gsp_args(&q, 0).is_err());
    }
}
