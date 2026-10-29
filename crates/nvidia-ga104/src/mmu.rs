//! I9a — the GPU's virtual memory, built by the driver: a VRAM allocator, a
//! GPU-VA allocator and the page tables (MMU format version 2, Ampere).
//! Built and tested **only in SIMULARE**; nothing here has run on the board.
//!
//! On the GSP path the driver makes the page tables itself and tells GSP-RM
//! where they are (study `studiu/2026-10-26_open-source/gpu.md`, lesson 12):
//! the golden image hands RM the levels of a reserved range
//! (`VASPACE_COPY_SERVER_RESERVED_PDES`), later channels use the root.
//!
//! Sources (read as the specification, code written by us):
//! - entry formats: generated `nvidia_rm570::regs::tu102_dev_mmu_ver2`
//!   (NVIDIA `dev_mmu.h`, 570.144, MIT);
//! - level layout: NVIDIA's comment in `kern_gmmu_fmt_gp10x.c` (570.144, MIT):
//!   PD3 [48:47] → PD2 [46:38] → PD1 [37:29] → PD0 [28:21] (dual 16-byte
//!   entries) → PT [20:12];
//! - choices tinygrad makes (`nvdev.py`, MIT, commit 9cf7829a6), DEDUS:
//!   `NO_ATS = 1` in directory entries, PTE `KIND = 6` (NVIDIA's
//!   `NV_MMU_PTE_KIND_GENERIC_MEMORY`), the GPU-VA window from 64 GiB.
//!
//! Only 4 KiB pages for now (the dual PDE's big half stays invalid). Every
//! table is cleared before it is linked (VRAM is not cleared at power-on).
//! All writes go through [`VramWrite`]; on the board that will be a BAR
//! window behind `hw-write`, followed by a TLB invalidate (I9b, not written).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};
use gpu_memory::RangeAllocator;
use nvidia_rm570::EntryField;
use nvidia_rm570::regs::tu102_dev_mmu_ver2 as f;

use crate::gsp::golden::{GoldenMemory, Mapping};
use crate::gsp::rmapi::PdeLevel;

pub const PAGE: u64 = 0x1000;
/// Virtual addresses have 49 bits (PD3 covers bits 48:47).
pub const VA_BITS: u32 = 49;
/// Where the driver hands out GPU virtual addresses (tinygrad: from 64 GiB,
/// 2^44 bytes). DEDUS.
pub const VA_BASE: u64 = 0x10_0000_0000;
pub const VA_SIZE: u64 = 1 << 44;
/// Physical VRAM addresses must fit `ADDRESS_VID` (25 bits of 4 KiB = 128 GiB).
pub const PA_LIMIT: u64 = 1 << (25 + 12);
/// PTE kind for plain memory (`NV_MMU_PTE_KIND_GENERIC_MEMORY` = 6; tinygrad uses 6).
pub const KIND_GENERIC: u128 = 6;

/// One level of the walk: its index bits in the VA and its entry size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Level {
    pub hi: u32,
    pub lo: u32,
    pub entry_size: u64,
}

impl Level {
    pub const fn entries(self) -> u64 {
        1 << (self.hi - self.lo + 1)
    }
    pub const fn index(self, va: u64) -> u64 {
        (va >> self.lo) & (self.entries() - 1)
    }
    /// Bytes of VA one entry covers.
    pub const fn covers(self) -> u64 {
        1 << self.lo
    }
    pub const fn table_bytes(self) -> u64 {
        self.entries() * self.entry_size
    }
}

/// PD3, PD2, PD1, PD0, PT (4 KiB pages).
pub const LEVELS: [Level; 5] = [
    Level {
        hi: 48,
        lo: 47,
        entry_size: f::NV_MMU_VER2_PDE__SIZE as u64,
    },
    Level {
        hi: 46,
        lo: 38,
        entry_size: f::NV_MMU_VER2_PDE__SIZE as u64,
    },
    Level {
        hi: 37,
        lo: 29,
        entry_size: f::NV_MMU_VER2_PDE__SIZE as u64,
    },
    Level {
        hi: 28,
        lo: 21,
        entry_size: f::NV_MMU_VER2_DUAL_PDE__SIZE as u64,
    },
    Level {
        hi: 20,
        lo: 12,
        entry_size: f::NV_MMU_VER2_PTE__SIZE as u64,
    },
];

/// Where the driver writes VRAM. SIMULARE: a model; board: the BAR0 window
/// of [`crate::vram_window`] (I9b, behind `hw-write`).
pub trait VramWrite {
    fn write_u64(&mut self, pa: u64, value: u64) -> GpuResult<()>;
}

fn put(e: u128, field: EntryField, v: u128) -> GpuResult<u128> {
    field.set(e, v).ok_or(GpuError::InvalidArgument(
        "MMU: valoare prea mare pentru campul intrarii",
    ))
}

fn vid(pa: u64) -> GpuResult<u128> {
    if !pa.is_multiple_of(PAGE) || pa >= PA_LIMIT {
        return Err(GpuError::InvalidArgument(
            "MMU: adresa VRAM nealiniata sau prea mare",
        ));
    }
    Ok((pa >> f::NV_MMU_VER2_PDE_ADDRESS_SHIFT) as u128)
}

/// A PD3/PD2/PD1 entry pointing to the next table in VRAM.
pub fn encode_pde(next: u64) -> GpuResult<u64> {
    let mut e = put(
        0,
        f::NV_MMU_VER2_PDE_IS_PTE,
        f::NV_MMU_VER2_PDE_IS_PTE_FALSE as u128,
    )?;
    e = put(
        e,
        f::NV_MMU_VER2_PDE_APERTURE,
        f::NV_MMU_VER2_PDE_APERTURE_VIDEO_MEMORY as u128,
    )?;
    e = put(
        e,
        f::NV_MMU_VER2_PDE_NO_ATS,
        f::NV_MMU_VER2_PDE_NO_ATS_TRUE as u128,
    )?;
    e = put(e, f::NV_MMU_VER2_PDE_ADDRESS_VID, vid(next)?)?;
    Ok(e as u64)
}

/// A PD0 (dual) entry whose small half points to a 4 KiB page table; the big
/// (64 KiB) half stays invalid.
pub fn encode_dual_pde(small_pt: u64) -> GpuResult<u128> {
    let mut e = put(
        0,
        f::NV_MMU_VER2_DUAL_PDE_IS_PTE,
        f::NV_MMU_VER2_DUAL_PDE_IS_PTE_FALSE as u128,
    )?;
    e = put(
        e,
        f::NV_MMU_VER2_DUAL_PDE_APERTURE_BIG,
        f::NV_MMU_VER2_DUAL_PDE_APERTURE_BIG_INVALID as u128,
    )?;
    e = put(
        e,
        f::NV_MMU_VER2_DUAL_PDE_NO_ATS,
        f::NV_MMU_VER2_DUAL_PDE_NO_ATS_TRUE as u128,
    )?;
    e = put(
        e,
        f::NV_MMU_VER2_DUAL_PDE_APERTURE_SMALL,
        f::NV_MMU_VER2_DUAL_PDE_APERTURE_SMALL_VIDEO_MEMORY as u128,
    )?;
    // The small half counts in 4 KiB units like the other entries.
    const _: () = assert!(
        f::NV_MMU_VER2_DUAL_PDE_ADDRESS_SHIFT == f::NV_MMU_VER2_PDE_ADDRESS_SHIFT
            && f::NV_MMU_VER2_PTE_ADDRESS_SHIFT == f::NV_MMU_VER2_PDE_ADDRESS_SHIFT
    );
    put(e, f::NV_MMU_VER2_DUAL_PDE_ADDRESS_SMALL_VID, vid(small_pt)?)
}

/// Attributes of a mapped page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PageAttr {
    /// `VOL`: uncached by the GPU (tinygrad sets it for "uncached" buffers).
    pub volatile: bool,
    pub read_only: bool,
}

/// A 4 KiB PTE for a page of VRAM.
pub fn encode_pte(pa: u64, attr: PageAttr) -> GpuResult<u64> {
    let mut e = put(
        0,
        f::NV_MMU_VER2_PTE_VALID,
        f::NV_MMU_VER2_PTE_VALID_TRUE as u128,
    )?;
    e = put(
        e,
        f::NV_MMU_VER2_PTE_APERTURE,
        f::NV_MMU_VER2_PTE_APERTURE_VIDEO_MEMORY as u128,
    )?;
    e = put(e, f::NV_MMU_VER2_PTE_VOL, attr.volatile as u128)?;
    e = put(e, f::NV_MMU_VER2_PTE_READ_ONLY, attr.read_only as u128)?;
    e = put(e, f::NV_MMU_VER2_PTE_ADDRESS_VID, vid(pa)?)?;
    e = put(e, f::NV_MMU_VER2_PTE_KIND, KIND_GENERIC)?;
    Ok(e as u64)
}

/// The driver's memory: VRAM ranges, GPU VAs, page tables.
pub struct GpuMemory<W: VramWrite> {
    vram: RangeAllocator,
    va: RangeAllocator,
    w: W,
    root: u64,
    /// (level, VA >> (level's hi + 1)) → table address, for levels 1..=4.
    tables: BTreeMap<(usize, u64), u64>,
    /// Mapped 4 KiB pages: VA → PA (no page is ever mapped twice).
    pages: BTreeMap<u64, u64>,
    /// Number of 8-byte writes made through `w`.
    pub writes: u64,
}

impl<W: VramWrite> GpuMemory<W> {
    /// `vram_start..vram_end` is the VRAM the driver may use: below GSP-RM's
    /// regions (`FbLayout::free_below`) and **above whatever the screen uses**
    /// (UEFI GOP framebuffer; its place is learned in window B0). The root
    /// table is allocated and cleared here.
    pub fn new(vram_start: u64, vram_end: u64, w: W) -> GpuResult<Self> {
        if vram_end <= vram_start || vram_end > PA_LIMIT || !vram_start.is_multiple_of(PAGE) {
            return Err(GpuError::InvalidArgument("MMU: zona VRAM gresita"));
        }
        let mut m = GpuMemory {
            vram: RangeAllocator::new(vram_start, vram_end - vram_start)?,
            va: RangeAllocator::new(VA_BASE, VA_SIZE)?,
            w,
            root: 0,
            tables: BTreeMap::new(),
            pages: BTreeMap::new(),
            writes: 0,
        };
        m.root = m.new_table()?;
        Ok(m)
    }

    /// The PD3 table (what `SET_PAGE_DIRECTORY` will name for later channels).
    pub fn root(&self) -> u64 {
        self.root
    }

    /// Every page table made so far, the root included.
    pub fn table_addresses(&self) -> impl Iterator<Item = u64> + '_ {
        core::iter::once(self.root).chain(self.tables.values().copied())
    }

    pub fn writer(&self) -> &W {
        &self.w
    }

    /// The sink, e.g. to invalidate the TLB once the tables are written (I9b).
    pub fn writer_mut(&mut self) -> &mut W {
        &mut self.w
    }

    fn write(&mut self, pa: u64, v: u64) -> GpuResult<()> {
        self.w.write_u64(pa, v)?;
        self.writes += 1;
        Ok(())
    }

    /// A cleared 4 KiB table (every level fits in one page).
    fn new_table(&mut self) -> GpuResult<u64> {
        let pa = self.vram.alloc(PAGE, PAGE)?;
        for off in (0..PAGE).step_by(8) {
            self.write(pa + off, 0)?;
        }
        Ok(pa)
    }

    /// The table of `level` (1..=4) that covers `va`, created and linked
    /// into its parent if missing.
    fn table(&mut self, level: usize, va: u64) -> GpuResult<u64> {
        if level == 0 {
            return Ok(self.root);
        }
        let key = (level, va >> (LEVELS[level].hi + 1));
        if let Some(&pa) = self.tables.get(&key) {
            return Ok(pa);
        }
        let parent = self.table(level - 1, va)?;
        let pl = LEVELS[level - 1];
        let slot = parent + pl.index(va) * pl.entry_size;
        let pa = self.new_table()?;
        if level - 1 == 3 {
            let e = encode_dual_pde(pa)?;
            self.write(slot, e as u64)?;
            self.write(slot + 8, (e >> 64) as u64)?;
        } else {
            self.write(slot, encode_pde(pa)?)?;
        }
        self.tables.insert(key, pa);
        Ok(pa)
    }

    /// Maps `size` bytes of VRAM at `pa` to `va` with 4 KiB pages. Refuses
    /// (before writing anything) a page that is already mapped.
    pub fn map(&mut self, va: u64, pa: u64, size: u64, attr: PageAttr) -> GpuResult<()> {
        if size == 0 || !va.is_multiple_of(PAGE) || !pa.is_multiple_of(PAGE) {
            return Err(GpuError::InvalidArgument("MMU: mapare nealiniata"));
        }
        let n = size.div_ceil(PAGE);
        let (Some(va_end), Some(pa_end)) = (va.checked_add(n * PAGE), pa.checked_add(n * PAGE))
        else {
            return Err(GpuError::InvalidArgument("MMU: mapare prea mare"));
        };
        if va_end > 1 << VA_BITS || pa_end > PA_LIMIT {
            return Err(GpuError::InvalidArgument(
                "MMU: mapare in afara spatiului de adrese",
            ));
        }
        if self.pages.range(va..va_end).next().is_some() {
            return Err(GpuError::InvalidArgument("MMU: pagina deja mapata"));
        }
        for i in 0..n {
            let (v, p) = (va + i * PAGE, pa + i * PAGE);
            let pt = self.table(4, v)?;
            self.write(pt + LEVELS[4].index(v) * 8, encode_pte(p, attr)?)?;
            self.pages.insert(v, p);
        }
        Ok(())
    }

    /// Where a mapped VA lands, from the driver's own bookkeeping.
    pub fn lookup(&self, va: u64) -> Option<u64> {
        self.pages.get(&(va & !(PAGE - 1))).map(|p| p + (va % PAGE))
    }
}

/// I10a: 32-bit stores go to the writer (the page tables are not involved:
/// these are physical addresses of buffers the driver allocated).
impl<W: VramWrite + crate::gsp::work_queue::ChannelMemory> crate::gsp::work_queue::ChannelMemory
    for GpuMemory<W>
{
    fn write_u32(&mut self, pa: u64, v: u32) -> GpuResult<()> {
        self.writer_mut().write_u32(pa, v)
    }
    fn read_u32(&mut self, pa: u64) -> GpuResult<u32> {
        self.writer_mut().read_u32(pa)
    }
}

impl<W: VramWrite> GoldenMemory for GpuMemory<W> {
    fn alloc_mapped(&mut self, size: u64, align: u64) -> GpuResult<Mapping> {
        let pa = self.alloc_phys(size, align)?;
        let len = size.next_multiple_of(PAGE);
        let va = self.va.alloc(len, align.max(PAGE))?;
        self.map(va, pa, len, PageAttr::default())?;
        Ok(Mapping { va, pa, size })
    }

    fn alloc_phys(&mut self, size: u64, align: u64) -> GpuResult<u64> {
        if size == 0 {
            return Err(GpuError::InvalidArgument("MMU: marime zero"));
        }
        self.vram
            .alloc(size.next_multiple_of(PAGE), align.max(PAGE))
    }

    /// Reserves a VA range for GSP-RM that one PD1 entry covers (≤ 512 MiB,
    /// aligned to its size) and returns PD3, PD2 and PD1 — the levels above
    /// it, created now; RM fills PD0 and below itself. Sizes as tinygrad
    /// sends them: the whole root (here 4 entries × 8 B, NVIDIA's [48:47];
    /// tinygrad's 48-bit VA sends 2 × 8) and one page for the others.
    fn reserve_va(&mut self, size: u64) -> GpuResult<(u64, Vec<PdeLevel>)> {
        if size == 0 || !size.is_power_of_two() || size > LEVELS[2].covers() {
            return Err(GpuError::InvalidArgument(
                "MMU: zona rezervata trebuie sa incapa intr-o intrare PD1",
            ));
        }
        let va = self.va.alloc(size, size)?;
        let mut levels = Vec::with_capacity(3);
        for (l, lv) in LEVELS.iter().enumerate().take(3) {
            levels.push(PdeLevel {
                phys: self.table(l, va)?,
                size: if l == 0 { lv.table_bytes() } else { PAGE },
                page_shift: lv.lo as u8,
                aperture: f::NV_MMU_VER2_PDE_APERTURE_VIDEO_MEMORY,
            });
        }
        Ok((va, levels))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_sim::vram::{SimVram, Translation, WalkError, translate};

    struct Sim(SimVram);
    impl VramWrite for Sim {
        fn write_u64(&mut self, pa: u64, v: u64) -> GpuResult<()> {
            self.0.write_u64(pa, v).map_err(GpuError::InvalidArgument)
        }
    }

    const VRAM: u64 = 8 << 30;
    fn mem() -> GpuMemory<Sim> {
        // SIMULARE: 8 GiB, the first 64 MiB left alone (stand-in for the screen).
        GpuMemory::new(64 << 20, VRAM - (256 << 20), Sim(SimVram::new(VRAM))).unwrap()
    }

    #[test]
    fn levels_match_nvidia_layout() {
        assert_eq!(LEVELS.map(|l| l.entries()), [4, 512, 512, 256, 512]);
        assert_eq!(
            LEVELS.map(|l| l.table_bytes()),
            [32, 4096, 4096, 4096, 4096]
        );
        assert_eq!(LEVELS[2].covers(), 512 << 20);
        // Contiguous index bits from 48 down to 12.
        for w in LEVELS.windows(2) {
            assert_eq!(w[0].lo, w[1].hi + 1);
        }
    }

    /// Encoder against the hand reading of `dev_mmu.h` (gpu_sim::vram).
    #[test]
    fn entries_have_the_documented_bits() {
        assert_eq!(
            encode_pde(0x1234_5000).unwrap(),
            0b10 | 1 << 5 | 0x12345 << 8
        );
        let d = encode_dual_pde(0x6000).unwrap();
        assert_eq!(d as u64, 1 << 5); // big half invalid, NO_ATS
        assert_eq!((d >> 64) as u64, 0b10 | 6 << 8);
        let p = encode_pte(
            0x8_0000_0000,
            PageAttr {
                volatile: true,
                read_only: false,
            },
        )
        .unwrap();
        assert_eq!(p, 1 | 1 << 3 | (0x80_0000 << 8) | 6 << 56);
        // Addresses that do not fit or are not page aligned are refused.
        assert!(encode_pte(PA_LIMIT, PageAttr::default()).is_err());
        assert!(encode_pde(0x1234).is_err());
        assert!(encode_dual_pde(PA_LIMIT).is_err());
    }

    #[test]
    fn mappings_walk_to_the_right_place() {
        let mut m = mem();
        let a = m.alloc_mapped(0x3_0000, 0x1000).unwrap();
        let b = m.alloc_mapped(5000, 2 << 20).unwrap();
        assert!(b.va.is_multiple_of(2 << 20) && b.pa.is_multiple_of(2 << 20));
        assert!(a.va >= VA_BASE && b.va >= a.va + 0x3_0000);
        let root = m.root();
        for (map, len) in [(a, 0x3_0000u64), (b, 0x2000)] {
            for off in (0..len).step_by(0x800) {
                let t = translate(&m.w.0, root, map.va + off).unwrap();
                assert_eq!(t.pa, map.pa + off);
                assert_eq!(m.lookup(map.va + off), Some(map.pa + off));
                assert!(!t.volatile && !t.read_only && !t.privileged);
                assert_eq!(t.kind, 6);
            }
        }
        // Past the end of b (rounded to a page) nothing is mapped.
        assert_eq!(
            translate(&m.w.0, root, b.va + 0x2000),
            Err(WalkError::NotPresent(4))
        );
        // A VA in another PD1 slot: the PD1 table exists and is cleared.
        assert_eq!(
            translate(&m.w.0, root, a.va + (1 << 29)),
            Err(WalkError::NotPresent(2))
        );
        assert_eq!(
            translate(&m.w.0, root, 1 << 47),
            Err(WalkError::NotPresent(0))
        );
    }

    #[test]
    fn mapping_across_a_page_table_boundary() {
        let mut m = mem();
        let va = VA_BASE + (2 << 20) - 0x2000; // last 2 pages of one PT, first 2 of the next
        let pa = m.alloc_phys(0x4000, 0x1000).unwrap();
        m.map(va, pa, 0x4000, PageAttr::default()).unwrap();
        for i in 0..4 {
            let t = translate(&m.w.0, m.root(), va + i * 0x1000).unwrap();
            assert_eq!(t.pa, pa + i * 0x1000);
        }
    }

    #[test]
    fn bad_requests_are_refused_without_writes() {
        let mut m = mem();
        let pa = m.alloc_phys(0x2000, 0x1000).unwrap();
        m.map(VA_BASE, pa, 0x2000, PageAttr::default()).unwrap();
        let before = m.writes;
        // Overlapping an existing page: refused, nothing written.
        assert!(
            m.map(VA_BASE + 0x1000, pa, 0x1000, PageAttr::default())
                .is_err()
        );
        assert!(
            m.map(VA_BASE + 0x10, pa, 0x1000, PageAttr::default())
                .is_err()
        );
        assert!(
            m.map(1 << VA_BITS, pa, 0x1000, PageAttr::default())
                .is_err()
        );
        assert!(
            m.map(VA_BASE + 0x10_0000, PA_LIMIT, 0x1000, PageAttr::default())
                .is_err()
        );
        assert!(m.alloc_phys(0, 0x1000).is_err());
        assert!(m.alloc_phys(1 << 40, 0x1000).is_err()); // more than the VRAM
        assert!(m.reserve_va(3 << 20).is_err());
        assert!(m.reserve_va(1 << 30).is_err());
        assert_eq!(m.writes, before);
        assert!(GpuMemory::new(0x1000, 0x1000, Sim(SimVram::new(VRAM))).is_err());
    }

    /// Every table the driver made was cleared: no slot of any table reads
    /// as poison (the model's stand-in for VRAM garbage).
    #[test]
    fn every_table_is_cleared_before_use() {
        let mut m = mem();
        m.alloc_mapped(0x1000, 0x1000).unwrap();
        m.reserve_va(512 << 20).unwrap();
        let mut tables: Vec<u64> = m.tables.values().copied().collect();
        tables.push(m.root());
        // root, PD2, PD1 (shared by the page and the reserved range), PD0, PT.
        assert_eq!(tables.len(), 5);
        for t in tables {
            for off in (0..PAGE).step_by(8) {
                assert_ne!(m.w.0.read_u64(t + off).unwrap(), gpu_sim::vram::POISON);
            }
        }
    }

    #[test]
    fn reserved_range_levels_are_the_walk_of_that_range() {
        let mut m = mem();
        m.alloc_mapped(0x1000, 0x1000).unwrap();
        let (va, lv) = m.reserve_va(512 << 20).unwrap();
        assert!(va.is_multiple_of(512 << 20));
        assert_eq!(lv.len(), 3);
        assert_eq!(lv[0].phys, m.root());
        assert_eq!(
            lv.iter().map(|l| l.page_shift).collect::<Vec<_>>(),
            [47, 38, 29]
        );
        assert_eq!(
            lv.iter().map(|l| l.size).collect::<Vec<_>>(),
            [32, 4096, 4096]
        );
        assert!(lv.iter().all(|l| l.aperture == 1));
        // PD3 → PD2 and PD2 → PD1 as the walker reads them (hand positions).
        let next = |e: u64| ((e >> 8) & ((1 << 25) - 1)) << 12;
        let e3 =
            m.w.0
                .read_u64(lv[0].phys + LEVELS[0].index(va) * 8)
                .unwrap();
        let e2 =
            m.w.0
                .read_u64(lv[1].phys + LEVELS[1].index(va) * 8)
                .unwrap();
        assert_eq!((next(e3), next(e2)), (lv[1].phys, lv[2].phys));
        // RM fills the range's PD0 itself: our PD1 slot for it stays empty.
        let e1 =
            m.w.0
                .read_u64(lv[2].phys + LEVELS[2].index(va) * 8)
                .unwrap();
        assert_eq!(e1, 0);
        assert_eq!(
            translate(&m.w.0, m.root(), va),
            Err(WalkError::NotPresent(2))
        );
        // The VA allocator never hands out the reserved range again.
        for _ in 0..4 {
            let x = m.alloc_mapped(256 << 20, 0x1000).unwrap();
            assert!(x.va + x.size <= va || x.va >= va + (512 << 20));
        }
    }

    /// A planted mistake in one PTE (address off by one page) is caught by
    /// the independent walker.
    #[test]
    fn a_wrong_pte_is_caught() {
        let mut m = mem();
        let a = m.alloc_mapped(0x2000, 0x1000).unwrap();
        let pt = m.tables[&(4, a.va >> 21)];
        let slot = pt + LEVELS[4].index(a.va + 0x1000) * 8;
        m.w.0
            .write_u64(
                slot,
                encode_pte(a.pa + 0x2000, PageAttr::default()).unwrap(),
            )
            .unwrap();
        let t: Translation = translate(&m.w.0, m.root(), a.va + 0x1000).unwrap();
        assert_ne!(t.pa, m.lookup(a.va + 0x1000).unwrap());
    }
}
