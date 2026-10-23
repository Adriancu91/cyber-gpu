//! I3, second half — locate the FWSEC ucode inside the VBIOS (read-only).
//!
//! FWSEC is NVIDIA-signed firmware stored in the VBIOS. I6 will need to know
//! where it is and how large its parts are; this module finds that out and
//! nothing more. The ucode bytes themselves are never read: they are an
//! opaque NVIDIA blob (docs/PROVENANCE.md).
//!
//! The chain, as described in public sources:
//! - BIT token 0x70 (Falcon data) points, inside the PCI-AT image, at a u32:
//!   the offset of the PMU lookup table. That offset is counted from the
//!   start of the PCI-AT image *as if the EFI image did not exist*, i.e. it
//!   lands in the FWSEC images taken back to back — CONFIRMAT: Linux
//!   nova-core VBIOS documentation (docs.kernel.org/gpu/nova/core/vbios.html:
//!   "does not account for the EFI image located between the PciAt and
//!   FwSec images"). The exact arithmetic is DEDUS (to confirm on the board).
//! - PMU lookup table: header `version u8, header_len u8, entry_len u8,
//!   entry_count u8`, then entries `application_id u8, target_id u8,
//!   data u32` — DEDUS from nouveau `nvkm/subdev/bios/pmu.c` (MIT).
//!   Application 0x85 = FWSEC_PROD (nova-core documentation); its `data` is
//!   the offset of the ucode descriptor, counted the same way (DEDUS).
//! - FalconUCodeDescV3, 44 bytes: hdr (bits 31:16 = header size, ucode
//!   follows it; bits 15:8 = version), stored_size, pkc_data_offset,
//!   interface_offset, imem_phys_base, imem_load_size, imem_virt_base,
//!   dmem_phys_base, dmem_load_size (u32 each), engine_id_mask u16,
//!   ucode_id u8, signature_count u8, signature_versions u16, reserved u16 —
//!   CONFIRMAT: field list and meaning from the nova-core patch "firmware:
//!   add ucode descriptor used by FWSEC-FRTS" (LKML, May 2025); the version
//!   bits are DEDUS. IMEM is loaded from ucode offset 0, DMEM from offset
//!   imem_load_size — CONFIRMAT: nova-core patch v11 08/12 (LKML, Mar 2026).
//! - Signatures (RSA-3K, 384 bytes each) sit right after the 44-byte
//!   descriptor, inside the header — DEDUS (nova-core documentation places
//!   the signatures between descriptor and ucode; the size is our reading).

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, Mmio};

use crate::regs::INIT_WINDOW;
use crate::vbios::{BIT_TOKEN_FALCON_DATA, CODE_FWSEC, CODE_PCI_AT, Rom, VbiosLayout};

pub const APP_ID_FWSEC_PROD: u8 = 0x85;
pub const DESC_V3_LEN: u32 = 44;
pub const RSA3K_SIG_LEN: u32 = 384;
const MAX_PMU_ENTRIES: u8 = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FalconUcodeDescV3 {
    pub hdr: u32,
    pub stored_size: u32,
    pub pkc_data_offset: u32,
    pub interface_offset: u32,
    pub imem_phys_base: u32,
    pub imem_load_size: u32,
    pub imem_virt_base: u32,
    pub dmem_phys_base: u32,
    pub dmem_load_size: u32,
    pub engine_id_mask: u16,
    pub ucode_id: u8,
    pub signature_count: u8,
    pub signature_versions: u16,
}

impl FalconUcodeDescV3 {
    /// Header size: descriptor + signatures; the ucode starts after it.
    pub fn header_size(&self) -> u32 {
        self.hdr >> 16
    }
    pub fn version(&self) -> u8 {
        (self.hdr >> 8) as u8
    }
    /// One signature per bit of `signature_versions` (DEDUS, see
    /// [`signature_index`]).
    pub fn signatures_consistent(&self) -> bool {
        self.signature_versions.count_ones() == self.signature_count as u32
    }
}

/// A run of bytes in the PROM window (offset, length).
pub type PromRange = (u32, u32);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FwsecLocation {
    /// FWSEC images in ROM order: (PROM offset, size). Their concatenation is
    /// the address space the VBIOS pointers use (after subtracting PCI-AT).
    pub sections: Vec<PromRange>,
    /// PROM offset of the PMU lookup table and the application ids it lists.
    pub pmu_table: u32,
    pub pmu_apps: Vec<u8>,
    /// Descriptor position, relative to the FWSEC section and in PROM.
    pub desc_rel: u32,
    pub desc_prom: u32,
    pub desc: FalconUcodeDescV3,
}

impl FwsecLocation {
    fn total(&self) -> u32 {
        self.sections.iter().map(|s| s.1).sum()
    }
    /// Map `len` bytes at section offset `rel` to PROM ranges (several if
    /// they cross from one FWSEC image into the next).
    pub fn prom_ranges(&self, rel: u32, len: u32) -> GpuResult<Vec<PromRange>> {
        map_ranges(&self.sections, rel, len)
    }
    /// Where the signatures are (PROM).
    pub fn signature_ranges(&self) -> GpuResult<Vec<PromRange>> {
        self.prom_ranges(
            self.desc_rel + DESC_V3_LEN,
            self.desc.signature_count as u32 * RSA3K_SIG_LEN,
        )
    }
    /// Where the ucode is (PROM); I6 copies these bytes, untouched, to RAM.
    pub fn ucode_ranges(&self) -> GpuResult<Vec<PromRange>> {
        self.prom_ranges(
            self.desc_rel + self.desc.header_size(),
            self.desc.stored_size,
        )
    }
    pub fn ucode_crosses_images(&self) -> bool {
        self.ucode_ranges().is_ok_and(|r| r.len() > 1)
    }
    /// Total bytes of the PROM the ucode occupies; must fit the section.
    pub fn fits(&self) -> bool {
        (self.desc_rel as u64) + self.desc.header_size() as u64 + self.desc.stored_size as u64
            <= self.total() as u64
    }
}

fn map_ranges(sections: &[PromRange], rel: u32, len: u32) -> GpuResult<Vec<PromRange>> {
    let total: u64 = sections.iter().map(|s| s.1 as u64).sum();
    if rel as u64 + len as u64 > total {
        return Err(GpuError::OutOfBounds {
            offset: rel as u64,
            len: len as u64,
            limit: total,
        });
    }
    let mut out = Vec::new();
    let (mut rel, mut left) = (rel, len);
    for &(off, size) in sections {
        if left == 0 {
            break;
        }
        if rel >= size {
            rel -= size;
            continue;
        }
        let take = (size - rel).min(left);
        out.push((off + rel, take));
        left -= take;
        rel = 0;
    }
    Ok(out)
}

/// Index of the signature to use for a board whose fuse holds
/// `fuse_version`: bit `fuse_version` of `signature_versions` must be set,
/// and the index is the number of set bits below it. DEDUS from nouveau
/// (`nvkm/falcon/fw.c`, MIT) and nova-core (procedure only); the fuse
/// register itself is NECUNOSCUT on GA10x, so nothing calls this on a board.
pub fn signature_index(signature_versions: u16, fuse_version: u8) -> GpuResult<usize> {
    if fuse_version >= 16 {
        return Err(GpuError::InvalidArgument(
            "FWSEC: versiunea din siguranta > 15",
        ));
    }
    let bit = 1u16 << fuse_version;
    if signature_versions & bit == 0 {
        return Err(GpuError::Blocked(
            "FWSEC: nicio semnatura pentru versiunea din siguranta placii",
        ));
    }
    Ok((signature_versions & (bit - 1)).count_ones() as usize)
}

/// Locate FWSEC_PROD. `layout` must come from [`crate::vbios::read_layout`]
/// on the same window. Reads only the PROM window, never the ucode.
pub fn locate<M: Mmio + ?Sized, K: Clock + ?Sized>(
    bar0: &mut M,
    journal: &mut Journal,
    clock: &mut K,
    budget_us: u64,
    layout: &VbiosLayout,
) -> GpuResult<FwsecLocation> {
    if bar0.len() < INIT_WINDOW {
        return Err(GpuError::InvalidArgument(
            "fereastra BAR0 prea mica pentru I3 (trebuie 4 MiB)",
        ));
    }
    let mut rom = Rom::new(bar0, journal, clock, budget_us);
    let r = walk(&mut rom, layout);
    journal.push(Event::Step {
        name: "I3: descriptorul FWSEC",
        ok: r.as_ref().is_ok_and(|l| l.fits()),
    });
    r
}

struct Section<'r, 'a, M: Mmio + ?Sized, K: Clock + ?Sized> {
    rom: &'r mut Rom<'a, M, K>,
    sections: &'r [PromRange],
}

impl<M: Mmio + ?Sized, K: Clock + ?Sized> Section<'_, '_, M, K> {
    fn u8(&mut self, rel: u32) -> GpuResult<u8> {
        let r = map_ranges(self.sections, rel, 1)?;
        self.rom.u8(r[0].0)
    }
    fn u16(&mut self, rel: u32) -> GpuResult<u16> {
        Ok(self.u8(rel)? as u16 | (self.u8(rel + 1)? as u16) << 8)
    }
    fn u32(&mut self, rel: u32) -> GpuResult<u32> {
        Ok(self.u16(rel)? as u32 | (self.u16(rel + 2)? as u32) << 16)
    }
}

fn walk<M: Mmio + ?Sized, K: Clock + ?Sized>(
    rom: &mut Rom<'_, M, K>,
    layout: &VbiosLayout,
) -> GpuResult<FwsecLocation> {
    let pci_at = match layout.images.first() {
        Some(i) if i.code_type == CODE_PCI_AT => *i,
        _ => return Err(GpuError::Blocked("FWSEC: lipseste imaginea PCI-AT")),
    };
    let sections: Vec<PromRange> = layout
        .images
        .iter()
        .filter(|i| i.code_type == CODE_FWSEC)
        .map(|i| (i.offset, i.size))
        .collect();
    if sections.is_empty() {
        return Err(GpuError::Blocked("FWSEC: nicio imagine FWSEC (0xe0)"));
    }
    let token = layout
        .bit
        .as_ref()
        .and_then(|(_, t)| t.iter().find(|t| t.id == BIT_TOKEN_FALCON_DATA))
        .copied()
        .ok_or(GpuError::Blocked("FWSEC: tabela BIT fara jetonul 0x70"))?;
    if token.length < 4 || token.data_offset as u32 + 4 > pci_at.size {
        return Err(GpuError::InvalidArgument(
            "FWSEC: jetonul 0x70 nu incape in imaginea PCI-AT",
        ));
    }
    let to_section = |ptr: u32| -> GpuResult<u32> {
        ptr.checked_sub(pci_at.size)
            .ok_or(GpuError::InvalidArgument(
                "FWSEC: pointer in imaginea PCI-AT, nu in FWSEC",
            ))
    };
    let table_ptr = rom.u32(pci_at.offset + token.data_offset as u32)?;
    let table_rel = to_section(table_ptr)?;
    let mut s = Section {
        rom,
        sections: &sections,
    };
    let header_len = s.u8(table_rel + 1)?;
    let entry_len = s.u8(table_rel + 2)?;
    let count = s.u8(table_rel + 3)?;
    if header_len < 4 || entry_len < 6 || count == 0 || count > MAX_PMU_ENTRIES {
        return Err(GpuError::InvalidArgument(
            "FWSEC: antetul tabelei PMU nu are forma asteptata",
        ));
    }
    let mut apps = Vec::with_capacity(count as usize);
    let mut desc_ptr = None;
    for k in 0..count as u32 {
        let e = table_rel + header_len as u32 + k * entry_len as u32;
        let app = s.u8(e)?;
        apps.push(app);
        if app == APP_ID_FWSEC_PROD && desc_ptr.is_none() {
            desc_ptr = Some(s.u32(e + 2)?);
        }
    }
    let desc_ptr = desc_ptr.ok_or(GpuError::Blocked(
        "FWSEC: tabela PMU nu are aplicatia 0x85 (FWSEC_PROD)",
    ))?;
    let desc_rel = to_section(desc_ptr)?;
    let w = |s: &mut Section<'_, '_, M, K>, k: u32| s.u32(desc_rel + 4 * k);
    let desc = FalconUcodeDescV3 {
        hdr: w(&mut s, 0)?,
        stored_size: w(&mut s, 1)?,
        pkc_data_offset: w(&mut s, 2)?,
        interface_offset: w(&mut s, 3)?,
        imem_phys_base: w(&mut s, 4)?,
        imem_load_size: w(&mut s, 5)?,
        imem_virt_base: w(&mut s, 6)?,
        dmem_phys_base: w(&mut s, 7)?,
        dmem_load_size: w(&mut s, 8)?,
        engine_id_mask: s.u16(desc_rel + 36)?,
        ucode_id: s.u8(desc_rel + 38)?,
        signature_count: s.u8(desc_rel + 39)?,
        signature_versions: s.u16(desc_rel + 40)?,
    };
    if desc.version() != 3 {
        return Err(GpuError::Blocked(
            "FWSEC: descriptorul nu e versiunea 3 (asteptata pe GA10x)",
        ));
    }
    if desc.signature_count == 0
        || desc.header_size() < DESC_V3_LEN + desc.signature_count as u32 * RSA3K_SIG_LEN
    {
        return Err(GpuError::InvalidArgument(
            "FWSEC: antetul descriptorului nu cuprinde semnaturile",
        ));
    }
    let loaded = desc.imem_load_size as u64 + desc.dmem_load_size as u64;
    if desc.stored_size == 0 || loaded > desc.stored_size as u64 {
        return Err(GpuError::InvalidArgument(
            "FWSEC: IMEM + DMEM mai mari decat ucode-ul stocat",
        ));
    }
    let loc = FwsecLocation {
        pmu_table: map_ranges(&sections, table_rel, 1)?[0].0,
        desc_prom: map_ranges(&sections, desc_rel, 1)?[0].0,
        sections,
        pmu_apps: apps,
        desc_rel,
        desc,
    };
    if !loc.fits() {
        return Err(GpuError::OutOfBounds {
            offset: desc_rel as u64,
            len: desc.header_size() as u64 + desc.stored_size as u64,
            limit: loc.total() as u64,
        });
    }
    Ok(loc)
}

#[cfg(test)]
mod tests {
    //! All tests: SIMULARE, on synthetic ROM bytes (gpu-sim::rom). They prove
    //! the walk follows our reading of the public descriptions, nothing more.
    use super::*;
    use crate::regs::{PROM_BASE, PROM_SIZE};
    use crate::vbios::{VBIOS_TIMEOUT_US, read_layout};
    use gpu_sim::clock::SimClock;
    use gpu_sim::mmio::SimMmio;
    use gpu_sim::rom::{self, SimFwsec};

    fn board(bytes: &[u8]) -> SimMmio {
        let mut m = SimMmio::new(INIT_WINDOW);
        m.load_bytes(PROM_BASE, bytes);
        m
    }

    fn run(m: &mut SimMmio) -> (GpuResult<FwsecLocation>, Journal) {
        let mut j = Journal::new(32);
        let mut c = SimClock::default();
        let layout = read_layout(m, &mut j, &mut c, VBIOS_TIMEOUT_US).unwrap();
        m.reads.clear();
        let r = locate(m, &mut j, &mut c, VBIOS_TIMEOUT_US, &layout);
        (r, j)
    }

    fn with(f: SimFwsec) -> SimMmio {
        board(&rom::ga10x_like_chain_with_fwsec(&f))
    }

    #[test]
    fn fwsec_prod_is_located_without_reading_the_ucode() {
        let mut m = with(SimFwsec::ga10x_like());
        let (r, j) = run(&mut m);
        let l = r.unwrap();
        assert_eq!(l.sections, [(0x18000, 0x4000), (0x1c000, 0x4000)]);
        assert_eq!(l.pmu_table, 0x18200);
        assert_eq!(l.pmu_apps, [0x05, 0x85]);
        assert_eq!((l.desc_rel, l.desc_prom), (0x400, 0x18400));
        let d = l.desc;
        assert_eq!(d.version(), 3);
        assert_eq!(d.header_size(), 44 + 2 * 384);
        assert_eq!((d.imem_load_size, d.dmem_load_size), (0x3000, 0x1000));
        assert_eq!(d.stored_size, 0x4000);
        assert_eq!((d.signature_count, d.signature_versions), (2, 0b0110));
        assert!(d.signatures_consistent());
        assert_eq!(l.signature_ranges().unwrap(), [(0x1842c, 768)]);
        let ucode = l.ucode_ranges().unwrap();
        assert_eq!(ucode, [(0x1872c, 0x38d4), (0x1c000, 0x72c)]);
        assert!(l.ucode_crosses_images() && l.fits());
        // Read-only, inside the PROM window, and never inside the ucode.
        assert!(m.writes.is_empty());
        assert_eq!(j.writes(), 0);
        assert!(!m.reads.is_empty());
        for &o in &m.reads {
            assert!((PROM_BASE..PROM_BASE + PROM_SIZE).contains(&o));
            let p = (o - PROM_BASE) as u32;
            for &(s, n) in &ucode {
                assert!(p + 4 <= s || p >= s + n, "ucode read at {p:#x}");
            }
        }
        assert!(j.events().any(|e| matches!(
            e,
            Event::Step {
                name: "I3: descriptorul FWSEC",
                ok: true
            }
        )));
    }

    #[test]
    fn missing_fwsec_prod_is_blocked() {
        let mut f = SimFwsec::ga10x_like();
        f.entries = alloc::vec![(0x05, 0, 0x100), (0x45, 0, 0x400)];
        let mut m = with(f);
        assert!(matches!(run(&mut m).0, Err(GpuError::Blocked(_))));
    }

    #[test]
    fn other_descriptor_versions_are_blocked() {
        let mut f = SimFwsec::ga10x_like();
        f.desc_version = 2;
        let (r, j) = run(&mut with(f));
        assert!(matches!(r, Err(GpuError::Blocked(_))));
        assert!(j.events().any(|e| matches!(
            e,
            Event::Step {
                name: "I3: descriptorul FWSEC",
                ok: false
            }
        )));
    }

    #[test]
    fn header_without_room_for_signatures_is_refused() {
        let mut bytes = rom::ga10x_like_chain_with_fwsec(&SimFwsec::ga10x_like());
        bytes[0x18400 + 39] = 5; // 5 signatures claimed, header holds 2
        assert!(matches!(
            run(&mut board(&bytes)).0,
            Err(GpuError::InvalidArgument(_))
        ));
    }

    #[test]
    fn ucode_running_past_the_fwsec_images_is_refused() {
        let mut f = SimFwsec::ga10x_like();
        f.imem_load_size = 0x8000;
        assert!(matches!(
            run(&mut with(f)).0,
            Err(GpuError::OutOfBounds { .. })
        ));
        let mut f = SimFwsec::ga10x_like();
        f.dmem_load_size = 0x1000;
        f.stored_extra = 0;
        let mut bytes = rom::ga10x_like_chain_with_fwsec(&f);
        // stored_size smaller than IMEM + DMEM
        bytes[0x18404..0x18408].copy_from_slice(&0x100u32.to_le_bytes());
        assert!(matches!(
            run(&mut board(&bytes)).0,
            Err(GpuError::InvalidArgument(_))
        ));
    }

    #[test]
    fn pointer_into_pci_at_is_refused() {
        let mut bytes = rom::ga10x_like_chain_with_fwsec(&SimFwsec::ga10x_like());
        bytes[0x3000..0x3004].copy_from_slice(&0x100u32.to_le_bytes());
        assert!(matches!(
            run(&mut board(&bytes)).0,
            Err(GpuError::InvalidArgument(_))
        ));
    }

    #[test]
    fn ifr_prefix_shifts_everything() {
        let chain = rom::ga10x_like_chain_with_fwsec(&SimFwsec::ga10x_like());
        let bytes = rom::with_ifr_v3(&chain);
        let start = (bytes.len() - chain.len()) as u32;
        let l = run(&mut board(&bytes)).0.unwrap();
        assert_eq!(l.desc_prom, start + 0x18400);
        assert_eq!(l.ucode_ranges().unwrap()[1], (start + 0x1c000, 0x72c));
    }

    #[test]
    fn no_fwsec_image_or_token_is_blocked() {
        let bytes = rom::build_chain(&[rom::SimImage {
            code_type: CODE_PCI_AT,
            blocks: 8,
            npde: false,
            bit: None,
        }]);
        let mut m = board(&bytes);
        assert!(matches!(run(&mut m).0, Err(GpuError::Blocked(_))));
    }

    #[test]
    fn signature_choice() {
        assert_eq!(signature_index(0b0110, 1), Ok(0));
        assert_eq!(signature_index(0b0110, 2), Ok(1));
        assert!(matches!(
            signature_index(0b0110, 0),
            Err(GpuError::Blocked(_))
        ));
        assert!(matches!(
            signature_index(0xffff, 16),
            Err(GpuError::InvalidArgument(_))
        ));
        assert_eq!(signature_index(0x8001, 15), Ok(1));
    }

    #[test]
    fn range_mapping() {
        let s = [(0x100, 0x10), (0x300, 0x10)];
        assert_eq!(map_ranges(&s, 0, 0x10), Ok(alloc::vec![(0x100, 0x10)]));
        assert_eq!(
            map_ranges(&s, 0x8, 0x10),
            Ok(alloc::vec![(0x108, 0x8), (0x300, 0x8)])
        );
        assert!(matches!(
            map_ranges(&s, 0x18, 0x9),
            Err(GpuError::OutOfBounds { .. })
        ));
    }

    #[test]
    fn needs_init_window() {
        let layout = VbiosLayout {
            ifr_version: None,
            rom_start: 0,
            images: Vec::new(),
            bit: None,
            prom_reads: 0,
        };
        let mut m = SimMmio::new(0x1000);
        let mut j = Journal::new(4);
        let r = locate(&mut m, &mut j, &mut SimClock::default(), 1, &layout);
        assert!(matches!(r, Err(GpuError::InvalidArgument(_))));
        assert!(m.reads.is_empty());
    }
}
