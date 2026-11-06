//! SIMULARE: a sparse model of the board's VRAM and an independent walker of
//! the GPU page tables (MMU format version 2, used by Ampere).
//!
//! The walker is the second, independent reading of the format that checks
//! the driver's encoder (`nvidia_ga104::mmu`). Its bit positions are written
//! by hand from NVIDIA's `dev_mmu.h` (open-gpu-kernel-modules 570.144,
//! `kernel-open/nvidia-uvm/hwref/turing/tu102/dev_mmu.h`, MIT) and the level
//! layout from NVIDIA's comment in `kern_gmmu_fmt_gp10x.c` (same tag, MIT):
//! PD3 [48:47] → PD2 [46:38] → PD1 [37:29] → PD0 [28:21] (16-byte dual
//! entries) → PT [20:12] (4 KiB pages). It deliberately does **not** use the
//! generated constants of `nvidia-rm570`.
//!
//! Real VRAM is not cleared at power-on; this model returns a poison pattern
//! for every byte nobody wrote, so a page table the driver forgot to clear is
//! caught (poison decodes as "system memory", which the walker refuses).
//! Nothing here says how the real MMU behaves; it only checks that what the
//! driver writes means what the driver intends, by NVIDIA's documented format.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;

/// What an unwritten 8-byte word of the model reads as.
pub const POISON: u64 = 0xa5a5_a5a5_a5a5_a5a5;

const PAGE: u64 = 0x1000;

/// Sparse VRAM: only the 4 KiB pages somebody wrote exist.
pub struct SimVram {
    size: u64,
    pages: BTreeMap<u64, Box<[u64; 512]>>,
    /// Number of 8-byte writes (to count what the driver did).
    pub writes: u64,
}

impl SimVram {
    pub fn new(size: u64) -> Self {
        SimVram {
            size,
            pages: BTreeMap::new(),
            writes: 0,
        }
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn write_u64(&mut self, pa: u64, v: u64) -> Result<(), &'static str> {
        if !pa.is_multiple_of(8) {
            return Err("VRAM: scriere nealiniata");
        }
        if pa.checked_add(8).is_none_or(|e| e > self.size) {
            return Err("VRAM: scriere in afara memoriei");
        }
        let page = self
            .pages
            .entry(pa & !(PAGE - 1))
            .or_insert_with(|| Box::new([POISON; 512]));
        page[((pa % PAGE) / 8) as usize] = v;
        self.writes += 1;
        Ok(())
    }

    pub fn read_u64(&self, pa: u64) -> Result<u64, &'static str> {
        if !pa.is_multiple_of(8) || pa.checked_add(8).is_none_or(|e| e > self.size) {
            return Err("VRAM: citire nealiniata sau in afara memoriei");
        }
        Ok(self
            .pages
            .get(&(pa & !(PAGE - 1)))
            .map_or(POISON, |p| p[((pa % PAGE) / 8) as usize]))
    }

    /// Whether any byte of `[pa, pa+len)` was ever written.
    pub fn touched(&self, pa: u64, len: u64) -> bool {
        let first = pa & !(PAGE - 1);
        self.pages
            .range(first..pa.saturating_add(len))
            .next()
            .is_some()
    }
}

/// A successful walk: where the VA lands and the attributes of its PTE.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Translation {
    pub pa: u64,
    pub volatile: bool,
    pub privileged: bool,
    pub read_only: bool,
    pub kind: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WalkError {
    /// The VA has more than 49 bits.
    VaTooWide,
    /// Level `0` = PD3 … `4` = PT: no table/page there (aperture invalid or PTE not valid).
    NotPresent(u8),
    /// The entry points to system memory (or is poison): not modelled.
    SystemMemory(u8),
    /// A big page (2 MiB / 512 MiB / 64 KiB table) — the driver does not make them yet.
    BigPage(u8),
    /// A peer GPU's memory.
    Peer(u8),
    /// A table address outside VRAM.
    OutOfVram(u8),
}

/// `ADDRESS_VID` is bits 32:8 of an entry, in 4 KiB units.
fn vid_addr(word: u64) -> u64 {
    ((word >> 8) & ((1 << 25) - 1)) << 12
}

/// Translates `va` starting from the root table at `root` (VRAM).
pub fn translate(vram: &SimVram, root: u64, va: u64) -> Result<Translation, WalkError> {
    if va >> 49 != 0 {
        return Err(WalkError::VaTooWide);
    }
    let rd = |lvl: u8, pa: u64| vram.read_u64(pa).map_err(|_| WalkError::OutOfVram(lvl));
    // PD3, PD2, PD1: 8-byte entries. Bit 0 = IS_PTE, 2:1 aperture (1 = video
    // memory), 32:8 address, 35:33 peer.
    let mut table = root;
    for (lvl, shift, bits) in [(0u8, 47u32, 2u32), (1, 38, 9), (2, 29, 9)] {
        let idx = (va >> shift) & ((1 << bits) - 1);
        let e = rd(lvl, table + idx * 8)?;
        match (e >> 1) & 3 {
            0 => return Err(WalkError::NotPresent(lvl)),
            1 => {}
            _ => return Err(WalkError::SystemMemory(lvl)),
        }
        if e & 1 != 0 {
            return Err(WalkError::BigPage(lvl));
        }
        if (e >> 33) & 7 != 0 {
            return Err(WalkError::Peer(lvl));
        }
        table = vid_addr(e);
    }
    // PD0: 16-byte dual entries. Low word: bit 0 IS_PTE, 2:1 big aperture.
    // High word (bits 64..127): 66:65 small aperture → bits 2:1, 96:72 small
    // address → bits 32:8, 99:97 peer → bits 35:33.
    let idx = (va >> 21) & 0xff;
    let lo = rd(3, table + idx * 16)?;
    let hi = rd(3, table + idx * 16 + 8)?;
    if lo & 1 != 0 || (lo >> 1) & 3 != 0 {
        return Err(WalkError::BigPage(3));
    }
    match (hi >> 1) & 3 {
        0 => return Err(WalkError::NotPresent(3)),
        1 => {}
        _ => return Err(WalkError::SystemMemory(3)),
    }
    if (hi >> 33) & 7 != 0 {
        return Err(WalkError::Peer(3));
    }
    let pt = vid_addr(hi);
    // PT: bit 0 valid, 2:1 aperture (0 = video memory here!), 3 volatile,
    // 5 privilege, 6 read-only, 32:8 address, 35:33 peer, 63:56 kind.
    let e = rd(4, pt + ((va >> 12) & 0x1ff) * 8)?;
    if e & 1 == 0 {
        return Err(WalkError::NotPresent(4));
    }
    match (e >> 1) & 3 {
        0 => {}
        1 => return Err(WalkError::Peer(4)),
        _ => return Err(WalkError::SystemMemory(4)),
    }
    if (e >> 33) & 7 != 0 {
        return Err(WalkError::Peer(4));
    }
    Ok(Translation {
        pa: vid_addr(e) + (va & 0xfff),
        volatile: e & (1 << 3) != 0,
        privileged: e & (1 << 5) != 0,
        read_only: e & (1 << 6) != 0,
        kind: (e >> 56) as u8,
    })
}

/// C-83 (I11): where a page lands — VRAM or host RAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Aperture {
    Vram,
    /// `SYSTEM_COHERENT_MEMORY` (2): host RAM, snooped.
    SysCoherent,
    /// `SYSTEM_NON_COHERENT_MEMORY` (3): host RAM, not snooped.
    SysNonCoherent,
}

/// C-83 (I11): a walk that also accepts host-RAM pages (the last level only;
/// directories must still be in VRAM).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AnyTranslation {
    pub aperture: Aperture,
    /// VRAM address or host bus address.
    pub addr: u64,
    pub volatile: bool,
    pub read_only: bool,
    pub kind: u8,
}

/// C-83 (I11): like [`translate`], but a PTE with aperture 2 or 3 is a
/// host-RAM page: `ADDRESS_SYS` = bits 53:8, in 4 KiB units (hand reading of
/// `dev_mmu.h`, like the rest of this walker). A PTE equal to the poison
/// pattern is refused as [`WalkError::SystemMemory`] (it would decode as a
/// valid coherent page — exactly the garbage a forgotten table holds).
/// [`translate`] is unchanged: it still refuses every host-RAM page.
pub fn translate_any(vram: &SimVram, root: u64, va: u64) -> Result<AnyTranslation, WalkError> {
    match translate(vram, root, va) {
        Ok(t) => {
            return Ok(AnyTranslation {
                aperture: Aperture::Vram,
                addr: t.pa,
                volatile: t.volatile,
                read_only: t.read_only,
                kind: t.kind,
            });
        }
        Err(WalkError::SystemMemory(4)) => {}
        Err(e) => return Err(e),
    }
    // Only the PT entry pointed to host RAM: walk again to read it.
    let pt = pt_of(vram, root, va)?;
    let e = vram
        .read_u64(pt + ((va >> 12) & 0x1ff) * 8)
        .map_err(|_| WalkError::OutOfVram(4))?;
    if e == POISON {
        return Err(WalkError::SystemMemory(4));
    }
    let aperture = match (e >> 1) & 3 {
        2 => Aperture::SysCoherent,
        3 => Aperture::SysNonCoherent,
        _ => return Err(WalkError::SystemMemory(4)),
    };
    let page = (e >> 8) & ((1 << 46) - 1);
    Ok(AnyTranslation {
        aperture,
        addr: (page << 12) + (va & 0xfff),
        volatile: e & (1 << 3) != 0,
        read_only: e & (1 << 6) != 0,
        kind: (e >> 56) as u8,
    })
}

/// The PT address for `va` (the directory part of [`translate`], which has
/// already accepted it when this is called).
fn pt_of(vram: &SimVram, root: u64, va: u64) -> Result<u64, WalkError> {
    let rd = |lvl: u8, pa: u64| vram.read_u64(pa).map_err(|_| WalkError::OutOfVram(lvl));
    let mut table = root;
    for (lvl, shift, bits) in [(0u8, 47u32, 2u32), (1, 38, 9), (2, 29, 9)] {
        let idx = (va >> shift) & ((1 << bits) - 1);
        table = vid_addr(rd(lvl, table + idx * 8)?);
    }
    let idx = (va >> 21) & 0xff;
    Ok(vid_addr(rd(3, table + idx * 16 + 8)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unwritten_vram_is_poison_and_bounds_hold() {
        let mut v = SimVram::new(1 << 20);
        assert_eq!(v.read_u64(0x100), Ok(POISON));
        assert!(v.write_u64(0x100, 7).is_ok());
        assert_eq!(v.read_u64(0x100), Ok(7));
        assert_eq!(v.read_u64(0x108), Ok(POISON));
        assert!(v.write_u64(0x101, 0).is_err());
        assert!(v.write_u64((1 << 20) - 4, 0).is_err());
        assert!(v.read_u64(1 << 20).is_err());
        assert!(v.touched(0x0, 0x1000) && !v.touched(0x1000, 0x1000));
        assert_eq!(v.writes, 1);
    }

    /// C-83: a host-RAM PTE (aperture 2, `ADDRESS_SYS`) is found by
    /// `translate_any`, refused by `translate`; poison is refused by both.
    #[test]
    fn host_ram_pages_translate_only_with_translate_any() {
        let mut v = SimVram::new(64 << 20);
        let (pd3, pd2, pd1, pd0, pt) = (0x1000u64, 0x2000, 0x3000, 0x4000, 0x5000);
        let pde = |next: u64| (1 << 1) | ((next >> 12) << 8);
        let va: u64 = (1 << 38) | (2 << 29) | (3 << 21) | (4 << 12) | 0x40;
        v.write_u64(pd3, pde(pd2)).unwrap();
        v.write_u64(pd2 + 8, pde(pd1)).unwrap();
        v.write_u64(pd1 + 2 * 8, pde(pd0)).unwrap();
        v.write_u64(pd0 + 3 * 16, 0).unwrap();
        v.write_u64(pd0 + 3 * 16 + 8, pde(pt)).unwrap();
        let bus = 0x0012_3456_7000u64; // above 4 GiB, bits beyond ADDRESS_VID
        v.write_u64(
            pt + 4 * 8,
            1 | (2 << 1) | (1 << 3) | ((bus >> 12) << 8) | (6 << 56),
        )
        .unwrap();
        assert_eq!(translate(&v, pd3, va), Err(WalkError::SystemMemory(4)));
        assert_eq!(
            translate_any(&v, pd3, va),
            Ok(AnyTranslation {
                aperture: Aperture::SysCoherent,
                addr: bus + 0x40,
                volatile: true,
                read_only: false,
                kind: 6
            })
        );
        // The neighbour PTE was never written: poison, refused.
        assert_eq!(
            translate_any(&v, pd3, va + 0x1000),
            Err(WalkError::SystemMemory(4))
        );
        // A VRAM page still comes out as VRAM.
        v.write_u64(pt + 5 * 8, 1 | (0x0234_5000u64 >> 12) << 8)
            .unwrap();
        assert_eq!(
            translate_any(&v, pd3, va + 0x1000 - 0x40).unwrap().aperture,
            Aperture::Vram
        );
    }

    /// A hand-built walk (tables written here word by word) translates, and
    /// poison at any level is refused instead of followed.
    #[test]
    fn hand_built_tables_translate() {
        let mut v = SimVram::new(64 << 20);
        let (pd3, pd2, pd1, pd0, pt) = (0x1000u64, 0x2000, 0x3000, 0x4000, 0x5000);
        let pde = |next: u64| (1 << 1) | ((next >> 12) << 8);
        let va: u64 = (1 << 47) | (3 << 38) | (5 << 29) | (7 << 21) | (9 << 12) | 0x123;
        v.write_u64(pd3 + 8, pde(pd2)).unwrap();
        v.write_u64(pd2 + 3 * 8, pde(pd1)).unwrap();
        v.write_u64(pd1 + 5 * 8, pde(pd0)).unwrap();
        v.write_u64(pd0 + 7 * 16, 0).unwrap();
        v.write_u64(pd0 + 7 * 16 + 8, pde(pt)).unwrap();
        v.write_u64(
            pt + 9 * 8,
            1 | (1 << 3) | ((0x0234_5000u64 >> 12) << 8) | (6 << 56),
        )
        .unwrap();
        assert_eq!(
            translate(&v, pd3, va),
            Ok(Translation {
                pa: 0x0234_5123,
                volatile: true,
                privileged: false,
                read_only: false,
                kind: 6
            })
        );
        // The neighbour page: PT entry never written → poison → refused.
        assert_eq!(
            translate(&v, pd3, va + 0x1000),
            Err(WalkError::SystemMemory(4))
        );
        // Another PD2 slot: poison at level 1.
        assert_eq!(
            translate(&v, pd3, va ^ (1 << 38)),
            Err(WalkError::SystemMemory(1))
        );
        assert_eq!(translate(&v, pd3, 1 << 49), Err(WalkError::VaTooWide));
        v.write_u64(pd3, 0).unwrap();
        assert_eq!(translate(&v, pd3, 0), Err(WalkError::NotPresent(0)));
    }
}
