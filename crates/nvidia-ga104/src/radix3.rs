//! I5 preparation — the "radix3" table that tells GSP where the pages of
//! the GSP-RM image are in system RAM (pure data, no device access).
//!
//! Format (CONFIRMAT: description in the nova-core patch "firmware: process
//! and prepare the GSP firmware", v4 07/11, LKML Sep 2025; no code copied):
//! 4 KiB pages, 8-byte little-endian entries holding DMA addresses.
//! - level 0: one page, entry 0 = address of the level-1 page, rest zero;
//! - level 1: one entry per level-2 page;
//! - level 2: one entry per 4 KiB page of the firmware image.
//!
//! Level 0 has a single entry, so level 1 must be one contiguous run; we
//! keep it to one page (512 level-2 pages = 1 GiB of image, far above the
//! tens of MiB of GSP-RM). With VT-d off on the workstation (FAPT, K0), DMA
//! addresses are physical addresses. The firmware bytes are not handled
//! here: only the addresses of the pages that will hold them.

use alloc::vec;
use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

pub const PAGE: u64 = 4096;
pub const ENTRIES_PER_PAGE: usize = (PAGE / 8) as usize;

/// How many pages each level needs for an image of `image_size` bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Radix3Shape {
    pub data_pages: usize,
    pub level2_pages: usize,
    pub level1_pages: usize,
}

impl Radix3Shape {
    pub fn for_size(image_size: u64) -> GpuResult<Self> {
        if image_size == 0 {
            return Err(GpuError::InvalidArgument("radix3: imagine goala"));
        }
        let data_pages = image_size.div_ceil(PAGE) as usize;
        let level2_pages = data_pages.div_ceil(ENTRIES_PER_PAGE);
        if level2_pages > ENTRIES_PER_PAGE {
            return Err(GpuError::NotPermitted(
                "radix3: imagine > 1 GiB (nivelul 1 ar avea mai mult de o pagina)",
            ));
        }
        Ok(Radix3Shape {
            data_pages,
            level2_pages,
            level1_pages: 1,
        })
    }
    /// Pages to allocate for the table itself (level 0 + 1 + 2).
    pub fn table_pages(&self) -> usize {
        1 + self.level1_pages + self.level2_pages
    }
}

/// The three levels, as 512-entry pages.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Radix3 {
    pub level0: Vec<u64>,
    pub level1: Vec<u64>,
    pub level2: Vec<Vec<u64>>,
}

fn page_addr(a: u64) -> GpuResult<u64> {
    if a == 0 || !a.is_multiple_of(PAGE) {
        return Err(GpuError::InvalidArgument(
            "radix3: adresa de pagina nula sau nealiniata la 4 KiB",
        ));
    }
    Ok(a)
}

impl Radix3 {
    /// `data` = DMA address of each image page in order; `level1` = address
    /// of the level-1 page; `level2` = addresses of the level-2 pages.
    pub fn build(data: &[u64], level1: u64, level2: &[u64]) -> GpuResult<Self> {
        let shape = Radix3Shape::for_size(data.len() as u64 * PAGE)?;
        if level2.len() != shape.level2_pages {
            return Err(GpuError::InvalidArgument(
                "radix3: numar gresit de pagini pentru nivelul 2",
            ));
        }
        let mut level0 = vec![0u64; ENTRIES_PER_PAGE];
        level0[0] = page_addr(level1)?;
        let mut l1 = vec![0u64; ENTRIES_PER_PAGE];
        for (k, &a) in level2.iter().enumerate() {
            l1[k] = page_addr(a)?;
        }
        let mut l2 = Vec::with_capacity(level2.len());
        for chunk in data.chunks(ENTRIES_PER_PAGE) {
            let mut p = vec![0u64; ENTRIES_PER_PAGE];
            for (k, &a) in chunk.iter().enumerate() {
                p[k] = page_addr(a)?;
            }
            l2.push(p);
        }
        // The table pages must not be image pages, nor each other.
        let mut all: Vec<u64> = data.iter().chain(level2).copied().collect();
        all.push(level1);
        all.sort_unstable();
        if all.windows(2).any(|w| w[0] == w[1]) {
            return Err(GpuError::InvalidArgument(
                "radix3: aceeasi pagina folosita de doua ori",
            ));
        }
        Ok(Radix3 {
            level0,
            level1: l1,
            level2: l2,
        })
    }

    /// Bytes of one level page, as GSP will read them (little endian).
    pub fn page_bytes(entries: &[u64]) -> Vec<u8> {
        entries.iter().flat_map(|e| e.to_le_bytes()).collect()
    }

    /// Walk the table like GSP: image offset → DMA address. `level2_addrs`
    /// says which level-2 page lives at which address (as passed to build).
    pub fn translate(&self, level2_addrs: &[u64], offset: u64) -> Option<u64> {
        let page = (offset / PAGE) as usize;
        let l2_index = page / ENTRIES_PER_PAGE;
        let l2_addr = *self.level1.get(l2_index)?;
        if l2_addr == 0 {
            return None;
        }
        let at = level2_addrs.iter().position(|&a| a == l2_addr)?;
        let a = self.level2.get(at)?[page % ENTRIES_PER_PAGE];
        (a != 0).then_some(a + offset % PAGE)
    }
}

#[cfg(test)]
mod tests {
    //! Pure data; addresses are SIM.
    use super::*;

    /// Non-contiguous, out-of-order pages above 16 MiB (the OS DMA window).
    fn scattered() -> Vec<u64> {
        alloc::vec![0x0150_0000, 0x0100_3000, 0x0180_0000, 0x0100_1000]
    }

    #[test]
    fn shape() {
        assert!(Radix3Shape::for_size(0).is_err());
        assert_eq!(
            Radix3Shape::for_size(1).unwrap(),
            Radix3Shape {
                data_pages: 1,
                level2_pages: 1,
                level1_pages: 1
            }
        );
        // 30 MiB SIM image: 7680 pages, 15 level-2 pages.
        let s = Radix3Shape::for_size(30 << 20).unwrap();
        assert_eq!(
            (s.data_pages, s.level2_pages, s.table_pages()),
            (7680, 15, 17)
        );
        assert!(Radix3Shape::for_size((1 << 30) + 1).is_err());
        assert!(Radix3Shape::for_size(1 << 30).is_ok());
    }

    #[test]
    fn walk_matches_the_pages_for_every_offset() {
        let n = 1300; // three level-2 pages, the last one partial
        let mut data: Vec<u64> = (0..n as u64).map(|k| 0x4000_0000 + k * 2 * PAGE).collect();
        data.reverse(); // order in the image ≠ order in RAM
        let l2 = [0x0200_0000, 0x0200_3000, 0x0200_1000];
        let r = Radix3::build(&data, 0x0200_2000, &l2).unwrap();
        assert_eq!(r.level0[0], 0x0200_2000);
        assert!(r.level0[1..].iter().all(|&e| e == 0));
        assert_eq!(&r.level1[..3], &l2);
        for (k, &page) in data.iter().enumerate() {
            for off in [0, 1, 4095] {
                let o = k as u64 * PAGE + off;
                assert_eq!(r.translate(&l2, o), Some(page + off));
            }
        }
        assert_eq!(r.translate(&l2, n as u64 * PAGE), None, "past the image");
        let bytes = Radix3::page_bytes(&r.level0);
        assert_eq!(bytes.len(), PAGE as usize);
        assert_eq!(&bytes[..8], &0x0200_2000u64.to_le_bytes());
    }

    #[test]
    fn bad_addresses_are_refused() {
        let data = scattered();
        assert!(Radix3::build(&data, 0x0200_0800, &[0x0300_0000]).is_err());
        assert!(Radix3::build(&data, 0, &[0x0300_0000]).is_err());
        assert!(Radix3::build(&data, 0x0200_0000, &[]).is_err());
        assert!(Radix3::build(&[0x1000, 0x1001], 0x0200_0000, &[0x0300_0000]).is_err());
        // A table page that is also an image page.
        assert!(Radix3::build(&data, data[0], &[0x0300_0000]).is_err());
        assert!(Radix3::build(&data, 0x0200_0000, &[0x0300_0000]).is_ok());
    }
}
