//! I4c: the container headers of the linux-firmware files
//! `bootloader-570.144.bin` and `booter_load-570.144.bin` — *where* the
//! pieces are, the way any loader reads them. The pieces themselves (code,
//! data, signatures) stay opaque bytes: nothing here looks inside them.
//! **Tested only on synthetic images (SIMULARE)**. C-39 (29 oct): on the
//! real files the first version refused 3 of 4 (`bin_size` taken for the
//! file length, the signature count read as a value instead of through its
//! pointer); the header words of the official files were then read and the
//! stand-ins `gpu-sim::fw_stick::*_570_144` reproduce them exactly.
//!
//! Sources (read as specification, re-written, docs/PROVENANCE.md):
//! - `RM_RISCV_UCODE_DESC` (84 B): NVIDIA `rmRiscvUcode.h`, 570.144, MIT —
//!   generated into `nvidia-rm570::structs` with bindgen size/offset checks;
//! - the outer header and the "HS v2" headers of Booter: nouveau
//!   `include/nvfw/fw.h` and `include/nvfw/hs.h` (MIT; `nvfw_bin_hdr`,
//!   `nvfw_hs_header_v2`, `nvfw_hs_load_header_v2`). Not NVIDIA documents:
//!   DEDUS, then checked against the header words of the official 570.144
//!   files (C-39);
//! - which descriptor fields feed `GspFwWprMeta`: nouveau `r535.c`.

use alloc::vec::Vec;
use core::mem::{offset_of, size_of};
use core::ops::Range;

use gpu_core::error::{GpuError, GpuResult};
use nvidia_rm570::structs::RM_RISCV_UCODE_DESC;

/// `nvfw_bin_hdr.bin_magic` (nouveau `fw.h`).
pub const BIN_MAGIC: u32 = 0x10de;
/// `nvfw_bin_hdr`: 6 x u32.
pub const BIN_HDR_SIZE: usize = 24;

fn u32_at(f: &[u8], off: usize) -> GpuResult<u32> {
    let b = f
        .get(
            off..off
                .checked_add(4)
                .ok_or(GpuError::InvalidArgument("antet: offset prea mare"))?,
        )
        .ok_or(GpuError::OutOfBounds {
            offset: off as u64,
            len: 4,
            limit: f.len() as u64,
        })?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

fn range(f: &[u8], off: u32, size: u32) -> GpuResult<Range<usize>> {
    let (o, s) = (off as usize, size as usize);
    match o.checked_add(s) {
        Some(e) if e <= f.len() => Ok(o..e),
        _ => Err(GpuError::OutOfBounds {
            offset: u64::from(off),
            len: u64::from(size),
            limit: f.len() as u64,
        }),
    }
}

/// The outer header every one of these files starts with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BinHdr {
    pub bin_ver: u32,
    pub bin_size: u32,
    pub header_offset: u32,
    pub data_offset: u32,
    pub data_size: u32,
}

impl BinHdr {
    pub fn parse(f: &[u8]) -> GpuResult<Self> {
        if u32_at(f, 0)? != BIN_MAGIC {
            return Err(GpuError::InvalidArgument(
                "firmware: antetul nu incepe cu 0x10de",
            ));
        }
        let h = BinHdr {
            bin_ver: u32_at(f, 4)?,
            bin_size: u32_at(f, 8)?,
            header_offset: u32_at(f, 12)?,
            data_offset: u32_at(f, 16)?,
            data_size: u32_at(f, 20)?,
        };
        // `bin_size` is NOT the file length: on the real 570.144 files
        // (C-39, workstation 29 oct) it is larger than the file
        // (booter_load: 0xf000 for 61 304 B). Loaders (nouveau, nova-core)
        // never use it; neither do we. It is kept only for the report.
        // What must hold: the outer header and the data blob are inside
        // the file, and the blob starts after the outer header.
        if (h.data_offset as usize) < BIN_HDR_SIZE || (h.header_offset as usize) < BIN_HDR_SIZE {
            return Err(GpuError::InvalidArgument(
                "firmware: antetul interior sau datele suprapuse peste antetul exterior",
            ));
        }
        range(f, h.data_offset, h.data_size)?;
        Ok(h)
    }

    pub fn data(&self) -> Range<usize> {
        self.data_offset as usize..(self.data_offset + self.data_size) as usize
    }
}

/// `bootloader-570.144.bin`: the GSP RISC-V bootloader and its descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RiscvBootloader {
    pub hdr: BinHdr,
    pub version: u32,
    pub app_version: u32,
    /// Offsets inside the data blob (`hdr.data()`), as `GspFwWprMeta` wants.
    pub code_offset: u32,
    pub data_offset: u32,
    pub manifest_offset: u32,
    pub manifest_size: u32,
}

fn desc_field(f: &[u8], base: usize, off: usize) -> GpuResult<u32> {
    u32_at(f, base + off)
}

impl RiscvBootloader {
    pub fn parse(f: &[u8]) -> GpuResult<Self> {
        let hdr = BinHdr::parse(f)?;
        let base = hdr.header_offset as usize;
        range(
            f,
            hdr.header_offset,
            size_of::<RM_RISCV_UCODE_DESC>() as u32,
        )?;
        let d = |o| desc_field(f, base, o);
        let version = d(offset_of!(RM_RISCV_UCODE_DESC, version))?;
        let code = (
            d(offset_of!(RM_RISCV_UCODE_DESC, monitorCodeOffset))?,
            d(offset_of!(RM_RISCV_UCODE_DESC, monitorCodeSize))?,
        );
        let data = (
            d(offset_of!(RM_RISCV_UCODE_DESC, monitorDataOffset))?,
            d(offset_of!(RM_RISCV_UCODE_DESC, monitorDataSize))?,
        );
        let man = (
            d(offset_of!(RM_RISCV_UCODE_DESC, manifestOffset))?,
            d(offset_of!(RM_RISCV_UCODE_DESC, manifestSize))?,
        );
        // Every piece must lie inside the data blob.
        let blob = &f[hdr.data()];
        for (o, s) in [code, data, man] {
            if s == 0 {
                return Err(GpuError::InvalidArgument(
                    "bootloader: o parte are marimea 0",
                ));
            }
            range(blob, o, s)?;
        }
        Ok(RiscvBootloader {
            hdr,
            version,
            app_version: d(offset_of!(RM_RISCV_UCODE_DESC, appVersion))?,
            code_offset: code.0,
            data_offset: data.0,
            manifest_offset: man.0,
            manifest_size: man.1,
        })
    }
}

/// One application of a Booter image (`nvfw_hs_load_header_v2_app`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HsApp {
    pub offset: u32,
    pub size: u32,
}

/// `booter_load-570.144.bin` (and `booter_unload`): "HS v2" layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HsV2 {
    pub hdr: BinHdr,
    /// All production signatures, `num_sig` of equal size, back to back.
    pub sig_prod: Range<usize>,
    pub num_sig: u32,
    /// Where in the image the chosen signature is copied, and which one
    /// (both read through the pointers `patch_loc` / `patch_sig`).
    pub patch_loc: u32,
    pub patch_sig: u32,
    pub meta_data: Range<usize>,
    pub os_code: HsApp,
    pub os_data: HsApp,
    pub apps: Vec<HsApp>,
}

impl HsV2 {
    pub fn sig_size(&self) -> usize {
        self.sig_prod.len() / self.num_sig as usize
    }

    pub fn parse(f: &[u8]) -> GpuResult<Self> {
        let hdr = BinHdr::parse(f)?;
        let h = hdr.header_offset as usize;
        let w = |i: usize| u32_at(f, h + 4 * i);
        let (sig_off, sig_size, loc_ptr, sig_ptr) = (w(0)?, w(1)?, w(2)?, w(3)?);
        // Word 6 is a *pointer* to the signature count, like patch_loc and
        // patch_sig (nouveau `nvkm_falcon_fw_ctor_hs_v2`; confirmed on the
        // real 570.144 headers, C-39), not the count itself.
        let (meta_off, meta_size, num_sig_ptr, lh_off) = (w(4)?, w(5)?, w(6)?, w(7)?);
        let num_sig = u32_at(f, num_sig_ptr as usize)?;
        if num_sig == 0 || sig_size == 0 || !sig_size.is_multiple_of(num_sig) {
            return Err(GpuError::InvalidArgument(
                "booter: numar sau marime de semnaturi imposibile",
            ));
        }
        let sig_prod = range(f, sig_off, sig_size)?;
        let meta_data = range(f, meta_off, meta_size)?;
        let patch_loc = u32_at(f, loc_ptr as usize)?;
        let patch_sig = u32_at(f, sig_ptr as usize)?;
        let l = lh_off as usize;
        let lw = |i: usize| u32_at(f, l + 4 * i);
        let os_code = HsApp {
            offset: lw(0)?,
            size: lw(1)?,
        };
        let os_data = HsApp {
            offset: lw(2)?,
            size: lw(3)?,
        };
        let num_apps = lw(4)?;
        if num_apps > 16 {
            return Err(GpuError::InvalidArgument("booter: prea multe aplicatii"));
        }
        let mut apps = Vec::new();
        for i in 0..num_apps as usize {
            apps.push(HsApp {
                offset: lw(5 + 2 * i)?,
                size: lw(6 + 2 * i)?,
            });
        }
        let blob = &f[hdr.data()];
        for a in [os_code, os_data].iter().chain(apps.iter()) {
            range(blob, a.offset, a.size)?;
        }
        // The signature is patched into the image: it must fit there.
        let sig_len = (sig_size / num_sig) as usize;
        if (patch_loc as usize)
            .checked_add(sig_len)
            .is_none_or(|e| e > blob.len())
        {
            return Err(GpuError::OutOfBounds {
                offset: u64::from(patch_loc),
                len: sig_len as u64,
                limit: blob.len() as u64,
            });
        }
        if patch_sig >= num_sig {
            return Err(GpuError::InvalidArgument(
                "booter: indexul semnaturii in afara listei",
            ));
        }
        Ok(HsV2 {
            hdr,
            sig_prod,
            num_sig,
            patch_loc,
            patch_sig,
            meta_data,
            os_code,
            os_data,
            apps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(f: &mut [u8], o: usize, v: u32) {
        f[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }

    /// SIMULARE: a synthetic bootloader file (filler bytes, no real code).
    fn bootloader() -> Vec<u8> {
        let mut f = alloc::vec![0xa5u8; 0x2000];
        for (i, v) in [BIN_MAGIC, 1, 0x2000, 0x100, 0x1000, 0x1000]
            .iter()
            .enumerate()
        {
            put(&mut f, 4 * i, *v);
        }
        let d = 0x100;
        for (field, v) in [
            (offset_of!(RM_RISCV_UCODE_DESC, version), 1),
            (offset_of!(RM_RISCV_UCODE_DESC, appVersion), 0x1234),
            (offset_of!(RM_RISCV_UCODE_DESC, manifestOffset), 0xc00),
            (offset_of!(RM_RISCV_UCODE_DESC, manifestSize), 0x400),
            (offset_of!(RM_RISCV_UCODE_DESC, monitorDataOffset), 0x800),
            (offset_of!(RM_RISCV_UCODE_DESC, monitorDataSize), 0x400),
            (offset_of!(RM_RISCV_UCODE_DESC, monitorCodeOffset), 0),
            (offset_of!(RM_RISCV_UCODE_DESC, monitorCodeSize), 0x800),
        ] {
            put(&mut f, d + field, v);
        }
        f
    }

    #[test]
    fn riscv_descriptor_offsets_come_from_the_generated_struct() {
        assert_eq!(size_of::<RM_RISCV_UCODE_DESC>(), 84);
        assert_eq!(offset_of!(RM_RISCV_UCODE_DESC, appVersion), 28);
        assert_eq!(offset_of!(RM_RISCV_UCODE_DESC, monitorCodeOffset), 48);
        let b = RiscvBootloader::parse(&bootloader()).unwrap();
        assert_eq!(b.app_version, 0x1234);
        assert_eq!(
            (b.code_offset, b.data_offset, b.manifest_offset),
            (0, 0x800, 0xc00)
        );
        assert_eq!(b.hdr.data(), 0x1000..0x2000);
    }

    #[test]
    fn broken_bootloader_files_are_refused() {
        let mut f = bootloader();
        f[0] = 0; // magic
        assert!(BinHdr::parse(&f).is_err());
        let mut f = bootloader();
        f.truncate(0x1fff); // data blob cut short
        assert!(BinHdr::parse(&f).is_err());
        let mut f = bootloader();
        put(&mut f, 16, 0x10); // data blob over the outer header
        assert!(BinHdr::parse(&f).is_err());
        let mut f = bootloader();
        put(&mut f, 0x100 + 52, 0x1001); // monitorCodeSize past the blob
        assert!(matches!(
            RiscvBootloader::parse(&f),
            Err(GpuError::OutOfBounds { .. })
        ));
        let mut f = bootloader();
        put(&mut f, 0x100 + 36, 0); // manifestSize 0
        assert!(RiscvBootloader::parse(&f).is_err());
        assert!(RiscvBootloader::parse(&f[..10]).is_err());
    }

    #[test]
    fn files_with_the_shape_of_the_official_570_144_ones_are_read() {
        // C-39: the three files our first parser refused on the workstation.
        use gpu_sim::fw_stick::{booter_570_144, bootloader_570_144};
        let f = bootloader_570_144();
        let b = RiscvBootloader::parse(&f).unwrap();
        assert_eq!(b.hdr.bin_size, 0x6100);
        assert_ne!(b.hdr.bin_size as usize, f.len());
        assert_eq!(b.hdr.data(), 0x6c..24_684);
        assert_eq!(b.version, 5);
        assert_eq!(
            (
                b.code_offset,
                b.data_offset,
                b.manifest_offset,
                b.manifest_size
            ),
            (0x1800, 0x800, 0, 0x800)
        );
        for (load, len, data, loc, app) in [
            (true, 61_304, 0xec00, 0x8a10, (0x100, 0x8900)),
            (false, 41_080, 0x9d00, 0x5010, (0x100, 0x4f00)),
        ] {
            let f = booter_570_144(load);
            assert_eq!(f.len(), len);
            let h = HsV2::parse(&f).unwrap();
            assert_eq!(h.hdr.data(), 0x378..len);
            assert_eq!(h.hdr.data_size, data);
            assert_eq!((h.num_sig, h.sig_size()), (2, 384));
            assert_eq!(h.sig_prod, 0x3c..0x33c);
            assert_eq!((h.patch_loc, h.patch_sig), (loc, 0));
            assert_eq!(h.meta_data, 0x344..0x350);
            assert_eq!(
                h.apps,
                [HsApp {
                    offset: app.0,
                    size: app.1
                }]
            );
        }
    }

    /// SIMULARE: a synthetic Booter file with 3 signatures of 384 bytes.
    fn booter() -> Vec<u8> {
        let mut f = alloc::vec![0u8; 0x3000];
        for (i, v) in [BIN_MAGIC, 1, 0x3000, 0x40, 0x2000, 0x1000]
            .iter()
            .enumerate()
        {
            put(&mut f, 4 * i, *v);
        }
        // hs header v2 at 0x40
        for (i, v) in [0x400, 3 * 384, 0x200, 0x204, 0x900, 0x40, 0x208, 0x100]
            .iter()
            .enumerate()
        {
            put(&mut f, 0x40 + 4 * i, *v);
        }
        put(&mut f, 0x200, 0x800); // patch location inside the image
        put(&mut f, 0x204, 2); // signature index
        put(&mut f, 0x208, 3); // signature count (word 6 points here)
        // load header v2 at 0x100: os code, os data, 1 app
        for (i, v) in [0, 0x600, 0x600, 0x200, 1, 0x800, 0x400].iter().enumerate() {
            put(&mut f, 0x100 + 4 * i, *v);
        }
        f
    }

    #[test]
    fn booter_hs_v2_headers_parse_and_are_bounded() {
        let b = HsV2::parse(&booter()).unwrap();
        assert_eq!(b.num_sig, 3);
        assert_eq!(b.sig_size(), 384);
        assert_eq!(b.sig_prod, 0x400..0x400 + 3 * 384);
        assert_eq!((b.patch_loc, b.patch_sig), (0x800, 2));
        assert_eq!(
            b.os_code,
            HsApp {
                offset: 0,
                size: 0x600
            }
        );
        assert_eq!(
            b.apps,
            [HsApp {
                offset: 0x800,
                size: 0x400
            }]
        );
        let mut f = booter();
        put(&mut f, 0x204, 3); // signature index past the list
        assert!(HsV2::parse(&f).is_err());
        let mut f = booter();
        put(&mut f, 0x200, 0xf00); // signature would not fit in the image
        assert!(HsV2::parse(&f).is_err());
        let mut f = booter();
        put(&mut f, 0x208, 0); // num_sig 0
        assert!(HsV2::parse(&f).is_err());
        let mut f = booter();
        put(&mut f, 0x40 + 24, 0xffff_fff0); // num_sig pointer outside the file
        assert!(HsV2::parse(&f).is_err());
        let mut f = booter();
        put(&mut f, 0x100 + 16, 1000); // num_apps absurd
        assert!(HsV2::parse(&f).is_err());
    }
}
