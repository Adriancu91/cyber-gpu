//! SIMULATION of an NVIDIA VBIOS image, for the read-only I3 parser.
//!
//! The layout follows *our reading* of public sources: PCI ROM header and
//! PCIR (PCI Firmware Specification 3.0), the NVIDIA PCI Data Extension
//! (NPDE) and the BIT table as read in nouveau (`nvkm/subdev/bios/`, MIT),
//! the IFR header as described in the Linux nova-core VBIOS documentation
//! (docs.kernel.org/gpu/nova/core/vbios.html). Every byte here is synthetic
//! (`SIM`); none comes from a real VBIOS, and no NVIDIA firmware is included.

use alloc::vec;
use alloc::vec::Vec;

/// A BIT token: `(id, version, length, data offset)`.
pub type SimBitToken = (u8, u8, u16, u16);

/// One synthetic image of the ROM chain.
pub struct SimImage {
    pub code_type: u8,
    /// Size in 512-byte blocks (written to PCIR and, if `npde`, NPDE).
    pub blocks: u16,
    pub npde: bool,
    /// Offset of a BIT table inside this image, with the given tokens
    /// `(id, version, length, data offset)`.
    pub bit: Option<(usize, Vec<SimBitToken>)>,
}

fn put16(b: &mut [u8], at: usize, v: u16) {
    b[at..at + 2].copy_from_slice(&v.to_le_bytes());
}
fn put32(b: &mut [u8], at: usize, v: u32) {
    b[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// PCIR at image offset 0x40, NPDE at the next 16-byte boundary after it.
pub const SIM_PCIR_AT: usize = 0x40;
pub const SIM_PCIR_LEN: u16 = 0x18;

/// Build the ROM image chain (no IFR header).
pub fn build_chain(images: &[SimImage]) -> Vec<u8> {
    let mut out = Vec::new();
    for (n, img) in images.iter().enumerate() {
        let last = n + 1 == images.len();
        let mut b = vec![0u8; img.blocks as usize * 512];
        put16(&mut b, 0, 0xaa55);
        put16(&mut b, 0x18, SIM_PCIR_AT as u16);
        let p = SIM_PCIR_AT;
        b[p..p + 4].copy_from_slice(b"PCIR");
        put16(&mut b, p + 4, 0x10de);
        put16(&mut b, p + 6, 0x2482);
        put16(&mut b, p + 0x0a, SIM_PCIR_LEN);
        put16(&mut b, p + 0x10, img.blocks);
        b[p + 0x14] = img.code_type;
        // With an NPDE, the PCIR "last" flag is left clear on purpose: the
        // parser must take the NPDE value (as nouveau does).
        b[p + 0x15] = if last && !img.npde { 0x80 } else { 0 };
        if img.npde {
            let q = (p + SIM_PCIR_LEN as usize + 0xf) & !0xf;
            b[q..q + 4].copy_from_slice(b"NPDE");
            put16(&mut b, q + 4, 0x0100);
            put16(&mut b, q + 6, 0x0c);
            put16(&mut b, q + 8, img.blocks);
            b[q + 0x0a] = if last { 0x80 } else { 0 };
        }
        if let Some((at, tokens)) = &img.bit {
            let at = *at;
            b[at..at + 6].copy_from_slice(b"\xff\xb8BIT\0");
            put16(&mut b, at + 6, 0x0100);
            b[at + 8] = 12; // header size
            b[at + 9] = 6; // token size
            b[at + 10] = tokens.len() as u8;
            for (k, &(id, ver, len, off)) in tokens.iter().enumerate() {
                let e = at + 12 + 6 * k;
                b[e] = id;
                b[e + 1] = ver;
                put16(&mut b, e + 2, len);
                put16(&mut b, e + 4, off);
            }
        }
        out.extend_from_slice(&b);
    }
    out
}

/// The chain we expect on an Ampere board: PCI-AT (with BIT), EFI, two
/// FWSEC images (nova-core documentation). Sizes are SIM.
pub fn ga10x_like_chain() -> Vec<u8> {
    build_chain(&[
        SimImage {
            code_type: 0x00,
            blocks: 128,
            npde: true,
            bit: Some((0x1000, vec![(0x42, 1, 0x20, 0x2000), (0x70, 1, 4, 0x3000)])),
        },
        SimImage {
            code_type: 0x03,
            blocks: 64,
            npde: true,
            bit: None,
        },
        SimImage {
            code_type: 0xe0,
            blocks: 32,
            npde: true,
            bit: None,
        },
        SimImage {
            code_type: 0xe0,
            blocks: 32,
            npde: true,
            bit: None,
        },
    ])
}

/// Prefix `chain` with an IFR header ("NVGI"), version 3, whose ROM
/// directory ("RFRD") points at the chain. Field placement per the nova-core
/// VBIOS documentation; values are SIM.
pub fn with_ifr_v3(chain: &[u8]) -> Vec<u8> {
    let fixed_size: u32 = 0x40;
    let total: u32 = 0x100;
    let flash_status: u32 = 0x200;
    let dir = (flash_status + 4096) as usize;
    let rom_start = (dir + 0x100) as u32;
    let mut b = vec![0u8; rom_start as usize];
    b[0..4].copy_from_slice(b"NVGI");
    put32(&mut b, 4, (fixed_size << 16) | (3 << 8));
    put32(&mut b, 8, total);
    put32(&mut b, total as usize, flash_status);
    b[dir..dir + 4].copy_from_slice(b"RFRD");
    put32(&mut b, dir + 8, rom_start);
    b.extend_from_slice(chain);
    b
}

/// Where the synthetic FWSEC pieces go, as offsets into the FWSEC images
/// taken back to back (the address space the VBIOS pointers use after the
/// PCI-AT size is subtracted). All values are SIM.
pub struct SimFwsec {
    /// Offset of the Falcon data u32 inside the PCI-AT image (the BIT token
    /// 0x70 data offset).
    pub falcon_data_at: usize,
    pub pmu_table_rel: u32,
    /// PMU entries `(application id, target id, descriptor offset in the
    /// FWSEC section)`.
    pub entries: Vec<(u8, u8, u32)>,
    pub desc_rel: u32,
    pub desc_version: u8,
    pub signature_count: u8,
    pub signature_versions: u16,
    pub imem_load_size: u32,
    pub dmem_load_size: u32,
    /// Extra bytes stored after IMEM + DMEM.
    pub stored_extra: u32,
}

impl SimFwsec {
    pub fn ga10x_like() -> Self {
        SimFwsec {
            falcon_data_at: 0x3000,
            pmu_table_rel: 0x200,
            entries: vec![(0x05, 0, 0x100), (0x85, 0, 0x400)],
            desc_rel: 0x400,
            desc_version: 3,
            signature_count: 2,
            signature_versions: 0b0110,
            // The ucode runs from the first FWSEC image into the second.
            imem_load_size: 0x3000,
            dmem_load_size: 0x1000,
            stored_extra: 0,
        }
    }
}

/// Write `f` into `chain` (built by [`build_chain`]), whose first image is
/// PCI-AT of `pci_at_size` bytes; `fwsec` lists the FWSEC images as
/// `(offset in chain, size)`. The ucode bytes stay zero: no firmware here.
pub fn put_fwsec(chain: &mut [u8], pci_at_size: u32, fwsec: &[(usize, u32)], f: &SimFwsec) {
    let at = |rel: u32| -> usize {
        let mut r = rel;
        for &(off, size) in fwsec {
            if r < size {
                return off + r as usize;
            }
            r -= size;
        }
        panic!("SIM: offset outside the FWSEC images");
    };
    let mut put8 = |chain: &mut [u8], rel: u32, v: u8| chain[at(rel)] = v;
    let put =
        |chain: &mut [u8], rel: u32, v: u32, n: u32, put8: &mut dyn FnMut(&mut [u8], u32, u8)| {
            for k in 0..n {
                put8(chain, rel + k, (v >> (8 * k)) as u8);
            }
        };
    put32(chain, f.falcon_data_at, pci_at_size + f.pmu_table_rel);
    let t = f.pmu_table_rel;
    put(chain, t, 1, 1, &mut put8); // version
    put(chain, t + 1, 6, 1, &mut put8); // header length
    put(chain, t + 2, 6, 1, &mut put8); // entry length
    put(chain, t + 3, f.entries.len() as u32, 1, &mut put8);
    for (k, &(app, target, desc)) in f.entries.iter().enumerate() {
        let e = t + 6 + 6 * k as u32;
        put(chain, e, app as u32, 1, &mut put8);
        put(chain, e + 1, target as u32, 1, &mut put8);
        put(chain, e + 2, pci_at_size + desc, 4, &mut put8);
    }
    let d = f.desc_rel;
    let header = 44 + f.signature_count as u32 * 384;
    let stored = f.imem_load_size + f.dmem_load_size + f.stored_extra;
    let words = [
        (header << 16) | ((f.desc_version as u32) << 8) | 1,
        stored,
        0x10, // pkc_data_offset
        0x20, // interface_offset
        0,    // imem_phys_base
        f.imem_load_size,
        0, // imem_virt_base
        0, // dmem_phys_base
        f.dmem_load_size,
    ];
    for (k, &w) in words.iter().enumerate() {
        put(chain, d + 4 * k as u32, w, 4, &mut put8);
    }
    put(chain, d + 36, 0x0400, 2, &mut put8); // engine_id_mask (SIM)
    put(chain, d + 38, 0x0a, 1, &mut put8); // ucode_id (SIM)
    put(chain, d + 39, f.signature_count as u32, 1, &mut put8);
    put(chain, d + 40, f.signature_versions as u32, 2, &mut put8);
}

/// [`ga10x_like_chain`] with the FWSEC pieces of [`SimFwsec::ga10x_like`].
pub fn ga10x_like_chain_with_fwsec(f: &SimFwsec) -> Vec<u8> {
    let mut b = ga10x_like_chain();
    put_fwsec(&mut b, 0x10000, &[(0x18000, 0x4000), (0x1c000, 0x4000)], f);
    b
}
