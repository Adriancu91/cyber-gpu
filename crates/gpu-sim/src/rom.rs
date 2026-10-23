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
