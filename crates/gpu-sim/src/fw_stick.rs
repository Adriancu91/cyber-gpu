//! SIMULARE: a synthetic "stick" with stand-ins for the four GSP 570.144
//! files, for host tests and for the OS command `gpu firmware --sim`.
//!
//! **None of these bytes is NVIDIA firmware.** They are filler bytes laid out
//! the way our reading of the public container formats says the real files
//! are laid out (outer `0x10de` header, `RM_RISCV_UCODE_DESC`, "HS v2"
//! headers, an ELF-64 with `.fwimage` and `.fwsignature_ga10x`). That reading
//! was DEDUS until C-39: on 29 oct the four official files were on the
//! stick (SHA-256 = official) and our parser refused three of them. The
//! container headers of the official files were then read (only the
//! header words: offsets, sizes, counts — no code, data or signatures,
//! docs/PROVENANCE.md) and the `*_570_144` stand-ins below reproduce those
//! header values exactly, around filler bytes.
//!
//! Written independently of the parsers in `nvidia-ga104` (`fw_hdr`,
//! `gsp::elf`), so a test that passes is two readings agreeing, not one
//! reading checking itself.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use core::mem::offset_of;

use nvidia_rm570::structs::RM_RISCV_UCODE_DESC;

/// Paths of the stand-ins, relative to the stick firmware directory, in the
/// order booter_load, booter_unload, bootloader, gsp. They live under `sim/`
/// so they can never be mistaken for the real tree `nvidia/ga102/gsp/`.
pub const SIM_PATHS: [&str; 4] = [
    "sim/booter_load-570.144.bin",
    "sim/booter_unload-570.144.bin",
    "sim/bootloader-570.144.bin",
    "sim/gsp-570.144.bin",
];

/// Licence stand-in name (the real one is `LICENCE.nvidia`, next to the tree).
pub const SIM_LICENCE: &str = "sim/LICENCE.nvidia";

const MAGIC: u32 = 0x10de;

fn put(f: &mut [u8], o: usize, v: u32) {
    f[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

fn put64(f: &mut [u8], o: usize, v: u64) {
    f[o..o + 8].copy_from_slice(&v.to_le_bytes());
}

/// Outer header: magic, version, size, header offset, data offset, data size.
/// `bin_size` is not the file length on the real files (C-39); the small
/// stand-ins write `len + 0x100` so no parser can lean on it.
fn outer(f: &mut [u8], hdr_off: u32, data_off: u32, data_size: u32) {
    let bin_size = f.len() as u32 + 0x100;
    outer_with(f, bin_size, hdr_off, data_off, data_size);
}

fn outer_with(f: &mut [u8], bin_size: u32, hdr_off: u32, data_off: u32, data_size: u32) {
    for (i, v) in [MAGIC, 1, bin_size, hdr_off, data_off, data_size]
        .iter()
        .enumerate()
    {
        put(f, 4 * i, *v);
    }
}

/// Stand-in for `bootloader-570.144.bin`: descriptor at 0x100, data blob
/// 0x1000..0x2000 holding code, data and manifest.
pub fn bootloader() -> Vec<u8> {
    let mut f = vec![0x5au8; 0x2000];
    outer(&mut f, 0x100, 0x1000, 0x1000);
    let d = 0x100;
    for (field, v) in [
        (offset_of!(RM_RISCV_UCODE_DESC, version), 1),
        (offset_of!(RM_RISCV_UCODE_DESC, monitorCodeOffset), 0),
        (offset_of!(RM_RISCV_UCODE_DESC, monitorCodeSize), 0x800),
        (offset_of!(RM_RISCV_UCODE_DESC, monitorDataOffset), 0x800),
        (offset_of!(RM_RISCV_UCODE_DESC, monitorDataSize), 0x400),
        (offset_of!(RM_RISCV_UCODE_DESC, manifestOffset), 0xc00),
        (offset_of!(RM_RISCV_UCODE_DESC, manifestSize), 0x400),
    ] {
        put(&mut f, d + field, v);
    }
    f
}

/// Stand-in for a Booter file ("HS v2"): `num_sig` signatures of 384 bytes,
/// `fill` makes load and unload differ.
pub fn booter(num_sig: u32, fill: u8) -> Vec<u8> {
    let mut f = vec![fill; 0x3000];
    outer(&mut f, 0x40, 0x2000, 0x1000);
    // HS header v2 at 0x40: sig offset/size, patch_loc ptr, patch_sig ptr,
    // meta offset/size, num_sig ptr, load header offset.
    for (i, v) in [
        0x400,
        num_sig * 384,
        0x200,
        0x204,
        0x900,
        0x40,
        0x208,
        0x100,
    ]
    .iter()
    .enumerate()
    {
        put(&mut f, 0x40 + 4 * i, *v);
    }
    put(&mut f, 0x200, 0x800); // where the signature goes in the image
    put(&mut f, 0x204, 0); // which signature
    put(&mut f, 0x208, num_sig); // how many signatures
    // Load header v2 at 0x100: os code, os data, 1 app.
    for (i, v) in [0, 0x600, 0x600, 0x200, 1, 0x800, 0x400].iter().enumerate() {
        put(&mut f, 0x100 + 4 * i, *v);
    }
    f
}

/// The header words of `booter_load-570.144.bin` / `booter_unload-570.144.bin`
/// as read from the official files (C-39): (file length, outer header,
/// patch_loc, load header). Shared by both: the HS v2 header, 2 signatures
/// of 384 B, patch_sig 0, one application.
const BOOTER_LOAD: (usize, [u32; 6], u32, [u32; 7]) = (
    61_304,
    [MAGIC, 1, 0xf000, 0x18, 0x378, 0xec00],
    0x8a10,
    [0, 0x100, 0x8a00, 0x6200, 1, 0x100, 0x8900],
);
const BOOTER_UNLOAD: (usize, [u32; 6], u32, [u32; 7]) = (
    41_080,
    [MAGIC, 1, 0xa100, 0x18, 0x378, 0x9d00],
    0x5010,
    [0, 0x100, 0x5000, 0x4d00, 1, 0x100, 0x4f00],
);
/// HS v2 header of both Booter files (at 0x18): sig offset/size, patch_loc
/// ptr, patch_sig ptr, meta offset/size, num_sig ptr, load header offset.
const BOOTER_HS_V2: [u32; 8] = [0x3c, 0x300, 0x33c, 0x340, 0x344, 0xc, 0x350, 0x354];

/// Stand-in with the exact header shape of the official Booter file
/// (`load = true`: booter_load, else booter_unload). Signatures, meta data,
/// code and data are filler / SIMULARE values (meta: fuse 0, engine 0x4,
/// ucode 2 — NOT read from the real file).
pub fn booter_570_144(load: bool) -> Vec<u8> {
    let (len, out, patch_loc, lh) = if load { BOOTER_LOAD } else { BOOTER_UNLOAD };
    let mut f = vec![if load { 0x11 } else { 0x22 }; len];
    outer_with(&mut f, out[2], out[3], out[4], out[5]);
    for (i, v) in BOOTER_HS_V2.iter().enumerate() {
        put(&mut f, 0x18 + 4 * i, *v);
    }
    for k in 0..2 {
        f[0x3c + k * 384..0x3c + (k + 1) * 384].fill(0xc0 + k as u8);
    }
    put(&mut f, 0x33c, patch_loc);
    put(&mut f, 0x340, 0); // patch_sig
    for (i, v) in [0, 0x4, 2].iter().enumerate() {
        put(&mut f, 0x344 + 4 * i, *v); // meta: SIMULARE values
    }
    put(&mut f, 0x350, 2); // num_sig
    for (i, v) in lh.iter().enumerate() {
        put(&mut f, 0x354 + 4 * i, *v);
    }
    f
}

/// `RM_RISCV_UCODE_DESC` words of `bootloader-570.144.bin` (C-39), in field
/// order (version, bootloader off/size, param off/size, ELF off/size,
/// appVersion, manifest off/size, monitor data off/size, monitor code
/// off/size, monitor enabled, swbrom code off/size, swbrom data off/size,
/// fbReservedSize, signed as code).
const BOOTLOADER_DESC: [u32; 21] = [
    5, 0x5000, 0x880, 0x5880, 0x10, 0, 0, 0, 0, 0x800, 0x800, 0x1000, 0x1800, 0x2900, 1, 0, 0, 0,
    0, 0x6000, 0,
];

/// Stand-in with the exact header shape of the official
/// `bootloader-570.144.bin`: 24 684 B, descriptor at 0x18, data blob
/// 0x6c..end. The blob is filler.
pub fn bootloader_570_144() -> Vec<u8> {
    let mut f = vec![0x5au8; 24_684];
    outer_with(&mut f, 0x6100, 0x18, 0x6c, 0x6000);
    for (i, v) in BOOTLOADER_DESC.iter().enumerate() {
        put(&mut f, 0x18 + 4 * i, *v);
    }
    f
}

/// Minimal ELF-64 LE container with the given sections (plus `.shstrtab`).
pub fn elf(sections: &[(&str, &[u8])]) -> Vec<u8> {
    let mut names = vec![0u8];
    let mut name_at = Vec::new();
    for (n, _) in sections {
        name_at.push(names.len());
        names.extend_from_slice(n.as_bytes());
        names.push(0);
    }
    let shstr_name = names.len();
    names.extend_from_slice(b".shstrtab\0");

    let mut out = vec![0u8; 64];
    out[..4].copy_from_slice(b"\x7fELF");
    out[4] = 2; // ELFCLASS64
    out[5] = 1; // little-endian
    out[6] = 1; // EV_CURRENT
    let mut placed = Vec::new();
    for (_, data) in sections {
        placed.push((out.len(), data.len()));
        out.extend_from_slice(data);
    }
    let names_at = out.len();
    out.extend_from_slice(&names);
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    let shoff = out.len();
    let shnum = sections.len() + 2; // null + sections + shstrtab
    out.resize(shoff + 64 * shnum, 0);
    let mut sh = |i: usize, name: usize, ty: u32, off: usize, size: usize| {
        let b = shoff + 64 * i;
        put(&mut out, b, name as u32);
        put(&mut out, b + 4, ty);
        put64(&mut out, b + 0x18, off as u64);
        put64(&mut out, b + 0x20, size as u64);
    };
    for (i, (off, size)) in placed.iter().enumerate() {
        sh(i + 1, name_at[i], 1, *off, *size); // SHT_PROGBITS
    }
    sh(shnum - 1, shstr_name, 3, names_at, names.len()); // SHT_STRTAB
    put64(&mut out, 0x28, shoff as u64);
    out[0x34] = 64; // e_ehsize
    out[0x3a] = 64; // e_shentsize
    out[0x3c..0x3e].copy_from_slice(&(shnum as u16).to_le_bytes());
    out[0x3e..0x40].copy_from_slice(&((shnum - 1) as u16).to_le_bytes());
    out
}

/// Stand-in for `gsp-570.144.bin`: an ELF with an 8 KiB `.fwimage` and a
/// 384-byte `.fwsignature_ga10x`.
pub fn gsp() -> Vec<u8> {
    let image: Vec<u8> = (0..0x2000u32).map(|i| (i * 7) as u8).collect();
    elf(&[(".fwimage", &image), (".fwsignature_ga10x", &[0x33; 384])])
}

/// The four stand-ins, in [`SIM_PATHS`] order.
pub fn files() -> [Vec<u8>; 4] {
    [
        booter_570_144(true),
        booter_570_144(false),
        bootloader_570_144(),
        gsp(),
    ]
}

/// The licence stand-in text.
pub fn licence() -> Vec<u8> {
    String::from("SIMULARE: nu e licenta NVIDIA, doar un fisier de proba\n").into_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stand_ins_have_the_outer_header_and_differ() {
        let f = files();
        let w = |b: &[u8], o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]);
        for (b, len) in f[..3].iter().zip([61_304, 41_080, 24_684]) {
            assert_eq!(w(b, 0), MAGIC);
            assert_eq!(b.len(), len, "mărimile fișierelor oficiale");
            // As on the real files: bin_size is not the length, the data
            // blob ends exactly at the end of the file.
            assert_ne!(w(b, 8) as usize, b.len());
            assert_eq!((w(b, 16) + w(b, 20)) as usize, b.len());
        }
        for b in [booter(3, 1), bootloader()] {
            assert_ne!(w(&b, 8) as usize, b.len());
        }
        assert!(f[3].starts_with(b"\x7fELF"));
        assert_ne!(f[0], f[1]);
        assert!(SIM_PATHS.iter().all(|p| p.starts_with("sim/")));
    }
}
