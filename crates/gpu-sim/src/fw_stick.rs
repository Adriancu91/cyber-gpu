//! SIMULARE: a synthetic "stick" with stand-ins for the four GSP 570.144
//! files, for host tests and for the OS command `gpu firmware --sim`.
//!
//! **None of these bytes is NVIDIA firmware.** They are filler bytes laid out
//! the way our reading of the public container formats says the real files
//! are laid out (outer `0x10de` header, `RM_RISCV_UCODE_DESC`, "HS v2"
//! headers, an ELF-64 with `.fwimage` and `.fwsignature_ga10x`). That reading
//! is DEDUS (docs/COMPATIBILITATE_FIRMWARE.md rows I4b/I4c) until the real
//! files are checked on the stick at the workstation window.
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
fn outer(f: &mut [u8], hdr_off: u32, data_off: u32, data_size: u32) {
    let len = f.len() as u32;
    for (i, v) in [MAGIC, 1, len, hdr_off, data_off, data_size]
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
    // meta offset/size, num_sig, load header offset, header size.
    for (i, v) in [
        0x400,
        num_sig * 384,
        0x200,
        0x204,
        0x900,
        0x40,
        num_sig,
        0x100,
        0x30,
    ]
    .iter()
    .enumerate()
    {
        put(&mut f, 0x40 + 4 * i, *v);
    }
    put(&mut f, 0x200, 0x800); // where the signature goes in the image
    put(&mut f, 0x204, 0); // which signature
    // Load header v2 at 0x100: os code, os data, 1 app.
    for (i, v) in [0, 0x600, 0x600, 0x200, 1, 0x800, 0x400].iter().enumerate() {
        put(&mut f, 0x100 + 4 * i, *v);
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
    [booter(3, 0x11), booter(2, 0x22), bootloader(), gsp()]
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
        for b in &f[..3] {
            assert_eq!(u32::from_le_bytes([b[0], b[1], b[2], b[3]]), MAGIC);
            assert_eq!(
                u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize,
                b.len()
            );
        }
        assert!(f[3].starts_with(b"\x7fELF"));
        assert_ne!(f[0], f[1]);
        assert!(SIM_PATHS.iter().all(|p| p.starts_with("sim/")));
    }
}
