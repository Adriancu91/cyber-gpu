//! Named sections of the GSP-RM ELF container (`gsp-570.144.bin`).
//!
//! The driver needs two byte ranges from the container: the image GSP runs
//! (`.fwimage`, described to GSP through radix3) and the signature Booter
//! checks (`.fwsignature_ga10x` on GA10x). Finding them is what every loader
//! does with an ELF file; the bytes inside stay opaque — we never decode or
//! analyse them (licence §2.1.3, C-18).
//!
//! Format: the ELF-64 specification (System V ABI, public). Section names
//! DEDUS from nouveau `gsp/r535.c` and nova-core's description (MIT / GPL,
//! read, not copied). On the real file: NOT TESTED (we do not keep it).

use gpu_core::error::{GpuError, GpuResult};

use super::{get_u16, get_u32, get_u64};

/// Section holding the GSP-RM image (goes into radix3).
pub const FWIMAGE: &str = ".fwimage";
/// Section holding the signature for GA10x (goes into WPR meta).
pub const FWSIGNATURE_GA10X: &str = ".fwsignature_ga10x";

const EHDR_SIZE: usize = 64;
const SHDR_SIZE: usize = 64;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// Returns the bytes of section `name`. Every offset is bounds-checked; a
/// malformed container is an error, never a panic.
pub fn section<'a>(elf: &'a [u8], name: &str) -> GpuResult<&'a [u8]> {
    if elf.len() < EHDR_SIZE || !elf.starts_with(b"\x7fELF") {
        return Err(bad("ELF: antet lipsa"));
    }
    if elf[4] != 2 || elf[5] != 1 {
        return Err(bad("ELF: nu e ELF-64 little-endian"));
    }
    let shoff = get_u64(elf, 0x28) as usize;
    let shentsize = get_u16(elf, 0x3a) as usize;
    let shnum = get_u16(elf, 0x3c) as usize;
    let shstrndx = get_u16(elf, 0x3e) as usize;
    if shentsize != SHDR_SIZE || shnum == 0 || shstrndx >= shnum {
        return Err(bad("ELF: tabela de sectiuni invalida"));
    }
    let table_end = shnum
        .checked_mul(SHDR_SIZE)
        .and_then(|n| n.checked_add(shoff))
        .ok_or(bad("ELF: tabela de sectiuni in afara fisierului"))?;
    if table_end > elf.len() {
        return Err(bad("ELF: tabela de sectiuni in afara fisierului"));
    }
    let shdr = |i: usize| &elf[shoff + i * SHDR_SIZE..shoff + (i + 1) * SHDR_SIZE];
    let range = |h: &[u8]| -> GpuResult<(usize, usize)> {
        let off = get_u64(h, 0x18) as usize;
        let size = get_u64(h, 0x20) as usize;
        let end = off
            .checked_add(size)
            .ok_or(bad("ELF: sectiune in afara fisierului"))?;
        if end > elf.len() {
            return Err(bad("ELF: sectiune in afara fisierului"));
        }
        Ok((off, end))
    };
    let (str_off, str_end) = range(shdr(shstrndx))?;
    let names = &elf[str_off..str_end];
    for i in 0..shnum {
        let h = shdr(i);
        let n = get_u32(h, 0) as usize;
        if n >= names.len() {
            continue;
        }
        let rest = &names[n..];
        let len = rest.iter().position(|&c| c == 0).unwrap_or(rest.len());
        if &rest[..len] == name.as_bytes() {
            // SHT_NOBITS (8) has no bytes in the file.
            if get_u32(h, 4) == 8 {
                return Err(bad("ELF: sectiunea nu are continut in fisier"));
            }
            let (a, b) = range(h)?;
            return Ok(&elf[a..b]);
        }
    }
    Err(bad("ELF: sectiunea cautata lipseste"))
}

/// Test helper: a minimal ELF-64 container with the given sections (synthetic).
#[cfg(test)]
pub(crate) fn synthetic_elf(sections: &[(&str, &[u8])]) -> alloc::vec::Vec<u8> {
    use super::{put_u32, put_u64};
    let mut shstr = alloc::vec![0u8];
    let mut name_off = alloc::vec::Vec::new();
    for (n, _) in sections {
        name_off.push(shstr.len());
        shstr.extend_from_slice(n.as_bytes());
        shstr.push(0);
    }
    let shstr_name = shstr.len();
    shstr.extend_from_slice(b".shstrtab\0");
    let mut out = alloc::vec![0u8; EHDR_SIZE];
    out[..4].copy_from_slice(b"\x7fELF");
    out[4] = 2;
    out[5] = 1;
    out[6] = 1;
    let mut placed = alloc::vec::Vec::new();
    for (_, data) in sections {
        while !out.len().is_multiple_of(8) {
            out.push(0);
        }
        placed.push((out.len(), data.len()));
        out.extend_from_slice(data);
    }
    let shstr_at = out.len();
    out.extend_from_slice(&shstr);
    while !out.len().is_multiple_of(8) {
        out.push(0);
    }
    let shoff = out.len();
    let shnum = sections.len() + 2;
    out.resize(shoff + shnum * SHDR_SIZE, 0);
    for (i, ((off, len), n)) in placed.iter().zip(&name_off).enumerate() {
        let h = shoff + (i + 1) * SHDR_SIZE;
        put_u32(&mut out, h, *n as u32);
        put_u32(&mut out, h + 4, 1); // PROGBITS
        put_u64(&mut out, h + 0x18, *off as u64);
        put_u64(&mut out, h + 0x20, *len as u64);
    }
    let h = shoff + (shnum - 1) * SHDR_SIZE;
    put_u32(&mut out, h, shstr_name as u32);
    put_u32(&mut out, h + 4, 3); // STRTAB
    put_u64(&mut out, h + 0x18, shstr_at as u64);
    put_u64(&mut out, h + 0x20, shstr.len() as u64);
    put_u64(&mut out, 0x28, shoff as u64);
    out[0x3a..0x3c].copy_from_slice(&(SHDR_SIZE as u16).to_le_bytes());
    out[0x3c..0x3e].copy_from_slice(&(shnum as u16).to_le_bytes());
    out[0x3e..0x40].copy_from_slice(&((shnum - 1) as u16).to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_sections_in_a_synthetic_container() {
        let img = [7u8; 300];
        let sig = [9u8; 96];
        let e = synthetic_elf(&[(FWIMAGE, &img), (FWSIGNATURE_GA10X, &sig)]);
        assert_eq!(section(&e, FWIMAGE).unwrap(), &img[..]);
        assert_eq!(section(&e, FWSIGNATURE_GA10X).unwrap(), &sig[..]);
        assert!(section(&e, ".fwsignature_tu10x").is_err());
    }

    #[test]
    fn malformed_containers_are_errors_not_panics() {
        let e = synthetic_elf(&[(FWIMAGE, &[1u8; 64])]);
        assert!(section(&e[..40], FWIMAGE).is_err());
        assert!(section(b"MZ\0\0", FWIMAGE).is_err());
        let mut big = e.clone();
        // Section size past the end of the file.
        let shoff = get_u64(&big, 0x28) as usize;
        super::super::put_u64(&mut big, shoff + SHDR_SIZE + 0x20, u64::MAX / 2);
        assert!(section(&big, FWIMAGE).is_err());
        let mut cut = e.clone();
        cut.truncate(e.len() - 10);
        assert!(section(&cut, FWIMAGE).is_err());
    }
}
