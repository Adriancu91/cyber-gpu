//! I3 diagnostic (hub command C-30): list the *structure* of the VBIOS
//! chain through the PROM window of BAR0, read-only, without copying it.
//!
//! Why: on the workstation's RTX 3070 Ti (ASUS, subsystem 1043:8813) the I3
//! parser stopped at 0x24600 ("expected 0xaa55, read 0x4e56"). This walk
//! never stops at the first surprise: for every image it reports the offset,
//! the ROM signature, the PCIR fields and the NPDE fields; where the chain
//! breaks it reports the first 16 bytes there and looks forward in 512-byte
//! steps for the next image; after the last image it reports any further
//! signatures in the window. Image contents (NVIDIA/ASUS firmware) are never
//! copied: besides the header fields, only 16 bytes per break and 16 bytes at
//! an image whose signature is not 0xAA55 are shown.
//!
//! Formats, from public descriptions (same as [`crate::vbios`]):
//! - ROM signatures 0xAA55 (PCI Firmware Specification 3.0), 0xBB77 and
//!   0x4E56 ("NV") — accepted by nouveau `nvkm/subdev/bios/image.c` (MIT);
//!   nova-core `vbios.rs` (GPL-2.0, read as specification only) accepts
//!   0xAA55 and 0x4E56;
//! - PCIR structure signatures "PCIR", "RGIS", "NPDS" — nouveau
//!   `bios/pcir.c` (MIT); fields at +4 vendor, +6 device, +0x0a length,
//!   +0x0c revision, +0x0d class code, +0x10 length in 512-byte blocks,
//!   +0x12 image revision, +0x14 code type, +0x15 indicator;
//! - NPDE after PCIR at the next 16-byte boundary: +4 revision, +6 length,
//!   +8 sub-image blocks, +0x0a last-image flag — nouveau `bios/npde.c`.
//!
//! Nothing here writes: all reads go through `ReadOnlyBar0`, which has no
//! write path. The number of PROM reads is bounded by the reader of
//! [`crate::vbios`] (budget and read cap).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, Mmio};

use crate::regs::{INIT_WINDOW, PROM_SIZE};
use crate::vbios::{
    BIT_TOKEN_FALCON_DATA, BitToken, CODE_FWSEC, CODE_PCI_AT, Rom, RomImage, code_name, find_bit,
    in_window,
};

/// Signatures that start a ROM image (see the module documentation).
pub const ROM_SIGNATURES: [u16; 3] = [0xaa55, 0xbb77, 0x4e56];
/// Signatures of the PCI data structure: "PCIR", "RGIS", "NPDS".
pub const PCIR_SIGNATURES: [u32; 3] = [0x5249_4350, 0x5349_4752, 0x5344_504e];
const SIG_NVGI: u32 = 0x4947_564e;
const SIG_RFRD: u32 = 0x4452_4652;
const SIG_NPDE: u32 = 0x4544_504e;

/// Bounds that keep the listing to a few dozen lines.
pub const DIAG_MAX_IMAGES: usize = 16;
pub const DIAG_MAX_BREAKS: usize = 4;
pub const DIAG_MAX_CANDIDATES: usize = 16;
/// Step of the forward search (PCI ROM images start on 512-byte boundaries).
pub const DIAG_SCAN_STEP: u32 = 512;

pub fn signature_name(sig: u16) -> &'static str {
    match sig {
        0xaa55 => "PCI",
        0xbb77 => "BB77",
        0x4e56 => "NV",
        _ => "necunoscuta",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiagIfr {
    pub version: u8,
    pub fixed_size: u16,
    pub total_size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiagPcir {
    /// Offset of the structure in the PROM window.
    pub at: u32,
    pub signature: u32,
    pub vendor: u16,
    pub device: u16,
    pub length: u16,
    pub revision: u8,
    pub class_code: [u8; 3],
    pub blocks: u16,
    pub image_revision: u16,
    pub code_type: u8,
    pub indicator: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiagNpde {
    pub at: u32,
    pub revision: u16,
    pub length: u16,
    pub blocks: u16,
    pub last_flag: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiagImage {
    pub offset: u32,
    pub signature: u16,
    /// The PCIR pointer at image +0x18.
    pub pcir_pointer: u16,
    pub pcir: DiagPcir,
    pub npde: Option<DiagNpde>,
    /// Size used to reach the next image (NPDE sub-image length if non-zero,
    /// else the PCIR length) and the last-image decision (NPDE wins).
    pub size: u32,
    pub last: bool,
    /// First 16 bytes, only for an image whose signature is not 0xAA55.
    pub head: Option<[u8; 16]>,
}

impl DiagImage {
    pub fn rom_image(&self) -> RomImage {
        RomImage {
            offset: self.offset,
            size: self.size,
            code_type: self.pcir.code_type,
            vendor: self.pcir.vendor,
            device: self.pcir.device,
            last: self.last,
            has_npde: self.npde.is_some(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BreakReason {
    UnknownSignature(u16),
    BadPcir {
        at: u32,
        signature: u32,
    },
    ZeroSize,
    OutsideWindow {
        offset: u64,
    },
    TooManyImages,
    /// The IFR ("NVGI") header is there but does not lead to the images.
    NoRomStart,
    Read(GpuError),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiagBreak {
    pub offset: u32,
    pub reason: BreakReason,
    /// First 16 bytes at `offset` (None if they could not be read).
    pub bytes: Option<[u8; 16]>,
    /// Where the walk resumed after the forward search, if it did.
    pub resumed_at: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiagCandidate {
    pub offset: u32,
    pub signature: u16,
    /// The PCIR pointer leads to a known PCIR signature.
    pub pcir_ok: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VbiosDiag {
    /// First dword of the PROM window.
    pub head: u32,
    pub ifr: Option<DiagIfr>,
    /// Where the walk started (None: no VBIOS signature found at all).
    pub rom_start: Option<u32>,
    pub images: Vec<DiagImage>,
    pub breaks: Vec<DiagBreak>,
    /// Signatures found after the last image, up to the end of the window.
    pub after_last: Vec<DiagCandidate>,
    /// Offset where the search after the last image started, if it ran.
    pub after_last_from: Option<u32>,
    pub bit: Option<(u32, Vec<BitToken>)>,
    pub bit_error: Option<GpuError>,
    pub prom_reads: u32,
    /// Writes recorded in the journal during the diagnostic (always 0: the
    /// reader has no write path; reported so the screen can show it).
    pub writes: usize,
}

impl VbiosDiag {
    /// The I3 criterion on this listing: a clean chain that starts with a
    /// PCI-AT image whose BIT table has the Falcon data token, plus at least
    /// one FWSEC image.
    pub fn i3_criterion(&self) -> bool {
        self.breaks.is_empty()
            && self.images.first().map(|i| i.pcir.code_type) == Some(CODE_PCI_AT)
            && self.images.iter().any(|i| i.pcir.code_type == CODE_FWSEC)
            && self
                .bit
                .as_ref()
                .is_some_and(|(_, t)| t.iter().any(|t| t.id == BIT_TOKEN_FALCON_DATA))
    }

    /// The listing as text lines (ASCII, Romanian without diacritics, for
    /// the OS screen and report).
    pub fn lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.push(format!(
            "vbios diagnostic (doar citire): primul dword 0x{:08x}",
            self.head
        ));
        match (&self.ifr, self.rom_start) {
            (Some(i), Some(s)) => out.push(format!(
                "antet NVGI versiunea {}, marime fixa 0x{:x}, marime totala 0x{:x}; imaginile PCI incep la 0x{:x}",
                i.version, i.fixed_size, i.total_size, s
            )),
            (Some(i), None) => out.push(format!(
                "antet NVGI versiunea {}, marime fixa 0x{:x}, marime totala 0x{:x}; inceputul imaginilor PCI negasit",
                i.version, i.fixed_size, i.total_size
            )),
            (None, Some(s)) => out.push(format!(
                "fara antet NVGI; imaginile PCI incep la 0x{:x}",
                s
            )),
            (None, None) => out.push(String::from(
                "nicio semnatura VBIOS la inceputul ferestrei PROM",
            )),
        }
        for (k, img) in self.images.iter().enumerate() {
            let p = &img.pcir;
            out.push(format!(
                "imagine {} la 0x{:05x}: semnatura 0x{:04x} ({}), PCIR la +0x{:x}",
                k,
                img.offset,
                img.signature,
                signature_name(img.signature),
                img.pcir_pointer
            ));
            out.push(format!(
                "  PCIR \"{}\" rev {}: {:04x}:{:04x} clasa {:02x}{:02x}{:02x}, tip cod 0x{:02x} ({}), lungime 0x{:x}, {} blocuri ({} octeti), revizia imaginii 0x{:x}, indicator 0x{:02x} (ultima: {})",
                ascii4(p.signature),
                p.revision,
                p.vendor,
                p.device,
                p.class_code[2],
                p.class_code[1],
                p.class_code[0],
                p.code_type,
                code_name(p.code_type),
                p.length,
                p.blocks,
                p.blocks as u32 * 512,
                p.image_revision,
                p.indicator,
                yes_no(p.indicator & 0x80 != 0)
            ));
            match &img.npde {
                Some(n) => out.push(format!(
                    "  NPDE la 0x{:05x}: rev 0x{:04x}, lungime 0x{:x}, {} blocuri ({} octeti), indicator 0x{:02x} (ultima: {})",
                    n.at,
                    n.revision,
                    n.length,
                    n.blocks,
                    n.blocks as u32 * 512,
                    n.last_flag,
                    yes_no(n.last_flag & 0x80 != 0)
                )),
                None => out.push(String::from("  fara NPDE")),
            }
            if let Some(h) = &img.head {
                out.push(format!("  primii 16 octeti: {}", hex16(h)));
            }
            if img.last {
                out.push(format!(
                    "  -> marime folosita 0x{:x}; ultima imagine (se termina la 0x{:x})",
                    img.size,
                    img.offset as u64 + img.size as u64
                ));
            } else {
                out.push(format!(
                    "  -> marime folosita 0x{:x}; urmatoarea la 0x{:x}",
                    img.size,
                    img.offset as u64 + img.size as u64
                ));
            }
        }
        for b in &self.breaks {
            out.push(format!(
                "RUPTURA la 0x{:05x}: {}",
                b.offset,
                reason_text(&b.reason)
            ));
            match &b.bytes {
                Some(x) => out.push(format!("  primii 16 octeti: {}", hex16(x))),
                None => out.push(String::from("  primii 16 octeti: necititi")),
            }
            match b.resumed_at {
                Some(r) => out.push(format!(
                    "  reluat la 0x{:05x} (cautare inainte, pas {} B)",
                    r, DIAG_SCAN_STEP
                )),
                None => out.push(String::from("  nereluat")),
            }
        }
        if let Some(from) = self.after_last_from {
            if self.after_last.is_empty() {
                out.push(format!(
                    "dupa ultima imagine (0x{:x}..0x{:x}, pas {} B): nicio alta semnatura",
                    from, PROM_SIZE, DIAG_SCAN_STEP
                ));
            } else {
                out.push(format!(
                    "dupa ultima imagine (0x{:x}..0x{:x}, pas {} B): {} semnaturi",
                    from,
                    PROM_SIZE,
                    DIAG_SCAN_STEP,
                    self.after_last.len()
                ));
                for c in &self.after_last {
                    out.push(format!(
                        "  0x{:05x}: semnatura 0x{:04x} ({}), PCIR valid: {}",
                        c.offset,
                        c.signature,
                        signature_name(c.signature),
                        yes_no(c.pcir_ok)
                    ));
                }
            }
        }
        match (&self.bit, &self.bit_error) {
            (Some((at, tokens)), _) => {
                let mut s = format!("tabela BIT la 0x{:05x}: {} jetoane:", at, tokens.len());
                for t in tokens {
                    s.push_str(&format!(" 0x{:02x}v{}", t.id, t.version));
                }
                out.push(s);
            }
            (None, Some(e)) => out.push(format!("tabela BIT: eroare la citire: {}", e)),
            (None, None) => out.push(String::from(
                "tabela BIT: negasita (sau prima imagine nu e PCI-AT)",
            )),
        }
        let types: Vec<&str> = self
            .images
            .iter()
            .map(|i| code_name(i.pcir.code_type))
            .collect();
        out.push(format!(
            "rezumat: {} imagini [{}], {} rupturi, criteriul I3: {}, {} citiri PROM, {} scrieri",
            self.images.len(),
            types.join(", "),
            self.breaks.len(),
            if self.i3_criterion() {
                "indeplinit"
            } else {
                "neindeplinit"
            },
            self.prom_reads,
            self.writes
        ));
        out
    }
}

fn yes_no(b: bool) -> &'static str {
    if b { "da" } else { "nu" }
}

fn ascii4(v: u32) -> String {
    v.to_le_bytes()
        .iter()
        .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
        .collect()
}

fn hex16(b: &[u8; 16]) -> String {
    let mut s = String::new();
    for (k, x) in b.iter().enumerate() {
        if k > 0 {
            s.push(' ');
        }
        s.push_str(&format!("{:02x}", x));
    }
    s
}

fn reason_text(r: &BreakReason) -> String {
    match r {
        BreakReason::UnknownSignature(s) => format!(
            "semnatura de imagine necunoscuta 0x{:04x} (cunoscute: aa55, bb77, 4e56)",
            s
        ),
        BreakReason::BadPcir { at, signature } => format!(
            "structura PCIR lipsa: la 0x{:05x} e 0x{:08x} (\"{}\")",
            at,
            signature,
            ascii4(*signature)
        ),
        BreakReason::ZeroSize => String::from("imagine de marime 0"),
        BreakReason::OutsideWindow { offset } => {
            format!("offset 0x{:x} in afara ferestrei PROM", offset)
        }
        BreakReason::TooManyImages => format!("mai mult de {} imagini", DIAG_MAX_IMAGES),
        BreakReason::NoRomStart => String::from("antetul NVGI nu duce la inceputul imaginilor PCI"),
        BreakReason::Read(e) => format!("eroare la citire: {}", e),
    }
}

/// Read one image header at `off` with the rules shared by the I3 parser:
/// known ROM signature, PCIR through the pointer at +0x18, NPDE overriding
/// size (if non-zero) and the last-image flag.
pub(crate) fn read_image<M: Mmio + ?Sized, K: Clock + ?Sized>(
    rom: &mut Rom<'_, M, K>,
    off: u32,
) -> Result<DiagImage, BreakReason> {
    let rd = BreakReason::Read;
    let signature = rom.u16(off).map_err(rd)?;
    if !ROM_SIGNATURES.contains(&signature) {
        return Err(BreakReason::UnknownSignature(signature));
    }
    let pcir_pointer = rom.u16(off + 0x18).map_err(rd)?;
    let at64 = off as u64 + pcir_pointer as u64;
    let at = in_window(at64).map_err(|_| BreakReason::OutsideWindow { offset: at64 })?;
    let psig = rom.u32(at).map_err(rd)?;
    if !PCIR_SIGNATURES.contains(&psig) {
        return Err(BreakReason::BadPcir {
            at,
            signature: psig,
        });
    }
    let pcir = DiagPcir {
        at,
        signature: psig,
        vendor: rom.u16(at + 4).map_err(rd)?,
        device: rom.u16(at + 6).map_err(rd)?,
        length: rom.u16(at + 0x0a).map_err(rd)?,
        revision: rom.u8(at + 0x0c).map_err(rd)?,
        class_code: [
            rom.u8(at + 0x0d).map_err(rd)?,
            rom.u8(at + 0x0e).map_err(rd)?,
            rom.u8(at + 0x0f).map_err(rd)?,
        ],
        blocks: rom.u16(at + 0x10).map_err(rd)?,
        image_revision: rom.u16(at + 0x12).map_err(rd)?,
        code_type: rom.u8(at + 0x14).map_err(rd)?,
        indicator: rom.u8(at + 0x15).map_err(rd)?,
    };
    let q64 = (at as u64 + pcir.length as u64 + 0xf) & !0xf;
    let q = in_window(q64).map_err(|_| BreakReason::OutsideWindow { offset: q64 })?;
    let npde = if rom.u32(q).map_err(rd)? == SIG_NPDE {
        Some(DiagNpde {
            at: q,
            revision: rom.u16(q + 4).map_err(rd)?,
            length: rom.u16(q + 6).map_err(rd)?,
            blocks: rom.u16(q + 8).map_err(rd)?,
            last_flag: rom.u8(q + 0x0a).map_err(rd)?,
        })
    } else {
        None
    };
    let mut size = pcir.blocks as u32 * 512;
    let mut last = pcir.indicator & 0x80 != 0;
    if let Some(n) = &npde {
        if n.blocks != 0 {
            size = n.blocks as u32 * 512;
        }
        last = n.last_flag & 0x80 != 0;
    }
    if size == 0 {
        return Err(BreakReason::ZeroSize);
    }
    let head = if signature != 0xaa55 {
        read16(rom, off)
    } else {
        None
    };
    Ok(DiagImage {
        offset: off,
        signature,
        pcir_pointer,
        pcir,
        npde,
        size,
        last,
        head,
    })
}

fn read16<M: Mmio + ?Sized, K: Clock + ?Sized>(
    rom: &mut Rom<'_, M, K>,
    off: u32,
) -> Option<[u8; 16]> {
    let mut b = [0u8; 16];
    for (k, x) in b.iter_mut().enumerate() {
        *x = rom.u8(off.checked_add(k as u32)?).ok()?;
    }
    Some(b)
}

/// Look for image signatures from `from` (rounded up to the step) to the
/// end of the window. Stops after `max` hits, or at the first hit with a
/// valid PCIR if `first_valid`. A read error ends the search.
fn search<M: Mmio + ?Sized, K: Clock + ?Sized>(
    rom: &mut Rom<'_, M, K>,
    from: u64,
    max: usize,
    first_valid: bool,
) -> Result<Vec<DiagCandidate>, GpuError> {
    let step = DIAG_SCAN_STEP as u64;
    let mut at = from.div_ceil(step) * step;
    let mut hits = Vec::new();
    while at < PROM_SIZE && hits.len() < max {
        let off = at as u32;
        let sig = rom.u16(off)?;
        if ROM_SIGNATURES.contains(&sig) {
            let ptr = rom.u16(off + 0x18)? as u64;
            let pcir_ok = match in_window(at + ptr) {
                Ok(p) if p as u64 + 4 <= PROM_SIZE => PCIR_SIGNATURES.contains(&rom.u32(p)?),
                _ => false,
            };
            hits.push(DiagCandidate {
                offset: off,
                signature: sig,
                pcir_ok,
            });
            if first_valid && pcir_ok {
                break;
            }
        }
        at += step;
    }
    Ok(hits)
}

/// Where the PCI images start, following the IFR header if present (same
/// rules as the I3 parser). Errors name what was found.
fn start<M: Mmio + ?Sized, K: Clock + ?Sized>(
    rom: &mut Rom<'_, M, K>,
    head: u32,
) -> Result<(Option<DiagIfr>, Option<u32>), GpuError> {
    if ROM_SIGNATURES.contains(&(head as u16)) {
        return Ok((None, Some(0)));
    }
    if head != SIG_NVGI {
        return Ok((None, None));
    }
    let fixed1 = rom.u32(4)?;
    let total = rom.u32(8)? & 0xf_ffff;
    let ifr = DiagIfr {
        version: ((fixed1 >> 8) & 0xff) as u8,
        fixed_size: ((fixed1 >> 16) & 0x7fff) as u16,
        total_size: total,
    };
    let s = match ifr.version {
        1 | 2 => rom.u32(in_window(ifr.fixed_size as u64 + 4)?)?,
        3 => {
            let flash_status = rom.u32(in_window(total as u64)?)?;
            let dir = in_window(flash_status as u64 + 4096)?;
            if rom.u32(dir)? != SIG_RFRD {
                return Ok((Some(ifr), None));
            }
            rom.u32(in_window(dir as u64 + 8)?)?
        }
        _ => return Ok((Some(ifr), None)),
    };
    if s & 3 != 0 || s as u64 >= PROM_SIZE {
        return Ok((Some(ifr), None));
    }
    Ok((Some(ifr), Some(s)))
}

/// Run the diagnostic. `bar0` must map at least [`INIT_WINDOW`] bytes.
/// Only an unusable window (too small, board absent, first read failing) is
/// an error; everything else is reported in the listing.
pub fn diagnose<M: Mmio + ?Sized, K: Clock + ?Sized>(
    bar0: &mut M,
    journal: &mut Journal,
    clock: &mut K,
    budget_us: u64,
) -> GpuResult<VbiosDiag> {
    if bar0.len() < INIT_WINDOW {
        return Err(GpuError::InvalidArgument(
            "fereastra BAR0 prea mica pentru I3 (trebuie 4 MiB)",
        ));
    }
    let writes_before = journal.writes();
    let mut rom = Rom::new(bar0, journal, clock, budget_us);
    let r = walk(&mut rom);
    let reads = rom.reads;
    let ok = r.as_ref().is_ok_and(|d| d.i3_criterion());
    journal.push(Event::Step {
        name: "I3: diagnosticul VBIOS",
        ok,
    });
    let writes = journal.writes() - writes_before;
    r.map(|mut d| {
        d.prom_reads = reads;
        d.writes = writes;
        d
    })
}

fn walk<M: Mmio + ?Sized, K: Clock + ?Sized>(rom: &mut Rom<'_, M, K>) -> GpuResult<VbiosDiag> {
    let head = rom.u32(0)?;
    let mut d = VbiosDiag {
        head,
        ifr: None,
        rom_start: None,
        images: Vec::new(),
        breaks: Vec::new(),
        after_last: Vec::new(),
        after_last_from: None,
        bit: None,
        bit_error: None,
        prom_reads: 0,
        writes: 0,
    };
    let mut next = match start(rom, head) {
        Ok((ifr, s)) => {
            d.ifr = ifr;
            d.rom_start = s;
            s
        }
        Err(e) => {
            d.breaks.push(DiagBreak {
                offset: 0,
                reason: BreakReason::Read(e),
                bytes: read16(rom, 0),
                resumed_at: None,
            });
            None
        }
    };
    if next.is_none() && d.breaks.is_empty() {
        // No signature at the start: report the bytes there and search.
        d.breaks.push(DiagBreak {
            offset: 0,
            reason: if d.ifr.is_some() {
                BreakReason::NoRomStart
            } else {
                BreakReason::UnknownSignature(head as u16)
            },
            bytes: read16(rom, 0),
            resumed_at: None,
        });
        next = resume(rom, &mut d, 0);
    } else if next.is_none() {
        next = resume(rom, &mut d, 0);
    }
    while let Some(off) = next {
        if d.images.len() == DIAG_MAX_IMAGES {
            d.breaks.push(DiagBreak {
                offset: off,
                reason: BreakReason::TooManyImages,
                bytes: None,
                resumed_at: None,
            });
            break;
        }
        match read_image(rom, off) {
            Ok(img) => {
                d.images.push(img);
                let end = off as u64 + img.size as u64;
                if img.last {
                    d.after_last_from = Some(end.min(PROM_SIZE) as u32);
                    match search(rom, end, DIAG_MAX_CANDIDATES, false) {
                        Ok(c) => d.after_last = c,
                        Err(e) => d.breaks.push(DiagBreak {
                            offset: end.min(PROM_SIZE) as u32,
                            reason: BreakReason::Read(e),
                            bytes: None,
                            resumed_at: None,
                        }),
                    }
                    next = None;
                } else if end >= PROM_SIZE {
                    d.breaks.push(DiagBreak {
                        offset: off,
                        reason: BreakReason::OutsideWindow { offset: end },
                        bytes: None,
                        resumed_at: None,
                    });
                    next = None;
                } else {
                    next = Some(end as u32);
                }
            }
            Err(reason) => {
                let bytes = match reason {
                    BreakReason::Read(_) => None,
                    _ => read16(rom, off),
                };
                d.breaks.push(DiagBreak {
                    offset: off,
                    reason,
                    bytes,
                    resumed_at: None,
                });
                next = if d.breaks.len() < DIAG_MAX_BREAKS {
                    resume(rom, &mut d, off as u64 + 1)
                } else {
                    None
                };
            }
        }
    }
    if let Some(first) = d.images.first().copied()
        && first.pcir.code_type == CODE_PCI_AT
    {
        match find_bit(rom, &first.rom_image()) {
            Ok(b) => d.bit = b,
            Err(e) => d.bit_error = Some(e),
        }
    }
    Ok(d)
}

/// Search forward for the next image with a valid PCIR; note it on the
/// last break. Read errors end the walk (recorded on that break).
fn resume<M: Mmio + ?Sized, K: Clock + ?Sized>(
    rom: &mut Rom<'_, M, K>,
    d: &mut VbiosDiag,
    from: u64,
) -> Option<u32> {
    let found = match search(rom, from, DIAG_MAX_CANDIDATES, true) {
        Ok(hits) => hits.into_iter().find(|c| c.pcir_ok).map(|c| c.offset),
        Err(_) => None,
    };
    if let Some(b) = d.breaks.last_mut() {
        b.resumed_at = found;
    }
    found
}

#[cfg(test)]
mod tests {
    //! All tests: SIMULARE, on synthetic ROM bytes (gpu-sim::rom). They prove
    //! the listing follows our reading of the public formats, nothing more.
    use super::*;
    use crate::regs::PROM_BASE;
    use crate::vbios::{CODE_EFI, VBIOS_TIMEOUT_US};
    use gpu_sim::clock::SimClock;
    use gpu_sim::mmio::SimMmio;
    use gpu_sim::rom::{self, SimImage};

    fn board(bytes: &[u8]) -> SimMmio {
        let mut m = SimMmio::new(INIT_WINDOW);
        m.load_bytes(PROM_BASE, bytes);
        m
    }

    fn run(m: &mut SimMmio) -> (GpuResult<VbiosDiag>, Journal) {
        let mut j = Journal::new(32);
        let mut c = SimClock::default();
        let r = diagnose(m, &mut j, &mut c, VBIOS_TIMEOUT_US);
        (r, j)
    }

    fn read_only(m: &SimMmio, j: &Journal, d: &VbiosDiag) {
        assert!(m.writes.is_empty());
        assert_eq!(j.writes(), 0);
        assert_eq!(d.writes, 0);
        assert!(
            m.reads
                .iter()
                .all(|&o| (PROM_BASE..PROM_BASE + PROM_SIZE).contains(&o))
        );
        assert_eq!(d.prom_reads as usize, m.reads.len());
    }

    fn text(d: &VbiosDiag) -> String {
        let l = d.lines();
        assert!(l.len() <= 200, "listing stays short: {} lines", l.len());
        assert!(l.iter().all(|s| s.is_ascii()));
        l.join("\n")
    }

    #[test]
    fn clean_chain_is_listed_field_by_field() {
        let mut m = board(&rom::ga10x_like_chain());
        let (r, j) = run(&mut m);
        let d = r.unwrap();
        read_only(&m, &j, &d);
        assert_eq!(d.head & 0xffff, 0xaa55);
        assert_eq!((d.ifr, d.rom_start), (None, Some(0)));
        assert!(d.breaks.is_empty());
        let offs: Vec<u32> = d.images.iter().map(|i| i.offset).collect();
        assert_eq!(offs, [0, 0x10000, 0x18000, 0x1c000]);
        let i0 = d.images[0];
        assert_eq!(i0.signature, 0xaa55);
        assert_eq!(i0.pcir_pointer as usize, rom::SIM_PCIR_AT);
        assert_eq!(i0.pcir.at as usize, rom::SIM_PCIR_AT);
        assert_eq!((i0.pcir.vendor, i0.pcir.device), (0x10de, 0x2482));
        assert_eq!(i0.pcir.length, rom::SIM_PCIR_LEN);
        assert_eq!((i0.pcir.blocks, i0.pcir.code_type), (128, CODE_PCI_AT));
        let n = i0.npde.unwrap();
        assert_eq!(
            (n.at, n.revision, n.length, n.blocks),
            (0x60, 0x0100, 0x0c, 128)
        );
        assert!(i0.head.is_none(), "0xAA55 images: no bytes shown");
        assert!(d.images[3].last && !d.images[2].last);
        assert_eq!(d.after_last_from, Some(0x20000));
        assert!(d.after_last.is_empty());
        assert!(
            d.bit
                .as_ref()
                .is_some_and(|(at, t)| *at == 0x1000 && t.len() == 2)
        );
        assert!(d.i3_criterion());
        let t = text(&d);
        assert!(t.contains("imagine 2 la 0x18000: semnatura 0xaa55 (PCI)"));
        assert!(t.contains("tip cod 0xe0 (FWSEC)"));
        assert!(t.contains("NPDE la 0x00060"));
        assert!(t.contains("nicio alta semnatura"));
        assert!(t.contains("criteriul I3: indeplinit"));
        assert!(t.ends_with(" 0 scrieri"));
        assert!(j.events().any(|e| matches!(
            e,
            Event::Step {
                name: "I3: diagnosticul VBIOS",
                ok: true
            }
        )));
    }

    #[test]
    fn workstation_case_nv_signature_at_0x24600_is_listed_not_a_break() {
        let mut m = board(&rom::asus_like_chain());
        let (r, j) = run(&mut m);
        let d = r.unwrap();
        read_only(&m, &j, &d);
        assert!(d.breaks.is_empty());
        let sigs: Vec<(u32, u16)> = d.images.iter().map(|i| (i.offset, i.signature)).collect();
        assert_eq!(
            sigs,
            [
                (0, 0xaa55),
                (0x12000, 0xaa55),
                (0x24600, 0x4e56),
                (0x28600, 0x4e56)
            ]
        );
        let h = d.images[2].head.unwrap();
        assert_eq!(&h[..2], b"VN", "the bytes the board showed");
        assert!(d.i3_criterion());
        let t = text(&d);
        assert!(t.contains("imagine 2 la 0x24600: semnatura 0x4e56 (NV)"));
        assert!(t.contains("  primii 16 octeti: 56 4e "));
    }

    #[test]
    fn unknown_signature_is_a_break_with_16_bytes_and_the_walk_resumes() {
        let mut bytes = rom::ga10x_like_chain();
        bytes[0x10000..0x10004].copy_from_slice(b"ABCD");
        let mut m = board(&bytes);
        let (r, j) = run(&mut m);
        let d = r.unwrap();
        read_only(&m, &j, &d);
        assert_eq!(d.breaks.len(), 1);
        let b = &d.breaks[0];
        assert_eq!(b.offset, 0x10000);
        assert_eq!(b.reason, BreakReason::UnknownSignature(0x4241));
        assert_eq!(&b.bytes.unwrap()[..4], b"ABCD");
        assert_eq!(b.resumed_at, Some(0x18000));
        let offs: Vec<u32> = d.images.iter().map(|i| i.offset).collect();
        assert_eq!(offs, [0, 0x18000, 0x1c000]);
        assert!(!d.i3_criterion(), "a broken chain never passes I3");
        let t = text(&d);
        assert!(t.contains("RUPTURA la 0x10000: semnatura de imagine necunoscuta 0x4241"));
        assert!(t.contains("  primii 16 octeti: 41 42 43 44"));
        assert!(t.contains("  reluat la 0x18000"));
        assert!(t.contains("criteriul I3: neindeplinit"));
    }

    #[test]
    fn missing_pcir_and_zero_size_are_named() {
        let mut bytes = rom::ga10x_like_chain();
        bytes[0x10000 + rom::SIM_PCIR_AT] = b'X';
        let mut m = board(&bytes);
        let d = run(&mut m).0.unwrap();
        assert_eq!(
            d.breaks[0].reason,
            BreakReason::BadPcir {
                at: 0x10040,
                signature: 0x5249_4358
            }
        );
        let mut bytes = rom::build_chain(&[SimImage {
            code_type: CODE_EFI,
            blocks: 4,
            npde: false,
            bit: None,
        }]);
        bytes[rom::SIM_PCIR_AT + 0x10] = 0;
        let mut m = board(&bytes);
        let d = run(&mut m).0.unwrap();
        assert_eq!(d.breaks[0].reason, BreakReason::ZeroSize);
        assert_eq!(d.breaks[0].resumed_at, None);
        assert!(d.images.is_empty());
    }

    #[test]
    fn signatures_after_the_last_image_are_listed() {
        let mut bytes = rom::ga10x_like_chain();
        // Image 1 claims to be the last one (NPDE flag at NPDE + 0x0a).
        bytes[0x10000 + 0x60 + 0x0a] = 0x80;
        let mut m = board(&bytes);
        let d = run(&mut m).0.unwrap();
        assert_eq!(d.images.len(), 2);
        assert_eq!(d.after_last_from, Some(0x18000));
        let c: Vec<(u32, u16, bool)> = d
            .after_last
            .iter()
            .map(|c| (c.offset, c.signature, c.pcir_ok))
            .collect();
        assert_eq!(c, [(0x18000, 0xaa55, true), (0x1c000, 0xaa55, true)]);
        assert!(!d.i3_criterion(), "no FWSEC inside the chain");
        assert!(
            text(&d).contains("dupa ultima imagine (0x18000..0x100000, pas 512 B): 2 semnaturi")
        );
    }

    #[test]
    fn no_signature_anywhere_is_reported_not_an_error() {
        let mut m = board(&[0x11u8; 64]);
        let (r, j) = run(&mut m);
        let d = r.unwrap();
        read_only(&m, &j, &d);
        assert_eq!(d.rom_start, None);
        assert_eq!(d.breaks.len(), 1);
        assert_eq!(d.breaks[0].resumed_at, None);
        assert!(d.images.is_empty());
        // The whole window is searched once, one dword per 512-byte step.
        assert!(d.prom_reads as u64 <= PROM_SIZE / DIAG_SCAN_STEP as u64 + 8);
        assert!(text(&d).contains("nicio semnatura VBIOS"));
    }

    #[test]
    fn ifr_v3_header_is_shown() {
        let bytes = rom::with_ifr_v3(&rom::ga10x_like_chain());
        let mut m = board(&bytes);
        let d = run(&mut m).0.unwrap();
        assert_eq!(d.ifr.map(|i| i.version), Some(3));
        assert_eq!(d.rom_start, Some(0x200 + 4096 + 0x100));
        assert!(d.i3_criterion());
        assert!(text(&d).contains("antet NVGI versiunea 3"));
        // An IFR header whose directory is missing: reported, not guessed.
        let mut bytes = rom::with_ifr_v3(&rom::ga10x_like_chain());
        bytes[0x200 + 4096] = b'X';
        let mut m = board(&bytes);
        let d = run(&mut m).0.unwrap();
        assert_eq!(d.rom_start, None);
        assert_eq!(d.breaks[0].reason, BreakReason::NoRomStart);
    }

    #[test]
    fn listing_is_bounded_for_a_long_chain() {
        let imgs: Vec<SimImage> = (0..24)
            .map(|_| SimImage {
                code_type: CODE_EFI,
                blocks: 1,
                npde: false,
                bit: None,
            })
            .collect();
        let mut m = board(&rom::build_chain(&imgs));
        let d = run(&mut m).0.unwrap();
        assert_eq!(d.images.len(), DIAG_MAX_IMAGES);
        assert_eq!(d.breaks.last().unwrap().reason, BreakReason::TooManyImages);
        text(&d);
    }

    #[test]
    fn dead_board_and_small_window_are_errors() {
        let mut m = board(&[]);
        m.absent = true;
        assert_eq!(run(&mut m).0, Err(GpuError::NotResponding));
        let mut m = SimMmio::new(0x1000);
        assert!(matches!(run(&mut m).0, Err(GpuError::InvalidArgument(_))));
        assert!(m.reads.is_empty());
    }
}
