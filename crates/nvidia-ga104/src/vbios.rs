//! I3 — read the VBIOS layout through the PROM window of BAR0 (read-only).
//!
//! What is parsed, and from which public description:
//! - optional IFR header "NVGI" at ROM offset 0 and, for version 3, the ROM
//!   directory "RFRD" that gives the start of the PCI ROM images — Linux
//!   nova-core VBIOS documentation (docs.kernel.org/gpu/nova/core/vbios.html);
//! - the PCI expansion ROM chain: 0xAA55 signature (also 0xBB77 and 0x4E56
//!   "NV", and the PCIR signatures "NPDS"/"RGIS", as nouveau `bios/image.c`,
//!   `pcir.c` accept them — MIT; the workstation's ASUS RTX 3070 Ti has an
//!   "NV" image at 0x24600, hub C-30), pointer to the PCIR
//!   structure at +0x18, PCIR fields (vendor +4, device +6, length +0x0a,
//!   image length in 512-byte units +0x10, code type +0x14, indicator +0x15,
//!   bit 7 = last image) — PCI Firmware Specification 3.0;
//! - the NVIDIA PCI Data Extension "NPDE" at the 16-byte boundary after
//!   PCIR, sub-image length (512-byte units) at +8 and last-image flag (bit 7)
//!   at +0x0a, which override the PCIR values — DEDUS from nouveau
//!   `nvkm/subdev/bios/npde.c`, `image.c` (MIT); to confirm on the board;
//! - image code types: 0x00 PCI-AT, 0x03 EFI, 0x70 NBSI, 0xE0 FWSEC (nova-core
//!   documentation; GA102 has no 0x70 image per a nouveau fix, Linux 5.10);
//! - the BIT table ("\xff\xb8BIT") inside the PCI-AT image: token size at +9,
//!   token count at +10, tokens from +12, each `id, version, length u16,
//!   offset u16` — DEDUS from nouveau `bios/bit.c` (MIT). Token 0x70 is the
//!   Falcon data that leads to FWSEC (nova-core documentation);
//! - the VBIOS version from BIT token 'i' (0x69, "BIOS info"): 5 bytes at the
//!   token's data offset, printed `major.chip.minor.micro.patch` with the
//!   first four bytes read as a little-endian u32 (byte 3 = major … byte 0 =
//!   micro) and byte 4 = patch — as nouveau `nvkm/subdev/bios/base.c` does
//!   when it prints "version %02x.%02x.%02x.%02x.%02x" (MIT, read as a
//!   specification). This is the string Windows / NVML report as the VBIOS
//!   version (workstation: 94.04.65.00.3e; laptop: 94.06.1E.00.23), so the
//!   first journal from a board can be checked against the Windows report.
//!
//! Locating the FWSEC ucode itself (PMU lookup table, application 0x85,
//! FalconUCodeDescV3) is done by [`crate::fwsec`], with the same reader; its
//! interior is NVIDIA firmware we treat as an opaque blob and never read.
//!
//! Nothing here writes. If the PROM window does not show a VBIOS signature,
//! the result is `Blocked`: nouveau disables ROM shadowing first (a config
//! write), which this step is not allowed to do.

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, Mmio};

use crate::regs::{INIT_WINDOW, PROM_SIZE, ReadOnlyBar0, is_priv_error};
use crate::vbios_diag::{BreakReason, ROM_SIGNATURES};

pub const CODE_PCI_AT: u8 = 0x00;
pub const CODE_EFI: u8 = 0x03;
pub const CODE_NBSI: u8 = 0x70;
pub const CODE_FWSEC: u8 = 0xe0;
pub const BIT_TOKEN_FALCON_DATA: u8 = 0x70;
/// BIT token 'i': BIOS info, carries the VBIOS version (see module doc).
pub const BIT_TOKEN_BIOS_INFO: u8 = 0x69;
/// Bytes of token 'i' needed for a full version (4 version + 1 patch).
pub const BIOS_INFO_MIN_LEN: u16 = 5;

/// Wall-clock budget for the whole I3 scan.
pub const VBIOS_TIMEOUT_US: u64 = 2_000_000;
/// Hard bound on PROM dword reads (the whole 1 MiB window is 262 144).
pub const VBIOS_MAX_READS: u32 = 300_000;
const MAX_IMAGES: usize = 16;
const MAX_BIT_TOKENS: usize = 64;

const SIG_NVGI: u32 = 0x4947_564e; // "NVGI"
const SIG_RFRD: u32 = 0x4452_4652; // "RFRD"
const SIG_PCIR: u32 = 0x5249_4350; // "PCIR"
const SIG_BIT: [u8; 5] = [0xff, 0xb8, b'B', b'I', b'T'];

pub fn code_name(t: u8) -> &'static str {
    match t {
        CODE_PCI_AT => "PCI-AT",
        CODE_EFI => "EFI",
        CODE_NBSI => "NBSI",
        CODE_FWSEC => "FWSEC",
        _ => "necunoscut",
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RomImage {
    /// Offset in the PROM window.
    pub offset: u32,
    pub size: u32,
    pub code_type: u8,
    pub vendor: u16,
    pub device: u16,
    pub last: bool,
    pub has_npde: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BitToken {
    pub id: u8,
    pub version: u8,
    pub length: u16,
    pub data_offset: u16,
}

/// The VBIOS version as NVIDIA tools print it: `94.04.65.00.3e`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BiosVersion {
    pub major: u8,
    pub chip: u8,
    pub minor: u8,
    pub micro: u8,
    pub patch: u8,
}

impl BiosVersion {
    /// The 5 bytes at the data offset of BIT token 'i', in ROM order.
    pub fn from_bytes(b: [u8; 5]) -> Self {
        BiosVersion {
            micro: b[0],
            minor: b[1],
            chip: b[2],
            major: b[3],
            patch: b[4],
        }
    }

    /// True when `reported` (the string Windows / NVML / a BIOS screen
    /// shows, e.g. "94.06.1E.00.23") names this version; case-insensitive,
    /// surrounding spaces ignored. Anything else — including a string in
    /// another shape — is `false`, never a guess.
    pub fn matches_report(&self, reported: &str) -> bool {
        let r = reported.trim();
        let mine = alloc::format!("{self}");
        r.len() == mine.len()
            && r.bytes()
                .zip(mine.bytes())
                .all(|(a, b)| a.to_ascii_lowercase() == b)
    }
}

impl core::fmt::Display for BiosVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{:02x}.{:02x}.{:02x}.{:02x}.{:02x}",
            self.major, self.chip, self.minor, self.micro, self.patch
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VbiosLayout {
    /// IFR version if the ROM starts with an "NVGI" header.
    pub ifr_version: Option<u8>,
    /// Offset of the first PCI ROM image in the PROM window.
    pub rom_start: u32,
    pub images: Vec<RomImage>,
    /// Offset of the BIT table (PROM window) and its tokens.
    pub bit: Option<(u32, Vec<BitToken>)>,
    /// The VBIOS version from BIT token 'i', when the token is there with
    /// at least [`BIOS_INFO_MIN_LEN`] bytes inside the PCI-AT image.
    /// `None` is "not read", never an invented value.
    pub version: Option<BiosVersion>,
    pub prom_reads: u32,
}

impl VbiosLayout {
    pub fn fwsec_images(&self) -> usize {
        self.images
            .iter()
            .filter(|i| i.code_type == CODE_FWSEC)
            .count()
    }
    pub fn has_falcon_data(&self) -> bool {
        self.bit
            .as_ref()
            .is_some_and(|(_, t)| t.iter().any(|t| t.id == BIT_TOKEN_FALCON_DATA))
    }
    /// The I3 pass criterion: a PCI-AT image with a BIT table that has the
    /// Falcon data token, and at least one FWSEC image.
    pub fn complete(&self) -> bool {
        self.images.first().map(|i| i.code_type) == Some(CODE_PCI_AT)
            && self.fwsec_images() > 0
            && self.has_falcon_data()
    }
}

/// Byte-addressed reader over the PROM window, with a one-dword cache, a
/// read counter and a time budget.
pub(crate) struct Rom<'a, M: Mmio + ?Sized, K: Clock + ?Sized> {
    bar: ReadOnlyBar0<'a, M>,
    clock: &'a mut K,
    start: u64,
    budget_us: u64,
    pub(crate) reads: u32,
    cache: Option<(u32, u32)>,
    /// First dword of the window, for the journal.
    head: Option<u32>,
}

impl<'a, M: Mmio + ?Sized, K: Clock + ?Sized> Rom<'a, M, K> {
    pub(crate) fn new(
        bar0: &'a mut M,
        journal: &'a mut Journal,
        clock: &'a mut K,
        budget_us: u64,
    ) -> Self {
        let start = clock.now_us();
        Rom {
            bar: ReadOnlyBar0::new(bar0, journal),
            clock,
            start,
            budget_us,
            reads: 0,
            cache: None,
            head: None,
        }
    }
    fn dword(&mut self, aligned: u32) -> GpuResult<u32> {
        if let Some((o, v)) = self.cache
            && o == aligned
        {
            return Ok(v);
        }
        if self.reads >= VBIOS_MAX_READS {
            return Err(GpuError::NotPermitted("I3: prea multe citiri PROM"));
        }
        if self.reads.is_multiple_of(256) {
            let waited = self.clock.now_us().saturating_sub(self.start);
            if waited >= self.budget_us {
                return Err(GpuError::Timeout { waited_us: waited });
            }
        }
        self.reads += 1;
        let v = self.bar.read_prom(aligned as u64)?;
        if v == 0xffff_ffff && aligned == 0 {
            return Err(GpuError::NotResponding);
        }
        if is_priv_error(v) {
            return Err(GpuError::HardwareFault { code: v });
        }
        if aligned == 0 && self.head.is_none() {
            self.head = Some(v);
        }
        self.cache = Some((aligned, v));
        Ok(v)
    }
    pub(crate) fn u8(&mut self, off: u32) -> GpuResult<u8> {
        if off as u64 >= PROM_SIZE {
            return Err(GpuError::OutOfBounds {
                offset: off as u64,
                len: 1,
                limit: PROM_SIZE,
            });
        }
        Ok((self.dword(off & !3)? >> ((off & 3) * 8)) as u8)
    }
    pub(crate) fn u16(&mut self, off: u32) -> GpuResult<u16> {
        Ok(self.u8(off)? as u16 | (self.u8(off.wrapping_add(1))? as u16) << 8)
    }
    pub(crate) fn u32(&mut self, off: u32) -> GpuResult<u32> {
        Ok(self.u16(off)? as u32 | (self.u16(off.wrapping_add(2))? as u32) << 16)
    }
}

pub(crate) fn in_window(off: u64) -> GpuResult<u32> {
    if off >= PROM_SIZE {
        return Err(GpuError::OutOfBounds {
            offset: off,
            len: 4,
            limit: PROM_SIZE,
        });
    }
    Ok(off as u32)
}

/// Read the VBIOS layout. `bar0` must map at least [`INIT_WINDOW`] bytes.
pub fn read_layout<M: Mmio + ?Sized, K: Clock + ?Sized>(
    bar0: &mut M,
    journal: &mut Journal,
    clock: &mut K,
    budget_us: u64,
) -> GpuResult<VbiosLayout> {
    if bar0.len() < INIT_WINDOW {
        return Err(GpuError::InvalidArgument(
            "fereastra BAR0 prea mica pentru I3 (trebuie 4 MiB)",
        ));
    }
    let mut rom = Rom::new(bar0, journal, clock, budget_us);
    let r = scan(&mut rom);
    let reads = rom.reads;
    let head = rom.head;
    if let Some(v) = head {
        journal.push(Event::MmioRead {
            bar: 0,
            offset: crate::regs::PROM_BASE,
            value: v,
        });
    }
    journal.push(Event::Step {
        name: "I3: VBIOS prin PROM",
        ok: r.as_ref().is_ok_and(|l| l.complete()),
    });
    r.map(|mut l| {
        l.prom_reads = reads;
        l
    })
}

/// The I3 parser stops at the first image that does not follow the rules
/// (the listing of [`crate::vbios_diag`] goes on); the error names it.
fn strict_error(off: u32, r: BreakReason) -> GpuError {
    match r {
        BreakReason::UnknownSignature(s) => GpuError::VerifyFailed {
            offset: off,
            expected: 0xaa55,
            actual: s as u32,
        },
        BreakReason::BadPcir { at, signature } => GpuError::VerifyFailed {
            offset: at,
            expected: SIG_PCIR,
            actual: signature,
        },
        BreakReason::ZeroSize => GpuError::InvalidArgument("I3: imagine ROM de marime 0"),
        BreakReason::OutsideWindow { offset } => GpuError::OutOfBounds {
            offset,
            len: 4,
            limit: PROM_SIZE,
        },
        BreakReason::TooManyImages => GpuError::NotPermitted("I3: prea multe imagini ROM"),
        BreakReason::NoRomStart => GpuError::Blocked("I3: inceputul imaginilor PCI negasit"),
        BreakReason::Read(e) => e,
    }
}

fn scan<M: Mmio + ?Sized, K: Clock + ?Sized>(rom: &mut Rom<'_, M, K>) -> GpuResult<VbiosLayout> {
    let head = rom.u32(0)?;
    let (ifr_version, rom_start) = if ROM_SIGNATURES.contains(&(head as u16)) {
        (None, 0)
    } else if head == SIG_NVGI {
        let fixed1 = rom.u32(4)?;
        let fixed_size = (fixed1 >> 16) & 0x7fff;
        let version = ((fixed1 >> 8) & 0xff) as u8;
        let start = match version {
            1 | 2 => rom.u32(in_window(fixed_size as u64 + 4)?)?,
            3 => {
                let total = rom.u32(8)? & 0xf_ffff;
                let flash_status = rom.u32(in_window(total as u64)?)?;
                let dir = in_window(flash_status as u64 + 4096)?;
                if rom.u32(dir)? != SIG_RFRD {
                    return Err(GpuError::Blocked(
                        "I3: directorul ROM (RFRD) lipseste din antetul IFR v3",
                    ));
                }
                rom.u32(in_window(dir as u64 + 8)?)?
            }
            _ => return Err(GpuError::Blocked("I3: versiune IFR necunoscuta")),
        };
        if start & 3 != 0 {
            return Err(GpuError::InvalidArgument(
                "I3: inceputul imaginilor PCI nealiniat",
            ));
        }
        (Some(version), in_window(start as u64)?)
    } else {
        return Err(GpuError::Blocked(
            "I3: fereastra PROM nu arata semnatura VBIOS (poate cere oprirea umbririi ROM = scriere, nepermisa)",
        ));
    };

    let mut images = Vec::new();
    let mut off = rom_start;
    loop {
        if images.len() == MAX_IMAGES {
            return Err(GpuError::NotPermitted("I3: prea multe imagini ROM"));
        }
        let img = crate::vbios_diag::read_image(rom, off).map_err(|r| strict_error(off, r))?;
        images.push(img.rom_image());
        if img.last {
            break;
        }
        off = in_window(off as u64 + img.size as u64)?;
    }

    let bit = match images.first() {
        Some(img) if img.code_type == CODE_PCI_AT => find_bit(rom, img)?,
        _ => None,
    };
    let version = match (&bit, images.first()) {
        (Some((_, tokens)), Some(pci_at)) => read_bios_version(rom, pci_at, tokens)?,
        _ => None,
    };
    Ok(VbiosLayout {
        ifr_version,
        rom_start,
        images,
        bit,
        version,
        prom_reads: 0,
    })
}

/// The VBIOS version from BIT token 'i' (5 reads at most, read-only). The
/// token's data offset is relative to the PCI-AT image, like token 0x70's.
/// Missing token, too short a token or data outside the image → `Ok(None)`.
pub(crate) fn read_bios_version<M: Mmio + ?Sized, K: Clock + ?Sized>(
    rom: &mut Rom<'_, M, K>,
    pci_at: &RomImage,
    tokens: &[BitToken],
) -> GpuResult<Option<BiosVersion>> {
    let Some(t) = tokens.iter().find(|t| t.id == BIT_TOKEN_BIOS_INFO) else {
        return Ok(None);
    };
    if t.length < BIOS_INFO_MIN_LEN || t.data_offset as u32 + BIOS_INFO_MIN_LEN as u32 > pci_at.size
    {
        return Ok(None);
    }
    let at = pci_at.offset + t.data_offset as u32;
    let mut b = [0u8; 5];
    for (k, slot) in b.iter_mut().enumerate() {
        *slot = rom.u8(at + k as u32)?;
    }
    Ok(Some(BiosVersion::from_bytes(b)))
}

pub(crate) fn find_bit<M: Mmio + ?Sized, K: Clock + ?Sized>(
    rom: &mut Rom<'_, M, K>,
    img: &RomImage,
) -> GpuResult<Option<(u32, Vec<BitToken>)>> {
    let end = img.offset + img.size;
    let mut at = img.offset;
    while at + SIG_BIT.len() as u32 <= end {
        let mut hit = true;
        for (k, &b) in SIG_BIT.iter().enumerate() {
            if rom.u8(at + k as u32)? != b {
                hit = false;
                break;
            }
        }
        if hit {
            let token_size = rom.u8(at + 9)? as u32;
            let count = (rom.u8(at + 10)? as usize).min(MAX_BIT_TOKENS);
            if token_size < 6 {
                return Err(GpuError::InvalidArgument(
                    "I3: tabela BIT cu jetoane < 6 octeti",
                ));
            }
            let mut tokens = Vec::with_capacity(count);
            for k in 0..count as u32 {
                let e = at + 12 + k * token_size;
                tokens.push(BitToken {
                    id: rom.u8(e)?,
                    version: rom.u8(e + 1)?,
                    length: rom.u16(e + 2)?,
                    data_offset: rom.u16(e + 4)?,
                });
            }
            return Ok(Some((at, tokens)));
        }
        at += 1;
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    //! All tests: SIMULARE, on synthetic ROM bytes (gpu-sim::rom). They prove
    //! the parser follows our reading of the public formats, nothing more.
    use super::*;
    use crate::regs::PROM_BASE;
    use gpu_sim::clock::SimClock;
    use gpu_sim::mmio::SimMmio;
    use gpu_sim::rom::{self, SimImage};

    fn board(bytes: &[u8]) -> SimMmio {
        let mut m = SimMmio::new(INIT_WINDOW);
        m.load_bytes(PROM_BASE, bytes);
        m
    }

    fn run(m: &mut SimMmio) -> (GpuResult<VbiosLayout>, Journal) {
        let mut j = Journal::new(32);
        let mut c = SimClock::default();
        let r = read_layout(m, &mut j, &mut c, VBIOS_TIMEOUT_US);
        (r, j)
    }

    #[test]
    fn ga10x_like_chain_is_complete() {
        let mut m = board(&rom::ga10x_like_chain());
        let (r, j) = run(&mut m);
        let l = r.unwrap();
        let types: Vec<u8> = l.images.iter().map(|i| i.code_type).collect();
        assert_eq!(types, [CODE_PCI_AT, CODE_EFI, CODE_FWSEC, CODE_FWSEC]);
        assert_eq!(
            l.images.iter().map(|i| i.offset).collect::<Vec<_>>(),
            [0, 0x10000, 0x18000, 0x1c000]
        );
        assert!(l.images.iter().all(|i| i.has_npde && i.vendor == 0x10de));
        assert!(l.images[3].last && !l.images[2].last, "NPDE flag wins");
        let (bit_at, tokens) = l.bit.clone().unwrap();
        assert_eq!(bit_at, 0x1000);
        assert_eq!(tokens.len(), 2);
        assert!(l.complete());
        assert_eq!(l.ifr_version, None);
        assert!(m.writes.is_empty());
        assert_eq!(j.writes(), 0);
        assert!(
            m.reads
                .iter()
                .all(|&o| (PROM_BASE..PROM_BASE + PROM_SIZE).contains(&o))
        );
        assert!(l.prom_reads > 0 && (l.prom_reads as usize) == m.reads.len());
    }

    #[test]
    fn version_is_none_when_token_i_is_missing_or_short() {
        // ga10x_like_chain: tokens 0x42 and 0x70 only.
        let mut m = board(&rom::ga10x_like_chain());
        let l = run(&mut m).0.unwrap();
        assert_eq!(l.version, None);
        // Token 'i' there, but only 4 bytes: no patch byte → not read.
        let mut m = board(&rom::with_bios_info([0, 0x65, 0x04, 0x94, 0x3e], 4));
        let l = run(&mut m).0.unwrap();
        assert_eq!(l.version, None);
        assert!(l.bit.as_ref().unwrap().1.iter().any(|t| t.id == 0x69));
        assert!(m.writes.is_empty());
    }

    #[test]
    fn version_is_read_from_token_i_in_nouveau_order() {
        // Test bytes shaped like the workstation's Windows-reported
        // "94.04.65.00.3e" (HARDWARE_WORKSTATION.md), not copied from a ROM.
        let v = rom::bios_info_bytes(0x94, 0x04, 0x65, 0x00, 0x3e);
        let mut m = board(&rom::with_bios_info(v, 5));
        let (r, j) = run(&mut m);
        let l = r.unwrap();
        let ver = l.version.expect("token 'i' with 5 bytes");
        assert_eq!(
            ver,
            BiosVersion {
                major: 0x94,
                chip: 0x04,
                minor: 0x65,
                micro: 0x00,
                patch: 0x3e
            }
        );
        assert_eq!(alloc::format!("{ver}"), "94.04.65.00.3e");
        assert!(ver.matches_report("94.04.65.00.3e"));
        assert!(ver.matches_report(" 94.04.65.00.3E "));
        assert!(!ver.matches_report("94.04.65.00.3f"));
        assert!(
            !ver.matches_report("94.4.65.0.3e"),
            "another shape is not a match"
        );
        assert!(!ver.matches_report(""));
        assert!(l.complete());
        assert!(m.writes.is_empty());
        assert_eq!(j.writes(), 0);
        // The laptop's Windows report (hub, 12 Nov 23:03): "94.06.1E.00.23".
        let lap = BiosVersion::from_bytes(rom::bios_info_bytes(0x94, 0x06, 0x1e, 0x00, 0x23));
        assert!(lap.matches_report("94.06.1E.00.23"));
        assert!(!ver.matches_report("94.06.1E.00.23"));
    }

    #[test]
    fn ifr_v3_header_is_followed() {
        let bytes = rom::with_ifr_v3(&rom::ga10x_like_chain());
        let mut m = board(&bytes);
        let l = run(&mut m).0.unwrap();
        assert_eq!(l.ifr_version, Some(3));
        assert_eq!(l.rom_start, 0x200 + 4096 + 0x100);
        assert_eq!(l.images[0].offset, l.rom_start);
        assert!(l.complete());
    }

    #[test]
    fn plain_pcir_chain_without_npde() {
        let bytes = rom::build_chain(&[
            SimImage {
                code_type: CODE_PCI_AT,
                blocks: 8,
                npde: false,
                bit: None,
            },
            SimImage {
                code_type: CODE_EFI,
                blocks: 4,
                npde: false,
                bit: None,
            },
        ]);
        let mut m = board(&bytes);
        let l = run(&mut m).0.unwrap();
        assert_eq!(l.images.len(), 2);
        assert!(l.images[1].last && !l.images[1].has_npde);
        assert!(!l.complete(), "no FWSEC, no BIT: I3 is not passed");
    }

    #[test]
    fn workstation_chain_with_nv_images_is_complete() {
        // SIMULARE of the 28 Oct board result: "NV" (0x4E56) at 0x24600.
        let mut m = board(&rom::asus_like_chain());
        let (r, j) = run(&mut m);
        let l = r.unwrap();
        let got: Vec<(u32, u8)> = l.images.iter().map(|i| (i.offset, i.code_type)).collect();
        assert_eq!(
            got,
            [
                (0, CODE_PCI_AT),
                (0x12000, CODE_EFI),
                (0x24600, CODE_FWSEC),
                (0x28600, CODE_FWSEC)
            ]
        );
        assert!(l.complete());
        assert!(m.writes.is_empty() && j.writes() == 0);
    }

    #[test]
    fn npds_and_rgis_pcir_signatures_are_accepted_others_not() {
        for (sig, ok) in [(*b"NPDS", true), (*b"RGIS", true), (*b"PCIX", false)] {
            let mut bytes = rom::ga10x_like_chain();
            let at = 0x10000 + rom::SIM_PCIR_AT;
            bytes[at..at + 4].copy_from_slice(&sig);
            let mut m = board(&bytes);
            let r = run(&mut m).0;
            if ok {
                assert!(r.unwrap().complete());
            } else {
                assert_eq!(
                    r,
                    Err(GpuError::VerifyFailed {
                        offset: at as u32,
                        expected: SIG_PCIR,
                        actual: u32::from_le_bytes(sig)
                    })
                );
            }
        }
    }

    #[test]
    fn unknown_image_signature_is_still_refused() {
        let mut bytes = rom::asus_like_chain();
        rom::set_signature(&mut bytes, 0x24600, 0x4e57);
        let mut m = board(&bytes);
        assert_eq!(
            run(&mut m).0,
            Err(GpuError::VerifyFailed {
                offset: 0x24600,
                expected: 0xaa55,
                actual: 0x4e57
            })
        );
    }

    #[test]
    fn missing_signature_is_blocked_not_guessed() {
        let mut m = board(&[0u8; 64]);
        let (r, j) = run(&mut m);
        assert!(matches!(r, Err(GpuError::Blocked(_))));
        assert!(
            j.events()
                .any(|e| matches!(e, Event::Step { ok: false, .. }))
        );
        assert!(m.writes.is_empty());
    }

    #[test]
    fn broken_chain_and_dead_board() {
        let mut bytes = rom::ga10x_like_chain();
        bytes[0x10000] = 0; // second image loses its 0xAA55
        let mut m = board(&bytes);
        assert!(matches!(
            run(&mut m).0,
            Err(GpuError::VerifyFailed {
                offset: 0x10000,
                ..
            })
        ));
        let mut m = board(&[]);
        m.absent = true;
        assert_eq!(run(&mut m).0, Err(GpuError::NotResponding));
        assert_eq!(m.reads.len(), 1);
    }

    #[test]
    fn chain_running_past_the_window_is_refused() {
        let mut bytes = rom::build_chain(&[SimImage {
            code_type: CODE_PCI_AT,
            blocks: 8,
            npde: false,
            bit: None,
        }]);
        bytes[rom::SIM_PCIR_AT + 0x15] = 0; // not last
        bytes[rom::SIM_PCIR_AT + 0x10..rom::SIM_PCIR_AT + 0x12]
            .copy_from_slice(&0x0800u16.to_le_bytes()); // 1 MiB
        let mut m = board(&bytes);
        assert!(matches!(run(&mut m).0, Err(GpuError::OutOfBounds { .. })));
    }

    #[test]
    fn time_budget_is_enforced() {
        struct Slow(u64);
        impl Clock for Slow {
            fn now_us(&mut self) -> u64 {
                self.0 += 10_000; // every look at the clock costs 10 ms
                self.0
            }
            fn delay_us(&mut self, us: u32) {
                self.0 += us as u64;
            }
        }
        let mut m = board(&rom::ga10x_like_chain());
        let mut j = Journal::new(8);
        let r = read_layout(&mut m, &mut j, &mut Slow(0), 50_000);
        assert!(matches!(r, Err(GpuError::Timeout { .. })));
    }

    #[test]
    fn needs_init_window() {
        let mut m = SimMmio::new(0x1000);
        assert!(matches!(run(&mut m).0, Err(GpuError::InvalidArgument(_))));
        assert!(m.reads.is_empty());
    }
}
