//! I4 diagnostic for the workstation window (hub C-36): a **read-only**
//! check of the GSP 570.144 files Adrian put on the stick.
//!
//! Unlike [`crate::firmware::load_gsp_set`] (which stops at the first bad
//! file and keeps the bytes for loading), this looks at **all four** files,
//! keeps nothing, and reports for each one:
//! 1. present / missing / unreadable;
//! 2. size and SHA-256 against [`crate::firmware::GSP_570_144`];
//! 3. only when the file is the official one: whether *our* parser reads its
//!    container header (I4b/I4c — the part a loader must read to know where
//!    the pieces are). The pieces themselves stay opaque (licence §2.1.3).
//!
//! The two verdicts are kept apart on purpose: a wrong digest means "wrong
//! file on the stick" (Adrian copies it again); a right digest with a header
//! our parser refuses means "our DEDUS reading of the format is wrong" (we fix
//! the code; the file is fine).
//!
//! **No access to the GPU at all** (no config space, no BAR); the only
//! input is the [`FirmwareSource`] the OS gives us. Tested in SIMULARE only
//! (stand-ins from `gpu-sim::fw_stick`); never run on the real files here.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::firmware::{FirmwareSource, GspRole, Pinned, STICK_DIR};
use crate::fw_hdr::{HsV2, RiscvBootloader};
use crate::gsp::elf;
use crate::sha256;

/// Order in which the files are checked and listed.
pub const ROLE_ORDER: [GspRole; 4] = [
    GspRole::BooterLoad,
    GspRole::BooterUnload,
    GspRole::Bootloader,
    GspRole::GspRm,
];

/// Licence file the linux-firmware licence asks to keep next to the files
/// (`STICK_DIR` + this). Informative only: not part of the I4 criterion.
pub const LICENCE: &str = "LICENCE.nvidia";

/// What was found for one file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FileVerdict {
    /// The OS says the file is not there.
    Missing,
    /// The OS could not read it (the error, as text).
    Unreadable(String),
    /// Size differs from the pinned one (wrong file or other version).
    WrongSize { got: u64 },
    /// Same size, different SHA-256 (corrupt or modified copy).
    WrongDigest,
    /// Size and SHA-256 are those of the official 570.144 file.
    Official,
}

/// What our parser made of the container header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HeaderVerdict {
    /// Not looked at: the file is not the official one.
    NotChecked,
    /// Our parser read it; a short description of where the pieces are.
    Read(String),
    /// Our parser refused it (the error, as text): fix the code, not the file.
    Refused(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FileCheck {
    pub pin: Pinned,
    pub file: FileVerdict,
    pub header: HeaderVerdict,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FwCheck {
    pub files: Vec<FileCheck>,
    /// `Some(true)` present, `Some(false)` missing, `None` unreadable.
    pub licence: Option<bool>,
}

impl FwCheck {
    /// All four files are the official ones.
    pub fn files_ok(&self) -> bool {
        self.files.len() == 4 && self.files.iter().all(|f| f.file == FileVerdict::Official)
    }

    /// All four headers were read by our parser.
    pub fn headers_ok(&self) -> bool {
        self.files.len() == 4
            && self
                .files
                .iter()
                .all(|f| matches!(f.header, HeaderVerdict::Read(_)))
    }

    /// The I4 criterion for the window: files official AND headers read.
    pub fn passes(&self) -> bool {
        self.files_ok() && self.headers_ok()
    }

    /// The report as text lines (ASCII, Romanian without diacritics).
    pub fn lines(&self) -> Vec<String> {
        let mut out = Vec::new();
        out.push(String::from(
            "firmware GSP 570.144 (doar citire de pe stick, 0 accese la placa):",
        ));
        for f in &self.files {
            let name = f.pin.path.rsplit('/').next().unwrap_or(f.pin.path);
            let file = match &f.file {
                FileVerdict::Missing => String::from("LIPSESTE de pe stick"),
                FileVerdict::Unreadable(e) => format!("NU SE POATE CITI ({e})"),
                FileVerdict::WrongSize { got } => format!(
                    "ALT FISIER: {got} octeti, asteptat {} (alta versiune sau copiere incompleta)",
                    f.pin.size
                ),
                FileVerdict::WrongDigest => format!(
                    "ALT CONTINUT: marimea e buna ({}), suma SHA-256 nu (fisier stricat sau modificat)",
                    f.pin.size
                ),
                FileVerdict::Official => format!(
                    "ok: {} octeti, SHA-256 {}... = oficial",
                    f.pin.size,
                    hex_prefix(&f.pin.sha256, 8)
                ),
            };
            out.push(format!("  {name}: {file}"));
            match &f.header {
                HeaderVerdict::NotChecked => {}
                HeaderVerdict::Read(s) => out.push(format!("    antet citit: {s}")),
                HeaderVerdict::Refused(e) => out.push(format!(
                    "    antet NECITIT de parserul nostru: {e} (fisierul e cel oficial: de reparat in cod, nu pe stick)"
                )),
            }
        }
        out.push(String::from(match self.licence {
            Some(true) => "  LICENCE.nvidia: prezenta",
            Some(false) => {
                "  LICENCE.nvidia: lipseste (nu opreste verificarea; pune-o langa fisiere)"
            }
            None => "  LICENCE.nvidia: nu se poate citi",
        }));
        let n_ok = self
            .files
            .iter()
            .filter(|f| f.file == FileVerdict::Official)
            .count();
        out.push(format!(
            "criteriul I4: {} (fisiere oficiale {}/4, antete citite {}/4); 0 scrieri, 0 accese la placa",
            if self.passes() { "TRECE" } else { "NU TRECE" },
            n_ok,
            self.files
                .iter()
                .filter(|f| matches!(f.header, HeaderVerdict::Read(_)))
                .count()
        ));
        out
    }
}

fn hex_prefix(d: &[u8; 32], n: usize) -> String {
    let mut s = String::new();
    for b in d.iter().take(n / 2) {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn header_of(role: GspRole, bytes: &[u8]) -> HeaderVerdict {
    let r: GpuResult<String> = match role {
        GspRole::BooterLoad | GspRole::BooterUnload => HsV2::parse(bytes).map(|h| {
            format!(
                "HS v2: {} semnaturi de {} octeti, {} aplicatii, toate in limitele fisierului",
                h.num_sig,
                h.sig_size(),
                h.apps.len()
            )
        }),
        GspRole::Bootloader => RiscvBootloader::parse(bytes).map(|b| {
            format!(
                "descriptor RISC-V: cod, date si manifest ({} octeti) in limitele fisierului",
                b.manifest_size
            )
        }),
        GspRole::GspRm => elf::section(bytes, elf::FWIMAGE).and_then(|img| {
            elf::section(bytes, elf::FWSIGNATURE_GA10X).map(|sig| {
                format!(
                    "ELF-64: {} = {} octeti, {} = {} octeti",
                    elf::FWIMAGE,
                    img.len(),
                    elf::FWSIGNATURE_GA10X,
                    sig.len()
                )
            })
        }),
    };
    match r {
        Ok(s) => HeaderVerdict::Read(s),
        Err(e) => HeaderVerdict::Refused(format!("{e}")),
    }
}

/// Checks one file against its pin; the bytes are dropped on return.
pub fn check_file(src: &mut dyn FirmwareSource, pin: &Pinned) -> FileCheck {
    let mut path = String::from(STICK_DIR);
    path.push_str(pin.path);
    let (file, header) = match src.read_file(&path) {
        Err(GpuError::NotPresent) => (FileVerdict::Missing, HeaderVerdict::NotChecked),
        Err(e) => (
            FileVerdict::Unreadable(format!("{e}")),
            HeaderVerdict::NotChecked,
        ),
        Ok(b) if b.len() as u64 != pin.size => (
            FileVerdict::WrongSize {
                got: b.len() as u64,
            },
            HeaderVerdict::NotChecked,
        ),
        Ok(b) if sha256::digest(&b) != pin.sha256 => {
            (FileVerdict::WrongDigest, HeaderVerdict::NotChecked)
        }
        Ok(b) => (FileVerdict::Official, header_of(pin.role, &b)),
    };
    FileCheck {
        pin: *pin,
        file,
        header,
    }
}

/// I4 for the window: checks every file of `manifest` (production:
/// [`crate::firmware::GSP_570_144`]) in [`ROLE_ORDER`], one at a time (only
/// one file in memory at once, ~64 MB for GSP-RM), plus the licence file.
/// Never stops early; never touches the GPU.
pub fn check_stick(src: &mut dyn FirmwareSource, manifest: &[Pinned; 4]) -> FwCheck {
    let mut files = Vec::new();
    for role in ROLE_ORDER {
        if let Some(pin) = manifest.iter().find(|p| p.role == role) {
            files.push(check_file(src, pin));
        }
    }
    let mut lic = String::from(STICK_DIR);
    lic.push_str(LICENCE);
    let licence = match src.read_file(&lic) {
        Ok(_) => Some(true),
        Err(GpuError::NotPresent) => Some(false),
        Err(_) => None,
    };
    FwCheck { files, licence }
}

/// A pin for a file the caller made (tests and the OS `--sim` mode, with
/// the `gpu-sim::fw_stick` stand-ins). Never used for the real set.
pub fn pin_of(role: GspRole, path: &'static str, bytes: &[u8]) -> Pinned {
    Pinned {
        role,
        path,
        size: bytes.len() as u64,
        sha256: sha256::digest(bytes),
    }
}

/// An in-memory [`FirmwareSource`] (tests, OS `--sim`).
#[derive(Default)]
pub struct MemSource {
    files: Vec<(String, Vec<u8>)>,
    /// Reads served, for "one file at a time" checks.
    pub reads: u32,
}

impl MemSource {
    pub fn new() -> Self {
        Self::default()
    }
    /// `path` is the full path (`STICK_DIR` + relative path).
    pub fn insert(&mut self, path: String, bytes: Vec<u8>) {
        self.files.retain(|(p, _)| *p != path);
        self.files.push((path, bytes));
    }
}

impl FirmwareSource for MemSource {
    fn read_file(&mut self, path: &str) -> GpuResult<Vec<u8>> {
        self.reads += 1;
        self.files
            .iter()
            .find(|(p, _)| p == path)
            .map(|(_, b)| b.clone())
            .ok_or(GpuError::NotPresent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::firmware::GSP_570_144;
    use gpu_sim::fw_stick;

    /// SIMULARE: the gpu-sim stand-ins on an in-memory stick, pinned like
    /// the real set.
    fn sim() -> ([Pinned; 4], MemSource) {
        let files = fw_stick::files();
        let man =
            core::array::from_fn(|i| pin_of(ROLE_ORDER[i], fw_stick::SIM_PATHS[i], &files[i]));
        let mut src = MemSource::new();
        for (p, b) in fw_stick::SIM_PATHS.iter().zip(files) {
            src.insert(format!("{STICK_DIR}{p}"), b);
        }
        src.insert(format!("{STICK_DIR}{LICENCE}"), fw_stick::licence());
        (man, src)
    }

    #[test]
    fn simulated_stick_passes_and_every_header_is_read() {
        let (man, mut src) = sim();
        let r = check_stick(&mut src, &man);
        assert!(r.passes(), "{:#?}", r.lines());
        assert_eq!(r.licence, Some(true));
        assert_eq!(src.reads, 5, "4 fisiere + licenta, fiecare o data");
        let text = r.lines().join("\n");
        assert!(text.contains("criteriul I4: TRECE (fisiere oficiale 4/4, antete citite 4/4)"));
        assert!(text.contains("HS v2: 3 semnaturi de 384 octeti, 1 aplicatii"));
        assert!(text.contains("HS v2: 2 semnaturi"));
        assert!(text.contains(".fwimage = 8192 octeti, .fwsignature_ga10x = 384 octeti"));
        assert!(text.contains("0 accese la placa"));
        assert!(r.lines().len() < 20);
    }

    #[test]
    fn every_file_is_reported_even_after_a_bad_one() {
        let (man, mut src) = sim();
        // 1: missing; 2: one byte flipped; 3: truncated.
        src.files
            .retain(|(p, _)| !p.ends_with(fw_stick::SIM_PATHS[0]));
        src.files
            .iter_mut()
            .find(|(p, _)| p.ends_with(fw_stick::SIM_PATHS[1]))
            .unwrap()
            .1[0x700] ^= 1;
        src.files
            .iter_mut()
            .find(|(p, _)| p.ends_with(fw_stick::SIM_PATHS[2]))
            .unwrap()
            .1
            .pop();
        src.files.retain(|(p, _)| !p.ends_with(LICENCE));
        let r = check_stick(&mut src, &man);
        let v: Vec<_> = r.files.iter().map(|f| f.file.clone()).collect();
        assert_eq!(
            v,
            [
                FileVerdict::Missing,
                FileVerdict::WrongDigest,
                FileVerdict::WrongSize { got: 0x1fff },
                FileVerdict::Official
            ]
        );
        // Bad files are not parsed; the good one still is.
        assert!(
            r.files[..3]
                .iter()
                .all(|f| f.header == HeaderVerdict::NotChecked)
        );
        assert!(matches!(r.files[3].header, HeaderVerdict::Read(_)));
        assert!(!r.passes());
        assert_eq!(r.licence, Some(false));
        let text = r.lines().join("\n");
        assert!(text.contains("LIPSESTE"));
        assert!(text.contains("ALT CONTINUT"));
        assert!(text.contains("ALT FISIER: 8191 octeti, asteptat 8192"));
        assert!(text.contains("criteriul I4: NU TRECE (fisiere oficiale 1/4, antete citite 1/4)"));
    }

    #[test]
    fn official_file_with_a_header_we_cannot_read_blames_the_code() {
        // The "official" pin is computed on a file our parser refuses: this is
        // the case "our DEDUS format reading is wrong", not "wrong file".
        let mut odd = fw_stick::bootloader();
        odd[0] = 0; // no 0x10de magic
        let files = fw_stick::files();
        let mut man: [Pinned; 4] =
            core::array::from_fn(|i| pin_of(ROLE_ORDER[i], fw_stick::SIM_PATHS[i], &files[i]));
        man[2] = pin_of(GspRole::Bootloader, fw_stick::SIM_PATHS[2], &odd);
        let mut src = MemSource::new();
        for (i, (p, b)) in fw_stick::SIM_PATHS.iter().zip(files).enumerate() {
            src.insert(
                format!("{STICK_DIR}{p}"),
                if i == 2 { odd.clone() } else { b },
            );
        }
        let r = check_stick(&mut src, &man);
        assert!(r.files_ok());
        assert!(!r.headers_ok() && !r.passes());
        assert!(matches!(r.files[2].header, HeaderVerdict::Refused(_)));
        let text = r.lines().join("\n");
        assert!(text.contains("de reparat in cod, nu pe stick"));
        assert!(text.contains("antete citite 3/4"));
    }

    #[test]
    fn gsp_elf_without_the_ga10x_signature_is_refused() {
        let no_sig = fw_stick::elf(&[(".fwimage", &[1u8; 64])]);
        let pin = pin_of(GspRole::GspRm, fw_stick::SIM_PATHS[3], &no_sig);
        let mut src = MemSource::new();
        src.insert(format!("{STICK_DIR}{}", fw_stick::SIM_PATHS[3]), no_sig);
        let f = check_file(&mut src, &pin);
        assert_eq!(f.file, FileVerdict::Official);
        assert!(matches!(f.header, HeaderVerdict::Refused(ref e) if e.contains("lipseste")));
    }

    #[test]
    fn real_manifest_on_an_empty_stick_reports_four_missing_files() {
        // What the OS sees in QEMU (no real files anywhere): 4 x missing.
        let mut src = MemSource::new();
        let r = check_stick(&mut src, &GSP_570_144);
        assert_eq!(r.files.len(), 4);
        assert!(r.files.iter().all(|f| f.file == FileVerdict::Missing));
        assert_eq!(
            r.files.iter().map(|f| f.pin.path).collect::<Vec<_>>(),
            [
                "nvidia/ga102/gsp/booter_load-570.144.bin",
                "nvidia/ga102/gsp/booter_unload-570.144.bin",
                "nvidia/ga102/gsp/bootloader-570.144.bin",
                "nvidia/ga102/gsp/gsp-570.144.bin"
            ]
        );
        assert!(
            r.lines()
                .join("\n")
                .contains("NU TRECE (fisiere oficiale 0/4")
        );
    }

    #[test]
    fn stand_ins_never_match_the_real_pins() {
        // The simulated bytes must never pass as the real 570.144 files.
        let files = fw_stick::files();
        for (pin, b) in GSP_570_144.iter().zip(files.iter()) {
            assert_ne!(sha256::digest(b), pin.sha256);
        }
    }

    #[test]
    fn unreadable_file_is_reported_with_the_os_error() {
        struct Broken;
        impl FirmwareSource for Broken {
            fn read_file(&mut self, _: &str) -> GpuResult<Vec<u8>> {
                Err(GpuError::InvalidArgument("citirea de pe stick a esuat"))
            }
        }
        let r = check_stick(&mut Broken, &GSP_570_144);
        assert!(
            r.files
                .iter()
                .all(|f| matches!(f.file, FileVerdict::Unreadable(_)))
        );
        assert_eq!(r.licence, None);
        assert!(r.lines()[1].contains("NU SE POATE CITI"));
    }
}
