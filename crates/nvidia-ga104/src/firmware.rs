//! Inventory of the NVIDIA-signed firmware the GA104 sequence needs.
//!
//! Source: linux-firmware `WHENCE` and tree (nvidia/ga104 -> mostly symlinks
//! to nvidia/ga102), licence `LICENSES/LICENCE.nvidia`. The files are NOT in
//! this repository and are never modified (the licence forbids it). We do not
//! produce or load unsigned firmware: Ampere verifies signatures in hardware.
//!
//! This module only describes and checks files the OS hands us.

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use crate::sha256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// GSP-RM path (nouveau default on Ampere since ~6.18, nova-core).
    Gsp,
    /// Pre-GSP path: ACR on SEC2 + signed FECS/GPCCS (nouveau GA102 GR, Linux 6.2).
    AcrGr,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FwFile {
    /// Path inside linux-firmware.
    pub path: &'static str,
    pub path_kind: Path,
    /// Expected container format, when known from public loaders.
    pub format: Format,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Format {
    /// ELF image (nouveau parses gsp-*.bin as ELF; inferred from its loader).
    Elf,
    /// Opaque NVIDIA blob with an NVIDIA header we do not parse yet.
    Opaque,
}

/// GSP firmware version we plan against — fixed by Adrian on 26 Oct 2026
/// (hub C-18): **570.144**, the version nova-core uses. Every other version is
/// refused by [`verify`]. (Before C-18 this was "535.113.01".)
pub const GSP_VERSION: &str = "570.144";

pub const GA104_FIRMWARE: &[FwFile] = &[
    FwFile {
        path: "nvidia/ga102/gsp/booter_load-570.144.bin",
        path_kind: Path::Gsp,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/gsp/booter_unload-570.144.bin",
        path_kind: Path::Gsp,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/gsp/bootloader-570.144.bin",
        path_kind: Path::Gsp,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/gsp/gsp-570.144.bin",
        path_kind: Path::Gsp,
        format: Format::Elf,
    },
    FwFile {
        path: "nvidia/ga104/gr/fecs_bl.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga104/gr/fecs_sig.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga104/gr/gpccs_bl.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga104/gr/gpccs_sig.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga104/gr/NET_img.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/acr/ucode_ahesasc.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/acr/ucode_asb.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/acr/ucode_unload.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/sec2/desc.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/sec2/image.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/sec2/sig.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/sec2/hs_bl_sig.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/nvdec/scrubber.bin",
        path_kind: Path::AcrGr,
        format: Format::Opaque,
    },
];

pub fn files_for(p: Path) -> impl Iterator<Item = &'static FwFile> {
    GA104_FIRMWARE.iter().filter(move |f| f.path_kind == p)
}

/// Structural sanity check of a file handed to us. It does *not* verify the
/// NVIDIA signature (the hardware does); it rejects obviously wrong input
/// such as an empty file or a non-ELF GSP image.
pub fn check(file: &FwFile, bytes: &[u8]) -> GpuResult<()> {
    if bytes.is_empty() {
        return Err(GpuError::InvalidArgument("fisier firmware gol"));
    }
    if file.format == Format::Elf && !bytes.starts_with(b"\x7fELF") {
        return Err(GpuError::InvalidArgument("imaginea GSP nu e ELF"));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// I4 — the pinned GSP firmware set (C-18) and its verified loading.
// ---------------------------------------------------------------------------

/// Where on the OS stick Adrian puts the files: `STICK_DIR` + [`Pinned::path`],
/// e.g. `/usb/firmware/nvidia/ga102/gsp/gsp-570.144.bin`. The tree below
/// `firmware/` mirrors the `linux-firmware` package, so the files can be
/// copied as they are.
pub const STICK_DIR: &str = "/usb/firmware/";

/// linux-firmware commit that added the 570.144 GSP images (May 2025); the
/// files are unchanged in the `20261019` release tag (checked 26 Oct 2026).
pub const LINUX_FIRMWARE_COMMIT: &str = "050f0a1e3a76d671c073b59c813a4119b610b725";

/// What each file of the GSP set is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GspRole {
    /// Booter (load): HS ucode on SEC2 that authenticates and starts GSP-RM (I7).
    BooterLoad,
    /// Booter (unload): clean shutdown of GSP-RM.
    BooterUnload,
    /// RISC-V boot loader of GSP-RM (I7).
    Bootloader,
    /// GSP-RM itself, an ELF container (I5: radix3 image).
    GspRm,
}

/// One firmware file as it must arrive: exact name, exact size, exact digest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pinned {
    pub role: GspRole,
    /// Path inside linux-firmware (and under [`STICK_DIR`] on the stick).
    pub path: &'static str,
    pub size: u64,
    pub sha256: [u8; 32],
}

/// The GSP-RM 570.144 set for GA102/GA104 (ga104/gsp is a link to ga102/gsp).
/// Sizes and digests computed on 26 Oct 2026 from the official linux-firmware
/// tree (commit [`LINUX_FIRMWARE_COMMIT`], tag 20261019) with `sha256sum`;
/// only the hash was computed, the contents were not inspected, and the files
/// were deleted afterwards (licence §2.1.3, C-18).
pub const GSP_570_144: [Pinned; 4] = [
    Pinned {
        role: GspRole::BooterLoad,
        path: "nvidia/ga102/gsp/booter_load-570.144.bin",
        size: 61_304,
        sha256: sha256::parse_hex(
            "4497e3eff7e95c774b8a569d17b27c08c9650158d10b229d2be81cdcad9a085b",
        ),
    },
    Pinned {
        role: GspRole::BooterUnload,
        path: "nvidia/ga102/gsp/booter_unload-570.144.bin",
        size: 41_080,
        sha256: sha256::parse_hex(
            "8e63db5b78d7d3e349f20a2d11099c3d7109081393cb09ffc0a28133324ae009",
        ),
    },
    Pinned {
        role: GspRole::Bootloader,
        path: "nvidia/ga102/gsp/bootloader-570.144.bin",
        size: 24_684,
        sha256: sha256::parse_hex(
            "82428f532240727e95bb3083fbaaba9b2cc7b937314323f2d546ce7245f27fad",
        ),
    },
    Pinned {
        role: GspRole::GspRm,
        path: "nvidia/ga102/gsp/gsp-570.144.bin",
        size: 63_571_696,
        sha256: sha256::parse_hex(
            "a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f",
        ),
    },
];

/// The OS side of I4: read a whole file from the stick. The kernel adapter
/// implements it over its FAT driver; tests implement it over memory.
pub trait FirmwareSource {
    /// `path` is [`STICK_DIR`] + [`Pinned::path`]. A missing file must be
    /// `Err(GpuError::NotPresent)`.
    fn read_file(&mut self, path: &str) -> GpuResult<Vec<u8>>;
}

/// Checks one file against its pin: size first (cheap), then SHA-256, then
/// the container magic for the ELF image. Nothing is loaded anywhere.
pub fn verify(pin: &Pinned, bytes: &[u8]) -> GpuResult<()> {
    if bytes.len() as u64 != pin.size {
        return Err(GpuError::InvalidArgument(
            "firmware: marimea nu e cea a versiunii 570.144 (alt fisier sau alta versiune)",
        ));
    }
    if sha256::digest(bytes) != pin.sha256 {
        return Err(GpuError::InvalidArgument(
            "firmware: suma SHA-256 nu se potriveste (fisier corupt sau modificat)",
        ));
    }
    if pin.role == GspRole::GspRm && !bytes.starts_with(b"\x7fELF") {
        return Err(GpuError::InvalidArgument("firmware: imaginea GSP nu e ELF"));
    }
    Ok(())
}

/// The four verified files, owned. Only constructed by [`load_gsp_set`].
pub struct GspFirmware {
    pub booter_load: Vec<u8>,
    pub booter_unload: Vec<u8>,
    pub bootloader: Vec<u8>,
    pub gsp_rm: Vec<u8>,
}

impl GspFirmware {
    pub fn get(&self, role: GspRole) -> &[u8] {
        match role {
            GspRole::BooterLoad => &self.booter_load,
            GspRole::BooterUnload => &self.booter_unload,
            GspRole::Bootloader => &self.bootloader,
            GspRole::GspRm => &self.gsp_rm,
        }
    }
}

/// I4: reads and verifies the whole GSP set (`GSP_570_144` in production;
/// tests pass a synthetic manifest). Stops at the first missing or wrong file
/// and returns its error; there is no partial success. Journals one step per
/// file. Touches no hardware.
pub fn load_gsp_set(
    src: &mut dyn FirmwareSource,
    manifest: &[Pinned; 4],
    journal: &mut Journal,
) -> GpuResult<GspFirmware> {
    let order = [
        GspRole::BooterLoad,
        GspRole::BooterUnload,
        GspRole::Bootloader,
        GspRole::GspRm,
    ];
    let mut got: [Option<Vec<u8>>; 4] = [None, None, None, None];
    for (slot, role) in got.iter_mut().zip(order) {
        let pin = manifest
            .iter()
            .find(|p| p.role == role)
            .ok_or(GpuError::InvalidArgument("firmware: manifest incomplet"))?;
        let mut path = alloc::string::String::from(STICK_DIR);
        path.push_str(pin.path);
        let r = src
            .read_file(&path)
            .and_then(|b| verify(pin, &b).map(|()| b));
        journal.push(Event::Step {
            name: step_name(role),
            ok: r.is_ok(),
        });
        match r {
            Ok(b) => *slot = Some(b),
            Err(GpuError::NotPresent) => {
                journal.push(Event::Note(
                    "I4: fisierul lipseste de pe stick (vezi docs/TEST_MATRIX.md partea B)",
                ));
                return Err(GpuError::NotPresent);
            }
            Err(e) => return Err(e),
        }
    }
    let [a, b, c, d] = got;
    match (a, b, c, d) {
        (Some(booter_load), Some(booter_unload), Some(bootloader), Some(gsp_rm)) => {
            Ok(GspFirmware {
                booter_load,
                booter_unload,
                bootloader,
                gsp_rm,
            })
        }
        _ => Err(GpuError::InvalidArgument("firmware: set incomplet")),
    }
}

fn step_name(role: GspRole) -> &'static str {
    match role {
        GspRole::BooterLoad => "I4: booter_load-570.144",
        GspRole::BooterUnload => "I4: booter_unload-570.144",
        GspRole::Bootloader => "I4: bootloader-570.144",
        GspRole::GspRm => "I4: gsp-570.144",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inventory() {
        assert_eq!(files_for(Path::Gsp).count(), 4);
        assert!(files_for(Path::AcrGr).count() >= 10);
        assert!(
            GA104_FIRMWARE
                .iter()
                .filter(|f| f.path_kind == Path::Gsp)
                .all(|f| f.path.contains(GSP_VERSION))
        );
        let gsp = GA104_FIRMWARE
            .iter()
            .find(|f| f.format == Format::Elf)
            .unwrap();
        assert!(check(gsp, b"\x7fELF\x02\x01").is_ok());
        assert!(check(gsp, b"MZ").is_err());
        assert!(check(&GA104_FIRMWARE[0], &[]).is_err());
    }

    #[test]
    fn pinned_set_is_570_144_and_matches_inventory() {
        for p in &GSP_570_144 {
            assert!(p.path.contains(GSP_VERSION), "{}", p.path);
            assert!(p.size > 0 && p.sha256 != [0; 32]);
            assert!(
                files_for(Path::Gsp).any(|f| f.path == p.path),
                "{} lipseste din inventar",
                p.path
            );
        }
        let gsp = GSP_570_144
            .iter()
            .find(|p| p.role == GspRole::GspRm)
            .unwrap();
        assert!(gsp.size > 32 << 20, "GSP-RM are zeci de MB");
    }

    /// SIMULARE: a stick in memory.
    struct MemStick(Vec<(alloc::string::String, Vec<u8>)>);
    impl FirmwareSource for MemStick {
        fn read_file(&mut self, path: &str) -> GpuResult<Vec<u8>> {
            self.0
                .iter()
                .find(|(p, _)| p == path)
                .map(|(_, b)| b.clone())
                .ok_or(GpuError::NotPresent)
        }
    }

    fn synthetic() -> ([Pinned; 4], MemStick) {
        // Synthetic stand-ins (NOT NVIDIA firmware): bytes we made up, pinned
        // the same way the real set is.
        let mut gsp = b"\x7fELF".to_vec();
        gsp.extend((0..5000u32).map(|i| i as u8));
        let files: [(GspRole, &'static str, Vec<u8>); 4] = [
            (
                GspRole::BooterLoad,
                "sim/booter_load-570.144.bin",
                vec![1; 300],
            ),
            (
                GspRole::BooterUnload,
                "sim/booter_unload-570.144.bin",
                vec![2; 200],
            ),
            (
                GspRole::Bootloader,
                "sim/bootloader-570.144.bin",
                vec![3; 100],
            ),
            (GspRole::GspRm, "sim/gsp-570.144.bin", gsp),
        ];
        let man = files.clone().map(|(role, path, b)| Pinned {
            role,
            path,
            size: b.len() as u64,
            sha256: sha256::digest(&b),
        });
        let stick = MemStick(
            files
                .into_iter()
                .map(|(_, p, b)| (alloc::format!("{STICK_DIR}{p}"), b))
                .collect(),
        );
        (man, stick)
    }

    #[test]
    fn load_set_simulated_ok_and_journaled() {
        let (man, mut stick) = synthetic();
        let mut j = Journal::new(16);
        let fw = load_gsp_set(&mut stick, &man, &mut j).unwrap();
        assert_eq!(fw.get(GspRole::Bootloader), &[3u8; 100][..]);
        assert!(fw.get(GspRole::GspRm).starts_with(b"\x7fELF"));
        assert_eq!(j.len(), 4);
        assert_eq!(j.writes(), 0);
    }

    #[test]
    fn load_set_refuses_missing_wrong_size_and_one_flipped_byte() {
        // Missing file.
        let (man, mut stick) = synthetic();
        stick.0.remove(2);
        let mut j = Journal::new(16);
        assert_eq!(
            load_gsp_set(&mut stick, &man, &mut j).err(),
            Some(GpuError::NotPresent)
        );
        // One byte changed, same size -> digest refuses.
        let (man, mut stick) = synthetic();
        stick.0[3].1[100] ^= 1;
        let e = load_gsp_set(&mut stick, &man, &mut Journal::new(16)).err();
        assert!(matches!(e, Some(GpuError::InvalidArgument(m)) if m.contains("SHA-256")));
        // Truncated file -> size refuses.
        let (man, mut stick) = synthetic();
        stick.0[0].1.pop();
        let e = load_gsp_set(&mut stick, &man, &mut Journal::new(16)).err();
        assert!(matches!(e, Some(GpuError::InvalidArgument(m)) if m.contains("marimea")));
    }

    #[test]
    fn real_manifest_refuses_anything_but_the_real_files() {
        // The real pins with a fake stick: every file has the right name but
        // wrong content, so nothing passes (we do not have, and never ship,
        // the real files).
        struct Fake;
        impl FirmwareSource for Fake {
            fn read_file(&mut self, path: &str) -> GpuResult<Vec<u8>> {
                assert!(path.starts_with(STICK_DIR) && path.contains("570.144"));
                Ok(vec![0; 61_304])
            }
        }
        let e = load_gsp_set(&mut Fake, &GSP_570_144, &mut Journal::new(8)).err();
        assert!(matches!(e, Some(GpuError::InvalidArgument(m)) if m.contains("SHA-256")));
    }
}
