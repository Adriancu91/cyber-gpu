//! Inventory of the NVIDIA-signed firmware the GA104 sequence needs.
//!
//! Source: linux-firmware `WHENCE` and tree (nvidia/ga104 -> mostly symlinks
//! to nvidia/ga102), licence `LICENSES/LICENCE.nvidia`. The files are NOT in
//! this repository and are never modified (the licence forbids it). We do not
//! produce or load unsigned firmware: Ampere verifies signatures in hardware.
//!
//! This module only describes and checks files the OS hands us.

use gpu_core::error::{GpuError, GpuResult};

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

/// GSP firmware version we plan against (one of the two shipped for ga102).
pub const GSP_VERSION: &str = "535.113.01";

pub const GA104_FIRMWARE: &[FwFile] = &[
    FwFile {
        path: "nvidia/ga102/gsp/booter_load-535.113.01.bin",
        path_kind: Path::Gsp,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/gsp/booter_unload-535.113.01.bin",
        path_kind: Path::Gsp,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/gsp/bootloader-535.113.01.bin",
        path_kind: Path::Gsp,
        format: Format::Opaque,
    },
    FwFile {
        path: "nvidia/ga102/gsp/gsp-535.113.01.bin",
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
}
