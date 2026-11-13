//! C-116 (3, the reading part): what the laptop's RTX 3060 Laptop GPU
//! (GA106) actually returned on 13 Nov 2026, read by Cyber Universe OS
//! `os-301` (cyber-gpu `835f24d`) with `gpu tur --trezire` — hub
//! `rezultate/2026-11-13-laptop-asus-os301/`: `gpu-jurnal-003.txt` (the 8
//! BAR0 reads and the 2 Command writes) and the photo `gpu-tur-trezire.jpg`
//! (the 45 `fapt` lines; the first ones — ids, Command, ReBAR — are cut off
//! in the photo and are not repeated here). That run wrote 2 configuration
//! registers (Memory Space on, then back to 0) and **0 GPU registers**.
//!
//! Only register values, positions, sizes and ids are kept — never the
//! content of the VBIOS or of NVIDIA firmware (the first PROM dword
//! `0xeb7faa55` is deliberately not reproduced: its upper half is VBIOS
//! code). [`observed_laptop_config`], [`observed_laptop_bar0`] and
//! [`observed_laptop_rom`] build the SIMULATED laptop from these facts, on
//! top of [`crate::laptop`] (what the OS and Windows had reported before).
//! What is still a guess is in [`STILL_SIM`]. A test passing against this
//! model is SIMULARE: it proves the model matches one journal, not that the
//! laptop will behave the same way tomorrow.

use alloc::vec::Vec;

use crate::laptop as lp;
use crate::mmio::SimMmio;
use crate::pci::SimConfig;
use crate::rom::{SimFwsec, SimImage, build_chain, put_fwsec, set_signature};

// ---- PCI configuration (photo; `pci.*` facts) ----
/// BAR3 32 MiB at 0x62_0000_0000 (was SIM in session 95/96).
pub const BAR3_BASE: u64 = 0x62_0000_0000;
/// BAR5: I/O at 0x4000.
pub const BAR5_IO_BASE: u16 = 0x4000;
/// Command as the firmware left it and as the tour left it again:
/// `cfg w 0x004: 0x00100000 <- 0x00000002 => 0x00100002`, then back.
pub const STATUS_AND_COMMAND: u32 = 0x0010_0000;

// ---- BAR0 registers (journal 003, in reading order) ----
pub const BOOT_0: u32 = 0xb760_00a1;
pub const BOOT_42: u32 = 0x176a_1000;
/// `NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK` (0x118128) — the
/// same value as on the workstation.
pub const PRIV_LEVEL_MASK: u32 = 0x0000_8b8f;
/// `GFW_BOOT` (0x118234): progress 0xff at the first look ("runde 1").
pub const GFW_BOOT: u32 = 0x0000_03ff;
/// Fuse `0x8241e0` = 3 → "versiunea 2 semnatura 2" (I3c), as on GA104.
pub const FUSE_UCODE_VERSION: u32 = 0x0000_0003;
/// `0x1183a4` = 0x1800 → 6144 MiB of VRAM (the laptop's 6 GiB).
pub const FB_SIZE_REG: u32 = 0x0000_1800;
/// `0x1fa828` = 0 → WPR2 down (cold boot), `i6.trece = da`.
pub const WPR2_HI: u32 = 0;
/// The 8 BAR0 reads of the tour, `(offset, value)`, in journal order.
pub const BAR0_READS: [(u64, u32); 8] = [
    (0x00_0000, BOOT_0),
    (0x00_0a00, BOOT_42),
    (0x11_8128, PRIV_LEVEL_MASK),
    (0x11_8234, GFW_BOOT),
    (0x30_0000, 0xeb7f_aa55), // first PROM dword: journaled by the OS, NOT reproduced in the model
    (0x82_41e0, FUSE_UCODE_VERSION),
    (0x11_83a4, FB_SIZE_REG),
    (0x1f_a828, WPR2_HI),
];

// ---- VBIOS layout (photo; `vbios.*` facts) ----
/// No IFR header: the first image is at 0.
pub const ROM_START: u32 = 0;
/// `(offset, code type, size in bytes, vendor, device)` of the 3 images.
/// Unlike the workstation's VBIOS there is **no EFI image** (type 0x03):
/// PCI-AT, then the two FWSEC images, which carry the generic id
/// `10de:2200` as on the workstation.
pub const IMAGES: [(u32, u8, u32, u16, u16); 3] = [
    (0x00000, 0x00, 65024, 0x10de, 0x2520),
    (0x0fe00, 0xe0, 22016, 0x10de, 0x2200),
    (0x15400, 0xe0, 415744, 0x10de, 0x2200),
];
/// The images end here (0x15400 + 415744).
pub const ROM_END: u32 = 0x7ac00;
/// BIT table offset and its 17 tokens `(id, version)` — the same list, in
/// the same order, as the workstation's ([`crate::observed::BIT_TOKENS`]).
pub const BIT_AT: u32 = 0x1b0;
pub const BIT_TOKENS: [(u8, u8); 17] = crate::observed::BIT_TOKENS;
/// `vbios.versiune = 94.06.1e.00.23` read from BIT token 'i' — identical
/// to what Windows / NVML report ([`crate::laptop::VBIOS_VERSION`]). This
/// is the first board where our reading of token 'i' was checked against
/// another tool: the byte order of `vbios::read_bios_version` is OBSERVED.
pub const VBIOS_VERSION_BYTES: [u8; 5] = crate::rom::bios_info_bytes(0x94, 0x06, 0x1e, 0x00, 0x23);

// ---- FWSEC (photo; `fwsec.*` facts) ----
pub const PMU_TABLE: u32 = 0x7aafe;
/// The same 16 application ids as on the workstation (0x85 = FWSEC_PROD).
pub const PMU_APPS: [u8; 16] = crate::observed::PMU_APPS;
pub const DESC_AT: u32 = 0x2c834;
pub const DESC_VERSION: u8 = 3;
pub const DESC_HEADER: u32 = 0x4ac;
/// IMEM 0xe100 (workstation: 0xe200) + DMEM 0x800 = ucode 0xe900: the
/// laptop's FWSEC is a different build from the workstation's.
pub const IMEM_LOAD: u32 = 0xe100;
pub const DMEM_LOAD: u32 = 0x800;
pub const SIGNATURES: u8 = 3;
pub const SIGNATURE_VERSIONS: u16 = 0x0007;
pub const ENGINE_MASK: u16 = 0x0400;
pub const UCODE_ID: u8 = 9;

/// What this model still invents, because the laptop has not shown it or
/// it must not be copied.
pub const STILL_SIM: &[&str] = &[
    "configuratia PCI: pozitia capabilitatilor, starea (Status) in afara bitului 'lista de capabilitati', marimile suportate de Resizable BAR",
    "VBIOS: continutul imaginilor (intentionat: nu copiem VBIOS-ul; primul cuvant PROM 0xeb7faa55 nu e reprodus), pozitia PCIR/NPDE in imagini, lungimile si datele jetoanelor BIT in afara jetonului 0x69 (5 octeti, OBSERVATI prin vbios.versiune)",
    "FWSEC: antetul tabelei PMU si descriptorii celorlalte aplicatii; octetii ucode-ului si ai semnaturilor (zero)",
    "orice registru necitit pe laptop: 0 in simulare (placa reala poate da 0xbadf....); registrele GSP/SEC2/FBIF nu au fost citite niciodata pe GA106",
];

/// SIMULATED configuration space of the laptop as the firmware leaves it:
/// [`SimConfig::ga106_laptop_asus`] with the observed BAR3 base and BAR5.
/// Command stays 0 (Memory Space off): the tour must wake the board itself.
pub fn observed_laptop_config() -> SimConfig {
    let mut s = SimConfig::ga106_laptop_asus();
    s.write_raw(0x04, STATUS_AND_COMMAND);
    s.write_raw(0x1c, (BAR3_BASE as u32) | 0xc);
    s.write_raw(0x20, (BAR3_BASE >> 32) as u32);
    s.write_raw(0x24, BAR5_IO_BASE as u32 | 1);
    s.set_power_state(lp::POWER_STATE);
    s
}

/// SIMULATED 16 MiB BAR0 with the 7 observed registers (the PROM dword
/// excepted) and the [`observed_laptop_rom`] layout in the PROM window.
pub fn observed_laptop_bar0() -> SimMmio {
    let mut m = SimMmio::new(16 << 20);
    for &(off, v) in &BAR0_READS {
        if off != 0x30_0000 {
            m.set(off, v);
        }
    }
    m.load_bytes(0x30_0000, &observed_laptop_rom());
    m
}

fn blocks(size: u32) -> u16 {
    (size / 512) as u16
}

/// A synthetic ROM with the observed layout: the 3 images at the same
/// offsets and sizes with the same ids, the BIT table at the same place
/// with the same token ids/versions (token 'i' = the observed version),
/// the FWSEC PMU table and descriptor where the laptop has them, with the
/// observed descriptor fields. Every content byte is zero or SIM.
pub fn observed_laptop_rom() -> Vec<u8> {
    const FALCON_DATA_AT: usize = 0x3000;
    let mut bios_info_at = None;
    let tokens: Vec<crate::rom::SimBitToken> = BIT_TOKENS
        .iter()
        .enumerate()
        .map(|(k, &(id, ver))| {
            let data = if id == 0x70 {
                FALCON_DATA_AT as u16
            } else {
                0x2000 + 0x10 * k as u16
            };
            let len = if id == 0x69 {
                bios_info_at = Some(data as usize);
                5u16
            } else {
                4u16
            };
            (id, ver, len, data)
        })
        .collect();
    let images: Vec<SimImage> = IMAGES
        .iter()
        .enumerate()
        .map(|(n, &(_, ty, size, _, _))| SimImage {
            code_type: ty,
            blocks: blocks(size),
            npde: true,
            bit: (n == 0).then(|| (BIT_AT as usize, tokens.clone())),
        })
        .collect();
    let mut b = build_chain(&images);
    if let Some(at) = bios_info_at {
        b[at..at + 5].copy_from_slice(&VBIOS_VERSION_BYTES);
    }
    // PCIR device ids: build_chain writes the workstation's; put the
    // laptop's. FWSEC images: "NV" signature, "NPDS" data structure.
    for &(off, ty, _, _, dev) in &IMAGES {
        let off = off as usize;
        let p = off + crate::rom::SIM_PCIR_AT;
        b[p + 6..p + 8].copy_from_slice(&dev.to_le_bytes());
        if ty == 0xe0 {
            set_signature(&mut b, off, 0x4e56);
            b[p..p + 4].copy_from_slice(b"NPDS");
        }
    }
    let sections = [
        (IMAGES[1].0 as usize, IMAGES[1].2),
        (IMAGES[2].0 as usize, IMAGES[2].2),
    ];
    let rel = |abs: u32| -> u32 {
        let mut base = 0;
        for &(off, size) in &sections {
            if abs >= off as u32 && abs < off as u32 + size {
                return base + abs - off as u32;
            }
            base += size;
        }
        panic!("SIM: offset outside the FWSEC images");
    };
    let desc_rel = rel(DESC_AT);
    let f = SimFwsec {
        falcon_data_at: FALCON_DATA_AT,
        pmu_table_rel: rel(PMU_TABLE),
        // SIM: every listed application points at the FWSEC_PROD descriptor.
        entries: PMU_APPS.iter().map(|&a| (a, 0, desc_rel)).collect(),
        desc_rel,
        desc_version: DESC_VERSION,
        signature_count: SIGNATURES,
        signature_versions: SIGNATURE_VERSIONS,
        imem_load_size: IMEM_LOAD,
        dmem_load_size: DMEM_LOAD,
        stored_extra: 0,
    };
    put_fwsec(&mut b, IMAGES[1].0, &sections, &f);
    // put_fwsec writes a SIM ucode_id; the laptop's is UCODE_ID.
    b[DESC_AT as usize + 38] = UCODE_ID;
    debug_assert_eq!(b.len(), ROM_END as usize);
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::hal::{ConfigSpace, Mmio};

    #[test]
    fn rom_has_the_observed_image_boundaries() {
        let b = observed_laptop_rom();
        assert_eq!(b.len(), ROM_END as usize);
        for &(off, ty, size, _, dev) in &IMAGES {
            let o = off as usize;
            let sig = u16::from_le_bytes([b[o], b[o + 1]]);
            assert_eq!(sig, if ty == 0xe0 { 0x4e56 } else { 0xaa55 });
            let p = o + crate::rom::SIM_PCIR_AT;
            assert_eq!(u16::from_le_bytes([b[p + 6], b[p + 7]]), dev);
            assert_eq!(
                u16::from_le_bytes([b[p + 0x10], b[p + 0x11]]) as u32 * 512,
                size
            );
        }
        assert_eq!(&b[BIT_AT as usize + 2..BIT_AT as usize + 5], b"BIT");
        assert_eq!(DESC_HEADER, 44 + SIGNATURES as u32 * 384);
        // The PMU table lies inside the last image, near its end.
        assert!(PMU_TABLE > IMAGES[2].0 && PMU_TABLE < ROM_END);
    }

    #[test]
    fn bar0_and_config_carry_the_observed_values() {
        let mut m = observed_laptop_bar0();
        for &(off, v) in &BAR0_READS {
            if off == 0x30_0000 {
                assert_ne!(m.read32_raw(off), v, "VBIOS bytes are never reproduced");
            } else {
                assert_eq!(m.read32_raw(off), v, "{off:#x}");
            }
        }
        let mut c = observed_laptop_config();
        assert_eq!(
            c.read32(0x04),
            STATUS_AND_COMMAND,
            "Memory Space off, as found"
        );
        assert_eq!(c.read32(0x24), 0x4001);
        assert_eq!(c.writes, 0);
    }
}
