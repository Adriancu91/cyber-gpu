//! C-54 (3): what the workstation's RTX 3070 Ti actually returned, read by
//! Cyber Universe OS on 28 Oct, 29 Oct and 1 Nov 2026 (hub `rezultate/`:
//! `2026-10-28-workstation-kronos-in-os/raport-20261028-1.txt`,
//! `2026-10-29-workstation-kronos-gpu/raport-20261029-1.txt`,
//! `2026-11-01-workstation-477-i4/raport-20261101-1.txt`, section
//! `== jurnal gpu ==`). All of those runs were read-only (0 writes).
//!
//! Only register values, positions, sizes and ids are kept here — never the
//! content of the VBIOS or of NVIDIA firmware (the first PROM dword is
//! deliberately not reproduced: its upper half is VBIOS code).
//!
//! [`observed_config`], [`observed_bar0`] and [`observed_rom`] build the
//! SIMULATED board from these facts, so the simulation guesses less. What is
//! still a guess is listed in [`STILL_SIM`]. A test passing against this
//! model is still SIMULARE, not proof of hardware behaviour.

use alloc::vec::Vec;

use crate::mmio::SimMmio;
use crate::pci::SimConfig;
use crate::rom::{SimFwsec, SimImage, build_chain, put_fwsec, set_signature};

// ---- PCI configuration (28 Oct, `gpu` level 0) ----
/// "Command 0003 (memorie da, bus master nu)": I/O + memory decode, no DMA.
pub const COMMAND: u16 = 0x0003;
/// "stare D0".
pub const POWER_STATE: u8 = 0;
/// "Resizable BAR: BAR0 = 16 MiB acum / BAR1 = 256 MiB / BAR3 = 32 MiB":
/// the capability is present with three entries (Resizable BAR is off in
/// the BIOS, so these are the default sizes). `(bar index, current log2 MiB)`.
pub const REBAR_CURRENT: [(u8, u8); 3] = [(0, 4), (1, 8), (3, 5)];

// ---- BAR0 registers (28 Oct) ----
pub const BOOT_0: u32 = 0xb740_00a1;
pub const BOOT_42: u32 = 0x174a_1000;
/// `NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK` (0x118128).
pub const PRIV_LEVEL_MASK: u32 = 0x0000_8b8f;
/// `NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT` (0x118234): progress
/// 0xff = done, at the first look ("dupa 1 runde").
pub const GFW_BOOT: u32 = 0x0000_03ff;

// ---- VBIOS layout (29 Oct, `gpu vbios` + `gpu vbios detalii`) ----
/// No IFR ("NVGI") header: the first image is at 0.
pub const ROM_START: u32 = 0;
/// `(offset, code type, size in bytes, vendor, device)` of the 4 images.
pub const IMAGES: [(u32, u8, u32, u16, u16); 4] = [
    (0x00000, 0x00, 65024, 0x10de, 0x2482),
    (0x0fe00, 0x03, 83968, 0x10de, 0x2482),
    (0x24600, 0xe0, 22016, 0x10de, 0x2200),
    (0x29c00, 0xe0, 386048, 0x10de, 0x2200),
];
/// The images end here; nothing after it up to 1 MiB.
pub const ROM_END: u32 = 0x88000;
/// BIT table offset and its 17 tokens `(id, version)`, in ROM order.
pub const BIT_AT: u32 = 0x1b0;
pub const BIT_TOKENS: [(u8, u8); 17] = [
    (0x32, 1),
    (0x42, 2),
    (0x43, 2),
    (0x44, 1),
    (0x49, 1),
    (0x4d, 2),
    (0x4e, 0),
    (0x50, 2),
    (0x53, 2),
    (0x54, 1),
    (0x55, 1),
    (0x56, 1),
    (0x78, 1),
    (0x64, 1),
    (0x70, 2),
    (0x75, 1),
    (0x69, 2),
];

// ---- FWSEC (29 Oct, `gpu fwsec`, I3b) ----
pub const PMU_TABLE: u32 = 0x87ecb;
/// Application ids listed by the PMU lookup table (0x85 = FWSEC_PROD).
pub const PMU_APPS: [u8; 16] = [
    0x01, 0x00, 0x00, 0x00, 0x00, 0x07, 0x08, 0x00, 0x45, 0x85, 0x49, 0x89, 0x00, 0x00, 0x00, 0x00,
];
pub const DESC_AT: u32 = 0x41208;
pub const DESC_VERSION: u8 = 3;
pub const DESC_HEADER: u32 = 0x4ac;
pub const IMEM_LOAD: u32 = 0xe200;
pub const DMEM_LOAD: u32 = 0x800;
pub const SIGNATURES: u8 = 3;
pub const SIGNATURE_VERSIONS: u16 = 0x0007;
pub const ENGINE_MASK: u16 = 0x0400;
pub const UCODE_ID: u8 = 9;

/// What [`observed_config`] / [`observed_bar0`] / [`observed_rom`] still
/// invent, because the board has not shown it yet (or it must not be
/// copied). Printed by the comparison tool so nobody mistakes it for data.
pub const STILL_SIM: &[&str] = &[
    "configuratia PCI: pozitia capabilitatilor, starea (Status), legatura PCIe, marimile suportate de Resizable BAR",
    "VBIOS: continutul imaginilor (intentionat: nu copiem VBIOS-ul), pozitia PCIR/NPDE in imagini, lungimile si datele jetoanelor BIT",
    "FWSEC: antetul tabelei PMU si descriptorii celorlalte aplicatii; octetii ucode-ului si ai semnaturilor (zero)",
    "registrele I6 (siguranta 0x8241e0, VRAM 0x1183a4, WPR2 0x1fa828): inca necitite pe placa",
    "orice registru necitit pe placa: 0 in simulare (placa reala poate da 0xbadf....)",
];

/// SIMULATED configuration space with the observed command register, power
/// state and Resizable BAR entries on top of
/// [`SimConfig::workstation_rtx3070ti`] (ids and BARs from the K0 inventory).
pub fn observed_config() -> SimConfig {
    let mut s = SimConfig::workstation_rtx3070ti();
    let status_and_cmd = (s.read_raw(0x04) & 0xffff_0000) | COMMAND as u32;
    s.write_raw(0x04, status_and_cmd);
    s.set_power_state(POWER_STATE);
    // ReBAR @0x200: entry k at 0x204 + 8k (capability) / 0x208 + 8k (control).
    // Supported sizes are SIM: only the current one.
    for (k, &(bar, log2)) in REBAR_CURRENT.iter().enumerate() {
        let k = k as u16;
        s.write_raw(0x204 + 8 * k, 1u32 << (log2 as u32 + 4));
        let count = if k == 0 {
            (REBAR_CURRENT.len() as u32) << 5
        } else {
            0
        };
        s.write_raw(0x208 + 8 * k, ((log2 as u32) << 8) | count | bar as u32);
    }
    s
}

/// SIMULATED 16 MiB BAR0 with the observed registers and the
/// [`observed_rom`] layout in the PROM window.
pub fn observed_bar0() -> SimMmio {
    let mut m = SimMmio::new(16 << 20);
    m.set(0x0000_0000, BOOT_0);
    m.set(0x0000_0a00, BOOT_42);
    m.set(0x0011_8128, PRIV_LEVEL_MASK);
    m.set(0x0011_8234, GFW_BOOT);
    m.load_bytes(0x30_0000, &observed_rom());
    m
}

fn blocks(size: u32) -> u16 {
    (size / 512) as u16
}

/// A synthetic ROM with the observed layout: the same 4 images at the same
/// offsets and sizes, the BIT table at the same place with the same token
/// ids/versions, the FWSEC PMU table and descriptor where the board has
/// them, with the observed descriptor fields. Every content byte is zero or
/// SIM ([`STILL_SIM`]).
pub fn observed_rom() -> Vec<u8> {
    // SIM: token lengths 4, data offsets in an empty area of image 0; the
    // Falcon data token (0x70) points at FALCON_DATA_AT.
    const FALCON_DATA_AT: usize = 0x3000;
    let tokens = BIT_TOKENS
        .iter()
        .enumerate()
        .map(|(k, &(id, ver))| {
            let data = if id == 0x70 {
                FALCON_DATA_AT as u16
            } else {
                0x2000 + 0x10 * k as u16
            };
            (id, ver, 4u16, data)
        })
        .collect();
    let mut b = build_chain(&[
        SimImage {
            code_type: IMAGES[0].1,
            blocks: blocks(IMAGES[0].2),
            npde: true,
            bit: Some((BIT_AT as usize, tokens)),
        },
        SimImage {
            code_type: IMAGES[1].1,
            blocks: blocks(IMAGES[1].2),
            npde: true,
            bit: None,
        },
        SimImage {
            code_type: IMAGES[2].1,
            blocks: blocks(IMAGES[2].2),
            npde: true,
            bit: None,
        },
        SimImage {
            code_type: IMAGES[3].1,
            blocks: blocks(IMAGES[3].2),
            npde: true,
            bit: None,
        },
    ]);
    // FWSEC images: "NV" signature, "NPDS" data structure, device 0x2200.
    for &(off, _, _, _, dev) in &IMAGES[2..] {
        let off = off as usize;
        set_signature(&mut b, off, 0x4e56);
        let p = off + crate::rom::SIM_PCIR_AT;
        b[p..p + 4].copy_from_slice(b"NPDS");
        b[p + 6..p + 8].copy_from_slice(&dev.to_le_bytes());
    }
    let sections = [
        (IMAGES[2].0 as usize, IMAGES[2].2),
        (IMAGES[3].0 as usize, IMAGES[3].2),
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
    // put_fwsec writes a SIM ucode_id; the board's is UCODE_ID. Byte 38 of
    // the descriptor (see put_fwsec), in the image that holds it.
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
        let b = observed_rom();
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
    }

    #[test]
    fn bar0_and_config_carry_the_observed_values() {
        let mut m = observed_bar0();
        assert_eq!(m.read32_raw(0), BOOT_0);
        assert_eq!(m.read32_raw(0xa00), BOOT_42);
        assert_eq!(m.read32_raw(0x118128), PRIV_LEVEL_MASK);
        assert_eq!(m.read32_raw(0x118234) & 0xff, 0xff);
        let mut c = observed_config();
        assert_eq!(c.read32(0x04) as u16, COMMAND);
        assert_eq!(c.writes, 0);
    }
}
