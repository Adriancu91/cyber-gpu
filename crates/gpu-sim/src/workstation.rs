//! FACTS about the one physical target board, from the K0 inventory run by
//! Adrian on 2026-10-23 09:33 (Windows, read-only script
//! `tools/k0-inventar-workstation.ps1`) and the BIOS photos of 09:45
//! (hub: `rezultate/2026-10-23-k0-workstation/`,
//! `rezultate/2026-10-23-workstation-os/`). See docs/HARDWARE_WORKSTATION.md.
//!
//! These constants describe what Windows reported. They are used to build
//! the SIMULATED configuration space in [`crate::pci`]; a test that passes
//! against that simulation still proves nothing about the real board.

pub const VENDOR: u16 = 0x10de;
pub const DEVICE: u16 = 0x2482;
pub const SUBSYS_VENDOR: u16 = 0x1043; // ASUS
pub const SUBSYS_DEVICE: u16 = 0x8813;
/// "rev A1" in the inventory.
pub const REVISION: u8 = 0xa1;
/// PCI bus 0xB3 (179), device 0, function 0 — on a separate root bus of the
/// Cascade Lake-X IIO stacks, not reachable through bridges from bus 0.
pub const BUS: u8 = 0xb3;
pub const VBIOS_VERSION: &str = "94.04.65.00.3e";

pub const BAR0_BASE: u64 = 0xf900_0000;
pub const BAR0_SIZE: u64 = 16 << 20;
pub const BAR1_BASE: u64 = 0xe000_0000;
pub const BAR1_SIZE: u64 = 256 << 20;
pub const BAR3_BASE: u64 = 0xf000_0000;
pub const BAR3_SIZE: u64 = 32 << 20;
pub const IO_BASE: u64 = 0xf000;
pub const IO_SIZE: u64 = 0x80;

/// Firmware settings seen in the BIOS photos / inventory.
pub const RESIZABLE_BAR_ENABLED: bool = false;
/// All BARs below 4 GiB; "Above 4G Decoding" probably off (not photographed).
pub const ALL_BARS_BELOW_4G: bool = true;
pub const VTD_ENABLED: bool = false;
/// Off for the OS test only; Adrian turns it back on for Windows.
pub const SECURE_BOOT_OFF_FOR_TEST: bool = true;
