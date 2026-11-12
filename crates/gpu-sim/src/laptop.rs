//! FACTS about the second test machine, the ASUS TUF Gaming F15 (FX506HM)
//! laptop with an RTX 3060 Laptop GPU (GA106), as reported by two read-only
//! sources (hub `rezultate/2026-11-11-laptop-asus-tuf/REZULTAT.md`):
//! - Cyber Universe OS on 12 Nov 2026 20:20 (`os-295`, `gpu identifica`):
//!   PCI id, bus address, Command, power state, BAR sizes, BAR0 base;
//! - Adrian's UHL inventory + report from Windows (12 Nov 2026 22:14 and
//!   23:03; NVML/CUDA, no serial numbers): subsystem, revision, VBIOS
//!   version string, SM count, L2, VRAM width, BAR1 base, PCIe link, D3cold.
//!
//! These constants describe what those tools reported. They shape the
//! SIMULATED laptop in [`crate::pci::SimConfig::ga106_laptop_asus`]; a test
//! passing against that model still proves nothing about the real laptop.
//! Nothing below was read by our driver from the board's registers yet
//! (Memory Space was off: see docs/GA106_LAPTOP.md §0 and §6).

pub const VENDOR: u16 = 0x10de;
pub const DEVICE: u16 = 0x2520;
pub const SUBSYS_VENDOR: u16 = 0x1043; // ASUS
pub const SUBSYS_DEVICE: u16 = 0x16f2;
/// "rev A1" in the Windows inventory.
pub const REVISION: u8 = 0xa1;
/// PCI 01:00.0 (bus 1, device 0, function 0), behind the Intel PEG root
/// port `\_SB.PC00.PEG1.PEGP` (Tiger Lake-H PEG10, id 8086:9a01).
pub const BUS: u8 = 0x01;
/// The string Windows / NVML show for the VBIOS ("Version94.6.1e.0.23" in
/// the driver registry). The first laptop journal must show the same
/// `vbios.versiune` (read from BIT token 'i'); a mismatch means our reading
/// of the ROM is wrong or the board shows another image.
pub const VBIOS_VERSION: &str = "94.06.1e.00.23";

/// Command 0x0000 as the firmware left it: Memory Space, I/O and Bus Master
/// all off (hybrid/Optimus laptop) — the reason BAR0 did not answer.
pub const COMMAND: u16 = 0x0000;
/// D0 (not put to sleep by the firmware; ACPI not involved).
pub const POWER_STATE: u8 = 0;

pub const BAR0_BASE: u64 = 0x8500_0000;
pub const BAR0_SIZE: u64 = 16 << 20;
/// Resizable BAR on by the firmware: BAR1 = the whole 8 GiB, above 4 GiB.
pub const BAR1_BASE: u64 = 0x60_0000_0000;
pub const BAR1_SIZE: u64 = 8 << 30;
pub const BAR3_SIZE: u64 = 32 << 20;
pub const RESIZABLE_BAR_ENABLED: bool = true;

/// PCIe Gen4 (16 GT/s), x8 active out of x16 maximum; payload 256 B.
pub const LINK_SPEED_GTS: u8 = 16;
pub const LINK_WIDTH_ACTIVE: u8 = 8;
pub const LINK_WIDTH_MAX: u8 = 16;
/// D3cold supported (ACPI; the laptop can cut the board's power — a laptop
/// risk, docs/GA106_LAPTOP.md §3).
pub const D3COLD_SUPPORTED: bool = true;

/// GA106M: compute capability 8.6 (same SASS as GA104), 30 SMs (GA104 on
/// the workstation: 48), L2 3 MiB, 6 GiB VRAM on a 192-bit bus.
pub const COMPUTE_CAPABILITY: (u8, u8) = (8, 6);
pub const SM_COUNT: u32 = 30;
pub const L2_BYTES: u64 = 3 << 20;
pub const VRAM_BYTES: u64 = 6 << 30;
pub const VRAM_BUS_BITS: u32 = 192;
/// Our own limit for the laptop (C-116 step 4), below the 6 GiB of VRAM.
pub const VRAM_BUDGET_BYTES: u64 = 5_500 << 20;

/// Power limits reported by NVML: default 80 W, enforced 95 W, range
/// 1–95 W; idle ~20–26 W at 58–62 °C in P0. Informational only.
pub const POWER_LIMIT_DEFAULT_W: u32 = 80;
pub const POWER_LIMIT_ENFORCED_W: u32 = 95;
