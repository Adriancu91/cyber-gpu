//! C-116 — the "minimal wake-up": Memory Space on (one Command-register
//! write, read back), so a GPU the firmware left with memory decode OFF
//! answers reads through its BARs. Nothing else changes: Bus Master and
//! I/O keep the value found, no GPU register is written, no power state
//! is changed.
//!
//! Why: on Adrian's ASUS TUF F15 (hybrid/Optimus laptop) `gpu identifica`
//! showed the RTX 3060 Laptop (GA106) in D0 with **Command 0x0000**
//! (observed 12 Nov 2026, os-295). With bit 1 clear the function does not
//! decode BAR0, so BOOT_0 / GFW / VBIOS cannot be read; the workstation's
//! firmware leaves the bit on, which is why this never came up there.
//!
//! Sources ("three sources" rule): PCI Local Bus Specification 3.0 §6.2.2
//! (Command register, bit 1 "Memory Space"); Linux `pci_enable_device` /
//! `pci_enable_resources` (sets exactly this bit, after checking that the
//! BARs are assigned) — read as specification; nouveau, nova-core and
//! tinygrad all rely on that call before touching BAR0. Original code.
//!
//! Procedure: read Command; refuse if the function does not answer, is not
//! in D0, is not a type-0 header, or has a memory BAR with address 0 (the
//! firmware did not assign it: decoding would claim address 0). If bit 1
//! is already set, write nothing. Otherwise write the 16-bit Command with
//! only bit 1 added (Status half is RW1C, so 0 is written there) and read
//! it back; a read-back that differs is restored and reported as
//! `VerifyFailed`. [`restore_memory_space`] puts the old value back at the
//! end of the window, so the board is left as the firmware had it.
//!
//! Behind an explicit [`MemorySpacePermit`]: the OS asks Adrian's "DA"
//! and only the test stick (`gpu-scrie`) can construct the writable
//! configuration adapter. SIMULATION only so far: not run on the laptop.

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::ConfigSpace;

use crate::bar::{self, BarKind};
use crate::bus_master::write_checked;
use crate::caps;
use crate::header::{self, CMD_MEMORY, REG_COMMAND};

/// Explicit acknowledgement that the caller lets the function decode its
/// memory BARs (C-116). Carried, never read: its existence is the point.
pub struct MemorySpacePermit {
    pub reason: &'static str,
}

impl MemorySpacePermit {
    pub fn new(reason: &'static str) -> Self {
        MemorySpacePermit { reason }
    }
}

/// The journal line written when the bit was already on.
pub const ALREADY_ON: &str = "Memory Space: deja pornit; nicio scriere";
/// The step line written after the one write.
pub const STEP_ON: &str = "C-116 trezire minima: Memory Space pornit";
/// The step line written after the restore.
pub const STEP_RESTORED: &str = "C-116 trezire minima: Command repus (Memory Space oprit)";

/// Read-only checks that must pass before the one write. Public so the
/// OS can show *why* before asking for "DA".
pub fn check_before_enable<C: ConfigSpace + ?Sized>(cfg: &mut C) -> GpuResult<u16> {
    let dword = cfg.read32(REG_COMMAND);
    if dword == 0xffff_ffff {
        return Err(GpuError::NotResponding);
    }
    if header::header_type(cfg) != 0 {
        return Err(GpuError::Blocked(
            "Memory Space: nu e un antet de tip 0 (endpoint); nu schimb nimic",
        ));
    }
    if caps::power_state(cfg).is_some_and(|d| d != 0) {
        return Err(GpuError::Blocked(
            "Memory Space: placa nu e in D0; trezirea minima nu schimba starea de energie",
        ));
    }
    // `decode_bars` drops BARs that read 0 (unassigned or unimplemented look
    // the same without sizing). BAR0 is always implemented on a GPU, so a
    // missing BAR0 means the firmware did not assign it; for the others the
    // Resizable BAR capability says which BARs exist, so each one it names
    // must have an address too. Decoding an unassigned BAR would claim 0.
    let bars = bar::decode_bars(cfg);
    match bars[0] {
        Some(b) if b.kind != BarKind::Io => {}
        _ => {
            return Err(GpuError::Blocked(
                "Memory Space: BAR0 nu are adresa (nealocat de firmware); decodarea ar revendica adresa 0",
            ));
        }
    }
    for r in caps::resizable_bars(cfg) {
        let i = usize::from(r.bar_index);
        if i >= bars.len() || bars[i].is_none() {
            return Err(GpuError::Blocked(
                "Memory Space: un BAR redimensionabil nu are adresa (nealocat de firmware); nu pornesc decodarea",
            ));
        }
    }
    Ok(dword as u16)
}

/// Turns Memory Space on. Returns the Command value found before, so the
/// caller can put it back with [`restore_memory_space`].
pub fn enable_memory_space<C: ConfigSpace + ?Sized>(
    cfg: &mut C,
    _permit: &MemorySpacePermit,
    j: &mut Journal,
) -> GpuResult<u16> {
    let cmd = check_before_enable(cfg)?;
    j.push(Event::ConfigRead {
        offset: REG_COMMAND,
        value: u32::from(cmd),
    });
    if cmd & CMD_MEMORY != 0 {
        j.push(Event::Note(ALREADY_ON));
        return Ok(cmd);
    }
    let want = cmd | CMD_MEMORY;
    write_checked(cfg, j, cmd, want)?;
    j.push(Event::Step {
        name: STEP_ON,
        ok: true,
    });
    Ok(cmd)
}

/// Puts the Command register back to `old` (as returned by
/// [`enable_memory_space`]); only bit 1 may differ from the current value.
/// Writes nothing when the value is already `old` (the bit was on before).
pub fn restore_memory_space<C: ConfigSpace + ?Sized>(
    cfg: &mut C,
    _permit: &MemorySpacePermit,
    j: &mut Journal,
    old: u16,
) -> GpuResult<()> {
    let dword = cfg.read32(REG_COMMAND);
    if dword == 0xffff_ffff {
        return Err(GpuError::NotResponding);
    }
    let cmd = dword as u16;
    if cmd == old {
        return Ok(());
    }
    if (cmd ^ old) & !CMD_MEMORY != 0 {
        return Err(GpuError::Blocked(
            "COMMAND schimbat si in alti biti decat Memory Space; nu restaurez orbeste",
        ));
    }
    write_checked(cfg, j, cmd, old)?;
    j.push(Event::Step {
        name: STEP_RESTORED,
        ok: true,
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::header::CMD_BUS_MASTER;
    use gpu_sim::pci::SimConfig;

    fn permit() -> MemorySpacePermit {
        MemorySpacePermit::new("test SIMULARE")
    }

    /// SIMULARE: the laptop as the firmware leaves it (Command 0x0000).
    #[test]
    fn laptop_one_write_then_restore_one_write() {
        let mut c = SimConfig::ga106_laptop_asus();
        let status = c.read32(REG_COMMAND) & 0xffff_0000;
        assert_eq!(
            c.read32(REG_COMMAND) & 0xffff,
            0x0000,
            "observed on the laptop"
        );
        let mut j = Journal::new(16);
        let old = enable_memory_space(&mut c, &permit(), &mut j).unwrap();
        assert_eq!(old, 0x0000);
        assert_eq!(c.read32(REG_COMMAND), status | 0x0002);
        assert_eq!(c.writes, 1, "exactly one write");
        assert_eq!(j.writes(), 1);
        assert_eq!(
            c.read32(REG_COMMAND) & u32::from(CMD_BUS_MASTER),
            0,
            "Bus Master stays off"
        );
        assert!(
            j.events()
                .any(|e| matches!(e, Event::Step { name, ok: true } if *name == STEP_ON))
        );
        let line = j
            .events()
            .find(|e| matches!(e, Event::ConfigWrite { .. }))
            .map(|e| alloc::format!("{e}"))
            .unwrap();
        assert_eq!(line, "cfg  w 0x004: 0x00100000 <- 0x00000002 => 0x00100002");
        restore_memory_space(&mut c, &permit(), &mut j, old).unwrap();
        assert_eq!(c.read32(REG_COMMAND), status, "Command back to 0x0000");
        assert_eq!(c.writes, 2);
        assert!(
            j.events()
                .any(|e| matches!(e, Event::Step { name, .. } if *name == STEP_RESTORED))
        );
    }

    #[test]
    fn workstation_already_on_writes_nothing() {
        let mut c = SimConfig::ga104_rtx3070ti(); // SIM: memory decode already on
        let mut j = Journal::new(16);
        let old = enable_memory_space(&mut c, &permit(), &mut j).unwrap();
        assert_eq!(old & CMD_MEMORY, CMD_MEMORY);
        assert_eq!(c.writes, 0);
        assert!(
            j.events()
                .any(|e| matches!(e, Event::Note(n) if *n == ALREADY_ON))
        );
        restore_memory_space(&mut c, &permit(), &mut j, old).unwrap();
        assert_eq!(c.writes, 0);
    }

    #[test]
    fn refused_absent_d3_or_unassigned_bar() {
        let mut j = Journal::new(16);
        let mut c = SimConfig::absent();
        assert!(matches!(
            enable_memory_space(&mut c, &permit(), &mut j),
            Err(GpuError::NotResponding)
        ));
        assert_eq!(c.writes, 0);
        // D3: the laptop dGPU put to sleep.
        let mut c = SimConfig::ga106_laptop_asus();
        c.set_power_state(3);
        assert!(matches!(
            enable_memory_space(&mut c, &permit(), &mut j),
            Err(GpuError::Blocked(_))
        ));
        assert_eq!(c.writes, 0);
        // BAR0 not assigned by the firmware.
        let mut c = SimConfig::ga106_laptop_asus();
        c.write_raw(0x10, 0x0000_0000);
        assert!(matches!(
            enable_memory_space(&mut c, &permit(), &mut j),
            Err(GpuError::Blocked(_))
        ));
        assert_eq!(c.writes, 0);
        // BAR1 (named by Resizable BAR) not assigned: refused too.
        let mut c = SimConfig::ga106_laptop_asus();
        c.write_raw(0x14, 0x0000_000c);
        c.write_raw(0x18, 0x0000_0000);
        assert!(matches!(
            enable_memory_space(&mut c, &permit(), &mut j),
            Err(GpuError::Blocked(_))
        ));
        assert_eq!(c.writes, 0);
        // Not a type-0 header (a bridge): refused.
        let mut c = SimConfig::ga106_laptop_asus();
        c.write_raw(0x0c, 0x0001_0000);
        assert!(matches!(
            enable_memory_space(&mut c, &permit(), &mut j),
            Err(GpuError::Blocked(_))
        ));
        assert_eq!(c.writes, 0);
        assert_eq!(j.writes(), 0);
    }

    #[test]
    fn restore_refuses_other_changed_bits() {
        let mut c = SimConfig::ga106_laptop_asus();
        let mut j = Journal::new(16);
        let old = enable_memory_space(&mut c, &permit(), &mut j).unwrap();
        // Someone turned Bus Master on meanwhile: not ours to undo.
        c.write32(REG_COMMAND, 0x0006);
        c.writes = 0;
        assert!(matches!(
            restore_memory_space(&mut c, &permit(), &mut j, old),
            Err(GpuError::Blocked(_))
        ));
        assert_eq!(c.writes, 0);
    }

    /// Command register whose bit 1 is stuck at 0 (fault injection).
    struct StuckMem(SimConfig);
    impl ConfigSpace for StuckMem {
        fn read32(&mut self, o: u16) -> u32 {
            self.0.read32(o)
        }
        fn write32(&mut self, o: u16, v: u32) {
            self.0.write32(o, if o == REG_COMMAND { v & !2 } else { v })
        }
    }

    #[test]
    fn read_back_mismatch_restores_and_fails() {
        let mut c = StuckMem(SimConfig::ga106_laptop_asus());
        let mut j = Journal::new(16);
        assert!(matches!(
            enable_memory_space(&mut c, &permit(), &mut j),
            Err(GpuError::VerifyFailed { .. })
        ));
        assert_eq!(c.0.read32(REG_COMMAND) & 0xffff, 0x0000);
        assert_eq!(j.writes(), 2, "the write and the restore, both journaled");
    }
}
