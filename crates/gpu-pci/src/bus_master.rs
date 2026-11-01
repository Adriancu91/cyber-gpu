//! I5g — Bus Master on/off (one Command-register write, read back).
//!
//! Bus Master (Command bit 2, PCI Local Bus 3.0 §6.2.2) lets the device read
//! and write system memory by DMA. The GSP falcon needs it from I6 on: it
//! fetches the signed FWSEC image from RAM. Decision D-013: one write, with
//! the value before and after in the journal, behind an explicit permit.
//!
//! Procedure: read Command; refuse if memory decode is off (the BARs would
//! not answer — not our state to change); if the bit already has the wanted
//! value, write nothing; otherwise write the 16-bit Command with only bit 2
//! changed (the Status half is RW1C, so 0 is written there) and read it back.
//! A read-back that differs in any bit is restored to the old value and
//! reported as `VerifyFailed`.

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::ConfigSpace;

use crate::header::{CMD_BUS_MASTER, CMD_MEMORY, REG_COMMAND};

/// Explicit acknowledgement that the caller lets the device do DMA (I5g).
pub struct BusMasterPermit {
    pub reason: &'static str,
}

impl BusMasterPermit {
    pub fn new(reason: &'static str) -> Self {
        BusMasterPermit { reason }
    }
}

/// Sets Bus Master to `on`. Returns the Command value found before, so the
/// caller can put it back with [`restore_command`] at the end of the window.
pub fn set_bus_master<C: ConfigSpace + ?Sized>(
    cfg: &mut C,
    _permit: &BusMasterPermit,
    j: &mut Journal,
    on: bool,
) -> GpuResult<u16> {
    let dword = cfg.read32(REG_COMMAND);
    j.push(Event::ConfigRead {
        offset: REG_COMMAND,
        value: dword,
    });
    if dword == 0xffff_ffff {
        return Err(GpuError::NotResponding);
    }
    let cmd = dword as u16;
    if cmd & CMD_MEMORY == 0 {
        return Err(GpuError::Blocked(
            "Bus Master: decodarea memoriei e oprita; nu schimb starea placii",
        ));
    }
    let want = if on {
        cmd | CMD_BUS_MASTER
    } else {
        cmd & !CMD_BUS_MASTER
    };
    if want == cmd {
        j.push(Event::Note(
            "Bus Master: deja in starea ceruta; nicio scriere",
        ));
        return Ok(cmd);
    }
    write_checked(cfg, j, cmd, want)?;
    j.push(Event::Step {
        name: if on {
            "I5g Bus Master pornit"
        } else {
            "I5g Bus Master oprit"
        },
        ok: true,
    });
    Ok(cmd)
}

/// Puts the Command register back to `old` (as returned by
/// [`set_bus_master`]); only bit 2 may differ from the current value.
pub fn restore_command<C: ConfigSpace + ?Sized>(
    cfg: &mut C,
    _permit: &BusMasterPermit,
    j: &mut Journal,
    old: u16,
) -> GpuResult<()> {
    let cmd = cfg.read32(REG_COMMAND) as u16;
    if cmd == old {
        return Ok(());
    }
    if (cmd ^ old) & !CMD_BUS_MASTER != 0 {
        return Err(GpuError::Blocked(
            "COMMAND schimbat si in alti biti decat Bus Master; nu restaurez orbeste",
        ));
    }
    write_checked(cfg, j, cmd, old)
}

fn write_checked<C: ConfigSpace + ?Sized>(
    cfg: &mut C,
    j: &mut Journal,
    before16: u16,
    want: u16,
) -> GpuResult<()> {
    let before = cfg.read32(REG_COMMAND);
    cfg.write32(REG_COMMAND, u32::from(want));
    let after = cfg.read32(REG_COMMAND);
    j.push(Event::ConfigWrite {
        offset: REG_COMMAND,
        before,
        written: u32::from(want),
        after,
    });
    if after as u16 != want {
        // Put the old value back once; the device is then Faulted for us.
        cfg.write32(REG_COMMAND, u32::from(before16));
        let back = cfg.read32(REG_COMMAND);
        j.push(Event::ConfigWrite {
            offset: REG_COMMAND,
            before: after,
            written: u32::from(before16),
            after: back,
        });
        return Err(GpuError::VerifyFailed {
            offset: u32::from(REG_COMMAND),
            expected: u32::from(want),
            actual: after & 0xffff,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_sim::pci::SimConfig;

    fn permit() -> BusMasterPermit {
        BusMasterPermit::new("test SIMULARE")
    }

    /// SIM board with only memory decode on (Bus Master and I/O off).
    fn board_bm_off() -> SimConfig {
        let mut c = SimConfig::ga104_rtx3070ti();
        c.write32(REG_COMMAND, 0x0002);
        c.writes = 0;
        c
    }

    #[test]
    fn on_then_restore_one_write_each_status_untouched() {
        let mut c = board_bm_off();
        let status = c.read32(REG_COMMAND) & 0xffff_0000;
        let mut j = Journal::new(16);
        let old = set_bus_master(&mut c, &permit(), &mut j, true).unwrap();
        assert_eq!(old, 0x0002);
        assert_eq!(c.read32(REG_COMMAND), status | 0x0006);
        assert_eq!(c.writes, 1);
        assert_eq!(j.writes(), 1);
        restore_command(&mut c, &permit(), &mut j, old).unwrap();
        assert_eq!(c.read32(REG_COMMAND), status | 0x0002);
        assert_eq!(c.writes, 2);
        // The journal line round-trips (what C-51 compares after the window).
        let line = j
            .events()
            .find(|e| matches!(e, Event::ConfigWrite { .. }))
            .map(|e| alloc::format!("{e}"))
            .unwrap();
        assert_eq!(line, "cfg  w 0x004: 0x00100002 <- 0x00000006 => 0x00100006");
    }

    #[test]
    fn already_on_writes_nothing() {
        let mut c = SimConfig::ga104_rtx3070ti(); // SIM: bus master already on
        let mut j = Journal::new(16);
        let old = set_bus_master(&mut c, &permit(), &mut j, true).unwrap();
        assert_eq!(old & CMD_BUS_MASTER, CMD_BUS_MASTER);
        assert_eq!(c.writes, 0);
        restore_command(&mut c, &permit(), &mut j, old).unwrap();
        assert_eq!(c.writes, 0);
    }

    #[test]
    fn refused_without_memory_decode_or_board() {
        let mut c = SimConfig::ga104_rtx3070ti();
        c.write32(REG_COMMAND, 0x0000);
        c.writes = 0;
        let mut j = Journal::new(16);
        assert!(matches!(
            set_bus_master(&mut c, &permit(), &mut j, true),
            Err(GpuError::Blocked(_))
        ));
        assert_eq!(c.writes, 0);
        let mut c = SimConfig::absent();
        assert!(matches!(
            set_bus_master(&mut c, &permit(), &mut j, true),
            Err(GpuError::NotResponding)
        ));
        assert_eq!(c.writes, 0);
    }

    #[test]
    fn restore_refuses_other_changed_bits() {
        let mut c = board_bm_off();
        let mut j = Journal::new(16);
        // Someone turned I/O decode off meanwhile: not ours to undo.
        c.write32(REG_COMMAND, 0x0006);
        c.writes = 0;
        assert!(matches!(
            restore_command(&mut c, &permit(), &mut j, 0x0003),
            Err(GpuError::Blocked(_))
        ));
        assert_eq!(c.writes, 0);
    }

    /// Command register whose bit 2 is stuck at 0 (fault injection).
    struct StuckBm(SimConfig);
    impl ConfigSpace for StuckBm {
        fn read32(&mut self, o: u16) -> u32 {
            self.0.read32(o)
        }
        fn write32(&mut self, o: u16, v: u32) {
            self.0.write32(o, if o == REG_COMMAND { v & !4 } else { v })
        }
    }

    #[test]
    fn read_back_mismatch_restores_and_fails() {
        let mut c = StuckBm(board_bm_off());
        let mut j = Journal::new(16);
        assert!(matches!(
            set_bus_master(&mut c, &permit(), &mut j, true),
            Err(GpuError::VerifyFailed { .. })
        ));
        assert_eq!(c.0.read32(REG_COMMAND) & 0xffff, 0x0002);
        assert_eq!(j.writes(), 2, "the write and the restore, both journaled");
    }
}
