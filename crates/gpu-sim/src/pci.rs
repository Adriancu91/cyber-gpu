//! SIMULATION of a type-0 PCI configuration space shaped like a GA104 board.
//!
//! Layout choices (addresses, capability offsets, ReBAR sizes) are `SIM`
//! values chosen to exercise the code paths; BAR sizes follow publicly
//! reported GA104 boards (docs/GA104_INITIALIZATION.md §2).

use alloc::vec;
use alloc::vec::Vec;

use gpu_core::hal::ConfigSpace;

const DWORDS: usize = 1024; // 4 KiB ECAM page

pub struct SimConfig {
    regs: Vec<u32>,
    /// Per BAR slot: writable mask and read-only type bits.
    bar_mask: [u32; 6],
    bar_ro: [u32; 6],
    absent: bool,
    /// Number of write32 calls seen.
    pub writes: usize,
    /// Set if a BAR was written while memory or I/O decode was enabled.
    pub decode_was_enabled_during_bar_write: bool,
    /// Fault injection: this BAR slot keeps all-ones and ignores the restore.
    pub fault_sticky_bar: Option<usize>,
}

impl SimConfig {
    /// SIM: RTX 3070 Ti (10de:2482) with BAR0 16 MiB, BAR1 256 MiB 64-bit
    /// prefetchable, BAR3 32 MiB 64-bit prefetchable, BAR5 128 B I/O.
    pub fn ga104_rtx3070ti() -> Self {
        let mut regs = vec![0u32; DWORDS];
        let mut set = |off: usize, v: u32| regs[off / 4] = v;
        set(0x00, 0x2482_10de);
        set(0x04, 0x0010_0007); // status: cap list; command: io + mem + bus master
        set(0x08, 0x0300_00a1); // VGA-compatible display controller, rev a1 (SIM)
        set(0x0c, 0x0080_0000); // multi-function (HDA audio on .1), header type 0
        set(0x10, 0xf600_0000); // BAR0
        set(0x14, 0xe000_000c); // BAR1 lo: mem64 prefetchable
        set(0x18, 0x0000_0000); // BAR1 hi
        set(0x1c, 0xf000_000c); // BAR3 lo
        set(0x20, 0x0000_0000); // BAR3 hi
        set(0x24, 0x0000_e001); // BAR5: I/O
        set(0x2c, 0x0000_0000); // subsystem (SIM)
        set(0x34, 0x60);
        // Capabilities: PM @0x60 -> MSI @0x68 -> PCIe @0x78
        set(0x60, 0x0003_6801);
        set(0x64, 0x0000_0000); // PMCSR: D0
        set(0x68, 0x0080_7805);
        set(0x78, 0x0002_0010); // PCIe cap v2, endpoint, next 0
        set(0x84, 0x0000_0104); // LinkCap: 16 GT/s x16
        set(0x88, 0x0104_0000); // LinkSta: 16 GT/s x16
        // Extended: AER @0x100 -> ReBAR @0x200
        set(0x100, 0x2001_0001);
        set(0x200, 0x0001_0015);
        set(0x204, ((1u32 << 14) - (1 << 6)) << 4); // BAR1: 64 MiB .. 8 GiB
        set(0x208, 0x0000_0841); // BAR1, 2 resizable BARs, current 2^8 MiB
        set(0x20c, (1u32 << 5) << 4); // BAR3: 32 MiB
        set(0x210, 0x0000_0503); // BAR3, current 2^5 MiB
        SimConfig {
            regs,
            bar_mask: [
                0xff00_0000,
                0xf000_0000,
                0xffff_ffff,
                0xfe00_0000,
                0xffff_ffff,
                0xffff_ff80,
            ],
            bar_ro: [0x0, 0xc, 0x0, 0xc, 0x0, 0x1],
            absent: false,
            writes: 0,
            decode_was_enabled_during_bar_write: false,
            fault_sticky_bar: None,
        }
    }

    /// SIM: an empty slot, every read returns all ones.
    pub fn absent() -> Self {
        let mut s = Self::ga104_rtx3070ti();
        s.absent = true;
        s
    }

    pub fn snapshot(&self) -> Vec<u32> {
        self.regs.clone()
    }

    /// Corrupt both capability lists into loops.
    pub fn make_cap_loop(&mut self) {
        self.regs[0x78 / 4] = 0x0002_6010; // PCIe -> PM again
        self.regs[0x200 / 4] = 0x1001_0015; // ReBAR -> AER (backwards)
    }

    pub fn set_device_id(&mut self, device: u16) {
        self.regs[0] = ((device as u32) << 16) | (self.regs[0] & 0xffff);
    }

    pub fn set_power_state(&mut self, d: u8) {
        self.regs[0x64 / 4] = d as u32 & 3;
    }
}

impl ConfigSpace for SimConfig {
    fn read32(&mut self, offset: u16) -> u32 {
        if self.absent {
            return 0xffff_ffff;
        }
        self.regs[(offset as usize & 0xffc) / 4]
    }

    fn write32(&mut self, offset: u16, value: u32) {
        self.writes += 1;
        if self.absent {
            return;
        }
        let off = offset as usize & 0xffc;
        match off {
            0x04 => {
                // Command is RW (low half); status is RW1C, writing 0 keeps it.
                let status = self.regs[1] & 0xffff_0000;
                self.regs[1] = status | (value & 0xffff);
            }
            0x10..=0x24 => {
                let slot = (off - 0x10) / 4;
                if self.regs[1] & 0x3 != 0 {
                    self.decode_was_enabled_during_bar_write = true;
                }
                if self.fault_sticky_bar == Some(slot) && value != 0xffff_ffff {
                    return;
                }
                self.regs[off / 4] = (value & self.bar_mask[slot]) | self.bar_ro[slot];
            }
            // Everything else in this model is read-only.
            _ => {}
        }
    }
}
