//! SIMULARE: BAR0 as the I9b code sees it — the 1 MiB VRAM window ("PRAMIN")
//! that `NV_PBUS_BAR0_WINDOW` points somewhere in VRAM, and the MMU TLB
//! invalidate registers — over the sparse [`SimVram`].
//!
//! Offsets and fields are written **by hand** (an independent reading, not the
//! generated constants of `nvidia-rm570`) from NVIDIA's MIT headers in
//! open-gpu-kernel-modules 570.144: `maxwell/gm107/dev_bus.h`
//! (`NV_PBUS_BAR0_WINDOW` = 0x1700, BASE 23:0 in 64 KiB units, TARGET 25:24,
//! 0 = video memory), `turing/tu102/dev_ram.h` (`NV_PRAMIN` =
//! 0x700000..0x7fffff), `turing/tu102/dev_vm.h` (the PF copy of the
//! "virtual function" registers at 0xb80000; `MMU_INVALIDATE_PDB` 0x30a0,
//! `_UPPER_PDB` 0x30a4, `MMU_INVALIDATE` 0x30b0 with ALL_VA bit 0, ALL_PDB
//! bit 1, TRIGGER bit 31).
//!
//! Nothing here says how the real board behaves (whether the window is
//! enabled after UEFI, how long an invalidate takes): it records what the
//! driver did so tests can check it, and it can play three faults.

use alloc::vec::Vec;
use gpu_core::hal::Mmio;

use crate::vram::SimVram;

/// `NV_PBUS_BAR0_WINDOW`.
pub const BAR0_WINDOW: u64 = 0x1700;
/// First byte of the window in BAR0, and its size.
pub const PRAMIN: u64 = 0x70_0000;
pub const PRAMIN_LEN: u64 = 0x10_0000;
const VF: u64 = 0xb8_0000;
pub const INVALIDATE_PDB: u64 = VF + 0x30a0;
pub const INVALIDATE_UPPER_PDB: u64 = VF + 0x30a4;
pub const INVALIDATE: u64 = VF + 0x30b0;
const TRIGGER: u32 = 1 << 31;
/// What a register outside this model reads as (a "priv error").
const PRIV_ERROR: u32 = 0xbadf_1100;

/// One TLB invalidate the driver started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Invalidate {
    /// The page directory named in `_PDB` / `_UPPER_PDB`, as a byte address.
    pub pdb: u64,
    /// `_PDB_APERTURE` (0 = video memory).
    pub aperture: u32,
    pub all_va: bool,
    pub all_pdb: bool,
    /// VRAM words written through the window before this invalidate.
    pub after_words: u64,
}

/// Faults the model can play.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Faults {
    /// TRIGGER never clears (the invalidate never finishes).
    pub stuck_invalidate: bool,
    /// A write through the window to this VRAM byte address is lost.
    pub lost_write: Option<u64>,
    /// The window register ignores writes (stays where it was).
    pub window_frozen: bool,
}

/// BAR0 (16 MiB) with only the I9b registers alive.
pub struct SimBar0Vram {
    pub vram: SimVram,
    pub faults: Faults,
    window: u32,
    pdb: u32,
    upper_pdb: u32,
    stuck: bool,
    /// Every invalidate started, in order.
    pub invalidates: Vec<Invalidate>,
    /// 32-bit VRAM writes made through the window.
    pub words: u64,
    /// Writes to any other register (tests expect none).
    pub stray_writes: Vec<(u64, u32)>,
}

impl SimBar0Vram {
    /// `window` = the value `NV_PBUS_BAR0_WINDOW` holds before the driver
    /// touches it (whatever UEFI left; unknown on the real board).
    pub fn new(vram: SimVram, window: u32) -> Self {
        SimBar0Vram {
            vram,
            faults: Faults::default(),
            window,
            pdb: 0,
            upper_pdb: 0,
            stuck: false,
            invalidates: Vec::new(),
            words: 0,
            stray_writes: Vec::new(),
        }
    }

    pub fn window_register(&self) -> u32 {
        self.window
    }

    /// The VRAM byte address behind BAR0 offset `off` of the window, if the
    /// window points to video memory.
    fn vram_address(&self, off: u64) -> Option<u64> {
        let target = (self.window >> 24) & 3;
        if target != 0 {
            return None;
        }
        Some((u64::from(self.window & 0x00ff_ffff) << 16) + (off - PRAMIN))
    }

    fn read_vram32(&self, pa: u64) -> u32 {
        match self.vram.read_u64(pa & !7) {
            Ok(w) => (w >> ((pa & 4) * 8)) as u32,
            Err(_) => PRIV_ERROR,
        }
    }

    fn write_vram32(&mut self, pa: u64, v: u32) {
        self.words += 1;
        if self.faults.lost_write == Some(pa) {
            return;
        }
        let Ok(old) = self.vram.read_u64(pa & !7) else {
            return;
        };
        let shift = (pa & 4) * 8;
        let new = (old & !(0xffff_ffffu64 << shift)) | (u64::from(v) << shift);
        let _ = self.vram.write_u64(pa & !7, new);
    }
}

impl Mmio for SimBar0Vram {
    fn len(&self) -> u64 {
        16 << 20
    }

    fn read32_raw(&mut self, off: u64) -> u32 {
        match off {
            BAR0_WINDOW => self.window,
            o if (PRAMIN..PRAMIN + PRAMIN_LEN).contains(&o) => match self.vram_address(o) {
                Some(pa) => self.read_vram32(pa),
                None => PRIV_ERROR,
            },
            INVALIDATE_PDB => self.pdb,
            INVALIDATE_UPPER_PDB => self.upper_pdb,
            // TRIGGER reads back as 1 while the invalidate is pending.
            INVALIDATE => {
                if self.stuck {
                    TRIGGER
                } else {
                    0
                }
            }
            _ => PRIV_ERROR,
        }
    }

    fn write32_raw(&mut self, off: u64, v: u32) {
        match off {
            BAR0_WINDOW => {
                if !self.faults.window_frozen {
                    self.window = v & 0x03ff_ffff;
                }
            }
            o if (PRAMIN..PRAMIN + PRAMIN_LEN).contains(&o) => {
                if let Some(pa) = self.vram_address(o) {
                    self.write_vram32(pa, v);
                }
            }
            INVALIDATE_PDB => self.pdb = v,
            INVALIDATE_UPPER_PDB => self.upper_pdb = v & 0xf_ffff,
            INVALIDATE => {
                if v & TRIGGER != 0 {
                    let page = (u64::from(self.upper_pdb) << 28) | u64::from(self.pdb >> 4);
                    self.invalidates.push(Invalidate {
                        pdb: page << 12,
                        aperture: (self.pdb >> 1) & 1,
                        all_va: v & 1 != 0,
                        all_pdb: v & 2 != 0,
                        after_words: self.words,
                    });
                    self.stuck = self.faults.stuck_invalidate;
                }
            }
            _ => self.stray_writes.push((off, v)),
        }
    }

    fn is_simulation(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_reads_and_writes_land_in_vram() {
        let mut b = SimBar0Vram::new(SimVram::new(1 << 30), 0);
        b.write32_raw(BAR0_WINDOW, 0x30); // 0x30 << 16 = 3 MiB
        b.write32_raw(PRAMIN + 0x14, 0xdead_beef);
        assert_eq!(b.vram.read_u64(0x30_0010).unwrap() >> 32, 0xdead_beef);
        assert_eq!(b.read32_raw(PRAMIN + 0x14), 0xdead_beef);
        // The other half of the word was never written: still poison.
        assert_eq!(b.read32_raw(PRAMIN + 0x10), 0xa5a5_a5a5);
        // A window pointed at system memory is not modelled: priv error.
        b.write32_raw(BAR0_WINDOW, 2 << 24);
        assert_eq!(b.read32_raw(PRAMIN), PRIV_ERROR);
        b.write32_raw(0x1234, 1);
        assert_eq!(b.stray_writes, [(0x1234, 1)]);
    }
}
