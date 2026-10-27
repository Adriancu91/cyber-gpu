//! I9b: writing the page tables into the board's VRAM through the BAR0 VRAM
//! window ("PRAMIN"), then the MMU TLB invalidate. **Tested only in SIMULARE**
//! (`gpu_sim::pramin`); never run on the RTX 3070 Ti.
//!
//! Sources (read as specification, code written by us, docs/PROVENANCE.md):
//! - registers generated from NVIDIA's MIT headers, open-gpu-kernel-modules
//!   570.144 (`nvidia-rm570`): `gm107/dev_bus.h` (`NV_PBUS_BAR0_WINDOW`),
//!   `tu102/dev_ram.h` (`NV_PRAMIN`), `tu102/dev_vm.h` (`MMU_INVALIDATE*`,
//!   `NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET`);
//! - that GA104 uses those layouts: RM 570.144's HAL tables
//!   (`g_kern_bus_nvoc.c`: `kbusSetBAR0WindowVidOffset_GM107` on TU102..AD107;
//!   `g_kern_gmmu_nvoc.c`: `kgmmuSetPdbToInvalidate_TU102`,
//!   `kgmmuCommitTlbInvalidate_TU102`, `kgmmuCheckPendingInvalidates_TU102`;
//!   `kern_gpu_tu102.c`: the PF reaches the "virtual function" registers at
//!   `NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET`), MIT;
//! - DEDUS: the invalidate value (ALL_VA, one PDB, cache level ALL) — RM's
//!   general case; tinygrad (`nvdev.py`, MIT) also invalidates after mapping.
//!   Whether the window works right after UEFI hands over: unknown until B0.
//!
//! Write policy (stricter than `gsp::falcon::GuardedBar0`, which it does not
//! widen): this module can write exactly four registers — the window and the
//! three invalidate registers — and VRAM through the window, never inside
//! the ranges given as protected (the screen's framebuffer, GSP-RM's
//! regions). Every register write is journaled with the value before and
//! after; every VRAM word is read back (a mismatch is an error, not a retry).
//! On a real BAR it needs a [`HardwareWritePermit`] (`hw-write`, off by
//! default; only in the workstation window with Adrian present).

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, Mmio, checked_read32, checked_write32, poll_until};
use nvidia_rm570::regs::gm107_dev_bus as bus;
use nvidia_rm570::regs::tu102_dev_ram as ram;
use nvidia_rm570::regs::tu102_dev_vm as vm;

use crate::gsp::HardwareWritePermit;
use crate::mmu::VramWrite;
use crate::regs::is_priv_error;

/// BAR0 offset of a "virtual function" register as the PF sees it.
const fn vreg(off: u32) -> u32 {
    vm::NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET.lo + off
}

/// The only registers this module writes.
pub const CONTROL_REGISTERS: [u32; 4] = [
    bus::NV_PBUS_BAR0_WINDOW,
    vreg(vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PDB),
    vreg(vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_UPPER_PDB),
    vreg(vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE),
];

/// Size of the window (1 MiB) and the step its base moves in (64 KiB).
pub const WINDOW_LEN: u64 = (ram::NV_PRAMIN.hi - ram::NV_PRAMIN.lo + 1) as u64;
pub const WINDOW_STEP: u64 = 1 << bus::NV_PBUS_BAR0_WINDOW_BASE_SHIFT;

/// How long an invalidate may take before we give up (DEDUS; RM uses its
/// general GPU timeout, seconds).
pub const INVALIDATE_TIMEOUT_US: u64 = 100_000;

/// VRAM seen through the BAR0 window, with the write policy above.
pub struct VramWindow<'a, M: Mmio + ?Sized> {
    mmio: &'a mut M,
    journal: &'a mut Journal,
    vram_size: u64,
    protected: Vec<(u64, u64)>,
    /// The window register as found (restored by [`Self::restore`]).
    original: u32,
    /// The VRAM address the window currently starts at, once we set it.
    base: Option<u64>,
    /// 32-bit VRAM words written and verified.
    pub words: u64,
}

impl<'a, M: Mmio + ?Sized> VramWindow<'a, M> {
    /// `protected` are `[start, end)` VRAM ranges never to be written (the
    /// screen's framebuffer — learned in window B0 —, GSP-RM's regions). A
    /// model from `gpu-sim` needs no permit; a real BAR always does.
    pub fn new(
        mmio: &'a mut M,
        journal: &'a mut Journal,
        permit: Option<&HardwareWritePermit>,
        vram_size: u64,
        protected: &[(u64, u64)],
    ) -> GpuResult<Self> {
        if !mmio.is_simulation() && permit.is_none() {
            return Err(GpuError::NotPermitted(
                "scriere in placa reala fara permis (optiunea hw-write, doar cu Adrian de fata)",
            ));
        }
        if protected.iter().any(|&(s, e)| e <= s) {
            return Err(GpuError::InvalidArgument("VRAM: zona protejata goala"));
        }
        if let Some(p) = permit {
            journal.push(Event::Note(p.reason));
        }
        let original = checked_read32(mmio, u64::from(bus::NV_PBUS_BAR0_WINDOW))?;
        if is_priv_error(original) {
            return Err(GpuError::HardwareFault { code: original });
        }
        Ok(VramWindow {
            mmio,
            journal,
            vram_size,
            protected: protected.to_vec(),
            original,
            base: None,
            words: 0,
        })
    }

    /// Writes one control register, journaled with before/after.
    fn write_reg(&mut self, off: u32, v: u32) -> GpuResult<u32> {
        debug_assert!(CONTROL_REGISTERS.contains(&off));
        let before = checked_read32(self.mmio, u64::from(off))?;
        checked_write32(self.mmio, u64::from(off), v)?;
        let after = checked_read32(self.mmio, u64::from(off))?;
        self.journal.push(Event::MmioWrite {
            bar: 0,
            offset: u64::from(off),
            before,
            written: v,
            after,
        });
        if is_priv_error(after) {
            return Err(GpuError::HardwareFault { code: after });
        }
        Ok(after)
    }

    /// Points the window at `base` (64 KiB aligned) in video memory.
    fn set_base(&mut self, base: u64) -> GpuResult<()> {
        let units = base / WINDOW_STEP;
        let f = bus::NV_PBUS_BAR0_WINDOW_BASE;
        if units > u64::from(f.mask() >> f.lo) {
            return Err(GpuError::InvalidArgument(
                "VRAM: adresa prea mare pentru fereastra",
            ));
        }
        let cur = checked_read32(self.mmio, u64::from(bus::NV_PBUS_BAR0_WINDOW))?;
        let v = bus::NV_PBUS_BAR0_WINDOW_BASE.set(
            bus::NV_PBUS_BAR0_WINDOW_TARGET.set(cur, bus::NV_PBUS_BAR0_WINDOW_TARGET_VID_MEM),
            units as u32,
        );
        let after = self.write_reg(bus::NV_PBUS_BAR0_WINDOW, v)?;
        if after != v {
            // The window did not move: writing now would land elsewhere.
            self.base = None;
            return Err(GpuError::HardwareFault { code: after });
        }
        self.base = Some(base);
        Ok(())
    }

    fn check(&self, pa: u64) -> GpuResult<()> {
        if !pa.is_multiple_of(4) || pa.checked_add(4).is_none_or(|e| e > self.vram_size) {
            return Err(GpuError::InvalidArgument(
                "VRAM: adresa nealiniata sau in afara memoriei",
            ));
        }
        if self.protected.iter().any(|&(s, e)| pa + 4 > s && pa < e) {
            return Err(GpuError::NotPermitted(
                "VRAM: zona protejata (ecranul sau GSP-RM)",
            ));
        }
        Ok(())
    }

    /// BAR0 offset of `pa`, moving the window if `pa` is outside it.
    fn offset_of(&mut self, pa: u64) -> GpuResult<u64> {
        let base = match self.base {
            Some(b) if pa >= b && pa + 4 <= b + WINDOW_LEN => b,
            _ => {
                let b = pa - pa % WINDOW_STEP;
                self.set_base(b)?;
                b
            }
        };
        Ok(u64::from(ram::NV_PRAMIN.lo) + (pa - base))
    }

    pub fn read_u32(&mut self, pa: u64) -> GpuResult<u32> {
        if !pa.is_multiple_of(4) || pa.checked_add(4).is_none_or(|e| e > self.vram_size) {
            return Err(GpuError::InvalidArgument(
                "VRAM: adresa nealiniata sau in afara memoriei",
            ));
        }
        let off = self.offset_of(pa)?;
        checked_read32(self.mmio, off)
    }

    /// Writes one 32-bit word of VRAM and reads it back.
    pub fn write_u32(&mut self, pa: u64, v: u32) -> GpuResult<()> {
        self.check(pa)?;
        let off = self.offset_of(pa)?;
        checked_write32(self.mmio, off, v)?;
        let back = checked_read32(self.mmio, off)?;
        if back != v {
            return Err(GpuError::HardwareFault { code: back });
        }
        self.words += 1;
        Ok(())
    }

    /// Invalidates the MMU's cached translations of the page directory at
    /// `pdb` (video memory) for every VA, and waits until the MMU is done.
    pub fn invalidate_tlb<C: Clock + ?Sized>(&mut self, clock: &mut C, pdb: u64) -> GpuResult<()> {
        let align = vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PDB_ADDR_ALIGNMENT;
        if !pdb.is_multiple_of(1 << align) || pdb >= self.vram_size {
            return Err(GpuError::InvalidArgument("TLB: director de pagini gresit"));
        }
        let page = pdb >> align;
        let addr = vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PDB_ADDR;
        let lo = addr.set(
            vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PDB_APERTURE.set(
                0,
                vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PDB_APERTURE_VID_MEM,
            ),
            (page & ((1 << addr.width()) - 1)) as u32,
        );
        let hi = vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_UPPER_PDB_ADDR
            .set(0, (page >> addr.width()) as u32);
        self.write_reg(vreg(vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_PDB), lo)?;
        self.write_reg(
            vreg(vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_UPPER_PDB),
            hi,
        )?;
        let mut cmd = vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_ALL_VA
            .set(0, vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_ALL_VA_TRUE);
        cmd = vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_ALL_PDB.set(
            cmd,
            vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_ALL_PDB_FALSE,
        );
        cmd = vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_CACHE_LEVEL.set(
            cmd,
            vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_CACHE_LEVEL_ALL,
        );
        cmd = vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_TRIGGER.set(
            cmd,
            vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_TRIGGER_TRUE,
        );
        let reg = vreg(vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE);
        self.write_reg(reg, cmd)?;
        let mmio = &mut *self.mmio;
        poll_until(clock, INVALIDATE_TIMEOUT_US, 10, || {
            let v = checked_read32(mmio, u64::from(reg)).ok()?;
            let pending = vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_TRIGGER.get(v)
                == vm::NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE_TRIGGER_TRUE;
            (!pending).then_some(())
        })
    }

    /// Puts the window register back to what it was before us.
    pub fn restore(&mut self) -> GpuResult<()> {
        let cur = checked_read32(self.mmio, u64::from(bus::NV_PBUS_BAR0_WINDOW))?;
        if cur != self.original {
            self.write_reg(bus::NV_PBUS_BAR0_WINDOW, self.original)?;
        }
        self.base = None;
        Ok(())
    }
}

/// The VRAM range the screen uses, from what the OS prints in `hw` (C-26,
/// os-192: the framebuffer's physical address and size, from Limine) and
/// the board's BAR1 (`lspci` / `gpu identifica`), rounded out to 64 KiB.
/// `None` if the framebuffer is not inside BAR1 (then it is not in this
/// board's VRAM, or BAR1 is not a plain view of VRAM).
///
/// DEDUS: UEFI GOP leaves BAR1 as a linear view of VRAM from its start, so
/// VRAM address = physical address − BAR1 base. To confirm with the B0
/// numbers; until then [`VramWindow`] callers keep a wider margin.
pub fn screen_vram_range(
    fb_phys: u64,
    fb_size: u64,
    bar1_base: u64,
    bar1_len: u64,
) -> Option<(u64, u64)> {
    let end = fb_phys.checked_add(fb_size)?;
    if fb_size == 0 || fb_phys < bar1_base || end > bar1_base.checked_add(bar1_len)? {
        return None;
    }
    let start = (fb_phys - bar1_base) / WINDOW_STEP * WINDOW_STEP;
    Some((start, (end - bar1_base).next_multiple_of(WINDOW_STEP)))
}

impl<M: Mmio + ?Sized> VramWrite for VramWindow<'_, M> {
    /// Little-endian: the low word at `pa`, the high word at `pa + 4`.
    fn write_u64(&mut self, pa: u64, value: u64) -> GpuResult<()> {
        if !pa.is_multiple_of(8) {
            return Err(GpuError::InvalidArgument(
                "VRAM: scriere de 8 octeti nealiniata",
            ));
        }
        self.write_u32(pa, value as u32)?;
        self.write_u32(pa + 4, (value >> 32) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mmu::{GpuMemory, PageAttr};
    use gpu_sim::clock::SimClock;
    use gpu_sim::pramin::{self as sim, SimBar0Vram};
    use gpu_sim::vram::{SimVram, translate};

    const VRAM: u64 = 8 << 30;
    /// SIMULARE stand-in for the UEFI framebuffer (its real place: window B0).
    const SCREEN: (u64, u64) = (0, 16 << 20);

    #[test]
    fn registers_match_the_hand_reading() {
        assert_eq!(
            CONTROL_REGISTERS,
            [
                sim::BAR0_WINDOW as u32,
                sim::INVALIDATE_PDB as u32,
                sim::INVALIDATE_UPPER_PDB as u32,
                sim::INVALIDATE as u32
            ]
        );
        assert_eq!(u64::from(ram::NV_PRAMIN.lo), sim::PRAMIN);
        assert_eq!(WINDOW_LEN, sim::PRAMIN_LEN);
        assert_eq!(WINDOW_STEP, 0x1_0000);
    }

    /// The I9a page tables, written into (simulated) VRAM through the window,
    /// read back by the independent walker, then one invalidate of the root.
    #[test]
    fn page_tables_through_the_window_then_invalidate() {
        let mut bar = SimBar0Vram::new(SimVram::new(VRAM), 0x0000_0012);
        let mut j = Journal::new(64);
        let mut clock = SimClock::default();
        let w = VramWindow::new(&mut bar, &mut j, None, VRAM, &[SCREEN]).unwrap();
        let mut mem = GpuMemory::new(64 << 20, VRAM - (256 << 20), w).unwrap();
        let va = 0x10_0000_0000;
        let pa = 0x1_2340_0000;
        mem.map(va, pa, 3 << 20, PageAttr::default()).unwrap();
        let root = mem.root();
        let table_writes = mem.writes;
        let words = mem.writer().words;
        assert_eq!(words, 2 * table_writes);
        mem.writer_mut().invalidate_tlb(&mut clock, root).unwrap();
        mem.writer_mut().restore().unwrap();
        drop(mem);

        assert!(bar.stray_writes.is_empty());
        assert_eq!(bar.words, words);
        // Every mapped page lands where the driver intended (walker = hand reading).
        for i in 0..(3 << 20) / 0x1000 {
            let t = translate(&bar.vram, root, va + i * 0x1000 + 8).unwrap();
            assert_eq!(t.pa, pa + i * 0x1000 + 8);
        }
        // One invalidate, after all the table writes, naming our root in VRAM.
        assert_eq!(bar.invalidates.len(), 1);
        let inv = bar.invalidates[0];
        assert_eq!(
            (inv.pdb, inv.aperture, inv.all_va, inv.all_pdb),
            (root, 0, true, false)
        );
        assert_eq!(inv.after_words, words);
        // Nothing written in the protected range; window put back.
        assert!(!bar.vram.touched(SCREEN.0, SCREEN.1 - SCREEN.0));
        assert_eq!(bar.window_register(), 0x12);
        // Register writes journaled: window moves + 3 invalidate registers + restore.
        assert!(j.writes() >= 5);
    }

    #[test]
    fn protected_and_out_of_range_writes_are_refused_before_any_access() {
        let mut bar = SimBar0Vram::new(SimVram::new(VRAM), 0);
        let mut j = Journal::new(16);
        let mut w = VramWindow::new(&mut bar, &mut j, None, VRAM, &[SCREEN]).unwrap();
        assert!(matches!(
            w.write_u32(0x10_0000, 1),
            Err(GpuError::NotPermitted(_))
        ));
        assert!(matches!(
            w.write_u32(SCREEN.1 - 2, 1),
            Err(GpuError::InvalidArgument(_))
        ));
        assert!(matches!(
            w.write_u32(SCREEN.1 - 4, 1),
            Err(GpuError::NotPermitted(_))
        ));
        assert!(w.write_u32(VRAM, 1).is_err());
        assert!(w.write_u64(SCREEN.1 + 4, 1).is_err());
        assert_eq!(w.words, 0);
        drop(w);
        assert_eq!(bar.words, 0);
        assert_eq!(bar.window_register(), 0, "window never moved");
        assert!(VramWindow::new(&mut bar, &mut j, None, VRAM, &[(5, 5)]).is_err());
    }

    #[test]
    fn faults_are_reported_not_hidden() {
        // A lost write is caught by the read-back.
        let mut bar = SimBar0Vram::new(SimVram::new(VRAM), 0);
        bar.faults.lost_write = Some(0x400_0004);
        let mut j = Journal::new(16);
        let mut w = VramWindow::new(&mut bar, &mut j, None, VRAM, &[]).unwrap();
        w.write_u32(0x400_0000, 7).unwrap();
        assert!(matches!(
            w.write_u32(0x400_0004, 7),
            Err(GpuError::HardwareFault { .. })
        ));
        drop(w);

        // A window that does not move: refused before writing anywhere.
        let mut bar = SimBar0Vram::new(SimVram::new(VRAM), 0);
        bar.faults.window_frozen = true;
        let mut w = VramWindow::new(&mut bar, &mut j, None, VRAM, &[]).unwrap();
        assert!(matches!(
            w.write_u32(0x400_0000, 7),
            Err(GpuError::HardwareFault { .. })
        ));
        drop(w);
        assert_eq!(bar.words, 0);

        // An invalidate that never finishes: a timeout, never success.
        let mut bar = SimBar0Vram::new(SimVram::new(VRAM), 0);
        bar.faults.stuck_invalidate = true;
        let mut clock = SimClock::default();
        let mut w = VramWindow::new(&mut bar, &mut j, None, VRAM, &[]).unwrap();
        assert!(matches!(
            w.invalidate_tlb(&mut clock, 0x400_0000),
            Err(GpuError::Timeout { .. })
        ));
        assert!(
            w.invalidate_tlb(&mut clock, 0x400_0010).is_err(),
            "unaligned PDB"
        );
    }

    /// A planted mistake in the invalidate (wrong page number) is caught.
    #[test]
    fn upper_pdb_bits_are_split_like_the_header_says() {
        let mut bar = SimBar0Vram::new(SimVram::new(1 << 45), 0);
        let mut j = Journal::new(16);
        let mut clock = SimClock::default();
        let pdb = (1u64 << 40) + 0x5000; // needs _UPPER_PDB
        let mut w = VramWindow::new(&mut bar, &mut j, None, 1 << 45, &[]).unwrap();
        w.invalidate_tlb(&mut clock, pdb).unwrap();
        drop(w);
        assert_eq!(bar.invalidates[0].pdb, pdb);
    }

    #[test]
    fn screen_range_from_the_hw_line() {
        // Workstation (K0): BAR1 256 MiB; a 1920x1080x4 framebuffer at BAR1 + 0.
        let bar1 = 0xc000_0000;
        let fb = 1920 * 1080 * 4;
        assert_eq!(
            screen_vram_range(bar1, fb, bar1, 256 << 20),
            Some((0, 0x7f_0000))
        );
        // Not at the start of BAR1: rounded out to whole 64 KiB steps.
        assert_eq!(
            screen_vram_range(bar1 + 0x1_2340, 0x100, bar1, 256 << 20),
            Some((0x1_0000, 0x2_0000))
        );
        // Outside BAR1 (QEMU's 0x80000000, or another board): not ours.
        assert_eq!(screen_vram_range(0x8000_0000, fb, bar1, 256 << 20), None);
        assert_eq!(
            screen_vram_range(bar1 + (256 << 20) - 4, 8, bar1, 256 << 20),
            None
        );
        assert_eq!(screen_vram_range(bar1, 0, bar1, 256 << 20), None);
    }

    /// A real BAR (not a model) without the permit: refused, nothing touched.
    #[test]
    fn real_bar_needs_the_permit() {
        struct Real(u32);
        impl Mmio for Real {
            fn len(&self) -> u64 {
                16 << 20
            }
            fn read32_raw(&mut self, _: u64) -> u32 {
                self.0 += 1;
                0
            }
            fn write32_raw(&mut self, _: u64, _: u32) {
                self.0 += 1000;
            }
        }
        let mut real = Real(0);
        let mut j = Journal::new(4);
        assert!(matches!(
            VramWindow::new(&mut real, &mut j, None, VRAM, &[]),
            Err(GpuError::NotPermitted(_))
        ));
        assert_eq!(real.0, 0);
        assert!(HardwareWritePermit::new("test").is_err() || cfg!(feature = "hw-write"));
    }
}
