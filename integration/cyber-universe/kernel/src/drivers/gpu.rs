//! Compute GPU, level 0 (C-11 / K4): identification only. The driver logic lives in the `cyber-gpu` crates
//! (tested on the host against SIMULATED devices); this file gives them the hardware: the ECAM configuration
//! space of the card and an uncached mapping of the first 4 KiB of BAR0.
//!
//! What it does on a real card: reads the PCI header, BARs and capabilities, then reads exactly two BAR0
//! registers (NV_PMC_BOOT_0 @0x0, NV_PMC_BOOT_42 @0xA00). No write to the card, no Bus Master, no DMA.
//! `gpu dimensiuni da` is the only path that writes (PCI BAR sizing, decode off, restored and verified); it is
//! an explicit shell option (D-013) and stays off by default.

use crate::mem::{map::phys_to_virt, paging};
use crate::pci::Device;
use alloc::{format, string::String};
use core::arch::asm;
use gpu_core::diag::Journal;
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{ConfigSpace, Mmio};
use gpu_core::ids::PciAddress;
use gpu_pci::bar::{SizingPermit, size_bars};
use nvidia_ga104::ids;
use nvidia_ga104::init::{GA104_PLAN, StepStatus};
use nvidia_ga104::probe;
use nvidia_ga104::regs::PROBE_WINDOW;

/// Configuration space of one PCI function through ECAM.
struct KernelCfg(Device);

impl ConfigSpace for KernelCfg {
    fn read32(&mut self, offset: u16) -> u32 {
        self.0.read32(offset)
    }
    fn write32(&mut self, offset: u16, value: u32) {
        self.0.write32(offset, value)
    }
}

/// The first page of BAR0, mapped uncached (PCD|PWT) by `paging::map_mmio`.
struct Bar0Window {
    virt: u64,
}

// SAFETY (both accesses): `virt` is the uncached mapping of the first PROBE_WINDOW bytes of BAR0, created by
// `map_page` below only after checking the page was not mapped before (so its caching is ours). `gpu_core::hal`
// checks offset < PROBE_WINDOW and 4-byte alignment before calling these.
impl Mmio for Bar0Window {
    fn len(&self) -> u64 {
        PROBE_WINDOW
    }
    fn read32_raw(&mut self, offset: u64) -> u32 {
        unsafe { ((self.virt + offset) as *const u32).read_volatile() }
    }
    fn write32_raw(&mut self, offset: u64, value: u32) {
        // Unreachable from nvidia-ga104 (its BAR0 accessor is read-only); kept for the trait.
        unsafe { ((self.virt + offset) as *mut u32).write_volatile(value) }
    }
}

fn find_card() -> Option<Device> {
    crate::pci::devices().into_iter().find(|d| ids::lookup(d.vendor, d.device).is_some() && d.class == 0x03)
}

fn address(d: &Device) -> PciAddress {
    PciAddress { segment: 0, bus: d.bus, dev: d.dev, func: d.func }
}

/// Map the first BAR0 page uncached. If the page is already mapped (e.g. inside an HHDM huge page) its caching
/// is unknown, so the probe refuses instead of reading device registers through a possibly cached mapping.
fn map_page(bar0: u64) -> GpuResult<Bar0Window> {
    let virt = phys_to_virt(bar0) as u64;
    if paging::translate(virt).is_some() {
        return Err(GpuError::Blocked("BAR0 deja mapat de altcineva: atributele de cache nu pot fi verificate"));
    }
    paging::map_mmio(bar0, PROBE_WINDOW).map_err(|_| GpuError::Blocked("map_mmio a esuat pentru BAR0"))?;
    Ok(Bar0Window { virt })
}

/// `gpu` / `gpu identifica`: the level-0 report.
pub fn identify() -> String {
    let Some(dev) = find_card() else {
        return String::from("gpu: nicio placa GA104 (nivel 0 cauta doar 10de:24xx, clasa 03)\n");
    };
    let mut cfg = KernelCfg(dev);
    let mut journal = Journal::new(64);
    let candidate = match probe::probe_config(&mut cfg, Some(address(&dev))) {
        Ok(c) => c,
        Err(e) => return format!("gpu: {}: {}\n", dev.address(), e),
    };
    crate::info!("gpu: {} {:04x}:{:04x} candidat {}, BAR0 {:#x}", dev.address(), dev.vendor, dev.device, candidate.model, candidate.bar0);
    let mut win = match map_page(candidate.bar0) {
        Ok(w) => w,
        Err(e) => return format!("{}gpu: BAR0 necitit: {}\n", candidate.report, e),
    };
    let out = match probe::probe_mmio(&mut win, &mut journal) {
        Ok(chip) => format!("{}", probe::Level0 { candidate, chip }),
        Err(e) => format!("{}gpu: BOOT_0: {}\n", candidate.report, e),
    };
    for e in journal.events() {
        crate::info!("gpu: {}", e);
    }
    crate::info!("gpu: nivel 0 gata, {} scrieri in placa", journal.writes());
    out
}

/// Runs `f` with maskable interrupts off on this (only) CPU, restoring the previous state.
fn without_interrupts<T>(f: impl FnOnce() -> T) -> T {
    let flags: u64;
    // SAFETY: pushfq/pop reads RFLAGS through the stack (so no `nostack`); cli only masks maskable interrupts on
    // this CPU and changes IF (so no `preserves_flags`). No other memory is touched.
    unsafe { asm!("pushfq", "pop {}", "cli", out(reg) flags, options(nomem)) };
    let r = f();
    if flags & (1 << 9) != 0 {
        // SAFETY: re-enables interrupts only if they were enabled on entry.
        unsafe { asm!("sti", options(nomem, nostack)) };
    }
    r
}

/// `gpu dimensiuni da`: PCI BAR sizing (writes to configuration space). Interrupts are off for the whole
/// sequence so nothing (console, framebuffer) touches the card while its decode is disabled.
pub fn size() -> String {
    let Some(dev) = find_card() else {
        return String::from("gpu: nicio placa GA104\n");
    };
    let mut cfg = KernelCfg(dev);
    let mut journal = Journal::new(128);
    let permit = SizingPermit::new("comanda shell: gpu dimensiuni da");
    let r = without_interrupts(|| size_bars(&mut cfg, &permit, &mut journal));
    for e in journal.events() {
        crate::info!("gpu: {}", e);
    }
    match r {
        Ok(bars) => {
            let mut s = String::new();
            for b in bars.iter().flatten() {
                s += &format!("  BAR{} @ {:#x} marime {:#x}\n", b.index, b.base, b.size.unwrap_or(0));
            }
            s
        }
        Err(e) => format!("gpu: dimensionare esuata: {} (placa trebuie considerata in eroare)\n", e),
    }
}

/// `gpu plan`: the GA104 bring-up steps and where they stop.
pub fn plan() -> String {
    let mut s = String::new();
    for st in GA104_PLAN {
        let status = match st.status {
            StepStatus::Implemented => String::from("implementat"),
            StepStatus::NotImplemented => String::from("neimplementat"),
            StepStatus::Blocked(why) => format!("blocat: {}", why),
        };
        s += &format!("  {:<4} {} — {}\n", st.id, st.what, status);
    }
    s
}
