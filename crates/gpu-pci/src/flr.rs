//! C-151 point 3 — Function Level Reset (FLR) of the GPU function, behind an
//! explicit permit. **Never called by this crate on its own**; the OS calls it
//! only when WPR2 is up at the start of the chain (variant 1 of
//! `docs/C-142_K513_VITEZA_FLR.md` §4), see `docs/C-151_RESET_FLR.md`.
//!
//! The order is the one of NVIDIA's RM 570.144 (MIT, read as a specification:
//! `kbifDoFunctionLevelReset_TU102`, `kernel_bif_tu102.c`), checked against the
//! PCIe Base Spec (Device Capabilities bit 28, Device Control bit 15, Device
//! Status bit 5 "Transactions Pending", "wait 100 ms after FLR before a
//! configuration request") and tinygrad (`nvdev.py`, MIT: Bus Master off,
//! reset, 100 ms). Our own code; nothing translated.
//!
//! 1. the device must announce FLR (Device Capabilities bit 28), else refused;
//! 2. the configuration is saved ([`save_config`]: header dwords 0..15, the
//!    PCIe control registers, the MSI capability) — reads only;
//! 3. Bus Master off (one Command write, read back): no more DMA to host RAM;
//! 4. wait (bounded) for Transactions Pending = 0; like RM and Linux, a
//!    timeout here is journaled and the reset goes on;
//! 5. Device Control bit 15 written (the status half written 0: it is RW1C);
//! 6. 100 ms, then poll Vendor ID until the function answers (not all ones,
//!    not the CRS value `0x0001`), bounded by [`FLR_READY_TIMEOUT_US`];
//! 7. the saved configuration written back (only dwords that differ; BARs
//!    while Command is still 0 after the reset, Command last) and re-read;
//!    any difference is an error, named.
//!
//! What this does **not** do (on purpose): it writes no BAR0 register (RM also
//! clears an XVE debug counter after FLR — an NVIDIA-internal register we do
//! not touch), and it does not wait for the VBIOS (GFW boot) — that needs BAR0
//! and is the next step of the caller (`nvidia_ga104::flr_reset`).

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, ConfigSpace, poll_until};

use crate::caps::{self, CAP_MSI, CAP_PCIE};
use crate::header::{CMD_BUS_MASTER, REG_COMMAND};

/// PCIe spec: no configuration request earlier than 100 ms after FLR.
pub const FLR_WAIT_US: u32 = 100_000;
/// How long the function may answer "not ready" after those 100 ms (SIM
/// choice; Linux waits much longer, RM reads once). 1 s.
pub const FLR_READY_TIMEOUT_US: u64 = 1_000_000;
/// Bound for Transactions Pending to clear before the reset (100 ms).
pub const PENDING_TIMEOUT_US: u64 = 100_000;
/// Poll step for both waits.
pub const POLL_US: u32 = 1_000;

/// Explicit acknowledgement that the caller resets the function (C-151:
/// the OS, in the automatic cycle approved by Adrian on 19 Nov 2026).
pub struct FlrPermit {
    pub reason: &'static str,
}

impl FlrPermit {
    pub fn new(reason: &'static str) -> Self {
        FlrPermit { reason }
    }
}

/// The configuration saved before the reset: `(offset, value)` dwords.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SavedConfig {
    /// Header dwords 0x00..0x3c (16 of them).
    pub header: [u32; 16],
    /// PCIe control dwords (DevCtl, LnkCtl, DevCtl2, LnkCtl2), as read.
    pub pcie: Vec<(u16, u32)>,
    /// MSI capability dwords (control dword last when restored).
    pub msi: Vec<(u16, u32)>,
    /// Offset of the PCIe capability.
    pub pcie_cap: u16,
}

/// Header dwords that are read-only in a type-0 header (ID, class/revision,
/// CardBus CIS, subsystem, capability pointer): never written back.
const HEADER_RO: [usize; 5] = [0, 2, 10, 11, 13];

/// Reads only. `Err(NotResponding)` when the device answers all ones;
/// `Err(Blocked)` without a PCIe capability.
pub fn save_config<C: ConfigSpace + ?Sized>(cfg: &mut C) -> GpuResult<SavedConfig> {
    let mut header = [0u32; 16];
    for (i, h) in header.iter_mut().enumerate() {
        *h = cfg.read32((i * 4) as u16);
    }
    if header[0] == 0xffff_ffff {
        return Err(GpuError::NotResponding);
    }
    let p = caps::find(cfg, CAP_PCIE)
        .ok_or(GpuError::Blocked("FLR: placa nu are capabilitate PCIe"))?;
    let pcie = [0x08u16, 0x10, 0x28, 0x30]
        .iter()
        .map(|&o| (p + o, cfg.read32(p + o)))
        .collect();
    let msi = match caps::find(cfg, CAP_MSI) {
        Some(m) => [4u16, 8, 0xc, 0]
            .iter()
            .map(|&o| (m + o, cfg.read32(m + o)))
            .collect(),
        None => Vec::new(),
    };
    Ok(SavedConfig {
        header,
        pcie,
        msi,
        pcie_cap: p,
    })
}

/// What the reset saw, for the journal and the OS report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlrDone {
    /// Transactions Pending was clear before the trigger (else: timed out,
    /// journaled, reset done anyway).
    pub pending_cleared: bool,
    /// Microseconds after the trigger until Vendor ID answered (≥ 100 000).
    pub ready_after_us: u64,
    /// Configuration dwords written back.
    pub restored: u32,
}

fn write_logged<C: ConfigSpace + ?Sized>(cfg: &mut C, j: &mut Journal, off: u16, v: u32) -> u32 {
    let before = cfg.read32(off);
    cfg.write32(off, v);
    let after = cfg.read32(off);
    j.push(Event::ConfigWrite {
        offset: off,
        before,
        written: v,
        after,
    });
    after
}

/// The reset itself (steps 1, 3–7 of the module comment; `saved` from
/// [`save_config`], read just before). On `Ok` the function answers again
/// with the saved configuration; the caller then waits for GFW boot.
pub fn function_level_reset<C: ConfigSpace + ?Sized, K: Clock + ?Sized>(
    cfg: &mut C,
    clock: &mut K,
    _permit: &FlrPermit,
    j: &mut Journal,
    saved: &SavedConfig,
) -> GpuResult<FlrDone> {
    let rc = caps::reset_capabilities(cfg)
        .ok_or(GpuError::Blocked("FLR: placa nu are capabilitate PCIe"))?;
    if !rc.flr_capable {
        j.push(Event::Step {
            name: "C-151 FLR: placa nu anunta FLR",
            ok: false,
        });
        return Err(GpuError::Blocked(
            "FLR: placa nu anunta FLR (Device Capabilities bit 28)",
        ));
    }
    let p = saved.pcie_cap;
    // 3. Bus Master off (status half 0: RW1C).
    let cmd = cfg.read32(REG_COMMAND) as u16;
    if cmd & CMD_BUS_MASTER != 0 {
        let after = write_logged(cfg, j, REG_COMMAND, u32::from(cmd & !CMD_BUS_MASTER));
        if after as u16 & CMD_BUS_MASTER != 0 {
            return Err(GpuError::Blocked("FLR: Bus Master nu s-a oprit"));
        }
    }
    j.push(Event::Step {
        name: "C-151 FLR: Bus Master oprit",
        ok: true,
    });
    // 4. Transactions Pending (Device Status bit 5 = dword bit 21).
    let pending = poll_until(clock, PENDING_TIMEOUT_US, POLL_US, || {
        (cfg.read32(p + 0x08) & (1 << 21) == 0).then_some(())
    });
    let pending_cleared = pending.is_ok();
    if !pending_cleared {
        j.push(Event::Note(
            "C-151 FLR: tranzactii in asteptare dupa 100 ms; resetul merge mai departe (ca RM si Linux)",
        ));
    }
    // 5. Trigger: Device Control bit 15, Device Status half written 0.
    let devctl = cfg.read32(p + 0x08) & 0xffff;
    let t0 = clock.now_us();
    cfg.write32(p + 0x08, devctl | 0x8000);
    j.push(Event::ConfigWrite {
        offset: p + 0x08,
        before: devctl,
        written: devctl | 0x8000,
        // No read-back: the function is in reset for 100 ms.
        after: 0xffff_ffff,
    });
    j.push(Event::Step {
        name: "C-151 FLR declansat",
        ok: true,
    });
    // 6. 100 ms, then until Vendor ID answers.
    clock.delay_us(FLR_WAIT_US);
    let id = poll_until(clock, FLR_READY_TIMEOUT_US, POLL_US, || {
        let v = cfg.read32(0);
        (v != 0xffff_ffff && v as u16 != 0x0001).then_some(v)
    });
    let ready_after_us = clock.now_us().saturating_sub(t0);
    let id = match id {
        Ok(v) => v,
        Err(_) => {
            j.push(Event::Step {
                name: "C-151 FLR: placa nu mai raspunde",
                ok: false,
            });
            return Err(GpuError::NotResponding);
        }
    };
    j.push(Event::Waited {
        what: "C-151 FLR: placa raspunde din nou",
        waited_us: ready_after_us,
    });
    if id != saved.header[0] {
        j.push(Event::Value {
            what: "C-151 FLR: ID dupa reset",
            value: id,
        });
        return Err(GpuError::NotPresent);
    }
    // 7. Restore: BARs and the rest while Command is 0, Command last.
    let mut restored = 0u32;
    let mut put = |cfg: &mut C, j: &mut Journal, off: u16, want: u32, mask: u32| {
        if cfg.read32(off) & mask != want & mask {
            restored += 1;
            write_logged(cfg, j, off, want);
        }
    };
    for i in (2..16).rev() {
        if !HEADER_RO.contains(&i) {
            // dword 3: only cache line size / latency timer (header type RO, BIST)
            let mask = if i == 3 { 0xffff } else { 0xffff_ffff };
            put(cfg, j, (i * 4) as u16, saved.header[i], mask);
        }
    }
    for &(off, v) in &saved.pcie {
        // control halves only; status halves (RW1C) written 0
        put(cfg, j, off, v & 0xffff & !0x8000, 0xffff);
    }
    for &(off, v) in &saved.msi {
        put(cfg, j, off, v, 0xffff_ffff);
    }
    put(cfg, j, REG_COMMAND, saved.header[1] & 0xffff, 0xffff);
    // Re-read and compare everything restored.
    let mut ok = true;
    for (i, &want) in saved.header.iter().enumerate().skip(1) {
        if HEADER_RO.contains(&i) {
            continue;
        }
        let mask = if i == 1 || i == 3 {
            0xffff
        } else {
            0xffff_ffff
        };
        ok &= cfg.read32((i * 4) as u16) & mask == want & mask;
    }
    for &(off, v) in &saved.pcie {
        ok &= cfg.read32(off) & 0x7fff == v & 0x7fff;
    }
    for &(off, v) in &saved.msi {
        ok &= cfg.read32(off) == v;
    }
    if !ok {
        j.push(Event::Step {
            name: "C-151 FLR: configuratia nu s-a rescris",
            ok: false,
        });
        return Err(GpuError::Blocked(
            "FLR: configuratia PCI rescrisa difera de cea salvata",
        ));
    }
    j.push(Event::Step {
        name: "C-151 FLR: configuratia rescrisa",
        ok: true,
    });
    Ok(FlrDone {
        pending_cleared,
        ready_after_us,
        restored,
    })
}
