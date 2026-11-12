//! I6–I7: the register side of booting GSP-RM on GA102 — falcon reset,
//! core selection, start/halt, the FWSEC-FRTS result (WPR2) and Booter on
//! SEC2. **Tested only in SIMULARE** against `gpu_sim::fake_gsp`; never run on
//! the RTX 3070 Ti.
//!
//! Sources (read as specification, code written by us, docs/PROVENANCE.md):
//! register offsets and fields generated from NVIDIA's MIT headers
//! (`nvidia-rm570`: `dev_falcon_v4.h`, `dev_gsp.h`, `dev_sec_pri.h`,
//! `dev_riscv_pri.h`, `dev_fb.h`, `dev_bus.h`, `dev_gc6_island*.h`, 570.144);
//! the *order* of the steps from nouveau `nvkm/falcon/ga102.c`,
//! `nvkm/subdev/gsp/r535.c` and tinygrad `support/nv/ip.py` (both MIT).
//! DEDUS (from those sources, not from NVIDIA documentation): the FRTS error
//! code in `NV_PBUS_SW_SCRATCH(0xe)[31:16]`; the timeouts; using
//! `CPUCTL_ALIAS` when `CPUCTL.ALIAS_EN` is set.
//!
//! Every access goes through [`GuardedBar0`]: writes only inside the GSP and
//! SEC2 windows, reads only there plus a few named status registers, every
//! write journaled with the value before and after. On a real BAR it needs a
//! [`HardwareWritePermit`] (build option `hw-write`, off by default).
//!
//! Loading the signed ucode (IMEM/DMEM over DMA) is in [`super::hs_load`]
//! (session 12); [`fwsec_frts_dma`] / [`booter_load_dma`] use it. The older
//! entry points with a `load` step stay for compatibility and for tests.

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, Mmio, checked_read32, checked_write32, poll_until};
use nvidia_rm570::regs::ga102_dev_falcon_second_pri as f2;
use nvidia_rm570::regs::ga102_dev_falcon_v4 as fv4;
use nvidia_rm570::regs::ga102_dev_falcon_v4_addendum as fv4a;
use nvidia_rm570::regs::ga102_dev_gc6_island as gc6;
use nvidia_rm570::regs::ga102_dev_gc6_island_addendum as gc6a;
use nvidia_rm570::regs::ga102_dev_gsp as gsp;
use nvidia_rm570::regs::ga102_dev_riscv_pri as riscv;
use nvidia_rm570::regs::ga102_dev_sec_pri as sec;
use nvidia_rm570::regs::tu102_dev_bus as bus;
use nvidia_rm570::regs::tu102_dev_fb as fb;

use super::HardwareWritePermit;
use crate::regs::is_priv_error;

/// The two windows this module may write (`NV_PGSP`, `NV_PSEC`).
pub const WRITE_RANGES: [(u32, u32); 2] = [
    (gsp::NV_PGSP.lo, gsp::NV_PGSP.hi),
    (sec::NV_PSEC.lo, sec::NV_PSEC.hi),
];

/// Status registers outside those windows that may be *read*.
pub const EXTRA_READS: [u32; 5] = [
    0, // NV_PMC_BOOT_0
    fb::NV_PFB_PRI_MMU_WPR2_ADDR_LO,
    fb::NV_PFB_PRI_MMU_WPR2_ADDR_HI,
    bus::NV_PBUS_SW_SCRATCH(0xe),
    gc6::NV_PGC6_BSI_SECURE_SCRATCH_14,
];

fn in_write_ranges(off: u32) -> bool {
    WRITE_RANGES.iter().any(|&(lo, hi)| off >= lo && off <= hi)
}

/// BAR0 with the write policy of I6–I8 (see the module comment).
pub struct GuardedBar0<'a, M: Mmio + ?Sized> {
    mmio: &'a mut M,
    journal: &'a mut Journal,
}

impl<'a, M: Mmio + ?Sized> GuardedBar0<'a, M> {
    /// A model from `gpu-sim` needs no permit; a real BAR always does.
    pub fn new(
        mmio: &'a mut M,
        journal: &'a mut Journal,
        permit: Option<&HardwareWritePermit>,
    ) -> GpuResult<Self> {
        if !mmio.is_simulation() && permit.is_none() {
            return Err(GpuError::NotPermitted(
                "scriere in placa reala fara permis (optiunea hw-write, doar cu Adrian de fata)",
            ));
        }
        if let Some(p) = permit {
            journal.push(Event::Note(p.reason));
        }
        Ok(GuardedBar0 { mmio, journal })
    }

    pub fn read(&mut self, off: u32) -> GpuResult<u32> {
        if !in_write_ranges(off) && !EXTRA_READS.contains(&off) {
            return Err(GpuError::NotPermitted(
                "citire in afara ferestrelor GSP/SEC2 si a registrelor de stare permise",
            ));
        }
        let v = checked_read32(self.mmio, u64::from(off))?;
        if is_priv_error(v) {
            return Err(GpuError::HardwareFault { code: v });
        }
        Ok(v)
    }

    pub fn write(&mut self, off: u32, v: u32) -> GpuResult<()> {
        if !in_write_ranges(off) {
            return Err(GpuError::NotPermitted(
                "scriere in afara ferestrelor GSP/SEC2",
            ));
        }
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
        Ok(())
    }

    pub fn journal(&mut self) -> &mut Journal {
        self.journal
    }

    /// Tests only: reach the model to "load" a program (no real loader yet).
    #[cfg(test)]
    pub(crate) fn model(&mut self) -> &mut M {
        self.mmio
    }
}

/// The two falcons used to boot GSP-RM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Engine {
    Gsp,
    Sec2,
}

impl Engine {
    pub const fn base(self) -> u32 {
        match self {
            Engine::Gsp => gsp::NV_PGSP.lo,
            Engine::Sec2 => sec::NV_PSEC.lo,
        }
    }
    /// Base of the RISC-V/"falcon2" registers of the same engine.
    pub const fn riscv_base(self) -> u32 {
        match self {
            Engine::Gsp => f2::NV_FALCON2_GSP_BASE,
            Engine::Sec2 => f2::NV_FALCON2_SEC_BASE,
        }
    }
    const fn engine_reg(self) -> u32 {
        match self {
            Engine::Gsp => gsp::NV_PGSP_FALCON_ENGINE,
            Engine::Sec2 => sec::NV_PSEC_FALCON_ENGINE,
        }
    }
}

/// Timeouts (DEDUS: nouveau/nova use the same orders of magnitude).
pub const RESET_READY_US: u64 = 150;
/// C-96: RM 570.144 waits for scrubbing with its default GPU timeout
/// (`GPU_TIMEOUT_DEFAULT`, 4 s on silicon; nouveau: 20 ms). The board
/// (10 Nov 2026) answered priv errors right after a reset of a running
/// GSP-RM, so the wait is RM's; it ends as soon as scrubbing reads DONE.
pub const SCRUB_US: u64 = 4_000_000;
pub const CORE_SELECT_US: u64 = 10_000;
pub const HALT_US: u64 = 2_000_000;
/// os-295, journal 011 (12 Nov 2026): was 2 s; the board once answered
/// HALTED after it. nova-core waits up to 5 s for the core
/// (`docs/GA104_INITIALIZATION.md`, the longest wait in the public sources),
/// so the bound is nova's now. A longer wait writes nothing; if the core is
/// not active after 5 s, the bound is not the cause.
pub const RISCV_ACTIVE_US: u64 = 5_000_000;
const STEP_US: u32 = 10;

/// C-96: a poll loop in the RM 570.144 / nouveau sense. A read that answers
/// a priv error (`0xbadf....`) is "not there yet", not a fault: both drivers
/// keep polling through it (RM `kflcnWaitForResetToFinish_GA102`,
/// `kflcnSwitchToFalcon_GA102`; nouveau `nvkm_msec` loops read the raw
/// value). If the register still answers a priv error when the time is up,
/// the error is reported with that value (not as a plain timeout), so the
/// journal shows what the board said. Any other error stops at once.
pub(crate) fn poll<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    timeout_us: u64,
    off: u32,
    mut ok: impl FnMut(u32) -> bool,
) -> GpuResult<u32> {
    let mut err = None;
    let mut last_priv = None;
    let r = poll_until(clock, timeout_us, STEP_US, || match bar.read(off) {
        Ok(v) if ok(v) => Some(v),
        Ok(_) => {
            last_priv = None;
            None
        }
        Err(GpuError::HardwareFault { code }) if is_priv_error(code) => {
            last_priv = Some(code);
            None
        }
        Err(e) => {
            err = Some(e);
            Some(0)
        }
    });
    match (err, r, last_priv) {
        (Some(e), _, _) => Err(e),
        (None, Err(GpuError::Timeout { .. }), Some(code)) => Err(GpuError::HardwareFault { code }),
        (None, r, _) => r,
    }
}

/// RM 570.144 `FLCN_RESET_PROPAGATION_DELAY_COUNT` (`falcon_common.h`, MIT):
/// register reads of `NV_P*_FALCON_ENGINE` after each write of it, so the
/// reset signal propagates before the next access.
pub const RESET_PROPAGATION_READS: u32 = 10;

/// C-96: RM's `kgspResetHw_TU102` order for the engine reset register:
/// RESET=1, propagation reads, RESET=0, propagation reads. The reads' values
/// are not used (a priv error while in reset is expected).
fn engine_reset_pulse<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    e: Engine,
) -> GpuResult<()> {
    bar.write(e.engine_reg(), 1)?;
    for _ in 0..RESET_PROPAGATION_READS {
        let _ = bar.read(e.engine_reg());
    }
    clock.delay_us(10);
    bar.write(e.engine_reg(), 0)?;
    for _ in 0..RESET_PROPAGATION_READS {
        let _ = bar.read(e.engine_reg());
    }
    Ok(())
}

/// Engine reset + wait for memory scrubbing (nouveau `ga102_flcn_reset_eng`
/// order: wait RESET_READY (not fatal), RESET=1, RESET=0, wait scrubbing).
///
/// C-96 (board, 10 Nov 2026): on a GSP that was running GSP-RM the first
/// HWCFG2 read after the release answered `0xbadf5620`; the scrub wait now
/// polls through priv errors (as RM and nouveau do) and the wait is RM's
/// default bound ([`SCRUB_US`]) instead of nouveau's 20 ms. The two notes
/// tell apart "RESET_READY never set" (RM: expected sometimes) from
/// "HWCFG2 unreadable before the reset".
pub fn reset<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    e: Engine,
) -> GpuResult<()> {
    let hwcfg2 = e.base() + fv4::NV_PFALCON_FALCON_HWCFG2;
    match poll(bar, clock, RESET_READY_US, hwcfg2, |v| {
        fv4a::NV_PFALCON_FALCON_HWCFG2_RESET_READY.get(v) == 1
    }) {
        Ok(_) => {}
        Err(GpuError::HardwareFault { .. }) => bar.journal().push(Event::Note(
            "falcon: HWCFG2 inaccesibil (0xbadf....) inainte de reset; continui",
        )),
        Err(_) => bar
            .journal()
            .push(Event::Note("falcon: RESET_READY nu a venit; continui")),
    }
    engine_reset_pulse(bar, clock, e)?;
    poll(bar, clock, SCRUB_US, hwcfg2, |v| {
        fv4::NV_PFALCON_FALCON_HWCFG2_MEM_SCRUBBING.get(v)
            == fv4::NV_PFALCON_FALCON_HWCFG2_MEM_SCRUBBING_DONE
    })?;
    Ok(())
}

/// C-96: RM 570.144 `kflcnSwitchToFalcon_GA102` — after a reset, if the
/// engine's `BCR_CTRL` does not already say CORE_SELECT = FALCON (a priv
/// error counts as "not falcon": RM reads it the same way), select the
/// falcon core and wait for VALID. RM's comment: this also releases the
/// RISC-V priv lockdown. Part of RM's `kflcnReset` (sequencer CORE_RESET).
pub fn switch_to_falcon<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    e: Engine,
) -> GpuResult<()> {
    let bcr = e.riscv_base() + riscv::NV_PRISCV_RISCV_BCR_CTRL;
    let already_falcon = match bar.read(bcr) {
        Ok(v) => {
            riscv::NV_PRISCV_RISCV_BCR_CTRL_CORE_SELECT.get(v)
                == riscv::NV_PRISCV_RISCV_BCR_CTRL_CORE_SELECT_FALCON
        }
        Err(GpuError::HardwareFault { code }) if is_priv_error(code) => false,
        Err(e) => return Err(e),
    };
    if already_falcon {
        return Ok(());
    }
    select_falcon_core(bar, clock, e)
}

/// Selects the falcon core (not RISC-V) for a signed HS program.
pub fn select_falcon_core<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    e: Engine,
) -> GpuResult<()> {
    let bcr = e.riscv_base() + riscv::NV_PRISCV_RISCV_BCR_CTRL;
    bar.write(
        bcr,
        riscv::NV_PRISCV_RISCV_BCR_CTRL_CORE_SELECT
            .set(0, riscv::NV_PRISCV_RISCV_BCR_CTRL_CORE_SELECT_FALCON),
    )?;
    poll(bar, clock, CORE_SELECT_US, bcr, |v| {
        riscv::NV_PRISCV_RISCV_BCR_CTRL_VALID.get(v) == riscv::NV_PRISCV_RISCV_BCR_CTRL_VALID_TRUE
    })?;
    Ok(())
}

/// `STARTCPU` the way RM (`kflcnStartCpu_TU102`) and nouveau
/// (`nvkm_falcon_v1_start`) do it: through `CPUCTL_ALIAS` when
/// `CPUCTL.ALIAS_EN` is set, else through `CPUCTL`. C-99 (board, 10 Nov
/// 2026): after Booter halted, SEC2's CPUCTL read `0x50` (HALTED |
/// ALIAS_EN) and a start written to CPUCTL was dropped — the sequencer's
/// CORE_RESUME waited 2 s for a SEC2 that never ran.
pub fn start_cpu<M: Mmio + ?Sized>(bar: &mut GuardedBar0<'_, M>, e: Engine) -> GpuResult<()> {
    let b = e.base();
    let cpuctl = bar.read(b + fv4::NV_PFALCON_FALCON_CPUCTL)?;
    let start = fv4::NV_PFALCON_FALCON_CPUCTL_STARTCPU.set(0, 1);
    if fv4::NV_PFALCON_FALCON_CPUCTL_ALIAS_EN.get(cpuctl) == 1 {
        bar.write(b + fv4::NV_PFALCON_FALCON_CPUCTL_ALIAS, start)
    } else {
        bar.write(b + fv4::NV_PFALCON_FALCON_CPUCTL, start)
    }
}

/// Writes the two mailboxes, starts the CPU, waits for HALTED and returns
/// the mailboxes as the program left them.
pub fn run_to_halt<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    e: Engine,
    mailbox: (u32, u32),
) -> GpuResult<(u32, u32)> {
    let b = e.base();
    bar.write(b + fv4::NV_PFALCON_FALCON_MAILBOX0, mailbox.0)?;
    bar.write(b + fv4::NV_PFALCON_FALCON_MAILBOX1, mailbox.1)?;
    start_cpu(bar, e)?;
    poll(
        bar,
        clock,
        HALT_US,
        b + fv4::NV_PFALCON_FALCON_CPUCTL,
        |v| {
            fv4::NV_PFALCON_FALCON_CPUCTL_HALTED.get(v) == fv4::NV_PFALCON_FALCON_CPUCTL_HALTED_TRUE
        },
    )?;
    Ok((
        bar.read(b + fv4::NV_PFALCON_FALCON_MAILBOX0)?,
        bar.read(b + fv4::NV_PFALCON_FALCON_MAILBOX1)?,
    ))
}

/// The protected region FWSEC-FRTS created, as the MMU reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Wpr2 {
    pub start: u64,
    pub end: u64,
}

/// `NV_PFB_PRI_MMU_WPR2_ADDR_{LO,HI}`: VAL is bits 31:4, address = VAL << 12.
pub fn decode_wpr2(lo: u32, hi: u32) -> Wpr2 {
    let a = |r: u32| {
        u64::from(fb::NV_PFB_PRI_MMU_WPR2_ADDR_LO_VAL.get(r))
            << fb::NV_PFB_PRI_MMU_WPR2_ADDR_LO_ALIGNMENT
    };
    Wpr2 {
        start: a(lo),
        end: a(hi),
    }
}

/// Reads WPR2 without writing anything: a non-zero HI means some driver
/// already ran FWSEC on this board since power-on (tinygrad checks the same).
pub fn read_wpr2<M: Mmio + ?Sized>(bar: &mut GuardedBar0<'_, M>) -> GpuResult<Option<Wpr2>> {
    let hi = bar.read(fb::NV_PFB_PRI_MMU_WPR2_ADDR_HI)?;
    if hi == 0 {
        return Ok(None);
    }
    let lo = bar.read(fb::NV_PFB_PRI_MMU_WPR2_ADDR_LO)?;
    Ok(Some(decode_wpr2(lo, hi)))
}

/// How the WPR2 HI register encoded the end of the region FWSEC created.
///
/// Board, 3 Nov 2026 (`rezultate/2026-11-03-workstation-os214-i6`): LO was
/// exactly the FRTS start we asked for, HI was *not* the exclusive end. The
/// public references only check LO (RM 570.144 `kernel_gsp_frts_tu102.c`:
/// `HI != 0` and `LO == frtsOffset`; nova-core: the same, its comment calls HI
/// "exclusive" but never checks it). The most likely encoding is the last
/// 4 KiB page (inclusive); the raw HI from `gpu-jurnal-002.txt` will confirm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Wpr2End {
    /// HI = end of the region (first byte after it).
    Exclusive,
    /// HI = start of the last 4 KiB page of the region.
    InclusiveLastPage,
}

/// Everything `check_frts` saw, for the log line and for the window report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrtsCheck {
    /// The region, end normalised to exclusive (= `frts` when it passed).
    pub wpr2: Wpr2,
    /// Raw `NV_PFB_PRI_MMU_WPR2_ADDR_LO` / `_HI`.
    pub lo_raw: u32,
    pub hi_raw: u32,
    pub end_form: Wpr2End,
}

/// WPR2 granularity: VAL counts 4 KiB pages (`_ALIGNMENT` 0xc).
const WPR2_PAGE: u64 = 1 << fb::NV_PFB_PRI_MMU_WPR2_ADDR_HI_ALIGNMENT;

/// Pure part of the check: compares LO/HI with the region we asked for.
/// LO must be exactly `frts.0` (what RM checks); HI must be either the
/// exclusive end or the inclusive last page of exactly `frts` - anything else
/// is reported against the HI register, with expected = exclusive end.
pub fn compare_frts(lo: u32, hi: u32, frts: (u64, u64)) -> GpuResult<FrtsCheck> {
    let raw = decode_wpr2(lo, hi);
    if raw.start != frts.0 {
        return Err(GpuError::VerifyFailed {
            offset: fb::NV_PFB_PRI_MMU_WPR2_ADDR_LO,
            expected: (frts.0 >> 12) as u32,
            actual: (raw.start >> 12) as u32,
        });
    }
    let end_form = if raw.end == frts.1 {
        Wpr2End::Exclusive
    } else if frts.1 >= frts.0 + WPR2_PAGE && raw.end == frts.1 - WPR2_PAGE {
        Wpr2End::InclusiveLastPage
    } else {
        return Err(GpuError::VerifyFailed {
            offset: fb::NV_PFB_PRI_MMU_WPR2_ADDR_HI,
            expected: (frts.1 >> 12) as u32,
            actual: (raw.end >> 12) as u32,
        });
    };
    Ok(FrtsCheck {
        wpr2: Wpr2 {
            start: frts.0,
            end: frts.1,
        },
        lo_raw: lo,
        hi_raw: hi,
        end_form,
    })
}

/// C-66: every HI encoding the driver accepts. Extends [`Wpr2End`] (kept as
/// it was, for compatibility) with the form the board actually showed on
/// 4 Nov 2026 (`rezultate/2026-11-04-workstation-os222`): LO `0x1ffe00` =
/// the FRTS start, HI `0x1ffee0` = the start of the **last 128 KiB block** of
/// the 1 MiB region. RM 570.144 (`kernel_gsp_frts_tu102.c`) and nova-core
/// only check `HI != 0` and LO, so they accept this too. DEDUS: we read HI as
/// "last 128 KiB block, inclusive"; the other reading (a region 128 KiB
/// shorter, exclusive end) is not excluded by any public source, but no
/// driver we read treats it as a failure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Wpr2EndForm {
    /// HI = end of the region (first byte after it).
    Exclusive,
    /// HI = start of the last 4 KiB page of the region.
    InclusiveLastPage,
    /// HI = start of the last 128 KiB block of the region (board, 4 Nov 2026).
    InclusiveLastBlock128K,
}

impl Wpr2EndForm {
    /// Short Romanian label for log lines and reports.
    pub fn label(self) -> &'static str {
        match self {
            Wpr2EndForm::Exclusive => "exclusiv",
            Wpr2EndForm::InclusiveLastPage => "ultima pagina de 4 KiB",
            Wpr2EndForm::InclusiveLastBlock128K => "ultimul bloc de 128 KiB",
        }
    }
}

/// Like [`FrtsCheck`], with [`Wpr2EndForm`] (C-66).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrtsCheckFull {
    /// The region, end normalised to exclusive (= `frts` when it passed).
    pub wpr2: Wpr2,
    /// Raw `NV_PFB_PRI_MMU_WPR2_ADDR_LO` / `_HI`.
    pub lo_raw: u32,
    pub hi_raw: u32,
    pub form: Wpr2EndForm,
}

/// The 128 KiB block of the board's WPR2 HI (4 Nov 2026).
pub const WPR2_BLOCK_128K: u64 = 128 << 10;

/// C-66: the rule the driver uses. LO must be exactly `frts.0`; HI must be
/// the exclusive end, the last 4 KiB page or the last 128 KiB block of
/// exactly `frts` (the block only when `frts.1` is 128 KiB aligned and the
/// region holds at least one block). Anything else is reported at HI, with
/// expected = exclusive end, as before.
pub fn compare_frts_full(lo: u32, hi: u32, frts: (u64, u64)) -> GpuResult<FrtsCheckFull> {
    let raw = decode_wpr2(lo, hi);
    if raw.start != frts.0 {
        return Err(GpuError::VerifyFailed {
            offset: fb::NV_PFB_PRI_MMU_WPR2_ADDR_LO,
            expected: (frts.0 >> 12) as u32,
            actual: (raw.start >> 12) as u32,
        });
    }
    let back = |n: u64| frts.1 >= frts.0 + n && raw.end == frts.1 - n;
    let form = if raw.end == frts.1 {
        Wpr2EndForm::Exclusive
    } else if back(WPR2_PAGE) {
        Wpr2EndForm::InclusiveLastPage
    } else if frts.1.is_multiple_of(WPR2_BLOCK_128K) && back(WPR2_BLOCK_128K) {
        Wpr2EndForm::InclusiveLastBlock128K
    } else {
        return Err(GpuError::VerifyFailed {
            offset: fb::NV_PFB_PRI_MMU_WPR2_ADDR_HI,
            expected: (frts.1 >> 12) as u32,
            actual: (raw.end >> 12) as u32,
        });
    };
    Ok(FrtsCheckFull {
        wpr2: Wpr2 {
            start: frts.0,
            end: frts.1,
        },
        lo_raw: lo,
        hi_raw: hi,
        form,
    })
}

/// The status reads of [`check_frts_full`], from any BAR0 reader.
fn frts_full_with(
    mut read: impl FnMut(u32) -> GpuResult<u32>,
    frts: (u64, u64),
) -> GpuResult<FrtsCheckFull> {
    let err = read(bus::NV_PBUS_SW_SCRATCH(0xe))? >> 16;
    if err != 0 {
        return Err(GpuError::HardwareFault { code: err });
    }
    let hi = read(fb::NV_PFB_PRI_MMU_WPR2_ADDR_HI)?;
    if hi == 0 {
        return Err(GpuError::HardwareFault { code: 0 });
    }
    let lo = read(fb::NV_PFB_PRI_MMU_WPR2_ADDR_LO)?;
    compare_frts_full(lo, hi, frts)
}

/// C-66: [`check_frts_detail`] with every accepted HI form.
pub fn check_frts_full<M: Mmio + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    frts: (u64, u64),
) -> GpuResult<FrtsCheckFull> {
    frts_full_with(|o| bar.read(o), frts)
}

/// C-66: BAR0 that can only **read** the few status registers of
/// [`EXTRA_READS`] (WPR2 LO/HI, the FRTS error scratch, BOOT_0, BSI scratch).
/// It has no `write` at all, so it needs no [`HardwareWritePermit`]: this is
/// what the OS uses to show WPR2 after I6, once the writes are over (on
/// 4 Nov the permit check refused that read-back: "nepermis de politica de
/// siguranta"). Every read is journaled (value included), so the stick
/// journal keeps them.
pub struct StatusReadBar0<'a, M: Mmio + ?Sized> {
    mmio: &'a mut M,
    journal: &'a mut Journal,
}

impl<'a, M: Mmio + ?Sized> StatusReadBar0<'a, M> {
    pub fn new(mmio: &'a mut M, journal: &'a mut Journal) -> Self {
        StatusReadBar0 { mmio, journal }
    }

    pub fn read(&mut self, off: u32) -> GpuResult<u32> {
        if !EXTRA_READS.contains(&off) {
            return Err(GpuError::NotPermitted(
                "citire doar-de-stare in afara registrelor de stare permise",
            ));
        }
        let v = checked_read32(self.mmio, u64::from(off))?;
        self.journal.push(Event::MmioRead {
            bar: 0,
            offset: u64::from(off),
            value: v,
        });
        if is_priv_error(v) {
            return Err(GpuError::HardwareFault { code: v });
        }
        Ok(v)
    }

    pub fn journal(&mut self) -> &mut Journal {
        self.journal
    }
}

/// C-66: WPR2 read back after I6 with reads only (no permit), checked with
/// [`compare_frts_full`]. Three reads: FRTS error scratch, HI, LO.
pub fn read_frts_after_i6<M: Mmio + ?Sized>(
    mmio: &mut M,
    journal: &mut Journal,
    frts: (u64, u64),
) -> GpuResult<FrtsCheckFull> {
    let mut b = StatusReadBar0::new(mmio, journal);
    frts_full_with(|o| b.read(o), frts)
}

/// C-69: the three registers read after I7 to show its result.
pub const I7_STATUS_READS: [u32; 3] = [
    sec::NV_PSEC.lo + fv4::NV_PFALCON_FALCON_MAILBOX0,
    sec::NV_PSEC.lo + fv4::NV_PFALCON_FALCON_MAILBOX1,
    f2::NV_FALCON2_GSP_BASE + riscv::NV_PRISCV_RISCV_CPUCTL,
];

/// C-69: what the board says after I7, read with reads only.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I7Status {
    /// SEC2 mailbox 0 after Booter halted: 0 = Booter succeeded.
    pub sec2_mailbox0: u32,
    pub sec2_mailbox1: u32,
    /// GSP `NV_PRISCV_RISCV_CPUCTL` (bit 7 = `ACTIVE_STAT`).
    pub riscv_cpuctl: u32,
}

impl I7Status {
    /// Booter returned 0 and the GSP RISC-V core runs (GSP-RM started).
    pub fn passed(&self) -> bool {
        self.sec2_mailbox0 == 0
            && riscv::NV_PRISCV_RISCV_CPUCTL_ACTIVE_STAT.get(self.riscv_cpuctl)
                == riscv::NV_PRISCV_RISCV_CPUCTL_ACTIVE_STAT_ACTIVE
    }
}

/// C-69: the I7 result read back with reads only (no permit, no write
/// function), each read journaled with its value — so the stick journal
/// shows the end of I7 even when `booter_load_dma` returned an error
/// (the same lesson as C-66 for WPR2 after I6). Only the three registers of
/// [`I7_STATUS_READS`].
pub fn read_after_i7<M: Mmio + ?Sized>(mmio: &mut M, journal: &mut Journal) -> GpuResult<I7Status> {
    let mut v = [0u32; 3];
    for (k, &off) in I7_STATUS_READS.iter().enumerate() {
        let x = checked_read32(mmio, u64::from(off))?;
        journal.push(Event::MmioRead {
            bar: 0,
            offset: u64::from(off),
            value: x,
        });
        if is_priv_error(x) {
            return Err(GpuError::HardwareFault { code: x });
        }
        v[k] = x;
    }
    Ok(I7Status {
        sec2_mailbox0: v[0],
        sec2_mailbox1: v[1],
        riscv_cpuctl: v[2],
    })
}

/// After FWSEC-FRTS halted: no error code, WPR2 up, and where we asked
/// (`frts` = the region from `fb_layout`, start..end). A wrong start is
/// reported at LO (0x1fa824), a wrong end at HI (0x1fa828).
/// Knows only the two forms of [`Wpr2End`] (kept for compatibility); the
/// board's 128 KiB form fails here — use [`check_frts_full`] (C-66).
pub fn check_frts_detail<M: Mmio + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    frts: (u64, u64),
) -> GpuResult<FrtsCheck> {
    let err = bar.read(bus::NV_PBUS_SW_SCRATCH(0xe))? >> 16;
    if err != 0 {
        return Err(GpuError::HardwareFault { code: err });
    }
    let hi = bar.read(fb::NV_PFB_PRI_MMU_WPR2_ADDR_HI)?;
    if hi == 0 {
        return Err(GpuError::HardwareFault { code: 0 });
    }
    let lo = bar.read(fb::NV_PFB_PRI_MMU_WPR2_ADDR_LO)?;
    compare_frts(lo, hi, frts)
}

/// [`check_frts_full`] without the raw values (kept for compatibility).
/// C-66: accepts every form of [`Wpr2EndForm`], including the board's
/// last 128 KiB block.
pub fn check_frts<M: Mmio + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    frts: (u64, u64),
) -> GpuResult<Wpr2> {
    check_frts_full(bar, frts).map(|c| c.wpr2)
}

/// Kept for compatibility: the loader is now `hs_load::dma_load`.
pub fn load_not_written<M: Mmio + ?Sized>(
    _bar: &mut GuardedBar0<'_, M>,
    _e: Engine,
) -> GpuResult<()> {
    Err(GpuError::NotImplemented(
        "I6/I7: incarcarea ucode-ului semnat in falcon (DMA) nu e scrisa",
    ))
}

/// I6: FWSEC-FRTS on the GSP falcon. `load` puts FWSEC in the falcon.
pub fn fwsec_frts<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    load: impl FnOnce(&mut GuardedBar0<'_, M>, Engine) -> GpuResult<()>,
    frts: (u64, u64),
) -> GpuResult<Wpr2> {
    fwsec_frts_with(bar, clock, |b, _, e| load(b, e), frts)
}

/// I6 with the real loader: FWSEC from `image` over DMA ([`super::hs_load`]).
pub fn fwsec_frts_dma<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    ucode: &super::hs_load::HsUcode,
    image: &super::hs_load::DmaImage,
    frts: (u64, u64),
) -> GpuResult<Wpr2> {
    fwsec_frts_with(
        bar,
        clock,
        |b, c, e| super::hs_load::dma_load(b, c, e, ucode, image),
        frts,
    )
}

fn fwsec_frts_with<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    load: impl FnOnce(&mut GuardedBar0<'_, M>, &mut C, Engine) -> GpuResult<()>,
    frts: (u64, u64),
) -> GpuResult<Wpr2> {
    if let Some(w) = read_wpr2(bar)? {
        // Already up: FWSEC must not run twice; the board needs a reset.
        bar.journal()
            .push(Event::Note("WPR2 deja ridicat inainte de FWSEC"));
        return Err(GpuError::HardwareFault {
            code: (w.end >> 12) as u32,
        });
    }
    reset(bar, clock, Engine::Gsp)?;
    select_falcon_core(bar, clock, Engine::Gsp)?;
    load(bar, clock, Engine::Gsp)?;
    run_to_halt(bar, clock, Engine::Gsp, (0, 0))?;
    let w = check_frts(bar, frts)?;
    bar.journal().push(Event::Step {
        name: "I6 FWSEC-FRTS",
        ok: true,
    });
    Ok(w)
}

/// I7: Booter on SEC2 with the address of `GspFwWprMeta` in mailbox 0/1,
/// then wait for the GSP RISC-V core to run GSP-RM.
pub fn booter_load<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    load: impl FnOnce(&mut GuardedBar0<'_, M>, Engine) -> GpuResult<()>,
    wpr_meta_addr: u64,
) -> GpuResult<()> {
    booter_load_with(bar, clock, |b, _, e| load(b, e), wpr_meta_addr)
}

/// I7 with the real loader: Booter from `image` over DMA ([`super::hs_load`]).
pub fn booter_load_dma<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    ucode: &super::hs_load::HsUcode,
    image: &super::hs_load::DmaImage,
    wpr_meta_addr: u64,
) -> GpuResult<()> {
    booter_load_with(
        bar,
        clock,
        |b, c, e| super::hs_load::dma_load(b, c, e, ucode, image),
        wpr_meta_addr,
    )
}

fn booter_load_with<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    load: impl FnOnce(&mut GuardedBar0<'_, M>, &mut C, Engine) -> GpuResult<()>,
    wpr_meta_addr: u64,
) -> GpuResult<()> {
    if read_wpr2(bar)?.is_none() {
        return Err(GpuError::InvalidArgument(
            "I7 inainte de I6: WPR2 nu e ridicat",
        ));
    }
    reset(bar, clock, Engine::Sec2)?;
    select_falcon_core(bar, clock, Engine::Sec2)?;
    load(bar, clock, Engine::Sec2)?;
    let (mb0, _) = run_to_halt(
        bar,
        clock,
        Engine::Sec2,
        (wpr_meta_addr as u32, (wpr_meta_addr >> 32) as u32),
    )?;
    if mb0 != 0 {
        journal_i7_failure(bar, I7_FAIL_BOOTER_CODE);
        return Err(GpuError::HardwareFault { code: mb0 });
    }
    if let Err(e) = wait_riscv_active(bar, clock) {
        journal_i7_failure(bar, I7_FAIL_RISCV_OFF);
        return Err(e);
    }
    bar.journal().push(Event::Step {
        name: "I7 Booter + GSP RISC-V",
        ok: true,
    });
    Ok(())
}

/// os-295, `gpu-jurnal-011.txt` (12 Nov 2026): Booter Load halted SEC2 with
/// MAILBOX0 = 0 (its success code), `FALCON_OS` was written — and the GSP
/// RISC-V core never became active within [`RISCV_ACTIVE_US`]; `CPUCTL` read
/// `0x10` (HALTED) afterwards. The journal held only the three reads of
/// [`read_after_i7`], so *where* the boot stopped could not be told. The
/// public drivers expect the core up right after Booter (nova waits at most
/// 5 s), so a longer wait is not the fix; what was missing is the board's
/// state at that moment.
pub const I7_FAIL_RISCV_OFF: &str = "I7: Booter a terminat cu MAILBOX0 = 0, dar RISC-V-ul GSP nu a devenit activ; starea placii (doar citiri):";
/// Booter Load halted with an error code in MAILBOX0 (the code is in the
/// step's reason row; Booter's codes are not published — RM only prints them).
pub const I7_FAIL_BOOTER_CODE: &str =
    "I7: Booter a oprit SEC2 cu MAILBOX0 != 0 (eroare Booter); starea placii (doar citiri):";

/// C-103 / os-295: Booter Unload halted with an error code (journal 012: 0x88).
pub const UNLOAD_FAIL_STATE: &str =
    "C-103 Booter Unload: MAILBOX0 != 0; starea placii (doar citiri):";

/// The two registers read after a failed I7 on top of
/// [`RESUME_STATE_READS`]: the GSP core select we programmed before Booter
/// (did it stick?) and `FALCON_OS` we wrote after it (os-246: GSP mailbox
/// writes once did not stick). Both are registers I7 already writes.
pub const I7_FAIL_EXTRA_READS: [(&str, u32); 2] = [
    (
        "GSP RISCV_BCR_CTRL (0x111 = RISC-V ales)",
        f2::NV_FALCON2_GSP_BASE + riscv::NV_PRISCV_RISCV_BCR_CTRL,
    ),
    (
        "GSP FALCON_OS (570144 = versiunea scrisa)",
        gsp::NV_PGSP.lo + fv4::NV_PFALCON_FALCON_OS,
    ),
];

/// Journals `why` and the board's state after a failed I7: the seven
/// registers of [`journal_resume_state`] plus [`I7_FAIL_EXTRA_READS`], each
/// as `valoare … = 0x…`. Reads only; a register that cannot be read is
/// journaled with its priv error (diagnosis never changes the result).
pub fn journal_i7_failure<M: Mmio + ?Sized>(bar: &mut GuardedBar0<'_, M>, why: &'static str) {
    bar.journal().push(Event::Note(why));
    journal_resume_state(bar);
    for &(what, off) in I7_FAIL_EXTRA_READS.iter() {
        let value = match bar.read(off) {
            Ok(v) => v,
            Err(GpuError::HardwareFault { code }) => code,
            Err(_) => continue,
        };
        bar.journal().push(Event::Value { what, value });
    }
}

/// C-103: the mailbox arguments of a normal (non-suspend, non-GC6) Booter
/// Unload — RM `kgspExecuteBooterUnloadIfNeeded_TU102` (570.144, MIT).
pub const BOOTER_UNLOAD_MAILBOX: (u32, u32) = (0xff, 0xff);

/// C-103: what [`booter_unload`] found and did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unloaded {
    /// WPR2 was not up: nothing to tear down, Booter Unload not run (as RM).
    Wpr2NuEraRidicat,
    /// Booter Unload ran, halted with mailbox 0 = 0, and WPR2 reads 0 now.
    Wpr2Coborat,
}

/// C-103: brings the board back to the state I6 expects **without a reset of
/// the computer** — WPR2 torn down by NVIDIA's signed `booter_unload` on
/// SEC2, the same program the NVIDIA driver runs at unload and nouveau at
/// `fini`. Order read in RM 570.144 (MIT, `kgspTeardown_TU102` +
/// `kgspExecuteBooterUnloadIfNeeded_TU102`): GSP falcon reset (GSP-RM, in
/// whatever state it is, stops), SEC2 reset, Booter Unload with mailboxes
/// (0xFF, 0xFF), mailbox 0 must come back 0, then `WPR2_ADDR_HI` must read 0.
/// RM also runs FWSEC-SB before it ("put back PreOsApps", display-related);
/// we do not — nothing of ours depends on the pre-OS apps, and running a
/// second signed program with no need widens the surface. If WPR2 is not up
/// the program is not run at all (RM does the same). GSP/SEC2 windows only,
/// through the write policy. SIMULARE: never run on the board.
pub fn booter_unload<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    load: impl FnOnce(&mut GuardedBar0<'_, M>, &mut C, Engine) -> GpuResult<()>,
) -> GpuResult<Unloaded> {
    let Some(w) = read_wpr2(bar)? else {
        bar.journal().push(Event::Note(
            "C-103 Booter Unload: WPR2 nu e ridicat, nimic de coborat (sarit, ca RM)",
        ));
        return Ok(Unloaded::Wpr2NuEraRidicat);
    };
    bar.journal().push(Event::Value {
        what: "C-103 Booter Unload: WPR2 inainte (inceput >> 12)",
        value: (w.start >> 12) as u32,
    });
    bar.journal().push(Event::Value {
        what: "C-103 Booter Unload: WPR2 inainte (capat >> 12)",
        value: (w.end >> 12) as u32,
    });
    // GSP-RM (or what is left of it) stops first: RM resets the GSP falcon
    // before the teardown programs run.
    reset(bar, clock, Engine::Gsp)?;
    reset(bar, clock, Engine::Sec2)?;
    select_falcon_core(bar, clock, Engine::Sec2)?;
    load(bar, clock, Engine::Sec2)?;
    let (mb0, _) = run_to_halt(bar, clock, Engine::Sec2, BOOTER_UNLOAD_MAILBOX)?;
    if mb0 != 0 {
        bar.journal().push(Event::Value {
            what: "C-103 Booter Unload: MAILBOX0 (Booter a dat eroare)",
            value: mb0,
        });
        // os-295, `gpu-jurnal-012.txt` (12 Nov 2026): the board answered
        // 0x88 here, after an I7 whose GSP core never ran GSP-RM; the code is
        // not published (RM only prints it). The state rows give the next
        // such journal what this one lacked. Reads only.
        journal_i7_failure(bar, UNLOAD_FAIL_STATE);
        return Err(GpuError::HardwareFault { code: mb0 });
    }
    if let Some(still) = read_wpr2(bar)? {
        bar.journal().push(Event::Value {
            what: "C-103 Booter Unload: WPR2 inca ridicat dupa Booter (capat >> 12)",
            value: (still.end >> 12) as u32,
        });
        return Err(GpuError::InvalidArgument(
            "C-103 Booter Unload: WPR2 e inca ridicat dupa Booter Unload",
        ));
    }
    bar.journal().push(Event::Step {
        name: "C-103 Booter Unload: WPR2 coborat",
        ok: true,
    });
    Ok(Unloaded::Wpr2Coborat)
}

/// C-103 with the real loader: `booter_unload` from `image` over DMA.
pub fn booter_unload_dma<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    ucode: &super::hs_load::HsUcode,
    image: &super::hs_load::DmaImage,
) -> GpuResult<Unloaded> {
    booter_unload(bar, clock, |b, c, e| {
        super::hs_load::dma_load(b, c, e, ucode, image)
    })
}

pub fn wait_riscv_active<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
) -> GpuResult<()> {
    wait_riscv_active_for(bar, clock, RISCV_ACTIVE_US)
}

/// `wait_riscv_active` with a caller-chosen bound (C-99: the sequencer's
/// CORE_RESUME waits long, see `sequencer::LONG_WAIT_US`).
pub fn wait_riscv_active_for<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    timeout_us: u64,
) -> GpuResult<()> {
    poll(
        bar,
        clock,
        timeout_us,
        f2::NV_FALCON2_GSP_BASE + riscv::NV_PRISCV_RISCV_CPUCTL,
        |v| {
            riscv::NV_PRISCV_RISCV_CPUCTL_ACTIVE_STAT.get(v)
                == riscv::NV_PRISCV_RISCV_CPUCTL_ACTIVE_STAT_ACTIVE
        },
    )?;
    Ok(())
}

/// C-75: the GSP-RM version GSP-RM checks in `NV_PFALCON_FALCON_OS` after
/// Booter (570.144 → 570144; the same value the sequencer's CORE_RESUME
/// writes).
pub const GSP_RM_APP_VERSION_570_144: u32 = 570_144;

/// C-75: `NV_PRISCV_RISCV_BCR_CTRL` value that hands the GSP engine to its
/// RISC-V core: CORE_SELECT = RISCV, BRFETCH = TRUE, VALID = TRUE (the
/// `0x111` nouveau `ga102_gsp_reset` sets; field positions from
/// `dev_riscv_pri.h`, MIT).
pub const GSP_BCR_RISCV: u32 = riscv::NV_PRISCV_RISCV_BCR_CTRL_VALID_TRUE
    | (riscv::NV_PRISCV_RISCV_BCR_CTRL_CORE_SELECT_RISCV << 4)
    | (riscv::NV_PRISCV_RISCV_BCR_CTRL_BRFETCH_TRUE << 8);

/// C-75: after FWSEC-FRTS the GSP falcon is left as FWSEC left it (falcon
/// core, HS); on the board (5 Nov 2026, os-246) the LibOS address written
/// to its mailboxes then did not stick (read back 0) and GSP-RM never
/// created its queues. Order read in nouveau (`r535_gsp_oneinit`: "Reset
/// GSP into RISC-V mode", `ga102_gsp_reset`) and RM 570.144: engine reset +
/// scrubbing, then BCR_CTRL = RISC-V. Only GSP-window writes. SIMULARE.
pub fn gsp_reset_into_riscv<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
) -> GpuResult<()> {
    reset(bar, clock, Engine::Gsp)?;
    let bcr = Engine::Gsp.riscv_base() + riscv::NV_PRISCV_RISCV_BCR_CTRL;
    bar.write(bcr, GSP_BCR_RISCV)?;
    poll(bar, clock, CORE_SELECT_US, bcr, |v| {
        riscv::NV_PRISCV_RISCV_BCR_CTRL_CORE_SELECT.get(v)
            == riscv::NV_PRISCV_RISCV_BCR_CTRL_CORE_SELECT_RISCV
    })?;
    Ok(())
}

/// C-75: writes the LibOS address in GSP mailbox 0/1 and reads both back.
/// A mailbox that did not keep its value stops here (before Booter), with
/// the value read: no Booter run on a GSP that will not find its queues.
pub fn write_libos_mailboxes<M: Mmio + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    libos_addr: u64,
) -> GpuResult<()> {
    if libos_addr == 0 {
        return Err(GpuError::InvalidArgument("I7: adresa LibOS este 0"));
    }
    let g = Engine::Gsp.base();
    let want = [libos_addr as u32, (libos_addr >> 32) as u32];
    let regs = [
        fv4::NV_PFALCON_FALCON_MAILBOX0,
        fv4::NV_PFALCON_FALCON_MAILBOX1,
    ];
    for (&r, &v) in regs.iter().zip(want.iter()) {
        bar.write(g + r, v)?;
    }
    for (&r, &v) in regs.iter().zip(want.iter()) {
        let got = bar.read(g + r)?;
        if got != v {
            bar.journal().push(Event::Note(
                "I7: cutia GSP nu a pastrat adresa LibOS; opresc inainte de Booter",
            ));
            return Err(GpuError::HardwareFault { code: got });
        }
    }
    Ok(())
}

/// C-75: I7 in the order the board needs: GSP reset into RISC-V
/// ([`gsp_reset_into_riscv`]) → LibOS address in the GSP mailboxes, read
/// back ([`write_libos_mailboxes`]) → Booter on SEC2 over DMA with the
/// `GspFwWprMeta` address (as [`booter_load_dma`]) → GSP-RM version in
/// `NV_PFALCON_FALCON_OS` → wait for the RISC-V core. [`booter_load_dma`]
/// stays as it was (compatibility). Tested only in SIMULARE.
pub fn booter_load_dma_libos<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    ucode: &super::hs_load::HsUcode,
    image: &super::hs_load::DmaImage,
    wpr_meta_addr: u64,
    libos_addr: u64,
) -> GpuResult<()> {
    if read_wpr2(bar)?.is_none() {
        return Err(GpuError::InvalidArgument(
            "I7 inainte de I6: WPR2 nu e ridicat",
        ));
    }
    gsp_reset_into_riscv(bar, clock)?;
    write_libos_mailboxes(bar, libos_addr)?;
    reset(bar, clock, Engine::Sec2)?;
    select_falcon_core(bar, clock, Engine::Sec2)?;
    super::hs_load::dma_load(bar, clock, Engine::Sec2, ucode, image)?;
    let (mb0, _) = run_to_halt(
        bar,
        clock,
        Engine::Sec2,
        (wpr_meta_addr as u32, (wpr_meta_addr >> 32) as u32),
    )?;
    if mb0 != 0 {
        journal_i7_failure(bar, I7_FAIL_BOOTER_CODE);
        return Err(GpuError::HardwareFault { code: mb0 });
    }
    bar.write(
        Engine::Gsp.base() + fv4::NV_PFALCON_FALCON_OS,
        GSP_RM_APP_VERSION_570_144,
    )?;
    // os-295, journal 011: this wait expired on the board once (Booter
    // reported 0, core stayed HALTED); the state rows say what the board
    // looked like when it did. RM/nouveau check the core here without any wait.
    if let Err(e) = wait_riscv_active(bar, clock) {
        journal_i7_failure(bar, I7_FAIL_RISCV_OFF);
        return Err(e);
    }
    bar.journal().push(Event::Step {
        name: "I7 GSP RISC-V + LibOS + Booter",
        ok: true,
    });
    Ok(())
}

/// "Doorbell": tells GSP-RM there is something in the command queue
/// (`NV_PGSP_QUEUE_HEAD(0) = 0`, tinygrad/nouveau).
pub fn ring_doorbell<M: Mmio + ?Sized>(bar: &mut GuardedBar0<'_, M>) -> GpuResult<()> {
    bar.write(gsp::NV_PGSP_QUEUE_HEAD(0), 0)
}

/// C-102: the registers that tell where I8 stands, read (reads only) and
/// journaled by name as `valoare … = 0x…` — before and after SEC2 is
/// started at CORE_RESUME, and when the wait for `GSP_INIT_DONE` fails.
/// On the board of 11 Nov 2026 the journal showed only the writes, so the
/// error 0x59 could not be placed; these rows place the next one.
pub const RESUME_STATE_READS: [(&str, u32); 7] = [
    (
        "PGC6 SCRATCH_14 (bit 26 = predarea etapei 3)",
        gc6::NV_PGC6_BSI_SECURE_SCRATCH_14,
    ),
    (
        "SEC2 CPUCTL",
        sec::NV_PSEC.lo + fv4::NV_PFALCON_FALCON_CPUCTL,
    ),
    (
        "SEC2 MAILBOX0",
        sec::NV_PSEC.lo + fv4::NV_PFALCON_FALCON_MAILBOX0,
    ),
    (
        "SEC2 MAILBOX1",
        sec::NV_PSEC.lo + fv4::NV_PFALCON_FALCON_MAILBOX1,
    ),
    (
        "GSP RISCV_CPUCTL (bit 7 = activ)",
        f2::NV_FALCON2_GSP_BASE + riscv::NV_PRISCV_RISCV_CPUCTL,
    ),
    (
        "GSP MAILBOX0",
        gsp::NV_PGSP.lo + fv4::NV_PFALCON_FALCON_MAILBOX0,
    ),
    (
        "GSP MAILBOX1",
        gsp::NV_PGSP.lo + fv4::NV_PFALCON_FALCON_MAILBOX1,
    ),
];

/// C-102: reads [`RESUME_STATE_READS`] and journals each as `Event::Value`;
/// a register that cannot be read is journaled with its priv error and
/// skipped (diagnosis never stops the step).
pub fn journal_resume_state<M: Mmio + ?Sized>(bar: &mut GuardedBar0<'_, M>) {
    for &(what, off) in RESUME_STATE_READS.iter() {
        let value = match bar.read(off) {
            Ok(v) => v,
            Err(GpuError::HardwareFault { code }) => code,
            Err(_) => continue,
        };
        bar.journal().push(Event::Value { what, value });
    }
}

/// Boot stage 3 hand-off flag (`NV_PGC6_BSI_SECURE_SCRATCH_14[26]`), used by
/// the sequencer's CORE_RESUME.
pub fn stage3_handoff_done<M: Mmio + ?Sized>(bar: &mut GuardedBar0<'_, M>) -> GpuResult<bool> {
    let v = bar.read(gc6::NV_PGC6_BSI_SECURE_SCRATCH_14)?;
    Ok(
        gc6a::NV_PGC6_BSI_SECURE_SCRATCH_14_BOOT_STAGE_3_HANDOFF.get(v)
            == gc6a::NV_PGC6_BSI_SECURE_SCRATCH_14_BOOT_STAGE_3_HANDOFF_VALUE_DONE,
    )
}

/// Tests of later steps start from a board where I6 + I7 succeeded (SIM).
#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;
    use gpu_sim::clock::SimClock;
    use gpu_sim::fake_gsp::{FakeGsp, SimUcode};

    pub(crate) fn booted() -> (FakeGsp, Journal, SimClock) {
        let mut g = FakeGsp::new();
        let mut j = Journal::new(512);
        let mut c = SimClock::default();
        let frts = g.frts;
        let load = |b: &mut GuardedBar0<'_, FakeGsp>, e: Engine| {
            let u = match e {
                Engine::Gsp => SimUcode::FwsecFrts,
                Engine::Sec2 => SimUcode::BooterLoad,
            };
            b.model().sim_load(e.base(), u);
            Ok(())
        };
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            fwsec_frts(&mut b, &mut c, load, frts).unwrap();
            booter_load(&mut b, &mut c, load, 0x1_0000_0000).unwrap();
        }
        (g, j, c)
    }
}

#[cfg(test)]
mod tests_c103 {
    //! C-103 (T118): Booter Unload in SIMULARE.
    use super::tests_support::booted;
    use super::*;
    use gpu_core::diag::Journal;
    use gpu_sim::clock::SimClock;
    use gpu_sim::fake_gsp::{FakeGsp, SimFault, SimUcode};

    fn load_unload(b: &mut GuardedBar0<'_, FakeGsp>, _: &mut SimClock, e: Engine) -> GpuResult<()> {
        assert_eq!(e, Engine::Sec2);
        b.model().sim_load(e.base(), SimUcode::BooterUnload);
        Ok(())
    }

    #[test]
    fn booter_unload_tears_wpr2_down_after_a_full_boot() {
        // After I6 + I7 (WPR2 up, GSP-RM "running"): unload → WPR2 reads 0,
        // RISC-V no longer active, and I6 could run again.
        let (mut g, mut j, mut c) = booted();
        assert!(g.riscv_active());
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        assert!(read_wpr2(&mut b).unwrap().is_some());
        let r = booter_unload(&mut b, &mut c, load_unload).unwrap();
        assert_eq!(r, Unloaded::Wpr2Coborat);
        assert!(read_wpr2(&mut b).unwrap().is_none());
        let _ = b;
        assert!(!g.riscv_active());
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let k = rows
            .iter()
            .position(|r| {
                r.starts_with("valoare C-103 Booter Unload: WPR2 inainte (inceput >> 12) = ")
            })
            .unwrap();
        assert!(
            rows[k + 1].starts_with("valoare C-103 Booter Unload: WPR2 inainte (capat >> 12) = ")
        );
        assert!(rows.contains(&String::from("pas C-103 Booter Unload: WPR2 coborat: ok")));
        // the mailboxes carried RM's normal-unload arguments
        assert!(
            rows.iter()
                .any(|r| r.starts_with("bar0 w 0x840040: ") && r.contains("<- 0x000000ff"))
        );
        assert!(
            rows.iter()
                .any(|r| r.starts_with("bar0 w 0x840044: ") && r.contains("<- 0x000000ff"))
        );
        // only GSP/SEC2 windows were written (the policy), WPR2 registers never
        assert!(
            !g.writes
                .iter()
                .any(|&(o, _)| o == 0x1fa824 || o == 0x1fa828)
        );
        // and I6 can run again on the same "power-on"
        let frts = g.frts;
        let load = |b: &mut GuardedBar0<'_, FakeGsp>, e: Engine| {
            b.model().sim_load(e.base(), SimUcode::FwsecFrts);
            assert_eq!(e, Engine::Gsp);
            Ok(())
        };
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        fwsec_frts(&mut b, &mut c, load, frts).unwrap();
        assert!(read_wpr2(&mut b).unwrap().is_some());
    }

    #[test]
    fn booter_unload_is_skipped_when_wpr2_is_down() {
        let mut g = FakeGsp::new();
        let mut j = Journal::new(64);
        let mut c = SimClock::default();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let r = booter_unload(&mut b, &mut c, |_, _, _| {
            panic!("no program must be loaded")
        })
        .unwrap();
        assert_eq!(r, Unloaded::Wpr2NuEraRidicat);
        let _ = b;
        assert_eq!(g.writes.len(), 0);
        assert!(
            j.events()
                .any(|e| alloc::format!("{e}").contains("WPR2 nu e ridicat"))
        );
    }

    #[test]
    fn booter_unload_errors_are_errors() {
        // Booter halts with an error code in mailbox 0.
        let (mut g, mut j, mut c) = booted();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let wrong = |b: &mut GuardedBar0<'_, FakeGsp>, _: &mut SimClock, e: Engine| {
            // a program for the other falcon: the SIM "boot ROM" halts it
            // with its error code (0xdead), never 0
            b.model().sim_load(e.base(), SimUcode::FwsecFrts);
            Ok(())
        };
        let r = booter_unload(&mut b, &mut c, wrong);
        assert!(
            matches!(r, Err(GpuError::HardwareFault { code: 0xdead })),
            "{r:?}"
        );
        let _ = b;
        assert!(
            j.events()
                .any(|e| alloc::format!("{e}").contains("MAILBOX0 (Booter a dat eroare)"))
        );
        // WPR2 stays up after a Booter that said "ok": refused, named.
        let (mut g, mut j, mut c) = booted();
        g.fault = SimFault::Wpr2Stuck;
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let r = booter_unload(&mut b, &mut c, load_unload);
        assert!(
            matches!(r, Err(GpuError::InvalidArgument(m)) if m.contains("inca ridicat")),
            "{r:?}"
        );
        assert!(read_wpr2(&mut b).unwrap().is_some());
        let _ = b;
        assert!(
            j.events()
                .any(|e| alloc::format!("{e}").contains("WPR2 inca ridicat dupa Booter"))
        );
    }

    /// os-295, journal 012 (T125c): after a Booter Unload that halts with a
    /// code, the journal names the code and then the board's state — the
    /// nine `valoare` rows — with no write journaled by the diagnosis. SIMULARE.
    #[test]
    fn booter_unload_error_journals_the_board_state() {
        let (mut g, mut j, mut c) = booted();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let wrong = |b: &mut GuardedBar0<'_, FakeGsp>, _: &mut SimClock, e: Engine| {
            b.model().sim_load(e.base(), SimUcode::FwsecFrts);
            Ok(())
        };
        assert!(booter_unload(&mut b, &mut c, wrong).is_err());
        let _ = b;
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let k = rows
            .iter()
            .position(|r| r.contains(UNLOAD_FAIL_STATE))
            .expect("the state note");
        assert!(rows[k - 1].contains("MAILBOX0 (Booter a dat eroare) = 0x0000dead"));
        super::tests::assert_state_rows(&rows[k + 1..], 9);
        assert!(
            !rows[k..].iter().any(|r| r.starts_with("bar0 w")),
            "the diagnosis wrote nothing"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_sim::clock::SimClock;
    use gpu_sim::fake_gsp::{FakeGsp, SimFault, SimUcode};
    use gpu_sim::mmio::SimMmio;

    const FRTS: (u64, u64) = ((8 << 30) - (2 << 20), (8 << 30) - (1 << 20));
    const META: u64 = 0x1_2345_6000;

    /// os-295 (T125): `rows` starts with the `n` state rows a failed I7 /
    /// Booter Unload journals — the names of [`RESUME_STATE_READS`] then
    /// [`I7_FAIL_EXTRA_READS`], in order, each `valoare … = 0x…`.
    pub(super) fn assert_state_rows(rows: &[String], n: usize) {
        let names: Vec<&str> = RESUME_STATE_READS
            .iter()
            .chain(I7_FAIL_EXTRA_READS.iter())
            .map(|&(what, _)| what)
            .collect();
        assert_eq!(names.len(), n);
        assert!(rows.len() >= n, "only {} rows: {rows:?}", rows.len());
        for (row, what) in rows.iter().zip(names) {
            let want = alloc::format!("valoare {what} = 0x");
            assert!(row.starts_with(&want), "row {row:?}, wanted {want:?}");
        }
    }

    /// os-295, journal 011 (T125a): Booter halts with 0, the GSP RISC-V core
    /// stays off — the error is still the timeout, and right after it the
    /// journal names the moment and the board's state (nine reads, no write):
    /// SEC2 halted, GSP CPUCTL without bit 7, the core select still RISC-V.
    /// SIMULARE (the board's 12 Nov journal held only three reads).
    #[test]
    fn i7_with_riscv_off_journals_the_board_state() {
        let mut g = FakeGsp::new();
        g.fault = SimFault::RiscvStaysOff;
        let mut j = Journal::new(512);
        let mut c = SimClock::default();
        let r = {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            fwsec_frts(&mut b, &mut c, load_ok, FRTS).unwrap();
            booter_load(&mut b, &mut c, load_ok, META)
        };
        assert!(matches!(r, Err(GpuError::Timeout { .. })), "{r:?}");
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let k = rows
            .iter()
            .position(|r| r.contains(I7_FAIL_RISCV_OFF))
            .expect("the state note");
        assert!(rows[k].starts_with("nota: "));
        assert_state_rows(&rows[k + 1..], 9);
        let value = |what: &str| -> u32 {
            let row = rows[k + 1..]
                .iter()
                .find(|r| r.contains(what))
                .unwrap_or_else(|| panic!("no row for {what}"));
            u32::from_str_radix(row.rsplit("0x").next().unwrap(), 16).unwrap()
        };
        assert_eq!(value("GSP RISCV_CPUCTL") & (1 << 7), 0, "core off");
        assert_eq!(value("SEC2 MAILBOX0"), 0, "Booter's success code");
        assert!(
            rows[k + 1..]
                .iter()
                .any(|r| r.contains("GSP RISCV_BCR_CTRL"))
        );
        assert!(
            !rows[k..].iter().any(|r| r.starts_with("bar0 w")),
            "the diagnosis wrote nothing"
        );
        // Nothing of this on a passing I7.
        let mut g = FakeGsp::new();
        let mut j = Journal::new(512);
        let mut c = SimClock::default();
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            fwsec_frts(&mut b, &mut c, load_ok, FRTS).unwrap();
            booter_load(&mut b, &mut c, load_ok, META).unwrap();
        }
        assert!(
            !j.events()
                .any(|e| alloc::format!("{e}").contains("starea placii"))
        );
    }

    /// os-295 (T125b): a Booter that halts with an error code journals the
    /// other note and the same state rows. SIMULARE.
    #[test]
    fn i7_with_booter_code_journals_the_board_state() {
        let mut g = FakeGsp::new();
        g.frts = FRTS;
        // A meta without the magic: the SIM Booter refuses it with 0xbad2.
        g.wpr_meta_mem = Some((META, alloc::vec![0u8; 4096]));
        let mut j = Journal::new(512);
        let mut c = SimClock::default();
        let r = {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            fwsec_frts(&mut b, &mut c, load_ok, FRTS).unwrap();
            booter_load(&mut b, &mut c, load_ok, META)
        };
        assert!(
            matches!(r, Err(GpuError::HardwareFault { code: 0xbad2 })),
            "{r:?}"
        );
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let k = rows
            .iter()
            .position(|r| r.contains(I7_FAIL_BOOTER_CODE))
            .expect("the state note");
        assert_state_rows(&rows[k + 1..], 9);
        assert!(
            rows[k + 1..]
                .iter()
                .any(|r| r.contains("SEC2 MAILBOX0 = 0x0000bad2"))
        );
        assert!(!rows.iter().any(|r| r.contains(I7_FAIL_RISCV_OFF)));
    }

    /// C-75 (T87): after FWSEC-FRTS the simulated GSP drops mailbox writes
    /// (as the board on 5 Nov): `write_libos_mailboxes` notices it and
    /// stops before any SEC2 write; after `gsp_reset_into_riscv` the
    /// address is kept and the core reads RISC-V. SIMULARE.
    #[test]
    fn libos_mailboxes_need_the_gsp_reset_after_fwsec() {
        let mut g = FakeGsp::new();
        let mut j = Journal::new(512);
        let mut c = SimClock::default();
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            fwsec_frts(&mut b, &mut c, load_ok, FRTS).unwrap();
        }
        assert!(g.gsp_mailbox_locked);
        let n = g.writes.len();
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            assert_eq!(
                write_libos_mailboxes(&mut b, 0x2_0000_0000),
                Err(GpuError::HardwareFault { code: 0 })
            );
        }
        assert!(
            g.writes[n..]
                .iter()
                .all(|&(o, _)| (0x110000..0x114000).contains(&o))
        );
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            gsp_reset_into_riscv(&mut b, &mut c).unwrap();
            write_libos_mailboxes(&mut b, 0x2_0000_0000).unwrap();
            assert_eq!(
                write_libos_mailboxes(&mut b, 0),
                Err(GpuError::InvalidArgument("I7: adresa LibOS este 0"))
            );
        }
        assert!(!g.gsp_mailbox_locked);
        let bcr = g
            .writes
            .iter()
            .rev()
            .find(|&&(o, _)| o == 0x111668)
            .unwrap();
        assert_eq!(bcr.1, 0x111);
    }

    fn load_ok(b: &mut GuardedBar0<'_, FakeGsp>, e: Engine) -> GpuResult<()> {
        let u = match e {
            Engine::Gsp => SimUcode::FwsecFrts,
            Engine::Sec2 => SimUcode::BooterLoad,
        };
        b.model().sim_load(e.base(), u);
        Ok(())
    }

    /// A BAR that is not a model, to check the permit rule.
    struct RealLike(SimMmio);
    impl Mmio for RealLike {
        fn len(&self) -> u64 {
            self.0.len()
        }
        fn read32_raw(&mut self, o: u64) -> u32 {
            self.0.read32_raw(o)
        }
        fn write32_raw(&mut self, o: u64, v: u32) {
            self.0.write32_raw(o, v)
        }
    }

    #[test]
    fn real_bar_needs_the_permit_and_writes_stay_in_the_windows() {
        let mut j = Journal::new(64);
        let mut real = RealLike(SimMmio::ga104_bar0());
        let r = GuardedBar0::new(&mut real, &mut j, None);
        assert!(matches!(r, Err(GpuError::NotPermitted(_))));
        assert!(real.0.writes.is_empty());
        let mut g = FakeGsp::new();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        assert!(matches!(b.write(0x1700, 1), Err(GpuError::NotPermitted(_))));
        assert!(matches!(b.write(0x0, 1), Err(GpuError::NotPermitted(_))));
        assert!(matches!(b.read(0x9a0000), Err(GpuError::NotPermitted(_))));
        assert_eq!(b.read(0).unwrap(), 0x1740_00a1);
        let _ = b;
        assert!(g.writes.is_empty());
    }

    /// C-71 (T92): the simulated Booter lowers WPR2 to `gspFwWprStart` from
    /// the `GspFwWprMeta` we wrote, as the board did on 5 Nov 2026 (after I7:
    /// `0x01f41000/0x01ffee00`, HI the same as after I6), so
    /// `FbLayout::free_after_i7` passes on the model too. A meta with a bad
    /// magic: Booter refuses (SIM code), WPR2 stays FRTS only, the free zone
    /// is refused. SIMULARE.
    #[test]
    fn booter_widens_wpr2_from_the_meta_like_the_board() {
        use crate::fb_layout::{FbLayout, FbParams, GIB, MIB};
        use crate::gsp::wpr_meta::{self, SysmemInputs};
        let l = FbLayout::compute(&FbParams {
            fb_size: 8 * GIB,
            vga_workspace_reg: None,
            bootloader_size: 24_684,
            gsp_elf_size: 60 * MIB,
        })
        .unwrap();
        let frts = (l.frts.start, l.frts.end);
        let inputs = SysmemInputs {
            radix3_elf_addr: 0x0200_0000,
            radix3_elf_size: 60 * MIB,
            bootloader_addr: 0x0600_0000,
            bootloader_size: 24_684,
            bootloader_code_offset: 0x100,
            bootloader_data_offset: 0x4000,
            bootloader_manifest_offset: 0x5000,
            signature_addr: 0x0610_0000,
            signature_size: 4096,
        };
        let meta = wpr_meta::encode(&l, &inputs).unwrap();
        let board = decode_wpr2(0x01f4_1000, 0x01ff_ee00);
        let run = |meta: &[u8]| {
            let mut g = FakeGsp::new();
            g.frts = frts;
            g.wpr_meta_mem = Some((META, meta.to_vec()));
            let mut j = Journal::new(256);
            let mut c = SimClock::default();
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            fwsec_frts(&mut b, &mut c, load_ok, frts).unwrap();
            let after_i6 = read_wpr2(&mut b).unwrap().unwrap();
            let r = booter_load(&mut b, &mut c, load_ok, META);
            let after_i7 = read_wpr2(&mut b).unwrap().unwrap();
            let mb0 = b.read(0x840040).unwrap();
            (after_i6, r, after_i7, mb0)
        };
        let (i6, r, i7, mb0) = run(&meta);
        assert!(r.is_ok());
        assert_eq!(mb0, 0);
        assert_eq!((i6.start, i6.end), (frts.0, board.end));
        assert_eq!(i7, board);
        assert_eq!(l.free_after_i7((i7.start, i7.end)).unwrap(), l.free_below());
        let mut bad = meta;
        bad[0] ^= 1;
        let (_, r, i7, mb0) = run(&bad);
        assert!(r.is_err());
        assert_eq!(mb0, 0xbad2);
        assert_eq!(i7.start, frts.0);
        assert!(l.free_after_i7((i7.start, i7.end)).is_err());
    }

    #[test]
    fn i6_i7_full_sequence_simulated() {
        let mut g = FakeGsp::new();
        g.frts = FRTS;
        let mut j = Journal::new(256);
        let mut c = SimClock::default();
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            let w = fwsec_frts(&mut b, &mut c, load_ok, FRTS).unwrap();
            assert_eq!((w.start, w.end), FRTS);
            booter_load(&mut b, &mut c, load_ok, META).unwrap();
            assert!(stage3_handoff_done(&mut b).unwrap());
            ring_doorbell(&mut b).unwrap();
        }
        assert!(g.riscv_active());
        assert_eq!(g.booter_arg, Some(META));
        assert_eq!(g.doorbells, 1);
        // Every write landed inside GSP/SEC2 and is journaled before/after.
        assert!(g.writes.iter().all(|&(o, _)| in_write_ranges(o)));
        assert_eq!(j.writes(), g.writes.len());
        // Bounded: simulated time well under the sum of the timeouts.
        assert!(c.now < 10_000, "{} us", c.now);
    }

    #[test]
    fn i6_on_the_fake_with_both_hi_forms_and_a_wrong_end() {
        // SIM: the board's start (3 Nov 2026) with each HI encoding.
        let frts = (0x1_ffe0_0000u64, 0x1_fff0_0000u64);
        for (last_page, form) in [
            (true, Wpr2End::InclusiveLastPage),
            (false, Wpr2End::Exclusive),
        ] {
            let mut g = FakeGsp::new();
            g.frts = frts;
            g.wpr2_hi_last_page = last_page;
            g.wpr2_hi_back = None;
            let mut j = Journal::new(256);
            let mut c = SimClock::default();
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            let w = fwsec_frts(&mut b, &mut c, load_ok, frts).unwrap();
            assert_eq!((w.start, w.end), frts);
            assert_eq!(check_frts_detail(&mut b, frts).unwrap().end_form, form);
        }
        // Same start, end 4 KiB further than asked: fails, and names HI.
        let mut g = FakeGsp::new();
        g.frts = (frts.0, frts.1 + 0x2000);
        g.wpr2_hi_back = None;
        let mut j = Journal::new(256);
        let mut c = SimClock::default();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let e = fwsec_frts(&mut b, &mut c, load_ok, frts).unwrap_err();
        assert_eq!(
            format!("{e}"),
            "verificare esuata la 0x1fa828: asteptat 0x1fff00, citit 0x1fff01"
        );
    }

    /// C-66 (T70): the board's values of 4 Nov 2026 (os-222): LO 0x1ffe00,
    /// HI 0x1ffee0 = the last 128 KiB block of FRTS 0x1ffe00000..0x1fff00000.
    /// SIMULARE: pure comparison, no board.
    #[test]
    fn frts_last_128k_block_from_the_board_passes() {
        let frts = (0x1_ffe0_0000u64, 0x1_fff0_0000u64);
        let enc = |a: u64| ((a >> 12) << 4) as u32;
        let (lo, hi) = (enc(frts.0), enc(0x1_ffee_0000));
        // The message of 4 Nov, as the older rule still gives it.
        let e = compare_frts(lo, hi, frts).unwrap_err();
        assert_eq!(
            format!("{e}"),
            "verificare esuata la 0x1fa828: asteptat 0x1fff00, citit 0x1ffee0"
        );
        let c = compare_frts_full(lo, hi, frts).unwrap();
        assert_eq!(c.form, Wpr2EndForm::InclusiveLastBlock128K);
        assert_eq!((c.wpr2.start, c.wpr2.end), frts);
        assert_eq!((c.lo_raw >> 4, c.hi_raw >> 4), (0x1ffe00, 0x1ffee0));
        // The two older forms still pass with the new rule.
        let f = |h: u64| compare_frts_full(lo, enc(h), frts).map(|c| c.form);
        assert_eq!(f(frts.1), Ok(Wpr2EndForm::Exclusive));
        assert_eq!(f(frts.1 - 0x1000), Ok(Wpr2EndForm::InclusiveLastPage));
        // Anything else between them still fails, at HI.
        for bad in [
            frts.1 - 0x2000,
            frts.1 - 0x1f000,
            frts.1 - 0x21000,
            frts.1 - 0x40000,
            frts.0,
            frts.1 + 0x1000,
        ] {
            assert_eq!(
                f(bad),
                Err(GpuError::VerifyFailed {
                    offset: 0x1fa828,
                    expected: 0x1fff00,
                    actual: (bad >> 12) as u32,
                })
            );
        }
        // A region smaller than one block, or not 128 KiB aligned: no block form.
        let small = (frts.0, frts.0 + 0x10000);
        assert!(compare_frts_full(lo, enc(frts.0 - 0x10000), small).is_err());
        let odd = (frts.0, frts.1 - 0x1000);
        assert!(compare_frts_full(lo, enc(odd.1 - 0x20000), odd).is_err());
        // Wrong start: still LO.
        assert!(matches!(
            compare_frts_full(enc(frts.0 + 0x1000), hi, frts),
            Err(GpuError::VerifyFailed {
                offset: 0x1fa824,
                ..
            })
        ));
        assert_eq!(c.form.label(), "ultimul bloc de 128 KiB");
    }

    /// C-66 (T71): I6 on the fake with the board's 128 KiB form (the default
    /// of `FakeGsp` now) passes `fwsec_frts`, and the read-back after I6 works
    /// with reads only — no permit, nothing written, 3 reads journaled.
    /// SIMULARE.
    #[test]
    fn i6_with_the_board_hi_and_read_back_without_permit() {
        let frts = (0x1_ffe0_0000u64, 0x1_fff0_0000u64);
        let mut g = FakeGsp::new();
        g.frts = frts;
        let mut j = Journal::new(256);
        let mut c = SimClock::default();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let w = fwsec_frts(&mut b, &mut c, load_ok, frts).unwrap();
        assert_eq!((w.start, w.end), frts);
        assert_eq!(
            check_frts_full(&mut b, frts).unwrap().form,
            Wpr2EndForm::InclusiveLastBlock128K
        );
        // The older detailed check (2 forms) still refuses it, at HI.
        assert!(format!("{}", check_frts_detail(&mut b, frts).unwrap_err()).contains("0x1fa828"));
        let writes = g.writes.len();
        let mut j2 = Journal::new(16);
        let r = read_frts_after_i6(&mut g, &mut j2, frts).unwrap();
        assert_eq!(r.form, Wpr2EndForm::InclusiveLastBlock128K);
        assert_eq!(r.hi_raw >> 4, 0x1ffee0);
        assert_eq!(g.writes.len(), writes, "read-back must not write");
        assert_eq!(j2.writes(), 0);
        let reads: Vec<u64> = j2
            .events()
            .filter_map(|e| match *e {
                Event::MmioRead { offset, .. } => Some(offset),
                _ => None,
            })
            .collect();
        let scratch = u64::from(bus::NV_PBUS_SW_SCRATCH(0xe));
        assert_eq!(reads, [scratch, 0x1fa828, 0x1fa824]);
        // Only status registers: a GSP register is refused by the read-only bar.
        let mut sb = StatusReadBar0::new(&mut g, &mut j2);
        assert!(matches!(sb.read(0x110100), Err(GpuError::NotPermitted(_))));
        // Cold board (WPR2 down): the read-back reports it, does not pass.
        let mut cold = FakeGsp::new();
        let mut j3 = Journal::new(16);
        assert_eq!(
            read_frts_after_i6(&mut cold, &mut j3, frts),
            Err(GpuError::HardwareFault { code: 0 })
        );
    }

    #[test]
    fn every_failure_is_reported_not_skipped() {
        let run = |fault: SimFault| {
            let mut g = FakeGsp::new();
            g.fault = fault;
            let mut j = Journal::new(256);
            let mut c = SimClock::default();
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            fwsec_frts(&mut b, &mut c, load_ok, FRTS)
                .and_then(|_| booter_load(&mut b, &mut c, load_ok, META))
        };
        assert!(matches!(
            run(SimFault::ScrubStuck),
            Err(GpuError::Timeout { .. })
        ));
        assert!(matches!(
            run(SimFault::FrtsError(0x0c)),
            Err(GpuError::HardwareFault { code: 0x0c })
        ));
        assert!(matches!(
            run(SimFault::NoWpr2),
            Err(GpuError::HardwareFault { code: 0 })
        ));
        assert!(matches!(
            run(SimFault::NeverHalts),
            Err(GpuError::Timeout { .. })
        ));
        assert!(matches!(
            run(SimFault::RiscvStaysOff),
            Err(GpuError::Timeout { .. })
        ));
        assert!(run(SimFault::None).is_ok());
    }

    /// C-96 (T108), from the board (10 Nov 2026, os-276): after I7 the GSP
    /// runs GSP-RM; the sequencer's CORE_RESET resets it and the first
    /// HWCFG2 read answers `0xbadf5620`. Before the fix the first such read
    /// stopped I8 ("eroare hardware 0xbadf5620"); now the scrub wait polls
    /// through it as RM/nouveau do, the reset finishes and the core switch
    /// to falcon follows. A priv error that never clears is still reported,
    /// with the board's value, not as a bare timeout. SIMULARE.
    #[test]
    fn reset_of_a_running_gsp_rm_polls_through_the_priv_error() {
        let (mut g, _j, _c) = tests_support::booted();
        assert!(g.riscv_active());
        // The model answers the board's value right after the release.
        g.write32_raw(0x1103c0, 1);
        g.write32_raw(0x1103c0, 0);
        assert_eq!(
            g.read32_raw(0x1100f4),
            gpu_sim::fake_gsp::PRIV_ERROR_AFTER_RESET
        );
        // Fresh booted model: the whole reset, as CORE_RESET runs it.
        let (mut g, mut j2, mut c2) = tests_support::booted();
        let n = j2.len();
        {
            let mut b = GuardedBar0::new(&mut g, &mut j2, None).unwrap();
            reset(&mut b, &mut c2, Engine::Gsp).unwrap();
            switch_to_falcon(&mut b, &mut c2, Engine::Gsp).unwrap();
        }
        assert!(!g.riscv_active());
        // SIM: RESET_READY is always set and HWCFG2 is readable before the
        // reset, so no pre-reset note is expected here (on the board either
        // note may appear; both are journaled, never silent). No step failed.
        assert!(
            j2.events()
                .skip(n)
                .all(|e| !matches!(e, Event::Step { ok: false, .. }))
        );
        let bcr = g
            .writes
            .iter()
            .rev()
            .find(|&&(o, _)| o == 0x111668)
            .unwrap();
        assert_eq!(bcr.1 & 0x10, 0, "core switched to falcon: {bcr:x?}");
        // Priv error that never clears: the board's value comes back.
        let (mut g, mut j3, mut c3) = tests_support::booted();
        g.fault = SimFault::PrivErrorStuck;
        let mut b = GuardedBar0::new(&mut g, &mut j3, None).unwrap();
        let t0 = c3.now;
        assert_eq!(
            reset(&mut b, &mut c3, Engine::Gsp),
            Err(GpuError::HardwareFault {
                code: gpu_sim::fake_gsp::PRIV_ERROR_AFTER_RESET
            })
        );
        assert!(c3.now - t0 >= SCRUB_US, "waited the whole RM bound");
    }

    #[test]
    fn wrong_order_wrong_region_and_missing_loader_are_refused() {
        let mut g = FakeGsp::new();
        let mut j = Journal::new(256);
        let mut c = SimClock::default();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        // Booter before FWSEC.
        assert!(matches!(
            booter_load(&mut b, &mut c, load_ok, META),
            Err(GpuError::InvalidArgument(_))
        ));
        // The real loader is not written: an error, never a fake success.
        assert!(matches!(
            fwsec_frts(&mut b, &mut c, load_not_written, FRTS),
            Err(GpuError::NotImplemented(_))
        ));
        // Nothing loaded: FWSEC "halts" without creating WPR2.
        assert!(matches!(
            fwsec_frts(&mut b, &mut c, |_, _| Ok(()), FRTS),
            Err(GpuError::HardwareFault { code: 0 })
        ));
        // FRTS somewhere else than asked.
        let other = (FRTS.0 - (1 << 20), FRTS.1 - (1 << 20));
        assert!(matches!(
            fwsec_frts(&mut b, &mut c, load_ok, other),
            Err(GpuError::VerifyFailed { .. })
        ));
        // And now WPR2 is up: FWSEC a second time is refused.
        assert!(matches!(
            fwsec_frts(&mut b, &mut c, load_ok, FRTS),
            Err(GpuError::HardwareFault { .. })
        ));
        // Booter with nothing loaded: SIM error 0xdead in mailbox 0.
        assert!(matches!(
            booter_load(&mut b, &mut c, |_, _| Ok(()), META),
            Err(GpuError::HardwareFault { code: 0xdead })
        ));
        // Booter with a zero WPR-meta address: SIM error 0xbad1.
        assert!(matches!(
            booter_load(&mut b, &mut c, load_ok, 0),
            Err(GpuError::HardwareFault { code: 0xbad1 })
        ));
    }

    #[test]
    fn frts_end_exclusive_or_last_page_and_hi_reported() {
        // Board values, 3 Nov 2026: FRTS 0x1ffe00000..0x1fff00000, LO = start.
        let frts = (0x1_ffe0_0000u64, 0x1_fff0_0000u64);
        let enc = |a: u64| ((a >> 12) << 4) as u32;
        let lo = enc(frts.0);
        let c = compare_frts(lo, enc(frts.1), frts).unwrap();
        assert_eq!(c.end_form, Wpr2End::Exclusive);
        assert_eq!((c.wpr2.start, c.wpr2.end), frts);
        let c = compare_frts(lo, enc(frts.1 - 0x1000), frts).unwrap();
        assert_eq!(c.end_form, Wpr2End::InclusiveLastPage);
        assert_eq!((c.wpr2.start, c.wpr2.end), frts);
        assert_eq!(c.hi_raw, enc(0x1_ffef_f000));
        // Any other end: reported at HI, not at LO (the 3 Nov message
        // "la 0x1fa824: asteptat 0x1ffe00, citit 0x1ffe00" named the wrong register).
        for bad in [frts.1 + 0x1000, frts.1 - 0x2000, frts.0, 0x2_0000_0000] {
            let e = compare_frts(lo, enc(bad), frts).unwrap_err();
            assert_eq!(
                e,
                GpuError::VerifyFailed {
                    offset: 0x1fa828,
                    expected: 0x1fff00,
                    actual: (bad >> 12) as u32,
                }
            );
            assert!(format!("{e}").contains("0x1fa828"));
        }
        // Wrong start: still LO.
        let e = compare_frts(enc(frts.0 + 0x1000), enc(frts.1), frts).unwrap_err();
        assert!(matches!(
            e,
            GpuError::VerifyFailed {
                offset: 0x1fa824,
                ..
            }
        ));
    }

    #[test]
    fn wpr2_register_encoding() {
        // VAL = bits 31:4, address = VAL << 12 (dev_fb.h ALIGNMENT 0xc).
        let w = decode_wpr2(0x0001_fe00 << 4, 0x0001_ff00 << 4);
        assert_eq!(w.start, 0x1fe0_0000);
        assert_eq!(w.end, 0x1ff0_0000);
    }
}
