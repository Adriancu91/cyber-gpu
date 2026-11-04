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
pub const SCRUB_US: u64 = 20_000;
pub const CORE_SELECT_US: u64 = 10_000;
pub const HALT_US: u64 = 2_000_000;
pub const RISCV_ACTIVE_US: u64 = 2_000_000;
const STEP_US: u32 = 10;

fn poll<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    timeout_us: u64,
    off: u32,
    mut ok: impl FnMut(u32) -> bool,
) -> GpuResult<u32> {
    let mut err = None;
    let r = poll_until(clock, timeout_us, STEP_US, || match bar.read(off) {
        Ok(v) if ok(v) => Some(v),
        Ok(_) => None,
        Err(e) => {
            err = Some(e);
            Some(0)
        }
    });
    match err {
        Some(e) => Err(e),
        None => r,
    }
}

/// Engine reset + wait for memory scrubbing (nouveau `ga102_flcn_reset_eng`
/// order: wait RESET_READY (not fatal), RESET=1, RESET=0, wait scrubbing).
pub fn reset<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    e: Engine,
) -> GpuResult<()> {
    let hwcfg2 = e.base() + fv4::NV_PFALCON_FALCON_HWCFG2;
    if poll(bar, clock, RESET_READY_US, hwcfg2, |v| {
        fv4a::NV_PFALCON_FALCON_HWCFG2_RESET_READY.get(v) == 1
    })
    .is_err()
    {
        bar.journal()
            .push(Event::Note("falcon: RESET_READY nu a venit; continui"));
    }
    bar.write(e.engine_reg(), 1)?;
    clock.delay_us(10);
    bar.write(e.engine_reg(), 0)?;
    poll(bar, clock, SCRUB_US, hwcfg2, |v| {
        fv4::NV_PFALCON_FALCON_HWCFG2_MEM_SCRUBBING.get(v)
            == fv4::NV_PFALCON_FALCON_HWCFG2_MEM_SCRUBBING_DONE
    })?;
    Ok(())
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
    let cpuctl = bar.read(b + fv4::NV_PFALCON_FALCON_CPUCTL)?;
    let start = fv4::NV_PFALCON_FALCON_CPUCTL_STARTCPU.set(0, 1);
    if fv4::NV_PFALCON_FALCON_CPUCTL_ALIAS_EN.get(cpuctl) == 1 {
        bar.write(b + fv4::NV_PFALCON_FALCON_CPUCTL_ALIAS, start)?;
    } else {
        bar.write(b + fv4::NV_PFALCON_FALCON_CPUCTL, start)?;
    }
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
        return Err(GpuError::HardwareFault { code: mb0 });
    }
    wait_riscv_active(bar, clock)?;
    bar.journal().push(Event::Step {
        name: "I7 Booter + GSP RISC-V",
        ok: true,
    });
    Ok(())
}

pub fn wait_riscv_active<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
) -> GpuResult<()> {
    poll(
        bar,
        clock,
        RISCV_ACTIVE_US,
        f2::NV_FALCON2_GSP_BASE + riscv::NV_PRISCV_RISCV_CPUCTL,
        |v| {
            riscv::NV_PRISCV_RISCV_CPUCTL_ACTIVE_STAT.get(v)
                == riscv::NV_PRISCV_RISCV_CPUCTL_ACTIVE_STAT_ACTIVE
        },
    )?;
    Ok(())
}

/// "Doorbell": tells GSP-RM there is something in the command queue
/// (`NV_PGSP_QUEUE_HEAD(0) = 0`, tinygrad/nouveau).
pub fn ring_doorbell<M: Mmio + ?Sized>(bar: &mut GuardedBar0<'_, M>) -> GpuResult<()> {
    bar.write(gsp::NV_PGSP_QUEUE_HEAD(0), 0)
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
mod tests {
    use super::*;
    use gpu_sim::clock::SimClock;
    use gpu_sim::fake_gsp::{FakeGsp, SimFault, SimUcode};
    use gpu_sim::mmio::SimMmio;

    const FRTS: (u64, u64) = ((8 << 30) - (2 << 20), (8 << 30) - (1 << 20));
    const META: u64 = 0x1_2345_6000;

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
