//! I2 — wait for GFW boot (read-only).
//!
//! Since Turing the GPU runs DEVINIT itself after power-on (FWSEC loaded from
//! the ROM); the driver only waits for it to finish. Procedure, as done by
//! nouveau `nvkm/subdev/devinit/tu102.c` (MIT) and described by nova-core
//! `gfw.rs` (GPL-2.0, read for the procedure only, no code taken):
//! 1. read SCRATCH_GROUP_05_PRIV_LEVEL_MASK; until bit 0 is set the progress
//!    register must not be read (it is still protected);
//! 2. then read SCRATCH_GROUP_05_0_GFW_BOOT; bits 7:0 == 0xff = completed;
//! 3. sleep ~1 ms and repeat; nova-core gives up after 4 s, nouveau after
//!    ~2050 iterations of 1–2 ms.
//!
//! This module only reads two allow-listed registers. It writes nothing.

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, Mmio, poll_until};

use crate::regs::{
    INIT_WINDOW, NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT as GFW_BOOT,
    NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK as PRIV_MASK, ReadOnlyBar0, is_priv_error,
};

pub const GFW_TIMEOUT_US: u64 = 4_000_000;
pub const GFW_POLL_US: u32 = 1_000;
pub const GFW_COMPLETED: u8 = 0xff;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GfwStatus {
    /// Number of polling rounds (1 = completed at the first look).
    pub polls: u32,
    pub waited_us: u64,
    pub priv_level_mask: u32,
    pub progress: u8,
}

enum Look {
    Done(GfwStatus),
    Fail(GpuError),
}

/// Poll until GFW boot reports completion, at most `timeout_us`.
pub fn wait_gfw_boot<M: Mmio + ?Sized, K: Clock + ?Sized>(
    bar0: &mut M,
    journal: &mut Journal,
    clock: &mut K,
    timeout_us: u64,
) -> GpuResult<GfwStatus> {
    if bar0.len() < INIT_WINDOW {
        return Err(GpuError::InvalidArgument(
            "fereastra BAR0 prea mica pentru I2 (trebuie 4 MiB)",
        ));
    }
    let start = clock.now_us();
    let mut polls = 0u32;
    let mut last = (0u32, 0u32);
    let r = {
        let mut bar = ReadOnlyBar0::new(bar0, journal);
        poll_until(clock, timeout_us, GFW_POLL_US, || {
            polls += 1;
            let mask = match bar.read_quiet(PRIV_MASK) {
                Ok(v) => v,
                Err(e) => return Some(Look::Fail(e)),
            };
            if mask == 0xffff_ffff {
                return Some(Look::Fail(GpuError::NotResponding));
            }
            if is_priv_error(mask) {
                return Some(Look::Fail(GpuError::HardwareFault { code: mask }));
            }
            last.0 = mask;
            if mask & 1 == 0 {
                return None; // still protected: do not touch GFW_BOOT
            }
            let progress = match bar.read_quiet(GFW_BOOT) {
                Ok(v) => v,
                Err(e) => return Some(Look::Fail(e)),
            };
            if is_priv_error(progress) {
                return Some(Look::Fail(GpuError::HardwareFault { code: progress }));
            }
            last.1 = progress;
            ((progress & 0xff) as u8 == GFW_COMPLETED).then_some(Look::Done(GfwStatus {
                polls,
                waited_us: 0,
                priv_level_mask: mask,
                progress: GFW_COMPLETED,
            }))
        })
    };
    // Journal only the final values (a poll can be thousands of reads).
    journal.push(Event::MmioRead {
        bar: 0,
        offset: PRIV_MASK.offset,
        value: last.0,
    });
    if last.0 & 1 != 0 {
        journal.push(Event::MmioRead {
            bar: 0,
            offset: GFW_BOOT.offset,
            value: last.1,
        });
    }
    let waited = clock.now_us().saturating_sub(start);
    let out = match r {
        Ok(Look::Done(mut s)) => {
            s.waited_us = waited;
            Ok(s)
        }
        Ok(Look::Fail(e)) | Err(e) => Err(e),
    };
    journal.push(Event::Step {
        name: "I2: GFW_BOOT",
        ok: out.is_ok(),
    });
    if matches!(out, Err(GpuError::Timeout { .. })) {
        journal.push(Event::Note(if last.0 & 1 == 0 {
            "I2: masca de protectie n-a fost coborata (FWSEC nu a terminat?)"
        } else {
            "I2: GFW_BOOT nu a ajuns la 0xff"
        }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_sim::clock::SimClock;
    use gpu_sim::mmio::SimMmio;

    const MASK: u64 = 0x11_8128;
    const PROG: u64 = 0x11_8234;

    fn board() -> SimMmio {
        SimMmio::new(INIT_WINDOW)
    }

    // All tests: SIMULARE. They prove the polling logic, not GA104 behaviour.

    #[test]
    fn completed_at_once() {
        let mut m = board();
        m.set(MASK, 1);
        m.set(PROG, 0xff);
        let (mut j, mut c) = (Journal::new(16), SimClock::default());
        let s = wait_gfw_boot(&mut m, &mut j, &mut c, GFW_TIMEOUT_US).unwrap();
        assert_eq!((s.polls, s.waited_us, s.progress), (1, 0, 0xff));
        assert!(m.writes.is_empty());
        assert_eq!(m.reads, [MASK, PROG]);
        assert_eq!(j.writes(), 0);
    }

    #[test]
    fn waits_for_mask_then_progress_without_early_reads() {
        let mut m = board();
        m.set(MASK, 0);
        m.set_after(MASK, 5, 1); // protection lowered after 5 more reads
        m.set(PROG, 0x40);
        m.set_after(PROG, 3, 0xff);
        let (mut j, mut c) = (Journal::new(16), SimClock::default());
        let s = wait_gfw_boot(&mut m, &mut j, &mut c, GFW_TIMEOUT_US).unwrap();
        // Rounds 1-5: mask 0. Round 6: mask 1, progress 0x40 (x3). Round 9: 0xff.
        assert_eq!(s.polls, 9);
        assert_eq!(s.waited_us, 8 * GFW_POLL_US as u64);
        // GFW_BOOT is never read while the mask bit is clear.
        let first_prog = m.reads.iter().position(|&o| o == PROG).unwrap();
        assert_eq!(first_prog, 6, "5 mask-only rounds, then mask+progress");
        assert!(m.reads[..first_prog].iter().all(|&o| o == MASK));
        assert!(m.writes.is_empty());
    }

    #[test]
    fn times_out_honestly() {
        let mut m = board();
        m.set(MASK, 1);
        m.set(PROG, 0x12);
        let (mut j, mut c) = (Journal::new(16), SimClock::default());
        let r = wait_gfw_boot(&mut m, &mut j, &mut c, GFW_TIMEOUT_US);
        assert_eq!(
            r,
            Err(GpuError::Timeout {
                waited_us: GFW_TIMEOUT_US
            })
        );
        assert!(j.events().any(|e| matches!(e, Event::Note(_))));
        assert!(
            j.events()
                .any(|e| matches!(e, Event::Step { ok: false, .. }))
        );
        assert!(m.writes.is_empty());
    }

    #[test]
    fn dead_board_and_priv_error_stop_immediately() {
        let mut m = board();
        m.absent = true;
        let (mut j, mut c) = (Journal::new(16), SimClock::default());
        assert_eq!(
            wait_gfw_boot(&mut m, &mut j, &mut c, GFW_TIMEOUT_US),
            Err(GpuError::NotResponding)
        );
        assert_eq!(m.reads.len(), 1);
        let mut m = board();
        m.set(MASK, 0xbadf_1100);
        assert_eq!(
            wait_gfw_boot(&mut m, &mut j, &mut c, GFW_TIMEOUT_US),
            Err(GpuError::HardwareFault { code: 0xbadf_1100 })
        );
    }

    #[test]
    fn needs_init_window() {
        let mut m = SimMmio::new(0x1000);
        let (mut j, mut c) = (Journal::new(4), SimClock::default());
        assert!(matches!(
            wait_gfw_boot(&mut m, &mut j, &mut c, 10),
            Err(GpuError::InvalidArgument(_))
        ));
        assert!(m.reads.is_empty());
    }
}
