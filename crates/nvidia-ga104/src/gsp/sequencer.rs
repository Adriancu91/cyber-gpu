//! I8d: the CPU sequencer (`GSP_RUN_CPU_SEQUENCER`) and the wait for
//! `GSP_INIT_DONE`. **SIMULARE only.**
//!
//! While it boots, GSP-RM asks the driver to run a short list of register
//! commands. Opcodes and payload sizes are generated from NVIDIA's MIT header
//! `rmgspseq.h` (570.144, crate `nvidia-rm570`); what each command means
//! follows nouveau `r535_gsp_msg_run_cpu_sequencer` (MIT) — read, re-written.
//! Every register access goes through [`GuardedBar0`], so a command outside
//! the GSP/SEC2 windows is refused (the boot stops with an error) instead of
//! being executed: the firmware does not get to pick arbitrary registers.

use alloc::vec::Vec;
use core::mem::size_of;

use gpu_core::diag::Event;
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, Mmio};
use nvidia_rm570::regs::ga102_dev_falcon_v4 as fv4;
use nvidia_rm570::regs::ga102_dev_fbif_v4 as fbif;
use nvidia_rm570::regs::ga102_dev_gsp_addendum as gspa;
use nvidia_rm570::structs as s;

use super::falcon::{self, Engine, GuardedBar0};
use super::msgq::{CpuQueues, SharedMem};
use super::rpc::{self, event};

/// One decoded sequencer command.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqCmd {
    RegWrite {
        addr: u32,
        val: u32,
    },
    RegModify {
        addr: u32,
        mask: u32,
        val: u32,
    },
    RegPoll {
        addr: u32,
        mask: u32,
        val: u32,
        timeout_us: u32,
        error: u32,
    },
    DelayUs(u32),
    RegStore {
        addr: u32,
        index: u32,
    },
    CoreReset,
    CoreStart,
    CoreWaitForHalt,
    CoreResume,
}

const fn words<T>() -> usize {
    size_of::<T>() / 4
}

/// Splits the command words (variable length: opcode + payload).
pub fn parse(cmds: &[u32]) -> GpuResult<Vec<SeqCmd>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < cmds.len() {
        let op = cmds[i];
        let n = match op {
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_WRITE => {
                words::<s::GSP_SEQ_BUF_PAYLOAD_REG_WRITE>()
            }
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_MODIFY => {
                words::<s::GSP_SEQ_BUF_PAYLOAD_REG_MODIFY>()
            }
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_POLL => {
                words::<s::GSP_SEQ_BUF_PAYLOAD_REG_POLL>()
            }
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_DELAY_US => {
                words::<s::GSP_SEQ_BUF_PAYLOAD_DELAY_US>()
            }
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_STORE => {
                words::<s::GSP_SEQ_BUF_PAYLOAD_REG_STORE>()
            }
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_RESET
            | s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_START
            | s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_WAIT_FOR_HALT
            | s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_RESUME => 0,
            _ => {
                return Err(GpuError::InvalidArgument(
                    "secventiator: cod de comanda necunoscut",
                ));
            }
        };
        let p = cmds.get(i + 1..i + 1 + n).ok_or(GpuError::InvalidArgument(
            "secventiator: comanda trunchiata",
        ))?;
        out.push(match op {
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_WRITE => SeqCmd::RegWrite {
                addr: p[0],
                val: p[1],
            },
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_MODIFY => SeqCmd::RegModify {
                addr: p[0],
                mask: p[1],
                val: p[2],
            },
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_POLL => SeqCmd::RegPoll {
                addr: p[0],
                mask: p[1],
                val: p[2],
                timeout_us: p[3],
                error: p[4],
            },
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_DELAY_US => SeqCmd::DelayUs(p[0]),
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_STORE => SeqCmd::RegStore {
                addr: p[0],
                index: p[1],
            },
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_RESET => SeqCmd::CoreReset,
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_START => SeqCmd::CoreStart,
            s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_WAIT_FOR_HALT => SeqCmd::CoreWaitForHalt,
            _ => SeqCmd::CoreResume,
        });
        i += 1 + n;
    }
    Ok(out)
}

/// What CORE_RESUME needs from the boot (nouveau: the LibOS arguments
/// address and the GSP-RM application version written to falcon `OS`).
#[derive(Clone, Copy, Debug)]
pub struct SeqContext {
    pub libos_addr: u64,
    pub app_version: u32,
}

/// Default bound for REG_POLL when GSP-RM sends a timeout of 0 (DEDUS).
pub const POLL_DEFAULT_US: u64 = 4_000_000;
/// Upper bound for one DELAY_US, so a corrupt value cannot stall the boot.
pub const MAX_DELAY_US: u32 = 1_000_000;

/// Runs the commands in order; stops at the first error.
pub fn execute<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    cmds: &[SeqCmd],
    reg_save: &mut [u32; 8],
    ctx: SeqContext,
) -> GpuResult<()> {
    let gsp = Engine::Gsp.base();
    for &c in cmds {
        match c {
            SeqCmd::RegWrite { addr, val } => bar.write(addr, val)?,
            SeqCmd::RegModify { addr, mask, val } => {
                let v = bar.read(addr)?;
                bar.write(addr, (v & !mask) | val)?; // as nvkm_mask()
            }
            SeqCmd::RegPoll {
                addr,
                mask,
                val,
                timeout_us,
                error,
            } => {
                let t = if timeout_us == 0 {
                    POLL_DEFAULT_US
                } else {
                    u64::from(timeout_us)
                };
                let mut err = None;
                let r = gpu_core::hal::poll_until(clock, t, 10, || match bar.read(addr) {
                    Ok(v) if v & mask == val => Some(()),
                    Ok(_) => None,
                    Err(e) => {
                        err = Some(e);
                        Some(())
                    }
                });
                if let Some(e) = err {
                    return Err(e);
                }
                if r.is_err() {
                    return Err(GpuError::HardwareFault { code: error });
                }
            }
            SeqCmd::DelayUs(us) => {
                if us > MAX_DELAY_US {
                    return Err(GpuError::InvalidArgument("secventiator: pauza prea lunga"));
                }
                clock.delay_us(us);
            }
            SeqCmd::RegStore { addr, index } => {
                let slot = reg_save
                    .get_mut(index as usize)
                    .ok_or(GpuError::InvalidArgument(
                        "secventiator: index de salvare in afara celor 8",
                    ))?;
                *slot = bar.read(addr)?;
            }
            SeqCmd::CoreReset => {
                falcon::reset(bar, clock, Engine::Gsp)?;
                // nouveau: FBIF_CTL |= ALLOW_PHYS_NO_CTX, DMACTL = 0.
                let ctl = gspa::NV_PGSP_FBIF_BASE + fbif::NV_PFALCON_FBIF_CTL;
                let v = bar.read(ctl)?;
                bar.write(ctl, fbif::NV_PFALCON_FBIF_CTL_ALLOW_PHYS_NO_CTX.set(v, 1))?;
                bar.write(gsp + fv4::NV_PFALCON_FALCON_DMACTL, 0)?;
            }
            SeqCmd::CoreStart => {
                let cpuctl = bar.read(gsp + fv4::NV_PFALCON_FALCON_CPUCTL)?;
                let start = fv4::NV_PFALCON_FALCON_CPUCTL_STARTCPU.set(0, 1);
                let r = if fv4::NV_PFALCON_FALCON_CPUCTL_ALIAS_EN.get(cpuctl) == 1 {
                    fv4::NV_PFALCON_FALCON_CPUCTL_ALIAS
                } else {
                    fv4::NV_PFALCON_FALCON_CPUCTL
                };
                bar.write(gsp + r, start)?;
            }
            SeqCmd::CoreWaitForHalt => {
                let mut err = None;
                let r = gpu_core::hal::poll_until(clock, falcon::HALT_US, 10, || {
                    match bar.read(gsp + fv4::NV_PFALCON_FALCON_CPUCTL) {
                        Ok(v) if fv4::NV_PFALCON_FALCON_CPUCTL_HALTED.get(v) == 1 => Some(()),
                        Ok(_) => None,
                        Err(e) => {
                            err = Some(e);
                            Some(())
                        }
                    }
                });
                if let Some(e) = err {
                    return Err(e);
                }
                r?;
            }
            SeqCmd::CoreResume => {
                // nouveau: GSP mailboxes = LibOS args, start SEC2, wait the
                // stage-3 hand-off, SEC2 mailbox 0 must be 0, GSP OS = app
                // version, GSP RISC-V must be active.
                let sec2 = Engine::Sec2.base();
                bar.write(gsp + fv4::NV_PFALCON_FALCON_MAILBOX0, ctx.libos_addr as u32)?;
                bar.write(
                    gsp + fv4::NV_PFALCON_FALCON_MAILBOX1,
                    (ctx.libos_addr >> 32) as u32,
                )?;
                bar.write(
                    sec2 + fv4::NV_PFALCON_FALCON_CPUCTL,
                    fv4::NV_PFALCON_FALCON_CPUCTL_STARTCPU.set(0, 1),
                )?;
                let mut err = None;
                let r = gpu_core::hal::poll_until(clock, falcon::HALT_US, 10, || {
                    match falcon::stage3_handoff_done(bar) {
                        Ok(true) => Some(()),
                        Ok(false) => None,
                        Err(e) => {
                            err = Some(e);
                            Some(())
                        }
                    }
                });
                if let Some(e) = err {
                    return Err(e);
                }
                r?;
                let mb0 = bar.read(sec2 + fv4::NV_PFALCON_FALCON_MAILBOX0)?;
                if mb0 != 0 {
                    return Err(GpuError::HardwareFault { code: mb0 });
                }
                bar.write(gsp + fv4::NV_PFALCON_FALCON_OS, ctx.app_version)?;
                falcon::wait_riscv_active(bar, clock)?;
            }
        }
    }
    Ok(())
}

/// What happened while waiting for `GSP_INIT_DONE`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InitLog {
    pub sequencer_runs: u32,
    pub sequencer_commands: u32,
    /// Other events seen and skipped (function numbers), in order.
    pub other_events: Vec<u32>,
}

/// Bound for the whole wait (DEDUS; nouveau waits seconds).
pub const INIT_DONE_US: u64 = 10_000_000;

/// Reads the status queue until `GSP_INIT_DONE`, running every sequencer
/// request on the way. `poll_gsp` lets a simulation advance its model
/// between two empty reads; on hardware it does nothing.
pub fn wait_init_done<M: Mmio + ?Sized, C: Clock + ?Sized>(
    q: &mut CpuQueues,
    mem: &mut dyn SharedMem,
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    ctx: SeqContext,
    timeout_us: u64,
    mut poll_gsp: impl FnMut(&mut dyn SharedMem),
) -> GpuResult<InitLog> {
    let mut log = InitLog::default();
    let start = clock.now_us();
    loop {
        match q.recv_rpc(mem)? {
            Some(m) if m.function == event::GSP_INIT_DONE => {
                if m.rpc_result != 0 {
                    return Err(GpuError::HardwareFault { code: m.rpc_result });
                }
                bar.journal().push(Event::Step {
                    name: "I8 GSP_INIT_DONE",
                    ok: true,
                });
                return Ok(log);
            }
            Some(m) if m.function == event::GSP_RUN_CPU_SEQUENCER => {
                let h = rpc::decode_sequencer(&m.payload)?;
                let cmds = parse(&h.commands)?;
                let mut save = h.reg_save;
                execute(bar, clock, &cmds, &mut save, ctx)?;
                log.sequencer_runs += 1;
                log.sequencer_commands += cmds.len() as u32;
            }
            Some(m) => log.other_events.push(m.function),
            None => {
                if clock.now_us().saturating_sub(start) >= timeout_us {
                    return Err(GpuError::Timeout {
                        waited_us: clock.now_us() - start,
                    });
                }
                clock.delay_us(100);
                poll_gsp(mem);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gsp::falcon::tests_support::booted;
    use crate::gsp::msgq::QueueLayout;
    use crate::gsp::rpc::function;
    use crate::gsp::sim_peer::SimGsp;
    use gpu_core::diag::Journal;
    use gpu_sim::clock::SimClock;
    use gpu_sim::fake_gsp::FakeGsp;

    const W: u32 = 0;
    const MOD: u32 = 1;
    const POLL: u32 = 2;
    const DELAY: u32 = 3;
    const STORE: u32 = 4;
    const RESET: u32 = 5;
    const START: u32 = 6;
    const HALT: u32 = 7;
    const RESUME: u32 = 8;

    const CTX: SeqContext = SeqContext {
        libos_addr: 0x0000_0002_0000_0000,
        app_version: 570_144,
    };

    fn payload(cmds: &[u32]) -> Vec<u8> {
        // Real GSP shape: capacity (bufferSizeDWord) far above the filled
        // count (cmdIndex); only cmdIndex words are sent.
        let mut p = alloc::vec![0u8; 40];
        p[0..4].copy_from_slice(&0x4000u32.to_le_bytes());
        p[4..8].copy_from_slice(&(cmds.len() as u32).to_le_bytes());
        for w in cmds {
            p.extend_from_slice(&w.to_le_bytes());
        }
        p
    }

    #[test]
    fn opcodes_and_sizes_are_the_generated_ones() {
        assert_eq!(
            [W, MOD, POLL, DELAY, STORE, RESET, START, HALT, RESUME],
            [
                s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_WRITE,
                s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_MODIFY,
                s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_POLL,
                s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_DELAY_US,
                s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_REG_STORE,
                s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_RESET,
                s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_START,
                s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_WAIT_FOR_HALT,
                s::GSP_SEQ_BUF_OPCODE_GSP_SEQ_BUF_OPCODE_CORE_RESUME,
            ]
        );
        let c = parse(&[
            W, 0x110084, 7, MOD, 0x110084, 0xf0, 0x30, POLL, 0x110084, 0xff, 0x37, 0, 9, DELAY, 5,
            STORE, 0x110084, 3, RESET, START, HALT, RESUME,
        ])
        .unwrap();
        assert_eq!(c.len(), 9);
        assert_eq!(
            c[2],
            SeqCmd::RegPoll {
                addr: 0x110084,
                mask: 0xff,
                val: 0x37,
                timeout_us: 0,
                error: 9
            }
        );
        assert!(parse(&[9]).is_err());
        assert!(parse(&[POLL, 1, 2, 3, 4]).is_err());
    }

    #[test]
    fn register_commands_run_through_the_write_policy() {
        let mut g = FakeGsp::new();
        let mut j = Journal::new(64);
        let mut c = SimClock::default();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let mut save = [0u32; 8];
        let rm = 0x110084; // NV_PGSP falcon RM scratch (inside the window)
        let cmds = parse(&[
            W,
            rm,
            0x1234_5600,
            MOD,
            rm,
            0xff,
            0x78,
            POLL,
            rm,
            0xffff,
            0x5678,
            100,
            1,
            STORE,
            rm,
            7,
            DELAY,
            50,
        ])
        .unwrap();
        execute(&mut b, &mut c, &cmds, &mut save, CTX).unwrap();
        assert_eq!(save[7], 0x1234_5678);
        assert!(c.now >= 50);
        // A poll that never matches returns GSP's own error code.
        let p = parse(&[POLL, rm, 0xffff_ffff, 1, 200, 0x42]).unwrap();
        assert!(matches!(
            execute(&mut b, &mut c, &p, &mut save, CTX),
            Err(GpuError::HardwareFault { code: 0x42 })
        ));
        // Outside GSP/SEC2: refused, not executed.
        for bad in [
            &[W, 0x0000_0200, 1][..],      // NV_PMC_ENABLE
            &[MOD, 0x0008_8004, 4, 4][..], // PCI config mirror
            &[STORE, 0x0010_0000, 0][..],
            &[STORE, rm, 8][..],
            &[DELAY, MAX_DELAY_US + 1][..],
        ] {
            let p = parse(bad).unwrap();
            assert!(
                execute(&mut b, &mut c, &p, &mut save, CTX).is_err(),
                "{bad:x?}"
            );
        }
        let _ = b;
        assert!(!g.writes.iter().any(|&(o, _)| o == 0x200 || o == 0x88004));
    }

    #[test]
    fn boot_to_init_done_with_a_sequencer_request_simulated() {
        // I6 + I7 on the fake registers, then GSP-RM "runs": it sets up the
        // status queue, asks for a sequencer run, then sends INIT_DONE.
        let (mut g, mut j, mut c) = booted();
        let l = QueueLayout::standard();
        let mut mem = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let mut q = CpuQueues::init(&mut mem, l, &pages).unwrap();
        // The real payloads (I8c', I8c), built for the workstation's board.
        let sys = crate::gsp::sysinfo::encode(&crate::gsp::sysinfo::tests::workstation()).unwrap();
        q.send_rpc(&mut mem, function::GSP_SET_SYSTEM_INFO, &sys)
            .unwrap();
        let reg = crate::gsp::registry::encode(&crate::gsp::registry::boot_entries()).unwrap();
        q.send_rpc(&mut mem, function::SET_REGISTRY, &reg).unwrap();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        falcon::ring_doorbell(&mut b).unwrap();
        let mut peer = SimGsp::start(&mut mem, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&mem).unwrap();
        let seq = payload(&[
            STORE, 0x110040, 0, RESET, W, 0x110084, 0xc0de, START, HALT, RESUME,
        ]);
        let mut step = 0;
        let log = wait_init_done(&mut q, &mut mem, &mut b, &mut c, CTX, INIT_DONE_US, |m| {
            step += 1;
            match step {
                1 => peer.service(m),
                2 => peer.post(m, event::GSP_RUN_CPU_SEQUENCER, 0, &seq),
                3 => peer.post(m, event::OS_ERROR_LOG, 0, b"log"),
                4 => peer.post(m, event::GSP_INIT_DONE, 0, &[]),
                _ => {}
            }
        })
        .unwrap();
        assert_eq!(log.sequencer_runs, 1);
        assert_eq!(log.sequencer_commands, 6);
        // The two replies to our RPCs, then the error-log event, were skipped.
        assert_eq!(
            log.other_events,
            [
                function::GSP_SET_SYSTEM_INFO,
                function::SET_REGISTRY,
                event::OS_ERROR_LOG
            ]
        );
        let _ = b;
        assert!(g.riscv_active());
        assert!(g.writes.contains(&(0x110084, 0xc0de)));
        assert!(g.writes.contains(&(0x110080, 570_144)));
    }

    #[test]
    fn a_silent_gsp_is_a_timeout() {
        let (mut g, mut j, mut c) = booted();
        let l = QueueLayout::standard();
        let mut mem = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let mut q = CpuQueues::init(&mut mem, l, &pages).unwrap();
        let _peer = SimGsp::start(&mut mem, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&mem).unwrap();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let t0 = c.now;
        let r = wait_init_done(&mut q, &mut mem, &mut b, &mut c, CTX, 50_000, |_| {});
        assert!(matches!(r, Err(GpuError::Timeout { .. })));
        assert!(c.now - t0 >= 50_000 && c.now - t0 < 51_000);
    }
}
