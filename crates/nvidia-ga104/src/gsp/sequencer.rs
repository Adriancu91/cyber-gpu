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
use super::nocat;
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
/// C-99 (Adrian, 10 Nov 2026, os-277: "why not wait longer, until the
/// answer comes?"): the waits after the doorbell that depend on GSP-RM —
/// CORE_WAIT_FOR_HALT, CORE_RESUME (SEC2 done, RISC-V active) and
/// GSP_INIT_DONE — are bounded by this (nouveau: 2 s, RM: 4 s). A slow
/// first start then shows in the journal as `asteptat …: raspuns dupa N us`
/// instead of a timeout; the OS can cut a wait short by hand
/// (`Clock::abort_requested` → `GpuError::Aborted`).
pub const LONG_WAIT_US: u64 = 60_000_000;
/// Upper bound for one DELAY_US, so a corrupt value cannot stall the boot.
pub const MAX_DELAY_US: u32 = 1_000_000;

impl SeqCmd {
    /// The command as a journal event (C-96: every request is journaled in
    /// full before it runs, so the board's log shows what GSP-RM asked).
    pub fn event(self, index: u32) -> Event {
        let (op, n, args): (&'static str, u8, [u32; 5]) = match self {
            SeqCmd::RegWrite { addr, val } => ("WRITE", 2, [addr, val, 0, 0, 0]),
            SeqCmd::RegModify { addr, mask, val } => ("MODIFY", 3, [addr, mask, val, 0, 0]),
            SeqCmd::RegPoll {
                addr,
                mask,
                val,
                timeout_us,
                error,
            } => ("POLL", 5, [addr, mask, val, timeout_us, error]),
            SeqCmd::DelayUs(us) => ("DELAY_US", 1, [us, 0, 0, 0, 0]),
            SeqCmd::RegStore { addr, index } => ("STORE", 2, [addr, index, 0, 0, 0]),
            SeqCmd::CoreReset => ("CORE_RESET", 0, [0; 5]),
            SeqCmd::CoreStart => ("CORE_START", 0, [0; 5]),
            SeqCmd::CoreWaitForHalt => ("CORE_WAIT_FOR_HALT", 0, [0; 5]),
            SeqCmd::CoreResume => ("CORE_RESUME", 0, [0; 5]),
        };
        Event::SeqCmd { index, op, n, args }
    }
}

/// C-96: a register poll in RM's sense — a priv error (`0xbadf....`) is
/// "not yet" and the loop goes on; when the time is up with the register
/// still unreadable, the error carries that value. Other errors stop.
fn poll_reg<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    timeout_us: u64,
    mut read: impl FnMut(&mut GuardedBar0<'_, M>) -> GpuResult<bool>,
) -> GpuResult<()> {
    let mut err = None;
    let mut last_priv = None;
    let r = gpu_core::hal::poll_until(clock, timeout_us, 10, || match read(bar) {
        Ok(true) => Some(()),
        Ok(false) => {
            last_priv = None;
            None
        }
        Err(GpuError::HardwareFault { code }) if crate::regs::is_priv_error(code) => {
            last_priv = Some(code);
            None
        }
        Err(e) => {
            err = Some(e);
            Some(())
        }
    });
    match (err, r, last_priv) {
        (Some(e), _, _) => Err(e),
        (None, Err(GpuError::Timeout { .. }), Some(code)) => Err(GpuError::HardwareFault { code }),
        (None, r, _) => r,
    }
}

/// Runs the commands in order; stops at the first error. The failing
/// command is journaled again (`secv #i ...` after `pas ...: ESEC`).
pub fn execute<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    cmds: &[SeqCmd],
    reg_save: &mut [u32; 8],
    ctx: SeqContext,
) -> GpuResult<()> {
    for (i, &c) in cmds.iter().enumerate() {
        if let Err(e) = execute_one(bar, clock, c, reg_save, ctx) {
            bar.journal().push(Event::Step {
                name: "I8 secventiator: comanda de mai jos s-a oprit",
                ok: false,
            });
            bar.journal().push(c.event(i as u32));
            return Err(e);
        }
    }
    Ok(())
}

fn execute_one<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    c: SeqCmd,
    reg_save: &mut [u32; 8],
    ctx: SeqContext,
) -> GpuResult<()> {
    let gsp = Engine::Gsp.base();
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
            // RM: a poll that runs out of time reports GSP's own error code;
            // a register unreadable the whole time reports the priv error.
            match poll_reg(bar, clock, t, |b| Ok(b.read(addr)? & mask == val)) {
                Ok(()) => {}
                Err(GpuError::Timeout { .. }) => {
                    return Err(GpuError::HardwareFault { code: error });
                }
                Err(e) => return Err(e),
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
            // RM 570.144 `kflcnReset` (TU102, used on GA10x): pre-reset wait,
            // reset pulse, wait scrubbing, switch the core to falcon; then
            // `kflcnDisableCtxReq`: FBIF_CTL |= ALLOW_PHYS_NO_CTX, DMACTL = 0
            // (nouveau does the same without the core switch). C-96: the
            // board answered 0xbadf5620 right after the release; the waits
            // now poll through it.
            falcon::reset(bar, clock, Engine::Gsp)?;
            falcon::switch_to_falcon(bar, clock, Engine::Gsp)?;
            let ctl = gspa::NV_PGSP_FBIF_BASE + fbif::NV_PFALCON_FBIF_CTL;
            let v = bar.read(ctl)?;
            bar.write(ctl, fbif::NV_PFALCON_FBIF_CTL_ALLOW_PHYS_NO_CTX.set(v, 1))?;
            bar.write(gsp + fv4::NV_PFALCON_FALCON_DMACTL, 0)?;
        }
        SeqCmd::CoreStart => falcon::start_cpu(bar, Engine::Gsp)?,
        SeqCmd::CoreWaitForHalt => {
            let t0 = clock.now_us();
            poll_reg(bar, clock, LONG_WAIT_US, |b| {
                let v = b.read(gsp + fv4::NV_PFALCON_FALCON_CPUCTL)?;
                Ok(fv4::NV_PFALCON_FALCON_CPUCTL_HALTED.get(v) == 1)
            })?;
            waited(bar, clock, "CORE_WAIT_FOR_HALT (falconul GSP oprit)", t0);
        }
        SeqCmd::CoreResume => {
            // RM 570.144 `kgspExecuteSequencerCommand_GA102` / nouveau
            // CORE_RESUME: GSP reset into RISC-V first (C-96: this step was
            // missing), GSP mailboxes = LibOS args, start SEC2 (C-99: through
            // CPUCTL_ALIAS when ALIAS_EN is set — RM `kflcnStartCpu`; the
            // board dropped the plain CPUCTL write after Booter), wait the
            // stage-3 hand-off, SEC2 mailbox 0 must be 0, GSP OS = app
            // version, GSP RISC-V must be active.
            let sec2 = Engine::Sec2.base();
            falcon::gsp_reset_into_riscv(bar, clock)?;
            bar.write(gsp + fv4::NV_PFALCON_FALCON_MAILBOX0, ctx.libos_addr as u32)?;
            bar.write(
                gsp + fv4::NV_PFALCON_FALCON_MAILBOX1,
                (ctx.libos_addr >> 32) as u32,
            )?;
            // C-102: the state before SEC2 starts (reads only) — on 11 Nov
            // the hand-off flag answered after 2 µs, too fast for Booter:
            // these rows show whether it was already set.
            bar.journal().push(Event::Note(
                "I8 CORE_RESUME: starea inainte de pornirea SEC2",
            ));
            falcon::journal_resume_state(bar);
            falcon::start_cpu(bar, Engine::Sec2)?;
            let t0 = clock.now_us();
            poll_reg(bar, clock, LONG_WAIT_US, falcon::stage3_handoff_done)?;
            waited(bar, clock, "CORE_RESUME: SEC2 (Booter) gata", t0);
            bar.journal()
                .push(Event::Note("I8 CORE_RESUME: starea dupa predarea etapei 3"));
            falcon::journal_resume_state(bar);
            let mb0 = bar.read(sec2 + fv4::NV_PFALCON_FALCON_MAILBOX0)?;
            if mb0 != 0 {
                bar.journal().push(Event::Value {
                    what: "CORE_RESUME: SEC2 MAILBOX0 (Booter a dat eroare)",
                    value: mb0,
                });
                return Err(GpuError::HardwareFault { code: mb0 });
            }
            bar.write(gsp + fv4::NV_PFALCON_FALCON_OS, ctx.app_version)?;
            let t0 = clock.now_us();
            falcon::wait_riscv_active_for(bar, clock, LONG_WAIT_US)?;
            waited(bar, clock, "CORE_RESUME: GSP RISC-V activ", t0);
        }
    }
    Ok(())
}

/// C-99: journals how long an answer took (`asteptat …: raspuns dupa N us`).
fn waited<M: Mmio + ?Sized, C: Clock + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    clock: &mut C,
    what: &'static str,
    t0: u64,
) {
    let waited_us = clock.now_us().saturating_sub(t0);
    bar.journal().push(Event::Waited { what, waited_us });
}

/// What happened while waiting for `GSP_INIT_DONE`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct InitLog {
    pub sequencer_runs: u32,
    pub sequencer_commands: u32,
    /// Other events seen and skipped (function numbers), in order.
    pub other_events: Vec<u32>,
    /// os-279: `GSP_POST_NOCAT_RECORD` messages received (843 on the board).
    pub nocat_records: u32,
    /// The distinct complaints among them (`NocatRecord::kind`), in order of
    /// first appearance; at most [`NOCAT_KINDS_MAX`] are kept.
    pub nocat_kinds: Vec<(u8, u64, u32)>,
    /// Records of a kind not seen before that arrived after `nocat_kinds`
    /// was full (counted, not journaled).
    pub nocat_kinds_dropped: u32,
    /// os-279: `UCODE_LIBOS_PRINT` messages received (2 on the board).
    pub libos_prints: u32,
}

/// The first NOCAT records are journaled in full whatever they say…
pub const NOCAT_FULL_FIRST: u32 = 4;
/// …after that only a record of a kind not seen before is, up to this many
/// kinds; the rest are counted (843 records × 10 rows would bury the log).
pub const NOCAT_KINDS_MAX: usize = 16;
/// `UCODE_LIBOS_PRINT` messages journaled with their fields; the rest counted.
pub const LIBOS_PRINTS_FULL: u32 = 8;

/// os-279: the count of what GSP-RM sent besides the sequencer request and
/// `GSP_INIT_DONE`, written at every end of the wait (good or bad).
fn journal_skipped_summary<M: Mmio + ?Sized>(bar: &mut GuardedBar0<'_, M>, log: &InitLog) {
    bar.journal().push(Event::Value {
        what: "I8 mesaje GSP-RM sarite in asteptarea GSP_INIT_DONE: total",
        value: log.other_events.len() as u32,
    });
    if log.nocat_records != 0 {
        bar.journal().push(Event::Value {
            what: "I8 NOCAT (0x1020): inregistrari de diagnostic primite",
            value: log.nocat_records,
        });
        bar.journal().push(Event::Value {
            what: "I8 NOCAT: feluri distincte (recType, errorCode, subsystem)",
            value: log.nocat_kinds.len() as u32,
        });
        if log.nocat_kinds_dropped != 0 {
            bar.journal().push(Event::Value {
                what: "I8 NOCAT: inregistrari de feluri noi peste limita de 16 (nejurnalizate)",
                value: log.nocat_kinds_dropped,
            });
        }
    }
    if log.libos_prints != 0 {
        bar.journal().push(Event::Value {
            what: "I8 UCODE_LIBOS_PRINT (0x100c): jurnale de microcod primite",
            value: log.libos_prints,
        });
    }
}

/// os-279: one skipped message — named, and the two kinds the board sent
/// decoded (NOCAT records, LibOS prints); the rest get their number and name.
fn journal_skipped<M: Mmio + ?Sized>(
    bar: &mut GuardedBar0<'_, M>,
    log: &mut InitLog,
    m: &rpc::Message,
) {
    log.other_events.push(m.function);
    match m.function {
        event::GSP_POST_NOCAT_RECORD => {
            log.nocat_records += 1;
            let Some(r) = nocat::NocatRecord::decode(&m.payload) else {
                bar.journal().push(Event::Value {
                    what: "I8 NOCAT: mesaj mai scurt decat inregistrarea (octeti)",
                    value: m.payload.len() as u32,
                });
                return;
            };
            let kind = r.kind();
            let mut new_kind = !log.nocat_kinds.contains(&kind);
            if new_kind {
                if log.nocat_kinds.len() < NOCAT_KINDS_MAX {
                    log.nocat_kinds.push(kind);
                } else {
                    log.nocat_kinds_dropped += 1;
                    new_kind = false;
                }
            }
            if log.nocat_records <= NOCAT_FULL_FIRST || new_kind {
                r.journal(bar.journal(), log.nocat_records);
            }
        }
        event::UCODE_LIBOS_PRINT => {
            log.libos_prints += 1;
            if log.libos_prints <= LIBOS_PRINTS_FULL {
                match nocat::LibosPrint::decode(&m.payload) {
                    Some(l) => l.journal(bar.journal()),
                    None => bar.journal().push(Event::Value {
                        what: "I8 UCODE_LIBOS_PRINT: mesaj trunchiat (octeti)",
                        value: m.payload.len() as u32,
                    }),
                }
            }
        }
        _ => {
            // C-102: every other message is named in the journal (its
            // function number, name and rpc_result), so the board's log
            // shows what GSP-RM said on the way to INIT_DONE.
            bar.journal().push(Event::Value {
                what: "I8 mesaj GSP-RM sarit (functia)",
                value: m.function,
            });
            if let Some(n) = rpc::name(m.function) {
                bar.journal()
                    .push(Event::text("I8 mesaj GSP-RM sarit (numele)", n.as_bytes()));
            }
            if m.rpc_result != 0 {
                bar.journal().push(Event::Value {
                    what: "I8 mesaj GSP-RM sarit: rpc_result",
                    value: m.rpc_result,
                });
            }
        }
    }
}

/// Bound for the whole wait (DEDUS; nouveau waits seconds). C-99: raised
/// from 10 s to `LONG_WAIT_US` — the OS passes it unchanged.
pub const INIT_DONE_US: u64 = LONG_WAIT_US;

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
        let next = match q.recv_rpc(mem) {
            Ok(n) => n,
            Err(e) => {
                // C-102: an error from the queue itself is named — the board
                // showed a bare "eroare hardware 0x59" with no row saying
                // whether it was a queue pointer or a GSP-RM result.
                if let GpuError::HardwareFault { code } = e {
                    bar.journal().push(Event::Value {
                        what: "I8 coada de stare: pointer in afara cozii (0..62)",
                        value: code,
                    });
                }
                bar.journal()
                    .push(Event::Note("I8: eroarea vine din coada de stare"));
                journal_skipped_summary(bar, &log);
                falcon::journal_resume_state(bar);
                return Err(e);
            }
        };
        match next {
            Some(m) if m.function == event::GSP_INIT_DONE => {
                bar.journal().push(Event::Value {
                    what: "GSP_INIT_DONE: rpc_result (0 = bine; altfel cod NV_STATUS)",
                    value: m.rpc_result,
                });
                if m.rpc_result != 0 {
                    bar.journal().push(Event::Value {
                        what: "GSP_INIT_DONE: rpc_result_private",
                        value: m.rpc_result_private,
                    });
                    // os-279: the code's name from nvstatuscodes.h (0x59 =
                    // NV_ERR_OPERATING_SYSTEM), so the log reads itself.
                    if let Some(n) = nvidia_rm570::status_name(m.rpc_result) {
                        bar.journal().push(Event::text(
                            "GSP_INIT_DONE: rpc_result (nume NV_STATUS)",
                            n.as_bytes(),
                        ));
                    }
                    bar.journal()
                        .push(Event::Note("I8: eroarea vine din GSP_INIT_DONE"));
                    journal_skipped_summary(bar, &log);
                    falcon::journal_resume_state(bar);
                    return Err(GpuError::HardwareFault { code: m.rpc_result });
                }
                waited(bar, clock, "GSP_INIT_DONE (de la sonerie)", start);
                journal_skipped_summary(bar, &log);
                bar.journal().push(Event::Step {
                    name: "I8 GSP_INIT_DONE",
                    ok: true,
                });
                return Ok(log);
            }
            Some(m) if m.function == event::GSP_RUN_CPU_SEQUENCER => {
                let h = rpc::decode_sequencer(&m.payload)?;
                let cmds = parse(&h.commands)?;
                // C-96: the whole request goes to the journal before any
                // command runs (what GSP-RM asked, even if a command stops).
                bar.journal().push(Event::Note(
                    "I8 secventiator: cererea GSP-RM, comenzile urmeaza",
                ));
                for (i, c) in cmds.iter().enumerate() {
                    bar.journal().push(c.event(i as u32));
                }
                let mut save = h.reg_save;
                execute(bar, clock, &cmds, &mut save, ctx)?;
                log.sequencer_runs += 1;
                log.sequencer_commands += cmds.len() as u32;
            }
            Some(m) => journal_skipped(bar, &mut log, &m),
            None => {
                let waited_us = clock.now_us().saturating_sub(start);
                if waited_us >= timeout_us {
                    bar.journal()
                        .push(Event::Note("I8: GSP_INIT_DONE nu a venit (timp expirat)"));
                    journal_skipped_summary(bar, &log);
                    falcon::journal_resume_state(bar);
                    return Err(GpuError::Timeout { waited_us });
                }
                if clock.abort_requested() {
                    bar.journal()
                        .push(Event::Note("I8: oprit manual in asteptarea GSP_INIT_DONE"));
                    journal_skipped_summary(bar, &log);
                    falcon::journal_resume_state(bar);
                    return Err(GpuError::Aborted { waited_us });
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

    /// C-96 (T109): the board's I8 of 10 Nov 2026 — GSP-RM runs, the
    /// request is `WRITE MAILBOX0 0`, `CORE_RESET`, … — now gets past the
    /// reset (the first HWCFG2 read answers 0xbadf5620 in the model, as on
    /// the board); the whole request is journaled before any command runs;
    /// CORE_RESUME resets the GSP into RISC-V before the mailboxes (RM and
    /// nouveau order; the step was missing). SIMULARE.
    #[test]
    fn board_request_of_10_nov_runs_and_is_journaled() {
        let (mut g, mut j, mut c) = booted();
        assert!(g.riscv_active());
        let l = QueueLayout::standard();
        let mut mem = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let mut q = CpuQueues::init(&mut mem, l, &pages).unwrap();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        falcon::ring_doorbell(&mut b).unwrap();
        let mut peer = SimGsp::start(&mut mem, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&mem).unwrap();
        let seq = payload(&[W, 0x110040, 0, RESET, START, HALT, RESUME]);
        let mut step = 0;
        let n_before = b.journal().len();
        let log = wait_init_done(&mut q, &mut mem, &mut b, &mut c, CTX, INIT_DONE_US, |m| {
            step += 1;
            match step {
                1 => peer.post(m, event::GSP_RUN_CPU_SEQUENCER, 0, &seq),
                2 => peer.post(m, event::GSP_INIT_DONE, 0, &[]),
                _ => {}
            }
        })
        .unwrap();
        assert_eq!((log.sequencer_runs, log.sequencer_commands), (1, 5));
        // The request, in order, before the first command's write.
        let events: Vec<&Event> = b.journal().events().skip(n_before).collect();
        let first_cmd = events
            .iter()
            .position(|e| matches!(e, Event::SeqCmd { index: 0, .. }))
            .unwrap();
        let first_write = events
            .iter()
            .position(|e| {
                matches!(
                    e,
                    Event::MmioWrite {
                        offset: 0x110040,
                        ..
                    }
                )
            })
            .unwrap();
        assert!(first_cmd < first_write);
        let ops: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                Event::SeqCmd { op, .. } => Some(*op),
                _ => None,
            })
            .collect();
        assert_eq!(
            ops,
            [
                "WRITE",
                "CORE_RESET",
                "CORE_START",
                "CORE_WAIT_FOR_HALT",
                "CORE_RESUME"
            ]
        );
        assert_eq!(
            alloc::format!("{}", events[first_cmd]),
            "secv #0 WRITE 0x110040 0x0"
        );
        let _ = b;
        // CORE_RESET ended with the falcon core selected; CORE_RESUME put the
        // RISC-V core back (BCR 0x111) and GSP-RM is active again.
        let bcrs: Vec<u32> = g
            .writes
            .iter()
            .filter(|&&(o, _)| o == 0x111668)
            .map(|&(_, v)| v)
            .collect();
        assert!(bcrs.ends_with(&[0x0, 0x111]), "{bcrs:x?}");
        assert!(g.riscv_active());
        assert!(g.writes.contains(&(0x110080, 570_144)));
    }

    /// C-96 (T110): a command that stops is journaled again after the
    /// `ESEC` step, so the board's log names it; a poll on a register that
    /// stays unreadable reports the priv error value, not GSP's code.
    #[test]
    fn the_failing_command_is_named_in_the_journal() {
        let (mut g, mut j, mut c) = booted();
        g.fault = gpu_sim::fake_gsp::SimFault::PrivErrorStuck;
        let mut save = [0u32; 8];
        let cmds = parse(&[W, 0x110040, 0, RESET]).unwrap();
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            assert_eq!(
                execute(&mut b, &mut c, &cmds, &mut save, CTX),
                Err(GpuError::HardwareFault {
                    code: gpu_sim::fake_gsp::PRIV_ERROR_AFTER_RESET
                })
            );
        }
        let tail: Vec<alloc::string::String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let k = tail
            .iter()
            .position(|t| t.contains("comanda de mai jos s-a oprit"))
            .unwrap();
        assert_eq!(tail[k + 1], "secv #1 CORE_RESET");
    }

    /// C-99 (T111): the board's CORE_RESUME of 10 Nov 2026 (os-277,
    /// `gpu-jurnal-005.txt`) — after Booter, SEC2's CPUCTL reads `0x50`
    /// (HALTED | ALIAS_EN) and the plain `0x840100 <- 2` was dropped (read
    /// back `0x50`), so the stage-3 hand-off never came: timeout after 2 s.
    /// The model reproduces it; the driver now starts SEC2 through
    /// `CPUCTL_ALIAS` (0x840130), as RM `kflcnStartCpu` does. SIMULARE.
    #[test]
    fn core_resume_starts_sec2_through_the_alias_after_booter() {
        let (mut g, mut j, mut c) = booted();
        let before = g.writes.len();
        let cmds = parse(&[RESUME]).unwrap();
        let mut save = [0u32; 8];
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            // The board's reading after Booter.
            assert_eq!(b.read(0x840100).unwrap(), 0x50);
            execute(&mut b, &mut c, &cmds, &mut save, CTX).unwrap();
        }
        let sec2: Vec<(u32, u32)> = g.writes[before..]
            .iter()
            .copied()
            .filter(|&(o, _)| o == 0x840100 || o == 0x840130)
            .collect();
        assert_eq!(sec2, [(0x840130, 2)]);
        assert!(g.riscv_active());
        // The old driver's write, on the same model: the board's 2 s timeout.
        let (mut g, mut j, mut c) = booted();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        falcon::gsp_reset_into_riscv(&mut b, &mut c).unwrap();
        b.write(0x840100, 2).unwrap();
        assert_eq!(b.read(0x840100).unwrap(), 0x50);
        assert_eq!(
            poll_reg(&mut b, &mut c, falcon::HALT_US, falcon::stage3_handoff_done),
            Err(GpuError::Timeout {
                waited_us: 2_000_000
            })
        );
    }

    /// C-99 (T112, Adrian's idea): the waits after the doorbell are long
    /// (`LONG_WAIT_US`), the journal says how long each answer took, and a
    /// clock that asks to stop (the OS: Esc) ends the wait with `Aborted`
    /// and the time waited — not with a timeout. SIMULARE.
    #[test]
    fn long_waits_are_journaled_and_can_be_stopped_by_hand() {
        // 1. A good CORE_RESUME: two `asteptat` lines with the time taken.
        let (mut g, mut j, mut c) = booted();
        let cmds = parse(&[RESUME]).unwrap();
        let mut save = [0u32; 8];
        {
            let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
            execute(&mut b, &mut c, &cmds, &mut save, CTX).unwrap();
        }
        let lines: Vec<alloc::string::String> = j
            .events()
            .filter(|e| matches!(e, Event::Waited { .. }))
            .map(|e| alloc::format!("{e}"))
            .collect();
        assert_eq!(lines.len(), 2, "{lines:?}");
        assert!(lines[0].starts_with("asteptat CORE_RESUME: SEC2 (Booter) gata: raspuns dupa "));
        assert!(lines[1].starts_with("asteptat CORE_RESUME: GSP RISC-V activ: raspuns dupa "));
        assert!(lines.iter().all(|l| l.ends_with(" us")));
        // 2. A SEC2 that never answers, Esc at 5 s: stopped by hand (the wait
        //    itself began a little after the reset, so it counts slightly
        //    less), well before the 60 s bound; the command is named in the
        //    journal.
        let (mut g, mut j, mut c) = booted();
        g.fault = gpu_sim::fake_gsp::SimFault::NeverHalts;
        c.abort_at = Some(c.now + 5_000_000);
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let r = execute(&mut b, &mut c, &cmds, &mut save, CTX);
        assert!(
            matches!(r, Err(GpuError::Aborted { waited_us }) if (4_900_000..=5_000_000).contains(&waited_us)),
            "{r:?}"
        );
        assert!(alloc::format!("{}", r.unwrap_err()).starts_with("oprit manual dupa 49"));
        let _ = b;
        let tail: Vec<alloc::string::String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let k = tail
            .iter()
            .position(|t| t.contains("comanda de mai jos s-a oprit"))
            .unwrap();
        assert_eq!(tail[k + 1], "secv #0 CORE_RESUME");
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
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        assert!(
            rows.iter()
                .any(|r| r == "nota: I8: GSP_INIT_DONE nu a venit (timp expirat)")
        );
        assert!(
            rows.iter()
                .any(|r| r.starts_with("valoare GSP MAILBOX0 = "))
        );
    }

    fn queues_after_i7() -> (QueueLayout, Vec<u8>, CpuQueues) {
        let l = QueueLayout::standard();
        let mut mem = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let q = CpuQueues::init(&mut mem, l, &pages).unwrap();
        (l, mem, q)
    }

    /// C-102 (T113): the board of 11 Nov 2026 said only "eroare hardware
    /// 0x59". If GSP-RM's `GSP_INIT_DONE` carries `rpc_result` 0x59
    /// (NV_ERR_OPERATING_SYSTEM), the journal now names it and the
    /// registers around it — and the error is still an error (SIMULATION).
    #[test]
    fn init_done_with_an_error_code_is_named_in_the_journal() {
        let (mut g, mut j, mut c) = booted();
        let (l, mut mem, mut q) = queues_after_i7();
        let mut peer = SimGsp::start(&mut mem, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&mem).unwrap();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let mut step = 0;
        let r = wait_init_done(&mut q, &mut mem, &mut b, &mut c, CTX, INIT_DONE_US, |m| {
            step += 1;
            match step {
                1 => peer.post(m, event::OS_ERROR_LOG, 0, b"log"),
                2 => peer.post(m, event::GSP_INIT_DONE, 0x59, &[]),
                _ => {}
            }
        });
        assert!(matches!(r, Err(GpuError::HardwareFault { code: 0x59 })));
        let _ = b;
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let k = rows
            .iter()
            .position(|r| r == "valoare I8 mesaj GSP-RM sarit (functia) = 0x00001006")
            .expect("the skipped event is named");
        // os-279: the number's name from rpc_global_enums.h follows it
        assert_eq!(
            rows[k + 1],
            "text I8 mesaj GSP-RM sarit (numele) = \"OS_ERROR_LOG\""
        );
        assert_eq!(
            rows[k + 2],
            "valoare GSP_INIT_DONE: rpc_result (0 = bine; altfel cod NV_STATUS) = 0x00000059"
        );
        assert_eq!(
            rows[k + 3],
            "valoare GSP_INIT_DONE: rpc_result_private = 0x00000000"
        );
        assert_eq!(
            rows[k + 4],
            "text GSP_INIT_DONE: rpc_result (nume NV_STATUS) = \"NV_ERR_OPERATING_SYSTEM\""
        );
        assert_eq!(rows[k + 5], "nota: I8: eroarea vine din GSP_INIT_DONE");
        assert_eq!(
            rows[k + 6],
            "valoare I8 mesaje GSP-RM sarite in asteptarea GSP_INIT_DONE: total = 0x00000001"
        );
        assert!(rows[k + 7].starts_with("valoare PGC6 SCRATCH_14 (bit 26 = predarea etapei 3) = "));
        assert!(rows[k + 13].starts_with("valoare GSP MAILBOX1 = "));
        assert_eq!(rows.len(), k + 14);
    }

    /// os-279 (T116): what the board sent on 11 Nov 2026 — many
    /// `GSP_POST_NOCAT_RECORD` messages, two `UCODE_LIBOS_PRINT`, then
    /// `GSP_INIT_DONE` with 0x59. The journal decodes the records (the first
    /// four and each new kind), counts the rest, names the error code, and
    /// the error is still an error (SIMULATION).
    #[test]
    fn nocat_records_and_libos_prints_are_decoded_not_just_counted() {
        use crate::gsp::nocat::tests::record;
        let (mut g, mut j, mut c) = booted();
        let (l, mut mem, mut q) = queues_after_i7();
        let mut peer = SimGsp::start(&mut mem, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&mem).unwrap();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let mut step = 0;
        let r = wait_init_done(&mut q, &mut mem, &mut b, &mut c, CTX, INIT_DONE_US, |m| {
            step += 1;
            match step {
                // 30 records of one kind (ENGINE, NV_ERR_INVALID_STATE 0x40,
                // subsystem 5): the first four in full, then counted
                1..=30 => peer.post(
                    m,
                    event::GSP_POST_NOCAT_RECORD,
                    0,
                    &record(2, 0x40, 5, b"GSP-RM"),
                ),
                // a new kind (RC, NV_ERR_TIMEOUT 0x65): in full, once
                31 => peer.post(
                    m,
                    event::GSP_POST_NOCAT_RECORD,
                    0,
                    &record(4, 0x65, 9, b"GSP-RM"),
                ),
                32 => peer.post(
                    m,
                    event::GSP_POST_NOCAT_RECORD,
                    0,
                    &record(4, 0x65, 9, b"GSP-RM"),
                ),
                // a record shorter than the struct: noted, not decoded
                33 => peer.post(m, event::GSP_POST_NOCAT_RECORD, 0, &[0u8; 100]),
                34 | 35 => {
                    let mut p = alloc::vec![0u8; 8 + 64];
                    p[0..4].copy_from_slice(&0x1234u32.to_le_bytes());
                    p[4..8].copy_from_slice(&64u32.to_le_bytes());
                    peer.post(m, event::UCODE_LIBOS_PRINT, 0, &p)
                }
                36 => peer.post(m, event::GSP_INIT_DONE, 0x59, &[]),
                _ => {}
            }
        });
        assert!(matches!(r, Err(GpuError::HardwareFault { code: 0x59 })));
        let _ = b;
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let count = |needle: &str| rows.iter().filter(|r| r.contains(needle)).count();
        // 4 (first) + 1 (new kind) = 5 records in full; the other 27 counted
        assert_eq!(count("valoare NOCAT: inregistrarea nr. = "), 5);
        assert!(rows.contains(&String::from(
            "valoare NOCAT: inregistrarea nr. = 0x00000004"
        )));
        assert!(rows.contains(&String::from(
            "valoare NOCAT: inregistrarea nr. = 0x0000001f"
        )));
        assert!(!rows.contains(&String::from(
            "valoare NOCAT: inregistrarea nr. = 0x00000005"
        )));
        assert_eq!(
            count("text NOCAT: errorCode (nume NV_STATUS) = \"NV_ERR_INVALID_STATE\""),
            4
        );
        assert_eq!(
            count("text NOCAT: errorCode (nume NV_STATUS) = \"NV_ERR_TIMEOUT\""),
            1
        );
        assert_eq!(count("text NOCAT: recType = \"RC\""), 1);
        assert_eq!(count("text NOCAT: source = \"GSP-RM\""), 5);
        assert_eq!(
            count("I8 NOCAT: mesaj mai scurt decat inregistrarea (octeti) = 0x00000064"),
            1
        );
        assert_eq!(count("UCODE_LIBOS_PRINT: ucodeEngDesc"), 2);
        assert_eq!(
            count(
                "UCODE_LIBOS_PRINT: libosPrintBufSize (jurnal LibOS binar, nedecodat) = 0x00000040"
            ),
            2
        );
        // no bare "sarit (functia)" rows for the decoded kinds
        assert_eq!(count("mesaj GSP-RM sarit (functia)"), 0);
        // the summary after the error, before the registers
        let e = rows
            .iter()
            .position(|r| r == "nota: I8: eroarea vine din GSP_INIT_DONE")
            .unwrap();
        assert_eq!(
            rows[e - 1],
            "text GSP_INIT_DONE: rpc_result (nume NV_STATUS) = \"NV_ERR_OPERATING_SYSTEM\""
        );
        assert_eq!(
            rows[e + 1],
            "valoare I8 mesaje GSP-RM sarite in asteptarea GSP_INIT_DONE: total = 0x00000023"
        );
        assert_eq!(
            rows[e + 2],
            "valoare I8 NOCAT (0x1020): inregistrari de diagnostic primite = 0x00000021"
        );
        assert_eq!(
            rows[e + 3],
            "valoare I8 NOCAT: feluri distincte (recType, errorCode, subsystem) = 0x00000002"
        );
        assert_eq!(
            rows[e + 4],
            "valoare I8 UCODE_LIBOS_PRINT (0x100c): jurnale de microcod primite = 0x00000002"
        );
        assert!(rows[e + 5].starts_with("valoare PGC6 SCRATCH_14"));
        // the first NOCAT record comes right after the doorbell-less start:
        // nothing was written to the board for any of this
        assert!(!g.writes.iter().any(|&(o, _)| o == 0x110c00));

        // 40 records, each of a new kind: 16 kinds kept (the first 4 are
        // among them), 24 counted as dropped kinds, INIT_DONE fine → Ok
        let (mut g, mut j, mut c) = booted();
        let (l, mut mem, mut q) = queues_after_i7();
        let mut peer = SimGsp::start(&mut mem, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&mem).unwrap();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let mut step = 0u32;
        let r = wait_init_done(&mut q, &mut mem, &mut b, &mut c, CTX, INIT_DONE_US, |m| {
            step += 1;
            match step {
                1..=40 => peer.post(
                    m,
                    event::GSP_POST_NOCAT_RECORD,
                    0,
                    &record(2, 0x100 + u64::from(step), 1, b"GSP-RM"),
                ),
                41 => peer.post(m, event::GSP_INIT_DONE, 0, &[]),
                _ => {}
            }
        })
        .unwrap();
        assert_eq!(r.nocat_records, 40);
        assert_eq!(r.nocat_kinds.len(), NOCAT_KINDS_MAX);
        assert_eq!(r.nocat_kinds_dropped, 24);
        assert_eq!(r.other_events.len(), 40);
        let _ = b;
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let count = |needle: &str| rows.iter().filter(|r| r.contains(needle)).count();
        assert_eq!(
            count("valoare NOCAT: inregistrarea nr. = "),
            NOCAT_KINDS_MAX
        );
        assert!(rows.contains(&String::from(
            "valoare I8 NOCAT: inregistrari de feluri noi peste limita de 16 (nejurnalizate) = 0x00000018"
        )));
        assert!(rows.contains(&String::from("pas I8 GSP_INIT_DONE: ok")));
    }

    /// os-280 preparation (T120): the journal the next run will write — NOCAT
    /// records (asserts and an engine error) before and after the sequencer
    /// request, LibOS prints, then `GSP_INIT_DONE` with 0x59 — printed as
    /// the OS prints it on the stick, read back by [`nocat_log::digest`]:
    /// the digest says exactly what the driver journaled, nothing more
    /// (SIMULATION; the board's records will say something else).
    #[test]
    fn the_stick_journal_rows_are_read_back_into_a_digest() {
        use crate::gsp::nocat::tests::{assert_record, record};
        use crate::gsp::nocat_log;
        let (mut g, mut j, mut c) = booted();
        let l = QueueLayout::standard();
        let mut mem = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let mut q = CpuQueues::init(&mut mem, l, &pages).unwrap();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        falcon::ring_doorbell(&mut b).unwrap();
        let mut peer = SimGsp::start(&mut mem, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&mem).unwrap();
        let seq = payload(&[W, 0x110040, 0, RESET, START, HALT, RESUME]);
        let a = assert_record(0x0001_2340, 321, &[0x10, 0x20], 1);
        let b2 = assert_record(0x0005_6780, 99, &[0x30], 3);
        let mut step = 0;
        let r = wait_init_done(&mut q, &mut mem, &mut b, &mut c, CTX, INIT_DONE_US, |m| {
            step += 1;
            match step {
                // six asserts at one place: the first four in full, two counted
                1..=6 => peer.post(m, event::GSP_POST_NOCAT_RECORD, 0, &a),
                // a second assert place: new kind, in full
                7 => peer.post(m, event::GSP_POST_NOCAT_RECORD, 0, &b2),
                8 => peer.post(m, event::GSP_RUN_CPU_SEQUENCER, 0, &seq),
                // after the sequencer: an ENGINE record (new kind) twice
                9 | 10 => peer.post(
                    m,
                    event::GSP_POST_NOCAT_RECORD,
                    0,
                    &record(2, 0x40, 5, b"GSP-RM"),
                ),
                11 => {
                    let mut p = alloc::vec![0u8; 8 + 16];
                    p[0..4].copy_from_slice(&0x1234u32.to_le_bytes());
                    p[4..8].copy_from_slice(&16u32.to_le_bytes());
                    // Session 87: the content words travel to the stick too.
                    p[8..12].copy_from_slice(&0xdead_beefu32.to_le_bytes());
                    p[20..24].copy_from_slice(&0x0102_0304u32.to_le_bytes());
                    peer.post(m, event::UCODE_LIBOS_PRINT, 0, &p)
                }
                12 => peer.post(m, event::GSP_INIT_DONE, 0x59, &[]),
                _ => {}
            }
        });
        assert!(matches!(r, Err(GpuError::HardwareFault { code: 0x59 })));
        let _ = b;
        // the stick file: one row per event, with the OS's log prefix
        let mut text = String::from("# jurnal gpu P6-40, cyber-gpu test, comanda: gpu lant\n");
        for (i, e) in j.events().enumerate() {
            text.push_str(&alloc::format!(
                "[   {}.{:06}] gpu: {e}\n",
                60 + i / 50,
                i * 20_000
            ));
        }
        let d = nocat_log::digest(&text);
        assert!(!d.is_empty());
        assert!(d.sequencer_seen);
        assert_eq!(d.summary_received, Some(9));
        assert_eq!(d.summary_kinds, Some(3));
        assert_eq!(d.summary_kinds_dropped, None);
        assert_eq!(d.summary_libos, Some(1));
        assert_eq!(d.summary_skipped_total, Some(10));
        assert_eq!(d.init_done_result, Some(0x59));
        assert_eq!(d.init_done_name.as_deref(), Some("NV_ERR_OPERATING_SYSTEM"));
        assert_eq!(d.error_origin.as_deref(), Some("GSP_INIT_DONE"));
        assert_eq!(d.libos_sizes, [16]);
        assert_eq!(d.libos_words, [alloc::vec![0xdead_beef, 0, 0, 0x0102_0304]]);
        assert!(alloc::format!("{d}").contains(
            "continut #1 (cuvinte LE, asa cum au venit): deadbeef 00000000 00000000 01020304"
        ));
        assert_eq!(
            (d.named_only_nocat, d.named_only_libos, d.short_records),
            (0, 0, 0)
        );
        // journaled in full: records 1–4 (first four), 7 (new kind), 8 (new kind)
        let idx: Vec<u32> = d.records.iter().map(|r| r.index).collect();
        assert_eq!(idx, [1, 2, 3, 4, 7, 8]);
        assert_eq!(d.before_after_sequencer(), (5, 1));
        let r1 = &d.records[0];
        assert_eq!(r1.rec_type.as_deref(), Some("ASSERT"));
        assert_eq!(r1.source.as_deref(), Some("ASSERT"));
        assert_eq!(
            (r1.assert_addr, r1.assert_line),
            (Some(0x0001_2340), Some(321))
        );
        assert_eq!(r1.stack, [0x10, 0x20]);
        assert_eq!(r1.count, None);
        assert_eq!(r1.error_code, None);
        assert_eq!(r1.faulting_engine.as_deref(), Some("GSP"));
        let r7 = &d.records[4];
        assert_eq!(
            (r7.assert_addr, r7.assert_line, r7.count),
            (Some(0x0005_6780), Some(99), Some(3))
        );
        let r8 = &d.records[5];
        assert_eq!(r8.rec_type.as_deref(), Some("ENGINE"));
        assert_eq!(r8.error_code, Some(0x40));
        assert_eq!(r8.error_name.as_deref(), Some("NV_ERR_INVALID_STATE"));
        assert_eq!(r8.subsystem, Some(5));
        assert!(r8.after_sequencer);
        // kinds: the first assert place covers 4 journaled records
        let kinds = d.kinds();
        assert_eq!(kinds.len(), 3);
        assert_eq!((kinds[0].0.index, kinds[0].1), (1, 4));
        assert_eq!(kinds[1].1, 1);
        let s = alloc::format!("{d}");
        assert!(s.contains("NOCAT: 9 inregistrari de diagnostic primite de la GSP-RM, 3 feluri distincte; 1 jurnale de microcod"));
        assert!(s.contains("jurnalizate in intregime: 6 (inainte de secventiator: 5, dupa: 1)"));
        assert!(s.contains("x4   ASSERT   adresa 0x00012340 linia 321 source \"ASSERT\" subsystem 0x0 motor \"GSP\" stiva [0x10, 0x20] (prima: nr. 1)"));
        assert!(s.contains("count 3"));
        assert!(s.contains(
            "ENGINE   errorCode 0x40 NV_ERR_INVALID_STATE source \"GSP-RM\" subsystem 0x5"
        ));
        assert!(s.contains("GSP_INIT_DONE: rpc_result 0x59 (NV_ERR_OPERATING_SYSTEM)\nI8: eroarea vine din GSP_INIT_DONE\n"));
    }

    /// C-102 (T114): the other way 0x59 can appear — the status queue's
    /// write pointer read as 89, outside 0..62. The error is the same
    /// `HardwareFault { 0x59 }`; the journal now says it came from the queue.
    #[test]
    fn a_queue_pointer_out_of_range_is_named_in_the_journal() {
        let (mut g, mut j, mut c) = booted();
        let (l, mut mem, mut q) = queues_after_i7();
        let _peer = SimGsp::start(&mut mem, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&mem).unwrap();
        // GSP's tx header of the status queue: write pointer = 89.
        let wp = l.stat_queue_offset + crate::gsp::msgq::qoff::WRITE_PTR;
        mem[wp..wp + 4].copy_from_slice(&89u32.to_le_bytes());
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let r = wait_init_done(&mut q, &mut mem, &mut b, &mut c, CTX, INIT_DONE_US, |_| {});
        assert!(matches!(r, Err(GpuError::HardwareFault { code: 0x59 })));
        let _ = b;
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let k = rows
            .iter()
            .position(|r| {
                r == "valoare I8 coada de stare: pointer in afara cozii (0..62) = 0x00000059"
            })
            .expect("the queue pointer is named");
        assert_eq!(rows[k + 1], "nota: I8: eroarea vine din coada de stare");
        assert_eq!(
            rows[k + 2],
            "valoare I8 mesaje GSP-RM sarite in asteptarea GSP_INIT_DONE: total = 0x00000000"
        );
        assert!(rows[k + 3].starts_with("valoare PGC6 SCRATCH_14"));
    }

    /// C-102 (T115): CORE_RESUME journals the state before SEC2 starts and
    /// after the stage-3 hand-off (reads only), in this order.
    #[test]
    fn core_resume_journals_the_state_before_and_after_sec2() {
        let (mut g, mut j, mut c) = booted();
        let mut b = GuardedBar0::new(&mut g, &mut j, None).unwrap();
        let mut save = [0u32; 8];
        execute(&mut b, &mut c, &[SeqCmd::CoreResume], &mut save, CTX).unwrap();
        let _ = b;
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        let before = rows
            .iter()
            .position(|r| r == "nota: I8 CORE_RESUME: starea inainte de pornirea SEC2")
            .unwrap();
        let after = rows
            .iter()
            .position(|r| r == "nota: I8 CORE_RESUME: starea dupa predarea etapei 3")
            .unwrap();
        assert!(before < after);
        assert!(rows[before + 1].starts_with("valoare PGC6 SCRATCH_14"));
        assert!(rows[before + 2].starts_with("valoare SEC2 CPUCTL = "));
        let alias = rows
            .iter()
            .position(|r| r.starts_with("bar0 w 0x840130"))
            .expect("SEC2 started through CPUCTL_ALIAS");
        assert!(before < alias && alias < after);
        // the reads are reads: no new write besides the sequence's own
        assert!(!g.writes.iter().any(|&(o, _)| o == 0x1180f8));
    }
}
