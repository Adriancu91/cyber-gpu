//! I9c — the driver's working client and its VA space, on the driver's own
//! page tables. Built and tested **only in SIMULARE** (against
//! `gpu_sim::fake_rm` behind the real queue code); nothing here has run on
//! the board, and nothing here writes a register.
//!
//! After the golden image (I8e), tinygrad (`ops_nv.py` `PCIIface` +
//! `NVDevice.__init__`, `ip.py` `rpc_rm_alloc` / `rpc_set_page_directory`,
//! MIT, commit 9cf7829a6; read as the specification and re-written) opens a
//! second client for the real work:
//! 1. client (`NV01_ROOT`, handle `0xc1000000`), device, subdevice;
//! 2. a VA space that is *externally owned* (the driver makes the page
//!    tables) with page faulting on, `vaBase = 0x1000`;
//! 3. the `SET_PAGE_DIRECTORY` RPC: GSP-RM is told the address of the root
//!    table (PD3) — the same root the golden image's reserved range hangs
//!    from (one set of page tables for everything, as tinygrad).
//!
//! RM 570.144 (`dma.c` `deviceCtrlCmdDmaSetPageDirectory`, `gpu_vaspace.c`
//! `gvaspaceExternalRootDirCommit`, MIT) is what says why: the call is
//! accepted only on an externally-owned (or shared-management) VA space, the
//! entries must cover the VA space's limit, and an externally-owned VA space
//! can get its root only once.
//!
//! I9d (also SIMULARE only): the working client's channel group (TSG, class
//! `KEPLER_CHANNEL_GROUP_A`, engine GR0) and its context share
//! (`FERMI_CONTEXT_SHARE_A`, a subcontext on the same VA space, `ASYNC` as
//! tinygrad). Rules read in RM 570.144 `kernel_channel_group_api.c` and
//! `kernel_ctxshare.c` (MIT): a TSG's parent is a device; with per-runlist
//! channel RAM (Ampere) the engine must be valid, and without MIG only GR0
//! among the GR engines; a context share's parent must be a TSG; `ASYNC`
//! takes a subcontext from the top of the group downwards.
//!
//! I9e (also SIMULARE only): the working GPFIFO channel in that group, its
//! compute object, the GR context promotion for it, the doorbell token and
//! the group's schedule — tinygrad `ops_nv.py` (`NVDevice.fifos`,
//! `_new_gpu_fifo`) and `ip.py` (`rpc_rm_alloc`, `promote_ctx`), MIT,
//! commit 9cf7829a6, read as the specification and re-written; the rules in
//! RM 570.144 `kernel_channel.c` (`kchannelConstruct`,
//! `kchannelCtrlCmdGpfifoGetWorkSubmitToken`) and `kernel_fifo_ga100.c`
//! (MIT). Order: channel (parent = TSG, with the context share) → compute
//! object → PROMOTE_CTX for MAIN/PM/PATCH, first the physical half, then the
//! virtual half, on new per-channel buffers → work submit token (+ runlist)
//! → schedule the TSG. Ringing the doorbell is a register write: not here.
//!
//! I10c (also SIMULARE only): GR0's shape (GPCs, TPCs per GPC, SMs per TPC,
//! warps per SM) and SM version, asked from GSP-RM on the working client's
//! subdevice with `NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO`, as tinygrad
//! does on its GSP path (`ops_nv.py` `_query_gpu_info`, MIT, commit
//! f5b17be). The local memory size (`work_queue::local_memory_size`) takes
//! this shape; a board that is not `sm_86` is refused before any launch.

use gpu_core::error::{GpuError, GpuResult};

use alloc::vec::Vec;

use super::golden::{GoldenImage, GoldenMemory, Mapping, PROMOTE_PHYS_ATTR, Rm, RmTransport};
use super::rmapi::{self, GrInfo, PromoteEntry, WorkChannelSetup, class, cmd};
use super::rpc::function;
use super::work_queue::ChannelMemory;
use crate::mmu::LEVELS;

/// The client handle tinygrad uses for its work (`PCIIface.root`). DEDUS.
pub const USER_CLIENT: u32 = 0xc100_0000;

/// What the working client is made of (kept for the channels of I9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UserSpace {
    pub client: u32,
    pub device: u32,
    pub subdevice: u32,
    pub vaspace: u32,
    /// The root table (PD3) GSP-RM was told about, and its entry count.
    pub root: u64,
    pub root_entries: u32,
    /// The first handle still free after this client.
    pub next_handle: u32,
}

/// Opens the working client. `first_handle` comes from
/// [`super::golden::GoldenImage::next_handle`]; `root` is the driver's PD3
/// (`mmu::GpuMemory::root`). Stops at the first refusal from GSP-RM (the
/// error carries RM's status); never reports a VA space as usable unless
/// GSP-RM accepted `SET_PAGE_DIRECTORY`.
pub fn open_user_space<T: RmTransport + ?Sized>(
    t: &mut T,
    first_handle: u32,
    root: u64,
) -> GpuResult<UserSpace> {
    let entries = LEVELS[0].entries() as u32;
    // Checked before the first message, so a bad root sends nothing.
    rmapi::check_root(root)?;
    let mut rm = Rm {
        t,
        client: USER_CLIENT,
        next: first_handle,
    };
    rm.alloc(0, class::ROOT, &rmapi::root_params())?;
    let device = rm.alloc(
        USER_CLIENT,
        class::DEVICE,
        &rmapi::device_params(USER_CLIENT),
    )?;
    let subdevice = rm.alloc(device, class::SUBDEVICE, &rmapi::subdevice_params())?;
    let vaspace = rm.alloc(device, class::VASPACE, &rmapi::user_vaspace_params())?;
    let msg = rmapi::set_page_directory_rpc(USER_CLIENT, device, vaspace, root, entries)?;
    // The RPC result is GSP-RM's status; a non-zero one is an error
    // (`RmTransport::call`). Its reply payload carries nothing we use.
    rm.t.call(function::SET_PAGE_DIRECTORY, &msg)?;
    let next_handle = rm.next;
    if next_handle <= vaspace {
        return Err(GpuError::InvalidArgument("RM: handle-urile s-au terminat"));
    }
    Ok(UserSpace {
        client: USER_CLIENT,
        device,
        subdevice,
        vaspace,
        root,
        root_entries: entries,
        next_handle,
    })
}

/// I10c: GR0's shape and SM version, from GSP-RM only (never assumed).
/// One control on `u.subdevice`; GSP-RM's refusal comes back as its status.
/// C-87 (recovery): takes the channel group off the runlist (`NVA06C_CTRL_CMD_GPFIFO_SCHEDULE`, `bEnable = 0`) — the
/// same RM control that put it there, with the other value; no register is written by us. After a hang the
/// channel's work is no longer scheduled; GSP-RM must answer for this to be known.
pub fn stop_compute_group<T: RmTransport + ?Sized>(
    t: &mut T,
    u: &UserSpace,
    cg: &ComputeGroup,
) -> GpuResult<()> {
    let mut rm = Rm {
        t,
        client: u.client,
        next: u.next_handle,
    };
    rm.control(
        cg.channel_group,
        cmd::TSG_SCHEDULE,
        &rmapi::schedule_params(false),
    )?;
    Ok(())
}

pub fn query_gr_info<T: RmTransport + ?Sized>(t: &mut T, u: &UserSpace) -> GpuResult<GrInfo> {
    let mut rm = Rm {
        t,
        client: u.client,
        next: u.next_handle,
    };
    let got = rm.control(u.subdevice, cmd::KGR_GET_INFO, &rmapi::gr_info_request())?;
    rmapi::parse_gr_info(&got)
}

/// The working client's channel group and its context share (I9d).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeGroup {
    pub channel_group: u32,
    pub ctxshare: u32,
    /// The subcontext (VEID) GSP-RM gave the context share.
    pub subctx_id: u32,
    /// The first handle still free after the group.
    pub next_handle: u32,
}

/// Creates the channel group (TSG) for GR0 on the working client's VA space
/// and one context share in it. `u` must come from [`open_user_space`] (so
/// GSP-RM already accepted our page tables for that VA space). Stops at the
/// first refusal; the subcontext id is taken only from GSP-RM's answer.
pub fn open_compute_group<T: RmTransport + ?Sized>(
    t: &mut T,
    u: &UserSpace,
) -> GpuResult<ComputeGroup> {
    rmapi::check_root(u.root)?;
    let mut rm = Rm {
        t,
        client: u.client,
        next: u.next_handle,
    };
    let channel_group = rm.alloc(
        u.device,
        class::CHANNEL_GROUP,
        &rmapi::channel_group_params(u.vaspace)?,
    )?;
    let sent = rmapi::ctxshare_params(u.vaspace, rmapi::subcontext::ASYNC)?;
    let (ctxshare, got) = rm.alloc_reply(channel_group, class::CONTEXT_SHARE, &sent)?;
    let subctx_id = rmapi::parse_ctxshare_reply(&sent, &got)?;
    let next_handle = rm.next;
    if next_handle <= ctxshare {
        return Err(GpuError::InvalidArgument("RM: handle-urile s-au terminat"));
    }
    Ok(ComputeGroup {
        channel_group,
        ctxshare,
        subctx_id,
        next_handle,
    })
}

/// GR context buffers promoted per working channel (tinygrad: `k in [0, 1, 2]`
/// = MAIN, PM, PATCH). DEDUS: proven only by tinygrad running them.
pub const WORK_CTX_BUFFERS: [u16; 3] = [0, 1, 2];

/// The working compute channel (I9e).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorkChannel {
    pub channel: u32,
    pub compute: u32,
    /// Ring + USERD buffer (ring at its start, USERD at `userd_offset`).
    pub ring: Mapping,
    pub entries: u32,
    pub userd_offset: u64,
    pub inst_pa: u64,
    pub mthd_pa: u64,
    pub notifier_pa: u64,
    /// The per-channel GR buffers promoted (id, mapping).
    pub ctx_bufs: Vec<(u16, Mapping)>,
    /// Channel id from GSP-RM and the value the doorbell would get.
    pub chid: u32,
    pub doorbell_token: u32,
    /// The first handle still free after the channel.
    pub next_handle: u32,
}

/// Creates the working compute channel in the group `cg` of the working
/// client `u`, using the GR buffer sizes and the runlists the golden image
/// `g` learned. Stops at the first refusal; the doorbell token is built
/// only from GSP-RM's answer and the runlist GSP-RM reported for GR0.
/// Nothing is written to the board (the doorbell is not rung).
pub fn open_work_channel<T: RmTransport + ?Sized, M: GoldenMemory + ?Sized>(
    t: &mut T,
    mem: &mut M,
    u: &UserSpace,
    cg: &ComputeGroup,
    g: &GoldenImage,
) -> GpuResult<WorkChannel> {
    open_channel(t, mem, u, cg, g, |_, _| Ok(()))
}

/// I10a: like [`open_work_channel`], but USERD is zeroed **before** the
/// channel is allocated, so `GPGet = GPPut = 0` when GSP-RM schedules it.
/// RM 570.144 clears a client's USERD itself only on vGPU
/// (`kernel_channel.c`, `kfifoSetupUserD`); on bare metal it is the
/// driver's job. Use this one for a channel that will get work
/// ([`super::work_queue::WorkQueue`]).
pub fn open_work_channel_cleared<
    T: RmTransport + ?Sized,
    M: GoldenMemory + ChannelMemory + ?Sized,
>(
    t: &mut T,
    mem: &mut M,
    u: &UserSpace,
    cg: &ComputeGroup,
    g: &GoldenImage,
) -> GpuResult<WorkChannel> {
    open_channel(t, mem, u, cg, g, |m, userd_pa| {
        for off in (0..USERD_SIZE).step_by(4) {
            m.write_u32(userd_pa + off, 0)?;
        }
        Ok(())
    })
}

/// Bytes of USERD (`AmpereAControlGPFifo`, generated).
pub const USERD_SIZE: u64 =
    core::mem::size_of::<nvidia_rm570::rmapi::AmpereAControlGPFifo>() as u64;

fn open_channel<T: RmTransport + ?Sized, M: GoldenMemory + ?Sized>(
    t: &mut T,
    mem: &mut M,
    u: &UserSpace,
    cg: &ComputeGroup,
    g: &GoldenImage,
    clear_userd: impl FnOnce(&mut M, u64) -> GpuResult<()>,
) -> GpuResult<WorkChannel> {
    rmapi::check_root(u.root)?;
    let runlist = g
        .runlists
        .iter()
        .find(|&&(e, _)| e == rmapi::ENGINE_GR0)
        .map(|&(_, r)| r)
        .ok_or(GpuError::InvalidArgument(
            "GSP-RM nu a raportat runlist-ul lui GR0",
        ))?;
    let plan: Vec<_> = WORK_CTX_BUFFERS
        .iter()
        .map(|id| {
            g.gr_bufs
                .iter()
                .find(|(b, _)| b.id == *id)
                .map(|(b, _)| *b)
                .ok_or(GpuError::InvalidArgument(
                    "imaginea golden nu are bufferul de context cerut",
                ))
        })
        .collect::<GpuResult<_>>()?;
    let entries = rmapi::work_channel::ENTRIES;
    let ring = mem.alloc_mapped(WorkChannelSetup::buffer_size(entries), 0x1000)?;
    let inst_pa = mem.alloc_phys(rmapi::golden_channel::INSTANCE_SIZE, 0x1000)?;
    let mthd_pa = mem.alloc_phys(rmapi::golden_channel::METHOD_BUFFER_SIZE, 0x1000)?;
    let notifier_pa = mem.alloc_phys(0x1000, 0x1000)?;
    let setup = WorkChannelSetup {
        ring_va: ring.va,
        ring_pa: ring.pa,
        entries,
        ctxshare: cg.ctxshare,
        inst_pa,
        mthd_pa,
        notifier_pa,
    };
    // Built before the first message, so a bad setup sends nothing.
    let params = rmapi::work_channel_params(&setup)?;
    clear_userd(mem, ring.pa + setup.userd_offset())?;
    let mut rm = Rm {
        t,
        client: u.client,
        next: cg.next_handle,
    };
    let channel = rm.alloc(cg.channel_group, class::CHANNEL_GPFIFO, &params)?;
    let compute = rm.alloc(channel, class::COMPUTE, &[])?;

    // The GR context of this channel: physical half, then virtual half.
    let mut ctx_bufs = Vec::with_capacity(plan.len());
    for b in &plan {
        ctx_bufs.push((b.id, mem.alloc_mapped(b.size, b.align)?));
    }
    let phys: Vec<_> = plan
        .iter()
        .zip(&ctx_bufs)
        .map(|(b, (_, m))| PromoteEntry {
            buffer_id: b.id,
            gpu_va: 0,
            gpu_pa: m.pa,
            size: b.size,
            phys_attr: PROMOTE_PHYS_ATTR,
            initialize: true,
            nonmapped: true,
        })
        .collect();
    let virt: Vec<_> = plan
        .iter()
        .zip(&ctx_bufs)
        .map(|(b, (_, m))| PromoteEntry {
            buffer_id: b.id,
            gpu_va: m.va,
            gpu_pa: 0,
            size: 0,
            phys_attr: 0,
            initialize: false,
            nonmapped: false,
        })
        .collect();
    for half in [&phys, &virt] {
        rm.control(
            u.subdevice,
            cmd::PROMOTE_CTX,
            &rmapi::promote_params(rmapi::ENGINE_GR0, u.client, channel, half)?,
        )?;
    }

    let got = rm.control(
        channel,
        cmd::WORK_SUBMIT_TOKEN,
        &rmapi::work_submit_token_request(),
    )?;
    let chid = rmapi::parse_work_submit_token(&got)?;
    let doorbell_token = rmapi::doorbell_token(chid, runlist)?;
    rm.control(
        cg.channel_group,
        cmd::TSG_SCHEDULE,
        &rmapi::schedule_params(true),
    )?;
    let next_handle = rm.next;
    Ok(WorkChannel {
        channel,
        compute,
        ring,
        entries,
        userd_offset: setup.userd_offset(),
        inst_pa,
        mthd_pa,
        notifier_pa,
        ctx_bufs,
        chid,
        doorbell_token,
        next_handle,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gsp::golden::{GoldenMemory, QueueTransport, RM_REPLY_US, init_golden_image};
    use crate::gsp::msgq::{CpuQueues, QueueLayout, SharedMem};
    use crate::gsp::sim_peer::SimGsp;
    use crate::mmu::{GpuMemory, VramWrite};
    use alloc::vec::Vec;
    use gpu_sim::clock::SimClock;
    use gpu_sim::fake_rm::{FakeRm, class as c, cmd as c_cmd, status};
    use gpu_sim::vram::{SimVram, translate};

    struct Sim(SimVram);
    impl VramWrite for Sim {
        fn write_u64(&mut self, pa: u64, v: u64) -> GpuResult<()> {
            self.0.write_u64(pa, v).map_err(GpuError::InvalidArgument)
        }
    }

    /// Direct transport to the model (no queues).
    struct Direct(FakeRm);
    impl RmTransport for Direct {
        fn call(&mut self, f: u32, p: &[u8]) -> GpuResult<Vec<u8>> {
            match self.0.handle(f, p) {
                Some((0, r)) => Ok(r),
                Some((code, _)) => Err(GpuError::HardwareFault { code }),
                None => Err(GpuError::NotImplemented("functie necunoscuta modelului")),
            }
        }
    }

    /// SIMULARE end to end: golden image and working client through the real
    /// queues, both on the same page tables (`mmu::GpuMemory` in a simulated
    /// VRAM); the model records the root it was given, and the independent
    /// walker (`gpu_sim::vram::translate`) resolves a golden buffer from it.
    #[test]
    fn working_client_gets_the_drivers_page_tables() {
        let l = QueueLayout::standard();
        let mut shm = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let mut q = CpuQueues::init(&mut shm, l, &pages).unwrap();
        let mut gsp = SimGsp::start(&mut shm, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&shm).unwrap();
        let mut rm = FakeRm::new();
        let mut clock = SimClock::default();
        let vram = 8u64 << 30;
        let mut mem =
            GpuMemory::new(64 << 20, vram - (256 << 20), Sim(SimVram::new(vram))).unwrap();
        let (g, u, w) = {
            let mut t = QueueTransport {
                q: &mut q,
                mem: &mut shm,
                clock: &mut clock,
                timeout_us: RM_REPLY_US,
                poll: |m: &mut dyn SharedMem| {
                    gsp.service_with(m, |f, p| rm.handle(f, p));
                },
                events: Vec::new(),
            };
            let g = init_golden_image(&mut t, &mut mem).unwrap();
            let u = open_user_space(&mut t, g.next_handle(), mem.root()).unwrap();
            let cg = open_compute_group(&mut t, &u).unwrap();
            assert_eq!(
                (cg.channel_group, cg.ctxshare, cg.subctx_id, cg.next_handle),
                (0xcf00_000b, 0xcf00_000c, 63, 0xcf00_000d)
            );
            // I9e: the working channel in the group, through the same queues.
            let w = open_work_channel(&mut t, &mut mem, &u, &cg, &g).unwrap();
            assert_eq!(
                (w.channel, w.compute, w.next_handle),
                (0xcf00_000d, 0xcf00_000e, 0xcf00_000f)
            );
            // The golden channel got id 0, this one 1; GR0 is on runlist 0 (SIM).
            assert_eq!((w.chid, w.doorbell_token), (1, 1));
            (g, u, w)
        };
        assert_eq!(rm.parent_of(w.channel), Some(0xcf00_000b));
        assert_eq!(rm.promoted(w.channel), [0, 1, 2, 0, 1, 2]);
        assert!(rm.is_scheduled(0xcf00_000b));
        // Ring, USERD and the per-channel GR buffers resolve through the root
        // GSP-RM was given (the independent walker).
        let root = rm.page_directory(USER_CLIENT, u.vaspace).unwrap().0;
        for (va, pa) in [
            (w.ring.va, w.ring.pa),
            (w.ring.va + w.userd_offset, w.ring.pa + w.userd_offset),
        ]
        .into_iter()
        .chain(w.ctx_bufs.iter().map(|(_, m)| (m.va, m.pa)))
        {
            assert_eq!(translate(&mem.writer().0, root, va).unwrap().pa, pa);
        }
        assert_eq!(g.next_handle(), 0xcf00_0007);
        // The root allocation takes a handle too (0xcf00_0007), as in the golden image.
        assert_eq!(
            (u.client, u.device, u.subdevice, u.vaspace, u.next_handle),
            (
                USER_CLIENT,
                0xcf00_0008,
                0xcf00_0009,
                0xcf00_000a,
                0xcf00_000b
            )
        );
        assert_eq!((u.root, u.root_entries), (mem.root(), 4));
        // The model saw the golden image's 7 allocations, then these 4 + the
        // group's 2 + the working channel and its compute object.
        assert_eq!(
            rm.log.allocs[7..],
            [
                c::ROOT,
                c::DEVICE,
                c::SUBDEVICE,
                c::VASPACE,
                c::CHANNEL_GROUP,
                c::CONTEXT_SHARE,
                c::CHANNEL_GPFIFO,
                c::COMPUTE
            ]
            .map(|x| (x, status::OK))
        );
        // Both are on the VA space that has our page tables.
        assert_eq!(rm.bound_vaspace(USER_CLIENT, 0xcf00_000b), Some(u.vaspace));
        assert_eq!(rm.bound_vaspace(USER_CLIENT, 0xcf00_000c), Some(u.vaspace));
        assert_eq!(rm.log.page_dirs, [(u.vaspace, status::OK)]);
        assert_eq!(
            rm.page_directory(USER_CLIENT, u.vaspace),
            Some((mem.root(), 4))
        );
        // From the root the model was given, a golden GR buffer resolves.
        let (_, m) = g.gr_bufs[0];
        let tr = translate(
            &mem.writer().0,
            rm.page_directory(USER_CLIENT, u.vaspace).unwrap().0,
            m.va,
        )
        .unwrap();
        assert_eq!(tr.pa, m.pa);
        // A buffer mapped after the RPC is visible through the same root.
        let late = mem.alloc_mapped(0x2000, 0x1000).unwrap();
        assert_eq!(
            translate(&mem.writer().0, u.root, late.va + 0x1008)
                .unwrap()
                .pa,
            late.pa + 0x1008
        );
    }

    fn opened() -> (Direct, UserSpace) {
        let mut d = Direct(FakeRm::new());
        let u = open_user_space(&mut d, 0x100, 0x7_0000_0000).unwrap();
        (d, u)
    }

    #[test]
    fn bad_roots_send_nothing() {
        for root in [0, 0x7_0000_0800, crate::mmu::PA_LIMIT] {
            let mut d = Direct(FakeRm::new());
            assert!(open_user_space(&mut d, 0x100, root).is_err());
            assert!(d.0.log.allocs.is_empty() && d.0.log.page_dirs.is_empty());
        }
    }

    /// The model's reading of RM's rules for SET_PAGE_DIRECTORY: each
    /// refusal comes back as RM's status code, never as success.
    #[test]
    fn rm_refusals_are_errors() {
        let (mut d, u) = opened();
        // A second root for an externally-owned VA space: INVALID_STATE.
        let again =
            rmapi::set_page_directory_rpc(USER_CLIENT, u.device, u.vaspace, u.root, 4).unwrap();
        assert_eq!(
            d.call(function::SET_PAGE_DIRECTORY, &again),
            Err(GpuError::HardwareFault {
                code: status::INVALID_STATE
            })
        );
        // Too few entries for the VA space's limit: INVALID_ARGUMENT.
        let (mut d, u) = {
            let mut d = Direct(FakeRm::new());
            let mut rm = Rm {
                t: &mut d,
                client: USER_CLIENT,
                next: 0x100,
            };
            rm.alloc(0, class::ROOT, &rmapi::root_params()).unwrap();
            let dev = rm
                .alloc(
                    USER_CLIENT,
                    class::DEVICE,
                    &rmapi::device_params(USER_CLIENT),
                )
                .unwrap();
            let va = rm
                .alloc(dev, class::VASPACE, &rmapi::user_vaspace_params())
                .unwrap();
            let plain = rm
                .alloc(dev, class::VASPACE, &rmapi::vaspace_params())
                .unwrap();
            (d, (dev, va, plain))
        };
        let (dev, va, plain) = u;
        for (msg, code) in [
            (
                rmapi::set_page_directory_rpc(USER_CLIENT, dev, va, 0x7_0000_0000, 2).unwrap(),
                status::INVALID_ARGUMENT,
            ),
            (
                rmapi::set_page_directory_rpc(USER_CLIENT, dev, va, 0x7_0000_0000, 5).unwrap(),
                status::INVALID_ARGUMENT,
            ),
            // A VA space RM manages itself: NOT_SUPPORTED.
            (
                rmapi::set_page_directory_rpc(USER_CLIENT, dev, plain, 0x7_0000_0000, 4).unwrap(),
                status::NOT_SUPPORTED,
            ),
            // The device handle where the VA space should be, another client.
            (
                rmapi::set_page_directory_rpc(USER_CLIENT, dev, dev, 0x7_0000_0000, 4).unwrap(),
                status::INVALID_OBJECT_HANDLE,
            ),
            (
                rmapi::set_page_directory_rpc(0xc1e0_0004, dev, va, 0x7_0000_0000, 4).unwrap(),
                status::INVALID_CLIENT,
            ),
        ] {
            assert_eq!(
                d.call(function::SET_PAGE_DIRECTORY, &msg),
                Err(GpuError::HardwareFault { code })
            );
        }
        // A truncated message: INVALID_PARAM_STRUCT.
        let ok = rmapi::set_page_directory_rpc(USER_CLIENT, dev, va, 0x7_0000_0000, 4).unwrap();
        assert_eq!(
            d.call(function::SET_PAGE_DIRECTORY, &ok[..40]),
            Err(GpuError::HardwareFault {
                code: status::INVALID_PARAM_STRUCT
            })
        );
        // After all the refusals the VA space has no root; then it gets one.
        assert_eq!(d.0.page_directory(USER_CLIENT, va), None);
        d.call(function::SET_PAGE_DIRECTORY, &ok).unwrap();
        assert_eq!(
            d.0.page_directory(USER_CLIENT, va),
            Some((0x7_0000_0000, 4))
        );
    }

    /// Planted mistakes in the message (the bytes our encoder writes) are
    /// caught by the model, which parses with its own hand-written offsets.
    #[test]
    fn planted_mistakes_are_caught() {
        let (_, u) = opened();
        let good =
            rmapi::set_page_directory_rpc(USER_CLIENT, u.device, u.vaspace, 0x7_0000_0000, 4)
                .unwrap();
        let fresh = || {
            let mut d = Direct(FakeRm::new());
            let u2 = {
                let mut rm = Rm {
                    t: &mut d,
                    client: USER_CLIENT,
                    next: 0x100,
                };
                rm.alloc(0, class::ROOT, &rmapi::root_params()).unwrap();
                let dev = rm
                    .alloc(
                        USER_CLIENT,
                        class::DEVICE,
                        &rmapi::device_params(USER_CLIENT),
                    )
                    .unwrap();
                rm.alloc(dev, class::SUBDEVICE, &rmapi::subdevice_params())
                    .unwrap();
                rm.alloc(dev, class::VASPACE, &rmapi::user_vaspace_params())
                    .unwrap()
            };
            assert_eq!(u2, u.vaspace);
            d
        };
        assert!(fresh().call(function::SET_PAGE_DIRECTORY, &good).is_ok());
        // (offset, value): params start at 16 in the RPC.
        for (off, v) in [
            (16usize, 0x0000_0800u32), // root not 4 KiB aligned (low half)
            (16 + 12, 3),              // aperture 3 (no such aperture)
            (16 + 16, u.device),       // hVASpace = the device
            (4, u.vaspace),            // hDevice = the VA space
            (16 + 24, 5),              // subDeviceId of a 5th GPU
        ] {
            let mut bad = good.clone();
            bad[off..off + 4].copy_from_slice(&v.to_le_bytes());
            assert!(
                fresh().call(function::SET_PAGE_DIRECTORY, &bad).is_err(),
                "greseala la {off} neprinsa"
            );
        }
    }

    // ---- I9d: channel group + context share (SIMULARE) ----

    #[test]
    fn compute_group_on_the_working_vaspace() {
        let (mut d, u) = opened();
        let cg = open_compute_group(&mut d, &u).unwrap();
        assert_eq!(
            (cg.channel_group, cg.ctxshare, cg.subctx_id, cg.next_handle),
            (0x104, 0x105, 63, 0x106)
        );
        assert_eq!(d.0.bound_vaspace(USER_CLIENT, 0x104), Some(u.vaspace));
        assert_eq!(d.0.bound_vaspace(USER_CLIENT, 0x105), Some(u.vaspace));
        assert_eq!(d.0.subcontexts(0x104), [63]);
        // A second group on the same client gets its own subcontext heap.
        let u2 = UserSpace {
            next_handle: cg.next_handle,
            ..u
        };
        let cg2 = open_compute_group(&mut d, &u2).unwrap();
        assert_eq!((cg2.channel_group, cg2.subctx_id), (0x106, 63));
    }

    /// Refused before anything is sent: no VA space, an unknown mode,
    /// SPECIFIED (verification only); answers that cannot be trusted.
    #[test]
    fn driver_side_refusals() {
        assert!(rmapi::channel_group_params(0).is_err());
        assert!(rmapi::ctxshare_params(0, rmapi::subcontext::ASYNC).is_err());
        assert!(rmapi::ctxshare_params(0x103, rmapi::subcontext::SPECIFIED).is_err());
        assert!(rmapi::ctxshare_params(0x103, 4).is_err());
        let sent = rmapi::ctxshare_params(0x103, rmapi::subcontext::ASYNC).unwrap();
        let mut got = sent.clone();
        got[8..12].copy_from_slice(&5u32.to_le_bytes());
        assert_eq!(rmapi::parse_ctxshare_reply(&sent, &got), Ok(5));
        // The "lower" success bit (31) is not part of the id.
        got[8..12].copy_from_slice(&(0x8000_0000u32 | 7).to_le_bytes());
        assert_eq!(rmapi::parse_ctxshare_reply(&sent, &got), Ok(7));
        for (off, v) in [(8usize, 64u32), (8, 0x7fff_ffff), (4, 0), (0, 0x104)] {
            let mut bad = sent.clone();
            bad[off..off + 4].copy_from_slice(&v.to_le_bytes());
            assert!(
                rmapi::parse_ctxshare_reply(&sent, &bad).is_err(),
                "{off}={v:#x}"
            );
        }
        assert!(rmapi::parse_ctxshare_reply(&sent, &sent[..8]).is_err());
        // A UserSpace with a bad root never reaches GSP-RM.
        let (mut d, u) = opened();
        let n = d.0.log.allocs.len();
        let bad = UserSpace { root: 0x800, ..u };
        assert!(open_compute_group(&mut d, &bad).is_err());
        assert_eq!(d.0.log.allocs.len(), n);
    }

    fn rm_on(d: &mut Direct, next: u32) -> Rm<'_, Direct> {
        Rm {
            t: d,
            client: USER_CLIENT,
            next,
        }
    }

    fn fault(code: u32) -> GpuResult<u32> {
        Err(GpuError::HardwareFault { code })
    }

    /// The model's reading of RM's rules for TSG and context share: each
    /// refusal comes back as RM's status code, never as success.
    #[test]
    fn rm_refusals_for_the_group() {
        let (mut d, u) = opened();
        let tsg_p = |vas: u32, eng: u32| {
            let mut b = rmapi::channel_group_params(u.vaspace).unwrap();
            b[8..12].copy_from_slice(&vas.to_le_bytes());
            b[12..16].copy_from_slice(&eng.to_le_bytes());
            b
        };
        let mut rm = rm_on(&mut d, 0x200);
        for (parent, p, code) in [
            (u.subdevice, tsg_p(u.vaspace, 1), status::INVALID_ARGUMENT), // not a device
            (u.device, tsg_p(u.vaspace, 0), status::INVALID_ARGUMENT),    // NULL engine
            (u.device, tsg_p(u.vaspace, 2), status::INVALID_ARGUMENT),    // GR1 without MIG
            (u.device, tsg_p(u.device, 1), status::INVALID_OBJECT_HANDLE),
            (u.device, tsg_p(0, 1), status::NOT_SUPPORTED), // device default: not modelled
        ] {
            assert_eq!(rm.alloc(parent, class::CHANNEL_GROUP, &p), fault(code));
        }
        let mut short = tsg_p(u.vaspace, 1);
        short.pop();
        assert_eq!(
            rm.alloc(u.device, class::CHANNEL_GROUP, &short),
            fault(status::INVALID_PARAM_STRUCT)
        );
        let tsg = rm
            .alloc(u.device, class::CHANNEL_GROUP, &tsg_p(u.vaspace, 1))
            .unwrap();
        let cs = |flags: u32, want: u32| {
            let mut b = alloc::vec![0u8; 12];
            b[0..4].copy_from_slice(&u.vaspace.to_le_bytes());
            b[4..8].copy_from_slice(&flags.to_le_bytes());
            b[8..12].copy_from_slice(&want.to_le_bytes());
            b
        };
        for (parent, p, code) in [
            (u.device, cs(1, 0), status::INVALID_OBJECT_HANDLE), // parent not a TSG
            (tsg, cs(4, 0), status::INVALID_ARGUMENT),           // no such mode
            (tsg, cs(2, 64), status::INVALID_ARGUMENT),          // SPECIFIED outside
        ] {
            assert_eq!(rm.alloc(parent, class::CONTEXT_SHARE, &p), fault(code));
        }
        // ASYNC fills 63 down to 1, then 0; the 65th is refused.
        let mut ids = Vec::new();
        for _ in 0..64 {
            let (_, got) = rm
                .alloc_reply(tsg, class::CONTEXT_SHARE, &cs(1, 0))
                .unwrap();
            ids.push(rmapi::parse_ctxshare_reply(&cs(1, 0), &got).unwrap());
        }
        assert_eq!(ids[..2], [63, 62]);
        assert_eq!(ids[62..], [1, 0]);
        assert_eq!(
            rm.alloc(tsg, class::CONTEXT_SHARE, &cs(1, 0)),
            fault(status::INSUFFICIENT_RESOURCES)
        );
        // SYNC (0) and SPECIFIED on a taken id: no resources.
        assert_eq!(
            rm.alloc(tsg, class::CONTEXT_SHARE, &cs(0, 0)),
            fault(status::INSUFFICIENT_RESOURCES)
        );
        assert_eq!(
            rm.alloc(tsg, class::CONTEXT_SHARE, &cs(2, 5)),
            fault(status::INSUFFICIENT_RESOURCES)
        );
    }

    /// Planted mistakes in the bytes our encoders write are caught by the
    /// model (its own hand-written offsets).
    #[test]
    fn planted_group_mistakes_are_caught() {
        let (_, u) = opened();
        let good = rmapi::channel_group_params(u.vaspace).unwrap();
        for (off, v) in [(12usize, 2u32), (12, 0), (8, u.device), (8, 0), (0, 0)] {
            let mut bad = good.clone();
            bad[off..off + 4].copy_from_slice(&v.to_le_bytes());
            let (mut d, _) = opened();
            let r = rm_on(&mut d, 0x200).alloc(u.device, class::CHANNEL_GROUP, &bad);
            // (0, 0) leaves the params unchanged: it must still pass.
            assert_eq!(r.is_ok(), off == 0, "greseala TSG la {off} = {v:#x}");
        }
        let good = rmapi::ctxshare_params(u.vaspace, rmapi::subcontext::ASYNC).unwrap();
        for (off, v) in [(0usize, u.device), (0, 0), (4, 9), (4, 2)] {
            let mut bad = good.clone();
            bad[off..off + 4].copy_from_slice(&v.to_le_bytes());
            let (mut d, _) = opened();
            let mut rm = rm_on(&mut d, 0x200);
            let tsg = rm
                .alloc(
                    u.device,
                    class::CHANNEL_GROUP,
                    &rmapi::channel_group_params(u.vaspace).unwrap(),
                )
                .unwrap();
            let r = rm.alloc(tsg, class::CONTEXT_SHARE, &bad);
            // (4, 2) = SPECIFIED with id 0: RM accepts it; our encoder never sends it.
            assert_eq!(
                r.is_ok(),
                (off, v) == (4, 2),
                "greseala ctxshare la {off} = {v:#x}"
            );
        }
    }

    // ---- I9e: the working channel (SIMULARE) ----

    type Mem = GpuMemory<Sim>;

    /// Golden image + working client + group over the direct transport, on
    /// real page tables in a simulated VRAM.
    fn grouped() -> (Direct, Mem, GoldenImage, UserSpace, ComputeGroup) {
        let mut d = Direct(FakeRm::new());
        let vram = 8u64 << 30;
        let mut mem =
            GpuMemory::new(64 << 20, vram - (256 << 20), Sim(SimVram::new(vram))).unwrap();
        let g = init_golden_image(&mut d, &mut mem).unwrap();
        let u = open_user_space(&mut d, g.next_handle(), mem.root()).unwrap();
        let cg = open_compute_group(&mut d, &u).unwrap();
        (d, mem, g, u, cg)
    }

    #[test]
    fn work_channel_in_the_group() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let w = open_work_channel(&mut d, &mut mem, &u, &cg, &g).unwrap();
        assert_eq!(d.0.parent_of(w.channel), Some(cg.channel_group));
        assert_eq!(d.0.parent_of(w.compute), Some(w.channel));
        assert_eq!(d.0.chid(w.channel), Some(w.chid));
        assert_eq!(w.doorbell_token, w.chid); // runlist 0 in the model
        assert_eq!(
            w.ctx_bufs.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            WORK_CTX_BUFFERS
        );
        // New buffers, not the golden ones.
        for (_, m) in &w.ctx_bufs {
            assert!(g.gr_bufs.iter().all(|(_, gm)| gm.pa != m.pa));
        }
        assert_eq!(w.userd_offset, 0x2000);
        assert!(d.0.is_scheduled(cg.channel_group));
        // C-87: the group can be taken off again (recovery), and put back.
        stop_compute_group(&mut d, &u, &cg).unwrap();
        assert!(!d.0.is_scheduled(cg.channel_group));
        assert_eq!(
            d.0.log.controls.last(),
            Some(&(c_cmd::TSG_SCHEDULE, status::OK))
        );
        d.0.log.controls.pop();
        Rm {
            t: &mut d,
            client: u.client,
            next: w.next_handle,
        }
        .control(
            cg.channel_group,
            cmd::TSG_SCHEDULE,
            &rmapi::schedule_params(true),
        )
        .unwrap();
        d.0.log.controls.pop();
        assert!(d.0.is_scheduled(cg.channel_group));
        // The order GSP-RM saw: channel, compute, 2 promotes, token, schedule.
        let n = d.0.log.controls.len();
        assert_eq!(
            d.0.log.controls[n - 4..],
            [
                (c_cmd::PROMOTE_CTX, status::OK),
                (c_cmd::PROMOTE_CTX, status::OK),
                (c_cmd::WORK_SUBMIT_TOKEN, status::OK),
                (c_cmd::TSG_SCHEDULE, status::OK)
            ]
        );
        // A second channel in the same group gets the next id.
        let cg2 = ComputeGroup {
            next_handle: w.next_handle,
            ..cg
        };
        let w2 = open_work_channel(&mut d, &mut mem, &u, &cg2, &g).unwrap();
        assert_eq!(w2.chid, w.chid + 1);
    }

    #[test]
    fn doorbell_token_rules() {
        assert_eq!(rmapi::doorbell_token(5, 0), Ok(5));
        assert_eq!(rmapi::doorbell_token(0x7ff, 3), Ok(0x3_07ff));
        assert_eq!(rmapi::doorbell_token(0, 0x7f), Ok(0x7f_0000));
        // Not a channel id (the request's -1 back, or high bits): refused.
        for t in [u32::MAX, 0x1000, 0x1_0005] {
            assert!(rmapi::doorbell_token(t, 0).is_err(), "{t:#x}");
        }
        assert!(rmapi::doorbell_token(1, 0x80).is_err());
        assert!(rmapi::parse_work_submit_token(&[0; 3]).is_err());
        assert_eq!(rmapi::work_submit_token_request(), [0xff; 4]);
        assert_eq!(rmapi::schedule_params(true), [1, 0]);
    }

    fn setup_of(w: &WorkChannel, ctxshare: u32) -> WorkChannelSetup {
        WorkChannelSetup {
            ring_va: w.ring.va,
            ring_pa: w.ring.pa,
            entries: w.entries,
            ctxshare,
            inst_pa: w.inst_pa,
            mthd_pa: w.mthd_pa,
            notifier_pa: w.notifier_pa,
        }
    }

    /// Refused by the driver before anything is sent.
    #[test]
    fn work_channel_driver_side_refusals() {
        let (_, _, g, u, cg) = grouped();
        let ok = WorkChannelSetup {
            ring_va: 0x10_0000,
            ring_pa: 0x0400_0000,
            entries: 0x400,
            ctxshare: cg.ctxshare,
            inst_pa: 0x0401_0000,
            mthd_pa: 0x0402_0000,
            notifier_pa: 0x0403_0000,
        };
        assert!(rmapi::work_channel_params(&ok).is_ok());
        for bad in [
            WorkChannelSetup {
                entries: 0x300,
                ..ok
            },
            WorkChannelSetup { entries: 1, ..ok },
            WorkChannelSetup { ring_va: 0, ..ok },
            WorkChannelSetup {
                ring_va: 0x10_0800,
                ..ok
            },
            WorkChannelSetup { ctxshare: 0, ..ok },
            WorkChannelSetup {
                inst_pa: 0x0401_0100,
                ..ok
            },
            WorkChannelSetup { mthd_pa: 0, ..ok },
            // A PA whose low 32 bits are zero cannot be a non-zero handle.
            WorkChannelSetup {
                ring_pa: 1 << 32,
                ..ok
            },
            WorkChannelSetup {
                notifier_pa: 2 << 32,
                ..ok
            },
        ] {
            assert!(rmapi::work_channel_params(&bad).is_err(), "{bad:?}");
        }
        // No GR0 runlist or a golden image without the buffers: nothing sent.
        let mut d = Direct(FakeRm::new());
        let mut mem = GpuMemory::new(64 << 20, 1 << 30, Sim(SimVram::new(2 << 30))).unwrap();
        for g2 in [
            GoldenImage {
                runlists: alloc::vec![(9, 1)],
                ..g.clone()
            },
            GoldenImage {
                gr_bufs: g.gr_bufs[3..].to_vec(),
                ..g.clone()
            },
        ] {
            assert!(open_work_channel(&mut d, &mut mem, &u, &cg, &g2).is_err());
        }
        assert!(d.0.log.allocs.is_empty() && d.0.log.controls.is_empty());
    }

    /// The model's reading of RM's rules for a TSG channel, the token and
    /// the schedule: each refusal comes back as RM's status code.
    #[test]
    fn rm_refusals_for_the_work_channel() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let w = open_work_channel(&mut d, &mut mem, &u, &cg, &g).unwrap();
        let good = rmapi::work_channel_params(&setup_of(&w, cg.ctxshare)).unwrap();
        let with = |off: usize, v: u32| {
            let mut b = good.clone();
            b[off..off + 4].copy_from_slice(&v.to_le_bytes());
            b
        };
        // A second group with its own context share, to name the wrong one.
        let mut rm = rm_on(&mut d, 0x300);
        let tsg2 = rm
            .alloc(
                u.device,
                class::CHANNEL_GROUP,
                &rmapi::channel_group_params(u.vaspace).unwrap(),
            )
            .unwrap();
        let cs2 = rm
            .alloc(
                tsg2,
                class::CONTEXT_SHARE,
                &rmapi::ctxshare_params(u.vaspace, rmapi::subcontext::ASYNC).unwrap(),
            )
            .unwrap();
        for (parent, p, code) in [
            (
                cg.channel_group,
                with(28, u.vaspace),
                status::INVALID_ARGUMENT,
            ), // VA space too
            (cg.channel_group, with(24, 0), status::NOT_SUPPORTED), // legacy mode
            (
                cg.channel_group,
                with(24, u.vaspace),
                status::OBJECT_NOT_FOUND,
            ), // not a ctxshare
            (
                cg.channel_group,
                with(24, cs2),
                status::INVALID_OBJECT_PARENT,
            ), // other TSG's
            (cg.channel_group, with(128, 2), status::INVALID_ARGUMENT), // GR1
            (cg.channel_group, with(20, 1), status::NOT_SUPPORTED), // not physical
            (u.device, good.clone(), status::INVALID_ARGUMENT), // device channel with a ctxshare
        ] {
            assert_eq!(rm.alloc(parent, class::CHANNEL_GPFIFO, &p), fault(code));
        }
        // A channel with no object bound: the group cannot be scheduled.
        let lone = rm
            .alloc(tsg2, class::CHANNEL_GPFIFO, &with(24, cs2))
            .unwrap();
        assert_eq!(
            rm.control(tsg2, cmd::TSG_SCHEDULE, &rmapi::schedule_params(true)),
            Err(GpuError::HardwareFault {
                code: status::INVALID_STATE
            })
        );
        // The token is a channel command; on the group it is refused.
        assert_eq!(
            rm.control(
                tsg2,
                cmd::WORK_SUBMIT_TOKEN,
                &rmapi::work_submit_token_request()
            ),
            Err(GpuError::HardwareFault {
                code: status::INVALID_OBJECT_HANDLE
            })
        );
        rm.alloc(lone, class::COMPUTE, &[]).unwrap();
        rm.control(tsg2, cmd::TSG_SCHEDULE, &rmapi::schedule_params(true))
            .unwrap();
    }

    /// Without a golden context, no compute object on a TSG channel.
    #[test]
    fn no_golden_no_compute() {
        let mut d = Direct(FakeRm::new());
        let u = open_user_space(&mut d, 0x100, 0x7_0000_0000).unwrap();
        let cg = open_compute_group(&mut d, &u).unwrap();
        let setup = WorkChannelSetup {
            ring_va: 0x10_0000,
            ring_pa: 0x0400_0000,
            entries: 0x400,
            ctxshare: cg.ctxshare,
            inst_pa: 0x0401_0000,
            mthd_pa: 0x0402_0000,
            notifier_pa: 0x0403_0000,
        };
        let mut rm = rm_on(&mut d, cg.next_handle);
        let ch = rm
            .alloc(
                cg.channel_group,
                class::CHANNEL_GPFIFO,
                &rmapi::work_channel_params(&setup).unwrap(),
            )
            .unwrap();
        assert_eq!(
            rm.alloc(ch, class::COMPUTE, &[]),
            fault(status::INVALID_STATE)
        );
    }

    /// Planted mistakes in the bytes `work_channel_params` writes are caught
    /// by the model (its own hand-written offsets).
    #[test]
    fn planted_work_channel_mistakes_are_caught() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let w = open_work_channel(&mut d, &mut mem, &u, &cg, &g).unwrap();
        let good = rmapi::work_channel_params(&setup_of(&w, cg.ctxshare)).unwrap();
        let mut rm = rm_on(&mut d, 0x400);
        assert!(
            rm.alloc(cg.channel_group, class::CHANNEL_GPFIFO, &good)
                .is_ok()
        );
        for (off, v) in [
            (24usize, u.vaspace), // context share = the VA space
            (28, u.vaspace),      // VA space set as well
            (16, 0x300),          // entries not a power of 2
            (8, 0x800),           // ring VA not aligned
            (64, 0x100),          // USERD inside the ring
            (144 + 16, 1),        // instance not in VRAM
            (168 + 8, 0x100),     // USERD too small
            (216 + 8, 0x1000),    // method buffer too small
            (192, 0x10),          // RAMFC base not aligned (low half)
        ] {
            let mut bad = good.clone();
            bad[off..off + 4].copy_from_slice(&v.to_le_bytes());
            assert!(
                rm.alloc(cg.channel_group, class::CHANNEL_GPFIFO, &bad)
                    .is_err(),
                "greseala la {off} = {v:#x} neprinsa"
            );
        }
    }

    // ---- I10a: the first pushbuffer on the working channel (SIMULARE) ----

    use crate::gsp::work_queue::{
        COMPUTE_SUBCH, ChannelMemory, Doorbell, LOCAL_MEM_WINDOW, SHARED_MEM_WINDOW, WorkQueue,
        compute_setup,
    };
    use crate::push::{DOORBELL_INFERRED, PushBuf};
    use gpu_sim::cmdproc::{CommandProcessor, Event, USERD_GP_GET};

    /// 32-bit stores on the 64-bit simulated VRAM (read-modify-write; SIM).
    impl ChannelMemory for Sim {
        fn write_u32(&mut self, pa: u64, v: u32) -> GpuResult<()> {
            let (a, sh) = (pa & !7, (pa & 4) * 8);
            let old = self.0.read_u64(a).map_err(GpuError::InvalidArgument)?;
            let new = (old & !(0xffff_ffff << sh)) | ((v as u64) << sh);
            self.0.write_u64(a, new).map_err(GpuError::InvalidArgument)
        }
        fn read_u32(&mut self, pa: u64) -> GpuResult<u32> {
            let v = self
                .0
                .read_u64(pa & !7)
                .map_err(GpuError::InvalidArgument)?;
            Ok((v >> ((pa & 4) * 8)) as u32)
        }
    }

    fn setup_words() -> Vec<u32> {
        let mut buf = [0u32; 16];
        let mut pb = PushBuf::new(&mut buf);
        compute_setup(&mut pb, LOCAL_MEM_WINDOW, SHARED_MEM_WINDOW).unwrap();
        pb.words().to_vec()
    }

    /// The GPU's side: from the root GSP-RM was given, the model fetches
    /// ring, USERD and pushbuffer, runs the ring and writes GPGet back.
    fn gpu_runs(mem: &mut Mem, root: u64, w: &WorkChannel, push: &Mapping) -> CommandProcessor {
        let mut cp = CommandProcessor::new();
        let v = &mem.writer().0;
        cp.mem
            .load_from_vram(v, root, w.ring.va, w.ring.size)
            .unwrap();
        cp.mem.load_from_vram(v, root, push.va, push.size).unwrap();
        let userd = w.ring.va + w.userd_offset;
        cp.run_channel(w.ring.va, w.entries, userd).unwrap();
        cp.mem
            .store_to_vram(&mut mem.writer_mut().0, root, userd + USERD_GP_GET, 8)
            .unwrap();
        cp
    }

    /// SIMULARE end to end: golden image → working client → group → working
    /// channel (I9c–I9e), then the compute setup written by the driver into
    /// a pushbuffer, the GPFIFO entry into the ring and GPPut into USERD;
    /// the model reads them only through the page tables GSP-RM was given.
    #[test]
    fn first_pushbuffer_on_the_work_channel() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let w = open_work_channel_cleared(&mut d, &mut mem, &u, &cg, &g).unwrap();
        let root = d.0.page_directory(USER_CLIENT, u.vaspace).unwrap().0;
        // USERD was zeroed before the channel existed.
        let userd_pa = w.ring.pa + w.userd_offset;
        assert_eq!(mem.writer().0.read_u64(userd_pa + USERD_GP_GET), Ok(0));
        let push = mem.alloc_mapped(0x1000, 0x1000).unwrap();
        let mut q = WorkQueue::new(&w, push).unwrap();
        let words = setup_words();
        let bell = q.submit(mem.writer_mut(), &words).unwrap();
        // The doorbell is only described: tinygrad's offset, the token.
        assert_eq!(
            bell,
            Doorbell {
                bar0_offset: DOORBELL_INFERRED,
                value: w.doorbell_token
            }
        );
        assert_eq!((q.put(), q.get()), (1, 0));
        let cp = gpu_runs(&mut mem, root, &w, &push);
        assert_eq!(
            cp.events,
            [Event::SetObject {
                subch: COMPUTE_SUBCH,
                class: class::COMPUTE
            }]
        );
        assert_eq!(cp.compute.local_window, Some(LOCAL_MEM_WINDOW));
        assert_eq!(cp.compute.shared_window, Some(SHARED_MEM_WINDOW));
        assert_eq!(cp.compute.local_memory, None);
        assert_eq!(cp.entries_taken, 1);
        // The driver reads GPGet back and accepts it.
        assert_eq!(q.update_get(mem.writer_mut()), Ok(1));
        assert_eq!(q.get(), 1);

        // Wrap-around: 1023 one-dword NOPs fill the ring up to the slot
        // before GET; one more is refused without writing anything.
        let nop = PushBuf::new(&mut [0u32; 2]).words().len(); // 0
        assert_eq!(nop, 0);
        let mut nb = [0u32; 2];
        let mut pb = PushBuf::new(&mut nb);
        pb.inc(
            COMPUTE_SUBCH,
            nvidia_rm570::regs::clc7c0::NVC7C0_SET_OBJECT,
            &[class::COMPUTE],
        )
        .unwrap();
        let one = pb.words().to_vec();
        let big = mem.alloc_mapped(0x4000, 0x1000).unwrap();
        let mut q2 = WorkQueue::new(&w, big).unwrap();
        // Bring q2 to the channel's current state (GET = PUT = 1).
        q2.submit(mem.writer_mut(), &one).unwrap();
        mem.writer_mut()
            .write_u32(w.ring.pa + w.userd_offset + USERD_GP_GET, 1)
            .unwrap();
        assert_eq!(q2.update_get(mem.writer_mut()), Ok(1));
        for _ in 0..w.entries - 1 {
            q2.submit(mem.writer_mut(), &one).unwrap();
        }
        let before = mem
            .writer()
            .0
            .read_u64(w.ring.pa + w.userd_offset + 0x88)
            .unwrap();
        assert!(q2.submit(mem.writer_mut(), &one).is_err(), "inelul plin");
        assert_eq!(
            mem.writer()
                .0
                .read_u64(w.ring.pa + w.userd_offset + 0x88)
                .unwrap(),
            before
        );
        assert_eq!(q2.put(), 0, "PUT a trecut peste capat");
        let cp = gpu_runs(&mut mem, root, &w, &big);
        assert_eq!(cp.entries_taken, w.entries - 1);
        assert_eq!(cp.events.len(), (w.entries - 1) as usize);
        assert_eq!(q2.update_get(mem.writer_mut()), Ok(w.entries - 1));
    }

    /// Without the clearing, USERD holds whatever VRAM had (the model's
    /// poison): the GPU would read a junk GPGet/GPPut — the model refuses.
    #[test]
    fn uncleared_userd_is_junk() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let w = open_work_channel(&mut d, &mut mem, &u, &cg, &g).unwrap();
        let root = d.0.page_directory(USER_CLIENT, u.vaspace).unwrap().0;
        let mut cp = CommandProcessor::new();
        cp.mem
            .load_from_vram(&mem.writer().0, root, w.ring.va, w.ring.size)
            .unwrap();
        assert!(
            cp.run_channel(w.ring.va, w.entries, w.ring.va + w.userd_offset)
                .is_err()
        );
    }

    /// A GPGet the GPU could not have written is refused; a pushbuffer area
    /// that overlaps the ring is refused.
    #[test]
    fn work_queue_refusals() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let w = open_work_channel_cleared(&mut d, &mut mem, &u, &cg, &g).unwrap();
        assert!(WorkQueue::new(&w, w.ring).is_err());
        let push = mem.alloc_mapped(0x1000, 0x1000).unwrap();
        let mut q = WorkQueue::new(&w, push).unwrap();
        q.submit(mem.writer_mut(), &setup_words()).unwrap();
        let get_pa = w.ring.pa + w.userd_offset + USERD_GP_GET;
        for bad in [2, w.entries, u32::MAX] {
            mem.writer_mut().write_u32(get_pa, bad).unwrap();
            assert!(q.update_get(mem.writer_mut()).is_err(), "GPGet {bad}");
        }
        assert_eq!(q.get(), 0);
        // Too big for the area: refused, nothing written.
        let tiny = Mapping { size: 8, ..push };
        let mut q = WorkQueue::new(&w, tiny).unwrap();
        let put_pa = w.ring.pa + w.userd_offset + 0x88;
        let before = mem.writer().0.read_u64(put_pa).unwrap();
        assert!(q.submit(mem.writer_mut(), &setup_words()).is_err());
        assert_eq!(mem.writer().0.read_u64(put_pa).unwrap(), before);
    }

    /// Mistakes planted in what the driver writes are caught by the model
    /// (it reads only what the headers say).
    #[test]
    fn planted_submit_mistakes_are_caught() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let w = open_work_channel_cleared(&mut d, &mut mem, &u, &cg, &g).unwrap();
        let root = d.0.page_directory(USER_CLIENT, u.vaspace).unwrap().0;
        let good = setup_words();
        let mut cases: Vec<(&str, Vec<u32>)> = Vec::new();
        let mut m = good.clone();
        m.swap(3, 4); // window halves swapped (upper=0, lower=0x7293)
        cases.push(("jumatati inversate", m));
        let mut m = good.clone();
        m[2] &= !(1 << 13); // window methods on subchannel 0 (no compute object)
        cases.push(("subcanal gresit", m));
        let mut m = good.clone();
        m[0] = (m[0] & !0x0fff) | (0x7b4 >> 2); // SET_OBJECT header points at WINDOW_B
        cases.push(("metoda gresita", m));
        let mut m = good.clone();
        m.truncate(7); // last dword lost
        cases.push(("trunchiat", m));
        for (what, words) in cases {
            let push = mem.alloc_mapped(0x1000, 0x1000).unwrap();
            // A fresh channel each time, so GET = PUT = 0.
            let cg2 = ComputeGroup {
                next_handle: 0x500 + 0x10 * what.len() as u32,
                ..cg
            };
            let w2 = open_work_channel_cleared(&mut d, &mut mem, &u, &cg2, &g).unwrap();
            let mut q = WorkQueue::new(&w2, push).unwrap();
            q.submit(mem.writer_mut(), &words).unwrap();
            let mut cp = CommandProcessor::new();
            let v = &mem.writer().0;
            cp.mem
                .load_from_vram(v, root, w2.ring.va, w2.ring.size)
                .unwrap();
            cp.mem.load_from_vram(v, root, push.va, push.size).unwrap();
            let r = cp.run_channel(w2.ring.va, w2.entries, w2.ring.va + w2.userd_offset);
            let wrong = r.is_err()
                || cp.compute.local_window != Some(LOCAL_MEM_WINDOW)
                || cp.compute.shared_window != Some(SHARED_MEM_WINDOW);
            assert!(wrong, "greseala „{what}” neprinsa");
        }
        let _ = w;
    }

    // ---- I10b: the first launch on the working channel (SIMULARE) ----

    use crate::gsp::work_queue::{invalidate_caches, launch, read_semaphore, release};
    use crate::qmd::ComputeQmd;

    const ADD64: &[u8] = include_bytes!("../../../../tools/nak/out/add64.sm86.sass");
    const DONE: u64 = 0x1_0000_0007; // both halves of the 64-bit payload matter

    /// Driver-side buffers for one `add64` launch, all mapped in our VA.
    struct Add64 {
        prog: Mapping,
        cbuf: Mapping,
        a: Mapping,
        b: Mapping,
        c: Mapping,
        qmd: Mapping,
        sem: Mapping,
        want: Vec<u32>,
    }

    fn put_bytes(w: &mut Sim, pa: u64, bytes: &[u8]) {
        for (i, ch) in bytes.chunks(4).enumerate() {
            let mut b = [0u8; 4];
            b[..ch.len()].copy_from_slice(ch);
            w.write_u32(pa + 4 * i as u64, u32::from_le_bytes(b))
                .unwrap();
        }
    }

    /// The driver writes program, arguments, inputs, QMD and a zeroed
    /// semaphore into VRAM by physical address (what the OS would do
    /// through BAR1 in the window with Adrian).
    fn add64_buffers(mem: &mut Mem) -> Add64 {
        let mut m = || mem.alloc_mapped(0x1000, 0x1000).unwrap();
        let (prog, cbuf, a, b, c, qmd, sem) = (m(), m(), m(), m(), m(), m(), m());
        let x: Vec<f32> = (0..64).map(|i| i as f32 * 0.25 - 3.0).collect();
        let y: Vec<f32> = (0..64).map(|i| (i * i) as f32 / 7.0).collect();
        let mut want = alloc::vec![0.0f32; 64];
        gpu_compute::reference::add_ftz(&x, &y, &mut want).unwrap();
        let w = mem.writer_mut();
        put_bytes(w, prog.pa, ADD64);
        let mut cb = alloc::vec![0u8; 0x100];
        for (off, va) in [(0x28, a.va), (0x30, b.va), (0x38, c.va)] {
            cb[off..off + 8].copy_from_slice(&va.to_le_bytes());
        }
        put_bytes(w, cbuf.pa, &cb);
        let f = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        put_bytes(w, a.pa, &f(&x));
        put_bytes(w, b.pa, &f(&y));
        put_bytes(w, c.pa, &f(&[f32::NAN; 64]));
        let q = ComputeQmd {
            program_va: prog.va,
            registers: 19,
            cbuf0_va: cbuf.va,
            cbuf0_size: 0x100,
            grid: [1, 1, 1],
            block: [64, 1, 1],
            release: None,
        };
        put_bytes(w, qmd.pa, &q.to_bytes().unwrap());
        put_bytes(w, sem.pa, &[0; 8]);
        Add64 {
            prog,
            cbuf,
            a,
            b,
            c,
            qmd,
            sem,
            want: want.iter().map(|v| v.to_bits()).collect(),
        }
    }

    /// The GPU's side for a launch: everything it reads is fetched through
    /// the page tables GSP-RM was given; what it writes (GPGet, C, the
    /// semaphore) goes back the same way. Returns the model and its result.
    fn gpu_launches(
        mem: &mut Mem,
        root: u64,
        w: &WorkChannel,
        push: &Mapping,
        k: &Add64,
    ) -> (CommandProcessor, GpuResult<u32>) {
        let mut cp = CommandProcessor::new();
        cp.register_program(ADD64, gpu_sim::cmdproc::add64_on_cpu);
        let v = &mem.writer().0;
        for m in [
            &w.ring, push, &k.prog, &k.cbuf, &k.a, &k.b, &k.c, &k.qmd, &k.sem,
        ] {
            cp.mem.load_from_vram(v, root, m.va, m.size).unwrap();
        }
        let userd = w.ring.va + w.userd_offset;
        let r = cp.run_channel(w.ring.va, w.entries, userd);
        if r.is_ok() {
            let v = &mut mem.writer_mut().0;
            cp.mem
                .store_to_vram(v, root, userd + USERD_GP_GET, 8)
                .unwrap();
            cp.mem.store_to_vram(v, root, k.c.va, k.c.size).unwrap();
            cp.mem.store_to_vram(v, root, k.sem.va, 16).unwrap();
        }
        (cp, r)
    }

    fn read_c(mem: &mut Mem, k: &Add64) -> Vec<u32> {
        (0..64)
            .map(|i| mem.writer_mut().read_u32(k.c.pa + 4 * i).unwrap())
            .collect()
    }

    /// SIMULARE end to end: golden image → I9c–I9e → I10a setup → I10b
    /// launch of the NAK-compiled `add64` and the completion semaphore,
    /// two GPFIFO entries, read by the model only through the page tables;
    /// the driver sees the semaphore and the result in VRAM.
    #[test]
    fn first_launch_on_the_work_channel() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let w = open_work_channel_cleared(&mut d, &mut mem, &u, &cg, &g).unwrap();
        // I10c: the board must be sm_86 before our program is launched.
        query_gr_info(&mut d, &u).unwrap().check_sm86().unwrap();
        let root = d.0.page_directory(USER_CLIENT, u.vaspace).unwrap().0;
        let k = add64_buffers(&mut mem);
        let push = mem.alloc_mapped(0x1000, 0x1000).unwrap();
        let mut q = WorkQueue::new(&w, push).unwrap();
        q.submit(mem.writer_mut(), &setup_words()).unwrap();
        let bell = q
            .launch_and_signal(mem.writer_mut(), k.qmd.va, k.sem.va, DONE)
            .unwrap();
        assert_eq!(bell.value, w.doorbell_token);
        assert_eq!(read_semaphore(mem.writer_mut(), k.sem.pa), Ok(0));
        let (cp, r) = gpu_launches(&mut mem, root, &w, &push, &k);
        assert_eq!(r, Ok(2));
        assert_eq!(
            cp.events,
            [
                Event::SetObject {
                    subch: COMPUTE_SUBCH,
                    class: class::COMPUTE
                },
                Event::Launch {
                    qmd: k.qmd.va,
                    program: 0
                },
                Event::Release {
                    addr: k.sem.va,
                    payload: DONE
                },
            ]
        );
        assert_eq!(cp.cache_invalidations, 1);
        // The driver's view, by physical address.
        assert_eq!(q.update_get(mem.writer_mut()), Ok(2));
        assert_eq!(read_semaphore(mem.writer_mut(), k.sem.pa), Ok(DONE));
        assert_eq!(read_c(&mut mem, &k), k.want, "bit cu bit, add_ftz");
    }

    /// One more pass of the GPU over the channel, with the same model (it
    /// keeps the compute object bound by the first pass); everything is
    /// fetched again through the page tables and written back the same way.
    fn gpu_runs_again(
        cp: &mut CommandProcessor,
        mem: &mut Mem,
        root: u64,
        w: &WorkChannel,
        push: &Mapping,
        k: &Add64,
    ) -> GpuResult<u32> {
        let v = &mem.writer().0;
        for m in [
            &w.ring, push, &k.prog, &k.cbuf, &k.a, &k.b, &k.c, &k.qmd, &k.sem,
        ] {
            cp.mem.load_from_vram(v, root, m.va, m.size).unwrap();
        }
        let userd = w.ring.va + w.userd_offset;
        let r = cp.run_channel(w.ring.va, w.entries, userd)?;
        let v = &mut mem.writer_mut().0;
        cp.mem
            .store_to_vram(v, root, userd + USERD_GP_GET, 8)
            .unwrap();
        cp.mem.store_to_vram(v, root, k.c.va, k.c.size).unwrap();
        cp.mem.store_to_vram(v, root, k.sem.va, 8).unwrap();
        Ok(r)
    }

    fn push_words(mem: &mut Mem, push: &Mapping) -> Vec<u32> {
        (0..push.size / 4)
            .map(|i| mem.writer_mut().read_u32(push.pa + 4 * i).unwrap())
            .collect()
    }

    /// SIMULARE: the pushbuffer area is a circle. A small area (setup + two
    /// launches) carries 12 launches: a stretch is written again only after
    /// the completion semaphore of a later (or its own) launch says the GPU
    /// read it; while something is in flight, the words the GPU has not read
    /// are never overwritten, and a launch that does not fit is refused with
    /// nothing written. Each launch recomputes `add64` bit for bit.
    #[test]
    fn pushbuffer_area_is_reused_after_the_semaphore() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let w = open_work_channel_cleared(&mut d, &mut mem, &u, &cg, &g).unwrap();
        let root = d.0.page_directory(USER_CLIENT, u.vaspace).unwrap().0;
        let k = add64_buffers(&mut mem);
        let setup = 4 * setup_words().len() as u64;
        let one = 48; // invalidate + launch + release: 12 words
        assert!(setup < one);
        let area = mem.alloc_mapped(0x1000, 0x1000).unwrap();
        let push = Mapping {
            size: setup + 2 * one,
            ..area
        };
        let mut q = WorkQueue::new(&w, push).unwrap();
        let put_pa = w.ring.pa + w.userd_offset + crate::gsp::work_queue::USERD_GP_PUT;
        let nan = alloc::vec![0x7fc0_0000u32; 64];
        let reset_c = |mem: &mut Mem| {
            for (i, &b) in nan.iter().enumerate() {
                mem.writer_mut()
                    .write_u32(k.c.pa + 4 * i as u64, b)
                    .unwrap();
            }
        };

        // Fill the area: setup + launches 1 and 2. Launch 3 does not fit.
        q.submit(mem.writer_mut(), &setup_words()).unwrap();
        q.launch_and_signal(mem.writer_mut(), k.qmd.va, k.sem.va, 1)
            .unwrap();
        q.launch_and_signal(mem.writer_mut(), k.qmd.va, k.sem.va, 2)
            .unwrap();
        assert_eq!(q.push_in_flight(), push.size);
        let words = push_words(&mut mem, &push);
        let put = mem.writer_mut().read_u32(put_pa).unwrap();
        assert!(
            q.launch_and_signal(mem.writer_mut(), k.qmd.va, k.sem.va, 3)
                .is_err()
        );
        assert_eq!(push_words(&mut mem, &push), words, "nimic suprascris");
        assert_eq!(mem.writer_mut().read_u32(put_pa).unwrap(), put);

        // Nothing is freed before the GPU has run (semaphore still 0).
        assert_eq!(read_semaphore(mem.writer_mut(), k.sem.pa), Ok(0));
        assert_eq!(q.retire(k.sem.va, 0), Ok(0));
        // A value the GPU was never asked to write is a fault.
        assert!(q.retire(k.sem.va, 3).is_err());
        assert_eq!(q.push_in_flight(), push.size);

        let mut cp = CommandProcessor::new();
        cp.register_program(ADD64, gpu_sim::cmdproc::add64_on_cpu);
        reset_c(&mut mem);
        assert_eq!(
            gpu_runs_again(&mut cp, &mut mem, root, &w, &push, &k),
            Ok(3)
        );
        assert_eq!(read_semaphore(mem.writer_mut(), k.sem.pa), Ok(2));
        assert_eq!(read_c(&mut mem, &k), k.want);
        assert_eq!(q.update_get(mem.writer_mut()), Ok(3));

        // The driver frees only up to launch 1 (setup + launch 1): launch 3
        // goes to the start of the area, before launch 2 (wrap-around);
        // launch 4 would reach launch 2 and is refused.
        assert_eq!(q.retire(k.sem.va, 1), Ok(2));
        assert_eq!(q.push_in_flight(), one);
        q.launch_and_signal(mem.writer_mut(), k.qmd.va, k.sem.va, 3)
            .unwrap();
        let slot = w.ring.pa + 8 * 3;
        assert_eq!(
            mem.writer_mut().read_u32(slot).unwrap(),
            push.va as u32,
            "lansarea 3 la inceputul zonei"
        );
        let words = push_words(&mut mem, &push);
        assert!(
            q.launch_and_signal(mem.writer_mut(), k.qmd.va, k.sem.va, 4)
                .is_err()
        );
        assert_eq!(push_words(&mut mem, &push), words, "lansarea 2 neatinsa");
        // The value must grow on the same semaphore.
        assert!(
            q.launch_and_signal(mem.writer_mut(), k.qmd.va, k.sem.va, 3)
                .is_err()
        );
        reset_c(&mut mem);
        assert_eq!(
            gpu_runs_again(&mut cp, &mut mem, root, &w, &push, &k),
            Ok(1)
        );
        assert_eq!(read_c(&mut mem, &k), k.want);
        let done = read_semaphore(mem.writer_mut(), k.sem.pa).unwrap();
        assert_eq!(done, 3);
        assert_eq!(q.retire(k.sem.va, done), Ok(2));
        assert_eq!(q.push_in_flight(), 0);

        // Steady state: 9 more rounds of launch → GPU → semaphore → retire
        // in the same 128-byte area (12 launches in all, the ring keeps going).
        for v in 4..=12u64 {
            q.launch_and_signal(mem.writer_mut(), k.qmd.va, k.sem.va, v)
                .unwrap();
            reset_c(&mut mem);
            assert_eq!(
                gpu_runs_again(&mut cp, &mut mem, root, &w, &push, &k),
                Ok(1)
            );
            assert_eq!(read_c(&mut mem, &k), k.want, "lansarea {v}");
            let done = read_semaphore(mem.writer_mut(), k.sem.pa).unwrap();
            assert_eq!(done, v);
            assert_eq!(q.retire(k.sem.va, done), Ok(1));
            q.update_get(mem.writer_mut()).unwrap();
        }
        assert_eq!(q.get(), q.put());
        assert_eq!(q.put(), 13);
        // Room again, but the same value on the same semaphore is refused.
        let put = mem.writer_mut().read_u32(put_pa).unwrap();
        assert!(
            q.launch_and_signal(mem.writer_mut(), k.qmd.va, k.sem.va, 12)
                .is_err()
        );
        assert_eq!(mem.writer_mut().read_u32(put_pa).unwrap(), put);
        assert_eq!(
            cp.events
                .iter()
                .filter(|e| matches!(e, Event::Launch { .. }))
                .count(),
            12
        );
    }

    /// What the driver refuses before writing anything.
    #[test]
    fn launch_driver_side_refusals() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let w = open_work_channel_cleared(&mut d, &mut mem, &u, &cg, &g).unwrap();
        let k = add64_buffers(&mut mem);
        let push = mem.alloc_mapped(0x1000, 0x1000).unwrap();
        let mut q = WorkQueue::new(&w, push).unwrap();
        let put_pa = w.ring.pa + w.userd_offset + 0x88;
        let before = mem.writer().0.read_u64(put_pa).unwrap();
        for (qmd, sem) in [
            (k.qmd.va + 0x80, k.sem.va),             // QMD not 256-aligned
            (LOCAL_MEM_WINDOW, k.sem.va),            // QMD outside our VA
            (k.qmd.va, k.sem.va + 4),                // semaphore not 8-aligned
            (k.qmd.va, SHARED_MEM_WINDOW),           // semaphore outside our VA
            (crate::mmu::VA_BASE - 0x100, k.sem.va), // QMD just below our VA
        ] {
            assert!(
                q.launch_and_signal(mem.writer_mut(), qmd, sem, DONE)
                    .is_err(),
                "{qmd:#x} {sem:#x}"
            );
        }
        assert_eq!(mem.writer().0.read_u64(put_pa).unwrap(), before);
        assert_eq!(q.put(), 0);
    }

    /// Mistakes planted in the launch words are caught: by the model (it
    /// reads only what the headers say) or by the driver's checks of the
    /// semaphore and the result.
    #[test]
    fn planted_launch_mistakes_are_caught() {
        let (mut d, mut mem, g, u, cg) = grouped();
        let root = d.0.page_directory(USER_CLIENT, u.vaspace).unwrap().0;
        let k = add64_buffers(&mut mem);
        let good = {
            let mut buf = [0u32; 16];
            let mut pb = PushBuf::new(&mut buf);
            invalidate_caches(&mut pb).unwrap();
            launch(&mut pb, k.qmd.va).unwrap();
            release(&mut pb, k.sem.va, DONE).unwrap();
            pb.words().to_vec()
        };
        // [0..2) invalidate, [2..4) PCAS_A, [4..6) PCAS2_B, [6..12) semaphore.
        assert_eq!(good.len(), 12);
        let mut cases: Vec<(&str, Vec<u32>)> = Vec::new();
        let mut m = good.clone();
        m[1] |= 1 << 8; // a cache bit the model does not know
        cases.push(("bit necunoscut la invalidare", m));
        let mut m = good.clone();
        m[3] += 1; // QMD address one 256-byte step off
        cases.push(("adresa QMD", m));
        let mut m = good.clone();
        m[5] = 0; // PCAS action NOP: nothing launched
        cases.push(("actiunea PCAS", m));
        let mut m = good.clone();
        m.drain(2..4); // no SEND_PCAS_A
        cases.push(("fara SEND_PCAS_A", m));
        let mut m = good.clone();
        m.swap(9, 10); // payload halves swapped
        cases.push(("jumatatile valorii", m));
        let mut m = good.clone();
        // timestamp release (16-byte record): since C-69 the model writes
        // it, so the mistake shows as the 8 bytes after the semaphore
        // overwritten (checked with the sentinel below).
        m[11] |= 1 << 25;
        cases.push(("timestamp", m));
        let mut m = good.clone();
        m[11] &= !(1 << 24); // 32-bit payload: the high half never written
        cases.push(("semafor pe 32 de biti", m));
        let mut m = good.clone();
        m[7] += 4; // semaphore address off by 4
        cases.push(("adresa semaforului", m));
        for (i, (what, words)) in cases.into_iter().enumerate() {
            // A fresh channel and clean output each time.
            let cg2 = ComputeGroup {
                next_handle: 0x600 + 0x10 * i as u32,
                ..cg
            };
            let w2 = open_work_channel_cleared(&mut d, &mut mem, &u, &cg2, &g).unwrap();
            put_bytes(mem.writer_mut(), k.sem.pa, &[0; 8]);
            put_bytes(mem.writer_mut(), k.sem.pa + 8, &[0xa5; 8]);
            put_bytes(mem.writer_mut(), k.c.pa, &[0xff; 256]);
            let push = mem.alloc_mapped(0x1000, 0x1000).unwrap();
            let mut q = WorkQueue::new(&w2, push).unwrap();
            q.submit(mem.writer_mut(), &setup_words()).unwrap();
            q.submit(mem.writer_mut(), &words).unwrap();
            let (_, r) = gpu_launches(&mut mem, root, &w2, &push, &k);
            let after = [0, 4].map(|o| mem.writer_mut().read_u32(k.sem.pa + 8 + o).unwrap());
            let caught = r.is_err()
                || read_semaphore(mem.writer_mut(), k.sem.pa) != Ok(DONE)
                || read_c(&mut mem, &k) != k.want
                || after != [0xa5a5_a5a5; 2];
            assert!(caught, "greseala „{what}” neprinsa");
        }
    }

    // ---- I10c: GR0's shape (SIMULARE) ----

    use crate::gsp::work_queue::{GrShape, local_memory_size};

    /// The shape and SM version come back from the model's answer
    /// (`SIM_GR_INFO`), and feed the local memory size.
    #[test]
    fn gr_info_from_gsp_rm() {
        let (mut d, u) = opened();
        let n = d.0.log.controls.len();
        let info = query_gr_info(&mut d, &u).unwrap();
        assert_eq!(d.0.log.controls[n..], [(c_cmd::KGR_GET_INFO, status::OK)]);
        assert_eq!(
            info,
            GrInfo {
                shape: GrShape {
                    gpcs: 6,
                    tpc_per_gpc: 4,
                    sm_per_tpc: 2,
                    max_warps_per_sm: 48
                },
                sm_version: rmapi::SM_VERSION_SM86
            }
        );
        assert_eq!(info.check_sm86(), Ok(()));
        // 1 B per thread → 32 B → 0x400 per warp → 48 × 2 warps = 0x18000
        // per TPC (rounded to 0x8000) → × 24 TPC = 0x240000 (to 0x20000).
        let lm = local_memory_size(1, info.shape).unwrap().unwrap();
        assert_eq!((lm.per_tpc, lm.total), (0x1_8000, 0x24_0000));
    }

    /// GSP-RM's refusals come back as its status, never as a shape.
    #[test]
    fn rm_refusals_for_gr_info() {
        let (mut d, u) = opened();
        // On the VA space instead of the subdevice.
        let wrong = UserSpace {
            subdevice: u.vaspace,
            ..u
        };
        assert_eq!(
            query_gr_info(&mut d, &wrong),
            Err(GpuError::HardwareFault {
                code: status::INVALID_OBJECT_HANDLE
            })
        );
        // A size other than 570.144's.
        let mut rm = rm_on(&mut d, u.next_handle);
        assert_eq!(
            rm.control(u.subdevice, cmd::KGR_GET_INFO, &[0u8; 464])
                .map(|_| ()),
            Err(GpuError::HardwareFault {
                code: status::INVALID_PARAM_STRUCT
            })
        );
    }

    /// Planted mistakes in GSP-RM's answer: each one is refused.
    #[test]
    fn planted_gr_info_mistakes_are_caught() {
        let (mut d, u) = opened();
        let mut rm = rm_on(&mut d, u.next_handle);
        let good = rm
            .control(u.subdevice, cmd::KGR_GET_INFO, &rmapi::gr_info_request())
            .unwrap();
        assert!(rmapi::parse_gr_info(&good).is_ok());
        let put =
            |b: &mut Vec<u8>, at: usize, v: u32| b[at..at + 4].copy_from_slice(&v.to_le_bytes());
        let mut planted: Vec<Vec<u8>> = Vec::new();
        // 1. Each of the five values zero.
        for i in [12usize, 13, 20, 23, 32] {
            let mut b = good.clone();
            put(&mut b, i * 8 + 4, 0);
            planted.push(b);
        }
        // 2. An entry that names another index than its position.
        let mut b = good.clone();
        put(&mut b, 20 * 8, 23);
        planted.push(b);
        // 3. One entry short, one byte long.
        planted.push(good[..good.len() - 8].to_vec());
        let mut b = good.clone();
        b.push(0);
        planted.push(b);
        for (n, b) in planted.iter().enumerate() {
            assert!(rmapi::parse_gr_info(b).is_err(), "greseala {n} neprinsa");
        }
        // 4. GR1's entries (engine 1) are not read: a shape there changes nothing.
        let mut b = good.clone();
        put(&mut b, 58 * 8 + 20 * 8 + 4, 99);
        assert_eq!(rmapi::parse_gr_info(&b), rmapi::parse_gr_info(&good));
        // 5. Another SM version is read as is, and refused for our programs.
        let mut b = good.clone();
        put(&mut b, 12 * 8 + 4, 0x807);
        let other = rmapi::parse_gr_info(&b).unwrap();
        assert_eq!(other.sm_version, 0x807);
        assert!(other.check_sm86().is_err());
        // 6. index = 0 (if GSP-RM does not fill it) is accepted.
        let mut b = good.clone();
        put(&mut b, 20 * 8, 0);
        assert_eq!(rmapi::parse_gr_info(&b), rmapi::parse_gr_info(&good));
    }

    // ---- C-71: the whole window chain, I6 → first computation (SIMULARE) ----

    /// What one simulated window left behind.
    struct FullWindow {
        report: crate::chain::ChainReport,
        command_after: u16,
        gsp_doorbells: u32,
        rpc_bells: u32,
        writes: Vec<(u32, u32)>,
        median_ns: Option<u64>,
        /// C-71: the whole journal as the OS would save it on the stick.
        journal_text: alloc::string::String,
        frts: (u64, u64),
        wpr_meta: u64,
    }

    /// `gpu lant` with [`crate::chain::WINDOW_CHAIN_FULL`] on the simulated
    /// board: Bus Master on the observed config space, I6/I7 on `FakeGsp`,
    /// I8 and every RM call through the real queues to `SimGsp` + `FakeRm`,
    /// the work channel on our page tables in a simulated VRAM, the launches
    /// on the command processor model. The simulated GSP answers a command
    /// only after a doorbell (as the board). `bell_each_rpc = false` uses the
    /// plain `QueueTransport`, which never rings.
    fn full_window(bell_each_rpc: bool) -> FullWindow {
        full_window_with(bell_each_rpc, true)
    }

    /// `libos_order = false`: I7 with the old [`crate::gsp::falcon::booter_load_dma`]
    /// (no GSP reset, no LibOS mailboxes), as the OS ran it on 5 Nov.
    fn full_window_with(bell_each_rpc: bool, libos_order: bool) -> FullWindow {
        use crate::chain::{READY_FOR_WINDOW, WINDOW_CHAIN_FULL, run_chain_ready};
        use crate::gsp::falcon::{self, GuardedBar0};
        use crate::gsp::golden::QueueTransportBell;
        use crate::gsp::rpc::{event, function};
        use crate::launch_cost;
        use core::cell::Cell;
        use gpu_core::diag::Journal;
        use gpu_core::hal::ConfigSpace;
        use gpu_pci::bus_master::{BusMasterPermit, restore_command, set_bus_master};
        use gpu_sim::observed;

        const WPR_META: u64 = 0x1_2340_0000;
        const LIBOS: u64 = 0x0000_0002_0000_0000;
        let mut cfg = observed::observed_config();
        let permit = BusMasterPermit::new("test C-71 (SIMULARE)");
        let mut old = None;
        let (mut g, [(fw_u, fw_img), (bo_u, bo_img)]) =
            crate::gsp::hs_load::tests::i6_i7_dma_board();
        let frts = g.frts;
        let mut clock = SimClock::default();
        let l = QueueLayout::standard();
        let mut shm = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let mut q = CpuQueues::init(&mut shm, l, &pages).unwrap();
        let mut gsp: Option<SimGsp> = None;
        let mut rm = FakeRm::new();
        let vram = 8u64 << 30;
        let mut mem: Mem =
            GpuMemory::new(64 << 20, vram - (256 << 20), Sim(SimVram::new(vram))).unwrap();
        let bells = Cell::new(0u32);
        let served = Cell::new(0u32);
        let mut golden = None;
        let mut space = None;
        let mut launch = None;
        let mut median = None;
        let mut j = Journal::new(1 << 16);
        let fail = |w: &'static str| GpuError::InvalidArgument(w);
        let report = run_chain_ready(WINDOW_CHAIN_FULL, READY_FOR_WINDOW, true, &mut j, |s, j| {
            // One RM call: the doorbell after each message (or never), the
            // simulated GSP answering only after a doorbell it has not served.
            macro_rules! with_rm {
                ($g:expr, |$t:ident| $body:expr) => {{
                    let gsp = gsp.as_mut().ok_or(fail("GSP neinitializat"))?;
                    let mut b = GuardedBar0::new($g, j, None)?;
                    let poll = |m: &mut dyn SharedMem| {
                        if bells.get() > served.get() {
                            served.set(bells.get());
                            gsp.service_with(m, |f, p| rm.handle(f, p));
                        }
                    };
                    if bell_each_rpc {
                        let mut $t = QueueTransportBell {
                            q: &mut q,
                            mem: &mut shm,
                            clock: &mut clock,
                            timeout_us: RM_REPLY_US,
                            poll,
                            events: Vec::new(),
                            bell: || {
                                falcon::ring_doorbell(&mut b)?;
                                bells.set(bells.get() + 1);
                                Ok(())
                            },
                            rung: 0,
                        };
                        $body
                    } else {
                        let mut $t = QueueTransport {
                            q: &mut q,
                            mem: &mut shm,
                            clock: &mut clock,
                            timeout_us: RM_REPLY_US,
                            poll,
                            events: Vec::new(),
                        };
                        $body
                    }
                }};
            }
            match s.id {
                "pregatire-i6" => {
                    let mut b = GuardedBar0::new(&mut g, j, None)?;
                    match falcon::read_wpr2(&mut b)? {
                        None => Ok(()),
                        Some(_) => Err(fail("WPR2 deja ridicat")),
                    }
                }
                "bus-master-pornit" => {
                    old = Some(set_bus_master(&mut cfg, &permit, j, true)?);
                    Ok(())
                }
                "i6-fwsec-frts" => {
                    let mut b = GuardedBar0::new(&mut g, j, None)?;
                    falcon::fwsec_frts_dma(&mut b, &mut clock, &fw_u, &fw_img, frts).map(|_| ())
                }
                "i6-wpr2-citit" => falcon::read_frts_after_i6(&mut g, j, frts).map(|_| ()),
                "i7-booter-gsp" => {
                    let mut b = GuardedBar0::new(&mut g, j, None)?;
                    if libos_order {
                        falcon::booter_load_dma_libos(
                            &mut b, &mut clock, &bo_u, &bo_img, WPR_META, LIBOS,
                        )
                    } else {
                        falcon::booter_load_dma(&mut b, &mut clock, &bo_u, &bo_img, WPR_META)
                    }
                }
                "i7-citit" => match falcon::read_after_i7(&mut g, j)?.passed() {
                    true => Ok(()),
                    false => Err(fail("I7: Booter refuzat sau RISC-V oprit")),
                },
                "i8-gsp-init" => {
                    let sys =
                        crate::gsp::sysinfo::encode(&crate::gsp::sysinfo::tests::workstation())?;
                    q.send_rpc(&mut shm, function::GSP_SET_SYSTEM_INFO, &sys)?;
                    let reg = crate::gsp::registry::encode(&crate::gsp::registry::boot_entries())?;
                    q.send_rpc(&mut shm, function::SET_REGISTRY, &reg)?;
                    let mut b = GuardedBar0::new(&mut g, j, None)?;
                    falcon::ring_doorbell(&mut b)?;
                    // C-75: GSP-RM finds its queues only through the LibOS
                    // address it had when Booter started it (board, 5 Nov).
                    if g.libos_at_booter != Some(LIBOS) {
                        return Err(fail("cozi GSP: GSP-RM pornit fara adresa LibOS"));
                    }
                    let mut b = GuardedBar0::new(&mut g, j, None)?;
                    let mut peer = SimGsp::start(&mut shm, l.cmd_queue_offset, l.stat_queue_offset);
                    q.link_rx(&shm)?;
                    let mut step = 0;
                    let ctx = crate::gsp::sequencer::SeqContext {
                        libos_addr: LIBOS,
                        app_version: 570_144,
                    };
                    crate::gsp::sequencer::wait_init_done(
                        &mut q,
                        &mut shm,
                        &mut b,
                        &mut clock,
                        ctx,
                        crate::gsp::sequencer::INIT_DONE_US,
                        |m| {
                            step += 1;
                            match step {
                                1 => peer.service(m),
                                2 => peer.post(m, event::GSP_INIT_DONE, 0, &[]),
                                _ => {}
                            }
                        },
                    )?;
                    gsp = Some(peer);
                    Ok(())
                }
                "i8e-golden" => {
                    let gi = with_rm!(&mut g, |t| init_golden_image(&mut t, &mut mem))?;
                    golden = Some(gi);
                    Ok(())
                }
                "i9-canal-lucru" => {
                    let gi = golden.as_ref().ok_or(fail("fara golden image"))?;
                    let (u, w) = with_rm!(&mut g, |t| {
                        let u = open_user_space(&mut t, gi.next_handle(), mem.root())?;
                        let cg = open_compute_group(&mut t, &u)?;
                        let w = open_work_channel_cleared(&mut t, &mut mem, &u, &cg, gi)?;
                        query_gr_info(&mut t, &u)?.check_sm86()?;
                        Ok::<_, GpuError>((u, w))
                    })?;
                    space = Some((u, w));
                    Ok(())
                }
                "i10-add64" => {
                    let (u, w) = space.as_ref().ok_or(fail("fara canal de lucru"))?;
                    let root = rm
                        .page_directory(USER_CLIENT, u.vaspace)
                        .ok_or(fail("RM nu are tabelele"))?
                        .0;
                    let k = add64_buffers(&mut mem);
                    let push = mem.alloc_mapped(0x1000, 0x1000)?;
                    let mut wq = WorkQueue::new(w, push)?;
                    wq.submit(mem.writer_mut(), &setup_words())?;
                    wq.launch_and_signal(mem.writer_mut(), k.qmd.va, k.sem.va, DONE)?;
                    let (cp, r) = gpu_launches(&mut mem, root, w, &push, &k);
                    r?;
                    if read_semaphore(mem.writer_mut(), k.sem.pa)? != DONE
                        || read_c(&mut mem, &k) != k.want
                    {
                        return Err(fail("add64: rezultat diferit de procesor"));
                    }
                    wq.update_get(mem.writer_mut())?;
                    launch = Some((cp, root, push, k, wq));
                    Ok(())
                }
                "i10-cost-lansare" => {
                    let (cp, root, push, k, wq) = launch.as_mut().ok_or(fail("fara add64"))?;
                    let (_, w) = space.as_ref().unwrap();
                    let rec = mem.alloc_mapped(0x1000, 0x1000)?;
                    put_bytes(mem.writer_mut(), rec.pa, &[0; 17 * 16]);
                    let mut words = alloc::vec![0u32; 256];
                    let mut pb = PushBuf::new(&mut words);
                    launch_cost::encode(&mut pb, k.qmd.va, rec.va, 16, 0x500)?;
                    let n = pb.len();
                    wq.submit(mem.writer_mut(), &words[..n])?;
                    // The GPU's side (SIMULARE clock: 4000 ns per launch).
                    cp.sim_launch_ns = 4000;
                    cp.mem
                        .load_from_vram(&mem.writer().0, *root, rec.va, rec.size)?;
                    gpu_runs_again(cp, &mut mem, *root, w, push, k)?;
                    cp.mem
                        .store_to_vram(&mut mem.writer_mut().0, *root, rec.va, 17 * 16)?;
                    let mut bytes = Vec::new();
                    for o in (0..17 * 16).step_by(4) {
                        bytes.extend_from_slice(
                            &mem.writer_mut().read_u32(rec.pa + o)?.to_le_bytes(),
                        );
                    }
                    median = Some(launch_cost::analyze(&bytes, 16, 0x500)?.median_ns);
                    Ok(())
                }
                "bus-master-oprit" => match old {
                    Some(o) => restore_command(&mut cfg, &permit, j, o),
                    None => Ok(()),
                },
                "tur-dupa" => Ok(()),
                _ => Err(GpuError::NotPermitted("pas fara implementare in test")),
            }
        });
        FullWindow {
            report,
            command_after: cfg.read32(0x04) as u16,
            gsp_doorbells: g.doorbells,
            rpc_bells: bells.get(),
            writes: g.writes.clone(),
            median_ns: median,
            journal_text: j.events().map(|e| alloc::format!("{e}\n")).collect(),
            frts,
            wpr_meta: WPR_META,
        }
    }

    /// C-71 (T78): every step of the full window chain passes on the
    /// simulated board, with the doorbell after every RM call.
    #[test]
    fn full_window_chain_passes_every_step_simulated() {
        let w = full_window(true);
        assert!(w.report.all_passed(), "{}", w.report);
        assert_eq!(w.report.steps.len(), crate::chain::WINDOW_CHAIN_FULL.len());
        // One doorbell for I8, then one per RM call (golden + I9 + GR info).
        assert!(w.rpc_bells > 10, "{}", w.rpc_bells);
        assert_eq!(w.gsp_doorbells, 1 + w.rpc_bells);
        // All register writes in the GSP/SEC2 windows.
        assert!(w.writes.iter().all(|&(o, _)| {
            (0x110000..=0x113fff).contains(&o) || (0x840000..=0x843fff).contains(&o)
        }));
        assert_eq!(w.median_ns, Some(4000));
        assert_eq!(w.command_after, gpu_sim::observed::COMMAND);
    }

    /// C-71 (T79): the plain transport never rings; a GSP that waits for
    /// the doorbell never answers → the chain stops at the golden image
    /// (timeout), nothing after it runs, Bus Master is put back.
    #[test]
    fn without_a_doorbell_per_rpc_the_chain_stops_at_the_golden_image() {
        use crate::chain::Outcome;
        let w = full_window(false);
        let (at, why) = w.report.stopped_at.clone().unwrap();
        assert_eq!(at, "i8e-golden", "{why}");
        assert!(why.starts_with("timeout"), "{why}");
        assert_eq!(
            w.report.outcome("i10-add64"),
            Some(&Outcome::Skipped("lantul s-a oprit inainte"))
        );
        assert_eq!(w.report.outcome("bus-master-oprit"), Some(&Outcome::Passed));
        assert_eq!(w.command_after, gpu_sim::observed::COMMAND);
        assert_eq!(w.gsp_doorbells, 1);
    }

    /// C-75 (T86): I7 in the old order (as `os-246` on 5 Nov: no GSP reset
    /// after FWSEC, the LibOS mailboxes written and not kept) gives on the
    /// simulated board what the real board gave: I7 passes (Booter 0, RISC-V
    /// active), I8 stops at the GSP queues, Bus Master is put back; the
    /// journal check names the cause at the first I7 write. In the new order
    /// ([`crate::gsp::falcon::booter_load_dma_libos`]) the same chain passes
    /// and the journal shows the LibOS address kept. SIMULARE.
    #[test]
    fn old_i7_order_reproduces_the_board_and_the_libos_order_passes() {
        use crate::chain::Outcome;
        use crate::chain_log::{ChainVerdict, check_chain_journal};
        let w = full_window_with(true, false);
        assert_eq!(w.report.outcome("i7-booter-gsp"), Some(&Outcome::Passed));
        assert_eq!(w.report.outcome("i7-citit"), Some(&Outcome::Passed));
        let (at, why) = w.report.stopped_at.clone().unwrap();
        assert_eq!(at, "i8-gsp-init", "{why}");
        assert!(why.contains("LibOS"), "{why}");
        assert_eq!(w.report.outcome("bus-master-oprit"), Some(&Outcome::Passed));
        assert_eq!(w.command_after, gpu_sim::observed::COMMAND);
        // The old function writes no GSP mailbox: the chain_log rule for
        // the old form (SEC2 only) still applies to it.
        let r = check_chain_journal(&w.journal_text, w.frts, w.wpr_meta, None);
        assert_eq!(r.i7_libos, None);

        let n = full_window_with(true, true);
        assert!(n.report.all_passed(), "{}", n.report);
        let r = check_chain_journal(&n.journal_text, n.frts, n.wpr_meta, Some(n.rpc_bells));
        assert_eq!(r.verdict(), ChainVerdict::Passed, "{r:?}");
        let s = r.i7_libos.unwrap();
        assert_eq!(s.libos, 0x0000_0002_0000_0000);
        assert_eq!(r.i7.as_ref(), Some(&s.booter));
        // 5 GSP writes before Booter + 1 after.
        assert_eq!(s.writes, s.booter.writes + 6);
        // Planted mistakes in the new I7 part are caught.
        let bad = |from: &str, to: &str| {
            let t = n.journal_text.replacen(from, to, 1);
            assert_ne!(t, n.journal_text, "{from}");
            match check_chain_journal(&t, n.frts, n.wpr_meta, Some(n.rpc_bells)).verdict() {
                ChainVerdict::ToAnalyse(why) => why,
                v => panic!("{from}: {v:?}"),
            }
        };
        let why = bad("<- 0x0008b320 => 0x0008b320", "<- 0x0008b321 => 0x0008b321");
        assert!(why.contains("FALCON_OS"), "{why}");
        let why = bad(
            "0x110044: 0x00000000 <- 0x00000002 => 0x00000002",
            "0x110044: 0x00000000 <- 0x00000002 => 0x00000000",
        );
        assert!(why.contains("LibOS"), "{why}");
    }

    /// C-71 (T81): the stick journal of the simulated full window, read
    /// back with `chain_log`: every step conform, one GSP doorbell per RM
    /// call; planted mistakes in the text are all caught. SIMULARE.
    #[test]
    fn full_window_journal_checks_step_by_step() {
        use crate::chain_log::{ChainVerdict, check_chain_journal};
        let w = full_window(true);
        assert!(w.report.all_passed(), "{}", w.report);
        let check = |t: &str, n: Option<u32>| check_chain_journal(t, w.frts, w.wpr_meta, n);
        let r = check(&w.journal_text, Some(w.rpc_bells));
        assert_eq!(r.verdict(), ChainVerdict::Passed, "{r:?}");
        assert_eq!(r.steps.len(), crate::chain::WINDOW_CHAIN_FULL.len());
        assert_eq!(r.rm_doorbells, w.rpc_bells);
        assert_eq!(r.step("i8-gsp-init").unwrap().gsp_doorbells, 1);
        assert!(r.i6.is_some() && r.i7.is_some());
        let lines: Vec<&str> = w.journal_text.lines().collect();
        let at = |id: &str| {
            lines
                .iter()
                .position(|l| *l == alloc::format!("pas {id}: ok"))
                .unwrap()
        };
        let edit = |f: &dyn Fn(&mut Vec<alloc::string::String>)| {
            let mut v: Vec<alloc::string::String> = lines.iter().map(|l| (*l).into()).collect();
            f(&mut v);
            v.join("\n")
        };
        let bad = |t: &str, n: Option<u32>, want: &str| match check(t, n).verdict() {
            ChainVerdict::ToAnalyse(why) => assert!(why.contains(want), "{why}"),
            v => panic!("{want}: {v:?}"),
        };
        // 1. one doorbell fewer than the RM calls the OS printed.
        bad(&w.journal_text, Some(w.rpc_bells + 1), "una pe mesaj");
        let golden = at("i8e-golden");
        let gb = (0..golden)
            .rev()
            .find(|&i| lines[i].starts_with("bar0 w 0x110c00"))
            .unwrap();
        bad(
            &edit(&|v| {
                v.remove(gb);
            }),
            Some(w.rpc_bells),
            "una pe mesaj",
        );
        // 2. a register write in golden that is not a doorbell.
        bad(
            &edit(&|v| v.insert(golden, "bar0 w 0x001200: 0x0 <- 0x1 => 0x1".into())),
            None,
            "i8e-golden: scriere care nu e sonerie",
        );
        // 3. Bus Master restored to a different value.
        let off = at("bus-master-oprit") - 1;
        bad(
            &edit(&|v| v[off] = "cfg  w 0x004: 0x00100007 <- 0x00100007 => 0x00100007".into()),
            None,
            "bus-master-oprit: Command pus",
        );
        // 4. a read-only step that wrote.
        let r7 = at("i7-citit");
        bad(
            &edit(&|v| v.insert(r7, "bar0 w 0x110040: 0x0 <- 0x1 => 0x1".into())),
            None,
            "i7-citit: pas doar-citire",
        );
        // 5. I8 that writes outside the GSP/SEC2 windows.
        let i8 = at("i8-gsp-init");
        bad(
            &edit(&|v| v.insert(i8, "bar0 w 0x000200: 0x0 <- 0x1 => 0x1".into())),
            None,
            "afara ferestrelor GSP/SEC2",
        );
        // 6. steps out of order.
        let (a, b) = (at("i9-canal-lucru"), at("i10-add64"));
        bad(&edit(&|v| v.swap(a, b)), None, "alta ordine");
        // 8. (T85) on the board I9 writes our page tables through the VRAM
        //    window: the real `VramWindow` journal (SIMULARE PRAMIN model)
        //    put into the I9 segment keeps the chain conform; a window not
        //    put back, or one that did not move, is caught.
        {
            use crate::mmu::{GpuMemory, PageAttr};
            use crate::vram_window::VramWindow;
            use gpu_core::diag::Journal;
            use gpu_sim::pramin::SimBar0Vram;
            let vram = 8u64 << 30;
            let mut bar = SimBar0Vram::new(SimVram::new(vram), 0x0000_0012);
            let mut wj = Journal::new(256);
            let mut clock = SimClock::default();
            let win = VramWindow::new(&mut bar, &mut wj, None, vram, &[(0, 16 << 20)]).unwrap();
            let mut m = GpuMemory::new(64 << 20, vram - (256 << 20), win).unwrap();
            m.map(0x10_0000_0000, 0x1_2340_0000, 3 << 20, PageAttr::default())
                .unwrap();
            let root = m.root();
            m.writer_mut().invalidate_tlb(&mut clock, root).unwrap();
            m.writer_mut().restore().unwrap();
            drop(m);
            let wl: Vec<alloc::string::String> =
                wj.events().map(|e| alloc::format!("{e}")).collect();
            let moves: Vec<usize> = wl
                .iter()
                .enumerate()
                .filter(|(_, l)| l.starts_with("bar0 w 0x001700"))
                .map(|(i, _)| i)
                .collect();
            assert!(moves.len() >= 2, "{wl:?}");
            let with = |f: &dyn Fn(&mut Vec<alloc::string::String>)| {
                let mut wl = wl.clone();
                f(&mut wl);
                let i9 = at("i9-canal-lucru");
                edit(&|v| {
                    for (k, l) in wl.iter().enumerate() {
                        v.insert(i9 + k, l.clone());
                    }
                })
            };
            let good = with(&|_| {});
            let r = check(&good, Some(w.rpc_bells));
            assert_eq!(r.verdict(), ChainVerdict::Passed, "{r:?}");
            let last = *moves.last().unwrap();
            bad(
                &with(&|v| {
                    v.remove(last);
                }),
                None,
                "fereastra VRAM (0x1700) lasata la",
            );
            let first = moves[0];
            bad(
                &with(&|v| {
                    let l = v[first].clone();
                    let (head, _) = l.rsplit_once("=> ").unwrap();
                    v[first] = alloc::format!("{head}=> 0x00000000");
                }),
                None,
                "nu s-a mutat",
            );
        }
        // 7. without a doorbell per RM call the chain stops at golden, and
        //    the journal shows a clean stop with Bus Master put back.
        let n = full_window(false);
        let r = check_chain_journal(&n.journal_text, n.frts, n.wpr_meta, None);
        assert_eq!(
            r.verdict(),
            ChainVerdict::StoppedCleanly("i8e-golden".into()),
            "{r:?}"
        );
    }
}
