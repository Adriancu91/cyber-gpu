//! I8e — the "golden image": the first RM client the driver creates right
//! after `GSP_INIT_DONE`, whose channel gets a promoted GR context that GSP-RM
//! keeps as the template ("golden" context) for every later compute channel.
//! Built and tested **only in SIMULARE** (against `gpu_sim::fake_rm` behind
//! the real queue code).
//!
//! Order (tinygrad `ip.py` `init_golden_image`, MIT, commit 9cf7829a6, read as
//! the specification and re-written; DEDUS where tinygrad itself is empirical):
//! 1. client (`NV01_ROOT`), device (`NV01_DEVICE_0`), subdevice
//!    (`NV20_SUBDEVICE_0`), VA space (`FERMI_VASPACE_A`);
//! 2. `FIFO_GET_DEVICE_INFO_TABLE` → which runlist serves which engine;
//! 3. `VASPACE_COPY_SERVER_RESERVED_PDES`: hand GSP-RM the page-table levels
//!    of a 512 MiB VA range the driver reserves for RM;
//! 4. the channel (`AMPERE_CHANNEL_GPFIFO_A`, 32 entries, GR0);
//! 5. `KGR_GET_CONTEXT_BUFFERS_INFO` → sizes/alignments of the GR context
//!    buffers; allocate them; `GPU_PROMOTE_CTX` (9 buffers, PM stays local);
//! 6. the compute object (`AMPERE_COMPUTE_B`) and the copy object
//!    (`AMPERE_DMA_COPY_B`) on the channel.
//!
//! The memory (VRAM + GPU virtual addresses + page tables) is the driver's
//! job and comes through [`GoldenMemory`]; on the board it will be the I9
//! memory manager, in the tests a bump allocator. The RPC transport comes
//! through [`RmTransport`]; [`QueueTransport`] is the real one over the
//! command/status queues. Nothing here writes a register.

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::Clock;

use super::msgq::{CpuQueues, SharedMem};
use super::rmapi::{self, ChannelSetup, PdeLevel, PromoteEntry, class, cmd};
use super::rpc::function;

/// The internal client handle tinygrad uses (`priv_root`). DEDUS.
pub const CLIENT: u32 = 0xc1e0_0004;
/// First handle for the objects under it (tinygrad `handle_gen`). DEDUS.
pub const FIRST_HANDLE: u32 = 0xcf00_0000;
/// VA range reserved for GSP-RM's own mappings (tinygrad: 512 MiB).
pub const RESERVED_VA_SIZE: u64 = 512 << 20;
/// tinygrad adds 256 KiB to the main GR image size. DEDUS (no reason given).
pub const GR_MAIN_EXTRA: u64 = 0x4_0000;
/// The attribute circular buffer is aligned to 2 MiB (tinygrad). DEDUS.
pub const ATTRIBUTE_CB_ALIGN: u64 = 2 << 20;
/// `physAttr` tinygrad sets on the physical half of a promoted buffer. DEDUS.
pub const PROMOTE_PHYS_ATTR: u32 = 0x4;

/// One RPC to GSP-RM and its reply payload.
pub trait RmTransport {
    fn call(&mut self, function: u32, payload: &[u8]) -> GpuResult<Vec<u8>>;
}

/// A piece of VRAM the driver allocated: physical address in VRAM and, if
/// mapped, the GPU virtual address in the golden VA space.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mapping {
    pub va: u64,
    pub pa: u64,
    pub size: u64,
}

/// What the golden image needs from the driver's memory manager.
pub trait GoldenMemory {
    /// Contiguous VRAM, `align`-aligned, mapped at a GPU VA.
    fn alloc_mapped(&mut self, size: u64, align: u64) -> GpuResult<Mapping>;
    /// Contiguous VRAM without a mapping (method buffer, instance block).
    fn alloc_phys(&mut self, size: u64, align: u64) -> GpuResult<u64>;
    /// Reserves `size` bytes of GPU VA and returns its start plus the page
    /// table levels that cover it (root first).
    fn reserve_va(&mut self, size: u64) -> GpuResult<(u64, Vec<PdeLevel>)>;
}

/// One GR context buffer as the golden image allocates and promotes it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrBuf {
    /// `NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_*`.
    pub id: u16,
    pub size: u64,
    pub align: u64,
    pub virt: bool,
    pub phys: bool,
    /// Kept by the driver, not promoted (the PM buffer).
    pub local: bool,
}

/// `NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_*` → `ENGINE_ID_*` (generated
/// constants; the mapping is tinygrad's: ids 3..=10 ↔ engine ids 17..=24).
fn engine_id_of(buffer: u16) -> usize {
    use nvidia_rm570::rmapi as r;
    match buffer as u32 {
        r::NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_MAIN => {
            r::NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS as usize
        }
        r::NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PM
        | r::NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_PATCH => {
            r::NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PATCH as usize
        }
        // UNRESTRICTED_PRIV_ACCESS_MAP reuses PRIV_ACCESS_MAP's size (tinygrad note).
        r::NV2080_CTRL_GPU_PROMOTE_CTX_BUFFER_ID_UNRESTRICTED_PRIV_ACCESS_MAP => {
            r::NV0080_CTRL_FIFO_GET_ENGINE_CONTEXT_PROPERTIES_ENGINE_ID_GRAPHICS_PRIV_ACCESS_MAP
                as usize
        }
        b => b as usize + 14,
    }
}

/// The GR buffer plan from the sizes GSP-RM reported (tinygrad's table).
pub fn gr_buffer_plan(info: &[(u32, u32); rmapi::CTX_ENGINE_IDS]) -> GpuResult<Vec<GrBuf>> {
    // (id, virt, phys, local)
    const PLAN: [(u16, bool, bool, bool); 10] = [
        (0, true, true, false),
        (1, true, true, true),
        (2, true, true, false),
        (3, true, false, false),
        (4, true, false, false),
        (5, true, false, false),
        (6, true, false, false),
        (9, true, true, false),
        (10, false, true, false),
        (11, true, true, false),
    ];
    PLAN.iter()
        .map(|&(id, virt, phys, local)| {
            let (size, align) = info[engine_id_of(id)];
            let mut align = align as u64;
            if id == 5 {
                align = ATTRIBUTE_CB_ALIGN;
            }
            if size == 0 || align == 0 || !align.is_power_of_two() {
                return Err(GpuError::InvalidArgument(
                    "GSP-RM a raportat un buffer de context gol sau nealiniat",
                ));
            }
            let extra = if id == 0 { GR_MAIN_EXTRA } else { 0 };
            Ok(GrBuf {
                id,
                size: (size as u64 + extra).next_multiple_of(align),
                align,
                virt,
                phys,
                local,
            })
        })
        .collect()
}

/// Handles and facts the golden image produced (kept for I9).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GoldenImage {
    pub client: u32,
    pub device: u32,
    pub subdevice: u32,
    pub vaspace: u32,
    pub channel: u32,
    pub compute: u32,
    pub dma: u32,
    /// (RM engine type, runlist).
    pub runlists: Vec<(u32, u32)>,
    pub gr_bufs: Vec<(GrBuf, Mapping)>,
    pub reserved_va: u64,
}

impl GoldenImage {
    /// The first object handle not used by the golden image (handles come
    /// from one counter, as in tinygrad's `handle_gen`); the next client
    /// (I9c) starts there.
    pub fn next_handle(&self) -> u32 {
        [
            self.device,
            self.subdevice,
            self.vaspace,
            self.channel,
            self.compute,
            self.dma,
        ]
        .into_iter()
        .max()
        .unwrap_or(FIRST_HANDLE)
            + 1
    }
}

/// One RM client's allocations and controls, handles from a counter.
pub(crate) struct Rm<'a, T: RmTransport + ?Sized> {
    pub(crate) t: &'a mut T,
    pub(crate) client: u32,
    pub(crate) next: u32,
}

impl<T: RmTransport + ?Sized> Rm<'_, T> {
    pub(crate) fn alloc(&mut self, parent: u32, class: u32, params: &[u8]) -> GpuResult<u32> {
        let h = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or(GpuError::InvalidArgument("RM: handle-urile s-au terminat"))?;
        let sent = rmapi::encode_alloc(self.client, parent, h, class, params);
        let reply = self.t.call(function::GSP_RM_ALLOC, &sent)?;
        rmapi::check_alloc_reply(&sent, &reply)?;
        Ok(h)
    }

    pub(crate) fn control(
        &mut self,
        object: u32,
        command: u32,
        params: &[u8],
    ) -> GpuResult<Vec<u8>> {
        let sent = rmapi::encode_control(self.client, object, command, params);
        let reply = self.t.call(function::GSP_RM_CONTROL, &sent)?;
        rmapi::check_control_reply(&sent, &reply)
    }
}

/// Runs the whole golden-image sequence. Stops at the first RM error (the
/// error carries GSP-RM's status code); never reports success for a step
/// GSP-RM refused.
pub fn init_golden_image<T: RmTransport + ?Sized, M: GoldenMemory + ?Sized>(
    t: &mut T,
    mem: &mut M,
) -> GpuResult<GoldenImage> {
    let mut rm = Rm {
        t,
        client: CLIENT,
        next: FIRST_HANDLE,
    };
    // 1. Objects. The root allocation names the client in hClient; GSP-RM
    // does not use hObject for it (tinygrad passes a fresh handle). DEDUS.
    rm.alloc(0, class::ROOT, &rmapi::root_params())?;
    let device = rm.alloc(CLIENT, class::DEVICE, &rmapi::device_params(CLIENT))?;
    let subdevice = rm.alloc(device, class::SUBDEVICE, &rmapi::subdevice_params())?;
    let vaspace = rm.alloc(device, class::VASPACE, &rmapi::vaspace_params())?;

    // 2. Runlists.
    let di = rm.control(
        subdevice,
        cmd::GET_DEVICE_INFO_TABLE,
        &rmapi::device_info_request(),
    )?;
    let runlists = rmapi::parse_device_info(&di)?;
    if !runlists.iter().any(|&(e, _)| e == rmapi::ENGINE_GR0) {
        return Err(GpuError::InvalidArgument(
            "GSP-RM nu raporteaza motorul GR0",
        ));
    }

    // 3. Reserved PDEs.
    let (reserved_va, levels) = mem.reserve_va(RESERVED_VA_SIZE)?;
    rm.control(
        vaspace,
        cmd::COPY_SERVER_RESERVED_PDES,
        &rmapi::reserved_pdes_params(reserved_va, RESERVED_VA_SIZE, &levels)?,
    )?;

    // 4. The channel.
    let gpfifo = mem.alloc_mapped(0x1000, 0x1000)?;
    let inst = mem.alloc_phys(rmapi::golden_channel::INSTANCE_SIZE, 0x1000)?;
    let mthd = mem.alloc_phys(rmapi::golden_channel::METHOD_BUFFER_SIZE, 0x1000)?;
    let setup = ChannelSetup::golden(gpfifo.va, gpfifo.pa, vaspace, inst, mthd);
    let channel = rm.alloc(
        device,
        class::CHANNEL_GPFIFO,
        &rmapi::channel_params(&setup)?,
    )?;

    // 5. GR context buffers, then promote everything that is not local.
    let info = rm.control(
        subdevice,
        cmd::KGR_GET_CONTEXT_BUFFERS_INFO,
        &rmapi::ctx_buffers_info_request(),
    )?;
    let plan = gr_buffer_plan(&rmapi::parse_ctx_buffers_info(&info)?)?;
    let mut gr_bufs = Vec::with_capacity(plan.len());
    let mut entries = Vec::new();
    for b in plan {
        let m = mem.alloc_mapped(b.size, b.align)?;
        if !b.local {
            entries.push(PromoteEntry {
                buffer_id: b.id,
                gpu_va: if b.virt { m.va } else { 0 },
                gpu_pa: if b.phys { m.pa } else { 0 },
                size: if b.phys { b.size } else { 0 },
                phys_attr: if b.phys { PROMOTE_PHYS_ATTR } else { 0 },
                initialize: b.phys,
                nonmapped: b.phys && !b.virt,
            });
        }
        gr_bufs.push((b, m));
    }
    rm.control(
        subdevice,
        cmd::PROMOTE_CTX,
        &rmapi::promote_params(rmapi::ENGINE_GR0, CLIENT, channel, &entries)?,
    )?;

    // 6. Compute and copy objects (no parameters).
    let compute = rm.alloc(channel, class::COMPUTE, &[])?;
    let dma = rm.alloc(channel, class::DMA_COPY, &[])?;

    Ok(GoldenImage {
        client: CLIENT,
        device,
        subdevice,
        vaspace,
        channel,
        compute,
        dma,
        runlists,
        gr_bufs,
        reserved_va,
    })
}

/// The real transport: send on the command queue, wait on the status queue
/// for the reply with the same function. GSP-RM's own events on the way
/// (log prints, error logs) are kept in `events`, not mistaken for replies.
pub struct QueueTransport<'a, C: Clock + ?Sized, P: FnMut(&mut dyn SharedMem)> {
    pub q: &'a mut CpuQueues,
    pub mem: &'a mut dyn SharedMem,
    pub clock: &'a mut C,
    pub timeout_us: u64,
    /// Lets a simulation run its side between two empty reads; nothing on hardware.
    pub poll: P,
    pub events: Vec<u32>,
}

/// Bound for one RM reply (DEDUS; GSP-RM answers allocations in ms).
pub const RM_REPLY_US: u64 = 5_000_000;

impl<C: Clock + ?Sized, P: FnMut(&mut dyn SharedMem)> RmTransport for QueueTransport<'_, C, P> {
    fn call(&mut self, func: u32, payload: &[u8]) -> GpuResult<Vec<u8>> {
        self.q.send_rpc(self.mem, func, payload)?;
        let start = self.clock.now_us();
        loop {
            match self.q.recv_rpc(self.mem)? {
                Some(m) if m.function == func => {
                    if m.rpc_result != 0 {
                        return Err(GpuError::HardwareFault { code: m.rpc_result });
                    }
                    return Ok(m.payload);
                }
                Some(m) if m.is_event() => self.events.push(m.function),
                Some(_) => {
                    return Err(GpuError::InvalidArgument(
                        "GSP-RM a raspuns la alta functie",
                    ));
                }
                None => {
                    if self.clock.now_us().saturating_sub(start) >= self.timeout_us {
                        return Err(GpuError::Timeout {
                            waited_us: self.clock.now_us() - start,
                        });
                    }
                    self.clock.delay_us(100);
                    (self.poll)(self.mem);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gsp::msgq::QueueLayout;
    use crate::gsp::sim_peer::SimGsp;
    use gpu_sim::clock::SimClock;
    use gpu_sim::fake_rm::{FakeRm, SIM_CTX_INFO, status};

    /// SIMULARE: a bump allocator over a pretend 8 GiB VRAM.
    struct Bump {
        pa: u64,
        va: u64,
        allocs: Vec<(u64, u64)>,
    }
    impl Bump {
        fn new() -> Self {
            Bump {
                pa: 0x1_0000_0000,
                va: 0x10_0000_0000,
                allocs: Vec::new(),
            }
        }
    }
    impl GoldenMemory for Bump {
        fn alloc_mapped(&mut self, size: u64, align: u64) -> GpuResult<Mapping> {
            let pa = self.alloc_phys(size, align)?;
            self.va = self.va.next_multiple_of(align.max(0x1000));
            let va = self.va;
            self.va += size.next_multiple_of(0x1000);
            Ok(Mapping { va, pa, size })
        }
        fn alloc_phys(&mut self, size: u64, align: u64) -> GpuResult<u64> {
            self.pa = self.pa.next_multiple_of(align.max(0x1000));
            let pa = self.pa;
            self.pa += size.next_multiple_of(0x1000);
            self.allocs.push((pa, size));
            Ok(pa)
        }
        fn reserve_va(&mut self, size: u64) -> GpuResult<(u64, Vec<PdeLevel>)> {
            self.va = self.va.next_multiple_of(size);
            let va = self.va;
            self.va += size;
            let pt = self.alloc_phys(0x3000, 0x1000)?;
            Ok((
                va,
                (0..3)
                    .map(|i| PdeLevel {
                        phys: pt + i * 0x1000,
                        size: 0x1000,
                        page_shift: [47, 38, 29][i as usize],
                        aperture: 1,
                    })
                    .collect(),
            ))
        }
    }

    /// Direct transport to the model (no queues), for the order/refusal tests.
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

    #[test]
    fn plan_follows_the_reported_sizes() {
        let plan = gr_buffer_plan(&SIM_CTX_INFO).unwrap();
        assert_eq!(
            plan.iter().map(|b| b.id).collect::<Vec<_>>(),
            [0, 1, 2, 3, 4, 5, 6, 9, 10, 11]
        );
        assert_eq!(plan[0].size, 0x4_2000 + GR_MAIN_EXTRA);
        assert_eq!(plan[5].align, 2 << 20);
        assert_eq!(plan[5].size, 2 << 20); // 0x130000 rounded up to 2 MiB
        assert_eq!(plan[9].size, plan[8].size); // 11 reuses 10's size
        assert!(plan[1].local && !plan[8].virt && !plan[3].phys);
        let mut bad = SIM_CTX_INFO;
        bad[16] = (0, 0x1000);
        assert!(gr_buffer_plan(&bad).is_err());
    }

    #[test]
    fn golden_image_runs_end_to_end_through_the_queues() {
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
        let mut mem = Bump::new();
        let g = {
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
            init_golden_image(&mut t, &mut mem).unwrap()
        };
        assert_eq!(
            (g.client, g.device, g.subdevice, g.vaspace),
            (CLIENT, 0xcf00_0001, 0xcf00_0002, 0xcf00_0003)
        );
        assert_eq!(
            (g.channel, g.compute, g.dma),
            (0xcf00_0004, 0xcf00_0005, 0xcf00_0006)
        );
        assert_eq!(g.runlists[0], (1, 0));
        assert_eq!(g.gr_bufs.len(), 10);
        // Order: 7 allocations and 4 controls, all accepted by the model.
        use gpu_sim::fake_rm::{class as c, cmd as k};
        assert_eq!(
            rm.log.allocs,
            [
                c::ROOT,
                c::DEVICE,
                c::SUBDEVICE,
                c::VASPACE,
                c::CHANNEL_GPFIFO,
                c::COMPUTE,
                c::DMA_COPY
            ]
            .map(|x| (x, status::OK))
        );
        assert_eq!(
            rm.log.controls,
            [
                k::GET_DEVICE_INFO_TABLE,
                k::COPY_SERVER_RESERVED_PDES,
                k::KGR_GET_CONTEXT_BUFFERS_INFO,
                k::PROMOTE_CTX
            ]
            .map(|x| (x, status::OK))
        );
        // No two VRAM allocations overlap.
        let mut a = mem.allocs.clone();
        a.sort();
        assert!(a.windows(2).all(|w| w[0].0 + w[0].1 <= w[1].0));
    }

    /// The model refuses a compute object without a promoted context; our
    /// sequence must surface that as an error, never as success.
    #[test]
    fn rm_refusals_stop_the_sequence_with_the_rm_code() {
        // Memory that returns an empty PDE list: refused before sending.
        struct NoPdes(Bump);
        impl GoldenMemory for NoPdes {
            fn alloc_mapped(&mut self, s: u64, a: u64) -> GpuResult<Mapping> {
                self.0.alloc_mapped(s, a)
            }
            fn alloc_phys(&mut self, s: u64, a: u64) -> GpuResult<u64> {
                self.0.alloc_phys(s, a)
            }
            fn reserve_va(&mut self, s: u64) -> GpuResult<(u64, Vec<PdeLevel>)> {
                Ok((self.0.reserve_va(s)?.0, Vec::new()))
            }
        }
        let mut t = Direct(FakeRm::new());
        assert!(matches!(
            init_golden_image(&mut t, &mut NoPdes(Bump::new())),
            Err(GpuError::InvalidArgument(_))
        ));
        assert_eq!(t.0.log.controls.len(), 1); // stopped before COPY_SERVER_RESERVED_PDES

        // A PDE level with a zero address: the model refuses (status 0x1f),
        // which must come back as that RM code.
        struct ZeroPde(Bump);
        impl GoldenMemory for ZeroPde {
            fn alloc_mapped(&mut self, s: u64, a: u64) -> GpuResult<Mapping> {
                self.0.alloc_mapped(s, a)
            }
            fn alloc_phys(&mut self, s: u64, a: u64) -> GpuResult<u64> {
                self.0.alloc_phys(s, a)
            }
            fn reserve_va(&mut self, s: u64) -> GpuResult<(u64, Vec<PdeLevel>)> {
                let (va, mut l) = self.0.reserve_va(s)?;
                l[1].phys = 0;
                Ok((va, l))
            }
        }
        let mut t = Direct(FakeRm::new());
        assert_eq!(
            init_golden_image(&mut t, &mut ZeroPde(Bump::new())),
            Err(GpuError::HardwareFault {
                code: status::INVALID_ARGUMENT
            })
        );
        assert!(
            t.0.log
                .allocs
                .iter()
                .all(|&(c, _)| c != gpu_sim::fake_rm::class::CHANNEL_GPFIFO)
        );

        // Without the promote, the model refuses the compute object.
        let mut rm = FakeRm::new();
        let mut mem = Bump::new();
        let mut r2 = Direct(core::mem::take(&mut rm));
        let mut seq = Rm {
            t: &mut r2,
            client: CLIENT,
            next: FIRST_HANDLE,
        };
        seq.alloc(0, class::ROOT, &rmapi::root_params()).unwrap();
        let dev = seq
            .alloc(CLIENT, class::DEVICE, &rmapi::device_params(CLIENT))
            .unwrap();
        let va = seq
            .alloc(dev, class::VASPACE, &rmapi::vaspace_params())
            .unwrap();
        let gp = mem.alloc_mapped(0x1000, 0x1000).unwrap();
        let ch_setup = ChannelSetup::golden(
            gp.va,
            gp.pa,
            va,
            mem.alloc_phys(0x1000, 0x1000).unwrap(),
            mem.alloc_phys(0x5000, 0x1000).unwrap(),
        );
        let ch = seq
            .alloc(
                dev,
                class::CHANNEL_GPFIFO,
                &rmapi::channel_params(&ch_setup).unwrap(),
            )
            .unwrap();
        assert_eq!(
            seq.alloc(ch, class::COMPUTE, &[]),
            Err(GpuError::HardwareFault {
                code: status::INVALID_STATE
            })
        );
        // The copy object needs no context.
        assert!(seq.alloc(ch, class::DMA_COPY, &[]).is_ok());
    }

    #[test]
    fn a_silent_gsp_is_a_timeout() {
        let l = QueueLayout::standard();
        let mut shm = alloc::vec![0u8; l.total_size];
        let pages: Vec<u64> = (0..l.page_table_entries as u64)
            .map(|i| 0x0500_0000 + i * 0x1000)
            .collect();
        let mut q = CpuQueues::init(&mut shm, l, &pages).unwrap();
        let _gsp = SimGsp::start(&mut shm, l.cmd_queue_offset, l.stat_queue_offset);
        q.link_rx(&shm).unwrap();
        let mut clock = SimClock::default();
        let mut t = QueueTransport {
            q: &mut q,
            mem: &mut shm,
            clock: &mut clock,
            timeout_us: 10_000,
            poll: |_: &mut dyn SharedMem| {},
            events: Vec::new(),
        };
        assert!(matches!(
            init_golden_image(&mut t, &mut Bump::new()),
            Err(GpuError::Timeout { .. })
        ));
    }

    /// I9a: the same sequence through the queues, with the driver's real
    /// memory manager (`mmu::GpuMemory`) instead of the bump allocator. Every
    /// mapped GR buffer and the GPFIFO are then walked by the independent
    /// page-table walker of `gpu_sim::vram`. SIMULARE.
    #[test]
    fn golden_image_with_real_page_tables() {
        use crate::mmu::{GpuMemory, VramWrite};
        use gpu_sim::vram::{SimVram, translate};
        struct Sim(SimVram);
        impl VramWrite for Sim {
            fn write_u64(&mut self, pa: u64, v: u64) -> GpuResult<()> {
                self.0.write_u64(pa, v).map_err(GpuError::InvalidArgument)
            }
        }
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
            GpuMemory::new(64 << 20, vram - (512 << 20), Sim(SimVram::new(vram))).unwrap();
        let g = {
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
            init_golden_image(&mut t, &mut mem).unwrap()
        };
        assert_eq!(rm.log.allocs.len(), 7);
        assert!(rm.log.allocs.iter().all(|&(_, s)| s == status::OK));
        assert!(rm.log.controls.iter().all(|&(_, s)| s == status::OK));
        let v = &mem.writer().0;
        let mut mapped = 0;
        for (b, m) in &g.gr_bufs {
            for off in (0..b.size).step_by(0x1000).chain([b.size - 1]) {
                let t = translate(v, mem.root(), m.va + off).unwrap();
                assert_eq!(t.pa, m.pa + off, "buffer {} + {off:#x}", b.id);
            }
            mapped += b.size;
        }
        assert_eq!(mapped, 0x35_5000); // the SIM context sizes, all walked
        // The reserved range: tables up to PD1, nothing below (RM's part).
        assert!(translate(v, mem.root(), g.reserved_va).is_err());
        // No page table lies inside a buffer, and all buffers lie in the zone.
        for t in mem.table_addresses() {
            assert!(
                g.gr_bufs
                    .iter()
                    .all(|(b, m)| t + 0x1000 <= m.pa || t >= m.pa + b.size)
            );
        }
        assert!(
            g.gr_bufs
                .iter()
                .all(|(b, m)| m.pa >= 64 << 20 && m.pa + b.size <= vram - (512 << 20))
        );
    }

    /// I9b: the same, but the page tables go into (simulated) VRAM through
    /// the BAR0 window (`vram_window::VramWindow` on `gpu_sim::pramin`), with
    /// the screen and GSP-RM's area protected, then one TLB invalidate. The
    /// GR buffers are walked by the independent walker afterwards. SIMULARE.
    #[test]
    fn golden_image_through_the_bar0_window() {
        use crate::mmu::GpuMemory;
        use crate::vram_window::VramWindow;
        use gpu_core::diag::Journal;
        use gpu_core::hal::Mmio;
        use gpu_sim::pramin::SimBar0Vram;
        use gpu_sim::vram::{SimVram, translate};
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
        let screen = (0, 16 << 20);
        let gsp_area = (vram - (512 << 20), vram);
        let mut bar = SimBar0Vram::new(SimVram::new(vram), 0);
        let mut j = Journal::new(256);
        let w = VramWindow::new(&mut bar, &mut j, None, vram, &[screen, gsp_area]).unwrap();
        let mut mem = GpuMemory::new(64 << 20, gsp_area.0, w).unwrap();
        let g = {
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
            init_golden_image(&mut t, &mut mem).unwrap()
        };
        let root = mem.root();
        let words = mem.writer().words;
        assert_eq!(words, 2 * mem.writes);
        mem.writer_mut().invalidate_tlb(&mut clock, root).unwrap();
        mem.writer_mut().restore().unwrap();
        drop(mem);
        assert_eq!(rm.log.allocs.len(), 7);
        assert!(bar.stray_writes.is_empty());
        assert_eq!(bar.invalidates.len(), 1);
        assert_eq!(bar.invalidates[0].pdb, root);
        assert_eq!(bar.invalidates[0].after_words, words);
        for (b, m) in &g.gr_bufs {
            for off in (0..b.size).step_by(0x1000).chain([b.size - 1]) {
                let t = translate(&bar.vram, root, m.va + off).unwrap();
                assert_eq!(t.pa, m.pa + off, "buffer {} + {off:#x}", b.id);
            }
        }
        assert!(!bar.vram.touched(screen.0, screen.1 - screen.0));
        assert!(!bar.vram.touched(gsp_area.0, gsp_area.1 - gsp_area.0));
        assert_eq!(bar.read32_raw(gpu_sim::pramin::BAR0_WINDOW), 0);
    }
}
