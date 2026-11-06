//! SIMULARE rig shared by the C-83 tests (I11, I12): golden image, working
//! client and work channel on `FakeRm`, the driver's page tables in a
//! simulated VRAM, host RAM as a second `SimVram` (bus addresses), and the
//! GPU side (`gpu_sim::cmdproc`) fetching everything through the page tables.
#![allow(dead_code)]

use gpu_core::error::{GpuError, GpuResult};
use gpu_sim::cmdproc::{CommandProcessor, Needs, ProgramFn};
use gpu_sim::fake_rm::FakeRm;
use gpu_sim::vram::SimVram;
use nvidia_ga104::gsp::golden::{GoldenMemory, Mapping, RmTransport, init_golden_image};
use nvidia_ga104::gsp::user_space::{
    USER_CLIENT, WorkChannel, open_compute_group, open_user_space, open_work_channel_cleared,
};
use nvidia_ga104::gsp::work_queue::{
    ChannelMemory, LOCAL_MEM_WINDOW, SHARED_MEM_WINDOW, USERD_GP_GET, WorkQueue, compute_setup,
};
use nvidia_ga104::mmu::{GpuMemory, SysMapping, VramWrite};
use nvidia_ga104::push::PushBuf;

pub struct Sim(pub SimVram);
impl VramWrite for Sim {
    fn write_u64(&mut self, pa: u64, v: u64) -> GpuResult<()> {
        self.0.write_u64(pa, v).map_err(GpuError::InvalidArgument)
    }
}
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

pub type Mem = GpuMemory<Sim>;
/// The host RAM model: 64 GiB of bus addresses (sparse).
pub const HOST_RAM: u64 = 64 << 30;

pub fn put_bytes(w: &mut Sim, pa: u64, bytes: &[u8]) {
    for (i, ch) in bytes.chunks(4).enumerate() {
        let mut b = [0u8; 4];
        b[..ch.len()].copy_from_slice(ch);
        w.write_u32(pa + 4 * i as u64, u32::from_le_bytes(b))
            .unwrap();
    }
}

pub fn get_bytes(w: &mut Sim, pa: u64, len: u64) -> Vec<u8> {
    (0..len.div_ceil(4))
        .flat_map(|i| w.read_u32(pa + 4 * i).unwrap().to_le_bytes())
        .take(len as usize)
        .collect()
}

/// The OS's DMA pages: deliberately not contiguous and above 4 GiB
/// (page i of buffer `which` at a scattered bus address).
pub fn dma_pages(which: u64, n: usize) -> Vec<u64> {
    (0..n as u64)
        .map(|i| 0x1_2000_0000 + which * 0x1000_0000 + ((i * 7) % n as u64) * 0x2000)
        .collect()
}

pub fn sys_write(sys: &mut SimVram, m: &SysMapping, data: &[u8]) {
    for (i, ch) in data.chunks(8).enumerate() {
        let off = 8 * i as u64;
        let mut b = [0u8; 8];
        b[..ch.len()].copy_from_slice(ch);
        let bus = m.pages[(off / 0x1000) as usize] + off % 0x1000;
        sys.write_u64(bus, u64::from_le_bytes(b)).unwrap();
    }
}

pub fn sys_read(sys: &SimVram, m: &SysMapping, len: u64) -> Vec<u8> {
    (0..len.div_ceil(8))
        .flat_map(|i| {
            let off = 8 * i;
            let bus = m.pages[(off / 0x1000) as usize] + off % 0x1000;
            sys.read_u64(bus).unwrap().to_le_bytes()
        })
        .take(len as usize)
        .collect()
}

pub struct Rig {
    pub mem: Mem,
    pub root: u64,
    pub w: WorkChannel,
    pub push: Mapping,
    pub wq: WorkQueue,
    pub sys: SimVram,
    /// The GPU model, kept between doorbells (it keeps the compute object bound by the first pass).
    pub cp: Option<CommandProcessor>,
}

/// Golden image → working client → work channel → compute setup submitted.
pub fn rig() -> Rig {
    let mut d = Direct(FakeRm::new());
    let vram = 8u64 << 30;
    let mut mem = GpuMemory::new(64 << 20, vram - (256 << 20), Sim(SimVram::new(vram))).unwrap();
    let g = init_golden_image(&mut d, &mut mem).unwrap();
    let u = open_user_space(&mut d, g.next_handle(), mem.root()).unwrap();
    let cg = open_compute_group(&mut d, &u).unwrap();
    let w = open_work_channel_cleared(&mut d, &mut mem, &u, &cg, &g).unwrap();
    let root = d.0.page_directory(USER_CLIENT, u.vaspace).unwrap().0;
    let push = mem.alloc_mapped(0x1000, 0x1000).unwrap();
    let mut wq = WorkQueue::new(&w, push).unwrap();
    let mut buf = [0u32; 16];
    let mut pb = PushBuf::new(&mut buf);
    compute_setup(&mut pb, LOCAL_MEM_WINDOW, SHARED_MEM_WINDOW).unwrap();
    let n = pb.len();
    wq.submit(mem.writer_mut(), &buf[..n]).unwrap();
    Rig {
        mem,
        root,
        w,
        push,
        wq,
        sys: SimVram::new(HOST_RAM),
        cp: None,
    }
}

/// The GPU's side of one doorbell: `programs` registered (on the first doorbell), every `(va, len)` of `loads`
/// fetched through the page tables (VRAM or host RAM), the channel run, `GPGet` and every `(va, len)` of
/// `stores` written back. The model is kept in the rig for the next doorbell (as the board keeps its state).
/// Simulated clock: 250 us per launch, 100 ns per method. Returns a snapshot of the events/timestamps of this
/// doorbell only.
pub fn gpu_runs(
    r: &mut Rig,
    programs: &[(&[u8], ProgramFn, Needs)],
    loads: &[(u64, u64)],
    stores: &[(u64, u64)],
) -> GpuResult<CommandProcessor> {
    let mut cp = r.cp.take().unwrap_or_else(|| {
        let mut cp = CommandProcessor::new();
        for &(code, f, n) in programs {
            cp.register_program_with_needs(code, f, n);
        }
        cp.sim_launch_ns = 250_000;
        cp.sim_method_ns = 100;
        cp
    });
    cp.events.clear();
    cp.timestamps.clear();
    let v = &r.mem.writer().0;
    let base = [(r.w.ring.va, r.w.ring.size), (r.push.va, r.push.size)];
    for &(va, len) in base.iter().chain(loads) {
        cp.mem
            .load_from(v, &r.sys, r.root, va, len.next_multiple_of(8))?;
    }
    let userd = r.w.ring.va + r.w.userd_offset;
    let res = cp.run_channel(r.w.ring.va, r.w.entries, userd);
    let v = &mut r.mem.writer_mut().0;
    let stored = (|| {
        res?;
        cp.mem
            .store_to(v, &mut r.sys, r.root, userd + USERD_GP_GET, 8)?;
        for &(va, len) in stores {
            cp.mem
                .store_to(v, &mut r.sys, r.root, va, len.next_multiple_of(8))?;
        }
        Ok(())
    })();
    let mut snap = CommandProcessor::new();
    snap.events = cp.events.clone();
    snap.timestamps = cp.timestamps.clone();
    r.cp = Some(cp);
    stored.map(|()| snap)
}

/// `(va, size)` of a VRAM mapping, for `gpu_runs`.
pub fn span(m: &Mapping) -> (u64, u64) {
    (m.va, m.size)
}
