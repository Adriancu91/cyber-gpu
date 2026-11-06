//! C-83 I11, SIMULARE: a large copy host RAM → VRAM → host RAM through the
//! work channel, end to end on the driver's real code: golden image, working
//! client, work channel (FakeRm), our page tables with **host-RAM PTEs**
//! (`mmu::alloc_sys_mapped`), the NAK-compiled `copy_b128`
//! (`tools/nak/out/copy_b128.sm86.sass`), one GPFIFO entry with three
//! timestamp records. The "GPU" is `gpu_sim::cmdproc`, which fetches
//! everything through the page tables with the independent walker
//! (`translate_any`) — host RAM is a second `SimVram` indexed by bus address.
//! Nothing here touches or proves anything about the real board; the speeds
//! are the test's own simulated clock.

use gpu_core::error::{GpuError, GpuResult};
use gpu_sim::cmdproc::{CommandProcessor, copy_b128_on_cpu};
use gpu_sim::fake_rm::FakeRm;
use gpu_sim::vram::SimVram;
use nvidia_ga104::gsp::golden::{GoldenMemory, Mapping, RmTransport, init_golden_image};
use nvidia_ga104::gsp::user_space::{
    USER_CLIENT, WorkChannel, open_compute_group, open_user_space, open_work_channel_cleared,
};
use nvidia_ga104::gsp::work_queue::{
    ChannelMemory, LOCAL_MEM_WINDOW, SHARED_MEM_WINDOW, USERD_GP_GET, WorkQueue, compute_setup,
};
use nvidia_ga104::mmu::{GpuMemory, PageAttr, SysMapping, VramWrite};
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::transfer::{self, COPY_B128, ROUND_TRIP_WORDS};

struct Sim(SimVram);
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

type Mem = GpuMemory<Sim>;
const SEED: u32 = 0x83;
const FIRST: u32 = 0x600;
/// The host RAM model: 64 GiB of bus addresses (sparse).
const HOST_RAM: u64 = 64 << 30;

fn put_bytes(w: &mut Sim, pa: u64, bytes: &[u8]) {
    for (i, ch) in bytes.chunks(4).enumerate() {
        let mut b = [0u8; 4];
        b[..ch.len()].copy_from_slice(ch);
        w.write_u32(pa + 4 * i as u64, u32::from_le_bytes(b))
            .unwrap();
    }
}

/// The OS's DMA pages: deliberately not contiguous and above 4 GiB
/// (page i of buffer `which` at a scattered bus address).
fn dma_pages(which: u64, n: usize) -> Vec<u64> {
    (0..n as u64)
        .map(|i| 0x1_2000_0000 + which * 0x1000_0000 + ((i * 7) % n as u64) * 0x2000)
        .collect()
}

fn sys_write(sys: &mut SimVram, m: &SysMapping, data: &[u8]) {
    for (i, ch) in data.chunks(8).enumerate() {
        let off = 8 * i as u64;
        let mut b = [0u8; 8];
        b[..ch.len()].copy_from_slice(ch);
        let bus = m.pages[(off / 0x1000) as usize] + off % 0x1000;
        sys.write_u64(bus, u64::from_le_bytes(b)).unwrap();
    }
}

fn sys_read(sys: &SimVram, m: &SysMapping, len: u64) -> Vec<u8> {
    (0..len.div_ceil(8))
        .flat_map(|i| {
            let off = 8 * i;
            let bus = m.pages[(off / 0x1000) as usize] + off % 0x1000;
            sys.read_u64(bus).unwrap().to_le_bytes()
        })
        .take(len as usize)
        .collect()
}

struct Rig {
    mem: Mem,
    root: u64,
    w: WorkChannel,
    push: Mapping,
    wq: WorkQueue,
    sys: SimVram,
}

fn rig() -> Rig {
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
    }
}

/// Everything one round trip needs, written by the driver.
struct Trip {
    bytes: u64,
    src: SysMapping,
    dst: SysMapping,
    vbuf: Mapping,
    prog: Mapping,
    cbufs: [Mapping; 2],
    qmds: [Mapping; 2],
    records: Mapping,
}

fn prepare(r: &mut Rig, bytes: u64, up_grid_override: Option<u32>) -> Trip {
    let pages = bytes.div_ceil(0x1000) as usize;
    let src = r.mem.alloc_sys_mapped(&dma_pages(1, pages)).unwrap();
    let dst = r.mem.alloc_sys_mapped(&dma_pages(2, pages)).unwrap();
    // The OS fills its RAM with the pattern; the destination with junk.
    let mut data = vec![0u8; bytes as usize];
    transfer::fill_pattern(&mut data, SEED).unwrap();
    sys_write(&mut r.sys, &src, &data);
    sys_write(&mut r.sys, &dst, &vec![0xee; bytes as usize]);
    let vbuf = r.mem.alloc_mapped(bytes, 0x1000).unwrap();
    let mut m = || r.mem.alloc_mapped(0x1000, 0x1000).unwrap();
    let (prog, cb0, cb1, q0, q1, records) = (m(), m(), m(), m(), m(), m());
    let w = r.mem.writer_mut();
    put_bytes(w, prog.pa, COPY_B128);
    put_bytes(
        w,
        cb0.pa,
        &transfer::copy_cbuf(src.va, vbuf.va, bytes).unwrap(),
    );
    put_bytes(
        w,
        cb1.pa,
        &transfer::copy_cbuf(vbuf.va, dst.va, bytes).unwrap(),
    );
    let mut up = transfer::copy_qmd(prog.va, cb0.va, bytes).unwrap();
    if let Some(g) = up_grid_override {
        up.grid[0] = g;
    }
    put_bytes(w, q0.pa, &up.to_bytes().unwrap());
    put_bytes(
        w,
        q1.pa,
        &transfer::copy_qmd(prog.va, cb1.va, bytes)
            .unwrap()
            .to_bytes()
            .unwrap(),
    );
    put_bytes(w, records.pa, &[0; 48]);
    Trip {
        bytes,
        src,
        dst,
        vbuf,
        prog,
        cbufs: [cb0, cb1],
        qmds: [q0, q1],
        records,
    }
}

fn submit(r: &mut Rig, t: &Trip) {
    let mut words = [0u32; ROUND_TRIP_WORDS];
    let mut pb = PushBuf::new(&mut words);
    transfer::encode_round_trip(&mut pb, t.qmds[0].va, t.qmds[1].va, t.records.va, FIRST).unwrap();
    r.wq.submit(r.mem.writer_mut(), &words).unwrap();
}

/// The GPU's side: every buffer fetched through the page tables (VRAM or
/// host RAM), the channel run, what the GPU wrote stored back the same way.
fn gpu_runs(r: &mut Rig, t: &Trip) -> GpuResult<CommandProcessor> {
    let mut cp = CommandProcessor::new();
    cp.register_program(COPY_B128, copy_b128_on_cpu);
    cp.sim_launch_ns = 250_000;
    cp.sim_method_ns = 100;
    let v = &r.mem.writer().0;
    let maps: Vec<(u64, u64)> = [
        &r.w.ring,
        &r.push,
        &t.prog,
        &t.cbufs[0],
        &t.cbufs[1],
        &t.qmds[0],
        &t.qmds[1],
        &t.records,
        &t.vbuf,
    ]
    .iter()
    .map(|m| (m.va, m.size.next_multiple_of(8)))
    .chain([(t.src.va, t.src.size), (t.dst.va, t.dst.size)])
    .collect();
    for &(va, len) in &maps {
        cp.mem.load_from(v, &r.sys, r.root, va, len)?;
    }
    let userd = r.w.ring.va + r.w.userd_offset;
    cp.run_channel(r.w.ring.va, r.w.entries, userd)?;
    let v = &mut r.mem.writer_mut().0;
    cp.mem
        .store_to(v, &mut r.sys, r.root, userd + USERD_GP_GET, 8)?;
    for (va, len) in [
        (t.vbuf.va, t.vbuf.size.next_multiple_of(8)),
        (t.dst.va, t.dst.size),
        (t.records.va, 48),
    ] {
        cp.mem.store_to(v, &mut r.sys, r.root, va, len)?;
    }
    Ok(cp)
}

fn records(r: &mut Rig, t: &Trip) -> Vec<u8> {
    (0..12)
        .flat_map(|i| {
            r.mem
                .writer_mut()
                .read_u32(t.records.pa + 4 * i)
                .unwrap()
                .to_le_bytes()
        })
        .collect()
}

#[test]
fn round_trip_ram_vram_ram_bit_for_bit() {
    for bytes in [16u64, 0x3230, 256 << 10] {
        let mut r = rig();
        let t = prepare(&mut r, bytes, None);
        submit(&mut r, &t);
        let cp = gpu_runs(&mut r, &t).unwrap();
        assert_eq!(cp.timestamps.len(), 3);
        // The driver's checks: the second RAM buffer, then VRAM spot words.
        let back = sys_read(&r.sys, &t.dst, t.bytes);
        assert_eq!(
            transfer::first_mismatch(&back, SEED),
            None,
            "{bytes} octeti"
        );
        for off in transfer::spot_offsets(t.bytes) {
            let got = r.mem.writer_mut().read_u32(t.vbuf.pa + off).unwrap();
            assert_eq!(
                got,
                transfer::pattern_word(SEED, off / 4),
                "VRAM la {off:#x}"
            );
        }
        // The source was not touched.
        let src = sys_read(&r.sys, &t.src, t.bytes);
        assert_eq!(transfer::first_mismatch(&src, SEED), None);
        // SIMULARE clock: each copy = one launch (250 us) + methods.
        let s = transfer::analyze_round_trip(&records(&mut r, &t), t.bytes, FIRST).unwrap();
        assert!(s.up_ns >= 250_000 && s.down_ns >= 250_000, "{s:?}");
        assert_eq!(r.wq.update_get(r.mem.writer_mut()), Ok(2));
    }
}

/// The host RAM really is reached through the PTEs: the copy lands on the
/// bus addresses the OS gave, page by page (scattered), not anywhere else.
#[test]
fn host_pages_are_the_ones_the_os_gave() {
    let mut r = rig();
    let t = prepare(&mut r, 3 * 0x1000, None);
    submit(&mut r, &t);
    gpu_runs(&mut r, &t).unwrap();
    for (i, &bus) in t.dst.pages.iter().enumerate() {
        let w = r.sys.read_u64(bus).unwrap() as u32;
        assert_eq!(
            w,
            transfer::pattern_word(SEED, i as u64 * 1024),
            "pagina {i}"
        );
        assert_eq!(r.mem.lookup_sys(t.dst.va + i as u64 * 0x1000), Some(bus));
    }
    assert_eq!(r.mem.lookup(t.dst.va), None, "nu e VRAM");
}

/// Mistakes the checks must catch (each in a fresh rig).
#[test]
fn planted_mistakes_are_caught() {
    // 1. A grid one block short for RAM→VRAM: the last 4 KiB never arrive.
    let mut r = rig();
    let t = prepare(&mut r, 256 << 10, Some(((256 << 10) / 16 / 256) - 1));
    submit(&mut r, &t);
    gpu_runs(&mut r, &t).unwrap();
    let back = sys_read(&r.sys, &t.dst, t.bytes);
    assert_eq!(
        transfer::first_mismatch(&back, SEED).map(|m| m.0),
        Some((256 << 10) - 4096)
    );

    // 2. The OS's page list for the destination has a page twice: the copy
    // "succeeds" for the GPU, the check names the page that is wrong.
    let mut r = rig();
    let t = prepare(&mut r, 4 * 0x1000, None);
    let mut bad = t.dst.pages.clone();
    bad[3] = bad[1];
    let wrong = r.mem.alloc_sys_mapped(&bad).unwrap();
    let w = r.mem.writer_mut();
    put_bytes(
        w,
        t.cbufs[1].pa,
        &transfer::copy_cbuf(t.vbuf.va, wrong.va, t.bytes).unwrap(),
    );
    submit(&mut r, &t);
    let mut t2 = t;
    t2.dst = wrong;
    gpu_runs(&mut r, &t2).unwrap();
    let mut page1 = vec![0u8; 0x1000];
    for (i, b) in page1.chunks_mut(8).enumerate() {
        b.copy_from_slice(&r.sys.read_u64(bad[1] + 8 * i as u64).unwrap().to_le_bytes());
    }
    // Page 1's bus address holds page 3's words: a mismatch at its first word.
    assert_eq!(
        u32::from_le_bytes(page1[..4].try_into().unwrap()),
        transfer::pattern_word(SEED, 3 * 1024)
    );

    // 3. A destination mapped read-only: the GPU's store is refused (fault).
    let mut r = rig();
    let t = prepare(&mut r, 0x1000, None);
    let ro = r.mem.alloc_sys_mapped(&dma_pages(3, 1)).unwrap();
    let ro_va = ro.va + (1 << 30);
    r.mem
        .map_sys(
            ro_va,
            &ro.pages,
            PageAttr {
                volatile: true,
                read_only: true,
            },
        )
        .unwrap();
    let w = r.mem.writer_mut();
    put_bytes(
        w,
        t.cbufs[1].pa,
        &transfer::copy_cbuf(t.vbuf.va, ro_va, t.bytes).unwrap(),
    );
    submit(&mut r, &t);
    let mut t3 = t;
    t3.dst = SysMapping { va: ro_va, ..ro };
    assert!(gpu_runs(&mut r, &t3).is_err());

    // 4. Records: a run that never happened (zeros) is named, not guessed.
    let mut r = rig();
    let t = prepare(&mut r, 0x1000, None);
    let e = transfer::analyze_round_trip(&records(&mut r, &t), t.bytes, FIRST).unwrap_err();
    assert!(format!("{e:?}").contains("inregistrarea 0"));
}

/// Host-RAM mappings are refused before anything is written when wrong.
#[test]
fn bad_host_mappings_are_refused() {
    let mut r = rig();
    let before = r.mem.writes;
    assert!(r.mem.alloc_sys_mapped(&[]).is_err());
    assert!(r.mem.alloc_sys_mapped(&[0x1234]).is_err(), "nealiniat");
    assert!(r.mem.alloc_sys_mapped(&[1 << 52]).is_err(), "prea mare");
    assert_eq!(r.mem.writes, before);
    let a = r.mem.alloc_sys_mapped(&[0x2000_0000]).unwrap();
    let v = r.mem.alloc_mapped(0x1000, 0x1000).unwrap();
    let w = r.mem.writes;
    assert!(
        r.mem
            .map_sys(a.va, &[0x3000_0000], PageAttr::default())
            .is_err(),
        "deja mapata"
    );
    assert!(
        r.mem
            .map_sys(v.va, &[0x3000_0000], PageAttr::default())
            .is_err(),
        "peste VRAM"
    );
    assert!(
        r.mem
            .map(a.va, 0x10_0000, 0x1000, PageAttr::default())
            .is_err(),
        "VRAM peste RAM"
    );
    assert_eq!(r.mem.writes, w);
}
