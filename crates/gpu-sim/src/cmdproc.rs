//! # SIMULARE — a model of the Ampere command processor (C-20 D, part 1)
//!
//! Reads a GPFIFO, decodes the pushbuffer it points at, and executes the few
//! methods the first compute launch needs:
//! - channel class C56F: `SEM_ADDR_LO/HI`, `SEM_PAYLOAD_LO/HI`, `SEM_EXECUTE`
//!   (release 32/64-bit, acquire ≥ — an acquire that is not yet satisfied is
//!   an error here, since nothing else runs in parallel);
//! - compute class C7C0: `SET_OBJECT`, `SEND_PCAS_A` (QMD address >> 8) and
//!   `SEND_SIGNALING_PCAS2_B` (launch), reading a QMD V03_00.
//!
//! A launched program is *recognised by its exact bytes* (registered with
//! [`CommandProcessor::register_program`]) and "run" by a Rust function on
//! the CPU. The model does not interpret SASS. Every number comes from the
//! generated crate `nvidia-rm570` (NVIDIA open-gpu-doc, MIT), not from our
//! encoder in `nvidia-ga104::push`, so a test that drives this model with
//! that encoder checks the encoder against the headers.
//!
//! What a passing test proves: our pushbuffer, GPFIFO entries and QMD are
//! consistent with NVIDIA's published layouts *as this model reads them*.
//! It does not prove the RTX 3070 Ti accepts them.
//! Idea (not code) from tinygrad `test/mockgpu/nv/nvgpu.py` (MIT).

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h, qmd_v03_00 as q};
use nvidia_rm570::{QMD_SIZE, QmdField};

const PAGE: u64 = 4096;

/// GPU memory as seen through GPU virtual addresses (identity; no MMU yet).
#[derive(Default)]
pub struct GpuMem {
    pages: BTreeMap<u64, Box<[u8; PAGE as usize]>>,
}

impl GpuMem {
    pub fn write(&mut self, va: u64, data: &[u8]) {
        for (i, &b) in data.iter().enumerate() {
            let a = va + i as u64;
            self.pages
                .entry(a / PAGE)
                .or_insert_with(|| Box::new([0; PAGE as usize]))[(a % PAGE) as usize] = b;
        }
    }

    /// Reading memory that was never written is an error (a real GPU would
    /// fault on an unmapped address).
    pub fn read(&self, va: u64, out: &mut [u8]) -> GpuResult<()> {
        for (i, b) in out.iter_mut().enumerate() {
            let a = va + i as u64;
            let p = self
                .pages
                .get(&(a / PAGE))
                .ok_or(GpuError::InvalidArgument(
                    "SIM: citire din memorie GPU nemapata",
                ))?;
            *b = p[(a % PAGE) as usize];
        }
        Ok(())
    }

    pub fn read_u32(&self, va: u64) -> GpuResult<u32> {
        let mut b = [0; 4];
        self.read(va, &mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    pub fn read_u64(&self, va: u64) -> GpuResult<u64> {
        let mut b = [0; 8];
        self.read(va, &mut b)?;
        Ok(u64::from_le_bytes(b))
    }

    pub fn write_u32(&mut self, va: u64, v: u32) {
        self.write(va, &v.to_le_bytes());
    }

    pub fn read_f32s(&self, va: u64, n: usize) -> GpuResult<Vec<f32>> {
        (0..n)
            .map(|i| self.read_u32(va + 4 * i as u64).map(f32::from_bits))
            .collect()
    }

    pub fn write_f32s(&mut self, va: u64, v: &[f32]) {
        for (i, x) in v.iter().enumerate() {
            self.write_u32(va + 4 * i as u64, x.to_bits());
        }
    }
}

/// What a recognised program sees when "launched".
pub struct Launch<'a> {
    pub mem: &'a mut GpuMem,
    /// Constant buffer 0 address (from the QMD).
    pub cbuf0: u64,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

pub type ProgramFn = fn(&mut Launch<'_>) -> GpuResult<()>;

/// What a program needs from its QMD besides the code (session 8). The model
/// refuses a launch whose QMD gives less, as a real SM would misbehave.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Needs {
    pub shared_bytes: u32,
    pub barriers: u32,
}

/// Events the model records, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Event {
    SetObject { subch: u8, class: u32 },
    Release { addr: u64, payload: u64 },
    Acquire { addr: u64, at_least: u64 },
    Launch { qmd: u64, program: usize },
}

pub struct CommandProcessor {
    pub mem: GpuMem,
    programs: Vec<(Vec<u8>, ProgramFn, Needs)>,
    objects: [u32; 8],
    sem_addr: u64,
    sem_payload: u64,
    pcas_qmd: Option<u64>,
    pub events: Vec<Event>,
}

impl Default for CommandProcessor {
    fn default() -> Self {
        Self::new()
    }
}

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

impl CommandProcessor {
    pub fn new() -> Self {
        CommandProcessor {
            mem: GpuMem::default(),
            programs: Vec::new(),
            objects: [0; 8],
            sem_addr: 0,
            sem_payload: 0,
            pcas_qmd: None,
            events: Vec::new(),
        }
    }

    /// The model will "run" `run` when a QMD points at exactly `code`.
    /// Returns the program index used in [`Event::Launch`].
    pub fn register_program(&mut self, code: &[u8], run: ProgramFn) -> usize {
        self.register_program_with_needs(code, run, Needs::default())
    }

    /// Like [`Self::register_program`], for a program that uses shared memory
    /// and barriers: the QMD must declare at least `needs`.
    pub fn register_program_with_needs(
        &mut self,
        code: &[u8],
        run: ProgramFn,
        needs: Needs,
    ) -> usize {
        self.programs.push((code.to_vec(), run, needs));
        self.programs.len() - 1
    }

    /// Processes GPFIFO entries (pairs of u32 as in clc56f `GP_ENTRY0/1`).
    pub fn run_gpfifo(&mut self, entries: &[(u32, u32)]) -> GpuResult<()> {
        for &(e0, e1) in entries {
            if h::NVC56F_GP_ENTRY0_FETCH.get(e0) != 0 {
                return Err(bad("SIM: GPFIFO cu FETCH conditional (nemodelat)"));
            }
            let va = ((h::NVC56F_GP_ENTRY1_GET_HI.get(e1) as u64) << 32)
                | ((h::NVC56F_GP_ENTRY0_GET.get(e0) as u64) << 2);
            let len = h::NVC56F_GP_ENTRY1_LENGTH.get(e1) as usize;
            if len == 0 {
                return Err(bad("SIM: GPFIFO cu lungime 0"));
            }
            let words: Vec<u32> = (0..len)
                .map(|i| self.mem.read_u32(va + 4 * i as u64))
                .collect::<GpuResult<_>>()?;
            self.run_pushbuffer(&words)?;
        }
        Ok(())
    }

    fn run_pushbuffer(&mut self, w: &[u32]) -> GpuResult<()> {
        let mut i = 0;
        while i < w.len() {
            let hdr = w[i];
            i += 1;
            let op = h::NVC56F_DMA_SEC_OP.get(hdr);
            let subch = h::NVC56F_DMA_METHOD_SUBCHANNEL.get(hdr) as u8;
            let method = h::NVC56F_DMA_METHOD_ADDRESS.get(hdr) << 2;
            if op == h::NVC56F_DMA_SEC_OP_IMMD_DATA_METHOD {
                self.method(subch, method, h::NVC56F_DMA_IMMD_DATA.get(hdr))?;
                continue;
            }
            let n = h::NVC56F_DMA_METHOD_COUNT.get(hdr) as usize;
            let data = w.get(i..i + n).ok_or(bad("SIM: pushbuffer trunchiat"))?;
            i += n;
            for (k, &d) in data.iter().enumerate() {
                let m = match op {
                    h::NVC56F_DMA_SEC_OP_INC_METHOD => method + 4 * k as u32,
                    h::NVC56F_DMA_SEC_OP_NON_INC_METHOD => method,
                    h::NVC56F_DMA_SEC_OP_ONE_INC => method + if k == 0 { 0 } else { 4 },
                    _ => return Err(bad("SIM: SEC_OP nemodelat")),
                };
                self.method(subch, m, d)?;
            }
        }
        Ok(())
    }

    fn method(&mut self, subch: u8, m: u32, d: u32) -> GpuResult<()> {
        // Host (channel) methods are below 0x100 on every subchannel.
        if m < 0x100 && m != c::NVC7C0_SET_OBJECT {
            return self.host_method(m, d);
        }
        if m == c::NVC7C0_SET_OBJECT {
            let class = c::NVC7C0_SET_OBJECT_CLASS_ID.get(d);
            self.objects[subch as usize] = class;
            self.events.push(Event::SetObject { subch, class });
            return Ok(());
        }
        if self.objects[subch as usize] != c::AMPERE_COMPUTE_B {
            return Err(bad(
                "SIM: metoda de calcul pe un subcanal fara AMPERE_COMPUTE_B",
            ));
        }
        match m {
            c::NVC7C0_SEND_PCAS_A => {
                self.pcas_qmd =
                    Some((c::NVC7C0_SEND_PCAS_A_QMD_ADDRESS_SHIFTED8.get(d) as u64) << 8);
                Ok(())
            }
            c::NVC7C0_SEND_SIGNALING_PCAS2_B => {
                let a = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION.get(d);
                if a != c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE
                    && a != c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE
                {
                    return Err(bad("SIM: actiune PCAS nemodelata"));
                }
                let qmd = self
                    .pcas_qmd
                    .take()
                    .ok_or(bad("SIM: lansare fara SEND_PCAS_A"))?;
                self.launch(qmd)
            }
            _ => Err(bad("SIM: metoda de calcul nemodelata")),
        }
    }

    fn host_method(&mut self, m: u32, d: u32) -> GpuResult<()> {
        match m {
            h::NVC56F_SEM_ADDR_LO => {
                self.sem_addr = (self.sem_addr & !0xffff_ffff) | (d & !3) as u64
            }
            h::NVC56F_SEM_ADDR_HI => {
                self.sem_addr = (self.sem_addr & 0xffff_ffff) | ((d & 0x1ff_ffff) as u64) << 32
            }
            h::NVC56F_SEM_PAYLOAD_LO => {
                self.sem_payload = (self.sem_payload & !0xffff_ffff) | d as u64
            }
            h::NVC56F_SEM_PAYLOAD_HI => {
                self.sem_payload = (self.sem_payload & 0xffff_ffff) | (d as u64) << 32
            }
            h::NVC56F_SEM_EXECUTE => {
                let wide = h::NVC56F_SEM_EXECUTE_PAYLOAD_SIZE.get(d)
                    == h::NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT;
                match h::NVC56F_SEM_EXECUTE_OPERATION.get(d) {
                    h::NVC56F_SEM_EXECUTE_OPERATION_RELEASE => {
                        let p = if wide {
                            self.sem_payload
                        } else {
                            self.sem_payload & 0xffff_ffff
                        };
                        let bytes = p.to_le_bytes();
                        self.mem
                            .write(self.sem_addr, &bytes[..if wide { 8 } else { 4 }]);
                        self.events.push(Event::Release {
                            addr: self.sem_addr,
                            payload: p,
                        });
                    }
                    h::NVC56F_SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ => {
                        let cur = if wide {
                            self.mem.read_u64(self.sem_addr)?
                        } else {
                            self.mem.read_u32(self.sem_addr)? as u64
                        };
                        if cur < self.sem_payload {
                            return Err(bad("SIM: semaforul asteptat nu vine niciodata"));
                        }
                        self.events.push(Event::Acquire {
                            addr: self.sem_addr,
                            at_least: self.sem_payload,
                        });
                    }
                    _ => return Err(bad("SIM: operatie de semafor nemodelata")),
                }
            }
            _ => return Err(bad("SIM: metoda de canal nemodelata")),
        }
        Ok(())
    }

    fn launch(&mut self, qmd_va: u64) -> GpuResult<()> {
        let mut raw = [0u8; QMD_SIZE];
        self.mem.read(qmd_va, &mut raw)?;
        let mut qmd = [0u32; QMD_SIZE / 4];
        for (i, w) in qmd.iter_mut().enumerate() {
            *w = u32::from_le_bytes([raw[4 * i], raw[4 * i + 1], raw[4 * i + 2], raw[4 * i + 3]]);
        }
        let g = |f: QmdField| f.get(&qmd);
        if g(q::NVC6C0_QMDV03_00_QMD_MAJOR_VERSION) != 3 || g(q::NVC6C0_QMDV03_00_QMD_VERSION) != 0
        {
            return Err(bad("SIM: QMD nu e V03_00"));
        }
        let prog = ((g(q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER) as u64) << 32)
            | g(q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER) as u64;
        let index = self
            .programs
            .iter()
            .position(|(code, _, _)| {
                let mut buf = alloc::vec![0u8; code.len()];
                self.mem.read(prog, &mut buf).is_ok() && &buf == code
            })
            .ok_or(bad("SIM: program necunoscut la adresa din QMD"))?;
        if g(q::NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID(0)) != 1 {
            return Err(bad("SIM: bufferul constant 0 nu e valid in QMD"));
        }
        let cbuf0 = ((g(q::NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(0)) as u64) << 32)
            | g(q::NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(0)) as u64;
        let grid = [
            g(q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH),
            g(q::NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT),
            g(q::NVC6C0_QMDV03_00_CTA_RASTER_DEPTH),
        ];
        let block = [
            g(q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0),
            g(q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION1),
            g(q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION2),
        ];
        if grid.contains(&0) || block.contains(&0) {
            return Err(bad("SIM: dimensiune 0 in QMD"));
        }
        let (run, needs) = (self.programs[index].1, self.programs[index].2);
        if g(q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE) < needs.shared_bytes {
            return Err(bad(
                "SIM: QMD da mai putina memorie partajata decat cere programul",
            ));
        }
        if g(q::NVC6C0_QMDV03_00_BARRIER_COUNT) < needs.barriers {
            return Err(bad(
                "SIM: QMD declara mai putine bariere decat foloseste programul",
            ));
        }
        run(&mut Launch {
            mem: &mut self.mem,
            cbuf0,
            grid,
            block,
        })?;
        self.events.push(Event::Launch {
            qmd: qmd_va,
            program: index,
        });
        if g(q::NVC6C0_QMDV03_00_RELEASE0_ENABLE) == q::NVC6C0_QMDV03_00_RELEASE0_ENABLE_TRUE {
            let addr = ((g(q::NVC6C0_QMDV03_00_RELEASE0_ADDRESS_UPPER) as u64) << 32)
                | g(q::NVC6C0_QMDV03_00_RELEASE0_ADDRESS_LOWER) as u64;
            let p = g(q::NVC6C0_QMDV03_00_RELEASE0_PAYLOAD_LOWER);
            self.mem.write_u32(addr, p);
            self.events.push(Event::Release {
                addr,
                payload: p as u64,
            });
        }
        Ok(())
    }
}

/// The CPU stand-in for `tools/nak/out/add64.sm86.sass`: reads the three
/// addresses from cbuf0 at 0x28/0x30/0x38 (tools/nak/README.md) and does
/// c[i] = a[i] + b[i] for every thread of a 64×1×1 block, through
/// `gpu-compute::reference::add_ftz` — NAK compiled the sum to `FADD.FTZ`
/// (subnormals flushed), so that is the oracle, not `reference::add`.
pub fn add64_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.grid != [1, 1, 1] || l.block != [64, 1, 1] {
        return Err(bad("SIM: add64 cere un bloc de 64x1x1"));
    }
    let a = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let b = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let c = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let (x, y) = (l.mem.read_f32s(a, 64)?, l.mem.read_f32s(b, 64)?);
    let mut out = alloc::vec![0.0f32; 64];
    gpu_compute::reference::add_ftz(&x, &y, &mut out)?;
    l.mem.write_f32s(c, &out);
    Ok(())
}

/// The CPU stand-in for `tools/nak/out/matmul_naive.sm86.sass` (C-20 E,
/// step 1). Layout read from the NAK dump (`matmul_naive.sm86.txt`): the
/// workgroup base at cbuf0 0x0/0x4 (NVK's dispatch base; added to the block
/// index), a/b/c at 0x28/0x30/0x38, m/n/k at 0x40/0x44/0x48; thread x → row,
/// y → column; 16×16 threads per block; threads outside C return. Each
/// element is `reference::matmul_fma_ftz` (the dump shows `ffma.ftz`).
pub fn matmul_naive_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    matmul_on_cpu(l, false)
}

/// Step 2 (`matmul_coalesced.sm86.sass`): the same, with thread x → column
/// and y → row (the grid's x covers the columns).
pub fn matmul_coalesced_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    matmul_on_cpu(l, true)
}

fn matmul_on_cpu(l: &mut Launch<'_>, x_is_col: bool) -> GpuResult<()> {
    if l.block != [16, 16, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: matmul_naive cere blocuri de 16x16x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let a = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let b = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let c = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let m = l.mem.read_u32(l.cbuf0 + 0x40)? as usize;
    let n = l.mem.read_u32(l.cbuf0 + 0x44)? as usize;
    let k = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let x = l.mem.read_f32s(a, m * k)?;
    let y = l.mem.read_f32s(b, k * n)?;
    let mut out = alloc::vec![0.0f32; m * n];
    gpu_compute::reference::matmul_fma_ftz(&x, &y, &mut out, m, k, n)?;
    let (gx, gy) = ((l.grid[0] * 16) as usize, (l.grid[1] * 16) as usize);
    let (rows, cols) = if x_is_col { (gy, gx) } else { (gx, gy) };
    // Only the elements some thread owns are written (a grid that is too
    // small leaves the rest of C untouched, as the real kernel would).
    for r in 0..m.min(rows) {
        for col in 0..n.min(cols) {
            l.mem
                .write_u32(c + 4 * (r * n + col) as u64, out[r * n + col].to_bits());
        }
    }
    Ok(())
}

/// What `tools/nak/out/matmul_smem.sm86.sass` (step 3) needs: two 16×16 f32
/// tiles (`as_` at 0, `bs_` at 0x400 in the dump) and one `bar.sync` barrier.
pub const MATMUL_SMEM_NEEDS: Needs = Needs {
    shared_bytes: 2 * 16 * 16 * 4,
    barriers: 1,
};

/// The CPU stand-in for step 3 (`matmul_smem.sm86.sass`, shared-memory
/// tiles). Unlike steps 1–2 it follows the kernel's own algorithm: for each
/// workgroup, tile by tile, it fills the two 16×16 "shared" arrays exactly as
/// the threads would (zeros outside A/B), then every "thread" does its
/// `ffma.ftz` chain over min(16, K − t) entries, then the next tile. So the
/// test checks the tiling index arithmetic and the edge handling, and that
/// the result equals the plain FMA+FTZ oracle bit for bit. Thread x → column,
/// y → row; cbuf0 layout as steps 1–2.
pub fn matmul_smem_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    const T: usize = 16;
    if l.block != [16, 16, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: matmul_smem cere blocuri de 16x16x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let a = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let b = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let c = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let m = l.mem.read_u32(l.cbuf0 + 0x40)? as usize;
    let n = l.mem.read_u32(l.cbuf0 + 0x44)? as usize;
    let k = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let x = l.mem.read_f32s(a, m * k)?;
    let y = l.mem.read_f32s(b, k * n)?;
    let fma_f32_ftz = gpu_compute::math::fma_f32_ftz;
    for gy in 0..l.grid[1] as usize {
        for gx in 0..l.grid[0] as usize {
            let mut acc = [[0.0f32; T]; T]; // [ty][tx]
            let mut t = 0;
            while t < k {
                let (mut as_, mut bs_) = ([[0.0f32; T]; T], [[0.0f32; T]; T]);
                for ty in 0..T {
                    for tx in 0..T {
                        let (row, col) = (gy * T + ty, gx * T + tx);
                        if row < m && t + tx < k {
                            as_[ty][tx] = x[row * k + t + tx];
                        }
                        if t + ty < k && col < n {
                            bs_[ty][tx] = y[(t + ty) * n + col];
                        }
                    }
                }
                // barrier()
                let len = T.min(k - t);
                for (ty, acc_row) in acc.iter_mut().enumerate() {
                    for (tx, acc_el) in acc_row.iter_mut().enumerate() {
                        for p in 0..len {
                            *acc_el = fma_f32_ftz(as_[ty][p], bs_[p][tx], *acc_el);
                        }
                    }
                }
                // barrier()
                t += T;
            }
            for (ty, acc_row) in acc.iter().enumerate() {
                for (tx, &v) in acc_row.iter().enumerate() {
                    let (row, col) = (gy * T + ty, gx * T + tx);
                    if row < m && col < n {
                        l.mem.write_u32(c + 4 * (row * n + col) as u64, v.to_bits());
                    }
                }
            }
        }
    }
    Ok(())
}

/// What `tools/nak/out/matmul_1d.sm86.sass` (step 4) needs: the 64×8 tile of
/// A at 0 and the 8×64 tile of B at 0x800 (4 KiB) and one barrier.
pub const MATMUL_1D_NEEDS: Needs = Needs {
    shared_bytes: (64 * 8 + 8 * 64) * 4,
    barriers: 1,
};

/// The CPU stand-in for step 4 (`matmul_1d.sm86.sass`, siboehm "1D
/// blocktiling"): workgroups of 512 threads, each owning a 64×64 block of C;
/// thread `tid` owns column `tid % 64` and the 8 rows `(tid / 64) * 8 ..+8`.
/// It follows the kernel's own algorithm: per K-tile of 8, the 64×8 and 8×64
/// "shared" tiles are filled exactly as the 512 threads load them (zeros
/// outside A/B), then each thread reuses one B value for its 8 accumulators,
/// over min(8, K − t) entries only. Grid x → column blocks, y → row blocks;
/// cbuf0 layout as steps 1–3. Oracle: `reference::matmul_fma_ftz`.
pub fn matmul_1d_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    const BM: usize = 64;
    const BN: usize = 64;
    const BK: usize = 8;
    const TM: usize = 8;
    const THREADS: usize = BM * BN / TM;
    if l.block != [THREADS as u32, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: matmul_1d cere blocuri de 512x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let a = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let b = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let c = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let m = l.mem.read_u32(l.cbuf0 + 0x40)? as usize;
    let n = l.mem.read_u32(l.cbuf0 + 0x44)? as usize;
    let k = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let x = l.mem.read_f32s(a, m * k)?;
    let y = l.mem.read_f32s(b, k * n)?;
    let fma_f32_ftz = gpu_compute::math::fma_f32_ftz;
    for gy in 0..l.grid[1] as usize {
        for gx in 0..l.grid[0] as usize {
            let (row0, col0) = (gy * BM, gx * BN);
            let mut acc = alloc::vec![[0.0f32; TM]; THREADS];
            let mut t = 0;
            while t < k {
                let mut as_ = [0.0f32; BM * BK];
                let mut bs_ = [0.0f32; BK * BN];
                for tid in 0..THREADS {
                    let (ar, ac) = (tid / BK, tid % BK);
                    if row0 + ar < m && t + ac < k {
                        as_[ar * BK + ac] = x[(row0 + ar) * k + t + ac];
                    }
                    let (br, bc) = (tid / BN, tid % BN);
                    if t + br < k && col0 + bc < n {
                        bs_[br * BN + bc] = y[(t + br) * n + col0 + bc];
                    }
                }
                // barrier()
                let len = BK.min(k - t);
                for (tid, acc_t) in acc.iter_mut().enumerate() {
                    let (tcol, trow) = (tid % BN, tid / BN);
                    for p in 0..len {
                        let bv = bs_[p * BN + tcol];
                        for (i, acc_el) in acc_t.iter_mut().enumerate() {
                            *acc_el = fma_f32_ftz(as_[(trow * TM + i) * BK + p], bv, *acc_el);
                        }
                    }
                }
                // barrier()
                t += BK;
            }
            for (tid, acc_t) in acc.iter().enumerate() {
                let col = col0 + tid % BN;
                for (i, &v) in acc_t.iter().enumerate() {
                    let row = row0 + (tid / BN) * TM + i;
                    if row < m && col < n {
                        l.mem.write_u32(c + 4 * (row * n + col) as u64, v.to_bits());
                    }
                }
            }
        }
    }
    Ok(())
}

/// What `tools/nak/out/matmul_2d.sm86.sass` (step 5) needs: the 64×8 tile of
/// A and the 8×64 tile of B (4 KiB) and one barrier.
pub const MATMUL_2D_NEEDS: Needs = Needs {
    shared_bytes: (64 * 8 + 8 * 64) * 4,
    barriers: 1,
};

/// The CPU stand-in for step 5 (`matmul_2d.sm86.sass`, siboehm "2D
/// blocktiling"): workgroups of 64 threads, each owning a 64×64 block of C;
/// thread `tid` owns the 8×8 cells at rows `(tid / 8) * 8 ..+8`, columns
/// `(tid % 8) * 8 ..+8`. It follows the kernel's own algorithm: per K-tile of
/// 8, the "shared" tiles are filled exactly as the 64 threads load them (A:
/// rows `tid / 8 + 8r`, column `tid % 8`; B: rows `r`, column `tid`; zeros
/// outside A/B), then for each p ≤ min(8, K − t) the thread copies 8 values of
/// A and 8 of B to "registers" and accumulates the 64 products. Grid x →
/// column blocks, y → row blocks; cbuf0 as steps 1–4. Oracle:
/// `reference::matmul_fma_ftz`.
pub fn matmul_2d_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    const BM: usize = 64;
    const BN: usize = 64;
    const BK: usize = 8;
    const TM: usize = 8;
    const TN: usize = 8;
    const THREADS: usize = (BM / TM) * (BN / TN);
    if l.block != [THREADS as u32, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: matmul_2d cere blocuri de 64x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let a = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let b = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let c = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let m = l.mem.read_u32(l.cbuf0 + 0x40)? as usize;
    let n = l.mem.read_u32(l.cbuf0 + 0x44)? as usize;
    let k = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let x = l.mem.read_f32s(a, m * k)?;
    let y = l.mem.read_f32s(b, k * n)?;
    let fma_f32_ftz = gpu_compute::math::fma_f32_ftz;
    for gy in 0..l.grid[1] as usize {
        for gx in 0..l.grid[0] as usize {
            let (row0, col0) = (gy * BM, gx * BN);
            let mut acc = alloc::vec![[0.0f32; TM * TN]; THREADS];
            let mut t = 0;
            while t < k {
                let mut as_ = [0.0f32; BM * BK];
                let mut bs_ = [0.0f32; BK * BN];
                for tid in 0..THREADS {
                    let (ar, ac) = (tid / BK, tid % BK);
                    for off in (0..BM).step_by(THREADS / BK) {
                        if row0 + ar + off < m && t + ac < k {
                            as_[(ar + off) * BK + ac] = x[(row0 + ar + off) * k + t + ac];
                        }
                    }
                    let bc = tid % BN;
                    for r in 0..BK {
                        if t + r < k && col0 + bc < n {
                            bs_[r * BN + bc] = y[(t + r) * n + col0 + bc];
                        }
                    }
                }
                // barrier()
                let len = BK.min(k - t);
                for (tid, acc_t) in acc.iter_mut().enumerate() {
                    let (trow, tcol) = (tid / (BN / TN), tid % (BN / TN));
                    for p in 0..len {
                        let reg_m: [f32; TM] =
                            core::array::from_fn(|i| as_[(trow * TM + i) * BK + p]);
                        let reg_n: [f32; TN] =
                            core::array::from_fn(|j| bs_[p * BN + tcol * TN + j]);
                        for i in 0..TM {
                            for j in 0..TN {
                                acc_t[i * TN + j] =
                                    fma_f32_ftz(reg_m[i], reg_n[j], acc_t[i * TN + j]);
                            }
                        }
                    }
                }
                // barrier()
                t += BK;
            }
            for (tid, acc_t) in acc.iter().enumerate() {
                let (trow, tcol) = (tid / (BN / TN), tid % (BN / TN));
                for i in 0..TM {
                    for j in 0..TN {
                        let (row, col) = (row0 + trow * TM + i, col0 + tcol * TN + j);
                        if row < m && col < n {
                            l.mem.write_u32(
                                c + 4 * (row * n + col) as u64,
                                acc_t[i * TN + j].to_bits(),
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
