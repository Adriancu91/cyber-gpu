//! # SIMULARE — a model of the Ampere command processor (C-20 D, part 1)
//!
//! Reads a GPFIFO, decodes the pushbuffer it points at, and executes the few
//! methods the first compute launch needs:
//! - channel class C56F: `SEM_ADDR_LO/HI`, `SEM_PAYLOAD_LO/HI`, `SEM_EXECUTE`
//!   (release 32/64-bit, acquire ≥ — an acquire that is not yet satisfied is
//!   an error here, since nothing else runs in parallel; I10b: `RELEASE_WFI`
//!   accepted; C-69: `RELEASE_TIMESTAMP` accepted — a 16-byte record,
//!   payload at +0, the *simulated* GPU time in ns at +8; any other bit
//!   refused);
//! - compute class C7C0: `SET_OBJECT`, `SEND_PCAS_A` (QMD address >> 8) and
//!   `SEND_SIGNALING_PCAS2_B` (launch), reading a QMD V03_00; I10b:
//!   `INVALIDATE_SHADER_CACHES_NO_WFI` (fields checked, counted);
//! - I10a: the once-per-channel compute setup — the local and shared memory
//!   windows, the local memory buffer and its per-TPC size — recorded in
//!   [`ComputeState`] (the model does not use them for anything else);
//! - I10a: a channel's ring as the GPU sees it ([`CommandProcessor::run_channel`]):
//!   `GPGet`/`GPPut` from USERD (`AmpereAControlGPFifo`), the entries between
//!   them, then `GPGet` written back.
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
use alloc::collections::btree_map::Entry;
use alloc::vec::Vec;
use core::cell::RefCell;

use gpu_core::error::{GpuError, GpuResult};
use nvidia_rm570::regs::{clc7b5 as ce, clc7c0 as c, clc56f as h, qmd_v03_00 as q};
use nvidia_rm570::rmapi::AmpereAControlGPFifo;
use nvidia_rm570::{Field, QMD_SIZE, QmdField};

use crate::vram::{AnyTranslation, Aperture, SimVram, translate, translate_any};

const PAGE: u64 = 4096;

/// GPU memory as seen through GPU virtual addresses (identity; no MMU yet).
#[derive(Default)]
pub struct GpuMem {
    pages: BTreeMap<u64, Box<[u8; PAGE as usize]>>,
    /// C-139: the optional model of a non-coherent read cache
    /// ([`CommandProcessor::model_stale_reads`]). Off by default.
    stale: RefCell<StaleReads>,
}

/// C-139, SIMULARE of a **hypothesis** (docs/os-305_ANALIZA_I14.md), not of
/// observed hardware: the loads of a running program go through a cache that
/// nothing keeps coherent — a line read once is served again from the cache,
/// whatever was written to memory since (by a program or by the host), until
/// the caches are invalidated.
#[derive(Default)]
struct StaleReads {
    /// Bytes per cache line (a power of two up to a page); 0 = the model is
    /// off and memory is coherent, as it always was.
    line: u64,
    /// A program is running: only its loads go through the cache.
    in_program: bool,
    lines: BTreeMap<u64, Vec<u8>>,
    /// Bytes served from the cache that differed from memory at that time.
    stale_bytes: u64,
}

impl GpuMem {
    pub fn write(&mut self, va: u64, data: &[u8]) {
        // Page by page (one map lookup per page, not per byte); the same bytes land where they did.
        let mut done = 0usize;
        while done < data.len() {
            let a = va + done as u64;
            let at = (a % PAGE) as usize;
            let n = (PAGE as usize - at).min(data.len() - done);
            let page = self
                .pages
                .entry(a / PAGE)
                .or_insert_with(|| Box::new([0; PAGE as usize]));
            page[at..at + n].copy_from_slice(&data[done..done + n]);
            done += n;
        }
    }

    fn byte(&self, a: u64) -> GpuResult<u8> {
        self.pages
            .get(&(a / PAGE))
            .map(|p| p[(a % PAGE) as usize])
            .ok_or(GpuError::InvalidArgument(
                "SIM: citire din memorie GPU nemapata",
            ))
    }

    /// Reading memory that was never written is an error (a real GPU would
    /// fault on an unmapped address). With the stale-read model on and a
    /// program running, the bytes come from the model's cache lines.
    pub fn read(&self, va: u64, out: &mut [u8]) -> GpuResult<()> {
        let mut st = self.stale.borrow_mut();
        if st.line == 0 || !st.in_program {
            // Page by page; a page never written is the same error as before.
            let mut done = 0usize;
            while done < out.len() {
                let a = va + done as u64;
                let at = (a % PAGE) as usize;
                let n = (PAGE as usize - at).min(out.len() - done);
                let page = self
                    .pages
                    .get(&(a / PAGE))
                    .ok_or(GpuError::InvalidArgument(
                        "SIM: citire din memorie GPU nemapata",
                    ))?;
                out[done..done + n].copy_from_slice(&page[at..at + n]);
                done += n;
            }
            return Ok(());
        }
        let line = st.line;
        for (i, b) in out.iter_mut().enumerate() {
            let a = va + i as u64;
            let base = a - a % line;
            let now = self.byte(a)?;
            let at = (a - base) as usize;
            let cached = match st.lines.entry(base) {
                Entry::Occupied(l) => l.get()[at],
                Entry::Vacant(v) => {
                    let fetched = (0..line)
                        .map(|k| self.byte(base + k))
                        .collect::<GpuResult<Vec<u8>>>()?;
                    v.insert(fetched)[at]
                }
            };
            if cached != now {
                st.stale_bytes += 1;
            }
            *b = cached;
        }
        Ok(())
    }

    /// C-139: marks the start/end of a program run for the stale-read model;
    /// returns the previous state.
    fn set_in_program(&self, on: bool) -> bool {
        core::mem::replace(&mut self.stale.borrow_mut().in_program, on)
    }

    /// C-139: what an invalidation of the shader data caches does in the model.
    fn drop_cached_lines(&self) {
        self.stale.borrow_mut().lines.clear();
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

    /// I10a: copies `len` bytes at GPU VA `va` (8-byte aligned) out of a
    /// simulated VRAM, through the page tables at `root`, read by the
    /// independent walker [`translate`] — what the GPU would fetch.
    pub fn load_from_vram(
        &mut self,
        vram: &SimVram,
        root: u64,
        va: u64,
        len: u64,
    ) -> GpuResult<()> {
        if !va.is_multiple_of(8) || !len.is_multiple_of(8) {
            return Err(bad("SIM: copiere din VRAM nealiniata"));
        }
        for off in (0..len).step_by(8) {
            let pa = translate(vram, root, va + off)
                .map_err(|_| bad("SIM: adresa GPU fara traducere"))?
                .pa;
            let v = vram.read_u64(pa).map_err(bad)?;
            self.write(va + off, &v.to_le_bytes());
        }
        Ok(())
    }

    /// I10a: the reverse — what the GPU wrote (e.g. `GPGet`) back into the
    /// simulated VRAM, through the same page tables.
    pub fn store_to_vram(&self, vram: &mut SimVram, root: u64, va: u64, len: u64) -> GpuResult<()> {
        if !va.is_multiple_of(8) || !len.is_multiple_of(8) {
            return Err(bad("SIM: copiere in VRAM nealiniata"));
        }
        for off in (0..len).step_by(8) {
            let pa = translate(vram, root, va + off)
                .map_err(|_| bad("SIM: adresa GPU fara traducere"))?
                .pa;
            vram.write_u64(pa, self.read_u64(va + off)?).map_err(bad)?;
        }
        Ok(())
    }
}

impl GpuMem {
    /// C-83 (I11): like [`Self::load_from_vram`], but a page may also be in
    /// host RAM (`sys`, a second [`SimVram`] used as the model of the
    /// OS's DMA memory, indexed by bus address), found by [`translate_any`].
    pub fn load_from(
        &mut self,
        vram: &SimVram,
        sys: &SimVram,
        root: u64,
        va: u64,
        len: u64,
    ) -> GpuResult<()> {
        if !va.is_multiple_of(8) || !len.is_multiple_of(8) {
            return Err(bad("SIM: copiere nealiniata"));
        }
        let mut tlb = Tlb::default();
        for off in (0..len).step_by(8) {
            let t = tlb.get(vram, root, va + off)?;
            let v = match t.aperture {
                Aperture::Vram => vram.read_u64(t.addr),
                _ => sys.read_u64(t.addr),
            }
            .map_err(bad)?;
            self.write(va + off, &v.to_le_bytes());
        }
        Ok(())
    }

    /// C-83 (I11): the reverse of [`Self::load_from`]; a read-only page is
    /// refused (the GPU would fault).
    pub fn store_to(
        &self,
        vram: &mut SimVram,
        sys: &mut SimVram,
        root: u64,
        va: u64,
        len: u64,
    ) -> GpuResult<()> {
        if !va.is_multiple_of(8) || !len.is_multiple_of(8) {
            return Err(bad("SIM: copiere nealiniata"));
        }
        let mut tlb = Tlb::default();
        for off in (0..len).step_by(8) {
            let t = tlb.get(vram, root, va + off)?;
            if t.read_only {
                return Err(bad("SIM: scriere GPU intr-o pagina doar-citire"));
            }
            let v = self.read_u64(va + off)?;
            match t.aperture {
                Aperture::Vram => vram.write_u64(t.addr, v),
                _ => sys.write_u64(t.addr, v),
            }
            .map_err(bad)?;
        }
        Ok(())
    }
}

/// The last 4 KiB page [`translate_any`] walked, reused for the next addresses of the same page (every page size
/// the MMU maps is a multiple of 4 KiB, so the page's bytes are contiguous at the other end too). Only within one
/// `load_from` / `store_to`: the page tables cannot change in between.
#[derive(Default)]
struct Tlb {
    last: Option<(u64, AnyTranslation)>,
}

impl Tlb {
    fn get(&mut self, vram: &SimVram, root: u64, va: u64) -> GpuResult<AnyTranslation> {
        let page = va & !(PAGE - 1);
        if let Some((p, t)) = self.last
            && p == page
        {
            return Ok(AnyTranslation {
                addr: t.addr + (va - page),
                ..t
            });
        }
        let t = translate_any(vram, root, va).map_err(|_| bad("SIM: adresa GPU fara traducere"))?;
        self.last = Some((
            page,
            AnyTranslation {
                addr: t.addr - (va - page),
                ..t
            },
        ));
        Ok(t)
    }
}

/// I10a: what the compute setup methods left in the model (`None` = never
/// set). Addresses are whole 64-bit values: `_A` (upper) then `_B` (lower).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ComputeState {
    pub local_window: Option<u64>,
    pub shared_window: Option<u64>,
    pub local_memory: Option<u64>,
    /// `SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A/B` and `_C` (max SM count).
    pub local_per_tpc: Option<u64>,
    pub local_max_sm: Option<u32>,
}

/// `USERD` offsets, from the generated `AmpereAControlGPFifo`.
pub const USERD_GP_GET: u64 = core::mem::offset_of!(AmpereAControlGPFifo, GPGet) as u64;
pub const USERD_GP_PUT: u64 = core::mem::offset_of!(AmpereAControlGPFifo, GPPut) as u64;

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
    SetObject {
        subch: u8,
        class: u32,
    },
    Release {
        addr: u64,
        payload: u64,
    },
    Acquire {
        addr: u64,
        at_least: u64,
    },
    Launch {
        qmd: u64,
        program: usize,
    },
    /// C-157 (RAM ↔ VRAM): one copy of the copy engine (`AMPERE_DMA_COPY_B`).
    CeCopy {
        src: u64,
        dst: u64,
        bytes: u64,
        pipelined: bool,
    },
}

/// C-157 (RAM ↔ VRAM), SIMULARE: the methods of the copy engine object seen so far (`clc7b5.h`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CeState {
    pub offset_in: u64,
    pub offset_out: u64,
    pub line_length: u32,
    pub sem_addr: u64,
    pub sem_payload: u32,
}

pub struct CommandProcessor {
    pub mem: GpuMem,
    programs: Vec<(Vec<u8>, ProgramFn, Needs)>,
    objects: [u32; 8],
    sem_addr: u64,
    sem_payload: u64,
    pcas_qmd: Option<u64>,
    pub events: Vec<Event>,
    /// I10a: the compute setup seen so far.
    pub compute: ComputeState,
    /// I10a: an `_A` (upper half) waiting for its `_B`, by method.
    upper: Option<(u32, u32)>,
    /// I10a: GPFIFO entries taken from rings by [`Self::run_channel`].
    pub entries_taken: u32,
    /// I10b: `INVALIDATE_SHADER_CACHES_NO_WFI` seen (the model has no
    /// caches; it only checks the fields).
    pub cache_invalidations: u32,
    /// C-69, SIMULARE: the model's GPU clock (ns), written by a timestamp
    /// release. It is *not* a measurement: it advances only by the costs
    /// below, which the test chooses.
    pub gpu_time_ns: u64,
    /// C-69, SIMULARE: simulated cost of one launch (added at each launch).
    pub sim_launch_ns: u64,
    /// C-69, SIMULARE: simulated cost of one method (added at each method).
    pub sim_method_ns: u64,
    /// C-69: `(addr, payload, time_ns)` of every timestamp release.
    pub timestamps: Vec<(u64, u64, u64)>,
    /// C-157, SIMULARE: the copy engine's method state.
    pub ce: CeState,
    /// C-157, SIMULARE: the copy engine's simulated speed (MB/s = bytes per µs; 0 = copies take no time). Pipelined
    /// copies are NOT overlapped by the model: it runs one copy after the other.
    pub sim_ce_mb_s: u64,
    /// C-157, SIMULARE: simulated fixed cost of one copy (ns).
    pub sim_ce_copy_ns: u64,
    /// C-165 (`docs/C-165_OCUPAREA.md`), SIMULARE: `[MIN, TARGET, MAX]` of the QMD's SM shared-memory step, one entry
    /// per launch that set a step (QMD units: KiB / 4 + 1). The model has no occupancy: it only checks and records.
    pub sm_configs: Vec<[u32; 3]>,
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
            compute: ComputeState::default(),
            upper: None,
            entries_taken: 0,
            cache_invalidations: 0,
            gpu_time_ns: 0,
            sim_launch_ns: 0,
            sim_method_ns: 0,
            timestamps: Vec::new(),
            ce: CeState::default(),
            sim_ce_mb_s: 0,
            sim_ce_copy_ns: 0,
            sm_configs: Vec::new(),
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

    /// C-139, SIMULARE of a **hypothesis** raised by os-305 (I13 passed on the
    /// board, I14 did not): the loads of a program go through a read cache of
    /// `line_bytes`-byte lines that is **not coherent** with later writes and
    /// is emptied only by `INVALIDATE_SHADER_CACHES_NO_WFI` (global data or
    /// constant) or by a QMD that asks for it
    /// (`INVALIDATE_SHADER_DATA_CACHE`). `0` turns the model off (the
    /// default: coherent memory). It is our reading of why every working
    /// stack invalidates before each launch; nothing here was observed on
    /// the board.
    pub fn model_stale_reads(&mut self, line_bytes: u64) -> GpuResult<()> {
        if line_bytes != 0 && (!line_bytes.is_power_of_two() || line_bytes > PAGE) {
            return Err(bad(
                "SIM: linia de cache trebuie sa fie putere a lui 2, cel mult o pagina",
            ));
        }
        let mut st = self.mem.stale.borrow_mut();
        // The same size again keeps the lines: only an invalidation empties
        // the cache, not a new doorbell.
        if st.line != line_bytes {
            st.line = line_bytes;
            st.lines.clear();
        }
        Ok(())
    }

    /// C-139: bytes the stale-read model served that differed from memory
    /// (0 = the model changed nothing, or is off).
    pub fn stale_bytes_served(&self) -> u64 {
        self.mem.stale.borrow().stale_bytes
    }

    /// I10a: a channel as the GPU sees it after the doorbell: reads `GPGet`
    /// and `GPPut` from USERD at `userd_va`, runs the ring entries from GET
    /// up to PUT (wrapping), then writes `GPGet = GPPut` back. A PUT outside
    /// the ring is an error (the host broke the protocol). Returns the
    /// number of entries taken.
    pub fn run_channel(&mut self, ring_va: u64, entries: u32, userd_va: u64) -> GpuResult<u32> {
        if entries < 2 || !entries.is_power_of_two() {
            return Err(bad("SIM: inel GPFIFO cu numar de intrari invalid"));
        }
        let get = self.mem.read_u32(userd_va + USERD_GP_GET)?;
        let put = self.mem.read_u32(userd_va + USERD_GP_PUT)?;
        if get >= entries || put >= entries {
            return Err(bad("SIM: GPGet/GPPut in afara inelului"));
        }
        let mut list = Vec::new();
        let mut i = get;
        while i != put {
            let a = ring_va + 8 * i as u64;
            list.push((self.mem.read_u32(a)?, self.mem.read_u32(a + 4)?));
            i = (i + 1) & (entries - 1);
        }
        self.run_gpfifo(&list)?;
        self.mem.write_u32(userd_va + USERD_GP_GET, put);
        self.entries_taken += list.len() as u32;
        Ok(list.len() as u32)
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
        self.gpu_time_ns += self.sim_method_ns;
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
        if self.objects[subch as usize] == ce::AMPERE_DMA_COPY_B {
            return self.ce_method(m, d);
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
            c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI => {
                let known = [
                    c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION,
                    c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA,
                    c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT,
                ];
                if known.iter().fold(d, |r, f| f.set(r, 0)) != 0 {
                    return Err(bad("SIM: biti nemodelati in INVALIDATE_SHADER_CACHES"));
                }
                self.cache_invalidations += 1;
                // C-139: the stale-read model (if on) forgets its lines.
                if c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA.get(d) != 0
                    || c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT.get(d) != 0
                {
                    self.mem.drop_cached_lines();
                }
                Ok(())
            }
            m => self.setup_method(m, d),
        }
    }

    /// I10a: the `_A`/`_B` address pairs of the compute setup. An `_A` is
    /// kept until its `_B`; bits outside a field are refused (the hardware
    /// would drop them and the address would be wrong).
    fn setup_method(&mut self, m: u32, d: u32) -> GpuResult<()> {
        let pairs: [(u32, Field, u32); 4] = [
            (
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A,
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A_BASE_ADDRESS_UPPER,
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_B,
            ),
            (
                c::NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A,
                c::NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A_BASE_ADDRESS_UPPER,
                c::NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_B,
            ),
            (
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_A,
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_A_ADDRESS_UPPER,
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_B,
            ),
            (
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A,
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A_SIZE_UPPER,
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_B,
            ),
        ];
        if let Some(&(a, f, _)) = pairs.iter().find(|p| p.0 == m) {
            if f.get(d) << f.lo != d {
                return Err(bad("SIM: biti in afara campului _A"));
            }
            self.upper = Some((a, f.get(d)));
            return Ok(());
        }
        if let Some(&(a, _, _)) = pairs.iter().find(|p| p.2 == m) {
            let hi = match self.upper.take() {
                Some((pa, hi)) if pa == a => hi,
                _ => return Err(bad("SIM: _B fara _A-ul lui")),
            };
            let v = Some(((hi as u64) << 32) | d as u64);
            let st = &mut self.compute;
            match a {
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A => st.local_window = v,
                c::NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A => st.shared_window = v,
                c::NVC7C0_SET_SHADER_LOCAL_MEMORY_A => st.local_memory = v,
                _ => st.local_per_tpc = v,
            }
            return Ok(());
        }
        if m == c::NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C {
            let f = c::NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C_MAX_SM_COUNT;
            if f.get(d) << f.lo != d {
                return Err(bad("SIM: biti in afara campului MAX_SM_COUNT"));
            }
            self.compute.local_max_sm = Some(f.get(d));
            return Ok(());
        }
        Err(bad("SIM: metoda de calcul nemodelata"))
    }

    /// C-157 (RAM ↔ VRAM): the copy engine methods a one-line copy needs (`clc7b5.h`, tinygrad's copy as the
    /// specification): source / destination VAs, the length, the semaphore, `LAUNCH_DMA`. Any bit of `LAUNCH_DMA`
    /// the model does not know is refused (block-linear layouts, several lines, remap, physical addresses,
    /// interrupts, reductions).
    fn ce_method(&mut self, m: u32, d: u32) -> GpuResult<()> {
        let upper = |f: Field| -> GpuResult<u64> {
            if f.get(d) << f.lo != d {
                return Err(bad("SIM: biti in afara campului UPPER (copiere)"));
            }
            Ok((f.get(d) as u64) << 32)
        };
        let st = &mut self.ce;
        match m {
            ce::NVC7B5_OFFSET_IN_UPPER => {
                st.offset_in =
                    upper(ce::NVC7B5_OFFSET_IN_UPPER_UPPER)? | (st.offset_in & 0xffff_ffff)
            }
            ce::NVC7B5_OFFSET_IN_LOWER => st.offset_in = (st.offset_in & !0xffff_ffff) | d as u64,
            ce::NVC7B5_OFFSET_OUT_UPPER => {
                st.offset_out =
                    upper(ce::NVC7B5_OFFSET_OUT_UPPER_UPPER)? | (st.offset_out & 0xffff_ffff)
            }
            ce::NVC7B5_OFFSET_OUT_LOWER => {
                st.offset_out = (st.offset_out & !0xffff_ffff) | d as u64
            }
            ce::NVC7B5_LINE_LENGTH_IN => st.line_length = d,
            ce::NVC7B5_SET_SEMAPHORE_A => {
                st.sem_addr = upper(ce::NVC7B5_SET_SEMAPHORE_A_UPPER)? | (st.sem_addr & 0xffff_ffff)
            }
            ce::NVC7B5_SET_SEMAPHORE_B => st.sem_addr = (st.sem_addr & !0xffff_ffff) | d as u64,
            ce::NVC7B5_SET_SEMAPHORE_PAYLOAD => st.sem_payload = d,
            ce::NVC7B5_LAUNCH_DMA => return self.ce_launch(d),
            _ => return Err(bad("SIM: metoda de copiere nemodelata")),
        }
        Ok(())
    }

    fn ce_launch(&mut self, d: u32) -> GpuResult<()> {
        let known = [
            ce::NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE,
            ce::NVC7B5_LAUNCH_DMA_FLUSH_ENABLE,
            ce::NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE,
            ce::NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT,
            ce::NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT,
        ];
        if known.iter().fold(d, |r, f| f.set(r, 0)) != 0 {
            return Err(bad("SIM: biti nemodelati in LAUNCH_DMA"));
        }
        let pitch = ce::NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH;
        if ce::NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT.get(d) != pitch
            || ce::NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT.get(d)
                != ce::NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH
        {
            return Err(bad("SIM: copiere block-linear nemodelata"));
        }
        let sem = ce::NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE.get(d);
        let four = ce::NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE;
        if sem != ce::NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE_NONE && sem != four {
            return Err(bad("SIM: semafor de copiere nemodelat"));
        }
        let st = self.ce;
        let pipelined = match ce::NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE.get(d) {
            ce::NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_PIPELINED => true,
            ce::NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED => false,
            _ => return Err(bad("SIM: LAUNCH_DMA fara copiere (nemodelat)")),
        };
        let bytes = st.line_length as u64;
        if bytes == 0 {
            return Err(bad("SIM: copiere de 0 octeti"));
        }
        let (se, de) = (st.offset_in + bytes, st.offset_out + bytes);
        if st.offset_in < de && st.offset_out < se {
            return Err(bad("SIM: copiere cu sursa si destinatia suprapuse"));
        }
        let mut buf = alloc::vec![0u8; bytes as usize];
        self.mem.read(st.offset_in, &mut buf)?;
        self.mem.write(st.offset_out, &buf);
        self.events.push(Event::CeCopy {
            src: st.offset_in,
            dst: st.offset_out,
            bytes,
            pipelined,
        });
        self.gpu_time_ns += self.sim_ce_copy_ns;
        if self.sim_ce_mb_s != 0 {
            self.gpu_time_ns += (bytes as u128 * 1000 / self.sim_ce_mb_s as u128) as u64;
        }
        if sem == four {
            // DEDUS: the four-word release = payload at +0, the GPU time at +8 (as the channel's timestamp record).
            if ce::NVC7B5_LAUNCH_DMA_FLUSH_ENABLE.get(d) != ce::NVC7B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE
            {
                return Err(bad(
                    "SIM: semafor de copiere fara FLUSH_ENABLE (datele pot sosi dupa el)",
                ));
            }
            if !st.sem_addr.is_multiple_of(16) {
                return Err(bad("SIM: semafor de copiere nealiniat la 16"));
            }
            let mut rec = [0u8; 16];
            rec[..4].copy_from_slice(&st.sem_payload.to_le_bytes());
            rec[8..].copy_from_slice(&self.gpu_time_ns.to_le_bytes());
            self.mem.write(st.sem_addr, &rec);
            self.timestamps
                .push((st.sem_addr, st.sem_payload as u64, self.gpu_time_ns));
            self.events.push(Event::Release {
                addr: st.sem_addr,
                payload: st.sem_payload as u64,
            });
        }
        Ok(())
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
                // I10b: only the fields the model knows; a timestamp release
                // (16-byte record) is not modelled. `RELEASE_WFI` needs
                // nothing here: the model runs one method at a time.
                let known = [
                    h::NVC56F_SEM_EXECUTE_OPERATION,
                    h::NVC56F_SEM_EXECUTE_ACQUIRE_SWITCH_TSG,
                    h::NVC56F_SEM_EXECUTE_RELEASE_WFI,
                    h::NVC56F_SEM_EXECUTE_PAYLOAD_SIZE,
                    h::NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP,
                ];
                if known.iter().fold(d, |r, f| f.set(r, 0)) != 0 {
                    return Err(bad("SIM: biti nemodelati in SEM_EXECUTE"));
                }
                let wide = h::NVC56F_SEM_EXECUTE_PAYLOAD_SIZE.get(d)
                    == h::NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT;
                if wide && !self.sem_addr.is_multiple_of(8) {
                    return Err(bad("SIM: semafor pe 64 de biti nealiniat la 8"));
                }
                // C-69: the 16-byte record (payload, then the time at +8;
                // layout DEDUS, to be confirmed on the board: value at 0,
                // timestamp at 8, as tinygrad's signal). Only on a release,
                // aligned to 16.
                let stamp = h::NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP.get(d)
                    == h::NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN;
                if stamp {
                    if h::NVC56F_SEM_EXECUTE_OPERATION.get(d)
                        != h::NVC56F_SEM_EXECUTE_OPERATION_RELEASE
                    {
                        return Err(bad("SIM: marca de timp doar la eliberare"));
                    }
                    if !self.sem_addr.is_multiple_of(16) {
                        return Err(bad("SIM: semafor cu marca de timp nealiniat la 16"));
                    }
                    self.mem
                        .write(self.sem_addr + 8, &self.gpu_time_ns.to_le_bytes());
                    let p = if wide {
                        self.sem_payload
                    } else {
                        self.sem_payload & 0xffff_ffff
                    };
                    self.timestamps.push((self.sem_addr, p, self.gpu_time_ns));
                }
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
        // C-165: the SM shared-memory step. 0 everywhere = a QMD without shared memory (`ComputeQmd::encode`).
        let sm = [
            g(q::NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE),
            g(q::NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE),
            g(q::NVC6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE),
        ];
        if sm != [0, 0, 0] {
            if sm[0] > sm[1] || sm[1] > sm[2] {
                return Err(bad(
                    "SIM: QMD cu MIN > TARGET sau TARGET > MAX la treapta SM",
                ));
            }
            // Units are KiB / 4 + 1 (Mesa NAK `gv100_sm_config_smem_size`).
            let target_bytes = sm[1].saturating_sub(1) * 4096;
            if target_bytes < g(q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE) {
                return Err(bad(
                    "SIM: treapta SM e mai mica decat memoria comuna a blocului",
                ));
            }
        }
        if g(q::NVC6C0_QMDV03_00_BARRIER_COUNT) < needs.barriers {
            return Err(bad(
                "SIM: QMD declara mai putine bariere decat foloseste programul",
            ));
        }
        // C-139: a QMD may ask for the shader data cache to be invalidated
        // before its launch (tinygrad's QMDs do; ours do not set it yet).
        if g(q::NVC6C0_QMDV03_00_INVALIDATE_SHADER_DATA_CACHE)
            == q::NVC6C0_QMDV03_00_INVALIDATE_SHADER_DATA_CACHE_TRUE
        {
            self.mem.drop_cached_lines();
        }
        let was = self.mem.set_in_program(true);
        let ran = run(&mut Launch {
            mem: &mut self.mem,
            cbuf0,
            grid,
            block,
        });
        self.mem.set_in_program(was);
        ran?;
        if sm != [0, 0, 0] {
            self.sm_configs.push(sm);
        }
        self.gpu_time_ns += self.sim_launch_ns;
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

/// C-160 point 4: what `tools/nak/out/mm_train.sm86.sass` needs — the same two
/// 16×16 f32 tiles and one barrier as step 3.
pub const MM_TRAIN_NEEDS: Needs = MATMUL_SMEM_NEEDS;

/// `mm_train` flags (cbuf0 0x54): A stored transposed, B stored transposed, add bias.
pub const MM_TRAIN_TA: u32 = 1;
pub const MM_TRAIN_TB: u32 = 2;
pub const MM_TRAIN_BIAS: u32 = 4;

/// The CPU stand-in for C-160 point 4's training matmul (`mm_train.sm86.sass`):
/// `C = op(A)·op(B) (+ bias)`, the three products of a `Linear` layer
/// (forward NT + bias, dX = NN, dW = TN). It follows the kernel's algorithm
/// like [`matmul_smem_on_cpu`] (tiles of op(A) and op(B) filled as the
/// threads would, zeros outside, `ffma.ftz` chain over min(16, K − t)
/// entries), then `fadd.ftz` of `bias[col]`. cbuf0: a/b/c/bias at
/// 0x28/0x30/0x38/0x40, m/n/k/flags at 0x48/0x4c/0x50/0x54 (NAK dump).
/// Unknown flag bits are refused: the program would ignore them, a caller
/// that sets them has a bug.
pub fn mm_train_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_train_blocks(l, MmEpilogue::Bias)
}

/// Order 20:11 step 3, second part: what `tools/nak/out/mm_train_gelu.sm86.sass` needs — `mm_train`'s two tiles and
/// one barrier.
pub const MM_TRAIN_GELU_NEEDS: Needs = MM_TRAIN_NEEDS;

/// The CPU stand-in for `mm_train_gelu.sm86.sass` (order 20:11 step 3, `docs/C-160_P7_LIPIRE_GELU.md`): the loop of
/// [`mm_train_on_cpu`] unchanged, then, instead of the bias, each element of C is
/// `gpu_compute::layer::gelu_tr_gpu`(u[row, col], Some(acc)) — `gelu_tr` mode 1 with acc in place of g. cbuf0 as
/// `mm_train` with `u` in the bias slot (0x40); the BIAS flag is refused (the program has no bias). A launch whose C
/// overlaps u is refused (other blocks would read u already overwritten).
pub fn mm_train_gelu_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_train_blocks(
        l,
        MmEpilogue::GeluBack(gpu_compute::layer::gelu_tr_gpu_back),
    )
}

/// [`mm_train_gelu_on_cpu`] with `gpu_compute::layer::gelu_tr_back_unfused_gpu` (Φ + u·φ with two roundings, no
/// FFMA): what the program would do if NAK had compiled its epilogue unlike `gelu_tr`'s. Not a model of our program —
/// T201 registers it to show the bit-for-bit comparison catches that difference.
pub fn mm_train_gelu_unfused_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_train_blocks(
        l,
        MmEpilogue::GeluBack(gpu_compute::layer::gelu_tr_back_unfused_gpu),
    )
}

/// C-157 p.2x (`docs/C-157_P2x_MM_TRAIN_TILE.md`): what `tools/nak/out/mm_train_tile.sm86.sass` needs — the 16×64
/// tile of op(A) and the 16×64 tile of op(B) (8 KiB) and one barrier.
pub const MM_TRAIN_TILE_NEEDS: Needs = Needs {
    shared_bytes: 2 * 16 * 64 * 4,
    barriers: 1,
};

/// The CPU stand-in for `mm_train_tile.sm86.sass`: `mm_train`'s job and cbuf0 (a/b/c/bias at 0x28/0x30/0x38/0x40,
/// m/n/k/flags at 0x48/0x4c/0x50/0x54) on blocks of 256 threads, each block a 64×64 piece of C; thread
/// (tx = tid % 16, ty = tid / 16) owns rows `ty + 16i`, columns `tx + 16j`. Per K-tile of 16 the "shared" tiles
/// `as_[p][r]` / `bs_[p][c]` hold op(A) / op(B) (zeros outside), then for p < min(16, K − t) the thread copies 4 + 4
/// values to "registers" and does its 16 `ffma.ftz`; finally `fadd.ftz` of `bias[col]` with BIAS. Same chains as
/// [`mm_train_on_cpu`], so the same bits; oracle `reference::matmul_t_fma_ftz`.
pub fn mm_train_tile_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_train_tile_blocks(l, MmEpilogue::Bias)
}

/// C-165 follow-up (`docs/C-165_MM_TILE_GELU.md`): what `tools/nak/out/mm_train_tile_gelu.sm86.sass` needs — as
/// `mm_train_tile` (the two 16×64 tiles, 8 KiB, one barrier).
pub const MM_TRAIN_TILE_GELU_NEEDS: Needs = MM_TRAIN_TILE_NEEDS;

/// The CPU stand-in for `mm_train_tile_gelu.sm86.sass`: the loads, tiles and `ffma.ftz` chains of
/// [`mm_train_tile_on_cpu`] unchanged, then each element of C is `gpu_compute::layer::gelu_tr_gpu_back`(u[row, col],
/// acc) — the epilogue of [`mm_train_gelu_on_cpu`]. cbuf0 as `mm_train` with `u` at the bias slot (0x40); the BIAS
/// flag is refused, and so is a C overlapping u. Same bits as [`mm_train_gelu_on_cpu`].
pub fn mm_train_tile_gelu_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_train_tile_blocks(
        l,
        MmEpilogue::GeluBack(gpu_compute::layer::gelu_tr_gpu_back),
    )
}

fn mm_train_tile_blocks(l: &mut Launch<'_>, epi: MmEpilogue) -> GpuResult<()> {
    const B: usize = 64;
    const BK: usize = 16;
    const TM: usize = 4;
    const THREADS: usize = 256;
    if l.block != [THREADS as u32, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: mm_train_tile cere blocuri de 256x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let a = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let b = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let c = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let bias_va = l.mem.read_u64(l.cbuf0 + 0x40)?;
    let m = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let n = l.mem.read_u32(l.cbuf0 + 0x4c)? as usize;
    let k = l.mem.read_u32(l.cbuf0 + 0x50)? as usize;
    let flags = l.mem.read_u32(l.cbuf0 + 0x54)?;
    if flags & !(MM_TRAIN_TA | MM_TRAIN_TB | MM_TRAIN_BIAS) != 0 {
        return Err(bad("SIM: mm_train_tile: biti de optiuni necunoscuti"));
    }
    let (ta, tb) = (flags & MM_TRAIN_TA != 0, flags & MM_TRAIN_TB != 0);
    let x = l.mem.read_f32s(a, m * k)?;
    let y = l.mem.read_f32s(b, k * n)?;
    let gelu = match epi {
        MmEpilogue::Bias => None,
        MmEpilogue::GeluBack(f) => {
            if flags & MM_TRAIN_BIAS != 0 {
                return Err(bad("SIM: mm_train_tile_gelu nu are bias"));
            }
            let bytes = 4 * (m * n) as u64;
            if c < bias_va + bytes && bias_va < c + bytes {
                return Err(bad("SIM: mm_train_tile_gelu: C se suprapune cu u"));
            }
            Some((f, l.mem.read_f32s(bias_va, m * n)?))
        }
    };
    let bias = if gelu.is_none() && flags & MM_TRAIN_BIAS != 0 {
        Some(l.mem.read_f32s(bias_va, n)?)
    } else {
        None
    };
    let op_a = |r: usize, p: usize| if ta { x[p * m + r] } else { x[r * k + p] };
    let op_b = |p: usize, col: usize| if tb { y[col * k + p] } else { y[p * n + col] };
    let fma = gpu_compute::math::fma_f32_ftz;
    for gy in 0..l.grid[1] as usize {
        for gx in 0..l.grid[0] as usize {
            let (row0, col0) = (gy * B, gx * B);
            let mut acc = alloc::vec![[0.0f32; TM * TM]; THREADS];
            let mut t = 0;
            while t < k {
                let mut as_ = [0.0f32; BK * B];
                let mut bs_ = [0.0f32; BK * B];
                for p in 0..BK {
                    for r in 0..B {
                        if row0 + r < m && t + p < k {
                            as_[p * B + r] = op_a(row0 + r, t + p);
                        }
                        if t + p < k && col0 + r < n {
                            bs_[p * B + r] = op_b(t + p, col0 + r);
                        }
                    }
                }
                // barrier()
                let len = BK.min(k - t);
                for (tid, acc_t) in acc.iter_mut().enumerate() {
                    let (tx, ty) = (tid % 16, tid / 16);
                    for p in 0..len {
                        let reg_m: [f32; TM] = core::array::from_fn(|i| as_[p * B + ty + 16 * i]);
                        let reg_n: [f32; TM] = core::array::from_fn(|j| bs_[p * B + tx + 16 * j]);
                        for i in 0..TM {
                            for j in 0..TM {
                                acc_t[i * TM + j] = fma(reg_m[i], reg_n[j], acc_t[i * TM + j]);
                            }
                        }
                    }
                }
                // barrier()
                t += BK;
            }
            for (tid, acc_t) in acc.iter().enumerate() {
                let (tx, ty) = (tid % 16, tid / 16);
                for i in 0..TM {
                    for j in 0..TM {
                        let (row, col) = (row0 + ty + 16 * i, col0 + tx + 16 * j);
                        if row < m && col < n {
                            let v = acc_t[i * TM + j];
                            let v = match (&bias, &gelu) {
                                (Some(bv), _) => gpu_compute::math::add_f32_ftz(v, bv[col]),
                                (None, Some((f, u))) => f(u[row * n + col], v),
                                (None, None) => v,
                            };
                            l.mem.write_u32(c + 4 * (row * n + col) as u64, v.to_bits());
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// C-165 (`docs/C-165_CURSA_MM.md`): the tile geometry of one race variant of `mm_train_tile`
/// (`tools/nak/mm_tv_<name>.comp` over `tools/nak/mm_train_tv.glsl`). A block of `(bm/tm)·(bn/tn)` threads computes a
/// `bm × bn` piece of C, `tm × tn` results per thread, K in slices of `bk`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmTileGeometry {
    pub bm: u32,
    pub bn: u32,
    pub bk: u32,
    pub tm: u32,
    pub tn: u32,
}

impl MmTileGeometry {
    /// Threads per block (x only).
    pub const fn threads(&self) -> u32 {
        (self.bm / self.tm) * (self.bn / self.tn)
    }
    /// What the program needs from its QMD: the two shared tiles and one barrier.
    pub const fn needs(&self) -> Needs {
        Needs {
            shared_bytes: 4 * self.bk * (self.bm + self.bn),
            barriers: 1,
        }
    }
}

/// C-165: `mm_tv_128x64` — 128 × 64 of C per block, BK 16, 8 × 4 per thread.
pub const MM_TV_128X64: MmTileGeometry = MmTileGeometry {
    bm: 128,
    bn: 64,
    bk: 16,
    tm: 8,
    tn: 4,
};
/// C-165: `mm_tv_64x128` — 64 × 128 of C per block, BK 16, 4 × 8 per thread.
pub const MM_TV_64X128: MmTileGeometry = MmTileGeometry {
    bm: 64,
    bn: 128,
    bk: 16,
    tm: 4,
    tn: 8,
};
/// C-165: `mm_tv_128x128` — 128 × 128 of C per block, BK 8, 8 × 8 per thread.
pub const MM_TV_128X128: MmTileGeometry = MmTileGeometry {
    bm: 128,
    bn: 128,
    bk: 8,
    tm: 8,
    tn: 8,
};
/// C-165: `mm_tv_64x64k8` — 64 × 64 of C per block, BK 8, 4 × 4 per thread.
pub const MM_TV_64X64K8: MmTileGeometry = MmTileGeometry {
    bm: 64,
    bn: 64,
    bk: 8,
    tm: 4,
    tn: 4,
};

/// C-165 (`docs/C-165_DIETA_128.md`): `mm_tv_128x128t512` — the register diet of `mm_tv_128x128`: 128 × 128 of C per
/// block, BK 8, but 512 threads with 8 × 4 results each.
pub const MM_TV_128X128T512: MmTileGeometry = MmTileGeometry {
    bm: 128,
    bn: 128,
    bk: 8,
    tm: 8,
    tn: 4,
};

/// CPU stand-in for `mm_tv_128x64.sm86.sass` (see [`mm_train_tv_on_cpu`]).
pub fn mm_tv_128x64_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_train_tv_on_cpu(l, MM_TV_128X64)
}
/// CPU stand-in for `mm_tv_64x128.sm86.sass` (see [`mm_train_tv_on_cpu`]).
pub fn mm_tv_64x128_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_train_tv_on_cpu(l, MM_TV_64X128)
}
/// CPU stand-in for `mm_tv_128x128.sm86.sass` (see [`mm_train_tv_on_cpu`]).
pub fn mm_tv_128x128_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_train_tv_on_cpu(l, MM_TV_128X128)
}
/// CPU stand-in for `mm_tv_64x64k8.sm86.sass` (see [`mm_train_tv_on_cpu`]).
pub fn mm_tv_64x64k8_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_train_tv_on_cpu(l, MM_TV_64X64K8)
}

/// CPU stand-in for `mm_tv_128x128t512.sm86.sass` (see [`mm_train_tv_on_cpu`]).
pub fn mm_tv_128x128t512_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_train_tv_on_cpu(l, MM_TV_128X128T512)
}

/// C-165: the CPU model of `tools/nak/mm_train_tv.glsl` for geometry `g`: `mm_train`'s job and cbuf0 on blocks of
/// `g.threads()` threads; thread (tx = tid % TX, ty = tid / TX), TX = bn/tn, TY = bm/tm, owns rows `ty + TY·i` and
/// columns `tx + TX·j`. Per K-slice the "shared" tiles hold op(A) / op(B) (zeros outside C), filled element by element
/// as the threads do (`idx = tid + threads·e`, the same mapping as the GLSL); then for p < min(bk, K − t) each thread
/// copies tm + tn values to "registers" and does its tm·tn `ffma.ftz`; finally `fadd.ftz` of `bias[col]` with BIAS.
/// Same chains as [`mm_train_on_cpu`], so the same bits; oracle `reference::matmul_t_fma_ftz`.
pub fn mm_train_tv_on_cpu(l: &mut Launch<'_>, g: MmTileGeometry) -> GpuResult<()> {
    let (bm, bn, bk, tm, tn) = (
        g.bm as usize,
        g.bn as usize,
        g.bk as usize,
        g.tm as usize,
        g.tn as usize,
    );
    let threads = g.threads() as usize;
    let (tx_n, ty_n) = (bn / tn, bm / tm);
    if l.block != [threads as u32, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: mm_train_tv: blocuri gresite pentru varianta"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let a = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let b = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let c = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let bias_va = l.mem.read_u64(l.cbuf0 + 0x40)?;
    let m = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let n = l.mem.read_u32(l.cbuf0 + 0x4c)? as usize;
    let k = l.mem.read_u32(l.cbuf0 + 0x50)? as usize;
    let flags = l.mem.read_u32(l.cbuf0 + 0x54)?;
    if flags & !(MM_TRAIN_TA | MM_TRAIN_TB | MM_TRAIN_BIAS) != 0 {
        return Err(bad("SIM: mm_train_tv: biti de optiuni necunoscuti"));
    }
    let (ta, tb) = (flags & MM_TRAIN_TA != 0, flags & MM_TRAIN_TB != 0);
    let x = l.mem.read_f32s(a, m * k)?;
    let y = l.mem.read_f32s(b, k * n)?;
    let bias = if flags & MM_TRAIN_BIAS != 0 {
        Some(l.mem.read_f32s(bias_va, n)?)
    } else {
        None
    };
    let fma = gpu_compute::math::fma_f32_ftz;
    let mut as_ = alloc::vec![0.0f32; bk * bm];
    let mut bs_ = alloc::vec![0.0f32; bk * bn];
    for gy in 0..l.grid[1] as usize {
        for gx in 0..l.grid[0] as usize {
            let (row0, col0) = (gy * bm, gx * bn);
            let mut acc = alloc::vec![0.0f32; threads * tm * tn];
            let mut t = 0;
            while t < k {
                // The loads, thread by thread, as in the GLSL (each element written exactly once).
                for tid in 0..threads {
                    for e in 0..bk * bm / threads {
                        let idx = tid + threads * e;
                        let (r, p) = if ta {
                            (idx % bm, idx / bm)
                        } else {
                            (idx / bk, idx % bk)
                        };
                        let (ar, ak) = (row0 + r, t + p);
                        as_[p * bm + r] = if ar < m && ak < k {
                            if ta { x[ak * m + ar] } else { x[ar * k + ak] }
                        } else {
                            0.0
                        };
                    }
                    for e in 0..bk * bn / threads {
                        let idx = tid + threads * e;
                        let (q, cc) = if tb {
                            (idx % bk, idx / bk)
                        } else {
                            (idx / bn, idx % bn)
                        };
                        let (bkk, bc) = (t + q, col0 + cc);
                        bs_[q * bn + cc] = if bkk < k && bc < n {
                            if tb { y[bc * k + bkk] } else { y[bkk * n + bc] }
                        } else {
                            0.0
                        };
                    }
                }
                // barrier()
                let len = bk.min(k - t);
                for (tid, acc_t) in acc.chunks_mut(tm * tn).enumerate() {
                    let (tx, ty) = (tid % tx_n, tid / tx_n);
                    for p in 0..len {
                        for i in 0..tm {
                            let am = as_[p * bm + ty + ty_n * i];
                            for j in 0..tn {
                                let bv = bs_[p * bn + tx + tx_n * j];
                                acc_t[i * tn + j] = fma(am, bv, acc_t[i * tn + j]);
                            }
                        }
                    }
                }
                // barrier()
                t += bk;
            }
            for (tid, acc_t) in acc.chunks(tm * tn).enumerate() {
                let (tx, ty) = (tid % tx_n, tid / tx_n);
                for i in 0..tm {
                    for j in 0..tn {
                        let (row, col) = (row0 + ty + ty_n * i, col0 + tx + tx_n * j);
                        if row < m && col < n {
                            let mut v = acc_t[i * tn + j];
                            if let Some(bv) = &bias {
                                v = gpu_compute::math::add_f32_ftz(v, bv[col]);
                            }
                            l.mem.write_u32(c + 4 * (row * n + col) as u64, v.to_bits());
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// What `mm_train` / `mm_train_gelu` do with a finished sum.
#[derive(Clone, Copy)]
enum MmEpilogue {
    /// `mm_train`: `fadd.ftz` of bias[col] with the BIAS flag, nothing otherwise.
    Bias,
    /// `mm_train_gelu`: `f(u[row, col], acc)`, u at the bias slot.
    GeluBack(fn(f32, f32) -> f32),
}

fn mm_train_blocks(l: &mut Launch<'_>, epi: MmEpilogue) -> GpuResult<()> {
    const T: usize = 16;
    if l.block != [16, 16, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: mm_train cere blocuri de 16x16x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let a = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let b = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let c = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let bias_va = l.mem.read_u64(l.cbuf0 + 0x40)?;
    let m = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let n = l.mem.read_u32(l.cbuf0 + 0x4c)? as usize;
    let k = l.mem.read_u32(l.cbuf0 + 0x50)? as usize;
    let flags = l.mem.read_u32(l.cbuf0 + 0x54)?;
    if flags & !(MM_TRAIN_TA | MM_TRAIN_TB | MM_TRAIN_BIAS) != 0 {
        return Err(bad("SIM: mm_train: biti de optiuni necunoscuti"));
    }
    let (ta, tb) = (flags & MM_TRAIN_TA != 0, flags & MM_TRAIN_TB != 0);
    let x = l.mem.read_f32s(a, m * k)?;
    let y = l.mem.read_f32s(b, k * n)?;
    let gelu = match epi {
        MmEpilogue::Bias => None,
        MmEpilogue::GeluBack(f) => {
            if flags & MM_TRAIN_BIAS != 0 {
                return Err(bad("SIM: mm_train_gelu nu are bias"));
            }
            let bytes = 4 * (m * n) as u64;
            if c < bias_va + bytes && bias_va < c + bytes {
                return Err(bad("SIM: mm_train_gelu: C se suprapune cu u"));
            }
            Some((f, l.mem.read_f32s(bias_va, m * n)?))
        }
    };
    let bias = if gelu.is_none() && flags & MM_TRAIN_BIAS != 0 {
        Some(l.mem.read_f32s(bias_va, n)?)
    } else {
        None
    };
    let op_a = |r: usize, p: usize| if ta { x[p * m + r] } else { x[r * k + p] };
    let op_b = |p: usize, col: usize| if tb { y[col * k + p] } else { y[p * n + col] };
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
                            as_[ty][tx] = op_a(row, t + tx);
                        }
                        if t + ty < k && col < n {
                            bs_[ty][tx] = op_b(t + ty, col);
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
                        let v = match (&bias, &gelu) {
                            (Some(bv), _) => gpu_compute::math::add_f32_ftz(v, bv[col]),
                            (None, Some((f, u))) => f(u[row * n + col], v),
                            (None, None) => v,
                        };
                        l.mem.write_u32(c + 4 * (row * n + col) as u64, v.to_bits());
                    }
                }
            }
        }
    }
    Ok(())
}

/// C-160 point 4, program 7: what `tools/nak/out/xent.sm86.sass` needs — 32 bytes of shared memory (8 warp
/// results) and one barrier.
pub const XENT_NEEDS: Needs = Needs {
    shared_bytes: 32,
    barriers: 1,
};

/// The CPU stand-in for C-160 point 4's loss program (`xent.sm86.sass`): one block of 256 per row (grid x = rows);
/// cbuf0 logits/targets/dlogits/loss at 0x28/0x30/0x38/0x40, v at 0x48, scale at 0x4c (NAK dump). Each row is
/// `gpu_compute::layer::xent_gpu256` (the program's order of the sums; `exp`/`ln`/`1/s` correctly rounded, the
/// board's `MUFU` approximate). A target outside the row gives a NaN loss and no one-hot, as the program.
pub fn xent_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad("SIM: xent cere blocuri de 256x1x1, grila = randuri"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (z, tg, d, loss) = (r(l, 0x28)?, r(l, 0x30)?, r(l, 0x38)?, r(l, 0x40)?);
    let v = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let scale = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x4c)?);
    if v == 0 {
        return Ok(()); // the program returns at once
    }
    for row in 0..l.grid[0] as usize {
        let zr = l.mem.read_f32s(z + 4 * (row * v) as u64, v)?;
        let tgt = l.mem.read_u32(tg + 4 * row as u64)?;
        let (dr, lr) = gpu_compute::layer::xent_gpu256(&zr, tgt, scale);
        l.mem.write_f32s(d + 4 * (row * v) as u64, &dr);
        l.mem.write_u32(loss + 4 * row as u64, lr.to_bits());
    }
    Ok(())
}

/// C-160 point 4, program 4 (`dx` half): what `tools/nak/out/ln_back.sm86.sass` needs — 64 bytes of shared memory
/// (the 8 warp results of both sums) and one barrier.
pub const LN_BACK_NEEDS: Needs = Needs {
    shared_bytes: 64,
    barriers: 1,
};

/// The CPU stand-in for C-160 point 4's LayerNorm-backward program (`ln_back.sm86.sass`): one block of 256 per row
/// (grid x = rows); cbuf0 x/dy/γ/μ/rstd/dx at 0x28/0x30/0x38/0x40/0x48/0x50, c at 0x58, 1/c at 0x5c (NAK dump). Each
/// row is `gpu_compute::layer::ln_back_gpu256` (the program's order; only add / mul / fma).
pub fn ln_back_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad("SIM: ln_back cere blocuri de 256x1x1, grila = randuri"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (x, dy, gm, mu, rs, dx) = (
        r(l, 0x28)?,
        r(l, 0x30)?,
        r(l, 0x38)?,
        r(l, 0x40)?,
        r(l, 0x48)?,
        r(l, 0x50)?,
    );
    let c = l.mem.read_u32(l.cbuf0 + 0x58)? as usize;
    let inv_c = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x5c)?);
    if c == 0 {
        return Ok(()); // the program returns at once
    }
    let gamma = l.mem.read_f32s(gm, c)?;
    for row in 0..l.grid[0] as usize {
        let off = 4 * (row * c) as u64;
        let xr = l.mem.read_f32s(x + off, c)?;
        let dyr = l.mem.read_f32s(dy + off, c)?;
        let m = f32::from_bits(l.mem.read_u32(mu + 4 * row as u64)?);
        let s = f32::from_bits(l.mem.read_u32(rs + 4 * row as u64)?);
        let out = gpu_compute::layer::ln_back_gpu256(&xr, &dyr, &gamma, m, s, inv_c);
        l.mem.write_f32s(dx + off, &out);
    }
    Ok(())
}

/// Order 20:11 step 3: what `tools/nak/out/ln_back_res.sm86.sass` needs — the same as [`LN_BACK_NEEDS`].
pub const LN_BACK_RES_NEEDS: Needs = LN_BACK_NEEDS;

/// The CPU stand-in for `ln_back_res.sm86.sass` (order 20:11 step 3, `docs/C-160_P7_LIPIRE_INAPOI.md`): as
/// [`ln_back_on_cpu`], with `dres` at 0x58, c at 0x60, 1/c at 0x64 (NAK dump); each row is
/// `gpu_compute::layer::ln_back_res_gpu256`. All the rows' `dres` are read before any row is written: a launch whose
/// `dx` overlaps `dres` is refused (the board's threads would race), as is one where `dx` overlaps `dy`.
pub fn ln_back_res_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    ln_back_res_rows(l, gpu_compute::layer::ln_back_res_gpu256)
}

/// [`ln_back_res_on_cpu`] with `gpu_compute::layer::ln_back_res_contracted_gpu256` (the last multiplication and the
/// residual addition as one FFMA): what the program would do without `add_exact`. Not a model of our program —
/// T200 registers it to show the bit-for-bit comparison catches that contraction.
pub fn ln_back_res_contracted_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    ln_back_res_rows(l, gpu_compute::layer::ln_back_res_contracted_gpu256)
}

type LnBackResRow = fn(&[f32], &[f32], &[f32], f32, f32, f32, &[f32]) -> Vec<f32>;

fn ln_back_res_rows(l: &mut Launch<'_>, row_fn: LnBackResRow) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad(
            "SIM: ln_back_res cere blocuri de 256x1x1, grila = randuri",
        ));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (x, dy, gm, mu, rs, dx, dres) = (
        r(l, 0x28)?,
        r(l, 0x30)?,
        r(l, 0x38)?,
        r(l, 0x40)?,
        r(l, 0x48)?,
        r(l, 0x50)?,
        r(l, 0x58)?,
    );
    let c = l.mem.read_u32(l.cbuf0 + 0x60)? as usize;
    let inv_c = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x64)?);
    if c == 0 {
        return Ok(()); // the program returns at once
    }
    let rows = l.grid[0] as usize;
    let bytes = 4 * (rows * c) as u64;
    let overlap = |a: u64, b: u64| a < b + bytes && b < a + bytes;
    if overlap(dx, dres) || overlap(dx, dy) {
        return Err(bad("SIM: ln_back_res: dx se suprapune cu dres sau dy"));
    }
    let gamma = l.mem.read_f32s(gm, c)?;
    for row in 0..rows {
        let off = 4 * (row * c) as u64;
        let xr = l.mem.read_f32s(x + off, c)?;
        let dyr = l.mem.read_f32s(dy + off, c)?;
        let dr = l.mem.read_f32s(dres + off, c)?;
        let m = f32::from_bits(l.mem.read_u32(mu + 4 * row as u64)?);
        let s = f32::from_bits(l.mem.read_u32(rs + 4 * row as u64)?);
        let out = row_fn(&xr, &dyr, &gamma, m, s, inv_c, &dr);
        l.mem.write_f32s(dx + off, &out);
    }
    Ok(())
}

/// C-160 point 4, program 3: what `tools/nak/out/ln_fwd.sm86.sass` needs — 32 bytes of shared memory (the 8 warp
/// results, reused by both sums) and one barrier.
pub const LN_FWD_NEEDS: Needs = Needs {
    shared_bytes: 32,
    barriers: 1,
};

/// The CPU stand-in for C-160 point 4's training LayerNorm-forward program (`ln_fwd.sm86.sass`): one block of 256
/// per row (grid x = rows); cbuf0 x/γ/β/y/μ/rstd at 0x28/0x30/0x38/0x40/0x48/0x50, c at 0x58, 1/c at 0x5c, ε at 0x60
/// (NAK dump). Each row is `gpu_compute::layer::ln_fwd_gpu256` (the program's order; `MUFU.RSQ` modelled correctly
/// rounded); y of the row, then μ and rstd of the row.
pub fn ln_fwd_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad("SIM: ln_fwd cere blocuri de 256x1x1, grila = randuri"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (x, gm, bt, y, mu, rs) = (
        r(l, 0x28)?,
        r(l, 0x30)?,
        r(l, 0x38)?,
        r(l, 0x40)?,
        r(l, 0x48)?,
        r(l, 0x50)?,
    );
    let c = l.mem.read_u32(l.cbuf0 + 0x58)? as usize;
    let inv_c = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x5c)?);
    let eps = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x60)?);
    if c == 0 {
        return Ok(()); // the program returns at once
    }
    let gamma = l.mem.read_f32s(gm, c)?;
    let beta = l.mem.read_f32s(bt, c)?;
    for row in 0..l.grid[0] as usize {
        let off = 4 * (row * c) as u64;
        let xr = l.mem.read_f32s(x + off, c)?;
        let (out, m, s) = gpu_compute::layer::ln_fwd_gpu256(&xr, &gamma, &beta, inv_c, eps);
        l.mem.write_f32s(y + off, &out);
        l.mem.write_u32(mu + 4 * row as u64, m.to_bits());
        l.mem.write_u32(rs + 4 * row as u64, s.to_bits());
    }
    Ok(())
}

/// The CPU stand-in for C-160 point 4's training GELU program (`gelu_tr.sm86.sass`, program 5): blocks of 256, one
/// element per thread (grid x = ⌈n/256⌉); cbuf0 u/g/y at 0x28/0x30/0x38, n at 0x40, the mode at 0x44 (0 = forward,
/// anything else = backward, as the program's `isetp.eq`) (NAK dump). Each element is
/// `gpu_compute::layer::gelu_tr_gpu` (the program's order; `MUFU.RCP` / `MUFU.EX2` modelled correctly rounded);
/// threads with i ≥ n write nothing; g is read only in the backward mode.
pub fn gelu_tr_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad("SIM: gelu_tr cere blocuri de 256x1x1, grila 1D"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (u, g, y) = (r(l, 0x28)?, r(l, 0x30)?, r(l, 0x38)?);
    let n = (l.mem.read_u32(l.cbuf0 + 0x40)? as u64).min(l.grid[0] as u64 * 256) as usize;
    let back = l.mem.read_u32(l.cbuf0 + 0x44)? != 0;
    if n == 0 {
        return Ok(());
    }
    let us = l.mem.read_f32s(u, n)?;
    let out: Vec<f32> = if back {
        let gs = l.mem.read_f32s(g, n)?;
        us.iter()
            .zip(&gs)
            .map(|(&ui, &gi)| gpu_compute::layer::gelu_tr_gpu(ui, Some(gi)))
            .collect()
    } else {
        us.iter()
            .map(|&ui| gpu_compute::layer::gelu_tr_gpu(ui, None))
            .collect()
    };
    l.mem.write_f32s(y, &out);
    Ok(())
}

/// The CPU stand-in for C-160 point 4's attention backward (`attn_back.sm86.sass`, program 6): blocks of 32; cbuf0
/// qkv/p/dout/ds/dqkv at 0x28/0x30/0x38/0x40/0x48, T at 0x50, C at 0x54, H at 0x58, d at 0x5c, the mode at 0x60
/// (0 = rows, anything else = columns, as the program's `isetp`), the scale at 0x64 (NAK dump). The batch is the
/// grid's z (the program never reads it). Mode 0, grid [⌈T/32⌉, ≤ H, ≤ B]: every row (b, h, i < min(T, 32·x)) of dS
/// is `gpu_compute::layer::attn_back_row_gpu`; mode 1, grid [≤ H, ≤ T, ≤ B]: every element is `attn_back_col_gpu`.
/// A grid past H / T / B (the program does not guard those) or a d ≠ C / H is refused as not modelled.
pub fn attn_back_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    attn_back_blocks(l, 32)
}

/// The CPU stand-in for C-160 point 7's `attn_back128.sm86.sass` (`docs/C-160_P7_FIR_LIMITAT.md`): the program of
/// [`attn_back_on_cpu`] with blocks of 128 — the same code per thread, so every element is the same
/// `attn_back_row_gpu` / `attn_back_col_gpu`. Mode 0, grid [⌈T/128⌉, ≤ H, ≤ B]: rows i < min(T, 128·x). Mode 1,
/// grid [≤ ⌈H/4⌉, ≤ T, ≤ B]: warp w of block x takes head 4x + w; heads ≥ H leave (the program guards them), so the
/// heads written are h < min(H, 4·x).
pub fn attn_back128_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    attn_back_blocks(l, 128)
}

fn attn_back_blocks(l: &mut Launch<'_>, block: u32) -> GpuResult<()> {
    use gpu_compute::layer::{AttnShape, attn_back_col_gpu, attn_back_row_gpu};
    if l.block != [block, 1, 1] {
        return Err(bad(if block == 32 {
            "SIM: attn_back cere blocuri de 32x1x1"
        } else {
            "SIM: attn_back128 cere blocuri de 128x1x1"
        }));
    }
    let warps = (block / 32) as usize;
    let rows_per_block = block as usize;
    if l.mem.read_u64(l.cbuf0)? != 0 || l.mem.read_u32(l.cbuf0 + 0x8)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0..0x8) diferita de 0, nemodelata",
        ));
    }
    let r64 = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let r32 = |l: &Launch<'_>, o: u64| l.mem.read_u32(l.cbuf0 + o);
    let (qkv, p, dout, ds, dqkv) = (
        r64(l, 0x28)?,
        r64(l, 0x30)?,
        r64(l, 0x38)?,
        r64(l, 0x40)?,
        r64(l, 0x48)?,
    );
    let (t, c, heads, d) = (
        r32(l, 0x50)? as usize,
        r32(l, 0x54)? as usize,
        r32(l, 0x58)? as usize,
        r32(l, 0x5c)? as usize,
    );
    let cols = r32(l, 0x60)? != 0;
    let scale = f32::from_bits(r32(l, 0x64)?);
    let [gx, gy, gz] = l.grid.map(|g| g as usize);
    if heads == 0 || d == 0 || d * heads != c || t == 0 {
        return Err(bad("SIM: attn_back: d != C/H sau forma goala, nemodelat"));
    }
    let (h_n, b_n) = if cols {
        (heads.min(warps * gx), gz)
    } else {
        (gy, gz)
    };
    let h_grid_bad = if cols {
        gx > heads.div_ceil(warps)
    } else {
        gy > heads
    };
    if h_grid_bad || (cols && gy > t) || (!cols && gx > t.div_ceil(rows_per_block)) {
        return Err(bad("SIM: attn_back: grila trece de H / T, nemodelata"));
    }
    if b_n == 0 || h_n == 0 {
        return Ok(());
    }
    let s = AttnShape {
        batch: b_n,
        t,
        c,
        heads,
    };
    let (nq, no, np) = s
        .sizes()
        .ok_or_else(|| bad("SIM: attn_back: forma prea mare"))?;
    let qkv_v = l.mem.read_f32s(qkv, nq)?;
    let p_v = l.mem.read_f32s(p, np)?;
    let dout_v = l.mem.read_f32s(dout, no)?;
    if !cols {
        let rows = t.min(rows_per_block * gx);
        for b in 0..b_n {
            for h in 0..h_n {
                for i in 0..rows {
                    let row = attn_back_row_gpu(&s, &qkv_v, &p_v, &dout_v, scale, b, h, i);
                    let at = (((b * heads + h) * t + i) * t) as u64;
                    l.mem.write_f32s(ds + 4 * at, &row);
                }
            }
        }
    } else {
        let ds_v = l.mem.read_f32s(ds, np)?;
        for b in 0..b_n {
            for h in 0..h_n {
                for tp in 0..gy {
                    let r = b * t + tp;
                    for e in 0..d {
                        let (q, k, v) =
                            attn_back_col_gpu(&s, &qkv_v, &p_v, &dout_v, &ds_v, b, h, tp, e);
                        let col = h * d + e;
                        for (part, x) in [q, k, v].into_iter().enumerate() {
                            let at = (r * 3 * c + part * c + col) as u64;
                            l.mem.write_f32s(dqkv + 4 * at, &[x]);
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// The CPU stand-in for C-160 point 7's `attn_back_tile.sm86.sass` (`docs/C-160_P7_ATENTIA_INAPOI_PE_PLACI.md`): mode 1
/// (columns) of the attention backward on 64-position tiles, blocks of 256; the cbuf0 of `attn_back` (qkv/p/dout/ds/dqkv
/// at 0x28/0x30/0x38/0x40/0x48, T 0x50, C 0x54, H 0x58, d 0x5c, mode 0x60). A mode other than 1 makes the program exit
/// at once: nothing is written. Grid [⌈T/64⌉, ≤ H, ≤ B]: every element (b, h, position tp < min(T, 64·x), e) of dQ |
/// dK | dV is `gpu_compute::layer::attn_back_col_gpu` (the program sums the very same chains). d > 64 (the shared
/// tiles), a grid past H or past ⌈T/64⌉, or d ≠ C / H are refused as not modelled.
pub fn attn_back_tile_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::layer::{ATTN_TILE_MAX_D, AttnShape, attn_back_col_gpu};
    if l.block != [256, 1, 1] {
        return Err(bad("SIM: attn_back_tile cere blocuri de 256x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 || l.mem.read_u32(l.cbuf0 + 0x8)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0..0x8) diferita de 0, nemodelata",
        ));
    }
    let r64 = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let r32 = |l: &Launch<'_>, o: u64| l.mem.read_u32(l.cbuf0 + o);
    if r32(l, 0x60)? != 1 {
        return Ok(());
    }
    let (qkv, p, dout, ds, dqkv) = (
        r64(l, 0x28)?,
        r64(l, 0x30)?,
        r64(l, 0x38)?,
        r64(l, 0x40)?,
        r64(l, 0x48)?,
    );
    let (t, c, heads, d) = (
        r32(l, 0x50)? as usize,
        r32(l, 0x54)? as usize,
        r32(l, 0x58)? as usize,
        r32(l, 0x5c)? as usize,
    );
    let [gx, gy, gz] = l.grid.map(|g| g as usize);
    if heads == 0 || d == 0 || d * heads != c || t == 0 {
        return Err(bad(
            "SIM: attn_back_tile: d != C/H sau forma goala, nemodelat",
        ));
    }
    if d > ATTN_TILE_MAX_D {
        return Err(bad(
            "SIM: attn_back_tile: d > 64 (placile partajate), nemodelat",
        ));
    }
    if gy > heads || gx > t.div_ceil(64) {
        return Err(bad("SIM: attn_back_tile: grila trece de H / T, nemodelata"));
    }
    if gz == 0 || gy == 0 || gx == 0 {
        return Ok(());
    }
    let s = AttnShape {
        batch: gz,
        t,
        c,
        heads,
    };
    let (nq, no, np) = s
        .sizes()
        .ok_or_else(|| bad("SIM: attn_back_tile: forma prea mare"))?;
    let qkv_v = l.mem.read_f32s(qkv, nq)?;
    let p_v = l.mem.read_f32s(p, np)?;
    let dout_v = l.mem.read_f32s(dout, no)?;
    let ds_v = l.mem.read_f32s(ds, np)?;
    let positions = t.min(64 * gx);
    for b in 0..gz {
        for h in 0..gy {
            for tp in 0..positions {
                let r = b * t + tp;
                for e in 0..d {
                    let (q, k, v) =
                        attn_back_col_gpu(&s, &qkv_v, &p_v, &dout_v, &ds_v, b, h, tp, e);
                    let col = h * d + e;
                    for (part, x) in [q, k, v].into_iter().enumerate() {
                        let at = (r * 3 * c + part * c + col) as u64;
                        l.mem.write_f32s(dqkv + 4 * at, &[x]);
                    }
                }
            }
        }
    }
    Ok(())
}

/// C-157 (session 244, `docs/C-157_ATTN_BACK_TC.md` §5): what `tools/nak/out2/attn_back_tile_tc.sm86.sass` needs —
/// two f16 tiles 64 × 64 and the f32 result tile 64 × 64 (32 768 bytes, the dump's `Shared size`) and one barrier.
pub const ATTN_BACK_TILE_TC_NEEDS: Needs = Needs {
    shared_bytes: 2 * 64 * 64 * 2 + 64 * 64 * 4,
    barriers: 1,
};

/// Where NVK 26.3.1 puts the push constants of the second-tool programs in cbuf 0 (`attn_back_tile_tc` dump:
/// `c[0x0][+0x700]` … `c[0x0][+0x738]`).
const ATTN_TC_PUSH_BASE: u64 = 0x700;

/// C-157 (session 244): the CPU stand-in for `tools/nak/out2/attn_back_tile_tc.sm86.sass` — mode 1 of the attention
/// backward (dQ = dS·K, dK = dSᵀ·Q, dV = Pᵀ·dO) on the tensor cores, blocks of 256. **cbuf0 is NOT `attn_back`'s:**
/// the second NAK tool (Mesa 26.3.1) reads the arguments at 0x700 — qkv/p/dout/ds/dqkv at 0x700/0x708/0x710/0x718/0x720,
/// T 0x728, C 0x72c, H 0x730, d 0x734, mode 0x738 (the dump). A mode other than 1 makes the program exit at once:
/// nothing is written. Grid [⌈T/64⌉, ≤ H, ≤ B]: every element (b, h, position < min(T, 64·x), e) of dQ | dK | dV is
/// `gpu_compute::attn_tc::model` — the inputs rounded to binary16, the tensor-core adder fitted on os-335 (a
/// HYPOTHESIS, not a fact). d > 64, a grid past H or past ⌈T/64⌉, d ≠ C / H, or an input the program would read that is
/// not a finite binary16 (on the board it would become ±inf / NaN) are refused (no write) instead of guessed.
pub fn attn_back_tile_tc_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::layer::{ATTN_TILE_MAX_D, AttnShape};
    if l.block != [256, 1, 1] {
        return Err(bad("SIM: attn_back_tile_tc cere blocuri de 256x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 || l.mem.read_u32(l.cbuf0 + 0x8)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0..0x8) diferita de 0, nemodelata",
        ));
    }
    let base = l.cbuf0 + ATTN_TC_PUSH_BASE;
    let r64 = |l: &Launch<'_>, o: u64| l.mem.read_u64(base + o);
    let r32 = |l: &Launch<'_>, o: u64| l.mem.read_u32(base + o);
    if r32(l, 0x38)? != 1 {
        return Ok(());
    }
    let (qkv, p, dout, ds, dqkv) = (
        r64(l, 0x00)?,
        r64(l, 0x08)?,
        r64(l, 0x10)?,
        r64(l, 0x18)?,
        r64(l, 0x20)?,
    );
    let (t, c, heads, d) = (
        r32(l, 0x28)? as usize,
        r32(l, 0x2c)? as usize,
        r32(l, 0x30)? as usize,
        r32(l, 0x34)? as usize,
    );
    let [gx, gy, gz] = l.grid.map(|g| g as usize);
    if heads == 0 || d == 0 || d * heads != c || t == 0 {
        return Err(bad(
            "SIM: attn_back_tile_tc: d != C/H sau forma goala, nemodelat",
        ));
    }
    if d > ATTN_TILE_MAX_D {
        return Err(bad(
            "SIM: attn_back_tile_tc: d > 64 (placile partajate), nemodelat",
        ));
    }
    if gy > heads || gx > t.div_ceil(64) {
        return Err(bad(
            "SIM: attn_back_tile_tc: grila trece de H / T, nemodelata",
        ));
    }
    if gz == 0 || gy == 0 || gx == 0 {
        return Ok(());
    }
    let s = AttnShape {
        batch: gz,
        t,
        c,
        heads,
    };
    let (nq, no, np) = s
        .sizes()
        .ok_or_else(|| bad("SIM: attn_back_tile_tc: forma prea mare"))?;
    let qkv_v = l.mem.read_f32s(qkv, nq)?;
    let p_v = l.mem.read_f32s(p, np)?;
    let dout_v = l.mem.read_f32s(dout, no)?;
    let ds_v = l.mem.read_f32s(ds, np)?;
    let out = gpu_compute::attn_tc::model(&s, &qkv_v, &p_v, &dout_v, &ds_v).ok_or_else(|| {
        bad("SIM: attn_back_tile_tc: o intrare nu e f16 finit (pe placa ar iesi inf / NaN), nemodelat")
    })?;
    let positions = t.min(64 * gx);
    for b in 0..gz {
        for tp in 0..positions {
            let r = b * t + tp;
            for part in 0..3 {
                for h in 0..gy {
                    let at = r * 3 * c + part * c + h * d;
                    l.mem.write_f32s(dqkv + 4 * at as u64, &out[at..at + d]);
                }
            }
        }
    }
    Ok(())
}

/// C-157 (session 244, `docs/C-157_ATTN_BACK_TC.md` §6): what `tools/nak/out2/attn_back_tile_tc16.sm86.sass` needs —
/// one array of 4 096 words (16 384 bytes) and one barrier.
pub const ATTN_BACK_TILE_TC16_NEEDS: Needs = Needs {
    shared_bytes: 64 * 64 * 4,
    barriers: 1,
};

/// C-157 (session 244): the CPU stand-in for `tools/nak/out2/attn_back_tile_tc16.sm86.sass` — the same arithmetic as
/// `attn_back_tile_tc` in 16 KiB of shared memory (only where the tiles live changes), so the stand-in is
/// [`attn_back_tile_tc_on_cpu`]: the same cbuf (arguments at 0x700), the same `gpu_compute::attn_tc::model` — a
/// HYPOTHESIS. That the board gives the same words for both programs is what the board test checks.
pub fn attn_back_tile_tc16_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    attn_back_tile_tc_on_cpu(l)
}

/// C-157 (session 248, `docs/C-157_ATTN_BACK_TC16_V4.md`): what `tools/nak/out2/attn_back_tile_tc16_v4.sm86.sass`
/// needs — exactly `attn_back_tile_tc16`'s 16 KiB and one barrier (the 128-bit loads read ahead land in registers).
pub const ATTN_BACK_TILE_TC16_V4_NEEDS: Needs = ATTN_BACK_TILE_TC16_NEEDS;

/// C-157 (session 248): the CPU stand-in for `tools/nak/out2/attn_back_tile_tc16_v4.sm86.sass` — `attn_back_tile_tc16`
/// with 128-bit global loads and the next tile read ahead. Valid only for aligned shapes: T and d multiples of 4 and
/// qkv / p / dout / dS at multiples of 16 bytes (read from cbuf0 at 0x700, as the board would see them); anything else
/// would be a misaligned 128-bit load on the board, so the model REFUSES it (no write) instead of guessing. For
/// aligned shapes the values put in shared memory are tc16's (T258), so the stand-in is [`attn_back_tile_tc_on_cpu`]
/// (a mode other than 1 still exits at once, before any load).
pub fn attn_back_tile_tc16_v4_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    let base = l.cbuf0 + ATTN_TC_PUSH_BASE;
    if l.mem.read_u32(base + 0x38)? == 1 {
        let mut aligned = true;
        for o in [0x00, 0x08, 0x10, 0x18] {
            aligned &= l.mem.read_u64(base + o)?.is_multiple_of(16);
        }
        let t = l.mem.read_u32(base + 0x28)?;
        let d = l.mem.read_u32(base + 0x34)?;
        if !aligned || !t.is_multiple_of(4) || !d.is_multiple_of(4) {
            return Err(bad(
                "SIM: attn_back_tile_tc16_v4: forma nealiniata (T, d multiplu de 4, adrese multiplu de 16)",
            ));
        }
    }
    attn_back_tile_tc_on_cpu(l)
}

/// C-157 (session 248): what `tools/nak/out2/attn_back_tile_tc16_v4pf.sm86.sass` needs — tc16's 16 KiB, one barrier.
pub const ATTN_BACK_TILE_TC16_V4PF_NEEDS: Needs = ATTN_BACK_TILE_TC16_NEEDS;

/// C-157 (session 248): the CPU stand-in for `tools/nak/out2/attn_back_tile_tc16_v4pf.sm86.sass` —
/// `attn_back_tile_tc16_v4` with the next tile read ahead of the products: the same loads (only issued earlier), the
/// same alignment rule and refusals, the same values, so the stand-in is [`attn_back_tile_tc16_v4_on_cpu`].
pub fn attn_back_tile_tc16_v4pf_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    attn_back_tile_tc16_v4_on_cpu(l)
}

/// C-157 (session 250, `docs/C-157_ATTN_BACK0_TC_LEGAT.md`): what `tools/nak/out2/attn_back_tile0_tc.sm86.sass` needs —
/// the dump's `Shared size: 33792` (the f16 dO / V tiles, then the f32 dP tile 64 × 68; os-467: rows of 65 = 33 024 B
/// stopped the board, `docs/C-157_OS467_ATTN0_TC_BLOCAJ.md`) and one barrier.
pub const ATTN_BACK_TILE0_TC_NEEDS: Needs = Needs {
    shared_bytes: 33_792,
    barriers: 1,
};

/// C-157 (session 250, `docs/C-157_ATTN_BACK0_TC_LEGAT.md`): the CPU stand-in for
/// `tools/nak/out2/attn_back_tile0_tc.sm86.sass` — mode 0 of the attention backward (dP = dO·Vᵀ on the tensor cores,
/// dot and dS in F32), blocks of 256. **cbuf0 is NOT `attn_back`'s:** the arguments sit at 0x700 (qkv / p / dout / ds at
/// 0x700 / 0x708 / 0x710 / 0x718, T 0x728, C 0x72c, H 0x730, d 0x734, mode 0x738, scale 0x73c; the dump). A mode
/// other than 0 makes the program exit at once: nothing is written. Grid [⌈T/64⌉, ≤ H, ≤ B]: every row
/// (b, h, i < min(T, 64·x)) of dS is `gpu_compute::attn_tc::back0_model` — dO and V rounded to binary16, the
/// tensor-core adder fitted on os-335 (a HYPOTHESIS, not a fact). d > 64, a grid past H or past ⌈T/64⌉, d ≠ C / H, or a
/// dO / V value that is not a finite binary16 are refused (no write) instead of guessed.
pub fn attn_back_tile0_tc_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::layer::{ATTN_TILE_MAX_D, AttnShape};
    if l.block != [256, 1, 1] {
        return Err(bad("SIM: attn_back_tile0_tc cere blocuri de 256x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 || l.mem.read_u32(l.cbuf0 + 0x8)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0..0x8) diferita de 0, nemodelata",
        ));
    }
    let base = l.cbuf0 + ATTN_TC_PUSH_BASE;
    let r64 = |l: &Launch<'_>, o: u64| l.mem.read_u64(base + o);
    let r32 = |l: &Launch<'_>, o: u64| l.mem.read_u32(base + o);
    if r32(l, 0x38)? != 0 {
        return Ok(());
    }
    let (qkv, p, dout, ds) = (r64(l, 0x00)?, r64(l, 0x08)?, r64(l, 0x10)?, r64(l, 0x18)?);
    let (t, c, heads, d) = (
        r32(l, 0x28)? as usize,
        r32(l, 0x2c)? as usize,
        r32(l, 0x30)? as usize,
        r32(l, 0x34)? as usize,
    );
    let scale = f32::from_bits(r32(l, 0x3c)?);
    let [gx, gy, gz] = l.grid.map(|g| g as usize);
    if heads == 0 || d == 0 || d * heads != c || t == 0 {
        return Err(bad(
            "SIM: attn_back_tile0_tc: d != C/H sau forma goala, nemodelat",
        ));
    }
    if d > ATTN_TILE_MAX_D {
        return Err(bad(
            "SIM: attn_back_tile0_tc: d > 64 (placile partajate), nemodelat",
        ));
    }
    if gy > heads || gx > t.div_ceil(64) {
        return Err(bad(
            "SIM: attn_back_tile0_tc: grila trece de H / T, nemodelata",
        ));
    }
    if gz == 0 || gy == 0 || gx == 0 {
        return Ok(());
    }
    let s = AttnShape {
        batch: gz,
        t,
        c,
        heads,
    };
    let (nq, no, np) = s
        .sizes()
        .ok_or_else(|| bad("SIM: attn_back_tile0_tc: forma prea mare"))?;
    let qkv_v = l.mem.read_f32s(qkv, nq)?;
    let p_v = l.mem.read_f32s(p, np)?;
    let dout_v = l.mem.read_f32s(dout, no)?;
    let out = gpu_compute::attn_tc::back0_model(&s, &qkv_v, &p_v, &dout_v, scale).ok_or_else(|| {
        bad("SIM: attn_back_tile0_tc: o intrare nu e f16 finit (pe placa ar iesi inf / NaN), nemodelat")
    })?;
    let rows = t.min(64 * gx);
    for b in 0..gz {
        for h in 0..gy {
            for i in 0..rows {
                let at = ((b * heads + h) * t + i) * t;
                l.mem.write_f32s(ds + 4 * at as u64, &out[at..at + t]);
            }
        }
    }
    Ok(())
}

/// C-157 (session 251, `docs/C-157_ATTN_BACK0_TC_V4.md`): what `tools/nak/out2/attn_back_tile0_tc_v4.sm86.sass` needs —
/// exactly `attn_back_tile0_tc`'s 33 792 bytes and one barrier (the 128-bit loads land in registers).
pub const ATTN_BACK_TILE0_TC_V4_NEEDS: Needs = ATTN_BACK_TILE0_TC_NEEDS;

/// C-157 (session 251, `docs/C-157_ATTN_BACK0_TC_V4.md`): the CPU stand-in for
/// `tools/nak/out2/attn_back_tile0_tc_v4.sm86.sass` — `attn_back_tile0_tc` with 128-bit global loads and stores. Valid
/// only for aligned shapes: T and d multiples of 4 and qkv / p / dout / dS at multiples of 16 bytes (read from cbuf0 at
/// 0x700, as the board would see them); anything else would be a misaligned 128-bit access on the board, so the model
/// REFUSES it (no write) instead of guessing. For aligned shapes the values in shared memory and the final dS are
/// `attn_back_tile0_tc`'s (T261), so the stand-in is [`attn_back_tile0_tc_on_cpu`] (a mode other than 0 still exits at
/// once, before any load).
pub fn attn_back_tile0_tc_v4_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    let base = l.cbuf0 + ATTN_TC_PUSH_BASE;
    if l.mem.read_u32(base + 0x38)? == 0 {
        let mut aligned = true;
        for o in [0x00, 0x08, 0x10, 0x18] {
            aligned &= l.mem.read_u64(base + o)?.is_multiple_of(16);
        }
        let t = l.mem.read_u32(base + 0x28)?;
        let d = l.mem.read_u32(base + 0x34)?;
        if !aligned || !t.is_multiple_of(4) || !d.is_multiple_of(4) {
            return Err(bad(
                "SIM: attn_back_tile0_tc_v4: forma nealiniata (T, d multiplu de 4, adrese multiplu de 16)",
            ));
        }
    }
    attn_back_tile0_tc_on_cpu(l)
}

/// The CPU stand-in for C-160 point 7's `attn_back_tile0.sm86.sass` (`docs/C-160_P7_ATENTIA_INAPOI_MOD0_PE_PLACI.md`):
/// mode 0 (the rows of dS) of the attention backward on 64-row tiles, blocks of 256; the cbuf0 of `attn_back`
/// (qkv/p/dout/ds at 0x28/0x30/0x38/0x40, T 0x50, C 0x54, H 0x58, d 0x5c, mode 0x60, scale 0x64). A mode other than 0
/// makes the program exit at once: nothing is written. Grid [⌈T/64⌉, ≤ H, ≤ B]: every row (b, h, i < min(T, 64·x)) of
/// dS is `gpu_compute::layer::attn_back_row_gpu` (the program sums the very same chains). d > 64 (the shared tiles), a
/// grid past H or past ⌈T/64⌉, or d ≠ C / H are refused as not modelled.
pub fn attn_back_tile0_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::layer::{ATTN_TILE_MAX_D, AttnShape, attn_back_row_gpu};
    if l.block != [256, 1, 1] {
        return Err(bad("SIM: attn_back_tile0 cere blocuri de 256x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 || l.mem.read_u32(l.cbuf0 + 0x8)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0..0x8) diferita de 0, nemodelata",
        ));
    }
    let r64 = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let r32 = |l: &Launch<'_>, o: u64| l.mem.read_u32(l.cbuf0 + o);
    if r32(l, 0x60)? != 0 {
        return Ok(());
    }
    let (qkv, p, dout, ds) = (r64(l, 0x28)?, r64(l, 0x30)?, r64(l, 0x38)?, r64(l, 0x40)?);
    let (t, c, heads, d) = (
        r32(l, 0x50)? as usize,
        r32(l, 0x54)? as usize,
        r32(l, 0x58)? as usize,
        r32(l, 0x5c)? as usize,
    );
    let scale = f32::from_bits(r32(l, 0x64)?);
    let [gx, gy, gz] = l.grid.map(|g| g as usize);
    if heads == 0 || d == 0 || d * heads != c || t == 0 {
        return Err(bad(
            "SIM: attn_back_tile0: d != C/H sau forma goala, nemodelat",
        ));
    }
    if d > ATTN_TILE_MAX_D {
        return Err(bad(
            "SIM: attn_back_tile0: d > 64 (placile partajate), nemodelat",
        ));
    }
    if gy > heads || gx > t.div_ceil(64) {
        return Err(bad(
            "SIM: attn_back_tile0: grila trece de H / T, nemodelata",
        ));
    }
    if gz == 0 || gy == 0 || gx == 0 {
        return Ok(());
    }
    let s = AttnShape {
        batch: gz,
        t,
        c,
        heads,
    };
    let (nq, no, np) = s
        .sizes()
        .ok_or_else(|| bad("SIM: attn_back_tile0: forma prea mare"))?;
    let qkv_v = l.mem.read_f32s(qkv, nq)?;
    let p_v = l.mem.read_f32s(p, np)?;
    let dout_v = l.mem.read_f32s(dout, no)?;
    let rows = t.min(64 * gx);
    for b in 0..gz {
        for h in 0..gy {
            for i in 0..rows {
                let row = attn_back_row_gpu(&s, &qkv_v, &p_v, &dout_v, scale, b, h, i);
                let at = (((b * heads + h) * t + i) * t) as u64;
                l.mem.write_f32s(ds + 4 * at, &row);
            }
        }
    }
    Ok(())
}

/// The CPU stand-in for C-160 point 4's attention forward (`attn_fwd.sm86.sass`, program 6, forward half): blocks of
/// 32; cbuf0 qkv/p/att at 0x28/0x30/0x38, T at 0x40, C at 0x44, H at 0x48, d at 0x4c, the scale at 0x50 (NAK dump).
/// The batch is the grid's z (the program never reads it). Grid [⌈T/32⌉, ≤ H, ≤ B]: every row (b, h, i < min(T, 32·x))
/// is `gpu_compute::layer::attn_fwd_row_gpu` (row i of P, all T values, and the d values of att). A grid past H / B or
/// past ⌈T/32⌉ (the program does not guard those) or a d ≠ C / H is refused as not modelled.
pub fn attn_fwd_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    attn_fwd_blocks(l, 32)
}

/// The CPU stand-in for C-160 point 7's `attn_p.sm86.sass` (`docs/C-160_P7_ATTN_P.md`): `attn_fwd` without P·V —
/// the same cbuf0 (att at 0x38 is read by nobody, nothing is written there), blocks of 32, grid [⌈T/32⌉, ≤ H, ≤ B];
/// every row (b, h, i < min(T, 32·x)) of P is `gpu_compute::layer::attn_p_row_gpu` (the P part of
/// `attn_fwd_row_gpu`, operation for operation).
pub fn attn_p_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    attn_fwd_blocks_with(l, 32, false)
}

/// The CPU stand-in for C-160 point 7's `attn_fwd128.sm86.sass` (`docs/C-160_P7_FIR_LIMITAT.md`): the program of
/// [`attn_fwd_on_cpu`] with blocks of 128 — the same code per thread, so every row is the same `attn_fwd_row_gpu`.
/// Grid [⌈T/128⌉, ≤ H, ≤ B]: rows i < min(T, 128·x).
pub fn attn_fwd128_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    attn_fwd_blocks(l, 128)
}

/// The CPU stand-in for C-160 point 7's `attn_fwd_tile.sm86.sass` (`docs/C-160_P7_ATENTIA_PE_PLACI.md`): blocks of 256
/// threads, each owning 64 rows; the cbuf0 of `attn_fwd` (qkv/p/att at 0x28/0x30/0x38, T 0x40, C 0x44, H 0x48, d 0x4c,
/// scale 0x50). Grid [⌈T/64⌉, ≤ H, ≤ B]: every row (b, h, i < min(T, 64·x)) is
/// `gpu_compute::layer::attn_fwd_tile_row_gpu`. d > 64 (the shared tiles), a grid past H or past ⌈T/64⌉, or d ≠ C / H
/// are refused as not modelled.
pub fn attn_fwd_tile_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::layer::{ATTN_TILE_MAX_D, AttnShape, attn_fwd_tile_row_gpu};
    if l.block != [256, 1, 1] {
        return Err(bad("SIM: attn_fwd_tile cere blocuri de 256x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 || l.mem.read_u32(l.cbuf0 + 0x8)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0..0x8) diferita de 0, nemodelata",
        ));
    }
    let r64 = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let r32 = |l: &Launch<'_>, o: u64| l.mem.read_u32(l.cbuf0 + o);
    let (qkv, p, att) = (r64(l, 0x28)?, r64(l, 0x30)?, r64(l, 0x38)?);
    let (t, c, heads, d) = (
        r32(l, 0x40)? as usize,
        r32(l, 0x44)? as usize,
        r32(l, 0x48)? as usize,
        r32(l, 0x4c)? as usize,
    );
    let scale = f32::from_bits(r32(l, 0x50)?);
    let [gx, gy, gz] = l.grid.map(|g| g as usize);
    if heads == 0 || d == 0 || d * heads != c || t == 0 {
        return Err(bad(
            "SIM: attn_fwd_tile: d != C/H sau forma goala, nemodelat",
        ));
    }
    if d > ATTN_TILE_MAX_D {
        return Err(bad(
            "SIM: attn_fwd_tile: d > 64 (placile partajate), nemodelat",
        ));
    }
    if gy > heads || gx > t.div_ceil(64) {
        return Err(bad("SIM: attn_fwd_tile: grila trece de H / T, nemodelata"));
    }
    if gz == 0 || gy == 0 || gx == 0 {
        return Ok(());
    }
    let s = AttnShape {
        batch: gz,
        t,
        c,
        heads,
    };
    let (nq, _, _) = s
        .sizes()
        .ok_or_else(|| bad("SIM: attn_fwd_tile: forma prea mare"))?;
    let qkv_v = l.mem.read_f32s(qkv, nq)?;
    let rows = t.min(64 * gx);
    for b in 0..gz {
        for h in 0..gy {
            for i in 0..rows {
                let (row, a) = attn_fwd_tile_row_gpu(&s, &qkv_v, scale, b, h, i);
                let at = (((b * heads + h) * t + i) * t) as u64;
                l.mem.write_f32s(p + 4 * at, &row);
                let o = ((b * t + i) * c + h * d) as u64;
                l.mem.write_f32s(att + 4 * o, &a);
            }
        }
    }
    Ok(())
}

/// The CPU stand-in for C-160 point 7's `attn_p_tile.sm86.sass` (`docs/C-160_P7_ATTN_P_TILE.md`): `attn_fwd_tile2`
/// without P·V — the cbuf0, blocks and grid of [`attn_fwd_tile_on_cpu`], but only P is written: every row
/// (b, h, i < min(T, 64·x)) is `gpu_compute::layer::attn_p_tile_row_gpu`; `att` (0x38) is never touched. d > 64, a grid
/// past H or past ⌈T/64⌉, or d ≠ C / H are refused as not modelled.
pub fn attn_p_tile_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::layer::{ATTN_TILE_MAX_D, AttnShape, attn_p_tile_row_gpu};
    if l.block != [256, 1, 1] {
        return Err(bad("SIM: attn_p_tile cere blocuri de 256x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 || l.mem.read_u32(l.cbuf0 + 0x8)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0..0x8) diferita de 0, nemodelata",
        ));
    }
    let r64 = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let r32 = |l: &Launch<'_>, o: u64| l.mem.read_u32(l.cbuf0 + o);
    let (qkv, p) = (r64(l, 0x28)?, r64(l, 0x30)?);
    let (t, c, heads, d) = (
        r32(l, 0x40)? as usize,
        r32(l, 0x44)? as usize,
        r32(l, 0x48)? as usize,
        r32(l, 0x4c)? as usize,
    );
    let scale = f32::from_bits(r32(l, 0x50)?);
    let [gx, gy, gz] = l.grid.map(|g| g as usize);
    if heads == 0 || d == 0 || d * heads != c || t == 0 {
        return Err(bad("SIM: attn_p_tile: d != C/H sau forma goala, nemodelat"));
    }
    if d > ATTN_TILE_MAX_D {
        return Err(bad(
            "SIM: attn_p_tile: d > 64 (placile partajate), nemodelat",
        ));
    }
    if gy > heads || gx > t.div_ceil(64) {
        return Err(bad("SIM: attn_p_tile: grila trece de H / T, nemodelata"));
    }
    if gz == 0 || gy == 0 || gx == 0 {
        return Ok(());
    }
    let s = AttnShape {
        batch: gz,
        t,
        c,
        heads,
    };
    let (nq, _, _) = s
        .sizes()
        .ok_or_else(|| bad("SIM: attn_p_tile: forma prea mare"))?;
    let qkv_v = l.mem.read_f32s(qkv, nq)?;
    let rows = t.min(64 * gx);
    for b in 0..gz {
        for h in 0..gy {
            for i in 0..rows {
                let row = attn_p_tile_row_gpu(&s, &qkv_v, scale, b, h, i);
                let at = (((b * heads + h) * t + i) * t) as u64;
                l.mem.write_f32s(p + 4 * at, &row);
            }
        }
    }
    Ok(())
}

fn attn_fwd_blocks(l: &mut Launch<'_>, block: u32) -> GpuResult<()> {
    attn_fwd_blocks_with(l, block, true)
}

fn attn_fwd_blocks_with(l: &mut Launch<'_>, block: u32, with_att: bool) -> GpuResult<()> {
    use gpu_compute::layer::{AttnShape, attn_fwd_row_gpu, attn_p_row_gpu};
    if l.block != [block, 1, 1] {
        return Err(bad(if !with_att {
            "SIM: attn_p cere blocuri de 32x1x1"
        } else if block == 32 {
            "SIM: attn_fwd cere blocuri de 32x1x1"
        } else {
            "SIM: attn_fwd128 cere blocuri de 128x1x1"
        }));
    }
    let rows_per_block = block as usize;
    if l.mem.read_u64(l.cbuf0)? != 0 || l.mem.read_u32(l.cbuf0 + 0x8)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0..0x8) diferita de 0, nemodelata",
        ));
    }
    let r64 = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let r32 = |l: &Launch<'_>, o: u64| l.mem.read_u32(l.cbuf0 + o);
    let (qkv, p, att) = (r64(l, 0x28)?, r64(l, 0x30)?, r64(l, 0x38)?);
    let (t, c, heads, d) = (
        r32(l, 0x40)? as usize,
        r32(l, 0x44)? as usize,
        r32(l, 0x48)? as usize,
        r32(l, 0x4c)? as usize,
    );
    let scale = f32::from_bits(r32(l, 0x50)?);
    let [gx, gy, gz] = l.grid.map(|g| g as usize);
    if heads == 0 || d == 0 || d * heads != c || t == 0 {
        return Err(bad("SIM: attn_fwd: d != C/H sau forma goala, nemodelat"));
    }
    if gy > heads || gx > t.div_ceil(rows_per_block) {
        return Err(bad("SIM: attn_fwd: grila trece de H / T, nemodelata"));
    }
    if gz == 0 || gy == 0 || gx == 0 {
        return Ok(());
    }
    let s = AttnShape {
        batch: gz,
        t,
        c,
        heads,
    };
    let (nq, _, _) = s
        .sizes()
        .ok_or_else(|| bad("SIM: attn_fwd: forma prea mare"))?;
    let qkv_v = l.mem.read_f32s(qkv, nq)?;
    let rows = t.min(rows_per_block * gx);
    for b in 0..gz {
        for h in 0..gy {
            for i in 0..rows {
                let at = (((b * heads + h) * t + i) * t) as u64;
                if !with_att {
                    l.mem
                        .write_f32s(p + 4 * at, &attn_p_row_gpu(&s, &qkv_v, scale, b, h, i));
                    continue;
                }
                let (row, a) = attn_fwd_row_gpu(&s, &qkv_v, scale, b, h, i);
                l.mem.write_f32s(p + 4 * at, &row);
                let o = ((b * t + i) * c + h * d) as u64;
                l.mem.write_f32s(att + 4 * o, &a);
            }
        }
    }
    Ok(())
}

/// The CPU stand-in for C-160 point 4's embedding forward (`emb_fwd.sm86.sass`, program 8): blocks of 64 along k, grid
/// [≤ ⌈C/64⌉, T, B] (position i = the block's y, text b = its z); cbuf0 tok/emb/pos/x at 0x28/0x30/0x38/0x40, C at 0x48,
/// T at 0x4c (NAK dump). Every block (i, b): r = b·T + i, id = tok[r], `x[r, k] = fadd.ftz(emb[id, k], pos[i, k])` for
/// k < min(C, 64·x) — the program reads the table at id·C as it is (the driver checks the ids, the program does not)
/// and does not compare i with T (a grid y past T is the driver's mistake, modelled as the program does it).
pub fn emb_fwd_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::math::add_f32_ftz;
    if l.block != [64, 1, 1] {
        return Err(bad("SIM: emb_fwd cere blocuri de 64x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 || l.mem.read_u32(l.cbuf0 + 0x8)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0..0x8) diferita de 0, nemodelata",
        ));
    }
    let r64 = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (tok, emb, pos, x) = (r64(l, 0x28)?, r64(l, 0x30)?, r64(l, 0x38)?, r64(l, 0x40)?);
    let c = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let t = l.mem.read_u32(l.cbuf0 + 0x4c)? as usize;
    let cols = c.min(64 * l.grid[0] as usize);
    if cols == 0 {
        return Ok(());
    }
    for b in 0..l.grid[2] as usize {
        for i in 0..l.grid[1] as usize {
            let r = b * t + i;
            let id = l.mem.read_u32(tok + 4 * r as u64)? as u64;
            let e = l.mem.read_f32s(emb + 4 * id * c as u64, cols)?;
            let p = l.mem.read_f32s(pos + 4 * (i * c) as u64, cols)?;
            let out: Vec<f32> = e.iter().zip(&p).map(|(&a, &b)| add_f32_ftz(a, b)).collect();
            l.mem.write_f32s(x + 4 * (r * c) as u64, &out);
        }
    }
    Ok(())
}

/// The CPU stand-in for C-160 point 4's embedding backward (`emb_back.sm86.sass`, program 8): blocks of 64 along k,
/// grid [≤ ⌈C/64⌉, S, 1] (segment s = the block's y); cbuf0 dx/off/dst/rows/g at 0x28/0x30/0x38/0x40/0x48, C at 0x50,
/// the mode at 0x54 (0 = write, anything else = add, as the program's `isetp`) (NAK dump). Every segment s < grid y,
/// k < min(C, 64·x): the chain `a = 0; a = fadd.ftz(a, dx[rows[j], k])` for j in off[s]..off[s+1], then
/// `g[dst[s], k] = a` or `fadd.ftz(g, a)` — the lists are read as they are (the driver checks them). Segments run in
/// order s = 0, 1, …: with repeated destinations (which the driver refuses) the board's result would depend on
/// timing, the model's does not. A grid z ≠ 1 is refused as not modelled.
pub fn emb_back_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::math::add_f32_ftz;
    if l.block != [64, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: emb_back cere blocuri de 64x1x1, grila 2D"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let r64 = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (dx, off, dst, rows, g) = (
        r64(l, 0x28)?,
        r64(l, 0x30)?,
        r64(l, 0x38)?,
        r64(l, 0x40)?,
        r64(l, 0x48)?,
    );
    let c = l.mem.read_u32(l.cbuf0 + 0x50)? as usize;
    let add = l.mem.read_u32(l.cbuf0 + 0x54)? != 0;
    let cols = c.min(64 * l.grid[0] as usize);
    if cols == 0 {
        return Ok(());
    }
    for s in 0..l.grid[1] as u64 {
        let lo = l.mem.read_u32(off + 4 * s)? as u64;
        let hi = l.mem.read_u32(off + 4 * (s + 1))? as u64;
        let mut a = alloc::vec![0.0f32; cols];
        for j in lo..hi {
            let r = l.mem.read_u32(rows + 4 * j)? as u64;
            let v = l.mem.read_f32s(dx + 4 * r * c as u64, cols)?;
            for (ak, vk) in a.iter_mut().zip(v) {
                *ak = add_f32_ftz(*ak, vk);
            }
        }
        let o = g + 4 * l.mem.read_u32(dst + 4 * s)? as u64 * c as u64;
        if add {
            let old = l.mem.read_f32s(o, cols)?;
            for (ak, ok) in a.iter_mut().zip(old) {
                *ak = add_f32_ftz(ok, *ak);
            }
        }
        l.mem.write_f32s(o, &a);
    }
    Ok(())
}

/// C-160 point 4, program 9: what `tools/nak/out/gnorm.sm86.sass` needs — 32 bytes of shared memory (the 8 warp sums)
/// and one barrier.
pub const GNORM_NEEDS: Needs = Needs {
    shared_bytes: 32,
    barriers: 1,
};

/// The CPU stand-in for C-160 point 4's gradient-norm program (`gnorm.sm86.sass`, program 9): blocks of 256, grid
/// [blocks, 1, 1]; cbuf0 x/part/res at 0x28/0x30/0x38, n 0x40, chunk 0x44, the mode 0x48 (bit 0 squares, bit 1
/// finish), max_norm 0x4c (NAK dump). Block b covers `x[b·chunk .. min((b+1)·chunk, n))` (`gpu_compute::layer::
/// gnorm_block_gpu`); with bit 1, thread 0 of EVERY block writes `res[0..2]` (`gnorm_finish_gpu`; the driver
/// launches one block, a grid > 1 is refused as not modelled: on the board the blocks would race on res), else
/// `part[b]`. A chunk · grid past 32 bits is refused (the program's `b·chunk` wraps).
pub fn gnorm_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::layer::{gnorm_block_gpu, gnorm_finish_gpu};
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad("SIM: gnorm cere blocuri de 256x1x1, grila 1D"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (x, part, res) = (r(l, 0x28)?, r(l, 0x30)?, r(l, 0x38)?);
    let n = l.mem.read_u32(l.cbuf0 + 0x40)? as usize;
    let chunk = l.mem.read_u32(l.cbuf0 + 0x44)? as u64;
    let mode = l.mem.read_u32(l.cbuf0 + 0x48)?;
    let max_norm = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x4c)?);
    let finish = mode & 2 != 0;
    if finish && l.grid[0] != 1 {
        return Err(bad("SIM: gnorm cu terminare cere un singur bloc"));
    }
    if chunk * l.grid[0] as u64 > u32::MAX as u64 {
        return Err(bad("SIM: gnorm: b*chunk trece de 32 de biti, nemodelat"));
    }
    for b in 0..l.grid[0] as usize {
        let lo = (b as u64 * chunk) as usize;
        let hi = (lo + chunk as usize).min(n);
        let xs = if hi > lo {
            l.mem.read_f32s(x + 4 * lo as u64, hi - lo)?
        } else {
            alloc::vec::Vec::new()
        };
        let s = gnorm_block_gpu(&xs, 0, xs.len(), mode & 1 != 0);
        if finish {
            let (nrm, c) = gnorm_finish_gpu(s, max_norm);
            l.mem.write_u32(res, nrm.to_bits());
            l.mem.write_u32(res + 4, c.to_bits());
        } else {
            l.mem.write_u32(part + 4 * b as u64, s.to_bits());
        }
    }
    Ok(())
}

/// The CPU stand-in for C-160 point 4's optimizer program (`opt_step.sm86.sass`, program 10): blocks of 256, one
/// element per thread (grid x = ⌈n/256⌉); cbuf0 g/coef/w/m/v at 0x28/0x30/0x38/0x40/0x48, n 0x50, the mode 0x54
/// (0 = SGD, anything else = AdamW, as the program's `isetp`), the 8 scalars from 0x58 (NAK dump). Each element is
/// `gpu_compute::layer::opt_step_gpu` (the √ and the divisions modelled correctly rounded); threads with i ≥ n do
/// nothing; m and v are neither read nor written in SGD.
pub fn opt_step_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::layer::{OptScalars, opt_step_gpu};
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad("SIM: opt_step cere blocuri de 256x1x1, grila 1D"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (g, coef, w, m, v) = (
        r(l, 0x28)?,
        r(l, 0x30)?,
        r(l, 0x38)?,
        r(l, 0x40)?,
        r(l, 0x48)?,
    );
    let n = (l.mem.read_u32(l.cbuf0 + 0x50)? as u64).min(l.grid[0] as u64 * 256) as usize;
    let adam = l.mem.read_u32(l.cbuf0 + 0x54)? != 0;
    let f = |l: &Launch<'_>, i: u64| -> GpuResult<f32> {
        Ok(f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x58 + 4 * i)?))
    };
    let s = OptScalars {
        neg_lr: f(l, 0)?,
        decay: f(l, 1)?,
        w1: f(l, 2)?,
        b2: f(l, 3)?,
        omb2: f(l, 4)?,
        bc2_sqrt: f(l, 5)?,
        neg_step: f(l, 6)?,
        eps: f(l, 7)?,
    };
    if n == 0 {
        return Ok(());
    }
    let c = f32::from_bits(l.mem.read_u32(coef)?);
    let gs = l.mem.read_f32s(g, n)?;
    let mut ws = l.mem.read_f32s(w, n)?;
    if adam {
        let mut ms = l.mem.read_f32s(m, n)?;
        let mut vs = l.mem.read_f32s(v, n)?;
        for i in 0..n {
            (ws[i], ms[i], vs[i]) = opt_step_gpu(true, &s, c, gs[i], ws[i], ms[i], vs[i]);
        }
        l.mem.write_f32s(m, &ms);
        l.mem.write_f32s(v, &vs);
    } else {
        for i in 0..n {
            ws[i] = opt_step_gpu(false, &s, c, gs[i], ws[i], 0.0, 0.0).0;
        }
    }
    l.mem.write_f32s(w, &ws);
    Ok(())
}

/// The CPU stand-in for C-160 point 4's column-sum program (`colsum.sm86.sass`): blocks of 256 (one thread per
/// column, grid x = ⌈c/256⌉); cbuf0 dy/x/μ/rstd/s0/s1 at 0x28/0x30/0x38/0x40/0x48/0x50, rows at 0x58, c at 0x5c, the
/// mode at 0x60 (0 = bias, else LayerNorm: dγ in s0, dβ in s1) (NAK dump). The columns are
/// `gpu_compute::layer::colsum_gpu`; threads past `c` (and blocks past the grid) do nothing, as the program.
pub fn colsum_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad("SIM: colsum cere blocuri de 256x1x1, grila 1D"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (dy, x, mu, rs, s0, s1) = (
        r(l, 0x28)?,
        r(l, 0x30)?,
        r(l, 0x38)?,
        r(l, 0x40)?,
        r(l, 0x48)?,
        r(l, 0x50)?,
    );
    let rows = l.mem.read_u32(l.cbuf0 + 0x58)? as usize;
    let c = l.mem.read_u32(l.cbuf0 + 0x5c)? as usize;
    let hat = l.mem.read_u32(l.cbuf0 + 0x60)? != 0;
    // Only the columns the launched threads reach.
    let cols = c.min(256 * l.grid[0] as usize);
    if cols == 0 {
        return Ok(());
    }
    let dyv = l.mem.read_f32s(dy, rows * c)?;
    let (a, b) = if hat {
        let xv = l.mem.read_f32s(x, rows * c)?;
        let mv = l.mem.read_f32s(mu, rows)?;
        let rv = l.mem.read_f32s(rs, rows)?;
        let st = gpu_compute::layer::LnStats {
            x: &xv,
            mean: &mv,
            rstd: &rv,
        };
        gpu_compute::layer::colsum_gpu(&dyv, rows, c, Some(st))
    } else {
        gpu_compute::layer::colsum_gpu(&dyv, rows, c, None)
    };
    l.mem.write_f32s(s0, &a[..cols]);
    if hat {
        l.mem.write_f32s(s1, &b[..cols]);
    }
    Ok(())
}

/// The CPU stand-in for the first pass of the split column sum (`colsum_part.sm86.sass`, C-160 point 7): blocks of 256
/// (one thread per column), grid ⌈c/256⌉ × G; cbuf0 as `colsum` (dy/x/μ/rstd/p0/p1 at 0x28…0x50, rows 0x58, c 0x5c,
/// mode 0x60) plus the rows per chunk at 0x64. Block row g writes `p0[g·c + k]` (and `p1`) for the columns its
/// threads reach (`gpu_compute::layer::colsum_part_gpu`); block rows past ⌈rows/chunk⌉ add nothing and write 0.0, as
/// the program (its loop is empty, the store still happens).
pub fn colsum_part_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: colsum_part cere blocuri de 256x1x1, grila 2D"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (dy, x, mu, rs, p0, p1) = (
        r(l, 0x28)?,
        r(l, 0x30)?,
        r(l, 0x38)?,
        r(l, 0x40)?,
        r(l, 0x48)?,
        r(l, 0x50)?,
    );
    let rows = l.mem.read_u32(l.cbuf0 + 0x58)? as usize;
    let c = l.mem.read_u32(l.cbuf0 + 0x5c)? as usize;
    let hat = l.mem.read_u32(l.cbuf0 + 0x60)? != 0;
    let chunk = l.mem.read_u32(l.cbuf0 + 0x64)? as usize;
    let cols = c.min(256 * l.grid[0] as usize);
    let groups = l.grid[1] as usize;
    if cols == 0 || groups == 0 {
        return Ok(());
    }
    if chunk == 0 {
        return Err(bad(
            "SIM: colsum_part cu bucata 0 (programul ar scrie 0 peste tot), nemodelat",
        ));
    }
    let real = rows.div_ceil(chunk).min(groups);
    let dyv = l.mem.read_f32s(dy, rows * c)?;
    let (a, b) = if hat {
        let xv = l.mem.read_f32s(x, rows * c)?;
        let mv = l.mem.read_f32s(mu, rows)?;
        let rv = l.mem.read_f32s(rs, rows)?;
        let st = gpu_compute::layer::LnStats {
            x: &xv,
            mean: &mv,
            rstd: &rv,
        };
        gpu_compute::layer::colsum_part_gpu(&dyv, rows, c, Some(st), chunk)
    } else {
        gpu_compute::layer::colsum_part_gpu(&dyv, rows, c, None, chunk)
    };
    for g in 0..groups {
        let off = (g * c * 4) as u64;
        if g < real {
            l.mem.write_f32s(p0 + off, &a[g * c..g * c + cols]);
            if hat {
                l.mem.write_f32s(p1 + off, &b[g * c..g * c + cols]);
            }
        } else {
            let zeros = alloc::vec![0.0f32; cols];
            l.mem.write_f32s(p0 + off, &zeros);
            if hat {
                l.mem.write_f32s(p1 + off, &zeros);
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

/// What `tools/nak/out/matmul_vec4.sm86.sass` (step 6) needs: the same 4 KiB
/// of shared memory as step 5 (A transposed + B) and one barrier.
pub const MATMUL_VEC4_NEEDS: Needs = Needs {
    shared_bytes: (64 * 8 + 8 * 64) * 4,
    barriers: 1,
};

/// The CPU stand-in for step 6 (`matmul_vec4.sm86.sass`, siboehm
/// "vectorize"): step 5's blocking (64 threads, a 64×64 block of C, 8×8
/// cells per thread, K-tiles of 8) with 128-bit accesses. It follows the
/// kernel: per K-tile, vector `q = tid + 64s` (s = 0, 1) of A is row `q / 2`,
/// columns `(q % 2) * 4 ..+4`, stored **transposed** (`as_[p * 64 + row]`);
/// vector `q` of B is row `q / 16`, columns `(q % 16) * 4 ..+4`; vectors
/// outside A/B are zeros; then for each p ≤ min(8, K − t) the thread reads
/// 8 values of A and 8 of B (two vec4 each) and accumulates the 64 products;
/// C is written as vec4. The kernel itself does not check its requirements,
/// so the model refuses (no write) what it cannot run correctly: K or N not
/// a multiple of 4, or a/b/c not aligned to 16 bytes (on the card a
/// misaligned 128-bit access faults). Oracle: `reference::matmul_fma_ftz`.
pub fn matmul_vec4_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    const BM: usize = 64;
    const BN: usize = 64;
    const BK: usize = 8;
    const TM: usize = 8;
    const TN: usize = 8;
    const THREADS: usize = (BM / TM) * (BN / TN);
    if l.block != [THREADS as u32, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: matmul_vec4 cere blocuri de 64x1x1"));
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
    if !k.is_multiple_of(4) || !n.is_multiple_of(4) {
        return Err(bad("SIM: matmul_vec4 cere K si N multipli de 4"));
    }
    if !(a | b | c).is_multiple_of(16) {
        return Err(bad("SIM: matmul_vec4 cere a/b/c aliniate la 16 octeti"));
    }
    let x = l.mem.read_f32s(a, m * k)?;
    let y = l.mem.read_f32s(b, k * n)?;
    let fma_f32_ftz = gpu_compute::math::fma_f32_ftz;
    for gy in 0..l.grid[1] as usize {
        for gx in 0..l.grid[0] as usize {
            let (row0, col0) = (gy * BM, gx * BN);
            let mut acc = alloc::vec![[0.0f32; TM * TN]; THREADS];
            let mut t = 0;
            while t < k {
                let mut as_ = [0.0f32; BK * BM]; // transposed
                let mut bs_ = [0.0f32; BK * BN];
                for tid in 0..THREADS {
                    for s in 0..2 {
                        let q = tid + s * THREADS;
                        let (r, kq) = (q / (BK / 4), q % (BK / 4));
                        let (ar, ak) = (row0 + r, t + kq * 4);
                        if ar < m && ak < k {
                            for e in 0..4 {
                                as_[(kq * 4 + e) * BM + r] = x[ar * k + ak + e];
                            }
                        }
                        let (r, cq) = (q / (BN / 4), q % (BN / 4));
                        let (bk, bc) = (t + r, col0 + cq * 4);
                        if bk < k && bc < n {
                            for e in 0..4 {
                                bs_[r * BN + cq * 4 + e] = y[bk * n + bc + e];
                            }
                        }
                    }
                }
                // barrier()
                let len = BK.min(k - t);
                for (tid, acc_t) in acc.iter_mut().enumerate() {
                    let (trow, tcol) = (tid / (BN / TN), tid % (BN / TN));
                    for p in 0..len {
                        let reg_m: [f32; TM] =
                            core::array::from_fn(|i| as_[p * BM + trow * TM + i]);
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
                    let row = row0 + trow * TM + i;
                    for j in (0..TN).step_by(4) {
                        let col = col0 + tcol * TN + j;
                        if row < m && col < n {
                            for e in 0..4 {
                                l.mem.write_u32(
                                    c + 4 * (row * n + col + e) as u64,
                                    acc_t[i * TN + j + e].to_bits(),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// C-83 I11: the CPU stand-in for `tools/nak/out/copy_b128.sm86.sass`.
/// Arguments in cbuf0 as NVK places the push constants (source 0x28,
/// destination 0x30, count of 16-byte units 0x38); blocks of 256×1×1; thread
/// `i` (block × 256 + lane) copies unit `i` if `i < n` — so, like the program,
/// a grid too small copies only what its threads cover. Bytes are copied as
/// they are (the program moves `uvec4` integers).
pub fn copy_b128_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad("SIM: copy_b128 cere blocuri de 256x1x1, grila 1D"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad("SIM: baza grupurilor (cbuf0 0x0) trebuie sa fie 0"));
    }
    let src = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let dst = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let n = l.mem.read_u32(l.cbuf0 + 0x38)? as u64;
    if !src.is_multiple_of(16) || !dst.is_multiple_of(16) {
        return Err(bad("SIM: copy_b128 cu adrese nealiniate la 16"));
    }
    let units = n.min(l.grid[0] as u64 * 256);
    let mut buf = [0u8; 16];
    for i in 0..units {
        l.mem.read(src + 16 * i, &mut buf)?;
        l.mem.write(dst + 16 * i, &buf);
    }
    Ok(())
}

/// C-83 I12: what `q8_matvec` needs from its QMD: 128 bytes of shared memory
/// (32 partial sums). No hardware barrier: with a workgroup of 32 threads (one
/// warp) NAK compiled `barrier()` to `MEMBAR` only — the dump has no
/// `BAR.SYNC` (`q8_matvec.sm86.txt`).
pub const Q8_MATVEC_NEEDS: Needs = Needs {
    shared_bytes: 128,
    barriers: 0,
};

/// C-83 I12: the CPU stand-in for `tools/nak/out/q8_matvec.sm86.sass`:
/// cbuf0 0x28 = W (Q8_0 rows), 0x30 = x, 0x38 = y, 0x40 = rows,
/// 0x44 = blocks per row; one block of 32×1×1 per row (grid x = rows);
/// each row through `gpu_compute::q8::row_gpu_order` (the program's exact
/// order and operations). Blocks past `rows` write nothing, like the program.
pub fn q8_matvec_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [32, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad("SIM: q8_matvec cere blocuri de 32x1x1, grila 1D"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad("SIM: baza grupurilor (cbuf0 0x0) trebuie sa fie 0"));
    }
    let w = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let x = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let y = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let rows = l.mem.read_u32(l.cbuf0 + 0x40)? as u64;
    let blocks = l.mem.read_u32(l.cbuf0 + 0x44)? as usize;
    if !(w | x | y).is_multiple_of(4) {
        return Err(bad("SIM: q8_matvec cu adrese nealiniate la 4"));
    }
    let row_bytes = blocks * gpu_compute::q8::BLOCK_BYTES;
    let xs = l.mem.read_f32s(x, blocks * gpu_compute::q8::BLOCK)?;
    let mut row = alloc::vec![0u8; row_bytes];
    for r in 0..rows.min(l.grid[0] as u64) {
        l.mem.read(w + r * row_bytes as u64, &mut row)?;
        let v = gpu_compute::q8::row_gpu_order(&row, &xs);
        l.mem.write_u32(y + 4 * r, v.to_bits());
    }
    Ok(())
}

/// C-147 step 3: what `q8_matvec32` needs: the same 32 partial sums as `q8_matvec` (128 B), no hardware
/// barrier (one warp; `q8_matvec32.sm86.txt` has no `BAR.SYNC`).
pub const Q8_MATVEC32_NEEDS: Needs = Needs {
    shared_bytes: 128,
    barriers: 0,
};

/// C-147 step 3: the CPU stand-in for `tools/nak/out/q8_matvec32.sm86.sass` — the cbuf0 of `q8_matvec`
/// (W 0x28, x 0x30, y 0x38, rows 0x40, blocks 0x44), x 16-aligned (vec4 loads); every row through
/// `gpu_compute::q8::row_gpu_order_w32` (quarter blocks, all 32 threads; the program's exact order).
pub fn q8_matvec32_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [32, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 {
        return Err(bad("SIM: q8_matvec32 cere blocuri de 32x1x1, grila 1D"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad("SIM: baza grupurilor (cbuf0 0x0) trebuie sa fie 0"));
    }
    let w = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let x = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let y = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let rows = l.mem.read_u32(l.cbuf0 + 0x40)? as u64;
    let blocks = l.mem.read_u32(l.cbuf0 + 0x44)? as usize;
    if !(w | y).is_multiple_of(4) || !x.is_multiple_of(16) {
        return Err(bad("SIM: q8_matvec32 cere W, y aliniate la 4 si x la 16"));
    }
    let row_bytes = blocks * gpu_compute::q8::BLOCK_BYTES;
    let xs = l.mem.read_f32s(x, blocks * gpu_compute::q8::BLOCK)?;
    let mut row = alloc::vec![0u8; row_bytes];
    for r in 0..rows.min(l.grid[0] as u64) {
        l.mem.read(w + r * row_bytes as u64, &mut row)?;
        let v = gpu_compute::q8::row_gpu_order_w32(&row, &xs);
        l.mem.write_u32(y + 4 * r, v.to_bits());
    }
    Ok(())
}

/// C-147 point 5: where NVK 26.3.1 puts the push constants of `mma16_f16` in cbuf 0 (`c[0x0][0x700]` in
/// `tools/nak/out2/mma16_f16.sm86.txt`; the first NAK tool, Mesa 25.0.7, put them at 0x28).
pub const MMA16_PUSH_BASE: u64 = 0x700;

/// C-147 point 5: what `mma16_f16` needs: no shared memory, no barrier (one warp; the dump says
/// `Shared size: 0`, no `BAR.SYNC`).
pub const MMA16_F16_NEEDS: Needs = Needs {
    shared_bytes: 0,
    barriers: 0,
};

/// C-147 point 5: the CPU stand-in for `tools/nak/out2/mma16_f16.sm86.sass` (second NAK tool, Mesa
/// 26.3.1; two `hmma.m16n8k16.f32`): cbuf0 a 0x700, b 0x708, c 0x710, d 0x718 (all 16-aligned); one
/// block of 32×1×1; D = A·B + C (16×16, A/B binary16, C/D f32, row-major). The arithmetic is
/// `gpu_compute::mma::mma16_block8_trunc` — a HYPOTHESIS of how the tensor core adds, not a fact.
pub fn mma16_f16_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::mma;
    if l.block != [32, 1, 1] || l.grid != [1, 1, 1] {
        return Err(bad("SIM: mma16_f16 cere un bloc de 32x1x1"));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + MMA16_PUSH_BASE + o);
    let (a, b, c, d) = (r(l, 0)?, r(l, 8)?, r(l, 0x10)?, r(l, 0x18)?);
    if !(a | b | c | d).is_multiple_of(16) {
        return Err(bad("SIM: mma16_f16 cere a, b, c, d aliniate la 16"));
    }
    let halves = |l: &Launch<'_>, va: u64| -> GpuResult<Vec<u16>> {
        let mut raw = alloc::vec![0u8; 2 * mma::M * mma::K];
        l.mem.read(va, &mut raw)?;
        Ok(raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|h| u16::from_le_bytes(*h))
            .collect())
    };
    let (ha, hb) = (halves(l, a)?, halves(l, b)?);
    let fc = l.mem.read_f32s(c, mma::M * mma::N)?;
    let out =
        mma::mma16_block8_trunc(&ha, &hb, &fc).ok_or_else(|| bad("SIM: mma16_f16 dimensiuni"))?;
    l.mem.write_f32s(d, &out);
    Ok(())
}

/// B1: what `tools/nak/out2/mma16_s8.sm86.sass` needs: no shared memory, no barrier (one warp; the dump
/// says `Shared size: 0`, no `BAR.SYNC`).
pub const MMA16_S8_NEEDS: Needs = Needs {
    shared_bytes: 0,
    barriers: 0,
};

/// B1: the CPU stand-in for `tools/nak/out2/mma16_s8.sm86.sass` (second NAK tool, Mesa 26.3.1; two
/// `imma.m16n8k32.i8.i8`): cbuf0 a 0x700, b 0x708, c 0x710, d 0x718 (all 16-aligned); one block of 32×1×1;
/// D = A·B + C (A 16×32 and B 32×16 in i8, C/D 16×16 in i32, row-major). The arithmetic is
/// `gpu_compute::mma_s8::mma16_s8_wrap`: exact whenever the result fits in i32; the wrap on overflow is a
/// HYPOTHESIS, not a fact.
pub fn mma16_s8_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::mma_s8 as s8;
    if l.block != [32, 1, 1] || l.grid != [1, 1, 1] {
        return Err(bad("SIM: mma16_s8 cere un bloc de 32x1x1"));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + MMA16_PUSH_BASE + o);
    let (a, b, c, d) = (r(l, 0)?, r(l, 8)?, r(l, 0x10)?, r(l, 0x18)?);
    if [a, b, c, d].contains(&0) || !(a | b | c | d).is_multiple_of(16) {
        return Err(bad(
            "SIM: mma16_s8 cere a, b, c, d nenule si aliniate la 16",
        ));
    }
    let bytes = |l: &Launch<'_>, va: u64, n: usize| -> GpuResult<Vec<u8>> {
        let mut raw = alloc::vec![0u8; n];
        l.mem.read(va, &mut raw)?;
        Ok(raw)
    };
    let ia: Vec<i8> = bytes(l, a, s8::M * s8::K)?
        .into_iter()
        .map(|v| v as i8)
        .collect();
    let ib: Vec<i8> = bytes(l, b, s8::K * s8::N)?
        .into_iter()
        .map(|v| v as i8)
        .collect();
    let ic: Vec<i32> = bytes(l, c, 4 * s8::M * s8::N)?
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| i32::from_le_bytes(*w))
        .collect();
    let out = s8::mma16_s8_wrap(&ia, &ib, &ic).ok_or_else(|| bad("SIM: mma16_s8 dimensiuni"))?;
    let raw: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    l.mem.write(d, &raw);
    Ok(())
}

/// C-160 point 7: what `tools/nak/out2/mm_tc.sm86.sass` needs — the f16 slices of op(A) (64×32) and op(B)
/// (32×64) plus the 64×64 f32 block for the end: 4 + 4 + 16 KiB of shared memory, one barrier.
pub const MM_TC_NEEDS: Needs = Needs {
    shared_bytes: (64 * 32 + 32 * 64) * 2 + 64 * 64 * 4,
    barriers: 1,
};

/// C-160 point 7: the CPU stand-in for `tools/nak/out2/mm_tc.sm86.sass` (second NAK tool, Mesa 26.3.1): the
/// training matmul on the tensor cores, `C = op(A)·op(B) (+ bias)` with op(A), op(B) rounded to f16. cbuf0 (NVK
/// 26.3.1 puts the push constants at 0x700): a/b/c/bias at 0x700/0x708/0x710/0x718, m/n/k/flags at
/// 0x720/0x724/0x728/0x72c. Blocks of 128×1×1, grid (⌈n/64⌉, ⌈m/64⌉, 1). The arithmetic is
/// `gpu_compute::mm_tc::model` — the tensor-core adder fitted on os-335, a HYPOTHESIS, not a fact. Unknown flags,
/// a bias flag without a bias, or inputs outside the f16 range are refused (no write).
pub fn mm_tc_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [128, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: mm_tc cere blocuri de 128x1x1"));
    }
    mm_tc_tiles_on_cpu(l, 64)
}

/// C-157 p.2 after os-423 (session 229): what `tools/nak/out2/mm_tc_pf.sm86.sass` needs — exactly `mm_tc`'s shared
/// memory (the f16 slices + the 64×64 f32 tile, 24 KiB) and one barrier; the loads read ahead live in registers.
pub const MM_TC_PF_NEEDS: Needs = MM_TC_NEEDS;

/// C-157 p.2 after os-423 (session 229): the CPU stand-in for `tools/nak/out2/mm_tc_pf.sm86.sass` — `mm_tc` with the
/// global loads issued ahead (branch-free, clamped index, zero chosen after the load; the next K-slice read while the
/// tensor cores work on the current one). The values copied to shared memory, the f16 rounding, the K order and the
/// bias add are `mm_tc`'s, so the stand-in is [`mm_tc_on_cpu`] (blocks of 128×1×1, 64×64 tiles) — the same
/// `gpu_compute::mm_tc::model`, a HYPOTHESIS fitted on os-335 and matched bit for bit by `mm_tc` on the board (os-423).
pub fn mm_tc_pf_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_tc_on_cpu(l)
}

/// C-157 p.2 (session 230, `docs/C-157_P2_CITIRI_128.md`): what `tools/nak/out2/mm_tc_v4.sm86.sass` needs — exactly
/// `mm_tc`'s shared memory and one barrier (the 128-bit loads land in registers).
pub const MM_TC_V4_NEEDS: Needs = MM_TC_NEEDS;

/// C-157 p.2 (session 230): the CPU stand-in for `tools/nak/out2/mm_tc_v4.sm86.sass` — `mm_tc_pf` with 128-bit global
/// loads (4 neighbouring floats per load). Valid only for aligned products: the stored row length of A (k, or m with
/// TA) and of B (n, or k with TB) a multiple of 4, and A / B at multiples of 16 bytes; anything else would be a
/// misaligned or straddling 128-bit load on the board, so the model REFUSES it (no write) instead of guessing. For
/// aligned products the values copied to shared memory are `mm_tc`'s (T236), so the stand-in is [`mm_tc_on_cpu`].
pub fn mm_tc_v4_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    let base = l.cbuf0 + MMA16_PUSH_BASE;
    let a = l.mem.read_u64(base)?;
    let b = l.mem.read_u64(base + 0x08)?;
    let m = l.mem.read_u32(base + 0x20)?;
    let n = l.mem.read_u32(base + 0x24)?;
    let k = l.mem.read_u32(base + 0x28)?;
    let flags = l.mem.read_u32(base + 0x2c)?;
    let a_row = if flags & 1 != 0 { m } else { k };
    let b_row = if flags & 2 != 0 { k } else { n };
    if !a_row.is_multiple_of(4)
        || !b_row.is_multiple_of(4)
        || !a.is_multiple_of(16)
        || !b.is_multiple_of(16)
    {
        return Err(bad(
            "SIM: mm_tc_v4: produs nealiniat (randuri multiplu de 4, adrese multiplu de 16)",
        ));
    }
    mm_tc_on_cpu(l)
}

/// C-165 after os-416 (session 220): what `tools/nak/out2/mm_tc128.sm86.sass` needs — the f16 slices of op(A)
/// (128×32) and op(B) (32×128) plus one 16×16 f32 scratch per subgroup (8 of them): 8 + 8 + 8 KiB, one barrier.
pub const MM_TC128_NEEDS: Needs = Needs {
    shared_bytes: (128 * 32 + 32 * 128) * 2 + 8 * 16 * 16 * 4,
    barriers: 1,
};

/// C-165 after os-416 (session 220): the CPU stand-in for `tools/nak/out2/mm_tc128.sm86.sass` — `mm_tc`'s job,
/// cbuf0 and arithmetic (every element: the same chain of tensor-core steps, K in steps of 16 in order from 0, then
/// one f32 add of the bias) on 128×128 tiles: blocks of 256×1×1, grid (⌈n/128⌉, ⌈m/128⌉, 1). The arithmetic is the
/// same `gpu_compute::mm_tc::model` as [`mm_tc_on_cpu`] — a HYPOTHESIS fitted on os-335, not a fact.
pub fn mm_tc128_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: mm_tc128 cere blocuri de 256x1x1"));
    }
    mm_tc_tiles_on_cpu(l, 128)
}

/// C-157 p.2 (session 231, `docs/C-157_P2_MM_TC128_V4.md`): what `tools/nak/out2/mm_tc128_v4.sm86.sass` needs —
/// exactly `mm_tc128`'s shared memory and one barrier (the 128-bit loads read ahead land in registers).
pub const MM_TC128_V4_NEEDS: Needs = MM_TC128_NEEDS;

/// C-157 p.2 (session 231): the CPU stand-in for `tools/nak/out2/mm_tc128_v4.sm86.sass` — `mm_tc128` with the next
/// K-slice read ahead in 128-bit global loads (as `mm_tc_v4`). Valid only for aligned products (the same rule as
/// [`mm_tc_v4_on_cpu`]); anything else is REFUSED (no write) instead of guessed. For aligned products the values
/// copied to shared memory are `mm_tc128`'s (T237), so the stand-in is [`mm_tc128_on_cpu`].
pub fn mm_tc128_v4_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: mm_tc128_v4 cere blocuri de 256x1x1"));
    }
    if !tc_aligned(l)? {
        return Err(bad(
            "SIM: mm_tc128_v4: produs nealiniat (randuri multiplu de 4, adrese multiplu de 16)",
        ));
    }
    mm_tc128_on_cpu(l)
}

/// The alignment rule of the 128-bit programs (`mm_tc_v4`, `mm_tc128_v4`), read from cbuf0 as the board would see it.
fn tc_aligned(l: &Launch<'_>) -> GpuResult<bool> {
    let base = l.cbuf0 + MMA16_PUSH_BASE;
    let a = l.mem.read_u64(base)?;
    let b = l.mem.read_u64(base + 0x08)?;
    let m = l.mem.read_u32(base + 0x20)?;
    let n = l.mem.read_u32(base + 0x24)?;
    let k = l.mem.read_u32(base + 0x28)?;
    let flags = l.mem.read_u32(base + 0x2c)?;
    let a_row = if flags & 1 != 0 { m } else { k };
    let b_row = if flags & 2 != 0 { k } else { n };
    Ok(a_row.is_multiple_of(4)
        && b_row.is_multiple_of(4)
        && a.is_multiple_of(16)
        && b.is_multiple_of(16))
}

/// B1 step 3 (session 225, `docs/B1_MM_S8.md`): what `tools/nak/out2/mm_s8.sm86.sass` needs — the i8 slices of
/// op(A) (128×32) and op(B) (32×128) plus one 16×16 i32 scratch per subgroup (8 of them): 4 + 4 + 8 KiB, one barrier.
pub const MM_S8_NEEDS: Needs = Needs {
    shared_bytes: 128 * 32 + 32 * 128 + 8 * 16 * 16 * 4,
    barriers: 1,
};

/// B1 step 3 (session 225): the CPU stand-in for `tools/nak/out2/mm_s8.sm86.sass` (second NAK tool, Mesa 26.3.1;
/// `imma.m16n8k32.i8.i8`): `C = op(A)·op(B) (+ bias)` with A, B in i8 and C, bias in i32. cbuf0 as `mm_tc`
/// (a/b/c/bias at 0x700/0x708/0x710/0x718, m/n/k/flags at 0x720/0x724/0x728/0x72c, flags TA/TB/BIAS); blocks of
/// 256×1×1, grid (⌈n/128⌉, ⌈m/128⌉, 1). The arithmetic is `gpu_compute::mma_s8::mm_s8_wrap`: exact whenever the
/// result fits in i32, in any order (integers); the wrap on overflow is a HYPOTHESIS, not a fact.
pub fn mm_s8_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    use gpu_compute::{mm_tc, mma_s8};
    if l.block != [256, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: mm_s8 cere blocuri de 256x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let base = l.cbuf0 + MMA16_PUSH_BASE;
    let a = l.mem.read_u64(base)?;
    let b = l.mem.read_u64(base + 0x08)?;
    let c = l.mem.read_u64(base + 0x10)?;
    let bias_va = l.mem.read_u64(base + 0x18)?;
    let m = l.mem.read_u32(base + 0x20)? as usize;
    let n = l.mem.read_u32(base + 0x24)? as usize;
    let k = l.mem.read_u32(base + 0x28)? as usize;
    let flags = l.mem.read_u32(base + 0x2c)?;
    if flags & !(mm_tc::TA | mm_tc::TB | mm_tc::BIAS) != 0 {
        return Err(bad("SIM: mm_s8: biti de optiuni necunoscuti"));
    }
    if (l.grid[0] as usize) < n.div_ceil(128) || (l.grid[1] as usize) < m.div_ceil(128) {
        return Err(bad("SIM: mm_s8: grila nu acopera C (nemodelat)"));
    }
    let i8s = |l: &Launch<'_>, va: u64, len: usize| -> GpuResult<Vec<i8>> {
        let mut raw = alloc::vec![0u8; len];
        l.mem.read(va, &mut raw)?;
        Ok(raw.into_iter().map(|v| v as i8).collect())
    };
    let x = i8s(l, a, m * k)?;
    let y = i8s(l, b, k * n)?;
    let bias = if flags & mm_tc::BIAS != 0 {
        let mut raw = alloc::vec![0u8; 4 * n];
        l.mem.read(bias_va, &mut raw)?;
        Some(
            raw.as_chunks::<4>()
                .0
                .iter()
                .map(|w| i32::from_le_bytes(*w))
                .collect::<Vec<i32>>(),
        )
    } else {
        None
    };
    let p = mma_s8::ProductS8 {
        a: &x,
        b: &y,
        bias: bias.as_deref(),
        m,
        n,
        k,
        flags,
    };
    let out = mma_s8::mm_s8_wrap(&p).ok_or_else(|| bad("SIM: mm_s8: intrari invalide"))?;
    let raw: Vec<u8> = out.iter().flat_map(|v| v.to_le_bytes()).collect();
    l.mem.write(c, &raw);
    Ok(())
}

/// `mm_tc` / `mm_tc128` after the block check: cbuf0, refusals and the product, for C tiles of `tile`×`tile`.
fn mm_tc_tiles_on_cpu(l: &mut Launch<'_>, tile: usize) -> GpuResult<()> {
    use gpu_compute::mm_tc;
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let base = l.cbuf0 + MMA16_PUSH_BASE;
    let a = l.mem.read_u64(base)?;
    let b = l.mem.read_u64(base + 0x08)?;
    let c = l.mem.read_u64(base + 0x10)?;
    let bias_va = l.mem.read_u64(base + 0x18)?;
    let m = l.mem.read_u32(base + 0x20)? as usize;
    let n = l.mem.read_u32(base + 0x24)? as usize;
    let k = l.mem.read_u32(base + 0x28)? as usize;
    let flags = l.mem.read_u32(base + 0x2c)?;
    if flags & !(mm_tc::TA | mm_tc::TB | mm_tc::BIAS) != 0 {
        return Err(bad("SIM: mm_tc: biti de optiuni necunoscuti"));
    }
    if (l.grid[0] as usize) < n.div_ceil(tile) || (l.grid[1] as usize) < m.div_ceil(tile) {
        return Err(bad("SIM: mm_tc: grila nu acopera C (nemodelat)"));
    }
    let x = l.mem.read_f32s(a, m * k)?;
    let y = l.mem.read_f32s(b, k * n)?;
    let bias = if flags & mm_tc::BIAS != 0 {
        Some(l.mem.read_f32s(bias_va, n)?)
    } else {
        None
    };
    let p = mm_tc::Product {
        a: &x,
        b: &y,
        bias: bias.as_deref(),
        m,
        n,
        k,
        flags,
    };
    let out =
        mm_tc::model(&p).ok_or_else(|| bad("SIM: mm_tc: intrari invalide (f16 sau dimensiuni)"))?;
    l.mem.write_f32s(c, &out);
    Ok(())
}

/// C-157 p.2 (`docs/C-160_P7_LIPIRE_GELU_TC.md`): what `tools/nak/out2/mm_tc_gelu.sm86.sass` needs — `mm_tc`'s.
pub const MM_TC_GELU_NEEDS: Needs = MM_TC_NEEDS;

/// C-157 p.2: the CPU stand-in for `tools/nak/out2/mm_tc_gelu.sm86.sass` (session 199): [`mm_tc_on_cpu`]'s product
/// unchanged, then each element of C is `gpu_compute::layer::gelu_tr_gpu_back`(u[row, col], v) — `gelu_tr` mode 1
/// with the tensor-core sum v in place of g. cbuf0 as `mm_tc` with `u` in the bias slot (0x718); the BIAS flag is
/// refused (the program has no bias), and so is a C overlapping u (other blocks would read u already overwritten).
pub fn mm_tc_gelu_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_tc_gelu_with(l, gpu_compute::layer::gelu_tr_gpu_back)
}

/// [`mm_tc_gelu_on_cpu`] with `gpu_compute::layer::gelu_tr_back_unfused_gpu` (Φ + u·φ in two roundings): what the
/// program would do if NAK had compiled its epilogue unlike `gelu_tr`'s. Not a model of our program — T207 registers
/// it to show the bit-for-bit comparison catches that difference.
pub fn mm_tc_gelu_unfused_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    mm_tc_gelu_with(l, gpu_compute::layer::gelu_tr_back_unfused_gpu)
}

fn mm_tc_gelu_with(l: &mut Launch<'_>, f: fn(f32, f32) -> f32) -> GpuResult<()> {
    use gpu_compute::mm_tc;
    if l.block != [128, 1, 1] || l.grid[2] != 1 {
        return Err(bad("SIM: mm_tc_gelu cere blocuri de 128x1x1"));
    }
    if l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: baza grupurilor (cbuf0 0x0) diferita de 0, nemodelata",
        ));
    }
    let base = l.cbuf0 + MMA16_PUSH_BASE;
    let a = l.mem.read_u64(base)?;
    let b = l.mem.read_u64(base + 0x08)?;
    let c = l.mem.read_u64(base + 0x10)?;
    let u_va = l.mem.read_u64(base + 0x18)?;
    let m = l.mem.read_u32(base + 0x20)? as usize;
    let n = l.mem.read_u32(base + 0x24)? as usize;
    let k = l.mem.read_u32(base + 0x28)? as usize;
    let flags = l.mem.read_u32(base + 0x2c)?;
    if flags & !(mm_tc::TA | mm_tc::TB) != 0 {
        return Err(bad("SIM: mm_tc_gelu: biti de optiuni necunoscuti sau BIAS"));
    }
    if (l.grid[0] as usize) < n.div_ceil(64) || (l.grid[1] as usize) < m.div_ceil(64) {
        return Err(bad("SIM: mm_tc_gelu: grila nu acopera C (nemodelat)"));
    }
    let bytes = 4 * (m * n) as u64;
    if c < u_va + bytes && u_va < c + bytes {
        return Err(bad("SIM: mm_tc_gelu: C se suprapune cu u"));
    }
    let x = l.mem.read_f32s(a, m * k)?;
    let y = l.mem.read_f32s(b, k * n)?;
    let u = l.mem.read_f32s(u_va, m * n)?;
    let p = mm_tc::Product {
        a: &x,
        b: &y,
        bias: None,
        m,
        n,
        k,
        flags,
    };
    let v = mm_tc::model(&p)
        .ok_or_else(|| bad("SIM: mm_tc_gelu: intrari invalide (f16 sau dimensiuni)"))?;
    let out: Vec<f32> = v.iter().zip(&u).map(|(&v, &u)| f(u, v)).collect();
    l.mem.write_f32s(c, &out);
    Ok(())
}

/// C-83 I13: `vadd.comp` — cbuf0 a 0x28, b 0x30, c 0x38, n 0x40; blocks of 256.
pub fn vadd_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 || l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad("SIM: vadd cere blocuri de 256x1x1, grila 1D, baza 0"));
    }
    let (a, b, c) = (
        l.mem.read_u64(l.cbuf0 + 0x28)?,
        l.mem.read_u64(l.cbuf0 + 0x30)?,
        l.mem.read_u64(l.cbuf0 + 0x38)?,
    );
    let n = (l.mem.read_u32(l.cbuf0 + 0x40)? as u64).min(l.grid[0] as u64 * 256) as usize;
    let out = gpu_compute::layer::vadd_gpu(&l.mem.read_f32s(a, n)?, &l.mem.read_f32s(b, n)?);
    l.mem.write_f32s(c, &out);
    Ok(())
}

/// C-83 I13: what `layernorm` / `attention` need: shared memory (128 B / 2 176 B), no hardware barrier (one warp).
pub const LAYERNORM_NEEDS: Needs = Needs {
    shared_bytes: 128,
    barriers: 0,
};
pub const ATTENTION_NEEDS: Needs = Needs {
    shared_bytes: 4 * (512 + 32),
    barriers: 0,
};

/// C-155 point 2: what `layernorm256` needs: 32 bytes of shared memory (8 warp sums) and one hardware barrier.
pub const LAYERNORM256_NEEDS: Needs = Needs {
    shared_bytes: 32,
    barriers: 1,
};

/// C-155 point 2: `layernorm256.comp` — the same cbuf as `layernorm.comp` (x 0x28, g 0x30, b 0x38, y 0x40, n 0x48,
/// inv_n 0x4c, eps 0x50); one block of 256; n at most 1024.
pub fn layernorm256_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid != [1, 1, 1] {
        return Err(bad("SIM: layernorm256 cere un bloc de 256x1x1"));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (x, g, b, y) = (r(l, 0x28)?, r(l, 0x30)?, r(l, 0x38)?, r(l, 0x40)?);
    let n = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let inv_n = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x4c)?);
    let eps = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x50)?);
    if n == 0 || n > gpu_compute::layer::LN256_MAX_N {
        return Err(bad("SIM: layernorm256 cu n in afara 1..1024"));
    }
    if inv_n.to_bits() != (1.0 / n as f32).to_bits() {
        return Err(bad("SIM: layernorm256 cu inv_n diferit de 1/n"));
    }
    let out = gpu_compute::layer::layernorm_gpu256(
        &l.mem.read_f32s(x, n)?,
        &l.mem.read_f32s(g, n)?,
        &l.mem.read_f32s(b, n)?,
        eps,
    )?;
    l.mem.write_f32s(y, &out);
    Ok(())
}

/// C-155 point 3: what `argmax` needs: 96 bytes of shared memory (8 warp picks) and one hardware barrier.
pub const ARGMAX_NEEDS: Needs = Needs {
    shared_bytes: 96,
    barriers: 1,
};

/// C-155 point 3: `argmax.comp` — logits 0x28, result 0x30 (16-aligned), n 0x38; one block of 256. Writes the 4
/// words of `gpu_compute::model::argmax_result` (the program's result for any split of the elements: a maximum
/// with the lower index on ties needs no arithmetic) at the result address.
pub fn argmax_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid != [1, 1, 1] {
        return Err(bad("SIM: argmax cere un bloc de 256x1x1"));
    }
    let logits = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let res = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let n = l.mem.read_u32(l.cbuf0 + 0x38)? as usize;
    if !logits.is_multiple_of(4) || !res.is_multiple_of(16) {
        return Err(bad("SIM: argmax cu adrese nealiniate"));
    }
    let words = gpu_compute::model::argmax_result(&l.mem.read_f32s(logits, n)?);
    for (i, w) in words.iter().enumerate() {
        l.mem.write_u32(res + 4 * i as u64, *w);
    }
    Ok(())
}

/// C-83 I13: `layernorm.comp` — x 0x28, g 0x30, b 0x38, y 0x40, n 0x48, inv_n 0x4c, eps 0x50; one block of 32.
pub fn layernorm_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [32, 1, 1] || l.grid != [1, 1, 1] {
        return Err(bad("SIM: layernorm cere un bloc de 32x1x1"));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (x, g, b, y) = (r(l, 0x28)?, r(l, 0x30)?, r(l, 0x38)?, r(l, 0x40)?);
    let n = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let inv_n = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x4c)?);
    let eps = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x50)?);
    if n == 0 || inv_n.to_bits() != (1.0 / n as f32).to_bits() {
        return Err(bad("SIM: layernorm cu inv_n diferit de 1/n"));
    }
    let out = gpu_compute::layer::layernorm_gpu(
        &l.mem.read_f32s(x, n)?,
        &l.mem.read_f32s(g, n)?,
        &l.mem.read_f32s(b, n)?,
        eps,
    );
    l.mem.write_f32s(y, &out);
    Ok(())
}

/// C-83 I13: `attention.comp` — q 0x28, kc 0x30, vc 0x38, o 0x40, n_pos 0x48, n_emb 0x4c, d 0x50, scale 0x54; one
/// block of 32 per head (grid x = heads).
pub fn attention_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [32, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 || l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad("SIM: attention cere blocuri de 32x1x1, grila = capete"));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (q, kc, vc, o) = (r(l, 0x28)?, r(l, 0x30)?, r(l, 0x38)?, r(l, 0x40)?);
    let n_pos = l.mem.read_u32(l.cbuf0 + 0x48)? as usize;
    let e = l.mem.read_u32(l.cbuf0 + 0x4c)? as usize;
    let d = l.mem.read_u32(l.cbuf0 + 0x50)? as usize;
    let scale = f32::from_bits(l.mem.read_u32(l.cbuf0 + 0x54)?);
    let heads = l.grid[0] as usize;
    if n_pos == 0 || n_pos > 512 || d == 0 || d > 128 || heads * d > e {
        return Err(bad("SIM: attention cu forme in afara programului"));
    }
    let qs = l.mem.read_f32s(q, e)?;
    let ks = l.mem.read_f32s(kc, n_pos * e)?;
    let vs = l.mem.read_f32s(vc, n_pos * e)?;
    let args = gpu_compute::layer::AttnArgs { n_pos, e, d, scale };
    for h in 0..heads {
        let out = gpu_compute::layer::attention_head_gpu(&qs, &ks, &vs, &args, h);
        l.mem.write_f32s(o + 4 * (h * d) as u64, &out);
    }
    Ok(())
}

/// C-83 I13: `gelu.comp` — x 0x28, y 0x30, n 0x38; blocks of 256.
pub fn gelu_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 || l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad("SIM: gelu cere blocuri de 256x1x1, grila 1D, baza 0"));
    }
    let (x, y) = (
        l.mem.read_u64(l.cbuf0 + 0x28)?,
        l.mem.read_u64(l.cbuf0 + 0x30)?,
    );
    let n = (l.mem.read_u32(l.cbuf0 + 0x38)? as u64).min(l.grid[0] as u64 * 256) as usize;
    let out: Vec<f32> = l
        .mem
        .read_f32s(x, n)?
        .into_iter()
        .map(gpu_compute::layer::gelu_gpu)
        .collect();
    l.mem.write_f32s(y, &out);
    Ok(())
}

/// C-147 step 2: what `q8_fused` needs: the 32 partial sums of `q8_matvec` (128 B), no hardware barrier (one warp).
pub const Q8_FUSED_NEEDS: Needs = Needs {
    shared_bytes: 128,
    barriers: 0,
};

/// C-147 step 2: the CPU stand-in for `tools/nak/out/q8_fused.sm86.sass` — W 0x28, x 0x30, bias 0x38, res 0x40,
/// y 0x48, k 0x50, v 0x58, rows 0x60, blocks 0x64, flags 0x68 (1 residual, 2 GELU, 4 K/V), split 0x6c; one block of
/// 32 per row. Every row: `q8::row_gpu_order` (as `q8_matvec`), `+ bias` and `res +` with `FADD.FTZ` (as `vadd`),
/// `gelu_gpu` (as `gelu`): the composition of the separate launches' models, in their order. Blocks past `rows`
/// write nothing; a pointer the flags do not use is never read.
pub fn q8_fused_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    q8_fused_with(l, false)
}

/// C-155 point 4: what `q8_fused32` needs: the same 32 partial sums (128 B), no hardware barrier (one warp).
pub const Q8_FUSED32_NEEDS: Needs = Needs {
    shared_bytes: 128,
    barriers: 0,
};

/// C-155 point 4: the CPU stand-in for `tools/nak/out/q8_fused32.sm86.sass` — the cbuf of `q8_fused`, byte for byte;
/// x must be 16-aligned (the 128-bit loads: otherwise the launch is refused, as an unaligned `LDG.128` would fault).
/// Every row: `q8::row_gpu_order_w32` (as `q8_matvec32`), then the tail of `q8_fused`.
pub fn q8_fused32_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    q8_fused_with(l, true)
}

fn q8_fused_with(l: &mut Launch<'_>, w32: bool) -> GpuResult<()> {
    use gpu_compute::math::add_f32_ftz;
    if l.block != [32, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 || l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: q8_fused cere blocuri de 32x1x1, grila 1D, baza 0",
        ));
    }
    let r = |l: &Launch<'_>, o: u64| l.mem.read_u64(l.cbuf0 + o);
    let (w, x, bias, res) = (r(l, 0x28)?, r(l, 0x30)?, r(l, 0x38)?, r(l, 0x40)?);
    let (y, k, v) = (r(l, 0x48)?, r(l, 0x50)?, r(l, 0x58)?);
    let rows = l.mem.read_u32(l.cbuf0 + 0x60)? as u64;
    let blocks = l.mem.read_u32(l.cbuf0 + 0x64)? as usize;
    let flags = l.mem.read_u32(l.cbuf0 + 0x68)?;
    let split = l.mem.read_u32(l.cbuf0 + 0x6c)? as u64;
    if !(w | x | bias | res | y | k | v).is_multiple_of(4) {
        return Err(bad("SIM: q8_fused cu adrese nealiniate la 4"));
    }
    if w32 && !x.is_multiple_of(16) {
        return Err(bad("SIM: q8_fused32 cere x aliniat la 16"));
    }
    let row_bytes = blocks * gpu_compute::q8::BLOCK_BYTES;
    let xs = l.mem.read_f32s(x, blocks * gpu_compute::q8::BLOCK)?;
    let mut row = alloc::vec![0u8; row_bytes];
    for i in 0..rows.min(l.grid[0] as u64) {
        l.mem.read(w + i * row_bytes as u64, &mut row)?;
        let dot = if w32 {
            gpu_compute::q8::row_gpu_order_w32(&row, &xs)
        } else {
            gpu_compute::q8::row_gpu_order(&row, &xs)
        };
        let mut val = add_f32_ftz(dot, f32::from_bits(l.mem.read_u32(bias + 4 * i)?));
        if flags & 1 != 0 {
            val = add_f32_ftz(f32::from_bits(l.mem.read_u32(res + 4 * i)?), val);
        }
        if flags & 2 != 0 {
            val = gpu_compute::layer::gelu_gpu(val);
        }
        l.mem.write_u32(y + 4 * i, val.to_bits());
        if flags & 4 != 0 && i >= split {
            if i < 2 * split {
                l.mem.write_u32(k + 4 * (i - split), val.to_bits());
            } else if i < 3 * split {
                l.mem.write_u32(v + 4 * (i - 2 * split), val.to_bits());
            }
        }
    }
    Ok(())
}

/// C-83 I14: `embed.comp` — token row 0x28, position row 0x30 (both Q8_0), x 0x38, n 0x40; blocks of 256.
pub fn embed_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 || l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad("SIM: embed cere blocuri de 256x1x1, grila 1D, baza 0"));
    }
    let (t, p, x) = (
        l.mem.read_u64(l.cbuf0 + 0x28)?,
        l.mem.read_u64(l.cbuf0 + 0x30)?,
        l.mem.read_u64(l.cbuf0 + 0x38)?,
    );
    let n = (l.mem.read_u32(l.cbuf0 + 0x40)? as u64).min(l.grid[0] as u64 * 256) as usize;
    let bytes = n.div_ceil(gpu_compute::q8::BLOCK) * gpu_compute::q8::BLOCK_BYTES;
    let (mut tr, mut pr) = (alloc::vec![0u8; bytes], alloc::vec![0u8; bytes]);
    l.mem.read(t, &mut tr)?;
    l.mem.read(p, &mut pr)?;
    l.mem
        .write_f32s(x, &gpu_compute::model::embed_gpu(&tr, &pr, n));
    Ok(())
}

/// C-157 point 7: `embed_dev.comp` — token table 0x28, argmax result 0x30 (its word 0 = the token), position row 0x38,
/// x 0x40, n 0x48, vocab 0x4c, row stride in bytes 0x50; blocks of 256. The token row is `table + idx * stride`, read
/// only when `idx < vocab`; otherwise x[0..n] = NaN (0x7fc00000) and the table is not touched. The value is the same
/// `embed_gpu` as [`embed_on_cpu`] (the program keeps embed.comp's expression).
pub fn embed_dev_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [256, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 || l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad(
            "SIM: embed_dev cere blocuri de 256x1x1, grila 1D, baza 0",
        ));
    }
    let (t, r, p, x) = (
        l.mem.read_u64(l.cbuf0 + 0x28)?,
        l.mem.read_u64(l.cbuf0 + 0x30)?,
        l.mem.read_u64(l.cbuf0 + 0x38)?,
        l.mem.read_u64(l.cbuf0 + 0x40)?,
    );
    let n = (l.mem.read_u32(l.cbuf0 + 0x48)? as u64).min(l.grid[0] as u64 * 256) as usize;
    let vocab = l.mem.read_u32(l.cbuf0 + 0x4c)?;
    let stride = l.mem.read_u32(l.cbuf0 + 0x50)? as u64;
    let idx = l.mem.read_u32(r)?;
    if idx >= vocab {
        l.mem
            .write_f32s(x, &alloc::vec![f32::from_bits(0x7fc0_0000); n]);
        return Ok(());
    }
    let bytes = n.div_ceil(gpu_compute::q8::BLOCK) * gpu_compute::q8::BLOCK_BYTES;
    let (mut tr, mut pr) = (alloc::vec![0u8; bytes], alloc::vec![0u8; bytes]);
    l.mem.read(t + idx as u64 * stride, &mut tr)?;
    l.mem.read(p, &mut pr)?;
    l.mem
        .write_f32s(x, &gpu_compute::model::embed_gpu(&tr, &pr, n));
    Ok(())
}

/// C-87 step 2: what `token.comp` (one program per token) needs: the attention's 2 176 B of shared memory.
pub const TOKEN_NEEDS: Needs = Needs {
    shared_bytes: 4 * (512 + 32),
    barriers: 0,
};

/// C-87 step 2: `token.comp` — list 0x28, sync 0x30, n 0x38; G workgroups of 32. The model walks the work list (256 B
/// per operation: operation at 0x20, grid at 0x24, arguments from 0x28) and runs each operation through the SAME model
/// as its separate launch, with the separate launch's block size; between operations it adds G to the barrier
/// counter, as the G workgroups would. It shows the list and its arguments are right; it does NOT model the grid
/// barrier's timing or coherence on a real SM (one operation is finished before the next starts, by construction).
pub fn token_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    // C-139: `token.comp` reads its work data with coherent loads
    // (`ld.global.strong.sys`, tools/nak/README.md) and its weights never
    // change, so the stale-read model does not apply inside it.
    let was = l.mem.set_in_program(false);
    let ran = token_ops(l);
    l.mem.set_in_program(was);
    ran
}

/// C-157 (session 239, `docs/C-157_TOKEN32.md`): `token32.comp` — the same list, constant buffer and checks as
/// `token.comp`, with the 13 operations of `kronos_token::op_number`; each runs through its separate program's model
/// with that program's block size (256 for `layernorm256`, `argmax`, `embed_dev`). What it shows and does not show is
/// what [`token_on_cpu`] says: the list and the arguments, not the grid barrier on a real SM; and that the 32-thread
/// bodies of `layernorm256` / `argmax` give the bits of the 256-thread programs is DEDUS from the source and the
/// listing (`docs/C-157_TOKEN32.md`), not modelled here.
pub fn token32_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    let was = l.mem.set_in_program(false);
    let ran = token_ops_with(l, &TOKEN32_OPS);
    l.mem.set_in_program(was);
    ran
}

const TOKEN32_OPS: [(ProgramFn, u32); 13] = [
    (copy_b128_on_cpu, 256),
    (q8_matvec_on_cpu, 32),
    (vadd_on_cpu, 256),
    (layernorm_on_cpu, 32),
    (attention_on_cpu, 32),
    (gelu_on_cpu, 256),
    (embed_on_cpu, 256),
    (q8_fused_on_cpu, 32),
    (layernorm256_on_cpu, 256),
    (argmax_on_cpu, 256),
    (q8_fused32_on_cpu, 32),
    (q8_matvec32_on_cpu, 32),
    (embed_dev_on_cpu, 256),
];

/// What `token32.comp` needs: the attention's 2 176 B of shared memory, as `token.comp`.
pub const TOKEN32_NEEDS: Needs = TOKEN_NEEDS;

fn token_ops(l: &mut Launch<'_>) -> GpuResult<()> {
    const OPS: [(ProgramFn, u32); 7] = [
        (copy_b128_on_cpu, 256),
        (q8_matvec_on_cpu, 32),
        (vadd_on_cpu, 256),
        (layernorm_on_cpu, 32),
        (attention_on_cpu, 32),
        (gelu_on_cpu, 256),
        (embed_on_cpu, 256),
    ];
    token_ops_with(l, &OPS)
}

fn token_ops_with(l: &mut Launch<'_>, ops: &[(ProgramFn, u32)]) -> GpuResult<()> {
    if l.block != [32, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 || l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad("SIM: token cere blocuri de 32x1x1, grila 1D, baza 0"));
    }
    let (list, sync) = (
        l.mem.read_u64(l.cbuf0 + 0x28)?,
        l.mem.read_u64(l.cbuf0 + 0x30)?,
    );
    let n = l.mem.read_u32(l.cbuf0 + 0x38)? as u64;
    if n == 0 {
        return Err(bad("SIM: token fara operatii"));
    }
    // Session 237: `token.comp` reads gl_NumWorkGroups.x from NVK's root table at 0x0c (listing lines 3188 / 3226:
    // c[0x0][0xc]) for its block stride and the barrier target; 0 there = `vb += 0` (a hang) on the board.
    if l.mem.read_u32(l.cbuf0 + 0x0c)? != l.grid[0] {
        return Err(bad(
            "SIM: token: numarul grupurilor de la 0x0c difera de grila (programul il citeste de acolo)",
        ));
    }
    if l.mem.read_u32(sync)? != 0 {
        return Err(bad(
            "SIM: token: contorul barierei nu a fost golit inainte de lansare",
        ));
    }
    let g = l.grid[0];
    for k in 0..n {
        let e = list + 256 * k;
        let op = l.mem.read_u32(e + 0x20)? as usize;
        let grid = l.mem.read_u32(e + 0x24)?;
        let &(run, block) = ops
            .get(op)
            .ok_or(bad("SIM: token: operatie necunoscuta in lista"))?;
        if grid == 0 {
            return Err(bad("SIM: token: operatie cu grila 0"));
        }
        run(&mut Launch {
            mem: l.mem,
            cbuf0: e,
            grid: [grid, 1, 1],
            block: [block, 1, 1],
        })?;
        if k + 1 < n {
            l.mem.write_u32(sync, (k as u32 + 1) * g);
        }
    }
    Ok(())
}

/// C-157 (docs/C-157_PROBA_BARIERA.md): what `grid_barrier.comp` needs — no shared memory, no hardware barrier
/// beyond the workgroup's own.
pub const GRID_BARRIER_NEEDS: Needs = Needs {
    shared_bytes: 0,
    barriers: 0,
};

/// C-157: the grid-barrier probe — sync 0x28, data 0x30, out 0x38, rounds 0x40; G workgroups of 32. The model plays
/// the G workgroups round by round as a CORRECT grid barrier would order them: every workgroup writes its word, the
/// counter reaches k·G, then every workgroup reads its neighbour's word (in a rotated order each round, so no single
/// order is baked in). It shows the arguments, the two halves and the expected results (counter R·G, out = 0); it says
/// NOTHING about whether the barrier holds on a real SM (residency, L1 coherence, hangs) — that is the board's job.
pub fn grid_barrier_on_cpu(l: &mut Launch<'_>) -> GpuResult<()> {
    if l.block != [32, 1, 1] || l.grid[1] != 1 || l.grid[2] != 1 || l.mem.read_u64(l.cbuf0)? != 0 {
        return Err(bad("SIM: bariera cere blocuri de 32x1x1, grila 1D, baza 0"));
    }
    let (sync, data, out) = (
        l.mem.read_u64(l.cbuf0 + 0x28)?,
        l.mem.read_u64(l.cbuf0 + 0x30)?,
        l.mem.read_u64(l.cbuf0 + 0x38)?,
    );
    let rounds = l.mem.read_u32(l.cbuf0 + 0x40)?;
    let g = l.grid[0] as u64;
    if g == 0 || rounds == 0 {
        return Err(bad("SIM: bariera fara grupuri sau fara runde"));
    }
    // the program reads gl_NumWorkGroups.x from NVK's root table at 0x0c (listing: c[0x0][0xc])
    if l.mem.read_u32(l.cbuf0 + 0x0c)? as u64 != g {
        return Err(bad(
            "SIM: bariera: numarul grupurilor de la 0x0c difera de grila (programul il citeste de acolo)",
        ));
    }
    if l.mem.read_u32(sync)? != 0 {
        return Err(bad(
            "SIM: bariera: contorul nu a fost golit inainte de lansare",
        ));
    }
    // the buffers must exist (a real program would fault on them)
    l.mem.read_u32(data + 4 * (2 * g - 1))?;
    l.mem.read_u32(out + 4 * (g - 1))?;
    let mut wrong = alloc::vec![0u32; g as usize];
    for k in 1..=rounds {
        let half = (k as u64 & 1) * g;
        for w in 0..g {
            l.mem.write_u32(data + 4 * (half + w), k);
            let c = l.mem.read_u32(sync)?;
            l.mem.write_u32(sync, c.wrapping_add(1));
        }
        if l.mem.read_u32(sync)? != k.wrapping_mul(g as u32) {
            return Err(bad("SIM: bariera: contorul nu a ajuns la k*G"));
        }
        for i in 0..g {
            let w = (i + k as u64) % g;
            if l.mem.read_u32(data + 4 * (half + (w + 1) % g))? != k {
                wrong[w as usize] += 1;
            }
        }
    }
    for (w, &n) in wrong.iter().enumerate() {
        l.mem.write_u32(out + 4 * w as u64, n);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Headers written by hand here (INC = 1 << 29, count 16, subch 13),
    // independent of nvidia-ga104::push.
    fn inc(subch: u32, m: u32, n: u32) -> u32 {
        (1 << 29) | (n << 16) | (subch << 13) | (m >> 2)
    }

    fn run(words: &[u32]) -> (CommandProcessor, GpuResult<()>) {
        let mut cp = CommandProcessor::new();
        for (i, &w) in words.iter().enumerate() {
            cp.mem.write_u32(0x1000 + 4 * i as u64, w);
        }
        let e0 = 0x1000;
        let e1 = (words.len() as u32) << 10;
        let r = cp.run_gpfifo(&[(e0, e1)]);
        (cp, r)
    }

    #[test]
    fn setup_methods_are_recorded() {
        let (cp, r) = run(&[
            inc(1, 0, 1),
            c::AMPERE_COMPUTE_B,
            inc(1, 0x7b0, 2),
            0x7293,
            0x10,
            inc(1, 0x790, 2),
            0x10,
            0x2000,
            inc(1, 0x2e4, 3),
            0,
            0x18000,
            0xff,
        ]);
        r.unwrap();
        assert_eq!(cp.compute.local_window, Some(0x7293_0000_0010));
        assert_eq!(cp.compute.local_memory, Some(0x10_0000_2000));
        assert_eq!(cp.compute.local_per_tpc, Some(0x18000));
        assert_eq!(cp.compute.local_max_sm, Some(0xff));
        assert_eq!(cp.compute.shared_window, None);
    }

    #[test]
    fn setup_mistakes_are_refused() {
        let obj = [inc(1, 0, 1), c::AMPERE_COMPUTE_B];
        for (what, tail) in [
            ("B fara A", &[inc(1, 0x7b4, 1), 0][..]),
            ("A peste camp", &[inc(1, 0x7b0, 2), 1 << 17, 0][..]),
            (
                "B al altei perechi",
                &[inc(1, 0x7b0, 1), 1, inc(1, 0x2a4, 1), 0][..],
            ),
            ("MAX_SM peste camp", &[inc(1, 0x2ec, 1), 0x200][..]),
        ] {
            let w: Vec<u32> = obj.iter().chain(tail).copied().collect();
            assert!(run(&w).1.is_err(), "{what}");
        }
        // Without SET_OBJECT first: refused too.
        assert!(run(&[inc(1, 0x7b0, 2), 1, 0]).1.is_err());
    }

    #[test]
    fn channel_ring_wraps_and_writes_gpget() {
        let mut cp = CommandProcessor::new();
        let (ring, userd) = (0x10_0000u64, 0x20_0000u64);
        cp.mem.write_u32(0x1000, inc(1, 0, 1));
        cp.mem.write_u32(0x1004, c::AMPERE_COMPUTE_B);
        // Slots 3 and 0 (wrap in a 4-entry ring), GET = 3, PUT = 1.
        for slot in [3u64, 0] {
            cp.mem.write_u32(ring + 8 * slot, 0x1000);
            cp.mem.write_u32(ring + 8 * slot + 4, 2 << 10);
        }
        cp.mem.write_u32(userd + USERD_GP_GET, 3);
        cp.mem.write_u32(userd + USERD_GP_PUT, 1);
        assert_eq!(cp.run_channel(ring, 4, userd), Ok(2));
        assert_eq!(cp.mem.read_u32(userd + USERD_GP_GET), Ok(1));
        assert_eq!(cp.events.len(), 2);
        // Nothing new: nothing taken.
        assert_eq!(cp.run_channel(ring, 4, userd), Ok(0));
        // PUT outside the ring.
        cp.mem.write_u32(userd + USERD_GP_PUT, 4);
        assert!(cp.run_channel(ring, 4, userd).is_err());
        assert!(cp.run_channel(ring, 3, userd).is_err());
    }
}
