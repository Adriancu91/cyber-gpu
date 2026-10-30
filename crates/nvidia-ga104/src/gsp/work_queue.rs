//! I10a — the first pushbuffer on the working channel (I9e), written into its
//! GPFIFO ring and announced in USERD. Built and tested **only in SIMULARE**
//! (the `gpu_sim::cmdproc` model reads the ring back); nothing here has run
//! on the board, and nothing here writes a register: the doorbell is only
//! *described* ([`Doorbell`]), never rung.
//!
//! Specification read (not translated), re-written here:
//! - tinygrad `ops_nv.py` (MIT, commit f5b17be): `nv_submit` (~l. 96–101:
//!   the GPFIFO entry is the pushbuffer address + `len << 42` + bit 41, i.e.
//!   `LEVEL = SUBROUTINE`; the ring slot is written, then `GPPut`, then the
//!   token goes to the doorbell), `NVDevice.fifos` (~l. 618–620: once per
//!   compute channel, `SET_OBJECT` + the local and shared memory windows on
//!   subchannel 1), the window addresses (~l. 602) and
//!   `_ensure_has_local_memory` (~l. 680–689: the local memory buffer and its
//!   per-TPC size);
//! - NVIDIA `clc56f.h` (`AmpereAControlGPFifo`: `GPGet` 0x88, read-only for
//!   the host; `GPPut` 0x8c) and `clc7c0.h` (the method numbers and fields),
//!   both through the generated crate `nvidia-rm570` (MIT).
//!
//! What a passing test proves: the bytes we put in memory are the ones the
//! headers (and the model built on them) expect, in the order tinygrad uses.
//! It does not prove the RTX 3070 Ti accepts them, and on the board the
//! writes would also need ordering (a write barrier before `GPPut` and before
//! the doorbell) — the memory path for that is the OS's part (BAR1 or system
//! memory) and is not here.

use alloc::collections::VecDeque;
use gpu_command::ring::Ring;
use gpu_core::error::{GpuError, GpuResult};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};
use nvidia_rm570::rmapi::AmpereAControlGPFifo;

use super::golden::Mapping;
use super::user_space::WorkChannel;
use crate::mmu::{VA_BASE, VA_BITS, VA_SIZE};
use crate::push::{AMPERE_COMPUTE_B, DOORBELL_INFERRED, PushBuf, gpfifo_entry};

/// The subchannel tinygrad binds the compute class to (`nvm(1, ...)`).
pub const COMPUTE_SUBCH: u8 = 1;

/// tinygrad's window addresses ("to not collide with other allocated
/// buffers", ~l. 602). Outside our VA range ([`VA_BASE`], [`VA_SIZE`]).
pub const LOCAL_MEM_WINDOW: u64 = 0x7293_0000_0000;
pub const SHARED_MEM_WINDOW: u64 = 0x7294_0000_0000;
/// How much VA we keep free around each window: tinygrad's spacing (4 GiB).
/// The real size of a window is not in the headers (NECUNOSCUT); NVK uses
/// windows 16 MiB apart, so 4 GiB is on the safe side.
pub const WINDOW_SPAN: u64 = 1 << 32;

/// `USERD` offsets, from the generated `AmpereAControlGPFifo`.
pub const USERD_GP_GET: u64 = core::mem::offset_of!(AmpereAControlGPFifo, GPGet) as u64;
pub const USERD_GP_PUT: u64 = core::mem::offset_of!(AmpereAControlGPFifo, GPPut) as u64;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// `[upper, lower]` of a GPU VA for an `_A`/`_B` method pair whose upper
/// field is `upper` (17 bits on C7C0 = a 49-bit VA).
fn split(va: u64, upper: nvidia_rm570::Field) -> GpuResult<[u32; 2]> {
    let hi = (va >> 32) as u32;
    if va >> VA_BITS != 0 || upper.set(0, hi) >> upper.lo != hi {
        return Err(bad("adresa GPU nu incape in campul metodei"));
    }
    Ok([hi, va as u32])
}

/// The two windows must be outside our VA range and must not overlap.
pub fn check_windows(local: u64, shared: u64) -> GpuResult<()> {
    let ours = VA_BASE..VA_BASE + VA_SIZE;
    for w in [local, shared] {
        if w % WINDOW_SPAN != 0 || w.checked_add(WINDOW_SPAN).is_none_or(|e| e >> VA_BITS != 0) {
            return Err(bad("fereastra: nealiniata la 4 GiB sau peste 49 de biti"));
        }
        if w < ours.end && ours.start < w + WINDOW_SPAN {
            return Err(bad("fereastra se suprapune cu adresele noastre GPU"));
        }
    }
    if local.abs_diff(shared) < WINDOW_SPAN {
        return Err(bad("ferestrele locala si partajata se suprapun"));
    }
    Ok(())
}

/// The once-per-channel compute setup, as tinygrad: `SET_OBJECT` (compute
/// class on [`COMPUTE_SUBCH`]), then `SET_SHADER_LOCAL_MEMORY_WINDOW_A/B`,
/// then `SET_SHADER_SHARED_MEMORY_WINDOW_A/B`.
pub fn compute_setup(pb: &mut PushBuf<'_>, local: u64, shared: u64) -> GpuResult<()> {
    check_windows(local, shared)?;
    let lw = split(
        local,
        c::NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A_BASE_ADDRESS_UPPER,
    )?;
    let sw = split(
        shared,
        c::NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A_BASE_ADDRESS_UPPER,
    )?;
    let class = c::NVC7C0_SET_OBJECT_CLASS_ID.set(0, AMPERE_COMPUTE_B);
    pb.inc(COMPUTE_SUBCH, c::NVC7C0_SET_OBJECT, &[class])?;
    pb.inc(
        COMPUTE_SUBCH,
        c::NVC7C0_SET_SHADER_LOCAL_MEMORY_WINDOW_A,
        &lw,
    )?;
    pb.inc(
        COMPUTE_SUBCH,
        c::NVC7C0_SET_SHADER_SHARED_MEMORY_WINDOW_A,
        &sw,
    )
}

/// What the local memory size depends on. All four come from GSP-RM
/// (`user_space::query_gr_info`, I10c: `NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO`);
/// they are inputs here, never assumed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GrShape {
    pub gpcs: u32,
    pub tpc_per_gpc: u32,
    pub sm_per_tpc: u32,
    pub max_warps_per_sm: u32,
}

/// Local memory sizes, tinygrad's arithmetic (~l. 683–685).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocalMemory {
    pub per_thread: u32,
    pub per_tpc: u64,
    pub total: u64,
}

fn round_up(v: u64, to: u64) -> Option<u64> {
    v.checked_add(to - 1).map(|x| x / to * to)
}

/// `required` bytes of local memory per thread (from the compiler's info
/// sheet) → the buffer the channel needs. 0 needs nothing (`None`).
pub fn local_memory_size(required: u32, g: GrShape) -> GpuResult<Option<LocalMemory>> {
    if required == 0 {
        return Ok(None);
    }
    if [g.gpcs, g.tpc_per_gpc, g.sm_per_tpc, g.max_warps_per_sm].contains(&0) {
        return Err(bad("forma GR cu 0 (raspuns GSP-RM lipsa)"));
    }
    let too_big = || bad("memoria locala ceruta e prea mare");
    let per_thread = round_up(required as u64, 32).ok_or_else(too_big)?;
    let per_warp = round_up(per_thread * 32, 0x200).ok_or_else(too_big)?;
    let per_tpc = per_warp
        .checked_mul(g.max_warps_per_sm as u64 * g.sm_per_tpc as u64)
        .and_then(|v| round_up(v, 0x8000))
        .ok_or_else(too_big)?;
    let total = per_tpc
        .checked_mul(g.tpc_per_gpc as u64 * g.gpcs as u64)
        .and_then(|v| round_up(v, 0x2_0000))
        .ok_or_else(too_big)?;
    if per_thread > u32::MAX as u64 || total >= VA_SIZE {
        return Err(too_big());
    }
    Ok(Some(LocalMemory {
        per_thread: per_thread as u32,
        per_tpc,
        total,
    }))
}

/// `SET_SHADER_LOCAL_MEMORY_A/B` = the buffer `buf`, then
/// `SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A/B/C` = per-TPC size + `0xff`
/// (tinygrad's max SM count). The buffer must be ours and big enough.
pub fn local_memory_setup(pb: &mut PushBuf<'_>, buf: &Mapping, lm: &LocalMemory) -> GpuResult<()> {
    if buf.size < lm.total || buf.va < VA_BASE || buf.va + buf.size > VA_BASE + VA_SIZE {
        return Err(bad(
            "bufferul de memorie locala: prea mic sau in afara adreselor noastre",
        ));
    }
    let a = split(buf.va, c::NVC7C0_SET_SHADER_LOCAL_MEMORY_A_ADDRESS_UPPER)?;
    let s = split(
        lm.per_tpc,
        c::NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A_SIZE_UPPER,
    )?;
    let max_sm = c::NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_C_MAX_SM_COUNT.set(0, 0xff);
    pb.inc(COMPUTE_SUBCH, c::NVC7C0_SET_SHADER_LOCAL_MEMORY_A, &a)?;
    pb.inc(
        COMPUTE_SUBCH,
        c::NVC7C0_SET_SHADER_LOCAL_MEMORY_NON_THROTTLED_A,
        &[s[0], s[1], max_sm],
    )
}

// ---- I10b: one launch and its completion semaphore (SIMULARE) ----

/// `addr` must be inside our VA range with `len` bytes after it.
fn in_our_va(addr: u64, len: u64) -> bool {
    addr >= VA_BASE
        && addr
            .checked_add(len)
            .is_some_and(|e| e <= VA_BASE + VA_SIZE)
}

/// `INVALIDATE_SHADER_CACHES_NO_WFI` with instruction, global data and
/// constant caches — tinygrad's `memory_barrier` (~l. 154–157), so the GPU
/// does not run a stale copy of a program, QMD or argument just written.
pub fn invalidate_caches(pb: &mut PushBuf<'_>) -> GpuResult<()> {
    let t = c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION_TRUE;
    let mut v = c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_INSTRUCTION.set(0, t);
    v = c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_GLOBAL_DATA.set(v, t);
    v = c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI_CONSTANT.set(v, t);
    pb.inc(
        COMPUTE_SUBCH,
        c::NVC7C0_INVALIDATE_SHADER_CACHES_NO_WFI,
        &[v],
    )
}

/// One launch, as tinygrad's `exec` for the first QMD of a queue (~l.
/// 179–181): `SEND_PCAS_A` = QMD address >> 8, then
/// `SEND_SIGNALING_PCAS2_B` = `PREFETCH_SCHEDULE`. The QMD (256 B,
/// 256-aligned) must be in our VA range.
pub fn launch(pb: &mut PushBuf<'_>, qmd_va: u64) -> GpuResult<()> {
    if !qmd_va.is_multiple_of(256) || !in_our_va(qmd_va, nvidia_rm570::QMD_SIZE as u64) {
        return Err(bad("QMD: nealiniat la 256 sau in afara adreselor noastre"));
    }
    let shifted = qmd_va >> 8;
    let f = c::NVC7C0_SEND_PCAS_A_QMD_ADDRESS_SHIFTED8;
    if f.set(0, shifted as u32) >> f.lo != shifted as u32 || shifted >> 32 != 0 {
        return Err(bad("QMD: adresa nu incape in SEND_PCAS_A"));
    }
    pb.inc(
        COMPUTE_SUBCH,
        c::NVC7C0_SEND_PCAS_A,
        &[f.set(0, shifted as u32)],
    )?;
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION.set(
        0,
        c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_PREFETCH_SCHEDULE,
    );
    pb.inc(COMPUTE_SUBCH, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
}

/// The host semaphore release tinygrad puts after work (`release`, ~l.
/// 111–119): `SEM_ADDR_LO/HI`, `SEM_PAYLOAD_LO/HI`, `SEM_EXECUTE` =
/// `RELEASE` + `RELEASE_WFI_EN` (wait for the launch to finish) + 64-bit
/// payload, no timestamp, on subchannel 0. tinygrad then sends
/// `NON_STALL_INTERRUPT`; we do not (we poll, and there is no interrupt
/// handler) — noted in `COMPATIBILITATE_FIRMWARE.md`.
pub fn release(pb: &mut PushBuf<'_>, sem_va: u64, value: u64) -> GpuResult<()> {
    if !sem_va.is_multiple_of(8) || !in_our_va(sem_va, 8) {
        return Err(bad(
            "semafor: nealiniat la 8 sau in afara adreselor noastre",
        ));
    }
    let mut exec = h::NVC56F_SEM_EXECUTE_OPERATION.set(0, h::NVC56F_SEM_EXECUTE_OPERATION_RELEASE);
    exec = h::NVC56F_SEM_EXECUTE_RELEASE_WFI.set(exec, h::NVC56F_SEM_EXECUTE_RELEASE_WFI_EN);
    exec = h::NVC56F_SEM_EXECUTE_PAYLOAD_SIZE.set(exec, h::NVC56F_SEM_EXECUTE_PAYLOAD_SIZE_64BIT);
    pb.inc(
        0,
        h::NVC56F_SEM_ADDR_LO,
        &[
            sem_va as u32,
            (sem_va >> 32) as u32,
            value as u32,
            (value >> 32) as u32,
            exec,
        ],
    )
}

/// Reads a 64-bit semaphore at physical `pa` with 32-bit loads: high,
/// low, high again; a high half that changed in between is re-read once
/// (the GPU writes the payload in one go, we read it in two).
pub fn read_semaphore<W: ChannelMemory + ?Sized>(w: &mut W, pa: u64) -> GpuResult<u64> {
    if !pa.is_multiple_of(8) {
        return Err(bad("semafor nealiniat la 8"));
    }
    for _ in 0..2 {
        let hi = w.read_u32(pa + 4)?;
        let lo = w.read_u32(pa)?;
        if w.read_u32(pa + 4)? == hi {
            return Ok(((hi as u64) << 32) | lo as u64);
        }
    }
    Err(GpuError::Timeout { waited_us: 0 })
}

/// 32-bit stores into VRAM at physical addresses (ring, USERD, pushbuffer).
/// SIMULARE: a model; on the board: the OS's mapping, in the window with
/// Adrian. `GPPut` is a 32-bit field next to the read-only `GPGet`, so a
/// 64-bit store would be wrong.
pub trait ChannelMemory {
    fn write_u32(&mut self, pa: u64, v: u32) -> GpuResult<()>;
    fn read_u32(&mut self, pa: u64) -> GpuResult<u32>;
}

/// The register write that would start the channel. **Never performed
/// here.** `bar0_offset` is tinygrad's (usermode region at 0xbb0000 + 0x90).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Doorbell {
    pub bar0_offset: u64,
    pub value: u32,
}

/// One GPFIFO entry as tinygrad writes it: [`gpfifo_entry`] (no SYNC) with
/// `LEVEL = SUBROUTINE` (bit 41 of the 64-bit entry).
pub fn tinygrad_entry(va: u64, dwords: u32) -> GpuResult<(u32, u32)> {
    let e = gpfifo_entry(va, dwords, false)?;
    Ok((
        e.entry0,
        h::NVC56F_GP_ENTRY1_LEVEL.set(e.entry1, h::NVC56F_GP_ENTRY1_LEVEL_SUBROUTINE),
    ))
}

/// One stretch of the pushbuffer area still owned by the GPU: `[start, end)`
/// (offsets in the area) and the completion semaphore that frees it
/// (`(semaphore VA, value)`), known once a later [`WorkQueue::launch_and_signal`]
/// releases one. The channel runs its GPFIFO entries in order, so a
/// semaphore released by a later entry means the earlier words were read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct InFlight {
    start: u64,
    end: u64,
    fence: Option<(u64, u64)>,
}

/// The host side of the working channel: the ring indices, where ring and
/// USERD live, and the pushbuffer area, used as a circle: words are written
/// only where no submission still in flight lives, and a stretch is given
/// back only by [`WorkQueue::retire`], after its completion semaphore.
#[derive(Debug)]
pub struct WorkQueue {
    ring: Ring,
    ring_pa: u64,
    userd_pa: u64,
    push: Mapping,
    in_flight: VecDeque<InFlight>,
    last_fence: Option<(u64, u64)>,
    token: u32,
}

impl WorkQueue {
    /// `push` must be a mapped buffer of ours, apart from the ring.
    pub fn new(ch: &WorkChannel, push: Mapping) -> GpuResult<Self> {
        let ring = Ring::new(ch.entries)?;
        if ch.userd_offset < ch.entries as u64 * 8 || !ch.userd_offset.is_multiple_of(0x200) {
            return Err(bad("USERD nealiniat sau peste inel"));
        }
        let (r0, r1) = (ch.ring.pa, ch.ring.pa + ch.ring.size);
        if push.size == 0
            || !push.va.is_multiple_of(4)
            || (push.pa < r1 && r0 < push.pa + push.size)
        {
            return Err(bad("bufferul de comenzi: gol, nealiniat sau peste inel"));
        }
        Ok(WorkQueue {
            ring,
            ring_pa: ch.ring.pa,
            userd_pa: ch.ring.pa + ch.userd_offset,
            push,
            in_flight: VecDeque::new(),
            last_fence: None,
            token: ch.doorbell_token,
        })
    }

    pub fn put(&self) -> u32 {
        self.ring.put()
    }

    pub fn get(&self) -> u32 {
        self.ring.get()
    }

    /// Copies `words` into the pushbuffer area, writes one GPFIFO entry in
    /// the next ring slot, then `GPPut` — in that order — and returns the
    /// doorbell write that would follow (not performed). On a refusal
    /// nothing is written.
    pub fn submit<W: ChannelMemory + ?Sized>(
        &mut self,
        w: &mut W,
        words: &[u32],
    ) -> GpuResult<Doorbell> {
        self.submit_inner(w, words, None)
    }

    /// Where `bytes` fit in the area without touching a stretch in flight:
    /// after the newest one, or from the start of the area when the end is
    /// too short and the oldest one starts far enough. `None` = no room
    /// until something is retired.
    fn place(&self, bytes: u64) -> Option<u64> {
        let size = self.push.size;
        let (Some(old), Some(new)) = (self.in_flight.front(), self.in_flight.back()) else {
            return (bytes <= size).then_some(0);
        };
        if new.start >= old.start {
            // Not wrapped: free = [new.end, size) and [0, old.start).
            if size - new.end >= bytes {
                Some(new.end)
            } else if old.start >= bytes {
                Some(0)
            } else {
                None
            }
        } else {
            // Wrapped: free = [new.end, old.start).
            (old.start - new.end >= bytes).then_some(new.end)
        }
    }

    fn submit_inner<W: ChannelMemory + ?Sized>(
        &mut self,
        w: &mut W,
        words: &[u32],
        fence: Option<(u64, u64)>,
    ) -> GpuResult<Doorbell> {
        let bytes = 4 * words.len() as u64;
        let off = self
            .place(bytes)
            .ok_or(GpuError::OutOfMemory { requested: bytes })?;
        let va = self.push.va + off;
        let entry = tinygrad_entry(va, words.len() as u32)?;
        let mut ring = self.ring;
        let slot = ring.reserve(1)?;
        let pa = self.push.pa + off;
        for (i, &d) in words.iter().enumerate() {
            w.write_u32(pa + 4 * i as u64, d)?;
        }
        let slot_pa = self.ring_pa + 8 * slot as u64;
        w.write_u32(slot_pa, entry.0)?;
        w.write_u32(slot_pa + 4, entry.1)?;
        w.write_u32(self.userd_pa + USERD_GP_PUT, ring.put())?;
        self.ring = ring;
        if let Some(f) = fence {
            // Earlier stretches without a semaphore of their own are freed
            // by this one (the entries run in order).
            for r in self.in_flight.iter_mut().rev() {
                if r.fence.is_some() {
                    break;
                }
                r.fence = Some(f);
            }
            self.last_fence = Some(f);
        }
        self.in_flight.push_back(InFlight {
            start: off,
            end: off + bytes,
            fence,
        });
        Ok(Doorbell {
            bar0_offset: DOORBELL_INFERRED,
            value: self.token,
        })
    }

    /// I10b: one pushbuffer with [`invalidate_caches`], [`launch`] of the
    /// QMD at `qmd_va` and [`release`] of `value` at `sem_va`, submitted as
    /// one GPFIFO entry. The driver then reads the semaphore
    /// ([`read_semaphore`]) to know the work is done. Nothing is written on
    /// a refusal.
    pub fn launch_and_signal<W: ChannelMemory + ?Sized>(
        &mut self,
        w: &mut W,
        qmd_va: u64,
        sem_va: u64,
        value: u64,
    ) -> GpuResult<Doorbell> {
        let mut buf = [0u32; 16];
        let mut pb = PushBuf::new(&mut buf);
        invalidate_caches(&mut pb)?;
        launch(&mut pb, qmd_va)?;
        release(&mut pb, sem_va, value)?;
        if let Some((va, last)) = self.last_fence
            && va == sem_va
            && value <= last
        {
            return Err(bad("valoarea semaforului nu creste"));
        }
        let n = pb.len();
        self.submit_inner(w, &buf[..n], Some((sem_va, value)))
    }

    /// Gives back the pushbuffer stretches whose completion semaphore at
    /// `sem_va` has reached `completed` (the value the driver read with
    /// [`read_semaphore`]), oldest first; stops at the first one not yet
    /// done. A value above the last one we asked the GPU to write is a
    /// fault (nothing is freed). Returns how many stretches were freed.
    pub fn retire(&mut self, sem_va: u64, completed: u64) -> GpuResult<u32> {
        match self.last_fence {
            Some((va, last)) if va == sem_va && completed > last => {
                return Err(bad("semafor peste ultima valoare trimisa"));
            }
            None => return Err(bad("niciun semafor trimis")),
            _ => {}
        }
        let mut freed = 0;
        while let Some(r) = self.in_flight.front() {
            match r.fence {
                Some((va, v)) if va == sem_va && v <= completed => {
                    self.in_flight.pop_front();
                    freed += 1;
                }
                _ => break,
            }
        }
        Ok(freed)
    }

    /// Bytes of the pushbuffer area still owned by the GPU (in flight).
    pub fn push_in_flight(&self) -> u64 {
        self.in_flight.iter().map(|r| r.end - r.start).sum()
    }

    /// Reads `GPGet` from USERD; a value outside [get, put] is a fault.
    /// Returns how many entries the GPU took since the last call.
    pub fn update_get<W: ChannelMemory + ?Sized>(&mut self, w: &mut W) -> GpuResult<u32> {
        let g = w.read_u32(self.userd_pa + USERD_GP_GET)?;
        self.ring.update_get(g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;
    use nvidia_rm570::regs::clc56f;

    /// The circle's arithmetic alone: where a stretch of `bytes` may go,
    /// given what is in flight (offsets in a 128-byte area).
    #[test]
    fn placement_never_reaches_a_stretch_in_flight() {
        let push = Mapping {
            va: VA_BASE,
            pa: 0x10_0000,
            size: 128,
        };
        let q = |live: &[(u64, u64)]| WorkQueue {
            ring: Ring::new(8).unwrap(),
            ring_pa: 0,
            userd_pa: 0x40,
            push,
            in_flight: live
                .iter()
                .map(|&(start, end)| InFlight {
                    start,
                    end,
                    fence: None,
                })
                .collect(),
            last_fence: None,
            token: 0,
        };
        // (in flight, bytes, expected place)
        type Case = (&'static [(u64, u64)], u64, Option<u64>);
        let cases: &[Case] = &[
            (&[], 128, Some(0)),
            (&[], 132, None),
            (&[(0, 48)], 80, Some(48)),
            (&[(0, 48)], 84, None),
            (&[(40, 100)], 28, Some(100)),
            (&[(40, 100)], 32, Some(0)), // end too short, start long enough
            (&[(40, 100)], 44, None),    // would reach the stretch at 40
            (&[(40, 100), (0, 16)], 24, Some(16)), // wrapped: [16, 40) free
            (&[(40, 100), (0, 16)], 28, None),
            (&[(40, 100), (100, 128), (0, 40)], 4, None), // full
        ];
        for &(live, bytes, want) in cases {
            assert_eq!(q(live).place(bytes), want, "{live:?} {bytes}");
        }
    }

    fn words(f: impl FnOnce(&mut PushBuf<'_>) -> GpuResult<()>) -> Vec<u32> {
        let mut mem = [0u32; 32];
        let mut pb = PushBuf::new(&mut mem);
        f(&mut pb).unwrap();
        pb.words().to_vec()
    }

    #[test]
    fn userd_offsets_from_the_headers() {
        assert_eq!((USERD_GP_GET, USERD_GP_PUT), (0x88, 0x8c));
    }

    /// The exact dwords tinygrad's `nvm(1, ...)` gives for the same setup.
    #[test]
    fn compute_setup_matches_tinygrad() {
        let w = words(|pb| compute_setup(pb, LOCAL_MEM_WINDOW, SHARED_MEM_WINDOW));
        let nvm = |m: u32, n: u32| (2 << 28) | (n << 16) | (1 << 13) | (m >> 2);
        assert_eq!(
            w,
            [
                nvm(0x0, 1),
                0xc7c0,
                nvm(0x7b0, 2),
                0x7293,
                0,
                nvm(0x2a0, 2),
                0x7294,
                0
            ]
        );
    }

    #[test]
    fn windows_are_checked() {
        assert!(check_windows(LOCAL_MEM_WINDOW, SHARED_MEM_WINDOW).is_ok());
        // Inside our VA range, overlapping, unaligned, past 49 bits.
        for (l, s) in [
            (VA_BASE, SHARED_MEM_WINDOW),
            (LOCAL_MEM_WINDOW, VA_BASE + VA_SIZE - WINDOW_SPAN),
            (LOCAL_MEM_WINDOW, LOCAL_MEM_WINDOW),
            (LOCAL_MEM_WINDOW + 0x1000, SHARED_MEM_WINDOW),
            (1 << 49, SHARED_MEM_WINDOW),
            ((1 << 49) - WINDOW_SPAN, SHARED_MEM_WINDOW),
        ] {
            assert!(check_windows(l, s).is_err(), "{l:#x} {s:#x}");
        }
        // Just below our range is fine, just touching it is not.
        assert!(check_windows(VA_BASE - WINDOW_SPAN, SHARED_MEM_WINDOW).is_ok());
        assert!(check_windows(VA_BASE + VA_SIZE, SHARED_MEM_WINDOW).is_ok());
    }

    #[test]
    fn local_memory_arithmetic() {
        let g = GrShape {
            gpcs: 6,
            tpc_per_gpc: 4,
            sm_per_tpc: 2,
            max_warps_per_sm: 48,
        };
        assert_eq!(local_memory_size(0, g), Ok(None));
        // 1 B → 32 B per thread, 1 KiB per warp → 0x200 rounded: 0x400;
        // ×48×2 = 0x18000 per TPC (already a multiple of 0x8000); ×24 TPC.
        let lm = local_memory_size(1, g).unwrap().unwrap();
        assert_eq!(
            (lm.per_thread, lm.per_tpc, lm.total),
            (32, 0x18000, 0x24_0000)
        );
        let lm = local_memory_size(40, g).unwrap().unwrap();
        assert_eq!(lm.per_thread, 64);
        assert_eq!(lm.per_tpc, 0x30000);
        assert!(local_memory_size(1, GrShape { gpcs: 0, ..g }).is_err());
        assert!(local_memory_size(u32::MAX, g).is_err());
        let buf = Mapping {
            va: VA_BASE + 0x10_0000,
            pa: 0x400_0000,
            size: 0x24_0000,
        };
        let lm = local_memory_size(1, g).unwrap().unwrap();
        let w = words(|pb| local_memory_setup(pb, &buf, &lm));
        assert_eq!(w[1..3], [0x10, 0x10_0000]);
        assert_eq!(w[4..7], [0, 0x18000, 0xff]);
        let small = Mapping {
            size: 0x1000,
            ..buf
        };
        let mut mem = [0u32; 16];
        assert!(local_memory_setup(&mut PushBuf::new(&mut mem), &small, &lm).is_err());
    }

    #[test]
    fn entry_is_tinygrads() {
        let (e0, e1) = tinygrad_entry(0x12_3456_7000, 8).unwrap();
        let tg: u64 = 0x12_3456_7000 + ((8u64 << 42) | (1 << 41));
        assert_eq!(((e1 as u64) << 32) | e0 as u64, tg);
        assert_eq!(
            clc56f::NVC56F_GP_ENTRY1_LEVEL.get(e1),
            clc56f::NVC56F_GP_ENTRY1_LEVEL_SUBROUTINE
        );
    }

    /// I10b: the exact dwords tinygrad's `memory_barrier`, `exec` (first
    /// QMD) and `release` give: `nvm(1, 0x1698, 0x1011)`, `nvm(1, 0x2b4,
    /// qmd >> 8)`, `nvm(1, 0x2c0, 9)`, `nvm(0, 0x5c, lo, hi, lo, hi,
    /// 0x0110_0001)`.
    #[test]
    fn launch_words_match_tinygrad() {
        let qmd = VA_BASE + 0x12_3400;
        let sem = VA_BASE + 0x8;
        let w = words(|pb| {
            invalidate_caches(pb)?;
            launch(pb, qmd)?;
            release(pb, sem, 0x1_0000_0007)
        });
        let nvm = |s: u32, m: u32, n: u32| (2 << 28) | (n << 16) | (s << 13) | (m >> 2);
        assert_eq!(
            w,
            [
                nvm(1, 0x1698, 1),
                0x1011,
                nvm(1, 0x2b4, 1),
                (qmd >> 8) as u32,
                nvm(1, 0x2c0, 1),
                9,
                nvm(0, 0x5c, 5),
                sem as u32,
                (sem >> 32) as u32,
                7,
                1,
                0x0110_0001,
            ]
        );
    }

    /// A 64-bit semaphore read as two halves: a high half that moves
    /// between the reads is re-read; one that never settles is an error.
    #[test]
    fn semaphore_reads_survive_a_carry() {
        struct Moving(alloc::vec::Vec<u32>);
        impl ChannelMemory for Moving {
            fn write_u32(&mut self, _: u64, _: u32) -> GpuResult<()> {
                unreachable!()
            }
            fn read_u32(&mut self, _: u64) -> GpuResult<u32> {
                Ok(self.0.remove(0))
            }
        }
        // hi, lo, hi: settled.
        assert_eq!(
            read_semaphore(&mut Moving([1, 5, 1].into()), 0x100),
            Ok(0x1_0000_0005)
        );
        // The carry happens between the two reads: 0 then 1 → read again.
        let mut m = Moving([0, 0, 1, 1, 0, 1].into());
        assert_eq!(read_semaphore(&mut m, 0x100), Ok(0x1_0000_0000));
        // Never settles.
        let mut m = Moving([0, 0, 1, 1, 0, 2].into());
        assert!(read_semaphore(&mut m, 0x100).is_err());
        assert!(read_semaphore(&mut Moving([].into()), 0x104).is_err());
    }
}
