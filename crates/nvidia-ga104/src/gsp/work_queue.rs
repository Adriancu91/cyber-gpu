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
/// (`NV2080_CTRL_CMD_GR_GET_INFO`, not implemented yet); they are inputs
/// here, never assumed.
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

/// The host side of the working channel: the ring indices, where ring and
/// USERD live, and a linear pushbuffer area (not recycled yet).
#[derive(Debug)]
pub struct WorkQueue {
    ring: Ring,
    ring_pa: u64,
    userd_pa: u64,
    push: Mapping,
    push_used: u64,
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
            push_used: 0,
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
        let bytes = 4 * words.len() as u64;
        if self.push.size - self.push_used < bytes {
            return Err(GpuError::OutOfMemory { requested: bytes });
        }
        let va = self.push.va + self.push_used;
        let entry = tinygrad_entry(va, words.len() as u32)?;
        let mut ring = self.ring;
        let slot = ring.reserve(1)?;
        let pa = self.push.pa + self.push_used;
        for (i, &d) in words.iter().enumerate() {
            w.write_u32(pa + 4 * i as u64, d)?;
        }
        let slot_pa = self.ring_pa + 8 * slot as u64;
        w.write_u32(slot_pa, entry.0)?;
        w.write_u32(slot_pa + 4, entry.1)?;
        w.write_u32(self.userd_pa + USERD_GP_PUT, ring.put())?;
        self.ring = ring;
        self.push_used += bytes;
        Ok(Doorbell {
            bar0_offset: DOORBELL_INFERRED,
            value: self.token,
        })
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
}
