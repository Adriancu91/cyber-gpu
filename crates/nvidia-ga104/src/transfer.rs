//! C-83 I11 — a large, safe copy between host RAM and VRAM **through the work
//! channel** (not word by word through the VRAM window), checked bit for bit,
//! its speed measured with the board's own clock.
//!
//! How (our choice, written down in `docs/GA104_INITIALIZATION.md`, I11):
//! the copy is a compute program (`tools/nak/copy_b128.comp`, compiled by Mesa
//! NAK on the host, 16 bytes per thread) launched on the same work channel and
//! compute object as I10 — no new RM object, no new register: the only new
//! thing on the board is that some PTEs point at **host RAM**
//! (`mmu::encode_pte_sys`, aperture `SYSTEM_COHERENT_MEMORY`, DEDUS). A copy
//! engine channel (`AMPERE_DMA_COPY_B`, tinygrad's `dma_gpfifo`) would need a
//! second channel and object: later, if the compute copy is too slow.
//!
//! One round trip, one GPFIFO entry ([`encode_round_trip`]):
//! record 0 (timestamp) → copy RAM→VRAM → record 1 → copy VRAM→RAM (into a
//! second RAM buffer) → record 2. Each record is a host semaphore release with
//! `RELEASE_WFI` + `RELEASE_TIMESTAMP` (as [`crate::launch_cost`]), so a copy
//! has finished before its record is written and the next copy starts after.
//! The driver then checks: the second RAM buffer equals the pattern, word for
//! word ([`first_mismatch`]); a few VRAM words read back through the window
//! (first, last, middle) equal it too; the records give the time of each
//! direction ([`analyze_round_trip`]).
//!
//! Tested **only in SIMULARE** (`gpu_sim::cmdproc` with host RAM modelled by a
//! second `SimVram`); the speeds there are the test's own numbers. Nothing in
//! this module touches a board; it only encodes and checks bytes.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::gsp::work_queue::{invalidate_caches, launch};
use crate::launch_cost::{RECORD, timestamp_release};
use crate::push::PushBuf;
use crate::qmd::ComputeQmd;

/// `tools/nak/out/copy_b128.sm86.sass` (Mesa NAK 25.0.7 for SM 8.6).
pub const COPY_B128: &[u8] = include_bytes!("../../../tools/nak/out/copy_b128.sm86.sass");
/// Registers per thread, from the NAK dump (`copy_b128.sm86.txt`).
pub const COPY_REGISTERS: u32 = 19;
/// Threads per block (the program's `local_size_x`).
pub const COPY_BLOCK: u32 = 256;
/// Bytes each thread copies.
pub const COPY_UNIT: u64 = 16;
/// Where NVK puts the push constants in cbuf 0 (as for `add64`): source,
/// destination (64-bit GPU VAs), count of 16-byte units (32-bit).
pub const CBUF_SRC: usize = 0x28;
pub const CBUF_DST: usize = 0x30;
pub const CBUF_N: usize = 0x38;
/// cbuf 0 size we give the QMD (as `add64`).
pub const CBUF_SIZE: u32 = 0x100;
/// The size suggested for the first window: 8 MiB (big enough that the
/// launch cost is small next to the copy, small enough for any DMA pool).
pub const I11_DEFAULT_BYTES: u64 = 8 << 20;
/// Largest copy one launch may do (n fits the 32-bit count; grid x < 2^31).
pub const MAX_COPY_BYTES: u64 =
    (u32::MAX as u64 / COPY_BLOCK as u64) * COPY_BLOCK as u64 * COPY_UNIT;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The 256-byte constant buffer of one copy of `bytes` bytes from `src` to
/// `dst` (GPU VAs). Refuses before anything is built: 0 bytes, a size or
/// address not a multiple of 16, too much, overlapping ranges.
pub fn copy_cbuf(src: u64, dst: u64, bytes: u64) -> GpuResult<[u8; CBUF_SIZE as usize]> {
    if bytes == 0 || !bytes.is_multiple_of(COPY_UNIT) || bytes > MAX_COPY_BYTES {
        return Err(bad("copiere: marime 0, nemultiplu de 16 sau prea mare"));
    }
    if !src.is_multiple_of(COPY_UNIT) || !dst.is_multiple_of(COPY_UNIT) {
        return Err(bad("copiere: adresa nealiniata la 16"));
    }
    let (Some(se), Some(de)) = (src.checked_add(bytes), dst.checked_add(bytes)) else {
        return Err(bad("copiere: adresa prea mare"));
    };
    if src < de && dst < se {
        return Err(bad("copiere: sursa si destinatia se suprapun"));
    }
    let mut cb = [0u8; CBUF_SIZE as usize];
    cb[CBUF_SRC..CBUF_SRC + 8].copy_from_slice(&src.to_le_bytes());
    cb[CBUF_DST..CBUF_DST + 8].copy_from_slice(&dst.to_le_bytes());
    cb[CBUF_N..CBUF_N + 4].copy_from_slice(&((bytes / COPY_UNIT) as u32).to_le_bytes());
    Ok(cb)
}

/// The QMD of one copy: grid = ⌈units / 256⌉ blocks of 256 threads.
pub fn copy_qmd(program_va: u64, cbuf_va: u64, bytes: u64) -> GpuResult<ComputeQmd> {
    if bytes == 0 || !bytes.is_multiple_of(COPY_UNIT) || bytes > MAX_COPY_BYTES {
        return Err(bad("copiere: marime 0, nemultiplu de 16 sau prea mare"));
    }
    let units = bytes / COPY_UNIT;
    Ok(ComputeQmd {
        program_va,
        registers: COPY_REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: CBUF_SIZE,
        grid: [units.div_ceil(COPY_BLOCK as u64) as u32, 1, 1],
        block: [COPY_BLOCK, 1, 1],
        release: None,
    })
}

/// The test pattern, word `i` of a buffer for `seed`: a fixed mix (splitmix64
/// finaliser) of seed and index, so neither the OS nor the board needs a
/// stored copy and a shifted, repeated or zero page is caught.
pub fn pattern_word(seed: u32, i: u64) -> u32 {
    let mut z = (((seed as u64) << 40) ^ i).wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    (z ^ (z >> 31)) as u32
}

/// Fills `buf` (length a multiple of 4) with the pattern.
pub fn fill_pattern(buf: &mut [u8], seed: u32) -> GpuResult<()> {
    if !buf.len().is_multiple_of(4) {
        return Err(bad("model: lungime nemultiplu de 4"));
    }
    for (i, w) in buf.chunks_exact_mut(4).enumerate() {
        w.copy_from_slice(&pattern_word(seed, i as u64).to_le_bytes());
    }
    Ok(())
}

/// The first word of `data` (read back) that differs from the pattern:
/// `(byte offset, got, wanted)`. `None` = identical bit for bit.
pub fn first_mismatch(data: &[u8], seed: u32) -> Option<(u64, u32, u32)> {
    data.chunks(4).enumerate().find_map(|(i, w)| {
        let mut b = [0u8; 4];
        b[..w.len()].copy_from_slice(w);
        let (got, want) = (u32::from_le_bytes(b), pattern_word(seed, i as u64));
        (got != want || w.len() != 4).then_some((4 * i as u64, got, want))
    })
}

/// Word offsets of the VRAM buffer the driver reads back through the window
/// as a spot check (first, middle, last word of a `bytes`-long buffer).
pub fn spot_offsets(bytes: u64) -> [u64; 3] {
    let last = bytes.saturating_sub(4);
    [0, (bytes / 2) & !3, last]
}

/// Payloads of the three records of a round trip.
pub fn round_trip_payloads(first: u32) -> [u32; 3] {
    [first, first.wrapping_add(1), first.wrapping_add(2)]
}

/// One GPFIFO entry for the round trip: invalidate the shader caches (the
/// program, its cbufs and the QMDs were just written), record 0, launch the
/// QMD at `up_qmd_va` (RAM→VRAM), record 1, launch `down_qmd_va` (VRAM→RAM),
/// record 2. `records_va` = 3 × 16 bytes, 16-aligned; the compute object is
/// already bound on subchannel 1 (I10 setup).
pub fn encode_round_trip(
    p: &mut PushBuf<'_>,
    up_qmd_va: u64,
    down_qmd_va: u64,
    records_va: u64,
    first_payload: u32,
) -> GpuResult<()> {
    let [p0, p1, p2] = round_trip_payloads(first_payload);
    invalidate_caches(p)?;
    timestamp_release(p, records_va, p0)?;
    launch(p, up_qmd_va)?;
    timestamp_release(p, records_va + RECORD, p1)?;
    launch(p, down_qmd_va)?;
    timestamp_release(p, records_va + 2 * RECORD, p2)
}

/// Words [`encode_round_trip`] writes (for sizing the pushbuffer area).
pub const ROUND_TRIP_WORDS: usize = 2 + 3 * 6 + 2 * 4;

/// C-83 (I12+): the general form of [`encode_round_trip`]: invalidate the
/// shader caches, record 0, then for each QMD `k`: launch it, record `k + 1`
/// (payload `first_payload + k + 1`). Each record waits for the launch
/// before it (WFI), so the launches run one after the other and record
/// `k + 1` − record `k` is the time of launch `k`.
pub fn encode_sequence(
    p: &mut PushBuf<'_>,
    qmds: &[u64],
    records_va: u64,
    first_payload: u32,
) -> GpuResult<()> {
    if qmds.is_empty() {
        return Err(bad("secventa fara lansari"));
    }
    invalidate_caches(p)?;
    timestamp_release(p, records_va, first_payload)?;
    for (k, &q) in qmds.iter().enumerate() {
        launch(p, q)?;
        timestamp_release(
            p,
            records_va + (k as u64 + 1) * RECORD,
            first_payload.wrapping_add(k as u32 + 1),
        )?;
    }
    Ok(())
}

/// Words [`encode_sequence`] writes for `n` launches.
pub const fn sequence_words(n: usize) -> usize {
    2 + 6 + n * (4 + 6)
}

/// Reads the `n + 1` records of [`encode_sequence`]: every payload as asked,
/// time never going back; returns the `n` durations (ns). A wrong payload
/// names the first launch that did not finish.
pub fn analyze_sequence(records: &[u8], n: usize, first_payload: u32) -> GpuResult<Vec<u64>> {
    if records.len() < (n + 1) * RECORD as usize {
        return Err(bad("secventa: prea putine inregistrari"));
    }
    let mut t = Vec::with_capacity(n + 1);
    for k in 0..=n {
        let r = &records[k * RECORD as usize..(k + 1) * RECORD as usize];
        let got = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
        if got != first_payload.wrapping_add(k as u32) {
            return Err(if k == 0 {
                bad("secventa: inregistrarea 0 lipseste (canalul nu a pornit)")
            } else {
                bad("secventa: o lansare nu s-a terminat (inregistrarea ei lipseste)")
            });
        }
        t.push(u64::from_le_bytes(r[8..16].try_into().unwrap_or([0; 8])));
    }
    if t.windows(2).any(|w| w[1] < w[0]) {
        return Err(bad("secventa: ceasul placii merge inapoi"));
    }
    Ok(t.windows(2).map(|w| w[1] - w[0]).collect())
}

/// The two directions, measured by the GPU's clock.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopySpeed {
    pub bytes: u64,
    /// RAM→VRAM: record 1 − record 0, ns.
    pub up_ns: u64,
    /// VRAM→RAM: record 2 − record 1, ns.
    pub down_ns: u64,
}

impl CopySpeed {
    /// MB/s (10^6 bytes per second), rounded down; 0 if the time is 0.
    pub fn mb_per_s(bytes: u64, ns: u64) -> u64 {
        if ns == 0 {
            return 0;
        }
        (bytes as u128 * 1000 / ns as u128) as u64
    }

    pub fn up_mb_s(&self) -> u64 {
        Self::mb_per_s(self.bytes, self.up_ns)
    }

    pub fn down_mb_s(&self) -> u64 {
        Self::mb_per_s(self.bytes, self.down_ns)
    }

    /// One journal line.
    pub fn line(&self) -> String {
        format!(
            "i11: {} octeti; RAM->VRAM {} ns ({} MB/s); VRAM->RAM {} ns ({} MB/s)",
            self.bytes,
            self.up_ns,
            self.up_mb_s(),
            self.down_ns,
            self.down_mb_s()
        )
    }
}

/// Reads the 48 bytes of records: each payload must be the one asked for and
/// the times must not go backwards, else an error naming the record (a
/// record with the wrong payload = that release never happened).
pub fn analyze_round_trip(records: &[u8], bytes: u64, first_payload: u32) -> GpuResult<CopySpeed> {
    if records.len() < 3 * RECORD as usize {
        return Err(bad("i11: mai putin de 3 inregistrari"));
    }
    let mut t = [0u64; 3];
    for (k, want) in round_trip_payloads(first_payload).into_iter().enumerate() {
        let r = &records[k * RECORD as usize..(k + 1) * RECORD as usize];
        let got = u32::from_le_bytes([r[0], r[1], r[2], r[3]]);
        if got != want {
            return Err(match k {
                0 => bad("i11: inregistrarea 0 lipseste (canalul nu a pornit)"),
                1 => bad("i11: inregistrarea 1 lipseste (copierea RAM->VRAM nu s-a terminat)"),
                _ => bad("i11: inregistrarea 2 lipseste (copierea VRAM->RAM nu s-a terminat)"),
            });
        }
        t[k] = u64::from_le_bytes(r[8..16].try_into().unwrap_or([0; 8]));
    }
    if t[1] < t[0] || t[2] < t[1] {
        return Err(bad("i11: ceasul placii merge inapoi intre inregistrari"));
    }
    Ok(CopySpeed {
        bytes,
        up_ns: t[1] - t[0],
        down_ns: t[2] - t[1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbuf_and_qmd_for_a_copy() {
        let cb = copy_cbuf(0x10_0000_0000, 0x10_0100_0000, 8 << 20).unwrap();
        assert_eq!(
            u64::from_le_bytes(cb[0x28..0x30].try_into().unwrap()),
            0x10_0000_0000
        );
        assert_eq!(
            u64::from_le_bytes(cb[0x30..0x38].try_into().unwrap()),
            0x10_0100_0000
        );
        assert_eq!(
            u32::from_le_bytes(cb[0x38..0x3c].try_into().unwrap()),
            1 << 19
        );
        // NVK's base workgroup (0x0/0x4) stays 0.
        assert!(cb[..8].iter().all(|&b| b == 0));
        let q = copy_qmd(0x1000, 0x2000, 8 << 20).unwrap();
        assert_eq!(q.grid, [2048, 1, 1]);
        assert_eq!(q.block, [256, 1, 1]);
        assert_eq!(copy_qmd(0, 0, 16).unwrap().grid, [1, 1, 1]);
        assert_eq!(copy_qmd(0, 0, 4096 + 16).unwrap().grid, [2, 1, 1]);
        q.to_bytes().unwrap();
    }

    #[test]
    fn bad_copies_are_refused() {
        let a = 0x10_0000_0000u64;
        assert!(copy_cbuf(a, a + 0x1000, 0).is_err());
        assert!(copy_cbuf(a, a + 0x1000, 20).is_err());
        assert!(copy_cbuf(a + 8, a + 0x1000, 32).is_err());
        assert!(copy_cbuf(a, a + 0x1008, 32).is_err());
        assert!(copy_cbuf(a, a + 0x10, 0x20).is_err(), "suprapunere");
        assert!(copy_cbuf(a + 0x10, a, 0x20).is_err(), "suprapunere");
        assert!(copy_cbuf(a, a + 0x20, 0x20).is_ok(), "vecine, nu suprapuse");
        assert!(copy_cbuf(a, u64::MAX - 15, 16).is_err());
        assert!(copy_cbuf(a, a + (1 << 40), MAX_COPY_BYTES + 16).is_err());
        assert!(copy_qmd(0, 0, 0).is_err());
    }

    #[test]
    fn pattern_catches_shifts_zeros_and_repeats() {
        let mut b = alloc::vec![0u8; 64 << 10];
        fill_pattern(&mut b, 7).unwrap();
        assert_eq!(first_mismatch(&b, 7), None);
        assert!(first_mismatch(&b, 8).is_some(), "alta samanta");
        // A page copied twice (the second page = the first).
        let mut r = b.clone();
        r.copy_within(0..4096, 4096);
        assert_eq!(first_mismatch(&r, 7).map(|m| m.0), Some(4096));
        // One bit flipped at the very end.
        let mut f = b.clone();
        *f.last_mut().unwrap() ^= 0x80;
        assert_eq!(first_mismatch(&f, 7).map(|m| m.0), Some((64 << 10) - 4));
        // All words of a buffer differ from each other at least in the first 64 KiB.
        let mut seen = alloc::collections::BTreeSet::new();
        assert!((0..16384).all(|i| seen.insert(pattern_word(7, i))));
        assert!(fill_pattern(&mut [0u8; 6], 1).is_err());
        assert_eq!(spot_offsets(8 << 20), [0, 4 << 20, (8 << 20) - 4]);
    }

    fn rec(payload: u32, t: u64) -> [u8; 16] {
        let mut r = [0u8; 16];
        r[..4].copy_from_slice(&payload.to_le_bytes());
        r[8..].copy_from_slice(&t.to_le_bytes());
        r
    }

    #[test]
    fn records_give_the_speed_or_name_the_missing_one() {
        let mut r = [0u8; 48];
        r[..16].copy_from_slice(&rec(0x600, 1_000));
        r[16..32].copy_from_slice(&rec(0x601, 1_000 + 400_000));
        r[32..].copy_from_slice(&rec(0x602, 1_000 + 400_000 + 800_000));
        let s = analyze_round_trip(&r, 8 << 20, 0x600).unwrap();
        assert_eq!((s.up_ns, s.down_ns), (400_000, 800_000));
        assert_eq!((s.up_mb_s(), s.down_mb_s()), (20_971, 10_485));
        assert!(s.line().contains("RAM->VRAM 400000 ns (20971 MB/s)"));
        let mut miss = r;
        miss[32] = 0;
        let e = analyze_round_trip(&miss, 8 << 20, 0x600).unwrap_err();
        assert!(format!("{e:?}").contains("VRAM->RAM"));
        let mut back = r;
        back[16..32].copy_from_slice(&rec(0x601, 10));
        assert!(analyze_round_trip(&back, 8 << 20, 0x600).is_err());
        assert!(analyze_round_trip(&r[..40], 8 << 20, 0x600).is_err());
        assert_eq!(CopySpeed::mb_per_s(1, 0), 0);
    }

    #[test]
    fn sequences_encode_and_analyze() {
        let va = crate::mmu::VA_BASE;
        let mut w = [0u32; 64];
        let mut p = PushBuf::new(&mut w);
        encode_sequence(
            &mut p,
            &[va + 0x100, va + 0x200, va + 0x300],
            va + 0x400,
            0x700,
        )
        .unwrap();
        assert_eq!(p.len(), sequence_words(3));
        let mut w = [0u32; 64];
        let mut p = PushBuf::new(&mut w);
        assert!(encode_sequence(&mut p, &[], va + 0x400, 0x700).is_err());
        let mut r = [0u8; 64];
        for (k, t) in [10u64, 30, 60, 100].into_iter().enumerate() {
            r[16 * k..16 * k + 16].copy_from_slice(&rec(0x700 + k as u32, t));
        }
        assert_eq!(analyze_sequence(&r, 3, 0x700).unwrap(), [20, 30, 40]);
        let mut miss = r;
        miss[48] = 0;
        assert!(analyze_sequence(&miss, 3, 0x700).is_err());
        assert!(analyze_sequence(&r[..48], 3, 0x700).is_err());
        // The round trip is the sequence of two.
        assert_eq!(sequence_words(2), ROUND_TRIP_WORDS);
    }

    #[test]
    fn round_trip_pushbuffer_has_the_documented_size() {
        let mut w = [0u32; 64];
        let mut p = PushBuf::new(&mut w);
        let va = crate::mmu::VA_BASE;
        encode_round_trip(&mut p, va + 0x100, va + 0x200, va + 0x300, 0x600).unwrap();
        assert_eq!(p.len(), ROUND_TRIP_WORDS);
        let mut w = [0u32; 64];
        let mut p = PushBuf::new(&mut w);
        assert!(encode_round_trip(&mut p, va + 0x100, va + 0x200, va + 0x308, 0x600).is_err());
    }
}
