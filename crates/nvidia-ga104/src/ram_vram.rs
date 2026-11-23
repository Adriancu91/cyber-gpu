//! C-157, Adrian's point of 23 Nov (a) — **the speed of RAM ↔ VRAM through the board's copy engine**, both ways,
//! big blocks, several copies in flight. Plan and predictions written before the code: `docs/C-157_RAM_VRAM_PROBA.md`.
//!
//! Until now every big copy went through the compute program `copy_b128` ([`crate::transfer`], C-83 I11); the copy
//! engine object (`AMPERE_DMA_COPY_B`, class `0xC7B5`) has been allocated on the work channel since the golden image
//! (`gsp::golden`, step 6) but never sent a method. This module encodes, for that object, on subchannel
//! [`CE_SUBCH`]:
//! - [`encode_bind`]: `SET_OBJECT` with the class;
//! - [`encode_copy`]: one one-line copy (`OFFSET_IN/OUT`, `LINE_LENGTH_IN`, `LAUNCH_DMA` pitch → pitch, virtual
//!   addresses), optionally followed by the copy engine's own four-word semaphore with the GPU time
//!   (`SEMAPHORE_TYPE = RELEASE_FOUR_WORD_SEMAPHORE`, with `FLUSH_ENABLE` so the data is out before the record);
//!   a copy may be `PIPELINED` (it may start before the one before it has finished);
//! - [`encode_probe`]: the whole measurement in one submission ([`groups`]: up and down at `M/64`, `M/16`, `M/4`,
//!   `M`, then 4 × `M/4` pipelined each way, every group [`REPEATS`] times), [`analyze`] for the records,
//!   [`judge`] for the bytes, [`Probe`] for the `kt ramvram …` journal rows, [`parse_ramvram`] to read them back.
//!
//! Every method number and field comes from NVIDIA's `clc7b5.h` (open-gpu-doc, MIT), generated into
//! `nvidia_rm570::regs::clc7b5`; tinygrad's `ops_nv.py` (MIT) was read as the specification of a one-line copy.
//! The layout of the four-word record (payload at +0, time at +8) is DEDUS — the same as the channel's timestamp
//! record the board already writes ([`crate::launch_cost`]).
//!
//! Tested **only in SIMULARE** (T192, `tests/sim_c157_ram_vram.rs`, `gpu_sim::cmdproc` with its copy engine model);
//! the speeds there are the test's own numbers. Nothing here touches a board; it only encodes and checks bytes.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};
use nvidia_rm570::regs::{clc7b5 as ce, clc7c0 as c};

use crate::launch_cost::{RECORD, timestamp_release};
use crate::push::PushBuf;
use crate::transfer;

/// The subchannel of the copy object (tinygrad's and nouveau's choice; compute stays on 1).
pub const CE_SUBCH: u8 = 4;
/// Each group is measured this many times (the rows give the median and the extremes).
pub const REPEATS: usize = 3;
/// Copies in flight in a parallel group.
pub const PARALLEL: u64 = 4;
/// The size divisors of the single copies: `M/64`, `M/16`, `M/4`, `M`.
pub const SIZE_DIVS: [u64; 4] = [64, 16, 4, 1];
/// The default `M` on the board: 64 MiB (A 64 MiB + B 128 MiB of host RAM, V1 + V2 128 MiB of VRAM).
pub const DEFAULT_MAX_BYTES: u64 = 64 << 20;
/// `M` must be a multiple of this (so `M/64` is whole 4 KiB pages).
pub const MAX_BYTES_UNIT: u64 = 256 << 10;
/// Largest `M` (one `LINE_LENGTH_IN` is 32 bits; 1 GiB keeps every row in a u32 at > 16 MB/s).
pub const MAX_MAX_BYTES: u64 = 1 << 30;
/// The fill of B before the probe (as I11).
pub const FILL: u8 = 0xee;
/// Predictions written before any measurement (MB/s for the `M` = 64 MiB single copy, and the parallel group
/// relative to it, ‰): `docs/C-157_RAM_VRAM_PROBA.md`.
pub const PREDICTED_UP_MB_S: (u64, u64) = (11_000, 13_000);
pub const PREDICTED_DOWN_MB_S: (u64, u64) = (11_000, 13_500);
pub const PREDICTED_PARALLEL_PERMILLE: (u64, u64) = (950, 1050);

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The direction of a copy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    /// RAM → VRAM.
    Up,
    /// VRAM → RAM.
    Down,
}

impl Dir {
    pub fn code(self) -> u32 {
        match self {
            Dir::Up => 0,
            Dir::Down => 1,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Dir::Up => "sus (RAM->VRAM)",
            Dir::Down => "jos (VRAM->RAM)",
        }
    }
}

/// One measured group: a single copy of `bytes`, or [`PARALLEL`] pipelined copies totalling `bytes`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    pub dir: Dir,
    pub bytes: u64,
    pub parallel: bool,
}

/// The groups of the probe for a maximum size `m`, in the order they are sent.
pub fn groups(m: u64) -> Vec<Group> {
    let mut g = Vec::new();
    for dir in [Dir::Up, Dir::Down] {
        for d in SIZE_DIVS {
            g.push(Group {
                dir,
                bytes: m / d,
                parallel: false,
            });
        }
    }
    for dir in [Dir::Up, Dir::Down] {
        g.push(Group {
            dir,
            bytes: m,
            parallel: true,
        });
    }
    g
}

/// Groups per probe.
pub const GROUPS: usize = 2 * SIZE_DIVS.len() + 2;
/// Records per probe: the start record plus one per group repetition.
pub const RECORDS: usize = 1 + GROUPS * REPEATS;
/// Bytes of the record area.
pub const RECORD_BYTES: u64 = RECORDS as u64 * RECORD;

/// Checks a maximum size.
pub fn check_max(m: u64) -> GpuResult<()> {
    if m == 0 || !m.is_multiple_of(MAX_BYTES_UNIT) || m > MAX_MAX_BYTES {
        return Err(bad(
            "ram-vram: marimea maxima nu e multiplu de 256 KiB sau e peste 1 GiB",
        ));
    }
    Ok(())
}

/// The `LAUNCH_DMA` word: pitch → pitch, virtual addresses, non-pipelined or pipelined; with a record: the
/// four-word semaphore (with the GPU time) and the flush before it.
pub fn launch_dma(pipelined: bool, record: bool) -> u32 {
    let mut d = ce::NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE.set(
        0,
        if pipelined {
            ce::NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_PIPELINED
        } else {
            ce::NVC7B5_LAUNCH_DMA_DATA_TRANSFER_TYPE_NON_PIPELINED
        },
    );
    d = ce::NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT
        .set(d, ce::NVC7B5_LAUNCH_DMA_SRC_MEMORY_LAYOUT_PITCH);
    d = ce::NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT
        .set(d, ce::NVC7B5_LAUNCH_DMA_DST_MEMORY_LAYOUT_PITCH);
    if record {
        d = ce::NVC7B5_LAUNCH_DMA_FLUSH_ENABLE.set(d, ce::NVC7B5_LAUNCH_DMA_FLUSH_ENABLE_TRUE);
        d = ce::NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE.set(
            d,
            ce::NVC7B5_LAUNCH_DMA_SEMAPHORE_TYPE_RELEASE_FOUR_WORD_SEMAPHORE,
        );
    }
    d
}

/// `SET_OBJECT` of the copy object on [`CE_SUBCH`] (once per submission; harmless to repeat).
pub fn encode_bind(p: &mut PushBuf<'_>) -> GpuResult<()> {
    p.inc(
        CE_SUBCH,
        c::NVC7C0_SET_OBJECT,
        &[c::NVC7C0_SET_OBJECT_CLASS_ID.set(0, ce::AMPERE_DMA_COPY_B)],
    )
}

/// Words [`encode_bind`] writes.
pub const BIND_WORDS: usize = 2;

fn split(va: u64, f: nvidia_rm570::Field) -> GpuResult<[u32; 2]> {
    let hi = (va >> 32) as u32;
    if f.set(0, hi) >> f.lo != hi {
        return Err(bad("ram-vram: adresa prea mare pentru motorul de copiere"));
    }
    Ok([f.set(0, hi), va as u32])
}

/// One copy of `bytes` from `src` to `dst` (GPU VAs), `pipelined` or not, then optionally the four-word record
/// `(va, payload)`. Refused before anything is written: 0 bytes, more than 32 bits, a size or address not a
/// multiple of 16, overlapping ranges, a record not aligned to 16.
pub fn encode_copy(
    p: &mut PushBuf<'_>,
    src: u64,
    dst: u64,
    bytes: u64,
    pipelined: bool,
    record: Option<(u64, u32)>,
) -> GpuResult<()> {
    if bytes == 0 || bytes > u32::MAX as u64 || !bytes.is_multiple_of(16) {
        return Err(bad(
            "ram-vram: marime 0, peste 32 de biti sau nemultiplu de 16",
        ));
    }
    if !src.is_multiple_of(16) || !dst.is_multiple_of(16) {
        return Err(bad("ram-vram: adresa nealiniata la 16"));
    }
    let (Some(se), Some(de)) = (src.checked_add(bytes), dst.checked_add(bytes)) else {
        return Err(bad("ram-vram: adresa prea mare"));
    };
    if src < de && dst < se {
        return Err(bad("ram-vram: sursa si destinatia se suprapun"));
    }
    let [si, sl] = split(src, ce::NVC7B5_OFFSET_IN_UPPER_UPPER)?;
    let [di, dl] = split(dst, ce::NVC7B5_OFFSET_OUT_UPPER_UPPER)?;
    if let Some((va, _)) = record
        && !va.is_multiple_of(RECORD)
    {
        return Err(bad("ram-vram: inregistrare nealiniata la 16"));
    }
    let sem = record
        .map(|(va, payload)| split(va, ce::NVC7B5_SET_SEMAPHORE_A_UPPER).map(|s| (s, payload)))
        .transpose()?;
    p.inc(CE_SUBCH, ce::NVC7B5_OFFSET_IN_UPPER, &[si, sl, di, dl])?;
    p.inc(CE_SUBCH, ce::NVC7B5_LINE_LENGTH_IN, &[bytes as u32])?;
    if let Some(([hi, lo], payload)) = sem {
        p.inc(CE_SUBCH, ce::NVC7B5_SET_SEMAPHORE_A, &[hi, lo, payload])?;
    }
    p.inc(
        CE_SUBCH,
        ce::NVC7B5_LAUNCH_DMA,
        &[launch_dma(pipelined, record.is_some())],
    )
}

/// Words [`encode_copy`] writes.
pub const fn copy_words(record: bool) -> usize {
    5 + 2 + 2 + if record { 4 } else { 0 }
}

/// The GPU VAs of the probe's buffers (all set up by the OS): `a` = host RAM, `m` bytes, the pattern; `b` = host
/// RAM, `2m` bytes, filled with [`FILL`]; `v1`, `v2` = VRAM, `m` bytes each; `records` = [`RECORD_BYTES`], 16-aligned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buffers {
    pub a: u64,
    pub b: u64,
    pub v1: u64,
    pub v2: u64,
    pub records: u64,
}

/// Words [`encode_probe`] writes.
pub const fn probe_words() -> usize {
    let single = 2 * SIZE_DIVS.len() * REPEATS * copy_words(true);
    let par = 2 * REPEATS * (copy_words(true) + (PARALLEL as usize - 1) * copy_words(false));
    BIND_WORDS + 6 + single + par
}

/// The whole probe in one submission: bind, record 0 (the channel's, as before), then every [`groups`] entry
/// [`REPEATS`] times, record `k` after the `k`-th repetition (payload `first_payload + k`). Singles: up `a → v1`,
/// down `v1 → b[0..]`; parallel: up `a → v2` as 4 quarters, down `v2 → b[m..2m]`, the first quarter
/// non-pipelined (it waits for what came before), the other three pipelined, the record after the last.
pub fn encode_probe(
    p: &mut PushBuf<'_>,
    bufs: &Buffers,
    m: u64,
    first_payload: u32,
) -> GpuResult<()> {
    check_max(m)?;
    for va in [bufs.a, bufs.b, bufs.v1, bufs.v2] {
        if !va.is_multiple_of(0x1000) {
            return Err(bad("ram-vram: buffer nealiniat la 4 KiB"));
        }
    }
    encode_bind(p)?;
    timestamp_release(p, bufs.records, first_payload)?;
    let mut k = 0u32;
    for g in groups(m) {
        for _ in 0..REPEATS {
            k += 1;
            let rec = Some((
                bufs.records + k as u64 * RECORD,
                first_payload.wrapping_add(k),
            ));
            if !g.parallel {
                let (s, d) = match g.dir {
                    Dir::Up => (bufs.a, bufs.v1),
                    Dir::Down => (bufs.v1, bufs.b),
                };
                encode_copy(p, s, d, g.bytes, false, rec)?;
                continue;
            }
            let q = g.bytes / PARALLEL;
            let (s, d) = match g.dir {
                Dir::Up => (bufs.a, bufs.v2),
                Dir::Down => (bufs.v2, bufs.b + m),
            };
            for i in 0..PARALLEL {
                let last = i + 1 == PARALLEL;
                encode_copy(
                    p,
                    s + i * q,
                    d + i * q,
                    q,
                    i != 0,
                    if last { rec } else { None },
                )?;
            }
        }
    }
    Ok(())
}

/// The times of one group (ns, one per repetition).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GroupTime {
    pub group: Group,
    pub ns: [u64; REPEATS],
}

impl GroupTime {
    fn sorted(&self) -> [u64; REPEATS] {
        let mut v = self.ns;
        v.sort_unstable();
        v
    }
    pub fn median_ns(&self) -> u64 {
        self.sorted()[REPEATS / 2]
    }
    pub fn min_ns(&self) -> u64 {
        self.sorted()[0]
    }
    pub fn max_ns(&self) -> u64 {
        self.sorted()[REPEATS - 1]
    }
    /// MB/s (10^6 bytes per second) at the median.
    pub fn mb_s(&self) -> u64 {
        transfer::CopySpeed::mb_per_s(self.group.bytes, self.median_ns())
    }
    /// MB/s at the fastest repetition.
    pub fn best_mb_s(&self) -> u64 {
        transfer::CopySpeed::mb_per_s(self.group.bytes, self.min_ns())
    }
}

/// Reads the [`RECORDS`] records (the bytes of the record area): payloads as asked, time never going back (as
/// [`transfer::analyze_sequence`]); a missing record names how far the board got.
pub fn analyze(records: &[u8], m: u64, first_payload: u32) -> GpuResult<Vec<GroupTime>> {
    check_max(m)?;
    let d = transfer::analyze_sequence(records, RECORDS - 1, first_payload)?;
    Ok(groups(m)
        .into_iter()
        .enumerate()
        .map(|(i, group)| {
            let mut ns = [0; REPEATS];
            ns.copy_from_slice(&d[i * REPEATS..(i + 1) * REPEATS]);
            GroupTime { group, ns }
        })
        .collect())
}

/// What the bytes say.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bytes {
    /// The first word of B (both halves) that differs from the pattern: (byte offset, got, wanted).
    pub mismatch: Option<(u64, u32, u32)>,
    /// A still equals the pattern.
    pub source_ok: bool,
}

/// Judges B (read back, `2m` bytes) and A (`m` bytes) against the pattern of `seed`.
pub fn judge(b: &[u8], a: &[u8], m: u64, seed: u32) -> GpuResult<Bytes> {
    check_max(m)?;
    if b.len() as u64 != 2 * m || a.len() as u64 != m {
        return Err(bad("ram-vram: B trebuie sa aiba 2M octeti si A M octeti"));
    }
    let lo = transfer::first_mismatch(&b[..m as usize], seed);
    let hi = transfer::first_mismatch(&b[m as usize..], seed).map(|(o, g, w)| (o + m, g, w));
    Ok(Bytes {
        mismatch: lo.or(hi),
        source_ok: transfer::first_mismatch(a, seed).is_none(),
    })
}

/// The whole probe: the maximum size, the group times, the bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probe {
    pub max_bytes: u64,
    pub times: Vec<GroupTime>,
    pub bytes: Bytes,
}

impl Probe {
    /// 0 = every byte right, 1 = B wrong, 2 = A touched.
    pub fn verdict(&self) -> u32 {
        if self.bytes.mismatch.is_some() {
            1
        } else if !self.bytes.source_ok {
            2
        } else {
            0
        }
    }
    pub fn passes(&self) -> bool {
        self.verdict() == 0 && self.times.len() == GROUPS
    }
    fn find(&self, dir: Dir, parallel: bool) -> Option<&GroupTime> {
        self.times.iter().find(|t| {
            t.group.dir == dir && t.group.parallel == parallel && t.group.bytes == self.max_bytes
        })
    }
    /// MB/s of the single `M` copy in a direction (median).
    pub fn max_mb_s(&self, dir: Dir) -> Option<u64> {
        self.find(dir, false).map(GroupTime::mb_s)
    }
    /// The parallel group's speed relative to the single `M` copy, ‰.
    pub fn parallel_permille(&self, dir: Dir) -> Option<u64> {
        let (s, p) = (self.find(dir, false)?, self.find(dir, true)?);
        (s.mb_s() != 0).then(|| p.mb_s() * 1000 / s.mb_s())
    }

    /// The `kt ramvram …` rows (read back by [`parse_ramvram`]).
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        let c = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        let mut out = alloc::vec![
            ("kt ramvram marime maxima (KiB)", c(self.max_bytes >> 10)),
            ("kt ramvram grupuri", self.times.len() as u32),
            ("kt ramvram verdict", self.verdict()),
            (
                "kt ramvram primul cuvant gresit (octet)",
                self.bytes.mismatch.map_or(u32::MAX, |(o, _, _)| c(o))
            ),
        ];
        for t in &self.times {
            out.extend([
                ("kt ramvram grup: sens", t.group.dir.code()),
                ("kt ramvram grup: paralel", u32::from(t.group.parallel)),
                ("kt ramvram grup: KiB", c(t.group.bytes >> 10)),
                ("kt ramvram grup: timp median (ns)", c(t.median_ns())),
                ("kt ramvram grup: timp minim (ns)", c(t.min_ns())),
                ("kt ramvram grup: timp maxim (ns)", c(t.max_ns())),
                ("kt ramvram grup: MB/s", c(t.mb_s())),
            ]);
        }
        out
    }

    pub fn journal(&self, j: &mut Journal) {
        for (what, value) in self.rows() {
            j.push(Event::Value { what, value });
        }
    }

    /// Report lines (`label`: "SIMULARE" or "placa").
    pub fn lines(&self, label: &str) -> Vec<String> {
        let mut out = alloc::vec![format!(
            "ram-vram ({label}): M = {} KiB, {}; sus {} MB/s, jos {} MB/s (copie de M, mediana)",
            self.max_bytes >> 10,
            ["octeti corecti", "B GRESIT", "A ATINS"][self.verdict() as usize],
            self.max_mb_s(Dir::Up).unwrap_or(0),
            self.max_mb_s(Dir::Down).unwrap_or(0)
        )];
        for t in &self.times {
            out.push(format!(
                "ram-vram {} {}{} KiB ({label}): {} ns mediana ({}..{}), {} MB/s, cel mai bun {} MB/s",
                t.group.dir.name(),
                if t.group.parallel { "4 in paralel, " } else { "" },
                t.group.bytes >> 10,
                t.median_ns(),
                t.min_ns(),
                t.max_ns(),
                t.mb_s(),
                t.best_mb_s()
            ));
        }
        out
    }
}

/// What [`parse_ramvram`] reads back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedRamVram {
    pub max_kib: Option<u32>,
    pub groups: Option<u32>,
    pub verdict: Option<u32>,
    pub first_wrong: Option<u32>,
    /// Per group, in order: (direction, parallel, KiB, median ns, min ns, max ns, MB/s).
    pub per_group: Vec<[u32; 7]>,
}

/// Reads the rows of [`Probe::rows`] (`gpu: valoare kt ramvram … = 0x…`). A new `marime maxima` row starts over.
pub fn parse_ramvram(text: &str) -> ParsedRamVram {
    let mut p = ParsedRamVram::default();
    for line in text.lines() {
        let Some(rest) = line.split("valoare kt ramvram ").nth(1) else {
            continue;
        };
        let Some((name, val)) = rest.rsplit_once(" = 0x") else {
            continue;
        };
        let Ok(v) = u32::from_str_radix(val.trim(), 16) else {
            continue;
        };
        let last = |p: &mut ParsedRamVram, k: usize| {
            if let Some(s) = p.per_group.last_mut() {
                s[k] = v;
            }
        };
        match name {
            "marime maxima (KiB)" => {
                p = ParsedRamVram {
                    max_kib: Some(v),
                    ..ParsedRamVram::default()
                }
            }
            "grupuri" => p.groups = Some(v),
            "verdict" => p.verdict = Some(v),
            "primul cuvant gresit (octet)" => p.first_wrong = Some(v),
            "grup: sens" => p.per_group.push([v, 0, 0, 0, 0, 0, 0]),
            "grup: paralel" => last(&mut p, 1),
            "grup: KiB" => last(&mut p, 2),
            "grup: timp median (ns)" => last(&mut p, 3),
            "grup: timp minim (ns)" => last(&mut p, 4),
            "grup: timp maxim (ns)" => last(&mut p, 5),
            "grup: MB/s" => last(&mut p, 6),
            _ => {}
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn launch_dma_words_are_the_header_fields() {
        // NON_PIPELINED (2) | SRC pitch (bit 7) | DST pitch (bit 8) = tinygrad's 0x182.
        assert_eq!(launch_dma(false, false), 0x182);
        assert_eq!(launch_dma(true, false), 0x181);
        // + FLUSH_ENABLE (bit 2) + FOUR_WORD semaphore (2 << 3).
        assert_eq!(launch_dma(false, true), 0x182 | 0x4 | 0x10);
    }

    #[test]
    fn groups_and_sizes() {
        let g = groups(DEFAULT_MAX_BYTES);
        assert_eq!(g.len(), GROUPS);
        assert_eq!(g[0].bytes, 1 << 20);
        assert_eq!(g[3].bytes, 64 << 20);
        assert!(g[8].parallel && g[8].dir == Dir::Up && g[9].dir == Dir::Down);
        assert_eq!(RECORDS, 31);
        assert!(
            check_max(0).is_err() && check_max(100 << 10).is_err() && check_max(2 << 30).is_err()
        );
        assert!(check_max(MAX_BYTES_UNIT).is_ok());
    }

    #[test]
    fn bad_copies_are_refused_and_nothing_is_written() {
        let mut w = [0u32; 32];
        let mut p = PushBuf::new(&mut w);
        assert!(encode_copy(&mut p, 0x1000, 0x2000, 0, false, None).is_err());
        assert!(encode_copy(&mut p, 0x1000, 0x2000, 8, false, None).is_err());
        assert!(encode_copy(&mut p, 0x1008, 0x9000, 16, false, None).is_err());
        assert!(encode_copy(&mut p, 0x1000, 0x1800, 0x1000, false, None).is_err());
        assert!(encode_copy(&mut p, 0x1000, 0x9000, 1 << 32, false, None).is_err());
        assert!(encode_copy(&mut p, 1 << 50, 0x9000, 16, false, None).is_err());
        assert!(encode_copy(&mut p, 0x1000, 0x9000, 16, false, Some((0x8, 1))).is_err());
        assert!(p.is_empty());
        encode_copy(&mut p, 0x1000, 0x9000, 16, true, Some((0x10, 1))).unwrap();
        assert_eq!(p.len(), copy_words(true));
    }

    #[test]
    fn rows_parse_back() {
        let m = MAX_BYTES_UNIT;
        let times = groups(m)
            .into_iter()
            .enumerate()
            .map(|(i, group)| GroupTime {
                group,
                ns: [1000 + i as u64, 900, 1100],
            })
            .collect();
        let pr = Probe {
            max_bytes: m,
            times,
            bytes: Bytes {
                mismatch: Some((0x40, 1, 2)),
                source_ok: true,
            },
        };
        let text: String = pr
            .rows()
            .iter()
            .map(|(w, v)| format!("gpu: valoare {w} = {v:#x}\n"))
            .collect();
        let back = parse_ramvram(&text);
        assert_eq!(back.max_kib, Some(256));
        assert_eq!(back.groups, Some(GROUPS as u32));
        assert_eq!(back.verdict, Some(1));
        assert_eq!(back.first_wrong, Some(0x40));
        assert_eq!(back.per_group.len(), GROUPS);
        let t = &pr.times[3];
        assert_eq!(
            back.per_group[3],
            [0, 0, 256, t.median_ns() as u32, 900, 1100, t.mb_s() as u32]
        );
        assert!(!pr.passes());
    }
}
