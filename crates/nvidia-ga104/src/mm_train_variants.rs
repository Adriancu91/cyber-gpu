//! C-165 (`docs/C-165_CURSA_MM.md`): the race variants of [`crate::mm_train_tile`] for the night race on the board
//! (DECIZII 24 nov 17:35, H3). Every variant is the same program body (`tools/nak/mm_train_tv.glsl`) with another
//! tile geometry, compiled by Mesa NAK 25.0.7 for SM 8.6: **same job, same constant buffer** ([`mm::Args::cbuf`]) and
//! **the same bits** as `mm_train` / `mm_train_tile` (every element is the `ffma.ftz` chain p = 0..K from +0.0, then one
//! `fadd.ftz` of the bias; oracle `gpu_compute::reference::matmul_t_fma_ftz`). Only the launch changes: block size,
//! grid, shared memory, registers. No register and no command is new — a variant is another program address + QMD.
//! CPU models: `gpu_sim::cmdproc::mm_train_tv_on_cpu` (one per geometry). Tested **only in SIMULARE** (T217);
//! which one is fastest on which shape is for the race on the board to say.

use alloc::vec::Vec;

use crate::mm_train as mm;
use crate::mm_train_tile as mtt;
use crate::qmd::{MAX_SHARED_BYTES, SharedMemory};
use gpu_core::error::{GpuError, GpuResult};

use crate::train_step::{Launch, Prog};

/// One program of the race: the code, its checksum and how to launch it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    /// Short name (`mm_train_tile`, `mm_tv_128x64`, …) — the program's file name in `tools/nak/out/`.
    pub name: &'static str,
    /// The SM 8.6 code.
    pub code: &'static [u8],
    /// SHA-256 of [`Variant::code`], lower-case hex (checked by a test and by the OS before the race).
    pub sha256: &'static str,
    /// Registers per thread (NAK dump).
    pub registers: u32,
    /// Rows × columns of C per block.
    pub bm: u32,
    pub bn: u32,
    /// K slice per shared-memory tile.
    pub bk: u32,
    /// Results per thread (rows × columns).
    pub tm: u32,
    pub tn: u32,
}

impl Variant {
    /// Threads per block: `[(bm/tm)·(bn/tn), 1, 1]`.
    pub const fn block(&self) -> [u32; 3] {
        [(self.bm / self.tm) * (self.bn / self.tn), 1, 1]
    }
    /// Shared memory: the bk × bm tile of op(A) and the bk × bn tile of op(B); one barrier.
    pub const fn shared(&self) -> SharedMemory {
        SharedMemory {
            bytes: 4 * self.bk * (self.bm + self.bn),
            max_bytes: MAX_SHARED_BYTES,
            barriers: 1,
        }
    }
    /// The grid for one product: one block per bm × bn piece of C.
    pub fn grid(&self, args: &mm::Args) -> [u32; 3] {
        [args.n.div_ceil(self.bn), args.m.div_ceil(self.bm), 1]
    }
    /// Floating-point operations of one product (2·M·N·K), for TFLOP/s in the race report.
    pub fn flops(args: &mm::Args) -> u64 {
        2 * args.m as u64 * args.n as u64 * args.k as u64
    }
}

/// `tools/nak/out/mm_tv_128x64.sm86.sass`.
pub const MM_TV_128X64: &[u8] = include_bytes!("../../../tools/nak/out/mm_tv_128x64.sm86.sass");
/// `tools/nak/out/mm_tv_64x128.sm86.sass`.
pub const MM_TV_64X128: &[u8] = include_bytes!("../../../tools/nak/out/mm_tv_64x128.sm86.sass");
/// `tools/nak/out/mm_tv_128x128.sm86.sass`.
pub const MM_TV_128X128: &[u8] = include_bytes!("../../../tools/nak/out/mm_tv_128x128.sm86.sass");
/// `tools/nak/out/mm_tv_64x64k8.sm86.sass`.
pub const MM_TV_64X64K8: &[u8] = include_bytes!("../../../tools/nak/out/mm_tv_64x64k8.sm86.sass");

/// The race list: first the witness (`mm_train_tile`, the program the race compares against), then the variants.
/// Index 0 is always `mm_train_tile`; new variants are only appended.
pub const VARIANTS: [Variant; 5] = [
    Variant {
        name: "mm_train_tile",
        code: mtt::MM_TRAIN_TILE,
        sha256: mtt::MM_TRAIN_TILE_SHA256,
        registers: mtt::REGISTERS,
        bm: 64,
        bn: 64,
        bk: 16,
        tm: 4,
        tn: 4,
    },
    Variant {
        name: "mm_tv_128x64",
        code: MM_TV_128X64,
        sha256: "deb06f297f42fedca493d565010511987f7f413b3261d3b1ffb96857709c7a61",
        registers: 54,
        bm: 128,
        bn: 64,
        bk: 16,
        tm: 8,
        tn: 4,
    },
    Variant {
        name: "mm_tv_64x128",
        code: MM_TV_64X128,
        sha256: "e5bc59e3ad9af1109c85cd3f895e4f2b26749b460d5bb6740dd12f71df27f582",
        registers: 54,
        bm: 64,
        bn: 128,
        bk: 16,
        tm: 4,
        tn: 8,
    },
    Variant {
        name: "mm_tv_128x128",
        code: MM_TV_128X128,
        sha256: "5d3affd4fd8e9e63edd2719fc204cb7ad197f18d9c0c5c453a73cc0cb9e23ebc",
        registers: 90,
        bm: 128,
        bn: 128,
        bk: 8,
        tm: 8,
        tn: 8,
    },
    Variant {
        name: "mm_tv_64x64k8",
        code: MM_TV_64X64K8,
        sha256: "880884afec4b4beb8f7dedbd7d2145e9c1981dc1c007c3ef98c0b238fbb8f903",
        registers: 34,
        bm: 64,
        bn: 64,
        bk: 8,
        tm: 4,
        tn: 4,
    },
];

/// One shape of `mm_train` product in a step: the arguments without the addresses, and how many launches have it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Form {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// TA / TB / BIAS ([`mm::TA`], [`mm::TB`], [`mm::BIAS`]).
    pub flags: u32,
    /// Launches of the step with this shape.
    pub count: u32,
}

/// The distinct shapes of the `mm_train` / `mm_train_tile` / race-variant launches of a step (in first-seen order, with counts) —
/// the list the race runs on (`test-placa --cursa`). Other programs are ignored.
pub fn forms(launches: &[Launch]) -> Vec<Form> {
    let mut out: Vec<Form> = Vec::new();
    for l in launches {
        if l.prog != Prog::MmTrain && l.prog.race_index().is_none() {
            continue;
        }
        let u32_at =
            |o: usize| u32::from_le_bytes([l.cbuf[o], l.cbuf[o + 1], l.cbuf[o + 2], l.cbuf[o + 3]]);
        let (m, n, k, flags) = (u32_at(0x48), u32_at(0x4c), u32_at(0x50), u32_at(0x54));
        match out
            .iter_mut()
            .find(|f| (f.m, f.n, f.k, f.flags) == (m, n, k, flags))
        {
            Some(f) => f.count += 1,
            None => out.push(Form {
                m,
                n,
                k,
                flags,
                count: 1,
            }),
        }
    }
    out
}

/// How many shapes one [`MmChoice`] can hold (#513 has 15; the small golden step has fewer).
pub const MAX_CHOICE_FORMS: usize = 32;

/// How much faster than the witness a variant must be to be chosen by [`MmChoice::from_race`]: ≥ 3 % (median
/// against median), so that the noise between two runs does not flip the pick.
pub const MIN_GAIN_PERMILLE: u64 = 30;

/// One row of the race report (`test-placa --cursa`): one variant on one shape.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RaceRow {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub flags: u32,
    /// Index in [`VARIANTS`] (0 = the witness `mm_train_tile`).
    pub variant: usize,
    /// The median time of one launch, in nanoseconds (board clock).
    pub median_ns: u64,
    /// True when C was bit for bit the witness's C on the same board (the race's verdict).
    pub same_bits: bool,
}

/// C-165: the race's pick, one variant per `mm_train` shape — what [`crate::train_step::Plan::with_mm_choice`]
/// launches. A shape that is not in the table stays on the witness (`mm_train_tile`). A small fixed table (no
/// allocation, `Copy`), so that a plan carrying it stays as cheap to copy as before.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MmChoice {
    /// (M, N, K, flags, variant index), the first `len` used.
    forms: [(u32, u32, u32, u32, u8); MAX_CHOICE_FORMS],
    len: usize,
    /// The chosen variants (indices ≥ 1), once each, in [`VARIANTS`] order — [`MmChoice::programs`].
    progs: [Prog; VARIANTS.len() - 1],
    n_progs: usize,
}

impl Default for MmChoice {
    fn default() -> Self {
        Self::new()
    }
}

impl MmChoice {
    /// The empty choice: every shape on the witness (the plan is then exactly `with_mm_tile`).
    pub const fn new() -> Self {
        MmChoice {
            forms: [(0, 0, 0, 0, 0); MAX_CHOICE_FORMS],
            len: 0,
            progs: [Prog::MmTrainTile; VARIANTS.len() - 1],
            n_progs: 0,
        }
    }

    /// Puts shape (M, N, K, flags) on `VARIANTS[variant]` (replaces an earlier pick for the same shape). Refuses an
    /// index past [`VARIANTS`], unknown flag bits, a zero side, and a full table.
    pub fn set(&mut self, m: u32, n: u32, k: u32, flags: u32, variant: usize) -> GpuResult<()> {
        if variant >= VARIANTS.len() {
            return Err(GpuError::InvalidArgument("cursa: varianta nu exista"));
        }
        if m == 0 || n == 0 || k == 0 || flags & !(mm::TA | mm::TB | mm::BIAS) != 0 {
            return Err(GpuError::InvalidArgument("cursa: forma invalida"));
        }
        let v = variant as u8;
        match self.forms[..self.len]
            .iter_mut()
            .find(|f| (f.0, f.1, f.2, f.3) == (m, n, k, flags))
        {
            Some(f) => f.4 = v,
            None => {
                if self.len == MAX_CHOICE_FORMS {
                    return Err(GpuError::InvalidArgument("cursa: prea multe forme"));
                }
                self.forms[self.len] = (m, n, k, flags, v);
                self.len += 1;
            }
        }
        self.refresh_programs();
        Ok(())
    }

    /// The variant index for shape (M, N, K, flags): the pick, or 0 (the witness) when the shape is not in the table.
    pub fn of(&self, m: u32, n: u32, k: u32, flags: u32) -> usize {
        self.forms[..self.len]
            .iter()
            .find(|f| (f.0, f.1, f.2, f.3) == (m, n, k, flags))
            .map_or(0, |f| f.4 as usize)
    }

    /// The shapes in the table, with their variant index, in the order they were set.
    pub fn entries(&self) -> impl Iterator<Item = (Form, usize)> + '_ {
        self.forms[..self.len].iter().map(|&(m, n, k, flags, v)| {
            (
                Form {
                    m,
                    n,
                    k,
                    flags,
                    count: 0,
                },
                v as usize,
            )
        })
    }

    /// The programs the choice may launch besides `mm_train_tile` (each chosen variant once, in [`VARIANTS`] order).
    pub fn programs(&self) -> &[Prog] {
        &self.progs[..self.n_progs]
    }

    fn refresh_programs(&mut self) {
        self.n_progs = 0;
        for i in 1..VARIANTS.len() {
            if self.forms[..self.len].iter().any(|f| f.4 as usize == i)
                && let Some(p) = Prog::of_race_index(i)
            {
                self.progs[self.n_progs] = p;
                self.n_progs += 1;
            }
        }
    }

    /// The pick from a race report: for every shape, the fastest variant whose C was the witness's bit for bit, if
    /// its median is at least [`MIN_GAIN_PERMILLE`] below the witness's median on that shape; otherwise the witness
    /// (also when the witness has no row with `same_bits` for that shape — nothing to compare against). Rows of an
    /// unknown variant or with a zero median are ignored. A variant that changed the bits is never chosen.
    pub fn from_race(rows: &[RaceRow]) -> GpuResult<MmChoice> {
        let mut c = MmChoice::new();
        let mut seen: Vec<(u32, u32, u32, u32)> = Vec::new();
        for r in rows {
            let key = (r.m, r.n, r.k, r.flags);
            if seen.contains(&key) {
                continue;
            }
            seen.push(key);
            let same = |x: &&RaceRow| {
                (x.m, x.n, x.k, x.flags) == key
                    && x.same_bits
                    && x.median_ns > 0
                    && x.variant < VARIANTS.len()
            };
            let witness = rows
                .iter()
                .filter(same)
                .filter(|x| x.variant == 0)
                .map(|x| x.median_ns)
                .min();
            let Some(w) = witness else {
                continue;
            };
            let best = rows
                .iter()
                .filter(same)
                .filter(|x| x.variant > 0)
                .min_by_key(|x| (x.median_ns, x.variant));
            if let Some(b) = best
                && b.median_ns * 1_000 <= w * (1_000 - MIN_GAIN_PERMILLE)
            {
                c.set(r.m, r.n, r.k, r.flags, b.variant)?;
            }
        }
        Ok(c)
    }
}

/// What [`parse_race`] reads back from a report: the race rows, and what it left out.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedRace {
    /// Every timed row of the last race in the text, in report order.
    pub rows: Vec<RaceRow>,
    /// The launches per step of each shape (the report's `(xN in pas)`), first-seen order — for the saving of a pick.
    pub counts: Vec<Form>,
    /// Rows without a time (`- us`: skipped or unfinished) — never a [`RaceRow`].
    pub untimed: usize,
    /// Rows whose variant name is not in [`VARIANTS`] — never a [`RaceRow`].
    pub unknown: usize,
}

impl ParsedRace {
    /// [`MmChoice::from_race`] on the rows read.
    pub fn choice(&self) -> GpuResult<MmChoice> {
        MmChoice::from_race(&self.rows)
    }

    /// How much one step would save with `choice` (Σ over the shapes of (witness − chosen median) × launches), in ns —
    /// from the isolated race times, so an estimate (DEDUS), the same sum the OS prints.
    pub fn saved_ns(&self, choice: &MmChoice) -> u64 {
        let mut s = 0u64;
        for f in &self.counts {
            let v = choice.of(f.m, f.n, f.k, f.flags);
            let t = |want: usize| {
                self.rows
                    .iter()
                    .filter(|r| (r.m, r.n, r.k, r.flags) == (f.m, f.n, f.k, f.flags))
                    .filter(|r| r.variant == want && r.same_bits)
                    .map(|r| r.median_ns)
                    .min()
            };
            if v > 0
                && let (Some(w), Some(c)) = (t(0), t(v))
            {
                s += w.saturating_sub(c) * u64::from(f.count);
            }
        }
        s
    }
}

/// The header the OS writes before a race (`…: cursa mm (C-165, …): …`) — a new one starts the reading over.
const RACE_HEADER: &str = ": cursa mm (C-165";
/// The marker of one race row (`…: cursa: <variant> | …`).
const RACE_ROW: &str = ": cursa: ";

/// `"812.3"` → 812 300 ns (up to 3 decimals; `None` on anything else).
fn us_to_ns(s: &str) -> Option<u64> {
    let (whole, frac) = match s.split_once('.') {
        Some((w, f)) => (w, f),
        None => (s, ""),
    };
    if whole.is_empty() || frac.len() > 3 || !frac.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let w: u64 = whole.parse().ok()?;
    let mut f = 0u64;
    for (i, b) in frac.bytes().enumerate() {
        f += u64::from(b - b'0') * [100, 10, 1][i];
    }
    w.checked_mul(1_000)?.checked_add(f)
}

/// `"1024x2304x768 f2 (x12 in pas)"` → (M, N, K, flags, count).
fn parse_shape(s: &str) -> Option<(u32, u32, u32, u32, u32)> {
    let mut it = s.split_whitespace();
    let mnk = it.next()?;
    let flags = it.next()?.strip_prefix('f')?.parse().ok()?;
    let count = it.next()?.strip_prefix("(x")?.parse().ok()?;
    let mut d = mnk.split('x');
    let (m, n, k) = (
        d.next()?.parse().ok()?,
        d.next()?.parse().ok()?,
        d.next()?.parse().ok()?,
    );
    if d.next().is_some() {
        return None;
    }
    Some((m, n, k, flags, count))
}

/// C-165 step 3 (`docs/C-165_CURSA_MM.md` §„Pasul 3”): reads the race rows of an OS report (P6-190, `test-placa --cursa-mm`):
/// `…: cursa: <variant> | MxNxK f<flags> (x<count> in pas) | <µs> us median | <x.yy> TFLOP/s | biti <=|DIFERA|…>`.
/// Only the last race in the text is kept (a new `cursa mm (C-165, …)` header starts over). Rows with `- us`, unknown
/// variant names and broken rows never become a [`RaceRow`]; nothing panics. The time is the report's, rounded to 0.1 µs.
pub fn parse_race(text: &str) -> ParsedRace {
    let mut p = ParsedRace::default();
    for line in text.lines() {
        if line.contains(RACE_HEADER) {
            p = ParsedRace::default();
            continue;
        }
        let Some((_, rest)) = line.split_once(RACE_ROW) else {
            continue;
        };
        let parts: Vec<&str> = rest.split(" | ").map(str::trim).collect();
        let [name, shape, time, _tflops, bits] = parts[..] else {
            continue;
        };
        let Some((m, n, k, flags, count)) = parse_shape(shape) else {
            continue;
        };
        let Some(bits) = bits.strip_prefix("biti ") else {
            continue;
        };
        let Some(time) = time.strip_suffix(" us median") else {
            continue;
        };
        if !p
            .counts
            .iter()
            .any(|f| (f.m, f.n, f.k, f.flags) == (m, n, k, flags))
        {
            p.counts.push(Form {
                m,
                n,
                k,
                flags,
                count,
            });
        }
        if time == "-" {
            p.untimed += 1;
            continue;
        }
        let Some(median_ns) = us_to_ns(time) else {
            continue;
        };
        let Some(variant) = VARIANTS.iter().position(|v| v.name == name) else {
            p.unknown += 1;
            continue;
        };
        p.rows.push(RaceRow {
            m,
            n,
            k,
            flags,
            variant,
            median_ns,
            same_bits: bits == "=",
        });
    }
    p
}

#[cfg(test)]
mod race_report_tests {
    use super::*;

    #[test]
    fn microseconds_and_shapes() {
        assert_eq!(us_to_ns("812.3"), Some(812_300));
        assert_eq!(us_to_ns("0.0"), Some(0));
        assert_eq!(us_to_ns("12"), Some(12_000));
        assert_eq!(us_to_ns("1.234"), Some(1_234));
        for bad in ["", ".5", "1.2345", "1.2.3", "a.1", "1.a", "-"] {
            assert_eq!(us_to_ns(bad), None, "{bad}");
        }
        assert_eq!(
            parse_shape("1024x2304x768 f2 (x12 in pas)"),
            Some((1024, 2304, 768, 2, 12))
        );
        for bad in [
            "1024x2304 f2 (x12 in pas)",
            "1024x2304x768x1 f2 (x12 in pas)",
            "1024x2304x768 2 (x12 in pas)",
            "1024x2304x768 fA (x12 in pas)",
            "1024x2304x768 f2",
        ] {
            assert_eq!(parse_shape(bad), None, "{bad}");
        }
    }
}
