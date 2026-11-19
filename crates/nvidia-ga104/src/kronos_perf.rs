//! C-147 stage 0 — measurements for `os-309`, nothing changed in what the board computes.
//!
//! (a) **Time per kind of launch** ([`KindProfile`]): the timestamp records the board already writes after every
//! launch of a token (`transfer::analyze_sequence`) are split by what each launch does — the 16 kinds of a layer,
//! the embedding, the final LayerNorm, the logits and the copy of the logits down. For every kind: the sum over
//! one token (median over the tokens) and one launch (median over all of them). `kp …` rows + [`parse_journal`].
//! In `EachLaunch` the time between two records also holds the cache invalidation sent before the launch: the
//! row is "launch + its invalidation", the same quantity `kv` sums.
//!
//! (b) **Cost of an empty launch** in the journal: [`launch_cost_rows`] turns `launch_cost::LaunchCost`
//! (`i10-cost-lansare`) into `kl …` rows, so the number is on the stick and not only on the screen.
//!
//! (d) **VRAM → VRAM copy** ([`VramCopy`]): one big copy (64 MiB by default) with the existing `copy_b128`
//! program, repeated a few times in one submission, each followed by a timestamp record → the memory's real speed
//! (bytes copied per second, and the traffic = read + write). The source is the model's weights (already in VRAM,
//! the OS has the same bytes in RAM), so three words of the destination read back through the window are checked
//! against the host's copy. `kc …` rows.
//!
//! (c) The board's performance state (`PERF_GET_CURRENT_PSTATE`, …) is NOT here: see `docs/C-147_ETAPA0.md`.
//!
//! Tested **only in SIMULARE** (`gpu_sim::cmdproc`, its clock is the model's: 250 µs per launch, 100 ns per
//! method); the numbers there test the counting, they say nothing about the RTX 3070 Ti. Nothing in this module
//! touches a board: it only encodes and reads bytes.

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use crate::kronos_layer::Prog;
use crate::launch_cost::{LaunchCost, RECORD};
use crate::push::PushBuf;
use crate::transfer;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

fn sat(v: u64) -> u32 {
    v.min(u32::MAX as u64) as u32
}

fn median(v: &mut [u64]) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

// ------------------------------------------------------------------------------------------------ (a) kinds

/// What one launch of a token does (the order of `kronos_layer::token_launches` / `kronos_model::token_launches`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Embed,
    Ln1,
    Qkv,
    QkvBias,
    KCopy,
    VCopy,
    Attention,
    Out,
    OutBias,
    Residual1,
    Ln2,
    Fc,
    FcBias,
    Gelu,
    Proj,
    ProjBias,
    Residual2,
    FinalLn,
    Logits,
    /// The copy of the logits down into the OS's RAM (when the OS appends it).
    CopyDown,
}

/// The 16 kinds of one layer, in launch order.
pub const LAYER_KINDS: [Kind; 16] = [
    Kind::Ln1,
    Kind::Qkv,
    Kind::QkvBias,
    Kind::KCopy,
    Kind::VCopy,
    Kind::Attention,
    Kind::Out,
    Kind::OutBias,
    Kind::Residual1,
    Kind::Ln2,
    Kind::Fc,
    Kind::FcBias,
    Kind::Gelu,
    Kind::Proj,
    Kind::ProjBias,
    Kind::Residual2,
];

/// C-147 step 2: the kinds of the glued layer (`kronos_layer::token_launches_fused`, 7 launches). A glued launch
/// is counted under its matrix: `qkv` = QKV + bias + K/V into the caches, `out` = out + bias + residual, `fc` =
/// fc + bias + GELU, `proj` = proj + bias + residual. The other layer kinds then have no launches (no rows); the row
/// `kp lansari pe simbol` (88 instead of 196 for #513) says which list was measured.
pub const LAYER_KINDS_FUSED: [Kind; 7] = [
    Kind::Ln1,
    Kind::Qkv,
    Kind::Attention,
    Kind::Out,
    Kind::Ln2,
    Kind::Fc,
    Kind::Proj,
];

/// Every kind, in the order of the journal rows.
pub const ALL_KINDS: [Kind; 20] = [
    Kind::Embed,
    Kind::Ln1,
    Kind::Qkv,
    Kind::QkvBias,
    Kind::KCopy,
    Kind::VCopy,
    Kind::Attention,
    Kind::Out,
    Kind::OutBias,
    Kind::Residual1,
    Kind::Ln2,
    Kind::Fc,
    Kind::FcBias,
    Kind::Gelu,
    Kind::Proj,
    Kind::ProjBias,
    Kind::Residual2,
    Kind::FinalLn,
    Kind::Logits,
    Kind::CopyDown,
];

impl Kind {
    /// Index in [`ALL_KINDS`].
    pub fn index(self) -> usize {
        ALL_KINDS.iter().position(|&k| k == self).unwrap_or(0)
    }

    /// The program this kind launches.
    pub fn prog(self) -> Prog {
        match self {
            Kind::Embed => Prog::Embed,
            Kind::Ln1 | Kind::Ln2 | Kind::FinalLn => Prog::LayerNorm,
            Kind::Qkv | Kind::Out | Kind::Fc | Kind::Proj | Kind::Logits => Prog::Q8Matvec,
            Kind::QkvBias
            | Kind::OutBias
            | Kind::Residual1
            | Kind::FcBias
            | Kind::ProjBias
            | Kind::Residual2 => Prog::Vadd,
            Kind::KCopy | Kind::VCopy | Kind::CopyDown => Prog::Copy,
            Kind::Attention => Prog::Attention,
            Kind::Gelu => Prog::Gelu,
        }
    }

    /// Short name (ASCII, the journal's).
    pub fn name(self) -> &'static str {
        KIND_ROWS[self.index()].0
    }

    /// The two `kp` row names: sum over one token (median over tokens, µs), one launch (median, ns).
    pub fn rows(self) -> (&'static str, &'static str) {
        let r = KIND_ROWS[self.index()];
        (r.1, r.2)
    }
}

/// (name, sum row, one-launch row) for every kind, in [`ALL_KINDS`] order.
const KIND_ROWS: [(&str, &str, &str); 20] = [
    ("embed", "kp embed simbol (us)", "kp embed lansare (ns)"),
    ("ln1", "kp ln1 simbol (us)", "kp ln1 lansare (ns)"),
    ("qkv", "kp qkv simbol (us)", "kp qkv lansare (ns)"),
    ("qkv+b", "kp qkv+b simbol (us)", "kp qkv+b lansare (ns)"),
    ("k", "kp k simbol (us)", "kp k lansare (ns)"),
    ("v", "kp v simbol (us)", "kp v lansare (ns)"),
    (
        "atentie",
        "kp atentie simbol (us)",
        "kp atentie lansare (ns)",
    ),
    ("out", "kp out simbol (us)", "kp out lansare (ns)"),
    ("out+b", "kp out+b simbol (us)", "kp out+b lansare (ns)"),
    ("rez1", "kp rez1 simbol (us)", "kp rez1 lansare (ns)"),
    ("ln2", "kp ln2 simbol (us)", "kp ln2 lansare (ns)"),
    ("fc", "kp fc simbol (us)", "kp fc lansare (ns)"),
    ("fc+b", "kp fc+b simbol (us)", "kp fc+b lansare (ns)"),
    ("gelu", "kp gelu simbol (us)", "kp gelu lansare (ns)"),
    ("proj", "kp proj simbol (us)", "kp proj lansare (ns)"),
    ("proj+b", "kp proj+b simbol (us)", "kp proj+b lansare (ns)"),
    ("rez2", "kp rez2 simbol (us)", "kp rez2 lansare (ns)"),
    (
        "ln_final",
        "kp ln_final simbol (us)",
        "kp ln_final lansare (ns)",
    ),
    ("logit", "kp logit simbol (us)", "kp logit lansare (ns)"),
    (
        "copiere",
        "kp copiere simbol (us)",
        "kp copiere lansare (ns)",
    ),
];

/// The kinds of one token's launches, in order: the embedding (when the program `embed` is launched — not with
/// `EmbedBy::Driver`), `layers` × the 16 of a layer, the final LayerNorm, the logits, and the copy down when the
/// OS appends it.
pub fn token_kinds(layers: usize, embed: bool, copy_down: bool) -> Vec<Kind> {
    let mut v = Vec::with_capacity(16 * layers + 4);
    if embed {
        v.push(Kind::Embed);
    }
    for _ in 0..layers {
        v.extend_from_slice(&LAYER_KINDS);
    }
    v.push(Kind::FinalLn);
    v.push(Kind::Logits);
    if copy_down {
        v.push(Kind::CopyDown);
    }
    v
}

/// C-147 step 2: [`token_kinds`] for the glued list (`kronos_model::token_launches_fused`).
pub fn token_kinds_fused(layers: usize, embed: bool, copy_down: bool) -> Vec<Kind> {
    let mut v = Vec::with_capacity(7 * layers + 4);
    if embed {
        v.push(Kind::Embed);
    }
    for _ in 0..layers {
        v.extend_from_slice(&LAYER_KINDS_FUSED);
    }
    v.push(Kind::FinalLn);
    v.push(Kind::Logits);
    if copy_down {
        v.push(Kind::CopyDown);
    }
    v
}

/// The names of the summary `kp` rows.
pub mod rows {
    pub const TOKENS: &str = "kp simboluri";
    pub const LAUNCHES: &str = "kp lansari pe simbol";
    pub const TOTAL: &str = "kp toate simbol (us)";
    /// The kind that takes the most of a token (index in `ALL_KINDS`).
    pub const TOP: &str = "kp felul cel mai lung (index)";
}

/// The launch times of a generation, split by kind.
#[derive(Clone, Debug, Default)]
pub struct KindProfile {
    /// Per token: the sum of each kind (ns), indexed as [`ALL_KINDS`].
    tokens: Vec<[u64; 20]>,
    /// Every launch time, per kind (ns).
    singles: Vec<Vec<u64>>,
    launches: usize,
}

/// What [`KindProfile::summary`] gives for one kind.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KindTime {
    pub kind: Option<Kind>,
    /// Launches of this kind in one token.
    pub per_token: u32,
    /// Sum over one token, median over the tokens (ns).
    pub token_median_ns: u64,
    /// One launch, median over all (ns).
    pub launch_median_ns: u64,
}

impl KindProfile {
    pub fn new() -> Self {
        KindProfile {
            tokens: Vec::new(),
            singles: (0..ALL_KINDS.len()).map(|_| Vec::new()).collect(),
            launches: 0,
        }
    }

    /// One token: `kinds` (from [`token_kinds`]) and the launch times read from its records, in the same order
    /// (all its submissions put together). Different lengths, or a token with another launch count than the
    /// first, are refused: nothing is guessed.
    pub fn add_token(&mut self, kinds: &[Kind], ns: &[u64]) -> GpuResult<()> {
        if kinds.is_empty() || kinds.len() != ns.len() {
            return Err(bad("kp: felurile si timpii au lungimi diferite"));
        }
        if self.singles.len() != ALL_KINDS.len() {
            *self = KindProfile::new();
        }
        if !self.tokens.is_empty() && self.launches != kinds.len() {
            return Err(bad("kp: un simbol cu alt numar de lansari decat primul"));
        }
        let mut sums = [0u64; 20];
        for (&k, &t) in kinds.iter().zip(ns) {
            sums[k.index()] += t;
            self.singles[k.index()].push(t);
        }
        self.launches = kinds.len();
        self.tokens.push(sums);
        Ok(())
    }

    pub fn tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Every kind that appeared, in [`ALL_KINDS`] order, plus the total (kind `None`) last.
    pub fn summary(&self) -> Vec<KindTime> {
        let mut out = Vec::new();
        let n = self.tokens.len().max(1);
        for (i, &k) in ALL_KINDS.iter().enumerate() {
            let Some(singles) = self.singles.get(i) else {
                continue;
            };
            if singles.is_empty() {
                continue;
            }
            let mut per: Vec<u64> = self.tokens.iter().map(|t| t[i]).collect();
            let mut one = singles.clone();
            out.push(KindTime {
                kind: Some(k),
                per_token: (singles.len() / n) as u32,
                token_median_ns: median(&mut per),
                launch_median_ns: median(&mut one),
            });
        }
        let mut totals: Vec<u64> = self.tokens.iter().map(|t| t.iter().sum()).collect();
        out.push(KindTime {
            kind: None,
            per_token: self.launches as u32,
            token_median_ns: median(&mut totals),
            launch_median_ns: 0,
        });
        out
    }

    /// The rows [`Self::journal`] writes: tokens, launches per token, then for every kind that appeared its two
    /// rows, then the total and the kind with the largest median per token.
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        let s = self.summary();
        let mut v = alloc::vec![
            (rows::TOKENS, sat(self.tokens.len() as u64)),
            (rows::LAUNCHES, sat(self.launches as u64)),
        ];
        let mut top: Option<(u64, Kind)> = None;
        for t in &s {
            match t.kind {
                Some(k) => {
                    let (a, b) = k.rows();
                    v.push((a, sat(t.token_median_ns / 1000)));
                    v.push((b, sat(t.launch_median_ns)));
                    if top.is_none_or(|(ns, _)| t.token_median_ns > ns) {
                        top = Some((t.token_median_ns, k));
                    }
                }
                None => v.push((rows::TOTAL, sat(t.token_median_ns / 1000))),
            }
        }
        v.push((rows::TOP, top.map_or(u32::MAX, |(_, k)| k.index() as u32)));
        v
    }

    /// The `kp` rows into the journal (once, at the end of the generation: 4 + 2 per kind, 44 for #513).
    pub fn journal(&self, j: &mut Journal) {
        for (what, value) in self.rows() {
            j.push(Event::Value { what, value });
        }
    }
}

// ------------------------------------------------------------------------------------------------ (b) empty launch

/// The names of the `kl` rows (the empty launches of `i10-cost-lansare`).
pub mod launch_rows {
    pub const LAUNCHES: &str = "kl lansari goale";
    pub const MIN: &str = "kl minim (ns)";
    pub const MEDIAN: &str = "kl median (ns)";
    pub const MAX: &str = "kl maxim (ns)";
    pub const TOTAL: &str = "kl total (us)";
}

/// `launch_cost::LaunchCost` as journal rows.
pub fn launch_cost_rows(c: &LaunchCost) -> Vec<(&'static str, u32)> {
    use launch_rows as n;
    alloc::vec![
        (n::LAUNCHES, c.launches),
        (n::MIN, sat(c.min_ns)),
        (n::MEDIAN, sat(c.median_ns)),
        (n::MAX, sat(c.max_ns)),
        (n::TOTAL, sat(c.total_ns / 1000)),
    ]
}

/// The `kl` rows into the journal.
pub fn journal_launch_cost(c: &LaunchCost, j: &mut Journal) {
    for (what, value) in launch_cost_rows(c) {
        j.push(Event::Value { what, value });
    }
}

// ------------------------------------------------------------------------------------------------ (d) VRAM copy

/// The suggested size of the VRAM → VRAM copy: 64 MiB (C-147).
pub const VRAM_COPY_BYTES: u64 = 64 << 20;
/// Copies of the same buffer in one submission (the first one may pay for cold caches and TLBs).
pub const VRAM_COPY_REPEATS: usize = 3;

/// One VRAM → VRAM copy measurement: `bytes` from `src` to `dst` (GPU VAs in VRAM, 16-aligned, not overlapping),
/// `repeats` times in one submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramCopy {
    pub src: u64,
    pub dst: u64,
    pub bytes: u64,
    pub repeats: usize,
}

/// What the records of a [`VramCopy`] say.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VramCopySpeed {
    pub bytes: u64,
    pub each_ns: Vec<u64>,
    pub min_ns: u64,
    pub median_ns: u64,
}

impl VramCopySpeed {
    /// Bytes copied per second, in MB/s (10^6), from the fastest copy.
    pub fn copy_mb_s(&self) -> u64 {
        transfer::CopySpeed::mb_per_s(self.bytes, self.min_ns)
    }

    /// The memory's traffic (every byte read once and written once), MB/s, from the fastest copy.
    pub fn traffic_mb_s(&self) -> u64 {
        transfer::CopySpeed::mb_per_s(2 * self.bytes, self.min_ns)
    }

    /// The traffic as a share of the RTX 3070 Ti's memory bandwidth on paper (608 GB/s, GDDR6X 19 Gbps × 256 bit;
    /// NVIDIA's product page), in tenths of a percent.
    pub fn share_permille(&self) -> u64 {
        self.traffic_mb_s() * 1000 / PAPER_MB_S
    }
}

/// The RTX 3070 Ti's memory bandwidth on paper, MB/s.
pub const PAPER_MB_S: u64 = 608_000;

/// The names of the `kc` rows.
pub mod copy_rows {
    pub const KIB: &str = "kc copiere VRAM KiB";
    pub const REPEATS: &str = "kc copieri";
    pub const EACH: &str = "kc copiere (us)";
    pub const MIN: &str = "kc cea mai rapida (us)";
    pub const MEDIAN: &str = "kc median (us)";
    pub const COPY_MB: &str = "kc copiere MB/s";
    pub const TRAFFIC_MB: &str = "kc trafic memorie MB/s (citire + scriere)";
    pub const SHARE: &str = "kc din 608 GB/s (promile)";
    pub const SPOTS: &str = "kc cuvinte verificate gresite (din 3)";
}

impl VramCopy {
    /// The measurement as asked; refused before anything is written: a size of 0 or not a multiple of 16, too
    /// big for one launch, addresses not 16-aligned, overlapping, no repeats or more than 8.
    pub fn new(src: u64, dst: u64, bytes: u64, repeats: usize) -> GpuResult<Self> {
        transfer::copy_cbuf(src, dst, bytes)?;
        if repeats == 0 || repeats > 8 {
            return Err(bad("kc: intre 1 si 8 copieri"));
        }
        Ok(VramCopy {
            src,
            dst,
            bytes,
            repeats,
        })
    }

    /// The copy's size for a source of `available` bytes: [`VRAM_COPY_BYTES`] or less, a multiple of 4 KiB.
    pub fn size_for(available: u64) -> u64 {
        VRAM_COPY_BYTES.min(available) & !0xfff
    }

    /// The cbuf of the copy (one launch; every repeat reuses the same cbuf and QMD).
    pub fn cbuf(&self) -> GpuResult<[u8; transfer::CBUF_SIZE as usize]> {
        transfer::copy_cbuf(self.src, self.dst, self.bytes)
    }

    /// The QMD of the copy, with the program at `program_va` and the cbuf at `cbuf_va`.
    pub fn qmd(&self, program_va: u64, cbuf_va: u64) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
        transfer::copy_qmd(program_va, cbuf_va, self.bytes)?.to_bytes()
    }

    /// Pushbuffer words of [`Self::encode`].
    pub fn words(&self) -> usize {
        transfer::sequence_words(self.repeats)
    }

    /// Bytes of records the submission writes (`repeats + 1` records of 16 bytes).
    pub fn record_bytes(&self) -> u64 {
        (self.repeats as u64 + 1) * RECORD
    }

    /// One submission: the caches invalidated, record 0, then `repeats` × (the copy, a record) —
    /// `transfer::encode_sequence` with the same QMD repeated.
    pub fn encode(
        &self,
        p: &mut PushBuf<'_>,
        qmd_va: u64,
        records_va: u64,
        first: u32,
    ) -> GpuResult<()> {
        let qmds = alloc::vec![qmd_va; self.repeats];
        transfer::encode_sequence(p, &qmds, records_va, first)
    }

    /// The records read back: every copy finished and in order, else an error.
    pub fn analyze(&self, records: &[u8], first: u32) -> GpuResult<VramCopySpeed> {
        let each = transfer::analyze_sequence(records, self.repeats, first)?;
        let mut s = each.clone();
        Ok(VramCopySpeed {
            bytes: self.bytes,
            min_ns: s.iter().copied().min().unwrap_or(0),
            median_ns: median(&mut s),
            each_ns: each,
        })
    }

    /// Byte offsets into the destination the driver reads back through the window (first, middle, last word).
    pub fn spots(&self) -> [u64; 3] {
        transfer::spot_offsets(self.bytes)
    }

    /// How many of the three words read back at [`Self::spots`] differ from the source's bytes (`src_bytes` = the
    /// host's copy of what is at `src`, at least `bytes` long; shorter = all three counted wrong).
    pub fn wrong_spots(&self, src_bytes: &[u8], got: &[u32; 3]) -> u32 {
        self.spots()
            .iter()
            .zip(got)
            .filter(|&(&off, &g)| {
                let o = off as usize;
                src_bytes
                    .get(o..o + 4)
                    .is_none_or(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]) != g)
            })
            .count() as u32
    }

    /// The `kc` rows: size, copies, each copy, fastest, median, MB/s copied, traffic, share of 608 GB/s, wrong
    /// words (`u32::MAX` = not checked).
    pub fn rows(&self, s: &VramCopySpeed, wrong: Option<u32>) -> Vec<(&'static str, u32)> {
        use copy_rows as n;
        let mut v = alloc::vec![
            (n::KIB, sat(self.bytes / 1024)),
            (n::REPEATS, self.repeats as u32)
        ];
        v.extend(s.each_ns.iter().map(|&t| (n::EACH, sat(t / 1000))));
        v.extend([
            (n::MIN, sat(s.min_ns / 1000)),
            (n::MEDIAN, sat(s.median_ns / 1000)),
            (n::COPY_MB, sat(s.copy_mb_s())),
            (n::TRAFFIC_MB, sat(s.traffic_mb_s())),
            (n::SHARE, sat(s.share_permille())),
            (n::SPOTS, wrong.unwrap_or(u32::MAX)),
        ]);
        v
    }

    pub fn journal(&self, s: &VramCopySpeed, wrong: Option<u32>, j: &mut Journal) {
        for (what, value) in self.rows(s, wrong) {
            j.push(Event::Value { what, value });
        }
    }
}

// ------------------------------------------------------------------------------------------------ reader

/// What a stick journal's `kp` / `kl` / `kc` rows say (the last value of each row; `ffffffff` = `None`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedPerf {
    pub tokens: Option<u32>,
    pub launches: Option<u32>,
    /// Per kind that appeared: (kind, sum over a token µs, one launch ns).
    pub kinds: Vec<(Kind, Option<u32>, Option<u32>)>,
    pub total_us: Option<u32>,
    pub top: Option<Kind>,
    /// `kl`: launches, min, median, max (ns), total (µs).
    pub empty_launch: Option<[Option<u32>; 5]>,
    /// `kc`: KiB, copies, each copy µs, fastest µs, median µs, MB/s, traffic MB/s, ‰ of 608 GB/s, wrong words.
    pub copy_kib: Option<u32>,
    pub copy_each_us: Vec<u32>,
    pub copy_min_us: Option<u32>,
    pub copy_mb_s: Option<u32>,
    pub traffic_mb_s: Option<u32>,
    pub share_permille: Option<u32>,
    pub copy_wrong: Option<u32>,
}

/// One `… valoare <name> = 0x<hex>` row of the stick journal.
pub fn value_row(line: &str) -> Option<(&str, u32)> {
    let rest = line.split("valoare ").nth(1)?;
    let (name, v) = rest.rsplit_once(" = ")?;
    let v = u32::from_str_radix(v.trim().strip_prefix("0x")?, 16).ok()?;
    Some((name, v))
}

/// Reads the `kp`, `kl` and `kc` rows of a stick journal. A new `kp simboluri` row starts the kinds again, a new
/// `kc copiere VRAM KiB` row the copy (only the last run of each is kept).
pub fn parse_journal(text: &str) -> ParsedPerf {
    let mut out = ParsedPerf::default();
    for (name, v) in text.lines().filter_map(value_row) {
        let some = (v != u32::MAX).then_some(v);
        if let Some(k) = ALL_KINDS.iter().find(|k| k.rows().0 == name) {
            match out.kinds.iter_mut().find(|e| e.0 == *k) {
                Some(e) => e.1 = some,
                None => out.kinds.push((*k, some, None)),
            }
            continue;
        }
        if let Some(k) = ALL_KINDS.iter().find(|k| k.rows().1 == name) {
            match out.kinds.iter_mut().find(|e| e.0 == *k) {
                Some(e) => e.2 = some,
                None => out.kinds.push((*k, None, some)),
            }
            continue;
        }
        let kl = |out: &mut ParsedPerf, i: usize| {
            out.empty_launch.get_or_insert([None; 5])[i] = some;
        };
        match name {
            rows::TOKENS => {
                out.tokens = some;
                out.kinds.clear();
                out.total_us = None;
                out.top = None;
            }
            rows::LAUNCHES => out.launches = some,
            rows::TOTAL => out.total_us = some,
            rows::TOP => out.top = some.and_then(|i| ALL_KINDS.get(i as usize).copied()),
            launch_rows::LAUNCHES => {
                out.empty_launch = None;
                kl(&mut out, 0)
            }
            launch_rows::MIN => kl(&mut out, 1),
            launch_rows::MEDIAN => kl(&mut out, 2),
            launch_rows::MAX => kl(&mut out, 3),
            launch_rows::TOTAL => kl(&mut out, 4),
            copy_rows::KIB => {
                out.copy_kib = some;
                out.copy_each_us.clear();
            }
            copy_rows::EACH => out.copy_each_us.push(v),
            copy_rows::MIN => out.copy_min_us = some,
            copy_rows::COPY_MB => out.copy_mb_s = some,
            copy_rows::TRAFFIC_MB => out.traffic_mb_s = some,
            copy_rows::SHARE => out.share_permille = some,
            copy_rows::SPOTS => out.copy_wrong = some,
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kronos_model as km;

    #[test]
    fn kinds_count_like_the_launch_list() {
        // The one-to-one check against `kronos_model::token_launches` is in `tests/sim_c147.rs`.
        assert_eq!(token_kinds(2, true, false).len(), km::launches_per_token(2));
        assert_eq!(token_kinds(12, true, true).len(), 196);
        assert_eq!(token_kinds_fused(12, true, true).len(), 88);
        assert_eq!(token_kinds(12, false, true).len(), 195);
        for k in ALL_KINDS {
            assert_eq!(ALL_KINDS[k.index()], k);
            assert!(k.rows().0.starts_with("kp ") && k.rows().1.starts_with("kp "));
        }
    }

    #[test]
    fn profile_sums_and_medians_per_kind() {
        let kinds = token_kinds(2, true, true);
        let mut p = KindProfile::new();
        for tok in 0..3u64 {
            let ns: Vec<u64> = kinds
                .iter()
                .enumerate()
                .map(|(i, k)| 1000 * (k.index() as u64 + 1) + i as u64 + tok)
                .collect();
            p.add_token(&kinds, &ns).unwrap();
        }
        let s = p.summary();
        let qkv = s.iter().find(|t| t.kind == Some(Kind::Qkv)).unwrap();
        assert_eq!(qkv.per_token, 2);
        // Token 1 (the median): layer 0 qkv at i = 2, layer 1 at i = 18: (3000+2+1) + (3000+18+1).
        assert_eq!(qkv.token_median_ns, 6022);
        let total = s.last().unwrap();
        assert_eq!((total.kind, total.per_token), (None, 36));
        let rows = p.rows();
        assert_eq!(rows.len(), 4 + 2 * 20);
        assert_eq!(rows[0], (rows::TOKENS, 3));
        // The copy down is the largest kind per token here (index 19 × 1000 ns, once).
        assert_eq!(rows.last().unwrap().0, rows::TOP);
        // Different lengths and a token with another launch count are refused.
        assert!(p.add_token(&kinds, &[1, 2]).is_err());
        let fewer = token_kinds(2, false, true);
        assert!(p.add_token(&fewer, &vec![1; fewer.len()]).is_err());
    }

    #[test]
    fn rows_round_trip_through_the_journal_text() {
        let kinds = token_kinds(1, true, false);
        let mut p = KindProfile::new();
        p.add_token(&kinds, &vec![2500; kinds.len()]).unwrap();
        let cost = LaunchCost {
            launches: 4,
            each_ns: alloc::vec![5000, 6000, 7000, 9000],
            min_ns: 5000,
            median_ns: 7000,
            max_ns: 9000,
            total_ns: 27_000,
        };
        let c = VramCopy::new(0x1000_0000, 0x2000_0000, VRAM_COPY_BYTES, 3).unwrap();
        let sp = VramCopySpeed {
            bytes: c.bytes,
            each_ns: alloc::vec![400_000, 300_000, 310_000],
            min_ns: 300_000,
            median_ns: 310_000,
        };
        let mut text = alloc::string::String::new();
        let mut put = |rows: Vec<(&'static str, u32)>| {
            for (n, v) in rows {
                text.push_str(&alloc::format!("[1.0] gpu: valoare {n} = {v:#010x}\n"));
            }
        };
        put(p.rows());
        put(launch_cost_rows(&cost));
        put(c.rows(&sp, Some(0)));
        let r = parse_journal(&text);
        assert_eq!(r.tokens, Some(1));
        assert_eq!(r.launches, Some(19));
        assert_eq!(r.kinds.len(), 19);
        assert_eq!(r.kinds[0], (Kind::Embed, Some(2), Some(2500)));
        assert_eq!(r.total_us, Some(47));
        assert_eq!(
            r.empty_launch,
            Some([Some(4), Some(5000), Some(7000), Some(9000), Some(27)])
        );
        assert_eq!(r.copy_kib, Some(65_536));
        assert_eq!(r.copy_each_us, [400, 300, 310]);
        // 64 MiB in 300 µs: 223 696 MB/s copied, twice that of traffic, 735 ‰ of 608 GB/s.
        assert_eq!(r.copy_mb_s, Some(223_696));
        assert_eq!(r.traffic_mb_s, Some(447_392));
        assert_eq!(r.share_permille, Some(735));
        assert_eq!(r.copy_wrong, Some(0));
    }

    #[test]
    fn vram_copy_refuses_bad_requests_and_checks_spots() {
        assert!(VramCopy::new(0x1000, 0x2000, 0, 3).is_err());
        assert!(VramCopy::new(0x1000, 0x1800, 0x1000, 3).is_err()); // overlap
        assert!(VramCopy::new(0x1008, 0x9000, 0x1000, 3).is_err()); // alignment
        assert!(VramCopy::new(0x1000, 0x9000, 0x1000, 0).is_err());
        assert!(VramCopy::new(0x1000, 0x9000, 0x1000, 9).is_err());
        assert_eq!(VramCopy::size_for(110 << 20), 64 << 20);
        assert_eq!(VramCopy::size_for(0x12345), 0x12000);
        let c = VramCopy::new(0x1000, 0x10_0000, 0x1000, 2).unwrap();
        let src: Vec<u8> = (0..0x1000u32).map(|i| i as u8).collect();
        let w = |o: usize| u32::from_le_bytes([src[o], src[o + 1], src[o + 2], src[o + 3]]);
        let [a, b, d] = c.spots();
        let good = [w(a as usize), w(b as usize), w(d as usize)];
        assert_eq!(c.wrong_spots(&src, &good), 0);
        assert_eq!(c.wrong_spots(&src, &[good[0], 0xdead, good[2]]), 1);
        assert_eq!(c.wrong_spots(&src[..16], &good), 2);
        assert_eq!(c.words(), transfer::sequence_words(2));
        assert_eq!(c.record_bytes(), 48);
    }
}
