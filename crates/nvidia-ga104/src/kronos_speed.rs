//! C-142 — what a real Kronos model (#513: 12 layers, E 768, 12 heads, vocabulary 16 384, context 512) needs on
//! the work channel, and simple speed counters for `kronos --gpu`.
//!
//! **Limits** ([`ModelLimits`]): computed from the shapes alone, before anything is uploaded — launches per token
//! (with the copy of the logits down), the pushbuffer bytes of one submission in each way of sending
//! (`kronos_diag::SeqMode`), the pushbuffer area that holds two of the largest (one in flight while the next is
//! written), the payload stride a doorbell's records need, the VRAM budget (`kronos_gen::VramBudget`) and the upload
//! pieces through the DMA arena of I11 (at most [`crate::transfer::I11_DEFAULT_BYTES`] each). [`ModelLimits::check`]
//! names every limit the caller's setup does not meet, so the OS refuses **before** the first upload.
//!
//! **Speed** ([`Speed`]): the board's clock (the timestamps of the records: what the GPU did) kept apart from the
//! host's clock (what a user waits for, with the driver, the journal and the screen in between). The weights'
//! upload is counted apart from the tokens. [`Speed::journal`] writes `kv …` rows into the stick journal (values,
//! 32-bit, microseconds, saturated) and [`parse_journal`] reads them back for the host tools.
//!
//! In SIMULARE the board clock is the model's (250 µs per launch, 100 ns per method): the numbers there test the
//! counting, they say nothing about the RTX 3070 Ti.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::layer::Shape;
use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use crate::kronos_diag::{self as kd, SeqMode};
use crate::kronos_gen::{TokenOutput, VramBudget};
use crate::kronos_model as km;
use crate::transfer::I11_DEFAULT_BYTES;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The shape of Kronos #513 (hub `integrare/kronos/MODEL.md`): 12 layers, 12 heads, n_emb 768, context 512,
/// vocabulary 16 384, tied head. `eps` is the LayerNorm epsilon of `kronos_gpt` (1e-5).
pub const K513_LAYERS: usize = 12;
pub const K513_VOCAB: usize = 16_384;
pub const K513_CONTEXT: usize = 512;
pub const K513_SHAPE: Shape = Shape {
    e: 768,
    heads: 12,
    eps: 1e-5,
};

/// C-147 / C-150: the shape of Kronos #651 (183 M; cyberspace `config/conv_w2_mare_24zc.toml`: `n_straturi = 24`,
/// `n_capete = 12`, `n_emb = 768`, `context = 512`, tokenizer `bpe_16384`): #513's width, twice the layers. To be
/// checked against hub `integrare/kronos/MODEL_651.md` when the export (C-150) writes it.
pub const K651_LAYERS: usize = 24;
pub const K651_VOCAB: usize = 16_384;
pub const K651_CONTEXT: usize = 512;
pub const K651_SHAPE: Shape = K513_SHAPE;

// ------------------------------------------------------------------------------------------------ limits

/// What one model needs on the work channel (shapes only).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ModelLimits {
    pub layers: usize,
    /// Launches of one token, the copy of the logits down into RAM included.
    pub launches: usize,
    /// Pushbuffer bytes of the largest submission of one token, per way: `Once`, `EachLaunch`, `PerLayer`.
    pub push_once: usize,
    pub push_each: usize,
    pub push_layer: usize,
    /// The pushbuffer area that holds two of the largest submissions in the way that needs the most
    /// (`EachLaunch`), rounded up to a power of two (the OS maps it page-aligned).
    pub push_area: u64,
    /// Payload values one doorbell's records use (the records carry `first .. first + launches`; the next
    /// doorbell must start after them).
    pub stride: u32,
    pub vram: VramBudget,
    /// Upload pieces through the arena (at most `piece` bytes each) and the bytes uploaded (the blob).
    pub pieces: u64,
    pub piece: u64,
}

/// One limit the caller's setup does not meet.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Limit {
    /// The pushbuffer area is smaller than two submissions of `need` bytes.
    PushArea {
        have: u64,
        need: u64,
    },
    /// The payload stride per doorbell is not above the launches of one submission.
    Stride {
        have: u32,
        need: u32,
    },
    /// The model does not fit in the free VRAM (with the reserve), or the blob in the longest free piece.
    Vram {
        need: u64,
        free: u64,
    },
    VramPiece {
        need: u64,
        largest: u64,
    },
}

impl Limit {
    pub fn line(&self) -> String {
        match self {
            Limit::PushArea { have, need } => format!(
                "zona de comenzi: are {} octeti, trebuie cel putin {} (doua simboluri de {} octeti)",
                have,
                need,
                need / 2
            ),
            Limit::Stride { have, need } => format!(
                "pasul valorilor inregistrarilor pe sonerie: {have:#x}, trebuie cel putin {need:#x}"
            ),
            Limit::Vram { need, free } => format!(
                "VRAM: trebuie {} MiB (cu rezerva), libere {} MiB",
                need >> 20,
                free >> 20
            ),
            Limit::VramPiece { need, largest } => format!(
                "VRAM: greutatile cer o bucata continua de {} MiB, cea mai lunga libera are {} MiB",
                need >> 20,
                largest >> 20
            ),
        }
    }
}

impl ModelLimits {
    /// The limits of a model with these shapes; `piece` = the largest upload piece (the OS: 8 MiB, I11's arena).
    pub fn of(
        s: &Shape,
        layers: usize,
        vocab: usize,
        context: usize,
        piece: u64,
    ) -> GpuResult<Self> {
        if piece == 0 || !piece.is_multiple_of(16) {
            return Err(bad("limite: bucata de urcare 0 sau nemultiplu de 16"));
        }
        let vram = VramBudget::of(s, layers, vocab, context)?;
        let launches = km::launches_per_token(layers) + 1;
        let push_once = kd::push_bytes(SeqMode::Once, layers, launches);
        let push_each = kd::push_bytes(SeqMode::EachLaunch, layers, launches);
        let push_layer = kd::push_bytes(SeqMode::PerLayer, layers, launches);
        let worst = push_once.max(push_each).max(push_layer) as u64;
        Ok(ModelLimits {
            layers,
            launches,
            push_once,
            push_each,
            push_layer,
            push_area: (2 * worst).next_power_of_two().max(0x1000),
            stride: launches as u32 + 1,
            vram,
            pieces: vram.blob.div_ceil(piece),
            piece,
        })
    }

    /// #513 with the OS's upload pieces (8 MiB).
    pub fn k513() -> Self {
        // The shapes are constants that `VramBudget::of` accepts (E is a multiple of the Q8 block).
        match Self::of(
            &K513_SHAPE,
            K513_LAYERS,
            K513_VOCAB,
            K513_CONTEXT,
            I11_DEFAULT_BYTES,
        ) {
            Ok(l) => l,
            Err(_) => unreachable!("#513 shapes are valid"),
        }
    }

    /// C-150: #651 (24 layers) with the OS's upload pieces (8 MiB).
    pub fn k651() -> Self {
        match Self::of(
            &K651_SHAPE,
            K651_LAYERS,
            K651_VOCAB,
            K651_CONTEXT,
            I11_DEFAULT_BYTES,
        ) {
            Ok(l) => l,
            Err(_) => unreachable!("#651 shapes are valid"),
        }
    }

    /// Every limit the setup does not meet (empty = it fits). `push_area` = the work channel's pushbuffer bytes,
    /// `stride` = payload values between two doorbells, `free` / `largest` = `GpuMemory::vram_free()`,
    /// `reserve` = VRAM kept free after the upload.
    pub fn check(
        &self,
        push_area: u64,
        stride: u32,
        free: u64,
        largest: u64,
        reserve: u64,
    ) -> Vec<Limit> {
        let mut out = Vec::new();
        let need = 2 * self.push_each.max(self.push_once).max(self.push_layer) as u64;
        if push_area < need {
            out.push(Limit::PushArea {
                have: push_area,
                need,
            });
        }
        if stride < self.stride {
            out.push(Limit::Stride {
                have: stride,
                need: self.stride,
            });
        }
        let total = self.vram.total() + reserve;
        if total > free {
            out.push(Limit::Vram { need: total, free });
        }
        if self.vram.blob > largest {
            out.push(Limit::VramPiece {
                need: self.vram.blob,
                largest,
            });
        }
        out
    }

    /// The lines for the screen and the report.
    pub fn lines(&self) -> Vec<String> {
        alloc::vec![
            format!(
                "limite ({} straturi): {} lansari pe simbol (cu copierea logit-urilor in RAM)",
                self.layers, self.launches
            ),
            format!(
                "limite: comenzi pe simbol: {} B (o invalidare), {} B (invalidare inaintea fiecarei lansari), {} B (pe straturi); zona de comenzi: cel putin {} KiB",
                self.push_once,
                self.push_each,
                self.push_layer,
                self.push_area >> 10
            ),
            format!(
                "limite: pasul valorilor inregistrarilor pe sonerie: cel putin {:#x}",
                self.stride
            ),
            self.vram.line(self.layers),
            format!(
                "limite: urcarea greutatilor: cel mult {} bucati de {} MiB prin arena I11",
                self.pieces,
                self.piece >> 20
            ),
        ]
    }
}

// ------------------------------------------------------------------------------------------------ speed

/// Where the numbers come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    /// `gpu-sim`: the model's clock (says nothing about the board).
    Simulare,
    /// The RTX 3070 Ti: the timestamps of the records the board wrote.
    Placa,
}

impl Source {
    pub fn label(self) -> &'static str {
        match self {
            Source::Simulare => "SIMULARE",
            Source::Placa => "placa",
        }
    }
}

/// The weights' upload: bytes, pieces, the board's time (sum of the copy launches) and the host's (with the
/// driver's work and the journal; `None` when the caller has no clock).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Upload {
    pub bytes: u64,
    pub pieces: u32,
    pub board_ns: u64,
    pub host_us: Option<u64>,
}

/// One position: the board's time (sum of the launches; first to last record) and the host's.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TokenTime {
    pub pos: u32,
    pub busy_ns: u64,
    pub span_ns: u64,
    pub host_us: Option<u64>,
}

/// The counters of one `kronos --gpu` run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Speed {
    pub source: Source,
    pub upload: Upload,
    pub tokens: Vec<TokenTime>,
}

/// Medians and totals of [`Speed`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpeedSummary {
    pub tokens: usize,
    pub median_busy_ns: u64,
    pub median_span_ns: u64,
    pub slowest_span_ns: u64,
    /// Median host time of a position (`None` when no position had one).
    pub median_host_us: Option<u64>,
    /// Positions per second × 100: over the board's spans, and over the host's times.
    pub board_per_s_x100: Option<u64>,
    pub host_per_s_x100: Option<u64>,
    /// Upload MB/s (10^6 bytes) over the board's copy time, and over the host's.
    pub upload_board_mb_s: Option<u64>,
    pub upload_host_mb_s: Option<u64>,
}

fn median(mut v: Vec<u64>) -> Option<u64> {
    v.sort_unstable();
    v.get(v.len() / 2).copied()
}

fn per_s_x100(n: usize, ns: u64) -> Option<u64> {
    (ns > 0 && n > 0).then(|| (n as u128 * 100 * 1_000_000_000 / ns as u128) as u64)
}

fn mb_s(bytes: u64, ns: u64) -> Option<u64> {
    (ns > 0).then(|| (bytes as u128 * 1000 / ns as u128) as u64)
}

fn x100(v: Option<u64>) -> String {
    match v {
        Some(x) => format!("{}.{:02}", x / 100, x % 100),
        None => String::from("?"),
    }
}

fn us(ns: u64) -> u32 {
    (ns / 1000).min(u32::MAX as u64) as u32
}

fn sat(v: u64) -> u32 {
    v.min(u32::MAX as u64) as u32
}

impl Speed {
    pub fn new(source: Source) -> Self {
        Speed {
            source,
            upload: Upload::default(),
            tokens: Vec::new(),
        }
    }

    /// One upload piece: `bytes` copied in `board_ns` (the copy launch, from its records), `host_us` from before
    /// writing the piece into the arena to after its last record was read.
    pub fn add_piece(&mut self, bytes: u64, board_ns: u64, host_us: Option<u64>) {
        let u = &mut self.upload;
        u.bytes += bytes;
        u.pieces += 1;
        u.board_ns += board_ns;
        u.host_us = match (u.host_us, host_us) {
            (None, h) if u.pieces == 1 => h,
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }

    /// One position from what the runner gave back (`kronos_gen::TokenOutput`); `host_us` from before the first
    /// launch was written to after the logits were read. A clock that went back is an error (nothing counted).
    pub fn add_token(
        &mut self,
        pos: usize,
        out: &TokenOutput,
        host_us: Option<u64>,
    ) -> GpuResult<()> {
        if out.end_ns < out.start_ns {
            return Err(bad("viteza: ceasul placii merge inapoi"));
        }
        self.tokens.push(TokenTime {
            pos: sat(pos as u64),
            busy_ns: out.busy_ns,
            span_ns: out.end_ns - out.start_ns,
            host_us,
        });
        Ok(())
    }

    pub fn summary(&self) -> SpeedSummary {
        let t = &self.tokens;
        let hosts: Vec<u64> = t.iter().filter_map(|x| x.host_us).collect();
        let span_sum: u64 = t.iter().map(|x| x.span_ns).sum();
        let host_sum: u64 = hosts.iter().sum();
        SpeedSummary {
            tokens: t.len(),
            median_busy_ns: median(t.iter().map(|x| x.busy_ns).collect()).unwrap_or(0),
            median_span_ns: median(t.iter().map(|x| x.span_ns).collect()).unwrap_or(0),
            slowest_span_ns: t.iter().map(|x| x.span_ns).max().unwrap_or(0),
            median_host_us: median(hosts.clone()),
            board_per_s_x100: per_s_x100(t.len(), span_sum),
            host_per_s_x100: if hosts.len() == t.len() {
                per_s_x100(t.len(), host_sum.saturating_mul(1000))
            } else {
                None
            },
            upload_board_mb_s: mb_s(self.upload.bytes, self.upload.board_ns),
            upload_host_mb_s: self
                .upload
                .host_us
                .and_then(|h| mb_s(self.upload.bytes, h.saturating_mul(1000))),
        }
    }

    /// The lines for the screen and the report (Romanian, with the source in each).
    pub fn lines(&self) -> Vec<String> {
        let s = self.summary();
        let l = self.source.label();
        let u = &self.upload;
        let host = |v: Option<u64>| v.map_or(String::from("?"), |x| format!("{x}"));
        alloc::vec![
            format!(
                "viteza ({l}): urcarea greutatilor: {} octeti in {} bucati; placa {} us ({} MB/s), procesor {} us ({} MB/s)",
                u.bytes,
                u.pieces,
                u.board_ns / 1000,
                host(s.upload_board_mb_s),
                host(u.host_us),
                host(s.upload_host_mb_s)
            ),
            format!(
                "viteza ({l}): {} simboluri; pe placa: median {} us lansarile, {} us prima-ultima inregistrare (cel mai lent {} us) = {} simboluri/s",
                s.tokens,
                s.median_busy_ns / 1000,
                s.median_span_ns / 1000,
                s.slowest_span_ns / 1000,
                x100(s.board_per_s_x100)
            ),
            format!(
                "viteza ({l}): cu driverul, jurnalul si ecranul: median {} us pe simbol = {} simboluri/s",
                host(s.median_host_us),
                x100(s.host_per_s_x100)
            ),
        ]
    }

    /// The `kv …` rows into the journal: the source, the upload, (optionally) every position, then the summary.
    /// On the board each row costs a write on the stick (~17 ms at `os-306`), so `per_token` = false keeps a long
    /// generation's journal short (5 + 9 rows whatever the length).
    pub fn journal(&self, j: &mut Journal, per_token: bool) {
        for (what, value) in self.rows(per_token) {
            j.push(Event::Value { what, value });
        }
    }

    /// The rows [`Self::journal`] writes, in order.
    pub fn rows(&self, per_token: bool) -> Vec<(&'static str, u32)> {
        use rows as n;
        let s = self.summary();
        let none = u32::MAX;
        let u = &self.upload;
        let mut v = alloc::vec![
            (
                n::SOURCE,
                match self.source {
                    Source::Simulare => 0,
                    Source::Placa => 1,
                }
            ),
            (n::UP_KIB, sat(u.bytes.div_ceil(1024))),
            (n::UP_PIECES, u.pieces),
            (n::UP_BOARD, us(u.board_ns)),
            (n::UP_HOST, u.host_us.map_or(none, sat)),
        ];
        if per_token {
            for t in &self.tokens {
                v.push((n::POS, t.pos));
                v.push((n::BUSY, us(t.busy_ns)));
                v.push((n::SPAN, us(t.span_ns)));
                v.push((n::HOST, t.host_us.map_or(none, sat)));
            }
        }
        v.extend([
            (n::TOKENS, sat(s.tokens as u64)),
            (n::MED_BUSY, us(s.median_busy_ns)),
            (n::MED_SPAN, us(s.median_span_ns)),
            (n::SLOWEST, us(s.slowest_span_ns)),
            (n::MED_HOST, s.median_host_us.map_or(none, sat)),
            (n::BOARD_X100, s.board_per_s_x100.map_or(none, sat)),
            (n::HOST_X100, s.host_per_s_x100.map_or(none, sat)),
            (n::UP_BOARD_MB, s.upload_board_mb_s.map_or(none, sat)),
            (n::UP_HOST_MB, s.upload_host_mb_s.map_or(none, sat)),
        ]);
        v
    }
}

/// The names of the `kv` rows (what the stick journal prints after `valoare `).
pub mod rows {
    pub const SOURCE: &str = "kv sursa (0 SIMULARE, 1 placa)";
    pub const UP_KIB: &str = "kv urcare KiB";
    pub const UP_PIECES: &str = "kv urcare bucati";
    pub const UP_BOARD: &str = "kv urcare pe placa (us, copierile)";
    pub const UP_HOST: &str = "kv urcare procesor (us, ffffffff = fara ceas)";
    pub const POS: &str = "kv pozitia";
    pub const BUSY: &str = "kv simbol pe placa lansarile (us)";
    pub const SPAN: &str = "kv simbol pe placa prima-ultima inregistrare (us)";
    pub const HOST: &str = "kv simbol procesor (us, cu jurnalul; ffffffff = fara ceas)";
    pub const TOKENS: &str = "kv simboluri";
    pub const MED_BUSY: &str = "kv median lansarile (us)";
    pub const MED_SPAN: &str = "kv median prima-ultima inregistrare (us)";
    pub const SLOWEST: &str = "kv cel mai lent simbol pe placa (us)";
    pub const MED_HOST: &str = "kv median procesor (us)";
    pub const BOARD_X100: &str = "kv simboluri pe secunda x100 (placa)";
    pub const HOST_X100: &str = "kv simboluri pe secunda x100 (procesor)";
    pub const UP_BOARD_MB: &str = "kv urcare MB/s (placa)";
    pub const UP_HOST_MB: &str = "kv urcare MB/s (procesor)";
}

/// What a journal's `kv` rows say (the last run in the text; `None` = row absent or `ffffffff`).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedSpeed {
    pub source: Option<u32>,
    pub up_kib: Option<u32>,
    pub up_pieces: Option<u32>,
    pub up_board_us: Option<u32>,
    pub up_host_us: Option<u32>,
    /// (position, busy us, span us, host us) of each position journaled.
    pub tokens: Vec<(u32, u32, u32, Option<u32>)>,
    pub n_tokens: Option<u32>,
    pub median_busy_us: Option<u32>,
    pub median_span_us: Option<u32>,
    pub slowest_us: Option<u32>,
    pub median_host_us: Option<u32>,
    pub board_x100: Option<u32>,
    pub host_x100: Option<u32>,
    pub up_board_mb_s: Option<u32>,
    pub up_host_mb_s: Option<u32>,
}

/// Reads the `kv` rows of a stick journal (`… gpu: valoare <name> = 0x<hex>`); a new `kv sursa` row starts a new
/// run (only the last one is kept). Rows of other steps are skipped.
pub fn parse_journal(text: &str) -> ParsedSpeed {
    use rows as n;
    let mut out = ParsedSpeed::default();
    let mut pending: Option<(u32, Option<u32>, Option<u32>)> = None;
    for line in text.lines() {
        let Some(rest) = line.split("valoare ").nth(1) else {
            continue;
        };
        let Some((name, v)) = rest.rsplit_once(" = ") else {
            continue;
        };
        let Some(v) = v
            .trim()
            .strip_prefix("0x")
            .and_then(|h| u32::from_str_radix(h, 16).ok())
        else {
            continue;
        };
        let some = (v != u32::MAX).then_some(v);
        match name {
            x if x == n::SOURCE => {
                out = ParsedSpeed {
                    source: Some(v),
                    ..ParsedSpeed::default()
                };
                pending = None;
            }
            x if x == n::UP_KIB => out.up_kib = some,
            x if x == n::UP_PIECES => out.up_pieces = some,
            x if x == n::UP_BOARD => out.up_board_us = some,
            x if x == n::UP_HOST => out.up_host_us = some,
            x if x == n::POS => pending = Some((v, None, None)),
            x if x == n::BUSY => {
                if let Some(p) = pending.as_mut() {
                    p.1 = Some(v);
                }
            }
            x if x == n::SPAN => {
                if let Some(p) = pending.as_mut() {
                    p.2 = Some(v);
                }
            }
            x if x == n::HOST => {
                if let Some((pos, Some(b), Some(s))) = pending.take() {
                    out.tokens.push((pos, b, s, some));
                }
            }
            x if x == n::TOKENS => out.n_tokens = some,
            x if x == n::MED_BUSY => out.median_busy_us = some,
            x if x == n::MED_SPAN => out.median_span_us = some,
            x if x == n::SLOWEST => out.slowest_us = some,
            x if x == n::MED_HOST => out.median_host_us = some,
            x if x == n::BOARD_X100 => out.board_x100 = some,
            x if x == n::HOST_X100 => out.host_x100 = some,
            x if x == n::UP_BOARD_MB => out.up_board_mb_s = some,
            x if x == n::UP_HOST_MB => out.up_host_mb_s = some,
            _ => {}
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k513_limits_are_the_ones_the_notes_name() {
        let l = ModelLimits::k513();
        assert_eq!(l.launches, 196);
        assert_eq!(l.push_each, 9440);
        assert_eq!(l.push_once, crate::kronos_gen::push_bytes_with_copy(12));
        assert_eq!(l.push_layer, 712);
        // Two submissions of 9 440 B do not fit in 16 KiB; 32 KiB is the area.
        assert_eq!(l.push_area, 32 << 10);
        assert_eq!(l.stride, 197);
        assert!(l.stride <= 0x100, "the OS's STRIDE 0x100 is enough");
        // ~106 MiB of weights + scratch + logits → 14 pieces of 8 MiB.
        assert!(l.vram.blob > 100 << 20 && l.vram.blob < 112 << 20);
        assert_eq!(l.pieces, l.vram.blob.div_ceil(8 << 20));
        assert_eq!(l.pieces, 14);
        assert!(l.vram.total() < 160 << 20);
        assert_eq!(l.lines().len(), 5);
    }

    #[test]
    fn k651_limits() {
        let l = ModelLimits::k651();
        assert_eq!(l.layers, 24);
        // 16 × 24 + 3 = 387 launches of the model + the copy of the logits down.
        assert_eq!(l.launches, 388);
        assert_eq!(l.push_area, 64 << 10);
        // The payload stride: 389 — above the 0x100 the OS uses for #513 (it needs 0x200 for #651).
        assert_eq!(l.stride, 389);
        assert!(l.stride > 0x100 && l.stride <= 0x200);
        let gib = 1u64 << 30;
        assert!(
            l.check(64 << 10, 0x200, 7 * gib, 7 * gib, 4 << 20)
                .is_empty()
        );
        let bad = l.check(32 << 10, 0x100, 7 * gib, 7 * gib, 4 << 20);
        assert_eq!(bad.len(), 2, "{bad:?}");
        // ~183 M weights at Q8_0 (1.0625 B each) + scratch, caches, logits: ~270 MiB on the board.
        assert!(
            l.vram.blob > 190 << 20 && l.vram.blob < 215 << 20,
            "{}",
            l.vram.blob >> 20
        );
        assert!(
            l.vram.total() > 250 << 20 && l.vram.total() < 290 << 20,
            "{}",
            l.vram.total() >> 20
        );
        assert_eq!(l.pieces, l.vram.blob.div_ceil(8 << 20));
    }

    #[test]
    fn check_names_every_limit_not_met() {
        let l = ModelLimits::k513();
        let gib = 1u64 << 30;
        assert!(
            l.check(32 << 10, 0x100, 7 * gib, 7 * gib, 4 << 20)
                .is_empty()
        );
        let bad = l.check(16 << 10, 0x80, 100 << 20, 50 << 20, 4 << 20);
        assert_eq!(bad.len(), 4);
        assert!(matches!(
            bad[0],
            Limit::PushArea {
                have: 16384,
                need: 18880
            }
        ));
        assert!(matches!(
            bad[1],
            Limit::Stride {
                have: 0x80,
                need: 197
            }
        ));
        assert!(matches!(bad[2], Limit::Vram { .. }));
        assert!(matches!(bad[3], Limit::VramPiece { .. }));
        assert!(bad.iter().all(|b| !b.line().is_empty()));
        // 24 layers (the 183M of C-87): 64 KiB.
        let l24 = ModelLimits::of(&K513_SHAPE, 24, K513_VOCAB, K513_CONTEXT, 8 << 20).unwrap();
        assert_eq!(l24.push_area, 64 << 10);
        assert!(ModelLimits::of(&K513_SHAPE, 12, K513_VOCAB, K513_CONTEXT, 0).is_err());
        assert!(ModelLimits::of(&K513_SHAPE, 12, K513_VOCAB, K513_CONTEXT, 8).is_err());
    }

    fn out(start: u64, end: u64, busy: u64) -> TokenOutput {
        TokenOutput {
            logits: Vec::new(),
            start_ns: start,
            end_ns: end,
            busy_ns: busy,
        }
    }

    #[test]
    fn speed_counts_board_and_host_apart() {
        let mut s = Speed::new(Source::Simulare);
        s.add_piece(8 << 20, 2_000_000, Some(300_000));
        s.add_piece(4 << 20, 1_000_000, Some(200_000));
        assert_eq!(s.upload.pieces, 2);
        assert_eq!(s.upload.host_us, Some(500_000));
        s.add_token(0, &out(10, 2_000_010, 1_500_000), Some(270_000))
            .unwrap();
        s.add_token(1, &out(3_000_000, 4_000_000, 900_000), Some(260_000))
            .unwrap();
        s.add_token(2, &out(5_000_000, 8_000_000, 2_500_000), Some(280_000))
            .unwrap();
        assert!(s.add_token(3, &out(9, 8, 0), None).is_err());
        let m = s.summary();
        assert_eq!(m.tokens, 3);
        assert_eq!(m.median_busy_ns, 1_500_000);
        assert_eq!(m.median_span_ns, 2_000_000);
        assert_eq!(m.slowest_span_ns, 3_000_000);
        assert_eq!(m.median_host_us, Some(270_000));
        // 3 positions over 6 ms of board spans = 500/s; over 810 ms of host = 3.70/s.
        assert_eq!(m.board_per_s_x100, Some(50_000));
        assert_eq!(m.host_per_s_x100, Some(370));
        // 12 MiB in 3 ms = 4194 MB/s; in 500 ms = 25 MB/s.
        assert_eq!(m.upload_board_mb_s, Some(4194));
        assert_eq!(m.upload_host_mb_s, Some(25));
        let lines = s.lines();
        assert!(lines.iter().all(|l| l.contains("(SIMULARE)")));
        assert!(lines[2].contains("3.70 simboluri/s"));
    }

    #[test]
    fn a_missing_host_clock_is_said_not_guessed() {
        let mut s = Speed::new(Source::Placa);
        s.add_piece(16, 100, None);
        s.add_piece(16, 100, Some(5));
        assert_eq!(s.upload.host_us, None);
        s.add_token(0, &out(0, 1000, 800), None).unwrap();
        s.add_token(1, &out(1000, 2000, 800), Some(9)).unwrap();
        let m = s.summary();
        assert_eq!(m.host_per_s_x100, None);
        assert_eq!(m.median_host_us, Some(9));
        assert_eq!(m.upload_host_mb_s, None);
        assert!(s.lines()[0].contains("procesor ? us"));
    }

    #[test]
    fn journal_rows_read_back() {
        let mut s = Speed::new(Source::Placa);
        s.add_piece(8 << 20, 2_000_000, Some(300_000));
        s.add_token(5, &out(0, 2_000_000, 1_500_000), Some(270_000))
            .unwrap();
        s.add_token(6, &out(2_500_000, 4_000_000, 1_000_000), None)
            .unwrap();
        let mut j = Journal::new(64);
        s.journal(&mut j, true);
        assert_eq!(j.len(), 5 + 2 * 4 + 9);
        let mut j2 = Journal::new(64);
        s.journal(&mut j2, false);
        assert_eq!(j2.len(), 14);
        // An older run first: only the last one is kept.
        let mut old = Speed::new(Source::Simulare);
        old.add_token(0, &out(0, 1, 1), None).unwrap();
        let text: String = old
            .rows(true)
            .iter()
            .chain(s.rows(true).iter())
            .map(|(w, v)| format!("[  1.000000] gpu: valoare {w} = {v:#010x}\n"))
            .collect();
        let p = parse_journal(&format!("[  0.5] gpu: pas i14-simbol: ok\n{text}"));
        assert_eq!(p.source, Some(1));
        assert_eq!(p.up_kib, Some(8192));
        assert_eq!(p.up_pieces, Some(1));
        assert_eq!(p.up_board_us, Some(2000));
        assert_eq!(p.up_host_us, Some(300_000));
        assert_eq!(
            p.tokens,
            alloc::vec![(5, 1500, 2000, Some(270_000)), (6, 1000, 1500, None)]
        );
        assert_eq!(p.n_tokens, Some(2));
        assert_eq!(p.median_busy_us, Some(1500));
        assert_eq!(p.median_host_us, Some(270_000));
        assert_eq!(p.host_x100, None);
        assert_eq!(p.board_x100, Some(57_142));
        assert_eq!(p.up_board_mb_s, Some(4194));
    }
}
