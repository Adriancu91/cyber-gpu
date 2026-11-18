//! C-139 (after os-305: I9–I13 pass on the board, `i14-simbol` fails with "argmax diferit de vectorii lui Kronos"
//! and nothing else reaches the stick journal) — **where** a token goes wrong, written into the journal.
//!
//! Three things, all driven through three small operations the caller implements ([`TokenIo`]: send a list of
//! launches as one submission; read words of GPU memory back; write a few words — plus the journal the rows go in):
//!
//! - [`SeqMode`] / [`Way`]: how the launches of one token are sent. `Once` is what os-305 ran (C-83: the shader
//!   caches invalidated once per submission); `EachLaunch` is the repair (`transfer::encode_sequence_fresh`);
//!   `PerLayer` sends one submission per layer — the regime I13 passed on the board. A [`Way`] also says who writes
//!   the embedding: the program `embed` (the one program of a token that I13 never ran on the board) or the driver
//!   ([`driver_embedding`], as I13 wrote its x).
//! - [`probe`] / [`examine`]: after a run, the activation scratch, the new cache rows, the final vector and the
//!   logits are read back and every launch's output is compared (a) with what the launch should have written
//!   (`gpu_compute::model::trace_gpu`, the program models), (b) with the launch's model fed with the **board's own
//!   inputs** (does the output follow from what is in memory?), and (c), when it does not, with the model fed with an
//!   **older version** of an input (the value the same address held earlier in the token) — the signature of a
//!   stale read. The first stage outside the tolerance and its kind are the [`Verdict`].
//! - [`ladder`]: the first token of a control text in the three modes, then — only if none passes — prefixes of the
//!   launch list (the embedding alone; if it is the wrong one, the token again with the driver's embedding; then
//!   each layer with its input still in place), until the stage is named. Its answer is the [`Way`] the texts run in.
//! - [`explain`]: after a token of the caller's own run fails, memory as that token left it, judged (nothing sent);
//!   it also tells when only the copy of the logits into RAM is wrong ([`Explained::vram_pass`]).
//!
//! Everything is journaled ([`Probe::journal`], [`LogitReport::journal`]) with fixed row names ([`rows`]) and read
//! back from a stick journal by [`parse_journal`].
//!
//! Tested **only in SIMULARE** (`tests/sim_i14_diag.rs`, with the GPU model's stale-read option standing in for the
//! hypothesis). Nothing here touches a board; no register is named here.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::ops::Range;

use gpu_compute::layer::{self, LayerTrace, LayerWeights, Shape, TOL_ABS, TOL_REL};
use gpu_compute::model::{self, Caches, LOGIT_ABS, LOGIT_REL, ModelWeights, TokenTrace};
use gpu_compute::q8;
use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use crate::kronos_layer::{self as kl, LAUNCHES_PER_TOKEN, Launch, LayerVas};
use crate::kronos_model::{self as km, ModelVas};
use crate::push::PushBuf;
use crate::transfer;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

// ------------------------------------------------------------------------------------------------ how a token is sent

/// How the launches of one token are sent to the work channel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeqMode {
    /// One submission, the shader caches invalidated once at its start ([`transfer::encode_sequence`]): what
    /// `i14-simbol` ran in os-305.
    Once = 1,
    /// One submission, the caches invalidated before every launch ([`transfer::encode_sequence_fresh`]).
    EachLaunch = 2,
    /// One submission per layer, each as `Once`: `[embedding + layer 0]`, `[layer 1]`, …, `[final LayerNorm, logits
    /// (+ what follows)]` — inside one submission no address is read again after it was rewritten, as in I13.
    PerLayer = 3,
}

impl SeqMode {
    pub fn from_code(c: u32) -> Option<SeqMode> {
        match c {
            1 => Some(SeqMode::Once),
            2 => Some(SeqMode::EachLaunch),
            3 => Some(SeqMode::PerLayer),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SeqMode::Once => "o invalidare pe trimitere (ca in os-305)",
            SeqMode::EachLaunch => "invalidare inaintea fiecarei lansari",
            SeqMode::PerLayer => "o trimitere pe strat (ca la I13)",
        }
    }
}

/// Who puts the embedding (x before layer 0) in memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbedBy {
    /// Launch 0 of the token's list: the program `embed` (the one program I13 never ran on the board).
    Program = 0,
    /// The driver computes it ([`driver_embedding`]: E multiply-adds) and writes it at `x` before the first
    /// submission, as I13 wrote its x; launch 0 is not sent. A way AROUND a program that is wrong on the board: the
    /// rest of the token can still be judged, but a step that needed it has not passed.
    Driver = 1,
}

/// How a token is sent: the mode of its submissions and who writes the embedding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Way {
    pub mode: SeqMode,
    pub embed: EmbedBy,
}

impl Way {
    /// What `i14-simbol` ran in os-305.
    pub const OS_305: Way = Way::of(SeqMode::Once);
    /// The repair of C-139.
    pub const REPAIR: Way = Way::of(SeqMode::EachLaunch);

    pub const fn of(mode: SeqMode) -> Way {
        Way {
            mode,
            embed: EmbedBy::Program,
        }
    }

    pub const fn with_driver_embedding(self) -> Way {
        Way {
            mode: self.mode,
            embed: EmbedBy::Driver,
        }
    }

    pub fn name(&self) -> String {
        match self.embed {
            EmbedBy::Program => String::from(self.mode.name()),
            EmbedBy::Driver => format!("{}, embedding scris de driver", self.mode.name()),
        }
    }

    /// The submissions of the first `n` launches of a token's list in this way: [`submissions`], without launch 0
    /// when the driver writes the embedding. The ranges index the token's list (and its launch slots).
    pub fn submissions(&self, layers: usize, n: usize) -> Vec<Range<usize>> {
        let mut s = submissions(self.mode, layers, n);
        if self.embed == EmbedBy::Driver
            && let Some(first) = s.first_mut()
        {
            first.start = 1;
            if first.start >= first.end {
                s.remove(0);
            }
        }
        s
    }
}

impl From<SeqMode> for Way {
    /// The mode with the program's embedding.
    fn from(mode: SeqMode) -> Way {
        Way::of(mode)
    }
}

/// The embedding of `token` at `pos` as the driver computes it when it does not launch `embed`: the bits of E
/// values, to be written at `x` (`ModelVas::layers[0].x`) before the first submission. The same arithmetic as the
/// program (`gpu_compute::model::embed_gpu`), so the rest of the token is unchanged bit for bit.
pub fn driver_embedding(w: &ModelWeights<'_>, token: usize, pos: usize) -> GpuResult<Vec<u32>> {
    w.check()?;
    if token >= w.vocab || pos >= w.context {
        return Err(bad("model: simbol sau pozitie in afara tabelelor"));
    }
    Ok(model::embed_gpu(
        &w.emb_tok[w.row_offset(token)..],
        &w.emb_pos[w.row_offset(pos)..],
        w.shape.e,
    )
    .iter()
    .map(|v| v.to_bits())
    .collect())
}

/// The submissions of the first `n` launches of a token's list (`n` may include launches the caller appends after
/// [`km::token_launches`], e.g. the copy of the logits): index ranges, in order.
pub fn submissions(mode: SeqMode, layers: usize, n: usize) -> Vec<Range<usize>> {
    if n == 0 {
        return Vec::new();
    }
    match mode {
        SeqMode::Once | SeqMode::EachLaunch => alloc::vec![0..n],
        SeqMode::PerLayer => {
            let mut out = Vec::new();
            let mut start = 0;
            // Ends of the layers: the embedding rides with layer 0.
            for l in 0..layers {
                let end = (1 + LAUNCHES_PER_TOKEN * (l + 1)).min(n);
                if end > start {
                    out.push(start..end);
                    start = end;
                }
            }
            if n > start {
                out.push(start..n);
            }
            out
        }
    }
}

/// Encodes one submission of `qmds` for `mode` (the records are the same in every mode).
pub fn encode(
    mode: SeqMode,
    p: &mut PushBuf<'_>,
    qmds: &[u64],
    records_va: u64,
    first_payload: u32,
) -> GpuResult<()> {
    match mode {
        SeqMode::Once | SeqMode::PerLayer => {
            transfer::encode_sequence(p, qmds, records_va, first_payload)
        }
        SeqMode::EachLaunch => transfer::encode_sequence_fresh(p, qmds, records_va, first_payload),
    }
}

/// Pushbuffer words of one submission of `n` launches in `mode`.
pub const fn words(mode: SeqMode, n: usize) -> usize {
    match mode {
        SeqMode::Once | SeqMode::PerLayer => transfer::sequence_words(n),
        SeqMode::EachLaunch => transfer::sequence_words_fresh(n),
    }
}

/// Bytes of the largest pushbuffer stretch a token of `launches` launches (the model's, plus what the caller
/// appends) needs in `mode`: the work channel's pushbuffer area must hold it (twice, to write the next token while
/// one is in flight). #513 with the copy of the logits (196 launches): 7 872 B as before, 9 440 B with `EachLaunch`,
/// 712 B per layer.
pub fn push_bytes(mode: SeqMode, layers: usize, launches: usize) -> usize {
    submissions(mode, layers, launches)
        .iter()
        .map(|r| 4 * words(mode, r.len()))
        .max()
        .unwrap_or(0)
}

/// The three operations the diagnosis needs from the caller (the OS on the board, the tests on `gpu-sim`).
pub trait TokenIo {
    /// Sends `launches` as ONE submission encoded with [`encode`] for `mode` and waits for its last record
    /// (constant buffer + QMD of `launches[i]` written at launch slot `first_slot + i`). An error (a record that
    /// never comes) ends the diagnosis: nothing more is sent.
    fn submit(&mut self, launches: &[Launch], first_slot: usize, mode: SeqMode) -> GpuResult<()>;

    /// Reads GPU memory back: for each `(GPU VA, words)` the 32-bit words there (VRAM through the window).
    fn read(&mut self, ranges: &[(u64, usize)]) -> GpuResult<Vec<Vec<u32>>>;

    /// Writes `words` at GPU VA `va` (VRAM through the window; only used for the embedding written by the driver,
    /// E words at `x`). It may be queued, but must be in memory before the next [`Self::submit`] runs.
    fn write(&mut self, va: u64, words: &[u32]) -> GpuResult<()>;

    /// The journal the rows of the diagnosis go into — the one the caller's own accesses are written in (on the
    /// board: the stick journal), so a probe's rows follow the accesses of its submissions, in order.
    fn journal(&mut self) -> &mut Journal;
}

/// Sends `list` (the first launches of a token's list, slots = their indices) in `way`: the embedding written by
/// the driver first, if the way says so, then the way's submissions.
fn send<I: TokenIo + ?Sized>(
    io: &mut I,
    w: &ModelWeights<'_>,
    vas: &ModelVas,
    list: &[Launch],
    token: usize,
    pos: usize,
    way: Way,
) -> GpuResult<()> {
    vas.check()?;
    if way.embed == EmbedBy::Driver {
        io.write(vas.layers[0].x, &driver_embedding(w, token, pos)?)?;
    }
    for r in way.submissions(w.layers.len(), list.len()) {
        io.submit(&list[r.clone()], r.start, way.mode)?;
    }
    Ok(())
}

/// One whole token in `way`, as the texts run it once [`ladder`] chose the way: the model's launches
/// ([`km::token_launches`]) followed by `extra` (e.g. the copy of the logits down into RAM,
/// `kronos_layer::copy_launch`), the driver's embedding written first when the way says so, split into the way's
/// submissions. The launch area needs `launches_per_token(layers) + extra.len()` slots.
pub fn send_token<I: TokenIo + ?Sized>(
    io: &mut I,
    w: &ModelWeights<'_>,
    vas: &ModelVas,
    token: usize,
    pos: usize,
    way: Way,
    extra: &[Launch],
) -> GpuResult<()> {
    let mut list = km::token_launches(&w.shape, w.vocab, w.context, vas, token, pos)?;
    list.extend_from_slice(extra);
    send(io, w, vas, &list, token, pos, way)
}

// ------------------------------------------------------------------------------------------------ the logits

/// What the logits of one position say, next to the CPU engine's (and Kronos's vectors, for a control text).
#[derive(Clone, Debug, PartialEq)]
pub struct LogitReport {
    pub token: usize,
    pub pos: usize,
    /// Greedy token of the board, of the CPU engine, and Kronos's (`vectori.json`), when known.
    pub board: Option<usize>,
    pub cpu: Option<usize>,
    pub want: Option<usize>,
    /// Board logits that are NaN or infinite.
    pub not_finite: usize,
    /// Board logits whose bits are `0xeeeeeeee`: the pattern the OS fills the RAM buffer with — never written.
    pub fill: usize,
    /// The first 8 logits: (board, CPU).
    pub first: Vec<(f32, f32)>,
    /// The logit (board, CPU) at the board's greedy token and at the CPU's.
    pub at_board_top: Option<(f32, f32)>,
    pub at_cpu_top: Option<(f32, f32)>,
    /// Largest |board − CPU| over the finite pairs, and where.
    pub max_abs: f32,
    pub max_at: usize,
    /// Logits outside `LOGIT_ABS + LOGIT_REL·|CPU|` (or not finite).
    pub outside: usize,
    /// Top-1 minus top-2 on the CPU: how far a rounding difference is from changing the token.
    pub cpu_margin: f32,
    pub passes: bool,
}

/// The pattern the OS fills the logits' RAM buffer with before every token.
pub const RAM_FILL: u32 = 0xeeee_eeee;

fn top2_margin(v: &[f32]) -> f32 {
    let Some(top) = model::argmax(v) else {
        return f32::NAN;
    };
    let second = v
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != top)
        .fold(f32::NEG_INFINITY, |m, (_, &x)| m.max(x));
    v[top] - second
}

/// Compares the board's logits with the CPU engine's (the judgement of `kronos_model::judge` plus what the journal
/// needs to say where they differ).
pub fn logit_report(
    token: usize,
    pos: usize,
    board: &[f32],
    cpu: &[f32],
    want: Option<usize>,
) -> LogitReport {
    let (b, c) = (model::argmax(board), model::argmax(cpu));
    let mut max_abs = 0.0f32;
    let mut max_at = 0;
    let mut outside = 0;
    for i in 0..board.len().max(cpu.len()) {
        let g = board.get(i).copied().unwrap_or(f32::NAN);
        let w = cpu.get(i).copied().unwrap_or(f32::NAN);
        if g.is_finite() && w.is_finite() {
            let d = (g - w).abs();
            if d > max_abs {
                (max_abs, max_at) = (d, i);
            }
            if d > LOGIT_ABS + LOGIT_REL * w.abs() {
                outside += 1;
            }
        } else {
            outside += 1;
        }
    }
    let pair = |i: Option<usize>| i.and_then(|i| Some((*board.get(i)?, *cpu.get(i)?)));
    let judged = km::judge(board, cpu).passes();
    LogitReport {
        token,
        pos,
        board: b,
        cpu: c,
        want,
        not_finite: board.iter().filter(|v| !v.is_finite()).count(),
        fill: board.iter().filter(|v| v.to_bits() == RAM_FILL).count(),
        first: board
            .iter()
            .zip(cpu)
            .take(8)
            .map(|(&g, &w)| (g, w))
            .collect(),
        at_board_top: pair(b),
        at_cpu_top: pair(c),
        max_abs,
        max_at,
        outside,
        cpu_margin: top2_margin(cpu),
        passes: judged && want.is_none_or(|w| b == Some(w)),
    }
}

/// Row names of the journal (short ASCII; the stick journal prints `valoare <name> = 0x…` / `text <name> = "…"`).
/// Floats are journaled as their 32 bits (exact); [`parse_journal`] reads them back.
pub mod rows {
    pub const PROBE: &str =
        "i14d proba (1 o invalidare, 2 inaintea fiecarei lansari, 3 pe straturi, 0 simbolul rulat)";
    pub const EMBED_BY: &str = "i14d embedding scris de (0 programul embed, 1 driver)";
    pub const LAUNCHES: &str = "i14d lansari trimise";
    pub const TOKEN: &str = "i14d simbol";
    pub const POS: &str = "i14d pozitia";
    pub const VERDICT: &str = "i14d rezultat (1 trece, 2 date vechi, 3 lansare gresita, 4 mostenit, 5 placa nu a terminat, 6 doar logit-urile)";
    pub const STAGES_OK: &str = "i14d etape in toleranta";
    pub const STAGES_BAD: &str = "i14d etape in afara tolerantei";
    pub const STAGE: &str = "i14d etapa (strat << 8 | lansare; fe = embedding, ff = final)";
    pub const STAGE_NAME: &str = "i14d etapa nume";
    pub const STAGE_DEV: &str = "i14d etapa abatere fata de model (f32)";
    pub const STAGE_FLAGS: &str = "i14d etapa stare (b0 in toleranta, b1 verificata local, b2 local ok, b3 date vechi, b8-15 din stratul / fe embedding)";
    pub const STAGE_LOCAL: &str = "i14d etapa abatere fata de intrarile placii (f32)";
    pub const ARGMAX_BOARD: &str = "i14d argmax placa (ffffffff = niciunul)";
    pub const ARGMAX_CPU: &str = "i14d argmax procesor";
    pub const ARGMAX_WANT: &str = "i14d argmax vectori.json (ffffffff = nu e text de control)";
    pub const NOT_FINITE: &str = "i14d logit-uri NaN sau infinit";
    pub const FILL: &str = "i14d logit-uri ramase eeeeeeee (nescrise in RAM)";
    pub const LOGIT_BOARD: &str = "i14d logit placa (f32, primele 8 in ordine)";
    pub const LOGIT_CPU: &str = "i14d logit procesor (f32, primele 8 in ordine)";
    pub const TOP_BOARD_B: &str = "i14d logit placa la argmax-ul placii (f32)";
    pub const TOP_BOARD_C: &str = "i14d logit procesor la argmax-ul placii (f32)";
    pub const TOP_CPU_B: &str = "i14d logit placa la argmax-ul procesorului (f32)";
    pub const TOP_CPU_C: &str = "i14d logit procesor la argmax-ul procesorului (f32)";
    pub const MAX_ABS: &str = "i14d abaterea maxima a logit-urilor (f32)";
    pub const MAX_AT: &str = "i14d abaterea maxima la indexul";
    pub const OUTSIDE: &str = "i14d logit-uri in afara tolerantei 1e-3";
    pub const MARGIN: &str = "i14d marja top1-top2 pe procesor (f32)";
    pub const RAM_DIFF: &str = "i14d logit-uri diferite intre RAM si VRAM (copierea in jos)";
    pub const SUMMARY: &str = "i14d rezumat";
    pub const SUMMARY_MORE: &str = "i14d rezumat (urmare)";
    pub const CHOSEN: &str =
        "i14d modul ales pentru texte (0 niciunul, 2 inaintea fiecarei lansari, 3 pe straturi)";
    pub const CHOSEN_EMBED: &str = "i14d embedding pentru texte (0 programul embed, 1 driver)";
}

fn val(j: &mut Journal, what: &'static str, value: u32) {
    j.push(Event::Value { what, value });
}

fn opt(v: Option<usize>) -> u32 {
    v.map_or(u32::MAX, |x| x.min(u32::MAX as usize - 1) as u32)
}

impl LogitReport {
    /// One line for the screen / the chain report.
    pub fn line(&self) -> String {
        let s = |v: Option<usize>| v.map_or(String::from("-"), |x| format!("{x}"));
        format!(
            "i14: pozitia {} (simbolul {}): argmax placa {}, procesor {}, vectori {}; abaterea maxima a logit-urilor {:e} la {}; {} in afara tolerantei; {} nefinite; {} nescrise; marja pe procesor {:e}",
            self.pos,
            self.token,
            s(self.board),
            s(self.cpu),
            s(self.want),
            self.max_abs,
            self.max_at,
            self.outside,
            self.not_finite,
            self.fill,
            self.cpu_margin
        )
    }

    /// The rows C-139 asks for: position, the three greedy tokens, the first 8 logits of the board and of the CPU,
    /// the largest difference.
    pub fn journal(&self, j: &mut Journal) {
        use rows as n;
        val(j, n::TOKEN, self.token as u32);
        val(j, n::POS, self.pos as u32);
        val(j, n::ARGMAX_BOARD, opt(self.board));
        val(j, n::ARGMAX_CPU, opt(self.cpu));
        val(j, n::ARGMAX_WANT, opt(self.want));
        val(j, n::NOT_FINITE, self.not_finite as u32);
        val(j, n::FILL, self.fill as u32);
        for &(b, _) in &self.first {
            val(j, n::LOGIT_BOARD, b.to_bits());
        }
        for &(_, c) in &self.first {
            val(j, n::LOGIT_CPU, c.to_bits());
        }
        if let Some((b, c)) = self.at_board_top {
            val(j, n::TOP_BOARD_B, b.to_bits());
            val(j, n::TOP_BOARD_C, c.to_bits());
        }
        if let Some((b, c)) = self.at_cpu_top {
            val(j, n::TOP_CPU_B, b.to_bits());
            val(j, n::TOP_CPU_C, c.to_bits());
        }
        val(j, n::MAX_ABS, self.max_abs.to_bits());
        val(j, n::MAX_AT, self.max_at as u32);
        val(j, n::OUTSIDE, self.outside as u32);
        val(j, n::MARGIN, self.cpu_margin.to_bits());
    }
}

// ------------------------------------------------------------------------------------------------ the stages

/// `StageCheck::layer` of the embedding launch and of the two final launches.
pub const EMBED: u8 = 0xfe;
pub const FINAL: u8 = 0xff;

/// Names of the 16 launches of a layer, in order (`kronos_layer::token_launches`).
pub const LAYER_STAGES: [&str; 16] = [
    "normalizare 1",
    "qkv = Wqkv * h",
    "qkv + bqkv",
    "copiere K in cache",
    "copiere V in cache",
    "atentie",
    "o = Wout * att",
    "o + bout",
    "x1 = x + ob",
    "normalizare 2",
    "f = Wfc * h2",
    "f + bfc",
    "gelu",
    "p = Wproj * g",
    "p + bproj",
    "x = x1 + pb",
];

/// The name of a stage by its code.
pub fn stage_name(layer: u8, launch: u8) -> &'static str {
    match (layer, launch) {
        (EMBED, _) => "embedding (simbol + pozitie)",
        (FINAL, 0) => "normalizare finala",
        (FINAL, _) => "logit-uri = Emb * h",
        (_, l) => LAYER_STAGES.get(l as usize).copied().unwrap_or("?"),
    }
}

/// Where an older value came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stale {
    /// The layer whose version of the input reproduces the output ([`EMBED`]: x as the embedding wrote it).
    pub from: u8,
}

/// One launch's output, judged.
#[derive(Clone, Debug, PartialEq)]
pub struct StageCheck {
    /// Layer (0-based), [`EMBED`] or [`FINAL`].
    pub layer: u8,
    /// Launch within the layer (0..16); for [`FINAL`]: 0 = LayerNorm, 1 = logits.
    pub launch: u8,
    /// Largest |board − model| against what the launch should have written (the whole token's trace); infinite if a
    /// value is not finite.
    pub dev: f32,
    pub ok: bool,
    /// The same against the launch's model fed with the board's own inputs; `None` when an input is no longer in
    /// memory (it was rewritten later in the token).
    pub local: Option<(f32, bool)>,
    /// The output is what the launch gives with an older version of an input.
    pub stale: Option<Stale>,
}

impl StageCheck {
    pub fn name(&self) -> &'static str {
        stage_name(self.layer, self.launch)
    }

    pub fn code(&self) -> u32 {
        (self.layer as u32) << 8 | self.launch as u32
    }

    fn flags(&self) -> u32 {
        let mut f = self.ok as u32;
        if let Some((_, ok)) = self.local {
            f |= 2 | (ok as u32) << 2;
        }
        if let Some(s) = self.stale {
            f |= 8 | (s.from as u32) << 8;
        }
        f
    }

    /// One line for the screen / the chain report.
    pub fn line(&self) -> String {
        let place = match self.layer {
            EMBED => String::from("inainte de straturi"),
            FINAL => String::from("dupa straturi"),
            l => format!("stratul {l}, lansarea {}", self.launch),
        };
        let local = match self.local {
            Some((d, true)) => format!("; fata de intrarile placii {d:e} (urmeaza din ele)"),
            Some((d, false)) => format!("; fata de intrarile placii {d:e} (NU urmeaza din ele)"),
            None => String::new(),
        };
        let stale = match self.stale {
            Some(Stale { from: EMBED }) => {
                String::from("; iese exact cu valoarea VECHE a lui x (cea scrisa de embedding)")
            }
            Some(Stale { from }) => {
                format!("; iese exact cu valorile VECHI ale intrarii (cele din stratul {from})")
            }
            None => String::new(),
        };
        format!(
            "{} ({place}): abaterea fata de model {:e}{}{local}{stale}",
            self.name(),
            self.dev,
            if self.ok { "" } else { " - IN AFARA" }
        )
    }
}

/// What a run says.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// A whole token: the logits pass against the CPU engine and the greedy token is the wanted one (what the step
    /// itself asks of a token). A prefix: every stage in memory is inside the stage tolerance.
    Curat,
    /// `stages[i]` — the first stage outside the tolerance that proves something — is what its launch gives with an
    /// **older** value of an input: the launch read stale data.
    DateVechi(usize),
    /// `stages[i]` — the first stage outside the tolerance that proves something — does not follow from the inputs
    /// now in memory and matches no older value: the launch itself (program, arguments, shape) is wrong on the board.
    LansareGresita(usize),
    /// The first stage outside the tolerance (`stages[i]`) follows from its inputs (or they are no longer in memory)
    /// and no later stage proves anything: the error entered before it, in a launch this run no longer shows
    /// ([`locate`] finds it).
    Mostenit(usize),
    /// Every stage in memory is fine but the logits do not pass (only for logits the caller read elsewhere, e.g.
    /// from RAM after the copy down).
    DoarLogiti,
    /// A submission did not finish or memory could not be read: nothing more was sent.
    Neterminat(GpuError),
}

impl Verdict {
    pub fn code(&self) -> u32 {
        match self {
            Verdict::Curat => 1,
            Verdict::DateVechi(_) => 2,
            Verdict::LansareGresita(_) => 3,
            Verdict::Mostenit(_) => 4,
            Verdict::Neterminat(_) => 5,
            Verdict::DoarLogiti => 6,
        }
    }

    pub fn stage(&self) -> Option<usize> {
        match self {
            Verdict::DateVechi(i) | Verdict::LansareGresita(i) | Verdict::Mostenit(i) => Some(*i),
            _ => None,
        }
    }
}

/// One run read back and judged.
#[derive(Clone, Debug, PartialEq)]
pub struct Probe {
    /// How it was sent; `None` = nothing was sent by the diagnosis (memory as a token of the caller left it).
    pub way: Option<Way>,
    /// Launches of the token's list that ran.
    pub done: usize,
    pub token: usize,
    pub pos: usize,
    /// Every stage whose output is in memory, in launch order.
    pub stages: Vec<StageCheck>,
    /// The logits (whole tokens only).
    pub logits: Option<LogitReport>,
    /// Logits that differ between what the caller read from RAM and VRAM ([`explain`] only).
    pub ram_diff: Option<usize>,
    pub verdict: Verdict,
}

impl Probe {
    pub fn passes(&self) -> bool {
        self.verdict == Verdict::Curat
    }

    /// The first stage outside the tolerance, if any.
    pub fn first_bad(&self) -> Option<&StageCheck> {
        self.verdict.stage().and_then(|i| self.stages.get(i))
    }

    /// The verdict in one sentence (Romanian, for the screen, the report and the journal's summary row).
    pub fn summary(&self) -> String {
        let how = match self.way {
            Some(w) => format!("{} lansari, {}", self.done, w.name()),
            None => String::from("simbolul rulat"),
        };
        let what = match &self.verdict {
            Verdict::Curat => String::from("TRECE"),
            Verdict::DateVechi(_) => String::from("DATE VECHI"),
            Verdict::LansareGresita(_) => String::from("LANSARE GRESITA"),
            Verdict::Mostenit(_) => String::from("EROARE MOSTENITA"),
            Verdict::DoarLogiti => String::from("ETAPELE BUNE, LOGIT-URILE CITITE NU"),
            Verdict::Neterminat(e) => format!("PLACA NU A TERMINAT ({e})"),
        };
        match self.first_bad() {
            Some(s) => format!("{how}: {what}: {}", s.line()),
            None => format!("{how}: {what}"),
        }
    }

    /// Lines for the screen / the chain report: the summary, the logits, every stage outside the tolerance.
    pub fn lines(&self) -> Vec<String> {
        let mut out = alloc::vec![format!("i14 diagnostic: {}", self.summary())];
        if let Some(l) = &self.logits {
            out.push(l.line());
        }
        if let Some(d) = self.ram_diff {
            out.push(format!(
                "i14 diagnostic: {d} logit-uri difera intre RAM (dupa copierea in jos) si VRAM"
            ));
        }
        for s in self.stages.iter().filter(|s| !s.ok) {
            out.push(format!("i14 diagnostic: {}", s.line()));
        }
        out
    }

    /// The rows of this run: the header, the logits (whole tokens; all the detail only for a run that does not
    /// pass), then every stage outside the tolerance with its numbers. A clean run costs about a dozen rows.
    pub fn journal(&self, j: &mut Journal) {
        use rows as n;
        val(j, n::PROBE, self.way.map_or(0, |w| w.mode as u32));
        val(j, n::EMBED_BY, self.way.map_or(0, |w| w.embed as u32));
        val(j, n::LAUNCHES, self.done as u32);
        val(j, n::VERDICT, self.verdict.code());
        match &self.logits {
            Some(l) if !self.passes() => l.journal(j),
            Some(l) => {
                val(j, n::TOKEN, l.token as u32);
                val(j, n::POS, l.pos as u32);
                val(j, n::ARGMAX_BOARD, opt(l.board));
                val(j, n::ARGMAX_CPU, opt(l.cpu));
                val(j, n::ARGMAX_WANT, opt(l.want));
                val(j, n::MAX_ABS, l.max_abs.to_bits());
            }
            None => {
                val(j, n::TOKEN, self.token as u32);
                val(j, n::POS, self.pos as u32);
            }
        }
        if let Some(d) = self.ram_diff {
            val(j, n::RAM_DIFF, d as u32);
        }
        let bad = self.stages.iter().filter(|s| !s.ok).count();
        val(j, n::STAGES_OK, (self.stages.len() - bad) as u32);
        val(j, n::STAGES_BAD, bad as u32);
        // Also for a token that passes: a stage outside the stage tolerance is worth seeing (it does not fail it).
        {
            for s in self.stages.iter().filter(|s| !s.ok) {
                val(j, n::STAGE, s.code());
                j.push(Event::text(n::STAGE_NAME, s.name().as_bytes()));
                val(j, n::STAGE_DEV, s.dev.to_bits());
                val(j, n::STAGE_FLAGS, s.flags());
                if let Some((d, _)) = s.local {
                    val(j, n::STAGE_LOCAL, d.to_bits());
                }
            }
        }
        j.push_text(n::SUMMARY, n::SUMMARY_MORE, &self.summary());
    }
}

/// What a diagnosis needs to know about the token.
pub struct ProbeCtx<'a> {
    pub w: &'a ModelWeights<'a>,
    pub vas: &'a ModelVas,
    /// The caches of the positions **before** `pos` (the CPU engine's; rows from `pos` on are not read).
    pub prev: &'a Caches,
    pub token: usize,
    pub pos: usize,
    /// The CPU engine's logits for this (token, position) (`gpu_compute::model::step_cpu`).
    pub cpu_logits: &'a [f32],
    /// Kronos's greedy token for this position (`vectori.json`), for a prompt position of a control text.
    pub want: Option<usize>,
}

/// What was read back after `done` launches of the token's list.
#[derive(Clone, Debug, PartialEq)]
pub struct Board {
    /// The activation scratch (`kronos_layer::scratch_bytes(E) / 4` values from `x`).
    pub scratch: Vec<f32>,
    /// For every layer whose cache copies ran: the new key row and value row at `pos`.
    pub kv: Vec<(Vec<f32>, Vec<f32>)>,
    /// The final LayerNorm's output and the logits (empty when their launches did not run).
    pub h_final: Vec<f32>,
    pub logits: Vec<f32>,
}

fn floats(w: &[u32]) -> Vec<f32> {
    w.iter().map(|&b| f32::from_bits(b)).collect()
}

/// Launches after the embedding that ran in layers, split: `(layers complete, launches into the next one, final
/// launches)`.
fn split(done: usize, layers: usize) -> (usize, usize, usize) {
    let q = done.saturating_sub(1);
    let in_layers = q.min(LAUNCHES_PER_TOKEN * layers);
    (
        in_layers / LAUNCHES_PER_TOKEN,
        in_layers % LAUNCHES_PER_TOKEN,
        q - in_layers,
    )
}

/// Layers with a new cache row after `done` launches (the key copy, launch 3, ran; the value copy is launch 4).
fn layers_with_cache_rows(done: usize, layers: usize) -> usize {
    let (full, rem, _) = split(done, layers);
    full + (rem > 3) as usize
}

/// Reads back what `done` launches left ([`Board`]).
pub fn read_board<I: TokenIo + ?Sized>(
    io: &mut I,
    c: &ProbeCtx<'_>,
    done: usize,
) -> GpuResult<Board> {
    let e = c.w.shape.e;
    let layers = c.w.layers.len();
    c.vas.check()?;
    if c.vas.layers.len() != layers {
        return Err(bad("diagnostic: adrese pentru alt numar de straturi"));
    }
    let n = km::launches_per_token(layers);
    let row = 4 * (c.pos * e) as u64;
    let mut ranges = alloc::vec![(c.vas.layers[0].x, (kl::scratch_bytes(e) / 4) as usize)];
    let with_rows = layers_with_cache_rows(done, layers);
    for l in &c.vas.layers[..with_rows] {
        ranges.push((l.kc + row, e));
        ranges.push((l.vc + row, e));
    }
    if done + 1 >= n {
        ranges.push((c.vas.h_final, e));
    }
    if done >= n {
        ranges.push((c.vas.logits, c.w.vocab));
    }
    let got = io.read(&ranges)?;
    if got.len() != ranges.len() || got.iter().zip(&ranges).any(|(g, r)| g.len() != r.1) {
        return Err(bad("diagnostic: citirea a intors alt numar de cuvinte"));
    }
    let mut it = got.iter().map(|w| floats(w));
    let scratch = it.next().unwrap_or_default();
    let kv = (0..with_rows)
        .map(|_| (it.next().unwrap_or_default(), it.next().unwrap_or_default()))
        .collect();
    let h_final = if done + 1 >= n {
        it.next().unwrap_or_default()
    } else {
        Vec::new()
    };
    let logits = if done >= n {
        it.next().unwrap_or_default()
    } else {
        Vec::new()
    };
    Ok(Board {
        scratch,
        kv,
        h_final,
        logits,
    })
}

/// Largest |got − want| and whether every element is inside `abs + rel·|want|`; a missing or non-finite value is
/// never inside (and makes the deviation infinite).
fn deviation(got: &[f32], want: &[f32], abs: f32, rel: f32) -> (f32, bool) {
    if got.len() != want.len() {
        return (f32::INFINITY, false);
    }
    let mut max = 0.0f32;
    let mut ok = true;
    for (&g, &w) in got.iter().zip(want) {
        if !g.is_finite() || !w.is_finite() {
            return (f32::INFINITY, false);
        }
        let d = (g - w).abs();
        max = max.max(d);
        ok &= d <= abs + rel * w.abs();
    }
    (max, ok)
}

fn act(got: &[f32], want: &[f32]) -> (f32, bool) {
    deviation(got, want, TOL_ABS, TOL_REL)
}

fn matvec(w: &[u8], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
    let mut y = alloc::vec![0.0f32; rows];
    // The shapes were checked with the model's weights; a mismatch leaves NaN (never inside a tolerance).
    if q8::matvec_gpu_order(w, rows, cols, x, &mut y).is_err() {
        y.fill(f32::NAN);
    }
    y
}

/// The 14 vectors of the scratch as the board has them, by the layer's own addresses.
struct Scratch<'a> {
    e: usize,
    words: &'a [f32],
    v: &'a LayerVas,
}

impl<'a> Scratch<'a> {
    fn at(&self, va: u64, units: usize) -> &'a [f32] {
        // An address before x or past the scratch gives nothing (never a panic: this runs in the kernel).
        let range = va
            .checked_sub(self.v.x)
            .and_then(|off| usize::try_from(off / 4).ok())
            .and_then(|start| Some(start..start.checked_add(units.checked_mul(self.e)?)?));
        range.and_then(|r| self.words.get(r)).unwrap_or(&[])
    }
    /// The output of launch `s` (cache rows are not in the scratch).
    fn out(&self, s: usize) -> &'a [f32] {
        let v = self.v;
        match s {
            0 => self.at(v.h, 1),
            1 => self.at(v.qkv, 3),
            2 => self.at(v.qkvb, 3),
            5 => self.at(v.att, 1),
            6 => self.at(v.o, 1),
            7 => self.at(v.ob, 1),
            8 => self.at(v.x1, 1),
            9 => self.at(v.h2, 1),
            10 => self.at(v.f, 4),
            11 => self.at(v.fb, 4),
            12 => self.at(v.g, 4),
            13 => self.at(v.p, 1),
            14 => self.at(v.pb, 1),
            15 => self.at(v.x, 1),
            _ => &[],
        }
    }
}

/// What launch `s` of a layer should write (the trace's vector; for 3 and 4 the key / value part of `qkvb`).
fn expected(t: &LayerTrace, e: usize, s: usize) -> &[f32] {
    match s {
        0 => &t.h,
        1 => &t.qkv,
        2 => &t.qkvb,
        3 => &t.qkvb[e..2 * e],
        4 => &t.qkvb[2 * e..],
        5 => &t.att,
        6 => &t.o,
        7 => &t.ob,
        8 => &t.x1,
        9 => &t.h2,
        10 => &t.f,
        11 => &t.fb,
        12 => &t.g,
        13 => &t.p,
        14 => &t.pb,
        _ => &t.x_out,
    }
}

/// Launch `s` of a layer as its program model, on explicit inputs: `a` is the launch's (first) scratch input, `b`
/// the second one of the two additions that read two scratch vectors (8: x + ob, 15: x1 + pb); `kc`/`vc` are the
/// cache rows `0..=pos` for the attention.
#[allow(clippy::too_many_arguments)]
fn model_of(
    s: usize,
    sh: &Shape,
    w: &LayerWeights<'_>,
    a: &[f32],
    b: &[f32],
    kc: &[f32],
    vc: &[f32],
    pos: usize,
) -> Vec<f32> {
    let e = sh.e;
    match s {
        0 => layer::layernorm_gpu(a, w.ln1_g, w.ln1_b, sh.eps),
        1 => matvec(w.wqkv, 3 * e, e, a),
        2 => layer::vadd_gpu(a, w.bqkv),
        3 => a.get(e..2 * e).unwrap_or(&[]).to_vec(),
        4 => a.get(2 * e..3 * e).unwrap_or(&[]).to_vec(),
        5 if a.len() >= e && kc.len() >= (pos + 1) * e && vc.len() >= (pos + 1) * e => {
            layer::attention_gpu(sh, a, kc, vc, pos + 1)
        }
        // A cache row that was not read back: nothing to compare with.
        5 => alloc::vec![f32::NAN; e],
        6 => matvec(w.wout, e, e, a),
        7 => layer::vadd_gpu(a, w.bout),
        8 | 15 => layer::vadd_gpu(a, b),
        9 => layer::layernorm_gpu(a, w.ln2_g, w.ln2_b, sh.eps),
        10 => matvec(w.wfc, 4 * e, e, a),
        11 => layer::vadd_gpu(a, w.bfc),
        12 => a.iter().map(|&v| layer::gelu_gpu(v)).collect(),
        13 => matvec(w.wproj, e, 4 * e, a),
        _ => layer::vadd_gpu(a, w.bproj),
    }
}

/// The scratch input(s) of launch `s` in a layer's trace: `(first, second)`.
fn inputs_of(t: &LayerTrace, s: usize) -> (&[f32], &[f32]) {
    match s {
        0 => (&t.x_in, &[]),
        1 => (&t.h, &[]),
        2 => (&t.qkv, &[]),
        3..=5 => (&t.qkvb, &[]),
        6 => (&t.att, &[]),
        7 => (&t.o, &[]),
        8 => (&t.x_in, &t.ob),
        9 => (&t.x1, &[]),
        10 => (&t.h2, &[]),
        11 => (&t.f, &[]),
        12 => (&t.fb, &[]),
        13 => (&t.g, &[]),
        14 => (&t.p, &[]),
        _ => (&t.x1, &t.pb),
    }
}

/// Judges what is in memory after `done` launches of the token (pure: nothing is sent or read here).
pub fn examine(c: &ProbeCtx<'_>, way: Option<Way>, done: usize, b: &Board) -> Probe {
    let fail = |err: GpuError| Probe {
        way,
        done,
        token: c.token,
        pos: c.pos,
        stages: Vec::new(),
        logits: None,
        ram_diff: None,
        verdict: Verdict::Neterminat(err),
    };
    let sh = &c.w.shape;
    let (e, layers) = (sh.e, c.w.layers.len());
    let n = km::launches_per_token(layers);
    if done == 0 || done > n || c.vas.layers.len() != layers {
        return fail(bad("diagnostic: numar de lansari sau de straturi gresit"));
    }
    let trace: TokenTrace = match model::trace_gpu(c.w, c.prev, c.token, c.pos) {
        Ok(t) => t,
        Err(err) => return fail(err),
    };
    if b.scratch.len() != (kl::scratch_bytes(e) / 4) as usize {
        return fail(bad("diagnostic: spatiul de lucru citit are alta marime"));
    }
    let (full, rem, fin) = split(done, layers);
    let mut stages: Vec<StageCheck> = Vec::new();
    let mut push = |layer: u8, launch: usize, path: (f32, bool), local, stale| {
        stages.push(StageCheck {
            layer,
            launch: launch as u8,
            dev: path.0,
            ok: path.1,
            local,
            stale,
        })
    };
    // Every value x held before layer `l` read it: the embedding's, then each layer's output.
    let x_before = |l: usize| -> Vec<(u8, &[f32])> {
        let mut v: Vec<(u8, &[f32])> = alloc::vec![(EMBED, &trace.x0[..])];
        v.extend((0..l).map(|j| (j as u8, &trace.layers[j].x_out[..])));
        v
    };
    // The cache rows 0..=pos of a layer as the board has them: the CPU's rows before `pos`, the board's at `pos`.
    let rows_of = |l: usize, k: bool| -> Vec<f32> {
        let (prev, new) = if k {
            (&c.prev.k[l], b.kv.get(l).map(|r| &r.0))
        } else {
            (&c.prev.v[l], b.kv.get(l).map(|r| &r.1))
        };
        let mut v = prev.get(..c.pos * e).unwrap_or(&[]).to_vec();
        v.extend_from_slice(new.map_or(&[][..], |r| &r[..]));
        v
    };

    // The embedding alone (x still holds it), or the layer in progress with x = its input still in place.
    let (vis, m, x_intact) = if done == 1 {
        (None, 0, true)
    } else if rem > 0 {
        (Some(full), rem, true)
    } else {
        (Some(full - 1), LAUNCHES_PER_TOKEN, false)
    };
    let v0 = &c.vas.layers[vis.unwrap_or(0)];
    let sc = Scratch {
        e,
        words: &b.scratch,
        v: v0,
    };
    let x_board = sc.at(v0.x, 1);
    // x still holds what the embedding wrote (nothing after it, or layer 0 not finished). Its inputs are two rows
    // of weights, never rewritten: what it should write IS what follows from the board's inputs.
    if x_intact && vis.is_none_or(|l| l == 0) {
        let path = act(x_board, &trace.x0);
        push(EMBED, 0, path, Some(path), None);
    }
    // Layers before the visible one: only their cache rows are still in memory.
    for j in 0..vis.unwrap_or(0) {
        if let Some((k, v)) = b.kv.get(j) {
            let t = &trace.layers[j];
            for (s, row) in [(3, k), (4, v)] {
                let path = act(row, expected(t, e, s));
                // The row an earlier layer's `qkvb` would have given (the copy read stale data)?
                let stale = (!path.1)
                    .then(|| {
                        (0..j)
                            .rev()
                            .find(|&i| act(row, expected(&trace.layers[i], e, s)).1)
                    })
                    .flatten()
                    .map(|i| Stale { from: i as u8 });
                push(j as u8, s, path, None, stale);
            }
        }
    }
    // x still holds the output of the layer before the visible one (its last launch).
    if let Some(l) = vis.filter(|&l| x_intact && l > 0) {
        let path = act(x_board, &trace.layers[l - 1].x_out);
        push((l - 1) as u8, 15, path, None, None);
    }
    if let Some(l) = vis {
        let (t, w) = (&trace.layers[l], &c.w.layers[l]);
        let (kc, vc) = (rows_of(l, true), rows_of(l, false));
        for s in 0..m {
            let got: &[f32] = match s {
                3 => b.kv.get(l).map_or(&[][..], |r| &r.0),
                4 => b.kv.get(l).map_or(&[][..], |r| &r.1),
                _ => sc.out(s),
            };
            let path = act(got, expected(t, e, s));
            // The board's own inputs of this launch, where they are still in memory.
            let have_x = x_intact;
            let board_in: Option<(&[f32], &[f32])> = match s {
                0 => have_x.then_some((x_board, &[][..])),
                1 => Some((sc.out(0), &[])),
                2 => Some((sc.out(1), &[])),
                3..=5 => Some((sc.out(2), &[])),
                6 => Some((sc.out(5), &[])),
                7 => Some((sc.out(6), &[])),
                8 => have_x.then_some((x_board, sc.out(7))),
                9 => Some((sc.out(8), &[])),
                10 => Some((sc.out(9), &[])),
                11 => Some((sc.out(10), &[])),
                12 => Some((sc.out(11), &[])),
                13 => Some((sc.out(12), &[])),
                14 => Some((sc.out(13), &[])),
                _ => Some((sc.out(8), sc.out(14))),
            };
            let local =
                board_in.map(|(a, b2)| act(got, &model_of(s, sh, w, a, b2, &kc, &vc, c.pos)));
            // Older versions of the inputs: looked for only when the stage is outside the tolerance AND does not
            // follow from the inputs now in memory (an output that follows from them read nothing stale; and at
            // position 0 the attention does not depend on its query at all, so an old query would "match" too).
            let mut stale = None;
            if !path.1 && !matches!(local, Some((_, true))) {
                let (cur_a, cur_b) = inputs_of(t, s);
                let two = matches!(s, 8 | 15);
                // Versions of the first input, newest first; for x: every value it held before this layer.
                let olds_a: Vec<(u8, &[f32])> = if matches!(s, 0 | 8) {
                    let mut v = x_before(l);
                    v.pop(); // the last one is the current input
                    v
                } else {
                    (0..l)
                        .map(|j| (j as u8, inputs_of(&trace.layers[j], s).0))
                        .collect()
                };
                let olds_b: Vec<(u8, &[f32])> = if two {
                    (0..l)
                        .map(|j| (j as u8, inputs_of(&trace.layers[j], s).1))
                        .collect()
                } else {
                    Vec::new()
                };
                let mut tries: Vec<(u8, &[f32], &[f32])> = Vec::new();
                for &(from, a) in olds_a.iter().rev() {
                    tries.push((from, a, cur_b));
                    for &(_, b2) in olds_b.iter().rev() {
                        tries.push((from, a, b2));
                    }
                }
                for &(from, b2) in olds_b.iter().rev() {
                    tries.push((from, cur_a, b2));
                }
                stale = tries
                    .into_iter()
                    .find(|&(_, a, b2)| act(got, &model_of(s, sh, w, a, b2, &kc, &vc, c.pos)).1)
                    .map(|(from, _, _)| Stale { from });
            }
            push(l as u8, s, path, local, stale);
        }
    }
    // The final LayerNorm and the logits.
    if fin >= 1 {
        let path = act(&b.h_final, &trace.h_final);
        let local = act(
            &b.h_final,
            &layer::layernorm_gpu(x_board, c.w.ln_g, c.w.ln_b, sh.eps),
        );
        let stale = (!path.1 && !local.1)
            .then(|| {
                let mut olds = x_before(layers);
                olds.pop();
                olds.into_iter().rev().find(|&(_, x)| {
                    act(
                        &b.h_final,
                        &layer::layernorm_gpu(x, c.w.ln_g, c.w.ln_b, sh.eps),
                    )
                    .1
                })
            })
            .flatten()
            .map(|(from, _)| Stale { from });
        push(FINAL, 0, path, Some(local), stale);
    }
    let mut logits = None;
    if fin >= 2 {
        let path = deviation(&b.logits, &trace.logits, LOGIT_ABS, LOGIT_REL);
        let local = deviation(
            &b.logits,
            &matvec(c.w.emb_tok, c.w.vocab, e, &b.h_final),
            LOGIT_ABS,
            LOGIT_REL,
        );
        push(FINAL, 1, path, Some(local), None);
        logits = Some(logit_report(
            c.token,
            c.pos,
            &b.logits,
            c.cpu_logits,
            c.want,
        ));
    }
    // A whole token is judged by its logits, as the step itself judges it (`kronos_model::judge` + the wanted
    // greedy token): a stage a little outside the stage tolerance does not fail a token whose logits pass. Without
    // logits (a prefix) the stages decide.
    let first_bad = stages.iter().position(|s| !s.ok);
    let verdict = match (&logits, first_bad) {
        (Some(l), _) if l.passes => Verdict::Curat,
        (_, Some(i)) => {
            // The first stage outside the tolerance that PROVES something: its output is what an older input gives
            // (a stale read), or it does not follow from the inputs now in memory (the launch itself). The stages
            // outside before it only passed a wrong input on (or their inputs are no longer in memory).
            let proven = stages.iter().enumerate().skip(i).find_map(|(k, s)| {
                if s.ok {
                    None
                } else if s.stale.is_some() {
                    Some(Verdict::DateVechi(k))
                } else if matches!(s.local, Some((_, false))) {
                    Some(Verdict::LansareGresita(k))
                } else {
                    None
                }
            });
            proven.unwrap_or(Verdict::Mostenit(i))
        }
        (Some(_), None) => Verdict::DoarLogiti,
        (None, None) => Verdict::Curat,
    };
    Probe {
        way,
        done,
        token: c.token,
        pos: c.pos,
        stages,
        logits,
        ram_diff: None,
        verdict,
    }
}

/// Sends the first `done` launches of the token in `way` (the embedding written by the driver first, if `way` says
/// so), reads memory back and judges it.
pub fn probe<I: TokenIo + ?Sized>(io: &mut I, c: &ProbeCtx<'_>, way: Way, done: usize) -> Probe {
    let fail = |err: GpuError| Probe {
        way: Some(way),
        done,
        token: c.token,
        pos: c.pos,
        stages: Vec::new(),
        logits: None,
        ram_diff: None,
        verdict: Verdict::Neterminat(err),
    };
    let list = match km::token_launches(&c.w.shape, c.w.vocab, c.w.context, c.vas, c.token, c.pos) {
        Ok(l) => l,
        Err(err) => return fail(err),
    };
    if done == 0 || done > list.len() {
        return fail(bad("diagnostic: numar de lansari in afara listei"));
    }
    // With the driver's embedding at least one launch must follow it: the write may be queued until a submission.
    if way.embed == EmbedBy::Driver && done < 2 {
        return fail(bad(
            "diagnostic: dupa embedding-ul scris de driver trebuie o lansare",
        ));
    }
    if let Err(err) = send(io, c.w, c.vas, &list[..done], c.token, c.pos, way) {
        return fail(err);
    }
    match read_board(io, c, done) {
        Ok(b) => examine(c, Some(way), done, &b),
        Err(err) => fail(err),
    }
}

/// What [`explain`] found.
#[derive(Clone, Debug, PartialEq)]
pub struct Explained {
    pub probe: Probe,
    /// The logits as they are in VRAM (what the last launch of the model wrote), when memory could be read.
    pub vram_logits: Option<Vec<f32>>,
    /// The logits in VRAM pass on their own (the CPU judgement and the wanted greedy token) while the ones the
    /// caller read do not: only the copy into RAM is wrong. The caller may go on with `vram_logits` (reading the
    /// logits through the window from then on) so the rest of the text is still judged — and must report the copy
    /// as wrong.
    pub vram_pass: bool,
}

/// After a token of the caller's own run did not pass: memory **as that token left it** is read back and judged
/// (nothing is sent), next to the logits the caller read (`ram_logits`, e.g. from RAM after the copy down). The
/// rows go to the journal ([`TokenIo::journal`]); the probe is returned for the screen.
pub fn explain<I: TokenIo + ?Sized>(io: &mut I, c: &ProbeCtx<'_>, ram_logits: &[f32]) -> Explained {
    let n = km::launches_per_token(c.w.layers.len());
    let (mut p, vram) = match read_board(io, c, n) {
        Ok(b) => {
            // Judged with the logits in VRAM first: `Curat` here = the model's own launches are all right.
            let mut p = examine(c, None, n, &b);
            p.ram_diff = Some(
                (0..b.logits.len().max(ram_logits.len()))
                    .filter(|&i| {
                        b.logits.get(i).map(|v| v.to_bits())
                            != ram_logits.get(i).map(|v| v.to_bits())
                    })
                    .count(),
            );
            (p, Some(b.logits))
        }
        Err(err) => (
            Probe {
                way: None,
                done: n,
                token: c.token,
                pos: c.pos,
                stages: Vec::new(),
                logits: None,
                ram_diff: None,
                verdict: Verdict::Neterminat(err),
            },
            None,
        ),
    };
    // What the caller judged: the logits it read.
    let ram = logit_report(c.token, c.pos, ram_logits, c.cpu_logits, c.want);
    let vram_pass = p.verdict == Verdict::Curat && !ram.passes;
    if vram_pass {
        p.verdict = Verdict::DoarLogiti;
    }
    p.logits = Some(ram);
    p.journal(io.journal());
    Explained {
        probe: p,
        vram_logits: vram,
        vram_pass,
    }
}

/// The answer of [`ladder`].
#[derive(Clone, Debug, PartialEq)]
pub struct Ladder {
    pub probes: Vec<Probe>,
    /// How the texts should be sent: the first way whose whole token passed (never [`Way::OS_305`], which is only
    /// the control). With `embed == EmbedBy::Driver` the program `embed` is wrong on the board: the texts are worth
    /// running for what they tell, but the caller's step has not passed.
    pub chosen: Option<Way>,
}

impl Ladder {
    pub fn lines(&self) -> Vec<String> {
        let mut out: Vec<String> = self.probes.iter().flat_map(|p| p.lines()).collect();
        out.push(match self.chosen {
            Some(w) => format!("i14 diagnostic: textele ruleaza cu: {}", w.name()),
            None => String::from(
                "i14 diagnostic: niciun mod nu trece; etapa de mai sus e cea de reparat (nimic nu mai e trimis)",
            ),
        });
        out
    }
}

/// How a probe of the ladder ended.
enum Ran {
    Clean,
    /// A submission did not finish (or memory could not be read): nothing more may be sent.
    Dead,
    /// Not clean; `embedding`: the first stage outside the tolerance is the embedding.
    Bad {
        embedding: bool,
    },
}

/// One probe of the ladder: sent, judged, journaled, kept.
fn run<I: TokenIo + ?Sized>(
    io: &mut I,
    c: &ProbeCtx<'_>,
    probes: &mut Vec<Probe>,
    way: Way,
    done: usize,
) -> Ran {
    let p = probe(io, c, way, done);
    p.journal(io.journal());
    let r = if p.passes() {
        Ran::Clean
    } else if matches!(p.verdict, Verdict::Neterminat(_)) {
        Ran::Dead
    } else {
        Ran::Bad {
            embedding: p.first_bad().is_some_and(|s| s.layer == EMBED),
        }
    };
    probes.push(p);
    r
}

/// Whole tokens in `ways`, in order, until one passes: that way. `Err` = a submission did not finish.
fn first_clean<I: TokenIo + ?Sized>(
    io: &mut I,
    c: &ProbeCtx<'_>,
    probes: &mut Vec<Probe>,
    ways: [Way; 2],
) -> Result<Option<Way>, ()> {
    let n = km::launches_per_token(c.w.layers.len());
    for way in ways {
        match run(io, c, probes, way, n) {
            Ran::Clean => return Ok(Some(way)),
            Ran::Dead => return Err(()),
            Ran::Bad { .. } => {}
        }
    }
    Ok(None)
}

/// Every layer up to its 15th launch, each prefix sent from the token's start in `way` (its input still in x, so
/// each of its launches is judged against its own inputs), until one is not clean.
fn layer_prefixes<I: TokenIo + ?Sized>(
    io: &mut I,
    c: &ProbeCtx<'_>,
    probes: &mut Vec<Probe>,
    way: Way,
) {
    for l in 0..c.w.layers.len() {
        let done = 1 + LAUNCHES_PER_TOKEN * l + LAUNCHES_PER_TOKEN - 1;
        if !matches!(run(io, c, probes, way, done), Ran::Clean) {
            break;
        }
    }
}

/// Which launch is wrong at this (token, position), when a whole token of a text did not pass and [`explain`] only
/// says the error was inherited (it entered in a layer whose vectors were overwritten since): prefixes of the
/// token's list, one submission per layer — the embedding alone (when the program writes it), then every layer up
/// to its 15th launch — until one is not clean; that probe names the launch. Every probe is journaled. The cache
/// rows of the earlier positions must still be in memory as those positions left them.
pub fn locate<I: TokenIo + ?Sized>(io: &mut I, c: &ProbeCtx<'_>, embed: EmbedBy) -> Vec<Probe> {
    let way = Way {
        mode: SeqMode::PerLayer,
        embed,
    };
    let mut probes = Vec::new();
    if embed == EmbedBy::Driver || matches!(run(io, c, &mut probes, way, 1), Ran::Clean) {
        layer_prefixes(io, c, &mut probes, way);
    }
    probes
}

/// The first token of a control text, sent several ways, before the texts run:
/// 1. [`Way::OS_305`] — exactly what failed in os-305 (a **control**: it tells whether that failure is still there
///    and at which stage; it does not decide anything);
/// 2. [`Way::REPAIR`] (`EachLaunch`) — if the whole token passes, it is chosen and nothing more is sent;
/// 3. `PerLayer` — only if 2 did not pass; chosen if it passes;
/// 4. only if neither passed: the embedding alone (one launch). If **it** is wrong, the token is tried again with
///    the embedding written by the driver (`EachLaunch`, then `PerLayer`) and the first that passes is chosen;
/// 5. still nothing: each layer up to its 15th launch (its input still in x), until one is not clean — that probe
///    names the launch.
///
/// A submission that does not finish stops everything at once. Every probe is journaled as it ends, then the choice.
pub fn ladder<I: TokenIo + ?Sized>(io: &mut I, c: &ProbeCtx<'_>) -> Ladder {
    let n = km::launches_per_token(c.w.layers.len());
    let mut probes: Vec<Probe> = Vec::new();
    let per_layer = Way::of(SeqMode::PerLayer);
    let chosen = 'all: {
        if matches!(run(io, c, &mut probes, Way::OS_305, n), Ran::Dead) {
            break 'all None;
        }
        match first_clean(io, c, &mut probes, [Way::REPAIR, per_layer]) {
            Ok(None) => {}
            Ok(way) => break 'all way,
            Err(()) => break 'all None,
        }
        // Nothing passes: which launch? First the embedding alone.
        let mut prefix_way = per_layer;
        match run(io, c, &mut probes, per_layer, 1) {
            Ran::Dead => break 'all None,
            Ran::Bad { embedding: true } => {
                let ways = [
                    Way::REPAIR.with_driver_embedding(),
                    per_layer.with_driver_embedding(),
                ];
                match first_clean(io, c, &mut probes, ways) {
                    Ok(None) => prefix_way = per_layer.with_driver_embedding(),
                    Ok(way) => break 'all way,
                    Err(()) => break 'all None,
                }
            }
            Ran::Clean | Ran::Bad { .. } => {}
        }
        layer_prefixes(io, c, &mut probes, prefix_way);
        None
    };
    let j = io.journal();
    val(j, rows::CHOSEN, chosen.map_or(0, |w| w.mode as u32));
    val(j, rows::CHOSEN_EMBED, chosen.map_or(0, |w| w.embed as u32));
    Ladder { probes, chosen }
}

// ------------------------------------------------------------------------------------------------ reading a journal

/// One probe as read back from a stick journal.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedProbe {
    pub mode: u32,
    /// 1 = the embedding was written by the driver.
    pub embed_by: u32,
    pub launches: u32,
    pub verdict: u32,
    pub token: Option<u32>,
    pub pos: Option<u32>,
    pub argmax_board: Option<u32>,
    pub argmax_cpu: Option<u32>,
    pub argmax_want: Option<u32>,
    pub not_finite: Option<u32>,
    pub fill: Option<u32>,
    pub logits_board: Vec<f32>,
    pub logits_cpu: Vec<f32>,
    pub max_abs: Option<f32>,
    pub max_at: Option<u32>,
    pub outside: Option<u32>,
    pub margin: Option<f32>,
    pub ram_diff: Option<u32>,
    pub stages_ok: Option<u32>,
    pub stages_bad: Option<u32>,
    /// `(code, deviation, flags, local deviation)` of every stage outside the tolerance.
    pub stages: Vec<(u32, f32, u32, Option<f32>)>,
    pub summary: String,
}

/// What [`parse_journal`] found.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ParsedDiag {
    pub probes: Vec<ParsedProbe>,
    /// The `CHOSEN` row, if the ladder ran to its end, and who writes the embedding for the texts.
    pub chosen: Option<u32>,
    pub chosen_embed: Option<u32>,
}

fn row<'a>(line: &'a str, kind: &str) -> Option<(&'a str, &'a str)> {
    let rest = &line[line.find(kind)? + kind.len()..];
    let (name, value) = rest.rsplit_once(" = ")?;
    Some((name, value.trim()))
}

/// A `text <name> = "<bytes>"` row: the name and the bytes between the quotes (the text itself may hold ` = `).
fn text_row(line: &str) -> Option<(&str, &str)> {
    let rest = &line[line.find("text ")? + 5..];
    let (name, value) = rest.split_once(" = \"")?;
    Some((name, value.strip_suffix('"').unwrap_or(value)))
}

/// Reads the `i14d` rows of a stick journal (`… valoare <name> = 0x…`, `… text <name> = "…"`) back into probes, in
/// order. Rows of other steps are skipped; a journal without `i14d` rows gives an empty result.
pub fn parse_journal(text: &str) -> ParsedDiag {
    use rows as n;
    let mut out = ParsedDiag::default();
    for line in text.lines() {
        if let Some((name, v)) = row(line, "valoare ") {
            let Some(v) = v
                .strip_prefix("0x")
                .and_then(|h| u32::from_str_radix(h, 16).ok())
            else {
                continue;
            };
            if name == n::PROBE {
                out.probes.push(ParsedProbe {
                    mode: v,
                    ..ParsedProbe::default()
                });
                continue;
            }
            if name == n::CHOSEN {
                out.chosen = Some(v);
                continue;
            }
            if name == n::CHOSEN_EMBED {
                out.chosen_embed = Some(v);
                continue;
            }
            let Some(p) = out.probes.last_mut() else {
                continue;
            };
            let some = |v: u32| (v != u32::MAX).then_some(v);
            let f = f32::from_bits(v);
            match name {
                x if x == n::EMBED_BY => p.embed_by = v,
                x if x == n::LAUNCHES => p.launches = v,
                x if x == n::VERDICT => p.verdict = v,
                x if x == n::TOKEN => p.token = Some(v),
                x if x == n::POS => p.pos = Some(v),
                x if x == n::ARGMAX_BOARD => p.argmax_board = some(v),
                x if x == n::ARGMAX_CPU => p.argmax_cpu = some(v),
                x if x == n::ARGMAX_WANT => p.argmax_want = some(v),
                x if x == n::NOT_FINITE => p.not_finite = Some(v),
                x if x == n::FILL => p.fill = Some(v),
                x if x == n::LOGIT_BOARD => p.logits_board.push(f),
                x if x == n::LOGIT_CPU => p.logits_cpu.push(f),
                x if x == n::MAX_ABS => p.max_abs = Some(f),
                x if x == n::MAX_AT => p.max_at = Some(v),
                x if x == n::OUTSIDE => p.outside = Some(v),
                x if x == n::MARGIN => p.margin = Some(f),
                x if x == n::RAM_DIFF => p.ram_diff = Some(v),
                x if x == n::STAGES_OK => p.stages_ok = Some(v),
                x if x == n::STAGES_BAD => p.stages_bad = Some(v),
                x if x == n::STAGE => p.stages.push((v, f32::NAN, 0, None)),
                x if x == n::STAGE_DEV => {
                    if let Some(s) = p.stages.last_mut() {
                        s.1 = f;
                    }
                }
                x if x == n::STAGE_FLAGS => {
                    if let Some(s) = p.stages.last_mut() {
                        s.2 = v;
                    }
                }
                x if x == n::STAGE_LOCAL => {
                    if let Some(s) = p.stages.last_mut() {
                        s.3 = Some(f);
                    }
                }
                _ => {}
            }
        } else if let Some((name, v)) = text_row(line)
            && (name == n::SUMMARY || name == n::SUMMARY_MORE)
            && let Some(p) = out.probes.last_mut()
        {
            p.summary.push_str(v);
        }
    }
    out
}

impl ParsedProbe {
    /// The probe in plain Romanian, for the person reading the journal.
    pub fn lines(&self) -> Vec<String> {
        let by_driver = |e: u32| {
            if e == EmbedBy::Driver as u32 {
                ", embedding scris de driver"
            } else {
                ""
            }
        };
        let how = match SeqMode::from_code(self.mode) {
            Some(m) => format!(
                "proba cu {}{} ({} lansari)",
                m.name(),
                by_driver(self.embed_by),
                self.launches
            ),
            None => String::from("simbolul rulat de pasul I14 (memoria citita dupa el)"),
        };
        let verdict = match self.verdict {
            1 => "TRECE",
            2 => "DATE VECHI (o lansare a citit valori dinainte de rescriere)",
            3 => "LANSARE GRESITA (iesirea nu urmeaza din intrarile din memorie)",
            4 => "EROARE MOSTENITA (a intrat inaintea primei etape vizibile)",
            5 => "PLACA NU A TERMINAT",
            6 => "etapele sunt bune, dar logit-urile citite (din RAM) nu trec",
            _ => "?",
        };
        let mut out = alloc::vec![format!("{how}: {verdict}")];
        let s = |v: Option<u32>| v.map_or(String::from("-"), |x| format!("{x}"));
        if self.argmax_cpu.is_some() || self.argmax_board.is_some() {
            out.push(format!(
                "  simbolul {} la pozitia {}: argmax placa {}, procesor {}, vectori {}",
                s(self.token),
                s(self.pos),
                s(self.argmax_board),
                s(self.argmax_cpu),
                s(self.argmax_want)
            ));
        }
        if let Some(d) = self.max_abs {
            out.push(format!(
                "  abaterea maxima a logit-urilor {d:e}{}{}{}",
                self.max_at
                    .map_or(String::new(), |i| format!(" la indexul {i}")),
                self.outside
                    .map_or(String::new(), |n| format!("; {n} in afara tolerantei")),
                self.margin.map_or(String::new(), |m| format!(
                    "; marja top1-top2 pe procesor {m:e}"
                )),
            ));
        }
        if let (Some(nf), Some(fill)) = (self.not_finite, self.fill)
            && (nf > 0 || fill > 0)
        {
            out.push(format!(
                "  {nf} logit-uri nefinite, {fill} ramase cu umplutura eeeeeeee (nescrise in RAM)"
            ));
        }
        if let Some(d) = self.ram_diff {
            out.push(format!(
                "  copierea logit-urilor in RAM: {d} valori diferite de cele din VRAM"
            ));
        }
        for (i, (b, c)) in self.logits_board.iter().zip(&self.logits_cpu).enumerate() {
            out.push(format!("  logit {i}: placa {b:e}, procesor {c:e}"));
        }
        if let (Some(ok), Some(bad)) = (self.stages_ok, self.stages_bad) {
            out.push(format!(
                "  etape citite inapoi: {ok} in toleranta, {bad} in afara"
            ));
        }
        for &(code, dev, flags, local) in &self.stages {
            let (layer, launch) = ((code >> 8) as u8, code as u8);
            let place = match layer {
                EMBED => String::from("inainte de straturi"),
                FINAL => String::from("dupa straturi"),
                l => format!("stratul {l}, lansarea {launch}"),
            };
            let mut t = format!(
                "  IN AFARA: {} ({place}): abaterea fata de model {dev:e}",
                stage_name(layer, launch)
            );
            if flags & 2 != 0 {
                t.push_str(&format!(
                    "; fata de intrarile placii {}{}",
                    local.map_or(String::from("?"), |d| format!("{d:e}")),
                    if flags & 4 != 0 {
                        " (urmeaza din ele)"
                    } else {
                        " (NU urmeaza din ele)"
                    }
                ));
            }
            if flags & 8 != 0 {
                match (flags >> 8) as u8 {
                    EMBED => t.push_str("; = rezultatul cu x VECHI (cel scris de embedding)"),
                    l => t.push_str(&format!(
                        "; = rezultatul cu intrarea VECHE (din stratul {l})"
                    )),
                }
            }
            out.push(t);
        }
        if !self.summary.is_empty() {
            out.push(format!("  rezumatul din jurnal: {}", self.summary));
        }
        out
    }
}

impl ParsedDiag {
    pub fn lines(&self) -> Vec<String> {
        let mut out: Vec<String> = self.probes.iter().flat_map(|p| p.lines()).collect();
        if let Some(c) = self.chosen {
            out.push(match SeqMode::from_code(c) {
                Some(m) if self.chosen_embed == Some(EmbedBy::Driver as u32) => format!(
                    "modul ales pentru texte: {}, embedding scris de driver (programul embed e gresit pe placa)",
                    m.name()
                ),
                Some(m) => format!("modul ales pentru texte: {}", m.name()),
                None => String::from("niciun mod nu a trecut: textele nu au mai rulat"),
            });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The submissions as (start, end) pairs.
    fn subs(mode: SeqMode, layers: usize, n: usize) -> Vec<(usize, usize)> {
        submissions(mode, layers, n)
            .into_iter()
            .map(|r| (r.start, r.end))
            .collect()
    }

    #[test]
    fn submissions_per_mode() {
        // The control model: 2 layers, 35 launches (+ the caller's copy of the logits = 36).
        assert_eq!(subs(SeqMode::Once, 2, 36), [(0, 36)]);
        assert_eq!(subs(SeqMode::EachLaunch, 2, 35), [(0, 35)]);
        assert_eq!(
            subs(SeqMode::PerLayer, 2, 36),
            [(0, 17), (17, 33), (33, 36)]
        );
        assert_eq!(
            subs(SeqMode::PerLayer, 2, 35),
            [(0, 17), (17, 33), (33, 35)]
        );
        // Prefixes: the embedding alone, a layer up to its 15th launch, the layers without the final launches.
        assert_eq!(subs(SeqMode::PerLayer, 2, 1), [(0, 1)]);
        assert_eq!(subs(SeqMode::PerLayer, 2, 16), [(0, 16)]);
        assert_eq!(subs(SeqMode::PerLayer, 2, 32), [(0, 17), (17, 32)]);
        assert_eq!(subs(SeqMode::PerLayer, 2, 33), [(0, 17), (17, 33)]);
        assert_eq!(subs(SeqMode::PerLayer, 12, 196).len(), 13);
        assert!(subs(SeqMode::Once, 2, 0).is_empty());
        // Every launch is sent exactly once, in order, in every mode.
        for mode in [SeqMode::Once, SeqMode::EachLaunch, SeqMode::PerLayer] {
            for n in 1..=40 {
                let s = subs(mode, 2, n);
                assert_eq!(s.first().unwrap().0, 0);
                assert_eq!(s.last().unwrap().1, n);
                assert!(s.windows(2).all(|w| w[0].1 == w[1].0));
                assert!(s.iter().all(|r| r.0 < r.1));
            }
        }
        assert_eq!(words(SeqMode::Once, 36), transfer::sequence_words(36));
        assert_eq!(words(SeqMode::PerLayer, 17), transfer::sequence_words(17));
        assert_eq!(
            words(SeqMode::EachLaunch, 36),
            transfer::sequence_words(36) + 72
        );
        assert_eq!(SeqMode::from_code(2), Some(SeqMode::EachLaunch));
        assert_eq!(SeqMode::from_code(0), None);
        // The pushbuffer stretch of one token with the copy of the logits: the control model and #513.
        assert_eq!(push_bytes(SeqMode::Once, 2, 36), 4 * (8 + 36 * 10));
        assert_eq!(push_bytes(SeqMode::EachLaunch, 2, 36), 1760);
        assert_eq!(push_bytes(SeqMode::PerLayer, 2, 36), 4 * (8 + 17 * 10));
        assert_eq!(
            push_bytes(SeqMode::Once, 12, 196),
            crate::kronos_gen::push_bytes_with_copy(12)
        );
        assert_eq!(push_bytes(SeqMode::EachLaunch, 12, 196), 9440);
        assert!(
            2 * push_bytes(SeqMode::EachLaunch, 12, 196) > 16 << 10,
            "16 KiB nu mai ajung"
        );
        assert!(2 * push_bytes(SeqMode::EachLaunch, 24, 388) <= 64 << 10);
        assert_eq!(push_bytes(SeqMode::PerLayer, 12, 196), 712);
        assert_eq!(push_bytes(SeqMode::Once, 2, 0), 0);
    }

    #[test]
    fn ways_and_the_driver_embedding() {
        let way = |w: Way, n: usize| -> Vec<(usize, usize)> {
            w.submissions(2, n)
                .into_iter()
                .map(|r| (r.start, r.end))
                .collect()
        };
        // With the program's embedding a way is its mode.
        for mode in [SeqMode::Once, SeqMode::EachLaunch, SeqMode::PerLayer] {
            for n in 0..=40 {
                assert_eq!(way(Way::of(mode), n), subs(mode, 2, n));
            }
        }
        assert_eq!(Way::OS_305, Way::of(SeqMode::Once));
        assert_eq!(Way::REPAIR, Way::of(SeqMode::EachLaunch));
        // With the driver's embedding launch 0 is not sent; the slots stay the list's indices.
        let d = Way::REPAIR.with_driver_embedding();
        assert_eq!(d.embed, EmbedBy::Driver);
        assert_eq!(way(d, 36), [(1, 36)]);
        assert_eq!(way(d, 1), [], "doar embedding-ul: nimic de trimis");
        assert_eq!(way(d, 0), []);
        let p = Way::of(SeqMode::PerLayer).with_driver_embedding();
        assert_eq!(way(p, 36), [(1, 17), (17, 33), (33, 36)]);
        assert_eq!(way(p, 16), [(1, 16)]);
        assert_eq!(way(p, 1), []);
        assert_eq!(
            d.name(),
            "invalidare inaintea fiecarei lansari, embedding scris de driver"
        );
        assert_eq!(Way::OS_305.name(), SeqMode::Once.name());

        // The driver's embedding is the program's arithmetic (and the CPU engine's x before layer 0).
        let m = km::test_model(
            Shape {
                e: 32,
                heads: 4,
                eps: 1e-5,
            },
            1,
            64,
            16,
            3,
        )
        .unwrap();
        let w = m.weights();
        let x = driver_embedding(&w, 5, 2).unwrap();
        assert_eq!(x.len(), 32);
        let t = model::trace_gpu(&w, &Caches::new(1, 32), 5, 0);
        assert!(t.is_ok());
        let at_2 = model::embed_gpu(
            &w.emb_tok[w.row_offset(5)..],
            &w.emb_pos[w.row_offset(2)..],
            32,
        );
        assert_eq!(x, at_2.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
        let at_0 = driver_embedding(&w, 5, 0).unwrap();
        assert_eq!(
            at_0,
            t.unwrap()
                .x0
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        assert_ne!(x, at_0, "pozitia conteaza");
        assert!(driver_embedding(&w, 64, 0).is_err());
        assert!(driver_embedding(&w, 0, 16).is_err());
    }

    #[test]
    fn what_ran_after_n_launches() {
        // 2 layers: 1 embedding + 32 + 2 final.
        assert_eq!(split(1, 2), (0, 0, 0));
        assert_eq!(split(16, 2), (0, 15, 0));
        assert_eq!(split(17, 2), (1, 0, 0));
        assert_eq!(split(32, 2), (1, 15, 0));
        assert_eq!(split(33, 2), (2, 0, 0));
        assert_eq!(split(34, 2), (2, 0, 1));
        assert_eq!(split(35, 2), (2, 0, 2));
        assert_eq!(layers_with_cache_rows(1, 2), 0);
        assert_eq!(layers_with_cache_rows(4, 2), 0, "copierea K e lansarea 3");
        assert_eq!(layers_with_cache_rows(5, 2), 1);
        assert_eq!(layers_with_cache_rows(6, 2), 1);
        assert_eq!(layers_with_cache_rows(16, 2), 1);
        assert_eq!(layers_with_cache_rows(22, 2), 2);
        assert_eq!(layers_with_cache_rows(35, 2), 2);
    }

    #[test]
    fn logit_reports() {
        let cpu = [0.0f32, 2.0, 1.0, -1.0];
        let r = logit_report(7, 3, &[0.0, 2.0005, 1.0, -1.0], &cpu, Some(1));
        assert!(r.passes);
        assert_eq!((r.board, r.cpu, r.outside), (Some(1), Some(1), 0));
        assert_eq!(r.cpu_margin, 1.0);
        assert!((r.max_abs - 0.0005).abs() < 1e-6 && r.max_at == 1);
        // The wanted token differs: not passing, even with the CPU's logits.
        assert!(!logit_report(7, 3, &cpu, &cpu, Some(2)).passes);
        // Never written: every value is the fill, finite, argmax 0.
        let fill = [f32::from_bits(RAM_FILL); 4];
        let r = logit_report(7, 3, &fill, &cpu, None);
        assert_eq!(
            (r.fill, r.not_finite, r.board, r.passes),
            (4, 0, Some(0), false)
        );
        assert!(r.line().contains("4 nescrise"));
        // NaN: no greedy token.
        let r = logit_report(7, 3, &[f32::NAN, 2.0, 1.0, -1.0], &cpu, None);
        assert_eq!((r.board, r.not_finite, r.outside), (None, 1, 1));
        assert_eq!(r.at_board_top, None);
        // Lengths that differ never pass.
        assert!(!logit_report(7, 3, &cpu[..3], &cpu, None).passes);
    }

    #[test]
    fn a_logit_report_survives_the_journal() {
        let cpu: Vec<f32> = (0..16).map(|i| (i as f32 * 0.37).sin()).collect();
        let mut board = cpu.clone();
        board[5] += 0.25;
        let r = logit_report(65, 0, &board, &cpu, Some(9));
        let p = Probe {
            way: Some(Way::REPAIR.with_driver_embedding()),
            done: 34,
            token: 65,
            pos: 0,
            stages: alloc::vec![
                StageCheck {
                    layer: 0,
                    launch: 3,
                    dev: 0.0,
                    ok: true,
                    local: None,
                    stale: None
                },
                StageCheck {
                    layer: 1,
                    launch: 1,
                    dev: 0.5,
                    ok: false,
                    local: Some((0.5, false)),
                    stale: Some(Stale { from: 0 })
                },
                StageCheck {
                    layer: FINAL,
                    launch: 0,
                    dev: 0.75,
                    ok: false,
                    local: Some((0.0, true)),
                    stale: Some(Stale { from: EMBED })
                },
            ],
            logits: Some(r.clone()),
            ram_diff: None,
            verdict: Verdict::DateVechi(1),
        };
        let mut j = Journal::new(256);
        p.journal(&mut j);
        val(&mut j, rows::CHOSEN, 2);
        val(&mut j, rows::CHOSEN_EMBED, 1);
        // As the OS prints the journal on the stick.
        let text: String = j
            .events()
            .enumerate()
            .map(|(i, e)| format!("[  127.{i:06}] gpu: {e}\n"))
            .collect();
        let d = parse_journal(&text);
        assert_eq!((d.chosen, d.chosen_embed), (Some(2), Some(1)));
        assert_eq!(d.probes.len(), 1);
        let q = &d.probes[0];
        assert_eq!((q.mode, q.embed_by, q.launches, q.verdict), (2, 1, 34, 2));
        assert_eq!((q.token, q.pos), (Some(65), Some(0)));
        assert_eq!(q.argmax_want, Some(9));
        assert_eq!(q.argmax_board, r.board.map(|v| v as u32));
        assert_eq!(q.logits_board, board[..8]);
        assert_eq!(q.logits_cpu, cpu[..8]);
        assert_eq!(q.max_abs, Some(r.max_abs));
        assert_eq!(q.max_at, Some(5));
        assert_eq!((q.stages_ok, q.stages_bad), (Some(1), Some(2)));
        assert_eq!(q.stages.len(), 2);
        assert_eq!(q.stages[0], (0x101, 0.5, 0b1010, Some(0.5)));
        assert_eq!(q.stages[1].0, 0xff00);
        assert_eq!(q.stages[1].2, 0xfe00 | 0b1110);
        assert_eq!(q.summary, p.summary());
        let lines = d.lines().join("\n");
        assert!(lines.contains("DATE VECHI"), "{lines}");
        assert!(lines.contains("stratul 1, lansarea 1"), "{lines}");
        assert!(lines.contains("intrarea VECHE (din stratul 0)"), "{lines}");
        assert!(lines.contains("x VECHI"), "{lines}");
        assert!(
            lines.contains(
                "proba cu invalidare inaintea fiecarei lansari, embedding scris de driver (34 lansari)"
            ),
            "{lines}"
        );
        assert!(
            lines.contains("modul ales pentru texte: invalidare inaintea fiecarei lansari, embedding scris de driver"),
            "{lines}"
        );
        // A journal of another step has nothing for us.
        assert_eq!(
            parse_journal("[ 1.0] gpu: valoare i9 c135 canal (handle) = 0xcf00000d\n"),
            ParsedDiag::default()
        );
    }

    #[test]
    fn stage_names_fit_a_journal_text() {
        for (i, n) in LAYER_STAGES.iter().enumerate() {
            assert!(n.len() <= gpu_core::diag::TEXT_LEN, "{i}: {n}");
            assert!(n.is_ascii());
        }
        for (l, s) in [(EMBED, 0), (FINAL, 0), (FINAL, 1)] {
            assert!(stage_name(l, s).len() <= gpu_core::diag::TEXT_LEN);
        }
        assert_eq!(stage_name(3, 5), "atentie");
        assert_eq!(stage_name(3, 99), "?");
    }
}
