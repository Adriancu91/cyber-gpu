//! C-87 — continuous generation on the work channel, and the VRAM a model needs before anything is uploaded.
//!
//! **Generation** ([`Generation`]): a prompt, then `n` greedy tokens, one token = one fenced submission of
//! [`crate::kronos_model::token_launches`] (the key/value caches grow by one position each time). The loop is the
//! same for the simulated GPU and the board: the caller gives a [`TokenRunner`] (it writes the launches, submits with
//! `WorkQueue::submit_fenced`, rings, waits for the last record and reads the logits + records back), and
//! [`Generation::run`] calls it once per position. **Any error stops the loop at once** ([`Stop`]): nothing more is
//! submitted, the tokens made so far are kept and the reason is named; the channel's cleanup is the caller's
//! (C-87 step 3). Optionally every logit vector is judged against the CPU engine ([`TokenCheck`]).
//!
//! **Speed** ([`GenStats`]): only from the board's clock, the timestamps of the records — the sum of the launch
//! times (GPU busy) and the span from the first record of the first token to the last record of the last one (this
//! one includes the driver's time between tokens: what a user sees). In SIMULARE the clock is the model's and the
//! numbers say nothing about the board.
//!
//! **Budget** ([`VramBudget`]): every buffer the OS maps for a model (weights blob, caches, programs, launch area,
//! records), page-rounded, plus the page tables — computed from the shapes alone and checked against the free VRAM
//! **before** the first upload (C-87 step 4: 24 layers, 183M).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::layer::{MAX_POS, Shape};
use gpu_compute::model;
use gpu_compute::q8;
use gpu_core::error::{GpuError, GpuResult};

use crate::kronos_layer::{self as kl, PROGRAMS};
use crate::kronos_model::{self as km, OwnedModel, TokenVerdict};

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

const PAGE: u64 = 0x1000;

fn pages(b: u64) -> u64 {
    b.next_multiple_of(PAGE)
}

// ------------------------------------------------------------------------------------------------ generation

/// What one token's run gave back: the logits (vocab values) and the board's clock at its first and last record.
#[derive(Clone, Debug, PartialEq)]
pub struct TokenOutput {
    pub logits: Vec<f32>,
    /// Timestamps (ns, board clock) of record 0 and of the last record of this token's submission.
    pub start_ns: u64,
    pub end_ns: u64,
    /// Sum of the launch durations (`transfer::analyze_sequence`).
    pub busy_ns: u64,
}

/// One token on the GPU (simulated or real). Implemented by the caller: the OS on the board, the tests on `gpu-sim`.
pub trait TokenRunner {
    fn run_token(&mut self, token: usize, pos: usize) -> GpuResult<TokenOutput>;
}

/// The CPU engine's logits for the same (token, position), to judge the GPU's against (optional).
pub trait TokenCheck {
    fn cpu_logits(&mut self, token: usize, pos: usize) -> GpuResult<Vec<f32>>;
}

/// Why the loop stopped.
#[derive(Clone, Debug, PartialEq)]
pub enum Stop {
    /// All `n` tokens made.
    Gata,
    /// The end-of-text token came (if one was given); it is kept as the last token.
    Sfarsit,
    /// The model's context (or `MAX_POS`) is full.
    ContextPlin,
    /// The runner failed at `pos`: nothing more was submitted.
    Eroare { pos: usize, err: GpuError },
    /// The logits at `pos` have a NaN/infinity or no maximum.
    LogitiNefiniti { pos: usize },
    /// The CPU check refused the logits at `pos`.
    NuTrece { pos: usize, verdict: TokenVerdict },
    /// The board's clock went back between tokens (or within one).
    CeasInapoi { pos: usize },
}

impl Stop {
    pub fn is_clean_end(&self) -> bool {
        matches!(self, Stop::Gata | Stop::Sfarsit | Stop::ContextPlin)
    }

    pub fn line(&self) -> String {
        match self {
            Stop::Gata => String::from("toate simbolurile cerute"),
            Stop::Sfarsit => String::from("simbolul de sfarsit"),
            Stop::ContextPlin => String::from("contextul modelului e plin"),
            Stop::Eroare { pos, err } => {
                format!("EROARE la pozitia {pos}: {err} (nimic nou trimis dupa ea)")
            }
            Stop::LogitiNefiniti { pos } => {
                format!("EROARE la pozitia {pos}: logit-uri NaN/infinit (nimic nou trimis)")
            }
            Stop::NuTrece { pos, verdict } => format!(
                "EROARE la pozitia {pos}: fata de procesor: {}",
                verdict.line(*pos)
            ),
            Stop::CeasInapoi { pos } => {
                format!("EROARE la pozitia {pos}: ceasul placii merge inapoi")
            }
        }
    }
}

/// The plan of one generation, checked before the first submission.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GenPlan {
    pub prompt: Vec<usize>,
    /// Greedy tokens wanted after the prompt.
    pub n: usize,
    pub vocab: usize,
    pub context: usize,
    /// Stop early when this token is generated (kept as the last one).
    pub eos: Option<usize>,
}

impl GenPlan {
    pub fn new(prompt: &[usize], n: usize, vocab: usize, context: usize) -> GpuResult<Self> {
        if prompt.is_empty() || n == 0 {
            return Err(bad("generare: text gol sau n = 0"));
        }
        if prompt.iter().any(|&t| t >= vocab) {
            return Err(bad("generare: simbol in afara vocabularului"));
        }
        if context == 0 || context > MAX_POS || prompt.len() > context {
            return Err(bad("generare: textul nu incape in context"));
        }
        Ok(GenPlan {
            prompt: prompt.to_vec(),
            n,
            vocab,
            context,
            eos: None,
        })
    }

    /// Positions run on the GPU: every prompt token and every generated one but the last (it is read from the
    /// logits of the position before), capped by the context.
    pub fn positions(&self) -> usize {
        (self.prompt.len() + self.n - 1).min(self.context)
    }
}

/// Speed and size of a generation (board clock only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenStats {
    /// Positions run on the GPU.
    pub positions: usize,
    /// Tokens generated (after the prompt).
    pub generated: usize,
    /// Sum of the launch times of all positions.
    pub busy_ns: u64,
    /// From the first record of the first position to the last record of the last one.
    pub span_ns: u64,
    pub median_ns: u64,
    pub slowest_ns: u64,
}

impl GenStats {
    /// Positions per second × 100 over the span (`None` without a measured span).
    pub fn per_s_x100(&self) -> Option<u64> {
        (self.span_ns > 0)
            .then(|| (self.positions as u128 * 100 * 1_000_000_000 / self.span_ns as u128) as u64)
    }

    /// The same over the GPU busy time only (the ceiling without the driver's gaps).
    pub fn busy_per_s_x100(&self) -> Option<u64> {
        (self.busy_ns > 0)
            .then(|| (self.positions as u128 * 100 * 1_000_000_000 / self.busy_ns as u128) as u64)
    }

    /// One journal line; `label` is "SIMULARE" or "placa".
    pub fn line(&self, label: &str) -> String {
        let r = |v: Option<u64>| match v {
            Some(x) => format!("{}.{:02}", x / 100, x % 100),
            None => String::from("?"),
        };
        format!(
            "generare ({label}): {} pozitii, {} simboluri noi; {} simboluri/s (ceasul placii, cu pauzele driverului), {} simboluri/s doar lansarile; median {} ns, cel mai lent {} ns",
            self.positions,
            self.generated,
            r(self.per_s_x100()),
            r(self.busy_per_s_x100()),
            self.median_ns,
            self.slowest_ns
        )
    }
}

/// A generation in progress (or finished). `seq` = prompt + generated tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct Generation {
    pub plan: GenPlan,
    pub seq: Vec<usize>,
    pub stop: Option<Stop>,
    /// Worst absolute logit gap against the CPU (only with a check).
    pub worst_abs: f32,
    per_token: Vec<u64>,
    first_ns: Option<u64>,
    last_ns: u64,
    busy_ns: u64,
    pos: usize,
}

impl Generation {
    pub fn new(plan: GenPlan) -> Self {
        Generation {
            seq: plan.prompt.clone(),
            plan,
            stop: None,
            worst_abs: 0.0,
            per_token: Vec::new(),
            first_ns: None,
            last_ns: 0,
            busy_ns: 0,
            pos: 0,
        }
    }

    /// The next (token, position) to run, `None` once stopped.
    pub fn next(&self) -> Option<(usize, usize)> {
        if self.stop.is_some() {
            return None;
        }
        Some((self.seq[self.pos], self.pos))
    }

    /// Tokens generated after the prompt.
    pub fn generated(&self) -> &[usize] {
        &self.seq[self.plan.prompt.len()..]
    }

    fn end(&mut self, s: Stop) {
        self.stop = Some(s);
    }

    /// Records the runner's result for the position [`Self::next`] gave; decides the next token or the stop.
    pub fn accept(
        &mut self,
        out: GpuResult<TokenOutput>,
        cpu: Option<GpuResult<Vec<f32>>>,
    ) -> Option<&Stop> {
        let Some((_, pos)) = self.next() else {
            return self.stop.as_ref();
        };
        let out = match out {
            Ok(o) => o,
            Err(err) => {
                self.end(Stop::Eroare { pos, err });
                return self.stop.as_ref();
            }
        };
        if out.end_ns < out.start_ns || out.start_ns < self.last_ns {
            self.end(Stop::CeasInapoi { pos });
            return self.stop.as_ref();
        }
        if out.logits.len() != self.plan.vocab {
            self.end(Stop::Eroare {
                pos,
                err: bad("generare: logit-uri de alta marime decat vocabularul"),
            });
            return self.stop.as_ref();
        }
        let Some(top) = model::argmax(&out.logits) else {
            self.end(Stop::LogitiNefiniti { pos });
            return self.stop.as_ref();
        };
        if let Some(cpu) = cpu {
            match cpu {
                Err(err) => {
                    self.end(Stop::Eroare { pos, err });
                    return self.stop.as_ref();
                }
                Ok(c) => match km::judge(&out.logits, &c) {
                    TokenVerdict::Trece { max_abs, .. } => {
                        self.worst_abs = self.worst_abs.max(max_abs)
                    }
                    verdict => {
                        self.end(Stop::NuTrece { pos, verdict });
                        return self.stop.as_ref();
                    }
                },
            }
        }
        self.first_ns.get_or_insert(out.start_ns);
        self.last_ns = out.end_ns;
        self.busy_ns += out.busy_ns;
        self.per_token.push(out.end_ns - out.start_ns);
        self.pos += 1;
        if self.pos >= self.seq.len() {
            self.seq.push(top);
            if self.plan.eos == Some(top) {
                self.end(Stop::Sfarsit);
            }
        }
        if self.stop.is_none() {
            if self.generated().len() >= self.plan.n {
                self.end(Stop::Gata);
            } else if self.pos >= self.plan.context {
                self.end(Stop::ContextPlin);
            }
        }
        self.stop.as_ref()
    }

    /// Runs to the end: one `run_token` per position, the check (if any) after each; stops at the first error.
    pub fn run<R: TokenRunner + ?Sized>(
        &mut self,
        r: &mut R,
        mut check: Option<&mut dyn TokenCheck>,
    ) -> &Stop {
        while let Some((tok, pos)) = self.next() {
            let out = r.run_token(tok, pos);
            let cpu = match (&out, check.as_deref_mut()) {
                (Ok(_), Some(c)) => Some(c.cpu_logits(tok, pos)),
                _ => None,
            };
            self.accept(out, cpu);
        }
        self.stop.as_ref().unwrap_or(&Stop::Gata)
    }

    pub fn stats(&self) -> GenStats {
        let mut t = self.per_token.clone();
        t.sort_unstable();
        GenStats {
            positions: self.per_token.len(),
            generated: self.generated().len(),
            busy_ns: self.busy_ns,
            span_ns: self.first_ns.map_or(0, |f| self.last_ns - f),
            median_ns: t.get(t.len() / 2).copied().unwrap_or(0),
            slowest_ns: t.last().copied().unwrap_or(0),
        }
    }
}

// ------------------------------------------------------------------------------------------------ VRAM budget

/// Bytes of VRAM a model takes on the work channel, each piece page-rounded as the OS maps it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VramBudget {
    /// Weights (Q8_0 matrices + f32 vectors), the activation scratch and the final vector + logits: one blob.
    pub blob: u64,
    /// Key and value caches of every layer (`2 · layers · MAX_POS · E · 4`).
    pub caches: u64,
    /// The NAK programs.
    pub programs: u64,
    /// The launch area (constant buffer + QMD, 512 B each, + the logits copy) and the records.
    pub launches: u64,
    /// Page tables for all of the above (one 4 KiB table per 2 MiB of VA, plus the upper levels).
    pub tables: u64,
}

impl VramBudget {
    pub fn total(&self) -> u64 {
        self.blob + self.caches + self.programs + self.launches + self.tables
    }

    /// The largest single piece that must be contiguous (the blob).
    pub fn largest(&self) -> u64 {
        self.blob.max(self.caches / 2)
    }

    /// The budget of a model with these shapes (`layers` layers, the tied head).
    pub fn of(s: &Shape, layers: usize, vocab: usize, context: usize) -> GpuResult<Self> {
        let e = s.e;
        if layers == 0 || e == 0 || !e.is_multiple_of(q8::BLOCK) || vocab == 0 {
            return Err(bad("buget VRAM: forma modelului gresita"));
        }
        if context > MAX_POS {
            return Err(bad("buget VRAM: context peste MAX_POS"));
        }
        let q = |rows: usize, cols: usize| (rows * cols / q8::BLOCK * q8::BLOCK_BYTES) as u64;
        let f = |n: usize| 4 * n as u64;
        // Each tensor of the blob 256-aligned (the OS packs at 256).
        let al = |b: u64| b.next_multiple_of(256);
        let layer = al(f(e)) * 4 // ln1 g/b, ln2 g/b
            + al(q(3 * e, e))
            + al(f(3 * e))
            + al(q(e, e))
            + al(f(e))
            + al(q(4 * e, e))
            + al(f(4 * e))
            + al(q(e, 4 * e))
            + al(f(e));
        let blob = al(q(vocab, e))
            + al(q(context, e))
            + 2 * al(f(e))
            + layers as u64 * layer
            + al(kl::scratch_bytes(e))
            + al(f(e + vocab));
        let n = km::launches_per_token(layers) as u64 + 1;
        let b = VramBudget {
            blob: pages(blob),
            caches: 2 * layers as u64 * pages(kl::cache_bytes(e)),
            programs: PROGRAMS.iter().map(|p| pages(p.code().len() as u64)).sum(),
            launches: pages(512 * n) + 3 * PAGE + pages(16 * (n + 1)),
            tables: 0,
        };
        let mapped = b.blob + b.caches + b.programs + b.launches;
        // Level 4 (PTEs): one table per 2 MiB, every piece starting its own; levels 1–3: a handful.
        let pieces = 2 * layers as u64 + PROGRAMS.len() as u64 + 6;
        let pts = mapped.div_ceil(2 << 20) + pieces;
        Ok(VramBudget {
            tables: (pts + 8) * PAGE,
            ..b
        })
    }

    pub fn of_model(m: &OwnedModel) -> GpuResult<Self> {
        Self::of(&m.shape, m.layers.len(), m.vocab, m.context)
    }

    /// Refuses (before anything is uploaded) when the total does not fit in `free` bytes or the blob in
    /// `largest_free` contiguous ones; `reserve` is kept free for what comes after (screen, other steps).
    pub fn check(&self, free: u64, largest_free: u64, reserve: u64) -> GpuResult<()> {
        let need = self.total() + reserve;
        if need > free {
            return Err(GpuError::OutOfMemory { requested: need });
        }
        if self.blob > largest_free {
            return Err(GpuError::OutOfMemory {
                requested: self.blob,
            });
        }
        Ok(())
    }

    pub fn line(&self, layers: usize) -> String {
        let mib = |b: u64| format!("{}.{:01}", b >> 20, ((b & 0xF_FFFF) * 10) >> 20);
        format!(
            "buget VRAM ({layers} straturi): {} MiB = greutati+lucru {} + cache-uri {} + programe {} + lansari {} + tabele {} MiB",
            mib(self.total()),
            mib(self.blob),
            mib(self.caches),
            mib(self.programs),
            mib(self.launches),
            mib(self.tables)
        )
    }
}

/// The pushbuffer bytes of one position (+ the copy of the logits down, as the OS does) for `layers` layers;
/// with two in flight this is what the area must hold (C-87 step 4: 24 layers).
pub const fn push_bytes_with_copy(layers: usize) -> usize {
    4 * crate::transfer::sequence_words(km::launches_per_token(layers) + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake {
        calls: usize,
        fail_at: Option<usize>,
        vocab: usize,
        t: u64,
    }

    impl TokenRunner for Fake {
        fn run_token(&mut self, token: usize, pos: usize) -> GpuResult<TokenOutput> {
            self.calls += 1;
            if self.fail_at == Some(pos) {
                return Err(GpuError::Timeout { waited_us: 1000 });
            }
            let mut logits = alloc::vec![0.0f32; self.vocab];
            logits[(token * 7 + pos + 1) % self.vocab] = 1.0;
            let start = self.t + 50;
            self.t = start + 1000;
            Ok(TokenOutput {
                logits,
                start_ns: start,
                end_ns: self.t,
                busy_ns: 900,
            })
        }
    }

    fn fake(fail_at: Option<usize>) -> Fake {
        Fake {
            calls: 0,
            fail_at,
            vocab: 32,
            t: 0,
        }
    }

    #[test]
    fn a_plan_is_checked_before_anything_runs() {
        assert!(GenPlan::new(&[], 3, 32, 16).is_err());
        assert!(GenPlan::new(&[1], 0, 32, 16).is_err());
        assert!(GenPlan::new(&[32], 3, 32, 16).is_err());
        assert!(GenPlan::new(&[1; 17], 3, 32, 16).is_err());
        assert!(GenPlan::new(&[1], 3, 32, MAX_POS + 1).is_err());
        let p = GenPlan::new(&[1, 2], 5, 32, 16).unwrap();
        assert_eq!(p.positions(), 6);
    }

    #[test]
    fn n_tokens_in_a_row_with_speed() {
        let mut g = Generation::new(GenPlan::new(&[3, 4], 6, 32, 64).unwrap());
        let mut r = fake(None);
        assert_eq!(*g.run(&mut r, None), Stop::Gata);
        assert_eq!(g.generated().len(), 6);
        assert_eq!(r.calls, 7, "2 + 6 - 1 pozitii");
        let s = g.stats();
        assert_eq!((s.positions, s.generated), (7, 6));
        assert_eq!(s.busy_ns, 7 * 900);
        assert_eq!(s.span_ns, 7 * 1050 - 50);
        assert_eq!(s.median_ns, 1000);
        // 7 positions in 7300 ns = 958 904.10/s.
        assert_eq!(s.per_s_x100(), Some(95_890_410));
        assert!(s.line("SIMULARE").contains("SIMULARE"));
        // The tokens follow the fake's rule: next = (token·7 + pos + 1) mod 32.
        let mut want = alloc::vec![3usize, 4];
        for pos in 1..7 {
            want.push((want[pos] * 7 + pos + 1) % 32);
        }
        assert_eq!(g.seq, want);
    }

    #[test]
    fn an_error_stops_at_once_and_keeps_what_was_made() {
        let mut g = Generation::new(GenPlan::new(&[3], 10, 32, 64).unwrap());
        let mut r = fake(Some(4));
        let stop = g.run(&mut r, None).clone();
        assert_eq!(
            stop,
            Stop::Eroare {
                pos: 4,
                err: GpuError::Timeout { waited_us: 1000 }
            }
        );
        assert!(!stop.is_clean_end());
        assert_eq!(r.calls, 5, "nimic trimis dupa eroare");
        assert_eq!(g.generated().len(), 4);
        assert_eq!(g.next(), None);
        assert_eq!(g.stats().positions, 4);
        assert!(stop.line().contains("pozitia 4"));
    }

    #[test]
    fn context_eos_nan_and_clock_stop_cleanly() {
        let mut g = Generation::new(GenPlan::new(&[1, 2, 3], 100, 32, 8).unwrap());
        assert_eq!(*g.run(&mut fake(None), None), Stop::ContextPlin);
        assert_eq!(g.stats().positions, 8);
        assert_eq!(g.seq.len(), 9);

        let mut p = GenPlan::new(&[3], 100, 32, 64).unwrap();
        // 3 -> (21+1)=22 at pos 0, 22 -> (154+2)%32=28 at pos 1.
        p.eos = Some(28);
        let mut g = Generation::new(p);
        assert_eq!(*g.run(&mut fake(None), None), Stop::Sfarsit);
        assert_eq!(g.generated(), &[22, 28]);

        let mut g = Generation::new(GenPlan::new(&[3], 4, 32, 64).unwrap());
        let nan = TokenOutput {
            logits: alloc::vec![f32::NAN; 32],
            start_ns: 0,
            end_ns: 1,
            busy_ns: 1,
        };
        assert_eq!(
            g.accept(Ok(nan), None),
            Some(&Stop::LogitiNefiniti { pos: 0 })
        );

        let mut g = Generation::new(GenPlan::new(&[3], 4, 32, 64).unwrap());
        let mut r = fake(None);
        g.accept(r.run_token(3, 0), None);
        let mut back = r.run_token(22, 1).unwrap();
        back.start_ns = 10;
        assert_eq!(g.accept(Ok(back), None), Some(&Stop::CeasInapoi { pos: 1 }));
        // Wrong logit count is an error, not a token.
        let mut g = Generation::new(GenPlan::new(&[3], 4, 32, 64).unwrap());
        let mut o = fake(None).run_token(3, 0).unwrap();
        o.logits.pop();
        assert!(matches!(g.accept(Ok(o), None), Some(Stop::Eroare { .. })));
    }

    struct Cpu(Option<usize>);
    impl TokenCheck for Cpu {
        fn cpu_logits(&mut self, token: usize, pos: usize) -> GpuResult<Vec<f32>> {
            let mut l = alloc::vec![0.0f32; 32];
            let k = if self.0 == Some(pos) {
                0
            } else {
                (token * 7 + pos + 1) % 32
            };
            l[k] = 1.0;
            Ok(l)
        }
    }

    #[test]
    fn the_cpu_check_stops_a_wrong_token() {
        let mut g = Generation::new(GenPlan::new(&[3], 5, 32, 64).unwrap());
        assert_eq!(*g.run(&mut fake(None), Some(&mut Cpu(None))), Stop::Gata);
        assert_eq!(g.worst_abs, 0.0);
        let mut g = Generation::new(GenPlan::new(&[3], 5, 32, 64).unwrap());
        let mut r = fake(None);
        assert!(matches!(
            g.run(&mut r, Some(&mut Cpu(Some(2)))),
            Stop::NuTrece { pos: 2, .. }
        ));
        assert_eq!(r.calls, 3);
    }

    fn kronos(layers: usize) -> VramBudget {
        VramBudget::of(
            &Shape {
                e: 768,
                heads: 12,
                eps: 1e-5,
            },
            layers,
            16384,
            512,
        )
        .unwrap()
    }

    #[test]
    fn the_budget_of_513_and_of_the_24_layer_model() {
        let (b12, b24) = (kronos(12), kronos(24));
        // Q8_0 matrices of one layer: 12·768² params · 36/32 B = 7 962 624 B; caches 2·512·768·4 = 3 MiB per layer.
        assert_eq!(b24.caches, (24 * 3) << 20);
        let per_layer = b24.blob - b12.blob;
        assert!((12 * 7_962_624..12 * 7_962_624 + 12 * 0x10000).contains(&per_layer));
        assert!(b12.total() < 200 << 20, "{}", b12.line(12));
        assert!(
            b24.total() > 270 << 20 && b24.total() < 300 << 20,
            "{}",
            b24.line(24)
        );
        // 8 GiB board, ~1 GiB kept by GSP/screen: plenty.
        assert!(b24.check(7 << 30, 6 << 30, 64 << 20).is_ok());
        // Refused before any upload when the zone is too small or too broken up.
        assert!(matches!(
            b24.check(256 << 20, 256 << 20, 0),
            Err(GpuError::OutOfMemory { .. })
        ));
        assert!(b24.check(1 << 30, 100 << 20, 0).is_err());
        assert!(b24.line(24).starts_with("buget VRAM (24 straturi): 2"));
        // The pushbuffer stretch of one 24-layer position (387 launches + the copy).
        assert_eq!(km::launches_per_token(24), 387);
        assert_eq!(push_bytes_with_copy(24), 4 * (8 + 388 * 10));
        assert!(2 * push_bytes_with_copy(24) <= 32 << 10);
        assert!(
            2 * push_bytes_with_copy(24) > 16 << 10,
            "16 KiB nu mai ajung"
        );
        assert!(
            VramBudget::of(
                &Shape {
                    e: 30,
                    heads: 2,
                    eps: 1e-5
                },
                1,
                16,
                16
            )
            .is_err()
        );
        assert!(
            VramBudget::of(
                &Shape {
                    e: 64,
                    heads: 2,
                    eps: 1e-5
                },
                1,
                16,
                600
            )
            .is_err()
        );
    }
}
