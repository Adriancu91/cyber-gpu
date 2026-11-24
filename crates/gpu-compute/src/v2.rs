//! C-164 — **Kronos „croit pe placă"** (the next Kronos, from zero): the Cyber-GPU side of the shared sheet. For each
//! candidate choice (ReLU² instead of GELU, RMS norm instead of LayerNorm, no biases, widths that divide the tiles,
//! the head) this module says **what it does to the training step on our board**: which launches and programs
//! disappear or change, how many parameters, how the matrix work splits — and it holds the CPU oracles of the two new
//! pieces (RMS norm, ReLU²), so the board programs that would replace `ln_fwd` / `ln_back` / `gelu_tr` can later be
//! checked the way every training program is (C-157 point 9 b).
//!
//! Counting rules = today's [`crate::train`] step as the board runs it (`nvidia_ga104::train_step::Plan::step`):
//! per layer 31 launches (4 forward matmuls + 8 backward, 3 attention, 2 + 2 norms, 6 column sums, 2 GELU, 4 additions),
//! plus 13 per step at `adamw_513` (11 at `sgd`). [`StepCount::count`] with [`TODAY`] and no fusions gives exactly
//! that (checked against the plan itself in `nvidia-ga104/tests/c164_kronos_v2.rs`).
//!
//! Nothing here runs on a board and nothing here changes `kronos_gpt`: it is design + SIMULARE (CPU) only. The
//! choice of architecture is Kronos's and Adrian's (C-164: „doar proiect și simulare acum").
//!
//! Sources (formulas only, our own code): the PyTorch documentation of `RMSNorm` / `F.rms_norm` (BSD; default eps for
//! f32 = `torch.finfo(torch.float32).eps`), Zhang & Sennrich 2019 (RMSNorm), So et al. 2021 („Primer": squared ReLU);
//! the hub's `idei/2026-11-24-centru-cele-patru-piese.md` (the candidate list).

use alloc::vec;
use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::train::{Arch, sqrt64};

fn need(cond: bool, what: &'static str) -> GpuResult<()> {
    if cond {
        Ok(())
    } else {
        Err(GpuError::InvalidArgument(what))
    }
}

/// `torch.finfo(torch.float32).eps` = 2⁻²³: PyTorch's default eps of `RMSNorm` for f32 inputs.
pub const RMS_EPS_TORCH_F32: f64 = 1.192_092_895_507_812_5e-7;

// ------------------------------------------------------------------ the choices

/// The MLP's activation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    /// Today: exact GELU (erf).
    Gelu,
    /// `relu(u)²`; its derivative `2·relu(u)`.
    Relu2,
}

/// The normalization before attention, before the MLP and before the head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Norm {
    /// Today: LayerNorm with γ (and β when [`Choices::bias`]).
    LayerNorm,
    /// RMS norm with a learned γ.
    Rms,
    /// RMS norm without any learned weight (`F.rms_norm(x, (C,))`).
    RmsNoWeight,
}

/// One candidate architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Choices {
    pub act: Act,
    pub norm: Norm,
    /// Biases on the four linear layers (and β on a LayerNorm).
    pub bias: bool,
}

/// Today's `kronos_gpt`.
pub const TODAY: Choices = Choices {
    act: Act::Gelu,
    norm: Norm::LayerNorm,
    bias: true,
};

/// The candidate list of C-164, one change at a time, then all together (the order of the sheet).
pub const CANDIDATES: [(&str, Choices); 5] = [
    ("azi", TODAY),
    (
        "ReLU²",
        Choices {
            act: Act::Relu2,
            ..TODAY
        },
    ),
    (
        "fără deplasări",
        Choices {
            bias: false,
            ..TODAY
        },
    ),
    (
        "RMS cu γ",
        Choices {
            norm: Norm::Rms,
            ..TODAY
        },
    ),
    (
        "toate (RMS fără γ)",
        Choices {
            act: Act::Relu2,
            norm: Norm::RmsNoWeight,
            bias: false,
        },
    ),
];

/// The fusions a step may use. The first two exist today for GELU (`ln_back_res`, `mm_train_gelu`, SIMULARE); the
/// third does not exist for any activation yet.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct Fusions {
    /// The residual addition of the backward pass inside the norm's backward (`ln_back_res`): −2 `vadd` per layer.
    pub back_res: bool,
    /// The activation's backward inside the store of `proj-dX` (`mm_train_gelu`): −1 launch per layer.
    pub act_back_in_mm: bool,
    /// The activation's forward inside the store of `fc` (does not exist): −1 launch per layer.
    pub act_fwd_in_mm: bool,
}

// ------------------------------------------------------------------ parameters

impl Choices {
    /// Learned vectors of one norm (γ, β).
    fn norm_vectors(&self) -> usize {
        match self.norm {
            Norm::LayerNorm => 1 + usize::from(self.bias),
            Norm::Rms => 1,
            Norm::RmsNoWeight => 0,
        }
    }

    /// Parameter tensors: the two embeddings, per layer 4 weights + biases + norm vectors, the final norm.
    pub fn tensor_count(&self, a: &Arch) -> usize {
        let per_layer = 4 + if self.bias { 4 } else { 0 } + 2 * self.norm_vectors();
        2 + per_layer * a.layers + self.norm_vectors()
    }

    /// Parameters (the head is tied to `emb_token`, as today).
    pub fn params(&self, a: &Arch) -> usize {
        let c = a.n_emb;
        let per_layer =
            12 * c * c + if self.bias { 9 * c } else { 0 } + 2 * self.norm_vectors() * c;
        a.vocab * c + a.ctx * c + per_layer * a.layers + self.norm_vectors() * c
    }
}

// ------------------------------------------------------------------ launches of one step

/// The kinds of launches of a training step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Mm,
    /// A matmul with the activation's backward in its store (`mm_train_gelu` or its ReLU² sibling).
    MmActBack,
    /// A matmul with the activation's forward in its store (does not exist yet).
    MmActFwd,
    AttnFwd,
    AttnBack,
    NormFwd,
    NormBack,
    /// The norm's backward with the residual addition (`ln_back_res` or its RMS sibling).
    NormBackRes,
    Colsum,
    ActFwd,
    ActBack,
    Vadd,
    EmbFwd,
    EmbBack,
    Xent,
    Gnorm,
    Opt,
}

/// Every [`Kind`], in a fixed order.
pub const KINDS: [Kind; 17] = [
    Kind::Mm,
    Kind::MmActBack,
    Kind::MmActFwd,
    Kind::AttnFwd,
    Kind::AttnBack,
    Kind::NormFwd,
    Kind::NormBack,
    Kind::NormBackRes,
    Kind::Colsum,
    Kind::ActFwd,
    Kind::ActBack,
    Kind::Vadd,
    Kind::EmbFwd,
    Kind::EmbBack,
    Kind::Xent,
    Kind::Gnorm,
    Kind::Opt,
];

/// Launches of one step, per [`Kind`] (index = position in [`KINDS`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepCount {
    pub n: [usize; 17],
}

fn ix(k: Kind) -> usize {
    KINDS.iter().position(|&x| x == k).unwrap_or(0)
}

impl StepCount {
    /// The launches of one step of architecture `a` with `ch` and fusions `f` (`adamw`: the clipping norm's 2
    /// launches; otherwise the `sgd` recipe). Attention = 1 forward + 2 backward launches per layer, as in the plans
    /// on tiles (`Plan::new_tiles3`).
    pub fn count(a: &Arch, ch: Choices, f: Fusions, adamw: bool) -> StepCount {
        let mut n = [0usize; 17];
        let mut add = |k: Kind, v: usize| n[ix(k)] += v;
        let norm_colsum = usize::from(ch.norm != Norm::RmsNoWeight);
        for _ in 0..a.layers {
            // forward: norm1, qkv, attention, out, residual 1, norm2, fc, activation, proj, residual 2
            add(Kind::NormFwd, 2);
            add(Kind::Mm, 3);
            if f.act_fwd_in_mm {
                add(Kind::MmActFwd, 1);
            } else {
                add(Kind::Mm, 1);
                add(Kind::ActFwd, 1);
            }
            add(Kind::AttnFwd, 1);
            add(Kind::Vadd, 2);
            // backward: dX and dW of the four matmuls, biases, activation, two norms, two residuals, attention
            if f.act_back_in_mm {
                add(Kind::MmActBack, 1);
                add(Kind::Mm, 7);
            } else {
                add(Kind::Mm, 8);
                add(Kind::ActBack, 1);
            }
            if ch.bias {
                add(Kind::Colsum, 4);
            }
            if f.back_res {
                add(Kind::NormBackRes, 2);
            } else {
                add(Kind::NormBack, 2);
                add(Kind::Vadd, 2);
            }
            add(Kind::Colsum, 2 * norm_colsum);
            add(Kind::AttnBack, 2);
        }
        // once per step: embeddings, final norm, logits, loss, head dW / dX, final norm back, embeddings back
        add(Kind::EmbFwd, 1);
        add(Kind::NormFwd, 1);
        add(Kind::Mm, 3);
        add(Kind::Xent, 1);
        add(Kind::NormBack, 1);
        add(Kind::Colsum, norm_colsum);
        add(Kind::EmbBack, 2);
        if adamw {
            add(Kind::Gnorm, 2);
        }
        add(Kind::Opt, 1);
        StepCount { n }
    }

    pub fn get(&self, k: Kind) -> usize {
        self.n[ix(k)]
    }

    pub fn total(&self) -> usize {
        self.n.iter().sum()
    }

    /// Kinds with at least one launch.
    pub fn kinds_used(&self) -> usize {
        self.n.iter().filter(|&&v| v > 0).count()
    }

    /// Σ launches × µs per launch, for the kinds given in `us` (kinds not listed cost 0): the elementwise share of a
    /// step from measured per-launch times (DEDUS when the times come from another program).
    pub fn cost_us(&self, us: &[(Kind, f64)]) -> f64 {
        us.iter().map(|&(k, t)| self.get(k) as f64 * t).sum()
    }
}

/// Per-launch times on the board, **measured** in os-410 (hub `rezultate/os-automat/2026-11-24/ciclu-auto-20261124-
/// 1822.txt`, #513, 2 × 512, median over 10 steps): `colsum` 24 253 µs / 73, `gelu_tr` 1 448 / 24, `vadd` 817 / 48,
/// `ln_fwd` 570 / 25, `ln_back` 621 / 25. The RMS and ReLU² programs do not exist: [`OS410_SMALL_US`] gives them the
/// time of the program they replace (all are memory-bound passes over the same arrays — DEDUS).
pub const OS410_SMALL_US: [(Kind, f64); 7] = [
    (Kind::Colsum, 24_253.0 / 73.0),
    (Kind::ActFwd, 1_448.0 / 24.0),
    (Kind::ActBack, 1_448.0 / 24.0),
    (Kind::Vadd, 817.0 / 48.0),
    (Kind::NormFwd, 570.0 / 25.0),
    (Kind::NormBack, 621.0 / 25.0),
    (Kind::NormBackRes, 621.0 / 25.0),
];

// ------------------------------------------------------------------ matrix work and widths

/// FLOP of the matmuls of one step over `rows` = batch × context positions: `(layers, head)`, each = 3 passes
/// (forward, dX, dW) × 2·rows·m·k. Layers: 12·C² per position; head: C·V. (Attention is not a matmul here.)
pub fn mm_flop(a: &Arch, rows: usize) -> (f64, f64) {
    let (c, v) = (a.n_emb as f64, a.vocab as f64);
    let r = rows as f64;
    (6.0 * r * 12.0 * c * c * a.layers as f64, 6.0 * r * c * v)
}

/// The widths of the step's matrices that do **not** divide by `tile` (empty = every matmul and the attention's head
/// width fit whole tiles). `rows` = batch × context.
pub fn widths_off_tile(a: &Arch, rows: usize, tile: usize) -> Vec<(&'static str, usize)> {
    let c = a.n_emb;
    let mut out = Vec::new();
    for (name, w) in [
        ("C", c),
        ("3C", 3 * c),
        ("4C", 4 * c),
        ("V", a.vocab),
        ("T", a.ctx),
        ("rânduri", rows),
        ("d (capul de atenție)", c / a.heads.max(1)),
    ] {
        if tile == 0 || !w.is_multiple_of(tile) {
            out.push((name, w));
        }
    }
    out
}

// ------------------------------------------------------------------ CPU oracles of the new pieces

/// ReLU² forward: `relu(u)²`. The square of an f32 is exact in f64, so one rounding: the board's `r*r` in f32 gives
/// the same bits (no erf, no tolerance needed — unlike GELU).
pub fn relu2_rows(u: &[f32]) -> Vec<f32> {
    u.iter()
        .map(|&x| {
            let r = if x > 0.0 { x as f64 } else { 0.0 };
            (r * r) as f32
        })
        .collect()
}

/// ReLU² backward: `g · 2·relu(u)` (rounded once; `2·relu(u)·g` is exact in f64). At `u ≤ 0` the gradient is +0.0
/// times g (−0.0 for a negative g, as in PyTorch's `2·relu(u)·g`).
pub fn relu2_back_rows(g: &[f32], u: &[f32]) -> GpuResult<Vec<f32>> {
    need(g.len() == u.len(), "relu2 inapoi: forma")?;
    Ok(g.iter()
        .zip(u)
        .map(|(&gv, &x)| {
            let r = if x > 0.0 { x } else { 0.0f32 };
            (2.0 * r as f64 * gv as f64) as f32
        })
        .collect())
}

/// RMS norm forward over `rows × c`: `y = x · rstd · γ`, `rstd = 1/√(mean(x²) + eps)` (f64, rounded once). `g` = γ,
/// or `None` for the norm without weight. Returns `(y, rstd)` (rstd rounded to f32: what a board program keeps).
pub fn rms_rows(
    x: &[f32],
    g: Option<&[f32]>,
    rows: usize,
    c: usize,
    eps: f64,
) -> GpuResult<(Vec<f32>, Vec<f32>)> {
    need(
        rows > 0 && c > 0 && x.len() == rows * c && g.is_none_or(|g| g.len() == c) && eps > 0.0,
        "rms: forma",
    )?;
    let mut y = vec![0.0f32; rows * c];
    let mut rstd = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &x[r * c..(r + 1) * c];
        let ms = row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / c as f64;
        let rs = 1.0 / sqrt64(ms + eps);
        rstd[r] = rs as f32;
        for k in 0..c {
            let w = g.map_or(1.0, |g| g[k] as f64);
            y[r * c + k] = (row[k] as f64 * rs * w) as f32;
        }
    }
    Ok((y, rstd))
}

/// RMS norm backward over `rows × c` (`x` the forward input, `dy` the gradient of the output): returns `(dx, dγ)`,
/// dγ empty for the norm without weight. With x̂ = x·rstd and dx̂ = dy·γ:
/// `dx = rstd · (dx̂ − x̂ · mean(dx̂·x̂))`, `dγ = Σ_rows dy·x̂`. (LayerNorm's formula without the `− mean(dx̂)` term and
/// with μ = 0 — so dγ is today's `colsum` in its LayerNorm mode fed a zero mean; [`dgamma_is_colsum_with_zero_mean`].)
pub fn rms_back_rows(
    x: &[f32],
    dy: &[f32],
    g: Option<&[f32]>,
    rows: usize,
    c: usize,
    eps: f64,
) -> GpuResult<(Vec<f32>, Vec<f32>)> {
    need(
        rows > 0
            && c > 0
            && x.len() == rows * c
            && dy.len() == rows * c
            && g.is_none_or(|g| g.len() == c)
            && eps > 0.0,
        "rms inapoi: forma",
    )?;
    let mut dx = vec![0.0f32; rows * c];
    let mut acc_g = vec![0.0f64; c];
    let mut dxhat = vec![0.0f64; c];
    let mut xhat = vec![0.0f64; c];
    for r in 0..rows {
        let row = &x[r * c..(r + 1) * c];
        let ms = row.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / c as f64;
        let rs = 1.0 / sqrt64(ms + eps);
        let mut m2 = 0.0f64;
        for k in 0..c {
            let d = dy[r * c + k] as f64;
            xhat[k] = row[k] as f64 * rs;
            acc_g[k] += d * xhat[k];
            dxhat[k] = d * g.map_or(1.0, |g| g[k] as f64);
            m2 += dxhat[k] * xhat[k];
        }
        m2 /= c as f64;
        for k in 0..c {
            dx[r * c + k] = (rs * (dxhat[k] - xhat[k] * m2)) as f32;
        }
    }
    let dg = if g.is_some() {
        acc_g.iter().map(|&v| v as f32).collect()
    } else {
        Vec::new()
    };
    Ok((dx, dg))
}

/// `Σ_rows dy·(x − μ)·rstd` with μ = 0 and the RMS rstd — the formula of today's `colsum` program in its LayerNorm
/// mode (s0) — next to [`rms_back_rows`]'s dγ: the largest absolute difference. 0 or a last-bit rounding means the RMS
/// norm's dγ needs **no new program** (the OS passes a zeroed μ array, or the RMS forward writes one).
pub fn dgamma_is_colsum_with_zero_mean(
    x: &[f32],
    dy: &[f32],
    rows: usize,
    c: usize,
    eps: f64,
) -> GpuResult<f64> {
    let g = vec![1.0f32; c];
    let (_, dg) = rms_back_rows(x, dy, Some(&g), rows, c, eps)?;
    let (_, rstd) = rms_rows(x, None, rows, c, eps)?;
    let mut worst = 0.0f64;
    for k in 0..c {
        let mut s = 0.0f64;
        for r in 0..rows {
            let mean = 0.0f64;
            s += dy[r * c + k] as f64 * ((x[r * c + k] as f64 - mean) * rstd[r] as f64);
        }
        let d = (s as f32 as f64 - dg[k] as f64).abs();
        if d > worst {
            worst = d;
        }
    }
    Ok(worst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::{M513, MIC};

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    #[test]
    fn today_matches_the_trainer_tensors_and_the_plan_counts() {
        for a in [MIC, M513] {
            assert_eq!(TODAY.tensor_count(&a), a.tensor_count());
            let total: usize = a.tensors().iter().map(|s| s.elements()).sum();
            assert_eq!(TODAY.params(&a), total);
            let s = StepCount::count(&a, TODAY, Fusions::default(), true);
            assert_eq!(s.total(), 31 * a.layers + 13);
            let s = StepCount::count(&a, TODAY, Fusions::default(), false);
            assert_eq!(s.total(), 31 * a.layers + 11);
        }
        // os-410: 98 032 128 weights, 385 launches, 73 colsum, 24 gelu_tr, 48 vadd, 25 + 25 norms, 147 matmuls.
        assert_eq!(TODAY.params(&M513), 98_032_128);
        let s = StepCount::count(&M513, TODAY, Fusions::default(), true);
        assert_eq!(s.total(), 385);
        assert_eq!(s.get(Kind::Colsum), 73);
        assert_eq!(s.get(Kind::ActFwd) + s.get(Kind::ActBack), 24);
        assert_eq!(s.get(Kind::Vadd), 48);
        assert_eq!((s.get(Kind::NormFwd), s.get(Kind::NormBack)), (25, 25));
        assert_eq!(s.get(Kind::Mm), 147);
        assert_eq!(s.get(Kind::AttnFwd) + s.get(Kind::AttnBack), 36);
        // the measured small programs of os-410 add back up to the profile's own sums (27 709 µs)
        let us = s.cost_us(&OS410_SMALL_US);
        assert!((us - 27_709.0).abs() < 0.5, "{us}");
    }

    #[test]
    fn the_candidates_launches_and_parameters_at_513() {
        let f0 = Fusions::default();
        let rows: Vec<(usize, usize, usize)> = CANDIDATES
            .iter()
            .map(|&(_, ch)| {
                let s = StepCount::count(&M513, ch, f0, true);
                (s.total(), s.get(Kind::Colsum), ch.params(&M513))
            })
            .collect();
        assert_eq!(rows[0], (385, 73, 98_032_128));
        assert_eq!(rows[1], (385, 73, 98_032_128)); // ReLU²: same launches, cheaper program
        assert_eq!(rows[2], (337, 25, 98_032_128 - 12 * 9 * 768 - 25 * 768));
        assert_eq!(rows[3], (385, 73, 98_032_128 - 25 * 768));
        assert_eq!(rows[4], (312, 0, 98_032_128 - 12 * 9 * 768 - 50 * 768));
        // with the two fusions we already have (for GELU today): 385 → 349; all candidates → 276
        let f2 = Fusions {
            back_res: true,
            act_back_in_mm: true,
            act_fwd_in_mm: false,
        };
        assert_eq!(StepCount::count(&M513, TODAY, f2, true).total(), 349);
        assert_eq!(
            StepCount::count(&M513, CANDIDATES[4].1, f2, true).total(),
            276
        );
        let f3 = Fusions {
            act_fwd_in_mm: true,
            ..f2
        };
        assert_eq!(
            StepCount::count(&M513, CANDIDATES[4].1, f3, true).total(),
            264
        );
    }

    #[test]
    fn program_kinds_today_and_all_candidates() {
        // Today, plain: mm, attn ×2, norm ×2, colsum, act ×2, vadd, emb ×2, xent, gnorm, opt = 14 kinds
        // (the plan has 13 programs: GELU forward and backward are the one `gelu_tr`; attention back is 2 programs
        // on tiles, counted as one kind here).
        let s = StepCount::count(&M513, TODAY, Fusions::default(), true);
        assert_eq!(s.kinds_used(), 14);
        let all = CANDIDATES[4].1;
        let f2 = Fusions {
            back_res: true,
            act_back_in_mm: true,
            act_fwd_in_mm: true,
        };
        let s = StepCount::count(&M513, all, f2, true);
        // colsum, ActFwd, ActBack gone; MmActFwd, MmActBack, NormBackRes come (variants of programs we have):
        // the number of kinds does NOT fall to ~5 — the launches fall, the kinds stay 14
        assert_eq!(s.get(Kind::Colsum), 0);
        assert_eq!(s.get(Kind::ActFwd) + s.get(Kind::ActBack), 0);
        assert_eq!(s.kinds_used(), 14);
        assert_eq!(s.total(), 264);
    }

    #[test]
    fn the_513_widths_divide_the_tiles_and_the_head_is_13_percent() {
        assert!(widths_off_tile(&M513, 1024, 64).is_empty());
        // at 128 only the attention's head width (64) is off — the attention tiles are 64 wide anyway
        assert_eq!(
            widths_off_tile(&M513, 1024, 128),
            vec![("d (capul de atenție)", 64)]
        );
        assert_eq!(widths_off_tile(&M513, 1000, 64), vec![("rânduri", 1000)]);
        let (l, h) = mm_flop(&M513, 1024);
        // C-165: 599.1 GFLOP per step at #513, 2 × 512
        assert!(((l + h) / 1e9 - 599.1).abs() < 0.1, "{}", (l + h) / 1e9);
        let share = h / (l + h);
        assert!((share - 0.129).abs() < 0.001, "{share}");
    }

    #[test]
    fn relu2_forward_backward_against_finite_differences_and_bits() {
        let mut s = 7u64;
        let u: Vec<f32> = (0..4096).map(|_| 3.0 * lcg(&mut s)).collect();
        let g: Vec<f32> = (0..4096).map(|_| lcg(&mut s)).collect();
        let y = relu2_rows(&u);
        let d = relu2_back_rows(&g, &u).unwrap();
        for i in 0..u.len() {
            // the f32 board formula gives the same bits
            let r = u[i].max(0.0);
            assert_eq!(y[i].to_bits(), (r * r).to_bits());
            assert_eq!(d[i].to_bits(), ((2.0 * r) * g[i]).to_bits());
            // derivative by central differences (away from the kink)
            if u[i].abs() > 1e-2 {
                let h = 1e-3f64;
                let f = |x: f64| if x > 0.0 { x * x } else { 0.0 };
                let num = (f(u[i] as f64 + h) - f(u[i] as f64 - h)) / (2.0 * h) * g[i] as f64;
                assert!((num - d[i] as f64).abs() < 1e-5, "{i}");
            }
        }
        assert!(relu2_back_rows(&g[..3], &u).is_err());
    }

    fn rms_loss(x: &[f32], w: &[f32], g: Option<&[f32]>, rows: usize, c: usize) -> f64 {
        let (y, _) = rms_rows(x, g, rows, c, RMS_EPS_TORCH_F32).unwrap();
        y.iter().zip(w).map(|(&a, &b)| a as f64 * b as f64).sum()
    }

    #[test]
    fn rms_backward_against_finite_differences() {
        let (rows, c) = (5, 64);
        let mut s = 11u64;
        let x: Vec<f32> = (0..rows * c).map(|_| 2.0 * lcg(&mut s)).collect();
        let w: Vec<f32> = (0..rows * c).map(|_| lcg(&mut s)).collect();
        let gam: Vec<f32> = (0..c).map(|_| 1.0 + 0.5 * lcg(&mut s)).collect();
        for g in [Some(&gam[..]), None] {
            let (dx, dg) = rms_back_rows(&x, &w, g, rows, c, RMS_EPS_TORCH_F32).unwrap();
            assert_eq!(dg.len(), if g.is_some() { c } else { 0 });
            let h = 1e-2f32;
            for i in (0..rows * c).step_by(7) {
                let mut xp = x.clone();
                let mut xm = x.clone();
                xp[i] += h;
                xm[i] -= h;
                let num = (rms_loss(&xp, &w, g, rows, c) - rms_loss(&xm, &w, g, rows, c))
                    / (2.0 * h as f64);
                assert!((num - dx[i] as f64).abs() < 2e-3, "dx {i}: {num} {}", dx[i]);
            }
            if let Some(g) = g {
                for k in (0..c).step_by(5) {
                    let mut gp = g.to_vec();
                    let mut gm = g.to_vec();
                    gp[k] += h;
                    gm[k] -= h;
                    let num = (rms_loss(&x, &w, Some(&gp), rows, c)
                        - rms_loss(&x, &w, Some(&gm), rows, c))
                        / (2.0 * h as f64);
                    assert!((num - dg[k] as f64).abs() < 2e-3, "dg {k}");
                }
            }
        }
        // rows of a constant: rstd · x = ±1 (with eps ≪ x²)
        let (y, _) = rms_rows(&[3.0; 8], None, 1, 8, RMS_EPS_TORCH_F32).unwrap();
        assert!(y.iter().all(|&v| (v - 1.0).abs() < 1e-6));
        assert!(rms_rows(&[1.0; 8], None, 2, 8, 1e-6).is_err());
        assert!(rms_rows(&[1.0; 8], Some(&[1.0; 3]), 1, 8, 1e-6).is_err());
    }

    #[test]
    fn rms_dgamma_is_todays_colsum_formula_with_zero_mean() {
        let (rows, c) = (1024, 768);
        let mut s = 3u64;
        let x: Vec<f32> = (0..rows * c).map(|_| 2.0 * lcg(&mut s)).collect();
        let dy: Vec<f32> = (0..rows * c).map(|_| 0.01 * lcg(&mut s)).collect();
        let worst = dgamma_is_colsum_with_zero_mean(&x, &dy, rows, c, RMS_EPS_TORCH_F32).unwrap();
        // rstd rounded to f32 (what the board keeps) moves the sum by at most a few f32 ulps of the result
        assert!(worst < 1e-6, "{worst}");
    }
}
