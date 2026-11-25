//! C-157 (centre note 25 Nov 11:55, request 2; `docs/C-157_PROGRAM_PERSISTENT.md`): what a "persistent" program —
//! one launch that stays on the board for a whole layer — could win, before anyone writes one. Pure arithmetic on the
//! times the OS **measured** on the board (os-431 step profile, os-346 per-launch generation times) and on the
//! "waves" model ([`crate::wave_model`]); nothing is sent, nothing is simulated. Every number returned is **DEDUS**:
//! a ceiling or a prediction, never a measurement.
//!
//! A persistent program wins from three sources only: (1) the gaps between launches, (2) intermediate results that
//! stay in fast memory instead of a VRAM round trip, (3) the idle slots of a product's last wave, handed to
//! independent work. This module sizes each one.

use alloc::vec::Vec;

use crate::kronos_perf::PAPER_MB_S;
use crate::train_profile::ParsedProfile;
use crate::train_step::Launch;
use crate::wave_model::{self as wm, CAL_K, SMS};

/// L2 cache of the GA104 (public NVIDIA spec sheet: 4 MB).
pub const L2_BYTES: u64 = 4 << 20;

/// Shared memory one SM can give its blocks on SM 8.6 (public: 100 KiB of the 128 KiB L1).
pub const SHARED_PER_SM: u64 = 100 << 10;

/// Register file of one SM (64 Ki × 32 bit).
pub const REGS_PER_SM: u64 = 256 << 10;

/// The fixed cost of one launch in the training step: os-392 (`antrenare mini`, MĂSURAT) −26 µs for 6 fused launches.
pub const TRAIN_GAP_NS: u64 = 4_300;

/// Upper bound of the fixed cost of one generation launch: the smallest whole launch measured (os-346, `ln2`,
/// 3.87 µs — its own work included, so the fixed part is at most this).
pub const GEN_GAP_NS: u64 = 3_870;

/// Fast memory a whole-board persistent program could keep a result in, at best (L2; the shared memories of all SMs
/// together are about as large but split per block).
pub const fn fast_bytes() -> u64 {
    let smem = SMS as u64 * SHARED_PER_SM;
    if smem > L2_BYTES { smem } else { L2_BYTES }
}

/// One intermediate result of a training layer (f32), as the step writes it to VRAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Intermediate {
    pub name: &'static str,
    pub bytes: u64,
}

/// The intermediates of one GPT layer's forward pass at `rows` = batch × context and width `e` (f32): the LayerNorm
/// output, qkv, the attention output, the hidden layer before and after GELU.
pub fn layer_intermediates(rows: u64, e: u64) -> [Intermediate; 5] {
    let f = |name, cols: u64| Intermediate {
        name,
        bytes: rows * cols * 4,
    };
    [
        f("ln", e),
        f("qkv", 3 * e),
        f("atentia", e),
        f("ascuns", 4 * e),
        f("gelu", 4 * e),
    ]
}

/// The programs that are neither products, attention nor the optimizer (the "small" ones a persistent program would
/// fold into its neighbours).
pub const SMALL_PROGRAMS: [&str; 10] = [
    "colsum",
    "colsum_part",
    "gelu_tr",
    "vadd",
    "ln_fwd",
    "ln_back",
    "xent",
    "emb_fwd",
    "emb_back",
    "gnorm",
];

/// The training step's budget from one board profile (µs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainBudget {
    pub step_us: u64,
    pub launches: u32,
    /// Launches × [`TRAIN_GAP_NS`]: the ceiling of source (1).
    pub gaps_us: u64,
    /// The [`SMALL_PROGRAMS`]: the ceiling of source (2) on the board today (all their time, as if folded away).
    pub small_us: u64,
}

impl TrainBudget {
    pub fn from_profile(p: &ParsedProfile) -> Self {
        Self {
            step_us: p.total_us,
            launches: p.launches,
            gaps_us: p.launches as u64 * TRAIN_GAP_NS / 1_000,
            small_us: p.programs_sum(&SMALL_PROGRAMS),
        }
    }

    /// `part` as thousandths of the step.
    pub fn permille(&self, part_us: u64) -> u64 {
        part_us * 1_000 / self.step_us.max(1)
    }
}

/// One product C (M × N) += A·B over K.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Product {
    pub m: u32,
    pub n: u32,
    pub k: u32,
}

/// The two backward products of one weight of one layer: dX and dW do not depend on each other.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Pair {
    /// `qkv`, `out`, `fc`, `proj`.
    pub name: &'static str,
    pub dx: Product,
    pub dw: Product,
}

fn product(l: &Launch) -> Product {
    let at =
        |o: usize| u32::from_le_bytes([l.cbuf[o], l.cbuf[o + 1], l.cbuf[o + 2], l.cbuf[o + 3]]);
    Product {
        m: at(0x48),
        n: at(0x4c),
        k: at(0x50),
    }
}

/// The dX / dW pairs of `layer` in a step's launch list (tags `inapoi:<name>-dX` / `inapoi:<name>-dW`), in list order.
/// Only product launches are read (the race variants and `mm_train*`).
pub fn backward_pairs(launches: &[Launch], layer: u16) -> Vec<Pair> {
    let mut out = Vec::new();
    let mine: Vec<&Launch> = launches
        .iter()
        .filter(|l| l.layer == Some(layer) && l.prog.race_index().is_some())
        .collect();
    for dx in &mine {
        let Some(name) = dx
            .tag
            .strip_prefix("inapoi:")
            .and_then(|t| t.strip_suffix("-dX"))
        else {
            continue;
        };
        let dw = mine.iter().find(|l| {
            l.tag
                .strip_prefix("inapoi:")
                .and_then(|t| t.strip_suffix("-dW"))
                == Some(name)
        });
        if let Some(dw) = dw {
            out.push(Pair {
                name,
                dx: product(dx),
                dw: product(dw),
            });
        }
    }
    out
}

/// One pair on one variant at one occupancy (ns, race clock, DEDUS).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PairCost {
    /// dX then dW, two launches ([`wm::launch_ns`] each).
    pub separate_ns: u64,
    /// One launch whose blocks are both products' blocks, packed perfectly on the slots: the work in "block × K"
    /// spread over all slots, but never below the longest single block. A floor — no scheduler does better.
    pub dual_floor_ns: u64,
    /// The idle share of the separate launches' slots, in thousandths (1 − work / (time × slots)).
    pub idle_permille: u64,
}

/// [`PairCost`] of `p` on `variant` with `b` blocks per SM. `None` if the model has no times there.
pub fn pair_cost(variant: usize, b: u32, p: &Pair) -> Option<PairCost> {
    let one = |q: &Product| wm::launch_ns(variant, q.m, q.n, q.k, b);
    let separate_ns = one(&p.dx)? + one(&p.dw)?;
    let wave = wm::wave_ns(variant, b)?;
    let slots = (SMS * b) as u64;
    let work = wm::blocks(variant, p.dx.m, p.dx.n) as u64 * p.dx.k as u64
        + wm::blocks(variant, p.dw.m, p.dw.n) as u64 * p.dw.k as u64;
    let packed = (wave * work).div_ceil(slots * CAL_K);
    let longest = wave * p.dx.k.max(p.dw.k) as u64 / CAL_K;
    let dual_floor_ns = packed.max(longest);
    // work in "slot-ns": every block costs one wave's time at its K
    let busy = wave as u128 * work as u128 / CAL_K as u128;
    let offered = separate_ns as u128 * slots as u128;
    let idle_permille = (1_000 - (busy * 1_000 / offered.max(1)).min(1_000)) as u64;
    Some(PairCost {
        separate_ns,
        dual_floor_ns,
        idle_permille,
    })
}

/// The best separate cost of `p` (each product on its own best modelled variant at `step_bytes`) and the best dual
/// floor (one variant for both, the best one) — what a dual launch could still win on top of the per-form choice.
pub fn pair_best_at(p: &Pair, step_bytes: u32) -> Option<(u64, u64)> {
    let best = |q: &Product| {
        wm::modelled()
            .filter_map(|v| wm::launch_ns_at(v, q.m, q.n, q.k, step_bytes))
            .min()
    };
    let separate = best(&p.dx)? + best(&p.dw)?;
    let dual = wm::modelled()
        .filter_map(|v| {
            let b = wm::blocks_per_sm(v, step_bytes)?;
            pair_cost(v, b, p).map(|c| c.dual_floor_ns)
        })
        .min()?;
    Some((separate, dual))
}

/// The per-launch generation times of #513 on `lipita-ln-32` (os-346, MĂSURAT, ns): one layer's 7 launches.
pub const OS346_LAYER_NS: [(&str, u64); 7] = [
    ("ln1", 4_030),
    ("qkv", 14_690),
    ("atentia", 20_770),
    ("out", 6_050),
    ("ln2", 3_870),
    ("fc", 18_780),
    ("proj", 11_900),
];

/// os-346: the logit launch (ns) and the board time of one token (ns), MĂSURAT.
pub const OS346_LOGIT_NS: u64 = 88_000;
pub const OS346_BOARD_NS: u64 = 1_071_000;

/// Bytes of Q8_0 weights (34 bytes per 32 values) of one GPT layer of width `e` (qkv, out, fc, proj).
pub const fn q8_layer_bytes(e: u64) -> u64 {
    e * (3 * e + e + 4 * e + 4 * e) / 32 * 34
}

/// One generation layer's budget (ns, DEDUS from MĂSURAT launch times).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GenLayer {
    /// The 7 launches as measured.
    pub layer_ns: u64,
    /// 7 × [`GEN_GAP_NS`]: the ceiling of what one launch for the layer saves.
    pub gaps_ns: u64,
    /// The layer's Q8 weights read once at the paper bandwidth: no program can go under this.
    pub floor_ns: u64,
}

/// [`GenLayer`] for width `e` with [`OS346_LAYER_NS`].
pub fn gen_layer(e: u64) -> GenLayer {
    let layer_ns = OS346_LAYER_NS.iter().map(|x| x.1).sum();
    GenLayer {
        layer_ns,
        gaps_ns: OS346_LAYER_NS.len() as u64 * GEN_GAP_NS,
        // bytes / (MB/s) = µs·1e-0 … bytes × 1 000 / MB_S = ns
        floor_ns: q8_layer_bytes(e) * 1_000 / PAPER_MB_S,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes() {
        // #513 lot 2 × 512: the hidden layer is 12 MiB, qkv 9 MiB.
        let i = layer_intermediates(1024, 768);
        assert_eq!(i[1].bytes, 9 << 20);
        assert_eq!(i[3].bytes, 12 << 20);
        assert_eq!(fast_bytes(), 48 * (100 << 10));
        assert_eq!(q8_layer_bytes(768), 7_520_256);
    }

    #[test]
    fn a_pair_that_fills_the_tail() {
        // fc at #513: dX 1024 × 768 × 3072 (48 blocks on 128x128 = one half-empty wave at 2 per SM),
        // dW 3072 × 768 × 1024 (144 blocks); the same flops, so the dual floor = dX alone.
        let p = Pair {
            name: "fc",
            dx: Product {
                m: 1024,
                n: 768,
                k: 3072,
            },
            dw: Product {
                m: 3072,
                n: 768,
                k: 1024,
            },
        };
        let c = pair_cost(3, 2, &p).unwrap();
        assert_eq!(c.dual_floor_ns, 637_750 * 3072 / 768);
        assert!(c.dual_floor_ns < c.separate_ns);
    }
}
