//! C-157 (`docs/C-157_ATENTIE_TC.md`, predictions A1–A5 written before this code): what the causal attention of the
//! #513 training step could win on the tensor cores (f16 inputs, f32 sums), before anyone writes such a program.
//! Pure arithmetic on times the OS **measured** on the board (os-431: the three gen-3 attention programs; os-443:
//! `mm_tc_v4`); nothing is sent, nothing is simulated. Every number returned is **DEDUS**: an estimate, never a
//! measurement. Nothing here has run on the board.
//!
//! Model (stated, not measured):
//! - work = the 6 products of one layer on the 64 × 64 tile pairs below (and on) the diagonal: forward S = Q·Kᵀ and
//!   O = P·V, backward dP = dO·Vᵀ, dQ = dS·K, dK = dSᵀ·Q, dV = Pᵀ·dO; every pair is 64 × 64 × 64 multiply-adds;
//! - tensor-core cost of one pair = the whole-board time per pair of `mm_tc_v4` on qkv-513 at 2 blocks per SM
//!   (os-443), × 1 (low) or × 2 (high: small K, frequent tile stores);
//! - memory = the passes over P / S / dS (one causal T × T matrix per head each), f16 elements, at the paper
//!   bandwidth (low) or at [`HIGH_MB_S`] (high), added to the products (no overlap assumed in either bound).

use crate::kronos_perf::PAPER_MB_S;
use crate::wave_model::{self as wm, OS443_TC_V4, TC_V4_CAL_K, TC_V4_TILE};

/// The attention shape of the #513 step (lot 2 × 512, 12 heads of 64, 12 layers).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnForm {
    pub batch: u64,
    pub t: u64,
    pub heads: u64,
    pub d: u64,
    pub layers: u64,
}

/// #513 as the mixed step runs it (os-431 / os-445).
pub const M513_ATTN: AttnForm = AttnForm {
    batch: 2,
    t: 512,
    heads: 12,
    d: 64,
    layers: 12,
};

/// The tile edge of the gen-3 attention programs (and of `mm_tc_v4`).
pub const TILE: u64 = 64;

/// The bandwidth of the high bound: 75 % of the paper figure (an assumption, not measured on this card).
pub const HIGH_MB_S: u64 = PAPER_MB_S * 3 / 4;

/// One attention program of the step: its name in the OS profile, how many of the 6 products it does and how many
/// passes over a causal T × T matrix (P, S or dS) it makes in VRAM (read from `tools/nak/*_3.comp`, causal tiles only;
/// the zero stores above the diagonal are not counted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttnProgram {
    pub name: &'static str,
    pub products: u64,
    pub matrix_passes: u64,
}

/// The three gen-3 programs of the step, in the order they run.
pub const PROGRAMS: [AttnProgram; 3] = [
    // phase A stores S, phase B reads S and stores exp, phase C reads it and stores P
    AttnProgram {
        name: "attn_fwd_tile3",
        products: 2,
        matrix_passes: 5,
    },
    // reads P, stores dS
    AttnProgram {
        name: "attn_back_tile0_3",
        products: 1,
        matrix_passes: 2,
    },
    // reads dS (for dQ and for dK) and P (for dV)
    AttnProgram {
        name: "attn_back_tile_3",
        products: 3,
        matrix_passes: 3,
    },
];

impl AttnForm {
    /// Tiles along T.
    pub fn tiles(&self) -> u64 {
        self.t.div_ceil(TILE)
    }

    /// Tile pairs on and below the diagonal, per head: n (n + 1) / 2.
    pub fn causal_pairs(&self) -> u64 {
        let n = self.tiles();
        n * (n + 1) / 2
    }

    /// Tile pairs of one product over one layer (all heads of the lot).
    pub fn pairs_per_product(&self) -> u64 {
        self.causal_pairs() * self.heads * self.batch
    }

    /// FLOP (2 per multiply-add) of one product over one layer, on whole tiles of edge [`TILE`] (d ≤ 64 counts as 64).
    pub fn product_flop(&self) -> u64 {
        self.pairs_per_product() * 2 * TILE * TILE * TILE
    }

    /// Bytes of one pass over a causal T × T matrix of the layer (whole tiles), at `elem` bytes per element.
    pub fn matrix_pass_bytes(&self, elem: u64) -> u64 {
        self.pairs_per_product() * TILE * TILE * elem
    }
}

/// Whole-board ns per tile pair of `mm_tc_v4`, × 1 000 (os-443, qkv-513 at 2 blocks per SM): the launch time over
/// (blocks × K / 64) pairs.
pub fn tc_pair_ps() -> u64 {
    let (_, ns, _) = OS443_TC_V4[1];
    let pairs = wm::tc_v4_blocks(6144, 2304) as u64 * (TC_V4_CAL_K / TC_V4_TILE as u64);
    ns * 1_000 / pairs
}

/// ns to move `bytes` at `mb_s` MB/s.
pub fn move_ns(bytes: u64, mb_s: u64) -> u64 {
    bytes * 1_000 / mb_s
}

/// The tensor-core estimate of one program over the whole step (all layers), in ns: (low, high).
pub fn tc_program_ns(f: &AttnForm, p: &AttnProgram) -> (u64, u64) {
    let pairs = p.products * f.pairs_per_product() * f.layers;
    let mm = pairs * tc_pair_ps() / 1_000;
    let bytes = p.matrix_passes * f.matrix_pass_bytes(2) * f.layers;
    (
        mm + move_ns(bytes, PAPER_MB_S),
        2 * mm + move_ns(bytes, HIGH_MB_S),
    )
}

/// The same programs' memory passes today (f32 elements), whole step, in ns at the paper bandwidth and at
/// [`HIGH_MB_S`]: (low, high). A floor of today's memory time, not a measurement.
pub fn f32_memory_ns(f: &AttnForm) -> (u64, u64) {
    let passes: u64 = PROGRAMS.iter().map(|p| p.matrix_passes).sum();
    let bytes = passes * f.matrix_pass_bytes(4) * f.layers;
    (move_ns(bytes, PAPER_MB_S), move_ns(bytes, HIGH_MB_S))
}

/// The tensor-core estimate of the whole attention of the step, in ns: (low, high).
pub fn tc_total_ns(f: &AttnForm) -> (u64, u64) {
    PROGRAMS
        .iter()
        .map(|p| tc_program_ns(f, p))
        .fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
}

/// The attention at a given F32 rate (`flop_per_us` = MFLOP/s), whole step, in ns: what moving today's programs to
/// the efficiency of the step's F32 products would give, with no f16 at all.
pub fn f32_at_rate_ns(f: &AttnForm, flop_per_us: u64) -> u64 {
    let flop: u64 = PROGRAMS.iter().map(|p| p.products).sum::<u64>() * f.product_flop() * f.layers;
    flop * 1_000 / flop_per_us.max(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_of_513() {
        let f = M513_ATTN;
        assert_eq!(f.tiles(), 8);
        assert_eq!(f.causal_pairs(), 36);
        assert_eq!(f.pairs_per_product(), 864);
        assert_eq!(f.product_flop(), 864 * 524_288);
        // 36 tiles × 16 KiB × 24 heads = 13,5 MiB per f32 pass
        assert_eq!(f.matrix_pass_bytes(4), 27 << 19);
    }

    #[test]
    fn pair_cost_from_os443() {
        // 2 623 680 ns / (3 456 blocks × 12) = 63,26 ns
        assert_eq!(tc_pair_ps(), 63_263);
    }

    #[test]
    fn bounds_are_ordered() {
        let (lo, hi) = tc_total_ns(&M513_ATTN);
        assert!(lo < hi && hi < 2 * lo + 1);
    }
}
