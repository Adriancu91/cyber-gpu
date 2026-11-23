//! C-160 point 7 — **the probe that sends the training attention "on tiles" to the real board alone**: the chain
//! [`crate::attn_fwd_tile`] → [`crate::attn_back_tile0`] (mode 0, the rows of dS) → [`crate::attn_back_tile`] (mode 1,
//! dQ | dK | dV), outside any training step, and judges each program on its own (plan, margins and predictions written
//! before this code: `docs/C-160_P7_ATENTIA_PROBA.md`, commit `608f377`).
//!
//! - [`SETS`]: five small shapes read back whole and one **timing** set, `atentie-513-b4` (B 4, T 512, C 768, H 12:
//!   a third of a #513 chunk, one layer), read back in bands.
//! - The inputs come from a hash of (seed, element) ([`ProbeSet::qkv_at`], [`ProbeSet::dout_at`]), so the OS can write
//!   them piece by piece ([`ProbeSet::write_inputs`]).
//! - One buffer per set ([`ProbeSet::layout`]): qkv, dO, then P, att, dS, dqkv, each output followed by [`GUARD`]
//!   words; what is read back and the guards are filled with NaN ([`FILL`]) before the first launch.
//! - [`launch_bytes`]: cbuf 0 and the QMD (with the shared memory and the barrier) of each [`Prog`].
//! - [`judge_set`]: `attn_fwd_tile` against the CPU model `layer::attn_fwd_tile_row_gpu` in the **MUFU margin** (the
//!   model's EX2 / RCP are correctly rounded, the board's approximate); `attn_back_tile0` against
//!   `layer::attn_back_row_gpu` **on the P read back from the board**, and `attn_back_tile` against
//!   `layer::attn_back_col_gpu` on the P **and** dS read back — so neither depends on the board's MUFU and both are
//!   predicted bit for bit (counted separately; the verdict uses a small safety margin).
//! - [`Probe`]: the run, `kt atentie …` journal rows ([`Probe::rows`]), report lines, [`parse_attn_tile`].
//!
//! Tested **only in SIMULARE** (T176, `tests/sim_c160_p7_attn_tile_probe.rs`); nothing here has run on a board.
//!
//! C-157 point 2 adds, beside all of the above (unchanged), the probe **"old beside new"**: [`Prog2`] / [`PROGS2`] (the
//! second generation `attn_*_2`, same operations, same order), [`launch_bytes2`], [`judge_pair`] (the new chain judged
//! with the same margins and compared word for word with the old one on the same set), [`Probe2`] (`kt atentie2 …` rows)
//! and [`parse_attn_tile2`]. Plan: `docs/C-157_P2_ATTN_TILE2.md`; tested only in SIMULARE (T184).
//!
//! After os-364 it adds, again beside everything above (unchanged), the third generation: [`Prog3`] / [`PROGS3`]
//! (`attn_*_3`, branch-free loads, same operations in the same order), [`launch_bytes3`], [`Probe3`] (`kt atentie3 …`
//! rows, judged with [`judge_pair`] against the first generation, timed beside the second) and [`parse_attn_tile3`].
//! Plan: `docs/C-157_P2_ATTN_TILE3.md` (section of session 182); tested only in SIMULARE (T191).
//!
//! After os-375 it adds, again beside everything above (unchanged), the fourth generation: [`Prog4`] / [`PROGS4`]
//! (`attn_fwd_tile3` → `attn_back_tile0_3` → `attn_back_tile_4`: only the last program is new), [`launch_bytes4`],
//! [`Probe4`] (`kt atentie4 …` rows, judged with [`judge_pair`] against the first generation, timed beside the third)
//! and [`parse_attn_tile4`]. Plan: `docs/C-157_P2_ATTN_TILE4.md` (section of session 189); tested only in SIMULARE (T197).

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::layer::{AttnShape, attn_back_col_gpu, attn_back_row_gpu, attn_fwd_tile_row_gpu};
use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use crate::attn_back::{self as ab, Mode};
use crate::attn_back_tile_4 as b14;
use crate::attn_fwd as af;
use crate::qmd::{ComputeQmd, SharedMemory};
use crate::{attn_back_tile as b1, attn_back_tile0 as b0, attn_fwd_tile as ft};
use crate::{attn_back_tile_2 as b12, attn_back_tile0_2 as b02, attn_fwd_tile2 as ft2};
use crate::{attn_back_tile_3 as b13, attn_back_tile0_3 as b03, attn_fwd_tile3 as ft3};

/// Words after each output, filled with [`FILL`], that must stay untouched.
pub const GUARD: usize = 256;
/// The word outputs and guards are filled with before the first launch (a quiet NaN).
pub const FILL: u32 = 0x7fc0_0000;
/// Bytes of constant buffer 0 sent with every launch (the programs read 0x54 / 0x68 of them).
pub const CBUF_SIZE: usize = 0x100;
/// Index of the timing set in [`SETS`].
pub const TIMING_SET: usize = 5;
/// Launches of every program on the timing set (the time row is their median).
pub const REPEATS: u32 = 5;
/// Texts of the timing set that are read back.
pub const BAND_TEXTS: [usize; 2] = [0, 3];
/// Rows (positions) of the timing set whose P / dS / att rows are read back; the last one is T − 1, the only row
/// whose dK / dV need nothing more than its own rows of P and dS.
pub const BAND_ROWS: [usize; 3] = [0, 300, 511];

/// MUFU margin of `attn_fwd_tile` against its CPU model: 2⁻¹⁸ relative (written before the code).
const MUFU_REL: f64 = 1.0 / (1u64 << 18) as f64;
/// One unit in the last place of 1 (f32).
const U: f64 = 1.0 / (1u64 << 24) as f64;
/// The smallest normal f32 (what flush-to-zero can lose per operation).
const FTZ: f64 = 1.175_494_350_822_287_5e-38;
/// Safety margin of the backward programs (their prediction is bit for bit): 2⁻²⁰ of the row's largest value.
const BACK_REL: f64 = 1.0 / (1u64 << 20) as f64;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The three programs of the probe, in launch order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prog {
    /// [`crate::attn_fwd_tile`]: P and att.
    Fwd,
    /// [`crate::attn_back_tile0`]: the rows of dS (mode 0).
    Rows,
    /// [`crate::attn_back_tile`]: dQ | dK | dV (mode 1).
    Cols,
}

/// The programs in launch order.
pub const PROGS: [Prog; 3] = [Prog::Fwd, Prog::Rows, Prog::Cols];

impl Prog {
    pub fn name(self) -> &'static str {
        match self {
            Prog::Fwd => "attn_fwd_tile",
            Prog::Rows => "attn_back_tile0",
            Prog::Cols => "attn_back_tile",
        }
    }
    /// The program bytes the OS loads.
    pub fn binary(self) -> &'static [u8] {
        match self {
            Prog::Fwd => ft::ATTN_FWD_TILE,
            Prog::Rows => b0::ATTN_BACK_TILE0,
            Prog::Cols => b1::ATTN_BACK_TILE,
        }
    }
    pub fn sha256(self) -> &'static str {
        match self {
            Prog::Fwd => ft::ATTN_FWD_TILE_SHA256,
            Prog::Rows => b0::ATTN_BACK_TILE0_SHA256,
            Prog::Cols => b1::ATTN_BACK_TILE_SHA256,
        }
    }
    pub fn registers(self) -> u32 {
        match self {
            Prog::Fwd => ft::REGISTERS,
            Prog::Rows => b0::REGISTERS,
            Prog::Cols => b1::REGISTERS,
        }
    }
    pub fn shared(self) -> SharedMemory {
        match self {
            Prog::Fwd => ft::SHARED,
            Prog::Rows => b0::SHARED,
            Prog::Cols => b1::SHARED,
        }
    }
}

/// C-157 point 2 (the second part for os-358): the **second generation** of the three programs —
/// [`crate::attn_fwd_tile2`], [`crate::attn_back_tile0_2`], [`crate::attn_back_tile_2`]: the same operations in the
/// same order as the first (the same bits predicted), faster loops. Wraps the role ([`Prog`]); the first generation and
/// [`PROGS`] are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prog2(pub Prog);

/// The second generation in launch order.
pub const PROGS2: [Prog2; 3] = [Prog2(Prog::Fwd), Prog2(Prog::Rows), Prog2(Prog::Cols)];

impl Prog2 {
    pub fn name(self) -> &'static str {
        match self.0 {
            Prog::Fwd => "attn_fwd_tile2",
            Prog::Rows => "attn_back_tile0_2",
            Prog::Cols => "attn_back_tile_2",
        }
    }
    /// The first-generation program it sits beside.
    pub fn old(self) -> Prog {
        self.0
    }
    /// The program bytes the OS loads.
    pub fn binary(self) -> &'static [u8] {
        match self.0 {
            Prog::Fwd => ft2::ATTN_FWD_TILE2,
            Prog::Rows => b02::ATTN_BACK_TILE0_2,
            Prog::Cols => b12::ATTN_BACK_TILE_2,
        }
    }
    pub fn sha256(self) -> &'static str {
        match self.0 {
            Prog::Fwd => ft2::ATTN_FWD_TILE2_SHA256,
            Prog::Rows => b02::ATTN_BACK_TILE0_2_SHA256,
            Prog::Cols => b12::ATTN_BACK_TILE_2_SHA256,
        }
    }
    pub fn registers(self) -> u32 {
        match self.0 {
            Prog::Fwd => ft2::REGISTERS,
            Prog::Rows => b02::REGISTERS,
            Prog::Cols => b12::REGISTERS,
        }
    }
    pub fn shared(self) -> SharedMemory {
        match self.0 {
            Prog::Fwd => ft2::SHARED,
            Prog::Rows => b02::SHARED,
            Prog::Cols => b12::SHARED,
        }
    }
}

/// C-157 point 2 after os-364: the **third generation** — [`crate::attn_fwd_tile3`], [`crate::attn_back_tile0_3`],
/// [`crate::attn_back_tile_3`]: the same operations in the same order as the first two (the same bits predicted),
/// branch-free global loads. Wraps the role ([`Prog`]); [`PROGS`] and [`PROGS2`] are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prog3(pub Prog);

/// The third generation in launch order.
pub const PROGS3: [Prog3; 3] = [Prog3(Prog::Fwd), Prog3(Prog::Rows), Prog3(Prog::Cols)];

impl Prog3 {
    pub fn name(self) -> &'static str {
        match self.0 {
            Prog::Fwd => "attn_fwd_tile3",
            Prog::Rows => "attn_back_tile0_3",
            Prog::Cols => "attn_back_tile_3",
        }
    }
    /// The first-generation program it is compared with word for word.
    pub fn old(self) -> Prog {
        self.0
    }
    /// The second-generation program it is timed beside.
    pub fn prev(self) -> Prog2 {
        Prog2(self.0)
    }
    /// The program bytes the OS loads.
    pub fn binary(self) -> &'static [u8] {
        match self.0 {
            Prog::Fwd => ft3::ATTN_FWD_TILE3,
            Prog::Rows => b03::ATTN_BACK_TILE0_3,
            Prog::Cols => b13::ATTN_BACK_TILE_3,
        }
    }
    pub fn sha256(self) -> &'static str {
        match self.0 {
            Prog::Fwd => ft3::ATTN_FWD_TILE3_SHA256,
            Prog::Rows => b03::ATTN_BACK_TILE0_3_SHA256,
            Prog::Cols => b13::ATTN_BACK_TILE_3_SHA256,
        }
    }
    pub fn registers(self) -> u32 {
        match self.0 {
            Prog::Fwd => ft3::REGISTERS,
            Prog::Rows => b03::REGISTERS,
            Prog::Cols => b13::REGISTERS,
        }
    }
    pub fn shared(self) -> SharedMemory {
        match self.0 {
            Prog::Fwd => ft3::SHARED,
            Prog::Rows => b03::SHARED,
            Prog::Cols => b13::SHARED,
        }
    }
}

/// C-157 point 2 after os-375: the **fourth generation** — [`crate::attn_fwd_tile3`] and [`crate::attn_back_tile0_3`]
/// (the same binaries as [`Prog3`]; run again because the last program reads the P and dS they write) and
/// [`crate::attn_back_tile_4`] (the whole diagonal tile of dK / dV on the unguarded loop; the same bits predicted).
/// Wraps the role ([`Prog`]); [`PROGS`], [`PROGS2`] and [`PROGS3`] are unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Prog4(pub Prog);

/// The fourth generation in launch order.
pub const PROGS4: [Prog4; 3] = [Prog4(Prog::Fwd), Prog4(Prog::Rows), Prog4(Prog::Cols)];

impl Prog4 {
    pub fn name(self) -> &'static str {
        match self.0 {
            Prog::Cols => "attn_back_tile_4",
            _ => Prog3(self.0).name(),
        }
    }
    /// The first-generation program it is compared with word for word.
    pub fn old(self) -> Prog {
        self.0
    }
    /// The third-generation program it is timed beside.
    pub fn prev(self) -> Prog3 {
        Prog3(self.0)
    }
    /// Whether the binary differs from the third generation's (only `attn_back_tile_4`).
    pub fn is_new(self) -> bool {
        self.0 == Prog::Cols
    }
    /// The program bytes the OS loads.
    pub fn binary(self) -> &'static [u8] {
        match self.0 {
            Prog::Cols => b14::ATTN_BACK_TILE_4,
            _ => Prog3(self.0).binary(),
        }
    }
    pub fn sha256(self) -> &'static str {
        match self.0 {
            Prog::Cols => b14::ATTN_BACK_TILE_4_SHA256,
            _ => Prog3(self.0).sha256(),
        }
    }
    pub fn registers(self) -> u32 {
        match self.0 {
            Prog::Cols => b14::REGISTERS,
            _ => Prog3(self.0).registers(),
        }
    }
    pub fn shared(self) -> SharedMemory {
        match self.0 {
            Prog::Cols => b14::SHARED,
            _ => Prog3(self.0).shared(),
        }
    }
}

/// One probe set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProbeSet {
    pub name: &'static str,
    pub batch: u32,
    pub t: u32,
    pub c: u32,
    pub heads: u32,
    pub seed: u64,
    /// qkv uniform in [−amp, amp); dO uniform in [−1, 1).
    pub amp: f32,
    /// `None`: everything read back. `Some((texts, rows))`: only these texts and rows (the last row must be T − 1).
    pub bands: Option<(&'static [usize], &'static [usize])>,
    /// Launches of every program (the time row is their median).
    pub repeats: u32,
}

const fn small(
    name: &'static str,
    batch: u32,
    t: u32,
    c: u32,
    heads: u32,
    seed: u64,
    amp: f32,
) -> ProbeSet {
    ProbeSet {
        name,
        batch,
        t,
        c,
        heads,
        seed,
        amp,
        bands: None,
        repeats: 1,
    }
}

/// The probe: five small shapes and the timing set.
pub const SETS: [ProbeSet; 6] = [
    small("mic-b2-t16-d16", 2, 16, 32, 2, 1, 2.0),
    small("t130-d64", 1, 130, 128, 2, 2, 2.0),
    small("t64-d40", 2, 64, 40, 1, 3, 2.0),
    small("t200-h3", 1, 200, 192, 3, 4, 2.0),
    small("ascutit", 1, 96, 64, 1, 5, 6.0),
    ProbeSet {
        name: "atentie-513-b4",
        batch: 4,
        t: 512,
        c: 768,
        heads: 12,
        seed: 6,
        amp: 2.0,
        bands: Some((&BAND_TEXTS, &BAND_ROWS)),
        repeats: REPEATS,
    },
];

/// Byte offsets inside the buffer of one set (every region aligned to 256 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub qkv: usize,
    pub dout: usize,
    pub p: usize,
    pub p_guard: usize,
    pub att: usize,
    pub att_guard: usize,
    pub ds: usize,
    pub ds_guard: usize,
    pub dqkv: usize,
    pub dqkv_guard: usize,
    /// End of the last guard.
    pub end: usize,
    /// The buffer size the OS allocates (`end` rounded up to 4 KiB).
    pub bytes: usize,
}

fn up(v: usize, to: usize) -> usize {
    v.div_ceil(to) * to
}

fn splitmix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Uniform in [−1, 1) on a 2⁻²³ grid, from (seed, which array, element).
fn unit(seed: u64, which: u64, idx: usize) -> f32 {
    let h = splitmix(seed.wrapping_mul(0x1_0000_0001) ^ (which << 56) ^ idx as u64);
    ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

impl ProbeSet {
    pub fn shape(&self) -> AttnShape {
        AttnShape {
            batch: self.batch as usize,
            t: self.t as usize,
            c: self.c as usize,
            heads: self.heads as usize,
        }
    }

    fn dims(&self) -> (usize, usize, usize, usize, usize) {
        let s = self.shape();
        (s.batch, s.t, s.c, s.heads, s.d())
    }

    /// The scale 1/√d the training step uses.
    pub fn scale(&self) -> f32 {
        ab::Args::scale_for(self.c / self.heads)
    }

    /// Whether all three programs take this shape (d ≤ 64, C divisible by H, the bands inside).
    pub fn shape_ok(&self) -> bool {
        let fa = self.fwd_args(0x1000);
        let ok = ft::shape_ok(&fa)
            && b0::shape_ok(&self.back_args(0x1000, Mode::Rows))
            && b1::shape_ok(&self.back_args(0x1000, Mode::Columns));
        let bands_ok = match self.bands {
            None => true,
            Some((texts, rows)) => {
                !texts.is_empty()
                    && !rows.is_empty()
                    && texts.iter().all(|&b| b < self.batch as usize)
                    && rows.windows(2).all(|w| w[0] < w[1])
                    && rows.last() == Some(&(self.t as usize - 1))
            }
        };
        ok && bands_ok
    }

    pub fn layout(&self) -> Layout {
        let (b, t, c, h, _) = self.dims();
        let (nq, no, np) = (b * t * 3 * c, b * t * c, b * h * t * t);
        let g = 4 * GUARD;
        let qkv = 0;
        let dout = up(qkv + 4 * nq, 256);
        let p = up(dout + 4 * no, 256);
        let p_guard = p + 4 * np;
        let att = up(p_guard + g, 256);
        let att_guard = att + 4 * no;
        let ds = up(att_guard + g, 256);
        let ds_guard = ds + 4 * np;
        let dqkv = up(ds_guard + g, 256);
        let dqkv_guard = dqkv + 4 * nq;
        let end = dqkv_guard + g;
        Layout {
            qkv,
            dout,
            p,
            p_guard,
            att,
            att_guard,
            ds,
            ds_guard,
            dqkv,
            dqkv_guard,
            end,
            bytes: up(end, 4096),
        }
    }

    pub fn qkv_at(&self, idx: usize) -> f32 {
        self.amp * unit(self.seed, 1, idx)
    }

    pub fn dout_at(&self, idx: usize) -> f32 {
        unit(self.seed, 2, idx)
    }

    /// Writes qkv and dO (and the zero gap between them) through `put(offset, bytes)`, in pieces of at most 64 KiB,
    /// in increasing offsets. Everything before [`Layout::p`].
    pub fn write_inputs(&self, mut put: impl FnMut(usize, &[u8])) {
        let (b, t, c, _, _) = self.dims();
        let l = self.layout();
        let regions: [(usize, usize, u8); 2] = [(l.qkv, b * t * 3 * c, 1), (l.dout, b * t * c, 2)];
        let mut piece = Vec::with_capacity(64 << 10);
        for (at, count, which) in regions {
            let mut i = 0;
            while i < count {
                piece.clear();
                let stop = (i + (16 << 10)).min(count);
                for e in i..stop {
                    let v = if which == 1 {
                        self.qkv_at(e)
                    } else {
                        self.dout_at(e)
                    };
                    piece.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                put(at + 4 * i, &piece);
                i = stop;
            }
        }
    }

    /// FNV-1a (32 bit) of the input bytes (`[0, p)` of the buffer), computed piece by piece.
    pub fn inputs_fnv(&self) -> u32 {
        let mut h = Fnv::new();
        let mut at = 0;
        self.write_inputs(|o, bytes| {
            h.zeros(o - at);
            h.add(bytes);
            at = o + bytes.len();
        });
        h.zeros(self.layout().p - at);
        h.0
    }

    /// The texts read back and judged.
    pub fn texts(&self) -> Vec<usize> {
        match self.bands {
            None => (0..self.batch as usize).collect(),
            Some((texts, _)) => texts.to_vec(),
        }
    }

    /// The rows (positions) whose P / dS / att rows are read back and judged.
    pub fn rows(&self) -> Vec<usize> {
        match self.bands {
            None => (0..self.t as usize).collect(),
            Some((_, rows)) => rows.to_vec(),
        }
    }

    /// The positions whose dqkv rows are read back and judged (all, or only T − 1 on a banded set).
    pub fn dq_rows(&self) -> Vec<usize> {
        match self.bands {
            None => (0..self.t as usize).collect(),
            Some(_) => alloc::vec![self.t as usize - 1],
        }
    }

    /// The byte ranges `(offset, len)` read back after the launches, in the order [`judge_set`] expects them
    /// concatenated: P rows (b, h, i), P guard, att rows (b, i), att guard, dS rows, dS guard, dqkv rows, dqkv guard.
    /// Adjacent ranges are merged (a small set is at most 8 ranges).
    pub fn read_ranges(&self) -> Vec<(usize, usize)> {
        let (_, t, c, h, _) = self.dims();
        let l = self.layout();
        let (texts, rows, dq) = (self.texts(), self.rows(), self.dq_rows());
        let mut v: Vec<(usize, usize)> = Vec::new();
        let mut push = |o: usize, n: usize| match v.last_mut() {
            Some(last) if last.0 + last.1 == o => last.1 += n,
            _ => v.push((o, n)),
        };
        for (base, guard) in [(l.p, l.p_guard), (l.att, l.att_guard), (l.ds, l.ds_guard)] {
            let is_att = base == l.att;
            for &b in &texts {
                if is_att {
                    for &i in &rows {
                        push(base + 4 * (b * t + i) * c, 4 * c);
                    }
                } else {
                    for hh in 0..h {
                        for &i in &rows {
                            push(base + 4 * ((b * h + hh) * t + i) * t, 4 * t);
                        }
                    }
                }
            }
            push(guard, 4 * GUARD);
        }
        for &b in &texts {
            for &tp in &dq {
                push(l.dqkv + 4 * (b * t + tp) * 3 * c, 4 * 3 * c);
            }
        }
        push(l.dqkv_guard, 4 * GUARD);
        v
    }

    /// The ranges the OS fills with [`FILL`] before the first launch: everything read back (the guards included).
    pub fn fill_ranges(&self) -> Vec<(usize, usize)> {
        self.read_ranges()
    }

    /// Bytes read back in total.
    pub fn read_len(&self) -> usize {
        self.read_ranges().iter().map(|r| r.1).sum()
    }

    /// The whole buffer as bytes (inputs, the read-back regions = [`FILL`]). For the small sets and the tests; for
    /// the timing set the OS uses [`Self::write_inputs`] and [`Self::fill_ranges`] instead (~151 MB).
    pub fn buffer(&self) -> Vec<u8> {
        let l = self.layout();
        let mut b = alloc::vec![0u8; l.bytes];
        self.write_inputs(|at, bytes| b[at..at + bytes.len()].copy_from_slice(bytes));
        for (at, len) in self.fill_ranges() {
            for w in b[at..at + len].as_chunks_mut::<4>().0 {
                w.copy_from_slice(&FILL.to_le_bytes());
            }
        }
        b
    }

    /// The forward program's arguments for a buffer at `buf_va`.
    pub fn fwd_args(&self, buf_va: u64) -> af::Args {
        let l = self.layout();
        af::Args {
            qkv: buf_va + l.qkv as u64,
            p: buf_va + l.p as u64,
            att: buf_va + l.att as u64,
            batch: self.batch,
            t: self.t,
            c: self.c,
            heads: self.heads,
            scale: self.scale(),
        }
    }

    /// A backward program's arguments for a buffer at `buf_va` (dO is read from the dO region).
    pub fn back_args(&self, buf_va: u64, mode: Mode) -> ab::Args {
        let l = self.layout();
        ab::Args {
            qkv: buf_va + l.qkv as u64,
            p: buf_va + l.p as u64,
            dout: buf_va + l.dout as u64,
            ds: buf_va + l.ds as u64,
            dqkv: buf_va + l.dqkv as u64,
            batch: self.batch,
            t: self.t,
            c: self.c,
            heads: self.heads,
            scale: self.scale(),
            mode,
        }
    }

    /// qkv [T, 3C] and dO [T, C] of text `b`.
    fn text_inputs(&self, b: usize) -> (Vec<f32>, Vec<f32>) {
        let (_, t, c, _, _) = self.dims();
        let q0 = b * t * 3 * c;
        let o0 = b * t * c;
        (
            (q0..q0 + t * 3 * c).map(|i| self.qkv_at(i)).collect(),
            (o0..o0 + t * c).map(|i| self.dout_at(i)).collect(),
        )
    }

    /// What the CPU models say the board writes, as the bytes read back (guards = FILL): P and att from
    /// `attn_fwd_tile_row_gpu`, dS from `attn_back_row_gpu` on that P, dqkv from `attn_back_col_gpu` on both.
    pub fn model_out(&self) -> Vec<u8> {
        let e = evaluate(self, None);
        let mut out = Vec::with_capacity(self.read_len());
        let put = |out: &mut Vec<u8>, v: &[f32]| {
            for x in v {
                out.extend_from_slice(&x.to_bits().to_le_bytes());
            }
            for _ in 0..GUARD {
                out.extend_from_slice(&FILL.to_le_bytes());
            }
        };
        put(&mut out, &e.model[0]);
        put(&mut out, &e.model[1]);
        put(&mut out, &e.model[2]);
        put(&mut out, &e.model[3]);
        out
    }
}

/// FNV-1a, 32 bit, fed in pieces.
struct Fnv(u32);

impl Fnv {
    fn new() -> Self {
        Fnv(0x811c_9dc5)
    }
    fn add(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 ^= b as u32;
            self.0 = self.0.wrapping_mul(0x0100_0193);
        }
    }
    fn zeros(&mut self, n: usize) {
        for _ in 0..n {
            self.0 = self.0.wrapping_mul(0x0100_0193);
        }
    }
}

/// FNV-1a, 32 bit.
pub fn fnv(bytes: &[u8]) -> u32 {
    let mut h = Fnv::new();
    h.add(bytes);
    h.0
}

/// Constant buffer 0 ([`CBUF_SIZE`] bytes) and the QMD (grid [⌈T/64⌉, H, B], 256 threads, the program's registers,
/// shared memory and barrier) of program `prog` on set `set`, whose buffer is at `buf_va` (aligned to 4 KiB).
/// `prog_va`: where [`Prog::binary`] was loaded (aligned to 256); `cbuf_va`: where the cbuf bytes go (aligned to 256).
pub fn launch_bytes(
    set: &ProbeSet,
    prog: Prog,
    prog_va: u64,
    cbuf_va: u64,
    buf_va: u64,
) -> GpuResult<([u8; CBUF_SIZE], [u8; nvidia_rm570::QMD_SIZE])> {
    launch_common(set, prog, 1, prog_va, cbuf_va, buf_va)
}

/// [`launch_bytes`] for the second generation ([`Prog2`]): the same cbuf and grid, the QMD with the new program's
/// registers and shared memory.
pub fn launch_bytes2(
    set: &ProbeSet,
    prog: Prog2,
    prog_va: u64,
    cbuf_va: u64,
    buf_va: u64,
) -> GpuResult<([u8; CBUF_SIZE], [u8; nvidia_rm570::QMD_SIZE])> {
    launch_common(set, prog.0, 2, prog_va, cbuf_va, buf_va)
}

/// [`launch_bytes`] for the third generation ([`Prog3`]): the same cbuf and grid, the QMD with the new program's
/// registers and shared memory.
pub fn launch_bytes3(
    set: &ProbeSet,
    prog: Prog3,
    prog_va: u64,
    cbuf_va: u64,
    buf_va: u64,
) -> GpuResult<([u8; CBUF_SIZE], [u8; nvidia_rm570::QMD_SIZE])> {
    launch_common(set, prog.0, 3, prog_va, cbuf_va, buf_va)
}

/// [`launch_bytes`] for the fourth generation ([`Prog4`]): the same cbuf and grid, the QMD with the program's registers
/// and shared memory (for `fwd` / `back0` exactly [`launch_bytes3`]'s).
pub fn launch_bytes4(
    set: &ProbeSet,
    prog: Prog4,
    prog_va: u64,
    cbuf_va: u64,
    buf_va: u64,
) -> GpuResult<([u8; CBUF_SIZE], [u8; nvidia_rm570::QMD_SIZE])> {
    launch_common(set, prog.0, 4, prog_va, cbuf_va, buf_va)
}

fn launch_common(
    set: &ProbeSet,
    prog: Prog,
    generation: u8,
    prog_va: u64,
    cbuf_va: u64,
    buf_va: u64,
) -> GpuResult<([u8; CBUF_SIZE], [u8; nvidia_rm570::QMD_SIZE])> {
    if !buf_va.is_multiple_of(4096) || !cbuf_va.is_multiple_of(256) || !prog_va.is_multiple_of(256)
    {
        return Err(bad(
            "atentie proba: bufferul nealiniat la 4 KiB sau programul / cbuf nealiniate la 256",
        ));
    }
    if !set.shape_ok() {
        return Err(bad("atentie proba: forma setului"));
    }
    let mut cb = [0u8; CBUF_SIZE];
    let (bytes, grid, block): (Vec<u8>, [u32; 3], [u32; 3]) = match prog {
        Prog::Fwd => {
            let a = set.fwd_args(buf_va);
            let c = a.cbuf().ok_or(bad("atentie proba: argumente"))?;
            match generation {
                2 => (c.to_vec(), ft2::grid(&a), ft2::BLOCK),
                3 | 4 => (c.to_vec(), ft3::grid(&a), ft3::BLOCK),
                _ => (c.to_vec(), ft::grid(&a), ft::BLOCK),
            }
        }
        Prog::Rows => {
            let a = set.back_args(buf_va, Mode::Rows);
            let c = a.cbuf().ok_or(bad("atentie proba: argumente"))?;
            match generation {
                2 => (c.to_vec(), b02::grid(&a), b02::BLOCK),
                3 | 4 => (c.to_vec(), b03::grid(&a), b03::BLOCK),
                _ => (c.to_vec(), b0::grid(&a), b0::BLOCK),
            }
        }
        Prog::Cols => {
            let a = set.back_args(buf_va, Mode::Columns);
            let c = a.cbuf().ok_or(bad("atentie proba: argumente"))?;
            match generation {
                2 => (c.to_vec(), b12::grid(&a), b12::BLOCK),
                3 => (c.to_vec(), b13::grid(&a), b13::BLOCK),
                4 => (c.to_vec(), b14::grid(&a), b14::BLOCK),
                _ => (c.to_vec(), b1::grid(&a), b1::BLOCK),
            }
        }
    };
    let (registers, shared) = match generation {
        2 => (Prog2(prog).registers(), Prog2(prog).shared()),
        3 => (Prog3(prog).registers(), Prog3(prog).shared()),
        4 => (Prog4(prog).registers(), Prog4(prog).shared()),
        _ => (prog.registers(), prog.shared()),
    };
    cb[..bytes.len()].copy_from_slice(&bytes);
    let q = ComputeQmd {
        program_va: prog_va,
        registers,
        cbuf0_va: cbuf_va,
        cbuf0_size: CBUF_SIZE as u32,
        grid,
        block,
        release: None,
    }
    .to_bytes_with_shared(&shared)?;
    Ok((cb, q))
}

/// The verdict of one program on one set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every judged element within its written margin, guard untouched.
    InMargine,
    /// Some element outside its margin (written, finite).
    NuTrece,
    /// Wrong length, an element not written / not finite, or a guard word changed.
    Invalid,
}

/// What one program gave on one set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProgResult {
    pub prog: Prog,
    pub verdict: Verdict,
    /// Worst element, in ‰ of its margin (`None` when invalid or outside).
    pub worst_permille: Option<u32>,
    /// Elements bit for bit equal to the CPU model (on the board's own inputs for the backward programs).
    pub exact: u32,
    /// Elements judged (P and dS: only j ≤ i; att and dqkv: all).
    pub elements: u32,
    /// The guard words after this program's output(s) still [`FILL`].
    pub guards_ok: bool,
    /// FNV-1a of this program's output bytes read back (guards excluded).
    pub out_fnv: u32,
    /// Launch times on the board clock (ns), one per launch, if the OS measured them.
    pub ns: Vec<u64>,
}

impl ProgResult {
    pub fn passes(&self) -> bool {
        self.verdict == Verdict::InMargine
    }
    /// 1 = within the margin, 0 = outside, 2 = invalid.
    pub fn code(&self) -> u32 {
        match self.verdict {
            Verdict::InMargine => 1,
            Verdict::NuTrece => 0,
            Verdict::Invalid => 2,
        }
    }
    pub fn median_ns(&self) -> Option<u64> {
        median(&self.ns)
    }
}

fn median(v: &[u64]) -> Option<u64> {
    if v.is_empty() {
        return None;
    }
    let mut v = v.to_vec();
    v.sort_unstable();
    Some(v[v.len() / 2])
}

/// What one set gave: the three programs in launch order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetResult {
    /// Index in [`SETS`] (`usize::MAX` for a set outside the table, in tests).
    pub set: usize,
    pub progs: [ProgResult; 3],
}

impl SetResult {
    pub fn passes(&self) -> bool {
        self.progs.iter().all(ProgResult::passes)
    }
}

fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// The pieces of a read-back (or the model's), in read order: P rows, att rows, dS rows, dqkv rows.
struct Eval {
    /// The model's values of the four pieces (computed on the board's P and dS when those were given).
    model: [Vec<f32>; 4],
    /// The margin of every element (NaN = above the diagonal: must be exactly +0).
    margin: [Vec<f64>; 4],
}

/// Pieces of the board's output (P, att, dS, dqkv rows, guards stripped).
struct Got<'a> {
    p: &'a [f32],
    ds: &'a [f32],
}

/// Computes the model's pieces and margins. With `got`, the backward models use the board's P (and dS) rows.
fn evaluate(s: &ProbeSet, got: Option<&Got<'_>>) -> Eval {
    let (_, t, c, h, d) = s.dims();
    let one = AttnShape {
        batch: 1,
        t,
        c,
        heads: h,
    };
    let scale = s.scale();
    let (texts, rows, dq) = (s.texts(), s.rows(), s.dq_rows());
    let nr = rows.len();
    let mut model: [Vec<f32>; 4] = Default::default();
    let mut margin: [Vec<f64>; 4] = Default::default();
    for (bi, &b) in texts.iter().enumerate() {
        let (qkv, dout) = s.text_inputs(b);
        // Forward rows, every head, then the att rows (heads side by side).
        let mut att_rows = alloc::vec![0.0f32; nr * c];
        let mut att_m = alloc::vec![0.0f64; nr * c];
        // P and dS of this text, [H, T, T], only the judged rows filled (the models read only those).
        let mut p_b = alloc::vec![0.0f32; h * t * t];
        let mut ds_b = alloc::vec![0.0f32; h * t * t];
        for hh in 0..h {
            for (ri, &i) in rows.iter().enumerate() {
                let (prow, arow) = attn_fwd_tile_row_gpu(&one, &qkv, scale, 0, hh, i);
                for (j, &pj) in prow.iter().enumerate() {
                    model[0].push(pj);
                    margin[0].push(if j > i {
                        f64::NAN
                    } else {
                        MUFU_REL * pj as f64 + (i + 3) as f64 * FTZ
                    });
                }
                for (e, &ae) in arow.iter().enumerate() {
                    let col = hh * d + e;
                    let sum: f64 = (0..=i)
                        .map(|j| (prow[j] as f64 * qkv[j * 3 * c + 2 * c + col] as f64).abs())
                        .sum();
                    let n = (i + 1) as f64;
                    att_rows[ri * c + col] = ae;
                    att_m[ri * c + col] = (MUFU_REL + 2.0 * n * U) * sum + 2.0 * n * FTZ;
                }
                let at = (hh * t + i) * t;
                let src = match got {
                    // The board's row of P: its place in the read order.
                    Some(g) => &g.p[((bi * h + hh) * nr + ri) * t..][..t],
                    None => &prow[..],
                };
                p_b[at..at + t].copy_from_slice(src);
            }
        }
        model[1].extend_from_slice(&att_rows);
        margin[1].extend_from_slice(&att_m);
        for hh in 0..h {
            for (ri, &i) in rows.iter().enumerate() {
                let row = attn_back_row_gpu(&one, &qkv, &p_b, &dout, scale, 0, hh, i);
                let big = row.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()));
                for (j, &v) in row.iter().enumerate() {
                    model[2].push(v);
                    margin[2].push(if j > i {
                        f64::NAN
                    } else {
                        BACK_REL * big + FTZ
                    });
                }
                let at = (hh * t + i) * t;
                let src = match got {
                    Some(g) => &g.ds[((bi * h + hh) * nr + ri) * t..][..t],
                    None => &row[..],
                };
                ds_b[at..at + t].copy_from_slice(src);
            }
        }
        for &tp in &dq {
            let mut r = alloc::vec![0.0f32; 3 * c];
            for hh in 0..h {
                for e in 0..d {
                    let (q, k, v) = attn_back_col_gpu(&one, &qkv, &p_b, &dout, &ds_b, 0, hh, tp, e);
                    let col = hh * d + e;
                    r[col] = q;
                    r[c + col] = k;
                    r[2 * c + col] = v;
                }
            }
            let big = r.iter().fold(0.0f64, |m, &v| m.max((v as f64).abs()));
            model[3].extend_from_slice(&r);
            margin[3].extend(core::iter::repeat_n(BACK_REL * big + FTZ, 3 * c));
        }
    }
    Eval { model, margin }
}

/// Judges one piece: `got` against the model within the margins; returns (verdict, worst ‰, exact, elements).
fn judge_piece(got: &[u32], model: &[f32], margin: &[f64]) -> (Verdict, Option<u32>, u32, u32) {
    let (mut exact, mut elements, mut worst) = (0u32, 0u32, 0u32);
    let mut verdict = Verdict::InMargine;
    for ((&g, &m), &mg) in got.iter().zip(model).zip(margin) {
        if mg.is_nan() {
            // Above the diagonal: exactly +0.
            if g != 0 {
                return (Verdict::Invalid, None, exact, elements);
            }
            continue;
        }
        elements += 1;
        let gv = f32::from_bits(g);
        if !gv.is_finite() {
            return (Verdict::Invalid, None, exact, elements);
        }
        if g == m.to_bits() {
            exact += 1;
            continue;
        }
        let err = (gv as f64 - m as f64).abs();
        if err > mg {
            verdict = Verdict::NuTrece;
        } else if mg > 0.0 {
            let pm = (err / mg * 1000.0) as u32 + 1;
            worst = worst.max(pm.min(1000));
        }
    }
    match verdict {
        Verdict::InMargine => (verdict, Some(worst), exact, elements),
        v => (v, None, exact, elements),
    }
}

/// Judges the bytes read back for `s` ([`ProbeSet::read_ranges`], concatenated) and the launch times of the three
/// programs (in launch order). `set` is only recorded.
pub fn judge(s: &ProbeSet, set: usize, out: &[u8], ns: [&[u64]; 3]) -> GpuResult<SetResult> {
    if !s.shape_ok() {
        return Err(bad("atentie proba: forma setului"));
    }
    let (_, t, c, h, _) = s.dims();
    let (nt, nr, nd) = (s.texts().len(), s.rows().len(), s.dq_rows().len());
    let lens = [
        nt * h * nr * t,
        nt * nr * c,
        nt * h * nr * t,
        nt * nd * 3 * c,
    ];
    let blank = |prog: Prog, ns: &[u64]| ProgResult {
        prog,
        verdict: Verdict::Invalid,
        worst_permille: None,
        exact: 0,
        elements: 0,
        guards_ok: false,
        out_fnv: 0,
        ns: ns.to_vec(),
    };
    let mut res = SetResult {
        set,
        progs: [
            blank(Prog::Fwd, ns[0]),
            blank(Prog::Rows, ns[1]),
            blank(Prog::Cols, ns[2]),
        ],
    };
    if out.len() != s.read_len() {
        return Ok(res);
    }
    let w = words(out);
    // Split into the four pieces and their guards.
    let mut pieces: [Vec<u32>; 4] = Default::default();
    let mut guards = [false; 4];
    let mut at = 0;
    for k in 0..4 {
        pieces[k] = w[at..at + lens[k]].to_vec();
        at += lens[k];
        guards[k] = w[at..at + GUARD].iter().all(|&x| x == FILL);
        at += GUARD;
    }
    let fl = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
    let (p, ds) = (fl(&pieces[0]), fl(&pieces[2]));
    let e = evaluate(s, Some(&Got { p: &p, ds: &ds }));
    let bytes_of = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    // Fwd = P + att; Rows = dS; Cols = dqkv.
    let groups: [&[usize]; 3] = [&[0, 1], &[2], &[3]];
    for (pi, g) in groups.iter().enumerate() {
        let r = &mut res.progs[pi];
        r.guards_ok = g.iter().all(|&k| guards[k]);
        let mut all = Vec::new();
        for &k in g.iter() {
            all.extend_from_slice(&bytes_of(&pieces[k]));
        }
        r.out_fnv = fnv(&all);
        let mut verdict = Verdict::InMargine;
        let mut worst = Some(0u32);
        for &k in g.iter() {
            let (v, wp, ex, el) = judge_piece(&pieces[k], &e.model[k], &e.margin[k]);
            r.exact += ex;
            r.elements += el;
            match (v, verdict) {
                (Verdict::Invalid, _) | (_, Verdict::Invalid) => verdict = Verdict::Invalid,
                (Verdict::NuTrece, _) => verdict = Verdict::NuTrece,
                _ => {}
            }
            worst = match (worst, wp) {
                (Some(a), Some(b)) => Some(a.max(b)),
                _ => None,
            };
        }
        if !r.guards_ok {
            verdict = Verdict::Invalid;
        }
        r.verdict = verdict;
        r.worst_permille = if verdict == Verdict::InMargine {
            worst
        } else {
            None
        };
    }
    Ok(res)
}

/// [`judge`] for set `set` of [`SETS`].
pub fn judge_set(set: usize, out: &[u8], ns: [&[u64]; 3]) -> GpuResult<SetResult> {
    let s = SETS.get(set).ok_or(bad("atentie proba: set necunoscut"))?;
    judge(s, set, out, ns)
}

/// The whole probe (one [`SetResult`] per set run, in order).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Probe {
    pub sets: Vec<SetResult>,
}

impl Probe {
    pub fn add(&mut self, r: SetResult) {
        self.sets.push(r);
    }
    /// The timing set was not run (no ~151 MB of VRAM).
    pub fn timing_skipped(&self) -> bool {
        !self.sets.iter().any(|r| r.set == TIMING_SET)
    }
    /// Sets 0–4 all judged, every program of every set run within its margin (the timing set may be skipped).
    pub fn passes(&self) -> bool {
        (0..TIMING_SET).all(|i| self.sets.iter().any(|r| r.set == i))
            && self.sets.iter().all(SetResult::passes)
    }
    /// Sets in which all three programs are within their margins.
    pub fn in_margin(&self) -> usize {
        self.sets.iter().filter(|r| r.passes()).count()
    }
    /// Per program (launch order): exact elements, judged elements, worst ‰ (`u32::MAX` if any set had none).
    pub fn per_prog(&self) -> [(u32, u32, u32); 3] {
        let mut out = [(0u32, 0u32, 0u32); 3];
        for r in &self.sets {
            for (k, p) in r.progs.iter().enumerate() {
                out[k].0 += p.exact;
                out[k].1 += p.elements;
                out[k].2 = out[k].2.max(p.worst_permille.unwrap_or(u32::MAX));
            }
        }
        out
    }
    fn timing(&self) -> Option<&SetResult> {
        self.sets.iter().find(|r| r.set == TIMING_SET)
    }

    /// The `kt atentie …` rows (values in the stick journal, read back by [`parse_attn_tile`]).
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        let cl = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        let pp = self.per_prog();
        let mut out = alloc::vec![
            ("kt atentie seturi", self.sets.len() as u32),
            ("kt atentie seturi in margine", self.in_margin() as u32),
            ("kt atentie fwd exacte (model)", pp[0].0),
            ("kt atentie fwd elemente", pp[0].1),
            ("kt atentie fwd promile", pp[0].2),
            ("kt atentie back0 exacte (model)", pp[1].0),
            ("kt atentie back0 elemente", pp[1].1),
            ("kt atentie back0 promile", pp[1].2),
            ("kt atentie back1 exacte (model)", pp[2].0),
            ("kt atentie back1 elemente", pp[2].1),
            ("kt atentie back1 promile", pp[2].2),
            ("kt atentie timp omis", u32::from(self.timing_skipped())),
        ];
        if let Some(r) = self.timing() {
            const T: [&str; 3] = [
                "kt atentie timp fwd (ns)",
                "kt atentie timp back0 (ns)",
                "kt atentie timp back1 (ns)",
            ];
            for (k, p) in r.progs.iter().enumerate() {
                if let Some(ns) = p.median_ns() {
                    out.push((T[k], cl(ns)));
                }
            }
        }
        for r in &self.sets {
            out.push(("kt atentie set: index", r.set as u32));
            out.push((
                "kt atentie set: intrari (fnv)",
                SETS.get(r.set).map_or(0, ProbeSet::inputs_fnv),
            ));
            for p in &r.progs {
                out.extend([
                    ("kt atentie set: verdict", p.code()),
                    ("kt atentie set: exacte (model)", p.exact),
                    (
                        "kt atentie set: promile",
                        p.worst_permille.unwrap_or(u32::MAX),
                    ),
                    ("kt atentie set: garzi", u32::from(p.guards_ok)),
                    ("kt atentie set: iesire (fnv)", p.out_fnv),
                ]);
            }
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
        let pp = self.per_prog();
        let mut out = alloc::vec![format!(
            "atentie pe placi ({label}): {} / {} seturi in margine la toate trei programele",
            self.in_margin(),
            self.sets.len()
        )];
        for (k, prog) in PROGS.iter().enumerate() {
            let w = if pp[k].2 == u32::MAX {
                String::from("-")
            } else {
                format!("{}", pp[k].2)
            };
            out.push(format!(
                "atentie {} ({label}): {} / {} elemente exacte fata de model, cel mai rau {w} promile din margine",
                prog.name(),
                pp[k].0,
                pp[k].1
            ));
        }
        match self.timing() {
            None => out.push(format!(
                "atentie {} ({label}): omis (fara VRAM)",
                SETS[TIMING_SET].name
            )),
            Some(r) => {
                const EST: [&str; 3] = ["~0,53", "~0,49", "~0,42-0,68"];
                for (k, p) in r.progs.iter().enumerate() {
                    if let Some(ns) = p.median_ns() {
                        out.push(format!(
                            "atentie {} {} ({label}): mediana {ns} ns pe {} lansari (estimat {} ms); pe bucata #513 ~{} ms",
                            SETS[TIMING_SET].name,
                            p.prog.name(),
                            p.ns.len(),
                            EST[k],
                            ns * 36 / 1_000_000
                        ));
                    }
                }
            }
        }
        for r in &self.sets {
            let name = SETS.get(r.set).map_or("?", |s| s.name);
            for p in &r.progs {
                let w = p
                    .worst_permille
                    .map(|v| format!("{v}"))
                    .unwrap_or_else(|| String::from("-"));
                out.push(format!(
                    "atentie set {name} {} ({label}): {}, exacte {}/{}, {w} promile, garzi {}",
                    p.prog.name(),
                    ["NU TRECE", "in margine", "INVALID"][p.code() as usize],
                    p.exact,
                    p.elements,
                    if p.guards_ok { "neatinse" } else { "ATINSE" },
                ));
            }
        }
        out
    }
}

/// The read-back of one set split into its four pieces (P, att, dS, dqkv rows; guards stripped), or `None` for a
/// wrong length.
fn pieces_of(s: &ProbeSet, out: &[u8]) -> Option<[Vec<u32>; 4]> {
    if out.len() != s.read_len() {
        return None;
    }
    let (_, t, c, h, _) = s.dims();
    let (nt, nr, nd) = (s.texts().len(), s.rows().len(), s.dq_rows().len());
    let lens = [
        nt * h * nr * t,
        nt * nr * c,
        nt * h * nr * t,
        nt * nd * 3 * c,
    ];
    let w = words(out);
    let mut pieces: [Vec<u32>; 4] = Default::default();
    let mut at = 0;
    for k in 0..4 {
        pieces[k] = w[at..at + lens[k]].to_vec();
        at += lens[k] + GUARD;
    }
    Some(pieces)
}

/// The pieces each program writes (Fwd = P + att; Rows = dS; Cols = dqkv).
const GROUPS: [&[usize]; 3] = [&[0, 1], &[2], &[3]];

/// What the second-generation chain gave on one set, beside the first.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PairResult {
    /// The new chain judged exactly as the old one ([`judge`]: the same margins, the same models).
    pub new: SetResult,
    /// Per program (launch order): words of its read-back equal to the old chain's, words compared (guards excluded).
    /// A wrong length on either side: (0, words of the expected read-back).
    pub same: [(u32, u32); 3],
}

impl PairResult {
    /// Every compared word of every program equal to the old chain's.
    pub fn identical(&self) -> bool {
        self.same.iter().all(|&(a, b)| a == b)
    }
}

/// C-157 point 2: judges the second-generation chain on set `s` (`new_out`, read back as [`ProbeSet::read_ranges`], with
/// its launch times) and compares it word for word with the first-generation chain's read-back `old_out` on the same
/// set. `set` is only recorded.
pub fn judge_pair(
    s: &ProbeSet,
    set: usize,
    old_out: &[u8],
    new_out: &[u8],
    ns_new: [&[u64]; 3],
) -> GpuResult<PairResult> {
    let new = judge(s, set, new_out, ns_new)?;
    let (_, t, c, h, _) = s.dims();
    let (nt, nr, nd) = (s.texts().len(), s.rows().len(), s.dq_rows().len());
    let lens = [
        nt * h * nr * t,
        nt * nr * c,
        nt * h * nr * t,
        nt * nd * 3 * c,
    ];
    let mut same = [(0u32, 0u32); 3];
    let (po, pn) = (pieces_of(s, old_out), pieces_of(s, new_out));
    for (k, g) in GROUPS.iter().enumerate() {
        let total: usize = g.iter().map(|&i| lens[i]).sum();
        same[k].1 = total as u32;
        if let (Some(a), Some(b)) = (&po, &pn) {
            same[k].0 = g
                .iter()
                .map(|&i| a[i].iter().zip(&b[i]).filter(|(x, y)| x == y).count() as u32)
                .sum();
        }
    }
    Ok(PairResult { new, same })
}

/// [`judge_pair`] for set `set` of [`SETS`].
pub fn judge_pair_set(
    set: usize,
    old_out: &[u8],
    new_out: &[u8],
    ns_new: [&[u64]; 3],
) -> GpuResult<PairResult> {
    let s = SETS.get(set).ok_or(bad("atentie proba: set necunoscut"))?;
    judge_pair(s, set, old_out, new_out, ns_new)
}

/// The probe of the second generation beside the first (one [`PairResult`] per set run, in order).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Probe2 {
    pub sets: Vec<PairResult>,
}

/// Measured medians of the first generation on `atentie-513-b4` (os-358, 23 nov 2026, ns): the reference when the old
/// chain's own times are not given.
pub const OS358_NS: [u64; 3] = [2_490_000, 1_810_000, 3_990_000];

impl Probe2 {
    pub fn add(&mut self, r: PairResult) {
        self.sets.push(r);
    }
    /// The timing set was not run.
    pub fn timing_skipped(&self) -> bool {
        !self.sets.iter().any(|r| r.new.set == TIMING_SET)
    }
    /// Sets 0–4 all judged and every program of the new chain within its margin on every set run.
    pub fn passes(&self) -> bool {
        (0..TIMING_SET).all(|i| self.sets.iter().any(|r| r.new.set == i))
            && self.sets.iter().all(|r| r.new.passes())
    }
    /// Sets in which the new chain is within its margins.
    pub fn in_margin(&self) -> usize {
        self.sets.iter().filter(|r| r.new.passes()).count()
    }
    /// Sets in which every word read back equals the old chain's.
    pub fn identical_sets(&self) -> usize {
        self.sets.iter().filter(|r| r.identical()).count()
    }
    /// Every set run identical to the old chain (prediction 2) — false when nothing was run.
    pub fn identical(&self) -> bool {
        !self.sets.is_empty() && self.identical_sets() == self.sets.len()
    }
    /// Per program: exact elements, judged elements, worst ‰ (as [`Probe::per_prog`]).
    pub fn per_prog(&self) -> [(u32, u32, u32); 3] {
        Probe {
            sets: self.sets.iter().map(|r| r.new.clone()).collect(),
        }
        .per_prog()
    }
    /// Per program: words equal to the old chain's, words compared (all sets).
    pub fn same_per_prog(&self) -> [(u32, u32); 3] {
        let mut out = [(0u32, 0u32); 3];
        for r in &self.sets {
            for (o, s) in out.iter_mut().zip(&r.same) {
                o.0 += s.0;
                o.1 += s.1;
            }
        }
        out
    }
    fn timing(&self) -> Option<&PairResult> {
        self.sets.iter().find(|r| r.new.set == TIMING_SET)
    }

    /// The `kt atentie2 …` rows (read back by [`parse_attn_tile2`]; [`parse_attn_tile`] does not see them).
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        pair_rows(&self.sets, &NAMES2)
    }

    pub fn journal(&self, j: &mut Journal) {
        for (what, value) in self.rows() {
            j.push(Event::Value { what, value });
        }
    }

    /// Report lines (`label`: "SIMULARE" or "placa"). `old`: the first generation's probe from the same run (its
    /// timing-set medians are the reference); without it, the os-358 medians ([`OS358_NS`]).
    pub fn lines(&self, label: &str, old: Option<&Probe>) -> Vec<String> {
        let (old_ns, src) = match old.and_then(|p| p.timing()) {
            Some(r) => ([0, 1, 2].map(|k| r.progs[k].median_ns()), "aceeasi rulare"),
            None => (OS358_NS.map(Some), "os-358"),
        };
        pair_lines(
            &self.sets,
            &PairText {
                label,
                tag: "atentie2",
                names: PROGS2.map(Prog2::name),
                same_with: PROGS2.map(|p| p.old().name()),
                time_ref: ["vechiul"; 3],
                est: ["~0,6-1,2", "~0,7-1,1", "~1,2-2,0"],
                old_ns,
                src,
            },
        )
    }
}

/// Row names of one "old beside new" probe (`kt atentie2 …` or `kt atentie3 …`).
struct PairRowNames {
    sets: &'static str,
    in_margin: &'static str,
    identical: &'static str,
    prog: [[&'static str; 5]; 3],
    skipped: &'static str,
    time: [&'static str; 3],
    set: [&'static str; 8],
}

macro_rules! pair_row_names {
    ($g:literal) => {
        PairRowNames {
            sets: concat!("kt atentie", $g, " seturi"),
            in_margin: concat!("kt atentie", $g, " seturi in margine"),
            identical: concat!("kt atentie", $g, " seturi identice cu vechi"),
            prog: [
                [
                    concat!("kt atentie", $g, " fwd exacte (model)"),
                    concat!("kt atentie", $g, " fwd elemente"),
                    concat!("kt atentie", $g, " fwd promile"),
                    concat!("kt atentie", $g, " fwd identice"),
                    concat!("kt atentie", $g, " fwd cuvinte"),
                ],
                [
                    concat!("kt atentie", $g, " back0 exacte (model)"),
                    concat!("kt atentie", $g, " back0 elemente"),
                    concat!("kt atentie", $g, " back0 promile"),
                    concat!("kt atentie", $g, " back0 identice"),
                    concat!("kt atentie", $g, " back0 cuvinte"),
                ],
                [
                    concat!("kt atentie", $g, " back1 exacte (model)"),
                    concat!("kt atentie", $g, " back1 elemente"),
                    concat!("kt atentie", $g, " back1 promile"),
                    concat!("kt atentie", $g, " back1 identice"),
                    concat!("kt atentie", $g, " back1 cuvinte"),
                ],
            ],
            skipped: concat!("kt atentie", $g, " timp omis"),
            time: [
                concat!("kt atentie", $g, " timp fwd (ns)"),
                concat!("kt atentie", $g, " timp back0 (ns)"),
                concat!("kt atentie", $g, " timp back1 (ns)"),
            ],
            set: [
                concat!("kt atentie", $g, " set: index"),
                concat!("kt atentie", $g, " set: intrari (fnv)"),
                concat!("kt atentie", $g, " set: verdict"),
                concat!("kt atentie", $g, " set: exacte (model)"),
                concat!("kt atentie", $g, " set: promile"),
                concat!("kt atentie", $g, " set: garzi"),
                concat!("kt atentie", $g, " set: iesire (fnv)"),
                concat!("kt atentie", $g, " set: identice"),
            ],
        }
    };
}

const NAMES2: PairRowNames = pair_row_names!("2");
const NAMES3: PairRowNames = pair_row_names!("3");
const NAMES4: PairRowNames = pair_row_names!("4");

fn pair_probe(sets: &[PairResult]) -> Probe2 {
    Probe2 {
        sets: sets.to_vec(),
    }
}

fn pair_rows(sets: &[PairResult], n: &PairRowNames) -> Vec<(&'static str, u32)> {
    let p = pair_probe(sets);
    let cl = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
    let pp = p.per_prog();
    let sp = p.same_per_prog();
    let mut out = alloc::vec![
        (n.sets, p.sets.len() as u32),
        (n.in_margin, p.in_margin() as u32),
        (n.identical, p.identical_sets() as u32),
    ];
    for k in 0..3 {
        out.extend([
            (n.prog[k][0], pp[k].0),
            (n.prog[k][1], pp[k].1),
            (n.prog[k][2], pp[k].2),
            (n.prog[k][3], sp[k].0),
            (n.prog[k][4], sp[k].1),
        ]);
    }
    out.push((n.skipped, u32::from(p.timing_skipped())));
    if let Some(r) = p.timing() {
        for (k, pr) in r.new.progs.iter().enumerate() {
            if let Some(ns) = pr.median_ns() {
                out.push((n.time[k], cl(ns)));
            }
        }
    }
    for r in &p.sets {
        out.push((n.set[0], r.new.set as u32));
        out.push((
            n.set[1],
            SETS.get(r.new.set).map_or(0, ProbeSet::inputs_fnv),
        ));
        for (k, pr) in r.new.progs.iter().enumerate() {
            out.extend([
                (n.set[2], pr.code()),
                (n.set[3], pr.exact),
                (n.set[4], pr.worst_permille.unwrap_or(u32::MAX)),
                (n.set[5], u32::from(pr.guards_ok)),
                (n.set[6], pr.out_fnv),
                (n.set[7], r.same[k].0),
            ]);
        }
    }
    out
}

/// The words of one "old beside new" report.
struct PairText<'a> {
    label: &'a str,
    tag: &'static str,
    names: [&'static str; 3],
    /// The programs the read-back is compared with word for word.
    same_with: [&'static str; 3],
    /// What the timing-set median is compared with.
    time_ref: [&'static str; 3],
    est: [&'static str; 3],
    old_ns: [Option<u64>; 3],
    src: &'static str,
}

fn pair_lines(sets: &[PairResult], t: &PairText) -> Vec<String> {
    let p = pair_probe(sets);
    let (label, tag) = (t.label, t.tag);
    let pp = p.per_prog();
    let sp = p.same_per_prog();
    let mut out = alloc::vec![format!(
        "{tag} vechi langa nou ({label}): {} / {} seturi in margine, {} / {} seturi identice bit cu bit cu programele vechi",
        p.in_margin(),
        p.sets.len(),
        p.identical_sets(),
        p.sets.len()
    )];
    for k in 0..3 {
        let w = if pp[k].2 == u32::MAX {
            String::from("-")
        } else {
            format!("{}", pp[k].2)
        };
        out.push(format!(
            "{tag} {} ({label}): {} / {} elemente exacte fata de model, cel mai rau {w} promile; {} / {} cuvinte identice cu {}",
            t.names[k], pp[k].0, pp[k].1, sp[k].0, sp[k].1, t.same_with[k]
        ));
    }
    match p.timing() {
        None => out.push(format!("{tag} {} ({label}): omis", SETS[TIMING_SET].name)),
        Some(r) => {
            for (k, pr) in r.new.progs.iter().enumerate() {
                if let Some(ns) = pr.median_ns() {
                    let vs = match t.old_ns[k] {
                        Some(o) if ns > 0 => format!(
                            "; {} {o} ns ({}), de {},{:02} ori mai rapid",
                            t.time_ref[k],
                            t.src,
                            o / ns,
                            (o % ns) * 100 / ns
                        ),
                        _ => String::new(),
                    };
                    out.push(format!(
                        "{tag} {} {} ({label}): mediana {ns} ns pe {} lansari (estimat {} ms){vs}",
                        SETS[TIMING_SET].name,
                        t.names[k],
                        pr.ns.len(),
                        t.est[k],
                    ));
                }
            }
        }
    }
    for r in &p.sets {
        let name = SETS.get(r.new.set).map_or("?", |s| s.name);
        for (k, pr) in r.new.progs.iter().enumerate() {
            let w = pr
                .worst_permille
                .map(|v| format!("{v}"))
                .unwrap_or_else(|| String::from("-"));
            out.push(format!(
                "{tag} set {name} {} ({label}): {}, exacte {}/{}, {w} promile, garzi {}, identice {}/{}",
                t.names[k],
                ["NU TRECE", "in margine", "INVALID"][pr.code() as usize],
                pr.exact,
                pr.elements,
                if pr.guards_ok { "neatinse" } else { "ATINSE" },
                r.same[k].0,
                r.same[k].1,
            ));
        }
    }
    out
}

/// The probe of the third generation (`attn_*_3`) beside the first two: one [`PairResult`] per set run, in order, each
/// from [`judge_pair`] with the **first** generation's read-back as `old_out` (the reference of the margins; os-364
/// found the second generation identical to it on the board).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Probe3 {
    pub sets: Vec<PairResult>,
}

/// Measured medians of the second generation on `atentie-513-b4` (os-364, 23 nov 2026, ns): the reference when the
/// second chain's own times are not given.
pub const OS364_NS: [u64; 3] = [1_530_000, 1_540_000, 2_250_000];

impl Probe3 {
    pub fn add(&mut self, r: PairResult) {
        self.sets.push(r);
    }
    /// The timing set was not run.
    pub fn timing_skipped(&self) -> bool {
        pair_probe(&self.sets).timing_skipped()
    }
    /// Sets 0–4 all judged and every program of the third chain within its margin on every set run.
    pub fn passes(&self) -> bool {
        pair_probe(&self.sets).passes()
    }
    /// Sets in which the third chain is within its margins.
    pub fn in_margin(&self) -> usize {
        pair_probe(&self.sets).in_margin()
    }
    /// Sets in which every word read back equals the first chain's.
    pub fn identical_sets(&self) -> usize {
        pair_probe(&self.sets).identical_sets()
    }
    /// Every set run identical to the first chain — false when nothing was run.
    pub fn identical(&self) -> bool {
        pair_probe(&self.sets).identical()
    }
    /// Per program: exact elements, judged elements, worst ‰ (as [`Probe::per_prog`]).
    pub fn per_prog(&self) -> [(u32, u32, u32); 3] {
        pair_probe(&self.sets).per_prog()
    }
    /// Per program: words equal to the first chain's, words compared (all sets).
    pub fn same_per_prog(&self) -> [(u32, u32); 3] {
        pair_probe(&self.sets).same_per_prog()
    }

    /// The `kt atentie3 …` rows (read back by [`parse_attn_tile3`]; [`parse_attn_tile`] and [`parse_attn_tile2`] do
    /// not see them).
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        pair_rows(&self.sets, &NAMES3)
    }

    pub fn journal(&self, j: &mut Journal) {
        for (what, value) in self.rows() {
            j.push(Event::Value { what, value });
        }
    }

    /// Report lines (`label`: "SIMULARE" or "placa"). `prev`: the second generation's probe from the same run (its
    /// timing-set medians are the reference); without it, the os-364 medians ([`OS364_NS`]).
    pub fn lines(&self, label: &str, prev: Option<&Probe2>) -> Vec<String> {
        let (old_ns, src) = match prev.and_then(|p| p.timing()) {
            Some(r) => (
                [0, 1, 2].map(|k| r.new.progs[k].median_ns()),
                "aceeasi rulare",
            ),
            None => (OS364_NS.map(Some), "os-364"),
        };
        pair_lines(
            &self.sets,
            &PairText {
                label,
                tag: "atentie3",
                names: PROGS3.map(Prog3::name),
                same_with: PROGS3.map(|p| p.old().name()),
                time_ref: PROGS3.map(|p| p.prev().name()),
                est: ["~0,7-1,1", "~0,5-0,9", "~1,6-2,0"],
                old_ns,
                src,
            },
        )
    }
}

/// The probe of the fourth generation (`attn_fwd_tile3` → `attn_back_tile0_3` → `attn_back_tile_4`) beside the third:
/// one [`PairResult`] per set run, in order, each from [`judge_pair`] with the **first** generation's read-back as
/// `old_out` (the reference of the margins; os-375 found the third generation identical to it on the board).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Probe4 {
    pub sets: Vec<PairResult>,
}

/// Measured medians of the third generation on `atentie-513-b4` (os-375, 23 nov 2026, ns): the reference when the
/// third chain's own times are not given.
pub const OS375_NS: [u64; 3] = [1_000_000, 830_000, 1_597_000];

impl Probe4 {
    pub fn add(&mut self, r: PairResult) {
        self.sets.push(r);
    }
    /// The timing set was not run.
    pub fn timing_skipped(&self) -> bool {
        pair_probe(&self.sets).timing_skipped()
    }
    /// Sets 0–4 all judged and every program of the fourth chain within its margin on every set run.
    pub fn passes(&self) -> bool {
        pair_probe(&self.sets).passes()
    }
    /// Sets in which the fourth chain is within its margins.
    pub fn in_margin(&self) -> usize {
        pair_probe(&self.sets).in_margin()
    }
    /// Sets in which every word read back equals the first chain's.
    pub fn identical_sets(&self) -> usize {
        pair_probe(&self.sets).identical_sets()
    }
    /// Every set run identical to the first chain — false when nothing was run.
    pub fn identical(&self) -> bool {
        pair_probe(&self.sets).identical()
    }
    /// Per program: exact elements, judged elements, worst ‰ (as [`Probe::per_prog`]).
    pub fn per_prog(&self) -> [(u32, u32, u32); 3] {
        pair_probe(&self.sets).per_prog()
    }
    /// Per program: words equal to the first chain's, words compared (all sets).
    pub fn same_per_prog(&self) -> [(u32, u32); 3] {
        pair_probe(&self.sets).same_per_prog()
    }

    /// The `kt atentie4 …` rows (read back by [`parse_attn_tile4`]; the readers of the first three generations do not
    /// see them).
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        pair_rows(&self.sets, &NAMES4)
    }

    pub fn journal(&self, j: &mut Journal) {
        for (what, value) in self.rows() {
            j.push(Event::Value { what, value });
        }
    }

    /// Report lines (`label`: "SIMULARE" or "placa"). `prev`: the third generation's probe from the same run (its
    /// timing-set medians are the reference); without it, the os-375 medians ([`OS375_NS`]).
    pub fn lines(&self, label: &str, prev: Option<&Probe3>) -> Vec<String> {
        let prev = prev.map(|p| pair_probe(&p.sets));
        let (old_ns, src) = match prev.as_ref().and_then(Probe2::timing) {
            Some(r) => (
                [0, 1, 2].map(|k| r.new.progs[k].median_ns()),
                "aceeasi rulare",
            ),
            None => (OS375_NS.map(Some), "os-375"),
        };
        pair_lines(
            &self.sets,
            &PairText {
                label,
                tag: "atentie4",
                names: PROGS4.map(Prog4::name),
                same_with: PROGS4.map(|p| p.old().name()),
                time_ref: PROGS4.map(|p| p.prev().name()),
                est: ["~0,95-1,05", "~0,79-0,87", "~1,2-1,35"],
                old_ns,
                src,
            },
        )
    }
}

/// What [`parse_attn_tile2`] reads back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedAttnTile2 {
    pub sets: Option<u32>,
    pub in_margin: Option<u32>,
    pub identical_sets: Option<u32>,
    /// Per program: (exact, elements, ‰, identical words, words).
    pub per_prog: [[Option<u32>; 5]; 3],
    pub timing_skipped: Option<u32>,
    /// Median ns per program on the timing set.
    pub timing_ns: [Option<u32>; 3],
    /// Per set: (index, inputs fnv, then for each program: verdict, exact, ‰, guards, output fnv, identical words).
    pub per_set: Vec<[u32; 20]>,
}

/// What [`parse_attn_tile3`] reads back (the same fields as for the second generation).
pub type ParsedAttnTile3 = ParsedAttnTile2;

/// Reads the rows of [`Probe2::rows`] (`gpu: valoare kt atentie2 … = 0x…`). A new `kt atentie2 seturi` row starts over.
pub fn parse_attn_tile2(text: &str) -> ParsedAttnTile2 {
    parse_pair(text, "valoare kt atentie2 ")
}

/// Reads the rows of [`Probe3::rows`] (`gpu: valoare kt atentie3 … = 0x…`). A new `kt atentie3 seturi` row starts over.
pub fn parse_attn_tile3(text: &str) -> ParsedAttnTile3 {
    parse_pair(text, "valoare kt atentie3 ")
}

/// What [`parse_attn_tile4`] reads back (the same fields as for the second generation).
pub type ParsedAttnTile4 = ParsedAttnTile2;

/// Reads the rows of [`Probe4::rows`] (`gpu: valoare kt atentie4 … = 0x…`). A new `kt atentie4 seturi` row starts over.
pub fn parse_attn_tile4(text: &str) -> ParsedAttnTile4 {
    parse_pair(text, "valoare kt atentie4 ")
}

fn parse_pair(text: &str, key: &str) -> ParsedAttnTile2 {
    let mut p = ParsedAttnTile2::default();
    let mut slot = 0usize;
    for line in text.lines() {
        let Some(rest) = line.split(key).nth(1) else {
            continue;
        };
        let Some((name, val)) = rest.rsplit_once(" = 0x") else {
            continue;
        };
        let Ok(v) = u32::from_str_radix(val.trim(), 16) else {
            continue;
        };
        let pr = |name: &str| match name {
            "fwd" => Some(0),
            "back0" => Some(1),
            "back1" => Some(2),
            _ => None,
        };
        match name {
            "seturi" => {
                p = ParsedAttnTile2 {
                    sets: Some(v),
                    ..ParsedAttnTile2::default()
                }
            }
            "seturi in margine" => p.in_margin = Some(v),
            "seturi identice cu vechi" => p.identical_sets = Some(v),
            "timp omis" => p.timing_skipped = Some(v),
            "set: index" => {
                let mut row = [0u32; 20];
                row[0] = v;
                p.per_set.push(row);
                slot = 1;
            }
            "set: intrari (fnv)"
            | "set: verdict"
            | "set: exacte (model)"
            | "set: promile"
            | "set: garzi"
            | "set: iesire (fnv)"
            | "set: identice" => {
                if let Some(row) = p.per_set.last_mut()
                    && slot < 20
                {
                    row[slot] = v;
                    slot += 1;
                }
            }
            _ => {
                if let Some(t) = name
                    .strip_prefix("timp ")
                    .and_then(|r| r.strip_suffix(" (ns)"))
                    .and_then(pr)
                {
                    p.timing_ns[t] = Some(v);
                } else if let Some((prog, what)) = name.split_once(' ')
                    && let Some(k) = pr(prog)
                {
                    let i = match what {
                        "exacte (model)" => 0,
                        "elemente" => 1,
                        "promile" => 2,
                        "identice" => 3,
                        "cuvinte" => 4,
                        _ => continue,
                    };
                    p.per_prog[k][i] = Some(v);
                }
            }
        }
    }
    p
}

/// What [`parse_attn_tile`] reads back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedAttnTile {
    pub sets: Option<u32>,
    pub in_margin: Option<u32>,
    /// Per program: (exact, elements, ‰).
    pub per_prog: [[Option<u32>; 3]; 3],
    pub timing_skipped: Option<u32>,
    /// Median ns per program on the timing set.
    pub timing_ns: [Option<u32>; 3],
    /// Per set: (index, inputs fnv, then for each program: verdict, exact, ‰, guards, output fnv).
    pub per_set: Vec<[u32; 17]>,
}

/// Reads the rows of [`Probe::rows`] (`gpu: valoare kt atentie … = 0x…`). A new `kt atentie seturi` row starts over
/// (the last run in a journal is kept).
pub fn parse_attn_tile(text: &str) -> ParsedAttnTile {
    let mut p = ParsedAttnTile::default();
    let mut slot = 0usize;
    for line in text.lines() {
        let Some(rest) = line.split("valoare kt atentie ").nth(1) else {
            continue;
        };
        let Some((name, val)) = rest.rsplit_once(" = 0x") else {
            continue;
        };
        let Ok(v) = u32::from_str_radix(val.trim(), 16) else {
            continue;
        };
        let pr = |name: &str| match name {
            "fwd" => Some(0),
            "back0" => Some(1),
            "back1" => Some(2),
            _ => None,
        };
        match name {
            "seturi" => {
                p = ParsedAttnTile {
                    sets: Some(v),
                    ..ParsedAttnTile::default()
                }
            }
            "seturi in margine" => p.in_margin = Some(v),
            "timp omis" => p.timing_skipped = Some(v),
            "set: index" => {
                let mut row = [0u32; 17];
                row[0] = v;
                p.per_set.push(row);
                slot = 1;
            }
            "set: intrari (fnv)"
            | "set: verdict"
            | "set: exacte (model)"
            | "set: promile"
            | "set: garzi"
            | "set: iesire (fnv)" => {
                if let Some(row) = p.per_set.last_mut()
                    && slot < 17
                {
                    row[slot] = v;
                    slot += 1;
                }
            }
            _ => {
                if let Some(t) = name
                    .strip_prefix("timp ")
                    .and_then(|r| r.strip_suffix(" (ns)"))
                    .and_then(pr)
                {
                    p.timing_ns[t] = Some(v);
                } else if let Some((prog, what)) = name.split_once(' ')
                    && let Some(k) = pr(prog)
                {
                    match what {
                        "exacte (model)" => p.per_prog[k][0] = Some(v),
                        "elemente" => p.per_prog[k][1] = Some(v),
                        "promile" => p.per_prog[k][2] = Some(v),
                        _ => {}
                    }
                }
            }
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sets_take_all_three_programs_and_the_timing_set_is_about_151_mb() {
        for s in &SETS {
            assert!(s.shape_ok(), "{}", s.name);
            let l = s.layout();
            for o in [l.qkv, l.dout, l.p, l.att, l.ds, l.dqkv] {
                assert_eq!(o % 256, 0, "{}", s.name);
            }
            assert!(l.end <= l.bytes && l.bytes % 4096 == 0);
        }
        let b = SETS[TIMING_SET].layout().bytes;
        assert!(150_000_000 < b && b < 152_000_000, "{b}"); // ~151 MB (144 MiB)
        assert!(
            SETS[..TIMING_SET]
                .iter()
                .all(|s| s.layout().bytes <= 3 << 20)
        );
        // ~0,34 MB read back from the timing set.
        let r = SETS[TIMING_SET].read_len();
        assert!(330_000 < r && r < 340_000, "{r}");
    }

    #[test]
    fn ranges_are_inside_ordered_and_merged() {
        for s in &SETS {
            let l = s.layout();
            let r = s.read_ranges();
            for w in r.windows(2) {
                assert!(w[0].0 + w[0].1 < w[1].0, "{}: overlap or unmerged", s.name);
            }
            assert!(r.iter().all(|&(o, n)| o >= l.p && o + n <= l.end));
            if s.bands.is_none() {
                assert!(r.len() <= 8, "{}", s.name);
            }
        }
    }

    #[test]
    fn inputs_stream_and_sets_differ() {
        let f: Vec<u32> = SETS[..TIMING_SET]
            .iter()
            .map(ProbeSet::inputs_fnv)
            .collect();
        for i in 0..f.len() {
            for j in i + 1..f.len() {
                assert_ne!(f[i], f[j]);
            }
        }
        for s in &SETS[..TIMING_SET] {
            let b = s.buffer();
            assert_eq!(fnv(&b[..s.layout().p]), s.inputs_fnv(), "{}", s.name);
        }
    }

    #[test]
    fn launch_bytes_checks_alignment_and_declares_shared_memory() {
        let s = &SETS[1];
        for prog in PROGS {
            let (cb, q) = launch_bytes(s, prog, 0x1000, 0x2000, 0x10_000).unwrap();
            let at = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
            assert_eq!(at(0x28), 0x10_000 + s.layout().qkv as u64);
            assert_eq!(at(0x30), 0x10_000 + s.layout().p as u64);
            let plain = ComputeQmd {
                program_va: 0x1000,
                registers: prog.registers(),
                cbuf0_va: 0x2000,
                cbuf0_size: CBUF_SIZE as u32,
                grid: [3, 2, 1],
                block: [256, 1, 1],
                release: None,
            }
            .to_bytes()
            .unwrap();
            assert_ne!(q, plain, "{}: shared memory must be declared", prog.name());
        }
        // Mode words: 0 rows, 1 columns.
        let (cb, _) = launch_bytes(s, Prog::Rows, 0x1000, 0x2000, 0x10_000).unwrap();
        assert_eq!(cb[0x60], 0);
        let (cb, _) = launch_bytes(s, Prog::Cols, 0x1000, 0x2000, 0x10_000).unwrap();
        assert_eq!(cb[0x60], 1);
        assert!(launch_bytes(s, Prog::Fwd, 0x1000, 0x2000, 0x10_200).is_err());
        assert!(launch_bytes(s, Prog::Fwd, 0x1000, 0x2010, 0x10_000).is_err());
        assert!(launch_bytes(s, Prog::Fwd, 0x1010, 0x2000, 0x10_000).is_err());
        let wide = ProbeSet {
            c: 130,
            heads: 1,
            ..SETS[1]
        };
        assert!(launch_bytes(&wide, Prog::Fwd, 0x1000, 0x2000, 0x10_000).is_err());
    }

    #[test]
    fn model_output_passes_exactly_and_unwritten_is_invalid() {
        for (i, s) in SETS[..TIMING_SET].iter().enumerate() {
            let r = judge_set(i, &s.model_out(), [&[], &[], &[]]).unwrap();
            assert!(r.passes(), "{}: {r:?}", s.name);
            for p in &r.progs {
                assert_eq!(p.exact, p.elements, "{} {}", s.name, p.prog.name());
                assert_eq!(p.worst_permille, Some(0));
            }
            let b = s.buffer();
            let filled: Vec<u8> = s
                .read_ranges()
                .iter()
                .flat_map(|&(o, n)| b[o..o + n].to_vec())
                .collect();
            let r = judge_set(i, &filled, [&[], &[], &[]]).unwrap();
            assert!(r.progs.iter().all(|p| p.verdict == Verdict::Invalid));
        }
        assert!(judge_set(9, &[], [&[], &[], &[]]).is_err());
        let r = judge_set(0, &[0; 8], [&[], &[], &[]]).unwrap();
        assert!(r.progs.iter().all(|p| p.code() == 2));
    }
}
