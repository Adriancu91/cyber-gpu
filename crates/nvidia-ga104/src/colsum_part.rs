//! C-160 point 7 — the column sum split over rows (`docs/C-160_P7_COLSUM_IMPARTIT.md`), after os-349 showed
//! [`crate::colsum`] bit for bit on the board but latency bound (one thread adds every row of its column).
//!
//! Pass 1, `tools/nak/colsum_part.comp` (Mesa NAK 25.0.7, SM 8.6): grid ⌈c/256⌉ × G, G = ⌈rows / chunk⌉; block row g
//! adds rows g·chunk … min((g+1)·chunk, rows) − 1 in order, with exactly `colsum`'s operations, into `p0[g·c + k]`
//! (and `p1` for LayerNorm). Pass 2 is `colsum` itself in bias mode over the G rows of partials (`p0 → s0`,
//! `p1 → s1`): [`Plan::launches`] gives the 2 (bias) or 3 (LayerNorm) launches in order. CPU model:
//! `gpu_compute::layer::colsum_split_gpu`; margin: `gpu_compute::train::colsum_split_margin`.
//! Tested **only in SIMULARE** (T170).

use crate::colsum::{self, Mode};

/// `tools/nak/out/colsum_part.sm86.sass` (Mesa NAK 25.0.7, SM 8.6).
pub const COLSUM_PART: &[u8] = include_bytes!("../../../tools/nak/out/colsum_part.sm86.sass");
/// SHA-256 of [`COLSUM_PART`] (checked by a test).
pub const COLSUM_PART_SHA256: &str =
    "740f5d439b357d37c06685384cdfd4fb7ba5cbda84c3023f49a77565d59725ac";
/// Registers per thread (NAK dump `colsum_part.sm86.txt`).
pub const REGISTERS: u32 = 20;
/// Threads per block: 256, one column each.
pub const BLOCK: [u32; 3] = [256, 1, 1];
/// Bytes of constant buffer 0 the program reads (push constants end at 0x68).
pub const CBUF_BYTES: usize = 0x68;

/// One pass-1 launch: GPU virtual addresses, the shape and the rows per chunk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Args {
    /// `[rows, c]` f32.
    pub dy: u64,
    /// `[rows, c]` f32 ([`Mode::LayerNorm`]).
    pub x: u64,
    /// `[rows]` f32 μ ([`Mode::LayerNorm`]).
    pub mean: u64,
    /// `[rows]` f32 rstd ([`Mode::LayerNorm`]).
    pub rstd: u64,
    /// `[G, c]` f32 partials, written.
    pub p0: u64,
    /// `[G, c]` f32 partials, written ([`Mode::LayerNorm`]).
    pub p1: u64,
    pub rows: u32,
    pub c: u32,
    pub mode: Mode,
    /// Rows per chunk (≥ 1).
    pub chunk: u32,
}

impl Args {
    /// G = ⌈rows / chunk⌉ (0 for an invalid chunk).
    pub fn groups(&self) -> u32 {
        if self.chunk == 0 {
            0
        } else {
            self.rows.div_ceil(self.chunk)
        }
    }

    /// Constant buffer 0: `colsum`'s layout (dy/x/μ/rstd/p0/p1 at 0x28…0x50, rows 0x58, c 0x5c, mode 0x60) plus the
    /// chunk at 0x64. `None` for an empty launch, chunk 0, or an address the mode needs missing.
    pub fn cbuf(&self) -> Option<[u8; CBUF_BYTES]> {
        if self.chunk == 0 {
            return None;
        }
        let base = colsum::Args {
            dy: self.dy,
            x: self.x,
            mean: self.mean,
            rstd: self.rstd,
            s0: self.p0,
            s1: self.p1,
            rows: self.rows,
            c: self.c,
            mode: self.mode,
        }
        .cbuf()?;
        let mut cb = [0u8; CBUF_BYTES];
        cb[..colsum::CBUF_BYTES].copy_from_slice(&base);
        cb[0x64..0x68].copy_from_slice(&self.chunk.to_le_bytes());
        Some(cb)
    }

    /// The grid: ⌈c/256⌉ × G.
    pub fn grid(&self) -> [u32; 3] {
        [self.c.div_ceil(256), self.groups(), 1]
    }
}

/// One launch of the split sum, in order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Launch {
    /// Pass 1: `colsum_part` with these arguments.
    Part(Args),
    /// Pass 2: `colsum` (bias mode) over the partials.
    Sum(colsum::Args),
}

/// The whole split sum: pass 1 into the partials, then `colsum` over them into `s0` (and `s1`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plan {
    pub part: Args,
    /// `[c]` f32, written (dγ or the bias gradient).
    pub s0: u64,
    /// `[c]` f32, written (dβ, [`Mode::LayerNorm`]).
    pub s1: u64,
}

impl Plan {
    /// The launches in order: `Part`, `Sum(p0 → s0)`, and for LayerNorm `Sum(p1 → s1)`. `None` if any of them is
    /// invalid (see [`Args::cbuf`]; `s0` / `s1` missing).
    pub fn launches(&self) -> Option<([Launch; 3], usize)> {
        let p = self.part;
        p.cbuf()?;
        let sum = |from: u64, to: u64| colsum::Args {
            dy: from,
            x: 0,
            mean: 0,
            rstd: 0,
            s0: to,
            s1: 0,
            rows: p.groups(),
            c: p.c,
            mode: Mode::Bias,
        };
        let a = sum(p.p0, self.s0);
        a.cbuf()?;
        let ln = p.mode == Mode::LayerNorm;
        let b = sum(p.p1, self.s1);
        if ln {
            b.cbuf()?;
        }
        Some((
            [Launch::Part(p), Launch::Sum(a), Launch::Sum(b)],
            if ln { 3 } else { 2 },
        ))
    }
}

/// A chunk for `rows` rows: ⌈√rows⌉ rounded up to a multiple of 8 (pass 1 and pass 2 then add about as many rows
/// each), at least 1. DEDUCED, to be tuned on the board.
pub fn default_chunk(rows: u32) -> u32 {
    let mut s = 1u32;
    while s.saturating_mul(s) < rows {
        s += 1;
    }
    if rows <= 8 {
        rows.max(1)
    } else {
        s.div_ceil(8) * 8
    }
}
