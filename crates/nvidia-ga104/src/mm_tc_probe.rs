//! C-160 point 7, part four — **the probe that sends [`crate::mm_tc`] (the training matmul on the tensor cores) to
//! the real board alone**, before it enters any training step, and judges what comes back
//! (plan, margin and predictions written before this code: `docs/C-160_P7_MM_TC_PROBA.md`).
//!
//! - [`SETS`]: five small fixed products (every flag combination the step uses, partial blocks, values at the binary16
//!   limits) and one **timing** product, `qkv-513` (6 144 × 2 304 × 768, the forward qkv of one #513 chunk).
//! - The inputs come from a hash of (seed, element) ([`ProbeSet::a_at`], [`ProbeSet::b_at`], [`ProbeSet::bias_at`]),
//!   so the OS can write them piece by piece ([`ProbeSet::write_inputs`]) without holding an 83 MB matrix.
//! - One buffer per set ([`ProbeSet::layout`]): A, B, bias, C, then [`GUARD`] words; C and the guard are filled with
//!   NaN ([`FILL`]) before the launch ([`ProbeSet::fill_ranges`]), so an element the board did not write cannot pass
//!   and a write past C is seen.
//! - [`launch_bytes`]: cbuf 0 (0x800 bytes, arguments at 0x700) and the QMD with the 24 KiB of shared memory and the
//!   barrier ([`crate::mm_tc::SHARED`]).
//! - What is read back: all of C + guard for the small sets; for `qkv-513` three bands of 16 rows ([`QKV_BANDS`]) +
//!   the guard ([`ProbeSet::read_ranges`]).
//! - [`judge_set`] → [`SetResult`]: `gpu_compute::mm_tc::judge` (the reference on the f16-rounded inputs, margin
//!   written in T168), the guards, and how many elements are **bit for bit** the CPU model `gpu_compute::mm_tc::model`
//!   (the tensor-core adder fitted on os-335 — a hypothesis the board will confirm or not).
//! - [`Probe`]: the run, `kt mm_tc …` journal rows ([`Probe::rows`]), report lines, [`parse_mm_tc`].
//!
//! Tested **only in SIMULARE** (T171, `tests/sim_c160_p7_mm_tc_probe.rs`); nothing here has run on a board.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::mm_tc::{self as oracle, Product};
use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use crate::mm_tc::{self as tc, BIAS, TA, TB};
use crate::qmd::ComputeQmd;

/// Words after C, filled with [`FILL`], that must stay untouched.
pub const GUARD: usize = 256;
/// The word C and the guard are filled with before a launch (a quiet NaN).
pub const FILL: u32 = 0x7fc0_0000;
/// Rows per band read back from a large set (one 16-row tile of C).
pub const BAND: usize = 16;
/// The first rows of the bands read back from `qkv-513` (start, middle, last tile).
pub const QKV_BANDS: [usize; 3] = [0, 3072, 6128];
/// Launches of the timing set (same inputs; the time row is their median).
pub const QKV_REPEATS: u32 = 5;
/// Index of the timing set in [`SETS`].
pub const QKV_SET: usize = 5;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// How a set's values are drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// A, B and bias uniform in [−2, 2).
    Normal,
    /// op(A) rows by `row % 4`: [−2, 2) / tiny (|x| < 2⁻¹⁶: f16 subnormals, some → 0) / large (|x| ≤ 65 504) /
    /// alternating [−2, 2) and |x| < 2⁻²⁰ along K; A's first stored element is 65 504 exactly. B and bias in [−2, 2).
    Limits,
    /// Activations in [−2, 2), weights and bias in [−0,05, 0,05) (the forward qkv of #513).
    Qkv,
}

/// One probe set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeSet {
    pub name: &'static str,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub flags: u32,
    pub seed: u64,
    pub scale: Scale,
    /// Rows read back and judged: `None` = all of C; `Some(starts)` = bands of [`BAND`] rows.
    pub bands: Option<&'static [usize]>,
    /// Launches of this set (the time row is their median).
    pub repeats: u32,
}

/// The probe: six products of the same program.
pub const SETS: [ProbeSet; 6] = [
    ProbeSet {
        name: "16x16x16",
        m: 16,
        n: 16,
        k: 16,
        flags: 0,
        seed: 1,
        scale: Scale::Normal,
        bands: None,
        repeats: 1,
    },
    ProbeSet {
        name: "70x130x33-tb-bias",
        m: 70,
        n: 130,
        k: 33,
        flags: TB | BIAS,
        seed: 2,
        scale: Scale::Normal,
        bands: None,
        repeats: 1,
    },
    ProbeSet {
        name: "64x64x96-ta",
        m: 64,
        n: 64,
        k: 96,
        flags: TA,
        seed: 3,
        scale: Scale::Normal,
        bands: None,
        repeats: 1,
    },
    ProbeSet {
        name: "100x72x200-ta-tb",
        m: 100,
        n: 72,
        k: 200,
        flags: TA | TB,
        seed: 4,
        scale: Scale::Normal,
        bands: None,
        repeats: 1,
    },
    ProbeSet {
        name: "32x48x64-limite-f16",
        m: 32,
        n: 48,
        k: 64,
        flags: TB | BIAS,
        seed: 5,
        scale: Scale::Limits,
        bands: None,
        repeats: 1,
    },
    ProbeSet {
        name: "qkv-513",
        m: 6144,
        n: 2304,
        k: 768,
        flags: TB | BIAS,
        seed: 6,
        scale: Scale::Qkv,
        bands: Some(&QKV_BANDS),
        repeats: QKV_REPEATS,
    },
];

/// Byte offsets inside the buffer of one set (every region aligned to 256 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Layout {
    pub a: usize,
    pub b: usize,
    /// `n` words (written for every set; read by the program only with [`BIAS`]).
    pub bias: usize,
    pub c: usize,
    /// The guard: [`GUARD`] words right after C.
    pub guard: usize,
    /// End of the guard.
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

/// Uniform in [−1, 1) on a 2⁻²³ grid, from (seed, which matrix, element).
fn unit(seed: u64, which: u64, idx: usize) -> f32 {
    let h = splitmix(seed.wrapping_mul(0x1_0000_0001) ^ (which << 56) ^ idx as u64);
    ((h >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

impl ProbeSet {
    fn dims(&self) -> (usize, usize, usize) {
        (self.m as usize, self.n as usize, self.k as usize)
    }

    pub fn layout(&self) -> Layout {
        let (m, n, k) = self.dims();
        let a = 0;
        let b = up(a + 4 * m * k, 256);
        let bias = up(b + 4 * k * n, 256);
        let c = up(bias + 4 * n, 256);
        let guard = c + 4 * m * n;
        let end = guard + 4 * GUARD;
        Layout {
            a,
            b,
            bias,
            c,
            guard,
            end,
            bytes: up(end, 4096),
        }
    }

    /// Stored A element `idx` (`[m, k]`, or `[k, m]` with [`TA`]).
    pub fn a_at(&self, idx: usize) -> f32 {
        let u = unit(self.seed, 1, idx);
        match self.scale {
            Scale::Normal | Scale::Qkv => 2.0 * u,
            Scale::Limits => {
                if idx == 0 {
                    return 65504.0;
                }
                let (m, _, k) = self.dims();
                let (row, q) = if self.flags & TA != 0 {
                    (idx % m, idx / m)
                } else {
                    (idx / k, idx % k)
                };
                match row % 4 {
                    0 => 2.0 * u,
                    1 => u / 65536.0,
                    2 => u * 65504.0,
                    _ => {
                        if q % 2 == 0 {
                            2.0 * u
                        } else {
                            u / (1u32 << 20) as f32
                        }
                    }
                }
            }
        }
    }

    /// Stored B element `idx` (`[k, n]`, or `[n, k]` with [`TB`]).
    pub fn b_at(&self, idx: usize) -> f32 {
        let u = unit(self.seed, 2, idx);
        match self.scale {
            Scale::Qkv => 0.05 * u,
            _ => 2.0 * u,
        }
    }

    pub fn bias_at(&self, idx: usize) -> f32 {
        let u = unit(self.seed, 3, idx);
        match self.scale {
            Scale::Qkv => 0.05 * u,
            _ => 2.0 * u,
        }
    }

    /// Writes the inputs (A, B, bias; the gaps between them are zero) through `put(offset, bytes)`, in pieces of at
    /// most 64 KiB, in increasing offsets. Everything before [`Layout::c`].
    pub fn write_inputs(&self, mut put: impl FnMut(usize, &[u8])) {
        let (m, n, k) = self.dims();
        let l = self.layout();
        let regions: [(usize, usize, u8); 3] = [(l.a, m * k, 1), (l.b, k * n, 2), (l.bias, n, 3)];
        let mut piece = Vec::with_capacity(64 << 10);
        for (at, count, which) in regions {
            let mut i = 0;
            while i < count {
                piece.clear();
                let stop = (i + (16 << 10)).min(count);
                for e in i..stop {
                    let v = match which {
                        1 => self.a_at(e),
                        2 => self.b_at(e),
                        _ => self.bias_at(e),
                    };
                    piece.extend_from_slice(&v.to_bits().to_le_bytes());
                }
                put(at + 4 * i, &piece);
                i = stop;
            }
        }
    }

    /// The byte ranges `(offset, len)` the OS must fill with [`FILL`] before the first launch: everything it will
    /// read back (all of C for a small set; the bands for a large one) and the guard.
    pub fn fill_ranges(&self) -> Vec<(usize, usize)> {
        self.read_ranges()
    }

    /// The byte ranges `(offset, len)` read back after the launch(es), in the order [`judge_set`] expects them
    /// concatenated.
    pub fn read_ranges(&self) -> Vec<(usize, usize)> {
        let (m, n, _) = self.dims();
        let l = self.layout();
        match self.bands {
            None => alloc::vec![(l.c, 4 * (m * n + GUARD))],
            Some(starts) => {
                let mut v: Vec<(usize, usize)> = starts
                    .iter()
                    .map(|&r| (l.c + 4 * r * n, 4 * BAND * n))
                    .collect();
                v.push((l.guard, 4 * GUARD));
                v
            }
        }
    }

    /// Bytes read back in total.
    pub fn read_len(&self) -> usize {
        self.read_ranges().iter().map(|r| r.1).sum()
    }

    /// The rows of C that are judged, in read order.
    pub fn judged_rows(&self) -> Vec<usize> {
        match self.bands {
            None => (0..self.m as usize).collect(),
            Some(starts) => starts.iter().flat_map(|&r| r..r + BAND).collect(),
        }
    }

    /// The whole buffer as bytes (inputs, then C and the guard = [`FILL`]). For the small sets and the tests;
    /// for `qkv-513` the OS uses [`Self::write_inputs`] and [`Self::fill_ranges`] instead (83 MB).
    pub fn buffer(&self) -> Vec<u8> {
        let l = self.layout();
        let mut b = alloc::vec![0u8; l.bytes];
        self.write_inputs(|at, bytes| b[at..at + bytes.len()].copy_from_slice(bytes));
        for (at, len) in self.fill_ranges() {
            for w in b[at..at + len].chunks_exact_mut(4) {
                w.copy_from_slice(&FILL.to_le_bytes());
            }
        }
        b
    }

    /// FNV-1a (32 bit) of the input bytes (`[0, c)` of the buffer), computed piece by piece.
    pub fn inputs_fnv(&self) -> u32 {
        let mut h = Fnv::new();
        let mut at = 0;
        self.write_inputs(|o, bytes| {
            h.zeros(o - at);
            h.add(bytes);
            at = o + bytes.len();
        });
        h.zeros(self.layout().c - at);
        h.0
    }

    /// The program's arguments for a buffer at `buf_va`.
    pub fn args(&self, buf_va: u64) -> tc::Args {
        let l = self.layout();
        tc::Args {
            a: buf_va + l.a as u64,
            b: buf_va + l.b as u64,
            c: buf_va + l.c as u64,
            bias: if self.flags & BIAS != 0 {
                buf_va + l.bias as u64
            } else {
                0
            },
            m: self.m,
            n: self.n,
            k: self.k,
            flags: self.flags,
        }
    }

    /// The judged products: one for a small set; one per band (op(A)'s band rows, stored untransposed) for a large
    /// one. Each band starts on a multiple of 16, so the CPU model computes the same 16×16 tiles as the whole product.
    fn parts(&self) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>, u32) {
        let (m, n, k) = self.dims();
        let b: Vec<f32> = (0..k * n).map(|i| self.b_at(i)).collect();
        let bias: Vec<f32> = (0..n).map(|i| self.bias_at(i)).collect();
        match self.bands {
            None => (
                alloc::vec![(0..m * k).map(|i| self.a_at(i)).collect()],
                b,
                bias,
                self.flags,
            ),
            Some(starts) => {
                let a_op = |r: usize, q: usize| {
                    if self.flags & TA != 0 {
                        self.a_at(q * m + r)
                    } else {
                        self.a_at(r * k + q)
                    }
                };
                let a = starts
                    .iter()
                    .map(|&r0| {
                        (r0..r0 + BAND)
                            .flat_map(|r| (0..k).map(move |q| (r, q)))
                            .map(|(r, q)| a_op(r, q))
                            .collect()
                    })
                    .collect();
                (a, b, bias, self.flags & !TA)
            }
        }
    }

    /// What the CPU model `gpu_compute::mm_tc::model` says the board writes, as the bytes read back (guard = FILL).
    pub fn model_out(&self) -> Vec<u8> {
        let (as_, b, bias, flags) = self.parts();
        let (_, n, k) = self.dims();
        let mut out = Vec::with_capacity(self.read_len());
        for a in &as_ {
            let p = Product {
                a,
                b: &b,
                bias: Some(&bias),
                m: a.len() / k,
                n,
                k,
                flags,
            };
            for v in oracle::model(&p).expect("probe inputs are in the f16 range") {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
            }
        }
        for _ in 0..GUARD {
            out.extend_from_slice(&FILL.to_le_bytes());
        }
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

/// Constant buffer 0 (0x800 bytes, arguments at 0x700) and the QMD (grid ⌈n/64⌉ × ⌈m/64⌉, 128 threads, 64
/// registers, 24 KiB of shared memory, one barrier) for set `set` whose buffer is at `buf_va` (aligned to 4 KiB).
/// `prog_va`: where [`tc::MM_TC`] was loaded (aligned to 256); `cbuf_va`: where the cbuf bytes go (aligned to 256).
pub fn launch_bytes(
    set: usize,
    prog_va: u64,
    cbuf_va: u64,
    buf_va: u64,
) -> GpuResult<([u8; tc::CBUF_BYTES], [u8; nvidia_rm570::QMD_SIZE])> {
    let s = SETS.get(set).ok_or(bad("mm_tc proba: set necunoscut"))?;
    if !buf_va.is_multiple_of(4096) || !cbuf_va.is_multiple_of(256) || !prog_va.is_multiple_of(256)
    {
        return Err(bad(
            "mm_tc proba: bufferul nealiniat la 4 KiB sau programul / cbuf nealiniate la 256",
        ));
    }
    let a = s.args(buf_va);
    let cb = a.cbuf().ok_or(bad("mm_tc proba: argumente"))?;
    let q = ComputeQmd {
        program_va: prog_va,
        registers: tc::REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: tc::CBUF_BYTES as u32,
        grid: a.grid(),
        block: tc::BLOCK,
        release: None,
    }
    .to_bytes_with_shared(&tc::SHARED)?;
    Ok((cb, q))
}

/// The verdict of one set.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every judged element within its written margin, guard untouched.
    InMargine,
    /// Some element outside its margin (written, finite).
    NuTrece,
    /// Wrong length, an element not written / not finite, or a guard word changed.
    Invalid,
}

/// What one set gave.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetResult {
    /// Index in [`SETS`].
    pub set: usize,
    pub verdict: Verdict,
    /// Worst element, in ‰ of its margin (`None` when invalid or outside).
    pub worst_permille: Option<u32>,
    /// Elements bit for bit equal to the CPU model `gpu_compute::mm_tc::model`.
    pub exact_model: u32,
    /// Elements judged (judged rows × n).
    pub elements: u32,
    /// All guard words still [`FILL`].
    pub guards_ok: bool,
    /// FNV-1a of the bytes read back.
    pub out_fnv: u32,
    /// Launch times on the board clock (ns), one per launch, if the OS measured them.
    pub ns: Vec<u64>,
}

impl SetResult {
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
    /// GFLOP/s of the median launch (2·m·n·k / ns), if timed.
    pub fn gflops(&self) -> Option<u32> {
        let s = SETS[self.set];
        let flop = 2 * s.m as u64 * s.n as u64 * s.k as u64;
        self.median_ns()
            .filter(|&ns| ns > 0)
            .map(|ns| u32::try_from(flop / ns).unwrap_or(u32::MAX))
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

fn words(b: &[u8]) -> Vec<u32> {
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect()
}

/// Judges the bytes read back for set `set` ([`ProbeSet::read_ranges`], concatenated) and the launch times.
pub fn judge_set(set: usize, out: &[u8], ns: &[u64]) -> GpuResult<SetResult> {
    let s = SETS.get(set).ok_or(bad("mm_tc proba: set necunoscut"))?;
    let (_, n, k) = s.dims();
    let elements = s.judged_rows().len() * n;
    let mut res = SetResult {
        set,
        verdict: Verdict::Invalid,
        worst_permille: None,
        exact_model: 0,
        elements: elements as u32,
        guards_ok: false,
        out_fnv: fnv(out),
        ns: ns.to_vec(),
    };
    if out.len() != s.read_len() {
        return Ok(res);
    }
    let got = words(out);
    res.guards_ok = got[elements..].iter().all(|&w| w == FILL);
    let want = words(&s.model_out());
    res.exact_model = (0..elements).filter(|&i| got[i] == want[i]).count() as u32;
    if !res.guards_ok {
        return Ok(res);
    }
    let (as_, b, bias, flags) = s.parts();
    let mut worst = 0u32;
    let mut at = 0;
    for a in &as_ {
        let rows = a.len() / k;
        let p = Product {
            a,
            b: &b,
            bias: Some(&bias),
            m: rows,
            n,
            k,
            flags,
        };
        let g: Vec<f32> = got[at..at + rows * n]
            .iter()
            .map(|&w| f32::from_bits(w))
            .collect();
        at += rows * n;
        match oracle::judge(&p, &g) {
            oracle::Verdict::InMargine { worst_permille, .. } => worst = worst.max(worst_permille),
            oracle::Verdict::NuTrece { .. } => {
                res.verdict = Verdict::NuTrece;
                return Ok(res);
            }
            oracle::Verdict::Invalid => return Ok(res),
        }
    }
    res.worst_permille = Some(worst);
    res.verdict = Verdict::InMargine;
    Ok(res)
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
    /// `qkv-513` was not run (no 83 MB of VRAM).
    pub fn qkv_skipped(&self) -> bool {
        !self.sets.iter().any(|r| r.set == QKV_SET)
    }
    /// Sets 0–4 all judged, every set run within its margin (`qkv-513` may be skipped).
    pub fn passes(&self) -> bool {
        (0..QKV_SET).all(|i| self.sets.iter().any(|r| r.set == i))
            && self.sets.iter().all(SetResult::passes)
    }
    pub fn in_margin(&self) -> usize {
        self.sets.iter().filter(|r| r.passes()).count()
    }
    pub fn exact_model(&self) -> u32 {
        self.sets.iter().map(|r| r.exact_model).sum()
    }
    pub fn elements(&self) -> u32 {
        self.sets.iter().map(|r| r.elements).sum()
    }
    /// The worst ‰ over the sets (`u32::MAX` if any set had none).
    pub fn worst_permille(&self) -> u32 {
        self.sets
            .iter()
            .map(|r| r.worst_permille.unwrap_or(u32::MAX))
            .max()
            .unwrap_or(0)
    }
    fn qkv(&self) -> Option<&SetResult> {
        self.sets.iter().find(|r| r.set == QKV_SET)
    }

    /// The `kt mm_tc …` rows (values in the stick journal, read back by [`parse_mm_tc`]).
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        let c = |v: u64| u32::try_from(v).unwrap_or(u32::MAX);
        let mut out = alloc::vec![
            ("kt mm_tc seturi", self.sets.len() as u32),
            ("kt mm_tc seturi in margine", self.in_margin() as u32),
            ("kt mm_tc elemente", self.elements()),
            ("kt mm_tc elemente exacte (model)", self.exact_model()),
            (
                "kt mm_tc abaterea maxima (promile din margine)",
                self.worst_permille()
            ),
            ("kt mm_tc qkv omis", u32::from(self.qkv_skipped())),
        ];
        if let Some(q) = self.qkv() {
            if let Some(ns) = q.median_ns() {
                out.push(("kt mm_tc qkv timp median (ns)", c(ns)));
            }
            if let Some(g) = q.gflops() {
                out.push(("kt mm_tc qkv gflop/s", g));
            }
        }
        for r in &self.sets {
            out.extend([
                ("kt mm_tc set: index", r.set as u32),
                ("kt mm_tc set: intrari (fnv)", SETS[r.set].inputs_fnv()),
                ("kt mm_tc set: verdict", r.code()),
                ("kt mm_tc set: exacte (model)", r.exact_model),
                (
                    "kt mm_tc set: promile",
                    r.worst_permille.unwrap_or(u32::MAX),
                ),
                ("kt mm_tc set: garzi", u32::from(r.guards_ok)),
                ("kt mm_tc set: iesire (fnv)", r.out_fnv),
            ]);
            if let Some(ns) = r.median_ns() {
                out.push(("kt mm_tc set: timp (ns)", c(ns)));
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
        let mut out = alloc::vec![format!(
            "mm_tc ({label}): {} / {} seturi in margine, {} / {} elemente exacte fata de model, abaterea maxima {} promile din margine",
            self.in_margin(),
            self.sets.len(),
            self.exact_model(),
            self.elements(),
            self.worst_permille()
        )];
        match self.qkv() {
            None => out.push(format!("mm_tc qkv-513 ({label}): omis (fara VRAM)")),
            Some(q) => {
                if let (Some(ns), Some(g)) = (q.median_ns(), q.gflops()) {
                    out.push(format!(
                        "mm_tc qkv-513 ({label}): mediana {ns} ns pe {} lansari = {}.{:02} TFLOP/s (estimat 8-17)",
                        q.ns.len(),
                        g / 1000,
                        (g % 1000) / 10
                    ));
                }
            }
        }
        for r in &self.sets {
            let t = r
                .median_ns()
                .map(|n| format!(", {n} ns"))
                .unwrap_or_default();
            let w = r
                .worst_permille
                .map(|p| format!("{p}"))
                .unwrap_or_else(|| String::from("-"));
            out.push(format!(
                "mm_tc set {} ({label}): {}, exacte {}/{}, {} promile, garzi {}{}",
                SETS[r.set].name,
                ["NU TRECE", "in margine", "INVALID"][r.code() as usize],
                r.exact_model,
                r.elements,
                w,
                if r.guards_ok { "neatinse" } else { "ATINSE" },
                t
            ));
        }
        out
    }
}

/// What [`parse_mm_tc`] reads back.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsedMmTc {
    pub sets: Option<u32>,
    pub in_margin: Option<u32>,
    pub elements: Option<u32>,
    pub exact_model: Option<u32>,
    pub worst_permille: Option<u32>,
    pub qkv_skipped: Option<u32>,
    pub qkv_median_ns: Option<u32>,
    pub qkv_gflops: Option<u32>,
    /// Per set, in order: (index, inputs fnv, verdict, exact, ‰, guards, output fnv, ns or `u32::MAX`).
    pub per_set: Vec<[u32; 8]>,
}

/// Reads the rows of [`Probe::rows`] (`gpu: valoare kt mm_tc … = 0x…`). A new `kt mm_tc seturi` row starts over
/// (the last run in a journal is kept).
pub fn parse_mm_tc(text: &str) -> ParsedMmTc {
    let mut p = ParsedMmTc::default();
    for line in text.lines() {
        let Some(rest) = line.split("valoare kt mm_tc ").nth(1) else {
            continue;
        };
        let Some((name, val)) = rest.rsplit_once(" = 0x") else {
            continue;
        };
        let Ok(v) = u32::from_str_radix(val.trim(), 16) else {
            continue;
        };
        let last = |p: &mut ParsedMmTc, k: usize| {
            if let Some(s) = p.per_set.last_mut() {
                s[k] = v;
            }
        };
        match name {
            "seturi" => {
                p = ParsedMmTc {
                    sets: Some(v),
                    ..ParsedMmTc::default()
                }
            }
            "seturi in margine" => p.in_margin = Some(v),
            "elemente" => p.elements = Some(v),
            "elemente exacte (model)" => p.exact_model = Some(v),
            "abaterea maxima (promile din margine)" => p.worst_permille = Some(v),
            "qkv omis" => p.qkv_skipped = Some(v),
            "qkv timp median (ns)" => p.qkv_median_ns = Some(v),
            "qkv gflop/s" => p.qkv_gflops = Some(v),
            "set: index" => p.per_set.push([v, 0, 0, 0, 0, 0, 0, u32::MAX]),
            "set: intrari (fnv)" => last(&mut p, 1),
            "set: verdict" => last(&mut p, 2),
            "set: exacte (model)" => last(&mut p, 3),
            "set: promile" => last(&mut p, 4),
            "set: garzi" => last(&mut p, 5),
            "set: iesire (fnv)" => last(&mut p, 6),
            "set: timp (ns)" => last(&mut p, 7),
            _ => {}
        }
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_are_aligned_and_qkv_is_about_83_mb() {
        for s in &SETS {
            let l = s.layout();
            for o in [l.a, l.b, l.bias, l.c] {
                assert_eq!(o % 256, 0, "{}", s.name);
            }
            assert_eq!(l.bytes % 4096, 0);
            assert!(l.end <= l.bytes);
        }
        let q = SETS[QKV_SET].layout().bytes;
        assert!(82_000_000 < q && q < 83_000_000, "{q}"); // ~82,6 MB (78,8 MiB)
        assert!(
            SETS[..QKV_SET]
                .iter()
                .all(|s| s.layout().bytes <= 256 << 10)
        );
    }

    #[test]
    fn inputs_stream_matches_the_buffer_and_stay_in_f16_range() {
        for s in &SETS[..QKV_SET] {
            let b = s.buffer();
            assert_eq!(fnv(&b[..s.layout().c]), s.inputs_fnv(), "{}", s.name);
            let (m, n, k) = s.dims();
            assert!((0..m * k).all(|i| s.a_at(i).abs() <= 65504.0));
            assert!((0..k * n).all(|i| s.b_at(i).abs() <= 2.0));
        }
        let l = SETS[4];
        assert_eq!(l.a_at(0), 65504.0);
        // Row 1 of the limits set is tiny: below the smallest normal f16.
        assert!((0..64).all(|q| l.a_at(64 + q).abs() < 1.0 / 16384.0));
    }

    #[test]
    fn sets_differ_and_read_ranges_are_inside() {
        let f: Vec<u32> = SETS[..QKV_SET].iter().map(ProbeSet::inputs_fnv).collect();
        for i in 0..f.len() {
            for j in i + 1..f.len() {
                assert_ne!(f[i], f[j]);
            }
        }
        for s in &SETS {
            let l = s.layout();
            for (o, n) in s.read_ranges() {
                assert!(o >= l.c && o + n <= l.end);
            }
            assert_eq!(
                s.read_len(),
                4 * (s.judged_rows().len() * s.n as usize + GUARD)
            );
        }
        assert_eq!(SETS[QKV_SET].judged_rows().len(), 48);
        assert!(SETS[QKV_SET].bands.unwrap().iter().all(|r| r % 16 == 0));
    }

    #[test]
    fn launch_bytes_checks_alignment_and_declares_shared_memory() {
        let (cb, q) = launch_bytes(1, 0x1000, 0x2000, 0x10_000).unwrap();
        let at = |o: usize| u64::from_le_bytes(cb[o..o + 8].try_into().unwrap());
        let l = SETS[1].layout();
        assert_eq!(at(0x700), 0x10_000);
        assert_eq!(at(0x710), 0x10_000 + l.c as u64);
        assert_eq!(at(0x718), 0x10_000 + l.bias as u64);
        let plain = ComputeQmd {
            program_va: 0x1000,
            registers: tc::REGISTERS,
            cbuf0_va: 0x2000,
            cbuf0_size: tc::CBUF_BYTES as u32,
            grid: SETS[1].args(0x10_000).grid(),
            block: tc::BLOCK,
            release: None,
        }
        .to_bytes()
        .unwrap();
        assert_ne!(q, plain, "the shared memory must be declared");
        // No bias flag: no bias address.
        let (cb, _) = launch_bytes(2, 0x1000, 0x2000, 0x10_000).unwrap();
        assert_eq!(at_of(&cb, 0x718), 0);
        assert!(launch_bytes(1, 0x1000, 0x2000, 0x10_200).is_err());
        assert!(launch_bytes(1, 0x1000, 0x2010, 0x10_000).is_err());
        assert!(launch_bytes(1, 0x1010, 0x2000, 0x10_000).is_err());
        assert!(launch_bytes(9, 0x1000, 0x2000, 0x10_000).is_err());
    }

    fn at_of(cb: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(cb[o..o + 8].try_into().unwrap())
    }

    #[test]
    fn model_output_passes_and_unwritten_is_invalid() {
        for (i, s) in SETS[..QKV_SET].iter().enumerate() {
            let r = judge_set(i, &s.model_out(), &[]).unwrap();
            assert!(r.passes(), "{}: {r:?}", s.name);
            assert_eq!(r.exact_model, r.elements);
            let b = s.buffer();
            let (o, n) = s.read_ranges()[0];
            let r = judge_set(i, &b[o..o + n], &[]).unwrap();
            assert_eq!(r.verdict, Verdict::Invalid);
        }
        assert!(judge_set(9, &[], &[]).is_err());
        assert_eq!(judge_set(0, &[0; 8], &[]).unwrap().code(), 2);
    }
}
