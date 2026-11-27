//! C-157 p.2 (session 274, `docs/C-157_TOKEN32_CEAS.md`): `token32_v2c` — `token32_v2` with the SM clock read around
//! every operation, a MEASURING program (never used for generation): where do the ~0.67 ms of `token32_v2` at G = 768
//! that the model cannot explain go (`docs/C-157_TOKEN32_V2_PLACA.md` §3)?
//!
//! The program is [`crate::kronos_token::TOKEN32_V2`]'s arithmetic (same operations in the same order) plus reads of the
//! SM cycle counter and plain stores into a separate clock area, whose address is a new constant-buffer field at
//! [`CBUF_CLOCK`]. The clock is per SM, so only differences taken by the same workgroup mean anything. The area
//! ([`clock_words`] u32 words, n operations, G workgroups):
//! - `[k·G + g]`: cycles of workgroup g's work on operation k (its loop over blocks; the barrier is not inside);
//! - `[n·G + k]`, k < n: workgroup 0's clock at the start of operation k (after the barrier); `[n·G + n]`: after the
//!   last operation — so the window of operation k = start(k + 1) − start(k) = the slowest workgroup + the barrier;
//! - `[n·G + n + 1 + g]`: how many times workgroup g read the barrier counter over the token.
//!
//! [`read_clock`] checks and splits that area, [`by_part`] sums it over the kinds of today's list ([`Part`]), and
//! [`ClockSummary::rows`] gives the `kc` journal rows (ns, converted with the board time the OS measured for the token).
//! Nothing here runs anything; in `gpu-sim` the area is filled with invented numbers (SIMULARE), only to check the reading.

use alloc::vec::Vec;

use gpu_core::{GpuError, GpuResult};

use crate::kronos_layer::{Launch, Prog};
use crate::kronos_token as kt;
use crate::occupancy::{self, Kernel, Reserve, SM86};
use crate::qmd::{ComputeQmd, MAX_SHARED_BYTES, SM_CONFIG_STEPS, SharedMemory};

fn bad(what: &'static str) -> GpuError {
    GpuError::InvalidArgument(what)
}

/// `token32_v2c.comp` compiled by Mesa NAK 25.0.7 for SM 8.6 (`tools/nak/out/token32_v2c.sm86.txt`).
pub const TOKEN32_V2C: &[u8] = include_bytes!("../../../tools/nak/out/token32_v2c.sm86.sass");
/// Registers per thread NAK reported for `token32_v2c` (`Num GPRs`) — NOT those of `token32_v2`, so the QMD is
/// [`token32_v2c_qmd_wide`], never `kronos_token::token32_qmd_wide` (which writes 92).
pub const TOKEN32_V2C_REGISTERS: u32 = 99;
/// SHA-256 of [`TOKEN32_V2C`] (the OS checks it before sending).
pub const TOKEN32_V2C_SHA256: &str =
    "e899fa7cce1d7e31e511f0304bb6d82961c0e13bcd19d86827d58db4b6c18206";

/// `token32_v2c` for the occupancy arithmetic: 32 threads, [`TOKEN32_V2C_REGISTERS`], the shared memory of `token32`.
pub const TOKEN32_V2C_KERNEL: Kernel = Kernel {
    name: "token32_v2c",
    threads: kt::TOKEN_BLOCK,
    registers: TOKEN32_V2C_REGISTERS,
    shared_bytes: kt::TOKEN_SHARED,
};

/// The QMD of the one `token32_v2c` launch: as `kronos_token::token32_qmd_wide` (the smallest SM shared-memory step
/// at which all `groups` are resident, with the 1 KiB reserve) but with [`TOKEN32_V2C_REGISTERS`]. Refuses what
/// cannot all be resident at once (the barrier would never open).
pub fn token32_v2c_qmd_wide(
    prog_va: u64,
    cbuf_va: u64,
    groups: u32,
    sms: u32,
) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    if groups == 0 || sms == 0 {
        return Err(bad("ceasul: 0 grupuri sau niciun SM"));
    }
    let step = SM_CONFIG_STEPS
        .iter()
        .copied()
        .find(|&step| {
            occupancy::blocks_per_sm(&SM86, &TOKEN32_V2C_KERNEL, step, Reserve::OneKib)
                .is_some_and(|f| u64::from(f.blocks) * u64::from(sms) >= u64::from(groups))
        })
        .ok_or(bad(
            "ceasul: mai multe grupuri decat incap deodata pe placa",
        ))?;
    ComputeQmd {
        program_va: prog_va,
        registers: TOKEN32_V2C_REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: 0x100,
        grid: [groups, 1, 1],
        block: [kt::TOKEN_BLOCK, 1, 1],
        release: None,
    }
    .to_bytes_with_sm_config(
        &SharedMemory {
            bytes: kt::TOKEN_SHARED,
            max_bytes: MAX_SHARED_BYTES,
            barriers: 0,
        },
        step,
    )
}

/// The constant-buffer field with the clock area's address (u64), right after `n` (0x38) — push constants of
/// `token32_v2c.comp`: list 0x28, sync 0x30, n 0x38, clock 0x40.
pub const CBUF_CLOCK: usize = 0x40;

/// u32 words of the clock area for `n` operations and `groups` workgroups: n·G + n + 1 + G.
pub fn clock_words(n: usize, groups: u32) -> GpuResult<usize> {
    let g = groups as usize;
    if n == 0 || g == 0 {
        return Err(bad("ceasul: n sau G = 0"));
    }
    n.checked_mul(g)
        .and_then(|x| x.checked_add(n + 1 + g))
        .ok_or(bad("ceasul: zona prea mare"))
}

/// [`kt::token_cbuf_for`] plus the clock area's address at [`CBUF_CLOCK`] (aligned to 4).
pub fn token_cbuf_clock(
    list_va: u64,
    sync_va: u64,
    n: usize,
    groups: u32,
    clock_va: u64,
) -> GpuResult<[u8; 256]> {
    if clock_va == 0 || !clock_va.is_multiple_of(4) {
        return Err(bad("ceasul: adresa zonei nealiniata sau 0"));
    }
    let mut c = kt::token_cbuf_for(list_va, sync_va, n, groups)?;
    c[CBUF_CLOCK..CBUF_CLOCK + 8].copy_from_slice(&clock_va.to_le_bytes());
    Ok(c)
}

/// One operation of the clock area (cycles of the SM clock).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OpClock {
    /// Workgroup 0's window: start of the next operation − start of this one (slowest workgroup + barrier).
    pub window: u32,
    /// The longest work of a workgroup on this operation.
    pub max_work: u32,
    /// The mean work over the G workgroups (rounded down).
    pub mean_work: u32,
    /// Workgroup 0's own work.
    pub wg0_work: u32,
    /// Which workgroup worked longest (the lowest index on a tie).
    pub slowest: u32,
}

/// The whole clock area of one token.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockReport {
    pub groups: u32,
    pub ops: Vec<OpClock>,
    /// Barrier counter reads over the token, all workgroups together.
    pub polls_total: u64,
    /// The most reads by one workgroup.
    pub polls_max: u32,
}

/// Splits the clock area `words` of a token of `n` operations at `groups` workgroups. Refuses an area of the wrong
/// size, a window shorter than workgroup 0's own work (the clock would be wrong) and a workgroup that read the
/// counter fewer than n − 1 times (each barrier reads it at least once).
pub fn read_clock(words: &[u32], n: usize, groups: u32) -> GpuResult<ClockReport> {
    if words.len() != clock_words(n, groups)? {
        return Err(bad("ceasul: zona nu are n*G + n + 1 + G cuvinte"));
    }
    let g = groups as usize;
    let starts = &words[n * g..n * g + n + 1];
    let mut ops = Vec::with_capacity(n);
    for k in 0..n {
        let work = &words[k * g..(k + 1) * g];
        let window = starts[k + 1].wrapping_sub(starts[k]);
        let (mut max_work, mut slowest, mut sum) = (0u32, 0u32, 0u64);
        for (i, &w) in work.iter().enumerate() {
            if w > max_work {
                max_work = w;
                slowest = i as u32;
            }
            sum += w as u64;
        }
        if window < work[0] {
            return Err(bad("ceasul: fereastra mai scurta decat munca grupului 0"));
        }
        ops.push(OpClock {
            window,
            max_work,
            mean_work: (sum / g as u64) as u32,
            wg0_work: work[0],
            slowest,
        });
    }
    let polls = &words[n * g + n + 1..];
    if polls.iter().any(|&p| (p as usize) < n - 1) {
        return Err(bad(
            "ceasul: un grup a citit contorul mai rar decat barierele",
        ));
    }
    Ok(ClockReport {
        groups,
        ops,
        polls_total: polls.iter().map(|&p| p as u64).sum(),
        polls_max: polls.iter().copied().max().unwrap_or(0),
    })
}

/// The kinds of today's generation list (`kronos_model::token_launches_fused_ln32` + `argmax`), as the os-468 profile
/// names them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    Embed,
    Ln1,
    Qkv,
    Attention,
    Out,
    Ln2,
    Fc,
    Proj,
    FinalLn,
    Logits,
    Argmax,
}

/// Every part, in the order of the journal rows.
pub const PARTS: [Part; 11] = [
    Part::Embed,
    Part::Ln1,
    Part::Qkv,
    Part::Attention,
    Part::Out,
    Part::Ln2,
    Part::Fc,
    Part::Proj,
    Part::FinalLn,
    Part::Logits,
    Part::Argmax,
];

/// The glued layer's 7 launches (`kronos_layer::token_launches_fused`).
const LAYER: [Part; 7] = [
    Part::Ln1,
    Part::Qkv,
    Part::Attention,
    Part::Out,
    Part::Ln2,
    Part::Fc,
    Part::Proj,
];

impl Part {
    fn index(self) -> usize {
        PARTS.iter().position(|&p| p == self).unwrap_or(0)
    }

    /// The programs this part may run on (the list's swaps: `layernorm256`, the quarter-block products).
    fn progs(self) -> &'static [Prog] {
        match self {
            Part::Embed => &[Prog::Embed, Prog::EmbedDev],
            Part::Ln1 | Part::Ln2 | Part::FinalLn => &[Prog::LayerNorm, Prog::LayerNorm256],
            Part::Qkv | Part::Out | Part::Fc | Part::Proj => &[Prog::Q8Fused, Prog::Q8Fused32],
            Part::Attention => &[Prog::Attention],
            Part::Logits => &[Prog::Q8Matvec, Prog::Q8Matvec32],
            Part::Argmax => &[Prog::Argmax],
        }
    }

    /// The three `kc` row names: window, slowest work, mean work (ns per token).
    pub fn rows(self) -> [&'static str; 3] {
        PART_ROWS[self.index()]
    }
}

const PART_ROWS: [[&str; 3]; 11] = [
    [
        "kc embed fereastra (ns)",
        "kc embed munca max (ns)",
        "kc embed munca medie (ns)",
    ],
    [
        "kc ln1 fereastra (ns)",
        "kc ln1 munca max (ns)",
        "kc ln1 munca medie (ns)",
    ],
    [
        "kc qkv fereastra (ns)",
        "kc qkv munca max (ns)",
        "kc qkv munca medie (ns)",
    ],
    [
        "kc atentie fereastra (ns)",
        "kc atentie munca max (ns)",
        "kc atentie munca medie (ns)",
    ],
    [
        "kc out fereastra (ns)",
        "kc out munca max (ns)",
        "kc out munca medie (ns)",
    ],
    [
        "kc ln2 fereastra (ns)",
        "kc ln2 munca max (ns)",
        "kc ln2 munca medie (ns)",
    ],
    [
        "kc fc fereastra (ns)",
        "kc fc munca max (ns)",
        "kc fc munca medie (ns)",
    ],
    [
        "kc proj fereastra (ns)",
        "kc proj munca max (ns)",
        "kc proj munca medie (ns)",
    ],
    [
        "kc ln_final fereastra (ns)",
        "kc ln_final munca max (ns)",
        "kc ln_final munca medie (ns)",
    ],
    [
        "kc logit fereastra (ns)",
        "kc logit munca max (ns)",
        "kc logit munca medie (ns)",
    ],
    [
        "kc argmax fereastra (ns)",
        "kc argmax munca max (ns)",
        "kc argmax munca medie (ns)",
    ],
];

/// The parts of a token list: embedding, `layers` glued layers of 7, the final LayerNorm, the logits, and the argmax
/// when `launches` ends with it. Refuses a list of another shape or a program that does not fit its place.
pub fn parts_of(launches: &[Launch]) -> GpuResult<Vec<Part>> {
    let argmax = launches.last().is_some_and(|l| l.prog == Prog::Argmax);
    let body = launches.len().saturating_sub(3 + argmax as usize);
    if launches.len() < 3 + 7 || !body.is_multiple_of(7) {
        return Err(bad(
            "ceasul: lista nu e embed + 7 pe strat + ln + logit (+ argmax)",
        ));
    }
    let mut v = Vec::with_capacity(launches.len());
    v.push(Part::Embed);
    for _ in 0..body / 7 {
        v.extend_from_slice(&LAYER);
    }
    v.push(Part::FinalLn);
    v.push(Part::Logits);
    if argmax {
        v.push(Part::Argmax);
    }
    for (p, l) in v.iter().zip(launches) {
        if !p.progs().contains(&l.prog) {
            return Err(bad("ceasul: un program nu se potriveste locului"));
        }
    }
    Ok(v)
}

/// One part summed over the token (cycles).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PartClock {
    pub ops: u32,
    pub window: u64,
    pub max_work: u64,
    pub mean_work: u64,
}

/// A token's clock area summed per part.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClockSummary {
    /// Indexed as [`PARTS`].
    pub parts: [PartClock; 11],
    /// Sum of all windows = the token from the start of operation 0 to the end of the last (cycles).
    pub windows: u64,
    /// Sum over operations of (window − slowest work), never below 0: the barrier as workgroup 0 saw it (cycles).
    pub barrier: u64,
    pub polls_total: u64,
    pub polls_max: u32,
    pub groups: u32,
    pub ops: u32,
}

/// Sums `r` over `parts` ([`parts_of`] of the same list).
pub fn by_part(r: &ClockReport, parts: &[Part]) -> GpuResult<ClockSummary> {
    if parts.len() != r.ops.len() {
        return Err(bad("ceasul: felurile nu au lungimea listei"));
    }
    let mut s = ClockSummary {
        parts: [PartClock::default(); 11],
        windows: 0,
        barrier: 0,
        polls_total: r.polls_total,
        polls_max: r.polls_max,
        groups: r.groups,
        ops: r.ops.len() as u32,
    };
    for (o, p) in r.ops.iter().zip(parts) {
        let e = &mut s.parts[p.index()];
        e.ops += 1;
        e.window += o.window as u64;
        e.max_work += o.max_work as u64;
        e.mean_work += o.mean_work as u64;
        s.windows += o.window as u64;
        s.barrier += o.window.saturating_sub(o.max_work) as u64;
    }
    Ok(s)
}

/// The names of the summary `kc` rows.
pub mod rows {
    pub const GROUPS: &str = "kc grupuri";
    pub const OPS: &str = "kc operatii";
    /// The SM clock deduced from the windows and the board time of the token (MHz).
    pub const MHZ: &str = "kc ceasul SM dedus (MHz)";
    pub const WINDOWS: &str = "kc ferestre toate (ns)";
    pub const BARRIER: &str = "kc bariera dupa cel mai lent (ns)";
    /// Counter reads per workgroup and per barrier, ×10.
    pub const POLLS: &str = "kc citiri contor pe grup si bariera (x10)";
    pub const POLLS_MAX: &str = "kc citiri contor cel mai harnic grup";
}

fn sat(x: u64) -> u32 {
    x.min(u32::MAX as u64) as u32
}

impl ClockSummary {
    /// Cycles → ns with the board time of the same token (`board_ns`, MĂSURAT by the OS from the records): the windows
    /// cover the token from the start of operation 0 to the end of the last, so ns = cycles · board_ns / windows.
    pub fn ns(&self, cycles: u64, board_ns: u64) -> u64 {
        if self.windows == 0 {
            return 0;
        }
        ((cycles as u128 * board_ns as u128) / self.windows as u128) as u64
    }

    /// The `kc` rows: 7 summary rows, then 3 per part that has operations (window, slowest work, mean work, in ns).
    pub fn rows(&self, board_ns: u64) -> Vec<(&'static str, u32)> {
        let mhz = (self.windows * 1000).checked_div(board_ns).unwrap_or(0);
        let barriers = (self.ops.saturating_sub(1) as u64) * self.groups as u64;
        let polls10 = (self.polls_total * 10).checked_div(barriers).unwrap_or(0);
        let mut v = alloc::vec![
            (rows::GROUPS, self.groups),
            (rows::OPS, self.ops),
            (rows::MHZ, sat(mhz)),
            (rows::WINDOWS, sat(self.ns(self.windows, board_ns))),
            (rows::BARRIER, sat(self.ns(self.barrier, board_ns))),
            (rows::POLLS, sat(polls10)),
            (rows::POLLS_MAX, self.polls_max),
        ];
        for p in PARTS {
            let c = self.parts[p.index()];
            if c.ops == 0 {
                continue;
            }
            let [w, m, a] = p.rows();
            v.push((w, sat(self.ns(c.window, board_ns))));
            v.push((m, sat(self.ns(c.max_work, board_ns))));
            v.push((a, sat(self.ns(c.mean_work, board_ns))));
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An invented area (not the board): G = 4, 3 operations.
    fn area() -> Vec<u32> {
        let (n, g) = (3usize, 4usize);
        let mut w = alloc::vec![0u32; n * g + n + 1 + g];
        let work = [[100, 150, 120, 130], [50, 60, 70, 40], [10, 10, 30, 10]];
        for k in 0..n {
            for i in 0..g {
                w[k * g + i] = work[k][i];
            }
        }
        // starts: wrapping across u32::MAX on purpose.
        let s0 = u32::MAX - 100;
        let starts = [
            s0,
            s0.wrapping_add(200),
            s0.wrapping_add(300),
            s0.wrapping_add(350),
        ];
        w[n * g..n * g + n + 1].copy_from_slice(&starts);
        w[n * g + n + 1..].copy_from_slice(&[2, 9, 5, 4]);
        w
    }

    #[test]
    fn reads_the_area() {
        let r = read_clock(&area(), 3, 4).unwrap();
        assert_eq!(
            r.ops[0],
            OpClock {
                window: 200,
                max_work: 150,
                mean_work: 125,
                wg0_work: 100,
                slowest: 1
            }
        );
        assert_eq!(r.ops[1].window, 100);
        assert_eq!(
            r.ops[2],
            OpClock {
                window: 50,
                max_work: 30,
                mean_work: 15,
                wg0_work: 10,
                slowest: 2
            }
        );
        assert_eq!((r.polls_total, r.polls_max), (20, 9));
        let s = by_part(&r, &[Part::Ln1, Part::Qkv, Part::Qkv]).unwrap();
        assert_eq!(s.windows, 350);
        assert_eq!(s.barrier, 50 + 30 + 20);
        assert_eq!(
            s.parts[Part::Qkv.index()],
            PartClock {
                ops: 2,
                window: 150,
                max_work: 100,
                mean_work: 70
            }
        );
        // 350 cycles over a 700 ns token: 500 MHz; ns = cycles × 2.
        let rows = s.rows(700);
        assert_eq!(rows[2], (rows::MHZ, 500));
        assert_eq!(rows[3], (rows::WINDOWS, 700));
        assert_eq!(rows[4], (rows::BARRIER, 200));
        assert_eq!(
            rows[5],
            (rows::POLLS, 25),
            "20 reads / (2 barriers × 4 groups) = 2.5"
        );
        assert_eq!(rows.len(), 7 + 3 * 2);
        assert_eq!(rows[10], ("kc qkv fereastra (ns)", 300));
    }

    #[test]
    fn refuses_bad_areas() {
        assert!(read_clock(&area()[1..], 3, 4).is_err());
        let mut w = area();
        w[0] = 250; // workgroup 0 worked longer than its window
        assert!(read_clock(&w, 3, 4).is_err());
        let mut w = area();
        w[3 * 4 + 4] = 1; // fewer reads than the 2 barriers
        assert!(read_clock(&w, 3, 4).is_err());
        assert!(clock_words(0, 4).is_err() && clock_words(3, 0).is_err());
        assert_eq!(clock_words(88, 768).unwrap(), 68_441);
        assert!(token_cbuf_clock(0x1000, 0x2000, 3, 4, 0).is_err());
        assert!(token_cbuf_clock(0x1000, 0x2000, 3, 4, 0x3002).is_err());
        let c = token_cbuf_clock(0x1000, 0x2000, 3, 4, 0x3000).unwrap();
        assert_eq!(&c[CBUF_CLOCK..CBUF_CLOCK + 8], &0x3000u64.to_le_bytes());
        assert_eq!(
            &c[..CBUF_CLOCK],
            &kt::token_cbuf_for(0x1000, 0x2000, 3, 4).unwrap()[..CBUF_CLOCK]
        );
    }

    #[test]
    fn the_qmd_carries_v2c_registers() {
        // 99 registers still leave 16 blocks of 32 per SM: G = 768 on 48 SMs fits, 769 does not.
        let q = token32_v2c_qmd_wide(0x10_0000, 0x20_0000, 768, 48).unwrap();
        let old = kt::token32_qmd_wide(0x10_0000, 0x20_0000, 768, 48).unwrap();
        let diff = q.iter().zip(old.iter()).filter(|(a, b)| a != b).count();
        assert!(
            (1..=2).contains(&diff),
            "only the register count differs ({diff} bytes)"
        );
        assert!(token32_v2c_qmd_wide(0x10_0000, 0x20_0000, 769, 48).is_err());
        assert!(token32_v2c_qmd_wide(0x10_0000, 0x20_0000, 0, 48).is_err());
        assert_eq!(TOKEN32_V2C_REGISTERS, 99);
    }
}
