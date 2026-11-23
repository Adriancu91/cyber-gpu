//! C-160 point 6, second part — **the whole training step as a list of launches the OS can send**: the small golden
//! step (`pas_de_aur_mic`: 2 layers, C 64, 4 heads, vocab 256, batch 2 × 32) on the real board.
//!
//! This is the list T165 ran on the simulated board (`tests/sim_c160_p5_golden.rs`), moved into the library so the OS
//! sends exactly what the simulation checked:
//! - [`Prog`]: the 13 programs of a step (the 10 training programs + `vadd`, which already runs in generation), with
//!   their code, SHA-256, registers, block and shared memory;
//! - [`Plan`]: one arena of board memory — weights, gradients, AdamW's m and v (flat, `named_parameters()` order),
//!   the batch, the `emb_back` segment lists, every activation the backward pass keeps, the scratch — at fixed
//!   offsets from a base address the OS picks; [`Plan::uploads`] = what the CPU writes **once** before step 1;
//! - [`Plan::step`]: the launches of one step, in order (75 at `adamw_513`, 73 at `sgd`), each a [`Launch`] (program,
//!   constant buffer, grid, a tag saying what it computes);
//! - [`pack`]: the control zone of one step (a cbuf + a QMD per launch, 512 bytes each) and the QMD addresses for
//!   `transfer::encode_sequence_fresh` — one submission per step;
//! - [`judge`] → [`Verdict`]: the run (losses, norms, gradient of step 1, weights after step 1 and after the last
//!   step) against PyTorch's numbers (the hub's `pas_de_aur_mic` files), in the hub's tolerances
//!   (`gpu_compute::train::TOL_*`, the same as T151 / T165); `kt pas-aur …` rows, report lines, [`parse_rows`].
//!
//! Tested **only in SIMULARE** (T165 now runs this list; T167, `tests/sim_c160_p6_train_step.rs`); nothing here has
//! run on a board.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::layer::{EmbSegments, OptScalars};
use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use crate::qmd::{ComputeQmd, SharedMemory};
use crate::{
    attn_back as ab, attn_back_tile_2 as abt2, attn_back_tile0_2 as abt02, attn_fwd as af,
    attn_fwd_tile2 as aft2, attn_p as ap, attn_p_tile as apt, colsum as cs, emb_back as eb,
    emb_fwd as ef, gelu_tr as gt, gnorm as gn, kronos_layer, ln_back as lb, ln_fwd as lf,
    mm_train as mm, opt_step as os, xent as xe,
};

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The learning rate of the `sgd` recipe (`reteta.json`).
pub const SGD_LR: f32 = 0.01;
/// The clipping threshold of the `adamw_513` recipe.
pub const MAX_NORM: f32 = 1.0;
/// Elements per block of the norm's first launch (`gnorm` partial sums).
pub const GNORM_CHUNK: u32 = 4_096;
/// Bytes of one launch in the control zone: the cbuf (0x100) then the QMD (0x100).
pub const SLOT: u64 = 0x200;
/// SHA-256 of `vadd.sm86.sass` (Mesa NAK 25.0.7; the generation's residual addition, unchanged).
pub const VADD_SHA256: &str = "8757ddc1348f2ae034342c24a364639db3bdb85402e70b16a84410cb4e0487f5";

// ------------------------------------------------------------------ the programs

/// One program of the training step.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prog {
    MmTrain,
    Xent,
    LnFwd,
    LnBack,
    Colsum,
    GeluTr,
    AttnFwd,
    AttnBack,
    EmbFwd,
    EmbBack,
    Gnorm,
    OptStep,
    Vadd,
    /// C-160 point 7 (`docs/C-160_P7_ATTN_P_IN_PAS.md`): `attn_p`, P without P·V — only in [`PROGRAMS_P_ONLY`], for
    /// the re-launches of [`Plan::new_p_only`]. Not in [`PROGRAMS`] (what the OS loads for the other plans is unchanged).
    AttnP,
    /// C-160 point 7 (`docs/C-160_P7_PLACI_IN_PAS.md`): the forward attention on tiles (`attn_fwd_tile2`), only in
    /// the plans of [`Plan::new_tiles`] / [`Plan::new_p_only_tiles`].
    AttnFwdTile2,
    /// The rows of dS (mode 0) on tiles (`attn_back_tile0_2`), tiled plans only.
    AttnBackTile0_2,
    /// The columns dQ, dK, dV (mode 1) on tiles (`attn_back_tile_2`), tiled plans only.
    AttnBackTile2,
    /// P without P·V on tiles (`attn_p_tile`), the re-launches of [`Plan::new_p_only_tiles`] only.
    AttnPTile,
}

/// Every program a step uses (the OS loads all 13 once; `Gnorm` only runs at `adamw_513`).
pub const PROGRAMS: [Prog; 13] = [
    Prog::MmTrain,
    Prog::Xent,
    Prog::LnFwd,
    Prog::LnBack,
    Prog::Colsum,
    Prog::GeluTr,
    Prog::AttnFwd,
    Prog::AttnBack,
    Prog::EmbFwd,
    Prog::EmbBack,
    Prog::Gnorm,
    Prog::OptStep,
    Prog::Vadd,
];

/// Every program any plan may launch: [`PROGRAMS_P_ONLY`] then the four tiled ones, so every older program keeps its
/// [`Prog::index`] (`check_programs` checks all of them).
pub const PROGRAMS_ALL: [Prog; 18] = [
    Prog::MmTrain,
    Prog::Xent,
    Prog::LnFwd,
    Prog::LnBack,
    Prog::Colsum,
    Prog::GeluTr,
    Prog::AttnFwd,
    Prog::AttnBack,
    Prog::EmbFwd,
    Prog::EmbBack,
    Prog::Gnorm,
    Prog::OptStep,
    Prog::Vadd,
    Prog::AttnP,
    Prog::AttnFwdTile2,
    Prog::AttnBackTile0_2,
    Prog::AttnBackTile2,
    Prog::AttnPTile,
];

/// What a [`Plan::new_tiles`] step launches: the 11 programs without attention + the three tiled attention programs.
pub const PROGRAMS_TILES: [Prog; 14] = [
    Prog::MmTrain,
    Prog::Xent,
    Prog::LnFwd,
    Prog::LnBack,
    Prog::Colsum,
    Prog::GeluTr,
    Prog::EmbFwd,
    Prog::EmbBack,
    Prog::Gnorm,
    Prog::OptStep,
    Prog::Vadd,
    Prog::AttnFwdTile2,
    Prog::AttnBackTile0_2,
    Prog::AttnBackTile2,
];

/// [`PROGRAMS_TILES`] + [`Prog::AttnPTile`]: what a [`Plan::new_p_only_tiles`] step launches.
pub const PROGRAMS_TILES_P_ONLY: [Prog; 15] = [
    Prog::MmTrain,
    Prog::Xent,
    Prog::LnFwd,
    Prog::LnBack,
    Prog::Colsum,
    Prog::GeluTr,
    Prog::EmbFwd,
    Prog::EmbBack,
    Prog::Gnorm,
    Prog::OptStep,
    Prog::Vadd,
    Prog::AttnFwdTile2,
    Prog::AttnBackTile0_2,
    Prog::AttnBackTile2,
    Prog::AttnPTile,
];

/// [`PROGRAMS`] + [`Prog::AttnP`] at the end (so every old program keeps its [`Prog::index`]): what a
/// [`Plan::new_p_only`] step uses (`Plan::programs`).
pub const PROGRAMS_P_ONLY: [Prog; 14] = [
    Prog::MmTrain,
    Prog::Xent,
    Prog::LnFwd,
    Prog::LnBack,
    Prog::Colsum,
    Prog::GeluTr,
    Prog::AttnFwd,
    Prog::AttnBack,
    Prog::EmbFwd,
    Prog::EmbBack,
    Prog::Gnorm,
    Prog::OptStep,
    Prog::Vadd,
    Prog::AttnP,
];

impl Prog {
    pub fn code(self) -> &'static [u8] {
        match self {
            Prog::MmTrain => mm::MM_TRAIN,
            Prog::Xent => xe::XENT,
            Prog::LnFwd => lf::LN_FWD,
            Prog::LnBack => lb::LN_BACK,
            Prog::Colsum => cs::COLSUM,
            Prog::GeluTr => gt::GELU_TR,
            Prog::AttnFwd => af::ATTN_FWD,
            Prog::AttnBack => ab::ATTN_BACK,
            Prog::EmbFwd => ef::EMB_FWD,
            Prog::EmbBack => eb::EMB_BACK,
            Prog::Gnorm => gn::GNORM,
            Prog::OptStep => os::OPT_STEP,
            Prog::Vadd => kronos_layer::VADD,
            Prog::AttnP => ap::ATTN_P,
            Prog::AttnFwdTile2 => aft2::ATTN_FWD_TILE2,
            Prog::AttnBackTile0_2 => abt02::ATTN_BACK_TILE0_2,
            Prog::AttnBackTile2 => abt2::ATTN_BACK_TILE_2,
            Prog::AttnPTile => apt::ATTN_P_TILE,
        }
    }

    /// The SHA-256 the OS checks before loading (refuse otherwise).
    pub fn sha256(self) -> &'static str {
        match self {
            Prog::MmTrain => mm::MM_TRAIN_SHA256,
            Prog::Xent => xe::XENT_SHA256,
            Prog::LnFwd => lf::LN_FWD_SHA256,
            Prog::LnBack => lb::LN_BACK_SHA256,
            Prog::Colsum => cs::COLSUM_SHA256,
            Prog::GeluTr => gt::GELU_TR_SHA256,
            Prog::AttnFwd => af::ATTN_FWD_SHA256,
            Prog::AttnBack => ab::ATTN_BACK_SHA256,
            Prog::EmbFwd => ef::EMB_FWD_SHA256,
            Prog::EmbBack => eb::EMB_BACK_SHA256,
            Prog::Gnorm => gn::GNORM_SHA256,
            Prog::OptStep => os::OPT_STEP_SHA256,
            Prog::Vadd => VADD_SHA256,
            Prog::AttnP => ap::ATTN_P_SHA256,
            Prog::AttnFwdTile2 => aft2::ATTN_FWD_TILE2_SHA256,
            Prog::AttnBackTile0_2 => abt02::ATTN_BACK_TILE0_2_SHA256,
            Prog::AttnBackTile2 => abt2::ATTN_BACK_TILE_2_SHA256,
            Prog::AttnPTile => apt::ATTN_P_TILE_SHA256,
        }
    }

    pub fn registers(self) -> u32 {
        match self {
            Prog::MmTrain => mm::REGISTERS,
            Prog::Xent => xe::REGISTERS,
            Prog::LnFwd => lf::REGISTERS,
            Prog::LnBack => lb::REGISTERS,
            Prog::Colsum => cs::REGISTERS,
            Prog::GeluTr => gt::REGISTERS,
            Prog::AttnFwd => af::REGISTERS,
            Prog::AttnBack => ab::REGISTERS,
            Prog::EmbFwd => ef::REGISTERS,
            Prog::EmbBack => eb::REGISTERS,
            Prog::Gnorm => gn::REGISTERS,
            Prog::OptStep => os::REGISTERS,
            Prog::Vadd => kronos_layer::Prog::Vadd.registers(),
            Prog::AttnP => ap::REGISTERS,
            Prog::AttnFwdTile2 => aft2::REGISTERS,
            Prog::AttnBackTile0_2 => abt02::REGISTERS,
            Prog::AttnBackTile2 => abt2::REGISTERS,
            Prog::AttnPTile => apt::REGISTERS,
        }
    }

    pub fn block(self) -> [u32; 3] {
        match self {
            Prog::MmTrain => mm::BLOCK,
            Prog::Xent => xe::BLOCK,
            Prog::LnFwd => lf::BLOCK,
            Prog::LnBack => lb::BLOCK,
            Prog::Colsum => cs::BLOCK,
            Prog::GeluTr => gt::BLOCK,
            Prog::AttnFwd => af::BLOCK,
            Prog::AttnBack => ab::BLOCK,
            Prog::EmbFwd => ef::BLOCK,
            Prog::EmbBack => eb::BLOCK,
            Prog::Gnorm => gn::BLOCK,
            Prog::OptStep => os::BLOCK,
            Prog::Vadd => [kronos_layer::Prog::Vadd.block(), 1, 1],
            Prog::AttnP => ap::BLOCK,
            Prog::AttnFwdTile2 => aft2::BLOCK,
            Prog::AttnBackTile0_2 => abt02::BLOCK,
            Prog::AttnBackTile2 => abt2::BLOCK,
            Prog::AttnPTile => apt::BLOCK,
        }
    }

    pub fn shared(self) -> Option<SharedMemory> {
        match self {
            Prog::MmTrain => Some(mm::SHARED),
            Prog::Xent => Some(xe::SHARED),
            Prog::LnFwd => Some(lf::SHARED),
            Prog::LnBack => Some(lb::SHARED),
            Prog::Gnorm => Some(gn::SHARED),
            Prog::AttnFwdTile2 => Some(aft2::SHARED),
            Prog::AttnBackTile0_2 => Some(abt02::SHARED),
            Prog::AttnBackTile2 => Some(abt2::SHARED),
            Prog::AttnPTile => Some(apt::SHARED),
            _ => None,
        }
    }

    /// Short name for the report (the `.comp` file's name).
    pub fn name(self) -> &'static str {
        match self {
            Prog::MmTrain => "mm_train",
            Prog::Xent => "xent",
            Prog::LnFwd => "ln_fwd",
            Prog::LnBack => "ln_back",
            Prog::Colsum => "colsum",
            Prog::GeluTr => "gelu_tr",
            Prog::AttnFwd => "attn_fwd",
            Prog::AttnBack => "attn_back",
            Prog::EmbFwd => "emb_fwd",
            Prog::EmbBack => "emb_back",
            Prog::Gnorm => "gnorm",
            Prog::OptStep => "opt_step",
            Prog::Vadd => "vadd",
            Prog::AttnP => "attn_p",
            Prog::AttnFwdTile2 => "attn_fwd_tile2",
            Prog::AttnBackTile0_2 => "attn_back_tile0_2",
            Prog::AttnBackTile2 => "attn_back_tile_2",
            Prog::AttnPTile => "attn_p_tile",
        }
    }

    /// Index in [`PROGRAMS_ALL`] (= the index in [`PROGRAMS`] / [`PROGRAMS_P_ONLY`] for their programs).
    pub fn index(self) -> usize {
        PROGRAMS_ALL.iter().position(|&p| p == self).unwrap_or(0)
    }
}

/// Checks every program's bytes against its SHA-256 (what the OS does before loading; refuse on a mismatch).
pub fn check_programs() -> GpuResult<()> {
    for p in PROGRAMS_ALL {
        if hex(&crate::sha256::digest(p.code())) != p.sha256() {
            return Err(bad("pas-aur: SHA-256 al unui program nu se potriveste"));
        }
    }
    Ok(())
}

fn hex(d: &[u8; 32]) -> String {
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

// ------------------------------------------------------------------ one launch

/// One launch of the step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub prog: Prog,
    /// Constant buffer 0, 0x100 bytes (the program's arguments at 0x28…, zero after them).
    pub cbuf: [u8; 0x100],
    pub grid: [u32; 3],
    /// What this launch computes (`inainte:…`, `inapoi:…`, `opt:…`) — for the report and the tests.
    pub tag: &'static str,
    /// The layer, if the launch belongs to one.
    pub layer: Option<u16>,
}

impl Launch {
    /// One launch of `prog` with the constant buffer `cb` (≤ 0x100 bytes; `None` = arguments the program refused) —
    /// the same checks as the step's own launches (C-157, `adam_ram`).
    pub fn new(
        prog: Prog,
        cb: Option<&[u8]>,
        grid: [u32; 3],
        tag: &'static str,
        layer: Option<usize>,
    ) -> GpuResult<Self> {
        launch(prog, cb, grid, tag, layer)
    }

    /// The QMD bytes with the program at `prog_va` and this launch's cbuf at `cbuf_va` (both 256-byte aligned).
    pub fn qmd(&self, prog_va: u64, cbuf_va: u64) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
        let q = ComputeQmd {
            program_va: prog_va,
            registers: self.prog.registers(),
            cbuf0_va: cbuf_va,
            cbuf0_size: 0x100,
            grid: self.grid,
            block: self.prog.block(),
            release: None,
        };
        match self.prog.shared() {
            Some(s) => q.to_bytes_with_shared(&s),
            None => q.to_bytes(),
        }
    }
}

fn launch(
    prog: Prog,
    cb: Option<&[u8]>,
    grid: [u32; 3],
    tag: &'static str,
    layer: Option<usize>,
) -> GpuResult<Launch> {
    let cb = cb.ok_or(bad("pas-aur: argumentele unei lansari refuzate"))?;
    if cb.len() > 0x100 || grid.contains(&0) {
        return Err(bad("pas-aur: lansare invalida"));
    }
    let mut cbuf = [0u8; 0x100];
    cbuf[..cb.len()].copy_from_slice(cb);
    Ok(Launch {
        prog,
        cbuf,
        grid,
        tag,
        layer: layer.map(|l| l as u16),
    })
}

/// The control zone of one step: launch `k`'s cbuf at `zone_va + k·SLOT`, its QMD at `+ 0x100`. Returns the zone's
/// bytes (write them before the submission) and the QMD addresses, in order, for `transfer::encode_sequence_fresh`.
/// `prog_va(p)` = where the OS loaded program `p`.
pub fn pack(
    launches: &[Launch],
    prog_va: impl Fn(Prog) -> u64,
    zone_va: u64,
) -> GpuResult<(Vec<u8>, Vec<u64>)> {
    if !zone_va.is_multiple_of(256) || launches.is_empty() {
        return Err(bad("pas-aur: zona de control"));
    }
    let mut bytes = alloc::vec![0u8; launches.len() * SLOT as usize];
    let mut qmds = Vec::with_capacity(launches.len());
    for (k, l) in launches.iter().enumerate() {
        let at = k * SLOT as usize;
        let cb_va = zone_va + at as u64;
        let pv = prog_va(l.prog);
        if !pv.is_multiple_of(256) {
            return Err(bad("pas-aur: programul nealiniat"));
        }
        let q = l.qmd(pv, cb_va)?;
        bytes[at..at + 0x100].copy_from_slice(&l.cbuf);
        bytes[at + 0x100..at + 0x100 + q.len()].copy_from_slice(&q);
        qmds.push(cb_va + 0x100);
    }
    Ok((bytes, qmds))
}

/// The grid of a forward-attention-shaped launch (`attn_fwd::Args`): [⌈T/64⌉, H, B] for the tiled programs, the
/// program's own grid otherwise.
fn attn_fwd_grid(prog: Prog, a: &af::Args) -> [u32; 3] {
    match prog {
        Prog::AttnFwdTile2 => aft2::grid(a),
        Prog::AttnPTile => apt::grid(a),
        _ => a.grid(),
    }
}

// ------------------------------------------------------------------ the arena

/// What one LayerNorm keeps for the backward pass: its input, μ and rstd.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LnSaved {
    pub x: u64,
    pub mean: u64,
    pub rstd: u64,
}

/// What one layer keeps for the backward pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerBufs {
    pub ln1: LnSaved,
    pub h1: u64,
    pub qkv: u64,
    pub p: u64,
    pub att: u64,
    pub ln2: LnSaved,
    pub h2: u64,
    pub u: u64,
    pub act: u64,
}

/// One `emb_back` segment list on the board: (off, dst, rows, count).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SegVas {
    pub off: u64,
    pub dst: u64,
    pub rows: u64,
    pub count: usize,
}

/// The board memory of the step: every address absolute (`base` + offset), 256-byte aligned.
#[derive(Clone, Debug, PartialEq)]
pub struct Plan {
    pub arch: Arch,
    pub batch: usize,
    pub len: usize,
    pub base: u64,
    /// Offset (in elements) of every tensor in the flat buffers (`named_parameters()` order).
    pub off: Vec<usize>,
    pub n_params: usize,
    pub w: u64,
    pub g: u64,
    pub m: u64,
    pub v: u64,
    pub tok: u64,
    pub tgt: u64,
    pub seg_tok: SegVas,
    pub seg_pos: SegVas,
    pub layers: Vec<LayerBufs>,
    /// The residual stream: x0, then (x_mid, x_out) per layer.
    pub xs: Vec<u64>,
    pub lnf: LnSaved,
    pub hf: u64,
    pub logits: u64,
    pub dlogits: u64,
    /// The per-row losses (`batch·len` f32), read back after every step.
    pub loss_rows: u64,
    pub dx: [u64; 2],
    pub dtmp: u64,
    pub dwide: u64,
    pub dwide2: u64,
    pub dqkv: u64,
    pub ds: u64,
    pub part: u64,
    /// `res[0]` = the gradient norm, `res[1]` = the clipping factor (`sgd`: 1, written once by the CPU).
    pub res: u64,
    /// The segment lists themselves (written by [`Plan::uploads`]).
    segs: [EmbSegments; 2],
    /// The program of the backward pass's P re-launches: `attn_fwd` ([`Plan::new_p_recalc`]) or `attn_p`
    /// ([`Plan::new_p_only`]); never launched by [`Plan::new`].
    p_prog: Prog,
    /// True for [`Plan::new_tiles`] / [`Plan::new_p_only_tiles`]: the attention launches use the tiled programs.
    tiles: bool,
    /// True for the `_mv_ram` plans (C-157): AdamW's m and v live in host RAM ([`crate::adam_ram`]); [`Plan::m`] and
    /// [`Plan::v`] are 0 and the arena has no room for them.
    mv_ram: bool,
    /// End of the arena (exclusive).
    pub end: u64,
}

struct Bump(u64);

impl Bump {
    /// `n` f32 / u32, 256-byte aligned, 256 bytes of gap after (as on the simulated board, T165).
    fn alloc(&mut self, n: usize) -> u64 {
        let va = self.0;
        self.0 += (4 * n as u64).div_ceil(256) * 256 + 256;
        va
    }
}

impl Plan {
    /// The arena for `arch` with a batch of `batch` texts of `len` tokens (`inputs` = the batch's tokens, for the
    /// `emb_back` segment lists), from `base` (256-byte aligned) on.
    pub fn new(arch: Arch, batch: usize, len: usize, inputs: &[u32], base: u64) -> GpuResult<Self> {
        Self::build(arch, batch, len, inputs, base, false, false)
    }

    /// The same arena as [`Plan::new`] but with **one** attention-probability buffer P shared by all layers (C-160
    /// point 7, `docs/C-160_P7_P_RECALCULAT.md`): the backward pass of layer l re-launches `attn_fwd` on the layer's
    /// kept `qkv` to rebuild its P (same program, same inputs, fixed summation order, no atomics → the same bits) when
    /// a later layer overwrote it. Saves (L − 1)·B·H·T² f32; costs [`Plan::p_relaunches`] launches per step.
    pub fn new_p_recalc(
        arch: Arch,
        batch: usize,
        len: usize,
        inputs: &[u32],
        base: u64,
    ) -> GpuResult<Self> {
        Self::build(arch, batch, len, inputs, base, true, false)
    }

    /// The arena of [`Plan::new_p_recalc`] (one shared P), but the backward pass rebuilds a lost P with **`attn_p`**
    /// (P without P·V, ≈ 0,51 of `attn_fwd`, DEDUS) on exactly the forward launch's arguments — the same operations in
    /// the same order, so the same bits (`docs/C-160_P7_ATTN_P_IN_PAS.md`); `att` is not rewritten. Its programs are
    /// [`PROGRAMS_P_ONLY`] (`Plan::programs`).
    pub fn new_p_only(
        arch: Arch,
        batch: usize,
        len: usize,
        inputs: &[u32],
        base: u64,
    ) -> GpuResult<Self> {
        let mut plan = Self::build(arch, batch, len, inputs, base, true, false)?;
        plan.p_prog = Prog::AttnP;
        Ok(plan)
    }

    /// The arena of [`Plan::new`] (P kept per layer), but every attention launch on the **tiled** programs
    /// (`docs/C-160_P7_PLACI_IN_PAS.md`): `attn_fwd_tile2`, then `attn_back_tile0_2` (mode 0) and `attn_back_tile_2`
    /// (mode 1), the same cbufs, the grid [⌈T/64⌉, H, B]. Needs d = C / H ≤ 64. Its programs are [`PROGRAMS_TILES`].
    pub fn new_tiles(
        arch: Arch,
        batch: usize,
        len: usize,
        inputs: &[u32],
        base: u64,
    ) -> GpuResult<Self> {
        let mut plan = Self::build(arch, batch, len, inputs, base, false, false)?;
        plan.set_tiles(Prog::AttnFwdTile2)?;
        Ok(plan)
    }

    /// C-157 (`docs/C-157_ADAM_RAM.md`): [`Plan::new_tiles`] **without** AdamW's m and v on the board (8 bytes per
    /// parameter less): they live in host RAM and come up layer by layer ([`crate::adam_ram::RamMv`]). The AdamW step
    /// of such a plan is [`Plan::step_until_opt`] + [`crate::adam_ram::RamMv::opt_launches`]; [`Plan::step`] refuses it
    /// (the `sgd` recipe, which never touches m and v, still works).
    pub fn new_tiles_mv_ram(
        arch: Arch,
        batch: usize,
        len: usize,
        inputs: &[u32],
        base: u64,
    ) -> GpuResult<Self> {
        let mut plan = Self::build(arch, batch, len, inputs, base, false, true)?;
        plan.set_tiles(Prog::AttnFwdTile2)?;
        Ok(plan)
    }

    /// C-157: [`Plan::new_p_only_tiles`] without m and v on the board (as [`Plan::new_tiles_mv_ram`]).
    pub fn new_p_only_tiles_mv_ram(
        arch: Arch,
        batch: usize,
        len: usize,
        inputs: &[u32],
        base: u64,
    ) -> GpuResult<Self> {
        let mut plan = Self::build(arch, batch, len, inputs, base, true, true)?;
        plan.set_tiles(Prog::AttnPTile)?;
        Ok(plan)
    }

    /// True for the `_mv_ram` plans: m and v are in host RAM, not in the arena.
    pub fn mv_in_ram(&self) -> bool {
        self.mv_ram
    }

    /// The arena of [`Plan::new_p_only`] (one shared P) with the tiled attention of [`Plan::new_tiles`]; a lost P is
    /// rebuilt with **`attn_p_tile`** (phases A and B of `attn_fwd_tile2`, the same P bits, T186). Its programs are
    /// [`PROGRAMS_TILES_P_ONLY`].
    pub fn new_p_only_tiles(
        arch: Arch,
        batch: usize,
        len: usize,
        inputs: &[u32],
        base: u64,
    ) -> GpuResult<Self> {
        let mut plan = Self::build(arch, batch, len, inputs, base, true, false)?;
        plan.set_tiles(Prog::AttnPTile)?;
        Ok(plan)
    }

    fn set_tiles(&mut self, p_prog: Prog) -> GpuResult<()> {
        let (cc, h) = (self.arch.n_emb, self.arch.heads);
        if cc / h > aft2::MAX_D as usize {
            return Err(bad("pas-aur: atentia pe placi cere d <= 64"));
        }
        self.tiles = true;
        self.p_prog = p_prog;
        Ok(())
    }

    /// True for the tiled plans ([`Plan::new_tiles`], [`Plan::new_p_only_tiles`]).
    pub fn tiles(&self) -> bool {
        self.tiles
    }

    /// The programs this plan's steps launch (load these): [`PROGRAMS_P_ONLY`] for [`Plan::new_p_only`],
    /// [`PROGRAMS_TILES`] / [`PROGRAMS_TILES_P_ONLY`] for the tiled plans, [`PROGRAMS`] otherwise.
    pub fn programs(&self) -> &'static [Prog] {
        match (self.tiles, self.p_prog) {
            (true, Prog::AttnPTile) => &PROGRAMS_TILES_P_ONLY,
            (true, _) => &PROGRAMS_TILES,
            (false, Prog::AttnP) => &PROGRAMS_P_ONLY,
            _ => &PROGRAMS,
        }
    }

    /// The program of the P re-launches (`attn_fwd` or `attn_p`).
    pub fn p_program(&self) -> Prog {
        self.p_prog
    }

    /// True when the P buffer of layer `l` is overwritten by a later layer's forward pass (then the backward pass
    /// rebuilds it first). Depends only on the addresses: always false for [`Plan::new`].
    pub fn p_rebuilt(&self, l: usize) -> bool {
        self.layers
            .get(l)
            .is_some_and(|wl| self.layers[l + 1..].iter().any(|x| x.p == wl.p))
    }

    /// How many `attn_fwd` re-launches one step has (0 for [`Plan::new`], L − 1 for [`Plan::new_p_recalc`]).
    pub fn p_relaunches(&self) -> usize {
        (0..self.layers.len())
            .filter(|&l| self.p_rebuilt(l))
            .count()
    }

    fn build(
        arch: Arch,
        batch: usize,
        len: usize,
        inputs: &[u32],
        base: u64,
        shared_p: bool,
        mv_ram: bool,
    ) -> GpuResult<Self> {
        let (cc, v, h) = (arch.n_emb, arch.vocab, arch.heads);
        let n = batch * len;
        if !base.is_multiple_of(256)
            || n == 0
            || inputs.len() != n
            || len > arch.ctx
            || h == 0
            || cc % h != 0
            || arch.layers == 0
        {
            return Err(bad("pas-aur: forma lotului sau baza"));
        }
        let st =
            EmbSegments::tokens(inputs, v).ok_or(bad("pas-aur: simbol in afara vocabularului"))?;
        let sp = EmbSegments::positions(batch, len).ok_or(bad("pas-aur: pozitii"))?;
        if !st.check(n, v) || !sp.check(n, arch.ctx) {
            return Err(bad("pas-aur: segmentele"));
        }
        let f = 4 * cc;
        let mut off = Vec::new();
        let mut at = 0;
        for t in arch.tensors() {
            off.push(at);
            at += t.elements();
        }
        let mut b = Bump(base);
        let (w, g) = (b.alloc(at), b.alloc(at));
        let (m, vv) = if mv_ram {
            (0, 0)
        } else {
            (b.alloc(at), b.alloc(at))
        };
        let (tok, tgt) = (b.alloc(n), b.alloc(n));
        let seg = |b: &mut Bump, s: &EmbSegments| SegVas {
            off: b.alloc(s.off.len()),
            dst: b.alloc(s.dst.len()),
            rows: b.alloc(s.rows.len()),
            count: s.len(),
        };
        let seg_tok = seg(&mut b, &st);
        let seg_pos = seg(&mut b, &sp);
        let mut xs = alloc::vec![b.alloc(n * cc)];
        let mut layers: Vec<LayerBufs> = Vec::new();
        for _ in 0..arch.layers {
            let xin = *xs.last().unwrap_or(&0);
            let ln1 = LnSaved {
                x: xin,
                mean: b.alloc(n),
                rstd: b.alloc(n),
            };
            let xmid = b.alloc(n * cc);
            let ln2 = LnSaved {
                x: xmid,
                mean: b.alloc(n),
                rstd: b.alloc(n),
            };
            layers.push(LayerBufs {
                ln1,
                h1: b.alloc(n * cc),
                qkv: b.alloc(n * 3 * cc),
                p: match layers.first() {
                    Some(first) if shared_p => first.p,
                    _ => b.alloc(batch * h * len * len),
                },
                att: b.alloc(n * cc),
                ln2,
                h2: b.alloc(n * cc),
                u: b.alloc(n * f),
                act: b.alloc(n * f),
            });
            xs.push(xmid);
            xs.push(b.alloc(n * cc));
        }
        let lnf = LnSaved {
            x: *xs.last().unwrap_or(&0),
            mean: b.alloc(n),
            rstd: b.alloc(n),
        };
        let hf = b.alloc(n * cc);
        let logits = b.alloc(n * v);
        let dlogits = b.alloc(n * v);
        let loss_rows = b.alloc(n);
        let dx = [b.alloc(n * cc), b.alloc(n * cc)];
        let dtmp = b.alloc(n * cc);
        let dwide = b.alloc(n * f);
        let dwide2 = b.alloc(n * f);
        let dqkv = b.alloc(n * 3 * cc);
        let ds = b.alloc(batch * h * len * len);
        let blocks = (at as u32).div_ceil(GNORM_CHUNK) as usize;
        let part = b.alloc(blocks.max(4096));
        let res = b.alloc(64);
        Ok(Plan {
            arch,
            batch,
            len,
            base,
            off,
            n_params: at,
            w,
            g,
            m,
            v: vv,
            tok,
            tgt,
            seg_tok,
            seg_pos,
            layers,
            xs,
            lnf,
            hf,
            logits,
            dlogits,
            loss_rows,
            dx,
            dtmp,
            dwide,
            dwide2,
            dqkv,
            ds,
            part,
            res,
            segs: [st, sp],
            p_prog: Prog::AttnFwd,
            tiles: false,
            mv_ram,
            end: b.0,
        })
    }

    /// Bytes of board memory the arena needs (from `base`).
    pub fn bytes(&self) -> u64 {
        self.end - self.base
    }

    /// The board address of tensor `i` of the weights (`base` = [`Plan::w`]) or of the gradients ([`Plan::g`]).
    pub fn t(&self, base: u64, i: usize) -> u64 {
        base + 4 * self.off[i] as u64
    }

    /// Index of tensor `k` of layer `l` (12 per layer, after the two tables).
    pub fn li(l: usize, k: usize) -> usize {
        2 + train::PER_LAYER * l + k
    }

    /// What the CPU writes **once**, before step 1: (address, bytes). The initial weights (flat), zeros for the
    /// gradients (the rows of `emb_pozitie`'s gradient past the text length are never written and must stay 0) and
    /// for m, v, the inputs and targets, the two segment lists, and `res = [0, 1]` (the `sgd` factor).
    pub fn uploads(
        &self,
        params: &[Vec<f32>],
        inputs: &[u32],
        targets: &[u32],
    ) -> GpuResult<Vec<(u64, Vec<u8>)>> {
        self.arch.check_params(params)?;
        let n = self.batch * self.len;
        if inputs.len() != n
            || targets.len() != n
            || targets.iter().any(|&t| t as usize >= self.arch.vocab)
        {
            return Err(bad("pas-aur: intrarile sau tintele"));
        }
        if EmbSegments::tokens(inputs, self.arch.vocab).as_ref() != Some(&self.segs[0]) {
            return Err(bad("pas-aur: alt lot decat cel al planului"));
        }
        let f =
            |v: &mut dyn Iterator<Item = f32>| v.flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let u = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let zeros = alloc::vec![0u8; 4 * self.n_params];
        let mut out = alloc::vec![
            (self.w, f(&mut params.iter().flatten().copied())),
            (self.g, zeros.clone()),
        ];
        // C-157: a `_mv_ram` plan has no m, v on the board (`adam_ram::RamMv::uploads` zeroes them in RAM).
        if !self.mv_ram {
            out.push((self.m, zeros.clone()));
            out.push((self.v, zeros));
        }
        out.push((self.tok, u(inputs)));
        out.push((self.tgt, u(targets)));
        for (s, vas) in self.segs.iter().zip([self.seg_tok, self.seg_pos]) {
            out.push((vas.off, u(&s.off)));
            out.push((vas.dst, u(&s.dst)));
            out.push((vas.rows, u(&s.rows)));
        }
        out.push((self.res, f(&mut [0.0f32, 1.0].into_iter())));
        Ok(out)
    }

    /// The launches of step `step` (1, 2, …) of `recipe`, in order. `opt` = the recipe's AdamW (its scalars for
    /// `step`; unused at `sgd`). Every tensor stays on the board; nothing is read in the middle of a step.
    pub fn step(&self, recipe: Recipe, opt: &AdamW, step: u32) -> GpuResult<Vec<Launch>> {
        if step == 0 {
            return Err(bad("pas-aur: pasii incep de la 1"));
        }
        if self.mv_ram && recipe == Recipe::AdamW513 {
            return Err(bad("pas-aur: m, v sunt in RAM (step_until_opt + adam_ram)"));
        }
        let mut out = Vec::with_capacity(32 * self.arch.layers + 16);
        self.forward_backward(&mut out, 1.0)?;
        self.optimize(&mut out, recipe, opt, step, None)?;
        Ok(out)
    }

    /// C-157 (`docs/C-157_ADAM_RAM.md`): the launches of step `step` of `adamw_513` **up to** the optimizer step (the
    /// forward and backward passes and the two norm launches: every launch of [`Plan::step`] but the last), and the
    /// optimizer step's arguments over the whole flat buffers with `m`, `v` left 0 — [`crate::adam_ram::RamMv`] splits
    /// them into one launch per layer on its staging buffers. Works on every plan (on one with m, v on the board it is
    /// exactly [`Plan::step`] without its last launch).
    pub fn step_until_opt(&self, opt: &AdamW, step: u32) -> GpuResult<(Vec<Launch>, os::Args)> {
        if step == 0 {
            return Err(bad("pas-aur: pasii incep de la 1"));
        }
        let mut out = Vec::with_capacity(32 * self.arch.layers + 16);
        self.forward_backward(&mut out, 1.0)?;
        let mut a = self.opt_args(&mut out, Recipe::AdamW513, opt, step, None)?;
        a.m = 0;
        a.v = 0;
        Ok((out, a))
    }

    /// Launches per step: 31 per layer + 13 at `adamw_513` (11 at `sgd`).
    pub fn launches_per_step(&self, recipe: Recipe) -> usize {
        31 * self.arch.layers
            + self.p_relaunches()
            + if recipe == Recipe::AdamW513 { 13 } else { 11 }
    }

    #[allow(clippy::too_many_arguments)]
    fn mm(
        out: &mut Vec<Launch>,
        (a, b, c, bias): (u64, u64, u64, u64),
        (m, n, k): (usize, usize, usize),
        flags: u32,
        tag: &'static str,
        layer: Option<usize>,
    ) -> GpuResult<()> {
        let x = mm::Args {
            a,
            b,
            c,
            bias,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            flags: flags | if bias != 0 { mm::BIAS } else { 0 },
        };
        out.push(launch(
            Prog::MmTrain,
            x.cbuf().as_ref().map(|b| &b[..]),
            x.grid(),
            tag,
            layer,
        )?);
        Ok(())
    }

    fn vadd(
        out: &mut Vec<Launch>,
        (a, b, o): (u64, u64, u64),
        n: usize,
        tag: &'static str,
        layer: Option<usize>,
    ) -> GpuResult<()> {
        let cb = kronos_layer::cbuf(&[a, b, o], &[n as u32]);
        out.push(launch(
            Prog::Vadd,
            Some(&cb[..0x44]),
            [(n as u32).div_ceil(256), 1, 1],
            tag,
            layer,
        )?);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn ln_fwd(
        out: &mut Vec<Launch>,
        s: &LnSaved,
        gamma: u64,
        beta: u64,
        y: u64,
        rows: usize,
        c: usize,
        tag: &'static str,
        layer: Option<usize>,
    ) -> GpuResult<()> {
        let a = lf::Args {
            x: s.x,
            gamma,
            beta,
            y,
            mean: s.mean,
            rstd: s.rstd,
            rows: rows as u32,
            c: c as u32,
            eps: train::LN_EPS as f32,
        };
        out.push(launch(
            Prog::LnFwd,
            a.cbuf().as_ref().map(|b| &b[..]),
            a.grid(),
            tag,
            layer,
        )?);
        Ok(())
    }

    /// LayerNorm backward: dx (`ln_back`), then dγ, dβ (`colsum`, LayerNorm mode).
    #[allow(clippy::too_many_arguments)]
    fn ln_back(
        out: &mut Vec<Launch>,
        s: &LnSaved,
        (dy, gamma, dx, dg, db): (u64, u64, u64, u64, u64),
        rows: usize,
        c: usize,
        tags: [&'static str; 2],
        layer: Option<usize>,
    ) -> GpuResult<()> {
        let a = lb::Args {
            x: s.x,
            dy,
            gamma,
            mean: s.mean,
            rstd: s.rstd,
            dx,
            rows: rows as u32,
            c: c as u32,
        };
        out.push(launch(
            Prog::LnBack,
            a.cbuf().as_ref().map(|b| &b[..]),
            a.grid(),
            tags[0],
            layer,
        )?);
        let k = cs::Args {
            dy,
            x: s.x,
            mean: s.mean,
            rstd: s.rstd,
            s0: dg,
            s1: db,
            rows: rows as u32,
            c: c as u32,
            mode: cs::Mode::LayerNorm,
        };
        out.push(launch(
            Prog::Colsum,
            k.cbuf().as_ref().map(|b| &b[..]),
            k.grid(),
            tags[1],
            layer,
        )?);
        Ok(())
    }

    fn bias_grad(
        out: &mut Vec<Launch>,
        (dy, db): (u64, u64),
        rows: usize,
        c: usize,
        tag: &'static str,
        layer: Option<usize>,
    ) -> GpuResult<()> {
        let k = cs::Args {
            dy,
            x: 0,
            mean: 0,
            rstd: 0,
            s0: db,
            s1: 0,
            rows: rows as u32,
            c: c as u32,
            mode: cs::Mode::Bias,
        };
        out.push(launch(
            Prog::Colsum,
            k.cbuf().as_ref().map(|b| &b[..]),
            k.grid(),
            tag,
            layer,
        )?);
        Ok(())
    }

    fn gelu(
        out: &mut Vec<Launch>,
        (u, g, y): (u64, u64, u64),
        n: usize,
        mode: gt::Mode,
        tag: &'static str,
        layer: Option<usize>,
    ) -> GpuResult<()> {
        let a = gt::Args {
            u,
            g,
            y,
            n: n as u32,
            mode,
        };
        out.push(launch(
            Prog::GeluTr,
            a.cbuf().as_ref().map(|b| &b[..]),
            a.grid(),
            tag,
            layer,
        )?);
        Ok(())
    }

    /// Forward + loss + backward: every gradient ends up in `g` (the same order as T165).
    /// `loss_scale` multiplies d(logits) (1 = plain F32 step; S at the mixed step, see [`Plan::step_mixed`]).
    fn forward_backward(&self, out: &mut Vec<Launch>, loss_scale: f32) -> GpuResult<()> {
        let ar = self.arch;
        let (bsz, tl) = (self.batch, self.len);
        let (cc, v, h) = (ar.n_emb, ar.vocab, ar.heads);
        let n = bsz * tl;
        let f = 4 * cc;
        let scale = ab::Args::scale_for((cc / h) as u32);
        let (w, g) = (self.w, self.g);
        let lnf_i = 2 + train::PER_LAYER * ar.layers;

        // ---- forward
        let a = ef::Args {
            tok: self.tok,
            emb: self.t(w, 0),
            pos: self.t(w, 1),
            x: self.xs[0],
            batch: bsz as u32,
            c: cc as u32,
            t: tl as u32,
        };
        out.push(launch(
            Prog::EmbFwd,
            a.cbuf().as_ref().map(|b| &b[..]),
            a.grid(),
            "inainte:incorporari",
            None,
        )?);
        for l in 0..ar.layers {
            let wt = |k: usize| self.t(w, Self::li(l, k));
            let (xin, xmid, xout) = (self.xs[2 * l], self.xs[2 * l + 1], self.xs[2 * l + 2]);
            let wl = self.layers[l];
            let tmp = self.dtmp;
            let ly = Some(l);
            Self::ln_fwd(out, &wl.ln1, wt(0), wt(1), wl.h1, n, cc, "inainte:ln1", ly)?;
            Self::mm(
                out,
                (wl.h1, wt(2), wl.qkv, wt(3)),
                (n, 3 * cc, cc),
                mm::TB,
                "inainte:qkv",
                ly,
            )?;
            let a = af::Args {
                qkv: wl.qkv,
                p: wl.p,
                att: wl.att,
                batch: bsz as u32,
                t: tl as u32,
                c: cc as u32,
                heads: h as u32,
                scale,
            };
            let fwd = if self.tiles {
                Prog::AttnFwdTile2
            } else {
                Prog::AttnFwd
            };
            out.push(launch(
                fwd,
                a.cbuf().as_ref().map(|b| &b[..]),
                attn_fwd_grid(fwd, &a),
                "inainte:atentia",
                ly,
            )?);
            Self::mm(
                out,
                (wl.att, wt(4), tmp, wt(5)),
                (n, cc, cc),
                mm::TB,
                "inainte:out",
                ly,
            )?;
            Self::vadd(out, (xin, tmp, xmid), n * cc, "inainte:rezidual1", ly)?;
            Self::ln_fwd(out, &wl.ln2, wt(6), wt(7), wl.h2, n, cc, "inainte:ln2", ly)?;
            Self::mm(
                out,
                (wl.h2, wt(8), wl.u, wt(9)),
                (n, f, cc),
                mm::TB,
                "inainte:fc",
                ly,
            )?;
            Self::gelu(
                out,
                (wl.u, 0, wl.act),
                n * f,
                gt::Mode::Forward,
                "inainte:gelu",
                ly,
            )?;
            Self::mm(
                out,
                (wl.act, wt(10), tmp, wt(11)),
                (n, cc, f),
                mm::TB,
                "inainte:proj",
                ly,
            )?;
            Self::vadd(out, (xmid, tmp, xout), n * cc, "inainte:rezidual2", ly)?;
        }
        let lnf = self.lnf;
        Self::ln_fwd(
            out,
            &lnf,
            self.t(w, lnf_i),
            self.t(w, lnf_i + 1),
            self.hf,
            n,
            cc,
            "inainte:ln-final",
            None,
        )?;
        // The tied head: logits = hf · embᵀ (no bias).
        Self::mm(
            out,
            (self.hf, self.t(w, 0), self.logits, 0),
            (n, v, cc),
            mm::TB,
            "inainte:logit",
            None,
        )?;

        // ---- loss + d(logits) = (softmax − onehot) / N
        let a = xe::Args {
            logits: self.logits,
            targets: self.tgt,
            dlogits: self.dlogits,
            loss: self.loss_rows,
            rows: n as u32,
            v: v as u32,
            scale: loss_scale / n as f32,
        };
        out.push(launch(
            Prog::Xent,
            a.cbuf().as_ref().map(|b| &b[..]),
            a.grid(),
            "eroarea",
            None,
        )?);

        // ---- backward
        // The head's dW = dlogitsᵀ·hf goes straight into emb_token's gradient (emb_back adds the scatter on top).
        Self::mm(
            out,
            (self.dlogits, self.hf, self.t(g, 0), 0),
            (v, cc, n),
            mm::TA,
            "inapoi:cap-dW",
            None,
        )?;
        Self::mm(
            out,
            (self.dlogits, self.t(w, 0), self.dtmp, 0),
            (n, cc, v),
            0,
            "inapoi:cap-dX",
            None,
        )?;
        let cur = 0;
        Self::ln_back(
            out,
            &lnf,
            (
                self.dtmp,
                self.t(w, lnf_i),
                self.dx[cur],
                self.t(g, lnf_i),
                self.t(g, lnf_i + 1),
            ),
            n,
            cc,
            ["inapoi:ln-final-dx", "inapoi:ln-final-dgamma"],
            None,
        )?;
        for l in (0..ar.layers).rev() {
            let wt = |k: usize| self.t(w, Self::li(l, k));
            let gt_ = |k: usize| self.t(g, Self::li(l, k));
            let wl = self.layers[l];
            let (dx, dtmp, dwide, dwide2, dqkv, ds) = (
                self.dx[cur],
                self.dtmp,
                self.dwide,
                self.dwide2,
                self.dqkv,
                self.ds,
            );
            let dx2 = self.dx[1 - cur];
            let ly = Some(l);
            // MLP: x = x_mid + proj(gelu(fc(LN2(x_mid))))
            Self::mm(
                out,
                (dx, wt(10), dwide, 0),
                (n, f, cc),
                0,
                "inapoi:proj-dX",
                ly,
            )?;
            Self::mm(
                out,
                (dx, wl.act, gt_(10), 0),
                (cc, f, n),
                mm::TA,
                "inapoi:proj-dW",
                ly,
            )?;
            Self::bias_grad(out, (dx, gt_(11)), n, cc, "inapoi:proj-db", ly)?;
            Self::gelu(
                out,
                (wl.u, dwide, dwide2),
                n * f,
                gt::Mode::Backward,
                "inapoi:gelu",
                ly,
            )?;
            Self::mm(
                out,
                (dwide2, wt(8), dtmp, 0),
                (n, cc, f),
                0,
                "inapoi:fc-dX",
                ly,
            )?;
            Self::mm(
                out,
                (dwide2, wl.h2, gt_(8), 0),
                (f, cc, n),
                mm::TA,
                "inapoi:fc-dW",
                ly,
            )?;
            Self::bias_grad(out, (dwide2, gt_(9)), n, f, "inapoi:fc-db", ly)?;
            Self::ln_back(
                out,
                &wl.ln2,
                (dtmp, wt(6), dx2, gt_(6), gt_(7)),
                n,
                cc,
                ["inapoi:ln2-dx", "inapoi:ln2-dgamma"],
                ly,
            )?;
            Self::vadd(out, (dx, dx2, dx2), n * cc, "inapoi:rezidual-ln2", ly)?; // dx += d(x_mid) through LN2
            // Attention: x_mid = x_in + out(attn(in(LN1(x_in))))
            Self::mm(
                out,
                (dx2, wt(4), dtmp, 0),
                (n, cc, cc),
                0,
                "inapoi:out-dX",
                ly,
            )?;
            Self::mm(
                out,
                (dx2, wl.att, gt_(4), 0),
                (cc, cc, n),
                mm::TA,
                "inapoi:out-dW",
                ly,
            )?;
            Self::bias_grad(out, (dx2, gt_(5)), n, cc, "inapoi:out-db", ly)?;
            if self.p_rebuilt(l) {
                // A later layer's forward pass overwrote this layer's P: rebuild it, bit for bit (same program, same
                // qkv; att is rewritten with its own bits).
                let a = af::Args {
                    qkv: wl.qkv,
                    p: wl.p,
                    att: wl.att,
                    batch: bsz as u32,
                    t: tl as u32,
                    c: cc as u32,
                    heads: h as u32,
                    scale,
                };
                // `attn_p::Args` = `attn_fwd::Args`: the same cbuf and grid either way.
                out.push(launch(
                    self.p_prog,
                    a.cbuf().as_ref().map(|b| &b[..]),
                    attn_fwd_grid(self.p_prog, &a),
                    "inapoi:atentia-P",
                    ly,
                )?);
            }
            for (mode, tag) in [
                (ab::Mode::Rows, "inapoi:atentia-dS"),
                (ab::Mode::Columns, "inapoi:atentia-dqkv"),
            ] {
                let a = ab::Args {
                    qkv: wl.qkv,
                    p: wl.p,
                    dout: dtmp,
                    ds,
                    dqkv,
                    batch: bsz as u32,
                    t: tl as u32,
                    c: cc as u32,
                    heads: h as u32,
                    scale,
                    mode,
                };
                let (prog, grid) = match (self.tiles, mode) {
                    (false, _) => (Prog::AttnBack, a.grid()),
                    (true, ab::Mode::Rows) => (Prog::AttnBackTile0_2, abt02::grid(&a)),
                    (true, ab::Mode::Columns) => (Prog::AttnBackTile2, abt2::grid(&a)),
                };
                out.push(launch(
                    prog,
                    a.cbuf().as_ref().map(|b| &b[..]),
                    grid,
                    tag,
                    ly,
                )?);
            }
            Self::mm(
                out,
                (dqkv, wt(2), dtmp, 0),
                (n, cc, 3 * cc),
                0,
                "inapoi:qkv-dX",
                ly,
            )?;
            Self::mm(
                out,
                (dqkv, wl.h1, gt_(2), 0),
                (3 * cc, cc, n),
                mm::TA,
                "inapoi:qkv-dW",
                ly,
            )?;
            Self::bias_grad(out, (dqkv, gt_(3)), n, 3 * cc, "inapoi:qkv-db", ly)?;
            Self::ln_back(
                out,
                &wl.ln1,
                (dtmp, wt(0), dx, gt_(0), gt_(1)),
                n,
                cc,
                ["inapoi:ln1-dx", "inapoi:ln1-dgamma"],
                ly,
            )?;
            Self::vadd(out, (dx2, dx, dx), n * cc, "inapoi:rezidual-ln1", ly)?; // dx += d(x_in) through LN1
        }
        // Embeddings: positions (write), then tokens (add onto the head's dW).
        let dx = self.dx[cur];
        for (seg, base, mode, tag) in [
            (
                self.seg_pos,
                self.t(g, 1),
                eb::Mode::Write,
                "inapoi:pozitii",
            ),
            (
                self.seg_tok,
                self.t(g, 0),
                eb::Mode::Add,
                "inapoi:simboluri",
            ),
        ] {
            let a = eb::Args {
                dx,
                off: seg.off,
                dst: seg.dst,
                rows: seg.rows,
                g: base,
                segments: seg.count as u32,
                c: cc as u32,
                mode,
            };
            out.push(launch(
                Prog::EmbBack,
                a.cbuf().as_ref().map(|b| &b[..]),
                a.grid(),
                tag,
                None,
            )?);
        }
        Ok(())
    }

    /// Norm + clipping (`adamw_513`) or the factor 1 (`sgd`, already in `res[1]`), then the optimizer.
    fn optimize(
        &self,
        out: &mut Vec<Launch>,
        recipe: Recipe,
        opt: &AdamW,
        step: u32,
        loss_scale: Option<f32>,
    ) -> GpuResult<()> {
        let a = self.opt_args(out, recipe, opt, step, loss_scale)?;
        out.push(launch(
            Prog::OptStep,
            a.cbuf().as_ref().map(|b| &b[..]),
            a.grid(),
            "opt:pas",
            None,
        )?);
        Ok(())
    }

    /// The launches of the optimizer before its step (the norm, the mixed step's 1/S) pushed to `out`; returns the
    /// step's arguments.
    fn opt_args(
        &self,
        out: &mut Vec<Launch>,
        recipe: Recipe,
        opt: &AdamW,
        step: u32,
        loss_scale: Option<f32>,
    ) -> GpuResult<os::Args> {
        let n = self.n_params as u32;
        // At the mixed step the gradients are S times too big: the norm is compared with S·MAX_NORM (the same factor
        // c, up to the 1e-6 in the denominator) and c is divided by S (one exact 1×1×1 F32 product) before the step.
        let max_norm = MAX_NORM * loss_scale.unwrap_or(1.0);
        let (sc, mode) = match recipe {
            Recipe::Sgd => (OptScalars::sgd(SGD_LR), os::Mode::Sgd),
            Recipe::AdamW513 => {
                let blocks = n.div_ceil(GNORM_CHUNK);
                let a = gn::Args {
                    x: self.g,
                    part: self.part,
                    res: 0,
                    n,
                    chunk: GNORM_CHUNK,
                    mode: gn::Mode::Partial,
                    max_norm,
                };
                out.push(launch(
                    Prog::Gnorm,
                    a.cbuf().as_ref().map(|b| &b[..]),
                    a.grid(),
                    "opt:norma-parti",
                    None,
                )?);
                let fin = gn::Args {
                    x: self.part,
                    part: 0,
                    res: self.res,
                    n: blocks,
                    chunk: blocks.div_ceil(256) * 256,
                    mode: gn::Mode::Finish,
                    max_norm,
                };
                out.push(launch(
                    Prog::Gnorm,
                    fin.cbuf().as_ref().map(|b| &b[..]),
                    fin.grid(),
                    "opt:norma-taiere",
                    None,
                )?);
                (OptScalars::adamw(&opt.scalars(step)), os::Mode::AdamW)
            }
        };
        let coef = match loss_scale {
            None => self.res + 4,
            Some(_) => {
                Self::mm(
                    out,
                    (
                        self.res + 4,
                        self.res + RES_INV_SCALE,
                        self.res + RES_COEF_MIXED,
                        0,
                    ),
                    (1, 1, 1),
                    0,
                    TAG_UNSCALE,
                    None,
                )?;
                self.res + RES_COEF_MIXED
            }
        };
        let a = os::Args {
            g: self.g,
            coef,
            w: self.w,
            m: self.m,
            v: self.v,
            n,
            mode,
            scalars: [
                sc.neg_lr,
                sc.decay,
                sc.w1,
                sc.b2,
                sc.omb2,
                sc.bc2_sqrt,
                sc.neg_step,
                sc.eps,
            ],
        };
        Ok(a)
    }

    /// The flat f32 of the weights or the gradients (read back from [`Plan::w`] / [`Plan::g`], `4·n_params` bytes)
    /// split into tensors.
    pub fn unflat(&self, flat: &[f32]) -> GpuResult<Vec<Vec<f32>>> {
        if flat.len() != self.n_params {
            return Err(bad("pas-aur: marimea citita"));
        }
        Ok(self
            .arch
            .tensors()
            .iter()
            .zip(&self.off)
            .map(|(s, &o)| flat[o..o + s.elements()].to_vec())
            .collect())
    }
}

/// The f32 values of little-endian bytes.
pub fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

// ------------------------------------------------------------------ the verdict

/// What the board produced over `steps` steps (read back by the OS).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Outcome {
    /// Mean of the per-row losses of each step (summed on the CPU, in f64, rounded to f32 — as T165).
    pub losses: Vec<f32>,
    /// `res[0]` after each step (`adamw_513` only).
    pub norms: Vec<f32>,
    pub grad1: Vec<Vec<f32>>,
    pub after1: Vec<Vec<f32>>,
    pub after_last: Vec<Vec<f32>>,
    /// Board time of each step (ns, the records around the submission), if measured.
    pub step_ns: Vec<u64>,
}

impl Outcome {
    /// The mean of one step's per-row losses.
    pub fn mean_loss(rows: &[f32]) -> f32 {
        if rows.is_empty() {
            return f32::NAN;
        }
        (rows.iter().map(|&r| r as f64).sum::<f64>() / rows.len() as f64) as f32
    }
}

/// PyTorch's numbers (the hub's `pas_de_aur_mic`: `reteta.json` `pierderi` / `norme_gradient`, `grad_pas1/`,
/// `dupa_pas1/`, `dupa_pas10/`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Refs {
    pub losses: Vec<f64>,
    /// Empty at `sgd`.
    pub norms: Vec<f64>,
    pub grad1: Vec<Vec<f32>>,
    pub after1: Vec<Vec<f32>>,
    pub after_last: Vec<Vec<f32>>,
}

/// The worst deviation of a set of tensors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Dev {
    pub abs: f32,
    /// max over tensors of ‖ours − theirs‖ / ‖theirs‖.
    pub rel_norm: f64,
    /// Index of the tensor with the largest `abs`.
    pub at: usize,
    pub finite: bool,
}

/// Compares two lists of tensors (same shapes).
pub fn deviation(ours: &[Vec<f32>], theirs: &[Vec<f32>]) -> Dev {
    let mut d = Dev {
        finite: ours.len() == theirs.len() && !ours.is_empty(),
        ..Dev::default()
    };
    for (i, (o, t)) in ours.iter().zip(theirs).enumerate() {
        if o.len() != t.len() {
            d.finite = false;
            continue;
        }
        let (mut diff2, mut norm2) = (0.0f64, 0.0f64);
        for (&a, &b) in o.iter().zip(t) {
            let e = (a - b).abs();
            if !e.is_finite() {
                d.finite = false;
                continue;
            }
            if e > d.abs {
                d.abs = e;
                d.at = i;
            }
            diff2 += (e as f64) * (e as f64);
            norm2 += (b as f64) * (b as f64);
        }
        if norm2 > 0.0 {
            d.rel_norm = d.rel_norm.max(sqrt(diff2 / norm2));
        }
    }
    d
}

fn sqrt(x: f64) -> f64 {
    if x <= 0.0 || !x.is_finite() {
        return if x == 0.0 { 0.0 } else { f64::NAN };
    }
    let mut y = f64::from_bits((x.to_bits() >> 1) + (1023u64 << 51));
    for _ in 0..8 {
        y = 0.5 * (y + x / y);
    }
    y
}

/// The verdict of a run against PyTorch: every figure as ‰ of its tolerance (≤ 1000 = within; `u32::MAX` = not
/// finite / missing).
#[derive(Clone, Debug, PartialEq)]
pub struct Verdict {
    pub recipe: Recipe,
    pub steps: usize,
    pub launches_per_step: usize,
    /// Worst relative loss over the steps (tolerance `TOL_LOSS_REL`).
    pub loss_permille: u32,
    pub grad_abs_permille: u32,
    pub grad_rel_permille: u32,
    pub w1_permille: u32,
    pub wlast_permille: u32,
    /// Worst relative norm (tolerance 1e-5, as T165); `None` at `sgd`.
    pub norm_permille: Option<u32>,
    pub loss_first: f32,
    pub loss_last: f32,
    pub grad: Dev,
    pub w1: Dev,
    pub wlast: Dev,
    pub median_step_ns: Option<u64>,
}

/// The tolerance of the norms (relative, as T165).
pub const TOL_NORM_REL: f64 = 1e-5;

fn permille(dev: f64, tol: f64, ok: bool) -> u32 {
    if !ok || !dev.is_finite() {
        return u32::MAX;
    }
    let x = dev / tol * 1000.0;
    if x >= (u32::MAX - 1) as f64 {
        return u32::MAX - 1;
    }
    // no_std: no f64::ceil
    let t = x as u32;
    if (t as f64) < x { t + 1 } else { t }
}

/// Judges a run against PyTorch in the hub's tolerances. `out.losses` must have as many steps as `refs.losses` used
/// (the first `out.losses.len()`); the weights after the last step are compared with `refs.after_last` (the hub
/// gives them after step 10).
pub fn judge(recipe: Recipe, launches_per_step: usize, out: &Outcome, refs: &Refs) -> Verdict {
    let steps = out.losses.len();
    let mut lrel = 0.0f64;
    let mut lok = steps > 0 && refs.losses.len() >= steps;
    for (k, &o) in out.losses.iter().enumerate() {
        match refs.losses.get(k) {
            Some(&w) if o.is_finite() && w != 0.0 => {
                lrel = lrel.max((o as f64 - w).abs() / w.abs())
            }
            _ => lok = false,
        }
    }
    let grad = deviation(&out.grad1, &refs.grad1);
    let w1 = deviation(&out.after1, &refs.after1);
    let wlast = deviation(&out.after_last, &refs.after_last);
    let wtol = match recipe {
        Recipe::Sgd => train::TOL_SGD_W_ABS,
        Recipe::AdamW513 => train::TOL_ADAMW_W_ABS,
    } as f64;
    let norm_permille = match recipe {
        Recipe::Sgd => None,
        Recipe::AdamW513 => {
            let mut r = 0.0f64;
            let mut ok = out.norms.len() == steps && refs.norms.len() >= steps;
            for (&a, &b) in out.norms.iter().zip(&refs.norms) {
                if a.is_finite() && b != 0.0 {
                    r = r.max((a as f64 - b).abs() / b);
                } else {
                    ok = false;
                }
            }
            Some(permille(r, TOL_NORM_REL, ok))
        }
    };
    let mut ns = out.step_ns.clone();
    ns.sort_unstable();
    Verdict {
        recipe,
        steps,
        launches_per_step,
        loss_permille: permille(lrel, train::TOL_LOSS_REL, lok),
        grad_abs_permille: permille(grad.abs as f64, train::TOL_GRAD_ABS as f64, grad.finite),
        grad_rel_permille: permille(grad.rel_norm, train::TOL_GRAD_REL_NORM, grad.finite),
        w1_permille: permille(w1.abs as f64, wtol, w1.finite),
        wlast_permille: permille(wlast.abs as f64, wtol, wlast.finite),
        norm_permille,
        loss_first: out.losses.first().copied().unwrap_or(f32::NAN),
        loss_last: out.losses.last().copied().unwrap_or(f32::NAN),
        grad,
        w1,
        wlast,
        median_step_ns: ns.get(ns.len() / 2).copied(),
    }
}

impl Verdict {
    /// The worst figure (‰ of its tolerance).
    pub fn worst_permille(&self) -> u32 {
        [
            self.loss_permille,
            self.grad_abs_permille,
            self.grad_rel_permille,
            self.w1_permille,
            self.wlast_permille,
            self.norm_permille.unwrap_or(0),
        ]
        .into_iter()
        .max()
        .unwrap_or(u32::MAX)
    }

    /// Same numbers as PyTorch: every figure within its tolerance.
    pub fn passes(&self) -> bool {
        self.steps > 0 && self.worst_permille() <= 1000
    }

    /// The `kt pas-aur …` rows (values in the stick journal, read back by [`parse_rows`]).
    pub fn rows(&self) -> Vec<(&'static str, u32)> {
        let mut out = alloc::vec![
            (
                "kt pas-aur reteta (0 sgd, 1 adamw_513)",
                u32::from(self.recipe == Recipe::AdamW513)
            ),
            ("kt pas-aur pasi", self.steps as u32),
            ("kt pas-aur lansari pe pas", self.launches_per_step as u32),
            (
                "kt pas-aur verdict (1 aceleasi numere)",
                u32::from(self.passes())
            ),
            (
                "kt pas-aur cel mai rau (promile din toleranta)",
                self.worst_permille()
            ),
            ("kt pas-aur eroarea (promile)", self.loss_permille),
            (
                "kt pas-aur gradientul pasului 1 abs (promile)",
                self.grad_abs_permille
            ),
            (
                "kt pas-aur gradientul pasului 1 rel (promile)",
                self.grad_rel_permille
            ),
            (
                "kt pas-aur greutatile dupa pasul 1 (promile)",
                self.w1_permille
            ),
            (
                "kt pas-aur greutatile dupa ultimul pas (promile)",
                self.wlast_permille
            ),
            (
                "kt pas-aur eroarea pasului 1 (f32)",
                self.loss_first.to_bits()
            ),
            (
                "kt pas-aur eroarea ultimului pas (f32)",
                self.loss_last.to_bits()
            ),
        ];
        if let Some(p) = self.norm_permille {
            out.push(("kt pas-aur normele (promile)", p));
        }
        if let Some(ns) = self.median_step_ns {
            out.push((
                "kt pas-aur timp median pe pas (us)",
                (ns / 1000).min(u32::MAX as u64) as u32,
            ));
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
        let name = if self.recipe == Recipe::AdamW513 {
            "adamw_513"
        } else {
            "sgd"
        };
        let p = |v: u32| {
            if v == u32::MAX {
                String::from("NEFINIT")
            } else {
                format!("{v}")
            }
        };
        let mut out = alloc::vec![format!(
            "pas de aur mic {name} ({label}): {} — {} pasi x {} lansari, cel mai rau {} promile din toleranta",
            if self.passes() {
                "ACELEASI NUMERE ca PyTorch"
            } else {
                "NU TRECE"
            },
            self.steps,
            self.launches_per_step,
            p(self.worst_permille())
        )];
        out.push(format!(
            "pas de aur mic {name} ({label}): eroarea {} -> {} ({} promile); gradientul pasului 1 abs {:.2e} ({} promile), rel {:.2e} ({} promile); greutatile dupa pasul 1 {:.2e} ({} promile), dupa ultimul {:.2e} ({} promile){}",
            self.loss_first,
            self.loss_last,
            p(self.loss_permille),
            self.grad.abs,
            p(self.grad_abs_permille),
            self.grad.rel_norm,
            p(self.grad_rel_permille),
            self.w1.abs,
            p(self.w1_permille),
            self.wlast.abs,
            p(self.wlast_permille),
            match self.norm_permille {
                Some(v) => format!("; normele {} promile", p(v)),
                None => String::new(),
            }
        ));
        if let Some(ns) = self.median_step_ns {
            out.push(format!(
                "pas de aur mic {name} ({label}): timpul median al unui pas pe placa {} us",
                ns / 1000
            ));
        }
        out
    }
}

/// Reads the `kt pas-aur …` rows of a stick journal (`gpu: valoare kt pas-aur … = 0x…`), in order. A new
/// `kt pas-aur reteta …` row starts a new run.
pub fn parse_rows(text: &str) -> Vec<Vec<(String, u32)>> {
    let mut runs: Vec<Vec<(String, u32)>> = Vec::new();
    for line in text.lines() {
        let Some(rest) = line.split("valoare kt pas-aur ").nth(1) else {
            continue;
        };
        let Some((name, val)) = rest.rsplit_once(" = 0x") else {
            continue;
        };
        let Ok(v) = u32::from_str_radix(val.trim(), 16) else {
            continue;
        };
        if name.starts_with("reteta") || runs.is_empty() {
            runs.push(Vec::new());
        }
        if let Some(r) = runs.last_mut() {
            r.push((String::from(name), v));
        }
    }
    runs
}

// ------------------------------------------------------------------ C-160 point 7, third part: the mixed step

/// The loss scale of the mixed step (`docs/C-160_P7_PAS_MIXT.md` §0): d(logits) ≈ 6·10⁻⁵ sits on f16's smallest
/// normal number; × 2¹⁰ moves it to ~6·10⁻², far from both ends of f16.
pub const LOSS_SCALE: f32 = 1024.0;
/// `res + 16` = 1/S (written once by the CPU, [`Plan::mixed_uploads`]).
pub const RES_INV_SCALE: u64 = 16;
/// `res + 20` = c / S, what `opt_step` reads at the mixed step (inside the existing 64-byte `res`; nothing moves).
pub const RES_COEF_MIXED: u64 = 20;
/// The tag of the 1×1×1 F32 product c · (1/S).
pub const TAG_UNSCALE: &str = "opt:descalare";

/// The mixed step's tolerances against PyTorch F32, written before the first run (`docs/C-160_P7_PAS_MIXT.md` §0).
pub const MIX_TOL_LOSS_REL: f64 = 1e-3;
/// (loss₁ − loss_last) against PyTorch's, relative.
pub const MIX_TOL_DROP_REL: f64 = 2e-2;
/// Step-1 gradient (divided by S), ‖Δ‖/‖g‖ per tensor.
pub const MIX_TOL_GRAD_REL_NORM: f64 = 5e-3;
/// Norms (divided by S), relative (`adamw_513`).
pub const MIX_TOL_NORM_REL: f64 = 2e-3;
/// Weights after the last step, ‖Δ‖/‖w‖ per tensor.
pub const MIX_TOL_W_REL_NORM: f64 = 1e-3;

/// One launch of the mixed step: either a launch of the F32 list unchanged, or a matmul on `mm_tc` (its arguments
/// sit at 0x700, so its cbuf is 0x800 bytes — it does not fit [`Launch`]).
// ~76 launches per step: the 256-byte cbuf of `Std` inline costs nothing worth a Box.
#[allow(clippy::large_enum_variant)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MixedLaunch {
    Std(Launch),
    Tc {
        args: crate::mm_tc::Args,
        tag: &'static str,
        layer: Option<u16>,
    },
}

/// Bytes of one `mm_tc` launch in the control zone: the cbuf (0x800) then the QMD (0x100).
pub const SLOT_TC: u64 = crate::mm_tc::CBUF_BYTES as u64 + 0x100;

impl MixedLaunch {
    pub fn name(&self) -> &'static str {
        match self {
            MixedLaunch::Std(l) => l.prog.name(),
            MixedLaunch::Tc { .. } => "mm_tc",
        }
    }

    pub fn tag(&self) -> &'static str {
        match self {
            MixedLaunch::Std(l) => l.tag,
            MixedLaunch::Tc { tag, .. } => tag,
        }
    }

    /// Bytes this launch takes in the control zone.
    pub fn slot(&self) -> u64 {
        match self {
            MixedLaunch::Std(_) => SLOT,
            MixedLaunch::Tc { .. } => SLOT_TC,
        }
    }
}

fn mm_args_of(l: &Launch) -> mm::Args {
    let u64_at = |o: usize| u64::from_le_bytes(l.cbuf[o..o + 8].try_into().unwrap_or([0; 8]));
    let u32_at = |o: usize| u32::from_le_bytes(l.cbuf[o..o + 4].try_into().unwrap_or([0; 4]));
    mm::Args {
        a: u64_at(0x28),
        b: u64_at(0x30),
        c: u64_at(0x38),
        bias: u64_at(0x40),
        m: u32_at(0x48),
        n: u32_at(0x4c),
        k: u32_at(0x50),
        flags: u32_at(0x54),
    }
}

/// S must be a power of two in [1, 2²⁴] (so that dividing by it is exact).
fn check_scale(s: f32) -> GpuResult<()> {
    let b = s.to_bits();
    if !(1.0..=16_777_216.0).contains(&s) || b & 0x007f_ffff != 0 {
        return Err(bad(
            "pas-mixt: scara erorii trebuie sa fie putere a lui 2, 1..2^24",
        ));
    }
    Ok(())
}

impl Plan {
    /// What the CPU writes once more for the mixed step (after [`Plan::uploads`]): 1/S at `res + 16`.
    pub fn mixed_uploads(&self, loss_scale: f32) -> GpuResult<Vec<(u64, Vec<u8>)>> {
        check_scale(loss_scale)?;
        Ok(alloc::vec![(
            self.res + RES_INV_SCALE,
            (1.0 / loss_scale).to_le_bytes().to_vec()
        )])
    }

    /// The **mixed** step (C-160 point 7, third part; SIMULARE only): the same list as [`Plan::step`], but every
    /// matmul of the model (forward and backward) on `mm_tc` (inputs rounded to f16, f32 sums on the tensor cores),
    /// d(logits) multiplied by `loss_scale`, the norm compared with `loss_scale`·[`MAX_NORM`] and, before
    /// `opt_step`, one exact F32 product (`mm_train`, 1×1×1) c·(1/S) at `res + 20`, which `opt_step` reads.
    /// Weights, gradient sums, m and v stay F32. `res[0]` is then S times the norm.
    pub fn step_mixed(
        &self,
        recipe: Recipe,
        opt: &AdamW,
        step: u32,
        loss_scale: f32,
    ) -> GpuResult<Vec<MixedLaunch>> {
        check_scale(loss_scale)?;
        if step == 0 {
            return Err(bad("pas-aur: pasii incep de la 1"));
        }
        if self.mv_ram && recipe == Recipe::AdamW513 {
            return Err(bad("pas-aur: pasul mixt cu m, v in RAM nu e facut"));
        }
        let mut list = Vec::with_capacity(32 * self.arch.layers + 16);
        self.forward_backward(&mut list, loss_scale)?;
        self.optimize(&mut list, recipe, opt, step, Some(loss_scale))?;
        list.into_iter()
            .map(|l| {
                if l.prog != Prog::MmTrain || l.tag == TAG_UNSCALE {
                    return Ok(MixedLaunch::Std(l));
                }
                let x = mm_args_of(&l);
                let args = crate::mm_tc::Args {
                    a: x.a,
                    b: x.b,
                    c: x.c,
                    bias: x.bias,
                    m: x.m,
                    n: x.n,
                    k: x.k,
                    flags: x.flags,
                };
                args.cbuf()
                    .ok_or(bad("pas-mixt: argumentele mm_tc refuzate"))?;
                Ok(MixedLaunch::Tc {
                    args,
                    tag: l.tag,
                    layer: l.layer,
                })
            })
            .collect()
    }

    /// Launches per mixed step: one more than [`Plan::launches_per_step`] (the 1×1×1 c/S).
    pub fn launches_per_step_mixed(&self, recipe: Recipe) -> usize {
        self.launches_per_step(recipe) + 1
    }
}

/// The control zone of one mixed step: launches one after the other (`SLOT` bytes for an F32-list launch,
/// [`SLOT_TC`] for an `mm_tc` one), each cbuf then its QMD. `prog_va(p)` = where program `p` is loaded,
/// `tc_va` = where `mm_tc` is. Returns the zone and the QMD addresses for `transfer::encode_sequence_fresh`.
pub fn pack_mixed(
    launches: &[MixedLaunch],
    prog_va: impl Fn(Prog) -> u64,
    tc_va: u64,
    zone_va: u64,
) -> GpuResult<(Vec<u8>, Vec<u64>)> {
    use crate::mm_tc as tc;
    if !zone_va.is_multiple_of(256) || !tc_va.is_multiple_of(256) || launches.is_empty() {
        return Err(bad("pas-mixt: zona de control"));
    }
    let total: u64 = launches.iter().map(MixedLaunch::slot).sum();
    let mut bytes = alloc::vec![0u8; total as usize];
    let mut qmds = Vec::with_capacity(launches.len());
    let mut at = 0usize;
    for l in launches {
        let cb_va = zone_va + at as u64;
        match l {
            MixedLaunch::Std(x) => {
                let pv = prog_va(x.prog);
                if !pv.is_multiple_of(256) {
                    return Err(bad("pas-aur: programul nealiniat"));
                }
                let q = x.qmd(pv, cb_va)?;
                bytes[at..at + 0x100].copy_from_slice(&x.cbuf);
                bytes[at + 0x100..at + 0x100 + q.len()].copy_from_slice(&q);
                qmds.push(cb_va + 0x100);
            }
            MixedLaunch::Tc { args, .. } => {
                let cb = args
                    .cbuf()
                    .ok_or(bad("pas-mixt: argumentele mm_tc refuzate"))?;
                let q = ComputeQmd {
                    program_va: tc_va,
                    registers: tc::REGISTERS,
                    cbuf0_va: cb_va,
                    cbuf0_size: tc::CBUF_BYTES as u32,
                    grid: args.grid(),
                    block: tc::BLOCK,
                    release: None,
                }
                .to_bytes_with_shared(&tc::SHARED)?;
                let c = tc::CBUF_BYTES;
                bytes[at..at + c].copy_from_slice(&cb);
                bytes[at + c..at + c + q.len()].copy_from_slice(&q);
                qmds.push(cb_va + c as u64);
            }
        }
        at += l.slot() as usize;
    }
    Ok((bytes, qmds))
}

/// One figure of the mixed verdict: its name, the deviation, the tolerance written before the run.
#[derive(Clone, Debug, PartialEq)]
pub struct MixedFigure {
    pub name: &'static str,
    pub dev: f64,
    pub tol: f64,
}

impl MixedFigure {
    pub fn passes(&self) -> bool {
        self.dev.is_finite() && self.dev <= self.tol
    }

    pub fn permille(&self) -> u32 {
        permille(self.dev, self.tol, true)
    }
}

/// Judges a mixed run against PyTorch F32 in the tolerances of `docs/C-160_P7_PAS_MIXT.md` §0. `out.norms` are
/// the board's `res[0]` (S times the norm) and `out.grad1` the board's gradient (S times it): both are divided by
/// `loss_scale` here. Returns every figure (pass = all within).
pub fn judge_mixed(
    recipe: Recipe,
    out: &Outcome,
    refs: &Refs,
    loss_scale: f32,
) -> Vec<MixedFigure> {
    let steps = out.losses.len();
    let mut lrel = if steps == 0 || refs.losses.len() < steps {
        f64::NAN
    } else {
        0.0
    };
    for (&o, &w) in out.losses.iter().zip(&refs.losses) {
        let e = if o.is_finite() && w != 0.0 {
            (o as f64 - w).abs() / w.abs()
        } else {
            f64::NAN
        };
        lrel = if e.is_nan() || lrel.is_nan() {
            f64::NAN
        } else {
            lrel.max(e)
        };
    }
    let drop = match (out.losses.first(), out.losses.last()) {
        (Some(&a), Some(&b)) if steps >= 2 && refs.losses.len() >= steps => {
            let theirs = refs.losses[0] - refs.losses[steps - 1];
            let ours = a as f64 - b as f64;
            if theirs != 0.0 {
                (ours - theirs).abs() / theirs.abs()
            } else {
                f64::NAN
            }
        }
        _ => f64::NAN,
    };
    let inv = 1.0 / loss_scale;
    let g: Vec<Vec<f32>> = out
        .grad1
        .iter()
        .map(|t| t.iter().map(|&x| x * inv).collect())
        .collect();
    let grad = deviation(&g, &refs.grad1);
    let wl = deviation(&out.after_last, &refs.after_last);
    let mut f = alloc::vec![
        MixedFigure {
            name: "eroarea",
            dev: lrel,
            tol: MIX_TOL_LOSS_REL
        },
        MixedFigure {
            name: "scaderea",
            dev: drop,
            tol: MIX_TOL_DROP_REL
        },
        MixedFigure {
            name: "gradient",
            dev: if grad.finite { grad.rel_norm } else { f64::NAN },
            tol: MIX_TOL_GRAD_REL_NORM
        },
        MixedFigure {
            name: "greutati",
            dev: if wl.finite { wl.rel_norm } else { f64::NAN },
            tol: MIX_TOL_W_REL_NORM
        },
    ];
    if recipe == Recipe::AdamW513 {
        let mut r = if out.norms.len() == steps && refs.norms.len() >= steps {
            0.0
        } else {
            f64::NAN
        };
        for (&a, &b) in out.norms.iter().zip(&refs.norms) {
            let e = if a.is_finite() && b != 0.0 {
                ((a * inv) as f64 - b).abs() / b
            } else {
                f64::NAN
            };
            r = if e.is_nan() || r.is_nan() {
                f64::NAN
            } else {
                r.max(e)
            };
        }
        f.push(MixedFigure {
            name: "norme",
            dev: r,
            tol: MIX_TOL_NORM_REL,
        });
    }
    f
}
