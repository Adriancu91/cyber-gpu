//! C-160 point 5 — T165, SIMULARE: the small golden step **end to end on the simulated board**.
//!
//! The whole training step of `pas_de_aur_mic` (2 layers, C 64, 4 heads, vocab 256, batch 2 × 32) runs as a list of
//! launches of the 10 training programs (+ `vadd`, the generation's residual addition, which already ran on the
//! RTX 3070 Ti). Since C-160 point 6 (second part) the list is the library's — `nvidia_ga104::train_step::Plan::step`,
//! exactly what the OS sends — and **every step is one submission**: the control zone from `train_step::pack` (a
//! cbuf + a QMD per launch) and `transfer::encode_sequence_fresh` (caches invalidated before every launch, a
//! timestamp after each), through our pushbuffer on the command processor; every intermediate tensor stays in
//! (simulated) GPU memory. The CPU only writes `Plan::uploads` once, reads the per-row losses after every step (their
//! mean only for the report), the gradients and weights after step 1 and the weights after step 10 — compared with
//! **PyTorch** (`reteta.json` and the hub's tensors) in the hub's tolerances, through `train_step::judge` too, and with
//! the CPU oracle (`gpu_compute::train`, T151).
//!
//! The "programs" run on the CPU (`gpu_sim::cmdproc::*_on_cpu` → `gpu_compute::layer`), in the order of the code NAK
//! made, the approximate units modelled correctly rounded. Nothing here runs on the RTX 3070 Ti.

use std::path::{Path, PathBuf};

use gpu_compute::train::{self, AdamW, Arch, Batch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, Event, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS, LN_FWD_NEEDS,
    MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, MM_TRAIN_TILE_GELU_NEEDS, MM_TRAIN_TILE_NEEDS, Needs,
    XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu, attn_back_tile0_on_cpu, attn_fwd_on_cpu,
    attn_fwd_tile_on_cpu, attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, colsum_part_on_cpu,
    emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu,
    ln_back_res_contracted_on_cpu, ln_back_res_on_cpu, ln_fwd_on_cpu, mm_train_gelu_on_cpu,
    mm_train_gelu_unfused_on_cpu, mm_train_on_cpu, mm_train_tile_gelu_on_cpu, mm_train_tile_on_cpu,
    mm_tv_64x64k8_on_cpu, mm_tv_64x128_on_cpu, mm_tv_128x64_on_cpu, mm_tv_128x128_on_cpu,
    opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::mm_train_variants::{self as mv, MmChoice, RaceRow, VARIANTS};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, Launch, Outcome, PROGRAMS_ALL, PROGRAMS_ALL_3, PROGRAMS_ALL_CHOICE,
    PROGRAMS_ALL_FUSED, PROGRAMS_ALL_FUSED2, PROGRAMS_ALL_SPLIT, PROGRAMS_ALL_TILE_GELU, Plan,
    Prog, Refs,
};
use nvidia_ga104::{transfer, xent as xe};
use nvidia_rm570::regs::clc7c0 as c;

/// The small golden model (hub `integrare/kronos/pas_de_aur_mic/reteta.json`).
const MIC: Arch = train::MIC;

thread_local! {
    /// T232 (C-165): when set, every submission of this thread packs its list with `pack_with_sm_step` at this step.
    static SM_STEP: std::cell::Cell<Option<u32>> = const { std::cell::Cell::new(None) };
    /// T232: the distinct [MIN, TARGET, MAX] step fields the command processor saw on this thread.
    static SM_SEEN: std::cell::RefCell<Vec<[u32; 3]>> = const { std::cell::RefCell::new(Vec::new()) };
}
const B: usize = 2;
const T: usize = 32;

/// The start of our GPU virtual addresses (`mmu::VA_BASE`; the launch encoder refuses QMDs outside them).
const VA: u64 = nvidia_ga104::mmu::VA_BASE;
/// The control zone of one step (cbuf + QMD per launch, `train_step::SLOT` each).
const ZONE: u64 = VA + 0x20_0000;
/// The timestamp records of one submission (16 bytes each).
const REC: u64 = VA + 0x30_0000;
const PUSH: u64 = VA + 0x32_0000;
/// First program; one 64 KiB slot each.
const PROGS: u64 = VA + 0x40_0000;
/// The arena's base (`Plan::new`).
const HEAP: u64 = VA + 0x100_0000;

/// What the simulated board checks in a tiled program's QMD: its own shared memory and barriers.
fn needs(s: nvidia_ga104::qmd::SharedMemory) -> Needs {
    Needs {
        shared_bytes: s.bytes,
        barriers: u32::from(s.barriers),
    }
}

fn prog_va(p: Prog) -> u64 {
    PROGS + 0x1_0000 * p.index() as u64
}

// ------------------------------------------------------------------ the simulated board

struct Board {
    cp: CommandProcessor,
    /// Every launch sent, in order (program).
    log: Vec<Prog>,
    /// Board time of each submission (ns, from the records).
    step_ns: Vec<u64>,
    first: u32,
}

impl Board {
    fn new() -> Self {
        Self::with_models(ln_back_res_on_cpu, mm_train_gelu_on_cpu)
    }

    /// T200: the board with another CPU stand-in for `ln_back_res` (the contracted one, for the planted mistake);
    /// T201: the same for `mm_train_gelu` (the unfused epilogue).
    fn with_models(
        ln_back_res: gpu_sim::cmdproc::ProgramFn,
        mm_gelu: gpu_sim::cmdproc::ProgramFn,
    ) -> Self {
        ts::check_programs().expect("SHA-256 programe");
        let mut cp = CommandProcessor::new();
        // All 18 (the 13 of every plan + `attn_p`, used only by `Plan::new_p_only`, + the four tiled ones of
        // `Plan::new_tiles` / `new_p_only_tiles`).
        // T195: + the three `_3` programs (`PROGRAMS_ALL_3`), on the same CPU models as the first tiled ones.
        // T200: + `ln_back_res` (`PROGRAMS_ALL_FUSED`). T201: + `mm_train_gelu` (`PROGRAMS_ALL_FUSED2`).
        // T215: + `mm_train_tile` (`PROGRAMS_ALL_TILE`). T216: + `colsum_part` (`PROGRAMS_ALL_SPLIT`).
        // T219: + the four race variants (`PROGRAMS_ALL_CHOICE`), each on its own CPU model and its own shared memory.
        // T220: + `mm_train_tile_gelu` (`PROGRAMS_ALL_TILE_GELU`).
        for p in PROGRAMS_ALL_TILE_GELU {
            let code = p.code();
            match p {
                Prog::MmTrain => {
                    cp.register_program_with_needs(code, mm_train_on_cpu, MM_TRAIN_NEEDS)
                }
                Prog::Xent => cp.register_program_with_needs(code, xent_on_cpu, XENT_NEEDS),
                Prog::LnFwd => cp.register_program_with_needs(code, ln_fwd_on_cpu, LN_FWD_NEEDS),
                Prog::LnBack => cp.register_program_with_needs(code, ln_back_on_cpu, LN_BACK_NEEDS),
                Prog::Colsum => cp.register_program(code, colsum_on_cpu),
                Prog::GeluTr => cp.register_program(code, gelu_tr_on_cpu),
                Prog::AttnFwd => cp.register_program(code, attn_fwd_on_cpu),
                Prog::AttnBack => cp.register_program(code, attn_back_on_cpu),
                Prog::EmbFwd => cp.register_program(code, emb_fwd_on_cpu),
                Prog::EmbBack => cp.register_program(code, emb_back_on_cpu),
                Prog::Gnorm => cp.register_program_with_needs(code, gnorm_on_cpu, GNORM_NEEDS),
                Prog::OptStep => cp.register_program(code, opt_step_on_cpu),
                Prog::Vadd => cp.register_program(code, vadd_on_cpu),
                Prog::AttnP => cp.register_program(code, attn_p_on_cpu),
                Prog::AttnFwdTile2 => cp.register_program_with_needs(
                    code,
                    attn_fwd_tile_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::AttnBackTile0_2 => cp.register_program_with_needs(
                    code,
                    attn_back_tile0_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::AttnBackTile2 => cp.register_program_with_needs(
                    code,
                    attn_back_tile_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::AttnPTile => cp.register_program_with_needs(
                    code,
                    attn_p_tile_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::AttnFwdTile3 => cp.register_program_with_needs(
                    code,
                    attn_fwd_tile_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::AttnBackTile0_3 => cp.register_program_with_needs(
                    code,
                    attn_back_tile0_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::AttnBackTile3 => cp.register_program_with_needs(
                    code,
                    attn_back_tile_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::LnBackRes => {
                    cp.register_program_with_needs(code, ln_back_res, LN_BACK_RES_NEEDS)
                }
                Prog::MmTrainGelu => {
                    cp.register_program_with_needs(code, mm_gelu, MM_TRAIN_GELU_NEEDS)
                }
                Prog::MmTrainTile => {
                    cp.register_program_with_needs(code, mm_train_tile_on_cpu, MM_TRAIN_TILE_NEEDS)
                }
                Prog::ColsumPart => cp.register_program(code, colsum_part_on_cpu),
                Prog::MmTv128x64 => cp.register_program_with_needs(
                    code,
                    mm_tv_128x64_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::MmTv64x128 => cp.register_program_with_needs(
                    code,
                    mm_tv_64x128_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::MmTv128x128 => cp.register_program_with_needs(
                    code,
                    mm_tv_128x128_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::MmTv64x64K8 => cp.register_program_with_needs(
                    code,
                    mm_tv_64x64k8_on_cpu,
                    needs(p.shared().unwrap()),
                ),
                Prog::MmTrainTileGelu => cp.register_program_with_needs(
                    code,
                    mm_train_tile_gelu_on_cpu,
                    MM_TRAIN_TILE_GELU_NEEDS,
                ),
            };
            cp.mem.write(prog_va(p), code);
        }
        Board {
            cp,
            log: Vec::new(),
            step_ns: Vec::new(),
            first: 0x5000,
        }
    }

    fn read(&self, va: u64, n: usize) -> Vec<f32> {
        self.cp.mem.read_f32s(va, n).unwrap()
    }

    /// One step = one submission: the control zone, then `encode_sequence_fresh` over all the QMDs.
    fn submit(&mut self, launches: &[Launch]) {
        self.try_submit(launches, |_| {})
            .unwrap_or_else(|err| panic!("pasul: {err}"));
    }

    /// [`Board::submit`] with the control zone patched first (T187: a QMD without its shared memory); the command
    /// processor's refusal comes back as an error.
    fn try_submit(&mut self, launches: &[Launch], patch: impl Fn(&mut [u8])) -> Result<(), String> {
        let (mut zone, qmds) = match SM_STEP.with(|c| c.get()) {
            None => ts::pack(launches, prog_va, ZONE).unwrap(),
            Some(step) => ts::pack_with_sm_step(launches, prog_va, ZONE, step).unwrap(),
        };
        patch(&mut zone);
        assert!(zone.len() as u64 <= REC - ZONE, "zona de control prea mare");
        self.cp.mem.write(ZONE, &zone);
        let n = launches.len();
        self.cp.mem.write(REC, &vec![0u8; 16 * (n + 1)]);
        let mut w = vec![0u32; 2 + transfer::sequence_words_fresh(n)];
        let mut pb = PushBuf::new(&mut w);
        pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
            .unwrap();
        transfer::encode_sequence_fresh(&mut pb, &qmds, REC, self.first).unwrap();
        let nw = pb.len();
        let bytes: Vec<u8> = w[..nw].iter().flat_map(|x| x.to_le_bytes()).collect();
        self.cp.mem.write(PUSH, &bytes);
        let e = push::gpfifo_entry(PUSH, nw as u32, false).unwrap();
        self.cp
            .run_gpfifo(&[(e.entry0, e.entry1)])
            .map_err(|err| format!("{err:?}"))?;
        let rec: Vec<u8> = (0..4 * (n + 1))
            .flat_map(|k| {
                self.cp
                    .mem
                    .read_u32(REC + 4 * k as u64)
                    .unwrap()
                    .to_le_bytes()
            })
            .collect();
        let ns = transfer::analyze_sequence(&rec, n, self.first).unwrap();
        self.step_ns.push(ns.iter().sum());
        self.first = self.first.wrapping_add(n as u32 + 1);
        self.log.extend(launches.iter().map(|l| l.prog));
        SM_SEEN.with(|v| {
            let mut v = v.borrow_mut();
            for sm in &self.cp.sm_configs {
                if !v.contains(sm) {
                    v.push(*sm);
                }
            }
        });
        Ok(())
    }
}

// ------------------------------------------------------------------ mistakes planted in the driver's list

/// A mistake planted in the driver's list, to show the comparison catches it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fault {
    None,
    /// The residual addition through LN2 forgotten in the backward pass of every layer.
    NoBackResidual,
    /// The loss gradient scaled by 2/N instead of 1/N.
    XentScale,
    /// `adamw_513` without the clipping (factor 1 at `res + 8` instead of the board's c at `res + 4`).
    NoClip,
    /// The head's dW written again after the token scatter (the scatter lost).
    HeadOverwrites,
    /// C-160 point 7 (T177): the shared-P plan without its `attn_fwd` re-launches (every layer but the last then
    /// uses the last layer's P).
    NoPRebuild,
    /// C-160 point 7 (`docs/C-160_P7_ATTN_P_IN_PAS.md`): layer 0's P re-launch reads layer 1's `qkv`.
    PWrongQkv,
    /// T187 (`docs/C-160_P7_PLACI_IN_PAS.md` prediction 3 — not a mistake): the two backward attention launches of a
    /// tiled plan replaced by `Plan::new`'s `attn_back` launches (same arena), the tiled forward kept.
    OldBackward,
    /// T187: mode 0 (the rows of dS) launched on `attn_back_tile_2` (which exits writing nothing for mode 0).
    TileRowsOnCols,
    /// T195: mode 0 launched on `attn_back_tile_3`.
    Tile3RowsOnCols,
    /// T195 (prediction 4 — not a mistake): every `_3` launch swapped for its `_2` partner, one kind at a time.
    Swap3To2Fwd,
    Swap3To2Back,
    /// T195 (prediction 4): the backward `_2` launches of a `_2` plan swapped for their `_3` partners.
    Swap2To3Back,
    /// T200: every `ln_back_res` launch adds the wrong buffer (`dwide`) instead of the residual path's gradient.
    FusedWrongRes,
    /// T200: `ln_back_res` run by the contracted model (one FFMA for rstd·q + dres) — what NAK would make without
    /// `add_exact`.
    FusedContracted,
    /// T201: `mm_train_gelu` run by the unfused model (Φ + u·φ in two roundings) — what NAK would make if it compiled
    /// the epilogue unlike `gelu_tr`.
    GeluUnfused,
    /// T201: every `mm_train_gelu` launch reads the next layer's `u`.
    GeluWrongU,
    /// T201: every `mm_train_gelu` launch run as plain `mm_train` (GELU' forgotten; u's address in the bias slot,
    /// no BIAS flag, so it is not read).
    GeluForgotten,
    /// T203: the `recalcul:*` launches dropped (the earlier layers go back on the last layer's buffers).
    NoRecompute,
    /// T203: layer 0's `recalcul:ln1` reads layer 1's input.
    RecomputeWrongInput,
    /// T215: every `mm_train_tile` launch keeps `mm_train`'s grid (16 × 16 pieces: far too many blocks — not a
    /// mistake, the extra blocks write nothing).
    TileOldGrid,
    /// T215: every `mm_train_tile` launch with one block row too few (the last 64 rows of C never written).
    TileShortGrid,
}

/// Which arena: P kept per layer ([`Plan::new`]), one shared P rebuilt with `attn_fwd` ([`Plan::new_p_recalc`]) or
/// with `attn_p` ([`Plan::new_p_only`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    Kept,
    Recalc,
    POnly,
    /// [`Plan::new_tiles`] (P kept, attention on tiles).
    Tiles,
    /// [`Plan::new_p_only_tiles`] (one P rebuilt with `attn_p_tile`).
    POnlyTiles,
    /// T195: [`Plan::new_tiles3`] (P kept, attention on the `_3` tiles).
    Tiles3,
    /// T195: [`Plan::new_p_only_tiles3`].
    POnlyTiles3,
    /// T200: [`Plan::new`] + [`Plan::with_fused_back`].
    KeptFused,
    /// T200: [`Plan::new_tiles3`] + [`Plan::with_fused_back`].
    Tiles3Fused,
    /// T201: [`Plan::new`] + [`Plan::with_fused_gelu`].
    KeptGelu,
    /// T201: [`Plan::new_tiles3`] + [`Plan::with_fused_gelu`].
    Tiles3Gelu,
    /// T201: [`Plan::new_tiles3`] + both fusions.
    Tiles3Both,
    /// T203: [`Plan::new`] + [`Plan::with_recompute`].
    KeptRecompute,
    /// T203: [`Plan::new_tiles3`] + both fusions + [`Plan::with_recompute`].
    Tiles3BothRecompute,
    /// T203: [`Plan::new_p_only_tiles3`] + [`Plan::with_recompute`].
    POnlyTiles3Recompute,
    /// T215: [`Plan::new_tiles3`] + [`Plan::with_mm_tile`].
    Tiles3Tile,
    /// T215: [`Plan::new_tiles3`] + both fusions + [`Plan::with_mm_tile`].
    Tiles3BothTile,
    /// T215: [`Plan::new`] + [`Plan::with_mm_tile`].
    KeptTile,
    /// T216: [`Plan::new`] + [`Plan::with_colsum_split`].
    KeptSplit,
    /// T216: [`Plan::new_tiles3`] + [`Plan::with_mm_tile`] + [`Plan::with_colsum_split`] (what os-407 + `--mm-placi` +
    /// the split would send).
    Tiles3TileSplit,
    /// T216: [`Plan::new_tiles3`] + both fusions + [`Plan::with_mm_tile`] + [`Plan::with_colsum_split`].
    Tiles3BothTileSplit,
    /// T219: [`Plan::new_tiles3`] + [`Plan::with_mm_choice`] (every shape of the step on a race variant, in turn).
    Tiles3Choice,
    /// T219: [`Plan::new_tiles3`] + both fusions + [`Plan::with_mm_choice`].
    Tiles3BothChoice,
    /// T219: [`Plan::new`] + [`Plan::with_mm_choice`] with the empty choice (= [`Plan::with_mm_tile`]).
    KeptChoiceEmpty,
    /// T220: [`Plan::new_tiles3`] + both fusions + [`Plan::with_mm_tile_gelu`].
    Tiles3BothTileGelu,
    /// T220: [`Plan::new`] + [`Plan::with_fused_gelu`] + [`Plan::with_mm_tile_gelu`].
    KeptGeluTileGelu,
    /// T220: [`Plan::new_tiles3`] + [`Plan::with_mm_tile_gelu`] without [`Plan::with_fused_gelu`] (= `Tiles3Tile`).
    Tiles3TileGeluAlone,
    /// T220: [`Plan::new_tiles3`] + both fusions + [`Plan::with_mm_choice`] + [`Plan::with_mm_tile_gelu`].
    Tiles3BothChoiceTileGelu,
}

impl Kind {
    fn split(self) -> bool {
        matches!(
            self,
            Kind::KeptSplit | Kind::Tiles3TileSplit | Kind::Tiles3BothTileSplit
        )
    }
}

fn plant(plan: &Plan, list: &mut Vec<Launch>, f: Fault) {
    match f {
        Fault::None => {}
        Fault::TileOldGrid | Fault::TileShortGrid => {
            let tiled: Vec<&mut Launch> = list
                .iter_mut()
                .filter(|l| l.prog == Prog::MmTrainTile)
                .collect();
            assert!(!tiled.is_empty(), "niciun mm_train_tile");
            for l in tiled {
                let a = mm_args(l);
                if f == Fault::TileOldGrid {
                    l.grid = a.grid();
                } else if l.grid[1] > 1 {
                    l.grid[1] -= 1;
                } else {
                    l.grid[0] = l.grid[0].saturating_sub(1).max(1);
                }
            }
        }
        Fault::NoBackResidual => list.retain(|l| l.tag != "inapoi:rezidual-ln2"),
        Fault::XentScale => {
            let n = (plan.batch * plan.len) as u32;
            let a = xe::Args {
                logits: plan.logits,
                targets: plan.tgt,
                dlogits: plan.dlogits,
                loss: plan.loss_rows,
                rows: n,
                v: plan.arch.vocab as u32,
                scale: 2.0 / n as f32,
            };
            let l = list.iter_mut().find(|l| l.tag == "eroarea").unwrap();
            let cb = a.cbuf().unwrap();
            l.cbuf[..cb.len()].copy_from_slice(&cb);
        }
        Fault::NoClip => {
            let l = list.iter_mut().find(|l| l.tag == "opt:pas").unwrap();
            let (from, to) = ((plan.res + 4).to_le_bytes(), (plan.res + 8).to_le_bytes());
            let at = l.cbuf.windows(8).position(|w| w == from).unwrap();
            l.cbuf[at..at + 8].copy_from_slice(&to);
        }
        Fault::HeadOverwrites => {
            let head = list
                .iter()
                .find(|l| l.tag == "inapoi:cap-dW")
                .unwrap()
                .clone();
            let at = list.iter().position(|l| l.tag.starts_with("opt:")).unwrap();
            list.insert(at, head);
        }
        Fault::NoPRebuild => list.retain(|l| l.tag != "inapoi:atentia-P"),
        Fault::OldBackward => {}
        Fault::Tile3RowsOnCols => {
            for l in list.iter_mut() {
                if l.prog == Prog::AttnBackTile0_3 {
                    l.prog = Prog::AttnBackTile3;
                }
            }
        }
        Fault::Swap3To2Fwd => {
            for l in list.iter_mut().filter(|l| l.prog == Prog::AttnFwdTile3) {
                l.prog = Prog::AttnFwdTile2;
            }
        }
        Fault::Swap3To2Back => {
            for l in list.iter_mut() {
                l.prog = match l.prog {
                    Prog::AttnBackTile0_3 => Prog::AttnBackTile0_2,
                    Prog::AttnBackTile3 => Prog::AttnBackTile2,
                    p => p,
                };
            }
        }
        Fault::Swap2To3Back => {
            for l in list.iter_mut() {
                l.prog = match l.prog {
                    Prog::AttnBackTile0_2 => Prog::AttnBackTile0_3,
                    Prog::AttnBackTile2 => Prog::AttnBackTile3,
                    p => p,
                };
            }
        }
        Fault::TileRowsOnCols => {
            for l in list.iter_mut() {
                if l.prog == Prog::AttnBackTile0_2 {
                    l.prog = Prog::AttnBackTile2;
                }
            }
        }
        Fault::FusedContracted | Fault::GeluUnfused => {}
        Fault::GeluWrongU => {
            let ly = plan.arch.layers;
            let mut n = 0;
            for l in list.iter_mut().filter(|l| l.prog == Prog::MmTrainGelu) {
                let k = l.layer.unwrap() as usize;
                l.cbuf[0x40..0x48].copy_from_slice(&plan.layers[(k + 1) % ly].u.to_le_bytes());
                n += 1;
            }
            assert_eq!(n, ly);
        }
        Fault::GeluForgotten => {
            for l in list.iter_mut().filter(|l| l.prog == Prog::MmTrainGelu) {
                l.prog = Prog::MmTrain;
            }
        }
        Fault::NoRecompute => list.retain(|l| !l.tag.starts_with("recalcul:")),
        Fault::RecomputeWrongInput => {
            let l = list
                .iter_mut()
                .find(|l| l.tag == "recalcul:ln1" && l.layer == Some(0))
                .unwrap();
            let (from, to) = (plan.xs[0].to_le_bytes(), plan.xs[2].to_le_bytes());
            let at = l.cbuf.windows(8).position(|w| w == from).unwrap();
            l.cbuf[at..at + 8].copy_from_slice(&to);
        }
        Fault::FusedWrongRes => {
            let mut n = 0;
            for l in list.iter_mut().filter(|l| l.prog == Prog::LnBackRes) {
                l.cbuf[0x58..0x60].copy_from_slice(&plan.dwide.to_le_bytes());
                n += 1;
            }
            assert_eq!(n, 2 * plan.arch.layers);
        }
        Fault::PWrongQkv => {
            let l = list
                .iter_mut()
                .find(|l| l.tag == "inapoi:atentia-P" && l.layer == Some(0))
                .unwrap();
            let (from, to) = (
                plan.layers[0].qkv.to_le_bytes(),
                plan.layers[1].qkv.to_le_bytes(),
            );
            let at = l.cbuf.windows(8).position(|w| w == from).unwrap();
            l.cbuf[at..at + 8].copy_from_slice(&to);
        }
    }
}

// ------------------------------------------------------------------ one run

/// Everything one run of `steps` steps on the simulated board produced.
struct Run {
    out: Outcome,
    launches_per_step: Vec<(Prog, usize)>,
    total_per_step: usize,
    cp_launch_events: usize,
    board_bytes: u64,
}

/// `recalc`: [`Plan::new_p_recalc`] (one P shared by all layers, rebuilt in the backward pass) instead of
/// [`Plan::new`].
#[allow(clippy::too_many_arguments)]
fn run_plan(
    kind: Kind,
    ar: Arch,
    (bsz, tl): (usize, usize),
    params: &[Vec<f32>],
    (inputs, targets): (&[u32], &[u32]),
    recipe: Recipe,
    steps: u32,
    fault: Fault,
) -> Run {
    let plan = match kind {
        Kind::Kept => Plan::new(ar, bsz, tl, inputs, HEAP).unwrap(),
        Kind::Recalc => Plan::new_p_recalc(ar, bsz, tl, inputs, HEAP).unwrap(),
        Kind::POnly => Plan::new_p_only(ar, bsz, tl, inputs, HEAP).unwrap(),
        Kind::Tiles => Plan::new_tiles(ar, bsz, tl, inputs, HEAP).unwrap(),
        Kind::POnlyTiles => Plan::new_p_only_tiles(ar, bsz, tl, inputs, HEAP).unwrap(),
        Kind::Tiles3 => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP).unwrap(),
        Kind::POnlyTiles3 => Plan::new_p_only_tiles3(ar, bsz, tl, inputs, HEAP).unwrap(),
        Kind::KeptFused => Plan::new(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_fused_back(),
        Kind::Tiles3Fused => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_fused_back(),
        Kind::KeptGelu => Plan::new(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_fused_gelu(),
        Kind::Tiles3Gelu => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_fused_gelu(),
        Kind::Tiles3Both => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_fused_back()
            .with_fused_gelu(),
        Kind::KeptRecompute => Plan::new(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_recompute()
            .unwrap(),
        Kind::Tiles3BothRecompute => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_fused_back()
            .with_fused_gelu()
            .with_recompute()
            .unwrap(),
        Kind::POnlyTiles3Recompute => Plan::new_p_only_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_recompute()
            .unwrap(),
        Kind::Tiles3Tile => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_mm_tile(),
        Kind::Tiles3BothTile => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_fused_back()
            .with_fused_gelu()
            .with_mm_tile(),
        Kind::KeptTile => Plan::new(ar, bsz, tl, inputs, HEAP).unwrap().with_mm_tile(),
        Kind::KeptSplit => Plan::new(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_colsum_split(),
        Kind::Tiles3TileSplit => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_mm_tile()
            .with_colsum_split(),
        Kind::Tiles3BothTileSplit => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_fused_back()
            .with_fused_gelu()
            .with_mm_tile()
            .with_colsum_split(),
        Kind::Tiles3Choice => {
            let p = Plan::new_tiles3(ar, bsz, tl, inputs, HEAP).unwrap();
            let c = turn_choice(&p, params);
            p.with_mm_choice(c)
        }
        Kind::Tiles3BothChoice => {
            let p = Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
                .unwrap()
                .with_fused_back()
                .with_fused_gelu();
            let c = turn_choice(&p, params);
            p.with_mm_choice(c)
        }
        Kind::KeptChoiceEmpty => Plan::new(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_mm_choice(MmChoice::new()),
        Kind::Tiles3BothTileGelu => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_fused_back()
            .with_fused_gelu()
            .with_mm_tile_gelu(),
        Kind::KeptGeluTileGelu => Plan::new(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_fused_gelu()
            .with_mm_tile_gelu(),
        Kind::Tiles3TileGeluAlone => Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
            .unwrap()
            .with_mm_tile_gelu(),
        Kind::Tiles3BothChoiceTileGelu => {
            let p = Plan::new_tiles3(ar, bsz, tl, inputs, HEAP)
                .unwrap()
                .with_fused_back()
                .with_fused_gelu();
            let c = turn_choice(&p, params);
            p.with_mm_choice(c).with_mm_tile_gelu()
        }
    };
    // T187: the arena of `Plan::new` (the same addresses as `new_tiles`), for `Fault::OldBackward`.
    let old = Plan::new(ar, bsz, tl, inputs, HEAP).unwrap();
    let mut b = match fault {
        Fault::FusedContracted => {
            Board::with_models(ln_back_res_contracted_on_cpu, mm_train_gelu_on_cpu)
        }
        Fault::GeluUnfused => Board::with_models(ln_back_res_on_cpu, mm_train_gelu_unfused_on_cpu),
        _ => Board::new(),
    };
    for (va, bytes) in plan.uploads(params, inputs, targets).unwrap() {
        b.cp.mem.write(va, &bytes);
    }
    if fault == Fault::NoClip {
        b.cp.mem.write_f32s(plan.res + 8, &[1.0]);
    }
    let opt = AdamW::recipe_513(params);
    let mut out = Outcome::default();
    let mut per_step = Vec::new();
    for step in 1..=steps {
        let mut list = plan.step(recipe, &opt, step).unwrap();
        assert_eq!(list.len(), plan.launches_per_step(recipe));
        assert_eq!(
            list.iter().filter(|l| l.tag == "inapoi:atentia-P").count(),
            plan.p_relaunches()
        );
        assert_eq!(
            list.iter()
                .filter(|l| l.tag.starts_with("recalcul:"))
                .count(),
            plan.recompute_launches()
        );
        plant(&plan, &mut list, fault);
        if fault == Fault::OldBackward {
            assert_eq!(kind, Kind::Tiles);
            let ol = old.step(recipe, &opt, step).unwrap();
            assert_eq!(ol.len(), list.len());
            for (l, o) in list.iter_mut().zip(ol) {
                assert_eq!((l.tag, l.layer, l.cbuf), (o.tag, o.layer, o.cbuf));
                if l.tag.starts_with("inapoi:atentia-d") {
                    *l = o;
                }
            }
        }
        b.submit(&list);
        let rows = b.read(plan.loss_rows, bsz * tl);
        out.losses.push(Outcome::mean_loss(&rows));
        if recipe == Recipe::AdamW513 {
            let r = b.read(plan.res, 2);
            // The driver stops the run on a non-finite norm (C-160 point 4 note).
            assert!(r[0].is_finite(), "norma nu e finita: pasul se opreste");
            out.norms.push(r[0]);
        }
        if step == 1 {
            out.grad1 = plan.unflat(&b.read(plan.g, plan.n_params)).unwrap();
            out.after1 = plan.unflat(&b.read(plan.w, plan.n_params)).unwrap();
            let mut counts: Vec<(Prog, usize)> = Vec::new();
            for l in &list {
                match counts.iter_mut().find(|(q, _)| *q == l.prog) {
                    Some(e) => e.1 += 1,
                    None => counts.push((l.prog, 1)),
                }
            }
            per_step = counts;
        }
    }
    out.after_last = plan.unflat(&b.read(plan.w, plan.n_params)).unwrap();
    out.step_ns = b.step_ns.clone();
    let total_per_step = per_step.iter().map(|e| e.1).sum();
    let cp_launch_events =
        b.cp.events
            .iter()
            .filter(|e| matches!(e, Event::Launch { .. }))
            .count();
    assert_eq!(cp_launch_events, b.log.len());
    Run {
        out,
        launches_per_step: per_step,
        total_per_step,
        cp_launch_events,
        board_bytes: plan.bytes(),
    }
}

fn run_mic(recipe: Recipe, steps: u32, fault: Fault) -> Run {
    run_mic_plan(Kind::Kept, recipe, steps, fault)
}

fn run_mic_plan(kind: Kind, recipe: Recipe, steps: u32, fault: Fault) -> Run {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    run_plan(
        kind,
        MIC,
        (B, T),
        &tensors("initiale"),
        (&inputs, &targets),
        recipe,
        steps,
        fault,
    )
}

// ------------------------------------------------------------------ the golden step's data

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../gpu-compute/tests/date/pas_de_aur_mic")
}

fn read_f32(p: &Path, n: usize) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(b.len(), 4 * n, "{}", p.display());
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_i32(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::try_from(i32::from_le_bytes(*c)).unwrap())
        .collect()
}

fn tensors(sub: &str) -> Vec<Vec<f32>> {
    MIC.tensors()
        .iter()
        .map(|t| {
            read_f32(
                &dir().join(sub).join(format!("{}.f32", t.name)),
                t.elements(),
            )
        })
        .collect()
}

/// A list of numbers of one recipe in `reteta.json` (`pierderi`, `norme_gradient`), parsed by hand.
fn json_list(recipe: &str, key: &str) -> Vec<f64> {
    let s = std::fs::read_to_string(dir().join("reteta.json")).unwrap();
    let r = &s[s.find(&format!("\"{recipe}\"")).unwrap()..];
    let a = &r[r.find(&format!("\"{key}\"")).unwrap()..];
    let a = &a[a.find('[').unwrap() + 1..a.find(']').unwrap()];
    a.split(',').map(|v| v.trim().parse().unwrap()).collect()
}

struct Dev {
    abs: f32,
    rel_norm: f64,
    where_: String,
}

fn worst(ours: &[Vec<f32>], theirs: &[Vec<f32>]) -> Dev {
    let mut d = Dev {
        abs: 0.0,
        rel_norm: 0.0,
        where_: String::new(),
    };
    for ((o, t), spec) in ours.iter().zip(theirs).zip(MIC.tensors()) {
        let (mut diff2, mut norm2) = (0.0f64, 0.0f64);
        for (&a, &b) in o.iter().zip(t) {
            let e = (a - b).abs();
            assert!(e.is_finite(), "{}: valoare nefinita", spec.name);
            if e > d.abs {
                d.abs = e;
                d.where_ = spec.name.clone();
            }
            diff2 += (e as f64).powi(2);
            norm2 += (b as f64).powi(2);
        }
        if norm2 > 0.0 {
            d.rel_norm = d.rel_norm.max((diff2 / norm2).sqrt());
        }
    }
    d
}

fn batch_of<'a>(inputs: &'a [u32], targets: &'a [u32]) -> Batch<'a> {
    Batch {
        inputs,
        targets,
        batch: B,
        len: T,
    }
}

fn refs(name: &str) -> Refs {
    Refs {
        losses: json_list(name, "pierderi"),
        norms: if name == "sgd" {
            Vec::new()
        } else {
            json_list(name, "norme_gradient")
        },
        grad1: tensors(&format!("{name}/grad_pas1")),
        after1: tensors(&format!("{name}/dupa_pas1")),
        after_last: tensors(&format!("{name}/dupa_pas10")),
    }
}

/// Judges one run against PyTorch (the hub's tolerances, the same as T151 for the CPU oracle) and the CPU oracle.
fn check(recipe: Recipe, name: &str, w_tol: f32) {
    let r = run_mic(recipe, 10, Fault::None);
    let o = &r.out;
    // Every step is the same list of launches, all through the command processor.
    assert_eq!(r.cp_launch_events, 10 * r.total_per_step);
    let want = json_list(name, "pierderi");
    let mut loss_rel = 0.0f64;
    for (k, (&o, &w)) in o.losses.iter().zip(&want).enumerate() {
        let e = (o as f64 - w).abs() / w.abs();
        loss_rel = loss_rel.max(e);
        assert!(
            e <= train::TOL_LOSS_REL,
            "{name}: eroarea pasului {}: {o} fata de {w}",
            k + 1
        );
    }
    let g = worst(&o.grad1, &tensors(&format!("{name}/grad_pas1")));
    let w1 = worst(&o.after1, &tensors(&format!("{name}/dupa_pas1")));
    let w10 = worst(&o.after_last, &tensors(&format!("{name}/dupa_pas10")));

    // The library's verdict (what the OS prints) must agree.
    let v = ts::judge(recipe, r.total_per_step, o, &refs(name));
    for l in v.lines("SIMULARE") {
        println!("kt c165 {l}");
    }
    assert!(v.passes(), "{name}: verdictul bibliotecii");
    assert_eq!(
        v.launches_per_step,
        if recipe == Recipe::Sgd { 73 } else { 75 }
    );

    // The CPU oracle (T151) on the same data.
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let cpu = train::golden_run(
        &MIC,
        &tensors("initiale"),
        &batch_of(&inputs, &targets),
        recipe,
        10,
    )
    .unwrap();
    let og = worst(&o.grad1, &cpu.grad_step1);
    let o1 = worst(&o.after1, &cpu.after_step1);
    let o10 = worst(&o.after_last, &cpu.after_last);
    let ol = o
        .losses
        .iter()
        .zip(&cpu.losses)
        .map(|(&a, &b)| ((a - b) as f64).abs() / b as f64)
        .fold(0.0, f64::max);

    let list: Vec<String> = r
        .launches_per_step
        .iter()
        .map(|(p, k)| format!("{} {k}", p.name()))
        .collect();
    println!(
        "kt c165 {name} pe placa simulata: {} lansari pe pas ({}), o trimitere pe pas, memoria plăcii {} KiB; eroare_rel={loss_rel:.2e} grad_abs={:.2e} ({}) grad_rel_norma={:.2e} w1_abs={:.2e} ({}) w10_abs={:.2e} ({}) eroarea_1={} eroarea_10={}",
        r.total_per_step,
        list.join(", "),
        r.board_bytes / 1024,
        g.abs,
        g.where_,
        g.rel_norm,
        w1.abs,
        w1.where_,
        w10.abs,
        w10.where_,
        o.losses[0],
        o.losses[9]
    );
    println!(
        "kt c165 {name} fata de oracolul pe procesor: eroare_rel={ol:.2e} grad_abs={:.2e} ({}) grad_rel_norma={:.2e} w1_abs={:.2e} w10_abs={:.2e}",
        og.abs, og.where_, og.rel_norm, o1.abs, o10.abs
    );
    if !o.norms.is_empty() {
        let wn = json_list(name, "norme_gradient");
        let nrel = o
            .norms
            .iter()
            .zip(&wn)
            .map(|(&a, &b)| (a as f64 - b).abs() / b)
            .fold(0.0, f64::max);
        println!("kt c165 {name} normele celor 10 pasi fata de PyTorch: rel ≤ {nrel:.2e}");
        assert!(nrel <= 1e-5, "{name}: normele {nrel}");
    }
    assert!(
        g.abs <= train::TOL_GRAD_ABS,
        "{name}: gradient {} in {}",
        g.abs,
        g.where_
    );
    assert!(
        g.rel_norm <= train::TOL_GRAD_REL_NORM,
        "{name}: gradient rel {}",
        g.rel_norm
    );
    assert!(
        w1.abs <= w_tol,
        "{name}: greutati dupa pasul 1: {} in {}",
        w1.abs,
        w1.where_
    );
    assert!(
        w10.abs <= w_tol,
        "{name}: greutati dupa pasul 10: {} in {}",
        w10.abs,
        w10.where_
    );
}

#[test]
fn t165_small_golden_step_sgd_end_to_end_on_the_simulated_board() {
    check(Recipe::Sgd, "sgd", train::TOL_SGD_W_ABS);
}

#[test]
fn t165_small_golden_step_adamw_513_end_to_end_on_the_simulated_board() {
    check(Recipe::AdamW513, "adamw_513", train::TOL_ADAMW_W_ABS);
}

#[test]
fn t165_planted_mistakes_in_the_driver_are_caught() {
    let rf = refs("adamw_513");
    for f in [
        Fault::NoBackResidual,
        Fault::XentScale,
        Fault::NoClip,
        Fault::HeadOverwrites,
    ] {
        // AdamW's first step barely sees a constant scale of the gradient (m/√v ≈ ±1): the missing clipping shows
        // only once the factor changes from step to step, so that mistake runs 10 steps.
        let steps = if f == Fault::NoClip { 10 } else { 1 };
        let r = run_mic(Recipe::AdamW513, steps, f);
        let g = worst(&r.out.grad1, &rf.grad1);
        let w1 = worst(&r.out.after1, &rf.after1);
        let w10 = worst(&r.out.after_last, &rf.after_last);
        let l = (r.out.losses[0] as f64 - rf.losses[0]).abs() / rf.losses[0];
        let caught = g.abs > train::TOL_GRAD_ABS
            || g.rel_norm > train::TOL_GRAD_REL_NORM
            || w1.abs > train::TOL_ADAMW_W_ABS
            || (steps == 10 && w10.abs > train::TOL_ADAMW_W_ABS)
            || l > train::TOL_LOSS_REL;
        // The library's verdict sees it too (after 1 step the weights after "the last step" = after step 1, which
        // the hub gives only after step 10 → the verdict is a fail there anyway; what matters is it never passes).
        let v = ts::judge(Recipe::AdamW513, r.total_per_step, &r.out, &rf);
        println!(
            "kt c165 greseala {f:?} ({steps} pasi): grad_abs {:.2e} ({}), w1_abs {:.2e} ({}){} -> {}; verdictul bibliotecii: {}",
            g.abs,
            g.where_,
            w1.abs,
            w1.where_,
            if steps == 10 {
                format!(", w10_abs {:.2e} ({})", w10.abs, w10.where_)
            } else {
                String::new()
            },
            if caught { "PRINSA" } else { "SCAPATA" },
            if v.passes() {
                "TRECE (gresit)"
            } else {
                "NU TRECE"
            }
        );
        assert!(caught, "{f:?} nu a fost prinsa");
        assert!(!v.passes(), "{f:?}: verdictul bibliotecii a trecut");
    }
}

/// Deterministic numbers in [−1, 1).
fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// #513's real shapes (C 768, 12 heads of 64, vocab 16 384, one text of 64 — `pas_de_aur_513/reteta.json`) but
/// `layers` layers and **made-up weights**: #513's weights are not in the cloud (2.7 GB, the hub has only the recipe),
/// so this is not the 513 golden step — it checks that the same list of launches, at #513's widths, gives the CPU
/// oracle's numbers (one AdamW step) in the hub's tolerances.
fn shapes_513(layers: usize) {
    shapes_513_on(Kind::Kept, layers, (1, 64));
}

/// [`shapes_513`] on the plan `kind` with the batch `(bsz, tl)`; returns the run's first loss and norm (C-157 point 2
/// after os-403, T212: the same check on [`Plan::new_tiles3`]).
fn shapes_513_on(kind: Kind, layers: usize, (bsz, tl): (usize, usize)) -> (f32, f32) {
    let ar = Arch {
        layers,
        ..train::M513
    };
    let mut r = rng(513);
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with("ln1.weight")
                || s.name.ends_with("ln2.weight")
                || s.name == "ln.weight"
            {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.02 * r()).collect()
            }
        })
        .collect();
    let n = bsz * tl;
    let toks: Vec<u32> = (0..=n).map(|_| ((r() + 1.0) * 8_191.9) as u32).collect();
    let (inputs, targets) = (toks[..n].to_vec(), toks[1..].to_vec());
    let run = run_plan(
        kind,
        ar,
        (bsz, tl),
        &params,
        (&inputs, &targets),
        Recipe::AdamW513,
        1,
        Fault::None,
    );
    let (loss, norm) = (run.out.losses[0], run.out.norms[0]);

    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: bsz,
        len: tl,
    };
    let cpu = train::golden_run(&ar, &params, &batch, Recipe::AdamW513, 1).unwrap();
    let dev = |ours: &[Vec<f32>], theirs: &[Vec<f32>]| ts::deviation(ours, theirs);
    let g = dev(&run.out.grad1, &cpu.grad_step1);
    let w = dev(&run.out.after1, &cpu.after_step1);
    assert!(g.finite && w.finite);
    let names = ar.tensors();
    let lrel = ((loss - cpu.losses[0]) as f64).abs() / cpu.losses[0] as f64;
    let nrel =
        (norm as f64 - train::grad_norm(&cpu.grad_step1)).abs() / train::grad_norm(&cpu.grad_step1);
    println!(
        "kt c165 formele lui #513 ({kind:?}, {layers} straturi, greutati inventate, {bsz} x {tl}): {} lansari, memoria placii {} MiB; fata de oracol: eroarea {loss} rel {lrel:.2e}, norma rel {nrel:.2e}, grad_abs {:.2e} ({}) grad_rel_norma {:.2e}, w1_abs {:.2e} ({})",
        run.total_per_step,
        run.board_bytes >> 20,
        g.abs,
        names[g.at].name,
        g.rel_norm,
        w.abs,
        names[w.at].name
    );
    let extra = if kind.split() { 8 * layers + 2 } else { 0 };
    assert_eq!(run.total_per_step, 31 * layers + 13 + extra);
    assert!(lrel <= train::TOL_LOSS_REL, "eroarea {lrel}");
    assert!(nrel <= 1e-5, "norma {nrel}");
    assert!(g.abs <= train::TOL_GRAD_ABS && g.rel_norm <= train::TOL_GRAD_REL_NORM);
    assert!(w.abs <= train::TOL_ADAMW_W_ABS);
    (loss, norm)
}

#[test]
fn t165_one_layer_at_513_widths_matches_the_oracle() {
    shapes_513(1);
}

/// All 12 layers of #513 (minutes in the simulator; run by hand: `cargo test --release -- --ignored`).
#[test]
#[ignore]
fn t165_all_12_layers_at_513_widths_matches_the_oracle() {
    shapes_513(12);
}

// ------------------------------------------------------------------ T212: #513's step on the `_3` attention (os-403)

/// C-157 point 2 after os-403 (`docs/C-157_P2_OS403_ATENTIA_513.md`, prediction 1, on the CPU — nothing runs): the
/// step os-403 sent (`Plan::new(...).resident()`, #513, 2 x 512, 12 layers) beside the same step on the third tiled
/// attention generation (`Plan::new_tiles3(...).resident()`): the same arena, the same 385 launches with the same tags,
/// layers and cbufs; only the 36 attention launches change program (and grid, to [8, 12, 2]).
#[test]
fn t212_the_513_step_on_tiles3_is_the_os403_list_with_the_attention_swapped() {
    let (bsz, tl) = (2usize, 512usize);
    let ar = train::M513;
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let old = Plan::new(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    let new = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    assert!(new.tiles3() && !old.tiles3());
    assert_eq!(old.bytes(), new.bytes(), "arena");
    assert_eq!(new.programs(), &ts::PROGRAMS_TILES3[..]);
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    drop(params);
    let (lo, ln) = (
        old.step(Recipe::AdamW513, &opt, 1).unwrap(),
        new.step(Recipe::AdamW513, &opt, 1).unwrap(),
    );
    assert_eq!(lo.len(), 385);
    assert_eq!(ln.len(), 385);
    assert_eq!(new.launches_per_step(Recipe::AdamW513), 385);
    let (mut same, mut swapped) = (0usize, [0usize; 3]);
    for (o, n) in lo.iter().zip(&ln) {
        assert_eq!(
            (o.tag, o.layer, o.cbuf),
            (n.tag, n.layer, n.cbuf),
            "{}",
            o.tag
        );
        match (o.prog, n.prog) {
            (Prog::AttnFwd, Prog::AttnFwdTile3) => swapped[0] += 1,
            (Prog::AttnBack, Prog::AttnBackTile0_3) => swapped[1] += 1,
            (Prog::AttnBack, Prog::AttnBackTile3) => swapped[2] += 1,
            (a, b) => {
                assert_eq!(a, b, "{}", o.tag);
                assert_eq!(o.grid, n.grid, "{}", o.tag);
                same += 1;
                continue;
            }
        }
        assert_eq!(n.grid, [8, 12, 2], "{}", n.tag);
    }
    println!(
        "kt t212 pasul #513 2 x 512: arena {} MiB la ambele, {} lansari, {same} neschimbate, atentia schimbata {swapped:?} (fwd, dS, dqkv)",
        new.bytes() >> 20,
        ln.len()
    );
    assert_eq!(same, 349);
    assert_eq!(swapped, [12, 12, 12]);
    // The OS sends the step in parts of at most `split` launches: the same count → the same parts (255 + 130).
    assert_eq!(ln.len().div_ceil(255), lo.len().div_ceil(255));
}

/// Prediction 2 (SIMULARE): one layer at #513's widths, made-up weights, 2 x 128 (two tiles per text, the diagonal
/// tile included) on [`Plan::new_tiles3`] = the CPU oracle in the hub's tolerances; beside [`Plan::new`] on the same
/// data the loss differs by ≤ 1e-6 relative (not bit for bit: the tiled forward sums in another order).
#[test]
fn t212_one_layer_at_513_widths_on_tiles3_matches_the_oracle() {
    shapes_513_on(Kind::Tiles3, 1, (2, 128));
}

/// [`t212_one_layer_at_513_widths_on_tiles3_matches_the_oracle`] beside [`Plan::new`] on the same data (two runs,
/// ~3 minutes; run by hand: `cargo test --release -- --ignored t212`). Session 206: both 9.750025 / 3.738398.
#[test]
#[ignore]
fn t212_one_layer_at_513_widths_tiles3_beside_plan_new() {
    let (l3, n3) = shapes_513_on(Kind::Tiles3, 1, (2, 128));
    let (l0, n0) = shapes_513_on(Kind::Kept, 1, (2, 128));
    let lrel = ((l3 - l0) as f64).abs() / l0 as f64;
    let nrel = ((n3 - n0) as f64).abs() / n0 as f64;
    println!(
        "kt t212 _3 fata de Plan::new (1 strat, 2 x 128): eroarea {l3} / {l0} rel {lrel:.2e}, norma {n3} / {n0} rel {nrel:.2e}"
    );
    assert!(lrel <= 1e-6, "eroarea {lrel}");
}

// ------------------------------------------------------------------ T177: P rebuilt instead of kept (C-160 point 7)

/// Bit-for-bit equality of everything two runs read back.
fn same_bits(a: &Outcome, b: &Outcome) -> bool {
    let flat = |v: &[Vec<f32>]| {
        v.iter()
            .flatten()
            .map(|x| x.to_bits())
            .collect::<Vec<u32>>()
    };
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    bits(&a.losses) == bits(&b.losses)
        && bits(&a.norms) == bits(&b.norms)
        && flat(&a.grad1) == flat(&b.grad1)
        && flat(&a.after1) == flat(&b.after1)
        && flat(&a.after_last) == flat(&b.after_last)
}

/// Predictions 1 and 2 (`docs/C-160_P7_P_RECALCULAT.md`): the small golden step with one shared P, rebuilt in the
/// backward pass, gives the kept-P numbers bit for bit (SIMULARE), one launch more, one P buffer less.
#[test]
fn t177_p_rebuilt_small_golden_step_is_bit_for_bit_the_kept_p_step() {
    for (recipe, name, n) in [
        (Recipe::Sgd, "sgd", 73usize),
        (Recipe::AdamW513, "adamw_513", 75),
    ] {
        let kept = run_mic_plan(Kind::Kept, recipe, 10, Fault::None);
        let rec = run_mic_plan(Kind::Recalc, recipe, 10, Fault::None);
        assert_eq!(kept.total_per_step, n);
        assert_eq!(rec.total_per_step, n + 1);
        let p_bytes = (4 * (B * MIC.heads * T * T) as u64).div_ceil(256) * 256 + 256;
        assert_eq!(kept.board_bytes - rec.board_bytes, p_bytes);
        let same = same_bits(&kept.out, &rec.out);
        let v = ts::judge(recipe, rec.total_per_step, &rec.out, &refs(name));
        println!(
            "kt c177 {name}: P recalculat {} lansari (P pastrat {}), memoria {} B mai putin; 10 pasi bit cu bit = {}; verdictul fata de PyTorch: {}",
            rec.total_per_step,
            kept.total_per_step,
            kept.board_bytes - rec.board_bytes,
            if same { "DA" } else { "NU" },
            if v.passes() { "TRECE" } else { "NU TRECE" }
        );
        assert!(same, "{name}: P recalculat nu da aceiasi biti");
        assert!(v.passes());
    }
}

/// Prediction 3: four layers (made-up weights), three AdamW steps — still bit for bit, three re-launches.
#[test]
fn t177_p_rebuilt_four_layers_bit_for_bit() {
    let ar = Arch { layers: 4, ..MIC };
    let mut r = rng(177);
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with("ln1.weight")
                || s.name.ends_with("ln2.weight")
                || s.name == "ln.weight"
            {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.3 * r()).collect()
            }
        })
        .collect();
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let run = |kind: Kind, fault: Fault| {
        run_plan(
            kind,
            ar,
            (B, T),
            &params,
            (&inputs, &targets),
            Recipe::AdamW513,
            3,
            fault,
        )
    };
    let kept = run(Kind::Kept, Fault::None);
    let rec = run(Kind::Recalc, Fault::None);
    assert_eq!(rec.total_per_step, kept.total_per_step + 3);
    assert!(same_bits(&kept.out, &rec.out), "4 straturi: alti biti");
    // Prediction 4, here: without the re-launches layers 0..2 use layer 3's P.
    let bad = run(Kind::Recalc, Fault::NoPRebuild);
    assert_eq!(bad.total_per_step, kept.total_per_step);
    assert!(
        !same_bits(&kept.out, &bad.out),
        "fara relansari: aceiasi biti?"
    );
    println!(
        "kt c177 4 straturi: {} lansari (+3), 3 pasi AdamW bit cu bit = DA; fara relansari: biti diferiti (PRINSA)",
        rec.total_per_step
    );
}

/// Prediction 4: the shared-P plan without its re-launches is caught against PyTorch at step 1.
#[test]
fn t177_missing_p_rebuild_is_caught() {
    let rf = refs("sgd");
    let r = run_mic_plan(Kind::Recalc, Recipe::Sgd, 1, Fault::NoPRebuild);
    let g = worst(&r.out.grad1, &rf.grad1);
    let w1 = worst(&r.out.after1, &rf.after1);
    let caught = g.abs > train::TOL_GRAD_ABS
        || g.rel_norm > train::TOL_GRAD_REL_NORM
        || w1.abs > train::TOL_SGD_W_ABS;
    let v = ts::judge(Recipe::Sgd, r.total_per_step, &r.out, &rf);
    println!(
        "kt c177 greseala fara relansari (pasul mic, sgd, 1 pas): grad_abs {:.2e} ({}), w1_abs {:.2e} ({}) -> {}; verdictul: {}",
        g.abs,
        g.where_,
        w1.abs,
        w1.where_,
        if caught { "PRINSA" } else { "SCAPATA" },
        if v.passes() {
            "TRECE (gresit)"
        } else {
            "NU TRECE"
        }
    );
    assert!(caught);
    assert!(!v.passes());
}

/// Prediction 5 (DEDUS, addresses only — nothing runs): the arena of #513 / #651 at real batches, P kept or rebuilt.
#[test]
fn t177_arena_memory_at_513_and_651() {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let m651 = Arch {
        layers: 24,
        ..train::M513
    };
    for (name, ar, bsz, want_kept, want_rec) in [
        ("#513 12 x 512", train::M513, 12usize, 7.7f64, 6.2f64),
        ("#651 4 x 512", m651, 4, 6.5, 5.4),
        ("#651 6 x 512", m651, 6, 8.4, 6.8),
    ] {
        let tl = 512;
        let inputs: Vec<u32> = (0..(bsz * tl) as u32)
            .map(|i| (i * 7919) % 16_384)
            .collect();
        let kept = Plan::new(ar, bsz, tl, &inputs, HEAP).unwrap();
        let rec = Plan::new_p_recalc(ar, bsz, tl, &inputs, HEAP).unwrap();
        assert_eq!(kept.p_relaunches(), 0);
        assert_eq!(rec.p_relaunches(), ar.layers - 1);
        let (k, r) = (gib(kept.bytes()), gib(rec.bytes()));
        println!(
            "kt c177 memoria arenei {name}: P pastrat {k:.2} GiB (estimat {want_kept}), P recalculat {r:.2} GiB (estimat {want_rec}); sub 7 GiB: {} / {}",
            if k < 7.0 { "DA" } else { "NU" },
            if r < 7.0 { "DA" } else { "NU" }
        );
        assert!((k / want_kept - 1.0).abs() <= 0.03, "{name}: {k}");
        assert!((r / want_rec - 1.0).abs() <= 0.03, "{name}: {r}");
    }
}

// ------------------------------------------------------------------ C-160 point 7: `attn_p` in the step (T185)

/// Predictions 2 (`docs/C-160_P7_ATTN_P_IN_PAS.md`): the small golden step with one shared P rebuilt by `attn_p` gives
/// the kept-P numbers (and the `attn_fwd`-rebuilt numbers) bit for bit, 10 steps, both recipes (SIMULARE).
#[test]
fn t185_attn_p_rebuilt_small_golden_step_is_bit_for_bit() {
    for (recipe, name, n) in [
        (Recipe::Sgd, "sgd", 73usize),
        (Recipe::AdamW513, "adamw_513", 75),
    ] {
        let kept = run_mic_plan(Kind::Kept, recipe, 10, Fault::None);
        let rec = run_mic_plan(Kind::Recalc, recipe, 10, Fault::None);
        let only = run_mic_plan(Kind::POnly, recipe, 10, Fault::None);
        assert_eq!(only.total_per_step, n + 1);
        assert_eq!(only.board_bytes, rec.board_bytes);
        let count = |r: &Run, p: Prog| {
            r.launches_per_step
                .iter()
                .find(|e| e.0 == p)
                .map_or(0, |e| e.1)
        };
        assert_eq!(count(&only, Prog::AttnP), 1);
        assert_eq!(count(&only, Prog::AttnFwd), count(&kept, Prog::AttnFwd));
        assert_eq!(count(&rec, Prog::AttnP), 0);
        let same_kept = same_bits(&kept.out, &only.out);
        let same_rec = same_bits(&rec.out, &only.out);
        let v = ts::judge(recipe, only.total_per_step, &only.out, &refs(name));
        println!(
            "kt c185 {name}: attn_p in pas {} lansari (1 attn_p); 10 pasi bit cu bit = P pastrat {} / attn_fwd relansat {}; verdictul fata de PyTorch: {}",
            only.total_per_step,
            if same_kept { "DA" } else { "NU" },
            if same_rec { "DA" } else { "NU" },
            if v.passes() { "TRECE" } else { "NU TRECE" }
        );
        assert!(
            same_kept && same_rec,
            "{name}: attn_p in pas nu da aceiasi biti"
        );
        assert!(v.passes());
    }
}

/// Prediction 3: four layers, three AdamW steps — bit for bit, three `attn_p` re-launches; prediction 4: without
/// them, or on another layer's `qkv`, other bits.
#[test]
fn t185_attn_p_four_layers_and_planted_mistakes() {
    let ar = Arch { layers: 4, ..MIC };
    let mut r = rng(185);
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with("ln1.weight")
                || s.name.ends_with("ln2.weight")
                || s.name == "ln.weight"
            {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.3 * r()).collect()
            }
        })
        .collect();
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let run = |kind: Kind, fault: Fault| {
        run_plan(
            kind,
            ar,
            (B, T),
            &params,
            (&inputs, &targets),
            Recipe::AdamW513,
            3,
            fault,
        )
    };
    let kept = run(Kind::Kept, Fault::None);
    let only = run(Kind::POnly, Fault::None);
    assert_eq!(only.total_per_step, kept.total_per_step + 3);
    assert_eq!(
        only.launches_per_step
            .iter()
            .find(|e| e.0 == Prog::AttnP)
            .map(|e| e.1),
        Some(3)
    );
    assert!(same_bits(&kept.out, &only.out), "4 straturi: alti biti");
    let none = run(Kind::POnly, Fault::NoPRebuild);
    assert!(
        !same_bits(&kept.out, &none.out),
        "fara attn_p: aceiasi biti?"
    );
    let wrong = run(Kind::POnly, Fault::PWrongQkv);
    assert_eq!(wrong.total_per_step, only.total_per_step);
    assert!(
        !same_bits(&kept.out, &wrong.out),
        "qkv gresit: aceiasi biti?"
    );
    println!(
        "kt c185 4 straturi: {} lansari (3 attn_p), 3 pasi AdamW bit cu bit = DA; fara relansari: PRINSA; qkv-ul altui strat: PRINSA",
        only.total_per_step
    );
}

// ------------------------------------------------------------------ T187: the tiled programs in the step (C-160 point 7)

/// Prediction 1 (`docs/C-160_P7_PLACI_IN_PAS.md`): the tiled plans have the arenas of `new` / `new_p_only`, the same
/// launches with the same tags, layers and cbufs; only the program and grid of the attention launches change.
#[test]
fn t187_the_tiled_list() {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let params = tensors("initiale");
    let opt = AdamW::recipe_513(&params);
    let h = MIC.heads as u32;
    for recipe in [Recipe::Sgd, Recipe::AdamW513] {
        for (old, new, programs) in [
            (
                Plan::new(MIC, B, T, &inputs, HEAP).unwrap(),
                Plan::new_tiles(MIC, B, T, &inputs, HEAP).unwrap(),
                &ts::PROGRAMS_TILES[..],
            ),
            (
                Plan::new_p_only(MIC, B, T, &inputs, HEAP).unwrap(),
                Plan::new_p_only_tiles(MIC, B, T, &inputs, HEAP).unwrap(),
                &ts::PROGRAMS_TILES_P_ONLY[..],
            ),
        ] {
            assert_eq!(old.bytes(), new.bytes());
            assert!(new.tiles() && !old.tiles());
            assert_eq!(new.programs(), programs);
            assert_eq!(new.p_relaunches(), old.p_relaunches());
            let (lo, ln) = (
                old.step(recipe, &opt, 1).unwrap(),
                new.step(recipe, &opt, 1).unwrap(),
            );
            assert_eq!(lo.len(), ln.len());
            assert_eq!(ln.len(), new.launches_per_step(recipe));
            let mut swapped = 0;
            for (o, n) in lo.iter().zip(&ln) {
                assert_eq!((o.tag, o.layer, o.cbuf), (n.tag, n.layer, n.cbuf));
                assert!(
                    programs.contains(&n.prog),
                    "{} nu e in lista",
                    n.prog.name()
                );
                let want = match (o.prog, o.tag) {
                    (Prog::AttnFwd, "inainte:atentia") => Prog::AttnFwdTile2,
                    (Prog::AttnP, _) => Prog::AttnPTile,
                    (Prog::AttnBack, "inapoi:atentia-dS") => Prog::AttnBackTile0_2,
                    (Prog::AttnBack, "inapoi:atentia-dqkv") => Prog::AttnBackTile2,
                    _ => o.prog,
                };
                assert_eq!(n.prog, want, "{}", o.tag);
                if want != o.prog {
                    swapped += 1;
                    assert_eq!(n.grid, [1, h, B as u32], "{}", o.tag);
                    assert!(n.prog.shared().is_some());
                } else {
                    assert_eq!(n.grid, o.grid);
                }
            }
            let l = MIC.layers;
            assert_eq!(swapped, 3 * l + old.p_relaunches());
        }
    }
    // d > 64 refused.
    let wide = Arch {
        heads: 1,
        n_emb: 128,
        ..MIC
    };
    assert!(Plan::new_tiles(wide, B, T, &inputs, HEAP).is_err());
    assert!(Plan::new_p_only_tiles(wide, B, T, &inputs, HEAP).is_err());
    // Every program of every plan: the SHA-256 checked, the old indices unchanged.
    ts::check_programs().unwrap();
    for (k, p) in ts::PROGRAMS_P_ONLY.iter().enumerate() {
        assert_eq!(p.index(), k);
    }
    for (k, p) in ts::PROGRAMS_ALL.iter().enumerate() {
        assert_eq!(p.index(), k);
    }
    println!(
        "kt c187 lista: aceleasi lansari, etichete si cbuf-uri ca new / new_p_only; schimbate doar {} lansari de atentie pe pas (grila [1, {h}, {B}])",
        3 * MIC.layers + 1
    );
}

/// Predictions 2–4: the small golden step on tiles, 10 steps, both recipes — PyTorch's verdict passes; the tiled
/// backward = the old backward bit for bit on the same P; P rebuilt with `attn_p_tile` = P kept bit for bit.
#[test]
fn t187_tiled_small_golden_step() {
    for (recipe, name) in [(Recipe::Sgd, "sgd"), (Recipe::AdamW513, "adamw_513")] {
        let kept = run_mic_plan(Kind::Kept, recipe, 10, Fault::None);
        let tiles = run_mic_plan(Kind::Tiles, recipe, 10, Fault::None);
        let only = run_mic_plan(Kind::POnlyTiles, recipe, 10, Fault::None);
        let oldb = run_mic_plan(Kind::Tiles, recipe, 10, Fault::OldBackward);
        assert_eq!(tiles.total_per_step, kept.total_per_step);
        assert_eq!(only.total_per_step, kept.total_per_step + 1);
        let v = ts::judge(recipe, tiles.total_per_step, &tiles.out, &refs(name));
        let vo = ts::judge(recipe, only.total_per_step, &only.out, &refs(name));
        // Against `Plan::new` (the old attention): the losses and the weights after the last step.
        let loss_rel = tiles
            .out
            .losses
            .iter()
            .zip(&kept.out.losses)
            .map(|(&a, &b)| ((a - b) as f64 / b as f64).abs())
            .fold(0.0f64, f64::max);
        let w = worst(&tiles.out.after_last, &kept.out.after_last);
        // The K part of every `in_proj_bias` (entries C … 2C): its true gradient is 0 (softmax does not change when a
        // row of scores moves by a constant), so the gradient there is rounding noise and AdamW turns noise into
        // steps of the learning rate's size — it is judged apart (absolute), everything else in the relative norm.
        let cc = MIC.n_emb;
        let split = |ws: &[Vec<f32>]| {
            let mut rest = ws.to_vec();
            let mut kb = Vec::new();
            for (t, spec) in rest.iter_mut().zip(MIC.tensors()) {
                if spec.name.ends_with("atentie.in_proj_bias") {
                    kb.extend_from_slice(&t[cc..2 * cc]);
                    t[cc..2 * cc].fill(0.0);
                }
            }
            (rest, kb)
        };
        let ((rt, kt), (rk, kk)) = (split(&tiles.out.after_last), split(&kept.out.after_last));
        let w_rest = worst(&rt, &rk);
        let kbias_abs = kt
            .iter()
            .zip(&kk)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let same_kept = same_bits(&tiles.out, &kept.out);
        let same_oldb = same_bits(&tiles.out, &oldb.out);
        let same_only = same_bits(&tiles.out, &only.out);
        println!(
            "kt c187 {name}: pe placi {} lansari; verdictul fata de PyTorch: {} (P refacut: {}); fata de Plan::new: bit cu bit {}, eroarea rel {:.2e}, greutatile dupa 10 pasi rel {:.2e} (abs {:.2e}, {}), fara bias-ul K rel {:.2e}, bias-ul K abs {:.2e}; inapoi pe placi = inapoi vechi bit cu bit {}; P refacut cu attn_p_tile = P pastrat bit cu bit {}",
            tiles.total_per_step,
            if v.passes() { "TRECE" } else { "NU TRECE" },
            if vo.passes() { "TRECE" } else { "NU TRECE" },
            if same_kept { "DA" } else { "NU" },
            loss_rel,
            w.rel_norm,
            w.abs,
            w.where_,
            w_rest.rel_norm,
            kbias_abs,
            if same_oldb { "DA" } else { "NU" },
            if same_only { "DA" } else { "NU" },
        );
        assert!(v.passes() && vo.passes(), "{name}: verdictul");
        assert!(
            loss_rel <= 1e-6,
            "{name}: erorile prea departe de Plan::new"
        );
        // Prediction 2 said rel ≤ 1e-5 for every tensor: missed at `adamw_513` on the K bias only (3.9e-3 rel there,
        // 7.8e-6 abs — see the doc); the bound below for the K bias is written after the run.
        assert!(
            w_rest.rel_norm <= 1e-5,
            "{name}: greutatile prea departe de Plan::new"
        );
        assert!(kbias_abs <= 2e-5, "{name}: bias-ul K prea departe");
        assert!(same_oldb, "{name}: inapoi pe placi != inapoi vechi");
        assert!(same_only, "{name}: attn_p_tile != P pastrat");
    }
}

/// Prediction 4 on four layers (3 `attn_p_tile` re-launches) and prediction 5: the planted mistakes are caught.
#[test]
fn t187_tiled_four_layers_and_planted_mistakes() {
    let ar = Arch { layers: 4, ..MIC };
    let mut r = rng(187);
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with("ln1.weight")
                || s.name.ends_with("ln2.weight")
                || s.name == "ln.weight"
            {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.3 * r()).collect()
            }
        })
        .collect();
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let run = |kind: Kind, fault: Fault| {
        run_plan(
            kind,
            ar,
            (B, T),
            &params,
            (&inputs, &targets),
            Recipe::AdamW513,
            3,
            fault,
        )
    };
    let tiles = run(Kind::Tiles, Fault::None);
    let only = run(Kind::POnlyTiles, Fault::None);
    assert_eq!(only.total_per_step, tiles.total_per_step + 3);
    assert_eq!(
        only.launches_per_step
            .iter()
            .find(|e| e.0 == Prog::AttnPTile)
            .map(|e| e.1),
        Some(3)
    );
    assert!(same_bits(&tiles.out, &only.out), "4 straturi: alti biti");
    let oldb = run(Kind::Tiles, Fault::OldBackward);
    assert!(same_bits(&tiles.out, &oldb.out), "4 straturi: inapoi vechi");
    let none = run(Kind::POnlyTiles, Fault::NoPRebuild);
    assert!(
        !same_bits(&tiles.out, &none.out),
        "fara attn_p_tile: aceiasi biti?"
    );
    let wrong = run(Kind::POnlyTiles, Fault::PWrongQkv);
    assert!(
        !same_bits(&tiles.out, &wrong.out),
        "qkv gresit: aceiasi biti?"
    );
    // Mode 0 on `attn_back_tile_2`: dS is never written, so the columns read a dS no launch wrote — on the simulated
    // board that memory is unmapped and the step is refused; either way (refused or other bits) the mistake is caught.
    let rows = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(Kind::Tiles, Fault::TileRowsOnCols)
    }));
    let rows_caught = match &rows {
        Err(_) => "pasul refuzat",
        Ok(r) if !same_bits(&tiles.out, &r.out) => "alti biti",
        Ok(_) => "NEPRINSA",
    };
    assert_ne!(
        rows_caught, "NEPRINSA",
        "modul 0 pe attn_back_tile_2: aceiasi biti?"
    );

    // A tiled launch whose QMD lacks the program's shared memory: refused by the command processor.
    let plan = Plan::new_tiles(MIC, B, T, &inputs, HEAP).unwrap();
    let mic_params = tensors("initiale");
    let mut b = Board::new();
    for (va, bytes) in plan.uploads(&mic_params, &inputs, &targets).unwrap() {
        b.cp.mem.write(va, &bytes);
    }
    let list = plan
        .step(Recipe::Sgd, &AdamW::recipe_513(&mic_params), 1)
        .unwrap();
    let k = list
        .iter()
        .position(|l| l.prog == Prog::AttnFwdTile2)
        .unwrap();
    let l = list[k].clone();
    let refused = b.try_submit(&list, |zone| {
        let at = k * ts::SLOT as usize;
        let q = nvidia_ga104::qmd::ComputeQmd {
            program_va: prog_va(l.prog),
            registers: l.prog.registers(),
            cbuf0_va: ZONE + at as u64,
            cbuf0_size: 0x100,
            grid: l.grid,
            block: l.prog.block(),
            release: None,
        }
        .to_bytes()
        .unwrap();
        zone[at + 0x100..at + 0x100 + q.len()].copy_from_slice(&q);
    });
    assert!(refused.is_err(), "QMD fara memoria partajata: acceptat?");
    println!(
        "kt c187 4 straturi: {} lansari (3 attn_p_tile), 3 pasi AdamW bit cu bit = P pastrat DA, = inapoi vechi DA; fara relansari / qkv-ul altui strat / modul 0 pe attn_back_tile_2 ({rows_caught}) / QMD fara memoria partajata: PRINSE",
        only.total_per_step
    );
}

// ------------------------------------------------------------------ T195: the `_3` attention in the step (C-157 p.2)

/// Prediction 1 (`docs/C-157_P2_ATTN3_IN_PAS.md`): the `_3` plans have the arenas of `new_tiles` /
/// `new_p_only_tiles`, the same launches with the same tags, layers, cbufs and grids; only the program of the 3·L
/// attention launches changes (the P re-launches stay on `attn_p_tile`). Prediction 6: `PROGRAMS_ALL` unchanged.
#[test]
fn t195_the_tiles3_list() {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let params = tensors("initiale");
    let opt = AdamW::recipe_513(&params);
    for recipe in [Recipe::Sgd, Recipe::AdamW513] {
        for (old, new, programs) in [
            (
                Plan::new_tiles(MIC, B, T, &inputs, HEAP).unwrap(),
                Plan::new_tiles3(MIC, B, T, &inputs, HEAP).unwrap(),
                &ts::PROGRAMS_TILES3[..],
            ),
            (
                Plan::new_p_only_tiles(MIC, B, T, &inputs, HEAP).unwrap(),
                Plan::new_p_only_tiles3(MIC, B, T, &inputs, HEAP).unwrap(),
                &ts::PROGRAMS_TILES3_P_ONLY[..],
            ),
        ] {
            assert_eq!(old.bytes(), new.bytes());
            assert!(new.tiles() && new.tiles3() && !old.tiles3());
            assert_eq!(new.programs(), programs);
            assert_eq!(new.p_relaunches(), old.p_relaunches());
            let (lo, ln) = (
                old.step(recipe, &opt, 1).unwrap(),
                new.step(recipe, &opt, 1).unwrap(),
            );
            assert_eq!(lo.len(), ln.len());
            let mut swapped = 0;
            for (o, n) in lo.iter().zip(&ln) {
                assert_eq!(
                    (o.tag, o.layer, o.cbuf, o.grid),
                    (n.tag, n.layer, n.cbuf, n.grid)
                );
                assert!(
                    programs.contains(&n.prog),
                    "{} nu e in lista",
                    n.prog.name()
                );
                let want = match o.prog {
                    Prog::AttnFwdTile2 => Prog::AttnFwdTile3,
                    Prog::AttnBackTile0_2 => Prog::AttnBackTile0_3,
                    Prog::AttnBackTile2 => Prog::AttnBackTile3,
                    p => p,
                };
                assert_eq!(n.prog, want, "{}", o.tag);
                if want != o.prog {
                    swapped += 1;
                    assert!(n.prog.shared().unwrap().bytes >= o.prog.shared().unwrap().bytes);
                }
            }
            assert_eq!(swapped, 3 * MIC.layers);
        }
    }
    // d > 64 refused, as `new_tiles`.
    let wide = Arch {
        heads: 1,
        n_emb: 128,
        ..MIC
    };
    assert!(Plan::new_tiles3(wide, B, T, &inputs, HEAP).is_err());
    assert!(Plan::new_p_only_tiles3(wide, B, T, &inputs, HEAP).is_err());
    // Prediction 6: `PROGRAMS_ALL` unchanged (18, the same indices; the OS's simulated board walks it), the three `_3`
    // programs after it in `PROGRAMS_ALL_3`, all 21 SHA-256 checked.
    assert_eq!(PROGRAMS_ALL.len(), 18);
    assert_eq!(PROGRAMS_ALL_3[..18], PROGRAMS_ALL[..]);
    for (k, p) in PROGRAMS_ALL_3.iter().enumerate() {
        assert_eq!(p.index(), k);
    }
    ts::check_programs().unwrap();
    println!(
        "kt c195 lista: aceleasi lansari, etichete, cbuf-uri si grile ca new_tiles / new_p_only_tiles; schimbate doar {} lansari de atentie pe pas (relansarile P raman pe attn_p_tile); PROGRAMS_ALL neschimbat (18)",
        3 * MIC.layers
    );
}

/// Predictions 2 and 4: the small golden step, 10 steps, both recipes — the `_3` plan = the `_2` plan bit for bit
/// (SIMULARE: the same CPU models, so this checks the wiring), PyTorch's verdict passes; `_3` forward with `_2`
/// backward and the other way round give the same bits.
#[test]
fn t195_tiles3_small_golden_step() {
    for (recipe, name) in [(Recipe::Sgd, "sgd"), (Recipe::AdamW513, "adamw_513")] {
        let two = run_mic_plan(Kind::Tiles, recipe, 10, Fault::None);
        let three = run_mic_plan(Kind::Tiles3, recipe, 10, Fault::None);
        let only3 = run_mic_plan(Kind::POnlyTiles3, recipe, 10, Fault::None);
        let mix_f = run_mic_plan(Kind::Tiles3, recipe, 10, Fault::Swap3To2Fwd);
        let mix_b = run_mic_plan(Kind::Tiles3, recipe, 10, Fault::Swap3To2Back);
        let mix_23 = run_mic_plan(Kind::Tiles, recipe, 10, Fault::Swap2To3Back);
        assert_eq!(three.total_per_step, two.total_per_step);
        assert_eq!(only3.total_per_step, two.total_per_step + 1);
        assert_eq!(
            three
                .launches_per_step
                .iter()
                .filter(|e| matches!(
                    e.0,
                    Prog::AttnFwdTile3 | Prog::AttnBackTile0_3 | Prog::AttnBackTile3
                ))
                .map(|e| e.1)
                .sum::<usize>(),
            3 * MIC.layers
        );
        let v = ts::judge(recipe, three.total_per_step, &three.out, &refs(name));
        let vo = ts::judge(recipe, only3.total_per_step, &only3.out, &refs(name));
        let s32 = same_bits(&three.out, &two.out);
        let so = same_bits(&three.out, &only3.out);
        let sm = [
            same_bits(&three.out, &mix_f.out),
            same_bits(&three.out, &mix_b.out),
            same_bits(&three.out, &mix_23.out),
        ];
        let yn = |b: bool| if b { "DA" } else { "NU" };
        println!(
            "kt c195 {name}: _3 {} lansari; verdictul fata de PyTorch: {} (P refacut: {}); _3 = _2 bit cu bit {}; P refacut = P pastrat {}; amestec (fwd _2 / inapoi _2 / inapoi _3 in planul _2) {} / {} / {} (SIMULARE)",
            three.total_per_step,
            if v.passes() { "TRECE" } else { "NU TRECE" },
            if vo.passes() { "TRECE" } else { "NU TRECE" },
            yn(s32),
            yn(so),
            yn(sm[0]),
            yn(sm[1]),
            yn(sm[2]),
        );
        assert!(v.passes() && vo.passes(), "{name}: verdictul");
        assert!(s32, "{name}: _3 != _2");
        assert!(so, "{name}: P refacut != P pastrat");
        assert!(sm.iter().all(|&b| b), "{name}: amestecul");
    }
}

/// Predictions 3 and 5 on four layers (3 `attn_p_tile` re-launches): bit for bit, and the planted mistakes caught.
#[test]
fn t195_tiles3_four_layers_and_planted_mistakes() {
    let ar = Arch { layers: 4, ..MIC };
    let mut r = rng(195);
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with("ln1.weight")
                || s.name.ends_with("ln2.weight")
                || s.name == "ln.weight"
            {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.3 * r()).collect()
            }
        })
        .collect();
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let run = |kind: Kind, fault: Fault| {
        run_plan(
            kind,
            ar,
            (B, T),
            &params,
            (&inputs, &targets),
            Recipe::AdamW513,
            3,
            fault,
        )
    };
    let two = run(Kind::Tiles, Fault::None);
    let three = run(Kind::Tiles3, Fault::None);
    let only3 = run(Kind::POnlyTiles3, Fault::None);
    assert_eq!(only3.total_per_step, three.total_per_step + 3);
    assert_eq!(
        only3
            .launches_per_step
            .iter()
            .find(|e| e.0 == Prog::AttnPTile)
            .map(|e| e.1),
        Some(3)
    );
    assert!(same_bits(&three.out, &two.out), "4 straturi: _3 != _2");
    assert!(same_bits(&three.out, &only3.out), "4 straturi: P refacut");
    let none = run(Kind::POnlyTiles3, Fault::NoPRebuild);
    assert!(
        !same_bits(&three.out, &none.out),
        "fara attn_p_tile: aceiasi biti?"
    );
    let wrong = run(Kind::POnlyTiles3, Fault::PWrongQkv);
    assert!(
        !same_bits(&three.out, &wrong.out),
        "qkv gresit: aceiasi biti?"
    );
    // Mode 0 on `attn_back_tile_3`: dS never written → refused (unmapped) or other bits.
    let rows = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run(Kind::Tiles3, Fault::Tile3RowsOnCols)
    }));
    let rows_caught = match &rows {
        Err(_) => "pasul refuzat",
        Ok(r) if !same_bits(&three.out, &r.out) => "alti biti",
        Ok(_) => "NEPRINSA",
    };
    assert_ne!(
        rows_caught, "NEPRINSA",
        "modul 0 pe attn_back_tile_3: aceiasi biti?"
    );

    // Every `_3` launch whose QMD lacks the program's shared memory: refused by the command processor.
    let plan = Plan::new_tiles3(MIC, B, T, &inputs, HEAP).unwrap();
    let mic_params = tensors("initiale");
    let list = plan
        .step(Recipe::Sgd, &AdamW::recipe_513(&mic_params), 1)
        .unwrap();
    let mut refused_all = 0;
    for prog in [
        Prog::AttnFwdTile3,
        Prog::AttnBackTile0_3,
        Prog::AttnBackTile3,
    ] {
        let mut b = Board::new();
        for (va, bytes) in plan.uploads(&mic_params, &inputs, &targets).unwrap() {
            b.cp.mem.write(va, &bytes);
        }
        let k = list.iter().position(|l| l.prog == prog).unwrap();
        let l = list[k].clone();
        let refused = b.try_submit(&list, |zone| {
            let at = k * ts::SLOT as usize;
            let q = nvidia_ga104::qmd::ComputeQmd {
                program_va: prog_va(l.prog),
                registers: l.prog.registers(),
                cbuf0_va: ZONE + at as u64,
                cbuf0_size: 0x100,
                grid: l.grid,
                block: l.prog.block(),
                release: None,
            }
            .to_bytes()
            .unwrap();
            zone[at + 0x100..at + 0x100 + q.len()].copy_from_slice(&q);
        });
        assert!(
            refused.is_err(),
            "{}: QMD fara memoria partajata acceptat?",
            prog.name()
        );
        refused_all += 1;
    }
    println!(
        "kt c195 4 straturi: {} lansari (3 attn_p_tile), 3 pasi AdamW: _3 = _2 bit cu bit DA, P refacut = P pastrat DA; fara relansari / qkv-ul altui strat / modul 0 pe attn_back_tile_3 ({rows_caught}) / QMD fara memoria partajata ({refused_all} din 3): PRINSE (SIMULARE)",
        only3.total_per_step
    );
}

// ------------------------------------------------------------------ T200: ln_back + the residual addition fused

/// T200 prediction 3 (`docs/C-160_P7_LIPIRE_INAPOI.md`): `with_fused_back` = the same list without the 2·L backward
/// `vadd`, each backward `ln_back` launch now `ln_back_res` with the same arguments + `dres` (the vadd's other
/// operand); the same arena; `PROGRAMS_ALL` / `PROGRAMS_ALL_3` unchanged.
#[test]
fn t200_the_fused_list() {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let params = tensors("initiale");
    let opt = AdamW::recipe_513(&params);
    let l2 = 2 * MIC.layers;
    for recipe in [Recipe::Sgd, Recipe::AdamW513] {
        for old in [
            Plan::new(MIC, B, T, &inputs, HEAP).unwrap(),
            Plan::new_tiles3(MIC, B, T, &inputs, HEAP).unwrap(),
            Plan::new_p_only_tiles3(MIC, B, T, &inputs, HEAP).unwrap(),
            Plan::new_tiles3_mv_ram(MIC, B, T, &inputs, HEAP).unwrap(),
        ] {
            let new = old.clone().with_fused_back();
            assert!(new.fused_back() && !old.fused_back());
            assert_eq!(new.bytes(), old.bytes());
            assert_eq!(new.programs(), old.programs());
            assert_eq!(new.fused_programs(), &[Prog::LnBackRes]);
            assert!(old.fused_programs().is_empty());
            if recipe == Recipe::AdamW513 && old.mv_in_ram() {
                continue; // the m, v in RAM step is built by `adam_ram` (T193), not by `step`
            }
            let (lo, ln) = (
                old.step(recipe, &opt, 1).unwrap(),
                new.step(recipe, &opt, 1).unwrap(),
            );
            assert_eq!(ln.len(), lo.len() - l2);
            assert_eq!(ln.len(), new.launches_per_step(recipe));
            assert_eq!(lo.len(), old.launches_per_step(recipe));
            // The forward `vadd` stay (2·L), the backward ones are gone.
            let vadds = |v: &[Launch]| v.iter().filter(|l| l.prog == Prog::Vadd).count();
            assert_eq!((vadds(&lo), vadds(&ln)), (2 * l2, l2));
            let mut it = ln.iter();
            let mut fused = 0;
            let mut k = 0;
            while k < lo.len() {
                let o = &lo[k];
                let n = it.next().unwrap();
                assert!(
                    new.programs().contains(&n.prog) || new.fused_programs().contains(&n.prog),
                    "{} nu e de incarcat",
                    n.prog.name()
                );
                if o.tag == "inapoi:ln2-dx" || o.tag == "inapoi:ln1-dx" {
                    // ln_back, colsum, vadd → ln_back_res, colsum.
                    let (cs, va) = (&lo[k + 1], &lo[k + 2]);
                    assert_eq!(va.prog, Prog::Vadd);
                    assert!(va.tag.starts_with("inapoi:rezidual-ln"));
                    let want_tag = if o.tag == "inapoi:ln2-dx" {
                        "inapoi:ln2-dx+rezidual"
                    } else {
                        "inapoi:ln1-dx+rezidual"
                    };
                    assert_eq!(
                        (n.prog, n.tag, n.layer, n.grid),
                        (Prog::LnBackRes, want_tag, o.layer, o.grid)
                    );
                    // x, dy, γ, μ, rstd, dx: the same addresses; dres = the vadd's other operand.
                    assert_eq!(n.cbuf[..0x58], o.cbuf[..0x58]);
                    let (a, b, c) = (
                        &va.cbuf[0x28..0x30],
                        &va.cbuf[0x30..0x38],
                        &va.cbuf[0x38..0x40],
                    );
                    assert_eq!(c, &o.cbuf[0x50..0x58], "vadd scrie peste dx al lui ln_back");
                    let other = if a == c { b } else { a };
                    assert_eq!(&n.cbuf[0x58..0x60], other);
                    assert_eq!(n.cbuf[0x60..0x68], o.cbuf[0x58..0x60]); // c, 1/c
                    let n2 = it.next().unwrap();
                    assert_eq!(n2, cs);
                    fused += 1;
                    k += 3;
                } else {
                    assert_eq!(n, o, "{}", o.tag);
                    k += 1;
                }
            }
            assert!(it.next().is_none());
            assert_eq!(fused, l2);
        }
    }
    // dx may not be dres or dy: no cbuf.
    let a = nvidia_ga104::ln_back_res::Args {
        x: HEAP,
        dy: HEAP + 0x1000,
        gamma: HEAP + 0x2000,
        mean: HEAP + 0x3000,
        rstd: HEAP + 0x4000,
        dx: HEAP + 0x5000,
        dres: HEAP + 0x6000,
        rows: 4,
        c: 64,
    };
    assert!(a.cbuf().is_some());
    assert!(
        nvidia_ga104::ln_back_res::Args { dres: a.dx, ..a }
            .cbuf()
            .is_none()
    );
    assert!(
        nvidia_ga104::ln_back_res::Args { dy: a.dx, ..a }
            .cbuf()
            .is_none()
    );
    // `PROGRAMS_ALL` (18, the OS's simulated board) and `PROGRAMS_ALL_3` (21) unchanged; `ln_back_res` after them.
    assert_eq!(PROGRAMS_ALL.len(), 18);
    assert_eq!(PROGRAMS_ALL_3.len(), 21);
    assert_eq!(PROGRAMS_ALL_FUSED[..21], PROGRAMS_ALL_3[..]);
    for (k, p) in PROGRAMS_ALL_FUSED.iter().enumerate() {
        assert_eq!(p.index(), k);
    }
    ts::check_programs().unwrap();
    println!(
        "kt c200 lista: {l2} lansari mai putine pe pas (toate vadd inapoi), ln_back -> ln_back_res cu aceleasi adrese + dres, restul identic; PROGRAMS_ALL neschimbat (18)"
    );
}

/// T200 predictions 1 and 3: the small golden step, 10 steps, both recipes — the fused plan = the plan without the
/// fusion bit for bit (P kept and `_3` tiles), PyTorch's verdict passes, 2·L launches fewer.
#[test]
fn t200_fused_small_golden_step() {
    for (recipe, name) in [(Recipe::Sgd, "sgd"), (Recipe::AdamW513, "adamw_513")] {
        let mut lines = Vec::new();
        for (old_k, new_k) in [
            (Kind::Kept, Kind::KeptFused),
            (Kind::Tiles3, Kind::Tiles3Fused),
        ] {
            let old = run_mic_plan(old_k, recipe, 10, Fault::None);
            let new = run_mic_plan(new_k, recipe, 10, Fault::None);
            assert_eq!(new.total_per_step + 2 * MIC.layers, old.total_per_step);
            assert_eq!(new.board_bytes, old.board_bytes);
            assert_eq!(
                new.launches_per_step
                    .iter()
                    .find(|e| e.0 == Prog::LnBackRes)
                    .map(|e| e.1),
                Some(2 * MIC.layers)
            );
            let same = same_bits(&new.out, &old.out);
            let v = ts::judge(recipe, new.total_per_step, &new.out, &refs(name));
            lines.push(format!(
                "{old_k:?}: {} -> {} lansari, 10 pasi bit cu bit {}, verdictul {}",
                old.total_per_step,
                new.total_per_step,
                if same { "DA" } else { "NU" },
                if v.passes() { "TRECE" } else { "NU TRECE" }
            ));
            assert!(same, "{name} {old_k:?}: lipit != nelipit");
            assert!(v.passes(), "{name} {old_k:?}: verdictul");
        }
        println!("kt c200 {name}: {} (SIMULARE)", lines.join("; "));
    }
}

/// T200 predictions 1 and 4 on four layers: bit for bit; the contracted model and a wrong `dres` caught.
#[test]
fn t200_fused_four_layers_and_planted_mistakes() {
    let ar = Arch { layers: 4, ..MIC };
    let mut r = rng(200);
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with("ln1.weight")
                || s.name.ends_with("ln2.weight")
                || s.name == "ln.weight"
            {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.3 * r()).collect()
            }
        })
        .collect();
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let run = |kind: Kind, fault: Fault| {
        run_plan(
            kind,
            ar,
            (B, T),
            &params,
            (&inputs, &targets),
            Recipe::AdamW513,
            3,
            fault,
        )
    };
    let old = run(Kind::Tiles3, Fault::None);
    let new = run(Kind::Tiles3Fused, Fault::None);
    assert_eq!(new.total_per_step + 8, old.total_per_step);
    assert!(
        same_bits(&new.out, &old.out),
        "4 straturi: lipit != nelipit"
    );
    // Prediction 4 (i): one FFMA instead of FMUL + FADD → other bits.
    let fma = run(Kind::Tiles3Fused, Fault::FusedContracted);
    let n_diff = fma
        .out
        .grad1
        .iter()
        .flatten()
        .zip(old.out.grad1.iter().flatten())
        .filter(|(a, b)| a.to_bits() != b.to_bits())
        .count();
    assert!(n_diff > 0, "contractia ffma: aceiasi biti?");
    // Prediction 4 (ii): `dres` = another buffer → other bits, and against PyTorch the small step fails.
    let wrong = run(Kind::Tiles3Fused, Fault::FusedWrongRes);
    assert!(
        !same_bits(&wrong.out, &old.out),
        "dres gresit: aceiasi biti?"
    );
    let rf = refs("sgd");
    let w = run_mic_plan(Kind::Tiles3Fused, Recipe::Sgd, 1, Fault::FusedWrongRes);
    let g = worst(&w.out.grad1, &rf.grad1);
    assert!(
        g.abs > train::TOL_GRAD_ABS || g.rel_norm > train::TOL_GRAD_REL_NORM,
        "dres gresit neprins fata de PyTorch"
    );
    let n_grad = old.out.grad1.iter().map(Vec::len).sum::<usize>();
    println!(
        "kt c200 4 straturi: {} -> {} lansari, 3 pasi AdamW bit cu bit DA; contractia ffma: {n_diff} din {n_grad} gradienti cu alti biti (PRINSA); dres gresit: alti biti + fata de PyTorch grad_abs={:.2e} (PRINSA) (SIMULARE)",
        old.total_per_step, new.total_per_step, g.abs
    );
}

// ------------------------------------------------------------------ T201: GELU's backward fused into proj-dX

/// T201 prediction G3 (`docs/C-160_P7_LIPIRE_GELU.md`): `with_fused_gelu` = the same list without the L backward
/// `gelu_tr`, each "inapoi:proj-dX" launch now `mm_train_gelu` with the same a / b / shape, C = the `gelu_tr`'s output
/// (`dwide2`) and u = its u; the same arena; combines with `with_fused_back`.
#[test]
fn t201_the_fused_gelu_list() {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let params = tensors("initiale");
    let opt = AdamW::recipe_513(&params);
    let ly = MIC.layers;
    for recipe in [Recipe::Sgd, Recipe::AdamW513] {
        for (old, back) in [
            (Plan::new(MIC, B, T, &inputs, HEAP).unwrap(), false),
            (Plan::new_tiles3(MIC, B, T, &inputs, HEAP).unwrap(), false),
            (
                Plan::new_tiles3(MIC, B, T, &inputs, HEAP)
                    .unwrap()
                    .with_fused_back(),
                true,
            ),
            (
                Plan::new_p_only_tiles3(MIC, B, T, &inputs, HEAP).unwrap(),
                false,
            ),
        ] {
            let new = old.clone().with_fused_gelu();
            assert!(new.fused_gelu() && !old.fused_gelu());
            assert_eq!(new.bytes(), old.bytes());
            assert_eq!(new.programs(), old.programs());
            let want: &[Prog] = if back {
                &[Prog::LnBackRes, Prog::MmTrainGelu]
            } else {
                &[Prog::MmTrainGelu]
            };
            assert_eq!(new.fused_programs(), want);
            let (lo, ln) = (
                old.step(recipe, &opt, 1).unwrap(),
                new.step(recipe, &opt, 1).unwrap(),
            );
            assert_eq!(ln.len(), lo.len() - ly);
            assert_eq!(ln.len(), new.launches_per_step(recipe));
            // The forward `gelu_tr` stay (L), the backward ones are gone.
            let gelus = |v: &[Launch]| v.iter().filter(|l| l.prog == Prog::GeluTr).count();
            assert_eq!((gelus(&lo), gelus(&ln)), (2 * ly, ly));
            let mut it = ln.iter();
            let mut fused = 0;
            let mut k = 0;
            while k < lo.len() {
                let o = &lo[k];
                if o.tag == "inapoi:gelu" {
                    k += 1; // gone
                    continue;
                }
                let n = it.next().unwrap();
                assert!(
                    new.programs().contains(&n.prog) || new.fused_programs().contains(&n.prog),
                    "{} nu e de incarcat",
                    n.prog.name()
                );
                if o.tag == "inapoi:proj-dX" {
                    // The matching gelu_tr: u, g = this C, y.
                    let g = lo[k..]
                        .iter()
                        .find(|l| l.tag == "inapoi:gelu" && l.layer == o.layer)
                        .unwrap();
                    assert_eq!(g.cbuf[0x30..0x38], o.cbuf[0x38..0x40], "gelu citeste C");
                    assert_eq!(
                        (n.prog, n.tag, n.layer, n.grid),
                        (Prog::MmTrainGelu, "inapoi:proj-dX+gelu", o.layer, o.grid)
                    );
                    assert_eq!(n.cbuf[0x28..0x38], o.cbuf[0x28..0x38]); // a, b
                    assert_eq!(n.cbuf[0x38..0x40], g.cbuf[0x38..0x40]); // C = gelu's y
                    assert_eq!(n.cbuf[0x40..0x48], g.cbuf[0x28..0x30]); // u
                    assert_eq!(n.cbuf[0x48..0x58], o.cbuf[0x48..0x58]); // m, n, k, flags (0)
                    fused += 1;
                } else {
                    assert_eq!(n, o, "{}", o.tag);
                }
                k += 1;
            }
            assert!(it.next().is_none());
            assert_eq!(fused, ly);
        }
    }
    // No bias, no c == u, no missing address.
    let a = nvidia_ga104::mm_train_gelu::Args {
        a: HEAP,
        b: HEAP + 0x1000,
        c: HEAP + 0x2000,
        u: HEAP + 0x3000,
        m: 4,
        n: 64,
        k: 16,
        flags: 0,
    };
    assert!(a.cbuf().is_some());
    use nvidia_ga104::mm_train_gelu::Args as G;
    assert!(G { u: a.c, ..a }.cbuf().is_none());
    assert!(G { u: 0, ..a }.cbuf().is_none());
    assert!(
        G {
            flags: nvidia_ga104::mm_train::BIAS,
            ..a
        }
        .cbuf()
        .is_none()
    );
    assert_eq!(PROGRAMS_ALL_FUSED2[..22], PROGRAMS_ALL_FUSED[..]);
    for (k, p) in PROGRAMS_ALL_FUSED2.iter().enumerate() {
        assert_eq!(p.index(), k);
    }
    ts::check_programs().unwrap();
    println!(
        "kt c201 lista: {ly} lansari mai putine pe pas (toate gelu_tr inapoi), proj-dX -> mm_train_gelu cu aceleasi a, b, forma + u, C = dwide2; restul identic"
    );
}

/// T201 predictions G1 and G3: the small golden step, 10 steps, both recipes — with `mm_train_gelu` (alone and with
/// `ln_back_res`) = without bit for bit, PyTorch's verdict passes, L (3L) launches fewer.
#[test]
fn t201_fused_gelu_small_golden_step() {
    for (recipe, name) in [(Recipe::Sgd, "sgd"), (Recipe::AdamW513, "adamw_513")] {
        let mut lines = Vec::new();
        for (old_k, new_k, fewer) in [
            (Kind::Kept, Kind::KeptGelu, MIC.layers),
            (Kind::Tiles3, Kind::Tiles3Gelu, MIC.layers),
            (Kind::Tiles3, Kind::Tiles3Both, 3 * MIC.layers),
        ] {
            let old = run_mic_plan(old_k, recipe, 10, Fault::None);
            let new = run_mic_plan(new_k, recipe, 10, Fault::None);
            assert_eq!(new.total_per_step + fewer, old.total_per_step);
            assert_eq!(new.board_bytes, old.board_bytes);
            assert_eq!(
                new.launches_per_step
                    .iter()
                    .find(|e| e.0 == Prog::MmTrainGelu)
                    .map(|e| e.1),
                Some(MIC.layers)
            );
            let same = same_bits(&new.out, &old.out);
            let v = ts::judge(recipe, new.total_per_step, &new.out, &refs(name));
            lines.push(format!(
                "{new_k:?}: {} -> {} lansari, 10 pasi bit cu bit {}, verdictul {}",
                old.total_per_step,
                new.total_per_step,
                if same { "DA" } else { "NU" },
                if v.passes() { "TRECE" } else { "NU TRECE" }
            ));
            assert!(same, "{name} {new_k:?}: lipit != nelipit");
            assert!(v.passes(), "{name} {new_k:?}: verdictul");
        }
        println!("kt c201 {name}: {} (SIMULARE)", lines.join("; "));
    }
}

/// T201 predictions G1 and G4 on four layers: bit for bit; the unfused epilogue, the next layer's u and a forgotten
/// GELU' caught.
#[test]
fn t201_fused_gelu_four_layers_and_planted_mistakes() {
    let ar = Arch { layers: 4, ..MIC };
    let mut r = rng(201);
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with("ln1.weight")
                || s.name.ends_with("ln2.weight")
                || s.name == "ln.weight"
            {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.3 * r()).collect()
            }
        })
        .collect();
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let run = |kind: Kind, fault: Fault| {
        run_plan(
            kind,
            ar,
            (B, T),
            &params,
            (&inputs, &targets),
            Recipe::AdamW513,
            3,
            fault,
        )
    };
    let old = run(Kind::Tiles3, Fault::None);
    let new = run(Kind::Tiles3Both, Fault::None);
    assert_eq!(new.total_per_step + 12, old.total_per_step);
    assert!(
        same_bits(&new.out, &old.out),
        "4 straturi: lipit != nelipit"
    );
    let diff = |o: &Run| {
        o.out
            .grad1
            .iter()
            .flatten()
            .zip(old.out.grad1.iter().flatten())
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count()
    };
    // G2's risk: the epilogue compiled without the FFMA of gelu_tr → other bits.
    let unf = run(Kind::Tiles3Both, Fault::GeluUnfused);
    let n_unf = diff(&unf);
    assert!(n_unf > 0, "epilogul fara ffma: aceiasi biti?");
    // G4: the next layer's u → other bits; GELU' forgotten → other bits and PyTorch's verdict fails.
    let wu = run(Kind::Tiles3Both, Fault::GeluWrongU);
    assert!(!same_bits(&wu.out, &old.out), "u gresit: aceiasi biti?");
    let fg = run(Kind::Tiles3Both, Fault::GeluForgotten);
    assert!(!same_bits(&fg.out, &old.out), "GELU' uitat: aceiasi biti?");
    let rf = refs("sgd");
    let mut worst_abs = Vec::new();
    for fault in [Fault::GeluWrongU, Fault::GeluForgotten] {
        let w = run_mic_plan(Kind::Tiles3Gelu, Recipe::Sgd, 1, fault);
        let g = worst(&w.out.grad1, &rf.grad1);
        assert!(
            g.abs > train::TOL_GRAD_ABS || g.rel_norm > train::TOL_GRAD_REL_NORM,
            "{fault:?} neprins fata de PyTorch"
        );
        worst_abs.push(g.abs);
    }
    let n_grad = old.out.grad1.iter().map(Vec::len).sum::<usize>();
    println!(
        "kt c201 4 straturi: {} -> {} lansari, 3 pasi AdamW bit cu bit DA; epilogul fara ffma: {n_unf} din {n_grad} gradienti cu alti biti (PRINS); u al stratului urmator: grad_abs={:.2e} fata de PyTorch (PRINS); GELU' uitat: grad_abs={:.2e} (PRINS) (SIMULARE)",
        old.total_per_step, new.total_per_step, worst_abs[0], worst_abs[1]
    );
}

// ------------------------------------------------------------------ order 20:11 step 4: recomputation (T203)

/// Predictions 1 and 2 (`docs/C-160_P9_RECALCULARE.md`): the small golden step with every layer but the last re-run in
/// the backward pass gives the plan's own numbers bit for bit, 10 steps, both recipes (SIMULARE).
#[test]
fn t203_recompute_small_golden_step_is_bit_for_bit() {
    for (recipe, name, n) in [
        (Recipe::Sgd, "sgd", 73usize),
        (Recipe::AdamW513, "adamw_513", 75),
    ] {
        let mut lines = Vec::new();
        for (old_k, new_k, more) in [
            (Kind::Kept, Kind::KeptRecompute, 8isize),
            (Kind::Tiles3Both, Kind::Tiles3BothRecompute, 8),
            (Kind::POnlyTiles3, Kind::POnlyTiles3Recompute, 7),
        ] {
            let old = run_mic_plan(old_k, recipe, 10, Fault::None);
            let new = run_mic_plan(new_k, recipe, 10, Fault::None);
            if old_k == Kind::Kept {
                assert_eq!(old.total_per_step, n);
            }
            assert_eq!(
                new.total_per_step as isize,
                old.total_per_step as isize + more
            );
            assert!(new.board_bytes < old.board_bytes);
            let same = same_bits(&new.out, &old.out);
            let v = ts::judge(recipe, new.total_per_step, &new.out, &refs(name));
            lines.push(format!(
                "{new_k:?}: {} -> {} lansari, arena {} -> {} B, 10 pasi bit cu bit {}, verdictul {}",
                old.total_per_step,
                new.total_per_step,
                old.board_bytes,
                new.board_bytes,
                if same { "DA" } else { "NU" },
                if v.passes() { "TRECE" } else { "NU TRECE" }
            ));
            assert!(same, "{name} {new_k:?}: recalculat != pastrat");
            assert!(v.passes(), "{name} {new_k:?}: verdictul");
        }
        println!("kt c203 {name}: {} (SIMULARE)", lines.join("; "));
    }
}

/// Predictions 3 and 4 on four layers: bit for bit, +24 launches; the re-run forgotten or fed the wrong input caught.
#[test]
fn t203_recompute_four_layers_and_planted_mistakes() {
    let ar = Arch { layers: 4, ..MIC };
    let mut r = rng(203);
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with("ln1.weight")
                || s.name.ends_with("ln2.weight")
                || s.name == "ln.weight"
            {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.3 * r()).collect()
            }
        })
        .collect();
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let run = |kind: Kind, fault: Fault| {
        run_plan(
            kind,
            ar,
            (B, T),
            &params,
            (&inputs, &targets),
            Recipe::AdamW513,
            3,
            fault,
        )
    };
    let old = run(Kind::Tiles3Both, Fault::None);
    let new = run(Kind::Tiles3BothRecompute, Fault::None);
    assert_eq!(new.total_per_step, old.total_per_step + 24);
    assert!(
        same_bits(&new.out, &old.out),
        "4 straturi: recalculat != pastrat"
    );
    let nr = run(Kind::Tiles3BothRecompute, Fault::NoRecompute);
    assert_eq!(nr.total_per_step, old.total_per_step);
    assert!(
        !same_bits(&nr.out, &old.out),
        "fara recalculare: aceiasi biti?"
    );
    let wi = run(Kind::Tiles3BothRecompute, Fault::RecomputeWrongInput);
    assert!(
        !same_bits(&wi.out, &old.out),
        "intrarea gresita: aceiasi biti?"
    );
    let rf = refs("sgd");
    let w = run_mic_plan(Kind::KeptRecompute, Recipe::Sgd, 1, Fault::NoRecompute);
    let g = worst(&w.out.grad1, &rf.grad1);
    assert!(
        g.abs > train::TOL_GRAD_ABS || g.rel_norm > train::TOL_GRAD_REL_NORM,
        "fara recalculare neprins fata de PyTorch"
    );
    println!(
        "kt c203 4 straturi: {} -> {} lansari, arena {} -> {} B, 3 pasi AdamW bit cu bit DA; fara recalculare: biti diferiti (PRINS), pas mic sgd grad_abs={:.2e} fata de PyTorch (PRINS); intrarea stratului 1 la stratul 0: biti diferiti (PRINS) (SIMULARE)",
        old.total_per_step, new.total_per_step, old.board_bytes, new.board_bytes, g.abs
    );
}

/// Prediction 5: a resident plan keeps the recomputation; `rebatch` keeps every address.
#[test]
fn t203_recompute_and_resident() {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let a = Plan::new_tiles3(MIC, B, T, &inputs, HEAP)
        .unwrap()
        .with_recompute()
        .unwrap()
        .resident()
        .unwrap();
    let b = Plan::new_tiles3(MIC, B, T, &inputs, HEAP)
        .unwrap()
        .resident()
        .unwrap()
        .with_recompute()
        .unwrap();
    assert!(a.recomputes() && a.is_resident() && a.tiles3());
    assert_eq!(a, b);
    let again = a.clone().with_recompute().unwrap();
    assert_eq!(again, a);
    let other: Vec<u32> = inputs.iter().map(|&t| (t + 1) % MIC.vocab as u32).collect();
    let rb = a.rebatch(&other).unwrap();
    assert_eq!(
        (rb.layers.clone(), rb.xs.clone(), rb.end),
        (a.layers.clone(), a.xs.clone(), a.end)
    );
    assert_eq!(rb.recompute_launches(), 8 * (MIC.layers - 1));
    assert_eq!(rb.p_relaunches(), 0);
    println!(
        "kt c203 rezident: recalcularea pastrata, aceeasi arena ({} B) in ambele ordini, lotul nou aceleasi adrese (SIMULARE)",
        a.bytes()
    );
}

/// Prediction 6 (DEDUS, addresses only — nothing runs): the arena of #513 / #651 with recomputation.
#[test]
fn t203_arena_memory_with_recompute() {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    let m651 = Arch {
        layers: 24,
        ..train::M513
    };
    let tl = 512;
    let plan = |ar: Arch, bsz: usize, rc: bool| {
        let inputs: Vec<u32> = (0..(bsz * tl) as u32)
            .map(|i| (i * 7919) % 16_384)
            .collect();
        let p = Plan::new_p_recalc(ar, bsz, tl, &inputs, HEAP).unwrap();
        if rc { p.with_recompute().unwrap() } else { p }
    };
    let mut lines = Vec::new();
    for (name, ar, bsz, want) in [
        ("#513 12x512", train::M513, 12usize, 3.25f64),
        ("#651 4x512", m651, 4, 3.40),
        ("#651 6x512", m651, 6, 3.73),
        ("#651 12x512", m651, 12, 4.73),
    ] {
        let before = gib(plan(ar, bsz, false).bytes());
        let after = gib(plan(ar, bsz, true).bytes());
        lines.push(format!(
            "{name}: P recalculat {before:.2} -> stratul recalculat {after:.2} GiB (estimat {want})"
        ));
        assert!((after / want - 1.0).abs() <= 0.05, "{name}: {after}");
        assert!(after < 7.0);
    }
    let mut best = 0;
    for bsz in 1..=40 {
        if gib(plan(m651, bsz, true).bytes()) < 7.0 {
            best = bsz;
        }
    }
    println!(
        "kt c203 memoria arenei: {}; cel mai mare lot x 512 al lui #651 sub 7 GiB: {best} (estimat ~25) (DEDUS)",
        lines.join("; ")
    );
    assert!((23..=27).contains(&best), "{best}");
}

// ------------------------------------------------------------------ T215: `mm_train` on 64×64 tiles in the step

/// `mm_train`'s arguments back from a launch's cbuf (the layout of `mm_train::Args::cbuf`).
fn mm_args(l: &Launch) -> nvidia_ga104::mm_train::Args {
    let u64_at = |o: usize| u64::from_le_bytes(l.cbuf[o..o + 8].try_into().unwrap());
    let u32_at = |o: usize| u32::from_le_bytes(l.cbuf[o..o + 4].try_into().unwrap());
    nvidia_ga104::mm_train::Args {
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

/// Prediction 1 (`docs/C-157_P2x_MM_TRAIN_TILE.md`, "Plan::with_mm_tile"): the #513 step of os-407 (`Plan::new_tiles3`,
/// resident, 2 × 512) with [`Plan::with_mm_tile`] is the same list — same arena, 385 launches, same tags, layers,
/// cbufs — with exactly the 147 `mm_train` launches on `mm_train_tile` and grid ⌈N/64⌉ × ⌈M/64⌉; the option survives
/// `resident()` and `with_recompute()`; nothing else changes.
#[test]
fn t215_the_513_step_with_mm_tile_is_the_same_list_with_the_products_swapped() {
    let (bsz, tl) = (2usize, 512usize);
    let ar = train::M513;
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let old = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    let new = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile()
        .resident()
        .unwrap();
    assert!(new.mm_tile() && !old.mm_tile() && new.tiles3());
    assert!(new.clone().with_recompute().unwrap().mm_tile());
    assert_eq!(old.bytes(), new.bytes(), "arena");
    assert_eq!(new.programs(), old.programs());
    assert_eq!(new.mm_tile_programs(), &ts::PROGRAMS_MM_TILE_EXTRA[..]);
    assert!(old.mm_tile_programs().is_empty());
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    drop(params);
    let (lo, ln) = (
        old.step(Recipe::AdamW513, &opt, 1).unwrap(),
        new.step(Recipe::AdamW513, &opt, 1).unwrap(),
    );
    assert_eq!((lo.len(), ln.len()), (385, 385));
    assert_eq!(new.launches_per_step(Recipe::AdamW513), 385);
    let (mut same, mut swapped, mut blocks_old, mut blocks_new) = (0usize, 0usize, 0u64, 0u64);
    for (o, n) in lo.iter().zip(&ln) {
        assert_eq!(
            (o.tag, o.layer, o.cbuf),
            (n.tag, n.layer, n.cbuf),
            "{}",
            o.tag
        );
        if (o.prog, n.prog) == (Prog::MmTrain, Prog::MmTrainTile) {
            let a = mm_args(n);
            assert_eq!(n.grid, [a.n.div_ceil(64), a.m.div_ceil(64), 1], "{}", n.tag);
            assert_eq!(o.grid, a.grid(), "{}", o.tag);
            blocks_old += o.grid.iter().map(|&g| g as u64).product::<u64>();
            blocks_new += n.grid.iter().map(|&g| g as u64).product::<u64>();
            swapped += 1;
        } else {
            assert_eq!((o.prog, o.grid), (n.prog, n.grid), "{}", o.tag);
            same += 1;
        }
    }
    // The part of the step before the optimizer (`step_until_opt`, the `_mv_ram` path) is swapped the same way.
    let (lu, _) = new.step_until_opt(&opt, 1).unwrap();
    assert_eq!(&lu[..], &ln[..ln.len() - 1]);
    println!(
        "kt t215 pasul #513 2 x 512 (_3, rezident): {} lansari, {same} neschimbate, {swapped} mm_train -> mm_train_tile; blocuri {blocks_old} -> {blocks_new}",
        ln.len()
    );
    assert_eq!(swapped, 147);
    assert_eq!(same, 238);
    assert!(
        blocks_new * 4 <= blocks_old,
        "blocuri {blocks_old} -> {blocks_new}"
    );
}

/// Prediction 2 (SIMULARE): the small golden step with [`Plan::with_mm_tile`] gives the plan's numbers **bit for bit**
/// (sgd and adamw_513, 10 steps) on `Plan::new`, `Plan::new_tiles3` and `_3` + both fusions; the verdict against PyTorch
/// passes; the launch count is unchanged.
#[test]
fn t215_mm_tile_small_golden_step_is_bit_for_bit() {
    for (recipe, name) in [(Recipe::Sgd, "sgd"), (Recipe::AdamW513, "adamw_513")] {
        for (base, tiled) in [
            (Kind::Kept, Kind::KeptTile),
            (Kind::Tiles3, Kind::Tiles3Tile),
            (Kind::Tiles3Both, Kind::Tiles3BothTile),
        ] {
            let a = run_mic_plan(base, recipe, 10, Fault::None);
            let b = run_mic_plan(tiled, recipe, 10, Fault::None);
            assert_eq!(a.total_per_step, b.total_per_step);
            let n_tile = b
                .launches_per_step
                .iter()
                .filter(|e| e.0 == Prog::MmTrainTile)
                .map(|e| e.1)
                .sum::<usize>();
            let n_mm = a
                .launches_per_step
                .iter()
                .filter(|e| e.0 == Prog::MmTrain)
                .map(|e| e.1)
                .sum::<usize>();
            assert_eq!(n_tile, n_mm, "{name} {tiled:?}");
            assert!(b.launches_per_step.iter().all(|e| e.0 != Prog::MmTrain));
            let v = ts::judge(recipe, b.total_per_step, &b.out, &refs(name));
            let s = same_bits(&a.out, &b.out);
            println!(
                "kt t215 {name} {tiled:?}: {} lansari ({n_tile} pe mm_train_tile); fata de PyTorch {}; = {base:?} bit cu bit {} (SIMULARE)",
                b.total_per_step,
                if v.passes() { "TRECE" } else { "NU TRECE" },
                if s { "DA" } else { "NU" }
            );
            assert!(v.passes(), "{name} {tiled:?}: verdictul");
            assert!(s, "{name} {tiled:?}: nu e bit cu bit");
        }
    }
}

/// Prediction 3: a grid with one block row too few is caught (different bits); `mm_train`'s own, much larger grid is
/// harmless (the extra blocks write nothing: the same bits).
#[test]
fn t215_mm_tile_grid_mistakes() {
    let base = run_mic_plan(Kind::Tiles3Tile, Recipe::AdamW513, 3, Fault::None);
    let old_grid = run_mic_plan(Kind::Tiles3Tile, Recipe::AdamW513, 3, Fault::TileOldGrid);
    assert!(same_bits(&base.out, &old_grid.out), "grila veche");
    let short = std::panic::catch_unwind(|| {
        run_mic_plan(Kind::Tiles3Tile, Recipe::AdamW513, 3, Fault::TileShortGrid)
    });
    let caught = match short {
        Err(_) => true,
        Ok(r) => !same_bits(&base.out, &r.out),
    };
    println!(
        "kt t215 grila mai mica prinsa: {} (SIMULARE)",
        if caught { "DA" } else { "NU" }
    );
    assert!(caught, "grila mai mica nu e prinsa");
}

// ------------------------------------------------------------------ T216: `colsum` split over rows in the step

/// Prediction 1 (`docs/C-157_P2y_COLSUM_IMPARTIT.md`): the #513 step of os-407 (`new_tiles3`, resident, 2 × 512) with
/// [`Plan::with_colsum_split`] (and `--mm-placi`): 385 → 483 launches; every `colsum` becomes `colsum_part` + `colsum`
/// (+ `colsum` for a LayerNorm) with the same tag and layer, the rest of the list unchanged in order; the work zone
/// sits after `end`, inside `bytes()`, and is big enough.
#[test]
fn t216_the_513_step_with_colsum_split() {
    let ar = train::M513;
    let (bsz, tl) = (2, 512);
    let inputs: Vec<u32> = (0..(bsz * tl) as u32).map(|i| (i * 37) % 16_384).collect();
    let old = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile()
        .resident()
        .unwrap();
    let new = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split()
        .resident()
        .unwrap();
    assert!(new.colsum_split() && !old.colsum_split() && new.mm_tile());
    assert!(new.clone().with_recompute().unwrap().colsum_split());
    assert!(new.clone().with_batch_slots().unwrap().colsum_split());
    assert_eq!(
        new.colsum_split_programs(),
        &ts::PROGRAMS_COLSUM_SPLIT_EXTRA[..]
    );
    assert!(old.colsum_split_programs().is_empty() && old.colsum_scratch().is_none());
    let (p0, p1, words) = new.colsum_scratch().unwrap();
    assert_eq!(words, 32 * 4 * 768);
    assert!(p0 >= new.end && p1 >= p0 + 4 * words as u64);
    assert!(p1 + 4 * words as u64 <= new.base + new.bytes());
    assert_eq!(old.end, new.end);
    let extra = new.bytes() - old.bytes();
    assert!(extra <= 1 << 20, "{extra}");
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    drop(params);
    let (lo, ln) = (
        old.step(Recipe::AdamW513, &opt, 1).unwrap(),
        new.step(Recipe::AdamW513, &opt, 1).unwrap(),
    );
    assert_eq!((lo.len(), ln.len()), (385, 483));
    assert_eq!(new.launches_per_step(Recipe::AdamW513), 483);
    let (mut same, mut bias, mut ln_sums) = (0usize, 0usize, 0usize);
    let mut j = 0;
    for o in &lo {
        if o.prog != Prog::Colsum {
            assert_eq!(o, &ln[j], "{}", o.tag);
            same += 1;
            j += 1;
            continue;
        }
        let part = &ln[j];
        assert_eq!(part.prog, Prog::ColsumPart, "{}", o.tag);
        assert_eq!((part.tag, part.layer), (o.tag, o.layer));
        let u = |l: &Launch, at: usize| u32::from_le_bytes(l.cbuf[at..at + 4].try_into().unwrap());
        let v = |l: &Launch, at: usize| u64::from_le_bytes(l.cbuf[at..at + 8].try_into().unwrap());
        // The same input, rows, columns and mode as the old launch; partials into the work zone; chunk 32.
        for at in [0x28, 0x30, 0x38, 0x40] {
            assert_eq!(v(part, at), v(o, at), "{} 0x{at:x}", o.tag);
        }
        assert_eq!(
            (u(part, 0x58), u(part, 0x5c), u(part, 0x60)),
            (u(o, 0x58), u(o, 0x5c), u(o, 0x60))
        );
        assert_eq!((v(part, 0x48), u(part, 0x64)), (p0, 32));
        assert_eq!(part.grid, [u(o, 0x5c).div_ceil(256), 32, 1]);
        let sums = if u(o, 0x60) != 0 { 2 } else { 1 };
        for (i, (from, to)) in [(p0, v(o, 0x48)), (p1, v(o, 0x50))]
            .into_iter()
            .take(sums)
            .enumerate()
        {
            let s = &ln[j + 1 + i];
            assert_eq!((s.prog, s.tag, s.layer), (Prog::Colsum, o.tag, o.layer));
            assert_eq!(
                (v(s, 0x28), v(s, 0x48), u(s, 0x58), u(s, 0x5c), u(s, 0x60)),
                (from, to, 32, u(o, 0x5c), 0)
            );
        }
        if sums == 2 {
            ln_sums += 1;
        } else {
            bias += 1;
        }
        j += 1 + sums;
    }
    assert_eq!(j, ln.len());
    assert_eq!((same, bias, ln_sums), (312, 48, 25));
    let (lu, _) = new.step_until_opt(&opt, 1).unwrap();
    assert_eq!(&lu[..], &ln[..ln.len() - 1]);
    let mixed = new.step_mixed(Recipe::AdamW513, &opt, 1, 16_384.0).unwrap();
    assert_eq!(mixed.len(), new.launches_per_step_mixed(Recipe::AdamW513));
    println!(
        "kt t216 pasul #513 2 x 512 (_3, --mm-placi, rezident): {} -> {} lansari ({bias} sume de deplasare, {ln_sums} LayerNorm); zona de lucru {} KiB dupa arena",
        lo.len(),
        ln.len(),
        extra >> 10
    );
}

/// Prediction 2 and 3 (SIMULARE): the small golden step with [`Plan::with_colsum_split`] — not bit for bit (another
/// order of the sums), but the verdict against PyTorch passes, the losses stay within 1e-5 of the unsplit plan and the
/// step-1 gradients are as close to the f64 oracle as without the option.
#[test]
fn t216_colsum_split_small_golden_step() {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let params = tensors("initiale");
    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: B,
        len: T,
    };
    for (recipe, name) in [(Recipe::Sgd, "sgd"), (Recipe::AdamW513, "adamw_513")] {
        let cpu = train::golden_run(&MIC, &params, &batch, recipe, 1).unwrap();
        for (base, split) in [
            (Kind::Kept, Kind::KeptSplit),
            (Kind::Tiles3Tile, Kind::Tiles3TileSplit),
            (Kind::Tiles3BothTile, Kind::Tiles3BothTileSplit),
        ] {
            let a = run_mic_plan(base, recipe, 10, Fault::None);
            let b = run_mic_plan(split, recipe, 10, Fault::None);
            let count = |r: &Run, p: Prog| {
                r.launches_per_step
                    .iter()
                    .filter(|e| e.0 == p)
                    .map(|e| e.1)
                    .sum::<usize>()
            };
            assert_eq!(b.total_per_step, a.total_per_step + 8 * MIC.layers + 2);
            assert_eq!(count(&b, Prog::ColsumPart), count(&a, Prog::Colsum));
            assert_eq!(
                count(&b, Prog::Colsum),
                count(&a, Prog::Colsum) + 2 * MIC.layers + 1
            );
            let v = ts::judge(recipe, b.total_per_step, &b.out, &refs(name));
            let worst_loss = a
                .out
                .losses
                .iter()
                .zip(&b.out.losses)
                .map(|(x, y)| ((x - y).abs() / x.abs()) as f64)
                .fold(0.0, f64::max);
            let ga = ts::deviation(&a.out.grad1, &cpu.grad_step1);
            let gb = ts::deviation(&b.out.grad1, &cpu.grad_step1);
            println!(
                "kt t216 {name} {split:?}: {} lansari ({} colsum_part); fata de PyTorch {}; bit cu bit cu {base:?}: {}; pierderile fata de {base:?} {worst_loss:.1e}; gradientul pasului 1 fata de oracolul f64: {:.2e} (fara: {:.2e}), rel {:.2e} (fara: {:.2e}) (SIMULARE)",
                b.total_per_step,
                count(&b, Prog::ColsumPart),
                if v.passes() { "TRECE" } else { "NU TRECE" },
                if same_bits(&a.out, &b.out) {
                    "DA"
                } else {
                    "NU"
                },
                gb.abs,
                ga.abs,
                gb.rel_norm,
                ga.rel_norm
            );
            assert!(v.passes(), "{name} {split:?}: verdictul");
            assert!(worst_loss <= 1e-5, "{name} {split:?}: {worst_loss}");
            assert!(
                gb.finite
                    && gb.abs <= train::TOL_GRAD_ABS
                    && gb.rel_norm <= train::TOL_GRAD_REL_NORM
            );
            assert!(
                gb.abs <= 2.0 * ga.abs.max(1e-9),
                "{name} {split:?}: {} > {}",
                gb.abs,
                ga.abs
            );
        }
    }
}

/// One layer at the widths of #513 (2 × 128) with the split, against the f64 oracle (as T165 / T212).
#[test]
fn t216_one_layer_at_513_widths_with_the_split_matches_the_oracle() {
    let a = shapes_513_on(Kind::Tiles3Tile, 1, (2, 128));
    let b = shapes_513_on(Kind::Tiles3TileSplit, 1, (2, 128));
    println!(
        "kt t216 un strat #513 2 x 128: eroarea {} / {}, norma {} / {} (fara / cu impartire, SIMULARE)",
        a.0, b.0, a.1, b.1
    );
}

/// A planted mistake: the second LayerNorm sum (dβ) of every split dropped — dβ stays 0 and the step is caught.
#[test]
fn t216_a_dropped_second_sum_is_caught() {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let params = tensors("initiale");
    let plan = Plan::new(MIC, B, T, &inputs, HEAP)
        .unwrap()
        .with_colsum_split();
    let opt = AdamW::recipe_513(&params);
    let full = plan.step(Recipe::Sgd, &opt, 1).unwrap();
    let mut cut = Vec::new();
    for (i, l) in full.iter().enumerate() {
        // A `colsum` right after another `colsum` that follows a `colsum_part` = the second sum of a LayerNorm.
        let second = i >= 2
            && l.prog == Prog::Colsum
            && full[i - 1].prog == Prog::Colsum
            && full[i - 2].prog == Prog::ColsumPart;
        if !second {
            cut.push(l.clone());
        }
    }
    assert_eq!(cut.len(), full.len() - (2 * MIC.layers + 1));
    let run = |list: &[Launch]| {
        let mut b = Board::new();
        for (va, bytes) in plan.uploads(&params, &inputs, &targets).unwrap() {
            b.cp.mem.write(va, &bytes);
        }
        b.submit(list);
        plan.unflat(&b.read(plan.g, plan.n_params)).unwrap()
    };
    let (g_full, g_cut) = (run(&full), run(&cut));
    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: B,
        len: T,
    };
    let cpu = train::golden_run(&MIC, &params, &batch, Recipe::Sgd, 1).unwrap();
    let ok = ts::deviation(&g_full, &cpu.grad_step1);
    let bad = ts::deviation(&g_cut, &cpu.grad_step1);
    println!(
        "kt t216 suma a doua lipsa: gradientul {:.2e} fata de {:.2e} (prinsa: {})",
        bad.abs,
        ok.abs,
        bad.abs > train::TOL_GRAD_ABS
    );
    assert!(ok.abs <= train::TOL_GRAD_ABS);
    assert!(bad.abs > train::TOL_GRAD_ABS, "nu e prinsa");
}

// ------------------------------------------------------------------ T219: the race's pick in the step

/// T219: every `mm_train` shape of `plan`'s step on a race variant, in turn (shape i on `VARIANTS[1 + i % 4]`), so that
/// all four variants run in one step.
fn turn_choice(plan: &Plan, params: &[Vec<f32>]) -> MmChoice {
    let opt = AdamW::recipe_513(params);
    let list = plan.step(Recipe::AdamW513, &opt, 1).unwrap();
    let mut c = MmChoice::new();
    for (i, f) in mv::forms(&list).iter().enumerate() {
        c.set(f.m, f.n, f.k, f.flags, 1 + i % (VARIANTS.len() - 1))
            .unwrap();
    }
    c
}

fn row(f: &mv::Form, variant: usize, median_ns: u64, same_bits: bool) -> RaceRow {
    RaceRow {
        m: f.m,
        n: f.n,
        k: f.k,
        flags: f.flags,
        variant,
        median_ns,
        same_bits,
    }
}

/// Prediction 1 (`docs/C-165_CURSA_MM.md`, "Plan::with_mm_choice"): the #513 step of os-410 (`new_tiles3`, resident,
/// 2 × 512, `--mm-placi`) with a pick from a (made-up) race report is the `with_mm_tile` list with only the chosen
/// shapes' launches on their variant and that variant's grid; same arena, 385 launches, same tags, layers, cbufs; the
/// pick survives `resident()`, `with_recompute()` and `with_colsum_split()`; the empty pick is `with_mm_tile` exactly.
#[test]
fn t219_the_513_step_with_the_race_pick_swaps_only_the_chosen_shapes() {
    let (bsz, tl) = (2usize, 512usize);
    let ar = train::M513;
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let tile = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile()
        .resident()
        .unwrap();
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    drop(params);
    let lt = tile.step(Recipe::AdamW513, &opt, 1).unwrap();
    let forms = mv::forms(&lt);
    assert_eq!(forms.len(), 15);
    // A made-up race report: on shape i variant 1 + i % 4 is 10 % faster than the witness, another variant 1 % faster
    // (below the 3 % threshold), and shape 0's best variant changed the bits (never chosen: the next one is).
    let mut rows = Vec::new();
    for (i, f) in forms.iter().enumerate() {
        rows.push(row(f, 0, 1_000_000, true));
        let best = 1 + i % 4;
        let other = 1 + (i + 1) % 4;
        rows.push(row(f, other, 990_000, true));
        if i == 0 {
            rows.push(row(f, best, 500_000, false));
        }
        rows.push(row(f, best, 900_000, true));
    }
    let pick = MmChoice::from_race(&rows).unwrap();
    for (i, f) in forms.iter().enumerate() {
        assert_eq!(pick.of(f.m, f.n, f.k, f.flags), 1 + i % 4, "{f:?}");
    }
    assert_eq!(pick.programs(), &PROGRAMS_ALL_CHOICE[25..]);
    let chosen = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_choice(pick)
        .resident()
        .unwrap();
    assert!(chosen.mm_tile() && chosen.mm_choice() == Some(&pick));
    assert_eq!(chosen.mm_choice_programs(), pick.programs());
    assert!(tile.mm_choice_programs().is_empty());
    assert_eq!(
        chosen.clone().with_recompute().unwrap().mm_choice(),
        Some(&pick)
    );
    assert_eq!(chosen.clone().with_colsum_split().mm_choice(), Some(&pick));
    assert_eq!(chosen.bytes(), tile.bytes(), "arena");
    assert_eq!(chosen.programs(), tile.programs());
    assert_eq!(chosen.mm_tile_programs(), tile.mm_tile_programs());
    let lc = chosen.step(Recipe::AdamW513, &opt, 1).unwrap();
    assert_eq!((lt.len(), lc.len()), (385, 385));
    let mut per_variant = [0usize; 5];
    let (mut blocks_tile, mut blocks_chosen) = (0u64, 0u64);
    for (t, c) in lt.iter().zip(&lc) {
        assert_eq!(
            (t.tag, t.layer, t.cbuf),
            (c.tag, c.layer, c.cbuf),
            "{}",
            t.tag
        );
        match c.prog.race_index() {
            Some(v) => {
                assert_eq!(t.prog, Prog::MmTrainTile, "{}", t.tag);
                let a = mm_args(c);
                assert_eq!(v, pick.of(a.m, a.n, a.k, a.flags), "{}", c.tag);
                assert_eq!(c.grid, VARIANTS[v].grid(&a), "{}", c.tag);
                assert_eq!(c.prog.block(), VARIANTS[v].block());
                blocks_tile += t.grid.iter().map(|&g| g as u64).product::<u64>();
                blocks_chosen += c.grid.iter().map(|&g| g as u64).product::<u64>();
                per_variant[v] += 1;
            }
            None => assert_eq!((t.prog, t.grid), (c.prog, c.grid), "{}", t.tag),
        }
    }
    assert_eq!(per_variant.iter().sum::<usize>(), 147);
    assert_eq!(per_variant[0], 0, "every shape was chosen");
    // The part before the optimizer (the `_mv_ram` path) is swapped the same way.
    let (lu, _) = chosen.step_until_opt(&opt, 1).unwrap();
    assert_eq!(&lu[..], &lc[..lc.len() - 1]);
    println!(
        "kt t219 pasul #513 2 x 512 cu alegerea cursei: 385 lansari, pe variante {per_variant:?} (martor, 128x64, 64x128, 128x128, 64x64k8); blocuri {blocks_tile} -> {blocks_chosen}"
    );
    // The empty pick is `with_mm_tile`, launch for launch.
    let empty = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_choice(MmChoice::new())
        .resident()
        .unwrap();
    assert_eq!(empty.step(Recipe::AdamW513, &opt, 1).unwrap(), lt);
    assert!(empty.mm_choice_programs().is_empty());
}

/// Prediction 2 (SIMULARE): the small golden step with every shape on a race variant (all four variants in one step)
/// gives `with_mm_tile`'s numbers **bit for bit** (sgd and adamw_513, 10 steps) on `new_tiles3` and `_3` + both
/// fusions; the verdict against PyTorch passes; the launch count is unchanged; the empty pick on `Plan::new` too.
#[test]
fn t219_the_race_pick_small_golden_step_is_bit_for_bit() {
    for (recipe, name) in [(Recipe::Sgd, "sgd"), (Recipe::AdamW513, "adamw_513")] {
        for (base, chosen) in [
            (Kind::Tiles3Tile, Kind::Tiles3Choice),
            (Kind::Tiles3BothTile, Kind::Tiles3BothChoice),
            (Kind::KeptTile, Kind::KeptChoiceEmpty),
        ] {
            let a = run_mic_plan(base, recipe, 10, Fault::None);
            let b = run_mic_plan(chosen, recipe, 10, Fault::None);
            assert_eq!(a.total_per_step, b.total_per_step);
            let on_variants: Vec<(Prog, usize)> = b
                .launches_per_step
                .iter()
                .filter(|e| e.0.race_index().is_some_and(|v| v > 0))
                .copied()
                .collect();
            if chosen == Kind::KeptChoiceEmpty {
                assert!(on_variants.is_empty());
                assert_eq!(a.launches_per_step, b.launches_per_step);
            } else {
                assert_eq!(on_variants.len(), 4, "{name} {chosen:?}: {on_variants:?}");
            }
            let v = ts::judge(recipe, b.total_per_step, &b.out, &refs(name));
            let s = same_bits(&a.out, &b.out);
            println!(
                "kt t219 {name} {chosen:?}: {} lansari, pe variante {on_variants:?}; fata de PyTorch {}; = {base:?} bit cu bit {} (SIMULARE)",
                b.total_per_step,
                if v.passes() { "TRECE" } else { "NU TRECE" },
                if s { "DA" } else { "NU" }
            );
            assert!(v.passes(), "{name} {chosen:?}: verdictul");
            assert!(s, "{name} {chosen:?}: nu e bit cu bit");
        }
    }
}

/// Prediction 3: the rules of the pick — a variant that changed the bits is never chosen, a gain under 3 % keeps the
/// witness, no witness row = no pick, rows of unknown variants or zero medians are ignored, ties go to the lower
/// index; `set` refuses an unknown variant, unknown flags, a zero side and a full table; a re-set replaces.
#[test]
fn t219_the_pick_rules() {
    let f = mv::Form {
        m: 1_024,
        n: 768,
        k: 3_072,
        flags: 0,
        count: 1,
    };
    let pick = |rows: &[RaceRow]| {
        MmChoice::from_race(rows)
            .unwrap()
            .of(f.m, f.n, f.k, f.flags)
    };
    // Changed bits: never, however fast.
    assert_eq!(pick(&[row(&f, 0, 100, true), row(&f, 3, 1, false)]), 0);
    // 2 % faster: the witness stays; 3 % faster: chosen.
    assert_eq!(pick(&[row(&f, 0, 1_000, true), row(&f, 2, 980, true)]), 0);
    assert_eq!(pick(&[row(&f, 0, 1_000, true), row(&f, 2, 970, true)]), 2);
    // No witness row (or the witness itself changed bits): nothing to compare against.
    assert_eq!(pick(&[row(&f, 1, 10, true)]), 0);
    assert_eq!(pick(&[row(&f, 0, 1_000, false), row(&f, 1, 10, true)]), 0);
    // Unknown variant and zero median ignored; a tie goes to the lower index.
    assert_eq!(
        pick(&[
            row(&f, 0, 1_000, true),
            row(&f, 9, 1, true),
            row(&f, 4, 0, true),
            row(&f, 4, 800, true),
            row(&f, 1, 800, true),
        ]),
        1
    );
    // Several runs of the same variant: its best median counts for the witness, the best row for the variant.
    assert_eq!(
        pick(&[
            row(&f, 0, 1_000, true),
            row(&f, 0, 900, true),
            row(&f, 2, 880, true),
        ]),
        0
    );
    let mut c = MmChoice::new();
    assert!(c.set(1, 1, 1, 0, VARIANTS.len()).is_err());
    assert!(c.set(1, 1, 1, 0x80, 1).is_err());
    assert!(c.set(0, 1, 1, 0, 1).is_err());
    c.set(64, 64, 64, 0, 3).unwrap();
    c.set(64, 64, 64, 0, 1).unwrap();
    assert_eq!(c.of(64, 64, 64, 0), 1);
    assert_eq!(c.programs(), &[Prog::MmTv128x64][..]);
    assert_eq!(c.of(64, 64, 64, 1), 0, "another flag = another shape");
    for i in 1..mv::MAX_CHOICE_FORMS as u32 {
        c.set(64 + i, 64, 64, 0, 2).unwrap();
    }
    assert!(c.set(9_999, 64, 64, 0, 2).is_err(), "the table is full");
    assert_eq!(c.programs(), &[Prog::MmTv128x64, Prog::MmTv64x128][..]);
    assert_eq!(c.entries().count(), mv::MAX_CHOICE_FORMS);
    // The programs: indices 25..=28, the right code behind each, every SHA-256 checked.
    ts::check_programs().unwrap();
    for (i, v) in VARIANTS.iter().enumerate() {
        let p = Prog::of_race_index(i).unwrap();
        assert_eq!(p.race_index(), Some(i));
        assert_eq!(
            (p.name(), p.sha256(), p.registers()),
            (v.name, v.sha256, v.registers)
        );
        assert_eq!(p.code(), v.code);
        assert_eq!(p.block(), v.block());
        assert_eq!(p.shared(), Some(v.shared()));
        assert_eq!(p.index(), if i == 0 { 23 } else { 24 + i });
    }
    assert_eq!(Prog::of_race_index(VARIANTS.len()), None);
    for (k, p) in PROGRAMS_ALL_SPLIT.iter().enumerate() {
        assert_eq!(p.index(), k, "the old indices stay");
    }
}

// ------------------------------------------------------------------ T220: `mm_train_tile_gelu` in the step

/// Prediction 3 (`docs/C-165_MM_TILE_GELU.md`): the #513 step (`new_tiles3`, resident, 2 × 512, both fusions,
/// `--mm-placi`) with [`Plan::with_mm_tile_gelu`] is the `with_mm_tile` list with only the 12 "proj-dX+gelu" launches
/// on `mm_train_tile_gelu` and its grid (blocks 3 072 → 768 each); same arena, same tags, layers, cbufs, order.
/// Without `with_fused_gelu` the option changes nothing.
#[test]
fn t220_the_513_step_with_tile_gelu_swaps_only_proj_dx() {
    let (bsz, tl) = (2usize, 512usize);
    let ar = train::M513;
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let base = || {
        Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
            .unwrap()
            .with_fused_back()
            .with_fused_gelu()
    };
    let old = base().with_mm_tile().resident().unwrap();
    let new = base().with_mm_tile_gelu().resident().unwrap();
    assert!(new.mm_tile_gelu() && new.mm_tile() && !old.mm_tile_gelu());
    assert!(new.clone().with_recompute().unwrap().mm_tile_gelu());
    assert_eq!(old.bytes(), new.bytes(), "arena");
    assert_eq!(
        new.mm_tile_gelu_programs(),
        &ts::PROGRAMS_MM_TILE_GELU_EXTRA[..]
    );
    assert!(old.mm_tile_gelu_programs().is_empty());
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    drop(params);
    let (lo, ln) = (
        old.step(Recipe::AdamW513, &opt, 1).unwrap(),
        new.step(Recipe::AdamW513, &opt, 1).unwrap(),
    );
    assert_eq!(lo.len(), ln.len());
    let (mut swapped, mut blocks_old, mut blocks_new) = (0usize, 0u64, 0u64);
    for (o, n) in lo.iter().zip(&ln) {
        assert_eq!(
            (o.tag, o.layer, o.cbuf),
            (n.tag, n.layer, n.cbuf),
            "{}",
            o.tag
        );
        if (o.prog, n.prog) == (Prog::MmTrainGelu, Prog::MmTrainTileGelu) {
            let a = mm_args(n);
            assert_eq!(n.grid, [a.n.div_ceil(64), a.m.div_ceil(64), 1], "{}", n.tag);
            blocks_old += o.grid.iter().map(|&g| g as u64).product::<u64>();
            blocks_new += n.grid.iter().map(|&g| g as u64).product::<u64>();
            swapped += 1;
        } else {
            assert_eq!((o.prog, o.grid), (n.prog, n.grid), "{}", o.tag);
        }
    }
    assert!(ln.iter().all(|l| l.prog != Prog::MmTrainGelu));
    let (lu, _) = new.step_until_opt(&opt, 1).unwrap();
    assert_eq!(&lu[..], &ln[..ln.len() - 1]);
    println!(
        "kt t220 pasul #513 2 x 512 (_3, ambele lipiri, rezident): {} lansari, {swapped} mm_train_gelu -> mm_train_tile_gelu; blocuri {blocks_old} -> {blocks_new}",
        ln.len()
    );
    assert_eq!(swapped, ar.layers);
    // 16×16 pieces of C per block before, 64×64 now (256 threads either way): 16 times fewer blocks.
    assert_eq!(blocks_old, 16 * blocks_new);
    // Without the GELU fusion: exactly `with_mm_tile`.
    let plain_old = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile();
    let plain_new = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile_gelu();
    assert!(plain_new.mm_tile_gelu_programs().is_empty());
    assert_eq!(
        plain_old.step(Recipe::AdamW513, &opt, 1).unwrap(),
        plain_new.step(Recipe::AdamW513, &opt, 1).unwrap()
    );
}

/// Prediction 4: the small golden step, 10 steps, both recipes — with `mm_train_tile_gelu` = without, **bit for bit**
/// (on `_3` + both fusions, on `Plan::new` + the GELU fusion, with the race pick, and alone = `with_mm_tile`); the
/// verdict against PyTorch passes; the launch count is unchanged; no `mm_train_gelu` launch is left.
#[test]
fn t220_tile_gelu_small_golden_step_is_bit_for_bit() {
    for (recipe, name) in [(Recipe::Sgd, "sgd"), (Recipe::AdamW513, "adamw_513")] {
        for (base, new_k, on_tile_gelu) in [
            (Kind::Tiles3BothTile, Kind::Tiles3BothTileGelu, MIC.layers),
            (Kind::KeptGelu, Kind::KeptGeluTileGelu, MIC.layers),
            (
                Kind::Tiles3BothChoice,
                Kind::Tiles3BothChoiceTileGelu,
                MIC.layers,
            ),
            (Kind::Tiles3Tile, Kind::Tiles3TileGeluAlone, 0),
        ] {
            let a = run_mic_plan(base, recipe, 10, Fault::None);
            let b = run_mic_plan(new_k, recipe, 10, Fault::None);
            assert_eq!(a.total_per_step, b.total_per_step);
            let n_tg = b
                .launches_per_step
                .iter()
                .filter(|e| e.0 == Prog::MmTrainTileGelu)
                .map(|e| e.1)
                .sum::<usize>();
            assert_eq!(n_tg, on_tile_gelu, "{name} {new_k:?}");
            assert!(b.launches_per_step.iter().all(|e| e.0 != Prog::MmTrainGelu));
            let v = ts::judge(recipe, b.total_per_step, &b.out, &refs(name));
            let s = same_bits(&a.out, &b.out);
            println!(
                "kt t220 {name} {new_k:?}: {} lansari ({n_tg} pe mm_train_tile_gelu); fata de PyTorch {}; = {base:?} bit cu bit {} (SIMULARE)",
                b.total_per_step,
                if v.passes() { "TRECE" } else { "NU TRECE" },
                if s { "DA" } else { "NU" }
            );
            assert!(v.passes(), "{name} {new_k:?}: verdictul");
            assert!(s, "{name} {new_k:?}: nu e bit cu bit");
        }
    }
}

// ------------------------------------------------------------------ T232: one SM step for the whole step (C-165)

/// T232d (`docs/C-165_TREAPTA_PAS.md` P3): the small golden step (SGD and AdamW, 3 steps) and 4 layers, packed with
/// one shared-memory step for every launch (the list's best step and 64 KiB), give the same bits as `pack`; the
/// command processor sees exactly one step. SIMULARE: the model has no occupancy — it only checks that the QMDs are
/// read correctly and the results do not change.
#[test]
fn t232d_one_sm_step_for_the_whole_step_same_bits() {
    let opt_list = |kind: Kind, ar: Arch| {
        let inputs = read_i32(&dir().join("intrari.i32"));
        let plan = match kind {
            Kind::Kept => Plan::new(ar, B, T, &inputs, HEAP).unwrap(),
            Kind::Tiles3TileSplit => Plan::new_tiles3(ar, B, T, &inputs, HEAP)
                .unwrap()
                .with_mm_tile()
                .with_colsum_split(),
            _ => unreachable!(),
        };
        let params: Vec<Vec<f32>> = ar
            .tensors()
            .iter()
            .map(|t| vec![0.0; t.elements()])
            .collect();
        let opt = AdamW::recipe_513(&params);
        plan.step(Recipe::AdamW513, &opt, 1).unwrap()
    };
    let four = Arch { layers: 4, ..MIC };
    let mut r = rng(232);
    let params4: Vec<Vec<f32>> = four
        .tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with("ln1.weight")
                || s.name.ends_with("ln2.weight")
                || s.name == "ln.weight"
            {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.3 * r()).collect()
            }
        })
        .collect();
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    for kind in [Kind::Kept, Kind::Tiles3TileSplit, Kind::Tiles3BothChoice] {
        let list = if kind == Kind::Tiles3BothChoice {
            None
        } else {
            Some(opt_list(kind, MIC))
        };
        let best = list.as_ref().map(|l| {
            (
                ts::step_best_sm_step(l, nvidia_ga104::occupancy::Reserve::None),
                ts::step_best_sm_step(l, nvidia_ga104::occupancy::Reserve::OneKib),
            )
        });
        let mut steps = vec![64 << 10];
        if let Some((r0, r1)) = best {
            println!(
                "kt t232d {kind:?}: treapta R0 {} KiB, R1 {} KiB",
                r0 >> 10,
                r1 >> 10
            );
            for s in [r0, r1] {
                if !steps.contains(&s) {
                    steps.push(s);
                }
            }
        }
        for (recipe, ar, params) in [
            (Recipe::Sgd, MIC, tensors("initiale")),
            (Recipe::AdamW513, MIC, tensors("initiale")),
            (Recipe::AdamW513, four, params4.clone()),
        ] {
            let run = || {
                run_plan(
                    kind,
                    ar,
                    (B, T),
                    &params,
                    (&inputs, &targets),
                    recipe,
                    3,
                    Fault::None,
                )
            };
            SM_STEP.with(|c| c.set(None));
            let base = run();
            for &step in &steps {
                SM_STEP.with(|c| c.set(Some(step)));
                SM_SEEN.with(|v| v.borrow_mut().clear());
                let one = run();
                SM_STEP.with(|c| c.set(None));
                let seen = SM_SEEN.with(|v| v.borrow().clone());
                let unit = step / 4096 + 1;
                assert_eq!(
                    seen,
                    vec![[unit, unit, 17]],
                    "{kind:?} {recipe:?} {} KiB",
                    step >> 10
                );
                assert_eq!(one.total_per_step, base.total_per_step);
                assert!(
                    same_bits(&base.out, &one.out),
                    "{kind:?} {recipe:?} {} straturi, {} KiB: alti biti",
                    ar.layers,
                    step >> 10
                );
            }
            println!(
                "kt t232d {kind:?} {recipe:?} {} straturi: aceiasi biti la {:?} KiB (SIMULARE)",
                ar.layers,
                steps.iter().map(|s| s >> 10).collect::<Vec<_>>()
            );
        }
    }
}
