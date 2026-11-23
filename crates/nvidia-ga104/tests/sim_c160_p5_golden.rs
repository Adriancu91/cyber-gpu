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
    CommandProcessor, Event, GNORM_NEEDS, LN_BACK_NEEDS, LN_FWD_NEEDS, MM_TRAIN_NEEDS, Needs,
    XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu, attn_back_tile0_on_cpu, attn_fwd_on_cpu,
    attn_fwd_tile_on_cpu, attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, emb_back_on_cpu,
    emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_fwd_on_cpu, mm_train_on_cpu,
    opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, Launch, Outcome, PROGRAMS_ALL, PROGRAMS_ALL_3, Plan, Prog, Refs,
};
use nvidia_ga104::{transfer, xent as xe};
use nvidia_rm570::regs::clc7c0 as c;

/// The small golden model (hub `integrare/kronos/pas_de_aur_mic/reteta.json`).
const MIC: Arch = train::MIC;
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
        ts::check_programs().expect("SHA-256 programe");
        let mut cp = CommandProcessor::new();
        // All 18 (the 13 of every plan + `attn_p`, used only by `Plan::new_p_only`, + the four tiled ones of
        // `Plan::new_tiles` / `new_p_only_tiles`).
        // T195: + the three `_3` programs (`PROGRAMS_ALL_3`), on the same CPU models as the first tiled ones.
        for p in PROGRAMS_ALL_3 {
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
        let (mut zone, qmds) = ts::pack(launches, prog_va, ZONE).unwrap();
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
}

fn plant(plan: &Plan, list: &mut Vec<Launch>, f: Fault) {
    match f {
        Fault::None => {}
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

fn run_with(
    ar: Arch,
    shape: (usize, usize),
    params: &[Vec<f32>],
    data: (&[u32], &[u32]),
    recipe: Recipe,
    steps: u32,
    fault: Fault,
) -> Run {
    run_plan(Kind::Kept, ar, shape, params, data, recipe, steps, fault)
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
    };
    // T187: the arena of `Plan::new` (the same addresses as `new_tiles`), for `Fault::OldBackward`.
    let old = Plan::new(ar, bsz, tl, inputs, HEAP).unwrap();
    let mut b = Board::new();
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
    let ar = Arch {
        layers,
        ..train::M513
    };
    let (bsz, tl) = (1usize, 64usize);
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
    let toks: Vec<u32> = (0..=tl).map(|_| ((r() + 1.0) * 8_191.9) as u32).collect();
    let (inputs, targets) = (toks[..tl].to_vec(), toks[1..].to_vec());
    let run = run_with(
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
        "kt c165 formele lui #513 ({layers} straturi, greutati inventate, 1 x 64): {} lansari, memoria placii {} MiB; fata de oracol: eroarea {loss} rel {lrel:.2e}, norma rel {nrel:.2e}, grad_abs {:.2e} ({}) grad_rel_norma {:.2e}, w1_abs {:.2e} ({})",
        run.total_per_step,
        run.board_bytes >> 20,
        g.abs,
        names[g.at].name,
        g.rel_norm,
        w.abs,
        names[w.at].name
    );
    assert_eq!(run.total_per_step, 31 * layers + 13);
    assert!(lrel <= train::TOL_LOSS_REL, "eroarea {lrel}");
    assert!(nrel <= 1e-5, "norma {nrel}");
    assert!(g.abs <= train::TOL_GRAD_ABS && g.rel_norm <= train::TOL_GRAD_REL_NORM);
    assert!(w.abs <= train::TOL_ADAMW_W_ABS);
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
