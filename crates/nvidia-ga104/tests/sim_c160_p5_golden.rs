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
    CommandProcessor, Event, GNORM_NEEDS, LN_BACK_NEEDS, LN_FWD_NEEDS, MM_TRAIN_NEEDS, XENT_NEEDS,
    attn_back_on_cpu, attn_fwd_on_cpu, colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu,
    gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_fwd_on_cpu, mm_train_on_cpu, opt_step_on_cpu,
    vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{self as ts, Launch, Outcome, PROGRAMS, Plan, Prog, Refs};
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
        for p in PROGRAMS {
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
        let (zone, qmds) = ts::pack(launches, prog_va, ZONE).unwrap();
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
            .unwrap_or_else(|err| panic!("pasul: {err:?}"));
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
    (bsz, tl): (usize, usize),
    params: &[Vec<f32>],
    (inputs, targets): (&[u32], &[u32]),
    recipe: Recipe,
    steps: u32,
    fault: Fault,
) -> Run {
    let plan = Plan::new(ar, bsz, tl, inputs, HEAP).unwrap();
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
        plant(&plan, &mut list, fault);
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
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    run_with(
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
