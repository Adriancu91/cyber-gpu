//! C-157, the centre's note of 25 Nov 11:55, request 1 — T241, SIMULARE: the batch and the results of the **mixed fast**
//! training step through pinned RAM (`batch_ram::pack_mixed_fast_via_ram`: copies in, the `pack_mixed_fast` list, copies
//! out), alone and with **two slots** prepared while the other slot's step runs. Plan and predictions written before
//! the code (`docs/C-157_COPIERE_IN_PARALEL.md`, commit `cdec2d6`).
//!
//! Honesty: the simulator runs every submission to the end before returning, so "while the board runs step N" is
//! modelled by sending step N in two halves with the CPU's work for the other slot between them (as T204). The CPU
//! models of `mm_tc_pf` / `mm_tc_v4` are `mm_tc`'s, so "same bits" checks the **path** (addresses, copies, order,
//! picks), not the board. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS, LN_FWD_NEEDS,
    MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS, MM_TRAIN_GELU_NEEDS,
    MM_TRAIN_NEEDS, Needs, XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu,
    attn_back_tile0_on_cpu, attn_fwd_on_cpu, attn_fwd_tile_on_cpu, attn_p_on_cpu,
    attn_p_tile_on_cpu, colsum_on_cpu, copy_b128_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu,
    gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu, ln_fwd_on_cpu,
    mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu, mm_tc_v4_on_cpu, mm_train_gelu_on_cpu,
    mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::batch_ram::{self as br, ViaRam};
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::mm_tc_pf::MM_TC_PF;
use nvidia_ga104::mm_tc_v4::MM_TC_V4;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, LOSS_SCALE, LOSS_SCALE_513, PROGRAMS_ALL_FUSED2, Plan, Prog, SLOT, TcPick,
    TcPrograms,
};
use nvidia_ga104::transfer::{self, COPY_B128};
use nvidia_rm570::regs::clc7c0 as c;

const MINI: Arch = Arch {
    vocab: 16_384,
    ctx: 32,
    n_emb: 64,
    heads: 4,
    layers: 2,
};
const B: usize = 2;
const T: usize = 32;
const STEPS: usize = 6;
const LR: f64 = 1e-3;
const INIT_HALF_WIDTH: f64 = 0.034_641_016;
const STEP_64K: Option<u32> = Some(64 * 1024);

const VA: u64 = nvidia_ga104::mmu::VA_BASE;
/// Two control zones (one per slot) and two staging areas (one per slot; in the OS: pinned RAM).
const ZONES: [u64; 2] = [VA + 0x20_0000, VA + 0x28_0000];
const ZONE_MAX: u64 = 0x8_0000;
const REC: u64 = VA + 0x30_0000;
const PUSH: u64 = VA + 0x32_0000;
const STAGES: [u64; 2] = [VA + 0x34_0000, VA + 0x38_0000];
const COPY: u64 = VA + 0x3f_0000;
const PROGS: u64 = VA + 0x40_0000;
const TC: u64 = PROGS + 0x1_0000 * 23;
const TCG: u64 = PROGS + 0x1_0000 * 24;
const PF: u64 = PROGS + 0x1_0000 * 25;
const V4: u64 = PROGS + 0x1_0000 * 26;
const HEAP: u64 = VA + 0x100_0000;

const FAST: TcPrograms = TcPrograms {
    tc: TC,
    pf: PF,
    v4: V4,
    gelu: TCG,
};

fn prog_va(p: Prog) -> u64 {
    PROGS + 0x1_0000 * p.index() as u64
}

fn needs(s: nvidia_ga104::qmd::SharedMemory) -> Needs {
    Needs {
        shared_bytes: s.bytes,
        barriers: u32::from(s.barriers),
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn initial() -> Vec<Vec<f32>> {
    let mut g = Rng(0x00c1_9e10_2026_1125);
    MINI.tensors()
        .iter()
        .map(|t| {
            let n = t.elements();
            let leaf = t.name.rsplit('.').nth(1).unwrap_or("");
            if t.name.ends_with("bias") {
                vec![0.0f32; n]
            } else if leaf.starts_with("ln") {
                vec![1.0f32; n]
            } else {
                (0..n)
                    .map(|_| ((g.unit() * 2.0 - 1.0) * INIT_HALF_WIDTH) as f32)
                    .collect()
            }
        })
        .collect()
}

/// `STEPS` different batches (each its own token segment lists).
fn batches() -> Vec<(Vec<u32>, Vec<u32>)> {
    let mut g = Rng(0x5eed_0000_0000_0241);
    (0..STEPS)
        .map(|_| {
            let (mut inp, mut tgt) = (Vec::new(), Vec::new());
            for _ in 0..B {
                let row: Vec<u32> = (0..=T)
                    .map(|_| (g.next() % MINI.vocab as u64) as u32)
                    .collect();
                inp.extend_from_slice(&row[..T]);
                tgt.extend_from_slice(&row[1..]);
            }
            (inp, tgt)
        })
        .collect()
}

fn board() -> CommandProcessor {
    ts::check_programs().expect("SHA-256 programe");
    let mut cp = CommandProcessor::new();
    for p in PROGRAMS_ALL_FUSED2 {
        let code = p.code();
        let sh = || needs(p.shared().unwrap());
        match p {
            Prog::MmTrain => cp.register_program_with_needs(code, mm_train_on_cpu, MM_TRAIN_NEEDS),
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
            Prog::AttnFwdTile2 | Prog::AttnFwdTile3 => {
                cp.register_program_with_needs(code, attn_fwd_tile_on_cpu, sh())
            }
            Prog::AttnBackTile0_2 | Prog::AttnBackTile0_3 => {
                cp.register_program_with_needs(code, attn_back_tile0_on_cpu, sh())
            }
            Prog::AttnBackTile2 | Prog::AttnBackTile3 => {
                cp.register_program_with_needs(code, attn_back_tile_on_cpu, sh())
            }
            Prog::AttnPTile => cp.register_program_with_needs(code, attn_p_tile_on_cpu, sh()),
            Prog::LnBackRes => {
                cp.register_program_with_needs(code, ln_back_res_on_cpu, LN_BACK_RES_NEEDS)
            }
            Prog::MmTrainGelu => {
                cp.register_program_with_needs(code, mm_train_gelu_on_cpu, MM_TRAIN_GELU_NEEDS)
            }
            Prog::MmTrainTile
            | Prog::ColsumPart
            | Prog::MmTv128x64
            | Prog::MmTv64x128
            | Prog::MmTv128x128
            | Prog::MmTv64x64K8
            | Prog::MmTrainTileGelu => unreachable!("nu e in PROGRAMS_ALL_FUSED2"),
        };
        cp.mem.write(prog_va(p), code);
    }
    cp.register_program_with_needs(MM_TC, mm_tc_on_cpu, MM_TC_NEEDS);
    cp.mem.write(TC, MM_TC);
    cp.register_program_with_needs(MM_TC_GELU, mm_tc_gelu_on_cpu, MM_TC_GELU_NEEDS);
    cp.mem.write(TCG, MM_TC_GELU);
    cp.register_program_with_needs(MM_TC_PF, mm_tc_pf_on_cpu, MM_TC_PF_NEEDS);
    cp.mem.write(PF, MM_TC_PF);
    cp.register_program_with_needs(MM_TC_V4, mm_tc_v4_on_cpu, MM_TC_V4_NEEDS);
    cp.mem.write(V4, MM_TC_V4);
    cp.register_program(COPY_B128, copy_b128_on_cpu);
    cp.mem.write(COPY, COPY_B128);
    cp
}

/// One submission of `qmds` (a part of a step's zone, already in memory) on the simulated board.
fn submit(cp: &mut CommandProcessor, first: &mut u32, qmds: &[u64]) {
    let n = qmds.len();
    cp.mem.write(REC, &vec![0u8; 16 * (n + 1)]);
    let mut w = vec![0u32; 2 + transfer::sequence_words_fresh(n)];
    let mut pb = PushBuf::new(&mut w);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    transfer::encode_sequence_fresh(&mut pb, qmds, REC, *first).unwrap();
    let nw = pb.len();
    let bytes: Vec<u8> = w[..nw].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, nw as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
    let last = first.wrapping_add(n as u32);
    assert_eq!(
        cp.mem.read_u32(REC + 16 * n as u64).unwrap(),
        last,
        "placa nu a terminat"
    );
    *first = last.wrapping_add(1);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Way {
    /// Today's OS path: the batch written and the results read through the VRAM window, `pack_mixed_fast`.
    Window,
    /// The batch and the results through the staging area, one slot, step after step.
    Ram,
    /// Two slots; step N sent in two halves, between them the results of step N−1 read and step N+1 prepared in the
    /// slot step N−1 used.
    Parallel,
    /// Plant: the same order as `Parallel` with one slot (one staging area, one zone).
    ParallelOneSlot,
}

#[derive(Debug, PartialEq, Eq)]
struct Run {
    losses: Vec<u32>,
    norms: Vec<u32>,
    weights: Vec<u32>,
    /// Words of the batch written through the window, per step (the first batch comes with `uploads`).
    batch_window_words: Vec<usize>,
    /// Launches per step and control zone bytes of step 1.
    launches: usize,
    zone1: usize,
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

struct Pending {
    qmds: Vec<u64>,
    via: Option<ViaRam>,
    plan: Plan,
    slot: usize,
}

fn run(way: Way) -> Run {
    let data = batches();
    let params = initial();
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let plan0 = Plan::new_tiles3(MINI, B, T, &data[0].0, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    let plan0 = if way == Way::Parallel {
        plan0.with_batch_slots().unwrap()
    } else {
        plan0
    };
    let mut cp = board();
    cp.mem.write(HEAP, &vec![0u8; plan0.bytes() as usize]);
    for (va, b) in plan0.uploads(&params, &data[0].0, &data[0].1).unwrap() {
        cp.mem.write(va, &b);
    }
    for (va, b) in plan0.mixed_uploads(LOSS_SCALE).unwrap() {
        cp.mem.write(va, &b);
    }
    if way == Way::Parallel {
        let p1 = plan0.slot(1).unwrap();
        for (va, b) in p1.slot_uploads() {
            cp.mem.write(va, &b);
        }
        for (va, b) in p1.mixed_uploads(LOSS_SCALE).unwrap() {
            cp.mem.write(va, &b);
        }
    }
    let slots = if way == Way::Parallel { 2 } else { 1 };
    let mut out = Run {
        losses: Vec::new(),
        norms: Vec::new(),
        weights: Vec::new(),
        batch_window_words: Vec::new(),
        launches: 0,
        zone1: 0,
    };
    // The CPU's work for step k (0-based) before it is sent.
    let prepare = |cp: &mut CommandProcessor, out: &mut Run, k: usize| -> Pending {
        let slot = k % slots;
        let (x, y) = &data[k];
        let base = if slots == 2 {
            plan0.slot(slot).unwrap()
        } else {
            plan0.clone()
        };
        let plan = base.rebatch(x).unwrap();
        let list = plan
            .step_mixed(Recipe::AdamW513, &opt, k as u32 + 1, LOSS_SCALE)
            .unwrap();
        let (zone, qmds, via) = if way == Way::Window {
            let mut words = 0;
            if k > 0 {
                for (va, b) in plan.batch_uploads(x, y).unwrap() {
                    words += b.len() / 4;
                    cp.mem.write(va, &b);
                }
            }
            out.batch_window_words.push(words);
            let (z, q, _) =
                ts::pack_mixed_fast(&list, prog_va, &FAST, STEP_64K, ZONES[slot]).unwrap();
            (z, q, None)
        } else {
            let v = br::via_ram(&plan, x, y, STAGES[slot]).unwrap();
            cp.mem.write(STAGES[slot], &v.staging);
            out.batch_window_words.push(0);
            let (z, q, _) =
                br::pack_mixed_fast_via_ram(&list, prog_va, &FAST, STEP_64K, COPY, ZONES[slot], &v)
                    .unwrap();
            (z, q, Some(v))
        };
        assert!(zone.len() as u64 <= ZONE_MAX, "zona de control prea mare");
        if k == 0 {
            out.launches = qmds.len();
            out.zone1 = zone.len();
        }
        assert_eq!(qmds.len(), out.launches);
        cp.mem.write(ZONES[slot], &zone);
        Pending {
            qmds,
            via,
            plan,
            slot,
        }
    };
    let results = |cp: &mut CommandProcessor, out: &mut Run, p: &Pending| {
        let (rows, norm) = match &p.via {
            None => {
                let [(lr, ln), (nr, nn)] = p.plan.step_reads();
                (
                    cp.mem.read_f32s(lr, (ln / 4) as usize).unwrap(),
                    cp.mem.read_f32s(nr, (nn / 4) as usize).unwrap()[0],
                )
            }
            Some(v) => {
                let mut st = vec![0u8; v.staging.len()];
                cp.mem.read(STAGES[p.slot], &mut st).unwrap();
                br::results(&st, v).unwrap()
            }
        };
        out.losses.push(ts::Outcome::mean_loss(&rows).to_bits());
        out.norms.push(norm.to_bits());
    };
    let mut first = 0x7000u32;
    match way {
        Way::Window | Way::Ram => {
            for k in 0..STEPS {
                let p = prepare(&mut cp, &mut out, k);
                submit(&mut cp, &mut first, &p.qmds);
                results(&mut cp, &mut out, &p);
            }
        }
        Way::Parallel | Way::ParallelOneSlot => {
            let mut prev: Option<Pending> = None;
            let mut cur = prepare(&mut cp, &mut out, 0);
            for k in 0..STEPS {
                let h = cur.qmds.len() / 2;
                submit(&mut cp, &mut first, &cur.qmds[..h]);
                // "While the board runs step k": the results of step k−1, then step k+1 in the slot k−1 used.
                if let Some(p) = prev.take() {
                    results(&mut cp, &mut out, &p);
                }
                let next = (k + 1 < STEPS).then(|| prepare(&mut cp, &mut out, k + 1));
                submit(&mut cp, &mut first, &cur.qmds[h..]);
                prev = Some(cur);
                match next {
                    Some(n) => cur = n,
                    None => break,
                }
            }
            if let Some(p) = prev.take() {
                results(&mut cp, &mut out, &p);
            }
        }
    }
    out.weights = bits(&cp.mem.read_f32s(plan0.w, plan0.n_params).unwrap());
    out
}

/// The `Ram` run, shared by Q1–Q3 (each run is ~2 min in a debug build).
fn ram() -> &'static Run {
    static R: std::sync::OnceLock<Run> = std::sync::OnceLock::new();
    R.get_or_init(|| run(Way::Ram))
}

fn same_numbers(a: &Run, b: &Run) -> bool {
    a.losses == b.losses && a.norms == b.norms && a.weights == b.weights
}

/// Q1: through RAM = through the window, bit for bit; 7 copy launches more; no batch word through the window.
#[test]
fn t241_q1_via_ram_mixed_fast_same_bits() {
    let w = run(Way::Window);
    let r = ram();
    assert_eq!(w.losses.len(), STEPS);
    assert!(w.losses.iter().all(|x| f32::from_bits(*x).is_finite()));
    assert!(same_numbers(&w, r), "prin RAM != prin fereastra");
    assert_eq!(r.launches, w.launches + 7);
    assert_eq!(r.zone1, w.zone1 + 7 * SLOT as usize);
    assert!(w.batch_window_words[1..].iter().all(|&n| n > 0));
    assert!(r.batch_window_words.iter().all(|&n| n == 0));
    println!(
        "kt c241 q1 SIMULARE: {STEPS} pasi mixti rapizi (64 KiB), prin RAM = prin fereastra bit cu bit; lansari {} -> {} (+7 copii); cuvinte ale lotului prin fereastra pe pas {} -> 0",
        w.launches, r.launches, w.batch_window_words[1]
    );
}

/// Q2: two slots, step N+1 prepared while step N runs, results of N−1 read then = step after step, bit for bit.
#[test]
fn t241_q2_two_slots_in_parallel_same_bits() {
    let r = ram();
    let p = run(Way::Parallel);
    assert!(same_numbers(r, &p), "in paralel != pas cu pas");
    assert_eq!(p.launches, r.launches);
    println!(
        "kt c241 q2 SIMULARE: doua locuri, pasul N+1 pregatit intre cele doua bucati ale pasului N = pas cu pas bit cu bit ({STEPS} pasi)"
    );
}

/// Q3: one slot in the parallel order is caught; misalignments and an unloaded `mm_tc*` refused before sending.
#[test]
fn t241_q3_planted_mistakes_caught() {
    let r = ram();
    let one = run(Way::ParallelOneSlot);
    assert_ne!(
        r.losses, one.losses,
        "un singur loc trebuia sa strice pierderile"
    );
    assert_ne!(
        r.weights, one.weights,
        "un singur loc trebuia sa strice greutatile"
    );
    let bad_losses = r
        .losses
        .iter()
        .zip(&one.losses)
        .filter(|(a, b)| a != b)
        .count();
    let data = batches();
    let plan = Plan::new_tiles3(MINI, B, T, &data[0].0, HEAP)
        .unwrap()
        .resident()
        .unwrap();
    let (x, y) = &data[0];
    assert!(br::via_ram(&plan, x, y, STAGES[0] + 16).is_err());
    let v = br::via_ram(&plan, x, y, STAGES[0]).unwrap();
    let params = initial();
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let list = plan
        .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE)
        .unwrap();
    assert!(
        br::pack_mixed_fast_via_ram(&list, prog_va, &FAST, STEP_64K, COPY + 16, ZONES[0], &v)
            .is_err()
    );
    assert!(
        br::pack_mixed_fast_via_ram(&list, prog_va, &FAST, STEP_64K, COPY, ZONES[0] + 16, &v)
            .is_err()
    );
    let none = TcPrograms::default();
    assert!(
        br::pack_mixed_fast_via_ram(&list, prog_va, &none, STEP_64K, COPY, ZONES[0], &v).is_err()
    );
    assert!(
        br::pack_mixed_fast_via_ram(&list, prog_va, &FAST, Some(12 * 1024), COPY, ZONES[0], &v)
            .is_err()
    );
    // With no step the copies are `pack_via_ram`'s bytes (the list differs: mixed, not F32).
    let (z, q, picks) =
        br::pack_mixed_fast_via_ram(&list, prog_va, &FAST, None, COPY, ZONES[0], &v).unwrap();
    let (zf, qf, pf) =
        ts::pack_mixed_fast(&list, prog_va, &FAST, None, ZONES[0] + 5 * SLOT).unwrap();
    assert_eq!(picks, pf);
    assert_eq!(&q[5..5 + qf.len()], &qf[..]);
    assert_eq!(&z[5 * SLOT as usize..5 * SLOT as usize + zf.len()], &zf[..]);
    println!(
        "kt c241 q3 SIMULARE: un singur loc in ordinea in paralel -> {bad_losses}/{STEPS} pierderi alti biti, greutatile alti biti; nealinieri, mm_tc neincarcat si treapta 12 KiB refuzate"
    );
}

/// Q4: #513's bytes per step (from `Plan::traffic`) and the mixed list through RAM: 147 on `mm_tc_v4`, 7 copies more.
#[test]
fn t241_q4_plan_513_bytes_and_copies() {
    let m = train::M513;
    let base = VA + (1 << 30);
    let expect = [
        (2usize, 20_484u64, 4_100u64),
        (6, 61_444, 12_292),
        (8, 81_924, 16_388),
    ];
    for (bsz, up, down) in expect {
        let n = bsz * 512;
        let inputs: Vec<u32> = (0..n as u32).map(|i| (i * 7919) % 16_384).collect();
        let plan = Plan::new_tiles3(m, bsz, 512, &inputs, base)
            .unwrap()
            .resident()
            .unwrap();
        let tr = plan.traffic();
        assert_eq!((tr.step_up, tr.step_down), (up, down), "lot {bsz}");
        let ms_window = (up / 4) as f64 * 0.76e-3 + (down / 4) as f64 * 1.1e-3;
        let us_link = (up + down) as f64 / 5_970.0;
        println!(
            "kt c241 q4 #513 lot {bsz}: sus {up} B ({} cuvinte), jos {down} B -> DEDUS prin fereastra ~{ms_window:.1} ms, pe legatura PCIe ~{us_link:.0} us",
            up / 4
        );
        if bsz == 2 {
            let targets: Vec<u32> = inputs.iter().map(|t| (t + 1) % 16_384).collect();
            let plan = plan.with_mm_tile().with_colsum_split();
            let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
            let list = plan
                .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
                .unwrap();
            let v = br::via_ram(&plan, &inputs, &targets, STAGES[0]).unwrap();
            let (_, q, picks) =
                br::pack_mixed_fast_via_ram(&list, prog_va, &FAST, STEP_64K, COPY, ZONES[0], &v)
                    .unwrap();
            let v4 = picks.iter().filter(|p| **p == Some(TcPick::V4)).count();
            assert_eq!(v4, 147);
            assert_eq!(q.len(), list.len() + 7);
            println!(
                "kt c241 q4 #513 lot 2 SIMULARE: {} lansari mixte + 7 copii, {v4} pe mm_tc_v4, tranzitul {} B",
                list.len(),
                v.staging.len()
            );
        }
    }
}
