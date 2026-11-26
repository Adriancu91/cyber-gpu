//! C-157 — T253, SIMULARE: `attn_back_tile_tc` (dQ, dK, dV of the attention on the tensor cores, f16 inputs) as a
//! CPU stand-in in `gpu-sim` and as an option of the **fast mixed step** (`train_step::attn_tc_list` +
//! `train_step::pack_mixed_fast_attn`). Plan and predictions C1–C6 written before these tests
//! (`docs/C-157_ATTN_BACK_TC.md` §5, commit `408b73f`).
//!
//! Honesty: the stand-in is `gpu_compute::attn_tc::model`, a HYPOTHESIS of what the program computes (the
//! tensor-core adder fitted in os-335). Nothing here runs on the RTX 3070 Ti; block counts are DEDUS.

use gpu_compute::layer::AttnShape;
use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC_NEEDS, ATTN_BACK_TILE_TC16_NEEDS, CommandProcessor, GNORM_NEEDS,
    LN_BACK_NEEDS, LN_BACK_RES_NEEDS, LN_FWD_NEEDS, Launch as SimLaunch, MM_TC_GELU_NEEDS,
    MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS, MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, Needs,
    XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu, attn_back_tile_tc_on_cpu,
    attn_back_tile_tc16_on_cpu, attn_back_tile0_on_cpu, attn_fwd_on_cpu, attn_fwd_tile_on_cpu,
    attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu,
    gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu, ln_fwd_on_cpu,
    mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu, mm_tc_v4_on_cpu, mm_train_gelu_on_cpu,
    mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::attn_back::{Args as AbArgs, Mode};
use nvidia_ga104::attn_back_tile_tc::{self as atc, ATTN_BACK_TILE_TC};
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::mm_tc_pf::MM_TC_PF;
use nvidia_ga104::mm_tc_v4::MM_TC_V4;
use nvidia_ga104::occupancy::{self, Kernel, Reserve, SM86};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, AttnTcProgram, LOSS_SCALE, LOSS_SCALE_513, MixedLaunch, PROGRAMS_ALL_FUSED2, Plan,
    Prog, TcPick, TcPrograms,
};
use nvidia_ga104::transfer;
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
const STEPS: u32 = 5;
const LR: f64 = 1e-3;
const INIT_HALF_WIDTH: f64 = 0.034_641_016;
const KIB: u32 = 1024;

const VA: u64 = nvidia_ga104::mmu::VA_BASE;
const ZONE: u64 = VA + 0x20_0000;
const REC: u64 = VA + 0x30_0000;
const PUSH: u64 = VA + 0x32_0000;
const PROGS: u64 = VA + 0x40_0000;
const TC: u64 = PROGS + 0x1_0000 * 23;
const TCG: u64 = PROGS + 0x1_0000 * 24;
const PF: u64 = PROGS + 0x1_0000 * 25;
const V4: u64 = PROGS + 0x1_0000 * 26;
const ATC: u64 = PROGS + 0x1_0000 * 27;
const ATC16: u64 = PROGS + 0x1_0000 * 28;
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

fn batch() -> (Vec<u32>, Vec<u32>) {
    let mut g = Rng(0x5eed_0000_0000_0240);
    let (mut inp, mut tgt) = (Vec::new(), Vec::new());
    for _ in 0..B {
        let row: Vec<u32> = (0..=T)
            .map(|_| (g.next() % MINI.vocab as u64) as u32)
            .collect();
        inp.extend_from_slice(&row[..T]);
        tgt.extend_from_slice(&row[1..]);
    }
    (inp, tgt)
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
    cp.register_program_with_needs(
        ATTN_BACK_TILE_TC,
        attn_back_tile_tc_on_cpu,
        ATTN_BACK_TILE_TC_NEEDS,
    );
    cp.mem.write(ATC, ATTN_BACK_TILE_TC);
    cp.register_program_with_needs(
        ATTN_BACK_TILE_TC16,
        attn_back_tile_tc16_on_cpu,
        ATTN_BACK_TILE_TC16_NEEDS,
    );
    cp.mem.write(ATC16, ATTN_BACK_TILE_TC16);
    cp
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug, PartialEq)]
enum Layout {
    Simple,
    Tiles3,
    Tiles3Both,
}

fn plan_of(layout: Layout, inputs: &[u32]) -> Plan {
    match layout {
        Layout::Simple => Plan::new(MINI, B, T, inputs, HEAP),
        Layout::Tiles3 => Plan::new_tiles3(MINI, B, T, inputs, HEAP),
        Layout::Tiles3Both => Plan::new_tiles3(MINI, B, T, inputs, HEAP)
            .map(|p| p.with_fused_back().with_fused_gelu()),
    }
    .unwrap()
}

fn submit(
    cp: &mut CommandProcessor,
    first: &mut u32,
    zone: &[u8],
    qmds: &[u64],
) -> Result<(), String> {
    assert!(zone.len() as u64 <= REC - ZONE, "zona de control prea mare");
    cp.mem.write(ZONE, zone);
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
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
        .map_err(|err| format!("{err:?}"))?;
    *first = first.wrapping_add(n as u32 + 1);
    Ok(())
}

fn mini_list(layout: Layout) -> Vec<MixedLaunch> {
    let (inputs, _) = batch();
    let params = initial();
    let plan = plan_of(layout, &inputs);
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    plan.step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE)
        .unwrap()
}

fn count(picks: &[Option<TcPick>], p: TcPick) -> usize {
    picks.iter().filter(|x| **x == Some(p)).count()
}

/// What one run leaves: the losses of each step, the step-1 gradient, the last weights.
struct Out {
    losses: Vec<Vec<f32>>,
    grad1: Vec<f32>,
    weights: Vec<f32>,
}

/// `STEPS` mixed steps on the simulated board, packed by `pack_mixed_fast_attn` at the 64 KiB step; with `attn` the
/// list goes through `attn_tc_list` first; `edit` may change the list after that (plants).
fn run(
    layout: Layout,
    attn: Option<AttnTcProgram>,
    edit: &dyn Fn(&mut Vec<MixedLaunch>),
) -> Result<(Out, Vec<Option<TcPick>>), String> {
    run_zone(layout, attn, edit, &|_, _, _| {})
}

/// [`run`] with `zone_edit` applied to the packed control zone (zone, QMD addresses, picks) before each submission.
#[allow(clippy::type_complexity)]
fn run_zone(
    layout: Layout,
    attn: Option<AttnTcProgram>,
    edit: &dyn Fn(&mut Vec<MixedLaunch>),
    zone_edit: &dyn Fn(&mut Vec<u8>, &[u64], &[Option<TcPick>]),
) -> Result<(Out, Vec<Option<TcPick>>), String> {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = plan_of(layout, &inputs);
    let mut cp = board();
    for (va, b) in plan.uploads(&params, &inputs, &targets).unwrap() {
        cp.mem.write(va, &b);
    }
    for (va, b) in plan.mixed_uploads(LOSS_SCALE).unwrap() {
        cp.mem.write(va, &b);
    }
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let mut first = 0x7000u32;
    let mut out = Out {
        losses: Vec::new(),
        grad1: Vec::new(),
        weights: Vec::new(),
    };
    let mut picks1 = Vec::new();
    for step in 1..=STEPS {
        let mut list = plan
            .step_mixed(Recipe::AdamW513, &opt, step, LOSS_SCALE)
            .unwrap();
        if attn.is_some() {
            list = ts::attn_tc_list(list).map_err(|e| format!("{e:?}"))?;
        }
        edit(&mut list);
        let (mut zone, qmds, p) =
            ts::pack_mixed_fast_attn(&list, prog_va, &FAST, attn, Some(64 * KIB), ZONE)
                .map_err(|e| format!("{e:?}"))?;
        zone_edit(&mut zone, &qmds, &p);
        if step == 1 {
            picks1 = p;
        }
        submit(&mut cp, &mut first, &zone, &qmds).map_err(|e| format!("pasul {step}: {e}"))?;
        out.losses
            .push(cp.mem.read_f32s(plan.loss_rows, B * T).unwrap());
        if step == 1 {
            out.grad1 = cp.mem.read_f32s(plan.g, plan.n_params).unwrap();
        }
    }
    out.weights = cp.mem.read_f32s(plan.w, plan.n_params).unwrap();
    Ok((out, picks1))
}

fn rel_norm(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    let (mut d, mut n) = (0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        d += (x as f64 - y as f64).powi(2);
        n += (y as f64).powi(2);
    }
    (d / n).sqrt()
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}

fn lcg(seed: u64) -> impl FnMut() -> f32 {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    move || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((x >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

const QKV: u64 = VA + 0x200_0000;
const P: u64 = VA + 0x300_0000;
const DOUT: u64 = VA + 0x400_0000;
const DS: u64 = VA + 0x500_0000;
const DQKV: u64 = VA + 0x600_0000;
const CB: u64 = VA + 0x700_0000;
const POISON: f32 = -7.25;

/// One launch of the stand-in on (batch, t, c, heads) with grid `grid`; returns dqkv (poisoned before) and the model.
fn stand_in(
    s: AttnShape,
    grid: [u32; 3],
    seed: u64,
    ds_put: Option<(usize, f32)>,
    mode: Mode,
) -> (Result<(), String>, Vec<f32>, Option<Vec<f32>>) {
    let (nq, no, _) = s.sizes().unwrap();
    let mut r = lcg(seed);
    let scale = AbArgs::scale_for((s.c / s.heads) as u32);
    let qkv: Vec<f32> = (0..nq).map(|_| 1.7 * r()).collect();
    let dout: Vec<f32> = (0..no).map(|_| 0.16 * r()).collect();
    let (p, _) = gpu_compute::layer::attn_fwd_gpu(&s, &qkv, scale).unwrap();
    let (mut ds, _) = gpu_compute::layer::attn_back_gpu(&s, &qkv, &p, &dout, scale).unwrap();
    if let Some((i, v)) = ds_put {
        ds[i] = v;
    }
    let want = gpu_compute::attn_tc::model(&s, &qkv, &p, &dout, &ds);
    let mut cp = CommandProcessor::new();
    cp.mem.write_f32s(QKV, &qkv);
    cp.mem.write_f32s(P, &p);
    cp.mem.write_f32s(DOUT, &dout);
    cp.mem.write_f32s(DS, &ds);
    cp.mem.write_f32s(DQKV, &vec![POISON; nq]);
    let a = AbArgs {
        qkv: QKV,
        p: P,
        dout: DOUT,
        ds: DS,
        dqkv: DQKV,
        batch: s.batch as u32,
        t: s.t as u32,
        c: s.c as u32,
        heads: s.heads as u32,
        scale,
        mode: Mode::Columns,
    };
    let mut cb = atc::cbuf(&a).unwrap();
    if mode == Mode::Rows {
        cb[0x738..0x73c].copy_from_slice(&0u32.to_le_bytes());
    }
    cp.mem.write(CB, &cb);
    let mut l = SimLaunch {
        mem: &mut cp.mem,
        cbuf0: CB,
        grid,
        block: atc::BLOCK,
    };
    let res = attn_back_tile_tc_on_cpu(&mut l).map_err(|e| format!("{e:?}"));
    (res, cp.mem.read_f32s(DQKV, nq).unwrap(), want)
}

/// C1: the stand-in writes `attn_tc::model` bit for bit (only the rows / heads of the grid), nothing in mode 0, and
/// refuses d > 64 and an input past f16.
#[test]
fn t253_c1_stand_in_is_the_model() {
    let mut elems = 0usize;
    for (b, t, c, h, grid) in [
        (1usize, 64usize, 64usize, 1usize, None),
        (2, 100, 96, 2, None),
        (1, 130, 40, 1, None),
        (2, 192, 128, 2, Some([2u32, 1u32, 2u32])), // part of the grid: rows < 128, head 0
        (1, 17, 24, 3, None),
    ] {
        let s = AttnShape {
            batch: b,
            t,
            c,
            heads: h,
        };
        let full = [t.div_ceil(64) as u32, h as u32, b as u32];
        let g = grid.unwrap_or(full);
        let (res, got, want) = stand_in(s, g, (t * 31 + c) as u64, None, Mode::Columns);
        res.unwrap();
        let want = want.unwrap();
        let d = c / h;
        for bb in 0..b {
            for tp in 0..t {
                for col in 0..3 * c {
                    let at = (bb * t + tp) * 3 * c + col;
                    let head = (col % c) / d;
                    let inside = bb < g[2] as usize
                        && tp < (64 * g[0] as usize).min(t)
                        && head < g[1] as usize;
                    if inside {
                        assert_eq!(
                            got[at].to_bits(),
                            want[at].to_bits(),
                            "{b}x{t}x{c}/{h} la {at}"
                        );
                        elems += 1;
                    } else {
                        assert_eq!(
                            got[at], POISON,
                            "{b}x{t}x{c}/{h}: scris in afara grilei la {at}"
                        );
                    }
                }
            }
        }
    }
    let s = AttnShape {
        batch: 1,
        t: 64,
        c: 64,
        heads: 1,
    };
    let (res, got, _) = stand_in(s, [1, 1, 1], 9, None, Mode::Rows);
    res.unwrap();
    assert!(got.iter().all(|&x| x == POISON), "modul 0 a scris");
    let mut cp = CommandProcessor::new();
    let mut cb = vec![0u8; 0x800];
    for (o, v) in [
        (0x728usize, 64u32),
        (0x72c, 128),
        (0x730, 1),
        (0x734, 128),
        (0x738, 1),
    ] {
        cb[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    for (i, va) in [QKV, P, DOUT, DS, DQKV].into_iter().enumerate() {
        cb[0x700 + 8 * i..0x708 + 8 * i].copy_from_slice(&va.to_le_bytes());
    }
    cp.mem.write(CB, &cb);
    let e = attn_back_tile_tc_on_cpu(&mut SimLaunch {
        mem: &mut cp.mem,
        cbuf0: CB,
        grid: [1, 1, 1],
        block: atc::BLOCK,
    })
    .unwrap_err();
    assert!(format!("{e:?}").contains("d > 64"), "{e:?}");
    // dS[0, 0, 0, 0] (on the diagonal, read by dQ) past f16 → refused, nothing written.
    let (res, got, want) = stand_in(s, [1, 1, 1], 9, Some((0, 1e6)), Mode::Columns);
    assert!(want.is_none());
    assert!(res.unwrap_err().contains("f16"));
    assert!(got.iter().all(|&x| x == POISON));
    println!(
        "kt c253 C1 SIMULARE: inlocuitorul = attn_tc::model pe {elems} elemente; modul 0 nimic; d>64 si f16 depasit refuzate"
    );
}

/// C2: on the mini `new_tiles3` list, `attn_tc_list` moves exactly one launch per layer; the zone grows by
/// 2 × (0x900 − 0x200); with no `AttnTc` and address 0, `pack_mixed_fast_attn` = `pack_mixed_fast` byte for byte.
#[test]
fn t253_c2_path_on_the_mini_list() {
    let list = mini_list(Layout::Tiles3);
    let (a, qa, pa) = ts::pack_mixed_fast(&list, prog_va, &FAST, Some(64 * KIB), ZONE).unwrap();
    let (b, qb, pb) =
        ts::pack_mixed_fast_attn(&list, prog_va, &FAST, None, Some(64 * KIB), ZONE).unwrap();
    assert_eq!((a.clone(), qa.clone(), pa.clone()), (b, qb, pb));
    let moved = ts::attn_tc_list(list.clone()).unwrap();
    assert_eq!(moved.len(), list.len());
    let n = moved
        .iter()
        .filter(|l| matches!(l, MixedLaunch::AttnTc(_)))
        .count();
    assert_eq!(n, MINI.layers);
    for (x, y) in list.iter().zip(&moved) {
        assert_eq!(x.tag(), y.tag());
        if let MixedLaunch::AttnTc(l) = y {
            assert_eq!(l.prog, Prog::AttnBackTile3);
            assert_eq!(x, &MixedLaunch::Std(l.clone()));
        } else {
            assert_eq!(x, y);
        }
    }
    let (z, q, p) = ts::pack_mixed_fast_attn(
        &moved,
        prog_va,
        &FAST,
        Some(AttnTcProgram::Tc(ATC)),
        Some(64 * KIB),
        ZONE,
    )
    .unwrap();
    assert_eq!(z.len(), a.len() + MINI.layers * (0x900 - 0x200));
    assert_eq!(q.len(), qa.len());
    assert_eq!(count(&p, TcPick::AttnTc), MINI.layers);
    assert_eq!(count(&p, TcPick::V4), count(&pa, TcPick::V4));
    // Plain (non-_3) lists have no attn_back_tile_3: the identity.
    let simple = mini_list(Layout::Simple);
    assert_eq!(ts::attn_tc_list(simple.clone()).unwrap(), simple);
    println!(
        "kt c253 C2 SIMULARE: mini tiles3: {} AttnTc, zona {} -> {} B, aceiasi octeti fara AttnTc",
        n,
        a.len(),
        z.len()
    );
}

/// C3: 5 mixed steps of the mini model, attention backward on `attn_back_tile_tc` vs on `attn_back_tile_3`.
#[test]
fn t253_c3_training_mini() {
    let none = &|_: &mut Vec<MixedLaunch>| {};
    let (f32_run, _) = run(Layout::Tiles3, None, none).unwrap();
    let (tc_run, picks) = run(Layout::Tiles3, Some(AttnTcProgram::Tc(ATC)), none).unwrap();
    assert_eq!(count(&picks, TcPick::AttnTc), MINI.layers);
    let mut worst_loss = 0f64;
    for (a, b) in tc_run.losses.iter().zip(&f32_run.losses) {
        assert!(a.iter().all(|x| x.is_finite()));
        let e = (mean(a) - mean(b)).abs() / mean(b).abs();
        worst_loss = worst_loss.max(e);
    }
    let first = mean(&tc_run.losses[0]);
    let last = mean(&tc_run.losses[STEPS as usize - 1]);
    let same_bits = tc_run
        .grad1
        .iter()
        .zip(&f32_run.grad1)
        .all(|(x, y)| x.to_bits() == y.to_bits());
    let g = rel_norm(&tc_run.grad1, &f32_run.grad1);
    let w = rel_norm(&tc_run.weights, &f32_run.weights);
    println!(
        "kt c253 C3 SIMULARE: mini 5 pasi: pierderea {first:.6} -> {last:.6}; fata de atentia F32: pierderi <= {worst_loss:.2e}, gradientul pasului 1 {g:.2e}, greutatile {w:.2e}; aceiasi biti: {same_bits}"
    );
    assert!(last < first, "pierderea nu scade");
    assert!(!same_bits, "f16 in dQ/dK/dV ar trebui sa schimbe bitii");
    assert!(worst_loss <= 1e-4, "pierderi {worst_loss}");
    assert!((1e-5..=2e-3).contains(&g), "gradient {g}");
    // Predicted <= 1e-4 (written before the run): found 3.8e-4 — the prediction was WRONG (AdamW turns small gradient
    // differences of near-zero entries into whole lr-sized steps). Asserted: within the mixed step's own weight
    // tolerance against PyTorch (`MIX_TOL_W_REL_NORM`, 1e-3).
    assert!(w <= ts::MIX_TOL_W_REL_NORM, "greutati {w}");
}

fn list_513(bsz: usize) -> Vec<MixedLaunch> {
    let m = train::M513;
    let inputs: Vec<u32> = (0..bsz as u32 * 512).map(|i| (i * 7919) % 16_384).collect();
    let base = VA + (1 << 30);
    let plan = Plan::new_tiles3(m, bsz, 512, &inputs, base)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split();
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    plan.step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap()
}

fn qmd_field(qmd: &[u8], lo: u32, hi: u32) -> u64 {
    let mut v = 0u64;
    for i in lo..=hi {
        if qmd[(i / 8) as usize] >> (i % 8) & 1 == 1 {
            v |= 1 << (i - lo);
        }
    }
    v
}

/// C4: #513's mixed list (batch 2): 12 `AttnTc` + 147 `mm_tc_v4`, zone + 12 × 1 792 B, each `AttnTc` QMD with the
/// program at `ATC`, 48 registers, cbuf 0x800 and the grid [8, 12, 2].
#[test]
fn t253_c4_list_513() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let list = list_513(2);
    let (a, _, _) = ts::pack_mixed_fast(&list, prog_va, &FAST, Some(64 * KIB), ZONE).unwrap();
    let moved = ts::attn_tc_list(list).unwrap();
    let (z, qm, p) = ts::pack_mixed_fast_attn(
        &moved,
        prog_va,
        &FAST,
        Some(AttnTcProgram::Tc(ATC)),
        Some(64 * KIB),
        ZONE,
    )
    .unwrap();
    assert_eq!(count(&p, TcPick::AttnTc), 12);
    assert_eq!(count(&p, TcPick::V4), 147);
    assert_eq!(z.len(), a.len() + 12 * 1792);
    let f = |x: nvidia_rm570::QmdField, qmd: &[u8]| qmd_field(qmd, x.lo, x.hi);
    for ((l, pk), qva) in moved.iter().zip(&p).zip(&qm) {
        if *pk != Some(TcPick::AttnTc) {
            continue;
        }
        let at = (qva - ZONE) as usize;
        let qmd = &z[at..at + 0x100];
        let lo = f(q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER, qmd);
        let hi = f(q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER, qmd);
        assert_eq!(hi << 32 | lo, ATC, "{}", l.tag());
        assert_eq!(f(q::NVC6C0_QMDV03_00_REGISTER_COUNT_V, qmd), 48);
        assert_eq!(f(q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH, qmd), 8);
        assert_eq!(f(q::NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT, qmd), 12);
        assert_eq!(f(q::NVC6C0_QMDV03_00_CTA_RASTER_DEPTH, qmd), 2);
        let cb = &z[at - 0x800..at];
        assert_eq!(
            u32::from_le_bytes(cb[0x728..0x72c].try_into().unwrap()),
            512
        );
        assert_eq!(u32::from_le_bytes(cb[0x738..0x73c].try_into().unwrap()), 1);
    }
    println!(
        "kt c253 C4 SIMULARE: #513 lot 2: 12 attn_back_tile_tc + 147 mm_tc_v4, zona +{} B",
        z.len() - a.len()
    );
}

/// C5: planted mistakes.
#[test]
fn t253_c5_plants() {
    let moved = ts::attn_tc_list(mini_list(Layout::Tiles3)).unwrap();
    assert!(ts::pack_mixed_fast_attn(&moved, prog_va, &FAST, None, Some(64 * KIB), ZONE).is_err());
    assert!(ts::pack_mixed_fast(&moved, prog_va, &FAST, Some(64 * KIB), ZONE).is_err());
    assert!(ts::pack_mixed_gelu(&moved, prog_va, TC, TCG, ZONE).is_err());
    assert!(
        ts::pack_mixed_fast_attn(
            &moved,
            prog_va,
            &FAST,
            Some(AttnTcProgram::Tc(ATC + 4)),
            None,
            ZONE
        )
        .is_err()
    );
    assert!(
        ts::pack_mixed_fast_attn(
            &moved,
            prog_va,
            &FAST,
            Some(AttnTcProgram::Tc(ATC)),
            Some(16 * KIB),
            ZONE
        )
        .is_err()
    );
    // A mode-0 attention launch cannot be moved.
    let mut bad = mini_list(Layout::Tiles3);
    for l in bad.iter_mut() {
        if let MixedLaunch::Std(x) = l
            && x.prog == Prog::AttnBackTile3
        {
            x.cbuf[0x60..0x64].copy_from_slice(&0u32.to_le_bytes());
        }
    }
    assert!(ts::attn_tc_list(bad).is_err());
    // The old note's layout: the arguments at 0x28 (as `attn_back_tile_3`), nothing at 0x700. The simulated program
    // reads mode 0 at 0x738 and writes nothing → dQ, dK, dV stay what the buffer held → the gradient is visibly wrong.
    let none = &|_: &mut Vec<MixedLaunch>| {};
    let (good, _) = run(Layout::Tiles3, Some(AttnTcProgram::Tc(ATC)), none).unwrap();
    let at_0x28 = &|z: &mut Vec<u8>, q: &[u64], p: &[Option<TcPick>]| {
        for (qva, pk) in q.iter().zip(p) {
            if *pk == Some(TcPick::AttnTc) {
                let cb = (qva - ZONE) as usize - 0x800;
                let args: Vec<u8> = z[cb + 0x700..cb + 0x740].to_vec();
                z[cb + 0x700..cb + 0x740].fill(0);
                z[cb + 0x28..cb + 0x68].copy_from_slice(&args);
            }
        }
    };
    // Found (predicted: the gradient off by > 1 %): in the simulator the never-written dQKV rows are unmapped memory,
    // so the next launch that reads them refuses the step — caught even earlier. On the board they would hold stale
    // bytes and the step would train on them silently: the reason the cbuf test of `attn_back_tile_tc` exists.
    let e = match run_zone(Layout::Tiles3, Some(AttnTcProgram::Tc(ATC)), none, at_0x28) {
        Ok((silent, _)) => {
            let g = rel_norm(&silent.grad1, &good.grad1);
            assert!(g > 1e-2, "{g}");
            format!("gradientul abate {g:.2e}")
        }
        Err(e) => {
            assert!(e.contains("nemapata"), "{e}");
            e
        }
    };
    assert!(good.losses.iter().flatten().all(|x| x.is_finite()));
    println!("kt c253 C5 SIMULARE: argumentele la 0x28 -> dQ/dK/dV nescrise -> {e}");
}

/// C6 (DEDUS, `occupancy`): blocks per SM at the 64 KiB step of the fast mixed step.
#[test]
fn t253_c6_blocks_at_64k() {
    let k = |name, regs, shared| Kernel {
        name,
        threads: 256,
        registers: regs,
        shared_bytes: shared,
    };
    let old = k(
        "attn_back_tile_3",
        nvidia_ga104::attn_back_tile_3::REGISTERS,
        33_024,
    );
    let tc = k("attn_back_tile_tc", atc::REGISTERS, atc::SHARED.bytes);
    let tc16 = k("attn_back_tile_tc (16 KiB)", atc::REGISTERS, 16_384);
    let mut rows = Vec::new();
    for kk in [old, tc, tc16] {
        let r0 = occupancy::blocks_per_sm(&SM86, &kk, 64 * KIB, Reserve::None)
            .unwrap()
            .blocks;
        let r1 = occupancy::blocks_per_sm(&SM86, &kk, 64 * KIB, Reserve::OneKib)
            .unwrap()
            .blocks;
        rows.push((kk.name, r0, r1));
    }
    println!(
        "kt c253 C6 DEDUS la treapta 64 KiB (blocuri pe SM, fara / cu rezerva 1 KiB): {rows:?}"
    );
    assert_eq!((rows[0].1, rows[0].2), (1, 1));
    assert_eq!((rows[1].1, rows[1].2), (2, 1));
    assert!(rows[2].1 >= 3 && rows[2].2 >= 3);
}

/// D1–D3 (`docs/C-157_ATTN_BACK_TC.md` §6): `attn_back_tile_tc16` in the fast mixed step.
#[test]
fn t253_d_tc16_path() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    // D1: the mini step with tc16 = with tc, bit for bit (the same model: the path only).
    let none = &|_: &mut Vec<MixedLaunch>| {};
    let (a, pa) = run(Layout::Tiles3, Some(AttnTcProgram::Tc(ATC)), none).unwrap();
    let (b, pb) = run(Layout::Tiles3, Some(AttnTcProgram::Tc16(ATC16)), none).unwrap();
    let eq = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, r)| p.to_bits() == r.to_bits());
    assert!(a.losses.iter().zip(&b.losses).all(|(x, y)| eq(x, y)));
    assert!(eq(&a.grad1, &b.grad1) && eq(&a.weights, &b.weights));
    assert_eq!(
        (count(&pa, TcPick::AttnTc), count(&pb, TcPick::AttnTc16)),
        (MINI.layers, MINI.layers)
    );
    assert_eq!(count(&pb, TcPick::AttnTc), 0);
    // D2: #513 with tc16: 12 launches, 16 KiB shared in the QMD, 48 registers, the same zone as with tc.
    let moved = ts::attn_tc_list(list_513(2)).unwrap();
    let (z32, _, _) = ts::pack_mixed_fast_attn(
        &moved,
        prog_va,
        &FAST,
        Some(AttnTcProgram::Tc(ATC)),
        Some(64 * KIB),
        ZONE,
    )
    .unwrap();
    let (z, qm, p) = ts::pack_mixed_fast_attn(
        &moved,
        prog_va,
        &FAST,
        Some(AttnTcProgram::Tc16(ATC16)),
        Some(64 * KIB),
        ZONE,
    )
    .unwrap();
    assert_eq!(z.len(), z32.len());
    assert_eq!(count(&p, TcPick::AttnTc16), 12);
    let f = |x: nvidia_rm570::QmdField, qmd: &[u8]| qmd_field(qmd, x.lo, x.hi);
    let mut seen = 0;
    for (pk, qva) in p.iter().zip(&qm) {
        if *pk != Some(TcPick::AttnTc16) {
            continue;
        }
        let at = (qva - ZONE) as usize;
        let qmd = &z[at..at + 0x100];
        let lo = f(q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER, qmd);
        let hi = f(q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER, qmd);
        assert_eq!(hi << 32 | lo, ATC16);
        assert_eq!(f(q::NVC6C0_QMDV03_00_REGISTER_COUNT_V, qmd), 48);
        assert_eq!(f(q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE, qmd), 16_384);
        assert_eq!(z[at - 0x800..at], z32[at - 0x800..at], "acelasi cbuf");
        let q32 = &z32[at..at + 0x100];
        assert_eq!(f(q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE, q32), 32_768);
        seen += 1;
    }
    assert_eq!(seen, 12);
    // D3: through RAM, `None` = the old packer byte for byte; with tc16 the same picks as without RAM.
    let (inputs, targets) = batch();
    let plan = plan_of(Layout::Tiles3, &inputs).resident().unwrap();
    let v = nvidia_ga104::batch_ram::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let list = mini_list(Layout::Tiles3);
    let old = nvidia_ga104::batch_ram::pack_mixed_fast_via_ram(
        &list,
        prog_va,
        &FAST,
        Some(64 * KIB),
        TC,
        ZONE,
        &v,
    )
    .unwrap();
    let new = nvidia_ga104::batch_ram::pack_mixed_fast_via_ram_attn(
        &list,
        prog_va,
        &FAST,
        None,
        Some(64 * KIB),
        TC,
        ZONE,
        &v,
    )
    .unwrap();
    assert_eq!(old, new);
    let mv = ts::attn_tc_list(list).unwrap();
    let (_, _, pr) = nvidia_ga104::batch_ram::pack_mixed_fast_via_ram_attn(
        &mv,
        prog_va,
        &FAST,
        Some(AttnTcProgram::Tc16(ATC16)),
        Some(64 * KIB),
        TC,
        ZONE,
        &v,
    )
    .unwrap();
    assert_eq!(count(&pr, TcPick::AttnTc16), MINI.layers);
    assert!(
        nvidia_ga104::batch_ram::pack_mixed_fast_via_ram(
            &mv,
            prog_va,
            &FAST,
            Some(64 * KIB),
            TC,
            ZONE,
            &v
        )
        .is_err()
    );
    println!(
        "kt c253 D SIMULARE: tc16 = tc bit cu bit pe mini (drumul); #513: 12 attn_back_tile_tc16, QMD 16 384 B / 48 registre, acelasi cbuf; prin RAM: None = vechiul, tc16 ales"
    );
}
