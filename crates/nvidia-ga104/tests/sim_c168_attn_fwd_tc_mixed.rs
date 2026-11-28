//! C-168 p.4 step 4 (session 289) — T305, SIMULARE: `attn_fwd_tc` (the attention forward on the tensor cores) linked
//! in the **fast mixed step**: `train_step::attn_fwd_tc_list` moves every aligned `attn_fwd_tile3` launch to
//! `MixedLaunch::AttnFwdTc`, and `train_step::pack_mixed_fast_attn9` packs it on `attn_fwd_tc` (cbuf at 0x700, the
//! grid in pairs, 120 registers, 27 648 B). The plan and the predictions H6a–H6d / H7 are in
//! `docs/C-168_ATTN_FWD_TC_PLAN.md` §11 (written before the code).
//!
//! Honesty: the `attn_fwd_tc` stand-in is the CPU model of T299 (`gpu_compute::attn_fwd_tc`, its adder a HYPOTHESIS
//! fitted on the board's `mm_tc` bits). `attn_fwd_tc` alone gave `attn_fwd_tile3`'s margins on the RTX 3070 Ti in os-504;
//! the step with it has NOT run on the board. Nothing here runs on a board.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, ATTN_FWD_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS,
    LN_BACK_RES_NEEDS, LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS,
    MM_TC_V5_NEEDS, MM_TC_V6_NEEDS, MM_TC_V7_NEEDS, MM_TC_V8_NEEDS, MM_TC_V8N_NEEDS,
    MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, Needs, XENT_NEEDS, attn_back_on_cpu,
    attn_back_tile_on_cpu, attn_back_tile_tc16_on_cpu, attn_back_tile_tc16_v4_on_cpu,
    attn_back_tile_tc16_v4pf_on_cpu, attn_back_tile0_on_cpu, attn_back_tile0_tc_on_cpu,
    attn_fwd_on_cpu, attn_fwd_tc_on_cpu, attn_fwd_tile_on_cpu, attn_p_on_cpu, attn_p_tile_on_cpu,
    colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu,
    ln_back_res_on_cpu, ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu,
    mm_tc_v4_on_cpu, mm_tc_v5_on_cpu, mm_tc_v6_on_cpu, mm_tc_v7_on_cpu, mm_tc_v8_on_cpu,
    mm_tc_v8n_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu,
    xent_on_cpu,
};
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::attn_back_tile_tc16_v4::ATTN_BACK_TILE_TC16_V4;
use nvidia_ga104::attn_back_tile_tc16_v4pf::ATTN_BACK_TILE_TC16_V4PF;
use nvidia_ga104::attn_back_tile0_tc::ATTN_BACK_TILE0_TC;
use nvidia_ga104::attn_fwd_tc::{self as af, ATTN_FWD_TC};
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::mm_tc_pf::MM_TC_PF;
use nvidia_ga104::mm_tc_v4::MM_TC_V4;
use nvidia_ga104::mm_tc_v5::MM_TC_V5;
use nvidia_ga104::mm_tc_v6::MM_TC_V6;
use nvidia_ga104::mm_tc_v7::MM_TC_V7;
use nvidia_ga104::mm_tc_v8::{self as t8, MM_TC_V8, MM_TC_V8N};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, Attn0Programs, AttnTcFast, AttnTcProgram, LOSS_SCALE, LOSS_SCALE_513, MixedLaunch,
    PROGRAMS_ALL_FUSED2, Plan, Prog, TcPick, TcPrograms, V8Programs,
};
use nvidia_ga104::transfer;
use nvidia_rm570::regs::clc7c0 as c;

/// The mini model of T297 / T304 (d = 16: `attn_fwd_tc` does not take it).
const MINI16: Arch = Arch {
    vocab: 16_384,
    ctx: 32,
    n_emb: 64,
    heads: 4,
    layers: 2,
};
/// An aligned mini model: d = 128 / 2 = 64 and T = 64 — the shapes `attn_fwd_tc` takes (one row tile, a pair of one).
const MINI: Arch = Arch {
    vocab: 16_384,
    ctx: 64,
    n_emb: 128,
    heads: 2,
    layers: 2,
};
const B: usize = 2;
const T: usize = 64;
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
const ATC16: u64 = PROGS + 0x1_0000 * 28;
const ATC0: u64 = PROGS + 0x1_0000 * 29;
const ATC16V4: u64 = PROGS + 0x1_0000 * 31;
const ATC16V4PF: u64 = PROGS + 0x1_0000 * 32;
const V5: u64 = PROGS + 0x1_0000 * 33;
const V6: u64 = PROGS + 0x1_0000 * 34;
const V7: u64 = PROGS + 0x1_0000 * 35;
const V8: u64 = PROGS + 0x1_0000 * 36;
const V8N: u64 = PROGS + 0x1_0000 * 37;
const AFT: u64 = PROGS + 0x1_0000 * 38;
/// The os-500 rates, as the OS runs `--pas-v8` since os-503.
const P8: V8Programs = V8Programs {
    v8: V8,
    v8n: V8N,
    sms: 48,
    rates: t8::RATES_OS500,
};
const HEAP: u64 = VA + 0x100_0000;

const FAST: TcPrograms = TcPrograms {
    tc: TC,
    pf: PF,
    v4: V4,
    gelu: TCG,
};
const AT: Option<AttnTcProgram> = Some(AttnTcProgram::Tc16(ATC16));
const A0: Attn0Programs = Attn0Programs { tc: ATC0, v4: 0 };
const FV4: Option<AttnTcFast> = Some(AttnTcFast::V4(ATC16V4));

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
    let mut g = Rng(0x00c1_9e10_2026_1126);
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
    let mut g = Rng(0x5eed_0000_0000_0253);
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
    cp.register_program_with_needs(MM_TC_V5, mm_tc_v5_on_cpu, MM_TC_V5_NEEDS);
    cp.mem.write(V5, MM_TC_V5);
    cp.register_program_with_needs(MM_TC_V6, mm_tc_v6_on_cpu, MM_TC_V6_NEEDS);
    cp.mem.write(V6, MM_TC_V6);
    cp.register_program_with_needs(MM_TC_V7, mm_tc_v7_on_cpu, MM_TC_V7_NEEDS);
    cp.mem.write(V7, MM_TC_V7);
    cp.register_program_with_needs(MM_TC_V8, mm_tc_v8_on_cpu, MM_TC_V8_NEEDS);
    cp.mem.write(V8, MM_TC_V8);
    cp.register_program_with_needs(MM_TC_V8N, mm_tc_v8n_on_cpu, MM_TC_V8N_NEEDS);
    cp.mem.write(V8N, MM_TC_V8N);
    cp.register_program_with_needs(
        ATTN_BACK_TILE_TC16,
        attn_back_tile_tc16_on_cpu,
        ATTN_BACK_TILE_TC16_NEEDS,
    );
    cp.mem.write(ATC16, ATTN_BACK_TILE_TC16);
    cp.register_program_with_needs(
        ATTN_BACK_TILE0_TC,
        attn_back_tile0_tc_on_cpu,
        ATTN_BACK_TILE0_TC_NEEDS,
    );
    cp.mem.write(ATC0, ATTN_BACK_TILE0_TC);
    cp.register_program_with_needs(
        ATTN_BACK_TILE_TC16_V4,
        attn_back_tile_tc16_v4_on_cpu,
        ATTN_BACK_TILE_TC16_V4_NEEDS,
    );
    cp.mem.write(ATC16V4, ATTN_BACK_TILE_TC16_V4);
    cp.register_program_with_needs(
        ATTN_BACK_TILE_TC16_V4PF,
        attn_back_tile_tc16_v4pf_on_cpu,
        ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    );
    cp.mem.write(ATC16V4PF, ATTN_BACK_TILE_TC16_V4PF);
    cp.register_program_with_needs(ATTN_FWD_TC, attn_fwd_tc_on_cpu, ATTN_FWD_TC_NEEDS);
    cp.mem.write(AFT, ATTN_FWD_TC);
    cp
}

fn scale() -> f32 {
    ts::loss_scale_for(MINI.vocab, B * T)
}

fn plan_mini(inputs: &[u32]) -> Plan {
    Plan::new_tiles3(MINI, B, T, inputs, HEAP).unwrap()
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

/// The list of step 1 with both attention-backward programs on the tensor cores (as the OS runs it), and with the
/// forward moved too when `fwd`.
fn mini_list(fwd: bool) -> Vec<MixedLaunch> {
    let (inputs, _) = batch();
    let params = initial();
    let plan = plan_mini(&inputs);
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let l = plan.step_mixed(Recipe::AdamW513, &opt, 1, scale()).unwrap();
    let l = ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap();
    if fwd {
        ts::attn_fwd_tc_list(l).unwrap()
    } else {
        l
    }
}

fn list_513(bsz: usize, fwd: bool) -> Vec<MixedLaunch> {
    let m = train::M513;
    let inputs: Vec<u32> = (0..bsz as u32 * 512).map(|i| (i * 7919) % 16_384).collect();
    let base = VA + (1 << 30);
    let plan = Plan::new_tiles3(m, bsz, 512, &inputs, base)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split();
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    let l = plan
        .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap();
    let l = ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap();
    if fwd {
        ts::attn_fwd_tc_list(l).unwrap()
    } else {
        l
    }
}

fn count(picks: &[Option<TcPick>], p: TcPick) -> usize {
    picks.iter().filter(|x| **x == Some(p)).count()
}

fn is_fwd3(l: &MixedLaunch) -> bool {
    matches!(l, MixedLaunch::Std(x) if x.prog == Prog::AttnFwdTile3)
}

struct Out {
    losses: Vec<Vec<f32>>,
    grad1: Vec<f32>,
    weights: Vec<f32>,
}

/// `STEPS` mixed steps of the aligned mini model on the simulated board (v7 / v8 / v8n by shape, both attention
/// backward programs on the tensor cores, the fast dQ/dK/dV variant), packed by `pack_mixed_fast_attn9` at the 64 KiB
/// step; the forward on `attn_fwd_tc` when `fwd`, else on `attn_fwd_tile3` (`attn_fwd_va = 0`).
fn run(fwd: bool) -> Result<(Out, Vec<Option<TcPick>>), String> {
    run_with(true, fwd)
}

/// [`run`] with the attention backward on the tensor cores only when `back` (else the F32 `_3` programs).
fn run_with(back: bool, fwd: bool) -> Result<(Out, Vec<Option<TcPick>>), String> {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = plan_mini(&inputs);
    let mut cp = board();
    for (va, b) in plan.uploads(&params, &inputs, &targets).unwrap() {
        cp.mem.write(va, &b);
    }
    for (va, b) in plan.mixed_uploads(scale()).unwrap() {
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
        let list = plan
            .step_mixed(Recipe::AdamW513, &opt, step, scale())
            .unwrap();
        let list = if back {
            ts::attn0_tc_list(ts::attn_tc_list(list).unwrap()).unwrap()
        } else {
            list
        };
        let list = if fwd {
            ts::attn_fwd_tc_list(list).unwrap()
        } else {
            list
        };
        let (zone, qmds, p) = ts::pack_mixed_fast_attn9(
            &list,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8),
            t8::LONG_K_OS503,
            if fwd { AFT } else { 0 },
            AT,
            FV4,
            A0,
            Some(64 * KIB),
            ZONE,
        )
        .map_err(|e| format!("{e:?}"))?;
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

fn qmd_field(qmd: &[u8], lo: u32, hi: u32) -> u64 {
    let mut v = 0u64;
    for i in lo..=hi {
        if qmd[(i / 8) as usize] >> (i % 8) & 1 == 1 {
            v |= 1 << (i - lo);
        }
    }
    v
}

fn field(zone: &[u8], qva: u64, f: nvidia_rm570::QmdField) -> u64 {
    let at = (qva - ZONE) as usize;
    qmd_field(&zone[at..at + 0x100], f.lo, f.hi)
}

fn program_va(zone: &[u8], qva: u64) -> u64 {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    field(zone, qva, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER) << 32
        | field(zone, qva, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER)
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}

/// ‖a − b‖ / ‖b‖ over the slice.
fn rel_norm(a: &[f32], b: &[f32]) -> f64 {
    let (mut d, mut n) = (0f64, 0f64);
    for (x, y) in a.iter().zip(b) {
        d += (*x as f64 - *y as f64).powi(2);
        n += (*y as f64).powi(2);
    }
    (d / n.max(f64::MIN_POSITIVE)).sqrt()
}

/// The largest ‖a − b‖ / ‖b‖ over the model's tensors (in `MINI.tensors()` order, packed one after another); with
/// `skip_k`, the K third of every `in_proj_bias` (exact gradient 0: rounding noise only) is left out of that tensor.
fn worst_tensor(a: &[f32], b: &[f32], skip_k: bool) -> (f64, String) {
    let mut at = 0usize;
    let mut worst = (0f64, String::new());
    for t in MINI.tensors() {
        let n = t.elements();
        let r = if skip_k && t.name.ends_with("in_proj_bias") {
            let third = n / 3;
            let (x, y) = (&a[at..at + n], &b[at..at + n]);
            let xa: Vec<f32> = x[..third].iter().chain(&x[2 * third..]).copied().collect();
            let ya: Vec<f32> = y[..third].iter().chain(&y[2 * third..]).copied().collect();
            rel_norm(&xa, &ya)
        } else {
            rel_norm(&a[at..at + n], &b[at..at + n])
        };
        if r > worst.0 || !r.is_finite() {
            worst = (r, t.name.to_string());
        }
        at += n;
    }
    assert_eq!(at, a.len(), "tensorii nu acopera parametrii");
    worst
}

/// T305 S1, SIMULARE: nothing changes without `attn_fwd_tc` — `attn9` with `attn_fwd_va = 0` on a list without
/// `AttnFwdTc` = `attn8k` byte for byte (mini, #513 lot 2 and 6, guard on and off); `attn_fwd_tc_list` is the identity
/// on a list whose attention it does not take (d = 16).
#[test]
fn t305_s1_without_fwd_same_bytes() {
    let k = Some(64 * KIB);
    for l in [mini_list(false), list_513(2, false), list_513(6, false)] {
        for long_k in [0, t8::LONG_K_OS503] {
            let a = ts::pack_mixed_fast_attn8k(
                &l,
                prog_va,
                &FAST,
                V5,
                V6,
                V7,
                Some(P8),
                long_k,
                AT,
                FV4,
                A0,
                k,
                ZONE,
            )
            .unwrap();
            for va in [0, AFT] {
                let b = ts::pack_mixed_fast_attn9(
                    &l,
                    prog_va,
                    &FAST,
                    V5,
                    V6,
                    V7,
                    Some(P8),
                    long_k,
                    va,
                    AT,
                    FV4,
                    A0,
                    k,
                    ZONE,
                )
                .unwrap();
                assert_eq!(a, b);
            }
        }
    }
    let (inputs, _) = batch();
    let inp16: Vec<u32> = inputs[..B * 32].to_vec();
    let p16 = Plan::new_tiles3(MINI16, B, 32, &inp16, HEAP).unwrap();
    let params: Vec<Vec<f32>> = MINI16
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let l16 = p16
        .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE)
        .unwrap();
    assert!(l16.iter().any(is_fwd3));
    assert_eq!(ts::attn_fwd_tc_list(l16.clone()).unwrap(), l16);
    println!(
        "kt c305 S1 SIMULARE: fara attn_fwd_tc (va 0 sau lista nemutata) -> attn8k octet cu octet (mini, #513 lot 2 si 6, cu si fara garda); d = 16 ramane pe attn_fwd_tile3"
    );
}

/// T305 S2, SIMULARE: in #513 (lot 2 and 6) `attn_fwd_tc_list` moves exactly the 12 `attn_fwd_tile3` launches (one per
/// layer, nothing else); `attn9` puts them on `attn_fwd_tc` (program, 120 registers, 27 648 B, blocks of 256, the grid
/// in pairs [4, 12, B], cbuf = `attn_fwd_tc::cbuf_from_f32`); every other pick and QMD is `attn8k`'s (with the guard:
/// lot 6 v7 13 / v8 0 / v8n 134).
#[test]
fn t305_s2_picks_and_qmds() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let k = Some(64 * KIB);
    let mut lines = Vec::new();
    for bsz in [2usize, 6] {
        let old = list_513(bsz, false);
        let new = list_513(bsz, true);
        assert_eq!(old.len(), new.len());
        assert_eq!(old.iter().filter(|l| is_fwd3(l)).count(), 12);
        assert_eq!(new.iter().filter(|l| is_fwd3(l)).count(), 0);
        let (za, qa, pa) = ts::pack_mixed_fast_attn8k(
            &old,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8),
            t8::LONG_K_OS503,
            AT,
            FV4,
            A0,
            k,
            ZONE,
        )
        .unwrap();
        let (zb, qb, pb) = ts::pack_mixed_fast_attn9(
            &new,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8),
            t8::LONG_K_OS503,
            AFT,
            AT,
            FV4,
            A0,
            k,
            ZONE,
        )
        .unwrap();
        assert_eq!(count(&pb, TcPick::AttnFwdTc), 12);
        // The zone grows by 12 × (SLOT_TC − SLOT); the QMDs after the first moved launch shift.
        assert_eq!(zb.len() - za.len(), 12 * (ts::SLOT_TC - ts::SLOT) as usize);
        let mut moved = 0;
        for (i, ((o, n), (x, y))) in old.iter().zip(&new).zip(pa.iter().zip(&pb)).enumerate() {
            let (qo, qn) = (qa[i], qb[i]);
            if let (MixedLaunch::Std(x3), MixedLaunch::AttnFwdTc(xn)) = (o, n) {
                moved += 1;
                assert_eq!(x3, xn);
                assert_eq!(*x, None);
                assert_eq!(*y, Some(TcPick::AttnFwdTc));
                let at = (qn - ZONE) as usize;
                let cb = af::cbuf_from_f32(&x3.cbuf).unwrap();
                assert_eq!(&zb[at - af::CBUF_BYTES..at], &cb[..]);
                assert_eq!(program_va(&zb, qn), AFT);
                assert_eq!(
                    field(&zb, qn, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V),
                    u64::from(af::REGISTERS)
                );
                assert_eq!(
                    field(&zb, qn, q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE),
                    u64::from(af::SHARED.bytes)
                );
                assert_eq!(
                    field(&zb, qn, q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0),
                    u64::from(af::BLOCK[0])
                );
                assert_eq!(field(&zb, qn, q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH), 4);
                assert_eq!(field(&zb, qn, q::NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT), 12);
                assert_eq!(
                    field(&zb, qn, q::NVC6C0_QMDV03_00_CTA_RASTER_DEPTH),
                    bsz as u64
                );
                assert_eq!(x3.grid, [8, 12, bsz as u32]);
                continue;
            }
            assert_eq!(o, n);
            assert_eq!(x, y, "lansarea {i}");
            // Same QMD apart from the cbuf address it points to.
            let (ao, an) = ((qo - ZONE) as usize, (qn - ZONE) as usize);
            let fo = |z: &[u8], qv: u64| {
                (
                    program_va(z, qv),
                    field(z, qv, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V),
                    field(z, qv, q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE),
                    field(z, qv, q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH),
                )
            };
            assert_eq!(fo(&za, qo), fo(&zb, qn), "lansarea {i}");
            let cbl = (o.slot() - 0x100) as usize;
            assert_eq!(&za[ao - cbl..ao], &zb[an - cbl..an], "cbuf {i}");
        }
        assert_eq!(moved, 12);
        let c3 = [
            count(&pb, TcPick::V7),
            count(&pb, TcPick::V8),
            count(&pb, TcPick::V8n),
        ];
        if bsz == 6 {
            assert_eq!(c3, [13, 0, 134]);
        }
        lines.push(format!(
            "lot {bsz}: 12 attn_fwd_tc, grila [4, 12, {bsz}], v7/v8/v8n {}/{}/{}",
            c3[0], c3[1], c3[2]
        ));
    }
    println!(
        "kt c305 S2 SIMULARE: #513 exact 12 lansari attn_fwd_tile3 -> attn_fwd_tc (120 registre, 27648 B, 256 fire, perechi), restul neschimbat; {}",
        lines.join("; ")
    );
}

/// T305 S3, SIMULARE: refusals before anything is sent — a list with `AttnFwdTc` and `attn_fwd_va = 0` (and in every
/// older packer), an unaligned `attn_fwd_va`, a moved launch whose grid does not match its arguments.
#[test]
fn t305_s3_refusals() {
    let k = Some(64 * KIB);
    let l = list_513(2, true);
    let e = ts::pack_mixed_fast_attn9(
        &l,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8),
        0,
        0,
        AT,
        FV4,
        A0,
        k,
        ZONE,
    );
    assert!(
        format!("{e:?}").contains("attn_fwd_tc nu e incarcat"),
        "{e:?}"
    );
    assert!(
        ts::pack_mixed_fast_attn8k(
            &l,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8),
            0,
            AT,
            FV4,
            A0,
            k,
            ZONE
        )
        .is_err()
    );
    assert!(
        ts::pack_mixed_fast_attn8(
            &l,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8),
            AT,
            FV4,
            A0,
            k,
            ZONE
        )
        .is_err()
    );
    assert!(ts::pack_mixed_fast_attn2(&l, prog_va, &FAST, AT, ATC0, k, ZONE).is_err());
    assert!(ts::pack_mixed_fast(&l, prog_va, &FAST, k, ZONE).is_err());
    assert!(
        ts::pack_mixed_fast_attn9(
            &l,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8),
            0,
            AFT + 0x80,
            AT,
            FV4,
            A0,
            k,
            ZONE,
        )
        .is_err()
    );
    // A tile3 launch with its grid's y changed: refused by the list (and by the packer, if made by hand).
    let mut bad = list_513(2, false);
    let i = bad.iter().position(is_fwd3).unwrap();
    if let MixedLaunch::Std(x) = &mut bad[i] {
        x.grid[1] = 11;
    }
    assert!(ts::attn_fwd_tc_list(bad.clone()).is_err());
    if let MixedLaunch::Std(x) = bad[i].clone() {
        bad[i] = MixedLaunch::AttnFwdTc(x);
    }
    assert!(
        ts::pack_mixed_fast_attn9(
            &bad,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8),
            0,
            AFT,
            AT,
            FV4,
            A0,
            k,
            ZONE,
        )
        .is_err()
    );
    println!(
        "kt c305 S3 SIMULARE: refuzuri inainte de trimitere (attn_fwd_tc neincarcat, packerii vechi, adresa nealiniata, grila nepotrivita)"
    );
}

/// T305 S4, SIMULARE (H6b): five steps of the aligned mini model through pushbuffer + QMD, the forward on `attn_fwd_tc`
/// against the same steps on `attn_fwd_tile3`. Written before the run (§11): inside the mixed step's tolerances
/// (`MIX_TOL_*`). Outcome (recorded in §12, PARTIAL): the losses, the drop and the step-1 gradient are inside; the
/// weights after 5 steps are NOT inside 1e-3 per tensor on the bias tensors — and neither are they for the change
/// already proven on the board (the attention backward on the tensor cores against the F32 `_3` programs, measured
/// here as the yardstick): zero-initialised biases after 5 Adam steps at lr 1e-3, and the K third of `in_proj_bias`,
/// whose exact gradient is 0 (only rounding noise, which Adam turns into ±lr steps). So this test asserts the three
/// figures that passed, prints the weights next to the yardstick, and bounds them loosely (finite, ≤ 0.1 outside the K
/// bias); the verdict on the weights is the oracle on the board (H7).
#[test]
fn t305_s4_training_in_margin() {
    let (f32b, _) = run_with(false, false).unwrap();
    let (base, p0) = run(false).unwrap();
    let (r, p1) = run(true).unwrap();
    assert_eq!(count(&p0, TcPick::AttnFwdTc), 0);
    assert_eq!(count(&p1, TcPick::AttnFwdTc), MINI.layers);
    let mut worst_loss = 0f64;
    for (a, b) in r.losses.iter().zip(&base.losses) {
        assert!(a.iter().all(|x| x.is_finite()));
        let d = ((mean(a) - mean(b)) / mean(b)).abs();
        worst_loss = worst_loss.max(d);
        assert!(d <= ts::MIX_TOL_LOSS_REL, "pierderea {d:e}");
    }
    let drop = |o: &Out| mean(&o.losses[0]) - mean(&o.losses[STEPS as usize - 1]);
    let dd = ((drop(&r) - drop(&base)) / drop(&base)).abs();
    assert!(drop(&base) > 0.0);
    assert!(dd <= ts::MIX_TOL_DROP_REL, "scaderea {dd:e}");
    let (g, gn) = worst_tensor(&r.grad1, &base.grad1, false);
    assert!(g <= ts::MIX_TOL_GRAD_REL_NORM, "gradientul {gn} {g:e}");
    // The weights: printed beside the yardstick (backward tc vs F32), bounded loosely.
    let (w, wn) = worst_tensor(&r.weights, &base.weights, true);
    let (wy, wyn) = worst_tensor(&base.weights, &f32b.weights, true);
    let (wk, _) = worst_tensor(&r.weights, &base.weights, false);
    let (wky, _) = worst_tensor(&base.weights, &f32b.weights, false);
    assert!(r.weights.iter().all(|x| x.is_finite()));
    assert!(w <= 0.1, "greutatile {wn} {w:e}");
    assert!(
        wy > ts::MIX_TOL_W_REL_NORM,
        "reperul a trecut: criteriul greutatilor ar deosebi"
    );
    // Not the same bits: the products are f16 (if this ever holds, the stand-in is not the tensor-core model).
    assert!(
        r.grad1
            .iter()
            .zip(&base.grad1)
            .any(|(x, y)| x.to_bits() != y.to_bits())
    );
    println!(
        "kt c305 S4 SIMULARE: mini aliniat (d 64, T 64, 2 straturi) 5 pasi attn_fwd_tc fata de attn_fwd_tile3: pierderea max {:.2e} (tol {:.0e}) TRECE, scaderea {:.2e} (tol {:.0e}) TRECE, gradientul max {:.2e} ({gn}, tol {:.0e}) TRECE; greutatile max {:.2e} ({wn}) / cu biasul K {:.2e} PESTE tol {:.0e}; reperul dovedit pe placa e si el peste (inapoi tc fata de F32: {:.2e} ({wyn}) / cu biasul K {:.2e}); pierderile {:.4} -> {:.4} (tile3 {:.4} -> {:.4})",
        worst_loss,
        ts::MIX_TOL_LOSS_REL,
        dd,
        ts::MIX_TOL_DROP_REL,
        g,
        ts::MIX_TOL_GRAD_REL_NORM,
        w,
        wk,
        ts::MIX_TOL_W_REL_NORM,
        wy,
        wky,
        mean(&r.losses[0]),
        mean(&r.losses[STEPS as usize - 1]),
        mean(&base.losses[0]),
        mean(&base.losses[STEPS as usize - 1]),
    );
}

/// T305 S5, SIMULARE: through RAM — `attn_fwd_va = 0` = `via_ram_attn8k` byte for byte; with `attn_fwd_tc` the picks
/// are those of `pack_mixed_fast_attn9`, and the list part of the zone is the same bytes at its place.
#[test]
fn t305_s5_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let plan = plan_mini(&inputs).resident().unwrap();
    let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let k = Some(64 * KIB);
    let l0 = mini_list(false);
    let a = br::pack_mixed_fast_via_ram_attn8k(
        &l0,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8),
        t8::LONG_K_OS503,
        AT,
        FV4,
        A0,
        k,
        TC,
        ZONE,
        &v,
    )
    .unwrap();
    let b = br::pack_mixed_fast_via_ram_attn9(
        &l0,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8),
        t8::LONG_K_OS503,
        0,
        AT,
        FV4,
        A0,
        k,
        TC,
        ZONE,
        &v,
    )
    .unwrap();
    assert_eq!(a, b);
    let l1 = mini_list(true);
    let (zr, _, pr) = br::pack_mixed_fast_via_ram_attn9(
        &l1,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8),
        t8::LONG_K_OS503,
        AFT,
        AT,
        FV4,
        A0,
        k,
        TC,
        ZONE,
        &v,
    )
    .unwrap();
    let n_in = v.copies_in.len();
    let list_va = ZONE + ts::SLOT * n_in as u64;
    let (zd, _, pd) = ts::pack_mixed_fast_attn9(
        &l1,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8),
        t8::LONG_K_OS503,
        AFT,
        AT,
        FV4,
        A0,
        k,
        list_va,
    )
    .unwrap();
    assert_eq!(pr, pd);
    let off = (list_va - ZONE) as usize;
    assert_eq!(&zr[off..off + zd.len()], &zd[..]);
    assert_eq!(count(&pr, TcPick::AttnFwdTc), MINI.layers);
    println!(
        "kt c305 S5 prin RAM SIMULARE: va 0 -> via_ram_attn8k octet cu octet; cu attn_fwd_tc aceleasi alegeri si aceeasi lista ca fara RAM"
    );
}
