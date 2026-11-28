//! C-168 p.5 step P5b (session 291) — T307, SIMULARE: GELU's backward fused into the store of the **fast** tensor-core
//! programs (`mm_tc_v7_gelu` / `mm_tc_v8_gelu` / `mm_tc_v8n_gelu`, `Plan::with_fused_gelu`, `pack_mixed_fast_attn10`)
//! in the fast mixed step of P5a (`new_tiles3` + `with_colsum_split` + `with_fused_back`, v7 / v8 / v8n by shape,
//! attention on the tensor cores). The plan and the predictions J1–J7 are in `docs/C-168_P5_LIPIRE_PLAN.md` §7
//! (written before this test).
//!
//! Honesty: everything here runs on the simulated board (`gpu_sim`); none of the three GELU' programs has run on a
//! board.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, ATTN_FWD_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS,
    LN_BACK_RES_NEEDS, LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS,
    MM_TC_V5_NEEDS, MM_TC_V6_NEEDS, MM_TC_V7_GELU_NEEDS, MM_TC_V7_NEEDS, MM_TC_V8_GELU_NEEDS,
    MM_TC_V8_NEEDS, MM_TC_V8N_GELU_NEEDS, MM_TC_V8N_NEEDS, MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS,
    Needs, XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu, attn_back_tile_tc16_on_cpu,
    attn_back_tile_tc16_v4_on_cpu, attn_back_tile_tc16_v4pf_on_cpu, attn_back_tile0_on_cpu,
    attn_back_tile0_tc_on_cpu, attn_fwd_on_cpu, attn_fwd_tc_on_cpu, attn_fwd_tile_on_cpu,
    attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, colsum_part_on_cpu, emb_back_on_cpu,
    emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu,
    ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu, mm_tc_v4_on_cpu,
    mm_tc_v5_on_cpu, mm_tc_v6_on_cpu, mm_tc_v7_gelu_on_cpu, mm_tc_v7_on_cpu, mm_tc_v8_gelu_on_cpu,
    mm_tc_v8_on_cpu, mm_tc_v8n_gelu_on_cpu, mm_tc_v8n_on_cpu, mm_train_gelu_on_cpu,
    mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::attn_back_tile_tc16_v4::ATTN_BACK_TILE_TC16_V4;
use nvidia_ga104::attn_back_tile_tc16_v4pf::ATTN_BACK_TILE_TC16_V4PF;
use nvidia_ga104::attn_back_tile0_tc::ATTN_BACK_TILE0_TC;
use nvidia_ga104::attn_fwd_tc::ATTN_FWD_TC;
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_fast_gelu::{self as fg, MM_TC_V7_GELU, MM_TC_V8_GELU, MM_TC_V8N_GELU};
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::mm_tc_pf::MM_TC_PF;
use nvidia_ga104::mm_tc_v4::MM_TC_V4;
use nvidia_ga104::mm_tc_v5::MM_TC_V5;
use nvidia_ga104::mm_tc_v6::MM_TC_V6;
use nvidia_ga104::mm_tc_v7::MM_TC_V7;
use nvidia_ga104::mm_tc_v8::{self as t8, MM_TC_V8, MM_TC_V8N};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, Attn0Programs, AttnTcFast, AttnTcProgram, GeluFast, LOSS_SCALE_513, MixedLaunch,
    PROGRAMS_ALL_FUSED2, Plan, Prog, TcPick, TcPrograms, V8Programs,
};
use nvidia_ga104::transfer;
use nvidia_rm570::regs::clc7c0 as c;

/// The mini model of T297 / T304 (d = 16: `attn_fwd_tc` does not take it).
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
const G7: u64 = PROGS + 0x1_0000 * 41;
const G8: u64 = PROGS + 0x1_0000 * 42;
const G8N: u64 = PROGS + 0x1_0000 * 43;
const GF: GeluFast = GeluFast {
    v7: G7,
    v8: G8,
    v8n: G8N,
};
/// The os-500 rates, as the OS runs `--pas-v8` since os-503.
const P8: V8Programs = V8Programs {
    v8: V8,
    v8n: V8N,
    sms: 48,
    rates: t8::RATES_OS500,
};
/// Where `colsum_part` is loaded (see [`prog_va`]).
const CSP: u64 = PROGS + 0x1_0000 * 40;
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
    if p == Prog::ColsumPart {
        // Index 24 = TCG's slot in this harness: put `colsum_part` after every other program.
        return CSP;
    }
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
    cp.register_program_with_needs(MM_TC_V7_GELU, mm_tc_v7_gelu_on_cpu, MM_TC_V7_GELU_NEEDS);
    cp.mem.write(G7, MM_TC_V7_GELU);
    cp.register_program_with_needs(MM_TC_V8_GELU, mm_tc_v8_gelu_on_cpu, MM_TC_V8_GELU_NEEDS);
    cp.mem.write(G8, MM_TC_V8_GELU);
    cp.register_program_with_needs(MM_TC_V8N_GELU, mm_tc_v8n_gelu_on_cpu, MM_TC_V8N_GELU_NEEDS);
    cp.mem.write(G8N, MM_TC_V8N_GELU);
    cp.mem.write(AFT, ATTN_FWD_TC);
    let csp = Prog::ColsumPart.code();
    cp.register_program(csp, colsum_part_on_cpu);
    cp.mem.write(CSP, csp);
    cp
}

fn scale() -> f32 {
    ts::loss_scale_for(MINI.vocab, B * T)
}

/// The P5a plan shape on the mini model (`new_tiles3` + `with_colsum_split` + `with_fused_back`), + `with_fused_gelu`
/// when `gelu`.
fn plan_mini(inputs: &[u32], gelu: bool) -> Plan {
    let p = Plan::new_tiles3(MINI, B, T, inputs, HEAP)
        .unwrap()
        .with_colsum_split()
        .with_fused_back();
    if gelu { p.with_fused_gelu() } else { p }
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

/// The #513 list of step 1 as the OS would send it after P5a (`new_tiles3` + `with_mm_tile` + `with_colsum_split` +
/// `with_fused_back`, both attention backward programs and the forward on the tensor cores), + `with_fused_gelu` when
/// `gelu`.
fn list_513(bsz: usize, gelu: bool) -> Vec<MixedLaunch> {
    let m = train::M513;
    let inputs: Vec<u32> = (0..bsz as u32 * 512).map(|i| (i * 7919) % 16_384).collect();
    let base = VA + (1 << 30);
    let plan = Plan::new_tiles3(m, bsz, 512, &inputs, base)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split()
        .with_fused_back();
    let plan = if gelu { plan.with_fused_gelu() } else { plan };
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    let l = plan
        .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap();
    ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()).unwrap()
}

#[allow(clippy::type_complexity)]
fn pack10(
    list: &[MixedLaunch],
    gelu: Option<GeluFast>,
) -> Result<(Vec<u8>, Vec<u64>, Vec<Option<TcPick>>), String> {
    pack10_with(list, gelu, P8)
}

#[allow(clippy::type_complexity)]
fn pack10_with(
    list: &[MixedLaunch],
    gelu: Option<GeluFast>,
    p8: V8Programs,
) -> Result<(Vec<u8>, Vec<u64>, Vec<Option<TcPick>>), String> {
    ts::pack_mixed_fast_attn10(
        list,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(p8),
        t8::LONG_K_OS503,
        AFT,
        gelu,
        AT,
        FV4,
        A0,
        Some(64 * KIB),
        ZONE,
    )
    .map_err(|e| format!("{e:?}"))
}

fn pack9(list: &[MixedLaunch]) -> (Vec<u8>, Vec<u64>, Vec<Option<TcPick>>) {
    ts::pack_mixed_fast_attn9(
        list,
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
        Some(64 * KIB),
        ZONE,
    )
    .unwrap()
}

fn std_prog(l: &MixedLaunch) -> Option<Prog> {
    match l {
        MixedLaunch::Std(x) => Some(x.prog),
        _ => None,
    }
}

fn count_prog(list: &[MixedLaunch], p: Prog) -> usize {
    list.iter().filter(|l| std_prog(l) == Some(p)).count()
}

fn count(picks: &[Option<TcPick>], p: TcPick) -> usize {
    picks.iter().filter(|x| **x == Some(p)).count()
}

/// True for the launches `with_fused_gelu` changes: "inapoi:proj-dX" and the backward `gelu_tr` (old list),
/// "inapoi:proj-dX+gelu" (new list).
fn touched(l: &MixedLaunch) -> bool {
    match l {
        MixedLaunch::Std(x) => x.prog == Prog::GeluTr && x.tag == "inapoi:gelu",
        MixedLaunch::Tc { tag, .. } => *tag == "inapoi:proj-dX",
        MixedLaunch::TcGelu { .. } => true,
        _ => false,
    }
}

/// Reads one field of a 256-byte QMD.
fn qmd_field(qmd: &[u8], fl: nvidia_rm570::QmdField) -> u64 {
    let mut v = 0u64;
    for i in fl.lo..=fl.hi {
        if qmd[(i / 8) as usize] >> (i % 8) & 1 == 1 {
            v |= 1 << (i - fl.lo);
        }
    }
    v
}

struct Out {
    losses: Vec<Vec<f32>>,
    grad1: Vec<f32>,
    weights: Vec<f32>,
    launches: usize,
    picks: Vec<Option<TcPick>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Mode {
    /// P5a: no `with_fused_gelu` (proj-dX on v7 / v8 / v8n + backward `gelu_tr`), `attn9`.
    Base,
    /// `with_fused_gelu`, `attn9`: proj-dX+gelu on the slow `mm_tc_gelu`.
    Slow,
    /// `with_fused_gelu`, `attn10` with [`GF`]: proj-dX+gelu on the GELU' variant of the fast program.
    Fast,
}

/// `STEPS` fast mixed steps of the aligned mini model on the simulated board (64 KiB step, the os-503 guard).
fn run(mode: Mode) -> Result<Out, String> {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = plan_mini(&inputs, mode != Mode::Base);
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
        launches: 0,
        picks: Vec::new(),
    };
    for step in 1..=STEPS {
        let list = plan
            .step_mixed(Recipe::AdamW513, &opt, step, scale())
            .unwrap();
        let list =
            ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(list).unwrap()).unwrap())
                .unwrap();
        out.launches = list.len();
        let (zone, qmds, picks) = match mode {
            Mode::Fast => pack10(&list, Some(GF))?,
            _ => pack9(&list),
        };
        out.picks = picks;
        submit(&mut cp, &mut first, &zone, &qmds).map_err(|e| format!("pasul {step}: {e}"))?;
        out.losses
            .push(cp.mem.read_f32s(plan.loss_rows, B * T).unwrap());
        if step == 1 {
            out.grad1 = cp.mem.read_f32s(plan.g, plan.n_params).unwrap();
        }
    }
    out.weights = cp.mem.read_f32s(plan.w, plan.n_params).unwrap();
    Ok(out)
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn same(a: &Out, b: &Out, what: &str) {
    for (s, (x, y)) in a.losses.iter().zip(&b.losses).enumerate() {
        assert_eq!(bits(x), bits(y), "{what}: pierderile pasului {}", s + 1);
    }
    assert_eq!(
        bits(&a.grad1),
        bits(&b.grad1),
        "{what}: gradientul pasului 1"
    );
    assert_eq!(bits(&a.weights), bits(&b.weights), "{what}: greutatile");
}

/// T307 S0 (J1, J2 as written in the library), CALCUL: the three programs are the compiled ones (SHA-256), differ
/// from their non-GELU sources, and keep the blocks per SM of the plan.
#[test]
fn t307_s0_programs() {
    assert_ne!(MM_TC_V8_GELU, MM_TC_V8);
    assert_ne!(MM_TC_V8N_GELU, MM_TC_V8N);
    assert_ne!(MM_TC_V7_GELU, MM_TC_V7);
    fg::check().unwrap();
    println!(
        "kt c307 S0 CALCUL: mm_tc_v7_gelu {} / v8_gelu {} / v8n_gelu {} registre; blocuri pe SM {} / {} / {}",
        fg::REGISTERS_V7,
        fg::REGISTERS_V8,
        fg::REGISTERS_V8N,
        65_536 / (128 * fg::REGISTERS_V7),
        65_536 / (256 * fg::REGISTERS_V8),
        65_536 / (128 * fg::REGISTERS_V8N),
    );
}

/// T307 S1 (J3), SIMULARE: the aligned mini model, 5 fast mixed steps: with `with_fused_gelu` on the GELU' variants
/// of the fast programs the losses of every step, the gradient of step 1 and the final weights are the same bits as on
/// the slow `mm_tc_gelu` and as without `with_fused_gelu` (P5a); the list is L launches shorter.
#[test]
fn t307_s1_mini_same_bits() {
    let base = run(Mode::Base).unwrap();
    let slow = run(Mode::Slow).unwrap();
    let fast = run(Mode::Fast).unwrap();
    same(&base, &slow, "P5a fata de mm_tc_gelu");
    same(&slow, &fast, "mm_tc_gelu fata de variantele rapide");
    assert!(fast.weights.iter().all(|x| x.is_finite()));
    assert_eq!(base.launches - fast.launches, MINI.layers);
    assert_eq!(slow.launches, fast.launches);
    let g = |p| count(&fast.picks, p);
    assert_eq!(count(&slow.picks, TcPick::Gelu), MINI.layers);
    assert_eq!(g(TcPick::Gelu), 0, "niciun TcGelu nu ramane pe mm_tc_gelu");
    assert_eq!(
        g(TcPick::V7Gelu) + g(TcPick::V8Gelu) + g(TcPick::V8nGelu),
        MINI.layers
    );
    println!(
        "kt c307 S1 SIMULARE: modelul mic aliniat, 5 pasi rapizi: GELU' in coada lui v7/v8/v8n ({}/{}/{}) = aceiasi biti ca mm_tc_gelu si ca P5a fara lipire (pierderile, gradientul pasului 1, greutatile); {} -> {} lansari pe pas",
        g(TcPick::V7Gelu),
        g(TcPick::V8Gelu),
        g(TcPick::V8nGelu),
        base.launches,
        fast.launches
    );
}

/// T307 S2 (J4), SIMULARE: #513 lot 2 and lot 6 after P5a: with `with_fused_gelu` exactly 12 launches fewer (the 12
/// backward `gelu_tr`), the 12 "proj-dX" become "proj-dX+gelu" on the GELU' variant of the program they had, every other
/// launch the same and in the same order with the same pick; each GELU' QMD points at its program, registers, block.
#[test]
fn t307_s2_513_lists() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let mut lines = Vec::new();
    for bsz in [2usize, 6] {
        let old = list_513(bsz, false);
        let new = list_513(bsz, true);
        assert_eq!(old.len() - new.len(), 12, "lot {bsz}");
        assert_eq!(
            count_prog(&old, Prog::GeluTr) - count_prog(&new, Prog::GeluTr),
            12
        );
        let (_, _, pa) = pack9(&old);
        let (zb, qb, pb) = pack10(&new, Some(GF)).unwrap();
        // the rest: same launches, same order, same picks
        let rest =
            |l: &[MixedLaunch], p: &[Option<TcPick>]| -> Vec<(MixedLaunch, Option<TcPick>)> {
                l.iter()
                    .zip(p)
                    .filter(|(x, _)| !touched(x))
                    .map(|(x, y)| (x.clone(), *y))
                    .collect()
            };
        assert_eq!(rest(&old, &pa), rest(&new, &pb), "lot {bsz}: restul listei");
        // proj-dX: the old pick, in GELU' form, layer by layer
        let old_dx: Vec<Option<TcPick>> = old
            .iter()
            .zip(&pa)
            .filter(|(l, _)| matches!(l, MixedLaunch::Tc { tag, .. } if *tag == "inapoi:proj-dX"))
            .map(|(_, p)| *p)
            .collect();
        let new_dx: Vec<Option<TcPick>> = new
            .iter()
            .zip(&pb)
            .filter(|(l, _)| matches!(l, MixedLaunch::TcGelu { .. }))
            .map(|(_, p)| *p)
            .collect();
        assert_eq!(old_dx.len(), 12);
        let gelu_of = |p: Option<TcPick>| match p {
            Some(TcPick::V7) => Some(TcPick::V7Gelu),
            Some(TcPick::V8) => Some(TcPick::V8Gelu),
            Some(TcPick::V8n) => Some(TcPick::V8nGelu),
            x => panic!("proj-dX pe {x:?}"),
        };
        assert_eq!(
            old_dx.iter().map(|p| gelu_of(*p)).collect::<Vec<_>>(),
            new_dx,
            "lot {bsz}"
        );
        for ((l, qv), p) in new.iter().zip(&qb).zip(&pb) {
            if let MixedLaunch::TcGelu { args, .. } = l {
                let at = (qv - ZONE) as usize;
                let qmd = &zb[at..at + 0x100];
                let va = qmd_field(qmd, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER) << 32
                    | qmd_field(qmd, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER);
                let (pva, regs, block, grid) = match p {
                    Some(TcPick::V8nGelu) => {
                        (G8N, fg::REGISTERS_V8N, 128, t8::grid_n(&fg::tc_args(args)))
                    }
                    Some(TcPick::V8Gelu) => {
                        (G8, fg::REGISTERS_V8, 256, t8::grid(&fg::tc_args(args)))
                    }
                    Some(TcPick::V7Gelu) => (G7, fg::REGISTERS_V7, 128, fg::tc_args(args).grid()),
                    x => panic!("{x:?}"),
                };
                assert_eq!(va, pva);
                assert_eq!(
                    qmd_field(qmd, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V),
                    u64::from(regs)
                );
                assert_eq!(
                    qmd_field(qmd, q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0),
                    block
                );
                assert_eq!(
                    qmd_field(qmd, q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH),
                    u64::from(grid[0])
                );
                assert_eq!(
                    qmd_field(qmd, q::NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT),
                    u64::from(grid[1])
                );
                // the cbuf right before the QMD = mm_tc_gelu's
                let cb = args.cbuf().unwrap();
                assert_eq!(&zb[at - cb.len()..at], &cb[..]);
            }
        }
        let c3 = [
            count(&pb, TcPick::V7Gelu),
            count(&pb, TcPick::V8Gelu),
            count(&pb, TcPick::V8nGelu),
        ];
        lines.push(format!(
            "lot {bsz}: {} -> {} lansari (pe placa {} -> {}), proj-dX+gelu pe v7/v8/v8n_gelu {}/{}/{}",
            old.len(),
            new.len(),
            old.len() + 7,
            new.len() + 7,
            c3[0],
            c3[1],
            c3[2]
        ));
    }
    println!(
        "kt c307 S2 SIMULARE: #513 cu GELU' in coada rapida: exact 12 lansari mai putin (12 gelu_tr inapoi), proj-dX pe varianta GELU a aceluiasi program, restul si alegerile neschimbate; {}",
        lines.join("; ")
    );
}

/// T307 S3 (J5), SIMULARE: refusals before anything is sent, and `None` = `attn9` byte for byte.
#[test]
fn t307_s3_refusals() {
    let new = list_513(2, true);
    let (_, _, picks) = pack10(&new, Some(GF)).unwrap();
    // `None`: exactly attn9 (the GELU' launches on mm_tc_gelu)
    assert_eq!(pack10(&new, None).unwrap(), pack9(&new));
    // the variant the pick wants is not loaded
    for (g, p) in [
        (GeluFast { v8n: 0, ..GF }, TcPick::V8nGelu),
        (GeluFast { v8: 0, ..GF }, TcPick::V8Gelu),
        (GeluFast { v7: 0, ..GF }, TcPick::V7Gelu),
    ] {
        let r = pack10(&new, Some(g));
        if count(&picks, p) > 0 {
            assert!(r.unwrap_err().contains("varianta GELU"), "{p:?}");
        } else {
            assert!(r.is_ok(), "{p:?} nefolosit: lista trece");
        }
    }
    // unaligned address
    assert!(pack10(&new, Some(GeluFast { v8n: G8N + 4, ..GF })).is_err());
    // the CPU model refuses a BIAS flag (the program has no bias): the cbuf is refused by the library, and a zone
    // patched by hand is refused by the stand-in
    let mut cp = board();
    let l = vec![product_launch()];
    let (zone, qmds, p) = pack10(&l, Some(GF)).unwrap();
    assert!(matches!(
        p[0],
        Some(TcPick::V7Gelu | TcPick::V8Gelu | TcPick::V8nGelu)
    ));
    fill_product(&mut cp);
    let mut first = 1u32;
    submit(&mut cp, &mut first, &zone, &qmds).unwrap();
    let mut z2 = zone.clone();
    let at = (qmds[0] - ZONE) as usize - nvidia_ga104::mm_tc::CBUF_BYTES + 0x72c;
    z2[at] |= 4;
    assert!(submit(&mut cp, &mut first, &z2, &qmds).is_err());
    assert!(
        nvidia_ga104::mm_tc_gelu::Args {
            flags: nvidia_ga104::mm_tc::BIAS,
            ..gelu_args()
        }
        .cbuf()
        .is_none()
    );
    assert!(
        nvidia_ga104::mm_tc_gelu::Args {
            u: gelu_args().c,
            ..gelu_args()
        }
        .cbuf()
        .is_none()
    );
    println!(
        "kt c307 S3 SIMULARE: varianta GELU neincarcata / adresa nealiniata refuzate inainte de trimis; None = attn9 octet cu octet; BIAS refuzat de model"
    );
}

/// One aligned product C = GELU'(u) ⊙ (A·Bᵀ), m = 256, n = 192, k = 96 (TB), on the heap.
fn gelu_args() -> nvidia_ga104::mm_tc_gelu::Args {
    nvidia_ga104::mm_tc_gelu::Args {
        a: HEAP,
        b: HEAP + 0x10_0000,
        c: HEAP + 0x20_0000,
        u: HEAP + 0x30_0000,
        m: 256,
        n: 192,
        k: 96,
        flags: nvidia_ga104::mm_tc::TB,
    }
}

fn product_launch() -> MixedLaunch {
    MixedLaunch::TcGelu {
        args: gelu_args(),
        tag: "proba",
        layer: None,
    }
}

fn fill_product(cp: &mut CommandProcessor) {
    let g = gelu_args();
    let mut r = Rng(0x0c16_8f5b_2026_1128);
    let mut f = |n: usize, w: f64| -> Vec<u8> {
        (0..n)
            .flat_map(|_| (((r.unit() * 2.0 - 1.0) * w) as f32).to_le_bytes())
            .collect()
    };
    let (m, n, k) = (g.m as usize, g.n as usize, g.k as usize);
    let a = f(m * k, 1.0);
    let b = f(n * k, 1.0);
    let u = f(m * n, 3.0);
    cp.mem.write(g.a, &a);
    cp.mem.write(g.b, &b);
    cp.mem.write(g.u, &u);
    cp.mem.write(g.c, &vec![0xffu8; m * n * 4]);
}

/// T307 S4 (J3 on each program), SIMULARE: one aligned product forced onto each of the three GELU' programs (the pick
/// steered by the rates and the SM count) gives the same bits as on `mm_tc_gelu`; each QMD carries its program.
#[test]
fn t307_s4_each_program_same_bits() {
    let l = vec![product_launch()];
    let g = gelu_args();
    let read = |cp: &CommandProcessor| bits(&cp.mem.read_f32s(g.c, (g.m * g.n) as usize).unwrap());
    // the reference: mm_tc_gelu (attn9)
    let mut cp = board();
    fill_product(&mut cp);
    let (zone, qmds, p) = pack9(&l);
    assert_eq!(p, vec![Some(TcPick::Gelu)]);
    let mut first = 1u32;
    submit(&mut cp, &mut first, &zone, &qmds).unwrap();
    let want = read(&cp);
    assert!(want.iter().all(|x| f32::from_bits(*x).is_finite()));
    let mut seen = Vec::new();
    for (sms, rates) in [
        (48, t8::RATES_OS500),
        (
            1,
            t8::Rates {
                v8_pct: 400,
                v8n_pct: 118,
            },
        ),
        (
            1,
            t8::Rates {
                v8_pct: 110,
                v8n_pct: 400,
            },
        ),
    ] {
        let p8 = V8Programs { sms, rates, ..P8 };
        let (zone, qmds, p) = pack10_with(&l, Some(GF), p8).unwrap();
        let mut cp = board();
        fill_product(&mut cp);
        let mut first = 1u32;
        submit(&mut cp, &mut first, &zone, &qmds).unwrap();
        assert_eq!(read(&cp), want, "{:?}", p[0]);
        seen.push(p[0].unwrap());
    }
    assert_eq!(seen, [TcPick::V7Gelu, TcPick::V8Gelu, TcPick::V8nGelu]);
    println!(
        "kt c307 S4 SIMULARE: acelasi produs (256x192x96, TB) pe mm_tc_v7_gelu / mm_tc_v8_gelu / mm_tc_v8n_gelu = aceiasi biti ca mm_tc_gelu"
    );
}

/// T307 S5, SIMULARE: through RAM (the OS's board path) — `gelu_fast = None` = `via_ram_attn9` byte for byte; with the
/// GELU' programs the picks are those of `pack_mixed_fast_attn10` and the list part of the zone is the same bytes at
/// its place.
#[test]
fn t307_s5_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let plan = plan_mini(&inputs, true).resident().unwrap();
    let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let k = Some(64 * KIB);
    let opt = AdamW::new(&initial(), LR, 0.9, 0.999, 1e-8, 0.1);
    let l = plan.step_mixed(Recipe::AdamW513, &opt, 1, scale()).unwrap();
    let l = ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()).unwrap();
    let via = |g: Option<GeluFast>| {
        br::pack_mixed_fast_via_ram_attn10(
            &l,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8),
            t8::LONG_K_OS503,
            AFT,
            g,
            AT,
            FV4,
            A0,
            k,
            TC,
            ZONE,
            &v,
        )
        .unwrap()
    };
    let old = br::pack_mixed_fast_via_ram_attn9(
        &l,
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
    assert_eq!(via(None), old);
    let (zr, _, pr) = via(Some(GF));
    let n_in = v.copies_in.len();
    let list_va = ZONE + ts::SLOT * n_in as u64;
    let (zd, _, pd) = ts::pack_mixed_fast_attn10(
        &l,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8),
        t8::LONG_K_OS503,
        AFT,
        Some(GF),
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
    assert_eq!(
        count(&pr, TcPick::V7Gelu) + count(&pr, TcPick::V8Gelu) + count(&pr, TcPick::V8nGelu),
        MINI.layers
    );
    println!(
        "kt c307 S5 prin RAM SIMULARE: None -> via_ram_attn9 octet cu octet; cu variantele GELU aceleasi alegeri si aceeasi lista ca fara RAM"
    );
}
