//! C-168 p.3 G7 (session 280) — T297, SIMULARE: `mm_tc_v8` (128×128) and `mm_tc_v8n` (128×64) linked in the **fast
//! mixed step**: `train_step::pack_mixed_fast_attn8` sends every aligned `mm_tc` launch to the program
//! `mm_tc_v8::pick` gives by waves (v7 / v8 / v8n), else the rule of `attn7`. The plan and the predictions are in
//! `docs/C-168_V8_PLAN.md` (G7 written before the code; the rates of the pick are DEDUS until the probe `--mm-tc-v8`).
//!
//! Honesty: the v8 / v8n stand-ins are `mm_tc`'s arithmetic on their tiles (T295). Neither has run on the RTX 3070 Ti,
//! nor has the step with them. Nothing here runs on a board.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS,
    LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS, MM_TC_V5_NEEDS,
    MM_TC_V6_NEEDS, MM_TC_V7_NEEDS, MM_TC_V8_NEEDS, MM_TC_V8N_NEEDS, MM_TRAIN_GELU_NEEDS,
    MM_TRAIN_NEEDS, Needs, XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu,
    attn_back_tile_tc16_on_cpu, attn_back_tile_tc16_v4_on_cpu, attn_back_tile_tc16_v4pf_on_cpu,
    attn_back_tile0_on_cpu, attn_back_tile0_tc_on_cpu, attn_fwd_on_cpu, attn_fwd_tile_on_cpu,
    attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu,
    gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu, ln_fwd_on_cpu,
    mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu, mm_tc_v4_on_cpu, mm_tc_v5_on_cpu,
    mm_tc_v6_on_cpu, mm_tc_v7_on_cpu, mm_tc_v8_on_cpu, mm_tc_v8n_on_cpu, mm_train_gelu_on_cpu,
    mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::attn_back_tile_tc16_v4::ATTN_BACK_TILE_TC16_V4;
use nvidia_ga104::attn_back_tile_tc16_v4pf::ATTN_BACK_TILE_TC16_V4PF;
use nvidia_ga104::attn_back_tile0_tc::ATTN_BACK_TILE0_TC;
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
const ATC16: u64 = PROGS + 0x1_0000 * 28;
const ATC0: u64 = PROGS + 0x1_0000 * 29;
const ATC16V4: u64 = PROGS + 0x1_0000 * 31;
const ATC16V4PF: u64 = PROGS + 0x1_0000 * 32;
const V5: u64 = PROGS + 0x1_0000 * 33;
const V6: u64 = PROGS + 0x1_0000 * 34;
const V7: u64 = PROGS + 0x1_0000 * 35;
const V8: u64 = PROGS + 0x1_0000 * 36;
const V8N: u64 = PROGS + 0x1_0000 * 37;
const P8: V8Programs = V8Programs {
    v8: V8,
    v8n: V8N,
    sms: 48,
    rates: t8::RATES_PLAN,
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
    cp
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

/// The mini list with both attention-backward programs moved onto the tensor cores.
fn mini_list() -> Vec<MixedLaunch> {
    let (inputs, _) = batch();
    let params = initial();
    let plan = plan_mini(&inputs);
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let l = plan
        .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE)
        .unwrap();
    ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()
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
    let l = plan
        .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap();
    ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()
}

fn count(picks: &[Option<TcPick>], p: TcPick) -> usize {
    picks.iter().filter(|x| **x == Some(p)).count()
}

struct Out {
    losses: Vec<Vec<f32>>,
    grad1: Vec<f32>,
    weights: Vec<f32>,
}

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}

type ZoneEdit = dyn Fn(&mut Vec<u8>, &[u64], &[Option<TcPick>]);

const NONE: &ZoneEdit = &|_, _, _| {};

/// `STEPS` mixed steps of the mini model on the simulated board, both attention-backward programs on the tensor
/// cores, packed by `pack_mixed_fast_attn8` at the 64 KiB step.
#[allow(clippy::too_many_arguments)]
fn run(
    progs: &TcPrograms,
    v5: u64,
    v6: u64,
    v7: u64,
    v8: Option<V8Programs>,
    attn: Option<AttnTcProgram>,
    fast: Option<AttnTcFast>,
    list_edit: &dyn Fn(&mut [MixedLaunch]),
    zone_edit: &ZoneEdit,
) -> Result<(Out, Vec<Option<TcPick>>), String> {
    run_k(progs, v5, v6, v7, v8, 0, attn, fast, list_edit, zone_edit)
}

/// [`run`] packed by `pack_mixed_fast_attn8k` with the long-sum guard `long_k` (T304; 0 = `run`).
#[allow(clippy::too_many_arguments)]
fn run_k(
    progs: &TcPrograms,
    v5: u64,
    v6: u64,
    v7: u64,
    v8: Option<V8Programs>,
    long_k: u32,
    attn: Option<AttnTcProgram>,
    fast: Option<AttnTcFast>,
    list_edit: &dyn Fn(&mut [MixedLaunch]),
    zone_edit: &ZoneEdit,
) -> Result<(Out, Vec<Option<TcPick>>), String> {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = plan_mini(&inputs);
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
        let list = plan
            .step_mixed(Recipe::AdamW513, &opt, step, LOSS_SCALE)
            .unwrap();
        let mut list = ts::attn0_tc_list(ts::attn_tc_list(list).unwrap()).unwrap();
        list_edit(&mut list);
        let (mut zone, qmds, p) = ts::pack_mixed_fast_attn8k(
            &list,
            prog_va,
            progs,
            v5,
            v6,
            v7,
            v8,
            long_k,
            attn,
            fast,
            A0,
            Some(64 * KIB),
            ZONE,
        )
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

fn no_edit(_: &mut [MixedLaunch]) {}

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

fn bits(x: &[f32], y: &[f32]) -> bool {
    x.len() == y.len() && x.iter().zip(y).all(|(p, r)| p.to_bits() == r.to_bits())
}

/// (v7, v8, v8n) among the picks.
fn v8_counts(p: &[Option<TcPick>]) -> [usize; 3] {
    [
        count(p, TcPick::V7),
        count(p, TcPick::V8),
        count(p, TcPick::V8n),
    ]
}

const FORCE_V8: V8Programs = V8Programs {
    rates: t8::Rates {
        v8_pct: 1000,
        v8n_pct: 100,
    },
    ..P8
};
const FORCE_V8N: V8Programs = V8Programs {
    rates: t8::Rates {
        v8_pct: 100,
        v8n_pct: 1000,
    },
    ..P8
};

/// N1: without v8 nothing changes: attn8 with `v8 = None` = attn7 byte for byte (mini and #513, with and without v7,
/// with and without the attention fast variant).
#[test]
fn t297_n1_without_v8_same_bytes() {
    let k = Some(64 * KIB);
    for l in [mini_list(), list_513(2)] {
        for v7 in [0, V7] {
            for fast in [None, FV4] {
                let a = ts::pack_mixed_fast_attn7(
                    &l, prog_va, &FAST, V5, V6, v7, AT, fast, A0, k, ZONE,
                )
                .unwrap();
                let b = ts::pack_mixed_fast_attn8(
                    &l, prog_va, &FAST, V5, V6, v7, None, AT, fast, A0, k, ZONE,
                )
                .unwrap();
                assert_eq!(a, b);
                assert_eq!(count(&b.2, TcPick::V8) + count(&b.2, TcPick::V8n), 0);
            }
        }
    }
    assert_eq!(TcPick::V8.name(), "mm_tc_v8");
    assert_eq!(TcPick::V8n.name(), "mm_tc_v8n");
    println!(
        "kt c297 N1 SIMULARE: fara mm_tc_v8, attn8 = attn7 octet cu octet (mini si #513, cu si fara v7)"
    );
}

/// N2: with v8 every launch that went to v7 takes `mm_tc_v8::pick` (by waves, 48 SM, the plan's rates); nothing else
/// moves; the zone keeps its length and QMD addresses; a v8 / v8n QMD has its program, registers, block, grid and
/// shared size, the cbuf in front of it is the same bytes. Lot 6 = the split of the plan (G7): v7 12, v8 38, v8n 97.
#[test]
fn t297_n2_picks_and_qmds() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let k = Some(64 * KIB);
    for (name, l, expect) in [
        ("mini", mini_list(), None),
        ("#513 lot 2", list_513(2), None),
        ("#513 lot 6", list_513(6), Some([12, 38, 97])),
    ] {
        let (za, qa, pa) =
            ts::pack_mixed_fast_attn7(&l, prog_va, &FAST, V5, V6, V7, AT, FV4, A0, k, ZONE)
                .unwrap();
        let (zb, qb, pb) = ts::pack_mixed_fast_attn8(
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
            ZONE,
        )
        .unwrap();
        assert_eq!(qa, qb);
        assert_eq!(za.len(), zb.len());
        let v7a = count(&pa, TcPick::V7);
        assert!(v7a > 0, "{name}");
        let got = v8_counts(&pb);
        assert_eq!(got.iter().sum::<usize>(), v7a, "{name}");
        if let Some(e) = expect {
            assert_eq!(got, e, "{name}");
        }
        for (((qv, x), y), launch) in qb.iter().zip(&pa).zip(&pb).zip(&l) {
            let at = (qv - ZONE) as usize;
            if *x != Some(TcPick::V7) {
                assert_eq!(x, y);
                assert_eq!(&za[at..at + 0x100], &zb[at..at + 0x100]);
                continue;
            }
            let MixedLaunch::Tc { args, .. } = launch else {
                panic!("{name}: V7 pe altceva decat Tc")
            };
            let want = match t8::pick(args, 48, t8::RATES_PLAN, true) {
                t8::Pick::V7 => TcPick::V7,
                t8::Pick::V8 => TcPick::V8,
                t8::Pick::V8n => TcPick::V8n,
            };
            assert_eq!(*y, Some(want), "{name}");
            // The cbuf (the same 48 B of arguments) in front of the QMD is unchanged.
            assert_eq!(&za[at - 0x800..at], &zb[at - 0x800..at]);
            let (va, regs, shared, block, grid) = match want {
                TcPick::V7 => {
                    assert_eq!(&za[at..at + 0x100], &zb[at..at + 0x100]);
                    continue;
                }
                TcPick::V8 => (V8, 128, 24_576, 256, t8::grid(args)),
                _ => (V8N, 144, 18_432, 128, t8::grid_n(args)),
            };
            assert_eq!(program_va(&zb, *qv), va);
            assert_eq!(field(&zb, *qv, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V), regs);
            assert_eq!(
                field(&zb, *qv, q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE),
                shared
            );
            assert_eq!(
                field(&zb, *qv, q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0),
                block
            );
            assert_eq!(
                field(&zb, *qv, q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH),
                u64::from(grid[0])
            );
            assert_eq!(
                field(&zb, *qv, q::NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT),
                u64::from(grid[1])
            );
        }
        println!(
            "kt c297 N2 SIMULARE: {name}: {v7a} x mm_tc_v7 -> v7 {} / v8 {} / v8n {} (alegerea pe valuri, 48 SM, vitezele planului DEDUS 140 / 126 %); zona {} B la fel",
            got[0],
            got[1],
            got[2],
            zb.len()
        );
    }
}

/// N3: the mini training (5 steps) with v8 / v8n = with v7, bit for bit — with the plan's rates, without v7, and with
/// the rates forced so that every aligned product goes to v8 (then to v8n).
#[test]
fn t297_n3_training_same_bits() {
    let (base, pb) = run(&FAST, V5, V6, V7, None, AT, None, &no_edit, NONE).unwrap();
    let v7n = count(&pb, TcPick::V7);
    assert!(v7n > 0);
    let (mut n8, mut n8n) = (0, 0);
    for (v7, p8) in [
        (V7, P8),
        (0, P8),
        (0, FORCE_V8),
        (0, FORCE_V8N),
        (V7, FORCE_V8N),
    ] {
        let (r, p) = run(&FAST, V5, V6, v7, Some(p8), AT, None, &no_edit, NONE).unwrap();
        let [c7, c8, c8n] = v8_counts(&p);
        assert_eq!(c7 + c8 + c8n, v7n);
        if v7 == 0 {
            assert_eq!(c7, 0);
        }
        if p8 == FORCE_V8 {
            assert_eq!(c8, v7n);
        }
        if p8 == FORCE_V8N && v7 == 0 {
            assert_eq!(c8n, v7n);
        }
        n8 += c8;
        n8n += c8n;
        for (a, b) in r.losses.iter().zip(&base.losses) {
            assert!(bits(a, b));
        }
        assert!(bits(&r.grad1, &base.grad1));
        assert!(bits(&r.weights, &base.weights));
    }
    assert!(n8 > 0 && n8n > 0);
    let first = mean(&base.losses[0]);
    let last = mean(&base.losses[STEPS as usize - 1]);
    assert!(last < first);
    println!(
        "kt c297 N3 SIMULARE: mini 5 pasi cu mm_tc_v8 / v8n ({n8} / {n8n} lansari in total pe 5 variante) = cu mm_tc_v7 bit cu bit; pierderea {first:.6} -> {last:.6}"
    );
}

/// N4: refusals. A v8 / v8n address not 256-aligned or 0, no SMs, a zero rate → refused before anything is built; a
/// v8 QMD whose shared size is lowered to v7's 14 336 B → the stand-in refuses (v8 needs 24 576 B).
#[test]
fn t297_n4_refusals() {
    use nvidia_rm570::regs::qmd_v03_00 as qq;
    let k = Some(64 * KIB);
    let l = mini_list();
    for bad8 in [
        V8Programs { v8: V8 + 16, ..P8 },
        V8Programs {
            v8n: V8N + 128,
            ..P8
        },
        V8Programs { v8: 0, ..P8 },
        V8Programs { v8n: 0, ..P8 },
        V8Programs { sms: 0, ..P8 },
        V8Programs {
            rates: t8::Rates {
                v8_pct: 0,
                v8n_pct: 126,
            },
            ..P8
        },
    ] {
        assert!(
            ts::pack_mixed_fast_attn8(
                &l,
                prog_va,
                &FAST,
                V5,
                V6,
                V7,
                Some(bad8),
                AT,
                FV4,
                A0,
                k,
                ZONE
            )
            .is_err()
        );
    }
    let (z7, q7, p7) =
        ts::pack_mixed_fast_attn7(&l, prog_va, &FAST, V5, V6, V7, AT, FV4, A0, k, ZONE).unwrap();
    let i7 = p7.iter().position(|p| *p == Some(TcPick::V7)).unwrap();
    let at7 = (q7[i7] - ZONE) as usize;
    let qmd7: Vec<u8> = z7[at7..at7 + 0x100].to_vec();
    let to_v7_shared = move |z: &mut Vec<u8>, q: &[u64], p: &[Option<TcPick>]| {
        let f = qq::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE;
        let (lo, hi) = ((f.lo / 8) as usize, (f.hi / 8) as usize);
        for (qva, pk) in q.iter().zip(p) {
            if *pk == Some(TcPick::V8) {
                let at = (qva - ZONE) as usize;
                z[at + lo..=at + hi].copy_from_slice(&qmd7[lo..=hi]);
            }
        }
    };
    let e = run(
        &FAST,
        V5,
        V6,
        0,
        Some(FORCE_V8),
        AT,
        None,
        &no_edit,
        &to_v7_shared,
    )
    .err()
    .expect("mm_tc_v8 cu memoria comuna a lui v7 trebuia refuzat");
    println!(
        "kt c297 N4 SIMULARE: adrese v8 nealiniate / 0, fara SM-uri, viteza 0 -> refuz; QMD v8 cu memoria comuna a lui v7 -> {e}"
    );
}

/// N1 / N2 through RAM: attn8 with `v8 = None` = attn7 byte for byte; with v8 the picks are those without RAM.
#[test]
fn t297_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let plan = plan_mini(&inputs).resident().unwrap();
    let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let list = mini_list();
    let k = Some(64 * KIB);
    let old = br::pack_mixed_fast_via_ram_attn7(
        &list, prog_va, &FAST, V5, V6, V7, AT, FV4, A0, k, TC, ZONE, &v,
    )
    .unwrap();
    let new = br::pack_mixed_fast_via_ram_attn8(
        &list, prog_va, &FAST, V5, V6, V7, None, AT, FV4, A0, k, TC, ZONE, &v,
    )
    .unwrap();
    assert_eq!(old, new);
    let (_, _, pr) = br::pack_mixed_fast_via_ram_attn8(
        &list,
        prog_va,
        &FAST,
        V5,
        V6,
        0,
        Some(FORCE_V8N),
        AT,
        FV4,
        A0,
        k,
        TC,
        ZONE,
        &v,
    )
    .unwrap();
    let (_, _, pd) = ts::pack_mixed_fast_attn8(
        &list,
        prog_va,
        &FAST,
        V5,
        V6,
        0,
        Some(FORCE_V8N),
        AT,
        FV4,
        A0,
        k,
        ZONE,
    )
    .unwrap();
    assert_eq!(pr, pd);
    assert!(count(&pr, TcPick::V8n) > 0);
    println!(
        "kt c297 prin RAM SIMULARE: fara v8 -> attn7 octet cu octet; cu v8 aceleasi alegeri ca fara RAM ({} x mm_tc_v8n)",
        count(&pr, TcPick::V8n)
    );
}

// ------------------------------------------------------------------ T304 (session 288): the long-sum guard

/// The measured rates of os-500, as the OS ran os-503.
const P8M: V8Programs = V8Programs {
    rates: t8::RATES_OS500,
    ..P8
};

fn tc_k(l: &MixedLaunch) -> Option<u32> {
    match l {
        MixedLaunch::Tc { args, .. } => Some(args.k),
        _ => None,
    }
}

/// T304 S1, SIMULARE: the guard off (`long_k = 0`), or with nothing to fall back on (v7 not loaded, or no v8) =
/// `pack_mixed_fast_attn8` byte for byte (mini, #513 lot 2 and lot 6).
#[test]
fn t304_s1_guard_off_same_bytes() {
    let k = Some(64 * KIB);
    for l in [mini_list(), list_513(2), list_513(6)] {
        for (v7, v8, long_k) in [
            (V7, Some(P8M), 0),
            (0, Some(P8M), t8::LONG_K_OS503),
            (V7, None, t8::LONG_K_OS503),
            (0, None, 0),
        ] {
            let a =
                ts::pack_mixed_fast_attn8(&l, prog_va, &FAST, V5, V6, v7, v8, AT, FV4, A0, k, ZONE)
                    .unwrap();
            let b = ts::pack_mixed_fast_attn8k(
                &l, prog_va, &FAST, V5, V6, v7, v8, long_k, AT, FV4, A0, k, ZONE,
            )
            .unwrap();
            assert_eq!(a, b);
        }
    }
    println!(
        "kt c304 S1 SIMULARE: garda oprita (long_k 0, fara v7 sau fara v8) -> attn8 octet cu octet (mini, #513 lot 2 si 6)"
    );
}

/// T304 S2, SIMULARE: with the guard (`LONG_K_OS503`) and the os-500 rates, exactly the launches with k ≥ 8 192 move
/// to v7 — in #513 the head's dX alone (k = 16 384); their QMD = v7's (program, registers, shared, grid), the cbuf is
/// the same bytes; nothing else moves. Lot 6: os-503's 12 / 0 / 135 → 13 / 0 / 134.
#[test]
fn t304_s2_picks() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let k = Some(64 * KIB);
    let mut lines = Vec::new();
    for (name, l) in [
        ("mini", mini_list()),
        ("#513 lot 2", list_513(2)),
        ("#513 lot 6", list_513(6)),
    ] {
        let (za, qa, pa) = ts::pack_mixed_fast_attn8(
            &l,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8M),
            AT,
            FV4,
            A0,
            k,
            ZONE,
        )
        .unwrap();
        let (zb, qb, pb) = ts::pack_mixed_fast_attn8k(
            &l,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8M),
            t8::LONG_K_OS503,
            AT,
            FV4,
            A0,
            k,
            ZONE,
        )
        .unwrap();
        assert_eq!(qa, qb);
        assert_eq!(za.len(), zb.len());
        let mut moved = 0;
        let mut long = 0;
        for (((qv, x), y), launch) in qb.iter().zip(&pa).zip(&pb).zip(&l) {
            let at = (qv - ZONE) as usize;
            let is_long = matches!(x, Some(TcPick::V7 | TcPick::V8 | TcPick::V8n))
                && tc_k(launch).is_some_and(|kk| kk >= t8::LONG_K_OS503);
            if !is_long {
                assert_eq!(x, y, "{name}");
                assert_eq!(&za[at..at + 0x100], &zb[at..at + 0x100], "{name}");
                continue;
            }
            long += 1;
            assert_eq!(tc_k(launch), Some(16_384), "{name}");
            assert_eq!(*y, Some(TcPick::V7), "{name}");
            assert_eq!(&za[at - 0x800..at], &zb[at - 0x800..at], "{name}");
            if *x == Some(TcPick::V7) {
                assert_eq!(&za[at..at + 0x100], &zb[at..at + 0x100]);
                continue;
            }
            moved += 1;
            let MixedLaunch::Tc { args, .. } = launch else {
                unreachable!()
            };
            assert_eq!(program_va(&zb, *qv), V7);
            assert_eq!(
                field(&zb, *qv, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V),
                u64::from(nvidia_ga104::mm_tc_v7::REGISTERS)
            );
            assert_eq!(
                field(&zb, *qv, q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE),
                u64::from(nvidia_ga104::mm_tc_v7::SHARED.bytes)
            );
            assert_eq!(
                field(&zb, *qv, q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH),
                u64::from(nvidia_ga104::mm_tc_v7::grid(args)[0])
            );
        }
        // One head, one dX with the vocabulary as its sum.
        assert_eq!(long, 1, "{name}");
        let a = v8_counts(&pa);
        let b = v8_counts(&pb);
        if name == "#513 lot 6" {
            assert_eq!(a, [12, 0, 135]);
            assert_eq!(b, [13, 0, 134]);
            assert_eq!(moved, 1);
        }
        if name == "#513 lot 2" {
            assert_eq!(a, [73, 24, 50]);
        }
        lines.push(format!(
            "{name} {}/{}/{} -> {}/{}/{}",
            a[0], a[1], a[2], b[0], b[1], b[2]
        ));
    }
    println!(
        "kt c304 S2 SIMULARE: garda k >= {}: doar dX-ul capului (k 16384) trece pe v7, QMD-ul lui v7, cbuf neschimbat; v7/v8/v8n {}",
        t8::LONG_K_OS503,
        lines.join("; ")
    );
}

/// T304 S3, SIMULARE: five mini steps with the guard = without it, bit for bit (losses, first gradient, weights).
#[test]
fn t304_s3_training_same_bits() {
    let (base, pb) = run(&FAST, V5, V6, V7, Some(P8M), AT, None, &no_edit, NONE).unwrap();
    let (r, p) = run_k(
        &FAST,
        V5,
        V6,
        V7,
        Some(P8M),
        t8::LONG_K_OS503,
        AT,
        None,
        &no_edit,
        NONE,
    )
    .unwrap();
    for (a, b) in r.losses.iter().zip(&base.losses) {
        assert!(bits(a, b));
    }
    assert!(bits(&r.grad1, &base.grad1));
    assert!(bits(&r.weights, &base.weights));
    let (a, b) = (v8_counts(&pb), v8_counts(&p));
    assert_eq!(a.iter().sum::<usize>(), b.iter().sum::<usize>());
    assert_eq!(b[0], a[0] + 1);
    println!(
        "kt c304 S3 SIMULARE: mini 5 pasi cu garda (v7/v8/v8n {}/{}/{} -> {}/{}/{}) = fara garda bit cu bit",
        a[0], a[1], a[2], b[0], b[1], b[2]
    );
}

/// T304 S4, SIMULARE: through RAM — `long_k = 0` = `via_ram_attn8` byte for byte; with the guard the picks are those of
/// `pack_mixed_fast_attn8k`.
#[test]
fn t304_s4_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let plan = plan_mini(&inputs).resident().unwrap();
    let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let list = mini_list();
    let k = Some(64 * KIB);
    let old = br::pack_mixed_fast_via_ram_attn8(
        &list,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8M),
        AT,
        FV4,
        A0,
        k,
        TC,
        ZONE,
        &v,
    )
    .unwrap();
    let off = br::pack_mixed_fast_via_ram_attn8k(
        &list,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8M),
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
    assert_eq!(old, off);
    let (_, _, pr) = br::pack_mixed_fast_via_ram_attn8k(
        &list,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8M),
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
    let (_, _, pd) = ts::pack_mixed_fast_attn8k(
        &list,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8M),
        t8::LONG_K_OS503,
        AT,
        FV4,
        A0,
        k,
        ZONE,
    )
    .unwrap();
    assert_eq!(pr, pd);
    assert_eq!(count(&pr, TcPick::V7), count(&old.2, TcPick::V7) + 1);
    println!(
        "kt c304 S4 prin RAM SIMULARE: long_k 0 -> via_ram_attn8 octet cu octet; cu garda aceleasi alegeri ca fara RAM"
    );
}
