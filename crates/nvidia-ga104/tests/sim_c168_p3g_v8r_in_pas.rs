//! C-168 p.3 P3g (session 292) — T309, SIMULARE: `mm_tc_v8r` in the fast mixed step where the waves pick `mm_tc_v8`
//! (`TcPick::V8r`, `pack_mixed_fast_attn11`, `pack_mixed_fast_via_ram_attn11`). The plan and the predictions K1–K6 are
//! in `docs/C-168_V8R.md` §6 (written before this test).
//!
//! Honesty: everything here runs on the simulated board (`gpu_sim`); `mm_tc_v8r` passed the board probe (os-505, V1)
//! but has not run inside a step on the board.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, ATTN_FWD_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS,
    LN_BACK_RES_NEEDS, LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS,
    MM_TC_V5_NEEDS, MM_TC_V6_NEEDS, MM_TC_V7_GELU_NEEDS, MM_TC_V7_NEEDS, MM_TC_V8_GELU_NEEDS,
    MM_TC_V8_NEEDS, MM_TC_V8N_GELU_NEEDS, MM_TC_V8N_NEEDS, MM_TC_V8R_NEEDS, MM_TRAIN_GELU_NEEDS,
    MM_TRAIN_NEEDS, Needs, XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu,
    attn_back_tile_tc16_on_cpu, attn_back_tile_tc16_v4_on_cpu, attn_back_tile_tc16_v4pf_on_cpu,
    attn_back_tile0_on_cpu, attn_back_tile0_tc_on_cpu, attn_fwd_on_cpu, attn_fwd_tc_on_cpu,
    attn_fwd_tile_on_cpu, attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, colsum_part_on_cpu,
    emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu,
    ln_back_res_on_cpu, ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu,
    mm_tc_v4_on_cpu, mm_tc_v5_on_cpu, mm_tc_v6_on_cpu, mm_tc_v7_gelu_on_cpu, mm_tc_v7_on_cpu,
    mm_tc_v8_gelu_on_cpu, mm_tc_v8_on_cpu, mm_tc_v8n_gelu_on_cpu, mm_tc_v8n_on_cpu,
    mm_tc_v8r_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu,
    xent_on_cpu,
};
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::attn_back_tile_tc16_v4::ATTN_BACK_TILE_TC16_V4;
use nvidia_ga104::attn_back_tile_tc16_v4pf::ATTN_BACK_TILE_TC16_V4PF;
use nvidia_ga104::attn_back_tile0_tc::ATTN_BACK_TILE0_TC;
use nvidia_ga104::attn_fwd_tc::ATTN_FWD_TC;
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_fast_gelu::{MM_TC_V7_GELU, MM_TC_V8_GELU, MM_TC_V8N_GELU};
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::mm_tc_pf::MM_TC_PF;
use nvidia_ga104::mm_tc_v4::MM_TC_V4;
use nvidia_ga104::mm_tc_v5::MM_TC_V5;
use nvidia_ga104::mm_tc_v6::MM_TC_V6;
use nvidia_ga104::mm_tc_v7::MM_TC_V7;
use nvidia_ga104::mm_tc_v8::{self as t8, MM_TC_V8, MM_TC_V8N};
use nvidia_ga104::mm_tc_v8r::{self as t8r, MM_TC_V8R};
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
const V8R: u64 = PROGS + 0x1_0000 * 44;
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
    cp.register_program_with_needs(MM_TC_V8R, mm_tc_v8r_on_cpu, MM_TC_V8R_NEEDS);
    cp.mem.write(V8R, MM_TC_V8R);
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

type Packed = (Vec<u8>, Vec<u64>, Vec<Option<TcPick>>);

/// `pack_mixed_fast_attn11` with `v8r_va` (0 = `attn10`), the os-503 guard, the attention on the tensor cores.
fn pack11(
    list: &[MixedLaunch],
    gelu: Option<GeluFast>,
    p8: Option<V8Programs>,
    v8r_va: u64,
) -> Result<Packed, String> {
    ts::pack_mixed_fast_attn11(
        list,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        p8,
        t8::LONG_K_OS503,
        AFT,
        gelu,
        v8r_va,
        AT,
        FV4,
        A0,
        Some(64 * KIB),
        ZONE,
    )
    .map_err(|e| format!("{e:?}"))
}

fn pack10(list: &[MixedLaunch], gelu: Option<GeluFast>, p8: Option<V8Programs>) -> Packed {
    ts::pack_mixed_fast_attn10(
        list,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        p8,
        t8::LONG_K_OS503,
        AFT,
        gelu,
        AT,
        FV4,
        A0,
        Some(64 * KIB),
        ZONE,
    )
    .unwrap()
}

fn count(picks: &[Option<TcPick>], p: TcPick) -> usize {
    picks.iter().filter(|x| **x == Some(p)).count()
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

/// Rates that make the waves give v8 every aligned product (to exercise v8 / v8r on the mini model).
const P8_V8: V8Programs = V8Programs {
    rates: t8::Rates {
        v8_pct: 400,
        v8n_pct: 118,
    },
    sms: 1,
    ..P8
};

struct Out {
    losses: Vec<Vec<f32>>,
    grad1: Vec<f32>,
    weights: Vec<f32>,
    picks: Vec<Option<TcPick>>,
}

/// `STEPS` fast mixed steps of the aligned mini model (P5a plan + `with_fused_gelu`, the GELU' variants) on the
/// simulated board, packed by `attn11` with `p8` and `v8r_va`.
fn run(p8: V8Programs, v8r_va: u64) -> Out {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = plan_mini(&inputs, true);
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
        picks: Vec::new(),
    };
    for step in 1..=STEPS {
        let list = plan
            .step_mixed(Recipe::AdamW513, &opt, step, scale())
            .unwrap();
        let list =
            ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(list).unwrap()).unwrap())
                .unwrap();
        let (zone, qmds, picks) = pack11(&list, Some(GF), Some(p8), v8r_va).unwrap();
        out.picks = picks;
        submit(&mut cp, &mut first, &zone, &qmds).unwrap();
        out.losses
            .push(cp.mem.read_f32s(plan.loss_rows, B * T).unwrap());
        if step == 1 {
            out.grad1 = cp.mem.read_f32s(plan.g, plan.n_params).unwrap();
        }
    }
    out.weights = cp.mem.read_f32s(plan.w, plan.n_params).unwrap();
    out
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

/// T309 S1 (K1), SIMULARE: the mini model, 5 fast mixed steps, with the rates steered so that the waves give v8 every
/// aligned product: on v8r (`attn11`) the same bits as on v8 (`attn10`) and as with the os-500 rates; every V8 → V8r.
#[test]
fn t309_s1_mini_same_bits() {
    let v8 = run(P8_V8, 0);
    let v8r = run(P8_V8, V8R);
    let normal = run(P8, 0);
    same(&v8, &v8r, "v8 fata de v8r");
    same(&normal, &v8r, "alegerea os-500 fata de v8r");
    let n8 = count(&v8.picks, TcPick::V8);
    assert!(n8 > 0, "vitezele fortate dau v8");
    assert_eq!(count(&v8r.picks, TcPick::V8r), n8);
    assert_eq!(count(&v8r.picks, TcPick::V8), 0);
    // everything else identical
    let other = |p: &[Option<TcPick>]| -> Vec<Option<TcPick>> {
        p.iter()
            .map(|x| match x {
                Some(TcPick::V8r) => Some(TcPick::V8),
                y => *y,
            })
            .collect()
    };
    assert_eq!(other(&v8r.picks), v8.picks);
    assert!(v8r.weights.iter().all(|x| x.is_finite()));
    println!(
        "kt c309 S1 SIMULARE: modelul mic, 5 pasi rapizi: {n8} produse pe v8r = aceiasi biti ca pe v8 si ca alegerea os-500 (pierderile, gradientul pasului 1, greutatile)"
    );
}

/// T309 S2 / S3 (K2, K3), SIMULARE: #513 lot 2 and lot 6, the P5a list with and without the P5b GELU' fusion: exactly
/// the launches the waves give v8 move to v8r (lot 2 without P5b: 24), nothing else changes; lot 6 has none (zone
/// byte for byte the same as `attn10`). Each V8r QMD: v8r's address, 120 registers, block 256, v8's grid.
#[test]
fn t309_s2_513_lists() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let mut lines = Vec::new();
    for gelu in [false, true] {
        let gf = gelu.then_some(GF);
        for bsz in [2usize, 6] {
            let l = list_513(bsz, gelu);
            let (za, qa, pa) = pack10(&l, gf, Some(P8));
            let (zb, qb, pb) = pack11(&l, gf, Some(P8), V8R).unwrap();
            let n8 = count(&pa, TcPick::V8);
            assert_eq!(count(&pb, TcPick::V8r), n8);
            assert_eq!(count(&pb, TcPick::V8), 0);
            assert_eq!(qa, qb, "aceleasi locuri in zona");
            for (i, (a, b)) in pa.iter().zip(&pb).enumerate() {
                if *a == Some(TcPick::V8) {
                    assert_eq!(*b, Some(TcPick::V8r));
                    let MixedLaunch::Tc { args, .. } = &l[i] else {
                        panic!("V8 pe altceva decat Tc")
                    };
                    let at = (qb[i] - ZONE) as usize;
                    let qmd = &zb[at..at + 0x100];
                    let va = qmd_field(qmd, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER) << 32
                        | qmd_field(qmd, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER);
                    assert_eq!(va, V8R);
                    assert_eq!(
                        qmd_field(qmd, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V),
                        u64::from(t8r::REGISTERS)
                    );
                    assert_eq!(
                        qmd_field(qmd, q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0),
                        256
                    );
                    let g = t8r::grid(args);
                    assert_eq!(g, t8::grid(args));
                    assert_eq!(
                        qmd_field(qmd, q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH),
                        u64::from(g[0])
                    );
                    assert_eq!(
                        qmd_field(qmd, q::NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT),
                        u64::from(g[1])
                    );
                    // the cbuf before the QMD is the same as v8's
                    let c = nvidia_ga104::mm_tc::CBUF_BYTES;
                    assert_eq!(&zb[at - c..at], &za[at - c..at]);
                } else {
                    assert_eq!(a, b);
                    // every other launch: the same bytes
                    let at = (qb[i] - ZONE) as usize;
                    assert_eq!(&zb[at..at + 0x100], &za[at..at + 0x100], "lansarea {i}");
                }
            }
            if !gelu && bsz == 2 {
                assert_eq!(n8, 24, "K2: lot 2 fara P5b");
            }
            if bsz == 6 {
                assert_eq!(n8, 0, "K3: lot 6 fara v8");
                assert_eq!(za, zb, "K3: lot 6 octet cu octet");
            }
            lines.push(format!(
                "lot {bsz}{}: {n8} V8 -> V8r",
                if gelu { " cu P5b" } else { "" }
            ));
        }
    }
    println!(
        "kt c309 S2 SIMULARE: #513 (lista P5a): doar lansarile date lui v8 trec pe v8r, restul octet cu octet; {}",
        lines.join("; ")
    );
}

/// T309 S4 (K4), SIMULARE: `v8r_va = 0` = `attn10` byte for byte; refusals before anything is sent; one aligned
/// product forced onto v8r gives the same bits as on v8.
#[test]
fn t309_s4_refusals_and_one_product() {
    let l = list_513(2, false);
    let a10 = pack10(&l, None, Some(P8));
    assert_eq!(pack11(&l, None, Some(P8), 0).unwrap(), a10);
    assert!(pack11(&l, None, Some(P8), V8R + 4).is_err(), "nealiniat");
    let e = pack11(&l, None, None, V8R).unwrap_err();
    assert!(e.contains("mm_tc_v8r fara"), "{e}");
    // one product, 256 × 256 × 128 (TB), forced onto v8 / v8r
    let args = nvidia_ga104::mm_tc::Args {
        a: HEAP,
        b: HEAP + 0x10_0000,
        c: HEAP + 0x20_0000,
        bias: 0,
        m: 256,
        n: 256,
        k: 128,
        flags: nvidia_ga104::mm_tc::TB,
    };
    let one = vec![MixedLaunch::Tc {
        args,
        tag: "proba",
        layer: None,
    }];
    let fill = |cp: &mut CommandProcessor| {
        let mut r = Rng(0x0c16_8f5c_2026_1128);
        let mut f = |n: usize| -> Vec<u8> {
            (0..n)
                .flat_map(|_| ((r.unit() * 2.0 - 1.0) as f32).to_le_bytes())
                .collect()
        };
        cp.mem.write(args.a, &f(256 * 128));
        cp.mem.write(args.b, &f(256 * 128));
        cp.mem.write(args.c, &vec![0xffu8; 256 * 256 * 4]);
    };
    let mut outs = Vec::new();
    for v8r in [0, V8R] {
        let (zone, qmds, p) = pack11(&one, None, Some(P8_V8), v8r).unwrap();
        assert_eq!(p[0], Some(if v8r == 0 { TcPick::V8 } else { TcPick::V8r }));
        let mut cp = board();
        fill(&mut cp);
        let mut first = 1u32;
        submit(&mut cp, &mut first, &zone, &qmds).unwrap();
        outs.push(bits(&cp.mem.read_f32s(args.c, 256 * 256).unwrap()));
    }
    assert_eq!(outs[0], outs[1]);
    assert!(outs[1].iter().all(|x| f32::from_bits(*x).is_finite()));
    println!(
        "kt c309 S4 SIMULARE: v8r_va = 0 -> attn10 octet cu octet; nealiniat / v8r fara v8 refuzate; produsul 256x256x128 pe v8r = aceiasi biti ca pe v8"
    );
}

/// T309 S5 (K4), SIMULARE: through RAM — `v8r_va = 0` = `via_ram_attn10` byte for byte; with v8r the picks and the
/// list part of the zone are those of `pack_mixed_fast_attn11`.
#[test]
fn t309_s5_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let plan = plan_mini(&inputs, true).resident().unwrap();
    let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let k = Some(64 * KIB);
    let opt = AdamW::new(&initial(), LR, 0.9, 0.999, 1e-8, 0.1);
    let l = plan.step_mixed(Recipe::AdamW513, &opt, 1, scale()).unwrap();
    let l = ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()).unwrap();
    let via = |r: u64| {
        br::pack_mixed_fast_via_ram_attn11(
            &l,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8_V8),
            t8::LONG_K_OS503,
            AFT,
            Some(GF),
            r,
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
    let old = br::pack_mixed_fast_via_ram_attn10(
        &l,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8_V8),
        t8::LONG_K_OS503,
        AFT,
        Some(GF),
        AT,
        FV4,
        A0,
        k,
        TC,
        ZONE,
        &v,
    )
    .unwrap();
    assert_eq!(via(0), old);
    let (zr, _, pr) = via(V8R);
    let list_va = ZONE + ts::SLOT * v.copies_in.len() as u64;
    let (zd, _, pd) = ts::pack_mixed_fast_attn11(
        &l,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8_V8),
        t8::LONG_K_OS503,
        AFT,
        Some(GF),
        V8R,
        AT,
        FV4,
        A0,
        k,
        list_va,
    )
    .unwrap();
    assert_eq!(pr, pd);
    assert!(count(&pr, TcPick::V8r) > 0);
    let off = (list_va - ZONE) as usize;
    assert_eq!(&zr[off..off + zd.len()], &zd[..]);
    println!(
        "kt c309 S5 prin RAM SIMULARE: v8r_va = 0 -> via_ram_attn10 octet cu octet; cu v8r aceleasi alegeri si aceeasi lista ca fara RAM"
    );
}
