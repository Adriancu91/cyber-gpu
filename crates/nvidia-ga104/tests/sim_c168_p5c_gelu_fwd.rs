//! C-168 p.5 P5c (session 293) — T310, SIMULARE: GELU's forward fused into the store of "fc" (`mm_tc_v7_geluf` /
//! `mm_tc_v8_geluf` / `mm_tc_v8n_geluf`, `Plan::with_fused_gelu_fwd`, `MixedLaunch::TcGeluFwd`,
//! `pack_mixed_fast_attn12`, `pack_mixed_fast_via_ram_attn12`). The plan and the predictions L1–L7 are in
//! `docs/C-168_P5_LIPIRE_PLAN.md` §9 (written before this test, session 292).
//!
//! Honesty: everything here runs on the simulated board (`gpu_sim`); none of the three `geluf` programs has run on a
//! board.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, ATTN_FWD_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS,
    LN_BACK_RES_NEEDS, LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS,
    MM_TC_V5_NEEDS, MM_TC_V6_NEEDS, MM_TC_V7_GELU_NEEDS, MM_TC_V7_GELUF_NEEDS, MM_TC_V7_NEEDS,
    MM_TC_V8_GELU_NEEDS, MM_TC_V8_GELUF_NEEDS, MM_TC_V8_NEEDS, MM_TC_V8N_GELU_NEEDS,
    MM_TC_V8N_GELUF_NEEDS, MM_TC_V8N_NEEDS, MM_TC_V8R_NEEDS, MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS,
    Needs, XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu, attn_back_tile_tc16_on_cpu,
    attn_back_tile_tc16_v4_on_cpu, attn_back_tile_tc16_v4pf_on_cpu, attn_back_tile0_on_cpu,
    attn_back_tile0_tc_on_cpu, attn_fwd_on_cpu, attn_fwd_tc_on_cpu, attn_fwd_tile_on_cpu,
    attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, colsum_part_on_cpu, emb_back_on_cpu,
    emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu,
    ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu, mm_tc_v4_on_cpu,
    mm_tc_v5_on_cpu, mm_tc_v6_on_cpu, mm_tc_v7_gelu_on_cpu, mm_tc_v7_geluf_on_cpu, mm_tc_v7_on_cpu,
    mm_tc_v8_gelu_on_cpu, mm_tc_v8_geluf_on_cpu, mm_tc_v8_on_cpu, mm_tc_v8n_gelu_on_cpu,
    mm_tc_v8n_geluf_on_cpu, mm_tc_v8n_on_cpu, mm_tc_v8r_on_cpu, mm_train_gelu_on_cpu,
    mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::attn_back_tile_tc16_v4::ATTN_BACK_TILE_TC16_V4;
use nvidia_ga104::attn_back_tile_tc16_v4pf::ATTN_BACK_TILE_TC16_V4PF;
use nvidia_ga104::attn_back_tile0_tc::ATTN_BACK_TILE0_TC;
use nvidia_ga104::attn_fwd_tc::ATTN_FWD_TC;
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_fast_gelu::{MM_TC_V7_GELU, MM_TC_V8_GELU, MM_TC_V8N_GELU};
use nvidia_ga104::mm_tc_fast_gelu_fwd::{
    self as ff, MM_TC_V7_GELUF, MM_TC_V8_GELUF, MM_TC_V8N_GELUF,
};
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::mm_tc_pf::MM_TC_PF;
use nvidia_ga104::mm_tc_v4::MM_TC_V4;
use nvidia_ga104::mm_tc_v5::MM_TC_V5;
use nvidia_ga104::mm_tc_v6::MM_TC_V6;
use nvidia_ga104::mm_tc_v7::MM_TC_V7;
use nvidia_ga104::mm_tc_v8::{self as t8, MM_TC_V8, MM_TC_V8N};
use nvidia_ga104::mm_tc_v8r::MM_TC_V8R;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, Attn0Programs, AttnTcFast, AttnTcProgram, GeluFast, GeluFwdFast, LOSS_SCALE_513,
    MixedLaunch, PROGRAMS_ALL_FUSED2, Plan, Prog, TcPick, TcPrograms, V8Programs,
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
const F7: u64 = PROGS + 0x1_0000 * 48;
const F8: u64 = PROGS + 0x1_0000 * 50; // the geluf programs are ~70 KiB: two 64 KiB slots each
const F8N: u64 = PROGS + 0x1_0000 * 52;
const GFF: GeluFwdFast = GeluFwdFast {
    v7: F7,
    v8: F8,
    v8n: F8N,
};
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
    cp.register_program_with_needs(MM_TC_V7_GELUF, mm_tc_v7_geluf_on_cpu, MM_TC_V7_GELUF_NEEDS);
    cp.mem.write(F7, MM_TC_V7_GELUF);
    cp.register_program_with_needs(MM_TC_V8_GELUF, mm_tc_v8_geluf_on_cpu, MM_TC_V8_GELUF_NEEDS);
    cp.mem.write(F8, MM_TC_V8_GELUF);
    cp.register_program_with_needs(
        MM_TC_V8N_GELUF,
        mm_tc_v8n_geluf_on_cpu,
        MM_TC_V8N_GELUF_NEEDS,
    );
    cp.mem.write(F8N, MM_TC_V8N_GELUF);
    let csp = Prog::ColsumPart.code();
    cp.register_program(csp, colsum_part_on_cpu);
    cp.mem.write(CSP, csp);
    cp
}

fn scale() -> f32 {
    ts::loss_scale_for(MINI.vocab, B * T)
}

/// The P5b plan shape on the mini model (`new_tiles3` + `with_colsum_split` + `with_fused_back` + `with_fused_gelu`),
/// + `with_fused_gelu_fwd` when `fwd`.
fn plan_mini(inputs: &[u32], fwd: bool) -> Plan {
    let p = Plan::new_tiles3(MINI, B, T, inputs, HEAP)
        .unwrap()
        .with_colsum_split()
        .with_fused_back()
        .with_fused_gelu();
    if fwd { p.with_fused_gelu_fwd() } else { p }
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

/// The #513 list of step 1 as the OS would send it after P5b (`new_tiles3` + `with_mm_tile` + `with_colsum_split` +
/// `with_fused_back` + `with_fused_gelu`, the attention on the tensor cores), + `with_fused_gelu_fwd` when `fwd`.
fn list_513(bsz: usize, fwd: bool) -> (Plan, Vec<MixedLaunch>) {
    let m = train::M513;
    let inputs: Vec<u32> = (0..bsz as u32 * 512).map(|i| (i * 7919) % 16_384).collect();
    let base = VA + (1 << 30);
    let plan = Plan::new_tiles3(m, bsz, 512, &inputs, base)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split()
        .with_fused_back()
        .with_fused_gelu();
    let plan = if fwd {
        plan.with_fused_gelu_fwd()
    } else {
        plan
    };
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    let l = plan
        .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap();
    let l = ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()).unwrap();
    (plan, l)
}

type Packed = (Vec<u8>, Vec<u64>, Vec<Option<TcPick>>);

/// `pack_mixed_fast_attn12` with the P5b GELU' variants, `v8r_va`, `gelu_fwd`, the os-503 guard, the attention on the
/// tensor cores.
fn pack12(
    list: &[MixedLaunch],
    p8: Option<V8Programs>,
    v8r_va: u64,
    gelu_fwd: Option<GeluFwdFast>,
) -> Result<Packed, String> {
    ts::pack_mixed_fast_attn12(
        list,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        p8,
        t8::LONG_K_OS503,
        AFT,
        Some(GF),
        v8r_va,
        gelu_fwd,
        AT,
        FV4,
        A0,
        Some(64 * KIB),
        ZONE,
    )
    .map_err(|e| format!("{e:?}"))
}

fn pack11(list: &[MixedLaunch], p8: Option<V8Programs>, v8r_va: u64) -> Result<Packed, String> {
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
        Some(GF),
        v8r_va,
        AT,
        FV4,
        A0,
        Some(64 * KIB),
        ZONE,
    )
    .map_err(|e| format!("{e:?}"))
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

/// Rates that make the waves give v8 every aligned product (to exercise v8 / v8_geluf on the mini model).
const P8_V8: V8Programs = V8Programs {
    rates: t8::Rates {
        v8_pct: 400,
        v8n_pct: 118,
    },
    sms: 1,
    ..P8
};

/// The `geluf` variant of a plain pick (what P5c must give the fused "fc").
fn geluf_of(p: Option<TcPick>) -> Option<TcPick> {
    match p {
        Some(TcPick::V7) => Some(TcPick::V7GeluF),
        Some(TcPick::V8 | TcPick::V8r) => Some(TcPick::V8GeluF),
        Some(TcPick::V8n) => Some(TcPick::V8nGeluF),
        other => panic!("fc pe {other:?}: nu are varianta geluf"),
    }
}

struct Out {
    losses: Vec<Vec<f32>>,
    grad1: Vec<f32>,
    weights: Vec<f32>,
    picks: Vec<Option<TcPick>>,
    launches: usize,
}

/// `STEPS` fast mixed steps of the aligned mini model (P5b plan, + `with_fused_gelu_fwd` when `fwd`) on the simulated
/// board, packed by `attn12` with `p8`, `v8r_va` and the `geluf` programs.
fn run(p8: V8Programs, v8r_va: u64, fwd: bool) -> Out {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = plan_mini(&inputs, fwd);
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
        launches: 0,
    };
    for step in 1..=STEPS {
        let list = plan
            .step_mixed(Recipe::AdamW513, &opt, step, scale())
            .unwrap();
        assert_eq!(
            list.len(),
            plan.launches_per_step_mixed(Recipe::AdamW513),
            "launches_per_step_mixed"
        );
        let list =
            ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(list).unwrap()).unwrap())
                .unwrap();
        let (zone, qmds, picks) = pack12(&list, Some(p8), v8r_va, Some(GFF)).unwrap();
        out.picks = picks;
        out.launches = list.len();
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

/// T310 S1 (L4), SIMULARE: the mini model, 5 fast mixed steps (P5a + P5b): with `with_fused_gelu_fwd` the same bits
/// as without (losses, the gradient of step 1, the weights), one launch fewer per layer, "fc" on the `geluf` variant —
/// with the os-500 rates and with the rates steered to v8 (v8r loaded: plain products on v8r, "fc" on v8_geluf).
#[test]
fn t310_s1_mini_same_bits() {
    let mut lines = Vec::new();
    for (p8, v8r, name) in [(P8, 0, "os-500"), (P8_V8, V8R, "v8 + v8r")] {
        let a = run(p8, v8r, false);
        let b = run(p8, v8r, true);
        same(&a, &b, name);
        assert_eq!(b.launches + MINI.layers, a.launches, "{name}: -L lansari");
        let nf = count(&b.picks, TcPick::V7GeluF)
            + count(&b.picks, TcPick::V8GeluF)
            + count(&b.picks, TcPick::V8nGeluF);
        assert_eq!(nf, MINI.layers, "{name}: fc pe geluf");
        assert!(b.weights.iter().all(|x| x.is_finite()));
        lines.push(format!(
            "{name}: {} -> {} lansari, v7_geluf {} / v8_geluf {} / v8n_geluf {}",
            a.launches,
            b.launches,
            count(&b.picks, TcPick::V7GeluF),
            count(&b.picks, TcPick::V8GeluF),
            count(&b.picks, TcPick::V8nGeluF)
        ));
    }
    println!(
        "kt c310 S1 SIMULARE: modelul mic, 5 pasi rapizi P5a+P5b, cu fc+gelu = aceiasi biti ca fara (pierderile, gradientul pasului 1, greutatile); {}",
        lines.join("; ")
    );
}

/// T310 S2 (L5), SIMULARE: #513 lot 2 and lot 6 (the P5b list, v8r loaded): exactly −12 launches (the 12 forward
/// `gelu_tr`), each "fc" on the `geluf` variant of the program it had (lot 6: v8n_geluf), every other pick identical
/// and in order, every other launch's cbuf byte for byte; each fused QMD: the variant's address, registers, block, grid;
/// its cbuf = the plain "fc"'s with `act` (the old `gelu_tr`'s output) at 0x730.
#[test]
fn t310_s2_513_lists() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let mut lines = Vec::new();
    for bsz in [2usize, 6] {
        let (pa, la) = list_513(bsz, false);
        let (pb, lb) = list_513(bsz, true);
        assert_eq!(la.len(), lb.len() + 12, "L5: lot {bsz} exact -12 lansari");
        assert_eq!(la.len(), pa.launches_per_step_mixed(Recipe::AdamW513));
        assert_eq!(lb.len(), pb.launches_per_step_mixed(Recipe::AdamW513));
        let (za, qa, ka) = pack12(&la, Some(P8), V8R, Some(GFF)).unwrap();
        let (zb, qb, kb) = pack12(&lb, Some(P8), V8R, Some(GFF)).unwrap();
        assert_eq!(
            pack11(&la, Some(P8), V8R).unwrap(),
            (za.clone(), qa.clone(), ka.clone())
        );
        let (mut i, mut j, mut fused) = (0usize, 0usize, Vec::new());
        while i < la.len() {
            let (a, b) = (&la[i], &lb[j]);
            let ca = (qa[i] - ZONE) as usize;
            let cb_at = (qb[j] - ZONE) as usize;
            if let MixedLaunch::TcGeluFwd { args, tag, layer } = b {
                let MixedLaunch::Tc {
                    args: fa,
                    tag: ta,
                    layer: lya,
                } = a
                else {
                    panic!("lansarea {i}: fc+gelu fara fc")
                };
                assert_eq!((*ta, *tag), ("inainte:fc", "inainte:fc+gelu"));
                assert_eq!(layer, lya);
                assert_eq!(args.tc, *fa);
                let MixedLaunch::Std(g) = &la[i + 1] else {
                    panic!("dupa fc nu e gelu")
                };
                assert_eq!(g.prog, Prog::GeluTr);
                assert_eq!(g.tag, "inainte:gelu");
                assert_eq!(args.act.to_le_bytes(), g.cbuf[0x38..0x40]);
                assert_eq!(kb[j], geluf_of(ka[i]), "lansarea {i}");
                fused.push(kb[j].unwrap());
                // the cbuf: the plain fc's with act at 0x730
                let c = nvidia_ga104::mm_tc::CBUF_BYTES;
                let (cba, cbb) = (&za[ca - c..ca], &zb[cb_at - c..cb_at]);
                assert_eq!(cba[..ff::ACT_AT], cbb[..ff::ACT_AT]);
                assert_eq!(cbb[ff::ACT_AT..ff::ACT_AT + 8], args.act.to_le_bytes());
                assert!(cba[ff::ACT_AT..].iter().all(|&x| x == 0));
                // the QMD
                let qmd = &zb[cb_at..cb_at + 0x100];
                let va = qmd_field(qmd, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER) << 32
                    | qmd_field(qmd, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER);
                let (pva, regs, block, grid) = match kb[j] {
                    Some(TcPick::V8nGeluF) => (F8N, ff::REGISTERS_V8N, 128, t8::grid_n(fa)),
                    Some(TcPick::V8GeluF) => (F8, ff::REGISTERS_V8, 256, t8::grid(fa)),
                    _ => (F7, ff::REGISTERS_V7, 128, fa.grid()),
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
                // the plain QMD of that fc had the same grid and block (same program family)
                let qa_ = &za[ca..ca + 0x100];
                for fl in [
                    q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH,
                    q::NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT,
                    q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0,
                ] {
                    assert_eq!(qmd_field(qa_, fl), qmd_field(qmd, fl));
                }
                i += 2;
                j += 1;
                continue;
            }
            assert_eq!(a.tag(), b.tag(), "lansarea {i}");
            assert_eq!(ka[i], kb[j], "lansarea {i}: alegerea");
            // the cbuf before the QMD: the same bytes (the QMDs differ in where their cbuf is)
            let c = (a.slot() - 0x100) as usize;
            assert_eq!(&za[ca - c..ca], &zb[cb_at - c..cb_at], "lansarea {i}: cbuf");
            i += 1;
            j += 1;
        }
        assert_eq!(j, lb.len());
        assert_eq!(fused.len(), 12);
        if bsz == 6 {
            assert!(
                fused.iter().all(|p| *p == TcPick::V8nGeluF),
                "lot 6: v8n_geluf"
            );
        }
        lines.push(format!(
            "lot {bsz}: {} -> {} lansari, fc pe {:?}",
            la.len(),
            lb.len(),
            fused[0]
        ));
        assert!(fused.iter().all(|p| *p == fused[0]));
    }
    println!(
        "kt c310 S2 SIMULARE: #513 (lista P5b + v8r): exact -12 lansari, fc pe varianta geluf a programului de azi, restul alegerilor si cbuf-urilor identice; {}",
        lines.join("; ")
    );
}

/// T310 S3 (L4, L5), SIMULARE: refusals before anything is sent, and `gelu_fwd = None` = `attn11` byte for byte.
#[test]
fn t310_s3_refusals() {
    let (_, la) = list_513(2, false);
    let (_, lb) = list_513(2, true);
    // None on a list without fc+gelu = attn11
    assert_eq!(
        pack12(&la, Some(P8), V8R, None).unwrap(),
        pack11(&la, Some(P8), V8R).unwrap()
    );
    // a fused list: refused without the geluf programs, by attn11 and by the older packers
    let e = pack12(&lb, Some(P8), V8R, None).unwrap_err();
    assert!(e.contains("foloseste pack_mixed_fast_attn12"), "{e}");
    assert!(pack11(&lb, Some(P8), V8R).is_err());
    assert!(ts::pack_mixed_gelu(&lb, prog_va, TC, TCG, ZONE).is_err());
    // the variant the waves pick not loaded
    for gff in [GeluFwdFast { v8n: 0, ..GFF }, GeluFwdFast { v8: 0, ..GFF }] {
        let lot = if gff.v8n == 0 { 6 } else { 2 };
        let (_, l) = list_513(lot, true);
        let r = pack12(&l, Some(P8), V8R, Some(gff));
        let e = r.unwrap_err();
        assert!(e.contains("varianta geluf"), "{e}");
    }
    // unaligned address of a geluf program
    assert!(pack12(&lb, Some(P8), V8R, Some(GeluFwdFast { v7: F7 + 4, ..GFF })).is_err());
    // no v8 and no v7: no geluf variant
    let e = ts::pack_mixed_fast_attn12(
        &lb,
        prog_va,
        &FAST,
        V5,
        V6,
        0,
        None,
        0,
        AFT,
        Some(GF),
        0,
        Some(GFF),
        AT,
        FV4,
        A0,
        Some(64 * KIB),
        ZONE,
    )
    .unwrap_err();
    assert!(format!("{e:?}").contains("fara varianta geluf"), "{e:?}");
    // an fc+gelu with an unaligned product (k = 30: rows of A not a multiple of 4) is refused
    let unaligned = vec![MixedLaunch::TcGeluFwd {
        args: ff::Args {
            tc: nvidia_ga104::mm_tc::Args {
                a: HEAP,
                b: HEAP + 0x10_0000,
                c: HEAP + 0x20_0000,
                bias: 0,
                m: 64,
                n: 64,
                k: 30,
                flags: nvidia_ga104::mm_tc::TB,
            },
            act: HEAP + 0x30_0000,
        },
        tag: "proba",
        layer: None,
    }];
    let e = pack12(&unaligned, Some(P8), 0, Some(GFF)).unwrap_err();
    assert!(e.contains("fara varianta geluf"), "{e}");
    // a plan whose fc is not followed by its GELU cannot be built: fuse checks u == the GELU's input (unit: cbuf)
    println!(
        "kt c310 S3 SIMULARE: gelu_fwd = None -> attn11 octet cu octet; lista cu fc+gelu refuzata fara programele geluf, de attn11 si de pack_mixed_gelu; varianta neincarcata, adresa nealiniata, fara v7 / v8, produs nealiniat refuzate"
    );
}

/// T310 S4 (L4), SIMULARE: one aligned product with bias (256 × 512 × 128, TB + BIAS) forced onto each of the three
/// `geluf` programs: C = the plain program's bits, act = `gelu_tr` mode 0 on C, bit for bit, as the two launches
/// "fc" + `gelu_tr` give; act overlapping C refused by the simulated program with nothing written.
#[test]
fn t310_s4_one_product_each_program() {
    let args = nvidia_ga104::mm_tc::Args {
        a: HEAP,
        b: HEAP + 0x10_0000,
        c: HEAP + 0x20_0000,
        bias: HEAP + 0x30_0000,
        m: 256,
        n: 512,
        k: 128,
        flags: nvidia_ga104::mm_tc::TB | nvidia_ga104::mm_tc::BIAS,
    };
    let act = HEAP + 0x40_0000;
    let mn = 256 * 512;
    let fill = |cp: &mut CommandProcessor| {
        let mut r = Rng(0x0c16_8f5c_2026_1128);
        let mut f = |n: usize, s: f64| -> Vec<u8> {
            (0..n)
                .flat_map(|_| (((r.unit() * 2.0 - 1.0) * s) as f32).to_le_bytes())
                .collect()
        };
        cp.mem.write(args.a, &f(256 * 128, 2.0));
        cp.mem.write(args.b, &f(512 * 128, 1.0));
        cp.mem.write(args.bias, &f(512, 1.0));
        cp.mem.write(args.c, &vec![0xffu8; mn * 4]);
        cp.mem.write(act, &vec![0xffu8; mn * 4]);
    };
    let gelu = MixedLaunch::Std(
        ts::Launch::new(
            Prog::GeluTr,
            nvidia_ga104::gelu_tr::Args {
                u: args.c,
                g: 0,
                y: act,
                n: mn as u32,
                mode: nvidia_ga104::gelu_tr::Mode::Forward,
            }
            .cbuf()
            .as_ref()
            .map(|b| &b[..]),
            nvidia_ga104::gelu_tr::Args {
                u: args.c,
                g: 0,
                y: act,
                n: mn as u32,
                mode: nvidia_ga104::gelu_tr::Mode::Forward,
            }
            .grid(),
            "proba:gelu",
            None,
        )
        .unwrap(),
    );
    let mut lines = Vec::new();
    // (rates, v7 loaded) -> which program the waves pick for this product
    for (p8, want, plain) in [
        (P8_V8, TcPick::V8GeluF, TcPick::V8),
        (
            V8Programs {
                rates: t8::Rates {
                    v8_pct: 100,
                    v8n_pct: 400,
                },
                ..P8_V8
            },
            TcPick::V8nGeluF,
            TcPick::V8n,
        ),
    ] {
        let two = vec![
            MixedLaunch::Tc {
                args,
                tag: "proba:fc",
                layer: None,
            },
            gelu.clone(),
        ];
        let one = vec![MixedLaunch::TcGeluFwd {
            args: ff::Args { tc: args, act },
            tag: "proba:fc+gelu",
            layer: None,
        }];
        let mut outs = Vec::new();
        for (list, w) in [(&two, plain), (&one, want)] {
            let (zone, qmds, p) = pack12(list, Some(p8), 0, Some(GFF)).unwrap();
            assert_eq!(p[0], Some(w));
            let mut cp = board();
            fill(&mut cp);
            let mut first = 1u32;
            submit(&mut cp, &mut first, &zone, &qmds).unwrap();
            outs.push((
                bits(&cp.mem.read_f32s(args.c, mn).unwrap()),
                bits(&cp.mem.read_f32s(act, mn).unwrap()),
            ));
        }
        assert_eq!(outs[0], outs[1], "{want:?}");
        assert!(outs[1].1.iter().all(|x| f32::from_bits(*x).is_finite()));
        lines.push(want.name());
    }
    // v7: no v8 given -> v7_geluf
    {
        let two = vec![
            MixedLaunch::Tc {
                args,
                tag: "proba:fc",
                layer: None,
            },
            gelu.clone(),
        ];
        let one = vec![MixedLaunch::TcGeluFwd {
            args: ff::Args { tc: args, act },
            tag: "proba:fc+gelu",
            layer: None,
        }];
        let mut outs = Vec::new();
        for (list, w) in [(&two, TcPick::V7), (&one, TcPick::V7GeluF)] {
            let (zone, qmds, p) = ts::pack_mixed_fast_attn12(
                list,
                prog_va,
                &FAST,
                V5,
                V6,
                V7,
                None,
                0,
                AFT,
                Some(GF),
                0,
                Some(GFF),
                AT,
                FV4,
                A0,
                Some(64 * KIB),
                ZONE,
            )
            .unwrap();
            assert_eq!(p[0], Some(w));
            let mut cp = board();
            fill(&mut cp);
            let mut first = 1u32;
            submit(&mut cp, &mut first, &zone, &qmds).unwrap();
            outs.push((
                bits(&cp.mem.read_f32s(args.c, mn).unwrap()),
                bits(&cp.mem.read_f32s(act, mn).unwrap()),
            ));
        }
        assert_eq!(outs[0], outs[1], "v7_geluf");
        lines.push(TcPick::V7GeluF.name());
    }
    // the simulated program refuses act overlapping C (the packer does too, through Args::cbuf)
    assert_eq!(
        ff::Args {
            tc: args,
            act: args.c + 64
        }
        .cbuf(),
        None
    );
    println!(
        "kt c310 S4 SIMULARE: produsul 256x512x128 (TB + BIAS) pe {}: C si act = fc + gelu_tr, bit cu bit",
        lines.join(" / ")
    );
}

/// T310 S5 (L5), SIMULARE: through RAM — `gelu_fwd = None` = `via_ram_attn11` byte for byte; with the `geluf` programs
/// the picks and the list part of the zone are those of `pack_mixed_fast_attn12`.
#[test]
fn t310_s5_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let k = Some(64 * KIB);
    let opt = AdamW::new(&initial(), LR, 0.9, 0.999, 1e-8, 0.1);
    let list_of = |fwd: bool| {
        let plan = plan_mini(&inputs, fwd).resident().unwrap();
        let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
        let l = plan.step_mixed(Recipe::AdamW513, &opt, 1, scale()).unwrap();
        let l =
            ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()).unwrap();
        (plan, v, l)
    };
    let (pa, va, la) = list_of(false);
    let (pb, vb, lb) = list_of(true);
    assert!(pb.fused_gelu_fwd() && !pa.fused_gelu_fwd());
    assert_eq!(va.copies_in.len(), vb.copies_in.len());
    let via12 = |l: &[MixedLaunch], v: &br::ViaRam, g: Option<GeluFwdFast>| {
        br::pack_mixed_fast_via_ram_attn12(
            l,
            prog_va,
            &FAST,
            V5,
            V6,
            V7,
            Some(P8),
            t8::LONG_K_OS503,
            AFT,
            Some(GF),
            V8R,
            g,
            AT,
            FV4,
            A0,
            k,
            TC,
            ZONE,
            v,
        )
    };
    let old = br::pack_mixed_fast_via_ram_attn11(
        &la,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8),
        t8::LONG_K_OS503,
        AFT,
        Some(GF),
        V8R,
        AT,
        FV4,
        A0,
        k,
        TC,
        ZONE,
        &va,
    )
    .unwrap();
    assert_eq!(via12(&la, &va, None).unwrap(), old);
    assert!(via12(&lb, &vb, None).is_err());
    let (zr, _, pr) = via12(&lb, &vb, Some(GFF)).unwrap();
    let list_va = ZONE + ts::SLOT * vb.copies_in.len() as u64;
    let (zd, _, pd) = ts::pack_mixed_fast_attn12(
        &lb,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        Some(P8),
        t8::LONG_K_OS503,
        AFT,
        Some(GF),
        V8R,
        Some(GFF),
        AT,
        FV4,
        A0,
        k,
        list_va,
    )
    .unwrap();
    assert_eq!(pr, pd);
    assert_eq!(
        count(&pr, TcPick::V7GeluF) + count(&pr, TcPick::V8GeluF) + count(&pr, TcPick::V8nGeluF),
        MINI.layers
    );
    let off = (list_va - ZONE) as usize;
    assert_eq!(&zr[off..off + zd.len()], &zd[..]);
    println!(
        "kt c310 S5 prin RAM SIMULARE: gelu_fwd = None -> via_ram_attn11 octet cu octet; cu geluf aceleasi alegeri si aceeasi lista ca fara RAM"
    );
}
