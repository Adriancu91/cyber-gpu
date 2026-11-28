//! C-168 p.5 P5d (session 294) — T311, SIMULARE: the forward residuals fused into the store of "out" and "proj"
//! (`mm_tc_v7_res` / `mm_tc_v8_res` / `mm_tc_v8n_res` / `mm_tc_v8r_res`, `Plan::with_fused_res_fwd`,
//! `MixedLaunch::TcRes`, `pack_mixed_fast_attn13`, `pack_mixed_fast_via_ram_attn13`). The plan and the predictions M1–M7
//! are in `docs/C-168_P5_LIPIRE_PLAN.md` §12 (written before this test, session 294).
//!
//! Honesty: everything here runs on the simulated board (`gpu_sim`); none of the four `_res` programs has run on a
//! board.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, ATTN_FWD_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS,
    LN_BACK_RES_NEEDS, LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS,
    MM_TC_V5_NEEDS, MM_TC_V6_NEEDS, MM_TC_V7_GELU_NEEDS, MM_TC_V7_GELUF_NEEDS, MM_TC_V7_NEEDS,
    MM_TC_V7_RES_NEEDS, MM_TC_V8_GELU_NEEDS, MM_TC_V8_GELUF_NEEDS, MM_TC_V8_NEEDS,
    MM_TC_V8_RES_NEEDS, MM_TC_V8N_GELU_NEEDS, MM_TC_V8N_GELUF_NEEDS, MM_TC_V8N_NEEDS,
    MM_TC_V8N_RES_NEEDS, MM_TC_V8R_NEEDS, MM_TC_V8R_RES_NEEDS, MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS,
    Needs, XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu, attn_back_tile_tc16_on_cpu,
    attn_back_tile_tc16_v4_on_cpu, attn_back_tile_tc16_v4pf_on_cpu, attn_back_tile0_on_cpu,
    attn_back_tile0_tc_on_cpu, attn_fwd_on_cpu, attn_fwd_tc_on_cpu, attn_fwd_tile_on_cpu,
    attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, colsum_part_on_cpu, emb_back_on_cpu,
    emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu,
    ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu, mm_tc_v4_on_cpu,
    mm_tc_v5_on_cpu, mm_tc_v6_on_cpu, mm_tc_v7_gelu_on_cpu, mm_tc_v7_geluf_on_cpu, mm_tc_v7_on_cpu,
    mm_tc_v7_res_on_cpu, mm_tc_v8_gelu_on_cpu, mm_tc_v8_geluf_on_cpu, mm_tc_v8_on_cpu,
    mm_tc_v8_res_on_cpu, mm_tc_v8n_gelu_on_cpu, mm_tc_v8n_geluf_on_cpu, mm_tc_v8n_on_cpu,
    mm_tc_v8n_res_on_cpu, mm_tc_v8r_on_cpu, mm_tc_v8r_res_on_cpu, mm_train_gelu_on_cpu,
    mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::attn_back_tile_tc16_v4::ATTN_BACK_TILE_TC16_V4;
use nvidia_ga104::attn_back_tile_tc16_v4pf::ATTN_BACK_TILE_TC16_V4PF;
use nvidia_ga104::attn_back_tile0_tc::ATTN_BACK_TILE0_TC;
use nvidia_ga104::attn_fwd_tc::ATTN_FWD_TC;
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_fast_gelu::{MM_TC_V7_GELU, MM_TC_V8_GELU, MM_TC_V8N_GELU};
use nvidia_ga104::mm_tc_fast_gelu_fwd::{MM_TC_V7_GELUF, MM_TC_V8_GELUF, MM_TC_V8N_GELUF};
use nvidia_ga104::mm_tc_fast_res::{
    self as fr, MM_TC_V7_RES, MM_TC_V8_RES, MM_TC_V8N_RES, MM_TC_V8R_RES,
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
    MixedLaunch, PROGRAMS_ALL_FUSED2, Plan, Prog, ResFast, TcPick, TcPrograms, V8Programs,
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
const R7: u64 = PROGS + 0x1_0000 * 54;
const R8: u64 = PROGS + 0x1_0000 * 56; // two 64 KiB slots each, as the geluf programs
const R8N: u64 = PROGS + 0x1_0000 * 58;
const R8R: u64 = PROGS + 0x1_0000 * 60;
const RF: ResFast = ResFast {
    v7: R7,
    v8: R8,
    v8n: R8N,
    v8r: R8R,
};
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
const _: () = assert!(R8R + 0x2_0000 <= HEAP);

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
    cp.register_program_with_needs(MM_TC_V7_RES, mm_tc_v7_res_on_cpu, MM_TC_V7_RES_NEEDS);
    cp.mem.write(R7, MM_TC_V7_RES);
    cp.register_program_with_needs(MM_TC_V8_RES, mm_tc_v8_res_on_cpu, MM_TC_V8_RES_NEEDS);
    cp.mem.write(R8, MM_TC_V8_RES);
    cp.register_program_with_needs(MM_TC_V8N_RES, mm_tc_v8n_res_on_cpu, MM_TC_V8N_RES_NEEDS);
    cp.mem.write(R8N, MM_TC_V8N_RES);
    cp.register_program_with_needs(MM_TC_V8R_RES, mm_tc_v8r_res_on_cpu, MM_TC_V8R_RES_NEEDS);
    cp.mem.write(R8R, MM_TC_V8R_RES);
    let csp = Prog::ColsumPart.code();
    cp.register_program(csp, colsum_part_on_cpu);
    cp.mem.write(CSP, csp);
    cp
}

fn scale() -> f32 {
    ts::loss_scale_for(MINI.vocab, B * T)
}

/// The P5c plan shape on the mini model (`new_tiles3` + `with_colsum_split` + `with_fused_back` + `with_fused_gelu` +
/// `with_fused_gelu_fwd`), + `with_fused_res_fwd` when `res`.
fn plan_mini(inputs: &[u32], res: bool) -> Plan {
    let p = Plan::new_tiles3(MINI, B, T, inputs, HEAP)
        .unwrap()
        .with_colsum_split()
        .with_fused_back()
        .with_fused_gelu()
        .with_fused_gelu_fwd();
    if res { p.with_fused_res_fwd() } else { p }
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

/// The #513 list of step 1 as the OS would send it after P5c (`new_tiles3` + `with_mm_tile` + `with_colsum_split` +
/// `with_fused_back` + `with_fused_gelu` + `with_fused_gelu_fwd`, the attention on the tensor cores), +
/// `with_fused_res_fwd` when `res`.
fn list_513(bsz: usize, res: bool) -> (Plan, Vec<MixedLaunch>) {
    let m = train::M513;
    let inputs: Vec<u32> = (0..bsz as u32 * 512).map(|i| (i * 7919) % 16_384).collect();
    let base = VA + (1 << 30);
    let plan = Plan::new_tiles3(m, bsz, 512, &inputs, base)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split()
        .with_fused_back()
        .with_fused_gelu()
        .with_fused_gelu_fwd();
    let plan = if res { plan.with_fused_res_fwd() } else { plan };
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    let l = plan
        .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap();
    let l = ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()).unwrap();
    (plan, l)
}

type Packed = (Vec<u8>, Vec<u64>, Vec<Option<TcPick>>);

/// `pack_mixed_fast_attn13` with the P5b GELU' variants, `v8r_va`, the P5c `geluf` programs, `res_fwd`, the os-503
/// guard, the attention on the tensor cores.
fn pack13(
    list: &[MixedLaunch],
    p8: Option<V8Programs>,
    v8r_va: u64,
    res_fwd: Option<ResFast>,
) -> Result<Packed, String> {
    ts::pack_mixed_fast_attn13(
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
        Some(GFF),
        res_fwd,
        AT,
        FV4,
        A0,
        Some(64 * KIB),
        ZONE,
    )
    .map_err(|e| format!("{e:?}"))
}

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

/// The `_res` variant of a plain pick (what P5d must give the fused "out" / "proj").
fn res_of(p: Option<TcPick>) -> Option<TcPick> {
    match p {
        Some(TcPick::V7) => Some(TcPick::V7Res),
        Some(TcPick::V8) => Some(TcPick::V8Res),
        Some(TcPick::V8r) => Some(TcPick::V8rRes),
        Some(TcPick::V8n) => Some(TcPick::V8nRes),
        other => panic!("out / proj pe {other:?}: nu are varianta _res"),
    }
}

fn nres(p: &[Option<TcPick>]) -> usize {
    count(p, TcPick::V7Res)
        + count(p, TcPick::V8Res)
        + count(p, TcPick::V8nRes)
        + count(p, TcPick::V8rRes)
}

struct Out {
    losses: Vec<Vec<f32>>,
    grad1: Vec<f32>,
    weights: Vec<f32>,
    picks: Vec<Option<TcPick>>,
    launches: usize,
}

/// `STEPS` fast mixed steps of the aligned mini model (P5c plan, + `with_fused_res_fwd` when `res`) on the simulated
/// board, packed by `attn13` with `p8`, `v8r_va`, the `geluf` and the `_res` programs.
fn run(p8: V8Programs, v8r_va: u64, res: bool) -> Out {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = plan_mini(&inputs, res);
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
        let (zone, qmds, picks) = pack13(&list, Some(p8), v8r_va, Some(RF)).unwrap();
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

/// Rates that make the waves give v8n every aligned product.
const P8_V8N: V8Programs = V8Programs {
    rates: t8::Rates {
        v8_pct: 100,
        v8n_pct: 400,
    },
    ..P8_V8
};

/// T311 S1 (M4), SIMULARE: the mini model, 5 fast mixed steps (P5a + P5b + P5c): with `with_fused_res_fwd` the same
/// bits as without (losses, the gradient of step 1, the weights), two launches fewer per layer, "out" and "proj" on the
/// `_res` variant — with the os-500 rates, with the rates steered to v8 (with and without v8r loaded) and to v8n.
#[test]
fn t311_s1_mini_same_bits() {
    let mut lines = Vec::new();
    for (p8, v8r, name) in [
        (P8, 0, "os-500"),
        (P8_V8, 0, "v8"),
        (P8_V8, V8R, "v8 + v8r"),
        (P8_V8N, 0, "v8n"),
    ] {
        let a = run(p8, v8r, false);
        let b = run(p8, v8r, true);
        same(&a, &b, name);
        assert_eq!(
            b.launches + 2 * MINI.layers,
            a.launches,
            "{name}: -2L lansari"
        );
        assert_eq!(
            nres(&b.picks),
            2 * MINI.layers,
            "{name}: out / proj pe _res"
        );
        assert_eq!(nres(&a.picks), 0);
        assert!(b.weights.iter().all(|x| x.is_finite()));
        lines.push(format!(
            "{name}: {} -> {} lansari, v7_res {} / v8_res {} / v8n_res {} / v8r_res {}",
            a.launches,
            b.launches,
            count(&b.picks, TcPick::V7Res),
            count(&b.picks, TcPick::V8Res),
            count(&b.picks, TcPick::V8nRes),
            count(&b.picks, TcPick::V8rRes)
        ));
    }
    println!(
        "kt c311 S1 SIMULARE: modelul mic, 5 pasi rapizi P5a+P5b+P5c, cu out / proj + rezidual = aceiasi biti ca fara (pierderile, gradientul pasului 1, greutatile); {}",
        lines.join("; ")
    );
}

/// T311 S2 (M5), SIMULARE: #513 lot 2 and lot 6 (the P5c list, v8r loaded): exactly −24 launches (the 24 forward
/// `vadd`), each "out" / "proj" on the `_res` variant of the program it had, every other pick identical and in order,
/// every other launch's cbuf byte for byte; each fused launch: C = the old `vadd`'s output, `res` = its first input,
/// the variant's address, registers, block, grid in the QMD.
#[test]
fn t311_s2_513_lists() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let rd = |cb: &[u8; 0x100], at: usize| u64::from_le_bytes(cb[at..at + 8].try_into().unwrap());
    let mut lines = Vec::new();
    for bsz in [2usize, 6] {
        let (pa, la) = list_513(bsz, false);
        let (pb, lb) = list_513(bsz, true);
        assert_eq!(la.len(), lb.len() + 24, "M5: lot {bsz} exact -24 lansari");
        assert_eq!(la.len(), pa.launches_per_step_mixed(Recipe::AdamW513));
        assert_eq!(lb.len(), pb.launches_per_step_mixed(Recipe::AdamW513));
        let (za, qa, ka) = pack13(&la, Some(P8), V8R, Some(RF)).unwrap();
        let (zb, qb, kb) = pack13(&lb, Some(P8), V8R, Some(RF)).unwrap();
        assert_eq!(
            pack12(&la, Some(P8), V8R, Some(GFF)).unwrap(),
            (za.clone(), qa.clone(), ka.clone())
        );
        let (mut i, mut j) = (0usize, 0usize);
        let (mut outs, mut projs) = (Vec::new(), Vec::new());
        let vadd_fwd = la
            .iter()
            .filter(|l| matches!(l, MixedLaunch::Std(v) if v.prog == Prog::Vadd && v.tag.starts_with("inainte:")))
            .count();
        assert_eq!(vadd_fwd, 24);
        while i < la.len() {
            let (a, b) = (&la[i], &lb[j]);
            let ca = (qa[i] - ZONE) as usize;
            let cb_at = (qb[j] - ZONE) as usize;
            if let MixedLaunch::TcRes { args, tag, layer } = b {
                let MixedLaunch::Tc {
                    args: pa_,
                    tag: ta,
                    layer: lya,
                } = a
                else {
                    panic!("lansarea {i}: out / proj + rezidual fara produs")
                };
                let want = match *ta {
                    "inainte:out" => ("inainte:out+rez", "inainte:rezidual1"),
                    "inainte:proj" => ("inainte:proj+rez", "inainte:rezidual2"),
                    t => panic!("lansarea {i}: {t}"),
                };
                assert_eq!(*tag, want.0);
                assert_eq!(layer, lya);
                let MixedLaunch::Std(v) = &la[i + 1] else {
                    panic!("dupa {ta} nu e vadd")
                };
                assert_eq!((v.prog, v.tag), (Prog::Vadd, want.1));
                assert_eq!(rd(&v.cbuf, 0x30), pa_.c, "vadd citeste produsul");
                assert_eq!(
                    args.tc,
                    nvidia_ga104::mm_tc::Args {
                        c: rd(&v.cbuf, 0x38),
                        ..*pa_
                    }
                );
                assert_eq!(args.res, rd(&v.cbuf, 0x28));
                assert_eq!(kb[j], res_of(ka[i]), "lansarea {i}");
                if want.0 == "inainte:out+rez" {
                    outs.push(kb[j].unwrap());
                } else {
                    projs.push(kb[j].unwrap());
                }
                // the cbuf: mm_tc's of the product writing x' with res at 0x730
                let c = nvidia_ga104::mm_tc::CBUF_BYTES;
                let cbb = &zb[cb_at - c..cb_at];
                assert_eq!(cbb, &args.cbuf().unwrap()[..]);
                assert_eq!(cbb[fr::RES_AT..fr::RES_AT + 8], args.res.to_le_bytes());
                let cba = &za[ca - c..ca];
                assert_eq!(cba[..0x710], cbb[..0x710]);
                assert_eq!(cba[0x718..fr::RES_AT], cbb[0x718..fr::RES_AT]);
                // the QMD
                let qmd = &zb[cb_at..cb_at + 0x100];
                let va = qmd_field(qmd, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER) << 32
                    | qmd_field(qmd, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER);
                let (pva, regs, block, grid) = match kb[j] {
                    Some(TcPick::V8nRes) => (R8N, fr::REGISTERS_V8N, 128, t8::grid_n(pa_)),
                    Some(TcPick::V8Res) => (R8, fr::REGISTERS_V8, 256, t8::grid(pa_)),
                    Some(TcPick::V8rRes) => (R8R, fr::REGISTERS_V8R, 256, t8::grid(pa_)),
                    _ => (R7, fr::REGISTERS_V7, 128, pa_.grid()),
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
            let c = (a.slot() - 0x100) as usize;
            assert_eq!(&za[ca - c..ca], &zb[cb_at - c..cb_at], "lansarea {i}: cbuf");
            i += 1;
            j += 1;
        }
        assert_eq!(j, lb.len());
        assert_eq!((outs.len(), projs.len()), (12, 12));
        assert!(outs.iter().all(|p| *p == outs[0]));
        assert!(projs.iter().all(|p| *p == projs[0]));
        assert!(
            !lb.iter()
                .any(|l| matches!(l, MixedLaunch::Std(v) if v.prog == Prog::Vadd && v.tag.starts_with("inainte:")))
        );
        lines.push(format!(
            "lot {bsz}: {} -> {} lansari, out pe {}, proj pe {}",
            la.len(),
            lb.len(),
            outs[0].name(),
            projs[0].name()
        ));
    }
    println!(
        "kt c311 S2 SIMULARE: #513 (lista P5c + v8r): exact -24 lansari (24 vadd inainte), out / proj pe varianta _res a programului de azi, restul alegerilor si cbuf-urilor identice; {}",
        lines.join("; ")
    );
}

/// T311 S3 (M4, M5), SIMULARE: refusals before anything is sent, and `res_fwd = None` = `attn12` byte for byte.
#[test]
fn t311_s3_refusals() {
    let (_, la) = list_513(2, false);
    let (_, lb) = list_513(2, true);
    assert_eq!(
        pack13(&la, Some(P8), V8R, None).unwrap(),
        pack12(&la, Some(P8), V8R, Some(GFF)).unwrap()
    );
    let e = pack13(&lb, Some(P8), V8R, None).unwrap_err();
    assert!(e.contains("foloseste pack_mixed_fast_attn13"), "{e}");
    assert!(pack12(&lb, Some(P8), V8R, Some(GFF)).is_err());
    assert!(ts::pack_mixed_gelu(&lb, prog_va, TC, TCG, ZONE).is_err());
    // the variant the waves pick not loaded (lot 2 / lot 6 picks are printed by S2; every variant is tried)
    for (lot, rf) in [
        (2usize, ResFast { v8r: 0, ..RF }),
        (2, ResFast { v7: 0, ..RF }),
        (6, ResFast { v8n: 0, ..RF }),
        (6, ResFast { v7: 0, ..RF }),
    ] {
        let (_, l) = list_513(lot, true);
        let (_, _, k) = pack13(&l, Some(P8), V8R, Some(RF)).unwrap();
        let used = |p: TcPick| count(&k, p) > 0;
        let missing = (rf.v7 == 0 && used(TcPick::V7Res))
            || (rf.v8 == 0 && used(TcPick::V8Res))
            || (rf.v8n == 0 && used(TcPick::V8nRes))
            || (rf.v8r == 0 && used(TcPick::V8rRes));
        let r = pack13(&l, Some(P8), V8R, Some(rf));
        if missing {
            let e = r.unwrap_err();
            assert!(e.contains("varianta _res"), "{e}");
        } else {
            assert!(r.is_ok());
        }
    }
    // v8r not loaded: v8 picks go on v8_res
    let (_, l2) = list_513(2, true);
    let (_, _, k) = pack13(&l2, Some(P8), 0, Some(RF)).unwrap();
    assert_eq!(count(&k, TcPick::V8rRes), 0);
    // unaligned address of a _res program
    assert!(pack13(&lb, Some(P8), V8R, Some(ResFast { v7: R7 + 4, ..RF })).is_err());
    // no v8 and no v7: no _res variant
    let e = ts::pack_mixed_fast_attn13(
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
        Some(RF),
        AT,
        FV4,
        A0,
        Some(64 * KIB),
        ZONE,
    )
    .unwrap_err();
    assert!(format!("{e:?}").contains("fara varianta _res"), "{e:?}");
    // an unaligned product (k = 30)
    let unaligned = vec![MixedLaunch::TcRes {
        args: fr::Args {
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
            res: HEAP + 0x30_0000,
        },
        tag: "proba",
        layer: None,
    }];
    let e = pack13(&unaligned, Some(P8), 0, Some(RF)).unwrap_err();
    assert!(e.contains("fara varianta _res"), "{e}");
    println!(
        "kt c311 S3 SIMULARE: res_fwd = None -> attn12 octet cu octet; lista cu out / proj + rezidual refuzata fara programele _res, de attn12 si de pack_mixed_gelu; varianta neincarcata, adresa nealiniata, fara v7 / v8, produs nealiniat refuzate"
    );
}

/// T311 S4 (M4), SIMULARE: one aligned product with bias (256 × 512 × 128, TB + BIAS) forced onto each of the four
/// `_res` programs: C = the product + `vadd`'s output, bit for bit, as the two launches give; `res` overlapping C
/// refused.
#[test]
fn t311_s4_one_product_each_program() {
    let mn = 256 * 512;
    let tmp = HEAP + 0x20_0000;
    let args = nvidia_ga104::mm_tc::Args {
        a: HEAP,
        b: HEAP + 0x10_0000,
        c: tmp,
        bias: HEAP + 0x30_0000,
        m: 256,
        n: 512,
        k: 128,
        flags: nvidia_ga104::mm_tc::TB | nvidia_ga104::mm_tc::BIAS,
    };
    let x = HEAP + 0x40_0000;
    let xo = HEAP + 0x60_0000;
    let fill = |cp: &mut CommandProcessor| {
        let mut r = Rng(0x0c16_8f5d_2026_1128);
        let mut f = |n: usize, s: f64| -> Vec<u8> {
            (0..n)
                .flat_map(|_| (((r.unit() * 2.0 - 1.0) * s) as f32).to_le_bytes())
                .collect()
        };
        cp.mem.write(args.a, &f(256 * 128, 2.0));
        cp.mem.write(args.b, &f(512 * 128, 1.0));
        cp.mem.write(args.bias, &f(512, 1.0));
        cp.mem.write(x, &f(mn, 3.0));
        cp.mem.write(tmp, &vec![0xffu8; mn * 4]);
        cp.mem.write(xo, &vec![0xffu8; mn * 4]);
    };
    let mut vcb = [0u8; 0x44];
    vcb[0x28..0x30].copy_from_slice(&x.to_le_bytes());
    vcb[0x30..0x38].copy_from_slice(&tmp.to_le_bytes());
    vcb[0x38..0x40].copy_from_slice(&xo.to_le_bytes());
    vcb[0x40..0x44].copy_from_slice(&(mn as u32).to_le_bytes());
    let vadd = MixedLaunch::Std(
        ts::Launch::new(
            Prog::Vadd,
            Some(&vcb[..]),
            [(mn as u32).div_ceil(256), 1, 1],
            "proba:rezidual",
            None,
        )
        .unwrap(),
    );
    let two = vec![
        MixedLaunch::Tc {
            args,
            tag: "proba:out",
            layer: None,
        },
        vadd,
    ];
    let one = vec![MixedLaunch::TcRes {
        args: fr::Args {
            tc: nvidia_ga104::mm_tc::Args { c: xo, ..args },
            res: x,
        },
        tag: "proba:out+rez",
        layer: None,
    }];
    let mut lines = Vec::new();
    // (v8 programs, v7 loaded, v8r loaded) -> which program the waves pick for this product
    for (p8, v7, v8r, plain, want) in [
        (Some(P8_V8), V7, 0, TcPick::V8, TcPick::V8Res),
        (Some(P8_V8), V7, V8R, TcPick::V8r, TcPick::V8rRes),
        (Some(P8_V8N), V7, 0, TcPick::V8n, TcPick::V8nRes),
        (None, V7, 0, TcPick::V7, TcPick::V7Res),
    ] {
        let mut outs = Vec::new();
        for (list, w) in [(&two, plain), (&one, want)] {
            let (zone, qmds, p) = ts::pack_mixed_fast_attn13(
                list,
                prog_va,
                &FAST,
                V5,
                V6,
                v7,
                p8,
                if p8.is_some() { t8::LONG_K_OS503 } else { 0 },
                AFT,
                Some(GF),
                v8r,
                Some(GFF),
                Some(RF),
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
            outs.push(bits(&cp.mem.read_f32s(xo, mn).unwrap()));
        }
        assert_eq!(outs[0], outs[1], "{want:?}");
        assert!(outs[1].iter().all(|v| f32::from_bits(*v).is_finite()));
        lines.push(want.name());
    }
    assert_eq!(
        fr::Args {
            tc: nvidia_ga104::mm_tc::Args { c: xo, ..args },
            res: xo + 64
        }
        .cbuf(),
        None
    );
    println!(
        "kt c311 S4 SIMULARE: produsul 256x512x128 (TB + BIAS) pe {}: C = produs + vadd, bit cu bit",
        lines.join(" / ")
    );
}

/// T311 S5 (M5), SIMULARE: through RAM — `res_fwd = None` = `via_ram_attn12` byte for byte; with the `_res` programs
/// the picks and the list part of the zone are those of `pack_mixed_fast_attn13`.
#[test]
fn t311_s5_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let k = Some(64 * KIB);
    let opt = AdamW::new(&initial(), LR, 0.9, 0.999, 1e-8, 0.1);
    let list_of = |res: bool| {
        let plan = plan_mini(&inputs, res).resident().unwrap();
        let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
        let l = plan.step_mixed(Recipe::AdamW513, &opt, 1, scale()).unwrap();
        let l =
            ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()).unwrap();
        (plan, v, l)
    };
    let (pa, va, la) = list_of(false);
    let (pb, vb, lb) = list_of(true);
    assert!(pb.fused_res_fwd() && !pa.fused_res_fwd());
    assert_eq!(va.copies_in.len(), vb.copies_in.len());
    let via13 = |l: &[MixedLaunch], v: &br::ViaRam, r: Option<ResFast>| {
        br::pack_mixed_fast_via_ram_attn13(
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
            Some(GFF),
            r,
            AT,
            FV4,
            A0,
            k,
            TC,
            ZONE,
            v,
        )
    };
    let old = br::pack_mixed_fast_via_ram_attn12(
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
        Some(GFF),
        AT,
        FV4,
        A0,
        k,
        TC,
        ZONE,
        &va,
    )
    .unwrap();
    assert_eq!(via13(&la, &va, None).unwrap(), old);
    assert!(via13(&lb, &vb, None).is_err());
    let (zr, _, pr) = via13(&lb, &vb, Some(RF)).unwrap();
    let list_va = ZONE + ts::SLOT * vb.copies_in.len() as u64;
    let (zd, _, pd) = ts::pack_mixed_fast_attn13(
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
        Some(RF),
        AT,
        FV4,
        A0,
        k,
        list_va,
    )
    .unwrap();
    assert_eq!(pr, pd);
    assert_eq!(nres(&pr), 2 * MINI.layers);
    let off = (list_va - ZONE) as usize;
    assert_eq!(&zr[off..off + zd.len()], &zd[..]);
    println!(
        "kt c311 S5 prin RAM SIMULARE: res_fwd = None -> via_ram_attn12 octet cu octet; cu _res aceleasi alegeri si aceeasi lista ca fara RAM"
    );
}
