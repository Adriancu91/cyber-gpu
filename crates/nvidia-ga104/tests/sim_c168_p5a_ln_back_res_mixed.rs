//! C-168 p.5 step P5a (session 290) — T306, SIMULARE: the backward residual additions fused into the backward LayerNorm
//! (`ln_back_res`, `Plan::with_fused_back`) in the **fast mixed step** the OS runs today (`new_tiles3` +
//! `with_colsum_split`, v7 / v8 / v8n by shape, attention forward and backward on the tensor cores, `pack_mixed_fast_attn9`).
//! No new program and no new library code: `ln_back_res` ran on the RTX 3070 Ti in os-415 (the oracle passed), and the
//! fast packers take every `Std` launch by `prog_va`. The plan and the predictions I1–I4 are in
//! `docs/C-168_P5_LIPIRE_PLAN.md` §4 (written before this test).
//!
//! Honesty: everything here runs on the simulated board (`gpu_sim`); the step with `ln_back_res` on the fast path has NOT
//! run on a board.

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
    colsum_on_cpu, colsum_part_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu,
    gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu, ln_fwd_on_cpu, mm_tc_gelu_on_cpu,
    mm_tc_on_cpu, mm_tc_pf_on_cpu, mm_tc_v4_on_cpu, mm_tc_v5_on_cpu, mm_tc_v6_on_cpu,
    mm_tc_v7_on_cpu, mm_tc_v8_on_cpu, mm_tc_v8n_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu,
    opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::attn_back_tile_tc16_v4::ATTN_BACK_TILE_TC16_V4;
use nvidia_ga104::attn_back_tile_tc16_v4pf::ATTN_BACK_TILE_TC16_V4PF;
use nvidia_ga104::attn_back_tile0_tc::ATTN_BACK_TILE0_TC;
use nvidia_ga104::attn_fwd_tc::ATTN_FWD_TC;
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
    self as ts, Attn0Programs, AttnTcFast, AttnTcProgram, LOSS_SCALE_513, MixedLaunch,
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
    cp.mem.write(AFT, ATTN_FWD_TC);
    let csp = Prog::ColsumPart.code();
    cp.register_program(csp, colsum_part_on_cpu);
    cp.mem.write(CSP, csp);
    cp
}

fn scale() -> f32 {
    ts::loss_scale_for(MINI.vocab, B * T)
}

/// The OS's plan shape on the mini model: `new_tiles3` (+ `with_colsum_split` when `split`), + `with_fused_back` when
/// `fused`.
fn plan_mini(inputs: &[u32], fused: bool, split: bool) -> Plan {
    let p = Plan::new_tiles3(MINI, B, T, inputs, HEAP).unwrap();
    let p = if split { p.with_colsum_split() } else { p };
    if fused { p.with_fused_back() } else { p }
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
/// The #513 list of step 1 as the OS sends it (`new_tiles3` + `with_mm_tile` + `with_colsum_split`, both attention
/// backward programs and the forward on the tensor cores), with `with_fused_back` when `fused`.
fn list_513(bsz: usize, fused: bool) -> Vec<MixedLaunch> {
    let m = train::M513;
    let inputs: Vec<u32> = (0..bsz as u32 * 512).map(|i| (i * 7919) % 16_384).collect();
    let base = VA + (1 << 30);
    let plan = Plan::new_tiles3(m, bsz, 512, &inputs, base)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split();
    let plan = if fused { plan.with_fused_back() } else { plan };
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    let l = plan
        .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap();
    ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()).unwrap()
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

/// True for the launches `with_fused_back` changes: the per-layer backward `ln_back` and its residual `vadd` (old list),
/// `ln_back_res` (new list). The final LayerNorm's `ln_back` stays in both.
fn touched(l: &MixedLaunch) -> bool {
    match l {
        MixedLaunch::Std(x) => match x.prog {
            Prog::LnBackRes => true,
            Prog::Vadd => x.tag.starts_with("inapoi:rezidual-ln"),
            Prog::LnBack => x.tag != "inapoi:ln-final-dx",
            _ => false,
        },
        _ => false,
    }
}

struct Out {
    losses: Vec<Vec<f32>>,
    grad1: Vec<f32>,
    weights: Vec<f32>,
    launches: usize,
}

/// `STEPS` fast mixed steps of the aligned mini model on the simulated board, as the OS packs them (`attn9`, 64 KiB step,
/// the os-503 guard), with the plan of [`plan_mini`].
fn run(fused: bool, split: bool) -> Result<Out, String> {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = plan_mini(&inputs, fused, split);
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
    };
    for step in 1..=STEPS {
        let list = plan
            .step_mixed(Recipe::AdamW513, &opt, step, scale())
            .unwrap();
        let list =
            ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(list).unwrap()).unwrap())
                .unwrap();
        out.launches = list.len();
        let (zone, qmds, _) = pack9(&list);
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

/// T306 S1 (I1), SIMULARE: the aligned mini model, 5 fast mixed steps, with and without `with_colsum_split`: with
/// `with_fused_back` the losses of every step, the gradient of step 1 and the final weights are the same bits; the list
/// is 2 launches shorter per layer.
#[test]
fn t306_s1_mini_same_bits() {
    let mut lines = Vec::new();
    for split in [false, true] {
        let a = run(false, split).unwrap();
        let b = run(true, split).unwrap();
        assert_eq!(a.launches - b.launches, 2 * MINI.layers, "split {split}");
        for (s, (x, y)) in a.losses.iter().zip(&b.losses).enumerate() {
            assert_eq!(
                bits(x),
                bits(y),
                "pierderile pasului {} (split {split})",
                s + 1
            );
        }
        assert_eq!(
            bits(&a.grad1),
            bits(&b.grad1),
            "gradientul pasului 1 (split {split})"
        );
        assert_eq!(
            bits(&a.weights),
            bits(&b.weights),
            "greutatile (split {split})"
        );
        assert!(a.weights.iter().all(|x| x.is_finite()));
        lines.push(format!(
            "colsum impartit {}: {} -> {} lansari pe pas",
            if split { "da" } else { "nu" },
            a.launches,
            b.launches
        ));
    }
    println!(
        "kt c306 S1 SIMULARE: modelul mic aliniat, 5 pasi rapizi (attn9): cu ln_back_res aceleasi biti (pierderile, gradientul pasului 1, greutatile); {}",
        lines.join("; ")
    );
}

/// T306 S2 (I2), SIMULARE: #513 lot 2 and lot 6 as the OS sends it: exactly 24 launches fewer (24 `vadd` gone, 24
/// `ln_back` → `ln_back_res`), every other launch the same and in the same order; `attn9` keeps every pick (v7 / v8 /
/// v8n with the guard, `attn_fwd_tc`) and points each `ln_back_res` QMD at its program.
#[test]
fn t306_s2_513_lists() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let mut lines = Vec::new();
    for bsz in [2usize, 6] {
        let old = list_513(bsz, false);
        let new = list_513(bsz, true);
        assert_eq!(old.len() - new.len(), 24, "lot {bsz}");
        assert_eq!(
            count_prog(&old, Prog::Vadd) - count_prog(&new, Prog::Vadd),
            24
        );
        assert_eq!(count_prog(&old, Prog::LnBack), 25);
        assert_eq!(count_prog(&new, Prog::LnBack), 1);
        assert_eq!(count_prog(&new, Prog::LnBackRes), 24);
        assert_eq!(count_prog(&old, Prog::LnBackRes), 0);
        let rest_old: Vec<&MixedLaunch> = old.iter().filter(|l| !touched(l)).collect();
        let rest_new: Vec<&MixedLaunch> = new.iter().filter(|l| !touched(l)).collect();
        assert_eq!(rest_old, rest_new, "lot {bsz}: restul listei");
        let (_, _, pa) = pack9(&old);
        let (zb, qb, pb) = pack9(&new);
        for p in [TcPick::V7, TcPick::V8, TcPick::V8n, TcPick::AttnFwdTc] {
            assert_eq!(count(&pa, p), count(&pb, p), "lot {bsz} {p:?}");
        }
        let tc_old: Vec<_> = pa.iter().flatten().collect();
        let tc_new: Vec<_> = pb.iter().flatten().collect();
        assert_eq!(tc_old, tc_new, "lot {bsz}: alegerile in ordine");
        let mut lbr = 0;
        for (l, qv) in new.iter().zip(&qb) {
            if std_prog(l) == Some(Prog::LnBackRes) {
                lbr += 1;
                let at = (qv - ZONE) as usize;
                let qmd = &zb[at..at + 0x100];
                let f = |fl: nvidia_rm570::QmdField| {
                    let mut v = 0u64;
                    for i in fl.lo..=fl.hi {
                        if qmd[(i / 8) as usize] >> (i % 8) & 1 == 1 {
                            v |= 1 << (i - fl.lo);
                        }
                    }
                    v
                };
                let va = f(q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER) << 32
                    | f(q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER);
                assert_eq!(va, prog_va(Prog::LnBackRes));
                assert_eq!(
                    f(q::NVC6C0_QMDV03_00_REGISTER_COUNT_V),
                    u64::from(Prog::LnBackRes.registers())
                );
            }
        }
        assert_eq!(lbr, 24);
        let c3 = [
            count(&pb, TcPick::V7),
            count(&pb, TcPick::V8),
            count(&pb, TcPick::V8n),
        ];
        if bsz == 6 {
            assert_eq!(c3, [13, 0, 134]);
        }
        lines.push(format!(
            "lot {bsz}: {} -> {} lansari, v7/v8/v8n {}/{}/{}, attn_fwd_tc {}",
            old.len(),
            new.len(),
            c3[0],
            c3[1],
            c3[2],
            count(&pb, TcPick::AttnFwdTc)
        ));
    }
    println!(
        "kt c306 S2 SIMULARE: #513 cu ln_back_res: exact 24 lansari mai putin (24 vadd, 24 ln_back -> ln_back_res), restul neschimbat, alegerile la fel; {}",
        lines.join("; ")
    );
}

/// T306 S3, SIMULARE: what the OS loads in addition to today's programs is exactly `ln_back_res`
/// (`Plan::fused_programs`): every `Std` launch of the fused #513 list is a program of `programs()` +
/// `colsum_split_programs()` + `fused_programs()`. (The packer does not check that a program is loaded: the OS checks
/// its loaded set against the plan, as it did for os-415.)
#[test]
fn t306_s3_programs() {
    let inputs: Vec<u32> = (0..2u32 * 512).map(|i| (i * 7919) % 16_384).collect();
    let plan = Plan::new_tiles3(train::M513, 2, 512, &inputs, VA + (1 << 30))
        .unwrap()
        .with_mm_tile()
        .with_colsum_split()
        .with_fused_back();
    assert_eq!(plan.fused_programs(), &[Prog::LnBackRes][..]);
    let loaded: Vec<Prog> = plan
        .programs()
        .iter()
        .chain(plan.colsum_split_programs())
        .chain(plan.fused_programs())
        .copied()
        .collect();
    for l in list_513(2, true) {
        if let Some(p) = std_prog(&l) {
            assert!(loaded.contains(&p), "{p:?} nu e in programele planului");
        }
    }
    println!(
        "kt c306 S3 SIMULARE: programe in plus fata de azi = [ln_back_res]; toate lansarile Std ale listei lipite sunt in programele planului"
    );
}
