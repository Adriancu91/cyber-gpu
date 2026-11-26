//! C-157 — T263, SIMULARE: the fast variants of the second tensor-core attention backward program (dQ, dK, dV):
//! `attn_back_tile_tc16_v4` (128-bit loads) and `attn_back_tile_tc16_v4pf` (plus the next tile read ahead), linked in
//! the **fast mixed step**: `train_step::pack_mixed_fast_attn4` picks the loaded variant for every aligned `AttnTc`
//! launch, else `attn_back_tile_tc16`. Plan and predictions W1–W7 written before these tests
//! (`docs/C-157_ATTN_BACK_TC16_V4_LEGAT.md`, commit `a922bab`).
//!
//! Honesty: every stand-in is `gpu_compute::attn_tc::model` / `back0_model`, a HYPOTHESIS of what the programs compute
//! (the tensor-core adder fitted in os-335). Nothing here runs on the RTX 3070 Ti.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS,
    LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS,
    MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, Needs, XENT_NEEDS, attn_back_on_cpu,
    attn_back_tile_on_cpu, attn_back_tile_tc16_on_cpu, attn_back_tile_tc16_v4_on_cpu,
    attn_back_tile_tc16_v4pf_on_cpu, attn_back_tile0_on_cpu, attn_back_tile0_tc_on_cpu,
    attn_fwd_on_cpu, attn_fwd_tile_on_cpu, attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu,
    emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu,
    ln_back_res_on_cpu, ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu,
    mm_tc_v4_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu,
    xent_on_cpu,
};
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::attn_back_tile_tc16_v4::ATTN_BACK_TILE_TC16_V4;
use nvidia_ga104::attn_back_tile_tc16_v4pf::ATTN_BACK_TILE_TC16_V4PF;
use nvidia_ga104::attn_back_tile0_tc::ATTN_BACK_TILE0_TC;
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::mm_tc_pf::MM_TC_PF;
use nvidia_ga104::mm_tc_v4::MM_TC_V4;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, Attn0Programs, AttnTcFast, AttnTcProgram, LOSS_SCALE, LOSS_SCALE_513, MixedLaunch,
    PROGRAMS_ALL_FUSED2, Plan, Prog, TcPick, TcPrograms,
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
const FPF: Option<AttnTcFast> = Some(AttnTcFast::V4pf(ATC16V4PF));

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

/// Offset of the dS argument in the F32 cbuf of `attn_back_tile_3` (args at 0x28: qkv, p, dout, ds).
const DSOFF: usize = 0x40;

type ZoneEdit = dyn Fn(&mut Vec<u8>, &[u64], &[Option<TcPick>]);

const NONE: &ZoneEdit = &|_, _, _| {};

/// `STEPS` mixed steps of the mini model on the simulated board, both attention-backward programs on the tensor
/// cores, packed by `pack_mixed_fast_attn4` at the 64 KiB step.
fn run(
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
        let (mut zone, qmds, p) =
            ts::pack_mixed_fast_attn4(&list, prog_va, &FAST, attn, fast, A0, Some(64 * KIB), ZONE)
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

/// The dS argument of every `AttnTc` moved by 4 bytes: the shape stops being aligned.
fn misalign(list: &mut [MixedLaunch]) {
    for l in list.iter_mut() {
        if let MixedLaunch::AttnTc(x) = l {
            let v = u64::from_le_bytes(x.cbuf[DSOFF..DSOFF + 8].try_into().unwrap()) + 4;
            x.cbuf[DSOFF..DSOFF + 8].copy_from_slice(&v.to_le_bytes());
        }
    }
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

/// W1: without a fast variant nothing changes: attn4 with `None` = attn3 byte for byte.
#[test]
fn t263_w1_without_fast_same_bytes() {
    let k = Some(64 * KIB);
    for l in [mini_list(), list_513(2)] {
        let a = ts::pack_mixed_fast_attn3(&l, prog_va, &FAST, AT, A0, k, ZONE).unwrap();
        let b = ts::pack_mixed_fast_attn4(&l, prog_va, &FAST, AT, None, A0, k, ZONE).unwrap();
        assert_eq!(a, b);
        assert_eq!(count(&b.2, TcPick::AttnTc16V4), 0);
        assert_eq!(count(&b.2, TcPick::AttnTc16V4pf), 0);
    }
    println!(
        "kt c263 W1 SIMULARE: fara varianta rapida, attn4 = attn3 octet cu octet (mini si #513)"
    );
}

/// W2: the mini list with tc16 + v4 (and + v4pf): the `AttnTc` launches pick the fast variant; same zone length;
/// only the program address and the register count of those QMDs differ.
#[test]
fn t263_w2_mini_picks_fast() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let k = Some(64 * KIB);
    let l = mini_list();
    let (za, qa, pa) = ts::pack_mixed_fast_attn3(&l, prog_va, &FAST, AT, A0, k, ZONE).unwrap();
    for (fast, pick, va, regs) in [
        (FV4, TcPick::AttnTc16V4, ATC16V4, 72),
        (FPF, TcPick::AttnTc16V4pf, ATC16V4PF, 104),
    ] {
        let (zb, qb, pb) =
            ts::pack_mixed_fast_attn4(&l, prog_va, &FAST, AT, fast, A0, k, ZONE).unwrap();
        assert_eq!(qa, qb);
        assert_eq!(za.len(), zb.len());
        assert_eq!(count(&pb, pick), MINI.layers);
        assert_eq!(count(&pb, TcPick::AttnTc16), 0);
        for (x, y) in pa.iter().zip(&pb) {
            if *x == Some(TcPick::AttnTc16) {
                assert_eq!(*y, Some(pick));
            } else {
                assert_eq!(x, y);
            }
        }
        let diff: Vec<usize> = (0..za.len()).filter(|&i| za[i] != zb[i]).collect();
        assert!(
            !diff.is_empty() && diff.len() <= 2 * MINI.layers * 10,
            "{}",
            diff.len()
        );
        let in_qmd = |i: usize| {
            qb.iter().zip(&pb).any(|(qv, p)| {
                *p == Some(pick) && (i as u64) >= qv - ZONE && (i as u64) < qv - ZONE + 0x100
            })
        };
        assert!(diff.iter().all(|&i| in_qmd(i)));
        for (qv, p) in qb.iter().zip(&pb) {
            if *p == Some(pick) {
                assert_eq!(program_va(&zb, *qv), va);
                assert_eq!(program_va(&za, *qv), ATC16);
                assert_eq!(field(&zb, *qv, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V), regs);
                assert_eq!(field(&za, *qv, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V), 48);
            }
        }
        println!(
            "kt c263 W2 SIMULARE: mini: {} x {}, zona {} B la fel de lunga, {} octeti diferiti (adresa programului + registrele {regs})",
            MINI.layers,
            pick.name(),
            zb.len(),
            diff.len()
        );
    }
}

/// W3: the mini training (5 steps) with v4 = with v4pf = with tc16, bit for bit.
#[test]
fn t263_w3_training_same_bits() {
    let (tc, pt) = run(AT, None, &no_edit, NONE).unwrap();
    assert_eq!(count(&pt, TcPick::AttnTc16), MINI.layers);
    let bits = |x: &[f32], y: &[f32]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, r)| p.to_bits() == r.to_bits())
    };
    let mut n = 0;
    for (attn, fast, pick) in [
        (AT, FV4, TcPick::AttnTc16V4),
        (AT, FPF, TcPick::AttnTc16V4pf),
        (None, FV4, TcPick::AttnTc16V4),
        (None, FPF, TcPick::AttnTc16V4pf),
    ] {
        let (r, p) = run(attn, fast, &no_edit, NONE).unwrap();
        assert_eq!(count(&p, pick), MINI.layers);
        for (a, b) in r.losses.iter().zip(&tc.losses) {
            assert!(bits(a, b));
        }
        assert!(bits(&r.grad1, &tc.grad1));
        assert!(bits(&r.weights, &tc.weights));
        n = r.grad1.len() + r.weights.len() + r.losses.iter().map(Vec::len).sum::<usize>();
    }
    let first = mean(&tc.losses[0]);
    let last = mean(&tc.losses[STEPS as usize - 1]);
    assert!(last < first);
    println!(
        "kt c263 W3 SIMULARE: mini 5 pasi cu tc16_v4 = cu tc16_v4pf = cu tc16 bit cu bit ({n} valori; si fara tc16 incarcat); pierderea {first:.6} -> {last:.6}"
    );
}

/// W4: #513's mixed list (batch 2): 12 fast + 12 `Attn0Tc` + 147 `mm_tc_v4`; zone as attn3; the fast QMDs.
#[test]
fn t263_w4_list_513() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let k = Some(64 * KIB);
    let l = list_513(2);
    let (za, qa, _) = ts::pack_mixed_fast_attn3(&l, prog_va, &FAST, AT, A0, k, ZONE).unwrap();
    for (fast, pick, va, regs) in [
        (FV4, TcPick::AttnTc16V4, ATC16V4, 72),
        (FPF, TcPick::AttnTc16V4pf, ATC16V4PF, 104),
    ] {
        let (z, qm, p) =
            ts::pack_mixed_fast_attn4(&l, prog_va, &FAST, AT, fast, A0, k, ZONE).unwrap();
        assert_eq!(count(&p, pick), 12);
        assert_eq!(count(&p, TcPick::AttnTc16), 0);
        assert_eq!(count(&p, TcPick::Attn0Tc), 12);
        assert_eq!(count(&p, TcPick::V4), 147);
        assert_eq!(z.len(), za.len());
        assert_eq!(qm, qa);
        let mut seen = 0;
        for (pk, qva) in p.iter().zip(&qm) {
            if *pk != Some(pick) {
                continue;
            }
            seen += 1;
            let at = (qva - ZONE) as usize;
            assert_eq!(program_va(&z, *qva), va);
            assert_eq!(field(&z, *qva, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V), regs);
            assert_eq!(field(&z, *qva, q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH), 8);
            assert_eq!(field(&z, *qva, q::NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT), 12);
            assert_eq!(field(&z, *qva, q::NVC6C0_QMDV03_00_CTA_RASTER_DEPTH), 2);
            assert!(nvidia_ga104::attn_back_tile_tc16_v4::fits_cbuf(
                &z[at - 0x800..at]
            ));
            // The fast QMD differs from tc16's only in the program address and the register count.
            let (o, n) = (&za[at..at + 0x100], &z[at..at + 0x100]);
            let d = (0..0x100).filter(|&i| o[i] != n[i]).count();
            assert!(d <= 10, "{d}");
        }
        assert_eq!(seen, 12);
        println!(
            "kt c263 W4 SIMULARE: #513 lot 2: 12 {} + 12 attn_back_tile0_tc + 147 mm_tc_v4, zona {} B (cat attn3), {regs} registre",
            pick.name(),
            z.len()
        );
    }
}

/// W5: a misaligned shape falls back to tc16; refused if only the fast variant is loaded; a fast address not
/// 256-aligned refused; a zone where a misaligned launch's QMD points at v4 is refused by the stand-in.
#[test]
fn t263_w5_misaligned_falls_back() {
    let k = Some(64 * KIB);
    let mut l = mini_list();
    misalign(&mut l);
    for fast in [FV4, FPF] {
        let (_, _, p) =
            ts::pack_mixed_fast_attn4(&l, prog_va, &FAST, AT, fast, A0, k, ZONE).unwrap();
        assert_eq!(count(&p, TcPick::AttnTc16), MINI.layers);
        assert_eq!(
            count(&p, TcPick::AttnTc16V4) + count(&p, TcPick::AttnTc16V4pf),
            0
        );
        let e = ts::pack_mixed_fast_attn4(&l, prog_va, &FAST, None, fast, A0, k, ZONE).unwrap_err();
        assert!(format!("{e:?}").contains("nealiniata"), "{e:?}");
    }
    let good = mini_list();
    for bad_fast in [
        Some(AttnTcFast::V4(ATC16V4 + 16)),
        Some(AttnTcFast::V4pf(ATC16V4PF + 16)),
        Some(AttnTcFast::V4(0)),
    ] {
        assert!(
            ts::pack_mixed_fast_attn4(&good, prog_va, &FAST, AT, bad_fast, A0, k, ZONE).is_err()
        );
    }
    // The misaligned step itself runs on tc16 (dS read 4 bytes further; the step still finishes).
    let (fallback, pf) = run(AT, FV4, &misalign, NONE).unwrap();
    assert_eq!(count(&pf, TcPick::AttnTc16), MINI.layers);
    assert!(fallback.losses.iter().flatten().all(|x| x.is_finite()));
    // Planted: the QMD of a misaligned launch pointed at v4 by hand → the stand-in refuses.
    let to_v4 = &|z: &mut Vec<u8>, q: &[u64], p: &[Option<TcPick>]| {
        use nvidia_rm570::regs::qmd_v03_00 as qq;
        for (qva, pk) in q.iter().zip(p) {
            if *pk == Some(TcPick::AttnTc16) {
                let at = (qva - ZONE) as usize;
                let f = qq::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER;
                assert_eq!(f.lo % 8, 0);
                let b = at + (f.lo / 8) as usize;
                z[b..b + 4].copy_from_slice(&(ATC16V4 as u32).to_le_bytes());
            }
        }
    };
    let e = run(AT, FV4, &misalign, to_v4)
        .err()
        .expect("v4 pe forma nealiniata trebuia refuzat");
    assert!(e.contains("nealiniata"), "{e}");
    println!(
        "kt c263 W5 SIMULARE: forma nealiniata -> attn_back_tile_tc16; doar varianta rapida -> refuz; QMD mutat pe v4 -> {e}"
    );
}

/// W6: through RAM: attn4 with `None` = attn3 byte for byte; with the fast variant the picks are those without RAM.
#[test]
fn t263_w6_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let plan = plan_mini(&inputs).resident().unwrap();
    let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let list = mini_list();
    let k = Some(64 * KIB);
    let old =
        br::pack_mixed_fast_via_ram_attn3(&list, prog_va, &FAST, AT, A0, k, TC, ZONE, &v).unwrap();
    let new =
        br::pack_mixed_fast_via_ram_attn4(&list, prog_va, &FAST, AT, None, A0, k, TC, ZONE, &v)
            .unwrap();
    assert_eq!(old, new);
    for (fast, pick) in [(FV4, TcPick::AttnTc16V4), (FPF, TcPick::AttnTc16V4pf)] {
        let (_, _, pr) =
            br::pack_mixed_fast_via_ram_attn4(&list, prog_va, &FAST, AT, fast, A0, k, TC, ZONE, &v)
                .unwrap();
        let (_, _, pd) =
            ts::pack_mixed_fast_attn4(&list, prog_va, &FAST, AT, fast, A0, k, ZONE).unwrap();
        assert_eq!(pr, pd);
        assert_eq!(count(&pr, pick), MINI.layers);
    }
    println!(
        "kt c263 W6 SIMULARE: prin RAM: fara varianta rapida -> attn3 octet cu octet; cu v4 / v4pf aceleasi alegeri ca fara RAM"
    );
}
