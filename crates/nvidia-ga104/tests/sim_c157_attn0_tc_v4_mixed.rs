//! C-157 — T262, SIMULARE: `attn_back_tile0_tc_v4` (dS of the attention backward with 128-bit loads / stores, the
//! same arithmetic as `attn_back_tile0_tc`) linked in the **fast mixed step**: `train_step::pack_mixed_fast_attn3`
//! picks v4 for every aligned dS launch when it is loaded, else `attn_back_tile0_tc`. Plan and predictions V1–V7
//! written before these tests (`docs/C-157_ATTN_BACK0_TC_V4_LEGAT.md`, commit `140df52`).
//!
//! Honesty: both stand-ins are `gpu_compute::attn_tc::back0_model`, a HYPOTHESIS of what the programs compute (the
//! tensor-core adder fitted in os-335). Nothing here runs on the RTX 3070 Ti.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC_NEEDS, ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE0_TC_NEEDS,
    ATTN_BACK_TILE0_TC_V4_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS,
    LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS,
    MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, Needs, XENT_NEEDS, attn_back_on_cpu,
    attn_back_tile_on_cpu, attn_back_tile_tc_on_cpu, attn_back_tile_tc16_on_cpu,
    attn_back_tile0_on_cpu, attn_back_tile0_tc_on_cpu, attn_back_tile0_tc_v4_on_cpu,
    attn_fwd_on_cpu, attn_fwd_tile_on_cpu, attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu,
    emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu,
    ln_back_res_on_cpu, ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu,
    mm_tc_v4_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu,
    xent_on_cpu,
};
use nvidia_ga104::attn_back_tile_tc::ATTN_BACK_TILE_TC;
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::attn_back_tile0_tc::ATTN_BACK_TILE0_TC;
use nvidia_ga104::attn_back_tile0_tc_v4::ATTN_BACK_TILE0_TC_V4;
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::mm_tc_pf::MM_TC_PF;
use nvidia_ga104::mm_tc_v4::MM_TC_V4;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, Attn0Programs, AttnTcProgram, LOSS_SCALE, LOSS_SCALE_513, MixedLaunch,
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
const ATC: u64 = PROGS + 0x1_0000 * 27;
const ATC16: u64 = PROGS + 0x1_0000 * 28;
const ATC0: u64 = PROGS + 0x1_0000 * 29;
const ATC0V4: u64 = PROGS + 0x1_0000 * 30;
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
    cp.register_program_with_needs(
        ATTN_BACK_TILE0_TC,
        attn_back_tile0_tc_on_cpu,
        ATTN_BACK_TILE0_TC_NEEDS,
    );
    cp.mem.write(ATC0, ATTN_BACK_TILE0_TC);
    cp.register_program_with_needs(
        ATTN_BACK_TILE0_TC_V4,
        attn_back_tile0_tc_v4_on_cpu,
        ATTN_BACK_TILE0_TC_V4_NEEDS,
    );
    cp.mem.write(ATC0V4, ATTN_BACK_TILE0_TC_V4);
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

fn mean(v: &[f32]) -> f64 {
    v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
}

const DSOFF: usize = 0x40;

type ZoneEdit = dyn Fn(&mut Vec<u8>, &[u64], &[Option<TcPick>]);

const NONE: &ZoneEdit = &|_, _, _| {};

const BOTH: Attn0Programs = Attn0Programs {
    tc: ATC0,
    v4: ATC0V4,
};

/// `STEPS` mixed steps of the mini model on the simulated board, both attention-backward programs on the tensor
/// cores (`attn_tc_list` + `attn0_tc_list`), packed by `pack_mixed_fast_attn3` with `attn0` at the 64 KiB step;
/// `list_edit` may change each list before packing, `zone_edit` the packed zone (plants).
fn run(
    attn0: Attn0Programs,
    list_edit: &dyn Fn(&mut [MixedLaunch]),
    zone_edit: &ZoneEdit,
) -> Result<(Out, Vec<Option<TcPick>>), String> {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = plan_of(Layout::Tiles3, &inputs);
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
    let at = Some(AttnTcProgram::Tc(ATC));
    for step in 1..=STEPS {
        let list = plan
            .step_mixed(Recipe::AdamW513, &opt, step, LOSS_SCALE)
            .unwrap();
        let mut list = ts::attn0_tc_list(ts::attn_tc_list(list).unwrap()).unwrap();
        list_edit(&mut list);
        let (mut zone, qmds, p) =
            ts::pack_mixed_fast_attn3(&list, prog_va, &FAST, at, attn0, Some(64 * KIB), ZONE)
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

/// The dS argument of every `Attn0Tc` moved by 4 bytes (F32 layout: dS at 0x40): the shape stops being aligned.
fn misalign(list: &mut [MixedLaunch]) {
    for l in list.iter_mut() {
        if let MixedLaunch::Attn0Tc(x) = l {
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

fn program_va(zone: &[u8], qva: u64) -> u64 {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let at = (qva - ZONE) as usize;
    let qmd = &zone[at..at + 0x100];
    let lo = q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER;
    let hi = q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER;
    qmd_field(qmd, hi.lo, hi.hi) << 32 | qmd_field(qmd, lo.lo, lo.hi)
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

/// V1: without v4 nothing changes: `pack_mixed_fast_attn3` with `v4 = 0` = `pack_mixed_fast_attn2` byte for byte.
#[test]
fn t262_v1_without_v4_same_bytes() {
    let k = Some(64 * KIB);
    let mini = ts::attn0_tc_list(ts::attn_tc_list(mini_list(Layout::Tiles3)).unwrap()).unwrap();
    let big = ts::attn0_tc_list(ts::attn_tc_list(list_513(2)).unwrap()).unwrap();
    for (l, at) in [
        (&mini, AttnTcProgram::Tc(ATC)),
        (&big, AttnTcProgram::Tc16(ATC16)),
    ] {
        let a = ts::pack_mixed_fast_attn2(l, prog_va, &FAST, Some(at), ATC0, k, ZONE).unwrap();
        let b = ts::pack_mixed_fast_attn3(
            l,
            prog_va,
            &FAST,
            Some(at),
            Attn0Programs { tc: ATC0, v4: 0 },
            k,
            ZONE,
        )
        .unwrap();
        assert_eq!(a, b);
        assert_eq!(count(&b.2, TcPick::Attn0TcV4), 0);
    }
    println!("kt c262 V1 SIMULARE: fara v4, attn3 = attn2 octet cu octet (mini si #513)");
}

/// V2: on the mini list with both loaded, the dS launches pick v4; same zone length, only the program address bytes
/// of those QMDs differ.
#[test]
fn t262_v2_mini_picks_v4() {
    let k = Some(64 * KIB);
    let l = ts::attn0_tc_list(ts::attn_tc_list(mini_list(Layout::Tiles3)).unwrap()).unwrap();
    let at = Some(AttnTcProgram::Tc(ATC));
    let (za, qa, pa) = ts::pack_mixed_fast_attn2(&l, prog_va, &FAST, at, ATC0, k, ZONE).unwrap();
    let (zb, qb, pb) = ts::pack_mixed_fast_attn3(&l, prog_va, &FAST, at, BOTH, k, ZONE).unwrap();
    assert_eq!(qa, qb);
    assert_eq!(za.len(), zb.len());
    assert_eq!(count(&pb, TcPick::Attn0TcV4), MINI.layers);
    assert_eq!(count(&pb, TcPick::Attn0Tc), 0);
    for (x, y) in pa.iter().zip(&pb) {
        if *x == Some(TcPick::Attn0Tc) {
            assert_eq!(*y, Some(TcPick::Attn0TcV4));
        } else {
            assert_eq!(x, y);
        }
    }
    let diff: Vec<usize> = (0..za.len()).filter(|&i| za[i] != zb[i]).collect();
    assert!(
        !diff.is_empty() && diff.len() <= 2 * MINI.layers * 8,
        "{}",
        diff.len()
    );
    let in_qmd = |i: usize| {
        qb.iter().zip(&pb).any(|(q, p)| {
            *p == Some(TcPick::Attn0TcV4) && (i as u64) >= q - ZONE && (i as u64) < q - ZONE + 0x100
        })
    };
    assert!(diff.iter().all(|&i| in_qmd(i)));
    for (q, p) in qb.iter().zip(&pb) {
        if *p == Some(TcPick::Attn0TcV4) {
            assert_eq!(program_va(&zb, *q), ATC0V4);
            assert_eq!(program_va(&za, *q), ATC0);
        }
    }
    println!(
        "kt c262 V2 SIMULARE: mini: {} Attn0TcV4, zona {} B la fel de lunga, {} octeti diferiti (doar adresa programului)",
        MINI.layers,
        zb.len(),
        diff.len()
    );
}

/// V3: the mini training (5 steps) with v4 = with tc, bit for bit.
#[test]
fn t262_v3_training_same_bits() {
    let (tc, pt) = run(Attn0Programs { tc: ATC0, v4: 0 }, &no_edit, NONE).unwrap();
    let (v4, pv) = run(BOTH, &no_edit, NONE).unwrap();
    let (only_v4, _) = run(Attn0Programs { tc: 0, v4: ATC0V4 }, &no_edit, NONE).unwrap();
    assert_eq!(count(&pt, TcPick::Attn0Tc), MINI.layers);
    assert_eq!(count(&pv, TcPick::Attn0TcV4), MINI.layers);
    let bits = |x: &[f32], y: &[f32]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, r)| p.to_bits() == r.to_bits())
    };
    for r in [&v4, &only_v4] {
        for (a, b) in r.losses.iter().zip(&tc.losses) {
            assert!(bits(a, b));
        }
        assert!(bits(&r.grad1, &tc.grad1));
        assert!(bits(&r.weights, &tc.weights));
    }
    let first = mean(&v4.losses[0]);
    let last = mean(&v4.losses[STEPS as usize - 1]);
    assert!(last < first);
    let n = tc.grad1.len() + tc.weights.len() + tc.losses.iter().map(Vec::len).sum::<usize>();
    println!(
        "kt c262 V3 SIMULARE: mini 5 pasi cu attn_back_tile0_tc_v4 = cu attn_back_tile0_tc bit cu bit ({n} valori); pierderea {first:.6} -> {last:.6}"
    );
}

/// V4: #513's mixed list (batch 2): 12 `Attn0TcV4` + 12 `AttnTc16` + 147 `mm_tc_v4`; zone as attn2; the v4 QMDs.
#[test]
fn t262_v4_list_513() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let k = Some(64 * KIB);
    let moved = ts::attn0_tc_list(ts::attn_tc_list(list_513(2)).unwrap()).unwrap();
    let at = Some(AttnTcProgram::Tc16(ATC16));
    let (za, _, _) = ts::pack_mixed_fast_attn2(&moved, prog_va, &FAST, at, ATC0, k, ZONE).unwrap();
    let (z, qm, p) = ts::pack_mixed_fast_attn3(&moved, prog_va, &FAST, at, BOTH, k, ZONE).unwrap();
    assert_eq!(count(&p, TcPick::Attn0TcV4), 12);
    assert_eq!(count(&p, TcPick::Attn0Tc), 0);
    assert_eq!(count(&p, TcPick::AttnTc16), 12);
    assert_eq!(count(&p, TcPick::V4), 147);
    assert_eq!(z.len(), za.len());
    let f = |x: nvidia_rm570::QmdField, qmd: &[u8]| qmd_field(qmd, x.lo, x.hi);
    let mut seen = 0;
    for (pk, qva) in p.iter().zip(&qm) {
        if *pk != Some(TcPick::Attn0TcV4) {
            continue;
        }
        seen += 1;
        let at = (qva - ZONE) as usize;
        let qmd = &z[at..at + 0x100];
        assert_eq!(program_va(&z, *qva), ATC0V4);
        assert_eq!(f(q::NVC6C0_QMDV03_00_REGISTER_COUNT_V, qmd), 56);
        assert_eq!(f(q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH, qmd), 8);
        assert_eq!(f(q::NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT, qmd), 12);
        assert_eq!(f(q::NVC6C0_QMDV03_00_CTA_RASTER_DEPTH, qmd), 2);
        assert!(nvidia_ga104::attn_back_tile0_tc_v4::fits_cbuf(
            &z[at - 0x800..at]
        ));
    }
    assert_eq!(seen, 12);
    println!(
        "kt c262 V4 SIMULARE: #513 lot 2: 12 attn_back_tile0_tc_v4 + 12 attn_back_tile_tc16 + 147 mm_tc_v4, zona {} B (cat attn2)",
        z.len()
    );
}

/// V5: a misaligned shape falls back to tc; refused if only v4 is loaded; a v4 address not 256-aligned refused; a
/// zone where a misaligned launch's QMD points at v4 is refused by the stand-in.
#[test]
fn t262_v5_misaligned_falls_back() {
    let k = Some(64 * KIB);
    let at = Some(AttnTcProgram::Tc(ATC));
    let mut l = ts::attn0_tc_list(ts::attn_tc_list(mini_list(Layout::Tiles3)).unwrap()).unwrap();
    misalign(&mut l);
    let (_, _, p) = ts::pack_mixed_fast_attn3(&l, prog_va, &FAST, at, BOTH, k, ZONE).unwrap();
    assert_eq!(count(&p, TcPick::Attn0Tc), MINI.layers);
    assert_eq!(count(&p, TcPick::Attn0TcV4), 0);
    let e = ts::pack_mixed_fast_attn3(
        &l,
        prog_va,
        &FAST,
        at,
        Attn0Programs { tc: 0, v4: ATC0V4 },
        k,
        ZONE,
    )
    .unwrap_err();
    assert!(format!("{e:?}").contains("nealiniata"), "{e:?}");
    let good = ts::attn0_tc_list(ts::attn_tc_list(mini_list(Layout::Tiles3)).unwrap()).unwrap();
    assert!(
        ts::pack_mixed_fast_attn3(
            &good,
            prog_va,
            &FAST,
            at,
            Attn0Programs {
                tc: ATC0,
                v4: ATC0V4 + 16
            },
            k,
            ZONE
        )
        .is_err()
    );
    assert!(
        ts::pack_mixed_fast_attn3(
            &good,
            prog_va,
            &FAST,
            at,
            Attn0Programs { tc: 0, v4: 0 },
            k,
            ZONE
        )
        .is_err()
    );
    // The misaligned step itself runs on tc (the dS rows land 4 bytes further; the step still finishes).
    let (fallback, pf) = run(BOTH, &misalign, NONE).unwrap();
    assert_eq!(count(&pf, TcPick::Attn0Tc), MINI.layers);
    assert!(fallback.losses.iter().flatten().all(|x| x.is_finite()));
    // Planted: the QMD of a misaligned launch pointed at v4 by hand → the stand-in refuses.
    let to_v4 = &|z: &mut Vec<u8>, q: &[u64], p: &[Option<TcPick>]| {
        use nvidia_rm570::regs::qmd_v03_00 as qq;
        for (qva, pk) in q.iter().zip(p) {
            if *pk == Some(TcPick::Attn0Tc) {
                let at = (qva - ZONE) as usize;
                let f = qq::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER;
                assert_eq!(f.lo % 8, 0);
                let b = at + (f.lo / 8) as usize;
                z[b..b + 4].copy_from_slice(&(ATC0V4 as u32).to_le_bytes());
            }
        }
    };
    let e = run(BOTH, &misalign, to_v4)
        .err()
        .expect("v4 pe forma nealiniata trebuia refuzat");
    assert!(e.contains("nealiniata"), "{e}");
    println!(
        "kt c262 V5 SIMULARE: forma nealiniata -> attn_back_tile0_tc; doar v4 -> refuz; QMD mutat pe v4 -> {e}"
    );
}

/// V6: through RAM: attn3 with `v4 = 0` = attn2 byte for byte; with v4 the picks are those without RAM.
#[test]
fn t262_v6_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let plan = plan_of(Layout::Tiles3, &inputs).resident().unwrap();
    let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let list = ts::attn0_tc_list(ts::attn_tc_list(mini_list(Layout::Tiles3)).unwrap()).unwrap();
    let at = Some(AttnTcProgram::Tc16(ATC16));
    let k = Some(64 * KIB);
    let old = br::pack_mixed_fast_via_ram_attn2(&list, prog_va, &FAST, at, ATC0, k, TC, ZONE, &v)
        .unwrap();
    let new = br::pack_mixed_fast_via_ram_attn3(
        &list,
        prog_va,
        &FAST,
        at,
        Attn0Programs { tc: ATC0, v4: 0 },
        k,
        TC,
        ZONE,
        &v,
    )
    .unwrap();
    assert_eq!(old, new);
    let (_, _, pr) =
        br::pack_mixed_fast_via_ram_attn3(&list, prog_va, &FAST, at, BOTH, k, TC, ZONE, &v)
            .unwrap();
    let (_, _, pd) = ts::pack_mixed_fast_attn3(&list, prog_va, &FAST, at, BOTH, k, ZONE).unwrap();
    assert_eq!(pr, pd);
    assert_eq!(count(&pr, TcPick::Attn0TcV4), MINI.layers);
    println!(
        "kt c262 V6 SIMULARE: prin RAM: v4 = 0 -> attn2 octet cu octet; cu v4 aceleasi alegeri ca fara RAM"
    );
}
