//! C-157 p.2 / the centre's note of 25 Nov 14:40, point 1 — T240, SIMULARE: the **mixed** training step packed with the
//! **fast** `mm_tc` programs (`mm_tc_v4` where the product fits, else `mm_tc_pf`) and one shared-memory step of the SM
//! (64 KiB) for every launch: `train_step::pack_mixed_fast`. Plan and predictions written before the code
//! (`docs/C-157_P2_PAS_MIXT_RAPID.md`, commit `cc7621f`).
//!
//! Honesty: in the simulator the CPU models of `mm_tc_pf` / `mm_tc_v4` are `mm_tc`'s (the v4 one refuses misaligned
//! products), so "same bits" here checks the **path** (pick, arguments, accepted QMDs, refusals), not the board.
//! Nothing here runs on the RTX 3070 Ti.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS, LN_FWD_NEEDS,
    MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS, MM_TRAIN_GELU_NEEDS,
    MM_TRAIN_NEEDS, Needs, XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu,
    attn_back_tile0_on_cpu, attn_fwd_on_cpu, attn_fwd_tile_on_cpu, attn_p_on_cpu,
    attn_p_tile_on_cpu, colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu,
    gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu, ln_fwd_on_cpu, mm_tc_gelu_on_cpu,
    mm_tc_on_cpu, mm_tc_pf_on_cpu, mm_tc_v4_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu,
    opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::mm_tc_pf::MM_TC_PF;
use nvidia_ga104::mm_tc_v4::MM_TC_V4;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, LOSS_SCALE, LOSS_SCALE_513, MixedLaunch, PROGRAMS_ALL_FUSED2, Plan, Prog, TcPick,
    TcPrograms,
};
use nvidia_ga104::transfer;
use nvidia_rm570::QmdField;
use nvidia_rm570::regs::clc7c0 as c;
use nvidia_rm570::regs::qmd_v03_00 as q;

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
    cp
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Pack {
    /// Today's `pack_mixed_gelu` (`mm_tc` only, the one-block step).
    Old,
    /// `pack_mixed_fast` with these programs and this step.
    Fast(TcPrograms, Option<u32>),
}

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

/// Every float the step leaves behind, as bits: the losses of each step, the step-1 gradient, the last weights.
#[derive(Debug, PartialEq, Eq)]
struct Bits {
    losses: Vec<Vec<u32>>,
    grad1: Vec<u32>,
    weights: Vec<u32>,
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
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

/// `STEPS` mixed steps at S = `LOSS_SCALE` on the simulated board; `edit` may change the list of every step (plants).
/// Returns the bits and the picks of step 1 (`None` for `Pack::Old`).
fn run(
    layout: Layout,
    pack: Pack,
    edit: &dyn Fn(&mut Vec<MixedLaunch>),
) -> Result<(Bits, Vec<Option<TcPick>>), String> {
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
    let mut out = Bits {
        losses: Vec::new(),
        grad1: Vec::new(),
        weights: Vec::new(),
    };
    let mut picks1 = Vec::new();
    for step in 1..=STEPS {
        let mut list = plan
            .step_mixed(Recipe::AdamW513, &opt, step, LOSS_SCALE)
            .unwrap();
        edit(&mut list);
        let (zone, qmds) = match pack {
            Pack::Old => {
                ts::pack_mixed_gelu(&list, prog_va, TC, TCG, ZONE).map_err(|e| format!("{e:?}"))?
            }
            Pack::Fast(progs, s) => {
                let (z, qm, p) = ts::pack_mixed_fast(&list, prog_va, &progs, s, ZONE)
                    .map_err(|e| format!("{e:?}"))?;
                if step == 1 {
                    picks1 = p;
                }
                (z, qm)
            }
        };
        submit(&mut cp, &mut first, &zone, &qmds).map_err(|e| format!("pasul {step}: {e}"))?;
        out.losses
            .push(bits(&cp.mem.read_f32s(plan.loss_rows, B * T).unwrap()));
        if step == 1 {
            out.grad1 = bits(&cp.mem.read_f32s(plan.g, plan.n_params).unwrap());
        }
    }
    out.weights = bits(&cp.mem.read_f32s(plan.w, plan.n_params).unwrap());
    Ok((out, picks1))
}

fn count(picks: &[Option<TcPick>], p: TcPick) -> usize {
    picks.iter().filter(|x| **x == Some(p)).count()
}

/// Prediction 1: the fast packing (v4 / pf at 64 KiB, and pf alone at the old step) = today's packing, bit for bit,
/// on the three plans of `antrenare mini`.
#[test]
fn t240_p1_same_bits_as_mm_tc() {
    let none = &|_: &mut Vec<MixedLaunch>| {};
    for layout in [Layout::Simple, Layout::Tiles3, Layout::Tiles3Both] {
        let (old, _) = run(layout, Pack::Old, none).unwrap();
        let (fast, picks) = run(layout, Pack::Fast(FAST, Some(64 * KIB)), none).unwrap();
        let pf_only = TcPrograms { v4: 0, ..FAST };
        let (pf, picks_pf) = run(layout, Pack::Fast(pf_only, None), none).unwrap();
        println!(
            "kt c240 mini {layout:?} SIMULARE: rapid@64KiB {} v4 + {} pf + {} tc + {} gelu; doar pf {} pf; aceiasi biti: {} / {}",
            count(&picks, TcPick::V4),
            count(&picks, TcPick::Pf),
            count(&picks, TcPick::Tc),
            count(&picks, TcPick::Gelu),
            count(&picks_pf, TcPick::Pf),
            old == fast,
            old == pf
        );
        assert_eq!(old, fast, "{layout:?}: rapid la 64 KiB difera de mm_tc");
        assert_eq!(old, pf, "{layout:?}: mm_tc_pf difera de mm_tc");
        assert_eq!(count(&picks, TcPick::Tc), 0);
        assert!(count(&picks, TcPick::V4) > 0);
        assert_eq!(count(&picks_pf, TcPick::V4), 0);
        assert_eq!(count(&picks_pf, TcPick::Tc), 0);
        assert!(
            old.losses
                .iter()
                .flatten()
                .all(|b| f32::from_bits(*b).is_finite())
        );
    }
}

fn mini_list(layout: Layout) -> Vec<MixedLaunch> {
    let (inputs, _) = batch();
    let params = initial();
    let plan = plan_of(layout, &inputs);
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    plan.step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE)
        .unwrap()
}

/// Prediction 2: no fast program and no step → byte for byte `pack_mixed_gelu`.
#[test]
fn t240_p2_without_fast_programs_same_bytes() {
    for layout in [Layout::Simple, Layout::Tiles3, Layout::Tiles3Both] {
        let list = mini_list(layout);
        let (a, qa) = ts::pack_mixed_gelu(&list, prog_va, TC, TCG, ZONE).unwrap();
        let only = TcPrograms {
            pf: 0,
            v4: 0,
            ..FAST
        };
        let (b, qb, picks) = ts::pack_mixed_fast(&list, prog_va, &only, None, ZONE).unwrap();
        assert_eq!(a, b, "{layout:?}");
        assert_eq!(qa, qb);
        assert_eq!(count(&picks, TcPick::Pf) + count(&picks, TcPick::V4), 0);
    }
}

fn list_513(batch: usize, glued: bool) -> Vec<MixedLaunch> {
    let m = train::M513;
    let inputs: Vec<u32> = (0..batch as u32 * 512)
        .map(|i| (i * 7919) % 16_384)
        .collect();
    let base = VA + (1 << 30);
    let plan = if glued {
        Plan::new_tiles3(m, batch, 512, &inputs, base)
            .unwrap()
            .with_fused_back()
            .with_fused_gelu()
            .with_mm_tile_gelu()
            .with_colsum_split()
    } else {
        Plan::new_tiles3(m, batch, 512, &inputs, base)
            .unwrap()
            .with_mm_tile()
            .with_colsum_split()
    };
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    plan.step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap()
}

fn flop(l: &MixedLaunch) -> u64 {
    match l {
        MixedLaunch::Tc { args, .. } => args.flop(),
        MixedLaunch::TcGelu { args, .. } => 2 * args.m as u64 * args.n as u64 * args.k as u64,
        MixedLaunch::TcGeluFwd { args, .. } => args.tc.flop(),
        MixedLaunch::Std(_)
        | MixedLaunch::AttnTc(_)
        | MixedLaunch::Attn0Tc(_)
        | MixedLaunch::AttnFwdTc(_) => 0,
    }
}

/// Prediction 3 (+ 6, DEDUS arithmetic): #513's mixed lists — every `mm_tc` product fits `mm_tc_v4`; the FLOP per step.
#[test]
fn t240_p3_list_513_all_on_v4() {
    for (bsz, glued, v4_want, gelu_want) in
        [(2, false, 147, 0), (6, false, 147, 0), (2, true, 135, 12)]
    {
        let list = list_513(bsz, glued);
        let (zone, qmds, picks) =
            ts::pack_mixed_fast(&list, prog_va, &FAST, Some(64 * KIB), ZONE).unwrap();
        let (zone_old, _) = ts::pack_mixed_gelu(&list, prog_va, TC, TCG, ZONE).unwrap();
        let fl: u64 = list.iter().map(flop).sum();
        let tf = fl as f64 / 1e12;
        println!(
            "kt c240 #513 lot {bsz}x512 lipit={glued} SIMULARE: {} v4 + {} pf + {} gelu, zona {:.0} KiB; inmultirile {:.3} TFLOP/pas -> DEDUS {:.0} ms la 1,03 / {:.0} la 2,75 / {:.0}-{:.0} la 5,2-4,4 / {:.0} la 8 TFLOP/s",
            count(&picks, TcPick::V4),
            count(&picks, TcPick::Pf),
            count(&picks, TcPick::Gelu),
            zone.len() as f64 / 1024.0,
            tf,
            tf / 1.03e-3,
            tf / 2.75e-3,
            tf / 5.2e-3,
            tf / 4.4e-3,
            tf / 8e-3,
        );
        assert_eq!(
            count(&picks, TcPick::V4),
            v4_want,
            "lot {bsz} lipit={glued}"
        );
        assert_eq!(count(&picks, TcPick::Pf), 0);
        assert_eq!(count(&picks, TcPick::Gelu), gelu_want);
        assert_eq!(zone.len(), zone_old.len());
        assert_eq!(qmds.len(), list.len());
    }
}

fn bit(qmd: &[u8], i: u32) -> bool {
    qmd[(i / 8) as usize] >> (i % 8) & 1 == 1
}

fn inside(i: u32, f: QmdField) -> bool {
    (f.lo..=f.hi).contains(&i)
}

/// Prediction 4: QMD bits that change from the old packing — for an `mm_tc` launch moved to v4: the program address,
/// the register count and MIN / TARGET of the step; for an F32 launch: MIN / TARGET only. **Partly wrong** (written
/// before the run): an F32 launch **without shared memory** also gets MAX = 64 KiB (0 before), exactly as
/// `pack_with_sm_step` does (T232, on the board in os-431). Asserted: what was found.
#[test]
fn t240_p4_qmd_differs_only_where_expected() {
    let list = list_513(2, false);
    let (a, qa) = ts::pack_mixed_gelu(&list, prog_va, TC, TCG, ZONE).unwrap();
    let (b, qb, picks) = ts::pack_mixed_fast(&list, prog_va, &FAST, Some(64 * KIB), ZONE).unwrap();
    assert_eq!(qa, qb);
    let step = [
        q::NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE,
        q::NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE,
    ];
    let max = q::NVC6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE;
    let mut max_changed = 0;
    let prog = [
        q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER,
        q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER,
        q::NVC6C0_QMDV03_00_REGISTER_COUNT_V,
    ];
    let (mut seen_prog, mut seen_step) = (0, 0);
    for ((qva, l), p) in qa.iter().zip(&list).zip(&picks) {
        let at = (qva - ZONE) as usize;
        let cb = l.slot() as usize - 0x100;
        // The cbuf before the QMD is the same.
        assert_eq!(a[at - cb..at], b[at - cb..at]);
        let (x, y) = (&a[at..at + 0x100], &b[at..at + 0x100]);
        let mut any_prog = false;
        for i in 0..0x800u32 {
            if bit(x, i) != bit(y, i) {
                let in_step = step.iter().any(|&f| inside(i, f));
                let in_prog = prog.iter().any(|&f| inside(i, f));
                if inside(i, max) {
                    // Only F32 launches without shared memory (MAX was 0).
                    let MixedLaunch::Std(f) = l else {
                        panic!("{}: MAX schimbat la mm_tc*", l.tag())
                    };
                    assert!(f.prog.shared().is_none(), "{}: MAX", l.tag());
                    max_changed += 1;
                    continue;
                }
                match p {
                    None => assert!(in_step, "lansare F32 {}: bitul {i}", l.tag()),
                    Some(_) => assert!(in_step || in_prog, "{}: bitul {i}", l.tag()),
                }
                any_prog |= in_prog;
                seen_step += usize::from(in_step);
            }
        }
        if *p == Some(TcPick::V4) {
            assert!(any_prog, "{}: v4 fara alt program", l.tag());
            seen_prog += 1;
        }
    }
    assert_eq!(seen_prog, 147);
    assert!(seen_step > 0);
    println!(
        "kt c240 qmd #513 SIMULARE: 147 v4 (adresa + registre + treapta), MAX schimbat pe {max_changed} biti (lansari F32 fara memorie comuna)"
    );
}

/// Prediction 5: planted mistakes are caught.
#[test]
fn t240_p5_plants() {
    // (a) A moved by 4 bytes → `mm_tc_pf`, not v4.
    let list = mini_list(Layout::Tiles3);
    let k = list
        .iter()
        .position(|l| matches!(l, MixedLaunch::Tc { .. }))
        .unwrap();
    let mut moved = list.clone();
    if let MixedLaunch::Tc { args, .. } = &mut moved[k] {
        args.a += 4;
    }
    assert_eq!(ts::tc_pick(&list[k], &FAST), Some(TcPick::V4));
    assert_eq!(ts::tc_pick(&moved[k], &FAST), Some(TcPick::Pf));
    let shift = &move |l: &mut Vec<MixedLaunch>| {
        let k = l
            .iter()
            .position(|x| matches!(x, MixedLaunch::Tc { .. }))
            .unwrap();
        if let MixedLaunch::Tc { args, .. } = &mut l[k] {
            args.a += 4;
        }
    };
    let (_, picks) = run(Layout::Tiles3, Pack::Fast(FAST, Some(64 * KIB)), shift).unwrap();
    assert_eq!(count(&picks, TcPick::Pf), 1);
    // (b) v4 forced on that misaligned product (loaded where `mm_tc` should be) → the simulated board refuses.
    let forced = TcPrograms {
        tc: V4,
        pf: 0,
        v4: 0,
        gelu: TCG,
    };
    let e = run(Layout::Tiles3, Pack::Fast(forced, None), shift).unwrap_err();
    assert!(e.contains("nealiniat"), "{e}");
    // (c) A step below the 24 KiB of `mm_tc*` is refused before anything is sent; so is a step that is not one.
    for s in [16 * KIB, 48 * KIB] {
        assert!(ts::pack_mixed_fast(&list, prog_va, &FAST, Some(s), ZONE).is_err());
    }
    // (d) A launch falls to `mm_tc` but `mm_tc` is not loaded → refused; misaligned addresses → refused.
    let no_tc = TcPrograms {
        tc: 0,
        pf: 0,
        ..FAST
    };
    assert!(ts::pack_mixed_fast(&moved, prog_va, &no_tc, None, ZONE).is_err());
    let bad = TcPrograms { v4: V4 + 4, ..FAST };
    assert!(ts::pack_mixed_fast(&list, prog_va, &bad, None, ZONE).is_err());
    let no_gelu = TcPrograms { gelu: 0, ..FAST };
    assert!(
        ts::pack_mixed_fast(
            &mini_list(Layout::Tiles3Both),
            prog_va,
            &no_gelu,
            None,
            ZONE
        )
        .is_err()
    );
    assert!(ts::pack_mixed_fast(&[], prog_va, &FAST, None, ZONE).is_err());
}
