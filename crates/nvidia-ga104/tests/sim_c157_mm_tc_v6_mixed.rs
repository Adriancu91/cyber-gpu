//! C-157 p.2 (session 264) — T277, SIMULARE: `mm_tc_v6` (v5 with the f16 stores of the loop paired) linked in the
//! **fast mixed step**: `train_step::pack_mixed_fast_attn6` sends every aligned `mm_tc` launch to `mm_tc_v6` when it
//! is loaded (before `mm_tc_v5`), else the rule of `attn5`. Predictions L1–L4 written before these tests
//! (`docs/C-157_MM_TC_V6_IN_PAS.md` §3).
//!
//! Honesty: the `mm_tc_v6` stand-in is `mm_tc`'s arithmetic (T276: the same fragments as v5 lane by lane). `mm_tc_v6`
//! has **not run on the board** yet (probe P2zzg). Nothing here runs on the RTX 3070 Ti.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS,
    LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS, MM_TC_V5_NEEDS,
    MM_TC_V6_NEEDS, MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, Needs, XENT_NEEDS, attn_back_on_cpu,
    attn_back_tile_on_cpu, attn_back_tile_tc16_on_cpu, attn_back_tile_tc16_v4_on_cpu,
    attn_back_tile_tc16_v4pf_on_cpu, attn_back_tile0_on_cpu, attn_back_tile0_tc_on_cpu,
    attn_fwd_on_cpu, attn_fwd_tile_on_cpu, attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu,
    emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu,
    ln_back_res_on_cpu, ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu,
    mm_tc_v4_on_cpu, mm_tc_v5_on_cpu, mm_tc_v6_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu,
    opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
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
const V5: u64 = PROGS + 0x1_0000 * 33;
const V6: u64 = PROGS + 0x1_0000 * 34;
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
/// cores, packed by `pack_mixed_fast_attn6` at the 64 KiB step.
fn run(
    progs: &TcPrograms,
    v5: u64,
    v6: u64,
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
        let (mut zone, qmds, p) = ts::pack_mixed_fast_attn6(
            &list,
            prog_va,
            progs,
            v5,
            v6,
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

/// Every `mm_tc` launch's picks: (v4, pf, tc, v5, v6).
fn mm_counts(p: &[Option<TcPick>]) -> (usize, usize, usize, usize, usize) {
    (
        count(p, TcPick::V4),
        count(p, TcPick::Pf),
        count(p, TcPick::Tc),
        count(p, TcPick::V5),
        count(p, TcPick::V6),
    )
}

/// L1: without `mm_tc_v6` nothing changes: attn6 with `v6 = 0` = attn5 byte for byte (mini and #513, with and
/// without v5, with and without the attention fast variant).
#[test]
fn t277_l1_without_v6_same_bytes() {
    let k = Some(64 * KIB);
    for l in [mini_list(), list_513(2)] {
        for v5 in [0, V5] {
            for fast in [None, FV4] {
                let a = ts::pack_mixed_fast_attn5(&l, prog_va, &FAST, v5, AT, fast, A0, k, ZONE)
                    .unwrap();
                let b = ts::pack_mixed_fast_attn6(&l, prog_va, &FAST, v5, 0, AT, fast, A0, k, ZONE)
                    .unwrap();
                assert_eq!(a, b);
                assert_eq!(count(&b.2, TcPick::V6), 0);
            }
        }
    }
    assert_eq!(TcPick::V6.name(), "mm_tc_v6");
    println!(
        "kt c277 L1 SIMULARE: fara mm_tc_v6, attn6 = attn5 octet cu octet (mini si #513, cu si fara v5)"
    );
}

/// L2: with `mm_tc_v6` every launch that went to v5 (or to v4 when v5 is not loaded) goes to v6, nothing else moves;
/// the zone keeps its length and QMD addresses; a v6 QMD differs from the v5 one only in the program address.
#[test]
fn t277_l2_picks_and_qmds() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let k = Some(64 * KIB);
    for (name, l, expect) in [
        ("mini", mini_list(), None),
        ("#513 lot 2", list_513(2), Some(147)),
    ] {
        let (za, qa, pa) =
            ts::pack_mixed_fast_attn5(&l, prog_va, &FAST, V5, AT, FV4, A0, k, ZONE).unwrap();
        let (zb, qb, pb) =
            ts::pack_mixed_fast_attn6(&l, prog_va, &FAST, V5, V6, AT, FV4, A0, k, ZONE).unwrap();
        let (_, pfa, tca, v5a, _) = mm_counts(&pa);
        let (v4b, pfb, tcb, v5b, v6b) = mm_counts(&pb);
        if let Some(n) = expect {
            assert_eq!(v5a, n);
        }
        assert!(v5a > 0, "{name}: niciun mm_tc_v5 de mutat");
        assert_eq!((v4b, pfb, tcb, v5b, v6b), (0, pfa, tca, 0, v5a));
        assert_eq!(qa, qb);
        assert_eq!(za.len(), zb.len());
        let mut diff_bytes = 0;
        for ((qv, x), y) in qb.iter().zip(&pa).zip(&pb) {
            let at = (qv - ZONE) as usize;
            if *x == Some(TcPick::V5) {
                assert_eq!(*y, Some(TcPick::V6));
                assert_eq!(program_va(&zb, *qv), V6);
                assert_eq!(program_va(&za, *qv), V5);
                assert_eq!(
                    field(&zb, *qv, q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE),
                    12_288
                );
                assert_eq!(field(&zb, *qv, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V), 128);
                let (o, n) = (&za[at..at + 0x100], &zb[at..at + 0x100]);
                let d = (0..0x100).filter(|&i| o[i] != n[i]).count();
                assert!(d <= 4, "{d}");
                diff_bytes += d;
                assert_eq!(&za[at - 0x800..at], &zb[at - 0x800..at]);
            } else {
                assert_eq!(x, y);
                assert_eq!(&za[at..at + 0x100], &zb[at..at + 0x100]);
            }
        }
        // Without v5 loaded the same launches go from v4 to v6.
        let (_, _, p4) =
            ts::pack_mixed_fast_attn6(&l, prog_va, &FAST, 0, V6, AT, FV4, A0, k, ZONE).unwrap();
        assert_eq!(p4, pb);
        println!(
            "kt c277 L2 SIMULARE: {name}: {v6b} x mm_tc_v5 -> mm_tc_v6 (pf {pfb}, tc {tcb} neschimbate), zona {} B la fel, {diff_bytes} octeti diferiti (doar adresa programului)",
            zb.len()
        );
    }
}

/// L3: the mini training (5 steps) with `mm_tc_v6` = with `mm_tc_v5` (and = with v4), bit for bit; also with only v6
/// loaded and with the attention fast variant.
#[test]
fn t277_l3_training_same_bits() {
    let (base, pb) = run(&FAST, V5, 0, AT, None, &no_edit, NONE).unwrap();
    let v5n = count(&pb, TcPick::V5);
    assert!(v5n > 0);
    let (old, _) = run(&FAST, 0, 0, AT, None, &no_edit, NONE).unwrap();
    let bits = |x: &[f32], y: &[f32]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, r)| p.to_bits() == r.to_bits())
    };
    assert!(bits(&old.weights, &base.weights));
    let no_v4 = TcPrograms { v4: 0, ..FAST };
    let mut n = 0;
    for (progs, v5, fast) in [(&FAST, V5, None), (&FAST, V5, FV4), (&no_v4, 0, None)] {
        let (r, p) = run(progs, v5, V6, AT, fast, &no_edit, NONE).unwrap();
        assert_eq!(count(&p, TcPick::V6), v5n);
        assert_eq!(count(&p, TcPick::V5) + count(&p, TcPick::V4), 0);
        for (a, b) in r.losses.iter().zip(&base.losses) {
            assert!(bits(a, b));
        }
        assert!(bits(&r.grad1, &base.grad1));
        assert!(bits(&r.weights, &base.weights));
        n = r.grad1.len() + r.weights.len() + r.losses.iter().map(Vec::len).sum::<usize>();
    }
    let first = mean(&base.losses[0]);
    let last = mean(&base.losses[STEPS as usize - 1]);
    assert!(last < first);
    println!(
        "kt c277 L3 SIMULARE: mini 5 pasi cu mm_tc_v6 ({v5n} lansari) = cu mm_tc_v5 = cu mm_tc_v4 bit cu bit ({n} valori; si doar cu v6, si cu tc16_v4); pierderea {first:.6} -> {last:.6}"
    );
}

/// L4: refusals. A v6 address not 256-aligned → refused before anything is built; a v6 QMD pointed at v4 (24 KiB
/// needed, 12 KiB in the QMD) → the stand-in refuses.
#[test]
fn t277_l4_refusals() {
    let k = Some(64 * KIB);
    let l = mini_list();
    for bad_va in [V6 + 16, V6 + 128] {
        assert!(
            ts::pack_mixed_fast_attn6(&l, prog_va, &FAST, V5, bad_va, AT, FV4, A0, k, ZONE)
                .is_err()
        );
    }
    let to_v4 = &|z: &mut Vec<u8>, q: &[u64], p: &[Option<TcPick>]| {
        use nvidia_rm570::regs::qmd_v03_00 as qq;
        for (qva, pk) in q.iter().zip(p) {
            if *pk == Some(TcPick::V6) {
                let at = (qva - ZONE) as usize;
                let f = qq::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER;
                assert_eq!(f.lo % 8, 0);
                let b = at + (f.lo / 8) as usize;
                z[b..b + 4].copy_from_slice(&(V4 as u32).to_le_bytes());
            }
        }
    };
    let e = run(&FAST, V5, V6, AT, None, &no_edit, to_v4)
        .err()
        .expect("mm_tc_v4 cu 12 KiB trebuia refuzat");
    println!(
        "kt c277 L4 SIMULARE: adresa v6 nealiniata -> refuz; QMD v6 mutat pe v4 (12 KiB) -> {e}"
    );
}

/// L1 / L2 through RAM: attn6 with `v6 = 0` = attn5 byte for byte; with v6 the picks are those without RAM.
#[test]
fn t277_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let plan = plan_mini(&inputs).resident().unwrap();
    let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let list = mini_list();
    let k = Some(64 * KIB);
    let old =
        br::pack_mixed_fast_via_ram_attn5(&list, prog_va, &FAST, V5, AT, FV4, A0, k, TC, ZONE, &v)
            .unwrap();
    let new = br::pack_mixed_fast_via_ram_attn6(
        &list, prog_va, &FAST, V5, 0, AT, FV4, A0, k, TC, ZONE, &v,
    )
    .unwrap();
    assert_eq!(old, new);
    let (_, _, pr) = br::pack_mixed_fast_via_ram_attn6(
        &list, prog_va, &FAST, V5, V6, AT, FV4, A0, k, TC, ZONE, &v,
    )
    .unwrap();
    let (_, _, pd) =
        ts::pack_mixed_fast_attn6(&list, prog_va, &FAST, V5, V6, AT, FV4, A0, k, ZONE).unwrap();
    assert_eq!(pr, pd);
    assert!(count(&pr, TcPick::V6) > 0);
    println!(
        "kt c277 prin RAM SIMULARE: fara v6 -> attn5 octet cu octet; cu v6 aceleasi alegeri ca fara RAM ({} x mm_tc_v6)",
        count(&pr, TcPick::V6)
    );
}

/// §1 (CALCUL on the real list): the flags of the 147 `mm_tc` launches of the #513 step, lot 2 — forward = `TB`
/// (+ `BIAS`), dX = no flag, dW = `TA`, a third of the work each; so the qkv-513 probe (`TB | BIAS`) is the case T275
/// counted at 640 store cycles, not 256.
#[test]
fn t277_step_flags() {
    use nvidia_ga104::mm_tc::{BIAS, TA, TB};
    let l = list_513(2);
    let mut by = std::collections::BTreeMap::<u32, (u32, u64)>::new();
    for x in &l {
        if let MixedLaunch::Tc { args, .. } = x {
            let e = by.entry(args.flags).or_default();
            e.0 += 1;
            e.1 += 2 * args.m as u64 * args.n as u64 * args.k as u64;
        }
    }
    let keys: Vec<u32> = by.keys().copied().collect();
    assert_eq!(keys, [0, TA, TB, TB | BIAS]);
    let fwd = by[&TB].1 + by[&(TB | BIAS)].1;
    let (dx, dw) = (by[&0].1, by[&TA].1);
    assert_eq!(fwd, dx);
    assert_eq!(dx, dw);
    let n: u32 = by.values().map(|v| v.0).sum();
    assert_eq!(n, 147);
    println!(
        "kt c277 pas CALCUL: 147 mm_tc in #513 lot 2: inainte TB(+BIAS) {} x, dX fara steag {} x, dW TA {} x, cate {:.1} GFLOP",
        by[&TB].0 + by[&(TB | BIAS)].0,
        by[&0].0,
        by[&TA].0,
        dx as f64 / 1e9
    );
}
