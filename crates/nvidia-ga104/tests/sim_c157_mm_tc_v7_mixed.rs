//! C-157 p.2 (session 275) — T290, SIMULARE: `mm_tc_v7` (v6 with 16 bytes of padding on every shared slice row)
//! linked in the **fast mixed step**: `train_step::pack_mixed_fast_attn7` sends every aligned `mm_tc` launch to
//! `mm_tc_v7` when it is loaded (before `mm_tc_v6`), else the rule of `attn6`. Predictions M1–M5 written before these
//! tests (`docs/C-157_OS490_V7_IN_PAS.md` §3).
//!
//! Honesty: the `mm_tc_v7` stand-in is `mm_tc`'s arithmetic (T280: the same fragments as v6). On the board `mm_tc_v7`
//! gave `mm_tc`'s bits in its probe (os-486, os-490), but the **step** with v7 has not run there. Nothing here runs on
//! the RTX 3070 Ti.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS,
    LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS, MM_TC_V5_NEEDS,
    MM_TC_V6_NEEDS, MM_TC_V7_NEEDS, MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, Needs, XENT_NEEDS,
    attn_back_on_cpu, attn_back_tile_on_cpu, attn_back_tile_tc16_on_cpu,
    attn_back_tile_tc16_v4_on_cpu, attn_back_tile_tc16_v4pf_on_cpu, attn_back_tile0_on_cpu,
    attn_back_tile0_tc_on_cpu, attn_fwd_on_cpu, attn_fwd_tile_on_cpu, attn_p_on_cpu,
    attn_p_tile_on_cpu, colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu,
    gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu, ln_fwd_on_cpu, mm_tc_gelu_on_cpu,
    mm_tc_on_cpu, mm_tc_pf_on_cpu, mm_tc_v4_on_cpu, mm_tc_v5_on_cpu, mm_tc_v6_on_cpu,
    mm_tc_v7_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu,
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
use nvidia_ga104::mm_tc_v5::MM_TC_V5;
use nvidia_ga104::mm_tc_v6::MM_TC_V6;
use nvidia_ga104::mm_tc_v7::MM_TC_V7;
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
const V7: u64 = PROGS + 0x1_0000 * 35;
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
/// cores, packed by `pack_mixed_fast_attn7` at the 64 KiB step.
#[allow(clippy::too_many_arguments)]
fn run(
    progs: &TcPrograms,
    v5: u64,
    v6: u64,
    v7: u64,
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
        let (mut zone, qmds, p) = ts::pack_mixed_fast_attn7(
            &list,
            prog_va,
            progs,
            v5,
            v6,
            v7,
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

/// Every `mm_tc` launch's picks: (v4, pf, tc, v5, v6, v7).
fn mm_counts(p: &[Option<TcPick>]) -> [usize; 6] {
    [
        count(p, TcPick::V4),
        count(p, TcPick::Pf),
        count(p, TcPick::Tc),
        count(p, TcPick::V5),
        count(p, TcPick::V6),
        count(p, TcPick::V7),
    ]
}

/// M1: without `mm_tc_v7` nothing changes: attn7 with `v7 = 0` = attn6 byte for byte (mini and #513, with and
/// without v5 / v6, with and without the attention fast variant).
#[test]
fn t290_m1_without_v7_same_bytes() {
    let k = Some(64 * KIB);
    for l in [mini_list(), list_513(2)] {
        for (v5, v6) in [(0, 0), (V5, 0), (V5, V6), (0, V6)] {
            for fast in [None, FV4] {
                let a =
                    ts::pack_mixed_fast_attn6(&l, prog_va, &FAST, v5, v6, AT, fast, A0, k, ZONE)
                        .unwrap();
                let b =
                    ts::pack_mixed_fast_attn7(&l, prog_va, &FAST, v5, v6, 0, AT, fast, A0, k, ZONE)
                        .unwrap();
                assert_eq!(a, b);
                assert_eq!(count(&b.2, TcPick::V7), 0);
            }
        }
    }
    assert_eq!(TcPick::V7.name(), "mm_tc_v7");
    println!(
        "kt c290 M1 SIMULARE: fara mm_tc_v7, attn7 = attn6 octet cu octet (mini si #513, cu si fara v5 / v6)"
    );
}

/// M2: with `mm_tc_v7` every launch that went to v6 goes to v7, nothing else moves; the zone keeps its length and QMD
/// addresses; a v7 QMD differs from the v6 one only in the program address and the shared size.
#[test]
fn t290_m2_picks_and_qmds() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let k = Some(64 * KIB);
    for (name, l, expect) in [
        ("mini", mini_list(), None),
        ("#513 lot 2", list_513(2), Some(147)),
        ("#513 lot 6", list_513(6), Some(147)),
    ] {
        let (za, qa, pa) =
            ts::pack_mixed_fast_attn6(&l, prog_va, &FAST, V5, V6, AT, FV4, A0, k, ZONE).unwrap();
        let (zb, qb, pb) =
            ts::pack_mixed_fast_attn7(&l, prog_va, &FAST, V5, V6, V7, AT, FV4, A0, k, ZONE)
                .unwrap();
        let [_, pfa, tca, _, v6a, _] = mm_counts(&pa);
        let got = mm_counts(&pb);
        if let Some(n) = expect {
            assert_eq!(v6a, n, "{name}");
        }
        assert!(v6a > 0, "{name}: niciun mm_tc_v6 de mutat");
        assert_eq!(got, [0, pfa, tca, 0, 0, v6a], "{name}");
        assert_eq!(qa, qb);
        assert_eq!(za.len(), zb.len());
        let mut max_diff = 0;
        for ((qv, x), y) in qb.iter().zip(&pa).zip(&pb) {
            let at = (qv - ZONE) as usize;
            if *x == Some(TcPick::V6) {
                assert_eq!(*y, Some(TcPick::V7));
                assert_eq!(program_va(&zb, *qv), V7);
                assert_eq!(program_va(&za, *qv), V6);
                assert_eq!(
                    field(&za, *qv, q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE),
                    12_288
                );
                assert_eq!(
                    field(&zb, *qv, q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE),
                    14_336
                );
                assert_eq!(field(&zb, *qv, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V), 128);
                let (o, n) = (&za[at..at + 0x100], &zb[at..at + 0x100]);
                let d = (0..0x100).filter(|&i| o[i] != n[i]).count();
                assert!(d <= 6, "{d}");
                max_diff = max_diff.max(d);
                assert_eq!(&za[at - 0x800..at], &zb[at - 0x800..at]);
            } else {
                assert_eq!(x, y);
                assert_eq!(&za[at..at + 0x100], &zb[at..at + 0x100]);
            }
        }
        // Without v6 / v5 loaded the same launches go to v7.
        let (_, _, p0) =
            ts::pack_mixed_fast_attn7(&l, prog_va, &FAST, 0, 0, V7, AT, FV4, A0, k, ZONE).unwrap();
        assert_eq!(p0, pb);
        println!(
            "kt c290 M2 SIMULARE: {name}: {v6a} x mm_tc_v6 -> mm_tc_v7 (pf {pfa}, tc {tca} neschimbate), zona {} B la fel, cel mult {max_diff} octeti diferiti pe QMD (adresa + memoria comuna 12 288 -> 14 336)",
            zb.len()
        );
    }
}

/// M3: the mini training (5 steps) with `mm_tc_v7` = with `mm_tc_v6` (= v5 = v4), bit for bit; also with only v7
/// loaded and with the attention fast variant.
#[test]
fn t290_m3_training_same_bits() {
    let (base, pb) = run(&FAST, V5, V6, 0, AT, None, &no_edit, NONE).unwrap();
    let v6n = count(&pb, TcPick::V6);
    assert!(v6n > 0);
    let (old, _) = run(&FAST, 0, 0, 0, AT, None, &no_edit, NONE).unwrap();
    let bits = |x: &[f32], y: &[f32]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, r)| p.to_bits() == r.to_bits())
    };
    assert!(bits(&old.weights, &base.weights));
    let no_v4 = TcPrograms { v4: 0, ..FAST };
    let mut n = 0;
    for (progs, v5, v6, fast) in [
        (&FAST, V5, V6, None),
        (&FAST, V5, V6, FV4),
        (&no_v4, 0, 0, None),
    ] {
        let (r, p) = run(progs, v5, v6, V7, AT, fast, &no_edit, NONE).unwrap();
        assert_eq!(count(&p, TcPick::V7), v6n);
        assert_eq!(
            count(&p, TcPick::V6) + count(&p, TcPick::V5) + count(&p, TcPick::V4),
            0
        );
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
        "kt c290 M3 SIMULARE: mini 5 pasi cu mm_tc_v7 ({v6n} lansari) = cu mm_tc_v6 = cu mm_tc_v4 bit cu bit ({n} valori; si doar cu v7, si cu tc16_v4); pierderea {first:.6} -> {last:.6}"
    );
}

/// M4: refusals. A v7 address not 256-aligned → refused before anything is built; a v7 QMD whose shared size is
/// lowered to v6's 12 KiB → the stand-in refuses (v7 needs 14 336 B).
#[test]
fn t290_m4_refusals() {
    use nvidia_rm570::regs::qmd_v03_00 as qq;
    let k = Some(64 * KIB);
    let l = mini_list();
    for bad_va in [V7 + 16, V7 + 128] {
        assert!(
            ts::pack_mixed_fast_attn7(&l, prog_va, &FAST, V5, V6, bad_va, AT, FV4, A0, k, ZONE)
                .is_err()
        );
    }
    // Build the zone once with v6 to learn the bytes of a 12 KiB shared-size field, then copy them into the v7 QMDs.
    let (z6, q6, p6) =
        ts::pack_mixed_fast_attn6(&l, prog_va, &FAST, V5, V6, AT, FV4, A0, k, ZONE).unwrap();
    let i6 = p6.iter().position(|p| *p == Some(TcPick::V6)).unwrap();
    let at6 = (q6[i6] - ZONE) as usize;
    let qmd6: Vec<u8> = z6[at6..at6 + 0x100].to_vec();
    let to_12k = move |z: &mut Vec<u8>, q: &[u64], p: &[Option<TcPick>]| {
        let f = qq::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE;
        let (lo, hi) = ((f.lo / 8) as usize, (f.hi / 8) as usize);
        for (qva, pk) in q.iter().zip(p) {
            if *pk == Some(TcPick::V7) {
                let at = (qva - ZONE) as usize;
                z[at + lo..=at + hi].copy_from_slice(&qmd6[lo..=hi]);
            }
        }
    };
    let e = run(&FAST, V5, V6, V7, AT, None, &no_edit, &to_12k)
        .err()
        .expect("mm_tc_v7 cu 12 KiB trebuia refuzat");
    println!(
        "kt c290 M4 SIMULARE: adresa v7 nealiniata -> refuz; QMD v7 cu memoria comuna a lui v6 (12 KiB) -> {e}"
    );
}

/// M5 (CALCUL): `mm_tc_v7` keeps 4 blocks per SM at the 64 KiB step, as v6.
#[test]
fn t290_m5_blocks_per_sm() {
    use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
    let k7 = Kernel {
        name: "mm_tc_v7",
        threads: 128,
        registers: nvidia_ga104::mm_tc_v7::REGISTERS,
        shared_bytes: nvidia_ga104::mm_tc_v7::SHARED.bytes,
    };
    let k6 = Kernel {
        name: "mm_tc_v6",
        threads: 128,
        registers: nvidia_ga104::mm_tc_v6::REGISTERS,
        shared_bytes: nvidia_ga104::mm_tc_v6::SHARED.bytes,
    };
    for r in [Reserve::None, Reserve::OneKib] {
        let b7 = blocks_per_sm(&SM86, &k7, 64 * KIB, r).unwrap().blocks;
        let b6 = blocks_per_sm(&SM86, &k6, 64 * KIB, r).unwrap().blocks;
        assert_eq!(b7, b6);
        assert_eq!(b7, 4);
    }
    let b7 = 4;
    let b6 = 4;
    println!("kt c290 M5 CALCUL: la 64 KiB mm_tc_v7 {b7} blocuri pe SM (v6 {b6})");
}

/// M1 / M2 through RAM: attn7 with `v7 = 0` = attn6 byte for byte; with v7 the picks are those without RAM.
#[test]
fn t290_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let plan = plan_mini(&inputs).resident().unwrap();
    let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let list = mini_list();
    let k = Some(64 * KIB);
    let old = br::pack_mixed_fast_via_ram_attn6(
        &list, prog_va, &FAST, V5, V6, AT, FV4, A0, k, TC, ZONE, &v,
    )
    .unwrap();
    let new = br::pack_mixed_fast_via_ram_attn7(
        &list, prog_va, &FAST, V5, V6, 0, AT, FV4, A0, k, TC, ZONE, &v,
    )
    .unwrap();
    assert_eq!(old, new);
    let (_, _, pr) = br::pack_mixed_fast_via_ram_attn7(
        &list, prog_va, &FAST, V5, V6, V7, AT, FV4, A0, k, TC, ZONE, &v,
    )
    .unwrap();
    let (_, _, pd) =
        ts::pack_mixed_fast_attn7(&list, prog_va, &FAST, V5, V6, V7, AT, FV4, A0, k, ZONE).unwrap();
    assert_eq!(pr, pd);
    assert!(count(&pr, TcPick::V7) > 0);
    println!(
        "kt c290 prin RAM SIMULARE: fara v7 -> attn6 octet cu octet; cu v7 aceleasi alegeri ca fara RAM ({} x mm_tc_v7)",
        count(&pr, TcPick::V7)
    );
}
