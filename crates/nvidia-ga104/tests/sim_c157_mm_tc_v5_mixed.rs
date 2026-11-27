//! C-157 p.2 after os-476 — T274, SIMULARE: `mm_tc_v5` (v4's loop, 12 KiB shared → 4 blocks per SM at 64 KiB)
//! linked in the **fast mixed step**: `train_step::pack_mixed_fast_attn5` sends every aligned `mm_tc` launch to
//! `mm_tc_v5` when it is loaded, else the old rule (v4 → pf → tc). Plan and predictions V1–V6 written before these
//! tests (`docs/C-157_OS476.md` §3).
//!
//! Honesty: the `mm_tc_v5` stand-in is `mm_tc_v4`'s (same bits proven on the board in os-476 for 5 aligned sets and
//! qkv-513, not for the whole step). Nothing here runs on the RTX 3070 Ti.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_BACK_RES_NEEDS,
    LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS, MM_TC_V5_NEEDS,
    MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS, Needs, XENT_NEEDS, attn_back_on_cpu,
    attn_back_tile_on_cpu, attn_back_tile_tc16_on_cpu, attn_back_tile_tc16_v4_on_cpu,
    attn_back_tile_tc16_v4pf_on_cpu, attn_back_tile0_on_cpu, attn_back_tile0_tc_on_cpu,
    attn_fwd_on_cpu, attn_fwd_tile_on_cpu, attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu,
    emb_back_on_cpu, emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu,
    ln_back_res_on_cpu, ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu,
    mm_tc_v4_on_cpu, mm_tc_v5_on_cpu, mm_train_gelu_on_cpu, mm_train_on_cpu, opt_step_on_cpu,
    vadd_on_cpu, xent_on_cpu,
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
/// cores, packed by `pack_mixed_fast_attn5` at the 64 KiB step.
fn run(
    progs: &TcPrograms,
    v5: u64,
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
        let (mut zone, qmds, p) = ts::pack_mixed_fast_attn5(
            &list,
            prog_va,
            progs,
            v5,
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

/// Every `mm_tc` launch's picks: (v4, pf, tc, v5).
fn mm_counts(p: &[Option<TcPick>]) -> (usize, usize, usize, usize) {
    (
        count(p, TcPick::V4),
        count(p, TcPick::Pf),
        count(p, TcPick::Tc),
        count(p, TcPick::V5),
    )
}

/// V1: without `mm_tc_v5` nothing changes: attn5 with `v5 = 0` = attn4 byte for byte (mini and #513, with and
/// without the attention fast variant).
#[test]
fn t274_v1_without_v5_same_bytes() {
    let k = Some(64 * KIB);
    for l in [mini_list(), list_513(2)] {
        for fast in [None, FV4] {
            let a = ts::pack_mixed_fast_attn4(&l, prog_va, &FAST, AT, fast, A0, k, ZONE).unwrap();
            let b =
                ts::pack_mixed_fast_attn5(&l, prog_va, &FAST, 0, AT, fast, A0, k, ZONE).unwrap();
            assert_eq!(a, b);
            assert_eq!(count(&b.2, TcPick::V5), 0);
        }
    }
    println!("kt c274 V1 SIMULARE: fara mm_tc_v5, attn5 = attn4 octet cu octet (mini si #513)");
}

/// V2 / V3: with `mm_tc_v5` every launch that went to `mm_tc_v4` goes to `mm_tc_v5`, nothing else moves; the zone
/// keeps its length and QMD addresses; a v5 QMD differs from the v4 one only in the program address and the shared
/// size (12 288 bytes instead of 24 576); the 64 KiB step is the same.
#[test]
fn t274_v2_v3_picks_and_qmds() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    let k = Some(64 * KIB);
    for (name, l, expect_v4) in [
        ("mini", mini_list(), None),
        ("#513 lot 2", list_513(2), Some(147)),
    ] {
        let (za, qa, pa) =
            ts::pack_mixed_fast_attn4(&l, prog_va, &FAST, AT, FV4, A0, k, ZONE).unwrap();
        let (zb, qb, pb) =
            ts::pack_mixed_fast_attn5(&l, prog_va, &FAST, V5, AT, FV4, A0, k, ZONE).unwrap();
        let (v4a, pfa, tca, _) = mm_counts(&pa);
        let (v4b, pfb, tcb, v5b) = mm_counts(&pb);
        if let Some(n) = expect_v4 {
            assert_eq!(v4a, n);
        }
        assert!(v4a > 0, "{name}: niciun mm_tc_v4 de mutat");
        assert_eq!((v4b, pfb, tcb, v5b), (0, pfa, tca, v4a));
        assert_eq!(qa, qb);
        assert_eq!(za.len(), zb.len());
        for (x, y) in pa.iter().zip(&pb) {
            if *x == Some(TcPick::V4) {
                assert_eq!(*y, Some(TcPick::V5));
            } else {
                assert_eq!(x, y);
            }
        }
        let mut diff_bytes = 0;
        for (qv, p) in qb.iter().zip(&pb) {
            let at = (qv - ZONE) as usize;
            if *p == Some(TcPick::V5) {
                assert_eq!(program_va(&zb, *qv), V5);
                assert_eq!(program_va(&za, *qv), V4);
                assert_eq!(
                    field(&zb, *qv, q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE),
                    12_288
                );
                assert_eq!(
                    field(&za, *qv, q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE),
                    24_576
                );
                assert_eq!(field(&zb, *qv, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V), 128);
                for f in [
                    q::NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE,
                    q::NVC6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE,
                    q::NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE,
                ] {
                    assert_eq!(field(&zb, *qv, f), field(&za, *qv, f));
                }
                // Everything else in the QMD and the whole cbuf in front of it: identical.
                let (o, n) = (&za[at..at + 0x100], &zb[at..at + 0x100]);
                let d = (0..0x100).filter(|&i| o[i] != n[i]).count();
                assert!(d <= 8, "{d}");
                diff_bytes += d;
                assert_eq!(&za[at - 0x800..at], &zb[at - 0x800..at]);
            } else {
                assert_eq!(&za[at..at + 0x100], &zb[at..at + 0x100]);
            }
        }
        println!(
            "kt c274 V2/V3 SIMULARE: {name}: {v5b} x mm_tc_v4 -> mm_tc_v5 (pf {pfb}, tc {tcb} neschimbate), zona {} B la fel, {diff_bytes} octeti diferiti (adresa + memoria comuna 12 288)",
            zb.len()
        );
    }
}

/// V4: the mini training (5 steps) with `mm_tc_v5` = with `mm_tc_v4`, bit for bit; also with only v5 loaded (no v4)
/// and with the attention fast variant.
#[test]
fn t274_v4_training_same_bits() {
    let (base, pb) = run(&FAST, 0, AT, None, &no_edit, NONE).unwrap();
    let v4n = count(&pb, TcPick::V4);
    assert!(v4n > 0);
    let bits = |x: &[f32], y: &[f32]| {
        x.len() == y.len() && x.iter().zip(y).all(|(p, r)| p.to_bits() == r.to_bits())
    };
    let no_v4 = TcPrograms { v4: 0, ..FAST };
    let mut n = 0;
    for (progs, fast) in [(&FAST, None), (&FAST, FV4), (&no_v4, None)] {
        let (r, p) = run(progs, V5, AT, fast, &no_edit, NONE).unwrap();
        assert_eq!(count(&p, TcPick::V5), v4n);
        assert_eq!(count(&p, TcPick::V4), 0);
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
        "kt c274 V4 SIMULARE: mini 5 pasi cu mm_tc_v5 ({v4n} lansari) = cu mm_tc_v4 bit cu bit ({n} valori; si fara v4 incarcat, si cu tc16_v4); pierderea {first:.6} -> {last:.6}"
    );
}

/// V5: refusals. A v5 address not 256-aligned → refused before anything is built; a planted zone where a v5 QMD is
/// pointed back at v4 (24 KiB needed, 12 KiB in the QMD) → the stand-in refuses: the shared size really matters.
#[test]
fn t274_v5_refusals() {
    let k = Some(64 * KIB);
    let l = mini_list();
    for bad_va in [V5 + 16, V5 + 128] {
        assert!(
            ts::pack_mixed_fast_attn5(&l, prog_va, &FAST, bad_va, AT, FV4, A0, k, ZONE).is_err()
        );
    }
    let to_v4 = &|z: &mut Vec<u8>, q: &[u64], p: &[Option<TcPick>]| {
        use nvidia_rm570::regs::qmd_v03_00 as qq;
        for (qva, pk) in q.iter().zip(p) {
            if *pk == Some(TcPick::V5) {
                let at = (qva - ZONE) as usize;
                let f = qq::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER;
                assert_eq!(f.lo % 8, 0);
                let b = at + (f.lo / 8) as usize;
                z[b..b + 4].copy_from_slice(&(V4 as u32).to_le_bytes());
            }
        }
    };
    let e = run(&FAST, V5, AT, None, &no_edit, to_v4)
        .err()
        .expect("mm_tc_v4 cu 12 KiB trebuia refuzat");
    println!(
        "kt c274 V5 SIMULARE: adresa v5 nealiniata -> refuz; QMD v5 mutat pe v4 (12 KiB) -> {e}"
    );
}

/// V6: through RAM: attn5 with `v5 = 0` = attn4 byte for byte; with v5 the picks are those without RAM.
#[test]
fn t274_v6_via_ram() {
    use nvidia_ga104::batch_ram as br;
    let (inputs, targets) = batch();
    let plan = plan_mini(&inputs).resident().unwrap();
    let v = br::via_ram(&plan, &inputs, &targets, VA + 0x80_0000).unwrap();
    let list = mini_list();
    let k = Some(64 * KIB);
    let old =
        br::pack_mixed_fast_via_ram_attn4(&list, prog_va, &FAST, AT, FV4, A0, k, TC, ZONE, &v)
            .unwrap();
    let new =
        br::pack_mixed_fast_via_ram_attn5(&list, prog_va, &FAST, 0, AT, FV4, A0, k, TC, ZONE, &v)
            .unwrap();
    assert_eq!(old, new);
    let (_, _, pr) =
        br::pack_mixed_fast_via_ram_attn5(&list, prog_va, &FAST, V5, AT, FV4, A0, k, TC, ZONE, &v)
            .unwrap();
    let (_, _, pd) =
        ts::pack_mixed_fast_attn5(&list, prog_va, &FAST, V5, AT, FV4, A0, k, ZONE).unwrap();
    assert_eq!(pr, pd);
    assert!(count(&pr, TcPick::V5) > 0);
    println!(
        "kt c274 V6 SIMULARE: prin RAM: fara v5 -> attn4 octet cu octet; cu v5 aceleasi alegeri ca fara RAM ({} x mm_tc_v5)",
        count(&pr, TcPick::V5)
    );
}

/// V7 (CALCUL on the board's MEASURED numbers, os-476): the probe against the band written before (P2zzb V7), and the
/// step prediction rescaled to the step measured with dS on the tensor cores (lot 6, step 2, 264.857 ms).
#[test]
fn t274_v7_os476_numbers() {
    // qkv-513, ns, from ciclu-auto-20261127-0115.txt.
    let (v4_64k, v5_64k, v4, v5): (f64, f64, f64, f64) =
        (2_628_640.0, 1_868_192.0, 4_761_472.0, 4_738_048.0);
    let r = v4_64k / v5_64k;
    assert!((1.40..1.42).contains(&r), "{r}");
    assert!(
        (1.59e6..=1.94e6).contains(&v5_64k),
        "V7: in banda scrisa inainte"
    );
    assert!(
        (v4 / v5 - 1.0).abs() < 0.01,
        "fara 64 KiB: v5 = v4 (ocuparea e castigul)"
    );
    // The step: 216.6 ms of mm_tc_v4 at lot 6 (os-468), 264.857 ms measured (lot 6 with dS, step 2).
    let (mm, step) = (216.6f64, 264.857f64);
    let at = |x: f64| step - mm * (1.0 - 1.0 / x);
    let (lo, hi) = (at(r), at(1.25));
    assert!(
        (201.0..203.0).contains(&lo) && (221.0..223.0).contains(&hi),
        "{lo} {hi}"
    );
    let sps = |ms: f64| 6.0 * 512.0 / (ms / 1e3);
    println!(
        "kt c274 V7 CALCUL: qkv-513 v4@64k / v5@64k = x{r:.3} (banda x1,35-1,65); fara 64k x{:.3}; pasul lot 6 cu v5 {lo:.1}-{hi:.1} ms = {:.0}-{:.0} simb/s (azi {:.0})",
        v4 / v5,
        sps(hi),
        sps(lo),
        sps(step)
    );
}
