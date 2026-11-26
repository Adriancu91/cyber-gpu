//! T250 (C-157, session 241, `docs/C-157_VALURI_V4.md`): where the mixed #513 step's 106 ms go on the board, and what
//! a dual dX + dW launch could still win on `mm_tc_v4`. Pure arithmetic on processor: the board times are the OS's
//! (MĂSURAT: os-443 `tests/rapoarte/os443-mm-tc-64k.txt`, os-445 `os445-mixt-rapid.txt`, os-431 `os431-pas.txt`, copied
//! unchanged); every derived number is DEDUS; nothing runs on (or simulates) the board.

use gpu_compute::train::{self, AdamW, Recipe};
use nvidia_ga104::persistent_study::{self as ps};
use nvidia_ga104::train_profile::{ParsedProfile, StepProfile, parse_profiles};
use nvidia_ga104::train_step::{
    self as ts, LOSS_SCALE_513, MixedLaunch, Plan, Prog, TcPick, TcPrograms,
};
use nvidia_ga104::wave_model as wm;

const OS443: &str = include_str!("rapoarte/os443-mm-tc-64k.txt");
const OS445: &str = include_str!("rapoarte/os445-mixt-rapid.txt");
const OS431: &str = include_str!("rapoarte/os431-pas.txt");
/// 2 blocks of `mm_tc_v4` per SM: the step `pack_mixed_fast` launched os-445 with (64 KiB).
const B64: u32 = 2;

/// The number right before `unit` after the last `key` on the first line containing `line_has`.
fn number(text: &str, line_has: &str, key: &str, unit: &str) -> u64 {
    let line = text
        .lines()
        .find(|l| l.contains(line_has))
        .unwrap_or_else(|| panic!("no line with {line_has:?}"));
    let rest = &line[line.rfind(key).expect("key") + key.len()..];
    let end = rest.find(unit).expect("unit");
    rest[..end].trim().parse().expect("a number")
}

fn profile_431() -> ParsedProfile {
    parse_profiles(OS431)
        .into_iter()
        .find(|p| p.context.contains("antrenare #513"))
        .expect("the #513 profile")
}

fn os445_board_us() -> u64 {
    number(OS445, "timpul pe pas (median", "placa ", " us")
}

/// The board time of os-431 outside the products (same programs, same 64 KiB step as os-445's mixed list).
fn rest_us() -> u64 {
    let p = profile_431();
    p.total_us - p.program("mm_tv_128x128").expect("mm_tv_128x128").us
}

/// The mixed #513 list os-445 ran (lot 2 × 512, `with_mm_tile().with_colsum_split()`, as T240).
fn list_513() -> Vec<MixedLaunch> {
    let inputs: Vec<u32> = (0..1024u32).map(|i| (i * 7_919) % 16_384).collect();
    let base = nvidia_ga104::mmu::VA_BASE + (1 << 30);
    let plan = Plan::new_tiles3(train::M513, 2, 512, &inputs, base)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split();
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    plan.step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap()
}

fn products(list: &[MixedLaunch]) -> Vec<(&'static str, u32, u32, u32)> {
    list.iter()
        .filter_map(|l| match l {
            MixedLaunch::Tc { args, tag, .. } => Some((*tag, args.m, args.n, args.k)),
            _ => None,
        })
        .collect()
}

/// The calibration reproduces the two os-443 qkv-513 rows exactly (read from the report, not retyped).
#[test]
fn t250a_calibration_is_os443() {
    let at64 = number(OS443, "mm_tc_v4@64k set qkv-513", "mm_tc_v4@64k ", " ns");
    let at32 = number(OS443, "mm_tc_v4 set qkv-513", "mm_tc_v4 ", " ns");
    assert_eq!((at64, at32), (2_623_680, 4_737_216));
    assert_eq!(wm::tc_v4_launch_ns(6144, 2304, 768, 2), Some(at64));
    assert_eq!(wm::tc_v4_launch_ns(6144, 2304, 768, 1), Some(at32));
}

/// V1: the products' share of os-445 (board − os-431's rest) against the wave model over the 147 launches.
#[test]
fn t250b_v1_products_of_the_mixed_step() {
    let list = list_513();
    let prods = products(&list);
    assert_eq!(prods.len(), 147);
    let model_ns: u64 = prods
        .iter()
        .map(|&(_, m, n, k)| wm::tc_v4_launch_ns(m, n, k, B64).unwrap())
        .sum();
    let board = os445_board_us();
    let rest = rest_us();
    let deduced = board - rest;
    let flop: u64 = prods
        .iter()
        .map(|&(_, m, n, k)| 2 * m as u64 * n as u64 * k as u64)
        .sum();
    let dev = (model_ns as f64 / 1e3 - deduced as f64) / deduced as f64;
    println!(
        "kt c241 os445 placa {board} us; restul os431 {rest} us; inmultirile DEDUS {deduced} us = {:.2} TFLOP/s; modelul v4 {} us ({:+.1} %)",
        flop as f64 / deduced as f64 / 1e6,
        model_ns / 1_000,
        dev * 100.0
    );
    let mut tags: Vec<(&str, u64)> = Vec::new();
    for &(tag, m, n, k) in &prods {
        let t = wm::tc_v4_launch_ns(m, n, k, B64).unwrap();
        match tags.iter_mut().find(|x| x.0 == tag) {
            Some(x) => x.1 += t,
            None => tags.push((tag, t)),
        }
    }
    tags.sort_by_key(|x| core::cmp::Reverse(x.1));
    for (tag, t) in &tags {
        println!("  model v4 {tag}: {} us", t / 1_000);
    }
    assert_eq!((board, rest), (106_001, 32_662));
    assert_eq!(deduced, 73_339);
    assert!(dev.abs() <= 0.10, "V1: model {model_ns} ns vs {deduced} us");
}

/// V2: the idle slots of the 147 launches (last partial waves), as thousandths of the products' modelled time.
#[test]
fn t250c_v2_idle_slots() {
    let list = list_513();
    let mut total = 0u128;
    let mut idle = 0u128;
    let mut seen = Vec::new();
    for (tag, m, n, k) in products(&list) {
        let t = wm::tc_v4_launch_ns(m, n, k, B64).unwrap() as u128;
        let p = wm::tc_v4_idle_permille(m, n, k, B64).unwrap() as u128;
        if p > 0 && !seen.contains(&tag) {
            seen.push(tag);
            println!(
                "  {tag} {m}x{n}x{k}: {} blocuri, {p} ‰ goale",
                wm::tc_v4_blocks(m, n)
            );
        }
        total += t;
        idle += t * p;
    }
    let share = (idle / total.max(1)) as u64;
    println!("kt c241 locuri goale v4: {share} ‰ din timpul inmultirilor (DEDUS)");
    assert!(share < 50, "V2: {share} ‰");
}

/// V3: the dual dX + dW floor on `mm_tc_v4`, summed over all 12 layers and the head, as a share of os-445's board.
#[test]
fn t250d_v3_dual_launch_floor() {
    let list = list_513();
    let mut pairs = ps::backward_pairs_mixed(&list, None);
    for l in 0..12u16 {
        let p = ps::backward_pairs_mixed(&list, Some(l));
        assert_eq!(p.len(), 4, "layer {l}: qkv, out, fc, proj");
        pairs.extend(p);
    }
    assert_eq!(pairs.len(), 49);
    let (mut sep, mut dual) = (0u64, 0u64);
    for p in &pairs {
        let c = ps::tc_v4_pair_cost(B64, p).unwrap();
        assert!(c.dual_floor_ns <= c.separate_ns);
        sep += c.separate_ns;
        dual += c.dual_floor_ns;
    }
    let win_us = (sep - dual) / 1_000;
    let board = os445_board_us();
    let permille = win_us * 1_000 / board;
    println!(
        "kt c241 lansarea dubla dX+dW pe v4: separat {} us, podeaua {} us, castig {win_us} us = {permille} ‰ din placa os445 (DEDUS, plafon)",
        sep / 1_000,
        dual / 1_000
    );
    assert!(permille < 20, "V3: {permille} ‰");
}

/// V4: os-431's rest as a share of the mixed board, and the attention's part of it.
#[test]
fn t250e_v4_the_rest_and_the_attention() {
    let p = profile_431();
    let board = os445_board_us();
    let rest = rest_us();
    let attn = p.programs_sum(&["attn_back_tile_3", "attn_fwd_tile3", "attn_back_tile0_3"]);
    println!(
        "kt c241 restul {rest} us = {} ‰ din placa mixta; atentia {attn} us = {} ‰ din placa, {} ‰ din rest",
        rest * 1_000 / board,
        attn * 1_000 / board,
        attn * 1_000 / rest
    );
    assert_eq!(attn, 20_581);
    assert!((280..=340).contains(&(rest * 1_000 / board)));
    assert!(
        attn * 1_000 / rest > 500,
        "the attention is most of the rest"
    );
}

/// The profile names of a mixed step (`StepProfile::from_mixed`): the 484 launches of os-445's list, each `mm_tc`
/// named after the packer's pick. Fed with the model's times (synthetic, DEDUS — not board numbers), the profile adds
/// the products up under `mm_tc_v4` exactly as the model does; wrong lengths are refused.
#[test]
fn t250f_profile_of_the_mixed_step() {
    const VA: u64 = nvidia_ga104::mmu::VA_BASE;
    const PROGS: u64 = VA + 0x40_0000;
    let fast = TcPrograms {
        tc: PROGS + 0x1_0000 * 23,
        gelu: PROGS + 0x1_0000 * 24,
        pf: PROGS + 0x1_0000 * 25,
        v4: PROGS + 0x1_0000 * 26,
    };
    let list = list_513();
    let (_, qmds, picks) = ts::pack_mixed_fast(
        &list,
        |p: Prog| PROGS + 0x1_0000 * p.index() as u64,
        &fast,
        Some(64 << 10),
        VA + 0x20_0000,
    )
    .unwrap();
    assert_eq!(qmds.len(), 484);
    let mut prof = StepProfile::from_mixed(&list, &picks).unwrap();
    assert_eq!(prof.launches(), 484);
    let ns: Vec<u64> = list
        .iter()
        .map(|l| match l {
            MixedLaunch::Tc { args, .. } => {
                wm::tc_v4_launch_ns(args.m, args.n, args.k, B64).unwrap()
            }
            _ => 1_000,
        })
        .collect();
    prof.add_step(&ns).unwrap();
    let v4 = prof
        .by_prog()
        .into_iter()
        .find(|g| g.name == "mm_tc_v4")
        .expect("mm_tc_v4 row");
    assert_eq!(v4.launches, 147);
    assert_eq!(v4.median_ns / 1_000, 73_260);
    assert!(prof.by_prog().iter().all(|g| g.name != "mm_tc"));
    for line in prof.lines(4) {
        println!("  SIMULARE (timpi din model) {line}");
    }
    // refusals: other lengths
    assert!(StepProfile::from_mixed(&list, &picks[1..]).is_err());
    assert!(StepProfile::from_mixed(&[], &[]).is_err());
    assert!(prof.add_step(&ns[1..]).is_err());
    // without picks (plain mm_tc), the products are named mm_tc
    let plain = StepProfile::from_mixed(&list, &vec![None; list.len()]).unwrap();
    let mut plain = plain;
    plain.add_step(&ns).unwrap();
    assert!(
        plain
            .by_prog()
            .iter()
            .any(|g| g.name == "mm_tc" && g.launches == 147)
    );
    assert_eq!(
        picks.iter().filter(|p| **p == Some(TcPick::V4)).count(),
        147
    );
}
