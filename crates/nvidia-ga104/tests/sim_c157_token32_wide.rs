//! C-157 (session 240, T249), SIMULARE: `token32` with many workgroups (`docs/C-157_TOKEN32_G_MARE.md`, predictions
//! written before the test was run):
//! - P1: the SM shared-memory step for G on 48 SMs: ≤ 96 → 8 KiB, ≤ 240 → 16, ≤ 480 → 32, ≤ 768 → 64, 769 refused;
//! - P2: `token32_qmd_wide` at G ≤ 96 = the bytes of `token32_qmd`;
//! - P3: one token of today's list + its argmax with `token32` at G = 384 and 768 (the wide QMD) = the separate
//!   launches bit for bit, the barrier counter = (n − 1)·G, the model sees TARGET 9 (32 KiB) / 17 (64 KiB);
//! - P4: the grid-barrier probe at G = 384 and 768 holds; `groups_for_wide` refuses 0, 1, 2, 3, 17 per SM
//!   (session 245, after os-456: `groups_for_wide_sms` / `token32_groups_for_sms`; the old entries fail closed).
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models: nothing here says whether the board keeps 10 / 16
//! `token32` blocks per SM, nor whether its barrier holds with 768 workgroups.

mod common;

use common::token_sim::TokenSim;
use gpu_compute::layer::Shape;
use gpu_sim::cmdproc::{GpuMem, Launch as SimLaunch, grid_barrier_on_cpu};
use nvidia_ga104::argmax as am;
use nvidia_ga104::grid_barrier::{self as gb, Verdict};
use nvidia_ga104::gsp::work_queue::GrShape;
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenIo, TokenList, Way};
use nvidia_ga104::kronos_layer::Launch;
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::kronos_token as kt;

const GA104: GrShape = GrShape {
    gpcs: 6,
    tpc_per_gpc: 4,
    sm_per_tpc: 2,
    max_warps_per_sm: 48,
};
const SMS: u32 = 48;

#[test]
fn t249a_step_for_groups() {
    let k = 1024;
    for (g, step) in [
        (1, 8 * k),
        (96, 8 * k),
        (97, 16 * k),
        (240, 16 * k),
        (241, 32 * k),
        (384, 32 * k),
        (480, 32 * k),
        (481, 64 * k),
        (768, 64 * k),
    ] {
        assert_eq!(kt::token32_step_for(g, SMS).unwrap(), step, "G = {g}");
    }
    assert!(kt::token32_step_for(769, SMS).is_err());
    assert!(kt::token32_step_for(0, SMS).is_err());
    assert!(kt::token32_step_for(1, 0).is_err());
    assert_eq!(kt::token32_max_per_sm(), 16);
    // os-456 (session 245): the GrShape count is the design maximum, so the old entry fails closed; the SM count
    // comes from `rmapi::GrCounts`.
    assert!(kt::token32_groups_for(&GA104, 8).is_err());
    assert_eq!(kt::token32_groups_for_sms(SMS, 8).unwrap(), 384);
    assert_eq!(kt::token32_groups_for_sms(SMS, 16).unwrap(), 768);
    assert!(kt::token32_groups_for_sms(SMS, 0).is_err());
    assert!(kt::token32_groups_for_sms(SMS, 17).is_err());
    assert!(kt::token32_groups_for_sms(0, 8).is_err());
}

#[test]
fn t249b_small_g_keeps_todays_qmd() {
    for g in [1, 7, 48, 96] {
        assert_eq!(
            kt::token32_qmd_wide(0x10_0000, 0x20_0000, g, SMS).unwrap(),
            kt::token32_qmd(0x10_0000, 0x20_0000, g).unwrap(),
            "G = {g}"
        );
    }
    assert_ne!(
        kt::token32_qmd_wide(0x10_0000, 0x20_0000, 384, SMS).unwrap(),
        kt::token32_qmd(0x10_0000, 0x20_0000, 384).unwrap()
    );
    assert!(kt::token32_qmd_wide(0x10_0000, 0x20_0000, 769, SMS).is_err());
}

fn small() -> km::OwnedModel {
    km::test_model(
        Shape {
            e: 64,
            heads: 4,
            eps: 1e-5,
        },
        3,
        256,
        16,
        11,
    )
    .unwrap()
}

fn today(s: &TokenSim<'_>, tok: usize, pos: usize) -> Vec<Launch> {
    let m = s.m;
    let mut l =
        km::token_launches_fused_ln32(&m.shape, m.vocab, m.context, &s.vas, tok, pos).unwrap();
    l.push(am::launch(s.vas.logits, s.argmax_va(), m.vocab).unwrap());
    l
}

fn logits(s: &mut TokenSim<'_>) -> Vec<u32> {
    let (va, n) = (s.vas.logits, s.m.vocab);
    s.read(&[(va, n)]).unwrap().remove(0)
}

#[test]
fn t249c_one_token_at_384_and_768_groups() {
    let m = small();
    for (groups, target) in [(384u32, 9u32), (768, 17)] {
        let mut sep = TokenSim::new(&m, 0);
        let mut one = TokenSim::new(&m, 0);
        one.token32_sms = Some(SMS);
        for (pos, tok) in [5usize, 200].into_iter().enumerate() {
            let l = today(&sep, tok, pos);
            let vas = sep.vas.clone();
            kd::send_token_list(
                &mut sep,
                TokenList::FusedLn32,
                &m.weights(),
                &vas,
                tok,
                pos,
                Way::of(SeqMode::EachLaunch),
                &l[l.len() - 1..],
            )
            .unwrap();
            let counter = one.submit_token32(&l, groups, false, None).unwrap();
            assert_eq!(counter, (l.len() as u32 - 1) * groups, "contorul barierei");
            assert_eq!(
                logits(&mut sep),
                logits(&mut one),
                "G {groups} pozitia {pos}"
            );
            assert_eq!(sep.ram_argmax().words, one.ram_argmax().words);
        }
        let seen = &one.r.cp.as_ref().unwrap().sm_configs;
        assert_eq!(seen.len(), 2, "doua lansari token32 cu treapta in QMD");
        for sm in seen {
            assert_eq!([sm[0], sm[1]], [target, target], "G {groups}: MIN/TARGET");
            assert!(sm[2] >= target);
        }
    }
}

#[test]
fn t249d_barrier_probe_wide() {
    const CBUF: u64 = 0x10_0000;
    const SYNC: u64 = 0x20_0000;
    const DATA: u64 = 0x30_0000;
    const OUT: u64 = 0x40_0000;
    for per_sm in [8, 16] {
        let g = gb::groups_for_wide_sms(SMS, per_sm).unwrap();
        assert_eq!(g, SMS * per_sm);
        for rounds in [gb::ROUNDS_SHORT, gb::ROUNDS_LONG] {
            let mut mem = GpuMem::default();
            mem.write(CBUF, &gb::cbuf(SYNC, DATA, OUT, rounds, g).unwrap());
            let (s, d, o) = gb::buffer_bytes(g);
            mem.write(SYNC, &vec![0; s as usize]);
            mem.write(DATA, &vec![0; d as usize]);
            mem.write(OUT, &vec![0xff; o as usize]);
            grid_barrier_on_cpu(&mut SimLaunch {
                mem: &mut mem,
                cbuf0: CBUF,
                grid: [g, 1, 1],
                block: [gb::BLOCK, 1, 1],
            })
            .unwrap();
            let counter = mem.read_u32(SYNC).unwrap();
            let out: Vec<u32> = (0..g as u64)
                .map(|i| mem.read_u32(OUT + 4 * i).unwrap())
                .collect();
            assert_eq!(gb::judge(g, rounds, counter, &out).unwrap(), Verdict::Holds);
        }
    }
    for bad in [0, 1, 2, 3, 17] {
        assert!(gb::groups_for_wide_sms(SMS, bad).is_err(), "{bad} pe SM");
    }
    assert!(gb::groups_for_wide_sms(0, 8).is_err());
    // os-456: the old entry fails closed for every count.
    for per_sm in [8, 16] {
        assert!(gb::groups_for_wide(&GA104, per_sm).is_err());
    }
}
