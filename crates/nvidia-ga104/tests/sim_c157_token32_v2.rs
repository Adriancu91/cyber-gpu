//! C-157 after os-471 (session 260, T272): `token32_v2.comp` — `token32` with the reads of earlier results through L1
//! (`docs/C-157_OS471.md` §3, predictions V1–V5 written before the code). What must hold here:
//! - V2 (listing, Mesa NAK 25.0.7): the NAK IR of `token32_v2` is that of `token32` instruction for instruction apart
//!   from the kind of some global loads (`.strong.sys` → `.constant`); the x loads of the products (128 bit) are all
//!   `.constant`; the four `cctl` around the grid barrier stay; 7 438 instructions, 92 registers;
//! - V1 (SIMULARE): one token of today's list + argmax with `token32_v2` at G = 384 and 768 (wide QMD) = the separate
//!   launches, the same logits bit for bit and the same argmax words, the barrier counter (n − 1) · G.
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models: it runs the same model for both programs, so V1 here only
//! shows the driver side (program, list, QMD) is right. That the SASS gives the same bits is DEDUS from V2 (same
//! arithmetic in the same order); that L1 holds nothing stale after the barrier is DEDUS from NAK's `from_nir.rs`
//! (barrier with acquire semantics → `cctl` IVAll after `membar`). Nothing here proves anything about the board.

mod common;

use common::token_sim::TokenSim;
use gpu_compute::layer::Shape;
use nvidia_ga104::argmax as am;
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenIo, TokenList, Way};
use nvidia_ga104::kronos_layer::Launch;
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::kronos_token as kt;

const OLD: &str = include_str!("../../../tools/nak/out/token32.sm86.txt");
const NEW: &str = include_str!("../../../tools/nak/out/token32_v2.sm86.txt");
const SMS: u32 = 48;

/// The IR lines (up to the encoded words), comments dropped, the load kind replaced by `X`.
fn ir(listing: &str) -> Vec<String> {
    listing
        .lines()
        .take_while(|l| !l.starts_with("Encoded shader"))
        .map(|l| {
            let l = l.split("//").next().unwrap_or("").trim_end();
            l.replace("ld.global.a64.strong.sys", "ld.global.a64.X")
                .replace("ld.global.a64.constant", "ld.global.a64.X")
        })
        .collect()
}

fn count(listing: &str, what: &str) -> usize {
    listing
        .lines()
        .take_while(|l| !l.starts_with("Encoded shader"))
        .filter(|l| l.contains(what))
        .count()
}

#[test]
fn t272_v2_same_ir_apart_from_the_loads() {
    let (a, b) = (ir(OLD), ir(NEW));
    assert!(a.len() > 7438);
    assert_eq!(a, b, "only the kind of the global loads may differ");
    for l in [OLD, NEW] {
        assert!(l.contains("Instruction count: 7438"));
        assert!(l.contains("Num GPRs: 92"));
        assert_eq!(
            count(l, "cctl.global.a64"),
            4,
            "WBAll / IVAll around the two memory barriers of grid_sync"
        );
        assert_eq!(count(l, "membar.sc..gpu"), 59);
        assert_eq!(
            count(l, "atom.add.u32.global.a64.strong.sys"),
            2,
            "the barrier counter stays strong"
        );
    }
    // token32: 274 strong loads (262 × 32 bit + 12 × 128 bit), 216 constant (weights + list).
    assert_eq!(count(OLD, "ld.global.a64.strong.sys"), 274);
    assert_eq!(count(OLD, "ld.global.a64.constant"), 216);
    // v2: 173 of them through L1; the 4 x loads of the two quarter-block products among them (128 bit).
    assert_eq!(count(NEW, "ld.global.a64.strong.sys"), 101);
    assert_eq!(count(NEW, "ld.global.a64.constant"), 389);
    assert_eq!(count(NEW, "ld.global.a64.constant.b128"), 4);
    // Writes untouched: every store stays strong (`coherent`).
    assert_eq!(count(OLD, "st.global"), count(NEW, "st.global"));
    assert_eq!(
        count(NEW, "st.global.a64.strong.sys"),
        count(NEW, "st.global")
    );
    assert_eq!(kt::TOKEN32_V2.len(), kt::TOKEN32.len());
    assert_ne!(kt::TOKEN32_V2, kt::TOKEN32);
    assert_eq!(kt::TOKEN32_V2_REGISTERS, kt::TOKEN32_REGISTERS);
}

fn wide() -> km::OwnedModel {
    km::test_model(
        Shape {
            e: 768,
            heads: 12,
            eps: 1e-5,
        },
        1,
        512,
        8,
        9,
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
fn t272_v1_one_token_at_kronos_width_384_and_768() {
    let m = wide();
    for groups in [384u32, 768] {
        let mut sep = TokenSim::new(&m, 0);
        let mut one = TokenSim::new(&m, 0);
        one.token32_sms = Some(SMS);
        one.token32_v2 = true;
        let mut prev: Option<Vec<u8>> = None;
        for (pos, tok) in [5usize, 200, 17].into_iter().enumerate() {
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
            // The driver's side of cause 2: only a few words of the list change between tokens.
            let now = kt::work_list32(&l).unwrap();
            if let Some(p) = prev {
                let ch = kt::work_list_changes(&p, &now).unwrap();
                assert!(!ch.is_empty() && ch.len() < now.len() / 4 / 20);
            }
            prev = Some(now);
        }
        assert_eq!(one.submissions, vec![1; 3], "o lansare pe simbol");
    }
}
