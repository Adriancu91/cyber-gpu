//! C-157 p.2 (session 277, T292): `token32_v3.comp` — `token32_v2` with every `layernorm256` folded into the
//! quarter-block product that reads its output (`docs/C-157_TOKEN32_V3_LN.md`, predictions L1–L7 written before the
//! code). What must hold here:
//! - L2 (listing, Mesa NAK 25.0.7): no local memory, at most 128 registers (16 workgroups per SM at the 64 KiB step),
//!   the barrier of `token32_v2` unchanged (4 `cctl`, 2 atomic adds), the folded start with the shuffles and the
//!   `mufu.rsq` of one more layernorm per folded operation kind;
//! - L6 / L7 (SIMULARE): one token of today's list + argmax with `token32_v3` at G = 384 and 768 = the separate launches
//!   (logits, argmax words, the layernorm outputs), the barrier counter (ops − 1) · G with 25 operations fewer per
//!   12 layers; consecutive tokens change few words of the folded list.
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models: a folded entry runs the layernorm's model then the
//! product's, so L7 shows the driver side (program, folded list, QMD). That the SASS gives the same bits is DEDUS from
//! the source and the listing. Nothing here proves anything about the board.

mod common;

use common::token_sim::TokenSim;
use gpu_compute::layer::Shape;
use nvidia_ga104::argmax as am;
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenIo, TokenList, Way};
use nvidia_ga104::kronos_layer::{Launch, Prog};
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::token_ln as tl;

const V2: &str = include_str!("../../../tools/nak/out/token32_v2.sm86.txt");
const V3: &str = include_str!("../../../tools/nak/out/token32_v3.sm86.txt");
const SMS: u32 = 48;

fn count(listing: &str, what: &str) -> usize {
    listing
        .lines()
        .take_while(|l| !l.starts_with("Encoded shader"))
        .filter(|l| l.contains(what))
        .count()
}

#[test]
fn t292_l2_listing() {
    assert!(V3.contains(&format!("Num GPRs: {}\n", tl::TOKEN32_V3_REGISTERS)));
    // the count NAK printed, not the constant: 16 blocks of 32 threads per SM need ≤ 128 registers
    let gprs: u32 = V3
        .lines()
        .find_map(|l| l.strip_prefix("Num GPRs: "))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(gprs <= 128);
    assert!(
        !V3.contains("ld.local") && !V3.contains("st.local"),
        "no local memory"
    );
    assert_eq!(count(V3, "cctl.global.a64"), 4);
    assert_eq!(count(V3, "atom.add.u32.global.a64.strong.sys"), 2);
    // one more layernorm body: its 2 × 8 warp butterflies (5 shuffles each) and one reciprocal square root
    assert_eq!(count(V3, "shfl.bfly") - count(V2, "shfl.bfly"), 80);
    assert_eq!(count(V3, "mufu.rsq") - count(V2, "mufu.rsq"), 1);
    // every store stays strong (`coherent`)
    assert_eq!(
        count(V3, "st.global.a64.strong.sys"),
        count(V3, "st.global")
    );
    assert_ne!(tl::TOKEN32_V3, nvidia_ga104::kronos_token::TOKEN32_V2);
    assert_eq!(
        nvidia_ga104::sha256::digest(tl::TOKEN32_V3)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        tl::TOKEN32_V3_SHA256
    );
    assert!(V3.contains("Instruction count: 10548\n"));
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

fn read_f32(s: &mut TokenSim<'_>, va: u64, n: usize) -> Vec<u32> {
    s.read(&[(va, n)]).unwrap().remove(0)
}

#[test]
fn t292_l7_one_token_at_kronos_width_384_and_768() {
    let m = wide();
    for groups in [384u32, 768] {
        let mut sep = TokenSim::new(&m, 0);
        let mut one = TokenSim::new(&m, 0);
        one.token32_sms = Some(SMS);
        one.token32_v3 = true;
        let mut prev: Option<Vec<u8>> = None;
        for (pos, tok) in [5usize, 200, 17].into_iter().enumerate() {
            let l = today(&sep, tok, pos);
            let lns: Vec<(u64, usize)> = l
                .iter()
                .filter(|x| x.prog == Prog::LayerNorm256)
                .map(|x| {
                    let y = u64::from_le_bytes(x.cbuf[0x40..0x48].try_into().unwrap());
                    let n = u32::from_le_bytes(x.cbuf[0x48..0x4c].try_into().unwrap());
                    (y, n as usize)
                })
                .collect();
            assert!(!lns.is_empty());
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
            // every layernorm256 of today's list is folded
            assert_eq!(one.last_ops, l.len() - lns.len());
            assert_eq!(
                counter,
                (one.last_ops as u32 - 1) * groups,
                "contorul barierei"
            );
            let (va, n) = (sep.vas.logits, m.vocab);
            assert_eq!(
                read_f32(&mut sep, va, n),
                read_f32(&mut one, va, n),
                "G {groups} pozitia {pos}"
            );
            assert_eq!(sep.ram_argmax().words, one.ram_argmax().words);
            for &(y, n) in &lns {
                assert_eq!(
                    read_f32(&mut sep, y, n),
                    read_f32(&mut one, y, n),
                    "iesirea layernorm"
                );
            }
            let (now, _) = tl::work_list_ln(&l).unwrap();
            if let Some(p) = prev {
                let ch = nvidia_ga104::kronos_token::work_list_changes(&p, &now).unwrap();
                assert!(!ch.is_empty() && ch.len() < now.len() / 4 / 20);
            }
            prev = Some(now);
        }
        assert_eq!(one.submissions, vec![1; 3], "o lansare pe simbol");
    }
}

#[test]
fn t292_v3_needs_the_sm_count() {
    let m = wide();
    let mut one = TokenSim::new(&m, 0);
    one.token32_v3 = true;
    let l = today(&one, 5, 0);
    assert!(one.submit_token32(&l, 48, false, None).is_err());
}
