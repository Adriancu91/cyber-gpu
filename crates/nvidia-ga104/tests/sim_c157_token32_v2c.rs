//! C-157 p.2 (session 274, T288): `token32_v2c.comp` — `token32_v2` with the SM clock around every operation, the
//! measuring program of `docs/C-157_TOKEN32_CEAS.md` (predictions C1–C8 written there BEFORE the code). What must hold
//! here, without the board:
//! - listing (Mesa NAK 25.0.7): the arithmetic of `token32_v2` is untouched — the same count of every floating-point,
//!   conversion and shuffle instruction, the same barrier (4 `cctl`, 2 strong atomics), the same loads; the additions
//!   are the clock reads and the stores into the clock area; 16 blocks of 32 still fit on an SM (≤ 128 registers);
//! - SIMULARE: one token of today's list + argmax at Kronos width and G = 768 = the separate launches bit for bit (the
//!   `gpu-sim` model runs the same program models), and the driver reads the clock area the model filled with
//!   INVENTED numbers back exactly (`token_clock::read_clock`, `parts_of`, `by_part`, the `kc` rows).
//!
//! Nothing here says anything about the board: the clock numbers are invented, the model does not time anything.

mod common;

use common::token_sim::TokenSim;
use gpu_compute::layer::Shape;
use nvidia_ga104::argmax as am;
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenIo, TokenList, Way};
use nvidia_ga104::kronos_layer::Launch;
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::kronos_token as kt;
use nvidia_ga104::token_clock::{self as tc, PARTS, Part};

const V2: &str = include_str!("../../../tools/nak/out/token32_v2.sm86.txt");
const V2C: &str = include_str!("../../../tools/nak/out/token32_v2c.sm86.txt");
const SMS: u32 = 48;

fn ir(listing: &str) -> impl Iterator<Item = &str> {
    listing
        .lines()
        .take_while(|l| !l.starts_with("Encoded shader"))
        .map(|l| l.split("//").next().unwrap_or("").trim())
}

/// Instructions whose opcode (first word after an optional `rN =` / predicate) starts with `op`.
fn count_op(listing: &str, op: &str) -> usize {
    ir(listing)
        .filter(|l| {
            let body = l.rsplit(" = ").next().unwrap_or(l);
            body.split_whitespace()
                .find(|w| !w.starts_with('@'))
                .is_some_and(|w| w == op || w.starts_with(&format!("{op}.")))
        })
        .count()
}

fn count(listing: &str, what: &str) -> usize {
    ir(listing).filter(|l| l.contains(what)).count()
}

#[test]
fn t288_listing_same_arithmetic_plus_the_clock() {
    for op in [
        "ffma", "fadd", "fmul", "mufu", "fsetp", "fmnmx", "i2f", "f2i", "shfl", "fswzadd",
    ] {
        assert_eq!(count_op(V2, op), count_op(V2C, op), "{op}");
    }
    assert!(count_op(V2, "ffma") > 100 && count_op(V2, "shfl") > 0);
    assert_eq!(
        count(V2C, "cctl.global.a64"),
        4,
        "the barrier's WBAll / IVAll"
    );
    assert_eq!(count(V2C, "membar.sc..gpu"), count(V2, "membar.sc..gpu"));
    assert_eq!(count(V2C, "atom.add.u32.global.a64.strong.sys"), 2);
    // Loads: the same kinds and counts (the clock area is never read).
    for k in [
        "ld.global.a64.strong.sys",
        "ld.global.a64.constant",
        "ld.global.a64.constant.b128",
        "ld.shared",
    ] {
        assert_eq!(count(V2, k), count(V2C, k), "{k}");
    }
    // The additions: the clock reads (two per operation in the source) and the stores into the clock area.
    assert!(count(V2C, "cs2r") >= 2, "the SM clock");
    assert_eq!(count(V2, "cs2r"), 0);
    assert!(count(V2C, "st.global") >= count(V2, "st.global") + 4);
    let gprs: u32 = V2C
        .lines()
        .find_map(|l| l.strip_prefix("Num GPRs: "))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert_eq!(gprs, tc::TOKEN32_V2C_REGISTERS);
    assert!(
        gprs <= 128,
        "16 blocks of 32 threads per SM need ≤ 128 registers"
    );
    assert_eq!(
        nvidia_ga104::sha256::digest(tc::TOKEN32_V2C)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect::<String>(),
        tc::TOKEN32_V2C_SHA256
    );
    assert_ne!(tc::TOKEN32_V2C, kt::TOKEN32_V2);
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
fn t288_one_token_at_768_and_the_clock_area_read_back() {
    let m = wide();
    let groups = 768u32;
    let mut sep = TokenSim::new(&m, 0);
    let mut one = TokenSim::new(&m, 0);
    one.token32_sms = Some(SMS);
    one.token32_v2c = true;
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
        let n = l.len();
        assert_eq!(counter, (n as u32 - 1) * groups, "contorul barierei");
        assert_eq!(logits(&mut sep), logits(&mut one), "pozitia {pos}");
        assert_eq!(sep.ram_argmax().words, one.ram_argmax().words);

        // The list's parts: embed + 1 layer of 7 + final ln + logits + argmax (this model has one layer).
        let parts = tc::parts_of(&l).unwrap();
        assert_eq!(parts.len(), n);
        assert_eq!(parts[0], Part::Embed);
        assert_eq!(
            &parts[n - 3..],
            &[Part::FinalLn, Part::Logits, Part::Argmax]
        );
        assert_eq!(
            parts[1..8],
            [
                Part::Ln1,
                Part::Qkv,
                Part::Attention,
                Part::Out,
                Part::Ln2,
                Part::Fc,
                Part::Proj
            ]
        );

        // The area the SIMULARE model filled (invented numbers, see `gpu_sim::cmdproc::token32c_on_cpu`).
        let words = &one.last_clock;
        assert_eq!(words.len(), tc::clock_words(n, groups).unwrap());
        let r = tc::read_clock(words, n, groups).unwrap();
        let ops: Vec<u32> = kt::work_list32(&l)
            .unwrap()
            .chunks(kt::ENTRY)
            .map(|e| u32::from_le_bytes(e[kt::ENTRY_OP..kt::ENTRY_OP + 4].try_into().unwrap()))
            .collect();
        for (o, op) in r.ops.iter().zip(&ops) {
            let slowest = 1_000 + 37 * op + 12;
            assert_eq!(o.max_work, slowest);
            assert_eq!(
                o.window,
                slowest + 400,
                "the wrap at 0xffff_f000 is read right"
            );
            assert_eq!(o.wg0_work, 1_000 + 37 * op);
            assert_eq!(o.slowest, 4);
        }
        assert_eq!(r.polls_total, (n as u64 - 1) * groups as u64 + 768);
        let s = tc::by_part(&r, &parts).unwrap();
        assert_eq!(s.barrier, 400 * n as u64);
        let rows = s.rows(s.windows); // 1 ns per cycle: 1 000 MHz
        assert_eq!(rows[2], (tc::rows::MHZ, 1000));
        assert_eq!(rows.len(), 7 + 3 * PARTS.len());
        let qkv = rows
            .iter()
            .find(|r| r.0 == "kc qkv fereastra (ns)")
            .unwrap();
        assert_eq!(qkv.1, 1_000 + 37 * 10 + 12 + 400);
    }
}

#[test]
fn t288_parts_refuses_other_lists() {
    let m = wide();
    let s = TokenSim::new(&m, 0);
    let l = today(&s, 5, 0);
    assert!(tc::parts_of(&l[..l.len() - 2]).is_err(), "no logits");
    let mut swapped = l.clone();
    swapped.swap(2, 3);
    assert!(
        tc::parts_of(&swapped).is_err(),
        "a program in the wrong place"
    );
    assert!(tc::parts_of(&l[..l.len() - 1]).is_ok(), "without argmax");
}
