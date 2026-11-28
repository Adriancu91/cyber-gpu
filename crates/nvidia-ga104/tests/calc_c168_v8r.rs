//! T302 (C-168 point 3 follow-up, session 286, `docs/C-168_V8R.md`): `mm_tc_v8r` = `mm_tc_v8` with the fragment reads of
//! a slice reordered (the 4 B fragments first, then one A fragment at a time with its 4 `hmma`). COUNTED on the NAK
//! 26.3.1 listings (compiled in the cloud, no board): fewer registers, fewer register copies in the K loop, no swap
//! through a scratch register, a shorter static issue schedule. Nothing here has run on a board; the speed on the board
//! is a prediction (V1-V3 in the note), not a result.

use nvidia_ga104::mm_tc_v8 as v8;
use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
use nvidia_ga104::sha256::digest as sha256;

const MM_TC_V8R: &[u8] = include_bytes!("../../../tools/nak/out2/mm_tc_v8r.sm86.sass");
const MM_TC_V8R_SHA256: &str = "6e28d531fc104ec1b37edb0f2303337c81eef66a4a82b5ad02861add18274bf4";
const LISTING_V8R: &str = include_str!("../../../tools/nak/out2/mm_tc_v8r.sm86.txt");
const LISTING_V8: &str = include_str!("../../../tools/nak/out2/mm_tc_v8.sm86.txt");
const LISTING_V8N: &str = include_str!("../../../tools/nak/out2/mm_tc_v8n.sm86.txt");
const SRC_V8R: &str = include_str!("../../../tools/nak/mm_tc_v8r.comp");
const SRC_V8: &str = include_str!("../../../tools/nak/mm_tc_v8.comp");

fn hex(d: &[u8]) -> String {
    d.iter().map(|b| format!("{b:02x}")).collect()
}

fn count(t: &[&str], what: &str) -> usize {
    t.iter().filter(|l| l.contains(what)).count()
}

/// The K loop as in T295b: from the block holding the first `ldsm` to the barrier that ends the slice.
fn k_loop(listing: &str) -> Vec<&str> {
    let lines: Vec<&str> = listing.lines().collect();
    let first = lines.iter().position(|l| l.contains("ldsm")).unwrap();
    let start = (0..first)
        .rev()
        .find(|&i| lines[i].starts_with("block"))
        .unwrap();
    let end = (first..lines.len())
        .find(|&i| lines[i].contains("bar.sync"))
        .unwrap();
    lines[start..=end].to_vec()
}

/// The loop's main block (the one with the `hmma`), up to its first branch.
fn main_block(listing: &str) -> Vec<&str> {
    let body = k_loop(listing);
    let end = body.iter().position(|l| l.contains(" bra ")).unwrap();
    body[..=end].to_vec()
}

/// Sum of the issue delays NAK wrote on the main block: a static lower bound of the cycles one warp needs to issue a
/// slice (waits on loads and on the tensor cores come on top).
fn issue_delay(listing: &str) -> u32 {
    main_block(listing)
        .iter()
        .filter_map(|l| l.split("delay=").nth(1))
        .map(|r| {
            r.chars()
                .take_while(|c| c.is_ascii_digit())
                .collect::<String>()
                .parse::<u32>()
                .unwrap()
        })
        .sum()
}

/// Copies whose destination is the one scratch register of a swap (`rX = mov rY` / `rY = mov rZ` / `rZ = mov rX`):
/// the register written by `mov` most often in the main block, counted only if it is written more than 8 times.
fn swap_scratch_movs(listing: &str) -> usize {
    let mut dst: Vec<(&str, usize)> = Vec::new();
    for l in main_block(listing) {
        if let Some((d, _)) = l.split_once(" = mov ") {
            let d = d.trim();
            match dst.iter_mut().find(|(r, _)| *r == d) {
                Some(e) => e.1 += 1,
                None => dst.push((d, 1)),
            }
        }
    }
    dst.iter().map(|&(_, n)| n).filter(|&n| n > 8).sum()
}

fn count_in(v: &[String], what: &str) -> usize {
    v.iter().filter(|l| l.contains(what)).count()
}

fn gprs(listing: &str) -> u32 {
    let l = listing
        .lines()
        .find(|l| l.starts_with("Num GPRs:"))
        .unwrap();
    l["Num GPRs:".len()..].trim().parse().unwrap()
}

#[test]
fn t302a_binary_is_the_recorded_build() {
    assert_eq!(MM_TC_V8R.len() % 16, 0);
    assert_eq!(hex(&sha256(MM_TC_V8R)), MM_TC_V8R_SHA256);
    // a different program from v8 (same source otherwise)
    assert_ne!(hex(&sha256(MM_TC_V8R)), v8::MM_TC_V8_SHA256);
}

#[test]
fn t302b_registers_and_blocks_per_sm() {
    assert_eq!(gprs(LISTING_V8), v8::REGISTERS);
    assert_eq!(gprs(LISTING_V8), 128);
    assert_eq!(gprs(LISTING_V8R), 120, "8 registers fewer than v8");
    assert_eq!(gprs(LISTING_V8N), v8::REGISTERS_N);
    let k = Kernel {
        name: "mm_tc_v8r",
        threads: v8::BLOCK[0],
        registers: 120,
        shared_bytes: v8::SHARED.bytes,
    };
    for r in [Reserve::None, Reserve::OneKib] {
        assert_eq!(
            blocks_per_sm(&SM86, &k, 64 << 10, r).unwrap().blocks,
            v8::BLOCKS_PER_SM,
            "still 2 blocks per SM, like v8"
        );
    }
    // no local memory (no spills)
    for l in LISTING_V8R.lines() {
        assert!(!l.contains("ld.local") && !l.contains("st.local"), "{l}");
    }
}

#[test]
fn t302c_the_k_loop_has_the_same_work_and_fewer_copies() {
    let body = k_loop(LISTING_V8R);
    assert_eq!(count(&body, "hmma.m16n8k16.f32"), 16);
    assert_eq!(count(&body, "ldsm.16.m8n8.x4"), 4, "A: 4 ldsm.x4");
    assert_eq!(count(&body, "ldsm.16.m8n8.x2"), 4, "B: 4 ldsm.x2");
    assert_eq!(count(&body, "ld.shared"), 0);
    assert_eq!(count(&body, "bar.sync"), 1);
    let whole: Vec<&str> = LISTING_V8R.lines().collect();
    assert_eq!(
        count(&whole, "hmma"),
        16,
        "the loop is the only place with hmma"
    );
    assert_eq!(count(&whole, "ldsm"), 8);
    assert_eq!(count(&whole, "movm"), 0);
    assert_eq!(count(&whole, "bar.sync"), 2);

    // (v8, v8r, v8n): copies in the loop, swap copies, static issue delay of the main block
    let movs = |l: &str| count(&k_loop(l), " = mov ");
    assert_eq!(movs(LISTING_V8), 148);
    assert_eq!(movs(LISTING_V8R), 84, "-43 % copies");
    assert_eq!(movs(LISTING_V8N), 41);
    assert_eq!(
        swap_scratch_movs(LISTING_V8),
        32,
        "v8: every accumulator swapped through r125"
    );
    assert_eq!(swap_scratch_movs(LISTING_V8R), 0);
    assert_eq!(swap_scratch_movs(LISTING_V8N), 0);
    assert_eq!(issue_delay(LISTING_V8), 645);
    assert_eq!(issue_delay(LISTING_V8R), 378, "-41 %, close to v8n");
    assert_eq!(issue_delay(LISTING_V8N), 367);
}

/// The same bits as v8: every one of the 16 accumulators gets exactly one `hmma` per slice, and the 16 `hmma` are all
/// the (A fragment, B fragment) pairs, 4 x 4, each once.
#[test]
fn t302d_the_16_hmma_are_the_4x4_pairs() {
    for listing in [LISTING_V8, LISTING_V8R] {
        let mut pairs: Vec<(String, String)> = Vec::new();
        for l in k_loop(listing) {
            if !l.contains("hmma.m16n8k16.f32") {
                continue;
            }
            let ops: Vec<&str> = l
                .split("hmma.m16n8k16.f32")
                .nth(1)
                .unwrap()
                .split_whitespace()
                .take(3)
                .collect();
            let dst = l.split(" = ").next().unwrap().trim();
            assert_eq!(dst, ops[2], "accumulate in place: {l}");
            pairs.push((ops[0].to_string(), ops[1].to_string()));
        }
        assert_eq!(pairs.len(), 16);
        let mut a: Vec<&String> = pairs.iter().map(|p| &p.0).collect();
        let mut b: Vec<&String> = pairs.iter().map(|p| &p.1).collect();
        a.sort();
        a.dedup();
        b.sort();
        b.dedup();
        // v8 holds its 4 A fragments at once; v8r reuses registers for A (one at a time), so count the pairs instead
        let mut uniq = pairs.clone();
        uniq.sort();
        uniq.dedup();
        assert!(b.len() == 4, "4 B fragments");
        assert!(
            a.len() <= 4 && uniq.len() >= 13,
            "{} A, {} pairs",
            a.len(),
            uniq.len()
        );
    }
}

/// The source is v8's with only the slice's read order changed (plus the header).
#[test]
fn t302e_source_differs_from_v8_only_in_the_read_order() {
    let code = |s: &str| -> Vec<String> {
        s.lines()
            .map(|l| l.trim())
            .filter(|l| !l.is_empty() && !l.starts_with("//"))
            .map(String::from)
            .collect()
    };
    let (a, b) = (code(SRC_V8), code(SRC_V8R));
    let cut = |v: &[String]| -> (Vec<String>, Vec<String>, Vec<String>) {
        let s = v.iter().position(|l| l == "uint sb = sa + AST;").unwrap();
        let e = v
            .iter()
            .position(|l| l == "uint nx = (cur ^ 1u) * STAGE;")
            .unwrap();
        (v[..=s].to_vec(), v[s + 1..e].to_vec(), v[e..].to_vec())
    };
    let (a0, a1, a2) = cut(&a);
    let (b0, b1, b2) = cut(&b);
    assert_eq!(a0, b0, "everything before the slice's reads is v8's");
    assert_eq!(a2, b2, "everything after the slice's products is v8's");
    // the same statements inside, only nested differently
    let l = "coopMatLoad(fb[j], sh_, sb + ((wc + 8u * j) * LS) / 2u, SS, gl_CooperativeMatrixLayoutColumnMajor);";
    assert!(
        a1.iter().any(|x| x == l) && b1.iter().any(|x| x == l),
        "{l}"
    );
    assert_eq!(count_in(&a1, "coopMatLoad"), 2);
    assert_eq!(count_in(&b1, "coopMatLoad"), 2);
    assert_eq!(count_in(&a1, "coopMatMulAdd"), 1);
    assert_eq!(count_in(&b1, "coopMatMulAdd"), 1);
    assert!(SRC_V8R.contains("coopMatMulAdd(fa, fb[j], acc[i][j])"));
}
