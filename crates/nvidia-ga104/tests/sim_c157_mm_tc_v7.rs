//! T280 (C-157 p.2, `docs/C-157_CITIRI_CONFLICTE.md` §3 / §5, session 266), SIMULARE: `mm_tc_v6` with 16 bytes of
//! padding on every shared slice row (`tools/nak/mm_tc_v7.comp` → `tools/nak/out2/mm_tc_v7.sm86.sass`, second NAK
//! tool, Mesa 26.3.1). Checked: the binary is the recorded build (W1: the same loop instructions as v6 — 32 `lds`,
//! 8 `ldsm.x4`, the same `hmma`, 64-bit stores and paired conversions — with the padded offsets; W2: ≤ 128 registers,
//! nothing spilled, 14 336 B); the GLSL is `mm_tc_v6`'s with only the row lengths changed; the per-lane read
//! addresses of the listing, under the public CUDA 8.6 bank model, are conflict-free on every flag (128 wavefronts
//! per block and slice, as T279 counted); the padded stores and reads, written out in Rust lane by lane for every
//! flag combination, give every 16×16 fragment exactly the elements `mm_tc_v6` gives it (a planted mistake — one
//! layout read with the unpadded row — is caught), every half of a slice is written once and no padding half is read;
//! through our pushbuffer + QMD in the command-processor model `mm_tc_v7` gives the same bits as `mm_tc` and refuses
//! what `mm_tc_v6` refuses (W3). Nothing here runs on the RTX 3070 Ti.

use gpu_compute::mm_tc::Product;
use gpu_sim::cmdproc::{
    CommandProcessor, MM_TC_NEEDS, MM_TC_V6_NEEDS, MM_TC_V7_NEEDS, ProgramFn, mm_tc_on_cpu,
    mm_tc_v6_on_cpu, mm_tc_v7_on_cpu,
};
use nvidia_ga104::mm_tc::{self as tc, MM_TC};
use nvidia_ga104::mm_tc_v6::{self as v6, MM_TC_V6};
use nvidia_ga104::mm_tc_v7::{self as v7, MM_TC_V7};
use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const LISTING_V7: &str = include_str!("../../../tools/nak/out2/mm_tc_v7.sm86.txt");
const LISTING_V6: &str = include_str!("../../../tools/nak/out2/mm_tc_v6.sm86.txt");
const SRC_V7: &str = include_str!("../../../tools/nak/mm_tc_v7.comp");
const SRC_V6: &str = include_str!("../../../tools/nak/mm_tc_v6.comp");

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const A: u64 = 0x100_0000;
const B: u64 = 0x200_0000;
const BIAS: u64 = 0x2f0_0000;
const C: u64 = 0x300_0000;
const QMD: u64 = 0x70_0000;
const PUSH: u64 = 0x71_0000;
const SENTINEL: f32 = -12345.0;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

struct Prog {
    code: &'static [u8],
    run: ProgramFn,
    needs: gpu_sim::cmdproc::Needs,
    registers: u32,
    shared: SharedMemory,
}

const MM_TC_P: Prog = Prog {
    code: MM_TC,
    run: mm_tc_on_cpu,
    needs: MM_TC_NEEDS,
    registers: tc::REGISTERS,
    shared: tc::SHARED,
};
const MM_TC_V6_P: Prog = Prog {
    code: MM_TC_V6,
    run: mm_tc_v6_on_cpu,
    needs: MM_TC_V6_NEEDS,
    registers: v6::REGISTERS,
    shared: v6::SHARED,
};
const MM_TC_V7_P: Prog = Prog {
    code: MM_TC_V7,
    run: mm_tc_v7_on_cpu,
    needs: MM_TC_V7_NEEDS,
    registers: v7::REGISTERS,
    shared: v7::SHARED,
};

fn args(p: &Product<'_>) -> tc::Args {
    tc::Args {
        a: A,
        b: B,
        c: C,
        bias: if p.flags & tc::BIAS != 0 { BIAS } else { 0 },
        m: p.m as u32,
        n: p.n as u32,
        k: p.k as u32,
        flags: p.flags,
    }
}

/// One launch of `prog` on `p`, A and B placed `da` / `db` bytes after their usual addresses; `Ok(C)` or the refusal.
fn run_at(
    prog: &Prog,
    p: &Product<'_>,
    grid: [u32; 3],
    block: [u32; 3],
    da: u64,
    db: u64,
) -> Result<Vec<u32>, gpu_core::error::GpuError> {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(prog.code, prog.run, prog.needs);
    cp.mem.write(PROG, prog.code);
    let mut ar = args(p);
    ar.a += da;
    ar.b += db;
    cp.mem.write(CBUF, &ar.cbuf().unwrap());
    cp.mem.write_f32s(A + da, &p.a[..p.m * p.k]);
    cp.mem.write_f32s(B + db, &p.b[..p.k * p.n]);
    if let Some(bias) = p.bias {
        cp.mem.write_f32s(BIAS, bias);
    }
    cp.mem.write_f32s(C, &vec![SENTINEL; p.m * p.n + 16]);
    cp.mem.write_u32(SEM, 0);
    let q = ComputeQmd {
        program_va: PROG,
        registers: prog.registers,
        cbuf0_va: CBUF,
        cbuf0_size: tc::CBUF_BYTES as u32,
        grid,
        block,
        release: Some((SEM, 1)),
    };
    cp.mem
        .write(QMD, &q.to_bytes_with_shared(&prog.shared).unwrap());
    let mut w = [0u32; 16];
    let mut pb = PushBuf::new(&mut w);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    pb.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    pb.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = pb.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])?;
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let got = cp.mem.read_f32s(C, p.m * p.n + 16).unwrap();
    assert!(
        got[p.m * p.n..].iter().all(|&x| x == SENTINEL),
        "wrote after C"
    );
    Ok(got[..p.m * p.n].iter().map(|x| x.to_bits()).collect())
}

fn aligned_shapes(f: u32) -> Vec<(usize, usize, usize)> {
    [
        (64usize, 32usize, 64usize),
        (68, 36, 60),
        (4, 4, 4),
        (132, 96, 72),
        (8, 200, 132),
        (100, 64, 4),
        (61, 44, 57),
        (63, 33, 65),
    ]
    .into_iter()
    .filter(|&(m, k, n)| {
        v7::fits(&tc::Args {
            a: A,
            b: B,
            c: C,
            bias: 0,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            flags: f,
        })
    })
    .collect()
}

/// Row lengths in halves: (A row-major, A column-major, B row-major, B column-major).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Rows {
    a_row: u32,
    a_col: u32,
    b_row: u32,
    b_col: u32,
}
const DENSE: Rows = Rows {
    a_row: 32,
    a_col: 64,
    b_row: 64,
    b_col: 32,
};
const PADDED: Rows = Rows {
    a_row: 40,
    a_col: 72,
    b_row: 72,
    b_col: 40,
};
/// Halves per slice array in v7 (640 f16vec4).
const ARRAY: u32 = 2_560;

fn count(t: &str, what: &str) -> usize {
    t.lines().filter(|l| l.contains(what)).count()
}

/// Distinct 4-byte words per bank, max over banks, for one request (byte addresses, each `width` bytes).
fn wavefronts(addrs: &[u32], width: u32) -> u32 {
    let mut words: [Vec<u32>; 32] = Default::default();
    for &a in addrs {
        for w in 0..width / 4 {
            let word = a / 4 + w;
            let b = &mut words[(word % 32) as usize];
            if !b.contains(&word) {
                b.push(word);
            }
        }
    }
    words.iter().map(|w| w.len() as u32).max().unwrap()
}

/// The loop loads of a listing: (true = ldsm, byte offset), in order, without the ending's `st_` reads.
fn loop_loads(listing: &str) -> Vec<(bool, u32)> {
    let mut v = Vec::new();
    for line in listing.lines() {
        let is_ldsm = line.contains("ldsm.16.m8n8.x4 [");
        let is_lds = line.contains("ld.shared.strong.cta.b32 [") && !line.contains(".x4+");
        if !(is_ldsm || is_lds) {
            continue;
        }
        let inside = &line[line.find('[').unwrap() + 1..line.find(']').unwrap()];
        let off = inside
            .split('+')
            .find_map(|p| p.strip_prefix("0x"))
            .map(|h| u32::from_str_radix(h, 16).unwrap())
            .unwrap_or(0);
        v.push((is_ldsm, off));
    }
    v
}

/// The block-uniform offsets of the loop loads, from the GLSL formulas, for row lengths `r` (halves) and the
/// op(B) array at byte `b_base`; the order is the listing's (T279 for v6).
fn planned_offsets(r: Rows, b_base: u32) -> (Vec<u32>, Vec<u32>) {
    // lds path: op(A) column-major (TA) and op(B) row-major (no TB): per 16 × 16 fragment 4 loads, the two
    // fragments of a K = 16 step 16 columns apart; rows 8..15 of the fragment (k + 8) = 8 rows further.
    let (sa, sb) = (2 * r.a_col, 2 * r.b_row);
    let a_t = |kk: u32| {
        [
            0,
            0x10,
            8 * sa,
            8 * sa + 0x10,
            0x20,
            0x30,
            8 * sa + 0x20,
            8 * sa + 0x30,
        ]
        .map(|o| o + kk * sa)
    };
    let b_n = |kk: u32| {
        [
            0,
            8 * sb,
            0x10,
            8 * sb + 0x10,
            0x20,
            8 * sb + 0x20,
            0x30,
            8 * sb + 0x30,
        ]
        .map(|o| o + b_base + kk * sb)
    };
    let mut lds = Vec::new();
    for kk in [0, 16] {
        lds.extend(a_t(kk));
        lds.extend(b_n(kk));
    }
    // ldsm path: op(A) row-major (fragment i 16 rows further), op(B) column-major (fragment j 16 rows further)
    let (ra, rb) = (2 * r.a_row, 2 * r.b_col);
    let mut ldsm = Vec::new();
    for kk in [0u32, 16] {
        ldsm.extend([
            2 * kk,
            16 * ra + 2 * kk,
            b_base + 2 * kk,
            b_base + 16 * rb + 2 * kk,
        ]);
    }
    (lds, ldsm)
}

/// Per-lane byte offsets inside one load (the lane formulas of the listing, T279), row length `s` in bytes.
fn lds_lane(s: u32) -> impl Fn(u32) -> u32 {
    move |l| s * (l >> 2) + 4 * (l & 3)
}
fn ldsm_a_lane(s: u32) -> impl Fn(u32) -> u32 {
    move |l| s * (l & 15) + (l & 16)
}
fn ldsm_b_lane(s: u32) -> impl Fn(u32) -> u32 {
    move |l| s * ((l & 7) | ((l & 16) >> 1)) + 2 * (l & 8)
}

#[test]
fn t280a_binary_is_the_recorded_build_with_the_planned_shape() {
    let hex: String = nvidia_ga104::sha256::digest(MM_TC_V7)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, v7::MM_TC_V7_SHA256);
    assert_eq!(MM_TC_V7.len() % 16, 0, "whole 16-byte instructions");
    let (txt, old) = (LISTING_V7, LISTING_V6);
    // W1: the same instruction mix as v6 in every class the plan names
    for what in [
        "hmma",
        ".b128",
        "bar.sync",
        "f2fp",
        "st.shared.strong.cta.b64",
        "ldsm.16.m8n8.x4 [",
        "movm",
    ] {
        assert_eq!(count(txt, what), count(old, what), "{what}");
    }
    let (l7, l6) = (loop_loads(txt), loop_loads(old));
    assert_eq!(l7.len(), l6.len());
    assert_eq!(l7.iter().filter(|x| !x.0).count(), 32, "32 lds");
    assert_eq!(l7.iter().filter(|x| x.0).count(), 8, "8 ldsm");
    assert!(
        txt.contains("c[0x0][+0x700]"),
        "push constants not at 0x700"
    );
    // W2: registers and shared memory, nothing spilled to memory
    assert!(txt.contains("Spills to mem: 0") && txt.contains("SLM size: 0"));
    assert!(txt.contains(&format!("Num GPRs: {}", v7::REGISTERS)));
    const { assert!(v7::REGISTERS <= 128, "W2: more than 128 registers") };
    assert!(txt.contains(&format!("Shared size: {}", v7::SHARED.bytes)));
    assert_eq!(v7::SHARED.bytes, 14_336);
    assert_eq!(v7::SHARED.bytes, MM_TC_V7_NEEDS.shared_bytes);
    assert_eq!(v7::BLOCK, v6::BLOCK);
    let k = Kernel {
        name: "mm_tc_v7",
        threads: 128,
        registers: v7::REGISTERS,
        shared_bytes: v7::SHARED.bytes,
    };
    for r in [Reserve::None, Reserve::OneKib] {
        assert_eq!(blocks_per_sm(&SM86, &k, 64 << 10, r).unwrap().blocks, 4);
    }
}

#[test]
fn t280b_the_listing_reads_at_the_padded_offsets() {
    // v6 first: the formula reproduces T279's offsets (op(B) at 0x1000)
    let (lds6, ldsm6) = planned_offsets(DENSE, 0x1000);
    let v6l = loop_loads(LISTING_V6);
    let got6: Vec<u32> = v6l.iter().filter(|x| !x.0).map(|x| x.1).collect();
    let gotm6: Vec<u32> = v6l.iter().filter(|x| x.0).map(|x| x.1).collect();
    assert_eq!(got6, lds6);
    assert_eq!(gotm6, ldsm6);
    // v7: padded rows, op(B) after the 5 120 bytes of op(A)
    let (lds7, ldsm7) = planned_offsets(PADDED, 2 * ARRAY);
    let v7l = loop_loads(LISTING_V7);
    let got7: Vec<u32> = v7l.iter().filter(|x| !x.0).map(|x| x.1).collect();
    let gotm7: Vec<u32> = v7l.iter().filter(|x| x.0).map(|x| x.1).collect();
    assert_eq!(got7, lds7, "lds offsets");
    assert_eq!(gotm7, ldsm7, "ldsm offsets");
    // every load base 16-byte aligned (ldsm rows), every row start 16-byte aligned at the padded lengths
    for o in got7.iter().chain(&gotm7) {
        assert_eq!(o % 4, 0);
    }
    for o in &gotm7 {
        assert_eq!(o % 16, 0);
    }
    for l in 0..32 {
        assert_eq!(ldsm_a_lane(80)(l) % 16, 0);
        assert_eq!(ldsm_b_lane(80)(l) % 16, 0);
    }
}

#[test]
fn t280c_the_reads_are_conflict_free_on_every_flag() {
    let lds = |s| wavefronts(&(0..32).map(lds_lane(s)).collect::<Vec<_>>(), 4);
    let ldsm = |f: &dyn Fn(u32) -> u32| -> u32 {
        (0..4)
            .map(|m| wavefronts(&(8 * m..8 * m + 8).map(f).collect::<Vec<_>>(), 16))
            .sum()
    };
    // v6 (dense) as T279: 8-way and 4-way
    assert_eq!(lds(128), 8);
    assert_eq!(ldsm(&ldsm_a_lane(64)), 16);
    assert_eq!(ldsm(&ldsm_b_lane(64)), 16);
    // v7: one wavefront per lds, four per ldsm.x4 (one per matrix) — on the row lengths the listing reads at
    assert_eq!(lds(2 * PADDED.a_col), 1);
    assert_eq!(lds(2 * PADDED.b_row), 1);
    assert_eq!(ldsm(&ldsm_a_lane(2 * PADDED.a_row)), 4);
    assert_eq!(ldsm(&ldsm_b_lane(2 * PADDED.b_col)), 4);
    // per block (4 warps) and slice (2 K = 16 steps, 2 fragments per operand): 16 + 16 per warp on every flag
    for (ta, tb) in [(false, false), (false, true), (true, false), (true, true)] {
        let a = if ta {
            8 * lds(144)
        } else {
            2 * ldsm(&ldsm_a_lane(80))
        };
        let b = if tb {
            2 * ldsm(&ldsm_b_lane(80))
        } else {
            8 * lds(144)
        };
        assert_eq!(4 * 2 * (a + b), 128, "ta {ta} tb {tb}");
    }
}

#[test]
fn t280d_only_the_row_lengths_changed_in_the_glsl() {
    // Undo the renames of v7 and drop its new defines and its header: what is left is v6 without its header and its
    // four stride literals.
    let body = |s: &'static str| &s[s.find("#version 460").unwrap()..];
    let mut v7s = body(SRC_V7).to_string();
    // C-168 p.5 P5b (session 291): the `-DGELU` store (mm_tc_v7_gelu) is not part of mm_tc_v7 (its bytes are unchanged,
    // T307): drop the `#ifdef GELU ... #else` lines and the `#endif` before comparing with v6.
    // C-168 p.5 P5c (session 293, T310): the same for every `#ifdef GELU_FWD ... #endif` block (the act store and the
    // act push constant), first, since one of them sits inside the `#else` above.
    // C-168 p.5 P5d (session 294, T311): the same for every `#ifdef RES ... #endif` block (the res push constant and the
    // residual add).
    while let Some(a) = v7s.find("#ifdef RES\n") {
        let e = v7s[a..].find("#endif\n").unwrap() + a + "#endif\n".len();
        v7s.replace_range(a..e, "");
    }
    while let Some(a) = v7s.find("#ifdef GELU_FWD\n") {
        let e = v7s[a..].find("#endif\n").unwrap() + a + "#endif\n".len();
        v7s.replace_range(a..e, "");
    }
    if let Some(a) = v7s.find("#ifdef GELU\n") {
        let b = v7s[a..].find("#else\n").unwrap() + a + "#else\n".len();
        v7s.replace_range(a..b, "");
        let e = v7s[a..].find("#endif\n").unwrap() + a;
        v7s.replace_range(e..e + "#endif\n".len(), "");
    }
    for (a, b) in [
        ("r * LA + p, p * LAT + r", "r * BK + p, p * BM + r"),
        ("p * LB + cl, cl * LBT + p", "p * BN + cl, cl * BK + p"),
        ("kk * LAT + wr", "kk * BM + wr"),
        ("(wr + 16u * i) * LA + kk", "(wr + 16u * i) * BK + kk"),
        ("(wc + 16u * j) * LBT + kk", "(wc + 16u * j) * BK + kk"),
        ("kk * LB + wc", "kk * BN + wc"),
        ("as4_[AS4]", "as4_[(BM * BK) / 4u]"),
        ("bs4_[BS4]", "bs4_[(BK * BN) / 4u]"),
        ("(row LA)", "(stride BK)"),
        ("(row LAT)", "(stride BM)"),
        ("(row LB)", "(stride BN)"),
        ("(row LBT)", "(stride BK)"),
    ] {
        assert_eq!(v7s.matches(a).count(), 1, "{a}");
        v7s = v7s.replace(a, b);
    }
    let keep = |l: &str| {
        !(l.starts_with("#define L")
            || l.starts_with("#define S")
            || l.starts_with("#define AS4")
            || l.starts_with("#define BS4")
            || l.starts_with("// Row lengths in halves")
            || l.starts_with("// coopMatLoad strides")
            || l.starts_with("// f16vec4 per slice buffer"))
    };
    let strip = |s: &str| -> String {
        s.lines()
            .filter(|l| keep(l))
            .map(|l| format!("{l}\n"))
            .collect()
    };
    assert_eq!(strip(&v7s), strip(body(SRC_V6)));
    // the new stride literals are the row lengths / 4, and each row length is the dense one + 8 halves
    for (def, v) in [
        ("#define SA 10u", PADDED.a_row / 4),
        ("#define SAT 18u", PADDED.a_col / 4),
        ("#define SB 18u", PADDED.b_row / 4),
        ("#define SBT 10u", PADDED.b_col / 4),
        ("#define LA 40u", PADDED.a_row),
        ("#define LAT 72u", PADDED.a_col),
        ("#define LB 72u", PADDED.b_row),
        ("#define LBT 40u", PADDED.b_col),
        ("#define AS4 640u", ARRAY / 4),
        ("#define BS4 640u", ARRAY / 4),
    ] {
        assert!(SRC_V7.contains(def), "{def}");
        assert!(def.contains(&format!(" {v}u")), "{def} != {v}");
    }
    assert_eq!(PADDED.a_row, DENSE.a_row + 8);
    assert_eq!(PADDED.a_col, DENSE.a_col + 8);
    assert_eq!(PADDED.b_row, DENSE.b_row + 8);
    assert_eq!(PADDED.b_col, DENSE.b_col + 8);
}

/// One element of an f16 slice in shared memory: (which operand, row of op(X), column of op(X)); `None` = never
/// written.
type Half = Option<(char, u32, u32)>;

/// The slice of op(A) (64 × 32, (r, p)) or op(B) (32 × 64, (p, c)) as `mm_tc_v6.comp` / `mm_tc_v7.comp` write it
/// (one f16vec4 at `base / 4`, column-major when transposed) with row lengths `rows`. Returns the halves of the
/// array and how often each was written.
fn slice(op: char, trans: bool, rows: Rows) -> (Vec<Half>, Vec<u32>) {
    let mut mem: Vec<Half> = vec![None; ARRAY as usize];
    let mut writes = vec![0u32; ARRAY as usize];
    for tid in 0..128u32 {
        for s in 0..4u32 {
            let e = tid + s * 128;
            let (x, y) = if op == 'A' {
                if trans {
                    (4 * (e % 16), e / 16)
                } else {
                    (e / 8, 4 * (e % 8))
                }
            } else if trans {
                (4 * (e % 8), e / 8)
            } else {
                (e / 16, 4 * (e % 16))
            };
            let lane = |q: u32| -> (u32, u32) {
                match (op, trans) {
                    ('A', false) | ('B', false) => (x, y + q),
                    _ => (x + q, y),
                }
            };
            let base = match (op, trans) {
                ('A', false) => x * rows.a_row + y,
                ('A', true) => y * rows.a_col + x,
                ('B', false) => x * rows.b_row + y,
                _ => y * rows.b_col + x,
            };
            assert_eq!(base % 4, 0);
            for q in 0..4u32 {
                let at = (4 * (base / 4) + q) as usize;
                mem[at] = Some((op, lane(q).0, lane(q).1));
                writes[at] += 1;
            }
        }
    }
    (mem, writes)
}

/// `coopMatLoad` of one 16 × 16 fragment, in halves; `None` if it reads past the array.
fn fragment(mem: &[Half], start: u32, stride: u32, col_major: bool) -> Vec<Half> {
    let mut f = Vec::new();
    for row in 0..16u32 {
        for col in 0..16u32 {
            let at = if col_major {
                start + col * stride + row
            } else {
                start + row * stride + col
            };
            f.push(mem.get(at as usize).copied().flatten());
        }
    }
    f
}

/// Every fragment every warp loads, for one flag combination, with the stores at `store` row lengths and the
/// reads at `read` row lengths (the GLSL's element / stride expressions, in halves).
fn fragments(ta: bool, tb: bool, store: Rows, read: Rows) -> Vec<Vec<Half>> {
    let (ma, _) = slice('A', ta, store);
    let (mb, _) = slice('B', tb, store);
    let mut out = Vec::new();
    for w in 0..4u32 {
        let (wr, wc) = ((w / 2) * 32, (w % 2) * 32);
        for kk in [0u32, 16] {
            for i in 0..2u32 {
                out.push(if ta {
                    fragment(
                        &ma,
                        4 * ((kk * read.a_col + wr + 16 * i) / 4),
                        read.a_col,
                        true,
                    )
                } else {
                    fragment(
                        &ma,
                        4 * (((wr + 16 * i) * read.a_row + kk) / 4),
                        read.a_row,
                        false,
                    )
                });
            }
            for j in 0..2u32 {
                out.push(if tb {
                    fragment(
                        &mb,
                        4 * (((wc + 16 * j) * read.b_col + kk) / 4),
                        read.b_col,
                        true,
                    )
                } else {
                    fragment(
                        &mb,
                        4 * ((kk * read.b_row + wc + 16 * j) / 4),
                        read.b_row,
                        false,
                    )
                });
            }
        }
    }
    out
}

#[test]
fn t280e_every_fragment_gets_the_same_elements_as_in_mm_tc_v6() {
    for ta in [false, true] {
        for tb in [false, true] {
            let old = fragments(ta, tb, DENSE, DENSE);
            let new = fragments(ta, tb, PADDED, PADDED);
            assert!(
                old.iter().flatten().all(|h| h.is_some()),
                "v6 reads only written halves"
            );
            assert!(
                new.iter().flatten().all(|h| h.is_some()),
                "v7 reads a padding half"
            );
            assert_eq!(old, new, "ta {ta} tb {tb}: mm_tc_v7 feeds other values");
            // the planted mistake: the layout this flag case reads through, read with the unpadded row
            let mut wrong = PADDED;
            if ta {
                wrong.a_col = DENSE.a_col;
            } else {
                wrong.a_row = DENSE.a_row;
            }
            assert_ne!(
                fragments(ta, tb, PADDED, wrong),
                old,
                "planted mistake (A) ta {ta} tb {tb}"
            );
            let mut wrong = PADDED;
            if tb {
                wrong.b_col = DENSE.b_col;
            } else {
                wrong.b_row = DENSE.b_row;
            }
            assert_ne!(
                fragments(ta, tb, PADDED, wrong),
                old,
                "planted mistake (B) ta {ta} tb {tb}"
            );
        }
    }
}

#[test]
fn t280f_every_half_written_once_padding_never_touched() {
    for (op, trans, len, rows_n) in [
        ('A', false, PADDED.a_row, 64u32),
        ('A', true, PADDED.a_col, 32),
        ('B', false, PADDED.b_row, 32),
        ('B', true, PADDED.b_col, 64),
    ] {
        let (mem, writes) = slice(op, trans, PADDED);
        assert!(
            writes.iter().all(|&w| w <= 1),
            "{op} {trans}: written twice"
        );
        assert_eq!(
            writes.iter().filter(|&&w| w == 1).count(),
            2_048,
            "{op} {trans}"
        );
        // the 8 padding halves of each row are never written, and nothing lands past the last row
        for (i, h) in mem.iter().enumerate() {
            let (row, col) = (i as u32 / len, i as u32 % len);
            let data = row < rows_n && col < len - 8;
            assert_eq!(h.is_some(), data, "{op} {trans}: half {i}");
        }
        assert!(rows_n * len <= ARRAY);
    }
}

#[test]
fn t280g_same_bits_as_mm_tc_through_the_command_processor() {
    let mut r = rng(0x0266_c157_0007);
    let mut runs = 0;
    for flags in 0..8u32 {
        for (m, k, n) in aligned_shapes(flags & 3) {
            let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
            let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
            let bias: Vec<f32> = (0..n).map(|_| r()).collect();
            let p = Product {
                a: &a,
                b: &b,
                bias: (flags & tc::BIAS != 0).then_some(&bias[..]),
                m,
                n,
                k,
                flags,
            };
            let ar = args(&p);
            let old = run_at(&MM_TC_P, &p, ar.grid(), tc::BLOCK, 0, 0).unwrap();
            let six = run_at(&MM_TC_V6_P, &p, v6::grid(&ar), v6::BLOCK, 0, 0).unwrap();
            let new = run_at(&MM_TC_V7_P, &p, v7::grid(&ar), v7::BLOCK, 0, 0).unwrap();
            assert!(
                old.iter().all(|&x| x != SENTINEL.to_bits()),
                "C not written"
            );
            assert_eq!(old, new, "{m}x{k}x{n} flags {flags}: mm_tc_v7 != mm_tc");
            assert_eq!(six, new, "{m}x{k}x{n} flags {flags}: mm_tc_v7 != mm_tc_v6");
            runs += 1;
        }
    }
    eprintln!("T280g: {runs} products, same bits");
    assert!(runs >= 30);
}

#[test]
fn t280h_model_refuses_what_mm_tc_v6_refuses() {
    let mut r = rng(11);
    let (m, k, n) = (200usize, 32usize, 300usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    let p = Product {
        a: &a,
        b: &b,
        bias: None,
        m,
        n,
        k,
        flags: 0,
    };
    let g = v7::grid(&args(&p));
    assert!(
        run_at(&MM_TC_V7_P, &p, g, [256, 1, 1], 0, 0).is_err(),
        "256 threads"
    );
    assert!(
        run_at(&MM_TC_V7_P, &p, [4, 4, 1], v7::BLOCK, 0, 0).is_err(),
        "short grid"
    );
    assert!(run_at(&MM_TC_V7_P, &p, g, v7::BLOCK, 0, 0).is_ok());
    for (da, db) in [(4u64, 0u64), (0, 8), (12, 12)] {
        assert!(
            run_at(&MM_TC_V7_P, &p, g, v7::BLOCK, da, db).is_err(),
            "+{da}/+{db}"
        );
    }
    let (m, k, n) = (66usize, 34usize, 62usize);
    let a: Vec<f32> = (0..m * k).map(|_| r()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| r()).collect();
    for flags in 0..4u32 {
        let p = Product {
            a: &a,
            b: &b,
            bias: None,
            m,
            n,
            k,
            flags,
        };
        assert!(!v7::fits(&args(&p)));
        assert!(run_at(&MM_TC_V7_P, &p, v7::grid(&args(&p)), v7::BLOCK, 0, 0).is_err());
    }
}
