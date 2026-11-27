//! T276 (C-157 p.2, `docs/C-157_MM_TC_V6_PLAN.md`, session 263), SIMULARE: `mm_tc_v5` with the f16 slices written
//! four values per 64-bit store (`tools/nak/mm_tc_v6.comp` → `tools/nak/out2/mm_tc_v6.sm86.sass`, second NAK tool,
//! Mesa 26.3.1). Checked: the binary is the recorded build (V1: 128 registers, nothing spilled to memory, 12 288 B;
//! V2: 8 stores of 64 bits and 16 paired conversions in the loop instead of 32 + 32; the same `hmma` and 128-bit
//! loads); the GLSL outside the slice stores and the two `coopMatLoad` pairs is `mm_tc_v5`'s, character for character;
//! the new stores and reads, written out in Rust lane by lane for every flag combination, give every 16×16 fragment
//! exactly the elements `mm_tc_v5` gives it (a planted mistake — a transposed slice read `RowMajor` — is caught), and
//! every half of a slice is written exactly once; through our pushbuffer + QMD in the command-processor model
//! `mm_tc_v6` gives the same bits as `mm_tc` and refuses what `mm_tc_v5` refuses. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::mm_tc::Product;
use gpu_sim::cmdproc::{
    CommandProcessor, MM_TC_NEEDS, MM_TC_V6_NEEDS, ProgramFn, mm_tc_on_cpu, mm_tc_v6_on_cpu,
};
use nvidia_ga104::mm_tc::{self as tc, MM_TC};
use nvidia_ga104::mm_tc_v5::{self as v5};
use nvidia_ga104::mm_tc_v6::{self as v6, MM_TC_V6};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

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
        v6::fits(&tc::Args {
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

#[test]
fn t276a_binary_is_the_recorded_build_with_the_planned_shape() {
    let hex: String = nvidia_ga104::sha256::digest(MM_TC_V6)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, v6::MM_TC_V6_SHA256);
    assert_eq!(MM_TC_V6.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out2/mm_tc_v6.sm86.txt");
    let old = include_str!("../../../tools/nak/out2/mm_tc_v5.sm86.txt");
    let count = |t: &str, what: &str| t.lines().filter(|l| l.contains(what)).count();
    // the same tensor-core chain and 128-bit loads
    assert_eq!(count(txt, "hmma"), count(old, "hmma"), "hmma");
    assert_eq!(count(txt, ".b128"), count(old, ".b128"), "128-bit loads");
    assert_eq!(count(txt, "bar.sync"), count(old, "bar.sync"), "barriers");
    // V2 / V2b: in v5 the loop has 32 `f2f` and 32 16-bit stores; in v6 none of either, 16 paired conversions and
    // 8 64-bit stores (the prologue and the ending have no f16 stores in either)
    assert_eq!(count(old, "= f2f.f16.f32"), 32);
    assert_eq!(count(old, "st.shared.strong.cta.u16"), 32);
    assert_eq!(count(txt, "= f2f.f16.f32"), 0);
    assert_eq!(count(txt, "st.shared.strong.cta.u16"), 0);
    assert_eq!(count(txt, "f2fp"), 16);
    assert_eq!(
        count(txt, "st.shared.strong.cta.b64") - count(old, "st.shared.strong.cta.b64"),
        8
    );
    assert!(
        txt.contains("c[0x0][+0x700]"),
        "push constants not at 0x700"
    );
    // V1: registers and shared memory as planned, nothing spilled to memory
    assert!(txt.contains("Spills to mem: 0") && txt.contains("SLM size: 0"));
    assert!(txt.contains(&format!("Num GPRs: {}", v6::REGISTERS)));
    const { assert!(v6::REGISTERS <= 128, "V1: more than 128 registers") };
    assert!(txt.contains("Max warps/SM: 16"));
    assert!(txt.contains(&format!("Shared size: {}", v6::SHARED.bytes)));
    assert_eq!(v6::SHARED.bytes, MM_TC_V6_NEEDS.shared_bytes);
    assert_eq!(v6::SHARED.bytes, 12_288);
    assert_eq!(v6::BLOCK, v5::BLOCK);
}

#[test]
fn t276b_only_the_slice_stores_and_reads_changed() {
    let v5s = include_str!("../../../tools/nak/mm_tc_v5.comp");
    let v6s = include_str!("../../../tools/nak/mm_tc_v6.comp");
    let from = |s: &'static str, a: &str| &s[s.find(a).expect(a)..];
    // load_slice + the start of main (up to the K loop), and the whole ending: the same text
    let (a, b) = ("#define BM", "shared float16_t as_");
    // v6 adds only the four stride literals (and their comment line); each must be the expression it stands for
    let added = [
        "// coopMatLoad strides in f16vec4 units, written as literals so the stride lint (program_lint) can check them:",
        "#define SA 8u    // BK / 4: op(A) row-major, 64 bytes",
        "#define SAT 16u  // BM / 4: op(A) column-major, 128 bytes",
        "#define SB 16u   // BN / 4: op(B) row-major, 128 bytes",
        "#define SBT 8u   // BK / 4: op(B) column-major, 64 bytes",
    ];
    let (bm, bn, bk) = (64u32, 64u32, 32u32);
    assert_eq!((bk / 4, bm / 4, bn / 4, bk / 4), (8, 16, 16, 8));
    let decl6: String = {
        let t = from(v6s, a);
        t[..t.find("// op(A) slice: row-major").unwrap()]
            .lines()
            .filter(|l| !added.contains(l))
            .map(|l| format!("{l}\n"))
            .collect()
    };
    assert!(added.iter().all(|l| v6s.contains(l)));
    assert_eq!(&from(v5s, a)[..from(v5s, a).find(b).unwrap()], decl6);
    let span = |s: &'static str, a: &str, b: &str| {
        let t = from(s, a);
        &t[..t.find(b).expect(b)]
    };
    assert_eq!(
        span(
            v5s,
            "shared float st_",
            "    for (uint t = 0u; t < args.k; t += BK) {"
        ),
        span(
            v6s,
            "shared float st_",
            "    for (uint t = 0u; t < args.k; t += BK) {"
        )
    );
    assert_eq!(
        from(v5s, "        barrier();\n        // The next slice's loads")
            .split("        for (uint kk = 0u;")
            .next(),
        from(v6s, "        barrier();\n        // The next slice's loads")
            .split("        for (uint kk = 0u;")
            .next()
    );
    assert_eq!(
        from(
            v5s,
            "            for (uint i = 0u; i < 2u; i++) {\n                for (uint j = 0u; j < 2u; j++) {\n                    acc[i][j] = coopMatMulAdd"
        ),
        from(
            v6s,
            "            for (uint i = 0u; i < 2u; i++) {\n                for (uint j = 0u; j < 2u; j++) {\n                    acc[i][j] = coopMatMulAdd"
        )
    );
    assert!(!v6s.contains("float16_t(v."), "no single 16-bit store left");
    assert_eq!(v6s.matches("= f16vec4(v);").count(), 2);
    assert_eq!(
        v6s.matches("gl_CooperativeMatrixLayoutColumnMajor").count(),
        2
    );
}

/// One element of an f16 slice in shared memory: (which operand, row of op(X), column of op(X)); `None` = never
/// written.
type Half = Option<(char, u32, u32)>;

/// The slice of op(A) (64 × 32, (r, p)) or op(B) (32 × 64, (p, c)) as the threads of a block write it.
/// `v6 = false`: `mm_tc_v5.comp` (4 halves, step 1 or BK / BN, row-major); `v6 = true`: `mm_tc_v6.comp` (one f16vec4 at
/// `base / 4`, column-major when transposed). Returns the 2 048 halves and how often each was written.
fn slice(op: char, trans: bool, v6: bool) -> (Vec<Half>, Vec<u32>) {
    let (bm, bn, bk) = (64u32, 64u32, 32u32);
    let mut mem: Vec<Half> = vec![None; 2048];
    let mut writes = vec![0u32; 2048];
    for tid in 0..128u32 {
        for s in 0..4u32 {
            let e = tid + s * 128;
            // the element of op(X) each of the 4 vector lanes holds (load_slice, unchanged)
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
                // A^T vectors run along r, A along p; B^T along p, B along c
                match (op, trans) {
                    ('A', false) | ('B', false) => (x, y + q),
                    _ => (x + q, y),
                }
            };
            for q in 0..4u32 {
                let at = if !v6 {
                    let (base, step) = if op == 'A' {
                        (x * bk + y, if trans { bk } else { 1 })
                    } else {
                        (x * bn + y, if trans { bn } else { 1 })
                    };
                    base + q * step
                } else {
                    let base = match (op, trans) {
                        ('A', false) => x * bk + y,
                        ('A', true) => y * bm + x,
                        ('B', false) => x * bn + y,
                        _ => y * bk + x,
                    };
                    assert_eq!(base % 4, 0);
                    4 * (base / 4) + q
                };
                mem[at as usize] = Some((op, lane(q).0, lane(q).1));
                writes[at as usize] += 1;
            }
        }
    }
    (mem, writes)
}

/// `coopMatLoad` of one 16 × 16 fragment: element (row, col) of the fragment from `start + row * stride + col`
/// (`RowMajor`) or `start + col * stride + row` (`ColumnMajor`), in halves (the v6 element / stride are f16vec4
/// units × 4).
fn fragment(mem: &[Half], start: u32, stride: u32, col_major: bool) -> Vec<Half> {
    let mut f = Vec::new();
    for row in 0..16u32 {
        for col in 0..16u32 {
            let at = if col_major {
                start + col * stride + row
            } else {
                start + row * stride + col
            };
            f.push(mem[at as usize]);
        }
    }
    f
}

/// Every fragment every warp loads, for one flag combination, as `mm_tc_v5.comp` reads (`v6 = false`) or as
/// `mm_tc_v6.comp` reads (`v6 = true`; `wrong` = the planted mistake: the transposed slices read `RowMajor`).
fn fragments(ta: bool, tb: bool, v6: bool, wrong: bool) -> Vec<Vec<Half>> {
    let (bm, bn, bk) = (64u32, 64u32, 32u32);
    let (ma, _) = slice('A', ta, v6);
    let (mb, _) = slice('B', tb, v6);
    let mut out = Vec::new();
    for w in 0..4u32 {
        let (wr, wc) = ((w / 2) * 32, (w % 2) * 32);
        for kk in [0u32, 16] {
            for i in 0..2u32 {
                out.push(if v6 && ta {
                    let start = 4 * ((kk * bm + wr + 16 * i) / 4);
                    fragment(&ma, start, 4 * (bm / 4), !wrong)
                } else if v6 {
                    // coopMatLoad(fa[i], as4_, ((wr + 16i) * BK + kk) / 4, BK / 4, RowMajor), in halves
                    fragment(
                        &ma,
                        4 * (((wr + 16 * i) * bk + kk) / 4),
                        4 * (bk / 4),
                        false,
                    )
                } else {
                    fragment(&ma, (wr + 16 * i) * bk + kk, bk, false)
                });
            }
            for j in 0..2u32 {
                out.push(if v6 && tb {
                    let start = 4 * (((wc + 16 * j) * bk + kk) / 4);
                    fragment(&mb, start, 4 * (bk / 4), !wrong)
                } else if v6 {
                    fragment(&mb, 4 * ((kk * bn + wc + 16 * j) / 4), 4 * (bn / 4), false)
                } else {
                    fragment(&mb, kk * bn + wc + 16 * j, bn, false)
                });
            }
        }
    }
    out
}

#[test]
fn t276c_every_fragment_gets_the_same_elements_as_in_mm_tc_v5() {
    for ta in [false, true] {
        for tb in [false, true] {
            let old = fragments(ta, tb, false, false);
            let new = fragments(ta, tb, true, false);
            assert!(
                old.iter().flatten().all(|h| h.is_some()),
                "v5 reads only written halves"
            );
            // fragment A (i) holds op(A)[wr + 16 i + row][kk + col], fragment B (j) op(B)[kk + row][wc + 16 j + col]
            for (n, f) in old.iter().enumerate() {
                let (w, kk, x) = (n as u32 / 8, (n as u32 / 4) % 2 * 16, n as u32 % 4);
                let (wr, wc) = ((w / 2) * 32, (w % 2) * 32);
                for (at, h) in f.iter().enumerate() {
                    let (row, col) = (at as u32 / 16, at as u32 % 16);
                    let want = if x < 2 {
                        ('A', wr + 16 * x + row, kk + col)
                    } else {
                        ('B', kk + row, wc + 16 * (x - 2) + col)
                    };
                    assert_eq!(*h, Some(want), "v5 ta {ta} tb {tb} fragment {n}");
                }
            }
            assert_eq!(old, new, "ta {ta} tb {tb}: mm_tc_v6 feeds other values");
            if ta || tb {
                assert_ne!(
                    fragments(ta, tb, true, true),
                    old,
                    "the planted mistake must be caught (ta {ta} tb {tb})"
                );
            }
        }
    }
}

#[test]
fn t276d_every_half_of_a_slice_is_written_once() {
    for op in ['A', 'B'] {
        for trans in [false, true] {
            for v6 in [false, true] {
                let (mem, writes) = slice(op, trans, v6);
                assert!(writes.iter().all(|&w| w == 1), "{op} trans {trans} v6 {v6}");
                assert!(mem.iter().all(|h| h.is_some()));
            }
        }
    }
}

#[test]
fn t276e_same_bits_as_mm_tc_through_the_command_processor() {
    let mut r = rng(0x0263_c157_0006);
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
            let new = run_at(&MM_TC_V6_P, &p, v6::grid(&ar), v6::BLOCK, 0, 0).unwrap();
            assert!(
                old.iter().all(|&x| x != SENTINEL.to_bits()),
                "C not written"
            );
            assert_eq!(old, new, "{m}x{k}x{n} flags {flags}: mm_tc_v6 != mm_tc");
            runs += 1;
        }
    }
    eprintln!("T276e: {runs} products, same bits");
    assert!(runs >= 30);
}

#[test]
fn t276f_model_refuses_what_mm_tc_v5_refuses() {
    let mut r = rng(10);
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
    let g = v6::grid(&args(&p));
    assert!(
        run_at(&MM_TC_V6_P, &p, g, [256, 1, 1], 0, 0).is_err(),
        "256 threads"
    );
    assert!(
        run_at(&MM_TC_V6_P, &p, [4, 4, 1], v6::BLOCK, 0, 0).is_err(),
        "short grid"
    );
    assert!(run_at(&MM_TC_V6_P, &p, g, v6::BLOCK, 0, 0).is_ok());
    for (da, db) in [(4u64, 0u64), (0, 8), (12, 12)] {
        assert!(
            run_at(&MM_TC_V6_P, &p, g, v6::BLOCK, da, db).is_err(),
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
        assert!(!v6::fits(&args(&p)));
        assert!(run_at(&MM_TC_V6_P, &p, v6::grid(&args(&p)), v6::BLOCK, 0, 0).is_err());
    }
}
