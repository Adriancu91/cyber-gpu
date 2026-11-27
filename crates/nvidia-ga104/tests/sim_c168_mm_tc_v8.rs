//! T295 (C-168 p.3, `docs/C-168_V8_PLAN.md` §2 / §4, session 279), SIMULARE: `mm_tc_v8` (128×128, 256 threads) and
//! `mm_tc_v8n` (128×64, 128 threads), one source `tools/nak/mm_tc_v8.comp` → `tools/nak/out2/mm_tc_v8{,n}.sm86.sass`
//! (second NAK tool, Mesa 26.3.1). Checked:
//! - G1 (compilation): the binaries are the recorded builds; registers, shared memory, nothing spilled; blocks per SM
//!   at 64 KiB as the plan needs (v8: 2, v8n: 3) — or, if the compiler gave more registers, what it gives, written down.
//! - G2 (listing of the K loop): the shared reads are only `ldsm` (4 `.x4` for A + 4 `.x2` for B per warp and slice:
//!   the accumulators are 16×8), no 32-bit `lds` and no `movm` anywhere, one block barrier in the loop, 16 `hmma` per
//!   warp and slice, on every flag (no branch on the flags around the reads); the register copies in the loop are
//!   counted (v8 148, v8n 41: the finding of §7 of the plan).
//! - The layout, lane by lane in Rust for every flag and both programs: every 16×16 fragment the loop reads holds
//!   exactly op(A) / op(B) of that slice (0 outside the matrix), every half of a stage is written once, no padding half
//!   is written or read, the two stages never mix; the per-lane `ldsm` addresses are conflict-free (public CUDA 8.6
//!   bank model). A planted mistake (B read row-major) is caught.
//! - G3: through our pushbuffer + QMD in the command-processor model, `mm_tc_v8` / `mm_tc_v8n` give the same bits as
//!   `mm_tc` on the edge sets (TA / TB / BIAS, partial blocks) and refuse unaligned products and wrong blocks.
//!
//! Nothing here runs on the RTX 3070 Ti.

use gpu_compute::mm_tc::Product;
use gpu_sim::cmdproc::{
    CommandProcessor, MM_TC_NEEDS, MM_TC_V8_NEEDS, MM_TC_V8N_NEEDS, ProgramFn, mm_tc_on_cpu,
    mm_tc_v8_on_cpu, mm_tc_v8n_on_cpu,
};
use nvidia_ga104::mm_tc::{self as tc, MM_TC};
use nvidia_ga104::mm_tc_v8::{self as v8, MM_TC_V8, MM_TC_V8N};
use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const LISTING_V8: &str = include_str!("../../../tools/nak/out2/mm_tc_v8.sm86.txt");
const LISTING_V8N: &str = include_str!("../../../tools/nak/out2/mm_tc_v8n.sm86.txt");
const SRC: &str = include_str!("../../../tools/nak/mm_tc_v8.comp");

// ---------------------------------------------------------------------------------------------------------------
// The layout, lane by lane (the formulas of load_unit / store_unit / the coopMatLoad offsets in mm_tc_v8.comp).
// ---------------------------------------------------------------------------------------------------------------

/// Halves per shared row (16 values + 8 of padding = 48 bytes).
const LS: usize = 24;
const BM: usize = 128;
const BK: usize = 16;

#[derive(Clone, Copy, Debug)]
struct Shape {
    bn: usize,
    threads: usize,
    wcols: usize,
    nua: usize,
    nub: usize,
}
const V8S: Shape = Shape {
    bn: 128,
    threads: 256,
    wcols: 4,
    nua: 1,
    nub: 1,
};
const V8NS: Shape = Shape {
    bn: 64,
    threads: 128,
    wcols: 2,
    nua: 2,
    nub: 1,
};

/// What a shared half holds: `None` never written, `Some(None)` a zero for outside the matrix, `Some(Some(i))` the
/// stored element `i` of A (`i`) or of B (`i | 1 << 40`).
type Cell = Option<Option<u64>>;
const BTAG: u64 = 1 << 40;

/// The unit of load_unit: (x, p) of its first element and whether the vectors run along x (transposed).
fn unit_xp(u: usize, xq: usize, tr: bool) -> (usize, usize) {
    if tr {
        (4 * (u % xq), 2 * (u / xq))
    } else {
        (u / 2, 8 * (u % 2))
    }
}

/// Writes one operand of slice `t` into `sh` at half `base`, as the threads of the program do. `x_total`: rows of
/// the operand in the whole matrix (m for A, n for B); `xs`: the block's first one; `ld`: the stored row length.
#[allow(clippy::too_many_arguments)]
fn store_operand(
    sh: &mut [Cell],
    base: usize,
    s: Shape,
    nu: usize,
    xrows: usize,
    tr: bool,
    xs: usize,
    x_total: usize,
    k: usize,
    t: usize,
    ld: usize,
    tag: u64,
) {
    let xq = xrows / 4;
    for tid in 0..s.threads {
        for su in 0..nu {
            let u = tid + su * s.threads;
            let (x, p) = unit_xp(u, xq, tr);
            // the 8 elements of the unit: (dx, dp) pairs in the order of the two vec4
            let elems: Vec<(usize, usize)> = if tr {
                (0..2).flat_map(|v| (0..4).map(move |i| (i, v))).collect()
            } else {
                (0..8).map(|i| (0, i)).collect()
            };
            for (dx, dp) in elems {
                let (gx, gp) = (xs + x + dx, t + p + dp);
                // a vec4 is inside or outside as a whole: the bound is checked on its first element
                let (fx, fp) = if tr {
                    (xs + x, t + p + dp)
                } else {
                    (xs + x, t + p + 4 * (dp / 4))
                };
                let inside = fx < x_total && fp < k;
                let v = if inside {
                    assert!(gx < x_total && gp < k, "a vec4 straddles the edge");
                    let at = if tr { gp * ld + gx } else { gx * ld + gp };
                    Some(at as u64 | tag)
                } else {
                    None
                };
                let h = base + (x + dx) * LS + p + dp;
                assert!((p + dp) < BK, "writes the padding");
                assert!(sh[h].is_none(), "half {h} written twice");
                sh[h] = Some(v);
            }
        }
    }
}

/// The element `(r, c)` of a 16×16 fragment read at half `off` with row stride 24 halves: row-major for A
/// (r * 24 + c), column-major for B (c * 24 + r).
fn frag(sh: &[Cell], off: usize, col_major: bool, r: usize, c: usize) -> Option<u64> {
    let h = if col_major {
        off + c * LS + r
    } else {
        off + r * LS + c
    };
    let i = if col_major { r } else { c };
    assert!(i < BK, "reads the padding");
    sh[h].expect("reads a half nobody wrote")
}

#[derive(Clone, Copy)]
struct Case {
    m: usize,
    n: usize,
    k: usize,
    ta: bool,
    tb: bool,
    by: usize,
    bx: usize,
}

/// One block and one slice: fills a stage like the program does, then checks every fragment of every warp.
fn check_slice(s: Shape, cs: Case, t: usize, b_read_row_major: bool) -> Result<(), String> {
    let ast = BM * LS;
    let bst = s.bn * LS;
    let mut sh: Vec<Cell> = vec![None; ast + bst];
    let (row0, col0) = (cs.by * BM, cs.bx * s.bn);
    let lda = if cs.ta { cs.m } else { cs.k };
    let ldb = if cs.tb { cs.k } else { cs.n };
    store_operand(&mut sh, 0, s, s.nua, BM, cs.ta, row0, cs.m, cs.k, t, lda, 0);
    store_operand(
        &mut sh, ast, s, s.nub, s.bn, !cs.tb, col0, cs.n, cs.k, t, ldb, BTAG,
    );
    // every non-padding half of the stage is written exactly once
    for (h, cell) in sh.iter().enumerate() {
        // both operand areas start on a row boundary, so the column is h % LS
        if h % LS < BK {
            assert!(cell.is_some(), "half {h} never written");
        } else {
            assert!(cell.is_none(), "padding half {h} written");
        }
    }
    let warps = s.threads / 32;
    for w in 0..warps {
        let wr = (w / s.wcols) * 64;
        let wc = (w % s.wcols) * 32;
        // A fragments 16×16, B fragments 16×8 (as the program reads them)
        for i in 0..4 {
            let off = (wr + 16 * i) * LS;
            for r in 0..16 {
                for p in 0..16 {
                    let (gr, gp) = (row0 + wr + 16 * i + r, t + p);
                    let want = (gr < cs.m && gp < cs.k).then(|| {
                        (if cs.ta {
                            gp * cs.m + gr
                        } else {
                            gr * cs.k + gp
                        }) as u64
                    });
                    if frag(&sh, off, false, r, p) != want {
                        return Err(format!("A w{w} i{i} ({r},{p})"));
                    }
                }
            }
        }
        for j in 0..4 {
            let off = ast + (wc + 8 * j) * LS;
            for p in 0..16 {
                for cc in 0..8 {
                    let (gp, gc) = (t + p, col0 + wc + 8 * j + cc);
                    let want = (gp < cs.k && gc < cs.n).then(|| {
                        (if cs.tb {
                            gc * cs.k + gp
                        } else {
                            gp * cs.n + gc
                        }) as u64
                            | BTAG
                    });
                    let got = if b_read_row_major {
                        // the planted mistake: [p][c]
                        let h = off + p * LS + cc;
                        if cc >= BK {
                            return Err("reads the padding".into());
                        }
                        sh[h].expect("unwritten")
                    } else {
                        frag(&sh, off, true, p, cc)
                    };
                    if got != want {
                        return Err(format!("B w{w} j{j} ({p},{cc})"));
                    }
                }
            }
        }
    }
    Ok(())
}

fn cases() -> Vec<Case> {
    let mut v = Vec::new();
    for &(m, n, k) in &[
        (256usize, 256usize, 48usize),
        (132, 68, 20),
        (4, 4, 4),
        (200, 300, 36),
    ] {
        for ta in [false, true] {
            for tb in [false, true] {
                for by in 0..m.div_ceil(BM) {
                    for bx in 0..n.div_ceil(64) {
                        v.push(Case {
                            m,
                            n,
                            k,
                            ta,
                            tb,
                            by,
                            bx,
                        });
                    }
                }
            }
        }
    }
    v
}

#[test]
fn t295c_every_fragment_holds_op_a_and_op_b_on_every_flag() {
    let mut checked = 0;
    for s in [V8S, V8NS] {
        for cs in cases() {
            if cs.bx * s.bn >= cs.n.max(1) && cs.bx > 0 {
                continue;
            }
            for t in (0..cs.k).step_by(BK) {
                check_slice(s, cs, t, false).unwrap_or_else(|e| panic!("{s:?} {e}"));
                checked += 1;
            }
        }
    }
    assert!(checked > 200, "{checked}");
}

#[test]
fn t295d_a_planted_mistake_is_caught() {
    let cs = Case {
        m: 256,
        n: 256,
        k: 32,
        ta: false,
        tb: false,
        by: 0,
        bx: 0,
    };
    assert!(check_slice(V8S, cs, 0, true).is_err());
    assert!(check_slice(V8NS, cs, 16, true).is_err());
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

#[test]
fn t295e_ldsm_reads_are_conflict_free_with_48_byte_rows() {
    // ldsm.x4: four 8×8 matrices, one 16-byte row address per lane; each phase of 8 lanes = one matrix. With 48-byte
    // rows the 8 rows of a matrix cover 8 × 48 = 384 bytes in 8 distinct 16-byte groups of the 128-byte line.
    let row = 48u32;
    // A (row-major 16×16): lanes 0..15 rows 0..15 at column 0, lanes 16..31 rows 0..15 at column 8 halves (16 B)
    let a = |l: u32| row * (l & 15) + (l & 16);
    // B (column-major): lanes give the rows of the stored [c][p] matrix
    let b = |l: u32| row * ((l & 7) | ((l & 16) >> 1)) + 2 * (l & 8);
    for f in [&a as &dyn Fn(u32) -> u32, &b] {
        for phase in 0..4 {
            let addrs: Vec<u32> = (8 * phase..8 * phase + 8).map(f).collect();
            assert_eq!(wavefronts(&addrs, 16), 1, "phase {phase}");
        }
    }
    // the dense 32-byte rows (no padding) would conflict: the check can fail
    let dense = |l: u32| 32 * (l & 15) + (l & 16);
    let addrs: Vec<u32> = (0..8).map(dense).collect();
    assert_eq!(wavefronts(&addrs, 16), 2);
}

// ---------------------------------------------------------------------------------------------------------------
// The binaries (G1, G2) and the command-processor model (G3).
// ---------------------------------------------------------------------------------------------------------------

fn count(t: &str, what: &str) -> usize {
    t.lines().filter(|l| l.contains(what)).count()
}

/// The lines of the K loop: from the block holding the first `ldsm` to the block barrier that ends the slice.
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

#[test]
fn t295a_binaries_are_the_recorded_builds_g1() {
    for (bin, sha, listing, regs, shared, blocks, threads) in [
        (
            MM_TC_V8,
            v8::MM_TC_V8_SHA256,
            LISTING_V8,
            v8::REGISTERS,
            v8::SHARED,
            v8::BLOCKS_PER_SM,
            v8::BLOCK[0],
        ),
        (
            MM_TC_V8N,
            v8::MM_TC_V8N_SHA256,
            LISTING_V8N,
            v8::REGISTERS_N,
            v8::SHARED_N,
            v8::BLOCKS_PER_SM_N,
            v8::BLOCK_N[0],
        ),
    ] {
        let hex: String = nvidia_ga104::sha256::digest(bin)
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        assert_eq!(hex, sha);
        assert_eq!(bin.len() % 16, 0);
        assert!(listing.contains("Spills to mem: 0") && listing.contains("SLM size: 0"));
        assert!(listing.contains(&format!("Num GPRs: {regs}")));
        assert!(listing.contains(&format!("Shared size: {}", shared.bytes)));
        assert!(listing.contains("c[0x0][+0x700]"), "push constants");
        let k = Kernel {
            name: "mm_tc_v8",
            threads,
            registers: regs,
            shared_bytes: shared.bytes,
        };
        for r in [Reserve::None, Reserve::OneKib] {
            assert_eq!(
                blocks_per_sm(&SM86, &k, 64 << 10, r).unwrap().blocks,
                blocks
            );
        }
    }
    assert_eq!(v8::SHARED.bytes, 24_576);
    assert_eq!(v8::SHARED_N.bytes, 18_432);
    assert_eq!(v8::SHARED.bytes, MM_TC_V8_NEEDS.shared_bytes);
    assert_eq!(v8::SHARED_N.bytes, MM_TC_V8N_NEEDS.shared_bytes);
    // the source is one file: v8n differs only by the V8N define
    assert!(SRC.contains("#ifdef V8N"));
}

#[test]
fn t295b_the_k_loop_reads_only_with_ldsm_g2() {
    // (listing, register copies counted in the loop: session 279, docs/C-168_V8_PLAN.md §7)
    for (listing, movs) in [(LISTING_V8, 148), (LISTING_V8N, 41)] {
        let body = k_loop(listing);
        let txt = body.join("\n");
        assert_eq!(
            count(&txt, "hmma.m16n8k16.f32"),
            16,
            "16 hmma per warp and slice"
        );
        assert_eq!(count(&txt, "ldsm.16.m8n8.x4"), 4, "A: 4 ldsm.x4");
        assert_eq!(
            count(&txt, "ldsm.16.m8n8.x2"),
            4,
            "B: 4 ldsm.x2 (16x8 fragments)"
        );
        assert_eq!(
            count(&txt, "ld.shared"),
            0,
            "no plain shared loads in the loop"
        );
        assert_eq!(count(&txt, "bar.sync"), 1, "one barrier per slice");
        assert_eq!(count(&txt, " = mov "), movs, "register copies in the loop");
        assert_eq!(
            count(listing, "hmma"),
            16,
            "the loop is the only place with hmma"
        );
        assert_eq!(count(listing, "movm"), 0, "no movm");
        assert_eq!(count(listing, "ldsm"), 8, "the reads are the loop's only");
        assert_eq!(
            count(listing, "bar.sync"),
            2,
            "one before the loop, one in it"
        );
    }
}

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

fn run_at(
    prog: &Prog,
    p: &Product<'_>,
    grid: [u32; 3],
    block: [u32; 3],
    da: u64,
) -> Result<Vec<u32>, gpu_core::error::GpuError> {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(prog.code, prog.run, prog.needs);
    cp.mem.write(PROG, prog.code);
    let mut ar = args(p);
    ar.a += da;
    cp.mem.write(CBUF, &ar.cbuf().unwrap());
    cp.mem.write_f32s(A + da, &p.a[..p.m * p.k]);
    cp.mem.write_f32s(B, &p.b[..p.k * p.n]);
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

#[test]
fn t295f_same_bits_as_mm_tc_and_the_same_refusals_g3() {
    let base = Prog {
        code: MM_TC,
        run: mm_tc_on_cpu,
        needs: MM_TC_NEEDS,
        registers: tc::REGISTERS,
        shared: tc::SHARED,
    };
    let p8 = Prog {
        code: MM_TC_V8,
        run: mm_tc_v8_on_cpu,
        needs: MM_TC_V8_NEEDS,
        registers: v8::REGISTERS,
        shared: v8::SHARED,
    };
    let p8n = Prog {
        code: MM_TC_V8N,
        run: mm_tc_v8n_on_cpu,
        needs: MM_TC_V8N_NEEDS,
        registers: v8::REGISTERS_N,
        shared: v8::SHARED_N,
    };
    let mut r = rng(0x168_0003);
    let mut sets = 0;
    for &(m, k, n) in &[
        (128usize, 16usize, 128usize),
        (132, 36, 68),
        (4, 4, 4),
        (260, 48, 196),
        (8, 200, 132),
        (192, 64, 256),
    ] {
        for flags in 0..8u32 {
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
            assert!(v8::fits(&ar));
            let want = run_at(&base, &p, ar.grid(), tc::BLOCK, 0).unwrap();
            let g8 = run_at(&p8, &p, v8::grid(&ar), v8::BLOCK, 0).unwrap();
            let g8n = run_at(&p8n, &p, v8::grid_n(&ar), v8::BLOCK_N, 0).unwrap();
            assert_eq!(g8, want, "v8 {m}x{k}x{n} flags {flags}");
            assert_eq!(g8n, want, "v8n {m}x{k}x{n} flags {flags}");
            // the grids: one block per 128×128 / 128×64 piece of C
            assert_eq!(
                v8::grid(&ar),
                [n.div_ceil(128) as u32, m.div_ceil(128) as u32, 1]
            );
            assert_eq!(
                v8::grid_n(&ar),
                [n.div_ceil(64) as u32, m.div_ceil(128) as u32, 1]
            );
            // refusals: A not 16-byte aligned, the other program's block
            assert!(run_at(&p8, &p, v8::grid(&ar), v8::BLOCK, 4).is_err());
            assert!(run_at(&p8n, &p, v8::grid_n(&ar), v8::BLOCK_N, 4).is_err());
            assert!(run_at(&p8, &p, v8::grid(&ar), v8::BLOCK_N, 0).is_err());
            assert!(run_at(&p8n, &p, v8::grid_n(&ar), v8::BLOCK, 0).is_err());
            sets += 1;
        }
    }
    // a grid too small for C is refused, not half computed
    let a = vec![0.5f32; 256 * 16];
    let b = vec![0.25f32; 16 * 256];
    let p = Product {
        a: &a,
        b: &b,
        bias: None,
        m: 256,
        n: 256,
        k: 16,
        flags: 0,
    };
    assert!(run_at(&p8, &p, [2, 1, 1], v8::BLOCK, 0).is_err());
    assert!(run_at(&p8n, &p, [2, 2, 1], v8::BLOCK_N, 0).is_err());
    // unaligned rows: fits says no
    let mut ar = args(&p);
    ar.k = 18;
    assert!(!v8::fits(&ar));
    assert_eq!(sets, 48);
}
