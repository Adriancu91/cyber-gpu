//! T231 (B1 step 3, `docs/B1_MM_S8.md`), SIMULARE: the int8 tensor-core matmul `mm_s8` on 128×128 tiles
//! (`tools/nak/mm_s8.comp` → `tools/nak/out2/mm_s8.sm86.sass`, second NAK tool, Mesa 26.3.1) launched through our
//! pushbuffer + QMD in the command-processor model; the "program" runs on the CPU (`mm_s8_on_cpu`). Integer sums do
//! not depend on the order, so C is judged **bit for bit** against `gpu_compute::mma_s8::mm_s8_ref`. Nothing here
//! runs on the RTX 3070 Ti.

use gpu_compute::mma_s8::{self as s8, ProductS8, mm_s8_ref, mm_s8_wrap};
use gpu_compute::train::{self, AdamW, Recipe};
use gpu_sim::cmdproc::{CommandProcessor, MM_S8_NEEDS, mm_s8_on_cpu};
use nvidia_ga104::mm_s8::{self as ms8, MM_S8};
use nvidia_ga104::mm_tc as tc;
use nvidia_ga104::mm_train_variants as mv;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_ga104::train_step::Plan;
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
/// What C holds before a launch, so an element nobody wrote cannot pass.
const UNWRITTEN: i32 = 0x7fff_fff7;
const HEAP: u64 = nvidia_ga104::mmu::VA_BASE + 0x100_0000;

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn i8s(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| self.next() as u8 as i8).collect()
    }
    fn bias(&mut self, n: usize) -> Vec<i32> {
        (0..n)
            .map(|_| (self.next() % (1 << 21)) as i32 - (1 << 20))
            .collect()
    }
}

fn args(p: &ProductS8<'_>) -> tc::Args {
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

fn bytes_i8(v: &[i8]) -> Vec<u8> {
    v.iter().map(|x| *x as u8).collect()
}

fn bytes_i32(v: &[i32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// One launch of `mm_s8` on `p` with `grid` / `block`; `Ok(C)` or the processor's refusal. Checks the guard after C.
fn run_with(
    p: &ProductS8<'_>,
    grid: [u32; 3],
    block: [u32; 3],
) -> Result<Vec<i32>, gpu_core::error::GpuError> {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(MM_S8, mm_s8_on_cpu, MM_S8_NEEDS);
    cp.mem.write(PROG, MM_S8);
    cp.mem.write(CBUF, &ms8::cbuf(&args(p)).unwrap());
    cp.mem.write(A, &bytes_i8(&p.a[..p.m * p.k]));
    cp.mem.write(B, &bytes_i8(&p.b[..p.k * p.n]));
    if let Some(bias) = p.bias {
        cp.mem.write(BIAS, &bytes_i32(bias));
    }
    cp.mem
        .write(C, &bytes_i32(&vec![UNWRITTEN; p.m * p.n + 16]));
    cp.mem.write_u32(SEM, 0);
    let q = ComputeQmd {
        program_va: PROG,
        registers: ms8::REGISTERS,
        cbuf0_va: CBUF,
        cbuf0_size: ms8::CBUF_BYTES as u32,
        grid,
        block,
        release: Some((SEM, 1)),
    };
    cp.mem
        .write(QMD, &q.to_bytes_with_shared(&ms8::SHARED).unwrap());
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
    let raw: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &raw);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    let res = cp.run_gpfifo(&[(e.entry0, e.entry1)]);
    let mut out = vec![0u8; 4 * (p.m * p.n + 16)];
    cp.mem.read(C, &mut out).unwrap();
    let got: Vec<i32> = out
        .chunks(4)
        .map(|w| i32::from_le_bytes(w.try_into().unwrap()))
        .collect();
    assert!(
        got[p.m * p.n..].iter().all(|&x| x == UNWRITTEN),
        "wrote after C"
    );
    if res.is_err() {
        assert!(
            got.iter().all(|&x| x == UNWRITTEN),
            "a refused launch wrote C"
        );
    }
    res?;
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    Ok(got[..p.m * p.n].to_vec())
}

fn run(p: &ProductS8<'_>) -> Vec<i32> {
    run_with(p, ms8::grid(&args(p)), ms8::BLOCK).unwrap()
}

fn exact(what: &str, p: &ProductS8<'_>) {
    let want = mm_s8_ref(p).unwrap_or_else(|| panic!("{what}: the oracle refuses (overflow?)"));
    let got = run(p);
    assert!(!got.contains(&UNWRITTEN), "{what}: C not written");
    assert_eq!(got, want, "{what}: mm_s8 != oracle");
}

#[test]
fn t231a_binary_is_the_recorded_build_and_uses_imma() {
    let hex: String = nvidia_ga104::sha256::digest(MM_S8)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(hex, ms8::MM_S8_SHA256);
    assert_eq!(MM_S8.len() % 16, 0, "whole 16-byte instructions");
    let txt = include_str!("../../../tools/nak/out2/mm_s8.sm86.txt");
    let imma = txt.lines().filter(|l| l.contains("imma")).count();
    eprintln!("T231a: imma {imma}");
    assert!(
        imma > 0 && imma.is_multiple_of(16),
        "P2: 2 x 4 accumulators x 2 halves"
    );
    assert!(!txt.contains("hmma"), "no f16 tensor-core instruction");
    assert!(txt.contains("bar"), "no barrier in the dump");
    assert!(txt.contains("c[0x0][0x700]"), "push constants not at 0x700");
    assert!(
        !txt.contains("ldl") && !txt.contains("stl"),
        "register spills"
    );
    const { assert!(ms8::REGISTERS <= 128, "P1") };
    assert_eq!(ms8::SHARED.bytes, MM_S8_NEEDS.shared_bytes);
    assert_eq!(ms8::SHARED.bytes, 16 * 1024, "P3");
}

#[test]
fn t231b_edges_all_flags_bit_exact() {
    let mut g = Lcg(0x9e37_79b9_7f4a_7c15);
    for (m, k, n) in [
        (128usize, 32usize, 128usize),
        (130, 33, 129),
        (5, 1, 3),
        (200, 96, 70),
        (64, 200, 257),
        (257, 48, 16),
    ] {
        let a = g.i8s(m * k);
        let b = g.i8s(k * n);
        let bias = g.bias(n);
        for flags in 0..8u32 {
            let p = ProductS8 {
                a: &a,
                b: &b,
                bias: (flags & tc::BIAS != 0).then_some(&bias[..]),
                m,
                n,
                k,
                flags,
            };
            exact(&format!("{m}x{k}x{n} flags {flags}"), &p);
        }
    }
}

#[test]
fn t231c_extremes_fit_on_the_largest_k() {
    let (m, k, n) = (130usize, 3_072usize, 129usize);
    let neg = vec![-128i8; m * k];
    let negb = vec![-128i8; k * n];
    let pos = vec![127i8; k * n];
    let bias: Vec<i32> = (0..n)
        .map(|i| if i % 2 == 0 { 1 << 20 } else { -(1 << 20) })
        .collect();
    for (b, each) in [(&negb, 3_072 * 16_384i64), (&pos, -3_072 * 16_256i64)] {
        let p = ProductS8 {
            a: &neg,
            b,
            bias: Some(&bias),
            m,
            n,
            k,
            flags: tc::BIAS,
        };
        let want = mm_s8_ref(&p).expect("fits in i32");
        assert_eq!(want[0] as i64, each + (1 << 20));
        exact(&format!("extreme {each}"), &p);
    }
}

/// The 15 matmul shapes of step #513 (the os-416 race list), M and N cut to at most 160 / 288, K whole.
#[test]
fn t231d_the_513_shapes_bit_exact() {
    let (bsz, tl) = (2usize, 512usize);
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let plan = Plan::new_tiles3(train::M513, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile()
        .resident()
        .unwrap();
    let params: Vec<Vec<f32>> = train::M513
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    let forms = mv::forms(&plan.step(Recipe::AdamW513, &opt, 1).unwrap());
    assert_eq!(forms.len(), 15);
    let mut g = Lcg(0x5151_0513_0225_0231);
    for f in &forms {
        let full = tc::Args {
            a: A,
            b: B,
            c: C,
            bias: if f.flags & tc::BIAS != 0 { BIAS } else { 0 },
            m: f.m,
            n: f.n,
            k: f.k,
            flags: f.flags,
        };
        assert!(ms8::cbuf(&full).is_some(), "{f:?}: refused");
        assert_eq!(ms8::grid(&full), [f.n.div_ceil(128), f.m.div_ceil(128), 1]);
        let (m, n, k) = (
            (f.m as usize).min(160),
            (f.n as usize).min(288),
            f.k as usize,
        );
        let a = g.i8s(m * k);
        let b = g.i8s(k * n);
        let bias = g.bias(n);
        let p = ProductS8 {
            a: &a,
            b: &b,
            bias: (f.flags & tc::BIAS != 0).then_some(&bias[..]),
            m,
            n,
            k,
            flags: f.flags,
        };
        exact(&format!("#513 {f:?} cut to {m}x{k}x{n}"), &p);
    }
}

#[test]
fn t231e_refusals_leave_c_untouched() {
    let mut g = Lcg(31);
    let (m, k, n) = (200usize, 64usize, 300usize);
    let a = g.i8s(m * k);
    let b = g.i8s(k * n);
    let p = ProductS8 {
        a: &a,
        b: &b,
        bias: None,
        m,
        n,
        k,
        flags: 0,
    };
    let grid = ms8::grid(&args(&p));
    assert_eq!(grid, [3, 2, 1]);
    for (what, gr, bl) in [
        ("128 threads", grid, [128, 1, 1]),
        ("short grid", [2, 2, 1], ms8::BLOCK),
        ("grid z", [3, 2, 2], ms8::BLOCK),
    ] {
        let e = run_with(&p, gr, bl);
        assert!(e.is_err(), "{what}");
    }
    // Unknown flags / BIAS without an address: the constant buffer is refused before anything is sent.
    assert!(
        ms8::cbuf(&tc::Args {
            flags: 8,
            ..args(&p)
        })
        .is_none()
    );
    assert!(
        ms8::cbuf(&tc::Args {
            flags: tc::BIAS,
            bias: 0,
            ..args(&p)
        })
        .is_none()
    );
    // Overflow: the oracle refuses, the model wraps (HYPOTHESIS) — such data is never sent to a board.
    let ones = vec![127i8; 4 * 4];
    let big = vec![i32::MAX; 4];
    let q = ProductS8 {
        a: &ones,
        b: &ones,
        bias: Some(&big),
        m: 4,
        n: 4,
        k: 4,
        flags: tc::BIAS,
    };
    assert_eq!(mm_s8_ref(&q), None);
    let w = mm_s8_wrap(&q).unwrap();
    assert_eq!(run(&q), w);
}

#[test]
fn t231f_oracle_on_the_probe_shape_is_the_probe() {
    let mut g = Lcg(0x0b1);
    let a = g.i8s(s8::M * s8::K);
    let b = g.i8s(s8::K * s8::N);
    let c0 = [0i32; s8::M * s8::N];
    let p = ProductS8 {
        a: &a,
        b: &b,
        bias: None,
        m: s8::M,
        n: s8::N,
        k: s8::K,
        flags: 0,
    };
    let probe = s8::mma16_s8_ref(&a, &b, &c0).unwrap();
    assert_eq!(mm_s8_ref(&p).unwrap(), probe.to_vec());
    exact("probe shape", &p);
}

/// A literal CPU transcription of `tools/nak/mm_s8.comp` (the same index formulas for the shared slices, the same
/// subgroup pieces and 16×32·32×16 steps, the same epilogue), so a wrong index in the GLSL — not only in the model —
/// shows up as different bits. It emulates the GLSL as written, not what NAK compiled.
// The loops mirror the GLSL ones on purpose (index for index), so clippy's iterator form is not used.
#[allow(clippy::needless_range_loop)]
fn glsl_literal(p: &ProductS8<'_>) -> Vec<i32> {
    const BM: usize = 128;
    const BN: usize = 128;
    const BK: usize = 32;
    const THREADS: usize = 256;
    let (m, n, k) = (p.m, p.n, p.k);
    let ta = p.flags & tc::TA != 0;
    let tb = p.flags & tc::TB != 0;
    let mut c = vec![UNWRITTEN; m * n];
    for gy in 0..m.div_ceil(BM) {
        for gx in 0..n.div_ceil(BN) {
            let (row0, col0) = (gy * BM, gx * BN);
            let mut acc = vec![[[0i32; 256]; 4]; 2 * 8]; // [w*2 + i][j][16x16]
            let mut t = 0;
            while t < k {
                let mut as_ = [0i8; BM * BK];
                let mut bs_ = [0i8; BK * BN];
                for tid in 0..THREADS {
                    for s in 0..(BM * BK) / THREADS {
                        let e = tid + s * THREADS;
                        let r = if ta { e % BM } else { e / BK };
                        let pp = if ta { e / BM } else { e % BK };
                        let (gr, gp) = (row0 + r, t + pp);
                        let mut x = 0i8;
                        if gr < m && gp < k {
                            x = if ta {
                                p.a[gp * m + gr]
                            } else {
                                p.a[gr * k + gp]
                            };
                        }
                        as_[r * BK + pp] = x;
                    }
                    for s in 0..(BK * BN) / THREADS {
                        let e = tid + s * THREADS;
                        let pp = if tb { e % BK } else { e / BN };
                        let cl = if tb { e / BK } else { e % BN };
                        let (gp, gc) = (t + pp, col0 + cl);
                        let mut x = 0i8;
                        if gp < k && gc < n {
                            x = if tb {
                                p.b[gc * k + gp]
                            } else {
                                p.b[gp * n + gc]
                            };
                        }
                        bs_[pp * BN + cl] = x;
                    }
                }
                for w in 0..8 {
                    let (wr, wc) = ((w / 2) * 32, (w % 2) * 64);
                    for i in 0..2 {
                        for j in 0..4 {
                            let a0 = (wr + 16 * i) * BK; // coopMatLoad(fa[i], as_, ., BK)
                            let b0 = wc + 16 * j; // coopMatLoad(fb[j], bs_, ., BN)
                            let d = &mut acc[w * 2 + i][j];
                            for rr in 0..16 {
                                for cc in 0..16 {
                                    let mut s = d[rr * 16 + cc];
                                    for q in 0..32 {
                                        let x = as_[a0 + rr * BK + q] as i32;
                                        let y = bs_[b0 + q * BN + cc] as i32;
                                        s = s.wrapping_add(x * y);
                                    }
                                    d[rr * 16 + cc] = s;
                                }
                            }
                        }
                    }
                }
                t += BK;
            }
            for w in 0..8 {
                let (wr, wc) = ((w / 2) * 32, (w % 2) * 64);
                for i in 0..2 {
                    for j in 0..4 {
                        let cs = &acc[w * 2 + i][j];
                        for e in 0..256 {
                            let gr = row0 + wr + 16 * i + e / 16;
                            let gc = col0 + wc + 16 * j + e % 16;
                            if gr < m && gc < n {
                                let mut v = cs[e];
                                if p.flags & tc::BIAS != 0 {
                                    v = v.wrapping_add(p.bias.unwrap()[gc]);
                                }
                                c[gr * n + gc] = v;
                            }
                        }
                    }
                }
            }
        }
    }
    c
}

#[test]
fn t231g_the_glsl_as_written_gives_the_oracle() {
    let mut g = Lcg(0x6e51);
    for (m, k, n) in [
        (130usize, 33usize, 129usize),
        (5, 1, 3),
        (200, 96, 70),
        (257, 64, 300),
    ] {
        let a = g.i8s(m * k);
        let b = g.i8s(k * n);
        let bias = g.bias(n);
        for flags in 0..8u32 {
            let p = ProductS8 {
                a: &a,
                b: &b,
                bias: (flags & tc::BIAS != 0).then_some(&bias[..]),
                m,
                n,
                k,
                flags,
            };
            let want = mm_s8_ref(&p).unwrap();
            assert_eq!(glsl_literal(&p), want, "{m}x{k}x{n} flags {flags}");
        }
    }
}
