//! T299 (C-168 point 4, step 2, `docs/C-168_ATTN_FWD_TC_PLAN.md` §4 H3 and §9), SIMULARE: `attn_fwd_tc`
//! (`tools/nak/attn_fwd_tc.comp`, the attention forward on the tensor cores, second NAK tool, Mesa 26.3.1).
//! The prediction H3 was written and pushed BEFORE this code (`9743c50`):
//!   the emulation of the program = the CPU model bit for bit on the edges (d < 64 refused, T = 64 / 128 / 512, B, H,
//!   the paired row tiles); against `attn_fwd_tile3` (F32) on data of #513's size: O relative error (norm) ≤ 1·10⁻³,
//!   P ≤ 2·10⁻³; above the diagonal tile of P nothing written.
//!
//! How: `emulate` runs the GLSL literally — block by block, warp by warp, thread by thread, the barriers as phases,
//! shared memory as the program's 6 912 words (f16 pairs packed as `packFloat2x16`, f32 tiles in rows of 68), every
//! `coopMatLoad` / `coopMatStore` with its offset, stride and layout, every global vec4 checked aligned and inside its
//! buffer, and every word read and written between two barriers checked for a race. The products go through the same
//! tensor-core adder as the model (`mma16_hw`, `GA104_OS335`), so what the comparison checks is the program's data
//! movement and its scalar order, not the board's adder (that is a hypothesis fitted on os-335). Nothing here runs on
//! the RTX 3070 Ti.

// The emulation indexes per-thread arrays by the thread id, as the GLSL does (clarity over iterators).
#![allow(clippy::needless_range_loop)]

use gpu_compute::attn_fwd_tc::{self as m, FwdVariant, LOG2E, NEG, rel_norm};
use gpu_compute::layer::{AttnShape, attn_fwd_tile_gpu, ex2_ftz};
use gpu_compute::math::{add_f32_ftz, ftz};
use gpu_compute::mma::{GA104_OS335, f16_to_f32, f32_to_f16, mma16_hw};
use gpu_sim::cmdproc::{ATTN_FWD_TC_NEEDS, CommandProcessor, attn_fwd_tc_on_cpu};
use nvidia_ga104::attn_back as ab;
use nvidia_ga104::attn_fwd::Args;
use nvidia_ga104::attn_fwd_tc::{self as tc, ATTN_FWD_TC};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_rm570::regs::clc7c0 as c;

const ZW: usize = 0;
const XW: usize = 2304;
const YW: usize = 4608;
const RW: usize = 36;
const SW: usize = 68;
const WORDS: usize = 6912;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed | 1;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

struct Case {
    s: AttnShape,
    qkv: Vec<f32>,
}

impl Case {
    fn new(seed: u64, batch: usize, t: usize, heads: usize, amp: f32) -> Case {
        let s = AttnShape {
            batch,
            t,
            c: 64 * heads,
            heads,
        };
        let mut r = rng(seed);
        let qkv = (0..s.sizes().unwrap().0).map(|_| r() * amp).collect();
        Case { s, qkv }
    }
    fn scale(&self) -> f32 {
        ab::Args::scale_for(64)
    }
    fn np(&self) -> usize {
        self.s.sizes().unwrap().2
    }
    fn no(&self) -> usize {
        self.s.sizes().unwrap().1
    }
}

// ------------------------------------------------------------------ the GLSL, emulated literally

#[derive(Clone, Copy, PartialEq, Debug)]
enum Plant {
    None,
    /// The barrier between P's phase and `store_vt` left out (Vᵀ overwrites S while it is read).
    NoBarrierBeforeVt,
    /// V stored by rows (not turned): Vt[j][x] instead of Vt[x][j].
    VtUnturned,
    /// The second row tile of each pair forgotten.
    NoSecondTile,
    /// The barrier after the K store of pass 2 left out.
    NoBarrierAfterK2,
}

/// Shared memory with a race check: between two barriers no word may be both read and written.
struct Shared {
    w: Vec<u32>,
    flags: Vec<u8>,
    races: usize,
}

impl Shared {
    fn new() -> Shared {
        Shared {
            w: vec![u32::MAX; WORDS],
            flags: vec![0; WORDS],
            races: 0,
        }
    }
    fn rd(&mut self, i: usize) -> u32 {
        if self.flags[i] & 2 != 0 {
            self.races += 1;
        }
        self.flags[i] |= 1;
        self.w[i]
    }
    fn wr(&mut self, i: usize, v: u32) {
        if self.flags[i] & 1 != 0 {
            self.races += 1;
        }
        self.flags[i] |= 2;
        self.w[i] = v;
    }
    fn barrier(&mut self) {
        self.flags.iter_mut().for_each(|f| *f = 0);
    }
    fn half(&mut self, h: usize) -> u16 {
        let w = self.rd(h / 2);
        if h.is_multiple_of(2) {
            w as u16
        } else {
            (w >> 16) as u16
        }
    }
}

fn pack(a: f32, b: f32) -> u32 {
    f32_to_f16(a) as u32 | (f32_to_f16(b) as u32) << 16
}

/// Global memory as vec4: index checked inside the buffer (a vec4 index is aligned by construction).
fn ld4(buf: &[f32], i4: usize) -> [f32; 4] {
    assert!(4 * i4 + 4 <= buf.len(), "vec4 load past the buffer");
    [
        buf[4 * i4],
        buf[4 * i4 + 1],
        buf[4 * i4 + 2],
        buf[4 * i4 + 3],
    ]
}

fn st4(buf: &mut [f32], i4: usize, v: [f32; 4]) {
    assert!(4 * i4 + 4 <= buf.len(), "vec4 store past the buffer");
    buf[4 * i4..4 * i4 + 4].copy_from_slice(&v);
}

fn mul_ftz(a: f32, b: f32) -> f32 {
    ftz(ftz(a) * ftz(b))
}

fn ex2d(a: f32, b: f32) -> f32 {
    ex2_ftz(mul_ftz(add_f32_ftz(a, -b), f32::from_bits(LOG2E)))
}

fn fmax(a: f32, b: f32) -> f32 {
    ftz(a).max(ftz(b))
}

/// `coopMatLoad` of an A 16 × 16 f16 matrix, row-major, at word `off`, stride `RW` words → 16 × 16 halves.
fn load_a(sm: &mut Shared, off: usize) -> [u16; 256] {
    let mut a = [0u16; 256];
    for mm in 0..16 {
        for k in 0..16 {
            a[mm * 16 + k] = sm.half(2 * off + mm * 2 * RW + k);
        }
    }
    a
}

/// `coopMatLoad` of a B 16 × 8 f16 matrix, column-major, at word `off`, stride `RW` words → 16 × 16 (columns 8.. 0).
fn load_b(sm: &mut Shared, off: usize) -> [u16; 256] {
    let mut b = [0u16; 256];
    for n in 0..8 {
        for k in 0..16 {
            b[k * 16 + n] = sm.half(2 * off + n * 2 * RW + k);
        }
    }
    b
}

/// One `coopMatMulAdd` 16 × 8 × 16 (the accumulator kept 16 × 16, columns 8.. stay 0).
fn mma(a: &[u16; 256], b: &[u16; 256], acc: &[f32; 256]) -> [f32; 256] {
    mma16_hw(a, b, acc, GA104_OS335).expect("a non-finite f16 operand reached a product")
}

/// `coopMatStore` of a 16 × 8 f32 accumulator, row-major, at word `off`, stride `SW`.
fn store_acc(sm: &mut Shared, off: usize, acc: &[f32; 256]) {
    for mm in 0..16 {
        for n in 0..8 {
            sm.wr(off + mm * SW + n, acc[mm * 16 + n].to_bits());
        }
    }
}

struct Ctx<'a> {
    k: &'a Case,
}

impl Ctx<'_> {
    fn ld4(&self) -> usize {
        (3 * self.k.s.c) >> 2
    }
    fn load_rows(&self, row0: usize, off: usize, lid: usize) -> [[f32; 4]; 4] {
        let mut v = [[0.0; 4]; 4];
        for s in 0..4 {
            let e = lid + 256 * s;
            v[s] = ld4(
                &self.k.qkv,
                (row0 + (e >> 4)) * self.ld4() + ((off + ((e & 15) << 2)) >> 2),
            );
        }
        v
    }
    fn v_unit(u: usize) -> (usize, usize) {
        let l = u & 31;
        let wv = u >> 5;
        (4 * ((l & 3) + 4 * (wv & 3)), 2 * ((l >> 2) + 8 * (wv >> 2)))
    }
    fn load_v(&self, row0: usize, h: usize, lid: usize) -> [[f32; 4]; 4] {
        let off = 2 * self.k.s.c + h * 64;
        let mut v = [[0.0; 4]; 4];
        for s in 0..2 {
            let (x, j) = Self::v_unit(lid + 256 * s);
            v[2 * s] = ld4(&self.k.qkv, (row0 + j) * self.ld4() + ((off + x) >> 2));
            v[2 * s + 1] = ld4(&self.k.qkv, (row0 + j + 1) * self.ld4() + ((off + x) >> 2));
        }
        v
    }
}

fn store_rows(sm: &mut Shared, base: usize, lid: usize, v: &[[f32; 4]; 4]) {
    for s in 0..4 {
        let e = lid + 256 * s;
        let o = base + (e >> 4) * RW + ((e & 15) << 1);
        sm.wr(o, pack(v[s][0], v[s][1]));
        sm.wr(o + 1, pack(v[s][2], v[s][3]));
    }
}

fn store_vt(sm: &mut Shared, lid: usize, v: &[[f32; 4]; 4], plant: Plant) {
    for s in 0..2 {
        let (x, j) = Ctx::v_unit(lid + 256 * s);
        let (a, b) = (v[2 * s], v[2 * s + 1]);
        if plant == Plant::VtUnturned {
            // rows j and j + 1 as rows (columns x .. x + 3), the mistake
            for (rr, src) in [(j, a), (j + 1, b)] {
                let o = ZW + rr * RW + (x >> 1);
                sm.wr(o, pack(src[0], src[1]));
                sm.wr(o + 1, pack(src[2], src[3]));
            }
            continue;
        }
        let o = ZW + x * RW + (j >> 1);
        for i in 0..4 {
            sm.wr(o + i * RW, pack(a[i], b[i]));
        }
    }
}

/// The program on every block of the grid; `p` / `att` hold what the buffers held before the launch. Returns the
/// number of races found between barriers.
fn emulate(k: &Case, plant: Plant, p: &mut [f32], att: &mut [f32]) -> usize {
    let cx = Ctx { k };
    let (t, cc, heads) = (k.s.t, k.s.c, k.s.heads);
    let nt = t / 64;
    let scale = k.scale();
    let mut races = 0;
    for b in 0..k.s.batch {
        for h in 0..heads {
            for bx in 0..nt.div_ceil(2) {
                let mut sm = Shared::new();
                let prow = (b * heads + h) * t;
                let brow = b * t;
                for pr in 0..2 {
                    let rt = if pr == 0 { bx } else { nt - 1 - bx };
                    if pr == 1 && (rt <= bx || plant == Plant::NoSecondTile) {
                        break;
                    }
                    let i0 = rt * 64;
                    sm.barrier();
                    for lid in 0..256 {
                        let vq = cx.load_rows(brow + i0, h * 64, lid);
                        store_rows(&mut sm, ZW, lid, &vq);
                    }
                    sm.barrier();
                    // qa[w][ks]
                    let mut qa = vec![[[0u16; 256]; 4]; 8];
                    for w in 0..8 {
                        let wr = (w >> 1) * 16;
                        for ks in 0..4 {
                            qa[w][ks] = load_a(&mut sm, ZW + wr * RW + 8 * ks);
                        }
                    }
                    // ---- pass 1
                    let mut mrun = [NEG; 256];
                    let mut lrun = [0.0f32; 256];
                    let mut vk: Vec<_> = (0..256)
                        .map(|lid| cx.load_rows(brow, cc + h * 64, lid))
                        .collect();
                    for kt in 0..=rt {
                        for lid in 0..256 {
                            store_rows(&mut sm, YW, lid, &vk[lid]);
                        }
                        sm.barrier();
                        for lid in 0..256 {
                            vk[lid] = cx.load_rows(brow + (kt + 1).min(rt) * 64, cc + h * 64, lid);
                        }
                        let diag = kt == rt;
                        let accs = s_products(&mut sm, &qa, YW, diag);
                        sm.barrier();
                        store_s(&mut sm, &accs);
                        sm.barrier();
                        for lid in 0..256 {
                            let (r, q) = (lid >> 2, lid & 3);
                            let lim = if diag { r } else { 63 };
                            let mut xs = [0.0f32; 16];
                            let mut tmax = NEG;
                            for mm in 0..4 {
                                let o = r * SW + 4 * q + 16 * mm;
                                for i in 0..4 {
                                    let sv = f32::from_bits(sm.rd(o + i));
                                    let live = 4 * q + 16 * mm + i <= lim;
                                    let x = mul_ftz(sv, scale);
                                    xs[4 * mm + i] = if live { x } else { NEG };
                                    tmax = fmax(tmax, xs[4 * mm + i]);
                                }
                            }
                            let mn = fmax(mrun[lid], tmax);
                            lrun[lid] = mul_ftz(lrun[lid], ex2d(mrun[lid], mn));
                            for kk in 0..16 {
                                let live = 4 * q + 16 * (kk >> 2) + (kk & 3) <= lim;
                                let ek = ex2d(xs[kk], mn);
                                lrun[lid] = add_f32_ftz(lrun[lid], if live { ek } else { 0.0 });
                            }
                            mrun[lid] = mn;
                        }
                    }
                    // the join (subgroup shuffles: lanes lbase .. lbase + 3 of the same warp)
                    let mut big_m = [0.0f32; 256];
                    let mut inv = [0.0f32; 256];
                    for lid in 0..256 {
                        let lbase = lid & !3;
                        let mq: Vec<f32> = (0..4).map(|kk| mrun[lbase + kk]).collect();
                        let lq: Vec<f32> = (0..4).map(|kk| lrun[lbase + kk]).collect();
                        let mx = fmax(fmax(mq[0], mq[1]), fmax(mq[2], mq[3]));
                        let mut l = 0.0f32;
                        for kk in 0..4 {
                            l = add_f32_ftz(l, mul_ftz(lq[kk], ex2d(mq[kk], mx)));
                        }
                        big_m[lid] = mx;
                        inv[lid] = ftz(1.0 / l);
                    }
                    // ---- pass 2
                    let mut acco = vec![[[0.0f32; 256]; 4]; 8];
                    let mut vv = vec![[[0.0f32; 4]; 4]; 256];
                    for lid in 0..256 {
                        vk[lid] = cx.load_rows(brow, cc + h * 64, lid);
                    }
                    sm.barrier();
                    for kt in 0..=rt {
                        for lid in 0..256 {
                            store_rows(&mut sm, XW, lid, &vk[lid]);
                        }
                        if plant != Plant::NoBarrierAfterK2 {
                            sm.barrier();
                        }
                        let diag = kt == rt;
                        let accs = s_products(&mut sm, &qa, XW, diag);
                        sm.barrier();
                        for lid in 0..256 {
                            vv[lid] = cx.load_v(brow + kt * 64, h, lid);
                        }
                        store_s(&mut sm, &accs);
                        sm.barrier();
                        for lid in 0..256 {
                            let (r, q) = (lid >> 2, lid & 3);
                            let lim = if diag { r } else { 63 };
                            for mm in 0..4 {
                                let col = 4 * q + 16 * mm;
                                let o = r * SW + col;
                                let mut pv = [0.0f32; 4];
                                for i in 0..4 {
                                    let sv = f32::from_bits(sm.rd(o + i));
                                    let live = col + i <= lim;
                                    let x = mul_ftz(sv, scale);
                                    let pe = mul_ftz(ex2d(x, big_m[lid]), inv[lid]);
                                    pv[i] = if live { pe } else { 0.0 };
                                }
                                let gi = (prow + i0 + r) * t + kt * 64 + col;
                                assert_eq!(gi % 4, 0);
                                st4(p, gi >> 2, pv);
                                let po = YW + r * RW + (col >> 1);
                                sm.wr(po, pack(pv[0], pv[1]));
                                sm.wr(po + 1, pack(pv[2], pv[3]));
                            }
                        }
                        if plant != Plant::NoBarrierBeforeVt {
                            sm.barrier();
                        }
                        for lid in 0..256 {
                            store_vt(&mut sm, lid, &vv[lid], plant);
                        }
                        for lid in 0..256 {
                            vk[lid] = cx.load_rows(brow + (kt + 1).min(rt) * 64, cc + h * 64, lid);
                        }
                        sm.barrier();
                        for w in 0..8 {
                            let (wr, wc) = ((w >> 1) * 16, (w & 1) * 32);
                            let k1 = if diag { wr + 16 } else { 64 };
                            for kk in (0..64).step_by(16) {
                                if kk >= k1 {
                                    break;
                                }
                                let fp = load_a(&mut sm, YW + wr * RW + (kk >> 1));
                                for jj in 0..4 {
                                    let fv = load_b(&mut sm, ZW + (wc + 8 * jj) * RW + (kk >> 1));
                                    acco[w][jj] = mma(&fp, &fv, &acco[w][jj]);
                                }
                            }
                        }
                    }
                    sm.barrier();
                    store_s(&mut sm, &acco);
                    sm.barrier();
                    for lid in 0..256 {
                        for s in 0..4 {
                            let e = lid + 256 * s;
                            let rr = e >> 4;
                            let x = (e & 15) << 2;
                            let o = rr * SW + x;
                            let ov = [0, 1, 2, 3].map(|i| f32::from_bits(sm.rd(o + i)));
                            let gi = (brow + i0 + rr) * cc + h * 64 + x;
                            assert_eq!(gi % 4, 0);
                            st4(att, gi >> 2, ov);
                        }
                    }
                }
                races += sm.races;
            }
        }
    }
    races
}

/// S = Q·Kᵀ of the 8 warps from the K tile at word `kw`: acc[w][jj] (16 × 8 each).
fn s_products(
    sm: &mut Shared,
    qa: &[[[u16; 256]; 4]],
    kw: usize,
    diag: bool,
) -> Vec<[[f32; 256]; 4]> {
    let mut accs = vec![[[0.0f32; 256]; 4]; 8];
    for w in 0..8 {
        let (wr, wc) = ((w >> 1) * 16, (w & 1) * 32);
        if diag && wc >= wr + 16 {
            continue;
        }
        for ks in 0..4 {
            for jj in 0..4 {
                let fk = load_b(sm, kw + (wc + 8 * jj) * RW + 8 * ks);
                accs[w][jj] = mma(&qa[w][ks], &fk, &accs[w][jj]);
            }
        }
    }
    accs
}

fn store_s(sm: &mut Shared, accs: &[[[f32; 256]; 4]]) {
    for w in 0..8 {
        let (wr, wc) = ((w >> 1) * 16, (w & 1) * 32);
        for jj in 0..4 {
            store_acc(sm, wr * SW + wc + 8 * jj, &accs[w][jj]);
        }
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The emulation with NaN in P before the launch; (P, att, races).
fn run_emulation(k: &Case, plant: Plant) -> (Vec<f32>, Vec<f32>, usize) {
    let mut p = vec![f32::NAN; k.np()];
    let mut att = vec![f32::NAN; k.no()];
    let races = emulate(k, plant, &mut p, &mut att);
    (p, att, races)
}

/// The edges of H3: T = 64 (one tile, the pair is a single tile), 128 (one pair), 192 (odd: the middle block does one
/// tile), 320, B and H > 1; and the larger amplitude (sharper softmax, the f16 range of S·scale).
fn edge_cases() -> Vec<(&'static str, Case)> {
    vec![
        ("T64 B1 H1", Case::new(1, 1, 64, 1, 1.0)),
        ("T128 B2 H2", Case::new(2, 2, 128, 2, 1.0)),
        ("T192 B1 H2 (odd)", Case::new(3, 1, 192, 2, 1.0)),
        ("T320 B2 H1 (odd, 3 blocks)", Case::new(4, 2, 320, 1, 1.0)),
        ("T128 sharp (amp 4)", Case::new(5, 1, 128, 1, 4.0)),
        ("T256 very sharp (amp 12)", Case::new(6, 1, 256, 1, 12.0)),
    ]
}

/// H3, first half: the emulation = the CPU model bit for bit (P where written, att), no race, P above the diagonal
/// tile untouched (still NaN).
#[test]
fn t299_emulation_equals_model_bit_for_bit() {
    for (name, k) in edge_cases() {
        let (p, att, races) = run_emulation(&k, Plant::None);
        assert_eq!(races, 0, "{name}: a race between barriers");
        let (mp, ma) = m::fwd_model(&k.s, &k.qkv, k.scale(), f32::NAN).unwrap();
        assert_eq!(bits(&p), bits(&mp), "{name}: P");
        assert_eq!(bits(&att), bits(&ma), "{name}: att");
        let t = k.s.t;
        let mut untouched = 0usize;
        for row in 0..k.s.batch * k.s.heads * t {
            let i = row % t;
            let end = (i / 64 + 1) * 64;
            assert!(p[row * t + end..(row + 1) * t].iter().all(|x| x.is_nan()));
            untouched += t - end;
            assert!(p[row * t + i + 1..row * t + end].iter().all(|&x| x == 0.0));
        }
        if t > 64 {
            assert!(untouched > 0);
        }
        println!(
            "T299 {name}: emulation = model bit for bit, 0 races, {untouched} P values above the diagonal tile not written"
        );
    }
}

/// H3 on #513's head shape (T 512, d 64): the emulation = the model for the first and the last head (all 4 blocks,
/// all 8 row tiles), B 1, H 12 (only heads 0 and 11 emulated, to keep the test short).
#[test]
fn t299_emulation_equals_model_at_513() {
    let full = Case::new(513, 1, 512, 12, 1.0);
    for h in [0usize, 11] {
        // the head alone as a 1-head case (the program's arithmetic of a head does not depend on the others)
        let mut one = Case::new(0, 1, 512, 1, 1.0);
        for r in 0..512 {
            for part in 0..3 {
                for e in 0..64 {
                    one.qkv[r * 192 + part * 64 + e] =
                        full.qkv[r * 3 * 768 + part * 768 + h * 64 + e];
                }
            }
        }
        let (p, att, races) = run_emulation(&one, Plant::None);
        assert_eq!(races, 0);
        let (mp, ma) = m::fwd_model(&full.s, &full.qkv, full.scale(), f32::NAN)
            .map(|(p, a)| {
                let pp = p[h * 512 * 512..(h + 1) * 512 * 512].to_vec();
                let aa: Vec<f32> = (0..512)
                    .flat_map(|r| a[r * 768 + h * 64..r * 768 + h * 64 + 64].to_vec())
                    .collect();
                (pp, aa)
            })
            .unwrap();
        assert_eq!(bits(&p), bits(&mp), "head {h}: P");
        assert_eq!(bits(&att), bits(&ma), "head {h}: att");
    }
}

/// The comparisons are not blind: every planted mistake in the emulation (a missing barrier → a race; Vᵀ not
/// turned; the second tile of a pair forgotten) and in the model (the join order, no mask in pass 2, P truncated)
/// is caught.
#[test]
fn t299_planted_mistakes_are_caught() {
    let k = Case::new(7, 1, 192, 1, 1.0);
    let (mp, ma) = m::fwd_model(&k.s, &k.qkv, k.scale(), f32::NAN).unwrap();
    for plant in [Plant::NoBarrierBeforeVt, Plant::NoBarrierAfterK2] {
        let (_, _, races) = run_emulation(&k, plant);
        assert!(races > 0, "{plant:?}: no race found");
    }
    for plant in [Plant::VtUnturned, Plant::NoSecondTile] {
        let (p, att, races) = run_emulation(&k, plant);
        assert_eq!(races, 0);
        assert!(
            bits(&p) != bits(&mp) || bits(&att) != bits(&ma),
            "{plant:?} not caught"
        );
    }
    let sharp = Case::new(8, 1, 256, 2, 3.0);
    let (sp, sa) = m::fwd_model(&sharp.s, &sharp.qkv, sharp.scale(), f32::NAN).unwrap();
    for v in [
        FwdVariant {
            join_reversed: true,
            ..FwdVariant::PROGRAM
        },
        FwdVariant {
            no_mask_pass2: true,
            ..FwdVariant::PROGRAM
        },
        FwdVariant {
            p_truncated: true,
            ..FwdVariant::PROGRAM
        },
    ] {
        let (vp, va) = m::fwd_model_with(&sharp.s, &sharp.qkv, sharp.scale(), f32::NAN, v).unwrap();
        assert!(
            bits(&vp) != bits(&sp) || bits(&va) != bits(&sa),
            "{v:?} not caught"
        );
    }
}

// ------------------------------------------------------------------ the margin against attn_fwd_tile3 (F32)

/// H3, second half: against `attn_fwd_tile3`'s model (F32, = `attn_fwd_tile_gpu`), on data of #513's size
/// (B 1, T 512, C 768, H 12): O relative error (norm) ≤ 1·10⁻³, P ≤ 2·10⁻³ (P compared up to the diagonal).
#[test]
fn t299_margin_against_tile3_at_513() {
    for (name, amp, seed) in [("amp 1", 1.0f32, 5130u64), ("amp 2", 2.0, 5131)] {
        let k = Case::new(seed, 1, 512, 12, amp);
        let (tp, ta) = m::fwd_model(&k.s, &k.qkv, k.scale(), 0.0).unwrap();
        let (fp, fa) = attn_fwd_tile_gpu(&k.s, &k.qkv, k.scale()).unwrap();
        let t = k.s.t;
        let eo = rel_norm(&ta, &fa, |_| true).unwrap();
        let ep = rel_norm(&tp, &fp, |i| i % t <= (i / t) % t).unwrap();
        // the worst single P value, in absolute terms (P ≤ 1)
        let worst_p = tp
            .iter()
            .zip(&fp)
            .enumerate()
            .filter(|(i, _)| i % t <= (i / t) % t)
            .map(|(_, (&a, &b))| (a - b).abs())
            .fold(0.0f32, f32::max);
        println!("T299 #513 {name}: O rel {eo:.3e}, P rel {ep:.3e}, P worst abs {worst_p:.3e}");
        assert!(eo <= 1e-3, "{name}: O {eo}");
        assert!(ep <= 2e-3, "{name}: P {ep}");
        // both write exact 0 above the diagonal inside the diagonal tile
        for i in 0..t {
            for j in i + 1..(i / 64 + 1) * 64 {
                assert_eq!(tp[i * t + j], 0.0);
                assert_eq!(fp[i * t + j], 0.0);
            }
        }
    }
}

/// The backward programs ignore P above the diagonal tile: the tensor-core backward (`attn_tc::model`, the model of
/// `attn_back_tile_tc16_v4`) and mode 0 (`attn_tc::back0_model`) give the same bits with NaN there (what `attn_fwd_tc`
/// leaves) as with 0 (what `attn_fwd_tile3` wrote).
#[test]
fn t299_backward_ignores_the_unwritten_tiles() {
    let k = Case::new(9, 2, 192, 2, 1.0);
    let (p_nan, _) = m::fwd_model(&k.s, &k.qkv, k.scale(), f32::NAN).unwrap();
    let (p_zero, _) = m::fwd_model(&k.s, &k.qkv, k.scale(), 0.0).unwrap();
    assert!(p_nan.iter().any(|x| x.is_nan()));
    let mut r = rng(99);
    let dout: Vec<f32> = (0..k.no()).map(|_| r()).collect();
    let ds0 = gpu_compute::attn_tc::back0_model(&k.s, &k.qkv, &p_zero, &dout, k.scale()).unwrap();
    let ds1 = gpu_compute::attn_tc::back0_model(&k.s, &k.qkv, &p_nan, &dout, k.scale()).unwrap();
    let t = k.s.t;
    let below = |i: usize| i % t <= (i / t) % t;
    let pick = |v: &[f32]| -> Vec<u32> {
        v.iter()
            .enumerate()
            .filter(|(i, _)| below(*i))
            .map(|(_, x)| x.to_bits())
            .collect()
    };
    assert_eq!(pick(&ds0), pick(&ds1), "mode 0: dS below the diagonal");
    // dS above the diagonal: NaN in, whatever mode 0 writes there; the tc backward gets the tile3-style dS (0 above)
    let mut ds = ds0.clone();
    for (i, x) in ds.iter_mut().enumerate() {
        if !below(i) {
            *x = 0.0;
        }
    }
    let g0 = gpu_compute::attn_tc::model(&k.s, &k.qkv, &p_zero, &dout, &ds).unwrap();
    let g1 = gpu_compute::attn_tc::model(&k.s, &k.qkv, &p_nan, &dout, &ds).unwrap();
    assert_eq!(
        bits(&g0),
        bits(&g1),
        "the tc backward reads P above the diagonal tile"
    );
}

// ------------------------------------------------------------------ through the command-processor model

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
const QKV: u64 = 0x100_0000;
const P: u64 = 0x400_0000;
const ATT: u64 = 0x800_0000;
const SENTINEL: f32 = -12345.0;

fn args(k: &Case, shift: u64) -> Args {
    Args {
        qkv: QKV + shift,
        p: P,
        att: ATT,
        batch: k.s.batch as u32,
        t: k.s.t as u32,
        c: k.s.c as u32,
        heads: k.s.heads as u32,
        scale: k.scale(),
    }
}

/// One launch through our pushbuffer + QMD with the cbuf `cb`; P filled with NaN before, att with SENTINEL.
fn launch(
    k: &Case,
    a: &Args,
    cb: &[u8],
) -> (Result<(), gpu_core::error::GpuError>, Vec<f32>, Vec<f32>) {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(ATTN_FWD_TC, attn_fwd_tc_on_cpu, ATTN_FWD_TC_NEEDS);
    cp.mem.write(PROG, ATTN_FWD_TC);
    cp.mem.write(CBUF, cb);
    cp.mem.write_f32s(a.qkv, &k.qkv);
    cp.mem.write_f32s(P, &vec![f32::NAN; k.np() + 16]);
    cp.mem.write_f32s(ATT, &vec![SENTINEL; k.no() + 16]);
    cp.mem.write_u32(SEM, 0);
    let q = nvidia_ga104::qmd::ComputeQmd {
        program_va: PROG,
        registers: tc::REGISTERS,
        cbuf0_va: CBUF,
        cbuf0_size: tc::CBUF_BYTES as u32,
        grid: tc::grid(a),
        block: tc::BLOCK,
        release: Some((SEM, 1)),
    };
    cp.mem
        .write(QMD, &q.to_bytes_with_shared(&tc::SHARED).unwrap());
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
    let res = cp.run_gpfifo(&[(e.entry0, e.entry1)]).map(|_| ());
    let p = cp.mem.read_f32s(P, k.np() + 16).unwrap();
    let att = cp.mem.read_f32s(ATT, k.no() + 16).unwrap();
    if res.is_ok() {
        assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1, "semaphore released");
    }
    (res, p, att)
}

/// Through our pushbuffer + QMD, as the OS would launch: the stand-in gives the model's bits (P where written, att),
/// leaves P above the diagonal tile as it was (NaN) and writes nothing past the buffers.
#[test]
fn t299_launch_through_the_command_processor() {
    for (name, k) in edge_cases().into_iter().take(4) {
        let a = args(&k, 0);
        assert!(tc::fits(&a));
        let (res, p, att) = launch(&k, &a, &tc::cbuf(&a).unwrap());
        res.unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let (mp, ma) = m::fwd_model(&k.s, &k.qkv, k.scale(), f32::NAN).unwrap();
        assert_eq!(bits(&p[..k.np()]), bits(&mp), "{name}: P");
        assert_eq!(bits(&att[..k.no()]), bits(&ma), "{name}: att");
        assert!(
            p[k.np()..].iter().all(|x| x.is_nan()),
            "{name}: wrote after P"
        );
        assert!(
            att[k.no()..].iter().all(|&x| x == SENTINEL),
            "{name}: wrote after att"
        );
    }
}

/// The edges the program does not take are refused twice: by `fits` (the host keeps `attn_fwd_tile3`) and by the
/// stand-in (no write) when such a cbuf reaches it anyway: d = 32, T = 96, qkv at an address 8 mod 16, and a Q value
/// outside binary16.
#[test]
fn t299_unfit_shapes_are_refused() {
    // d = 32 (C 64, H 2) and T = 96, cbuf written by hand at 0x700 as the program reads it
    let bad_shapes = [
        AttnShape {
            batch: 1,
            t: 64,
            c: 64,
            heads: 2,
        },
        AttnShape {
            batch: 1,
            t: 96,
            c: 64,
            heads: 1,
        },
    ];
    for s in bad_shapes {
        let mut r = rng(3);
        let k = Case {
            s,
            qkv: (0..s.sizes().unwrap().0).map(|_| r()).collect(),
        };
        let a = args(&k, 0);
        assert!(!tc::fits(&a));
        let mut cb = [0u8; tc::CBUF_BYTES];
        let f = a.cbuf().unwrap();
        cb[0x700..0x72c].copy_from_slice(&f[0x28..0x54]);
        let (res, p, att) = launch(&k, &a, &cb);
        assert!(res.is_err());
        assert!(p.iter().all(|x| x.is_nan()));
        assert!(att.iter().all(|&x| x == SENTINEL));
    }
    // qkv 8 mod 16
    let k = Case::new(11, 1, 64, 1, 1.0);
    let a = args(&k, 8);
    assert!(!tc::fits(&a));
    let mut cb = [0u8; tc::CBUF_BYTES];
    cb[0x700..0x72c].copy_from_slice(&a.cbuf().unwrap()[0x28..0x54]);
    let (res, p, _) = launch(&k, &a, &cb);
    assert!(res.is_err());
    assert!(p.iter().all(|x| x.is_nan()));
    // a K value outside binary16: refused, nothing written
    let mut k = Case::new(12, 1, 128, 1, 1.0);
    k.qkv[64 * 192 + 64 + 5] = 1.0e6;
    let a = args(&k, 0);
    let (res, p, att) = launch(&k, &a, &tc::cbuf(&a).unwrap());
    assert!(res.is_err());
    assert!(p.iter().all(|x| x.is_nan()));
    assert!(att.iter().all(|&x| x == SENTINEL));
}

/// P rounded to f16 is what O uses: the model's att equals a plain f64 P·V over P rounded to f16 and V rounded to
/// f16 within the tensor-core adder's error (a check of the model against its own definition, not of the board).
#[test]
fn t299_att_is_p16_times_v16() {
    let k = Case::new(13, 1, 128, 1, 1.0);
    let (p, att) = m::fwd_model(&k.s, &k.qkv, k.scale(), 0.0).unwrap();
    let mut worst = 0.0f64;
    for i in 0..128 {
        for e in 0..64 {
            let mut s = 0.0f64;
            for j in 0..=i {
                let pv = f16_to_f32(f32_to_f16(p[i * 128 + j])) as f64;
                let vv = f16_to_f32(f32_to_f16(k.qkv[j * 192 + 128 + e])) as f64;
                s += pv * vv;
            }
            worst = worst.max((att[i * 64 + e] as f64 - s).abs());
        }
    }
    println!("T299 att vs f64 P16·V16: worst abs {worst:.3e}");
    assert!(worst < 2e-5, "{worst}");
}
