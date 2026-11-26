//! T251 (C-157, session 242, `docs/C-157_ATENTIE_TC.md`): what the causal attention of the mixed #513 step could win
//! on the tensor cores (f16 inputs, f32 sums). The board times are the OS's (MĂSURAT: os-431 `tests/rapoarte/
//! os431-pas.txt`, os-443 through `wave_model::OS443_TC_V4`), every derived time is DEDUS; the precision probe is a
//! processor computation (SIMULARE of the arithmetic, f64 reference). Nothing runs on (or simulates) the board.

use gpu_compute::mm_tc::round_f16;
use gpu_compute::train::{self, AdamW, Recipe};
use nvidia_ga104::attn_tc_study::{self as at, M513_ATTN, PROGRAMS};
use nvidia_ga104::kronos_perf::PAPER_MB_S;
use nvidia_ga104::train_profile::{ParsedProfile, parse_profiles};
use nvidia_ga104::train_step::{LOSS_SCALE_513, MixedLaunch, Plan};

const OS431: &str = include_str!("rapoarte/os431-pas.txt");

fn profile_431() -> ParsedProfile {
    parse_profiles(OS431)
        .into_iter()
        .find(|p| p.context.contains("antrenare #513"))
        .expect("the #513 profile")
}

fn measured_us(name: &str) -> u64 {
    profile_431().program(name).expect(name).us
}

/// FLOP of the 147 products of the mixed #513 list (the same products os-431 ran in F32 on `mm_tv_128x128`).
fn step_product_flop() -> u64 {
    let inputs: Vec<u32> = (0..1024u32).map(|i| (i * 7_919) % 16_384).collect();
    let base = nvidia_ga104::mmu::VA_BASE + (1 << 30);
    let plan = Plan::new_tiles3(train::M513, 2, 512, &inputs, base)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split();
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    plan.step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap()
        .iter()
        .filter_map(|l| match l {
            MixedLaunch::Tc { args, .. } => Some(2 * args.m as u64 * args.n as u64 * args.k as u64),
            _ => None,
        })
        .sum()
}

/// The three attention programs: os-431 total.
#[test]
fn t251a_os431_attention() {
    let us: Vec<u64> = PROGRAMS.iter().map(|p| measured_us(p.name)).collect();
    assert_eq!(us, vec![6_207, 5_182, 9_192]);
    assert_eq!(us.iter().sum::<u64>(), 20_581);
}

/// A1: the work of the 6 products and the rate each program reaches today.
#[test]
fn t251b_a1_rates_today() {
    let f = M513_ATTN;
    let per_layer = 6 * f.product_flop();
    let step = per_layer * f.layers;
    let total_us: u64 = PROGRAMS.iter().map(|p| measured_us(p.name)).sum();
    let mm_f32_us = profile_431().program("mm_tv_128x128").unwrap().us;
    let mm_flop = step_product_flop();
    println!(
        "kt c242 A1 atentia {} MFLOP pe strat, {} MFLOP pe pas; azi {:.2} TFLOP/s; inmultirile F32 ale pasului {:.2} TFLOP/s ({} MFLOP in {mm_f32_us} us)",
        per_layer / 1_000_000,
        step / 1_000_000,
        step as f64 / total_us as f64 / 1e6,
        mm_flop as f64 / mm_f32_us as f64 / 1e6,
        mm_flop / 1_000_000
    );
    for p in PROGRAMS {
        let us = measured_us(p.name);
        let flop = p.products * f.product_flop() * f.layers;
        println!(
            "kt c242 A1 {:<18} {us:>5} us, {} produse, {:.2} TFLOP/s",
            p.name,
            p.products,
            flop as f64 / us as f64 / 1e6
        );
    }
    assert_eq!(per_layer, 2_717_908_992);
    // today's attention is slower per FLOP than the step's F32 products
    assert!(step * mm_f32_us < mm_flop * total_us);
}

/// A2: today's P / S / dS passes in f32 against the measured time.
#[test]
fn t251c_a2_memory_today() {
    let f = M513_ATTN;
    let passes: u64 = PROGRAMS.iter().map(|p| p.matrix_passes).sum();
    let per_layer = passes * f.matrix_pass_bytes(4);
    let (lo, hi) = at::f32_memory_ns(&f);
    let total_us: u64 = PROGRAMS.iter().map(|p| measured_us(p.name)).sum();
    println!(
        "kt c242 A2 {passes} treceri x {:.1} MiB = {:.1} MiB pe strat; memoria F32 a pasului {}-{} us = {}-{} ‰ din {total_us} us",
        f.matrix_pass_bytes(4) as f64 / 1048576.0,
        per_layer as f64 / 1048576.0,
        lo / 1_000,
        hi / 1_000,
        lo / total_us,
        hi / total_us
    );
    assert_eq!(per_layer, 135 << 20);
    assert_eq!(lo, (135u64 << 20) * 12 * 1_000 / PAPER_MB_S);
}

/// A3: the tensor-core estimate against today and against F32 at the step's F32 product rate.
#[test]
fn t251d_a3_tensor_core_estimate() {
    let f = M513_ATTN;
    let total_us: u64 = PROGRAMS.iter().map(|p| measured_us(p.name)).sum();
    let (lo, hi) = at::tc_total_ns(&f);
    let mm_f32_us = profile_431().program("mm_tv_128x128").unwrap().us;
    let f32_rate = step_product_flop() / mm_f32_us; // MFLOP/s
    let f32_best = at::f32_at_rate_ns(&f, f32_rate);
    let board_us = 106_001u64; // os-445, read by T250
    println!(
        "kt c242 A3 pe circuitele speciale {}-{} us (azi {total_us}); castig {}-{} us = {:.1}-{:.1} % din placa mixta; F32 la viteza inmultirilor F32: {} us (castig {} us)",
        lo / 1_000,
        hi / 1_000,
        total_us - hi / 1_000,
        total_us - lo / 1_000,
        (total_us - hi / 1_000) as f64 * 100.0 / board_us as f64,
        (total_us - lo / 1_000) as f64 * 100.0 / board_us as f64,
        f32_best / 1_000,
        total_us - f32_best / 1_000
    );
    for p in PROGRAMS {
        let (l, h) = at::tc_program_ns(&f, &p);
        let us = measured_us(p.name);
        println!(
            "kt c242 A3/A5 {:<18} azi {us:>5} us -> {}-{} us (castig {}-{} us)",
            p.name,
            l / 1_000,
            h / 1_000,
            us - h / 1_000,
            us - l / 1_000
        );
    }
    assert!(lo < hi && hi / 1_000 < total_us);
    assert!(f32_best / 1_000 < total_us);
}

// ---- A4: precision of f16 inputs with f32 sums (processor, f64 reference) ----

struct Rng(u64);
impl Rng {
    fn unit(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Irwin–Hall of 4 uniforms, scaled to standard deviation `sd`.
    fn normal(&mut self, sd: f64) -> f64 {
        let s: f64 = (0..4).map(|_| self.unit()).sum::<f64>() - 2.0;
        s * sd * 3f64.sqrt()
    }
}

/// One head: Q, K, V, dO row-major [T, d].
struct Head {
    t: usize,
    d: usize,
    q: Vec<f64>,
    k: Vec<f64>,
    v: Vec<f64>,
    d_o: Vec<f64>,
}

fn head(t: usize, d: usize, do_sd: f64, seed: u64) -> Head {
    let mut r = Rng(seed);
    let mut g = |sd: f64| (0..t * d).map(|_| r.normal(sd)).collect::<Vec<f64>>();
    Head {
        t,
        d,
        q: g(1.0),
        k: g(1.0),
        v: g(1.0),
        d_o: g(do_sd),
    }
}

/// (O, dQ, dK, dV) of causal attention. `rd` rounds every input of the 6 products (and the stored P, dS); sums and
/// the softmax are in `f64` when `wide`, else in f32. `s` scales dO before the backward products and is divided out.
fn attention(h: &Head, rd: fn(f64) -> f64, wide: bool, s: f64) -> [Vec<f64>; 4] {
    let (t, d) = (h.t, h.d);
    let acc = |x: f64| if wide { x } else { x as f32 as f64 };
    let c = 1.0 / (d as f64).sqrt();
    let q: Vec<f64> = h.q.iter().map(|&x| rd(x)).collect();
    let k: Vec<f64> = h.k.iter().map(|&x| rd(x)).collect();
    let v: Vec<f64> = h.v.iter().map(|&x| rd(x)).collect();
    let dov: Vec<f64> = h.d_o.iter().map(|&x| rd(acc(x * s))).collect();
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).fold(0.0, |z, (&x, &y)| acc(z + acc(x * y)));
    // forward
    let mut p = vec![0.0; t * t];
    for i in 0..t {
        let row: Vec<f64> = (0..=i)
            .map(|j| acc(dot(&q[i * d..][..d], &k[j * d..][..d]) * c))
            .collect();
        let m = row.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = row.iter().map(|&x| acc((x - m).exp())).collect();
        let sum = e.iter().fold(0.0, |z, &x| acc(z + x));
        for j in 0..=i {
            p[i * t + j] = rd(acc(e[j] / sum));
        }
    }
    let mut o = vec![0.0; t * d];
    for i in 0..t {
        for x in 0..d {
            o[i * d + x] = (0..=i).fold(0.0, |z, j| acc(z + acc(p[i * t + j] * v[j * d + x])));
        }
    }
    // backward (all in units of s)
    let mut ds = vec![0.0; t * t];
    for i in 0..t {
        let dp: Vec<f64> = (0..=i)
            .map(|j| dot(&dov[i * d..][..d], &v[j * d..][..d]))
            .collect();
        let dt = (0..=i).fold(0.0, |z, j| acc(z + acc(p[i * t + j] * dp[j])));
        for j in 0..=i {
            ds[i * t + j] = rd(acc(acc(p[i * t + j] * acc(dp[j] - dt)) * c));
        }
    }
    let mut dq = vec![0.0; t * d];
    let mut dk = vec![0.0; t * d];
    let mut dv = vec![0.0; t * d];
    for i in 0..t {
        for x in 0..d {
            dq[i * d + x] =
                (0..=i).fold(0.0, |z, j| acc(z + acc(ds[i * t + j] * k[j * d + x]))) / s;
        }
    }
    for j in 0..t {
        for x in 0..d {
            dk[j * d + x] = (j..t).fold(0.0, |z, i| acc(z + acc(ds[i * t + j] * q[i * d + x]))) / s;
            dv[j * d + x] =
                (j..t).fold(0.0, |z, i| acc(z + acc(p[i * t + j] * dov[i * d + x]))) / s;
        }
    }
    [o, dq, dk, dv]
}

fn rel(a: &[f64], b: &[f64]) -> f64 {
    let n: f64 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
    let r: f64 = b.iter().map(|y| y * y).sum();
    (n / r).sqrt()
}

fn f16r(x: f64) -> f64 {
    round_f16(x as f32) as f64
}
fn f32r(x: f64) -> f64 {
    x as f32 as f64
}
fn same(x: f64) -> f64 {
    x
}

/// A4 (SIMULARE on the processor): one #513 head, T = 512, d = 64; dO of the step's size (~1e-5).
#[test]
fn t251e_a4_f16_precision() {
    let h = head(512, 64, 1e-5, 0x0251_0513);
    let want = attention(&h, same, true, 1.0);
    let runs = [
        ("F32 (azi)", attention(&h, f32r, false, 1.0)),
        ("f16, scara 2^14", attention(&h, f16r, false, 16_384.0)),
        ("f16, fara scara", attention(&h, f16r, false, 1.0)),
    ];
    let mut errs = Vec::new();
    for (name, got) in &runs {
        let e: Vec<f64> = (0..4).map(|n| rel(&got[n], &want[n])).collect();
        println!(
            "kt c242 A4 {name:<16} eroarea relativa: O {:.2e}, dQ {:.2e}, dK {:.2e}, dV {:.2e}",
            e[0], e[1], e[2], e[3]
        );
        errs.push(e);
    }
    // today's F32 path is the floor; f16 with the scale stays far above it but bounded
    assert!(errs[0].iter().all(|&e| e < 1e-5));
    assert!(errs[1].iter().all(|&e| e < 1e-2));
    // without the scale the backward loses more than with it
    assert!(errs[2][1] > errs[1][1] && errs[2][2] > errs[1][2]);
}
