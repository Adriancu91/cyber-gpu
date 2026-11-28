//! T300 (CALCUL on the board's numbers, session 283): C-168 point 3, the probe `--mm-tc-v8` as it ran on the
//! RTX 3070 Ti in os-500 (28 Nov 2026, cycle 00:49, `rezultate/os-automat/2026-11-28/ciclu-auto-20261128-0049.txt` in
//! the hub). The board numbers below are copied from that log (medians of 5 launches, at the 64 KiB step); nothing here
//! runs on a board. What is computed: the verdict of G4-G6 (`docs/C-168_V8_PLAN.md` §4), the per-SM rates of v8 / v8n
//! relative to v7 at full waves, the wave model checked on dW, and the pick of the step at lot 6 with the measured rates
//! (`mm_tc_v8::RATES_OS500`) — the last part DEDUS (the model of T294 calibrated on os-496).

use nvidia_ga104::mm_tc as tc;
use nvidia_ga104::mm_tc_v8 as t8;

const SM: u32 = 48;

/// (rows, cols of C per block, blocks per SM at 64 KiB)
const V7: (u32, u32, u32) = (64, 64, 4);
const V8: (u32, u32, u32) = (128, 128, 2);
const V8N: (u32, u32, u32) = (128, 64, 3);

/// One probe shape: M x N of C, K, and the medians in ns (v7@64k, v8@64k, v8n@64k; `None` = not at full clock).
struct Probe {
    name: &'static str,
    m: u32,
    n: u32,
    k: u32,
    v7: f64,
    v8: f64,
    v8n: Option<f64>,
}

/// os-500, lines 175-177 / 193-195 / 211-213 of the log. v8n@64k on qkv-513-dw ran at seconds 1815.8-1817.5 of the
/// 1 800 s full-clock request ("IN AFARA FERESTREI DE VITEZA"): its 12.46 ms is not a full-clock time and is left out.
fn probes() -> [Probe; 3] {
    [
        Probe {
            name: "qkv-513",
            m: 6144,
            n: 2304,
            k: 768,
            v7: 1_198_432.0,
            v8: 1_063_328.0,
            v8n: Some(1_012_160.0),
        },
        Probe {
            name: "qkv-513-dx",
            m: 6144,
            n: 768,
            k: 2304,
            v7: 1_146_592.0,
            v8: 1_036_640.0,
            v8n: Some(929_120.0),
        },
        Probe {
            name: "qkv-513-dw",
            m: 2304,
            n: 768,
            k: 6144,
            v7: 1_469_120.0,
            v8: 1_650_368.0,
            v8n: None,
        },
    ]
}

fn blocks(m: u32, n: u32, t: (u32, u32, u32)) -> u32 {
    m.div_ceil(t.0) * n.div_ceil(t.1)
}

/// Useful share of the slots: blocks / (waves * slots).
fn eff(m: u32, n: u32, t: (u32, u32, u32)) -> f64 {
    let b = blocks(m, n, t);
    let slots = SM * t.2;
    b as f64 / (b.div_ceil(slots) * slots) as f64
}

fn tflops(p: &Probe, ns: f64) -> f64 {
    2.0 * p.m as f64 * p.n as f64 * p.k as f64 / ns / 1e3
}

/// G4-G6: the verdict, written against the bands of the plan (§4).
#[test]
fn t300_g4_g6_verdict() {
    let [qkv, dx, dw] = probes();
    // G4: same bits as mm_tc on 7 / 7 aligned sets, for v8, v8@64k, v8n, v8n@64k (log lines 187, 196, 205, 214).
    let g4_sets = [7u32, 7, 7, 7];
    assert!(g4_sets.iter().all(|&s| s == 7));
    // G5: qkv-513 v8 0,75-0,96 ms -> measured 1,063 ms: MISSED (above the band).
    assert!(qkv.v8 > 0.96e6);
    // G6: dX v8 0,72-0,92 -> 1,037 MISSED; dW v8n 1,02-1,30 -> not at full clock; v8 on dW 1,22-1,56 -> 1,650 MISSED.
    assert!(dx.v8 > 0.92e6 && dw.v8 > 1.56e6 && dw.v8n.is_none());
    // The threshold written in §4: v8 under x1,15 against v7 on qkv-513.
    let r = qkv.v7 / qkv.v8;
    assert!(r < 1.15, "{r}");
    println!(
        "kt c300 G4-G6 CALCUL (os-500): G4 CONFIRMATA (7/7 x4 aceiasi biti); G5 RATATA: v8 qkv-513 {:.3} ms = {:.1} TFLOP/s (banda 0,75-0,96), x{r:.3} fata de v7 (prag 1,15); G6 RATATA: v8 dX {:.3} (banda 0,72-0,92), v8 dW {:.3} (banda 1,22-1,56); v8n dW in afara ferestrei de viteza (nemasurat)",
        qkv.v8 / 1e6,
        tflops(&qkv, qkv.v8),
        dx.v8 / 1e6,
        dw.v8 / 1e6
    );
    println!(
        "kt c300 v8n (lipsea din nota centrului de comanda): qkv-513 {:.3} ms = {:.1} TFLOP/s (x{:.3} fata de v7), dX {:.3} ms = {:.1} TFLOP/s (x{:.3})",
        qkv.v8n.unwrap() / 1e6,
        tflops(&qkv, qkv.v8n.unwrap()),
        qkv.v7 / qkv.v8n.unwrap(),
        dx.v8n.unwrap() / 1e6,
        tflops(&dx, dx.v8n.unwrap()),
        dx.v7 / dx.v8n.unwrap()
    );
}

/// The per-SM rates at full waves. On qkv-513 and qkv-513-dx every program fills its waves exactly, so the time ratio
/// is the rate ratio; `RATES_OS500` is the lower of the two, rounded down.
#[test]
fn t300_rates_at_full_waves() {
    let ps = probes();
    let mut r8 = Vec::new();
    let mut r8n = Vec::new();
    for p in ps.iter().take(2) {
        for t in [V7, V8, V8N] {
            assert!((eff(p.m, p.n, t) - 1.0).abs() < 1e-12, "{} {t:?}", p.name);
        }
        r8.push(100.0 * p.v7 / p.v8);
        r8n.push(100.0 * p.v7 / p.v8n.unwrap());
    }
    let lo8 = r8.iter().cloned().fold(f64::MAX, f64::min);
    let lo8n = r8n.iter().cloned().fold(f64::MAX, f64::min);
    assert_eq!(t8::RATES_OS500.v8_pct, lo8.floor() as u32);
    assert_eq!(t8::RATES_OS500.v8n_pct, lo8n.floor() as u32);
    // The plan assumed v8 = 125-160 % and v8n = 0,9 x that. Measured: v8 under the band, and v8n ABOVE v8.
    assert!(lo8 < 125.0 && lo8n > lo8);
    println!(
        "kt c300 vitezele pe SM la valuri pline CALCUL: v8 {:.1} / {:.1} %, v8n {:.1} / {:.1} % din v7 (qkv / dX) -> RATES_OS500 = {} / {} (planul DEDUS 140 / 126)",
        r8[0],
        r8[1],
        r8n[0],
        r8n[1],
        t8::RATES_OS500.v8_pct,
        t8::RATES_OS500.v8n_pct
    );
}

/// The wave model on dW (the only probe shape with half-empty last waves): v8 on dW predicted from its dX rate and
/// the waves, against the board. Also: what v8n on dW should give (DEDUS, to be measured inside the window).
#[test]
fn t300_waves_on_dw() {
    let [_, dx, dw] = probes();
    let (m, n) = (dw.m, dw.n);
    assert_eq!(blocks(m, n, V7), 432);
    assert_eq!(blocks(m, n, V8), 108);
    assert_eq!(blocks(m, n, V8N), 216);
    // Same FLOP as dX; time = v7's dX time / rate / eff.
    let rate8 = dx.v7 / dx.v8;
    let rate8n = dx.v7 / dx.v8n.unwrap();
    let v7_pred = dx.v7 / eff(m, n, V7);
    let v8_pred = dx.v7 / rate8 / eff(m, n, V8);
    let v8n_pred = dx.v7 / rate8n / eff(m, n, V8N);
    let e7 = v7_pred / dw.v7 - 1.0;
    let e8 = v8_pred / dw.v8 - 1.0;
    // The model is within 6 % on v7 and within 12 % (pessimistic) on v8: a lone last wave runs a bit faster than a
    // full one (less pressure on L2), which the model does not see.
    assert!(e7.abs() < 0.06, "{e7}");
    assert!((0.0..0.12).contains(&e8), "{e8}");
    // v8n on dW: predicted below v7 (1,469) — the one number os-500 lost to the clock window.
    assert!(v8n_pred < dw.v7);
    println!(
        "kt c300 dW DEDUS: modelul valurilor: v7 {:.3} ms (masurat {:.3}, {:+.1} %), v8 {:.3} (masurat {:.3}, {:+.1} %); v8n prezis {:.3} ms (de masurat in fereastra; v7 {:.3})",
        v7_pred / 1e6,
        dw.v7 / 1e6,
        e7 * 100.0,
        v8_pred / 1e6,
        dw.v8 / 1e6,
        e8 * 100.0,
        v8n_pred / 1e6,
        dw.v7 / 1e6
    );
}

/// Where the gain went (CALCUL on the probe numbers and the compiled listings of session 279, `docs/C-168_V8_PLAN.md`
/// §7): the traffic from L2 to the SMs per program, and the warps per SM, against the rates.
#[test]
fn t300_where_the_gain_went() {
    let [qkv, ..] = probes();
    // f32 operands: every block reads its rows of A and its columns of B over the whole K.
    let traffic = |t: (u32, u32, u32)| -> f64 {
        let a = qkv.m as f64 * qkv.k as f64 * 4.0 * qkv.n.div_ceil(t.1) as f64;
        let b = qkv.k as f64 * qkv.n as f64 * 4.0 * qkv.m.div_ceil(t.0) as f64;
        a + b
    };
    let gbs = |t, ns: f64| traffic(t) / ns; // bytes per ns = GB/s
    let g7 = gbs(V7, qkv.v7);
    let g8 = gbs(V8, qkv.v8);
    let g8n = gbs(V8N, qkv.v8n.unwrap());
    // v8 halves the bytes per FLOP against v7 and still runs at a lower L2 rate than v7 and v8n: the L2 -> SM path
    // is not what holds v8 back.
    assert!((traffic(V7) / traffic(V8) - 2.0).abs() < 1e-9);
    assert!(g8 < g8n && g8n < g7);
    // Warps per SM: v8 2 x 8 = 16, v8n 3 x 4 = 12, v7 4 x 4 = 16. v8n has fewer warps and is faster: not occupancy.
    let warps = |threads: u32, per_sm: u32| threads / 32 * per_sm;
    assert_eq!(warps(t8::BLOCK[0], t8::BLOCKS_PER_SM), 16);
    assert_eq!(warps(t8::BLOCK_N[0], t8::BLOCKS_PER_SM_N), 12);
    // The difference left between the two halves of one source: register copies in the K loop (session 279 listing):
    // v8 148, v8n 41 per slice of 16 hmma -> 9,25 vs 2,56 per hmma; instructions per hmma 22,6 vs 20,0.
    let (mov8, mov8n) = (148.0 / 16.0, 41.0 / 16.0);
    assert!(mov8 > 3.0 * mov8n);
    println!(
        "kt c300 unde s-a dus castigul CALCUL: trafic L2->SM qkv-513 v7 {:.0} GB/s, v8 {:.0}, v8n {:.0} (v8 citeste jumatate din octetii lui v7 si tot nu e mai rapid pe octet); fire de 32 pe SM v7 16 / v8 16 / v8n 12; copii de registre pe hmma v8 {mov8:.2} / v8n {mov8n:.2}",
        g7, g8, g8n
    );
}

/// The 147 matmuls of the #513 step at lot 6 (as T294), each as an aligned `tc::Args`.
fn step_shapes() -> Vec<(u32, u32, u32, u32)> {
    let (t, e, v) = (3072u32, 768u32, 16_384u32);
    vec![
        (t, 3 * e, e, 12), // qkv
        (t, e, e, 12),     // out
        (t, 4 * e, e, 12), // fc
        (t, e, 4 * e, 12), // proj
        (t, v, e, 1),      // logit
        (t, e, 3 * e, 12), // qkv-dX
        (t, e, e, 12),     // out-dX
        (t, e, 4 * e, 12), // fc-dX
        (t, 4 * e, e, 12), // proj-dX
        (t, e, v, 1),      // cap-dX
        (e, 3 * e, t, 12), // qkv-dW
        (e, e, t, 12),     // out-dW
        (e, 4 * e, t, 12), // fc-dW
        (4 * e, e, t, 12), // proj-dW
        (e, v, t, 1),      // cap-dW
    ]
}

fn args(m: u32, n: u32, k: u32) -> tc::Args {
    tc::Args {
        a: 0x1000_0000,
        b: 0x2000_0000,
        c: 0x3000_0000,
        bias: 0,
        m,
        n,
        k,
        flags: 0,
    }
}

/// The pick at lot 6 with the measured rates, and the step it predicts (DEDUS: T294's model, R7 calibrated on the
/// 98,1 ms of os-496, the measured rates instead of the plan's band).
#[test]
fn t300_step_pick_with_measured_rates() {
    let sh = step_shapes();
    assert_eq!(sh.iter().map(|s| s.3).sum::<u32>(), 147);
    let count = |rates: t8::Rates| {
        let mut c = [0u32; 3];
        for &(m, n, k, cnt) in &sh {
            let a = args(m, n, k);
            assert!(t8::fits(&a));
            match t8::pick(&a, SM, rates, true) {
                t8::Pick::V7 => c[0] += cnt,
                t8::Pick::V8 => c[1] += cnt,
                t8::Pick::V8n => c[2] += cnt,
            }
        }
        c
    };
    let plan = count(t8::RATES_PLAN);
    assert_eq!(plan, [12, 38, 97]);
    let meas = count(t8::RATES_OS500);
    // With v8n faster per SM than v8, and its waves never worse on these shapes, v8 is never picked.
    assert_eq!(meas[1], 0);
    assert_eq!(meas.iter().sum::<u32>(), 147);
    // The step (DEDUS).
    let flop = |m: u32, n: u32, k: u32, c: u32| 2.0 * m as f64 * n as f64 * k as f64 * c as f64;
    let measured = 98.099e-3;
    let r7 = sh
        .iter()
        .map(|&(m, n, k, c)| flop(m, n, k, c) / eff(m, n, V7))
        .sum::<f64>()
        / measured;
    let f8 = t8::RATES_OS500.v8_pct as f64 / 100.0;
    let f8n = t8::RATES_OS500.v8n_pct as f64 / 100.0;
    let mm: f64 = sh
        .iter()
        .map(|&(m, n, k, c)| {
            let w = flop(m, n, k, c);
            let c7 = w / (r7 * eff(m, n, V7));
            let c8 = w / (f8 * r7 * eff(m, n, V8));
            let c8n = w / (f8n * r7 * eff(m, n, V8N));
            c7.min(c8).min(c8n)
        })
        .sum();
    let board = 147.810e-3;
    let rest = board - measured;
    let sps = |t: f64| 3072.0 / t;
    assert!(mm < measured && mm > 75e-3, "{mm}");
    println!(
        "kt c300 pasul lot 6 DEDUS: alegerea cu vitezele masurate v7 {} / v8 {} / v8n {} (planul: 12 / 38 / 97); inmultirile 98,1 -> {:.1} ms; placa 147,8 -> {:.1} ms = {:.0} simb/s (azi 20 795)",
        meas[0],
        meas[1],
        meas[2],
        mm * 1e3,
        (rest + mm) * 1e3,
        sps(rest + mm)
    );
}
