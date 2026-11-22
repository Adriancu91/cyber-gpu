//! C-160 point 4 — T163 / T164, SIMULARE: the last two programs of the training step.
//!
//! - T163, program 9 (`tools/nak/gnorm.comp` → `gnorm.sm86.sass`): the total gradient norm N = √(Σ g²) over all
//!   tensors and the clipping factor c = min(1, max_norm / (N + 1e-6)), without atomics (partial sums per block, then
//!   one block), judged against the f64 oracle `gpu_compute::train::grad_norm` in a margin written BEFORE the
//!   measurement, and against PyTorch's norm of step 1 of the small golden step (`reteta.json`).
//! - T164, program 10 (`tools/nak/opt_step.comp` → `opt_step.sm86.sass`): the optimizer step over all parameters —
//!   `sgd` and `adamw_513` — against the oracle `train::{sgd_step, AdamW::step}` (SGD and m, v bit for bit; the AdamW
//!   weights in a written margin with room for the board's approximate √ and divisions), then the whole small golden
//!   step with the optimizer on the simulated board: 10 steps, gradients from the CPU oracle, norm + clipping + AdamW
//!   (or SGD) launched through our pushbuffer + QMD, against PyTorch's weights after steps 1 and 10.
//!
//! The "programs" run on the CPU (`gnorm_on_cpu`, `opt_step_on_cpu` → `gpu_compute::layer`), in the order of the code
//! NAK made, the approximate units modelled correctly rounded. Nothing here runs on the RTX 3070 Ti.

use std::path::{Path, PathBuf};

use gpu_compute::layer::{self as gl, OptScalars};
use gpu_compute::train::{self, AdamW, Arch, Batch};
use gpu_sim::cmdproc::{CommandProcessor, Event, GNORM_NEEDS, gnorm_on_cpu, opt_step_on_cpu};
use nvidia_ga104::gnorm::{self as gn, GNORM};
use nvidia_ga104::opt_step::{self as os, OPT_STEP};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const PROG_N: u64 = 0x10_0000;
const PROG_O: u64 = 0x18_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
const X: u64 = 0x100_0000;
const PART: u64 = 0x500_0000;
const RES: u64 = 0x510_0000;
const W: u64 = 0x600_0000;
const M: u64 = 0xa00_0000;
const V: u64 = 0xe00_0000;
const SENTINEL: f32 = -12345.0;
/// 2⁻²⁴: one rounding of f32.
const U: f64 = 1.0 / 16_777_216.0;
/// Room for each of the board's approximate units (√, division) — relative 2⁻²², as for the other programs.
const E_MUFU: f64 = 4.0 * U;
/// 2⁻¹²⁶: the smallest normal f32.
const FTZ: f64 = 1.175_494_350_822_287_5e-38;

/// The small golden model (hub `integrare/kronos/pas_de_aur_mic/reteta.json`).
const MIC: Arch = Arch {
    vocab: 256,
    layers: 2,
    n_emb: 64,
    heads: 4,
    ctx: 32,
};

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn hex(b: &[u8]) -> String {
    nvidia_ga104::sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

// ------------------------------------------------------------------ launching

/// A command processor with both programs.
fn cp() -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program_with_needs(GNORM, gnorm_on_cpu, GNORM_NEEDS);
    cp.register_program(OPT_STEP, opt_step_on_cpu);
    cp.mem.write(PROG_N, GNORM);
    cp.mem.write(PROG_O, OPT_STEP);
    cp
}

fn launch(
    cp: &mut CommandProcessor,
    prog: u64,
    registers: u32,
    cb: &[u8],
    grid: [u32; 3],
    block: [u32; 3],
    shared: Option<SharedMemory>,
) -> Result<(), gpu_core::error::GpuError> {
    let mut buf = vec![0u8; 0x100];
    buf[..cb.len()].copy_from_slice(cb);
    cp.mem.write(CBUF, &buf);
    let q = ComputeQmd {
        program_va: prog,
        registers,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid,
        block,
        release: Some((SEM, 7)),
    };
    let bytes = match shared {
        Some(s) => q.to_bytes_with_shared(&s).unwrap(),
        None => q.to_bytes().unwrap(),
    };
    cp.mem.write(QMD, &bytes);
    cp.mem.write_u32(SEM, 0);
    let mut w = [0u32; 16];
    let mut p = PushBuf::new(&mut w);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = p.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])?;
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 7);
    Ok(())
}

fn launch_gnorm(cp: &mut CommandProcessor, a: &gn::Args) {
    launch(
        cp,
        PROG_N,
        gn::REGISTERS,
        &a.cbuf().unwrap(),
        a.grid(),
        gn::BLOCK,
        Some(gn::SHARED),
    )
    .unwrap();
}

/// The driver's norm: x already at X; one launch if it fits in one chunk, else partial sums then the finish.
/// Returns (N, c, number of launches).
fn board_norm(cp: &mut CommandProcessor, n: usize, chunk: u32, max_norm: f32) -> (f32, f32, usize) {
    cp.mem.write_f32s(RES, &[SENTINEL; 4]);
    let blocks = (n as u32).div_ceil(chunk);
    let before = launches(cp);
    if blocks == 1 {
        let a = gn::Args {
            x: X,
            part: 0,
            res: RES,
            n: n as u32,
            chunk,
            mode: gn::Mode::Whole,
            max_norm,
        };
        launch_gnorm(cp, &a);
    } else {
        cp.mem
            .write_f32s(PART, &vec![SENTINEL; blocks as usize + 64]);
        let a = gn::Args {
            x: X,
            part: PART,
            res: 0,
            n: n as u32,
            chunk,
            mode: gn::Mode::Partial,
            max_norm,
        };
        launch_gnorm(cp, &a);
        let p = cp.mem.read_f32s(PART, blocks as usize + 64).unwrap();
        assert!(
            p[blocks as usize..].iter().all(|&v| v == SENTINEL),
            "part scris dupa blocuri"
        );
        let fin = gn::Args {
            x: PART,
            part: 0,
            res: RES,
            n: blocks,
            chunk: blocks.div_ceil(256) * 256,
            mode: gn::Mode::Finish,
            max_norm,
        };
        launch_gnorm(cp, &fin);
    }
    let r = cp.mem.read_f32s(RES, 4).unwrap();
    assert!(r[2..].iter().all(|&v| v == SENTINEL), "res scris dupa 2");
    (r[0], r[1], launches(cp) - before)
}

fn launches(cp: &CommandProcessor) -> usize {
    cp.events
        .iter()
        .filter(|e| matches!(e, Event::Launch { .. }))
        .count()
}

fn norm_of(x: &[f32], chunk: u32, max_norm: f32) -> (f32, f32) {
    let mut cp = cp();
    cp.mem.write_f32s(X, x);
    let (n, c, _) = board_norm(&mut cp, x.len(), chunk, max_norm);
    (n, c)
}

// ------------------------------------------------------------------ T163: the margin, written before the measurement
//
// S = Σ x² ≥ 0 (all terms of one sign). A thread's ffma chain of k terms: |Ŝ_t − S_t| ≤ 1.01·k·u·S_t; the block sum
// adds 5 butterfly levels + 7 sequential adds (12 roundings): the block's partial ≤ 1.01·(k₁ + 12)·u relative. The
// finish adds the P partials the same way (k₂ = ⌈P/256⌉): e_S = 1.01·(k₁ + 12 + k₂ + 12)·u relative (k₂ = 0 for a
// single launch), + n·FTZ absolute (squares below 2⁻¹²⁶ are flushed). N = √Ŝ: ½·e_S + u + E_MUFU relative, plus
// √(n·FTZ). c = max · RCP(N + 1e-6) (NAK's division): rel(N) + 3u + E_MUFU (the add, the RCP, the product); with
// c_true ≥ 1 the board must give exactly 1 unless c_true is within that margin of 1.

struct NormBound {
    n: f64,
    c: f64,
}

fn norm_bound(n_el: usize, chunk: u32, s_true: f64, max_norm: f64, e_mufu: f64) -> NormBound {
    let blocks = n_el.div_ceil(chunk as usize);
    let k1 = (chunk as usize).min(n_el).div_ceil(256) as f64;
    let k2 = if blocks > 1 {
        blocks.div_ceil(256) as f64 + 12.0
    } else {
        0.0
    };
    let e_s = 1.01 * (k1 + 12.0 + k2) * U;
    let n_true = s_true.sqrt();
    let rel_n = 0.5 * e_s + U + e_mufu;
    let dn = n_true * rel_n + (n_el as f64 * FTZ).sqrt() + FTZ;
    let c_true = max_norm / (n_true + 1e-6);
    let dc = c_true.min(1.0) * (rel_n + 3.0 * U + e_mufu) * 1.01 + FTZ;
    NormBound { n: dn, c: dc }
}

/// Share of the margin used (max over N and c); `Err` if outside.
fn judge_norm(
    x: &[f32],
    chunk: u32,
    max_norm: f32,
    got: (f32, f32),
    e_mufu: f64,
) -> Result<f64, String> {
    let n_true = train::grad_norm(&[x.to_vec()]);
    let s_true = n_true * n_true;
    let b = norm_bound(x.len(), chunk, s_true, max_norm as f64, e_mufu);
    let c_true = (max_norm as f64 / (n_true + 1e-6)).min(1.0);
    let en = (got.0 as f64 - n_true).abs();
    let ec = (got.1 as f64 - c_true).abs();
    if en.is_nan() || en > b.n {
        return Err(format!("N {} fata de {n_true} (margine {})", got.0, b.n));
    }
    if ec.is_nan() || ec > b.c || got.1 > 1.0 {
        return Err(format!("c {} fata de {c_true} (margine {})", got.1, b.c));
    }
    Ok((en / b.n).max(ec / b.c))
}

fn vecf(seed: u64, n: usize, scale: f32) -> Vec<f32> {
    let mut r = rng(seed);
    (0..n).map(|_| r() * scale).collect()
}

#[test]
fn t163_sha_and_cbuf_layout() {
    assert_eq!(hex(GNORM), gn::GNORM_SHA256);
    let txt = include_str!("../../../tools/nak/out/gnorm.sm86.txt");
    assert_eq!(GNORM.len(), 135 * 16);
    assert!(txt.contains("Instruction count: 135") && txt.contains("Num GPRs: 19"));
    assert_eq!(txt.matches("mufu.").count(), 2, "un SQRT + un RCP");
    assert!(!txt.contains("atom"), "fara atomic");
    let a = gn::Args {
        x: 0x1111,
        part: 0x2222,
        res: 0x3333,
        n: 1000,
        chunk: 512,
        mode: gn::Mode::Partial,
        max_norm: 1.0,
    };
    let cb = a.cbuf().unwrap();
    assert_eq!(&cb[0x28..0x30], &0x1111u64.to_le_bytes());
    assert_eq!(&cb[0x30..0x38], &0x2222u64.to_le_bytes());
    assert_eq!(&cb[0x38..0x40], &0x3333u64.to_le_bytes());
    assert_eq!(&cb[0x40..0x44], &1000u32.to_le_bytes());
    assert_eq!(&cb[0x44..0x48], &512u32.to_le_bytes());
    assert_eq!(&cb[0x48..0x4c], &1u32.to_le_bytes());
    assert_eq!(&cb[0x4c..0x50], &1.0f32.to_le_bytes());
    assert_eq!(&cb[..0x28], &[0u8; 0x28]);
    assert_eq!(a.grid(), [2, 1, 1]);
    // Refusals: chunk not a multiple of 256, single block that does not cover n, missing addresses, empty.
    assert!(gn::Args { chunk: 300, ..a }.cbuf().is_none());
    assert!(gn::Args { chunk: 0, ..a }.cbuf().is_none());
    assert!(gn::Args { n: 0, ..a }.cbuf().is_none());
    assert!(gn::Args { part: 0, ..a }.cbuf().is_none());
    assert!(
        gn::Args {
            mode: gn::Mode::Whole,
            ..a
        }
        .cbuf()
        .is_none()
    );
    assert!(
        gn::Args {
            mode: gn::Mode::Finish,
            chunk: 1024,
            res: 0,
            ..a
        }
        .cbuf()
        .is_none()
    );
    assert!(
        gn::Args {
            mode: gn::Mode::Finish,
            chunk: 1024,
            part: 0,
            ..a
        }
        .cbuf()
        .is_some()
    );
    assert_eq!(
        gn::Args {
            mode: gn::Mode::Whole,
            chunk: 1024,
            ..a
        }
        .grid(),
        [1, 1, 1]
    );
}

#[test]
fn t163_norm_in_the_margin_on_many_shapes() {
    // (seed, n, chunk, scale)
    let cases: [(u64, usize, u32, f32); 10] = [
        (1, 1, 256, 1.0),
        (2, 255, 256, 0.3),
        (3, 256, 256, 2.0),
        (4, 257, 256, 0.01),
        (5, 5_000, 1_024, 0.05),
        (6, 70_000, 4_096, 1e-3),
        (7, 137_000, 1_024, 1e3),  // 134 blocks → finish over 134 partials
        (8, 300_000, 256, 1.0),    // 1172 blocks → finish chunk 1280
        (9, 1_000, 65_536, 1e-18), // squares near FTZ
        (10, 1_000, 1_024, 1e18),
    ];
    let mut worst = 0.0f64;
    for (seed, n, chunk, scale) in cases {
        let x = vecf(seed, n, scale);
        let got = norm_of(&x, chunk, 1.0);
        let (mn, mc, _) = gl::grad_norm_gpu(&x, chunk as usize, 1.0).unwrap();
        assert_eq!(
            (got.0.to_bits(), got.1.to_bits()),
            (mn.to_bits(), mc.to_bits()),
            "lansarea = modelul"
        );
        let share = judge_norm(&x, chunk, 1.0, got, E_MUFU)
            .unwrap_or_else(|e| panic!("n {n} chunk {chunk}: {e}"));
        // The model's own error (correctly rounded units) leaves room for the board's.
        let own = judge_norm(&x, chunk, 1.0, got, 0.0).unwrap();
        worst = worst.max(share);
        println!(
            "kt c163 n {n} chunk {chunk} scara {scale:e}: N {} c {} — {:.1}% din margine ({:.1}% fara loc MUFU)",
            got.0,
            got.1,
            100.0 * share,
            100.0 * own
        );
    }
    assert!(worst < 1.0);
}

#[test]
fn t163_clipping_factor_and_special_values() {
    // Norm below max: c = 1 exactly; above: c < 1; exactly at max_norm: 1/(1+1e-6) → < 1.
    let small = vec![0.1f32; 4];
    assert_eq!(norm_of(&small, 256, 1.0).1, 1.0);
    let big = vec![3.0f32; 4];
    let (n, c) = norm_of(&big, 256, 1.0);
    assert_eq!(n, 6.0);
    assert!(c < 1.0 && (c as f64 - 1.0 / 6.000001).abs() < 1e-7);
    let one = [1.0f32];
    let (n, c) = norm_of(&one, 256, 1.0);
    assert_eq!(n, 1.0);
    assert!(c < 1.0, "1/(1+1e-6) < 1");
    // max_norm other than 1.
    let (_, c) = norm_of(&big, 256, 3.0);
    assert!((c as f64 - 0.5).abs() < 1e-6);
    // All zeros: N = 0, c = max/1e-6 → clamped to 1.
    assert_eq!(norm_of(&[0.0; 600], 256, 1.0), (0.0, 1.0));
    // A NaN gradient: N = NaN, c = NaN (as torch.clamp: the step would poison the weights — the driver must stop).
    let mut x = vecf(11, 1_000, 1.0);
    x[777] = f32::NAN;
    let (n, c) = norm_of(&x, 256, 1.0);
    assert!(n.is_nan() && c.is_nan());
    // An infinite gradient: N = ∞, c = 0 (PyTorch: max/(∞) = 0 → weights unchanged by SGD; still a stop for the driver).
    x[777] = f32::INFINITY;
    let (n, c) = norm_of(&x, 256, 1.0);
    assert!(n.is_infinite() && c == 0.0);
}

#[test]
fn t163_same_bits_every_run_and_any_order_of_blocks() {
    let x = vecf(21, 100_000, 0.7);
    let a = norm_of(&x, 1_024, 1.0);
    let b = norm_of(&x, 1_024, 1.0);
    assert_eq!(
        (a.0.to_bits(), a.1.to_bits()),
        (b.0.to_bits(), b.1.to_bits())
    );
    // Two launches for 98 blocks, one for a single chunk.
    let mut c = cp();
    c.mem.write_f32s(X, &x);
    assert_eq!(board_norm(&mut c, x.len(), 1_024, 1.0).2, 2);
    assert_eq!(board_norm(&mut c, 1_000, 1_024, 1.0).2, 1);
    // Another chunk: other bits possibly, same margin.
    let d = norm_of(&x, 4_096, 1.0);
    judge_norm(&x, 4_096, 1.0, d, E_MUFU).unwrap();
}

#[test]
fn t163_refuses_what_it_does_not_model() {
    let mut c = cp();
    c.mem.write_f32s(X, &[1.0; 512]);
    let a = gn::Args {
        x: X,
        part: PART,
        res: RES,
        n: 512,
        chunk: 256,
        mode: gn::Mode::Finish,
        max_norm: 1.0,
    };
    // Finish with two blocks (they would race on res), wrong block, QMD without shared memory.
    let mut cb = gn::Args { chunk: 512, ..a }.cbuf().unwrap().to_vec();
    assert!(
        launch(
            &mut c,
            PROG_N,
            gn::REGISTERS,
            &cb,
            [2, 1, 1],
            gn::BLOCK,
            Some(gn::SHARED)
        )
        .is_err()
    );
    assert!(
        launch(
            &mut c,
            PROG_N,
            gn::REGISTERS,
            &cb,
            [1, 1, 1],
            [128, 1, 1],
            Some(gn::SHARED)
        )
        .is_err()
    );
    assert!(
        launch(
            &mut c,
            PROG_N,
            gn::REGISTERS,
            &cb,
            [1, 1, 1],
            gn::BLOCK,
            None
        )
        .is_err()
    );
    // Workgroup base ≠ 0.
    cb[0] = 1;
    assert!(
        launch(
            &mut c,
            PROG_N,
            gn::REGISTERS,
            &cb,
            [1, 1, 1],
            gn::BLOCK,
            Some(gn::SHARED)
        )
        .is_err()
    );
}

/// The golden step's data.
fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../gpu-compute/tests/date/pas_de_aur_mic")
}

fn read_f32(p: &Path, n: usize) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(b.len(), 4 * n, "{}", p.display());
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_i32(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::try_from(i32::from_le_bytes(*c)).unwrap())
        .collect()
}

fn tensors(sub: &str) -> Vec<Vec<f32>> {
    MIC.tensors()
        .iter()
        .map(|t| {
            read_f32(
                &dir().join(sub).join(format!("{}.f32", t.name)),
                t.elements(),
            )
        })
        .collect()
}

/// PyTorch's gradient norms of the 10 steps (`reteta.json`, `retete.<name>.norme_gradient`), parsed by hand.
fn pytorch_norms(name: &str) -> Vec<f64> {
    let s = std::fs::read_to_string(dir().join("reteta.json")).unwrap();
    let r = &s[s.find(&format!("\"{name}\"")).unwrap()..];
    let a = &r[r.find("\"norme_gradient\"").unwrap()..];
    let a = &a[a.find('[').unwrap() + 1..a.find(']').unwrap()];
    a.split(',').map(|v| v.trim().parse().unwrap()).collect()
}

fn flat(t: &[Vec<f32>]) -> Vec<f32> {
    t.iter().flatten().copied().collect()
}

#[test]
fn t163_pytorch_norm_of_the_golden_step() {
    let g = flat(&tensors("adamw_513/grad_pas1"));
    assert_eq!(
        g.len(),
        MIC.tensors().iter().map(|t| t.elements()).sum::<usize>()
    );
    let want = pytorch_norms("adamw_513")[0];
    for chunk in [256u32, 4_096, 131_072] {
        let (n, c) = norm_of(&g, chunk, 1.0);
        let rel = (n as f64 - want).abs() / want;
        judge_norm(&g, chunk, 1.0, (n, c), E_MUFU).unwrap();
        println!(
            "kt c163 pas_de_aur_mic chunk {chunk}: N {n} fata de PyTorch {want} (rel {rel:.2e}), c {c}"
        );
        assert!(rel <= 1e-6, "norma fata de PyTorch: {rel}");
        assert!((c as f64 - 1.0 / (want + 1e-6)).abs() <= 2e-6);
    }
}

#[test]
fn t163_planted_mistakes_are_caught() {
    let mut x = vecf(31, 50_000, 0.2);
    x[49_999] = 3.0;
    let chunk = 1_024u32;
    let (n, c, parts) = gl::grad_norm_gpu(&x, chunk as usize, 1.0).unwrap();
    judge_norm(&x, chunk, 1.0, (n, c), E_MUFU).unwrap();
    let fin = |s: f32| gl::gnorm_finish_gpu(s, 1.0);
    let s: f32 = parts.iter().sum();
    let wrong: Vec<(&str, (f32, f32))> = vec![
        ("fara patrate", fin(x.iter().sum::<f32>().abs())),
        (
            "ultimul bloc uitat",
            fin(parts[..parts.len() - 1].iter().sum()),
        ),
        ("doar primul bloc", fin(parts[0])),
        ("fara radacina", (s, (1.0 / (s + 1e-6)).min(1.0))),
        (
            "ultimul element uitat",
            fin(gl::gnorm_block_gpu(&x, 0, x.len() - 1, true)),
        ),
        ("N/2", (n / 2.0, c)),
        ("c al lui 2·max", (n, (2.0 / (n + 1e-6)).min(1.0))),
    ];
    for (what, got) in wrong {
        assert!(
            judge_norm(&x, chunk, 1.0, got, E_MUFU).is_err(),
            "nu a prins: {what}"
        );
    }
    // c without the clamp at 1 (a small norm).
    let y = vecf(32, 100, 0.01);
    let (n, _, _) = gl::grad_norm_gpu(&y, 256, 1.0).unwrap();
    assert!(
        judge_norm(&y, 256, 1.0, (n, 1.0 / (n + 1e-6)), E_MUFU).is_err(),
        "nu a prins: c fara limita 1"
    );
}

// ------------------------------------------------------------------ T164: the optimizer step

/// The board's optimizer over all elements of w (and m, v for AdamW), with clipping factor `coef`.
#[allow(clippy::too_many_arguments)]
fn board_opt(
    cp: &mut CommandProcessor,
    adam: bool,
    s: &OptScalars,
    coef: f32,
    g: &[f32],
    w: &mut [f32],
    m: &mut [f32],
    v: &mut [f32],
) {
    let n = g.len();
    cp.mem.write_f32s(X, g);
    cp.mem.write_f32s(RES, &[coef]);
    cp.mem.write_f32s(W, w);
    cp.mem.write_f32s(W + 4 * n as u64, &[SENTINEL; 300]);
    cp.mem.write_f32s(M, m);
    cp.mem.write_f32s(V, v);
    let a = os::Args {
        g: X,
        coef: RES,
        w: W,
        m: M,
        v: V,
        n: n as u32,
        mode: if adam { os::Mode::AdamW } else { os::Mode::Sgd },
        scalars: [
            s.neg_lr, s.decay, s.w1, s.b2, s.omb2, s.bc2_sqrt, s.neg_step, s.eps,
        ],
    };
    launch(
        cp,
        PROG_O,
        os::REGISTERS,
        &a.cbuf().unwrap(),
        a.grid(),
        os::BLOCK,
        None,
    )
    .unwrap();
    w.copy_from_slice(&cp.mem.read_f32s(W, n).unwrap());
    assert!(
        cp.mem
            .read_f32s(W + 4 * n as u64, 300)
            .unwrap()
            .iter()
            .all(|&x| x == SENTINEL),
        "w scris dupa n"
    );
    m.copy_from_slice(&cp.mem.read_f32s(M, n).unwrap());
    v.copy_from_slice(&cp.mem.read_f32s(V, n).unwrap());
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

// The AdamW margin, written before the measurement. m and v are plain f32 operations in the oracle's order: bit for
// bit (data without subnormals). The update upd = −step·(m / (√v / bc2 + eps)); NAK divides by multiplying with a
// reciprocal, so on the board: √ (u + E_MUFU), RCP(bc2) (u + E_MUFU) and the product (u), the add of eps (u), RCP(den)
// (u + E_MUFU), the product by m (u), by −step (u) → rel ≤ 3·E_MUFU + 7u; the oracle's own upd (√, 2 divisions, add,
// product) is off by ≤ 5u from the exact one. The final w = fl(w·decay + upd) rounds once on each side:
//   |w_board − w_oracle| ≤ 1.01·(3·E_MUFU + 12u)·|upd| + 2.01·u·|w_oracle| + 4·FTZ.
fn adam_bound(upd: f64, w: f64, e_mufu: f64) -> f64 {
    1.01 * (3.0 * e_mufu + 12.0 * U) * upd.abs() + 2.01 * U * w.abs() + 4.0 * FTZ
}

/// A random parameter vector with state as after a few steps.
struct OptCase {
    g: Vec<f32>,
    w: Vec<f32>,
    m: Vec<f32>,
    v: Vec<f32>,
}

fn opt_case(seed: u64, n: usize, gs: f32) -> OptCase {
    let mut r = rng(seed);
    let g: Vec<f32> = (0..n).map(|_| r() * gs).collect();
    let w: Vec<f32> = (0..n).map(|_| r() * 0.2).collect();
    let m: Vec<f32> = (0..n).map(|_| r() * gs * 0.1).collect();
    let v: Vec<f32> = (0..n).map(|_| (r() * gs).powi(2) * 0.01).collect();
    OptCase { g, w, m, v }
}

#[test]
fn t164_sha_and_cbuf_layout() {
    assert_eq!(hex(OPT_STEP), os::OPT_STEP_SHA256);
    let txt = include_str!("../../../tools/nak/out/opt_step.sm86.txt");
    assert_eq!(OPT_STEP.len(), 61 * 16);
    assert!(txt.contains("Instruction count: 61") && txt.contains("Num GPRs: 19"));
    assert_eq!(txt.matches("mufu.").count(), 3, "un SQRT + doua RCP");
    assert!(!txt.contains("ffma"), "precise: nimic contopit");
    let a = os::Args {
        g: 1,
        coef: 2,
        w: 3,
        m: 4,
        v: 5,
        n: 1000,
        mode: os::Mode::AdamW,
        scalars: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    };
    let cb = a.cbuf().unwrap();
    for (i, va) in [1u64, 2, 3, 4, 5].iter().enumerate() {
        assert_eq!(&cb[0x28 + 8 * i..0x30 + 8 * i], &va.to_le_bytes());
    }
    assert_eq!(&cb[0x50..0x54], &1000u32.to_le_bytes());
    assert_eq!(&cb[0x54..0x58], &1u32.to_le_bytes());
    for i in 0..8 {
        assert_eq!(
            &cb[0x58 + 4 * i..0x5c + 4 * i],
            &(i as f32 + 1.0).to_le_bytes()
        );
    }
    assert_eq!(&cb[..0x28], &[0u8; 0x28]);
    assert_eq!(a.grid(), [4, 1, 1]);
    assert!(os::Args { m: 0, ..a }.cbuf().is_none());
    assert!(
        os::Args {
            m: 0,
            v: 0,
            mode: os::Mode::Sgd,
            ..a
        }
        .cbuf()
        .is_some()
    );
    assert!(os::Args { coef: 0, ..a }.cbuf().is_none());
    assert!(os::Args { n: 0, ..a }.cbuf().is_none());
}

#[test]
fn t164_sgd_is_bit_for_bit_the_oracle() {
    for (seed, n) in [(41u64, 1usize), (42, 255), (43, 257), (44, 20_000)] {
        let k = opt_case(seed, n, 0.5);
        let mut want = vec![k.w.clone()];
        train::sgd_step(&mut want, std::slice::from_ref(&k.g), 0.01).unwrap();
        let mut c = cp();
        let (mut w, mut m, mut v) = (k.w.clone(), k.m.clone(), k.v.clone());
        board_opt(
            &mut c,
            false,
            &OptScalars::sgd(0.01),
            1.0,
            &k.g,
            &mut w,
            &mut m,
            &mut v,
        );
        assert_eq!(bits(&w), bits(&want[0]), "n {n}");
        assert_eq!(
            (bits(&m), bits(&v)),
            (bits(&k.m), bits(&k.v)),
            "SGD nu atinge m, v"
        );
    }
}

#[test]
fn t164_adamw_against_the_oracle() {
    let mut worst = 0.0f64;
    for (seed, n, gs, steps, coef) in [
        (51u64, 1usize, 1.0f32, 1u32, 1.0f32),
        (52, 1_000, 1e-3, 1, 1.0),
        (53, 20_000, 0.1, 3, 1.0),
        (54, 7_777, 10.0, 10, 0.0929),
        (55, 4_096, 1e-12, 2, 1.0), // v near FTZ: eps dominates
        (56, 3_000, 1e6, 1, 1e-6),
    ] {
        let k = opt_case(seed, n, gs);
        let mut adam = AdamW::recipe_513(std::slice::from_ref(&k.w));
        adam.t = steps - 1;
        adam.m = vec![k.m.clone()];
        adam.v = vec![k.v.clone()];
        let s = OptScalars::adamw(&adam.scalars(steps));
        let mut g = k.g.clone();
        if coef < 1.0 {
            for x in &mut g {
                *x *= coef;
            }
        }
        let mut want = vec![k.w.clone()];
        let decayed: Vec<f32> = k.w.iter().map(|&x| x * s.decay).collect();
        adam.step(&mut want, &[g]).unwrap();
        let mut c = cp();
        let (mut w, mut m, mut v) = (k.w.clone(), k.m.clone(), k.v.clone());
        board_opt(&mut c, true, &s, coef, &k.g, &mut w, &mut m, &mut v);
        if gs > 1e-10 {
            assert_eq!(bits(&m), bits(&adam.m[0]), "m bit cu bit, caz {seed}");
            assert_eq!(bits(&v), bits(&adam.v[0]), "v bit cu bit, caz {seed}");
        }
        let mut share = 0.0f64;
        let mut same = 0usize;
        for i in 0..n {
            let upd = want[0][i] as f64 - decayed[i] as f64;
            let b = adam_bound(upd, want[0][i] as f64, E_MUFU);
            let e = (w[i] as f64 - want[0][i] as f64).abs();
            assert!(
                e <= b,
                "caz {seed} elem {i}: {} fata de {} (margine {b})",
                w[i],
                want[0][i]
            );
            share = share.max(e / b);
            same += (w[i].to_bits() == want[0][i].to_bits()) as usize;
        }
        worst = worst.max(share);
        println!(
            "kt c164 adamw caz {seed} n {n} g~{gs:e} pas {steps} c {coef}: {:.1}% din margine, {same}/{n} biti identici cu oracolul",
            100.0 * share
        );
    }
    assert!(worst < 1.0);
}

#[test]
fn t164_margin_holds_for_units_two_ulps_off() {
    // The board's √ and divisions are approximate: redo the element with each unit moved by ±2 ulp and check that the
    // written margin still holds (the model rounds correctly; the board may not).
    fn nudge(x: f32, k: i32) -> f32 {
        f32::from_bits((x.to_bits() as i32 + k) as u32)
    }
    let k = opt_case(61, 5_000, 0.05);
    let adam = AdamW::recipe_513(std::slice::from_ref(&k.w));
    let s = OptScalars::adamw(&adam.scalars(4));
    let mut worst = 0.0f64;
    for i in 0..k.g.len() {
        let (want, mi, vi) = gl::opt_step_gpu(true, &s, 1.0, k.g[i], k.w[i], k.m[i], k.v[i]);
        let wi = k.w[i] * s.decay;
        for d in [[2, -2, 2], [-2, 2, -2], [2, 2, 2], [-2, -2, -2]] {
            // NAK's order: den = SQRT(v)·RCP(bc2) + eps; w = w′ + (m·RCP(den))·neg_step.
            let sq = nudge(gpu_compute::math::sqrt(vi), d[0]);
            let den = sq * nudge(1.0 / s.bc2_sqrt, d[1]) + s.eps;
            let q = mi * nudge(1.0 / den, d[2]);
            let w = wi + q * s.neg_step;
            let b = adam_bound(want as f64 - wi as f64, want as f64, E_MUFU);
            let e = (w as f64 - want as f64).abs();
            assert!(e <= b, "elem {i} d {d:?}");
            worst = worst.max(e / b);
        }
    }
    println!(
        "kt c164 unitati mutate cu 2 ulp: {:.1}% din margine",
        100.0 * worst
    );
}

#[test]
fn t164_planted_mistakes_are_caught() {
    let k = opt_case(71, 4_000, 0.05);
    let mut adam = AdamW::recipe_513(std::slice::from_ref(&k.w));
    adam.t = 2;
    adam.m = vec![k.m.clone()];
    adam.v = vec![k.v.clone()];
    let s = OptScalars::adamw(&adam.scalars(3));
    let mut want = vec![k.w.clone()];
    adam.step(&mut want, std::slice::from_ref(&k.g)).unwrap();
    let decayed: Vec<f32> = k.w.iter().map(|&x| x * s.decay).collect();
    let ok = |w: &[f32]| {
        (0..w.len()).all(|i| {
            let upd = want[0][i] as f64 - decayed[i] as f64;
            (w[i] as f64 - want[0][i] as f64).abs() <= adam_bound(upd, want[0][i] as f64, E_MUFU)
        })
    };
    let run = |s: &OptScalars, coef: f32, g: &[f32]| {
        let mut c = cp();
        let (mut w, mut m, mut v) = (k.w.clone(), k.m.clone(), k.v.clone());
        board_opt(&mut c, true, s, coef, g, &mut w, &mut m, &mut v);
        w
    };
    assert!(ok(&run(&s, 1.0, &k.g)));
    let neg: Vec<f32> = k.g.iter().map(|x| -x).collect();
    let mut shifted = k.g.clone();
    shifted.rotate_left(1);
    let wrong: Vec<(&str, Vec<f32>)> = vec![
        (
            "fara scaderea greutatii",
            run(&OptScalars { decay: 1.0, ..s }, 1.0, &k.g),
        ),
        (
            "pasul fara corectia lui β1",
            run(
                &OptScalars {
                    neg_step: -5e-5,
                    ..s
                },
                1.0,
                &k.g,
            ),
        ),
        (
            "fara corectia lui β2",
            run(&OptScalars { bc2_sqrt: 1.0, ..s }, 1.0, &k.g),
        ),
        (
            "β-urile inversate",
            run(
                &OptScalars {
                    w1: s.omb2,
                    omb2: s.w1,
                    ..s
                },
                1.0,
                &k.g,
            ),
        ),
        ("semnul gradientului", run(&s, 1.0, &neg)),
        ("gradientul deplasat cu unu", run(&s, 1.0, &shifted)),
        ("taiere aplicata cand nu trebuie", run(&s, 0.5, &k.g)),
        (
            "eps de 1000 de ori",
            run(
                &OptScalars {
                    eps: s.eps * 1000.0,
                    ..s
                },
                1.0,
                &k.g,
            ),
        ),
    ];
    for (what, w) in wrong {
        assert!(!ok(&w), "nu a prins: {what}");
    }
}

#[test]
fn t164_refuses_what_it_does_not_model() {
    let mut c = cp();
    let a = os::Args {
        g: X,
        coef: RES,
        w: W,
        m: M,
        v: V,
        n: 512,
        mode: os::Mode::Sgd,
        scalars: [0.0; 8],
    };
    let mut cb = a.cbuf().unwrap().to_vec();
    assert!(
        launch(
            &mut c,
            PROG_O,
            os::REGISTERS,
            &cb,
            [2, 1, 1],
            [128, 1, 1],
            None
        )
        .is_err()
    );
    assert!(
        launch(
            &mut c,
            PROG_O,
            os::REGISTERS,
            &cb,
            [2, 2, 1],
            os::BLOCK,
            None
        )
        .is_err()
    );
    cb[0] = 1;
    assert!(
        launch(
            &mut c,
            PROG_O,
            os::REGISTERS,
            &cb,
            [2, 1, 1],
            os::BLOCK,
            None
        )
        .is_err()
    );
}

// ------------------------------------------------------------------ the golden step with the board's optimizer

/// 10 steps of the small golden step: gradients from the CPU oracle (`train::loss_and_grad`, T151), then on the
/// simulated board: gnorm (adamw_513 only) → opt_step. Returns (weights after step 1, after step 10, norms).
fn golden_with_board_optimizer(adam: bool) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<f32>) {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: 2,
        len: 32,
    };
    let specs = MIC.tensors();
    let mut params = tensors("initiale");
    let n: usize = specs.iter().map(|t| t.elements()).sum();
    let opt = AdamW::recipe_513(&params);
    let (mut m, mut v) = (vec![0.0f32; n], vec![0.0f32; n]);
    let mut c = cp();
    let (mut after1, mut norms) = (Vec::new(), Vec::new());
    for step in 1..=10u32 {
        let (_, grads) = train::loss_and_grad(&MIC, &params, &batch).unwrap();
        let g = flat(&grads);
        let mut w = flat(&params);
        if adam {
            c.mem.write_f32s(X, &g);
            let (nrm, coef, launches) = board_norm(&mut c, n, 4_096, 1.0);
            assert_eq!(launches, 2);
            norms.push(nrm);
            board_opt(
                &mut c,
                true,
                &OptScalars::adamw(&opt.scalars(step)),
                coef,
                &g,
                &mut w,
                &mut m,
                &mut v,
            );
        } else {
            board_opt(
                &mut c,
                false,
                &OptScalars::sgd(0.01),
                1.0,
                &g,
                &mut w,
                &mut m,
                &mut v,
            );
        }
        let mut off = 0;
        for (p, sp) in params.iter_mut().zip(&specs) {
            p.copy_from_slice(&w[off..off + sp.elements()]);
            off += sp.elements();
        }
        if step == 1 {
            after1 = params.clone();
        }
    }
    (after1, params, norms)
}

fn worst_abs(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
    a.iter()
        .flatten()
        .zip(b.iter().flatten())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

#[test]
fn t164_golden_step_sgd_with_the_board_optimizer() {
    let (w1, w10, _) = golden_with_board_optimizer(false);
    let d1 = worst_abs(&w1, &tensors("sgd/dupa_pas1"));
    let d10 = worst_abs(&w10, &tensors("sgd/dupa_pas10"));
    // And bit for bit the CPU oracle's own run (the optimizer is the same f32 arithmetic).
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: 2,
        len: 32,
    };
    let cpu =
        train::golden_run(&MIC, &tensors("initiale"), &batch, train::Recipe::Sgd, 10).unwrap();
    assert_eq!(bits(&flat(&w10)), bits(&flat(&cpu.after_last)));
    println!(
        "kt c164 pas_de_aur_mic sgd cu optimizatorul pe placa simulata: dupa pasul 1 max|Δ| {d1:.2e}, dupa 10 {d10:.2e} (tol. {})",
        train::TOL_SGD_W_ABS
    );
    assert!(d1 <= train::TOL_SGD_W_ABS && d10 <= train::TOL_SGD_W_ABS);
}

#[test]
fn t164_golden_step_adamw_513_with_the_board_optimizer() {
    let (w1, w10, norms) = golden_with_board_optimizer(true);
    let d1 = worst_abs(&w1, &tensors("adamw_513/dupa_pas1"));
    let d10 = worst_abs(&w10, &tensors("adamw_513/dupa_pas10"));
    let want = pytorch_norms("adamw_513");
    let nrel = norms
        .iter()
        .zip(&want)
        .map(|(&a, &b)| (a as f64 - b).abs() / b)
        .fold(0.0, f64::max);
    println!(
        "kt c164 pas_de_aur_mic adamw_513 cu norma + taierea + AdamW pe placa simulata: dupa pasul 1 max|Δ| {d1:.2e}, dupa 10 {d10:.2e} (tol. {}); normele celor 10 pasi fata de PyTorch rel ≤ {nrel:.2e}",
        train::TOL_ADAMW_W_ABS
    );
    assert!(d1 <= train::TOL_ADAMW_W_ABS && d10 <= train::TOL_ADAMW_W_ABS);
    assert!(nrel <= 1e-5);
    // Against the CPU oracle's own run (T151: the oracle itself is 1.05e-5 from PyTorch after 10 steps). After step 1
    // the difference is only the approximate units (per element inside T164's margin: ≤ 1e-7 here). Through 10 steps
    // the gradients are recomputed from weights that differ in the last bits, and AdamW's m/(√v + eps) amplifies that
    // where g is near 0 (as between the oracle and PyTorch): a first guess of 1e-6 was too tight (measured 1.41e-6);
    // the criterion is the hub's tolerance.
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: 2,
        len: 32,
    };
    let cpu = train::golden_run(
        &MIC,
        &tensors("initiale"),
        &batch,
        train::Recipe::AdamW513,
        10,
    )
    .unwrap();
    let o1 = worst_abs(&w1, &cpu.after_step1);
    let o10 = worst_abs(&w10, &cpu.after_last);
    println!(
        "kt c164 adamw_513 fata de oracolul pe procesor: dupa pasul 1 max|Δ| {o1:.2e}, dupa 10 {o10:.2e}"
    );
    assert!(o1 <= 1e-7, "{o1}");
    assert!(o10 <= train::TOL_ADAMW_W_ABS, "{o10}");
}
