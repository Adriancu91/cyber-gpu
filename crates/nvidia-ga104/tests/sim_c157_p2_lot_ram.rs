//! C-157 point 2 after os-390 — T209, SIMULARE: the batch and the results of the resident training step through a
//! RAM staging area, moved by the copy program inside the step's list (`nvidia_ga104::batch_ram`), against the path
//! of os-390 (T206 with the cuts: batch written through the VRAM window, losses + norm read through it).
//! `antrenare mini` (vocabulary 16 384, context 32, E 64, 4 heads, 2 layers, batch 2 × 32, AdamW lr 1e-3), 10 steps
//! on 10 different batches, on `gpu_sim::cmdproc` (the model does not tell RAM from VRAM: one address space).
//!
//! Predictions written before the code: `docs/C-157_P2_LOT_PRIN_RAM.md` (R1–R5). Nothing here runs on the RTX 3070 Ti.

use std::sync::OnceLock;

use gpu_compute::train::{AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS, LN_FWD_NEEDS, MM_TRAIN_NEEDS, XENT_NEEDS,
    attn_back_on_cpu, attn_fwd_on_cpu, colsum_on_cpu, copy_b128_on_cpu, emb_back_on_cpu,
    emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_fwd_on_cpu, mm_train_on_cpu,
    opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::batch_ram::{self as br, ViaRam};
use nvidia_ga104::kronos_diet2::{RecordPlan, ends_span};
use nvidia_ga104::kronos_fast::WriteShadow;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{self as ts, PROGRAMS, Plan, Prog};
use nvidia_ga104::transfer::{self, COPY_B128};
use nvidia_rm570::regs::clc7c0 as c;

const MINI: Arch = Arch {
    vocab: 16_384,
    ctx: 32,
    n_emb: 64,
    heads: 4,
    layers: 2,
};
const B: usize = 2;
const T: usize = 32;
const STEPS: usize = 10;
const LR: f64 = 1e-3;

const VA: u64 = nvidia_ga104::mmu::VA_BASE;
const ZONE: u64 = VA + 0x20_0000;
const REC: u64 = VA + 0x30_0000;
/// The compute class is bound once, in its own stretch (in the OS: the channel's setup); the step's pushbuffer
/// follows at `PUSH`, as `WorkQueue::submit_fenced` places it once everything before is retired.
const SETUP: u64 = VA + 0x31_f000;
const PUSH: u64 = VA + 0x32_0000;
const PUSH_BYTES: u64 = 0x1_0000;
const PROGS: u64 = VA + 0x40_0000;
/// The RAM staging area (in the OS: pinned RAM next to the control zone) and the copy program.
const STAGE: u64 = VA + 0x38_0000;
const COPY: u64 = VA + 0x3f_0000;
const HEAP: u64 = VA + 0x100_0000;
/// Payloads move by more than one step's records every submission (the OS: `TOKEN_FIRST + STRIDE * subs`).
const FIRST: u32 = 0x7000;
const STRIDE: u32 = 0x100;

fn prog_va(p: Prog) -> u64 {
    PROGS + 0x1_0000 * p.index() as u64
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

fn initial() -> Vec<Vec<f32>> {
    let mut g = Rng(0x5eed_0000_0000_0209);
    MINI.tensors()
        .iter()
        .map(|t| {
            (0..t.elements())
                .map(|_| ((g.next() >> 11) as f64 / (1u64 << 53) as f64 * 0.07 - 0.035) as f32)
                .collect()
        })
        .collect()
}

fn batches() -> &'static Vec<(Vec<u32>, Vec<u32>)> {
    static D: OnceLock<Vec<(Vec<u32>, Vec<u32>)>> = OnceLock::new();
    D.get_or_init(|| {
        let mut g = Rng(0x5eed_0000_0000_2090);
        (0..STEPS)
            .map(|_| {
                let (mut inp, mut tgt) = (Vec::new(), Vec::new());
                for _ in 0..B {
                    let row: Vec<u32> = (0..=T)
                        .map(|_| (g.next() % MINI.vocab as u64) as u32)
                        .collect();
                    inp.extend_from_slice(&row[..T]);
                    tgt.extend_from_slice(&row[1..]);
                }
                (inp, tgt)
            })
            .collect()
    })
}

fn board() -> CommandProcessor {
    ts::check_programs().expect("SHA-256 programe");
    let mut cp = CommandProcessor::new();
    // The model's clock: a fixed cost per launch and per method, so the records carry growing times (SIMULARE).
    cp.sim_launch_ns = 3_000;
    cp.sim_method_ns = 10;
    for p in PROGRAMS {
        let code = p.code();
        match p {
            Prog::MmTrain => cp.register_program_with_needs(code, mm_train_on_cpu, MM_TRAIN_NEEDS),
            Prog::Xent => cp.register_program_with_needs(code, xent_on_cpu, XENT_NEEDS),
            Prog::LnFwd => cp.register_program_with_needs(code, ln_fwd_on_cpu, LN_FWD_NEEDS),
            Prog::LnBack => cp.register_program_with_needs(code, ln_back_on_cpu, LN_BACK_NEEDS),
            Prog::Colsum => cp.register_program(code, colsum_on_cpu),
            Prog::GeluTr => cp.register_program(code, gelu_tr_on_cpu),
            Prog::AttnFwd => cp.register_program(code, attn_fwd_on_cpu),
            Prog::AttnBack => cp.register_program(code, attn_back_on_cpu),
            Prog::EmbFwd => cp.register_program(code, emb_fwd_on_cpu),
            Prog::EmbBack => cp.register_program(code, emb_back_on_cpu),
            Prog::Gnorm => cp.register_program_with_needs(code, gnorm_on_cpu, GNORM_NEEDS),
            Prog::OptStep => cp.register_program(code, opt_step_on_cpu),
            Prog::Vadd => cp.register_program(code, vadd_on_cpu),
            other => panic!("program neasteptat in PROGRAMS: {other:?}"),
        };
        cp.mem.write(prog_va(p), code);
    }
    cp.register_program(COPY_B128, copy_b128_on_cpu);
    cp.mem.write(COPY, COPY_B128);
    let mut w = [0u32; 2];
    let mut pb = PushBuf::new(&mut w);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(SETUP, &bytes);
    let e = push::gpfifo_entry(SETUP, 2, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
    cp
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Way {
    /// os-390: the batch through the VRAM window, losses + norm read through it (T206 with the cuts).
    Window,
    /// The batch and the results through RAM, the copies in the step's list.
    Ram,
}

/// A mistake planted on purpose (R4).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Plant {
    None,
    /// From step 3 on the CPU forgets to write the staging area (the previous batch stays there).
    StaleStaging,
    /// From step 3 on the copies out are left off the list (nothing comes back to RAM).
    NoCopiesOut,
}

/// What the driver did on one step through the VRAM window (32-bit words) and in RAM (bytes).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct StepIo {
    push_written: usize,
    rec_read: usize,
    batch_window: usize,
    results_window: usize,
    staging_bytes: usize,
    launches: usize,
}

#[derive(Clone, Debug, PartialEq)]
struct Run {
    losses: Vec<f32>,
    norms: Vec<f32>,
    weights: Vec<Vec<f32>>,
    steps: Vec<StepIo>,
    /// The arena after the steps of [`ARENA_STEPS`].
    arenas: Vec<Vec<u8>>,
}

/// The steps after which the whole arena is kept for R2 (first batch through the window / RAM, middle, last).
const ARENA_STEPS: [usize; 3] = [1, 4, 9];

fn write_push(cp: &mut CommandProcessor, sh: &mut WriteShadow, words: &[u32]) -> usize {
    let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
    let runs = sh.changes(PUSH, &bytes).unwrap();
    let mut n = 0;
    for (pa, r) in runs {
        n += r.len() / 4;
        cp.mem.write(pa, &bytes[r]);
    }
    sh.remember(PUSH, &bytes).unwrap();
    n
}

fn plan0() -> Plan {
    Plan::new(MINI, B, T, &batches()[0].0, HEAP)
        .unwrap()
        .resident()
        .unwrap()
}

fn run(way: Way, plant: Plant) -> Run {
    let mut cp = board();
    let params = initial();
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let data = batches();
    let plan0 = plan0();
    // The whole arena made once (in the OS: the arena image urcata o data RAM -> VRAM), then the uploads.
    cp.mem.write(HEAP, &vec![0u8; plan0.bytes() as usize]);
    for (va, bytes) in plan0.uploads(&params, &data[0].0, &data[0].1).unwrap() {
        cp.mem.write(va, &bytes);
    }
    let mut shadow = WriteShadow::new(PUSH, PUSH_BYTES).unwrap();
    let (mut losses, mut norms, mut steps, mut arenas) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (k, (x, y)) in data.iter().enumerate() {
        let mut io = StepIo::default();
        let plan = plan0.rebatch(x).unwrap();
        let list = plan.step(Recipe::AdamW513, &opt, k as u32 + 1).unwrap();
        let (zone, qmds, via): (Vec<u8>, Vec<u64>, Option<ViaRam>) = match way {
            Way::Window => {
                if k > 0 {
                    for (va, bytes) in plan.batch_uploads(x, y).unwrap() {
                        io.batch_window += bytes.len() / 4;
                        cp.mem.write(va, &bytes);
                    }
                }
                let (z, q) = ts::pack(&list, prog_va, ZONE).unwrap();
                (z, q, None)
            }
            Way::Ram => {
                let mut v = br::via_ram(&plan, x, y, STAGE).unwrap();
                if !(plant == Plant::StaleStaging && k >= 3) {
                    cp.mem.write(STAGE, &v.staging);
                    io.staging_bytes = v.staging.len();
                }
                if plant == Plant::NoCopiesOut && k >= 3 {
                    v.copies_out.clear();
                }
                let (z, q) = br::pack_via_ram(&list, prog_va, COPY, ZONE, &v).unwrap();
                (z, q, Some(v))
            }
        };
        assert!(zone.len() as u64 <= REC - ZONE);
        cp.mem.write(ZONE, &zone);
        let n = qmds.len();
        io.launches = n;
        if k == 0 {
            cp.mem.write(REC, &vec![0u8; 16 * (n + 1)]);
        }
        let first = FIRST + STRIDE * k as u32;
        let mut w = vec![0u32; transfer::sequence_words_fresh(n)];
        let mut pb = PushBuf::new(&mut w);
        transfer::encode_sequence_fresh(&mut pb, &qmds, REC, first).unwrap();
        let nw = pb.len();
        io.push_written = write_push(&mut cp, &mut shadow, &w);
        let e = push::gpfifo_entry(PUSH, nw as u32, false).unwrap();
        cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
        let last = first + n as u32;
        assert_eq!(cp.mem.read_u32(REC + 16 * n as u64).unwrap(), last);
        let mut recs = vec![0u8; 16 * (n + 1)];
        cp.mem.read(REC, &mut recs).unwrap();
        let all_ns: u64 = transfer::analyze_sequence(&recs, n, first)
            .unwrap()
            .iter()
            .sum();
        if RecordPlan::DIET.reads_all(k) {
            io.rec_read = 4 * (n + 1);
        } else {
            io.rec_read = 8;
            assert_eq!(
                ends_span(&recs[..16], &recs[16 * n..16 * (n + 1)], n, first).unwrap(),
                all_ns
            );
        }
        let (rows, norm) = match &via {
            None => {
                let [(lr, ln), (nr, nn)] = plan.step_reads();
                io.results_window = ((ln + nn) / 4) as usize;
                (
                    cp.mem.read_f32s(lr, (ln / 4) as usize).unwrap(),
                    cp.mem.read_f32s(nr, (nn / 4) as usize).unwrap()[0],
                )
            }
            Some(v) => {
                let mut st = vec![0u8; v.staging.len()];
                cp.mem.read(STAGE, &mut st).unwrap();
                br::results(&st, v).unwrap()
            }
        };
        losses.push(ts::Outcome::mean_loss(&rows));
        norms.push(norm);
        steps.push(io);
        if plant == Plant::None && ARENA_STEPS.contains(&k) {
            let mut a = vec![0u8; plan0.bytes() as usize];
            cp.mem.read(HEAP, &mut a).unwrap();
            arenas.push(a);
        }
    }
    let weights = plan0
        .unflat(&cp.mem.read_f32s(plan0.w, plan0.n_params).unwrap())
        .unwrap();
    Run {
        losses,
        norms,
        weights,
        steps,
        arenas,
    }
}

fn window() -> &'static Run {
    static R: OnceLock<Run> = OnceLock::new();
    R.get_or_init(|| run(Way::Window, Plant::None))
}

fn ram() -> &'static Run {
    static R: OnceLock<Run> = OnceLock::new();
    R.get_or_init(|| run(Way::Ram, Plant::None))
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn t209_r1_same_numbers_bit_for_bit() {
    let (a, b) = (window(), ram());
    assert_eq!(a.losses.len(), STEPS);
    assert!(a.losses.iter().all(|l| l.is_finite()));
    assert_eq!(bits(&a.losses), bits(&b.losses), "pierderile");
    assert_eq!(bits(&a.norms), bits(&b.norms), "normele");
    for (ta, tb) in a.weights.iter().zip(&b.weights) {
        assert_eq!(bits(ta), bits(tb), "greutatile");
    }
    println!(
        "T209 R1 (SIMULARE): 10 pasi, pierderi {:.4} .. {:.4}; prin RAM = prin fereastra bit cu bit",
        a.losses[0],
        a.losses[STEPS - 1]
    );
}

#[test]
fn t209_r2_arena_differs_only_in_rounding_tails() {
    let (a, b) = (window(), ram());
    let p = plan0();
    let mut tails = 0usize;
    for (j, &k) in ARENA_STEPS.iter().enumerate() {
        let (x, y) = &batches()[k];
        // The bytes the RAM path may write beyond the window path: the 16-byte rounding of each area it copies in.
        let plan = p.rebatch(x).unwrap();
        let ups = plan.batch_uploads(x, y).unwrap();
        let in_tail = |addr: u64| {
            ups.iter().any(|(va, bytes)| {
                let end = va + bytes.len() as u64;
                addr >= end && addr < va + (bytes.len() as u64).div_ceil(16) * 16
            })
        };
        let (aa, ab) = (&a.arenas[j], &b.arenas[j]);
        assert_eq!(aa.len(), ab.len());
        for (c, (ca, cb)) in aa.chunks(4096).zip(ab.chunks(4096)).enumerate() {
            if ca == cb {
                continue;
            }
            for (i, (wa, wb)) in ca.iter().zip(cb).enumerate() {
                if wa != wb {
                    let addr = HEAP + (c * 4096 + i) as u64;
                    assert!(
                        in_tail(addr),
                        "pasul {k}: octetul {addr:#x} difera in afara cozilor"
                    );
                    assert_eq!(*wb, 0, "pasul {k}: coada pusa pe zero");
                    tails += 1;
                }
            }
        }
    }
    println!(
        "T209 R2 (SIMULARE): arena aceeasi dupa pasii 1, 4, 9; {tails} octeti diferiti (doar in cozile rotunjirii au voie, pusi pe 0)"
    );
}

#[test]
fn t209_r3_window_words_per_step() {
    let (a, b) = (window(), ram());
    let n = plan0().launches_per_step(Recipe::AdamW513);
    assert_eq!(n, 75);
    for (k, (sa, sb)) in a.steps.iter().zip(&b.steps).enumerate() {
        assert_eq!(sa.launches, n);
        assert_eq!(sb.launches, n + 7, "5 copieri in + 2 afara");
        assert_eq!(sb.batch_window, 0);
        assert_eq!(sb.results_window, 0);
        assert_eq!(sa.results_window, 65);
        if k >= 2 {
            assert_eq!(sa.push_written, n + 1, "pasul {k}");
            assert_eq!(
                sb.push_written,
                n + 8,
                "pasul {k}: doar valorile inregistrarilor"
            );
        }
        if k > 0 {
            assert_eq!(sa.batch_window, 321, "pasul {k}");
        }
    }
    let worst = |r: &Run, f: fn(&StepIo) -> (usize, usize)| {
        r.steps[2..]
            .iter()
            .filter(|s| s.rec_read == 8)
            .map(f)
            .max()
            .unwrap()
    };
    let io = |s: &StepIo| {
        (
            s.push_written + s.batch_window,
            s.rec_read + s.results_window,
        )
    };
    let (wa, ra) = worst(a, io);
    let (wb, rb) = worst(b, io);
    assert_eq!((wa, ra), (76 + 321, 8 + 65));
    assert_eq!((wb, rb), (83, 8));
    // R5: the costs MEASURED in os-390 (0.76 us per word written + read back, ~1.1 us per word read).
    let us = |w: usize, r: usize| w as f64 * 0.76 + r as f64 * 1.1;
    let staging = b.steps[2].staging_bytes;
    println!("T209 R3/R5 (SIMULARE + aritmetica cu costurile din os-390), {n} lansari pe pas:");
    println!(
        "  prin fereastra: scrise {wa} (pushbuffer 76, lotul 321), citite {ra} -> ~{:.0} us",
        us(wa, ra)
    );
    println!(
        "  prin RAM:       scrise {wb} (pushbuffer {wb}), citite {rb} -> ~{:.0} us; tranzitul {staging} B in RAM; +7 lansari pe placa",
        us(wb, rb)
    );
    assert!(us(wa, ra) - us(wb, rb) > 280.0);
}

#[test]
fn t209_r4_planted_mistakes_caught() {
    let good = ram();
    let stale = run(Way::Ram, Plant::StaleStaging);
    assert_eq!(bits(&stale.losses[..3]), bits(&good.losses[..3]));
    assert_ne!(
        bits(&stale.losses[3..]),
        bits(&good.losses[3..]),
        "lotul vechi din tranzit"
    );
    let no_out = run(Way::Ram, Plant::NoCopiesOut);
    assert_eq!(bits(&no_out.losses[..3]), bits(&good.losses[..3]));
    // The board still trains right (the same weights), but nothing reaches the staging area: the CPU wrote the result
    // areas on zero with the batch, so it reads loss 0 and norm 0 (impossible values: ln 16 384 ≈ 9.7 at the start),
    // never a plausible stale one.
    for (ta, tb) in no_out.weights.iter().zip(&good.weights) {
        assert_eq!(bits(ta), bits(tb));
    }
    assert!(
        no_out.losses[3..].iter().all(|&l| l == 0.0),
        "pierderea 0 = copierea lipsa"
    );
    assert!(
        no_out.norms[3..].iter().all(|&n| n == 0.0),
        "norma 0 = copierea lipsa"
    );
    assert_ne!(bits(&no_out.losses[3..]), bits(&good.losses[3..]));
    // Refused before anything is built.
    let p = plan0();
    let (x, y) = &batches()[1];
    let plan = p.rebatch(x).unwrap();
    assert!(
        br::via_ram(&plan, x, y, STAGE + 16).is_err(),
        "tranzit nealiniat"
    );
    assert!(
        br::via_ram(&plan, &batches()[2].0, y, STAGE).is_err(),
        "lot strain"
    );
    let v = br::via_ram(&plan, x, y, STAGE).unwrap();
    assert!(
        br::results(&v.staging[..v.norm_at.0], &v).is_err(),
        "tranzit prea scurt"
    );
    println!(
        "T209 R4 (SIMULARE): tranzitul nescris, copierile afara lipsa, tranzit nealiniat / lot strain / prea scurt: prinse"
    );
}
