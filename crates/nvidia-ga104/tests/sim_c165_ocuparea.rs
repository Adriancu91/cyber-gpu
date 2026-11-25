//! C-165 — T230, SIMULARE (`docs/C-165_OCUPAREA.md` §4): the SM shared-memory step of the QMD and the occupancy
//! probe. (a) the blocks-per-SM table (DEDUS arithmetic), (b)–(c) `ComputeQmd::encode_with_sm_config` (only an
//! addition), (d) `occupancy::best_step`, (e) the 10 probe runs through our pushbuffer + QMD in the command processor
//! model give the witness's bits, (f) the model refuses a wrong step. The model has no occupancy and no timing of its
//! own: nothing here says anything about speed. Nothing here runs on the RTX 3070 Ti.

use gpu_sim::cmdproc::{
    self as cpm, CommandProcessor, Event, MM_TRAIN_BIAS, MM_TRAIN_TB, MM_TRAIN_TILE_NEEDS, Needs,
    ProgramFn, mm_train_tile_on_cpu,
};
use nvidia_ga104::mm_train as mt;
use nvidia_ga104::mm_train_variants::VARIANTS;
use nvidia_ga104::occupancy::{
    self as oc, Bound, Fit, Kernel, PROBE, Reserve, SM86, best_step, blocks_per_sm,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SM_CONFIG_STEPS, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;
use nvidia_rm570::regs::qmd_v03_00 as q;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const A: u64 = 0x40_0000;
const B: u64 = 0x80_0000;
const BIAS: u64 = 0xbf_0000;
const C: u64 = 0xc0_0000;
const QMD: u64 = 0x100_0000;
const PUSH: u64 = 0x101_0000;
const SENTINEL: f32 = -12345.0;
const KIB: u32 = 1024;

fn model(i: usize) -> (ProgramFn, Needs) {
    match i {
        0 => (mm_train_tile_on_cpu, MM_TRAIN_TILE_NEEDS),
        3 => (cpm::mm_tv_128x128_on_cpu, cpm::MM_TV_128X128.needs()),
        4 => (cpm::mm_tv_64x64k8_on_cpu, cpm::MM_TV_64X64K8.needs()),
        _ => unreachable!("not in the probe"),
    }
}

fn fit(blocks: u32, bound: Bound) -> Option<Fit> {
    Some(Fit { blocks, bound })
}

/// T230a: the table of §2 (R0 / R1), with the named limit.
#[test]
fn t230a_blocks_per_sm_table() {
    let tile = oc::variant_kernel(0);
    let k8 = oc::variant_kernel(4);
    let big = oc::variant_kernel(3);
    let tc = oc::mm_tc_kernel();
    assert_eq!(
        (tile.threads, tile.registers, tile.shared_bytes),
        (256, 34, 8 * KIB)
    );
    assert_eq!(
        (k8.threads, k8.registers, k8.shared_bytes),
        (256, 34, 4 * KIB)
    );
    assert_eq!(
        (big.threads, big.registers, big.shared_bytes),
        (256, 90, 8 * KIB)
    );
    assert_eq!(
        (tc.threads, tc.registers, tc.shared_bytes),
        (128, 64, 24 * KIB)
    );
    let steps = [8, 16, 32, 64, 96];
    // (kernel, R0 blocks per step, R1 blocks per step, limit without shared memory)
    let table: [(Kernel, [u32; 5], [u32; 5], Fit); 4] = [
        (
            tile,
            [1, 2, 4, 6, 6],
            [0, 1, 3, 6, 6],
            Fit {
                blocks: 6,
                bound: Bound::Threads,
            },
        ),
        (
            k8,
            [2, 4, 6, 6, 6],
            [1, 3, 6, 6, 6],
            Fit {
                blocks: 6,
                bound: Bound::Threads,
            },
        ),
        (
            big,
            [1, 2, 2, 2, 2],
            [0, 1, 2, 2, 2],
            Fit {
                blocks: 2,
                bound: Bound::Registers,
            },
        ),
        (
            tc,
            [0, 0, 1, 2, 4],
            [0, 0, 1, 2, 3],
            Fit {
                blocks: 8,
                bound: Bound::Registers,
            },
        ),
    ];
    for (k, r0, r1, other) in table {
        assert_eq!(
            oc::blocks_without_shared(&SM86, &k),
            Some(other),
            "{}",
            k.name
        );
        for (s, kib) in steps.iter().enumerate() {
            for (r, want) in [(Reserve::None, r0[s]), (Reserve::OneKib, r1[s])] {
                let f = blocks_per_sm(&SM86, &k, kib * KIB, r).unwrap();
                assert_eq!(f.blocks, want, "{} {kib} KiB {r:?}", k.name);
                let bound = if want < other.blocks {
                    Bound::Shared
                } else {
                    other.bound
                };
                assert!(
                    f.bound == bound || f.bound == Bound::Shared,
                    "{} {kib} KiB",
                    k.name
                );
            }
        }
    }
    // The register threshold of `mm_tv_128x128`: 80 → 3 blocks, 85 (rounded to 88) → still 2.
    let at = |regs| {
        oc::blocks_without_shared(
            &SM86,
            &Kernel {
                registers: regs,
                ..big
            },
        )
    };
    assert_eq!(at(80), fit(3, Bound::Registers));
    assert_eq!(at(85), fit(2, Bound::Registers));
    assert_eq!(at(81), fit(2, Bound::Registers));
    assert_eq!(at(256), None);
    println!(
        "kt t230a: mm_train_tile 1/2/4/6 blocuri la 8/16/32/64 KiB (R0); 128x128 2 blocuri (registre), 3 la <=80 registre; DEDUS"
    );
}

fn qmd_for(i: usize) -> ComputeQmd {
    let v = &VARIANTS[i];
    ComputeQmd {
        program_va: PROG,
        registers: v.registers,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid: [4, 3, 1],
        block: v.block(),
        release: Some((SEM, 1)),
    }
}

/// T230b: MIN = TARGET = the step, MAX as today, every other field identical; at the one-block step, today's bytes.
#[test]
fn t230b_sm_config_is_only_the_step() {
    for i in [0, 3, 4] {
        let s = VARIANTS[i].shared();
        let q0 = qmd_for(i);
        let today = q0.encode_with_shared(&s).unwrap();
        let own = s.bytes.next_multiple_of(8 * KIB).max(8 * KIB);
        assert_eq!(q0.encode_with_sm_config(&s, own).unwrap(), today, "{i}");
        for step in SM_CONFIG_STEPS.iter().copied().filter(|&b| b >= s.bytes) {
            let m = q0.encode_with_sm_config(&s, step).unwrap();
            let units = step / 4096 + 1;
            assert_eq!(
                q::NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE.get(&m),
                units
            );
            assert_eq!(
                q::NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE.get(&m),
                units
            );
            assert_eq!(
                q::NVC6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE.get(&m),
                17
            );
            // Every other bit: identical to today's QMD.
            let mut a = m;
            let mut b = today;
            for f in [
                q::NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE,
                q::NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE,
            ] {
                f.set(&mut a, 0).unwrap();
                f.set(&mut b, 0).unwrap();
            }
            assert_eq!(a, b, "{i} {step}");
            assert_eq!(
                q0.to_bytes_with_sm_config(&s, step).unwrap().to_vec(),
                m.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()
            );
        }
    }
}

/// T230c: refused — a step that is not 8/16/32/64 KiB, below the block's memory, above MAX.
#[test]
fn t230c_bad_steps_refused() {
    let q0 = qmd_for(0);
    let s = VARIANTS[0].shared();
    for bad in [0, 4 * KIB, 12 * KIB, 96 * KIB, 100 * KIB, 8 * KIB + 1] {
        assert!(q0.encode_with_sm_config(&s, bad).is_err(), "{bad}");
    }
    let big = SharedMemory {
        bytes: 20 * KIB,
        ..s
    };
    assert!(q0.encode_with_sm_config(&big, 16 * KIB).is_err());
    assert!(q0.encode_with_sm_config(&big, 32 * KIB).is_ok());
    let small_max = SharedMemory {
        max_bytes: 16 * KIB,
        ..s
    };
    assert!(q0.encode_with_sm_config(&small_max, 32 * KIB).is_err());
    assert!(q0.encode_with_sm_config(&small_max, 16 * KIB).is_ok());
}

/// T230d: one step for a whole list.
#[test]
fn t230d_best_step() {
    let tile = oc::variant_kernel(0);
    let k8 = oc::variant_kernel(4);
    let big = oc::variant_kernel(3);
    for r in [Reserve::None, Reserve::OneKib] {
        assert_eq!(best_step(&SM86, &[tile], r), 64 * KIB, "{r:?}");
        assert_eq!(best_step(&SM86, &[k8], r), 32 * KIB, "{r:?}");
    }
    assert_eq!(best_step(&SM86, &[big], Reserve::None), 16 * KIB);
    assert_eq!(best_step(&SM86, &[big], Reserve::OneKib), 32 * KIB);
    assert_eq!(best_step(&SM86, &[big, k8], Reserve::None), 32 * KIB);
    assert_eq!(best_step(&SM86, &[big, tile], Reserve::None), 64 * KIB);
    // mm_tc never reaches its register limit (8) at 64 KiB: the largest step.
    assert_eq!(
        best_step(&SM86, &[oc::mm_tc_kernel()], Reserve::None),
        64 * KIB
    );
    assert_eq!(best_step(&SM86, &[], Reserve::None), 8 * KIB);
}

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

struct Job {
    a: Vec<f32>,
    b: Vec<f32>,
    bias: Vec<f32>,
    m: usize,
    n: usize,
    k: usize,
}

const FLAGS: u32 = MM_TRAIN_TB | MM_TRAIN_BIAS; // the qkv flags of the #513 step (6)

fn job(m: usize, n: usize, k: usize, seed: u64) -> Job {
    let mut r = rng(seed);
    Job {
        a: (0..m * k).map(|_| r()).collect(),
        b: (0..n * k).map(|_| r()).collect(),
        bias: (0..n).map(|_| r()).collect(),
        m,
        n,
        k,
    }
}

fn args(j: &Job) -> mt::Args {
    mt::Args {
        a: A,
        b: B,
        c: C,
        bias: BIAS,
        m: j.m as u32,
        n: j.n as u32,
        k: j.k as u32,
        flags: FLAGS,
    }
}

/// One launch of variant `i` with the QMD bytes `qmd`; `Ok(C)` or the model's refusal (C checked untouched then).
fn run_with(
    i: usize,
    j: &Job,
    qmd: &[u8],
) -> (Result<Vec<f32>, gpu_core::error::GpuError>, Vec<[u32; 3]>) {
    let mut cp = CommandProcessor::new();
    let (f, needs) = model(i);
    cp.register_program_with_needs(VARIANTS[i].code, f, needs);
    cp.mem.write(PROG, VARIANTS[i].code);
    let mut cb = vec![0u8; 0x100];
    cb[..mt::CBUF_BYTES].copy_from_slice(&args(j).cbuf().unwrap());
    cp.mem.write(CBUF, &cb);
    cp.mem.write_f32s(A, &j.a);
    cp.mem.write_f32s(B, &j.b);
    cp.mem.write_f32s(BIAS, &j.bias);
    cp.mem.write_f32s(C, &vec![SENTINEL; j.m * j.n + 16]);
    cp.mem.write_u32(SEM, 0);
    cp.mem.write(QMD, qmd);
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
    let ran = cp.run_gpfifo(&[(e.entry0, e.entry1)]);
    let got = cp.mem.read_f32s(C, j.m * j.n + 16).unwrap();
    assert!(
        got[j.m * j.n..].iter().all(|&x| x == SENTINEL),
        "wrote after C"
    );
    let out = match ran {
        Ok(()) => {
            assert!(cp.events.iter().any(|e| matches!(e, Event::Launch { .. })));
            assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
            Ok(got[..j.m * j.n].to_vec())
        }
        Err(e) => {
            assert!(
                got.iter().all(|&x| x == SENTINEL),
                "C touched by a refused launch"
            );
            Err(e)
        }
    };
    (out, cp.sm_configs.clone())
}

fn qmd_job(i: usize, j: &Job) -> ComputeQmd {
    ComputeQmd {
        grid: VARIANTS[i].grid(&args(j)),
        ..qmd_for(i)
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// T230e: the 10 probe runs give the witness's bits (and the oracle's), with the step recorded.
#[test]
fn t230e_probe_same_bits() {
    assert_eq!(PROBE.len(), 10);
    assert_eq!(
        (PROBE[0].variant, PROBE[0].step_bytes),
        (0, 8 * KIB),
        "the first run is today's launch"
    );
    // A scaled qkv-like job (same flags; full and partial tiles of every geometry) — the real qkv shape
    // (1024 × 2304 × 768) is for the board.
    for (m, n, k, seed) in [(256, 384, 128, 1), (200, 300, 72, 2)] {
        let j = job(m, n, k, seed);
        let mut oracle = vec![0.0; m * n];
        gpu_compute::reference::matmul_t_fma_ftz(
            &j.a,
            &j.b,
            Some(&j.bias),
            &mut oracle,
            m,
            k,
            n,
            false,
            true,
        )
        .unwrap();
        let witness = {
            let q0 = qmd_job(0, &j);
            let (c0, sm) = run_with(
                0,
                &j,
                &q0.to_bytes_with_shared(&VARIANTS[0].shared()).unwrap(),
            );
            assert_eq!(sm, vec![[3, 3, 17]]);
            c0.unwrap()
        };
        assert_eq!(bits(&witness), bits(&oracle));
        for p in PROBE {
            let s = VARIANTS[p.variant].shared();
            let qb = qmd_job(p.variant, &j)
                .to_bytes_with_sm_config(&s, p.step_bytes)
                .unwrap();
            let (got, sm) = run_with(p.variant, &j, &qb);
            let u = p.step_bytes / 4096 + 1;
            assert_eq!(sm, vec![[u, u, 17]], "{p:?}");
            assert_eq!(bits(&got.unwrap()), bits(&witness), "{p:?} {m}x{n}x{k}");
        }
    }
    println!(
        "kt t230e: 10 rulari ale probei (3 programe, trepte 8-64 KiB) = bitii martorului pe 2 forme; SIMULARE"
    );
}

/// T230f: the model refuses TARGET below the program's memory, MIN > TARGET, TARGET > MAX; C untouched.
#[test]
fn t230f_model_refuses_wrong_steps() {
    let j = job(128, 128, 32, 3);
    let i = 0;
    let base = qmd_job(i, &j)
        .encode_with_shared(&VARIANTS[i].shared())
        .unwrap();
    let words = |m: [u32; 64]| m.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
    let set = |min: u32, target: u32, max: u32| {
        let mut m = base;
        q::NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE
            .set(&mut m, min)
            .unwrap();
        q::NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE
            .set(&mut m, target)
            .unwrap();
        q::NVC6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE
            .set(&mut m, max)
            .unwrap();
        words(m)
    };
    // 8 KiB of shared memory: a 4 KiB step (units 2) is too small; MIN > TARGET; TARGET > MAX.
    for (min, target, max) in [(2, 2, 17), (5, 3, 17), (3, 25, 17)] {
        let (r, sm) = run_with(i, &j, &set(min, target, max));
        assert!(r.is_err(), "{min} {target} {max}");
        assert!(sm.is_empty());
    }
    // Accepted: MIN ≤ TARGET ≤ MAX, the step ≥ 8 KiB.
    let (r, sm) = run_with(i, &j, &set(3, 17, 17));
    assert!(r.is_ok());
    assert_eq!(sm, vec![[3, 17, 17]]);
    assert_eq!(words(base).len(), 256);
}
