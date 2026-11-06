//! C-83 I13, SIMULARE: one whole Kronos layer for successive positions on the work channel — 16 launches per
//! position (`kronos_layer::token_launches`) of our six NAK programs, each followed by a timestamp record, the
//! key/value caches growing in VRAM. The output x of every position is compared with the CPU engine
//! (`gpu_compute::layer::step_cpu`) in the tolerance fixed before the run. The "GPU" is `gpu_sim::cmdproc` with
//! the program models of `gpu_compute::layer`; nothing here proves anything about the board.

mod common;

use common::{Rig, get_bytes, gpu_runs, put_bytes, rig, span};
use gpu_compute::layer::{self, Shape};
use gpu_core::error::GpuResult;
use gpu_sim::cmdproc::{
    ATTENTION_NEEDS, CommandProcessor, LAYERNORM_NEEDS, Needs, ProgramFn, Q8_MATVEC_NEEDS,
    attention_on_cpu, copy_b128_on_cpu, gelu_on_cpu, layernorm_on_cpu, q8_matvec_on_cpu,
    vadd_on_cpu,
};
use nvidia_ga104::gsp::golden::{GoldenMemory, Mapping};
use nvidia_ga104::kronos_layer::{
    self as kl, LAUNCHES_PER_TOKEN, LayerVas, OwnedLayer, PROGRAMS, Prog, WeightVas,
};
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::transfer;

const FIRST: u32 = 0x900;

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Gpu {
    progs: Vec<(Prog, Mapping)>,
    weights: Vec<Mapping>,
    scratch: Mapping,
    kc: Mapping,
    vc: Mapping,
    launches: Mapping,
    records: Mapping,
    vas: LayerVas,
}

fn upload(r: &mut Rig, s: &Shape, l: &OwnedLayer) -> Gpu {
    let e = s.e;
    let progs: Vec<(Prog, Mapping)> = PROGRAMS
        .iter()
        .map(|&p| {
            (
                p,
                r.mem.alloc_mapped(p.code().len() as u64, 0x1000).unwrap(),
            )
        })
        .collect();
    let parts: Vec<Vec<u8>> = vec![
        f32s(&l.ln1_g),
        f32s(&l.ln1_b),
        l.wqkv.clone(),
        f32s(&l.bqkv),
        l.wout.clone(),
        f32s(&l.bout),
        f32s(&l.ln2_g),
        f32s(&l.ln2_b),
        l.wfc.clone(),
        f32s(&l.bfc),
        l.wproj.clone(),
        f32s(&l.bproj),
    ];
    let weights: Vec<Mapping> = parts
        .iter()
        .map(|b| r.mem.alloc_mapped(b.len() as u64, 0x1000).unwrap())
        .collect();
    let scratch = r.mem.alloc_mapped(kl::scratch_bytes(e), 0x1000).unwrap();
    let kc = r.mem.alloc_mapped(kl::cache_bytes(e), 0x1000).unwrap();
    let vc = r.mem.alloc_mapped(kl::cache_bytes(e), 0x1000).unwrap();
    let launches = r
        .mem
        .alloc_mapped(512 * LAUNCHES_PER_TOKEN as u64, 0x1000)
        .unwrap();
    let records = r.mem.alloc_mapped(0x1000, 0x1000).unwrap();
    let w = r.mem.writer_mut();
    for (p, m) in &progs {
        put_bytes(w, m.pa, p.code());
    }
    for (b, m) in parts.iter().zip(&weights) {
        put_bytes(w, m.pa, b);
    }
    // Junk (NaN) in the scratch and caches: anything not computed cannot pass.
    put_bytes(w, scratch.pa, &vec![0xff; scratch.size as usize]);
    let v = |i: usize| weights[i].va;
    let wv = WeightVas {
        ln1_g: v(0),
        ln1_b: v(1),
        wqkv: v(2),
        bqkv: v(3),
        wout: v(4),
        bout: v(5),
        ln2_g: v(6),
        ln2_b: v(7),
        wfc: v(8),
        bfc: v(9),
        wproj: v(10),
        bproj: v(11),
    };
    let vas = LayerVas::with_scratch(e, scratch.va, kc.va, vc.va, wv).unwrap();
    Gpu {
        progs,
        weights,
        scratch,
        kc,
        vc,
        launches,
        records,
        vas,
    }
}

fn prog_va(g: &Gpu, p: Prog) -> u64 {
    g.progs.iter().find(|(q, _)| *q == p).unwrap().1.va
}

/// Writes x, the 16 cbufs + QMDs of `pos`, clears the records, submits one GPFIFO entry.
fn submit(
    r: &mut Rig,
    g: &Gpu,
    s: &Shape,
    pos: usize,
    x: &[f32],
    tamper: impl Fn(&mut Vec<kl::Launch>),
) {
    let mut launches = kl::token_launches(s, &g.vas, pos).unwrap();
    tamper(&mut launches);
    let x_pa = g.scratch.pa + (g.vas.x - g.scratch.va);
    let w = r.mem.writer_mut();
    put_bytes(w, x_pa, &f32s(x));
    let mut qmds = Vec::new();
    for (k, l) in launches.iter().enumerate() {
        let cb = (
            g.launches.va + 512 * k as u64,
            g.launches.pa + 512 * k as u64,
        );
        let qm = (cb.0 + 256, cb.1 + 256);
        put_bytes(w, cb.1, &l.cbuf);
        put_bytes(w, qm.1, &l.qmd(prog_va(g, l.prog), cb.0).unwrap());
        qmds.push(qm.0);
    }
    put_bytes(w, g.records.pa, &[0; 16 * (LAUNCHES_PER_TOKEN + 1)]);
    let mut words = vec![0u32; transfer::sequence_words(qmds.len())];
    let mut pb = PushBuf::new(&mut words);
    transfer::encode_sequence(&mut pb, &qmds, g.records.va, FIRST).unwrap();
    r.wq.submit(r.mem.writer_mut(), &words).unwrap();
}

const MODELS: [(Prog, ProgramFn, Needs); 6] = [
    (
        Prog::Copy,
        copy_b128_on_cpu,
        Needs {
            shared_bytes: 0,
            barriers: 0,
        },
    ),
    (Prog::Q8Matvec, q8_matvec_on_cpu, Q8_MATVEC_NEEDS),
    (
        Prog::Vadd,
        vadd_on_cpu,
        Needs {
            shared_bytes: 0,
            barriers: 0,
        },
    ),
    (Prog::LayerNorm, layernorm_on_cpu, LAYERNORM_NEEDS),
    (Prog::Attention, attention_on_cpu, ATTENTION_NEEDS),
    (
        Prog::Gelu,
        gelu_on_cpu,
        Needs {
            shared_bytes: 0,
            barriers: 0,
        },
    ),
];

fn gpu(r: &mut Rig, g: &Gpu) -> GpuResult<CommandProcessor> {
    let programs: Vec<(&[u8], ProgramFn, Needs)> =
        MODELS.iter().map(|&(p, f, n)| (p.code(), f, n)).collect();
    let loads: Vec<(u64, u64)> = g
        .progs
        .iter()
        .map(|(_, m)| m)
        .chain(&g.weights)
        .chain([&g.scratch, &g.kc, &g.vc, &g.launches, &g.records])
        .map(span)
        .collect();
    let stores = [span(&g.scratch), span(&g.kc), span(&g.vc), span(&g.records)];
    // The caches start as junk only where nothing was written: load what VRAM has (poison = junk).
    gpu_runs(r, &programs, &loads, &stores)
}

fn read_x(r: &mut Rig, g: &Gpu, e: usize) -> Vec<f32> {
    let pa = g.scratch.pa + (g.vas.x - g.scratch.va);
    get_bytes(r.mem.writer_mut(), pa, 4 * e as u64)
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Runs `positions` through the layer on the GPU model and on the CPU engine; returns the verdicts.
fn run(s: Shape, positions: usize, seed: u32) -> Vec<kl::LayerVerdict> {
    let l = kl::test_layer(s.e, seed).unwrap();
    let mut r = rig();
    let g = upload(&mut r, &s, &l);
    let (mut kc, mut vc) = (
        vec![0.0f32; layer::MAX_POS * s.e],
        vec![0.0f32; layer::MAX_POS * s.e],
    );
    let mut out = Vec::new();
    for pos in 0..positions {
        let x = kl::test_input(s.e, seed, pos);
        submit(&mut r, &g, &s, pos, &x, |_| {});
        let cp = gpu(&mut r, &g).unwrap();
        assert_eq!(cp.timestamps.len(), LAUNCHES_PER_TOKEN + 1);
        let rec = get_bytes(
            r.mem.writer_mut(),
            g.records.pa,
            16 * (LAUNCHES_PER_TOKEN as u64 + 1),
        );
        assert_eq!(
            transfer::analyze_sequence(&rec, LAUNCHES_PER_TOKEN, FIRST)
                .unwrap()
                .len(),
            16
        );
        let mut want = x.clone();
        layer::step_cpu(&s, &l.weights(), &mut want, &mut kc, &mut vc, pos).unwrap();
        out.push(kl::judge(&read_x(&mut r, &g, s.e), &want));
    }
    out
}

#[test]
fn small_layer_over_six_positions() {
    let v = run(
        Shape {
            e: 64,
            heads: 2,
            eps: 1e-5,
        },
        6,
        1,
    );
    for (pos, x) in v.iter().enumerate() {
        assert!(x.passes(), "{}", x.line(pos));
    }
}

#[test]
fn kronos_sized_layer_two_positions() {
    let v = run(
        Shape {
            e: 768,
            heads: 12,
            eps: 1e-5,
        },
        2,
        2,
    );
    for (pos, x) in v.iter().enumerate() {
        assert!(x.passes(), "{}", x.line(pos));
        // The models are close to the CPU, far inside the tolerance.
        if let kl::LayerVerdict::Trece { max_abs } = x {
            assert!(*max_abs < 1e-5, "{max_abs}");
        }
    }
}

type Tamper = Box<dyn Fn(&mut Vec<kl::Launch>)>;

/// Mistakes planted in the launch list never pass.
#[test]
fn planted_mistakes_never_pass() {
    let s = Shape {
        e: 64,
        heads: 2,
        eps: 1e-5,
    };
    let l = kl::test_layer(64, 3).unwrap();
    let cases: Vec<(&str, Tamper)> = vec![
        (
            "fara rezidual dupa atentie",
            Box::new(|v: &mut Vec<kl::Launch>| {
                // x1 = 0 + ob instead of x + ob: point "a" at the bias of out (wrong input).
                let ob = v[7].cbuf[0x38..0x40].to_vec();
                v[8].cbuf[0x28..0x30].copy_from_slice(&ob);
            }),
        ),
        (
            "atentia vede o pozitie in minus",
            Box::new(|v: &mut Vec<kl::Launch>| {
                let n = u32::from_le_bytes(v[5].cbuf[0x48..0x4c].try_into().unwrap());
                v[5].cbuf[0x48..0x4c].copy_from_slice(&(n - 1).max(1).to_le_bytes());
            }),
        ),
        (
            "gelu lipsa",
            Box::new(|v: &mut Vec<kl::Launch>| {
                // The projection reads fb (before GELU) instead of g.
                let fb = v[11].cbuf[0x38..0x40].to_vec();
                v[13].cbuf[0x30..0x38].copy_from_slice(&fb);
            }),
        ),
    ];
    for (what, tamper) in cases {
        let mut r = rig();
        let g = upload(&mut r, &s, &l);
        let (mut kc, mut vc) = (
            vec![0.0f32; layer::MAX_POS * 64],
            vec![0.0f32; layer::MAX_POS * 64],
        );
        let mut passed_all = true;
        for pos in 0..3 {
            let x = kl::test_input(64, 3, pos);
            submit(&mut r, &g, &s, pos, &x, &tamper);
            gpu(&mut r, &g).unwrap();
            let mut want = x.clone();
            layer::step_cpu(&s, &l.weights(), &mut want, &mut kc, &mut vc, pos).unwrap();
            passed_all &= kl::judge(&read_x(&mut r, &g, 64), &want).passes();
        }
        assert!(!passed_all, "greseala „{what}” neprinsa");
    }
}
