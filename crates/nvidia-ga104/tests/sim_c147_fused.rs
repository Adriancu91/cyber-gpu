//! C-147 step 2 (etapa 3a), SIMULARE: the glued layer (`kronos_layer::token_launches_fused`, 7 launches with the
//! new `q8_fused` program) against the 16 launches of I13, position after position on the simulated work channel:
//! the layer output x and the key/value caches must be the same **bit for bit** (every value goes through the same
//! operations in the same order), and inside the I13 tolerance of the CPU engine. Mistakes planted in the glued
//! list never pass. The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about
//! the board.

mod common;

use common::token_sim::TokenSim;
use common::{Rig, get_bytes, gpu_runs, put_bytes, rig, span};
use gpu_compute::layer::{self, Shape};
use gpu_core::error::GpuResult;
use gpu_sim::cmdproc::{
    ATTENTION_NEEDS, CommandProcessor, LAYERNORM_NEEDS, Needs, ProgramFn, Q8_FUSED_NEEDS,
    Q8_MATVEC_NEEDS, attention_on_cpu, copy_b128_on_cpu, gelu_on_cpu, layernorm_on_cpu,
    q8_fused_on_cpu, q8_matvec_on_cpu, vadd_on_cpu,
};
use nvidia_ga104::gsp::golden::{GoldenMemory, Mapping};
use nvidia_ga104::kronos_diag::{SeqMode, TokenIo};
use nvidia_ga104::kronos_layer::{
    self as kl, LAUNCHES_PER_TOKEN, LAUNCHES_PER_TOKEN_FUSED, LayerVas, OwnedLayer, PROGRAMS_FUSED,
    Prog, WeightVas,
};
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::kronos_perf as kp;
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::q8_fused;
use nvidia_ga104::transfer;

const FIRST: u32 = 0xa00;

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
    let progs: Vec<(Prog, Mapping)> = PROGRAMS_FUSED
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
    let launches = r.mem.alloc_mapped(512 * 16, 0x1000).unwrap();
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

/// Writes x and the launches' cbufs + QMDs, clears the records, submits one GPFIFO entry.
fn submit(r: &mut Rig, g: &Gpu, x: &[f32], launches: &[kl::Launch]) {
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

const NONE: Needs = Needs {
    shared_bytes: 0,
    barriers: 0,
};

const MODELS: [(Prog, ProgramFn, Needs); 7] = [
    (Prog::Copy, copy_b128_on_cpu, NONE),
    (Prog::Q8Matvec, q8_matvec_on_cpu, Q8_MATVEC_NEEDS),
    (Prog::Vadd, vadd_on_cpu, NONE),
    (Prog::LayerNorm, layernorm_on_cpu, LAYERNORM_NEEDS),
    (Prog::Attention, attention_on_cpu, ATTENTION_NEEDS),
    (Prog::Gelu, gelu_on_cpu, NONE),
    (Prog::Q8Fused, q8_fused_on_cpu, Q8_FUSED_NEEDS),
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
    gpu_runs(r, &programs, &loads, &stores)
}

fn read(r: &mut Rig, m: &Mapping, va: u64, n: usize) -> Vec<u32> {
    get_bytes(r.mem.writer_mut(), m.pa + (va - m.va), 4 * n as u64)
        .chunks(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// One position through the layer: the output x and the caches up to `pos` (bits), and the record count.
struct Out {
    x: Vec<u32>,
    kc: Vec<u32>,
    vc: Vec<u32>,
    records: usize,
}

fn one(r: &mut Rig, g: &Gpu, e: usize, pos: usize, x: &[f32], launches: &[kl::Launch]) -> Out {
    submit(r, g, x, launches);
    let cp = gpu(r, g).unwrap();
    let rec = get_bytes(
        r.mem.writer_mut(),
        g.records.pa,
        16 * (launches.len() as u64 + 1),
    );
    let records = transfer::analyze_sequence(&rec, launches.len(), FIRST)
        .unwrap()
        .len();
    assert_eq!(cp.timestamps.len(), launches.len() + 1);
    Out {
        x: read(r, &g.scratch, g.vas.x, e),
        kc: read(r, &g.kc, g.vas.kc, (pos + 1) * e),
        vc: read(r, &g.vc, g.vas.vc, (pos + 1) * e),
        records,
    }
}

type Tamper = fn(&mut Vec<kl::Launch>);

/// `positions` through the layer, glued and not, on two rigs; returns how many positions were identical and the
/// verdicts of the glued one against the CPU engine.
fn compare(
    s: Shape,
    positions: usize,
    seed: u32,
    tamper: Tamper,
) -> (usize, Vec<kl::LayerVerdict>) {
    let l = kl::test_layer(s.e, seed).unwrap();
    let (mut ra, mut rb) = (rig(), rig());
    let (ga, gb) = (upload(&mut ra, &s, &l), upload(&mut rb, &s, &l));
    let (mut kc, mut vc) = (
        vec![0.0f32; layer::MAX_POS * s.e],
        vec![0.0f32; layer::MAX_POS * s.e],
    );
    let (mut same, mut verdicts) = (0, Vec::new());
    for pos in 0..positions {
        let x = kl::test_input(s.e, seed, pos);
        let old = kl::token_launches(&s, &ga.vas, pos).unwrap();
        let mut new = kl::token_launches_fused(&s, &gb.vas, pos).unwrap();
        assert_eq!(
            (old.len(), new.len()),
            (LAUNCHES_PER_TOKEN, LAUNCHES_PER_TOKEN_FUSED)
        );
        tamper(&mut new);
        let a = one(&mut ra, &ga, s.e, pos, &x, &old);
        let b = one(&mut rb, &gb, s.e, pos, &x, &new);
        assert_eq!((a.records, b.records), (16, 7));
        if a.x == b.x && a.kc == b.kc && a.vc == b.vc {
            same += 1;
        }
        let mut want = x.clone();
        layer::step_cpu(&s, &l.weights(), &mut want, &mut kc, &mut vc, pos).unwrap();
        let got: Vec<f32> = b.x.iter().map(|&w| f32::from_bits(w)).collect();
        verdicts.push(kl::judge(&got, &want));
    }
    (same, verdicts)
}

#[test]
fn fused_layer_same_bits_small() {
    let s = Shape {
        e: 64,
        heads: 2,
        eps: 1e-5,
    };
    let (same, v) = compare(s, 6, 1, |_| {});
    assert_eq!(same, 6);
    for (pos, x) in v.iter().enumerate() {
        assert!(x.passes(), "{}", x.line(pos));
    }
}

#[test]
fn fused_layer_same_bits_kronos_sized() {
    let s = Shape {
        e: 768,
        heads: 12,
        eps: 1e-5,
    };
    let (same, v) = compare(s, 2, 2, |_| {});
    assert_eq!(same, 2);
    for (pos, x) in v.iter().enumerate() {
        assert!(x.passes(), "{}", x.line(pos));
    }
}

#[test]
fn fused_counts_and_programs() {
    // #513: 12 layers -> 87 launches (+ the copy of the logits = 88) instead of 195 (196).
    assert_eq!(km::launches_per_token(12), 195);
    assert_eq!(km::launches_per_token_fused(12), 87);
    assert_eq!(PROGRAMS_FUSED.len(), 8);
    assert_eq!(Prog::Q8Fused.code(), q8_fused::Q8_FUSED);
    assert_eq!(Prog::Q8Fused.registers(), q8_fused::REGISTERS);
    // The old programs keep their 19 registers and their QMD bytes.
    for p in kl::PROGRAMS {
        assert_eq!(p.registers(), 19);
    }
    // `token.comp` has no glued operation: its work list refuses one.
    let s = Shape {
        e: 64,
        heads: 2,
        eps: 1e-5,
    };
    let l = kl::test_layer(64, 4).unwrap();
    let mut r = rig();
    let g = upload(&mut r, &s, &l);
    let list = kl::token_launches_fused(&s, &g.vas, 0).unwrap();
    assert!(nvidia_ga104::kronos_token::work_list(&list).is_err());
    let progs: Vec<Prog> = list.iter().map(|l| l.prog).collect();
    assert_eq!(
        progs,
        [
            Prog::LayerNorm,
            Prog::Q8Fused,
            Prog::Attention,
            Prog::Q8Fused,
            Prog::LayerNorm,
            Prog::Q8Fused,
            Prog::Q8Fused
        ]
    );
}

fn set_u32(l: &mut kl::Launch, at: usize, v: u32) {
    l.cbuf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

/// Mistakes planted in the glued list never pass (the comparison with the 16 launches and the CPU verdict).
#[test]
fn planted_mistakes_never_pass() {
    let s = Shape {
        e: 64,
        heads: 2,
        eps: 1e-5,
    };
    let cases: [(&str, Tamper); 4] = [
        ("fara rezidual dupa atentie", |v| {
            set_u32(&mut v[3], q8_fused::CBUF_FLAGS, 0)
        }),
        ("fara gelu", |v| set_u32(&mut v[5], q8_fused::CBUF_FLAGS, 0)),
        ("cheile si valorile nu intra in cache", |v| {
            set_u32(&mut v[1], q8_fused::CBUF_FLAGS, 0)
        }),
        ("fara termenul liber al proiectiei", |v| {
            // bias -> the bias of out (another vector)
            let b = v[3].cbuf[q8_fused::CBUF_BIAS..q8_fused::CBUF_BIAS + 8].to_vec();
            v[6].cbuf[q8_fused::CBUF_BIAS..q8_fused::CBUF_BIAS + 8].copy_from_slice(&b);
        }),
    ];
    for (what, tamper) in cases {
        let (same, v) = compare(s, 3, 3, tamper);
        assert!(
            same < 3 && !v.iter().all(|x| x.passes()),
            "greseala „{what}” neprinsa"
        );
    }
}

/// A whole (small) model, token after token, as the OS sends it (`TokenIo::submit`, one launch per record): the 88
/// glued launches give the same logits and caches, bit for bit, as the 196; the `kp` kinds of the glued list are
/// its launches, one to one.
#[test]
fn fused_model_same_logits() {
    let m = km::test_model(
        Shape {
            e: 64,
            heads: 4,
            eps: 1e-5,
        },
        3,
        256,
        16,
        11,
    )
    .unwrap();
    let (mut a, mut b) = (TokenSim::new(&m, 0), TokenSim::new(&m, 0));
    for (pos, tok) in [5usize, 200, 17, 5].into_iter().enumerate() {
        let old = km::token_launches(&m.shape, m.vocab, m.context, &a.vas, tok, pos).unwrap();
        let new = km::token_launches_fused(&m.shape, m.vocab, m.context, &b.vas, tok, pos).unwrap();
        assert_eq!(
            (old.len(), new.len()),
            (km::launches_per_token(3), km::launches_per_token_fused(3))
        );
        a.submit(&old, 0, SeqMode::EachLaunch).unwrap();
        b.submit(&new, 0, SeqMode::EachLaunch).unwrap();
        let ra = a.read(&[(a.vas.logits, m.vocab)]).unwrap();
        let rb = b.read(&[(b.vas.logits, m.vocab)]).unwrap();
        assert_eq!(ra, rb, "pozitia {pos}");
        assert!(ra[0].iter().all(|&w| f32::from_bits(w).is_finite()));
        for l in 0..3 {
            let (ka, kb) = (a.vas.layers[l].kc, b.vas.layers[l].kc);
            let (va, vb) = (a.vas.layers[l].vc, b.vas.layers[l].vc);
            let n = (pos + 1) * m.shape.e;
            assert_eq!(
                a.read(&[(ka, n), (va, n)]).unwrap(),
                b.read(&[(kb, n), (vb, n)]).unwrap()
            );
        }
        // The kp kinds of the glued list: one to one with what was sent.
        let kinds = kp::token_kinds_fused(3, true, false);
        assert_eq!(kinds.len(), new.len());
        for (k, l) in kinds.iter().zip(&new) {
            let want = match k {
                kp::Kind::Qkv | kp::Kind::Out | kp::Kind::Fc | kp::Kind::Proj => Prog::Q8Fused,
                _ => k.prog(),
            };
            assert_eq!(want, l.prog, "{k:?}");
        }
    }
}
