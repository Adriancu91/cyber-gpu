//! C-83 I14, SIMULARE: whole tokens of a (small, synthetic) Kronos model on the work channel — embedding, every
//! layer, final LayerNorm, logits — `16·layers + 3` launches per token in one GPFIFO entry, then greedy
//! generation: the GPU model's tokens and logits against the CPU engine (`gpu_compute::model::step_cpu`). The
//! "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about the board or #513.

mod common;

use common::{Rig, get_bytes, gpu_runs, put_bytes, rig, span};
use gpu_compute::layer::Shape;
use gpu_compute::model::{self, Caches};
use gpu_core::error::GpuResult;
use gpu_sim::cmdproc::{
    ATTENTION_NEEDS, CommandProcessor, LAYERNORM_NEEDS, Needs, ProgramFn, Q8_MATVEC_NEEDS,
    attention_on_cpu, copy_b128_on_cpu, embed_on_cpu, gelu_on_cpu, layernorm_on_cpu,
    q8_matvec_on_cpu, vadd_on_cpu,
};
use nvidia_ga104::gsp::golden::{GoldenMemory, Mapping};
use nvidia_ga104::kronos_layer::{self as kl, LayerVas, PROGRAMS, Prog, WeightVas};
use nvidia_ga104::kronos_model::{self as km, ModelVas, OwnedModel};
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::transfer;

const FIRST: u32 = 0xa00;
/// Payloads grow with the position (the last record is the fence of the pushbuffer stretch).
fn first(pos: usize) -> u32 {
    FIRST + 0x1000 * pos as u32
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
    (Prog::Embed, embed_on_cpu, NONE),
];

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Gpu {
    progs: Vec<(Prog, Mapping)>,
    all: Vec<Mapping>,
    out: Vec<Mapping>,
    launches: Mapping,
    records: Mapping,
    vas: ModelVas,
    n: usize,
}

fn put(r: &mut Rig, all: &mut Vec<Mapping>, bytes: &[u8]) -> u64 {
    let m = r.mem.alloc_mapped(bytes.len() as u64, 0x1000).unwrap();
    put_bytes(r.mem.writer_mut(), m.pa, bytes);
    all.push(m);
    m.va
}

fn upload(r: &mut Rig, m: &OwnedModel) -> Gpu {
    let e = m.shape.e;
    let progs: Vec<(Prog, Mapping)> = PROGRAMS
        .iter()
        .map(|&p| {
            let mm = r.mem.alloc_mapped(p.code().len() as u64, 0x1000).unwrap();
            put_bytes(r.mem.writer_mut(), mm.pa, p.code());
            (p, mm)
        })
        .collect();
    let mut all = Vec::new();
    let emb_tok = put(r, &mut all, &m.emb_tok);
    let emb_pos = put(r, &mut all, &m.emb_pos);
    let ln_g = put(r, &mut all, &f32s(&m.ln_g));
    let ln_b = put(r, &mut all, &f32s(&m.ln_b));
    let scratch = r.mem.alloc_mapped(kl::scratch_bytes(e), 0x1000).unwrap();
    let mut out = vec![scratch];
    let mut layers = Vec::new();
    for l in &m.layers {
        let wv = WeightVas {
            ln1_g: put(r, &mut all, &f32s(&l.ln1_g)),
            ln1_b: put(r, &mut all, &f32s(&l.ln1_b)),
            wqkv: put(r, &mut all, &l.wqkv),
            bqkv: put(r, &mut all, &f32s(&l.bqkv)),
            wout: put(r, &mut all, &l.wout),
            bout: put(r, &mut all, &f32s(&l.bout)),
            ln2_g: put(r, &mut all, &f32s(&l.ln2_g)),
            ln2_b: put(r, &mut all, &f32s(&l.ln2_b)),
            wfc: put(r, &mut all, &l.wfc),
            bfc: put(r, &mut all, &f32s(&l.bfc)),
            wproj: put(r, &mut all, &l.wproj),
            bproj: put(r, &mut all, &f32s(&l.bproj)),
        };
        let kc = r.mem.alloc_mapped(kl::cache_bytes(e), 0x1000).unwrap();
        let vc = r.mem.alloc_mapped(kl::cache_bytes(e), 0x1000).unwrap();
        out.extend([kc, vc]);
        layers.push(LayerVas::with_scratch(e, scratch.va, kc.va, vc.va, wv).unwrap());
    }
    let fin = r
        .mem
        .alloc_mapped(4 * (e + m.vocab) as u64, 0x1000)
        .unwrap();
    out.push(fin);
    let n = km::launches_per_token(m.layers.len());
    let launches = r.mem.alloc_mapped(512 * n as u64, 0x1000).unwrap();
    let records = r.mem.alloc_mapped(16 * (n as u64 + 1), 0x1000).unwrap();
    let vas = ModelVas {
        emb_tok,
        emb_pos,
        ln_g,
        ln_b,
        layers,
        h_final: fin.va,
        logits: fin.va + 4 * e as u64,
    };
    Gpu {
        progs,
        all,
        out,
        launches,
        records,
        vas,
        n,
    }
}

fn token(r: &mut Rig, g: &Gpu, m: &OwnedModel, tok: usize, pos: usize) -> GpuResult<Vec<f32>> {
    let list = km::token_launches(&m.shape, m.vocab, m.context, &g.vas, tok, pos)?;
    let va = |p: Prog| g.progs.iter().find(|(q, _)| *q == p).unwrap().1.va;
    let mut qmds = Vec::new();
    let w = r.mem.writer_mut();
    for (k, l) in list.iter().enumerate() {
        let (cb_va, cb_pa) = (
            g.launches.va + 512 * k as u64,
            g.launches.pa + 512 * k as u64,
        );
        put_bytes(w, cb_pa, &l.cbuf);
        put_bytes(w, cb_pa + 256, &l.qmd(va(l.prog), cb_va)?);
        qmds.push(cb_va + 256);
    }
    put_bytes(w, g.records.pa, &vec![0; 16 * (g.n + 1)]);
    let mut words = vec![0u32; transfer::sequence_words(qmds.len())];
    let mut pb = PushBuf::new(&mut words);
    transfer::encode_sequence(&mut pb, &qmds, g.records.va, first(pos))?;
    let fence = g.records.va + 16 * g.n as u64;
    let done = (first(pos) + g.n as u32) as u64;
    r.wq.submit_fenced(r.mem.writer_mut(), &words, fence, done)?;
    let programs: Vec<(&[u8], ProgramFn, Needs)> =
        MODELS.iter().map(|&(p, f, n)| (p.code(), f, n)).collect();
    let loads: Vec<(u64, u64)> = g
        .progs
        .iter()
        .map(|(_, m)| m)
        .chain(&g.all)
        .chain(&g.out)
        .chain([&g.launches, &g.records])
        .map(span)
        .collect();
    let stores: Vec<(u64, u64)> = g.out.iter().chain([&g.records]).map(span).collect();
    let cp: CommandProcessor = gpu_runs(r, &programs, &loads, &stores)?;
    assert_eq!(cp.timestamps.len(), g.n + 1);
    let rec = get_bytes(r.mem.writer_mut(), g.records.pa, 16 * (g.n as u64 + 1));
    transfer::analyze_sequence(&rec, g.n, first(pos))?;
    // The driver read the last record: its pushbuffer stretch can be reused.
    let fence = g.records.va + 16 * g.n as u64;
    let last = u32::from_le_bytes(rec[16 * g.n..16 * g.n + 4].try_into().unwrap()) as u64;
    assert!(r.wq.retire(fence, last)? >= 1);
    assert_eq!(r.wq.push_in_flight(), 0);
    let fin = g.out.last().unwrap();
    Ok(get_bytes(
        r.mem.writer_mut(),
        fin.pa + 4 * m.shape.e as u64,
        4 * m.vocab as u64,
    )
    .chunks(4)
    .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
    .collect())
}

/// Prompt, then greedy tokens, on the GPU model and on the CPU engine side by side.
fn generate(m: &OwnedModel, prompt: &[usize], more: usize) -> (Vec<usize>, Vec<km::TokenVerdict>) {
    let mut r = rig();
    let g = upload(&mut r, m);
    let w = m.weights();
    let mut c = Caches::new(m.layers.len(), m.shape.e);
    let (mut seq, mut verdicts) = (prompt.to_vec(), Vec::new());
    for pos in 0..prompt.len() + more {
        let tok = seq[pos];
        let got = token(&mut r, &g, m, tok, pos).unwrap();
        let cpu = model::step_cpu(&w, &mut c, tok, pos).unwrap();
        verdicts.push(km::judge(&got, &cpu));
        if pos + 1 >= seq.len() && seq.len() < prompt.len() + more {
            seq.push(model::argmax(&got).unwrap());
        }
    }
    (seq, verdicts)
}

#[test]
fn greedy_tokens_of_a_small_model_match_the_cpu() {
    let m = km::test_model(
        Shape {
            e: 64,
            heads: 2,
            eps: 1e-5,
        },
        2,
        256,
        16,
        7,
    )
    .unwrap();
    let (seq, v) = generate(&m, &[5, 17, 3], 5);
    for (pos, x) in v.iter().enumerate() {
        assert!(x.passes(), "{}", x.line(pos));
    }
    // The CPU alone gives the same continuation.
    let w = m.weights();
    let mut c = Caches::new(2, 64);
    let mut cpu_seq = vec![5, 17, 3];
    for pos in 0..7 {
        let l = model::step_cpu(&w, &mut c, cpu_seq[pos], pos).unwrap();
        if pos + 1 >= cpu_seq.len() {
            cpu_seq.push(model::argmax(&l).unwrap());
        }
    }
    assert_eq!(seq, cpu_seq);
    assert_eq!(seq.len(), 8);
}

#[test]
fn kronos_width_one_token() {
    let m = km::test_model(
        Shape {
            e: 768,
            heads: 12,
            eps: 1e-5,
        },
        1,
        512,
        8,
        9,
    )
    .unwrap();
    let (_, v) = generate(&m, &[42], 1);
    for (pos, x) in v.iter().enumerate() {
        assert!(x.passes(), "{}", x.line(pos));
    }
}

/// A wrong embedding row (token index off by one) never passes.
#[test]
fn wrong_token_row_is_caught() {
    let m = km::test_model(
        Shape {
            e: 64,
            heads: 2,
            eps: 1e-5,
        },
        1,
        256,
        16,
        11,
    )
    .unwrap();
    let mut r = rig();
    let g = upload(&mut r, &m);
    let got = token(&mut r, &g, &m, 6, 0).unwrap();
    let mut c = Caches::new(1, 64);
    let cpu = model::step_cpu(&m.weights(), &mut c, 5, 0).unwrap();
    assert!(!km::judge(&got, &cpu).passes());
    // A token outside the vocabulary is refused before anything is written.
    assert!(token(&mut r, &g, &m, 256, 1).is_err());
    // A fence value that does not grow is refused (the stretch could never be told apart).
    let fence = g.records.va + 16 * g.n as u64;
    let same = (first(0) + g.n as u32) as u64;
    assert!(
        r.wq.submit_fenced(r.mem.writer_mut(), &[0], fence, same)
            .is_err()
    );
    assert!(
        r.wq.submit_fenced(r.mem.writer_mut(), &[0], 0x10, same + 1)
            .is_err(),
        "in afara VA"
    );
}
