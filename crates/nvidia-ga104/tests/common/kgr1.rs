//! The KGR1 control vectors (hub `integrare/kronos/vectori/`, Kronos S39/C-24), shared by the tests and by the host
//! tool `examples/analiza_i14.rs`.
//!
//! The hub is private and this repo is public, so the vectors are NOT in the repo. They are prepared locally from a
//! clone of the hub by `tools/vectori-kgr1/pregateste.py` (into `target/vectori-kgr1/`, never committed) and read at
//! run time from the folder in `KGR1_VECTORI`. The files: a tiny Kronos (2 layers, 4 heads, n_emb 32, vocabulary
//! 16 384) with RANDOM weights, and what Kronos's own engine without PyTorch gives on three texts.
#![allow(dead_code)]

use gpu_compute::layer::Shape;
use nvidia_ga104::kronos_layer::OwnedLayer;
use nvidia_ga104::kronos_model::OwnedModel;

pub const VECTORS_NEED: &str = "vectorii KGR1 din hub: python3 -I tools/vectori-kgr1/pregateste.py <hub>/integrare/kronos/vectori target/vectori-kgr1, apoi KGR1_VECTORI=$PWD/target/vectori-kgr1 cargo test --release -p nvidia-ga104 --test sim_i14_token kgr1 -- --ignored";

pub struct Files {
    pub kgr: Vec<u8>,
    pub txt: String,
    pub logits: Vec<u8>,
}

pub fn files() -> Files {
    let dir = std::env::var("KGR1_VECTORI")
        .unwrap_or_else(|_| panic!("lipseste KGR1_VECTORI: {VECTORS_NEED}"));
    let read = |n: &str| {
        std::fs::read(format!("{dir}/{n}"))
            .unwrap_or_else(|e| panic!("{dir}/{n}: {e}; {VECTORS_NEED}"))
    };
    Files {
        kgr: read("kronos_mic_q8_0.kgr"),
        txt: String::from_utf8(read("kronos_mic_q8_0.txt")).unwrap(),
        logits: read("kronos_mic_q8_0_logits.f32"),
    }
}

/// The hub README's tolerance for the Q8_0 vectors (absolute, last position).
pub const VECTOR_TOL: f32 = 1e-3;

/// A minimal KGR1 reader (hub `format_greutati.md`), host-only, for the test file.
pub fn read_kgr1(f: &[u8]) -> OwnedModel {
    use std::collections::HashMap;
    let mut o = 0usize;
    let mut take = |n: usize| {
        let s = &f[o..o + n];
        o += n;
        s
    };
    assert_eq!(take(4), b"KGR1");
    let u32_ = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    let u64_ = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap());
    assert_eq!(u32_(take(4)), 1);
    let (nkv, nt) = (u64_(take(8)), u64_(take(8)));
    let mut ints = HashMap::new();
    let mut eps = 0.0f64;
    for _ in 0..nkv {
        let n = u64_(take(8)) as usize;
        let k = String::from_utf8(take(n).to_vec()).unwrap();
        match u32_(take(4)) {
            1 => {
                ints.insert(k, i64::from_le_bytes(take(8).try_into().unwrap()));
            }
            2 => {
                let v = f64::from_le_bytes(take(8).try_into().unwrap());
                if k == "ln_eps" {
                    eps = v;
                }
            }
            3 => {
                let n = u64_(take(8)) as usize;
                let v = take(n);
                if k == "arhitectura" {
                    assert_eq!(v, b"kronos_gpt");
                }
            }
            other => panic!("fel necunoscut {other}"),
        }
    }
    let mut tensors = HashMap::new();
    for _ in 0..nt {
        let n = u64_(take(8)) as usize;
        let name = String::from_utf8(take(n).to_vec()).unwrap();
        let nd = u32_(take(4)) as usize;
        let dims: Vec<usize> = (0..nd).map(|_| u64_(take(8)) as usize).collect();
        let kind = u32_(take(4));
        let off = u64_(take(8)) as usize;
        tensors.insert(name, (dims, kind, off));
    }
    let data = o.div_ceil(32) * 32;
    let bytes = |name: &str, kind: u32| -> &[u8] {
        let (dims, k, off) = &tensors[name];
        assert_eq!(*k, kind, "{name}");
        let n = match kind {
            0 => 4 * dims.iter().product::<usize>(),
            _ => dims[0] * dims[1] / 32 * 36,
        };
        &f[data + off..data + off + n]
    };
    let f32s = |name: &str| -> Vec<f32> {
        bytes(name, 0)
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let q8 = |name: &str| bytes(name, 1).to_vec();
    assert_eq!(ints["cap_legat"], 1);
    let n_layers = ints["n_straturi"] as usize;
    OwnedModel {
        shape: Shape {
            e: ints["n_emb"] as usize,
            heads: ints["n_capete"] as usize,
            eps: eps as f32,
        },
        vocab: ints["vocabular"] as usize,
        context: ints["context"] as usize,
        emb_tok: q8("emb_token.weight"),
        emb_pos: q8("emb_pozitie.weight"),
        layers: (0..n_layers)
            .map(|i| {
                let p = |s: &str| format!("straturi.{i}.{s}");
                OwnedLayer {
                    ln1_g: f32s(&p("ln1.weight")),
                    ln1_b: f32s(&p("ln1.bias")),
                    wqkv: q8(&p("atentie.in_proj_weight")),
                    bqkv: f32s(&p("atentie.in_proj_bias")),
                    wout: q8(&p("atentie.out_proj.weight")),
                    bout: f32s(&p("atentie.out_proj.bias")),
                    ln2_g: f32s(&p("ln2.weight")),
                    ln2_b: f32s(&p("ln2.bias")),
                    wfc: q8(&p("mlp.0.weight")),
                    bfc: f32s(&p("mlp.0.bias")),
                    wproj: q8(&p("mlp.2.weight")),
                    bproj: f32s(&p("mlp.2.bias")),
                }
            })
            .collect(),
        ln_g: f32s("ln.weight"),
        ln_b: f32s("ln.bias"),
    }
}

pub struct Vector {
    pub text: String,
    pub prompt: Vec<usize>,
    pub argmax: Vec<usize>,
    pub greedy: Vec<usize>,
    pub logits: Vec<f32>,
}

pub fn vectors(f: &Files, vocab: usize) -> Vec<Vector> {
    let nums = |l: &str| -> Vec<usize> { l.split(' ').map(|v| v.parse().unwrap()).collect() };
    let mut out: Vec<Vector> = Vec::new();
    for line in f.txt.lines() {
        let (k, v) = line.split_once(' ').unwrap();
        match k {
            "text" => out.push(Vector {
                text: v.to_string(),
                prompt: vec![],
                argmax: vec![],
                greedy: vec![],
                logits: vec![],
            }),
            "simboluri" => out.last_mut().unwrap().prompt = nums(v),
            "argmax" => out.last_mut().unwrap().argmax = nums(v),
            "greedy" => out.last_mut().unwrap().greedy = nums(v),
            _ => {}
        }
    }
    assert_eq!(f.logits.len(), 4 * vocab * out.len());
    for (i, t) in out.iter_mut().enumerate() {
        t.logits = f.logits[4 * vocab * i..4 * vocab * (i + 1)]
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
    }
    out
}

pub fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}
