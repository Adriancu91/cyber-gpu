//! C-157 point 3 ("the diet, part two"), SIMULARE / host only: what the OS's CPU does per token of `kronos --gpu`
//! (#513 shape, the glued list `Fused`, 88 launches + the copy of the logits), each part timed on THIS host.
//! Nothing touches a board; the absolute µs are the host's, not the OS's. The OS's own `kt median argmax (us)`
//! (85 µs on os-328) is printed as a scale (85 / host argmax) — DEDUS and too high for integer work: the OS kernel
//! is soft-float, so only float compares are that much slower there (`docs/C-157_P3_DIETA_DOI.md`).
//!
//!   cargo run --release -p nvidia-ga104 --example dieta_doi
//!
//! The parts follow `cyber-universe` `gpu_i14.rs` `BoardRunner::{submit, run_token}` (read only, `13582db`).

use gpu_compute::layer::Shape;
use gpu_compute::model as cm;
use nvidia_ga104::kronos_diag::{self as kd, SeqMode};
use nvidia_ga104::kronos_fast::WriteShadow;
use nvidia_ga104::kronos_layer::{self as kl, LayerVas, Prog, WeightVas};
use nvidia_ga104::kronos_model::{self as km, ModelVas};
use nvidia_ga104::mmu::VA_BASE;
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::transfer;
use std::hint::black_box;
use std::time::Instant;

const E: usize = 768;
const LAYERS: usize = 12;
const VOCAB: usize = 16_384;
const CONTEXT: usize = 512;
const ROUNDS: usize = 400;
/// The OS's `kt median argmax (us)` on os-328 (`gpu-jurnal-026.txt`, 0x55).
const OS_ARGMAX_US: f64 = 85.0;

fn vas() -> ModelVas {
    let mut next = VA_BASE + 0x1000_0000;
    let mut take = |bytes: u64| {
        let a = next;
        next += bytes.next_multiple_of(0x1000);
        a
    };
    let scratch = take(kl::scratch_bytes(E));
    let layers = (0..LAYERS)
        .map(|_| {
            let w = WeightVas {
                ln1_g: take(4 * E as u64),
                ln1_b: take(4 * E as u64),
                wqkv: take(3 * E as u64 * E as u64),
                bqkv: take(12 * E as u64),
                wout: take(E as u64 * E as u64),
                bout: take(4 * E as u64),
                ln2_g: take(4 * E as u64),
                ln2_b: take(4 * E as u64),
                wfc: take(4 * E as u64 * E as u64),
                bfc: take(16 * E as u64),
                wproj: take(4 * E as u64 * E as u64),
                bproj: take(4 * E as u64),
            };
            let kc = take(kl::cache_bytes(E));
            let vc = take(kl::cache_bytes(E));
            LayerVas::with_scratch(E, scratch, kc, vc, w).unwrap()
        })
        .collect();
    let fin = take(4 * (E + VOCAB) as u64);
    ModelVas {
        emb_tok: take(VOCAB as u64 * E as u64),
        emb_pos: take(CONTEXT as u64 * E as u64),
        ln_g: take(4 * E as u64),
        ln_b: take(4 * E as u64),
        layers,
        h_final: fin,
        logits: fin + 4 * E as u64,
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn fnv1a(mut h: u32, v: &[f32]) -> u32 {
    for x in v {
        for b in x.to_bits().to_le_bytes() {
            h ^= u32::from(b);
            h = h.wrapping_mul(0x0100_0193);
        }
    }
    h
}

fn prog_va(p: Prog) -> u64 {
    VA_BASE + 0x1_0000 * (p as u64)
}

fn main() {
    let s = Shape {
        e: E,
        heads: 12,
        eps: 1e-5,
    };
    let v = vas();
    let slots = VA_BASE + 0x0100_0000;
    let records = VA_BASE + 0x0200_0000;
    let down = kl::copy_launch(v.logits, VA_BASE + 0x0300_0000, 4 * VOCAB as u64).unwrap();
    let n = km::launches_per_token_fused(LAYERS) + 1;
    let mut shadow = WriteShadow::new(slots, 512 * n as u64).unwrap();
    let pn = kd::words(SeqMode::EachLaunch, n);
    let mut push_shadow = WriteShadow::new(0, 4 * pn as u64).unwrap();
    let ram: Vec<u8> = (0..4 * VOCAB)
        .map(|i| (i as u32).wrapping_mul(2_654_435_761).to_le_bytes()[i % 4] & 0x3f)
        .collect();
    let mut parts: [(&str, Vec<f64>); 10] = [
        (
            "1 lansarile construite (token_launches_fused + copiere)",
            vec![],
        ),
        ("2 QMD-urile codate (la.qmd x89)", vec![]),
        (
            "3 umbra lansarilor (changes + remember, cbuf + QMD)",
            vec![],
        ),
        ("4 pushbuffer-ul codat (kd::encode) + umbra lui", vec![]),
        ("5 inregistrarile analizate (analyze_sequence, 89)", vec![]),
        ("6 logit-urile citite din RAM (64 KiB -> Vec<f32>)", vec![]),
        ("7 argmax (gpu_compute::model::argmax)", vec![]),
        ("8 amprenta FNV-1a a logit-urilor", vec![]),
        ("9 logit-urile copiate de 2 ori (last_logits, out)", vec![]),
        ("D2 argmax pe biti (argmax_bits, doar intregi)", vec![]),
    ];
    let mut print = 0x811c_9dc5u32;
    for round in 0..ROUNDS {
        let pos = 1 + round % (CONTEXT - 2);
        let tok = (round * 7919) % VOCAB;
        let t = Instant::now();
        let mut ls = km::token_launches_fused(&s, VOCAB, CONTEXT, &v, tok, pos).unwrap();
        ls.push(down.clone());
        parts[0].1.push(t.elapsed().as_secs_f64() * 1e6);
        let t = Instant::now();
        let qmds: Vec<[u8; 256]> = ls
            .iter()
            .enumerate()
            .map(|(j, la)| {
                let cb = slots + 512 * j as u64;
                let q = la.qmd(prog_va(la.prog), cb).unwrap();
                let mut o = [0u8; 256];
                o[..q.len()].copy_from_slice(&q);
                o
            })
            .collect();
        parts[1].1.push(t.elapsed().as_secs_f64() * 1e6);
        let t = Instant::now();
        let mut changed = 0usize;
        for (j, la) in ls.iter().enumerate() {
            let cb = slots + 512 * j as u64;
            let q = &qmds[j][..nvidia_rm570::QMD_SIZE];
            changed += shadow.changes(cb, &la.cbuf).unwrap().len();
            shadow.remember(cb, &la.cbuf).unwrap();
            changed += shadow.changes(cb + 256, q).unwrap().len();
            shadow.remember(cb + 256, q).unwrap();
        }
        black_box(changed);
        parts[2].1.push(t.elapsed().as_secs_f64() * 1e6);
        let t = Instant::now();
        let qv: Vec<u64> = (0..n).map(|j| slots + 512 * j as u64 + 256).collect();
        let first = 0x1_0000u32 + 0x100 * round as u32;
        let mut words = vec![0u32; pn];
        let mut p = PushBuf::new(&mut words);
        kd::encode(SeqMode::EachLaunch, &mut p, &qv, records, first).unwrap();
        let len = p.len();
        let bytes: Vec<u8> = words[..len].iter().flat_map(|w| w.to_le_bytes()).collect();
        black_box(push_shadow.changes(0, &bytes).unwrap().len());
        push_shadow.remember(0, &bytes).unwrap();
        parts[3].1.push(t.elapsed().as_secs_f64() * 1e6);
        let rec: Vec<u8> = (0..=n)
            .flat_map(|k| {
                let mut r = [0u8; 16];
                r[..4].copy_from_slice(&(first + k as u32).to_le_bytes());
                r[8..].copy_from_slice(&(1000 * k as u64).to_le_bytes());
                r
            })
            .collect();
        let t = Instant::now();
        black_box(transfer::analyze_sequence(&rec, n, first).unwrap());
        parts[4].1.push(t.elapsed().as_secs_f64() * 1e6);
        let t = Instant::now();
        let logits: Vec<f32> = black_box(&ram)
            .chunks(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        parts[5].1.push(t.elapsed().as_secs_f64() * 1e6);
        let t = Instant::now();
        black_box(cm::argmax(black_box(&logits)));
        parts[6].1.push(t.elapsed().as_secs_f64() * 1e6);
        let t = Instant::now();
        print = fnv1a(print, black_box(&logits));
        parts[7].1.push(t.elapsed().as_secs_f64() * 1e6);
        let t = Instant::now();
        let a = black_box(logits.clone());
        let b = black_box(logits.clone());
        parts[8].1.push(t.elapsed().as_secs_f64() * 1e6);
        drop((a, b));
        let bits: Vec<u32> = logits.iter().map(|x| x.to_bits()).collect();
        let t = Instant::now();
        black_box(cm::argmax_bits(black_box(&bits)));
        parts[9].1.push(t.elapsed().as_secs_f64() * 1e6);
    }
    black_box(print);
    let host_argmax = median(parts[6].1.clone()).max(0.01);
    let k = OS_ARGMAX_US / host_argmax;
    println!(
        "SIMULARE (gazda, fara placa): #513, lista lipita, {n} lansari pe simbol, {ROUNDS} simboluri"
    );
    println!(
        "calibrare: argmax gazda {host_argmax:.1} us = argmax OS {OS_ARGMAX_US} us -> x{k:.1} (DEDUS)"
    );
    let mut total = 0.0;
    for (name, v) in &parts[..9] {
        let m = median(v.clone());
        total += m;
        println!("  {name:<58} gazda {m:>7.1} us   OS ~{:>6.0} us", m * k);
    }
    println!(
        "  {:<58} gazda {total:>7.1} us   OS ~{:>6.0} us",
        "suma (1-9, azi)",
        total * k
    );
    let d2 = median(parts[9].1.clone());
    println!(
        "  {:<58} gazda {d2:>7.1} us   (OS: fara emularea float, ~ca pe gazda x1,6)",
        parts[9].0
    );
}
