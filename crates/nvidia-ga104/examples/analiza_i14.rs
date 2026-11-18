//! C-139 (after os-305): the I14 analysis on the host — SIMULARE and arithmetic only, never a board.
//!
//!   python3 -I tools/vectori-kgr1/pregateste.py <hub>/integrare/kronos/vectori target/vectori-kgr1
//!   KGR1_VECTORI=$PWD/target/vectori-kgr1 cargo run --release -p nvidia-ga104 --example analiza_i14 [-- <gpu-jurnal.txt>]
//!
//! With Kronos's control model (`mic_q8_0.kgr`, prepared from the private hub; not in this repo) it prints:
//! 1. the gap between the first and the second logit at every control position, on our CPU engine — how far a
//!    rounding difference is from changing the greedy token;
//! 2. what the first token of the first control text (the one `i14-simbol` failed at in os-305) gives on the GPU
//!    **model** under each hypothesis: the greedy token and the first 8 logits — the signatures a board journal
//!    can be matched with;
//! 3. with a stick journal: the I14 diagnosis rows it holds (`kronos_diag::parse_journal`), in plain Romanian, and
//!    which hypothesis the board's first 8 logits match.
//!
//! Without `KGR1_VECTORI` only part 3 runs (a journal must be given).

#[path = "../tests/common/mod.rs"]
mod common;

use common::kgr1::{read_kgr1, vectors};
use common::token_sim::TokenSim;
use gpu_compute::model::{self, Caches};
use gpu_core::error::GpuResult;
use gpu_sim::cmdproc::{Launch, ProgramFn, embed_on_cpu};
use nvidia_ga104::kronos_diag::{self as kd, SeqMode};
use nvidia_ga104::kronos_layer::Prog;
use nvidia_ga104::kronos_model::OwnedModel;

/// The embedding without the position row (x = the token row only).
fn embed_without_position(l: &mut Launch<'_>) -> GpuResult<()> {
    let tok = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let x = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let n = l.mem.read_u32(l.cbuf0 + 0x40)? as usize;
    let mut row = vec![0u8; n / 32 * 36];
    l.mem.read(tok, &mut row)?;
    let zero = vec![0u8; row.len()];
    l.mem.write_f32s(x, &model::embed_gpu(&row, &zero, n));
    Ok(())
}

/// The embedding with the 8-bit numbers taken as unsigned (no sign extension).
fn embed_unsigned(l: &mut Launch<'_>) -> GpuResult<()> {
    let tok = l.mem.read_u64(l.cbuf0 + 0x28)?;
    let pos = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let x = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let n = l.mem.read_u32(l.cbuf0 + 0x40)? as usize;
    let bytes = n / 32 * 36;
    let (mut t, mut p) = (vec![0u8; bytes], vec![0u8; bytes]);
    l.mem.read(tok, &mut t)?;
    l.mem.read(pos, &mut p)?;
    let v = |row: &[u8], i: usize| {
        let b = &row[i / 32 * 36..];
        f32::from_le_bytes([b[0], b[1], b[2], b[3]]) * b[4 + i % 32] as f32
    };
    let out: Vec<f32> = (0..n).map(|i| v(&t, i) + v(&p, i)).collect();
    l.mem.write_f32s(x, &out);
    Ok(())
}

struct Signature {
    name: &'static str,
    logits: Vec<f32>,
}

fn run(m: &OwnedModel, tok: usize, line: u64, embed: ProgramFn, mode: SeqMode) -> Vec<f32> {
    let mut s = TokenSim::new(m, line);
    s.models.iter_mut().find(|x| x.0 == Prog::Embed).unwrap().1 = embed;
    s.run_token(tok, 0, mode).unwrap()
}

fn signatures(m: &OwnedModel, tok: usize) -> Vec<Signature> {
    let sig = |name, logits| Signature { name, logits };
    vec![
        sig(
            "totul corect (modelul programelor, memorie coerenta)",
            run(m, tok, 0, embed_on_cpu, SeqMode::Once),
        ),
        sig(
            "IPOTEZA: date vechi citite in aceeasi trimitere (cache necoerent, linii de 4..128 B)",
            run(m, tok, 128, embed_on_cpu, SeqMode::Once),
        ),
        sig(
            "date vechi, linii de 256 B (vectori vecini in aceeasi linie)",
            run(m, tok, 256, embed_on_cpu, SeqMode::Once),
        ),
        sig(
            "aceeasi ipoteza, cu reparatia (invalidare inaintea fiecarei lansari)",
            run(m, tok, 128, embed_on_cpu, SeqMode::EachLaunch),
        ),
        sig(
            "aceeasi ipoteza, cu o trimitere pe strat (ca la I13)",
            run(m, tok, 128, embed_on_cpu, SeqMode::PerLayer),
        ),
        sig(
            "logit-urile necopiate in RAM (umplutura 0xEE ramasa)",
            vec![f32::from_bits(kd::RAM_FILL); m.vocab],
        ),
        sig(
            "embedding fara randul pozitiei",
            run(m, tok, 0, embed_without_position, SeqMode::Once),
        ),
        sig(
            "embedding cu numerele luate fara semn",
            run(m, tok, 0, embed_unsigned, SeqMode::Once),
        ),
    ]
}

fn main() {
    let journal = std::env::args().nth(1).map(|p| {
        let b = std::fs::read(&p).unwrap_or_else(|e| {
            eprintln!("nu pot citi {p}: {e}");
            std::process::exit(1);
        });
        (p, String::from_utf8_lossy(&b).into_owned())
    });
    let dir = std::env::var("KGR1_VECTORI").ok();
    let mut sigs = Vec::new();
    if let Some(dir) = &dir {
        let read = |n: &str| {
            std::fs::read(format!("{dir}/{n}")).unwrap_or_else(|e| {
                eprintln!("{dir}/{n}: {e}\n{}", common::kgr1::VECTORS_NEED);
                std::process::exit(1);
            })
        };
        let f = common::kgr1::Files {
            kgr: read("kronos_mic_q8_0.kgr"),
            txt: String::from_utf8(read("kronos_mic_q8_0.txt")).unwrap(),
            logits: read("kronos_mic_q8_0_logits.f32"),
        };
        let m = read_kgr1(&f.kgr);
        let w = m.weights();
        let vs = vectors(&f, m.vocab);
        println!(
            "modelul de control: {} straturi, E {}, {} capete, vocabular {}, context {} (greutati la intamplare)",
            m.layers.len(),
            m.shape.e,
            m.shape.heads,
            m.vocab,
            m.context
        );
        println!(
            "\n1. Marja dintre primul si al doilea logit, pe procesor (motorul nostru), la fiecare pozitie de control:"
        );
        let mut smallest = (f32::INFINITY, String::new());
        for t in &vs {
            let mut c = Caches::new(m.layers.len(), m.shape.e);
            let mut seq = t.prompt.clone();
            for pos in 0..t.prompt.len() + t.greedy.len() - 1 {
                let l = model::step_cpu(&w, &mut c, seq[pos], pos).unwrap();
                let r = kd::logit_report(seq[pos], pos, &l, &l, None);
                let top = r.cpu.unwrap();
                println!(
                    "   \"{}\" pozitia {pos}: simbolul {} -> {top}, marja {:.4}",
                    t.text, seq[pos], r.cpu_margin
                );
                if r.cpu_margin < smallest.0 {
                    smallest = (r.cpu_margin, format!("\"{}\" pozitia {pos}", t.text));
                }
                if pos + 1 >= seq.len() {
                    seq.push(top);
                }
            }
        }
        println!(
            "   cea mai mica marja: {:.4} ({}) = de {:.0} ori toleranta logit-urilor (1e-3)",
            smallest.0,
            smallest.1,
            smallest.0 / model::LOGIT_ABS
        );
        let (tok, want) = (vs[0].prompt[0], vs[0].argmax[0]);
        let mut c = Caches::new(m.layers.len(), m.shape.e);
        let cpu = model::step_cpu(&w, &mut c, tok, 0).unwrap();
        println!(
            "\n2. Simbolul {tok} la pozitia 0 (\"{}\"; vectori.json: argmax {want}) pe MODELUL GPU (SIMULARE), pe ipoteze:",
            vs[0].text
        );
        sigs = signatures(&m, tok);
        for s in &sigs {
            let r = kd::logit_report(tok, 0, &s.logits, &cpu, Some(want));
            let top = r.board.map_or(String::from("niciunul"), |t| t.to_string());
            println!("   - {}:", s.name);
            println!(
                "       argmax {top}; {} logit-uri nefinite; abaterea maxima fata de procesor {:e}; {}",
                r.not_finite,
                r.max_abs,
                if r.passes {
                    "TRECE"
                } else if r.board.is_none() {
                    "NU TRECE - pe placa mesajul ar fi «logit-uri NaN/infinit», nu «argmax diferit»"
                } else if r.board == Some(want) {
                    "NU TRECE - acelasi simbol, logit-uri in afara tolerantei"
                } else {
                    "NU TRECE - «argmax diferit de vectorii lui Kronos», ca in os-305"
                }
            );
            println!("       primele 8 logit-uri: {:?}", &s.logits[..8]);
        }
    } else {
        println!(
            "(fara KGR1_VECTORI: partile 1 si 2 sar. {})",
            common::kgr1::VECTORS_NEED
        );
    }
    let Some((path, text)) = journal else {
        if dir.is_none() {
            eprintln!(
                "folosire: analiza_i14 [<gpu-jurnal.txt>] (cu KGR1_VECTORI pentru partile 1 si 2)"
            );
            std::process::exit(1);
        }
        return;
    };
    println!("\n3. Jurnalul {path}:");
    let d = kd::parse_journal(&text);
    if d.probes.is_empty() {
        println!("   niciun rand de diagnostic I14 (i14d) in jurnal.");
        if text.contains("pas i14-simbol: ESEC") {
            println!(
                "   pasul i14-simbol e ESEC; jurnalul e dinaintea diagnosticului (C-139): ce argmax a dat placa nu a ajuns pe stick."
            );
        }
        return;
    }
    for l in d.lines() {
        println!("   {l}");
    }
    // The board's first 8 logits of every probe that has them, against the signatures.
    for (i, p) in d.probes.iter().enumerate() {
        if p.logits_board.len() < 8 || p.pos != Some(0) {
            continue;
        }
        let hit = sigs.iter().find(|s| {
            s.logits[..8]
                .iter()
                .zip(&p.logits_board)
                .all(|(a, b)| (a - b).abs() <= 1e-3 + 1e-3 * a.abs())
        });
        match hit {
            Some(s) => println!(
                "   proba {}: primele 8 logit-uri ale placii se potrivesc cu: {}",
                i + 1,
                s.name
            ),
            None if !sigs.is_empty() => println!(
                "   proba {}: primele 8 logit-uri ale placii nu se potrivesc cu nicio ipoteza din lista de mai sus",
                i + 1
            ),
            None => {}
        }
    }
}
