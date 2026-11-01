//! C-54: compares a Cyber Universe report from the workstation with the
//! simulated board (`gpu_sim::observed`). Host tool, reads text only.
//!
//!   cargo run --release -p nvidia-ga104 --example compara_raport -- <raport.txt>...
//!
//! Several reports may be given (oldest first): their facts are merged, the
//! last value of a key wins. Exit code: 0 = no real difference, 2 = the
//! model differs from the board somewhere (fix the model or look at the
//! board), 1 = usage / file error.

use gpu_core::diag::Journal;
use gpu_sim::clock::SimClock;
use gpu_sim::observed;
use nvidia_ga104::board_compare::{compare, facts_from_report};
use nvidia_ga104::read_tour::run_tour;

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    if paths.is_empty() {
        eprintln!("folosire: compara_raport <raport.txt>...");
        std::process::exit(1);
    }
    let mut board = Vec::new();
    for p in &paths {
        match std::fs::read(p) {
            Ok(b) => board.extend(facts_from_report(&String::from_utf8_lossy(&b))),
            Err(e) => {
                eprintln!("nu pot citi {p}: {e}");
                std::process::exit(1);
            }
        }
    }
    let mut cfg = observed::observed_config();
    let mut bar0 = observed::observed_bar0();
    let mut journal = Journal::new(8192);
    let tour = match run_tour(
        &mut cfg,
        None,
        &mut bar0,
        &mut SimClock::default(),
        &mut journal,
    ) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("turul pe simulare a esuat: {e}");
            std::process::exit(1);
        }
    };
    let c = compare(&board, &tour.facts);
    println!("SIMULARE: modelul nu e placa; vezi lista de mai jos cu ce inca ghiceste.");
    println!("rapoarte: {}", paths.join(", "));
    print!("{c}");
    println!("ce inventeaza inca simularea:");
    for s in observed::STILL_SIM {
        println!("  - {s}");
    }
    if c.has_real_difference() {
        std::process::exit(2);
    }
}
