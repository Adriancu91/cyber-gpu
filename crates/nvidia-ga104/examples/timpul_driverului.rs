//! C-147 point 7: where the driver's time of one `kronos --gpu` token goes, from a stick journal's row clocks.
//! Host arithmetic only, never a board.
//!
//!   cargo run --release -p nvidia-ga104 --example timpul_driverului -- <gpu-jurnal.txt>
//!
//! Prints, for the token loop (the longest run of doorbell intervals with the same rows), the median of the steady
//! tokens: the time with each VRAM window open and the time with no window open (the CPU), and the slow tokens left
//! out (the first, cold ones and the ones with a journal flush to the stick). What each window holds is written in
//! `docs/C-147_PUNCTUL7_ESTIMARE.md` (DEDUS from the OS code, not from the journal).

use nvidia_ga104::driver_split::{split_by_bell, steady, token_loop};

fn main() {
    let Some(path) = std::env::args().nth(1) else {
        eprintln!("folosire: timpul_driverului <gpu-jurnal.txt>");
        std::process::exit(2);
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("nu pot citi {path}: {e}");
            std::process::exit(1);
        }
    };
    let all = split_by_bell(&text);
    let tokens = token_loop(&all);
    if tokens.is_empty() {
        println!("niciun interval intre doua sonerii in jurnal");
        return;
    }
    let s = steady(tokens);
    println!(
        "intervale intre sonerii: {} in jurnal; bucla simbolurilor: {} ({} mutari ale ferestrei fiecare); constante: {}",
        all.len(),
        s.intervals,
        tokens[0].moves,
        s.steady
    );
    println!("median pe simbol (constante): {} us", s.total_us);
    for (b, us) in &s.windows {
        println!("  fereastra 0x{b:04x}: {us:>6} us");
    }
    println!("  fara fereastra (procesorul): {:>6} us", s.outside_us);
    if !s.slow.is_empty() {
        let v: Vec<String> = s.slow.iter().map(|u| format!("{u}")).collect();
        println!(
            "lasate deoparte (mai lente de 1,5 x mediana): {} us",
            v.join(", ")
        );
    }
}
