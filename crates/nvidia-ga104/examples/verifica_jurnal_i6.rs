//! C-63: checks a stick journal from the workstation (`raport/gpu-jurnal-00N.txt`,
//! written by `gpu i6 --scrie`). Host tool, reads text only, never a board.
//!
//!   cargo run --release -p nvidia-ga104 --example verifica_jurnal_i6 -- <gpu-jurnal.txt> [frts_start frts_end]
//!
//! FRTS defaults to the region the board reported on 3 Nov 2026
//! (`gpu pregatire-i6`: 0x1ffe00000..0x1fff00000). Exit code: 0 = write order
//! and WPR2 both pass, 2 = something differs, 1 = usage / file error.

use nvidia_ga104::i6_log::check_stick_journal;

fn hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let frts = match a.len() {
        1 => (0x1_ffe0_0000, 0x1_fff0_0000),
        3 => match (hex(&a[1]), hex(&a[2])) {
            (Some(s), Some(e)) if s < e => (s, e),
            _ => {
                eprintln!("FRTS gresit: {} {}", a[1], a[2]);
                std::process::exit(1);
            }
        },
        _ => {
            eprintln!("folosire: verifica_jurnal_i6 <gpu-jurnal.txt> [frts_start frts_end]");
            std::process::exit(1);
        }
    };
    let text = match std::fs::read(&a[0]) {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) => {
            eprintln!("nu pot citi {}: {e}", a[0]);
            std::process::exit(1);
        }
    };
    let r = check_stick_journal(&text, frts);
    let mut ok = true;
    println!("jurnal: {} ({} accesuri citite)", a[0], r.accesses);
    println!("FRTS asteptat: {:#x}..{:#x}", frts.0, frts.1);
    println!(
        "scrieri in configuratie: {} (asteptat 2: Bus Master pornit, apoi la loc)",
        r.config_writes.len()
    );
    for (off, b, w, af) in &r.config_writes {
        println!("  cfg w {off:#05x}: {b:#010x} <- {w:#010x} => {af:#010x}");
    }
    match &r.i6 {
        None => println!("ordinea scrierilor I6: nicio scriere BAR0 in jurnal"),
        Some(Ok(s)) => println!(
            "ordinea scrierilor I6: TRECE ({} scrieri, sectiuni {:?})",
            s.writes, s.sections
        ),
        Some(Err(e)) => {
            ok = false;
            println!(
                "ordinea scrierilor I6: NU TRECE la scrierea #{} ({:#x}): {}",
                e.write_index, e.offset, e.what
            );
        }
    }
    match &r.wpr2 {
        None => {
            ok = false;
            println!(
                "WPR2 dupa I6: nicio citire HI nenula (FWSEC n-a ajuns acolo sau WPR2 a ramas coborat)"
            );
        }
        Some((lo, hi, Ok(c))) => println!(
            "WPR2 dupa I6: LO {lo:#010x} HI {hi:#010x} -> TRECE (capat: {:?})",
            c.end_form
        ),
        Some((lo, hi, Err(e))) => {
            ok = false;
            println!("WPR2 dupa I6: LO {lo:#010x} HI {hi:#010x} -> NU TRECE: {e}");
        }
    }
    if r.config_writes.len() != 2 {
        ok = false;
    }
    println!("verdict: {}", if ok { "TRECE" } else { "DE ANALIZAT" });
    if !ok {
        std::process::exit(2);
    }
}
