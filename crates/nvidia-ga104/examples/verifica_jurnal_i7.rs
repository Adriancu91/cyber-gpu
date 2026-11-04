//! C-69: checks a stick journal of a window where I6 and then I7 (Booter on
//! SEC2, GSP-RM start) ran in the same boot. Host tool, reads text only,
//! never a board.
//!
//!   cargo run --release -p nvidia-ga104 --example verifica_jurnal_i7 -- <gpu-jurnal.txt> <adresa_GspFwWprMeta> [frts_start frts_end]
//!
//! The `GspFwWprMeta` address is the one the OS printed for this run (it goes
//! into SEC2 mailbox 0/1). FRTS defaults to the board's region (3–4 Nov 2026:
//! 0x1ffe00000..0x1fff00000). Exit code: 0 = I6 order, WPR2, I7 order and the
//! GSP RISC-V state all pass, 2 = something differs, 1 = usage / file error.

use nvidia_ga104::i6_log::check_i6_i7_journal;

fn hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
}

fn usage() -> ! {
    eprintln!(
        "folosire: verifica_jurnal_i7 <gpu-jurnal.txt> <adresa_GspFwWprMeta> [frts_start frts_end]"
    );
    std::process::exit(1);
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() != 2 && a.len() != 4 {
        usage();
    }
    let Some(meta) = hex(&a[1]) else { usage() };
    let frts = if a.len() == 4 {
        match (hex(&a[2]), hex(&a[3])) {
            (Some(s), Some(e)) if s < e => (s, e),
            _ => usage(),
        }
    } else {
        (0x1_ffe0_0000, 0x1_fff0_0000)
    };
    let text = match std::fs::read(&a[0]) {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) => {
            eprintln!("nu pot citi {}: {e}", a[0]);
            std::process::exit(1);
        }
    };
    let r = check_i6_i7_journal(&text, frts, meta);
    let mut ok = true;
    println!("jurnal: {}", a[0]);
    println!(
        "FRTS asteptat: {:#x}..{:#x}; GspFwWprMeta: {meta:#x}",
        frts.0, frts.1
    );
    match &r.i6_part.i6 {
        None => {
            ok = false;
            println!("I6: nicio scriere BAR0 inainte de I7");
        }
        Some(Ok(s)) => println!("I6 ordinea scrierilor: TRECE ({} scrieri)", s.writes),
        Some(Err(e)) => {
            ok = false;
            println!(
                "I6 ordinea scrierilor: NU TRECE la scrierea #{} ({:#x}): {}",
                e.write_index, e.offset, e.what
            );
        }
    }
    match &r.i6_part.wpr2_full {
        Some((lo, hi, Ok(c))) => println!(
            "WPR2 dupa I6: LO {lo:#010x} HI {hi:#010x} -> TRECE (capat: {})",
            c.form.label()
        ),
        Some((lo, hi, Err(e))) => {
            ok = false;
            println!("WPR2 dupa I6: LO {lo:#010x} HI {hi:#010x} -> NU TRECE: {e}");
        }
        None => {
            ok = false;
            println!("WPR2 dupa I6: nicio citire HI nenula inainte de I7");
        }
    }
    match &r.i7 {
        None => {
            ok = false;
            println!("I7: nicio scriere SEC2 (I7 nu a pornit)");
        }
        Some(Ok(s)) => println!(
            "I7 ordinea scrierilor: TRECE ({} scrieri, sectiuni {:?})",
            s.writes, s.sections
        ),
        Some(Err(e)) => {
            ok = false;
            println!(
                "I7 ordinea scrierilor: NU TRECE la scrierea #{} ({:#x}): {}",
                e.write_index, e.offset, e.what
            );
        }
    }
    match r.riscv_cpuctl {
        Some(v) if r.riscv_active => println!("GSP RISC-V: ACTIV (CPUCTL {v:#010x})"),
        Some(v) => {
            ok = false;
            println!("GSP RISC-V: NU e activ (CPUCTL {v:#010x})");
        }
        None => {
            ok = false;
            println!("GSP RISC-V: nicio citire in jurnal (lipseste `read_after_i7`)");
        }
    }
    let cfg = r.i6_part.config_writes.len() + r.i7_config_writes.len();
    println!("scrieri in configuratie: {cfg} (asteptat 2: Bus Master pornit, apoi la loc)");
    if cfg != 2 {
        ok = false;
    }
    println!("verdict: {}", if ok { "TRECE" } else { "DE ANALIZAT" });
    if !ok {
        std::process::exit(2);
    }
}
