//! C-71: checks the stick journal of a whole `gpu lant` window
//! (`WINDOW_CHAIN_FULL`: I6 → I7 → I8 → golden → I9 → I10 → launch cost →
//! Bus Master back → tour), step by step. Host tool, reads text only, never
//! a board.
//!
//!   cargo run --release -p nvidia-ga104 --example verifica_jurnal_lant -- <gpu-jurnal.txt> <adresa_GspFwWprMeta> [apeluri_rm] [frts_start frts_end]
//!
//! `apeluri_rm` = the number of RM calls the OS printed for golden + I9
//! (then the GSP doorbells must match it: one per message); `-` = unknown.
//! FRTS defaults to the board's region (3–4 Nov 2026: 0x1ffe00000..0x1fff00000).
//! Exit code: 0 = TRECE, 3 = stopped cleanly at a step (Bus Master put back),
//! 2 = to analyse, 1 = usage / file error.

use nvidia_ga104::chain_log::{ChainVerdict, check_chain_journal, step_ids};

fn hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
}

fn usage() -> ! {
    eprintln!(
        "folosire: verifica_jurnal_lant <gpu-jurnal.txt> <adresa_GspFwWprMeta> [apeluri_rm|-] [frts_start frts_end]"
    );
    std::process::exit(1);
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if !(2..=5).contains(&a.len()) || a.len() == 4 {
        usage();
    }
    let Some(meta) = hex(&a[1]) else { usage() };
    let rm_calls = match a.get(2).map(String::as_str) {
        None | Some("-") => None,
        Some(n) => match n.parse::<u32>() {
            Ok(n) => Some(n),
            Err(_) => usage(),
        },
    };
    let frts = if a.len() == 5 {
        match (hex(&a[3]), hex(&a[4])) {
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
    let r = check_chain_journal(&text, frts, meta, rm_calls);
    println!("jurnal: {}", a[0]);
    println!(
        "FRTS asteptat: {:#x}..{:#x}; GspFwWprMeta: {meta:#x}",
        frts.0, frts.1
    );
    for id in step_ids() {
        match r.step(id) {
            None => println!("  {id}: nu e in jurnal"),
            Some(s) => {
                let res = if s.ok { "ok" } else { "ESEC" };
                let rule = s.problem.as_deref().unwrap_or("conform");
                print!(
                    "  {id}: {res}; {} accese, {} scrieri BAR0, {} in configuratie",
                    s.accesses, s.bar0_writes, s.config_writes
                );
                if s.gsp_doorbells > 0 {
                    print!(", {} sonerii GSP", s.gsp_doorbells);
                }
                if s.work_doorbells > 0 {
                    print!(
                        ", {} sonerii canal de lucru (0xbb0090, DEDUS)",
                        s.work_doorbells
                    );
                }
                println!(" -> {rule}");
            }
        }
    }
    for p in &r.problems {
        println!("problema: {p}");
    }
    if let Some(s) = &r.i6 {
        println!("I6: {} scrieri, sectiuni {:?}", s.writes, s.sections);
    }
    if let Some(s) = &r.i7 {
        println!("I7: {} scrieri, sectiuni {:?}", s.writes, s.sections);
    }
    match rm_calls {
        Some(n) => println!(
            "sonerii GSP la golden + I9: {} (apeluri RM tiparite de OS: {n})",
            r.rm_doorbells
        ),
        None => println!(
            "sonerii GSP la golden + I9: {} (numarul apelurilor RM nu a fost dat)",
            r.rm_doorbells
        ),
    }
    match r.verdict() {
        ChainVerdict::Passed => println!("verdict: TRECE"),
        ChainVerdict::StoppedCleanly(at) => {
            println!("verdict: OPRIT CURAT la {at} (Bus Master pus la loc)");
            std::process::exit(3);
        }
        ChainVerdict::ToAnalyse(why) => {
            println!("verdict: DE ANALIZAT: {why}");
            std::process::exit(2);
        }
    }
}
