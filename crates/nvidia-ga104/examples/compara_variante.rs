//! C-108: compares the stick journals of the variant runs (`gpu lant
//! --varianta N`, one cold boot each) with a reference run (os-283,
//! `gpu-jurnal-009.txt`): which variant each journal is, how far it went,
//! what `GSP_INIT_DONE` said, how many NOCAT records came, and which rows
//! differ from the reference once the prefix, header, NOCAT timestamps and
//! the variant rows themselves are ignored. Host tool, reads text only,
//! never a board. Each journal is also checked step by step like
//! `verifica_jurnal_lant` (chain verdict), with the `GspFwWprMeta` address.
//!
//!   cargo run --release -p nvidia-ga104 --example compara_variante -- <adresa_GspFwWprMeta> <referinta.txt> <jurnal.txt>...
//!
//! Exit code: 0 = at least one journal passed I8, 3 = none did (all stopped
//! cleanly), 2 = a journal is to be analysed (a rule broken), 1 = usage /
//! file error.

use nvidia_ga104::chain_log::{ChainVerdict, chain_for_journal, check_chain_journal_for};
use nvidia_ga104::variant_log::{self, Table, TableRow};

fn hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.trim_start_matches("0x"), 16).ok()
}

fn usage() -> ! {
    eprintln!("folosire: compara_variante <adresa_GspFwWprMeta> <referinta.txt> <jurnal.txt>...");
    std::process::exit(1);
}

fn read(path: &str) -> String {
    match std::fs::read(path) {
        Ok(b) => String::from_utf8_lossy(&b).into_owned(),
        Err(e) => {
            eprintln!("nu pot citi {path}: {e}");
            std::process::exit(1);
        }
    }
}

fn label(path: &str) -> String {
    path.rsplit(['/', '\\']).next().unwrap_or(path).to_string()
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    if a.len() < 3 {
        usage();
    }
    let Some(meta) = hex(&a[0]) else { usage() };
    let frts = (0x1_ffe0_0000, 0x1_fff0_0000);
    let reference = read(&a[1]);
    let mut rows = vec![TableRow {
        label: label(&a[1]),
        run: variant_log::read(&reference),
        difference: None,
    }];
    let mut verdicts = Vec::new();
    for p in &a[1..] {
        let text = if p == &a[1] {
            reference.clone()
        } else {
            read(p)
        };
        if p != &a[1] {
            rows.push(TableRow {
                label: label(p),
                run: variant_log::read(&text),
                difference: Some(variant_log::compare(&reference, &text)),
            });
        }
        let chain = chain_for_journal(&text);
        let r = check_chain_journal_for(chain, &text, frts, meta, None);
        verdicts.push((label(p), r.verdict_for(chain), r.problems));
    }
    let table = Table {
        rows: &rows,
        max_diff_rows: 12,
    };
    println!("referinta: {} (GspFwWprMeta {meta:#x})", a[1]);
    print!("{table}");
    println!("verdictele lantului (ca verifica_jurnal_lant):");
    let mut worst = 3;
    for (l, v, problems) in &verdicts {
        match v {
            ChainVerdict::Passed => {
                println!("  {l}: TRECE");
                worst = worst.min(0);
            }
            ChainVerdict::StoppedCleanly(at) => println!("  {l}: OPRIT CURAT la {at}"),
            ChainVerdict::ToAnalyse(why) => {
                println!("  {l}: DE ANALIZAT: {why}");
                worst = 2;
            }
        }
        for p in problems {
            println!("    problema: {p}");
        }
    }
    let passed = !table.passed().is_empty();
    std::process::exit(if worst == 2 {
        2
    } else if passed {
        0
    } else {
        3
    });
}
