//! C-108 (session 89): reading the stick journals of the *variant* runs
//! (`gpu lant --varianta N`, one cold boot each) and comparing them with
//! each other and with a reference run (os-283, `gpu-jurnal-009.txt`).
//!
//! Thursday 12 Nov 2026 Adrian tries up to ten variants in a row; each
//! `trimite` puts one journal of ~2500 rows in the hub. This module turns
//! those into one table: which variant each journal is (the rows
//! [`crate::gsp::variants::Settings::journal`] writes at the top), how far
//! the chain went, what `GSP_INIT_DONE` said, how many NOCAT records came,
//! and *what differs* from the reference once the rows that differ on every
//! run are ignored (the `[ time]` prefix the OS adds, the journal header with
//! the commit, the NOCAT timestamps, the variant rows themselves). The
//! sessions 87–88 did that comparison by hand (`diff` after stripping the
//! prefix); here it is one function with a test.
//!
//! Pure text processing (like [`crate::chain_log`] and
//! [`crate::gsp::nocat_log`]): never touches a board, never says more than
//! the journals do. The outcome of a run is read from the rows, not guessed:
//! a run "passed I8" only if the journal has `pas i8-gsp-init: ok`.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::fmt;

use crate::chain_log::parse_step;
use crate::gsp::nocat_log::{self, NocatDigest};

/// The step whose outcome the variants are about.
pub const I8_STEP: &str = "i8-gsp-init";

const WHAT_VARIANT_NR: &str = "varianta I8 (C-108): numarul";
const TEXT_VARIANT_NAME: &str = "varianta I8: numele";
const WHAT_SETTING_PREFIX: &str = "varianta: ";
const WHAT_NOCAT_TIMESTAMP: &str = "NOCAT: timestamp";
const HEADER_PREFIX: &str = "# jurnal gpu";
const WAITED_PREFIX: &str = "asteptat ";

/// What one journal says about its run.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VariantRun {
    /// `valoare varianta I8 (C-108): numarul` — `Some(0)` = no `--varianta`
    /// (the os-283 behaviour), `None` = the row is not there (a driver from
    /// before C-108).
    pub variant: Option<u32>,
    /// `text varianta I8: numele`.
    pub variant_name: Option<String>,
    /// The commit from the header `# jurnal gpu …, cyber-gpu <rev>, comanda: …`.
    pub cyber_gpu_rev: Option<String>,
    /// The `valoare varianta: <setting> = …` rows, in journal order.
    pub settings: Vec<(String, u32)>,
    /// Every `pas <id>: ok|ESEC` row, in order.
    pub steps: Vec<(String, bool)>,
    /// What GSP-RM said before `GSP_INIT_DONE` (empty if the run stopped earlier).
    pub nocat: NocatDigest,
    /// Rows in the journal.
    pub rows: usize,
}

/// How far a run got, read from the step rows only.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// `pas i8-gsp-init: ok` is in the journal.
    PassedI8,
    /// `pas i8-gsp-init: ESEC`, with `GSP_INIT_DONE: rpc_result` when the
    /// journal has it (the code and its `NV_STATUS` name).
    RefusedAtI8(Option<(u32, String)>),
    /// The chain stopped at an earlier step (`ESEC` before I8).
    StoppedBefore(String),
    /// No `pas` row about I8 at all and no failed step (a journal cut short
    /// or from another command).
    Unknown,
}

impl fmt::Display for Outcome {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Outcome::PassedI8 => write!(f, "I8 TRECUT"),
            Outcome::RefusedAtI8(Some((code, name))) if name.is_empty() => {
                write!(f, "I8 refuzat: rpc_result {code:#x}")
            }
            Outcome::RefusedAtI8(Some((code, name))) => {
                write!(f, "I8 refuzat: rpc_result {code:#x} ({name})")
            }
            Outcome::RefusedAtI8(None) => write!(f, "I8 ESEC (fara GSP_INIT_DONE in jurnal)"),
            Outcome::StoppedBefore(s) => write!(f, "oprit inainte de I8, la {s}"),
            Outcome::Unknown => write!(f, "fara rand despre I8"),
        }
    }
}

/// `valoare <what> = 0x........`, possibly after a log prefix.
fn value_row(line: &str) -> Option<(&str, u32)> {
    let rest = &line[line.find("valoare ")? + 8..];
    let (what, v) = rest.rsplit_once(" = ")?;
    let v = u32::from_str_radix(v.trim().trim_start_matches("0x"), 16).ok()?;
    Some((what, v))
}

/// `text <what> = "..."`, possibly after a log prefix.
fn text_row(line: &str) -> Option<(&str, &str)> {
    let rest = &line[line.find("text ")? + 5..];
    let (what, v) = rest.split_once(" = \"")?;
    let v = v.trim_end().strip_suffix('"')?;
    Some((what, v))
}

/// The OS prefix `[   66.217252] ` (and `gpu: `) stripped from a row, so two
/// runs can be compared row by row. Rows without the prefix come back as
/// they are (trimmed at the end).
pub fn strip_prefix(line: &str) -> &str {
    let line = line.trim_end();
    let rest = match line.strip_prefix('[') {
        Some(r) => match r.split_once("] ") {
            Some((stamp, rest))
                if !stamp.is_empty()
                    && stamp.trim().chars().all(|c| c.is_ascii_digit() || c == '.') =>
            {
                rest
            }
            _ => line,
        },
        None => line,
    };
    rest.strip_prefix("gpu: ").unwrap_or(rest)
}

/// True for a row that differs between any two runs without meaning
/// anything: the header (commit, command), the NOCAT timestamps, the wait
/// durations (`asteptat …: raspuns dupa N us`), the variant rows themselves
/// (they *are* the difference being tested).
pub fn is_noise(stripped: &str) -> bool {
    stripped.starts_with(HEADER_PREFIX)
        || stripped.starts_with(WAITED_PREFIX)
        || stripped.is_empty()
        || match value_row(stripped) {
            Some((what, _)) => {
                what.starts_with(WHAT_NOCAT_TIMESTAMP)
                    || what == WHAT_VARIANT_NR
                    || what.starts_with(WHAT_SETTING_PREFIX)
            }
            None => matches!(text_row(stripped), Some((what, _)) if what == TEXT_VARIANT_NAME),
        }
}

/// Reads one journal.
pub fn read(text: &str) -> VariantRun {
    let mut r = VariantRun {
        nocat: nocat_log::digest(text),
        ..VariantRun::default()
    };
    for line in text.lines() {
        r.rows += 1;
        let s = strip_prefix(line);
        if let Some(rest) = s.strip_prefix(HEADER_PREFIX) {
            if let Some(i) = rest.find("cyber-gpu ") {
                let rev = rest[i + 10..].split([',', ' ']).next().unwrap_or("");
                if !rev.is_empty() {
                    r.cyber_gpu_rev = Some(rev.to_string());
                }
            }
            continue;
        }
        if let Some((id, ok)) = parse_step(s) {
            r.steps.push((id.to_string(), ok));
            continue;
        }
        if let Some((what, v)) = value_row(s) {
            if what == WHAT_VARIANT_NR {
                r.variant = Some(v);
            } else if let Some(setting) = what.strip_prefix(WHAT_SETTING_PREFIX) {
                r.settings.push((setting.to_string(), v));
            }
            continue;
        }
        if let Some((TEXT_VARIANT_NAME, v)) = text_row(s) {
            r.variant_name = Some(v.to_string());
        }
    }
    r
}

impl VariantRun {
    /// `varianta N (nume)` / `fara varianta (os-283)` / `driver dinainte de C-108`.
    pub fn variant_label(&self) -> String {
        match (self.variant, &self.variant_name) {
            (Some(0), _) => String::from("fara --varianta (ca os-283)"),
            (Some(n), Some(name)) => format!("varianta {n} ({name})"),
            (Some(n), None) => format!("varianta {n}"),
            (None, _) => String::from("driver dinainte de C-108 (fara randul variantei)"),
        }
    }

    /// The outcome, from the step rows (and `GSP_INIT_DONE` when I8 failed).
    pub fn outcome(&self) -> Outcome {
        match self.steps.iter().find(|(id, _)| id == I8_STEP) {
            Some((_, true)) => Outcome::PassedI8,
            Some((_, false)) => Outcome::RefusedAtI8(
                self.nocat
                    .init_done_result
                    .map(|c| (c, self.nocat.init_done_name.clone().unwrap_or_default())),
            ),
            None => match self.steps.iter().find(|(_, ok)| !ok) {
                Some((id, _)) => Outcome::StoppedBefore(id.clone()),
                None => Outcome::Unknown,
            },
        }
    }

    /// The last step row (`id`, passed), if any.
    pub fn last_step(&self) -> Option<&(String, bool)> {
        self.steps.last()
    }

    /// The assert addresses of the records journaled in full, sorted and
    /// deduplicated (what the sessions compared by hand between os-280 and os-283).
    pub fn assert_addresses(&self) -> Vec<u32> {
        let mut v: Vec<u32> = self
            .nocat
            .records
            .iter()
            .filter_map(|r| r.assert_addr)
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

/// The meaningful rows of a journal (prefix stripped, noise rows dropped),
/// as a multiset: row → how many times it appears.
fn row_bag(text: &str) -> BTreeMap<&str, u32> {
    let mut m = BTreeMap::new();
    for line in text.lines() {
        let s = strip_prefix(line);
        if !is_noise(s) {
            *m.entry(s).or_insert(0) += 1;
        }
    }
    m
}

/// What differs between two journals once the noise rows are ignored:
/// rows (with their count) only one of them has. Order-insensitive, so a
/// variant that adds rows at I7 does not make every later row "different".
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Difference {
    /// Meaningful rows counted in the reference / in the run.
    pub reference_rows: u32,
    pub run_rows: u32,
    /// Rows (and how many times) only the reference has, in row order.
    pub only_reference: Vec<(String, u32)>,
    /// Rows only the run has, in row order.
    pub only_run: Vec<(String, u32)>,
}

impl Difference {
    pub fn is_identical(&self) -> bool {
        self.only_reference.is_empty() && self.only_run.is_empty()
    }
}

/// Compares a run with the reference journal (os-283 for Thursday's runs).
pub fn compare(reference: &str, run: &str) -> Difference {
    let a = row_bag(reference);
    let b = row_bag(run);
    let mut d = Difference {
        reference_rows: a.values().sum(),
        run_rows: b.values().sum(),
        ..Difference::default()
    };
    // In row order (not BTreeMap order), each row once with its surplus count.
    let mut seen_a: BTreeMap<&str, ()> = BTreeMap::new();
    for line in reference.lines() {
        let s = strip_prefix(line);
        if is_noise(s) || seen_a.contains_key(s) {
            continue;
        }
        seen_a.insert(s, ());
        let na = a.get(s).copied().unwrap_or(0);
        let nb = b.get(s).copied().unwrap_or(0);
        if na > nb {
            d.only_reference.push((s.to_string(), na - nb));
        }
    }
    let mut seen_b: BTreeMap<&str, ()> = BTreeMap::new();
    for line in run.lines() {
        let s = strip_prefix(line);
        if is_noise(s) || seen_b.contains_key(s) {
            continue;
        }
        seen_b.insert(s, ());
        let na = a.get(s).copied().unwrap_or(0);
        let nb = b.get(s).copied().unwrap_or(0);
        if nb > na {
            d.only_run.push((s.to_string(), nb - na));
        }
    }
    d
}

/// One journal in the table: its label (file name), what it says, and how it
/// differs from the reference (`None` for the reference itself).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TableRow {
    pub label: String,
    pub run: VariantRun,
    pub difference: Option<Difference>,
}

/// The table of Thursday's runs, sorted by variant number (the reference and
/// runs without a variant row first), with the one-line conclusion.
pub struct Table<'a> {
    pub rows: &'a [TableRow],
    /// How many differing rows to print per journal.
    pub max_diff_rows: usize,
}

impl Table<'_> {
    /// The variants whose journal says `pas i8-gsp-init: ok`.
    pub fn passed(&self) -> Vec<&TableRow> {
        self.rows
            .iter()
            .filter(|r| r.run.outcome() == Outcome::PassedI8)
            .collect()
    }
}

impl fmt::Display for Table<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut order: Vec<&TableRow> = self.rows.iter().collect();
        order.sort_by_key(|r| (r.difference.is_some(), r.run.variant.unwrap_or(0)));
        for r in order {
            writeln!(f, "{}: {}", r.label, r.run.variant_label())?;
            if let Some(rev) = &r.run.cyber_gpu_rev {
                writeln!(f, "  cyber-gpu: {rev}; {} randuri", r.run.rows)?;
            }
            writeln!(f, "  rezultat: {}", r.run.outcome())?;
            if let Some((id, ok)) = r.run.last_step() {
                writeln!(
                    f,
                    "  ultimul pas: {id} ({})",
                    if *ok { "ok" } else { "ESEC" }
                )?;
            }
            if let Some(n) = r.run.nocat.summary_received {
                let addrs = r.run.assert_addresses();
                writeln!(
                    f,
                    "  NOCAT: {n} inregistrari primite; {} jurnalizate ({} adrese de assert distincte)",
                    r.run.nocat.records.len(),
                    addrs.len()
                )?;
            }
            if !r.run.settings.is_empty() {
                write!(f, "  setari:")?;
                for (s, v) in &r.run.settings {
                    let short = s.split(" (").next().unwrap_or(s);
                    write!(f, " {short}={v}")?;
                }
                writeln!(f)?;
            }
            match &r.difference {
                None => writeln!(f, "  (referinta)")?,
                Some(d) if d.is_identical() => writeln!(
                    f,
                    "  fata de referinta: IDENTIC (fara antet, marcaje de timp si randurile variantei; {} randuri)",
                    d.run_rows
                )?,
                Some(d) => {
                    writeln!(
                        f,
                        "  fata de referinta: {} randuri doar in referinta, {} doar aici ({} vs {} randuri cu sens)",
                        d.only_reference.len(),
                        d.only_run.len(),
                        d.reference_rows,
                        d.run_rows
                    )?;
                    for (s, n) in d.only_reference.iter().take(self.max_diff_rows) {
                        if *n > 1 {
                            writeln!(f, "    - {s}  (x{n})")?;
                        } else {
                            writeln!(f, "    - {s}")?;
                        }
                    }
                    if d.only_reference.len() > self.max_diff_rows {
                        writeln!(
                            f,
                            "    - … inca {} randuri doar in referinta",
                            d.only_reference.len() - self.max_diff_rows
                        )?;
                    }
                    for (s, n) in d.only_run.iter().take(self.max_diff_rows) {
                        if *n > 1 {
                            writeln!(f, "    + {s}  (x{n})")?;
                        } else {
                            writeln!(f, "    + {s}")?;
                        }
                    }
                    if d.only_run.len() > self.max_diff_rows {
                        writeln!(
                            f,
                            "    + … inca {} randuri doar aici",
                            d.only_run.len() - self.max_diff_rows
                        )?;
                    }
                }
            }
        }
        let passed = self.passed();
        if passed.is_empty() {
            writeln!(
                f,
                "concluzie: nicio varianta nu a trecut de I8 (din {} jurnale)",
                self.rows.len()
            )
        } else {
            write!(f, "concluzie: au trecut de I8:")?;
            for r in passed {
                write!(f, " {} [{}]", r.run.variant_label(), r.label)?;
            }
            writeln!(f)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Rows as the OS writes them on the stick (`[ time] gpu: ` prefix),
    /// after the header row.
    fn journal(rev: &str, rows: &[&str]) -> String {
        let mut s = format!("# jurnal gpu P6-40, cyber-gpu {rev}, comanda: gpu lant\n");
        for (i, r) in rows.iter().enumerate() {
            s.push_str(&format!(
                "[   {}.{:06}] gpu: {r}\n",
                60 + i / 10,
                i * 1234 % 1_000_000
            ));
        }
        s
    }

    const COMMON_I6_I7: &[&str] = &[
        "pas pregatire-i6: ok",
        "cfg  w 0x004: 0x00100003 <- 0x00000007 => 0x00100007",
        "pas bus-master-pornit: ok",
        "bar0 w 0x110100: 0x00000000 <- 0x00000002 => 0x00000002",
        "pas i6-fwsec-frts: ok",
        "pas i6-wpr2-citit: ok",
    ];

    fn os283_journal() -> String {
        let mut rows: Vec<&str> = alloc::vec![
            "valoare varianta I8 (C-108): numarul = 0x00000000",
            "text varianta I8: numele = \"os-283\"",
            "valoare varianta: RPC-urile de pornire inainte de Booter (1 = la I7, 0 = la I8) = 0x00000000",
            "valoare varianta: GSP_ARGUMENTS_CACHED.bDmemStack = 0x00000001",
        ];
        rows.extend_from_slice(COMMON_I6_I7);
        rows.push("pas i7-booter-gsp: ok");
        rows.push("pas i7-citit: ok");
        rows.push("bar0 w 0x110c00: 0x00000000 <- 0x00000000 => 0x00000000");
        rows.push("valoare NOCAT: inregistrarea nr. = 0x00000001");
        rows.push(
            "valoare NOCAT: recType (1 BUGCHECK, 2 ENGINE, 3 TDR, 4 RC, 5 ASSERT) = 0x00000005",
        );
        rows.push("text NOCAT: recType = \"ASSERT\"");
        rows.push(
            "valoare NOCAT ASSERT: adresa assert-ului (fata de inceputul GSP-RM) = 0x019ef3c6",
        );
        rows.push("valoare NOCAT ASSERT: linia din sursa RM = 0x00000000");
        rows.push("valoare NOCAT: timestamp (32 de biti de jos) = 0xd54e5d60");
        rows.push("valoare I8 NOCAT (0x1020): inregistrari de diagnostic primite = 0x0000034c");
        rows.push(
            "valoare GSP_INIT_DONE: rpc_result (0 = bine; altfel cod NV_STATUS) = 0x00000059",
        );
        rows.push("text GSP_INIT_DONE: rpc_result (nume NV_STATUS) = \"NV_ERR_OPERATING_SYSTEM\"");
        rows.push("pas i8-gsp-init: ESEC");
        rows.push("cfg  w 0x004: 0x00100007 <- 0x00000003 => 0x00100003");
        rows.push("pas bus-master-oprit: ok");
        rows.push("pas tur-dupa: ok");
        journal("2bbd0b3", &rows)
    }

    /// The same run, other cold boot: only the prefix, the commit and the
    /// NOCAT timestamp differ.
    fn same_run_again() -> String {
        os283_journal()
            .replace("cyber-gpu 2bbd0b3", "cyber-gpu 49e6685")
            .replace("0xd54e5d60", "0xdb3ee380")
            .replace("[   60.", "[   99.")
    }

    fn variant1_passed() -> String {
        let mut rows: Vec<&str> = alloc::vec![
            "valoare varianta I8 (C-108): numarul = 0x00000001",
            "text varianta I8: numele = \"rpc-inainte\"",
            "valoare varianta: RPC-urile de pornire inainte de Booter (1 = la I7, 0 = la I8) = 0x00000001",
            "valoare varianta: GSP_ARGUMENTS_CACHED.bDmemStack = 0x00000001",
        ];
        rows.extend_from_slice(COMMON_I6_I7);
        rows.push("gpu0: I7 pregatire (varianta C-108): SetSystemInfo + SetRegistry (2 intrari) puse in coada INAINTE de Booter; nimic scris in placa");
        rows.push("bar0 w 0x110c00: 0x00000000 <- 0x00000000 => 0x00000000");
        rows.push("pas i7-booter-gsp: ok");
        rows.push("pas i7-citit: ok");
        rows.push("valoare I8 NOCAT (0x1020): inregistrari de diagnostic primite = 0x00000000");
        rows.push(
            "valoare GSP_INIT_DONE: rpc_result (0 = bine; altfel cod NV_STATUS) = 0x00000000",
        );
        rows.push("text GSP_INIT_DONE: rpc_result (nume NV_STATUS) = \"NV_OK\"");
        rows.push("pas i8-gsp-init: ok");
        rows.push("pas i8e-golden: ok");
        rows.push("pas bus-master-oprit: ok");
        rows.push("pas tur-dupa: ok");
        journal("49e6685", &rows)
    }

    fn variant2_refused_differently() -> String {
        os283_journal()
            .replace("numarul = 0x00000000", "numarul = 0x00000002")
            .replace("\"os-283\"", "\"rpc-inainte-set-nova\"")
            .replace("0x019ef3c6", "0x019ef400")
            .replace("= 0x0000034c", "= 0x00000010")
    }

    fn stopped_at_i6() -> String {
        journal(
            "49e6685",
            &[
                "valoare varianta I8 (C-108): numarul = 0x00000003",
                "text varianta I8: numele = \"rpc-inainte-set-tinygrad\"",
                "pas pregatire-i6: ok",
                "pas bus-master-pornit: ok",
                "pas i6-fwsec-frts: ESEC",
                "pas bus-master-oprit: ok",
                "pas tur-dupa: ok",
            ],
        )
    }

    #[test]
    fn the_prefix_is_stripped_and_noise_rows_are_known() {
        assert_eq!(
            strip_prefix("[  100.296752] gpu: valoare NOCAT: inregistrarea nr. = 0x00000001"),
            "valoare NOCAT: inregistrarea nr. = 0x00000001"
        );
        assert_eq!(strip_prefix("pas i7-citit: ok  "), "pas i7-citit: ok");
        assert_eq!(strip_prefix("[abc] gpu: x"), "[abc] gpu: x");
        assert!(is_noise(
            "# jurnal gpu P6-40, cyber-gpu 2bbd0b3, comanda: gpu lant"
        ));
        assert!(is_noise(
            "valoare NOCAT: timestamp (32 de biti de jos) = 0xd54e5d60"
        ));
        assert!(is_noise(
            "valoare varianta I8 (C-108): numarul = 0x00000001"
        ));
        assert!(is_noise(
            "valoare varianta: GspFwWprMeta.flags = 0x00000000"
        ));
        assert!(is_noise("text varianta I8: numele = \"rpc-inainte\""));
        assert!(is_noise(
            "asteptat CORE_RESUME: SEC2 (Booter) gata: raspuns dupa 3 us"
        ));
        assert!(!is_noise(
            "valoare NOCAT ASSERT: adresa assert-ului (fata de inceputul GSP-RM) = 0x019ef3c6"
        ));
        assert!(!is_noise("pas i8-gsp-init: ESEC"));
        assert!(!is_noise(
            "bar0 w 0x110c00: 0x00000000 <- 0x00000000 => 0x00000000"
        ));
    }

    #[test]
    fn a_journal_is_read_variant_steps_and_outcome() {
        let r = read(&os283_journal());
        assert_eq!(r.variant, Some(0));
        assert_eq!(r.variant_name.as_deref(), Some("os-283"));
        assert_eq!(r.cyber_gpu_rev.as_deref(), Some("2bbd0b3"));
        assert_eq!(r.settings.len(), 2);
        assert_eq!(r.settings[1].1, 1);
        assert_eq!(r.steps.len(), 9);
        assert_eq!(r.last_step().unwrap().0, "tur-dupa");
        assert_eq!(
            r.outcome(),
            Outcome::RefusedAtI8(Some((0x59, String::from("NV_ERR_OPERATING_SYSTEM"))))
        );
        assert_eq!(r.nocat.summary_received, Some(0x34c));
        assert_eq!(r.assert_addresses(), alloc::vec![0x019e_f3c6]);
        assert_eq!(r.variant_label(), "fara --varianta (ca os-283)");

        let p = read(&variant1_passed());
        assert_eq!(p.variant, Some(1));
        assert_eq!(p.outcome(), Outcome::PassedI8);
        assert_eq!(p.variant_label(), "varianta 1 (rpc-inainte)");

        let s = read(&stopped_at_i6());
        assert_eq!(
            s.outcome(),
            Outcome::StoppedBefore(String::from("i6-fwsec-frts"))
        );

        let old = read(&journal("bcc314e", &["pas pregatire-i6: ok"]));
        assert_eq!(old.variant, None);
        assert_eq!(old.outcome(), Outcome::Unknown);
        assert!(old.variant_label().contains("dinainte de C-108"));
    }

    #[test]
    fn the_same_run_on_another_cold_boot_is_identical() {
        let d = compare(&os283_journal(), &same_run_again());
        assert!(d.is_identical(), "{d:?}");
        assert_eq!(d.reference_rows, d.run_rows);
        // 26 rows after the header; the 4 variant rows and the NOCAT timestamp are noise.
        assert_eq!(d.reference_rows, 21);
    }

    #[test]
    fn differences_are_listed_once_with_their_count_and_order_does_not_matter() {
        let d = compare(&os283_journal(), &variant2_refused_differently());
        assert_eq!(d.only_reference.len(), 2);
        assert_eq!(d.only_run.len(), 2);
        assert!(d.only_reference[0].0.contains("0x019ef3c6"));
        assert!(d.only_run[0].0.contains("0x019ef400"));
        assert!(d.only_reference[1].0.contains("= 0x0000034c"));
        assert!(d.only_run[1].0.contains("= 0x00000010"));

        // Variant 1 adds rows at I7 and ends differently: the shared rows are
        // not reported, only the real additions / removals.
        let d = compare(&os283_journal(), &variant1_passed());
        let only_run: Vec<&str> = d.only_run.iter().map(|(s, _)| s.as_str()).collect();
        assert!(
            only_run
                .iter()
                .any(|s| s.contains("puse in coada INAINTE de Booter"))
        );
        assert!(only_run.contains(&"pas i8-gsp-init: ok"));
        assert!(only_run.contains(&"pas i8e-golden: ok"));
        assert!(!only_run.contains(&"pas tur-dupa: ok"));
        let only_ref: Vec<&str> = d.only_reference.iter().map(|(s, _)| s.as_str()).collect();
        assert!(only_ref.contains(&"pas i8-gsp-init: ESEC"));
        assert!(only_ref.iter().any(|s| s.contains("0x019ef3c6")));
        // The doorbell row is in both (I8 in the reference, I7 in variant 1): not a difference.
        assert!(!only_ref.iter().any(|s| s.contains("0x110c00")));
        assert!(!only_run.iter().any(|s| s.contains("0x110c00")));

        // A row repeated more often in one journal shows its surplus.
        let twice = os283_journal().replace(
            "pas i7-citit: ok",
            "bar0 w 0x110100: 0x00000000 <- 0x00000002 => 0x00000002\npas i7-citit: ok",
        );
        let d = compare(&os283_journal(), &twice);
        assert_eq!(
            d.only_run,
            alloc::vec![(
                String::from("bar0 w 0x110100: 0x00000000 <- 0x00000002 => 0x00000002"),
                1
            )]
        );
        assert!(d.only_reference.is_empty());
    }

    #[test]
    fn the_table_sorts_by_variant_names_the_passed_one_and_the_identical_ones() {
        let reference = os283_journal();
        let mk = |label: &str, text: String, with_diff: bool| TableRow {
            label: String::from(label),
            run: read(&text),
            difference: with_diff.then(|| compare(&reference, &text)),
        };
        let rows = alloc::vec![
            mk("gpu-jurnal-012.txt", variant2_refused_differently(), true),
            mk("gpu-jurnal-013.txt", stopped_at_i6(), true),
            mk("gpu-jurnal-011.txt", variant1_passed(), true),
            mk("gpu-jurnal-010.txt", same_run_again(), true),
            mk("gpu-jurnal-009.txt", reference.clone(), false),
        ];
        let t = Table {
            rows: &rows,
            max_diff_rows: 3,
        };
        assert_eq!(t.passed().len(), 1);
        let s = format!("{t}");
        let pos = |needle: &str| {
            s.find(needle)
                .unwrap_or_else(|| panic!("{needle:?} not in:\n{s}"))
        };
        // Reference first, then variants 0, 1, 2, 3.
        assert!(
            pos("gpu-jurnal-009.txt: fara --varianta") < pos("gpu-jurnal-010.txt: fara --varianta")
        );
        assert!(pos("gpu-jurnal-010.txt") < pos("gpu-jurnal-011.txt: varianta 1 (rpc-inainte)"));
        assert!(pos("gpu-jurnal-011.txt") < pos("gpu-jurnal-012.txt: varianta 2"));
        assert!(pos("gpu-jurnal-012.txt") < pos("gpu-jurnal-013.txt: varianta 3"));
        assert!(s.contains("  (referinta)"));
        assert!(s.contains("fata de referinta: IDENTIC"));
        assert!(s.contains("rezultat: I8 TRECUT"));
        assert!(s.contains("rezultat: I8 refuzat: rpc_result 0x59 (NV_ERR_OPERATING_SYSTEM)"));
        assert!(s.contains("rezultat: oprit inainte de I8, la i6-fwsec-frts"));
        assert!(s.contains(
            "NOCAT: 844 inregistrari primite; 1 jurnalizate (1 adrese de assert distincte)"
        ));
        assert!(s.contains(
            "setari: RPC-urile de pornire inainte de Booter=1 GSP_ARGUMENTS_CACHED.bDmemStack=1"
        ));
        assert!(
            s.contains("… inca"),
            "variant 1 has more than 3 differing rows:\n{s}"
        );
        assert!(s.ends_with(
            "concluzie: au trecut de I8: varianta 1 (rpc-inainte) [gpu-jurnal-011.txt]\n"
        ));

        let none = [mk("a.txt", same_run_again(), true)];
        let t = Table {
            rows: &none,
            max_diff_rows: 3,
        };
        assert!(
            format!("{t}")
                .ends_with("concluzie: nicio varianta nu a trecut de I8 (din 1 jurnale)\n")
        );
    }
}
