//! Preparation for the `os-280` journal: a digest of what the stick journal
//! says about the messages GSP-RM sent before `GSP_INIT_DONE` — the NOCAT
//! records decoded by [`super::nocat`] (rows `valoare NOCAT: …` /
//! `text NOCAT: …`), the `UCODE_LIBOS_PRINT` rows, the summary rows written
//! by [`super::sequencer::wait_init_done`] and the `GSP_INIT_DONE` result.
//!
//! On the board (os-279, 11 Nov 2026) GSP-RM sent 843 records before
//! refusing to initialise with `0x59`; the next run journals the first four
//! and each new kind in full, which is up to 20 blocks of ~10 rows in a
//! 3000-row journal. This module turns them into a few lines: how many
//! records, which kinds (sorted by how often they appear among the
//! journaled ones), the asserts with address + line, where they came
//! relative to the sequencer (before / after `CORE_RESUME`), and what
//! `GSP_INIT_DONE` said. It also reads the *old* journal format (os-279:
//! `valoare I8 mesaj GSP-RM sarit (functia) = 0x00001020`) so the two runs
//! can be compared.
//!
//! Pure text processing (like [`crate::chain_log`]): never touches a board,
//! never says more than the journal does.

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use super::rpc::event;

/// One NOCAT record journaled in full.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecordDigest {
    /// `NOCAT: inregistrarea nr.` (1-based count of records received).
    pub index: u32,
    /// `text NOCAT: recType` (ASSERT, RC, ENGINE…).
    pub rec_type: Option<String>,
    pub source: Option<String>,
    pub subsystem: Option<u32>,
    /// Non-ASSERT records: `errorCode`, low 32 bits, and its `NV_STATUS` name.
    pub error_code: Option<u32>,
    pub error_name: Option<String>,
    /// ASSERT records: address relative to the RM image and source line.
    pub assert_addr: Option<u32>,
    pub assert_line: Option<u32>,
    /// ASSERT records: journaled call-stack entries.
    pub stack: Vec<u32>,
    /// ASSERT records: `count` when more than 1.
    pub count: Option<u32>,
    pub faulting_engine: Option<String>,
    /// The record arrived after the sequencer request (`CORE_RESUME` path).
    pub after_sequencer: bool,
}

impl RecordDigest {
    /// What makes two journaled records "the same complaint" — the same key
    /// the driver uses (`NocatRecord::kind`), read back from the rows.
    pub fn kind(&self) -> (Option<&str>, Option<u32>, Option<u32>, Option<u32>) {
        (
            self.rec_type.as_deref(),
            self.assert_addr.or(self.error_code),
            self.assert_line,
            self.subsystem,
        )
    }
}

/// Everything the journal says about GSP-RM's messages before `GSP_INIT_DONE`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NocatDigest {
    /// Records journaled in full, in journal order.
    pub records: Vec<RecordDigest>,
    /// Old-format rows (os-279): `0x1020` messages named but not decoded.
    pub named_only_nocat: u32,
    /// Old-format rows: `0x100c` messages named but not decoded.
    pub named_only_libos: u32,
    /// `I8 NOCAT: mesaj mai scurt decat inregistrarea`.
    pub short_records: u32,
    /// `UCODE_LIBOS_PRINT: libosPrintBufSize` rows (one per journaled print).
    pub libos_sizes: Vec<u32>,
    /// The summary rows at the end of the wait (`None` = row not in journal).
    pub summary_received: Option<u32>,
    pub summary_kinds: Option<u32>,
    pub summary_kinds_dropped: Option<u32>,
    pub summary_libos: Option<u32>,
    pub summary_skipped_total: Option<u32>,
    /// `GSP_INIT_DONE: rpc_result` and its `NV_STATUS` name.
    pub init_done_result: Option<u32>,
    pub init_done_name: Option<String>,
    /// `nota: I8: eroarea vine din …` (what follows "din ").
    pub error_origin: Option<String>,
    /// The sequencer request was in the journal (so "before / after" means something).
    pub sequencer_seen: bool,
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

/// `nota: <n>`, possibly after a log prefix.
fn note_row(line: &str) -> Option<&str> {
    Some(line[line.find("nota: ")? + 6..].trim_end())
}

const WHAT_INDEX: &str = "NOCAT: inregistrarea nr.";
const WHAT_SUBSYSTEM: &str = "NOCAT: subsystem";
const WHAT_ERROR_LO: &str = "NOCAT: errorCode (32 de biti de jos";
const WHAT_ASSERT_ADDR: &str = "NOCAT ASSERT: adresa assert-ului";
const WHAT_ASSERT_LINE: &str = "NOCAT ASSERT: linia din sursa RM";
const WHAT_ASSERT_CALL: &str = "NOCAT ASSERT: apel #";
const WHAT_ASSERT_COUNT: &str = "NOCAT ASSERT: de cate ori (count)";
const WHAT_SHORT: &str = "I8 NOCAT: mesaj mai scurt decat inregistrarea";
const WHAT_SUM_RECEIVED: &str = "I8 NOCAT (0x1020): inregistrari de diagnostic primite";
const WHAT_SUM_KINDS: &str = "I8 NOCAT: feluri distincte";
const WHAT_SUM_DROPPED: &str = "I8 NOCAT: inregistrari de feluri noi peste limita";
const WHAT_SUM_LIBOS: &str = "I8 UCODE_LIBOS_PRINT (0x100c): jurnale de microcod primite";
const WHAT_SUM_SKIPPED: &str = "I8 mesaje GSP-RM sarite in asteptarea GSP_INIT_DONE: total";
const WHAT_LIBOS_SIZE: &str = "UCODE_LIBOS_PRINT: libosPrintBufSize";
const WHAT_SKIPPED_FN: &str = "I8 mesaj GSP-RM sarit (functia)";
const WHAT_INIT_DONE: &str = "GSP_INIT_DONE: rpc_result (0 = bine";
const TEXT_REC_TYPE: &str = "NOCAT: recType";
const TEXT_ERROR_NAME: &str = "NOCAT: errorCode (nume NV_STATUS)";
const TEXT_SOURCE: &str = "NOCAT: source";
const TEXT_ENGINE: &str = "NOCAT: faultingEngine";
const TEXT_INIT_DONE_NAME: &str = "GSP_INIT_DONE: rpc_result (nume NV_STATUS)";
const NOTE_SEQUENCER: &str = "I8 secventiator: cererea GSP-RM";
const NOTE_ORIGIN: &str = "I8: eroarea vine din ";

/// Reads a whole journal (any other rows are ignored).
pub fn digest(text: &str) -> NocatDigest {
    let mut d = NocatDigest::default();
    for line in text.lines() {
        if let Some(n) = note_row(line) {
            if n.starts_with(NOTE_SEQUENCER) {
                d.sequencer_seen = true;
            } else if let Some(o) = n.strip_prefix(NOTE_ORIGIN) {
                d.error_origin = Some(String::from(o));
            }
            continue;
        }
        if let Some((what, v)) = text_row(line) {
            let Some(r) = d.records.last_mut() else {
                if what == TEXT_INIT_DONE_NAME {
                    d.init_done_name = Some(String::from(v));
                }
                continue;
            };
            match what {
                TEXT_REC_TYPE => r.rec_type = Some(String::from(v)),
                TEXT_ERROR_NAME => r.error_name = Some(String::from(v)),
                TEXT_SOURCE => r.source = Some(String::from(v)),
                TEXT_ENGINE => r.faulting_engine = Some(String::from(v)),
                TEXT_INIT_DONE_NAME => d.init_done_name = Some(String::from(v)),
                _ => {}
            }
            continue;
        }
        let Some((what, v)) = value_row(line) else {
            continue;
        };
        if what == WHAT_INDEX {
            d.records.push(RecordDigest {
                index: v,
                after_sequencer: d.sequencer_seen,
                ..RecordDigest::default()
            });
        } else if what.starts_with(WHAT_SHORT) {
            d.short_records += 1;
        } else if what.starts_with(WHAT_SUM_RECEIVED) {
            d.summary_received = Some(v);
        } else if what.starts_with(WHAT_SUM_KINDS) {
            d.summary_kinds = Some(v);
        } else if what.starts_with(WHAT_SUM_DROPPED) {
            d.summary_kinds_dropped = Some(v);
        } else if what.starts_with(WHAT_SUM_LIBOS) {
            d.summary_libos = Some(v);
        } else if what.starts_with(WHAT_SUM_SKIPPED) {
            d.summary_skipped_total = Some(v);
        } else if what.starts_with(WHAT_LIBOS_SIZE) {
            d.libos_sizes.push(v);
        } else if what == WHAT_SKIPPED_FN {
            match v {
                event::GSP_POST_NOCAT_RECORD => d.named_only_nocat += 1,
                event::UCODE_LIBOS_PRINT => d.named_only_libos += 1,
                _ => {}
            }
        } else if what.starts_with(WHAT_INIT_DONE) {
            d.init_done_result = Some(v);
        } else if let Some(r) = d.records.last_mut() {
            if what == WHAT_SUBSYSTEM {
                r.subsystem = Some(v);
            } else if what.starts_with(WHAT_ERROR_LO) {
                r.error_code = Some(v);
            } else if what.starts_with(WHAT_ASSERT_ADDR) {
                r.assert_addr = Some(v);
            } else if what == WHAT_ASSERT_LINE {
                r.assert_line = Some(v);
            } else if what.starts_with(WHAT_ASSERT_CALL) {
                r.stack.push(v);
            } else if what == WHAT_ASSERT_COUNT {
                r.count = Some(v);
            }
        }
    }
    d
}

impl NocatDigest {
    /// Nothing about GSP-RM's messages in this journal (a run that stopped
    /// before I8, or an older driver).
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
            && self.named_only_nocat == 0
            && self.named_only_libos == 0
            && self.summary_received.is_none()
            && self.init_done_result.is_none()
    }

    /// The distinct kinds among the journaled records with how many of those
    /// records they cover, most frequent first (ties: journal order). The
    /// counts are of *journaled* records only; the board's totals are in
    /// the summary rows.
    pub fn kinds(&self) -> Vec<(&RecordDigest, u32)> {
        let mut out: Vec<(&RecordDigest, u32)> = Vec::new();
        for r in &self.records {
            match out.iter_mut().find(|(k, _)| k.kind() == r.kind()) {
                Some((_, n)) => *n += 1,
                None => out.push((r, 1)),
            }
        }
        out.sort_by_key(|(_, n)| core::cmp::Reverse(*n));
        out
    }

    /// Journaled records before / after the sequencer request.
    pub fn before_after_sequencer(&self) -> (u32, u32) {
        let after = self.records.iter().filter(|r| r.after_sequencer).count() as u32;
        (self.records.len() as u32 - after, after)
    }
}

impl fmt::Display for NocatDigest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_empty() {
            return writeln!(f, "NOCAT: niciun rand despre mesajele GSP-RM in jurnal");
        }
        if self.named_only_nocat != 0 || self.named_only_libos != 0 {
            writeln!(
                f,
                "NOCAT (format vechi, ca os-279): {} mesaje 0x1020 (GSP_POST_NOCAT_RECORD) si {} mesaje 0x100c (UCODE_LIBOS_PRINT) numite, dar nedecodate",
                self.named_only_nocat, self.named_only_libos
            )?;
        }
        if let Some(n) = self.summary_received {
            write!(
                f,
                "NOCAT: {n} inregistrari de diagnostic primite de la GSP-RM"
            )?;
            if let Some(k) = self.summary_kinds {
                write!(f, ", {k} feluri distincte")?;
            }
            if let Some(dr) = self.summary_kinds_dropped {
                write!(
                    f,
                    " (+ {dr} inregistrari de feluri noi peste limita, nejurnalizate)"
                )?;
            }
            if let Some(l) = self.summary_libos {
                write!(f, "; {l} jurnale de microcod (LibOS, binare)")?;
            }
            if let Some(t) = self.summary_skipped_total {
                write!(f, "; mesaje sarite in total: {t}")?;
            }
            writeln!(f)?;
        }
        if !self.records.is_empty() {
            let (before, after) = self.before_after_sequencer();
            write!(f, "  jurnalizate in intregime: {}", self.records.len())?;
            if self.sequencer_seen {
                write!(f, " (inainte de secventiator: {before}, dupa: {after})")?;
            }
            if self.short_records != 0 {
                write!(
                    f,
                    "; mesaje mai scurte decat inregistrarea: {}",
                    self.short_records
                )?;
            }
            writeln!(f)?;
            writeln!(
                f,
                "  feluri, cele mai dese intai (numarul = cate dintre cele jurnalizate):"
            )?;
            for (r, n) in self.kinds() {
                write!(f, "    x{n:<3} {:<8}", r.rec_type.as_deref().unwrap_or("?"))?;
                match (r.assert_addr, r.assert_line) {
                    (Some(a), line) => {
                        write!(f, " adresa {a:#010x}")?;
                        if let Some(l) = line {
                            write!(f, " linia {l}")?;
                        }
                    }
                    (None, _) => {
                        if let Some(e) = r.error_code {
                            write!(f, " errorCode {e:#x}")?;
                        }
                        if let Some(name) = &r.error_name {
                            write!(f, " {name}")?;
                        }
                    }
                }
                if let Some(s) = &r.source {
                    write!(f, " source \"{s}\"")?;
                }
                if let Some(s) = r.subsystem {
                    write!(f, " subsystem {s:#x}")?;
                }
                if let Some(e) = &r.faulting_engine {
                    write!(f, " motor \"{e}\"")?;
                }
                if !r.stack.is_empty() {
                    write!(f, " stiva [")?;
                    for (i, a) in r.stack.iter().enumerate() {
                        if i != 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{a:#x}")?;
                    }
                    write!(f, "]")?;
                }
                if let Some(c) = r.count {
                    write!(f, " count {c}")?;
                }
                writeln!(f, " (prima: nr. {})", r.index)?;
            }
        }
        if !self.libos_sizes.is_empty() {
            write!(
                f,
                "  jurnale LibOS jurnalizate: {} (marimi:",
                self.libos_sizes.len()
            )?;
            for s in &self.libos_sizes {
                write!(f, " {s}")?;
            }
            writeln!(f, " octeti; continut binar, nedecodat)")?;
        }
        match self.init_done_result {
            Some(0) => writeln!(f, "GSP_INIT_DONE: rpc_result 0 (bine)")?,
            Some(v) => {
                write!(f, "GSP_INIT_DONE: rpc_result {v:#x}")?;
                if let Some(n) = &self.init_done_name {
                    write!(f, " ({n})")?;
                }
                writeln!(f)?;
            }
            None => writeln!(f, "GSP_INIT_DONE: nu e in jurnal")?,
        }
        if let Some(o) = &self.error_origin {
            writeln!(f, "I8: eroarea vine din {o}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    /// Rows as the OS writes them on the stick (`[ time] gpu: ` prefix).
    fn journal(rows: &[&str]) -> String {
        let mut s = String::new();
        for (i, r) in rows.iter().enumerate() {
            s.push_str(&format!(
                "[   {}.{:06}] gpu: {r}\n",
                60 + i / 10,
                i * 1234 % 1_000_000
            ));
        }
        s
    }

    #[test]
    fn the_old_format_is_counted_and_the_result_named() {
        let mut rows = alloc::vec![
            "nota: lant (C-54): incepe, scrieri PERMISE (fereastra cu Adrian)",
            "bar0 r 0x300000 = 0xeb7faa55",
        ];
        rows.extend(core::iter::repeat_n(
            "valoare I8 mesaj GSP-RM sarit (functia) = 0x00001020",
            843,
        ));
        rows.push("nota: I8 secventiator: cererea GSP-RM, comenzile urmeaza");
        rows.push("secv #0 POLL 0x110040 0x80000000 0x80000000 0x0 0x3");
        rows.push("valoare I8 mesaj GSP-RM sarit (functia) = 0x0000100c");
        rows.push("valoare I8 mesaj GSP-RM sarit (functia) = 0x0000100c");
        rows.push(
            "valoare GSP_INIT_DONE: rpc_result (0 = bine; altfel cod NV_STATUS) = 0x00000059",
        );
        rows.push("valoare GSP_INIT_DONE: rpc_result_private = 0x00000059");
        rows.push("nota: I8: eroarea vine din GSP_INIT_DONE");
        rows.push(
            "valoare I8 mesaje GSP-RM sarite in asteptarea GSP_INIT_DONE: total = 0x0000034d",
        );
        rows.push("pas i8-gsp-init: ESEC");
        let d = digest(&journal(&rows));
        assert!(!d.is_empty());
        assert_eq!(d.named_only_nocat, 843);
        assert_eq!(d.named_only_libos, 2);
        assert!(d.records.is_empty());
        assert_eq!(d.summary_skipped_total, Some(845));
        assert_eq!(d.summary_received, None);
        assert_eq!(d.init_done_result, Some(0x59));
        assert_eq!(d.init_done_name, None);
        assert_eq!(d.error_origin.as_deref(), Some("GSP_INIT_DONE"));
        assert!(d.sequencer_seen);
        let s = format!("{d}");
        assert!(s.contains("843 mesaje 0x1020 (GSP_POST_NOCAT_RECORD) si 2 mesaje 0x100c"));
        assert!(s.contains("GSP_INIT_DONE: rpc_result 0x59\n"));
        assert!(s.contains("I8: eroarea vine din GSP_INIT_DONE"));
    }

    #[test]
    fn a_journal_without_i8_rows_is_empty() {
        let d = digest(&journal(&[
            "pas i6-fwsec-frts: ok",
            "bar0 w 0x110040 = 0x0",
            "pas i7-booter-gsp: ESEC",
        ]));
        assert!(d.is_empty());
        assert_eq!(
            format!("{d}"),
            "NOCAT: niciun rand despre mesajele GSP-RM in jurnal\n"
        );
    }
}
