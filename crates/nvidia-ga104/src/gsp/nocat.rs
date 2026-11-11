//! The two messages GSP-RM sent on the board before refusing to initialise
//! (test `os-279`, 11 Nov 2026: 843 × `0x1020`, 2 × `0x100c`, then
//! `GSP_INIT_DONE` with `rpc_result = 0x59`), decoded field by field so the
//! journal can say what GSP-RM complained about. Decoding only; nothing here
//! touches the board.
//!
//! - `GSP_POST_NOCAT_RECORD` (0x1020): the payload is
//!   `rpc_gsp_post_nocat_record_v01_00 { NvU32 data; }` and the kernel driver
//!   reads `&data` as an `NV2080CtrlNocatJournalInsertRecord`
//!   (`kernel_gsp.c`, `_kgspRpcGspPostNocatRecord`): a diagnostic record RM on
//!   the GSP asks the CPU side to put in its journal (an assert, an RC error,
//!   a bugcheck…). Layout from `ctrl2080nvd.h`, natural alignment, `NvU64`
//!   fields 8-aligned (offsets checked with a C compiler, test below):
//!
//!   ```text
//!   +0    flags           u32
//!   +8    timestamp       u64
//!   +16   recType         u8      (0 UNKNOWN, 1 BUGCHECK, 2 ENGINE, 3 TDR, 4 RC, 5 ASSERT, 6 ANY)
//!   +20   bugcheck        u32
//!   +24   source[65]      C string
//!   +92   subsystem       u32
//!   +96   errorCode       u64
//!   +104  faultingEngine[65] C string
//!   +172  tdrReason       u32
//!   +176  diagBufferLen   u32
//!   +180  diagBuffer[1024]
//!   size 1208
//!   ```
//!
//!   `diagBuffer` is RM-internal binary data: only its length is journaled.
//! - `UCODE_LIBOS_PRINT` (0x100c): `rpc_ucode_libos_print_v1E_08
//!   { NvU32 ucodeEngDesc; NvU32 libosPrintBufSize; NvU8 libosPrintBuf[]; }`
//!   — a log buffer of another microcode (the PMU) forwarded by GSP-RM
//!   (`_kgspRpcUcodeLibosPrint`). Its content is in LibOS binary log format
//!   (decodable only with that microcode's own symbol tables, which we do not
//!   open — see RISKS.md); the engine descriptor and the size are journaled.
//!
//! Sources (MIT, open-gpu-kernel-modules@570.144): `ctrl2080nvd.h`,
//! `g_rpc-structures.h`, `kernel_gsp.c` — read, re-written (docs/PROVENANCE.md).

use gpu_core::diag::{Event, Journal};

use super::{get_u32, get_u64};

/// `NV2080_NOCAT_JOURNAL_MAX_STR_LEN`.
pub const STR_LEN: usize = 65;
/// `NV2080_NOCAT_JOURNAL_MAX_DIAG_BUFFER`.
pub const DIAG_BUFFER_LEN: usize = 1024;
/// `sizeof(NV2080CtrlNocatJournalInsertRecord)`.
pub const RECORD_SIZE: usize = 1208;

mod off {
    pub const FLAGS: usize = 0;
    pub const TIMESTAMP: usize = 8;
    pub const REC_TYPE: usize = 16;
    pub const BUGCHECK: usize = 20;
    pub const SOURCE: usize = 24;
    pub const SUBSYSTEM: usize = 92;
    pub const ERROR_CODE: usize = 96;
    pub const FAULTING_ENGINE: usize = 104;
    pub const TDR_REASON: usize = 172;
    pub const DIAG_BUFFER_LEN: usize = 176;
    pub const DIAG_BUFFER: usize = 180;
}

const _: () = assert!(off::DIAG_BUFFER + DIAG_BUFFER_LEN <= RECORD_SIZE);

/// `NV2080_NOCAT_JOURNAL_REC_TYPE_*` as a word for the journal.
pub fn rec_type_name(rec_type: u8) -> &'static str {
    match rec_type {
        0 => "UNKNOWN",
        1 => "BUGCHECK",
        2 => "ENGINE",
        3 => "TDR",
        4 => "RC",
        5 => "ASSERT",
        6 => "ANY",
        _ => "?",
    }
}

/// One `NV2080CtrlNocatJournalInsertRecord`, the fixed fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NocatRecord<'a> {
    pub flags: u32,
    pub timestamp: u64,
    pub rec_type: u8,
    pub bugcheck: u32,
    /// `source` up to its NUL (at most 64 bytes).
    pub source: &'a [u8],
    pub subsystem: u32,
    pub error_code: u64,
    /// `faultingEngine` up to its NUL.
    pub faulting_engine: &'a [u8],
    pub tdr_reason: u32,
    pub diag_buffer_len: u32,
}

fn c_str(b: &[u8]) -> &[u8] {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    &b[..end]
}

impl<'a> NocatRecord<'a> {
    /// Decodes the record at the start of a `GSP_POST_NOCAT_RECORD` payload.
    /// `None` if the payload is shorter than the record (the board's were
    /// one 4 KiB element: 80 + 1208 bytes fit).
    pub fn decode(payload: &'a [u8]) -> Option<Self> {
        if payload.len() < RECORD_SIZE {
            return None;
        }
        Some(NocatRecord {
            flags: get_u32(payload, off::FLAGS),
            timestamp: get_u64(payload, off::TIMESTAMP),
            rec_type: payload[off::REC_TYPE],
            bugcheck: get_u32(payload, off::BUGCHECK),
            source: c_str(&payload[off::SOURCE..off::SOURCE + STR_LEN]),
            subsystem: get_u32(payload, off::SUBSYSTEM),
            error_code: get_u64(payload, off::ERROR_CODE),
            faulting_engine: c_str(&payload[off::FAULTING_ENGINE..off::FAULTING_ENGINE + STR_LEN]),
            tdr_reason: get_u32(payload, off::TDR_REASON),
            diag_buffer_len: get_u32(payload, off::DIAG_BUFFER_LEN),
        })
    }

    /// What makes two records "the same complaint": type, error code,
    /// subsystem (the journal prints each distinct one once, then counts).
    pub fn kind(&self) -> (u8, u64, u32) {
        (self.rec_type, self.error_code, self.subsystem)
    }

    /// Writes the record to the journal, one named row per field that says
    /// something (zero fields that carry no information are left out).
    pub fn journal(&self, j: &mut Journal, index: u32) {
        j.push(Event::Value {
            what: "NOCAT: inregistrarea nr.",
            value: index,
        });
        j.push(Event::Value {
            what: "NOCAT: recType (1 BUGCHECK, 2 ENGINE, 3 TDR, 4 RC, 5 ASSERT)",
            value: u32::from(self.rec_type),
        });
        j.push(Event::text(
            "NOCAT: recType",
            rec_type_name(self.rec_type).as_bytes(),
        ));
        j.push(Event::Value {
            what: "NOCAT: errorCode (cod NV_STATUS, 32 de biti de jos)",
            value: self.error_code as u32,
        });
        if self.error_code >> 32 != 0 {
            j.push(Event::Value {
                what: "NOCAT: errorCode (32 de biti de sus)",
                value: (self.error_code >> 32) as u32,
            });
        }
        if let Some(name) = nvidia_rm570::status_name(self.error_code as u32) {
            j.push(Event::text(
                "NOCAT: errorCode (nume NV_STATUS)",
                name.as_bytes(),
            ));
        }
        j.push(Event::text("NOCAT: source", self.source));
        j.push(Event::Value {
            what: "NOCAT: subsystem",
            value: self.subsystem,
        });
        if !self.faulting_engine.is_empty() {
            j.push(Event::text("NOCAT: faultingEngine", self.faulting_engine));
        }
        if self.bugcheck != 0 {
            j.push(Event::Value {
                what: "NOCAT: bugcheck",
                value: self.bugcheck,
            });
        }
        if self.tdr_reason != 0 {
            j.push(Event::Value {
                what: "NOCAT: tdrReason",
                value: self.tdr_reason,
            });
        }
        if self.flags != 0 {
            j.push(Event::Value {
                what: "NOCAT: flags",
                value: self.flags,
            });
        }
        j.push(Event::Value {
            what: "NOCAT: diagBufferLen (continut RM intern, nejurnalizat)",
            value: self.diag_buffer_len,
        });
        j.push(Event::Value {
            what: "NOCAT: timestamp (32 de biti de jos)",
            value: self.timestamp as u32,
        });
    }
}

/// `rpc_ucode_libos_print_v1E_08`, the two fixed fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LibosPrint {
    pub ucode_eng_desc: u32,
    pub print_buf_size: u32,
    /// Bytes of `libosPrintBuf` actually present in the payload.
    pub present: u32,
}

impl LibosPrint {
    pub fn decode(payload: &[u8]) -> Option<Self> {
        if payload.len() < 8 {
            return None;
        }
        Some(LibosPrint {
            ucode_eng_desc: get_u32(payload, 0),
            print_buf_size: get_u32(payload, 4),
            present: (payload.len() - 8) as u32,
        })
    }

    pub fn journal(&self, j: &mut Journal) {
        j.push(Event::Value {
            what: "UCODE_LIBOS_PRINT: ucodeEngDesc (descriptorul motorului, RM: ENG_PMU)",
            value: self.ucode_eng_desc,
        });
        j.push(Event::Value {
            what: "UCODE_LIBOS_PRINT: libosPrintBufSize (jurnal LibOS binar, nedecodat)",
            value: self.print_buf_size,
        });
        if self.present != self.print_buf_size {
            j.push(Event::Value {
                what: "UCODE_LIBOS_PRINT: octeti prezenti in mesaj",
                value: self.present,
            });
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec::Vec;

    /// Builds a record the way the C struct lays it out.
    pub(crate) fn record(rec_type: u8, error: u64, subsystem: u32, source: &[u8]) -> Vec<u8> {
        let mut p = alloc::vec![0u8; RECORD_SIZE];
        super::super::put_u32(&mut p, off::FLAGS, 0);
        super::super::put_u64(&mut p, off::TIMESTAMP, 0x0000_0123_4567_89ab);
        p[off::REC_TYPE] = rec_type;
        p[off::SOURCE..off::SOURCE + source.len()].copy_from_slice(source);
        super::super::put_u32(&mut p, off::SUBSYSTEM, subsystem);
        super::super::put_u64(&mut p, off::ERROR_CODE, error);
        p[off::FAULTING_ENGINE..off::FAULTING_ENGINE + 3].copy_from_slice(b"GSP");
        super::super::put_u32(&mut p, off::DIAG_BUFFER_LEN, 40);
        p
    }

    #[test]
    fn offsets_are_the_c_layout() {
        // offsetof() printed by a C compiler for the struct in ctrl2080nvd.h
        // (natural alignment, NV_DECLARE_ALIGNED(…, 8) on the two u64s).
        assert_eq!(
            [
                off::FLAGS,
                off::TIMESTAMP,
                off::REC_TYPE,
                off::BUGCHECK,
                off::SOURCE,
                off::SUBSYSTEM,
                off::ERROR_CODE,
                off::FAULTING_ENGINE,
                off::TDR_REASON,
                off::DIAG_BUFFER_LEN,
                off::DIAG_BUFFER,
            ],
            [0, 8, 16, 20, 24, 92, 96, 104, 172, 176, 180]
        );
        assert_eq!(off::DIAG_BUFFER + DIAG_BUFFER_LEN, 1204);
        assert_eq!(RECORD_SIZE, 1208); // 1204 rounded up to the 8-byte alignment
        // The whole record fits one queue element with the headers.
        const { assert!(80 + RECORD_SIZE <= 0x1000) };
    }

    #[test]
    fn a_record_is_decoded_and_journaled() {
        let p = record(5, 0x59, 7, b"GSP-RM");
        let r = NocatRecord::decode(&p).unwrap();
        assert_eq!(r.rec_type, 5);
        assert_eq!(r.error_code, 0x59);
        assert_eq!(r.subsystem, 7);
        assert_eq!(r.source, b"GSP-RM");
        assert_eq!(r.faulting_engine, b"GSP");
        assert_eq!(r.diag_buffer_len, 40);
        assert_eq!(r.kind(), (5, 0x59, 7));
        assert!(NocatRecord::decode(&p[..RECORD_SIZE - 1]).is_none());
        let mut j = Journal::new(64);
        r.journal(&mut j, 3);
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        assert_eq!(
            rows,
            [
                "valoare NOCAT: inregistrarea nr. = 0x00000003",
                "valoare NOCAT: recType (1 BUGCHECK, 2 ENGINE, 3 TDR, 4 RC, 5 ASSERT) = 0x00000005",
                "text NOCAT: recType = \"ASSERT\"",
                "valoare NOCAT: errorCode (cod NV_STATUS, 32 de biti de jos) = 0x00000059",
                "text NOCAT: errorCode (nume NV_STATUS) = \"NV_ERR_OPERATING_SYSTEM\"",
                "text NOCAT: source = \"GSP-RM\"",
                "valoare NOCAT: subsystem = 0x00000007",
                "text NOCAT: faultingEngine = \"GSP\"",
                "valoare NOCAT: diagBufferLen (continut RM intern, nejurnalizat) = 0x00000028",
                "valoare NOCAT: timestamp (32 de biti de jos) = 0x456789ab",
            ]
        );
        // an error code the header does not list: no name row, the number stays
        let p = record(2, 0xffff_0001_dead_0000, 0, b"");
        let r = NocatRecord::decode(&p).unwrap();
        let mut j = Journal::new(64);
        r.journal(&mut j, 0);
        let rows: Vec<String> = j.events().map(|e| alloc::format!("{e}")).collect();
        assert!(rows.contains(&String::from(
            "valoare NOCAT: errorCode (cod NV_STATUS, 32 de biti de jos) = 0xdead0000"
        )));
        assert!(rows.contains(&String::from(
            "valoare NOCAT: errorCode (32 de biti de sus) = 0xffff0001"
        )));
        assert!(!rows.iter().any(|r| r.contains("nume NV_STATUS")));
        assert!(rows.contains(&String::from("text NOCAT: source = \"\"")));
    }

    #[test]
    fn libos_print_fields() {
        let mut p = alloc::vec![0u8; 8 + 100];
        super::super::put_u32(&mut p, 0, 0x0000_1234);
        super::super::put_u32(&mut p, 4, 100);
        let l = LibosPrint::decode(&p).unwrap();
        assert_eq!(
            (l.ucode_eng_desc, l.print_buf_size, l.present),
            (0x1234, 100, 100)
        );
        let mut j = Journal::new(8);
        l.journal(&mut j);
        assert_eq!(j.len(), 2);
        assert!(LibosPrint::decode(&p[..7]).is_none());
        let l = LibosPrint::decode(&p[..50]).unwrap();
        let mut j = Journal::new(8);
        l.journal(&mut j);
        assert_eq!(j.len(), 3);
    }
}
