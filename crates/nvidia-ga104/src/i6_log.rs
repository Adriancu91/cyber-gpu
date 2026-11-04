//! C-51: checking the I6 journal that comes back from the board.
//!
//! Without the tinygrad reference run (cancelled 1 nov: on the real board
//! only Cyber Universe runs), the evidence for I6 is our own journal, saved
//! by the OS on the stick, one line per access in the [`Event`] text format
//! (`bar0 w 0x110624: 0x00000000 <- 0x00000080 => 0x00000080`), possibly
//! after a log prefix (`[ 606.590] INFO  gpu: `).
//!
//! [`parse_line`] reads those lines back; [`check_i6`] checks the write
//! sequence against the order our I6 code must follow (the same order the
//! simulator runs, taken from RM 570.144 `kgspExecuteHsFalcon_GA102` and
//! `s_dmaTransfer_GA102`, MIT): reset → falcon core → FBIF/DMACTL/TRANSCFG
//! → per section the DMA base then one (MOFFS, FBOFFS, CMD) triple per
//! 256-byte block → the four boot-ROM registers → BOOTVEC → mailboxes → start.
//! Anything else — an unknown register, a write outside the GSP window, a
//! skipped or reordered block — is reported with its line index.
//!
//! Pure text processing: this module never touches a board.

use alloc::vec::Vec;

use gpu_core::diag::Event;

/// GSP falcon window (the only one I6 may write).
pub const GSP_LO: u64 = 0x11_0000;
pub const GSP_HI: u64 = 0x11_3fff;

const ENGINE: u64 = 0x1103c0;
const BCR_CTRL: u64 = 0x111668;
const FBIF_CTL: u64 = 0x110624;
const DMACTL: u64 = 0x11010c;
const TRANSCFG0: u64 = 0x110600;
const DMATRFBASE: u64 = 0x110110;
const DMATRFBASE1: u64 = 0x110128;
const DMATRFMOFFS: u64 = 0x110114;
const DMATRFFBOFFS: u64 = 0x11011c;
const DMATRFCMD: u64 = 0x110118;
const BROM_PARAADDR0: u64 = 0x111210;
const BROM_ENGIDMASK: u64 = 0x11119c;
const BROM_CURR_UCODE_ID: u64 = 0x111198;
const MOD_SEL: u64 = 0x111180;
const BOOTVEC: u64 = 0x110104;
const MAILBOX0: u64 = 0x110040;
const MAILBOX1: u64 = 0x110044;
const CPUCTL: u64 = 0x110100;
const CPUCTL_ALIAS: u64 = 0x110130;
/// `NV_PFALCON_FALCON_DMATRFCMD_IMEM` (bit 4): the block goes to IMEM.
const CMD_IMEM: u32 = 1 << 4;

/// Parses one journal line (any prefix before `bar`/`cfg` is ignored).
/// Lines that are not register accesses (`pas …`, `nota: …`) give `None`.
pub fn parse_line(line: &str) -> Option<Event> {
    let start = line
        .find("bar")
        .into_iter()
        .chain(line.find("cfg "))
        .min()?;
    let mut t = line[start..].split_whitespace();
    let head = t.next()?;
    let op = t.next()?;
    let off = hex(t.next()?.trim_end_matches(':'))?;
    if head == "cfg" {
        let offset = u16::try_from(off).ok()?;
        return match op {
            "r" => {
                (t.next()? == "=").then_some(())?;
                Some(Event::ConfigRead {
                    offset,
                    value: hex32(t.next()?)?,
                })
            }
            "w" => {
                let (before, written, after) = write_values(&mut t)?;
                Some(Event::ConfigWrite {
                    offset,
                    before,
                    written,
                    after,
                })
            }
            _ => None,
        };
    }
    let bar = head.strip_prefix("bar")?.parse::<u8>().ok()?;
    match op {
        "r" => {
            (t.next()? == "=").then_some(())?;
            Some(Event::MmioRead {
                bar,
                offset: off,
                value: hex32(t.next()?)?,
            })
        }
        "w" => {
            let (before, written, after) = write_values(&mut t)?;
            Some(Event::MmioWrite {
                bar,
                offset: off,
                before,
                written,
                after,
            })
        }
        _ => None,
    }
}

fn write_values<'a>(t: &mut impl Iterator<Item = &'a str>) -> Option<(u32, u32, u32)> {
    let before = hex32(t.next()?)?;
    (t.next()? == "<-").then_some(())?;
    let written = hex32(t.next()?)?;
    (t.next()? == "=>").then_some(())?;
    let after = hex32(t.next()?)?;
    Some((before, written, after))
}

fn hex(s: &str) -> Option<u64> {
    u64::from_str_radix(s.strip_prefix("0x")?, 16).ok()
}

fn hex32(s: &str) -> Option<u32> {
    u32::try_from(hex(s)?).ok()
}

/// What a correct I6 journal contained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct I6Summary {
    /// Blocks of 256 B per DMA section, in order (IMEM first, then DMEM).
    pub sections: Vec<(bool, u32)>,
    pub writes: usize,
}

/// First place where the journal leaves the expected order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I6LogError {
    /// Index among the BAR0 writes (0-based).
    pub write_index: usize,
    pub offset: u64,
    pub what: &'static str,
}

/// Checks the BAR0 writes of an I6 journal. Reads and other lines are
/// skipped; any write outside the GSP window is an error. Config-space
/// writes (I5g, Bus Master) are not part of I6 and are rejected here too.
pub fn check_i6(events: &[Event]) -> Result<I6Summary, I6LogError> {
    let mut w: Vec<(u64, u32)> = Vec::new();
    for e in events {
        match *e {
            Event::MmioWrite {
                bar: 0,
                offset,
                written,
                ..
            } => {
                if !(GSP_LO..=GSP_HI).contains(&offset) {
                    return Err(err(w.len(), offset, "scriere in afara ferestrei GSP"));
                }
                w.push((offset, written));
            }
            Event::MmioWrite { offset, .. } => {
                return Err(err(w.len(), offset, "scriere in alt BAR decat BAR0"));
            }
            Event::ConfigWrite { offset, .. } => {
                return Err(err(
                    w.len(),
                    u64::from(offset),
                    "scriere in configuratie in mijlocul I6",
                ));
            }
            _ => {}
        }
    }
    let mut i = 0usize;
    let expect = |i: &mut usize, off: u64, what: &'static str| -> Result<u32, I6LogError> {
        match w.get(*i) {
            Some(&(o, v)) if o == off => {
                *i += 1;
                Ok(v)
            }
            Some(&(o, _)) => Err(err(*i, o, what)),
            None => Err(err(*i, off, "jurnalul se termina inainte de sfarsitul I6")),
        }
    };
    if expect(&mut i, ENGINE, "lipseste resetarea (ENGINE=1)")? != 1
        || expect(&mut i, ENGINE, "lipseste ENGINE=0")? != 0
    {
        return Err(err(i - 1, ENGINE, "resetarea nu e 1 apoi 0"));
    }
    expect(&mut i, BCR_CTRL, "lipseste alegerea nucleului falcon")?;
    expect(&mut i, FBIF_CTL, "lipseste FBIF_CTL")?;
    if expect(&mut i, DMACTL, "lipseste DMACTL")? != 0 {
        return Err(err(i - 1, DMACTL, "DMACTL trebuie 0"));
    }
    expect(&mut i, TRANSCFG0, "lipseste TRANSCFG(0)")?;
    let mut sections = Vec::new();
    while w.get(i).map(|&(o, _)| o) == Some(DMATRFBASE) {
        expect(&mut i, DMATRFBASE, "")?;
        expect(&mut i, DMATRFBASE1, "lipseste DMATRFBASE1")?;
        let mut blocks = 0u32;
        let mut imem = None;
        let mut first = (0u32, 0u32);
        while w.get(i).map(|&(o, _)| o) == Some(DMATRFMOFFS) {
            let m = expect(&mut i, DMATRFMOFFS, "")?;
            let f = expect(&mut i, DMATRFFBOFFS, "lipseste DMATRFFBOFFS")?;
            let c = expect(&mut i, DMATRFCMD, "lipseste DMATRFCMD")?;
            if blocks == 0 {
                first = (m, f);
            } else if m != first.0 + 0x100 * blocks || f != first.1 + 0x100 * blocks {
                return Err(err(i - 3, DMATRFMOFFS, "bloc sarit sau in alta ordine"));
            }
            let is_imem = c & CMD_IMEM != 0;
            if *imem.get_or_insert(is_imem) != is_imem {
                return Err(err(
                    i - 1,
                    DMATRFCMD,
                    "IMEM si DMEM amestecate in aceeasi sectiune",
                ));
            }
            blocks += 1;
        }
        if blocks == 0 {
            return Err(err(i, DMATRFBASE, "sectiune DMA fara niciun bloc"));
        }
        sections.push((imem.unwrap_or(false), blocks));
    }
    if sections.is_empty() {
        return Err(err(i, DMATRFBASE, "nicio copiere DMA"));
    }
    expect(&mut i, BROM_PARAADDR0, "lipseste BROM_PARAADDR(0)")?;
    expect(&mut i, BROM_ENGIDMASK, "lipseste BROM_ENGIDMASK")?;
    expect(&mut i, BROM_CURR_UCODE_ID, "lipseste BROM_CURR_UCODE_ID")?;
    expect(&mut i, MOD_SEL, "lipseste MOD_SEL")?;
    expect(&mut i, BOOTVEC, "lipseste BOOTVEC")?;
    if expect(&mut i, MAILBOX0, "lipseste MAILBOX0")? != 0
        || expect(&mut i, MAILBOX1, "lipseste MAILBOX1")? != 0
    {
        return Err(err(i - 1, MAILBOX0, "cutiile postale trebuie 0 la FWSEC"));
    }
    let start = match w.get(i) {
        Some(&(o, v)) if o == CPUCTL || o == CPUCTL_ALIAS => v,
        Some(&(o, _)) => return Err(err(i, o, "lipseste pornirea (CPUCTL)")),
        None => return Err(err(i, CPUCTL, "jurnalul se termina inainte de pornire")),
    };
    if start & 0b10 == 0 {
        return Err(err(i, CPUCTL, "CPUCTL fara STARTCPU"));
    }
    i += 1;
    if let Some(&(o, _)) = w.get(i) {
        return Err(err(i, o, "scrieri dupa pornirea FWSEC"));
    }
    Ok(I6Summary {
        sections,
        writes: w.len(),
    })
}

fn err(write_index: usize, offset: u64, what: &'static str) -> I6LogError {
    I6LogError {
        write_index,
        offset,
        what,
    }
}

/// C-58: the WPR2 registers as the journal saw them after FWSEC — the last
/// read of HI (0x1fa828) and of LO (0x1fa824) — checked with the same rule as
/// the driver ([`crate::gsp::falcon::compare_frts`]). `None` = the journal has
/// no non-zero HI read (FWSEC did not get that far, or WPR2 stayed down).
pub fn wpr2_after_i6(
    events: &[Event],
    frts: (u64, u64),
) -> Option<(
    u32,
    u32,
    gpu_core::error::GpuResult<crate::gsp::falcon::FrtsCheck>,
)> {
    let last = |reg: u64| {
        events.iter().rev().find_map(|e| match *e {
            Event::MmioRead {
                bar: 0,
                offset,
                value,
            } if offset == reg => Some(value),
            _ => None,
        })
    };
    let hi = last(0x1fa828).filter(|&v| v != 0)?;
    let lo = last(0x1fa824)?;
    Some((lo, hi, crate::gsp::falcon::compare_frts(lo, hi, frts)))
}

/// C-66: [`wpr2_after_i6`] with the driver's current rule
/// ([`crate::gsp::falcon::compare_frts_full`]: also the board's last 128 KiB
/// block).
pub fn wpr2_after_i6_full(
    events: &[Event],
    frts: (u64, u64),
) -> Option<(
    u32,
    u32,
    gpu_core::error::GpuResult<crate::gsp::falcon::FrtsCheckFull>,
)> {
    let (lo, hi, _) = wpr2_after_i6(events, frts)?;
    Some((lo, hi, crate::gsp::falcon::compare_frts_full(lo, hi, frts)))
}

/// C-63 window: everything a whole stick journal (`raport/gpu-jurnal-00N.txt`)
/// says about one I6 attempt, in one pass over the text.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StickJournal {
    /// Lines that parsed as register accesses.
    pub accesses: usize,
    /// Config-space writes `(offset, before, written, after)`: I5g Bus
    /// Master on and the restore afterwards (2 expected for one attempt).
    pub config_writes: Vec<(u16, u32, u32, u32)>,
    /// [`check_i6`] on the BAR0 accesses (config writes taken out first).
    /// `None` when the journal has no BAR0 write at all (e.g. only the
    /// read-only `gpu pregatire-i6`).
    pub i6: Option<Result<I6Summary, I6LogError>>,
    /// [`wpr2_after_i6`] on the same accesses.
    pub wpr2: Option<(
        u32,
        u32,
        gpu_core::error::GpuResult<crate::gsp::falcon::FrtsCheck>,
    )>,
    /// C-66: [`wpr2_after_i6_full`] on the same accesses (the rule the
    /// driver uses now; prefer it to `wpr2`).
    pub wpr2_full: Option<(
        u32,
        u32,
        gpu_core::error::GpuResult<crate::gsp::falcon::FrtsCheckFull>,
    )>,
}

/// Reads a whole stick journal: separates the config writes (Bus Master),
/// checks the I6 write order and the WPR2 registers after FWSEC. Pure text
/// processing, never touches a board.
pub fn check_stick_journal(text: &str, frts: (u64, u64)) -> StickJournal {
    let events: Vec<Event> = text.lines().filter_map(parse_line).collect();
    let mut config_writes = Vec::new();
    let mut rest = Vec::new();
    for e in &events {
        match *e {
            Event::ConfigWrite {
                offset,
                before,
                written,
                after,
            } => config_writes.push((offset, before, written, after)),
            other => rest.push(other),
        }
    }
    let any_bar_write = rest.iter().any(|e| matches!(e, Event::MmioWrite { .. }));
    StickJournal {
        accesses: events.len(),
        config_writes,
        i6: any_bar_write.then(|| check_i6(&rest)),
        wpr2: wpr2_after_i6(&rest, frts),
        wpr2_full: wpr2_after_i6_full(&rest, frts),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn lines_round_trip_with_and_without_os_prefix() {
        let evs = [
            Event::MmioWrite {
                bar: 0,
                offset: 0x110624,
                before: 0,
                written: 0x80,
                after: 0x80,
            },
            Event::MmioRead {
                bar: 0,
                offset: 0x1fa828,
                value: 0,
            },
            Event::ConfigWrite {
                offset: 4,
                before: 0x0010_0002,
                written: 0x0010_0006,
                after: 0x0010_0006,
            },
            Event::ConfigRead {
                offset: 0,
                value: 0x2482_10de,
            },
        ];
        for e in evs {
            assert_eq!(parse_line(&format!("{e}")), Some(e));
            assert_eq!(parse_line(&format!("[ 606.590] INFO  gpu:   {e}")), Some(e));
        }
        assert_eq!(parse_line("pas I6 FWSEC-FRTS: ok"), None);
        assert_eq!(parse_line("nota: WPR2 deja ridicat"), None);
        assert_eq!(parse_line("bar0 w 0xzz: 0x0 <- 0x1 => 0x1"), None);
        assert_eq!(parse_line("bar0 w 0x110624: 0x0 <- 0x1"), None);
    }

    #[test]
    fn wpr2_read_back_from_the_journal_text() {
        // SIM journal lines with the board's FRTS (3 Nov 2026), OS prefix included.
        let frts = (0x1_ffe0_0000u64, 0x1_fff0_0000u64);
        let text = |hi: u32| {
            format!(
                "[ 1.0] INFO  gpu: bar0 r 0x1fa828 = 0x00000000\n\
                 [ 9.0] INFO  gpu: bar0 r 0x1fa828 = {hi:#010x}\n\
                 [ 9.1] INFO  gpu: bar0 r 0x1fa824 = 0x01ffe000\n"
            )
        };
        let evs = |t: &str| t.lines().filter_map(parse_line).collect::<Vec<_>>();
        // Inclusive last page: passes.
        let (lo, hi, r) = wpr2_after_i6(&evs(&text(0x01ffeff0)), frts).unwrap();
        assert_eq!((lo, hi), (0x01ffe000, 0x01ffeff0));
        assert_eq!(
            r.unwrap().end_form,
            crate::gsp::falcon::Wpr2End::InclusiveLastPage
        );
        // Something else: the error names HI.
        let (_, _, r) = wpr2_after_i6(&evs(&text(0x02000000)), frts).unwrap();
        assert!(format!("{}", r.unwrap_err()).contains("0x1fa828"));
        // Only the cold read (HI = 0): nothing to check.
        // C-66: the board's HI of 4 Nov (last 128 KiB block) passes the new rule.
        let (_, hi, r) = wpr2_after_i6_full(&evs(&text(0x01ffee00)), frts).unwrap();
        assert_eq!(hi, 0x01ffee00);
        assert_eq!(
            r.unwrap().form,
            crate::gsp::falcon::Wpr2EndForm::InclusiveLastBlock128K
        );
        let cold = "bar0 r 0x1fa828 = 0x00000000";
        assert!(wpr2_after_i6(&evs(cold), frts).is_none());
    }

    #[test]
    fn empty_or_foreign_journal_is_refused() {
        assert!(check_i6(&[]).is_err());
        let e = check_i6(&[Event::MmioWrite {
            bar: 0,
            offset: 0x1700,
            before: 0,
            written: 1,
            after: 1,
        }])
        .unwrap_err();
        assert_eq!((e.write_index, e.offset), (0, 0x1700));
    }
}
