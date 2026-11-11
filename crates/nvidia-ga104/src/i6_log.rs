//! C-51: checking the I6 journal that comes back from the board
//! (C-69: and the I7 one — Booter on SEC2, same load order).
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

// Offsets inside a falcon window (GSP 0x110000, SEC2 0x840000), from
// `dev_falcon_v4.h` / `dev_riscv_pri.h` / `dev_falcon_second_pri.h`
// (570.144, MIT) as generated in `nvidia-rm570`.
const ENGINE: u64 = 0x3c0;
const BCR_CTRL: u64 = 0x1668;
const FBIF_CTL: u64 = 0x624;
const DMACTL: u64 = 0x10c;
const TRANSCFG0: u64 = 0x600;
const DMATRFBASE: u64 = 0x110;
const DMATRFBASE1: u64 = 0x128;
const DMATRFMOFFS: u64 = 0x114;
const DMATRFFBOFFS: u64 = 0x11c;
const DMATRFCMD: u64 = 0x118;
const BROM_PARAADDR0: u64 = 0x1210;
const BROM_ENGIDMASK: u64 = 0x119c;
const BROM_CURR_UCODE_ID: u64 = 0x1198;
const MOD_SEL: u64 = 0x1180;
const BOOTVEC: u64 = 0x104;
const MAILBOX0: u64 = 0x40;
const MAILBOX1: u64 = 0x44;
/// `NV_PGSP_QUEUE_HEAD(0)` − GSP window base (`dev_gsp.h`): the RPC doorbell.
const QUEUE_HEAD: u64 = 0xc00;
const CPUCTL: u64 = 0x100;
const CPUCTL_ALIAS: u64 = 0x130;
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
    check_hs_load(events, &GSP_I6, (0, 0))
}

/// SEC2 falcon window (the only one I7 may write; C-69).
pub const SEC2_LO: u64 = 0x84_0000;
pub const SEC2_HI: u64 = 0x84_3fff;

/// One signed-ucode load as the journal shows it: which falcon, and the
/// words used in the error messages.
struct HsLoad {
    lo: u64,
    hi: u64,
    outside: &'static str,
    mailboxes: &'static str,
    after_start: &'static str,
    truncated: &'static str,
    config: &'static str,
}

const GSP_I6: HsLoad = HsLoad {
    lo: GSP_LO,
    hi: GSP_HI,
    outside: "scriere in afara ferestrei GSP",
    mailboxes: "cutiile postale trebuie 0 la FWSEC",
    after_start: "scrieri dupa pornirea FWSEC",
    truncated: "jurnalul se termina inainte de sfarsitul I6",
    config: "scriere in configuratie in mijlocul I6",
};

const SEC2_I7: HsLoad = HsLoad {
    lo: SEC2_LO,
    hi: SEC2_HI,
    outside: "scriere in afara ferestrei SEC2",
    mailboxes: "cutiile postale trebuie sa fie adresa GspFwWprMeta",
    after_start: "scrieri dupa pornirea Booter",
    truncated: "jurnalul se termina inainte de sfarsitul I7",
    config: "scriere in configuratie in mijlocul I7",
};

/// C-69: checks the BAR0 writes of an I7 journal (Booter on SEC2) — the
/// same load order as I6 ([`check_i6`]) on the SEC2 window, with mailbox
/// 0/1 = the low/high half of the `GspFwWprMeta` address instead of 0.
/// The events must hold only the I7 part (see [`check_i6_i7_journal`] for
/// a whole stick journal). Pure text processing.
pub fn check_i7(events: &[Event], wpr_meta_addr: u64) -> Result<I6Summary, I6LogError> {
    check_hs_load(
        events,
        &SEC2_I7,
        (wpr_meta_addr as u32, (wpr_meta_addr >> 32) as u32),
    )
}

/// `NV_PFALCON_FALCON_OS` (GSP-RM version after Booter, C-75).
const FALCON_OS: u64 = 0x80;

/// C-75: what a correct I7 journal in the board's order contained.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct I7LibosSummary {
    /// The Booter load on SEC2 ([`check_i7`] on that part).
    pub booter: I6Summary,
    /// The LibOS address the GSP mailboxes kept (read back equal).
    pub libos: u64,
    /// All BAR0 writes of the step.
    pub writes: usize,
}

/// C-75: checks an I7 journal in the order the board needs
/// ([`crate::gsp::falcon::booter_load_dma_libos`]): on the GSP window
/// ENGINE = 1, ENGINE = 0, BCR_CTRL = RISC-V (read back with CORE_SELECT
/// set), MAILBOX0/1 = the LibOS address, **each read back equal**; then the
/// Booter load on SEC2 exactly as [`check_i7`]; then one write of the GSP
/// `FALCON_OS` = 570144, read back equal. `libos_addr = None` takes the
/// address from the journal (it must be non-zero). The journal of 5 Nov
/// 2026 (os-246: mailboxes written without the GSP reset, read back 0) is
/// reported at its first write. Pure text processing.
pub fn check_i7_libos(
    events: &[Event],
    wpr_meta_addr: u64,
    libos_addr: Option<u64>,
) -> Result<I7LibosSummary, I6LogError> {
    let in_sec2 = |o: u64| (SEC2_LO..=SEC2_HI).contains(&o);
    let is_sec2_write =
        |e: &Event| matches!(*e, Event::MmioWrite { bar: 0, offset, .. } if in_sec2(offset));
    let writes: Vec<(u64, u32, u32)> = events
        .iter()
        .filter_map(|e| match *e {
            Event::MmioWrite {
                bar: 0,
                offset,
                written,
                after,
                ..
            } => Some((offset, written, after)),
            _ => None,
        })
        .collect();
    let Some(first) = events.iter().position(is_sec2_write) else {
        return Err(err(writes.len(), SEC2_LO, "lipseste Booter pe SEC2"));
    };
    let last = events.iter().rposition(is_sec2_write).unwrap_or(first);
    if let Some(k) = events.iter().position(|e| {
        matches!(*e, Event::ConfigWrite { .. })
            || matches!(*e, Event::MmioWrite { bar, .. } if bar != 0)
    }) {
        let n = events[..k]
            .iter()
            .filter(|x| matches!(x, Event::MmioWrite { bar: 0, .. }))
            .count();
        return Err(err(
            n,
            0,
            "scriere in configuratie sau in alt BAR in mijlocul I7",
        ));
    }
    let before = |e: &[Event]| {
        e.iter()
            .filter(|x| matches!(x, Event::MmioWrite { bar: 0, .. }))
            .count()
    };
    let n_pre = before(&events[..first]);
    let mut pre = &writes[..n_pre.min(writes.len())];
    let g = GSP_LO;
    // C-108 variant 1 (os-283 analysis): the init RPCs' doorbell
    // (`NV_PGSP_QUEUE_HEAD(0) = 0`) may come first, before the GSP reset, as
    // in RM / nouveau / tinygrad / nova. Optional: the os-283 order has none.
    let doorbell_first = matches!(pre.first(), Some(&(o, 0, _)) if o == g + QUEUE_HEAD);
    if doorbell_first {
        pre = &pre[1..];
    }
    let skip = usize::from(doorbell_first);
    // The os-246 form: mailboxes first, no GSP reset.
    if let Some(&(o, wr, af)) = pre.first()
        && o == g + MAILBOX0
    {
        return Err(err(
            0,
            o,
            if wr != af {
                "cutia GSP nu a pastrat adresa LibOS: GSP nu a fost resetat in RISC-V dupa FWSEC"
            } else {
                "lipseste resetarea GSP in RISC-V inainte de cutiile LibOS"
            },
        ));
    }
    let exp = [
        (ENGINE, "lipseste resetarea GSP (ENGINE=1)"),
        (ENGINE, "lipseste ENGINE=0 la GSP"),
        (BCR_CTRL, "lipseste trecerea GSP pe RISC-V (BCR_CTRL)"),
        (MAILBOX0, "lipseste cutia GSP 0 (adresa LibOS)"),
        (MAILBOX1, "lipseste cutia GSP 1 (adresa LibOS)"),
    ];
    for (k, &(off, what)) in exp.iter().enumerate() {
        match pre.get(k) {
            Some(&(o, _, _)) if o == g + off => {}
            Some(&(o, _, _)) => return Err(err(k + skip, o, what)),
            None => return Err(err(k + skip, g + off, what)),
        }
    }
    if pre.len() > exp.len() {
        return Err(err(
            exp.len() + skip,
            pre[exp.len()].0,
            "scriere in plus inainte de Booter",
        ));
    }
    if pre[0].1 != 1 || pre[1].1 != 0 {
        return Err(err(1 + skip, g + ENGINE, "resetarea GSP nu e 1 apoi 0"));
    }
    if pre[2].2 & (1 << 4) == 0 {
        return Err(err(
            2 + skip,
            g + BCR_CTRL,
            "GSP nu a trecut pe RISC-V (CORE_SELECT citit 0)",
        ));
    }
    for k in [3, 4] {
        if pre[k].1 != pre[k].2 {
            return Err(err(
                k + skip,
                pre[k].0,
                "cutia GSP nu a pastrat adresa LibOS",
            ));
        }
    }
    let libos = u64::from(pre[3].1) | u64::from(pre[4].1) << 32;
    if libos == 0 || libos_addr.is_some_and(|a| a != libos) {
        return Err(err(
            3 + skip,
            g + MAILBOX0,
            "cutiile GSP nu au adresa LibOS asteptata",
        ));
    }
    let booter = check_i7(&events[first..=last], wpr_meta_addr).map_err(|e| I6LogError {
        write_index: e.write_index + n_pre,
        ..e
    })?;
    let n_mid = before(&events[first..=last]);
    let post = &writes[(n_pre + n_mid).min(writes.len())..];
    let at = n_pre + n_mid;
    match post.first() {
        None => {
            return Err(err(
                at,
                g + FALCON_OS,
                "lipseste versiunea GSP-RM (FALCON_OS) dupa Booter",
            ));
        }
        Some(&(o, _, _)) if o != g + FALCON_OS => {
            return Err(err(
                at,
                o,
                "dupa Booter e permisa doar versiunea GSP-RM (FALCON_OS)",
            ));
        }
        Some(&(o, wr, af)) => {
            if wr != crate::gsp::falcon::GSP_RM_APP_VERSION_570_144 || wr != af {
                return Err(err(at, o, "FALCON_OS trebuie 570144, recitit egal"));
            }
        }
    }
    if let Some(&(o, _, _)) = post.get(1) {
        return Err(err(at + 1, o, "scriere in plus dupa versiunea GSP-RM"));
    }
    Ok(I7LibosSummary {
        booter,
        libos,
        writes: writes.len(),
    })
}

fn check_hs_load(
    events: &[Event],
    f: &HsLoad,
    mailboxes: (u32, u32),
) -> Result<I6Summary, I6LogError> {
    let mut w: Vec<(u64, u32)> = Vec::new();
    for e in events {
        match *e {
            Event::MmioWrite {
                bar: 0,
                offset,
                written,
                ..
            } => {
                if !(f.lo..=f.hi).contains(&offset) {
                    return Err(err(w.len(), offset, f.outside));
                }
                w.push((offset, written));
            }
            Event::MmioWrite { offset, .. } => {
                return Err(err(w.len(), offset, "scriere in alt BAR decat BAR0"));
            }
            Event::ConfigWrite { offset, .. } => {
                return Err(err(w.len(), u64::from(offset), f.config));
            }
            _ => {}
        }
    }
    let b = f.lo;
    let mut i = 0usize;
    let expect = |i: &mut usize, off: u64, what: &'static str| -> Result<u32, I6LogError> {
        let off = b + off;
        match w.get(*i) {
            Some(&(o, v)) if o == off => {
                *i += 1;
                Ok(v)
            }
            Some(&(o, _)) => Err(err(*i, o, what)),
            None => Err(err(*i, off, f.truncated)),
        }
    };
    let at = |i: usize| w.get(i).map(|&(o, _)| o - b);
    if expect(&mut i, ENGINE, "lipseste resetarea (ENGINE=1)")? != 1
        || expect(&mut i, ENGINE, "lipseste ENGINE=0")? != 0
    {
        return Err(err(i - 1, b + ENGINE, "resetarea nu e 1 apoi 0"));
    }
    expect(&mut i, BCR_CTRL, "lipseste alegerea nucleului falcon")?;
    expect(&mut i, FBIF_CTL, "lipseste FBIF_CTL")?;
    if expect(&mut i, DMACTL, "lipseste DMACTL")? != 0 {
        return Err(err(i - 1, b + DMACTL, "DMACTL trebuie 0"));
    }
    expect(&mut i, TRANSCFG0, "lipseste TRANSCFG(0)")?;
    let mut sections = Vec::new();
    while at(i) == Some(DMATRFBASE) {
        expect(&mut i, DMATRFBASE, "")?;
        expect(&mut i, DMATRFBASE1, "lipseste DMATRFBASE1")?;
        let mut blocks = 0u32;
        let mut imem = None;
        let mut first = (0u32, 0u32);
        while at(i) == Some(DMATRFMOFFS) {
            let m = expect(&mut i, DMATRFMOFFS, "")?;
            let fo = expect(&mut i, DMATRFFBOFFS, "lipseste DMATRFFBOFFS")?;
            let c = expect(&mut i, DMATRFCMD, "lipseste DMATRFCMD")?;
            if blocks == 0 {
                first = (m, fo);
            } else if m != first.0 + 0x100 * blocks || fo != first.1 + 0x100 * blocks {
                return Err(err(i - 3, b + DMATRFMOFFS, "bloc sarit sau in alta ordine"));
            }
            let is_imem = c & CMD_IMEM != 0;
            if *imem.get_or_insert(is_imem) != is_imem {
                return Err(err(
                    i - 1,
                    b + DMATRFCMD,
                    "IMEM si DMEM amestecate in aceeasi sectiune",
                ));
            }
            blocks += 1;
        }
        if blocks == 0 {
            return Err(err(i, b + DMATRFBASE, "sectiune DMA fara niciun bloc"));
        }
        sections.push((imem.unwrap_or(false), blocks));
    }
    if sections.is_empty() {
        return Err(err(i, b + DMATRFBASE, "nicio copiere DMA"));
    }
    expect(&mut i, BROM_PARAADDR0, "lipseste BROM_PARAADDR(0)")?;
    expect(&mut i, BROM_ENGIDMASK, "lipseste BROM_ENGIDMASK")?;
    expect(&mut i, BROM_CURR_UCODE_ID, "lipseste BROM_CURR_UCODE_ID")?;
    expect(&mut i, MOD_SEL, "lipseste MOD_SEL")?;
    expect(&mut i, BOOTVEC, "lipseste BOOTVEC")?;
    if expect(&mut i, MAILBOX0, "lipseste MAILBOX0")? != mailboxes.0
        || expect(&mut i, MAILBOX1, "lipseste MAILBOX1")? != mailboxes.1
    {
        return Err(err(i - 1, b + MAILBOX0, f.mailboxes));
    }
    let start = match w.get(i) {
        Some(&(o, v)) if o == b + CPUCTL || o == b + CPUCTL_ALIAS => v,
        Some(&(o, _)) => return Err(err(i, o, "lipseste pornirea (CPUCTL)")),
        None => {
            return Err(err(i, b + CPUCTL, "jurnalul se termina inainte de pornire"));
        }
    };
    if start & 0b10 == 0 {
        return Err(err(i, b + CPUCTL, "CPUCTL fara STARTCPU"));
    }
    i += 1;
    if let Some(&(o, _)) = w.get(i) {
        return Err(err(i, o, f.after_start));
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
    stick_events(&events, frts)
}

fn stick_events(events: &[Event], frts: (u64, u64)) -> StickJournal {
    let mut config_writes = Vec::new();
    let mut rest = Vec::new();
    for e in events {
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

/// `NV_PRISCV_RISCV_CPUCTL` of the GSP (`NV_FALCON2_GSP_BASE` 0x111000 +
/// 0x388); bit 7 = `ACTIVE_STAT` (570.144 `dev_riscv_pri.h`, MIT).
pub const GSP_RISCV_CPUCTL: u64 = 0x11_1388;

/// C-69: a whole stick journal of a window where I6 and then I7 ran in the
/// same boot (Bus Master on → I6 → WPR2 read-back → I7 → Bus Master off).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct I6I7Journal {
    /// The part before the first SEC2 write, read as [`check_stick_journal`].
    pub i6_part: StickJournal,
    /// [`check_i7`] on the BAR0 accesses from the first SEC2 write on
    /// (config writes taken out). `None` = no SEC2 write: I7 did not start.
    pub i7: Option<Result<I6Summary, I6LogError>>,
    /// Config-space writes in the I7 part (the Bus Master restore).
    pub i7_config_writes: Vec<(u16, u32, u32, u32)>,
    /// Last value read from the GSP RISC-V `CPUCTL`, if any.
    pub riscv_cpuctl: Option<u32>,
    /// `ACTIVE_STAT` set in that last value: GSP-RM runs (end of I7).
    pub riscv_active: bool,
}

/// C-69: reads the I6 + I7 stick journal in one pass. `wpr_meta_addr` is
/// the `GspFwWprMeta` address the OS printed for this run. Pure text
/// processing, never touches a board.
pub fn check_i6_i7_journal(text: &str, frts: (u64, u64), wpr_meta_addr: u64) -> I6I7Journal {
    let events: Vec<Event> = text.lines().filter_map(parse_line).collect();
    let split = events
        .iter()
        .position(|e| {
            matches!(*e, Event::MmioWrite { bar: 0, offset, .. }
                if (SEC2_LO..=SEC2_HI).contains(&offset))
        })
        .unwrap_or(events.len());
    let (a, b) = events.split_at(split);
    let mut i7_config_writes = Vec::new();
    let mut rest = Vec::new();
    for e in b {
        match *e {
            Event::ConfigWrite {
                offset,
                before,
                written,
                after,
            } => i7_config_writes.push((offset, before, written, after)),
            other => rest.push(other),
        }
    }
    let riscv_cpuctl = events.iter().rev().find_map(|e| match *e {
        Event::MmioRead {
            bar: 0,
            offset: GSP_RISCV_CPUCTL,
            value,
        } => Some(value),
        _ => None,
    });
    I6I7Journal {
        i6_part: stick_events(a, frts),
        i7: (!b.is_empty()).then(|| check_i7(&rest, wpr_meta_addr)),
        i7_config_writes,
        riscv_cpuctl,
        riscv_active: riscv_cpuctl.is_some_and(|v| v & (1 << 7) != 0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The I7 lines of the board journal of 5 Nov 2026 (os-246), shortened:
    /// the two GSP mailbox writes that read back 0, then the first SEC2
    /// writes. Real values, copied from `os246-gpu-jurnal-002.txt`.
    const OS246_I7_START: &str = "\
[  298.172009] gpu: bar0 w 0x110040: 0x00000000 <- 0x09c71000 => 0x00000000
[  298.195009] gpu: bar0 w 0x110044: 0x00000000 <- 0x00000000 => 0x00000000
[  298.218027] gpu: nota: falcon: RESET_READY nu a venit; continui
[  298.241119] gpu: bar0 w 0x8403c0: 0x00000000 <- 0x00000001 => 0x00000001
[  298.264045] gpu: bar0 w 0x8403c0: 0x00000001 <- 0x00000000 => 0x00000000
[  298.287289] gpu: bar0 w 0x841668: 0x00000110 <- 0x00000000 => 0x00000001
";

    /// C-75 (T88): the real os-246 I7 journal is reported at its first
    /// write with the cause (GSP not reset into RISC-V after FWSEC); the
    /// same lines with the mailbox kept say the reset is missing.
    #[test]
    fn i7_libos_rule_names_the_cause_on_the_board_journal() {
        let ev: Vec<Event> = OS246_I7_START.lines().filter_map(parse_line).collect();
        let e = check_i7_libos(&ev, 0x09c6_1000, None).unwrap_err();
        assert_eq!((e.write_index, e.offset), (0, 0x110040));
        assert!(e.what.contains("nu a fost resetat in RISC-V"), "{}", e.what);
        let kept = OS246_I7_START.replacen("=> 0x00000000", "=> 0x09c71000", 1);
        let ev: Vec<Event> = kept.lines().filter_map(parse_line).collect();
        let e = check_i7_libos(&ev, 0x09c6_1000, None).unwrap_err();
        assert!(e.what.contains("lipseste resetarea GSP"), "{}", e.what);
        // No SEC2 write at all.
        let e = check_i7_libos(&ev[..2], 0x09c6_1000, None).unwrap_err();
        assert_eq!(e.what, "lipseste Booter pe SEC2");
    }
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
