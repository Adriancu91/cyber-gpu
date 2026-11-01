//! C-54 (2): compare what the board said with what the simulation says.
//!
//! [`facts_from_report`] reads a Cyber Universe report (`raport-*.txt` from
//! the stick, as it lands in hub `rezultate/`) and extracts [`Fact`]s:
//! - `fapt <cheie> = <valoare>` lines (the format of [`crate::read_tour`]);
//! - the lines the OS prints today for `gpu`, `gpu gfw`, `gpu vbios`,
//!   `gpu fwsec` and `gpu firmware` (reports of 28 Oct – 1 Nov), so the old
//!   reports can be compared too;
//! - journal lines `jurnal cyber-gpu: bar0 r 0x… = 0x…`.
//!
//! [`compare`] puts them next to the facts of the same tour run on the
//! simulated board and says, for each key: same, different, only on the
//! board, only in the simulation. Some differences are expected and are
//! labelled as such (VBIOS content and NVIDIA firmware are never put in
//! the simulation). Pure text processing: never touches a board.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt;

use crate::read_tour::{Fact, bar0_key, desc_value, image_value, sig_value};

fn hex(s: &str) -> Option<u64> {
    let s = s.trim().trim_end_matches([',', ':', ')', ';']);
    u64::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16).ok()
}

fn after<'a>(line: &'a str, pat: &str) -> Option<&'a str> {
    line.find(pat).map(|i| &line[i + pat.len()..])
}

fn word(s: &str, n: usize) -> Option<&str> {
    s.split_whitespace().nth(n)
}

/// Extracts the facts of one line (zero or more).
fn line_facts(line: &str, out: &mut Vec<Fact>) {
    // New format first.
    if let Some(rest) = after(line, "fapt ") {
        if let Some((k, v)) = rest.split_once(" = ") {
            out.push(Fact::new(k.trim(), v.trim()));
        }
        return;
    }
    // Journal lines of our driver.
    if line.contains("jurnal cyber-gpu:") {
        if let Some(gpu_core::diag::Event::MmioRead {
            bar: 0,
            offset,
            value,
        }) = crate::i6_log::parse_line(line)
        {
            out.push(Fact::new(bar0_key(offset), format!("{value:#010x}")));
        }
        return;
    }
    // `gpu` (level 0).
    if let Some(r) = after(line, "NVIDIA (clasa ") {
        // "10de:2482 NVIDIA (clasa 03:00:00) subsistem 88131043"
        let pre = &line[..line.find(" NVIDIA (clasa ").unwrap_or(0)];
        if let Some(id) = pre.split_whitespace().last() {
            out.push(Fact::new("pci.id", id));
        }
        if let Some(c) = r.split(')').next() {
            out.push(Fact::new("pci.clasa", c.replace(':', "")));
        }
        if let Some(s) = after(line, "subsistem ").and_then(|s| word(s, 0))
            && s.len() == 8
        {
            out.push(Fact::new(
                "pci.subsistem",
                format!("{}:{}", &s[4..], &s[..4]),
            ));
        }
        return;
    }
    if let Some(r) = after(line, "Command ") {
        if let Some(c) = word(r, 0).and_then(hex) {
            out.push(Fact::new("pci.command", format!("{c:#06x}")));
            out.push(Fact::new(
                "pci.bus_master",
                if c & 4 != 0 { "da" } else { "nu" },
            ));
        }
        if let Some(d) = after(line, "stare ").and_then(|s| word(s, 0)) {
            out.push(Fact::new("pci.stare", d));
        }
        return;
    }
    if let Some(r) = after(line, "Resizable BAR: BAR") {
        // "Resizable BAR: BAR1 = 256 MiB acum"
        if let (Some(n), Some(sz)) = (word(r, 0), word(r, 2)) {
            out.push(Fact::new(format!("pci.rebar.bar{n}"), format!("{sz} MiB")));
        }
        return;
    }
    if let Some(r) = after(line, "gpu0: BAR") {
        // "BAR0 0xf9000000 memorie" / "BAR1 0xe0000000 memorie, 64 biti, prefetch" / "BAR5 0xf000 I/O"
        if let (Some(n), Some(base)) = (word(r, 0), word(r, 1).and_then(hex))
            && n.len() == 1
        {
            let v = if r.contains("I/O") {
                format!("{base:#x} io")
            } else {
                format!(
                    "{base:#x} mem{}{}",
                    if r.contains("64 biti") { " 64b" } else { "" },
                    if r.contains("prefetch") {
                        " prefetch"
                    } else {
                        ""
                    }
                )
            };
            out.push(Fact::new(format!("pci.bar{n}"), v));
        }
        return;
    }
    if let Some(r) = after(line, "NV_PMC_BOOT_0 = ") {
        // "NV_PMC_BOOT_0 = b74000a1, NV_PMC_BOOT_42 = 174a1000 (cyber-gpu, consecvent true)"
        if let Some(v) = word(r, 0).and_then(hex) {
            out.push(Fact::new(bar0_key(0), format!("{v:#010x}")));
        }
        if let Some(v) = after(line, "NV_PMC_BOOT_42 = ")
            .and_then(|s| word(s, 0))
            .and_then(hex)
        {
            out.push(Fact::new(bar0_key(0xa00), format!("{v:#010x}")));
        }
        if let Some(c) = after(line, "consecvent ").and_then(|s| word(s, 0)) {
            let c = c.trim_end_matches(')');
            out.push(Fact::new(
                "chip.consecvent",
                if c == "true" || c == "da" { "da" } else { "nu" },
            ));
        }
        return;
    }
    if let Some(r) = after(line, "gpu0: cip 0x") {
        if let Some(n) = word(r, 1) {
            out.push(Fact::new("chip.nume", n));
        }
        return;
    }
    if let Some(r) = after(line, "GFW gata (progres ") {
        if let Some(p) = word(r, 0).and_then(hex) {
            out.push(Fact::new("gfw.progres", format!("{p:#04x}")));
        }
        if let Some(n) = after(line, "dupa ").and_then(|s| word(s, 0)) {
            out.push(Fact::new("gfw.runde", n));
        }
        if let Some(m) = after(line, "PRIV_LEVEL_MASK ")
            .and_then(|s| word(s, 0))
            .and_then(hex)
        {
            out.push(Fact::new(bar0_key(0x118128), format!("{m:#010x}")));
        }
        return;
    }
    // `gpu vbios`
    if let Some(r) = after(line, "VBIOS: antet ") {
        // "antet fara IFR, prima imagine la 0x0, 4 imagini"
        out.push(Fact::new(
            "vbios.ifr",
            if r.starts_with("fara IFR") {
                "nu"
            } else {
                "da"
            },
        ));
        if let Some(n) = r
            .rsplit(", ")
            .next()
            .and_then(|s| s.strip_suffix(" imagini"))
        {
            out.push(Fact::new("vbios.imagini", n));
        }
        return;
    }
    if let Some(r) = after(line, "  imaginea ") {
        // "imaginea 0: offset 0x00000, tip 0x00 PCI-AT, 65024 octeti, 10de:2482, NPDE"
        let n = r.split(':').next().unwrap_or("");
        let off = after(r, "offset ").and_then(|s| word(s, 0)).and_then(hex);
        let ty = after(r, "tip ").and_then(|s| word(s, 0)).and_then(hex);
        let size = r
            .split(", ")
            .find(|p| p.ends_with(" octeti"))
            .and_then(|p| p.strip_suffix(" octeti"))
            .and_then(|p| p.parse::<u32>().ok());
        let id = r
            .split(", ")
            .find(|p| p.len() == 9 && p.as_bytes()[4] == b':');
        if let (Some(off), Some(ty), Some(size), Some(id)) = (off, ty, size, id) {
            let (v, d) = id.split_at(4);
            if let (Ok(v), Ok(d)) = (u16::from_str_radix(v, 16), u16::from_str_radix(&d[1..], 16)) {
                out.push(Fact::new(
                    format!("vbios.imagine{n}"),
                    image_value(off as u32, ty as u8, size, v, d),
                ));
            }
        }
        return;
    }
    if let Some(r) = after(line, "tabela BIT la ") {
        // "tabela BIT la 0x001b0: 17 jetoane: 0x32v1 0x42v2 ..."
        if let (Some(at), Some(n)) = (word(r, 0).and_then(hex), word(r, 1)) {
            out.push(Fact::new("vbios.bit", format!("{at:#x} {n}")));
        }
        if let Some(t) = r.splitn(3, ": ").nth(2) {
            let toks: Vec<String> = t
                .split_whitespace()
                .filter_map(|x| {
                    let (id, v) = x.split_once('v')?;
                    Some(format!("{:#04x}v{}", hex(id)?, v))
                })
                .collect();
            out.push(Fact::new("vbios.bit.jetoane", toks.join(" ")));
        }
        return;
    }
    // `gpu fwsec`
    if let Some(r) = after(line, "tabela PMU la ") {
        if let Some(at) = word(r, 0).and_then(hex) {
            out.push(Fact::new("fwsec.pmu", format!("{at:#x}")));
        }
        if let Some(a) = after(r, "aplicatii ") {
            let apps: Vec<&str> = a
                .split(" (")
                .next()
                .unwrap_or("")
                .split_whitespace()
                .collect();
            out.push(Fact::new("fwsec.aplicatii", apps.join(" ")));
        }
        return;
    }
    if let Some(r) = after(line, "  descriptor v") {
        // "descriptor v3 la 0x41208: antet 0x4ac octeti, ucode 0xea00 octeti (IMEM 0xe200, DMEM 0x800)"
        let v = word(r, 0).and_then(|s| s.parse::<u8>().ok());
        let at = word(r, 2).and_then(hex);
        let hdr = after(r, "antet ").and_then(|s| word(s, 0)).and_then(hex);
        let st = after(r, "ucode ").and_then(|s| word(s, 0)).and_then(hex);
        let im = after(r, "IMEM ").and_then(|s| word(s, 0)).and_then(hex);
        let dm = after(r, "DMEM ").and_then(|s| word(s, 0)).and_then(hex);
        if let (Some(v), Some(at), Some(h), Some(s), Some(i), Some(d)) = (v, at, hdr, st, im, dm) {
            out.push(Fact::new(
                "fwsec.descriptor",
                desc_value(v, at as u32, h as u32, s as u32, i as u32, d as u32),
            ));
        }
        return;
    }
    if let Some(r) = after(line, "  semnaturi: ") {
        // "semnaturi: 3 (masca versiunilor 0x0007, potrivita), motoare 0x0400, ucode_id 9"
        let n = word(r, 0).and_then(|s| s.parse::<u8>().ok());
        let m = after(r, "versiunilor ")
            .and_then(|s| word(s, 0))
            .and_then(hex);
        let e = after(r, "motoare ").and_then(|s| word(s, 0)).and_then(hex);
        let u = after(r, "ucode_id ")
            .and_then(|s| word(s, 0))
            .and_then(|s| s.parse::<u8>().ok());
        if let (Some(n), Some(m), Some(e), Some(u)) = (n, m, e, u) {
            out.push(Fact::new(
                "fwsec.semnaturi",
                sig_value(n, m as u16, e as u16, u),
            ));
        }
        return;
    }
    // `gpu firmware`: "booter_load-570.144.bin: ok: 61304 octeti, SHA-256 4497e3ef... = oficial"
    if let Some(i) = line.find("-570.144.bin: ") {
        let name_start = line[..i].rfind(' ').map(|s| s + 1).unwrap_or(0);
        let name = &line[name_start..i + "-570.144.bin".len()];
        let r = &line[i + "-570.144.bin: ".len()..];
        let size = after(r, "ok: ").and_then(|s| word(s, 0));
        let sha = after(r, "SHA-256 ").and_then(|s| word(s, 0));
        if let (Some(size), Some(sha)) = (size, sha) {
            out.push(Fact::new(
                format!("fw.{name}"),
                format!(
                    "{size} octeti sha256 {}{}",
                    sha.trim_end_matches('.'),
                    if r.contains("= oficial") {
                        " oficial"
                    } else {
                        ""
                    }
                ),
            ));
        }
    }
}

/// All facts of a report, in order. A key seen twice keeps both entries;
/// [`compare`] uses the last one (the most recent run in the report).
pub fn facts_from_report(text: &str) -> Vec<Fact> {
    let mut out = Vec::new();
    for line in text.lines() {
        line_facts(line, &mut out);
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Board and simulation agree.
    Same,
    /// Both have a value, and they differ: the model is wrong (or the board changed).
    Different,
    /// Expected difference: content the simulation must never hold.
    DifferentOnPurpose,
    /// The board showed it; the simulation has nothing for it.
    OnlyBoard,
    /// The simulation has it; this report did not show it.
    OnlySim,
}

impl Verdict {
    pub fn label(self) -> &'static str {
        match self {
            Verdict::Same => "LA FEL",
            Verdict::Different => "DIFERIT",
            Verdict::DifferentOnPurpose => "diferit intentionat",
            Verdict::OnlyBoard => "doar pe placa",
            Verdict::OnlySim => "doar in simulare",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Row {
    pub key: String,
    pub board: Option<String>,
    pub sim: Option<String>,
    pub verdict: Verdict,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Comparison {
    pub rows: Vec<Row>,
}

impl Comparison {
    pub fn count(&self, v: Verdict) -> usize {
        self.rows.iter().filter(|r| r.verdict == v).count()
    }
    /// The model needs fixing (or the board changed): a real difference.
    pub fn has_real_difference(&self) -> bool {
        self.count(Verdict::Different) > 0
    }
}

/// Keys whose value the simulation must never reproduce: VBIOS bytes read
/// through PROM, NVIDIA firmware files.
fn on_purpose(key: &str) -> bool {
    key.starts_with("fw.")
        || key.strip_prefix("bar0.").and_then(hex).is_some_and(|o| {
            (crate::regs::PROM_BASE..crate::regs::PROM_BASE + crate::regs::PROM_SIZE).contains(&o)
        })
}

fn last<'a>(facts: &'a [Fact], key: &str) -> Option<&'a str> {
    facts
        .iter()
        .rev()
        .find(|f| f.key == key)
        .map(|f| f.value.as_str())
}

/// Compare board facts with simulation facts, keys in first-seen order
/// (board keys first).
pub fn compare(board: &[Fact], sim: &[Fact]) -> Comparison {
    let mut keys: Vec<&str> = Vec::new();
    for f in board.iter().chain(sim) {
        if !keys.contains(&f.key.as_str()) {
            keys.push(&f.key);
        }
    }
    let rows = keys
        .into_iter()
        .map(|k| {
            let b = last(board, k);
            let s = last(sim, k);
            let verdict = match (b, s) {
                (Some(b), Some(s)) if b == s => Verdict::Same,
                (Some(_), _) if on_purpose(k) => Verdict::DifferentOnPurpose,
                (Some(_), Some(_)) => Verdict::Different,
                (Some(_), None) => Verdict::OnlyBoard,
                (None, _) => Verdict::OnlySim,
            };
            Row {
                key: String::from(k),
                board: b.map(String::from),
                sim: s.map(String::from),
                verdict,
            }
        })
        .collect();
    Comparison { rows }
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "comparatie placa / simulare (SIMULARE = modelul gpu-sim::observed): {} la fel, {} DIFERIT, {} diferit intentionat, {} doar pe placa, {} doar in simulare",
            self.count(Verdict::Same),
            self.count(Verdict::Different),
            self.count(Verdict::DifferentOnPurpose),
            self.count(Verdict::OnlyBoard),
            self.count(Verdict::OnlySim)
        )?;
        for r in &self.rows {
            let b = r.board.as_deref().unwrap_or("-");
            let s = r.sim.as_deref().unwrap_or("-");
            if r.verdict == Verdict::Same {
                writeln!(f, "  [{}] {} = {}", r.verdict.label(), r.key, b)?;
            } else {
                writeln!(
                    f,
                    "  [{}] {}: placa {} | simulare {}",
                    r.verdict.label(),
                    r.key,
                    b,
                    s
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::diag::Journal;
    use gpu_sim::clock::SimClock;
    use gpu_sim::observed;

    /// Lines copied from the `== jurnal gpu ==` sections of the board
    /// reports (hub `rezultate/`, 28 Oct, 29 Oct, 1 Nov). No VBIOS bytes
    /// except the first PROM dword the OS journaled.
    const BOARD: &str = "\
[1313.390] INFO  gpu: gpu0: b3:00.0 10de:2482 NVIDIA (clasa 03:00:00) subsistem 88131043
[1315.060] INFO  gpu: gpu0: Command 0003 (memorie da, bus master nu), stare D0
[1316.730] INFO  gpu: gpu0: BAR0 0xf9000000 memorie
[1318.400] INFO  gpu: gpu0: BAR1 0xe0000000 memorie, 64 biti, prefetch
[1320.070] INFO  gpu: gpu0: BAR3 0xf0000000 memorie, 64 biti, prefetch
[1321.740] INFO  gpu: gpu0: BAR5 0xf000 I/O
[1323.400] INFO  gpu: gpu0: Resizable BAR: BAR0 = 16 MiB acum
[1325.070] INFO  gpu: gpu0: Resizable BAR: BAR1 = 256 MiB acum
[1326.740] INFO  gpu: gpu0: Resizable BAR: BAR3 = 32 MiB acum
[1330.080] INFO  gpu: gpu0: NV_PMC_BOOT_0 = b74000a1, NV_PMC_BOOT_42 = 174a1000 (cyber-gpu, consecvent true)
[1331.750] INFO  gpu: gpu0: cip 0x174 GA104 (arhitectura Ampere)
[1434.160] INFO  gpu: gpu0: jurnal cyber-gpu: bar0 r 0x118128 = 0x00008b8f
[1435.240] INFO  gpu: gpu0: jurnal cyber-gpu: bar0 r 0x118234 = 0x000003ff
[1440.220] INFO  gpu: gpu0: GFW gata (progres 0xff) dupa 1 runde, 3 us; PRIV_LEVEL_MASK 0x00008b8f
[1023.550] INFO  gpu: gpu0: jurnal cyber-gpu: bar0 r 0x300000 = 0xeb7faa55
[1023.560] INFO  gpu: gpu0: VBIOS: antet fara IFR, prima imagine la 0x0, 4 imagini
[1023.560] INFO  gpu: gpu0:   imaginea 0: offset 0x00000, tip 0x00 PCI-AT, 65024 octeti, 10de:2482, NPDE
[1023.560] INFO  gpu: gpu0:   imaginea 1: offset 0x0fe00, tip 0x03 EFI, 83968 octeti, 10de:2482, NPDE
[1023.570] INFO  gpu: gpu0:   imaginea 2: offset 0x24600, tip 0xe0 FWSEC, 22016 octeti, 10de:2200, NPDE
[1023.570] INFO  gpu: gpu0:   imaginea 3: offset 0x29c00, tip 0xe0 FWSEC, 386048 octeti, 10de:2200, NPDE, ultima
[1107.170] INFO  gpu: gpu0: tabela BIT la 0x001b0: 17 jetoane: 0x32v1 0x42v2 0x43v2 0x44v1 0x49v1 0x4dv2 0x4ev0 0x50v2 0x53v2 0x54v1 0x55v1 0x56v1 0x78v1 0x64v1 0x70v2 0x75v1 0x69v2
[1206.790] INFO  gpu: gpu0:   tabela PMU la 0x87ecb: aplicatii 0x01 0x00 0x00 0x00 0x00 0x07 0x08 0x00 0x45 0x85 0x49 0x89 0x00 0x00 0x00 0x00 (0x85 = FWSEC_PROD: da)
[1206.800] INFO  gpu: gpu0:   descriptor v3 la 0x41208: antet 0x4ac octeti, ucode 0xea00 octeti (IMEM 0xe200, DMEM 0x800)
[1206.800] INFO  gpu: gpu0:   semnaturi: 3 (masca versiunilor 0x0007, potrivita), motoare 0x0400, ucode_id 9
[ 606.630] INFO  gpu:   booter_load-570.144.bin: ok: 61304 octeti, SHA-256 4497e3ef... = oficial
";

    fn sim_facts() -> Vec<Fact> {
        let mut cfg = observed::observed_config();
        let mut bar0 = observed::observed_bar0();
        let mut j = Journal::new(4096);
        crate::read_tour::run_tour(&mut cfg, None, &mut bar0, &mut SimClock::default(), &mut j)
            .unwrap()
            .facts
    }

    #[test]
    fn old_report_lines_become_facts() {
        let f = facts_from_report(BOARD);
        let get = |k: &str| last(&f, k);
        assert_eq!(get("pci.id"), Some("10de:2482"));
        assert_eq!(get("pci.subsistem"), Some("1043:8813"));
        assert_eq!(get("pci.clasa"), Some("030000"));
        assert_eq!(get("pci.bar1"), Some("0xe0000000 mem 64b prefetch"));
        assert_eq!(get("bar0.0x000000"), Some("0xb74000a1"));
        assert_eq!(get("bar0.0x118234"), Some("0x000003ff"));
        assert_eq!(
            get("vbios.bit.jetoane").map(|s| s.split(' ').count()),
            Some(17)
        );
        assert_eq!(
            get("fw.booter_load-570.144.bin"),
            Some("61304 octeti sha256 4497e3ef oficial")
        );
    }

    #[test]
    fn board_reports_agree_with_the_observed_model() {
        let c = compare(&facts_from_report(BOARD), &sim_facts());
        assert!(!c.has_real_difference(), "{c}");
        // Every hardware fact of these reports is reproduced by the model.
        assert_eq!(c.count(Verdict::OnlyBoard), 0, "{c}");
        // The VBIOS dword and the firmware file are different on purpose.
        assert_eq!(c.count(Verdict::DifferentOnPurpose), 2, "{c}");
        assert!(c.count(Verdict::Same) >= 28, "{c}");
    }

    #[test]
    fn a_wrong_model_is_caught() {
        // The pre-C-54 guess for the VBIOS split (rom::ASUS_LIKE_OFFSETS)
        // and BOOT_0 (SimMmio::ga104_bar0) differ from the board.
        let mut sim = sim_facts();
        for f in sim.iter_mut() {
            if f.key == "vbios.imagine1" {
                f.value = image_value(0x12000, 0x03, 0x93 * 512, 0x10de, 0x2482);
            }
            if f.key == "bar0.0x000000" {
                f.value = String::from("0x174000a1");
            }
        }
        let c = compare(&facts_from_report(BOARD), &sim);
        assert_eq!(c.count(Verdict::Different), 2, "{c}");
        let t = format!("{c}");
        assert!(t.contains("[DIFERIT] bar0.0x000000: placa 0xb74000a1 | simulare 0x174000a1"));
    }

    #[test]
    fn new_fact_lines_round_trip() {
        let sim = sim_facts();
        let text: String = sim
            .iter()
            .map(|f| format!("[  1.000] INFO  gpu: {f}\n"))
            .collect();
        let back = facts_from_report(&text);
        assert_eq!(back, sim);
        let c = compare(&back, &sim);
        assert_eq!(c.count(Verdict::Same), c.rows.len());
    }

    #[test]
    fn unrelated_lines_give_nothing() {
        let t = "[ 1.170] WARN  usb: port 25: incercarea 1, TEST UNIT READY\n\
                 [ 0.130] INFO  pci: 00:1f.6 8086:15e3 clasa 02:00:00 placa de retea\n\
                 == jurnal gpu ==\n";
        assert!(facts_from_report(t).is_empty());
    }
}
