//! Parser for the "golden log" of C-20 proposal A (preparation only).
//!
//! In the workstation window (with Adrian present, from a Linux stick) the
//! tinygrad driver runs with `NV_DEBUG=4`; every BAR0 register write it makes
//! goes through `NVDev.wreg`, which prints exactly
//! `wreg: {hex(addr)} = {hex(value)}` (tinygrad `support/nv/nvdev.py`, MIT,
//! read at commit 9cf7829a6). Other lines (DEBUG RPC prints, Python messages)
//! are kept only as a count. This module turns the log into an ordered list
//! of writes that a test can compare with our own driver, step by step, and
//! replay into [`SimMmio`].
//!
//! Not in the log (tinygrad limitation, so not in the golden list either):
//! register *reads*, the doorbell page (`ops_nv.py` maps BAR0+0xbb0000
//! separately), VRAM writes through BAR1 and system-memory buffers.

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::mmio::SimMmio;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Write {
    /// 1-based line number in the log, for error reports.
    pub line: usize,
    pub addr: u32,
    pub value: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GoldenLog {
    pub writes: Vec<Write>,
    /// Lines that are not register writes (ignored).
    pub other_lines: usize,
}

const PREFIX: &str = "wreg: ";

fn hex(s: &str) -> Option<u64> {
    let d = s.strip_prefix("0x")?;
    if d.is_empty() || d.len() > 16 {
        return None;
    }
    u64::from_str_radix(d, 16).ok()
}

/// Parses the whole log. A line that starts with `wreg: ` but is malformed,
/// an unaligned address or a value wider than 32 bits is an error with the
/// line number: a damaged golden log must not be half-trusted.
pub fn parse(text: &str) -> GpuResult<GoldenLog> {
    let mut log = GoldenLog::default();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim_end_matches('\r');
        let Some(rest) = line.strip_prefix(PREFIX) else {
            log.other_lines += 1;
            continue;
        };
        let bad = GpuError::InvalidArgument("jurnal: linie wreg invalida");
        let (a, v) = rest.split_once(" = ").ok_or(bad)?;
        let (a, v) = (hex(a.trim()).ok_or(bad)?, hex(v.trim()).ok_or(bad)?);
        if a > u32::MAX as u64 || v > u32::MAX as u64 || a % 4 != 0 {
            return Err(bad);
        }
        log.writes.push(Write {
            line: i + 1,
            addr: a as u32,
            value: v as u32,
        });
    }
    Ok(log)
}

impl GoldenLog {
    /// Replays the writes, in order, into a simulated BAR0 (SIMULARE). Writes
    /// outside the window are refused, not dropped.
    pub fn replay_into(&self, bar0: &mut SimMmio) -> GpuResult<usize> {
        use gpu_core::hal::Mmio;
        for w in &self.writes {
            if w.addr as u64 + 4 > bar0.len() {
                return Err(GpuError::InvalidArgument("jurnal: adresa in afara BAR0"));
            }
        }
        for w in &self.writes {
            bar0.write32_raw(w.addr as u64, w.value);
        }
        Ok(self.writes.len())
    }

    /// First position where `ours` differs from the golden list
    /// (`None` = identical). Used to compare our driver with the log.
    pub fn first_difference(&self, ours: &[(u32, u32)]) -> Option<usize> {
        let golden = self.writes.iter().map(|w| (w.addr, w.value));
        let mut n = 0;
        for (g, o) in golden.zip(ours.iter().copied()) {
            if g != o {
                return Some(n);
            }
            n += 1;
        }
        if n == self.writes.len() && n == ours.len() {
            None
        } else {
            Some(n)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::hal::Mmio;

    /// Written by hand in the format of `nvdev.py` (not a real log): the
    /// GSP mailbox, a queue head and the falcon CPUCTL alias, mixed with
    /// the kind of other output tinygrad prints.
    const HAND_WRITTEN: &str = "\
nv 0000:b3:00.0: WPR2 is up. Issuing a full reset.
wreg: 0x110040 = 0x0
wreg: 0x110044 = 0x0
some rpc print line
wreg: 0x110c00 = 0x0\r
wreg: 0x110130 = 0x2
";

    #[test]
    fn parses_the_tinygrad_format() {
        let log = parse(HAND_WRITTEN).unwrap();
        assert_eq!(log.other_lines, 2);
        assert_eq!(log.writes.len(), 4);
        assert_eq!(
            log.writes[0],
            Write {
                line: 2,
                addr: 0x110040,
                value: 0
            }
        );
        assert_eq!(
            log.writes[3],
            Write {
                line: 6,
                addr: 0x110130,
                value: 2
            }
        );
    }

    #[test]
    fn replays_into_the_simulated_bar0_and_compares() {
        let log = parse(HAND_WRITTEN).unwrap();
        let mut bar0 = SimMmio::ga104_bar0();
        bar0.set(0x110130, 0xdead);
        assert_eq!(log.replay_into(&mut bar0).unwrap(), 4);
        assert_eq!(bar0.read32_raw(0x110130), 2);

        let same = [(0x110040, 0), (0x110044, 0), (0x110c00, 0), (0x110130, 2)];
        assert_eq!(log.first_difference(&same), None);
        let mut other = same;
        other[2].1 = 1;
        assert_eq!(log.first_difference(&other), Some(2));
        assert_eq!(log.first_difference(&same[..3]), Some(3));
    }

    #[test]
    fn refuses_damaged_lines() {
        for bad in [
            "wreg: 0x110040 0x0",
            "wreg: 110040 = 0x0",
            "wreg: 0x110041 = 0x0",
            "wreg: 0x110040 = 0x100000000",
            "wreg: 0x = 0x1",
        ] {
            assert!(parse(bad).is_err(), "{bad}");
        }
        let far = parse("wreg: 0x2000000 = 0x1").unwrap();
        assert!(far.replay_into(&mut SimMmio::ga104_bar0()).is_err());
    }
}
