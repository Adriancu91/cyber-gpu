//! C-157 point 5 (= C-151 point 2) — the dashboard used as a **thermal watch** during long jobs, and read before
//! and after the max-speed request. Pure logic, built and checked **only in SIMULARE**: nothing here sends a
//! control or touches a register; it reads the RUSD page the OS already has (see [`super::rusd`]) and decides.
//!
//! - [`ThermalLimits::PROPOSED`]: warn at 80 °C, stop at 85 °C, start again only at or below 70 °C, a dashboard
//!   whose temperature timestamp has not moved for 10 s = stop (we can no longer see the board). Plus an immediate
//!   stop on a thermal or emergency slow-down reason in the page (0x08, 0x20, 0x40, 0x80). DEDUS proposal: NVIDIA
//!   lists 93 °C as the RTX 3070 Ti's maximum GPU temperature; Ampere boards report a "max operating" of ~87 °C.
//!   Sources and reasoning: `docs/C-157_P5_CEAS_TERMIC.md`.
//! - [`Watch`]: `due` → read the page → `observe`; the stop latches; few `kw` rows (one per verdict change, one
//!   summary), never one per reading.
//! - [`compare_boost`]: the P-state and the core clock before and after the max-speed request.

use super::perf::Pstate;
use super::rusd::{Dashboard, RusdMem, Section, off, read_section, reason};
use gpu_core::diag::{Event, Journal};

/// The warn / stop / start-again temperatures (whole °C) and how long a silent dashboard is tolerated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ThermalLimits {
    pub warn_c: i32,
    pub stop_c: i32,
    /// A new job may start only at or below this temperature.
    pub resume_c: i32,
    /// The temperature timestamp unchanged for longer than this = stop.
    pub silent_ms: u64,
}

impl ThermalLimits {
    /// The proposal (DEDUS, `docs/C-157_P5_CEAS_TERMIC.md` §3).
    pub const PROPOSED: ThermalLimits = ThermalLimits {
        warn_c: 80,
        stop_c: 85,
        resume_c: 70,
        silent_ms: 10_000,
    };
    /// No stop limit above this is accepted (NVIDIA's 93 °C maximum minus a margin).
    pub const STOP_C_MAX: i32 = 90;

    /// Refuses limits without sense before anything is watched.
    pub fn check(&self) -> Result<(), &'static str> {
        if self.stop_c > Self::STOP_C_MAX {
            return Err("ceas termic: oprirea peste 90 C");
        }
        if self.warn_c >= self.stop_c {
            return Err("ceas termic: avertizarea nu e sub oprire");
        }
        if self.resume_c >= self.warn_c {
            return Err("ceas termic: pornirea din nou nu e sub avertizare");
        }
        if self.silent_ms < 1_000 || self.silent_ms > 60_000 {
            return Err("ceas termic: tabloul mut in afara 1-60 s");
        }
        Ok(())
    }
}

impl Default for ThermalLimits {
    fn default() -> Self {
        Self::PROPOSED
    }
}

/// The slow-down reasons that stop the work at once (thermal, hardware slow-down, power brakes).
pub const STOP_REASONS: u32 =
    reason::HW_SLOWDOWN | reason::THERMAL | reason::HW_POWER_BRAKES_SLOWDOWN;

/// Why the watch stopped the work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// The temperature reached `stop_c` (the value read, whole °C).
    Hot(i32),
    /// A slow-down reason from [`STOP_REASONS`] (the bits seen).
    Slowdown(u32),
    /// The temperature timestamp did not move for this many ms.
    Silent(u64),
    /// The temperature section could not be read at all (missing or busy).
    NoTemperature,
}

impl StopReason {
    /// The code in the `kw` rows: 1 hot, 2 slow-down, 3 silent, 4 no temperature.
    pub fn code(self) -> u32 {
        match self {
            StopReason::Hot(_) => 1,
            StopReason::Slowdown(_) => 2,
            StopReason::Silent(_) => 3,
            StopReason::NoTemperature => 4,
        }
    }
}

/// One reading's verdict.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    Continue,
    Warn,
    Stop(StopReason),
}

impl Verdict {
    /// The code in the `kw` rows: 0 continue, 1 warn, 2 stop.
    pub fn code(self) -> u32 {
        match self {
            Verdict::Continue => 0,
            Verdict::Warn => 1,
            Verdict::Stop(_) => 2,
        }
    }
}

/// The names of the `kw` rows.
pub mod rows {
    pub const VERDICT: &str = "kw ceas termic verdict (0 continua, 1 avertizare, 2 oprire)";
    pub const AT_MS: &str = "kw ceas termic la (ms de la pornirea lucrarii)";
    pub const TEMP: &str = "kw ceas termic temperatura (grade C)";
    pub const STOP: &str = "kw ceas termic motivul opririi (1 fierbinte, 2 incetinire, 3 tablou mut, 4 fara temperatura)";
    pub const STOP_DETAIL: &str =
        "kw ceas termic detaliu oprire (grade C / masca motivelor / ms fara schimbare)";
    pub const READS: &str = "kw ceas termic rezumat: citiri";
    pub const TEMP_MIN: &str = "kw ceas termic rezumat: temperatura minima (grade C)";
    pub const TEMP_MAX: &str = "kw ceas termic rezumat: temperatura maxima (grade C)";
    pub const POWER_MAX: &str = "kw ceas termic rezumat: puterea maxima (mW)";
    pub const SILENT_READS: &str = "kw ceas termic rezumat: citiri cu tabloul neschimbat";
    pub const BOOST: &str =
        "kw viteza maxima vazuta in tablou (0 urcata, 1 deja sus, 2 fara efect, 3 necunoscut)";
    pub const BOOST_P_BEFORE: &str = "kw viteza maxima: starea P inainte (tablou)";
    pub const BOOST_P_AFTER: &str = "kw viteza maxima: starea P dupa (tablou)";
    pub const BOOST_CLK_BEFORE: &str = "kw viteza maxima: ceasul nucleului inainte (MHz, tablou)";
    pub const BOOST_CLK_AFTER: &str = "kw viteza maxima: ceasul nucleului dupa (MHz, tablou)";
}

/// The temperature section's timestamp (the "is the page still being written?" mark), if valid.
pub fn temp_stamp(mem: &mut dyn RusdMem) -> Option<u64> {
    match read_section::<{ off::TEMP_LEN }>(mem, off::TEMP_GPU) {
        Section::Ok { ts, .. } => Some(ts),
        _ => None,
    }
}

/// The thermal watch over one long job. Times are ms from the job's start (the OS's clock).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Watch {
    pub limits: ThermalLimits,
    /// How often to read (ms).
    pub every_ms: u64,
    next_ms: u64,
    last_stamp: Option<u64>,
    stamp_since_ms: u64,
    last_verdict: Option<Verdict>,
    stopped: Option<StopReason>,
    pub reads: u32,
    pub silent_reads: u32,
    pub temp_min: Option<i32>,
    pub temp_max: Option<i32>,
    pub power_max_mw: Option<u32>,
}

impl Watch {
    /// The default reading interval (ms): ten dashboard polls of 500 ms.
    pub const EVERY_MS_DEFAULT: u64 = 5_000;

    /// A watch; refuses limits without sense and intervals outside 1–60 s, or longer than the silent limit.
    pub fn new(limits: ThermalLimits, every_ms: u64) -> Result<Watch, &'static str> {
        limits.check()?;
        if !(1_000..=60_000).contains(&every_ms) || every_ms > limits.silent_ms {
            return Err("ceas termic: intervalul in afara 1-60 s sau peste limita tabloului mut");
        }
        Ok(Watch {
            limits,
            every_ms,
            next_ms: 0,
            last_stamp: None,
            stamp_since_ms: 0,
            last_verdict: None,
            stopped: None,
            reads: 0,
            silent_reads: 0,
            temp_min: None,
            temp_max: None,
            power_max_mw: None,
        })
    }

    /// True when a reading is due at `now_ms` (the first one at once).
    pub fn due(&self, now_ms: u64) -> bool {
        now_ms >= self.next_ms
    }

    /// The latched stop, if any.
    pub fn stopped(&self) -> Option<StopReason> {
        self.stopped
    }

    /// One reading: the dashboard and the temperature timestamp read at `now_ms`. Writes the `kw` rows only when
    /// the verdict changes. Once stopped, stays stopped (whatever the next readings say).
    pub fn observe(
        &mut self,
        now_ms: u64,
        dash: &Dashboard,
        stamp: Option<u64>,
        j: &mut Journal,
    ) -> Verdict {
        self.reads = self.reads.saturating_add(1);
        self.next_ms = now_ms.saturating_add(self.every_ms);
        let temp = dash.temp_c();
        if let Some(t) = temp {
            self.temp_min = Some(self.temp_min.map_or(t, |m| m.min(t)));
            self.temp_max = Some(self.temp_max.map_or(t, |m| m.max(t)));
        }
        if let Some(p) = dash.avg_power_mw {
            self.power_max_mw = Some(self.power_max_mw.map_or(p, |m| m.max(p)));
        }
        // The page's heartbeat: a new timestamp restarts the silent clock.
        let silent_for = match (stamp, self.last_stamp) {
            (Some(s), Some(l)) if s == l => {
                self.silent_reads = self.silent_reads.saturating_add(1);
                now_ms.saturating_sub(self.stamp_since_ms)
            }
            (Some(s), _) => {
                self.last_stamp = Some(s);
                self.stamp_since_ms = now_ms;
                0
            }
            (None, _) => {
                self.silent_reads = self.silent_reads.saturating_add(1);
                if self.last_stamp.is_none() && self.reads == 1 {
                    self.stamp_since_ms = now_ms;
                }
                now_ms.saturating_sub(self.stamp_since_ms)
            }
        };
        let fresh = if let Some(r) = self.stopped {
            Verdict::Stop(r)
        } else if let Some(bits) = dash.reasons.map(|r| r & STOP_REASONS).filter(|&b| b != 0) {
            Verdict::Stop(StopReason::Slowdown(bits))
        } else if let Some(t) = temp.filter(|&t| t >= self.limits.stop_c) {
            Verdict::Stop(StopReason::Hot(t))
        } else if temp.is_none() && silent_for > self.limits.silent_ms {
            Verdict::Stop(StopReason::NoTemperature)
        } else if silent_for > self.limits.silent_ms {
            Verdict::Stop(StopReason::Silent(silent_for))
        } else if temp.is_some_and(|t| t >= self.limits.warn_c) {
            Verdict::Warn
        } else {
            Verdict::Continue
        };
        if let Verdict::Stop(r) = fresh {
            self.stopped = Some(r);
        }
        if self.last_verdict != Some(fresh) {
            self.last_verdict = Some(fresh);
            let mut row = |what: &'static str, value: u32| j.push(Event::Value { what, value });
            row(rows::VERDICT, fresh.code());
            row(rows::AT_MS, now_ms.min(u32::MAX as u64) as u32);
            if let Some(t) = temp {
                row(rows::TEMP, t as u32);
            }
            if let Verdict::Stop(r) = fresh {
                row(rows::STOP, r.code());
                let detail = match r {
                    StopReason::Hot(t) => t as u32,
                    StopReason::Slowdown(b) => b,
                    StopReason::Silent(ms) => ms.min(u32::MAX as u64) as u32,
                    StopReason::NoTemperature => 0,
                };
                row(rows::STOP_DETAIL, detail);
            }
        }
        fresh
    }

    /// The summary rows at the end of the job.
    pub fn summary(&self, j: &mut Journal) {
        let mut row = |what: &'static str, value: u32| j.push(Event::Value { what, value });
        row(rows::READS, self.reads);
        if let Some(t) = self.temp_min {
            row(rows::TEMP_MIN, t as u32);
        }
        if let Some(t) = self.temp_max {
            row(rows::TEMP_MAX, t as u32);
        }
        if let Some(p) = self.power_max_mw {
            row(rows::POWER_MAX, p);
        }
        row(rows::SILENT_READS, self.silent_reads);
    }
}

/// Whether a new job may start now: the temperature is read and at or below `resume_c`, and no stop reason is set.
pub fn may_start(limits: &ThermalLimits, dash: &Dashboard) -> bool {
    dash.temp_c().is_some_and(|t| t <= limits.resume_c)
        && dash.reasons.is_some_and(|r| r & STOP_REASONS == 0)
}

/// What the dashboard says the max-speed request did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoostSeen {
    /// Reached P0, or the core clock rose by at least 20 %.
    Raised,
    /// Already in P0 before the request (os-323).
    AlreadyUp,
    /// Neither.
    NoEffect,
    /// The P-state or the core clock is missing on one side.
    Unknown,
}

impl BoostSeen {
    pub fn code(self) -> u32 {
        match self {
            BoostSeen::Raised => 0,
            BoostSeen::AlreadyUp => 1,
            BoostSeen::NoEffect => 2,
            BoostSeen::Unknown => 3,
        }
    }
}

/// The wait between the request and the "after" reading: two dashboard polls, so the page shows the new state.
pub fn after_wait_ms(poll_ms: u32) -> u64 {
    2 * poll_ms as u64
}

/// The dashboard before and after the max-speed request; writes the `kw viteza maxima` rows.
pub fn compare_boost(before: &Dashboard, after: &Dashboard, j: &mut Journal) -> BoostSeen {
    let seen = match (
        before.pstate(),
        after.pstate(),
        before.clk_graphics_mhz,
        after.clk_graphics_mhz,
    ) {
        (Some(Pstate::P(0)), Some(_), Some(_), Some(_)) => BoostSeen::AlreadyUp,
        (Some(_), Some(pa), Some(cb), Some(ca)) => {
            if pa == Pstate::P(0) || ca as u64 * 5 >= cb as u64 * 6 {
                BoostSeen::Raised
            } else {
                BoostSeen::NoEffect
            }
        }
        _ => BoostSeen::Unknown,
    };
    let mut row = |what: &'static str, v: Option<u32>| {
        if let Some(value) = v {
            j.push(Event::Value { what, value });
        }
    };
    row(rows::BOOST, Some(seen.code()));
    row(rows::BOOST_P_BEFORE, before.pstate().map(Pstate::value));
    row(rows::BOOST_P_AFTER, after.pstate().map(Pstate::value));
    row(rows::BOOST_CLK_BEFORE, before.clk_graphics_mhz);
    row(rows::BOOST_CLK_AFTER, after.clk_graphics_mhz);
    seen
}

/// What a stick journal's `kw` rows say.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParsedKw {
    /// How many verdict rows (= verdict changes).
    pub changes: u32,
    pub last_verdict: Option<u32>,
    pub stop: Option<u32>,
    pub stop_detail: Option<u32>,
    pub reads: Option<u32>,
    pub temp_min: Option<i32>,
    pub temp_max: Option<i32>,
    pub power_max_mw: Option<u32>,
    pub silent_reads: Option<u32>,
    pub boost: Option<u32>,
    pub boost_clk_before: Option<u32>,
    pub boost_clk_after: Option<u32>,
}

fn kw_value(line: &str) -> Option<(&str, u32)> {
    let rest = line.split("valoare ").nth(1)?;
    let (name, v) = rest.rsplit_once(" = ")?;
    let v = u32::from_str_radix(v.trim().strip_prefix("0x")?, 16).ok()?;
    Some((name, v))
}

/// Reads the `kw` rows of a stick journal (the last value of each; `changes` counts the verdict rows).
pub fn parse_kw(text: &str) -> ParsedKw {
    let mut k = ParsedKw::default();
    for line in text.lines() {
        let Some((name, v)) = kw_value(line) else {
            continue;
        };
        match name {
            rows::VERDICT => {
                k.changes += 1;
                k.last_verdict = Some(v);
            }
            rows::STOP => k.stop = Some(v),
            rows::STOP_DETAIL => k.stop_detail = Some(v),
            rows::READS => k.reads = Some(v),
            rows::TEMP_MIN => k.temp_min = Some(v as i32),
            rows::TEMP_MAX => k.temp_max = Some(v as i32),
            rows::POWER_MAX => k.power_max_mw = Some(v),
            rows::SILENT_READS => k.silent_reads = Some(v),
            rows::BOOST => k.boost = Some(v),
            rows::BOOST_CLK_BEFORE => k.boost_clk_before = Some(v),
            rows::BOOST_CLK_AFTER => k.boost_clk_after = Some(v),
            _ => {}
        }
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proposed_limits_are_sane() {
        assert_eq!(ThermalLimits::PROPOSED.check(), Ok(()));
        let mut l = ThermalLimits::PROPOSED;
        l.stop_c = 91;
        assert!(l.check().is_err());
        l = ThermalLimits::PROPOSED;
        l.warn_c = 85;
        assert!(l.check().is_err());
        l = ThermalLimits::PROPOSED;
        l.resume_c = 80;
        assert!(l.check().is_err());
        assert!(Watch::new(ThermalLimits::PROPOSED, 500).is_err());
        assert!(Watch::new(ThermalLimits::PROPOSED, 20_000).is_err());
    }

    #[test]
    fn stop_reasons_are_the_header_bits() {
        assert_eq!(STOP_REASONS, 0x08 | 0x20 | 0x40 | 0x80);
        assert_eq!(STOP_REASONS & reason::GPU_IDLE, 0);
    }
}
