//! C-151 point 1 — the board's dashboard: "RM User Shared Data" (RUSD), a page of **system memory** GSP-RM
//! fills by itself with the temperature, the clocks, the P-state, the load, the slow-down reasons and the power.
//! Built and read **only in SIMULARE**; OFF by default ([`RusdPlan::enabled`] = false). Nothing here touches the
//! board: it builds the two control payloads and reads bytes the OS hands it.
//!
//! **The "three sources" rule (sesiunea 124, `docs/C-151_TABLOU_RUSD.md`):**
//! - the messages: NVIDIA 570.144 `ctrl2080internal.h` (`INTERNAL_INIT_USER_SHARED_DATA` 0x20800afe `{NvU64
//!   physAddr}`, 0 = de-initialise; `INTERNAL_USER_SHARED_DATA_SET_DATA_POLL` 0x20800aff `{NvU64 polledDataMask;
//!   NvU32 pollFrequencyMs}`), NVIDIA 570.144 `gpu_user_shared_data.c` (who sends them, in which order and with
//!   which values), `g_subdevice_nvoc.c` (flags `0xc0` = `INTERNAL | ROUTE_TO_PHYSICAL`, kernel-privileged — the
//!   same class as `INTERNAL_STATIC_KGR_GET_INFO`, accepted from our client on the RTX 3070 Ti at I10c), and
//!   nouveau's series "GSP telemetry via RUSD" (Mohamed Ahmed, 14 Jul 2026, r570 only): the same two controls,
//!   INIT first, then the interval once with an empty mask, then the mask with interval 0 ("keep the current").
//! - the memory: `gpuCreateRusdMemory` — `sizeof(NV00DE_SHARED_DATA)`, **system memory, physically contiguous**,
//!   zero-filled; the address given is the one the GPU sees (`AT_GPU`); nouveau: "a zero-filled, physically
//!   contiguous, coherent sysmem buffer, hand its bus address to the firmware". GSP-RM writes it, the CPU only reads.
//! - the reading: `cl00de.h` (`RUSD_READ_DATA`: timestamp, copy, timestamp again; equal = valid; up to 10 tries;
//!   `0` = not available, `u64::MAX` = write in progress) and nouveau's reader (the same 10 tries).
//!
//! **One difference, written down (DEDUS):** NVIDIA's driver and nouveau send both controls with GSP-RM's
//! *internal* client (`hInternalClient`/`hInternalSubdevice` from `GSP_STATIC_CONFIG_INFO`); we do not read that
//! RPC and send them with **our** client and subdevice, as for the boost (C-147 point 4). The flags allow it (the
//! same as `KGR_GET_INFO`, which GSP-RM took from our client); whether GSP-RM ties RUSD to its internal client is
//! not known. A refusal is a journal row.
//!
//! The layout (x86-64, computed by the C compiler from `cl00de.h` 570.144; 464 bytes): see [`off`].

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use super::perf::{Pstate, RmControl};
use super::rmapi;

/// `NV2080_CTRL_CMD_INTERNAL_INIT_USER_SHARED_DATA` (`ctrl2080internal.h`, 570.144).
pub const CMD_INIT: u32 = 0x2080_0afe;
/// `NV2080_CTRL_CMD_INTERNAL_USER_SHARED_DATA_SET_DATA_POLL` (`ctrl2080internal.h`, 570.144).
pub const CMD_SET_DATA_POLL: u32 = 0x2080_0aff;
/// `sizeof(NV00DE_SHARED_DATA)` in 570.144 (x86-64). The firmware writes this many bytes; the OS gives one page.
pub const SHARED_DATA_SIZE: usize = 464;

/// `NV00DE_RUSD_POLL_*` (`cl00de.h`).
pub mod poll {
    pub const CLOCK: u64 = 0x1;
    pub const PERF: u64 = 0x2;
    pub const MEMORY: u64 = 0x4;
    pub const POWER: u64 = 0x8;
    pub const THERMAL: u64 = 0x10;
    pub const PCI: u64 = 0x20;
    /// What the dashboard asks for: clocks, perf (P-state, load, slow-down reasons), power, temperature.
    pub const DASHBOARD: u64 = CLOCK | PERF | POWER | THERMAL;
    pub const ALL: u64 = 0x3f;
}

/// The interval NVIDIA's driver uses on GeForce (`NV_REG_STR_RM_RUSD_POLLING_INTERVAL_DEFAULT`, ms).
pub const POLL_MS_DEFAULT: u32 = 500;
/// The bounds NVIDIA's registry allows (`…_MIN`, `…_MAX`, ms); we never send outside them.
pub const POLL_MS_MIN: u32 = 100;
pub const POLL_MS_MAX: u32 = 1000;

/// `RUSD_TIMESTAMP_INVALID`, `RUSD_TIMESTAMP_WRITE_IN_PROGRESS`, `RUSD_SEQ_START` (`cl00de.h`).
pub const TS_INVALID: u64 = 0;
pub const TS_WRITE_IN_PROGRESS: u64 = u64::MAX;
pub const SEQ_START: u64 = 0xff00_0000_0000_0000;
/// `RUSD_READ_DATA`: tries before giving up.
pub const READ_TRIES: u32 = 10;

/// Byte offsets inside `NV00DE_SHARED_DATA` (570.144, x86-64; printed by `offsetof` — `docs/C-151_TABLOU_RUSD.md`).
/// Each polled section starts with its `volatile NvU64 lastModifiedTimestamp`.
pub mod off {
    /// `clkPublicDomainInfos`: ts, then `targetClkMHz` for GRAPHICS, MEMORY, VIDEO, SM.
    pub const CLOCKS: usize = 72;
    pub const CLOCKS_LEN: usize = 16;
    /// `clkThrottleReason`: ts, `reasonMask`.
    pub const THROTTLE: usize = 96;
    pub const THROTTLE_LEN: usize = 4;
    /// `perfDevUtil`: ts, `gpuPercentBusy`, `memoryPercentBusy`, 4 × (`clkPercentBusy`, `samplingPeriodUs`).
    pub const UTIL: usize = 112;
    pub const UTIL_LEN: usize = 40;
    /// `perfCurrentPstate`: ts, `currentPstate` (the mask of `PERF_GET_CURRENT_PSTATE`).
    pub const PSTATE: usize = 264;
    pub const PSTATE_LEN: usize = 4;
    /// `powerLimitGpu`: ts, `requestedmW`, `enforcedmW`.
    pub const POWER_LIMIT: usize = 280;
    pub const POWER_LIMIT_LEN: usize = 8;
    /// `temperatures[RUSD_TEMPERATURE_TYPE_GPU]`: ts, `NvTemp` (signed 24.8, °C).
    pub const TEMP_GPU: usize = 296;
    pub const TEMP_LEN: usize = 4;
    /// `avgPowerUsage`: ts, `averageGpuPower`, `averageModulePower`, `averageMemoryPower` (mW).
    pub const AVG_POWER: usize = 368;
    pub const AVG_POWER_LEN: usize = 12;
}

/// `RUSD_CLK_THROTTLE_REASON_*` (`cl00de.h`).
pub mod reason {
    pub const GPU_IDLE: u32 = 1 << 0;
    pub const APPLICATION_CLOCK_SETTING: u32 = 1 << 1;
    pub const SW_POWER_CAP: u32 = 1 << 2;
    pub const HW_SLOWDOWN: u32 = 1 << 3;
    pub const SYNC_BOOST: u32 = 1 << 4;
    pub const SW_THERMAL_SLOWDOWN: u32 = 1 << 5;
    pub const HW_THERMAL_SLOWDOWN: u32 = 1 << 6;
    pub const HW_POWER_BRAKES_SLOWDOWN: u32 = 1 << 7;
    pub const DISPLAY_CLOCK_SETTING: u32 = 1 << 8;
    /// Any thermal slow-down (software or hardware).
    pub const THERMAL: u32 = SW_THERMAL_SLOWDOWN | HW_THERMAL_SLOWDOWN;
}

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The dashboard, prepared. `enabled` stays false unless the OS sets it on purpose (C-151 point 1: on the board
/// only at the end of the chain, after the results are on the stick).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RusdPlan {
    pub enabled: bool,
    pub mask: u64,
    pub poll_ms: u32,
}

impl Default for RusdPlan {
    fn default() -> Self {
        RusdPlan {
            enabled: false,
            mask: poll::DASHBOARD,
            poll_ms: POLL_MS_DEFAULT,
        }
    }
}

/// The payload of `INIT_USER_SHARED_DATA` with this GPU-visible address (`0` = de-initialise, see
/// [`encode_deinit`]). The address must be 8-byte aligned (the structure's alignment).
pub fn encode_init(h_client: u32, h_subdevice: u32, phys: u64) -> GpuResult<Vec<u8>> {
    if phys == 0 {
        return Err(bad("rusd: adresa 0 (pentru oprire: encode_deinit)"));
    }
    if !phys.is_multiple_of(8) {
        return Err(bad("rusd: adresa nealiniata la 8 octeti"));
    }
    Ok(rmapi::encode_control(
        h_client,
        h_subdevice,
        CMD_INIT,
        &phys.to_le_bytes(),
    ))
}

/// `INIT_USER_SHARED_DATA` with `physAddr = 0`: GSP-RM stops writing the page ("0 to de-initialize").
pub fn encode_deinit(h_client: u32, h_subdevice: u32) -> Vec<u8> {
    rmapi::encode_control(h_client, h_subdevice, CMD_INIT, &[0; 8])
}

/// The payload of `SET_DATA_POLL`: `{NvU64 mask; NvU32 ms; 4 bytes padding}` (16 bytes). `ms = 0` keeps the
/// current interval (nouveau's reading of NVIDIA's code); otherwise it must be within 100..=1000.
pub fn encode_set_poll(h_client: u32, h_subdevice: u32, mask: u64, ms: u32) -> GpuResult<Vec<u8>> {
    if mask & !poll::ALL != 0 {
        return Err(bad("rusd: masca are biti necunoscuti"));
    }
    if ms != 0 && !(POLL_MS_MIN..=POLL_MS_MAX).contains(&ms) {
        return Err(bad("rusd: intervalul in afara 100..1000 ms"));
    }
    let mut p = [0u8; 16];
    p[..8].copy_from_slice(&mask.to_le_bytes());
    p[8..12].copy_from_slice(&ms.to_le_bytes());
    Ok(rmapi::encode_control(
        h_client,
        h_subdevice,
        CMD_SET_DATA_POLL,
        &p,
    ))
}

/// GSP-RM's status as a row value: `0` accepted, else `0x8000_0000 | status`.
fn answer(sent: &[u8], reply: &[u8]) -> GpuResult<u32> {
    let (st, _) = rmapi::control_reply_status(sent, reply)?;
    Ok(if st == 0 {
        0
    } else {
        0x8000_0000 | (st & 0x7fff_ffff)
    })
}

/// The names of the `kd` rows.
pub mod rows {
    pub const OFF: &str = "kd tabloul de bord: oprit, nimic trimis";
    pub const INIT: &str = "kd pornirea zonei comune (0 = acceptata; 8000xxxx = refuz)";
    pub const INTERVAL: &str = "kd intervalul de citire (0 = acceptat; 8000xxxx = refuz)";
    pub const MASK: &str = "kd cererea datelor (0 = acceptata; 8000xxxx = refuz)";
    pub const SECTIONS: &str = "kd sectiuni citite (biti: 0 ceasuri, 1 motive, 2 incarcare, 3 stare, 4 temperatura, 5 putere medie, 6 limita)";
    pub const BUSY: &str = "kd sectiuni ocupate dupa 10 incercari (aceiasi biti)";
    pub const TEMP: &str = "kd temperatura placii (grade C x 256, cu semn)";
    pub const CLK_GR: &str = "kd ceasul nucleului (MHz)";
    pub const CLK_MEM: &str = "kd ceasul memoriei (MHz)";
    pub const PSTATE: &str = "kd stare de performanta (0 = P0; 1xxxx = masca neobisnuita)";
    pub const LOAD_GPU: &str = "kd incarcarea nucleului (valoarea bruta)";
    pub const LOAD_MEM: &str = "kd incarcarea memoriei (valoarea bruta)";
    pub const REASONS: &str = "kd motivele incetinirii (masca; 0x20/0x40 = termic)";
    pub const POWER: &str = "kd puterea medie a placii (mW)";
    pub const LIMIT: &str = "kd limita de putere impusa (mW)";
    pub const STOP_MASK: &str = "kd oprirea citirilor (0 = acceptata; 8000xxxx = refuz)";
    pub const STOP_INIT: &str = "kd oprirea zonei comune (0 = acceptata; 8000xxxx = refuz)";
    pub const KEEP_PAGE: &str = "kd zona comuna NU s-a oprit: pagina ramane rezervata pana la repornire (nu se refoloseste)";
}

/// What the start did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RusdStart {
    /// `None` = the plan is off, nothing sent.
    pub init: Option<u32>,
    pub interval: Option<u32>,
    pub mask: Option<u32>,
}

impl RusdStart {
    /// True when GSP-RM took the page (it may write it until [`rusd_stop`] is accepted).
    pub fn page_given(&self) -> bool {
        self.init == Some(0)
    }

    /// True when the data was asked for and accepted: wait at least two intervals, then read.
    pub fn polling(&self) -> bool {
        self.page_given() && self.interval == Some(0) && self.mask == Some(0)
    }
}

/// C-151 point 1, the start (in NVIDIA's and nouveau's order): `INIT(phys)` → `SET_DATA_POLL(0, poll_ms)` (the
/// interval, with an empty mask) → `SET_DATA_POLL(mask, 0)`. Each answer is a `kd` row; a refusal stops the start
/// (never an error); only a transport error or a reply for another command is (`?`).
///
/// `phys` = the GPU-visible address of a zero-filled, physically contiguous page of system memory, which the OS
/// keeps reserved until [`rusd_stop`] says the page was taken back.
pub fn rusd_start(
    plan: &RusdPlan,
    phys: u64,
    h_client: u32,
    h_subdevice: u32,
    rm: &mut RmControl<'_>,
    j: &mut Journal,
) -> GpuResult<RusdStart> {
    let mut s = RusdStart {
        init: None,
        interval: None,
        mask: None,
    };
    if !plan.enabled {
        j.push(Event::Note(rows::OFF));
        return Ok(s);
    }
    if plan.poll_ms == 0 || plan.mask == 0 {
        return Err(bad("rusd: planul cere interval si masca nenule"));
    }
    // Everything is checked before the first byte goes out.
    let init = encode_init(h_client, h_subdevice, phys)?;
    let interval = encode_set_poll(h_client, h_subdevice, 0, plan.poll_ms)?;
    let mask = encode_set_poll(h_client, h_subdevice, plan.mask, 0)?;
    let mut send = |sent: &[u8], what: &'static str, j: &mut Journal| -> GpuResult<u32> {
        let reply = rm(sent)?;
        let v = answer(sent, &reply)?;
        j.push(Event::Value { what, value: v });
        Ok(v)
    };
    let v = send(&init, rows::INIT, j)?;
    s.init = Some(v);
    if v != 0 {
        return Ok(s);
    }
    let v = send(&interval, rows::INTERVAL, j)?;
    s.interval = Some(v);
    if v != 0 {
        return Ok(s);
    }
    s.mask = Some(send(&mask, rows::MASK, j)?);
    Ok(s)
}

/// What the stop did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RusdStop {
    pub mask: Option<u32>,
    pub deinit: Option<u32>,
}

impl RusdStop {
    /// True when the OS may give the page back (GSP-RM accepted `physAddr = 0`, or never took the page).
    pub fn page_free(&self, start: &RusdStart) -> bool {
        !start.page_given() || self.deinit == Some(0)
    }
}

/// The stop: `SET_DATA_POLL(0, 0)` (only if a mask was accepted) → `INIT(0)` (only if the page was taken). If
/// GSP-RM refuses `INIT(0)`, the `KEEP_PAGE` note says the page stays reserved until the next power-on.
pub fn rusd_stop(
    start: &RusdStart,
    h_client: u32,
    h_subdevice: u32,
    rm: &mut RmControl<'_>,
    j: &mut Journal,
) -> GpuResult<RusdStop> {
    let mut out = RusdStop {
        mask: None,
        deinit: None,
    };
    if !start.page_given() {
        return Ok(out);
    }
    if start.mask == Some(0) {
        let sent = encode_set_poll(h_client, h_subdevice, 0, 0)?;
        let reply = rm(&sent)?;
        let v = answer(&sent, &reply)?;
        j.push(Event::Value {
            what: rows::STOP_MASK,
            value: v,
        });
        out.mask = Some(v);
    }
    let sent = encode_deinit(h_client, h_subdevice);
    let reply = rm(&sent)?;
    let v = answer(&sent, &reply)?;
    j.push(Event::Value {
        what: rows::STOP_INIT,
        value: v,
    });
    out.deinit = Some(v);
    if v != 0 {
        j.push(Event::Note(rows::KEEP_PAGE));
    }
    Ok(out)
}

// ------------------------------------------------------------------------------ the reading

/// The page as the OS sees it. The OS implements it with **volatile** 64-bit reads for timestamps and an acquire
/// fence between the reads (`portAtomicMemoryFenceLoad` in `RUSD_READ_DATA`); the copy may be plain volatile
/// byte reads. GSP-RM writes the page at any moment.
pub trait RusdMem {
    /// One volatile little-endian 64-bit read at `offset` (8-byte aligned).
    fn read_u64(&mut self, offset: usize) -> u64;
    /// Copies `out.len()` bytes from `offset`.
    fn read_bytes(&mut self, offset: usize, out: &mut [u8]);
}

/// A plain byte slice (a snapshot: host tests and tools).
impl RusdMem for &[u8] {
    fn read_u64(&mut self, offset: usize) -> u64 {
        let mut b = [0u8; 8];
        b.copy_from_slice(&self[offset..offset + 8]);
        u64::from_le_bytes(b)
    }
    fn read_bytes(&mut self, offset: usize, out: &mut [u8]) {
        out.copy_from_slice(&self[offset..offset + out.len()]);
    }
}

/// `RUSD_SEQ_DATA_VALID` (`cl00de.h`).
pub fn ts_valid(ts: u64) -> bool {
    (ts < SEQ_START && ts != TS_INVALID) || (ts >= SEQ_START && ts & 1 == 0)
}

/// One section's read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Section<const N: usize> {
    /// Two equal valid timestamps around the copy.
    Ok { ts: u64, data: [u8; N] },
    /// The timestamp is `0` on every try: not written (not asked for, not supported, or not yet polled).
    Missing,
    /// Ten tries without two equal valid timestamps (GSP-RM was writing each time).
    Busy,
}

/// `RUSD_READ_DATA` for the section at `base` (timestamp at `base`, data at `base + 8`).
pub fn read_section<const N: usize>(mem: &mut dyn RusdMem, base: usize) -> Section<N> {
    let mut saw_invalid_only = true;
    for _ in 0..READ_TRIES {
        let t0 = mem.read_u64(base);
        if !ts_valid(t0) {
            if t0 != TS_INVALID {
                saw_invalid_only = false;
            }
            continue;
        }
        let mut data = [0u8; N];
        mem.read_bytes(base + 8, &mut data);
        let t1 = mem.read_u64(base);
        if t1 == t0 {
            return Section::Ok { ts: t0, data };
        }
        saw_invalid_only = false;
    }
    if saw_invalid_only {
        Section::Missing
    } else {
        Section::Busy
    }
}

fn u32_at(d: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([d[o], d[o + 1], d[o + 2], d[o + 3]])
}

/// The dashboard read once. `None` = that section was not read (see `sections` / `busy`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Dashboard {
    /// `NvTemp`: °C × 256, signed.
    pub temp_x256: Option<i32>,
    pub clk_graphics_mhz: Option<u32>,
    pub clk_memory_mhz: Option<u32>,
    /// The raw `currentPstate` mask.
    pub pstate_mask: Option<u32>,
    pub load_gpu: Option<u32>,
    pub load_mem: Option<u32>,
    pub reasons: Option<u32>,
    pub avg_power_mw: Option<u32>,
    pub power_limit_mw: Option<u32>,
    /// Bits of the sections read (see [`rows::SECTIONS`]).
    pub sections: u32,
    /// Bits of the sections still being written after 10 tries.
    pub busy: u32,
}

impl Dashboard {
    /// The temperature in whole °C (rounded towards minus infinity).
    pub fn temp_c(&self) -> Option<i32> {
        self.temp_x256.map(|t| t >> 8)
    }

    /// The P-state decoded like `PERF_GET_CURRENT_PSTATE`.
    pub fn pstate(&self) -> Option<Pstate> {
        self.pstate_mask.map(|m| {
            if m.count_ones() == 1 && m <= 0x8000 {
                Pstate::P(m.trailing_zeros() as u8)
            } else {
                Pstate::Odd(m)
            }
        })
    }

    /// True when a thermal slow-down is reported.
    pub fn thermal_slowdown(&self) -> bool {
        self.reasons.is_some_and(|r| r & reason::THERMAL != 0)
    }
}

/// Reads every dashboard section once and writes the `kd` rows (a row per value read, then the two masks).
pub fn read_dashboard(mem: &mut dyn RusdMem, j: &mut Journal) -> Dashboard {
    let mut d = Dashboard::default();
    let mark = |bit: u32, s_ok: bool, s_busy: bool, d: &mut Dashboard| {
        if s_ok {
            d.sections |= 1 << bit;
        }
        if s_busy {
            d.busy |= 1 << bit;
        }
    };
    macro_rules! sec {
        ($bit:expr, $base:expr, $len:expr, |$data:ident| $body:block) => {
            match read_section::<{ $len }>(mem, $base) {
                Section::Ok { data: $data, .. } => {
                    $body;
                    mark($bit, true, false, &mut d);
                }
                Section::Busy => mark($bit, false, true, &mut d),
                Section::Missing => {}
            }
        };
    }
    sec!(0, off::CLOCKS, off::CLOCKS_LEN, |x| {
        d.clk_graphics_mhz = Some(u32_at(&x, 0));
        d.clk_memory_mhz = Some(u32_at(&x, 4));
    });
    sec!(1, off::THROTTLE, off::THROTTLE_LEN, |x| {
        d.reasons = Some(u32_at(&x, 0));
    });
    sec!(2, off::UTIL, off::UTIL_LEN, |x| {
        d.load_gpu = Some(u32_at(&x, 0));
        d.load_mem = Some(u32_at(&x, 4));
    });
    sec!(3, off::PSTATE, off::PSTATE_LEN, |x| {
        d.pstate_mask = Some(u32_at(&x, 0));
    });
    sec!(4, off::TEMP_GPU, off::TEMP_LEN, |x| {
        d.temp_x256 = Some(u32_at(&x, 0) as i32);
    });
    sec!(5, off::AVG_POWER, off::AVG_POWER_LEN, |x| {
        d.avg_power_mw = Some(u32_at(&x, 0));
    });
    sec!(6, off::POWER_LIMIT, off::POWER_LIMIT_LEN, |x| {
        d.power_limit_mw = Some(u32_at(&x, 4));
    });
    let mut row = |what: &'static str, v: Option<u32>| {
        if let Some(value) = v {
            j.push(Event::Value { what, value });
        }
    };
    row(rows::TEMP, d.temp_x256.map(|t| t as u32));
    row(rows::CLK_GR, d.clk_graphics_mhz);
    row(rows::CLK_MEM, d.clk_memory_mhz);
    row(rows::PSTATE, d.pstate().map(Pstate::value));
    row(rows::LOAD_GPU, d.load_gpu);
    row(rows::LOAD_MEM, d.load_mem);
    row(rows::REASONS, d.reasons);
    row(rows::POWER, d.avg_power_mw);
    row(rows::LIMIT, d.power_limit_mw);
    row(rows::SECTIONS, Some(d.sections));
    row(rows::BUSY, Some(d.busy));
    d
}

/// C-151 point 1 on the board, **at the end of the chain, after the results are on the stick**: start → wait
/// (`sleep_ms(3 × poll_ms)`, so GSP-RM polled at least twice) → read → stop. With the plan off: one note, nothing
/// sent, nothing read. Returns the start, the dashboard (if it was read) and the stop.
#[allow(clippy::too_many_arguments)]
pub fn dashboard_at_end(
    plan: &RusdPlan,
    phys: u64,
    h_client: u32,
    h_subdevice: u32,
    rm: &mut RmControl<'_>,
    mem: &mut dyn RusdMem,
    sleep_ms: &mut dyn FnMut(u32),
    j: &mut Journal,
) -> GpuResult<(RusdStart, Option<Dashboard>, RusdStop)> {
    let s = rusd_start(plan, phys, h_client, h_subdevice, rm, j)?;
    let d = if s.polling() {
        sleep_ms(plan.poll_ms.saturating_mul(3));
        Some(read_dashboard(mem, j))
    } else {
        None
    };
    let e = rusd_stop(&s, h_client, h_subdevice, rm, j)?;
    Ok((s, d, e))
}

/// What a stick journal's `kd` rows say (the last value of each).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParsedKd {
    pub off: bool,
    pub init: Option<u32>,
    pub interval: Option<u32>,
    pub mask: Option<u32>,
    pub temp_x256: Option<i32>,
    pub clk_graphics_mhz: Option<u32>,
    pub clk_memory_mhz: Option<u32>,
    pub pstate: Option<u32>,
    pub load_gpu: Option<u32>,
    pub load_mem: Option<u32>,
    pub reasons: Option<u32>,
    pub avg_power_mw: Option<u32>,
    pub power_limit_mw: Option<u32>,
    pub sections: Option<u32>,
    pub busy: Option<u32>,
    pub stop_mask: Option<u32>,
    pub stop_init: Option<u32>,
    pub keep_page: bool,
}

fn kd_value(line: &str) -> Option<(&str, u32)> {
    let rest = line.split("valoare ").nth(1)?;
    let (name, v) = rest.rsplit_once(" = ")?;
    let v = u32::from_str_radix(v.trim().strip_prefix("0x")?, 16).ok()?;
    Some((name, v))
}

/// Reads the `kd` rows of a stick journal.
pub fn parse_kd(text: &str) -> ParsedKd {
    let mut k = ParsedKd::default();
    for line in text.lines() {
        if line.contains(rows::OFF) {
            k.off = true;
        }
        if line.contains(rows::KEEP_PAGE) {
            k.keep_page = true;
        }
        let Some((name, v)) = kd_value(line) else {
            continue;
        };
        if name == rows::TEMP {
            k.temp_x256 = Some(v as i32);
            continue;
        }
        let slot = match name {
            rows::INIT => &mut k.init,
            rows::INTERVAL => &mut k.interval,
            rows::MASK => &mut k.mask,
            rows::CLK_GR => &mut k.clk_graphics_mhz,
            rows::CLK_MEM => &mut k.clk_memory_mhz,
            rows::PSTATE => &mut k.pstate,
            rows::LOAD_GPU => &mut k.load_gpu,
            rows::LOAD_MEM => &mut k.load_mem,
            rows::REASONS => &mut k.reasons,
            rows::POWER => &mut k.avg_power_mw,
            rows::LIMIT => &mut k.power_limit_mw,
            rows::SECTIONS => &mut k.sections,
            rows::BUSY => &mut k.busy,
            rows::STOP_MASK => &mut k.stop_mask,
            rows::STOP_INIT => &mut k.stop_init,
            _ => continue,
        };
        *slot = Some(v);
    }
    k
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn payloads_have_the_570_144_shape() {
        let p0 = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_PARAMS as usize;
        let c = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_CMD as usize;
        let i = encode_init(1, 2, 0x1234_5000).unwrap();
        assert_eq!(&i[c..c + 4], &CMD_INIT.to_le_bytes());
        assert_eq!(&i[p0..], &0x1234_5000u64.to_le_bytes());
        assert!(encode_init(1, 2, 0).is_err());
        assert!(encode_init(1, 2, 0x1004).is_err());
        assert_eq!(&encode_deinit(1, 2)[p0..], &[0; 8]);
        let s = encode_set_poll(1, 2, poll::DASHBOARD, 500).unwrap();
        assert_eq!(&s[c..c + 4], &CMD_SET_DATA_POLL.to_le_bytes());
        assert_eq!(
            &s[p0..],
            &[0x1b, 0, 0, 0, 0, 0, 0, 0, 0xf4, 0x01, 0, 0, 0, 0, 0, 0]
        );
        assert!(encode_set_poll(1, 2, 0x40, 0).is_err());
        assert!(encode_set_poll(1, 2, 1, 99).is_err());
        assert!(encode_set_poll(1, 2, 1, 1001).is_err());
        assert!(encode_set_poll(1, 2, 1, 0).is_ok());
    }

    #[test]
    fn layout_fits_the_structure() {
        // Every section's end is before the next one and inside the 464 bytes.
        let secs = [
            (off::CLOCKS, off::CLOCKS_LEN),
            (off::THROTTLE, off::THROTTLE_LEN),
            (off::UTIL, off::UTIL_LEN),
            (off::PSTATE, off::PSTATE_LEN),
            (off::POWER_LIMIT, off::POWER_LIMIT_LEN),
            (off::TEMP_GPU, off::TEMP_LEN),
            (off::AVG_POWER, off::AVG_POWER_LEN),
        ];
        for w in secs.windows(2) {
            assert!(w[0].0 + 8 + w[0].1 <= w[1].0, "{w:?}");
        }
        for (b, l) in secs {
            assert_eq!(b % 8, 0);
            assert!(b + 8 + l <= SHARED_DATA_SIZE);
        }
    }

    #[test]
    fn timestamp_validity_follows_cl00de() {
        assert!(!ts_valid(0));
        assert!(ts_valid(1));
        assert!(ts_valid(SEQ_START - 1));
        assert!(ts_valid(SEQ_START));
        assert!(!ts_valid(SEQ_START + 1));
        assert!(!ts_valid(TS_WRITE_IN_PROGRESS));
    }
}
