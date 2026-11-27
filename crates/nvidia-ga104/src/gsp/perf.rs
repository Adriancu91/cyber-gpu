//! C-147 stage 0 (c) and point 4 — the board's performance state through `GSP_RM_CONTROL`, built and read
//! **only in SIMULARE**. Nothing here touches the board: it only builds and reads bytes.
//!
//! **Read only (stage 0 c):** `NV2080_CTRL_CMD_PERF_GET_CURRENT_PSTATE` (0x20802068) on the subdevice; the answer
//! is one `NvU32 currPstate`, a bit mask of `NV2080_CTRL_PERF_PSTATES_P0..P15` (bit n = Pn). Sources (all three
//! read, `docs/C-147_ETAPA0.md` §3): NVIDIA 570.144 `ctrl2080perf.h` (command, structure, the P-state bits);
//! NVIDIA 570.144 `g_subdevice_nvoc.c` (export table: flags `0x40048` = `ROUTE_TO_PHYSICAL` — NVIDIA's own driver
//! passes this control **unchanged** to GSP-RM, which is what we do); NVIDIA 570.144 `rpc.c`
//! (`RmRpcPerfGetCurrentPstate`: the same single word on the vGPU path). Tinygrad, nouveau and nova do not use it.
//!
//! **Not sent:** `PERF_GET_LEVEL_INFO_V2` — the caller must fill the clock domains it asks about
//! (`NV2080_CTRL_PERF_GET_CLK_INFO.domain`), and the domain values (`NV2080_CTRL_CLK_DOMAIN_*`) are not in the
//! 570.144 public headers (`ctrl2080clk.h` is empty there). A guessed domain is not sent. The temperature: no
//! documented path either (`ctrl2080thermal.h` is empty in 570.144; no `dev_therm.h` for GA10x is published).
//!
//! **The boost (point 4, prepared, OFF by default — `BoostPlan::enabled` = false):** two shapes, both 8 bytes on
//! the subdevice. (A) what NVIDIA's own driver sends GSP-RM (`kern_perf_boost.c`, `kperfBoostSet`: the public
//! `PERF_BOOST` is handled on the CPU side — export flags `0x318`, not routed to GSP — and GSP-RM receives
//! `NV2080_CTRL_CMD_INTERNAL_PERF_BOOST_SET_2X` 0x20800a9a with `{NvBool flags; NvU32 duration}`, `ctrl2080internal.h`);
//! (B) what tinygrad sends GSP-RM from its own driver (`ops_nv.py`, `NVDevice.__init__`: the public `PERF_BOOST`
//! 0x2080200a `{NvU32 flags; NvU32 duration}`, duration `0xffffffff`, the answer's status **not checked**).
//! Our choice: (A) first, a finite duration (never `DURATION_INFINITE`), the same flags as tinygrad
//! (`BOOST_TO_MAX | CUDA_YES | CUDA_PRIORITY_HIGH` = 0x52), and the clear (`CMD_CLEAR | CUDA_YES` = 0x10, duration 0)
//! at the end of the test. A refusal is a journal row and the chain goes on as today.

use alloc::vec::Vec;

use gpu_core::diag::{Event, Journal};
use gpu_core::error::{GpuError, GpuResult};

use super::rmapi;

/// `NV2080_CTRL_CMD_PERF_GET_CURRENT_PSTATE` (`ctrl2080perf.h`, 570.144).
pub const CMD_GET_CURRENT_PSTATE: u32 = 0x2080_2068;
/// `NV2080_CTRL_CMD_PERF_BOOST` (public; shape B).
pub const CMD_PERF_BOOST: u32 = 0x2080_200a;
/// `NV2080_CTRL_CMD_INTERNAL_PERF_BOOST_SET_2X` (`ctrl2080internal.h`, 570.144; shape A).
pub const CMD_INTERNAL_BOOST_SET_2X: u32 = 0x2080_0a9a;

/// `NV2080_CTRL_PERF_BOOST_FLAGS_*` (`ctrl2080perf.h`): CMD 1:0, CUDA 4:4, ASYNC 5:5, CUDA_PRIORITY 6:6.
pub mod boost_flags {
    pub const CMD_CLEAR: u32 = 0;
    pub const CMD_BOOST_TO_MAX: u32 = 2;
    pub const CUDA_YES: u32 = 1 << 4;
    pub const CUDA_PRIORITY_HIGH: u32 = 1 << 6;
    /// What tinygrad asks for (and we would): to the highest P-state, as a compute ("CUDA") client, high priority.
    pub const TO_MAX_COMPUTE: u32 = CMD_BOOST_TO_MAX | CUDA_YES | CUDA_PRIORITY_HIGH;
    /// The clear of that boost (boosts are tracked per client and per CUDA flag).
    pub const CLEAR_COMPUTE: u32 = CMD_CLEAR | CUDA_YES;
}

/// `NV2080_CTRL_PERF_BOOST_DURATION_MAX` (seconds).
pub const BOOST_DURATION_MAX_S: u32 = 3600;
/// `NV2080_CTRL_PERF_BOOST_DURATION_INFINITE` — never sent by us.
pub const BOOST_DURATION_INFINITE: u32 = 0xffff_ffff;
/// Our boost's duration: 10 minutes (a test of `kronos --gpu` takes well under that).
pub const BOOST_DURATION_S: u32 = 600;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The payload of `GSP_RM_CONTROL` that reads the current P-state.
pub fn encode_get_pstate(h_client: u32, h_subdevice: u32) -> Vec<u8> {
    rmapi::encode_control(h_client, h_subdevice, CMD_GET_CURRENT_PSTATE, &[0; 4])
}

/// What one reading said.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pstate {
    /// `NV_OK`, one bit set: P0 (the highest) … P15.
    P(u8),
    /// `NV_OK` but the mask is not one bit (0, several, or bits above P15): kept as read, never interpreted.
    Odd(u32),
    /// GSP-RM refused (its `NV_STATUS`).
    Refused(u32),
}

impl Pstate {
    /// The journal value: 0..15 = Pn; `0x1_0000 | mask` = an odd mask; `0x8000_0000 | status` = refused.
    pub fn value(self) -> u32 {
        match self {
            Pstate::P(n) => n as u32,
            Pstate::Odd(m) => 0x1_0000 | (m & 0xffff),
            Pstate::Refused(s) => 0x8000_0000 | (s & 0x7fff_ffff),
        }
    }
}

/// Reads GSP-RM's answer to [`encode_get_pstate`]. A reply for another command or of another length is an error;
/// a refusal is not (it is a [`Pstate::Refused`] to journal).
pub fn read_pstate(sent: &[u8], reply: &[u8]) -> GpuResult<Pstate> {
    let (st, params) = rmapi::control_reply_status(sent, reply)?;
    if st != 0 {
        return Ok(Pstate::Refused(st));
    }
    let m = params
        .get(..4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .ok_or(bad("pstate: raspuns fara parametri"))?;
    Ok(if m.count_ones() == 1 && m <= 0x8000 {
        Pstate::P(m.trailing_zeros() as u8)
    } else {
        Pstate::Odd(m)
    })
}

/// When the P-state was read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum When {
    /// Before the boost (or at the end of the chain when there is no boost).
    Before,
    /// Right after the boost was asked for.
    AfterBoost,
    /// At the end of the test (after the clear, when there was a boost).
    End,
}

/// The names of the `ks` rows.
pub mod rows {
    pub const BEFORE: &str = "ks stare de performanta inainte (0 = P0; 8000xxxx = refuz)";
    pub const AFTER: &str = "ks stare de performanta dupa cerere (0 = P0; 8000xxxx = refuz)";
    pub const END: &str = "ks stare de performanta la final (0 = P0; 8000xxxx = refuz)";
    pub const BOOST: &str = "ks cerere viteza maxima (0 = acceptata; 8000xxxx = refuz)";
    pub const CLEAR: &str = "ks stergerea cererii (0 = acceptata; 8000xxxx = refuz)";
    pub const NO_LEVELS: &str =
        "ks nivelurile si temperatura: necitite (fara cale documentata in 570.144)";
    /// C-147 point 4: the P-state read while the boost still holds, right before its clear.
    pub const BEFORE_CLEAR: &str =
        "ks stare de performanta inainte de stergere (0 = P0; 8000xxxx = refuz)";
    /// C-147 point 4: the plan is off, nothing was sent (only the readings).
    pub const OFF: &str = "ks viteza maxima: oprita, nimic trimis (doar citiri)";
    /// C-147 point 4: one row with what the boost did (see [`super::BoostVerdict`]).
    pub const VERDICT: &str =
        "ks viteza maxima verdict (0 oprita, 1 urcata, 2 acceptata fara efect, 3 refuzata)";
}

/// One `ks` row for a P-state reading.
pub fn journal_pstate(j: &mut Journal, when: When, p: Pstate) {
    let what = match when {
        When::Before => rows::BEFORE,
        When::AfterBoost => rows::AFTER,
        When::End => rows::END,
    };
    j.push(Event::Value {
        what,
        value: p.value(),
    });
}

/// Which message the boost would be.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoostShape {
    /// NVIDIA's driver → GSP-RM: `INTERNAL_PERF_BOOST_SET_2X`, `{NvBool flags; NvU32 duration}`.
    Internal2x,
    /// tinygrad → GSP-RM: the public `PERF_BOOST`, `{NvU32 flags; NvU32 duration}`.
    Public,
}

/// The boost, prepared. `enabled` is false unless the OS sets it on purpose (the cycle mode, after the command
/// centre confirms in the hub that stage 0 showed a low state — C-147 completion of 19 nov 22:50).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoostPlan {
    pub enabled: bool,
    pub shape: BoostShape,
    pub duration_s: u32,
}

impl Default for BoostPlan {
    fn default() -> Self {
        BoostPlan {
            enabled: false,
            shape: BoostShape::Internal2x,
            duration_s: BOOST_DURATION_S,
        }
    }
}

fn boost_params(shape: BoostShape, flags: u32, duration: u32) -> [u8; 8] {
    let mut p = [0u8; 8];
    match shape {
        // NvBool (one byte) at 0, three bytes of padding, NvU32 at 4 (natural alignment).
        BoostShape::Internal2x => p[0] = flags as u8,
        BoostShape::Public => p[..4].copy_from_slice(&flags.to_le_bytes()),
    }
    p[4..].copy_from_slice(&duration.to_le_bytes());
    p
}

impl BoostPlan {
    fn command(&self) -> u32 {
        match self.shape {
            BoostShape::Internal2x => CMD_INTERNAL_BOOST_SET_2X,
            BoostShape::Public => CMD_PERF_BOOST,
        }
    }

    /// The `GSP_RM_CONTROL` payload asking for the highest P-state for `duration_s` seconds. Refused when the plan
    /// is not enabled, when the duration is 0, above `DURATION_MAX` or infinite.
    pub fn encode_boost(&self, h_client: u32, h_subdevice: u32) -> GpuResult<Vec<u8>> {
        if !self.enabled {
            return Err(GpuError::Blocked(
                "viteza maxima: oprita implicit (C-147 punctul 4, doar in modul ciclu, dupa confirmarea centrului)",
            ));
        }
        if self.duration_s == 0 || self.duration_s > BOOST_DURATION_MAX_S {
            return Err(bad("viteza maxima: durata 0 sau peste 3600 s"));
        }
        Ok(rmapi::encode_control(
            h_client,
            h_subdevice,
            self.command(),
            &boost_params(self.shape, boost_flags::TO_MAX_COMPUTE, self.duration_s),
        ))
    }

    /// The clear of the same boost (same shape, `CLEAR_COMPUTE`, duration 0). Allowed even when the plan is not
    /// enabled: clearing a boost nobody asked for changes nothing.
    pub fn encode_clear(&self, h_client: u32, h_subdevice: u32) -> Vec<u8> {
        rmapi::encode_control(
            h_client,
            h_subdevice,
            self.command(),
            &boost_params(self.shape, boost_flags::CLEAR_COMPUTE, 0),
        )
    }

    /// C-157 p.2 after os-485 (session 269): true when a piece of work that starts `since_request_ms` after the
    /// accepted boost and lasts `run_ms` ends inside the boost's `duration_s`. Pure arithmetic, nothing is sent.
    /// After the duration the boost ends by itself and the board may fall back to its low state (os-485: requested
    /// at 114.3 s, P8 at 210 MHz at the end, 1 698 s); a time measured outside the window is not a time at full speed.
    pub fn covers_ms(&self, since_request_ms: u64, run_ms: u64) -> bool {
        self.enabled && since_request_ms.saturating_add(run_ms) <= u64::from(self.duration_s) * 1000
    }
}

// ------------------------------------------------------------------------------ point 4, ready to link

/// What GSP-RM said to the boost.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoostAnswer {
    /// The plan is off: nothing was sent.
    Off,
    /// `NV_OK`.
    Accepted,
    /// Refused with this `NV_STATUS` (a row; the chain goes on as today).
    Refused(u32),
}

/// What the boost did, judged from the two readings around it (one `ks … verdict` row).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoostVerdict {
    /// Nothing sent.
    Off = 0,
    /// Accepted and the P-state read after is a higher state (a lower number) than before.
    Raised = 1,
    /// Accepted, but the P-state did not move (or a reading was odd / refused): "other factors can limit
    /// P-States" (`ctrl2080perf.h`) — the reading before the clear says whether it moved later.
    NoEffect = 2,
    /// Refused by GSP-RM.
    Refused = 3,
}

/// The boost's start: the reading before, the answer, the reading after (only when accepted).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoostStart {
    pub before: Pstate,
    pub answer: BoostAnswer,
    pub after: Option<Pstate>,
}

impl BoostStart {
    pub fn verdict(&self) -> BoostVerdict {
        match (self.answer, self.before, self.after) {
            (BoostAnswer::Off, ..) => BoostVerdict::Off,
            (BoostAnswer::Refused(_), ..) => BoostVerdict::Refused,
            (BoostAnswer::Accepted, Pstate::P(b), Some(Pstate::P(a))) if a < b => {
                BoostVerdict::Raised
            }
            (BoostAnswer::Accepted, ..) => BoostVerdict::NoEffect,
        }
    }

    /// True when the boost holds on the board and must be cleared at the end.
    pub fn needs_clear(&self) -> bool {
        self.answer == BoostAnswer::Accepted
    }
}

/// The boost's end: the reading before the clear (only if a boost holds), the clear's answer (`0` or
/// `8000xxxx`; `None` = nothing to clear), the reading at the end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoostEnd {
    pub before_clear: Option<Pstate>,
    pub clear: Option<u32>,
    pub end: Pstate,
}

/// One `GSP_RM_CONTROL`: the payload in, GSP-RM's reply out (the OS's `OsRm::call(GSP_RM_CONTROL, …)`).
pub type RmControl<'a> = dyn FnMut(&[u8]) -> GpuResult<Vec<u8>> + 'a;

fn read_now(h_client: u32, h_subdevice: u32, rm: &mut RmControl<'_>) -> GpuResult<Pstate> {
    let sent = encode_get_pstate(h_client, h_subdevice);
    let reply = rm(&sent)?;
    read_pstate(&sent, &reply)
}

/// C-147 point 4, **once, right after the work channel is open** (before `i14-simbol` / `kronos --gpu`):
/// `ks inainte` → the boost (`ks cerere`, only if `plan.enabled`) → `ks dupa cerere` (only if accepted) →
/// `ks … verdict`. A refusal of GSP-RM (of the reading or of the boost) is a row, never an error; only a transport
/// error or a reply for another command is (`?`). With the plan off nothing is sent but the reading.
pub fn boost_start(
    plan: &BoostPlan,
    h_client: u32,
    h_subdevice: u32,
    rm: &mut RmControl<'_>,
    j: &mut Journal,
) -> GpuResult<BoostStart> {
    let before = read_now(h_client, h_subdevice, rm)?;
    journal_pstate(j, When::Before, before);
    let (answer, after) = if plan.enabled {
        let sent = plan.encode_boost(h_client, h_subdevice)?;
        let reply = rm(&sent)?;
        let v = read_boost(&sent, &reply)?;
        j.push(Event::Value {
            what: rows::BOOST,
            value: v,
        });
        if v == 0 {
            let a = read_now(h_client, h_subdevice, rm)?;
            journal_pstate(j, When::AfterBoost, a);
            (BoostAnswer::Accepted, Some(a))
        } else {
            (BoostAnswer::Refused(v & 0x7fff_ffff), None)
        }
    } else {
        j.push(Event::Note(rows::OFF));
        (BoostAnswer::Off, None)
    };
    let s = BoostStart {
        before,
        answer,
        after,
    };
    j.push(Event::Value {
        what: rows::VERDICT,
        value: s.verdict() as u32,
    });
    Ok(s)
}

/// C-147 point 4, **at the end of the test** (after `kronos --gpu`, `kp`, `kc`; the place of today's `ks la final`):
/// if a boost holds — `ks inainte de stergere` → the clear in the same shape (`ks stergerea`); then always
/// `ks la final`. Call it on the error paths too, when the channel still answers: the boost also ends by itself
/// after `plan.duration_s`.
pub fn boost_finish(
    start: &BoostStart,
    plan: &BoostPlan,
    h_client: u32,
    h_subdevice: u32,
    rm: &mut RmControl<'_>,
    j: &mut Journal,
) -> GpuResult<BoostEnd> {
    let (before_clear, clear) = if start.needs_clear() {
        let b = read_now(h_client, h_subdevice, rm)?;
        j.push(Event::Value {
            what: rows::BEFORE_CLEAR,
            value: b.value(),
        });
        let sent = plan.encode_clear(h_client, h_subdevice);
        let reply = rm(&sent)?;
        let c = read_boost(&sent, &reply)?;
        j.push(Event::Value {
            what: rows::CLEAR,
            value: c,
        });
        (Some(b), Some(c))
    } else {
        (None, None)
    };
    let end = read_now(h_client, h_subdevice, rm)?;
    journal_pstate(j, When::End, end);
    Ok(BoostEnd {
        before_clear,
        clear,
        end,
    })
}

/// What a stick journal's `ks` rows say (the last value of each; for the cycle report and the host tools).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ParsedKs {
    pub before: Option<u32>,
    pub boost: Option<u32>,
    pub after: Option<u32>,
    pub verdict: Option<u32>,
    pub before_clear: Option<u32>,
    pub clear: Option<u32>,
    pub end: Option<u32>,
    /// The `ks viteza maxima: oprita` note was there.
    pub off: bool,
}

/// One `… valoare <name> = 0x<hex>` row (the same shape `kronos_perf::value_row` reads).
fn ks_value(line: &str) -> Option<(&str, u32)> {
    let rest = line.split("valoare ").nth(1)?;
    let (name, v) = rest.rsplit_once(" = ")?;
    let v = u32::from_str_radix(v.trim().strip_prefix("0x")?, 16).ok()?;
    Some((name, v))
}

/// Reads the `ks` rows of a stick journal.
pub fn parse_ks(text: &str) -> ParsedKs {
    let mut k = ParsedKs::default();
    for line in text.lines() {
        if line.contains(rows::OFF) {
            k.off = true;
        }
        let Some((name, v)) = ks_value(line) else {
            continue;
        };
        let slot = match name {
            rows::BEFORE => &mut k.before,
            rows::BOOST => &mut k.boost,
            rows::AFTER => &mut k.after,
            rows::VERDICT => &mut k.verdict,
            rows::BEFORE_CLEAR => &mut k.before_clear,
            rows::CLEAR => &mut k.clear,
            rows::END => &mut k.end,
            _ => continue,
        };
        *slot = Some(v);
    }
    k
}

/// GSP-RM's status for a boost or a clear: `0` accepted, else `0x8000_0000 | status` (the `ks` row's value).
pub fn read_boost(sent: &[u8], reply: &[u8]) -> GpuResult<u32> {
    let (st, _) = rmapi::control_reply_status(sent, reply)?;
    Ok(if st == 0 {
        0
    } else {
        0x8000_0000 | (st & 0x7fff_ffff)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reply(sent: &[u8], status: u32, params: &[u8]) -> Vec<u8> {
        let mut r = sent.to_vec();
        let st = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_STATUS as usize;
        r[st..st + 4].copy_from_slice(&status.to_le_bytes());
        let p0 = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_PARAMS as usize;
        r[p0..p0 + params.len()].copy_from_slice(params);
        r
    }

    #[test]
    fn pstate_read_refused_or_odd() {
        let s = encode_get_pstate(0xc1e0_0004, 0x5d1d_0001);
        assert_eq!(
            s.len(),
            nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_PARAMS as usize + 4
        );
        assert_eq!(
            read_pstate(&s, &reply(&s, 0, &1u32.to_le_bytes())),
            Ok(Pstate::P(0))
        );
        assert_eq!(
            read_pstate(&s, &reply(&s, 0, &0x100u32.to_le_bytes())),
            Ok(Pstate::P(8))
        );
        assert_eq!(
            read_pstate(&s, &reply(&s, 0, &3u32.to_le_bytes())),
            Ok(Pstate::Odd(3))
        );
        assert_eq!(
            read_pstate(&s, &reply(&s, 0x56, &[0; 4])),
            Ok(Pstate::Refused(0x56))
        );
        assert_eq!(Pstate::Refused(0x56).value(), 0x8000_0056);
        assert_eq!(Pstate::P(2).value(), 2);
        // A reply for another command is an error, not a reading.
        let mut other = reply(&s, 0, &1u32.to_le_bytes());
        let c = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_CMD as usize;
        other[c] ^= 1;
        assert!(read_pstate(&s, &other).is_err());
    }

    #[test]
    fn boost_is_off_by_default_finite_and_cleared_in_the_same_shape() {
        let p = BoostPlan::default();
        assert!(!p.enabled);
        assert!(matches!(p.encode_boost(1, 2), Err(GpuError::Blocked(_))));
        let on = BoostPlan { enabled: true, ..p };
        let b = on.encode_boost(1, 2).unwrap();
        let p0 = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_PARAMS as usize;
        let c = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_CMD as usize;
        assert_eq!(&b[c..c + 4], &CMD_INTERNAL_BOOST_SET_2X.to_le_bytes());
        assert_eq!(&b[p0..], &[0x52, 0, 0, 0, 0x58, 0x02, 0, 0]);
        let cl = on.encode_clear(1, 2);
        assert_eq!(&cl[p0..], &[0x10, 0, 0, 0, 0, 0, 0, 0]);
        let pubb = BoostPlan {
            shape: BoostShape::Public,
            ..on
        };
        let b = pubb.encode_boost(1, 2).unwrap();
        assert_eq!(&b[c..c + 4], &CMD_PERF_BOOST.to_le_bytes());
        assert_eq!(&b[p0..], &[0x52, 0, 0, 0, 0x58, 0x02, 0, 0]);
        for d in [0, BOOST_DURATION_MAX_S + 1, BOOST_DURATION_INFINITE] {
            let x = BoostPlan {
                duration_s: d,
                ..on
            };
            assert!(x.encode_boost(1, 2).is_err(), "{d}");
        }
        assert_eq!(read_boost(&cl, &reply(&cl, 0, &[0; 8])), Ok(0));
        assert_eq!(read_boost(&cl, &reply(&cl, 0x56, &[0; 8])), Ok(0x8000_0056));
    }

    #[test]
    fn covers_ms_the_window() {
        let off = BoostPlan::default();
        assert!(!off.covers_ms(0, 1), "plan off: nothing holds");
        let p = BoostPlan {
            enabled: true,
            ..BoostPlan::default()
        };
        assert!(p.covers_ms(0, 600_000));
        assert!(!p.covers_ms(0, 600_001));
        // os-485: requested at 114.3 s of the journal; v5@64k dX at 677.6 s, v6 dX at 722.9 s.
        assert!(p.covers_ms(677_600 - 114_300, 5_000));
        assert!(!p.covers_ms(722_900 - 114_300, 5_000));
        assert!(!p.covers_ms(u64::MAX, u64::MAX), "no overflow");
        let long = BoostPlan {
            duration_s: 1800,
            ..p
        };
        assert!(long.covers_ms(1_697_600 - 114_300, 0));
    }
}
