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
}
