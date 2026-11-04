//! C-69 — measurement 1 of the "one program per symbol" target
//! (`docs/TINTA_UN_SINGUR_PROGRAM.md`): **what one launch costs** on the
//! board, before we decide whether one big program is worth it.
//!
//! The method: a start marker, then N launches, each followed by a host
//! semaphore release *with a timestamp* (`SEM_EXECUTE.RELEASE_TIMESTAMP`,
//! `RELEASE_WFI` on, so the time is taken after the launch finished). The
//! GPU writes a 16-byte record per release — payload at +0, its own clock
//! in ns at +8 (layout DEDUS from tinygrad's signal: value at 0, timestamp
//! at 8; MIT, read as specification). The differences between consecutive
//! records are the cost of one launch + its (tiny) work, measured by the
//! GPU itself, without the CPU in the loop.
//!
//! Tested **only in SIMULARE** (`gpu_sim::cmdproc` with a simulated clock):
//! the numbers there are the test's own, only the method is checked.
//! Nothing here touches a board; this module only encodes and reads bytes.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};
use nvidia_rm570::regs::{clc7c0 as c, clc56f as h};

use crate::push::PushBuf;

/// Bytes per timestamp record (payload, then the time at +8).
pub const RECORD: u64 = 16;

/// The `SEM_EXECUTE` word: 32-bit release, wait for idle, with timestamp.
pub fn sem_execute_timestamp() -> u32 {
    let d = h::NVC56F_SEM_EXECUTE_OPERATION.set(0, h::NVC56F_SEM_EXECUTE_OPERATION_RELEASE);
    let d = h::NVC56F_SEM_EXECUTE_RELEASE_WFI.set(d, h::NVC56F_SEM_EXECUTE_RELEASE_WFI_EN);
    h::NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP.set(d, h::NVC56F_SEM_EXECUTE_RELEASE_TIMESTAMP_EN)
}

/// A timestamp release of `payload` into the record at `va` (16-aligned).
pub fn timestamp_release(p: &mut PushBuf<'_>, va: u64, payload: u32) -> GpuResult<()> {
    if !va.is_multiple_of(RECORD) || va >> 57 != 0 {
        return Err(GpuError::InvalidArgument(
            "marca de timp: adresa nealiniata la 16 sau prea mare",
        ));
    }
    p.inc(
        0,
        h::NVC56F_SEM_ADDR_LO,
        &[
            va as u32,
            (va >> 32) as u32,
            payload,
            0,
            sem_execute_timestamp(),
        ],
    )
}

/// The whole measurement as one pushbuffer: record 0 = start marker, then
/// for k in 1..=n: launch the QMD at `qmd_va`, release record k with payload
/// `first_payload + k`. The compute object must already be bound on
/// subchannel 1 (`SET_OBJECT`), as in every launch of the work channel.
pub fn encode(
    p: &mut PushBuf<'_>,
    qmd_va: u64,
    records_va: u64,
    n: u32,
    first_payload: u32,
) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::InvalidArgument("masuratoare cu 0 lansari"));
    }
    if !qmd_va.is_multiple_of(256) {
        return Err(GpuError::InvalidArgument("QMD nealiniat la 256"));
    }
    timestamp_release(p, records_va, first_payload)?;
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    for k in 1..=n {
        p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(qmd_va >> 8) as u32])?;
        p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])?;
        timestamp_release(
            p,
            records_va + RECORD * u64::from(k),
            first_payload.wrapping_add(k),
        )?;
    }
    Ok(())
}

/// What the records say.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LaunchCost {
    pub launches: u32,
    /// Per-launch times in ns, in order.
    pub each_ns: Vec<u64>,
    pub min_ns: u64,
    pub median_ns: u64,
    pub max_ns: u64,
    pub total_ns: u64,
}

impl LaunchCost {
    /// One line for the OS screen and the stick journal.
    pub fn line(&self) -> String {
        format!(
            "cost lansare: {} lansari, mediana {} ns, minim {} ns, maxim {} ns, total {} ns",
            self.launches, self.median_ns, self.min_ns, self.max_ns, self.total_ns
        )
    }
}

/// Reads `n + 1` records (as copied back from GPU memory) and checks them:
/// every payload present and in order (each launch finished and released),
/// time never going back. A missing or out-of-order payload is an error,
/// never a guessed number.
pub fn analyze(bytes: &[u8], n: u32, first_payload: u32) -> GpuResult<LaunchCost> {
    let need = (u64::from(n) + 1) * RECORD;
    if n == 0 || (bytes.len() as u64) < need {
        return Err(GpuError::InvalidArgument(
            "marca de timp: prea putine inregistrari",
        ));
    }
    let rec = |k: usize| {
        let b = &bytes[k * RECORD as usize..(k + 1) * RECORD as usize];
        let payload = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
        let mut t = [0u8; 8];
        t.copy_from_slice(&b[8..16]);
        (payload, u64::from_le_bytes(t))
    };
    let mut each = Vec::with_capacity(n as usize);
    let (p0, mut prev) = rec(0);
    if p0 != first_payload {
        return Err(GpuError::InvalidArgument(
            "marca de timp: lipseste marcajul de start",
        ));
    }
    for k in 1..=n as usize {
        let (p, t) = rec(k);
        if p != first_payload.wrapping_add(k as u32) {
            return Err(GpuError::InvalidArgument(
                "marca de timp: o lansare nu s-a terminat (valoare lipsa sau in alta ordine)",
            ));
        }
        if t < prev {
            return Err(GpuError::InvalidArgument(
                "marca de timp: timpul merge inapoi",
            ));
        }
        each.push(t - prev);
        prev = t;
    }
    let mut sorted = each.clone();
    sorted.sort_unstable();
    Ok(LaunchCost {
        launches: n,
        min_ns: sorted[0],
        median_ns: sorted[sorted.len() / 2],
        max_ns: sorted[sorted.len() - 1],
        total_ns: each.iter().sum(),
        each_ns: each,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn records(v: &[(u32, u64)]) -> Vec<u8> {
        v.iter()
            .flat_map(|&(p, t)| {
                let mut r = [0u8; 16];
                r[..4].copy_from_slice(&p.to_le_bytes());
                r[8..].copy_from_slice(&t.to_le_bytes());
                r
            })
            .collect()
    }

    #[test]
    fn records_give_min_median_max() {
        let b = records(&[(100, 1000), (101, 1500), (102, 3500), (103, 4200)]);
        let c = analyze(&b, 3, 100).unwrap();
        assert_eq!(c.each_ns, [500, 2000, 700]);
        assert_eq!(
            (c.min_ns, c.median_ns, c.max_ns, c.total_ns),
            (500, 700, 2000, 3200)
        );
        assert!(c.line().contains("mediana 700 ns"));
    }

    #[test]
    fn missing_or_wrong_records_are_errors() {
        // A launch that never finished: its payload is still the old value.
        let b = records(&[(100, 1000), (101, 1500), (0, 0), (103, 4200)]);
        assert!(analyze(&b, 3, 100).is_err());
        // Time going back.
        let b = records(&[(100, 1000), (101, 900)]);
        assert!(analyze(&b, 1, 100).is_err());
        // No start marker, too few bytes, zero launches.
        assert!(analyze(&records(&[(7, 0), (101, 1)]), 1, 100).is_err());
        assert!(analyze(&records(&[(100, 0)]), 1, 100).is_err());
        assert!(analyze(&records(&[(100, 0)]), 0, 100).is_err());
    }

    #[test]
    fn encoding_refuses_bad_addresses() {
        let mut w = [0u32; 64];
        let mut p = PushBuf::new(&mut w);
        assert!(timestamp_release(&mut p, 0x1008, 1).is_err());
        assert!(encode(&mut p, 0x1080, 0x2000, 1, 0).is_err());
        assert!(encode(&mut p, 0x1000, 0x2000, 0, 0).is_err());
        assert!(p.is_empty(), "nothing encoded after a refusal");
    }
}
