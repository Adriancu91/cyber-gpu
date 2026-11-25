//! C-157 (`docs/C-157_PROBA_BARIERA.md`): the grid-barrier probe — the smallest program that tests what `token.comp`
//! (one program per token, C-87) relies on: `G` workgroups resident at once meeting at a barrier through a counter in
//! memory, and a word written by one workgroup before the barrier seen by another after it.
//!
//! This module builds the probe's constant buffer and QMD, chooses `G`, judges the results the board would return and
//! turns two board times into the cost of one barrier. In SIMULARE the program's model
//! (`gpu-sim::cmdproc::grid_barrier_on_cpu`) shows the arguments and the expected results; whether the barrier holds
//! on a real SM (no hang, no stale word) is measured only on the board, with a time limit and the recovery ready.

use gpu_core::error::{GpuError, GpuResult};

use crate::gsp::work_queue::GrShape;
use crate::qmd::{ComputeQmd, MAX_SHARED_BYTES, SharedMemory};

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// `grid_barrier.comp` compiled by Mesa NAK 25.0.7 for SM 8.6 (`tools/nak/out/grid_barrier.sm86.txt`; SHA-256
/// `535f3537…3b12d`, session 237). COMPILAT, NERULAT pe placă.
pub const PROGRAM: &[u8] = include_bytes!("../../../tools/nak/out/grid_barrier.sm86.sass");
/// Registers per thread NAK reported (`Num GPRs`).
pub const REGISTERS: u32 = 19;
/// Threads per workgroup (one warp, as `token.comp`).
pub const BLOCK: u32 = 32;
/// NVK's root table: `cs.group_count[3]` at 0x0c / 0x10 / 0x14 (Mesa 25.0.7 `nvk_root_descriptor_table`). The program
/// reads `gl_NumWorkGroups.x` from 0x0c (listing: `c[0x0][0xc]`), so the driver MUST write G there — left at 0 the
/// barrier target would be 0 (no wait) and the neighbour index would divide by 0.
pub const CBUF_GROUP_COUNT: usize = 0x0c;
/// The constant buffer: sync 0x28, data 0x30, out 0x38, rounds 0x40.
pub const CBUF_SYNC: usize = 0x28;
pub const CBUF_DATA: usize = 0x30;
pub const CBUF_OUT: usize = 0x38;
pub const CBUF_ROUNDS: usize = 0x40;
/// The two round counts the OS runs: the difference / 1 000 = one barrier.
pub const ROUNDS_SHORT: u32 = 1;
pub const ROUNDS_LONG: u32 = 1_001;
/// At most two workgroups per SM in the first board run (`docs/C-157_PROBA_BARIERA.md` §2.3).
pub const MAX_PER_SM: u32 = 2;

/// Bytes the program needs: the counter (4), the data words (2·G·4), the results (G·4).
pub fn buffer_bytes(groups: u32) -> (u64, u64, u64) {
    (4, 8 * groups as u64, 4 * groups as u64)
}

/// The constant buffer for `groups` workgroups. The counter must be cleared (by the driver) before every launch.
pub fn cbuf(
    sync_va: u64,
    data_va: u64,
    out_va: u64,
    rounds: u32,
    groups: u32,
) -> GpuResult<[u8; 256]> {
    if rounds == 0
        || groups == 0
        || !sync_va.is_multiple_of(4)
        || !data_va.is_multiple_of(4)
        || !out_va.is_multiple_of(4)
    {
        return Err(bad("proba barierei: adrese nealiniate sau 0 runde"));
    }
    let mut c = [0u8; 256];
    c[CBUF_SYNC..CBUF_SYNC + 8].copy_from_slice(&sync_va.to_le_bytes());
    c[CBUF_DATA..CBUF_DATA + 8].copy_from_slice(&data_va.to_le_bytes());
    c[CBUF_OUT..CBUF_OUT + 8].copy_from_slice(&out_va.to_le_bytes());
    c[CBUF_ROUNDS..CBUF_ROUNDS + 4].copy_from_slice(&rounds.to_le_bytes());
    for (i, v) in [groups, 1, 1].into_iter().enumerate() {
        let at = CBUF_GROUP_COUNT + 4 * i;
        c[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    Ok(c)
}

/// `per_sm` workgroups on each SM GSP-RM reports (1 or 2): all resident, since a 32-thread workgroup without shared
/// memory is far below an SM's limits. Anything else is refused — the first run stays at ≤ 2 per SM.
pub fn groups_for(gr: &GrShape, per_sm: u32) -> GpuResult<u32> {
    let sms = gr.gpcs * gr.tpc_per_gpc * gr.sm_per_tpc;
    if sms == 0 {
        return Err(bad("proba barierei: niciun SM raportat"));
    }
    if per_sm == 0 || per_sm > MAX_PER_SM {
        return Err(bad("proba barierei: 1 sau 2 grupuri pe SM"));
    }
    Ok(sms * per_sm)
}

/// The QMD: `groups` workgroups of 32 threads, no shared memory. `registers` = what NAK reported for the program.
pub fn qmd(
    prog_va: u64,
    registers: u32,
    cbuf_va: u64,
    groups: u32,
) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    if groups == 0 {
        return Err(bad("proba barierei: 0 grupuri"));
    }
    ComputeQmd {
        program_va: prog_va,
        registers,
        cbuf0_va: cbuf_va,
        cbuf0_size: 0x100,
        grid: [groups, 1, 1],
        block: [BLOCK, 1, 1],
        release: None,
    }
    .to_bytes_with_shared(&SharedMemory {
        bytes: 0,
        max_bytes: MAX_SHARED_BYTES,
        barriers: 0,
    })
}

/// What one launch returned.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Counter = R·G and every workgroup saw its neighbour's word in every round.
    Holds,
    /// The counter ended elsewhere (a workgroup skipped or repeated a barrier).
    BadCounter { got: u32, want: u32 },
    /// Some rounds read a stale word: (workgroups with errors, total wrong rounds).
    StaleWords { groups: u32, rounds: u64 },
}

/// Judges the counter and the `out` words after a launch that finished (a launch that did not finish in time is the
/// caller's verdict, not this one).
pub fn judge(groups: u32, rounds: u32, counter: u32, out: &[u32]) -> GpuResult<Verdict> {
    if out.len() != groups as usize || groups == 0 {
        return Err(bad("proba barierei: rezultate de alta lungime decat G"));
    }
    let want = rounds.wrapping_mul(groups);
    if counter != want {
        return Ok(Verdict::BadCounter { got: counter, want });
    }
    let bad_groups = out.iter().filter(|&&w| w != 0).count() as u32;
    if bad_groups == 0 {
        return Ok(Verdict::Holds);
    }
    Ok(Verdict::StaleWords {
        groups: bad_groups,
        rounds: out.iter().map(|&w| w as u64).sum(),
    })
}

/// The cost of one barrier in ns from the two board times (`ROUNDS_SHORT`, `ROUNDS_LONG`). DEDUS from MĂSURAT.
pub fn barrier_ns(short_ns: u64, long_ns: u64) -> GpuResult<u64> {
    if long_ns <= short_ns {
        return Err(bad("proba barierei: rularea lunga nu e mai lunga"));
    }
    Ok((long_ns - short_ns) / (ROUNDS_LONG - ROUNDS_SHORT) as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdicts() {
        assert_eq!(judge(4, 3, 12, &[0; 4]).unwrap(), Verdict::Holds);
        assert_eq!(
            judge(4, 3, 11, &[0; 4]).unwrap(),
            Verdict::BadCounter { got: 11, want: 12 }
        );
        assert_eq!(
            judge(4, 3, 12, &[0, 2, 0, 1]).unwrap(),
            Verdict::StaleWords {
                groups: 2,
                rounds: 3
            }
        );
        assert!(judge(4, 3, 12, &[0; 3]).is_err());
        assert_eq!(barrier_ns(10_000, 1_510_000).unwrap(), 1_500);
        assert!(barrier_ns(10, 10).is_err());
        assert!(cbuf(2, 0, 0, 1, 4).is_err());
        assert!(cbuf(0, 0, 0, 0, 4).is_err());
        assert!(cbuf(0, 0, 0, 1, 0).is_err());
        let c = cbuf(0x100, 0x200, 0x300, 7, 96).unwrap();
        assert_eq!(&c[0x0c..0x18], &[96, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);
        assert_eq!(&c[..0x0c], &[0; 12]);
    }
}
