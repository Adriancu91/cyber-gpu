//! C-87 step 2 — "one program per token": the whole launch list of a token ([`crate::kronos_model::token_launches`],
//! `16·layers + 3` operations, 195 for #513) run by **one** launch of the persistent program `token.comp` (Mesa NAK,
//! SM 8.6), instead of one launch per operation.
//!
//! The driver writes the **work list** ([`work_list`]): per operation 256 bytes = the separate launch's constant
//! buffer (arguments from 0x28, same layout) + the operation number at 0x20 and its grid at 0x24. The program's own
//! constant buffer ([`token_cbuf`]) names the list, the barrier counter (one u32, cleared before every launch) and the
//! number of operations. `G` workgroups of 32 threads ([`token_qmd`]) share each operation's blocks and meet at a grid
//! barrier between operations — correct **only if all G are resident at once**: G ≤ the SM count GSP-RM reports
//! ([`groups_for`]).
//!
//! What is proven where: in SIMULARE the program's model (`gpu-sim::cmdproc::token_on_cpu`) runs the list through the
//! same program models as the separate launches, so it shows the list, the constant buffer and the QMD are right and
//! the results equal the 195 launches. It says **nothing** about the grid barrier on the board (a real SM could hang
//! there if G were too big, or read stale data if the coherence were wrong): that is measured only in the window, after
//! the separate launches passed, with the recovery (C-87 step 3) ready.

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::gsp::work_queue::GrShape;
use crate::kronos_layer::{Launch, Prog};
use crate::qmd::{ComputeQmd, MAX_SHARED_BYTES, SharedMemory};

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// `token.comp` compiled by Mesa NAK 25.0.7 for SM 8.6 (`tools/nak/out/token.sm86.txt`).
pub const TOKEN: &[u8] = include_bytes!("../../../tools/nak/out/token.sm86.sass");
/// Registers per thread NAK reported for it (`Num GPRs`).
pub const TOKEN_REGISTERS: u32 = 35;
/// Threads per workgroup.
pub const TOKEN_BLOCK: u32 = 32;
/// Shared memory: the attention's scores (512 f32) + the 32 partial sums.
pub const TOKEN_SHARED: u32 = 4 * (512 + 32);
/// Bytes per operation in the work list.
pub const ENTRY: usize = 256;
/// Where the operation number and its grid sit in an entry (below the arguments at 0x28).
pub const ENTRY_OP: usize = 0x20;
pub const ENTRY_GRID: usize = 0x24;
/// The program's own constant buffer: list 0x28, sync 0x30, n 0x38.
pub const CBUF_LIST: usize = 0x28;
pub const CBUF_SYNC: usize = 0x30;
pub const CBUF_N: usize = 0x38;

/// The operation number `token.comp` dispatches on (= the order of `kronos_layer::PROGRAMS`).
pub fn op_number(p: Prog) -> u32 {
    match p {
        Prog::Copy => 0,
        Prog::Q8Matvec => 1,
        Prog::Vadd => 2,
        Prog::LayerNorm => 3,
        Prog::Attention => 4,
        Prog::Gelu => 5,
        Prog::Embed => 6,
    }
}

/// The work list of `launches` (256 bytes each). Refuses an empty list or a constant buffer that uses bytes 0x20..0x28
/// (they carry the operation and grid here; ours never do).
pub fn work_list(launches: &[Launch]) -> GpuResult<Vec<u8>> {
    if launches.is_empty() {
        return Err(bad("lista de lucru goala"));
    }
    let mut out = Vec::with_capacity(ENTRY * launches.len());
    for l in launches {
        if l.cbuf[..CBUF_LIST].iter().any(|&b| b != 0) || l.grid == 0 {
            return Err(bad(
                "lista de lucru: bufferul constant foloseste 0x00..0x28 sau grila 0",
            ));
        }
        let mut e = l.cbuf;
        e[ENTRY_OP..ENTRY_OP + 4].copy_from_slice(&op_number(l.prog).to_le_bytes());
        e[ENTRY_GRID..ENTRY_GRID + 4].copy_from_slice(&l.grid.to_le_bytes());
        out.extend_from_slice(&e);
    }
    Ok(out)
}

/// The program's constant buffer.
pub fn token_cbuf(list_va: u64, sync_va: u64, n: usize) -> GpuResult<[u8; 256]> {
    if n == 0 || n > u32::MAX as usize || !list_va.is_multiple_of(16) || !sync_va.is_multiple_of(4)
    {
        return Err(bad(
            "programul pe simbol: lista sau contorul nealiniate, ori n = 0",
        ));
    }
    let mut c = [0u8; 256];
    c[CBUF_LIST..CBUF_LIST + 8].copy_from_slice(&list_va.to_le_bytes());
    c[CBUF_SYNC..CBUF_SYNC + 8].copy_from_slice(&sync_va.to_le_bytes());
    c[CBUF_N..CBUF_N + 4].copy_from_slice(&(n as u32).to_le_bytes());
    Ok(c)
}

/// How many workgroups: one per SM (all resident: 32 threads and 2 176 B of shared memory each are far below an SM's
/// limits), capped by `max` (the widest operation needs no more than its grid; more only wait at the barriers).
pub fn groups_for(gr: &GrShape, max: u32) -> GpuResult<u32> {
    let sms = gr.gpcs * gr.tpc_per_gpc * gr.sm_per_tpc;
    if sms == 0 || max == 0 {
        return Err(bad("programul pe simbol: niciun SM raportat"));
    }
    Ok(sms.min(max))
}

/// The QMD of the one launch: `groups` workgroups of 32 threads.
pub fn token_qmd(
    prog_va: u64,
    cbuf_va: u64,
    groups: u32,
) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    if groups == 0 {
        return Err(bad("programul pe simbol: 0 grupuri"));
    }
    ComputeQmd {
        program_va: prog_va,
        registers: TOKEN_REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: 0x100,
        grid: [groups, 1, 1],
        block: [TOKEN_BLOCK, 1, 1],
        release: None,
    }
    .to_bytes_with_shared(&SharedMemory {
        bytes: TOKEN_SHARED,
        max_bytes: MAX_SHARED_BYTES,
        barriers: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kronos_layer::PROGRAMS;

    #[test]
    fn the_list_carries_every_launch() {
        for (i, p) in PROGRAMS.iter().enumerate() {
            assert_eq!(op_number(*p), i as u32);
        }
        let mut cb = [0u8; 256];
        cb[0x28] = 7;
        let l = Launch {
            prog: Prog::Gelu,
            cbuf: cb,
            grid: 3,
        };
        let w = work_list(&[l.clone(), l.clone()]).unwrap();
        assert_eq!(w.len(), 512);
        assert_eq!(&w[256 + 0x20..256 + 0x28], &[5, 0, 0, 0, 3, 0, 0, 0]);
        assert_eq!(w[256 + 0x28], 7);
        assert_eq!(&w[..0x20], &[0u8; 0x20]);
        assert!(work_list(&[]).is_err());
        let mut odd = l.clone();
        odd.cbuf[0x20] = 1;
        assert!(work_list(&[odd]).is_err());
        assert!(work_list(&[Launch { grid: 0, ..l }]).is_err());
    }

    #[test]
    fn cbuf_qmd_and_groups() {
        let c = token_cbuf(0x10_0000_1000, 0x10_0000_2000, 195).unwrap();
        assert_eq!(
            u64::from_le_bytes(c[0x28..0x30].try_into().unwrap()),
            0x10_0000_1000
        );
        assert_eq!(u32::from_le_bytes(c[0x38..0x3c].try_into().unwrap()), 195);
        assert!(token_cbuf(0x10_0000_1008, 0x10_0000_2000, 1).is_err());
        assert!(token_cbuf(0x10_0000_1000, 0x10_0000_2000, 0).is_err());
        // The RTX 3070 Ti: 6 GPCs × 4 TPCs × 2 SMs = 48.
        let gr = GrShape {
            gpcs: 6,
            tpc_per_gpc: 4,
            sm_per_tpc: 2,
            max_warps_per_sm: 48,
        };
        assert_eq!(groups_for(&gr, 1000).unwrap(), 48);
        assert_eq!(groups_for(&gr, 12).unwrap(), 12);
        assert!(groups_for(&GrShape { gpcs: 0, ..gr }, 10).is_err());
        assert!(token_qmd(0x10_0000_0000, 0x10_0000_1000, 48).is_ok());
        assert!(token_qmd(0x10_0000_0000, 0x10_0000_1000, 0).is_err());
        assert!(TOKEN.len() >= 16 && TOKEN.len().is_multiple_of(16));
        // The register count is NAK's, from its dump.
        let dump = include_str!("../../../tools/nak/out/token.sm86.txt");
        assert!(dump.contains(&alloc::format!("Num GPRs: {TOKEN_REGISTERS}\n")));
    }
}
