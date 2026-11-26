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
use crate::occupancy::{self, Kernel, Reserve, SM86};
use crate::qmd::{ComputeQmd, MAX_SHARED_BYTES, SM_CONFIG_STEPS, SharedMemory};

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
        // C-147: `token.comp` has no such operation; `work_list` refuses it.
        Prog::Q8Fused => 7,
        // C-155: nor these two; `work_list` refuses them as well.
        Prog::LayerNorm256 => 8,
        Prog::Argmax => 9,
        // C-155 point 4: nor this one.
        Prog::Q8Fused32 => 10,
        Prog::Q8Matvec32 => 11,
        // C-157 point 7: nor this one.
        Prog::EmbedDev => 12,
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
        if l.prog == Prog::Q8Fused {
            return Err(bad("lista de lucru: token.comp nu are programul q8_fused"));
        }
        if matches!(
            l.prog,
            Prog::LayerNorm256 | Prog::Argmax | Prog::Q8Fused32 | Prog::Q8Matvec32 | Prog::EmbedDev
        ) {
            return Err(bad(
                "lista de lucru: token.comp nu are programele layernorm256, argmax, q8_fused32, q8_matvec32",
            ));
        }
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

/// C-157 (session 239, `docs/C-157_TOKEN32.md`): `token32.comp` — `token.comp` plus the programs of today's generation
/// list (`layernorm256`, `q8_fused`, `q8_fused32`, `q8_matvec32`, `argmax`, `embed_dev`), compiled by Mesa NAK 25.0.7
/// for SM 8.6 (`tools/nak/out/token32.sm86.txt`). The same list layout, constant buffer and QMD shape as [`TOKEN`].
pub const TOKEN32: &[u8] = include_bytes!("../../../tools/nak/out/token32.sm86.sass");
/// Registers per thread NAK reported for `token32` (`Num GPRs`).
pub const TOKEN32_REGISTERS: u32 = 92;

/// C-157 after os-471 (session 260, `docs/C-157_OS471.md`): `token32_v2.comp` — [`TOKEN32`] with the reads of what
/// earlier operations wrote (x of the products, q and K/V of the attention, residual, bias, LayerNorm inputs, logits,
/// the argmax slot) as `ld.global.constant` (through L1) instead of `ld.global.strong.sys`; the NAK IR is instruction
/// for instruction that of [`TOKEN32`] apart from those loads (same arithmetic, same order, 7 438 instructions,
/// 92 registers, the same shared memory), so the same list, constant buffer, QMDs ([`token32_qmd`],
/// [`token32_qmd_wide`]) and groups apply. Correct only because every operation is separated by the grid barrier,
/// whose `cctl` (IVAll) drops L1 — DEDUS from the listing; on the board: the same 16 tokens and fingerprint.
pub const TOKEN32_V2: &[u8] = include_bytes!("../../../tools/nak/out/token32_v2.sm86.sass");
/// Registers per thread NAK reported for `token32_v2` (`Num GPRs`) — the same as [`TOKEN32_REGISTERS`].
pub const TOKEN32_V2_REGISTERS: u32 = 92;
/// SHA-256 of [`TOKEN32_V2`] (the OS checks it before sending, as for `token32`).
pub const TOKEN32_V2_SHA256: &str =
    "5f4d22a0d01f35d2bd3b88f0e43e492acede1a1a8d912b80fc0712d012f11171";

/// The work list of `launches` for [`TOKEN32`]: every program of [`crate::kronos_layer::Prog`] is an operation of it
/// (the numbers of [`op_number`]). Refuses what [`work_list`] refuses apart from the program kind.
pub fn work_list32(launches: &[Launch]) -> GpuResult<Vec<u8>> {
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

/// C-157 after os-471 (session 260, `docs/C-157_OS471.md`): the 32-bit words where the work list `next` differs from
/// `prev` (byte offset, new word), in order. The OS wrote the whole list (88 × 256 B = 5 632 words at #513) before
/// every token; consecutive tokens change only a few words (the position, the token), so writing only these keeps
/// the driver's part near today's list. Both lists must have the same length (same operations), a multiple of 4.
pub fn work_list_changes(prev: &[u8], next: &[u8]) -> GpuResult<Vec<(usize, u32)>> {
    if prev.len() != next.len() || !next.len().is_multiple_of(4) || next.is_empty() {
        return Err(bad(
            "lista de lucru: listele au lungimi diferite (alte operatii) sau nemultiple de 4",
        ));
    }
    // Both lengths are multiples of 4 (checked above): `as_chunks` leaves no remainder.
    let (prev_words, _) = prev.as_chunks::<4>();
    let (next_words, _) = next.as_chunks::<4>();
    Ok(prev_words
        .iter()
        .zip(next_words)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (_, b))| (4 * i, u32::from_le_bytes(*b)))
        .collect())
}

/// The QMD of the one `token32` launch: `groups` workgroups of 32 threads (the shared memory of [`TOKEN`]).
pub fn token32_qmd(
    prog_va: u64,
    cbuf_va: u64,
    groups: u32,
) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    qmd_with(prog_va, cbuf_va, groups, TOKEN32_REGISTERS)
}

/// C-157 (session 240, `docs/C-157_TOKEN32_G_MARE.md`): `token32` as the occupancy arithmetic sees it — 32 threads,
/// [`TOKEN32_REGISTERS`], [`TOKEN_SHARED`] bytes of shared memory.
pub const TOKEN32_KERNEL: Kernel = Kernel {
    name: "token32",
    threads: TOKEN_BLOCK,
    registers: TOKEN32_REGISTERS,
    shared_bytes: TOKEN_SHARED,
};

/// The most `token32` workgroups one SM can keep at once (the 64 KiB step, either reading of the 1 KiB reserve):
/// 16, the SM's block limit. More than `16 × SMs` workgroups cannot all be resident, and the grid barrier would
/// wait forever for the ones not started — refused.
pub fn token32_max_per_sm() -> u32 {
    let last = SM_CONFIG_STEPS[SM_CONFIG_STEPS.len() - 1];
    occupancy::blocks_per_sm(&SM86, &TOKEN32_KERNEL, last, Reserve::OneKib).map_or(0, |f| f.blocks)
}

/// The smallest SM shared-memory step (8/16/32/64 KiB, [`SM_CONFIG_STEPS`]) at which `groups` workgroups of
/// `token32` are all resident on `sms` SMs — counted with the 1 KiB reserve per block (the stricter reading of
/// [`Reserve`]): 2 / 5 / 10 / 16 per SM. Refuses `groups` = 0, `sms` = 0 and `groups` > 16 × `sms`. That the board
/// keeps that many blocks per SM at that step is DEDUS (the occupancy probe and the barrier probe at G = 384 / 768
/// decide).
pub fn token32_step_for(groups: u32, sms: u32) -> GpuResult<u32> {
    if groups == 0 || sms == 0 {
        return Err(bad("programul pe simbol: 0 grupuri sau niciun SM"));
    }
    SM_CONFIG_STEPS
        .iter()
        .copied()
        .find(|&step| {
            occupancy::blocks_per_sm(&SM86, &TOKEN32_KERNEL, step, Reserve::OneKib)
                .is_some_and(|f| u64::from(f.blocks) * u64::from(sms) >= u64::from(groups))
        })
        .ok_or_else(|| {
            bad("programul pe simbol: mai multe grupuri decat incap deodata pe placa (16 pe SM)")
        })
}

/// `per_sm` `token32` workgroups on each SM GSP-RM reports (1..=16). Refuses anything that would not be resident.
///
/// **os-456: always refused now** (fails closed): the `GrShape` product is the design maximum (84 on the RTX 3070 Ti,
/// which has 48 SMs), so 16 per SM would ask for 1 344 workgroups — more than fit at once, a barrier that never opens.
/// Use [`token32_groups_for_sms`] with `rmapi::GrCounts::sms`.
pub fn token32_groups_for(_gr: &GrShape, _per_sm: u32) -> GpuResult<u32> {
    Err(bad(
        "programul pe simbol: numarul de SM din GrShape e maximul de proiect (os-456); foloseste token32_groups_for_sms",
    ))
}

/// C-157 after os-456: `per_sm` `token32` workgroups (1..=16) on the SMs GR0 really has (`rmapi::GrCounts::sms`).
pub fn token32_groups_for_sms(sms: u32, per_sm: u32) -> GpuResult<u32> {
    if sms == 0 {
        return Err(bad("programul pe simbol: niciun SM raportat"));
    }
    if per_sm == 0 || per_sm > token32_max_per_sm() {
        return Err(bad("programul pe simbol: intre 1 si 16 grupuri pe SM"));
    }
    Ok(sms * per_sm)
}

/// The QMD of the one `token32` launch with the SM step for many workgroups: the step of [`token32_step_for`]
/// (`MIN = TARGET =` step, `ComputeQmd::to_bytes_with_sm_config`), everything else as [`token32_qmd`]. At the 8 KiB
/// step (G ≤ 2 × SMs) the bytes are those of [`token32_qmd`]. Refuses G > 16 × SMs.
pub fn token32_qmd_wide(
    prog_va: u64,
    cbuf_va: u64,
    groups: u32,
    sms: u32,
) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    let step = token32_step_for(groups, sms)?;
    ComputeQmd {
        program_va: prog_va,
        registers: TOKEN32_REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: 0x100,
        grid: [groups, 1, 1],
        block: [TOKEN_BLOCK, 1, 1],
        release: None,
    }
    .to_bytes_with_sm_config(
        &SharedMemory {
            bytes: TOKEN_SHARED,
            max_bytes: MAX_SHARED_BYTES,
            barriers: 0,
        },
        step,
    )
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

/// NVK's root table: `cs.group_count[3]` at 0x0c / 0x10 / 0x14 (Mesa 25.0.7 `nvk_root_descriptor_table`).
/// `token.comp` reads `gl_NumWorkGroups.x` from 0x0c (`tools/nak/out/token.sm86.txt`, `c[0x0][0xc]`) for its block
/// stride and the barrier target (found in session 237 while compiling the barrier probe).
pub const CBUF_GROUP_COUNT: usize = 0x0c;

/// The program's constant buffer **with the workgroup count** at 0x0c — the one to use. [`token_cbuf`] (kept for
/// compatibility) leaves 0x0c at 0: on the board `vb += G` would then be `vb += 0` (a hang) and the barrier would
/// not wait; the SIMULARE model now refuses such a launch.
pub fn token_cbuf_for(list_va: u64, sync_va: u64, n: usize, groups: u32) -> GpuResult<[u8; 256]> {
    if groups == 0 {
        return Err(bad("programul pe simbol: 0 grupuri"));
    }
    let mut c = token_cbuf(list_va, sync_va, n)?;
    for (i, v) in [groups, 1, 1].into_iter().enumerate() {
        let at = CBUF_GROUP_COUNT + 4 * i;
        c[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }
    Ok(c)
}

/// C-157 after os-456: [`groups_for`] from the SMs GR0 really has (`rmapi::GrCounts::sms`).
pub fn groups_for_sms(sms: u32, max: u32) -> GpuResult<u32> {
    if sms == 0 || max == 0 {
        return Err(bad("programul pe simbol: niciun SM raportat"));
    }
    Ok(sms.min(max))
}

/// How many workgroups: one per SM (all resident: 32 threads and 2 176 B of shared memory each are far below an SM's
/// limits), capped by `max` (the widest operation needs no more than its grid; more only wait at the barriers).
///
/// **os-456:** the `GrShape` product is the design maximum (84 on the 3070 Ti, 48 SMs): up to 84 workgroups, still all
/// resident (16 per SM fit), so kept for compatibility; use [`groups_for_sms`].
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
    qmd_with(prog_va, cbuf_va, groups, TOKEN_REGISTERS)
}

fn qmd_with(
    prog_va: u64,
    cbuf_va: u64,
    groups: u32,
    registers: u32,
) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    if groups == 0 {
        return Err(bad("programul pe simbol: 0 grupuri"));
    }
    ComputeQmd {
        program_va: prog_va,
        registers,
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
        // token32: the same for its own dump; its QMD only differs in the register count.
        let dump = include_str!("../../../tools/nak/out/token32.sm86.txt");
        assert!(dump.contains(&alloc::format!("Num GPRs: {TOKEN32_REGISTERS}\n")));
        assert!(TOKEN32.len() >= 16 && TOKEN32.len().is_multiple_of(16));
        assert!(token32_qmd(0x10_0000_0000, 0x10_0000_1000, 0).is_err());
        let (a, b) = (
            token_qmd(0x10_0000_0000, 0x10_0000_1000, 48).unwrap(),
            token32_qmd(0x10_0000_0000, 0x10_0000_1000, 48).unwrap(),
        );
        assert_ne!(a, b);
        // Every program kind goes into the token32 list.
        for p in [
            Prog::LayerNorm256,
            Prog::Argmax,
            Prog::Q8Fused,
            Prog::Q8Fused32,
            Prog::Q8Matvec32,
            Prog::EmbedDev,
        ] {
            let l = Launch {
                prog: p,
                cbuf: [0u8; 256],
                grid: 1,
            };
            assert!(work_list(core::slice::from_ref(&l)).is_err());
            let w = work_list32(&[l]).unwrap();
            assert_eq!(
                u32::from_le_bytes(w[ENTRY_OP..ENTRY_OP + 4].try_into().unwrap()),
                op_number(p)
            );
        }
    }
}
