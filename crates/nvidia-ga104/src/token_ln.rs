//! C-157 p.2 (session 277, `docs/C-157_TOKEN32_V3_LN.md`): `token32_v3` — `token32_v2` with every `layernorm256` folded
//! into the quarter-block product that reads its output (os-495: the 25 layernorms of `token32_v2` cost ~0.46 ms per
//! token at G = 768, done by ONE workgroup each).
//!
//! The program has two operations more than `token32_v2` ([`OP_LN_FUSED32`], [`OP_LN_MATVEC32`]): every workgroup that
//! has rows computes the layernorm's mean and inverse deviation with exactly `layernorm256`'s additions, then the 24
//! normalized values its lane reads in every row of a quarter-block product, kept in registers; workgroup 0 also writes
//! the layernorm's output. [`work_list_ln`] builds the list: a `layernorm256` followed by the `q8_fused32` /
//! `q8_matvec32` that reads exactly its output becomes one entry (the product's, op 13 / 14, with the layernorm's
//! arguments copied at [`ENTRY_LN`]); anything else stays as in `kronos_token::work_list32`.
//!
//! What is proven where: in SIMULARE (`gpu-sim::cmdproc::token32v3_on_cpu`) the folded entry runs the layernorm's model
//! and then the product's — it shows the list and the arguments are right. That the SASS gives the bits of the
//! separate programs is DEDUS from the source and the listing; on the board: the same 16 tokens and fingerprint (L1).

use alloc::vec::Vec;

use gpu_core::{GpuError, GpuResult};

use crate::kronos_layer::{Launch, Prog};
use crate::kronos_token as kt;
use crate::occupancy::{self, Kernel, Reserve, SM86};
use crate::qmd::{ComputeQmd, MAX_SHARED_BYTES, SM_CONFIG_STEPS, SharedMemory};

fn bad(what: &'static str) -> GpuError {
    GpuError::InvalidArgument(what)
}

/// `token32_v3.comp` compiled by Mesa NAK 25.0.7 for SM 8.6 (`tools/nak/out/token32_v3.sm86.txt`).
pub const TOKEN32_V3: &[u8] = include_bytes!("../../../tools/nak/out/token32_v3.sm86.sass");
/// Registers per thread NAK reported for `token32_v3` (`Num GPRs`) — NOT those of `token32_v2`, so the QMD is
/// [`token32_v3_qmd_wide`].
pub const TOKEN32_V3_REGISTERS: u32 = 97;
/// SHA-256 of [`TOKEN32_V3`] (the OS checks it before sending).
pub const TOKEN32_V3_SHA256: &str =
    "8ecf4fed9ec987af4de86aeba024556f00ce1a7310842e46fa0e10b6e5c5b719";

/// Operation 13: a `layernorm256` folded into the `q8_fused32` that follows it.
pub const OP_LN_FUSED32: u32 = 13;
/// Operation 14: a `layernorm256` folded into the `q8_matvec32` that follows it (the logits).
pub const OP_LN_MATVEC32: u32 = 14;
/// Where the layernorm's arguments (its cbuf bytes 0x28..0x54: x, g, b, y, n, 1/n, eps) sit in a folded entry.
pub const ENTRY_LN: usize = 0xa0;
/// The layernorm's argument bytes copied (from its cbuf at 0x28).
pub const LN_ARGS: usize = 0x54 - 0x28;
/// The widest folded layernorm: 24 values per lane × 32 lanes.
pub const MAX_N: u32 = 768;

/// `token32_v3` for the occupancy arithmetic: 32 threads, [`TOKEN32_V3_REGISTERS`], the shared memory of `token32`.
pub const TOKEN32_V3_KERNEL: Kernel = Kernel {
    name: "token32_v3",
    threads: kt::TOKEN_BLOCK,
    registers: TOKEN32_V3_REGISTERS,
    shared_bytes: kt::TOKEN_SHARED,
};

fn u64_at(c: &[u8; 256], at: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&c[at..at + 8]);
    u64::from_le_bytes(b)
}

fn u32_at(c: &[u8; 256], at: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&c[at..at + 4]);
    u32::from_le_bytes(b)
}

/// The folded entry for `ln` followed by `mm`, or `None` when the pair cannot be folded: `ln` is not a
/// `layernorm256` on grid 1, `mm` is not a `q8_fused32` / `q8_matvec32` reading exactly `ln`'s output, n is not
/// 32 × the product's blocks or is above [`MAX_N`], or the product's bytes from [`ENTRY_LN`] are not zero.
pub fn fold(ln: &Launch, mm: &Launch) -> Option<[u8; kt::ENTRY]> {
    if ln.prog != Prog::LayerNorm256 || ln.grid != 1 {
        return None;
    }
    let (op, blocks_at) = match mm.prog {
        Prog::Q8Fused32 => (OP_LN_FUSED32, crate::q8_fused::CBUF_BLOCKS),
        Prog::Q8Matvec32 => (OP_LN_MATVEC32, crate::q8_matvec::CBUF_BLOCKS),
        _ => return None,
    };
    // both products read x at 0x30; the layernorm writes y at 0x40, its n at 0x48
    let (y, n) = (u64_at(&ln.cbuf, 0x40), u32_at(&ln.cbuf, 0x48));
    let blocks = u32_at(&mm.cbuf, blocks_at);
    if u64_at(&mm.cbuf, crate::q8_fused::CBUF_X) != y
        || n == 0
        || n > MAX_N
        || blocks.checked_mul(32) != Some(n)
        || mm.grid == 0
        || mm.cbuf[ENTRY_LN..].iter().any(|&b| b != 0)
        || ln.cbuf[..0x28].iter().any(|&b| b != 0)
        || mm.cbuf[..0x28].iter().any(|&b| b != 0)
    {
        return None;
    }
    let mut e = mm.cbuf;
    e[kt::ENTRY_OP..kt::ENTRY_OP + 4].copy_from_slice(&op.to_le_bytes());
    e[kt::ENTRY_GRID..kt::ENTRY_GRID + 4].copy_from_slice(&mm.grid.to_le_bytes());
    e[ENTRY_LN..ENTRY_LN + LN_ARGS].copy_from_slice(&ln.cbuf[0x28..0x28 + LN_ARGS]);
    Some(e)
}

/// The work list of `launches` for [`TOKEN32_V3`]: every foldable pair ([`fold`]) becomes one entry, every other
/// launch the entry of `kronos_token::work_list32`. Returns the list and its number of operations.
pub fn work_list_ln(launches: &[Launch]) -> GpuResult<(Vec<u8>, usize)> {
    if launches.is_empty() {
        return Err(bad("lista de lucru goala"));
    }
    let mut out = Vec::with_capacity(kt::ENTRY * launches.len());
    let mut n = 0;
    let mut i = 0;
    while i < launches.len() {
        if let Some(e) = launches.get(i + 1).and_then(|mm| fold(&launches[i], mm)) {
            out.extend_from_slice(&e);
            i += 2;
        } else {
            out.extend_from_slice(&kt::work_list32(&launches[i..i + 1])?);
            i += 1;
        }
        n += 1;
    }
    Ok((out, n))
}

/// The QMD of the one `token32_v3` launch: as `kronos_token::token32_qmd_wide` (the smallest SM shared-memory step at
/// which all `groups` are resident, with the 1 KiB reserve) but with [`TOKEN32_V3_REGISTERS`]. Refuses what cannot all
/// be resident at once (the barrier would never open).
pub fn token32_v3_qmd_wide(
    prog_va: u64,
    cbuf_va: u64,
    groups: u32,
    sms: u32,
) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
    if groups == 0 || sms == 0 {
        return Err(bad("token32_v3: 0 grupuri sau niciun SM"));
    }
    let step = SM_CONFIG_STEPS
        .iter()
        .copied()
        .find(|&step| {
            occupancy::blocks_per_sm(&SM86, &TOKEN32_V3_KERNEL, step, Reserve::OneKib)
                .is_some_and(|f| u64::from(f.blocks) * u64::from(sms) >= u64::from(groups))
        })
        .ok_or(bad(
            "token32_v3: mai multe grupuri decat incap deodata pe placa",
        ))?;
    ComputeQmd {
        program_va: prog_va,
        registers: TOKEN32_V3_REGISTERS,
        cbuf0_va: cbuf_va,
        cbuf0_size: 0x100,
        grid: [groups, 1, 1],
        block: [kt::TOKEN_BLOCK, 1, 1],
        release: None,
    }
    .to_bytes_with_sm_config(
        &SharedMemory {
            bytes: kt::TOKEN_SHARED,
            max_bytes: MAX_SHARED_BYTES,
            barriers: 0,
        },
        step,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ln(x: u64, y: u64, n: u32) -> Launch {
        let mut c = [0u8; 256];
        c[0x28..0x30].copy_from_slice(&x.to_le_bytes());
        c[0x30..0x38].copy_from_slice(&0x7000u64.to_le_bytes());
        c[0x38..0x40].copy_from_slice(&0x8000u64.to_le_bytes());
        c[0x40..0x48].copy_from_slice(&y.to_le_bytes());
        c[0x48..0x4c].copy_from_slice(&n.to_le_bytes());
        c[0x4c..0x50].copy_from_slice(&(1.0f32 / n as f32).to_bits().to_le_bytes());
        c[0x50..0x54].copy_from_slice(&1e-5f32.to_bits().to_le_bytes());
        Launch {
            prog: Prog::LayerNorm256,
            cbuf: c,
            grid: 1,
        }
    }

    fn mv(x: u64, blocks: u32, rows: u32) -> Launch {
        let mut c = [0u8; 256];
        c[0x28..0x30].copy_from_slice(&0x9000u64.to_le_bytes());
        c[0x30..0x38].copy_from_slice(&x.to_le_bytes());
        c[0x38..0x40].copy_from_slice(&0xa000u64.to_le_bytes());
        c[0x40..0x44].copy_from_slice(&rows.to_le_bytes());
        c[0x44..0x48].copy_from_slice(&blocks.to_le_bytes());
        Launch {
            prog: Prog::Q8Matvec32,
            cbuf: c,
            grid: rows,
        }
    }

    #[test]
    fn folds_only_what_reads_the_layernorm() {
        let (a, b) = (ln(0x1000, 0x2000, 768), mv(0x2000, 24, 50));
        let e = fold(&a, &b).unwrap();
        assert_eq!(u32_at(&e, kt::ENTRY_OP), OP_LN_MATVEC32);
        assert_eq!(u32_at(&e, kt::ENTRY_GRID), 50);
        assert_eq!(&e[0x28..0xa0], &b.cbuf[0x28..0xa0]);
        assert_eq!(&e[ENTRY_LN..ENTRY_LN + LN_ARGS], &a.cbuf[0x28..0x54]);
        assert!(e[ENTRY_LN + LN_ARGS..].iter().all(|&x| x == 0));
        // another x, a wider vector, blocks that do not match, a layernorm on 2 blocks: not folded
        assert!(fold(&a, &mv(0x2004, 24, 50)).is_none());
        assert!(fold(&ln(0x1000, 0x2000, 800), &mv(0x2000, 25, 50)).is_none());
        assert!(fold(&a, &mv(0x2000, 23, 50)).is_none());
        assert!(
            fold(
                &Launch {
                    grid: 2,
                    ..a.clone()
                },
                &b
            )
            .is_none()
        );
        assert!(fold(&b, &a).is_none());
        let mut dirty = b.clone();
        dirty.cbuf[0xf0] = 1;
        assert!(fold(&a, &dirty).is_none());
        // the list: [ln, mv, ln(other), mv] -> 3 operations
        let other = ln(0x1000, 0x3000, 768);
        let (list, n) = work_list_ln(&[a.clone(), b.clone(), other.clone(), b.clone()]).unwrap();
        assert_eq!(n, 3);
        assert_eq!(list.len(), 3 * kt::ENTRY);
        assert_eq!(&list[..kt::ENTRY], &e);
        assert_eq!(
            &list[kt::ENTRY + kt::ENTRY_OP..kt::ENTRY + kt::ENTRY_OP + 4],
            &[8, 0, 0, 0]
        );
        assert!(work_list_ln(&[]).is_err());
    }

    #[test]
    fn program_registers_and_qmd() {
        assert!(TOKEN32_V3.len() >= 16 && TOKEN32_V3.len().is_multiple_of(16));
        let dump = include_str!("../../../tools/nak/out/token32_v3.sm86.txt");
        assert!(dump.contains(&alloc::format!("Num GPRs: {TOKEN32_V3_REGISTERS}\n")));
        // 16 workgroups per SM must still fit (G = 768 on 48 SMs)
        assert!(token32_v3_qmd_wide(0x10_0000_0000, 0x10_0000_1000, 768, 48).is_ok());
        assert!(token32_v3_qmd_wide(0x10_0000_0000, 0x10_0000_1000, 769, 48).is_err());
        assert!(token32_v3_qmd_wide(0x10_0000_0000, 0x10_0000_1000, 0, 48).is_err());
    }
}
