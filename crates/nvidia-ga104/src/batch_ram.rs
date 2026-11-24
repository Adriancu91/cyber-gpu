//! C-157 point 2 after os-390 (`docs/C-157_P2_LOT_PRIN_RAM.md`): the batch and the results of a **resident** training
//! step through a pinned RAM staging area, moved by the copy program (`transfer::COPY_B128`) inside the step's own
//! list, instead of word by word through the VRAM window.
//!
//! - [`via_ram`]: the staging bytes (what the CPU writes before the step), the copies RAM → arena (one per area of
//!   [`Plan::batch_uploads`]) and the copies arena → RAM (the areas of [`Plan::step_reads`]);
//! - [`pack_via_ram`]: the control zone with the copies in front of the step's launches and behind them (the same
//!   slot layout as [`crate::train_step::pack`]);
//! - [`results`]: the per-row losses and the gradient norm, read from the staging area after the wait.
//!
//! Only additions; nothing in [`crate::train_step`] changes. Tested on `gpu_sim` only (T209, SIMULARE).

use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::train_step::{Launch, Plan, Prog, SLOT, pack};
use crate::transfer::{COPY_UNIT, copy_cbuf, copy_qmd};

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// Where each area starts in the staging area (256-byte steps, as the arena).
const STAGE_ALIGN: usize = 256;

/// One copy the copy program does (GPU VAs, `bytes` a multiple of 16).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RamCopy {
    pub src: u64,
    pub dst: u64,
    pub bytes: u64,
}

/// What one step needs to move its batch in and its results out through RAM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ViaRam {
    /// The staging area as the CPU writes it before the step (batch areas, zero-padded; the result areas zero).
    pub staging: Vec<u8>,
    /// RAM → arena, one per area of [`Plan::batch_uploads`], in its order.
    pub copies_in: Vec<RamCopy>,
    /// Arena → RAM: the per-row losses, then `res` (the gradient norm).
    pub copies_out: Vec<RamCopy>,
    /// (offset in the staging area, bytes) of the per-row losses and of the norm, for [`results`].
    pub loss_at: (usize, usize),
    pub norm_at: (usize, usize),
}

fn round16(n: u64) -> u64 {
    n.div_ceil(COPY_UNIT) * COPY_UNIT
}

fn round_stage(n: usize) -> usize {
    n.div_ceil(STAGE_ALIGN) * STAGE_ALIGN
}

/// The staging bytes and copies for the batch (`inputs`, `targets`) of `plan` (a resident plan, after
/// [`Plan::rebatch`] for this batch), with the staging area at `staging_va` (a GPU VA of pinned RAM, 256-byte
/// aligned). Refuses everything [`Plan::batch_uploads`] refuses, a misaligned staging area, or an area of the arena
/// whose 16-byte rounding would not fit in its 256-byte allocation.
pub fn via_ram(plan: &Plan, inputs: &[u32], targets: &[u32], staging_va: u64) -> GpuResult<ViaRam> {
    if !staging_va.is_multiple_of(STAGE_ALIGN as u64) {
        return Err(bad("lot prin RAM: zona de tranzit nealiniata la 256"));
    }
    let ups = plan.batch_uploads(inputs, targets)?;
    let mut staging: Vec<u8> = Vec::new();
    let mut copies_in = Vec::with_capacity(ups.len());
    for (dst, bytes) in &ups {
        if !dst.is_multiple_of(STAGE_ALIGN as u64) {
            return Err(bad("lot prin RAM: zona arenei nealiniata"));
        }
        let at = staging.len();
        let n = round16(bytes.len() as u64);
        // The arena allocates every area in 256-byte blocks plus a 256-byte gap, so the rounded copy stays inside.
        if n > round_stage(bytes.len()) as u64 + STAGE_ALIGN as u64 {
            return Err(bad("lot prin RAM: rotunjirea iese din zona arenei"));
        }
        staging.extend_from_slice(bytes);
        staging.resize(at + round_stage(n as usize), 0);
        copies_in.push(RamCopy {
            src: staging_va + at as u64,
            dst: *dst,
            bytes: n,
        });
    }
    let reads = plan.step_reads();
    let mut copies_out = Vec::with_capacity(reads.len());
    let mut at_out = [(0usize, 0usize); 2];
    for (k, (src, len)) in reads.iter().enumerate() {
        if !src.is_multiple_of(STAGE_ALIGN as u64) {
            return Err(bad("lot prin RAM: rezultatul nealiniat"));
        }
        let at = staging.len();
        let n = round16(*len);
        staging.resize(at + round_stage(n as usize), 0);
        copies_out.push(RamCopy {
            src: *src,
            dst: staging_va + at as u64,
            bytes: n,
        });
        at_out[k] = (at, *len as usize);
    }
    Ok(ViaRam {
        staging,
        copies_in,
        copies_out,
        loss_at: at_out[0],
        norm_at: at_out[1],
    })
}

/// How many launches [`pack_via_ram`] adds to a step's list.
pub fn extra_launches(v: &ViaRam) -> usize {
    v.copies_in.len() + v.copies_out.len()
}

/// The control zone (bytes at `zone_va`) and the QMD addresses, in launch order: the copies in, the step's `list`,
/// the copies out. The copy program is at `copy_va`; every slot is [`SLOT`] bytes (cbuf at +0, QMD at +0x100), as
/// [`crate::train_step::pack`] lays out the step's own launches.
pub fn pack_via_ram(
    list: &[Launch],
    prog_va: impl Fn(Prog) -> u64,
    copy_va: u64,
    zone_va: u64,
    v: &ViaRam,
) -> GpuResult<(Vec<u8>, Vec<u64>)> {
    if !copy_va.is_multiple_of(256) || !zone_va.is_multiple_of(256) {
        return Err(bad(
            "lot prin RAM: programul de copiere sau zona nealiniate",
        ));
    }
    let n_in = v.copies_in.len();
    let list_va = zone_va + SLOT * n_in as u64;
    let (list_bytes, list_qmds) = pack(list, prog_va, list_va)?;
    let mut bytes = Vec::with_capacity((extra_launches(v) + list.len()) * SLOT as usize);
    let mut qmds = Vec::with_capacity(extra_launches(v) + list.len());
    let put_copy = |bytes: &mut Vec<u8>, qmds: &mut Vec<u64>, c: &RamCopy| -> GpuResult<()> {
        let at = bytes.len();
        let cb_va = zone_va + at as u64;
        let cb = copy_cbuf(c.src, c.dst, c.bytes)?;
        let q = copy_qmd(copy_va, cb_va, c.bytes)?.to_bytes()?;
        bytes.resize(at + SLOT as usize, 0);
        bytes[at..at + cb.len()].copy_from_slice(&cb);
        bytes[at + 0x100..at + 0x100 + q.len()].copy_from_slice(&q);
        qmds.push(cb_va + 0x100);
        Ok(())
    };
    for c in &v.copies_in {
        put_copy(&mut bytes, &mut qmds, c)?;
    }
    bytes.extend_from_slice(&list_bytes);
    qmds.extend_from_slice(&list_qmds);
    for c in &v.copies_out {
        put_copy(&mut bytes, &mut qmds, c)?;
    }
    Ok((bytes, qmds))
}

/// The per-row losses and the gradient norm from the staging area read back after the wait (`staging` = the
/// area's bytes, at least as long as [`ViaRam::staging`]).
pub fn results(staging: &[u8], v: &ViaRam) -> GpuResult<(Vec<f32>, f32)> {
    let f = |(at, len): (usize, usize)| -> GpuResult<Vec<f32>> {
        let b = staging
            .get(at..at + len)
            .ok_or(bad("lot prin RAM: zona de tranzit prea scurta"))?;
        Ok(b.chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    };
    let rows = f(v.loss_at)?;
    let norm = *f(v.norm_at)?
        .first()
        .ok_or(bad("lot prin RAM: norma lipseste"))?;
    Ok((rows, norm))
}
