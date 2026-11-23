//! C-157 (`docs/C-157_ADAM_RAM.md`, step 2 of `docs/C-157_RAM_OFFLOAD.md`) — **AdamW's m and v in host RAM, brought
//! to the board layer by layer** by the copy engine, so a `_mv_ram` training plan
//! ([`Plan::new_tiles_mv_ram`] / [`Plan::new_p_only_tiles_mv_ram`]) keeps 8 bytes per parameter less on the board.
//!
//! - [`chunks`]: the pieces of the flat buffers, at tensor borders: the two tables, then the 12 tensors of every
//!   layer, then the final LayerNorm;
//! - [`RamMv`]: where m and v live in RAM (the OS's DMA memory, mapped into our GPU VAs), one staging place on the
//!   board as big as the largest piece (m then v), and the copy engine's records;
//! - [`RamMv::opt_launches`]: the optimizer step of [`Plan::step_until_opt`] split into one `opt_step` launch per
//!   piece, on the staging place. `opt_step` works element by element, so the bits are the same as one launch over
//!   everything (T193);
//! - [`encode_step`]: the whole step in one submission — the launches before the optimizer exactly as
//!   `transfer::encode_sequence_fresh` sends them, then per piece: copy m, v up (the copy engine's record with
//!   `FLUSH`), the channel **waits** for that record (`SEM_EXECUTE` `ACQ_STRICT_GEQ`, 32 bits), the launch and its
//!   timestamp record with WFI (as every launch today), copy m, v down (a record); at the end the channel waits for
//!   the last record, so when the submission is done m, v are back in RAM.
//!
//! The ordering copy engine → compute rests on the wait (a method the simulated board checks); compute → copy
//! engine rests on the WFI of the timestamp release before the copies down — that it also covers the copy engine
//! reading what the program wrote is **DEDUS** (same L2; tinygrad and NVK also put a barrier between the queues).
//! Method numbers: NVIDIA's `clc56f.h` / `clc7b5.h` (MIT), generated into `nvidia_rm570::regs`.
//!
//! Tested **only in SIMULARE** (T193, `tests/sim_c157_adam_ram.rs`); nothing here has run on a board.

use alloc::vec::Vec;

use gpu_compute::train::PER_LAYER;
use gpu_core::error::{GpuError, GpuResult};
use nvidia_rm570::regs::clc56f as h;

use crate::gsp::work_queue::{invalidate_caches, launch};
use crate::launch_cost::{RECORD, timestamp_release};
use crate::opt_step as os;
use crate::push::PushBuf;
use crate::ram_vram::{BIND_WORDS, copy_words, encode_bind, encode_copy};
use crate::train_step::{Launch, Plan, Prog};
use crate::transfer;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The tag of one per-piece optimizer launch.
pub const TAG: &str = "opt:pas-bucata";

/// One piece of the flat buffers: tensors `first .. first + tensors`, elements `off .. off + n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub first: usize,
    pub tensors: usize,
    pub off: usize,
    pub n: usize,
    /// The layer, for the 12 tensors of a layer; `None` for the tables and the final LayerNorm.
    pub layer: Option<usize>,
}

/// The pieces of `plan`'s flat buffers: tables, every layer, the final LayerNorm (L + 2). Every piece starts and
/// ends on 16 bytes (a copy-engine line); refused otherwise.
pub fn chunks(plan: &Plan) -> GpuResult<Vec<Chunk>> {
    let count = plan.off.len();
    let layers = plan.arch.layers;
    if count != 2 + PER_LAYER * layers + 2 {
        return Err(bad("adam-ram: numarul tensorilor"));
    }
    let mut out = Vec::with_capacity(layers + 2);
    let mut push = |first: usize, tensors: usize, layer: Option<usize>| {
        let off = plan.off[first];
        let end = plan
            .off
            .get(first + tensors)
            .copied()
            .unwrap_or(plan.n_params);
        out.push(Chunk {
            first,
            tensors,
            off,
            n: end - off,
            layer,
        });
    };
    push(0, 2, None);
    for l in 0..layers {
        push(2 + PER_LAYER * l, PER_LAYER, Some(l));
    }
    push(2 + PER_LAYER * layers, 2, None);
    if out
        .iter()
        .any(|c| c.n == 0 || !c.off.is_multiple_of(4) || !c.n.is_multiple_of(4))
    {
        return Err(bad("adam-ram: o bucata nu e pe 16 octeti"));
    }
    Ok(out)
}

/// Where m and v live and pass. `m`, `v`: host RAM, `4·n_params` bytes each; `stage_m`, `stage_v`: board memory,
/// `4·max piece` each; `sync`: the copy engine's records, [`RamMv::sync_bytes`] (16 per copy up or down).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamMv {
    pub m: u64,
    pub v: u64,
    pub stage_m: u64,
    pub stage_v: u64,
    pub sync: u64,
    pub chunks: Vec<Chunk>,
    pub n_params: usize,
}

fn round(bytes: u64, to: u64) -> u64 {
    bytes.div_ceil(to) * to
}

impl RamMv {
    /// For a `_mv_ram` plan: m at `ram`, v right after it (4 KiB aligned); the staging place at `stage` (m, then v,
    /// 256-byte aligned); the records at `sync` (16-aligned). `ram`, `stage` 4 KiB aligned.
    pub fn new(plan: &Plan, ram: u64, stage: u64, sync: u64) -> GpuResult<Self> {
        if !plan.mv_in_ram() {
            return Err(bad("adam-ram: planul tine m, v pe placa"));
        }
        if !ram.is_multiple_of(0x1000) || !stage.is_multiple_of(0x1000) {
            return Err(bad(
                "adam-ram: RAM sau locul de pe placa nealiniat la 4 KiB",
            ));
        }
        if !sync.is_multiple_of(RECORD) {
            return Err(bad("adam-ram: inregistrarile nealiniate la 16"));
        }
        let chunks = chunks(plan)?;
        let max = chunks.iter().map(|c| c.n).max().unwrap_or(0) as u64;
        Ok(RamMv {
            m: ram,
            v: ram + round(4 * plan.n_params as u64, 0x1000),
            stage_m: stage,
            stage_v: stage + round(4 * max, 256),
            sync,
            chunks,
            n_params: plan.n_params,
        })
    }

    /// Bytes of host RAM for m and v (from `ram`).
    pub fn ram_bytes(&self) -> u64 {
        self.v - self.m + 4 * self.n_params as u64
    }

    /// Bytes of board memory for the staging place (from `stage`).
    pub fn stage_bytes(&self) -> u64 {
        2 * (self.stage_v - self.stage_m)
    }

    /// Bytes of the copy engine's records: two per piece.
    pub fn sync_bytes(&self) -> u64 {
        2 * self.chunks.len() as u64 * RECORD
    }

    /// What the CPU writes once before step 1: zeros for m and v in RAM, zeros for the records.
    pub fn uploads(&self) -> Vec<(u64, Vec<u8>)> {
        let z = alloc::vec![0u8; 4 * self.n_params];
        alloc::vec![
            (self.m, z.clone()),
            (self.v, z),
            (self.sync, alloc::vec![0u8; self.sync_bytes() as usize]),
        ]
    }

    /// The optimizer step `a` ([`Plan::step_until_opt`]) as one `opt_step` launch per piece: g, w = the piece's
    /// slice, m, v = the staging place, n = the piece; the scalars and the clipping factor unchanged.
    pub fn opt_launches(&self, a: &os::Args) -> GpuResult<Vec<Launch>> {
        if a.mode != os::Mode::AdamW || a.n as usize != self.n_params {
            return Err(bad("adam-ram: pasul nu e AdamW pe tot modelul"));
        }
        self.chunks
            .iter()
            .map(|c| {
                let x = os::Args {
                    g: a.g + 4 * c.off as u64,
                    w: a.w + 4 * c.off as u64,
                    m: self.stage_m,
                    v: self.stage_v,
                    n: c.n as u32,
                    ..*a
                };
                Launch::new(
                    Prog::OptStep,
                    x.cbuf().as_ref().map(|b| &b[..]),
                    x.grid(),
                    TAG,
                    c.layer,
                )
            })
            .collect()
    }

    /// Record of piece `k`'s copy up (`down = false`) or down.
    pub fn record(&self, k: usize, down: bool) -> u64 {
        self.sync + (2 * k as u64 + down as u64) * RECORD
    }
}

/// The channel waits until the 32-bit word at `va` is ≥ `payload` (`SEM_EXECUTE` `ACQ_STRICT_GEQ`, subchannel 0).
pub fn acquire_geq(p: &mut PushBuf<'_>, va: u64, payload: u32) -> GpuResult<()> {
    if !va.is_multiple_of(4) || va >> 57 != 0 {
        return Err(bad("adam-ram: semafor nealiniat la 4 sau prea mare"));
    }
    let exec =
        h::NVC56F_SEM_EXECUTE_OPERATION.set(0, h::NVC56F_SEM_EXECUTE_OPERATION_ACQ_STRICT_GEQ);
    p.inc(
        0,
        h::NVC56F_SEM_ADDR_LO,
        &[va as u32, (va >> 32) as u32, payload, 0, exec],
    )
}

/// Words [`acquire_geq`] writes.
pub const ACQUIRE_WORDS: usize = 6;

/// Words [`encode_step`] writes for `before` launches and `pieces` pieces.
pub const fn step_words(before: usize, pieces: usize) -> usize {
    transfer::sequence_words_fresh(before)
        + BIND_WORDS
        + pieces * (2 * copy_words(false) + 2 * copy_words(true) + ACQUIRE_WORDS + 2 + 4 + 6)
        + ACQUIRE_WORDS
}

/// The payload of the copy engine's record of piece `k`, up or down.
pub fn sync_payload(first_sync: u32, k: usize, down: bool) -> u32 {
    first_sync.wrapping_add(1 + 2 * k as u32 + down as u32)
}

/// One step in one submission: `before` (the QMDs of [`Plan::step_until_opt`]'s launches) exactly as
/// `transfer::encode_sequence_fresh` sends them (records 0..=n), then piece `k` of `rm` with `pieces[k]` (the QMD of
/// [`RamMv::opt_launches`]' launch `k`): m, v up, wait, launch, record n + 1 + k (as if it were the next launch of
/// the sequence: `transfer::analyze_sequence` reads all n + pieces durations), m, v down; the channel waits for the
/// last copy down. Copy-engine payloads: [`sync_payload`].
pub fn encode_step(
    p: &mut PushBuf<'_>,
    rm: &RamMv,
    before: &[u64],
    pieces: &[u64],
    records_va: u64,
    first_payload: u32,
    first_sync: u32,
) -> GpuResult<()> {
    if pieces.len() != rm.chunks.len() {
        return Err(bad("adam-ram: cate o lansare pe bucata"));
    }
    transfer::encode_sequence_fresh(p, before, records_va, first_payload)?;
    encode_bind(p)?;
    let n = before.len();
    for (k, (c, &q)) in rm.chunks.iter().zip(pieces).enumerate() {
        let (off, bytes) = (4 * c.off as u64, 4 * c.n as u64);
        let up = (rm.record(k, false), sync_payload(first_sync, k, false));
        let down = (rm.record(k, true), sync_payload(first_sync, k, true));
        encode_copy(p, rm.m + off, rm.stage_m, bytes, false, None)?;
        encode_copy(p, rm.v + off, rm.stage_v, bytes, false, Some(up))?;
        acquire_geq(p, up.0, up.1)?;
        invalidate_caches(p)?;
        launch(p, q)?;
        timestamp_release(
            p,
            records_va + (n + 1 + k) as u64 * RECORD,
            first_payload.wrapping_add((n + 1 + k) as u32),
        )?;
        encode_copy(p, rm.stage_m, rm.m + off, bytes, false, None)?;
        encode_copy(p, rm.stage_v, rm.v + off, bytes, false, Some(down))?;
    }
    let last = rm.chunks.len() - 1;
    acquire_geq(
        p,
        rm.record(last, true),
        sync_payload(first_sync, last, true),
    )
}

/// The copy engine's records after a step: every payload as [`sync_payload`] says. A wrong one names the first
/// piece whose copy did not finish (`Err`).
pub fn check_sync(records: &[u8], pieces: usize, first_sync: u32) -> GpuResult<()> {
    if records.len() < 2 * pieces * RECORD as usize {
        return Err(bad("adam-ram: prea putine inregistrari"));
    }
    for k in 0..pieces {
        for down in [false, true] {
            let at = (2 * k + down as usize) * RECORD as usize;
            let got = u32::from_le_bytes([
                records[at],
                records[at + 1],
                records[at + 2],
                records[at + 3],
            ]);
            if got != sync_payload(first_sync, k, down) {
                return Err(bad(if down {
                    "adam-ram: o copiere inapoi in RAM nu s-a terminat"
                } else {
                    "adam-ram: o copiere spre placa nu s-a terminat"
                }));
            }
        }
    }
    Ok(())
}
