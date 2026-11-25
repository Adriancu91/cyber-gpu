//! C-165 (`docs/C-165_OCUPAREA.md`): how many blocks of a program fit on one SM of the GA104 at a given
//! shared-memory step, and the occupancy probe for the board. Pure arithmetic; nothing is sent.
//!
//! The limits are the public table for compute capability 8.6 (CUDA programming guide; ideas only): 1 536 threads,
//! 48 warps, 16 blocks, 65 536 registers per SM, registers given per warp in units of 256. Whether the board really
//! keeps more blocks per SM when the QMD asks for a larger step is **DEDUS** — the probe ([`PROBE`]) decides.
//! The 1 KiB per block that the CUDA driver reserves on 8.x may or may not exist for our launches: both cases are
//! computed ([`Reserve`]).

use crate::mm_tc;
use crate::mm_train_variants::VARIANTS;
use crate::qmd::SM_CONFIG_STEPS;

/// Per-SM limits of compute capability 8.6 (public table).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SmLimits {
    pub threads: u32,
    pub warps: u32,
    pub blocks: u32,
    pub registers: u32,
    /// Registers are allocated per warp in units of this many (256 = 8 per thread).
    pub reg_unit_per_warp: u32,
    pub max_regs_per_thread: u32,
}

/// The RTX 3070 Ti's SM (GA104, capability 8.6).
pub const SM86: SmLimits = SmLimits {
    threads: 1536,
    warps: 48,
    blocks: 16,
    registers: 65536,
    reg_unit_per_warp: 256,
    max_regs_per_thread: 255,
};

/// The two readings of the CUDA guide's per-block reserve (DEDUS which one the board follows).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reserve {
    /// R0: a block takes exactly its shared memory (Mesa NVK's view).
    None,
    /// R1: a block takes its shared memory + 1 KiB.
    OneKib,
}

impl Reserve {
    pub const fn bytes(self) -> u32 {
        match self {
            Reserve::None => 0,
            Reserve::OneKib => 1024,
        }
    }
}

/// What stops more blocks from fitting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Bound {
    Threads,
    Blocks,
    Registers,
    Shared,
}

/// One program as the occupancy arithmetic sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Kernel {
    pub name: &'static str,
    pub threads: u32,
    pub registers: u32,
    pub shared_bytes: u32,
}

/// Blocks per SM and the limit that sets it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fit {
    pub blocks: u32,
    pub bound: Bound,
}

/// Blocks per SM without the shared-memory limit (threads, warps, blocks, registers). `None` for a program the SM
/// cannot run at all (0 threads, too many registers).
pub fn blocks_without_shared(lim: &SmLimits, k: &Kernel) -> Option<Fit> {
    if k.threads == 0 || k.registers == 0 || k.registers > lim.max_regs_per_thread {
        return None;
    }
    let warps = k.threads.div_ceil(32);
    let regs_per_warp = (k.registers * 32).next_multiple_of(lim.reg_unit_per_warp);
    let regs_per_block = regs_per_warp * warps;
    let by_threads = (lim.threads / k.threads).min(lim.warps / warps);
    let by_regs = lim.registers / regs_per_block;
    let mut fit = Fit {
        blocks: lim.blocks,
        bound: Bound::Blocks,
    };
    if by_threads < fit.blocks {
        fit = Fit {
            blocks: by_threads,
            bound: Bound::Threads,
        };
    }
    if by_regs < fit.blocks {
        fit = Fit {
            blocks: by_regs,
            bound: Bound::Registers,
        };
    }
    Some(fit)
}

/// Blocks per SM at the shared-memory step `step_bytes` (8/16/32/64/96 KiB of the SM's L1 given to shared memory).
/// `blocks == 0` = one block does not fit in that step.
pub fn blocks_per_sm(lim: &SmLimits, k: &Kernel, step_bytes: u32, r: Reserve) -> Option<Fit> {
    let other = blocks_without_shared(lim, k)?;
    let per_block = k.shared_bytes.next_multiple_of(128) + r.bytes();
    if per_block == 0 {
        return Some(other);
    }
    let by_shared = step_bytes / per_block;
    // Shared memory wins a tie: it is the knob the probe turns.
    if by_shared <= other.blocks {
        Some(Fit {
            blocks: by_shared,
            bound: Bound::Shared,
        })
    } else {
        Some(other)
    }
}

/// The smallest step of [`SM_CONFIG_STEPS`] at which **every** program of `ks` reaches its limit without shared
/// memory; if none does, the largest step (64 KiB = our `MAX`). One step for the whole list: changing the step between
/// launches most likely needs an empty SM (DEDUS), to be avoided between the launches of a training step.
pub fn best_step(lim: &SmLimits, ks: &[Kernel], r: Reserve) -> u32 {
    let last = SM_CONFIG_STEPS[SM_CONFIG_STEPS.len() - 1];
    SM_CONFIG_STEPS
        .iter()
        .copied()
        .find(|&step| {
            ks.iter().all(|k| {
                match (
                    blocks_without_shared(lim, k),
                    blocks_per_sm(lim, k, step, r),
                ) {
                    (Some(o), Some(f)) => f.blocks >= o.blocks,
                    _ => true,
                }
            })
        })
        .unwrap_or(last)
}

/// The race variant `i` of [`VARIANTS`] as a [`Kernel`].
pub fn variant_kernel(i: usize) -> Kernel {
    let v = &VARIANTS[i];
    Kernel {
        name: v.name,
        threads: v.block()[0],
        registers: v.registers,
        shared_bytes: v.shared().bytes,
    }
}

/// C-165 (`docs/C-165_DIETA_128.md`): the register diet of `mm_tv_128x128`
/// ([`crate::mm_train_variants::MM_TV_128X128T512_VARIANT`], 512 threads, 54 registers, 8 KiB) as a [`Kernel`].
pub fn diet_128_kernel() -> Kernel {
    let v = &crate::mm_train_variants::MM_TV_128X128T512_VARIANT;
    Kernel {
        name: v.name,
        threads: v.block()[0],
        registers: v.registers,
        shared_bytes: v.shared().bytes,
    }
}

/// `mm_tc` (128 threads, 64 registers, 24 KiB) as a [`Kernel`].
pub fn mm_tc_kernel() -> Kernel {
    Kernel {
        name: "mm_tc",
        threads: mm_tc::BLOCK[0] * mm_tc::BLOCK[1] * mm_tc::BLOCK[2],
        registers: mm_tc::REGISTERS,
        shared_bytes: mm_tc::SHARED.bytes,
    }
}

/// One launch of the occupancy probe: variant index in [`VARIANTS`] and the step in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProbeRun {
    pub variant: usize,
    pub step_bytes: u32,
}

const fn run(variant: usize, kib: u32) -> ProbeRun {
    ProbeRun {
        variant,
        step_bytes: kib << 10,
    }
}

/// The occupancy probe on the qkv shape of the #513 step (`docs/C-165_OCUPAREA.md` §3), in order: the witness
/// `mm_train_tile` at 8 → 16 → 32 → 64 KiB, then `mm_tv_64x64k8` at 8 → 16 → 32, then `mm_tv_128x128` at 8 → 16 → 32.
/// The first run is today's launch (the step of one block): its bits are the probe's reference.
pub const PROBE: [ProbeRun; 10] = [
    run(0, 8),
    run(0, 16),
    run(0, 32),
    run(0, 64),
    run(4, 8),
    run(4, 16),
    run(4, 32),
    run(3, 8),
    run(3, 16),
    run(3, 32),
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn witness_today_is_one_block() {
        let f = blocks_per_sm(&SM86, &variant_kernel(0), 8 << 10, Reserve::None).unwrap();
        assert_eq!(
            f,
            Fit {
                blocks: 1,
                bound: Bound::Shared
            }
        );
    }
}
