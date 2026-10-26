//! I5–I8 — the data GSP-RM 570.144 expects from the driver, built and tested
//! **only in SIMULARE** (no access to the board anywhere in this module).
//!
//! What is here (each submodule states its sources and confidence):
//! - [`falcon`]: I6–I7 on registers (falcon reset/start, FWSEC-FRTS result,
//!   Booter on SEC2), behind a write policy; SIMULARE only;
//! - [`sequencer`]: GSP-RM's CPU sequencer requests and the wait for
//!   `GSP_INIT_DONE` (I8), through the same write policy; SIMULARE only;
//! - [`elf`]: finding a named section in the GSP-RM ELF container (like any
//!   loader; the section contents stay opaque);
//! - [`wpr_meta`]: the 256-byte `GspFwWprMeta` Booter reads (I7);
//! - [`libos`]: the LibOS memory-region arguments and `GSP_ARGUMENTS_CACHED`;
//! - [`msgq`]: the shared memory with the command/status queues (I5, I8);
//! - [`registry`]: the `SET_REGISTRY` table (`PACKED_REGISTRY_TABLE`);
//! - [`rpc`]: one queue element = element header + RPC header + payload, with
//!   the XOR checksum and sequence numbers (I8).
//!
//! The layouts follow NVIDIA's MIT headers from open-gpu-kernel-modules at tag
//! `570.144` (`gsp_fw_wpr_meta.h`, `msgq_priv.h`, `message_queue_priv.h`,
//! `g_rpc-message-header.h`, `rpc_headers.h`, `rpc_global_enums.h`,
//! `gsp_init_args.h`, `libos_init_args.h`) — read as interface descriptions,
//! re-written here (docs/PROVENANCE.md). They are firmware ABI: valid for
//! 570.144 only, which is what [`crate::firmware`] enforces.
//!
//! Anything that would touch the real board (Bus Master, falcon registers,
//! the doorbell) is *not* implemented; it needs a [`HardwareWritePermit`],
//! which only exists when the crate is built with the `hw-write` feature
//! (off by default) and is meant for the workstation window with Adrian.

pub mod elf;
pub mod falcon;
pub mod libos;
pub mod msgq;
pub mod registry;
pub mod rpc;
pub mod sequencer;
#[cfg(test)]
pub(crate) mod sim_peer;
pub mod wpr_meta;

use gpu_core::error::{GpuError, GpuResult};

/// Proof that the caller is in the approved hardware window. Without the
/// `hw-write` feature it cannot be created at all.
#[derive(Debug)]
pub struct HardwareWritePermit {
    pub reason: &'static str,
}

impl HardwareWritePermit {
    /// `reason` is journaled with every write made under the permit.
    pub fn new(reason: &'static str) -> GpuResult<Self> {
        if cfg!(feature = "hw-write") {
            Ok(HardwareWritePermit { reason })
        } else {
            Err(GpuError::NotPermitted(
                "scrierile in placa sunt oprite (build fara optiunea hw-write); doar in fereastra pe workstation, cu Adrian",
            ))
        }
    }
}

/// Little-endian helpers over byte buffers (all GSP structures are LE).
pub(crate) fn put_u32(b: &mut [u8], off: usize, v: u32) {
    b[off..off + 4].copy_from_slice(&v.to_le_bytes());
}
pub(crate) fn put_u64(b: &mut [u8], off: usize, v: u64) {
    b[off..off + 8].copy_from_slice(&v.to_le_bytes());
}
pub(crate) fn get_u16(b: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([b[off], b[off + 1]])
}
pub(crate) fn get_u32(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([b[off], b[off + 1], b[off + 2], b[off + 3]])
}
pub(crate) fn get_u64(b: &[u8], off: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[off..off + 8]);
    u64::from_le_bytes(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permit_only_with_the_feature() {
        let p = HardwareWritePermit::new("test");
        if cfg!(feature = "hw-write") {
            assert_eq!(p.unwrap().reason, "test");
        } else {
            assert!(matches!(p, Err(GpuError::NotPermitted(_))));
        }
    }
}
