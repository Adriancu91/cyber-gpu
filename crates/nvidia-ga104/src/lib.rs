//! NVIDIA GA104 backend.
//!
//! What is implemented and safe today:
//! - [`ids`]: PCI device id table (pci.ids, cross-checked with NVIDIA's README);
//! - [`boot`]: decoding of NV_PMC_BOOT_0 / NV_PMC_BOOT_42 (open-gpu-doc, MIT);
//! - [`regs`]: the register allow-list and a read-only BAR0 accessor — there
//!   is no write path to BAR0 in this crate;
//! - [`probe`]: level-0 identification (config space + two BAR0 reads);
//! - [`init`]: the GA10x bring-up sequence as an explicit plan whose
//!   unimplemented steps return `Blocked` with the exact blocker;
//! - [`fb_layout`]: where GSP-RM's regions go in VRAM (pure arithmetic, no access);
//! - [`vbios_diag`]: I3 diagnostic — the VBIOS chain structure listed line by
//!   line (offsets, signatures, PCIR/NPDE fields, breaks), read-only (C-30);
//! - [`fwsec`]: I3, second half — where FWSEC_PROD sits in the VBIOS (read-only,
//!   the ucode bytes are never read);
//! - [`fuse`]: the ucode fuse versions (read-only) that pick the FWSEC/Booter
//!   signature the boot ROM accepts;
//! - [`firmware`]: inventory of the NVIDIA-signed firmware files the sequence
//!   needs; I4: the pinned 570.144 GSP set (size + SHA-256) read from the stick;
//! - [`gsp`]: I5–I8 data structures for GSP-RM 570.144 (WPR meta, LibOS
//!   arguments, message queues, RPC elements) — SIMULARE only, no board access;
//! - [`sha256`]: FIPS 180-4 digest for the I4 check;
//! - [`push`]: pushbuffer method headers and GPFIFO entries (clc56f.h, MIT) —
//!   encoding only, unverified on hardware;
//! - [`mmu`]: I9a — VRAM/GPU-VA allocation and the MMU v2 page tables, written
//!   through a `VramWrite` sink — SIMULARE only, no board access;
//! - [`vram_window`]: I9b — that sink on the board: VRAM through the BAR0
//!   window + the MMU TLB invalidate; four registers, journaled, behind
//!   `hw-write` on a real BAR — SIMULARE only;
//! - [`radix3`]: the page list GSP reads to find its image in RAM (pure data);
//! - [`backend`]: the I1 contract for GA104.
//!
//! Sources and licences: docs/GA104_INITIALIZATION.md, docs/PROVENANCE.md.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod backend;
pub mod boot;
pub mod fb_layout;
pub mod firmware;
pub mod fuse;
pub mod fw_check;
pub mod fw_hdr;
pub mod fwsec;
pub mod gfw;
pub mod gsp;
pub mod ids;
pub mod init;
pub mod mmu;
pub mod probe;
pub mod push;
pub mod qmd;
pub mod radix3;
pub mod regs;
pub mod sha256;
pub mod vbios;
pub mod vbios_diag;
pub mod vram_window;

pub use backend::Ga104Backend;
