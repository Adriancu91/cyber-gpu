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
//! - [`i6_check`]: what I6 (FWSEC-FRTS) needs, read without writing (fuse,
//!   VRAM size, WPR2 down, FRTS region) — plan docs/FEREASTRA_C-49.md;
//! - [`i6_log`]: reads the I6 journal back from the stick and checks the
//!   write order (C-51: our journal replaces the cancelled tinygrad run);
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
//! - [`read_tour`]: C-54 — the read-only tour for the automatic loop (facts
//!   `cheie = valoare`, 0 writes, refused if a write is journaled);
//! - [`board_compare`]: C-54 — facts from a board report vs the same tour on
//!   the `gpu-sim` model (pure text);
//! - [`chain`]: C-54 — the window steps in a chain (stop at the first failure,
//!   cleanup always, no writing step without permission, blocked steps never tried);
//! - [`chain_log`]: C-71 — the stick journal of a whole `gpu lant` window,
//!   checked step by step (pure text);
//! - [`launch_cost`]: C-69 — measurement 1 of the one-program target: the
//!   cost of one launch from GPU timestamps (encoding + reading, SIMULARE);
//! - [`backend`]: the I1 contract for GA104.
//!
//! Sources and licences: docs/GA104_INITIALIZATION.md, docs/PROVENANCE.md.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod backend;
pub mod board_compare;
pub mod boot;
pub mod chain;
pub mod chain_log;
pub mod fb_layout;
pub mod firmware;
pub mod fuse;
pub mod fw_check;
pub mod fw_hdr;
pub mod fwsec;
pub mod gfw;
pub mod gsp;
pub mod i6_check;
pub mod i6_log;
pub mod ids;
pub mod init;
pub mod kronos_gen;
pub mod kronos_layer;
pub mod kronos_model;
pub mod kronos_token;
pub mod launch_cost;
pub mod mmu;
pub mod probe;
pub mod push;
pub mod q8_matvec;
pub mod qmd;
pub mod radix3;
pub mod read_tour;
pub mod recovery;
pub mod regs;
pub mod sha256;
pub mod transfer;
pub mod variant_log;
pub mod vbios;
pub mod vbios_diag;
pub mod vram_window;

pub use backend::Ga104Backend;
