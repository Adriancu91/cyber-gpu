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
//! - [`fwsec`]: I3, second half — where FWSEC_PROD sits in the VBIOS (read-only,
//!   the ucode bytes are never read);
//! - [`firmware`]: inventory of the NVIDIA-signed firmware files the sequence
//!   needs, with their licence status;
//! - [`push`]: pushbuffer method headers and GPFIFO entries (clc56f.h, MIT) —
//!   encoding only, unverified on hardware;
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
pub mod fwsec;
pub mod gfw;
pub mod ids;
pub mod init;
pub mod probe;
pub mod push;
pub mod radix3;
pub mod regs;
pub mod vbios;

pub use backend::Ga104Backend;
