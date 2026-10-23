//! # SIMULARE — nu este hardware
//!
//! Software models used by host-side tests, following the pattern of
//! `cyber-universe/rtl8168/src/model.rs`. They encode *our reading* of public
//! specifications; a test passing against them proves the driver logic is
//! consistent with that reading, never that the RTX 3070 Ti behaves this way.
//! Register values that are not from a public source are marked `SIM`.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod clock;
pub mod mmio;
pub mod pci;
pub mod workstation;
