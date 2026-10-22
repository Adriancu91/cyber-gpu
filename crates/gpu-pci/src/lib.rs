//! PCI/PCIe discovery for compute devices.
//!
//! Source: PCI Local Bus Specification 3.0 (header type 0, BAR encoding,
//! capability list) and PCI Express Base Specification (PCIe capability,
//! extended capabilities, Resizable BAR). Original code, no foreign code.
//!
//! Two levels of access:
//! - **read-only** ([`header`], [`bar::decode_bars`], [`caps`], [`report`]):
//!   safe on any device, including the primary display GPU;
//! - **BAR sizing** ([`bar::size_bars`]): writes to configuration space with
//!   memory decode disabled. It requires an explicit [`bar::SizingPermit`]
//!   and restores and verifies every register it touches.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod bar;
pub mod caps;
pub mod header;
pub mod report;

pub use bar::{Bar, BarKind, SizingPermit};
pub use report::DeviceReport;
