//! Common vocabulary for every compute backend of Cyber Universe OS.
//!
//! This crate is hardware-neutral: nothing here knows about GA104 registers.
//! It defines
//! - the hardware access traits the kernel implements ([`hal`]),
//! - structured errors ([`error`]),
//! - device identity and capabilities ([`ids`], [`caps`]),
//! - the backend contract I1 ([`backend`]),
//! - the device lifecycle state machine ([`state`]),
//! - a bounded diagnostic journal ([`diag`]).
//!
//! The crate is `no_std` + `alloc` so it links into the kernel; host tests use std.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod backend;
pub mod caps;
pub mod diag;
pub mod error;
pub mod hal;
pub mod ids;
pub mod state;

pub use backend::{
    BufferDesc, BufferId, ComputeDevice, DeviceStatus, Fence, MemLocation, Submission,
};
pub use caps::{Capabilities, DTypeSet, Feature};
pub use error::{GpuError, GpuResult};
pub use hal::{Clock, ConfigSpace, Mmio};
pub use ids::{PciAddress, PciId};
pub use state::DeviceState;
