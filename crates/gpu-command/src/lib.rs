//! Hardware-neutral command submission primitives: ring indices, fence
//! timelines and bounded waits. Vendor encodings (NVIDIA pushbuffer methods,
//! GPFIFO entries) live in the vendor crate.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

pub mod fence;
pub mod ring;

pub use fence::FenceTimeline;
pub use ring::Ring;
