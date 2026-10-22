//! Memory bookkeeping for compute backends.
//!
//! Invariants (also in docs/MEMORY_INVARIANTS.md):
//! - **M1** CPU physical addresses, device bus addresses, VRAM offsets and GPU
//!   virtual addresses are different types ([`addr`]); nothing converts one
//!   into another implicitly.
//! - **M2** A CPU physical address becomes a bus address only when the IOMMU
//!   state is *known* to be off/pass-through ([`addr::IommuState`]).
//! - **M3** Allocated ranges never overlap and always lie inside the managed
//!   region; free + allocated = total ([`range::RangeAllocator::check`]).
//! - **M4** Every access through a buffer is checked against its size with
//!   overflow-safe arithmetic ([`region::check_access`]).
//! - **M5** Host memory given to a device for DMA is page aligned, inside the
//!   kernel's DMA window and pinned ([`dma::DmaPolicy`]).
//! - **M6** Buffer handles carry a generation; a released handle can never
//!   reach a reused slot ([`buffers::BufferTable`]).

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod addr;
pub mod buffers;
pub mod dma;
pub mod range;
pub mod region;

pub use range::RangeAllocator;
