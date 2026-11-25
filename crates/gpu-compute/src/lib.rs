//! Compute layer above the backend contract.
//!
//! - [`tensor`]: explicit tensor format (dtype, shape, strides) — contract I2;
//! - [`math`]: `no_std` f32 functions (exp, sqrt, tanh, erf) — the kernel has
//!   no libm;
//! - [`reference`]: straightforward f32 implementations of the operators
//!   Kronos W2 needs. They are the numerical oracle every GPU kernel is
//!   compared against;
//! - [`kernels`]: faster CPU kernels (blocked matmul, bf16 weights),
//!   bit-identical to the reference;
//! - [`mma`]: binary16 conversions and the oracle + tolerance of the first
//!   tensor-core program (C-147 point 5; SIMULARE only);
//! - [`mma_s8`]: the bit-exact oracle of the int8 tensor-core probe (B1; SIMULARE only);
//! - [`rows`]: row-range versions of softmax (causal), attention and
//!   cross-entropy, so one operator can be split over several cores;
//!   bit-identical to the reference, no global state;
//! - [`train`]: the CPU oracle of a whole training step (C-160 point 3: forward,
//!   loss, backward, clipping, SGD / AdamW), checked against PyTorch's golden step;
//! - [`card`]: the model card read from a KGR1 header (C-157 point 8), which
//!   picks the compute pack;
//! - [`cpu`]: a [`gpu_core::ComputeDevice`] that executes on the **CPU**. It is
//!   a real backend for the runtime (plan v0.2: CPU first) and a test double
//!   for the runtime; it is never evidence of GPU execution.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod card;
pub mod cpu;
pub mod kernels;
pub mod layer;
pub mod math;
pub mod mm_tc;
pub mod mma;
pub mod mma_s8;
pub mod model;
pub mod op;
pub mod q8;
pub mod reference;
pub mod rows;
pub mod tensor;
pub mod train;
pub mod v2;

pub use op::OpCode;
pub use tensor::{DType, TensorDesc};
