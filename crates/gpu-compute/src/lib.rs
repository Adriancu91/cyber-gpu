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
//! - [`cpu`]: a [`gpu_core::ComputeDevice`] that executes on the **CPU**. It is
//!   a real backend for the runtime (plan v0.2: CPU first) and a test double
//!   for the runtime; it is never evidence of GPU execution.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

extern crate alloc;

pub mod cpu;
pub mod kernels;
pub mod math;
pub mod op;
pub mod reference;
pub mod tensor;

pub use op::OpCode;
pub use tensor::{DType, TensorDesc};
