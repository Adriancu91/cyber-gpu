//! Structured errors. Every operation that is not implemented, or that is
//! blocked by a missing hardware mechanism, returns a *distinct* error;
//! nothing in this project fakes success.

use core::fmt;

use crate::state::DeviceState;

pub type GpuResult<T> = Result<T, GpuError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuError {
    /// The code path exists in the contract but has not been written yet.
    NotImplemented(&'static str),
    /// The step cannot be done with what we know today (missing documentation,
    /// firmware, OS mechanism). The string names the blocker; see docs/BLOCKERS.md.
    Blocked(&'static str),
    /// No device at this address, or the vendor/device id is not the expected one.
    NotPresent,
    /// The device was found but this backend does not support it.
    Unsupported { vendor: u16, device: u16 },
    /// Reads return all ones: device absent, in D3cold, or link down.
    NotResponding,
    /// An argument violates the contract (zero size, bad alignment, ...).
    InvalidArgument(&'static str),
    /// Access outside a mapped region or buffer.
    OutOfBounds { offset: u64, len: u64, limit: u64 },
    /// The allocator has no free range that satisfies the request.
    OutOfMemory { requested: u64 },
    /// Unknown buffer handle, or a handle used after release.
    InvalidHandle,
    /// The operation needs another lifecycle state.
    InvalidState {
        expected: DeviceState,
        actual: DeviceState,
    },
    /// A bounded wait expired.
    Timeout { waited_us: u64 },
    /// The safety policy refused the operation (e.g. a write to a register
    /// that is not on the allow-list, or a risky probe without a permit).
    NotPermitted(&'static str),
    /// The device reported an error; `code` is backend specific.
    HardwareFault { code: u32 },
    /// A read-back check failed after a register restore.
    VerifyFailed {
        offset: u32,
        expected: u32,
        actual: u32,
    },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::NotImplemented(w) => write!(f, "neimplementat: {w}"),
            GpuError::Blocked(w) => write!(f, "blocat: {w}"),
            GpuError::NotPresent => write!(f, "dispozitivul nu exista"),
            GpuError::Unsupported { vendor, device } => {
                write!(f, "dispozitiv nesuportat {vendor:04x}:{device:04x}")
            }
            GpuError::NotResponding => write!(f, "dispozitivul nu raspunde (0xffffffff)"),
            GpuError::InvalidArgument(w) => write!(f, "argument invalid: {w}"),
            GpuError::OutOfBounds { offset, len, limit } => {
                write!(
                    f,
                    "acces in afara limitelor: {offset:#x}+{len:#x} > {limit:#x}"
                )
            }
            GpuError::OutOfMemory { requested } => {
                write!(f, "memorie insuficienta: {requested} octeti")
            }
            GpuError::InvalidHandle => write!(f, "handle invalid"),
            GpuError::InvalidState { expected, actual } => {
                write!(f, "stare gresita: asteptat {expected:?}, actual {actual:?}")
            }
            GpuError::Timeout { waited_us } => write!(f, "timeout dupa {waited_us} us"),
            GpuError::NotPermitted(w) => write!(f, "nepermis de politica de siguranta: {w}"),
            GpuError::HardwareFault { code } => write!(f, "eroare hardware {code:#x}"),
            GpuError::VerifyFailed {
                offset,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "verificare esuata la {offset:#x}: asteptat {expected:#x}, citit {actual:#x}"
                )
            }
        }
    }
}
