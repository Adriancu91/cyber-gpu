//! Contract I1 (kernel <-> compute backend). Small on purpose so that a CPU
//! backend, an AMD backend and the GA104 backend can all implement it without
//! the runtime above them changing.
//!
//! Rules every implementation follows:
//! - an operation that is not supported yet returns `NotImplemented` or
//!   `Blocked`, never `Ok`;
//! - every buffer access is bounds-checked against the buffer size;
//! - every wait is bounded by a timeout;
//! - no GA104 register appears in this API.

use crate::caps::Capabilities;
use crate::error::GpuResult;
use crate::ids::{PciAddress, PciId};
use crate::state::DeviceState;

/// Opaque buffer handle. Generation-tagged so a stale handle is detected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BufferId {
    pub index: u32,
    pub generation: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemLocation {
    /// Device-local memory (VRAM on a discrete GPU, RAM for the CPU backend).
    Device,
    /// Host memory the device can reach by DMA.
    HostVisible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BufferDesc {
    pub size: u64,
    pub align: u64,
    pub location: MemLocation,
}

impl BufferDesc {
    pub const fn device(size: u64) -> Self {
        BufferDesc {
            size,
            align: 256,
            location: MemLocation::Device,
        }
    }
}

/// Monotonic completion counter. Fence `n` is complete once the backend's
/// completed value is >= n.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fence(pub u64);

/// Binary formats a backend may accept. `NvidiaCubinSm86` is listed so the
/// contract can name it; the GA104 backend refuses it until a loader exists
/// (docs/TOOLCHAIN.md).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryFormat {
    NvidiaCubinSm86,
    NvidiaRawSass,
    AmdGcnCodeObject,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelRef<'a> {
    /// An operation the backend implements itself (op codes: crate gpu-compute).
    Builtin(u32),
    /// A device binary. The backend must validate format and contents.
    Binary {
        format: BinaryFormat,
        bytes: &'a [u8],
        entry: &'a str,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Submission<'a> {
    pub kernel: KernelRef<'a>,
    pub buffers: &'a [BufferId],
    /// Small immediate arguments (dimensions, flags, f32 bit patterns).
    pub scalars: &'a [u32],
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceStatus {
    pub state: DeviceState,
    pub submitted: Fence,
    pub completed: Fence,
    /// Last backend specific error code (0 = none).
    pub last_error: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub address: Option<PciAddress>,
    pub pci: Option<PciId>,
    /// Chip name as confirmed from the device, e.g. "GA104".
    pub chip: &'static str,
    /// Raw identity register as read (e.g. NV_PMC_BOOT_0), if any.
    pub raw_id: Option<u32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecoveryLevel {
    /// Drop pending work, keep the device configured.
    Soft,
    /// Reinitialize the device. Backends may refuse (`NotPermitted`).
    Full,
}

/// The backend contract.
pub trait ComputeDevice {
    fn identify(&mut self) -> GpuResult<Identity>;
    fn capabilities(&self) -> Capabilities;
    fn initialize(&mut self) -> GpuResult<()>;
    fn allocate(&mut self, desc: BufferDesc) -> GpuResult<BufferId>;
    fn release(&mut self, buf: BufferId) -> GpuResult<()>;
    fn copy_host_to_device(&mut self, src: &[u8], dst: BufferId, dst_offset: u64) -> GpuResult<()>;
    fn copy_device_to_host(
        &mut self,
        src: BufferId,
        src_offset: u64,
        dst: &mut [u8],
    ) -> GpuResult<()>;
    fn submit(&mut self, work: &Submission<'_>) -> GpuResult<Fence>;
    fn synchronize(&mut self, fence: Fence, timeout_us: u64) -> GpuResult<()>;
    fn query_status(&mut self) -> GpuResult<DeviceStatus>;
    fn reset_or_recover(&mut self, level: RecoveryLevel) -> GpuResult<()>;
}
