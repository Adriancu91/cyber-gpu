//! DMA buffer policy (invariant M5).
//!
//! cyber-universe today allocates DMA memory with
//! `alloc_contiguous_between(pages, 16 << 20, 1 << 32)` and pins it with
//! `frame::pin`. The default policy mirrors that window. A GPU can address
//! more than 32 bits, but widening the window is a separate, tested change.

use gpu_core::error::{GpuError, GpuResult};

use crate::addr::CpuPhys;

pub const PAGE: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DmaPolicy {
    pub low: u64,
    pub high: u64,
}

impl DmaPolicy {
    /// The window used by every cyber-universe driver today.
    pub const KERNEL_DEFAULT: DmaPolicy = DmaPolicy {
        low: 16 << 20,
        high: 1 << 32,
    };

    pub fn validate(&self, phys: CpuPhys, len: u64, pinned: bool) -> GpuResult<()> {
        if len == 0 {
            return Err(GpuError::InvalidArgument("buffer DMA gol"));
        }
        if !phys.0.is_multiple_of(PAGE) || !len.is_multiple_of(PAGE) {
            return Err(GpuError::InvalidArgument("buffer DMA nealiniat la pagina"));
        }
        let end = phys.0.checked_add(len).ok_or(GpuError::OutOfBounds {
            offset: phys.0,
            len,
            limit: self.high,
        })?;
        if phys.0 < self.low || end > self.high {
            return Err(GpuError::OutOfBounds {
                offset: phys.0,
                len,
                limit: self.high,
            });
        }
        if !pinned {
            return Err(GpuError::NotPermitted("buffer DMA nefixat (frame::pin)"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window() {
        let p = DmaPolicy::KERNEL_DEFAULT;
        assert!(p.validate(CpuPhys(16 << 20), PAGE, true).is_ok());
        assert!(p.validate(CpuPhys((1 << 32) - PAGE), PAGE, true).is_ok());
        assert!(p.validate(CpuPhys(1 << 32), PAGE, true).is_err());
        assert!(p.validate(CpuPhys(0x1000), PAGE, true).is_err());
        assert!(p.validate(CpuPhys((16 << 20) + 8), PAGE, true).is_err());
        assert!(p.validate(CpuPhys(16 << 20), PAGE, false).is_err());
        assert!(p.validate(CpuPhys(u64::MAX - 4095), PAGE, true).is_err());
    }
}
