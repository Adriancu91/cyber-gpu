//! Overflow-safe bounds checks (invariant M4).

use gpu_core::error::{GpuError, GpuResult};

/// Check that `[offset, offset + len)` lies inside `[0, limit)`.
pub fn check_access(offset: u64, len: u64, limit: u64) -> GpuResult<()> {
    match offset.checked_add(len) {
        Some(end) if end <= limit => Ok(()),
        _ => Err(GpuError::OutOfBounds { offset, len, limit }),
    }
}

pub fn align_up(v: u64, align: u64) -> Option<u64> {
    debug_assert!(align.is_power_of_two());
    v.checked_add(align - 1).map(|x| x & !(align - 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds() {
        assert!(check_access(0, 16, 16).is_ok());
        assert!(check_access(16, 0, 16).is_ok());
        assert!(check_access(1, 16, 16).is_err());
        assert!(check_access(u64::MAX, 2, u64::MAX).is_err());
        assert_eq!(align_up(5, 4), Some(8));
        assert_eq!(align_up(8, 4), Some(8));
        assert_eq!(align_up(u64::MAX, 4096), None);
    }
}
