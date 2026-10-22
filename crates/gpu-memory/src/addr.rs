//! Distinct address types (invariant M1) and the IOMMU gate (M2).

use gpu_core::error::{GpuError, GpuResult};

/// Physical address as the CPU sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct CpuPhys(pub u64);

/// Address a device uses on the bus to reach host memory (after the IOMMU, if any).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct BusAddr(pub u64);

/// Offset inside device-local memory (VRAM), from the start of VRAM.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct VramOffset(pub u64);

/// Address in a GPU virtual address space (needs GPU page tables).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct GpuVa(pub u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IommuState {
    /// Nobody checked (the default today: cyber-universe has no IOMMU code).
    Unknown,
    /// Verified absent, disabled in firmware, or in pass-through mode.
    OffOrPassthrough,
    /// Enabled with translation: needs IOMMU page tables we do not have.
    Translating,
}

impl BusAddr {
    /// The only CPU -> bus conversion. Identity mapping is valid only when
    /// the IOMMU is known not to translate.
    pub fn from_cpu_phys(p: CpuPhys, iommu: IommuState) -> GpuResult<BusAddr> {
        match iommu {
            IommuState::OffOrPassthrough => Ok(BusAddr(p.0)),
            IommuState::Unknown => Err(GpuError::Blocked("starea IOMMU necunoscuta (K0)")),
            IommuState::Translating => Err(GpuError::Blocked(
                "IOMMU activ: lipsesc tabelele IOMMU in OS",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iommu_gate() {
        let p = CpuPhys(0x1234_5000);
        assert_eq!(
            BusAddr::from_cpu_phys(p, IommuState::OffOrPassthrough),
            Ok(BusAddr(0x1234_5000))
        );
        assert!(matches!(
            BusAddr::from_cpu_phys(p, IommuState::Unknown),
            Err(GpuError::Blocked(_))
        ));
        assert!(matches!(
            BusAddr::from_cpu_phys(p, IommuState::Translating),
            Err(GpuError::Blocked(_))
        ));
    }
}
