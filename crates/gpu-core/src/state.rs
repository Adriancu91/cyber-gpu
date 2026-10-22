//! Device lifecycle. Transitions are explicit and checked so that, for
//! example, nothing can be submitted to a device that was never initialized.
//!
//! ```text
//! Discovered -> Identified -> Mapped -> Initialized -> Ready
//!      \            \           \           \           \
//!       +------------+-----------+-----------+-----------+--> Faulted
//! Faulted -> Identified   (only through reset_or_recover)
//! ```

use crate::error::{GpuError, GpuResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviceState {
    /// Seen on the PCI bus, nothing read from the device itself yet.
    Discovered,
    /// Identity confirmed from device registers (e.g. NV_PMC_BOOT_0).
    Identified,
    /// MMIO windows mapped and bounds known.
    Mapped,
    /// Engines/firmware brought up; memory management usable.
    Initialized,
    /// Can accept submissions.
    Ready,
    /// An error was detected; only recovery is allowed.
    Faulted,
}

impl DeviceState {
    pub fn can_go(self, to: DeviceState) -> bool {
        use DeviceState::*;
        matches!(
            (self, to),
            (Discovered, Identified)
                | (Identified, Mapped)
                | (Mapped, Initialized)
                | (Initialized, Ready)
                | (Faulted, Identified)
        ) || (to == Faulted && self != Faulted)
    }

    /// Checked transition.
    pub fn go(&mut self, to: DeviceState) -> GpuResult<()> {
        if self.can_go(to) {
            *self = to;
            Ok(())
        } else {
            Err(GpuError::InvalidState {
                expected: to.predecessor(),
                actual: *self,
            })
        }
    }

    /// Require at least `min` (and not Faulted).
    pub fn require(self, min: DeviceState) -> GpuResult<()> {
        if self != DeviceState::Faulted && self >= min {
            Ok(())
        } else {
            Err(GpuError::InvalidState {
                expected: min,
                actual: self,
            })
        }
    }

    fn predecessor(self) -> DeviceState {
        use DeviceState::*;
        match self {
            Discovered | Faulted => Discovered,
            Identified => Discovered,
            Mapped => Identified,
            Initialized => Mapped,
            Ready => Initialized,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DeviceState::*;
    use super::*;

    #[test]
    fn legal_path() {
        let mut s = Discovered;
        for t in [Identified, Mapped, Initialized, Ready] {
            s.go(t).unwrap();
        }
        assert_eq!(s, Ready);
        s.go(Faulted).unwrap();
        assert!(s.go(Ready).is_err());
        s.go(Identified).unwrap();
    }

    #[test]
    fn no_skipping() {
        let mut s = Discovered;
        assert_eq!(
            s.go(Ready),
            Err(GpuError::InvalidState {
                expected: Initialized,
                actual: Discovered
            })
        );
        assert!(s.require(Identified).is_err());
        s.go(Identified).unwrap();
        assert!(s.require(Identified).is_ok());
        assert!(Faulted.require(Discovered).is_err());
    }
}
