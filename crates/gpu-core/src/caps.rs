//! Capabilities a backend reports. A capability is only reported as present
//! after it has been demonstrated by that backend; GA104 reports nothing
//! beyond identification until G5 passes on the physical card.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Feature {
    /// Device identity was read from the device itself.
    Identify,
    /// Device memory can be allocated through this backend.
    Allocate,
    /// Host <-> device copies work.
    Copy,
    /// Compute programs can be launched and complete.
    Compute,
    /// Completion is signalled by interrupt (otherwise: polling).
    Interrupts,
}

/// Bit set of numeric types a backend can compute with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DTypeSet(u8);

impl DTypeSet {
    pub const F32: DTypeSet = DTypeSet(1 << 0);
    pub const BF16: DTypeSet = DTypeSet(1 << 1);
    pub const F16: DTypeSet = DTypeSet(1 << 2);
    pub const I32: DTypeSet = DTypeSet(1 << 3);

    pub const fn empty() -> Self {
        DTypeSet(0)
    }
    pub const fn union(self, o: DTypeSet) -> Self {
        DTypeSet(self.0 | o.0)
    }
    pub const fn contains(self, o: DTypeSet) -> bool {
        self.0 & o.0 == o.0
    }
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Capabilities {
    /// Human readable backend name, e.g. "cpu-referinta" or "nvidia-ga104".
    pub backend: &'static str,
    /// Device memory known to the backend, in bytes (0 = unknown).
    pub memory_bytes: u64,
    /// Largest single allocation the backend accepts (0 = none possible).
    pub max_allocation: u64,
    pub dtypes: DTypeSet,
    features: u8,
}

impl Capabilities {
    pub fn new(backend: &'static str) -> Self {
        Capabilities {
            backend,
            memory_bytes: 0,
            max_allocation: 0,
            dtypes: DTypeSet::empty(),
            features: 0,
        }
    }
    fn bit(f: Feature) -> u8 {
        1 << (f as u8)
    }
    pub fn with(mut self, f: Feature) -> Self {
        self.features |= Self::bit(f);
        self
    }
    pub fn has(&self, f: Feature) -> bool {
        self.features & Self::bit(f) != 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn features_and_dtypes() {
        let c = Capabilities::new("x").with(Feature::Identify);
        assert!(c.has(Feature::Identify));
        assert!(!c.has(Feature::Compute));
        let d = DTypeSet::F32.union(DTypeSet::BF16);
        assert!(d.contains(DTypeSet::F32) && d.contains(DTypeSet::BF16));
        assert!(!d.contains(DTypeSet::F16));
        assert!(DTypeSet::empty().is_empty());
    }
}
