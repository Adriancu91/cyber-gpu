//! Producer/consumer ring indices (the shape of a GPFIFO, an NVMe queue or an
//! xHCI ring). The driver owns PUT; the device reports GET. A GET value read
//! from hardware is validated before it is trusted.

use gpu_core::error::{GpuError, GpuResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ring {
    entries: u32,
    put: u32,
    get: u32,
}

impl Ring {
    /// `entries` must be a power of two >= 2. One slot stays empty so that
    /// full and empty are distinguishable.
    pub fn new(entries: u32) -> GpuResult<Self> {
        if entries < 2 || !entries.is_power_of_two() {
            return Err(GpuError::InvalidArgument("inel: numar de intrari invalid"));
        }
        Ok(Ring {
            entries,
            put: 0,
            get: 0,
        })
    }

    pub fn entries(&self) -> u32 {
        self.entries
    }
    pub fn put(&self) -> u32 {
        self.put
    }
    pub fn get(&self) -> u32 {
        self.get
    }
    fn mask(&self) -> u32 {
        self.entries - 1
    }
    pub fn pending(&self) -> u32 {
        self.put.wrapping_sub(self.get) & self.mask()
    }
    pub fn space(&self) -> u32 {
        self.entries - 1 - self.pending()
    }
    pub fn is_empty(&self) -> bool {
        self.put == self.get
    }

    /// Reserve `n` slots; returns the first slot index. Fails without side
    /// effects when there is not enough space.
    pub fn reserve(&mut self, n: u32) -> GpuResult<u32> {
        if n == 0 || n > self.space() {
            return Err(GpuError::OutOfMemory {
                requested: n as u64,
            });
        }
        let first = self.put;
        self.put = (self.put + n) & self.mask();
        Ok(first)
    }

    /// Accept a GET value reported by the device. It must lie in the
    /// pending window [get, put]; anything else is a hardware/protocol fault.
    pub fn update_get(&mut self, hw_get: u32) -> GpuResult<u32> {
        if hw_get >= self.entries {
            return Err(GpuError::HardwareFault { code: hw_get });
        }
        let advanced = hw_get.wrapping_sub(self.get) & self.mask();
        if advanced > self.pending() {
            return Err(GpuError::HardwareFault { code: hw_get });
        }
        self.get = hw_get;
        Ok(advanced)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_and_validate() {
        assert!(Ring::new(0).is_err());
        assert!(Ring::new(3).is_err());
        let mut r = Ring::new(8).unwrap();
        assert_eq!(r.space(), 7);
        assert_eq!(r.reserve(5), Ok(0));
        assert!(r.reserve(3).is_err(), "only 2 left");
        assert_eq!(r.update_get(4), Ok(4));
        assert_eq!(r.reserve(4), Ok(5));
        assert_eq!(r.put(), 1, "wrapped");
        assert_eq!(r.pending(), 5);
        assert!(
            r.update_get(3).is_err(),
            "GET behind current GET is outside the window"
        );
        assert!(r.update_get(2).is_err(), "beyond PUT");
        assert!(r.update_get(9).is_err());
        assert_eq!(r.update_get(1), Ok(5));
        assert!(r.is_empty());
    }
}
