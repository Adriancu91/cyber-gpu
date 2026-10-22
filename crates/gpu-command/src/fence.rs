//! Fence timeline: monotonic 64-bit sequence numbers. The device (or the CPU
//! backend) reports the last completed value; reports that go backwards or
//! beyond what was submitted are treated as faults.

use gpu_core::backend::Fence;
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{Clock, poll_until};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FenceTimeline {
    submitted: u64,
    completed: u64,
}

impl FenceTimeline {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn next_fence(&mut self) -> Fence {
        self.submitted += 1;
        Fence(self.submitted)
    }
    pub fn submitted(&self) -> Fence {
        Fence(self.submitted)
    }
    pub fn completed(&self) -> Fence {
        Fence(self.completed)
    }
    pub fn is_done(&self, f: Fence) -> bool {
        self.completed >= f.0
    }

    pub fn signal(&mut self, value: u64) -> GpuResult<()> {
        if value < self.completed || value > self.submitted {
            return Err(GpuError::HardwareFault { code: value as u32 });
        }
        self.completed = value;
        Ok(())
    }

    /// Drop everything pending (soft recovery): pending fences are marked
    /// complete so no waiter hangs; the caller reports the loss.
    pub fn abandon(&mut self) -> u64 {
        let lost = self.submitted - self.completed;
        self.completed = self.submitted;
        lost
    }

    /// Wait for `f`, polling `read_hw` for the completed value.
    pub fn wait<C: Clock + ?Sized>(
        &mut self,
        f: Fence,
        clock: &mut C,
        timeout_us: u64,
        mut read_hw: impl FnMut() -> u64,
    ) -> GpuResult<()> {
        if f.0 > self.submitted {
            return Err(GpuError::InvalidArgument("fence netrimis"));
        }
        let mut fault = None;
        let r = poll_until(clock, timeout_us, 10, || {
            if self.is_done(f) {
                return Some(());
            }
            if let Err(e) = self.signal(read_hw()) {
                fault = Some(e);
                return Some(());
            }
            self.is_done(f).then_some(())
        });
        match fault {
            Some(e) => Err(e),
            None => r,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::hal::Clock;

    struct C(u64);
    impl Clock for C {
        fn now_us(&mut self) -> u64 {
            self.0
        }
        fn delay_us(&mut self, us: u32) {
            self.0 += us as u64;
        }
    }

    #[test]
    fn monotonic_and_bounded() {
        let mut t = FenceTimeline::new();
        let a = t.next_fence();
        let b = t.next_fence();
        let mut hw = 0u64;
        let mut c = C(0);
        assert_eq!(
            t.wait(b, &mut c, 50, || hw),
            Err(GpuError::Timeout { waited_us: 50 })
        );
        hw = 1;
        assert!(t.wait(a, &mut c, 50, || hw).is_ok());
        assert!(t.signal(0).is_err(), "backwards");
        assert!(t.signal(3).is_err(), "beyond submitted");
        let mut step = 1u64;
        assert!(
            t.wait(b, &mut c, 1000, || {
                step += 1;
                step.min(2)
            })
            .is_ok()
        );
        assert!(t.wait(Fence(9), &mut c, 10, || 2).is_err());
        let _ = t.next_fence();
        assert_eq!(t.abandon(), 1);
    }
}
