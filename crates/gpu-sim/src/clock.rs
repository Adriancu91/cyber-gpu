//! SIMULATION clock: time advances only when someone waits.

use gpu_core::hal::Clock;

#[derive(Default)]
pub struct SimClock {
    pub now: u64,
}

impl Clock for SimClock {
    fn now_us(&mut self) -> u64 {
        self.now
    }
    fn delay_us(&mut self, us: u32) {
        self.now += us as u64;
    }
}
