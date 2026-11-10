//! SIMULATION clock: time advances only when someone waits.

use gpu_core::hal::Clock;

#[derive(Default)]
pub struct SimClock {
    pub now: u64,
    /// C-99: SIM stand-in for the Esc key — once `now` reaches this, every
    /// wait answers `abort_requested() == true`. `None` = never.
    pub abort_at: Option<u64>,
}

impl Clock for SimClock {
    fn now_us(&mut self) -> u64 {
        self.now
    }
    fn delay_us(&mut self, us: u32) {
        self.now += us as u64;
    }
    fn abort_requested(&mut self) -> bool {
        self.abort_at.is_some_and(|t| self.now >= t)
    }
}
