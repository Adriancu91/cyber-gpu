//! The I1 contract for GA104. Today it identifies the card (read-only) and
//! then reports exactly where initialization stops. Memory, copies and
//! submission return errors until the device reaches `Ready`, which requires
//! the blocked steps in [`crate::init`].

use gpu_command::FenceTimeline;
use gpu_core::backend::{
    BufferDesc, BufferId, ComputeDevice, DeviceStatus, Fence, Identity, RecoveryLevel, Submission,
};
use gpu_core::caps::{Capabilities, Feature};
use gpu_core::diag::Journal;
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::{ConfigSpace, Mmio};
use gpu_core::ids::PciAddress;
use gpu_core::state::DeviceState;

use crate::init::InitPlan;
use crate::probe::{self, Level0};

pub struct Ga104Backend<C: ConfigSpace, M: Mmio> {
    cfg: C,
    bar0: M,
    address: Option<PciAddress>,
    state: DeviceState,
    level0: Option<Level0>,
    plan: InitPlan,
    fences: FenceTimeline,
    pub journal: Journal,
}

impl<C: ConfigSpace, M: Mmio> Ga104Backend<C, M> {
    /// `bar0` must be an uncached mapping of at least the first 4 KiB of BAR0.
    pub fn new(cfg: C, bar0: M, address: Option<PciAddress>) -> Self {
        Ga104Backend {
            cfg,
            bar0,
            address,
            state: DeviceState::Discovered,
            level0: None,
            plan: InitPlan::ga104(),
            fences: FenceTimeline::new(),
            journal: Journal::new(512),
        }
    }

    pub fn level0(&self) -> Option<&Level0> {
        self.level0.as_ref()
    }

    pub fn state(&self) -> DeviceState {
        self.state
    }

    fn not_ready(&self) -> GpuError {
        match self.plan.first_gap() {
            Some(step) => match step.status {
                crate::init::StepStatus::Blocked(why) => GpuError::Blocked(why),
                _ => GpuError::NotImplemented(step.what),
            },
            None => GpuError::InvalidState {
                expected: DeviceState::Ready,
                actual: self.state,
            },
        }
    }
}

impl<C: ConfigSpace, M: Mmio> ComputeDevice for Ga104Backend<C, M> {
    fn identify(&mut self) -> GpuResult<Identity> {
        if self.level0.is_none() {
            let l0 = probe::level0(
                &mut self.cfg,
                self.address,
                &mut self.bar0,
                &mut self.journal,
            )?;
            self.level0 = Some(l0);
            if self.state == DeviceState::Discovered {
                self.state.go(DeviceState::Identified)?;
            }
        }
        let l0 = self.level0.as_ref().ok_or(GpuError::NotPresent)?;
        Ok(Identity {
            address: self.address,
            pci: Some(l0.candidate.report.id),
            chip: "GA104",
            raw_id: Some(l0.chip.boot0.raw),
        })
    }

    fn capabilities(&self) -> Capabilities {
        let c = Capabilities::new("nvidia-ga104");
        if self.level0.is_some() {
            c.with(Feature::Identify)
        } else {
            c
        }
    }

    fn initialize(&mut self) -> GpuResult<()> {
        self.identify()?;
        let state = &mut self.state;
        self.plan.run(|step| {
            // I0 and I1 were executed by identify(); the probe results are
            // recorded in the journal. Mapped = BAR0 probe window usable.
            if step.id == "I1" && *state == DeviceState::Identified {
                state.go(DeviceState::Mapped)?;
            }
            Ok(())
        })
    }

    fn allocate(&mut self, _desc: BufferDesc) -> GpuResult<BufferId> {
        self.state
            .require(DeviceState::Ready)
            .map_err(|_| self.not_ready())?;
        Err(GpuError::NotImplemented("ga104: alocare VRAM"))
    }

    fn release(&mut self, _buf: BufferId) -> GpuResult<()> {
        Err(GpuError::InvalidHandle)
    }

    fn copy_host_to_device(&mut self, _src: &[u8], _dst: BufferId, _off: u64) -> GpuResult<()> {
        self.state
            .require(DeviceState::Ready)
            .map_err(|_| self.not_ready())?;
        Err(GpuError::NotImplemented("ga104: copiere gazda->dispozitiv"))
    }

    fn copy_device_to_host(&mut self, _src: BufferId, _off: u64, _dst: &mut [u8]) -> GpuResult<()> {
        self.state
            .require(DeviceState::Ready)
            .map_err(|_| self.not_ready())?;
        Err(GpuError::NotImplemented("ga104: copiere dispozitiv->gazda"))
    }

    fn submit(&mut self, _work: &Submission<'_>) -> GpuResult<Fence> {
        self.state
            .require(DeviceState::Ready)
            .map_err(|_| self.not_ready())?;
        Err(GpuError::NotImplemented("ga104: trimitere"))
    }

    fn synchronize(&mut self, fence: Fence, _timeout_us: u64) -> GpuResult<()> {
        if fence > self.fences.submitted() {
            return Err(GpuError::InvalidArgument("fence netrimis"));
        }
        Ok(()) // nothing can have been submitted
    }

    fn query_status(&mut self) -> GpuResult<DeviceStatus> {
        Ok(DeviceStatus {
            state: self.state,
            submitted: self.fences.submitted(),
            completed: self.fences.completed(),
            last_error: 0,
        })
    }

    fn reset_or_recover(&mut self, level: RecoveryLevel) -> GpuResult<()> {
        match level {
            RecoveryLevel::Soft => {
                self.fences.abandon();
                Ok(())
            }
            RecoveryLevel::Full => Err(GpuError::NotPermitted(
                "reset complet al GPU-ului principal fara protocol de recuperare",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_sim::mmio::SimMmio;
    use gpu_sim::pci::SimConfig;

    fn sim() -> Ga104Backend<SimConfig, SimMmio> {
        let mut m = SimMmio::ga104_bar0();
        m.set(0xa00, 0x174a_1000);
        Ga104Backend::new(SimConfig::ga104_rtx3070ti(), m, None)
    }

    #[test]
    fn identifies_then_reports_exact_blocker() {
        let mut d = sim();
        let id = d.identify().unwrap();
        assert_eq!(id.chip, "GA104");
        assert_eq!(id.raw_id, Some(0x1740_00a1));
        assert!(d.capabilities().has(Feature::Identify));
        assert!(!d.capabilities().has(Feature::Compute));
        let r = d.initialize();
        assert!(matches!(r, Err(GpuError::NotImplemented(w)) if w.contains("GFW_BOOT")));
        assert_eq!(d.state(), DeviceState::Mapped);
        assert!(matches!(
            d.allocate(BufferDesc::device(256)),
            Err(GpuError::NotImplemented(_))
        ));
        assert!(
            d.submit(&Submission {
                kernel: gpu_core::backend::KernelRef::Builtin(1),
                buffers: &[],
                scalars: &[],
                grid: [1; 3],
                block: [1; 3]
            })
            .is_err()
        );
        assert_eq!(d.journal.writes(), 0, "nothing written to the device");
        assert!(matches!(
            d.reset_or_recover(RecoveryLevel::Full),
            Err(GpuError::NotPermitted(_))
        ));
    }

    #[test]
    fn absent_card() {
        let mut d = Ga104Backend::new(SimConfig::absent(), SimMmio::ga104_bar0(), None);
        assert_eq!(d.identify().unwrap_err(), GpuError::NotPresent);
        assert_eq!(d.initialize().unwrap_err(), GpuError::NotPresent);
    }
}
