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
use gpu_core::hal::{Clock, ConfigSpace, Mmio};
use gpu_core::ids::PciAddress;
use gpu_core::state::DeviceState;

use crate::gfw::{self, GfwStatus};
use crate::init::InitPlan;
use crate::probe::{self, Level0};
use crate::vbios::{self, VbiosLayout};

pub struct Ga104Backend<C: ConfigSpace, M: Mmio, K: Clock> {
    cfg: C,
    bar0: M,
    clock: K,
    address: Option<PciAddress>,
    state: DeviceState,
    level0: Option<Level0>,
    gfw: Option<GfwStatus>,
    vbios: Option<VbiosLayout>,
    plan: InitPlan,
    fences: FenceTimeline,
    pub journal: Journal,
}

impl<C: ConfigSpace, M: Mmio, K: Clock> Ga104Backend<C, M, K> {
    /// `bar0` must be an uncached mapping of BAR0: at least the first 4 KiB
    /// for `identify`, at least [`crate::regs::INIT_WINDOW`] (4 MiB) for the
    /// read-only steps I2–I3 of `initialize`.
    pub fn new(cfg: C, bar0: M, clock: K, address: Option<PciAddress>) -> Self {
        Ga104Backend {
            cfg,
            bar0,
            clock,
            address,
            state: DeviceState::Discovered,
            level0: None,
            gfw: None,
            vbios: None,
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

    /// Result of I2, once run.
    pub fn gfw(&self) -> Option<&GfwStatus> {
        self.gfw.as_ref()
    }

    /// Result of I3, once run (also kept when I3 found an incomplete VBIOS).
    pub fn vbios(&self) -> Option<&VbiosLayout> {
        self.vbios.as_ref()
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

impl<C: ConfigSpace, M: Mmio, K: Clock> ComputeDevice for Ga104Backend<C, M, K> {
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
        let Ga104Backend {
            bar0,
            clock,
            state,
            gfw,
            vbios,
            plan,
            journal,
            ..
        } = self;
        plan.run(|step| {
            match step.id {
                // I0 and I1 were executed by identify(); the probe results are
                // recorded in the journal. Mapped = BAR0 probe window usable.
                "I1" if *state == DeviceState::Identified => state.go(DeviceState::Mapped)?,
                "I2" if gfw.is_none() => {
                    *gfw = Some(gfw::wait_gfw_boot(
                        bar0,
                        journal,
                        clock,
                        gfw::GFW_TIMEOUT_US,
                    )?);
                }
                "I3" if vbios.is_none() => {
                    let l = vbios::read_layout(bar0, journal, clock, vbios::VBIOS_TIMEOUT_US)?;
                    let complete = l.complete();
                    *vbios = Some(l);
                    if !complete {
                        return Err(GpuError::Blocked(
                            "I3: VBIOS citit, dar fara PCI-AT+BIT+FWSEC (vezi raportul)",
                        ));
                    }
                }
                _ => {}
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
    use gpu_sim::clock::SimClock;
    use gpu_sim::mmio::SimMmio;
    use gpu_sim::pci::SimConfig;

    // SIMULARE: a 4 MiB BAR0 model with BOOT_0/42, GFW completed and a
    // synthetic VBIOS in the PROM window.
    fn sim() -> Ga104Backend<SimConfig, SimMmio, SimClock> {
        let mut m = SimMmio::ga104_bar0();
        m.set(0xa00, 0x174a_1000);
        m.set(0x11_8128, 1);
        m.set(0x11_8234, 0xff);
        m.load_bytes(crate::regs::PROM_BASE, &gpu_sim::rom::ga10x_like_chain());
        Ga104Backend::new(
            SimConfig::workstation_rtx3070ti(),
            m,
            SimClock::default(),
            None,
        )
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
        assert!(matches!(r, Err(GpuError::Blocked(w)) if w.contains("LICENCE.nvidia")));
        assert_eq!(d.state(), DeviceState::Mapped);
        assert_eq!(d.gfw().unwrap().progress, 0xff);
        assert!(d.vbios().unwrap().complete());
        assert!(matches!(
            d.allocate(BufferDesc::device(256)),
            Err(GpuError::Blocked(_))
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
        let mut d = Ga104Backend::new(
            SimConfig::absent(),
            SimMmio::ga104_bar0(),
            SimClock::default(),
            None,
        );
        assert_eq!(d.identify().unwrap_err(), GpuError::NotPresent);
        assert_eq!(d.initialize().unwrap_err(), GpuError::NotPresent);
    }

    #[test]
    fn gfw_never_finishing_stops_at_i2() {
        let mut d = sim();
        d.bar0.set(0x11_8234, 0x10);
        assert!(matches!(d.initialize(), Err(GpuError::Timeout { .. })));
        assert!(d.gfw().is_none() && d.vbios().is_none());
        assert_eq!(d.journal.writes(), 0);
    }

    #[test]
    fn probe_window_only_cannot_run_i2() {
        let mut m = SimMmio::new(0x1000);
        m.set(0, 0x1740_00a1);
        m.set(0xa00, 0x174a_1000);
        let mut d = Ga104Backend::new(SimConfig::ga104_rtx3070ti(), m, SimClock::default(), None);
        d.identify().unwrap();
        assert!(matches!(d.initialize(), Err(GpuError::InvalidArgument(_))));
    }
}
