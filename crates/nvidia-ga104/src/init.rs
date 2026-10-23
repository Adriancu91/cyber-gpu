//! GA10x bring-up as an explicit plan.
//!
//! The step list follows what nouveau (`nvkm/subdev/gsp/tu102.c`, reused by
//! `ga102.c`, MIT) and nova-core (`gsp/boot.rs`, GPL-2.0, read for the
//! sequence only) do on Ampere with GSP-RM. Each step states whether we
//! implement it, and if not, what blocks it. [`InitPlan::run`] executes the
//! implemented prefix and stops at the first step that is not implemented,
//! returning `GpuError::Blocked` with that step's blocker — it never reports
//! success for work it did not do.

use gpu_core::error::{GpuError, GpuResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StepStatus {
    /// Implemented and covered by tests (simulation); hardware confirmation pending.
    Implemented,
    /// Sequence known from public sources; code not written yet.
    NotImplemented,
    /// Cannot be done yet; the string is the blocker.
    Blocked(&'static str),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub id: &'static str,
    pub what: &'static str,
    pub status: StepStatus,
    /// Writes to the device (config or BAR0) or enables DMA.
    pub writes_hardware: bool,
}

pub const GA104_PLAN: &[Step] = &[
    Step {
        id: "I0",
        what: "identificare PCI (vendor/device/clasa, BAR-uri, capabilitati) — doar citire",
        status: StepStatus::Implemented,
        writes_hardware: false,
    },
    Step {
        id: "I1",
        what: "maparea primei pagini din BAR0 (UC) si citirea NV_PMC_BOOT_0/BOOT_42",
        status: StepStatus::Implemented,
        writes_hardware: false,
    },
    Step {
        id: "I2",
        what: "asteptarea GFW_BOOT (devinit facut de firmware-ul placii la pornire) — doar citire",
        status: StepStatus::Implemented,
        writes_hardware: false,
    },
    Step {
        id: "I3",
        what: "citirea VBIOS prin fereastra PROM: imaginile, tabela BIT, imaginile FWSEC — doar citire",
        status: StepStatus::Implemented,
        writes_hardware: false,
    },
    Step {
        id: "I4",
        what: "incarcarea fisierelor de firmware semnate (booter_load, bootloader, gsp-*.bin) in memorie",
        status: StepStatus::Blocked(
            "decizia de licenta LICENCE.nvidia + transportul fisierelor in OS (initrd/stick)",
        ),
        writes_hardware: false,
    },
    Step {
        id: "I5",
        what: "buffere DMA in RAM pentru WPR meta, radix3 si cozile GSP; Bus Master",
        status: StepStatus::Blocked(
            "starea IOMMU pe workstation necunoscuta (K0); fereastra DMA a OS-ului e sub 4 GiB",
        ),
        writes_hardware: true,
    },
    Step {
        id: "I6",
        what: "FWSEC-FRTS pe falcon-ul GSP (creeaza WPR2), verificare SCRATCH_E",
        status: StepStatus::Blocked("interfata falcon GA10x neimplementata; necesita I3"),
        writes_hardware: true,
    },
    Step {
        id: "I7",
        what: "pornirea GSP RISC-V si booter_load pe SEC2 cu adresa WPR meta",
        status: StepStatus::Blocked("necesita I4-I6"),
        writes_hardware: true,
    },
    Step {
        id: "I8",
        what: "protocolul RPC GSP-RM (SetSystemInfo, SetRegistry, secventiator, init done)",
        status: StepStatus::Blocked(
            "structurile RPC depind de versiunea firmware-ului (r535/r570); nedocumentate oficial",
        ),
        writes_hardware: true,
    },
    Step {
        id: "I9",
        what: "alocarea canalului GPFIFO (AMPERE_CHANNEL_GPFIFO_A) si a obiectului AMPERE_COMPUTE_B prin RM",
        status: StepStatus::Blocked("necesita I8"),
        writes_hardware: true,
    },
    Step {
        id: "I10",
        what: "tabele de pagini GPU, QMD v3, lansarea primului program si citirea rezultatului",
        status: StepStatus::Blocked("necesita I9 si un program SM86 valid (docs/TOOLCHAIN.md)"),
        writes_hardware: true,
    },
];

pub struct InitPlan {
    steps: &'static [Step],
    done: usize,
}

impl InitPlan {
    pub fn ga104() -> Self {
        InitPlan {
            steps: GA104_PLAN,
            done: 0,
        }
    }

    pub fn steps(&self) -> &'static [Step] {
        self.steps
    }

    /// Number of steps already executed.
    pub fn done(&self) -> usize {
        self.done
    }

    /// Mark the implemented prefix as done after `run_step` succeeds for
    /// each; stop at the first step that is not implemented.
    pub fn run(&mut self, mut run_step: impl FnMut(&Step) -> GpuResult<()>) -> GpuResult<()> {
        while let Some(step) = self.steps.get(self.done) {
            match step.status {
                StepStatus::Implemented => {
                    run_step(step)?;
                    self.done += 1;
                }
                StepStatus::NotImplemented => return Err(GpuError::NotImplemented(step.what)),
                StepStatus::Blocked(why) => return Err(GpuError::Blocked(why)),
            }
        }
        Ok(())
    }

    /// The first step that stops initialization.
    pub fn first_gap(&self) -> Option<&'static Step> {
        self.steps
            .iter()
            .find(|s| s.status != StepStatus::Implemented)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_honestly_at_first_gap() {
        let mut p = InitPlan::ga104();
        let mut ran = alloc::vec::Vec::new();
        let r = p.run(|s| {
            ran.push(s.id);
            Ok(())
        });
        assert_eq!(ran, ["I0", "I1", "I2", "I3"]);
        assert!(matches!(r, Err(GpuError::Blocked(_))));
        assert_eq!(p.first_gap().unwrap().id, "I4");
        assert_eq!(p.done(), 4);
    }

    #[test]
    fn implemented_steps_do_not_write() {
        assert!(
            GA104_PLAN
                .iter()
                .filter(|s| s.status == StepStatus::Implemented)
                .all(|s| !s.writes_hardware)
        );
    }

    #[test]
    fn step_error_propagates() {
        let mut p = InitPlan::ga104();
        let r = p.run(|_| Err(GpuError::NotResponding));
        assert_eq!(r, Err(GpuError::NotResponding));
        assert_eq!(p.done(), 0);
    }
}
