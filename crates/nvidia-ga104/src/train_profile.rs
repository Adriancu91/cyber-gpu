//! C-157 point 2 after os-392 (`docs/C-157_P2_PROFIL_PAS.md`): **where the board's time goes inside one training
//! step**, split by launch.
//!
//! The OS already writes one timestamp record after every launch of the step and reads all of them on a few steps
//! (`RecordPlan::DIET`). [`StepProfile`] names every launch (its `train_step::Launch::tag` and program), takes the
//! per-launch times of each fully read step (directly, or from the records through `transfer::analyze_sequence`,
//! with the same refusals) and gives the median per step, summed by tag (the layers of one tag added together),
//! by program and by stage (`inainte`, `eroarea`, `inapoi`, `opt`, the batch copies).
//!
//! As in `kronos_perf`, the time between two records also holds whatever the list puts before the launch (an
//! invalidation, if any): the row is "launch + what precedes it", and the rows add up to the board time of the step.
//!
//! Only additions; nothing in [`crate::train_step`] changes. Tested on host arithmetic and SIMULARE only (T210):
//! the numbers in the tests are synthetic and say nothing about the RTX 3070 Ti.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_core::error::{GpuError, GpuResult};

use crate::train_step::Launch;
use crate::transfer;

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

fn median(v: &mut [u64]) -> u64 {
    if v.is_empty() {
        return 0;
    }
    v.sort_unstable();
    v[v.len() / 2]
}

/// The tag of the copies RAM → arena put in front of the step (the batch through RAM, `batch_ram`).
pub const COPY_UP: &str = "lot:copiere-sus";
/// The tag of the copies arena → RAM put behind the step (losses + norm through RAM, `batch_ram`).
pub const COPY_DOWN: &str = "lot:copiere-jos";
/// The program name of those copies (`transfer::COPY_B128`).
pub const COPY_PROG: &str = "copy_b128";

/// The stage of a tag: the text before `:` (`inainte`, `inapoi`, `opt`, `lot`), or the whole tag (`eroarea`).
pub fn stage(tag: &'static str) -> &'static str {
    match tag.find(':') {
        Some(i) => &tag[..i],
        None => tag,
    }
}

/// One group of launches (one tag, one program or one stage).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Group {
    pub name: &'static str,
    /// Launches of this group in one step.
    pub launches: u32,
    /// The group's sum over one step, median over the steps (ns).
    pub median_ns: u64,
    /// That median as a share of the median step total, in thousandths.
    pub permille: u32,
}

/// The per-launch times of the fully read steps of one training list.
#[derive(Clone, Debug, Default)]
pub struct StepProfile {
    /// (tag, program) of every launch, in list order.
    names: Vec<(&'static str, &'static str)>,
    /// Per step: the time of every launch (ns), same order as `names`.
    steps: Vec<Vec<u64>>,
}

impl StepProfile {
    /// The names of a step's launches, as `Plan::step` gives them.
    pub fn from_launches(list: &[Launch]) -> GpuResult<Self> {
        if list.is_empty() {
            return Err(bad("profil: lista pasului e goala"));
        }
        Ok(StepProfile {
            names: list.iter().map(|l| (l.tag, l.prog.name())).collect(),
            steps: Vec::new(),
        })
    }

    /// The same list with `up` copies in front and `down` copies behind (the batch through RAM: 5 and 2).
    /// Refused once steps were added (the launch count would change under them).
    pub fn with_copies(mut self, up: usize, down: usize) -> GpuResult<Self> {
        if !self.steps.is_empty() {
            return Err(bad("profil: copierile se adauga inaintea pasilor"));
        }
        let mut names = Vec::with_capacity(self.names.len() + up + down);
        names.extend((0..up).map(|_| (COPY_UP, COPY_PROG)));
        names.extend_from_slice(&self.names);
        names.extend((0..down).map(|_| (COPY_DOWN, COPY_PROG)));
        self.names = names;
        Ok(self)
    }

    /// Launches per step (copies included).
    pub fn launches(&self) -> usize {
        self.names.len()
    }

    /// Fully read steps added so far.
    pub fn steps(&self) -> usize {
        self.steps.len()
    }

    /// One step's per-launch times (ns), in list order. Another length is refused: nothing is guessed.
    pub fn add_step(&mut self, ns: &[u64]) -> GpuResult<()> {
        if ns.len() != self.names.len() {
            return Err(bad("profil: alt numar de timpi decat de lansari"));
        }
        self.steps.push(ns.to_vec());
        Ok(())
    }

    /// One step from its records (`launches() + 1` records of `launch_cost::RECORD` bytes, payloads from
    /// `first_payload`), through `transfer::analyze_sequence`: a missing record or a clock going back is an error.
    pub fn add_records(&mut self, records: &[u8], first_payload: u32) -> GpuResult<()> {
        let ns = transfer::analyze_sequence(records, self.names.len(), first_payload)?;
        self.add_step(&ns)
    }

    /// The median step total (ns): the board time of the step, first record → last.
    pub fn total_ns(&self) -> u64 {
        let mut t: Vec<u64> = self.steps.iter().map(|s| s.iter().sum()).collect();
        median(&mut t)
    }

    fn grouped(&self, key: impl Fn(&'static str, &'static str) -> &'static str) -> Vec<Group> {
        let mut keys: Vec<&'static str> = Vec::new();
        let mut index = Vec::with_capacity(self.names.len());
        for &(tag, prog) in &self.names {
            let k = key(tag, prog);
            let i = match keys.iter().position(|&x| x == k) {
                Some(i) => i,
                None => {
                    keys.push(k);
                    keys.len() - 1
                }
            };
            index.push(i);
        }
        let total = self.total_ns();
        let mut out: Vec<Group> = keys
            .iter()
            .enumerate()
            .map(|(g, &name)| {
                let mut sums: Vec<u64> = self
                    .steps
                    .iter()
                    .map(|s| {
                        s.iter()
                            .zip(&index)
                            .filter(|&(_, &i)| i == g)
                            .map(|(&t, _)| t)
                            .sum()
                    })
                    .collect();
                let m = median(&mut sums);
                Group {
                    name,
                    launches: index.iter().filter(|&&i| i == g).count() as u32,
                    median_ns: m,
                    permille: m
                        .saturating_mul(1000)
                        .checked_div(total)
                        .unwrap_or(0)
                        .min(u32::MAX as u64) as u32,
                }
            })
            .collect();
        // Largest first; equal times keep the list order (stable sort).
        out.sort_by_key(|g| core::cmp::Reverse(g.median_ns));
        out
    }

    /// By tag (`inapoi:cap-dX`, …): the layers of one tag added together.
    pub fn by_tag(&self) -> Vec<Group> {
        self.grouped(|tag, _| tag)
    }

    /// By program (`mm_train`, `colsum`, …).
    pub fn by_prog(&self) -> Vec<Group> {
        self.grouped(|_, prog| prog)
    }

    /// By stage (`inainte`, `eroarea`, `inapoi`, `opt`, `lot`).
    pub fn by_stage(&self) -> Vec<Group> {
        self.grouped(|tag, _| stage(tag))
    }

    /// The text rows for the OS's report (ASCII, as the rest of its journal): one header, the stages, then the `top`
    /// largest tags and the programs. Empty when no step was added.
    pub fn lines(&self, top: usize) -> Vec<String> {
        if self.steps.is_empty() {
            return Vec::new();
        }
        let row = |g: &Group| {
            format!(
                "  {:<26} {:>6} us ({:>3}.{} %, {} lansari)",
                g.name,
                g.median_ns / 1000,
                g.permille / 10,
                g.permille % 10,
                g.launches
            )
        };
        let mut v = Vec::new();
        v.push(format!(
            "profilul pasului (placa, median pe {} pasi cititi intregi): {} us, {} lansari",
            self.steps.len(),
            self.total_ns() / 1000,
            self.names.len()
        ));
        v.push(String::from(" pe etape:"));
        v.extend(self.by_stage().iter().map(row));
        v.push(format!(" cele mai lungi {top} etichete:"));
        v.extend(self.by_tag().iter().take(top).map(row));
        v.push(String::from(" pe programe:"));
        v.extend(self.by_prog().iter().map(row));
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stage_of_tags() {
        assert_eq!(stage("inapoi:cap-dX"), "inapoi");
        assert_eq!(stage("eroarea"), "eroarea");
        assert_eq!(stage(COPY_UP), "lot");
    }
}
