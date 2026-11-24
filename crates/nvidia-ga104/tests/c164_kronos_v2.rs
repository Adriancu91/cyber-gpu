//! T218 (C-164, `docs/C-164_KRONOS_V2_PLACA.md`): the launch counts of `gpu_compute::v2::StepCount` for today's
//! architecture are the board plan's own lists — `Plan::step` of the tiled `_3` plans, plain and with the two fusions
//! that exist today (`with_fused_back`, `with_fused_gelu`), at MIC and at #513 (2 × 512, the os-410 step). So the
//! sheet's counts for the candidates start from the real list, not from a hand count. Counting only: nothing runs.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_compute::v2::{Fusions, Kind, StepCount, TODAY};
use nvidia_ga104::train_step::{Launch, Plan, Prog};

const HEAP: u64 = 0x2_0000_0000;

fn kind(l: &Launch) -> Kind {
    match l.prog {
        Prog::MmTrain
        | Prog::MmTrainTile
        | Prog::MmTv128x64
        | Prog::MmTv64x128
        | Prog::MmTv128x128
        | Prog::MmTv64x64K8 => Kind::Mm,
        Prog::MmTrainGelu => Kind::MmActBack,
        Prog::AttnFwd | Prog::AttnFwdTile2 | Prog::AttnFwdTile3 => Kind::AttnFwd,
        Prog::AttnBack
        | Prog::AttnP
        | Prog::AttnPTile
        | Prog::AttnBackTile0_2
        | Prog::AttnBackTile2
        | Prog::AttnBackTile0_3
        | Prog::AttnBackTile3 => Kind::AttnBack,
        Prog::LnFwd => Kind::NormFwd,
        Prog::LnBack => Kind::NormBack,
        Prog::LnBackRes => Kind::NormBackRes,
        Prog::Colsum | Prog::ColsumPart => Kind::Colsum,
        Prog::GeluTr if l.tag.starts_with("inainte") => Kind::ActFwd,
        Prog::GeluTr => Kind::ActBack,
        Prog::Vadd => Kind::Vadd,
        Prog::EmbFwd => Kind::EmbFwd,
        Prog::EmbBack => Kind::EmbBack,
        Prog::Xent => Kind::Xent,
        Prog::Gnorm => Kind::Gnorm,
        Prog::OptStep => Kind::Opt,
    }
}

fn check(ar: Arch, bsz: usize, tl: usize) {
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % ar.vocab) as u32)
        .collect();
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    drop(params);
    for (back_res, gelu) in [(false, false), (true, false), (false, true), (true, true)] {
        let mut plan = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP).unwrap();
        if back_res {
            plan = plan.with_fused_back();
        }
        if gelu {
            plan = plan.with_fused_gelu();
        }
        let f = Fusions {
            back_res,
            act_back_in_mm: gelu,
            act_fwd_in_mm: false,
        };
        for (recipe, adamw) in [(Recipe::AdamW513, true), (Recipe::Sgd, false)] {
            let list = plan.step(recipe, &opt, 1).unwrap();
            let mut got = StepCount { n: [0; 17] };
            for l in &list {
                let k = gpu_compute::v2::KINDS
                    .iter()
                    .position(|&x| x == kind(l))
                    .unwrap();
                got.n[k] += 1;
            }
            let want = StepCount::count(&ar, TODAY, f, adamw);
            assert_eq!(got, want, "{back_res} {gelu} {adamw}");
            assert_eq!(want.total(), plan.launches_per_step(recipe));
        }
    }
}

#[test]
fn t218_today_counts_are_the_plan_lists_at_mic() {
    check(train::MIC, 2, 32);
}

#[test]
fn t218_today_counts_are_the_plan_lists_at_513() {
    check(train::M513, 2, 512);
}
