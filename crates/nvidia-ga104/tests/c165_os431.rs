//! T238 (C-165 / C-157 p.2, `docs/C-165_VALURI.md`): os-423 (the occupancy probe) and os-431 (the #513 step at one
//! shared-memory step, 64 KiB) — the board's rows copied unchanged into `tests/rapoarte/os423-ocupare.txt` and
//! `os431-pas.txt` — against the "waves" model ([`nvidia_ga104::wave_model`]). On the processor: the board times are
//! the board's (measured by the OS); the model's numbers are DEDUS; nothing here runs on (or simulates) the board.

use gpu_compute::train::{self, AdamW, Recipe};
use nvidia_ga104::mm_train_variants::{self as mv, RACE_WINNER_513, VARIANTS, parse_race};
use nvidia_ga104::train_profile::{ParsedProfile, parse_profiles};
use nvidia_ga104::train_step::Plan;
use nvidia_ga104::wave_model::{self as wm, STEP_OVER_RACE_PERMILLE};

const OS423: &str = include_str!("rapoarte/os423-ocupare.txt");
const OS431: &str = include_str!("rapoarte/os431-pas.txt");
const OS418: &str = include_str!("rapoarte/os418-pas.txt");
const OS416_RACE: &str = include_str!("rapoarte/os416-cursa.txt");
const HEAP: u64 = nvidia_ga104::mmu::VA_BASE + 0x100_0000;
const KIB64: u32 = 64 << 10;

fn the_513_profile(text: &str) -> ParsedProfile {
    parse_profiles(text)
        .into_iter()
        .find(|p| p.context.contains("antrenare #513"))
        .expect("the #513 profile")
}

fn forms_513() -> Vec<mv::Form> {
    let (plan, opt) = plan_513();
    mv::forms(&plan.step(Recipe::AdamW513, &opt, 1).unwrap())
}

fn plan_513() -> (Plan, AdamW) {
    let (bsz, tl) = (2usize, 512usize);
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let plan = Plan::new_tiles3(train::M513, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile()
        .resident()
        .unwrap();
    let params: Vec<Vec<f32>> = train::M513
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    (plan, opt)
}

/// `ocupare mm_train_tile 8 KiB: 2840.0 us, …` → (name, KiB, ns).
fn probe_rows() -> Vec<(String, u32, u64)> {
    OS423
        .lines()
        .filter_map(|l| {
            let r = l.split("): ocupare ").nth(1)?;
            let (name, rest) = r.split_once(' ')?;
            let (kib, rest) = rest.split_once(" KiB: ")?;
            let (us, _) = rest.split_once(" us,")?;
            let (a, b) = us.split_once('.')?;
            let ns = a.parse::<u64>().ok()? * 1000 + b.parse::<u64>().ok()? * 100;
            Some((name.to_string(), kib.parse().ok()?, ns))
        })
        .collect()
}

/// §1: os-423 — 10 rows, all the same bits; the model's table is these rows / their waves.
#[test]
fn t238a_the_probe_and_the_table() {
    let rows = probe_rows();
    assert_eq!(rows.len(), 10);
    assert!(OS423.contains("proba ocuparii: 10 / 10 aceiasi biti"));
    let qkv = (1024u32, 2304u32, 768u32);
    for (name, kib, ns) in &rows {
        let v = VARIANTS.iter().position(|x| x.name == name).unwrap();
        let b = wm::blocks_per_sm(v, kib << 10).unwrap();
        let model = wm::launch_ns(v, qkv.0, qkv.1, qkv.2, b).unwrap();
        // The table was made from these rows: every probe launch within 1 % (128x128's last wave runs at 1 block).
        assert!(
            model.abs_diff(*ns) * 100 <= *ns,
            "{name} {kib} KiB: model {model} ns, board {ns} ns"
        );
    }
    // P5 → R0: at 16 KiB, 128x128 has 2 blocks only without the 1 KiB reserve, and the board got faster.
    let k3 = nvidia_ga104::occupancy::variant_kernel(3);
    use nvidia_ga104::occupancy::{Reserve, SM86, blocks_per_sm};
    assert_eq!(
        blocks_per_sm(&SM86, &k3, 16 << 10, Reserve::None)
            .unwrap()
            .blocks,
        2
    );
    assert_eq!(
        blocks_per_sm(&SM86, &k3, 16 << 10, Reserve::OneKib)
            .unwrap()
            .blocks,
        1
    );
    let t = |kib: u32| {
        rows.iter()
            .find(|r| r.0 == "mm_tv_128x128" && r.1 == kib)
            .unwrap()
            .2
    };
    assert!(t(16) * 1000 / t(8) <= 700, "R0: 16 KiB is faster");
}

/// V1: the model on the os-416 race (b = 1): every 128x128 shape within 11 %.
#[test]
fn t238b_model_on_the_race() {
    let race = parse_race(OS416_RACE);
    let mut worst = 0u64;
    for f in &race.counts {
        let r = race
            .rows
            .iter()
            .find(|r| {
                (r.m, r.n, r.k, r.flags, r.variant) == (f.m, f.n, f.k, f.flags, RACE_WINNER_513)
            })
            .unwrap();
        let m = wm::launch_ns(RACE_WINNER_513, f.m, f.n, f.k, 1).unwrap();
        let pm = m.abs_diff(r.median_ns) * 1000 / r.median_ns;
        worst = worst.max(pm);
        assert!(pm <= 110, "{f:?}: model {m}, race {}", r.median_ns);
    }
    println!("kt t238b modelul pe cursa os-416 (128x128, b=1): abaterea maxima {worst} promile");
}

/// The 7 tags present in both os-418 and os-431, with their shape.
const TAGS: [(&str, u32, u32, u32); 7] = [
    ("inainte:fc", 1024, 3072, 768),
    ("inapoi:proj-dX", 1024, 3072, 768),
    ("inapoi:proj-dW", 768, 3072, 1024),
    ("inapoi:fc-dW", 3072, 768, 1024),
    ("inapoi:qkv-dW", 2304, 768, 1024),
    ("inainte:proj", 1024, 768, 3072),
    ("inapoi:fc-dX", 1024, 768, 3072),
];

/// V2: the model on the step: os-418 (b = 1) and os-431 (b = 2), totals within 4 %, each tag within 7 %; and the tags
/// with 48 blocks did not move.
#[test]
fn t238c_model_on_the_step() {
    let forms = forms_513();
    assert_eq!(forms.len(), 15);
    for (_, m, n, k) in TAGS {
        assert!(
            forms.iter().any(|f| (f.m, f.n, f.k) == (m, n, k)),
            "{m}x{n}x{k}"
        );
    }
    let a = the_513_profile(OS418);
    let b = the_513_profile(OS431);
    assert_eq!((b.steps, b.total_us, b.launches), (10, 255_896, 483));
    assert!(OS431.contains("treapta pas: 64 KiB pe 483 lansari"));
    assert!(OS431.contains("TRECE (pasul #513 cu arena in VRAM = oracolul"));
    let name = VARIANTS[RACE_WINNER_513].name;
    let (ma, mb) = (a.program(name).unwrap(), b.program(name).unwrap());
    assert_eq!((ma.us, mb.us, mb.launches), (294_946, 223_234, 147));
    // Nothing else on the plate moved (≤ 0.5 ms).
    assert!((a.total_us - ma.us).abs_diff(b.total_us - mb.us) <= 500);

    let step = |bps: u32| -> u64 {
        forms
            .iter()
            .map(|f| wm::launch_ns(RACE_WINNER_513, f.m, f.n, f.k, bps).unwrap() * f.count as u64)
            .sum::<u64>()
            * STEP_OVER_RACE_PERMILLE
            / 1000
            / 1000
    };
    let (pa, pb) = (step(1), step(2));
    for (model, board) in [(pa, ma.us), (pb, mb.us)] {
        assert!(
            model.abs_diff(board) * 100 <= board * 4,
            "model {model} us, board {board} us"
        );
    }
    let mut worst = 0u64;
    for (tag, m, n, k) in TAGS {
        let (ta, tb) = (a.tag(tag).unwrap(), b.tag(tag).unwrap());
        assert_eq!((ta.launches, tb.launches), (12, 12));
        for (bps, board_us) in [(1u32, ta.us / 12), (2, tb.us / 12)] {
            let model_us = wm::launch_ns(RACE_WINNER_513, m, n, k, bps).unwrap()
                * STEP_OVER_RACE_PERMILLE
                / 1000
                / 1000;
            let pm = model_us.abs_diff(board_us) * 1000 / board_us;
            worst = worst.max(pm);
            assert!(
                pm <= 70,
                "{tag} b={bps}: model {model_us} us, board {board_us} us"
            );
        }
        if wm::blocks(RACE_WINNER_513, m, n) == 48 {
            assert!(
                ta.us.abs_diff(tb.us) * 100 <= ta.us,
                "{tag}: 48 blocks, no gain"
            );
        }
    }
    println!(
        "kt t238c mm_tv_128x128 in pas: os-418 placa {} us (model {pa}), os-431 {} us (model {pb}); etichete: abaterea maxima {worst} promile",
        ma.us, mb.us
    );
}

/// V3 + V4: the model's pick at 64 KiB and the predicted #513 step (the prediction the next board run judges).
#[test]
fn t238d_the_pick_at_64_kib() {
    let forms = forms_513();
    let (pick, race_ns) = wm::predicted_choice(&forms, KIB64).unwrap();
    assert_eq!(pick.entries().count(), 15);
    // The step with the model's pick: 3 matmul programs (the witness + 2 variants), still one 64 KiB step.
    assert_eq!(pick.programs().len(), 2);
    let (plan, opt) = plan_513();
    let list = plan
        .with_mm_choice(pick)
        .step(Recipe::AdamW513, &opt, 1)
        .unwrap();
    assert_eq!(list.len(), 483 - 98);
    assert_eq!(mv::forms(&list).iter().map(|f| f.count).sum::<u32>(), 147);
    assert_eq!(
        nvidia_ga104::train_step::step_best_sm_step(
            &list,
            nvidia_ga104::occupancy::Reserve::OneKib
        ),
        KIB64
    );
    let mm_us = race_ns * STEP_OVER_RACE_PERMILLE / 1000 / 1000;
    for f in &forms {
        let v = pick.of(f.m, f.n, f.k, f.flags);
        let g128 = wm::blocks(RACE_WINNER_513, f.m, f.n);
        if g128 >= 192 && f.k <= 1024 && (f.n % 3072 == 0 || f.n == 16_384 || f.m == 16_384) {
            assert_eq!(v, RACE_WINNER_513, "{f:?}: many blocks → 128x128");
        }
        if f.m == 1024 && f.n == 768 {
            assert_ne!(v, RACE_WINNER_513, "{f:?}: 48 blocks → a 64×64 variant");
        }
        println!(
            "kt t238d {}x{}x{} f{} x{} -> {} ({} us in pas)",
            f.m,
            f.n,
            f.k,
            f.flags,
            f.count,
            VARIANTS[v].name,
            wm::launch_ns_at(v, f.m, f.n, f.k, KIB64).unwrap() * STEP_OVER_RACE_PERMILLE
                / 1000
                / 1000
        );
    }
    // The rest of the step (plate without matmuls + outside the plate) is os-431's.
    let b = the_513_profile(OS431);
    let rest_plate = b.total_us - b.program(VARIANTS[RACE_WINNER_513].name).unwrap().us;
    let outside = 266_847 - b.total_us;
    let step_us = mm_us + rest_plate + outside;
    // V4: ~204 ms, the interval 195–215 ms.
    assert!((195_000..=215_000).contains(&step_us), "step {step_us} us");
    println!(
        "kt t238d alegerea modelului la 64 KiB: inmultirile ~{mm_us} us, pasul ~{step_us} us = ~{} simb/s (DEDUS)",
        1024 * 1_000_000 / step_us
    );
}
