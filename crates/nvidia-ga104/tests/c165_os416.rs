//! T225 (C-165, `docs/C-165_CURSA_MM.md` §„os-416”): the full night race of os-416 (15 shapes × 4 variants, the real
//! board report, rows 214–292 of `ciclu-auto-20261125-0135.txt`, copied unchanged into `tests/rapoarte/os416-cursa.txt`)
//! read with `parse_race`, the pick, the saving, and the #513 plan with the pick. On the processor: the board times
//! are the board's (measured by the OS); nothing here runs on (or simulates) the board.

use gpu_compute::train::{self, AdamW, Recipe};
use nvidia_ga104::mm_train_variants::{
    self as mv, MmChoice, RACE_WINNER_513, VARIANTS, parse_race,
};
use nvidia_ga104::train_step::{Plan, Prog};

const HEAP: u64 = nvidia_ga104::mmu::VA_BASE + 0x100_0000;
const REPORT: &str = include_str!("rapoarte/os416-cursa.txt");
/// What the OS printed: "pasul ar scadea cu ~141387 us" (from its unrounded times; DEDUS).
const OS_SAVED_US: u64 = 141_387;
/// The board's step time in os-416 (median of 10): plate 463 740 µs, whole step 474 176 µs.
const BOARD_PLATE_US: u64 = 463_740;
const BOARD_STEP_US: u64 = 474_176;

fn plan_513() -> (Plan, Vec<u32>, Vec<mv::Form>, AdamW) {
    let (bsz, tl) = (2usize, 512usize);
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let tile = Plan::new_tiles3(train::M513, bsz, tl, &inputs, HEAP)
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
    let forms = mv::forms(&tile.step(Recipe::AdamW513, &opt, 1).unwrap());
    (tile, inputs, forms, opt)
}

/// Predictions 1–4: 60 rows, 15 shapes = the #513 step's, all on `mm_tv_128x128`, the OS's saving.
#[test]
fn t225_the_real_os416_race() {
    let p = parse_race(REPORT);
    assert_eq!(p.rows.len(), 60, "timed rows");
    assert_eq!((p.untimed, p.unknown), (0, 0));
    assert!(p.rows.iter().all(|r| r.same_bits), "all = the witness");
    assert_eq!(p.counts.len(), 15);
    assert_eq!(p.counts.iter().map(|f| f.count).sum::<u32>(), 147);

    let (_, _, forms, _) = plan_513();
    assert_eq!(
        p.counts, forms,
        "the race's shapes = the #513 step's, in step order"
    );

    let pick = p.choice().unwrap();
    for f in &forms {
        assert_eq!(pick.of(f.m, f.n, f.k, f.flags), RACE_WINNER_513, "{f:?}");
    }
    assert_eq!(pick.entries().count(), 15);
    assert_eq!(pick.programs().len(), 1);
    assert_eq!(VARIANTS[RACE_WINNER_513].name, "mm_tv_128x128");
    assert_eq!(pick, MmChoice::all(&forms, RACE_WINNER_513).unwrap());

    let saved_us = p.saved_ns(&pick) / 1000;
    assert!(
        saved_us.abs_diff(OS_SAVED_US) <= 20,
        "ours {saved_us} us, the OS {OS_SAVED_US} us"
    );

    // Σ witness × launches and Σ winner × launches (isolated board times).
    let t = |v: usize| -> u64 {
        p.counts
            .iter()
            .map(|f| {
                let r = p
                    .rows
                    .iter()
                    .find(|r| (r.m, r.n, r.k, r.flags, r.variant) == (f.m, f.n, f.k, f.flags, v))
                    .unwrap();
                r.median_ns * u64::from(f.count)
            })
            .sum::<u64>()
            / 1000
    };
    let (w, win) = (t(0), t(RACE_WINNER_513));
    assert!(
        w <= BOARD_PLATE_US,
        "the products alone cannot be longer than the plate's step"
    );
    let rest = BOARD_PLATE_US - w;
    let predicted = BOARD_STEP_US - saved_us;
    println!(
        "kt t225 os-416: 60 randuri, 15 forme, 147 lansari; alegerea = mm_tv_128x128 pe toate 15; \
         economia {saved_us} us (OS: {OS_SAVED_US}); inmultirile martor {w} us = {}% din placa ({BOARD_PLATE_US} us), \
         restul {rest} us; cu 128x128 inmultirile {win} us; pasul prezis {predicted} us = {} simb/s (DEDUS)",
        w * 100 / BOARD_PLATE_US,
        1024 * 1_000_000 / predicted
    );
    // The race ran at 1 785 MHz, the steps at 1 950 (os-414): scale the in-step products (os-412 profile,
    // `mm_train_tile` 430.5 ms) by the race's winner / witness ratio instead (DEDUS).
    let in_step_mm_us = 430_500u64;
    let in_step_saved = in_step_mm_us - in_step_mm_us * win / w;
    let predicted2 = BOARD_STEP_US - in_step_saved;
    println!(
        "kt t225 cu raportul 128x128/martor ({} promile) pe inmultirile din pas (430.5 ms): economia {in_step_saved} us, \
         pasul prezis {predicted2} us = {} simb/s (DEDUS)",
        win * 1000 / w,
        1024 * 1_000_000 / predicted2
    );
    // Smallest gain among the shapes (the pick is not marginal anywhere).
    let min_gain_pm = p
        .counts
        .iter()
        .map(|f| {
            let g = |v: usize| {
                p.rows
                    .iter()
                    .find(|r| (r.m, r.n, r.k, r.flags, r.variant) == (f.m, f.n, f.k, f.flags, v))
                    .unwrap()
                    .median_ns
            };
            1000 - g(RACE_WINNER_513) * 1000 / g(0)
        })
        .min()
        .unwrap();
    println!(
        "kt t225 castigul cel mai mic pe o forma: {} promile",
        min_gain_pm
    );
    assert!(min_gain_pm >= mv::MIN_GAIN_PERMILLE);
}

/// Prediction 5: the #513 plan with the pick moves exactly the 147 products, nothing else.
#[test]
fn t225_the_plan_with_the_os416_pick() {
    let (tile, inputs, forms, opt) = plan_513();
    let pick = parse_race(REPORT).choice().unwrap();
    let chosen = Plan::new_tiles3(train::M513, 2, 512, &inputs, HEAP)
        .unwrap()
        .with_mm_choice(pick)
        .resident()
        .unwrap();
    let lt = tile.step(Recipe::AdamW513, &opt, 1).unwrap();
    let lc = chosen.step(Recipe::AdamW513, &opt, 1).unwrap();
    assert_eq!(lt.len(), lc.len());
    assert_eq!(chosen.bytes(), tile.bytes());
    let moved: Vec<_> = lt
        .iter()
        .zip(&lc)
        .filter(|(a, b)| a.prog != b.prog)
        .collect();
    assert_eq!(
        moved.len() as u32,
        forms.iter().map(|f| f.count).sum::<u32>()
    );
    assert_eq!(moved.len(), 147);
    let winner = Prog::of_race_index(RACE_WINNER_513).unwrap();
    assert!(moved.iter().all(|(_, b)| b.prog == winner));
    assert!(
        lc.iter()
            .all(|l| l.prog != Prog::MmTrain && l.prog != Prog::MmTrainTile)
    );
    assert_eq!(chosen.mm_choice_programs(), &[winner]);
}

/// `MmChoice::all`: refusals and the witness.
#[test]
fn t225_choice_all() {
    let (_, _, forms, _) = plan_513();
    assert_eq!(MmChoice::all(&forms, 0).unwrap(), MmChoice::new());
    assert!(MmChoice::all(&forms, VARIANTS.len()).is_err());
    let bad = [mv::Form {
        m: 0,
        n: 1,
        k: 1,
        flags: 0,
        count: 1,
    }];
    assert!(MmChoice::all(&bad, RACE_WINNER_513).is_err());
}
