//! T242 (C-157, centre note 25 Nov 11:55 request 2, `docs/C-157_PROGRAM_PERSISTENT.md`): the budget of a "persistent"
//! program on one layer, from the board's own numbers — os-431 (#513 step profile, rows copied unchanged into
//! `tests/rapoarte/os431-pas.txt`) and os-346 (generation, per-launch times). On the processor: the board times are
//! the board's (MĂSURAT by the OS); every derived number here is DEDUS; nothing runs on (or simulates) the board.

use gpu_compute::train::{self, AdamW, Recipe};
use nvidia_ga104::kronos_model::launches_per_token_fused;
use nvidia_ga104::persistent_study::{self as ps, Pair};
use nvidia_ga104::train_profile::{ParsedProfile, parse_profiles};
use nvidia_ga104::train_step::{Launch, Plan};
use nvidia_ga104::wave_model::{self as wm, STEP_OVER_RACE_PERMILLE};

const OS431: &str = include_str!("rapoarte/os431-pas.txt");
const HEAP: u64 = nvidia_ga104::mmu::VA_BASE + 0x100_0000;
const KIB64: u32 = 64 << 10;
/// `mm_tv_128x128`, the variant os-431 ran every product on.
const V128: usize = 3;

fn profile_513() -> ParsedProfile {
    parse_profiles(OS431)
        .into_iter()
        .find(|p| p.context.contains("antrenare #513"))
        .expect("the #513 profile")
}

fn step_513() -> Vec<Launch> {
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
    plan.step(Recipe::AdamW513, &AdamW::recipe_513(&params), 1)
        .unwrap()
}

/// P1: no forward intermediate of a #513 layer (lot 2 × 512) fits in the board's fast memory except the
/// LayerNorm / attention outputs (3 MiB) — a whole-layer program would still go through VRAM for qkv and the hidden layer.
#[test]
fn t242a_p1_intermediates_do_not_fit() {
    let fast = ps::fast_bytes();
    let items = ps::layer_intermediates(1024, 768);
    let fit: Vec<&str> = items
        .iter()
        .filter(|i| i.bytes <= fast)
        .map(|i| i.name)
        .collect();
    let big: Vec<&str> = items
        .iter()
        .filter(|i| i.bytes > fast)
        .map(|i| i.name)
        .collect();
    println!(
        "memoria rapida {} KiB; incap {fit:?}; nu incap {big:?}",
        fast >> 10
    );
    assert_eq!(fit, ["ln", "atentia"]);
    assert_eq!(big, ["qkv", "ascuns", "gelu"]);
    // lot 6: nothing fits.
    assert!(
        ps::layer_intermediates(6 * 512, 768)
            .iter()
            .all(|i| i.bytes > fast)
    );
}

/// P2 + P3: gaps and small programs, as shares of the os-431 step.
#[test]
fn t242b_p2_p3_gaps_and_small_programs() {
    let b = ps::TrainBudget::from_profile(&profile_513());
    println!(
        "pas {} us, {} lansari; goluri {} us ({} ‰); marunte {} us ({} ‰)",
        b.step_us,
        b.launches,
        b.gaps_us,
        b.permille(b.gaps_us),
        b.small_us,
        b.permille(b.small_us)
    );
    assert_eq!((b.step_us, b.launches), (255_896, 483));
    assert_eq!(b.gaps_us, 2_076);
    assert!(b.permille(b.gaps_us) <= 10, "P2");
    // colsum 845 + colsum_part 1 572 + gelu_tr 1 472 + vadd 832 + ln_fwd 581 + ln_back 654 + xent 522 + emb 78 + gnorm 734
    assert_eq!(b.small_us, 7_290);
    assert!(b.permille(b.small_us) <= 30, "P3");
}

/// The four backward pairs of every layer are found, with the shapes of the plan.
#[test]
fn t242c_pairs_of_a_layer() {
    let l = step_513();
    for layer in 0..12u16 {
        let p = ps::backward_pairs(&l, layer);
        let names: Vec<&str> = p.iter().map(|x| x.name).collect();
        assert_eq!(names, ["proj", "fc", "out", "qkv"], "layer {layer}");
    }
    let p = ps::backward_pairs(&l, 5);
    let fc = p.iter().find(|x| x.name == "fc").unwrap();
    assert_eq!((fc.dx.m, fc.dx.n, fc.dx.k), (1024, 768, 3072));
    assert_eq!((fc.dw.m, fc.dw.n, fc.dw.k), (3072, 768, 1024));
    assert!(ps::backward_pairs(&l, 12).is_empty());
}

fn pairs() -> Vec<Pair> {
    ps::backward_pairs(&step_513(), 0)
}

/// P4, part 1: on the os-431 run (all products on 128x128, 2 blocks per SM at 64 KiB), the dual floor of the four
/// pairs × 12 layers, at the step clock.
#[test]
fn t242d_p4_dual_launch_on_os431() {
    let b = wm::blocks_per_sm(V128, KIB64).unwrap();
    assert_eq!(b, 2);
    let (mut sep, mut dual) = (0u64, 0u64);
    for p in pairs() {
        let c = ps::pair_cost(V128, b, &p).unwrap();
        println!(
            "{}: separat {} us, dublu (podea) {} us, locuri goale {} ‰",
            p.name,
            c.separate_ns / 1000,
            c.dual_floor_ns / 1000,
            c.idle_permille
        );
        sep += c.separate_ns;
        dual += c.dual_floor_ns;
    }
    let step = |ns: u64| ns * 12 * STEP_OVER_RACE_PERMILLE / 1000 / 1000;
    let (sep_us, dual_us) = (step(sep), step(dual));
    let total = profile_513().total_us;
    let win = sep_us - dual_us;
    println!(
        "12 straturi, ceasul pasului: separat {sep_us} us, dublu {dual_us} us, castig {win} us = {} ‰ din pasul os-431",
        win * 1000 / total
    );
    // The model against the board on the same launches: os-431 printed only its 8 longest tags, 6 of them are these
    // products (out-dX / out-dW are shorter and were not printed).
    let prof = profile_513();
    let six = ["proj-dX", "proj-dW", "fc-dX", "fc-dW", "qkv-dX", "qkv-dW"];
    let board: u64 = six
        .iter()
        .map(|t| prof.tag(&format!("inapoi:{t}")).expect(t).us)
        .sum();
    let out = pairs().into_iter().find(|p| p.name == "out").unwrap();
    let model6 = sep_us - step(ps::pair_cost(V128, b, &out).unwrap().separate_ns);
    println!("cele 6 etichete: placa {board} us, modelul {model6} us");
    assert!(
        board.abs_diff(model6) * 100 <= 3 * board,
        "the model reproduces the board within 3 %"
    );
    // P4 predicted 5–15 % of the os-431 step: MISSED — the floor is 19.1 % (the 48-block dX forms leave more than
    // half of the slots idle). The test states what was found.
    let pm = win * 1000 / total;
    assert!((185..=195).contains(&pm), "P4 (os-431): {pm} ‰");
}

/// P4, part 2: on top of the per-form choice at 64 KiB (the model's best variant for every product).
#[test]
fn t242e_p4_dual_launch_on_the_per_form_choice() {
    let (mut sep, mut dual) = (0u64, 0u64);
    for p in pairs() {
        let (s, d) = ps::pair_best_at(&p, KIB64).unwrap();
        println!(
            "{}: separat (alegerea pe forma) {} us, dublu {} us",
            p.name,
            s / 1000,
            d / 1000
        );
        sep += s;
        dual += d.min(s);
    }
    let step = |ns: u64| ns * 12 * STEP_OVER_RACE_PERMILLE / 1000 / 1000;
    let win = step(sep) - step(dual);
    let total = profile_513().total_us;
    println!(
        "peste alegerea pe forma: separat {} us, dublu {} us, castig {win} us = {} ‰ din os-431",
        step(sep),
        step(dual),
        win * 1000 / total
    );
    let pm = win * 1000 / total;
    assert!((20..=80).contains(&pm), "P4 (pe forma): {pm} ‰");
}

/// P5 + P6: generation, one layer and one token of #513 (os-346).
#[test]
fn t242f_p5_p6_generation() {
    let g = ps::gen_layer(768);
    let n = launches_per_token_fused(12) as u64;
    let gaps = n * ps::GEN_GAP_NS;
    let floor =
        12 * g.floor_ns + 16_384 * 768 / 32 * 34 * 1000 / nvidia_ga104::kronos_perf::PAPER_MB_S;
    let measured_token = 12 * g.layer_ns + ps::OS346_LOGIT_NS;
    println!(
        "strat: {} ns masurat, goluri <= {} ns, podeaua latimii de banda {} ns",
        g.layer_ns, g.gaps_ns, g.floor_ns
    );
    println!(
        "simbol: {n} lansari, goluri <= {} us = {} ‰ din placa {} us; podeaua {} us (placa / podea = {:.1})",
        gaps / 1000,
        gaps * 1000 / ps::OS346_BOARD_NS,
        ps::OS346_BOARD_NS / 1000,
        floor / 1000,
        ps::OS346_BOARD_NS as f64 / floor as f64
    );
    assert_eq!(g.layer_ns, 80_090);
    assert_eq!(n, 87);
    // the 12 layers + logit account for the board time within 3 % (the rest: final LayerNorm, embedding, argmax)
    assert!(measured_token.abs_diff(ps::OS346_BOARD_NS) * 100 <= 3 * ps::OS346_BOARD_NS);
    let pm = gaps * 1000 / ps::OS346_BOARD_NS;
    assert!((300..=400).contains(&pm), "P5: {pm} ‰");
    let ratio = ps::OS346_BOARD_NS / floor;
    assert!((5..=7).contains(&ratio), "P6: x{ratio}");
}
