//! T239 (C-157 p.2, `docs/C-157_P2_OS436.md` §4): os-436, the #513 step with the batch 6 × 512 at the 64 KiB step (all
//! 15 shapes on `mm_tv_128x128`) — the board's rows copied unchanged into `tests/rapoarte/os436-lot6.txt` — against the
//! "waves" model ([`nvidia_ga104::wave_model`], unchanged since `8fe7637`, i.e. before this run). The thresholds are
//! the ones written for V2 in `docs/C-165_VALURI.md` (total ≤ 4 %, each tag ≤ 7 %). On the processor: the board times
//! are the board's; the model's numbers are DEDUS; nothing here runs on (or simulates) the board.

use gpu_compute::train::{self, AdamW, Recipe};
use nvidia_ga104::mm_train_variants::{self as mv, RACE_WINNER_513, VARIANTS};
use nvidia_ga104::train_profile::{ParsedProfile, parse_profiles};
use nvidia_ga104::train_step::Plan;
use nvidia_ga104::wave_model::{self as wm, STEP_OVER_RACE_PERMILLE};

const LOT6: &str = include_str!("rapoarte/os436-lot6.txt");
const OS431: &str = include_str!("rapoarte/os431-pas.txt");
const HEAP: u64 = nvidia_ga104::mmu::VA_BASE + 0x100_0000;
const KIB64: u32 = 64 << 10;
/// os-436: the whole step (median of 10), µs.
const STEP_US: u64 = 656_392;

fn profile(text: &str) -> ParsedProfile {
    parse_profiles(text)
        .into_iter()
        .find(|p| p.context.contains("antrenare #513"))
        .expect("the #513 profile")
}

fn plan(bsz: usize) -> (Plan, AdamW) {
    let tl = 512usize;
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

fn forms(bsz: usize) -> Vec<mv::Form> {
    let (p, opt) = plan(bsz);
    mv::forms(&p.step(Recipe::AdamW513, &opt, 1).unwrap())
}

/// µs in the step of one product on `variant` with `b` blocks per SM (model, DEDUS).
fn step_us(variant: usize, m: u32, n: u32, k: u32, b: u32) -> u64 {
    wm::launch_ns(variant, m, n, k, b).unwrap() * STEP_OVER_RACE_PERMILLE / 1000 / 1000
}

/// The 7 tags of T238 at the batch 6: the row count (1024 → 3072) moves to M for the forward / dX products and to K
/// for the dW products.
const TAGS6: [(&str, u32, u32, u32); 7] = [
    ("inainte:fc", 3072, 3072, 768),
    ("inapoi:proj-dX", 3072, 3072, 768),
    ("inapoi:proj-dW", 768, 3072, 3072),
    ("inapoi:fc-dW", 3072, 768, 3072),
    ("inapoi:qkv-dW", 2304, 768, 3072),
    ("inainte:proj", 3072, 768, 3072),
    ("inapoi:fc-dX", 3072, 768, 3072),
];

/// The board's rows: 10 steps, 483 launches, all matmuls on 128x128, the 64 KiB step; not a pass (the synthetic
/// batches, as os-395 — the criterion, not the plate).
#[test]
fn t239a_the_rows() {
    let p = profile(LOT6);
    assert_eq!((p.steps, p.total_us, p.launches), (10, 637_873, 483));
    assert!(LOT6.contains("lot 6 x 512 (--lot-513 6)"));
    assert!(LOT6.contains("treapta pas: 64 KiB pe 483 lansari"));
    assert!(LOT6.contains("0 care nu sunt numere"));
    let mm = p.program(VARIANTS[RACE_WINNER_513].name).unwrap();
    assert_eq!((mm.us, mm.launches), (560_816, 147));
    // The clock during the steps (the dashboard): 1 950 MHz, P0, no slowdown — the same as os-414 / os-431.
    assert!(LOT6.contains("kt tablou pasi-513 ceasul nucleului median MHz 1950"));
    assert!(LOT6.contains("kt tablou pasi-513 motivele incetinirii (sau pe biti) 0"));
}

/// §4 V6: the unchanged model at b = 2 on the batch-6 shapes — total within 4 %, each of the 7 tags within 7 %.
#[test]
fn t239b_the_model_on_the_batch_6() {
    let f6 = forms(6);
    assert_eq!(f6.len(), 15);
    assert_eq!(f6.iter().map(|f| f.count).sum::<u32>(), 147);
    for (tag, m, n, k) in TAGS6 {
        assert!(
            f6.iter().any(|f| (f.m, f.n, f.k) == (m, n, k)),
            "{tag}: {m}x{n}x{k}"
        );
    }
    let p = profile(LOT6);
    let board = p.program(VARIANTS[RACE_WINNER_513].name).unwrap().us;
    let model: u64 = f6
        .iter()
        .map(|f| step_us(RACE_WINNER_513, f.m, f.n, f.k, 2) * f.count as u64)
        .sum();
    let pm = model.abs_diff(board) * 1000 / board;
    println!("kt t239b mm_tv_128x128 lot 6: placa {board} us, modelul {model} us ({pm} promile)");
    let mut worst = 0u64;
    for (tag, m, n, k) in TAGS6 {
        let t = p.tag(tag).unwrap();
        assert_eq!(t.launches, 12);
        let board_us = t.us / 12;
        let model_us = step_us(RACE_WINNER_513, m, n, k, 2);
        let tp = model_us.abs_diff(board_us) * 1000 / board_us;
        worst = worst.max(tp);
        println!(
            "kt t239b {tag} {m}x{n}x{k}: {} blocuri, placa {board_us} us, modelul {model_us} us ({tp} promile)",
            wm::blocks(RACE_WINNER_513, m, n)
        );
        assert!(tp <= 70, "{tag}: model {model_us} us, board {board_us} us");
    }
    assert!(pm <= 40, "total: model {model} us, board {board} us");
    println!("kt t239b etichete: abaterea maxima {worst} promile");
}

/// Batch 6 against batch 2 (os-431, same step, same program): the matmuls ×3 in work. The shapes that had 48 blocks
/// at batch 2 (1024 × 768) have 144 at batch 6 and stop being "one block per SM".
#[test]
fn t239c_batch_6_against_batch_2() {
    let (a, b) = (profile(OS431), profile(LOT6));
    let name = VARIANTS[RACE_WINNER_513].name;
    let (ma, mb) = (a.program(name).unwrap().us, b.program(name).unwrap().us);
    let per_token_permille = mb * 1000 / (3 * ma);
    println!(
        "kt t239c inmultirile: lot 2 {ma} us, lot 6 {mb} us = x{}.{:03} pentru x3 lucru ({per_token_permille} promile pe simbol)",
        mb / ma,
        mb * 1000 / ma % 1000
    );
    assert!(per_token_permille < 900, "batch 6 is cheaper per token");
    for tag6 in ["inainte:proj", "inapoi:fc-dX"] {
        let (t2, t6) = (a.tag(tag6).unwrap().us, b.tag(tag6).unwrap().us);
        assert_eq!(wm::blocks(RACE_WINNER_513, 1024, 768), 48);
        assert_eq!(wm::blocks(RACE_WINNER_513, 3072, 768), 144);
        println!(
            "kt t239c {tag6}: lot 2 {} us, lot 6 {} us pe lansare = x{}.{:03}",
            t2 / 12,
            t6 / 12,
            t6 / t2,
            t6 * 1000 / t2 % 1000
        );
        // 3× the work in the same 2 waves (48 → 144 blocks at 96 per wave): far less than ×3.
        assert!(t6 * 10 < t2 * 25, "{tag6}: ×{}", t6 as f64 / t2 as f64);
    }
    let (ra, rb) = (a.total_us - ma, b.total_us - mb);
    println!(
        "kt t239c restul placii: lot 2 {ra} us, lot 6 {rb} us = x{}.{:03}",
        rb / ra,
        rb * 1000 / ra % 1000
    );
}

/// §5 V7 (the prediction for the next batch-6 run): the model's pick per shape at 64 KiB on the batch-6 shapes, and
/// the step it predicts. Written before any such run.
#[test]
fn t239d_the_pick_at_batch_6() {
    let f6 = forms(6);
    let (pick, race_ns) = wm::predicted_choice(&f6, KIB64).unwrap();
    assert_eq!(pick.entries().count(), 15);
    let b = profile(LOT6);
    let now_mm = b.program(VARIANTS[RACE_WINNER_513].name).unwrap().us;
    let model_now: u64 = f6
        .iter()
        .map(|f| step_us(RACE_WINNER_513, f.m, f.n, f.k, 2) * f.count as u64)
        .sum();
    let mm_us = race_ns * STEP_OVER_RACE_PERMILLE / 1000 / 1000;
    for f in &f6 {
        let v = pick.of(f.m, f.n, f.k, f.flags);
        println!(
            "kt t239d {}x{}x{} f{} x{} -> {} ({} us in pas; 128x128: {} us)",
            f.m,
            f.n,
            f.k,
            f.flags,
            f.count,
            VARIANTS[v].name,
            wm::launch_ns_at(v, f.m, f.n, f.k, KIB64).unwrap() * STEP_OVER_RACE_PERMILLE
                / 1000
                / 1000,
            step_us(RACE_WINNER_513, f.m, f.n, f.k, 2)
        );
    }
    // The model's own gain, applied to the board's matmul time (the model's error cancels to first order).
    let mm_pred = now_mm * mm_us / model_now;
    let rest = b.total_us - now_mm + (STEP_US - b.total_us);
    let step = mm_pred + rest;
    println!(
        "kt t239d alegerea la 64 KiB, lot 6: inmultirile ~{mm_pred} us (azi {now_mm}), pasul ~{step} us = ~{} simb/s (azi {} us = {} simb/s) (DEDUS)",
        3072 * 1_000_000 / step,
        STEP_US,
        3072 * 1_000_000 / STEP_US
    );
    assert!(mm_pred < now_mm);
    // V7: ~554 ms (interval 530–580 ms) ≈ 5 300–5 800 simb/s.
    assert!((530_000..=580_000).contains(&step), "step {step} us");
}

/// §1 + §3: the `mm_tc*` programs (24 KiB) launched with `pack` sit at the 32 KiB step = 1 block per SM; at 64 KiB all
/// three fit twice (registers 64 / 248 / 128), and the QMD accepts that step for their shared memory.
#[test]
fn t239e_mm_tc_occupancy_and_the_64_kib_step() {
    use nvidia_ga104::occupancy::{Kernel, Reserve, SM86, blocks_per_sm, mm_tc_kernel};
    use nvidia_ga104::qmd::sm_config_shared_units;
    use nvidia_ga104::{mm_tc, mm_tc_pf, mm_tc_v4};
    let tc = mm_tc_kernel();
    let pf = Kernel {
        name: "mm_tc_pf",
        registers: mm_tc_pf::REGISTERS,
        ..tc
    };
    let v4 = Kernel {
        name: "mm_tc_v4",
        registers: mm_tc_v4::REGISTERS,
        ..tc
    };
    // `pack`: the smallest step that holds 24 KiB = 32 KiB.
    assert_eq!(
        sm_config_shared_units(mm_tc::SHARED.bytes),
        sm_config_shared_units(32 << 10)
    );
    for k in [tc, pf, v4] {
        let at = |s: u32| blocks_per_sm(&SM86, &k, s, Reserve::None).unwrap().blocks;
        assert_eq!((at(32 << 10), at(KIB64)), (1, 2), "{}", k.name);
    }
    // qkv-513 of the probe: 6144 × 2304 → 3 456 blocks of 64 × 64 = 72 waves at 1 block per SM.
    let a = mm_tc::Args {
        a: 0,
        b: 0,
        c: 0,
        bias: 0,
        m: 6144,
        n: 2304,
        k: 768,
        flags: 0,
    };
    let g = a.grid();
    assert_eq!(g[0] * g[1], 3456);
    assert_eq!((g[0] * g[1]).div_ceil(wm::SMS), 72);
    // The QMD takes the 64 KiB step for the 24 KiB programs (the call the OS would make).
    let q = nvidia_ga104::qmd::ComputeQmd {
        program_va: 0x1_0000_0000,
        registers: mm_tc_pf::REGISTERS,
        cbuf0_va: 0x1_0010_0000,
        cbuf0_size: 0x800,
        grid: g,
        block: mm_tc_pf::BLOCK,
        release: None,
    };
    assert!(q.to_bytes_with_sm_config(&mm_tc_pf::SHARED, KIB64).is_ok());
    assert!(q.to_bytes_with_sm_config(&mm_tc_v4::SHARED, KIB64).is_ok());
}
