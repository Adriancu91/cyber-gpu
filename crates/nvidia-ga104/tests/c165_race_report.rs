//! T221 (C-165 step 3, `docs/C-165_CURSA_MM.md` §„Pasul 3”): the race report reader `parse_race`, on the processor.
//! The report text is made here with the OS's own row templates (cyber-universe P6-190, `gpu_i14.rs`: the header,
//! `race_row`, the "fastest" line and the summary), so the reader is checked against the exact format the board's
//! report will have. Nothing here runs on (or simulates) the board: it is text in, rows and a pick out.

use gpu_compute::train::{self, AdamW, Recipe};
use nvidia_ga104::mm_train_variants::{self as mv, MmChoice, RaceRow, VARIANTS, parse_race};
use nvidia_ga104::train_step::Plan;

const HEAP: u64 = nvidia_ga104::mmu::VA_BASE + 0x100_0000;
const P: &str = "gpu";
const TAG: &str = "antrenare-513";

/// The OS's `race_row` (P6-190), copied: `cursa: varianta | forma | us median | TFLOP/s | biti`.
fn os_row(name: &str, f: &mv::Form, ns: Option<u64>, bits: &str) -> String {
    let flops = 2 * u64::from(f.m) * u64::from(f.n) * u64::from(f.k);
    let (us, tf) = match ns {
        Some(n) if n > 0 => (
            format!("{}.{} us", n / 1000, (n % 1000) / 100),
            format!(
                "{}.{:02} TFLOP/s",
                flops / n / 1000,
                (flops * 100 / n / 1000) % 100
            ),
        ),
        _ => (String::from("- us"), String::from("- TFLOP/s")),
    };
    let row = format!(
        "cursa: {} | {}x{}x{} f{} (x{} in pas) | {} median | {} | biti {}",
        name, f.m, f.n, f.k, f.flags, f.count, us, tf, bits
    );
    format!("{}: kronos --gpu: {}: {}", P, TAG, row)
}

fn os_header(forms: usize, launches: u32) -> String {
    format!(
        "{}: kronos --gpu: {}: cursa mm (C-165, 5 reps): {} forme ale lui mm_train din pas ({} lansari) x 5 variante (mm_train_tile, ...); martorul mm_train_tile (trecut pe placa la os-410); C pe o bucata de 64 MiB in VRAM + 256 octeti de paza",
        P, TAG, forms, launches
    )
}

fn os_fastest(f: &mv::Form, name: &str, t: u64, w: u64) -> String {
    format!(
        "{}: kronos --gpu: {}: cursa: forma {}x{}x{} f{}: cea mai rapida cu aceiasi biti = {} ({}.{} us; martorul {}.{} us)",
        P,
        TAG,
        f.m,
        f.n,
        f.k,
        f.flags,
        name,
        t / 1000,
        (t % 1000) / 100,
        w / 1000,
        (w % 1000) / 100
    )
}

/// The #513 step of os-410 / os-412 (`new_tiles3`, `with_mm_tile`, resident) and its 15 `mm_train` shapes.
fn plan_513() -> (Plan, Vec<u32>, Vec<mv::Form>, AdamW) {
    let (bsz, tl) = (2usize, 512usize);
    let ar = train::M513;
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let tile = Plan::new_tiles3(ar, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile()
        .resident()
        .unwrap();
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    let forms = mv::forms(&tile.step(Recipe::AdamW513, &opt, 1).unwrap());
    (tile, inputs, forms, opt)
}

/// A made-up race (no board): times in ns that differ by more than 0.1 µs between variants, so the report's rounding
/// cannot change the pick. On shape i: variant 1 + i % 4 is ~12 % faster, another ~1 % (under the 3 % rule); shape 0's
/// fastest changed the bits; shape 1's variant 4 was skipped (`- us`); shape 2 has an unfinished row.
fn made_up(forms: &[mv::Form]) -> (Vec<RaceRow>, String) {
    let mut rows = Vec::new();
    let mut text = Vec::new();
    text.push(String::from(
        "gpu: kronos --gpu: tablou: P0, 1 905 MHz, 61 C (inainte de cursa)",
    ));
    let launches: u32 = forms.iter().map(|f| f.count).sum();
    text.push(os_header(forms.len(), launches));
    for (i, f) in forms.iter().enumerate() {
        let w = 1_000_000 + 37_000 * i as u64 + 456; // 456 ns: lost by the report's rounding
        let best = 1 + i % 4;
        let other = 1 + (i + 1) % 4;
        let mut fastest = (0usize, w);
        for (v, var) in VARIANTS.iter().enumerate() {
            let (ns, same) = match v {
                0 => (Some(w), true),
                _ if v == best && i == 0 => (Some(w / 2), false),
                _ if v == best => (Some(w * 88 / 100), true),
                _ if v == other => (Some(w * 99 / 100), true),
                4 if i == 1 => (None, false),
                _ => (Some(w * 120 / 100), true),
            };
            let bits = match (ns, same) {
                (None, _) => "SARIT",
                (Some(_), true) => "=",
                (Some(_), false) => "DIFERA",
            };
            if i == 2 && v == 4 {
                text.push(os_row(var.name, f, None, "NETERMINAT"));
                continue;
            }
            text.push(os_row(var.name, f, ns, bits));
            if let Some(t) = ns {
                // The truth, with full nanoseconds.
                rows.push(RaceRow {
                    m: f.m,
                    n: f.n,
                    k: f.k,
                    flags: f.flags,
                    variant: v,
                    median_ns: t,
                    same_bits: same,
                });
                if same && t < fastest.1 {
                    fastest = (v, t);
                }
            }
        }
        if fastest.0 > 0 {
            text.push(os_fastest(f, VARIANTS[fastest.0].name, fastest.1, w));
        }
        text.push(format!(
            "gpu: kronos --gpu: tablou in timpul cursei (dupa forma {} din {}): P0, 1 860 MHz, 70 C",
            i + 1,
            forms.len()
        ));
    }
    text.push(format!(
        "gpu: kronos --gpu: {}: cursa mm: gata: 70 randuri cu aceiasi biti, 3 diferite; cu castigatoarele pe fiecare forma pasul ar scadea cu ~9000 us (din timpii izolati; DEDUS)",
        TAG
    ));
    (rows, text.join("\n"))
}

/// Predictions 1, 2 and 4: the 15 × 5 race of #513 in the OS's format → 75 − 2 timed rows, the times rounded to
/// 0.1 µs, the pick = the pick from the true rows, and `with_mm_choice` swaps exactly the chosen shapes.
#[test]
fn t221_the_513_race_report_gives_the_same_pick_as_the_true_rows() {
    let (tile, inputs, forms, opt) = plan_513();
    assert_eq!(forms.len(), 15);
    let (truth, text) = made_up(&forms);
    let p = parse_race(&text);
    assert_eq!(p.rows.len(), truth.len(), "timed rows");
    assert_eq!(p.rows.len(), 15 * 5 - 2);
    assert_eq!((p.untimed, p.unknown), (2, 0));
    assert_eq!(
        p.counts, forms,
        "shapes and launches per step, in step order"
    );
    for (a, b) in p.rows.iter().zip(&truth) {
        assert_eq!(
            (a.m, a.n, a.k, a.flags, a.variant, a.same_bits),
            (b.m, b.n, b.k, b.flags, b.variant, b.same_bits)
        );
        assert_eq!(a.median_ns, b.median_ns / 100 * 100, "rounded to 0.1 µs");
    }
    let pick = p.choice().unwrap();
    assert_eq!(pick, MmChoice::from_race(&truth).unwrap());
    for (i, f) in forms.iter().enumerate() {
        // Shape 0: its fastest changed the bits; the next one (1 %) is under the rule → the witness stays.
        let want = if i == 0 { 0 } else { 1 + i % 4 };
        assert_eq!(pick.of(f.m, f.n, f.k, f.flags), want, "shape {i}: {f:?}");
    }
    // The saving the OS would print (DEDUS), from the rounded report times: Σ (w − 0.88 w) × count over shapes 1..15.
    let saved = p.saved_ns(&pick);
    let expect: u64 = forms
        .iter()
        .enumerate()
        .skip(1)
        .map(|(i, f)| {
            let w = (1_000_000 + 37_000 * i as u64 + 456) / 100 * 100;
            let c = ((1_000_000 + 37_000 * i as u64 + 456) * 88 / 100) / 100 * 100;
            (w - c) * u64::from(f.count)
        })
        .sum();
    assert_eq!(saved, expect);
    assert_eq!(p.saved_ns(&MmChoice::new()), 0);
    // The plan with the pick: only the chosen shapes move (as T219), same arena, same number of launches.
    let chosen = Plan::new_tiles3(train::M513, 2, 512, &inputs, HEAP)
        .unwrap()
        .with_mm_choice(pick)
        .resident()
        .unwrap();
    let lt = tile.step(Recipe::AdamW513, &opt, 1).unwrap();
    let lc = chosen.step(Recipe::AdamW513, &opt, 1).unwrap();
    assert_eq!((lt.len(), lc.len()), (385, 385));
    assert_eq!(chosen.bytes(), tile.bytes());
    let moved = lt.iter().zip(&lc).filter(|(a, b)| a.prog != b.prog).count();
    let expect_moved: u32 = forms.iter().skip(1).map(|f| f.count).sum();
    assert_eq!(moved as u32, expect_moved);
}

/// Prediction 3: two races in one journal → only the last one is read.
#[test]
fn t221_only_the_last_race_is_kept() {
    let (_, _, forms, _) = plan_513();
    let (_, first) = made_up(&forms);
    let f = forms[3];
    let second = [
        os_header(1, f.count),
        os_row("mm_train_tile", &f, Some(2_000_000), "="),
        os_row("mm_tv_128x128", &f, Some(1_500_000), "="),
    ]
    .join("\n");
    let p = parse_race(&format!("{first}\n{second}"));
    assert_eq!(p.rows.len(), 2);
    assert_eq!(p.counts, vec![f]);
    let pick = p.choice().unwrap();
    assert_eq!(pick.of(f.m, f.n, f.k, f.flags), 3);
    assert_eq!(pick.entries().count(), 1);
    assert_eq!(p.saved_ns(&pick), 500_000 * u64::from(f.count));
}

/// Predictions 2 and 5: unknown names, broken rows and other lines never become rows; nothing panics.
#[test]
fn t221_broken_and_foreign_rows_are_left_out() {
    let f = mv::Form {
        m: 1024,
        n: 768,
        k: 768,
        flags: 4,
        count: 36,
    };
    let good = os_row("mm_train_tile", &f, Some(812_345), "=");
    let lines = [
        good.clone(),
        os_row("mm_tv_256x256", &f, Some(700_000), "="),
        good.replace("1024x768x768", "1024x768"),
        good.replace(" f4 ", " fX "),
        good.replace("(x36", "(36"),
        good.replace("812.3 us", "81.2.3 us"),
        good.replace(" | biti =", ""),
        good.replace("biti =", "bitii ="),
        good.replace(" us median", " ms median"),
        String::from(
            "gpu: kronos --gpu: antrenare-513: cursa: forma 1024x768x768 f4: cea mai rapida cu aceiasi biti = mm_tv_128x64 (1.0 us; martorul 2.0 us)",
        ),
        String::from("cursa: "),
        String::from(": cursa:  |  |  |  | "),
        String::new(),
    ];
    let p = parse_race(&lines.join("\n"));
    assert_eq!(
        p.rows,
        vec![RaceRow {
            m: 1024,
            n: 768,
            k: 768,
            flags: 4,
            variant: 0,
            median_ns: 812_300,
            same_bits: true,
        }]
    );
    assert_eq!((p.untimed, p.unknown), (0, 1));
    assert_eq!(p.counts, vec![f]);
    // A witness alone: nothing to choose.
    assert_eq!(p.choice().unwrap(), MmChoice::new());
    // An empty or unrelated text.
    assert_eq!(parse_race(""), Default::default());
    assert_eq!(
        parse_race("gpu: valoare kt colsum seturi = 0x5\n"),
        Default::default()
    );
}
