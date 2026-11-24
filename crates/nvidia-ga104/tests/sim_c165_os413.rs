//! T222 (C-165, os-413): why `mm_tv_64x64k8` did not finish on the board, checked on the CPU only (nothing is
//! simulated on a board here). The hypothesis (DEDUS): the instruction fetcher reads ahead past the end of the code;
//! the OS mapped every program rounded up to 4 KiB pages only, and `mm_tv_64x64k8` was the last program mapped, with
//! its code ending 16 bytes before the end of its pages. These tests check that this explains exactly the one variant
//! that stopped, that [`program_map_bytes`] leaves a whole page after every program, and the race list without it.

use nvidia_ga104::mm_train_variants::{RACE_HOLD, VARIANTS, in_race, race_list};
use nvidia_ga104::qmd::{PROGRAM_TAIL_BYTES, program_map_bytes, program_page_slack};

const PAGE: u64 = 4096;
/// Mesa nouveau/nvc0's figure ("a 2K (from NVIDIA) overallocation at the end to avoid prefetch bugs").
const NVIDIA_2K: u64 = 2048;

#[test]
fn t222_map_bytes() {
    assert_eq!(program_map_bytes(0), 0);
    assert_eq!(program_map_bytes(1), 2 * PAGE);
    assert_eq!(program_map_bytes(PAGE), 2 * PAGE);
    assert_eq!(program_map_bytes(16_368), 5 * PAGE);
    assert_eq!(program_page_slack(16_368), 16);
    assert_eq!(program_page_slack(PAGE), 0);
    for v in VARIANTS.iter() {
        let len = v.code.len() as u64;
        let map = program_map_bytes(len);
        assert!(map - len >= PROGRAM_TAIL_BYTES, "{}", v.name);
        assert_eq!(map % PAGE, 0, "{}", v.name);
    }
}

/// The os-413 race: the five programs mapped one after the other (each rounded up to 4 KiB, consecutive VAs, as
/// `map_vram` in the OS), nothing mapped after the last. Mapped bytes after the end of each program's code:
/// only `mm_tv_64x64k8` has fewer than 2 KiB — the one that stopped.
#[test]
fn t222_os413_layout_explains_only_the_last() {
    let mut after = Vec::new();
    for (i, v) in VARIANTS.iter().enumerate() {
        let len = v.code.len() as u64;
        let next: u64 = VARIANTS
            .get(i + 1)
            .map_or(0, |n| (n.code.len() as u64).div_ceil(PAGE) * PAGE);
        after.push((
            v.name,
            len,
            program_page_slack(len),
            program_page_slack(len) + next,
        ));
    }
    for (name, len, slack, mapped_after) in &after {
        println!(
            "kt t222 {name}: {len} B, rest in pagina {slack} B, mapat dupa cod {mapped_after} B"
        );
    }
    let short: Vec<&str> = after
        .iter()
        .filter(|a| a.3 < NVIDIA_2K)
        .map(|a| a.0)
        .collect();
    assert_eq!(short, ["mm_tv_64x64k8"]);
    assert_eq!(after[4].2, 16);
}

#[test]
fn t222_race_list_without_hold() {
    assert_eq!(RACE_HOLD, [4]);
    assert_eq!(VARIANTS[4].name, "mm_tv_64x64k8");
    let names: Vec<(usize, &str)> = race_list().map(|(i, v)| (i, v.name)).collect();
    assert_eq!(
        names,
        [
            (0, "mm_train_tile"),
            (1, "mm_tv_128x64"),
            (2, "mm_tv_64x128"),
            (3, "mm_tv_128x128")
        ]
    );
    assert!(!in_race(4));
    assert!(!in_race(VARIANTS.len()));
    assert!(in_race(0));
}

/// The race rows of the real os-413 report (`ciclu-auto-20261124-2223.txt`, rows 204–210, copied as they are):
/// `parse_race` reads 4 timed rows + 1 unfinished, and the pick on qkv (1024 × 2304 × 768, f6) is `mm_tv_128x128`
/// (1 889,8 µs against 2 838,3 µs: −33 %), saving ~11,4 ms on the 12 qkv launches of a step (DEDUS, isolated times).
#[test]
fn t222_parse_the_real_os413_report() {
    use nvidia_ga104::mm_train_variants::parse_race;
    let p = "kronos: lant: gpu0: kronos --gpu: antrenare #513 pe placa (arena in VRAM)";
    let text = format!(
        "{p}: cursa mm (C-165, placa): 15 forme ale lui mm_train din pas (147 lansari) x 5 variante\n\
         {p}: cursa: mm_train_tile | 1024x2304x768 f6 (x12 in pas) | 2838.3 us median | 1.27 TFLOP/s | biti =\n\
         {p}: cursa: mm_tv_128x64 | 1024x2304x768 f6 (x12 in pas) | 2720.3 us median | 1.33 TFLOP/s | biti =\n\
         {p}: cursa: mm_tv_64x128 | 1024x2304x768 f6 (x12 in pas) | 2709.2 us median | 1.33 TFLOP/s | biti =\n\
         {p}: cursa: mm_tv_128x128 | 1024x2304x768 f6 (x12 in pas) | 1889.8 us median | 1.91 TFLOP/s | biti =\n\
         {p}: cursa: mm_tv_64x64k8 | 1024x2304x768 f6 (x12 in pas) | - us median | - TFLOP/s | biti NETERMINAT\n\
         {p}: cursa mm: NU A TERMINAT (placa nu a terminat mm_tv_64x64k8 pe forma 1 (ultima inregistrare 0x0, asteptat 0x18f08)): 4 randuri cu aceiasi biti, 0 diferite\n"
    );
    let r = parse_race(&text);
    assert_eq!(r.rows.len(), 4);
    assert_eq!(r.untimed, 1);
    assert_eq!(r.unknown, 0);
    let c = r.choice().unwrap();
    assert_eq!(c.of(1024, 2304, 768, 6), 3);
    assert_eq!(c.entries().count(), 1);
    let saved = r.saved_ns(&c);
    println!(
        "kt t222 os-413: alegerea qkv = mm_tv_128x128, pasul ar scadea cu {} us (DEDUS)",
        saved / 1000
    );
    assert_eq!(saved, (2_838_300 - 1_889_800) * 12);
}
