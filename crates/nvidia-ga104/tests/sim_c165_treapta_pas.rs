//! T232 (C-165, `docs/C-165_TREAPTA_PAS.md`): one shared-memory step of the SM for every launch of the training step —
//! (a) `Prog::kernel`, (b) the best step of the #513 step as the OS sends it, (c) `pack_with_sm_step` = `pack` outside
//! the three step fields, (e) the refusals. (d), the simulated board with the same bits, is in `sim_c160_p5_golden.rs`.
//! SIMULARE / DEDUS: the occupancy arithmetic is the public table for capability 8.6; nothing here runs on the board.

use gpu_compute::train::{self, AdamW, Recipe};
use nvidia_ga104::mm_train_variants::{self as mv, MmChoice, RACE_WINNER_513};
use nvidia_ga104::occupancy::{self as oc, Reserve, SM86};
use nvidia_ga104::qmd::{SM_CONFIG_STEPS, sm_config_shared_units};
use nvidia_ga104::train_step::{self as ts, Launch, Plan, Prog};
use nvidia_rm570::QmdField;
use nvidia_rm570::regs::qmd_v03_00 as q;

const KIB: u32 = 1024;
const HEAP: u64 = nvidia_ga104::mmu::VA_BASE + 0x100_0000;
const ZONE: u64 = nvidia_ga104::mmu::VA_BASE + 0x20_0000;

fn prog_va(p: Prog) -> u64 {
    nvidia_ga104::mmu::VA_BASE + 0x40_0000 + 0x1_0000 * p.index() as u64
}

/// The #513 step as os-418 sends it (`new_tiles3` + `--mm-placi` + `--colsum-impartit` + the race's pick), 2 × 512.
fn list_513() -> Vec<Launch> {
    let (bsz, tl) = (2usize, 512usize);
    let inputs: Vec<u32> = (0..bsz * tl)
        .map(|i| ((i * 7_919) % 16_384) as u32)
        .collect();
    let base = Plan::new_tiles3(train::M513, bsz, tl, &inputs, HEAP)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split()
        .resident()
        .unwrap();
    let params: Vec<Vec<f32>> = train::M513
        .tensors()
        .iter()
        .map(|t| vec![0.0; t.elements()])
        .collect();
    let opt = AdamW::recipe_513(&params);
    let forms = mv::forms(&base.step(Recipe::AdamW513, &opt, 1).unwrap());
    let plan = base.with_mm_choice(MmChoice::all(&forms, RACE_WINNER_513).unwrap());
    plan.step(Recipe::AdamW513, &opt, 1).unwrap()
}

/// T232a: `Prog::kernel` = the program's constants.
#[test]
fn t232a_prog_kernel() {
    for p in ts::PROGRAMS_ALL_CHOICE {
        let k = p.kernel();
        let b = p.block();
        assert_eq!(k.name, p.name());
        assert_eq!(k.threads, b[0] * b[1] * b[2]);
        assert_eq!(k.registers, p.registers());
        assert_eq!(k.shared_bytes, p.shared().map_or(0, |s| s.bytes));
    }
    let k = Prog::MmTrainTile.kernel();
    assert_eq!((k.threads, k.registers, k.shared_bytes), (256, 34, 8 * KIB));
}

/// T232b (P1): the best step of the #513 step, with every program's limit.
#[test]
fn t232b_best_step_of_the_513_step() {
    let list = list_513();
    assert_eq!(list.len(), 483, "os-418: 483 launches");
    let ks = ts::step_kernels(&list);
    for k in &ks {
        let free = oc::blocks_without_shared(&SM86, k).unwrap();
        let at = |s: u32, r| oc::blocks_per_sm(&SM86, k, s, r).unwrap().blocks;
        println!(
            "kt t232b {:<20} {:>4} fire {:>3} reg {:>6} B | fara memorie {} ({:?}) | R0 8/16/32/64: {}/{}/{}/{} | R1: {}/{}/{}/{}",
            k.name,
            k.threads,
            k.registers,
            k.shared_bytes,
            free.blocks,
            free.bound,
            at(8 * KIB, Reserve::None),
            at(16 * KIB, Reserve::None),
            at(32 * KIB, Reserve::None),
            at(64 * KIB, Reserve::None),
            at(8 * KIB, Reserve::OneKib),
            at(16 * KIB, Reserve::OneKib),
            at(32 * KIB, Reserve::OneKib),
            at(64 * KIB, Reserve::OneKib),
        );
    }
    let r0 = ts::step_best_sm_step(&list, Reserve::None);
    let r1 = ts::step_best_sm_step(&list, Reserve::OneKib);
    println!(
        "kt t232b pasul #513 (483 lansari, {} programe): treapta R0 {} KiB, R1 {} KiB (DEDUS)",
        ks.len(),
        r0 >> 10,
        r1 >> 10
    );
    assert!(SM_CONFIG_STEPS.contains(&r0) && SM_CONFIG_STEPS.contains(&r1));
    assert!(r0 <= r1);
    // Every program reaches its limit without shared memory at the chosen step.
    for (step, r) in [(r0, Reserve::None), (r1, Reserve::OneKib)] {
        for k in &ks {
            let free = oc::blocks_without_shared(&SM86, k).unwrap().blocks;
            assert!(
                oc::blocks_per_sm(&SM86, k, step, r).unwrap().blocks >= free || step == 64 * KIB,
                "{}",
                k.name
            );
        }
    }
    // The list really packs at both.
    for s in [r0, r1] {
        ts::pack_with_sm_step(&list, prog_va, ZONE, s).unwrap();
    }
}

fn field_words(f: QmdField) -> core::ops::RangeInclusive<usize> {
    (f.lo as usize / 32)..=(f.hi as usize / 32)
}

fn get(qmd: &[u8], f: QmdField) -> u32 {
    let (hi, lo) = (f.hi, f.lo);
    let w = lo as usize / 32;
    let x = u32::from_le_bytes(qmd[4 * w..4 * w + 4].try_into().unwrap());
    let width = hi - lo + 1;
    (x >> (lo % 32))
        & if width == 32 {
            u32::MAX
        } else {
            (1 << width) - 1
        }
}

/// T232c (P2): byte for byte = `pack`, outside the words of MIN / TARGET / MAX.
#[test]
fn t232c_same_bytes_outside_the_step_fields() {
    let list = list_513();
    let (a, qa) = ts::pack(&list, prog_va, ZONE).unwrap();
    let fields = [
        q::NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE,
        q::NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE,
        q::NVC6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE,
    ];
    let step_words: Vec<usize> = fields.iter().flat_map(|&f| field_words(f)).collect();
    for step in SM_CONFIG_STEPS {
        let (b, qb) = match ts::pack_with_sm_step(&list, prog_va, ZONE, step) {
            Ok(x) => x,
            Err(_) => {
                // Only refused below the largest program's shared memory.
                assert!(list.iter().any(|l| {
                    l.prog
                        .shared()
                        .is_some_and(|s| s.bytes.next_multiple_of(256) > step)
                }));
                continue;
            }
        };
        assert_eq!(qa, qb);
        assert_eq!(a.len(), b.len());
        let unit = sm_config_shared_units(step);
        let mut changed_none = 0;
        for &qva in &qa {
            let at = (qva - ZONE) as usize;
            // The cbuf before the QMD.
            assert_eq!(a[at - 0x100..at], b[at - 0x100..at]);
            let (x, y) = (&a[at..at + 0x100], &b[at..at + 0x100]);
            for w in 0..0x40 {
                if !step_words.contains(&w) {
                    assert_eq!(x[4 * w..4 * w + 4], y[4 * w..4 * w + 4], "cuvantul {w}");
                }
            }
            assert_eq!(get(y, fields[0]), unit);
            assert_eq!(get(y, fields[1]), unit);
            assert_eq!(get(y, fields[2]), sm_config_shared_units(64 * KIB));
            if get(x, fields[1]) == 0 {
                changed_none += 1;
            }
        }
        // Everything outside the QMDs and cbufs (padding) is identical too.
        let mut ma = a.clone();
        let mut mb = b.clone();
        for &qva in &qa {
            let at = (qva - ZONE) as usize;
            ma[at - 0x100..at + 0x100].fill(0);
            mb[at - 0x100..at + 0x100].fill(0);
        }
        assert_eq!(ma, mb);
        println!(
            "kt t232c treapta {} KiB: {} QMD-uri, doar MIN/TARGET/MAX schimbate; {} lansari fara memorie comuna (azi 0/0/0)",
            step >> 10,
            qa.len(),
            changed_none
        );
    }
}

/// T232e (P4): refused before anything is sent.
#[test]
fn t232e_refusals() {
    let list = list_513();
    for bad in [0, 4 * KIB, 12 * KIB, 96 * KIB, 128 * KIB] {
        assert!(
            ts::pack_with_sm_step(&list, prog_va, ZONE, bad).is_err(),
            "{bad}"
        );
    }
    let tile = list.iter().find(|l| l.prog == Prog::MmTv128x128).unwrap();
    let need = tile.prog.shared().unwrap().bytes;
    for s in SM_CONFIG_STEPS {
        assert_eq!(
            tile.qmd_with_sm_step(0x10_0000, 0x20_0000, s).is_ok(),
            s >= need,
            "{s}"
        );
    }
    assert!(ts::pack_with_sm_step(&[], prog_va, ZONE, 64 * KIB).is_err());
}
