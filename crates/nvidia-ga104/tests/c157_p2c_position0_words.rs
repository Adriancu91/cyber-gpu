//! C-157 point 2 (T149), SIMULARE / arithmetic only: the one slow position left in `os-329` (21 nov 15:56) is
//! position 0 — driver 9 863 µs against a median of 844 µs. In the stick journal (`gpu-jurnal-028.txt`) the
//! difference sits in the launch-slot window (`0x1b2d`): 7.43 ms at position 0, ~0.04 ms at every other one. The
//! launch shadow starts empty, the ladder used only a few slots of the plain list, so the first glued token writes
//! every word of its 88 slots (cbuf + QMD) and its whole pushbuffer. This test counts those words for a model with
//! #513's shape, checks that the per-word cost it implies agrees with the cold position 0 of `os-324` (plain list,
//! 16.9 ms), and shows what a warm-up (the same position 0 sent once before the clock) leaves for the timed
//! position 0: only the records' payloads. Nothing runs here; nothing new is proposed for the board.

use gpu_compute::layer::Shape;
use nvidia_ga104::kronos_diag::{self as kd, SeqMode};
use nvidia_ga104::kronos_layer::{self as kl, Launch, LayerVas, Prog, WeightVas};
use nvidia_ga104::kronos_model::{self as km, ModelVas};
use nvidia_ga104::mmu::VA_BASE;
use nvidia_ga104::push::PushBuf;

const E: usize = 768;
const LAYERS: usize = 12;
const VOCAB: usize = 16_384;
const CONTEXT: usize = 512;
const STRIDE: u32 = 0x100;
const RING_WORDS: usize = 3;

/// Board, `os-329`, position 0: the launch-slot window `0x1b2d` stayed open 129.906462 → 129.913895 s.
const OS329_SLOTS_US: f64 = 7_433.0;
/// Board, `os-329`, position 0: the pushbuffer window `0x10e6` stayed open 129.913895 → 129.914585 s (~0.07 ms later).
const OS329_PUSH_US: f64 = 690.0;
/// Board, `os-324`, position 0 (plain list, cold shadow): ~16.9 ms in the same window (note C-157 point 2b).
const OS324_SLOTS_US: f64 = 16_900.0;
/// Board, `os-329`: median VRAM words per token (`kt cuvinte VRAM pe simbol (median)` = 0x82).
const OS329_WORDS_MEDIAN: usize = 130;

fn shape() -> Shape {
    Shape {
        e: E,
        heads: 12,
        eps: 1e-5,
    }
}

fn vas() -> ModelVas {
    let mut next = VA_BASE + 0x1000_0000;
    let mut take = |bytes: u64| {
        let a = next;
        next += bytes.next_multiple_of(0x1000);
        a
    };
    let scratch = take(kl::scratch_bytes(E));
    let layers = (0..LAYERS)
        .map(|_| {
            let w = WeightVas {
                ln1_g: take(4 * E as u64),
                ln1_b: take(4 * E as u64),
                wqkv: take(3 * E as u64 * E as u64),
                bqkv: take(12 * E as u64),
                wout: take(E as u64 * E as u64),
                bout: take(4 * E as u64),
                ln2_g: take(4 * E as u64),
                ln2_b: take(4 * E as u64),
                wfc: take(4 * E as u64 * E as u64),
                bfc: take(16 * E as u64),
                wproj: take(4 * E as u64 * E as u64),
                bproj: take(4 * E as u64),
            };
            let kc = take(kl::cache_bytes(E));
            let vc = take(kl::cache_bytes(E));
            LayerVas::with_scratch(E, scratch, kc, vc, w).unwrap()
        })
        .collect();
    let fin = take(4 * (E + VOCAB) as u64);
    ModelVas {
        emb_tok: take(VOCAB as u64 * E as u64),
        emb_pos: take(CONTEXT as u64 * E as u64),
        ln_g: take(4 * E as u64),
        ln_b: take(4 * E as u64),
        layers,
        h_final: fin,
        logits: fin + 4 * E as u64,
    }
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(w)
        })
        .collect()
}

fn prog_va(p: Prog) -> u64 {
    VA_BASE + 0x1_0000 * (p as u64)
}

const SLOTS: u64 = VA_BASE + 0x0100_0000;
const RECORDS: u64 = VA_BASE + 0x0200_0000;

/// The words of one token as the shadows see them: (slot words — cbuf + QMD of every launch —, pushbuffer words).
fn token_words(list: &[Launch], first: u32) -> (Vec<Vec<u32>>, Vec<u32>) {
    let slots = list
        .iter()
        .enumerate()
        .map(|(j, l)| {
            let cb = SLOTS + 512 * j as u64;
            let mut w = words(&l.cbuf);
            w.extend(words(&l.qmd(prog_va(l.prog), cb).unwrap()));
            w
        })
        .collect();
    let qmds: Vec<u64> = (0..list.len())
        .map(|j| SLOTS + 512 * j as u64 + 256)
        .collect();
    let n = kd::words(SeqMode::EachLaunch, list.len());
    let mut push = vec![0u32; n];
    let mut p = PushBuf::new(&mut push);
    kd::encode(SeqMode::EachLaunch, &mut p, &qmds, RECORDS, first).unwrap();
    assert_eq!(p.len(), n);
    (slots, push)
}

/// Words a shadow that remembers `before` (or nothing, `None`) writes for `now`.
fn written(
    before: Option<&(Vec<Vec<u32>>, Vec<u32>)>,
    now: &(Vec<Vec<u32>>, Vec<u32>),
) -> (usize, usize) {
    match before {
        None => (now.0.iter().map(Vec::len).sum(), now.1.len()),
        Some(b) => {
            let s =
                b.0.iter()
                    .zip(&now.0)
                    .map(|(x, y)| x.iter().zip(y).filter(|(a, c)| a != c).count())
                    .sum();
            let p = b.1.iter().zip(&now.1).filter(|(a, c)| a != c).count();
            (s, p)
        }
    }
}

fn with_copy(mut l: Vec<Launch>, v: &ModelVas) -> Vec<Launch> {
    l.push(kl::copy_launch(v.logits, VA_BASE + 0x0300_0000, 4 * VOCAB as u64).unwrap());
    l
}

#[test]
fn t149_position0_cold_shadow_explains_os329_and_warm_up_removes_it() {
    let (s, v) = (shape(), vas());
    // "Buna ziua": position 0 = 2612, position 1 = 7464 (the same two tokens as T144; any two give the same counts).
    let fused = |tok, pos| {
        with_copy(
            km::token_launches_fused(&s, VOCAB, CONTEXT, &v, tok, pos).unwrap(),
            &v,
        )
    };
    let plain = |tok, pos| {
        with_copy(
            km::token_launches(&s, VOCAB, CONTEXT, &v, tok, pos).unwrap(),
            &v,
        )
    };
    let first = 0x1_0000u32;
    let p0 = token_words(&fused(2612, 0), first);
    assert_eq!(p0.0.len(), 88, "#513 glued + copy of the logits");

    // (1) Cold (os-329 today): nothing remembered → every word of the 88 slots and the whole pushbuffer.
    let (cold_slots, cold_push) = written(None, &p0);
    // (2) Warm-up: position 0 sent once before the clock (first), then the timed position 0 (first + STRIDE).
    let p0_again = token_words(&fused(2612, 0), first + STRIDE);
    let (warm_slots, warm_push) = written(Some(&p0), &p0_again);
    // (3) Every later position (sanity: the board's median, 130 words).
    let p1 = token_words(&fused(7464, 1), first + 2 * STRIDE);
    let (next_slots, next_push) = written(Some(&p0_again), &p1);

    let us_per_word = OS329_SLOTS_US / cold_slots as f64;
    // The same per-word cost must explain os-324's cold position 0 (plain list, 196 slots) on its own.
    let pl = token_words(&plain(2612, 0), first);
    let (plain_slots, _) = written(None, &pl);
    let plain_us = plain_slots as f64 * us_per_word;
    println!(
        "T149 (#513, lista lipita): pozitia 0 rece = {cold_slots} cuvinte in sloturi + {cold_push} pushbuffer; \
         dupa incalzire = {warm_slots} + {warm_push} (+{RING_WORDS} inel); pozitia 1 = {next_slots} + {next_push} (+{RING_WORDS}); \
         {us_per_word:.3} us/cuvant (os-329) → lista simpla rece {plain_slots} cuvinte ≈ {plain_us:.0} us (os-324 pe placa: ~{OS324_SLOTS_US:.0})"
    );

    // Warm-up leaves only the records' payloads: one per launch + the opening record, nothing in the slots.
    assert_eq!(
        warm_slots, 0,
        "same token, same position: cbufs and QMDs identical"
    );
    assert_eq!(warm_push, 88 + 1);
    // A later position writes what the board measured (median 130 words).
    assert_eq!(next_slots + next_push + RING_WORDS, OS329_WORDS_MEDIAN);
    // The cold count is two orders of magnitude above that: the whole of position 0's extra driver time.
    assert!(cold_slots > 50 * OS329_WORDS_MEDIAN);
    // The same cost explains the pushbuffer window of position 0 (within 25 %).
    let push_us = cold_push as f64 * us_per_word;
    assert!(
        (push_us / OS329_PUSH_US - 1.0).abs() < 0.25,
        "pushbuffer {push_us:.0} us vs board {OS329_PUSH_US:.0} us"
    );
    // One per-word cost explains both boards' cold position 0 (within 25 %): the explanation is not fitted twice.
    assert!(
        (plain_us / OS324_SLOTS_US - 1.0).abs() < 0.25,
        "os-324 cold slots {plain_us:.0} us vs board ~{OS324_SLOTS_US:.0} us"
    );
}
