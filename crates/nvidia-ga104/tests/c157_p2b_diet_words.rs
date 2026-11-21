//! C-157 point 2b (T144), SIMULARE / arithmetic only: why the board wrote a median of **238** VRAM words per token
//! with the diet on (`os-324`, 21 nov 11:01; `kt cuvinte VRAM pe simbol (median)` = 0xee), when the small models of
//! the simulation write ~50. Nothing runs here: the words the diet must still write are counted from the bytes the
//! driver builds for two consecutive positions of a model with #513's shape (12 layers, E 768, 12 heads, vocabulary
//! 16 384) — the launch slots (cbuf + QMD, through the launch shadow) and the pushbuffer (through its shadow). The
//! ring entry and `GP_PUT` (3 more words the OS writes outside both shadows) are added by hand. Addresses are
//! made-up but fixed, as on the board (the same slots, the same records, the same programs every token).

use gpu_compute::layer::Shape;
use nvidia_ga104::kronos_diag::{self as kd, SeqMode};
use nvidia_ga104::kronos_layer::{self as kl, LayerVas, Prog, WeightVas};
use nvidia_ga104::kronos_model::{self as km, ModelVas};
use nvidia_ga104::mmu::VA_BASE;
use nvidia_ga104::push::PushBuf;

const E: usize = 768;
const LAYERS: usize = 12;
const VOCAB: usize = 16_384;
const CONTEXT: usize = 512;
/// The OS's payload step between two submissions (`gpu_i14.rs`, `STRIDE`).
const STRIDE: u32 = 0x100;
/// Ring entry (2 words) + `GP_PUT` in USERD (1 word): written by the OS outside the two shadows.
const RING_WORDS: usize = 3;

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
    let scratch = take(nvidia_ga104::kronos_layer::scratch_bytes(E));
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
            let kc = take(nvidia_ga104::kronos_layer::cache_bytes(E));
            let vc = take(nvidia_ga104::kronos_layer::cache_bytes(E));
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
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn differ(a: &[u32], b: &[u32]) -> usize {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).filter(|(x, y)| x != y).count()
}

/// Program VAs: one fixed address per program, as the OS loads them once.
fn prog_va(p: Prog) -> u64 {
    VA_BASE + 0x1_0000 * (p as u64)
}

#[test]
fn t144_diet_words_per_token_match_the_board() {
    let (s, v) = (shape(), vas());
    let slots = VA_BASE + 0x0100_0000;
    let records = VA_BASE + 0x0200_0000;
    // Two consecutive positions of the generation (prompt tokens of "Buna ziua", then the first generated).
    // + the copy of the logits down into RAM (`kl::copy_launch`), as the OS sends it (`kp copiere`: 1 launch).
    let down = kl::copy_launch(v.logits, VA_BASE + 0x0300_0000, 4 * VOCAB as u64).unwrap();
    let mut a = km::token_launches(&s, VOCAB, CONTEXT, &v, 2612, 5).unwrap();
    let mut b = km::token_launches(&s, VOCAB, CONTEXT, &v, 7464, 6).unwrap();
    a.push(down.clone());
    b.push(down);
    assert_eq!(
        a.len(),
        196,
        "#513: 16 per layer + embed + ln_final + logits + copy"
    );
    // (1) Launch slots, word by word (cbuf at +0, QMD at +256 of each 512-byte slot), as the launch shadow sees them.
    let mut cbuf_words = 0;
    let mut qmd_words = 0;
    for (j, (la, lb)) in a.iter().zip(&b).enumerate() {
        let cb = slots + 512 * j as u64;
        cbuf_words += differ(&words(&la.cbuf), &words(&lb.cbuf));
        let (qa, qb) = (
            la.qmd(prog_va(la.prog), cb).unwrap(),
            lb.qmd(prog_va(lb.prog), cb).unwrap(),
        );
        qmd_words += differ(&words(&qa), &words(&qb));
    }
    // (2) The pushbuffer (`EachLaunch`, one submission per token), at the same place, payloads one STRIDE apart.
    let qmds: Vec<u64> = (0..a.len()).map(|j| slots + 512 * j as u64 + 256).collect();
    let n = kd::words(SeqMode::EachLaunch, a.len());
    let (mut wa, mut wb) = (vec![0u32; n], vec![0u32; n]);
    let first = 0x1_0000u32;
    {
        let mut p = PushBuf::new(&mut wa);
        kd::encode(SeqMode::EachLaunch, &mut p, &qmds, records, first).unwrap();
        assert_eq!(p.len(), n);
    }
    {
        let mut p = PushBuf::new(&mut wb);
        kd::encode(SeqMode::EachLaunch, &mut p, &qmds, records, first + STRIDE).unwrap();
    }
    let push_words = differ(&wa, &wb);
    let total = cbuf_words + qmd_words + push_words + RING_WORDS;
    println!(
        "T144 (#513, dieta): cbuf {cbuf_words} + QMD {qmd_words} + pushbuffer {push_words} (din {n}) + inel {RING_WORDS} = {total} cuvinte pe simbol (placa os-324: 238)"
    );
    // The payload of every record changes every token: one word per launch + the opening record.
    assert_eq!(push_words, a.len() + 1);
    // The QMDs never change (same program, same slot).
    assert_eq!(qmd_words, 0);
    // What the board measured, to the word.
    assert_eq!(total, 238, "board: median 238 words per token (os-324)");
}
