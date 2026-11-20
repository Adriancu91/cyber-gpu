//! C-147 point 7 (T141), SIMULARE: the driver's diet. On os-322 the driver took 3,1 ms of a 5,8 ms token, and the
//! stick journal's clocks (`driver_split`) put ~1,5 ms of it in rewriting the whole pushbuffer (2 360 words, each
//! read back) and ~0,5 ms in zeroing the 197 timestamp records — both every token. Here, on the simulated work
//! channel, a model run with (a) the pushbuffer written through a shadow (`kronos_fast::ShadowedMemory`: only the
//! words that changed since the last token) and (b) the records not zeroed must give the same logits and caches
//! **bit for bit** as the run that writes everything, every token; a stale record must still be caught. The "GPU"
//! is `gpu_sim::cmdproc`; nothing here proves anything about the board (where the gain is DEDUS, not measured).

mod common;

use common::token_sim::TokenSim;
use gpu_compute::layer::Shape;
use nvidia_ga104::kronos_diag::{SeqMode, TokenIo};
use nvidia_ga104::kronos_fast::WriteShadow;
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::transfer;

fn model() -> km::OwnedModel {
    km::test_model(
        Shape {
            e: 64,
            heads: 4,
            eps: 1e-5,
        },
        3,
        256,
        16,
        11,
    )
    .unwrap()
}

fn diet(s: &mut TokenSim<'_>) {
    s.push_shadow = Some(WriteShadow::new(s.r.push.pa, s.r.push.size).unwrap());
    s.zero_records = false;
}

/// Both lists (196-like and glued), six positions: the same bits with and without the diet.
#[test]
fn diet_same_logits_and_caches() {
    let m = model();
    for fused in [false, true] {
        let (mut a, mut b) = (TokenSim::new(&m, 0), TokenSim::new(&m, 0));
        diet(&mut b);
        for (pos, tok) in [5usize, 200, 17, 5, 99, 3].into_iter().enumerate() {
            let list = |s: &TokenSim<'_>| {
                if fused {
                    km::token_launches_fused(&m.shape, m.vocab, m.context, &s.vas, tok, pos)
                        .unwrap()
                } else {
                    km::token_launches(&m.shape, m.vocab, m.context, &s.vas, tok, pos).unwrap()
                }
            };
            let (la, lb) = (list(&a), list(&b));
            a.submit(&la, 0, SeqMode::EachLaunch).unwrap();
            b.submit(&lb, 0, SeqMode::EachLaunch).unwrap();
            let ra = a.read(&[(a.vas.logits, m.vocab)]).unwrap();
            let rb = b.read(&[(b.vas.logits, m.vocab)]).unwrap();
            assert_eq!(ra, rb, "fused {fused}, pozitia {pos}");
            for l in 0..3 {
                let n = (pos + 1) * m.shape.e;
                let (al, bl) = (&a.vas.layers[l], &b.vas.layers[l]);
                assert_eq!(
                    a.read(&[(al.kc, n), (al.vc, n)]).unwrap(),
                    b.read(&[(bl.kc, n), (bl.vc, n)]).unwrap()
                );
            }
            // The board's per-launch durations (from the records) are the same too.
            assert_eq!(a.durations.last(), b.durations.last());
        }
        // What the driver writes: the full run every token; the diet the whole pushbuffer while its place still
        // moves (token 0 sits after the setup stretch, token 1 at offset 0 once everything before is retired), then
        // about one word per record (the payloads) plus the ring entry, and never the records.
        let n = a.submissions[0] as u64;
        let full = a.driver_words[1].0;
        assert!(a.driver_words.iter().all(|w| w.2 == 16 * (n + 1)));
        assert!(b.driver_words.iter().all(|w| w.2 == 0));
        assert_eq!(
            b.driver_words[0].0, full,
            "the first token writes everything"
        );
        for (i, w) in b.driver_words.iter().enumerate().skip(2) {
            assert_eq!(
                w.0 + w.1,
                full,
                "token {i}: every word either written or left out"
            );
            assert!(
                w.0 <= n + 1 + 3 + 2,
                "token {i}: {} words written for {n} launches",
                w.0
            );
        }
        eprintln!(
            "fused {fused}: {n} lansari; cuvinte scrise pe simbol: {} fara dieta, {} cu dieta (+ {} octeti de inregistrari zerorizati fara dieta)",
            full,
            b.driver_words[2].0,
            16 * (n + 1)
        );
    }
}

/// Records not zeroed: what the previous submission left in them can never pass for this one (the payloads move
/// by the stride every submission).
#[test]
fn stale_records_are_caught() {
    const FIRST: u32 = 0x1_0100;
    const STRIDE: u32 = 0x100;
    let n = 5;
    let rec = |first: u32, upto: usize| {
        let mut b = vec![0u8; 16 * (n + 1)];
        for k in 0..=upto {
            b[16 * k..16 * k + 4].copy_from_slice(&(first + k as u32).to_le_bytes());
            b[16 * k + 8..16 * k + 16].copy_from_slice(&(1000 * k as u64 + 1).to_le_bytes());
        }
        b
    };
    // The whole submission done: fine.
    assert!(transfer::analyze_sequence(&rec(FIRST, n), n, FIRST).is_ok());
    // The previous submission's records, untouched: not this one's.
    let old = rec(FIRST - STRIDE, n);
    assert!(transfer::analyze_sequence(&old, n, FIRST).is_err());
    assert_eq!(transfer::records_seen(&old, n, FIRST), 0);
    // This submission stopped after launch 2 (records 0..=2 new, the rest old): caught, and counted right.
    let mut mixed = old.clone();
    mixed[..48].copy_from_slice(&rec(FIRST, 2)[..48]);
    assert!(transfer::analyze_sequence(&mixed, n, FIRST).is_err());
    assert_eq!(transfer::records_seen(&mixed, n, FIRST), 3);
    // The last record is the one the OS waits for: an old one never equals the new last payload.
    assert_ne!(FIRST - STRIDE + n as u32, FIRST + n as u32);
}
