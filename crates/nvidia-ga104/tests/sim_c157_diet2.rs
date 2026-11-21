//! C-157 point 3 (T147), SIMULARE: the second diet of the driver — nothing new sent to the board, only less CPU work
//! per token. A text runs as the OS runs it (the glued list `Fused`, the first diet on, every token followed by the
//! copy of the logits down into RAM), and at every position:
//! - the two-record read (`kronos_diet2::ends_span`, record 0 and record n) gives the board's span of the token
//!   **exactly** = the sum of the durations of the full read (`transfer::analyze_sequence`) = what the OS's
//!   `TokenOutput` carries today (`end − start`, `busy`);
//! - `gpu_compute::model::argmax_bits` on the RAM buffer's words gives exactly `argmax` on the floats, so the
//!   generated text is the same token for token;
//! - `RecordPlan::DIET` reads 8 words instead of 4·(n+1) on all positions but 0, 1 and every 8th;
//! - a wrong record (a launch that did not finish, a clock going back) is still caught with two records.
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about the board.

mod common;

use common::token_sim::TokenSim;
use common::{get_bytes, sys_write};
use gpu_compute::layer::Shape;
use gpu_compute::model;
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenList, Way};
use nvidia_ga104::kronos_diet2::{self as d2, RecordPlan};
use nvidia_ga104::kronos_fast::WriteShadow;
use nvidia_ga104::kronos_layer as kl;
use nvidia_ga104::kronos_model as km;

const PROMPT: [usize; 3] = [5, 200, 17];

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

/// The records of the last submission (`n` launches): (all of them, first payload).
fn last_records(s: &mut TokenSim<'_>, n: usize) -> Vec<u8> {
    let pa = s.records.pa;
    get_bytes(s.r.mem.writer_mut(), pa, 16 * (n as u64 + 1))
}

#[test]
fn t147_two_records_and_integer_argmax_change_nothing() {
    let m = model();
    let mut s = TokenSim::new(&m, 0);
    s.push_shadow = Some(WriteShadow::new(s.r.push.pa, s.r.push.size).unwrap());
    s.zero_records = false;
    let way = Way::of(SeqMode::EachLaunch);
    let n = TokenList::Fused.launches_per_token(m.layers.len()) + 1;
    let plan = RecordPlan::DIET;
    let (mut words_today, mut words_diet) = (0usize, 0usize);
    let mut tok = PROMPT[0];
    let mut first_payload = 0x1_0000u32;
    for pos in 0..m.context {
        let (vas, bytes) = (s.vas.clone(), (4 * m.vocab as u64).next_multiple_of(16));
        let down = kl::copy_launch(vas.logits, s.host.va, bytes).unwrap();
        let host = s.host.clone();
        sys_write(&mut s.r.sys, &host, &vec![0xee; bytes as usize]);
        kd::send_token_list(
            &mut s,
            TokenList::Fused,
            &m.weights(),
            &vas,
            tok,
            pos,
            way,
            &[down],
        )
        .unwrap();
        let rec = last_records(&mut s, n);
        let full: u64 = s.durations.last().unwrap().iter().sum();
        let (start, end, busy) = *s.times.last().unwrap();
        let span = d2::ends_span(&rec[..16], &rec[16 * n..], n, first_payload).unwrap();
        assert_eq!(span, full, "pozitia {pos}: capetele = suma duratelor");
        assert_eq!(span, end - start, "pozitia {pos}: capetele = end - start");
        assert_eq!(span, busy, "pozitia {pos}: capetele = busy");
        // A record that did not arrive, or a clock going back, is caught with two records too.
        let mut bad_last = rec[16 * n..].to_vec();
        bad_last[0] ^= 1;
        assert!(d2::ends_span(&rec[..16], &bad_last, n, first_payload).is_err());
        let mut back = rec[16 * n..].to_vec();
        back[8..16].copy_from_slice(&0u64.to_le_bytes());
        if span > 0 {
            assert!(d2::ends_span(&rec[..16], &back, n, first_payload).is_err());
        }
        // The argmax on the words = the argmax on the floats.
        let logits = s.ram_logits();
        let bits: Vec<u32> = logits.iter().map(|x| x.to_bits()).collect();
        let a = model::argmax(&logits);
        assert_eq!(model::argmax_bits(&bits), a, "pozitia {pos}");
        words_today += RecordPlan::ALL.words(n, pos);
        words_diet += plan.words(n, pos);
        tok = if pos + 1 < PROMPT.len() {
            PROMPT[pos + 1]
        } else {
            a.unwrap()
        };
        first_payload += 0x100;
    }
    eprintln!(
        "T147 (SIMULARE): {} pozitii, {n} lansari; cuvinte de inregistrari citite: azi {words_today}, dieta 2 {words_diet}",
        m.context
    );
    // 16 positions: 0, 1 and 8 read all (3 × 4·(n+1)), the other 13 read 8 words.
    assert_eq!(words_diet, 3 * 4 * (n + 1) + 13 * 8);
    assert!(words_diet * 3 < words_today);
}

/// #513's numbers (87 launches + the copy = 88, so 89 records): 356 words today (os-328), 8 with the diet on 7 of
/// 8 positions.
#[test]
fn t147_k513_record_words() {
    let n = 88;
    assert_eq!(RecordPlan::ALL.words(n, 5), 356);
    assert_eq!(RecordPlan::DIET.words(n, 5), 8);
    let avg: usize = (0..64).map(|p| RecordPlan::DIET.words(n, p)).sum::<usize>() / 64;
    assert!(avg < 60, "{avg}");
}
