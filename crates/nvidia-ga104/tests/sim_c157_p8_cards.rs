//! T211 — C-157 point 8 step 2: the reference cards of #513 / #651 are the one source of their numbers
//! (`docs/C-157_P8_FISA_MODELULUI.md` §5, predictions written before the code). Pure CPU checks; nothing here is
//! executed on a GPU.

use gpu_compute::card::{ModelCard, kgr1_header, kronos_keys};
use nvidia_ga104::kronos_diag::TokenList;
use nvidia_ga104::kronos_layer::Prog;
use nvidia_ga104::kronos_speed::{ModelLimits, card_k513, card_k651};
use nvidia_ga104::packs::{self, KRONOS_GPT_Q8_GEN, Mismatch, Proof};
use nvidia_ga104::transfer::I11_DEFAULT_BYTES;

fn from_file(layers: i64) -> ModelCard {
    ModelCard::from_kgr1(&kgr1_header(&kronos_keys(layers, 12, 768, 512, 16_384))).expect("fisa")
}

#[test]
fn t211_reference_cards_equal_the_files() {
    assert_eq!(card_k513(), from_file(12));
    assert_eq!(card_k651(), from_file(24));
}

#[test]
fn t211_limits_from_the_cards_equal_the_old_ones() {
    assert_eq!(
        ModelLimits::of_card(&card_k513(), I11_DEFAULT_BYTES).unwrap(),
        ModelLimits::k513()
    );
    assert_eq!(
        ModelLimits::of_card(&card_k651(), I11_DEFAULT_BYTES).unwrap(),
        ModelLimits::k651()
    );
    let fused = [TokenList::FusedLn32];
    let l = ModelLimits::of_card_lists(&card_k651(), I11_DEFAULT_BYTES, &fused, &[]).unwrap();
    assert_eq!(l, ModelLimits::k651_lists(&fused).unwrap());
    assert_eq!(l.launches, 172);
}

#[test]
fn t211_reference_cards_are_proved_and_the_proofs_use_their_forms() {
    for (c, run) in [(card_k513(), "#513, os-346"), (card_k651(), "#651, os-355")] {
        let ch = packs::select(&c).expect("pachet");
        match ch.proof {
            Proof::Board(b) => assert!(b.run.starts_with(run), "{}", b.run),
            Proof::Needed => panic!("{run} trebuia dovedit pe placa"),
        }
    }
    let forms: Vec<_> = KRONOS_GPT_Q8_GEN.proofs.iter().map(|p| p.form).collect();
    assert_eq!(forms, vec![card_k513().form(), card_k651().form()]);
}

#[test]
fn t211_argmax_after_the_list_is_one_more_launch() {
    let ch = packs::select(&card_k651()).expect("pachet");
    let plain = ch.limits(I11_DEFAULT_BYTES).unwrap();
    let with_argmax = ch.limits_with(I11_DEFAULT_BYTES, &[Prog::Argmax]).unwrap();
    assert_eq!(with_argmax.launches, plain.launches + 1);
    assert_eq!(plain, ch.limits_with(I11_DEFAULT_BYTES, &[]).unwrap());
}

#[test]
fn t211_kronos_card_is_only_data() {
    let c = ModelCard::kronos(12, 12, 100, 512, 16_384);
    assert_eq!(c.e, 100);
    let r = packs::select(&c).expect_err("trebuia refuz");
    assert!(
        r.reasons
            .iter()
            .any(|(_, ms)| ms.contains(&Mismatch::EMultiple {
                e: 100,
                multiple: 32
            }))
    );
}
