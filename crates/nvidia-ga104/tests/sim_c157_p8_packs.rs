//! T208 — C-157 point 8: the model card picks the compute pack (`docs/C-157_P8_FISA_MODELULUI.md`, predictions
//! written before the code). Pure CPU checks; nothing here is executed on a GPU.

use gpu_compute::card::{KeyValue, ModelCard, kgr1_header, kronos_keys};
use nvidia_ga104::kronos_diag::TokenList;
use nvidia_ga104::kronos_speed::ModelLimits;
use nvidia_ga104::packs::{self, KRONOS_GPT_Q8_GEN, Mismatch, Proof};
use nvidia_ga104::transfer::I11_DEFAULT_BYTES;

fn card(keys: &[(&str, KeyValue<'_>)]) -> ModelCard {
    ModelCard::from_kgr1(&kgr1_header(keys)).expect("fisa")
}

fn with(
    mut keys: Vec<(&'static str, KeyValue<'static>)>,
    k: &'static str,
    v: KeyValue<'static>,
) -> ModelCard {
    keys.push((k, v));
    card(&keys)
}

fn refused_with(c: &ModelCard) -> Vec<Mismatch> {
    let r = packs::select(c).expect_err("trebuia refuz");
    assert!(r.lines()[0].starts_with("REFUZ"));
    assert_eq!(
        r.lines().len(),
        1 + r.reasons.iter().map(|(_, m)| m.len()).sum::<usize>()
    );
    r.reasons.into_iter().flat_map(|(_, m)| m).collect()
}

#[test]
fn t208_513_and_651_are_proved_on_the_board() {
    for (layers, run) in [(12, "#513"), (24, "#651")] {
        let c = card(&kronos_keys(layers, 12, 768, 512, 16_384));
        let ch = packs::select(&c).expect("pachet");
        assert_eq!(ch.pack.name, "kronos-gpt-q8-generare");
        match ch.proof {
            Proof::Board(b) => assert!(b.run.starts_with(run)),
            Proof::Needed => panic!("{run} trebuia dovedit pe placa"),
        }
        assert!(ch.line().contains("dovedit pe placa"));
    }
}

#[test]
fn t208_tiny_vector_model_needs_the_proof() {
    let c = card(&kronos_keys(2, 4, 32, 512, 16_384));
    let ch = packs::select(&c).expect("pachet");
    assert_eq!(ch.proof, Proof::Needed);
    assert!(ch.line().contains("PROBA NECESARA"));
}

#[test]
fn t208_refusals() {
    let base = || kronos_keys(12, 12, 768, 512, 16_384);
    assert_eq!(
        refused_with(&with(base(), "fel", KeyValue::Text("f32"))),
        vec![Mismatch::Quant("f32".into())]
    );
    assert_eq!(
        refused_with(&with(base(), "n_capete", KeyValue::Int(4))),
        vec![Mismatch::HeadDim {
            have: 192,
            max: 128
        }]
    );
    assert_eq!(
        refused_with(&with(base(), "context", KeyValue::Int(1024))),
        vec![Mismatch::Context {
            have: 1024,
            max: 512
        }]
    );
    let m = refused_with(&with(base(), "n_emb", KeyValue::Int(100)));
    assert!(m.contains(&Mismatch::EMultiple {
        e: 100,
        multiple: 32
    }));
    assert_eq!(
        refused_with(&with(base(), "arhitectura", KeyValue::Text("llama"))),
        vec![Mismatch::Arch("llama".into())]
    );
    assert_eq!(
        refused_with(&with(base(), "cap_legat", KeyValue::Int(0))),
        vec![Mismatch::TiedHead]
    );
    assert_eq!(
        refused_with(&with(base(), "n_straturi", KeyValue::Int(65))),
        vec![Mismatch::Layers { have: 65, max: 64 }]
    );
    assert_eq!(
        refused_with(&with(base(), "vocabular", KeyValue::Int(0))),
        vec![Mismatch::Vocab]
    );
    assert_eq!(
        refused_with(&with(base(), "n_capete", KeyValue::Int(7))),
        vec![Mismatch::Heads { e: 768, heads: 7 }]
    );
    // Several reasons at once are all listed.
    let mut k = base();
    k.push(("fel", KeyValue::Text("f32")));
    k.push(("context", KeyValue::Int(2048)));
    assert_eq!(refused_with(&card(&k)).len(), 2);
}

#[test]
fn t208_limits_come_from_the_card() {
    let c = card(&kronos_keys(24, 12, 768, 512, 16_384));
    let ch = packs::select(&c).unwrap();
    assert_eq!(
        ch.limits(I11_DEFAULT_BYTES).unwrap(),
        ModelLimits::k651_lists(&[TokenList::FusedLn32]).unwrap()
    );
}

#[test]
fn t208_every_pack_proof_is_a_form_the_pack_takes() {
    for p in packs::PACKS {
        for b in p.proofs {
            let (l, h, e, ctx, v) = b.form;
            let c = card(&kronos_keys(
                l as i64, h as i64, e as i64, ctx as i64, v as i64,
            ));
            assert!(p.mismatches(&c).is_empty(), "{}: {}", p.name, b.run);
        }
    }
    assert_eq!(packs::PACKS[0], KRONOS_GPT_Q8_GEN);
}
