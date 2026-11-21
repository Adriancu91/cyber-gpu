//! C-157 point 2 after os-335 (T156): which adder model explains what the real RTX 3070 Ti wrote in the P8 probe.
//!
//! The board's figures (`OS335_EXACT`, `OS335_PERMILLE`) are MEASURED (os-335, 21 Nov 2026 22:21). Everything
//! else here runs on the host: the models are computed on the CPU and compared with those figures. A match is a
//! match on the **fingerprint** (12 numbers), not bit for bit; `fit_rows` is what lets the next board run say so.

use gpu_compute::mma::{self, ADD_MODELS, GA104_OS335, HwAdder, MmaVerdict};
use gpu_core::diag::Journal;
use nvidia_ga104::mma16_probe::{
    self as mp, D_BYTES, OS335_EXACT, OS335_PERMILLE, SETS, fit_rows, ga104_fit_bytes,
    parse_mma16_fit,
};

/// (exact per set, worst ‰ per set) of a D-maker over the 6 sets.
fn fingerprint(f: impl Fn(&mp::Inputs) -> [f32; 256]) -> (Vec<u32>, Vec<u32>) {
    let mut ex = Vec::new();
    let mut pm = Vec::new();
    for s in SETS.iter() {
        let x = s.inputs();
        let d = f(&x);
        match mma::judge(&x.a, &x.b, &x.c, &d) {
            MmaVerdict::InToleranta { exact } => ex.push(exact as u32),
            v => panic!("{}: {v:?}", s.name),
        }
        pm.push(mma::worst_permille(&x.a, &x.b, &x.c, &d).unwrap());
    }
    (ex, pm)
}

#[test]
fn t156_the_ga104_model_reproduces_the_os335_fingerprint() {
    let (ex, pm) = fingerprint(|x| mma::mma16_hw(&x.a, &x.b, &x.c, GA104_OS335).unwrap());
    assert_eq!(ex, OS335_EXACT);
    assert_eq!(pm, OS335_PERMILLE);
}

#[test]
fn t156_none_of_the_ten_older_models_does() {
    // This is what os-335 said: "modele potrivite: niciunul".
    for m in ADD_MODELS {
        let (ex, pm) = fingerprint(|x| mma::mma16_blocked(&x.a, &x.b, &x.c, m).unwrap());
        assert!(ex != OS335_EXACT || pm != OS335_PERMILLE, "{}", m.name());
    }
}

#[test]
fn t156_neighbours_of_the_fit_do_not_match() {
    for block in [1, 2, 4, 8, 16] {
        for extra_bits in 0..=3 {
            let h = HwAdder { block, extra_bits };
            if h == GA104_OS335 {
                continue;
            }
            let (ex, pm) = fingerprint(|x| mma::mma16_hw(&x.a, &x.b, &x.c, h).unwrap());
            assert!(ex != OS335_EXACT || pm != OS335_PERMILLE, "{h:?}");
        }
    }
}

#[test]
fn t156_fit_rows_say_yes_only_for_the_fit() {
    let mut j = Journal::new(64);
    for (i, set) in SETS.iter().enumerate() {
        let fit = ga104_fit_bytes(i).unwrap();
        let r = fit_rows(i, &fit).unwrap();
        assert_eq!(r[0], ("kt mma16 set: iesire (fnv)", mp::d_fnv(&fit)));
        assert_eq!(r[1], ("kt mma16 set: model ga104", 1));
        // The simulator's older hypothesis (block 8, f32 truncation of exact sums) is not the fit on the
        // non-identity sets.
        let x = set.inputs();
        let old = mma::mma16_block8_trunc(&x.a, &x.b, &x.c).unwrap();
        let old_bytes: Vec<u8> = old.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect();
        let want = if i == 0 { 1 } else { 0 };
        assert_eq!(fit_rows(i, &old_bytes).unwrap()[1].1, want, "set {i}");
        mp::journal_fit(&mut j, i, &fit).unwrap();
    }
    assert_eq!(fit_rows(1, &[0u8; 8]).unwrap()[1].1, 2);
    assert!(fit_rows(9, &[0u8; D_BYTES]).is_err());

    let text: String = j.events().map(|e| format!("gpu: {e}\n")).collect();
    let back = parse_mma16_fit(&text);
    assert_eq!(back.len(), SETS.len());
    assert!(back.iter().all(|&(_, m)| m == 1));
    assert_eq!(back[2].0, mp::d_fnv(&ga104_fit_bytes(2).unwrap()));
}
