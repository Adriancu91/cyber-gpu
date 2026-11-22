//! C-160 point 6, second part (T167): the library's list of the small golden step (`nvidia_ga104::train_step`) —
//! what the OS sends. The numbers of the list are checked end to end in T165 (SIMULARE, `sim_c160_p5_golden.rs`);
//! here: the programs' SHA-256, the shape of the list, the control zone, the uploads, the refusals and the verdict's
//! rows read back. No board.

use gpu_compute::train::{self, AdamW, Recipe};
use gpu_core::diag::Journal;
use nvidia_ga104::mmu::VA_BASE;
use nvidia_ga104::train_step::{
    self as ts, Outcome, PROGRAMS, Plan, Prog, Refs, SLOT, check_programs, judge, parse_rows,
};

const MIC_B: usize = 2;
const MIC_T: usize = 32;

fn batch() -> (Vec<u32>, Vec<u32>) {
    let toks: Vec<u32> = (0..=(MIC_B * MIC_T) as u32)
        .map(|i| (i * 37 + 11) % 256)
        .collect();
    (toks[..MIC_B * MIC_T].to_vec(), toks[1..].to_vec())
}

fn params() -> Vec<Vec<f32>> {
    train::MIC
        .tensors()
        .iter()
        .map(|s| vec![0.01; s.elements()])
        .collect()
}

#[test]
fn t167_programs_match_their_sha256() {
    check_programs().unwrap();
    assert_eq!(PROGRAMS.len(), 13);
    for (i, p) in PROGRAMS.iter().enumerate() {
        assert_eq!(p.index(), i);
        assert!(p.registers() <= 25, "{}", p.name());
        assert_eq!(p.sha256().len(), 64);
    }
}

#[test]
fn t167_the_list_of_one_step() {
    let (inp, _) = batch();
    let plan = Plan::new(train::MIC, MIC_B, MIC_T, &inp, VA_BASE + 0x100_0000).unwrap();
    let opt = AdamW::recipe_513(&params());
    for (recipe, n) in [(Recipe::AdamW513, 75), (Recipe::Sgd, 73)] {
        let l = plan.step(recipe, &opt, 1).unwrap();
        assert_eq!(l.len(), n);
        assert_eq!(plan.launches_per_step(recipe), n);
        let count = |p: Prog| l.iter().filter(|x| x.prog == p).count();
        assert_eq!(count(Prog::MmTrain), 27);
        assert_eq!(count(Prog::Colsum), 13);
        assert_eq!(count(Prog::Vadd), 8);
        assert_eq!(
            count(Prog::Gnorm),
            if recipe == Recipe::Sgd { 0 } else { 2 }
        );
        assert_eq!(l[0].tag, "inainte:incorporari");
        assert_eq!(l.last().unwrap().tag, "opt:pas");
        // Every tag of a layer has its layer.
        assert!(l.iter().all(|x| x.layer.is_some()
            == (x.tag.contains("rezidual")
                || x.tag.starts_with("inainte:ln1")
                || x.tag.starts_with("inainte:ln2")
                || x.tag.starts_with("inainte:qkv")
                || x.tag.starts_with("inainte:atentia")
                || x.tag.starts_with("inainte:out")
                || x.tag.starts_with("inainte:fc")
                || x.tag.starts_with("inainte:gelu")
                || x.tag.starts_with("inainte:proj")
                || (x.tag.starts_with("inapoi:")
                    && !x.tag.contains("cap")
                    && !x.tag.contains("final")
                    && x.tag != "inapoi:pozitii"
                    && x.tag != "inapoi:simboluri"))));
    }
    // Step 2 differs from step 1 only in the optimizer's scalars (AdamW's bias corrections).
    let a = plan.step(Recipe::AdamW513, &opt, 1).unwrap();
    let b = plan.step(Recipe::AdamW513, &opt, 2).unwrap();
    let diff: Vec<&str> = a
        .iter()
        .zip(&b)
        .filter(|(x, y)| x != y)
        .map(|(x, _)| x.tag)
        .collect();
    assert_eq!(diff, ["opt:pas"]);
    assert!(plan.step(Recipe::Sgd, &opt, 0).is_err());
    // ~2.8 MiB of board memory, everything inside the arena.
    assert!(plan.bytes() < 3 << 20, "{}", plan.bytes());
    // Every launch points into the arena (and nowhere past it; other words of the cbuf may look like addresses
    // below the arena by chance, e.g. two counts side by side).
    for x in a.iter() {
        let vas: Vec<u64> = x.cbuf[0x28..0x78]
            .chunks_exact(8)
            .map(|w| u64::from_le_bytes(w.try_into().unwrap()))
            .filter(|&va| va >= plan.base && va < VA_BASE + (1 << 32))
            .collect();
        assert!(!vas.is_empty(), "{}", x.tag);
        assert!(vas.iter().all(|&va| va < plan.end), "{}", x.tag);
    }
}

#[test]
fn t167_control_zone() {
    let (inp, _) = batch();
    let plan = Plan::new(train::MIC, MIC_B, MIC_T, &inp, VA_BASE + 0x100_0000).unwrap();
    let l = plan
        .step(Recipe::AdamW513, &AdamW::recipe_513(&params()), 1)
        .unwrap();
    let pv = |p: Prog| VA_BASE + 0x40_0000 + 0x1_0000 * p.index() as u64;
    let zone = VA_BASE + 0x20_0000;
    let (bytes, qmds) = ts::pack(&l, pv, zone).unwrap();
    assert_eq!(bytes.len() as u64, 75 * SLOT);
    assert_eq!(qmds.len(), 75);
    for (k, (x, q)) in l.iter().zip(&qmds).enumerate() {
        assert_eq!(*q, zone + k as u64 * SLOT + 0x100);
        let at = k * SLOT as usize;
        assert_eq!(&bytes[at..at + 0x100], &x.cbuf[..]);
        let want = x.qmd(pv(x.prog), zone + k as u64 * SLOT).unwrap();
        assert_eq!(&bytes[at + 0x100..at + 0x100 + want.len()], &want[..]);
    }
    assert!(ts::pack(&l, pv, zone + 8).is_err());
    assert!(ts::pack(&l, |_| VA_BASE + 4, zone).is_err());
    assert!(ts::pack(&[], pv, zone).is_err());
}

#[test]
fn t167_uploads_and_refusals() {
    let (inp, tgt) = batch();
    let base = VA_BASE + 0x100_0000;
    let plan = Plan::new(train::MIC, MIC_B, MIC_T, &inp, base).unwrap();
    let up = plan.uploads(&params(), &inp, &tgt).unwrap();
    // weights, g, m, v, tokens, targets, 2 × 3 segment lists, res
    assert_eq!(up.len(), 13);
    assert_eq!(up[0].0, plan.w);
    assert_eq!(up[0].1.len(), 4 * plan.n_params);
    assert!(up[1].1.iter().all(|&b| b == 0));
    assert_eq!(
        up.last().unwrap().1,
        [0.0f32.to_le_bytes(), 1.0f32.to_le_bytes()].concat()
    );
    for (va, b) in &up {
        assert!(*va >= plan.base && va + b.len() as u64 <= plan.end);
    }
    // Another batch than the plan's (the segment lists would be wrong) is refused, so are bad shapes.
    let mut other = inp.clone();
    other.swap(0, 1);
    assert!(plan.uploads(&params(), &other, &tgt).is_err());
    let mut bad_t = tgt.clone();
    bad_t[0] = 256;
    assert!(plan.uploads(&params(), &inp, &bad_t).is_err());
    assert!(plan.uploads(&params()[1..], &inp, &tgt).is_err());
    assert!(Plan::new(train::MIC, MIC_B, MIC_T, &inp, base + 4).is_err());
    assert!(Plan::new(train::MIC, MIC_B, MIC_T + 1, &inp, base).is_err());
    let mut big = inp.clone();
    big[3] = 300;
    assert!(Plan::new(train::MIC, MIC_B, MIC_T, &big, base).is_err());
    // unflat
    let flat = vec![1.0f32; plan.n_params];
    let t = plan.unflat(&flat).unwrap();
    assert_eq!(t.len(), train::MIC.tensor_count());
    assert!(plan.unflat(&flat[1..]).is_err());
}

fn fake_refs() -> (Outcome, Refs) {
    let p = params();
    let refs = Refs {
        losses: vec![5.0, 4.5],
        norms: vec![2.0, 1.9],
        grad1: p.clone(),
        after1: p.clone(),
        after_last: p.clone(),
    };
    let out = Outcome {
        losses: vec![5.0, 4.5],
        norms: vec![2.0, 1.9],
        grad1: p.clone(),
        after1: p.clone(),
        after_last: p,
        step_ns: vec![900_000, 700_000],
    };
    (out, refs)
}

#[test]
fn t167_verdict_rows_and_parse() {
    let (mut out, refs) = fake_refs();
    let v = judge(Recipe::AdamW513, 75, &out, &refs);
    assert!(v.passes());
    // f32 against f64 references: a few ‰ at most (1.9 is not exact in f32).
    assert!(v.worst_permille() <= 10, "{}", v.worst_permille());
    assert_eq!(v.median_step_ns, Some(900_000));
    let mut j = Journal::new(64);
    v.journal(&mut j);
    let text: String = j.events().map(|e| format!("gpu: {e}\n")).collect();
    let runs = parse_rows(&text);
    assert_eq!(runs.len(), 1);
    let get = |k: &str| runs[0].iter().find(|(n, _)| n.starts_with(k)).unwrap().1;
    assert_eq!(get("verdict"), 1);
    assert_eq!(get("lansari pe pas"), 75);
    assert_eq!(get("reteta"), 1);
    assert_eq!(f32::from_bits(get("eroarea pasului 1")), 5.0);
    assert_eq!(get("timp median"), 900);
    assert!(v.lines("placa")[0].contains("ACELEASI NUMERE"));

    // Just over a tolerance → not passing, the figure says how far.
    out.after_last[3][0] += 2.0 * train::TOL_ADAMW_W_ABS;
    let v = judge(Recipe::AdamW513, 75, &out, &refs);
    assert!(!v.passes());
    assert!(
        v.wlast_permille > 1000 && v.wlast_permille < 2100,
        "{}",
        v.wlast_permille
    );
    assert_eq!(v.wlast.at, 3);
    // A NaN anywhere → not passing, "NEFINIT".
    let (mut out, refs) = fake_refs();
    out.grad1[0][0] = f32::NAN;
    let v = judge(Recipe::AdamW513, 75, &out, &refs);
    assert!(!v.passes());
    assert_eq!(v.grad_abs_permille, u32::MAX);
    assert!(v.lines("placa")[1].contains("NEFINIT"));
    // A missing step / missing norms → not passing.
    let (mut out, refs) = fake_refs();
    out.norms.pop();
    assert!(!judge(Recipe::AdamW513, 75, &out, &refs).passes());
    assert!(judge(Recipe::Sgd, 73, &out, &refs).passes());
    let (mut out, refs) = fake_refs();
    out.losses.push(4.0);
    assert!(!judge(Recipe::Sgd, 73, &out, &refs).passes());
    assert!(!judge(Recipe::Sgd, 73, &Outcome::default(), &refs).passes());
}
