//! C-160 point 3 — T151: the CPU oracle (`gpu_compute::train`) against PyTorch's golden step for the small model
//! (hub `integrare/kronos/pas_de_aur_mic/`, copied into `tests/date/pas_de_aur_mic/`; `reteta.json` SHA-256
//! `5e3c5824…e3a8`, every data file checked against its SHA-256 when it was copied).
//!
//! What is compared, with the tolerances written in the hub's `PAS_DE_AUR.md` (fixed before this oracle existed):
//! the loss of the 10 steps, the gradient of step 1 of every tensor, the weights after step 1 and after step 10 —
//! first recipe `sgd`, then `adamw_513`. CPU against CPU: nothing here runs on a GPU.

use std::path::{Path, PathBuf};

use gpu_compute::train::{self, Batch, MIC, Recipe};

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/date/pas_de_aur_mic")
}

fn read_f32(p: &Path, n: usize) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(b.len(), 4 * n, "{}", p.display());
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_i32(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| {
            let v = i32::from_le_bytes(*c);
            assert!(v >= 0);
            v as u32
        })
        .collect()
}

fn tensors(sub: &str) -> Vec<Vec<f32>> {
    MIC.tensors()
        .iter()
        .map(|t| {
            read_f32(
                &dir().join(sub).join(format!("{}.f32", t.name)),
                t.elements(),
            )
        })
        .collect()
}

/// The 10 losses of a recipe, read from `reteta.json` without a JSON parser: the list after
/// `"<recipe>": {` … `"pierderi": [`.
fn losses(recipe: &str) -> Vec<f64> {
    let s = std::fs::read_to_string(dir().join("reteta.json")).unwrap();
    let r = s.find(&format!("\"{recipe}\": {{")).expect("reteta");
    let p = r + s[r..].find("\"pierderi\": [").expect("pierderi") + "\"pierderi\": [".len();
    let end = p + s[p..].find(']').unwrap();
    s[p..end]
        .split(',')
        .map(|x| x.trim().parse().unwrap())
        .collect()
}

struct Dev {
    abs: f32,
    rel_norm: f64,
    where_: String,
}

fn worst(ours: &[Vec<f32>], theirs: &[Vec<f32>]) -> Dev {
    let mut d = Dev {
        abs: 0.0,
        rel_norm: 0.0,
        where_: String::new(),
    };
    for ((o, t), spec) in ours.iter().zip(theirs).zip(MIC.tensors()) {
        let mut diff2 = 0.0f64;
        let mut norm2 = 0.0f64;
        for (&a, &b) in o.iter().zip(t) {
            let e = (a - b).abs();
            if e > d.abs {
                d.abs = e;
                d.where_ = spec.name.clone();
            }
            diff2 += (e as f64).powi(2);
            norm2 += (b as f64).powi(2);
        }
        if norm2 > 0.0 {
            d.rel_norm = d.rel_norm.max((diff2 / norm2).sqrt());
        }
    }
    d
}

fn check(recipe: Recipe, name: &str, w_tol: f32) {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: 2,
        len: 32,
    };
    let initial = tensors("initiale");
    let run = train::golden_run(&MIC, &initial, &batch, recipe, 10).unwrap();

    let want = losses(name);
    assert_eq!(want.len(), 10);
    let mut loss_rel = 0.0f64;
    for (k, (&o, &w)) in run.losses.iter().zip(&want).enumerate() {
        let r = (o as f64 - w).abs() / w.abs();
        loss_rel = loss_rel.max(r);
        assert!(
            r <= train::TOL_LOSS_REL,
            "{name}: eroarea pasului {}: {o} fata de {w}",
            k + 1
        );
    }
    let g = worst(&run.grad_step1, &tensors(&format!("{name}/grad_pas1")));
    let w1 = worst(&run.after_step1, &tensors(&format!("{name}/dupa_pas1")));
    let w10 = worst(&run.after_last, &tensors(&format!("{name}/dupa_pas10")));
    println!(
        "kt c160p3 {name} eroare_rel={loss_rel:.2e} grad_abs={:.2e} ({}) grad_rel_norma={:.2e} w1_abs={:.2e} ({}) w10_abs={:.2e} ({}) eroarea_1={} eroarea_10={}",
        g.abs,
        g.where_,
        g.rel_norm,
        w1.abs,
        w1.where_,
        w10.abs,
        w10.where_,
        run.losses[0],
        run.losses[9]
    );
    assert!(
        g.abs <= train::TOL_GRAD_ABS,
        "{name}: gradient {} in {}",
        g.abs,
        g.where_
    );
    assert!(
        g.rel_norm <= train::TOL_GRAD_REL_NORM,
        "{name}: gradient rel {}",
        g.rel_norm
    );
    assert!(
        w1.abs <= w_tol,
        "{name}: greutati dupa pasul 1: {} in {}",
        w1.abs,
        w1.where_
    );
    assert!(
        w10.abs <= w_tol,
        "{name}: greutati dupa pasul 10: {} in {}",
        w10.abs,
        w10.where_
    );
}

#[test]
fn t151a_sgd_matches_pytorch() {
    check(Recipe::Sgd, "sgd", train::TOL_SGD_W_ABS);
}

#[test]
fn t151b_adamw_513_matches_pytorch() {
    check(Recipe::AdamW513, "adamw_513", train::TOL_ADAMW_W_ABS);
}

/// The check can fail: a wrong gradient (one tensor scaled), a wrong recipe or a shifted batch must be caught.
#[test]
fn t151c_the_comparison_catches_mistakes() {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: 2,
        len: 32,
    };
    let initial = tensors("initiale");
    let want_g = tensors("sgd/grad_pas1");
    let (_, mut g) = train::loss_and_grad(&MIC, &initial, &batch).unwrap();
    assert!(worst(&g, &want_g).abs <= train::TOL_GRAD_ABS);
    // One tensor's gradient 0.2 % too large (in_proj_weight of layer 0) must be caught.
    for x in g[4].iter_mut() {
        *x *= 1.002;
    }
    let d = worst(&g, &want_g);
    assert!(d.rel_norm > train::TOL_GRAD_REL_NORM || d.abs > train::TOL_GRAD_ABS);
    // SGD weights compared with AdamW's: far outside.
    let sgd = train::golden_run(&MIC, &initial, &batch, Recipe::Sgd, 1).unwrap();
    assert!(worst(&sgd.after_step1, &tensors("adamw_513/dupa_pas1")).abs > train::TOL_ADAMW_W_ABS);
    // Targets shifted by one position: the loss is no longer PyTorch's.
    let mut shifted = targets.clone();
    shifted.rotate_left(1);
    let b2 = Batch {
        targets: &shifted,
        ..batch
    };
    let (l, _) = train::loss_and_grad(&MIC, &initial, &b2).unwrap();
    assert!((l as f64 - losses("sgd")[0]).abs() / losses("sgd")[0] > train::TOL_LOSS_REL);
}
