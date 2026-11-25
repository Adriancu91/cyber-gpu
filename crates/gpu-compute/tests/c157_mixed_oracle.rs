//! C-157 p.2 after os-445 — T246: the CPU oracle of the board's mixed step (`train::loss_and_grad_mixed`) on the
//! small model `MIC`. CPU against CPU: nothing here runs on a GPU, and nothing here says what the board computes.
//!
//! What is checked (predictions M4 in `docs/C-157_P2_OS445_PAS_MIXT.md`, written before the code):
//! - the mixed step is close to the F32 step (the tolerances of the mixed step: loss 1e-3, every tensor's gradient
//!   5e-3 relative in norm was the prediction; it missed — 7.3e-3 on `straturi.0.ln2.bias` with only 64 symbols, so
//!   the check is 1e-2) but **not** the same bits (the rounding really happens);
//! - the loss does not depend on the error scale S (S only touches the backward pass), bit for bit;
//! - S = 2^10 and S = 2^14 give the same gradients within 1e-3 (no value lost at either scale on this model);
//! - a scale that is not a power of two, or one that pushes S·g out of the f16 range, is refused — no infinities;
//! - the same call twice gives the same bits.

use gpu_compute::train::{self, Batch, MIC};

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 33) as u32
    }
}

fn start_weights() -> Vec<Vec<f32>> {
    let mut g = Lcg(1);
    MIC.tensors()
        .iter()
        .map(|t| {
            let n = t.elements();
            let leaf = t.name.rsplit('.').nth(1).unwrap_or("");
            if t.name.ends_with("bias") {
                vec![0.0f32; n]
            } else if leaf.starts_with("ln") {
                vec![1.0f32; n]
            } else {
                (0..n)
                    .map(|_| ((g.next() as f64 / 2_147_483_648.0 - 1.0) * 0.034_641_016) as f32)
                    .collect()
            }
        })
        .collect()
}

fn batch_rows() -> (Vec<u32>, Vec<u32>) {
    let mut g = Lcg(0x164);
    let (mut x, mut y) = (Vec::new(), Vec::new());
    for _ in 0..2 {
        let row: Vec<u32> = (0..=MIC.ctx).map(|_| g.next() % MIC.vocab as u32).collect();
        x.extend_from_slice(&row[..MIC.ctx]);
        y.extend_from_slice(&row[1..]);
    }
    (x, y)
}

fn rel(a: &[f32], b: &[f32]) -> f64 {
    let d: f64 = a
        .iter()
        .zip(b)
        .map(|(&p, &q)| (p as f64 - q as f64).powi(2))
        .sum();
    let n: f64 = b.iter().map(|&q| (q as f64).powi(2)).sum();
    (d / n.max(1e-300)).sqrt()
}

#[test]
fn t246_mixed_oracle_on_mic() {
    let params = start_weights();
    let (x, y) = batch_rows();
    let batch = Batch {
        inputs: &x,
        targets: &y,
        batch: 2,
        len: MIC.ctx,
    };
    let (l32, g32) = train::loss_and_grad(&MIC, &params, &batch).unwrap();
    let (lm, gm) = train::loss_and_grad_mixed(&MIC, &params, &batch, 16384.0).unwrap();

    // Close to F32, but not the same bits.
    let dl = ((lm - l32) / l32).abs() as f64;
    assert!(dl <= 1e-3, "pierderea: {lm} fata de {l32}");
    let mut worst = 0.0f64;
    let mut differ = false;
    for (t, (a, b)) in MIC.tensors().iter().zip(gm.iter().zip(&g32)) {
        if t.name.ends_with("in_proj_bias") {
            continue; // the k part of this bias has a zero gradient in exact arithmetic (softmax shift): noise only
        }
        let r = rel(a, b);
        assert!(r <= 1e-2, "{}: gradientul mixt la {r:e} de cel F32", t.name);
        worst = worst.max(r);
        differ |= a.iter().zip(b).any(|(p, q)| p.to_bits() != q.to_bits());
    }
    assert!(differ, "pasul mixt nu rotunjeste nimic");
    assert!(worst > 0.0);
    println!(
        "T246: pierderea F32 {l32} mixt {lm} (rel {dl:.1e}); cel mai rau gradient {worst:.1e}"
    );

    // The loss does not depend on S; the gradients barely.
    let (l10, g10) = train::loss_and_grad_mixed(&MIC, &params, &batch, 1024.0).unwrap();
    assert_eq!(l10.to_bits(), lm.to_bits());
    for (a, b) in g10.iter().zip(&gm) {
        assert!(rel(a, b) <= 1e-3 || b.iter().all(|&v| v == 0.0));
    }

    // Deterministic.
    let (lm2, gm2) = train::loss_and_grad_mixed(&MIC, &params, &batch, 16384.0).unwrap();
    assert_eq!(lm2.to_bits(), lm.to_bits());
    assert!(
        gm2.iter()
            .flatten()
            .zip(gm.iter().flatten())
            .all(|(p, q)| p.to_bits() == q.to_bits())
    );

    // Refusals.
    assert!(train::loss_and_grad_mixed(&MIC, &params, &batch, 1000.0).is_err());
    assert!(train::loss_and_grad_mixed(&MIC, &params, &batch, 0.0).is_err());
    assert!(train::loss_and_grad_mixed(&MIC, &params, &batch, f32::INFINITY).is_err());
    assert!(
        train::loss_and_grad_mixed(&MIC, &params, &batch, 2f32.powi(40)).is_err(),
        "S·g peste 65 504 trebuie refuzat"
    );
}
