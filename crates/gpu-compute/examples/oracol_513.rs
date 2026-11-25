//! C-157 point 2 after os-395: the CPU oracle of the #513 training run that the OS did on the board
//! (`test-placa --antrenare-513`, P6-172), with the same start weights, the same synthetic batches and the same
//! optimizer, so the board's losses and gradient norms can be judged.
//!
//! What the OS used (cyber-universe `kernel/src/drivers/gpu_i14.rs::train_513_resident` and
//! `kernel/src/shell/antrenare_mini.rs`): `train::M513`; batch 2 × 512; start weights uniform in ±0.034641016
//! (std 0.02) from MT19937 seeded with 1, biases 0, LayerNorm weights 1; batches from MT19937 seeded with 0x164
//! (`next_u32() % 16384`, 513 symbols per row: inputs = first 512, targets = last 512); AdamW lr 1e-3, betas
//! (0.9, 0.999), eps 1e-8, weight decay 0.1; clipping at 1.0; 10 steps.
//!
//! Run: `cargo run --release -p gpu-compute --example oracol_513 -- [steps] [--mixt]` (single thread, minutes per step;
//! `--mixt` = the mixed step's oracle, `train::loss_and_grad_mixed` with S = 2^14).
//! Output: one line per step `pas k: pierderea L norma N secunde S`. Nothing here touches a board.

use gpu_compute::train::{self, AdamW, Batch};
use std::time::Instant;

/// MT19937 with 32-bit outputs, seeded like `std::mt19937(seed)` (the standard reference algorithm).
struct Mt {
    s: [u32; 624],
    i: usize,
}

impl Mt {
    fn new(seed: u32) -> Mt {
        let mut s = [0u32; 624];
        s[0] = seed;
        for j in 1..624 {
            s[j] = 1_812_433_253u32
                .wrapping_mul(s[j - 1] ^ (s[j - 1] >> 30))
                .wrapping_add(j as u32);
        }
        Mt { s, i: 624 }
    }

    fn next(&mut self) -> u32 {
        if self.i >= 624 {
            for k in 0..624 {
                let y = (self.s[k] & 0x8000_0000) | (self.s[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.s[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.s[k] = v;
            }
            self.i = 0;
        }
        let mut y = self.s[self.i];
        self.i += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }
}

const HALF_WIDTH: f64 = 0.034_641_016;

/// The error scale of the board's mixed #513 step (`nvidia_ga104::train_step::LOSS_SCALE_513`, 2^14); written here
/// as a number so the example needs no driver crate.
const LOSS_SCALE_513: f32 = 16384.0;

fn start_weights(a: &train::Arch) -> Vec<Vec<f32>> {
    let mut g = Mt::new(1);
    a.tensors()
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
                    .map(|_| ((g.next() as f64 / 4_294_967_296.0 * 2.0 - 1.0) * HALF_WIDTH) as f32)
                    .collect()
            }
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    // `--mixt`: the board's mixed step (C-157 p.2 after os-445) — the matmul inputs rounded to f16, S = 2^14.
    let mixed = args.iter().any(|a| a == "--mixt");
    let steps: usize = args.iter().find_map(|s| s.parse().ok()).unwrap_or(10);
    let a = train::M513;
    let (bsz, tl) = (2usize, 512usize);
    let mut g = Mt::new(0x164);
    let batches: Vec<(Vec<u32>, Vec<u32>)> = (0..10)
        .map(|_| {
            let (mut x, mut y) = (Vec::new(), Vec::new());
            for _ in 0..bsz {
                let row: Vec<u32> = (0..=tl).map(|_| g.next() % a.vocab as u32).collect();
                x.extend_from_slice(&row[..tl]);
                y.extend_from_slice(&row[1..]);
            }
            (x, y)
        })
        .collect();
    let mut params = start_weights(&a);
    println!(
        "oracol #513 (procesor): {} greutati, lot {bsz} x {tl}, primele simboluri ale lotului 1: {:?}",
        params.iter().map(|p| p.len()).sum::<usize>(),
        &batches[0].0[..4]
    );
    let mut adam = AdamW::new(&params, 1e-3, 0.9, 0.999, 1e-8, 0.1);
    for (k, (x, y)) in batches.iter().enumerate().take(steps) {
        let t0 = Instant::now();
        let batch = Batch {
            inputs: x,
            targets: y,
            batch: bsz,
            len: tl,
        };
        let (loss, mut grads) = if mixed {
            train::loss_and_grad_mixed(&a, &params, &batch, LOSS_SCALE_513).expect("pasul mixt")
        } else {
            train::loss_and_grad(&a, &params, &batch).expect("pasul")
        };
        let norm = train::grad_norm(&grads);
        train::clip_grad_norm(&mut grads, 1.0);
        adam.step(&mut params, &grads).expect("adamw");
        println!(
            "pas {}{}: pierderea {:.4} norma {:.4} (exact {:.7} / {:.7}) secunde {:.0}",
            k + 1,
            if mixed { " (mixt)" } else { "" },
            loss,
            norm,
            loss,
            norm,
            t0.elapsed().as_secs_f64()
        );
    }
}
