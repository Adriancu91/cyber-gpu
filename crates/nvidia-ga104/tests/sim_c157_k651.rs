//! C-157 point 4 (T148), SIMULARE: Kronos **#651** (24 layers, E 768, 12 heads, vocabulary 16 384, context 512;
//! hub `integrare/kronos/MODEL_651.md`) on the simulated work channel, set up with the OS's constants of today
//! (`cyber-universe` `gpu_i14.rs`: pushbuffer area 32 KiB, payload stride `0x100` per doorbell, upload pieces of 8 MiB,
//! 4 MiB VRAM reserve). Weights GENERATED here, not #651's; the file itself is read only by the OS.
//!
//! What must hold:
//! - the shapes give exactly the file's tensor bytes (206 665 728 of the 206 898 464; the rest is the header with the
//!   tokenizer) and its 292 tensors (98 Q8_0, 194 F32);
//! - the **plain** list does NOT fit the OS's stride (388 launches + 1 > `0x100`) and its pushbuffer needs more than
//!   32 KiB in `EachLaunch`; the OS's check today (`ModelLimits::of`, always the plain list) would therefore refuse
//!   #651 before any upload, even when only the glued list runs;
//! - the **glued** lists (`Fused`, `FusedLn`, `Fused32`, with or without `argmax`) fit all of the OS's constants;
//! - a whole #651-shaped model uploaded through the arena and run with the glued list (the cycle's default since
//!   P6-112), one doorbell per position, gives the CPU engine's tokens.
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here says anything about the RTX 3070 Ti.

mod common;

use common::sys_write;
use common::token_sim::TokenSim;
use gpu_compute::model::{self as cm, Caches, ModelWeights};
use gpu_compute::q8;
use gpu_core::error::GpuResult;
use nvidia_ga104::kronos_diag::{self as kd, EmbedBy, SeqMode, TokenList, Way};
use nvidia_ga104::kronos_gen::{
    GenPlan, Generation, Stop, TokenCheck, TokenOutput, TokenRunner, VramBudget,
};
use nvidia_ga104::kronos_layer::{self as kl, Prog};
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::kronos_speed::{
    K513_CONTEXT, K513_LAYERS, K513_SHAPE, K513_VOCAB, K651_CONTEXT, K651_LAYERS, K651_SHAPE,
    K651_VOCAB, Limit, ModelLimits,
};
use nvidia_ga104::transfer::I11_DEFAULT_BYTES;

/// The OS's constants today (`cyber-universe/kernel/src/drivers/gpu_i10.rs` `PUSH_BYTES`, `gpu_i14.rs` `STRIDE`,
/// `VRAM_RESERVE`), read from its code on 21 Nov 2026.
const OS_PUSH: u64 = 0x8000;
const OS_STRIDE: u32 = 0x100;
const OS_RESERVE: u64 = 4 << 20;
/// The RTX 3070 Ti: 8 GiB, of which ~1 GiB is kept by GSP-RM, the golden image and the screen (a round guess).
const FREE: u64 = 7 << 30;

const WAY: Way = Way {
    mode: SeqMode::EachLaunch,
    embed: EmbedBy::Program,
};

const GLUED: [TokenList; 4] = [
    TokenList::Fused,
    TokenList::FusedLn,
    TokenList::Fused32,
    TokenList::FusedLn32,
];

/// The bytes of the tensors of the KGR1 file of a model with these shapes (Q8_0 matrices, F32 vectors).
fn tensor_bytes(layers: usize, e: usize, vocab: usize, context: usize) -> (u64, usize, usize) {
    let q = |r: usize, c: usize| (r * c / q8::BLOCK * q8::BLOCK_BYTES) as u64;
    let f = |n: usize| 4 * n as u64;
    let layer = q(3 * e, e) + q(e, e) + q(4 * e, e) + q(e, 4 * e)  // the 4 matrices
        + f(e) * 4 + f(3 * e) + f(e) + f(4 * e) + f(e); // ln1, ln2 (g, b), the 4 biases
    let bytes = q(vocab, e) + q(context, e) + 2 * f(e) + layers as u64 * layer;
    // emb_token, emb_pozitie + 4 matrices per layer; ln_f g/b + 8 vectors per layer.
    (bytes, 2 + 4 * layers, 2 + 8 * layers)
}

#[test]
fn t148a_the_shapes_are_the_files() {
    let (bytes, q8s, f32s) = tensor_bytes(K651_LAYERS, K651_SHAPE.e, K651_VOCAB, K651_CONTEXT);
    assert_eq!(bytes, 206_665_728);
    assert_eq!((q8s, f32s, q8s + f32s), (98, 194, 292));
    // MODEL_651.md: the file is 206 898 464 bytes; the rest is the header (keys, tokenizer of 214 851 characters,
    // the tensor list, zeros to 32).
    let header = 206_898_464 - bytes;
    assert!((214_851..400_000).contains(&header), "{header}");
    // #513: 12 layers, 148 tensors.
    let (_, a, b) = tensor_bytes(K513_LAYERS, K513_SHAPE.e, K513_VOCAB, K513_CONTEXT);
    assert_eq!(a + b, 148);
    // The blob holds all the tensors + the scratch + the final vector and the logits, each 256-aligned.
    let lim = ModelLimits::k651();
    assert!(lim.vram.blob >= bytes && lim.vram.blob - bytes < 1 << 20);
    assert_eq!(lim.pieces, 25, "the blob in pieces of 8 MiB");
}

#[test]
fn t148b_the_plain_list_does_not_fit_the_os_the_glued_ones_do() {
    // What the OS checks today: `ModelLimits::of` = the plain list, whatever runs.
    let plain = ModelLimits::k651();
    assert_eq!(plain.launches, 388);
    assert_eq!(plain.stride, 389);
    let refused = plain.check(OS_PUSH, OS_STRIDE, FREE, FREE, OS_RESERVE);
    assert!(
        refused.iter().any(|r| matches!(r, Limit::Stride { .. })),
        "{refused:?}"
    );
    assert!(
        refused.iter().any(|r| matches!(r, Limit::PushArea { .. })),
        "{refused:?}"
    );
    assert_eq!(refused.len(), 2, "VRAM is not the problem: {refused:?}");
    for r in &refused {
        println!("SIMULARE #651, lista simpla: refuzat: {}", r.line());
    }
    // With the plain list (alone or compared): stride 0x200 and a pushbuffer area of 64 KiB.
    assert_eq!(plain.push_area, 64 << 10);
    assert!(
        plain
            .check(plain.push_area, 0x200, FREE, FREE, OS_RESERVE)
            .is_empty()
    );
    let both = ModelLimits::k651_lists(&[TokenList::Plain, TokenList::Fused]).unwrap();
    assert_eq!((both.launches, both.stride), (388, 389));
    assert_eq!(both.push_area, plain.push_area);

    // The glued lists alone: everything fits the OS's constants as they are.
    for list in GLUED {
        let lim = ModelLimits::k651_lists(&[list]).unwrap();
        assert_eq!(lim.launches, 7 * 24 + 3 + 1, "{list:?}");
        assert!(lim.stride <= OS_STRIDE, "{list:?}");
        assert!(lim.push_area <= OS_PUSH, "{list:?}");
        assert!(
            lim.check(OS_PUSH, OS_STRIDE, FREE, FREE, OS_RESERVE)
                .is_empty(),
            "{list:?}"
        );
        // + argmax on the board (C-155 point 3).
        let top = ModelLimits::of_lists(
            &K651_SHAPE,
            K651_LAYERS,
            K651_VOCAB,
            K651_CONTEXT,
            I11_DEFAULT_BYTES,
            &[list],
            &[Prog::Argmax],
        )
        .unwrap();
        assert_eq!(top.launches, lim.launches + 1);
        assert!(top.stride <= OS_STRIDE && top.push_area <= OS_PUSH);
        assert!(top.vram.programs > plain.vram.programs);
    }
    let fused = ModelLimits::k651_lists(&[TokenList::Fused]).unwrap();
    for l in fused.lines() {
        println!("SIMULARE #651, lista lipita: {l}");
    }
    assert!(ModelLimits::k651_lists(&[]).is_err());

    // #513 keeps fitting either way (nothing changes for it).
    let k513 = ModelLimits::k513();
    assert!(
        k513.check(OS_PUSH, OS_STRIDE, FREE, FREE, OS_RESERVE)
            .is_empty()
    );
    let k513f = ModelLimits::of_lists(
        &K513_SHAPE,
        K513_LAYERS,
        K513_VOCAB,
        K513_CONTEXT,
        I11_DEFAULT_BYTES,
        &[TokenList::Fused],
        &[],
    )
    .unwrap();
    assert_eq!(k513f.launches, 88);
}

#[test]
fn t148c_the_vram_budget() {
    let all = ModelLimits::of_lists(
        &K651_SHAPE,
        K651_LAYERS,
        K651_VOCAB,
        K651_CONTEXT,
        I11_DEFAULT_BYTES,
        &[
            TokenList::Plain,
            TokenList::Fused,
            TokenList::FusedLn,
            TokenList::Fused32,
            TokenList::FusedLn32,
        ],
        &[Prog::Argmax],
    )
    .unwrap();
    let v = all.vram;
    println!("SIMULARE {}", v.line(K651_LAYERS));
    // Weights ~197 MiB, caches 24 · 2 · 512 · 768 · 4 = 72 MiB, the rest small.
    assert_eq!(v.caches, 72 << 20);
    assert!(
        v.total() > 270 << 20 && v.total() < 290 << 20,
        "{}",
        v.total()
    );
    assert!(v.programs + v.launches + v.tables < 2 << 20);
    // Twice #513 in the layers, the same embeddings and logits.
    let k513 = ModelLimits::k513().vram;
    assert!(v.total() < 2 * k513.total());
    // Fits a free VRAM of 7 GiB with room for 20 such models; refused on a board with < ~275 MiB free.
    assert!(v.check(FREE, FREE, OS_RESERVE).is_ok());
    assert!(v.check(272 << 20, 272 << 20, OS_RESERVE).is_err());
    // The blob must be one contiguous piece of ~198 MiB.
    assert!(v.check(FREE, 150 << 20, OS_RESERVE).is_err());
    // `with_programs` counts every program once, and none of the plain list again.
    let p0 = VramBudget::of(&K651_SHAPE, K651_LAYERS, K651_VOCAB, K651_CONTEXT).unwrap();
    assert_eq!(p0.with_programs(&kl::PROGRAMS), p0);
    let one = p0.with_programs(&[Prog::Argmax, Prog::Argmax]);
    assert_eq!(
        one.programs - p0.programs,
        (Prog::Argmax.code().len() as u64).next_multiple_of(0x1000)
    );
    assert_eq!(one.tables - p0.tables, 0x1000);
}

struct Runner<'s, 'a> {
    s: &'s mut TokenSim<'a>,
    list: TokenList,
}

impl TokenRunner for Runner<'_, '_> {
    fn run_token(&mut self, token: usize, pos: usize) -> GpuResult<TokenOutput> {
        let before = self.s.times.len();
        let (m, vas) = (self.s.m, self.s.vas.clone());
        let bytes = (4 * m.vocab as u64).next_multiple_of(16);
        let down = kl::copy_launch(vas.logits, self.s.host.va, bytes)?;
        let host = self.s.host.clone();
        sys_write(&mut self.s.r.sys, &host, &vec![0xee; bytes as usize]);
        kd::send_token_list(
            self.s,
            self.list,
            &m.weights(),
            &vas,
            token,
            pos,
            WAY,
            &[down],
        )?;
        let mine = &self.s.times[before..];
        Ok(TokenOutput {
            logits: self.s.ram_logits(),
            start_ns: mine.first().map_or(0, |t| t.0),
            end_ns: mine.last().map_or(0, |t| t.1),
            busy_ns: mine.iter().map(|t| t.2).sum(),
        })
    }
}

struct Cpu<'a> {
    w: ModelWeights<'a>,
    caches: Caches,
}

impl TokenCheck for Cpu<'_> {
    fn cpu_logits(&mut self, token: usize, pos: usize) -> GpuResult<Vec<f32>> {
        cm::step_cpu(&self.w, &mut self.caches, token, pos)
    }
}

#[test]
fn t148d_651_shape_runs_glued_with_the_os_constants() {
    let lim = ModelLimits::k651_lists(&[TokenList::Fused]).unwrap();
    let m = km::test_model(K651_SHAPE, K651_LAYERS, K651_VOCAB, K651_CONTEXT, 651).unwrap();
    assert_eq!(m.layers.len(), 24);
    // The OS's pushbuffer area (32 KiB), the blob through the arena in pieces of 8 MiB, the stride 0x100 (TokenSim).
    let mut s = TokenSim::with(&m, 0, OS_PUSH, Some(I11_DEFAULT_BYTES)).unwrap();
    assert_eq!(s.upload.len() as u64, lim.pieces);
    let vas = s.vas.clone();
    let last = vas.layers.last().unwrap().wproj;
    let got = {
        use nvidia_ga104::kronos_diag::TokenIo;
        s.read(&[(last, m.layers[23].wproj.len() / 4)]).unwrap()
    };
    let words: Vec<u32> = m.layers[23]
        .wproj
        .chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(
        got[0], words,
        "the 24th layer arrived whole through the arena"
    );

    // "Ana are mere." (first 3 tokens) + 2 greedy tokens: 4 positions, the glued list, judged by the CPU engine.
    let plan = GenPlan::new(&[65, 1988, 1585], 2, m.vocab, m.context).unwrap();
    let mut g = Generation::new(plan);
    let mut cpu = Cpu {
        w: m.weights(),
        caches: Caches::new(K651_LAYERS, K651_SHAPE.e),
    };
    let mut run = Runner {
        s: &mut s,
        list: TokenList::Fused,
    };
    let stop = g.run(&mut run, Some(&mut cpu)).clone();
    assert_eq!(stop, Stop::Gata, "{}", stop.line());
    assert_eq!(g.generated().len(), 2);
    assert!(g.worst_abs <= cm::LOGIT_ABS, "worst {}", g.worst_abs);
    // One doorbell per position, 172 launches each: under the stride, in half the 32 KiB area.
    assert_eq!(s.submissions, vec![lim.launches; 4]);
    assert!((lim.launches as u32) < OS_STRIDE);
    assert!(2 * 4 * kd::words(SeqMode::EachLaunch, lim.launches) as u64 <= OS_PUSH);
    println!(
        "SIMULARE #651 (greutati generate), lista lipita: 4 pozitii, {} lansari pe pozitie, abaterea maxima fata de procesor {:e}, simbolurile generate {:?}",
        lim.launches,
        g.worst_abs,
        g.generated()
    );
}
