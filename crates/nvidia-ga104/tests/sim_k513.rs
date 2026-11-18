//! C-142 (1), SIMULARE: a model with **#513's shape** (12 layers, E 768, 12 heads, vocabulary 16 384, context 512;
//! weights GENERATED here, not #513's) on the simulated work channel, set up as the OS will for `kronos --gpu`:
//! pushbuffer area 32 KiB, the whole blob (~106 MiB) uploaded through a host-RAM arena in pieces of at most 8 MiB
//! (one `copy_b128` launch each, I11), then every position sent as ONE submission of 196 launches (the token's 195 +
//! the copy of the logits down into RAM) in the way that passed on the board at `os-306` (`EachLaunch`: the shader
//! caches invalidated before every launch), the logits read from RAM and judged against the CPU engine, greedy
//! generation. The speed counters (`kronos_speed::Speed`) are filled as the OS will fill them and their journal rows
//! read back. The "GPU" is `gpu_sim::cmdproc` with the program models; its clock is the model's. Nothing here says
//! anything about the RTX 3070 Ti, its speed, or #513's own weights.

mod common;

use std::time::Instant;

use common::token_sim::TokenSim;
use gpu_compute::model::{self as cm, Caches, ModelWeights};
use gpu_core::diag::Journal;
use gpu_core::error::GpuResult;
use nvidia_ga104::kronos_diag::{self as kd, EmbedBy, SeqMode, TokenIo, Way};
use nvidia_ga104::kronos_gen::{
    GenPlan, Generation, Stop, TokenCheck, TokenOutput, TokenRunner, VramBudget,
};
use nvidia_ga104::kronos_model as km;
use nvidia_ga104::kronos_speed::{
    self as ks, K513_CONTEXT, K513_LAYERS, K513_SHAPE, K513_VOCAB, ModelLimits, Source, Speed,
};
use nvidia_ga104::transfer::I11_DEFAULT_BYTES;

/// The way `os-306` chose on the board (C-139 repair): invalidation before every launch, embedding by the program.
const WAY: Way = Way {
    mode: SeqMode::EachLaunch,
    embed: EmbedBy::Program,
};

struct Runner<'s, 'a> {
    s: &'s mut TokenSim<'a>,
    speed: Speed,
}

impl TokenRunner for Runner<'_, '_> {
    fn run_token(&mut self, token: usize, pos: usize) -> GpuResult<TokenOutput> {
        let before = self.s.times.len();
        let t0 = Instant::now();
        let logits = self.s.run_token_ram(token, pos, WAY)?;
        let host_us = t0.elapsed().as_micros() as u64;
        let mine = &self.s.times[before..];
        let out = TokenOutput {
            logits,
            start_ns: mine.first().map_or(0, |t| t.0),
            end_ns: mine.last().map_or(0, |t| t.1),
            busy_ns: mine.iter().map(|t| t.2).sum(),
        };
        self.speed.add_token(pos, &out, Some(host_us))?;
        Ok(out)
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
fn k513_shape_runs_on_the_simulated_channel_with_32_kib_and_the_arena() {
    // The limits, from the shapes alone (what the OS checks before uploading anything).
    let lim = ModelLimits::k513();
    assert_eq!(lim.launches, 196);
    assert_eq!(lim.push_each, 9440);
    assert_eq!(lim.push_area, 32 << 10);
    assert!(lim.stride <= 0x100);
    for l in lim.lines() {
        println!("SIMULARE {l}");
    }

    let m = km::test_model(K513_SHAPE, K513_LAYERS, K513_VOCAB, K513_CONTEXT, 513).unwrap();
    assert_eq!(VramBudget::of_model(&m).unwrap(), lim.vram);
    assert_eq!(m.shape.d(), 64);

    // 32 KiB pushbuffer area, the blob uploaded through the arena in pieces of at most 8 MiB.
    let t0 = Instant::now();
    let mut s = TokenSim::with(&m, 0, 32 << 10, Some(I11_DEFAULT_BYTES)).unwrap();
    let upload_host_us = t0.elapsed().as_micros() as u64;
    assert_eq!(
        s.upload.len() as u64,
        lim.pieces,
        "pieces as the limits say"
    );
    assert!(
        s.upload
            .iter()
            .all(|&(b, ns)| b <= I11_DEFAULT_BYTES && ns > 0)
    );
    assert!(
        s.upload[..s.upload.len() - 1]
            .iter()
            .all(|&(b, _)| b == I11_DEFAULT_BYTES)
    );
    let blob_bytes: u64 = s.upload.iter().map(|&(b, _)| b).sum();
    assert_eq!(blob_bytes.next_multiple_of(0x1000), lim.vram.blob);

    // The weights arrived whole: the token table (the first piece), the head of the last layer (the last pieces).
    let vas = s.vas.clone();
    let words = |b: &[u8]| -> Vec<u32> {
        b.chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    let last = vas.layers.last().unwrap().wproj;
    let got = s
        .read(&[
            (vas.emb_tok, m.emb_tok.len() / 4),
            (vas.emb_pos, m.emb_pos.len() / 4),
            (last, m.layers[K513_LAYERS - 1].wproj.len() / 4),
        ])
        .unwrap();
    assert_eq!(got[0], words(&m.emb_tok), "token table through the arena");
    assert_eq!(
        got[1],
        words(&m.emb_pos),
        "position table through the arena"
    );
    assert_eq!(
        got[2],
        words(&m.layers[K513_LAYERS - 1].wproj),
        "last layer through the arena"
    );

    let mut speed = Speed::new(Source::Simulare);
    for &(b, ns) in &s.upload {
        speed.add_piece(b, ns, None);
    }
    speed.upload.host_us = Some(upload_host_us);

    // "Ana are mere." (first 3 tokens of the control text) + 2 greedy tokens: 4 positions on the "GPU".
    let plan = GenPlan::new(&[65, 1988, 1585], 2, m.vocab, m.context).unwrap();
    let mut g = Generation::new(plan);
    let mut cpu = Cpu {
        w: m.weights(),
        caches: Caches::new(K513_LAYERS, K513_SHAPE.e),
    };
    let mut run = Runner { s: &mut s, speed };
    let stop = g.run(&mut run, Some(&mut cpu)).clone();
    assert_eq!(stop, Stop::Gata, "{}", stop.line());
    assert_eq!(g.generated().len(), 2);
    assert!(g.worst_abs <= cm::LOGIT_ABS, "worst {}", g.worst_abs);
    let speed = run.speed;

    // One doorbell per position, 196 launches each, in the 32 KiB area.
    assert_eq!(s.submissions, vec![196; 4]);
    assert_eq!(4 * kd::words(SeqMode::EachLaunch, 196), 9440);

    // The counters and their journal rows (as `gpu-jurnal` will carry them).
    assert_eq!(speed.tokens.len(), 4);
    assert_eq!(speed.upload.pieces as u64, lim.pieces);
    let sum = speed.summary();
    assert!(sum.median_busy_ns > 0 && sum.median_span_ns >= sum.median_busy_ns);
    // The model's clock: 250 µs per launch → ~49 ms per position (says nothing about the board).
    assert!(sum.median_busy_ns >= 196 * 250_000);
    let mut j = Journal::new(256);
    speed.journal(&mut j, true);
    let text: String = j
        .events()
        .map(|e| format!("[  400.000000] gpu: {e}\n"))
        .collect();
    let p = ks::parse_journal(&text);
    assert_eq!(p.source, Some(0));
    assert_eq!(p.up_pieces, Some(lim.pieces as u32));
    assert_eq!(p.up_kib, Some(blob_bytes.div_ceil(1024) as u32));
    assert_eq!(p.n_tokens, Some(4));
    assert_eq!(
        p.tokens.iter().map(|t| t.0).collect::<Vec<_>>(),
        vec![0, 1, 2, 3]
    );
    assert_eq!(p.median_busy_us, Some((sum.median_busy_ns / 1000) as u32));
    for l in speed.lines() {
        println!("{l}");
    }
}

#[test]
fn the_os_check_refuses_16_kib_and_a_small_vram_before_any_upload() {
    let lim = ModelLimits::k513();
    // The OS's check (`run_model`): one submission must fit in half the area (one in flight, the next written).
    let refused = lim.check(16 << 10, 0x100, 8 << 30, 8 << 30, 4 << 20);
    assert_eq!(refused.len(), 1);
    assert!(refused[0].line().contains("zona de comenzi"));
    // The RTX 3070 Ti has 8 GiB; GSP-RM, the golden image and the screen take some: #513 needs well under 200 MiB.
    assert!(lim.vram.total() + (4 << 20) < 200 << 20);
    let refused = lim.check(32 << 10, 0x100, 100 << 20, 100 << 20, 4 << 20);
    assert_eq!(refused.len(), 2, "{refused:?}");
}
