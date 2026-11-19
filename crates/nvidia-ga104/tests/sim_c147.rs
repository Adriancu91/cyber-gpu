//! C-147 stage 0, SIMULARE: the measurements change nothing the GPU computes and count what they say.
//! (a) the kinds of `kronos_perf::token_kinds` are the launches `kronos_model::token_launches` really sends, one to
//! one, and the time per kind adds up to the `kv` busy time of the same tokens; (d) the VRAM → VRAM copy with
//! `copy_b128`, repeated in one submission, copies bit for bit and its records give one time per copy. The "GPU" is
//! `gpu_sim::cmdproc` with its own clock (250 µs per launch): nothing here says anything about the RTX 3070 Ti.

mod common;

use common::token_sim::TokenSim;
use common::{get_bytes, gpu_runs, put_bytes, rig, span};
use gpu_compute::layer::Shape;
use gpu_sim::cmdproc::{Needs, copy_b128_on_cpu};
use nvidia_ga104::gsp::golden::GoldenMemory;
use nvidia_ga104::gsp::work_queue::ChannelMemory;
use nvidia_ga104::kronos_diag::{EmbedBy, SeqMode, Way};
use nvidia_ga104::kronos_layer as kl;
use nvidia_ga104::kronos_model::{self as km, OwnedModel};
use nvidia_ga104::kronos_perf::{self as kp, Kind, KindProfile, VramCopy};
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::transfer::{self, COPY_B128};

fn small() -> OwnedModel {
    km::test_model(
        Shape {
            e: 32,
            heads: 4,
            eps: 1e-5,
        },
        2,
        512,
        32,
        7,
    )
    .unwrap()
}

#[test]
fn kinds_are_the_launches_sent_and_add_up_to_the_busy_time() {
    let m = small();
    for (mode, embed) in [
        (SeqMode::EachLaunch, EmbedBy::Program),
        (SeqMode::PerLayer, EmbedBy::Program),
        (SeqMode::EachLaunch, EmbedBy::Driver),
    ] {
        let mut t = TokenSim::new(&m, 0);
        let way = Way { mode, embed };
        let kinds = kp::token_kinds(2, embed == EmbedBy::Program, true);
        // One to one with the list the driver builds (+ the copy down the OS appends).
        let mut list = km::token_launches(&m.shape, m.vocab, m.context, &t.vas, 3, 0).unwrap();
        list.push(kl::copy_launch(t.vas.logits, t.host.va, 4 * m.vocab as u64).unwrap());
        let sent: Vec<_> = if embed == EmbedBy::Program {
            list.iter().collect()
        } else {
            list.iter().skip(1).collect()
        };
        assert_eq!(kinds.len(), sent.len());
        for (k, l) in kinds.iter().zip(&sent) {
            assert_eq!(k.prog(), l.prog, "{k:?}");
        }
        let mut p = KindProfile::new();
        let mut busy = Vec::new();
        for (pos, tok) in [3usize, 9, 1].into_iter().enumerate() {
            let (d0, t0) = (t.durations.len(), t.times.len());
            t.run_token_ram(tok, pos, way).unwrap();
            let ns: Vec<u64> = t.durations[d0..].concat();
            busy.push(t.times[t0..].iter().map(|x| x.2).sum::<u64>());
            p.add_token(&kinds, &ns).unwrap();
        }
        let s = p.summary();
        let total = s.last().unwrap();
        busy.sort_unstable();
        assert_eq!(total.token_median_ns, busy[1], "{mode:?} {embed:?}");
        assert_eq!(total.per_token as usize, kinds.len());
        // SIMULARE clock: every launch ≥ 250 µs; 2 q8 launches per layer kind "qkv" × 2 layers.
        let qkv = s.iter().find(|x| x.kind == Some(Kind::Qkv)).unwrap();
        assert_eq!(qkv.per_token, 2);
        assert!(qkv.launch_median_ns >= 250_000);
        let embed_row = s.iter().any(|x| x.kind == Some(Kind::Embed));
        assert_eq!(embed_row, embed == EmbedBy::Program);
        // The rows read back from the journal text say the same.
        let mut j = gpu_core::diag::Journal::new(256);
        p.journal(&mut j);
        let text: String = j
            .events()
            .map(|e| format!("[  126.000000] gpu: {e}\n"))
            .collect();
        let r = kp::parse_journal(&text);
        assert_eq!(r.tokens, Some(3));
        assert_eq!(r.launches, Some(kinds.len() as u32));
        assert_eq!(r.total_us, Some((busy[1] / 1000) as u32));
    }
}

#[test]
fn vram_copy_bit_for_bit_with_one_time_per_copy() {
    const FIRST: u32 = 0x700;
    let mut r = rig();
    let bytes: u64 = 1 << 20;
    let src = r.mem.alloc_mapped(bytes, 0x1000).unwrap();
    let dst = r.mem.alloc_mapped(bytes, 0x1000).unwrap();
    let mut m = || r.mem.alloc_mapped(0x1000, 0x1000).unwrap();
    let (prog, cb, q, rec) = (m(), m(), m(), m());
    let mut data = vec![0u8; bytes as usize];
    transfer::fill_pattern(&mut data, 0x147).unwrap();
    let c = VramCopy::new(src.va, dst.va, VramCopy::size_for(bytes), 3).unwrap();
    let w = r.mem.writer_mut();
    put_bytes(w, src.pa, &data);
    put_bytes(w, dst.pa, &vec![0xee; bytes as usize]);
    put_bytes(w, prog.pa, COPY_B128);
    put_bytes(w, cb.pa, &c.cbuf().unwrap());
    put_bytes(w, q.pa, &c.qmd(prog.va, cb.va).unwrap());
    put_bytes(w, rec.pa, &[0; 64]);
    let mut words = vec![0u32; c.words()];
    let mut pb = PushBuf::new(&mut words);
    c.encode(&mut pb, q.va, rec.va, FIRST).unwrap();
    assert_eq!(pb.len(), c.words());
    r.wq.submit(r.mem.writer_mut(), &words).unwrap();
    let loads: Vec<(u64, u64)> = [&src, &dst, &prog, &cb, &q, &rec]
        .iter()
        .map(|x| span(x))
        .collect();
    let cp = gpu_runs(
        &mut r,
        &[(COPY_B128, copy_b128_on_cpu, Needs::default())],
        &loads,
        &[span(&dst), (rec.va, c.record_bytes())],
    )
    .unwrap();
    assert_eq!(cp.timestamps.len(), 4);
    let back = get_bytes(r.mem.writer_mut(), dst.pa, bytes);
    assert_eq!(transfer::first_mismatch(&back, 0x147), None);
    let spots = c
        .spots()
        .map(|o| r.mem.writer_mut().read_u32(dst.pa + o).unwrap());
    assert_eq!(c.wrong_spots(&data, &spots), 0);
    let recs = get_bytes(r.mem.writer_mut(), rec.pa, c.record_bytes());
    let s = c.analyze(&recs, FIRST).unwrap();
    assert_eq!(s.each_ns.len(), 3);
    assert!(s.min_ns >= 250_000, "{s:?}");
    let rows = c.rows(&s, Some(0));
    assert_eq!(rows.len(), 2 + 3 + 6);
    // A record that never came is an error, not a number.
    let mut cut = recs.clone();
    cut[48..52].copy_from_slice(&0u32.to_le_bytes());
    assert!(c.analyze(&cut, FIRST).is_err());
}
