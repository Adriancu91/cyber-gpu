//! C-139 (after os-305), SIMULARE: the diagnosis of a token on the work channel (`kronos_diag`) and the repair
//! (`transfer::encode_sequence_fresh`).
//!
//! What the board showed: I10–I13 pass, `i14-simbol` fails at the first token with every logit finite and another
//! greedy token. The **hypothesis** (docs/os-305_ANALIZA_I14.md): a launch read, through its `ld.global…constant`
//! loads, a value that an earlier launch of the same submission had read and that was rewritten since — I14 is the
//! first step that does that (layer 2 reuses layer 1's scratch; the final LayerNorm reads x a third time). Here the
//! GPU model can be asked to behave that way (`CommandProcessor::model_stale_reads`): with it, the I13 sequence
//! still passes and the I14 sequence fails exactly like on the board, and the tests show what the diagnosis writes
//! and that the repair passes. None of this is a measurement: on the real board the hypothesis may be wrong, which
//! is why the diagnosis also names a wrong launch that has nothing to do with stale data.

mod common;

use common::kgr1::{VECTOR_TOL, files, max_abs, read_kgr1, vectors};
use common::put_bytes;
use common::token_sim::TokenSim;
use gpu_compute::layer::{self, Shape};
use gpu_compute::model::{self, Caches};
use gpu_core::diag::Journal;
use gpu_core::error::{GpuError, GpuResult};
use gpu_sim::cmdproc::{Launch as SimLaunch, embed_on_cpu, gelu_on_cpu};
use nvidia_ga104::kronos_diag::{
    self as kd, EMBED, FINAL, Ladder, ProbeCtx, SeqMode, Stale, TokenIo, Verdict,
};
use nvidia_ga104::kronos_gen::{GenPlan, Generation, Stop, TokenCheck, TokenOutput, TokenRunner};
use nvidia_ga104::kronos_layer::{self as kl, Launch, Prog};
use nvidia_ga104::kronos_model::{self as km, OwnedModel};

/// The control model's shape (2 layers, E 32, 4 heads) with a small vocabulary, synthetic weights. Vocabulary 512
/// and context 32 put the activation scratch at a multiple of 128 bytes in the blob, as the control model's is
/// (there: a multiple of 256) — with cache lines of 128 bytes no line holds parts of two vectors.
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

const TOKEN: usize = 65;

/// The CPU engine's logits for the first token and the (empty) caches before it.
fn first_token(m: &OwnedModel) -> (Vec<f32>, Caches) {
    let mut c = Caches::new(m.layers.len(), m.shape.e);
    let cpu = model::step_cpu(&m.weights(), &mut c, TOKEN, 0).unwrap();
    (cpu, Caches::new(m.layers.len(), m.shape.e))
}

fn ladder_of(s: &mut TokenSim<'_>, m: &OwnedModel) -> (Ladder, Journal) {
    let (cpu, prev) = first_token(m);
    let w = m.weights();
    let vas = s.vas.clone();
    let ctx = ProbeCtx {
        w: &w,
        vas: &vas,
        prev: &prev,
        token: TOKEN,
        pos: 0,
        cpu_logits: &cpu,
        want: model::argmax(&cpu),
    };
    let mut j = Journal::new(4096);
    let l = kd::ladder(s, &ctx, &mut j);
    (l, j)
}

fn journal_text(j: &Journal) -> String {
    j.events()
        .enumerate()
        .map(|(i, e)| format!("[  126.{i:06}] gpu: {e}\n"))
        .collect()
}

/// The picture of the board under the hypothesis: one layer per submission passes (I13), a whole token in one
/// submission does not (I14), with finite logits and another greedy token; both repairs pass.
#[test]
fn the_board_picture_under_the_stale_read_hypothesis() {
    let m = small();
    let (cpu, _) = first_token(&m);
    for line in [4u64, 128] {
        // I13's regime: the 16 launches of ONE layer per submission, x written by the driver, 4 positions.
        let mut s = TokenSim::new(&m, line);
        let (lw, lv) = (m.layers[0].weights(), s.vas.layers[0]);
        let e = m.shape.e;
        let x_pa = s.blob.pa + (lv.x - s.blob.va);
        let (mut kc, mut vc) = (
            vec![0.0f32; layer::MAX_POS * e],
            vec![0.0f32; layer::MAX_POS * e],
        );
        for pos in 0..4 {
            let x = kl::test_input(e, 0x13, pos);
            let bytes: Vec<u8> = x.iter().flat_map(|v| v.to_le_bytes()).collect();
            put_bytes(s.r.mem.writer_mut(), x_pa, &bytes);
            let list = kl::token_launches(&m.shape, &lv, pos).unwrap();
            s.submit(&list, 0, SeqMode::Once).unwrap();
            let got: Vec<f32> = s.read(&[(lv.x, e)]).unwrap()[0]
                .iter()
                .map(|&b| f32::from_bits(b))
                .collect();
            let mut want = x.clone();
            layer::step_cpu(&m.shape, &lw, &mut want, &mut kc, &mut vc, pos).unwrap();
            let v = kl::judge(&got, &want);
            assert!(v.passes(), "linie {line}: {}", v.line(pos));
        }
        assert_eq!(
            s.stale_bytes(),
            0,
            "un strat pe trimitere nu reciteste nimic rescris"
        );

        // I14's regime: the whole token in one submission.
        let mut s = TokenSim::new(&m, line);
        let got = s.run_token(TOKEN, 0, SeqMode::Once).unwrap();
        let r = kd::logit_report(TOKEN, 0, &got, &cpu, None);
        assert!(!r.passes, "linie {line}: {}", r.line());
        assert_eq!(r.not_finite, 0, "ca pe placa: toate logit-urile finite");
        assert!(r.board.is_some() && r.outside > m.vocab / 2, "{}", r.line());
        assert!(s.stale_bytes() > 0);
        assert_eq!(
            (s.vas.layers[0].x - s.blob.va) % 128,
            0,
            "spatiul de lucru aliniat la linie"
        );

        // Both repairs: nothing stale is read, the logits are the CPU's.
        for mode in [SeqMode::EachLaunch, SeqMode::PerLayer] {
            let mut s = TokenSim::new(&m, line);
            let got = s.run_token(TOKEN, 0, mode).unwrap();
            let r = kd::logit_report(TOKEN, 0, &got, &cpu, None);
            assert!(r.passes, "linie {line}, {mode:?}: {}", r.line());
            assert_eq!(s.stale_bytes(), 0, "{mode:?}");
        }
    }
    // With coherent memory (the model as it always was) the three modes give the same bits.
    let logits: Vec<Vec<f32>> = [SeqMode::Once, SeqMode::EachLaunch, SeqMode::PerLayer]
        .into_iter()
        .map(|mode| TokenSim::new(&m, 0).run_token(TOKEN, 0, mode).unwrap())
        .collect();
    assert!(kd::logit_report(TOKEN, 0, &logits[0], &cpu, None).passes);
    for l in &logits[1..] {
        assert!(
            l.iter()
                .zip(&logits[0])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
    }
}

/// The ladder under the hypothesis: the control (as os-305) names the first stale read — layer 1's LayerNorm
/// reading x as the embedding left it — the repair passes and is chosen; nothing else is sent.
#[test]
fn the_ladder_confirms_stale_data_and_picks_the_repair() {
    let m = small();
    let mut s = TokenSim::new(&m, 4);
    let (l, j) = ladder_of(&mut s, &m);
    assert_eq!(l.chosen, Some(SeqMode::EachLaunch));
    assert_eq!(l.probes.len(), 2);
    assert_eq!(s.submissions, [35, 35]);
    let p = &l.probes[0];
    assert_eq!(p.mode, Some(SeqMode::Once));
    let first = p.first_bad().expect("o etapa in afara");
    assert!(
        matches!(p.verdict, Verdict::DateVechi(_)),
        "{}",
        p.summary()
    );
    assert_eq!((first.layer, first.launch), (1, 0), "{}", first.line());
    assert_eq!(first.stale, Some(Stale { from: EMBED }));
    assert_eq!(first.local, None, "x a fost rescris de ultima lansare");
    // Layer 0 ran on fresh data: its cache rows are fine.
    assert!(p.stages[..2].iter().all(|s| s.ok && s.layer == 0));
    // Every launch of layer 1 is off; each one whose output does not follow from what is in memory now is what
    // the launch gives with layer 0's values. The others only passed a wrong input on (at position 0 the attention
    // does not even depend on its query).
    let mut proven = 0;
    for s in p.stages.iter().filter(|s| s.layer == 1 && s.launch >= 1) {
        assert!(!s.ok, "{}", s.line());
        match s.local {
            Some((_, true)) => assert_eq!(s.stale, None, "{}", s.line()),
            _ => {
                assert!(s.stale.is_some(), "{}", s.line());
                proven += 1;
            }
        }
    }
    assert!(proven >= 8, "{proven} lansari cu date vechi dovedite");
    let att = p.stages.iter().find(|s| (s.layer, s.launch) == (1, 5));
    assert_eq!(att.unwrap().local, Some((0.0, true)));
    // The final LayerNorm read x as the embedding left it; the logits follow from that h.
    let fin: Vec<_> = p.stages.iter().filter(|s| s.layer == FINAL).collect();
    assert_eq!(fin[0].stale, Some(Stale { from: EMBED }));
    assert_eq!(fin[1].local.map(|l| l.1), Some(true));
    let lg = p.logits.as_ref().unwrap();
    assert!(!lg.passes && lg.not_finite == 0 && lg.fill == 0);
    assert!(l.probes[1].passes(), "{}", l.probes[1].summary());
    assert_eq!(l.probes[1].stages.len(), 20);

    // The journal says the same when read back from the stick.
    let d = kd::parse_journal(&journal_text(&j));
    assert_eq!(d.chosen, Some(2));
    assert_eq!(d.probes.len(), 2);
    assert_eq!((d.probes[0].mode, d.probes[0].verdict), (1, 2));
    assert_eq!((d.probes[1].mode, d.probes[1].verdict), (2, 1));
    assert_eq!(d.probes[0].stages[0].0, 0x100);
    assert_eq!(d.probes[0].stages[0].2 >> 8, EMBED as u32);
    assert_eq!(
        d.probes[0].logits_board,
        lg.first.iter().map(|p| p.0).collect::<Vec<_>>()
    );
    assert_eq!(d.probes[0].argmax_board, lg.board.map(|v| v as u32));
    assert_eq!(d.probes[0].summary, p.summary());
    let text = d.lines().join("\n");
    assert!(text.contains("DATE VECHI"), "{text}");
    assert!(text.contains("modul ales pentru texte: invalidare inaintea fiecarei lansari"));
    // No row of the diagnosis looks like a step line or a register access to the chain's journal checker.
    for line in journal_text(&j).lines() {
        assert_eq!(nvidia_ga104::chain_log::parse_step(line), None, "{line}");
        assert!(
            !matches!(
                nvidia_ga104::i6_log::parse_line(line),
                Some(gpu_core::diag::Event::MmioWrite { .. })
                    | Some(gpu_core::diag::Event::ConfigWrite { .. })
            ),
            "{line}"
        );
    }
    // A healthy board (coherent memory): the control passes too; the repair is still what the texts use.
    let mut s = TokenSim::new(&m, 0);
    let (l, _) = ladder_of(&mut s, &m);
    assert_eq!(l.chosen, Some(SeqMode::EachLaunch));
    assert!(l.probes.iter().all(|p| p.passes()) && l.probes.len() == 2);
}

/// Three layers: after the whole token only the last layer's vectors are in the scratch, but the cache rows of the
/// middle layer are still there — and they are what layer 0's `qkvb` gives: the stale read is named there.
#[test]
fn a_hidden_layer_shows_the_stale_read_in_its_cache_rows() {
    let m = km::test_model(
        Shape {
            e: 32,
            heads: 4,
            eps: 1e-5,
        },
        3,
        512,
        32,
        9,
    )
    .unwrap();
    let mut s = TokenSim::new(&m, 128);
    let (l, _) = ladder_of(&mut s, &m);
    assert_eq!(l.chosen, Some(SeqMode::EachLaunch));
    let p = &l.probes[0];
    let bad = p.first_bad().unwrap();
    assert!(
        matches!(p.verdict, Verdict::DateVechi(_)),
        "{}",
        p.summary()
    );
    assert_eq!((bad.layer, bad.launch), (1, 3), "{}", bad.line());
    assert_eq!(bad.stale, Some(Stale { from: 0 }));
    // Layer 0's own rows are fine; layer 2 is the visible one (16 launches) and the two final launches follow.
    assert!(p.stages[..2].iter().all(|s| s.ok && s.layer == 0));
    assert_eq!(p.stages.len(), 2 + 2 + 16 + 2);
    assert_eq!(s.submissions, [51, 51]);
}

/// A board where invalidating before each launch changes nothing: `EachLaunch` behaves as `Once`.
struct Deaf<'a>(TokenSim<'a>);

impl TokenIo for Deaf<'_> {
    fn submit(&mut self, l: &[Launch], slot: usize, mode: SeqMode) -> GpuResult<()> {
        let mode = if mode == SeqMode::EachLaunch {
            SeqMode::Once
        } else {
            mode
        };
        self.0.submit(l, slot, mode)
    }
    fn read(&mut self, r: &[(u64, usize)]) -> GpuResult<Vec<Vec<u32>>> {
        self.0.read(r)
    }
}

/// If the invalidate method does not help, one submission per layer (I13's regime) is tried and chosen.
#[test]
fn the_ladder_falls_back_to_one_submission_per_layer() {
    let m = small();
    let (cpu, prev) = first_token(&m);
    let w = m.weights();
    let mut io = Deaf(TokenSim::new(&m, 4));
    let vas = io.0.vas.clone();
    let ctx = ProbeCtx {
        w: &w,
        vas: &vas,
        prev: &prev,
        token: TOKEN,
        pos: 0,
        cpu_logits: &cpu,
        want: None,
    };
    let mut j = Journal::new(4096);
    let l = kd::ladder(&mut io, &ctx, &mut j);
    assert_eq!(l.chosen, Some(SeqMode::PerLayer));
    let v: Vec<u32> = l.probes.iter().map(|p| p.verdict.code()).collect();
    assert_eq!(v, [2, 2, 1]);
    assert_eq!(io.0.submissions, [35, 35, 17, 16, 2]);
    assert!(l.lines().last().unwrap().contains("o trimitere pe strat"));
}

/// GELU with one wrong element (a launch that is wrong on its own, nothing stale).
fn gelu_wrong(l: &mut SimLaunch<'_>) -> GpuResult<()> {
    gelu_on_cpu(l)?;
    let y = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let v = f32::from_bits(l.mem.read_u32(y + 12)?);
    l.mem.write_u32(y + 12, (v + 0.5).to_bits());
    Ok(())
}

/// The embedding with one wrong element.
fn embed_wrong(l: &mut SimLaunch<'_>) -> GpuResult<()> {
    embed_on_cpu(l)?;
    let x = l.mem.read_u64(l.cbuf0 + 0x38)?;
    let v = f32::from_bits(l.mem.read_u32(x + 20)?);
    l.mem.write_u32(x + 20, (v + 0.25).to_bits());
    Ok(())
}

fn never(_: &mut SimLaunch<'_>) -> GpuResult<()> {
    Err(GpuError::InvalidArgument("SIM: programul nu se termina"))
}

fn with_model(m: &OwnedModel, prog: Prog, f: gpu_sim::cmdproc::ProgramFn) -> TokenSim<'_> {
    let mut s = TokenSim::new(m, 0);
    let slot = s.models.iter_mut().find(|x| x.0 == prog).unwrap();
    slot.1 = f;
    s
}

/// A launch that is wrong by itself (no mode helps): the prefixes name it — program and layer — and stop there.
#[test]
fn a_wrong_launch_is_named_by_the_prefixes() {
    let m = small();
    let mut s = with_model(&m, Prog::Gelu, gelu_wrong);
    let (l, j) = ladder_of(&mut s, &m);
    assert_eq!(l.chosen, None);
    // Control, repair, per layer (none passes), the embedding alone (clean), layer 0 up to its 15th launch.
    let v: Vec<(u32, usize, u32)> = l
        .probes
        .iter()
        .map(|p| (p.mode.unwrap() as u32, p.done, p.verdict.code()))
        .collect();
    assert_eq!(
        v,
        [(1, 35, 4), (2, 35, 4), (3, 35, 4), (3, 1, 1), (3, 16, 3)]
    );
    // The whole tokens only see that layer 1 started from a wrong x ("inherited").
    let whole = l.probes[0].first_bad().unwrap();
    assert_eq!((whole.layer, whole.launch), (1, 0));
    assert_eq!((whole.local, whole.stale), (None, None));
    // The prefix sees the launch itself: its output does not follow from its input.
    let p = l.probes.last().unwrap();
    let bad = p.first_bad().unwrap();
    assert_eq!((bad.layer, bad.launch, bad.name()), (0, 12, "gelu"));
    assert!(matches!(bad.local, Some((d, false)) if (d - 0.5).abs() < 1e-3));
    assert_eq!(bad.stale, None);
    // Everything before it in that layer is clean, and x still holds the embedding.
    let at = p.verdict.stage().unwrap();
    assert!(p.stages[..at].iter().all(|s| s.ok));
    assert_eq!(p.stages[0].layer, EMBED);
    let text = kd::parse_journal(&journal_text(&j)).lines().join("\n");
    assert!(text.contains("LANSARE GRESITA"), "{text}");
    assert!(text.contains("gelu (stratul 0, lansarea 12)"), "{text}");
    assert!(text.contains("niciun mod nu a trecut"), "{text}");

    // A wrong embedding: the very first prefix (one launch) says so.
    let mut s = with_model(&m, Prog::Embed, embed_wrong);
    let (l, _) = ladder_of(&mut s, &m);
    assert_eq!(l.chosen, None);
    let p = l.probes.last().unwrap();
    assert_eq!((p.done, p.verdict.code()), (1, 3), "{}", p.summary());
    assert_eq!(p.first_bad().unwrap().layer, EMBED);
    assert_eq!(l.probes.len(), 4);
}

/// A submission that never finishes: the diagnosis stops at once, nothing more is sent.
#[test]
fn a_submission_that_does_not_finish_stops_the_ladder() {
    let m = small();
    let mut s = with_model(&m, Prog::Attention, never);
    let (l, j) = ladder_of(&mut s, &m);
    assert_eq!(l.chosen, None);
    assert_eq!(l.probes.len(), 1);
    assert!(matches!(l.probes[0].verdict, Verdict::Neterminat(_)));
    assert!(
        s.submissions.is_empty(),
        "nimic nu s-a terminat, nimic nou trimis"
    );
    let d = kd::parse_journal(&journal_text(&j));
    assert_eq!(
        (d.probes.len(), d.probes[0].verdict, d.chosen),
        (1, 5, Some(0))
    );
}

/// After a token of the step's own run fails: memory as that token left it, judged without sending anything.
#[test]
fn explaining_a_token_that_already_ran() {
    let m = small();
    let (cpu, prev) = first_token(&m);
    let w = m.weights();
    // Stale data, the OS's run (list + copy of the logits down, one submission).
    let mut s = TokenSim::new(&m, 4);
    let ram = s.run_token_ram(TOKEN, 0, SeqMode::Once).unwrap();
    let before = s.submissions.len();
    let vas = s.vas.clone();
    let ctx = ProbeCtx {
        w: &w,
        vas: &vas,
        prev: &prev,
        token: TOKEN,
        pos: 0,
        cpu_logits: &cpu,
        want: model::argmax(&cpu),
    };
    let mut j = Journal::new(1024);
    let p = kd::explain(&mut s, &ctx, &ram, &mut j);
    assert_eq!(s.submissions.len(), before, "nimic trimis");
    assert_eq!(p.mode, None);
    assert!(
        matches!(p.verdict, Verdict::DateVechi(_)),
        "{}",
        p.summary()
    );
    assert_eq!(p.ram_diff, Some(0), "copierea in RAM a adus ce era in VRAM");
    let d = kd::parse_journal(&journal_text(&j));
    assert_eq!((d.probes[0].mode, d.probes[0].verdict), (0, 2));
    assert_eq!(d.probes[0].ram_diff, Some(0));
    assert_eq!(d.probes[0].logits_board.len(), 8);
    assert_eq!(d.probes[0].logits_cpu, cpu[..8]);

    // A healthy token whose logits never reached RAM (the buffer still holds the 0xEE fill).
    let mut s = TokenSim::new(&m, 0);
    s.run_token(TOKEN, 0, SeqMode::EachLaunch).unwrap();
    let fill = vec![f32::from_bits(kd::RAM_FILL); m.vocab];
    let vas = s.vas.clone();
    let ctx = ProbeCtx { vas: &vas, ..ctx };
    let mut j = Journal::new(1024);
    let p = kd::explain(&mut s, &ctx, &fill, &mut j);
    assert_eq!(p.verdict, Verdict::DoarLogiti, "{}", p.summary());
    assert!(p.stages.iter().all(|s| s.ok));
    assert_eq!(p.ram_diff, Some(m.vocab));
    let lg = p.logits.unwrap();
    assert_eq!((lg.fill, lg.board), (m.vocab, Some(0)));
    let text = kd::parse_journal(&journal_text(&j)).lines().join("\n");
    assert!(text.contains("512 ramase cu umplutura"), "{text}");
    assert!(
        text.contains("512 valori diferite de cele din VRAM"),
        "{text}"
    );
}

/// The step's run of a text in a mode: one token = the list + the copy of the logits down, read from RAM.
struct TextRunner<'s, 'm> {
    s: &'s mut TokenSim<'m>,
    mode: SeqMode,
}

impl TokenRunner for TextRunner<'_, '_> {
    fn run_token(&mut self, token: usize, pos: usize) -> GpuResult<TokenOutput> {
        let logits = self.s.run_token_ram(token, pos, self.mode)?;
        let t = 1000 * (self.s.submissions.len() as u64 + 1);
        Ok(TokenOutput {
            logits,
            start_ns: t,
            end_ns: t + 500,
            busy_ns: 400,
        })
    }
}

struct Cpu<'a> {
    w: model::ModelWeights<'a>,
    c: Caches,
}

impl TokenCheck for Cpu<'_> {
    fn cpu_logits(&mut self, token: usize, pos: usize) -> GpuResult<Vec<f32>> {
        model::step_cpu(&self.w, &mut self.c, token, pos)
    }
}

/// A whole text after the ladder, under the hypothesis: in the chosen mode every position passes against the CPU
/// (the caches grow, the K/V rows of earlier positions are read again in later submissions); as os-305 sent it,
/// the first position stops the text.
#[test]
fn a_text_in_the_chosen_mode() {
    let m = small();
    for line in [4u64, 128] {
        let mut s = TokenSim::new(&m, line);
        let (l, _) = ladder_of(&mut s, &m);
        let mode = l.chosen.unwrap();
        let mut check = Cpu {
            w: m.weights(),
            c: Caches::new(2, 32),
        };
        let mut g =
            Generation::new(GenPlan::new(&[TOKEN, 17, 300], 8, m.vocab, m.context).unwrap());
        let stop = g
            .run(&mut TextRunner { s: &mut s, mode }, Some(&mut check))
            .clone();
        assert_eq!(stop, Stop::Gata, "linie {line}: {}", stop.line());
        assert_eq!(g.generated().len(), 8);
        // The same text, one submission per layer.
        let mut s2 = TokenSim::new(&m, line);
        let mut check = Cpu {
            w: m.weights(),
            c: Caches::new(2, 32),
        };
        let mut g2 =
            Generation::new(GenPlan::new(&[TOKEN, 17, 300], 8, m.vocab, m.context).unwrap());
        let stop = g2
            .run(
                &mut TextRunner {
                    s: &mut s2,
                    mode: SeqMode::PerLayer,
                },
                Some(&mut check),
            )
            .clone();
        assert_eq!(stop, Stop::Gata, "{}", stop.line());
        assert_eq!(g2.seq, g.seq);
        // 36 launches per position: [17, 16, 3] per layer.
        assert_eq!(s2.submissions[..3], [17, 16, 3]);
        // As os-305 sent it: stopped at position 0 by the CPU check.
        let mut s3 = TokenSim::new(&m, line);
        let mut check = Cpu {
            w: m.weights(),
            c: Caches::new(2, 32),
        };
        let mut g3 =
            Generation::new(GenPlan::new(&[TOKEN, 17, 300], 8, m.vocab, m.context).unwrap());
        let stop = g3
            .run(
                &mut TextRunner {
                    s: &mut s3,
                    mode: SeqMode::Once,
                },
                Some(&mut check),
            )
            .clone();
        assert!(
            matches!(stop, Stop::NuTrece { pos: 0, .. }),
            "{}",
            stop.line()
        );
    }
}

/// A later position (the caches hold earlier rows): the diagnosis of a whole token and of a prefix agree with the
/// program models bit for bit on a healthy model, in every mode.
#[test]
fn a_later_position_is_clean_on_a_healthy_model() {
    let m = small();
    let w = m.weights();
    let mut s = TokenSim::new(&m, 0);
    let mut c = Caches::new(2, 32);
    let seq = [TOKEN, 17, 300, 9];
    for (pos, &tok) in seq.iter().enumerate() {
        // The caches before this position, then the CPU's logits for it.
        let prev = Caches {
            k: c.k.clone(),
            v: c.v.clone(),
        };
        let cpu = model::step_cpu(&w, &mut c, tok, pos).unwrap();
        let vas = s.vas.clone();
        let ctx = ProbeCtx {
            w: &w,
            vas: &vas,
            prev: &prev,
            token: tok,
            pos,
            cpu_logits: &cpu,
            want: None,
        };
        let mode = [SeqMode::Once, SeqMode::EachLaunch, SeqMode::PerLayer][pos % 3];
        for done in [1, 16, 22, 32, 34, 35] {
            let p = kd::probe(&mut s, &ctx, mode, done);
            assert!(p.passes(), "pozitia {pos}, {done} lansari: {}", p.summary());
            // In SIMULARE the model's memory is the trace, bit for bit — except where the trace used the CPU
            // engine's rows of earlier positions (the attention, from position 1 on).
            let worst = p.stages.iter().fold(0.0f32, |m, s| m.max(s.dev));
            assert!(worst <= 1e-5, "pozitia {pos}, {done}: {worst:e}");
            if pos == 0 {
                assert_eq!(worst, 0.0);
            }
        }
    }
}

// ------------------------------------------------------------------ the KGR1 control vectors (hub, private)

/// The real control model (`mic_q8_0.kgr`, the one `i14-simbol` runs) under the hypothesis: the first token of the
/// first text, sent as in os-305, gives finite logits and another greedy token — the board's message — and the
/// ladder picks the repair; then the three control texts, sent with the repair, give Kronos's vectors exactly.
#[test]
#[ignore = "cere KGR1_VECTORI (vectorii din hub-ul privat, pregatiti local)"]
fn kgr1_control_model_under_the_hypothesis() {
    let f = files();
    let m = read_kgr1(&f.kgr);
    let w = m.weights();
    let vs = vectors(&f, m.vocab);
    let (tok, want) = (vs[0].prompt[0], vs[0].argmax[0]);
    assert_eq!((tok, want), (65, 10337));
    let mut c = Caches::new(m.layers.len(), m.shape.e);
    let prev = Caches::new(m.layers.len(), m.shape.e);
    let cpu = model::step_cpu(&w, &mut c, tok, 0).unwrap();
    let mut s = TokenSim::new(&m, 128);
    let vas = s.vas.clone();
    let ctx = ProbeCtx {
        w: &w,
        vas: &vas,
        prev: &prev,
        token: tok,
        pos: 0,
        cpu_logits: &cpu,
        want: Some(want),
    };
    let mut j = Journal::new(4096);
    let l = kd::ladder(&mut s, &ctx, &mut j);
    assert_eq!(l.chosen, Some(SeqMode::EachLaunch));
    let lg = l.probes[0].logits.as_ref().unwrap();
    eprintln!("SIMULARE (ipoteza): {}", lg.line());
    eprintln!("SIMULARE (ipoteza): primele 8 logit-uri {:?}", lg.first);
    assert_eq!(lg.not_finite, 0);
    assert_eq!(lg.cpu, Some(want));
    assert!(lg.board.is_some() && lg.board != Some(want));
    // How far a rounding difference is from changing this token: not the cause.
    assert!(lg.cpu_margin > 0.28, "{}", lg.cpu_margin);
    eprintln!("SIMULARE: {} randuri de jurnal pentru scara", j.len());

    // The three control texts with the repair, the logits copied down into RAM as the OS does.
    for t in &vs {
        let mut s = TokenSim::new(&m, 128);
        let mut seq = t.prompt.clone();
        let mut last = Vec::new();
        for pos in 0..t.prompt.len() + t.greedy.len() - 1 {
            let got = s.run_token_ram(seq[pos], pos, SeqMode::EachLaunch).unwrap();
            let top = model::argmax(&got).unwrap();
            if pos < t.prompt.len() {
                assert_eq!(top, t.argmax[pos], "{}: pozitia {pos}", t.text);
            }
            if pos + 1 == t.prompt.len() {
                last = got.clone();
            }
            if pos + 1 >= seq.len() {
                seq.push(top);
            }
        }
        assert_eq!(&seq[t.prompt.len()..], &t.greedy[..], "{}", t.text);
        assert!(max_abs(&last, &t.logits) <= VECTOR_TOL, "{}", t.text);
        assert_eq!(s.stale_bytes(), 0);
    }
}

/// The margins of the control vectors on our CPU engine: at every one of the 29 positions the greedy token is far
/// from a tie (the smallest gap is 28 times the logit tolerance), so a rounding difference cannot give "argmax
/// diferit"; the first position of the first text has a gap of 0.28.
#[test]
#[ignore = "cere KGR1_VECTORI (vectorii din hub-ul privat, pregatiti local)"]
fn kgr1_margins_rule_out_rounding() {
    let f = files();
    let m = read_kgr1(&f.kgr);
    let w = m.weights();
    let mut smallest = f32::INFINITY;
    let mut n = 0;
    for (i, t) in vectors(&f, m.vocab).iter().enumerate() {
        let mut c = Caches::new(m.layers.len(), m.shape.e);
        let mut seq = t.prompt.clone();
        for pos in 0..t.prompt.len() + t.greedy.len() - 1 {
            let l = model::step_cpu(&w, &mut c, seq[pos], pos).unwrap();
            let r = kd::logit_report(seq[pos], pos, &l, &l, None);
            eprintln!(
                "{}: pozitia {pos}: simbolul {} -> {}, marja {:.4}",
                t.text,
                seq[pos],
                r.cpu.unwrap(),
                r.cpu_margin
            );
            if (i, pos) == (0, 0) {
                assert!((r.cpu_margin - 0.2849).abs() < 1e-3);
            }
            smallest = smallest.min(r.cpu_margin);
            n += 1;
            if pos + 1 >= seq.len() {
                seq.push(r.cpu.unwrap());
            }
        }
    }
    assert_eq!(n, 29);
    assert!(smallest > 28.0 * model::LOGIT_ABS, "{smallest}");
    // Two logits moved against each other by the whole tolerance still do not reach it.
    assert!(smallest > 2.0 * (model::LOGIT_ABS + model::LOGIT_REL * 8.4));
}
