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
use gpu_core::error::{GpuError, GpuResult};
use gpu_sim::cmdproc::{
    Launch as SimLaunch, attention_on_cpu, copy_b128_on_cpu, embed_on_cpu, gelu_on_cpu, vadd_on_cpu,
};
use nvidia_ga104::kronos_diag::{
    self as kd, EMBED, EmbedBy, FINAL, Ladder, ProbeCtx, SeqMode, Stale, TokenIo, Verdict, Way,
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

/// The ladder on the first token, and the rows it wrote, as the OS prints them on the stick.
fn ladder_of(s: &mut TokenSim<'_>, m: &OwnedModel) -> (Ladder, String) {
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
    let l = kd::ladder(s, &ctx);
    (l, s.take_journal_text())
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
    assert_eq!(l.chosen, Some(Way::REPAIR));
    assert_eq!(l.probes.len(), 2);
    assert_eq!(s.submissions, [35, 35]);
    let p = &l.probes[0];
    assert_eq!(p.way, Some(Way::OS_305));
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
    let d = kd::parse_journal(&j);
    assert_eq!((d.chosen, d.chosen_embed), (Some(2), Some(0)));
    assert_eq!(d.probes.len(), 2);
    assert_eq!((d.probes[0].mode, d.probes[0].verdict), (1, 2));
    assert_eq!((d.probes[1].mode, d.probes[1].verdict), (2, 1));
    assert!(d.probes.iter().all(|p| p.embed_by == 0));
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
    assert!(!text.contains("scris de driver"), "{text}");
    // No row of the diagnosis looks like a step line or a register access to the chain's journal checker.
    for line in j.lines() {
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
    assert_eq!(l.chosen, Some(Way::REPAIR));
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
    assert_eq!(l.chosen, Some(Way::REPAIR));
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
    fn write(&mut self, va: u64, words: &[u32]) -> GpuResult<()> {
        self.0.write(va, words)
    }
    fn journal(&mut self) -> &mut gpu_core::diag::Journal {
        self.0.journal()
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
    let l = kd::ladder(&mut io, &ctx);
    assert_eq!(l.chosen, Some(Way::of(SeqMode::PerLayer)));
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

/// `(mode, embedding by the driver, launches, verdict code)` of every probe.
fn shape_of(l: &Ladder) -> Vec<(u32, u32, usize, u32)> {
    l.probes
        .iter()
        .map(|p| {
            let w = p.way.unwrap();
            (w.mode as u32, w.embed as u32, p.done, p.verdict.code())
        })
        .collect()
}

/// A launch that is wrong by itself (no mode helps): the prefixes name it — program and layer — and stop there.
#[test]
fn a_wrong_launch_is_named_by_the_prefixes() {
    let m = small();
    let mut s = with_model(&m, Prog::Gelu, gelu_wrong);
    let (l, j) = ladder_of(&mut s, &m);
    assert_eq!(l.chosen, None);
    // Control, repair, per layer (none passes), the embedding alone (clean), layer 0 up to its 15th launch.
    assert_eq!(
        shape_of(&l),
        [
            (1, 0, 35, 3),
            (2, 0, 35, 3),
            (3, 0, 35, 3),
            (3, 0, 1, 1),
            (3, 0, 16, 3)
        ]
    );
    // The whole tokens: layer 1 started from a wrong x (LayerNorm 1 only passed it on — its input is no longer in
    // memory), but the same program is wrong in layer 1 too, and there its input IS in memory: named at once.
    let whole = &l.probes[0];
    let first = whole.stages.iter().find(|s| !s.ok).unwrap();
    assert_eq!((first.layer, first.launch), (1, 0));
    assert_eq!((first.local, first.stale), (None, None));
    let named = whole.first_bad().unwrap();
    assert_eq!(
        (named.layer, named.launch, named.name()),
        (1, 12, "gelu"),
        "{}",
        whole.summary()
    );
    assert!(matches!(named.local, Some((d, false)) if (d - 0.5).abs() < 1e-3));
    // The prefix sees the first wrong launch itself: its output does not follow from its input.
    let p = l.probes.last().unwrap();
    let bad = p.first_bad().unwrap();
    assert_eq!((bad.layer, bad.launch, bad.name()), (0, 12, "gelu"));
    assert!(matches!(bad.local, Some((d, false)) if (d - 0.5).abs() < 1e-3));
    assert_eq!(bad.stale, None);
    // Everything before it in that layer is clean, and x still holds the embedding.
    let at = p.verdict.stage().unwrap();
    assert!(p.stages[..at].iter().all(|s| s.ok));
    assert_eq!(p.stages[0].layer, EMBED);
    let d = kd::parse_journal(&j);
    assert_eq!((d.chosen, d.chosen_embed), (Some(0), Some(0)));
    let text = d.lines().join("\n");
    assert!(text.contains("LANSARE GRESITA"), "{text}");
    assert!(text.contains("gelu (stratul 0, lansarea 12)"), "{text}");
    assert!(text.contains("niciun mod nu a trecut"), "{text}");

    // A launch wrong only in the FIRST layer (`fb = f + bfc` there): after a whole token its vectors are gone
    // (layer 1 reused the scratch) and layer 1's launches are all right given their inputs — "inherited"; the
    // prefix of layer 0 names it.
    let mut s = with_model(&m, Prog::Vadd, vadd_wrong_at_the_marked_bias);
    WRONG_BIAS.with(|b| b.set(s.vas.layers[0].bfc));
    let (l, j) = ladder_of(&mut s, &m);
    assert_eq!(l.chosen, None);
    assert_eq!(
        shape_of(&l),
        [
            (1, 0, 35, 4),
            (2, 0, 35, 4),
            (3, 0, 35, 4),
            (3, 0, 1, 1),
            (3, 0, 16, 3)
        ]
    );
    let whole = l.probes[0].first_bad().unwrap();
    assert_eq!((whole.layer, whole.launch), (1, 0));
    assert_eq!((whole.local, whole.stale), (None, None));
    let bad = l.probes.last().unwrap().first_bad().unwrap();
    assert_eq!((bad.layer, bad.launch, bad.name()), (0, 11, "f + bfc"));
    let text = kd::parse_journal(&j).lines().join("\n");
    assert!(text.contains("EROARE MOSTENITA"), "{text}");
    assert!(text.contains("f + bfc (stratul 0, lansarea 11)"), "{text}");

    // The same launch wrong only in the SECOND layer: the whole tokens name it (the last layer's vectors are all in
    // memory); layer 0's prefix is clean, layer 1's names it again.
    let mut s = with_model(&m, Prog::Vadd, vadd_wrong_at_the_marked_bias);
    WRONG_BIAS.with(|b| b.set(s.vas.layers[1].bfc));
    let (l, _) = ladder_of(&mut s, &m);
    assert_eq!(l.chosen, None);
    assert_eq!(
        shape_of(&l),
        [
            (1, 0, 35, 3),
            (2, 0, 35, 3),
            (3, 0, 35, 3),
            (3, 0, 1, 1),
            (3, 0, 16, 1),
            (3, 0, 32, 3)
        ]
    );
    let whole = l.probes[0].first_bad().unwrap();
    assert_eq!((whole.layer, whole.launch), (1, 11));
    let p = l.probes.last().unwrap();
    let bad = p.first_bad().unwrap();
    assert_eq!((bad.layer, bad.launch, bad.name()), (1, 11, "f + bfc"));
    assert!(matches!(bad.local, Some((d, false)) if (d - 0.5).abs() < 1e-3));
    // Before it: layer 0's two cache rows and its output (still in x), then layer 1's launches — all clean.
    let at = p.verdict.stage().unwrap();
    assert!(p.stages[..at].iter().all(|s| s.ok));
    assert_eq!(
        p.stages[..3]
            .iter()
            .map(|s| (s.layer, s.launch))
            .collect::<Vec<_>>(),
        [(0, 3), (0, 4), (0, 15)]
    );
}

thread_local! {
    /// The bias address at which `vadd_wrong_at_the_marked_bias` goes wrong (a plain `fn` cannot capture it; every
    /// test runs in its own thread).
    static WRONG_BIAS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The vector addition, wrong in one element only where its second input is the marked bias: one launch of one
/// layer, however the token is sent.
fn vadd_wrong_at_the_marked_bias(l: &mut SimLaunch<'_>) -> GpuResult<()> {
    vadd_on_cpu(l)?;
    if l.mem.read_u64(l.cbuf0 + 0x30)? == WRONG_BIAS.with(|b| b.get()) {
        let y = l.mem.read_u64(l.cbuf0 + 0x38)?;
        let v = f32::from_bits(l.mem.read_u32(y + 8)?);
        l.mem.write_u32(y + 8, (v + 0.5).to_bits());
    }
    Ok(())
}

/// The program `embed` wrong on the board (the one program of a token that I13 never ran there): the embedding
/// alone says so, the token passes with the embedding written by the driver, and the texts run that way.
#[test]
fn a_wrong_embedding_is_replaced_by_the_drivers() {
    let m = small();
    let mut s = with_model(&m, Prog::Embed, embed_wrong);
    let (l, j) = ladder_of(&mut s, &m);
    let by_driver = Way::REPAIR.with_driver_embedding();
    assert_eq!(l.chosen, Some(by_driver));
    assert_eq!(
        shape_of(&l),
        [
            (1, 0, 35, 4),
            (2, 0, 35, 4),
            (3, 0, 35, 4),
            (3, 0, 1, 3),
            (2, 1, 35, 1)
        ]
    );
    // The last submission is the token without its launch 0.
    assert_eq!(s.submissions, [35, 35, 17, 16, 2, 1, 34]);
    let p = &l.probes[3];
    let bad = p.first_bad().unwrap();
    assert_eq!(bad.layer, EMBED);
    assert!(matches!(bad.local, Some((d, false)) if (d - 0.25).abs() < 1e-3));
    assert!(
        l.lines()
            .last()
            .unwrap()
            .contains("embedding scris de driver")
    );
    let d = kd::parse_journal(&j);
    assert_eq!((d.chosen, d.chosen_embed), (Some(2), Some(1)));
    assert_eq!(d.probes[4].embed_by, 1);
    let text = d.lines().join("\n");
    assert!(
        text.contains("embedding (simbol + pozitie) (inainte de straturi)"),
        "{text}"
    );
    assert!(
        text.contains("embedding scris de driver (programul embed e gresit pe placa)"),
        "{text}"
    );

    // A text that way: every position passes against the CPU; with the program's embedding the first one stops it.
    for (way, passes) in [(by_driver, true), (Way::REPAIR, false)] {
        let mut check = Cpu {
            w: m.weights(),
            c: Caches::new(2, 32),
        };
        let mut g =
            Generation::new(GenPlan::new(&[TOKEN, 17, 300], 8, m.vocab, m.context).unwrap());
        let stop = g
            .run(&mut TextRunner { s: &mut s, way }, Some(&mut check))
            .clone();
        if passes {
            assert_eq!(stop, Stop::Gata, "{}", stop.line());
            assert_eq!(g.generated().len(), 8);
        } else {
            assert!(
                matches!(stop, Stop::NuTrece { pos: 0, .. }),
                "{}",
                stop.line()
            );
        }
    }

    // The embedding wrong AND a launch of a layer wrong: the driver's embedding does not save the token, and the
    // prefixes (with the driver's embedding in place) name the other launch.
    let mut s = with_model(&m, Prog::Embed, embed_wrong);
    s.models.iter_mut().find(|x| x.0 == Prog::Gelu).unwrap().1 = gelu_wrong;
    let (l, _) = ladder_of(&mut s, &m);
    assert_eq!(l.chosen, None);
    assert_eq!(
        shape_of(&l)[3..],
        [(3, 0, 1, 3), (2, 1, 35, 3), (3, 1, 35, 3), (3, 1, 16, 3)]
    );
    let p = l.probes.last().unwrap();
    assert_eq!(p.way.unwrap().embed, EmbedBy::Driver);
    let bad = p.first_bad().unwrap();
    assert_eq!((bad.layer, bad.launch, bad.name()), (0, 12, "gelu"));
    assert!(p.stages[0].layer == EMBED && p.stages[0].ok);
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
    let d = kd::parse_journal(&j);
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
    let ex = kd::explain(&mut s, &ctx, &ram);
    let j = s.take_journal_text();
    assert_eq!(s.submissions.len(), before, "nimic trimis");
    assert!(!ex.vram_pass, "logit-urile din VRAM sunt la fel de gresite");
    assert_eq!(ex.vram_logits.as_deref(), Some(&ram[..]));
    let p = ex.probe;
    assert_eq!(p.way, None);
    assert!(
        matches!(p.verdict, Verdict::DateVechi(_)),
        "{}",
        p.summary()
    );
    assert_eq!(p.ram_diff, Some(0), "copierea in RAM a adus ce era in VRAM");
    let d = kd::parse_journal(&j);
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
    let ex = kd::explain(&mut s, &ctx, &fill);
    let j = s.take_journal_text();
    // Only the copy is wrong: the logits in VRAM pass, and the step may go on with them.
    assert!(ex.vram_pass);
    let vram = ex.vram_logits.unwrap();
    assert!(kd::logit_report(TOKEN, 0, &vram, &cpu, ctx.want).passes);
    let p = ex.probe;
    assert_eq!(p.verdict, Verdict::DoarLogiti, "{}", p.summary());
    assert!(p.stages.iter().all(|s| s.ok));
    assert_eq!(p.ram_diff, Some(m.vocab));
    let lg = p.logits.unwrap();
    assert_eq!((lg.fill, lg.board), (m.vocab, Some(0)));
    let text = kd::parse_journal(&j).lines().join("\n");
    assert!(text.contains("512 ramase cu umplutura"), "{text}");
    assert!(
        text.contains("512 valori diferite de cele din VRAM"),
        "{text}"
    );
}

/// The step's run of a text in a way: one token = the list + the copy of the logits down, read from RAM.
struct TextRunner<'s, 'm> {
    s: &'s mut TokenSim<'m>,
    way: Way,
}

impl TokenRunner for TextRunner<'_, '_> {
    fn run_token(&mut self, token: usize, pos: usize) -> GpuResult<TokenOutput> {
        let logits = self.s.run_token_ram(token, pos, self.way)?;
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
        let way = l.chosen.unwrap();
        let mut check = Cpu {
            w: m.weights(),
            c: Caches::new(2, 32),
        };
        let mut g =
            Generation::new(GenPlan::new(&[TOKEN, 17, 300], 8, m.vocab, m.context).unwrap());
        let stop = g
            .run(&mut TextRunner { s: &mut s, way }, Some(&mut check))
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
                    way: Way::of(SeqMode::PerLayer),
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
                    way: Way::OS_305,
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

// ------------------------------------------------------------------ the step, as the note for the OS describes it

/// Where the step reads a token's logits from.
#[derive(Clone, Copy, Debug, PartialEq)]
enum LogitsFrom {
    /// The OS's RAM buffer, after the copy down (one more launch).
    Ram,
    /// VRAM, through the window (no copy launch).
    Vram,
}

/// How `i14-simbol` ends (`integration/cyber-universe/C-139_i14-diagnostic-si-reparatie.md`).
#[derive(Debug, PartialEq)]
enum StepEnd {
    /// Every text passed with every launch on the board, sent in this mode: the step is `ok`.
    Trece(SeqMode),
    /// The texts pass only AROUND a launch that is wrong on the board — the program `embed` (the driver wrote the
    /// embedding) and / or the copy of the logits into RAM (they were read from VRAM): the step reports ESEC with
    /// that reason; the texts were run to the end only for what they tell.
    Ocolit {
        mode: SeqMode,
        embed_gresit: bool,
        copiere_gresita: bool,
    },
    /// Not passing; what and where is in the journal.
    Esec,
}

/// One position for `Generation`: the token in the chosen way, the logits from RAM or from VRAM.
struct StepRunner<'s, 'm> {
    s: &'s mut TokenSim<'m>,
    way: Way,
    from: LogitsFrom,
}

impl TokenRunner for StepRunner<'_, '_> {
    fn run_token(&mut self, token: usize, pos: usize) -> GpuResult<TokenOutput> {
        let logits = match self.from {
            LogitsFrom::Ram => self.s.run_token_ram(token, pos, self.way)?,
            LogitsFrom::Vram => self.s.run_token(token, pos, self.way)?,
        };
        let t = 1000 * (self.s.submissions.len() as u64 + 1);
        Ok(TokenOutput {
            logits,
            start_ns: t,
            end_ns: t + 500,
            busy_ns: 400,
        })
    }
}

/// The whole step as the note describes it: (1) the ladder on the first token of the first text; (2) the texts in
/// the chosen way, every position judged against the CPU engine; (3) at the first position that does not pass,
/// memory as that token left it is explained in the journal — if only the copy into RAM is wrong the text starts
/// again with the logits read from VRAM, if the error was inherited the launch is located.
fn step(s: &mut TokenSim<'_>, m: &OwnedModel, texts: &[&[usize]]) -> StepEnd {
    let w = m.weights();
    let vas = s.vas.clone();
    let (layers, e) = (m.layers.len(), m.shape.e);
    // 1. The ladder.
    let tok = texts[0][0];
    let cpu = model::step_cpu(&w, &mut Caches::new(layers, e), tok, 0).unwrap();
    let prev = Caches::new(layers, e);
    let ctx = ProbeCtx {
        w: &w,
        vas: &vas,
        prev: &prev,
        token: tok,
        pos: 0,
        cpu_logits: &cpu,
        want: model::argmax(&cpu),
    };
    let Some(way) = kd::ladder(s, &ctx).chosen else {
        return StepEnd::Esec;
    };
    // 2. The texts.
    let mut from = LogitsFrom::Ram;
    for t in texts {
        loop {
            let mut check = Cpu {
                w: m.weights(),
                c: Caches::new(layers, e),
            };
            let mut g = Generation::new(GenPlan::new(t, 3, m.vocab, m.context).unwrap());
            let stop = g
                .run(&mut StepRunner { s, way, from }, Some(&mut check))
                .clone();
            let pos = match stop {
                Stop::Gata => break,
                Stop::NuTrece { pos, .. } | Stop::LogitiNefiniti { pos } => pos,
                _ => return StepEnd::Esec,
            };
            // 3. What that token left in memory. The CPU engine's caches hold the rows of the earlier positions
            //    (the row of `pos` is rewritten with the same values by asking again).
            let tok = g.seq[pos];
            let cpu = check.cpu_logits(tok, pos).unwrap();
            let ctx = ProbeCtx {
                w: &w,
                vas: &vas,
                prev: &check.c,
                token: tok,
                pos,
                cpu_logits: &cpu,
                want: None,
            };
            let read = match from {
                LogitsFrom::Ram => s.ram_logits(),
                LogitsFrom::Vram => {
                    let got = s.read(&[(vas.logits, m.vocab)]).unwrap();
                    got[0].iter().map(|&b| f32::from_bits(b)).collect()
                }
            };
            let ex = kd::explain(s, &ctx, &read);
            if ex.vram_pass && from == LogitsFrom::Ram {
                from = LogitsFrom::Vram;
                continue;
            }
            if matches!(ex.probe.verdict, Verdict::Mostenit(_)) {
                kd::locate(s, &ctx, way.embed);
            }
            return StepEnd::Esec;
        }
    }
    let (embed_gresit, copiere_gresita) = (way.embed == EmbedBy::Driver, from == LogitsFrom::Vram);
    if embed_gresit || copiere_gresita {
        StepEnd::Ocolit {
            mode: way.mode,
            embed_gresit,
            copiere_gresita,
        }
    } else {
        StepEnd::Trece(way.mode)
    }
}

thread_local! {
    /// The VA the marked wrong programs below look for (a cache of one layer, or the RAM buffer of the logits).
    static MARK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The copy program, except that nothing reaches the marked destination (the OS's RAM buffer).
fn copy_not_into_the_marked_buffer(l: &mut SimLaunch<'_>) -> GpuResult<()> {
    if l.mem.read_u64(l.cbuf0 + 0x30)? == MARK.with(|m| m.get()) {
        return Ok(());
    }
    copy_b128_on_cpu(l)
}

/// The attention, wrong in one element from the second position on, where its key cache is the marked one (one
/// layer): the first token of a text passes, a later one does not.
fn attention_wrong_later_at_the_marked_cache(l: &mut SimLaunch<'_>) -> GpuResult<()> {
    attention_on_cpu(l)?;
    let later = l.mem.read_u32(l.cbuf0 + 0x48)? > 1;
    if later && l.mem.read_u64(l.cbuf0 + 0x30)? == MARK.with(|m| m.get()) {
        let o = l.mem.read_u64(l.cbuf0 + 0x40)?;
        let v = f32::from_bits(l.mem.read_u32(o + 16)?);
        l.mem.write_u32(o + 16, (v + 0.5).to_bits());
    }
    Ok(())
}

/// The step from its first submission to its verdict, on a healthy model, under the hypothesis and with each kind
/// of fault the diagnosis tells apart.
#[test]
fn the_step_as_the_note_describes_it() {
    let m = small();
    let texts: [&[usize]; 2] = [&[TOKEN, 17, 300], &[9, 400]];
    // 2 texts: 3 + 2 prompt positions and 3 generated tokens each = (3 + 2) + (2 + 2) positions.
    let positions = 9;

    // Healthy memory and under the hypothesis: the repair is chosen, every text passes, nothing is explained.
    for line in [0u64, 4, 128] {
        let mut s = TokenSim::new(&m, line);
        assert_eq!(
            step(&mut s, &m, &texts),
            StepEnd::Trece(SeqMode::EachLaunch),
            "linie {line}"
        );
        // The ladder (control + repair, 35 launches each), then 36 launches per position.
        assert_eq!(s.submissions.len(), 2 + positions, "linie {line}");
        assert!(s.submissions[2..].iter().all(|&n| n == 36));
        let d = kd::parse_journal(&s.take_journal_text());
        assert_eq!(d.probes.len(), 2, "doar scara");
        assert_eq!(d.probes[0].verdict, if line == 0 { 1 } else { 2 });
    }

    // The copy into RAM never arrives: the ladder passes (it reads VRAM), the first position read from RAM does
    // not; the explanation tells the two apart and the texts go on with the logits from VRAM.
    let mut s = with_model(&m, Prog::Copy, copy_not_into_the_marked_buffer);
    MARK.with(|k| k.set(s.host.va));
    assert_eq!(
        step(&mut s, &m, &texts),
        StepEnd::Ocolit {
            mode: SeqMode::EachLaunch,
            embed_gresit: false,
            copiere_gresita: true
        }
    );
    // Ladder, the one position read from RAM, then every position again without the copy launch.
    assert_eq!(s.submissions.len(), 2 + 1 + positions);
    assert_eq!(s.submissions[2], 36);
    assert!(s.submissions[3..].iter().all(|&n| n == 35));
    let d = kd::parse_journal(&s.take_journal_text());
    assert_eq!(d.probes.len(), 3);
    let ex = &d.probes[2];
    assert_eq!((ex.mode, ex.verdict), (0, 6));
    assert_eq!((ex.ram_diff, ex.fill), (Some(512), Some(512)));
    assert_eq!(ex.argmax_board, Some(0), "umplutura: argmax 0");
    let text = d.lines().join("\n");
    assert!(
        text.contains("etapele sunt bune, dar logit-urile citite (din RAM) nu trec"),
        "{text}"
    );

    // The program `embed` wrong: the texts run with the driver's embedding (and the step says why it is not ok).
    let mut s = with_model(&m, Prog::Embed, embed_wrong);
    assert_eq!(
        step(&mut s, &m, &texts),
        StepEnd::Ocolit {
            mode: SeqMode::EachLaunch,
            embed_gresit: true,
            copiere_gresita: false
        }
    );
    assert!(s.submissions[7..].iter().all(|&n| n == 35), "36 fara embed");

    // A launch wrong at every position: the ladder names it and no text is sent.
    let mut s = with_model(&m, Prog::Gelu, gelu_wrong);
    assert_eq!(step(&mut s, &m, &texts), StepEnd::Esec);
    assert_eq!(s.submissions, [35, 35, 17, 16, 2, 1, 16]);

    // A launch wrong only from the second position on, in the last layer: the ladder passes, the text stops at
    // position 1 and the explanation names the launch (its inputs are still in memory).
    let mut s = with_model(
        &m,
        Prog::Attention,
        attention_wrong_later_at_the_marked_cache,
    );
    MARK.with(|k| k.set(s.vas.layers[1].kc));
    assert_eq!(step(&mut s, &m, &texts), StepEnd::Esec);
    assert_eq!(s.submissions, [35, 35, 36, 36], "nimic trimis dupa esec");
    let d = kd::parse_journal(&s.take_journal_text());
    let ex = d.probes.last().unwrap();
    assert_eq!((ex.mode, ex.verdict, ex.pos), (0, 3, Some(1)));
    let text = d.lines().join("\n");
    assert!(text.contains("atentie (stratul 1, lansarea 5)"), "{text}");
    assert!(text.contains("NU urmeaza din ele"), "{text}");

    // The same in a layer whose vectors are gone after the whole token (3 layers, the middle one): the explanation
    // can only say "inherited"; the prefixes at that position name the launch.
    let m3 = km::test_model(m.shape, 3, 512, 32, 9).unwrap();
    let mut s = with_model(
        &m3,
        Prog::Attention,
        attention_wrong_later_at_the_marked_cache,
    );
    MARK.with(|k| k.set(s.vas.layers[1].kc));
    assert_eq!(step(&mut s, &m3, &texts), StepEnd::Esec);
    // Ladder (51 + 51), two positions (52 each), then the prefixes: the embedding, layer 0, layers 0 + 1.
    assert_eq!(s.submissions, [51, 51, 52, 52, 1, 16, 17, 15]);
    let d = kd::parse_journal(&s.take_journal_text());
    let v: Vec<(u32, u32, u32)> = d.probes[2..]
        .iter()
        .map(|p| (p.mode, p.launches, p.verdict))
        .collect();
    assert_eq!(v, [(0, 51, 4), (3, 1, 1), (3, 16, 1), (3, 32, 3)]);
    assert!(d.probes[3..].iter().all(|p| p.pos == Some(1)));
    let text = d.lines().join("\n");
    assert!(text.contains("EROARE MOSTENITA"), "{text}");
    assert!(text.contains("atentie (stratul 1, lansarea 5)"), "{text}");
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
        // Odd positions: the embedding written by the driver (launch 0 not sent) — the same bits.
        let way = if pos % 2 == 1 {
            Way::of(mode).with_driver_embedding()
        } else {
            Way::of(mode)
        };
        if way.embed == EmbedBy::Driver {
            // The driver's embedding alone is not a probe: nothing would be sent after the write.
            let p = kd::probe(&mut s, &ctx, way, 1);
            assert!(matches!(p.verdict, Verdict::Neterminat(_)));
        }
        for done in [1, 16, 22, 32, 34, 35] {
            if done == 1 && way.embed == EmbedBy::Driver {
                continue;
            }
            let before = s.submissions.len();
            let p = kd::probe(&mut s, &ctx, way, done);
            assert!(p.passes(), "pozitia {pos}, {done} lansari: {}", p.summary());
            let sent: usize = s.submissions[before..].iter().sum();
            assert_eq!(sent, done - (pos % 2), "pozitia {pos}, {done} lansari");
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
    let l = kd::ladder(&mut s, &ctx);
    let j = s.take_journal_text();
    assert_eq!(l.chosen, Some(Way::REPAIR));
    let lg = l.probes[0].logits.as_ref().unwrap();
    eprintln!("SIMULARE (ipoteza): {}", lg.line());
    eprintln!("SIMULARE (ipoteza): primele 8 logit-uri {:?}", lg.first);
    assert_eq!(lg.not_finite, 0);
    assert_eq!(lg.cpu, Some(want));
    assert!(lg.board.is_some() && lg.board != Some(want));
    // How far a rounding difference is from changing this token: not the cause.
    assert!(lg.cpu_margin > 0.28, "{}", lg.cpu_margin);
    eprintln!(
        "SIMULARE: {} randuri de jurnal pentru scara",
        j.lines().count()
    );

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

    // The fallback mode on the real model, under the hypothesis: one submission per layer gives the CPU's logits.
    let mut s = TokenSim::new(&m, 128);
    let got = s.run_token(tok, 0, SeqMode::PerLayer).unwrap();
    let r = kd::logit_report(tok, 0, &got, &cpu, Some(want));
    assert!(r.passes, "{}", r.line());
    assert_eq!(
        (s.submissions.clone(), s.stale_bytes()),
        (vec![17, 16, 2], 0)
    );

    // The first text again with the embedding written by the driver (the fallback if the program `embed` is the
    // wrong one on the board): the same greedy tokens at every position, launch 0 never sent.
    let t = &vs[0];
    let mut s = TokenSim::new(&m, 128);
    let way = Way::REPAIR.with_driver_embedding();
    let mut seq = t.prompt.clone();
    for pos in 0..t.prompt.len() + t.greedy.len() - 1 {
        let got = s.run_token_ram(seq[pos], pos, way).unwrap();
        let top = model::argmax(&got).unwrap();
        if pos < t.prompt.len() {
            assert_eq!(top, t.argmax[pos], "{}: pozitia {pos}", t.text);
        }
        if pos + 1 >= seq.len() {
            seq.push(top);
        }
    }
    assert_eq!(&seq[t.prompt.len()..], &t.greedy[..], "{}", t.text);
    assert!(s.submissions.iter().all(|&n| n == 35), "35 = 36 fara embed");
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
