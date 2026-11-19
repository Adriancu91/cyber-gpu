//! C-145, SIMULARE: the three ways to shorten the driver's pauses change nothing the GPU computes.
//! (1) the launch slots written through `kronos_fast::WriteShadow` (only the changed words) give the same tokens and
//! the same logits, bit for bit, as writing every slot whole, with far fewer bytes; (2) the CPU check after the board
//! (`Generation::run_check_after`) gives the same tokens and the same verdict as the check between tokens, and stops
//! at the same position when the GPU is made wrong; (3) the stick journal written with the routine rows held gives
//! the same rows in the same order as the journal flushed row by row. The "GPU" is `gpu_sim::cmdproc` with the
//! program models; nothing here says anything about the RTX 3070 Ti or its speed.

mod common;

use std::sync::Mutex;

use common::token_sim::TokenSim;
use gpu_compute::layer::Shape;
use gpu_compute::model::{self as cm, Caches, ModelWeights};
use gpu_core::diag::{Event, Flush, Journal, LiveSink};
use gpu_core::error::GpuResult;
use nvidia_ga104::kronos_diag::{EmbedBy, SeqMode, Way};
use nvidia_ga104::kronos_fast::{self as kf, WriteShadow};
use nvidia_ga104::kronos_gen::{GenPlan, Generation, Stop, TokenCheck, TokenOutput, TokenRunner};
use nvidia_ga104::kronos_model::{self as km, OwnedModel};

const WAY: Way = Way {
    mode: SeqMode::EachLaunch,
    embed: EmbedBy::Program,
};

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

struct Runner<'s, 'a> {
    s: &'s mut TokenSim<'a>,
    logits: Vec<Vec<f32>>,
}

impl TokenRunner for Runner<'_, '_> {
    fn run_token(&mut self, token: usize, pos: usize) -> GpuResult<TokenOutput> {
        let before = self.s.times.len();
        let logits = self.s.run_token_ram(token, pos, WAY)?;
        self.logits.push(logits.clone());
        let mine = &self.s.times[before..];
        Ok(TokenOutput {
            logits,
            start_ns: mine.first().map_or(0, |t| t.0),
            end_ns: mine.last().map_or(0, |t| t.1),
            busy_ns: mine.iter().map(|t| t.2).sum(),
        })
    }
}

struct Cpu<'a> {
    w: ModelWeights<'a>,
    caches: Caches,
    /// When set, the CPU's logits at this position get a different maximum (a "wrong board" seen from the check).
    spoil: Option<usize>,
    calls: Vec<usize>,
}

impl TokenCheck for Cpu<'_> {
    fn cpu_logits(&mut self, token: usize, pos: usize) -> GpuResult<Vec<f32>> {
        self.calls.push(pos);
        let mut l = cm::step_cpu(&self.w, &mut self.caches, token, pos)?;
        if self.spoil == Some(pos) {
            let top = cm::argmax(&l).unwrap();
            let other = (top + 1) % l.len();
            l[other] = l[top] + 10.0;
        }
        Ok(l)
    }
}

fn cpu(m: &OwnedModel, spoil: Option<usize>) -> Cpu<'_> {
    Cpu {
        w: m.weights(),
        caches: Caches::new(m.layers.len(), m.shape.e),
        spoil,
        calls: Vec::new(),
    }
}

/// One generation of `n` tokens after the prompt; `shadow` on/off, check between tokens or after.
fn generate(
    m: &OwnedModel,
    n: usize,
    shadow: bool,
    after: bool,
    spoil: Option<usize>,
) -> (Generation, Vec<Vec<f32>>, u64, Vec<usize>) {
    let mut s = TokenSim::new(m, 0);
    if shadow {
        s.shadow = Some(WriteShadow::new(s.launches.pa, s.launches.size).unwrap());
    }
    let plan = GenPlan::new(&[65, 198, 185], n, m.vocab, m.context).unwrap();
    let mut g = Generation::new(plan);
    let mut c = cpu(m, spoil);
    let mut r = Runner {
        s: &mut s,
        logits: Vec::new(),
    };
    if after {
        g.run_check_after(&mut r, &mut c);
    } else {
        g.run(&mut r, Some(&mut c));
    }
    let logits = r.logits;
    (g, logits, s.launch_bytes, c.calls)
}

#[test]
fn shadow_writes_fewer_bytes_and_the_gpu_computes_the_same() {
    let m = small();
    let (g_full, l_full, bytes_full, _) = generate(&m, 6, false, false, None);
    let (g_sh, l_sh, bytes_sh, _) = generate(&m, 6, true, false, None);
    assert_eq!(g_full.stop, Some(Stop::Gata));
    assert_eq!(g_sh.stop, Some(Stop::Gata));
    assert_eq!(g_full.seq, g_sh.seq, "same tokens");
    assert_eq!(l_full.len(), 8);
    for (a, b) in l_full.iter().zip(&l_sh) {
        let (a, b): (Vec<u32>, Vec<u32>) = (
            a.iter().map(|x| x.to_bits()).collect(),
            b.iter().map(|x| x.to_bits()).collect(),
        );
        assert_eq!(a, b, "the same logits, bit for bit");
    }
    // Whole slots every time vs the first token whole and then only the changed words.
    let n = km::launches_per_token(m.layers.len()) + 1;
    assert_eq!(bytes_full, 8 * 512 * n as u64);
    assert!(
        bytes_sh < bytes_full / 4,
        "with the shadow {bytes_sh} B, without {bytes_full} B"
    );
    println!(
        "SIMULARE C-145: sloturile lansarilor, 8 pozitii x {n} lansari: {bytes_full} B scrisi intregi, {bytes_sh} B cu umbra"
    );
}

#[test]
fn check_after_gives_the_same_verdict_and_stops_at_the_same_position() {
    let m = small();
    let (g_each, _, _, calls_each) = generate(&m, 5, true, false, None);
    let (g_after, _, _, calls_after) = generate(&m, 5, true, true, None);
    assert_eq!(g_each.stop, Some(Stop::Gata));
    assert_eq!(g_after.stop, Some(Stop::Gata));
    assert_eq!(g_each.seq, g_after.seq);
    assert_eq!(g_each.worst_abs, g_after.worst_abs);
    assert_eq!(calls_each, calls_after, "every position checked, in order");
    assert_eq!(g_after.deferred_bytes(), 7 * 512 * 4);

    // The CPU disagrees at position 4 (a generated position): both stop there, with the same tokens kept.
    let (g_each, _, _, _) = generate(&m, 5, true, false, Some(4));
    let (g_after, _, _, calls_after) = generate(&m, 5, true, true, Some(4));
    assert!(matches!(g_each.stop, Some(Stop::NuTrece { pos: 4, .. })));
    assert!(matches!(g_after.stop, Some(Stop::NuTrece { pos: 4, .. })));
    assert_eq!(
        g_each.seq, g_after.seq,
        "nothing from the refused position on"
    );
    assert_eq!(g_after.seq.len(), 5);
    assert_eq!(
        calls_after,
        vec![0, 1, 2, 3, 4],
        "the check stops at the first refusal"
    );

    // At a prompt position: the prompt stays whole, nothing generated survives.
    let (g_after, _, _, _) = generate(&m, 5, true, true, Some(1));
    assert!(matches!(g_after.stop, Some(Stop::NuTrece { pos: 1, .. })));
    assert_eq!(g_after.seq, vec![65, 198, 185]);
}

static ROWS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static STICK: Mutex<Vec<String>> = Mutex::new(Vec::new());
static HELD: Mutex<Vec<String>> = Mutex::new(Vec::new());

fn line_now(e: &Event) {
    ROWS.lock().unwrap().push(e.to_string());
}

/// A stick that holds `Later` rows in memory and writes them before the next `Now` row or at `flush`.
fn line_live(e: &Event, f: Flush) {
    let mut held = HELD.lock().unwrap();
    held.push(e.to_string());
    if f == Flush::Now {
        STICK.lock().unwrap().append(&mut held);
    }
}

fn flush_live() {
    STICK.lock().unwrap().append(&mut HELD.lock().unwrap());
}

/// The rows of one os-307 token, in its order (the permit note, the window, the invalidate, the doorbell, the window).
fn token_rows(j: &mut Journal, pos: u32) {
    let w = |off: u64, v: u32| Event::MmioWrite {
        bar: 0,
        offset: off,
        before: v ^ 1,
        written: v,
        after: v,
    };
    let permit = "C-55 gpu lant: fereastra cu Adrian (P6-42)";
    for e in [
        Event::Note(permit),
        w(0x1700, 0x1b2d),
        w(0x1700, 0x10e6),
        w(0x1700, 0x101f),
        w(0x1700, 0x1fff0),
        Event::Note(permit),
        w(0xb830a0, 0x100000),
        w(0xb830a4, 0),
        w(0xb830b0, 0x8000_0001),
        Event::Note("soneria canalului de lucru (I10): scriere fara recitire"),
        w(0xbb0090, 1 + pos),
        Event::Note(permit),
        w(0x1700, 0x1b2f),
        w(0x1700, 0x101f),
        w(0x1700, 0x1fff0),
    ] {
        j.push(e);
    }
}

#[test]
fn held_routine_rows_reach_the_stick_whole_and_in_order() {
    const NOTES: [&str; 2] = [
        "C-55 gpu lant: fereastra cu Adrian (P6-42)",
        "soneria canalului de lucru (I10): scriere fara recitire",
    ];
    let run = |j: &mut Journal, routine: bool| {
        j.push(Event::Value {
            what: "kg simboluri cerute",
            value: 16,
        });
        if routine {
            j.set_routine(Some(kf::gen_routine(&NOTES)));
        }
        for pos in 0..18 {
            token_rows(j, pos);
            j.flush_live(); // the OS: once per token
        }
        j.set_routine(None);
        j.push(Event::Value {
            what: "kv simboluri",
            value: 18,
        });
        j.push(Event::Step {
            name: "i14-simbol",
            ok: true,
        });
    };
    let mut a = Journal::with_sink(4096, line_now);
    run(&mut a, false);
    let mut b = Journal::with_live_sink(
        4096,
        LiveSink {
            line: line_live,
            flush: flush_live,
        },
    );
    run(&mut b, true);
    let rows = ROWS.lock().unwrap().clone();
    let stick = STICK.lock().unwrap().clone();
    assert_eq!(rows.len(), 3 + 18 * 15);
    assert_eq!(stick, rows, "the same rows, the same order");
    assert!(HELD.lock().unwrap().is_empty(), "nothing left in memory");
    let st = b.live_stats();
    assert_eq!(st.held, 18 * 15, "every routine row of the tokens was held");
    assert_eq!(
        st.flushes, 18,
        "one flush of the stick per token instead of 15"
    );
    // The chain checker reads both the same (it reads text).
    let text = |v: &[String]| -> String {
        v.iter()
            .enumerate()
            .map(|(i, r)| format!("[  126.{i:06}] gpu: {r}\n"))
            .collect()
    };
    assert_eq!(text(&stick), text(&rows));
}
