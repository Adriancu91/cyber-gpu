//! C-155 point 3 (T145), SIMULARE: the argmax on the board (`argmax.comp`): the token's list (`TokenList::FusedLn`,
//! with the driver's diet) followed by the argmax launch, the 16-byte result in RAM. What must hold:
//! - with the copy of the logits AND the argmax in the same token (the proof run for the board), the 4 words are
//!   exactly `argmax_result` of the copied logits and the token is the CPU engine's, at every position;
//! - with the argmax alone (the fast run), the logits' RAM buffer is never written, the generation (`run_top`) makes
//!   the same tokens as the one with the copy (`run`), and the oracle passes on the copy run;
//! - the program's algorithm (threads, rounds of 8, butterfly, 8 warp picks), replayed on the processor, gives
//!   `argmax_result` for every n and for ties, -0.0/+0.0, NaN and infinities placed anywhere;
//! - NaN in the logits = "no token" (`LogitiNefiniti`), a result never written = an error, never a token;
//! - a planted wrong program is caught.
//!
//! The "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about the board.

mod common;

use common::sys_write;
use common::token_sim::TokenSim;
use gpu_compute::layer::Shape;
use gpu_compute::model::{self, ARGMAX_NONE, Caches};
use gpu_core::error::GpuResult;
use nvidia_ga104::argmax::{self as am, ArgmaxResult};
use nvidia_ga104::kronos_diag::{self as kd, SeqMode, TokenIo, TokenList, Way};
use nvidia_ga104::kronos_fast::WriteShadow;
use nvidia_ga104::kronos_gen::{
    GenPlan, Generation, Stop, TokenOutput, TokenRunner, TopOutput, TopRunner,
};
use nvidia_ga104::kronos_layer::{self as kl, PROGRAMS_ARGMAX, Prog};
use nvidia_ga104::kronos_model as km;

const PROMPT: [usize; 3] = [5, 200, 17];
const FILL: u8 = 0xee;

fn model() -> km::OwnedModel {
    km::test_model(
        Shape {
            e: 64,
            heads: 4,
            eps: 1e-5,
        },
        3,
        256,
        16,
        11,
    )
    .unwrap()
}

fn diet(s: &mut TokenSim<'_>) {
    s.push_shadow = Some(WriteShadow::new(s.r.push.pa, s.r.push.size).unwrap());
    s.zero_records = false;
}

fn fill_ram(s: &mut TokenSim<'_>) {
    let bytes = (4 * s.m.vocab as u64).next_multiple_of(16) + 16;
    let host = s.host.clone();
    sys_write(&mut s.r.sys, &host, &vec![FILL; bytes as usize]);
}

/// One token: the list, then (if `copy`) the copy of the logits down, then the argmax; RAM filled before.
fn token(s: &mut TokenSim<'_>, tok: usize, pos: usize, way: Way, copy: bool) -> GpuResult<()> {
    let (m, vas) = (s.m, s.vas.clone());
    let bytes = (4 * m.vocab as u64).next_multiple_of(16);
    let mut extra = Vec::new();
    if copy {
        extra.push(kl::copy_launch(vas.logits, s.host.va, bytes)?);
    }
    extra.push(am::launch(vas.logits, s.argmax_va(), m.vocab)?);
    fill_ram(s);
    kd::send_token_list(
        s,
        TokenList::FusedLn,
        &m.weights(),
        &vas,
        tok,
        pos,
        way,
        &extra,
    )
}

fn last_times(s: &TokenSim<'_>) -> (u64, u64, u64) {
    s.times.last().copied().unwrap_or((0, 0, 0))
}

struct CopyRunner<'s, 'a>(&'s mut TokenSim<'a>, Way);

impl TokenRunner for CopyRunner<'_, '_> {
    fn run_token(&mut self, tok: usize, pos: usize) -> GpuResult<TokenOutput> {
        token(self.0, tok, pos, self.1, true)?;
        let (start_ns, end_ns, busy_ns) = last_times(self.0);
        Ok(TokenOutput {
            logits: self.0.ram_logits(),
            start_ns,
            end_ns,
            busy_ns,
        })
    }
}

struct TopOnly<'s, 'a>(&'s mut TokenSim<'a>, Way);

impl TopRunner for TopOnly<'_, '_> {
    fn run_top(&mut self, tok: usize, pos: usize) -> GpuResult<TopOutput> {
        token(self.0, tok, pos, self.1, false)?;
        let (start_ns, end_ns, busy_ns) = last_times(self.0);
        Ok(TopOutput {
            result: self.0.ram_argmax(),
            start_ns,
            end_ns,
            busy_ns,
        })
    }
}

#[test]
fn t145a_proof_run_copy_and_argmax_in_the_same_token() {
    let m = model();
    let w = m.weights();
    for mode in [SeqMode::EachLaunch, SeqMode::PerLayer] {
        let way = Way::of(mode);
        let mut s = TokenSim::new(&m, 0);
        diet(&mut s);
        let mut c = Caches::new(m.layers.len(), m.shape.e);
        let mut tok = PROMPT[0];
        for pos in 0..m.context {
            token(&mut s, tok, pos, way, true).unwrap();
            let logits = s.ram_logits();
            let r = s.ram_argmax();
            assert!(r.agrees(&logits), "{mode:?} pozitia {pos}: {:?}", r.words);
            let cpu = model::step_cpu(&w, &mut c, tok, pos).unwrap();
            let v = km::judge(&logits, &cpu);
            assert!(v.passes(), "{mode:?}: {}", v.line(pos));
            let board = r.token(m.vocab).unwrap().unwrap();
            assert_eq!(Some(board), model::argmax(&cpu), "{mode:?} pozitia {pos}");
            tok = if pos + 1 < PROMPT.len() {
                PROMPT[pos + 1]
            } else {
                board
            };
        }
        // The list + copy + argmax on every position.
        let n = TokenList::FusedLn.launches_per_token(m.layers.len()) + 2;
        assert_eq!(s.submissions.iter().sum::<usize>(), m.context * n);
    }
}

#[test]
fn t145b_fast_run_argmax_only_same_tokens() {
    let m = model();
    let w = m.weights();
    let way = Way::of(SeqMode::EachLaunch);
    let plan = GenPlan::new(&PROMPT, m.context - PROMPT.len(), m.vocab, m.context).unwrap();

    let mut a = TokenSim::new(&m, 0);
    diet(&mut a);
    let mut with_copy = Generation::new(plan.clone());
    struct Cpu<'w>(model::ModelWeights<'w>, Caches);
    impl nvidia_ga104::kronos_gen::TokenCheck for Cpu<'_> {
        fn cpu_logits(&mut self, tok: usize, pos: usize) -> GpuResult<Vec<f32>> {
            model::step_cpu(&self.0, &mut self.1, tok, pos)
        }
    }
    let mut cpu = Cpu(w, Caches::new(m.layers.len(), m.shape.e));
    let stop = with_copy
        .run(&mut CopyRunner(&mut a, way), Some(&mut cpu))
        .clone();
    assert_eq!(stop, Stop::Gata);

    let mut b = TokenSim::new(&m, 0);
    diet(&mut b);
    let mut fast = Generation::new(plan);
    let stop = fast.run_top(&mut TopOnly(&mut b, way)).clone();
    assert_eq!(stop, Stop::Gata, "{}", stop.line());
    assert_eq!(fast.seq, with_copy.seq);
    assert_eq!(fast.generated().len(), m.context - PROMPT.len());
    // The logits never came down: the RAM buffer still holds the fill.
    assert!(b.ram_logits().iter().all(|v| v.to_bits() == 0xeeee_eeee));
    // One launch less per token; the clock was kept.
    // (the last generated token is not run: positions = prompt + generated - 1)
    let p = fast.stats().positions;
    assert_eq!(p, m.context - 1);
    let n = TokenList::FusedLn.launches_per_token(m.layers.len()) + 1;
    assert_eq!(b.submissions.iter().sum::<usize>(), p * n);
    assert_eq!(a.submissions.iter().sum::<usize>(), p * (n + 1));
    eprintln!("{p} pozitii, aceleasi simboluri: {:?}", fast.generated());
}

/// The program's algorithm (`argmax.comp`) on the processor, thread by thread: what each of the 256 threads keeps
/// after its rounds of 8, the butterfly in each warp (xor 16, 8, 4, 2, 1), lane 0's pick per warp, thread 0's
/// combination of the 8 in order. Independent of `argmax_result`: the two must agree.
fn key_of(f: f32) -> u32 {
    let b = if f.to_bits() == 0x8000_0000 {
        0
    } else {
        f.to_bits()
    };
    if b & 0x8000_0000 != 0 {
        !b
    } else {
        b | 0x8000_0000
    }
}

fn bits_of(k: u32) -> u32 {
    match k {
        0 => 0xff80_0000,
        k if k & 0x8000_0000 != 0 => k & 0x7fff_ffff,
        k => !k,
    }
}

fn program_on_cpu(v: &[f32]) -> [u32; 4] {
    let n = v.len();
    let beats = |a: (u32, u32), b: (u32, u32)| a.0 > b.0 || (a.0 == b.0 && a.1 < b.1);
    let mut th: Vec<(u32, u32, u32)> = (0..256)
        .map(|t| {
            let (mut bv, mut bi, mut c) = (0u32, ARGMAX_NONE, 0u32);
            let mut base = 0;
            while base < n {
                for k in 0..8 {
                    let i = base + t + 256 * k;
                    if i < n {
                        let x = v[i];
                        if x.to_bits() & 0x7f80_0000 == 0x7f80_0000 {
                            c += 1;
                        } else if key_of(x) > bv {
                            bv = key_of(x);
                            bi = i as u32;
                        }
                    }
                }
                base += 2048;
            }
            (bv, bi, c)
        })
        .collect();
    let mut s = 16;
    while s > 0 {
        let prev = th.clone();
        for (t, cur) in th.iter_mut().enumerate() {
            let o = prev[(t & !31) | ((t & 31) ^ s)];
            cur.2 += o.2;
            if beats((o.0, o.1), (cur.0, cur.1)) {
                cur.0 = o.0;
                cur.1 = o.1;
            }
        }
        s >>= 1;
    }
    // Every lane of a warp ends with the same pick.
    for w in 0..8 {
        assert!(
            (0..32).all(|l| th[32 * w + l].1 == th[32 * w].1 && th[32 * w + l].2 == th[32 * w].2)
        );
    }
    let (mut rv, mut ri, mut rc) = th[0];
    for w in 1..8 {
        let p = th[32 * w];
        if beats((p.0, p.1), (rv, ri)) {
            rv = p.0;
            ri = p.1;
        }
        rc += p.2;
    }
    [
        if rc == 0 { ri } else { ARGMAX_NONE },
        bits_of(rv),
        rc,
        n as u32,
    ]
}

fn lcg(seed: &mut u32) -> u32 {
    *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
    *seed >> 8
}

#[test]
fn t145c_the_programs_algorithm_is_the_oracle() {
    let mut seed = 7u32;
    for n in [
        1usize, 2, 31, 255, 256, 257, 2047, 2048, 2049, 4097, 16384, 50257,
    ] {
        for case in 0..8 {
            // Few distinct values, so ties are everywhere.
            let mut v: Vec<f32> = (0..n).map(|_| (lcg(&mut seed) % 9) as f32 - 4.0).collect();
            match case {
                1 => v.iter_mut().for_each(|x| *x = 0.0),
                2 => {
                    // -0.0 and +0.0 as the maximum
                    v.iter_mut().for_each(|x| *x = -1.0);
                    let a = lcg(&mut seed) as usize % n;
                    let b = lcg(&mut seed) as usize % n;
                    v[a] = -0.0;
                    v[b] = 0.0;
                }
                3 => {
                    let a = lcg(&mut seed) as usize % n;
                    v[a] = f32::NAN;
                }
                4 => {
                    let a = lcg(&mut seed) as usize % n;
                    v[a] = if a.is_multiple_of(2) {
                        f32::INFINITY
                    } else {
                        f32::NEG_INFINITY
                    };
                }
                5 => {
                    // the maximum only at the very end (the last round of the last thread)
                    let last = n - 1;
                    v[last] = 100.0;
                }
                6 => {
                    // subnormals around 0 as the maximum (a float compare with FTZ would call them equal)
                    v.iter_mut().for_each(|x| *x = -1.0);
                    for _ in 0..3 {
                        let a = lcg(&mut seed) as usize % n;
                        v[a] = f32::from_bits(1 + lcg(&mut seed) % 5);
                    }
                    let a = lcg(&mut seed) as usize % n;
                    v[a] = 0.0;
                }
                7 => {
                    // only negative values, the largest -FLT_MAX-ish
                    v.iter_mut().for_each(|x| *x = f32::MIN);
                }
                _ => {}
            }
            let want = model::argmax_result(&v);
            assert_eq!(program_on_cpu(&v), want, "n {n}, cazul {case}");
            assert_eq!(
                want[0],
                model::argmax(&v).map_or(ARGMAX_NONE, |i| i as u32),
                "n {n}, cazul {case}"
            );
        }
    }
}

#[test]
fn t145d_program_bytes_qmd_and_programs() {
    let sha = nvidia_ga104::sha256::digest(am::ARGMAX);
    let hex: String = sha.iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(hex, am::ARGMAX_SHA256);
    assert_eq!(am::ARGMAX.len() % 16, 0);
    let txt = include_str!("../../../tools/nak/out/argmax.sm86.txt");
    assert!(txt.contains("bar"), "fara bariera");
    assert!(txt.contains("shfl"), "fara shfl");
    assert_eq!(Prog::Argmax.code(), am::ARGMAX);
    assert_eq!(Prog::Argmax.block(), 256);
    assert_eq!(Prog::Argmax.registers(), am::REGISTERS);
    let sh = Prog::Argmax.shared().unwrap();
    assert_eq!((sh.bytes, sh.barriers), (96, 1));
    assert!(PROGRAMS_ARGMAX.contains(&Prog::Argmax));
    assert!(!TokenList::FusedLn.programs().contains(&Prog::Argmax));
    // The one-program-per-token list (`token.comp`) has no argmax: refused, not sent.
    let l = am::launch(0x10_0000, 0x20_0000, 16384).unwrap();
    assert!(nvidia_ga104::kronos_token::work_list(std::slice::from_ref(&l)).is_err());
    let q = l.qmd(0x1_0000, 0x2_0000).unwrap();
    assert_eq!(q.len(), nvidia_rm570::QMD_SIZE);
}

#[test]
fn t145e_nan_in_the_logits_is_no_token() {
    let m = model();
    let way = Way::of(SeqMode::EachLaunch);
    let mut s = TokenSim::new(&m, 0);
    token(&mut s, PROMPT[0], 0, way, false).unwrap();
    let good = s.ram_argmax();
    assert!(good.token(m.vocab).unwrap().is_some());
    // A NaN planted in the logits in VRAM, then the argmax alone again.
    let vas = s.vas.clone();
    s.write(vas.logits + 4 * 77, &[f32::NAN.to_bits()]).unwrap();
    fill_ram(&mut s);
    let l = am::launch(vas.logits, s.argmax_va(), m.vocab).unwrap();
    s.submit(&[l], 0, SeqMode::EachLaunch).unwrap();
    let r = s.ram_argmax();
    assert_eq!(r.words[0], ARGMAX_NONE);
    assert_eq!(r.nonfinite(), 1);
    assert_eq!(r.token(m.vocab).unwrap(), None);
    // In a generation: the stop is "logits not finite", at that position.
    let plan = GenPlan::new(&PROMPT, 2, m.vocab, m.context).unwrap();
    let mut g = Generation::new(plan);
    g.accept_top(Ok(TopOutput {
        result: r,
        start_ns: 1,
        end_ns: 2,
        busy_ns: 1,
    }));
    assert_eq!(g.stop, Some(Stop::LogitiNefiniti { pos: 0 }));
}

fn writes_nothing(_: &mut gpu_sim::cmdproc::Launch<'_>) -> GpuResult<()> {
    Ok(())
}

fn one_off(l: &mut gpu_sim::cmdproc::Launch<'_>) -> GpuResult<()> {
    gpu_sim::cmdproc::argmax_on_cpu(l)?;
    let res = l.mem.read_u64(l.cbuf0 + 0x30)?;
    let i = l.mem.read_u32(res)?;
    l.mem.write_u32(res, i + 1);
    Ok(())
}

#[test]
fn t145f_planted_wrong_programs_are_caught() {
    let m = model();
    let way = Way::of(SeqMode::EachLaunch);
    // Writes nothing: the fill is left, the generation stops with an error at position 0 (no token guessed).
    let mut s = TokenSim::new(&m, 0);
    for p in s.models.iter_mut() {
        if p.0 == Prog::Argmax {
            p.1 = writes_nothing;
        }
    }
    let plan = GenPlan::new(&PROMPT, 2, m.vocab, m.context).unwrap();
    let mut g = Generation::new(plan.clone());
    let stop = g.run_top(&mut TopOnly(&mut s, way)).clone();
    assert!(matches!(stop, Stop::Eroare { pos: 0, .. }), "{stop:?}");
    assert_eq!(g.seq, PROMPT.to_vec());
    // Off by one: caught by the proof run (copy + argmax), at the first position.
    let mut s = TokenSim::new(&m, 0);
    for p in s.models.iter_mut() {
        if p.0 == Prog::Argmax {
            p.1 = one_off;
        }
    }
    token(&mut s, PROMPT[0], 0, way, true).unwrap();
    assert!(!s.ram_argmax().agrees(&s.ram_logits()));
}

#[test]
fn t145g_result_reading_rejects_what_is_not_a_result() {
    let fill = ArgmaxResult::from_words([0xeeee_eeee; 4]);
    assert!(fill.token(256).is_err());
    let r = ArgmaxResult::from_words(model::argmax_result(&[0.5, 2.0, 2.0]));
    assert_eq!(r.token(3).unwrap(), Some(1));
    assert!(r.token(4).is_err());
    let plan = GenPlan::new(&PROMPT, 2, 256, 16).unwrap();
    let mut g = Generation::new(plan);
    g.accept_top(Ok(TopOutput {
        result: fill,
        start_ns: 1,
        end_ns: 2,
        busy_ns: 1,
    }));
    assert!(matches!(g.stop, Some(Stop::Eroare { pos: 0, .. })));
}
