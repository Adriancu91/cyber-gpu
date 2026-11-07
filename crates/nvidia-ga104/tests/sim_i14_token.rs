//! C-83 I14, SIMULARE: whole tokens of a (small, synthetic) Kronos model on the work channel — embedding, every
//! layer, final LayerNorm, logits — `16·layers + 3` launches per token in one GPFIFO entry, then greedy
//! generation: the GPU model's tokens and logits against the CPU engine (`gpu_compute::model::step_cpu`). The
//! "GPU" is `gpu_sim::cmdproc` with the program models; nothing here proves anything about the board or #513.

mod common;

use common::{Rig, get_bytes, gpu_runs, put_bytes, rig, rig_with_push, span};
use gpu_compute::layer::Shape;
use gpu_compute::model::{self, Caches};
use gpu_core::error::GpuResult;
use gpu_sim::cmdproc::{
    ATTENTION_NEEDS, CommandProcessor, LAYERNORM_NEEDS, Needs, ProgramFn, Q8_MATVEC_NEEDS,
    attention_on_cpu, copy_b128_on_cpu, embed_on_cpu, gelu_on_cpu, layernorm_on_cpu,
    q8_matvec_on_cpu, vadd_on_cpu,
};
use nvidia_ga104::gsp::golden::{GoldenMemory, Mapping};
use nvidia_ga104::kronos_gen::{
    self as kgen, GenPlan, Generation, Stop, TokenCheck, TokenOutput, TokenRunner, VramBudget,
};
use nvidia_ga104::kronos_layer::{self as kl, LayerVas, PROGRAMS, Prog, WeightVas};
use nvidia_ga104::kronos_model::{self as km, ModelVas, OwnedModel};
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::transfer;

const FIRST: u32 = 0xa00;
/// Payloads grow with the position (the last record is the fence of the pushbuffer stretch).
fn first(pos: usize) -> u32 {
    FIRST + 0x1000 * pos as u32
}
const NONE: Needs = Needs {
    shared_bytes: 0,
    barriers: 0,
};
const MODELS: [(Prog, ProgramFn, Needs); 7] = [
    (Prog::Copy, copy_b128_on_cpu, NONE),
    (Prog::Q8Matvec, q8_matvec_on_cpu, Q8_MATVEC_NEEDS),
    (Prog::Vadd, vadd_on_cpu, NONE),
    (Prog::LayerNorm, layernorm_on_cpu, LAYERNORM_NEEDS),
    (Prog::Attention, attention_on_cpu, ATTENTION_NEEDS),
    (Prog::Gelu, gelu_on_cpu, NONE),
    (Prog::Embed, embed_on_cpu, NONE),
];

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Gpu {
    progs: Vec<(Prog, Mapping)>,
    all: Vec<Mapping>,
    out: Vec<Mapping>,
    launches: Mapping,
    records: Mapping,
    vas: ModelVas,
    n: usize,
}

fn put(r: &mut Rig, all: &mut Vec<Mapping>, bytes: &[u8]) -> u64 {
    let m = r.mem.alloc_mapped(bytes.len() as u64, 0x1000).unwrap();
    put_bytes(r.mem.writer_mut(), m.pa, bytes);
    all.push(m);
    m.va
}

fn upload(r: &mut Rig, m: &OwnedModel) -> Gpu {
    let e = m.shape.e;
    let progs: Vec<(Prog, Mapping)> = PROGRAMS
        .iter()
        .map(|&p| {
            let mm = r.mem.alloc_mapped(p.code().len() as u64, 0x1000).unwrap();
            put_bytes(r.mem.writer_mut(), mm.pa, p.code());
            (p, mm)
        })
        .collect();
    let mut all = Vec::new();
    let emb_tok = put(r, &mut all, &m.emb_tok);
    let emb_pos = put(r, &mut all, &m.emb_pos);
    let ln_g = put(r, &mut all, &f32s(&m.ln_g));
    let ln_b = put(r, &mut all, &f32s(&m.ln_b));
    let scratch = r.mem.alloc_mapped(kl::scratch_bytes(e), 0x1000).unwrap();
    let mut out = vec![scratch];
    let mut layers = Vec::new();
    for l in &m.layers {
        let wv = WeightVas {
            ln1_g: put(r, &mut all, &f32s(&l.ln1_g)),
            ln1_b: put(r, &mut all, &f32s(&l.ln1_b)),
            wqkv: put(r, &mut all, &l.wqkv),
            bqkv: put(r, &mut all, &f32s(&l.bqkv)),
            wout: put(r, &mut all, &l.wout),
            bout: put(r, &mut all, &f32s(&l.bout)),
            ln2_g: put(r, &mut all, &f32s(&l.ln2_g)),
            ln2_b: put(r, &mut all, &f32s(&l.ln2_b)),
            wfc: put(r, &mut all, &l.wfc),
            bfc: put(r, &mut all, &f32s(&l.bfc)),
            wproj: put(r, &mut all, &l.wproj),
            bproj: put(r, &mut all, &f32s(&l.bproj)),
        };
        let kc = r.mem.alloc_mapped(kl::cache_bytes(e), 0x1000).unwrap();
        let vc = r.mem.alloc_mapped(kl::cache_bytes(e), 0x1000).unwrap();
        out.extend([kc, vc]);
        layers.push(LayerVas::with_scratch(e, scratch.va, kc.va, vc.va, wv).unwrap());
    }
    let fin = r
        .mem
        .alloc_mapped(4 * (e + m.vocab) as u64, 0x1000)
        .unwrap();
    out.push(fin);
    let n = km::launches_per_token(m.layers.len());
    let launches = r.mem.alloc_mapped(512 * n as u64, 0x1000).unwrap();
    let records = r.mem.alloc_mapped(16 * (n as u64 + 1), 0x1000).unwrap();
    let vas = ModelVas {
        emb_tok,
        emb_pos,
        ln_g,
        ln_b,
        layers,
        h_final: fin.va,
        logits: fin.va + 4 * e as u64,
    };
    Gpu {
        progs,
        all,
        out,
        launches,
        records,
        vas,
        n,
    }
}

fn token(r: &mut Rig, g: &Gpu, m: &OwnedModel, tok: usize, pos: usize) -> GpuResult<Vec<f32>> {
    token_out(r, g, m, tok, pos, first(pos)).map(|o| o.logits)
}

/// One token with its records' times (board clock = the model's in SIMULARE); `first` = payload of record 0.
fn token_out(
    r: &mut Rig,
    g: &Gpu,
    m: &OwnedModel,
    tok: usize,
    pos: usize,
    first: u32,
) -> GpuResult<TokenOutput> {
    let list = km::token_launches(&m.shape, m.vocab, m.context, &g.vas, tok, pos)?;
    let va = |p: Prog| g.progs.iter().find(|(q, _)| *q == p).unwrap().1.va;
    let mut qmds = Vec::new();
    let w = r.mem.writer_mut();
    for (k, l) in list.iter().enumerate() {
        let (cb_va, cb_pa) = (
            g.launches.va + 512 * k as u64,
            g.launches.pa + 512 * k as u64,
        );
        put_bytes(w, cb_pa, &l.cbuf);
        put_bytes(w, cb_pa + 256, &l.qmd(va(l.prog), cb_va)?);
        qmds.push(cb_va + 256);
    }
    put_bytes(w, g.records.pa, &vec![0; 16 * (g.n + 1)]);
    let mut words = vec![0u32; transfer::sequence_words(qmds.len())];
    let mut pb = PushBuf::new(&mut words);
    transfer::encode_sequence(&mut pb, &qmds, g.records.va, first)?;
    let fence = g.records.va + 16 * g.n as u64;
    let done = (first + g.n as u32) as u64;
    r.wq.submit_fenced(r.mem.writer_mut(), &words, fence, done)?;
    let programs: Vec<(&[u8], ProgramFn, Needs)> =
        MODELS.iter().map(|&(p, f, n)| (p.code(), f, n)).collect();
    let loads: Vec<(u64, u64)> = g
        .progs
        .iter()
        .map(|(_, m)| m)
        .chain(&g.all)
        .chain(&g.out)
        .chain([&g.launches, &g.records])
        .map(span)
        .collect();
    let stores: Vec<(u64, u64)> = g.out.iter().chain([&g.records]).map(span).collect();
    let cp: CommandProcessor = gpu_runs(r, &programs, &loads, &stores)?;
    assert_eq!(cp.timestamps.len(), g.n + 1);
    let rec = get_bytes(r.mem.writer_mut(), g.records.pa, 16 * (g.n as u64 + 1));
    let durations = transfer::analyze_sequence(&rec, g.n, first)?;
    // The driver read the last record: its pushbuffer stretch can be reused.
    let fence = g.records.va + 16 * g.n as u64;
    let last = u32::from_le_bytes(rec[16 * g.n..16 * g.n + 4].try_into().unwrap()) as u64;
    assert!(r.wq.retire(fence, last)? >= 1);
    assert_eq!(r.wq.push_in_flight(), 0);
    let fin = g.out.last().unwrap();
    let time = |k: usize| u64::from_le_bytes(rec[16 * k + 8..16 * k + 16].try_into().unwrap());
    Ok(TokenOutput {
        logits: get_bytes(
            r.mem.writer_mut(),
            fin.pa + 4 * m.shape.e as u64,
            4 * m.vocab as u64,
        )
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect(),
        start_ns: time(0),
        end_ns: time(g.n),
        busy_ns: durations.iter().sum(),
    })
}

/// Prompt, then greedy tokens, on the GPU model and on the CPU engine side by side.
fn generate(m: &OwnedModel, prompt: &[usize], more: usize) -> (Vec<usize>, Vec<km::TokenVerdict>) {
    let mut r = rig();
    let g = upload(&mut r, m);
    let w = m.weights();
    let mut c = Caches::new(m.layers.len(), m.shape.e);
    let (mut seq, mut verdicts) = (prompt.to_vec(), Vec::new());
    for pos in 0..prompt.len() + more {
        let tok = seq[pos];
        let got = token(&mut r, &g, m, tok, pos).unwrap();
        let cpu = model::step_cpu(&w, &mut c, tok, pos).unwrap();
        verdicts.push(km::judge(&got, &cpu));
        if pos + 1 >= seq.len() && seq.len() < prompt.len() + more {
            seq.push(model::argmax(&got).unwrap());
        }
    }
    (seq, verdicts)
}

#[test]
fn greedy_tokens_of_a_small_model_match_the_cpu() {
    let m = km::test_model(
        Shape {
            e: 64,
            heads: 2,
            eps: 1e-5,
        },
        2,
        256,
        16,
        7,
    )
    .unwrap();
    let (seq, v) = generate(&m, &[5, 17, 3], 5);
    for (pos, x) in v.iter().enumerate() {
        assert!(x.passes(), "{}", x.line(pos));
    }
    // The CPU alone gives the same continuation.
    let w = m.weights();
    let mut c = Caches::new(2, 64);
    let mut cpu_seq = vec![5, 17, 3];
    for pos in 0..7 {
        let l = model::step_cpu(&w, &mut c, cpu_seq[pos], pos).unwrap();
        if pos + 1 >= cpu_seq.len() {
            cpu_seq.push(model::argmax(&l).unwrap());
        }
    }
    assert_eq!(seq, cpu_seq);
    assert_eq!(seq.len(), 8);
}

#[test]
fn kronos_width_one_token() {
    let m = km::test_model(
        Shape {
            e: 768,
            heads: 12,
            eps: 1e-5,
        },
        1,
        512,
        8,
        9,
    )
    .unwrap();
    let (_, v) = generate(&m, &[42], 1);
    for (pos, x) in v.iter().enumerate() {
        assert!(x.passes(), "{}", x.line(pos));
    }
}

/// A wrong embedding row (token index off by one) never passes.
#[test]
fn wrong_token_row_is_caught() {
    let m = km::test_model(
        Shape {
            e: 64,
            heads: 2,
            eps: 1e-5,
        },
        1,
        256,
        16,
        11,
    )
    .unwrap();
    let mut r = rig();
    let g = upload(&mut r, &m);
    let got = token(&mut r, &g, &m, 6, 0).unwrap();
    let mut c = Caches::new(1, 64);
    let cpu = model::step_cpu(&m.weights(), &mut c, 5, 0).unwrap();
    assert!(!km::judge(&got, &cpu).passes());
    // A token outside the vocabulary is refused before anything is written.
    assert!(token(&mut r, &g, &m, 256, 1).is_err());
    // A fence value that does not grow is refused (the stretch could never be told apart).
    let fence = g.records.va + 16 * g.n as u64;
    let same = (first(0) + g.n as u32) as u64;
    assert!(
        r.wq.submit_fenced(r.mem.writer_mut(), &[0], fence, same)
            .is_err()
    );
    assert!(
        r.wq.submit_fenced(r.mem.writer_mut(), &[0], 0x10, same + 1)
            .is_err(),
        "in afara VA"
    );
}

// ------------------------------------------------------------------ C-87 step 1: continuous generation, SIMULARE

/// `kronos_gen::TokenRunner` over the simulated channel; payloads keep growing across positions (fence values).
struct SimRunner<'a> {
    r: Rig,
    g: Gpu,
    m: &'a OwnedModel,
    calls: usize,
    /// Before this position, the GELU program's code in VRAM is wiped (a fault the board could show).
    break_at: Option<usize>,
}

impl TokenRunner for SimRunner<'_> {
    fn run_token(&mut self, token: usize, pos: usize) -> GpuResult<TokenOutput> {
        self.calls += 1;
        if self.break_at == Some(pos) {
            let (_, pm) = self.g.progs.iter().find(|(p, _)| *p == Prog::Gelu).unwrap();
            put_bytes(self.r.mem.writer_mut(), pm.pa, &vec![0; pm.size as usize]);
        }
        token_out(&mut self.r, &self.g, self.m, token, pos, first(pos))
    }
}

struct CpuCheck<'a> {
    w: gpu_compute::model::ModelWeights<'a>,
    c: Caches,
}

impl TokenCheck for CpuCheck<'_> {
    fn cpu_logits(&mut self, token: usize, pos: usize) -> GpuResult<Vec<f32>> {
        model::step_cpu(&self.w, &mut self.c, token, pos)
    }
}

fn sim_runner(m: &OwnedModel, break_at: Option<usize>) -> SimRunner<'_> {
    // Two positions in flight, as the OS sizes it (kronos_gen::push_bytes_with_copy).
    let push = (2 * kgen::push_bytes_with_copy(m.layers.len()) as u64).next_multiple_of(0x1000);
    let mut r = rig_with_push(push);
    let g = upload(&mut r, m);
    SimRunner {
        r,
        g,
        m,
        calls: 0,
        break_at,
    }
}

fn small(seed: u32, context: usize) -> OwnedModel {
    km::test_model(
        Shape {
            e: 64,
            heads: 2,
            eps: 1e-5,
        },
        2,
        256,
        context,
        seed,
    )
    .unwrap()
}

/// 40 greedy tokens in a row: the caches grow by one position each time, every position one fenced submission whose
/// pushbuffer stretch is given back before the next; every logit vector judged against the CPU; the speed from the
/// (model's) clock.
#[test]
fn continuous_generation_of_40_tokens() {
    let m = small(7, 64);
    let mut run = sim_runner(&m, None);
    let mut check = CpuCheck {
        w: m.weights(),
        c: Caches::new(2, 64),
    };
    let mut gn = Generation::new(GenPlan::new(&[5, 17, 3], 40, m.vocab, m.context).unwrap());
    let stop = gn.run(&mut run, Some(&mut check)).clone();
    assert_eq!(stop, Stop::Gata, "{}", stop.line());
    assert_eq!(gn.generated().len(), 40);
    assert_eq!(run.calls, 42);
    assert_eq!(run.r.wq.push_in_flight(), 0);
    // The CPU alone, greedy, gives the same 43 tokens.
    let w = m.weights();
    let mut c = Caches::new(2, 64);
    let mut cpu = vec![5, 17, 3];
    for pos in 0..42 {
        let l = model::step_cpu(&w, &mut c, cpu[pos], pos).unwrap();
        if pos + 1 >= cpu.len() {
            cpu.push(model::argmax(&l).unwrap());
        }
    }
    assert_eq!(gn.seq, cpu);
    let s = gn.stats();
    eprintln!(
        "{} (abaterea maxima {:e})",
        s.line("SIMULARE"),
        gn.worst_abs
    );
    assert_eq!(s.positions, 42);
    // The model's clock: 35 launches of 250 µs + methods per position.
    assert!(s.busy_ns >= 42 * 35 * 250_000 && s.span_ns >= s.busy_ns);
    assert!(s.per_s_x100().unwrap() > 0);
}

/// A fault in the middle (a program wiped before position 5): the loop stops at that position, nothing more is
/// submitted, the tokens made before are kept and are the CPU's. The stretch of the failed submission stays owned by
/// the "GPU" (`push_in_flight` > 0): giving it back is the recovery's job (C-87 step 3), never a guess here.
#[test]
fn generation_stops_cleanly_at_a_fault() {
    let m = small(7, 64);
    let mut run = sim_runner(&m, Some(5));
    let mut gn = Generation::new(GenPlan::new(&[5, 17, 3], 40, m.vocab, m.context).unwrap());
    let stop = gn.run(&mut run, None).clone();
    assert!(
        matches!(stop, Stop::Eroare { pos: 5, .. }),
        "{}",
        stop.line()
    );
    assert_eq!(run.calls, 6, "nimic trimis dupa eroare");
    assert_eq!(gn.generated().len(), 3);
    assert_eq!(gn.next(), None);
    assert!(run.r.wq.push_in_flight() > 0);
    let (seq, _) = generate(&m, &[5, 17, 3], 3);
    assert_eq!(gn.seq, seq);
}

/// C-87 step 3, SIMULARE: after the fault above, the recovery with GSP-RM's model (`FakeRm`) and a simulated
/// config space: the group is taken off the runlist by the RM control, Bus Master goes off (re-read), the RAM pages
/// are given back; with GSP silent the same steps leave them in quarantine and the board off until a restart.
#[test]
fn recovery_after_a_fault_without_a_restart() {
    use gpu_core::diag::Journal;
    use gpu_core::error::GpuError;
    use gpu_pci::bus_master::{BusMasterPermit, set_bus_master};
    use gpu_sim::pci::SimConfig;
    use nvidia_ga104::gsp::user_space::{query_gr_info, stop_compute_group};
    use nvidia_ga104::gsp::work_queue::{ChannelMemory, USERD_GP_GET, USERD_GP_PUT};
    use nvidia_ga104::recovery::{self, NextChain, RamPages, RecoveryOps};

    struct Ops<'a> {
        r: &'a mut Rig,
        cfg: SimConfig,
        silent: bool,
    }
    impl RecoveryOps for Ops<'_> {
        fn gsp_ping(&mut self) -> GpuResult<()> {
            if self.silent {
                return Err(GpuError::Timeout { waited_us: 500_000 });
            }
            query_gr_info(&mut self.r.rm, &self.r.u).map(|_| ())
        }
        fn stop_group(&mut self) -> GpuResult<()> {
            stop_compute_group(&mut self.r.rm, &self.r.u, &self.r.cg)
        }
        fn ring_pointers(&mut self) -> GpuResult<(u32, u32)> {
            let userd = self.r.w.ring.pa + self.r.w.userd_offset;
            let m = self.r.mem.writer_mut();
            Ok((
                m.read_u32(userd + USERD_GP_GET)?,
                m.read_u32(userd + USERD_GP_PUT)?,
            ))
        }
        fn bus_master_off(&mut self) -> GpuResult<bool> {
            let mut j = Journal::new(16);
            set_bus_master(
                &mut self.cfg,
                &BusMasterPermit::new("recuperare"),
                &mut j,
                false,
            )?;
            Ok(self.cfg.read_raw(4) & 4 == 0)
        }
    }

    for silent in [false, true] {
        let m = small(7, 64);
        let mut run = sim_runner(&m, Some(5));
        let mut gn = Generation::new(GenPlan::new(&[5, 17, 3], 40, m.vocab, m.context).unwrap());
        assert!(matches!(
            gn.run(&mut run, None),
            Stop::Eroare { pos: 5, .. }
        ));
        assert!(run.r.rm.0.is_scheduled(run.r.cg.channel_group));
        let mut cfg = SimConfig::workstation_rtx3070ti();
        cfg.write_raw(4, cfg.read_raw(4) | 0x6); // memory decode + Bus Master on, as in the window
        let mut ops = Ops {
            r: &mut run.r,
            cfg,
            silent,
        };
        let got = recovery::recover(&mut ops);
        for l in &got.lines {
            eprintln!("SIMULARE: {l}");
        }
        assert!(got.bus_master_off);
        assert!(got.ring.is_some());
        let tsg = run.r.cg.channel_group;
        if silent {
            assert_eq!(
                (got.ram, got.next),
                (RamPages::Carantina, NextChain::DoarDupaRepornire)
            );
            assert!(
                run.r.rm.0.is_scheduled(tsg),
                "fara GSP, grupul nu a putut fi oprit"
            );
        } else {
            assert_eq!(
                (got.ram, got.next),
                (RamPages::Eliberate, NextChain::Permis)
            );
            assert!(!run.r.rm.0.is_scheduled(tsg));
        }
    }
}

/// The context is the limit: a 16-position model stops at 16 positions with "context plin", not with an error.
#[test]
fn generation_stops_at_a_full_context() {
    let m = small(3, 16);
    let mut run = sim_runner(&m, None);
    let mut gn = Generation::new(GenPlan::new(&[1, 2], 100, m.vocab, m.context).unwrap());
    assert_eq!(*gn.run(&mut run, None), Stop::ContextPlin);
    assert_eq!(gn.stats().positions, 16);
    assert_eq!(run.calls, 16);
}

// ------------------------------------------------------------------ C-87 step 2: one program per token, SIMULARE

/// The token through ONE launch of `token.comp` (work list + barrier counter), on its own rig.
struct OneProgram<'a> {
    r: Rig,
    g: Gpu,
    m: &'a OwnedModel,
    prog: Mapping,
    list: Mapping,
    launch: Mapping,
    groups: u32,
    launches_seen: usize,
    /// Plants an unknown operation number in this entry of the list (after it is written).
    tamper: Option<usize>,
}

fn one_program(m: &OwnedModel, groups: u32) -> OneProgram<'_> {
    use nvidia_ga104::kronos_token as kt;
    let mut r = rig();
    let g = upload(&mut r, m);
    let prog = r.mem.alloc_mapped(kt::TOKEN.len() as u64, 0x1000).unwrap();
    put_bytes(r.mem.writer_mut(), prog.pa, kt::TOKEN);
    let list = r
        .mem
        .alloc_mapped((kt::ENTRY * g.n) as u64, 0x1000)
        .unwrap();
    // cbuf (256) + QMD (256) + the barrier counter (16).
    let launch = r.mem.alloc_mapped(0x1000, 0x1000).unwrap();
    OneProgram {
        r,
        g,
        m,
        prog,
        list,
        launch,
        groups,
        launches_seen: 0,
        tamper: None,
    }
}

impl TokenRunner for OneProgram<'_> {
    fn run_token(&mut self, tok: usize, pos: usize) -> GpuResult<TokenOutput> {
        use gpu_sim::cmdproc::{TOKEN_NEEDS, token_on_cpu};
        use nvidia_ga104::kronos_token as kt;
        let (m, g) = (self.m, &self.g);
        let list = km::token_launches(&m.shape, m.vocab, m.context, &g.vas, tok, pos)?;
        let w = self.r.mem.writer_mut();
        put_bytes(w, self.list.pa, &kt::work_list(&list)?);
        if let Some(k) = self.tamper {
            put_bytes(
                w,
                self.list.pa + (kt::ENTRY * k + kt::ENTRY_OP) as u64,
                &[9, 0, 0, 0],
            );
        }
        let (cb, qmd, sync) = (self.launch.va, self.launch.va + 256, self.launch.va + 512);
        put_bytes(
            w,
            self.launch.pa,
            &kt::token_cbuf(self.list.va, sync, list.len())?,
        );
        put_bytes(
            w,
            self.launch.pa + 256,
            &kt::token_qmd(self.prog.va, cb, self.groups)?,
        );
        put_bytes(w, self.launch.pa + 512, &[0; 16]);
        put_bytes(w, g.records.pa, &[0; 32]);
        let first = first(pos);
        let mut words = vec![0u32; transfer::sequence_words(1)];
        let mut pb = PushBuf::new(&mut words);
        transfer::encode_sequence(&mut pb, &[qmd], g.records.va, first)?;
        let done = (first + 1) as u64;
        self.r
            .wq
            .submit_fenced(self.r.mem.writer_mut(), &words, g.records.va + 16, done)?;
        let mut programs: Vec<(&[u8], ProgramFn, Needs)> =
            MODELS.iter().map(|&(p, f, n)| (p.code(), f, n)).collect();
        programs.push((kt::TOKEN, token_on_cpu, TOKEN_NEEDS));
        let loads: Vec<(u64, u64)> = g
            .all
            .iter()
            .chain(&g.out)
            .chain([&g.records, &self.prog, &self.list, &self.launch])
            .map(span)
            .collect();
        let stores: Vec<(u64, u64)> = g
            .out
            .iter()
            .chain([&g.records, &self.launch])
            .map(span)
            .collect();
        let cp = gpu_runs(&mut self.r, &programs, &loads, &stores)?;
        self.launches_seen += cp
            .events
            .iter()
            .filter(|e| matches!(e, gpu_sim::cmdproc::Event::Launch { .. }))
            .count();
        let rec = get_bytes(self.r.mem.writer_mut(), g.records.pa, 32);
        let d = transfer::analyze_sequence(&rec, 1, first)?;
        let last = u32::from_le_bytes(rec[16..20].try_into().unwrap()) as u64;
        self.r.wq.retire(g.records.va + 16, last)?;
        // The barrier counter after the launch: (operations - 1) · G.
        let c = get_bytes(self.r.mem.writer_mut(), self.launch.pa + 512, 4);
        assert_eq!(
            u32::from_le_bytes(c[..4].try_into().unwrap()),
            (list.len() as u32 - 1) * self.groups
        );
        let fin = g.out.last().unwrap();
        let time = |k: usize| u64::from_le_bytes(rec[16 * k + 8..16 * k + 16].try_into().unwrap());
        Ok(TokenOutput {
            logits: get_bytes(
                self.r.mem.writer_mut(),
                fin.pa + 4 * m.shape.e as u64,
                4 * m.vocab as u64,
            )
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
            start_ns: time(0),
            end_ns: time(1),
            busy_ns: d[0],
        })
    }
}

/// One launch per token gives, bit for bit, the logits of the 35 separate launches, position after position, and the
/// same greedy tokens; the pushbuffer of a token shrinks from 35 launches to 1.
#[test]
fn one_program_per_token_equals_the_separate_launches() {
    let m = small(7, 64);
    let mut sep = sim_runner(&m, None);
    let mut one = one_program(&m, 48);
    let mut seq = vec![5usize, 17, 3];
    for pos in 0..12 {
        let a = sep.run_token(seq[pos], pos).unwrap();
        let b = one.run_token(seq[pos], pos).unwrap();
        assert!(
            a.logits
                .iter()
                .zip(&b.logits)
                .all(|(x, y)| x.to_bits() == y.to_bits()),
            "pozitia {pos}: logit-urile difera"
        );
        if pos + 1 >= seq.len() {
            seq.push(model::argmax(&b.logits).unwrap());
        }
    }
    assert_eq!(one.launches_seen, 12, "o singura lansare pe simbol");
    assert_eq!(km::launches_per_token(2), 35);
    assert_eq!(transfer::sequence_words(1), 18);
    // And as a generation with the CPU check.
    let mut one = one_program(&m, 7);
    let mut check = CpuCheck {
        w: m.weights(),
        c: Caches::new(2, 64),
    };
    let mut gn = Generation::new(GenPlan::new(&[5, 17, 3], 10, m.vocab, m.context).unwrap());
    assert_eq!(*gn.run(&mut one, Some(&mut check)), Stop::Gata);
    assert_eq!(gn.seq, seq);
}

/// The same at Kronos's width (E 768, 12 heads, 1 layer): the widest operations (3072 rows) spread over 48 groups.
#[test]
fn one_program_per_token_at_kronos_width() {
    let m = km::test_model(
        Shape {
            e: 768,
            heads: 12,
            eps: 1e-5,
        },
        1,
        512,
        8,
        9,
    )
    .unwrap();
    let mut sep = sim_runner(&m, None);
    let mut one = one_program(&m, 48);
    for (pos, tok) in [42usize, 7].into_iter().enumerate() {
        let a = sep.run_token(tok, pos).unwrap();
        let b = one.run_token(tok, pos).unwrap();
        assert!(
            a.logits
                .iter()
                .zip(&b.logits)
                .all(|(x, y)| x.to_bits() == y.to_bits())
        );
    }
}

/// A list the program cannot run is refused by the model; `work_list` refuses an empty one before anything is sent.
#[test]
fn one_program_refuses_a_bad_list() {
    use nvidia_ga104::kronos_token as kt;
    let m = small(7, 64);
    let mut one = one_program(&m, 4);
    assert!(one.run_token(5, 0).is_ok());
    // An unknown operation number in entry 3: the model stops, the launch's record never comes, no logits.
    one.tamper = Some(3);
    assert!(one.run_token(17, 1).is_err());
    assert!(kt::work_list(&[]).is_err());
}

/// C-87 step 4: the budget of the test model matches what the rig really mapped (an upper bound, page-rounded).
#[test]
fn the_vram_budget_covers_what_is_uploaded() {
    let m = small(7, 64);
    let b = VramBudget::of_model(&m).unwrap();
    let mut r = rig();
    let before = r.mem.vram_free().0;
    let _g = upload(&mut r, &m);
    let used = before - r.mem.vram_free().0;
    eprintln!("{} ; rig-ul a folosit {} octeti", b.line(2), used);
    assert!(
        b.total() >= used,
        "bugetul {} < folosit {}",
        b.total(),
        used
    );
    assert!(b.total() < 2 * used);
    assert!(b.check(before, r.mem.vram_free().1 + used, 0).is_ok());
}

/// C-87 step 4: the 24-layer path (the 183M model's depth: `16·24 + 3 = 387` launches per position, one pushbuffer
/// stretch of 15 512 B), narrow (E 128) so the simulation stays quick; 4 positions, every one judged against the CPU.
/// The real width (768) is covered by `kronos_width_one_token` (1 layer) and the budget by `kronos_gen`'s tests.
#[test]
fn a_24_layer_model_on_the_work_channel() {
    let m = km::test_model(
        Shape {
            e: 128,
            heads: 4,
            eps: 1e-5,
        },
        24,
        512,
        32,
        24,
    )
    .unwrap();
    assert_eq!(km::launches_per_token(24), 387);
    let mut run = sim_runner(&m, None);
    let mut check = CpuCheck {
        w: m.weights(),
        c: Caches::new(24, 128),
    };
    let mut gn = Generation::new(GenPlan::new(&[9, 99], 3, m.vocab, m.context).unwrap());
    let stop = gn.run(&mut run, Some(&mut check)).clone();
    assert_eq!(stop, Stop::Gata, "{}", stop.line());
    assert_eq!(run.calls, 4);
    eprintln!(
        "24 straturi: {} (abaterea maxima {:e})",
        gn.stats().line("SIMULARE"),
        gn.worst_abs
    );
}

// ------------------------------------------------------------------ the KGR1 control vectors (hub, S39/C-24)
//
// The hub is private and this repo is public, so the vectors are NOT in the repo. They are prepared locally from a
// clone of the hub by `tools/vectori-kgr1/pregateste.py` (into `target/vectori-kgr1/`, never committed) and read at
// run time from the folder in `KGR1_VECTORI`. Without it these tests are ignored; run with `--ignored` and no folder,
// they FAIL (never a silent pass). The files: a tiny Kronos (2 layers, 4 heads, n_emb 32, vocabulary 16 384) with
// RANDOM weights, and what Kronos's own engine without PyTorch (`unelte/inferenta_bruta.py`, cyberspace) gives on
// three texts. The checks are the hub README's: argmax on every position identical, 5 greedy tokens identical, the
// last-position logits within 1e-3 (absolute). It ties our CPU oracle — and, in SIMULATION, our program models on
// the work channel — to Kronos's reference engine. It says nothing about the board.

const VECTORS_NEED: &str = "vectorii KGR1 din hub: python3 -I tools/vectori-kgr1/pregateste.py <hub>/integrare/kronos/vectori target/vectori-kgr1, apoi KGR1_VECTORI=$PWD/target/vectori-kgr1 cargo test --release -p nvidia-ga104 --test sim_i14_token kgr1 -- --ignored";

struct Files {
    kgr: Vec<u8>,
    txt: String,
    logits: Vec<u8>,
}

fn files() -> Files {
    let dir = std::env::var("KGR1_VECTORI")
        .unwrap_or_else(|_| panic!("lipseste KGR1_VECTORI: {VECTORS_NEED}"));
    let read = |n: &str| {
        std::fs::read(format!("{dir}/{n}"))
            .unwrap_or_else(|e| panic!("{dir}/{n}: {e}; {VECTORS_NEED}"))
    };
    Files {
        kgr: read("kronos_mic_q8_0.kgr"),
        txt: String::from_utf8(read("kronos_mic_q8_0.txt")).unwrap(),
        logits: read("kronos_mic_q8_0_logits.f32"),
    }
}

/// The hub README's tolerance for the Q8_0 vectors (absolute, last position).
const VECTOR_TOL: f32 = 1e-3;

/// A minimal KGR1 reader (hub `format_greutati.md`), host-only, for the test file.
fn read_kgr1(f: &[u8]) -> OwnedModel {
    use std::collections::HashMap;
    let mut o = 0usize;
    let mut take = |n: usize| {
        let s = &f[o..o + n];
        o += n;
        s
    };
    assert_eq!(take(4), b"KGR1");
    let u32_ = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    let u64_ = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap());
    assert_eq!(u32_(take(4)), 1);
    let (nkv, nt) = (u64_(take(8)), u64_(take(8)));
    let mut ints = HashMap::new();
    let mut eps = 0.0f64;
    for _ in 0..nkv {
        let n = u64_(take(8)) as usize;
        let k = String::from_utf8(take(n).to_vec()).unwrap();
        match u32_(take(4)) {
            1 => {
                ints.insert(k, i64::from_le_bytes(take(8).try_into().unwrap()));
            }
            2 => {
                let v = f64::from_le_bytes(take(8).try_into().unwrap());
                if k == "ln_eps" {
                    eps = v;
                }
            }
            3 => {
                let n = u64_(take(8)) as usize;
                let v = take(n);
                if k == "arhitectura" {
                    assert_eq!(v, b"kronos_gpt");
                }
            }
            other => panic!("fel necunoscut {other}"),
        }
    }
    let mut tensors = HashMap::new();
    for _ in 0..nt {
        let n = u64_(take(8)) as usize;
        let name = String::from_utf8(take(n).to_vec()).unwrap();
        let nd = u32_(take(4)) as usize;
        let dims: Vec<usize> = (0..nd).map(|_| u64_(take(8)) as usize).collect();
        let kind = u32_(take(4));
        let off = u64_(take(8)) as usize;
        tensors.insert(name, (dims, kind, off));
    }
    let data = o.div_ceil(32) * 32;
    let bytes = |name: &str, kind: u32| -> &[u8] {
        let (dims, k, off) = &tensors[name];
        assert_eq!(*k, kind, "{name}");
        let n = match kind {
            0 => 4 * dims.iter().product::<usize>(),
            _ => dims[0] * dims[1] / 32 * 36,
        };
        &f[data + off..data + off + n]
    };
    let f32s = |name: &str| -> Vec<f32> {
        bytes(name, 0)
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect()
    };
    let q8 = |name: &str| bytes(name, 1).to_vec();
    assert_eq!(ints["cap_legat"], 1);
    let n_layers = ints["n_straturi"] as usize;
    OwnedModel {
        shape: Shape {
            e: ints["n_emb"] as usize,
            heads: ints["n_capete"] as usize,
            eps: eps as f32,
        },
        vocab: ints["vocabular"] as usize,
        context: ints["context"] as usize,
        emb_tok: q8("emb_token.weight"),
        emb_pos: q8("emb_pozitie.weight"),
        layers: (0..n_layers)
            .map(|i| {
                let p = |s: &str| format!("straturi.{i}.{s}");
                kl::OwnedLayer {
                    ln1_g: f32s(&p("ln1.weight")),
                    ln1_b: f32s(&p("ln1.bias")),
                    wqkv: q8(&p("atentie.in_proj_weight")),
                    bqkv: f32s(&p("atentie.in_proj_bias")),
                    wout: q8(&p("atentie.out_proj.weight")),
                    bout: f32s(&p("atentie.out_proj.bias")),
                    ln2_g: f32s(&p("ln2.weight")),
                    ln2_b: f32s(&p("ln2.bias")),
                    wfc: q8(&p("mlp.0.weight")),
                    bfc: f32s(&p("mlp.0.bias")),
                    wproj: q8(&p("mlp.2.weight")),
                    bproj: f32s(&p("mlp.2.bias")),
                }
            })
            .collect(),
        ln_g: f32s("ln.weight"),
        ln_b: f32s("ln.bias"),
    }
}

struct Vector {
    text: String,
    prompt: Vec<usize>,
    argmax: Vec<usize>,
    greedy: Vec<usize>,
    logits: Vec<f32>,
}

fn vectors(f: &Files, vocab: usize) -> Vec<Vector> {
    let nums = |l: &str| -> Vec<usize> { l.split(' ').map(|v| v.parse().unwrap()).collect() };
    let mut out: Vec<Vector> = Vec::new();
    for line in f.txt.lines() {
        let (k, v) = line.split_once(' ').unwrap();
        match k {
            "text" => out.push(Vector {
                text: v.to_string(),
                prompt: vec![],
                argmax: vec![],
                greedy: vec![],
                logits: vec![],
            }),
            "simboluri" => out.last_mut().unwrap().prompt = nums(v),
            "argmax" => out.last_mut().unwrap().argmax = nums(v),
            "greedy" => out.last_mut().unwrap().greedy = nums(v),
            _ => {}
        }
    }
    assert_eq!(f.logits.len(), 4 * vocab * out.len());
    for (i, t) in out.iter_mut().enumerate() {
        t.logits = f.logits[4 * vocab * i..4 * vocab * (i + 1)]
            .chunks(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
    }
    out
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
}

/// Our CPU oracle alone (no GPU, no simulation) against Kronos's reference engine.
#[test]
#[ignore = "cere KGR1_VECTORI (vectorii din hub-ul privat, pregatiti local)"]
fn kgr1_control_vectors_cpu_oracle() {
    let f = files();
    let m = read_kgr1(&f.kgr);
    assert_eq!(
        (m.shape.e, m.shape.heads, m.layers.len(), m.vocab, m.context),
        (32, 4, 2, 16384, 64)
    );
    let w = m.weights();
    let vs = vectors(&f, m.vocab);
    assert_eq!(vs.len(), 3);
    for t in &vs {
        let mut c = Caches::new(m.layers.len(), m.shape.e);
        let mut seq = t.prompt.clone();
        for pos in 0..t.prompt.len() + t.greedy.len() - 1 {
            let l = model::step_cpu(&w, &mut c, seq[pos], pos).unwrap();
            let top = model::argmax(&l).unwrap();
            if pos < t.prompt.len() {
                assert_eq!(top, t.argmax[pos], "{}: pozitia {pos}", t.text);
            }
            if pos + 1 == t.prompt.len() {
                let d = max_abs(&l, &t.logits);
                eprintln!(
                    "vectori KGR1, procesor: {}: abaterea logit-urilor {d:e}",
                    t.text
                );
                assert!(d <= VECTOR_TOL, "{}: logit-uri {d:e}", t.text);
            }
            if pos + 1 >= seq.len() {
                seq.push(top);
            }
        }
        assert_eq!(&seq[t.prompt.len()..], &t.greedy[..], "{}", t.text);
    }
}

/// SIMULARE: the same vectors through the work channel (`16·2 + 3 = 35` launches per token, program models).
#[test]
#[ignore = "cere KGR1_VECTORI (vectorii din hub-ul privat, pregatiti local)"]
fn kgr1_control_vectors_on_the_work_channel() {
    let f = files();
    let m = read_kgr1(&f.kgr);
    for t in vectors(&f, m.vocab) {
        let (seq, v) = generate(&m, &t.prompt, t.greedy.len());
        for (pos, x) in v.iter().enumerate() {
            assert!(x.passes(), "{}: {}", t.text, x.line(pos));
        }
        assert_eq!(&seq[t.prompt.len()..], &t.greedy[..], "{}", t.text);
    }
    // The last-position logits of the channel against the reference engine's, for one text.
    let t = &vectors(&f, m.vocab)[0];
    let mut r = rig();
    let g = upload(&mut r, &m);
    let mut got = Vec::new();
    for (pos, &tok) in t.prompt.iter().enumerate() {
        got = token(&mut r, &g, &m, tok, pos).unwrap();
        assert_eq!(model::argmax(&got), Some(t.argmax[pos]), "pozitia {pos}");
    }
    let d = max_abs(&got, &t.logits);
    eprintln!(
        "vectori KGR1, SIMULARE pe canal: {}: abaterea logit-urilor {d:e}",
        t.text
    );
    assert!(d <= VECTOR_TOL, "logit-uri {d:e}");
}

/// The check can fail: one scale of the first token's embedding row 1 % off moves the logits out of 1e-3.
#[test]
#[ignore = "cere KGR1_VECTORI (vectorii din hub-ul privat, pregatiti local)"]
fn kgr1_control_vectors_catch_a_changed_weight() {
    let f = files();
    let mut m = read_kgr1(&f.kgr);
    let t = &vectors(&f, m.vocab)[0];
    let row = t.prompt[0] * m.shape.e / 32 * 36;
    let s = f32::from_le_bytes(m.emb_tok[row..row + 4].try_into().unwrap()) * 1.01;
    m.emb_tok[row..row + 4].copy_from_slice(&s.to_le_bytes());
    let w = m.weights();
    let mut c = Caches::new(m.layers.len(), m.shape.e);
    let mut l = Vec::new();
    for (pos, &tok) in t.prompt.iter().enumerate() {
        l = model::step_cpu(&w, &mut c, tok, pos).unwrap();
    }
    assert!(max_abs(&l, &t.logits) > VECTOR_TOL);
}
