//! C-157 (T193, `docs/C-157_ADAM_RAM.md`): AdamW's m and v in host RAM, brought to the board layer by layer by the
//! copy engine (`nvidia_ga104::adam_ram`), against the tiled plan of today with m, v on the board. **SIMULARE**: the
//! programs run on the CPU (`gpu_sim::cmdproc::*_on_cpu`), the copy engine is the model of T192, "RAM" is a range of
//! the simulated GPU VAs. No board.

use std::path::{Path, PathBuf};

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, Event, GNORM_NEEDS, LN_BACK_NEEDS, LN_FWD_NEEDS, MM_TRAIN_NEEDS, Needs,
    XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu, attn_back_tile0_on_cpu, attn_fwd_on_cpu,
    attn_fwd_tile_on_cpu, attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, emb_back_on_cpu,
    emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_fwd_on_cpu, mm_train_on_cpu,
    opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::adam_ram::{self as ar, RamMv};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{self as ts, Launch, PROGRAMS_ALL_3, Plan, Prog};
use nvidia_ga104::transfer;
use nvidia_rm570::regs::clc7c0 as c;

const MIC: Arch = train::MIC;
const VA: u64 = nvidia_ga104::mmu::VA_BASE;
const ZONE: u64 = VA + 0x20_0000;
const REC: u64 = VA + 0x30_0000;
/// The copy engine's records (2 per piece).
const SYNC: u64 = VA + 0x31_8000;
const PUSH: u64 = VA + 0x32_0000;
const PROGS: u64 = VA + 0x40_0000;
const HEAP: u64 = VA + 0x100_0000;
/// "Host RAM": far from the arena (on the board it is the OS's DMA memory mapped into our VAs).
const RAM: u64 = VA + 0x4000_0000;

fn prog_va(p: Prog) -> u64 {
    PROGS + 0x1_0000 * p.index() as u64
}

fn needs(s: nvidia_ga104::qmd::SharedMemory) -> Needs {
    Needs {
        shared_bytes: s.bytes,
        barriers: u32::from(s.barriers),
    }
}

fn board() -> CommandProcessor {
    ts::check_programs().expect("SHA-256 programe");
    let mut cp = CommandProcessor::new();
    for p in PROGRAMS_ALL_3 {
        let code = p.code();
        let sh = || needs(p.shared().unwrap());
        match p {
            Prog::MmTrain => cp.register_program_with_needs(code, mm_train_on_cpu, MM_TRAIN_NEEDS),
            Prog::Xent => cp.register_program_with_needs(code, xent_on_cpu, XENT_NEEDS),
            Prog::LnFwd => cp.register_program_with_needs(code, ln_fwd_on_cpu, LN_FWD_NEEDS),
            Prog::LnBack => cp.register_program_with_needs(code, ln_back_on_cpu, LN_BACK_NEEDS),
            Prog::Colsum => cp.register_program(code, colsum_on_cpu),
            Prog::GeluTr => cp.register_program(code, gelu_tr_on_cpu),
            Prog::AttnFwd => cp.register_program(code, attn_fwd_on_cpu),
            Prog::AttnBack => cp.register_program(code, attn_back_on_cpu),
            Prog::EmbFwd => cp.register_program(code, emb_fwd_on_cpu),
            Prog::EmbBack => cp.register_program(code, emb_back_on_cpu),
            Prog::Gnorm => cp.register_program_with_needs(code, gnorm_on_cpu, GNORM_NEEDS),
            Prog::OptStep => cp.register_program(code, opt_step_on_cpu),
            Prog::Vadd => cp.register_program(code, vadd_on_cpu),
            Prog::AttnP => cp.register_program(code, attn_p_on_cpu),
            Prog::AttnFwdTile2 => cp.register_program_with_needs(code, attn_fwd_tile_on_cpu, sh()),
            Prog::AttnBackTile0_2 => {
                cp.register_program_with_needs(code, attn_back_tile0_on_cpu, sh())
            }
            Prog::AttnBackTile2 => {
                cp.register_program_with_needs(code, attn_back_tile_on_cpu, sh())
            }
            Prog::AttnPTile => cp.register_program_with_needs(code, attn_p_tile_on_cpu, sh()),
            // The `_3` programs: the same operations in the same order as the first tiled ones (T188–T190), the
            // same CPU models.
            Prog::AttnFwdTile3 => cp.register_program_with_needs(code, attn_fwd_tile_on_cpu, sh()),
            Prog::AttnBackTile0_3 => {
                cp.register_program_with_needs(code, attn_back_tile0_on_cpu, sh())
            }
            Prog::AttnBackTile3 => {
                cp.register_program_with_needs(code, attn_back_tile_on_cpu, sh())
            }
            Prog::LnBackRes | Prog::MmTrainGelu | Prog::MmTrainTile | Prog::ColsumPart => {
                unreachable!("nu e in PROGRAMS_ALL_3")
            }
        };
        cp.mem.write(prog_va(p), code);
    }
    cp
}

fn run_words(cp: &mut CommandProcessor, w: &[u32]) {
    let bytes: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, w.len() as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
        .unwrap_or_else(|err| panic!("pasul: {err:?}"));
}

fn records(cp: &CommandProcessor, va: u64, n: usize) -> Vec<u8> {
    let mut b = vec![0u8; 16 * n];
    cp.mem.read(va, &mut b).unwrap();
    b
}

// ------------------------------------------------------------------ the two ways of running the step

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    /// Piece 1's launch replaced by piece 0's.
    PieceRepeated,
    /// m and v swapped in the launch of the last layer's piece **at every step**: consistent, so the weights stay the
    /// same bits (found at the first run of T193) — only m, v in RAM show it.
    SwapMv,
    /// The same swap at step 2 only: the weights change.
    SwapMvStep2,
    /// The copy down of piece 0 lost (its RAM m, v restored after the submission).
    NoCopyDown,
}

#[derive(Default)]
struct Run {
    losses: Vec<u32>,
    norms: Vec<u32>,
    after1: Vec<u32>,
    after_last: Vec<u32>,
    m: Vec<u32>,
    v: Vec<u32>,
    per_step: usize,
    board_bytes: u64,
    events: Vec<Event>,
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn mean_bits(cp: &CommandProcessor, plan: &Plan) -> u32 {
    let rows = cp
        .mem
        .read_f32s(plan.loss_rows, plan.batch * plan.len)
        .unwrap();
    ts::Outcome::mean_loss(&rows).to_bits()
}

/// Today's tiled plan (m, v on the board), `adamw_513`, `steps` steps.
fn run_board(plan: &Plan, params: &[Vec<f32>], data: (&[u32], &[u32]), steps: u32) -> Run {
    let mut cp = board();
    for (va, b) in plan.uploads(params, data.0, data.1).unwrap() {
        cp.mem.write(va, &b);
    }
    let opt = AdamW::recipe_513(params);
    let mut r = Run::default();
    let mut first = 0x5000u32;
    for step in 1..=steps {
        let list = plan.step(Recipe::AdamW513, &opt, step).unwrap();
        r.per_step = list.len();
        let (zone, qmds) = ts::pack(&list, prog_va, ZONE).unwrap();
        cp.mem.write(ZONE, &zone);
        let mut w = vec![0u32; 2 + transfer::sequence_words_fresh(list.len())];
        let mut pb = PushBuf::new(&mut w);
        pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
            .unwrap();
        transfer::encode_sequence_fresh(&mut pb, &qmds, REC, first).unwrap();
        let nw = pb.len();
        run_words(&mut cp, &w[..nw]);
        transfer::analyze_sequence(&records(&cp, REC, list.len() + 1), list.len(), first).unwrap();
        first = first.wrapping_add(list.len() as u32 + 1);
        r.losses.push(mean_bits(&cp, plan));
        r.norms.push(cp.mem.read_u32(plan.res).unwrap());
        if step == 1 {
            r.after1 = bits(&cp.mem.read_f32s(plan.w, plan.n_params).unwrap());
        }
    }
    r.after_last = bits(&cp.mem.read_f32s(plan.w, plan.n_params).unwrap());
    r.m = bits(&cp.mem.read_f32s(plan.m, plan.n_params).unwrap());
    r.v = bits(&cp.mem.read_f32s(plan.v, plan.n_params).unwrap());
    r.board_bytes = plan.bytes();
    r
}

/// The `_mv_ram` plan with `adam_ram`: one submission per step (the launches before the optimizer, then the pieces).
fn run_ram(
    plan: &Plan,
    params: &[Vec<f32>],
    data: (&[u32], &[u32]),
    steps: u32,
    fault: Fault,
) -> Run {
    assert!(plan.mv_in_ram());
    let stage = plan.end.div_ceil(0x1000) * 0x1000;
    let rm = RamMv::new(plan, RAM, stage, SYNC).unwrap();
    let mut cp = board();
    for (va, b) in plan.uploads(params, data.0, data.1).unwrap() {
        cp.mem.write(va, &b);
    }
    for (va, b) in rm.uploads() {
        cp.mem.write(va, &b);
    }
    let opt = AdamW::recipe_513(params);
    let mut r = Run::default();
    let (mut first, mut sync) = (0x5000u32, 0x9000u32);
    let pieces = rm.chunks.len();
    for step in 1..=steps {
        let (before, a) = plan.step_until_opt(&opt, step).unwrap();
        let mut opt_l = rm.opt_launches(&a).unwrap();
        match fault {
            Fault::PieceRepeated => opt_l[1] = opt_l[0].clone(),
            Fault::SwapMv | Fault::SwapMvStep2 if fault == Fault::SwapMv || step == 2 => {
                let l = &mut opt_l[pieces - 2];
                let (m, v) = (l.cbuf[0x40..0x48].to_vec(), l.cbuf[0x48..0x50].to_vec());
                l.cbuf[0x40..0x48].copy_from_slice(&v);
                l.cbuf[0x48..0x50].copy_from_slice(&m);
            }
            _ => {}
        }
        let all: Vec<Launch> = before.iter().chain(&opt_l).cloned().collect();
        r.per_step = all.len();
        let (zone, qmds) = ts::pack(&all, prog_va, ZONE).unwrap();
        assert!(zone.len() as u64 <= REC - ZONE);
        cp.mem.write(ZONE, &zone);
        let n = before.len();
        let mut w = vec![0u32; 2 + ar::step_words(n, pieces)];
        let mut pb = PushBuf::new(&mut w);
        pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
            .unwrap();
        ar::encode_step(&mut pb, &rm, &qmds[..n], &qmds[n..], REC, first, sync).unwrap();
        let nw = pb.len();
        assert_eq!(nw, w.len(), "step_words");
        let keep = (fault == Fault::NoCopyDown).then(|| {
            let c0 = rm.chunks[0];
            let at = 4 * c0.off as u64;
            (
                cp.mem.read_f32s(rm.m + at, c0.n).unwrap(),
                cp.mem.read_f32s(rm.v + at, c0.n).unwrap(),
            )
        });
        let ev0 = cp.events.len();
        run_words(&mut cp, &w[..nw]);
        if let Some((m0, v0)) = keep {
            let at = 4 * rm.chunks[0].off as u64;
            cp.mem.write_f32s(rm.m + at, &m0);
            cp.mem.write_f32s(rm.v + at, &v0);
        }
        transfer::analyze_sequence(&records(&cp, REC, n + pieces + 1), n + pieces, first).unwrap();
        ar::check_sync(&records(&cp, SYNC, 2 * pieces), pieces, sync).unwrap();
        if step == 1 {
            r.events = cp.events[ev0..].to_vec();
        }
        first = first.wrapping_add((n + pieces) as u32 + 1);
        sync = sync.wrapping_add(2 * pieces as u32);
        r.losses.push(mean_bits(&cp, plan));
        r.norms.push(cp.mem.read_u32(plan.res).unwrap());
        if step == 1 {
            r.after1 = bits(&cp.mem.read_f32s(plan.w, plan.n_params).unwrap());
        }
    }
    r.after_last = bits(&cp.mem.read_f32s(plan.w, plan.n_params).unwrap());
    r.m = bits(&cp.mem.read_f32s(rm.m, plan.n_params).unwrap());
    r.v = bits(&cp.mem.read_f32s(rm.v, plan.n_params).unwrap());
    r.board_bytes = plan.bytes() + rm.stage_bytes();
    r
}

fn same(a: &Run, b: &Run) -> bool {
    a.losses == b.losses
        && a.norms == b.norms
        && a.after1 == b.after1
        && a.after_last == b.after_last
        && a.m == b.m
        && a.v == b.v
}

// ------------------------------------------------------------------ data

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../gpu-compute/tests/date/pas_de_aur_mic")
}

fn read_bytes4(p: &Path) -> Vec<[u8; 4]> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    b.as_chunks::<4>().0.to_vec()
}

fn golden() -> (Vec<Vec<f32>>, Vec<u32>, Vec<u32>) {
    let i32s = |name: &str| {
        read_bytes4(&dir().join(name))
            .iter()
            .map(|c| u32::try_from(i32::from_le_bytes(*c)).unwrap())
            .collect::<Vec<u32>>()
    };
    let params = MIC
        .tensors()
        .iter()
        .map(|t| {
            let v: Vec<f32> = read_bytes4(&dir().join("initiale").join(format!("{}.f32", t.name)))
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            assert_eq!(v.len(), t.elements());
            v
        })
        .collect();
    (params, i32s("intrari.i32"), i32s("tinte.i32"))
}

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn made_up(arch: Arch, seed: u64) -> Vec<Vec<f32>> {
    let mut r = rng(seed);
    arch.tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with(".weight") && s.shape.len() == 1 {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.3 * r()).collect()
            }
        })
        .collect()
}

// ------------------------------------------------------------------ the tests

/// Predictions 1 and 2: the small golden step, 5 steps of `adamw_513`, both tiled plans: m, v in RAM = m, v on the
/// board, bit for bit (losses, norms, weights after step 1 and after the last, m and v); L + 2 optimizer launches.
#[test]
fn t193_small_golden_step_bit_for_bit() {
    let (params, inp, tgt) = golden();
    let (bsz, tl) = (2, 32);
    for only in [false, true] {
        let (old, new) = if only {
            (
                Plan::new_p_only_tiles(MIC, bsz, tl, &inp, HEAP).unwrap(),
                Plan::new_p_only_tiles_mv_ram(MIC, bsz, tl, &inp, HEAP).unwrap(),
            )
        } else {
            (
                Plan::new_tiles(MIC, bsz, tl, &inp, HEAP).unwrap(),
                Plan::new_tiles_mv_ram(MIC, bsz, tl, &inp, HEAP).unwrap(),
            )
        };
        assert_eq!(new.programs(), old.programs());
        let a = run_board(&old, &params, (&inp, &tgt), 5);
        let b = run_ram(&new, &params, (&inp, &tgt), 5, Fault::None);
        let ok = same(&a, &b);
        println!(
            "kt c157 adam-ram mic {}: lansari pe pas {} -> {} (L + 2 bucati), placa {} -> {} octeti, bit cu bit {} (SIMULARE)",
            if only { "P refacut" } else { "P pastrat" },
            a.per_step,
            b.per_step,
            a.board_bytes,
            b.board_bytes,
            if ok { "DA" } else { "NU" }
        );
        assert_eq!(b.per_step, a.per_step - 1 + MIC.layers + 2);
        assert!(ok, "m, v in RAM != m, v pe placa");
        // m, v really moved: nonzero after the steps.
        assert!(b.m.iter().any(|&x| x != 0) && b.v.iter().any(|&x| x != 0));
    }
}

/// Prediction 1 on four layers with made-up weights.
#[test]
fn t193_four_layers_bit_for_bit() {
    let arch = Arch { layers: 4, ..MIC };
    let params = made_up(arch, 193);
    let (bsz, tl) = (2, 32);
    let toks: Vec<u32> = (0..=(bsz * tl) as u32)
        .map(|i| (i * 37 + 11) % 256)
        .collect();
    let (inp, tgt) = (&toks[..bsz * tl], &toks[1..]);
    let old = Plan::new_tiles(arch, bsz, tl, inp, HEAP).unwrap();
    let new = Plan::new_tiles_mv_ram(arch, bsz, tl, inp, HEAP).unwrap();
    let a = run_board(&old, &params, (inp, tgt), 3);
    let b = run_ram(&new, &params, (inp, tgt), 3, Fault::None);
    assert_eq!(b.per_step, a.per_step - 1 + 6);
    assert!(same(&a, &b), "4 straturi: m, v in RAM != m, v pe placa");
}

/// The order the simulated board saw, per piece: m up, v up, the copy engine's record, the channel's wait on it, the
/// launch, its record, m down, v down, the record; at the end the wait for the last copy down.
#[test]
fn t193_order_of_one_step() {
    let (params, inp, tgt) = golden();
    let plan = Plan::new_tiles_mv_ram(MIC, 2, 32, &inp, HEAP).unwrap();
    let rm = RamMv::new(&plan, RAM, plan.end.div_ceil(0x1000) * 0x1000, SYNC).unwrap();
    let r = run_ram(&plan, &params, (&inp, &tgt), 1, Fault::None);
    let opt = AdamW::recipe_513(&params);
    let n = plan.step_until_opt(&opt, 1).unwrap().0.len();
    let mut ev = r.events.iter();
    let mut launches = 0;
    for e in ev.by_ref() {
        if matches!(e, Event::Launch { .. }) {
            launches += 1;
            if launches == n {
                break;
            }
        }
    }
    // the record of launch n, then the copy object
    assert!(matches!(ev.next(), Some(Event::Release { .. })));
    assert!(matches!(ev.next(), Some(Event::SetObject { subch: 4, .. })));
    for (k, ch) in rm.chunks.iter().enumerate() {
        let (off, bytes) = (4 * ch.off as u64, 4 * ch.n as u64);
        let want_copy = |e: Option<&Event>, src: u64, dst: u64| {
            assert_eq!(
                e,
                Some(&Event::CeCopy {
                    src,
                    dst,
                    bytes,
                    pipelined: false
                }),
                "bucata {k}"
            )
        };
        want_copy(ev.next(), rm.m + off, rm.stage_m);
        want_copy(ev.next(), rm.v + off, rm.stage_v);
        let up = rm.record(k, false);
        assert!(matches!(ev.next(), Some(Event::Release { addr, .. }) if *addr == up));
        assert!(matches!(ev.next(), Some(Event::Acquire { addr, .. }) if *addr == up));
        assert!(matches!(ev.next(), Some(Event::Launch { .. })));
        assert!(matches!(ev.next(), Some(Event::Release { .. })));
        want_copy(ev.next(), rm.stage_m, rm.m + off);
        want_copy(ev.next(), rm.stage_v, rm.v + off);
        let down = rm.record(k, true);
        assert!(matches!(ev.next(), Some(Event::Release { addr, .. }) if *addr == down));
    }
    let last = rm.record(rm.chunks.len() - 1, true);
    assert!(matches!(ev.next(), Some(Event::Acquire { addr, .. }) if *addr == last));
    assert!(ev.next().is_none());
}

/// Prediction 3: the planted mistakes change the numbers.
#[test]
fn t193_planted_mistakes_are_caught() {
    let (params, inp, tgt) = golden();
    let new = Plan::new_tiles_mv_ram(MIC, 2, 32, &inp, HEAP).unwrap();
    let good = run_ram(&new, &params, (&inp, &tgt), 2, Fault::None);
    for f in [
        Fault::PieceRepeated,
        Fault::SwapMv,
        Fault::SwapMvStep2,
        Fault::NoCopyDown,
    ] {
        let bad = run_ram(&new, &params, (&inp, &tgt), 2, f);
        assert!(!same(&good, &bad), "{f:?} nu a fost prinsa");
        if f == Fault::SwapMv {
            // Consistent at every step: same weights, m and v exchanged in RAM.
            assert_eq!(good.after_last, bad.after_last);
            let ch = ar::chunks(&new).unwrap();
            let c = ch[ch.len() - 2];
            let r = c.off..c.off + c.n;
            assert_eq!(good.m[r.clone()], bad.v[r.clone()]);
            assert_eq!(good.v[r.clone()], bad.m[r.clone()]);
            assert_eq!(good.m[..c.off], bad.m[..c.off]);
        } else {
            assert_ne!(good.after_last, bad.after_last, "{f:?}: greutatile");
        }
    }
}

/// The pieces, the refusals and the words.
#[test]
fn t193_pieces_and_refusals() {
    let (params, inp, tgt) = golden();
    let old = Plan::new_tiles(MIC, 2, 32, &inp, HEAP).unwrap();
    let new = Plan::new_tiles_mv_ram(MIC, 2, 32, &inp, HEAP).unwrap();
    let ch = ar::chunks(&new).unwrap();
    assert_eq!(ch.len(), MIC.layers + 2);
    assert_eq!(ch.iter().map(|c| c.n).sum::<usize>(), new.n_params);
    assert!(ch.windows(2).all(|w| w[0].off + w[0].n == w[1].off));
    assert_eq!(ch[1].layer, Some(0));
    assert_eq!((ch[0].layer, ch.last().unwrap().layer), (None, None));
    // m, v gone from the arena: 8 bytes a parameter (+ the two gaps).
    let saved = old.bytes() - new.bytes();
    let want = 2 * ((4 * new.n_params as u64).div_ceil(256) * 256 + 256);
    assert_eq!(saved, want);
    assert_eq!((new.m, new.v), (0, 0));
    assert_eq!(
        new.uploads(&params, &inp, &tgt).unwrap().len(),
        old.uploads(&params, &inp, &tgt).unwrap().len() - 2
    );
    let opt = AdamW::recipe_513(&params);
    assert!(new.step(Recipe::AdamW513, &opt, 1).is_err());
    assert!(new.step(Recipe::Sgd, &opt, 1).is_ok());
    assert!(new.step_mixed(Recipe::AdamW513, &opt, 1, 1024.0).is_err());
    assert!(new.step_until_opt(&opt, 0).is_err());
    // On a plan with m, v on the board, `step_until_opt` = `step` without its last launch.
    let (before, a) = old.step_until_opt(&opt, 3).unwrap();
    let full = old.step(Recipe::AdamW513, &opt, 3).unwrap();
    assert_eq!(before[..], full[..full.len() - 1]);
    assert_eq!(full.last().unwrap().tag, "opt:pas");
    assert_eq!((a.m, a.v, a.n as usize), (0, 0, old.n_params));
    assert!(RamMv::new(&old, RAM, HEAP + 0x1000_0000, SYNC).is_err());
    assert!(RamMv::new(&new, RAM + 8, HEAP + 0x1000_0000, SYNC).is_err());
    assert!(RamMv::new(&new, RAM, HEAP + 0x1000_0000, SYNC + 4).is_err());
    let rm = RamMv::new(&new, RAM, HEAP + 0x1000_0000, SYNC).unwrap();
    let max = ch.iter().map(|c| c.n).max().unwrap() as u64;
    assert_eq!(rm.stage_bytes(), 2 * (4 * max).div_ceil(256) * 256);
    let l = rm.opt_launches(&a).unwrap();
    assert!(
        l.iter()
            .all(|x| x.tag == ar::TAG && x.prog == Prog::OptStep)
    );
    let mut sgd = a;
    sgd.mode = nvidia_ga104::opt_step::Mode::Sgd;
    assert!(rm.opt_launches(&sgd).is_err());
    // A missing copy record is seen.
    let mut rec = vec![0u8; 16 * 2 * ch.len()];
    for k in 0..ch.len() {
        for d in [false, true] {
            let at = 16 * (2 * k + d as usize);
            rec[at..at + 4].copy_from_slice(&ar::sync_payload(7, k, d).to_le_bytes());
        }
    }
    assert!(ar::check_sync(&rec, ch.len(), 7).is_ok());
    rec[16 * 3] ^= 1;
    assert!(ar::check_sync(&rec, ch.len(), 7).is_err());
}

/// Prediction 4: what the board saves at #513 and #651 (12 / 24 layers × 768, one text of 512).
#[test]
fn t193_board_memory_at_513_and_651() {
    let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
    for (name, layers, want) in [("#513", 12usize, 0.63f64), ("#651", 24, 1.26)] {
        let arch = Arch {
            layers,
            ..train::M513
        };
        let tl = 512;
        let inputs: Vec<u32> = (0..tl as u32).map(|i| (i * 7919) % 16_384).collect();
        let old = Plan::new_tiles(arch, 1, tl, &inputs, HEAP).unwrap();
        let new = Plan::new_tiles_mv_ram(arch, 1, tl, &inputs, HEAP).unwrap();
        let rm = RamMv::new(&new, RAM, HEAP + 0x1_0000_0000, SYNC).unwrap();
        let saved = gib(old.bytes() - new.bytes() - rm.stage_bytes());
        println!(
            "kt c157 adam-ram {name}: placa {:.2} -> {:.2} GiB (+ locul {:.3} GiB), economie {saved:.2} GiB (prezis {want}), RAM {:.2} GiB",
            gib(old.bytes()),
            gib(new.bytes()),
            gib(rm.stage_bytes()),
            gib(rm.ram_bytes())
        );
        assert!((saved / want - 1.0).abs() <= 0.03, "{name}: {saved}");
    }
}

// ------------------------------------------------------------------ T195: the `_3` plans with m, v in RAM

/// T195 (`docs/C-157_P2_ATTN3_IN_PAS.md`): the `_3` plans with m, v in RAM — 5 steps of `adamw_513` on the small
/// golden step, bit for bit = the `_3` plan with m, v on the board = the `_2` plan (SIMULARE: the `_3` programs run on
/// the same CPU models as `_2`, so this checks the wiring, not the board's arithmetic).
#[test]
fn t195_tiles3_mv_ram_bit_for_bit() {
    let (params, inp, tgt) = golden();
    let (bsz, tl) = (2, 32);
    for only in [false, true] {
        let (old2, board3, ram3) = if only {
            (
                Plan::new_p_only_tiles(MIC, bsz, tl, &inp, HEAP).unwrap(),
                Plan::new_p_only_tiles3(MIC, bsz, tl, &inp, HEAP).unwrap(),
                Plan::new_p_only_tiles3_mv_ram(MIC, bsz, tl, &inp, HEAP).unwrap(),
            )
        } else {
            (
                Plan::new_tiles(MIC, bsz, tl, &inp, HEAP).unwrap(),
                Plan::new_tiles3(MIC, bsz, tl, &inp, HEAP).unwrap(),
                Plan::new_tiles3_mv_ram(MIC, bsz, tl, &inp, HEAP).unwrap(),
            )
        };
        assert!(ram3.tiles3() && ram3.mv_in_ram() && board3.tiles3() && !old2.tiles3());
        assert_eq!(ram3.programs(), board3.programs());
        let a = run_board(&old2, &params, (&inp, &tgt), 5);
        let b = run_board(&board3, &params, (&inp, &tgt), 5);
        let c = run_ram(&ram3, &params, (&inp, &tgt), 5, Fault::None);
        let (ab, bc) = (same(&a, &b), same(&b, &c));
        println!(
            "kt c195 adam-ram _3 {}: _3 = _2 bit cu bit {}, _3 m,v in RAM = _3 m,v pe placa bit cu bit {} (SIMULARE)",
            if only { "P refacut" } else { "P pastrat" },
            if ab { "DA" } else { "NU" },
            if bc { "DA" } else { "NU" },
        );
        assert!(ab, "_3 != _2");
        assert!(bc, "_3: m, v in RAM != pe placa");
    }
}
