//! C-160 point 5 — T165, SIMULARE: the small golden step **end to end on the simulated board**.
//!
//! The whole training step of `pas_de_aur_mic` (2 layers, C 64, 4 heads, vocab 256, batch 2 × 32) runs as a list of
//! launches of the 10 training programs (+ `vadd`, the generation's residual addition, which already ran on the
//! RTX 3070 Ti), each through our pushbuffer + QMD on the command processor, every intermediate tensor in (simulated)
//! GPU memory: embeddings → 2 × (LayerNorm, in_proj, attention, out_proj, +, LayerNorm, fc, GELU, proj, +) → final
//! LayerNorm → tied head → loss + its gradient → the same backward → embedding gradients → norm + clipping →
//! SGD / AdamW. The CPU only uploads the batch and the initial weights once, builds the segment lists of `emb_back`
//! (integers, from the tokens) and reads back what we compare: the per-row losses (summed on the CPU only for the
//! report), the gradients of step 1 and the weights after steps 1 and 10 — against **PyTorch** (`reteta.json` and the
//! hub's tensors), in the hub's tolerances, and against the CPU oracle (`gpu_compute::train`, T151).
//!
//! The "programs" run on the CPU (`gpu_sim::cmdproc::*_on_cpu` → `gpu_compute::layer`), in the order of the code NAK
//! made, the approximate units modelled correctly rounded. Nothing here runs on the RTX 3070 Ti.

use std::path::{Path, PathBuf};

use gpu_compute::layer::{EmbSegments, OptScalars};
use gpu_compute::train::{self, AdamW, Arch, Batch, Recipe};
use gpu_sim::cmdproc::{
    CommandProcessor, Event, GNORM_NEEDS, LN_BACK_NEEDS, LN_FWD_NEEDS, MM_TRAIN_NEEDS, XENT_NEEDS,
    attn_back_on_cpu, attn_fwd_on_cpu, colsum_on_cpu, emb_back_on_cpu, emb_fwd_on_cpu,
    gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_fwd_on_cpu, mm_train_on_cpu, opt_step_on_cpu,
    vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_ga104::{
    attn_back as ab, attn_fwd as af, colsum as cs, emb_back as eb, emb_fwd as ef, gelu_tr as gt,
    gnorm as gn, kronos_layer, ln_back as lb, ln_fwd as lf, mm_train as mm, opt_step as os,
    xent as xe,
};
use nvidia_rm570::regs::clc7c0 as c;

/// The small golden model (hub `integrare/kronos/pas_de_aur_mic/reteta.json`).
const MIC: Arch = train::MIC;
const B: usize = 2;
const T: usize = 32;

const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
/// First program; one 64 KiB slot each.
const PROGS: u64 = 0x40_0000;
/// The bump allocator's start (tensors, 256-byte aligned).
const HEAP: u64 = 0x100_0000;

// ------------------------------------------------------------------ the simulated board and its launches

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum P {
    MmTrain,
    Xent,
    LnFwd,
    LnBack,
    Colsum,
    GeluTr,
    AttnFwd,
    AttnBack,
    EmbFwd,
    EmbBack,
    Gnorm,
    OptStep,
    Vadd,
}

/// (program, bytes, registers, threads per block, shared memory).
type ProgRow = (P, &'static [u8], u32, [u32; 3], Option<SharedMemory>);

fn table() -> Vec<ProgRow> {
    vec![
        (
            P::MmTrain,
            mm::MM_TRAIN,
            mm::REGISTERS,
            mm::BLOCK,
            Some(mm::SHARED),
        ),
        (
            P::Xent,
            xe::XENT,
            xe::REGISTERS,
            xe::BLOCK,
            Some(xe::SHARED),
        ),
        (
            P::LnFwd,
            lf::LN_FWD,
            lf::REGISTERS,
            lf::BLOCK,
            Some(lf::SHARED),
        ),
        (
            P::LnBack,
            lb::LN_BACK,
            lb::REGISTERS,
            lb::BLOCK,
            Some(lb::SHARED),
        ),
        (P::Colsum, cs::COLSUM, cs::REGISTERS, cs::BLOCK, None),
        (P::GeluTr, gt::GELU_TR, gt::REGISTERS, gt::BLOCK, None),
        (P::AttnFwd, af::ATTN_FWD, af::REGISTERS, af::BLOCK, None),
        (P::AttnBack, ab::ATTN_BACK, ab::REGISTERS, ab::BLOCK, None),
        (P::EmbFwd, ef::EMB_FWD, ef::REGISTERS, ef::BLOCK, None),
        (P::EmbBack, eb::EMB_BACK, eb::REGISTERS, eb::BLOCK, None),
        (
            P::Gnorm,
            gn::GNORM,
            gn::REGISTERS,
            gn::BLOCK,
            Some(gn::SHARED),
        ),
        (P::OptStep, os::OPT_STEP, os::REGISTERS, os::BLOCK, None),
        (P::Vadd, kronos_layer::VADD, 19, [256, 1, 1], None),
    ]
}

struct Board {
    cp: CommandProcessor,
    next: u64,
    /// Every launch, in order (program, grid).
    log: Vec<(P, [u32; 3])>,
}

impl Board {
    fn new() -> Self {
        let mut cp = CommandProcessor::new();
        for (i, (p, code, ..)) in table().into_iter().enumerate() {
            match p {
                P::MmTrain => cp.register_program_with_needs(code, mm_train_on_cpu, MM_TRAIN_NEEDS),
                P::Xent => cp.register_program_with_needs(code, xent_on_cpu, XENT_NEEDS),
                P::LnFwd => cp.register_program_with_needs(code, ln_fwd_on_cpu, LN_FWD_NEEDS),
                P::LnBack => cp.register_program_with_needs(code, ln_back_on_cpu, LN_BACK_NEEDS),
                P::Colsum => cp.register_program(code, colsum_on_cpu),
                P::GeluTr => cp.register_program(code, gelu_tr_on_cpu),
                P::AttnFwd => cp.register_program(code, attn_fwd_on_cpu),
                P::AttnBack => cp.register_program(code, attn_back_on_cpu),
                P::EmbFwd => cp.register_program(code, emb_fwd_on_cpu),
                P::EmbBack => cp.register_program(code, emb_back_on_cpu),
                P::Gnorm => cp.register_program_with_needs(code, gnorm_on_cpu, GNORM_NEEDS),
                P::OptStep => cp.register_program(code, opt_step_on_cpu),
                P::Vadd => cp.register_program(code, vadd_on_cpu),
            };
            cp.mem.write(PROGS + 0x1_0000 * i as u64, code);
        }
        Board {
            cp,
            next: HEAP,
            log: Vec::new(),
        }
    }

    /// `n` f32 (or u32) of GPU memory, 256-byte aligned.
    fn alloc(&mut self, n: usize) -> u64 {
        let va = self.next;
        self.next += (4 * n as u64).div_ceil(256) * 256 + 256;
        va
    }

    fn upload(&mut self, v: &[f32]) -> u64 {
        let va = self.alloc(v.len());
        self.cp.mem.write_f32s(va, v);
        va
    }

    fn upload_u32(&mut self, v: &[u32]) -> u64 {
        let va = self.alloc(v.len());
        let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        self.cp.mem.write(va, &b);
        va
    }

    fn read(&self, va: u64, n: usize) -> Vec<f32> {
        self.cp.mem.read_f32s(va, n).unwrap()
    }

    fn launch(&mut self, p: P, cb: Option<&[u8]>, grid: [u32; 3]) {
        let cb = cb.unwrap_or_else(|| panic!("{p:?}: argumente refuzate"));
        let (i, (_, _, registers, block, shared)) = table()
            .into_iter()
            .enumerate()
            .find(|(_, t)| t.0 == p)
            .unwrap();
        let mut buf = vec![0u8; 0x100];
        buf[..cb.len()].copy_from_slice(cb);
        self.cp.mem.write(CBUF, &buf);
        let q = ComputeQmd {
            program_va: PROGS + 0x1_0000 * i as u64,
            registers,
            cbuf0_va: CBUF,
            cbuf0_size: 0x100,
            grid,
            block,
            release: Some((SEM, 7)),
        };
        let bytes = match shared {
            Some(s) => q.to_bytes_with_shared(&s).unwrap(),
            None => q.to_bytes().unwrap(),
        };
        self.cp.mem.write(QMD, &bytes);
        self.cp.mem.write_u32(SEM, 0);
        let mut w = [0u32; 16];
        let mut pb = PushBuf::new(&mut w);
        pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
            .unwrap();
        pb.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
            .unwrap();
        let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
        pb.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
            .unwrap();
        let n = pb.len();
        let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
        self.cp.mem.write(PUSH, &bytes);
        let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
        self.cp
            .run_gpfifo(&[(e.entry0, e.entry1)])
            .unwrap_or_else(|err| panic!("{p:?}: {err:?}"));
        assert_eq!(self.cp.mem.read_u32(SEM).unwrap(), 7, "{p:?}: semaforul");
        self.log.push((p, grid));
    }

    // -------------------------------------------------------------- one launch per call

    /// `c = op(a)·op(b) (+ bias)`, `op(a)` m × k, `op(b)` k × n.
    #[allow(clippy::too_many_arguments)]
    fn mm(&mut self, a: u64, b: u64, cc: u64, bias: u64, m: usize, n: usize, k: usize, flags: u32) {
        let x = mm::Args {
            a,
            b,
            c: cc,
            bias,
            m: m as u32,
            n: n as u32,
            k: k as u32,
            flags: flags | if bias != 0 { mm::BIAS } else { 0 },
        };
        self.launch(P::MmTrain, x.cbuf().as_ref().map(|b| &b[..]), x.grid());
    }

    fn vadd(&mut self, a: u64, b: u64, out: u64, n: usize) {
        let mut cb = [0u8; 0x44];
        for (i, va) in [a, b, out].iter().enumerate() {
            cb[0x28 + 8 * i..0x30 + 8 * i].copy_from_slice(&va.to_le_bytes());
        }
        cb[0x40..0x44].copy_from_slice(&(n as u32).to_le_bytes());
        self.launch(P::Vadd, Some(&cb), [(n as u32).div_ceil(256), 1, 1]);
    }

    #[allow(clippy::too_many_arguments)]
    fn ln_fwd(
        &mut self,
        x: u64,
        gamma: u64,
        beta: u64,
        y: u64,
        mean: u64,
        rstd: u64,
        rows: usize,
        cc: usize,
    ) {
        let a = lf::Args {
            x,
            gamma,
            beta,
            y,
            mean,
            rstd,
            rows: rows as u32,
            c: cc as u32,
            eps: train::LN_EPS as f32,
        };
        self.launch(P::LnFwd, a.cbuf().as_ref().map(|b| &b[..]), a.grid());
    }

    /// LayerNorm backward: dx (`ln_back`), then dγ, dβ (`colsum`, LayerNorm mode).
    #[allow(clippy::too_many_arguments)]
    fn ln_back(
        &mut self,
        s: &LnSaved,
        dy: u64,
        gamma: u64,
        dx: u64,
        dg: u64,
        db: u64,
        rows: usize,
        cc: usize,
    ) {
        let a = lb::Args {
            x: s.x,
            dy,
            gamma,
            mean: s.mean,
            rstd: s.rstd,
            dx,
            rows: rows as u32,
            c: cc as u32,
        };
        self.launch(P::LnBack, a.cbuf().as_ref().map(|b| &b[..]), a.grid());
        let k = cs::Args {
            dy,
            x: s.x,
            mean: s.mean,
            rstd: s.rstd,
            s0: dg,
            s1: db,
            rows: rows as u32,
            c: cc as u32,
            mode: cs::Mode::LayerNorm,
        };
        self.launch(P::Colsum, k.cbuf().as_ref().map(|b| &b[..]), k.grid());
    }

    fn bias_grad(&mut self, dy: u64, db: u64, rows: usize, cc: usize) {
        let k = cs::Args {
            dy,
            x: 0,
            mean: 0,
            rstd: 0,
            s0: db,
            s1: 0,
            rows: rows as u32,
            c: cc as u32,
            mode: cs::Mode::Bias,
        };
        self.launch(P::Colsum, k.cbuf().as_ref().map(|b| &b[..]), k.grid());
    }

    fn gelu(&mut self, u: u64, g: u64, y: u64, n: usize, mode: gt::Mode) {
        let a = gt::Args {
            u,
            g,
            y,
            n: n as u32,
            mode,
        };
        self.launch(P::GeluTr, a.cbuf().as_ref().map(|b| &b[..]), a.grid());
    }
}

/// What one LayerNorm keeps for the backward pass: its input, μ and rstd (on the board).
#[derive(Clone, Copy)]
struct LnSaved {
    x: u64,
    mean: u64,
    rstd: u64,
}

/// What one layer keeps for the backward pass (all on the board).
struct LayerBufs {
    ln1: LnSaved,
    h1: u64,
    qkv: u64,
    p: u64,
    att: u64,
    ln2: LnSaved,
    h2: u64,
    u: u64,
    act: u64,
}

/// The whole step's memory: weights, gradients, AdamW state (flat, `named_parameters()` order) and the activations.
/// A mistake planted in the driver's list, to show the comparison catches it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Fault {
    None,
    /// The residual addition through LN2 forgotten in the backward pass of every layer.
    NoBackResidual,
    /// The loss gradient scaled by 2/N instead of 1/N.
    XentScale,
    /// `adamw_513` without the clipping (factor 1 instead of the board's c).
    NoClip,
    /// The head's dW written after the token scatter instead of under it (the scatter lost).
    HeadOverwrites,
}

struct Step {
    b: Board,
    ar: Arch,
    bsz: usize,
    tl: usize,
    fault: Fault,
    /// Offset (in elements) of every tensor in the flat buffers.
    off: Vec<usize>,
    n_params: usize,
    w: u64,
    g: u64,
    m: u64,
    v: u64,
    tok: u64,
    tgt: u64,
    seg_tok: (u64, u64, u64, usize),
    seg_pos: (u64, u64, u64, usize),
    layers: Vec<LayerBufs>,
    xs: Vec<u64>,
    lnf: LnSaved,
    hf: u64,
    logits: u64,
    dlogits: u64,
    loss_rows: u64,
    // backward scratch
    dx: [u64; 2],
    dtmp: u64,
    dwide: u64,
    dwide2: u64,
    dqkv: u64,
    ds: u64,
    part: u64,
    res: u64,
}

fn segs(b: &mut Board, s: &EmbSegments) -> (u64, u64, u64, usize) {
    (
        b.upload_u32(&s.off),
        b.upload_u32(&s.dst),
        b.upload_u32(&s.rows),
        s.len(),
    )
}

impl Step {
    fn new(
        ar: Arch,
        (bsz, tl): (usize, usize),
        params: &[Vec<f32>],
        inputs: &[u32],
        targets: &[u32],
    ) -> Self {
        let mut b = Board::new();
        let (cc, v) = (ar.n_emb, ar.vocab);
        let n = bsz * tl;
        let f = 4 * cc;
        let mut off = Vec::new();
        let mut at = 0;
        for p in params {
            off.push(at);
            at += p.len();
        }
        let flat: Vec<f32> = params.iter().flatten().copied().collect();
        let w = b.upload(&flat);
        // Zeroed once by the driver: the rows of `emb_pozitie`'s gradient past the text length are never written by
        // `emb_back` (positions ≥ T), and must stay 0 (PyTorch's gradient there).
        let g = b.upload(&vec![0.0; at]);
        let m = b.upload(&vec![0.0; at]);
        let vv = b.upload(&vec![0.0; at]);
        let tok = b.upload_u32(inputs);
        let tgt = b.upload_u32(targets);
        let st = EmbSegments::tokens(inputs, v).unwrap();
        let sp = EmbSegments::positions(bsz, tl).unwrap();
        assert!(st.check(n, v) && sp.check(n, ar.ctx));
        let seg_tok = segs(&mut b, &st);
        let seg_pos = segs(&mut b, &sp);
        let mut xs = vec![b.alloc(n * cc)];
        let mut layers = Vec::new();
        for _ in 0..ar.layers {
            let ln = |b: &mut Board, x: u64| LnSaved {
                x,
                mean: b.alloc(n),
                rstd: b.alloc(n),
            };
            let xin = *xs.last().unwrap();
            let ln1 = ln(&mut b, xin);
            let xmid = b.alloc(n * cc);
            let ln2 = ln(&mut b, xmid);
            layers.push(LayerBufs {
                ln1,
                h1: b.alloc(n * cc),
                qkv: b.alloc(n * 3 * cc),
                p: b.alloc(bsz * ar.heads * tl * tl),
                att: b.alloc(n * cc),
                ln2,
                h2: b.alloc(n * cc),
                u: b.alloc(n * f),
                act: b.alloc(n * f),
            });
            xs.push(xmid);
            xs.push(b.alloc(n * cc));
        }
        let lnf = LnSaved {
            x: *xs.last().unwrap(),
            mean: b.alloc(n),
            rstd: b.alloc(n),
        };
        Step {
            ar,
            bsz,
            tl,
            fault: Fault::None,
            off,
            n_params: at,
            w,
            g,
            m,
            v: vv,
            tok,
            tgt,
            seg_tok,
            seg_pos,
            layers,
            lnf,
            hf: b.alloc(n * cc),
            logits: b.alloc(n * v),
            dlogits: b.alloc(n * v),
            loss_rows: b.alloc(n),
            dx: [b.alloc(n * cc), b.alloc(n * cc)],
            dtmp: b.alloc(n * cc),
            dwide: b.alloc(n * f),
            dwide2: b.alloc(n * f),
            dqkv: b.alloc(n * 3 * cc),
            ds: b.alloc(bsz * ar.heads * tl * tl),
            part: b.alloc(4096),
            res: b.alloc(64),
            xs,
            b,
        }
    }

    /// The board address of tensor `i` of the weights (`base` = w) or the gradients (`base` = g).
    fn t(&self, base: u64, i: usize) -> u64 {
        base + 4 * self.off[i] as u64
    }

    /// Index of tensor `k` of layer `l` (12 per layer, after the two tables).
    fn li(l: usize, k: usize) -> usize {
        2 + train::PER_LAYER * l + k
    }

    /// Forward + loss + backward: every gradient ends up in `g`. Returns the per-row losses (read back for the report).
    fn forward_backward(&mut self) -> Vec<f32> {
        let (ar, bsz, tl) = (self.ar, self.bsz, self.tl);
        let (cc, v, h) = (ar.n_emb, ar.vocab, ar.heads);
        let n = bsz * tl;
        let f = 4 * cc;
        let scale = ab::Args::scale_for((cc / h) as u32);
        let (w, g) = (self.w, self.g);
        let lnf_i = 2 + train::PER_LAYER * ar.layers;

        // ---- forward
        let a = ef::Args {
            tok: self.tok,
            emb: self.t(w, 0),
            pos: self.t(w, 1),
            x: self.xs[0],
            batch: bsz as u32,
            c: cc as u32,
            t: tl as u32,
        };
        self.b
            .launch(P::EmbFwd, a.cbuf().as_ref().map(|b| &b[..]), a.grid());
        for l in 0..ar.layers {
            let wt: [u64; train::PER_LAYER] = std::array::from_fn(|k| self.t(w, Self::li(l, k)));
            let wt = |k: usize| wt[k];
            let (xin, xmid, xout) = (self.xs[2 * l], self.xs[2 * l + 1], self.xs[2 * l + 2]);
            let wl = &self.layers[l];
            let (ln1, h1, qkv, p, att, ln2, h2, u, act) = (
                wl.ln1, wl.h1, wl.qkv, wl.p, wl.att, wl.ln2, wl.h2, wl.u, wl.act,
            );
            let tmp = self.dtmp;
            self.b
                .ln_fwd(xin, wt(0), wt(1), h1, ln1.mean, ln1.rstd, n, cc);
            self.b.mm(h1, wt(2), qkv, wt(3), n, 3 * cc, cc, mm::TB);
            let a = af::Args {
                qkv,
                p,
                att,
                batch: bsz as u32,
                t: tl as u32,
                c: cc as u32,
                heads: h as u32,
                scale,
            };
            self.b
                .launch(P::AttnFwd, a.cbuf().as_ref().map(|b| &b[..]), a.grid());
            self.b.mm(att, wt(4), tmp, wt(5), n, cc, cc, mm::TB);
            self.b.vadd(xin, tmp, xmid, n * cc);
            self.b
                .ln_fwd(xmid, wt(6), wt(7), h2, ln2.mean, ln2.rstd, n, cc);
            self.b.mm(h2, wt(8), u, wt(9), n, f, cc, mm::TB);
            self.b.gelu(u, 0, act, n * f, gt::Mode::Forward);
            self.b.mm(act, wt(10), tmp, wt(11), n, cc, f, mm::TB);
            self.b.vadd(xmid, tmp, xout, n * cc);
        }
        let lnf = self.lnf;
        self.b.ln_fwd(
            lnf.x,
            self.t(w, lnf_i),
            self.t(w, lnf_i + 1),
            self.hf,
            lnf.mean,
            lnf.rstd,
            n,
            cc,
        );
        // The tied head: logits = hf · embᵀ (no bias).
        self.b
            .mm(self.hf, self.t(w, 0), self.logits, 0, n, v, cc, mm::TB);

        // ---- loss + d(logits) = (softmax − onehot) / N
        let a = xe::Args {
            logits: self.logits,
            targets: self.tgt,
            dlogits: self.dlogits,
            loss: self.loss_rows,
            rows: n as u32,
            v: v as u32,
            scale: if self.fault == Fault::XentScale {
                2.0
            } else {
                1.0
            } / n as f32,
        };
        self.b
            .launch(P::Xent, a.cbuf().as_ref().map(|b| &b[..]), a.grid());

        // ---- backward
        // The head's dW = dlogitsᵀ·hf goes straight into emb_token's gradient (emb_back adds the scatter on top).
        self.b
            .mm(self.dlogits, self.hf, self.t(g, 0), 0, v, cc, n, mm::TA);
        self.b
            .mm(self.dlogits, self.t(w, 0), self.dtmp, 0, n, cc, v, 0);
        let cur = 0;
        self.b.ln_back(
            &lnf,
            self.dtmp,
            self.t(w, lnf_i),
            self.dx[cur],
            self.t(g, lnf_i),
            self.t(g, lnf_i + 1),
            n,
            cc,
        );
        for l in (0..ar.layers).rev() {
            let wt: [u64; train::PER_LAYER] = std::array::from_fn(|k| self.t(w, Self::li(l, k)));
            let wt = |k: usize| wt[k];
            let gt_: [u64; train::PER_LAYER] = std::array::from_fn(|k| self.t(g, Self::li(l, k)));
            let gt_ = |k: usize| gt_[k];
            let wl = &self.layers[l];
            let (ln1, h1, qkv, p, att, ln2, h2, u, act) = (
                wl.ln1, wl.h1, wl.qkv, wl.p, wl.att, wl.ln2, wl.h2, wl.u, wl.act,
            );
            let (dx, dtmp, dwide, dwide2, dqkv, ds) = (
                self.dx[cur],
                self.dtmp,
                self.dwide,
                self.dwide2,
                self.dqkv,
                self.ds,
            );
            let dx2 = self.dx[1 - cur];
            // MLP: x = x_mid + proj(gelu(fc(LN2(x_mid))))
            self.b.mm(dx, wt(10), dwide, 0, n, f, cc, 0); // d(act)
            self.b.mm(dx, act, gt_(10), 0, cc, f, n, mm::TA); // dW proj
            self.b.bias_grad(dx, gt_(11), n, cc);
            self.b.gelu(u, dwide, dwide2, n * f, gt::Mode::Backward); // du
            self.b.mm(dwide2, wt(8), dtmp, 0, n, cc, f, 0); // dh2
            self.b.mm(dwide2, h2, gt_(8), 0, f, cc, n, mm::TA); // dW fc
            self.b.bias_grad(dwide2, gt_(9), n, f);
            self.b
                .ln_back(&ln2, dtmp, wt(6), dx2, gt_(6), gt_(7), n, cc);
            if self.fault != Fault::NoBackResidual {
                self.b.vadd(dx, dx2, dx2, n * cc); // dx += d(x_mid) through LN2
            }
            // Attention: x_mid = x_in + out(attn(in(LN1(x_in))))
            self.b.mm(dx2, wt(4), dtmp, 0, n, cc, cc, 0); // d(att)
            self.b.mm(dx2, att, gt_(4), 0, cc, cc, n, mm::TA); // dW out
            self.b.bias_grad(dx2, gt_(5), n, cc);
            for mode in [ab::Mode::Rows, ab::Mode::Columns] {
                let a = ab::Args {
                    qkv,
                    p,
                    dout: dtmp,
                    ds,
                    dqkv,
                    batch: bsz as u32,
                    t: tl as u32,
                    c: cc as u32,
                    heads: h as u32,
                    scale,
                    mode,
                };
                self.b
                    .launch(P::AttnBack, a.cbuf().as_ref().map(|b| &b[..]), a.grid());
            }
            self.b.mm(dqkv, wt(2), dtmp, 0, n, cc, 3 * cc, 0); // dh1
            self.b.mm(dqkv, h1, gt_(2), 0, 3 * cc, cc, n, mm::TA); // dW in
            self.b.bias_grad(dqkv, gt_(3), n, 3 * cc);
            self.b.ln_back(&ln1, dtmp, wt(0), dx, gt_(0), gt_(1), n, cc);
            self.b.vadd(dx2, dx, dx, n * cc); // dx += d(x_in) through LN1
            // dx is back in self.dx[cur].
        }
        // Embeddings: positions (write), then tokens (add onto the head's dW).
        let dx = self.dx[cur];
        for (seg, base, mode) in [
            (self.seg_pos, self.t(g, 1), eb::Mode::Write),
            (self.seg_tok, self.t(g, 0), eb::Mode::Add),
        ] {
            let a = eb::Args {
                dx,
                off: seg.0,
                dst: seg.1,
                rows: seg.2,
                g: base,
                segments: seg.3 as u32,
                c: cc as u32,
                mode,
            };
            self.b
                .launch(P::EmbBack, a.cbuf().as_ref().map(|b| &b[..]), a.grid());
        }
        if self.fault == Fault::HeadOverwrites {
            self.b
                .mm(self.dlogits, self.hf, self.t(g, 0), 0, v, cc, n, mm::TA);
        }
        self.b.read(self.loss_rows, n)
    }

    /// Norm + clipping (`adamw_513`: max 1.0) or the factor 1 (`sgd`), then the optimizer over all parameters.
    fn optimize(&mut self, recipe: Recipe, opt: &AdamW, step: u32) -> Option<f32> {
        let n = self.n_params;
        let (norm, sc, mode) = match recipe {
            Recipe::Sgd => {
                self.b.cp.mem.write_f32s(self.res, &[0.0, 1.0]);
                (None, OptScalars::sgd(0.01), os::Mode::Sgd)
            }
            Recipe::AdamW513 => {
                const CHUNK: u32 = 4_096;
                let blocks = (n as u32).div_ceil(CHUNK);
                let a = gn::Args {
                    x: self.g,
                    part: self.part,
                    res: 0,
                    n: n as u32,
                    chunk: CHUNK,
                    mode: gn::Mode::Partial,
                    max_norm: 1.0,
                };
                self.b
                    .launch(P::Gnorm, a.cbuf().as_ref().map(|b| &b[..]), a.grid());
                let fin = gn::Args {
                    x: self.part,
                    part: 0,
                    res: self.res,
                    n: blocks,
                    chunk: blocks.div_ceil(256) * 256,
                    mode: gn::Mode::Finish,
                    max_norm: 1.0,
                };
                self.b
                    .launch(P::Gnorm, fin.cbuf().as_ref().map(|b| &b[..]), fin.grid());
                if self.fault == Fault::NoClip {
                    self.b.cp.mem.write_f32s(self.res + 8, &[1.0]);
                }
                let r = self.b.read(self.res, 2);
                // The driver stops the step on a non-finite norm (C-160 point 4 note).
                assert!(r[0].is_finite(), "norma nu e finita: pasul se opreste");
                (
                    Some(r[0]),
                    OptScalars::adamw(&opt.scalars(step)),
                    os::Mode::AdamW,
                )
            }
        };
        let a = os::Args {
            g: self.g,
            coef: if self.fault == Fault::NoClip {
                self.res + 8
            } else {
                self.res + 4
            },
            w: self.w,
            m: self.m,
            v: self.v,
            n: n as u32,
            mode,
            scalars: [
                sc.neg_lr,
                sc.decay,
                sc.w1,
                sc.b2,
                sc.omb2,
                sc.bc2_sqrt,
                sc.neg_step,
                sc.eps,
            ],
        };
        self.b
            .launch(P::OptStep, a.cbuf().as_ref().map(|b| &b[..]), a.grid());
        norm
    }

    fn unflat(&self, base: u64) -> Vec<Vec<f32>> {
        self.ar
            .tensors()
            .iter()
            .enumerate()
            .map(|(i, s)| self.b.read(self.t(base, i), s.elements()))
            .collect()
    }
}

// ------------------------------------------------------------------ the golden step's data

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../gpu-compute/tests/date/pas_de_aur_mic")
}

fn read_f32(p: &Path, n: usize) -> Vec<f32> {
    let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    assert_eq!(b.len(), 4 * n, "{}", p.display());
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_le_bytes(*c))
        .collect()
}

fn read_i32(p: &Path) -> Vec<u32> {
    let b = std::fs::read(p).unwrap();
    b.as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::try_from(i32::from_le_bytes(*c)).unwrap())
        .collect()
}

fn tensors(sub: &str) -> Vec<Vec<f32>> {
    MIC.tensors()
        .iter()
        .map(|t| {
            read_f32(
                &dir().join(sub).join(format!("{}.f32", t.name)),
                t.elements(),
            )
        })
        .collect()
}

/// A list of numbers of one recipe in `reteta.json` (`pierderi`, `norme_gradient`), parsed by hand.
fn json_list(recipe: &str, key: &str) -> Vec<f64> {
    let s = std::fs::read_to_string(dir().join("reteta.json")).unwrap();
    let r = &s[s.find(&format!("\"{recipe}\"")).unwrap()..];
    let a = &r[r.find(&format!("\"{key}\"")).unwrap()..];
    let a = &a[a.find('[').unwrap() + 1..a.find(']').unwrap()];
    a.split(',').map(|v| v.trim().parse().unwrap()).collect()
}

struct Dev {
    abs: f32,
    rel_norm: f64,
    where_: String,
}

fn worst(ours: &[Vec<f32>], theirs: &[Vec<f32>]) -> Dev {
    let mut d = Dev {
        abs: 0.0,
        rel_norm: 0.0,
        where_: String::new(),
    };
    for ((o, t), spec) in ours.iter().zip(theirs).zip(MIC.tensors()) {
        let (mut diff2, mut norm2) = (0.0f64, 0.0f64);
        for (&a, &b) in o.iter().zip(t) {
            let e = (a - b).abs();
            assert!(e.is_finite(), "{}: valoare nefinita", spec.name);
            if e > d.abs {
                d.abs = e;
                d.where_ = spec.name.clone();
            }
            diff2 += (e as f64).powi(2);
            norm2 += (b as f64).powi(2);
        }
        if norm2 > 0.0 {
            d.rel_norm = d.rel_norm.max((diff2 / norm2).sqrt());
        }
    }
    d
}

/// Everything one run of 10 steps on the simulated board produced.
struct Run {
    losses: Vec<f32>,
    norms: Vec<f32>,
    grad1: Vec<Vec<f32>>,
    after1: Vec<Vec<f32>>,
    after10: Vec<Vec<f32>>,
    launches_per_step: Vec<(P, usize)>,
    total_per_step: usize,
    cp_launch_events: usize,
    board_bytes: u64,
}

fn run_on_board(recipe: Recipe, steps: u32) -> Run {
    run_with(recipe, steps, Fault::None)
}

fn run_with(recipe: Recipe, steps: u32, fault: Fault) -> Run {
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let params = tensors("initiale");
    let opt = AdamW::recipe_513(&params);
    let mut s = Step::new(MIC, (B, T), &params, &inputs, &targets);
    s.fault = fault;
    let (mut losses, mut norms) = (Vec::new(), Vec::new());
    let (mut grad1, mut after1) = (Vec::new(), Vec::new());
    let mut per_step = Vec::new();
    for step in 1..=steps {
        let before = s.b.log.len();
        let rows = s.forward_backward();
        losses.push((rows.iter().map(|&r| r as f64).sum::<f64>() / rows.len() as f64) as f32);
        if step == 1 {
            grad1 = s.unflat(s.g);
        }
        if let Some(nrm) = s.optimize(recipe, &opt, step) {
            norms.push(nrm);
        }
        if step == 1 {
            after1 = s.unflat(s.w);
            let mut counts: Vec<(P, usize)> = Vec::new();
            for (p, _) in &s.b.log[before..] {
                match counts.iter_mut().find(|(q, _)| q == p) {
                    Some(e) => e.1 += 1,
                    None => counts.push((*p, 1)),
                }
            }
            per_step = counts;
        }
    }
    let total_per_step = per_step.iter().map(|e| e.1).sum();
    let cp_launch_events =
        s.b.cp
            .events
            .iter()
            .filter(|e| matches!(e, Event::Launch { .. }))
            .count();
    Run {
        losses,
        norms,
        grad1,
        after1,
        after10: s.unflat(s.w),
        launches_per_step: per_step,
        total_per_step,
        cp_launch_events,
        board_bytes: s.b.next - HEAP,
    }
}

fn batch_of<'a>(inputs: &'a [u32], targets: &'a [u32]) -> Batch<'a> {
    Batch {
        inputs,
        targets,
        batch: B,
        len: T,
    }
}

/// Judges one run against PyTorch (the hub's tolerances, the same as T151 for the CPU oracle) and the CPU oracle.
fn check(recipe: Recipe, name: &str, w_tol: f32) {
    let r = run_on_board(recipe, 10);
    // Every step is the same list of launches, all through the command processor.
    assert_eq!(r.cp_launch_events, 10 * r.total_per_step);
    let want = json_list(name, "pierderi");
    let mut loss_rel = 0.0f64;
    for (k, (&o, &w)) in r.losses.iter().zip(&want).enumerate() {
        let e = (o as f64 - w).abs() / w.abs();
        loss_rel = loss_rel.max(e);
        assert!(
            e <= train::TOL_LOSS_REL,
            "{name}: eroarea pasului {}: {o} fata de {w}",
            k + 1
        );
    }
    let g = worst(&r.grad1, &tensors(&format!("{name}/grad_pas1")));
    let w1 = worst(&r.after1, &tensors(&format!("{name}/dupa_pas1")));
    let w10 = worst(&r.after10, &tensors(&format!("{name}/dupa_pas10")));

    // The CPU oracle (T151) on the same data.
    let inputs = read_i32(&dir().join("intrari.i32"));
    let targets = read_i32(&dir().join("tinte.i32"));
    let cpu = train::golden_run(
        &MIC,
        &tensors("initiale"),
        &batch_of(&inputs, &targets),
        recipe,
        10,
    )
    .unwrap();
    let og = worst(&r.grad1, &cpu.grad_step1);
    let o1 = worst(&r.after1, &cpu.after_step1);
    let o10 = worst(&r.after10, &cpu.after_last);
    let ol = r
        .losses
        .iter()
        .zip(&cpu.losses)
        .map(|(&a, &b)| ((a - b) as f64).abs() / b as f64)
        .fold(0.0, f64::max);

    let list: Vec<String> = r
        .launches_per_step
        .iter()
        .map(|(p, k)| format!("{p:?} {k}"))
        .collect();
    println!(
        "kt c165 {name} pe placa simulata: {} lansari pe pas ({}), memoria plăcii {} KiB; eroare_rel={loss_rel:.2e} grad_abs={:.2e} ({}) grad_rel_norma={:.2e} w1_abs={:.2e} ({}) w10_abs={:.2e} ({}) eroarea_1={} eroarea_10={}",
        r.total_per_step,
        list.join(", "),
        r.board_bytes / 1024,
        g.abs,
        g.where_,
        g.rel_norm,
        w1.abs,
        w1.where_,
        w10.abs,
        w10.where_,
        r.losses[0],
        r.losses[9]
    );
    println!(
        "kt c165 {name} fata de oracolul pe procesor: eroare_rel={ol:.2e} grad_abs={:.2e} ({}) grad_rel_norma={:.2e} w1_abs={:.2e} w10_abs={:.2e}",
        og.abs, og.where_, og.rel_norm, o1.abs, o10.abs
    );
    if !r.norms.is_empty() {
        let wn = json_list(name, "norme_gradient");
        let nrel = r
            .norms
            .iter()
            .zip(&wn)
            .map(|(&a, &b)| (a as f64 - b).abs() / b)
            .fold(0.0, f64::max);
        println!("kt c165 {name} normele celor 10 pasi fata de PyTorch: rel ≤ {nrel:.2e}");
        assert!(nrel <= 1e-5, "{name}: normele {nrel}");
    }
    assert!(
        g.abs <= train::TOL_GRAD_ABS,
        "{name}: gradient {} in {}",
        g.abs,
        g.where_
    );
    assert!(
        g.rel_norm <= train::TOL_GRAD_REL_NORM,
        "{name}: gradient rel {}",
        g.rel_norm
    );
    assert!(
        w1.abs <= w_tol,
        "{name}: greutati dupa pasul 1: {} in {}",
        w1.abs,
        w1.where_
    );
    assert!(
        w10.abs <= w_tol,
        "{name}: greutati dupa pasul 10: {} in {}",
        w10.abs,
        w10.where_
    );
}

#[test]
fn t165_small_golden_step_sgd_end_to_end_on_the_simulated_board() {
    check(Recipe::Sgd, "sgd", train::TOL_SGD_W_ABS);
}

#[test]
fn t165_small_golden_step_adamw_513_end_to_end_on_the_simulated_board() {
    check(Recipe::AdamW513, "adamw_513", train::TOL_ADAMW_W_ABS);
}

#[test]
fn t165_planted_mistakes_in_the_driver_are_caught() {
    let want_g = tensors("adamw_513/grad_pas1");
    let want_w1 = tensors("adamw_513/dupa_pas1");
    let want_w10 = tensors("adamw_513/dupa_pas10");
    let want_l = json_list("adamw_513", "pierderi");
    for f in [
        Fault::NoBackResidual,
        Fault::XentScale,
        Fault::NoClip,
        Fault::HeadOverwrites,
    ] {
        // AdamW's first step barely sees a constant scale of the gradient (m/√v ≈ ±1): the missing clipping shows
        // only once the factor changes from step to step, so that mistake runs 10 steps.
        let steps = if f == Fault::NoClip { 10 } else { 1 };
        let r = run_with(Recipe::AdamW513, steps, f);
        let w10 = worst(&r.after10, &want_w10);
        let g = worst(&r.grad1, &want_g);
        let w1 = worst(&r.after1, &want_w1);
        let l = (r.losses[0] as f64 - want_l[0]).abs() / want_l[0];
        let caught = g.abs > train::TOL_GRAD_ABS
            || g.rel_norm > train::TOL_GRAD_REL_NORM
            || w1.abs > train::TOL_ADAMW_W_ABS
            || (steps == 10 && w10.abs > train::TOL_ADAMW_W_ABS)
            || l > train::TOL_LOSS_REL;
        println!(
            "kt c165 greseala {f:?} ({steps} pasi): grad_abs {:.2e} ({}), w1_abs {:.2e} ({}){} -> {}",
            g.abs,
            g.where_,
            w1.abs,
            w1.where_,
            if steps == 10 {
                format!(", w10_abs {:.2e} ({})", w10.abs, w10.where_)
            } else {
                String::new()
            },
            if caught { "PRINSA" } else { "SCAPATA" }
        );
        assert!(caught, "{f:?} nu a fost prinsa");
    }
}

/// Deterministic numbers in [−1, 1).
fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// #513's real shapes (C 768, 12 heads of 64, vocab 16 384, one text of 64 — `pas_de_aur_513/reteta.json`) but
/// `layers` layers and **made-up weights**: #513's weights are not in the cloud (2.7 GB, the hub has only the recipe),
/// so this is not the 513 golden step — it checks that the same list of launches, at #513's widths, gives the CPU
/// oracle's numbers (one AdamW step) in the hub's tolerances.
fn shapes_513(layers: usize) {
    let ar = Arch {
        layers,
        ..train::M513
    };
    let (bsz, tl) = (1usize, 64usize);
    let mut r = rng(513);
    let params: Vec<Vec<f32>> = ar
        .tensors()
        .iter()
        .map(|s| {
            let n = s.elements();
            if s.name.ends_with("ln1.weight")
                || s.name.ends_with("ln2.weight")
                || s.name == "ln.weight"
            {
                (0..n).map(|_| 1.0 + 0.1 * r()).collect()
            } else {
                (0..n).map(|_| 0.02 * r()).collect()
            }
        })
        .collect();
    let toks: Vec<u32> = (0..=tl).map(|_| ((r() + 1.0) * 8_191.9) as u32).collect();
    let (inputs, targets) = (toks[..tl].to_vec(), toks[1..].to_vec());
    let opt = AdamW::recipe_513(&params);
    let mut s = Step::new(ar, (bsz, tl), &params, &inputs, &targets);
    let rows = s.forward_backward();
    let loss = (rows.iter().map(|&x| x as f64).sum::<f64>() / rows.len() as f64) as f32;
    let grad = s.unflat(s.g);
    let norm = s.optimize(Recipe::AdamW513, &opt, 1).unwrap();
    let w1 = s.unflat(s.w);

    let batch = Batch {
        inputs: &inputs,
        targets: &targets,
        batch: bsz,
        len: tl,
    };
    let cpu = train::golden_run(&ar, &params, &batch, Recipe::AdamW513, 1).unwrap();
    let dev = |ours: &[Vec<f32>], theirs: &[Vec<f32>]| {
        let mut d = Dev {
            abs: 0.0,
            rel_norm: 0.0,
            where_: String::new(),
        };
        for ((o, t), spec) in ours.iter().zip(theirs).zip(ar.tensors()) {
            let (mut diff2, mut norm2) = (0.0f64, 0.0f64);
            for (&a, &b) in o.iter().zip(t) {
                let e = (a - b).abs();
                assert!(e.is_finite(), "{}: valoare nefinita", spec.name);
                if e > d.abs {
                    d.abs = e;
                    d.where_ = spec.name.clone();
                }
                diff2 += (e as f64).powi(2);
                norm2 += (b as f64).powi(2);
            }
            if norm2 > 0.0 {
                d.rel_norm = d.rel_norm.max((diff2 / norm2).sqrt());
            }
        }
        d
    };
    let g = dev(&grad, &cpu.grad_step1);
    let w = dev(&w1, &cpu.after_step1);
    let lrel = ((loss - cpu.losses[0]) as f64).abs() / cpu.losses[0] as f64;
    let nrel =
        (norm as f64 - train::grad_norm(&cpu.grad_step1)).abs() / train::grad_norm(&cpu.grad_step1);
    println!(
        "kt c165 formele lui #513 ({layers} straturi, greutati inventate, 1 x 64): {} lansari, memoria placii {} MiB; fata de oracol: eroarea {loss} rel {lrel:.2e}, norma rel {nrel:.2e}, grad_abs {:.2e} ({}) grad_rel_norma {:.2e}, w1_abs {:.2e} ({})",
        s.b.log.len(),
        (s.b.next - HEAP) >> 20,
        g.abs,
        g.where_,
        g.rel_norm,
        w.abs,
        w.where_
    );
    assert!(lrel <= train::TOL_LOSS_REL, "eroarea {lrel}");
    assert!(nrel <= 1e-5, "norma {nrel}");
    assert!(g.abs <= train::TOL_GRAD_ABS && g.rel_norm <= train::TOL_GRAD_REL_NORM);
    assert!(w.abs <= train::TOL_ADAMW_W_ABS);
}

#[test]
fn t165_one_layer_at_513_widths_matches_the_oracle() {
    shapes_513(1);
}

/// All 12 layers of #513 (minutes in the simulator; run by hand: `cargo test --release -- --ignored`).
#[test]
#[ignore]
fn t165_all_12_layers_at_513_widths_matches_the_oracle() {
    shapes_513(12);
}
