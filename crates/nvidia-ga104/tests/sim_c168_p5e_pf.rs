//! C-168 p.5 P5e (session 295) — T312, SIMULARE: the `_pf` programs (the tail reads of `u` / `res` issued one fragment
//! ahead, `tools/nak/mm_tc_v8.comp` / `mm_tc_v7.comp` / `mm_tc_v8r.comp` with `-DPF`) put in the slots of the `_gelu` /
//! `_res` programs: the same launches, the same QMDs, the same bits. The plan and the predictions N1–N7 are in
//! `docs/C-168_P5_LIPIRE_PLAN.md` §15 (written before this test, session 295).
//!
//! Honesty: the simulated board runs the same CPU model for a `_pf` program as for its `_gelu` / `_res` program (the
//! arithmetic is the same by construction; that the compiled tails do the same operations is checked on the listings,
//! `tools/nak/compara_pf.py`, N3). What this test shows: the `_pf` programs are the checked bytes, they fit their slots,
//! their registers are the ones the packer writes into the QMD, and a whole fused step packed with them gives the same
//! bits. None of the seven `_pf` programs has run on a board.

use gpu_compute::train::{self, AdamW, Arch, Recipe};
use gpu_sim::cmdproc::{
    ATTN_BACK_TILE_TC16_NEEDS, ATTN_BACK_TILE_TC16_V4_NEEDS, ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    ATTN_BACK_TILE0_TC_NEEDS, ATTN_FWD_TC_NEEDS, CommandProcessor, GNORM_NEEDS, LN_BACK_NEEDS,
    LN_BACK_RES_NEEDS, LN_FWD_NEEDS, MM_TC_GELU_NEEDS, MM_TC_NEEDS, MM_TC_PF_NEEDS, MM_TC_V4_NEEDS,
    MM_TC_V5_NEEDS, MM_TC_V6_NEEDS, MM_TC_V7_GELU_NEEDS, MM_TC_V7_GELUF_NEEDS, MM_TC_V7_NEEDS,
    MM_TC_V7_RES_NEEDS, MM_TC_V8_GELU_NEEDS, MM_TC_V8_GELUF_NEEDS, MM_TC_V8_NEEDS,
    MM_TC_V8_RES_NEEDS, MM_TC_V8N_GELU_NEEDS, MM_TC_V8N_GELUF_NEEDS, MM_TC_V8N_NEEDS,
    MM_TC_V8N_RES_NEEDS, MM_TC_V8R_NEEDS, MM_TC_V8R_RES_NEEDS, MM_TRAIN_GELU_NEEDS, MM_TRAIN_NEEDS,
    Needs, XENT_NEEDS, attn_back_on_cpu, attn_back_tile_on_cpu, attn_back_tile_tc16_on_cpu,
    attn_back_tile_tc16_v4_on_cpu, attn_back_tile_tc16_v4pf_on_cpu, attn_back_tile0_on_cpu,
    attn_back_tile0_tc_on_cpu, attn_fwd_on_cpu, attn_fwd_tc_on_cpu, attn_fwd_tile_on_cpu,
    attn_p_on_cpu, attn_p_tile_on_cpu, colsum_on_cpu, colsum_part_on_cpu, emb_back_on_cpu,
    emb_fwd_on_cpu, gelu_tr_on_cpu, gnorm_on_cpu, ln_back_on_cpu, ln_back_res_on_cpu,
    ln_fwd_on_cpu, mm_tc_gelu_on_cpu, mm_tc_on_cpu, mm_tc_pf_on_cpu, mm_tc_v4_on_cpu,
    mm_tc_v5_on_cpu, mm_tc_v6_on_cpu, mm_tc_v7_gelu_on_cpu, mm_tc_v7_geluf_on_cpu, mm_tc_v7_on_cpu,
    mm_tc_v7_res_on_cpu, mm_tc_v8_gelu_on_cpu, mm_tc_v8_geluf_on_cpu, mm_tc_v8_on_cpu,
    mm_tc_v8_res_on_cpu, mm_tc_v8n_gelu_on_cpu, mm_tc_v8n_geluf_on_cpu, mm_tc_v8n_on_cpu,
    mm_tc_v8n_res_on_cpu, mm_tc_v8r_on_cpu, mm_tc_v8r_res_on_cpu, mm_train_gelu_on_cpu,
    mm_train_on_cpu, opt_step_on_cpu, vadd_on_cpu, xent_on_cpu,
};
use nvidia_ga104::attn_back_tile_tc16::ATTN_BACK_TILE_TC16;
use nvidia_ga104::attn_back_tile_tc16_v4::ATTN_BACK_TILE_TC16_V4;
use nvidia_ga104::attn_back_tile_tc16_v4pf::ATTN_BACK_TILE_TC16_V4PF;
use nvidia_ga104::attn_back_tile0_tc::ATTN_BACK_TILE0_TC;
use nvidia_ga104::attn_fwd_tc::ATTN_FWD_TC;
use nvidia_ga104::mm_tc::MM_TC;
use nvidia_ga104::mm_tc_fast_gelu::{MM_TC_V7_GELU, MM_TC_V8_GELU, MM_TC_V8N_GELU};
use nvidia_ga104::mm_tc_fast_gelu_fwd::{MM_TC_V7_GELUF, MM_TC_V8_GELUF, MM_TC_V8N_GELUF};
use nvidia_ga104::mm_tc_fast_pf::{
    self as fp, MM_TC_V7_GELU_PF, MM_TC_V7_RES_PF, MM_TC_V8_GELU_PF, MM_TC_V8_RES_PF,
    MM_TC_V8N_GELU_PF, MM_TC_V8N_RES_PF, MM_TC_V8R_RES_PF,
};
use nvidia_ga104::mm_tc_fast_res::{MM_TC_V7_RES, MM_TC_V8_RES, MM_TC_V8N_RES, MM_TC_V8R_RES};
use nvidia_ga104::mm_tc_gelu::MM_TC_GELU;
use nvidia_ga104::mm_tc_pf::MM_TC_PF;
use nvidia_ga104::mm_tc_v4::MM_TC_V4;
use nvidia_ga104::mm_tc_v5::MM_TC_V5;
use nvidia_ga104::mm_tc_v6::MM_TC_V6;
use nvidia_ga104::mm_tc_v7::MM_TC_V7;
use nvidia_ga104::mm_tc_v8::{self as t8, MM_TC_V8, MM_TC_V8N};
use nvidia_ga104::mm_tc_v8r::MM_TC_V8R;
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::train_step::{
    self as ts, Attn0Programs, AttnTcFast, AttnTcProgram, GeluFast, GeluFwdFast, LOSS_SCALE_513,
    MixedLaunch, PROGRAMS_ALL_FUSED2, Plan, Prog, ResFast, TcPick, TcPrograms, V8Programs,
};
use nvidia_ga104::transfer;
use nvidia_rm570::regs::clc7c0 as c;

/// The mini model of T297 / T304 (d = 16: `attn_fwd_tc` does not take it).
const MINI: Arch = Arch {
    vocab: 16_384,
    ctx: 64,
    n_emb: 128,
    heads: 2,
    layers: 2,
};
const B: usize = 2;
const T: usize = 64;
const STEPS: u32 = 5;
const LR: f64 = 1e-3;
const INIT_HALF_WIDTH: f64 = 0.034_641_016;
const KIB: u32 = 1024;

const VA: u64 = nvidia_ga104::mmu::VA_BASE;
const ZONE: u64 = VA + 0x20_0000;
const REC: u64 = VA + 0x30_0000;
const PUSH: u64 = VA + 0x32_0000;
const PROGS: u64 = VA + 0x40_0000;
const TC: u64 = PROGS + 0x1_0000 * 23;
const TCG: u64 = PROGS + 0x1_0000 * 24;
const PF: u64 = PROGS + 0x1_0000 * 25;
const V4: u64 = PROGS + 0x1_0000 * 26;
const ATC16: u64 = PROGS + 0x1_0000 * 28;
const ATC0: u64 = PROGS + 0x1_0000 * 29;
const ATC16V4: u64 = PROGS + 0x1_0000 * 31;
const ATC16V4PF: u64 = PROGS + 0x1_0000 * 32;
const V5: u64 = PROGS + 0x1_0000 * 33;
const V6: u64 = PROGS + 0x1_0000 * 34;
const V7: u64 = PROGS + 0x1_0000 * 35;
const V8: u64 = PROGS + 0x1_0000 * 36;
const V8N: u64 = PROGS + 0x1_0000 * 37;
const AFT: u64 = PROGS + 0x1_0000 * 38;
const G7: u64 = PROGS + 0x1_0000 * 41;
const G8: u64 = PROGS + 0x1_0000 * 42;
const G8N: u64 = PROGS + 0x1_0000 * 43;
const V8R: u64 = PROGS + 0x1_0000 * 44;
const F7: u64 = PROGS + 0x1_0000 * 48;
const F8: u64 = PROGS + 0x1_0000 * 50; // the geluf programs are ~70 KiB: two 64 KiB slots each
const F8N: u64 = PROGS + 0x1_0000 * 52;
const R7: u64 = PROGS + 0x1_0000 * 54;
const R8: u64 = PROGS + 0x1_0000 * 56; // two 64 KiB slots each, as the geluf programs
const R8N: u64 = PROGS + 0x1_0000 * 58;
const R8R: u64 = PROGS + 0x1_0000 * 60;
const RF: ResFast = ResFast {
    v7: R7,
    v8: R8,
    v8n: R8N,
    v8r: R8R,
};
const GFF: GeluFwdFast = GeluFwdFast {
    v7: F7,
    v8: F8,
    v8n: F8N,
};
const GF: GeluFast = GeluFast {
    v7: G7,
    v8: G8,
    v8n: G8N,
};
/// The os-500 rates, as the OS runs `--pas-v8` since os-503.
const P8: V8Programs = V8Programs {
    v8: V8,
    v8n: V8N,
    sms: 48,
    rates: t8::RATES_OS500,
};
/// Where `colsum_part` is loaded (see [`prog_va`]).
const CSP: u64 = PROGS + 0x1_0000 * 40;
const HEAP: u64 = VA + 0x100_0000;
const _: () = assert!(R8R + 0x2_0000 <= HEAP);

const FAST: TcPrograms = TcPrograms {
    tc: TC,
    pf: PF,
    v4: V4,
    gelu: TCG,
};
const AT: Option<AttnTcProgram> = Some(AttnTcProgram::Tc16(ATC16));
const A0: Attn0Programs = Attn0Programs { tc: ATC0, v4: 0 };
const FV4: Option<AttnTcFast> = Some(AttnTcFast::V4(ATC16V4));

fn prog_va(p: Prog) -> u64 {
    if p == Prog::ColsumPart {
        // Index 24 = TCG's slot in this harness: put `colsum_part` after every other program.
        return CSP;
    }
    PROGS + 0x1_0000 * p.index() as u64
}

fn needs(s: nvidia_ga104::qmd::SharedMemory) -> Needs {
    Needs {
        shared_bytes: s.bytes,
        barriers: u32::from(s.barriers),
    }
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn initial() -> Vec<Vec<f32>> {
    let mut g = Rng(0x00c1_9e10_2026_1126);
    MINI.tensors()
        .iter()
        .map(|t| {
            let n = t.elements();
            let leaf = t.name.rsplit('.').nth(1).unwrap_or("");
            if t.name.ends_with("bias") {
                vec![0.0f32; n]
            } else if leaf.starts_with("ln") {
                vec![1.0f32; n]
            } else {
                (0..n)
                    .map(|_| ((g.unit() * 2.0 - 1.0) * INIT_HALF_WIDTH) as f32)
                    .collect()
            }
        })
        .collect()
}

fn batch() -> (Vec<u32>, Vec<u32>) {
    let mut g = Rng(0x5eed_0000_0000_0253);
    let (mut inp, mut tgt) = (Vec::new(), Vec::new());
    for _ in 0..B {
        let row: Vec<u32> = (0..=T)
            .map(|_| (g.next() % MINI.vocab as u64) as u32)
            .collect();
        inp.extend_from_slice(&row[..T]);
        tgt.extend_from_slice(&row[1..]);
    }
    (inp, tgt)
}

/// The simulated board; with `pf` the seven `_pf` programs sit in the slots of their `_gelu` / `_res` programs.
fn board(pf: bool) -> CommandProcessor {
    ts::check_programs().expect("SHA-256 programe");
    let mut cp = CommandProcessor::new();
    for p in PROGRAMS_ALL_FUSED2 {
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
            Prog::AttnFwdTile2 | Prog::AttnFwdTile3 => {
                cp.register_program_with_needs(code, attn_fwd_tile_on_cpu, sh())
            }
            Prog::AttnBackTile0_2 | Prog::AttnBackTile0_3 => {
                cp.register_program_with_needs(code, attn_back_tile0_on_cpu, sh())
            }
            Prog::AttnBackTile2 | Prog::AttnBackTile3 => {
                cp.register_program_with_needs(code, attn_back_tile_on_cpu, sh())
            }
            Prog::AttnPTile => cp.register_program_with_needs(code, attn_p_tile_on_cpu, sh()),
            Prog::LnBackRes => {
                cp.register_program_with_needs(code, ln_back_res_on_cpu, LN_BACK_RES_NEEDS)
            }
            Prog::MmTrainGelu => {
                cp.register_program_with_needs(code, mm_train_gelu_on_cpu, MM_TRAIN_GELU_NEEDS)
            }
            Prog::MmTrainTile
            | Prog::ColsumPart
            | Prog::MmTv128x64
            | Prog::MmTv64x128
            | Prog::MmTv128x128
            | Prog::MmTv64x64K8
            | Prog::MmTrainTileGelu => unreachable!("nu e in PROGRAMS_ALL_FUSED2"),
        };
        cp.mem.write(prog_va(p), code);
    }
    cp.register_program_with_needs(MM_TC, mm_tc_on_cpu, MM_TC_NEEDS);
    cp.mem.write(TC, MM_TC);
    cp.register_program_with_needs(MM_TC_GELU, mm_tc_gelu_on_cpu, MM_TC_GELU_NEEDS);
    cp.mem.write(TCG, MM_TC_GELU);
    cp.register_program_with_needs(MM_TC_PF, mm_tc_pf_on_cpu, MM_TC_PF_NEEDS);
    cp.mem.write(PF, MM_TC_PF);
    cp.register_program_with_needs(MM_TC_V4, mm_tc_v4_on_cpu, MM_TC_V4_NEEDS);
    cp.mem.write(V4, MM_TC_V4);
    cp.register_program_with_needs(MM_TC_V5, mm_tc_v5_on_cpu, MM_TC_V5_NEEDS);
    cp.mem.write(V5, MM_TC_V5);
    cp.register_program_with_needs(MM_TC_V6, mm_tc_v6_on_cpu, MM_TC_V6_NEEDS);
    cp.mem.write(V6, MM_TC_V6);
    cp.register_program_with_needs(MM_TC_V7, mm_tc_v7_on_cpu, MM_TC_V7_NEEDS);
    cp.mem.write(V7, MM_TC_V7);
    cp.register_program_with_needs(MM_TC_V8, mm_tc_v8_on_cpu, MM_TC_V8_NEEDS);
    cp.mem.write(V8, MM_TC_V8);
    cp.register_program_with_needs(MM_TC_V8N, mm_tc_v8n_on_cpu, MM_TC_V8N_NEEDS);
    cp.mem.write(V8N, MM_TC_V8N);
    cp.register_program_with_needs(MM_TC_V8R, mm_tc_v8r_on_cpu, MM_TC_V8R_NEEDS);
    cp.mem.write(V8R, MM_TC_V8R);
    cp.register_program_with_needs(
        ATTN_BACK_TILE_TC16,
        attn_back_tile_tc16_on_cpu,
        ATTN_BACK_TILE_TC16_NEEDS,
    );
    cp.mem.write(ATC16, ATTN_BACK_TILE_TC16);
    cp.register_program_with_needs(
        ATTN_BACK_TILE0_TC,
        attn_back_tile0_tc_on_cpu,
        ATTN_BACK_TILE0_TC_NEEDS,
    );
    cp.mem.write(ATC0, ATTN_BACK_TILE0_TC);
    cp.register_program_with_needs(
        ATTN_BACK_TILE_TC16_V4,
        attn_back_tile_tc16_v4_on_cpu,
        ATTN_BACK_TILE_TC16_V4_NEEDS,
    );
    cp.mem.write(ATC16V4, ATTN_BACK_TILE_TC16_V4);
    cp.register_program_with_needs(
        ATTN_BACK_TILE_TC16_V4PF,
        attn_back_tile_tc16_v4pf_on_cpu,
        ATTN_BACK_TILE_TC16_V4PF_NEEDS,
    );
    cp.mem.write(ATC16V4PF, ATTN_BACK_TILE_TC16_V4PF);
    cp.register_program_with_needs(ATTN_FWD_TC, attn_fwd_tc_on_cpu, ATTN_FWD_TC_NEEDS);
    cp.register_program_with_needs(
        if pf { MM_TC_V7_GELU_PF } else { MM_TC_V7_GELU },
        mm_tc_v7_gelu_on_cpu,
        MM_TC_V7_GELU_NEEDS,
    );
    cp.mem
        .write(G7, if pf { MM_TC_V7_GELU_PF } else { MM_TC_V7_GELU });
    cp.register_program_with_needs(
        if pf { MM_TC_V8_GELU_PF } else { MM_TC_V8_GELU },
        mm_tc_v8_gelu_on_cpu,
        MM_TC_V8_GELU_NEEDS,
    );
    cp.mem
        .write(G8, if pf { MM_TC_V8_GELU_PF } else { MM_TC_V8_GELU });
    cp.register_program_with_needs(
        if pf {
            MM_TC_V8N_GELU_PF
        } else {
            MM_TC_V8N_GELU
        },
        mm_tc_v8n_gelu_on_cpu,
        MM_TC_V8N_GELU_NEEDS,
    );
    cp.mem.write(
        G8N,
        if pf {
            MM_TC_V8N_GELU_PF
        } else {
            MM_TC_V8N_GELU
        },
    );
    cp.mem.write(AFT, ATTN_FWD_TC);
    cp.register_program_with_needs(MM_TC_V7_GELUF, mm_tc_v7_geluf_on_cpu, MM_TC_V7_GELUF_NEEDS);
    cp.mem.write(F7, MM_TC_V7_GELUF);
    cp.register_program_with_needs(MM_TC_V8_GELUF, mm_tc_v8_geluf_on_cpu, MM_TC_V8_GELUF_NEEDS);
    cp.mem.write(F8, MM_TC_V8_GELUF);
    cp.register_program_with_needs(
        MM_TC_V8N_GELUF,
        mm_tc_v8n_geluf_on_cpu,
        MM_TC_V8N_GELUF_NEEDS,
    );
    cp.mem.write(F8N, MM_TC_V8N_GELUF);
    cp.register_program_with_needs(
        if pf { MM_TC_V7_RES_PF } else { MM_TC_V7_RES },
        mm_tc_v7_res_on_cpu,
        MM_TC_V7_RES_NEEDS,
    );
    cp.mem
        .write(R7, if pf { MM_TC_V7_RES_PF } else { MM_TC_V7_RES });
    cp.register_program_with_needs(
        if pf { MM_TC_V8_RES_PF } else { MM_TC_V8_RES },
        mm_tc_v8_res_on_cpu,
        MM_TC_V8_RES_NEEDS,
    );
    cp.mem
        .write(R8, if pf { MM_TC_V8_RES_PF } else { MM_TC_V8_RES });
    cp.register_program_with_needs(
        if pf { MM_TC_V8N_RES_PF } else { MM_TC_V8N_RES },
        mm_tc_v8n_res_on_cpu,
        MM_TC_V8N_RES_NEEDS,
    );
    cp.mem
        .write(R8N, if pf { MM_TC_V8N_RES_PF } else { MM_TC_V8N_RES });
    cp.register_program_with_needs(
        if pf { MM_TC_V8R_RES_PF } else { MM_TC_V8R_RES },
        mm_tc_v8r_res_on_cpu,
        MM_TC_V8R_RES_NEEDS,
    );
    cp.mem
        .write(R8R, if pf { MM_TC_V8R_RES_PF } else { MM_TC_V8R_RES });
    let csp = Prog::ColsumPart.code();
    cp.register_program(csp, colsum_part_on_cpu);
    cp.mem.write(CSP, csp);
    cp
}

fn scale() -> f32 {
    ts::loss_scale_for(MINI.vocab, B * T)
}

/// The P5c plan shape on the mini model (`new_tiles3` + `with_colsum_split` + `with_fused_back` + `with_fused_gelu` +
/// `with_fused_gelu_fwd`), + `with_fused_res_fwd` when `res`.
fn plan_mini(inputs: &[u32], res: bool) -> Plan {
    let p = Plan::new_tiles3(MINI, B, T, inputs, HEAP)
        .unwrap()
        .with_colsum_split()
        .with_fused_back()
        .with_fused_gelu()
        .with_fused_gelu_fwd();
    if res { p.with_fused_res_fwd() } else { p }
}

fn submit(
    cp: &mut CommandProcessor,
    first: &mut u32,
    zone: &[u8],
    qmds: &[u64],
) -> Result<(), String> {
    assert!(zone.len() as u64 <= REC - ZONE, "zona de control prea mare");
    cp.mem.write(ZONE, zone);
    let n = qmds.len();
    cp.mem.write(REC, &vec![0u8; 16 * (n + 1)]);
    let mut w = vec![0u32; 2 + transfer::sequence_words_fresh(n)];
    let mut pb = PushBuf::new(&mut w);
    pb.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    transfer::encode_sequence_fresh(&mut pb, qmds, REC, *first).unwrap();
    let nw = pb.len();
    let bytes: Vec<u8> = w[..nw].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, nw as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])
        .map_err(|err| format!("{err:?}"))?;
    *first = first.wrapping_add(n as u32 + 1);
    Ok(())
}

/// The #513 list of step 1 as the OS would send it after P5c (`new_tiles3` + `with_mm_tile` + `with_colsum_split` +
/// `with_fused_back` + `with_fused_gelu` + `with_fused_gelu_fwd`, the attention on the tensor cores), +
/// `with_fused_res_fwd` when `res`.
fn list_513(bsz: usize, res: bool) -> (Plan, Vec<MixedLaunch>) {
    let m = train::M513;
    let inputs: Vec<u32> = (0..bsz as u32 * 512).map(|i| (i * 7919) % 16_384).collect();
    let base = VA + (1 << 30);
    let plan = Plan::new_tiles3(m, bsz, 512, &inputs, base)
        .unwrap()
        .with_mm_tile()
        .with_colsum_split()
        .with_fused_back()
        .with_fused_gelu()
        .with_fused_gelu_fwd();
    let plan = if res { plan.with_fused_res_fwd() } else { plan };
    let opt = AdamW::new(&[], 1e-3, 0.9, 0.999, 1e-8, 0.1);
    let l = plan
        .step_mixed(Recipe::AdamW513, &opt, 1, LOSS_SCALE_513)
        .unwrap();
    let l = ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(l).unwrap()).unwrap()).unwrap();
    (plan, l)
}

type Packed = (Vec<u8>, Vec<u64>, Vec<Option<TcPick>>);

/// `pack_mixed_fast_attn13` with the P5b GELU' variants, `v8r_va`, the P5c `geluf` programs, `res_fwd`, the os-503
/// guard, the attention on the tensor cores.
fn pack13(
    list: &[MixedLaunch],
    p8: Option<V8Programs>,
    v8r_va: u64,
    res_fwd: Option<ResFast>,
) -> Result<Packed, String> {
    ts::pack_mixed_fast_attn13(
        list,
        prog_va,
        &FAST,
        V5,
        V6,
        V7,
        p8,
        t8::LONG_K_OS503,
        AFT,
        Some(GF),
        v8r_va,
        Some(GFF),
        res_fwd,
        AT,
        FV4,
        A0,
        Some(64 * KIB),
        ZONE,
    )
    .map_err(|e| format!("{e:?}"))
}

fn count(picks: &[Option<TcPick>], p: TcPick) -> usize {
    picks.iter().filter(|x| **x == Some(p)).count()
}

/// Reads one field of a 256-byte QMD.
fn qmd_field(qmd: &[u8], fl: nvidia_rm570::QmdField) -> u64 {
    let mut v = 0u64;
    for i in fl.lo..=fl.hi {
        if qmd[(i / 8) as usize] >> (i % 8) & 1 == 1 {
            v |= 1 << (i - fl.lo);
        }
    }
    v
}

/// Rates that make the waves give v8 every aligned product (to exercise v8 / v8_geluf on the mini model).
const P8_V8: V8Programs = V8Programs {
    rates: t8::Rates {
        v8_pct: 400,
        v8n_pct: 118,
    },
    sms: 1,
    ..P8
};

fn nres(p: &[Option<TcPick>]) -> usize {
    count(p, TcPick::V7Res)
        + count(p, TcPick::V8Res)
        + count(p, TcPick::V8nRes)
        + count(p, TcPick::V8rRes)
}

struct Out {
    losses: Vec<Vec<f32>>,
    grad1: Vec<f32>,
    weights: Vec<f32>,
    picks: Vec<Option<TcPick>>,
    launches: usize,
}

/// `STEPS` fast mixed steps of the aligned mini model (P5c plan, + `with_fused_res_fwd` when `res`) on the simulated
/// board, packed by `attn13` with `p8`, `v8r_va`, the `geluf` and the `_res` programs.
fn run(p8: V8Programs, v8r_va: u64, res: bool, pf: bool) -> Out {
    let (inputs, targets) = batch();
    let params = initial();
    let plan = plan_mini(&inputs, res);
    let mut cp = board(pf);
    for (va, b) in plan.uploads(&params, &inputs, &targets).unwrap() {
        cp.mem.write(va, &b);
    }
    for (va, b) in plan.mixed_uploads(scale()).unwrap() {
        cp.mem.write(va, &b);
    }
    let opt = AdamW::new(&params, LR, 0.9, 0.999, 1e-8, 0.1);
    let mut first = 0x7000u32;
    let mut out = Out {
        losses: Vec::new(),
        grad1: Vec::new(),
        weights: Vec::new(),
        picks: Vec::new(),
        launches: 0,
    };
    for step in 1..=STEPS {
        let list = plan
            .step_mixed(Recipe::AdamW513, &opt, step, scale())
            .unwrap();
        assert_eq!(
            list.len(),
            plan.launches_per_step_mixed(Recipe::AdamW513),
            "launches_per_step_mixed"
        );
        let list =
            ts::attn_fwd_tc_list(ts::attn0_tc_list(ts::attn_tc_list(list).unwrap()).unwrap())
                .unwrap();
        let (zone, qmds, picks) = pack13(&list, Some(p8), v8r_va, Some(RF)).unwrap();
        out.picks = picks;
        out.launches = list.len();
        submit(&mut cp, &mut first, &zone, &qmds).unwrap();
        out.losses
            .push(cp.mem.read_f32s(plan.loss_rows, B * T).unwrap());
        if step == 1 {
            out.grad1 = cp.mem.read_f32s(plan.g, plan.n_params).unwrap();
        }
    }
    out.weights = cp.mem.read_f32s(plan.w, plan.n_params).unwrap();
    out
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn same(a: &Out, b: &Out, what: &str) {
    for (s, (x, y)) in a.losses.iter().zip(&b.losses).enumerate() {
        assert_eq!(bits(x), bits(y), "{what}: pierderile pasului {}", s + 1);
    }
    assert_eq!(
        bits(&a.grad1),
        bits(&b.grad1),
        "{what}: gradientul pasului 1"
    );
    assert_eq!(bits(&a.weights), bits(&b.weights), "{what}: greutatile");
}

/// Rates that make the waves give v8n every aligned product.
const P8_V8N: V8Programs = V8Programs {
    rates: t8::Rates {
        v8_pct: 100,
        v8n_pct: 400,
    },
    ..P8_V8
};

/// T312 S1 (N4), SIMULARE: the mini model, 5 fast mixed steps with every fusion (P5a + P5b + P5c + P5d): with the
/// `_pf` programs in the slots the same bits as with the `_gelu` / `_res` programs (losses, the gradient of step 1, the
/// weights), the same picks and launches — on the os-500 rates and the rates steered to v8, v8 + v8r, v8n.
#[test]
fn t312_s1_mini_same_bits() {
    let mut lines = Vec::new();
    for (p8, v8r, name) in [
        (P8, 0, "os-500"),
        (P8_V8, 0, "v8"),
        (P8_V8, V8R, "v8 + v8r"),
        (P8_V8N, 0, "v8n"),
    ] {
        let a = run(p8, v8r, true, false);
        let b = run(p8, v8r, true, true);
        same(&a, &b, name);
        assert_eq!(a.picks, b.picks, "{name}: aceleasi alegeri");
        assert_eq!(a.launches, b.launches, "{name}: aceleasi lansari");
        let ngelu = count(&b.picks, TcPick::V7Gelu)
            + count(&b.picks, TcPick::V8Gelu)
            + count(&b.picks, TcPick::V8nGelu);
        assert_eq!(
            ngelu, MINI.layers,
            "{name}: proj-dX + GELU' pe un program _gelu (_pf)"
        );
        assert_eq!(
            nres(&b.picks),
            2 * MINI.layers,
            "{name}: out / proj pe un program _res (_pf)"
        );
        assert!(b.weights.iter().all(|x| x.is_finite()));
        lines.push(format!(
            "{name}: {} lansari, {ngelu} gelu + {} res pe _pf",
            b.launches,
            nres(&b.picks)
        ));
    }
    println!(
        "kt c312 S1 SIMULARE: modelul mic, 5 pasi rapizi P5a-P5d, programele _pf in locul celor _gelu / _res = aceiasi biti; {}",
        lines.join("; ")
    );
}

/// T312 S2 (N2 + N4), SIMULARE: the `_pf` programs are the checked bytes (SHA-256), differ from their `_gelu` / `_res`
/// programs, fit the slots the packer and the OS give those programs, and have their registers — so the QMDs the packer
/// writes for #513 lot 2 and lot 6 (registers, block, grid, address of the slot) are right for them as they are.
#[test]
fn t312_s2_513_qmds_fit_the_pf_programs() {
    use nvidia_rm570::regs::qmd_v03_00 as q;
    fp::check().unwrap();
    for (pf, plain, slot, room) in [
        (MM_TC_V7_GELU_PF, MM_TC_V7_GELU, G7, G8 - G7),
        (MM_TC_V8_GELU_PF, MM_TC_V8_GELU, G8, G8N - G8),
        (MM_TC_V8N_GELU_PF, MM_TC_V8N_GELU, G8N, V8R - G8N),
        (MM_TC_V7_RES_PF, MM_TC_V7_RES, R7, R8 - R7),
        (MM_TC_V8_RES_PF, MM_TC_V8_RES, R8, R8N - R8),
        (MM_TC_V8N_RES_PF, MM_TC_V8N_RES, R8N, R8R - R8N),
        (MM_TC_V8R_RES_PF, MM_TC_V8R_RES, R8R, HEAP - R8R),
    ] {
        assert_ne!(pf, plain);
        assert!(
            pf.len() as u64 <= room,
            "programul _pf de la {slot:#x} nu incape ({} B)",
            pf.len()
        );
    }
    let mut lines = Vec::new();
    for bsz in [2usize, 6] {
        let (_, l) = list_513(bsz, true);
        let (z, qs, k) = pack13(&l, Some(P8), V8R, Some(RF)).unwrap();
        let mut n = 0;
        for (i, pick) in k.iter().enumerate() {
            let want = match pick {
                Some(TcPick::V7Gelu) => (G7, fp::REGISTERS_V7_GELU),
                Some(TcPick::V8Gelu) => (G8, fp::REGISTERS_V8_GELU),
                Some(TcPick::V8nGelu) => (G8N, fp::REGISTERS_V8N_GELU),
                Some(TcPick::V7Res) => (R7, fp::REGISTERS_V7_RES),
                Some(TcPick::V8Res) => (R8, fp::REGISTERS_V8_RES),
                Some(TcPick::V8nRes) => (R8N, fp::REGISTERS_V8N_RES),
                Some(TcPick::V8rRes) => (R8R, fp::REGISTERS_V8R_RES),
                _ => continue,
            };
            let at = (qs[i] - ZONE) as usize;
            let qmd = &z[at..at + 0x100];
            let va = qmd_field(qmd, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER) << 32
                | qmd_field(qmd, q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER);
            assert_eq!(va, want.0, "lot {bsz}, lansarea {i}: adresa");
            assert_eq!(
                qmd_field(qmd, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V),
                u64::from(want.1),
                "lot {bsz}, lansarea {i}: registrele programului _pf"
            );
            n += 1;
        }
        assert_eq!(
            n,
            12 + 24,
            "lot {bsz}: 12 proj-dX + GELU' si 24 out / proj + rezidual"
        );
        lines.push(format!("lot {bsz}: {n} QMD-uri potrivite programelor _pf"));
    }
    println!("kt c312 S2 SIMULARE: {}", lines.join("; "));
}
