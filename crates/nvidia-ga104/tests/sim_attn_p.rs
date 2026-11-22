//! C-160 point 7 — T178, SIMULARE: `attn_p` (`tools/nak/attn_p.comp` → `tools/nak/out/attn_p.sm86.sass`, Mesa NAK
//! 25.0.7), the causal attention's P only, without P·V (`docs/C-160_P7_ATTN_P.md`). Launched through our pushbuffer +
//! QMD in the command processor model next to `attn_fwd`: the P it writes must be `attn_fwd`'s P **bit for bit** and
//! the `att` buffer must stay untouched. In the model "the same bits" holds by construction (both are the same CPU
//! function), so the test also compares the compiled code: the floating-point operations that lead to P must be the
//! same in both NAK dumps (that is what makes the bits equal on the board). Nothing here runs on the RTX 3070 Ti.

use gpu_compute::layer::AttnShape;
use gpu_sim::cmdproc::{CommandProcessor, attn_fwd_on_cpu, attn_p_on_cpu};
use nvidia_ga104::attn_fwd::{self as af, ATTN_FWD};
use nvidia_ga104::attn_p::{self as ap, ATTN_P};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::ComputeQmd;
use nvidia_rm570::regs::clc7c0 as c;

const PROG_FWD: u64 = 0x10_0000;
const PROG_P: u64 = 0x18_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
const QKV: u64 = 0x100_0000;
const QKV2: u64 = 0x200_0000;
const P: u64 = 0x400_0000;
const ATT: u64 = 0x800_0000;
const GUARD: usize = 64;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

fn shape(batch: usize, t: usize, c: usize, heads: usize) -> AttnShape {
    AttnShape { batch, t, c, heads }
}

fn qkv(s: &AttnShape, seed: u64, amp: f32) -> Vec<f32> {
    let (nq, _, _) = s.sizes().unwrap();
    let mut r = rng(seed);
    (0..nq).map(|_| r() * amp).collect()
}

fn args(s: &AttnShape, qkv: u64) -> af::Args {
    af::Args {
        qkv,
        p: P,
        att: ATT,
        batch: s.batch as u32,
        t: s.t as u32,
        c: s.c as u32,
        heads: s.heads as u32,
        scale: nvidia_ga104::attn_back::Args::scale_for((s.c / s.heads) as u32),
    }
}

fn cp() -> CommandProcessor {
    let mut cp = CommandProcessor::new();
    cp.register_program(ATTN_FWD, attn_fwd_on_cpu);
    cp.register_program(ATTN_P, attn_p_on_cpu);
    cp.mem.write(PROG_FWD, ATTN_FWD);
    cp.mem.write(PROG_P, ATTN_P);
    cp
}

fn launch(cp: &mut CommandProcessor, prog: u64, regs: u32, a: &af::Args, grid: [u32; 3], sem: u32) {
    let mut cb = vec![0u8; 0x100];
    let bytes = a.cbuf().unwrap();
    cb[..bytes.len()].copy_from_slice(&bytes);
    let cbuf = CBUF + 0x1000 * sem as u64;
    cp.mem.write(cbuf, &cb);
    let q = ComputeQmd {
        program_va: prog,
        registers: regs,
        cbuf0_va: cbuf,
        cbuf0_size: 0x100,
        grid,
        block: ap::BLOCK,
        release: Some((SEM, sem)),
    };
    let qmd = QMD + 0x1000 * sem as u64;
    cp.mem.write(qmd, &q.to_bytes().unwrap());
    let mut w = [0u32; 16];
    let mut p = PushBuf::new(&mut w);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(qmd >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = p.len();
    let words: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    let push_at = PUSH + 0x1000 * sem as u64;
    cp.mem.write(push_at, &words);
    let e = push::gpfifo_entry(push_at, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), sem);
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Fills P and att (+ guards) with NaN, loads qkv; returns (np, no).
fn load(cp: &mut CommandProcessor, s: &AttnShape, q: &[f32]) -> (usize, usize) {
    let (_, no, np) = s.sizes().unwrap();
    cp.mem.write_f32s(QKV, q);
    cp.mem.write_f32s(P, &vec![f32::NAN; np + GUARD]);
    cp.mem.write_f32s(ATT, &vec![f32::NAN; no + GUARD]);
    cp.mem.write_u32(SEM, 0);
    (np, no)
}

/// P of `attn_fwd` alone and P of `attn_p` alone (fresh boards), plus the att buffer after `attn_p`.
fn both(s: &AttnShape, q: &[f32]) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let mut f = cp();
    let (np, _) = load(&mut f, s, q);
    let a = args(s, QKV);
    launch(&mut f, PROG_FWD, af::REGISTERS, &a, a.grid(), 1);
    let pf = bits(&f.mem.read_f32s(P, np).unwrap());
    let mut g = cp();
    let (np, no) = load(&mut g, s, q);
    launch(&mut g, PROG_P, ap::REGISTERS, &a, a.grid(), 1);
    let all = g.mem.read_f32s(P, np + GUARD).unwrap();
    assert!(all[np..].iter().all(|v| v.is_nan()), "attn_p wrote after P");
    let att = bits(&g.mem.read_f32s(ATT, no + GUARD).unwrap());
    (pf, bits(&all[..np]), att)
}

/// The cases (written in the plan): the small golden model's shape, T not a multiple of 32, one head, d = 1, d = 64,
/// a sharp softmax, several texts.
fn cases() -> Vec<(&'static str, AttnShape, Vec<f32>)> {
    let mut v = Vec::new();
    for (name, s, seed, amp) in [
        ("pas-mic 4x32 C32 H4", shape(4, 32, 32, 4), 1u64, 1.0f32),
        ("T 45 (nemultiplu de 32)", shape(2, 45, 16, 2), 2, 1.0),
        ("o capatana H1 d40", shape(1, 37, 40, 1), 3, 0.7),
        ("d 1", shape(2, 33, 4, 4), 4, 2.0),
        ("d 64 H2", shape(1, 70, 128, 2), 5, 0.5),
        ("ascutit (P aproape 0/1)", shape(1, 40, 8, 1), 6, 9.0),
        ("loturi 3, T 96", shape(3, 96, 24, 3), 7, 1.0),
    ] {
        v.push((name, s, qkv(&s, seed, amp)));
    }
    v
}

#[test]
fn t178_attn_p_code_is_what_the_note_says() {
    let sum: String = nvidia_ga104::sha256::digest(ATTN_P)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(sum, ap::ATTN_P_SHA256);
    assert_eq!(ATTN_P.len() % 16, 0);
    assert!(
        ATTN_P.len() < ATTN_FWD.len(),
        "attn_p must be shorter than attn_fwd"
    );
    const { assert!(ap::REGISTERS <= af::REGISTERS) };
    assert_eq!(ap::CBUF_BYTES, af::CBUF_BYTES);
}

/// Prediction 2: the floating-point operations that lead to P are the same in both NAK dumps (same opcodes, same
/// modifiers); attn_fwd has in addition only the P·V chain (`ffma`).
#[test]
fn t178_compiled_p_ops_match_attn_fwd() {
    fn ops(txt: &str) -> Vec<String> {
        let mut v: Vec<String> = txt
            .lines()
            .filter_map(|l| {
                let rhs = l.split_once('=')?.1.trim();
                let op = rhs.split_whitespace().next()?;
                let base = op.split('.').next()?;
                ["ffma", "fmul", "fadd", "fmnmx", "mufu", "fsetp", "fswzadd"]
                    .contains(&base)
                    .then(|| op.to_string())
            })
            .collect();
        v.sort();
        v
    }
    let fwd = ops(include_str!("../../../tools/nak/out/attn_fwd.sm86.txt"));
    let p = ops(include_str!("../../../tools/nak/out/attn_p.sm86.txt"));
    let mut rest = fwd.clone();
    for o in &p {
        let at = rest
            .iter()
            .position(|x| x == o)
            .unwrap_or_else(|| panic!("attn_p has {o}, attn_fwd has no such operation left"));
        rest.remove(at);
    }
    assert!(
        rest.iter().all(|o| o.starts_with("ffma")),
        "attn_fwd's extra float ops are not only the P·V ffma chain: {rest:?}"
    );
    assert!(
        !rest.is_empty(),
        "attn_fwd should have the P·V ffma in addition"
    );
}

/// Prediction 3: P bit for bit equal to attn_fwd's, att untouched (still NaN), nothing written after P.
#[test]
fn t178_p_bit_for_bit_as_attn_fwd_and_att_untouched() {
    for (name, s, q) in cases() {
        let (pf, pp, att) = both(&s, &q);
        assert_eq!(pf, pp, "{name}: P of attn_p differs from attn_fwd");
        assert!(
            att.iter().all(|&b| b == f32::NAN.to_bits()),
            "{name}: attn_p touched att"
        );
    }
}

/// The re-launch of the step: attn_fwd writes P and att, something overwrites P, attn_p rebuilds it — P and att are
/// then exactly what attn_fwd left.
#[test]
fn t178_rebuilds_overwritten_p() {
    for (name, s, q) in cases() {
        let mut f = cp();
        let (np, no) = load(&mut f, &s, &q);
        let a = args(&s, QKV);
        launch(&mut f, PROG_FWD, af::REGISTERS, &a, a.grid(), 1);
        let p0 = bits(&f.mem.read_f32s(P, np).unwrap());
        let att0 = bits(&f.mem.read_f32s(ATT, no).unwrap());
        f.mem.write_f32s(P, &vec![0.25f32; np]);
        launch(&mut f, PROG_P, ap::REGISTERS, &a, a.grid(), 2);
        assert_eq!(bits(&f.mem.read_f32s(P, np).unwrap()), p0, "{name}");
        assert_eq!(bits(&f.mem.read_f32s(ATT, no).unwrap()), att0, "{name}");
    }
}

/// Prediction 4: planted mistakes are caught (the P differs from attn_fwd's).
#[test]
fn t178_planted_mistakes_caught() {
    let (_, s, q) = cases().swap_remove(1); // T 45: two blocks of rows
    let (pf, _, _) = both(&s, &q);
    let a = args(&s, QKV);
    let np = s.sizes().unwrap().2;
    // 1. the wrong scale
    let mut g = cp();
    load(&mut g, &s, &q);
    let wrong = af::Args {
        scale: a.scale * 1.0001,
        ..a
    };
    launch(&mut g, PROG_P, ap::REGISTERS, &wrong, a.grid(), 1);
    assert_ne!(
        bits(&g.mem.read_f32s(P, np).unwrap()),
        pf,
        "wrong scale not caught"
    );
    // 2. the grid without its last block of rows
    let mut g = cp();
    load(&mut g, &s, &q);
    let [x, y, z] = a.grid();
    launch(&mut g, PROG_P, ap::REGISTERS, &a, [x - 1, y, z], 1);
    assert_ne!(
        bits(&g.mem.read_f32s(P, np).unwrap()),
        pf,
        "short grid not caught"
    );
    // 3. another layer's qkv
    let mut g = cp();
    load(&mut g, &s, &q);
    g.mem.write_f32s(QKV2, &qkv(&s, 99, 1.0));
    launch(&mut g, PROG_P, ap::REGISTERS, &args(&s, QKV2), a.grid(), 1);
    assert_ne!(
        bits(&g.mem.read_f32s(P, np).unwrap()),
        pf,
        "wrong qkv not caught"
    );
}

/// The cbuf is attn_fwd's, byte for byte (the OS can reuse the same argument block).
#[test]
fn t178_same_cbuf_as_attn_fwd() {
    let s = shape(4, 32, 32, 4);
    let a = args(&s, QKV);
    let x: ap::Args = a;
    assert_eq!(x.cbuf(), a.cbuf());
    assert_eq!(x.grid(), [1, 4, 4]);
}
