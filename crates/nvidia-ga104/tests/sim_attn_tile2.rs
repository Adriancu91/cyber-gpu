//! C-157 point 2 after os-358 — T181, SIMULARE: `attn_fwd_tile2` (`tools/nak/attn_fwd_tile2.comp`, Mesa NAK 25.0.7;
//! `docs/C-157_P2_ATTN_TILE2.md`). The plan and the predictions were written and pushed BEFORE this code (`7fa6afa`):
//!   1. the same operations: 16 `mufu.ex2` + 1 `mufu.rcp` as `attn_fwd_tile`, no atomics, no local memory, 1 barrier;
//!   2. registers ≤ 128, 37 888 shared bytes in the QMD;
//!   3. loops (on tiles below the diagonal): phase A ≤ 28 instructions per 16 `ffma`, phase C ≤ 30, no `bssy` in either;
//!   4. the launch = `attn_fwd_tile`'s bit for bit on T173's 10 shapes; planted mistakes caught (old shared size, a grid
//!      one block short, a wrong block, d > 64).
//!
//! What the simulation can and cannot show: the "program" runs on the CPU (the same model as `attn_fwd_tile`,
//! `attn_fwd_tile_on_cpu`), so point 4 checks the driver side (QMD with the new shared size, grid, block, cbuf, which
//! rows get written), not the NAK code. That the board gives `attn_fwd_tile`'s bits is a prediction for the board,
//! as is the time. Points 1–3 are read from the NAK listing. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::layer::{AttnShape, attn_fwd_tile_gpu};
use gpu_sim::cmdproc::{CommandProcessor, Needs, attn_fwd_tile_on_cpu};
use nvidia_ga104::attn_back as ab;
use nvidia_ga104::attn_fwd as af;
use nvidia_ga104::attn_fwd_tile::{self as at, ATTN_FWD_TILE};
use nvidia_ga104::attn_fwd_tile2::{self as at2, ATTN_FWD_TILE2};
use nvidia_ga104::push::{self, PushBuf};
use nvidia_ga104::qmd::{ComputeQmd, SharedMemory};
use nvidia_rm570::regs::clc7c0 as c;

const PROG: u64 = 0x10_0000;
const CBUF: u64 = 0x20_0000;
const SEM: u64 = 0x30_0000;
const QMD: u64 = 0x31_0000;
const PUSH: u64 = 0x32_0000;
const QKV: u64 = 0x100_0000;
const P: u64 = 0x400_0000;
const ATT: u64 = 0x800_0000;
const SENTINEL: f32 = -12345.0;

fn rng(seed: u64) -> impl FnMut() -> f32 {
    let mut s = seed;
    move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

struct Case {
    s: AttnShape,
    qkv: Vec<f32>,
}

impl Case {
    fn new(seed: u64, s: AttnShape, amp: f32) -> Case {
        let (nq, _, _) = s.sizes().unwrap();
        let mut r = rng(seed);
        Case {
            s,
            qkv: (0..nq).map(|_| r() * amp).collect(),
        }
    }
    fn scale(&self) -> f32 {
        ab::Args::scale_for(self.s.d() as u32)
    }
}

fn args(k: &Case) -> af::Args {
    af::Args {
        qkv: QKV,
        p: P,
        att: ATT,
        batch: k.s.batch as u32,
        t: k.s.t as u32,
        c: k.s.c as u32,
        heads: k.s.heads as u32,
        scale: k.scale(),
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Prog {
    Tile,
    Tile2,
}

/// One launch through pushbuffer + QMD; `qmd_shared` is what the QMD declares, the simulator demands the program's own
/// shared size and barrier (as the board would need them).
fn launch(
    k: &Case,
    prog: Prog,
    grid: [u32; 3],
    block: [u32; 3],
    qmd_shared: SharedMemory,
) -> Result<(Vec<f32>, Vec<f32>), gpu_core::error::GpuError> {
    let (_, no, np) = k.s.sizes().unwrap();
    let mut cp = CommandProcessor::new();
    let (code, regs, own) = match prog {
        Prog::Tile => (ATTN_FWD_TILE, at::REGISTERS, at::SHARED),
        Prog::Tile2 => (ATTN_FWD_TILE2, at2::REGISTERS, at2::SHARED),
    };
    cp.register_program_with_needs(
        code,
        attn_fwd_tile_on_cpu,
        Needs {
            shared_bytes: own.bytes,
            barriers: u32::from(own.barriers),
        },
    );
    cp.mem.write(PROG, code);
    cp.mem.write_f32s(QKV, &k.qkv);
    cp.mem.write_f32s(P, &vec![SENTINEL; np + 64]);
    cp.mem.write_f32s(ATT, &vec![SENTINEL; no + 64]);
    cp.mem.write_u32(SEM, 0);
    let cbuf = args(k).cbuf().unwrap();
    let mut cb = vec![0u8; 0x100];
    cb[..cbuf.len()].copy_from_slice(&cbuf);
    cp.mem.write(CBUF, &cb);
    let q = ComputeQmd {
        program_va: PROG,
        registers: regs,
        cbuf0_va: CBUF,
        cbuf0_size: 0x100,
        grid,
        block,
        release: Some((SEM, 1)),
    };
    cp.mem
        .write(QMD, &q.to_bytes_with_shared(&qmd_shared).unwrap());
    let mut w = [0u32; 16];
    let mut p = PushBuf::new(&mut w);
    p.inc(1, c::NVC7C0_SET_OBJECT, &[c::AMPERE_COMPUTE_B])
        .unwrap();
    p.inc(1, c::NVC7C0_SEND_PCAS_A, &[(QMD >> 8) as u32])
        .unwrap();
    let action = c::NVC7C0_SEND_SIGNALING_PCAS2_B_PCAS_ACTION_INVALIDATE_COPY_SCHEDULE;
    p.inc(1, c::NVC7C0_SEND_SIGNALING_PCAS2_B, &[action])
        .unwrap();
    let n = p.len();
    let bytes: Vec<u8> = w[..n].iter().flat_map(|x| x.to_le_bytes()).collect();
    cp.mem.write(PUSH, &bytes);
    let e = push::gpfifo_entry(PUSH, n as u32, false).unwrap();
    cp.run_gpfifo(&[(e.entry0, e.entry1)])?;
    assert_eq!(cp.mem.read_u32(SEM).unwrap(), 1);
    let pv = cp.mem.read_f32s(P, np + 64).unwrap();
    let av = cp.mem.read_f32s(ATT, no + 64).unwrap();
    assert!(pv[np..].iter().all(|&v| v == SENTINEL), "wrote after P");
    assert!(av[no..].iter().all(|&v| v == SENTINEL), "wrote after att");
    Ok((pv[..np].to_vec(), av[..no].to_vec()))
}

fn run(k: &Case, prog: Prog) -> (Vec<f32>, Vec<f32>) {
    let a = args(k);
    match prog {
        Prog::Tile => launch(k, prog, at::grid(&a), at::BLOCK, at::SHARED).unwrap(),
        Prog::Tile2 => launch(k, prog, at2::grid(&a), at2::BLOCK, at2::SHARED).unwrap(),
    }
}

fn shape(batch: usize, t: usize, c: usize, heads: usize) -> AttnShape {
    AttnShape { batch, t, c, heads }
}

/// T173's 10 shapes (the same seeds and amplitudes).
fn cases() -> Vec<(&'static str, Case)> {
    vec![
        (
            "small model B2 T32 C64 H4",
            Case::new(1, shape(2, 32, 64, 4), 1.5),
        ),
        ("T70 C96 H3 (d 32)", Case::new(2, shape(1, 70, 96, 3), 1.0)),
        ("T33 C80 H2 (d 40)", Case::new(3, shape(2, 33, 80, 2), 1.0)),
        ("T5 C8 H1", Case::new(4, shape(3, 5, 8, 1), 2.0)),
        ("T17 C3 H3 (d 1)", Case::new(5, shape(1, 17, 3, 3), 2.0)),
        ("sharp softmax", Case::new(6, shape(1, 40, 64, 4), 6.0)),
        (
            "very sharp (FTZ range)",
            Case::new(7, shape(1, 24, 16, 1), 30.0),
        ),
        ("T130 C32 H2", Case::new(8, shape(1, 130, 32, 2), 1.0)),
        (
            "#513 head width, T 200 (d 64)",
            Case::new(9, shape(1, 200, 768, 12), 1.0),
        ),
        (
            "T 64 exactly, T 65",
            Case::new(10, shape(2, 65, 128, 2), 1.0),
        ),
    ]
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

fn hex(code: &[u8]) -> String {
    nvidia_ga104::sha256::digest(code)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

const LISTING: &str = include_str!("../../../tools/nak/out/attn_fwd_tile2.sm86.txt");
const LISTING_OLD: &str = include_str!("../../../tools/nak/out/attn_fwd_tile.sm86.txt");

/// The opcode of every instruction line (`r = op ...` or `op ...`, predicate dropped).
fn ops(txt: &str) -> Vec<&str> {
    txt.lines()
        .filter_map(|l| l.split_once("//").map(|(ins, _)| ins))
        .filter_map(|ins| {
            let ins = ins.split_once(" = ").map_or(ins, |(_, r)| r).trim();
            let ins = if ins.starts_with('@') {
                ins.split_once(char::is_whitespace)
                    .map_or("", |(_, r)| r)
                    .trim()
            } else {
                ins
            };
            ins.split_whitespace().next()
        })
        .collect()
}

// Prediction 2 (checked when compiling).
const _: () = assert!(at2::REGISTERS <= 128);
const _: () = assert!(at2::SHARED.bytes == 37_888);
const _: () = assert!(at2::SHARED.bytes <= at2::SHARED.max_bytes);

// ------------------------------------------------------------------ the tests

/// Predictions 1 and 2, from the NAK listing.
#[test]
fn t181_binary_and_listing() {
    assert_eq!(hex(ATTN_FWD_TILE2), at2::ATTN_FWD_TILE2_SHA256);
    assert!(LISTING.contains(&format!("Num GPRs: {}", at2::REGISTERS)));
    let (new, old) = (ops(LISTING), ops(LISTING_OLD));
    let count = |v: &[&str], p: &str| v.iter().filter(|o| o.starts_with(p)).count();
    for (what, v) in [("new", &new), ("old", &old)] {
        assert_eq!(count(v, "atom") + count(v, "red."), 0, "{what}: atomics");
    }
    // The same MUFU operations as attn_fwd_tile (the same exp and 1/sum, nothing else).
    let mufu = |v: &[&str]| {
        let mut m: Vec<String> = v
            .iter()
            .filter(|o| o.starts_with("mufu"))
            .map(|o| o.to_string())
            .collect();
        m.sort();
        m
    };
    assert_eq!(mufu(&new), mufu(&old), "MUFU differs from attn_fwd_tile");
    assert!(!mufu(&new).is_empty());
    // No local memory (no spill of the 16-value load arrays).
    assert!(
        !LISTING.contains("ld.local") && !LISTING.contains("st.local"),
        "local memory"
    );
    // One barrier id.
    assert!(count(&new, "bar.sync") >= 1);
    println!(
        "T181 listing: {} registers (old {}), {} ops (old {}), mufu {:?}",
        at2::REGISTERS,
        at::REGISTERS,
        new.len(),
        old.len(),
        mufu(&new)
    );
}

/// Prediction 4: the launch gives attn_fwd_tile's bits, on all 10 shapes, and the model's.
#[test]
fn t181_tile2_equals_tile_bit_for_bit() {
    for (name, k) in cases() {
        let (p2, a2) = run(&k, Prog::Tile2);
        let (p1, a1) = run(&k, Prog::Tile);
        assert_eq!(bits(&p2), bits(&p1), "{name}: P differs from attn_fwd_tile");
        assert_eq!(
            bits(&a2),
            bits(&a1),
            "{name}: att differs from attn_fwd_tile"
        );
        let (mp, ma) = attn_fwd_tile_gpu(&k.s, &k.qkv, k.scale()).unwrap();
        assert_eq!(bits(&p2), bits(&mp), "{name}: P = the model");
        assert_eq!(bits(&a2), bits(&ma), "{name}: att = the model");
        assert!(
            !p2.iter().chain(&a2).any(|&x| x == SENTINEL),
            "{name}: unwritten"
        );
    }
}

/// Prediction 4: planted mistakes.
#[test]
fn t181_planted_mistakes_are_caught() {
    let k = Case::new(21, shape(2, 130, 64, 2), 1.0);
    let a = args(&k);
    let g = at2::grid(&a);
    assert_eq!(g, [3, 2, 2]);
    // The QMD with attn_fwd_tile's shared size (512 bytes short): refused.
    assert!(launch(&k, Prog::Tile2, g, at2::BLOCK, at::SHARED).is_err());
    assert!(launch(&k, Prog::Tile2, g, at2::BLOCK, at2::SHARED).is_ok());
    // A grid one block short: rows 128, 129 stay sentinels.
    let (p, att) = launch(
        &k,
        Prog::Tile2,
        [g[0] - 1, g[1], g[2]],
        at2::BLOCK,
        at2::SHARED,
    )
    .unwrap();
    let left = p.iter().chain(&att).filter(|&&x| x == SENTINEL).count();
    assert_eq!(left, 2 * 2 * 2 * 130 + 2 * 2 * 64);
    // A wrong block or a grid past ⌈T/64⌉: refused.
    assert!(launch(&k, Prog::Tile2, g, [128, 1, 1], at2::SHARED).is_err());
    assert!(
        launch(
            &k,
            Prog::Tile2,
            [g[0] + 1, g[1], g[2]],
            at2::BLOCK,
            at2::SHARED
        )
        .is_err()
    );
    // d > 64: refused.
    let k65 = Case::new(22, shape(1, 8, 130, 2), 1.0);
    let a65 = args(&k65);
    assert!(!at2::shape_ok(&a65));
    assert!(launch(&k65, Prog::Tile2, at2::grid(&a65), at2::BLOCK, at2::SHARED).is_err());
}

/// Prediction 3, from the listing with tools/buget-antrenare/nak_bucle.py's rule (a loop = a back edge; its own
/// blocks): the inner loops with 16 `ffma` per iteration of 4 (phase A over e, phase C below the diagonal) have at most
/// 28 / 30 instructions per 16 `ffma` and no `bssy`.
#[test]
fn t181_inner_loops_are_short() {
    let loops = inner_loops(LISTING);
    println!("T181 loops (instr, ffma, bssy): {loops:?}");
    let unrolled: Vec<&(usize, usize, usize)> = loops.iter().filter(|l| l.1 == 64).collect();
    assert_eq!(
        unrolled.len(),
        2,
        "two loops with 4 x 16 ffma (A and C below the diagonal)"
    );
    for l in &unrolled {
        assert_eq!(l.2, 0, "bssy in an unrolled loop");
    }
    let per16: Vec<f64> = unrolled
        .iter()
        .map(|l| l.0 as f64 * 16.0 / l.1 as f64)
        .collect();
    assert!(per16.iter().all(|&x| x <= 30.0), "{per16:?}");
    assert!(per16.iter().any(|&x| x <= 28.0), "{per16:?}");
}

/// nak_bucle.py's loops: for every back edge tail → head (head ≤ tail) the natural loop (the head plus the blocks
/// reaching the tail without passing the head); (instr, ffma, bssy) summed over its OWN blocks (minus nested loops).
fn inner_loops(txt: &str) -> Vec<(usize, usize, usize)> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut blocks: BTreeMap<usize, (Vec<String>, Vec<usize>)> = BTreeMap::new();
    let mut cur: Option<usize> = None;
    for line in txt.lines() {
        let t = line.trim_start();
        if (t.starts_with("block ") || t.starts_with("block.u ")) && t.ends_with('{') {
            cur = t.split_whitespace().nth(1).and_then(|x| x.parse().ok());
            if let Some(i) = cur {
                blocks.insert(i, (Vec::new(), Vec::new()));
            }
            continue;
        }
        if let Some(rest) = t.strip_prefix("} -> [") {
            if let Some(i) = cur.take() {
                blocks.get_mut(&i).unwrap().1 = rest
                    .trim_end_matches(']')
                    .split(',')
                    .filter_map(|x| x.trim().parse().ok())
                    .collect();
            }
            continue;
        }
        if let Some(i) = cur
            && line.contains("//")
        {
            blocks.get_mut(&i).unwrap().0.push(line.to_string());
        }
    }
    let mut preds: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (&b, (_, succ)) in &blocks {
        for &t in succ {
            preds.entry(t).or_default().push(b);
        }
    }
    let mut found: Vec<BTreeSet<usize>> = Vec::new();
    for (&tail, (_, succ)) in &blocks {
        for &head in succ {
            if head > tail {
                continue;
            }
            let mut body: BTreeSet<usize> = [head, tail].into();
            let mut stack = vec![tail];
            while let Some(n) = stack.pop() {
                if n == head {
                    continue;
                }
                for &q in preds.get(&n).map_or(&[][..], |v| v) {
                    if body.insert(q) {
                        stack.push(q);
                    }
                }
            }
            found.push(body);
        }
    }
    found
        .iter()
        .map(|body| {
            let inner: BTreeSet<usize> = found
                .iter()
                .filter(|b| b.is_subset(body) && b.len() < body.len())
                .flatten()
                .copied()
                .collect();
            let mut s = (0, 0, 0);
            for i in body.difference(&inner) {
                for ins in &blocks[i].0 {
                    s.0 += 1;
                    s.1 += ins.contains("ffma") as usize;
                    s.2 += ins.contains("bssy") as usize;
                }
            }
            s
        })
        .collect()
}
