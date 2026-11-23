//! C-160 point 7 — T186, SIMULARE: `attn_p_tile` (`tools/nak/attn_p_tile.comp`, Mesa NAK 25.0.7;
//! `docs/C-160_P7_ATTN_P_TILE.md`) = `attn_fwd_tile2` without P·V. The plan and the predictions were written and pushed
//! BEFORE this code (`e80ffbe`):
//!   1. registers ≤ 117, fewer instructions than `attn_fwd_tile2`, 37 888 shared bytes, 1 barrier, no local memory, no
//!      atomics;
//!   2. the same `mufu` as `attn_fwd_tile2`, every floating-point opcode also in `attn_fwd_tile2`, `att` (cbuf 0x38)
//!      never read;
//!   3. P = `attn_fwd_tile`'s and `attn_fwd_tile2`'s bit for bit on T173 / T181's 10 shapes, `att` untouched;
//!   4. planted mistakes caught (old shared size, a grid one block short, a wrong block, d > 64, a wrong scale).
//!
//! What the simulation can and cannot show: the "program" runs on the CPU (`attn_p_tile_on_cpu` →
//! `gpu_compute::layer::attn_p_tile_row_gpu`, the P part of `attn_fwd_tile_row_gpu`), so point 3 checks the driver side
//! (QMD, grid, block, cbuf, which rows get written, nothing in `att`), not the NAK code; points 1–2 are read from the NAK
//! listing. That the board gives `attn_fwd_tile`'s P bits is a prediction. Nothing here runs on the RTX 3070 Ti.

use gpu_compute::layer::{AttnShape, attn_fwd_tile_gpu, attn_p_tile_gpu};
use gpu_sim::cmdproc::{
    CommandProcessor, Needs, ProgramFn, attn_fwd_tile_on_cpu, attn_p_tile_on_cpu,
};
use nvidia_ga104::attn_back as ab;
use nvidia_ga104::attn_fwd as af;
use nvidia_ga104::attn_fwd_tile::{self as at, ATTN_FWD_TILE};
use nvidia_ga104::attn_fwd_tile2::{self as at2, ATTN_FWD_TILE2};
use nvidia_ga104::attn_p_tile::{self as apt, ATTN_P_TILE};
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
    PTile,
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
    let (code, regs, own, model): (_, _, _, ProgramFn) = match prog {
        Prog::Tile => (
            ATTN_FWD_TILE,
            at::REGISTERS,
            at::SHARED,
            attn_fwd_tile_on_cpu,
        ),
        Prog::Tile2 => (
            ATTN_FWD_TILE2,
            at2::REGISTERS,
            at2::SHARED,
            attn_fwd_tile_on_cpu,
        ),
        Prog::PTile => (ATTN_P_TILE, apt::REGISTERS, apt::SHARED, attn_p_tile_on_cpu),
    };
    cp.register_program_with_needs(
        code,
        model,
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
        Prog::PTile => launch(k, prog, apt::grid(&a), apt::BLOCK, apt::SHARED).unwrap(),
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

const LISTING: &str = include_str!("../../../tools/nak/out/attn_p_tile.sm86.txt");
const LISTING_TILE2: &str = include_str!("../../../tools/nak/out/attn_fwd_tile2.sm86.txt");

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

// Prediction 1 (checked when compiling).
const _: () = assert!(apt::REGISTERS <= at2::REGISTERS);
const _: () = assert!(apt::SHARED.bytes == 37_888);
const _: () = assert!(apt::SHARED.barriers == 1);

/// The floating-point opcodes (with their modifiers) of a listing, sorted and deduplicated.
fn fp_kinds(v: &[&str]) -> Vec<String> {
    let mut k: Vec<String> = v
        .iter()
        .filter(|o| {
            ["ffma", "fmul", "fadd", "fmnmx", "mufu", "fsetp", "fsel"]
                .iter()
                .any(|p| o.starts_with(p))
        })
        .map(|o| o.to_string())
        .collect();
    k.sort();
    k.dedup();
    k
}

// ------------------------------------------------------------------ the tests

/// Predictions 1 and 2, from the NAK listing.
#[test]
fn t186_binary_and_listing() {
    assert_eq!(hex(ATTN_P_TILE), apt::ATTN_P_TILE_SHA256);
    assert!(LISTING.contains(&format!("Num GPRs: {}", apt::REGISTERS)));
    let (new, t2) = (ops(LISTING), ops(LISTING_TILE2));
    let count = |v: &[&str], p: &str| v.iter().filter(|o| o.starts_with(p)).count();
    assert_eq!(count(&new, "atom") + count(&new, "red."), 0, "atomics");
    assert!(
        new.len() < t2.len(),
        "not shorter: {} vs {}",
        new.len(),
        t2.len()
    );
    let mufu = |v: &[&str]| {
        let mut m: Vec<String> = v
            .iter()
            .filter(|o| o.starts_with("mufu"))
            .map(|o| o.to_string())
            .collect();
        m.sort();
        m
    };
    assert_eq!(mufu(&new), mufu(&t2), "MUFU differs from attn_fwd_tile2");
    assert!(!mufu(&new).is_empty());
    let (kn, k2) = (fp_kinds(&new), fp_kinds(&t2));
    for k in &kn {
        assert!(k2.contains(k), "{k} is not in attn_fwd_tile2");
    }
    assert!(
        !LISTING.contains("ld.local") && !LISTING.contains("st.local"),
        "local memory"
    );
    assert!(count(&new, "bar.sync") >= 1);
    // att's address (cbuf 0x38 / 0x3c) is never read; qkv (0x28) and p (0x30) are.
    assert!(!LISTING.contains("c[0x0][0x38]") && !LISTING.contains("c[0x0][0x3c]"));
    assert!(LISTING_TILE2.contains("c[0x0][0x38]"));
    assert!(LISTING.contains("c[0x0][0x28]") && LISTING.contains("c[0x0][0x30]"));
    println!(
        "T186 listing: {} registers (tile2 {}), {} ops (tile2 {}), mufu {:?}, fp {:?}",
        apt::REGISTERS,
        at2::REGISTERS,
        new.len(),
        t2.len(),
        mufu(&new),
        kn
    );
}

/// Prediction 3: P = attn_fwd_tile's and attn_fwd_tile2's bit for bit on the 10 shapes, att untouched.
#[test]
fn t186_p_equals_tile_bit_for_bit() {
    for (name, k) in cases() {
        let (pp, ap) = run(&k, Prog::PTile);
        let (p1, _) = run(&k, Prog::Tile);
        let (p2, _) = run(&k, Prog::Tile2);
        assert_eq!(bits(&pp), bits(&p1), "{name}: P differs from attn_fwd_tile");
        assert_eq!(
            bits(&pp),
            bits(&p2),
            "{name}: P differs from attn_fwd_tile2"
        );
        let (mp, _) = attn_fwd_tile_gpu(&k.s, &k.qkv, k.scale()).unwrap();
        assert_eq!(bits(&pp), bits(&mp), "{name}: P = the model");
        assert_eq!(
            bits(&attn_p_tile_gpu(&k.s, &k.qkv, k.scale()).unwrap()),
            bits(&mp),
            "{name}: attn_p_tile_gpu"
        );
        assert!(!pp.contains(&SENTINEL), "{name}: P unwritten");
        assert!(ap.iter().all(|&x| x == SENTINEL), "{name}: att touched");
    }
}

/// Prediction 3, the re-launch of the step: attn_fwd_tile writes P and att, P is overwritten, attn_p_tile rebuilds it —
/// P comes back bit for bit and att stays what attn_fwd_tile left.
#[test]
fn t186_rebuilds_an_overwritten_p() {
    let k = Case::new(31, shape(2, 130, 128, 2), 1.0);
    let (_, no, np) = k.s.sizes().unwrap();
    let a = args(&k);
    let mut cp = CommandProcessor::new();
    for (code, model, own) in [
        (ATTN_FWD_TILE, attn_fwd_tile_on_cpu as ProgramFn, at::SHARED),
        (ATTN_P_TILE, attn_p_tile_on_cpu, apt::SHARED),
    ] {
        cp.register_program_with_needs(
            code,
            model,
            Needs {
                shared_bytes: own.bytes,
                barriers: u32::from(own.barriers),
            },
        );
    }
    cp.mem.write(PROG, ATTN_FWD_TILE);
    cp.mem.write(PROG + 0x8_0000, ATTN_P_TILE);
    cp.mem.write_f32s(QKV, &k.qkv);
    cp.mem.write_f32s(P, &vec![SENTINEL; np]);
    cp.mem.write_f32s(ATT, &vec![SENTINEL; no]);
    let cbuf = a.cbuf().unwrap();
    let mut cb = vec![0u8; 0x100];
    cb[..cbuf.len()].copy_from_slice(&cbuf);
    cp.mem.write(CBUF, &cb);
    let go = |cp: &mut CommandProcessor, prog: u64, regs: u32, own: SharedMemory, sem: u32| {
        let q = ComputeQmd {
            program_va: prog,
            registers: regs,
            cbuf0_va: CBUF,
            cbuf0_size: 0x100,
            grid: apt::grid(&a),
            block: apt::BLOCK,
            release: Some((SEM, sem)),
        };
        cp.mem.write(QMD, &q.to_bytes_with_shared(&own).unwrap());
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
        cp.run_gpfifo(&[(e.entry0, e.entry1)]).unwrap();
        assert_eq!(cp.mem.read_u32(SEM).unwrap(), sem);
    };
    go(&mut cp, PROG, at::REGISTERS, at::SHARED, 1);
    let p0 = cp.mem.read_f32s(P, np).unwrap();
    let a0 = cp.mem.read_f32s(ATT, no).unwrap();
    cp.mem.write_f32s(P, &vec![f32::NAN; np]);
    go(&mut cp, PROG + 0x8_0000, apt::REGISTERS, apt::SHARED, 2);
    assert_eq!(
        bits(&cp.mem.read_f32s(P, np).unwrap()),
        bits(&p0),
        "P not rebuilt"
    );
    assert_eq!(
        bits(&cp.mem.read_f32s(ATT, no).unwrap()),
        bits(&a0),
        "att changed"
    );
}

/// Prediction 4: planted mistakes.
#[test]
fn t186_planted_mistakes_are_caught() {
    let k = Case::new(21, shape(2, 130, 64, 2), 1.0);
    let a = args(&k);
    let g = apt::grid(&a);
    assert_eq!(g, [3, 2, 2]);
    // The QMD with attn_fwd_tile's shared size (512 bytes short): refused.
    assert!(launch(&k, Prog::PTile, g, apt::BLOCK, at::SHARED).is_err());
    assert!(launch(&k, Prog::PTile, g, apt::BLOCK, apt::SHARED).is_ok());
    // A grid one block short: rows 128, 129 of every (b, h) stay sentinels; att untouched either way.
    let (p, att) = launch(
        &k,
        Prog::PTile,
        [g[0] - 1, g[1], g[2]],
        apt::BLOCK,
        apt::SHARED,
    )
    .unwrap();
    assert_eq!(
        p.iter().filter(|&&x| x == SENTINEL).count(),
        2 * 2 * 2 * 130
    );
    assert!(att.iter().all(|&x| x == SENTINEL));
    // A wrong block or a grid past ⌈T/64⌉: refused.
    assert!(launch(&k, Prog::PTile, g, [128, 1, 1], apt::SHARED).is_err());
    assert!(
        launch(
            &k,
            Prog::PTile,
            [g[0] + 1, g[1], g[2]],
            apt::BLOCK,
            apt::SHARED
        )
        .is_err()
    );
    // d > 64: refused.
    let k65 = Case::new(22, shape(1, 8, 130, 2), 1.0);
    let a65 = args(&k65);
    assert!(!apt::shape_ok(&a65));
    assert!(launch(&k65, Prog::PTile, apt::grid(&a65), apt::BLOCK, apt::SHARED).is_err());
    // A wrong scale (the d of another model) changes P.
    let good = run(&k, Prog::PTile).0;
    let mut bad = attn_p_tile_gpu(&k.s, &k.qkv, ab::Args::scale_for(48)).unwrap();
    assert_ne!(bits(&bad), bits(&good));
    bad = attn_p_tile_gpu(&k.s, &k.qkv, k.scale()).unwrap();
    assert_eq!(bits(&bad), bits(&good));
}

/// Prediction 5 (DEDUS), from the listing: on the tiles below the diagonal attn_p_tile keeps only phase A's unrolled
/// loop (4 x 16 `ffma`); attn_fwd_tile2 has two (A and C). Prints the instruction counts of both.
#[test]
fn t186_one_unrolled_loop_left() {
    let new = inner_loops(LISTING);
    let t2 = inner_loops(LISTING_TILE2);
    println!("T186 loops (instr, ffma, bssy): attn_p_tile {new:?}; attn_fwd_tile2 {t2:?}");
    assert_eq!(new.iter().filter(|l| l.1 == 64).count(), 1);
    assert_eq!(t2.iter().filter(|l| l.1 == 64).count(), 2);
    let a_new = new.iter().find(|l| l.1 == 64).unwrap();
    assert_eq!(a_new.2, 0, "bssy in phase A");
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
