//! T192 — C-157 (Adrian, 23 Nov, point a): the RAM ↔ VRAM probe with the board's **copy engine**, SIMULARE.
//! End to end on the driver's real code: golden image, working client, work channel (FakeRm) plus the copy object
//! allocated on it (`user_space::alloc_copy_object`), our page tables with host-RAM PTEs for A and B, VRAM for V1,
//! V2 and the records, one submission from `ram_vram::encode_probe`. The "GPU" is `gpu_sim::cmdproc` with its copy
//! engine model (methods and fields from `clc7b5.h`); the speeds are the test's own simulated numbers, not a
//! measurement. Nothing here touches or proves anything about the real board.

mod common;

use common::{Rig, dma_pages, gpu_runs, rig_with_push, span, sys_read, sys_write};
use gpu_core::error::GpuResult;
use gpu_sim::cmdproc::{CommandProcessor, Event};
use gpu_sim::fake_rm::class as fc;
use nvidia_ga104::gsp::golden::{GoldenMemory, Mapping};
use nvidia_ga104::gsp::user_space::alloc_copy_object;
use nvidia_ga104::mmu::SysMapping;
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::ram_vram::{self as rv, Buffers, Dir};
use nvidia_ga104::transfer;

const SEED: u32 = 0x157;
const FIRST: u32 = 0x900;
/// SIMULARE: the model's copy engine at 12 000 MB/s plus 5 µs per copy (the test's numbers).
const SIM_MB_S: u64 = 12_000;
const SIM_COPY_NS: u64 = 5_000;

struct Setup {
    m: u64,
    a: SysMapping,
    b: SysMapping,
    v1: Mapping,
    v2: Mapping,
    records: Mapping,
}

fn prepare(r: &mut Rig, m: u64) -> Setup {
    let pages = (m / 0x1000) as usize;
    let a = r.mem.alloc_sys_mapped(&dma_pages(1, pages)).unwrap();
    let b = r.mem.alloc_sys_mapped(&dma_pages(2, 2 * pages)).unwrap();
    let mut data = vec![0u8; m as usize];
    transfer::fill_pattern(&mut data, SEED).unwrap();
    sys_write(&mut r.sys, &a, &data);
    sys_write(&mut r.sys, &b, &vec![rv::FILL; 2 * m as usize]);
    let v1 = r.mem.alloc_mapped(m, 0x1000).unwrap();
    let v2 = r.mem.alloc_mapped(m, 0x1000).unwrap();
    let records = r.mem.alloc_mapped(0x1000, 0x1000).unwrap();
    common::put_bytes(r.mem.writer_mut(), records.pa, &[0; 0x1000]);
    Setup {
        m,
        a,
        b,
        v1,
        v2,
        records,
    }
}

fn bufs(s: &Setup) -> Buffers {
    Buffers {
        a: s.a.va,
        b: s.b.va,
        v1: s.v1.va,
        v2: s.v2.va,
        records: s.records.va,
    }
}

fn words(s: &Setup) -> Vec<u32> {
    let mut w = vec![0u32; rv::probe_words()];
    let mut pb = PushBuf::new(&mut w);
    rv::encode_probe(&mut pb, &bufs(s), s.m, FIRST).unwrap();
    assert_eq!(pb.len(), rv::probe_words());
    w
}

fn run(r: &mut Rig, s: &Setup, w: &[u32]) -> GpuResult<CommandProcessor> {
    let mut cp = CommandProcessor::new();
    cp.sim_launch_ns = 250_000;
    cp.sim_method_ns = 100;
    cp.sim_ce_mb_s = SIM_MB_S;
    cp.sim_ce_copy_ns = SIM_COPY_NS;
    r.cp = Some(cp);
    r.wq.submit(r.mem.writer_mut(), w).unwrap();
    let loads = [
        span(&s.v1),
        span(&s.v2),
        span(&s.records),
        (s.a.va, s.a.size),
        (s.b.va, s.b.size),
    ];
    let stores = [
        span(&s.v1),
        span(&s.v2),
        (s.records.va, rv::RECORD_BYTES),
        (s.b.va, s.b.size),
    ];
    gpu_runs(r, &[], &loads, &stores)
}

fn records(r: &mut Rig, s: &Setup) -> Vec<u8> {
    common::get_bytes(r.mem.writer_mut(), s.records.pa, rv::RECORD_BYTES)
}

fn probe(r: &mut Rig, s: &Setup) -> GpuResult<rv::Probe> {
    let times = rv::analyze(&records(r, s), s.m, FIRST)?;
    let b = sys_read(&r.sys, &s.b, 2 * s.m);
    let a = sys_read(&r.sys, &s.a, s.m);
    Ok(rv::Probe {
        max_bytes: s.m,
        times,
        bytes: rv::judge(&b, &a, s.m, SEED)?,
    })
}

#[test]
fn the_copy_object_is_allocated_on_the_work_channel() {
    let mut r = rig_with_push(0x4000);
    let (h, next) = alloc_copy_object(&mut r.rm, &r.u, &r.w).unwrap();
    assert_eq!(h, r.w.next_handle);
    assert_eq!(next, h + 1);
    let _ = fc::DMA_COPY;
}

#[test]
fn probe_passes_bit_for_bit_with_the_simulated_times() {
    for m in [rv::MAX_BYTES_UNIT, 4 * rv::MAX_BYTES_UNIT] {
        let mut r = rig_with_push(0x4000);
        alloc_copy_object(&mut r.rm, &r.u, &r.w).unwrap();
        let s = prepare(&mut r, m);
        let w = words(&s);
        let cp = run(&mut r, &s, &w).unwrap();
        // 1 channel record + 30 copy-engine records; 10 × 3 groups: 24 single copies + 6 × 4 parallel ones.
        assert_eq!(cp.timestamps.len(), rv::RECORDS);
        let copies: Vec<_> =
            r.cp.as_ref()
                .unwrap()
                .events
                .iter()
                .filter_map(|e| match e {
                    Event::CeCopy { pipelined, .. } => Some(*pipelined),
                    _ => None,
                })
                .collect();
        assert_eq!(copies.len(), 24 + 6 * 4);
        assert_eq!(copies.iter().filter(|p| **p).count(), 6 * 3);
        let p = probe(&mut r, &s).unwrap();
        assert!(p.passes(), "{:?}", p.lines("SIMULARE"));
        // The SIMULARE clock: bytes at 12 000 MB/s + 5 µs per copy + 100 ns per method (9 for a copy with its
        // record, 6 for one without: 4 addresses, the length, LAUNCH_DMA).
        for t in &p.times {
            let copies = if t.group.parallel { rv::PARALLEL } else { 1 };
            let methods = 9 + (copies - 1) * 6;
            let want = t.group.bytes * 1000 / SIM_MB_S + copies * SIM_COPY_NS + methods * 100;
            let got = t.median_ns();
            assert!(
                got.abs_diff(want) <= 20 + copies * 2,
                "{:?}: {got} fata de {want}",
                t.group
            );
            assert_eq!(t.ns[0], t.ns[1]);
        }
        let up = p.max_mb_s(Dir::Up).unwrap();
        assert!((9_000..12_000).contains(&up), "{up}");
        // The rows go through the journal text and back.
        let text: String = p
            .rows()
            .iter()
            .map(|(w, v)| format!("gpu: valoare {w} = {v:#x}\n"))
            .collect();
        let back = rv::parse_ramvram(&text);
        assert_eq!(back.verdict, Some(0));
        assert_eq!(back.per_group.len(), rv::GROUPS);
        assert_eq!(back.max_kib, Some((m >> 10) as u32));
    }
}

#[test]
fn a_wrong_parallel_copy_is_seen_in_the_second_half_of_b() {
    let mut r = rig_with_push(0x4000);
    alloc_copy_object(&mut r.rm, &r.u, &r.w).unwrap();
    let s = prepare(&mut r, rv::MAX_BYTES_UNIT);
    // A planted fault: the last pipelined quarter down lands 16 bytes off (OFFSET_OUT_LOWER of the last copy).
    let mut w = words(&s);
    let q = s.m / rv::PARALLEL;
    let target = (s.b.va + s.m + 3 * q) as u32;
    let at = w.iter().rposition(|&x| x == target).unwrap();
    w[at] += 16;
    run(&mut r, &s, &w).unwrap();
    let p = probe(&mut r, &s).unwrap();
    assert_eq!(p.verdict(), 1);
    let (off, _, _) = p.bytes.mismatch.unwrap();
    assert!(off >= s.m + 3 * q, "{off:#x}");
}

#[test]
fn a_missing_record_and_unknown_bits_stop_cleanly() {
    // The pushbuffer cut before the last group: record 30 never comes.
    let mut r = rig_with_push(0x4000);
    alloc_copy_object(&mut r.rm, &r.u, &r.w).unwrap();
    let s = prepare(&mut r, rv::MAX_BYTES_UNIT);
    let w = words(&s);
    let cut = w.len() - (rv::copy_words(true) + 3 * rv::copy_words(false));
    run(&mut r, &s, &w[..cut]).unwrap();
    assert!(rv::analyze(&records(&mut r, &s), s.m, FIRST).is_err());
    assert_eq!(
        transfer::records_seen(&records(&mut r, &s), rv::RECORDS - 1, FIRST),
        rv::RECORDS - 1
    );
    // A LAUNCH_DMA with a bit the model does not know (MULTI_LINE_ENABLE) is refused by the GPU model.
    let mut r = rig_with_push(0x4000);
    let s = prepare(&mut r, rv::MAX_BYTES_UNIT);
    let mut w = words(&s);
    let ld = rv::launch_dma(false, true);
    let at = w.iter().position(|&x| x == ld).unwrap();
    w[at] |= 1 << 9;
    assert!(run(&mut r, &s, &w).is_err());
}
