//! C-147 step 3, SIMULARE: `q8_matvec32` (all 32 threads, 128-bit x loads) on the work channel, end to end on the
//! driver's code, the same sequence as I12 (`sim_i12_q8.rs`): copy RAM→VRAM → `q8_matvec32` → copy y VRAM→RAM, each
//! followed by a timestamp record; y judged by `q8_matvec32::judge` (bit for bit with its own oracle). Plus: the
//! result is within I12's margin of `q8_matvec`'s, and on Kronos #513's shapes it is a different rounding (so the
//! board will show which program ran). The "GPU" is `gpu_sim::cmdproc`; nothing here proves anything about the board.

mod common;

use common::{Rig, dma_pages, gpu_runs, put_bytes, rig, span, sys_read, sys_write};
use gpu_compute::q8;
use gpu_core::error::GpuResult;
use gpu_sim::cmdproc::{
    CommandProcessor, Needs, ProgramFn, Q8_MATVEC_NEEDS, Q8_MATVEC32_NEEDS, copy_b128_on_cpu,
    q8_matvec_on_cpu, q8_matvec32_on_cpu,
};
use nvidia_ga104::gsp::golden::{GoldenMemory, Mapping};
use nvidia_ga104::mmu::SysMapping;
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::q8_matvec::{self as mv, Q8_MATVEC, Verdict};
use nvidia_ga104::q8_matvec32::{self as w32, Q8_MATVEC32};
use nvidia_ga104::transfer::{self, COPY_B128};

const FIRST: u32 = 0x700;

struct Job {
    rows: usize,
    w: Vec<u8>,
    x: Vec<f32>,
    /// W then x, in host RAM and in VRAM (same layout, 16-aligned parts).
    up_bytes: u64,
    x_off: u64,
    y_bytes: u64,
    host_in: SysMapping,
    host_y: SysMapping,
    vin: Mapping,
    vy: Mapping,
    progs: [Mapping; 2],
    cbufs: [Mapping; 3],
    qmds: [Mapping; 3],
    records: Mapping,
}

fn prepare(r: &mut Rig, rows: usize, cols: usize, seed: u32) -> Job {
    let (w, x) = mv::test_matrix(rows, cols, seed).unwrap();
    let x_off = (w.len() as u64).next_multiple_of(16);
    let up_bytes = (x_off + 4 * cols as u64).next_multiple_of(16);
    let y_bytes = (4 * rows as u64).next_multiple_of(16);
    let host_in = r
        .mem
        .alloc_sys_mapped(&dma_pages(1, up_bytes.div_ceil(0x1000) as usize))
        .unwrap();
    let host_y = r
        .mem
        .alloc_sys_mapped(&dma_pages(2, y_bytes.div_ceil(0x1000) as usize))
        .unwrap();
    let mut data = w.clone();
    data.resize(x_off as usize, 0);
    data.extend(x.iter().flat_map(|v| v.to_le_bytes()));
    data.resize(up_bytes as usize, 0);
    sys_write(&mut r.sys, &host_in, &data);
    sys_write(&mut r.sys, &host_y, &vec![0xee; y_bytes as usize]);
    let vin = r.mem.alloc_mapped(up_bytes, 0x1000).unwrap();
    let vy = r.mem.alloc_mapped(y_bytes, 0x1000).unwrap();
    let progs =
        [COPY_B128, Q8_MATVEC32].map(|p| r.mem.alloc_mapped(p.len() as u64, 0x1000).unwrap());
    let mut m = || r.mem.alloc_mapped(0x1000, 0x1000).unwrap();
    let cbufs = [m(), m(), m()];
    let qmds = [m(), m(), m()];
    let records = m();
    let wr = r.mem.writer_mut();
    put_bytes(wr, progs[0].pa, COPY_B128);
    put_bytes(wr, progs[1].pa, Q8_MATVEC32);
    put_bytes(
        wr,
        cbufs[0].pa,
        &transfer::copy_cbuf(host_in.va, vin.va, up_bytes).unwrap(),
    );
    put_bytes(
        wr,
        cbufs[1].pa,
        &w32::cbuf(vin.va, vin.va + x_off, vy.va, rows, cols).unwrap(),
    );
    put_bytes(
        wr,
        cbufs[2].pa,
        &transfer::copy_cbuf(vy.va, host_y.va, y_bytes).unwrap(),
    );
    let up = transfer::copy_qmd(progs[0].va, cbufs[0].va, up_bytes).unwrap();
    put_bytes(wr, qmds[0].pa, &up.to_bytes().unwrap());
    put_bytes(
        wr,
        qmds[1].pa,
        &w32::qmd(progs[1].va, cbufs[1].va, rows).unwrap(),
    );
    let down = transfer::copy_qmd(progs[0].va, cbufs[2].va, y_bytes).unwrap();
    put_bytes(wr, qmds[2].pa, &down.to_bytes().unwrap());
    // y in VRAM starts as junk (NaN), so a row nobody wrote cannot pass.
    put_bytes(wr, vy.pa, &vec![0xff; y_bytes as usize]);
    put_bytes(wr, records.pa, &[0; 64]);
    Job {
        rows,
        w,
        x,
        up_bytes,
        x_off,
        y_bytes,
        host_in,
        host_y,
        vin,
        vy,
        progs,
        cbufs,
        qmds,
        records,
    }
}

fn submit(r: &mut Rig, j: &Job) {
    let mut words = vec![0u32; transfer::sequence_words(3)];
    let mut pb = PushBuf::new(&mut words);
    let q = [j.qmds[0].va, j.qmds[1].va, j.qmds[2].va];
    transfer::encode_sequence(&mut pb, &q, j.records.va, FIRST).unwrap();
    r.wq.submit(r.mem.writer_mut(), &words).unwrap();
}

fn gpu(r: &mut Rig, j: &Job) -> GpuResult<CommandProcessor> {
    gpu_with(r, j, q8_matvec32_on_cpu)
}

/// The planted mistake 5: the GPU model runs `q8_matvec`'s order for the new program.
fn gpu_old_order(r: &mut Rig, j: &Job) -> GpuResult<CommandProcessor> {
    gpu_with(r, j, q8_matvec_on_cpu)
}

fn gpu_with(r: &mut Rig, j: &Job, model: ProgramFn) -> GpuResult<CommandProcessor> {
    let mut loads: Vec<(u64, u64)> = j
        .progs
        .iter()
        .chain(&j.cbufs)
        .chain(&j.qmds)
        .chain([&j.records, &j.vin, &j.vy])
        .map(span)
        .collect();
    loads.push((j.host_in.va, j.host_in.size));
    loads.push((j.host_y.va, j.host_y.size));
    let stores = [
        span(&j.vin),
        span(&j.vy),
        (j.host_y.va, j.host_y.size),
        (j.records.va, 64),
    ];
    gpu_runs(
        r,
        &[
            (COPY_B128, copy_b128_on_cpu, Needs::default()),
            (Q8_MATVEC32, model, Q8_MATVEC32_NEEDS),
        ],
        &loads,
        &stores,
    )
}

/// What the driver reads back: y from host RAM, the 4 records from VRAM.
fn y_and_records(r: &mut Rig, j: &Job) -> (Vec<f32>, Vec<u8>) {
    let y = sys_read(&r.sys, &j.host_y, 4 * j.rows as u64)
        .chunks(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let rec = common::get_bytes(r.mem.writer_mut(), j.records.pa, 64);
    (y, rec)
}

#[test]
fn matvec32_on_the_work_channel_is_identical_to_its_oracle() {
    for (rows, cols) in [
        (1, 32),
        (100, 96),
        (37, 1056),
        (mv::TEST_ROWS, mv::TEST_COLS),
        // 768 × 3072 = 651 upload pages: found the rig's `dma_pages` repeating pages (fixed, session 121).
        (768, 3072),
    ] {
        let mut r = rig();
        let j = prepare(&mut r, rows, cols, 0x12);
        submit(&mut r, &j);
        let cp = gpu(&mut r, &j).unwrap();
        assert_eq!(cp.timestamps.len(), 4);
        let (y, rec) = y_and_records(&mut r, &j);
        let v = w32::judge(&j.w, rows, cols, &j.x, &y).unwrap();
        assert_eq!(v, Verdict::Identic, "{rows}x{cols}: {}", v.line());
        let t = transfer::analyze_sequence(&rec, 3, FIRST).unwrap();
        assert_eq!(t.len(), 3);
        assert_eq!(j.y_bytes % 16, 0);
        // Within I12's margin of q8_matvec's result and of the CPU engine's.
        let (mut old, mut cpu) = (vec![0.0f32; rows], vec![0.0f32; rows]);
        q8::matvec_gpu_order(&j.w, rows, cols, &j.x, &mut old).unwrap();
        q8::matvec_cpu(&j.w, rows, cols, &j.x, &mut cpu).unwrap();
        let m = q8::margin(&j.w, rows, cols, &j.x).unwrap();
        assert_eq!(q8::first_outside_margin(&y, &old, &m), None);
        assert_eq!(q8::first_outside_margin(&y, &cpu, &m), None);
        // On Kronos's shapes the two programs round differently somewhere: the judge of q8_matvec
        // sees the new result as "in marja", never "identic" (so the board shows which one ran).
        if cols >= 768 {
            assert!(matches!(
                mv::judge(&j.w, rows, cols, &j.x, &y).unwrap(),
                Verdict::InMarja { .. }
            ));
        }
    }
}

/// C-139: under the GPU model's non-coherent read cache, still the oracle bit for bit.
#[test]
fn matvec32_passes_under_the_stale_read_model() {
    for line in [4u64, 128, 1024] {
        let mut r = rig();
        r.stale_line = line;
        let j = prepare(&mut r, 100, 768, 0x12);
        submit(&mut r, &j);
        gpu(&mut r, &j).unwrap();
        let (y, _) = y_and_records(&mut r, &j);
        let v = w32::judge(&j.w, 100, 768, &j.x, &y).unwrap();
        assert_eq!(v, Verdict::Identic, "linie {line}: {}", v.line());
    }
}

/// Mistakes planted in what the driver writes; none passes.
#[test]
fn planted_mistakes_never_pass() {
    // 1. x 4 bytes off (not 16-aligned): the driver refuses to build the cbuf; written by hand, the GPU model
    //    refuses the launch and the record is missing.
    let mut r = rig();
    let j = prepare(&mut r, 64, 256, 3);
    assert!(w32::cbuf(j.vin.va, j.vin.va + j.x_off + 4, j.vy.va, 64, 256).is_err());
    let cb = mv::q8_cbuf(j.vin.va, j.vin.va + j.x_off + 4, j.vy.va, 64, 256).unwrap();
    put_bytes(r.mem.writer_mut(), j.cbufs[1].pa, &cb);
    submit(&mut r, &j);
    assert!(gpu(&mut r, &j).is_err());
    let (_, rec) = y_and_records(&mut r, &j);
    assert!(transfer::analyze_sequence(&rec, 3, FIRST).is_err());

    // 2. x one quarter block too far (16-aligned, wrong offset): wrong numbers, named.
    let mut r = rig();
    let j = prepare(&mut r, 64, 256, 3);
    let cb = w32::cbuf(j.vin.va, j.vin.va + j.x_off + 32, j.vy.va, 64, 256).unwrap();
    put_bytes(r.mem.writer_mut(), j.cbufs[1].pa, &cb);
    submit(&mut r, &j);
    gpu(&mut r, &j).unwrap();
    let (y, _) = y_and_records(&mut r, &j);
    assert!(!w32::judge(&j.w, 64, 256, &j.x, &y).unwrap().passes());

    // 3. A grid one row short: the last row is left as junk → named.
    let mut r = rig();
    let j = prepare(&mut r, 64, 256, 3);
    put_bytes(
        r.mem.writer_mut(),
        j.qmds[1].pa,
        &w32::qmd(j.progs[1].va, j.cbufs[1].va, 63).unwrap(),
    );
    submit(&mut r, &j);
    gpu(&mut r, &j).unwrap();
    let (y, _) = y_and_records(&mut r, &j);
    match w32::judge(&j.w, 64, 256, &j.x, &y).unwrap() {
        Verdict::NuTrece { row, .. } => assert_eq!(row, 63),
        v => panic!("{v:?}"),
    }

    // 4. One number changed in host RAM after the oracle was computed: its row fails.
    let mut r = rig();
    let j = prepare(&mut r, 16, 64, 3);
    let off = 5 * 2 * 36 + 36 + 4 + 17; // row 5, block 1, number 17 (quarter 2)
    let mut data = sys_read(&r.sys, &j.host_in, j.up_bytes);
    data[off] = data[off].wrapping_add(100);
    sys_write(&mut r.sys, &j.host_in, &data);
    submit(&mut r, &j);
    gpu(&mut r, &j).unwrap();
    let (y, _) = y_and_records(&mut r, &j);
    match w32::judge(&j.w, 16, 64, &j.x, &y).unwrap() {
        Verdict::NuTrece { row, .. } => assert_eq!(row, 5),
        v => panic!("{v:?}"),
    }

    // 5. The GPU model running the OLD program's order for the new program: not "identic" on #513's shape.
    let mut r = rig();
    let j = prepare(&mut r, 64, 768, 3);
    submit(&mut r, &j);
    gpu_old_order(&mut r, &j).unwrap();
    let (y, _) = y_and_records(&mut r, &j);
    assert!(matches!(
        w32::judge(&j.w, 64, 768, &j.x, &y).unwrap(),
        Verdict::InMarja { .. }
    ));
}

/// C-147 step 3, the race in SIMULARE: two scaled-down lanes (the board runs `K513_LANES`), A B B A heats of
/// `launch_cost` launches with records, y judged after each heat by the entrant's own judge. The simulated
/// clock gives every launch the same time, so the method is checked, never a winner: the tie keeps `q8_matvec`.
#[test]
fn the_race_runs_on_the_work_channel() {
    use nvidia_ga104::launch_cost;
    use nvidia_ga104::q8_race::{self as race, Entrant, Lane, Standing, Winner};
    let lanes = [
        Lane {
            name: "qkv",
            rows: 96,
            cols: 768,
            per_token: 12,
        },
        Lane {
            name: "proj",
            rows: 48,
            cols: 3072,
            per_token: 12,
        },
    ];
    let mut standing = Standing::default();
    let mut text = String::new();
    for (li, lane) in lanes.iter().enumerate() {
        let mut r = rig();
        let j = prepare(&mut r, lane.rows, lane.cols, 0x147);
        submit(&mut r, &j);
        gpu_all(&mut r, &j).unwrap(); // the upload (and one q8_matvec32 run)
        let old = r.mem.alloc_mapped(Q8_MATVEC.len() as u64, 0x1000).unwrap();
        put_bytes(r.mem.writer_mut(), old.pa, Q8_MATVEC);
        let rec = r.mem.alloc_mapped(0x1000, 0x1000).unwrap();
        for (h, heat) in race::heats(lanes.len(), 4)
            .into_iter()
            .filter(|h| h.lane == li)
            .enumerate()
        {
            let prog = match heat.entrant {
                Entrant::Q8Matvec => old.va,
                Entrant::Q8Matvec32 => j.progs[1].va,
            };
            let wr = r.mem.writer_mut();
            let cb = heat
                .entrant
                .cbuf(j.vin.va, j.vin.va + j.x_off, j.vy.va, lane.rows, lane.cols)
                .unwrap();
            put_bytes(wr, j.cbufs[1].pa, &cb);
            put_bytes(
                wr,
                j.qmds[1].pa,
                &heat.entrant.qmd(prog, j.cbufs[1].va, lane.rows).unwrap(),
            );
            put_bytes(wr, j.vy.pa, &vec![0xff; j.y_bytes as usize]); // junk before every heat
            put_bytes(wr, rec.pa, &[0; 256]);
            let first = 0x900 + 0x10 * h as u32;
            let mut words = vec![0u32; 512];
            let mut p = PushBuf::new(&mut words);
            race::encode_heat(&mut p, j.qmds[1].va, rec.va, &heat, first).unwrap();
            let n = p.len();
            r.wq.submit(r.mem.writer_mut(), &words[..n]).unwrap();
            let mut loads: Vec<(u64, u64)> = vec![span(&old), span(&rec)];
            loads.extend(
                j.progs
                    .iter()
                    .chain(&j.cbufs)
                    .chain(&j.qmds)
                    .chain([&j.vin, &j.vy])
                    .map(span),
            );
            gpu_runs(
                &mut r,
                &[
                    (COPY_B128, copy_b128_on_cpu, Needs::default()),
                    (Q8_MATVEC32, q8_matvec32_on_cpu, Q8_MATVEC32_NEEDS),
                    (Q8_MATVEC, q8_matvec_on_cpu, Q8_MATVEC_NEEDS),
                ],
                &loads,
                &[span(&j.vy), (rec.va, 256)],
            )
            .unwrap();
            let bytes = common::get_bytes(r.mem.writer_mut(), rec.pa, 16 * (heat.n as u64 + 1));
            let cost = launch_cost::analyze(&bytes, heat.n, first).unwrap();
            let y: Vec<f32> = common::get_bytes(r.mem.writer_mut(), j.vy.pa, 4 * lane.rows as u64)
                .chunks(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let v = heat
                .entrant
                .judge(&j.w, lane.rows, lane.cols, &j.x, &y)
                .unwrap();
            assert_eq!(
                v,
                Verdict::Identic,
                "{} {:?}: {}",
                lane.name,
                heat.entrant,
                v.line()
            );
            standing.add(
                lanes.len(),
                &race::HeatResult {
                    heat,
                    cost,
                    passes: v.passes(),
                },
            );
        }
        assert_eq!(standing.winner(li), Winner::Stays("nu e mai rapid"));
    }
    let rows = race::rows(&standing, &lanes).unwrap();
    for (n, v) in &rows {
        text.push_str(&format!("[2.0] gpu: valoare {n} = {v:#010x}\n"));
    }
    let p = race::parse_journal(&text);
    assert_eq!(p.lanes, Some(2));
    for l in &p.lane[..2] {
        assert_eq!(l.passes, [Some(true), Some(true)]);
        assert_eq!(
            l.median[0], l.median[1],
            "SIMULARE: acelasi timp pe lansare"
        );
        assert_eq!(l.winner, Some(0));
    }
    assert_eq!(p.token_today_us, p.token_best_us);
}

fn gpu_all(r: &mut Rig, j: &Job) -> GpuResult<CommandProcessor> {
    let mut loads: Vec<(u64, u64)> = j
        .progs
        .iter()
        .chain(&j.cbufs)
        .chain(&j.qmds)
        .chain([&j.records, &j.vin, &j.vy])
        .map(span)
        .collect();
    loads.push((j.host_in.va, j.host_in.size));
    loads.push((j.host_y.va, j.host_y.size));
    gpu_runs(
        r,
        &[
            (COPY_B128, copy_b128_on_cpu, Needs::default()),
            (Q8_MATVEC32, q8_matvec32_on_cpu, Q8_MATVEC32_NEEDS),
            (Q8_MATVEC, q8_matvec_on_cpu, Q8_MATVEC_NEEDS),
        ],
        &loads,
        &[span(&j.vin), span(&j.vy), (j.records.va, 64)],
    )
}
