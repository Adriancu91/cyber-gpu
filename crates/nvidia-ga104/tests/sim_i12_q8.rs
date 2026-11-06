//! C-83 I12, SIMULARE: Kronos's basic operation y = W·x (W in Q8_0) on the
//! work channel, end to end on the driver's code: W and x sit in host RAM
//! (the OS's DMA pages), one GPFIFO entry does
//! copy RAM→VRAM (I11's `copy_b128`) → `q8_matvec` (NAK, from VRAM) →
//! copy y VRAM→RAM, each followed by a timestamp record; the driver judges y
//! with `q8_matvec::judge` (bit for bit with the program's oracle, else the
//! margin written before the run). The "GPU" is `gpu_sim::cmdproc` reading
//! everything through the page tables. Nothing here proves anything about the
//! real board; times are the model's own clock.

mod common;

use common::{Rig, dma_pages, gpu_runs, put_bytes, rig, span, sys_read, sys_write};
use gpu_core::error::GpuResult;
use gpu_sim::cmdproc::{
    CommandProcessor, Needs, Q8_MATVEC_NEEDS, copy_b128_on_cpu, q8_matvec_on_cpu,
};
use nvidia_ga104::gsp::golden::{GoldenMemory, Mapping};
use nvidia_ga104::gsp::work_queue::ChannelMemory;
use nvidia_ga104::mmu::SysMapping;
use nvidia_ga104::push::PushBuf;
use nvidia_ga104::q8_matvec::{self as mv, Q8_MATVEC, Verdict};
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
    let progs = [COPY_B128, Q8_MATVEC].map(|p| r.mem.alloc_mapped(p.len() as u64, 0x1000).unwrap());
    let mut m = || r.mem.alloc_mapped(0x1000, 0x1000).unwrap();
    let cbufs = [m(), m(), m()];
    let qmds = [m(), m(), m()];
    let records = m();
    let wr = r.mem.writer_mut();
    put_bytes(wr, progs[0].pa, COPY_B128);
    put_bytes(wr, progs[1].pa, Q8_MATVEC);
    put_bytes(
        wr,
        cbufs[0].pa,
        &transfer::copy_cbuf(host_in.va, vin.va, up_bytes).unwrap(),
    );
    put_bytes(
        wr,
        cbufs[1].pa,
        &mv::q8_cbuf(vin.va, vin.va + x_off, vy.va, rows, cols).unwrap(),
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
        &mv::q8_qmd(progs[1].va, cbufs[1].va, rows).unwrap(),
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
            (Q8_MATVEC, q8_matvec_on_cpu, Q8_MATVEC_NEEDS),
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
fn matvec_q8_on_the_work_channel_is_identical_to_the_oracle() {
    for (rows, cols) in [
        (1, 32),
        (100, 96),
        (37, 1056),
        (mv::TEST_ROWS, mv::TEST_COLS),
    ] {
        let mut r = rig();
        let j = prepare(&mut r, rows, cols, 0x12);
        submit(&mut r, &j);
        let cp = gpu(&mut r, &j).unwrap();
        assert_eq!(cp.timestamps.len(), 4);
        let (y, rec) = y_and_records(&mut r, &j);
        let v = mv::judge(&j.w, rows, cols, &j.x, &y).unwrap();
        assert_eq!(v, Verdict::Identic, "{rows}x{cols}: {}", v.line());
        let t = transfer::analyze_sequence(&rec, 3, FIRST).unwrap();
        assert_eq!(t.len(), 3);
        assert!(t.iter().all(|&ns| ns >= 250_000), "{t:?}");
        // W and x arrived in VRAM bit for bit (spot words).
        for off in transfer::spot_offsets(j.w.len() as u64) {
            let got = r.mem.writer_mut().read_u32(j.vin.pa + off).unwrap();
            let want = u32::from_le_bytes(j.w[off as usize..off as usize + 4].try_into().unwrap());
            assert_eq!(got, want, "W la {off:#x}");
        }
        let x0 = r.mem.writer_mut().read_u32(j.vin.pa + j.x_off).unwrap();
        assert_eq!(x0, j.x[0].to_bits());
        assert_eq!(j.up_bytes % 16, 0);
        assert_eq!(j.y_bytes % 16, 0);
    }
}

/// Mistakes planted in what the driver writes; each must end in a refusal or
/// in "nu trece" naming the row, never in a pass.
#[test]
fn planted_mistakes_never_pass() {
    // 1. x pointer one block too far (wrong offset into the upload).
    let mut r = rig();
    let j = prepare(&mut r, 64, 256, 3);
    let cb = mv::q8_cbuf(j.vin.va, j.vin.va + j.x_off + 128, j.vy.va, 64, 256).unwrap();
    put_bytes(r.mem.writer_mut(), j.cbufs[1].pa, &cb);
    submit(&mut r, &j);
    gpu(&mut r, &j).unwrap();
    let (y, _) = y_and_records(&mut r, &j);
    assert!(!mv::judge(&j.w, 64, 256, &j.x, &y).unwrap().passes());

    // 2. A grid one row short: the last row is left as junk → named.
    let mut r = rig();
    let j = prepare(&mut r, 64, 256, 3);
    let mut q = [0u8; 256];
    q.copy_from_slice(&mv::q8_qmd(j.progs[1].va, j.cbufs[1].va, 63).unwrap());
    put_bytes(r.mem.writer_mut(), j.qmds[1].pa, &q);
    submit(&mut r, &j);
    gpu(&mut r, &j).unwrap();
    let (y, _) = y_and_records(&mut r, &j);
    match mv::judge(&j.w, 64, 256, &j.x, &y).unwrap() {
        Verdict::NuTrece { row, .. } => assert_eq!(row, 63),
        v => panic!("{v:?}"),
    }

    // 3. The matvec QMD without its shared memory: the model refuses the
    // launch, the GPU stops, the driver sees the record missing.
    let mut r = rig();
    let j = prepare(&mut r, 8, 64, 3);
    let plain = nvidia_ga104::qmd::ComputeQmd {
        program_va: j.progs[1].va,
        registers: mv::Q8_REGISTERS,
        cbuf0_va: j.cbufs[1].va,
        cbuf0_size: mv::CBUF_SIZE,
        grid: [8, 1, 1],
        block: [32, 1, 1],
        release: None,
    };
    put_bytes(r.mem.writer_mut(), j.qmds[1].pa, &plain.to_bytes().unwrap());
    submit(&mut r, &j);
    assert!(gpu(&mut r, &j).is_err());
    let (_, rec) = y_and_records(&mut r, &j);
    assert!(transfer::analyze_sequence(&rec, 3, FIRST).is_err());

    // 4. One weight byte changed in host RAM after the oracle was computed
    // (a corrupted upload): the row it is in fails.
    let mut r = rig();
    let j = prepare(&mut r, 16, 64, 3);
    let row_bytes = 2 * 36;
    let off = 5 * row_bytes + 36 + 4 + 17; // row 5, block 1, number 17
    let mut data = sys_read(&r.sys, &j.host_in, j.up_bytes);
    data[off] = data[off].wrapping_add(100);
    sys_write(&mut r.sys, &j.host_in, &data);
    submit(&mut r, &j);
    gpu(&mut r, &j).unwrap();
    let (y, _) = y_and_records(&mut r, &j);
    match mv::judge(&j.w, 16, 64, &j.x, &y).unwrap() {
        Verdict::NuTrece { row, .. } => assert_eq!(row, 5),
        v => panic!("{v:?}"),
    }
}
