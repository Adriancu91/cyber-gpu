//! T140 — C-151 point 1, SIMULARE: the board's dashboard ("RM User Shared Data", `gsp::rusd`) against the fake
//! GSP-RM's RUSD model (`gpu_sim::fake_rm::SimRusd`, SIM values: 45.5 °C, 210/405 MHz, P8). Checked: the bytes and
//! the order on the wire (INIT → interval → mask … mask 0 → INIT 0), the seqlock reading (write in progress, a
//! changing timestamp, a missing section), every refusal the model can play, the plan off, the stop refused (the
//! page stays reserved), and the reader of the `kd` rows. Nothing here says what the RTX 3070 Ti answers.

use gpu_core::diag::Journal;
use gpu_sim::fake_rm::{
    FN_GSP_RM_ALLOC, FN_GSP_RM_CONTROL, FakeRm, RUSD_SIZE, RusdRule, SimRusd, cmd,
};
use nvidia_ga104::gsp::golden::CLIENT;
use nvidia_ga104::gsp::rmapi::{self, class};
use nvidia_ga104::gsp::rusd::{
    self as kd, Dashboard, RusdMem, RusdPlan, SHARED_DATA_SIZE, Section, TS_WRITE_IN_PROGRESS, off,
    poll, rows,
};

const SUB: u32 = 0xcf00_0002;
const PAGE: u64 = 0x7f3_2000;

fn rm_with(rusd: Option<SimRusd>) -> FakeRm {
    let mut rm = FakeRm::new();
    let alloc = |rm: &mut FakeRm, parent: u32, h: u32, c: u32, p: &[u8]| {
        let sent = rmapi::encode_alloc(CLIENT, parent, h, c, p);
        let (_, reply) = rm.handle(FN_GSP_RM_ALLOC, &sent).unwrap();
        rmapi::check_alloc_reply(&sent, &reply).unwrap();
    };
    alloc(&mut rm, 0, 0xcf00_0000, class::ROOT, &rmapi::root_params());
    alloc(
        &mut rm,
        CLIENT,
        0xcf00_0001,
        class::DEVICE,
        &rmapi::device_params(CLIENT),
    );
    alloc(
        &mut rm,
        0xcf00_0001,
        SUB,
        class::SUBDEVICE,
        &rmapi::subdevice_params(),
    );
    rm.rusd = rusd;
    rm
}

fn on() -> RusdPlan {
    RusdPlan {
        enabled: true,
        ..RusdPlan::default()
    }
}

/// The page GSP-RM writes, seen by the driver: the model renders it when the driver "sleeps" (one poll).
struct Page {
    bytes: [u8; RUSD_SIZE],
}

impl RusdMem for Page {
    fn read_u64(&mut self, o: usize) -> u64 {
        (&self.bytes[..]).read_u64(o)
    }
    fn read_bytes(&mut self, o: usize, out: &mut [u8]) {
        (&self.bytes[..]).read_bytes(o, out)
    }
}

struct Run {
    text: String,
    controls: Vec<(u32, u32)>,
    start: kd::RusdStart,
    dash: Option<Dashboard>,
    stop: kd::RusdStop,
    slept: u32,
    model: Option<SimRusd>,
}

fn run(rm: &mut FakeRm, plan: &RusdPlan) -> Run {
    let mut j = Journal::new(64);
    // The model renders the page during the sleep; the page lives in a cell the reader sees afterwards.
    let page = std::cell::RefCell::new(Page {
        bytes: [0; RUSD_SIZE],
    });
    let rm_cell = std::cell::RefCell::new(rm);
    let mut call = |sent: &[u8]| -> gpu_core::error::GpuResult<Vec<u8>> {
        let (_, reply) = rm_cell
            .borrow_mut()
            .handle(FN_GSP_RM_CONTROL, sent)
            .unwrap();
        Ok(reply)
    };
    let mut slept = 0;
    // The end-of-chain step split in two, so the model can render the page from its state during the wait.
    let start = kd::rusd_start(plan, PAGE, CLIENT, SUB, &mut call, &mut j).unwrap();
    let dash = if start.polling() {
        slept += plan.poll_ms * 3;
        let m = rm_cell.borrow().rusd.unwrap();
        page.borrow_mut().bytes = m.render(1_000_000_123);
        Some(kd::read_dashboard(&mut *page.borrow_mut(), &mut j))
    } else {
        None
    };
    let stop = kd::rusd_stop(&start, CLIENT, SUB, &mut call, &mut j).unwrap();
    let rm = rm_cell.into_inner();
    Run {
        text: j.events().map(|e| format!("{e}\n")).collect(),
        controls: rm.log.controls.clone(),
        start,
        dash,
        stop,
        slept,
        model: rm.rusd,
    }
}

#[test]
fn t140a_dashboard_read_at_the_end_and_stopped() {
    let mut rm = rm_with(Some(SimRusd::new(RusdRule::Accept)));
    let r = run(&mut rm, &on());
    assert!(r.start.polling());
    assert_eq!(r.slept, 1500);
    let d = r.dash.unwrap();
    assert_eq!(d.temp_x256, Some(45 * 256 + 128));
    assert_eq!(d.temp_c(), Some(45));
    assert_eq!(d.clk_graphics_mhz, Some(210));
    assert_eq!(d.clk_memory_mhz, Some(405));
    assert_eq!(d.pstate(), Some(nvidia_ga104::gsp::perf::Pstate::P(8)));
    assert_eq!(d.load_gpu, Some(3));
    assert_eq!(d.reasons, Some(1));
    assert!(!d.thermal_slowdown());
    assert_eq!(d.avg_power_mw, Some(21_000));
    assert_eq!(d.power_limit_mw, Some(290_000));
    assert_eq!(d.sections, 0x7f);
    assert_eq!(d.busy, 0);
    // The order on the wire: INIT, interval, mask | mask 0, INIT 0 — all accepted.
    let seq: Vec<u32> = r.controls.iter().map(|c| c.0).collect();
    assert_eq!(
        seq,
        [
            cmd::INTERNAL_INIT_USER_SHARED_DATA,
            cmd::INTERNAL_USER_SHARED_DATA_SET_DATA_POLL,
            cmd::INTERNAL_USER_SHARED_DATA_SET_DATA_POLL,
            cmd::INTERNAL_USER_SHARED_DATA_SET_DATA_POLL,
            cmd::INTERNAL_INIT_USER_SHARED_DATA,
        ]
    );
    assert!(r.controls.iter().all(|c| c.1 == 0));
    assert!(r.stop.page_free(&r.start));
    let m = r.model.unwrap();
    assert_eq!((m.phys, m.mask, m.poll_ms), (None, 0, 500));
    // The reader gives back what was written.
    let k = kd::parse_kd(&r.text);
    assert_eq!((k.init, k.interval, k.mask), (Some(0), Some(0), Some(0)));
    assert_eq!(k.temp_x256, Some(45 * 256 + 128));
    assert_eq!(
        (k.clk_graphics_mhz, k.clk_memory_mhz),
        (Some(210), Some(405))
    );
    assert_eq!(k.pstate, Some(8));
    assert_eq!(k.avg_power_mw, Some(21_000));
    assert_eq!((k.sections, k.busy), (Some(0x7f), Some(0)));
    assert_eq!((k.stop_mask, k.stop_init), (Some(0), Some(0)));
    assert!(!k.off && !k.keep_page);
}

#[test]
fn t140b_bytes_on_the_wire() {
    let mut rm = rm_with(Some(SimRusd::new(RusdRule::Accept)));
    let mut sent_all: Vec<Vec<u8>> = Vec::new();
    let mut call = |sent: &[u8]| -> gpu_core::error::GpuResult<Vec<u8>> {
        sent_all.push(sent.to_vec());
        let (_, reply) = rm.handle(FN_GSP_RM_CONTROL, sent).unwrap();
        Ok(reply)
    };
    let mut j = Journal::new(32);
    let s = kd::rusd_start(&on(), PAGE, CLIENT, SUB, &mut call, &mut j).unwrap();
    kd::rusd_stop(&s, CLIENT, SUB, &mut call, &mut j).unwrap();
    let p0 = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_PARAMS as usize;
    let params: Vec<&[u8]> = sent_all.iter().map(|s| &s[p0..]).collect();
    assert_eq!(params[0], &PAGE.to_le_bytes());
    assert_eq!(
        params[1],
        &[0, 0, 0, 0, 0, 0, 0, 0, 0xf4, 1, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(
        params[2],
        &[0x1b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    );
    assert_eq!(params[3], &[0; 16]);
    assert_eq!(params[4], &[0; 8]);
    // The client and subdevice are ours (the difference from NVIDIA/nouveau, written in the docs).
    for s in &sent_all {
        assert_eq!(&s[0..4], &CLIENT.to_le_bytes());
        assert_eq!(&s[4..8], &SUB.to_le_bytes());
    }
    assert_eq!(SHARED_DATA_SIZE, RUSD_SIZE);
}

#[test]
fn t140c_refusals_are_rows_and_nothing_else_is_sent() {
    // INIT refused: one row, no interval, no mask, no stop.
    let mut rm = rm_with(Some(SimRusd::new(RusdRule::RefuseInit(0x56))));
    let r = run(&mut rm, &on());
    assert_eq!(r.start.init, Some(0x8000_0056));
    assert!(r.dash.is_none());
    assert_eq!(r.controls.len(), 1);
    assert!(r.stop.page_free(&r.start));
    assert_eq!(kd::parse_kd(&r.text).init, Some(0x8000_0056));
    // The interval refused: the page was taken, so it is given back (INIT 0) without a mask clear.
    let mut rm = rm_with(Some(SimRusd::new(RusdRule::RefusePoll(0x1f))));
    let r = run(&mut rm, &on());
    assert_eq!(r.start.interval, Some(0x8000_001f));
    assert!(r.dash.is_none());
    let seq: Vec<u32> = r.controls.iter().map(|c| c.0).collect();
    assert_eq!(
        seq,
        [
            cmd::INTERNAL_INIT_USER_SHARED_DATA,
            cmd::INTERNAL_USER_SHARED_DATA_SET_DATA_POLL,
            cmd::INTERNAL_INIT_USER_SHARED_DATA,
        ]
    );
    assert_eq!(r.stop.mask, None);
    assert_eq!(r.stop.deinit, Some(0));
    // No RUSD in the model at all (GSP-RM without it): INVALID_COMMAND as a row.
    let mut rm = rm_with(None);
    let r = run(&mut rm, &on());
    assert_eq!(
        r.start.init,
        Some(0x8000_0000 | gpu_sim::fake_rm::status::INVALID_COMMAND)
    );
}

#[test]
fn t140d_stop_refused_keeps_the_page() {
    let mut rm = rm_with(Some(SimRusd::new(RusdRule::RefuseDeinit(0x40))));
    let r = run(&mut rm, &on());
    assert!(r.dash.is_some());
    assert_eq!(r.stop.deinit, Some(0x8000_0040));
    assert!(!r.stop.page_free(&r.start));
    assert!(r.text.contains(rows::KEEP_PAGE));
    assert!(kd::parse_kd(&r.text).keep_page);
}

#[test]
fn t140e_plan_off_sends_nothing() {
    assert!(!RusdPlan::default().enabled);
    let mut rm = rm_with(Some(SimRusd::new(RusdRule::Accept)));
    let r = run(&mut rm, &RusdPlan::default());
    assert!(r.controls.is_empty());
    assert!(r.dash.is_none());
    assert_eq!(r.slept, 0);
    assert!(kd::parse_kd(&r.text).off);
    // Bad plans are refused before the first byte.
    let mut call =
        |_: &[u8]| -> gpu_core::error::GpuResult<Vec<u8>> { panic!("nothing may be sent") };
    let mut j = Journal::new(8);
    for p in [
        RusdPlan {
            poll_ms: 50,
            ..on()
        },
        RusdPlan { mask: 0x80, ..on() },
        RusdPlan { mask: 0, ..on() },
    ] {
        assert!(kd::rusd_start(&p, PAGE, CLIENT, SUB, &mut call, &mut j).is_err());
    }
    assert!(kd::rusd_start(&on(), 0, CLIENT, SUB, &mut call, &mut j).is_err());
    assert!(kd::rusd_start(&on(), PAGE + 4, CLIENT, SUB, &mut call, &mut j).is_err());
}

/// A page GSP-RM is writing while we read: the timestamp reads come from a script.
struct Racing {
    bytes: [u8; RUSD_SIZE],
    ts_script: Vec<u64>,
    reads: usize,
}

impl RusdMem for Racing {
    fn read_u64(&mut self, o: usize) -> u64 {
        let v = self.ts_script.get(self.reads).copied();
        self.reads += 1;
        v.unwrap_or_else(|| (&self.bytes[..]).read_u64(o))
    }
    fn read_bytes(&mut self, o: usize, out: &mut [u8]) {
        (&self.bytes[..]).read_bytes(o, out)
    }
}

#[test]
fn t140f_seqlock_retries_missing_and_busy() {
    let mut m = SimRusd::new(RusdRule::Accept);
    m.phys = Some(PAGE);
    m.poll_ms = 500;
    m.mask = poll::THERMAL;
    let bytes = m.render(777);
    // Write in progress, then a timestamp that changes under the copy, then a clean read.
    let mut r = Racing {
        bytes,
        ts_script: vec![TS_WRITE_IN_PROGRESS, 700, 777],
        reads: 0,
    };
    match kd::read_section::<4>(&mut r, off::TEMP_GPU) {
        Section::Ok { ts, data } => {
            assert_eq!(ts, 777);
            assert_eq!(i32::from_le_bytes(data), 45 * 256 + 128);
        }
        other => panic!("{other:?}"),
    }
    // Only the temperature was asked for: the clocks are "missing", not busy.
    assert_eq!(
        kd::read_section::<16>(&mut &bytes[..], off::CLOCKS),
        Section::Missing
    );
    // Always being written: busy after 10 tries.
    let mut r = Racing {
        bytes,
        ts_script: vec![TS_WRITE_IN_PROGRESS; 10],
        reads: 0,
    };
    assert_eq!(kd::read_section::<4>(&mut r, off::TEMP_GPU), Section::Busy);
    assert_eq!(r.reads, 10);
    // The whole dashboard with only THERMAL: one value row, sections = bit 4.
    let mut j = Journal::new(16);
    let d = kd::read_dashboard(&mut &bytes[..], &mut j);
    assert_eq!(d.sections, 1 << 4);
    assert_eq!(d.clk_graphics_mhz, None);
    // A hot board with a thermal slow-down.
    let mut hot = SimRusd::new(RusdRule::Accept);
    hot.phys = Some(PAGE);
    hot.poll_ms = 500;
    hot.mask = poll::DASHBOARD;
    hot.temp_x256 = 87 * 256;
    hot.reasons = 0x40;
    let d = kd::read_dashboard(&mut &hot.render(5)[..], &mut Journal::new(16));
    assert_eq!(d.temp_c(), Some(87));
    assert!(d.thermal_slowdown());
    // Not polling (no page): all zero, nothing read.
    let none = SimRusd::new(RusdRule::Accept).render(5);
    let d = kd::read_dashboard(&mut &none[..], &mut Journal::new(16));
    assert_eq!((d.sections, d.busy), (0, 0));
}

#[test]
fn t140g_dashboard_at_end_in_one_call() {
    // The one-call form the OS links: the page here is a snapshot rendered beforehand (the model polls once).
    let mut rm = rm_with(Some(SimRusd::new(RusdRule::Accept)));
    let mut snap = SimRusd::new(RusdRule::Accept);
    snap.phys = Some(PAGE);
    snap.poll_ms = 500;
    snap.mask = poll::DASHBOARD;
    let bytes = snap.render(42);
    let mut mem: &[u8] = &bytes;
    let mut call = |sent: &[u8]| -> gpu_core::error::GpuResult<Vec<u8>> {
        let (_, reply) = rm.handle(FN_GSP_RM_CONTROL, sent).unwrap();
        Ok(reply)
    };
    let mut slept = 0;
    let mut sleep = |ms: u32| slept += ms;
    let mut j = Journal::new(64);
    let (s, d, e) = kd::dashboard_at_end(
        &on(),
        PAGE,
        CLIENT,
        SUB,
        &mut call,
        &mut mem,
        &mut sleep,
        &mut j,
    )
    .unwrap();
    assert!(s.polling());
    assert_eq!(d.unwrap().sections, 0x7f);
    assert!(e.page_free(&s));
    assert_eq!(slept, 1500);
}

#[test]
fn t140h_stop_without_deinit_as_on_the_board() {
    // os-323 on the RTX 3070 Ti: INIT(0) refused with 0x8000001f (NV_ERR_INVALID_ARGUMENT). The model plays that
    // refusal; the new stop never sends INIT(0), so the refusal (and its ASSERT on the board) cannot happen.
    let mut rm = rm_with(Some(SimRusd::new(RusdRule::RefuseDeinit(0x1f))));
    let mut snap = SimRusd::new(RusdRule::Accept);
    snap.phys = Some(PAGE);
    snap.poll_ms = 500;
    snap.mask = poll::DASHBOARD;
    let bytes = snap.render(42);
    let mut mem: &[u8] = &bytes;
    let mut sent_cmds = Vec::new();
    let mut call = |sent: &[u8]| -> gpu_core::error::GpuResult<Vec<u8>> {
        sent_cmds.push(sent.to_vec());
        let (_, reply) = rm.handle(FN_GSP_RM_CONTROL, sent).unwrap();
        Ok(reply)
    };
    let mut sleep = |_ms: u32| {};
    let mut j = Journal::new(64);
    let (s, d, e) = kd::dashboard_at_end_keep_page(
        &on(),
        PAGE,
        CLIENT,
        SUB,
        &mut call,
        &mut mem,
        &mut sleep,
        &mut j,
    )
    .unwrap();
    assert!(s.polling());
    assert_eq!(d.unwrap().sections, 0x7f);
    assert_eq!(e.mask, Some(0));
    assert_eq!(e.deinit, None);
    // INIT, interval, mask, mask 0: four controls, no INIT(0).
    assert_eq!(sent_cmds.len(), 4);
    assert_eq!(rm.log.controls.len(), 4);
    let text: String = j.events().map(|e| format!("{e}\n")).collect();
    assert!(text.contains(rows::KEEP_PAGE_ON_PURPOSE));
    assert!(!text.contains(rows::STOP_INIT));
    let k = kd::parse_kd(&text);
    assert!(k.keep_page);
    assert_eq!(k.stop_init, None);
    assert_eq!(k.stop_mask, Some(0));
}
