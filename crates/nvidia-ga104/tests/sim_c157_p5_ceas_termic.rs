//! T179 — C-157 point 5 (= C-151 point 2), SIMULARE: the dashboard as a thermal watch during a long job and before /
//! after the max-speed request (`gsp::rusd_watch`), on pages rendered by the fake GSP-RM's RUSD model
//! (`gpu_sim::fake_rm::SimRusd`). Predictions P1–P7 written before the code (`docs/C-157_P5_CEAS_TERMIC.md`, `54ff52d`).
//! Nothing here says what the RTX 3070 Ti does when it heats up.

use gpu_core::diag::Journal;
use gpu_sim::fake_rm::{RusdRule, SimRusd};
use nvidia_ga104::gsp::rusd::{Dashboard, read_dashboard};
use nvidia_ga104::gsp::rusd_watch::{
    self as kw, BoostSeen, StopReason, ThermalLimits, Verdict, Watch, compare_boost, may_start,
    parse_kw,
};

fn model() -> SimRusd {
    let mut m = SimRusd::new(RusdRule::Accept);
    m.phys = Some(0x7f3_2000);
    m.poll_ms = 500;
    m.mask = 0x1b;
    m
}

/// The page the model rendered at `stamp_ns`, read like the OS reads it.
fn read(m: &SimRusd, stamp_ns: u64) -> (Dashboard, Option<u64>) {
    let page = m.render(stamp_ns);
    let mut mem: &[u8] = &page;
    let mut scratch = Journal::new(32);
    let d = read_dashboard(&mut mem, &mut scratch);
    let s = kw::temp_stamp(&mut mem);
    (d, s)
}

fn text(j: &Journal) -> String {
    j.events().map(|e| format!("{e}\n")).collect()
}

/// One watch over `temps` (°C), a reading every 5 s with a fresh page each time.
fn run_temps(temps: &[i32], reasons: u32) -> (Vec<Verdict>, Watch, Journal) {
    let mut m = model();
    m.reasons = reasons;
    let mut w = Watch::new(ThermalLimits::PROPOSED, 5_000).unwrap();
    let mut j = Journal::new(256);
    let mut v = Vec::new();
    for (i, &t) in temps.iter().enumerate() {
        let now = i as u64 * 5_000;
        assert!(w.due(now));
        m.temp_x256 = t * 256 + 64;
        let (d, s) = read(&m, 1_000_000 + now * 1_000_000);
        v.push(w.observe(now, &d, s, &mut j));
        assert!(!w.due(now + 4_999));
    }
    (v, w, j)
}

#[test]
fn t179a_cool_board_continues_and_one_row() {
    // P1 (+ P7: rows only on change).
    let (v, w, mut j) = run_temps(&[45; 20], 1);
    assert!(v.iter().all(|&x| x == Verdict::Continue));
    assert_eq!(w.stopped(), None);
    assert_eq!(
        (w.temp_min, w.temp_max, w.reads, w.silent_reads),
        (Some(45), Some(45), 20, 0)
    );
    w.summary(&mut j);
    let k = parse_kw(&text(&j));
    assert_eq!(k.changes, 1, "a single verdict row for 20 quiet readings");
    assert_eq!(k.last_verdict, Some(0));
    assert_eq!(
        (k.reads, k.temp_min, k.temp_max, k.silent_reads),
        (Some(20), Some(45), Some(45), Some(0))
    );
    assert_eq!(k.power_max_mw, Some(21_000));
}

#[test]
fn t179b_heating_warns_then_stops_and_latches() {
    // P2.
    let (v, w, j) = run_temps(&[70, 79, 80, 84, 85, 60, 40], 1);
    assert_eq!(
        v,
        [
            Verdict::Continue,
            Verdict::Continue,
            Verdict::Warn,
            Verdict::Warn,
            Verdict::Stop(StopReason::Hot(85)),
            Verdict::Stop(StopReason::Hot(85)),
            Verdict::Stop(StopReason::Hot(85)),
        ]
    );
    assert_eq!(w.stopped(), Some(StopReason::Hot(85)));
    let k = parse_kw(&text(&j));
    assert_eq!(k.changes, 3, "continue → warn → stop");
    assert_eq!((k.stop, k.stop_detail), (Some(1), Some(85)));
}

#[test]
fn t179c_slowdown_reasons() {
    // P3: thermal at a cool temperature stops at once; GPU_IDLE (os-323's 0x1) never does.
    let (v, _, _) = run_temps(&[50, 50], 0x20);
    assert_eq!(v[0], Verdict::Stop(StopReason::Slowdown(0x20)));
    for bits in [0x08, 0x40, 0x80, 0x21] {
        let (v, _, _) = run_temps(&[50], bits);
        assert_eq!(v[0], Verdict::Stop(StopReason::Slowdown(bits & !1)));
    }
    let (v, _, _) = run_temps(&[50; 3], 0x1 | 0x4);
    assert!(
        v.iter().all(|&x| x == Verdict::Continue),
        "idle / SW power cap are not stops"
    );
}

#[test]
fn t179d_silent_dashboard_stops() {
    // P4: the same page (same timestamp) for longer than 10 s.
    let m = model();
    let (d, s) = read(&m, 7_000_000);
    let mut w = Watch::new(ThermalLimits::PROPOSED, 5_000).unwrap();
    let mut j = Journal::new(64);
    assert_eq!(w.observe(0, &d, s, &mut j), Verdict::Continue);
    assert_eq!(w.observe(5_000, &d, s, &mut j), Verdict::Continue);
    assert_eq!(w.observe(10_000, &d, s, &mut j), Verdict::Continue);
    assert_eq!(
        w.observe(10_001, &d, s, &mut j),
        Verdict::Stop(StopReason::Silent(10_001))
    );
    assert_eq!(w.silent_reads, 3);
    // A page that is never written (polling off): no temperature → stop after the same 10 s.
    let mut off = model();
    off.mask = 0;
    let (d0, s0) = read(&off, 7_000_000);
    assert_eq!((d0.temp_c(), s0), (None, None));
    let mut w = Watch::new(ThermalLimits::PROPOSED, 5_000).unwrap();
    assert_eq!(w.observe(0, &d0, s0, &mut j), Verdict::Continue);
    assert_eq!(w.observe(10_000, &d0, s0, &mut j), Verdict::Continue);
    assert_eq!(
        w.observe(15_000, &d0, s0, &mut j),
        Verdict::Stop(StopReason::NoTemperature)
    );
    // A heartbeat in between resets the clock.
    let mut w = Watch::new(ThermalLimits::PROPOSED, 5_000).unwrap();
    for i in 0..10u64 {
        let (d, s) = read(&m, 1 + i * 1_000_000_000);
        assert_eq!(w.observe(i * 5_000, &d, s, &mut j), Verdict::Continue);
    }
}

#[test]
fn t179e_boost_seen_in_the_dashboard() {
    // P5.
    let mut m = model();
    let (p8, _) = read(&m, 5);
    m.pstate_mask = 1;
    m.clk_graphics_mhz = 1_770;
    let (p0, _) = read(&m, 6);
    let mut j = Journal::new(64);
    assert_eq!(compare_boost(&p8, &p0, &mut j), BoostSeen::Raised);
    let k = parse_kw(&text(&j));
    assert_eq!(
        (k.boost, k.boost_clk_before, k.boost_clk_after),
        (Some(0), Some(210), Some(1_770))
    );
    assert_eq!(
        compare_boost(&p0, &p0, &mut j),
        BoostSeen::AlreadyUp,
        "os-323: P0 before the request"
    );
    assert_eq!(compare_boost(&p8, &p8, &mut j), BoostSeen::NoEffect);
    // The core clock alone, +20 %, in P5 → P5.
    let mut m5 = model();
    m5.pstate_mask = 1 << 5;
    m5.clk_graphics_mhz = 1_000;
    let (a, _) = read(&m5, 7);
    m5.clk_graphics_mhz = 1_200;
    let (b, _) = read(&m5, 8);
    assert_eq!(compare_boost(&a, &b, &mut j), BoostSeen::Raised);
    m5.clk_graphics_mhz = 1_199;
    let (c, _) = read(&m5, 9);
    assert_eq!(compare_boost(&a, &c, &mut j), BoostSeen::NoEffect);
    assert_eq!(
        compare_boost(&Dashboard::default(), &p0, &mut j),
        BoostSeen::Unknown
    );
    assert_eq!(kw::after_wait_ms(500), 1_000);
}

#[test]
fn t179f_may_start_again() {
    // P6.
    let mut m = model();
    m.temp_x256 = 71 * 256;
    assert!(!may_start(&ThermalLimits::PROPOSED, &read(&m, 1).0));
    m.temp_x256 = 70 * 256 + 255;
    assert!(may_start(&ThermalLimits::PROPOSED, &read(&m, 1).0));
    m.reasons = 0x40;
    assert!(!may_start(&ThermalLimits::PROPOSED, &read(&m, 1).0));
    assert!(
        !may_start(&ThermalLimits::PROPOSED, &Dashboard::default()),
        "no temperature = no start"
    );
}

#[test]
fn t179g_eight_hour_job_stays_small() {
    // 8 h at 5 s = 5 760 readings, temperature wandering 60–79 °C: no stop, only the warn-free rows.
    let mut m = model();
    let mut w = Watch::new(ThermalLimits::PROPOSED, 5_000).unwrap();
    let mut j = Journal::new(64);
    let mut now = 0u64;
    let mut n = 0;
    while now < 8 * 3_600_000 {
        if w.due(now) {
            m.temp_x256 = (60 + (n % 20)) * 256;
            let (d, s) = read(&m, 1 + now * 1_000_000);
            assert_eq!(w.observe(now, &d, s, &mut j), Verdict::Continue);
            n += 1;
        }
        now += 1_000;
    }
    assert_eq!(n, 5_760);
    w.summary(&mut j);
    let k = parse_kw(&text(&j));
    assert_eq!(k.changes, 1);
    assert_eq!(
        (k.reads, k.temp_min, k.temp_max),
        (Some(5_760), Some(60), Some(79))
    );
    assert!(j.events().count() < 12, "rows: {}", j.events().count());
}
