//! T139 — C-147 point 4, SIMULARE: the boost ready to link (`gsp::perf::{boost_start, boost_finish}`) against the
//! fake GSP-RM's performance model (`gpu_sim::fake_rm::SimPerf`, idle P8 / boosted P0 — SIM values). The rows, the
//! bytes sent, the clear and the reader are checked for every answer the model can play: accepted, accepted
//! without effect, refused, plan off, the public shape. Nothing here says what the RTX 3070 Ti answers.

use gpu_core::diag::Journal;
use gpu_core::error::GpuError;
use gpu_sim::fake_rm::{BoostRule, FN_GSP_RM_ALLOC, FN_GSP_RM_CONTROL, FakeRm, SimPerf, cmd};
use nvidia_ga104::gsp::golden::CLIENT;
use nvidia_ga104::gsp::perf::{
    self as gp, BoostAnswer, BoostPlan, BoostShape, BoostVerdict, Pstate, rows,
};
use nvidia_ga104::gsp::rmapi::{self, class};

const SUB: u32 = 0xcf00_0002;

fn rm_with(perf: Option<SimPerf>) -> FakeRm {
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
    rm.perf = perf;
    rm
}

fn on() -> BoostPlan {
    BoostPlan {
        enabled: true,
        ..BoostPlan::default()
    }
}

/// Runs start + finish like the OS would (start after the work channel, finish at the end) and returns the
/// journal's text, the controls the model saw, and both results.
fn run(
    rm: &mut FakeRm,
    plan: &BoostPlan,
) -> (String, Vec<(u32, u32)>, gp::BoostStart, gp::BoostEnd) {
    let mut j = Journal::new(64);
    let mut call = |sent: &[u8]| -> gpu_core::error::GpuResult<Vec<u8>> {
        let (_, reply) = rm.handle(FN_GSP_RM_CONTROL, sent).unwrap();
        Ok(reply)
    };
    let s = gp::boost_start(plan, CLIENT, SUB, &mut call, &mut j).unwrap();
    // … here the OS runs i14-simbol, kronos --gpu (kp), kc …
    let e = gp::boost_finish(&s, plan, CLIENT, SUB, &mut call, &mut j).unwrap();
    let text: String = j.events().map(|e| format!("{e}\n")).collect();
    (text, rm.log.controls.clone(), s, e)
}

#[test]
fn t139a_accepted_boost_raises_then_clears() {
    let mut rm = rm_with(Some(SimPerf::new(BoostRule::Accept)));
    let (text, controls, s, e) = run(&mut rm, &on());
    assert_eq!(s.before, Pstate::P(8));
    assert_eq!(s.answer, BoostAnswer::Accepted);
    assert_eq!(s.after, Some(Pstate::P(0)));
    assert_eq!(s.verdict(), BoostVerdict::Raised);
    assert_eq!(e.before_clear, Some(Pstate::P(0)));
    assert_eq!(e.clear, Some(0));
    assert_eq!(e.end, Pstate::P(8));
    // The order on the wire: read, boost, read | read, clear, read.
    let seq: Vec<u32> = controls.iter().map(|c| c.0).collect();
    let (r, b) = (
        cmd::PERF_GET_CURRENT_PSTATE,
        cmd::INTERNAL_PERF_BOOST_SET_2X,
    );
    assert_eq!(seq, [r, b, r, r, b, r]);
    assert!(controls.iter().all(|c| c.1 == 0));
    assert_eq!(rm.perf.unwrap().accepted, 2);
    assert_eq!(rm.perf.unwrap().active, None, "the clear ended the boost");
    let k = gp::parse_ks(&text);
    assert_eq!(
        (k.before, k.boost, k.after, k.verdict),
        (Some(8), Some(0), Some(0), Some(1))
    );
    assert_eq!(
        (k.before_clear, k.clear, k.end),
        (Some(0), Some(0), Some(8))
    );
    assert!(!k.off);
    assert_eq!(text.lines().count(), 7, "{text}");
}

#[test]
fn t139b_bytes_on_the_wire_are_nvidias_shape_finite_and_cleared_alike() {
    let p = on();
    let b = p.encode_boost(CLIENT, SUB).unwrap();
    let c = p.encode_clear(CLIENT, SUB);
    let o = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_PARAMS as usize;
    let k = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_CMD as usize;
    for m in [&b, &c] {
        assert_eq!(&m[k..k + 4], &0x2080_0a9au32.to_le_bytes());
        assert_eq!(m.len(), o + 8);
    }
    // flags 0x52 (BOOST_TO_MAX | CUDA | CUDA_PRIORITY_HIGH) in the NvBool, 600 s; the clear: 0x10, 0.
    assert_eq!(&b[o..], &[0x52, 0, 0, 0, 0x58, 0x02, 0, 0]);
    assert_eq!(&c[o..], &[0x10, 0, 0, 0, 0, 0, 0, 0]);
    // Never "until cleared", never above an hour, never 0.
    for d in [0, 3601, u32::MAX] {
        let x = BoostPlan { duration_s: d, ..p };
        assert!(x.encode_boost(CLIENT, SUB).is_err());
    }
}

#[test]
fn t139c_refused_boost_is_a_row_and_nothing_is_cleared() {
    let mut rm = rm_with(Some(SimPerf::new(BoostRule::Refuse(0x56))));
    let (text, controls, s, e) = run(&mut rm, &on());
    assert_eq!(s.answer, BoostAnswer::Refused(0x56));
    assert_eq!(s.after, None);
    assert_eq!(s.verdict(), BoostVerdict::Refused);
    assert_eq!((e.before_clear, e.clear, e.end), (None, None, Pstate::P(8)));
    let seq: Vec<u32> = controls.iter().map(|c| c.0).collect();
    let (r, b) = (
        cmd::PERF_GET_CURRENT_PSTATE,
        cmd::INTERNAL_PERF_BOOST_SET_2X,
    );
    assert_eq!(seq, [r, b, r]);
    let k = gp::parse_ks(&text);
    assert_eq!(
        (k.boost, k.verdict, k.clear),
        (Some(0x8000_0056), Some(3), None)
    );
}

#[test]
fn t139d_accepted_without_effect_is_said_so() {
    let mut rm = rm_with(Some(SimPerf::new(BoostRule::AcceptNoEffect)));
    let (text, _, s, e) = run(&mut rm, &on());
    assert_eq!(s.verdict(), BoostVerdict::NoEffect);
    assert_eq!(e.before_clear, Some(Pstate::P(8)));
    assert_eq!(e.clear, Some(0), "an accepted boost is always cleared");
    assert_eq!(gp::parse_ks(&text).verdict, Some(2));
}

#[test]
fn t139e_plan_off_sends_only_the_readings() {
    let mut rm = rm_with(Some(SimPerf::new(BoostRule::Accept)));
    let (text, controls, s, e) = run(&mut rm, &BoostPlan::default());
    assert_eq!(s.verdict(), BoostVerdict::Off);
    assert_eq!(e.clear, None);
    assert!(controls.iter().all(|c| c.0 == cmd::PERF_GET_CURRENT_PSTATE));
    assert_eq!(controls.len(), 2);
    let k = gp::parse_ks(&text);
    assert!(k.off);
    assert_eq!((k.before, k.end, k.verdict), (Some(8), Some(8), Some(0)));
}

#[test]
fn t139f_no_perf_model_today_readings_refused_still_no_error() {
    // The model as before C-147 point 4 (and what an unknown GSP-RM answer looks like): every control refused.
    let mut rm = rm_with(None);
    let (text, _, s, e) = run(&mut rm, &on());
    assert!(matches!(s.before, Pstate::Refused(_)));
    assert!(matches!(s.answer, BoostAnswer::Refused(_)));
    assert!(matches!(e.end, Pstate::Refused(_)));
    assert_eq!(gp::parse_ks(&text).verdict, Some(3));
}

#[test]
fn t139g_public_shape_refused_by_the_model_and_a_transport_error_stops() {
    let mut rm = rm_with(Some(SimPerf::new(BoostRule::Accept)));
    let public = BoostPlan {
        shape: BoostShape::Public,
        ..on()
    };
    let (_, _, s, _) = run(&mut rm, &public);
    assert_eq!(s.verdict(), BoostVerdict::Refused);
    // A transport error is an error (`?`), not a row.
    let mut j = Journal::new(8);
    let mut broken = |_: &[u8]| -> gpu_core::error::GpuResult<Vec<u8>> {
        Err(GpuError::Timeout { waited_us: 1 })
    };
    assert!(gp::boost_start(&on(), CLIENT, SUB, &mut broken, &mut j).is_err());
    // A reply for another command is an error too.
    let mut rm2 = rm_with(Some(SimPerf::new(BoostRule::Accept)));
    let mut wrong = |sent: &[u8]| -> gpu_core::error::GpuResult<Vec<u8>> {
        let (_, mut reply) = rm2.handle(FN_GSP_RM_CONTROL, sent).unwrap();
        let k = nvidia_rm570::rmapi::RPC_GSP_RM_CONTROL_CMD as usize;
        reply[k] ^= 1;
        Ok(reply)
    };
    assert!(gp::boost_start(&on(), CLIENT, SUB, &mut wrong, &mut j).is_err());
    assert_eq!(rows::VERDICT.split_whitespace().next(), Some("ks"));
}
