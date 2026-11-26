//! C-157 after os-456 (`docs/C-157_P2_OS456_SM.md`; G1–G4 written together with this code, not before) — T255, on the
//! processor: the number of SMs. os-456 (MĂSURAT by the OS on the RTX 3070 Ti) read 7 × 6 × 2 = 84 from the three
//! `LITTER_*` entries; the board has 48. The reply below is built by hand with the board's litter values and the
//! TPC count the public spec gives (24); what GSP-RM really writes at index 9 is seen only on the board.

use nvidia_ga104::grid_barrier as gb;
use nvidia_ga104::gsp::rmapi::{self, GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT, GrCounts};
use nvidia_ga104::kronos_token as kt;

/// `infoList[i] = {index i, data v}` for the given entries (8 bytes each), zero elsewhere.
fn reply(entries: &[(u32, u32)]) -> Vec<u8> {
    let mut b = rmapi::gr_info_request();
    for &(i, v) in entries {
        let at = i as usize * 8;
        b[at..at + 4].copy_from_slice(&i.to_le_bytes());
        b[at + 4..at + 8].copy_from_slice(&v.to_le_bytes());
    }
    b
}

/// The board's litter (os-456) + SM version, warps; `tpcs` at index 9 (None = left 0).
fn board(tpcs: Option<u32>) -> Vec<u8> {
    let mut e = vec![(12, 0x806), (13, 48), (20, 7), (23, 6), (32, 2)];
    if let Some(t) = tpcs {
        e.push((GR_INFO_INDEX_SHADER_PIPE_SUB_COUNT, t));
    }
    reply(&e)
}

#[test]
fn t255a_g1_litter_is_84_real_is_48() {
    let p = board(Some(24));
    let info = rmapi::parse_gr_info(&p).unwrap();
    let g = info.shape;
    assert_eq!(g.gpcs * g.tpc_per_gpc * g.sm_per_tpc, 84); // what os-456 read
    assert_eq!(
        rmapi::parse_gr_counts(&p).unwrap(),
        GrCounts { tpcs: 24, sms: 48 }
    );
    println!("G1: LITTER 7 x 6 x 2 = 84 (maximul de proiect); TPC 24 -> 48 SM");
}

#[test]
fn t255b_g2_refusals() {
    assert!(rmapi::parse_gr_counts(&board(None)).is_err()); // 0 TPCs
    assert!(rmapi::parse_gr_counts(&board(Some(43))).is_err()); // above 7 x 6
    assert!(rmapi::parse_gr_counts(&board(Some(42))).is_ok());
    let mut p = board(Some(24));
    p[9 * 8..9 * 8 + 4].copy_from_slice(&10u32.to_le_bytes()); // index of another entry
    assert!(rmapi::parse_gr_counts(&p).is_err());
    assert!(rmapi::parse_gr_counts(&p[..p.len() - 1]).is_err());
}

#[test]
fn t255c_g3_g4_groups_from_the_real_count() {
    let p = board(Some(24));
    let g = rmapi::parse_gr_info(&p).unwrap().shape;
    let sms = rmapi::parse_gr_counts(&p).unwrap().sms;
    // G3: the first barrier run, G = 48 / 96 (os-456 would have asked 84 / 168).
    assert_eq!(gb::groups_for_sms(sms, 1).unwrap(), 48);
    assert_eq!(gb::groups_for_sms(sms, 2).unwrap(), 96);
    assert_eq!(gb::groups_for(&g, 2).unwrap(), 168);
    assert!(gb::groups_for_sms(sms, 3).is_err());
    // G4: the wide runs and token32 from the real count; the old entries fail closed (84 x 16 = 1 344 > 768
    // resident would never open the barrier).
    assert_eq!(gb::groups_for_wide_sms(sms, 16).unwrap(), 768);
    assert!(gb::groups_for_wide(&g, 16).is_err());
    assert_eq!(kt::token32_groups_for_sms(sms, 16).unwrap(), 768);
    assert!(kt::token32_groups_for(&g, 16).is_err());
    assert!(kt::token32_step_for(768, sms).is_ok());
    assert!(kt::token32_step_for(84 * 16, sms).is_err());
    assert_eq!(kt::groups_for_sms(sms, 1000).unwrap(), 48);
    println!("G3/G4: bariera G = 48 / 96, larga 768; vechile intrari pentru G mare refuza");
}
