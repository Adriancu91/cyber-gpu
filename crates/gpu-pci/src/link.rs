//! PCIe link diagnosis (read-only): why a link runs slower than the board
//! can, decided from the board's link registers and those of the port
//! above it (C-157, Adrian's decision of 23 Nov 2026 20:11).
//!
//! Source: PCI Express Base Specification — PCIe Capabilities register
//! (device/port type, bits 7:4), Link Capabilities (max speed 3:0, max width
//! 9:4), Link Status (current speed 3:0, negotiated width 9:4); type-1
//! header Secondary Bus Number (offset 0x19). Original code, no foreign code.
//!
//! Nothing here writes to configuration space.

use gpu_core::hal::ConfigSpace;

use crate::caps::{self, PcieLink};

/// Type-1 header: Secondary Bus Number.
pub const REG_SECONDARY_BUS: u16 = 0x19;
/// PCIe Capabilities device/port type: Root Port of a Root Complex.
pub const PORT_ROOT: u8 = 0x4;
/// PCIe Capabilities device/port type: Downstream Port of a switch.
pub const PORT_SWITCH_DOWNSTREAM: u8 = 0x6;

/// Raw link ceiling per direction, in GB/s (10^9 bytes): transfer rate ×
/// encoding efficiency (8b/10b for gen1/2, 128b/130b for gen3–5) × width
/// ÷ 8. Real DMA reaches ~80–90 % of this (packet headers, flow control).
/// `None` for an unknown speed code.
pub fn link_bandwidth_gbps(speed: u8, width: u8) -> Option<f64> {
    let per_lane = match speed {
        1 => 2.5 * 0.8 / 8.0,
        2 => 5.0 * 0.8 / 8.0,
        3 => 8.0 * 128.0 / 130.0 / 8.0,
        4 => 16.0 * 128.0 / 130.0 / 8.0,
        5 => 32.0 * 128.0 / 130.0 / 8.0,
        _ => return None,
    };
    Some(per_lane * width as f64)
}

/// Why the link runs where it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkVerdict {
    /// Negotiated at the best both ends can do.
    Full,
    /// The port above has fewer lanes than the board (slot wired narrower,
    /// or a split lane group): only a different slot helps.
    SlotNarrow,
    /// The port could do more lanes than were negotiated: board seating,
    /// dirty contacts, a riser — reseat, do not move.
    TrainedNarrow,
    /// The port above is slower than the board (platform generation): no
    /// slot on this machine helps.
    SlotSlow,
    /// Width is fine but the link sits at gen1 — a GPU at rest lowers its
    /// speed to save power; only meaningful under load.
    Idle,
    /// Width fine, both ends faster than negotiated, link not at gen1:
    /// signal quality, a BIOS speed limit or a riser.
    TrainedSlow,
    /// Port registers not available: only the board side is known.
    NoPort,
}

/// The diagnosis plus the numbers it was made from.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinkDiagnosis {
    pub verdict: LinkVerdict,
    /// Best width / speed the two ends could agree on.
    pub best_width: u8,
    pub best_speed: u8,
    /// Raw ceilings (GB/s per direction): now, and at the best common link.
    pub now_gbps: Option<f64>,
    pub best_gbps: Option<f64>,
}

/// Decide the cause from the board's link and, if read, the port above.
/// Width is checked first (lanes trained, then lanes the slot has): it does
/// not change at rest, speed does.
pub fn diagnose(board: &PcieLink, port: Option<&PcieLink>) -> LinkDiagnosis {
    let (best_width, best_speed, verdict) = match port {
        None => (
            board.max_width,
            board.max_speed,
            if board.width >= board.max_width && board.speed >= board.max_speed {
                LinkVerdict::Full
            } else {
                LinkVerdict::NoPort
            },
        ),
        Some(p) => {
            let bw = board.max_width.min(p.max_width);
            let bs = board.max_speed.min(p.max_speed);
            let v = if board.width < bw {
                LinkVerdict::TrainedNarrow
            } else if p.max_width < board.max_width {
                LinkVerdict::SlotNarrow
            } else if board.speed < bs {
                if board.speed <= 1 {
                    LinkVerdict::Idle
                } else {
                    LinkVerdict::TrainedSlow
                }
            } else if p.max_speed < board.max_speed {
                LinkVerdict::SlotSlow
            } else {
                LinkVerdict::Full
            };
            (bw, bs, v)
        }
    };
    LinkDiagnosis {
        verdict,
        best_width,
        best_speed,
        now_gbps: link_bandwidth_gbps(board.speed, board.width),
        best_gbps: link_bandwidth_gbps(best_speed, best_width),
    }
}

impl LinkVerdict {
    /// Short Romanian text for the tour line `pci.legatura.cauza`.
    pub fn text(self) -> &'static str {
        match self {
            LinkVerdict::Full => "legatura la maximul comun al placii si al slotului",
            LinkVerdict::SlotNarrow => {
                "slotul are mai putine linii decat placa: doar alt slot (mai lat) ajuta"
            }
            LinkVerdict::TrainedNarrow => {
                "slotul poate mai multe linii decat s-au inteles: placa de reasezat / contacte / prelungitor"
            }
            LinkVerdict::SlotSlow => {
                "slotul e de generatie mai veche decat placa: niciun slot al acestui calculator nu ajuta"
            }
            LinkVerdict::Idle => "latimea e buna; viteza gen1 = repaus (de citit sub sarcina)",
            LinkVerdict::TrainedSlow => {
                "latimea e buna, viteza sub ce pot ambele capete: semnal / limita din BIOS / prelungitor"
            }
            LinkVerdict::NoPort => "portul de deasupra necitit: cauza nu se poate spune",
        }
    }
}

/// Is this function a port a board can sit under (root port or switch
/// downstream port)? Read-only.
pub fn is_downstream_port<C: ConfigSpace + ?Sized>(cfg: &mut C) -> bool {
    matches!(
        caps::pcie_link(cfg).map(|l| l.device_type),
        Some(PORT_ROOT) | Some(PORT_SWITCH_DOWNSTREAM)
    )
}

/// Secondary bus number of a type-1 (bridge) function. Read-only.
pub fn secondary_bus<C: ConfigSpace + ?Sized>(cfg: &mut C) -> Option<u8> {
    if crate::header::header_type(cfg) & 0x7f != 1 {
        return None;
    }
    Some(cfg.read8(REG_SECONDARY_BUS))
}

/// Among `bridges` (any order), the index of the downstream port whose
/// secondary bus is `board_bus`. Read-only.
pub fn find_upstream_port<C: ConfigSpace>(bridges: &mut [C], board_bus: u8) -> Option<usize> {
    bridges
        .iter_mut()
        .position(|b| secondary_bus(b) == Some(board_bus) && is_downstream_port(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    fn link(max_speed: u8, max_width: u8, speed: u8, width: u8) -> PcieLink {
        PcieLink {
            device_type: 0,
            max_speed,
            max_width,
            speed,
            width,
        }
    }

    /// Minimal type-1 configuration space (SIM) with only a PCIe capability.
    struct Bridge {
        regs: Vec<u32>,
        writes: usize,
    }

    impl Bridge {
        fn new(secondary: u8, port_type: u8, lcap: (u8, u8), lsta: (u8, u8)) -> Self {
            let mut regs = vec![0u32; 64];
            regs[0] = 0x2030_8086;
            regs[1] = 0x0010_0000; // status: capability list
            regs[3] = 0x0001_0000; // header type 1
            regs[0x18 / 4] = (secondary as u32) << 8 | 0xb2;
            regs[0x34 / 4] = 0x40;
            regs[0x40 / 4] = ((port_type as u32) << 4 | 0x2) << 16 | 0x0010;
            regs[0x4c / 4] = (lcap.1 as u32) << 4 | lcap.0 as u32;
            regs[0x50 / 4] = ((lsta.1 as u32) << 4 | lsta.0 as u32) << 16;
            Bridge { regs, writes: 0 }
        }
    }

    impl ConfigSpace for Bridge {
        fn read32(&mut self, offset: u16) -> u32 {
            self.regs.get(offset as usize / 4).copied().unwrap_or(0)
        }
        fn write32(&mut self, _offset: u16, _value: u32) {
            self.writes += 1;
        }
    }

    #[test]
    fn bandwidth_ceilings() {
        let g3x8 = link_bandwidth_gbps(3, 8).unwrap();
        let g3x16 = link_bandwidth_gbps(3, 16).unwrap();
        assert!((g3x8 - 7.877).abs() < 0.01, "{g3x8}");
        assert!((g3x16 - 15.754).abs() < 0.01, "{g3x16}");
        assert!((link_bandwidth_gbps(1, 8).unwrap() - 2.0).abs() < 1e-9);
        assert_eq!(link_bandwidth_gbps(0, 8), None);
        // os-376 measured 6.26 up / 6.76 down: 79 % / 86 % of the x8 ceiling.
        assert!((6.26 / g3x8 - 0.795).abs() < 0.01);
        assert!((6.76 / g3x8 - 0.858).abs() < 0.01);
    }

    #[test]
    fn workstation_today_is_slot_narrow() {
        // Board gen4 x16, link gen3 x8; predicted port (slot 1): gen3 x8.
        let board = link(4, 16, 3, 8);
        let d = diagnose(&board, Some(&link(3, 8, 3, 8)));
        assert_eq!(d.verdict, LinkVerdict::SlotNarrow);
        assert_eq!((d.best_speed, d.best_width), (3, 8));
        // The same board in a full x16 gen3 slot (slots 2 / 4): only gen3 is
        // left, which no slot of the Dell 5820 can fix.
        let moved = diagnose(&link(4, 16, 3, 16), Some(&link(3, 16, 3, 16)));
        assert_eq!(moved.verdict, LinkVerdict::SlotSlow);
        let gain = moved.now_gbps.unwrap() / d.now_gbps.unwrap();
        assert!((gain - 2.0).abs() < 1e-9);
    }

    #[test]
    fn port_wider_than_link_is_seating() {
        // P3: the port could do x16 but only x8 trained -> reseat, not move.
        let d = diagnose(&link(4, 16, 3, 8), Some(&link(3, 16, 3, 8)));
        assert_eq!(d.verdict, LinkVerdict::TrainedNarrow);
        assert_eq!(d.best_width, 16);
    }

    #[test]
    fn rest_is_not_a_speed_fault() {
        // At rest the board drops to gen1 but keeps x8: the width finding
        // (the slot) stands; idle shows only once the width is fine.
        let narrow_idle = diagnose(&link(4, 16, 1, 8), Some(&link(3, 8, 1, 8)));
        assert_eq!(narrow_idle.verdict, LinkVerdict::SlotNarrow);
        let full_idle = diagnose(&link(4, 16, 1, 16), Some(&link(3, 16, 1, 16)));
        assert_eq!(full_idle.verdict, LinkVerdict::Idle);
        let slow = diagnose(&link(4, 16, 2, 16), Some(&link(3, 16, 2, 16)));
        assert_eq!(slow.verdict, LinkVerdict::TrainedSlow);
        let full = diagnose(&link(3, 16, 3, 16), Some(&link(3, 16, 3, 16)));
        assert_eq!(full.verdict, LinkVerdict::Full);
    }

    #[test]
    fn without_port_only_board_side() {
        assert_eq!(
            diagnose(&link(4, 16, 3, 8), None).verdict,
            LinkVerdict::NoPort
        );
        assert_eq!(diagnose(&link(3, 8, 3, 8), None).verdict, LinkVerdict::Full);
        for v in [
            LinkVerdict::Full,
            LinkVerdict::SlotNarrow,
            LinkVerdict::TrainedNarrow,
            LinkVerdict::SlotSlow,
            LinkVerdict::Idle,
            LinkVerdict::TrainedSlow,
            LinkVerdict::NoPort,
        ] {
            assert!(v.text().is_ascii() && !v.text().is_empty());
        }
    }

    #[test]
    fn finds_port_by_secondary_bus_without_writes() {
        // The workstation's b2 bus (SIM shape of the lspci list): ports A, C, D.
        let mut ports = [
            Bridge::new(0xb3, PORT_ROOT, (3, 8), (3, 8)),
            Bridge::new(0xb4, PORT_ROOT, (3, 4), (3, 4)),
            Bridge::new(0xb5, PORT_ROOT, (3, 4), (1, 0)),
        ];
        let i = find_upstream_port(&mut ports, 0xb3).unwrap();
        assert_eq!(i, 0);
        let p = caps::pcie_link(&mut ports[i]).unwrap();
        assert_eq!((p.device_type, p.max_speed, p.max_width), (PORT_ROOT, 3, 8));
        assert_eq!(find_upstream_port(&mut ports, 0xb6), None);
        // An upstream switch port with the right bus is not a slot.
        let mut up = [Bridge::new(0xb3, 0x5, (3, 16), (3, 16))];
        assert_eq!(find_upstream_port(&mut up, 0xb3), None);
        assert!(ports.iter().chain(up.iter()).all(|b| b.writes == 0));
    }

    #[test]
    fn board_header_has_no_secondary_bus() {
        let mut cfg = gpu_sim::pci::SimConfig::workstation_rtx3070ti();
        assert_eq!(secondary_bus(&mut cfg), None);
        assert!(!is_downstream_port(&mut cfg));
        assert_eq!(cfg.writes, 0);
    }
}
