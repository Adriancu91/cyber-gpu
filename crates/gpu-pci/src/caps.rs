//! Capability lists (read-only).
//!
//! Standard list: PCI 3.0 §6.7. Extended list (offset 0x100+, needs ECAM):
//! PCIe Base Spec §7.6. Both walks are bounded so a corrupt or looping list
//! can never hang the kernel.

use alloc::vec::Vec;

use gpu_core::hal::ConfigSpace;

use crate::header::{self, REG_CAP_PTR, STATUS_CAP_LIST};

pub const CAP_PM: u8 = 0x01;
pub const CAP_MSI: u8 = 0x05;
pub const CAP_VENDOR: u8 = 0x09;
pub const CAP_PCIE: u8 = 0x10;
pub const CAP_MSIX: u8 = 0x11;

pub const EXT_AER: u16 = 0x0001;
pub const EXT_RESIZABLE_BAR: u16 = 0x0015;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cap {
    pub id: u8,
    pub offset: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExtCap {
    pub id: u16,
    pub version: u8,
    pub offset: u16,
}

pub fn list<C: ConfigSpace + ?Sized>(cfg: &mut C) -> Vec<Cap> {
    let mut out = Vec::new();
    if header::status(cfg) & STATUS_CAP_LIST == 0 {
        return out;
    }
    let mut ptr = (cfg.read8(REG_CAP_PTR) & 0xfc) as u16;
    // 48 = (256 - 64) / 4: the most entries that fit in the legacy space.
    for _ in 0..48 {
        if ptr < 0x40 {
            break;
        }
        let id = cfg.read8(ptr);
        out.push(Cap { id, offset: ptr });
        ptr = (cfg.read8(ptr + 1) & 0xfc) as u16;
    }
    out
}

pub fn find<C: ConfigSpace + ?Sized>(cfg: &mut C, id: u8) -> Option<u16> {
    list(cfg).into_iter().find(|c| c.id == id).map(|c| c.offset)
}

pub fn ext_list<C: ConfigSpace + ?Sized>(cfg: &mut C) -> Vec<ExtCap> {
    let mut out = Vec::new();
    let mut ptr = 0x100u16;
    // 960 = (4096 - 256) / 4
    for _ in 0..960 {
        let h = cfg.read32(ptr);
        if h == 0 || h == 0xffff_ffff {
            break;
        }
        out.push(ExtCap {
            id: h as u16,
            version: ((h >> 16) & 0xf) as u8,
            offset: ptr,
        });
        let next = ((h >> 20) & 0xffc) as u16;
        if next < 0x100 || next <= ptr {
            break;
        }
        ptr = next;
    }
    out
}

pub fn find_ext<C: ConfigSpace + ?Sized>(cfg: &mut C, id: u16) -> Option<u16> {
    ext_list(cfg)
        .into_iter()
        .find(|c| c.id == id)
        .map(|c| c.offset)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PcieLink {
    pub device_type: u8,
    /// Encoded speed: 1 = 2.5 GT/s, 2 = 5, 3 = 8, 4 = 16, 5 = 32.
    pub max_speed: u8,
    pub max_width: u8,
    pub speed: u8,
    pub width: u8,
}

impl PcieLink {
    pub fn speed_name(code: u8) -> &'static str {
        match code {
            1 => "2.5GT/s",
            2 => "5GT/s",
            3 => "8GT/s",
            4 => "16GT/s",
            5 => "32GT/s",
            6 => "64GT/s",
            _ => "?",
        }
    }
}

pub fn pcie_link<C: ConfigSpace + ?Sized>(cfg: &mut C) -> Option<PcieLink> {
    let p = find(cfg, CAP_PCIE)?;
    let caps = cfg.read16(p + 2);
    let lcap = cfg.read32(p + 0x0c);
    let lsta = cfg.read16(p + 0x12);
    Some(PcieLink {
        device_type: ((caps >> 4) & 0xf) as u8,
        max_speed: (lcap & 0xf) as u8,
        max_width: ((lcap >> 4) & 0x3f) as u8,
        speed: (lsta & 0xf) as u8,
        width: ((lsta >> 4) & 0x3f) as u8,
    })
}

/// Power state from the PM capability: 0 = D0 .. 3 = D3hot.
pub fn power_state<C: ConfigSpace + ?Sized>(cfg: &mut C) -> Option<u8> {
    let p = find(cfg, CAP_PM)?;
    Some((cfg.read16(p + 4) & 0x3) as u8)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResizableBar {
    pub bar_index: u8,
    /// Bit n set: size 2^n MiB supported (n = 0 is 1 MiB).
    pub supported_mib_log2: u32,
    /// Current size as log2(MiB).
    pub current_mib_log2: u8,
}

impl ResizableBar {
    pub fn current_bytes(&self) -> u64 {
        (1u64 << 20) << self.current_mib_log2
    }
    pub fn max_bytes(&self) -> Option<u64> {
        if self.supported_mib_log2 == 0 {
            return None;
        }
        Some((1u64 << 20) << (31 - self.supported_mib_log2.leading_zeros()))
    }
}

/// Resizable BAR extended capability (PCIe Base Spec §7.8.6), read-only.
pub fn resizable_bars<C: ConfigSpace + ?Sized>(cfg: &mut C) -> Vec<ResizableBar> {
    let mut out = Vec::new();
    let Some(base) = find_ext(cfg, EXT_RESIZABLE_BAR) else {
        return out;
    };
    let first_ctrl = cfg.read32(base + 8);
    let n = ((first_ctrl >> 5) & 0x7).clamp(1, 6) as u16;
    for k in 0..n {
        let cap = cfg.read32(base + 4 + 8 * k);
        let ctrl = cfg.read32(base + 8 + 8 * k);
        out.push(ResizableBar {
            bar_index: (ctrl & 0x7) as u8,
            supported_mib_log2: cap >> 4,
            current_mib_log2: ((ctrl >> 8) & 0x3f) as u8,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_sim::pci::SimConfig;

    #[test]
    fn walks_lists() {
        let mut cfg = SimConfig::ga104_rtx3070ti();
        let ids: Vec<u8> = list(&mut cfg).iter().map(|c| c.id).collect();
        assert_eq!(ids, [CAP_PM, CAP_MSI, CAP_PCIE]);
        assert!(find_ext(&mut cfg, EXT_RESIZABLE_BAR).is_some());
        let link = pcie_link(&mut cfg).unwrap();
        assert_eq!(
            (link.max_speed, link.max_width, link.speed, link.width),
            (4, 16, 4, 16)
        );
        assert_eq!(power_state(&mut cfg), Some(0));
        let rb = resizable_bars(&mut cfg);
        assert_eq!(rb.len(), 2);
        assert_eq!(rb[0].bar_index, 1);
        assert_eq!(rb[0].current_bytes(), 256 << 20);
        assert_eq!(rb[0].max_bytes(), Some(8 << 30));
    }

    #[test]
    fn looping_list_terminates() {
        let mut cfg = SimConfig::ga104_rtx3070ti();
        cfg.make_cap_loop();
        assert!(list(&mut cfg).len() <= 48);
        assert!(ext_list(&mut cfg).len() <= 960);
    }

    #[test]
    fn no_writes_during_walks() {
        let mut cfg = SimConfig::ga104_rtx3070ti();
        let _ = list(&mut cfg);
        let _ = ext_list(&mut cfg);
        let _ = resizable_bars(&mut cfg);
        let _ = pcie_link(&mut cfg);
        assert_eq!(cfg.writes, 0);
    }
}
