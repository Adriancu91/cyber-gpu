//! GA104 PCI device ids.
//!
//! Source: pci.ids (github.com/pciutils/pciids, `pci.ids` lines ~13578-13618,
//! GPL-2.0+/BSD-3-Clause data, cited not copied as code), cross-checked for
//! 2482/2484/2486/2488/2489 with NVIDIA open-gpu-kernel-modules README (MIT).
//!
//! The PCI id only makes a device a *candidate*; the chip is confirmed from
//! NV_PMC_BOOT_0 (see [`crate::probe`]).

pub const VENDOR_NVIDIA: u16 = 0x10de;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Model {
    pub device: u16,
    pub name: &'static str,
}

pub const GA104_MODELS: &[Model] = &[
    Model {
        device: 0x2482,
        name: "GeForce RTX 3070 Ti",
    },
    Model {
        device: 0x248c,
        name: "GeForce RTX 3070 Ti",
    },
    Model {
        device: 0x2484,
        name: "GeForce RTX 3070",
    },
    Model {
        device: 0x2488,
        name: "GeForce RTX 3070 LHR",
    },
    Model {
        device: 0x248d,
        name: "GeForce RTX 3070",
    },
    Model {
        device: 0x24c8,
        name: "GeForce RTX 3070 GDDR6X",
    },
    Model {
        device: 0x2486,
        name: "GeForce RTX 3060 Ti",
    },
    Model {
        device: 0x2489,
        name: "GeForce RTX 3060 Ti LHR",
    },
    Model {
        device: 0x248e,
        name: "GeForce RTX 3060 Ti",
    },
    Model {
        device: 0x24c9,
        name: "GeForce RTX 3060 Ti GDDR6X",
    },
    Model {
        device: 0x2487,
        name: "GeForce RTX 3060 (GA104)",
    },
    Model {
        device: 0x24c7,
        name: "GeForce RTX 3060 8GB (GA104)",
    },
    Model {
        device: 0x248a,
        name: "CMP 70HX",
    },
    Model {
        device: 0x249c,
        name: "GeForce RTX 3080 Mobile",
    },
    Model {
        device: 0x24dc,
        name: "GeForce RTX 3080 Mobile",
    },
    Model {
        device: 0x249d,
        name: "GeForce RTX 3070 Mobile",
    },
    Model {
        device: 0x24dd,
        name: "GeForce RTX 3070 Mobile",
    },
    Model {
        device: 0x24a0,
        name: "GeForce RTX 3070 Ti Laptop",
    },
    Model {
        device: 0x24e0,
        name: "GeForce RTX 3070 Ti Laptop",
    },
    Model {
        device: 0x24b0,
        name: "RTX A4000",
    },
    Model {
        device: 0x24b1,
        name: "RTX A4000H",
    },
];

/// GA106 PCI device ids (C-95 / C-116, the ASUS TUF F15 laptop: "RTX 3060 Laptop").
///
/// Source: NVIDIA open-gpu-kernel-modules@570.144 README (MIT), table of
/// supported GPUs; the GA106 attribution of 0x25xx below 0x2580 follows
/// pci.ids (cited, not copied). Note: the README also lists "RTX 3060
/// Laptop GPU" names on GA104 (0x249c, 0x24a0) and GA107 (0x25a0, 0x25a2)
/// ids, so the name proves nothing: the chip is confirmed from BOOT_0.
/// OBSERVED on Adrian's laptop (12 Nov 2026, os-295 `gpu identifica`):
/// `10de:2520` at 01:00.0. GA106 is recognised for READ-ONLY steps only
/// (identification, GFW status, VBIOS, firmware file check); the writing
/// chain stays GA104-only until Adrian's separate "DA" (C-116 step 3).
pub const GA106_MODELS: &[Model] = &[
    Model {
        device: 0x2503,
        name: "GeForce RTX 3060",
    },
    Model {
        device: 0x2504,
        name: "GeForce RTX 3060 LHR",
    },
    Model {
        device: 0x2507,
        name: "GeForce RTX 3050",
    },
    Model {
        device: 0x2508,
        name: "GeForce RTX 3050 OEM",
    },
    Model {
        device: 0x2520,
        name: "GeForce RTX 3060 Laptop",
    },
    Model {
        device: 0x2521,
        name: "GeForce RTX 3060 Laptop",
    },
    Model {
        device: 0x2523,
        name: "GeForce RTX 3050 Ti Laptop",
    },
    Model {
        device: 0x2531,
        name: "RTX A2000",
    },
    Model {
        device: 0x2544,
        name: "GeForce RTX 3060",
    },
    Model {
        device: 0x2560,
        name: "GeForce RTX 3060 Laptop",
    },
    Model {
        device: 0x2563,
        name: "GeForce RTX 3050 Ti Laptop",
    },
    Model {
        device: 0x2571,
        name: "RTX A2000 12GB",
    },
];

/// The laptop's own id (OBSERVED 12 Nov 2026, `hw` / `gpu identifica` on os-295).
pub const LAPTOP_DEVICE: u16 = 0x2520;

/// Which Ampere chip a PCI id is a candidate for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    /// The workstation board; the only chip the writing chain accepts.
    Ga104,
    /// C-116: read-only recognition (+ the minimal wake-up, behind a permit).
    Ga106,
}

impl Family {
    /// The chip id BOOT_0 must report for this family.
    pub fn chip_id(self) -> u16 {
        match self {
            Family::Ga104 => crate::boot::CHIP_GA104,
            Family::Ga106 => crate::boot::CHIP_GA106,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Family::Ga104 => "GA104",
            Family::Ga106 => "GA106",
        }
    }

    /// Whether any step that writes to the GPU registers may run on this
    /// family. Only GA104 (the workstation); GA106 needs Adrian's separate
    /// "DA" after the read-only journal from the laptop (C-116 step 3).
    pub fn writes_allowed(self) -> bool {
        matches!(self, Family::Ga104)
    }
}

/// GA106 table lookup (read-only recognition, C-116).
pub fn lookup_ga106(vendor: u16, device: u16) -> Option<&'static Model> {
    if vendor != VENDOR_NVIDIA {
        return None;
    }
    GA106_MODELS.iter().find(|m| m.device == device)
}

/// GA104 or GA106 candidate. [`lookup`] stays GA104-only.
pub fn lookup_ampere(vendor: u16, device: u16) -> Option<(Family, &'static Model)> {
    lookup(vendor, device)
        .map(|m| (Family::Ga104, m))
        .or_else(|| lookup_ga106(vendor, device).map(|m| (Family::Ga106, m)))
}

/// The board this project targets.
pub const TARGET_DEVICE: u16 = 0x2482;

/// HDA audio function on the same board (function 1). Never driven by us.
pub const GA104_HDA: u16 = 0x228b;

pub fn lookup(vendor: u16, device: u16) -> Option<&'static Model> {
    if vendor != VENDOR_NVIDIA {
        return None;
    }
    GA104_MODELS.iter().find(|m| m.device == device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table() {
        assert_eq!(lookup(0x10de, 0x2482).unwrap().name, "GeForce RTX 3070 Ti");
        assert!(
            lookup(0x10de, GA104_HDA).is_none(),
            "audio function is not a GPU"
        );
        assert!(lookup(0x1002, 0x2482).is_none(), "vendor checked");
        assert!(lookup(0x10de, 0x2204).is_none(), "GA102 is not GA104");
        let mut ids: alloc::vec::Vec<u16> = GA104_MODELS.iter().map(|m| m.device).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), GA104_MODELS.len(), "no duplicates");
    }

    #[test]
    fn ga106_table_is_separate_and_read_only() {
        assert!(lookup(0x10de, 0x2520).is_none(), "GA104 lookup unchanged");
        assert_eq!(
            lookup_ga106(0x10de, LAPTOP_DEVICE).unwrap().name,
            "GeForce RTX 3060 Laptop"
        );
        assert!(lookup_ga106(0x1002, 0x2520).is_none(), "vendor checked");
        assert_eq!(lookup_ampere(0x10de, 0x2482).unwrap().0, Family::Ga104);
        assert_eq!(lookup_ampere(0x10de, 0x2520).unwrap().0, Family::Ga106);
        assert!(
            lookup_ampere(0x10de, 0x25a0).is_none(),
            "GA107 not recognised"
        );
        assert!(!Family::Ga106.writes_allowed());
        assert!(Family::Ga104.writes_allowed());
        assert_eq!(Family::Ga106.chip_id(), 0x176);
        assert_eq!(Family::Ga106.name(), "GA106");
        for m in GA106_MODELS {
            assert!(lookup(0x10de, m.device).is_none(), "no id in both tables");
        }
        let mut ids: alloc::vec::Vec<u16> = GA106_MODELS.iter().map(|m| m.device).collect();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), GA106_MODELS.len(), "no duplicates");
    }
}
