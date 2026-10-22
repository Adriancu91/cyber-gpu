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
}
