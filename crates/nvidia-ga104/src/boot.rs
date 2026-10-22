//! NV_PMC_BOOT_0 and NV_PMC_BOOT_42 decoding.
//!
//! Source: NVIDIA open-gpu-doc `manuals/ampere/ga100/dev_boot.ref.txt` (MIT):
//! BOOT_0 @0x0: MINOR_REVISION 3:0, MAJOR_REVISION 7:4, IMPLEMENTATION 23:20,
//! ARCHITECTURE 28:24 (GA100 family = 0x17).
//! BOOT_42 @0xA00: MINOR_EXTENDED_REVISION 11:8, MINOR_REVISION 15:12,
//! MAJOR_REVISION 19:16, IMPLEMENTATION 23:20, ARCHITECTURE 28:24, CHIP_ID 28:20.
//! Linux nova-core declares BOOT_42 architecture as 29:24; we follow NVIDIA's
//! manual and report bit 29 separately so the hardware test can settle it.
//! nouveau computes the chipset as (boot0 & 0x1ff00000) >> 20; GA104 = 0x174.

pub const ARCH_AMPERE: u8 = 0x17;
pub const CHIP_GA104: u16 = 0x174;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Boot0 {
    pub raw: u32,
    pub architecture: u8,
    pub implementation: u8,
    pub major_revision: u8,
    pub minor_revision: u8,
}

impl Boot0 {
    pub fn decode(raw: u32) -> Boot0 {
        Boot0 {
            raw,
            architecture: ((raw >> 24) & 0x1f) as u8,
            implementation: ((raw >> 20) & 0xf) as u8,
            major_revision: ((raw >> 4) & 0xf) as u8,
            minor_revision: (raw & 0xf) as u8,
        }
    }

    /// Chip id as nouveau computes it (bits 28:20).
    pub fn chip_id(&self) -> u16 {
        ((self.raw >> 20) & 0x1ff) as u16
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Boot42 {
    pub raw: u32,
    pub chip_id: u16,
    pub major_revision: u8,
    pub minor_revision: u8,
    pub minor_extended_revision: u8,
    /// Bit 29: zero per NVIDIA's manual, part of `architecture` per nova-core.
    pub bit29: bool,
}

impl Boot42 {
    pub fn decode(raw: u32) -> Boot42 {
        Boot42 {
            raw,
            chip_id: ((raw >> 20) & 0x1ff) as u16,
            major_revision: ((raw >> 16) & 0xf) as u8,
            minor_revision: ((raw >> 12) & 0xf) as u8,
            minor_extended_revision: ((raw >> 8) & 0xf) as u8,
            bit29: raw & (1 << 29) != 0,
        }
    }
}

pub fn chip_name(chip_id: u16) -> &'static str {
    match chip_id {
        0x170 => "GA100",
        0x172 => "GA102",
        0x173 => "GA103",
        0x174 => "GA104",
        0x176 => "GA106",
        0x177 => "GA107",
        _ => "necunoscut",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_ga104() {
        // Field values built from the manual, not from a captured register.
        let raw = (0x17 << 24) | (0x4 << 20) | (0xa << 4) | 0x1;
        let b = Boot0::decode(raw);
        assert_eq!(b.architecture, ARCH_AMPERE);
        assert_eq!(b.implementation, 4);
        assert_eq!(b.chip_id(), CHIP_GA104);
        assert_eq!((b.major_revision, b.minor_revision), (0xa, 1));
        assert_eq!(chip_name(b.chip_id()), "GA104");
        // nouveau's mask gives the same value
        assert_eq!((raw & 0x1ff0_0000) >> 20, CHIP_GA104 as u32);
    }

    #[test]
    fn decode_boot42() {
        let raw = (0x174 << 20) | (0xa << 16) | (0x1 << 12) | (0x2 << 8);
        let b = Boot42::decode(raw);
        assert_eq!(b.chip_id, CHIP_GA104);
        assert_eq!(
            (
                b.major_revision,
                b.minor_revision,
                b.minor_extended_revision
            ),
            (0xa, 1, 2)
        );
        assert!(!b.bit29);
        assert!(Boot42::decode(raw | 1 << 29).bit29);
    }
}
