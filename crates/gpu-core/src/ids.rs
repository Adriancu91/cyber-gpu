//! PCI identity.

use core::fmt;

pub const VENDOR_NVIDIA: u16 = 0x10de;
pub const VENDOR_AMD: u16 = 0x1002;
pub const VENDOR_INTEL: u16 = 0x8086;
/// Class code 0x03: display controller.
pub const CLASS_DISPLAY: u8 = 0x03;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PciAddress {
    pub segment: u16,
    pub bus: u8,
    pub dev: u8,
    pub func: u8,
}

impl fmt::Display for PciAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:02x}:{:02x}.{}",
            self.segment, self.bus, self.dev, self.func
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PciId {
    pub vendor: u16,
    pub device: u16,
    pub subsys_vendor: u16,
    pub subsys_device: u16,
    pub revision: u8,
    pub class: u8,
    pub subclass: u8,
    pub prog_if: u8,
}

impl fmt::Display for PciId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04x}:{:04x} (subsys {:04x}:{:04x}) rev {:02x} clasa {:02x}:{:02x}:{:02x}",
            self.vendor,
            self.device,
            self.subsys_vendor,
            self.subsys_device,
            self.revision,
            self.class,
            self.subclass,
            self.prog_if
        )
    }
}
