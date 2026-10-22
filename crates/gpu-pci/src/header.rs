//! Type-0 configuration header (read-only helpers).

use gpu_core::error::{GpuError, GpuResult};
use gpu_core::hal::ConfigSpace;
use gpu_core::ids::PciId;

pub const REG_ID: u16 = 0x00;
pub const REG_COMMAND: u16 = 0x04;
pub const REG_STATUS: u16 = 0x06;
pub const REG_CLASS: u16 = 0x08;
pub const REG_HEADER: u16 = 0x0c;
pub const REG_BAR0: u16 = 0x10;
pub const REG_SUBSYS: u16 = 0x2c;
pub const REG_CAP_PTR: u16 = 0x34;

pub const CMD_IO: u16 = 1 << 0;
pub const CMD_MEMORY: u16 = 1 << 1;
pub const CMD_BUS_MASTER: u16 = 1 << 2;
pub const CMD_INTX_DISABLE: u16 = 1 << 10;
pub const STATUS_CAP_LIST: u16 = 1 << 4;

/// Read vendor/device/class/subsystem. Returns `NotPresent` when the
/// function does not exist (vendor reads as 0xffff).
pub fn read_identity<C: ConfigSpace + ?Sized>(cfg: &mut C) -> GpuResult<PciId> {
    let id = cfg.read32(REG_ID);
    let vendor = id as u16;
    if vendor == 0xffff || vendor == 0 {
        return Err(GpuError::NotPresent);
    }
    let class = cfg.read32(REG_CLASS);
    let sub = cfg.read32(REG_SUBSYS);
    Ok(PciId {
        vendor,
        device: (id >> 16) as u16,
        subsys_vendor: sub as u16,
        subsys_device: (sub >> 16) as u16,
        revision: class as u8,
        prog_if: (class >> 8) as u8,
        subclass: (class >> 16) as u8,
        class: (class >> 24) as u8,
    })
}

/// Header type without the multi-function bit. Only type 0 (endpoint) has six BARs.
pub fn header_type<C: ConfigSpace + ?Sized>(cfg: &mut C) -> u8 {
    cfg.read8(REG_HEADER + 2) & 0x7f
}

pub fn command<C: ConfigSpace + ?Sized>(cfg: &mut C) -> u16 {
    cfg.read16(REG_COMMAND)
}

pub fn status<C: ConfigSpace + ?Sized>(cfg: &mut C) -> u16 {
    cfg.read16(REG_STATUS)
}
