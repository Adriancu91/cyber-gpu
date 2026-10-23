//! SIMULATION of a BAR0 register window. Sparse: unknown offsets read as
//! `default` (real NVIDIA hardware returns values like 0xbadf.... for
//! unclaimed registers; we do not model that).

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use gpu_core::hal::Mmio;

pub struct SimMmio {
    len: u64,
    regs: BTreeMap<u64, u32>,
    pub default: u32,
    /// All reads return 0xffffffff (device in D3cold / link down).
    pub absent: bool,
    pub reads: Vec<u64>,
    pub writes: Vec<(u64, u32)>,
    /// Registers whose value changes after a number of reads of that
    /// offset: `offset -> (reads left, new value)`. Models a status that
    /// firmware updates while the driver polls.
    delayed: BTreeMap<u64, (u32, u32)>,
}

impl SimMmio {
    pub fn new(len: u64) -> Self {
        SimMmio {
            len,
            regs: BTreeMap::new(),
            default: 0,
            absent: false,
            reads: Vec::new(),
            writes: Vec::new(),
            delayed: BTreeMap::new(),
        }
    }

    /// SIM: GA104 BAR0 (16 MiB) with NV_PMC_BOOT_0 encoding chip id 0x174.
    /// The revision bits (0xa1) are a SIM value.
    pub fn ga104_bar0() -> Self {
        let mut m = Self::new(16 << 20);
        m.set(0x0000_0000, 0x1740_00a1);
        m
    }

    pub fn set(&mut self, off: u64, v: u32) {
        self.regs.insert(off, v);
    }

    /// After `reads` more reads of `off`, the register reads `v`.
    pub fn set_after(&mut self, off: u64, reads: u32, v: u32) {
        self.delayed.insert(off, (reads, v));
    }

    /// Place `bytes` at `base` as little-endian dwords (the PROM window).
    pub fn load_bytes(&mut self, base: u64, bytes: &[u8]) {
        for (i, c) in bytes.chunks(4).enumerate() {
            let mut w = [0u8; 4];
            w[..c.len()].copy_from_slice(c);
            self.set(base + 4 * i as u64, u32::from_le_bytes(w));
        }
    }
}

impl Mmio for SimMmio {
    fn len(&self) -> u64 {
        self.len
    }
    fn read32_raw(&mut self, offset: u64) -> u32 {
        self.reads.push(offset);
        if self.absent {
            return 0xffff_ffff;
        }
        if let Some((left, v)) = self.delayed.get_mut(&offset) {
            if *left == 0 {
                let v = *v;
                self.delayed.remove(&offset);
                self.regs.insert(offset, v);
            } else {
                *left -= 1;
            }
        }
        *self.regs.get(&offset).unwrap_or(&self.default)
    }
    fn write32_raw(&mut self, offset: u64, value: u32) {
        self.writes.push((offset, value));
        if !self.absent {
            self.regs.insert(offset, value);
        }
    }
}
