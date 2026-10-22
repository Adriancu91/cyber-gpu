//! Pushbuffer method headers and GPFIFO entries for AMPERE_CHANNEL_GPFIFO_A.
//!
//! Source: NVIDIA open-gpu-doc `classes/host/clc56f.h` (MIT), lines ~265-332:
//! - method header: SEC_OP 31:29, METHOD_COUNT / IMMD_DATA 28:16,
//!   METHOD_SUBCHANNEL 15:13, METHOD_ADDRESS 11:0 (= method byte offset >> 2);
//! - GPFIFO ENTRY0: FETCH 0:0, GET 31:2 (address bits 31:2);
//!   ENTRY1: GET_HI 7:0 (address bits 39:32), LEVEL 9:9, LENGTH 30:10 (dwords), SYNC 31:31.
//!
//! STATUS: encoding only, checked against the field definitions above.
//! Nothing here has been executed by a GPU; there is no channel yet (init.rs I9).

use gpu_core::error::{GpuError, GpuResult};

/// Class ids (open-gpu-doc, MIT).
pub const AMPERE_CHANNEL_GPFIFO_A: u32 = 0xc56f;
pub const AMPERE_COMPUTE_B: u32 = 0xc7c0;

/// Doorbell (NOTIFY_CHANNEL_PENDING) offset in BAR0 for GA10x.
/// INFERRED: nouveau vfn/ga100.c base 0xb80000 + user 0x030000 + 0x90.
/// Not on the read allow-list and never written by this crate yet.
pub const DOORBELL_INFERRED: u64 = 0x00bb_0090;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum SecOp {
    IncMethod = 1,
    NonIncMethod = 3,
    ImmdDataMethod = 4,
    OneInc = 5,
}

const MAX_COUNT: u32 = 0x1fff; // 13 bits
const MAX_ADDR_BYTES: u32 = 0x3ffc; // 12-bit dword address

fn check_method(subch: u8, method: u32) -> GpuResult<()> {
    if subch > 7 {
        return Err(GpuError::InvalidArgument("subcanal > 7"));
    }
    if method & 3 != 0 || method > MAX_ADDR_BYTES {
        return Err(GpuError::InvalidArgument("adresa metodei invalida"));
    }
    Ok(())
}

/// Header for a method group followed by `count` data dwords.
pub fn method_header(op: SecOp, subch: u8, method: u32, count: u32) -> GpuResult<u32> {
    check_method(subch, method)?;
    if op == SecOp::ImmdDataMethod {
        return Err(GpuError::InvalidArgument("folositi immediate()"));
    }
    if count == 0 || count > MAX_COUNT {
        return Err(GpuError::InvalidArgument("numar de date invalid"));
    }
    Ok(((op as u32) << 29) | (count << 16) | ((subch as u32) << 13) | (method >> 2))
}

/// Single-dword header carrying a 13-bit immediate value.
pub fn immediate(subch: u8, method: u32, data: u32) -> GpuResult<u32> {
    check_method(subch, method)?;
    if data > MAX_COUNT {
        return Err(GpuError::InvalidArgument("valoare imediata > 13 biti"));
    }
    Ok(((SecOp::ImmdDataMethod as u32) << 29)
        | (data << 16)
        | ((subch as u32) << 13)
        | (method >> 2))
}

/// Growable pushbuffer in host memory (dwords).
pub struct PushBuf<'a> {
    buf: &'a mut [u32],
    len: usize,
}

impl<'a> PushBuf<'a> {
    pub fn new(buf: &'a mut [u32]) -> Self {
        PushBuf { buf, len: 0 }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn words(&self) -> &[u32] {
        &self.buf[..self.len]
    }
    fn push(&mut self, w: u32) -> GpuResult<()> {
        let slot = self
            .buf
            .get_mut(self.len)
            .ok_or(GpuError::OutOfMemory { requested: 4 })?;
        *slot = w;
        self.len += 1;
        Ok(())
    }
    /// Incrementing method group: data[i] goes to method + 4*i.
    pub fn inc(&mut self, subch: u8, method: u32, data: &[u32]) -> GpuResult<()> {
        if self.len + 1 + data.len() > self.buf.len() {
            return Err(GpuError::OutOfMemory {
                requested: (4 * (1 + data.len())) as u64,
            });
        }
        self.push(method_header(
            SecOp::IncMethod,
            subch,
            method,
            data.len() as u32,
        )?)?;
        for &d in data {
            self.push(d)?;
        }
        Ok(())
    }
    pub fn imm(&mut self, subch: u8, method: u32, data: u32) -> GpuResult<()> {
        let h = immediate(subch, method, data)?;
        self.push(h)
    }
}

/// One 8-byte GPFIFO entry pointing at a pushbuffer segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GpFifoEntry {
    pub entry0: u32,
    pub entry1: u32,
}

/// `gpu_va` is the GPU virtual address of the segment (40 bits, dword aligned);
/// `dwords` its length.
pub fn gpfifo_entry(gpu_va: u64, dwords: u32, sync: bool) -> GpuResult<GpFifoEntry> {
    if gpu_va & 3 != 0 || gpu_va >> 40 != 0 {
        return Err(GpuError::InvalidArgument(
            "adresa GPFIFO: aliniere sau peste 40 de biti",
        ));
    }
    if dwords == 0 || dwords >= 1 << 21 {
        return Err(GpuError::InvalidArgument("lungime GPFIFO invalida"));
    }
    Ok(GpFifoEntry {
        entry0: (gpu_va as u32) & !3,
        entry1: ((gpu_va >> 32) as u32 & 0xff) | (dwords << 10) | ((sync as u32) << 31),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headers_follow_field_layout() {
        // INC, subchannel 1, method 0x100, 2 dwords
        assert_eq!(
            method_header(SecOp::IncMethod, 1, 0x100, 2),
            Ok(0x2002_2040)
        );
        // NON_INC, subchannel 0, method 0x0, 1 dword
        assert_eq!(method_header(SecOp::NonIncMethod, 0, 0, 1), Ok(0x6001_0000));
        // IMMD, subchannel 7, method 0x3ffc, data 0x1fff; bit 12 is outside
        // METHOD_ADDRESS (11:0) and stays 0
        assert_eq!(immediate(7, 0x3ffc, 0x1fff), Ok(0x9fff_efff));
        assert!(method_header(SecOp::IncMethod, 8, 0, 1).is_err());
        assert!(method_header(SecOp::IncMethod, 0, 0x4000, 1).is_err());
        assert!(method_header(SecOp::IncMethod, 0, 0x102, 1).is_err());
        assert!(method_header(SecOp::IncMethod, 0, 0, 0x2000).is_err());
        assert!(immediate(0, 0, 0x2000).is_err());
    }

    #[test]
    fn pushbuf_bounds() {
        let mut mem = [0u32; 4];
        let mut pb = PushBuf::new(&mut mem);
        pb.inc(0, 0x200, &[1, 2]).unwrap();
        assert!(pb.inc(0, 0x200, &[1, 2]).is_err(), "no partial group");
        assert_eq!(pb.len(), 3);
        pb.imm(0, 0x10, 5).unwrap();
        assert!(pb.imm(0, 0x10, 5).is_err());
        assert_eq!(pb.words()[0] >> 29, 1);
    }

    #[test]
    fn gpfifo_fields() {
        let e = gpfifo_entry(0x12_3456_7890, 0x100, true).unwrap();
        assert_eq!(e.entry0, 0x3456_7890);
        assert_eq!(e.entry1 & 0xff, 0x12);
        assert_eq!((e.entry1 >> 10) & 0x1f_ffff, 0x100);
        assert_eq!(e.entry1 >> 31, 1);
        assert_eq!((e.entry1 >> 9) & 1, 0, "LEVEL main");
        assert!(gpfifo_entry(1 << 40, 1, false).is_err());
        assert!(gpfifo_entry(0x1002, 1, false).is_err());
        assert!(gpfifo_entry(0x1000, 0, false).is_err());
    }
}
