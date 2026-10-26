//! I10 (part) — building a QMD V03_00 (the 256-byte "launch description" the
//! compute class reads after `SEND_PCAS_A`). Pure data; nothing is sent.
//!
//! Field positions come from the generated crate `nvidia-rm570`
//! (open-gpu-doc `clc6c0qmd.h`, MIT). Which fields a minimal launch fills was
//! read in Mesa NAK `qmd.rs` and tinygrad `ops_nv.py` (both MIT; study lessons
//! 15–16) — re-written here. The program's register count and memory sizes
//! come from the NAK "fișă" (`tools/nak/README.md`).
//!
//! STATUS: SIMULARE only (checked by `gpu-sim::cmdproc`). Fields a real board
//! will also need (shared/local memory windows, barrier count, caching
//! controls) are not set yet; they come with I9/I10 and the golden log.

use gpu_core::error::{GpuError, GpuResult};
use nvidia_rm570::regs::qmd_v03_00 as q;
use nvidia_rm570::{QMD_SIZE, QmdField};

/// Inputs for one launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComputeQmd {
    /// GPU virtual address of the program code (256-byte aligned).
    pub program_va: u64,
    /// Registers per thread (NAK fișă).
    pub registers: u32,
    /// Constant buffer 0: GPU address (256-byte aligned) and size in bytes.
    pub cbuf0_va: u64,
    pub cbuf0_size: u32,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Semaphore written by the GPU when the work finishes (`release0`).
    pub release: Option<(u64, u32)>,
}

fn put(qmd: &mut [u32; QMD_SIZE / 4], f: QmdField, v: u32) -> GpuResult<()> {
    f.set(qmd, v).map_err(GpuError::InvalidArgument)
}

impl ComputeQmd {
    pub fn encode(&self) -> GpuResult<[u32; QMD_SIZE / 4]> {
        // Ampere GPU virtual addresses are 49 bits wide.
        let va_ok = |va: u64, align: u64| va >> 49 == 0 && va.is_multiple_of(align);
        if !va_ok(self.program_va, 256) || !va_ok(self.cbuf0_va, 256) {
            return Err(GpuError::InvalidArgument(
                "QMD: adresa nealiniata sau prea mare",
            ));
        }
        if self.cbuf0_size == 0 || !self.cbuf0_size.is_multiple_of(16) || self.cbuf0_size > 0x1_0000
        {
            return Err(GpuError::InvalidArgument(
                "QMD: bufferul constant 0 are marime invalida",
            ));
        }
        if self.registers == 0 || self.registers > 255 {
            return Err(GpuError::InvalidArgument("QMD: numar de registre invalid"));
        }
        let mut m = [0u32; QMD_SIZE / 4];
        put(&mut m, q::NVC6C0_QMDV03_00_QMD_MAJOR_VERSION, 3)?;
        put(&mut m, q::NVC6C0_QMDV03_00_QMD_VERSION, 0)?;
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER,
            self.program_va as u32,
        )?;
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_UPPER,
            (self.program_va >> 32) as u32,
        )?;
        put(&mut m, q::NVC6C0_QMDV03_00_REGISTER_COUNT_V, self.registers)?;
        put(&mut m, q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH, self.grid[0])?;
        put(&mut m, q::NVC6C0_QMDV03_00_CTA_RASTER_HEIGHT, self.grid[1])?;
        put(&mut m, q::NVC6C0_QMDV03_00_CTA_RASTER_DEPTH, self.grid[2])?;
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION0,
            self.block[0],
        )?;
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION1,
            self.block[1],
        )?;
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_CTA_THREAD_DIMENSION2,
            self.block[2],
        )?;
        put(&mut m, q::NVC6C0_QMDV03_00_CONSTANT_BUFFER_VALID(0), 1)?;
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_LOWER(0),
            self.cbuf0_va as u32,
        )?;
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_CONSTANT_BUFFER_ADDR_UPPER(0),
            (self.cbuf0_va >> 32) as u32,
        )?;
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_CONSTANT_BUFFER_SIZE_SHIFTED4(0),
            self.cbuf0_size >> 4,
        )?;
        if let Some((va, payload)) = self.release {
            if !va.is_multiple_of(4) || va >> 40 != 0 {
                return Err(GpuError::InvalidArgument(
                    "QMD: adresa semaforului invalida",
                ));
            }
            put(
                &mut m,
                q::NVC6C0_QMDV03_00_RELEASE0_ENABLE,
                q::NVC6C0_QMDV03_00_RELEASE0_ENABLE_TRUE,
            )?;
            put(
                &mut m,
                q::NVC6C0_QMDV03_00_RELEASE0_ADDRESS_LOWER,
                va as u32,
            )?;
            put(
                &mut m,
                q::NVC6C0_QMDV03_00_RELEASE0_ADDRESS_UPPER,
                (va >> 32) as u32,
            )?;
            put(&mut m, q::NVC6C0_QMDV03_00_RELEASE0_PAYLOAD_LOWER, payload)?;
        }
        Ok(m)
    }

    /// The QMD as little-endian bytes, ready to be copied to GPU memory.
    pub fn to_bytes(&self) -> GpuResult<[u8; QMD_SIZE]> {
        let w = self.encode()?;
        let mut b = [0u8; QMD_SIZE];
        for (i, x) in w.iter().enumerate() {
            b[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
        }
        Ok(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn add64() -> ComputeQmd {
        ComputeQmd {
            program_va: 0x10_0000,
            registers: 19,
            cbuf0_va: 0x20_0000,
            cbuf0_size: 0x100,
            grid: [1, 1, 1],
            block: [64, 1, 1],
            release: Some((0x30_0000, 1)),
        }
    }

    #[test]
    fn fields_land_where_the_header_says() {
        let m = add64().encode().unwrap();
        assert_eq!(m[48], 0x10_0000); // PROGRAM_ADDRESS_LOWER = bits 1567:1536
        assert_eq!(m[18] >> 16, 64); // CTA_THREAD_DIMENSION0 = 607:592
        assert_eq!(m[12], 1); // CTA_RASTER_WIDTH = 415:384
        assert_eq!(m[32], 0x20_0000); // CONSTANT_BUFFER_ADDR_LOWER(0) = 1055:1024
        assert_eq!(m[24], 0x30_0000); // RELEASE0_ADDRESS_LOWER = 799:768
        assert_eq!(m[26], 1); // RELEASE0_PAYLOAD_LOWER = 863:832
        assert_eq!(q::NVC6C0_QMDV03_00_QMD_MAJOR_VERSION.get(&m), 3);
    }

    #[test]
    fn refuses_bad_inputs() {
        let mut x = add64();
        x.program_va += 4;
        assert!(x.encode().is_err());
        let mut x = add64();
        x.block = [1 << 16, 1, 1]; // 16-bit field
        assert!(x.encode().is_err());
        let mut x = add64();
        x.cbuf0_size = 8;
        assert!(x.encode().is_err());
        let mut x = add64();
        x.release = Some((0x30_0002, 1));
        assert!(x.encode().is_err());
    }
}
