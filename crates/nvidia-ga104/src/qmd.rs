//! I10 (part) — building a QMD V03_00 (the 256-byte "launch description" the
//! compute class reads after `SEND_PCAS_A`). Pure data; nothing is sent.
//!
//! Field positions come from the generated crate `nvidia-rm570`
//! (open-gpu-doc `clc6c0qmd.h`, MIT). Which fields a minimal launch fills was
//! read in Mesa NAK `qmd.rs` and tinygrad `ops_nv.py` (both MIT; study lessons
//! 15–16) — re-written here. The program's register count and memory sizes
//! come from the NAK "fișă" (`tools/nak/README.md`).
//!
//! STATUS: SIMULARE only (checked by `gpu-sim::cmdproc`). Shared memory and
//! the barrier count: [`ComputeQmd::encode_with_shared`] (session 8). Not set
//! yet, though Mesa NAK sets them in every QMD V03_00 (`Qmd3_0::new`):
//! `API_VISIBLE_CALL_LIMIT = NO_CHECK`, `SAMPLER_INDEX = INDEPENDENTLY`,
//! `SM_GLOBAL_CACHING_ENABLE = 1` — to align with the golden log before the
//! first real launch (docs/COMPATIBILITATE_FIRMWARE.md, „De verificat”).

use gpu_core::error::{GpuError, GpuResult};
use nvidia_rm570::regs::qmd_v03_00 as q;
use nvidia_rm570::{QMD_SIZE, QmdField};

/// Shared memory + barriers a program needs (from the NAK dump / the GLSL
/// declaration). Added in session 8 for matmul step 3; [`ComputeQmd::encode`]
/// alone leaves these fields 0 (enough for programs without shared memory).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedMemory {
    /// Bytes of `shared` variables per workgroup (rounded up to 256 here).
    pub bytes: u32,
    /// Largest per-workgroup size the driver allows (Mesa NVK: 48 KiB).
    pub max_bytes: u32,
    /// Hardware barriers used (`bar.sync` → 1).
    pub barriers: u8,
}

/// NVK's limit (`NVK_MAX_SHARED_SIZE`, Mesa 25.0.7).
pub const MAX_SHARED_BYTES: u32 = 48 * 1024;

/// The SM's L1/shared split for a size, in QMD units (Mesa NAK `qmd.rs`
/// `gv100_sm_config_smem_size`, MIT, re-written): 8/16/32/64/96 KiB → n/4 KiB + 1.
pub fn sm_config_shared_units(bytes: u32) -> u32 {
    let kib = match bytes {
        b if b > 64 * 1024 => 96,
        b if b > 32 * 1024 => 64,
        b if b > 16 * 1024 => 32,
        b if b > 8 * 1024 => 16,
        _ => 8,
    };
    kib / 4 + 1
}

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

    /// [`Self::encode`] plus shared memory and barriers, filled the way Mesa
    /// NAK fills QMD V03_00 (`set_smem_size` bounded, `set_barrier_count`).
    pub fn encode_with_shared(&self, s: &SharedMemory) -> GpuResult<[u32; QMD_SIZE / 4]> {
        let bytes = s.bytes.next_multiple_of(0x100);
        if s.max_bytes > MAX_SHARED_BYTES || bytes > s.max_bytes {
            return Err(GpuError::InvalidArgument(
                "QMD: memoria partajata depaseste limita",
            ));
        }
        if s.barriers > 16 {
            return Err(GpuError::InvalidArgument("QMD: prea multe bariere"));
        }
        let mut m = self.encode()?;
        put(&mut m, q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE, bytes)?;
        let units = sm_config_shared_units(bytes);
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE,
            units,
        )?;
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE,
            sm_config_shared_units(s.max_bytes),
        )?;
        put(
            &mut m,
            q::NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE,
            units,
        )?;
        put(&mut m, q::NVC6C0_QMDV03_00_BARRIER_COUNT, s.barriers as u32)?;
        Ok(m)
    }

    /// [`Self::encode_with_shared`] as little-endian bytes.
    pub fn to_bytes_with_shared(&self, s: &SharedMemory) -> GpuResult<[u8; QMD_SIZE]> {
        Ok(words_to_bytes(&self.encode_with_shared(s)?))
    }

    /// The QMD as little-endian bytes, ready to be copied to GPU memory.
    pub fn to_bytes(&self) -> GpuResult<[u8; QMD_SIZE]> {
        Ok(words_to_bytes(&self.encode()?))
    }
}

fn words_to_bytes(w: &[u32; QMD_SIZE / 4]) -> [u8; QMD_SIZE] {
    let mut b = [0u8; QMD_SIZE];
    for (i, x) in w.iter().enumerate() {
        b[4 * i..4 * i + 4].copy_from_slice(&x.to_le_bytes());
    }
    b
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

    #[test]
    fn shared_memory_like_mesa_nak() {
        // Units: 8/16/32/64/96 KiB -> 3/5/9/17/25.
        for (b, u) in [
            (0, 3),
            (2048, 3),
            (8192, 3),
            (8193, 5),
            (16 << 10, 5),
            (48 << 10, 17),
            (65 << 10, 25),
        ] {
            assert_eq!(sm_config_shared_units(b), u, "{b}");
        }
        let s = SharedMemory {
            bytes: 2000,
            max_bytes: MAX_SHARED_BYTES,
            barriers: 1,
        };
        let m = add64().encode_with_shared(&s).unwrap();
        assert_eq!(q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE.get(&m), 0x800); // rounded to 256
        assert_eq!(q::NVC6C0_QMDV03_00_MIN_SM_CONFIG_SHARED_MEM_SIZE.get(&m), 3);
        assert_eq!(
            q::NVC6C0_QMDV03_00_TARGET_SM_CONFIG_SHARED_MEM_SIZE.get(&m),
            3
        );
        assert_eq!(
            q::NVC6C0_QMDV03_00_MAX_SM_CONFIG_SHARED_MEM_SIZE.get(&m),
            17
        );
        assert_eq!(q::NVC6C0_QMDV03_00_BARRIER_COUNT.get(&m), 1);
        // Everything encode() wrote is unchanged.
        let plain = add64().encode().unwrap();
        for f in [
            q::NVC6C0_QMDV03_00_PROGRAM_ADDRESS_LOWER,
            q::NVC6C0_QMDV03_00_CTA_RASTER_WIDTH,
        ] {
            assert_eq!(f.get(&m), f.get(&plain));
        }
        assert_eq!(q::NVC6C0_QMDV03_00_SHARED_MEMORY_SIZE.get(&plain), 0);
        for bad in [
            SharedMemory {
                bytes: 49 << 10,
                ..s
            },
            SharedMemory {
                max_bytes: 64 << 10,
                ..s
            },
            SharedMemory { barriers: 17, ..s },
        ] {
            assert!(add64().encode_with_shared(&bad).is_err(), "{bad:?}");
        }
    }
}
