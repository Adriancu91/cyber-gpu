//! CPU backend: implements the I1 contract with host RAM as "device memory"
//! and synchronous execution of the built-in ops on the CPU.
//!
//! This is the backend Kronos uses first (plan v0.2, K6-CPU) and the test
//! double for everything above I1. Its identity says "cpu-referinta"; it
//! never reports itself as a GPU.

use alloc::vec;
use alloc::vec::Vec;

use gpu_command::FenceTimeline;
use gpu_core::backend::{
    BufferDesc, BufferId, ComputeDevice, DeviceStatus, Fence, Identity, KernelRef, RecoveryLevel,
    Submission,
};
use gpu_core::caps::{Capabilities, DTypeSet, Feature};
use gpu_core::error::{GpuError, GpuResult};
use gpu_core::state::DeviceState;
use gpu_memory::buffers::BufferTable;
use gpu_memory::range::RangeAllocator;

use crate::kernels;
use crate::op::OpCode;
use crate::reference as r;

pub struct CpuBackend {
    arena: Vec<u8>,
    table: BufferTable,
    fences: FenceTimeline,
    state: DeviceState,
    last_error: u32,
}

impl CpuBackend {
    /// `arena_bytes` of host RAM act as device memory.
    pub fn new(arena_bytes: usize) -> GpuResult<Self> {
        let space = RangeAllocator::new(0, arena_bytes as u64)?;
        Ok(CpuBackend {
            arena: vec![0; arena_bytes],
            table: BufferTable::new(space),
            fences: FenceTimeline::new(),
            state: DeviceState::Discovered,
            last_error: 0,
        })
    }

    fn range(&self, id: BufferId, off: u64, len: u64) -> GpuResult<core::ops::Range<usize>> {
        let start = self.table.resolve(id, off, len)? as usize;
        Ok(start..start + len as usize)
    }

    fn read_f32(&self, id: BufferId, n: usize) -> GpuResult<Vec<f32>> {
        let rg = self.range(id, 0, (n * 4) as u64)?;
        let (words, _) = self.arena[rg].as_chunks::<4>();
        Ok(words.iter().map(|c| f32::from_le_bytes(*c)).collect())
    }

    fn read_u32(&self, id: BufferId, n: usize) -> GpuResult<Vec<u32>> {
        let rg = self.range(id, 0, (n * 4) as u64)?;
        let (words, _) = self.arena[rg].as_chunks::<4>();
        Ok(words.iter().map(|c| u32::from_le_bytes(*c)).collect())
    }

    fn read_u16(&self, id: BufferId, n: usize) -> GpuResult<Vec<u16>> {
        let rg = self.range(id, 0, (n * 2) as u64)?;
        let (halves, _) = self.arena[rg].as_chunks::<2>();
        Ok(halves.iter().map(|c| u16::from_le_bytes(*c)).collect())
    }

    fn write_f32(&mut self, id: BufferId, v: &[f32]) -> GpuResult<()> {
        let rg = self.range(id, 0, (v.len() * 4) as u64)?;
        let (words, _) = self.arena[rg].as_chunks_mut::<4>();
        for (dst, x) in words.iter_mut().zip(v) {
            dst.copy_from_slice(&x.to_le_bytes());
        }
        Ok(())
    }

    fn run(&mut self, op: OpCode, b: &[BufferId], s: &[u32]) -> GpuResult<()> {
        let u = |i: usize| s[i] as usize;
        match op {
            OpCode::Add => {
                let n = u(0);
                let (x, y) = (self.read_f32(b[0], n)?, self.read_f32(b[1], n)?);
                let mut o = vec![0.0; n];
                r::add(&x, &y, &mut o)?;
                self.write_f32(b[2], &o)
            }
            OpCode::BiasAdd => {
                let (rows, cols) = (u(0), u(1));
                let mut x = self.read_f32(b[0], rows * cols)?;
                let bias = self.read_f32(b[1], cols)?;
                r::bias_add(&mut x, &bias, rows, cols)?;
                self.write_f32(b[0], &x)
            }
            OpCode::Matmul => {
                let (m, k, n) = (u(0), u(1), u(2));
                let (a, bb) = (self.read_f32(b[0], m * k)?, self.read_f32(b[1], k * n)?);
                let mut o = vec![0.0; m * n];
                // Blocked kernel: bit-identical to r::matmul (kernels.rs tests).
                kernels::matmul_blocked(&a, &bb, &mut o, m, k, n)?;
                self.write_f32(b[2], &o)
            }
            OpCode::MatmulBf16W => {
                let (m, k, n) = (u(0), u(1), u(2));
                let (a, w) = (self.read_f32(b[0], m * k)?, self.read_u16(b[1], k * n)?);
                let mut o = vec![0.0; m * n];
                kernels::matmul_bf16w(&a, &w, &mut o, m, k, n)?;
                self.write_f32(b[2], &o)
            }
            OpCode::LayerNorm => {
                let (rows, cols, eps) = (u(0), u(1), f32::from_bits(s[2]));
                let x = self.read_f32(b[0], rows * cols)?;
                let g = self.read_f32(b[1], cols)?;
                let be = self.read_f32(b[2], cols)?;
                let mut o = vec![0.0; rows * cols];
                r::layernorm(&x, &g, &be, &mut o, rows, cols, eps)?;
                self.write_f32(b[3], &o)
            }
            OpCode::GeluTanh => {
                let mut x = self.read_f32(b[0], u(0))?;
                r::gelu_tanh(&mut x);
                self.write_f32(b[0], &x)
            }
            OpCode::Softmax => {
                let (rows, cols) = (u(0), u(1));
                let mut x = self.read_f32(b[0], rows * cols)?;
                r::softmax_rows(&mut x, rows, cols, s[2] != 0)?;
                self.write_f32(b[0], &x)
            }
            OpCode::Embedding => {
                let (n, vocab, dim) = (u(0), u(1), u(2));
                let ids = self.read_u32(b[0], n)?;
                let table = self.read_f32(b[1], vocab * dim)?;
                let mut o = vec![0.0; n * dim];
                r::embedding(&ids, &table, vocab, dim, &mut o)?;
                self.write_f32(b[2], &o)
            }
            OpCode::Attention => {
                let (t, h, d) = (u(0), u(1), u(2));
                let n = t * h * d;
                let (q, k, v) = (
                    self.read_f32(b[0], n)?,
                    self.read_f32(b[1], n)?,
                    self.read_f32(b[2], n)?,
                );
                let mut o = vec![0.0; n];
                r::causal_attention(&q, &k, &v, &mut o, t, h, d)?;
                self.write_f32(b[3], &o)
            }
        }
    }
}

impl ComputeDevice for CpuBackend {
    fn identify(&mut self) -> GpuResult<Identity> {
        if self.state == DeviceState::Discovered {
            self.state.go(DeviceState::Identified)?;
        }
        Ok(Identity {
            address: None,
            pci: None,
            chip: "cpu-referinta",
            raw_id: None,
        })
    }

    fn capabilities(&self) -> Capabilities {
        let mut c = Capabilities::new("cpu-referinta")
            .with(Feature::Identify)
            .with(Feature::Allocate)
            .with(Feature::Copy)
            .with(Feature::Compute);
        c.memory_bytes = self.table.space().total();
        c.max_allocation = self.table.space().total();
        c.dtypes = DTypeSet::F32;
        // bf16 is accepted only as matmul weights (MATMUL_BF16W), not as a
        // general tensor type, so the capability set stays F32.
        c
    }

    fn initialize(&mut self) -> GpuResult<()> {
        self.identify()?;
        if self.state == DeviceState::Identified {
            self.state.go(DeviceState::Mapped)?;
        }
        if self.state == DeviceState::Mapped {
            self.state.go(DeviceState::Initialized)?;
            self.state.go(DeviceState::Ready)?;
        }
        self.state.require(DeviceState::Ready)
    }

    fn allocate(&mut self, desc: BufferDesc) -> GpuResult<BufferId> {
        self.state.require(DeviceState::Ready)?;
        let (id, a) = self.table.allocate(desc)?;
        self.arena[a.start as usize..(a.start + a.size) as usize].fill(0);
        Ok(id)
    }

    fn release(&mut self, buf: BufferId) -> GpuResult<()> {
        self.table.release(buf).map(|_| ())
    }

    fn copy_host_to_device(&mut self, src: &[u8], dst: BufferId, dst_offset: u64) -> GpuResult<()> {
        self.state.require(DeviceState::Ready)?;
        let rg = self.range(dst, dst_offset, src.len() as u64)?;
        self.arena[rg].copy_from_slice(src);
        Ok(())
    }

    fn copy_device_to_host(
        &mut self,
        src: BufferId,
        src_offset: u64,
        dst: &mut [u8],
    ) -> GpuResult<()> {
        self.state.require(DeviceState::Ready)?;
        let rg = self.range(src, src_offset, dst.len() as u64)?;
        dst.copy_from_slice(&self.arena[rg]);
        Ok(())
    }

    fn submit(&mut self, work: &Submission<'_>) -> GpuResult<Fence> {
        self.state.require(DeviceState::Ready)?;
        let code = match work.kernel {
            KernelRef::Builtin(c) => c,
            KernelRef::Binary { .. } => {
                return Err(GpuError::NotImplemented("cpu: binare de dispozitiv"));
            }
        };
        let op = OpCode::from_u32(code)
            .ok_or(GpuError::InvalidArgument("cod de operatie necunoscut"))?;
        let (nb, ns) = op.arity();
        if work.buffers.len() != nb || work.scalars.len() != ns {
            return Err(GpuError::InvalidArgument("numar gresit de argumente"));
        }
        if let Err(e) = self.run(op, work.buffers, work.scalars) {
            self.last_error = code;
            return Err(e);
        }
        let f = self.fences.next_fence();
        self.fences.signal(f.0)?;
        Ok(f)
    }

    fn synchronize(&mut self, fence: Fence, _timeout_us: u64) -> GpuResult<()> {
        if fence > self.fences.submitted() {
            return Err(GpuError::InvalidArgument("fence netrimis"));
        }
        Ok(()) // execution is synchronous
    }

    fn query_status(&mut self) -> GpuResult<DeviceStatus> {
        Ok(DeviceStatus {
            state: self.state,
            submitted: self.fences.submitted(),
            completed: self.fences.completed(),
            last_error: self.last_error,
        })
    }

    fn reset_or_recover(&mut self, _level: RecoveryLevel) -> GpuResult<()> {
        self.fences.abandon();
        self.last_error = 0;
        if self.state == DeviceState::Faulted {
            self.state.go(DeviceState::Identified)?;
            self.initialize()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpu_core::backend::BufferDesc;

    fn f32s(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn submit(dev: &mut CpuBackend, op: OpCode, b: &[BufferId], s: &[u32]) -> GpuResult<Fence> {
        dev.submit(&Submission {
            kernel: KernelRef::Builtin(op as u32),
            buffers: b,
            scalars: s,
            grid: [1; 3],
            block: [1; 3],
        })
    }

    #[test]
    fn lifecycle_enforced() {
        let mut d = CpuBackend::new(1 << 16).unwrap();
        assert!(matches!(
            d.allocate(BufferDesc::device(16)),
            Err(GpuError::InvalidState { .. })
        ));
        d.initialize().unwrap();
        assert_eq!(d.identify().unwrap().chip, "cpu-referinta");
        assert!(d.capabilities().has(Feature::Compute));
    }

    #[test]
    fn vector_add_end_to_end() {
        let mut d = CpuBackend::new(1 << 16).unwrap();
        d.initialize().unwrap();
        let n = 64;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| 0.5 * i as f32).collect();
        let ba = d.allocate(BufferDesc::device(256)).unwrap();
        let bb = d.allocate(BufferDesc::device(256)).unwrap();
        let bo = d.allocate(BufferDesc::device(256)).unwrap();
        d.copy_host_to_device(&f32s(&a), ba, 0).unwrap();
        d.copy_host_to_device(&f32s(&b), bb, 0).unwrap();
        let f = submit(&mut d, OpCode::Add, &[ba, bb, bo], &[n as u32]).unwrap();
        d.synchronize(f, 1000).unwrap();
        let mut out = vec![0u8; 256];
        d.copy_device_to_host(bo, 0, &mut out).unwrap();
        for i in 0..n {
            let v = f32::from_le_bytes(out[i * 4..i * 4 + 4].try_into().unwrap());
            assert_eq!(v, 1.5 * i as f32);
        }
        let st = d.query_status().unwrap();
        assert_eq!((st.submitted, st.completed), (Fence(1), Fence(1)));
    }

    #[test]
    fn bounds_and_handles() {
        let mut d = CpuBackend::new(1 << 12).unwrap();
        d.initialize().unwrap();
        let b = d.allocate(BufferDesc::device(64)).unwrap();
        assert!(matches!(
            d.copy_host_to_device(&[0; 65], b, 0),
            Err(GpuError::OutOfBounds { .. })
        ));
        assert!(matches!(
            d.copy_host_to_device(&[0; 8], b, 60),
            Err(GpuError::OutOfBounds { .. })
        ));
        // matmul reading past the buffer is refused, not truncated
        assert!(submit(&mut d, OpCode::Matmul, &[b, b, b], &[8, 8, 8]).is_err());
        assert!(submit(&mut d, OpCode::Add, &[b, b], &[1]).is_err(), "arity");
        assert!(matches!(
            d.submit(&Submission {
                kernel: KernelRef::Builtin(99),
                buffers: &[],
                scalars: &[],
                grid: [1; 3],
                block: [1; 3]
            }),
            Err(GpuError::InvalidArgument(_))
        ));
        d.release(b).unwrap();
        assert_eq!(d.release(b), Err(GpuError::InvalidHandle));
        assert!(matches!(
            d.allocate(BufferDesc::device(1 << 13)),
            Err(GpuError::OutOfMemory { .. })
        ));
        assert_eq!(
            d.query_status().unwrap().submitted,
            Fence(0),
            "failed submits consume no fence"
        );
    }

    #[test]
    fn matmul_through_backend_matches_reference() {
        let mut d = CpuBackend::new(1 << 16).unwrap();
        d.initialize().unwrap();
        let (m, k, n) = (4, 5, 3);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.11).cos()).collect();
        let ba = d.allocate(BufferDesc::device((m * k * 4) as u64)).unwrap();
        let bb = d.allocate(BufferDesc::device((k * n * 4) as u64)).unwrap();
        let bo = d.allocate(BufferDesc::device((m * n * 4) as u64)).unwrap();
        d.copy_host_to_device(&f32s(&a), ba, 0).unwrap();
        d.copy_host_to_device(&f32s(&b), bb, 0).unwrap();
        submit(
            &mut d,
            OpCode::Matmul,
            &[ba, bb, bo],
            &[m as u32, k as u32, n as u32],
        )
        .unwrap();
        let mut out = vec![0u8; m * n * 4];
        d.copy_device_to_host(bo, 0, &mut out).unwrap();
        let mut want = vec![0.0; m * n];
        r::matmul(&a, &b, &mut want, m, k, n).unwrap();
        assert_eq!(out, f32s(&want));
    }

    #[test]
    fn bf16_weight_matmul_through_backend() {
        let mut d = CpuBackend::new(1 << 16).unwrap();
        d.initialize().unwrap();
        let (m, k, n) = (3, 70, 5);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.21).sin()).collect();
        let wf: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.07).cos()).collect();
        let w: Vec<u16> = wf.iter().map(|&x| crate::tensor::f32_to_bf16(x)).collect();
        let wb: Vec<u8> = w.iter().flat_map(|h| h.to_le_bytes()).collect();
        let ba = d.allocate(BufferDesc::device((m * k * 4) as u64)).unwrap();
        let bw = d.allocate(BufferDesc::device((k * n * 2) as u64)).unwrap();
        let bo = d.allocate(BufferDesc::device((m * n * 4) as u64)).unwrap();
        d.copy_host_to_device(&f32s(&a), ba, 0).unwrap();
        d.copy_host_to_device(&wb, bw, 0).unwrap();
        submit(
            &mut d,
            OpCode::MatmulBf16W,
            &[ba, bw, bo],
            &[m as u32, k as u32, n as u32],
        )
        .unwrap();
        let mut out = vec![0u8; m * n * 4];
        d.copy_device_to_host(bo, 0, &mut out).unwrap();
        let wd: Vec<f32> = w.iter().map(|&h| crate::tensor::bf16_to_f32(h)).collect();
        let mut want = vec![0.0; m * n];
        r::matmul(&a, &wd, &mut want, m, k, n).unwrap();
        assert_eq!(out, f32s(&want));
        // A bf16 buffer that is too short is refused, not read past.
        assert!(
            submit(
                &mut d,
                OpCode::MatmulBf16W,
                &[ba, bw, bo],
                &[m as u32, k as u32, 2 * n as u32]
            )
            .is_err()
        );
    }
}
