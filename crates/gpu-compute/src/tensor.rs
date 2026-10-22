//! Explicit tensor format (contract I2): dtype + shape + strides in elements.
//! Row-major, up to 4 dimensions (Kronos W2 needs at most [batch, head, seq, dim]).

use gpu_core::error::{GpuError, GpuResult};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    BF16,
}

impl DType {
    pub const fn size(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
        }
    }
}

pub const MAX_RANK: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorDesc {
    pub dtype: DType,
    pub rank: usize,
    pub shape: [usize; MAX_RANK],
    pub strides: [usize; MAX_RANK],
}

impl TensorDesc {
    /// Contiguous row-major tensor. Fails on rank > 4, zero dims or overflow.
    pub fn contiguous(dtype: DType, shape: &[usize]) -> GpuResult<Self> {
        if shape.is_empty() || shape.len() > MAX_RANK {
            return Err(GpuError::InvalidArgument("rang tensor invalid"));
        }
        if shape.contains(&0) {
            return Err(GpuError::InvalidArgument("dimensiune zero"));
        }
        let mut s = [0usize; MAX_RANK];
        let mut st = [0usize; MAX_RANK];
        let mut acc = 1usize;
        for i in (0..shape.len()).rev() {
            s[i] = shape[i];
            st[i] = acc;
            acc = acc
                .checked_mul(shape[i])
                .ok_or(GpuError::InvalidArgument("depasire numar elemente"))?;
        }
        acc.checked_mul(dtype.size())
            .ok_or(GpuError::InvalidArgument("depasire octeti"))?;
        Ok(TensorDesc {
            dtype,
            rank: shape.len(),
            shape: s,
            strides: st,
        })
    }

    pub fn dims(&self) -> &[usize] {
        &self.shape[..self.rank]
    }

    pub fn numel(&self) -> usize {
        self.dims().iter().product()
    }

    pub fn bytes(&self) -> usize {
        self.numel() * self.dtype.size()
    }

    pub fn is_contiguous(&self) -> bool {
        let mut acc = 1;
        for i in (0..self.rank).rev() {
            if self.strides[i] != acc {
                return false;
            }
            acc *= self.shape[i];
        }
        true
    }
}

/// f32 -> bf16 with round-to-nearest-even; NaN stays a quiet NaN.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) as u16) | 0x0040;
    }
    let round = 0x7fff + ((b >> 16) & 1);
    (b.wrapping_add(round) >> 16) as u16
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        let t = TensorDesc::contiguous(DType::F32, &[12, 512, 768]).unwrap();
        assert_eq!(t.numel(), 12 * 512 * 768);
        assert_eq!(t.strides[..3], [512 * 768, 768, 1]);
        assert!(t.is_contiguous());
        assert_eq!(
            TensorDesc::contiguous(DType::BF16, &[16384, 768])
                .unwrap()
                .bytes(),
            16384 * 768 * 2
        );
        assert!(TensorDesc::contiguous(DType::F32, &[]).is_err());
        assert!(TensorDesc::contiguous(DType::F32, &[1, 2, 3, 4, 5]).is_err());
        assert!(TensorDesc::contiguous(DType::F32, &[0]).is_err());
        assert!(TensorDesc::contiguous(DType::F32, &[usize::MAX, 2]).is_err());
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(bf16_to_f32(0x3f80), 1.0);
        // 1 + 2^-8 is exactly halfway between bf16 1.0 and 1.0078125: ties to even -> 1.0
        assert_eq!(f32_to_bf16(1.0 + 1.0 / 256.0), 0x3f80);
        // 1 + 3*2^-8 halfway between 0x3f81 and 0x3f82: ties to even -> 0x3f82
        assert_eq!(f32_to_bf16(1.0 + 3.0 / 256.0), 0x3f82);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        assert_eq!(f32_to_bf16(f32::INFINITY), 0x7f80);
        for &x in &[1.234_5f32, -2.5e-3, 65504.0, 1e30] {
            let back = bf16_to_f32(f32_to_bf16(x));
            assert!(((back - x) / x).abs() <= 1.0 / 256.0, "{x}");
        }
    }
}
