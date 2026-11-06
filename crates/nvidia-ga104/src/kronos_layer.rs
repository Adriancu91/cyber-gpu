//! C-83 I13 — one whole layer of Kronos on the work channel, for one new position (a decoding step), as a list of
//! launches of our NAK programs: `layernorm`, `q8_matvec` (I12), `vadd`, `copy_b128` (I11, the new key/value into
//! the caches), `attention`, `gelu`. The driver writes each launch's cbuf and QMD ([`Launch::qmd`]) and submits
//! them in one GPFIFO entry with a timestamp record after each ([`crate::transfer::encode_sequence`]): the records
//! say which launch did not finish and how long each took.
//!
//! The order of the 16 launches is the CPU engine's (`gpu_compute::layer::step_cpu`), the layer's output is
//! written back into `x` (in place, element by element). The verdict ([`judge`]) compares x with the CPU engine in
//! the tolerance `gpu_compute::layer::{TOL_ABS, TOL_REL}` (fixed before any run): LayerNorm, softmax and GELU use
//! the GPU's approximate units, so bit for bit is not expected here (it still is for each `q8_matvec` alone, I12).
//!
//! Tested **only in SIMULARE**; nothing here touches a board.

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use gpu_compute::layer::{self, LayerWeights, MAX_POS, Shape};
use gpu_compute::q8;
use gpu_core::error::{GpuError, GpuResult};

use crate::qmd::{ComputeQmd, MAX_SHARED_BYTES, SharedMemory};
use crate::transfer::{self, pattern_word};

fn bad(why: &'static str) -> GpuError {
    GpuError::InvalidArgument(why)
}

/// The programs a layer uses (all compiled by Mesa NAK 25.0.7 for SM 8.6, 19 registers each — the dumps in
/// `tools/nak/out/`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Prog {
    Copy,
    Q8Matvec,
    Vadd,
    LayerNorm,
    Attention,
    Gelu,
    /// I14: the token + position embedding (`embed.comp`).
    Embed,
}

pub const PROGRAMS: [Prog; 7] = [
    Prog::Copy,
    Prog::Q8Matvec,
    Prog::Vadd,
    Prog::LayerNorm,
    Prog::Attention,
    Prog::Gelu,
    Prog::Embed,
];

pub const VADD: &[u8] = include_bytes!("../../../tools/nak/out/vadd.sm86.sass");
pub const LAYERNORM: &[u8] = include_bytes!("../../../tools/nak/out/layernorm.sm86.sass");
pub const ATTENTION: &[u8] = include_bytes!("../../../tools/nak/out/attention.sm86.sass");
pub const GELU: &[u8] = include_bytes!("../../../tools/nak/out/gelu.sm86.sass");
pub const EMBED: &[u8] = include_bytes!("../../../tools/nak/out/embed.sm86.sass");

impl Prog {
    pub fn code(self) -> &'static [u8] {
        match self {
            Prog::Copy => transfer::COPY_B128,
            Prog::Q8Matvec => crate::q8_matvec::Q8_MATVEC,
            Prog::Vadd => VADD,
            Prog::LayerNorm => LAYERNORM,
            Prog::Attention => ATTENTION,
            Prog::Gelu => GELU,
            Prog::Embed => EMBED,
        }
    }

    /// Threads per block (`local_size_x`).
    pub fn block(self) -> u32 {
        match self {
            Prog::Copy | Prog::Vadd | Prog::Gelu | Prog::Embed => 256,
            Prog::Q8Matvec | Prog::LayerNorm | Prog::Attention => 32,
        }
    }

    /// Shared memory (one warp each: NAK left no `BAR.SYNC`).
    pub fn shared(self) -> Option<SharedMemory> {
        let bytes = match self {
            Prog::Q8Matvec | Prog::LayerNorm => 128,
            Prog::Attention => 4 * (MAX_POS as u32 + 32),
            _ => return None,
        };
        Some(SharedMemory {
            bytes,
            max_bytes: MAX_SHARED_BYTES,
            barriers: 0,
        })
    }
}

/// One launch: program, its 256-byte cbuf, blocks in x.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Launch {
    pub prog: Prog,
    pub cbuf: [u8; 256],
    pub grid: u32,
}

impl Launch {
    /// The QMD bytes for this launch with the program at `prog_va` and the cbuf at `cbuf_va`.
    pub fn qmd(&self, prog_va: u64, cbuf_va: u64) -> GpuResult<[u8; nvidia_rm570::QMD_SIZE]> {
        let q = ComputeQmd {
            program_va: prog_va,
            registers: 19,
            cbuf0_va: cbuf_va,
            cbuf0_size: 0x100,
            grid: [self.grid, 1, 1],
            block: [self.prog.block(), 1, 1],
            release: None,
        };
        match self.prog.shared() {
            Some(s) => q.to_bytes_with_shared(&s),
            None => q.to_bytes(),
        }
    }
}

pub(crate) fn cbuf(vas: &[u64], words: &[u32]) -> [u8; 256] {
    let mut c = [0u8; 256];
    for (i, v) in vas.iter().enumerate() {
        c[0x28 + 8 * i..0x30 + 8 * i].copy_from_slice(&v.to_le_bytes());
    }
    let w0 = 0x28 + 8 * vas.len();
    for (i, w) in words.iter().enumerate() {
        c[w0 + 4 * i..w0 + 4 * i + 4].copy_from_slice(&w.to_le_bytes());
    }
    c
}

pub(crate) fn blocks(n: usize, per: u32) -> u32 {
    n.div_ceil(per as usize) as u32
}

/// GPU VAs of everything one layer touches. Activations live in one scratch buffer laid out by
/// [`LayerVas::with_scratch`]; caches are `[MAX_POS][E]` each.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerVas {
    pub x: u64,
    pub h: u64,
    pub qkv: u64,
    pub qkvb: u64,
    pub att: u64,
    pub o: u64,
    pub ob: u64,
    pub x1: u64,
    pub h2: u64,
    pub f: u64,
    pub fb: u64,
    pub g: u64,
    pub p: u64,
    pub pb: u64,
    pub kc: u64,
    pub vc: u64,
    pub ln1_g: u64,
    pub ln1_b: u64,
    pub wqkv: u64,
    pub bqkv: u64,
    pub wout: u64,
    pub bout: u64,
    pub ln2_g: u64,
    pub ln2_b: u64,
    pub wfc: u64,
    pub bfc: u64,
    pub wproj: u64,
    pub bproj: u64,
}

/// Bytes of the activation scratch for n_emb `e` (14 vectors, see [`LayerVas::with_scratch`]).
pub fn scratch_bytes(e: usize) -> u64 {
    4 * e as u64 * (1 + 1 + 3 + 3 + 1 + 1 + 1 + 1 + 1 + 4 + 4 + 4 + 1 + 1)
}

/// Bytes of one cache (keys or values): `MAX_POS × E` f32.
pub fn cache_bytes(e: usize) -> u64 {
    4 * (MAX_POS * e) as u64
}

impl LayerVas {
    /// Lays the activations out in the scratch at `scratch` (16-aligned; every vector is a multiple of 128 bytes
    /// since E is a multiple of 32) and takes the caches and weights as given.
    pub fn with_scratch(
        e: usize,
        scratch: u64,
        kc: u64,
        vc: u64,
        w: WeightVas,
    ) -> GpuResult<LayerVas> {
        if !scratch.is_multiple_of(16)
            || !kc.is_multiple_of(16)
            || !vc.is_multiple_of(16)
            || !e.is_multiple_of(32)
        {
            return Err(bad("strat: adrese nealiniate la 16 sau E nemultiplu de 32"));
        }
        let mut at = scratch;
        let mut take = |n: usize| {
            let a = at;
            at += 4 * (n * e) as u64;
            a
        };
        Ok(LayerVas {
            x: take(1),
            h: take(1),
            qkv: take(3),
            qkvb: take(3),
            att: take(1),
            o: take(1),
            ob: take(1),
            x1: take(1),
            h2: take(1),
            f: take(4),
            fb: take(4),
            g: take(4),
            p: take(1),
            pb: take(1),
            kc,
            vc,
            ln1_g: w.ln1_g,
            ln1_b: w.ln1_b,
            wqkv: w.wqkv,
            bqkv: w.bqkv,
            wout: w.wout,
            bout: w.bout,
            ln2_g: w.ln2_g,
            ln2_b: w.ln2_b,
            wfc: w.wfc,
            bfc: w.bfc,
            wproj: w.wproj,
            bproj: w.bproj,
        })
    }
}

/// GPU VAs of a layer's weights (the same order as `gpu_compute::layer::LayerWeights`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WeightVas {
    pub ln1_g: u64,
    pub ln1_b: u64,
    pub wqkv: u64,
    pub bqkv: u64,
    pub wout: u64,
    pub bout: u64,
    pub ln2_g: u64,
    pub ln2_b: u64,
    pub wfc: u64,
    pub bfc: u64,
    pub wproj: u64,
    pub bproj: u64,
}

/// The 16 launches of one position `pos` through the layer (CPU engine order); the output lands in `v.x`.
pub fn token_launches(s: &Shape, v: &LayerVas, pos: usize) -> GpuResult<Vec<Launch>> {
    let (e, d) = (s.e, s.d());
    if e == 0
        || s.heads == 0
        || !e.is_multiple_of(s.heads)
        || !e.is_multiple_of(q8::BLOCK)
        || d > 128
    {
        return Err(bad("strat: forma invalida pentru programe"));
    }
    if pos >= MAX_POS {
        return Err(bad("strat: pozitie peste cache (512)"));
    }
    let inv_n = (1.0 / e as f32).to_bits();
    let ln = |x: u64, g: u64, b: u64, y: u64| Launch {
        prog: Prog::LayerNorm,
        cbuf: cbuf(&[x, g, b, y], &[e as u32, inv_n, s.eps.to_bits()]),
        grid: 1,
    };
    let mv = |w: u64, x: u64, y: u64, rows: usize, cols: usize| -> GpuResult<Launch> {
        let mut c = [0u8; 256];
        c.copy_from_slice(&crate::q8_matvec::q8_cbuf(w, x, y, rows, cols)?);
        Ok(Launch {
            prog: Prog::Q8Matvec,
            cbuf: c,
            grid: rows as u32,
        })
    };
    let add = |a: u64, b: u64, c: u64, n: usize| Launch {
        prog: Prog::Vadd,
        cbuf: cbuf(&[a, b, c], &[n as u32]),
        grid: blocks(n, 256),
    };
    let copy = |src: u64, dst: u64, bytes: u64| -> GpuResult<Launch> {
        Ok(Launch {
            prog: Prog::Copy,
            cbuf: transfer::copy_cbuf(src, dst, bytes)?,
            grid: (bytes / transfer::COPY_UNIT).div_ceil(transfer::COPY_BLOCK as u64) as u32,
        })
    };
    let row = 4 * e as u64;
    let scale = 1.0 / gpu_compute::math::sqrt(d as f32);
    Ok(alloc::vec![
        ln(v.x, v.ln1_g, v.ln1_b, v.h),
        mv(v.wqkv, v.h, v.qkv, 3 * e, e)?,
        add(v.qkv, v.bqkv, v.qkvb, 3 * e),
        copy(v.qkvb + row, v.kc + pos as u64 * row, row)?,
        copy(v.qkvb + 2 * row, v.vc + pos as u64 * row, row)?,
        Launch {
            prog: Prog::Attention,
            cbuf: cbuf(
                &[v.qkvb, v.kc, v.vc, v.att],
                &[(pos + 1) as u32, e as u32, d as u32, scale.to_bits()]
            ),
            grid: s.heads as u32,
        },
        mv(v.wout, v.att, v.o, e, e)?,
        add(v.o, v.bout, v.ob, e),
        add(v.x, v.ob, v.x1, e),
        ln(v.x1, v.ln2_g, v.ln2_b, v.h2),
        mv(v.wfc, v.h2, v.f, 4 * e, e)?,
        add(v.f, v.bfc, v.fb, 4 * e),
        Launch {
            prog: Prog::Gelu,
            cbuf: cbuf(&[v.fb, v.g], &[(4 * e) as u32]),
            grid: blocks(4 * e, 256),
        },
        mv(v.wproj, v.g, v.p, e, 4 * e)?,
        add(v.p, v.bproj, v.pb, e),
        add(v.x1, v.pb, v.x, e),
    ])
}

/// Launches per position.
pub const LAUNCHES_PER_TOKEN: usize = 16;

/// A layer's weights owned (the window's synthetic test layer, or a layer read from a KGR1 file by the OS).
#[derive(Clone, Debug, PartialEq)]
pub struct OwnedLayer {
    pub ln1_g: Vec<f32>,
    pub ln1_b: Vec<f32>,
    pub wqkv: Vec<u8>,
    pub bqkv: Vec<f32>,
    pub wout: Vec<u8>,
    pub bout: Vec<f32>,
    pub ln2_g: Vec<f32>,
    pub ln2_b: Vec<f32>,
    pub wfc: Vec<u8>,
    pub bfc: Vec<f32>,
    pub wproj: Vec<u8>,
    pub bproj: Vec<f32>,
}

impl OwnedLayer {
    pub fn weights(&self) -> LayerWeights<'_> {
        LayerWeights {
            ln1_g: &self.ln1_g,
            ln1_b: &self.ln1_b,
            wqkv: &self.wqkv,
            bqkv: &self.bqkv,
            wout: &self.wout,
            bout: &self.bout,
            ln2_g: &self.ln2_g,
            ln2_b: &self.ln2_b,
            wfc: &self.wfc,
            bfc: &self.bfc,
            wproj: &self.wproj,
            bproj: &self.bproj,
        }
    }
}

/// The window's synthetic layer (no file): matrices quantized from the I11 pattern (±0.05), LayerNorm gains
/// 1 ± 0.1, offsets and biases ± 0.02. `seed` changes everything.
pub fn test_layer(e: usize, seed: u32) -> GpuResult<OwnedLayer> {
    let mut k = 0u32;
    let mut vec = |n: usize, mid: f32, amp: f32| -> Vec<f32> {
        k += 1;
        (0..n as u64)
            .map(|i| {
                mid + amp
                    * ((pattern_word(seed.wrapping_mul(31).wrapping_add(k), i) >> 8) as f32
                        / (1u32 << 23) as f32
                        - 1.0)
            })
            .collect()
    };
    let mat = |rows: usize, cols: usize, s: u32| {
        crate::q8_matvec::test_matrix(rows, cols, seed.wrapping_mul(17).wrapping_add(s))
            .map(|m| m.0)
    };
    Ok(OwnedLayer {
        wqkv: mat(3 * e, e, 1)?,
        wout: mat(e, e, 2)?,
        wfc: mat(4 * e, e, 3)?,
        wproj: mat(e, 4 * e, 4)?,
        ln1_g: vec(e, 1.0, 0.1),
        ln1_b: vec(e, 0.0, 0.02),
        bqkv: vec(3 * e, 0.0, 0.02),
        bout: vec(e, 0.0, 0.02),
        ln2_g: vec(e, 1.0, 0.1),
        ln2_b: vec(e, 0.0, 0.02),
        bfc: vec(4 * e, 0.0, 0.02),
        bproj: vec(e, 0.0, 0.02),
    })
}

/// The input of position `pos` for the window test (±1).
pub fn test_input(e: usize, seed: u32, pos: usize) -> Vec<f32> {
    (0..e as u64)
        .map(|i| {
            (pattern_word(seed ^ 0x5eed, (pos * e) as u64 + i) >> 8) as f32 / (1u32 << 23) as f32
                - 1.0
        })
        .collect()
}

/// What the board's layer output says.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerVerdict {
    /// Every element within the tolerance; the largest |GPU − CPU|.
    Trece {
        max_abs: f32,
    },
    NuTrece {
        index: usize,
        got: f32,
        cpu: f32,
    },
}

impl LayerVerdict {
    pub fn passes(&self) -> bool {
        matches!(self, LayerVerdict::Trece { .. })
    }

    pub fn line(&self, pos: usize) -> String {
        match self {
            LayerVerdict::Trece { max_abs } => {
                format!(
                    "i13: pozitia {pos}: TRECE - abaterea maxima fata de procesor {max_abs:e} (toleranta 1e-4 + 1e-4*|x|)"
                )
            }
            LayerVerdict::NuTrece { index, got, cpu } => {
                format!(
                    "i13: pozitia {pos}: NU TRECE - elementul {index}: placa {got:e}, procesor {cpu:e}"
                )
            }
        }
    }
}

/// Compares the board's x after the layer with the CPU engine's.
pub fn judge(got: &[f32], cpu: &[f32]) -> LayerVerdict {
    match layer::first_outside(got, cpu) {
        Some((index, got, cpu)) => LayerVerdict::NuTrece { index, got, cpu },
        None => LayerVerdict::Trece {
            max_abs: got
                .iter()
                .zip(cpu)
                .fold(0.0f32, |m, (g, c)| m.max((g - c).abs())),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn programs_and_launch_list() {
        for p in PROGRAMS {
            assert!(p.code().len().is_multiple_of(16) && !p.code().is_empty());
        }
        let s = Shape {
            e: 768,
            heads: 12,
            eps: 1e-5,
        };
        let base = crate::mmu::VA_BASE;
        let w = WeightVas {
            ln1_g: base,
            ln1_b: base + 0x1000,
            wqkv: base + 0x10_0000,
            bqkv: base + 0x2000,
            wout: base + 0x40_0000,
            bout: base + 0x3000,
            ln2_g: base + 0x4000,
            ln2_b: base + 0x5000,
            wfc: base + 0x80_0000,
            bfc: base + 0x6000,
            wproj: base + 0x100_0000,
            bproj: base + 0x9000,
        };
        let v = LayerVas::with_scratch(
            768,
            base + 0x200_0000,
            base + 0x300_0000,
            base + 0x400_0000,
            w,
        )
        .unwrap();
        assert_eq!(v.pb + 4 * 768, base + 0x200_0000 + scratch_bytes(768));
        let l = token_launches(&s, &v, 5).unwrap();
        assert_eq!(l.len(), LAUNCHES_PER_TOKEN);
        assert_eq!(l[5].prog, Prog::Attention);
        assert_eq!(l[5].grid, 12);
        assert_eq!(
            u32::from_le_bytes(l[5].cbuf[0x48..0x4c].try_into().unwrap()),
            6,
            "n_pos = pos + 1"
        );
        assert_eq!(
            f32::from_le_bytes(l[5].cbuf[0x54..0x58].try_into().unwrap()),
            0.125
        );
        // The new key goes to row 5 of the key cache.
        assert_eq!(
            u64::from_le_bytes(l[3].cbuf[0x30..0x38].try_into().unwrap()),
            v.kc + 5 * 3072
        );
        assert_eq!(l[1].grid, 2304);
        assert_eq!(l[12].grid, 12, "gelu: 3072 / 256");
        for x in &l {
            x.qmd(base, base + 0x100).unwrap();
        }
        assert!(token_launches(&s, &v, MAX_POS).is_err());
        assert!(
            token_launches(
                &Shape {
                    e: 768,
                    heads: 5,
                    eps: 1e-5
                },
                &v,
                0
            )
            .is_err()
        );
    }

    #[test]
    fn verdict_lines() {
        assert!(judge(&[1.0, 2.0], &[1.0, 2.00005]).passes());
        let v = judge(&[1.0, 2.1], &[1.0, 2.0]);
        assert!(!v.passes() && v.line(3).contains("elementul 1"));
        assert!(!judge(&[1.0], &[1.0, 2.0]).passes());
        let t = test_layer(64, 1).unwrap();
        Shape {
            e: 64,
            heads: 2,
            eps: 1e-5,
        }
        .check(&t.weights())
        .unwrap();
        assert_eq!(test_layer(64, 1).unwrap(), t);
        assert!(test_input(64, 1, 0).iter().all(|v| v.abs() <= 1.0));
    }
}
