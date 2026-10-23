//! Built-in operation codes for `KernelRef::Builtin` and their argument
//! conventions. Every backend that claims an op must accept exactly this
//! layout; differential tests compare backends op by op.
//!
//! | op | buffers | scalars |
//! |---|---|---|
//! | ADD | a, b, out (f32) | n |
//! | BIAS_ADD | x (in/out), bias | rows, cols |
//! | MATMUL | a[m,k], b[k,n], out[m,n] | m, k, n |
//! | LAYERNORM | x, gamma, beta, out | rows, cols, eps (f32 bits) |
//! | GELU_TANH | x (in/out) | n |
//! | SOFTMAX | x (in/out) | rows, cols, causal (0/1) |
//! | EMBEDDING | ids (u32), table, out | n_ids, vocab, dim |
//! | ATTENTION | q, k, v, out [T, H*D] | T, H, D |
//! | MATMUL_BF16W | a[m,k] f32, w[k,n] bf16, out[m,n] f32 | m, k, n |

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum OpCode {
    Add = 1,
    BiasAdd = 2,
    Matmul = 3,
    LayerNorm = 4,
    GeluTanh = 5,
    Softmax = 6,
    Embedding = 7,
    Attention = 8,
    MatmulBf16W = 9,
}

impl OpCode {
    pub fn from_u32(v: u32) -> Option<OpCode> {
        use OpCode::*;
        Some(match v {
            1 => Add,
            2 => BiasAdd,
            3 => Matmul,
            4 => LayerNorm,
            5 => GeluTanh,
            6 => Softmax,
            7 => Embedding,
            8 => Attention,
            9 => MatmulBf16W,
            _ => return None,
        })
    }

    /// (buffers, scalars) expected.
    pub fn arity(self) -> (usize, usize) {
        use OpCode::*;
        match self {
            Add => (3, 1),
            BiasAdd => (2, 2),
            Matmul => (3, 3),
            LayerNorm => (4, 3),
            GeluTanh => (1, 1),
            Softmax => (1, 3),
            Embedding => (3, 3),
            Attention => (4, 3),
            MatmulBf16W => (3, 3),
        }
    }
}
