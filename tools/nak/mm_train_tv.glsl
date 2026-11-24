// C-165 (docs/C-165_CURSA_MM.md): the body shared by the race variants of mm_train_tile. Included by the small
// mm_tv_*.comp files, which only set the tile geometry (BM, BN, BK, TM, TN) before including it.
// Same job and same push constants as mm_train.comp / mm_train_tile.comp: C[M,N] = op(A)[M,K] * op(B)[K,N]
// (+ bias[N]), f32, flags TA (A stored [K,M]), TB (B stored [N,K]), BIAS. Layout idea: siboehm SGEMM_CUDA kernel 5
// "2D blocktiling" (MIT, idea only) and our mm_train_tile.comp.
// One workgroup of THREADS = (BM/TM) * (BN/TN) threads computes a BM x BN block of C; thread (tx = tid % TX,
// ty = tid / TX), TX = BN/TN, TY = BM/TM, owns rows ty + TY*i and columns tx + TX*j (i < TM, j < TN).
// Per K-tile of BK: the threads copy the BK x BM tile of op(A) (as_[p][r]) and the BK x BN tile of op(B) (bs_[p][c])
// into shared memory; element number idx = tid + THREADS*e of a tile goes to the thread that makes neighbouring
// threads read neighbouring addresses (the values in shared memory are the same either way).
// Barrier, then for p in the tile: TM values of A and TN of B to registers, TM*TN fused multiply-adds. The inner
// loop stops at K exactly (no zero padding is accumulated), so every element of C is still the chain of fused
// multiply-adds p = 0..K-1 from +0.0, then one add of bias[col]: the oracle stays
// gpu-compute::reference::matmul_t_fma_ftz, the very bits of mm_train and mm_train_tile. Threads whose cells are
// outside C load zeros and wait at the barriers; they only skip the stores.
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_EXT_control_flow_attributes : require

#define TX (BN / TN)
#define TY (BM / TM)
#define THREADS (TX * TY)
#define LOADS_A (BK * BM / THREADS)
#define LOADS_B (BK * BN / THREADS)
#define F_TA 1u
#define F_TB 2u
#define F_BIAS 4u

layout(local_size_x = THREADS, local_size_y = 1, local_size_z = 1) in;

layout(buffer_reference, std430, buffer_reference_align = 4) readonly buffer In { float v[]; };
layout(buffer_reference, std430, buffer_reference_align = 4) writeonly buffer Out { float v[]; };

layout(push_constant) uniform Args {
    In a;
    In b;
    Out c;
    In bias;
    uint m;
    uint n;
    uint k;
    uint flags;
} args;

shared float as_[BK * BM];   // as_[p * BM + r] = op(A)[row0 + r, t + p]
shared float bs_[BK * BN];   // bs_[p * BN + c] = op(B)[t + p, col0 + c]

void main() {
    uint tid = gl_LocalInvocationID.x;
    uint tx = tid % TX;
    uint ty = tid / TX;
    uint row0 = gl_WorkGroupID.y * BM;
    uint col0 = gl_WorkGroupID.x * BN;
    bool ta = (args.flags & F_TA) != 0u;
    bool tb = (args.flags & F_TB) != 0u;
    float acc[TM * TN];
    [[unroll]] for (uint i = 0; i < TM * TN; i++) acc[i] = 0.0;
    float reg_m[TM];
    float reg_n[TN];
    for (uint t = 0; t < args.k; t += BK) {
        [[unroll]] for (uint e = 0; e < LOADS_A; e++) {
            // op(A) tile: element (r, p). A stored [M,K]: p fastest across threads; [K,M] (TA): r fastest.
            uint idx = tid + THREADS * e;
            uint r = ta ? idx % BM : idx / BK;
            uint p = ta ? idx / BM : idx % BK;
            uint ar = row0 + r, ak = t + p;
            float va = 0.0;
            if (ar < args.m && ak < args.k) {
                va = ta ? args.a.v[ak * args.m + ar] : args.a.v[ar * args.k + ak];
            }
            as_[p * BM + r] = va;
        }
        [[unroll]] for (uint e = 0; e < LOADS_B; e++) {
            // op(B) tile: element (p, c). B stored [K,N]: c fastest; [N,K] (TB): p fastest.
            uint idx = tid + THREADS * e;
            uint q = tb ? idx % BK : idx / BN;
            uint cc = tb ? idx / BK : idx % BN;
            uint bk = t + q, bc = col0 + cc;
            float vb = 0.0;
            if (bk < args.k && bc < args.n) {
                vb = tb ? args.b.v[bc * args.k + bk] : args.b.v[bk * args.n + bc];
            }
            bs_[q * BN + cc] = vb;
        }
        barrier();
        uint len = min(BK, args.k - t);
        for (uint p = 0; p < len; p++) {
            [[unroll]] for (uint i = 0; i < TM; i++) reg_m[i] = as_[p * BM + ty + TY * i];
            [[unroll]] for (uint j = 0; j < TN; j++) reg_n[j] = bs_[p * BN + tx + TX * j];
            [[unroll]] for (uint i = 0; i < TM; i++) {
                [[unroll]] for (uint j = 0; j < TN; j++) {
                    acc[i * TN + j] += reg_m[i] * reg_n[j];
                }
            }
        }
        barrier();
    }
    [[unroll]] for (uint i = 0; i < TM; i++) {
        uint row = row0 + ty + TY * i;
        [[unroll]] for (uint j = 0; j < TN; j++) {
            uint col = col0 + tx + TX * j;
            if (row < args.m && col < args.n) {
                float v = acc[i * TN + j];
                if ((args.flags & F_BIAS) != 0u) {
                    v += args.bias.v[col];
                }
                args.c.v[row * args.n + col] = v;
            }
        }
    }
}
