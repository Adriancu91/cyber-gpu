#!/usr/bin/env python3
"""Training memory budget for Kronos models on an 8 GB board (C-160 point 2).

Plain-Python, no dependencies. Counts bytes the backward pass needs kept from the
forward pass (activations), plus parameters, gradients and the two AdamW moments.
All counts are DEDUCED from the architecture in Cyber-hub integrare/kronos/PAS_DE_AUR.md;
nothing here is measured on a GPU.
"""

MODELS = {
    # name: (layers, n_emb, heads, vocab, context)
    "mic": (2, 64, 4, 256, 32),
    "#513": (12, 768, 12, 16384, 512),
    "#651": (24, 768, 12, 16384, 512),
}
BUDGET = 7.0 * 2**30  # Adrian's rule for the workstation (CLAUDE.md, 12 Oct 2026): keep VRAM under ~7 GB


def params(L, C, H, V, Tmax):
    per_layer = 12 * C * C + 13 * C  # ln1, in_proj(+b), out_proj(+b), ln2, mlp0(+b), mlp2(+b)
    return L * per_layer + V * C + Tmax * C + 2 * C  # tied head: emb_token counted once


def act_bytes(L, C, H, V, B, T, act=4, probs=True, logits=4):
    """Bytes kept for backward. `act` = bytes per kept matmul input/output (4 = F32, 2 = 16-bit)."""
    tok = B * T
    # per layer, per token: ln1 in (C, F32), ln1 out (C), qkv (3C), att out (C),
    # residual-2 = ln2 in (C, F32), ln2 out (C), fc pre-GELU (4C), GELU out (4C); LN mean+rstd x2 (F32)
    f32_part = 2 * C * 4 + 4 * 4          # residual inputs of both LayerNorms + mean/rstd, always F32
    mm_part = (1 + 3 + 1 + 1 + 4 + 4) * C * act
    att = H * T * 4 if probs else H * 2 * 4  # softmax probs (F32) or only row max+sum (recompute)
    layer = tok * (f32_part + mm_part + att)
    final = tok * (C * 4 + 2 * 4 + C * act)  # final LN in, mean/rstd, out
    return L * layer + final + tok * V * logits  # logits; their gradient is written in place


def gib(x):
    return x / 2**30


def table():
    out = []
    for name, (L, C, H, V, Tm) in MODELS.items():
        P = params(L, C, H, V, Tm)
        state = 16 * P  # weights + grads + m + v, all F32
        out.append(f"{name}: P = {P:,} parameters; weights+grads+m+v (F32) = {gib(state):.3f} GiB")
        for T in sorted({min(64, Tm), Tm // 2, Tm} if Tm > 32 else {Tm}):
            for mode, kw in (("F32", {}), ("F32 fara P", {"probs": False}),
                             ("16b", {"act": 2}), ("16b fara P", {"act": 2, "probs": False})):
                a1 = act_bytes(L, C, H, V, 1, T, **kw)
                free = BUDGET - state - 256 * 2**20  # 256 MiB margin for programs, temporaries, rings
                bmax = int(free // a1) if a1 else 0
                out.append(f"  T={T:4d} {mode:10s}: activari pe secventa {a1/2**20:9.2f} MiB; lot maxim in 7 GiB: {bmax}")
    return "\n".join(out)


if __name__ == "__main__":
    print(table())
