#!/usr/bin/env python3
"""C-160 p.7 — written-before-measurement estimate of training speed (DEDUCED, nothing measured here).

Usage: python3 tools/buget-antrenare/viteza.py

Inputs:
  * model shapes: integrare/kronos/MODEL.md, MODEL_651.md (same numbers as buget.py)
  * PyTorch speed on the same RTX 3070 Ti: integrare/kronos/FISA_ANTRENARE.md section 1 (MEASURED by Kronos)
  * GA104 ceilings: public specs (48 SMs, ~1.77 GHz boost, 128 FP32 FMA/clk/SM,
    256 f16 tensor FMA/clk/SM with f32 accumulate on GA10x) -> DEDUCED, the board clock is not measured
"""

SMS = 48
CLOCK_HZ = 1.77e9
FP32_PEAK = SMS * 128 * 2 * CLOCK_HZ        # FLOP/s
TC_F16_F32ACC_PEAK = SMS * 256 * 2 * CLOCK_HZ

MODELS = {
    # name: (params, layers, width, context, recompute_in_pytorch, pytorch_tokens_per_s)
    "#513": (98_032_128, 12, 768, 512, False, 39_265),
    "#651": (183_086_592, 24, 768, 512, True, 15_509),
}


def flop_per_token(params, layers, width, ctx, recompute):
    """6P for the matmuls (2 forward + 4 backward), +2P when activations are recomputed,
    plus causal attention ~12*L*C*T (forward + backward, Q.K^T and P.V)."""
    mm = (8 if recompute else 6) * params
    attn = 12 * layers * width * ctx
    return mm + attn


def main():
    print(f"ceilings (DEDUCED from specs): FP32 {FP32_PEAK/1e12:.1f} TFLOP/s, "
          f"f16 tensor -> f32 {TC_F16_F32ACC_PEAK/1e12:.1f} TFLOP/s")
    # engines: (label, ceiling, fraction of ceiling reached by the matmuls)
    engines = [
        ("mm_train today (16x16 tiles, F32)", FP32_PEAK, 0.12),
        ("F32 with register tiles, low", FP32_PEAK, 0.50),
        ("F32 with register tiles, high", FP32_PEAK, 0.65),
        ("f16 tensor, low", TC_F16_F32ACC_PEAK, 0.40),
        ("f16 tensor, mid", TC_F16_F32ACC_PEAK, 0.55),
        ("f16 tensor, high", TC_F16_F32ACC_PEAK, 0.70),
    ]
    # non-matmul work (LayerNorm, GELU, softmax, loss, AdamW, driver) as a share of the step
    overhead = 0.15
    for name, (p, layers, c, t, rec, torch_tps) in MODELS.items():
        f = flop_per_token(p, layers, c, t, rec)
        eff = torch_tps * f
        print(f"\n{name}: {f/1e9:.3f} GFLOP per token (recompute={rec}); "
              f"PyTorch {torch_tps} tok/s = {eff/1e12:.1f} TFLOP/s effective "
              f"({eff/TC_F16_F32ACC_PEAK*100:.0f}% of the tensor ceiling, "
              f"{eff/FP32_PEAK*100:.0f}% of the FP32 ceiling)")
        for label, peak, frac in engines:
            mm_time = f / (peak * frac)
            tps = 1.0 / (mm_time / (1.0 - overhead))
            print(f"  {label:38s} {peak*frac/1e12:5.1f} TFLOP/s on matmuls -> "
                  f"{tps:8.0f} tok/s = x{tps/torch_tps:4.2f} PyTorch")
        need = eff / (1.0 - overhead) / TC_F16_F32ACC_PEAK
        print(f"  to match PyTorch with {overhead*100:.0f}% overhead the tensor matmuls need "
              f"{need*100:.0f}% of the tensor ceiling")


if __name__ == "__main__":
    main()
