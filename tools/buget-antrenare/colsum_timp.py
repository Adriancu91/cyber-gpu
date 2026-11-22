#!/usr/bin/env python3
"""C-157 p.2 after os-349: what the measured `colsum` times say about a real training step (DEDUCED from MEASURED).

Usage: python3 tools/buget-antrenare/colsum_timp.py

Inputs:
  * MEASURED on the RTX 3070 Ti (os-349, cycle 2026-11-22 12:36, `kt colsum set: timp (ns)`): one launch per set,
    the buffer in host RAM (the OS maps it with alloc_sys_mapped), so every load crosses PCIe.
  * colsum.comp: one thread per column, rows added in order, no unrolling -> one memory round trip per row.
  * #513 step shapes (C-160_P5_PAS_DE_AUR.md, scaled to 12 layers): per layer 4 bias sums (2304, 768, 3072, 768 columns)
    and 2 LayerNorm sums (768 columns, 2 inputs per element), + the final LayerNorm; a chunk = 12 x 512 = 6144 rows.
  * PyTorch: 39 265 tokens/s on #513 (FISA_ANTRENARE.md, MEASURED by Kronos) -> time budget of one chunk.
Nothing here is a board measurement except the five times below.
"""

# (set, rows, columns, LayerNorm mode, measured ns)
MEASURED = [
    ("deplasare-64x64", 64, 64, False, 46_496),
    ("ln-64x64", 64, 64, True, 55_040),
    ("ln-3x257", 3, 257, True, 6_816),
    ("ln-8x768", 8, 768, True, 15_712),
    ("ln-8x768-media100", 8, 768, True, 15_936),
]

ROWS = 12 * 512
BIAS_COLS = [2304, 768, 3072, 768]
LN_COLS = 768
LAYERS = 12
TORCH_TPS = 39_265
BANDWIDTH = 450e9  # bytes/s, DEDUCED effective VRAM read rate (the VRAM copy measured ~281 GB/s each way, 562 GB/s traffic)


def main():
    # per-row cost from the two sets with the same mode and the most rows apart (ln: 8 -> 64 rows)
    ln8 = (MEASURED[3][4] + MEASURED[4][4]) / 2
    ln64 = MEASURED[1][4]
    per_row_ln = (ln64 - ln8) / (64 - 8)
    base_ln = ln8 - 8 * per_row_ln
    per_row_bias = (MEASURED[0][4] - base_ln) / 64
    print(f"MEASURED fit (host RAM over PCIe): LayerNorm mode {per_row_ln/1000:.2f} us/row + {base_ln/1000:.1f} us; "
          f"bias mode {per_row_bias/1000:.2f} us/row (same base)")

    launches = LAYERS * (len(BIAS_COLS) + 2) + 1
    chunk_budget = ROWS / TORCH_TPS
    print(f"#513 step: {launches} colsum launches per chunk of {ROWS} rows; PyTorch does the whole chunk in "
          f"{chunk_budget*1e3:.0f} ms")

    # the program as it is: latency bound, one round trip per row; columns run in parallel (<= 12 blocks of 256)
    for label, scale in (("host RAM (as measured)", 1.0), ("VRAM, DEDUCED 0.6x latency", 0.6)):
        t_bias = base_ln + ROWS * per_row_bias * scale
        t_ln = base_ln + ROWS * per_row_ln * scale
        total = LAYERS * (len(BIAS_COLS) * t_bias + 2 * t_ln) + t_ln
        print(f"  colsum as is, {label:28s}: {t_bias/1e6:.2f} ms per bias sum, {t_ln/1e6:.2f} ms per LN sum -> "
              f"{total/1e9*1e3:.0f} ms per chunk = {total/1e9/chunk_budget*100:.0f}% of PyTorch's whole chunk")

    # the floor: read every element once
    bytes_bias = sum(BIAS_COLS) * ROWS * 4
    bytes_ln = 2 * LN_COLS * ROWS * 4 * 2  # dy and x, two LayerNorms
    total_bytes = LAYERS * (bytes_bias + bytes_ln) + LN_COLS * ROWS * 4 * 2
    floor = total_bytes / BANDWIDTH
    print(f"  floor (read each element once at {BANDWIDTH/1e9:.0f} GB/s): {total_bytes/1e9:.2f} GB -> "
          f"{floor*1e3:.1f} ms per chunk = {floor/chunk_budget*100:.1f}% of PyTorch's chunk")


if __name__ == "__main__":
    main()
