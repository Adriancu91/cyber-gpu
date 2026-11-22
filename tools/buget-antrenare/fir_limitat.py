#!/usr/bin/env python3
"""C-160 p.7: which training programs are limited by one thread doing too much (DEDUCED, nothing here is measured).

Usage: python3 tools/buget-antrenare/fir_limitat.py

The same look as colsum_timp.py (after os-349), for every program of the training step except the matmuls
(mm_train / mm_tc have their own estimate in viteza.py) and colsum (colsum_timp.py, colsum_part).

Shapes: #513 (12 layers, C 768, 12 heads, d 64, V 16 384, 98 032 128 parameters), one chunk = 12 texts x 512 = 6 144 rows,
the unit PyTorch trains at 39 265 tokens/s (FISA_ANTRENARE.md, MEASURED by Kronos) -> 156 ms per chunk.

For each program, from its .comp source and the NAK listing (registers, no unrolling: one load per loop iteration):
  * threads launched and the blocks' size -> waves on the board (GA104: 48 SMs, at most 48 warps and 16 blocks per SM,
    64 K registers per SM; public CUDA tables for compute capability 8.6);
  * the longest chain one thread runs (loop iterations) and how long one iteration waits (latency model below);
  * the bytes the program must move (floor at BANDWIDTH).
Time of one launch = max(bandwidth floor, waves x longest chain x cycles per iteration). Both are rough (factor 2-3).

Latency model (DEDUCED, public microbenchmark figures for Ampere, not measured on our board):
  L1 hit ~35 cycles, L2 hit ~200, DRAM ~450. An iteration with one dependent load costs about the load's latency when a
  scheduler has fewer than ~4 other warps to switch to, otherwise ~ issue-bound. We take per-iteration cycles as given in
  each row below (written from the access pattern of the source), not fitted to anything.
"""

SMS = 48
CLOCK_HZ = 1.77e9
MAX_WARPS_SM = 48
MAX_BLOCKS_SM = 16
REGS_SM = 65536
BANDWIDTH = 450e9  # bytes/s, the same DEDUCED effective VRAM rate as colsum_timp.py

L, C, H, D, V = 12, 768, 12, 64, 16_384
B, T = 12, 512
ROWS = B * T
N_PARAMS = 98_032_128
TORCH_TPS = 39_265
CHUNK_S = ROWS / TORCH_TPS


def resident_threads(block, regs):
    regs_alloc = -(-regs // 8) * 8
    warps_block = -(-block // 32)
    by_blocks = MAX_BLOCKS_SM
    by_warps = MAX_WARPS_SM // warps_block
    by_regs = REGS_SM // (regs_alloc * warps_block * 32)
    blocks = min(by_blocks, by_warps, by_regs)
    return blocks * block * SMS, blocks * warps_block


INSTR_PER_ITER = 6  # NAK listing: no unrolling -> load(s), fma, index add, compare, branch


def launch_time(threads, block, regs, chain, cyc_per_iter, bytes_moved):
    """chain: list of (iterations, load latency in cycles) segments of the longest thread."""
    resident, warps_sm = resident_threads(block, regs)
    waves = max(1, -(-threads // resident))
    warps_smsp = max(1, min(warps_sm, -(-threads // 32) // SMS) // 4)
    # one iteration of one warp: the load's latency, or the issue slots of the other warps sharing its scheduler
    cycles = sum(n * max(lat, warps_smsp * INSTR_PER_ITER) for n, lat in chain) * cyc_per_iter
    t_chain = waves * cycles / CLOCK_HZ
    t_bw = bytes_moved / BANDWIDTH
    return max(t_chain, t_bw), t_chain, t_bw, waves, warps_sm


def program_rows():
    tri = T * (T + 1) // 2  # (i, j) pairs with j <= i, per (text, head)
    rows = []
    # name, launches per chunk, threads, block, registers, longest chain (iterations), cycles per iteration, bytes, note
    rows.append((
        "ln_back", 2 * L + 1, ROWS * 256, 256, 19,
        [(2 * -(-C // 256), 35)], 1,
        ROWS * C * 4 * 5 + C * 4,  # x, dy read twice, dx written once
        "un bloc pe rand, 3 elemente pe fir: limitat de memorie",
    ))
    rows.append((
        "opt_step (AdamW)", 1, N_PARAMS, 256, 19,
        [(1, 450)], 1,
        N_PARAMS * 4 * 7,  # g, w, m, v read; w, m, v written
        "un fir pe parametru: limitat de memorie (o data pe pas)",
    ))
    gn_blocks = -(-N_PARAMS // 4096)
    rows.append((
        "gnorm (2 lansari)", 1, gn_blocks * 256, 256, 19,
        [(4096 // 256, 35)], 1,
        N_PARAMS * 4,
        "sume partiale pe blocuri de 4 096: limitat de memorie",
    ))
    rows.append((
        "xent", 1, ROWS * 256, 256, 19,
        [(3 * V // 256, 35)], 1,
        ROWS * V * 4 * 3,
        "un bloc pe rand, 64 de elemente pe fir",
    ))
    rows.append((
        "gelu_tr (inapoi)", L, ROWS * 4 * C, 256, 19,
        [(1, 450)], 1,
        ROWS * 4 * C * 4 * 3,
        "un fir pe element",
    ))
    rows.append((
        "emb_back (simboluri)", 1, ROWS * C, 64, 19,
        [(512, 450)], 1,  # a frequent token: hundreds of rows in one chain, each a DRAM row
        ROWS * C * 4 * 2,
        "un fir pe (segment, coloana); lantul = aparitiile simbolului",
    ))
    # attention, per launch; the causal triangle: thread i runs (i+1) iterations of each j loop
    # attn_fwd: scores (i+1)*d iterations (Q row per lane: L1; K broadcast) ~ 40 cycles,
    #           3 passes over the own P row (strided per lane, 1 sector of 8 per lane) ~ 90 cycles,
    #           att: d*(i+1) iterations re-reading the own P row (7 of 8 from L1, 1 from L2/DRAM) ~ 90 cycles
    fwd_chain = [(T * D, 40), (3 * T, 90), (D * T, 90)]
    rows.append((
        "attn_fwd", L, B * H * T, 32, 25,
        fwd_chain, 1,
        B * H * tri * 4 * 2 * 4 + ROWS * 3 * C * 4,  # P written / re-read (>= 4 touches, sector-inefficient)
        "UN FIR PE RAND (b, h, i): lant pana la 512 x 64 x 2 iteratii",
    ))
    bwd0_chain = [(T * D, 40), (3 * T, 90)]
    rows.append((
        "attn_back mod 0", L, B * H * T, 32, 25,
        bwd0_chain, 1,
        B * H * tri * 4 * 4 * 4 + ROWS * 3 * C * 4,
        "UN FIR PE RAND (b, h, i): lant pana la 512 x 64 iteratii",
    ))
    # mode 1: one warp per (b, h, t), each lane 2 columns e; per e: T iterations reading a dS and a P column
    # (stride T: one sector per element, L2 at best) -> ~200 cycles each
    rows.append((
        "attn_back mod 1", L, B * H * T * 32, 32, 25,
        [((D // 32) * T, 200)], 1,
        B * H * tri * 4 * 2 * 8 + ROWS * 3 * C * 4 * 2,  # columns of dS and P: 8x sector amplification
        "un warp pe pozitie; coloanele lui dS si P citite pe sarite",
    ))
    return rows


def main():
    print(f"#513, o bucata = {ROWS} randuri; PyTorch face toata bucata in {CHUNK_S*1e3:.0f} ms (MASURAT de Kronos)")
    print("Totul DEDUS (factor 2-3 de nesiguranta); nimic masurat pe placa.\n")
    total = 0.0
    attn = 0.0
    print(f"{'program':22s} {'lans.':>5s} {'fire':>10s} {'bloc':>4s} {'warps/SM':>8s} {'valuri':>6s} "
          f"{'lant ms':>8s} {'memorie ms':>10s} {'pe bucata ms':>12s}  cine limiteaza")
    for name, launches, threads, block, regs, chain, cyc, nbytes, note in program_rows():
        t, t_chain, t_bw, waves, warps_sm = launch_time(threads, block, regs, chain, cyc, nbytes)
        per_chunk = t * launches
        total += per_chunk
        if name.startswith("attn"):
            attn += per_chunk
        who = "FIRUL" if t_chain > t_bw else "memoria"
        print(f"{name:22s} {launches:5d} {threads:10d} {block:4d} {warps_sm:8d} {waves:6d} "
              f"{t_chain*1e3:8.2f} {t_bw*1e3:10.2f} {per_chunk*1e3:12.1f}  {who}: {note}")
    print(f"\nToate de mai sus: ~{total*1e3:.0f} ms pe bucata = {total/CHUNK_S*100:.0f}% din toata bucata PyTorch; "
          f"din care atentia ~{attn*1e3:.0f} ms.")

    # The cheap first change: blocks of 128 instead of 32 for the attention (same per-thread code, same bits).
    print("\nPasul ieftin (acelasi cod pe fir, aceiasi biti): blocuri de 128 in loc de 32 la atentie")
    for name, launches, threads, block, regs, chain, cyc, nbytes, note in program_rows():
        if not name.startswith("attn"):
            continue
        t32 = launch_time(threads, 32, regs, chain, cyc, nbytes)
        t128 = launch_time(threads, 128, regs, chain, cyc, nbytes)
        print(f"  {name:18s}: valuri {t32[3]} -> {t128[3]}, warps/SM {t32[4]} -> {t128[4]}; "
              f"pe bucata ~{t32[0]*launches*1e3:.0f} ms -> ~{t128[0]*launches*1e3:.0f} ms")

    flop = L * 6 * B * H * (T * (T + 1) // 2) * D * 2  # 2 matmuls forward + 4 backward, causal half
    print(f"\nTinta atentiei ca inmultiri pe placi (2 inainte + 4 inapoi, doar triunghiul): {flop/1e9:.0f} GFLOP pe bucata"
          f" -> ~{flop/10e12*1e3:.0f} ms la 10 TFLOP/s (F32 bun), ~{flop/15e12*1e3:.0f} ms la 15 (mm_tc)")


def tiled_forward():
    """C-160 p.7, session 163: the attention forward "pe placi" (attn_fwd_tile.comp, plan written BEFORE the code).

    Blocks of 256 threads (16 x 16) own 64 rows (b, h, i0..i0+63); every thread 4 x 4 results. Q tile once, then
    for every 64-column tile up to the diagonal: K tile in shared, 4 x 4 scores per thread (16 ffma per 8 shared
    loads). P passes through memory: s written, read back for exp (x written), read back once more for P = x/sum
    (P written, the tile also kept in shared for att = P.V with V tiles in shared). Above-diagonal tiles: zeros.
    Two numbers per launch: the arithmetic (FLOP actually computed, diagonal tiles in full) at an effective F32 rate,
    and the bytes (P traffic + Q, K, V re-read per row tile), at BANDWIDTH. The slower one wins (no overlap assumed
    beyond that). DEDUCED; factor 2.
    """
    tiles_rows = -(-T // 64)
    tiles_lower = tiles_rows * (tiles_rows + 1) // 2  # (row tile, column tile) pairs on or below the diagonal
    bh = B * H
    flop = bh * tiles_lower * (64 * 64 * D * 2) * 2  # scores + P.V, full 64 x 64 on every tile kept
    lower_bytes = bh * tiles_lower * 64 * 64 * 4
    full_p = bh * T * T * 4
    # s write, s read, x write, x read + final P write of the whole row (zeros above too)
    p_bytes = 4 * lower_bytes + full_p
    qkv_bytes = bh * tiles_lower * 64 * D * 4 * 2 + ROWS * 3 * C * 4  # K and V tiles per row tile (L2 mostly)
    out = []
    for rate in (4e12, 8e12):
        t_ar = flop / rate
        t_bw = (p_bytes + qkv_bytes) / BANDWIDTH
        out.append((rate, t_ar, t_bw, max(t_ar, t_bw) * L))
    return flop, p_bytes, out


def print_tiled():
    flop, p_bytes, out = tiled_forward()
    print("\nAtentia inainte \"pe placi\" (attn_fwd_tile, plan sesiunea 163, DEDUS):")
    print(f"  pe lansare: {flop/1e9:.1f} GFLOP calculate (diagonala intreaga), P mutat {p_bytes/1e6:.0f} MB")
    for rate, t_ar, t_bw, chunk in out:
        print(f"  la {rate/1e12:.0f} TFLOP/s F32 efectiv: calcul {t_ar*1e3:.2f} ms, memorie {t_bw*1e3:.2f} ms"
              f" -> ~{chunk*1e3:.0f} ms pe bucata (12 lansari)")


if __name__ == "__main__":
    main()
    print_tiled()
