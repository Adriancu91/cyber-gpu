#!/usr/bin/env python3
"""After os-358 (C-157 p.2): the loops of a NAK listing, counted (DEDUCED from the listing, nothing measured).

Usage: python3 tools/buget-antrenare/nak_bucle.py tools/nak/out/attn_back_tile.sm86.txt [...]   (the loops of listings)
       python3 tools/buget-antrenare/nak_bucle.py                                             (the os-358 model, below)

For every natural loop (back edge tail -> head; the body = blocks reaching the tail without passing the head) prints,
summed over its OWN blocks (the body minus the loops nested in it, which have their own row) (= the path where every `if` inside is taken, as on tiles below the diagonal):
  instr  instructions issued by one warp per iteration (bssy / bsync / nop included: they take issue slots),
  stall  the sum of the NAK stall counts ("delay=") = cycles one warp needs per iteration when no variable-latency
         result is waited on longer than the stall already covers (a LOWER bound on one warp's iteration),
  ffma, lds (b32 / b128), ldg (read-only `constant` loads included), stg, sts, branch regions (bssy), barriers.
"""
import re
import sys

BLOCK = re.compile(r"^block(?:\.u)? (\d+) L\d+ \[[^\]]*\] -> \{")
SUCC = re.compile(r"^\} -> \[([^\]]*)\]")
DELAY = re.compile(r"delay=(\d+)")


def parse(path):
    blocks = {}
    cur = None
    for line in open(path, encoding="utf-8"):
        m = BLOCK.match(line)
        if m:
            cur = int(m.group(1))
            blocks[cur] = {"ins": [], "succ": []}
            continue
        m = SUCC.match(line)
        if m and cur is not None:
            blocks[cur]["succ"] = [int(x) for x in m.group(1).split(",") if x.strip()]
            cur = None
            continue
        if cur is not None and "//" in line:
            blocks[cur]["ins"].append(line.rstrip())
    return blocks


def stats(blocks, ids):
    s = {"instr": 0, "stall": 0, "ffma": 0, "lds32": 0, "lds128": 0, "ldg": 0, "stg": 0, "sts": 0, "bssy": 0, "bar": 0}
    for b in ids:
        for ins in blocks[b]["ins"]:
            s["instr"] += 1
            d = DELAY.search(ins)
            s["stall"] += int(d.group(1)) if d else 1
            if "ffma" in ins:
                s["ffma"] += 1
            if "ld.shared" in ins:
                s["lds128" if ".b128" in ins else "lds32"] += 1
            if "st.shared" in ins:
                s["sts"] += 1
            if "ld.global" in ins:
                s["ldg"] += 1
            if "st.global" in ins:
                s["stg"] += 1
            if "bssy" in ins:
                s["bssy"] += 1
            if "bar.sync" in ins:
                s["bar"] += 1
    return s


def loops(blocks):
    """Natural loops: for a back edge tail -> head (head <= tail), the head plus every block that reaches the tail
    walking predecessors without passing the head. Innermost (fewest blocks) first."""
    preds = {b: [] for b in blocks}
    for b, info in blocks.items():
        for t in info["succ"]:
            preds.setdefault(t, []).append(b)
    out = []
    for tail, info in blocks.items():
        for head in info["succ"]:
            if head > tail:
                continue
            body = {head, tail}
            stack = [tail]
            while stack:
                n = stack.pop()
                if n == head:
                    continue
                for q in preds.get(n, []):
                    if q not in body:
                        body.add(q)
                        stack.append(q)
            out.append((head, tail, sorted(body)))
    return sorted(out, key=lambda x: (len(x[2]), x[0]))


def main():
    for path in sys.argv[1:]:
        blocks = parse(path)
        print(f"== {path}")
        print("  bucla (blocuri)   instr  stall  ffma lds32 lds128 ldg stg sts bssy bar")
        found = loops(blocks)
        for head, tail, body in found:
            inner = set()
            for h2, t2, b2 in found:
                if set(b2) < set(body):
                    inner |= set(b2)
            s = stats(blocks, [x for x in body if x not in inner])
            print(f"  {head:4d}..{tail:<6d} {s['instr']:9d} {s['stall']:6d} {s['ffma']:5d} {s['lds32']:5d} {s['lds128']:6d}"
                  f" {s['ldg']:3d} {s['stg']:3d} {s['sts']:3d} {s['bssy']:4d} {s['bar']:3d}")




# ---- os-358: the three attention programs on set atentie-513-b4 (B 4, T 512, H 12, d 64), DEDUCED ----
# Loop heads -> (what, how many times per (row tile, column tile) pair or per block, warps of 8 that run it,
# shared wavefronts per warp-iteration). Mapped by hand from the .comp sources; the listings are pinned by SHA-256.
# Waits not covered by the stall counts: +LDS_WAIT per iteration with a shared load, +LDG_WAIT per iteration that uses
# a global load's result (tile loads: one load then its store, so one full memory latency per iteration).
CLOCK_HZ = 1.77e9  # P0 reported by GSP in os-358; the real clock under load is NOT read (no documented path in 570.144)
SMSPS = 48 * 4
LDS_WAIT = 20
LDG_WAIT = 500
PAIRS = 48 * 36  # B*H x lower (row tile, column tile) pairs, T 512 -> 8 row tiles -> 36 pairs per (b, h)
BLOCKS = 48 * 8

PROGRAMS = {
    # name: (listing, warps per SMSP from occupancy, measured ns, loops)
    # loop: (head block, per "pair" | "block", count, warps (of 8) active, waits LDS?, waits LDG?, wavefronts)
    "attn_fwd_tile": ("tools/nak/out/attn_fwd_tile.sm86.txt", 4, 2_491_104, [
        (1, "block", 16, 8, False, True, 1),     # Q tile load
        (11, "pair", 16, 8, False, True, 1),     # K tile load
        (17, "pair", 64, 8, True, False, 72),    # phase A: qs 4 x 2-way + ks 4 x 16-way (rows of 64 floats, same bank)
        (7, "pair", 1, 8, False, False, 0),      # phase A: s -> P
        (74, "pair", 1, 8, False, True, 0),      # phase B: exp, P re-read and re-written
        (132, "pair", 1, 8, False, True, 8),     # phase C: P = x * inv, P tile to shared
        (281, "pair", 16, 8, False, True, 1),    # V tile load
        (287, "pair", 64, 8, True, False, 12),   # phase C: P.V (vb conflict-free, pa 2-way)
    ]),
    "attn_back_tile0": ("tools/nak/out/attn_back_tile0.sm86.txt", 6, 1_813_504, [
        (6, "pair", 16, 8, False, True, 1),      # dO + V tile load (V skewed)
        (15, "pair", 64, 8, True, False, 12),    # dP = dO . V^T (ob 2-way, vb skewed)
        (2, "pair", 1, 8, False, False, 4),      # dP -> dS, dP tile to shared
        (66, "pair", 16, 8, False, True, 1),     # P tile load
        (73, "pair", 64, 2, True, False, 2),     # dot chain: one thread per row, 2 warps of 8
        (90, "block", 8, 8, False, True, 0),     # pass 2 over the whole row: dS = P (dP - dot) scale
    ]),
    "attn_back_tile": ("tools/nak/out/attn_back_tile.sm86.txt", 6, 3_989_696, [
        (6, "pair", 16, 8, False, True, 1),      # dQ: dS + K tile load
        (15, "pair", 64, 8, True, False, 12),    # dQ inner
        (2, "pair", 1, 8, False, False, 0),
        (86, "pair2", 16, 8, False, True, 1),    # dK / dV: tile load (two passes)
        (104, "pair2", 64, 8, True, False, 5),   # dK / dV inner
        (82, "pair2", 1, 8, False, False, 0),
        (81, "block", 2, 8, False, False, 0),
    ]),
}


def os358():
    print("\nos-358, setul atentie-513-b4 (DEDUS din listari, ceasul presupus 1,77 GHz):")
    print("  program           emisie  lant-bloc partajata  -> model (max .. suma)   masurat   masurat/max")
    for name, (path, w, measured, items) in PROGRAMS.items():
        blocks = parse(path)
        found = {h: (h, t, b) for h, t, b in loops(blocks)}
        issue = lat = shared = 0.0
        for head, per, count, warps, lds, ldg, wf in items:
            h, t, body = found[head]
            inner = set()
            for h2, t2, b2 in found.values():
                if set(b2) < set(body):
                    inner |= set(b2)
            s = stats(blocks, [x for x in body if x not in inner])
            times = {"pair": PAIRS, "pair2": 2 * PAIRS, "block": BLOCKS}[per] * count
            wi = times * warps  # warp-iterations
            issue += wi * s["instr"]
            # the block's chain: between barriers every warp waits for the slowest, so a phase run by 2 warps of 8
            # costs the block as long as one run by all 8 (counted per block, not per warp)
            lat += times * 8 * (s["stall"] + (LDS_WAIT if lds else 0) + (LDG_WAIT if ldg else 0))
            shared += wi * wf
        t_issue = issue / SMSPS / CLOCK_HZ
        t_lat = lat / (SMSPS * w) / CLOCK_HZ
        t_sh = shared / 48 / CLOCK_HZ
        lo = max(t_issue, t_lat, t_sh)
        hi = t_issue + t_lat + t_sh
        m = measured / 1e9
        print(f"  {name:16s} {t_issue*1e3:6.2f} {t_lat*1e3:8.2f} {t_sh*1e3:10.2f}    {lo*1e3:5.2f} .. {hi*1e3:5.2f} ms"
              f"   {m*1e3:6.2f} ms   x{m/lo:.1f}")


if __name__ == "__main__" and len(sys.argv) == 1:
    os358()


if __name__ == "__main__" and len(sys.argv) > 1:
    main()
