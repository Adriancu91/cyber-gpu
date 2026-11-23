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


WT = re.compile(r"wt=([01]+)")
WR = re.compile(r"wr:(\d)")


def waits(blocks, ids):
    """Memory waits in a row (os-364, C-157 p.2): walking the loop's own blocks in listing order, a global load whose
    scoreboard ("wr:N") is waited on ("wt=" with bit N) before the next load is issued counts one full memory latency.
    Loads that share a scoreboard and are waited on together count once. Path = every `if` taken (as `stats`)."""
    pending = set()
    n = 0
    for b in sorted(ids):
        for ins in blocks[b]["ins"]:
            m = WT.search(ins)
            if m and pending:
                bits = m.group(1)[::-1]
                hit = {sb for sb in pending if sb < len(bits) and bits[sb] == "1"}
                if hit:
                    n += 1
                    pending -= hit
            if "ld.global" in ins:
                w = WR.search(ins)
                if w:
                    pending.add(int(w.group(1)))
    return n


def waits_fresh(blocks, ids):
    """Fresh memory latencies (session 179, for the branch-free `_3` programs): like `waits`, but a wait counts only
    when the load waited on was issued AFTER the previous counted wait (its latency starts after that wait ended);
    a wait on a load issued before it overlaps a latency already paid and is not counted. NAK has 6 scoreboards, so
    with more than 5 loads in flight a later load waits for an earlier one: 8 branch-free loads = 2 latencies.
    On code whose loads are waited on one at a time this gives the same number as `waits`."""
    pending = {}
    issued = 0
    mark = 0
    n = 0
    for b in sorted(ids):
        for ins in blocks[b]["ins"]:
            m = WT.search(ins)
            if m and pending:
                bits = m.group(1)[::-1]
                hit = [sb for sb in pending if sb < len(bits) and bits[sb] == "1"]
                if hit:
                    if max(pending[sb] for sb in hit) > mark:
                        n += 1
                        mark = issued
                    for sb in hit:
                        del pending[sb]
            if "ld.global" in ins:
                w = WR.search(ins)
                if w:
                    issued += 1
                    pending[int(w.group(1))] = issued
    return n


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
        print("  bucla (blocuri)   instr  stall  ffma lds32 lds128 ldg stg sts bssy bar sir nou")
        found = loops(blocks)
        for head, tail, body in found:
            inner = set()
            for h2, t2, b2 in found:
                if set(b2) < set(body):
                    inner |= set(b2)
            own = [x for x in body if x not in inner]
            s = stats(blocks, own)
            print(f"  {head:4d}..{tail:<6d} {s['instr']:9d} {s['stall']:6d} {s['ffma']:5d} {s['lds32']:5d} {s['lds128']:6d}"
                  f" {s['ldg']:3d} {s['stg']:3d} {s['sts']:3d} {s['bssy']:4d} {s['bar']:3d} {waits(blocks, own):3d} {waits_fresh(blocks, own):3d}")




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


# ---- os-364 (C-157 p.2, session 178): old and new side by side, memory waits COUNTED from the listing ----
# Same constants as os358(). Per loop: (head, per, count, warps of 8, waits LDS?, shared wavefronts, waits override).
# per: "pair" = every lower (row tile, column tile) pair, "below" = pairs under the diagonal, "diag" = diagonal pairs,
# "pair2" / "below2" / "diag2" = the same twice (dK and dV passes), "block" = once per block.
# Waits = `waits()` on the listing (the path where every `if` is taken), except where an override says how many of the
# counted loads really run (attn_back_tile / _2 tile loads: `pass == 0 ? ds : p` -> half of the loads in the listing).
BELOW = 48 * 28
DIAG = 48 * 8
PER = {"pair": PAIRS, "pair2": 2 * PAIRS, "below": BELOW, "below2": 2 * BELOW, "diag": DIAG, "diag2": 2 * DIAG,
       "block": BLOCKS}

PROGRAMS364 = {
    # name: (listing, warps per SMSP, measured ns in os-364, loops)
    "attn_fwd_tile": ("tools/nak/out/attn_fwd_tile.sm86.txt", 4, 2_497_280, [
        (1, "block", 16, 8, False, 1, None), (11, "pair", 16, 8, False, 1, None), (17, "pair", 64, 8, True, 72, None),
        (7, "pair", 1, 8, False, 0, None), (74, "pair", 1, 8, False, 0, None), (132, "pair", 1, 8, False, 8, None),
        (281, "pair", 16, 8, False, 1, None), (287, "pair", 64, 8, True, 12, None),
    ]),
    "attn_fwd_tile2": ("tools/nak/out/attn_fwd_tile2.sm86.txt", 4, 1_527_872, [
        (49, "pair", 1, 8, False, 16, None),     # K tile load (16 guarded loads) + S to P
        (101, "pair", 16, 8, True, 32, None),    # phase A, unrolled by 4, rows of 65
        (161, "pair", 1, 8, False, 0, None),     # phase B: exp, P re-read and re-written
        (219, "pair", 1, 8, False, 16, None),    # phase C outer: P = x * inv, V tile load
        (417, "below", 16, 8, True, 32, None),   # phase C under the diagonal, no guard
        (424, "diag", 64, 8, True, 8, None),     # phase C on the diagonal, guarded
    ]),
    "attn_back_tile0": ("tools/nak/out/attn_back_tile0.sm86.txt", 6, 1_814_272, [
        (6, "pair", 16, 8, False, 1, None), (15, "pair", 64, 8, True, 12, None), (2, "pair", 1, 8, False, 4, None),
        (66, "pair", 16, 8, False, 1, None), (73, "pair", 64, 2, True, 2, None), (90, "block", 8, 8, False, 0, None),
    ]),
    "attn_back_tile0_2": ("tools/nak/out/attn_back_tile0_2.sm86.txt", 6, 1_544_416, [
        (2, "pair", 1, 8, False, 4, None),       # dS store, dP tile to shared
        (6, "pair", 2, 8, False, 8, None),       # dO tile load, 8 at a time
        (33, "pair", 2, 8, False, 8, None),      # V tile load
        (162, "pair", 2, 8, False, 8, None),     # P tile load
        (60, "pair", 16, 8, True, 48, None),     # dP = dO . V^T, unrolled by 4
        (191, "below", 16, 2, True, 8, None),    # dot chain under the diagonal (2 warps of 8)
        (198, "diag", 16, 2, True, 8, None),     # dot chain on the diagonal
        (225, "block", 8, 8, False, 0, None),    # pass 2: dS = P (dP - dot) scale over the whole row
    ]),
    "attn_back_tile": ("tools/nak/out/attn_back_tile.sm86.txt", 6, 3_994_080, [
        (6, "pair", 16, 8, False, 1, None), (15, "pair", 64, 8, True, 12, None), (2, "pair", 1, 8, False, 0, None),
        (86, "pair2", 16, 8, False, 1, 2), (104, "pair2", 64, 8, True, 5, None), (82, "pair2", 1, 8, False, 0, None),
        (81, "block", 2, 8, False, 0, None),
    ]),
    "attn_back_tile_2": ("tools/nak/out/attn_back_tile_2.sm86.txt", 6, 2_250_880, [
        (2, "pair", 1, 8, False, 0, None),       # dQ outer
        (6, "pair", 8, 8, False, 4, None),       # dQ: dS + K tile load, 2 + 2 at a time
        (22, "below", 16, 8, True, 32, None),    # dQ inner under the diagonal
        (29, "diag", 16, 8, True, 32, None),     # dQ inner on the diagonal, guarded
        (132, "block", 2, 8, False, 0, None),    # pass (dK, dV)
        (133, "pair2", 1, 8, False, 0, None),    # it loop
        (137, "pair2", 4, 8, False, 8, 8),       # tile load, 4 + 4 at a time (16 in the listing, 8 run)
        (204, "below2", 16, 8, True, 32, None),  # dK / dV inner under the diagonal
        (211, "diag2", 16, 8, True, 32, None),   # dK / dV inner on the diagonal, guarded
    ]),
}


# Session 179: attn_back_tile0_3 (branch-free loads, docs/C-157_P2_ATTN_TILE3.md); waits counted with waits_fresh().
# Pass 2 is split: below the diagonal (loop 153, 28 tiles per (b, h) = "below"), above (loop 256, no loads, 28 tiles =
# "below" by symmetry); the diagonal tile of pass 2 is straight code (no loop) and is NOT in the model (1 tile of 8).
PROGRAMS179 = {
    "attn_back_tile0_3": ("tools/nak/out/attn_back_tile0_3.sm86.txt", 6, None, [
        (2, "pair", 1, 8, False, 4, None),       # dS store, dP tile to shared
        (6, "pair", 2, 8, False, 8, None),       # dO tile load, 8 at a time, branch-free
        (9, "pair", 2, 8, False, 8, None),       # V tile load
        (114, "pair", 2, 8, False, 8, None),     # P tile load
        (12, "pair", 16, 8, True, 48, None),     # dP = dO . V^T, unrolled by 4 (as _2)
        (119, "below", 16, 2, True, 8, None),    # dot chain under the diagonal
        (126, "diag", 16, 2, True, 8, None),     # dot chain on the diagonal
        (153, "below", 1, 8, False, 0, None),    # pass 2 under the diagonal: 32 loads first, then the arithmetic
        (256, "below", 1, 8, False, 0, None),    # pass 2 above the diagonal: 0.0 stored, no load
    ]),
}
PROGRAMS364.update(PROGRAMS179)

# Session 180: attn_back_tile_3 (branch-free tile loads, docs/C-157_P2_ATTN_TILE3.md); the inner loops = attn_back_tile_2's
# (same instruction counts in the listing); only the two load loops change. dK / dV loads 2 + 2 at a time (8 iterations).
PROGRAMS180 = {
    "attn_back_tile_3": ("tools/nak/out/attn_back_tile_3.sm86.txt", 6, None, [
        (2, "pair", 1, 8, False, 0, None),       # dQ outer
        (6, "pair", 8, 8, False, 4, None),       # dQ: dS + K tile load, 2 + 2 at a time, branch-free
        (10, "below", 16, 8, True, 32, None),    # dQ inner under the diagonal
        (17, "diag", 16, 8, True, 32, None),     # dQ inner on the diagonal, guarded
        (120, "block", 2, 8, False, 0, None),    # pass (dK, dV)
        (124, "pair2", 1, 8, False, 0, None),    # it loop
        (128, "pair2", 8, 8, False, 4, None),    # tile load, 2 + 2 at a time, branch-free
        (135, "below2", 16, 8, True, 32, None),  # dK / dV inner under the diagonal
        (142, "diag2", 16, 8, True, 32, None),   # dK / dV inner on the diagonal, guarded
    ]),
}
PROGRAMS364.update(PROGRAMS180)
PROGRAMS179.update(PROGRAMS180)  # kept out of os364()'s list of measured programs


# Session 181 (C-157 p.2 after os-364): attn_fwd_tile3 = attn_fwd_tile2 with branch-free loads (DEDUS, unmeasured).
PROGRAMS181 = {
    "attn_fwd_tile3": ("tools/nak/out/attn_fwd_tile3.sm86.txt", 4, None, [
        (1, "pair", 1, 8, False, 16, None),      # K tile load (2 x 8 branch-free loads) + S to P
        (5, "pair", 16, 8, True, 32, None),      # phase A, unrolled by 4, rows of 65 (as _2)
        (65, "pair", 1, 8, False, 0, None),      # phase B: 16 P loads first, exp, select, guarded stores
        (123, "pair", 1, 8, False, 16, None),    # phase C outer: 16 P loads first, P = x * inv, V tile load (2 x 8)
        (225, "below", 16, 8, True, 32, None),   # phase C under the diagonal (as _2)
        (232, "diag", 64, 8, True, 8, None),     # phase C on the diagonal, guarded (as _2)
    ]),
}
PROGRAMS364.update(PROGRAMS181)
PROGRAMS179.update(PROGRAMS181)  # kept out of os364()'s list of measured programs


# Session 188: attn_back_tile_4 = _3 with the full diagonal tile of dK / dV on the unguarded loop (K1). On atentie-513-b4
# (T 512) every tile is full: the unguarded loop runs for below + diagonal (= every pair), the guarded one never.
PROGRAMS188 = {
    "attn_back_tile_4": ("tools/nak/out/attn_back_tile_4.sm86.txt", 6, None, [
        (2, "pair", 1, 8, False, 0, None),       # dQ outer
        (6, "pair", 8, 8, False, 4, None),       # dQ: dS + K tile load (as _3)
        (10, "below", 16, 8, True, 32, None),    # dQ inner under the diagonal (as _3)
        (17, "diag", 16, 8, True, 32, None),     # dQ inner on the diagonal, guarded (as _3)
        (120, "block", 2, 8, False, 0, None),    # pass (dK, dV)
        (124, "pair2", 1, 8, False, 0, None),    # it loop
        (128, "pair2", 8, 8, False, 4, None),    # tile load (as _3)
        (132, "pair2", 16, 8, True, 32, None),   # K1: dK / dV inner, unguarded, below AND on the diagonal
    ]),
}
PROGRAMS364.update(PROGRAMS188)
PROGRAMS179.update(PROGRAMS188)  # kept out of os364()'s list of measured programs


def model364(name, lw=LDG_WAIT, drop_waits=False, fresh=False):
    """(issue, chain, shared, waits per block-pair summary) in seconds for one program; drop_waits=True models the
    branch-free loads (every tile load / pass waited on once per iteration instead of once per load)."""
    path, w, measured, items = PROGRAMS364[name]
    blocks = parse(path)
    found = {h: (h, t, b) for h, t, b in loops(blocks)}
    issue = lat = shared = 0.0
    mem_lat = 0.0
    for head, per, count, warps, lds, wf, override in items:
        h, t, body = found[head]
        inner = set()
        for h2, t2, b2 in found.values():
            if set(b2) < set(body):
                inner |= set(b2)
        own = [x for x in body if x not in inner]
        s = stats(blocks, own)
        nw = (waits_fresh(blocks, own) if fresh else waits(blocks, own)) if override is None else override
        if drop_waits and nw > 1:
            nw = 1
        times = PER[per] * count
        issue += times * warps * s["instr"]
        chain = s["stall"] + (LDS_WAIT if lds else 0) + nw * lw
        lat += times * 8 * chain
        mem_lat += times * 8 * nw * lw
        shared += times * warps * wf
    k = SMSPS * CLOCK_HZ
    return issue / k, lat / (k * w), shared / 48 / CLOCK_HZ, mem_lat / (k * w), (measured or 0) / 1e9


def os364():
    print("\nos-364, setul atentie-513-b4, vechi langa nou (DEDUS din listari; asteptarile memoriei NUMARATE; 1,77 GHz):")
    print("  program             emisie  lant-bloc (din el memoria)  partajata   model max   masurat   masurat/max")
    ratios = {}
    for name in [n for n in PROGRAMS364 if n not in PROGRAMS179]:
        i, l, sh, mem, m = model364(name)
        lo = max(i, l, sh)
        ratios[name] = (lo, m)
        print(f"  {name:18s} {i*1e3:6.2f} {l*1e3:9.2f}  ({mem*1e3:5.2f})        {sh*1e3:8.2f}    {lo*1e3:6.2f} ms"
              f"   {m*1e3:6.2f} ms   x{m/lo:.2f}")
    print("  castigul nou / vechi:   model      masurat")
    for old, new in (("attn_fwd_tile", "attn_fwd_tile2"), ("attn_back_tile0", "attn_back_tile0_2"),
                     ("attn_back_tile", "attn_back_tile_2")):
        print(f"  {new:18s}      x{ratios[old][0]/ratios[new][0]:.2f}      x{ratios[old][1]/ratios[new][1]:.2f}")
    print("  citiri fara ramura (o asteptare pe iteratie de incarcare), programele noi, model max x factorul masurat:")
    for name in ("attn_fwd_tile2", "attn_back_tile0_2", "attn_back_tile_2"):
        lo, m = ratios[name]
        i, l, sh, mem, _ = model364(name, drop_waits=True)
        lo2 = max(i, l, sh)
        print(f"  {name:18s}  model {lo*1e3:5.2f} -> {lo2*1e3:5.2f} ms; cu factorul x{m/lo:.2f}: {m*1e3:5.2f} -> "
              f"~{lo2*m/lo*1e3:5.2f} ms")


def s179():
    print("\nsesiunea 179, attn_back_tile0_3 (citiri fara ramura; asteptarile NUMARATE cu waits_fresh; DEDUS, nemasurat):")
    i2, l2, sh2, mem2, m2 = model364("attn_back_tile0_2", fresh=True)
    lo2 = max(i2, l2, sh2)
    i3, l3, sh3, mem3, _ = model364("attn_back_tile0_3", fresh=True)
    lo3 = max(i3, l3, sh3)
    f = m2 / lo2
    print(f"  attn_back_tile0_2  emisie {i2*1e3:5.2f} lant {l2*1e3:5.2f} (memoria {mem2*1e3:5.2f}) partajata {sh2*1e3:5.2f}"
          f" -> model {lo2*1e3:5.2f} ms, masurat os-364 {m2*1e3:5.2f} ms (x{f:.2f})")
    print(f"  attn_back_tile0_3  emisie {i3*1e3:5.2f} lant {l3*1e3:5.2f} (memoria {mem3*1e3:5.2f}) partajata {sh3*1e3:5.2f}"
          f" -> model {lo3*1e3:5.2f} ms; cu factorul x{f:.2f}: ~{lo3*f*1e3:5.2f} ms (castig x{lo2/lo3:.2f})")


def s180():
    print("\nsesiunea 180, attn_back_tile_3 (citiri fara ramura; asteptarile NUMARATE cu waits_fresh; DEDUS, nemasurat):")
    i2, l2, sh2, mem2, m2 = model364("attn_back_tile_2", fresh=True)
    lo2 = max(i2, l2, sh2)
    i3, l3, sh3, mem3, _ = model364("attn_back_tile_3", fresh=True)
    lo3 = max(i3, l3, sh3)
    f = m2 / lo2
    print(f"  attn_back_tile_2   emisie {i2*1e3:5.2f} lant {l2*1e3:5.2f} (memoria {mem2*1e3:5.2f}) partajata {sh2*1e3:5.2f}"
          f" -> model {lo2*1e3:5.2f} ms, masurat os-364 {m2*1e3:5.2f} ms (x{f:.2f})")
    print(f"  attn_back_tile_3   emisie {i3*1e3:5.2f} lant {l3*1e3:5.2f} (memoria {mem3*1e3:5.2f}) partajata {sh3*1e3:5.2f}"
          f" -> model {lo3*1e3:5.2f} ms; cu factorul x{f:.2f}: ~{lo3*f*1e3:5.2f} ms (castig x{lo2/lo3:.2f})")


def s181():
    print("\nsesiunea 181, attn_fwd_tile3 (citiri fara ramura; asteptarile NUMARATE cu waits_fresh; DEDUS, nemasurat):")
    i2, l2, sh2, mem2, m2 = model364("attn_fwd_tile2", fresh=True)
    lo2 = max(i2, l2, sh2)
    i3, l3, sh3, mem3, _ = model364("attn_fwd_tile3", fresh=True)
    lo3 = max(i3, l3, sh3)
    f = m2 / lo2
    print(f"  attn_fwd_tile2     emisie {i2*1e3:5.2f} lant {l2*1e3:5.2f} (memoria {mem2*1e3:5.2f}) partajata {sh2*1e3:5.2f}"
          f" -> model {lo2*1e3:5.2f} ms, masurat os-364 {m2*1e3:5.2f} ms (x{f:.2f})")
    print(f"  attn_fwd_tile3     emisie {i3*1e3:5.2f} lant {l3*1e3:5.2f} (memoria {mem3*1e3:5.2f}) partajata {sh3*1e3:5.2f}"
          f" -> model {lo3*1e3:5.2f} ms; cu factorul x{f:.2f}: ~{lo3*f*1e3:5.2f} ms (castig x{lo2/lo3:.2f})")


# Session 186: the 15:57 cycle (cyber-gpu d885efd), atentie-513-b4, three generations in one run (medians of 5, ns).
CYCLE1557_NS = {
    "attn_fwd_tile": 2488096, "attn_back_tile0": 1813952, "attn_back_tile": 3995008,
    "attn_fwd_tile2": 1524224, "attn_back_tile0_2": 1547136, "attn_back_tile_2": 2257792,
    "attn_fwd_tile3": 1000192, "attn_back_tile0_3": 829536, "attn_back_tile_3": 1596608,
}


def att3():
    print("\nciclul 15:57, atentie-513-b4, trei generatii (DEDUS din listari; masurat pe placa; 1,77 GHz):")
    print("  program             model max (lw=500)   masurat   masurat/max   limitat de")
    for name, m in CYCLE1557_NS.items():
        i, l, sh, mem, _ = model364(name, fresh=name.endswith("3"))
        lo = max(i, l, sh)
        by = "emisie" if lo == i else ("lant" if lo == l else "partajata")
        print(f"  {name:18s}  {lo*1e3:6.2f} ms            {m/1e6:6.3f} ms   x{m/1e9/lo:.2f}        {by}")
    print("  latenta memoriei lw (cicluri) -> rapoartele masurat/max: cel mai mic .. cel mai mare (imprastierea)")
    best = None
    for lw in (100, 200, 300, 400, 500, 700, 1000, 1500, 2000, 3000):
        rs = []
        for name, m in CYCLE1557_NS.items():
            i, l, sh, mem, _ = model364(name, lw=lw, fresh=name.endswith("3"))
            rs.append(m / 1e9 / max(i, l, sh))
        spread = max(rs) / min(rs)
        if best is None or spread < best[1]:
            best = (lw, spread)
        print(f"  lw {lw:5d}: x{min(rs):.2f} .. x{max(rs):.2f}  (x{spread:.2f})")
    print(f"  cel mai strans: lw = {best[0]} cicluri, imprastierea x{best[1]:.2f}")
    print("  fara suprapunere (emisie + lant + partajata, lw=500):")
    rs = []
    for name, m in CYCLE1557_NS.items():
        i, l, sh, mem, _ = model364(name, fresh=name.endswith("3"))
        rs.append(m / 1e9 / (i + l + sh))
        print(f"  {name:18s}  suma {(i+l+sh)*1e3:5.2f} ms   masurat/suma x{rs[-1]:.2f}")
    print(f"  imprastierea x{max(rs)/min(rs):.2f}")


def s188():
    print("\nsesiunea 188, attn_back_tile_4 (K1: placa intreaga a diagonalei dK / dV fara garda; DEDUS, nemasurat):")
    m3 = CYCLE1557_NS["attn_back_tile_3"] / 1e9
    i3, l3, sh3, _, _ = model364("attn_back_tile_3", fresh=True)
    i4, l4, sh4, _, _ = model364("attn_back_tile_4", fresh=True)
    lo3, lo4 = max(i3, l3, sh3), max(i4, l4, sh4)
    su3, su4 = i3 + l3 + sh3, i4 + l4 + sh4
    print(f"  attn_back_tile_3   emisie {i3*1e3:5.3f} lant {l3*1e3:5.3f} partajata {sh3*1e3:5.3f}  max {lo3*1e3:5.3f}"
          f"  suma {su3*1e3:5.3f} ms; masurat 15:57 {m3*1e3:5.3f} ms")
    print(f"  attn_back_tile_4   emisie {i4*1e3:5.3f} lant {l4*1e3:5.3f} partajata {sh4*1e3:5.3f}  max {lo4*1e3:5.3f}"
          f"  suma {su4*1e3:5.3f} ms")
    print(f"  pe placa cu factorul lui _3: dupa max ~{lo4*m3/lo3*1e3:5.2f} ms (x{lo3/lo4:.2f}),"
          f" dupa suma ~{su4*m3/su3*1e3:5.2f} ms (x{su3/su4:.2f})")

if __name__ == "__main__" and len(sys.argv) == 1:
    os358()
    os364()
    s179()
    s180()
    s181()
    att3()
    s188()


if __name__ == "__main__" and len(sys.argv) > 1:
    main()
