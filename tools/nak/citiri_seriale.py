#!/usr/bin/env python3
"""Count how many global loads of a NAK listing are waited on before the next global load is issued.

Usage: python3 -I tools/nak/citiri_seriale.py tools/nak/out2/mm_tc.sm86.txt [more listings...]

Reads the text listings written by tools/nak/build.sh / build2.sh (`*.sm86.txt`). For every `ld.global` it takes the
scoreboard the load writes (`wr:N`) and looks, up to the next `ld.global`, for an instruction whose wait mask (`wt=xxxxxx`,
rightmost digit = scoreboard 0) contains N. Such a load is "serial": the thread stops until the data arrives before it
can ask for the next value, so no two of its loads are in flight together. With few warps per SM nothing hides that wait.

Also prints the longest run of loads issued before the first wait ("in zbor" = in flight). Text only, no board access.
Our own script (C-157 p.2, os-423 comparison). Exit code 0 always; the numbers are the result.

Session 229 adds the STRICT count ("strict"): a scoreboard is a counter, and a wait on it waits for EVERY operation
counted there (data writes `wr:N` and operand reads `rd:N` alike). Walking the listing in order, a load stays in flight
until any instruction waits on its scoreboard (even a later load that needs a free scoreboard, or an unrelated read);
"strict" is the largest number of loads in flight at once. With 6 scoreboards NAK cannot keep more than ~6 apart.
"""
import re
import sys


def analyse(path):
    lines = open(path, encoding="utf-8").read().splitlines()
    loads = [i for i, line in enumerate(lines) if "ld.global" in line]
    serial = 0
    best_run = 0
    run = 0
    for a, i in enumerate(loads):
        m = re.search(r"wr:(\d)", lines[i])
        end = loads[a + 1] if a + 1 < len(loads) else len(lines)
        waited = False
        if m:
            sb = int(m.group(1))
            for x in lines[i + 1:end + 1]:
                w = re.search(r"wt=([01]{6})", x)
                if w and w.group(1)[::-1][sb] == "1":
                    waited = True
                    break
        run += 1
        best_run = max(best_run, run)
        if waited:
            serial += 1
            run = 0
    out = {}
    strict = 0
    for line in lines:
        w = re.search(r"wt=([01]{6})", line)
        if w:
            for sb, bit in enumerate(w.group(1)[::-1]):
                if bit == "1":
                    out.pop(sb, None)
        if "ld.global" in line:
            m = re.search(r"wr:(\d)", line)
            if m:
                out[int(m.group(1))] = out.get(int(m.group(1)), 0) + 1
            strict = max(strict, sum(out.values()))
    return len(loads), serial, best_run, strict


def main():
    for path in sys.argv[1:]:
        n, serial, best_run, strict = analyse(path)
        name = path.replace("\\", "/").split("/")[-1]
        print(f"{name}: ld.global {n}, asteptate inainte de urmatoarea citire {serial}, cel mult {best_run} in zbor, strict {strict} in zbor")


if __name__ == "__main__":
    main()
