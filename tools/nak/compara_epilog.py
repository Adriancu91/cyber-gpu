#!/usr/bin/env python3
"""C-165 follow-up (docs/C-165_MM_TILE_GELU.md): compare the GELU' epilogues of two NAK listings.

Each epilogue starts at the `fmul.ftz rX 0x3f3504f3` (u / sqrt(2)) and ends at the multiplication by the sum
(the last `fmul.ftz` before the address `iadd3`). Registers are replaced by the values they hold (dataflow form, see
`norm`), the `// delay` comments dropped; every epilogue of the second listing must equal the (single) epilogue of
the first one, instruction for instruction.

    python3 tools/nak/compara_epilog.py tools/nak/out/mm_train_gelu.sm86.txt tools/nak/out/mm_train_tile_gelu.sm86.txt
"""
import re
import sys


def epilogues(path):
    lines = [l.split("//")[0].strip() for l in open(path, encoding="utf-8")]
    out, cur = [], None
    for l in lines:
        if cur is None and re.search(r"fmul\.ftz \S+ 0x3f3504f3$", l):
            cur = []
        if cur is not None:
            if "iadd3" in l:
                out.append(cur)
                cur = None
                continue
            if l:
                cur.append(l)
    return out


def norm(ep):
    """Dataflow form: every written register becomes a new value t<k>; a read names the value it reads (an input
    never written inside the epilogue becomes in<k>). Two epilogues are equal when they do the same operations on
    the same values, whatever registers NAK picked."""
    cur, ins, res, k = {}, {}, [], 0
    for l in ep:
        l = re.sub(r"\s+", " ", l)
        dst, _, rhs = l.partition(" = ")
        def src(m):
            r = m.group(0)
            if r in cur:
                return cur[r]
            ins.setdefault(r, "in%d" % len(ins))
            return ins[r]
        rhs = re.sub(r"\b(?:r\d+|p\d+)\b", src, rhs)
        outs = []
        for d in dst.split():
            cur[d] = "t%d" % k
            outs.append(cur[d])
            k += 1
        res.append(" ".join(outs) + " = " + rhs)
    return res


def main():
    ref_all, new_all = epilogues(sys.argv[1]), epilogues(sys.argv[2])
    assert len(ref_all) == 1, f"{sys.argv[1]}: {len(ref_all)} epiloguri"
    ref = norm(ref_all[0])
    bad = [i for i, e in enumerate(new_all) if norm(e) != ref]
    print(f"referinta: {len(ref)} instructiuni; epiloguri noi: {len(new_all)}; diferite: {bad}")
    sys.exit(1 if bad or not new_all else 0)


if __name__ == "__main__":
    main()
