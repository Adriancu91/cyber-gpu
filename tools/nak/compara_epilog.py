#!/usr/bin/env python3
"""C-165 follow-up (docs/C-165_MM_TILE_GELU.md): compare the GELU' epilogues of two NAK listings.

Each epilogue starts at the `fmul.ftz rX 0x3f3504f3` (u / sqrt(2)) and ends at the multiplication by the sum
(the last `fmul.ftz` before the address `iadd3`). Registers are replaced by the values they hold (dataflow form, see
`norm`), the `// delay` comments dropped; every epilogue of the second listing must equal the (single) epilogue of
the first one, instruction for instruction.

    python3 tools/nak/compara_epilog.py tools/nak/out/mm_train_gelu.sm86.txt tools/nak/out/mm_train_tile_gelu.sm86.txt

With `--float` (C-168 p.5 P5b, session 291) only the floating-point operations are compared (f*, mufu, mov, sel), and
an epilogue also ends at the store (`st.global`): the address arithmetic (lea / iadd3) is left out, so a store
address NAK schedules between the last two multiplications does not count as a difference, and the epilogues are
compared as expression trees of the stored value (`tree`): every operation with its operands, recursively, the
inputs named in the order the tree reaches them. An independent instruction NAK moves (the sign test `fsetp` one
place earlier or later) is then not a difference; any other operation, constant, operand or order of operands is:

    python3 tools/nak/compara_epilog.py --float tools/nak/out2/mm_tc_gelu.sm86.txt tools/nak/out2/mm_tc_v8n_gelu.sm86.txt
"""
import re
import sys


FLOAT_OP = re.compile(r"=\s*(f\w*|mufu|mov|sel)\b")


def epilogues(path, only_float=False):
    lines = [l.split("//")[0].strip() for l in open(path, encoding="utf-8")]
    out, cur = [], None
    for l in lines:
        if cur is None and re.search(r"fmul\.ftz \S+ 0x3f3504f3$", l):
            cur = []
        if cur is not None:
            if "iadd3" in l or (only_float and "st.global" in l):
                out.append(cur)
                cur = None
                continue
            if l and (not only_float or FLOAT_OP.search(l)):
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


def tree(ep):
    """The stored value of one float-only epilogue (its last instruction) as an expression tree string: operands are
    replaced, in order, by the expression they hold at that point (registers are reused, so the substitution is done
    forward), inputs become leaves renamed in the order the final string reaches them."""
    cur, last = {}, None
    for l in ep:
        l = re.sub(r"\s+", " ", l)
        dst, _, rhs = l.partition(" = ")
        e = "(" + re.sub(r"\b(?:r\d+|p\d+)\b", lambda m: cur.get(m.group(0), "<" + m.group(0) + ">"), rhs) + ")"
        for d in dst.split():
            cur[d] = e
            last = d
    out, names = cur[last], {}
    return re.sub(r"<[^>]+>", lambda m: names.setdefault(m.group(0), "in%d" % len(names)), out)


def main():
    args = sys.argv[1:]
    only_float = args[:1] == ["--float"]
    if only_float:
        args = args[1:]
    sys.argv[1:] = args
    ref_all = epilogues(sys.argv[1], only_float)
    new_all = epilogues(sys.argv[2], only_float)
    assert len(ref_all) == 1, f"{sys.argv[1]}: {len(ref_all)} epiloguri"
    form = tree if only_float else norm
    ref = form(ref_all[0])
    bad = [i for i, e in enumerate(new_all) if form(e) != ref]
    size = len(ref_all[0])
    print(f"referinta: {size} instructiuni; epiloguri noi: {len(new_all)}; diferite: {bad}")
    sys.exit(1 if bad or not new_all else 0)


if __name__ == "__main__":
    main()
