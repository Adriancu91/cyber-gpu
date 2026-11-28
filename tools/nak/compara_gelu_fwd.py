#!/usr/bin/env python3
"""C-168 p.5 P5c (session 293, docs/C-168_P5_LIPIRE_PLAN.md section 9, prediction L3): is the `act` store of a `geluf`
program the board's `gelu_tr` mode 0, operation for operation?

    python3 tools/nak/compara_gelu_fwd.py tools/nak/out/gelu_tr.sm86.txt tools/nak/out2/mm_tc_v8n_geluf.sm86.txt \
        [tools/nak/out2/mm_tc_v8n.sm86.txt]

The reference `gelu_tr.sm86.txt` was made by the FIRST NAK tool (Mesa 25.0.7, the program on the board); the `geluf`
programs by the second one (Mesa 26.3.1). The two print the same arithmetic in two forms: 25.0.7 writes |x| and -x as
separate `fadd.ftz -rZ ...` instructions and loads constants with `mov` into the operand, 26.3.1 folds them into the
operand modifiers (`|r12|`, `fmul.ftz -r19 r13`) and names `mufu.rcp.f32`. Both listings are turned into expression
trees and then rewritten with EXACT identities only:
  - fadd.ftz(-0, X) = X           (X is already an ftz result, so flushing again changes nothing; -0 + x = x)
  - fmul.ftz(-a, b) = -fmul.ftz(a, b), fmul.ftz(a, -b) = -fmul.ftz(a, b)   (round-to-nearest is sign-symmetric)
  - -(-x) = x, |-x| = |x|, `mov C` = C, mufu.*.f32 = mufu.*
Anything else (another operation, constant, order of operands, rounding) stays a difference.
The reference epilogue = Phi(u) (from `fmul.ftz u 0x3f3504f3` to `fmul.ftz . 0x3f000000`), then mode 0's
`fmul.ftz u Phi`. Each epilogue of the new listing (from the same first instruction to its `st.global`) must give
`fmul.ftz u Phi(u)` with the same tree, `u` must be the register the program stored to C just before (`c[0x0][0x710]`),
and the epilogue's store must go to `act` (`c[0x0][0x730]`). With a third argument (the plain program) the float and
tensor-core operations of the new listing, the GELU epilogues (and the `mov 1.0` right before each) left out, must be
the plain program's, in the same order (the u store unchanged).
"""
import re
import sys

START = re.compile(r"fmul\.ftz (\S+) 0x3f3504f3$")
FLOAT_OP = re.compile(r"=\s*(f\w*|mufu\S*|mov|sel|hmma\S*)\b")


def clean(path):
    return [l.split("//")[0].strip() for l in open(path, encoding="utf-8")]


def operand(tok, env):
    neg = tok.startswith("-")
    t = tok[1:] if neg else tok
    ab = t.startswith("|") and t.endswith("|")
    if ab:
        t = t[1:-1]
    if t == "rZ":
        e = ("c", "0")
    elif re.fullmatch(r"0x[0-9a-f]+", t):
        e = ("c", t)
    elif re.fullmatch(r"[rp]\d+", t):
        e = env.get(t, ("in", t))
    else:
        e = ("tok", t)
    if ab:
        e = ("abs", e[1] if e[0] == "neg" else e)
    if neg:
        e = e[1] if e[0] == "neg" else ("neg", e)
    return e


def apply(op, args):
    op = re.sub(r"\.f32$", "", op)
    if op == "mov":
        return args[0]
    if op == "fadd.ftz" and args[0] == ("c", "0") or op == "fadd.ftz" and args[0] == ("neg", ("c", "0")):
        return args[1]
    if op == "fmul.ftz":
        sign = False
        a = []
        for x in args:
            if x[0] == "neg":
                sign = not sign
                x = x[1]
            a.append(x)
        e = ("op", op, tuple(a))
        return ("neg", e) if sign else e
    return ("op", op, tuple(args))


def build(lines, env):
    """Expression of every register written by `lines` (float-only instructions)."""
    last = None
    for l in lines:
        l = re.sub(r"\s+", " ", l)
        dst, _, rhs = l.partition(" = ")
        parts = rhs.split()
        e = apply(parts[0], [operand(t, env) for t in parts[1:]])
        for d in dst.split():
            env[d] = e
            last = d
    return env, last


def show(e):
    if e[0] == "op":
        return "(" + e[1] + " " + " ".join(show(x) for x in e[2]) + ")"
    if e[0] == "neg":
        return "-" + show(e[1])
    if e[0] == "abs":
        return "|" + show(e[1]) + "|"
    if e[0] == "in":
        return "u" if e[1] == "U" else "in:" + e[1]
    return e[1]


def reference(path):
    ls = clean(path)
    i = next(k for k, l in enumerate(ls) if START.search(l))
    u = START.search(ls[i]).group(1)
    j = next(k for k in range(i, len(ls)) if "0x3f000000" in ls[k])
    env = {u: ("in", "U")}
    body = [l for l in ls[i:j + 1] if FLOAT_OP.search(l)]
    env, phi = build(body, env)
    # mode 0 (block after the branch): `fmul.ftz <u> <phi>` then the store to y (0x38)
    m = next(l for l in ls[j:] if re.search(r"= fmul\.ftz %s %s$" % (u, phi), re.sub(r"\s+", " ", l)))
    assert m, "mode 0 nu e fmul.ftz u phi"
    return show(("op", "fmul.ftz", (("in", "U"), env[phi])))


def store_base(ls, k):
    """The cbuf slot (0x710 = C, 0x730 = act) the address of the store at line k was computed from: the `lea` that
    wrote the store's address register, searched backwards."""
    m = re.search(r"\[(r\d+)", ls[k])
    r = m.group(1)
    for x in range(k - 1, -1, -1):
        if re.match(r"%s(\s|$)" % r, ls[x]):
            c = re.search(r"c\[0x0\]\[(0x[0-9a-f]+)\]", ls[x])
            return c.group(1) if c else None
    return None


def epilogues(path):
    """Each GELU epilogue of a geluf listing: (tree of the act value, C store of u found at 0x710, act store at 0x730,
    first line, act store line, lines of the epilogue's float ops)."""
    ls = clean(path)
    out = []
    i = 0
    while i < len(ls):
        m = START.search(ls[i])
        if not m:
            i += 1
            continue
        u = m.group(1)
        env = {u: ("in", "U")}
        # an input register other than u (NAK keeps the constant 1.0 in a register): its last write before the
        # epilogue must be `mov <constant>`, which is then that constant
        j = i
        while not ("st.global" in ls[j] and "0x730" == store_base(ls, j)):
            j += 1
        body = [n for n in range(i, j) if FLOAT_OP.search(ls[n])]
        for r in sorted(set(re.findall(r"\br\d+\b", " ".join(ls[n].partition(" = ")[2] for n in body))) - {u}):
            w = next((x for x in range(i - 1, -1, -1) if re.match(r"%s(\s|$)" % r, ls[x])), None)
            if w is not None and re.fullmatch(r"%s\s*=\s*mov (0x[0-9a-f]+)" % r, ls[w]):
                env[r] = ("c", ls[w].split()[-1])
        env, _ = build([ls[n] for n in body], env)
        val = ls[j].split()[-1]
        # the C store of u: the last store before the epilogue, or one inside it, of register u, at 0x710
        su = False
        for k in list(range(i, j)) + [max(x for x in range(i) if "st.global" in ls[x])]:
            if "st.global" in ls[k] and ls[k].split()[-1] == u and store_base(ls, k) == "0x710":
                su = True
        out.append((show(env[val]), su, True, i, j, body))
        i = j + 1
    return out


def ops(lines, skip=()):
    res = []
    for n, l in enumerate(lines):
        if n in skip:
            continue
        m = FLOAT_OP.search(l)
        if m:
            res.append(re.sub(r"\s+", " ", l.partition(" = ")[2]).split()[0])
    return res


def main():
    ref = reference(sys.argv[1])
    eps = epilogues(sys.argv[2])
    bad = [n for n, (t, su, sa, _, _, _) in enumerate(eps) if t != ref or not su or not sa]
    print(f"referinta gelu_tr mod 0: {ref.count('(')} operatii; epiloguri act: {len(eps)}; diferite: {bad}")
    if bad:
        print("ref:", ref)
        print("nou:", eps[bad[0]][0], eps[bad[0]][1:3])
    ok = not bad and eps
    if len(sys.argv) > 3:
        ls = clean(sys.argv[2])
        skip = set()
        for _, _, _, i, j, body in eps:
            skip.update(body)
            if "mov 0x3f800000" in ls[i - 1]:
                skip.add(i - 1)
        a = ops(clean(sys.argv[3]))
        b = ops(ls, skip)
        same = a == b
        print(f"restul (virgula mobila + hmma): programul simplu {len(a)}, geluf fara cozi {len(b)}: "
              f"{'IDENTIC' if same else 'DIFERIT'}")
        ok = ok and same
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
