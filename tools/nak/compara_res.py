#!/usr/bin/env python3
"""C-168 p.5 P5d (session 294, docs/C-168_P5_LIPIRE_PLAN.md section 12, prediction M3): compare a `-DRES` NAK listing
with the listing of its plain program.

For every store of C (`st.global` whose address is built from c[0x0][0x710]) the `_res` listing must have, in the same
block: an address built from c[0x0][0x730] with the SAME index register as the C address, one `ld.global` from it,
and exactly one `fadd.ftz` whose operands are the value the plain program would store and the loaded value, whose
result is what is stored. Those four instructions (2 lea, ld, fadd) are then removed; what is left must equal the
plain listing as a sequence of operations with their constants (registers renamed away): every floating-point op
(f*, mufu, hmma), every load and store, in the same order. Exit 1 on any difference.

    python3 tools/nak/compara_res.py tools/nak/out2/mm_tc_v8.sm86.txt tools/nak/out2/mm_tc_v8_res.sm86.txt
    python3 tools/nak/compara_res.py --self-test tools/nak/out2/mm_tc_v8.sm86.txt tools/nak/out2/mm_tc_v8_res.sm86.txt
"""
import re
import sys

REG = re.compile(r"^(r\d+|p\d+|u?r?\w*\d+|rZ|pT)$")
KEEP = re.compile(r"=\s*(f\w*|mufu|hmma\S*|ld\.\S+)\b|^\s*st\.\S+")


def lines_of(text):
    return [l.split("//")[0].rstrip() for l in text.splitlines()]


def parse(l):
    """dest regs, op, operand tokens"""
    if "=" in l:
        d, r = l.split("=", 1)
        t = r.split()
        return d.split(), t[0], t[1:]
    t = l.split()
    return [], (t[0] if t else ""), t[1:]


def shape(l):
    d, op, args = parse(l)
    keep = [a for a in args if not re.match(r"^\[?(r|p|ur|up)\d+", a) and a not in ("rZ", "pT")]
    return op + " " + " ".join(keep)


def strip_res(lines):
    """Checks every C store of a -DRES listing and returns (listing without the residual instructions, stores, errors)."""
    out, errs, stores = list(lines), [], 0
    drop = set()
    for i, l in enumerate(lines):
        if "st.global" not in l:
            continue
        # the C address: the lea of 0x710 nearest before the store (NAK may put it before or after the res address)
        j = i - 1
        while j >= 0 and i - j <= 10 and "c[0x0][0x710]" not in lines[j]:
            j -= 1
        if j < 0 or i - j > 10:
            errs.append(f"linia {i+1}: scriere fara adresa C")
            continue
        stores += 1
        ci = parse(lines[j])[2][0]  # index register of C
        val = l.split()[-1]  # the stored value
        # backwards from the store: the nearest fadd writing the stored value (the bias add comes earlier, in its
        # own block, reading from 0x718; the residual add must be the last write of the value)
        k = i - 1
        fadd = None
        while k >= 0 and k >= i - 12:
            d, op, a = parse(lines[k])
            if d and d[0] == val:
                fadd = k if op.startswith("fadd") else None
                break
            k -= 1
        if fadd is None:
            errs.append(f"linia {i+1}: valoarea scrisa ({val}) nu vine dintr-o fadd chiar inainte")
            continue
        d, op, a = parse(lines[fadd])
        if op != "fadd.ftz" or len(a) != 2:
            errs.append(f"linia {fadd+1}: {lines[fadd].strip()} nu e o fadd.ftz simpla")
            continue
        ld = None
        for m in range(fadd - 1, max(fadd - 8, -1), -1):
            dd, oo, aa = parse(lines[m])
            if oo.startswith("ld.global") and dd and dd[0] in a:
                ld = m
                break
        if ld is None:
            errs.append(f"linia {fadd+1}: fadd fara citire globala")
            continue
        loaded = parse(lines[ld])[0][0]
        other = a[0] if a[1] == loaded else a[1]
        if other == loaded:
            errs.append(f"linia {fadd+1}: ambii operanzi sunt citirea")
        base = parse(lines[ld])[2][0].strip("[").split("..")[0]
        lo = hi = None
        for m in range(ld - 1, max(ld - 6, -1), -1):
            dd, oo, aa = parse(lines[m])
            if oo == "lea" and "c[0x0][0x730]" in lines[m] and dd and dd[0] == base:
                lo = m
            if oo.startswith("lea.x.hi") and "c[0x0][0x734]" in lines[m]:
                hi = m
        if lo is None or hi is None:
            errs.append(f"linia {ld+1}: citirea nu vine de la adresa din 0x730")
            continue
        if parse(lines[lo])[2][0] != ci:
            errs.append(f"linia {lo+1}: indicele lui res ({parse(lines[lo])[2][0]}) nu e al lui C ({ci})")
        # the value fadd adds to must be the one the plain program stores: rename the fadd's destination away
        out[i] = out[i].replace(val, other) if val != other else out[i]
        drop.update([lo, hi, ld, fadd])
    return [l for n, l in enumerate(out) if n not in drop], stores, errs


def ops(lines):
    return [shape(l) for l in lines if l.strip() and KEEP.search(l)]


def compare(plain_text, res_text):
    plain, res = lines_of(plain_text), lines_of(res_text)
    rest, stores, errs = strip_res(res)
    sp = sum(1 for l in plain if "st.global" in l)
    if stores != sp:
        errs.append(f"scrieri: plain {sp}, res {stores}")
    a, b = ops(plain), ops(rest)
    if a != b:
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        errs.append(f"restul operatiilor difera la {n}: {a[n:n+1]} / {b[n:n+1]} ({len(a)} / {len(b)})")
    return stores, len(a), errs


def main(argv):
    self_test = argv[:1] == ["--self-test"]
    if self_test:
        argv = argv[1:]
    p, r = (open(x, encoding="utf-8").read() for x in argv)
    stores, n, errs = compare(p, r)
    if self_test:
        planted = [
            ("fadd.ftz", "fmul.ftz", "fadd devenita fmul"),
            ("c[0x0][0x730]", "c[0x0][0x738]", "alta adresa pentru res"),
            ("0x3f800000", "0x3f800001", "o constanta"),
        ]
        caught = 0
        for old, new, what in planted:
            if old not in r:
                continue
            i = r.index(old) if old != "fadd.ftz" else r.rindex(old)
            bad = r[:i] + new + r[i + len(old):]
            if compare(p, bad)[2]:
                caught += 1
            else:
                print(f"GRESEALA PLANTATA NEPRINSA: {what}")
                return 1
        m = re.search(r"lea (r\d+) 2 c\[0x0\]\[0x730\]", r)
        bad = r[: m.start(1)] + "r1" + r[m.end(1) :]
        if not compare(p, bad)[2]:
            print("GRESEALA PLANTATA NEPRINSA: res citit la alt indice decat C")
            return 1
        caught += 1
        print(f"self-test: {caught} greseli plantate prinse")
        return 0
    if errs:
        for e in errs:
            print("DIFERENTA:", e)
        return 1
    print(f"{argv[1]}: {stores} / {stores} cozi = a programului simplu + o fadd.ftz cu res (0x730, acelasi indice ca C); restul ({n} operatii) identic, in ordine")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
