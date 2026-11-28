#!/usr/bin/env python3
"""C-168 p.5 P5e (session 295, docs/C-168_P5_LIPIRE_PLAN.md section 15, prediction N3): compare a `-DPF` NAK listing
(`_gelu_pf` / `_res_pf`) with the listing of the same program without `-DPF` (`_gelu` / `_res`).

Checks, exit 1 on any failure:
  1. the floating-point work is the same: every f* / mufu / hmma operation with its constants, in the same order, and
     the same number of C stores (st.global) -> the same arithmetic on the same values, the same bits;
  2. the tail reads (ld.global whose address is built from c[0x0][SRC], SRC = 0x718 for -DGELU (u at the bias slot),
     0x730 for -DRES) are issued ahead: the most reads waiting at once (issued, not yet used) is at least --ahead
     (4 for v8 / v8n / v8r, 8 for v7), while the program without -DPF waits for each read alone (1);
  3. there are between 0 and --extra more such reads than without -DPF (the source reads the last fragment once more,
     at the clamped index; NAK may drop that repeated read, as it does for v8 / v8n / v8r).

    python3 tools/nak/compara_pf.py --src 0x718 --ahead 4 --extra 4 out2/mm_tc_v8n_gelu.sm86.txt out2/mm_tc_v8n_gelu_pf.sm86.txt
    python3 tools/nak/compara_pf.py --self-test --src 0x718 --ahead 4 --extra 4 PLAIN PF
"""
import re
import sys

FP = re.compile(r"=\s*(f[a-z0-9]*|mufu|hmma)\S*\b")


def lines_of(text):
    return [l.split("//")[0].rstrip() for l in text.splitlines()]


def parse(l):
    if "=" in l:
        d, r = l.split("=", 1)
        t = r.split()
        return d.split(), (t[0] if t else ""), t[1:]
    t = l.split()
    return [], (t[0] if t else ""), t[1:]


def shape(l):
    d, op, args = parse(l)
    keep = [a for a in args if not re.match(r"^[\[\-|]*(r|p|ur|up)\d+", a) and a not in ("rZ", "pT", "-rZ")]
    return op + " " + " ".join(keep)


def fp_ops(lines):
    return [shape(l) for l in lines if l.strip() and FP.search(l)]


def tail_reads(lines, src):
    """indices of the ld.global whose address register comes from a lea of c[0x0][src]"""
    addr = {}  # reg -> True if last written by a lea of src
    reads = []
    for i, l in enumerate(lines):
        d, op, a = parse(l)
        if op.startswith("ld.global") and a:
            # [rA..B+rZ] (address built in a register by a lea of c[0x0][src]) or [rA..B+urX..Y] (the base loaded once
            # into a uniform register by an ldc of c[0x0][+src], the index in rA)
            parts = [x.split("..")[0] for x in a[0].strip("[],").split("+")]
            if any(addr.get(x) for x in parts):
                reads.append(i)
        for r in d:
            addr[r] = (op == "lea" and f"c[0x0][{src}]" in l) or (op.startswith("ldc") and f"c[0x0][+{src}]" in l)
    return reads


def max_in_flight(lines, reads):
    """most tail reads issued and not yet used at the same time (straight-line walk of the listing)"""
    pending = {}
    best = 0
    for i, l in enumerate(lines):
        d, op, a = parse(l)
        used = {r for r in pending if any(r == x.strip("[|-").split("..")[0] for x in a)}
        for r in used:
            del pending[r]
        if i in reads:
            pending[d[0]] = i
            best = max(best, len(pending))
        else:
            for r in d:
                pending.pop(r, None)
    return best


def compare(plain_text, pf_text, src, ahead, extra):
    p, q = lines_of(plain_text), lines_of(pf_text)
    errs = []
    a, b = fp_ops(p), fp_ops(q)
    if a != b:
        n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        errs.append(f"operatiile in virgula mobila difera la {n}: {a[n:n+1]} / {b[n:n+1]} ({len(a)} / {len(b)})")
    sp = sum(1 for l in p if "st.global" in l)
    sq = sum(1 for l in q if "st.global" in l)
    if sp != sq:
        errs.append(f"scrieri: fara PF {sp}, cu PF {sq}")
    rp, rq = tail_reads(p, src), tail_reads(q, src)
    if not len(rp) <= len(rq) <= len(rp) + extra:
        errs.append(f"citiri de la {src}: fara PF {len(rp)}, cu PF {len(rq)} (asteptat intre +0 si +{extra})")
    fp, fq = max_in_flight(p, rp), max_in_flight(q, rq)
    if fq < ahead:
        errs.append(f"citiri in asteptare deodata: cu PF {fq} (cerut >= {ahead})")
    return dict(fp=len(a), stores=sq, reads=(len(rp), len(rq)), flight=(fp, fq)), errs


def main(argv):
    self_test = "--self-test" in argv
    argv = [x for x in argv if x != "--self-test"]
    opt = {}
    while argv and argv[0].startswith("--"):
        opt[argv[0][2:]] = argv[1]
        argv = argv[2:]
    src, ahead, extra = opt["src"], int(opt["ahead"]), int(opt["extra"])
    p, q = (open(x, encoding="utf-8").read() for x in argv)
    info, errs = compare(p, q, src, ahead, extra)
    if self_test:
        planted = [
            ("fmul.ftz", "fadd.ftz", "o inmultire devenita adunare"),
            ("0x3f800000", "0x3f800001", "o constanta"),
            (f"c[0x0][+{src}]", "c[0x0][+0x738]", "alta adresa pentru citirea cozii"),
        ]
        planted.insert(1, ("fadd.ftz", "fmul.ftz", "o adunare devenita inmultire"))
        caught = 0
        for old, new, what in planted:
            if old not in q:
                continue
            i = q.rindex(old)
            if not compare(p, q[:i] + new + q[i + len(old):], src, ahead, extra)[1]:
                print(f"GRESEALA PLANTATA NEPRINSA: {what}")
                return 1
            caught += 1
        # the program without PF given as the PF one must fail check 2 / 3
        if not compare(p, p, src, ahead, extra)[1]:
            print("GRESEALA PLANTATA NEPRINSA: programul fara PF trecut drept PF")
            return 1
        caught += 1
        if caught < 3:
            print(f"self-test: doar {caught} greseli plantate")
            return 1
        print(f"self-test: {caught} greseli plantate prinse")
        return 0
    if errs:
        for e in errs:
            print("DIFERENTA:", e)
        return 1
    print(f"{argv[1]}: {info['fp']} operatii in virgula mobila identice, in ordine; {info['stores']} scrieri; citiri de la {src}: "
          f"{info['reads'][0]} -> {info['reads'][1]}; in asteptare deodata {info['flight'][0]} -> {info['flight'][1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
