#!/usr/bin/env python3
"""Prepares the KGR1 control vectors (hub `integrare/kronos/vectori/`, Kronos S39/C-24) for cyber-gpu's tests.

The hub is private; cyber-gpu is public. The outputs go to a LOCAL folder (e.g. `target/vectori-kgr1/`) and are
never committed; the tests read them from `KGR1_VECTORI`. Only what the tests need is written:
- `mic_q8_0.kgr` with the key `tokenizer.bpe` REMOVED (the weights are random, seed 20261028, not a trained model;
  the tokenizer stays in the hub) -> `kronos_mic_q8_0.kgr` (still a valid KGR1 file: same tensors, same data zone);
- from `vectori.json`, the `q8_0` part only: token ids, argmax per position, greedy_5, the top-1/top-2 margin
  -> `kronos_mic_q8_0.txt`, and the last-position logits as little-endian f32 -> `kronos_mic_q8_0_logits.f32`.

Usage: python3 -I tools/vectori-kgr1/pregateste.py <hub>/integrare/kronos/vectori target/vectori-kgr1
       KGR1_VECTORI=$PWD/target/vectori-kgr1 cargo test --release -p nvidia-ga104 --test sim_i14_token kgr1 -- --ignored
The source file's SHA-256 is checked against `vectori.json` first; the outputs are the same bit for bit every run.
"""
import hashlib
import json
import os
import struct
import sys


def main(src, out):
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(f"{src}/vectori.json", encoding="utf-8"))
    raw = open(f"{src}/mic_q8_0.kgr", "rb").read()
    want = meta["fisiere"]["q8_0"]["sha256"]
    if hashlib.sha256(raw).hexdigest() != want:
        sys.exit("mic_q8_0.kgr: suma SHA-256 nu se potriveste cu vectori.json")
    o = 0

    def take(fmt):
        nonlocal o
        v = struct.unpack_from(fmt, raw, o)
        o += struct.calcsize(fmt)
        return v[0]

    def text():
        nonlocal o
        n = take("<Q")
        s = raw[o:o + n]
        o += n
        return s

    assert raw[:4] == b"KGR1"
    o = 4
    ver, nkv, nt = take("<I"), take("<Q"), take("<Q")
    kvs = []
    for _ in range(nkv):
        k, kind = text(), take("<I")
        start = o
        if kind in (1, 2):
            o += 8
        else:
            text()
        kvs.append((k, kind, raw[start:o]))
    tensors = []
    for _ in range(nt):
        start = o
        text()
        nd = take("<I")
        o += 8 * nd + 4 + 8
        tensors.append(raw[start:o])
    data = raw[(o + 31) // 32 * 32:]

    kept = [kv for kv in kvs if kv[0] != b"tokenizer.bpe"]
    assert len(kept) == len(kvs) - 1
    head = b"KGR1" + struct.pack("<IQQ", ver, len(kept), nt)
    for k, kind, val in kept:
        head += struct.pack("<Q", len(k)) + k + struct.pack("<I", kind) + val
    head += b"".join(tensors)
    head += b"\0" * (-len(head) % 32)
    open(f"{out}/kronos_mic_q8_0.kgr", "wb").write(head + data)

    lines = [
        "# vectorii de control KGR1 Q8_0 (hub integrare/kronos/vectori, S39/C-24), partea q8_0; facut de tools/vectori-kgr1/pregateste.py",
        f"sursa_sha256 {want}",
    ]
    logits = b""
    for t in meta["q8_0"]:
        lines.append(f"text {t['text']}")
        for key, name in (("simboluri", "simboluri"), ("argmax_pe_pozitii", "argmax"), ("greedy_5", "greedy")):
            lines.append(name + " " + " ".join(str(v) for v in t[key]))
        lines.append(f"marja {t['marja_minima_greedy']}")
        logits += b"".join(struct.pack("<f", v) for v in t["logits_ultima_pozitie"])
    open(f"{out}/kronos_mic_q8_0.txt", "w", encoding="utf-8", newline="\n").write("\n".join(lines) + "\n")
    open(f"{out}/kronos_mic_q8_0_logits.f32", "wb").write(logits)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
