#!/usr/bin/env python3
"""Generate Rust register constants from NVIDIA's published GA102 headers (MIT).

Host-only tool (C-20 B). Reads the `#define` lines of the headers listed in
HEADERS from a checkout of NVIDIA open-gpu-kernel-modules at tag 570.144 and
writes `crates/nvidia-rm570/src/regs.rs`. Standard library only.

Only four shapes are translated; everything else is listed as skipped in the
output so nothing is silently dropped:
  * `0x...` / decimal          -> pub const NAME: u32
  * `hi:lo`                    -> pub const NAME: Field
  * `0xHI:0xLO` (device range)  -> pub const NAME: Range
  * `(0xBASE+(i)*STRIDE)`      -> pub const fn NAME(i: u32) -> u32
  * another NAME / NAME(n)     -> resolved alias (u32)

Usage: gen_regs.py <ogkm-checkout> <output.rs>
"""
import re
import sys

HEADERS = [
    ("ga102_dev_boot", "src/common/inc/swref/published/ampere/ga102/dev_boot.h"),
    ("ga102_dev_falcon_v4", "src/common/inc/swref/published/ampere/ga102/dev_falcon_v4.h"),
    ("ga102_dev_falcon_v4_addendum", "src/common/inc/swref/published/ampere/ga102/dev_falcon_v4_addendum.h"),
    ("ga102_dev_falcon_second_pri", "src/common/inc/swref/published/ampere/ga102/dev_falcon_second_pri.h"),
    ("ga102_dev_fbif_v4", "src/common/inc/swref/published/ampere/ga102/dev_fbif_v4.h"),
    ("ga102_dev_gsp", "src/common/inc/swref/published/ampere/ga102/dev_gsp.h"),
    ("ga102_dev_gsp_addendum", "src/common/inc/swref/published/ampere/ga102/dev_gsp_addendum.h"),
    ("ga102_dev_sec_pri", "src/common/inc/swref/published/ampere/ga102/dev_sec_pri.h"),
    ("ga102_dev_sec_addendum", "src/common/inc/swref/published/ampere/ga102/dev_sec_addendum.h"),
    ("ga102_dev_riscv_pri", "src/common/inc/swref/published/ampere/ga102/dev_riscv_pri.h"),
    ("ga102_dev_gc6_island", "src/common/inc/swref/published/ampere/ga102/dev_gc6_island.h"),
    ("ga102_dev_gc6_island_addendum", "src/common/inc/swref/published/ampere/ga102/dev_gc6_island_addendum.h"),
    ("tu102_dev_fb", "src/common/inc/swref/published/turing/tu102/dev_fb.h"),
]

DEFINE = re.compile(r"^#define\s+([A-Za-z_][A-Za-z0-9_]*)(\([a-z]\))?\s+(.*?)\s*(/\*\s*(.*?)\s*\*/)?\s*$")
NUM = re.compile(r"^(0x[0-9A-Fa-f]+|[0-9]+)[uU]?$")
FIELD = re.compile(r"^([0-9]+):([0-9]+)$")
RANGE = re.compile(r"^(0x[0-9A-Fa-f]+):(0x[0-9A-Fa-f]+)$")
INDEXED = re.compile(r"^\((0x[0-9A-Fa-f]+|[0-9]+)\+\(([a-z])\)\*(0x[0-9A-Fa-f]+|[0-9]+)\)$")
ALIAS = re.compile(r"^\(?([A-Z][A-Z0-9_]*)(\(([0-9]+)\))?\)?$")


def parse(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = DEFINE.match(line.rstrip("\n"))
            if m:
                out.append((m.group(1), m.group(2), m.group(3), (m.group(5) or "").strip()))
    return out


def main(root, dst):
    plain = {}  # name -> int (global, for alias resolution)
    indexed = {}  # name -> (base, stride)
    modules = []
    for mod, rel in HEADERS:
        items, skipped, seen = [], [], {}
        for name, param, value, access in parse(f"{root}/{rel}"):
            if name.startswith("__") or value == "":
                continue
            # C allows an identical redefinition; keep one, refuse a different one.
            if name in seen:
                if seen[name] != (param, value):
                    sys.exit(f"{rel}: {name} redefined with a different value")
                continue
            seen[name] = (param, value)
            if param is None and NUM.match(value):
                v = int(NUM.match(value).group(1), 0)
                plain[name] = v
                items.append(("const", name, v, access))
            elif param is None and FIELD.match(value):
                hi, lo = map(int, FIELD.match(value).groups())
                items.append(("field", name, (hi, lo), access))
            elif param is None and RANGE.match(value):
                hi, lo = (int(x, 0) for x in RANGE.match(value).groups())
                items.append(("range", name, (hi, lo), access))
            elif param is not None and INDEXED.match(value):
                b, var, s = INDEXED.match(value).groups()
                if "(" + var + ")" != param:
                    skipped.append((name, value))
                    continue
                indexed[name] = (int(b, 0), int(s, 0))
                items.append(("fn", name, indexed[name], access))
            elif param is None and ALIAS.match(value):
                a = ALIAS.match(value)
                tgt, idx = a.group(1), a.group(3)
                if idx is None and tgt in plain:
                    v = plain[tgt]
                elif idx is not None and tgt in indexed:
                    v = indexed[tgt][0] + int(idx) * indexed[tgt][1]
                else:
                    skipped.append((name, value))
                    continue
                plain[name] = v
                items.append(("const", name, v, f"alias of {value}"))
            else:
                skipped.append((name, value))
        modules.append((mod, rel, items, skipped))

    w = []
    w.append("// GENERATED by tools/gen-rm570/gen_regs.py — do not edit by hand.")
    w.append("// Source: NVIDIA open-gpu-kernel-modules, tag 570.144 (see crate README for the commit).")
    w.append("// The source headers are MIT licensed, Copyright NVIDIA CORPORATION & AFFILIATES;")
    w.append("// the full notice is in crates/nvidia-rm570/LICENSE-NVIDIA.")
    w.append("//")
    w.append("// Access codes in the doc comments are NVIDIA's (first letter R = readable,")
    w.append("// second W = writable). Nothing here performs any register access.")
    w.append("")
    w.append("use crate::{Field, Range};")
    for mod, rel, items, skipped in modules:
        w.append("")
        w.append(f"/// From `{rel}` (570.144).")
        w.append("#[allow(non_snake_case, missing_docs, clippy::identity_op)]")
        w.append(f"pub mod {mod} {{")
        if any(k == "field" for k, *_ in items):
            w.append("    use super::Field;")
        if any(k == "range" for k, *_ in items):
            w.append("    use super::Range;")
        for kind, name, v, access in items:
            doc = f"    /// `{access}`" if access else None
            if doc:
                w.append(doc)
            if kind == "const":
                ty = "u32" if v <= 0xFFFF_FFFF else "u64"
                w.append(f"    pub const {name}: {ty} = {v:#x};")
            elif kind == "field":
                w.append(f"    pub const {name}: Field = Field::new({v[0]}, {v[1]});")
            elif kind == "range":
                w.append(f"    pub const {name}: Range = Range::new({v[1]:#x}, {v[0]:#x});")
            else:
                w.append(f"    pub const fn {name}(i: u32) -> u32 {{")
                w.append(f"        {v[0]:#x} + i * {v[1]:#x}")
                w.append("    }")
        for name, value in skipped:
            w.append(f"    // skipped (shape not translated): {name} = {value}")
        w.append("}")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(w) + "\n")
    n = sum(len(i) for _, _, i, _ in modules)
    s = sum(len(k) for _, _, _, k in modules)
    print(f"regs: {n} definitions from {len(modules)} headers, {s} skipped (listed in the file)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
