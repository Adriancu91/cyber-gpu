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
  * in ENTRY_MODULES (I9, page-table entries): `a:b` with small sums such as
    `(35-3):8`                 -> pub const NAME: EntryField (bits of a u128)

Second source (C-20 B, methods and QMD): NVIDIA open-gpu-doc (MIT), for the
channel class C56F, the compute class C7C0 and the QMD V03_00 layout:
  * `MW(hi:lo)`                -> pub const NAME: QmdField (bit positions)
  * `MW((a+(i)*s):(b+(i)*s))`  -> pub const fn NAME(i: u32) -> QmdField

Usage: gen_regs.py <ogkm-checkout> <open-gpu-doc-checkout> <output.rs>
"""
import re
import sys

# (module, source key, path, only names starting with this prefix — or one of
#  a tuple of prefixes — or None)
HEADERS = [
    ("ga102_dev_boot", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_boot.h", None),
    ("ga102_dev_falcon_v4", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_falcon_v4.h", None),
    ("ga102_dev_falcon_v4_addendum", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_falcon_v4_addendum.h", None),
    ("ga102_dev_falcon_second_pri", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_falcon_second_pri.h", None),
    ("ga102_dev_fbif_v4", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_fbif_v4.h", None),
    ("ga102_dev_gsp", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_gsp.h", None),
    ("ga102_dev_gsp_addendum", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_gsp_addendum.h", None),
    ("ga102_dev_sec_pri", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_sec_pri.h", None),
    ("ga102_dev_sec_addendum", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_sec_addendum.h", None),
    ("ga102_dev_riscv_pri", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_riscv_pri.h", None),
    ("ga102_dev_gc6_island", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_gc6_island.h", None),
    ("ga102_dev_gc6_island_addendum", "ogkm", "src/common/inc/swref/published/ampere/ga102/dev_gc6_island_addendum.h", None),
    ("tu102_dev_fb", "ogkm", "src/common/inc/swref/published/turing/tu102/dev_fb.h", None),
    ("tu102_dev_bus", "ogkm", "src/common/inc/swref/published/turing/tu102/dev_bus.h", None),
    # I9: the GPU page-table entry formats (MMU version 2, used by Pascal..Ampere).
    # The published swref copy of dev_mmu.h is a stub; the full one ships with UVM.
    ("tu102_dev_mmu_ver2", "ogkm", "kernel-open/nvidia-uvm/hwref/turing/tu102/dev_mmu.h", "NV_MMU_VER2_"),
    # I9b: writing VRAM through the BAR0 window (PRAMIN) and the TLB invalidate.
    # NV_PBUS_BAR0_WINDOW is published only in the Maxwell header; RM 570.144
    # uses that layout (kbusSetBAR0WindowVidOffset_GM107) on TU10x..AD10x, GA104 included.
    ("tu102_dev_ram", "ogkm", "src/common/inc/swref/published/turing/tu102/dev_ram.h", "NV_PRAMIN"),
    ("gm107_dev_bus", "ogkm", "src/common/inc/swref/published/maxwell/gm107/dev_bus.h", "NV_PBUS_BAR0_WINDOW"),
    ("tu102_dev_vm", "ogkm", "src/common/inc/swref/published/turing/tu102/dev_vm.h",
     ("NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET", "NV_VIRTUAL_FUNCTION_PRIV_MMU_INVALIDATE")),
    ("clc56f", "ogd", "classes/host/clc56f.h", None),
    ("clc7c0", "ogd", "classes/compute/clc7c0.h", None),
    ("qmd_v03_00", "ogd", "classes/compute/clc6c0qmd.h", "NVC6C0_QMDV03_00_"),
]

# Modules whose bit fields describe page-table entries (64 or 128 bits wide,
# not 32-bit registers): every `hi:lo` there becomes an `EntryField`, and the
# two ends may be small sums such as `(35-3):8` or `99:(100-3)`.
ENTRY_MODULES = {"tu102_dev_mmu_ver2"}

DEFINE = re.compile(r"^#define\s+([A-Za-z_][A-Za-z0-9_]*)(\([a-z]\))?\s+(.*?)\s*(/\*\s*(.*?)\s*\*/)?\s*$")
NUM = re.compile(r"^\(?(0x[0-9A-Fa-f]+|[0-9]+)[uU]?\)?$")
MW = re.compile(r"^MW\(([0-9]+):([0-9]+)\)$")
MW_I = re.compile(r"^MW\(\(([0-9]+)\+\(i\)\*([0-9]+)\):\(([0-9]+)\+\(i\)\*([0-9]+)\)\)$")
FIELD = re.compile(r"^([0-9]+):([0-9]+)$")
RANGE = re.compile(r"^(0x[0-9A-Fa-f]+):(0x[0-9A-Fa-f]+)$")
INDEXED = re.compile(r"^\((0x[0-9A-Fa-f]+|[0-9]+)\+\(([a-z])\)\*(0x[0-9A-Fa-f]+|[0-9]+)\)$")
SUM = r"\(?[0-9]+(?:[-+][0-9]+)*\)?"
EFIELD = re.compile(r"^(" + SUM + r"):(" + SUM + r")$")


def eval_sum(text):
    """Value of `n`, `(n)`, `(n-m)`, `(n+m-k)`: digits, + and - only."""
    terms = re.findall(r"([-+]?)([0-9]+)", text.strip("()"))
    return sum(-int(v) if sign == "-" else int(v) for sign, v in terms)


ALIAS = re.compile(r"^\(?([A-Z][A-Z0-9_]*)(\(([0-9]+)\))?\)?$")


def parse(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            m = DEFINE.match(line.rstrip("\n"))
            if m:
                value = m.group(3).split("//")[0].strip()  # drop C++ line comments
                out.append((m.group(1), m.group(2), value, (m.group(5) or "").strip()))
    return out


def main(roots, dst):
    plain = {}  # name -> int (global, for alias resolution)
    indexed = {}  # name -> (base, stride)
    modules = []
    for mod, key, rel, prefix in HEADERS:
        root = roots[key]
        items, skipped, seen = [], [], {}
        for name, param, value, access in parse(f"{root}/{rel}"):
            if name.startswith("__") or value == "":
                continue
            if prefix is not None and not name.startswith(prefix):
                continue
            # C allows an identical redefinition; keep one, refuse a different one.
            if name in seen:
                if seen[name] != (param, value):
                    sys.exit(f"{rel}: {name} redefined with a different value")
                continue
            seen[name] = (param, value)
            if param is None and MW.match(value):
                hi, lo = map(int, MW.match(value).groups())
                items.append(("mw", name, (hi, lo), access))
            elif param == "(i)" and MW_I.match(value):
                a, sa, b, sb = map(int, MW_I.match(value).groups())
                if sa != sb:
                    skipped.append((name, value))
                    continue
                items.append(("mwfn", name, (a, b, sa), access))
            elif param is None and NUM.match(value):
                v = int(NUM.match(value).group(1), 0)
                plain[name] = v
                items.append(("const", name, v, access))
            elif mod in ENTRY_MODULES and param is None and EFIELD.match(value):
                hi, lo = (eval_sum(x) for x in EFIELD.match(value).groups())
                if not (0 <= lo <= hi < 128):
                    sys.exit(f"{rel}: {name} = {value} is not a field of a 128-bit entry")
                items.append(("efield", name, (hi, lo), access))
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
        modules.append((mod, key, rel, items, skipped))

    w = []
    w.append("// GENERATED by tools/gen-rm570/gen_regs.py — do not edit by hand.")
    w.append("// Sources: NVIDIA open-gpu-kernel-modules tag 570.144 and NVIDIA open-gpu-doc")
    w.append("// (commits pinned in tools/gen-rm570/generate.sh).")
    w.append("// The source headers are MIT licensed, Copyright NVIDIA CORPORATION & AFFILIATES;")
    w.append("// the full notice is in crates/nvidia-rm570/LICENSE-NVIDIA.")
    w.append("//")
    w.append("// Access codes in the doc comments are NVIDIA's (first letter R = readable,")
    w.append("// second W = writable). Nothing here performs any register access.")
    w.append("")
    w.append("use crate::{EntryField, Field, QmdField, Range};")
    for mod, key, rel, items, skipped in modules:
        w.append("")
        src = "open-gpu-kernel-modules 570.144" if key == "ogkm" else "open-gpu-doc"
        w.append(f"/// From `{rel}` ({src}).")
        w.append("#[allow(non_snake_case, missing_docs, clippy::identity_op)]")
        w.append(f"pub mod {mod} {{")
        if any(k == "field" for k, *_ in items):
            w.append("    use super::Field;")
        if any(k == "efield" for k, *_ in items):
            w.append("    use super::EntryField;")
        if any(k == "range" for k, *_ in items):
            w.append("    use super::Range;")
        if any(k in ("mw", "mwfn") for k, *_ in items):
            w.append("    use super::QmdField;")
        for kind, name, v, access in items:
            doc = f"    /// `{access}`" if access else None
            if doc:
                w.append(doc)
            if kind == "const":
                ty = "u32" if v <= 0xFFFF_FFFF else "u64"
                w.append(f"    pub const {name}: {ty} = {v:#x};")
            elif kind == "field":
                w.append(f"    pub const {name}: Field = Field::new({v[0]}, {v[1]});")
            elif kind == "efield":
                w.append(f"    pub const {name}: EntryField = EntryField::new({v[0]}, {v[1]});")
            elif kind == "mw":
                w.append(f"    pub const {name}: QmdField = QmdField::new({v[0]}, {v[1]});")
            elif kind == "mwfn":
                w.append(f"    pub const fn {name}(i: u32) -> QmdField {{")
                w.append(f"        QmdField::new({v[0]} + i * {v[2]}, {v[1]} + i * {v[2]})")
                w.append("    }")
            elif kind == "range":
                w.append(f"    pub const {name}: Range = Range::new({v[1]:#x}, {v[0]:#x});")
            else:
                w.append(f"    pub const fn {name}(i: u32) -> u32 {{")
                w.append(f"        {v[0]:#x} + i * {v[1]:#x}")
                w.append("    }")
        mws = [name for k, name, *_ in items if k == "mw"]
        if mws:
            w.append("    /// Every fixed-position field above (for whole-layout tests).")
            w.append("    pub const ALL_FIELDS: &[(&str, QmdField)] = &[")
            for name in mws:
                w.append(f'        ("{name}", {name}),')
            w.append("    ];")
        for name, value in skipped:
            w.append(f"    // skipped (shape not translated): {name} = {value}")
        w.append("}")
    with open(dst, "w", encoding="utf-8") as f:
        f.write("\n".join(w) + "\n")
    n = sum(len(i) for _, _, _, i, _ in modules)
    s = sum(len(k) for _, _, _, _, k in modules)
    print(f"regs: {n} definitions from {len(modules)} headers, {s} skipped (listed in the file)")


if __name__ == "__main__":
    main({"ogkm": sys.argv[1], "ogd": sys.argv[2]}, sys.argv[3])
