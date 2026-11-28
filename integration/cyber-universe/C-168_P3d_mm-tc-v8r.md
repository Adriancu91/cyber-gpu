# C-168 p.3 urmarea — `mm_tc_v8r`: v8 cu mai puține copii de registre (gata pentru probă, testat doar în SIMULARE)

Sesiunea Cyber-GPU 287, 28 nov 2026, în cloud, fără placă. De ce există și ce prezicem: `docs/C-168_V8R.md` (§1–§3, scrise înainte
de placă). Testele: T302 (numărătoarea pe listare) și T303 (`crates/nvidia-ga104/tests/sim_c168_mm_tc_v8r.rs`, SIMULARE).

## Ce e
`mm_tc_v8` cu o singură schimbare: în fiecare felie de 16 citește întâi cele 4 bucăți B, apoi câte o bucată A cu cele 4 `hmma` ale ei.
**Aceeași treabă, aceleași 48 B de argumente (cbuf ca `mm_tc`, argumentele la 0x700), aceeași regulă `fits`, aceeași grilă și aceiași
biți ca `mm_tc_v8` (= `mm_tc`).** Diferă doar binarul și **numărul de registre din QMD (120, nu 128)**.

| | `mm_tc_v8` | **`mm_tc_v8r`** |
|---|---|---|
| binar | `tools/nak/out2/mm_tc_v8.sm86.sass` | `tools/nak/out2/mm_tc_v8r.sm86.sass` |
| SHA-256 | `1b910a53…8c4b29` | `6e28d531fc104ec1b37edb0f2303337c81eef66a4a82b5ad02861add18274bf4` |
| bucata din C pe bloc / fire | 128 × 128 / 256 | la fel (`mm_tc_v8r::BLOCK`) |
| grila | `mm_tc_v8::grid` | `mm_tc_v8r::grid` (aceeași) |
| registre (QMD) | 128 | **`mm_tc_v8r::REGISTERS` = 120** |
| memoria comună (QMD) | 24 576 B, 1 barieră | `mm_tc_v8r::SHARED` (aceeași) |
| blocuri pe SM la 64 KiB | 2 | 2 |

Rust (doar adăugări): `nvidia_ga104::mm_tc_v8r::{MM_TC_V8R, MM_TC_V8R_SHA256, REGISTERS, BLOCK, BLOCK_M, BLOCK_NCOLS, SHARED,
BLOCKS_PER_SM, CBUF_BYTES, fits, grid}`; simulare: `gpu_sim::cmdproc::{mm_tc_v8r_on_cpu, MM_TC_V8R_NEEDS}`.

## Ce cerem OS-ului (când are placa)
1. Încă un rând în proba `--mm-tc-v8`: `mm_tc_v8r`, pe aceleași seturi aliniate + `qkv-513`, `qkv-513-dx`, `qkv-513-dw`, **în același
   ciclu cu v8 și v8n** (ca să comparăm pe aceeași placă, la aceeași căldură). De exemplu opțiunea `--mm-tc-v8r`, care adaugă v8r
   lângă v8 / v8n. Rândul propus:
   `test-placa --tablou-viteza --mm-tc --mm-tc-v7 --mm-tc-v8 --mm-tc-v8r --mm-tc-64k --mm-tc-dxdw`
2. **Așteptat (scris înainte, `docs/C-168_V8R.md` §3):**
   - **V1:** aceiași biți ca `mm_tc` (și ca v8) pe toate seturile aliniate, gărzile neatinse.
   - **V2:** `qkv-513` @64k → v8r **0,85–1,01 ms** (v8 1,063, v8n 1,012 în os-500).
   - **V3:** `qkv-513-dx` @64k → v8r **0,83–0,99 ms** (v8 1,037, v8n 0,929).
   - **V4:** `qkv-513-dw` @64k → v8r **1,40–1,60 ms** (v8 1,650).
   - **Cum citim:** v8r ≥ ×1,05 mai rapid ca v8 pe qkv și dX → copiile erau frâna lui v8; sub ×1,03 → nu contau.
3. În pas **nu** se pune încă: abia după V1 pe placă, iar alegerea v7 / v8 / v8n / v8r o pregătim noi după cifre.

## Ce e nou pentru placă
Doar programul `mm_tc_v8r` (SHA de mai sus), sub opțiune oprită implicit. Nicio scriere în registre, niciun firmware, nicio schimbare
la `mm_tc` … `mm_tc_v8n`. Nicio schimbare incompatibilă (doar adăugări). **Atenție la QMD:** 120 de registre (nu 128 ca v8).
