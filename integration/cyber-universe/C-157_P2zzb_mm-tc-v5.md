# C-157 p.2 — `mm_tc_v5`: `mm_tc_v4` cu 4 blocuri pe SM (gata pentru placă, testat doar în SIMULARE)

Sesiunea Cyber-GPU 258, 26 nov 2026, în cloud, fără placă. Detalii și predicții: `docs/C-157_MM_TC_V5_PLAN.md`.

## Ce e
Același program ca `mm_tc_v4` (aceeași lansare, același cbuf, 128 de fire, plăci 64 × 64, aceeași regulă `fits`, **aceiași biți**),
dar blocul cere **12 KiB** de memorie comună în loc de 24: rezultatul iese prin câte o zonă mică de 1 KiB pe grup de fire (finalul lui
`mm_tc128`, care a dat aceiași biți pe placă în os-423), nu printr-o placă întreagă de 16 KiB. La treapta 64 KiB intră **4 blocuri pe SM**
în loc de 2. Pasul #513 e 79 % înmulțiri pe `mm_tc_v4`, care azi merge cu 8,3 TFLOP/s (17 % din placă), încetinit de așteptare.

| | |
|---|---|
| Binarul | `tools/nak/out2/mm_tc_v5.sm86.sass`, SHA-256 `f34f4f934135b39d4eccb59ef969688864d4bf78938a32eaf23eca594aee23d6` |
| Rust | `nvidia_ga104::mm_tc_v5::{MM_TC_V5, MM_TC_V5_SHA256, REGISTERS (128), BLOCK, SHARED (12 288 B), grid, fits}` — `Args` / `cbuf` / `grid` = ale lui `mm_tc` |
| Simulare | `gpu_sim::cmdproc::{mm_tc_v5_on_cpu, MM_TC_V5_NEEDS}` (refuză produsele nealiniate, ca v4) |
| QMD | ca `mm_tc_v4` (registre 128, treapta 64 KiB), **dar memoria comună = `mm_tc_v5::SHARED` (12 KiB)** |

**Înainte de lansare: `mm_tc_v5::fits(&args)`** (= `mm_tc_v4::fits`). Dacă zice nu → `mm_tc_pf`.

## Ce cerem OS-ului (când are placa)
1. Încă un rând în proba `--mm-tc-64k`, **după `mm_tc_v4`**: aceleași seturi aliniate + qkv-513, cu `mm_tc_v5`.
   - **Așteptat:** aceiași biți ca `mm_tc` pe toate seturile aliniate, gărzile neatinse.
   - **Viteza prezisă (DEDUS, scrisă înainte):** qkv-513 **1,59–1,94 ms** (azi v4 2,62 ms; ×1,35–1,65) = **11,2–13,7 TFLOP/s**.
     Sub ×1,2 (peste 2,19 ms) → ocuparea nu e frâna și rămâne `mm_tc_v4`.
2. Dacă (1) trece: pasul #513 lot 6 cu `mm_tc_v5` în locul lui `mm_tc_v4` (în `pack_mixed_fast`, aceleași 147 de lansări, când `fits`).
   - **Așteptat:** pierderile și normele identice cu os-468 (aceiași biți); înmulțirile 216,6 → **131–160 ms**; pasul 273 → **188–217 ms**
     (~14 200–16 400 simb/s față de 11 256). Cu `--tablou-viteza`, ca să se vadă și frecvența plăcii.

## Ce e nou pentru placă
Doar programul `mm_tc_v5` (SHA de mai sus). Nicio scriere în registre, niciun firmware, nicio schimbare la `mm_tc` / `mm_tc_pf` /
`mm_tc_v4`. Nicio schimbare incompatibilă (doar adăugări).
