# C-157 p.2 — `mm_tc_v7`: `mm_tc_v6` cu rânduri umplute în memoria comună (gata pentru probă, testat doar în SIMULARE)

Sesiunea Cyber-GPU 266, 27 nov 2026, în cloud, fără placă. Socoteala și predicțiile (scrise înainte de cod):
`docs/C-157_CITIRI_CONFLICTE.md` (§3–§4), rezultatul compilării și al simulării: același fișier, §6.

## Ce e
Același program ca `mm_tc_v6` (aceeași lansare, același cbuf, 128 de fire, plăci 64 × 64, aceeași regulă `fits`, **aceiași biți**),
dar rândurile feliilor din memoria comună au **16 octeți de umplutură** (80 / 144 B în loc de 64 / 128). Socoteala (T279): în v6
citirile `coopMatLoad` se loveau în aceleași bănci (8 la 1 și 4 la 1); în v7 sunt fără conflict pe orice steag. Memoria comună crește
de la 12 288 la **14 336 B**, tot 4 blocuri pe SM la treapta de 64 KiB.

| | |
|---|---|
| Binarul | `tools/nak/out2/mm_tc_v7.sm86.sass`, SHA-256 `953ed6b85fb0022183ef0e5b4b3262cd3f3742c43dcea2397ed9e2d285e1cf6a` |
| Rust | `nvidia_ga104::mm_tc_v7::{MM_TC_V7, MM_TC_V7_SHA256, REGISTERS (128), BLOCK, SHARED (14 336 B), grid, fits}` — `Args` / `cbuf` / `grid` = ale lui `mm_tc` |
| Simulare | `gpu_sim::cmdproc::{mm_tc_v7_on_cpu, MM_TC_V7_NEEDS}` (refuză produsele nealiniate, ca v6) |
| QMD | ca `mm_tc_v6` (registre 128, treapta 64 KiB), dar cu `mm_tc_v7::SHARED` (14 336 B) — **nu** refolosiți `SHARED`-ul lui v6 |

**Înainte de lansare: `mm_tc_v7::fits(&args)`** (= `mm_tc_v4::fits`). Dacă zice nu → `mm_tc_pf`.

## Ce cerem OS-ului (când are placa; după proba v6 din P2zzg / P2zzh, nu în locul ei)
1. Încă un rând în proba `--mm-tc` (cu `--mm-tc-64k`), **după `mm_tc_v6`**: aceleași seturi aliniate + `qkv-513`, `qkv-513-dx`,
   `qkv-513-dw`, cu `mm_tc_v7` (de exemplu opțiunea `--mm-tc-v7`).
   - **Așteptat (W4):** aceiași biți ca `mm_tc` pe toate seturile aliniate, gărzile neatinse.
   - **Viteza prezisă (DEDUS, scrisă înainte de cod), v7 / v6:** `qkv-513` (`TB`) **×1,00–1,20**; `qkv-513-dx` (fără steag)
     **×1,05–1,40**; `qkv-513-dw` (`TA`) **×1,10–1,60**; ordinea câștigurilor dW > dX > înainte.
   - Rândul propus: `test-placa --tablou-viteza --mm-tc --mm-tc-v4 --mm-tc-v5 --mm-tc-v6 --mm-tc-v7 --mm-tc-64k`.
2. Legarea în pas o pregătim noi abia după ce proba v6 / v7 a rulat (ca la v5 / v6: o alegere `TcPick` nouă, doar adăugări).

## Ce e nou pentru placă
Doar programul `mm_tc_v7` (SHA de mai sus), sub opțiune oprită implicit. Nicio scriere în registre, niciun firmware, nicio schimbare
la `mm_tc` / `mm_tc_pf` / `mm_tc_v4` / `mm_tc_v5` / `mm_tc_v6`. Nicio schimbare incompatibilă (doar adăugări).
