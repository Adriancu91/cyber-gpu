# C-168 p.3 — `mm_tc_v8` / `mm_tc_v8n`: înmulțirea v8 (gata pentru probă, testată doar în SIMULARE)

Sesiunea Cyber-GPU 279, 27 nov 2026, în cloud, fără placă. Planul și predicțiile (scrise înainte de cod): `docs/C-168_V8_PLAN.md`
§1–§4; ce a ieșit la compilare și în simulare: același fișier, §7. Testul: T295 (`crates/nvidia-ga104/tests/sim_c168_mm_tc_v8.rs`).

## Ce e
Două programe din aceeași sursă (`tools/nak/mm_tc_v8.comp`), **aceeași treabă, aceleași 48 B de argumente (cbuf ca `mm_tc`,
argumentele la 0x700), aceeași regulă `fits` ca v4–v7 și aceiași biți ca `mm_tc`**. Diferă blocul, numărul de fire și memoria comună:

| | `mm_tc_v8` | `mm_tc_v8n` |
|---|---|---|
| binar | `tools/nak/out2/mm_tc_v8.sm86.sass` | `tools/nak/out2/mm_tc_v8n.sm86.sass` |
| SHA-256 | `1b910a5392cb5f9a67ee3ad340ef7dea94e5a56630ac77bcf0903d8b0f8c4b29` | `9a0702bccfe2d82e699a99ee22096fb1a8b8844a130b559aae136f9c7ef20aae` |
| bucata din C pe bloc | 128 × 128 | 128 rânduri × 64 coloane |
| fire pe bloc (`BLOCK` / `BLOCK_N`) | **256** × 1 × 1 | 128 × 1 × 1 |
| grila | `mm_tc_v8::grid` = (⌈n/128⌉, ⌈m/128⌉, 1) | `mm_tc_v8::grid_n` = (⌈n/64⌉, ⌈m/128⌉, 1) |
| registre (QMD) | `REGISTERS` = 128 | `REGISTERS_N` = 144 |
| memoria comună (QMD) | `SHARED` = 24 576 B, 1 barieră | `SHARED_N` = 18 432 B, 1 barieră |
| blocuri pe SM la treapta 64 KiB | 2 | 3 |

Rust (doar adăugări): `nvidia_ga104::mm_tc_v8::{MM_TC_V8, MM_TC_V8N, *_SHA256, REGISTERS, REGISTERS_N, BLOCK, BLOCK_N, SHARED,
SHARED_N, grid, grid_n, fits}`; simulare: `gpu_sim::cmdproc::{mm_tc_v8_on_cpu, mm_tc_v8n_on_cpu, MM_TC_V8_NEEDS, MM_TC_V8N_NEEDS}`.

**Atenție la QMD:** fiecare program cu **propriile** `REGISTERS` / `SHARED` și propriul număr de fire — nu refolosiți valorile lui
v7 (128 fire, 14 336 B). Înainte de lansare: `mm_tc_v8::fits(&args)`; dacă zice nu → `mm_tc_pf`.

## Ce cerem OS-ului (când are placa)
1. Încă două rânduri în proba `--mm-tc` (cu `--mm-tc-64k` și `--mm-tc-dxdw`), după `mm_tc_v7`: aceleași seturi aliniate +
   `qkv-513`, `qkv-513-dx`, `qkv-513-dw`, o dată cu `mm_tc_v8` și o dată cu `mm_tc_v8n` (de exemplu opțiunea `--mm-tc-v8`, care le
   rulează pe amândouă). Rândul propus:
   `test-placa --tablou-viteza --mm-tc --mm-tc-v7 --mm-tc-v8 --mm-tc-64k --mm-tc-dxdw`
2. **Așteptat (scris înainte de cod, `docs/C-168_V8_PLAN.md` §4):**
   - **G4:** aceiași biți ca `mm_tc` pe toate seturile aliniate, gărzile neatinse — ambele programe.
   - **G5:** `qkv-513` (6 144 × 2 304 × 768, v7 la 64 KiB: 1,196 ms) → v8 **0,75–0,96 ms** (22,7–29,1 TFLOP/s).
   - **G6:** `qkv-513-dx` (v7 1,147 ms) → v8 **0,72–0,92 ms**; `qkv-513-dw` (v7 1,464 ms) → v8n **1,02–1,30 ms**, v8 mai lent pe dW
     (1,22–1,56: ultimul val de blocuri).
   - **Prag scris:** dacă v8 rămâne sub ×1,15 față de v7 la `qkv-513`, frâna nu e la citirile din memoria comună.
3. Legarea în pas (alegerea v7 / v8 / v8n pe formă, `TcPick`) o pregătim noi; rulați-o abia după proba de mai sus.

## Ce e nou pentru placă
Doar cele două programe (SHA de mai sus), sub opțiune oprită implicit (`mm_tc_v8` are 256 de fire pe bloc, ca `mm_tc128`). Nicio scriere în registre, niciun firmware, nicio schimbare la
`mm_tc` … `mm_tc_v7`. Nicio schimbare incompatibilă (doar adăugări).
