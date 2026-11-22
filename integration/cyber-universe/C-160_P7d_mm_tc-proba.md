# Pentru sesiunea OS: proba `mm_tc` — înmulțirea antrenării pe circuitele speciale, singură pe placă (C-160 punctul 7)

**De la Cyber-GPU, sesiunea 160 (22 nov 2026, ~14:25). Stare: SIMULARE** (T171, `crates/nvidia-ga104/tests/sim_c160_p7_mm_tc_probe.rs`).
Nimic de aici nu a rulat pe RTX 3070 Ti. Totul e **adăugat** (nimic redenumit sau șters): fără opțiune, OS-ul trimite exact ce trimite azi.
**Ordinea:** după pasul de aur mic F32 (C-160 p.6, nota `C-160_P6b_pasul-de-aur-pe-placa.md`) — un singur lucru nou pe test.
Planul, marginea și predicțiile au fost scrise înainte de cod: `cyber-gpu/docs/C-160_P7_MM_TC_PROBA.md` (commit `42caf68`).

## Ce e
Un calcul mic, **în afara drumului generării**: placa face 6 înmulțiri de matrice cu programul `mm_tc` (intrări rotunjite la f16, adunarea în
f32 pe circuitele speciale — aceleași `hmma.m16n8k16.f32` care au trecut la P8, os-335). Cinci seturi mici fixe judecă **corectitudinea**;
al șaselea, `qkv-513` (6 144 × 2 304 × 768, qkv înainte pe o bucată #513), dă **primul timp** al unei înmulțiri de antrenare rapide —
cifra de comparat cu estimarea (8–17 TFLOP/s) și, mai departe, cu PyTorch.

## Ce e nou pentru placă (exact)
1. **Programul `mm_tc`** — niciodată rulat. Mesa NAK **26.3.1** (a doua unealtă, ca `mma16_f16` de la P8), SHA-256
   **`f60cd944ab67ae17ddc0d61931085791dc25f76397de82070fe0b0f1c3821ee3`** — `nvidia_ga104::mm_tc::{MM_TC, MM_TC_SHA256}`. **64 de registre**,
   blocuri de **128 de fire**, grila **⌈n/64⌉ × ⌈m/64⌉ × 1** (y > 1 pentru prima oară la un program de antrenare).
2. **Memoria partajată: 24 KiB pe bloc + o barieră** — cea mai mare de până acum. QMD-ul o declară (`ComputeQmd::to_bytes_with_shared(&mm_tc::SHARED)`,
   făcut deja în `launch_bytes`); fără ea placa ar calcula gunoi sau s-ar bloca (modelul nostru o refuză, T171c).
3. **cbuf 0 de 0x800 octeți**, argumentele la 0x700 (ca la P8).
4. Setul `qkv-513` cere **un buffer de ~82,6 MB (78,8 MiB) în VRAM**, aliniat la 4 KiB. Dacă nu se poate, setul se sare; seturile 0–4
   (≤ 256 KiB fiecare) rămân proba. Timpul are sens **numai cu bufferul în VRAM** (în RAM-ul procesorului ar măsura magistrala).
**Nimic altceva:** nicio scriere nouă în registre, nimic nou prin GSP, aceleași canale, aceeași secvență cu înregistrări de timp ca la P8 / `colsum`.

## Ce e nou la noi (ramura `g0-g2-fundatie`, commit de legat: cel din rândul `PREDARE` din hub)
- Modulul **`nvidia_ga104::mm_tc_probe`**:
  - `SETS` (6), `QKV_SET` (= 5), `QKV_REPEATS` (= 5), `QKV_BANDS`, `BAND`, `GUARD`, `FILL`, `Scale`;
  - `ProbeSet::{layout (→ Layout::{a, b, bias, c, guard, end, bytes}), a_at, b_at, bias_at, write_inputs, fill_ranges, read_ranges,
    read_len, judged_rows, buffer, inputs_fnv, args, model_out}`;
  - `launch_bytes(set, prog_va, cbuf_va, buf_va)` → (cbuf 0x800 octeți, QMD cu memoria partajată); verifică alinierile;
  - `judge_set(set, octetii_cititi, &timpi_ns)` → `SetResult` (verdict, ‰ din margine, `exact_model`, gărzi, amprenta ieșirii,
    `median_ns()`, `gflops()`); `Probe::{add, passes, qkv_skipped, rows, journal, lines}`; `parse_mm_tc(text)`.
- Existente: `nvidia_ga104::mm_tc::{MM_TC, MM_TC_SHA256, REGISTERS, BLOCK, SHARED, CBUF_BYTES, Args}`, `gpu_compute::mm_tc::{judge, model, margin}`.

## Ce legi (Release separat, opțiune oprită implicit, de ex. `test-placa --mm-tc`)
După ce lanțul de azi a pornit canalul de lucru (ca la P8 / `colsum`), o singură dată:
1. **Programul:** încarci `MM_TC` în VRAM, aliniat la 256; verifici întâi SHA-256 = `MM_TC_SHA256` (refuzi altfel).
2. **O zonă VRAM de 4 KiB** pentru cbuf + QMD: cbuf la +0 (0x800 octeți), QMD la +0x800.
3. Pentru fiecare set `i` în 0..5 (cele mici), **câte o trimitere**:
   - un buffer de `SETS[i].layout().bytes` (≤ 256 KiB, aliniat la 4 KiB; VRAM sau RAM mapat — ce aveți); scrii `SETS[i].buffer()` în el;
   - `let (cb, q) = mm_tc_probe::launch_bytes(i, prog_va, zona_va, buf_va)?`; `cb` la zona+0, `q` la zona+0x800;
   - `transfer::encode_sequence(&[zona_va + 0x800], rec.va, first)` → `run_records` → `analyze_sequence` = `ns`;
   - citești `SETS[i].read_ranges()` (un singur interval aici) și `probe.add(mm_tc_probe::judge_set(i, &octeti, &[ns])?)`.
4. **Setul `qkv-513`** (`QKV_SET`), doar dacă aveți ~82,6 MB de **VRAM** liberi:
   - scrii intrările bucată cu bucată: `SETS[5].write_inputs(|offset, octeti| scrie(buf + offset, octeti))` (bucăți ≤ 64 KiB, ~26 MB în
     total), apoi umpli cu `FILL` intervalele din `SETS[5].fill_ranges()` (3 benzi × 16 rânduri + garda, ~0,44 MB);
   - **5 trimiteri** (`QKV_REPEATS`) cu același cbuf + QMD, fiecare cu timpul ei (C se rescrie identic);
   - citești `SETS[5].read_ranges()` (4 intervale, în ordine, lipite) și `probe.add(judge_set(5, &octeti, &timpii)?)`. Judecata pe
     procesor durează ~0,6 s (modelul pe cele 3 benzi), **după** ceasul de timp.
   - fără VRAM: sari setul; rândul `kt mm_tc qkv omis` = 1 iese singur.
5. Dacă placa nu termină un set (înregistrarea nu vine): oprești proba, raportul spune „nu a terminat la setul i”; nu reîncerci.
6. La final: `probe.journal(&mut journal)` + `probe.lines("placa")`. **Nu atinge nimic din generare.**

## Ce trece și ce așteptăm (scris înainte; T171 în SIMULARE)
- **Trece** = `kt mm_tc seturi in margine` = numărul de seturi rulate (5 sau 6), `kt mm_tc set: garzi` = 1 peste tot. Marginea (T168,
  neschimbată): (K + 3)·2⁻²³·(Σ|â·b̂| + |deplasare|) + 2⁻¹²⁶ față de referința pe intrările rotunjite la f16.
- **Bit cu bit modelul nostru** (`kt mm_tc elemente exacte (model)` = `kt mm_tc elemente`, 22 188 fără qkv, 132 780 cu): predicție cu
  încredere doar ~50 % (sumatorul plăcii e o potrivire pe amprenta os-335, nu dovedit bit cu bit). Dacă nu e: verdictul tot poate trece,
  iar amprentele de mai jos ne spun unde să căutăm.
- **Abaterea maximă ≤ 100 ‰** (modelul: 69 ‰, pe setul cu limitele f16).
- **Timpul `qkv-513`: 1,3–2,7 ms ≈ 8–17 TFLOP/s** (`kt mm_tc qkv gflop/s` = 8 000–17 000). Sub 5 000 = frâna e altundeva decât credem.

Amprentele (SIMULARE; placa trebuie să le dea dacă e bit cu bit modelul):

| set | nume | intrări (fnv) | ieșire model (fnv) | elemente | ‰ model |
|---|---|---|---|---|---|
| 0 | `16x16x16` | `0x4269a6e4` | `0x5f30b5fa` | 256 | 37 |
| 1 | `70x130x33-tb-bias` | `0x1ce6707b` | `0x54ae06a1` | 9 100 | 40 |
| 2 | `64x64x96-ta` | `0x29e9fdbe` | `0xa15d5bcd` | 4 096 | 17 |
| 3 | `100x72x200-ta-tb` | `0xbcef59d2` | `0x2d7099cc` | 7 200 | 10 |
| 4 | `32x48x64-limite-f16` | `0xc8358166` | `0xcdfb5c85` | 1 536 | 69 |
| 5 | `qkv-513` (3 benzi) | `0x7f6be273` | `0x4dfae68e` | 110 592 | 5 |

## Rândurile din jurnal
`kt mm_tc seturi`, `… seturi in margine`, `… elemente`, `… elemente exacte (model)`, `… abaterea maxima (promile din margine)`,
`… qkv omis`, `… qkv timp median (ns)`, `… qkv gflop/s`, apoi pe set: `… set: index`, `intrari (fnv)`, `verdict` (1/0/2), `exacte (model)`,
`promile`, `garzi`, `iesire (fnv)`, `timp (ns)`. Le citim cu `parse_mm_tc`.
