# Pentru sesiunea OS: primul program „înapoi” al antrenării pe placa reală — `colsum` (C-160 punctul 6)

**De la Cyber-GPU, sesiunea 154 (22 nov 2026, ~09:00). Stare: SIMULARE** (T166, `crates/nvidia-ga104/tests/sim_c160_p6_colsum.rs`).
Nimic din ce scrie aici nu a rulat pe RTX 3070 Ti. Motorul de antrenare e aprobat de Adrian (21 nov 16:18, `DECIZII.md`); pe placă intră,
ca de obicei, doar printr-un Release separat și testul centrului de comandă, **un singur lucru nou**. Totul e **adăugat** (nimic redenumit
sau șters): fără opțiune, OS-ul trimite exact ce trimite azi.

## Ce e
Un calcul mic, **în afara drumului generării**: placa face sumele pe coloane ale unui gradient — gradientul unei deplasări (Σ dy) și, pentru
LayerNorm, dγ = Σ dy·x̂ și dβ = Σ dy — de 5 ori, cu 5 seturi de numere fixe. OS-ul citește sumele și le judecă pe procesor. Nimic din Kronos
nu se schimbă; niciun simbol nu depinde de acest test.

## Programul ales și de ce
**`colsum`** (`tools/nak/colsum.comp`, Mesa NAK **25.0.7** — aceeași unealtă ca toate programele care rulează azi).
- SHA-256: **`7a0fa2bde1b464cff50af588371bb09adf0f03399e880a07d75021431a47fb10`** (1 136 octeți) — `nvidia_ga104::colsum::{COLSUM, COLSUM_SHA256}`.
- De ce acesta (dintre cele 10 programe ale pasului): (1) **fără MUFU, fără memorie partajată, fără barieră, fără `atomic`** — doar adunări,
  înmulțiri, fma; modelul pe procesor face aceleași operații în ordinea codului NAK, deci **predicția e „bit cu bit”**, un da/nu fără
  interpretare (la `ln_back` intră bariera și memoria partajată, la celelalte MUFU); (2) e și programul 2 din studiu (gradientul oricărei
  deplasări) și jumătatea dγ/dβ a LayerNorm-ului înapoi — folosit de 5 ori pe strat la fiecare pas; (3) are deja oracolul f64 și marginea
  scrisă dinainte (T155).

## Ce e nou pentru placă (exact)
1. **Programul `colsum`** — niciodată rulat pe placă. 19 registre pe fir (sub maximul de azi, 25), blocuri de **256 de fire**, grila
   **⌈c/256⌉ × 1 × 1** (1, 2 sau 3 blocuri în probă), fără memorie partajată.
2. **Bufferul constant de 0x100 octeți** (`cbuf0_size` = 0x100, ca azi), adresele la 0x28…0x50, rows / c / modul la 0x58 / 0x5c / 0x60.
3. Programul **citește și scrie într-un buffer de 64 KiB** (16 pagini consecutive în adresele plăcii). RAM-ul gazdă mapat ca la P8 /
   copiere sau VRAM — programului îi e egal; alegeți ce aveți deja. Scrie doar în cele două zone de ieșire.

**Nimic altceva:** nicio scriere nouă în registre, nimic nou prin GSP, aceleași canale, aceeași secvență cu înregistrări de timp ca la P8.

## Ce e nou la noi (ramura `g0-g2-fundatie`, commit de legat: cel din rândul `PREDARE` din hub)
- Modulul **`nvidia_ga104::colsum_probe`**:
  - `SETS` (5 seturi: `deplasare-64x64`, `ln-64x64`, `ln-3x257`, `ln-8x768`, `ln-8x768-media100`), `ProbeSet::{layout, inputs, buffer,
    inputs_fnv, out_range, model_out}`;
  - bufferul unui set (`BUF_BYTES` = 64 KiB): dy, x, μ, rstd, apoi ieșirile s0 și s1, fiecare urmată de `GUARD` = 256 de cuvinte; ieșirile
    și gărzile **umplute cu NaN** (`FILL` = `0x7fc00000`) — un element nescris nu poate trece, o scriere după coloane se vede;
  - `launch_bytes(set, prog_va, cbuf_va, buf_va)` → (cbuf 0x100 octeți, QMD) — verifică alinierea (bufferul la 4 KiB, cbuf la 256);
  - `judge_set(set, iesire, ns)` → `SetResult` (verdictul, abaterea maximă în ‰ din margine, câte elemente sunt bit cu bit modelul,
    gărzile, amprenta ieșirii); `Probe::{add, passes, rows, journal, lines}`; `parse_colsum(text)` citește rândurile înapoi.
- În oracol (adăugat): `gpu_compute::train::colsum_margin` — marginea scrisă în T155, aceeași formulă, mutată în bibliotecă.
- Existente: `nvidia_ga104::colsum::{COLSUM, COLSUM_SHA256, REGISTERS, BLOCK, Args, Mode}`, `gpu_compute::layer::colsum_gpu`,
  `gpu_compute::train::{layernorm_stats, layernorm_back_rows}`.

## Ce legi (Release separat, opțiune oprită implicit, de ex. `test-placa --colsum`)
După ce lanțul de azi a pornit canalul de lucru (ca la P8), o singură dată:
1. **Programul:** încarcă `COLSUM` în VRAM ca pe celelalte; verifică întâi SHA-256 = `COLSUM_SHA256` (refuză altfel).
2. **O zonă VRAM de 4 KiB** pentru cbuf + QMD: cbuf la +0, QMD la +0x800 (ca la P8).
3. **Un buffer de 64 KiB** (`BUF_BYTES`), aliniat la 4 KiB în adresele plăcii.
4. Pentru fiecare set `i` din `SETS` (0..5), **câte o trimitere**:
   - scrii `SETS[i].buffer()` în buffer (toți cei 64 KiB: și ieșirile devin NaN);
   - `let (cb, q) = colsum_probe::launch_bytes(i, prog_va, zona_va, buf_va)?`; scrii `cb` la zona+0 și `q` la zona+0x800;
   - `transfer::encode_sequence(&[zona_va + 0x800], rec.va, first)` → `run_records` (2 înregistrări) → `analyze_sequence` = timpul `ns`;
   - `let (o, n) = SETS[i].out_range();` citești cei **n octeți de la buffer + o** și `probe.add(colsum_probe::judge_set(i, &iesire, Some(ns))?)`.
   - dacă placa nu termină (înregistrarea nu vine): oprești proba, rândul `kt colsum seturi` cu câte s-au făcut, raportul spune
     „nu a terminat la setul i”; nu reîncerci.
5. La final: `probe.journal(&mut journal)` + `probe.lines("placa")` în raport. **Nu atinge nimic din generare.**

## Oracolul, marginea, ce citim înapoi
- **Oracolul pe procesor:** `train::layernorm_back_rows` (γ = 1) — exact calculul care a dat numerele PyTorch la pasul de aur mic (T151), în f64.
- **Marginea (scrisă în T155, înainte de orice rulare, neschimbată):** pe coloană, R = numărul de rânduri, u = 2⁻²⁴:
  dβ (și deplasarea): R·u·Σ|dy| + u·|dβ| + (R+1)·2⁻¹²⁶; dγ: Σ|dy|·(rs·u·|μ| + 5u·|x̂|) + R·u·Σ|dy·x̂| + u·|dγ| + (R+1)·2⁻¹²⁶.
- **Ce citim:** doar cele două zone de ieșire cu gărzile lor (`out_range`, ≤ 8 KiB pe set) + cele 2 înregistrări de timp.
- **Predicția de verificat:** toate cele **3 778** de elemente egale **bit cu bit** cu modelul (`kt colsum elemente exacte (model)` =
  `kt colsum elemente`). Dacă trece marginea dar nu bit cu bit, placa adună altfel decât credem (FTZ, fma) — tot „trece”, dar notăm.

## Rândurile `kt` (le scrie `Probe::journal`)
| Rând | Ce înseamnă |
|---|---|
| `kt colsum seturi` | câte seturi s-au judecat (5) |
| `kt colsum seturi in margine` | **5 = trece** |
| `kt colsum elemente` / `kt colsum elemente exacte (model)` | 3 778 / câte sunt bit cu bit modelul (predicția: 3 778) |
| `kt colsum abaterea maxima (promile din margine)` | ≤ 1000 = în margine |
| `kt colsum timp median (ns)` | timpul unei lansări pe ceasul plăcii |
| `kt colsum set: index / intrari (fnv) / verdict / exacte / promile / garzi / iesire (fnv) / timp (ns)` | câte unul pe set; verdict 1 = în margine, 0 = nu, 2 = invalid (nescris / NaN / gardă atinsă); gărzi 1 = neatinse |

Amprentele (FNV-1a, 32 de biți) — intrările trimise și ieșirea pe care o dă modelul (dacă placa e bit cu bit modelul, `iesire (fnv)` = acestea):

| Set | intrări | ieșirea modelului |
|---|---|---|
| 0 `deplasare-64x64` | `0x602f78d4` | `0x24bebf23` |
| 1 `ln-64x64` | `0xc6ce6717` | `0x8a91cbf7` |
| 2 `ln-3x257` | `0x4a05914e` | `0xce87801f` |
| 3 `ln-8x768` | `0xd6b7e1b9` | `0x080b8c3e` |
| 4 `ln-8x768-media100` | `0x9476af44` | `0xfcf8f634` |

## Ce trebuie să iasă
- **Trece:** `kt colsum seturi in margine` = 5, gărzile neatinse pe toate seturile, abaterea maximă ≤ 1000 ‰.
- În SIMULARE: 5/5, **3 778 / 3 778 exacte**, cel mai rău 447 ‰ (setul `ln-3x257`; 24 / 19 / 197 / 319 la celelalte).
- Timpul (DEDUS, nemăsurat): câteva µs pe lansare (1–3 blocuri, 3–64 de rânduri) — nu e un test de viteză.
- În T166 sunt prinse 5 feluri de greșeli plantate (fără lansare, grilă prea scurtă, un rând lipsă, alt mod, μ/rstd ale altui rând) pe
  toate seturile unde au sens, plus o scriere în gardă, un element mutat, o unitate în ultimul bit (trece marginea, dar nu bit cu bit).

## După
Dacă trece: următorul program „înapoi” pe placă, apoi pasul de aur mic întreg pe placa reală (C-160 punctul 6, partea a doua; lista de
75 de lansări din `docs/C-160_P5_PAS_DE_AUR.md`). Dacă nu trece: raportul cu rândurile de mai sus ajunge — nu reîncercați.
