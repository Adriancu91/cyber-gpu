# C-168 punctul 3 — înmulțirea „v8”: planul și predicțiile (scrise ÎNAINTE de cod)

Sesiunea Cyber-GPU 278, 27 nov 2026, în cloud, FĂRĂ placă. Comanda: C-168 (Adrian, 27 nov 20:00), punctul 3: „înmulțirea v8
cu tehnicile publice Ampere: `ldmatrix` pe toate citirile din memoria comună, memoria comună în 2–3 trepte cu `cp.async`, bloc
128×128, aranjare contra conflictelor. Țintă: ≥ 28 TFLOP/s pe qkv-513 / dX / dW (azi 17–18), aceiași biți ca `mm_tc`.”
Punctul „înainte”: os-496, `mm_tc_v7` în pas = 98,1 ms din 147,8 la lot 6 = 18,3 TFLOP/s (`docs/C-157_OS496_V7_IN_PAS.md`).
Socotelile acestui plan: T294 (`crates/nvidia-ga104/tests/calc_c168_v8_plan.rs`, CALCUL + DEDUS, nimic pe placă).

## 1. Ce poate face compilatorul nostru (citit în sursa Mesa 26.3.1, MIT, ca specificație)

Decizia lui Adrian din 26 oct: programele GPU le compilează **Mesa NAK**. Am citit sursa uneltei a doua (Mesa 26.3.1,
`src/nouveau/compiler/`, SHA-256 al arhivei verificat ca în `tools/nak/build2.sh`):

1. **`cp.async` nu există în NAK 26.3.1.** Lista operațiilor din `nak/ir.rs` (110 feluri) nu are nicio copiere asincronă
   memorie-plăcii → memorie-comună; nici GLSL / SPIR-V nu are o instrucțiune pentru ea. Deci partea „`cp.async`” din C-168 p.3
   **nu se poate face cu unealta aleasă**. Ce facem în loc (tehnica publică dinainte de Ampere, aceeași idee): felia următoare
   se citește în **registre** cât circuitele speciale lucrează (v7 o face deja) și se scrie în **a doua treaptă** a memoriei
   comune → **o singură barieră pe felie** (v7: două). Să adăugăm noi `cp.async` în NAK ar cere codificarea instrucțiunii
   în binar, pe care NVIDIA nu o publică; nu o deducem din programele lor (regula clean-room). Dacă o versiune Mesa viitoare o
   aduce, o folosim atunci.
2. **`ldmatrix` (`ldsm.x4`) NAK îl alege singur**, doar în două cazuri (`nak_nir_lower_cmat.c`,
   `try_lower_cmat_load_to_ldsm`): pas constant multiplu de 16 B, în memoria comună, și **A pe rânduri / B pe coloane**. A pe
   coloane (16×16 = 4 matrice) nu primește niciodată `ldsm`; de aceea v7 are 8 `ldsm` + 32 `lds` (T279). Leacul: în memoria
   comună **A stă mereu pe rânduri [r][p] și B mereu pe coloane [c][p]** („K la rând”), oricare ar fi steagurile `TA` / `TB`;
   întoarcerea (transpunerea) se face în registre, la scriere.
3. Mărimile circuitelor în NVK 26.3.1: 16×16×16, 16×8×16, 16×8×8 (f16 → f32). Un 16×16×16 = două `hmma` 16×8×16 pe placă.

## 2. Programul (ce scriu în sesiunea următoare)

`tools/nak/mm_tc_v8.comp`, din aceeași sursă două programe (`#define`):

| | v8 | v8n |
|---|---|---|
| bloc (rânduri × coloane din C) | 128 × 128 | 128 × 64 |
| fire / grupuri de 32 | 256 / 8 (2 × 4) | 128 / 4 (2 × 2) |
| bucata fiecărui grup | 64 × 32 (4 × 2 acumulatori 16×16) | 64 × 32 (la fel) |
| felia K | 16 | 16 |
| memoria comună | 2 trepte × (128 + 128) rânduri × 48 B = **24 576 B** | 2 × (128 + 64) × 48 B = **18 432 B** |
| blocuri pe SM la 64 KiB (cu rezerva de 1 KiB) | **2** (16 grupuri) | **3** (12 grupuri) |

- Rândurile memoriei comune: 16 valori f16 + 8 de umplutură = **48 B** → cele 8 rânduri ale unei matrice `ldsm` cad în 8 grupuri
  de 16 B diferite (fără conflict, T294).
- Bucla pe felie: (1) citirile feliei t+1 (128 de biți, fără ramificări, ca v4–v7) pleacă în registre; (2) grupul calculează
  felia t din treapta t % 2 cu 4 + 2 `ldsm.x4` și 8 `coopMatMulAdd`; (3) registrele se convertesc în f16 și se scriu în treapta
  (t + 1) % 2; (4) **o barieră**. Scrierea în treapta cealaltă e sigură: toate firele au trecut bariera de după felia t − 1,
  ultima care a citit-o.
- Întoarcerea în registre: înainte (`TB`) nimic de întors; dX (fără steag) B se întoarce; dW (`TA`) A și B se întorc. Firul
  citește două `vec4` din rândurile p și p + 1 și scrie 4 perechi f16 (32 de biți) la rândurile r … r + 3.
- La fel ca `mm_tc`: produsele f16 ale acelorași valori (0 în afara matricei), sumele f32 pe K în pași de 16 **în ordine**,
  ieșirea prin câte 16×16 pe grup (finalul trecut pe placă în os-423; aici zona refolosește treptele după ultima felie), `bias`
  adunat în f32 → **aceiași biți ca `mm_tc`**. Regula de aliniere = `mm_tc_v4::fits`; aceleași 48 B de argumente.
- În Rust (doar adăugări): `nvidia_ga104::mm_tc_v8` (constantele, `fits`, QMD-ul cu 256 / 128 de fire), modelul pe procesor
  în `gpu-sim`, rândul probei `--mm-tc-v8` lângă v7, apoi alegerea pe formă în pas (`TcPick::V8` / `V8n`).

## 3. Valurile: unde ajută blocul mare și unde strică (T294, CALCUL)

Un bloc mare înseamnă mai puține blocuri; dacă ultimul val de blocuri e pe jumătate gol, placa stă. Pe cele 147 de înmulțiri
ale pasului (lot 6, 48 SM):

| Forma (lot 6) | × | blocuri v7 (192 locuri) | v8 (96) | v8n (144) |
|---|---|---|---|---|
| qkv înainte 3072×2304, K 768 | 12 | 1 728 (100 %) | 432 (90 %) | 864 (100 %) |
| fc înainte, proj-dX 3072×3072 | 24 | 2 304 (100 %) | 576 (100 %) | 1 152 (100 %) |
| out, proj înainte; qkv-, out-, fc-, cap-dX; proj-dW; fc-dW (N sau M = 768) | 85 | 576 (100 %) | 144 (75 %) | 288 (100 %) |
| qkv-dW 768×2304, K 3072 | 12 | 432 (75 %) | 108 (56 %) | 216 (75 %) |
| out-dW 768×768, K 3072 | 12 | 144 (75 %) | 36 (38 %) | 72 (50 %) |
| logit, cap-dW (16 384) | 2 | 100 % | 100 % | 97–99 % |

**Ce iese de aici:** v8 (128×128) singur ar pierde un sfert pe 85 din 147 de înmulțiri; de aceea două programe și alegerea pe
formă. **Bonus (DEDUS):** „dW e ×1,28 mai lent decât dX la aceeași muncă” (os-490, qkv-513) e aproape tot ultimul val: dX = 576
blocuri = 3 valuri pline, dW = 432 = 2,25 valuri → ×1,33 prezis din valuri, ×1,28 măsurat (T294). Nu e citirea întoarsă.

## 4. Predicțiile (scrise înainte de cod; verdictul pe placă = aceiași biți + timpul)

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| G1 | compilarea (NAK 26.3.1) | v8: **112–168 registre**, ≤ 128 (cât trebuie pentru 2 blocuri pe SM) cu 50 %; v8n ≤ 168 (3 blocuri); 0 memorie locală | 50 % / 70 % |
| G2 | listarea buclei K | citirile din memoria comună **doar `ldsm.x4`** (4 pentru A + 2 pentru B pe grup și felie), **0 `lds` / `movm`** în buclă, pe toate trei steagurile; **1 barieră** pe felie; 16 `hmma` pe grup și felie | 70 % |
| G3 | SIMULARE (`gpu-sim`) | aceiași biți ca `mm_tc` pe margini (TA / TB / BIAS, forme parțiale, nealiniate refuzate) și pe formele #513 tăiate | 90 % |
| G4 | pe placă, proba la 64 KiB | aceiași biți ca `mm_tc` pe toate seturile aliniate | 85 % |
| G5 | pe placă, qkv-513 înainte (6 144 × 2 304 × 768; v7@64k 1,196 ms) | v8 **0,75–0,96 ms = 22,7–29,1 TFLOP/s** | 45 % |
| G6 | pe placă, qkv-513-dx (v7 1,147 ms) / -dw (v7 1,464 ms) | dX: v8 **0,72–0,92 ms**; dW: v8n **1,02–1,30 ms** (v8 pe dW mai lent, 1,22–1,56: ultimul val) | 40 % |
| G7 | în pas, lot 6, cu alegerea pe formă | înmulțirile 98,1 → **67–85 ms**; placa **117–135 ms ≈ 22 800–26 300 simb/s** (alegerea: v7 12, v8 38, v8n 97) | 40 % |

Socoteala G5–G7 (DEDUS, T294): viteza unui SM plin cu v7 în pas R7 = 18,9 TFLOP/s; pentru v8 presupun **f = ×1,25–1,60** pe SM
la valuri pline (jumătate din traficul din L2 pe FLOP, o barieră pe felie în loc de două, toate citirile `ldsm`), v8n = 0,9 · f;
fiecare formă ia cel mai rapid program după valuri. Ținta C-168 (28 TFLOP/s pe qkv-513) e în banda G5, nu sigură: o scriu ca
atare. **Prag scris:** dacă G2 iese și pe placă v8 rămâne sub ×1,15 față de v7 la qkv-513, frâna nu e la citirile din memoria
comună / bariere, ci la citirile din memoria plăcii (f32, 8 B pe produs) — atunci pasul următor e copia f16 a operanzilor
(„v9”), nu blocuri mai mari.

## 5. Tăiat acum (cu motiv)

- `cp.async` și 3 trepte: nu există în NAK (§1); 3 trepte de 12 KiB ar lăsa 1 bloc pe SM la 64 KiB.
- Împărțirea pe K (split-K) pentru dW / out-dW: schimbă ordinea sumelor → **nu** aceiași biți. Leacul pentru dW e alt bloc.
- Copia f16 a greutăților și a activărilor (jumătate din octeți la citire): schimbă arena pasului; după ce v8 e măsurat.
- Treapta de 100 KiB a memoriei comune doar pentru înmulțiri: neverificată pe placă; nu o presupun.

## 6. Ce trebuie pentru cod (sesiunea următoare)

Unealta a doua (Mesa 26.3.1) nu e construită în această copie din cloud: `sh tools/nak/build2.sh mesa` (~20 min) cu pachetele
din antetul scriptului. Apoi `mm_tc_v8.comp` → listarea (G1, G2) → modelul + SIMULARE (G3) → nota pentru OS (proba
`--mm-tc-v8`, G4–G6) → alegerea în pas (G7).

**Nou pentru placă (la final):** două programe noi (`mm_tc_v8`, `mm_tc_v8n`), aceeași lansare ca `mm_tc_v7` cu alt număr de fire
și altă grilă; niciun registru, niciun firmware.

Surse (publice, citite ca specificație, nimic copiat): ghidul CUDA (memoria comună, bănci, capacitatea 8.6), documentația PTX
(`ldmatrix`, `mma.m16n8k16`), foaia publică NVIDIA GA104 (vârful f16 / sume f32); sursa Mesa 26.3.1 (MIT): `nak/ir.rs`,
`nak_nir_lower_cmat.c`, `nvk_physical_device.c` — doar citite, pentru ce poate compilatorul.

## 7. Ce a ieșit (sesiunea Cyber-GPU 279, 27 nov 2026, ~21:10–22:10, în cloud, FĂRĂ placă)

Unealta a doua (Mesa 26.3.1) reconstruită în cloud; **`mm_tc_v7` iese identic** (`953ed6b8…`), deci e aceeași unealtă ca pentru
programele de pe placă. Sursa: `tools/nak/mm_tc_v8.comp` (v8n = aceeași sursă cu `-DV8N`, `sh tools/nak/build2.sh mm_tc_v8n`).

| | v8 (128×128, 256 fire) | v8n (128×64, 128 fire) |
|---|---|---|
| SHA-256 | `1b910a53…8c4b29` | `9a0702bc…20aae` |
| registre / memorie comună / memorie locală | **128** / 24 576 B / 0 | **144** / 18 432 B / 0 |
| blocuri pe SM la 64 KiB (cu rezerva de 1 KiB) | **2** | **3** |
| bucla K (o felie de 16): `hmma` / `ldsm.x4` / `ldsm.x2` / `lds` / bariere | 16 / 4 / 4 / **0** / **1** | 16 / 4 / 4 / **0** / **1** |
| copii de registre (`mov`) în buclă | 148 | **41** |
| instrucțiuni în buclă pe `hmma` | 22,6 | 20,0 |

**Verdictul predicțiilor (cele scrise în §4):**
- **G1 — CONFIRMATĂ:** v8 = 128 registre (≤ 128, cât trebuie pentru 2 blocuri pe SM), v8n = 144 (≤ 168, 3 blocuri), 0 memorie locală.
- **G2 — CONFIRMATĂ** pe forma scrisă (prima compilare, acumulatori 16×16: 6 `ldsm.x4` = 4 pentru A + 2 pentru B, 0 `lds`, 0 `movm`,
  1 barieră, 16 `hmma`, fără ramificare pe steaguri în jurul citirilor). Programul final are acumulatori 16×8 (mai jos), deci B se
  citește cu 4 `ldsm.x2` în loc de 2 `ldsm.x4` — aceiași octeți, tot fără `lds`.
- **G3 — CONFIRMATĂ (SIMULARE, T295):** aceiași biți ca `mm_tc` pe 6 forme × 8 combinații de steaguri (TA / TB / BIAS, blocuri
  parțiale, forme mai mici decât un bloc), pentru ambele programe; refuzate: adresa nealiniată, blocul celuilalt program, grila
  prea mică. În plus, refăcut fir cu fir în Rust: fiecare bucată citită de buclă are exact op(A) / op(B) ale feliei (0 în afara
  matricei), fiecare jumătate a unei trepte e scrisă o dată, umplutura nu e nici scrisă, nici citită; citirile `ldsm` cu rânduri de
  48 B sunt fără conflict de bănci (o greșeală plantată — B citit pe rânduri — e prinsă).
- G4–G7: **NOT TESTED** (cer placa; nota pentru OS: `integration/cyber-universe/C-168_P3a_mm-tc-v8.md`).

**Ce am aflat nou (MĂSURAT pe listare, nepus în predicții):** compilatorul **copiază acumulatorii în și din registre în jurul
fiecărui `hmma`** — și în v7 (138 `mov` în bucla de 16 `hmma`, programul care merge azi pe placă). Cauza văzută în listare: un
`coopMatMulAdd` 16×16×16 devine două `hmma.m16n8k16`, iar alocatorul nu ține jumătățile acumulatorului lipite. Scris cu
acumulatori **16×8** (fiecare = exact registrele unui `hmma`), v8n coboară de la 161 la **41** de copii; v8 rămâne la 148 (la
128 de registre alocatorul nu are loc de manevră: NAK ia doar câte registre socotește că trăiesc, limita lui pentru 256 de fire
fiind 254 — citit în `nak/assign_regs.rs`, `nak/ir.rs::gpr_limit_from_local_size`). Aceleași `hmma` pe aceleași valori în
aceeași ordine → aceiași biți. Socoteala bugetului (DEDUS): la vârful f16 → f32 al plăcii un `hmma.m16n8k16` ține circuitul unei
sub-partiții ~32 de tacte, deci sub ~32 de instrucțiuni pe `hmma` copiile nu frânează prin număr, ci doar prin lanțurile de
așteptare. Dacă pe placă v8 iese sub prag (§4: < ×1,15 față de v7 la qkv-513) și v8n peste v8 pe aceleași forme, copiile din v8 sunt
primul suspect, înaintea citirilor din memoria plăcii.

**Următorul pas:** proba pe placă a sesiunii OS (`--mm-tc-v8`, G4–G6); între timp alegerea pe formă în pas (`TcPick::V8` /
`V8n`, G7, doar adăugări) și o încercare de a coborî copiile din v8 (ordinea buclelor `i`/`j` sau două unități de citire pe fir).
