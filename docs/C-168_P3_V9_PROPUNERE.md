# C-168 p.3 — înmulțirea nouă pentru formele lotului 6 (`mm_tc_v9`): ce lipsește și predicția (sesiunea 295, 28 nov 2026)

Răspunsul la cererea centrului de comandă din `COMENZI.md` (C-168, notele 16:30 și 18:36): „înmulțirea pentru formele lotului 6 (cele
122 de lansări v8n) — țintă ≥ 28 TFLOP/s; propunerea voastră pentru ce lipsește (bloc mai mare? 3 trepte? ldmatrix pe B?), cu predicție”.
Totul de mai jos e CALCUL (pe listările NAK din `tools/nak/out2/`) și DEDUS; nimic nou nu a rulat pe placă.

## 1. Ce face bucla principală azi, pe fiecare felie K (16 valori de K), pe fir (CALCUL, listările NAK)

Socotit de la blocul cu `hmma` până la bariera felei, pe drumul aliniat (ramura mai lungă la fiecare ramificație):

| Program | Instrucțiuni pe felie | `hmma` | Întregi (adrese, margini, alegeri) | `mov` (copii) | Restul (`ldsm`, `ld.global`, `st.shared`, `f2fp`, `bar`) | Instrucțiuni pe `hmma` |
|---|---|---|---|---|---|---|
| v8 (128×128, 8 grupuri de 32) | 352 | 16 | 157 | 138 | 41 | 22,0 |
| v8n (128×64, 4 grupuri) | 303 | 16 | 213 | 26 | 48 | 18,9 |
| v8r (128×128, 8 grupuri) | 283 | 16 | 157 | 74 | 36 | 17,7 |

Doar ~5 % din instrucțiuni sunt înmulțirea propriu-zisă. Întregii vin din trei locuri, vizibile în sursă (`tools/nak/mm_tc_v8.comp`):
(1) transpunerile (`tra`, `trb`) se aleg **la rulare**, din fanioane, la fiecare felie (`sel`-uri + ramificații în `load_unit` /
`store_unit`); (2) marginile (`row < m`, `k < K`) se verifică pe fiecare citire, deși la #513 toate produsele sunt aliniate;
(3) adresele se refac din indici la fiecare felie (`imad`, `shf`, `lea`), în loc să se mute cu un pas fix.

## 2. Un model care explică vitezele măsurate (CALCUL + cifrele de pe placă)

Pe fiecare din cele 4 sub-unități ale unui SM: circuitele tensoriale fac 128 FLOP pe ciclu (f16 cu sume f32), deci o `hmma` m16n8k16
(4 096 FLOP) ține **32 de cicluri**; o instrucțiune întreagă sau `mov` a unui grup de 32 ține **2 cicluri** (16 benzi). Ipoteza:
**timpul felei ≈ timpul tensorial + timpul întregilor** (nu se suprapun: fiecare felie are o barieră și lanțuri de dependențe scurte).

| Program | Grupuri pe sub-unitate | Tensorial (cicluri / felie) | Întregi + `mov` (cicluri / felie) | Randament prezis | Prezis la 43,5 TFLOP/s vârf | **Măsurat qkv-513 (os-513)** |
|---|---|---|---|---|---|---|
| v8 | 4 | 4 × 16 × 32 = 2 048 | 4 × 295 × 2 = 2 360 | 46,5 % | 20,2 | **20,4** |
| v8n | 3 | 1 536 | 3 × 239 × 2 = 1 434 | 51,7 % | 22,5 | **21,2** (fc la lot 6: **22,6**) |
| v8r | 4 | 2 048 | 4 × 231 × 2 = 1 848 | 52,6 % | 22,9 | **22,2** |

Modelul nu are niciun parametru potrivit după măsurători și cade la −6…+1 % de placă pe toate trei. De aici răspunsul:
**ce lipsește nu e un bloc mai mare, nici a treia treaptă, nici `ldmatrix` pe B** (toate citirile din memoria comună sunt deja `ldsm`,
din v8), ci **mai puține instrucțiuni întregi pe `hmma`**. Câtă vreme întregii nu se suprapun cu înmulțirea, fiecare instrucțiune
întreagă scoasă din buclă se vede în timp.

Ce nu explică modelul (cinstit): de ce nu se suprapun (presupus: bariera pe fiecare felie de 16 și dependențele imediate `ldsm` →
`hmma`); vârful exact al plăcii (43,5 TFLOP/s = 48 SM × 512 FLOP / ciclu × 1,77 GHz; frecvența reală sub sarcină nu e citită).

## 3. Propunerea: `mm_tc_v9` (aceiași biți ca `mm_tc`)

Același tipar ca v8n / v8r (bucăți 128×64 și 128×128, citirile din memoria comună cu `ldsm`, ordinea fragmentelor a lui v8r — fără
copiile de registre), cu patru schimbări, toate de **instrucțiuni**, nu de aritmetică:

1. **Transpunerile fixate la compilare:** un program pentru fiecare formă folosită în pas (NN, NT, TN — `-DTA`, `-DTB`), fără `sel` pe
   fanioane și fără ramificațiile din `load_unit` / `store_unit`. Alegerea pe formă (`pick`) alege și varianta.
2. **Doar produse aliniate, fără verificări de margini în buclă** (`fits` deja refuză restul; ele rămân pe v7 / `mm_tc`).
3. **Felii de 32 în loc de 16** (două `hmma` pe K pe barieră): bariera, numărătoarea buclei și adresele se plătesc o dată la 32 de
   `hmma` în loc de 16.
4. **Adrese mutate cu un pas fix** (un `iadd` pe indicator pe felie), nu refăcute din indici.

Fiecare elementul lui C primește aceleași produse f16 în aceeași ordine de adunare ca la `mm_tc` (felia de 32 = două felii de 16 la rând,
aceeași ordine), deci aceiași biți — verificat ca la v8: SIMULARE bit cu bit, apoi proba `--mm-tc-v9` pe placă lângă `mm_tc`.

Riscuri: (a) registrele — feliile de 32 dublează valorile ținute pentru felia următoare (v8n: +24 registre → ~168; la 168 rămân 3 blocuri
pe SM, peste = 2); (b) memoria comună — o treaptă de 32 la 128×64 are ~15 KiB, două trepte ~30 KiB pe bloc: la treapta de azi
(64 KiB) încap 2 blocuri pe SM, pentru 3 e nevoie de treapta de 100 KiB (același câmp din QMD ca `--treapta-pas`, altă valoare — nou
pentru placă, de cerut OS-ului separat); (c) NAK poate să nu scoată singur adresele din buclă chiar cu sursa scrisă așa — se vede pe listare
înainte de orice probă.

## 4. Predicțiile (scrise înainte de cod)

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| X1 | CALCUL, listare | instrucțiuni întregi + `mov` pe `hmma` **≤ 8** (azi 14,4 la v8r, 14,9 la v8n); 0 `sel` pe transpuneri în buclă | 60 % |
| X2 | CALCUL, listare | registre ≤ 168 (v9n, 3 blocuri) / ≤ 128 (v9, 2 blocuri); 0 „spills” în memorie | 50 % |
| X3 | SIMULARE | aceiași biți ca `mm_tc` pe toate seturile aliniate (NN, NT, TN) și în pasul #513 | 90 % |
| X4 | placa, proba `--mm-tc-v9` | aceiași biți ca `mm_tc` pe toate seturile rulate | 85 % |
| X5 | placa, DEDUS din model | qkv-513 / dX / dW la @64k: **27–31 TFLOP/s** (0,70–0,80 ms la qkv; azi v8r 0,981). Sub 25 = modelul din §2 e greșit | 40 % |
| X6 | placa, pasul lot 6 | înmulțirile 85,5 → **63–72 ms** → pasul ~118 → **~96–105 ms** (≈ 29 000–32 000 simb/s) | 35 % |

Dacă X1 iese (instrucțiunile scad) dar X5 nu (timpul nu), atunci întregii se suprapuneau totuși cu înmulțirea și frâna e altundeva —
atunci următorul candidat e așteptarea citirilor (proba de diagnostic din `docs/C-168_OS500_V8.md` §4: v8n cu citirile înlocuite de
valori fixe, doar timp).

## 5. Ordinea pașilor
1. `mm_tc_v9.comp` + compilarea; listarea → X1, X2 (o sesiune). Dacă X1 nu iese, mă opresc și scriu de ce, înainte de orice probă.
2. Modelul pe procesor + SIMULAREA → X3; proba `--mm-tc-v9` pentru OS (nota de predare) → X4, X5.
3. După probă: v9 legat în pas (alegerea pe formă, doar adăugări) → X6.
