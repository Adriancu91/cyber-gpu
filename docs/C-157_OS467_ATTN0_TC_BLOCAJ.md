# C-157 — os-467: de ce `attn_back_tile0_tc` nu termină pe placă, și reparația

Sesiunea 255, 26 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte de cod**.
Nimic de aici nu a rulat pe RTX 3070 Ti; ce iese pe procesor e **SIMULARE** sau **verificare a listării NAK**, timpii sunt **DEDUS**.

## 1. Ce a arătat placa (os-467, 26 nov 18:41, MĂSURAT de OS)

`atentie-tc0: placa nu a terminat la setul 0 (mic-b2-t16-d16), programul nou attn_back_tile0_tc: ultima inregistrare = 0x0`
→ oprit după 2 s, fără reîncercare; placa a revenit (lanțul de scris a rulat apoi întreg). `atentie-tc0-v4` nu a mai rulat.
Cererea OS: de ce, și o verificare în simulare care să prindă astfel de blocaje.

## 2. Cauza (DEDUS din listarea NAK, citită înainte de acest plan)

Ipoteza OS-ului a fost bariera. Am citit listarea (`tools/nak/out2/attn_back_tile0_tc.sm86.txt`):

- **Barierele sunt curate:** toate cele 6 `bar.sync` stau în blocuri uniforme (`block.u`), adică toate firele blocului ajung la ele
  (bucla `jt ≤ bx` și bucla `hh` au contoare egale pentru tot blocul).
- **Scrierea plăcii dP în memoria partajată nu e aliniată.** `coopMatStore(acc[j], cs, …, 65u, …)` — pasul de rând 65 de `float`
  (260 de octeți) — e tradus de NAK în **8 × `st.shared.strong.cta.b64`** (câte 2 numere pe fir), cu adresa
  `4 · (65 · rând + 2 · (lane & 3) + …)` (`imad r8 0x41 r9`). Pentru rândurile impare ale fragmentului (lane / 4 = 1, 3, 5, 7)
  adresa e `4 mod 8`: **o scriere de 64 de biți nealiniată la 8 octeți.** Pe plăcile NVIDIA o astfel de adresă este o eroare de
  fir („misaligned shared address”, documentația publică CUDA, lista excepțiilor `cuda-gdb`); fără un tratament al ei în driver,
  firul rămâne oprit și blocul nu se termină — exact „nu a terminat, ultima înregistrare = 0x0”.
- **De ce simularea nu a prins:** `gpu-sim` nu execută instrucțiunile programului; în locul lui rulează un înlocuitor pe procesor
  (modelul) care verifică doar argumentele, QMD-ul și rezultatul. Nicio aliniere a memoriei partajate nu ajunge acolo.
- **De ce celelalte programe pe circuite merg pe placă:** toate (`mm_tc*`, `attn_back_tile_tc`, `_tc16*`) scriu / citesc matricele
  cooperative cu pas de rând 16, 32 sau 64 de cuvinte (multiplu de 16 octeți). Doar `attn_back_tile0_tc` și `_tc_v4` folosesc 65.
  CUDA cere același lucru la `wmma::store_matrix_sync` (pasul „multiplu de 16 octeți”), idee din documentația publică.

## 3. Ce fac (plan)

1. **Verificarea nouă (pe procesor, fără placă), două reguli peste toate programele din `tools/nak/`:**
   (a) **pasul matricelor cooperative:** orice `coopMatLoad` / `coopMatStore` într-un tablou `shared` cu pasul × mărimea elementului
   ne-multiplu de 16 octeți → refuz; (b) **barierele:** în fiecare listare NAK, orice `bar.sync` dintr-un bloc neuniform → refuz.
   Test în `nvidia-ga104` (citește sursele `.comp` și listările `.sm86.txt`), cu greșelile plantate.
2. **Reparația:** placa dP din memoria partajată cu rânduri de **66** în loc de 65 (264 de octeți, multiplu de 8: fiecare scriere
   de 64 de biți aliniată), în `attn_back_tile0_tc.comp` și `attn_back_tile0_tc_v4.comp`. Ordinea calculelor nu se schimbă, deci
   **aceiași biți**; memoria partajată crește cu 256 B (33 024 → 33 280 B, tot 1 bloc pe SM la treapta 64 KiB). Programele se
   recompilează cu aceeași unealtă (Mesa NAK 26.3.1); numele modulelor rămân, se schimbă octeții, suma SHA-256, memoria partajată
   (și registrele, dacă se schimbă).
3. **Notă pentru OS** (`integration/cyber-universe/`): cauza, dependența nouă, același rând de test (os-467 din nou).

## 4. Predicțiile (scrise ÎNAINTE de cod)

- **Z1 — regula (a) prinde exact cele două programe.** Din toate `.comp` din `tools/nak/`, doar `attn_back_tile0_tc` și
  `attn_back_tile0_tc_v4` cad (pasul 65 · 4 B); toate programele care au trecut pe placă trec regula.
- **Z2 — regula (b) nu prinde nimic** în listările de azi (și cele două programe trec: bariera nu e cauza); o `bar.sync` mutată de
  mână într-un bloc neuniform e prinsă.
- **Z3 — listarea nouă:** `st.shared…b64` cu pas de rând 264 B (`imad … 0x42`), 0 memorie locală, 0 vărsări, 16 `hmma`,
  `Shared size: 33280`, registrele 56 ± 8, argumentele tot la 0x700. v4: aceleași citiri / scrieri globale de 128 de biți.
- **Z4 — SIMULARE:** modelul nu se schimbă; toate testele de azi ale celor două programe (T254, T260–T262 și pasul mixt) trec cu
  doar constantele actualizate (suma, memoria partajată); aceiași biți ca înainte.
- **Z5 — pe placă (NOT TESTED, de văzut în os-467 reluat):** programul termină; `atentie-tc0` 5 / 5 seturi; timpul cu cel mult
  ~3 % peste estimarea E7 (lanțul `dot` citește acum cu conflict de 2 pe bancă: firele `lid` și `lid + 16`).

## 4b. Corectură la plan, tot înainte de codul programelor (aceeași sesiune)

Scriind regula (a) am văzut că **66 nu ajunge pentru regula pe care o verific**: 66 · 4 = 264 B e multiplu de 8 (scrierile de 64 de
biți ar fi aliniate), dar nu de 16, cât cere documentația CUDA pentru pasul matricelor. Aleg **68** (272 B, multiplu de 16), ca regula
să fie una singură pentru toate programele. Prețul: lanțul `dot` (64 de fire, un rând pe fir) citește cu **conflict de 4 pe bancă**
în loc de 2. Se schimbă deci: **Z3** — `imad … 0x44`, `Shared size: 33792` (16 384 + 64 · 68 · 4); **Z5** — timpul pe placă cu
cel mult **~5 %** peste estimarea E7 (DEDUS: lanțul e oricum limitat de latența celor 64 de `fma` legate). Tot 1 bloc pe SM la treapta
64 KiB (2 blocuri ar cere 67 584 B). Z1, Z2, Z4 rămân.

## 5. Rezultatul (sesiunea 255, ~19:40; verificările = pe procesor; programele = COMPILATE, NU RULATE)

Unealta Mesa NAK 26.3.1 construită din nou în cloud; `attn_back_tile_tc16` iese identic (`8cbbab7e…`), deci aceeași unealtă.

| # | prezis | ieșit | verdict |
|---|---|---|---|
| Z1 | regula pasului prinde exact `attn_back_tile0_tc` și `_tc_v4` | din **74** de surse (15 cu matrice cooperative), refuzate **exact cele două** (`coopMatStore` în `cs`, 260 B); după reparație 0; pasul 65 pus înapoi de mână → prins | CONFIRMATĂ |
| Z2 | barierele: nimic prins; o barieră mutată într-un bloc neuniform → prinsă | cele **13** listări NAK 26.3.1 cu bariere: toate în blocuri uniforme; plantată → prinsă. **Dar:** la unealta veche (25.0.7) **11 din 31** de listări au bariere în blocuri pe care NAK-ul vechi nu le marchează uniforme — toate programe care au rulat bine pe placă, deci acolo regula nu dovedește nimic (doar numărate) | CONFIRMATĂ pentru unealta nouă; regula e doar o dovadă suficientă, nu o detecție |
| Z3 | `imad … 0x44`, `Shared size: 33792`, 56 ± 8 registre, 16 `hmma`, 0 locală / vărsări, argumentele la 0x700 | tc: **56 registre**, 33 792, 16 `hmma`, 8 `st.shared.b64` (rânduri de 272 B), 0 locală / vărsări; v4: 56, 33 792, aceleași 28 / 16 citiri / scrieri globale de 128 de biți, plus 4 `ld.shared.b128` (citirea plăcii dP, acum aliniată la 16) | CONFIRMATĂ |
| Z4 | SIMULARE: aceiași biți, testele vechi trec doar cu constantele noi | toate testele vechi ale celor două programe (T254, T260–T263, pasul mixt, prin RAM) trec cu doar constantele noi; verificările întregi **1 309 PASS** (8 ignorate), format, clippy, no_std | CONFIRMATĂ |
| Z5 | pe placă termină; timp ≤ ~5 % peste E7 | — | NOT TESTED |

SHA-256 noi: `attn_back_tile0_tc.sm86.sass` = `f904a08bd110296b38cbee87f4cc5af597f2f0ea8a911c4789c48fd40981b8e2`,
`attn_back_tile0_tc_v4.sm86.sass` = `0bbe619e5911131cc4c87c996839562df74f58eda1181c6916bc96fdd277d2ff`.

Am verificat și celelalte scrieri de 64 de biți din v4 (plăcile f16 dO / V, `q = 32 · r + 2 · tx`, mereu par): aliniate.

### Ce înseamnă (pe scurt)

1. Programul nou nu s-a blocat din cauza barierei, ci pentru că scria în memoria partajată la adrese „strâmbe” (nealiniate).
2. Reparația schimbă doar așezarea în memorie (rânduri de 68 în loc de 65); calculul e același, rezultatele identice în SIMULARE.
3. Avem acum o verificare automată care ar fi prins greșeala înainte de placă, rulată peste toate programele noastre.

### Ce NU e dovedit

- Că programul termină pe placă — doar reluarea os-467 o arată. Regula (a) acoperă doar matricele cooperative; alte accese
  late (64 / 128 de biți) în memoria partajată nu sunt verificate automat (le-am citit de mână doar la aceste două programe).

Cod (doar adăugări de nume): `nvidia_ga104::program_lint::{coopmat_shared_strides, barriers_outside_uniform, StrideFinding}`,
`attn_back_tile0_tc::DP_ROW`; testul `crates/nvidia-ga104/tests/lint_c157_os467.rs`. Valori schimbate: octeții, sumele și
`SHARED.bytes` ale celor două programe, `gpu_sim::cmdproc::ATTN_BACK_TILE0_TC_NEEDS` (33 792).
