# C-157 punctul 2 — după fiecare test pe placă: ce am estimat, ce a ieșit, ce tai mai departe

Fiecare test nou pe placă primește aici 5–10 rânduri (cel mai nou sus). Sursele: `Cyber-hub/rezultate/…/REZULTAT.md` și jurnalele brute.

## os-470 — dS al atenției pe circuitele speciale, programele reparate (`--atentie-tc0 --atentie-tc0-v4`, 26 nov 2026, 22:54) — sesiunea 259
Sursa: `rezultate/os-automat/2026-11-26/ciclu-auto-20261126-2254.txt` (rândurile 108–134, 217–218). Detalii: `docs/C-157_OS470.md`, T270.
- **Corectitudinea exact ca în SIMULARE:** ambele programe termină (cauza os-467 = scrierile nealiniate, confirmată); tc = modelul
  86 690 / 86 690 cuvinte, v4 = tc 138 432 / 138 432; abaterea față de F32 ≤ 2,84·10⁻⁴.
- **Viteza:** tc ×1,46 față de F32 (E7 prezisese ×1,63–2,89 — sub bandă); v4 ×1,47 față de tc (prezis ×1,2–1,8) → v4 ×2,16 față de F32.
  Citirile scalare ale lui tc mai țineau ~30 % din timp; v4 le-a tăiat.
- **Prezis pentru os-472 (scris înainte):** dS în pas 13,4 → 5,9–6,4 ms, pasul lot 6 ~264–268 ms. **Ce tai:** alte variante dS;
  urmează înmulțirile (`mm_tc_v5`). Nimic nou pentru placă.

## os-457 — atenția înapoi pe circuitele f16 (`--atentie-tc`, 26 nov 2026, 09:43) — sesiunea 247
Sursa: `rezultate/os-automat/2026-11-26/ciclu-auto-20261126-0943.txt` (rândurile 103–113, 193–195). Detalii: `docs/C-157_P2_OS457_ATTN_TC.md`, T257.
- **Corectitudinea exact ca în SIMULARE:** 4 / 5 cu marginea veche (`ascutit`, subnormale f16), tc16 = tc 5 / 5; T257 reface pe procesor
  toate cifrele plăcii (abaterea 5 / 5, ‰ 4 / 4, elementul 394 cu aceeași valoare). Cu marginea nouă: 5 / 5 (DEDUS).
- **Viteza sub predicție:** tc16 0,87 ms, tc 1,16 ms, `_3` 1,59 ms la b4 (×1,83 / ×1,38). Predicția noastră avea o greșeală de formă
  (cifra lotului 2 lângă setul lotului 4); corectată: 0,40–0,75 ms → tc16 ×1,2–2,2 peste. Cauza (DEDUS): ~8 așteptări de citiri pe pas
  de placă (8 runde de 4 citiri scalare). **Ce tai:** `attn_back_tile_tc16_v4` (citiri de 128 de biți + citirea înainte); OS-ul poate
  lega deja tc16 (~2,2–4,0 ms câștig pe pas, DEDUS).

## os-443 — `mm_tc*` la treapta 64 KiB (`--mm-tc-64k`, 25 nov 2026, 21:05) — sesiunea 237
Sursa: `rezultate/os-automat/2026-11-25/ciclu-auto-20261125-2105.txt` (rândurile 100–184). Detalii: `docs/C-157_P2_OS436.md` §6, T244.
- **Ieșit ca prezis, toate:** aceiași biți peste tot (6/6, 6/6, 5/5); qkv-513 `mm_tc` ×1,93 (prezis 1,7–2,0), `mm_tc_pf` ×1,76 =
  4,85 TFLOP/s (prezis 4,4–5,2), **`mm_tc_v4` ×1,81 = 8,28 TFLOP/s** (prezis 6,9–8,3, la margine); seturile mici neschimbate.
- **Unde suntem:** cel mai rapid program al nostru pe placă, ×2,3 față de cea mai bună înmulțire F32; trece pragul de 8 TFLOP/s pus
  pentru pasul mixt rapid. **Ce tăiem mai departe:** rularea pasului mixt rapid (P6-215, deja în `main` la OS) — prezis 72–87 ms de
  înmulțiri, pasul ~120–140 ms. Nimic nou pentru placă.

## os-423 — proba `mm_tc` pe placă (`--mm-tc`, circuitele speciale f16 → f32, 25 nov 2026, 10:46) — sesiunea 228
Sursa: `rezultate/os-automat/2026-11-25/ciclu-auto-20261125-1052.txt` (rândurile 94–104), `rezultate/2026-11-25-workstation-os423-mm-tc/`.
Predicțiile (scrise înainte, `docs/C-160_P7_MM_TC_PROBA.md`, sesiunea 160): 6/6 în margine (~85 %), bit cu bit modelul (~50 %),
abaterea ≤ 100 ‰, qkv-513 la 8–17 TFLOP/s, seturile mici ~5–10 µs.
- **Ieșit ca prezis — corectitudinea:** 6 / 6 în margine, gărzile neatinse, abaterea maximă **69 ‰** (aceeași cifră ca în SIMULARE).
  **Predicția 2 (încredere doar 50 %) a ieșit: 132 780 / 132 780 de elemente bit cu bit modelul** (`gpu_compute::mm_tc::model`, sumatorul
  `GA104_OS335` potrivit în os-335 pe 12 numere) — pe 6 forme, cu TA/TB/BIAS, valori f16 subnormale și K = 768. Sumatorul circuitelor
  speciale e acum dovedit pe placă, nu doar potrivit: `mm_tc` poate fi oracolul pasului mixt fără toleranță.
- **Ieșit altfel — viteza:** qkv-513 (6 144 × 2 304 × 768, 21,74 GFLOP) **21,15 ms = 1,03 TFLOP/s**, de **8–17 ori sub predicție**
  (pragul nostru scris: „sub 5 TFLOP/s = frâna e altundeva decât credem”). Seturile mici: 18–161 µs, nu 5–10.
- **Unde e frâna (din listare + timpi; DEDUS, nu măsurat separat):** fiecare fir copiază felia de K în 32 de elemente (16 din A, 16 din B),
  **câte unul**: citește, **așteaptă** valoarea, o rotunjește, o scrie în memoria partajată, abia apoi cere următoarea. Unealta nouă
  `tools/nak/citiri_seriale.py` pe `tools/nak/out2/mm_tc.sm86.txt`: **96 / 96 de `ld.global` așteptate înainte de următoarea citire,
  cel mult 1 în zbor**. Cu un singur bloc de 4 grupuri de fire pe unitate (24 KiB), nimic nu acoperă așteptarea.
  Socoteala: 3 456 blocuri / 48 unități = 72 de valuri → 293,8 µs pe bloc / 24 felii = **12,2 µs pe felie** ≈ 22–24 mii de cicluri
  (1,8–1,95 GHz) ≈ **~700 de cicluri × 32 de citiri la rând**; circuitele speciale au de lucru doar ~500 de cicluri pe felie.
  Seturile mici o confirmă: timpul crește cu **numărul de felii**, nu cu numărul de blocuri (1 felie 18,4 µs; 2 felii 36,8 / 38,9 µs;
  3 felii 75,3; 7 felii 161,2 → ~18–25 µs pe felie „la rece”; setul cu 6 blocuri costă cât cel cu 1 bloc la același număr de felii).
- **Aceeași formă de citire e în toate înmulțirile noastre** (listări): `mm_train_tile` 32/32, `mm_tv_128x128` 80/80,
  `mm_tv_64x64k8` 24/24, `mm_tc128` 128/128, `mm_s8` 128/128, `mm_tc_gelu` 65/65 — toate cu cel mult 1 în zbor. Excepția care
  arată leacul: **atenția gen 3** (`attn_fwd_tile3`, trecută pe placă) — indicele strâns cu `min()`, citirile într-un șir de registre
  fără ramificare, zeroul pus după → **până la 16 citiri în zbor**. La F32 frâna se amestecă cu calculul și memoria partajată (modelul
  simplu nu le ordonează bine cursa), deci partea ei acolo e NECUNOSCUTĂ; la `mm_tc`, unde calculul e mic, ea e aproape tot timpul.
- **Ce înseamnă pentru „ocuparea” (citirea centrului):** ocuparea e înmulțitorul, nu cauza. Treapta cea mai mare a noastră (64 KiB) dă
  `mm_tc` **2 blocuri** pe unitate (24 KiB + rezerva) → cel mult **×2 → ~2 TFLOP/s** (cât F32 de azi). Leacul e în program.
- **Ce tăiem mai departe (predicții DEDUS, scrise acum, înainte de cod):** (1) proba ocupării pe `mm_tc` la 64 KiB: ×1,7–2,0;
  (2) `mm_tc128` așa cum e (aceleași 32 de citiri la rând pe fir, de 4 ori mai mult calcul pe felie, 1 bloc): qkv-513 **~3–4,5 TFLOP/s**;
  (3) **`mm_tc` cu citirile în avans** (ca atenția gen 3: 32 de citiri în zbor, apoi rotunjirea și scrierea): felia ~2–3 mii de cicluri
  în loc de ~23 de mii → qkv-513 **~5–10 TFLOP/s** (plafonul de acolo = traficul ~1,4 GB din L2 / memoria plăcii, plăcile 64×64 cu f32).
  Ordinea propusă: (3) e următorul program (o sesiune, SIMULARE + listarea verificată cu `citiri_seriale.py`: ≥ 16 în zbor), apoi
  același leac în `mm_tc128` / `mm_s8` și în `mm_tv_*`. Nimic nou pentru placă din comparația asta.

## os-418 — pasul #513 cu alegerea cursei (`--mm-alegere`, 25 nov 2026, 04:33) — sesiunea 222
Sursa: `rezultate/os-automat/2026-11-25/ciclu-auto-20261125-0433.txt` (rândurile 158–218). Estimarea (sesiunea 219): pasul ~341 ms
(325–345), înmulțirile ~297 ms, aceiași biți.
- **Ieșit ca estimat:** pasul **338,4 ms = 3 026 simb/s** (−0,8 % față de predicție), înmulțirile 294,9 ms, aceleași pierderi și norme
  ca os-414 cifră cu cifră, TRECE = oracolul. Restul plăcii și partea din afara ei neschimbate (+0,1 ms fiecare).
- **Ieșit altfel:** nimic important. Raportul cursei (0,690) se păstrează în pas (0,684) → o cursă e de ajuns ca să prezicem pasul.
- **Unde suntem:** 2,03 TFLOP/s ≈ 8,5 % din vârful F32; 7,7 % din viteza Windows/PyTorch (bf16).
- **Ce tăiem mai departe:** cursa cu 5 variante (prezis 300–330 ms), apoi circuitele speciale (`mm_tc128`) după os-352.
  Detalii: `docs/C-165_CURSA_MM.md` §„os-418”; nota `integration/cyber-universe/C-165_os418-pas.md`; T228.

## os-417 — proba `mm_tv_64x64k8` singură, cu pagina de pază (25 nov 2026, 03:48) — sesiunea 221
Sursa: `rezultate/os-automat/2026-11-25/ciclu-auto-20261125-0348.txt` (rândurile 183–222). Estimarea (sesiunea 216): „termină cu aceiași
biți → cauza = citirea în avans”; timp neprezis.
- **Ieșit ca estimat:** termină, aceiași biți ca martorul → cauza confirmată; varianta se întoarce în cursă (`race_list_full()`).
- **Ieșit altfel:** pe qkv e **cea mai rapidă dintre toate** — 1 768 µs față de 1 890 la `128x128` (os-416) și 2 839 la martor (×1,61).
- Pasul #513 (aceeași listă ca os-414): 463,7 ms față de 463,6 — repetabil sub 1 ‰; înmulțirile = 93 % din placă.
- **Ce tăiem mai departe:** cursa completă cu 5 variante; bănuiala: `128x128` are doar 48 de blocuri pe formele cu 768 de coloane,
  acolo `64x64k8` (192 de blocuri) ar trebui să câștige. Nota `integration/cyber-universe/C-165_os417_64x64k8-intoarsa.md`.

## os-414 (A4, tabloul în timpul pașilor) + os-415 `--lipit-placi` (25 nov 2026, 00:18 / 00:58) — sesiunea 218
Surse: `Cyber-hub/rezultate/2026-11-25-workstation-os414-tablou/REZULTAT.md`, `rezultate/os-automat/2026-11-25/ciclu-auto-20261125-0018.txt`
(os-414: `--mm-placi --colsum-impartit`) și `ciclu-auto-20261125-0058.txt` (os-415: `--mm-placi --lipit-placi`, **fără** `--colsum-impartit`).
Predicțiile noastre (scrise înainte): A4 — „dacă ceasul e mic, întâi lămurit ceasul” (bănuiala nr. 1 de la os-410);
`--lipit-placi` (`docs/C-165_MM_TILE_GELU.md`) — **~2–4 ms pe pas mai puțin** decât nelipit, aceiași biți.

**Predicțiile cititorului de profil (T224), scrise înainte de cod:** P1 — în fiecare raport sunt 2 profiluri (mini, apoi #513), #513 = 463 594 µs /
483 de lansări (os-414) și 485 577 / 349 (os-415); P2 — suma rândurilor „pe programe” = totalul, la cel mult 1 µs pe rând (rotunjirea);
P3 — fără `colsum` + `colsum_part`, os-415 − os-414 = **+39 µs** (sub 0,1 ‰ din pas: lipirea nu schimbă nimic); P4 — înmulțirile
(`mm_train_tile` + `mm_train_tile_gelu`) **+1 190 µs**, programele mici lipite (`gelu_tr`, `vadd`, `ln_back`, `ln_back_res`) **−1 160 µs**;
P5 — eticheta `proj-dX` → `proj-dX+gelu`: **+1 256 µs = +105 µs pe lansare (+3,0 %)**.

**Ce a ieșit (T224 pe rândurile adevărate, 6/6 PASS pe procesor; P1–P5 toate confirmate):**
- **A4 (os-414): ceasul NU e frâna.** În timpul celor 10 pași #513: nucleul **1 950 MHz** constant (110 % din boost-ul public 1 770), memoria
  9 501 MHz, P0 ×10, niciun motiv de încetinire, 50 °C, puterea **127 W din 290 (44 %)**. **Bănuiala nr. 1 de la os-410 e exclusă.**
  La 1 950 MHz vârful F32 e ~24,0 TFLOP/s (48 SM × 128 benzi × 2 × 1,95 GHz, DEDUS din fișa publică); înmulțirile fac ~599 GFLOP în
  ~432 ms = **~1,39 TFLOP/s = ~5,8 % din vârf**. Placa e la ceas maxim, dar unitățile de calcul stau: **frâna e în program** (bănuiala nr. 2:
  citirile globale neascunse — fără plăci duble, două bariere pe felie; și citirile partajate). „Încărcarea 100 %” a tabloului înseamnă
  „SM-urile au avut de lucru”, nu „benzile au calculat”; puterea de 44 % e semnul bun.
- **os-415 `--lipit-placi`: `mm_train_tile_gelu` a mers pe placă din prima** (prima rulare: pierderile = oracolul, 1,5e-5 / 9,4e-5).
  Pasul a ieșit însă **+22,0 ms** (463,6 → 485,6 ms) — **nu din lipire**: lista OS-ului de la 00:58 n-a avut `--colsum-impartit`, iar
  `colsum` vechi = 24,3 ms față de 2,4 (exact ce a scos os-412). Fără `colsum`, cele două pași diferă cu **+39 µs** (0,08 ‰) — **lipirea e la egalitate**:
  - câștigat: 36 de lansări mici mai puține (`gelu_tr` 24 → 12, `vadd` 48 → 24): **−1,16 ms** cu tot cu `ln_back_res`;
  - pierdut: cele 12 `proj-dX+gelu` pe `mm_train_tile_gelu` sunt cu **+105 µs pe lansare (+3,0 %)** mai lente decât `proj-dX` pe `mm_train_tile` → **+1,26 ms**;
    restul de 135 de `mm_train_tile` neschimbate (−0,07 ms, zgomot).
- **Predicția noastră „~2–4 ms mai puțin” — GREȘITĂ** (a ieșit ~0). Am socotit cele două treceri prin `[1 024, 3 072]` la ~1 ms pe pas și n-am
  socotit costul finalului: citirea lui `u` (12,6 MB pe lansare) cu același tipar ca scrierea plăcii (4 × 4 pe fir, rânduri depărtate) →
  ~105 µs în loc de ~30 la citire unită (DEDUS, nedovedit). Lecția: la un program de înmulțire frânat de memorie, orice citire în plus în final
  costă cât ea însăși, nu „gratis”.
- **Ce propunem OS-ului:** lista #513 implicită = `--placi3 --mm-placi --colsum-impartit` (os-414, 463,6 ms); `--lipit-placi` poate rămâne
  opțiune (aceiași biți, −36 de lansări, ~0 timp) — cu `--colsum-impartit` alături ar da tot ~463,6 ms (DEDUS, T224 P3). Câștigul real rămâne
  **cursa completă (fără `64x64k8`) → `128x128`** (×1,50 pe qkv la os-413) și apoi `mm_tc`.
- Unealta nouă: `train_profile::parse_profiles` + `ParsedProfile::{diff_programs, total_without, programs_sum, tag, program}` — de acum
  comparațiile între două rulări se fac din textul raportului, nu de mână (doar adăugări).

## os-412 `--antrenare-513 --placi3 --mm-placi --colsum-impartit` (P6-189 + ceasul P6-191; 24 nov 2026, 20:58 / 21:03) — sesiunea 215
Surse: `Cyber-hub/rezultate/2026-11-24-workstation-os412-colsum/REZULTAT.md` (rapoartele `ciclu-auto-20261124-2058.txt` reper, `-2103.txt` nou).
Predicțiile noastre (`docs/C-157_P2y_COLSUM_IMPARTIT.md`, scrise înainte, T216): `colsum` 24,3 → **~2,5–5 ms**; pasul **−19–22 ms**; biții = oracolul.
- **Corect:** verdictul = oracolul (1,6e-5 / 9,6e-5, toleranța 1e-3) — `colsum_part` a mers pe placă din prima, ca în SIMULARE (aceeași abatere mică).
- **`colsum`: 2,4 ms** (`colsum_part` 1,5 + `colsum` 0,8; ×10) — cu 0,1 ms **sub** capătul de jos al intervalului. Podeaua pe lansare pe care am
  presupus-o (5–10 µs) e de fapt mai mică în lista lungă: 171 de lansări / 2,4 ms ≈ **14 µs pe lansare cu tot cu lucrul** (DEDUS) → costul fix
  ≤ ~5 µs. O cifră nouă, bună pentru estimările viitoare ale programelor mici.
- **Pasul: placa 485,6 → 463,0 ms (−22,6 ms)** — la marginea de sus a predicției (−19–22): aproape tot e `colsum` (−21,9), restul ±0,7 zgomot.
  Pasul întreg **473,5 ms = 2 162 simb/s**; între placă și pasul întreg rămân ~10 ms (citirile pierderilor + urcarea lotului; neschimbat).
- **Ceasul reparat (P6-191):** placa / pasul întreg = 977–981 ‰ (înainte 1 022 ‰) — confirmă socoteala de la os-407 (+3,4 %, frecvența de la
  pornire). Cifrele vechi „pas întreg” erau prea mici cu ~3,4 %; cele noi sunt bune.
- **Ce rămâne:** `mm_train_tile` **430,5 ms = 92,9 %** din pas; tot restul ~33 ms. Orice câștig de acum încolo e în înmulțire:
  (1) cursa C-165 (os-413, GATA DE TEST) + A4 → `MmChoice::from_race` (sesiunea aceasta: cititorul raportului, ca alegerea să iasă direct din
  textul raportului); (2) `mm_train_tile_gelu` (T220) dacă OS-ul vrea lipirile; (3) `mm_tc` / pasul mixt pe #513.

## os-410 `--antrenare-513 --placi3 --mm-placi` (P6-188: cele 147 de înmulțiri pe `mm_train_tile`; 24 nov 2026, 18:18 / 18:22) — sesiunea 211
- **Ce s-a potrivit:** biții (= oracolul, aceeași pierdere 9,8493 → 10,1735) — prima rulare a lui `mm_train_tile` pe placă, din prima.
- **Ce a ieșit altfel:** înmulțirile 718,1 → **428,7 ms (×1,68)**, prezis ×3–6; pasul 772 → **483 ms = 2 120 simb/s** (prezis 175–295 ms).
  Câștigul e **același pe toate formele** (×1,65–1,70), și la cele cu A transpus (dW, `f1`), și la cele cu B transpus (`f6`).
- **Socoteala plafonului (DEDUS, din cifrele publice Ampere: 128 de benzi F32 / ceas / SM, memoria partajată 128 B / ceas / SM):**
  `mm_train` face 2 citiri partajate pe FMA → plafon ~25 % din vârf (~5,4 TFLOP/s); `mm_train_tile` 0,5 → plafon ~50 % (~10,8).
  Măsurat: 0,84 și ~1,4 TFLOP/s = **~15 % și ~13 % din plafonul fiecăruia** — același factor comun ~1/7 la amândouă.
- **Ce NU e frâna comună (DEDUS):** ocuparea (34 de registre × 256 de fire, 8 KiB → 6 blocuri / SM = 48 de warp-uri, plin);
  conflictele de bancă la scrierea plăcilor (16 căi, dar doar la `f0` / `f6`; `f1` n-are și a câștigat la fel); lotul mic
  (192–768 de blocuri = 4–16 pe SM, iar câștigul e același și la formele mari).
- **Bănuiala nr. 1 (NECUNOSCUT până la A4):** placa nu calculează la ceasul maxim (stare P / ceas SM scăzut în timpul pasului).
  Un factor comun ~1/7 pe toate formele și pe ambele programe se potrivește exact cu un ceas mai mic, nu cu o problemă a programului.
  De aceea A4 (puterea + starea P + ceasul SM în timpul pasului) are întâietate; fără el, cursa C-165 măsoară programe sub frână.
- **Bănuiala nr. 2:** întârzierea citirilor globale neascunsă (fără plăci duble; două bariere pe felie). Cursa o lămurește parțial:
  `mm_tv_64x64k8` (felie 8, 4 KiB) vs martorul.
- **Ce tai mai departe:** variantele cursei (C-165) sunt gata (T217); propun pentru OS: cursa + A4 în aceeași pornire. Dacă A4 arată
  ceas mic, nu mai scriem programe până nu e lămurit ceasul.

## os-407 `--antrenare-513 --placi3` (P6-185: pasul #513 pe atenția gen 3; 24 nov 2026, 16:08 / 16:12) — sesiunea 210
Surse: `Cyber-hub/rezultate/2026-11-24-workstation-os407-atentie-513/REZULTAT.md`, `ciclu-auto-20261124-1612.txt` (rândurile 133–179).
Predicția noastră (`docs/C-157_P2_OS403_ATENTIA_513.md`, T212): **atenția 371 → ~20–40 ms, pasul ~770–800 ms ≈ 1 280–1 330 simb/s, pierderile = oracolul.**
- **Corect:** ambele teste = oracolul procesorului în pașii 1–9 (`Plan::new`: 2,3e-5 / 1,1e-4; `_3`: 1,5e-5 / 9,4e-5; toleranța 1e-3). Pasul 10 doar afișat (1,0e-3).
- **Atenția: exact în interval, la capătul de jos.** 20,5 ms pe pas (`attn_fwd_tile3` 6,18 + dS 5,16 + dQKV 9,18) = **1,71 ms pe strat la B 2**,
  adică jumătate din os-375 (3,43 ms pe strat la B 4): atenția scalează liniar cu lotul, cum am presupus. ×18 față de `Plan::new`.
- **Pasul: 778,2 ms pe placă = 1 315 simb/s** (prezis 770–800 / ~1 300). `mm_train` 723,9 ms (os-403: 718,9; +0,7 %, aceleași 147 lansări) = **93,0 %**.
  Restul: `colsum` 24,3 ms (3,1 %, 73 lansări, ~333 µs fiecare), atenția 20,5, AdamW + norma 5,5, restul < 4 ms.
- **Lămurit: „ceasurile placa / pasul întreg = 1 022 ‰”** (și vechiul NECUNOSCUT de la os-386: „așteptarea” < „placa”). Ceasul procesorului (TSC) e
  socotit cu frecvența măsurată **la pornire în 10 ms: 3 106 925 kHz**, dar cea măsurată acum în 5 × 50 ms e **3 002 035 kHz** (+34,9 ‰). Deci toți
  timpii pe ceasul procesorului din raport sunt **cu ~3,4 % prea mici** (DEDUS, aritmetică): pasul întreg 760,8 ms × 1,0349 = **787,4 ms** → placa / pas
  = 988 ‰ (cum trebuie, pasul cuprinde placa); la pasul mini „așteptarea” 3 519 µs × 1,0349 = 3 642 ≥ placa 3 620. Ipoteza (a) de la os-386 confirmată.
  Ceasul plăcii nu e atins. **Propunere OS (mică, fără placă):** timpii pe procesor socotiți cu frecvența „acum” (mediana 5 × 50 ms) sau calibrarea de la
  pornire pe 50 ms în loc de 10. Simbolurile/s „pasul întreg” de până azi sunt cu ~3,4 % prea mari (1 345 → ~1 300).
- **Predicția pentru `mm_train_tile` (OS-ul l-a legat deja: P6-188, `--mm-placi`), corectată:** în nota P2x scriam `mm_train` ~600 ms; e 724.
  Cu ×3–6: **`mm_train_tile` ~120–240 ms, pasul ~175–295 ms ≈ 3 500–5 850 simb/s** (DEDUS). Atunci `colsum` (24 ms) ar fi **8–14 %** din pas.
- **Ce tai mai departe:** (1) `--mm-placi` pe placă (gata în OS). (2) **`colsum` împărțit în pasul #513** (`colsum_part`, T170, gata din sesiunea 159,
  nepredat): `Plan::with_colsum_split` — sesiunea aceasta (T216, nota `integration/cyber-universe/C-157_P2y_colsum-impartit.md`).

## x16: placa mutată în SLOT2, `--ram-vram` + `--antrenare-mini --vram --lot-ram` (os-403; 24 nov 2026, 13:56 / 14:01) — sesiunea 207
Surse: `Cyber-hub/rezultate/2026-11-24-workstation-x16-ram-vram/REZULTAT.md` (rapoartele `ciclu-auto-20261124-1356.txt`, `-1401.txt`).
Predicția noastră (`docs/C-157_PCIE_LEGATURA.md`, P5; proba din OS): **sus 11–13 GB/s, jos 11–13,5 GB/s, ×1,8–2,0 față de x8; pasul mini neschimbat (~3,8 ms prin RAM).**
- **Legătura: exact cum am prezis.** Sus **12 379 MB/s** (×2,07), jos **13 529 MB/s** (×2,00) = 79 / 86 % din plafonul brut gen3 x16 (15,75 GB/s);
  4 copii în paralel = 1 000 / 999 promile din una (plafonul e legătura, nu motorul de copiere). P5 confirmat; cauza (slotul 1 = x8) confirmată.
- **Sus, bucățile sub 64 MiB merg cu 11,4–11,8 GB/s** (64 MiB: 12,4), jos aproape plat (13,2 → 13,5 GB/s). Nu e un cost fix pe copie (4 MiB e mai lent
  decât 1 MiB), ci o împrăștiere de ~5 % la urcare (DEDUS). Pentru lotul prin RAM (copii de câțiva KiB) legătura nu contează oricum.
- **Pasul mini prin RAM: 3 921 µs** (estimat ~3,8 ms, +3 %), 16 322 simb/s; arena de 26 MiB urcată în 2,44 ms (înainte ~4,9) și adusă în 2,06 ms (înainte 4,1) — ×2 tot.
  Pasul nu depinde de legătură (1,3 KiB sus / 260 B jos pe pas), cum am spus.
- **Ce n-am observat până azi — urcarea greutăților la generare:** placa 9,1 ms (12,2 GB/s, ×2 față de 18,0 ms pe x8), dar **procesorul 265–292 ms
  (~400 MB/s) — neschimbat față de x8 (271 ms)**. Deci **~97 % din urcare e pe procesor**, ~19 ms pe fiecare din cele 14 bucăți de 8 MiB, iar
  costurile pe care le știm nu ajung: o copie de 8 MiB în RAM cu cache ~1–3 ms, ~140 de cuvinte scrise în VRAM ~0,5 ms, o sonerie + așteptarea ~0,7 ms
  (DEDUS) → **~15 ms pe bucată nelămuriți**. Nu e o problemă azi (o dată pe pornire, ~0,3 s), dar e exact drumul pe care l-ar folosi mutarea
  greutăților / a stărilor AdamW prin RAM (`docs/C-157_RAM_OFFLOAD.md`): la 400 MB/s pe procesor ar fi de ~30 de ori prea încet.
- **Ce tai mai departe:** întâi măsurăm, nu ghicim: `nvidia_ga104::kronos_speed::UploadHostParts` (T213) desparte timpul procesorului pe bucată în
  3 părți (copia în RAM / pregătirea comenzilor / trimiterea + așteptarea), cu 4 citiri de ceas în bucla OS-ului; nimic nou pentru placă.
  Predicțiile, scrise înainte: §„Urcarea pe procesor” din nota `integration/cyber-universe/C-157_P2w_x16-urcare.md`.
- `docs/C-157_RAM_OFFLOAD.md` §5: coloana „slot x16” (~12,5 GB/s) e acum cifra măsurată; tabelele rămân valabile.

## ciclul automat 16:08 (`ciclu-auto-20261124-1608.txt`, cyber-gpu `408c69c`; 24 nov 2026) — sesiunea 209
Nu e un test nou: repetă os-403 (pasul #513 tot pe `Plan::new`, fără `--placi3`; urcarea fără cele 3 părți ale P2w).
- **Reproductibil:** placa 1 124 ms pe pas (910 simb/s), `mm_train` 718,9 ms / 147 lansări, atenția 18,8 % + 14,1 % — aceleași cifre ca
  os-403 la sub 0,1 %; pierderile = oracolul pe pașii 1–9. Pasul mini 3 617 µs, generarea 941 simb/s (placa), ca înainte.
- **Ce tai mai departe:** nimic nou de aici; așteptăm os-407 (`--placi3`) și urcarea pe părți. Între timp: `Plan::with_mm_tile` (T215).

## os-403 `--antrenare-mini --vram --antrenare-513` (P6-180..182: profilul pasului #513 pe programe; 24 nov 2026, 12:58) — sesiunea 206
Surse: `Cyber-hub/rezultate/2026-11-24-workstation-os403-profil-513/REZULTAT.md` (rapoartele `ciclu-auto-20261124-1258.txt`, `-1300.txt`).
Predicția noastră (os-395, mai jos; `tools/buget-antrenare/pas513.py`): **91 % înmulțiri `mm_train` la ~0,5–0,7 TFLOP/s, atenția ~9 %.**
- **Înmulțirile: aproape.** `mm_train` 718 ms = 63,9 % (147 lansări); pe ~599 GFLOP = **~0,83 TFLOP/s** (DEDUS), puțin peste intervalul nostru.
- **Atenția: RATAT de ~4 ori.** 371 ms = 33 % (`attn_fwd` 212 ms = **17,7 ms pe strat**, `attn_back` 159 ms) față de ~9 %: socoteala număra
  doar operațiile (58 GFLOP), nu cât de prost merg programele **de bază** (0,09 TFLOP/s înainte, 0,24 înapoi). Cauza, citită în OS
  (`gpu_i14.rs`, `train_513_resident`): pasul e făcut cu **`Plan::new`**, deci nu folosește atenția pe plăci gen 3, trecută pe placă la os-375
  (3,43 ms pe strat la B 4) și legată în `Plan::new_tiles3` de la sesiunea 187. Am presupus greșit că pasul #513 o folosește deja.
- **Pierderea: ca os-395** (9,85 → 10,18 pe loturi sintetice; criteriul „scade” nu se potrivește, vezi oracolul de mai jos). Restul (colsum 2 %,
  AdamW 0,4 %) = cât era de așteptat.
- **Ce tai mai departe — întâi atenția, nu înmulțirile:** e cea mai ieftină tăietură (niciun program nou, o linie în OS): `Plan::new` →
  `Plan::new_tiles3` în pasul #513. T212 (MĂSURAT în cloud): aceeași arenă 2 560 MiB, aceleași 385 lansări, doar cele 36 de atenție schimbă
  programul; SIMULARE: un strat la lățimile #513, 2 × 128, = oracolul. Estimat (DEDUS): atenția **371 → ~20–40 ms**, pasul **~770–800 ms ≈
  1 280–1 330 simb/s**. Apoi `mm_train` (~90 % din pasul nou): `--mixt` pe #513 (scara 2¹⁴), apoi `mm_train` F32 pe plăci.
  Nota: `integration/cyber-universe/C-157_P2v_atentia3-in-513.md`; plan + predicții: `docs/C-157_P2_OS403_ATENTIA_513.md`.

## os-395 `--antrenare-513` (P6-172: pasul de antrenare #513 pe placă, arena 2,5 GiB în VRAM; 24 nov 2026, 10:58) — sesiunea 205
Surse: `Cyber-hub/rezultate/2026-11-24-workstation-os395-antrenare-513/REZULTAT.md`, `ciclu-auto-20261124-1058.txt` (rândurile 112–126).
Predicția OS-ului: trece = pierderea 1 ≈ ln 16 384 (la 2 %), apoi scade. A noastră (`C-160_P7_VITEZA_ESTIMARE.md` §3): `mm_train` ~2,6 TFLOP/s → ~3 400 simb/s.
- **Corectitudinea: placa = oracolul pe procesor, MĂSURAT acum.** Oracolul nou (`crates/gpu-compute/examples/oracol_513.rs`: aceleași greutăți
  de start MT19937(1), aceleași loturi sintetice MT19937(0x164), AdamW lr 1e-3, tăiere 1,0; ~4 min pe pas, un fir, în cloud) dă
  aceleași pierderi și norme: pașii 1–7 identici la 4 zecimale (9,8493 / 3,2729 … 9,8472 / 0,8914), 8–9 la ≤ 1,1·10⁻⁴, iar pasul 10
  10,1628 față de 10,1774 pe placă (**1,4·10⁻³**, norma 4,8·10⁻⁴) — exact pasul în care rularea „sare” (norma 0,93 → 1,65), unde orice
  rotunjire diferită se mărește (DEDUS; greutățile finale necomparate). **Placa nu greșește: creșterea pierderii e a rețetei, nu a calculului.**
- **De ce urcă pierderea (DEDUS):** loturile sintetice sunt simboluri la întâmplare — nu e nimic de învățat, cel mai bun rezultat posibil e
  ln 16 384 = 9,704 pe orice lot nou. Pornim puțin deasupra (9,849: greutățile σ 0,02 pe E 768 dau logit-uri cu σ ~0,55, iar 9,704 + 0,55²/2 ≈ 9,85,
  exact cât a ieșit), iar AdamW cu lr 1e-3 (de 20 de ori rețeta lui #513, 5e-5) „învață” zgomotul fiecărui lot, care strică lotul următor.
  Criteriul „ultima pierdere sub prima” nu se potrivește cu date la întâmplare. **Propunere OS:** pe loturi sintetice, criteriul = cele 10
  pierderi și norme ale oracolului (`gpu_compute::train::ORACLE_513_SYNTH`), pașii 1–9 la 1e-3 (pasul 10 doar afișat); scăderea se cere doar pe datele lui Kronos.
- **Viteza: 1 123 ms pe pas = 911 simb/s, de ~3,7 ori sub estimarea noastră (~3 400) și de ~43 de ori sub Windows.** Socoteala
  (`tools/buget-antrenare/pas513.py`, DEDUS): 657 GFLOP pe pas → **0,59 TFLOP/s efectiv**; 91 % sunt înmulțiri `mm_train` (straturile 79 %,
  capul legat 12 %), atenția ~9 %. La 0,6 TFLOP/s înmulțirile singure ar lua ~1 000 ms — adică `mm_train` merge probabil tot la ~0,5–0,7
  TFLOP/s, ca programele de atenție măsurate la os-358 (de 4–9 ori sub estimări), nu la 2,6. **Nemăsurat pe program** — de aici cererea (2).
- **Ceasurile, lămurit (DEDUS):** placa / pasul întreg = 1 123 236 / 1 093 282 = **1,027** pe fiecare pas; la os-390/392 raportul era tot 0,973
  (invers) pe intervale de 4 ms. Același raport pe 4 ms și pe 1,1 s = **ritm diferit cu 2,7 %, nu decalaj fix.** Care ceas greșește:
  NECUNOSCUT (proba: amândouă față de un al treilea, ex. ceasul RTC pe 60 s). Nu schimbă concluziile (2,7 % din viteză).
- **Ce tai mai departe:** (1) profilul pasului #513 pe programe (`train_profile`, fără nimic nou pentru placă) — confirmă sau infirmă
  „91 % `mm_train`”; (2) circuitele speciale: proba `mm_tc` (os-352, `qkv-513`), apoi `--antrenare-513 --mixt` cu **scara S = 2¹⁴**
  (la 1 024 de simboluri pe lot d(logits) ≈ 6·10⁻⁸·S: cu S = 1 024 cade exact la marginea f16; 2¹⁴ dă ~10⁻³, iar ținta (p−1)/N·S ≈ −16 ≪ 65 504);
  dacă `mm_tc` atinge măcar 5 TFLOP/s, înmulțirile scad de la ~1 000 la ~120 ms → ~5 000–7 000 simb/s; pentru 39 300 mai trebuie lotul
  12 × 512 și restul pasului tăiat. Nimic nou trimis la placă din această comparație.
- **Generarea, testul 2:** 18/18, ca os-392.

## os-392 `--antrenare-mini --vram --pas-lipit` (P6-175; 24 nov 2026, 08:19) — sesiunea 203
Surse: `Cyber-hub/rezultate/2026-11-24-workstation-os392-pas-lipit/REZULTAT.md`, `ciclu-auto-20261124-0819.txt` (rândurile 89–124).
Predicții scrise înainte: `docs/C-160_P7_LIPIRE_INAPOI.md` (P1 biți, P3 lansări) și `docs/C-160_P7_LIPIRE_GELU.md` (G1); la os-390 (mai sus):
„după lotul prin RAM placa e ~95 % din pas, deci câștigul următor e pe placă”. Pentru modelul mini nu scrisesem un număr în µs.
- **Corectitudinea = predicția, la bit:** lipit = nelipit pe 10 / 10 pași (pierdere și normă), greutățile 0 biți diferiți din 1 150 720;
  ambele = oracolul (9,9e-8 / 6,3e-7 / 8,8e-6, ca os-390). Cele 2 programe noi (`ln_back_res`, `mm_train_gelu`) au mers din prima.
- **Lansările = predicția:** 75 → 69 (2·L = 4 din `ln_back_res` + L = 2 din `mm_train_gelu`, L = 2 straturi).
- **Timpul: placa 3 628 → 3 602 µs (−26 µs, −0,7 %), pasul întreg 4 166 → 4 141 µs (−25) = 15 455 simb/s.** DEDUS după: 26 µs / 6 lansări
  = **~4,3 µs pe lansare scutită**, exact costul unei lansări măsurat la os-331 (~4 µs); traficul scutit pe mini e ~0,4 MB (rânduri 64 × C 64)
  ≈ 1 µs. Deci pe mini lipirea câștigă doar lansări; câștigul de trafic (~2,7 GB, ~6–7 ms pe bucată) rămâne de văzut la #513.
- **Diferit (zgomot, nu lipire):** pushbuffer-ul median 71 µs lipit față de 46 nelipit — dar pe pași e **bimodal în ambele rulări**
  (42–46 µs la pașii 2–5 și 10, 71–73 µs la pașii 6–8/9); mediana a căzut pe cealaltă treaptă. Un salt de golire unic (pasul 4 nelipit: 584 µs).
- **NECUNOSCUTUL ceasurilor, încă o dată:** „așteptarea” − „placa” = −98 µs nelipit, −99 µs lipit (raport 0,973 în ambele). Intervalele sunt
  prea apropiate ca să despartă „decalaj fix” de „ceasuri diferite cu 2,7 %”; proba cu intervalul lung (urcarea arenei, 4,9 ms) rămâne.
- **Ce tai mai departe:** (1) lotul prin RAM (T209, predat, −~290 µs în afara plăcii); (2) **pe placă: unde se duc cei 3,6 ms?**
  69 de lansări × ~4 µs = ~0,3 ms, restul ~3,3 ms e calcul pe matrici mici, pe care nu l-am măsurat niciodată pe program.
  Bănuitul principal (DEDUS din formă): `inapoi:cap-dX` = `mm_train` cu doar **16 blocuri** pe 48 de multiprocesoare și K = 16 384 →
  1 024 de ture seriale de plăci 16×16 cu barieră; urmează `eroarea` (64 de rânduri × 16 384), `opt:pas` (1,15 M greutăți, ~32 MB trafic).
  Remediul: **profilul pasului pe programe** (`nvidia_ga104::train_profile`, T210, plan în `docs/C-157_P2_PROFIL_PAS.md`) — din
  înregistrările pe care OS-ul le citește deja la 3 pași; nimic nou pentru placă. Abia cu cifrele lui alegem tăietura (K împărțit, `mm_tc`, pasul mixt).
- **Generarea, testul 2:** 636,13 simb/s = os-390 (632,84), în zgomot.

## os-390 `--antrenare-mini --vram`, pasul întreg tăiat (P6-174 + tăieturile T206; 24 nov 2026, 07:37) — sesiunea 202
Surse: `Cyber-hub/rezultate/2026-11-24-workstation-os390-pas-taiat/REZULTAT.md`, `ciclu-auto-20261124-0737.txt`. Predicții scrise
înainte: os-389 (mai jos: „pasul întreg 4–5 ms”) și `integration/cyber-universe/C-157_P2_taieturi-pasul-intreg.md` (K5: driverul ~0,3 ms,
pasul ~4–4,5 ms, „la primii 2 pași umbra scrie tot”, pașii 0, 1, 8 citesc toate înregistrările).
- **Corectitudinea = predicția:** ACELEAȘI NUMERE (9,9e-8 / 6,3e-7 / 8,8e-6), 0 scrieri pe stick în pași, 0 rânduri aruncate.
- **Pasul întreg 4 150 µs = 15 421 simb/s** (estimat 4–4,5 ms; os-389: 370 ms, ×89). Placa 3 625 µs (= os-386) → placa e acum **87 %** din pas.
- **Încălzirea = predicția, la pas:** pasul 1 5 258 µs, pasul 2 4 574 (umbra pushbuffer-ului se umple de două ori), de la 3 constant ~4 135;
  pasul 9 4 271 (+121 µs = citirea tuturor înregistrărilor la `k = 8`, cum scria planul `RecordPlan::DIET`).
- **Diferit: în afara plăcii 525 µs, nu ~300.** Pe cuvânt (MĂSURAT): golirea 302 µs / 397 cuvinte = **0,76 µs** la scriere (luasem 0,66);
  citirile 13 + 70 µs / 73 cuvinte = **~1,1 µs** la citire (luasem 0,5 — de două ori mai scumpe). Și n-am numărat deloc partea de procesor:
  planul lotului 34 + lotul pus în coadă 49 + lista și zona de control 53 + umbra pushbuffer-ului 52 + TLB 24 + soneria 9 = **221 µs**.
- **Cea mai mare bucată din afara plăcii = lotul (1 284 B) scris cuvânt cu cuvânt prin fereastra VRAM, fiecare recitit:** 321 din cele 397
  de cuvinte → ~245 µs la golire + 49 µs pus în coadă ≈ **~290 µs**. Pierderile + norma (65 de cuvinte citite prin fereastră) = 70 µs.
- **NECUNOSCUT (mărunt):** „așteptarea” (3 525 µs, ceasul procesorului, de după sonerie) e mai mică decât „placa” (3 625 µs, ceasul plăcii,
  prima → ultima înregistrare). Ipoteze: (a) cele două ceasuri diferă cu ~2,8 %; (b) placa pornește înainte de semnul de după sonerie.
  Proba ieftină: același interval lung măsurat cu ambele ceasuri (ex. urcarea arenei, 4,9 ms) în același rând.
- **Ce tai mai departe (DEDUS, nimic nou pentru placă — programul de copiere `copy_b128` merge deja pe placă la urcarea arenei):**
  (1) **lotul prin RAM:** procesorul scrie lotul în zona de control din RAM (deja fixată, ~1 µs) și lista pornește cu 5 copieri mici
  RAM → arena (câte o zonă: intrări, ținte, cele 3 liste de segmente) → −~290 µs în afara plăcii, +~15–20 µs pe placă;
  (2) **pierderile + norma prin RAM:** 2 copieri la coada listei, arena → RAM, citite apoi din RAM → −~60 µs, +~8 µs pe placă;
  (3) apoi lotul în paralel (P10, predat) ascunde planul + lista (~90 µs). Estimat: **~3,75–3,85 ms pe pas ≈ 16 600–17 000 simb/s**;
  după asta placa e ~95 % din pas, deci câștigul următor e pe placă: `--pas-lipit` (os-392, următorul la rând), apoi pasul mixt.
- **De lămurit de OS (știut deja de centru):** rândul „REZULTAT TEST PLACA … antrenare mini pe placa: niciun rezultat” deși blocul spune TRECE.
- **Generarea, testul 2 (`--argmax-fara-copie --dieta2`):** 632,84 simb/s, driverul 405 µs = os-353 / os-355 (635,90 / 647,24), în zgomot.

## os-389 `--antrenare-mini --vram`, pasul întreg pe bucăți (24 nov 2026, 05:22) — sesiunea 200
Surse: `Cyber-hub/rezultate/2026-11-24-workstation-os389-bucati/REZULTAT.md`, `ciclu-auto-20261124-0522.txt` (rândurile 87–106),
`gpu-jurnal-072.txt` (rândurile 4250–4485). os-389 = doar măsurarea pe bucăți (P6-173); remediul jurnalului (P6-174) a intrat în OS după.
- **Corectitudinea și placa = os-386:** ACELEAȘI NUMERE (9,9e-8 / 6,3e-7 / 8,8e-6), placa 3,65 ms pe pas, pasul întreg 370 ms (median).
- **Bucățile = rândurile de jurnal din ele, ~16,7 ms fiecare (MĂSURAT):** golirea 102,0 ms = 6 rânduri (nota + 5 mutări ale ferestrei),
  TLB 67,4 = 4 (nota + 3), soneria 33,4 = 2, așteptarea + înregistrările + restul trimiterii 67,1 = 4, pierderile + norma 100,0 = 6.
  **22 de rânduri × 16,8 ms = 370 ms**, exact diagnosticul J1 din os-386. Pushbuffer-ul (4 µs), planul (33), lotul (49), lista (69) = nimic.
- **„Multiplii de ~33 ms” (observația centrului):** unitatea e de fapt **un rând = ~16,7 ms**; 33 = 2 rânduri, 17 = 1. Nu e un ceas: în tot
  `gpu-jurnal-072.txt` pauzele dintre rânduri sunt fie 16–19 ms (3 687 de rânduri „acum”, cele mai multe 16,1–17,5) sau 0–1 ms (rândurile ținute), fără trepte la
  16,7 / 33,3 → o operație de durată fixă, nu o pauză sau o întrerupere de ceas. Din cod (`gpu_i6::live_line`, DEDUS): tipărirea pe
  portul serial + ecran (~4 ms, os-308) + o scriere golită pe stick (~12,5 ms). Așteptarea plăcii **nu** e cu pauză: placa (3,65 ms)
  lucrează ascunsă în timpul celor 2 rânduri de după sonerie, așa că bucla de așteptare o găsește deja gata.
- **Ce a ieșit altfel:** nimic față de J1; vârful e acum la pasul 7 (678 ms, os-386: pasul 6) — tot o scriere USB lentă, alt loc.
- **Ce tai mai departe:** nimic nou de la noi — P6-174 (deja în OS, `28a4745`) face exact tăietura: cele 22 de rânduri sunt toate de rutină
  (`GEN_WRITES` = fereastra, cele 3 registre TLB, soneria; `GEN_NOTES` = cele 2 note), ținute în memorie sub `GenHold`, 220 < `HELD_MAX`
  1 024 → niciun rând pierdut. **Predicția pentru următoarea rulare (DEDUS):** 0 scrieri pe stick în pași; golirea / TLB / soneria câte
  zeci de µs; așteptarea ≈ placa (~3,6 ms, acum vizibilă); pierderile + norma < 0,5 ms; **pasul întreg 4–5 ms ≈ 13 000–16 000 simb/s**
  (azi 172); dacă iese peste 6 ms, următorul de căutat e golirea finală sau citirea înregistrărilor. După: lotul în paralel (P10) → ~4 ms,
  apoi placa însăși (pasul mixt os-384 + `mm_tc_gelu`).

## os-386 `--antrenare-mini --vram` + atenția gen 4 (24 nov 2026, 02:37) — sesiunea 197
Surse: `Cyber-hub/rezultate/2026-11-24-workstation-os386-arena-vram/REZULTAT.md`, `ciclu-auto-20261124-0237.txt`, `gpu-jurnal-069.txt`.
Predicții scrise înainte: `docs/C-160_P8_ARENA_VRAM.md` (R4), `docs/C-157_P2_OS386_PASUL_INTREG.md` (J1–J3, `b9d0899`).
- **Corectitudinea = predicția:** ACELEAȘI NUMERE (pierdere 9,9e-8, normă 6,3e-7, greutăți 8,8e-6); 1 284 B sus / 260 B jos pe pas, exact T202.
- **Placa: 3,65 ms pe pas** (estimat 1–5 ms; os-380: 25,6 ms) = 17 534 simb/s doar pe placă. Arena urcată o dată în 4,9 ms.
- **Ratat de ~100 de ori: pasul întreg ~370 ms** (estimat „câteva ms”). Cauza (MĂSURAT în jurnal + DEDUS din cod): cele **22 de rânduri de
  jurnal pe pas** merg fiecare pe stick (~16,5 ms unul) — bucla antrenării nu pornește porțiunea de rutină a generării. Fereastra fiecărui
  pas în jurnal urmează pasul întreg rând cu rând, inclusiv vârful de 664 ms de la pasul 6 (o scriere USB lentă).
- **Am greșit și în P8:** cele 286 ms din os-380 nu erau urcările de 31 MB, ci tot rândurile (corectat în nota P8).
- **Ce tai mai departe:** rutina jurnalului + `flush_live()` pe pas în OS (3 rânduri; T205: 0 scrieri pe stick în loc de 220) → **~5 ms pe
  pas ≈ 12 000–13 000 simb/s** (DEDUS: 1 533 cuvinte scrise + 369 citite pe pas ≈ 1,2 ms; procesorul ~26 µs). Apoi umbra pushbuffer-ului
  + doar prima/ultima înregistrare + lotul în paralel → ~4 ms. Nota: `integration/cyber-universe/C-157_P2_os-386_pasul-intreg.md`.
- **Atenția gen 4:** `attn_back_tile_4` 1,33 ms (estimat 1,2–1,35, gen 3: 1,60) — în interval; lanțul atenției ~3,16 ms (gen 1: 8,30).
  Identică bit cu bit, 6/6. Nimic de tăiat aici înainte de pasul întreg.

## os-376 `test-placa --i9 c135 --ram-vram` (viteza RAM ↔ VRAM cu motorul de copiere, 23 nov 2026 17:36) — sesiunea 188
Surse: `Cyber-hub/rezultate/2026-11-23-workstation-os376-ram-vram/REZULTAT.md`, `rezultate/os-automat/2026-11-23/ciclu-auto-20261123-1736.txt`
(rândurile `ram-vram …`). Nucleul: cyber-universe cu cyber-gpu `9359f21`. Predicțiile scrise înainte: `docs/C-157_RAM_VRAM_PROBA.md` (`477455c`).

### Ce a ieșit
- **Corectitudinea = predicțiile:** motorul de copiere pe canalul de lucru primește metodele (riscul 1 nu s-a întâmplat), toate cele
  31 de înregistrări, toți octeții corecți în ambele sensuri. Generarea neatinsă: 16 / 16 identice, amprenta `0x6c8501f0`.
- **Viteza ratată de 2 ori, în același fel în ambele sensuri:** sus (RAM → VRAM) 64 MiB **5,97 GB/s** mediana (cel mai bun 6,26; 16 MiB
  6,25), jos (VRAM → RAM) **6,76 GB/s** (prezis 11–13 / 11–13,5). 1 MiB: sus 5,30, jos 6,69 (prezis 6–11; costul fix ~22 µs sus,
  ~2 µs jos). 4 copii în paralel: 1 048 / 1 000 ‰ (prezis ±5 % → **în interval**). Față de `copy_b128` (6,2): ×0,96 la sus (prezis ×1,8–2,1 → ratat).

### De ce (FAPT + DEDUS)
- **Legătura e PCIe 3.0 x8, nu x16.** Nu e o presupunere: turul nostru citește starea legăturii plăcii din spațiul de configurare
  (`pci.legatura`) și în toate rapoartele de pe workstation (BOOT_0 `0xb74000a1`, `b3:00.0`) scrie **x8**: `gen3 x8 (max gen4 x16)`
  (de ex. `rezultate/os-automat/2026-11-11/20261103-1.txt`), `gen3 x8 (max gen3 x16)` (os-278), iar în repaus `gen1 x8` (os-275 —
  placa coboară singură viteza când nu lucrează). Lățimea e **mereu 8**, deși placa poate 16. Predicția mea a pus „x16” fără să citesc
  acest rând — greșeala mea.
- Plafonul PCIe 3.0 x8: 7,88 GB/s brut; cu antetele pachetelor ~7,20 GB/s la pachete de 256 B (scrieri, sensul „jos”) și ~6,63 GB/s
  la răspunsuri de 128 B (citiri, sensul „sus”). Măsurat: **jos 6,76 = 94 %**, **sus 6,26 (cel mai bun) = 94 %** din plafonul lui. Adică
  motorul de copiere umple legătura; nimic de câștigat în driver (de aceea nici 4 în paralel nu adaugă nimic).
- Același plafon se vedea deja: `copy_b128` 6,2 GB/s (C-83 I11) și urcarea greutăților la generare 6,15 GB/s (în același raport) —
  le credeam limitate de program; erau limitate de legătură.
- **De ce x8 (DEDUS, de verificat de Adrian):** Dell Precision 5820 are, după fișa Dell, două sloturi „PCIe x16” întregi, unul „x16
  cablat ca x8”, unul ca x4 și unul ca x1, toate Gen 3. Cel mai probabil placa stă în slotul cablat x8 (sau contactele nu sunt bune).
  În slotul x16 întreg: ~12–13 GB/s (predicția inițială), fără nicio schimbare de cod.

### Ce schimbă
- Studiul RAM ↔ VRAM refăcut cu cifra reală: `docs/C-157_RAM_OFFLOAD.md` §2–3 (coloana „~6,3 GB/s, azi” lângă „~12,5 GB/s, slot x16”).
- AdamW cu m, v în RAM (`ab9b537`): ~0,26 s pe pas la #513 (21 % din pas PyTorch) în loc de ~0,13 — tot merită doar cu suprapunere.
- Generarea: nimic (greutățile se urcă o dată, 18 ms).
- Propunere pentru OS (nota `integration/cyber-universe/C-157_P2p_os-376_ram-vram.md`): rândul `pci.legatura` scris și **în timpul**
  probei `--ram-vram` (doar citire), ca să vedem viteza legăturii în timpul copiei, nu doar la tur.

## Ciclul automat 15:57 `test-placa --i9 c135 --atentie-placi --atentie2 --atentie3` (a treia generație „vechi lângă nou”, 23 nov 2026) — sesiunea 186
Surse: `Cyber-hub/rezultate/os-automat/2026-11-23/ciclu-auto-20261123-1557.txt`, jurnalul brut `gpu-jurnal-060.txt` (rândurile `kt atentie3 …`).
Nucleul: cyber-universe 0.1.0 cu cyber-gpu `d885efd`. Predicțiile scrise înainte: `docs/C-157_P2_ATTN_TILE3.md` (sesiunea 182, `b6e3b5b`),
nota `integration/cyber-universe/C-157_P2n_atentie3-vechi-langa-nou.md`.

### Planul analizei (scris ÎNAINTE de socoteli, sesiunea 186)
Cifrele brute citite deja (deci **nu** sunt predicții): 6 / 6 în margine, 6 / 6 identice cu vechii; timpii `_3` (mediana a 5) 1,000 / 0,830 /
1,597 ms, `_2` în aceeași rulare 1,524 / 1,547 / 2,258, gen 1 2,488 / 1,814 / 3,995.
Plan: în `tools/buget-antrenare/nak_bucle.py`, funcția nouă `att3()`: (1) raportul măsurat / model pentru toate 9 programe (3 generații),
cu modelul de până acum (`waits` la gen 1 / `_2`, `waits_fresh` la `_3`, 500 de cicluri); (2) aceeași socoteală cu latența memoriei
`lw` de la 100 la 3 000 de cicluri, căutând `lw` la care cele 9 rapoarte sunt cel mai strânse.
Predicții (scrise acum):
- A1: la 500 de cicluri, rapoartele `_3` ies **mai mari** decât ale `_2` la fwd și back0 (programele scurtate pe partea de memorie, deci
  ce rămâne e partea pe care modelul o subestimează), iar la back1 (limitat de emisie în model) **la fel** (±10 %).
- A2: cel mai strâns `lw` e **sub** 500 de cicluri (adică partea lipsă din model nu e latența memoriei, ci altceva: emisia reală,
  barierele, ocuparea), iar împrăștierea acolo rămâne > ×1,3 (un singur factor comun nu explică totul).
- A3: tăietura următoare, dacă A1–A2 țin: la `attn_back_tile_3` (cel mai lung, limitat de emisie) — mai puține instrucțiuni pe
  buclă; la fwd3 / back0_3 — barierele și ocuparea, nu memoria.

### Ce a ieșit (sesiunea 186)
- **Corectitudinea = predicțiile, toate (P1, P2, P4):** 6 / 6 seturi în margine; **6 / 6 identice bit cu bit cu programele vechi**
  (281 032 / 209 096 / 206 592 de cuvinte; încrederea scrisă era ~65 % pentru toate trei); fwd3 exact ca fwd2 (69 097 / 178 162 față de
  model, cel mai rău 89 ‰), back0_3 și back1_3 100 % exacte; nicio lansare omisă, gărzile neatinse. Generarea neatinsă (16 / 16 identice).
- **Timpul (P3, setul mare, aceeași rulare `_2` → `_3`): toate trei în intervalele scrise:** fwd 1,52 → **1,00 ms** (×1,52; scris
  0,7–1,1, model 0,81), back0 1,55 → **0,83** (×1,86; scris 0,5–0,9, model 0,55), back1 2,26 → **1,60** (×1,41; scris 1,6–2,0, model
  1,60 — exact). Lanțul întreg **3,43 ms** (scris ~2,9; `_2` 5,33, gen 1 8,30 → ×2,42 față de prima generație). Pe bucata #513:
  ~191 → **~123 ms** (DEDUS, scalat din ~297 ms al gen 1; estimat ~113; PyTorch face toată bucata în 156 ms).
- **Analiza (`python3 tools/buget-antrenare/nak_bucle.py`, funcția `att3`, DEDUS):** măsurat / model (max, 500 de cicluri) `_3` = ×3,48 /
  ×4,70 / ×3,75 față de `_2` ×2,82 / ×3,12 / ×3,77. **A1: DA** — fwd și back0 (scurtate pe memorie) cresc, back1 (limitat de emisie)
  rămâne la fel (±1 %). **A2: NU pe jumătate** — cel mai strâns `lw` e chiar **500** (nu sub), cu împrăștierea ×1,67 pe 9 programe;
  latența memoriei nu e nici prea mică, nici prea mare în model. Ce explică mai bine: modelul **fără suprapunere** (emisie + lanț +
  partajată adunate): măsurat / sumă ×1,26–×1,85, împrăștierea **×1,47**; la back0_3 cele trei părți sunt aproape egale (0,17 / 0,18 /
  0,14 ms), deci „max” subestimează cel mai tare acolo. O constantă fixă pe lansare nu explică nimic (dreapta potrivită pe 9 puncte:
  b ≈ −0,01 ms). Factorul rămas ~×1,6 peste sumă: **NECUNOSCUT** (ceasul real sub sarcină necitit, barierele).
- **Ce tai mai departe (A3, DEDUS):** cel mai lung e acum **`attn_back_tile_3` (1,60 ms, 47 % din lanț)**, limitat de emisie în model →
  mai puține instrucțiuni în buclele dK / dV (modelul nu se poate „scurta” pe memorie acolo: memoria e doar 0,08 ms). La fwd3 / back0_3
  emisia, lanțul și memoria partajată sunt apropiate → o tăietură pe o singură parte câștigă puțin; candidat: mai multe blocuri pe SM.
- **Ce urmează din asta:** condiția din plan („leg `_3` în `train_step` după ce placa confirmă P2”) e **îndeplinită** — adăugarea
  `Plan::new_tiles3` (aceleași lansări, programele `_3`), cu plan și predicții întâi. Nimic nou de cerut OS-ului din această comparație.
- **În aceeași pornire, ciclul 15:59 (`--argmax-fara-copie --dieta2`, #513):** 18 / 18 identice cu procesorul, placa 1 076 µs, driverul
  **379 µs**, **642,55 simb/s** — la fel ca os-355 (379 µs, 647,24); nimic nou, repetabil.

## os-364 `test-placa --atentie-placi --atentie2` (atenția pe plăci „vechi lângă nou”, 23 nov 2026, 07:42) — sesiunea 178
Surse: `Cyber-hub/rezultate/2026-11-23-workstation-os364-atentie2/REZULTAT.md`, `rezultate/os-automat/2026-11-23/ciclu-auto-20261123-0742.txt`.
Predicțiile scrise înainte: `docs/C-157_P2_ATTN_TILE2.md`, nota `integration/cyber-universe/C-157_P2m_atentie2-vechi-langa-nou.md` (§5).
- **Corectitudinea = predicțiile, toate:** 6/6 seturi în margine; **6/6 identice bit cu bit cu programele vechi** (281 032 / 209 096 /
  206 592 de cuvinte; încrederea scrisă era ~80 %); exacte față de model ca la os-358 (fwd2 69 097 / 178 162, cel mai rău 89 ‰;
  back0_2 și back1_2 100 %). Generarea neatinsă (16 identice, 933 simb/s pe placă; 495,78 cu proba după).
- **Timpul (setul mare, aceeași rulare, vechi → nou): mai rapid, dar peste estimare la toate trei:** fwd 2,50 → **1,53 ms** (×1,63; estimat
  0,6–1,2), back0 1,81 → **1,54** (×1,17; estimat 0,7–1,1), back1 3,99 → **2,25** (×1,77; estimat 1,2–2,0). Pe bucata #513:
  ~297 → **~191 ms** (estimat ~90–155).

### Planul analizei (scris ÎNAINTE de socoteli, sesiunea 178)
Ce am văzut deja, înainte de plan (deci **nu** e predicție): în listările NAK ale programelor noi, fiecare citire globală păzită de un
`if` (marginea lui T sau d) stă în zona ei `bssy` / `bsync`, iar ieșirea din zonă **așteaptă rezultatul citirii** (`bra … wt=000001`).
Adică F3 („toate citirile emise deodată”) nu s-a întâmplat în cod: citirile tot una după alta, câte o latență întreagă a memoriei.
Plan: în `tools/buget-antrenare/nak_bucle.py` (1) număr automat, pe buclă, **așteptările memoriei puse în șir** (o citire globală al cărei
rezultat e așteptat înainte de următoarea); (2) modelul os-358 refăcut cu așteptările numărate (nu presupuse „una pe iterație”), aceleași
constante (1,77 GHz, 500 de cicluri de memorie, 20 partajată), pentru cele 3 programe vechi **și** cele 3 noi.
Predicții (scrise acum):
1. Pe încărcările de plăci, programele noi au **același număr** de așteptări în șir ca cele vechi (F3 = 0 câștig); în `fwd` faza B / C și
   în trecerea a doua din `back0` câte 16 / 32 de așteptări pe iterație, la vechi și la nou.
2. Cu așteptările numărate, raportul măsurat / model iese **mai strâns** decât la os-358 (×2,8 / ×5,2 / ×4,1): toate cele 6 rapoarte
   (3 vechi + 3 noi) într-un interval de cel mult ×1,6 între cel mai mic și cel mai mare.
3. Câștigul nou / vechi prezis de model (fără factorul comun) e în ±25 % din cel măsurat (1,63 / 1,17 / 1,77).
4. Remediul care urmează (dacă 1–3 țin): **citiri fără ramură** — adresa strânsă în marginea bufferului, citirea mereu, apoi `sel`
   0 / valoare (aceiași biți); modelul dă cât ar câștiga fiecare program.

### Rezultatul analizei (sesiunea 178, DEDUS; `python3 tools/buget-antrenare/nak_bucle.py`, funcția `os364`, coloana `sir`)
- **Predicția 1: DA.** Așteptări în șir pe o pereche de plăci, vechi = nou: `fwd` K 16 = 16, P + V 32 = 32, faza B 16 = 16; `back0`
  dO + V 32 = 32, P 16 = 16, trecerea a doua 32 = 32 pe iterație; `back1` dQ 32 = 32, dK / dV 32 = 32. **F3 n-a câștigat nimic**: NAK
  pune fiecare citire păzită în zona ei și așteaptă rezultatul la ieșire (îmbinarea cu 0,0), deci citirile au rămas una după alta.
  Tot câștigul măsurat vine din F1 (rânduri de 65), F2 (fără gardă sub diagonală) și F4 (bucle desfăcute).
- **Predicția 2: DA (la limită).** Măsurat / model: vechi ×2,83 / ×3,07 / ×4,14, nou ×2,82 / ×3,12 / ×3,76 → între cel mai mic și cel mai
  mare ×1,47 (pragul scris ×1,6; la os-358, fără așteptările numărate: ×2,8–×5,2). Factorul comun ~×3 rămâne **NECUNOSCUT**; o latență a
  memoriei mai mare decât 500 de cicluri nu-l explică uniform (la 1 000–3 000 de cicluri rapoartele se împrăștie mai tare, ×1,75–×2,27),
  iar `back1` iese mereu cel mai lent față de model (candidat: conflictele de bancă din dK / dV, socotite prea optimist).
- **Predicția 3: DA.** Câștigul nou / vechi prezis de model: ×1,63 / ×1,19 / ×1,61 față de ×1,63 / ×1,17 / ×1,77 măsurat (în ±10 %).
  Deci modelul pe bucle ține pentru **diferențe**, chiar dacă nu știe factorul absolut — bun pentru a alege ce tăiem.
- **Ce tai mai departe (predicția 4, DEDUS cu factorul măsurat al fiecărui program):** citiri fără ramură (adresa strânsă în buffer,
  citirea mereu, apoi `sel` 0 / valoare; aceleași sume, aceiași biți) → **fwd2 ~1,53 → ~0,76 ms, back0_2 ~1,54 → ~0,63, back1_2
  ~2,25 → ~1,76**; pe bucata #513 ~191 → **~113 ms** (PyTorch: 156 ms toată bucata). La `back0_2` cea mai mare parte e trecerea a doua
  (32 de citiri în șir × 8 plăci pe bloc); acolo și citirile de deasupra diagonalei se pot sări (j > i scrie doar 0). Candidat separat,
  NECUNOSCUT: citirile din bufferele scrise (`p` la fwd, `ds` la back0) ies `ld.global.strong.sys` (fără cache-ul L1), cele doar citite
  `constant`.
- **Ce NU s-a schimbat:** corectitudinea și pasul de aur nu sunt atinse; generarea neatinsă. Nimic nou de cerut OS-ului din această
  comparație; programele `_3` (citiri fără ramură) vin ca adăugare, cu plan și predicții întâi, și o probă „vechi lângă nou” ca os-364.

## os-358 `test-placa --atentie-placi` (atenția de antrenare „pe plăci”, 23 nov 2026, 00:36) — sesiunea 171
Surse: `Cyber-hub/rezultate/os-automat/2026-11-23/ciclu-auto-20261123-0036.txt`, `-0038.txt`, `gpu-jurnal-055.txt` (nucleul cu
`cyber-gpu 44a4d16`). Predicțiile scrise înainte: `docs/C-160_P7_ATENTIA_PROBA.md` (`608f377`).
- **Corectitudinea = predicțiile:** 6/6 seturi în margine la toate trei programele, gărzile neatinse; `attn_back_tile0` 106 226/106 226 și
  `attn_back_tile` 206 592/206 592 **exacte bit cu bit** (predicția 2); `attn_fwd_tile` 69 097/178 162 = **38,8 %** exacte (predicția
  20–80 %), cel mai rău **89 ‰** (predicția ≤ 300 ‰). Bufferul de 151 MB în VRAM, 25 MB urcați și recitiți identic.
- **Timpul pe setul mare (4 × 512, 12 capete × 64, mediana a 5 lansări) — ratat de 4–9 ori:** `attn_fwd_tile` **2,49 ms** (estimat ~0,53),
  `attn_back_tile0` **1,81 ms** (~0,49), `attn_back_tile` **3,99 ms** (~0,42–0,68). Adică **0,50–0,73 TFLOP/s** = 2–3 % din F32-ul plăcii
  (~21,7, DEDUS); pe bucata #513 ~89 + 65 + 143 = **~297 ms** (estimat 52–61; programele vechi de 32 de fire ~257, nemăsurate; toată
  bucata PyTorch 156). E **prima măsurătoare pe placă a unui program limitat de calcul**: toate estimările de viteză a antrenării se
  sprijineau pe „4–8 TFLOP/s efectiv”, nemăsurat până acum.
- **De ce (DEDUS, `tools/buget-antrenare/nak_bucle.py`, din listările NAK):** estimarea a socotit doar FLOP la 4–8 TFLOP/s și octeții la
  450 GB/s; n-a privit **codul scris de NAK**. Patru cauze, socotite pe bucle (ceasul presupus 1,77 GHz):
  1. **Bucle care emit mult pentru puțin calcul:** 16 `ffma` din 38 (fwd A), 53 (back0), 68 (fwd C), 76 (back1 dQ), 120 (back1 dK/dV)
     de instrucțiuni pe iterație; în `fwd` C și în `back1` garda `j <= i` e pusă **pe fiecare din cele 4 rânduri, la fiecare iterație**
     = 4 zone `bssy/bsync` (+ adresele refăcute) — și pe plăcile de sub diagonală, unde garda e mereu adevărată. Buclele nedesfăcute
     așteaptă 107–460 de cicluri pe iterație (suma întârzierilor scrise de NAK).
  2. **`attn_fwd_tile`, faza A: conflicte de bancă 16 la 1** — placa K e ținută `[j][e]` cu rânduri de 64 de float-uri, iar cele 16 fire
     `tx` citesc 16 rânduri diferite la același `e` → aceeași bancă. Singur ~0,88 ms din 2,49.
  3. **Plăcile urcate în memoria partajată câte o valoare pe iterație** (`dont_unroll`: citire → scriere → următoarea) = 16 așteptări
     ale memoriei pe placă în loc de una.
  4. **Ocupare mică:** 37 / 32 / 32 KiB partajați + 55 / 50 / 73 registre → 16 / 24 / 24 warp-uri pe SM din 48.
  Modelul pe bucle dă **0,88–1,83 / 0,35–0,76 / 0,96–2,00 ms** față de 2,49 / 1,81 / 3,99 măsurat: `fwd` aproape explicat, la cele două
  „înapoi” rămâne un **factor ~2 NECUNOSCUT** (candidați: ceasul real sub sarcină — necitit, „P0” nu spune frecvența; latența memoriei
  sub sarcină peste 500 de cicluri; `membar.sc.gpu` după fiecare barieră).
- **Al doilea test (după FLR, `--argmax-fara-copie --dieta2`):** placa 1 079 µs, driverul 404, **635,72 simb/s** pe mediană (621,16 media
  pe poziții) = os-353 (635,90). Generarea neatinsă de probă (505,56 cu proba după, ca os-346).
- **Ce NU blochează:** pasul de aur mic pe placă (os-351) folosește `attn_fwd` / `attn_back` (programele vechi, forme mici) — corectitudinea
  nu e atinsă de cifrele de aici. Blocate sunt doar **estimările de viteză** ale antrenării pe placă (`C-160_P7_VITEZA_ESTIMARE.md`), care
  trebuie înmulțite cu un factor de măsurat, nu presupus.
- **Ce tai mai departe (plan, cu aceiași biți — ordinea fiecărei sume neschimbată):** (F1) K în memoria partajată cu rânduri de 65 →
  fără conflicte; (F2) plăcile de sub diagonală fără gardă, doar placa diagonalei cu gardă; (F3) cele 16 citiri ale unei plăci emise
  toate, apoi scrise; (F4) bucla interioară desfăcută câte 4. Programe noi lângă cele vechi (`attn_*_tile2`), cronometrate **în aceeași
  rulare** cu cele vechi — diferența arată cât face fiecare cauză și dacă factorul ~2 e ceasul (ar rămâne la toate). Estimat (DEDUS,
  factor 2): ~0,6–1,0 / ~0,5–0,8 / ~1,0–1,6 ms → ~75–120 ms pe bucata #513. Nota pentru OS: `integration/cyber-universe/C-157_P2l_os-358_atentie.md`.

## os-355, ciclul 20:06 / 20:08 (două teste într-o pornire: #513 dieta 2 + #651 pe `lipita-ln-32`, 22 nov 2026) — sesiunea 167
Surse: `Cyber-hub/rezultate/os-automat/2026-11-22/ciclu-auto-20261122-2006.txt` și `-2008.txt` (nucleul cu `cyber-gpu 54563a5`).
- **Testul 1 (#513, `--argmax-fara-copie --dieta2`, a doua oară după os-353):** placa 1 074 µs, driverul **379 µs** (os-353: 390), jurnal 61,
  simbolul 1 549 µs, **647,24 simb/s** (os-353: 635,90); 18/18, 8 cuvinte de înregistrări, 0 poziții lente. Repetabil: ±11 µs la driver.
- **Testul 2 (#651, `--k651`, implicit `lipita-ln-32`, fără dieta 2 și cu copie; după resetul din interior):** **estimat** (os-343 mai jos,
  DEDUS): placa ~2 030 µs, driverul ~1 317, ~3,41 ms = **~293 simb/s**, peste ~3,8 ms = sferturile nu scalează. **Ieșit (MĂSURAT):** placa
  **2 045 µs** (+0,7 %), driverul **1 385** (+5 %), jurnal 64, simbolul **3 507 µs = 285,75 simb/s** (−2,5 %); 16/16 identice cu procesorul,
  0 poziții lente, abaterea logit-urilor 3,65e-5. Sferturile scalează la #651 ca la #513 (fc 18,75 / qkv 14,69 / proj 11,94 / out 6,08 µs
  pe lansare = #513 la 0,3 %). Amprenta de referință nouă pentru #651 + `lipita-ln-32`: **`0x79eec72f`**.
- **Diferit:** driverul +68 µs față de os-343 (1 317) la același număr de lansări — în zgomotul de ±5 % văzut între rulări; atenția pe
  lansare 21,09 µs (la #513 în același ciclu 20,03; NECUNOSCUT de ce +5 %, poate textul / pozițiile — nu contează la viteză acum).
- **Estimare pentru `--k651 --argmax-fara-copie --dieta2` (DEDUS, din #513):** fără copie −~217 µs (citirea 49 + argmax 85 + FNV ~83,
  nu depind de straturi); A+B: citirea a ~173 de înregistrări în loc de 2 (~1,9 µs fiecare) **~−320** + așteptarea ~−55 → driverul
  **~720–850 µs**; placa ~2 049 (argmax 16 în loc de copie 12) → simbolul ~2,85–2,95 ms = **~340–350 simb/s** (×1,2). Peste ~3,1 ms =
  dieta nu câștigă la #651 cât la #513.
- **Ce tai mai departe:** la #651 placa e 58 % din simbol; după dietă ~70 %, iar acolo atenția de generare (509 µs, 24 × 21 µs) e cel mai
  mare fel — același candidat ca la #513. Nota pentru OS: `integration/cyber-universe/C-157_P2k_os-355_k651.md`.

## os-353 `test-placa --argmax-fara-copie --dieta2` (dieta 2, bucățile A+B, fără copie; ciclul automat 22 nov 2026, 19:36) — sesiunea 166
Sursa: `Cyber-hub/rezultate/os-automat/2026-11-22/ciclu-auto-20261122-1936.txt` (nucleul cu `cyber-gpu 54563a5`; #513, `lipita-ln-32`).
- **Estimat** (`integration/cyber-universe/C-157_P2j_argmax-fara-copie.md` §3, DEDUS): placa neschimbată ~1 076 µs, 8 cuvinte de
  înregistrări pe simbol, driverul 608 → **~410–460 µs**, simbolul ~1 550–1 600 µs = **~625–645 simb/s**; peste ~520 µs = așteptarea nu prinde.
- **Ieșit (MĂSURAT):** corect 18/18, 16 simboluri identice, 0 poziții lente; placa **1 075 µs**, cuvinte de înregistrări **8**, două
  înregistrări pe 14 din 18 poziții (complete: 0, 1, 8, 16, ca în plan); **driverul 390 µs** (−218), jurnal 63, simbolul **1 551 µs**;
  **635,90 simb/s** în medie (644,74 pe mediană). În interval la viteză; driverul cu ~20 µs **mai bun** decât capătul de jos.
- **Diferit / de ce:** A+B au dat 218 µs, eu pusesem 150–200: citirea celor 89 de înregistrări costa ~166 µs (os-328), acum 8 cuvinte
  (~4 µs) → ~160 doar din A; restul ~55 µs din B (așteptarea la 100 µs pierdea ~50 în medie). **Așteptarea prinde:** 21 de citiri pe
  simbol = 1 imediată + ~20 la 5 µs în marja de 100 µs (DEDUS din `WaitPlan`: somn = cel mai scurt timp recent − 100 µs).
- **Ținta dietei (C-157 p.3, „driver sub 0,5 ms pe simbol”) e atinsă pe placă: 390 µs.** Windows/PyTorch 129,12 → ×4,9.
- **Ce rămâne în cei 390 µs (DEDUS, neîmpărțit încă pe placă):** 130 de cuvinte VRAM scrise prin umbră (~0,66 µs scriere + recitire →
  ~86 µs), soneria, construirea celor 88 de QMD-uri/pushbuffer-ului în umbră și ceasul; restul (~300 µs) **NECUNOSCUT** pe bucăți.
- **Ce tai mai departe:** (1) placa e acum 69 % din simbol; cel mai mare fel e **atenția de generare** (242 µs, 12 × ~20 µs: un singur
  rând de interogare; de verificat dacă e limitată de fir, ca atenția antrenării în C-160 p.7 — NECUNOSCUT) → candidatul următor pe placă, înaintea înmulțirilor (fc 224, qkv 175);
  (2) marja așteptării se poate coborî de la 100 la ~40 µs (diferența cel-mai-lent / median e ~85 µs, dar somnul ia minimul a 4 timpi):
  câștig ~0 la viteză, doar mai puține citiri — nu merită un test; (3) pentru driver, un rând `kt` pe bucăți (umbră / QMD / sonerie)
  ar arăta unde sunt cei ~300 µs — propunere mică pentru OS, nimic nou pe placă.

## Ciclul automat 15:06 `test-placa --i9 c135 --argmax-fara-copie` (argmax fără copie pe `lipita-ln-32`, 22 nov 2026) — sesiunea 161
Sursa: `Cyber-hub/rezultate/os-automat/2026-11-22/ciclu-auto-20261122-1506.txt` + `gpu-jurnal-048.txt` (nucleul cu `cyber-gpu 3b4ca77`;
#513, `lipita-ln-32` implicită, dieta 1 pornită, **dieta 2 oprită**: OS-ul o oprește când argmax e fără copie, `gpu_i14.rs` rândul 767).
- **Estimat** (sesiunea 151, mai sus, DEDUS): placa ~1 070 µs, driverul ~680 µs (fără citirea logit-urilor și argmax-ul pe procesor) →
  **~550 simb/s**; simbolurile = ale procesorului; lansarea argmax ~15 µs (os-340).
- **Ieșit (MĂSURAT):** **18/18 poziții, simbolul plăcii = al procesorului**, 16 simboluri identice, 0 poziții lente. Placa **1 076 µs**
  (+5: argmax 16,2 µs în locul copierii de 11,9), driverul **608 µs**, jurnalul 64 → simbolul median **1 778 µs = 562 simb/s**, media
  rulării **568,28 simb/s** (estimat ~550: **+3 %**). Windows/PyTorch 129,12 → **×4,4**; doar lansările 932,6 simb/s.
- **Ce a ieșit altfel:** driverul a scăzut mai mult decât am socotit: 825 (os-346) → 608 = **−217 µs**, nu −145. Citirea logit-urilor (49 µs)
  și argmax-ul pe procesor (85 µs) dau −134; restul de **~83 µs e amprenta FNV** pe cele 64 KiB de logit-uri, care fără copie nu se mai
  face deloc (la estimare o lăsasem în ceas). Socoteala închide la o microsecundă (DEDUS din `kt` os-346 + codul OS `run_top`).
- **Mărunt:** `kt median argmax placa` 16,19 µs (os-340: 15,04, +1,1 µs, în zgomot); `kt median citirea rezultatului` 0 µs (16 octeți).
  Rândul text `kt lista … cu copierea logit-urilor` e vechi pentru modul fără copie (lansarea a 88-a e argmax), iar legenda
  `kt lista (0 … 4)` nu spune ce e valoarea 7 — de corectat în OS când trece pe acolo.
- **Ce tai mai departe:** din cei 608 µs de driver, **~150–200 µs** sunt încă citirea celor 89 de înregistrări + așteptarea la 100 µs —
  exact bucățile A și B ale dietei 2, care **nu depind de logit-uri** (C și D, amprenta și argmax-ul pe biți, nu mai au obiect fără copie).
  Le-am oprit degeaba împreună: propun ca OS-ul să lase A+B pornite și cu argmax fără copie (4 schimbări mici, nota de mai jos) →
  driverul ~410–460 µs, simbolul ~1 550–1 600 µs = **~625–645 simb/s** (DEDUS; Windows ×4,9). Apoi pe placă rămân înmulțirile (~700 µs).
- Nota pentru OS: `integration/cyber-universe/C-157_P2j_argmax-fara-copie.md`. Nimic nou pentru placă.

## os-349 `test-placa --i9 c135 --colsum` (primul program „înapoi” al antrenării pe placă, ciclul automat 22 nov 2026, 12:36) — sesiunea 159
Sursa: `Cyber-hub/rezultate/2026-11-22-workstation-os349-colsum/REZULTAT.md` + `rezultate/os-automat/2026-11-22/ciclu-auto-20261122-1236.txt`
și `gpu-jurnal-046.txt` (nucleul cu `cyber-gpu 3b4ca77`; generarea pe `lipita-ln`, implicitul de atunci).
- **Estimat** (nota `integration/cyber-universe/C-160_P6_primul-program-inapoi.md`, SIMULARE T166): 5/5 seturi, **3 778 / 3 778 elemente
  bit cu bit modelul**, abaterea 24 / 19 / 447 / 197 / 319 ‰ din margine, gărzile neatinse.
- **Ieșit (MĂSURAT):** exact asta. 5/5 în margine, 3 778 / 3 778 exacte, **aceleași ‰ pe fiecare set** (24 / 19 / 447 / 197 / 319) și
  **toate cele 5 amprente ale ieșirii = ale modelului** (`0x24bebf23`, `0x8a91cbf7`, `0xce87801f`, `0x080b8c3e`, `0xfcf8f634`); amprentele
  intrărilor = cele din notă, gărzile neatinse. Deci placa adună în ordinea și cu rotunjirea pe care le credem (fără FTZ ascuns, fma
  unde l-a pus NAK): **modelul pe procesor al lui `colsum` e placa, bit cu bit** — primul program „înapoi” dovedit pe placa reală.
- **Timpul (MĂSURAT, o lansare pe set):** 46,5 / 55,0 / 6,8 / 15,7 / 15,9 µs. **Atenție, nu e timpul de la antrenare:** OS-ul ține
  bufferul în RAM-ul procesorului (`alloc_sys_mapped`), deci fiecare citire trece prin PCIe. Programul are un fir pe coloană și adună
  rândurile unul după altul → un drum dus-întors la memorie pe rând: **~0,6–0,7 µs pe rând** (potrivirea 8 → 64 rânduri; aproximativă,
  setul de 3 rânduri nu se așază pe dreaptă).
- **Ce înseamnă pentru motor (DEDUS, `python3 tools/buget-antrenare/colsum_timp.py`):** la #513 un pas are **73 de `colsum`** pe bucata de
  6 144 de rânduri. Așa cum e programul, ~2–4 ms pe lansare → **~165–275 ms pe bucată**, adică **106–176 % din tot timpul în care PyTorch
  face bucata întreagă** (156 ms). Podeaua (fiecare număr citit o dată, ~450 GB/s) e ~6,6 ms (4 %). Deci `colsum` e corect, dar trebuie
  **împărțit pe rânduri** (blocuri pe bucăți de rânduri + o a doua trecere peste sumele parțiale) înainte de orice măsurătoare de viteză a
  pasului. Previziunea „restul pasului ~15 %” din `C-160_P7_VITEZA_ESTIMARE.md` era valabilă doar cu un `colsum` împărțit; am scris asta acolo.
- **Generarea din același ciclu (MĂSURAT):** 16/16 simboluri identice cu procesorul, amprenta `0x46a07914` = referința `lipita-ln`,
  placa 1 976 µs (fc 487, proj 406, qkv 381 — ca în os-331), driverul 811, 351 simb/s cu driverul. Rândul „logit-urile identice cu os-322? NU”
  e cel deja lămurit (`0x37920000`, altă ordine a sumelor, de ~6 ori sub prag). Nimic de tăiat aici; `lipita-ln-32` implicită vine în os-350.
- **Ce tai mai departe:** nimic nou pentru placă. Următorul pe placă: **pasul de aur mic** (deja predat, `C-160_P6b_pasul-de-aur-pe-placa.md`).
  La noi intră în coadă **`colsum` împărțit pe rânduri** (SIMULARE întâi; ordinea sumelor se schimbă → model și margine noi, scrise înainte).

## os-346 `test-placa --i9 c135 --ln-32` (lista `lipita-ln-32` pe placă, ciclul automat 22 nov 2026, 10:06) — sesiunea 156
Sursa: `Cyber-hub/rezultate/os-automat/2026-11-22/ciclu-auto-20261122-1006.txt` + `gpu-jurnal-044.txt` (nucleul cu `cyber-gpu feb0819`;
#513 de două ori pe același text: întâi `lipita-ln` = implicitul, apoi `lipita-ln-32` = LayerNorm nou + toate înmulțirile pe sferturi).
- **Estimat** (sesiunea 151, mai jos, DEDUS): placa **≈ 1 070 µs** pe simbol, driverul ~830, jurnalul ~62 → ~1 960 µs = **~510 simb/s**;
  aceleași simboluri, abaterea față de lista lipită sub 1e-4.
- **Ieșit (MĂSURAT):** placa **1 071 µs** (la o microsecundă de estimare), driverul 825, jurnalul 60 → **1 986 µs = 503,5 simb/s** pe
  simbolul median (−1,3 % față de ~510); media rulării cu pauzele driverului 480,63 simb/s; doar lansările 935,9 simb/s. Corect: 16/16
  simboluri identice cu procesorul, față de `lipita-ln` max |d| 1,0e-5 (aceleași simboluri), 0 poziții lente. Windows/PyTorch 129,12 → **×3,9**.
- **Sferturile și LayerNorm-ul nou se adună exact (MĂSURAT, pe lansare):** fc 18,78 µs (ciclul 05:06: 18,72), qkv 14,69 (14,62),
  proj 11,90 (11,90), out 6,05 (6,08), logit 88,0 (87,9), atenția 20,77 (20,7), ln1/ln2 4,03/3,87 (os-331: ~4,0). Nicio interacțiune
  între cele două schimbări → socotelile pe lansare de până acum rămân bune (și estimarea #651 pe sferturi, ~293 simb/s, câștigă încredere).
- **Rândul „logit-urile identice bit cu bit cu os-322? NU” (cerut de OS să-l lămurim):** e așteptat. Pe sferturi fiecare fir adună alte
  bucăți ale rândului, deci altă ordine a sumelor → alți biți în ultimele zecimale. Abaterea față de procesor 1,74e-5 (`0x37920000`; os-322
  cu lista veche: 1,62e-5, `0x37880000`), ambele de ~6 ori sub pragul 1e-4. Amprenta de referință pentru `lipita-ln-32`: **`0x6c8501f0`**
  (de comparat cu ea, nu cu os-322, la rulările următoare ale acestei liste).
- **Unde e timpul acum (MĂSURAT, din 1 986 µs):** driverul 825 (**42 %**, cel mai mare bloc), înmulțirile 702 (35 %: fc 225, qkv 175,
  proj 142, logit 88, out 72), atenția 251 (13 %), LayerNorm 99, restul ~110. Deci următorul câștig mare e **pe procesor**, nu pe placă:
  dieta 2 (predată, `C-157_P3_dieta-doi.md`, driverul ~330–410) + argmax fără copie (`os-343`, încă netestat pe placă) →
  1 071 + ~330–410 + 60 ≈ 1 460–1 540 µs = **~650–685 simb/s** (DEDUS; Windows ×5,0–5,3).
- **Ce tai mai departe:** (1) propun **`lipita-ln-32` implicit** în `test-placa` (programele pe sferturi au trecut de două ori: 05:06 și
  10:06; `--lipita-ln` rămâne pentru comparație) și pentru `--k651`; (2) dieta 2 + argmax fără copie pe placă; (3) abia apoi înmulțirile
  pe rândurile scurte și atenția. Mărunt, rămas: rândul `kt lista` spune „87 lansări”, tabelul pe fel = 88.
- Nota pentru OS: `integration/cyber-universe/C-157_P2i_os-346_ln-32.md`. Nimic nou pentru placă.

## os-343 `test-placa --i9 c135 --k651` (Kronos #651 pe placă, primul test al modelului mare, 22 nov 2026, 07:36) — sesiunea 154
Sursa: `Cyber-hub/rezultate/2026-11-22-workstation-os343-k651/REZULTAT.md` + `rezultate/os-automat/2026-11-22/ciclu-auto-20261122-0736.txt`
(nucleul cu `cyber-gpu cf9e911`; lista lipită cu LayerNorm nou, **fără sferturi**).
- **Estimat** (`integration/cyber-universe/C-157_P4_k651-limite.md`, DEDUS): simbolul median **~5,1–5,6 ms** (~180–195 simb/s); „peste ~6 ms
  = ceva nu scalează cu straturile”; VRAM ~270 MiB, urcarea în 25 de bucăți, 172 de lansări pe simbol; simbolurile = ale procesorului.
- **Ieșit (MĂSURAT):** **5 161 µs** median (placa 3 747, driverul 1 317, jurnalul 62) = **194,34 simb/s**, la marginea bună a estimării;
  16/16 simboluri identice cu procesorul, 0 poziții lente, 25 de bucăți (206 817 280 octeți, 33,6 ms pe placă), 172 de lansări, amprenta
  de referință nouă `0x38b4270f`, abaterea maximă a logit-urilor 3,92e-5 (la #513: 1,74e-5; dublul straturilor, DEDUS normal). Procesorul
  (același test): 38,1 simb/s → placa ×5,1.
- **Placa scalează exact cu straturile (MĂSURAT, pe lansare):** fc 40,61 µs (la #513 în os-331: 487/12 = 40,58), proj 33,86 (33,75),
  qkv 31,74 (31,75), out 12,42 (12,33), atenția 20,70 (20,8), ln1/ln2 4,0/3,8 (LayerNorm nou), logit 186 (identic: același vocabular).
  Deci 24 de straturi = de două ori aceleași lansări; nimic nou nu frânează la modelul mare.
- **Driverul 1 317 µs (la #513 ~818–830):** crește cu numărul de lansări: din două puncte (88 → 172 de lansări) ies **~5,9 µs pe lansare +
  ~300 µs fix** (DEDUS). La #651 driverul e **26 %** din simbol azi și ar fi **~39 %** după sferturi → dieta 2 (predată, `C-157_P3_dieta-doi.md`)
  câștigă **mai mult la #651**: citirea înregistrărilor crește cu lansările (~173 în loc de 89, ~1,9 µs fiecare → ~−320 µs, nu ~−166).
- **Estimare pentru `#651` + `lipita-ln-32` (sferturi + LayerNorm nou, testul următor din REZULTAT):** pe lansare, din ciclul automat 05:06 (#513 pe sferturi,
  aceleași lățimi 768): 24 × (fc 18,72 + qkv 14,62 + proj 11,90 + out 6,08 + atenția 20,7 + ln 7,84) + logit 87,9 + rest 21 ≈ **2 030 µs**
  placa (−1 717); driverul tot ~1 317 (aceleași 172 de lansări) + jurnalul 62 → **~3,41 ms = ~293 simb/s** (DEDUS); cu dieta 2 (~−450 sau mai mult la
  #651) **~340**; cu argmax fără copie încă ~−150 → **~360 simb/s**. Peste ~3,8 ms ar însemna că sferturile nu scalează ca la #513.
- **Mărunt, de lămurit de OS (nu de noi):** citirea modelului de pe stick **29 s** pentru 197 MiB (~6,8 MB/s) și copierea pe procesor 774 ms
  (267 MB/s) — în afara ceasului; rândul „comparatie: placa ? simboluri/s (fara randul de viteza al nucleului)” la sfârșit nu ia cifra din `kt`.
- **Ce tai mai departe:** nimic nou pentru placă; ordinea de câștig la #651: (1) sferturile (`--ln-32`, gata în os-346), (2) dieta 2,
  (3) înmulțirile pe rândurile scurte + atenția (C-157_P2g). Nota pentru OS: `integration/cyber-universe/C-157_P2h_os-343_k651.md`.

## Ciclul automat 05:06 `test-placa --i9 c135 --sferturi` (cursa înmulțirilor pe sferturi, 22 nov 2026) — sesiunea 151
Sursa: `Cyber-hub/rezultate/os-automat/2026-11-22/ciclu-auto-20261122-0506.txt` + `gpu-jurnal-040.txt` (nucleul cu `cyber-gpu b157c7f`).
- **Estimat** (`C-155_cursa-pe-sferturi.md`, DEDUS): câștig mare pe fc / qkv / out / logit (acolo lucrau 24 din 32 de fire), **cel mai mic
  pe proj**; aceleași simboluri, abaterea față de lista lipită sub 1e-4.
- **Ieșit (MĂSURAT):** corect — 16/16 simboluri identice cu procesorul, față de lista lipită max |d| 1,14e-5 (`0x37400000`), față de
  procesor 1,62e-5; amprenta nouă `0x9d0d3c66`. **Toate cele 5 feluri de înmulțiri trec regula de 50 ‰:** fc +538, qkv +539, logit +527,
  out +510 și — **diferit de estimare — proj +648 ‰, cel mai mare câștig**. Placa 2 419 → **1 516 µs** pe simbol (−903 µs, +373 ‰);
  simbolul cu driverul 2 397 µs = **~417 simb/s** pe rularea pe sferturi (media celor două rulări 351,90 nu e cifra listei noi).
- **De ce proj a câștigat cel mai mult** (DEDUS din octeții citiți): pe sferturi, proj (3 072 de intrări, 12 sferturi pe fir) merge cu
  ~223 GB/s, fc / qkv / logit cu ~136–161 GB/s, out cu ~109 GB/s. Deci frâna rămasă nu mai e „24 din 32 de fire”, ci **rândurile scurte**:
  la 768 de intrări un fir face doar 3 sferturi → prea puține citiri în zbor pe grup de 32. A doua ipoteză din notă e cea care rămâne.
- **Lista aleasă de Cyber-GPU** (cerută de rândul `kt sferturi`): **toată pe sferturi**, iar cu LayerNorm-ul nou (implicit din os-336)
  împreună → lista nouă **`lipita-ln-32`** (`TokenList::FusedLn32`, T162 în SIMULARE). „Mixtă” ar fi aceeași listă: felurile care n-au
  trecut (ln, atenție, embed, copiere) au oricum același program în ambele.
- **Estimare pentru `lipita-ln-32`** (DEDUS): placa 1 516 − 544 (LayerNorm vechi) + 98 (LayerNorm nou, os-331) ≈ **1 070 µs**; cu driverul
  830 + jurnalul 62 → ~1 960 µs = **~510 simb/s**; cu argmax fără copie (os-343) driverul ~680 → **~550 simb/s** (Windows/PyTorch 129 → ×4,3).
- **Ce tai mai departe:** după `lipita-ln-32`, pe placă rămân înmulțirile ~700 µs (65 %) și **atenția 251 µs (23 %; ~21 µs pe lansare la
  doar ~20 de poziții)** → (1) înmulțiri cu mai multe citiri în zbor pe rândurile de 768 (2–4 rânduri pe grup sau citirile celor 3 sferturi
  puse înainte de calcul), (2) atenția; driverul (830 µs) rămâne cel mai mare bloc → dieta 2 (predată, ~330–410 µs).
- **Mărunt, rămas de lămurit de OS:** rândul `kt lista` spune tot „87 lansări”, tabelul pe fel = 88 (12 × 7 + 4).
- Nota pentru OS: `integration/cyber-universe/C-157_P2g_sferturi.md`. Nimic nou pentru placă (programele au rulat deja toate trei).

## os-340 `test-placa --argmax-placa` (argmax pe placă, proba, 22 nov 2026, 02:51) — sesiunea 149
- **Estimat** (`C-155_argmax-pe-placa.md`, DEDUS): cele 4 cuvinte ale programului = `argmax_result` al logit-urilor copiate la fiecare
  poziție (prag 0); lansarea argmax ~5–15 µs; în proba cu copie, o lansare în plus pe simbol (placa +~10 µs).
- **Ieșit:** **18/18 poziții identice**, simbolul plăcii = al procesorului; lansarea argmax **15,04 µs** median (marginea de sus a
  estimării: un singur grup de 256 de fire citește 64 KiB ≈ 4,4 GB/s); placa 1 977 → **1 989 µs** (+12, cât lansarea minus suprapunerea);
  16 simboluri identice, amprenta `0x46a07914` = aceeași, 0 poziții lente; **media cu driverul 338,91 simb/s** (os-336: 347,17).
- **De ce media a scăzut cu ~8 simb/s — doar proba, nu programul:** driverul 814 → 865 µs (+51), iar rândul „argmax” 85 → **158 µs**
  (+73). În OS (`gpu_i14.rs`, P6-123) ceasul „argmax” cuprinde acum și (a) **o a doua citire a întregii arene** (64 KiB + 16 octeți, ca
  să ia cei 16 octeți de la capăt; citirea logit-urilor singură = 49 µs) și (b) **verificarea** `agrees()` = încă o trecere de
  `argmax_result` pe 16 384 de logit-uri (~25 µs). 49 + 25 ≈ 73 µs: exact diferența (DEDUS din cod + cifre).
- **os-336** (LayerNorm nou implicit, 00:06), pe scurt: 347,17 simb/s față de ~348 estimat; placa 1 977, driver 814 — ca os-332.
- **Ce tai mai departe — argmax fără copie** (punctul 4 al notei): dispar copia (11,9 µs pe placă), citirea logit-urilor (49 µs) și
  argmax-ul pe procesor (85 µs); rămân lansarea argmax (15 µs) și citirea a **doar 16 octeți** (~1–2 µs). Estimare (DEDUS): placa
  ~1 980 µs, driverul ~680 µs, simbolul ~2 770 µs → **~360 simb/s** (os-336: 347). Proba: simbolurile = ale procesorului (oracolul
  după placă, ca acum), amprenta nu mai există pe placă (logit-urile nu se mai citesc) → oracolul pe procesor face amprenta.
- **Mai târziu, mărunt:** argmax pe mai multe grupuri (o a doua lansare mică) ar coborî 15 → ~6 µs; câștig ~9 µs, nu merită acum.
- Nota pentru OS: `integration/cyber-universe/C-157_P2f_os-340_argmax.md`. Nimic nou pentru placă.

## os-332 `test-placa --ln-nou` + încălzirea poziției 0 (21 nov 2026, ~19:16) — sesiunea 142
- **Estimat** (`C-157_P2c_pozitia-0.md` + os-331, DEDUS): 0 poziții lente; ~348 simb/s și la prima rulare cu LayerNorm nou; încălzirea
  scrie ~11 264 + 1 064 = 12 328 de cuvinte pe prima listă; nimic schimbat în amprentă.
- **Ieșit:** `pozitiile lente: 0 din 18`, cea mai lentă poziție 2 494 µs (lista veche) / 2 049 µs (LayerNorm nou), față de ~12 ms în os-329;
  **media cu driverul 351,39 simb/s** (estimat ~348, +1 %); amprenta `0x46a07914` ca în os-331; 16 simboluri identice cu procesorul.
- **Încălzirea:** 15 553 µs în afara ceasului, **12 687** cuvinte pe prima listă — cu 359 (+3 %) peste socoteala mea; mărunt, probabil
  cuvintele de început ale pushbuffer-ului pe care nu le-am numărat; nu schimbă nimic în ceas.
- **Driverul 818 µs** (os-331: 819): neschimbat, cum era de așteptat — dieta 2 încă nu e în OS. Placa 1 977 µs, la fel ca os-331.
- **Ce tai mai departe:** neschimbat față de os-331 — dieta 2 (driver ~818 → ~330–410 µs) și cursa înmulțirilor (fc + proj + qkv + logit
  = 1 458 din 1 977 µs). Cu dieta 2: ~2,47 ms pe simbol → **~405 simb/s** (DEDUS). Nimic nou de cerut OS-ului din acest test.

## os-331 `test-placa --ln-nou` (LayerNorm nou, 21 nov 2026, 17:26) — sesiunea 140
- **Estimat** (`C-155_LayerNorm-nou.md`, DEDUS): ~3–5 µs pe lansare în loc de ~21,8 → **~0,45 ms mai puțin pe simbol**; aceleași simboluri.
- **Ieșit:** **~4 µs pe lansare** (ln1 48 + ln2 46 + ln_final 4 = 98 µs pe simbol); placa 2 423 → **1 978 µs** (−445 µs) în aceeași pornire,
  același text; aceleași 16 simboluri, ln nou față de vechi max |d| `0x36e80000` = 6,9e-6. Ca estimat (DEDUS → MĂSURAT). Și pe ceasul
  soneriilor din `gpu-jurnal-030.txt` se vede la fel: 3,39 ms între sonerii în prima rulare, 2,96 ms în a doua (−0,43 ms).
- **Media 348,47 simb/s e a doua rulare** (lista lipită era deja scrisă de prima, deci umbra plină). Prima rulare (LayerNorm vechi) are încă
  poziția 0 rece: ~31 ms între „generarea incepe” și prima sonerie (os-329: ~25 ms pregătire + ~7 ms sloturile). Pentru o singură rulare
  rece cu LayerNorm nou: ~51,7 + ~8 ms = **~300 simb/s**; cu încălzirea (os-332) → **~348 și la prima rulare** (DEDUS). Estimarea mea
  din os-329 (încălzirea, fără LayerNorm) era ~297; cu LayerNorm adăugat ar fi dat ~342 → ieșit 348, +2 %.
- **„Logit-urile identice cu os-322? NU” (de văzut de noi):** e de așteptat, nu e o eroare. LayerNorm nou adună media și varianța pe 256 de
  fire, în copac, nu pe rând → rotunjiri în altă ordine. Abaterea maximă față de procesor: `0x37920000` = 1,74e-5 (os-322: `0x37880000`
  = 1,62e-5), de ~6 ori sub pragul 1e-4; simbolurile aceleași. Amprenta de referință de acum: `0x46a07914`.
- **Driverul 819 µs (os-329: 844):** −25 µs, în zgomot; dieta 2 încă nu e în os-331.
- **De lămurit de OS (mărunt):** rândurile „viteza (placa): 36 simboluri … cu driverul 299,83” și comparația „față de os-322” amestecă cele
  două rulări (vechi + nou); cifra curată a rulării noi e în rândul `kt` (2 885 µs, 346,6 / 348,5 simb/s). Iar „87 lansări” din rândul
  `kt lista` nu se potrivește cu tabelul pe fel (12 × 7 + 4 = 88). Nota: `integration/cyber-universe/C-157_P2d_os-331.md`.
- **Ce tai mai departe:** înmulțirile sunt acum **74 % din placă** (fc 487 + proj 405 + qkv 381 + logit 186 = 1 459 din 1 978 µs) →
  cursa pe sferturi (C-155 p.4, predată) e următorul test cu cel mai mare câștig pe placă; driverul (819 µs, 28 % din simbol) → dieta 2
  (predată, ~330–410 µs). Cu încălzirea + dieta 2: ~2,45 ms pe simbol → **~400 simb/s** (DEDUS), înainte de cursă.

## os-329 `test-placa` (pozițiile lente + lista lipită implicită, 21 nov 2026, 15:56) — sesiunea 138
- **Estimat** (`C-157_P2b_pozitii-lente.md`, cu P3 după): media ≈ mediana, **~280 simb/s** cu driverul; pozițiile lente → 0.
- **Ieșit:** **262,04 simb/s** în medie (suma 68 690 µs / 18), mediana 3 338 µs (299,6), placa 413,56 simb/s, 130 de cuvinte, amprenta
  `0xda9ae084` (identic bit cu bit), nicio scriere pe stick. Scrierea pe stick (poziția 15/16) a dispărut de tot, ca estimat.
- **Diferit: poziția 0 a rămas lentă** (12 250 µs; driverul 9 863). Din `gpu-jurnal-028.txt` (ceasul rândurilor): fereastra sloturilor
  `0x1b2d` stă deschisă **7,43 ms** la poziția 0 (~0,04 ms la celelalte), iar a pushbuffer-ului `0x10e6` **0,69 ms** (~0,07). Cauza
  (DEDUS, T149): `forget_range` a reparat uitarea, dar **umbra lansărilor pornește goală**, iar scara a folosit lista simplă — deci
  primul simbol lipit scrie toate cele **11 264** de cuvinte ale celor 88 de sloturi + **1 064** de pushbuffer. Același cost pe cuvânt
  (0,66 µs, scriere + recitire) explică și pushbuffer-ul (702 față de 690 µs) și poziția 0 rece din os-324 (16 555 față de ~16 900 µs).
  Am estimat greșit că `forget_range` ajunge: am uitat că lista lipită n-a mai fost scrisă niciodată înainte de poziția 0.
- **Driverul median 844 µs față de 759 (os-328), +85 µs:** NECUNOSCUT (dieta 2 nu e încă în os-329; candidații: inelul de rânduri
  ținute în memorie, rândurile `kt` pe poziții). Mic; dieta 2 lucrează exact în zona aceea.
- **Încă două lucruri mărunte:** (a) simbolurile 9–17 au ~0,12 ms mai mult decât 1–8 (sonerie la sonerie 3,31 → 3,44 ms) — atenția
  crește cu lungimea textului (DEDUS, normal); (b) ~25 ms între nota „generarea incepe” și prima scriere a poziției 0, în afara ceasului
  (nu intră în medie; DEDUS: pregătirea textului, între care cache-urile verificării pe procesor, ~38 MB puse pe zero).
- **Ce tai mai departe:** **încălzirea** — poziția 0 trimisă o dată înainte de ceas (aceleași lansări, același simbol; cache-urile K/V
  ale poziției 0 se rescriu cu aceiași biți). Poziția 0 cronometrată scrie atunci doar 89 + 3 cuvinte (T149) → −~8,1 ms → suma ~60,6 ms,
  **~297 simb/s** (DEDUS). Cu dieta 2 după: ~340. Nota pentru OS: `integration/cyber-universe/C-157_P2c_pozitia-0.md`.

## os-328 `--lipit` (P3, 21 nov 2026, 12:56) — sesiunea 135
- **Estimat** (`C-155_P3-gata-de-legat.md`, DEDUS): ~4,0 → **~3,6 ms** pe simbol (~280 simb/s); 88 de lansări; logit-uri identice bit cu bit cu lista simplă; ~130 de cuvinte VRAM pe simbol.
- **Ieșit:** **3 284 µs** (304,5 simb/s) — cu ~320 µs mai bine decât estimarea; 88 de lansări, 130 de cuvinte; amprenta `0xda9ae084` = cea din os-324 (lista simplă) → identic bit cu bit, 16/16 simboluri.
- **Unde a fost estimarea prea prudentă:** am pus tot câștigul pe placă. De fapt **driverul** a câștigat cel mai mult: 1 347 → **759 µs** (−588; 108 lansări mai puțin = mai puține QMD-uri și cuvinte de scris), iar **placa** doar 2 624 → 2 421 µs (−203).
- **Pe placă, pe fel (os-324 → os-328, median pe simbol):** cele 4 feluri mici (adunări/bias, ~107 µs, 48 de lansări) au dispărut; programele lipite au plătit coada: fc 471 → 486, qkv 368 → 381, proj 400 → 406, out 144 → 148 (+38 µs). Restul ~134 µs = pauzele dintre lansări (~1,2 µs pe lansare scoasă). ln1/ln2/atenție/logit neschimbate (±3 µs).
- **Prima rulare (lista simplă) la 17,57 simb/s** (de lămurit de OS, punctul 2 din REZULTAT): în `gpu-jurnal-026.txt`, după cele 14 sonerii ale urcării, singura pauză mare din prima generare e de **0,90 s între soneria 30 și 31 = înainte de poziția 16** — exact locul scrierii jurnalului pe stick „la 16 simboluri” (aceeași poziție ca cea lentă din lista lipită, 77 ms). E mai lungă pentru că prima golire duce și rândurile urcării greutăților (DEDUS din ora rândurilor; ora e a scrierii în memorie, nu a plăcii). Încă o pauză de 0,30 s după soneria 32 = sfârșitul primei generări. Concluzie: tot scrierea pe stick (P6-112 o repară); placa și driverul nu au nimic.
- **Ce tai mai departe:** înmulțirile (fc + proj + qkv + logit = **1 459 µs = 60 %** din placă) merg la **~60–75 GB/s**, adică ~12 % din 608 GB/s (copia VRAM: 282 GB/s în același test). Acolo e următorul câștig mare → C-155 punctul 4 (cursa înmulțirilor). Apoi ln1 + ln2 (522 µs) → LayerNorm nou (gata de legat). Driverul (759 µs) e acum sub placă; ținta 0,5 ms rămâne C-157 punctul 3.
- **De lămurit de OS (nu de noi):** rândul `kt lipit fata de simplu` (propus în nota P3) nu e în jurnal; amprenta ajunge pentru proba de azi.

## os-323 `--flr-proba` (21 nov 2026, 08:32) — sesiunea 132
- **Estimat** (`C-151_reset-flr.md`, sesiunea 122): placa răspunde după FLR în 100 ms + câteva citiri (limita 1 s); WPR2 coborât; al doilea lanț trece fără repornire („varianta 1”).
- **Ieșit:** placa răspunde după **131 ms**, programul de pornire al plăcii gata la **206 ms**, WPR2 = 0, al doilea lanț trece **tot** (I6…I14, turul de citire). Varianta 1 dovedită; rezerva (Booter Unload) nu mai e nevoie.
- **Diferit:** doar cele ~31 ms peste cele 100 ms de așteptare (citiri până răspunde); limita de 1 s are marjă de ×7. Nimic de schimbat la noi.
- **De lămurit de OS (nu de noi):** cele „11 scrieri refuzate în afara gardului” la al doilea lanț — gardul nostru pentru rescrierea configurației cere doar 0x04–0x3c, Device/Link Control și MSI; dacă cele 11 sunt în afara lor, sunt ale adaptoarelor OS, nu ale `gpu_pci::flr`.
- **Ce tai:** nimic. Deschide calea „mai multe teste într-o pornire” (C-158 punctul 5).

## os-323 `--tablou` (21 nov 2026, 08:21) — sesiunea 132
- **Estimat** (`C-151_tablou-rusd.md`): cele 3 comenzi acceptate, `sections` = 0x7f, 30–50 °C în repaus, P8. **Ieșit:** exact așa (35,8 °C, P8, 225/405 MHz, 25,2 W, limita 290 W, 0x7f).
- **Motivele încetinirii = 0x1:** e bitul `RUSD_CLK_THROTTLE_REASON_GPU_IDLE` din `cl00de.h` (570.144) — „placa e în repaus”, nu o încetinire. Deja în `gsp::rusd::reason::GPU_IDLE`.
- **Diferit: oprirea zonei comune refuzată** (`INIT(0)` → `0x8000001f` = `NV_ERR_INVALID_ARGUMENT`, plus un ASSERT intern). Cauza (DEDUS, două surse): driverul NVIDIA 570.144 **nu trimite niciodată** `INIT(0)` — `gpuDestroyRusdMemory` doar eliberează memoria, iar singurul care trimite `INIT` e `_gpushareddataInitGsp`, mereu cu o adresă reală. Comentariul „0 to de-initialize” din antet nu e urmat de GSP-RM 570.144.
- **Ce tai:** oprirea fără `INIT(0)` — `gsp::rusd::rusd_stop_poll_only` și `dashboard_at_end_keep_page` (doar adăugări; T140h în SIMULARE: 4 comenzi în loc de 5, niciun `INIT(0)`). Pagina rămâne rezervată până la FLR sau repornire (așa era oricum după refuz). Pe placă: **o comandă mai puțin, nimic nou**; un ASSERT mai puțin pe rulare (așteptat).
- **P0 înainte de viteza maximă (ieri P5):** NECUNOSCUT. Ipoteză (DEDUS): placa nu coboară încă din P0 după lucrul abia terminat al lanțului (I10–I14 rulează chiar înainte); tabloul arată P8 la 1,7 s după. Proba ieftină: aceeași citire a tabloului și **înainte** de lanț (C-151 punctul 2, în coadă).
- **Viteza (testul obișnuit):** 94,44 simb/s, driverul 3 087 µs/simbol — ca os-322 (94,84). Estimarea pentru dietă rămâne ~1,3 ms driver; se vede în `os-324`.

## os-462 + os-465 (26 nov 2026, 15:48–16:36) — sesiunea 254
- **os-462, tc16 în pasul mixt #513 (lot 2):** aceleași numere ca oracolul mixt ✔; câștig **−6,66 ms** (9,17 → 2,51 ms), prezis 2,2–4,0 — prea prudent. Cauza (DEDUS, T264): proba izolată rulează tc16 la treapta de 16 KiB (1 bloc pe SM), pasul la 64 KiB (3–4 blocuri); F32 `_3` are 1 bloc în ambele.
- **os-465, P2zq:** v4 = v4pf = tc16 bit cu bit (4 / 4 aliniate); v4 **0,52 ms (×1,67)** în banda 0,35–0,55; v4pf 0,64 — pierde chiar la 1 bloc pe SM. **Tai v4pf**; în pas v4 (3 blocuri) față de tc16 (3–4) câștigă doar ~0,2–1 ms (DEDUS).
- **os-465, P2zr:** W1–W5 toate confirmate; o barieră 2,03 µs la G = 384, 3,11 µs la G = 768. **G ales pentru `token32`: 384** (plafon ~0,16 ms pe simbol); 768 tăiat.
- **Rândul lot 6 sărit:** ordinea opțiunilor (`--atentie-tc16` trebuie după `--doua-locuri`). Nota întreagă: `docs/C-157_OS462_OS465.md`.
