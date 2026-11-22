# C-157 punctul 2 — după fiecare test pe placă: ce am estimat, ce a ieșit, ce tai mai departe

Fiecare test nou pe placă primește aici 5–10 rânduri (cel mai nou sus). Sursele: `Cyber-hub/rezultate/…/REZULTAT.md` și jurnalele brute.

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
