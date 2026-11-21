# C-157 punctul 2 — după fiecare test pe placă: ce am estimat, ce a ieșit, ce tai mai departe

Fiecare test nou pe placă primește aici 5–10 rânduri (cel mai nou sus). Sursele: `Cyber-hub/rezultate/…/REZULTAT.md` și jurnalele brute.

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
