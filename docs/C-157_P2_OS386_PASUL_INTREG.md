# C-157 punctul 2 — os-386: de ce pasul întreg e 370 ms când placa face 3,65 ms (plan + predicții, scrise ÎNAINTE de cod)

Sesiunea 197 (24 nov 2026, cloud, fără placă). Surse: `Cyber-hub/rezultate/2026-11-24-workstation-os386-arena-vram/REZULTAT.md`,
`rezultate/os-automat/2026-11-24/ciclu-auto-20261124-0237.txt` (rândurile 87–104) și `gpu-jurnal-069.txt` (rândurile 4177–4330),
codul OS-ului citit (doar citit): `cyber-universe/kernel/src/drivers/gpu_i14.rs` (`train_mini_resident`), `gpu_i11.rs`
(`run_records`), `gpu_i10.rs` (`kick_and_wait`), `gpu_i6.rs` (`live_line`, `GenHold`).

## Ce am estimat (C-160 P8, R4)
Placa 1–5 ms pe pas → **ieșit 3,65 ms** (în interval). Pasul întreg „cu câteva ms peste placă” → **ieșit ~370 ms**. Ratat de ~100 de ori.

## Ce citesc în jurnal (MĂSURAT, din ora rândurilor)
Fiecare pas lasă în `gpu-jurnal-069.txt` exact **22 de rânduri**, toate scrieri BAR0 la registrele ferestrei / TLB / soneriei și notele lor: golirea
(notă + 5 mutări ale ferestrei), invalidarea TLB (notă + 3), soneria (notă + 1), așteptarea (notă + 3), citirea pierderilor (notă + 2),
citirea normei (notă + 2). Între două rânduri trec **~16,5 ms**
(nu µs). Fereastra fiecărui pas în jurnal: 349, 347, 349, 351, 353, **646**, 362, 364, 388, 344 ms — aceeași formă ca „pasul întreg”
din raport (347, 364, 366, 368, 370, **664**, 379, 382, 404, 360), inclusiv vârful de la pasul 6. 22 × 16,5 ms ≈ 363 ms.

## Ipoteza (DEDUS din cod)
În afara generării, OS-ul trimite **fiecare** rând al jurnalului cu `Flush::Now`: tipărit pe portul serial și pe ecran (~4 ms, os-308)
și scris + golit pe stick (o scriere USB pe rând). În generare (`kronos --gpu`) aceleași rânduri sunt „de rutină”
(`journal.set_routine(Some(kronos_fast::gen_routine(&GEN_NOTES)))` + `GenHold` + `journal.flush_live()` o dată pe simbol): stau în
memorie și ajung pe stick la final. `train_mini_resident` **nu** pornește această porțiune de rutină → ~350 ms pe pas în scrieri pe stick.
Același lucru explică și os-380 (286 ms pe pas cu arena în RAM): nu urcările de 31 MB, cum am scris în P8, ci tot rândurile.

## Predicțiile (înainte de cod)
- **J1 (SIMULARE, jurnalul):** cele 22 de rânduri ale unui pas, trecute de 10 ori prin `Journal` cu o destinație numărată:
  (a) fără porțiune de rutină → **220** rânduri `Now` (= 220 scrieri pe stick); (b) cu `gen_routine(GEN_NOTES)` + `flush_live()` după
  fiecare pas → **0** `Now`; (c) cu rutina dar **fără** `flush_live()` → un `Now` de fiecare dată când se adună 64 de rânduri ținute
  (`GEN_MAX_PENDING`) = **3** în 10 pași (rândurile 65, 130, 195) → trei scrieri pe stick în mijlocul pașilor; deci `flush_live()` pe pas
  e obligatoriu, nu opțional.
- **J2 (numărătoare, forma lui `antrenare mini`, planul rezident):** pe pas driverul scrie în VRAM (cu recitire) ~1 500 de cuvinte
  (pushbuffer-ul `sequence_words_fresh(75)` = 908, înregistrările puse pe zero 4 × 76 = 304, lotul 1 284 B = 321) și citește
  ~370 (înregistrările 304 + pierderile 64 + norma). La 0,66 µs pe cuvânt scris și ~0,5 µs pe cuvânt citit: **~1,2 ms**.
- **J3 (DEDUS, pentru placă; NU se poate testa aici):** cu rândurile ținute în memorie, pasul întreg = 3,65 ms placa + ~1,2 ms cuvinte
  + procesorul (planul lotului, lista, împachetarea; măsurat pe gazdă, doar orientativ) ≈ **5–6 ms = ~11 000–13 000 simb/s** la
  antrenare (azi 173). Al doilea rând de tăieturi, după: umbra pushbuffer-ului (doar cuvintele care se schimbă), înregistrările doar
  prima + ultima, lotul în paralel (P10, deja predat) → ~4 ms.

## Ce NU face
Nu schimbă nicio lansare, niciun registru, nicio valoare scrisă pe placă: doar **când** ajung rândurile jurnalului pe stick. Codul e al
OS-ului (`gpu_i14.rs`); de la noi: diagnosticul, proba J1/J2 și nota. Nimic nu rulează pe RTX 3070 Ti.
