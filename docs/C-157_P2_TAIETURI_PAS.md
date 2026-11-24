# C-157 punctul 2 — al doilea rând de tăieturi pentru pasul întreg de antrenare (plan + predicții, scrise ÎNAINTE de cod)

Sesiunea 198 (24 nov 2026, cloud, fără placă). Continuă `docs/C-157_P2_OS386_PASUL_INTREG.md` (J1–J3): după ce rândurile
jurnalului nu mai merg pe stick la fiecare pas, din pasul întreg rămân placa (3,65 ms, os-386) + munca driverului (~1,2 ms, J2)
+ procesorul. Aici: cât din cei ~1,2 ms se poate tăia **cu unelte care există deja și merg deja pe placă în generare**
(`kronos --gpu` cu dieta 1 și 2, os-322 … os-353), fără nimic nou trimis la placă.

## Ce face azi `train_mini_resident` (citit în `cyber-universe/kernel/src/drivers/gpu_i14.rs`, doar citit)
La fiecare pas: (1) pune pe zero toate cele 76 de înregistrări (304 cuvinte, cu recitire); (2) scrie tot pushbuffer-ul
(`sequence_words_fresh(75)` = 908 cuvinte) prin `run_records` fără umbră; (3) citește toate cele 76 de înregistrări (304 cuvinte);
(4) urcă lotul (321 de cuvinte) și citește pierderile + norma (65 de cuvinte). Generarea face (1)–(3) altfel de la os-322 / os-353:
umbra pushbuffer-ului (`kronos_fast::ShadowedMemory`), înregistrările puse pe zero o singură dată (`rec_zeroed`) și doar prima +
ultima citite (`kronos_diet2::RecordPlan::DIET`, `ends_span`), plus așteptarea planificată (`WaitPlan`).

## Tăieturile (aceleași patru unelte, aplicate pasului de antrenare)
- **A. Umbra pushbuffer-ului:** de la un pas la altul se schimbă doar valorile înregistrărilor (prima valoare crește la fiecare
  trimitere); adresele QMD-urilor rămân (zona de control e aceeași, `pack` dă aceleași adrese).
- **B. Înregistrările pe zero o singură dată:** o înregistrare veche are valoarea trimiterii de dinainte și nu poate trece drept
  cea nouă (T141b, aceeași regulă).
- **C. Doar prima + ultima înregistrare** (8 cuvinte), toate doar la pașii care hrănesc profilul (`RecordPlan::DIET.reads_all`:
  pașii 0, 1 și fiecare al 8-lea).
- **D. Așteptarea planificată** (`WaitPlan`): ~50 µs în medie (DEDUS din generare; aici nu se poate măsura).

## Predicțiile (înainte de cod; T206, SIMULARE pe `gpu_sim::cmdproc`, forma lui `antrenare mini`, 10 pași, 10 loturi diferite)
- **K1 (numere):** cu A + B + C, pierderile, normele și greutățile după 10 pași sunt **bit cu bit** aceleași ca drumul de azi.
- **K2 (umbra):** pasul 0 scrie tot pushbuffer-ul (908); de la pasul 1, pe pas se scriu exact **76** de cuvinte (valorile celor
  76 de înregistrări), restul de 832 sunt lăsate (știute deja). Conținutul pushbuffer-ului în memorie = cel scris întreg.
- **K3 (înregistrările):** pe zero 304 cuvinte la pasul 0, apoi 0. Citite: 304 la pașii 0, 1, 8; **8** la ceilalți 7.
  La pașii citiți întregi, `ends_span` = suma duratelor `analyze_sequence` (același timp al plăcii).
- **K4 (siguranța):** fără punerea pe zero, un pas oprit la jumătate (pushbuffer tăiat după lansarea 40) e prins: ultima
  înregistrare are valoarea pasului de dinainte → așteptarea nu se termină, `ends_span` refuză, `records_seen` = 41.
- **K5 (aritmetică, DEDUS pentru placă):** pe pas, de la pasul 2: scrise 76 + 321 (lotul) = **397** cuvinte (azi ~1 533), citite
  8 + 65 = **73** (azi 369) → la 0,66 / 0,5 µs pe cuvânt **~0,3 ms** (azi ~1,2 ms): **−0,9 ms pe pas**. Cu lotul urcat în
  paralel (P10, deja predat) rămân ~0,09 ms. Pasul întreg estimat după remediul jurnalului + tăieturi: 3,65 + ~0,3 + procesorul
  ≈ **4–4,5 ms** (~14 000–16 000 simb/s la `antrenare mini`; azi 173). NU se poate testa aici.

## Ce NU face
Nicio lansare, niciun program, niciun registru, nicio valoare nouă pe placă: aceleași cuvinte ajung în pushbuffer, aceleași
înregistrări le scrie placa. Codul e al OS-ului (`gpu_i14.rs`, `run_records_d2` există deja în `gpu_i11.rs`); de la noi:
proba T206 și nota pentru OS. Nimic nu rulează pe RTX 3070 Ti.

## Rezultatul (după cod; T206, `crates/nvidia-ga104/tests/sim_c157_p2_taieturi.rs`, SIMULARE)
- **K1 — trece:** pierderile, normele și greutățile după 10 pași **bit cu bit** aceleași ca drumul de azi; pushbuffer-ul din
  memorie identic la sfârșit.
- **K2 — trece exact:** pasul 0 scrie 908 cuvinte, apoi **76** pe pas (832 lăsate, știute deja).
- **K3 — trece exact:** înregistrările pe zero doar la pasul 0; citite 304 la pașii 0, 1, 8 și **8** la ceilalți; timpul plăcii
  din cele două capete = suma duratelor la fiecare pas (ceasul modelului `gpu_sim`, cu un cost fix ales de test — nu e o măsurătoare).
- **K4 — trece:** pasul 2 oprit după lansarea 40 (fără punere pe zero): ultima înregistrare `0x714b` (a pasului 1), așteptat
  `0x724b` → așteptarea nu se termină, `ends_span` și `analyze_sequence` refuză, `records_seen` = 41.
- **K5 — aritmetica:** azi 1 533 scrise + 369 citite ≈ **1 196 µs**; cu tăieturi 397 + 73 ≈ **299 µs**; cu lotul în paralel (P10)
  ≈ **87 µs**. Pe placă: DEDUS, nemăsurat.
- Nimic n-a ieșit altfel decât prezis. `tools/check.sh`: 983 de teste, PASS.
