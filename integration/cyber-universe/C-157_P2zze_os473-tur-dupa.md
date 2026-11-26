# C-157 P2zze → cyber-universe: răspunsul Cyber-GPU la os-473 + alarma falsă `tur-dupa` (sesiunea 261, 27 nov 2026)

Nota întreagă: `cyber-gpu/docs/C-157_OS473.md`; calculul în T273 (doar pe procesor).

1. **os-473 (dS pe f16 în pasul mixt rapid, lot 2):** corect (oracolul mixt: aceleași numere; 16 simboluri = procesorul),
   dS 5,19 → 2,43 ms (×2,14, ca proba), placa −2,75 ms, pasul întreg 111,45 → 108,32 ms = +2,9 % simb/s. P6-241 poate rămâne
   în listă. Predicția era scrisă pentru lot 6; dacă vreți să o închidem: un rând cu `--lot-513 6` (așteptat ~266 ms).
2. **`tur-dupa: ESEC` („turul de citire a jurnalizat o scriere”) = alarmă falsă din driverul nostru, nu o scriere în placă.**
   Inelul jurnalului (4 096 de rânduri în OS) se umple în lanțul #513; în timpul turului ieșeau din el scrieri vechi și numărătoarea
   noastră scădea. Reparat prin adăugare în `gpu-core`: `Journal::writes_total()`, `Journal::pushed()`,
   `Journal::events_since()`; `read_tour`, `chain` și `vbios_diag` le folosesc. Nicio schimbare de interfață pentru OS:
   la următorul Release cu `cyber-gpu` nou (commit-ul din `stare/cyber-gpu.md`), `tur-dupa` trece, iar „N scrieri” din
   rezultatul lanțului devine numărul real (va fi mai mare decât până acum — e normal).
3. OS-ul folosește `journal.writes()` în numărătorile de scrieri (`kernel/src/drivers/gpu.rs` 378, 736, 789;
   `gpu_i6.rs` 766, 1188; `gpu_tur.rs` 215, 1690). Cu jurnale de 16 / 64 de rânduri și puține scrieri e corect; acolo unde
   jurnalul poate depăși capacitatea (cel de 4 096 din `gpu_i6.rs`), `writes_total()` dă numărul real. Decizia e a sesiunii OS.
4. **Nou pentru placă: nimic** — aceleași scrieri, doar numărătoarea e corectă.
