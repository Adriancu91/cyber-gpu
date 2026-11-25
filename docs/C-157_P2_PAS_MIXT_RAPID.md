# C-157 p.2 / nota centrului 14:40 punctul 1 — pasul mixt cu programele `mm_tc` rapide și treapta 64 KiB (SIMULARE)

Sesiunea 234, 25 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE** sau **DEDUS**: nimic nu a rulat pe RTX 3070 Ti.

## De ce
Ordinea centrului (25 nov 14:40): (1) `mm_tc` rapid (treapta + citirile în avans) → **apoi pasul mixt #513** (toate înmulțirile pe
`mm_tc`). Programele rapide există (`mm_tc_pf`, pe placă 2,75 TFLOP/s în os-437; `mm_tc_v4`, os-438 de rulat), dar
`Plan::step_mixed` + `pack_mixed(_gelu)` împachetează **doar vechiul `mm_tc`** (1,03 TFLOP/s pe placă, os-423), la treapta unui
singur bloc (32 KiB). Deci azi `--mixt` pe #513 ar rula cu cel mai lent program. Lipsește legătura.

## Ce se face (doar adăugări în `nvidia_ga104::train_step`; nimic schimbat în ce există)
- `TcPrograms { tc, pf, v4, gelu }` — unde a încărcat OS-ul fiecare program din familia `mm_tc` (0 = neîncărcat).
- `TcPick` + `tc_pick(&MixedLaunch, &TcPrograms)` — regula: `MixedLaunch::Tc` → `mm_tc_v4` dacă e încărcat și `mm_tc_v4::fits`,
  altfel `mm_tc_pf` dacă e încărcat, altfel `mm_tc`; `TcGelu` → `mm_tc_gelu` (nu are încă variantă rapidă).
- `pack_mixed_fast(launches, prog_va, &progs, step_bytes: Option<u32>, zone_va)` → zona, adresele QMD și alegerea pe fiecare
  lansare (pentru rândurile din raport). Cu `Some(treapta)`: **toate** lansările (F32 și `mm_tc*`) primesc aceeași treaptă
  (`MIN = TARGET = treapta`, ca `pack_with_sm_step` / `--treapta-pas` din os-431). Aceleași cbuf-uri, aceleași mărimi de zonă.

## Predicțiile, scrise ÎNAINTE de cod (se raportează oricum ar ieși)
1. **Aceiași biți (SIMULARE):** pe `antrenare mini` (5 pași, planurile simplu / plăci3 / lipit), pasul mixt împachetat rapid
   (v4 / pf, treapta 64 KiB) dă **aceleași pierderi, același gradient la pasul 1 și aceleași greutăți bit cu bit** ca
   `pack_mixed_gelu`. Cinste: în simulator modelele pe procesor ale `mm_tc_pf` / `mm_tc_v4` sunt ale `mm_tc`, deci proba
   verifică **drumul** (alegerea, argumentele, QMD-ul acceptat, refuzul nealinierii), nu placa; „aceiași biți pe placă” pentru
   programe e dovedit doar la `mm_tc_pf` (os-437); la `mm_tc_v4` așteaptă os-438.
2. **Fără programe rapide și fără treaptă** (`pf = v4 = 0`, `None`): zona e **octet cu octet** cea a lui `pack_mixed_gelu`.
3. **Lista #513** (lot 2 × 512 și 6 × 512, planul `placi3 + mm-placi + colsum`): toate **147** de înmulțiri `mm_tc` trec `fits`
   → **147 pe `mm_tc_v4`, 0 pe `mm_tc_pf`**; planul lipit: 135 v4 + 12 `mm_tc_gelu`.
4. **QMD-ul:** față de cel vechi, o lansare `mm_tc` mutată pe v4 la 64 KiB diferă **doar** în adresa programului, numărul de
   registre și câmpurile MIN / TARGET ale treptei; o lansare F32 doar în câmpurile treptei.
5. **Greșeli plantate prinse:** A mutat cu 4 octeți → `pf`, nu `v4`; v4 forțat pe un produs nealiniat → simulatorul refuză;
   treapta 16 KiB (sub cei 24 KiB ai `mm_tc*`) → refuzată înainte de trimitere; `tc = 0` când o lansare cade pe `mm_tc` → refuz.
6. **Timpul (DEDUS, aritmetică):** înmulțirile `mm_tc*` ale pasului #513 lot 2 ≈ **0,6 TFLOP** pe pas; la 1,03 TFLOP/s (`mm_tc`
   vechi) ~580 ms, la 2,75 (`mm_tc_pf`, 32 KiB) ~220 ms, la 4,4–5,2 (64 KiB, prezis în nota sesiunii 233) ~115–140 ms. Adică
   **pasul mixt câștigă față de F32 (266,8 ms în os-431) doar cu programele rapide la 64 KiB**, iar ținta centrului (≥ 8 TFLOP/s)
   ar da ~75 ms pe înmulțiri.

## Ce e nou pentru placă
Nimic nou ca program sau câmp: `mm_tc_pf` (pe placă în os-437), `mm_tc_v4` (în os-438), treapta 64 KiB (os-431). Nou e doar
**amestecul** lor în pasul mixt — de rulat de OS abia după os-438 (`mm_tc_v4` aceiași biți pe placă).
