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

## Rezultate (sesiunea 234, T240 `crates/nvidia-ga104/tests/sim_c157_mixed_fast.rs`, SIMULARE)
| Predicție | Ce a ieșit | Verdict |
|---|---|---|
| 1. Aceiași biți | mini, 5 pași, 3 planuri (simplu / plăci3 / lipit): rapid la 64 KiB **27 v4** (lipit: 25 v4 + 2 `mm_tc_gelu`), 0 pf, 0 `mm_tc`; „doar pf” 27 pf — pierderile, gradientul pasului 1 și greutățile **bit cu bit** ca `pack_mixed_gelu` | CONFIRMATĂ (drumul, nu placa) |
| 2. Fără programe rapide | zona octet cu octet = `pack_mixed_gelu` pe cele 3 planuri | CONFIRMATĂ |
| 3. Lista #513 | lot 2 și lot 6: **147 v4, 0 pf**; lipit: 135 v4 + 12 gelu; zona 499 / 481 KiB, cât cea veche | CONFIRMATĂ |
| 4. QMD-ul | v4: doar adresa programului, registrele, MIN / TARGET; **dar** lansările F32 fără memorie comună primesc și MAX = 64 KiB (494 de biți pe pas) — exact ce face `pack_with_sm_step` (os-431, deja pe placă) | **PARȚIAL GREȘITĂ** (uitasem MAX) |
| 5. Greșeli plantate | A + 4 octeți → pf (1 lansare), pasul trece; v4 forțat pe același produs → simulatorul refuză („nealiniat”); treptele 16 și 48 KiB refuzate; `mm_tc` / `mm_tc_gelu` neîncărcat când trebuie → refuz; adresă nealiniată → refuz; listă goală → refuz | CONFIRMATĂ |
| 6. Timpul (DEDUS) | înmulțirile #513 lot 2 = **0,599 TFLOP/pas** → 582 ms la 1,03 · 218 la 2,75 · 115–136 la 5,2–4,4 · **75 la 8 TFLOP/s**; lot 6: 1,797 TFLOP → 654 / 346–409 / 225 ms | aritmetică |

**Ce înseamnă (DEDUS):** pasul mixt #513 lot 2 cu vechiul `mm_tc` ar fi ~600+ ms (mai rău decât F32, 266,8 ms). Cu `mm_tc_pf` la
32 KiB ~220 ms doar înmulțirile + ~40–60 ms restul ≈ cât F32 azi. Câștigul real vine doar cu v4 la 64 KiB (~115–136 ms) și, pentru ținta
centrului, cu ≥ 8 TFLOP/s (~75 ms → pasul ~120–140 ms ≈ 7 500–8 500 simb/s la lot 2). De aceea os-438 (v4 pe placă) și treapta 64 KiB
pentru `mm_tc*` (nota sesiunii 233) decid dacă merită rulat pasul mixt rapid.
