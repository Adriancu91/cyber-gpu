# C-157 — `attn_back_tile0_tc_v4` legat în pasul mixt rapid (alegerea v4 / tc pe fiecare lansare)

Sesiunea 252, 26 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte de cod**.
Nimic de aici nu a rulat pe RTX 3070 Ti; tot ce iese pe procesor e **SIMULARE**, timpii sunt **DEDUS**.

## 1. De ce

Sesiunea 251 a scris `attn_back_tile0_tc_v4` (dS-ul atenției cu citiri / scrieri de 128 de biți; aceiași biți ca
`attn_back_tile0_tc` în SIMULARE, T261, PREDARE P2zt), dar **nu l-a legat**: pasul mixt rapid (`pack_mixed_fast_attn2`, sesiunea
250) știe o singură adresă pentru dS pe circuite. v4 merge doar pe forme aliniate (T și d multiplu de 4, adrese multiplu de 16),
deci gazda trebuie să aleagă pe fiecare lansare: v4 dacă încape, altfel `attn_back_tile0_tc` — aceeași regulă ca la `mm_tc_v4`.
Legătura se face acum în SIMULARE, ca să fie gata când vine proba P2zt; în pas intră doar după ce P2zt trece pe placă.

## 2. Ce va fi (plan, doar adăugări)

- `train_step::Attn0Programs { tc, v4 }` — unde a încărcat OS-ul cele două programe dS (0 = neîncărcat).
- `TcPick::Attn0TcV4` (nume „attn_back_tile0_tc_v4”).
- `pack_mixed_fast_attn3(…, attn, attn0: Attn0Programs, …)`: o lansare `MixedLaunch::Attn0Tc` merge la v4 dacă `v4 ≠ 0` și
  `attn_back_tile0_tc_v4::fits_cbuf` pe argumentele ei, altfel la `tc` (refuz dacă `tc = 0`). QMD-ul e același (56 registre,
  33 024 B, o barieră, blocuri de 256, aceeași grilă, același cbuf) — diferă doar adresa programului.
  `pack_mixed_fast_attn2(…, attn0_va)` devine `pack_mixed_fast_attn3(…, {tc: attn0_va, v4: 0})` (aceiași octeți).
- `batch_ram::pack_mixed_fast_via_ram_attn3` (pentru `--lot-prin-ram`), la fel.
- Testul nou `crates/nvidia-ga104/tests/sim_c157_attn0_tc_v4_mixed.rs` (T262).

## 3. Predicțiile (scrise ÎNAINTE de cod)

- **V1 — fără v4 nimic nu se schimbă.** `pack_mixed_fast_attn3` cu `v4 = 0` = `pack_mixed_fast_attn2` octet cu octet (mini și #513).
- **V2 — lista mini cu ambele încărcate.** Cele 2 lansări dS aleg `Attn0TcV4`; zona are aceeași lungime ca la attn2 și diferă
  **doar** în octeții adresei programului din cele 2 QMD-uri (≤ 2 × 8 octeți).
- **V3 — antrenarea mini (5 pași) cu v4 = cu tc, bit cu bit** (pierderi, gradientul pasului 1, greutăți), fiindcă v4 face
  aceeași aritmetică; ambele programe „înapoi” ale atenției pe circuite.
- **V4 — lista #513 (lot 2).** 12 `Attn0TcV4` + 12 `AttnTc16` + 147 `mm_tc_v4`; zona cât la attn2; QMD-ul: adresa v4,
  56 registre, grila [8, 12, 2].
- **V5 — forma nealiniată cade pe tc.** O lansare dS cu adresa dS + 4 → aleasă `Attn0Tc` (dacă tc e încărcat), refuz dacă doar
  v4 e încărcat; adresă v4 nealiniată la 256 → refuz; o zonă în care QMD-ul unei forme nealiniate e mutat pe v4 → înlocuitorul
  din `gpu-sim` refuză (pe placă ar fi o greșeală de acces).
- **V6 — prin RAM.** `pack_mixed_fast_via_ram_attn3` cu `v4 = 0` = attn2 octet cu octet; cu v4 aceleași alegeri ca fără RAM.
- **V7 — timpul (DEDUS, doar pe placă).** Ca la P2zt: ×1,2–1,8 pe programul dS la `atentie-513-b4`. NOT TESTED.
