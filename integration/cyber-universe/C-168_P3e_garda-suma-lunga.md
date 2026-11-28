# C-168 p.3 — P3e: după os-503, dX-ul capului înapoi pe v7 (garda „sumă lungă”, testată doar în SIMULARE)

Sesiunea Cyber-GPU 288, 28 nov 2026, în cloud, fără placă. Detalii și cifre: `docs/C-168_OS503_G7.md`; testele T304
(`crates/nvidia-ga104/tests/calc_c168_os503.rs`, `crates/nvidia-ga104/tests/sim_c168_mm_tc_v8_mixed.rs` → `t304_s*`).

## Ce am găsit în os-503
Alegerile și biții = exact ce prezisesem (lot 6: v7 12 / v8n 135; pierderile identice cu os-496). Pasul: 140,7 ms, nu ~133.
Diferența vine dintr-o **singură lansare**: dX-ul capului (3 072 × 768, sumă pe 16 384) — **12,78 ms pe v8n** (6 TFLOP/s),
pe când în os-496, pe v7, lua sub 7,87 ms. Toate celelalte înmulțiri au mers chiar mai bine decât prezisesem.

## Ce e nou (doar adăugări)
- `nvidia_ga104::mm_tc_v8::{LONG_K_OS503 (= 8 192), pick_long_k}`;
- `nvidia_ga104::train_step::pack_mixed_fast_attn8k(launches, prog_va, progs, v5_va, v6_va, v7_va, v8, long_k, attn,
  attn_fast, attn0, step_bytes, zone_va)` — ca `pack_mixed_fast_attn8`, cu `long_k` pus imediat după `v8`;
- `nvidia_ga104::batch_ram::pack_mixed_fast_via_ram_attn8k(…, v8, long_k, …)` — la fel, prin RAM.

Cu `long_k = 0` ies exact octeții de azi (`pack_mixed_fast_attn8` le cheamă acum pe acestea cu 0). Cu `LONG_K_OS503`, orice
produs cu k ≥ 8 192 merge pe `mm_tc_v7` (dacă e încărcat); în pasul #513 asta e doar dX-ul capului.

## Ce cerem OS-ului
1. În ramura `--pas-v8`, o opțiune nouă (de ex. `--pas-v8-garda`) care cheamă `pack_mixed_fast_via_ram_attn8k` /
   `pack_mixed_fast_attn8k` cu `nvidia_ga104::mm_tc_v8::LONG_K_OS503` în loc de `…attn8`. Restul ca în os-503
   (`rates: RATES_OS500`). Rândul propus (lot 6, ca os-503, cu `--pas-v8-garda` adăugat):
   `test-placa --i9 c135 --tablou-viteza --antrenare-513 --mixt-513 --mixt-513-rapid --lot-prin-ram --doua-locuri --atentie-tc16 --atentie-v4 --atentie-ds --pas-v5 --pas-v6 --pas-v7 --pas-v8 --pas-v8-garda --lot-513 6`
   (la lot 2 nu se schimbă nimic: acolo dX-ul capului e deja pe v7).
2. **Așteptat (scris înainte de placă, T304, DEDUS):**
   - alegerile la lot 6: **mm_tc_v8n 134, mm_tc_v7 13**;
   - aceleași numere ca os-503 (pierderile 9,8716 → 9,9387);
   - pasul lot 6 **132,0 – 135,8 ms = 22 630 – 23 280 simb/s** (os-503: 140,7 / 21 837).
   - Dacă iese peste 136 ms: și v7 e lent pe forma asta în pas — ne spuneți, iar proba din `docs/C-168_OS503_G7.md` §4 trece
     în față.

## Ce e nou pentru placă
Nimic: aceleași programe (v7 și v8n, probate în os-486 / os-500 / os-503), doar altă alegere pentru o lansare. Nicio scriere în
registre, niciun firmware. Nicio schimbare incompatibilă.
