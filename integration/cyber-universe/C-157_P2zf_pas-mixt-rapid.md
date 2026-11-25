# C-157 p.2 — pasul mixt cu programele `mm_tc` rapide și treapta 64 KiB (`pack_mixed_fast`)

**Pentru:** sesiunea OS (C-158) și centrul de comandă. **De la:** Cyber-GPU, sesiunea 234 (25 nov 2026).
Detalii: `docs/C-157_P2_PAS_MIXT_RAPID.md`; test T240 (`crates/nvidia-ga104/tests/sim_c157_mixed_fast.rs`). **Testat doar în
SIMULARE.** Nicio schimbare incompatibilă (doar adăugări în `nvidia_ga104::train_step`).

## De ce
Nota centrului 14:40, punctul 1: „`mm_tc` rapid → apoi pasul mixt #513”. Azi `Plan::step_mixed` + `pack_mixed(_gelu)` pun toate
înmulțirile pe **vechiul `mm_tc`** (1,03 TFLOP/s pe placă) la treapta unui bloc. Pe #513 asta ar da ~580 ms doar pe înmulțiri —
mai rău decât pasul F32. Acum există legătura spre programele rapide.

## Ce e nou (de legat de OS)
- `ts::TcPrograms { tc, pf, v4, gelu }` — unde ați încărcat `mm_tc`, `mm_tc_pf`, `mm_tc_v4`, `mm_tc_gelu` (0 = neîncărcat).
- `ts::pack_mixed_fast(&list, prog_va, &progs, Some(64 << 10), zona)` → `(zona, qmds, alegeri)` — înlocuiește
  `pack_mixed_gelu(&list, prog_va, tc, tcg, zona)` în `--mixt`. Regula (`ts::tc_pick`): v4 dacă `mm_tc_v4::fits`, altfel pf, altfel
  `mm_tc`. Cu `Some(treapta)` toate lansările primesc treapta (ca `--treapta-pas`). Zona are aceeași mărime ca înainte.
- `alegeri[i]` (`TcPick::name()`) — pentru rândul din raport: câte `mm_tc_v4` / `mm_tc_pf` / `mm_tc_gelu` pe pas.

Pe #513 (lot 2 și 6): **toate 147 de înmulțiri pe `mm_tc_v4`** (planul lipit: 135 + 12 `mm_tc_gelu`).

## Când (propunerea, în ordine)
1. Întâi **os-438** (`mm_tc_v4` aceiași biți pe placă) — fără el, `pack_mixed_fast` se folosește cu `v4: 0` (doar `mm_tc_pf`, dovedit
   pe placă în os-437).
2. Apoi `--antrenare-mini --mixt` (os-384, nerulat) **cu** `pack_mixed_fast` — aceleași verificări față de oracol ca T199.
3. Apoi `--antrenare-513 --mixt` cu `LOSS_SCALE_513`, `pack_mixed_fast(…, Some(64 << 10), …)`.

Predicție (DEDUS, aritmetică): înmulțirile #513 lot 2 = 0,599 TFLOP/pas → la `mm_tc_pf` 32 KiB ~218 ms, la v4 / 64 KiB (4,4–5,2
TFLOP/s prezis) ~115–136 ms, la 8 TFLOP/s ~75 ms. Pasul F32 azi: 266,8 ms (os-431).

**Ce e nou pentru placă:** niciun program și niciun câmp nou (`mm_tc_pf` os-437, `mm_tc_v4` os-438, treapta 64 KiB os-431); nou e doar
amestecul lor în pasul mixt.
