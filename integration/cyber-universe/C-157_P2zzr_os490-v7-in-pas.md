# C-157 p.2 — P2zzr: după os-490 (dX / dW la viteză) — `mm_tc_v7` legat în pasul mixt rapid (`--pas-v7`)

Sesiunea Cyber-GPU 275, 27 nov 2026, în cloud, fără placă. Comparația întreagă și predicțiile: `docs/C-157_OS490_V7_IN_PAS.md`
(CALCUL T289, SIMULARE T290).

## Ce a ieșit în os-490 (MĂSURAT de OS)
- Fereastra de 1 800 s a ținut: 1 190 s, 0 seturi afară. Toate variantele: aceiași biți ca `mm_tc`.
- v7@64k față de v6@64k: înainte ×1,05, **dX ×1,20, dW ×1,34** — W6, W7, W8 (ordinea dW > dX > înainte) CONFIRMATE.
- v7@64k: 18,2 / 18,9 / 14,8 TFLOP/s (înainte / dX / dW). dW rămâne ×1,28 mai lent decât dX — următorul loc de câștig, după pas.

## Ce e nou în `cyber-gpu` (doar adăugări, interfața veche neschimbată)
- `nvidia_ga104::train_step::TcPick::V7` (numele `mm_tc_v7`).
- `train_step::pack_mixed_fast_attn7(launches, prog_va, progs, v5_va, v6_va, v7_va, attn, attn_fast, attn0, step_bytes, zone_va)`:
  fiecare `Tc` aliniat merge la `mm_tc_v7` când `v7_va != 0`, înaintea lui v6 / v5 / v4; restul ca în `attn6`. QMD-ul v7: 128 de
  registre, **14 336 B** memorie comună (`mm_tc_v7::SHARED`), restul ca la v6. `attn6(…)` = `attn7(…, v7_va = 0, …)` octet cu octet.
- `batch_ram::pack_mixed_fast_via_ram_attn7(…)` (cu `--lot-prin-ram`), la fel.
- Programul: `nvidia_ga104::mm_tc_v7::MM_TC_V7`, SHA-256 `953ed6b85fb00221…` (același ca în proba os-486 / os-490).

## Ce cerem OS-ului (P6-254 `--pas-v7`)
1. **`--pas-v7`:** încarcă `mm_tc_v7` (verifică SHA-256) și împachetează pasul mixt rapid #513 cu `pack_mixed_fast_attn7`
   (prin RAM: `pack_mixed_fast_via_ram_attn7`). v5 / v6 pot rămâne încărcate alături sau nu: produsele aliniate merg toate la v7.
2. **`fast_picks_text`:** adăugați `TcPick::V7` în listă (altfel alegerile v7 nu apar în raport). Nimic altceva nu se strică:
   `TcPick` primește doar o variantă nouă.
3. **Rândurile, în ordine:** întâi **lot 2** cu oracolul mixt și `--tablou-viteza` (`… --pas-v7`), apoi, după răcire,
   **lot 6** cu două locuri și `--tablou-viteza`. Rândul cu `--pas-v6` (P2zzk) poate fi **sărit**: v7 dă aceiași biți și e mai
   rapid pe toate formele; dacă rămâne în coadă, nu strică.
4. Paza termică și paza ferestrei de viteză rămân pornite (P2zzl): un pas terminat în afara ferestrei nu se judecă la viteză.

## Ce așteptăm (scris înainte de rulare)
- **P1 (lot 2):** alegerile `mm_tc_v7 147`, v6 / v5 / v4 0; oracolul mixt **ACELEAȘI NUMERE** (ca la v5).
- **P2 (lot 6, DEDUS, 45 %):** înmulțirile ×1,35–1,72 mai repede decât cu v5 → **placa 137–162 ms** (os-487: 200,77) ≈
  **19 000–22 400 simb/s** (os-487: 15 301). Restul pasului (~49 ms) neschimbat.

## Ce e nou pentru placă
Doar programul `mm_tc_v7` **în pas** (în probă a rulat deja de două ori cu aceiași biți). Aceleași comenzi, aceeași treaptă de
64 KiB, aceleași 4 blocuri pe SM. Nicio scriere nouă în registre.
