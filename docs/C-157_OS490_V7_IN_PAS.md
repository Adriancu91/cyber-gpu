# C-157 punctul 2 — os-490 comparat (dX / dW la viteză) + `mm_tc_v7` legat în pasul mixt rapid (sesiunea 275)

Sesiunea Cyber-GPU 275, 27 nov 2026, ~16:13 (ora României), în cloud, **fără placă**. Secțiunile 1–3 sunt scrise
**înainte** de codul și testele T289 / T290. Cifrele „MĂSURAT” vin din raportul plăcii
(`Cyber-hub/rezultate/os-automat/2026-11-27/ciclu-auto-20261127-1520.txt`, rândurile `mm_tc*`); restul e CALCUL sau DEDUS.

## 1. os-490: ce a ieșit (MĂSURAT de OS) și verdictele

Rândul P2zzm (`test-placa --tablou-viteza --viteza-1800 --mm-tc --mm-tc-v6 --mm-tc-v7 --mm-tc-64k --mm-tc-dxdw`), cu cererea de
viteză maximă pe 1 800 s aprobată de Adrian (DECIZII 27 nov 14:35). Lanțul a mai rulat **1 190 s** după cerere: **0 seturi în
afara ferestrei** (estimat în P2zzm: ultimul program la ~1 150 s — în fereastră, cum am spus).

Biți: `mm_tc_v6`, `mm_tc_v6@64k`, `mm_tc_v7`, `mm_tc_v7@64k` = `mm_tc` pe **7 / 7** seturi aliniate (1 sărit, nealiniat, ca
întotdeauna); `mm_tc@64k` 8 / 8.

Mediana a 5 lansări (ns), la treapta de 64 KiB:

| Setul (aceeași muncă: 21,74 GFLOP) | v6@64k | v7@64k | v7 / v6 | TFLOP/s v7 |
|---|---|---|---|---|
| qkv-513 (înainte, `TB \| BIAS`) | 1 256 544 | 1 194 752 | ×1,052 | 18,19 |
| qkv-513-dx (fără steag) | 1 380 128 | 1 147 488 | ×1,203 | 18,94 |
| qkv-513-dw (`TA`) | 1 962 880 | 1 464 352 | ×1,340 | 14,84 |

| # | Predicția (scrisă înainte) | Placa | Verdict |
|---|---|---|---|
| fereastra (P2zzm) | ultimul program ~1 150 s, în cei 1 800 s | 1 190 s, 0 seturi afară | CONFIRMATĂ |
| W6 | dX v7 / v6 ×1,05–1,40 | ×1,203 | CONFIRMATĂ |
| W7 | dW v7 / v6 ×1,10–1,60 | ×1,340 | CONFIRMATĂ |
| W8 | ordinea câștigurilor dW > dX > înainte | 1,340 > 1,203 > 1,052 | CONFIRMATĂ |
| estimarea T283 | dX v6@64k ~1,48 ms | 1,380 ms | în bine cu 7 % |
| estimarea T283 | dW v6@64k 1,63–2,18 ms | 1,963 ms | în bandă |
| V9 (os-485) | dX v6 / v5 ×1,00–1,10 | 1,620 (os-485, în fereastră) / 1,380 = ×1,174 | NECONFIRMATĂ (mai repede; două rulări diferite) |
| V10 (os-485) | dW v6 / v5 ×1,15–1,70 | v5 pe dW n-a rulat niciodată la viteză | rămâne NEJUDECABILĂ |

**Ce înseamnă (pe scurt):** calea memoriei comune era frâna și pe dX / dW (pragul din `C-157_CITIRI_CONFLICTE.md` §4: W7 peste
×1,10 → v7 devine prioritar). `mm_tc_v7@64k` e cel mai rapid pe toate trei felurile de înmulțiri din pas. Ce rămâne: dW e încă
**×1,28 mai lent** decât dX la aceeași muncă (1,464 față de 1,147 ms) — e următorul loc unde se poate câștiga în înmulțiri
(nu-l tai acum; întâi pasul cu v7 pe placă).

## 2. Ce leg acum (doar adăugări)

- `TcPick::V7` (numele `mm_tc_v7`). OS-ul folosește `TcPick` doar într-o listă (`fast_picks_text`), nu într-un `match`
  complet, deci varianta nouă nu-i strică construcția; trebuie doar adăugată în listă ca să fie numărată.
- `train_step::pack_mixed_fast_attn7(…, v5_va, v6_va, v7_va, …)`: fiecare `Tc` aliniat (`mm_tc_v7::fits` = `mm_tc_v4::fits`)
  merge la `mm_tc_v7` când e încărcat, **înaintea** lui v6; altfel regula lui `attn6` (v6 → v5 → v4 → pf → tc). QMD-ul v7 =
  QMD-ul v6 cu altă adresă de program și **14 336 B** de memorie comună în loc de 12 288 (aceleași 128 de registre, același
  bloc, aceeași grilă, același cbuf). `pack_mixed_fast_attn6` = `attn7` cu `v7_va = 0`.
- `batch_ram::pack_mixed_fast_via_ram_attn7`, la fel; `attn6` prin RAM = `attn7` cu `v7_va = 0`.

## 3. Predicțiile (scrise înainte de teste și de orice rulare a pasului cu v7)

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| M1 | `attn7` cu `v7 = 0` | = `attn6` octet cu octet (mini și #513; cu / fără v5, v6; și prin RAM) | 95 % |
| M2 | cu v7 încărcat | toate cele 147 `Tc` aliniate din #513 lot 2 (câte erau la v6) → v7; pf / tc neschimbate; zona aceeași lungime; QMD-ul v7 diferă de cel v6 doar la adresa programului și la mărimea memoriei comune (≤ 6 octeți) | 85 % |
| M3 | antrenarea mini, 5 pași | aceiași biți ca cu v6 (pierderi, gradienți, greutăți); și cu v7 fără v6 / v5 / v4 încărcate | 90 % |
| M4 | refuzuri | adresa v7 nealiniată → refuz; un QMD v7 cu memoria comună scăzută la 12 KiB (cât cere v6) → refuzat de procesor | 85 % |
| M5 | blocuri pe SM la 64 KiB | v7 tot 4 (4 × 14 KiB + rezerva de 1 KiB pe bloc ≤ 64 KiB), ca v6 | 95 % (CALCUL) |
| P1 | pasul #513 lot 2 cu `--pas-v7` pe placă | alegerile `mm_tc_v7 147`, v6 / v5 0; oracolul mixt **ACELEAȘI NUMERE** (aceiași biți ca v5 / v6) | 85 % |
| P2 | pasul #513 lot 6 pe placă (os-487 cu v5: 200,77 ms, din care `mm_tc_v5` 75,4 % ≈ 151,4 ms) | înmulțirile ×1,35–1,72 mai repede → **placa 137–162 ms** (mijloc ~145) ≈ **19 000–22 400 simb/s** | 45 % |

P2, socoteala (DEDUS): pe forma qkv, v5@64k = 1,869 (înainte) + 1,620 (dX, os-485) + dW nemăsurat la viteză (2,28–3,05,
T283) = 5,77–6,54 ms; v7@64k = 1,195 + 1,147 + 1,464 = 3,806 ms → ×1,52–1,72. În pas înmulțirile mergeau cam ca proba
(11,85 față de 11,64 TFLOP/s), dar pasul are și alte forme (MLP, capul de 16 384), deci marginea de jos coborâtă la ×1,35.
Restul pasului (~49,4 ms) nu se schimbă. Riscul: cu înmulțirile mai repede placa se încălzește mai repede — paza termică
rămâne pornită; dacă taie, timpii nu se judecă.

## 4. Ce a ieșit (după teste)

| # | Rezultat | Stare |
|---|---|---|
| §1 | T289: fereastra, W6 ×1,203, W7 ×1,340, W8, estimările T283, V9 peste două rulări ×1,17 — cifrele din tabel, socotite din nou | PASS (CALCUL) |
| M1 | `attn7` cu `v7 = 0` = `attn6` octet cu octet (mini, #513 lot 2; cu / fără v5, v6; cu / fără `tc16_v4`); prin RAM la fel | PASS (SIMULARE) |
| M2 | mini 27, #513 lot 2 **147**, #513 lot 6 **147** × v6 → v7; pf / tc 0 neschimbate; zona aceeași lungime; **2 octeți** diferiți pe QMD (adresa + memoria comună 12 288 → 14 336); fără v6 / v5 aceleași alegeri | PASS (SIMULARE) |
| M3 | mini 5 pași cu v7 = cu v6 = cu v4, bit cu bit (2 301 760 valori; și doar cu v7, și cu `tc16_v4`); pierderea 9,7128 → 8,8200 | PASS (SIMULARE) |
| M4 | adresă v7 nealiniată → refuz; QMD v7 cu memoria comună a lui v6 (12 KiB) → „QMD dă mai puțină memorie partajată decât cere programul” | PASS (SIMULARE) |
| M5 | 4 blocuri pe SM la 64 KiB, ca v6 (cu și fără rezerva de 1 KiB) | PASS (CALCUL) |
| P1, P2 | pe placă | NOT TESTED |

Nota pentru OS: `integration/cyber-universe/C-157_P2zzr_os490-v7-in-pas.md`. **Nou pentru placă:** doar `mm_tc_v7` în pas (în
probă a rulat deja cu aceiași biți, os-486 și os-490).
