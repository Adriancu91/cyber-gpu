# C-157 p.2 — comparația os-496: `mm_tc_v7` în pasul #513 (`--pas-v7`, P2zzr)

Sesiunea Cyber-GPU 278, 27 nov 2026, ~20:11–21:05 (ora României), în cloud, FĂRĂ placă. Cifrele plăcii: MĂSURAT de OS
(`rezultate/os-automat/2026-11-27/ciclu-auto-20261127-1948.txt` = lot 2, `ciclu-auto-20261127-1954.txt` = lot 6; rezumatul
centrului: `rezultate/2026-11-27-workstation-os496-v7-in-pas-record/REZULTAT.md`). Socotelile: CALCUL (T293,
`crates/nvidia-ga104/tests/calc_c157_os496.rs`). Predicțiile au fost scrise înainte, în `docs/C-157_OS490_V7_IN_PAS.md` §3.

## 1. Față de predicții

| # | Predicția (P2zzr) | Placa (os-496) | Verdict |
|---|---|---|---|
| P1 | lot 2: alegerile `mm_tc_v7 147`, v6 / v5 0; oracolul mixt ACELEAȘI NUMERE | 147 / 0 / 0; **ACELEAȘI NUMERE** (pierderea 9,9·10⁻⁸, norma 6,3·10⁻⁷, greutățile 8,8·10⁻⁶) | CONFIRMATĂ |
| P2 | lot 6: placa **137–162 ms** (mijloc ~145) ≈ 19 000–22 400 simb/s | placa **147,81 ms**, pasul întreg **147,72 ms = 20 795 simb/s** | CONFIRMATĂ |
| P2 socoteala | înmulțirile ×1,35–1,72 față de v5; restul pasului (~49,4 ms) neschimbat | `mm_tc_v7` 98,10 ms față de ~151,5 (os-487) = **×1,545**; restul **49,7 ms** (+0,6 %) | CONFIRMATĂ |

Altele (MĂSURAT): `mm_tc_v7` în pas **18,3 TFLOP/s** la lot 6 (17,7 la lot 2) = **39 %** din vârful plăcii la ceasul măsurat
(47,6 TFLOP/s la 1 935 MHz, f16 cu sume f32, foaia publică GA104). Temperatura maximă 64 °C, fără oprire; „în afara plăcii” 0 µs
(două locuri). „NU TRECE” de la lot 6 e cel cunoscut (loturi sintetice: ultima pierdere nu poate fi sub prima; pierderile =
os-487 cifră cu cifră, deci aceeași aritmetică) — nu e o defecțiune.

**Cifra zilei:** 20 795 simb/s = **×1,36** față de os-487 (15 301) = **53 %** din Windows / PyTorch (~39 300).

## 2. Profilul „înainte” pentru C-168 (lot 6, 147,81 ms, MĂSURAT)

| Bucata | ms | % |
|---|---|---|
| `mm_tc_v7` (147 înmulțiri) | 98,1 | 66,4 |
| atenția înainte `attn_fwd_tile3` (12) | 16,1 | 10,9 |
| atenția înapoi (dS v4 + dQKV tc16_v4, 24) | 11,9 | 8,0 |
| mărunte (Adam 4,8, GELU 4,3, sume 4,7, vadd 2,5, LN 3,1, pierderea 1,5, rest) | 21,7 | 14,7 |

Cifrele pe programe adunate = placa la 0,5 % (T293). Acesta e punctul „înainte” al planului C-168 (punctele 3–5).

## 3. Ce urmează

Lucrul în curs s-a terminat (C-168, Adrian 20:00). Următorul: **C-168 punctul 3 — înmulțirea v8**, planul și predicțiile în
`docs/C-168_V8_PLAN.md` (scrise înainte de cod, în aceeași sesiune). **Nou pentru placă:** nimic.
