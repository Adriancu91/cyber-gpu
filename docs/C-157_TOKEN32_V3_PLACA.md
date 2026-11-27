# C-157 p.2 — `token32_v3` pe placă (os-498, ciclul 27 nov 22:11) comparat cu predicțiile L1–L7

Sesiunea Cyber-GPU 280, 27 nov 2026, în cloud, fără placă. Sursa (MĂSURAT de OS): hub
`rezultate/os-automat/2026-11-27/ciclu-auto-20261127-2211.txt` (rândurile 83–106; rândul de test
`test-placa --i9 c135 --tablou-viteza --token32 --token32-v2 --token32-v3`). Predicțiile erau scrise înainte, în
`docs/C-157_TOKEN32_V3_LN.md` §2. Socoteala: T296 (`crates/nvidia-ga104/tests/calc_c157_token32_v3_placa.rs`, CALCUL pe procesor).
Viteza e valabilă: cererea de viteză maximă acceptată (600 s), lanțul a mai rulat 50 s după ea, 0 seturi în afara ferestrei, P0 la final.

## 1. Ce s-a prezis și ce a ieșit

| # | Prezis (scris înainte) | Ieșit (MĂSURAT) | Verdict |
|---|---|---|---|
| L1 | aceiași biți ca `token32_v2`, amprenta `0x6c8501f0`, contorul (63 − 1) · G | TRECE la G = 384 și 768: 16 simboluri identice, `0x6c8501f0`, contorul 18 / 18 | CONFIRMATĂ |
| L2 | listarea: ≤ 128 registre, 0 memorie locală | 97 registre (sesiunea 277) | CONFIRMATĂ (listare) |
| L3 | placa la G = 768 **1,18–1,38 ms** | **1,453 ms** (v2 în același lanț 1,667) | NECONFIRMATĂ — 73 µs peste capăt |
| L4 | cu driverul ~505–560 simb/s | **2 091 µs = 478 simb/s** (lista obișnuită 2 030 µs = 493) | NECONFIRMATĂ |
| L5 | G = 768 tot mai repede | 768: 1 453 µs, 384: 1 682 µs | CONFIRMATĂ |
| L6 | puține cuvinte schimbate între simboluri | primul simbol 4 032, apoi 38 (din 4 032) | CONFIRMATĂ |
| L7 | SIMULARE = lansările separate | (sesiunea 277) | CONFIRMATĂ în SIMULARE |

Câștigul real: v2 → v3 **−214 µs la G = 768** (×1,147) și **−254 µs la G = 384** (×1,151). `token32_v2` repetă exact (1 667 µs; os-495 1 666,
10:01 1 668). Partea driverului a rămas ~0,6 ms (v2 614, v3 638 µs). Pe placă `token32_v3` e tot de ×1,36 mai lent decât lista
obișnuită (1 067 µs), cu driverul cu 3 % mai lent (2 091 față de 2 030).

## 2. Unde s-a dus diferența (DEDUS, T296)

Ce scoate v3 din v2 (cifrele măsurate pe `token32_v2c` în os-495, aduse la mediana lui v2): munca celor 25 de LayerNorm (~440 µs) +
25 de bariere (~48 µs) = **~490 µs**. Dacă începutul lipit n-ar costa nimic, v3 ar fi ~1,18 ms (= capătul de jos al benzii L3). A ieșit
1,453 ms → cele 25 de începuturi lipite costă **~275–280 µs, ~11 µs fiecare**; prezis 1,5–4 µs (de ×3–7 peste).

Bănuieli (nedovedite, de măsurat): (a) toate cele 768 de grupuri citesc deodată aceiași ~3 KiB (rândul x + γ + β) imediat după
golirea L1 de la barieră — o coadă pe aceleași linii din L2; (b) cele două reduceri (media, apoi abaterea) sunt în șir, fiecare după
o citire din L2, înaintea primei citiri a înmulțirii — adică ~2–3 drumuri dus-întors la L2 în plus pe operație, fără nimic care să le
acopere; (c) dezechilibrul între grupuri, pe care bariera îl plătește.

## 3. Ce tai, ce las

- **Las** `token32_v3` ca bază pentru „un program pe simbol” (e cel mai rapid dintre cele trei, aceiași biți).
- **Nu** trece încă lista obișnuită: la generare rămâne lista (493 simb/s cu driverul, față de 478).
- Pasul următor pe linia `token32` (cum scria planul pentru cazul „L3 peste bandă”): `token32_v3c` = v3 + ceasul SM-ului pe operație
  (ca v2c), ca să vedem cât din cei ~11 µs e citirea și cât reducerea; plan + predicții întâi. Prioritatea rămâne la C-168 (v8 în pas),
  deci acesta stă în coadă.
