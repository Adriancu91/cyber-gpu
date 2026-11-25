# C-157 p.2 — os-445 (pasul mixt rapid #513) comparat + oracolul pasului mixt pe procesor

Sesiunea 238, 25 nov 2026, în cloud, **fără placă**. Cifrele plăcii sunt MĂSURATE de OS (os-445, ciclul automat 22:02); tot
ce e scris aici în plus e **DEDUS** sau calculat pe procesor (cloud). Nimic nu a rulat pe RTX 3070 Ti în sesiunea asta.

Sursa: `Cyber-hub/rezultate/os-automat/2026-11-25/ciclu-auto-20261125-2202.txt` (rândurile 145–165, copiate neschimbate în
`crates/nvidia-ga104/tests/rapoarte/os445-mixt-rapid.txt`) și `Cyber-hub/rezultate/2026-11-25-workstation-os445-mixt-rapid/REZULTAT.md`.

## 1. Ce a ieșit față de ce am prezis

| # | prezis (înainte de rulare) | ieșit pe placă | verdict |
|---|---|---|---|
| alegerea | 147 / 147 pe `mm_tc_v4`, 0 pe `mm_tc_pf` (sesiunea 234, P3) | **147 `mm_tc_v4`** | CORECTĂ |
| mini | `antrenare mini --mixt --mixt-rapid` = oracolul (27 v4) | 27 / 27 v4, = oracolul | CORECTĂ |
| timpul | înmulțirile 72–87 ms, pasul ~120–140 ms (sesiunea 237, §6 din `C-157_P2_OS436.md`) | placa **106,0 ms**, pasul întreg **116,9 ms = 8 758 simb/s** | RATATĂ puțin (mai rapid) |
| pierderile | „aceleași pierderi ca oracolul” (pașii 1–9 la 1e-3) | **DIFERĂ**: cea mai mare abatere 7,2e-3 (pierderea), 1,3e-2 (norma) | **RATATĂ** |

Abaterile pe pași (placa mixt față de `ORACLE_513_SYNTH`, oracolul F32; relative):

| pas | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| pierderea | 0 | 1,0e-5 | 0 | 2,0e-5 | 6,1e-5 | 1,0e-5 | 2,6e-4 | **5,5e-3** | **7,2e-3** |
| norma | 6,1e-5 | 0 | 3,9e-4 | 1,1e-4 | 1,1e-4 | 8,9e-4 | **1,3e-2** | 5,9e-3 | 1,2e-3 |

**Ce se vede (DEDUS):** pașii 1–6 stau **în toleranța pasului mixt** (pierderea ≤ 6,1e-5, norma ≤ 8,9e-4 < 2e-3); abaterea pornește
la pasul 7 și crește. Rularea pe loturi sintetice e instabilă (pierderea urcă peste ln V; oracolul F32 însuși sare la pasul 10,
norma 0,93 → 1,66, și placa F32 din os-395 diferea deja acolo cu 1,4e-3). AdamW împarte fiecare gradient la mărimea lui, deci o
diferență mică pe gradientele mici se face pas întreg de lr; pe o traiectorie instabilă diferența crește de la pas la pas. Rotunjirea
f16 e o diferență mai mare decât ordinea sumelor, deci se vede mai devreme (pasul 7 în loc de 10).

Asta e o ipoteză. Ca s-o deosebim de „programul greșește ceva în pasul mare” trebuie **oracolul pasului mixt**: același pas pe
procesor, cu intrările celor 147 de înmulțiri rotunjite la f16 exact ca `mm_tc` (gradientele cu scara S = 2¹⁴), sumele în f64.

## 2. Predicțiile pentru oracolul mixt, scrise ÎNAINTE de cod și de rulare

- **M1.** Oracolul mixt, pasul 1: pierderea **9,8493** și norma **3,273 ± 0,001** (rotunjirea intrărilor schimbă pierderea sub 1e-4).
- **M2.** Oracolul mixt față de oracolul F32: pașii 1–6 sub 1e-3, apoi se despart la pașii 7–9 cu **≥ 2e-3** (aceeași formă ca placa).
- **M3.** Placa (os-445) față de oracolul mixt: **pașii 1–9 sub 1e-3** (ca placa F32 față de oracolul F32 în os-395: 1,1e-4). Dacă
  iese așa, programele pasului mixt fac exact ce trebuie, iar „DIFERĂ” e prețul f16, nu o greșeală. Dacă placa e la fel de departe
  de oracolul mixt ca de cel F32, atunci traiectoria e prea instabilă ca să judece 9 pași și verdictul trebuie scurtat (pașii 1–6).
- **M4.** Proba pe modelul mic (T246): oracolul mixt cu intrări deja exacte în f16 = oracolul F32 bit cu bit; S = 1 și S = 2¹⁴ dau
  aceleași numere cât timp nicio valoare nu e subnormală; oracolul mixt ≠ F32 pe intrări obișnuite.

## 3. Timpul (MĂSURAT față de DEDUS)

- Pasul întreg 116,9 ms, placa 106,0 ms (pașii 3–10; pașii 1–2 113,9 / 109,9 ms = încălzirea). Predicția: înmulțirile 72–87 ms +
  restul ~40–60 ms (cât în os-431 F32 fără înmulțiri) → pasul 120–140 ms. A ieșit sub capătul de jos: dacă înmulțirile au mers la
  viteza lui qkv-513 (8,28 TFLOP/s → 72 ms), restul plăcii e doar **~34 ms**, mai puțin decât socoteam (DEDUS; fără profilul pe
  lansări nu se poate despărți).
- ×2,28 față de F32 lot 2 (os-431, 266,8 ms); 22 % din Windows (~39 300 simb/s).

## 4. Rezultatele oracolului mixt

(se completează după rulare)
