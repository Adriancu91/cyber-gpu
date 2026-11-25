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

## 4. Rezultatele oracolului mixt (calculat pe procesor, cloud, sesiunea 238)

Cod (doar adăugare): `gpu_compute::train::loss_and_grad_mixed(arch, greutăți, lot, S)` — pasul de antrenare cu intrările celor 147 de
înmulțiri rotunjite la f16 (gradientele: `round_f16(g·S)/S`), restul neschimbat; `examples/oracol_513.rs -- 10 --mixt` (≈ 3,5 min pe pas,
un fir); constantele `gpu_compute::train::ORACLE_513_MIXED_SYNTH`. Funcțiile publice existente nu s-au schimbat.

| pas | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| oracol F32 (pierdere / normă) | 9,8493 / 3,2729 | 9,8750 / 1,1021 | 9,8293 / 1,0242 | 9,9083 / 0,9176 | 9,8642 / 0,8955 | 9,8817 / 0,8962 | 9,8472 / 0,8914 | 9,9391 / 0,9364 | 9,9929 / 0,9346 | 10,1628 / 1,6557 |
| **oracol mixt** (nou) | 9,8493 / 3,2731 | 9,8749 / 1,1021 | 9,8293 / 1,0246 | 9,9086 / 0,9175 | 9,8650 / 0,8956 | 9,8816 / 0,8952 | 9,8508 / 0,9079 | 10,0040 / 0,9288 | 9,9514 / 0,9380 | 9,9519 / 1,0257 |
| **placa os-445** (MĂSURAT) | 9,8493 / 3,2731 | 9,8749 / 1,1021 | 9,8293 / 1,0246 | 9,9085 / 0,9175 | 9,8648 / 0,8956 | 9,8816 / 0,8954 | 9,8498 / 0,9028 | 9,9936 / 0,9309 | 9,9207 / 0,9357 | 10,0178 / 1,0812 |

| Predicție | Ce a ieșit | Verdict |
|---|---|---|
| M1. pasul 1: 9,8493 și 3,273 ± 0,001 | 9,8493261 / 3,2731080 — **exact cifrele plăcii** (F32 dădea 3,2729) | CONFIRMATĂ |
| M2. mixt față de F32: 1–6 sub 1e-3, 7–9 ≥ 2e-3 | 1–5 sub 3,9e-4; **pasul 6 norma 1,1e-3** (puțin peste); pasul 7 norma 1,9e-2, 8 pierderea 6,5e-3 | CONFIRMATĂ în mare (pasul 6 la limită) |
| M3. placa față de oracolul mixt: 1–9 sub 1e-3 | **pașii 1–6: pierderea ≤ 2,0e-5, norma ≤ 2,2e-4** (pașii 1–3 aceleași 4 zecimale); pașii 7–9: norma 5,6e-3, pierderea 3,1e-3 | **PARȚIAL RATATĂ**: 1–6 da, 7–9 nu |
| M4. proba pe modelul mic (T246) | aproape de F32 (pierderea 2,1e-6) dar alți biți; pierderea nu depinde de S; S greșit sau prea mare refuzat; prezisesem gradientul ≤ 5e-3 pe tensor — a ieșit **7,3e-3** pe `straturi.0.ln2.bias` (doar 64 de simboluri, se compensează mai puțin) → testul cere 1e-2 | CONFIRMATĂ, cu toleranța gradientului ratată |

**Ce înseamnă (DEDUS):**
1. **Programele pasului mixt fac pe placă exact ce trebuie.** Pe pașii 1–6 placa e de 3–4 ori mai aproape de oracolul mixt decât de cel
   F32 (norma 2,2e-4 față de 8,9e-4), iar pașii 1–3 sunt aceleași cifre. „DIFERĂ” din raport nu e o greșeală de program.
2. **Rotunjirea f16 costă puțin pe un pas** (pierderea ≤ 8e-5, norma ≤ 1,1e-3 pe pașii 1–6), dar rularea sintetică e instabilă de la pasul
   7 (oracolul F32 sare și el la pasul 10). Acolo orice diferență crește — chiar și ordinea sumelor f32 ale plăcii, care în pasul mixt
   poate întoarce o rotunjire f16 (salt de 2⁻¹¹ în loc de 2⁻²⁴). De aceea placa F32 ține 9 pași (os-395), iar placa mixt doar 6.
3. **Verdictul propus pentru OS** (nota `integration/cyber-universe/C-157_P2zi_oracol-mixt.md`): pentru `--mixt-513` pe loturile
   sintetice, față de `ORACLE_513_MIXED_SYNTH` pe **pașii 1–6** la 1e-3 (pierderea) / 2e-3 (norma); pașii 7–10 doar afișați. Cu os-445
   ar fi ieșit **ACELEAȘI NUMERE (mixt)**: 2,0e-5 / 2,2e-4.
4. **Rămâne deschis (nu se poate răspunde pe date sintetice):** dacă f16 strică *învățarea*. Asta e cererea centrului (OS + Kronos):
   comparația F32 față de mixt pe text real, aceiași pași, aceeași sămânță. Nimic de făcut aici până atunci.
