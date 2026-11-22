# C-160 punctul 4 — programul 6 al antrenării, jumătatea „înapoi”: atenția cauzală înapoi (`attn_back`)

Sesiunea 148, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE**: nimic nu a rulat pe RTX 3070 Ti.

## Ce e
Programul care trece gradientul pasului de antrenare înapoi prin atenția lui Kronos (cu masca cauzală), pe toate capetele și
toate textele lotului. Intră: qkv [B·T, 3C] și P [B, H, T, T] (păstrate de la drumul înainte) și dO [B·T, C] (gradientul ieșirii
atenției, adus de `mm_train` prin `out_proj`). Iese dqkv [B·T, 3C] = dQ | dK | dV, exact așezarea lui qkv, gata pentru înmulțirea
înapoi prin `in_proj`. Un singur binar, **două lansări la rând**:
- **modul 0 (rânduri)**, grila [⌈T/32⌉, H, B]: un fir pe rândul (b, h, i) scrie rândul i din dS: dP_ij = Σ_e dO_ie·V_je,
  dot = Σ_j P_ij·dP_ij, dS_ij = P_ij·(dP_ij − dot)·(1/√d) pentru j ≤ i și **0 pentru j > i**;
- **modul 1 (coloane)**, grila [H, T, B], 32 de fire care merg pe e: dQ_te = Σ_{j≤t} dS_tj·K_je, dK_te = Σ_{i≥t} dS_it·Q_ie,
  dV_te = Σ_{i≥t} P_it·dO_ie.

**Masca** = marginile buclelor: ce e deasupra diagonalei în P nu se citește niciodată (testat cu NaN și 1e30 acolo: niciun bit
schimbat). Fiecare sumă e făcută de un singur fir, într-o ordine fixă: **fără `atomic`, fără memorie partajată, fără barieră, fără
MUFU** → aceiași biți la fiecare rulare, iar modelul de pe procesor **este** programul, operație cu operație.

`tools/nak/attn_back.comp` → Mesa NAK 25.0.7: **5 104 octeți**, 319 instrucțiuni, 25 de registre; SHA-256
`7092074774576fbe9d213b0d3adb4f02d6fa013771cde8315526879fb0079f54`. Bufferul constant 0: qkv / P / dO / dS / dqkv la
0x28 / 0x30 / 0x38 / 0x40 / 0x48, T la 0x50, C la 0x54, H la 0x58, d la 0x5c, modul la 0x60, scala la 0x64 (0x68 octeți); baza
grupurilor la 0x0 / 0x4 / 0x8 rămâne 0. B nu e citit: e z-ul grilei.

Aritmetica, citită din `tools/nak/out/attn_back.sm86.txt` (testul o numără): **5 lanțuri `ffma.ftz`** (dP, dot, dQ, dK, dV, fiecare
pornind de la 0, indicii crescător), `fadd.ftz`(−0, −dot), `fadd.ftz`(dP, −dot), `fmul.ftz`(P, ·), `fmul.ftz`(·, scala). Modelul:
`gpu_compute::layer::{attn_back_row_gpu, attn_back_col_gpu, attn_back_gpu}`. Oracolul: `gpu_compute::train::attention_back_rows`
(adăugat; exact liniile din `loss_and_grad`, care au dat numerele PyTorch la pasul de aur mic, T151), plus `train::attention_rows`
(drumul înainte, pentru P).

## Marginea, scrisă înainte de măsurare (T159)
u = 2⁻²⁴; un lanț `ffma` de n termeni greșește cu cel mult 1,01·n·u·Σ|termeni| (+ eroarea adusă de operanzi), plus 2⁻¹²⁶ pe pas (FTZ):
dP: 1,01·d·u·Σ|dO·V|; dot: Σ P·ΔdP + 1,01·(i+1)·u·Σ P·|dP|; dS: s·P·(ΔdP + Δdot) + 4,1·u·s·P·(|dP − dot| + ΔdP + Δdot);
dQ / dK: Σ ΔdS·|K| (|Q|) + 1,01·n·u·Σ (|dS| + ΔdS)·|K| (|Q|); dV: 1,01·n·u·Σ P·|dO|; plus rotunjirea oracolului la f32 (u·|valoare|).
Oracolul primește **aceleași** intrări ca placa (P rotunjit la f32).

## Ce a ieșit (SIMULARE, modelul lansat prin procesorul de comenzi față de oracol)
| Caz | dS (cât din margine) | dqkv (cât din margine) | max \|Δ\| dqkv |
|---|---|---|---|
| forma modelului mic: B 2, T 32, C 64, H 4 | 6,5 % | 55,4 % | 6,0e-7 |
| T 70 (trece de blocul de 32), C 96, H 3 (d 32) | 3,7 % | 55,1 % | 7,2e-7 |
| T 33, C 80, H 2 (d 40: modul 1 face două ture pe e) | 3,4 % | 60,7 % | 4,8e-7 |
| T 5, C 8, H 1 (B 3) | 8,9 % | 61,1 % | 6,0e-8 |
| T 17, C 3, H 3 (d 1) | 19,4 % | 44,0 % | 1,2e-7 |
| softmax ascuțit (P aproape 0/1: dP − dot se anulează) | 7,8 % | 65,8 % | 3,8e-6 |
| gradienți mici (dO ~1e-4, ca dlogits/(B·T)) | 6,4 % | 65,5 % | 5,8e-11 |

Partea mare din marginea lui dqkv e rotunjirea oracolului însuși la f32 (o jumătate de unitate), deci programul e cât se poate de
aproape. Și: **lanțul cu P-ul pasului de antrenare** (P în f64 de la drumul înainte, rotunjit la f32 pentru placă) față de
`attention_back_rows` cu P în f64 (exact `loss_and_grad`): pe fiecare parte (dQ, dK, dV) abaterea ≤ 2⁻¹⁶ din cea mai mare valoare,
pe toate cele 7 cazuri. Lansarea = modelul, bit cu bit; două rulări = aceiași biți; dS deasupra diagonalei = +0 exact; grila scurtă
lasă restul neatins (capete, texte, rânduri peste 32·x); refuzuri (grilă peste H / T, bloc greșit: nimic scris); **10 greșeli plantate
prinse** (fără scală; scala de două ori; dS transpus; semnul lui dS; fără „dot”; dQ ↔ dK; dK ↔ dV; dV pe triunghiul greșit; dK cu P în
loc de dS; capetele mutate) + modelul altui program înregistrat pentru acest cod.

**Ce NU e dovedit:** nimic pe placă. Viteza: un fir pe rând / pe element, fără memorie partajată — lent la T mare (T = 512: un fir
face ~T·d operații pentru dP), dar neglijabil la pașii de aur; varianta rapidă (plăci în memoria partajată, P recalculat ca la
FlashAttention, pentru #513) vine după ce merge drumul întreg.

## Ce e nou pentru placă (când va fi predat, C-160 punctul 6)
Un program nou, `attn_back`: bloc de **32** de fire și **grilă 3D** — primul program al nostru care citește `ctaid.z` (`sr[0x27]`,
textul lotului) și baza grupurilor la 0x8 (`ctaid.y` era deja în probele matmul); fără memorie partajată, fără barieră, fără MUFU;
două lansări la rând, a doua citește ce a scris prima (dS).

## Ce urmează (punctul 4, mai departe)
Atenția înainte pe T poziții (care scrie P), apoi încorporările (înainte / adunarea înapoi), norma + tăierea, AdamW.
