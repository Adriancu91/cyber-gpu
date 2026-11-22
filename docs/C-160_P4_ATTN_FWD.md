# C-160 punctul 4 — programul 6 al antrenării, jumătatea „înainte”: atenția cauzală pe T poziții (`attn_fwd`)

Sesiunea 149, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE**: nimic nu a rulat pe RTX 3070 Ti.

## Ce e
Programul care face atenția lui Kronos (cu masca cauzală) la **antrenare**: toate cele T poziții ale tuturor textelor lotului
odată (la generare, `attention.comp` face o singură poziție nouă pe cache-uri). Intră qkv [B·T, 3C] (ieșirea lui `in_proj`).
Ies **att** [B·T, C] (capetele unul lângă altul, gata pentru `out_proj`) și **P** [B, H, T, T] — probabilitățile softmax, cu +0
deasupra diagonalei — păstrate pentru drumul înapoi: exact ce citește `attn_back` (sesiunea 148). O singură lansare, grila
[⌈T/32⌉, H, B], blocuri de 32, un fir pe rândul (b, h, i):

s_j = (Σ_e Q_ie·K_je)·(1/√d) → m = max s_j → x_j = exp(s_j − m) → Σ x → P_ij = x_j·(1/Σ) pentru j ≤ i, 0 pentru j > i →
att_ie = Σ_j P_ij·V_je.

**Masca** = marginile buclelor (P deasupra diagonalei e scris +0, niciodată citit). Fiecare sumă e lanțul unui singur fir, în
ordine fixă: fără `atomic`, memorie partajată sau barieră → aceiași biți la fiecare rulare. Rândul lui P e folosit ca loc de
lucru (s, apoi x, apoi P), deci nu trebuie memorie în plus.

`tools/nak/attn_fwd.comp` → Mesa NAK 25.0.7: **3 536 octeți**, 221 de instrucțiuni, 25 de registre; SHA-256
`fe97ab49c5b04303d0e2a2e2a9d7bc9c22d8bfbaaf2e8cbfa3e41104dbaa5596`. Bufferul constant 0: qkv / P / att la 0x28 / 0x30 / 0x38,
T 0x40, C 0x44, H 0x48, d 0x4c, scala 0x50 (0x54 octeți); baza grupurilor la 0x0 / 0x4 / 0x8 rămâne 0; B = z-ul grilei.

Aritmetica, citită din `tools/nak/out/attn_fwd.sm86.txt` (testul o verifică): 2 lanțuri `ffma.ftz` (Q·K și P·V, de la 0,
crescător), `fmul.ftz` cu scala, `fmnmx.ftz`, `fadd.ftz`(s, −m), `fmul.ftz` cu log₂e = `0x3fb8aa3b`, **`mufu.exp2`**, lanțul
`fadd.ftz` al sumei, **`mufu.rcp`** (NAK nu adaugă pas Newton) și `fmul.ftz`(x, 1/Σ). Modelul: `gpu_compute::layer::{attn_fwd_row_gpu,
attn_fwd_gpu}` (operație cu operație; EX2 și RCP modelate rotunjit corect — ale plăcii sunt aproximative, deci judecăm în margine,
nu bit cu bit). Oracolul: `gpu_compute::train::attention_rows` (f64; exact liniile din `loss_and_grad`, T151).

## Marginea, scrisă înainte de măsurare (T160)
u = 2⁻²⁴, MUFU = 2⁻²² (locul pentru unitățile aproximative ale plăcii, ca la RSQ în T157). Pe rândul (b, h, i), cu valorile exacte s*, P*, a* = s* − max s*:
- s_j: Δs_j = s·(1,01·d·u·Σ|Q·K| + 2d·FTZ) + 2,01·u·|s*_j| (+ f64 al oracolului);
- x_j: ε_j = Δs_j + 3,01·u·|a*_j| + MUFU (relativ; m se anulează în softmax, deci nu aduce eroare proprie);
- Σ: ε_Σ = Σ_j P*_j·ε_j + 1,01·(i+1)·u;
- P_j: ΔP_j = 1,05·P*_j·(ε_j + ε_Σ + 2,01·u + MUFU) + (i+3)·FTZ;
- att: Σ_j ΔP_j·|V_je| + 1,01·(i+1)·u·Σ_j (P*_j + ΔP_j)·|V_je| + FTZ + rotunjirea oracolului la f32.

## Ce a ieșit (SIMULARE, modelul lansat prin procesorul de comenzi față de oracol)
| Caz | P (cât din margine) | att (cât din margine) | max \|Δ\| att |
|---|---|---|---|
| forma modelului mic: B 2, T 32, C 64, H 4 | 8,0 % | 3,7 % | 2,4e-7 |
| T 70 (trece de blocul de 32), C 96, H 3 | 4,4 % | 2,5 % | 1,2e-7 |
| T 33, C 80, H 2 (d 40) | 3,4 % | 1,9 % | 8,9e-8 |
| T 5, C 8, H 1 (B 3) | 10,9 % | 3,2 % | 1,8e-7 |
| T 17, C 3, H 3 (d 1) | 13,6 % | 4,0 % | 1,2e-7 |
| softmax ascuțit (qkv × 6) | 9,3 % | 2,8 % | 8,3e-6 |
| foarte ascuțit (qkv × 30: exponenți până în zona FTZ) | 4,4 % | < 0,1 % | 1,9e-6 |
| T 130, C 32, H 2 | 7,1 % | 6,5 % | 1,2e-7 |

La „foarte ascuțit” marginea lui att e largă (eroarea lui s, ~1e-3 relativ, trece prin exp); programul stă mult sub ea.
În plus: lansarea = modelul, bit cu bit; două rulări = aceiași biți; fiecare rând al lui P însumează 1; grila scurtă scrie doar
rândurile ei; NaN lăsat deasupra diagonalei de un P vechi e rescris cu +0; refuzuri (bloc de 64, grilă peste H sau peste ⌈T/32⌉,
formă greșită: nimic scris). **Lanțul pasului de antrenare:** P-ul acestui program dat direct lui `attn_back` → dQ, dK, dV la
≤ 2⁻¹⁴ din cea mai mare valoare față de `attention_back_rows` cu P-ul în f64 (6 forme); și **cele două programe pe același procesor
de comenzi** (înainte, apoi rândurile lui dS) = modelul lui `attn_back` cu acest P, bit cu bit. **8 greșeli plantate prinse**
(scala de două ori, fără scală, Q ↔ K, V citit din K, capetele mutate, P transpus, masca deplasată cu unu, fără 1/Σ).

**Ce NU e dovedit:** nimic pe placă; nici cât de aproximative sunt EX2 / RCP ale lui GA104 (am lăsat 2⁻²² fiecare). Viteza: un fir pe
rând, fără memorie partajată — bun pentru pașii de aur (T ≤ 64), lent la T = 512; varianta rapidă (plăci, P recalculat ca la
FlashAttention) după ce merge drumul întreg (`docs/C-160_MOTOR_STUDIU.md`).

## Ce e nou pentru placă (când va fi predat, C-160 punctul 6)
Programul `attn_fwd`: bloc de 32, grilă 3D (ca `attn_back`), **`mufu.exp2` și `mufu.rcp`** (aceleași unități ca în programul de atenție
al generării, `attention.comp`, care rulează pe placă la fiecare `test-placa`), fără memorie partajată / barieră.

## Ce urmează (punctul 4, mai departe)
Încorporările (înainte / adunarea înapoi pe tabela de simboluri), norma + tăierea gradientului, AdamW. Cu ele, punctul 5: pasul de
aur mic cap-coadă pe placa simulată.
