# C-160 punctul 7, partea a treia — pasul de aur mic „mixt”: înmulțirile pe `mm_tc` + scalarea erorii (SIMULARE)

Sesiunea 158, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE** sau **DEDUS**: nimic nu a rulat pe RTX 3070 Ti.

## 0. Toleranța, scrisă ÎNAINTE de prima rulare (commit separat, înaintea codului)

Pasul de aur mic (`pas_de_aur_mic`: 2 straturi, C 64, 4 capete, vocabular 256, lot 2 × 32), aceeași listă ca T165, dar:
- **toate înmulțirile** (qkv, out_proj, fc, proj, capul legat — înainte și înapoi, cu dW-urile) merg pe `mm_tc` (intrări rotunjite
  la f16, sume f32 pe circuitele speciale) în loc de `mm_train`;
- **scalarea erorii, fixă, S = 2¹⁰ = 1 024**: `xent` scrie d(logits) = (softmax − onehot) · S / N; norma (`adamw_513`) e comparată cu
  S · 1,0; înainte de `opt_step` o înmulțire 1 × 1 × 1 cu `mm_train` (F32, exactă: S e putere a lui 2) face factorul c / S, pe care
  `opt_step` îl citește. Greutățile, gradienții adunați, m și v rămân F32. Niciun program nou în afară de `mm_tc`.

De ce S = 1 024: d(logits) e cam 1/256 / 64 ≈ 6·10⁻⁵ — exact la marginea de jos a f16 (cel mai mic număr normal: 6,1·10⁻⁵); sub ea
cifrele se pierd. × 1 024 → ~6·10⁻², departe de ambele margini (cel mai mare gradient de pe drumul înapoi ar trebui să fie ≪ 65 504 / 1 024
= 64). Dacă o intrare trece de 65 504, modelul `mm_tc` refuză lansarea (pe placă ar ieși „infinit” → norma nefinită oprește pasul).

**Criteriile (față de PyTorch F32, fișierele hub-ului; toate trebuie să treacă, la ambele rețete, pe 10 pași):**

| Ce | Toleranța mixtă | (Toleranța F32 de până acum) |
|---|---|---|
| Eroarea fiecăruia din cei 10 pași, relativ | **≤ 1·10⁻³** | 1·10⁻⁵ |
| Scăderea erorii (pasul 1 − pasul 10), relativ față de a lui PyTorch | **≤ 2 %** | — |
| Gradientul pasului 1 (după împărțirea la S), ‖Δ‖/‖g‖ pe fiecare tensor | **≤ 5·10⁻³** | 1·10⁻³ |
| Normele gradientului (`adamw_513`, după împărțirea la S), relativ | **≤ 2·10⁻³** | 1·10⁻⁵ |
| Greutățile după pasul 10, ‖Δ‖/‖w‖ pe fiecare tensor | **≤ 1·10⁻³** | (abs. 1·10⁻⁶ / 1·10⁻⁴) |

De unde: o înmulțire `mm_tc` greșește ~3·10⁻⁴ relativ (T168); înlănțuite pe 2 straturi, înainte și înapoi, cel mult câteva ori asta → 5·10⁻³
pentru gradient lasă loc. Eroarea e o medie pe 64 de simboluri — abaterile se mediază → 1·10⁻³ e larg. Greutățile se mișcă puțin în 10 pași
(lr 0,01 la `sgd`, 5·10⁻⁵ la AdamW), deci abaterea relativă a greutăților rămâne mică chiar dacă pașii diferă cu câteva procente.

**Predicția despre scalare (se măsoară, se raportează oricum ar ieși):** cu S = 1 (fără scalare), gradientul pasului 1 față de PyTorch iese
**mai rău decât cu S = 1 024**, de cel puțin 1,5 ori în cel mai rău tensor (`emb_token`, care primește dW-ul capului din d(logits)).

**Greșeli plantate care trebuie prinse:** împărțirea la S uitată (gradienți × 1 024); S pus în eroare dar norma comparată cu 1,0 (tăiere greșită).

Ce NU dovedește: nimic despre placă (`mm_tc` n-a rulat pe RTX 3070 Ti); scalarea e fixă — scalarea dinamică (micșorarea lui S la „infinit”)
cere citirea normei de procesor între pași și vine doar dacă un antrenament lung o cere.
