# C-160 punctul 7, partea întâi — estimarea vitezei de antrenare, scrisă ÎNAINTE de orice măsurătoare

Sesiunea 156, 22 nov 2026, în cloud, **fără placă**. Totul de aici e **DEDUS** (socoteală), nimic MĂSURAT de noi. Cifrele se refac cu
`python3 tools/buget-antrenare/viteza.py`. Scopul: când vine prima măsurătoare de viteză a înmulțirilor de antrenare pe placă, avem
dinainte scris ce așteptăm și ce ar însemna o abatere.

## 1. Cifra de bătut (MĂSURAT de Kronos, `Cyber-hub/integrare/kronos/FISA_ANTRENARE.md` §1)
- #513: **~39 300 simboluri/s** (lot 12 × 512, 8 bucăți pe pas); #651: **~15 500 simboluri/s** (cu recalculul activărilor).
- PyTorch calculează în **bf16 (autocast)** pe circuitele speciale; greutățile, gradienții și AdamW în F32.

## 2. Ce înseamnă în operații (DEDUS)
- Pe simbol: înmulțirile 6·P (2 înainte + 4 înapoi; 8·P cu recalculul) + atenția ~12·L·C·T.
  #513: **0,645 GFLOP**; #651 (cu recalcul): **1,578 GFLOP**.
- Deci PyTorch scoate efectiv **~25 TFLOP/s** pe ambele modele.
- Plafoanele plăcii (din specificațiile publice: 48 de SM-uri, ~1,77 GHz, 128 FMA F32 pe ceas pe SM, 256 FMA f16→f32 pe circuitele
  speciale pe ceas pe SM; **ceasul real sub sarcină nu e măsurat**, poate fi ±10 %): **F32 ~21,7 TFLOP/s**, **f16 cu adunare F32
  ~43,5 TFLOP/s**.
- **Concluzia cea mai importantă:** PyTorch scoate deja ~116 % din plafonul F32. **Un motor numai F32 nu poate ajunge la PyTorch pe
  această placă, oricât de bun ar fi** — cel mult ~0,5×. Ca să fim „mai repede decât PyTorch” (ținta C-160) trebuie circuitele
  speciale (f16 × f16 → f32, singurul fel oferit de NAK pe GA104), la **~66–68 % din plafonul lor** pe înmulțiri, cu restul pasului
  (LayerNorm, GELU, softmax, eroarea, AdamW, driverul) sub ~15 %.

## 3. Estimarea, motor cu motor (DEDUS; 15 % din pas pentru ce nu e înmulțire)

| Motorul | TFLOP/s pe înmulțiri | #513 simb/s | față de PyTorch | #651 simb/s |
|---|---|---|---|---|
| `mm_train` de azi (plăci 16×16, F32, un rezultat pe fir) | ~2,6 (12 % din F32) | ~3 400 | ×0,09 | ~1 400 |
| F32 cu plăci în registre (ca SGEMM-urile clasice) | 10,9–14,1 (50–65 %) | 14 300–18 600 | ×0,37–0,47 | 5 900–7 600 |
| f16 pe circuitele speciale, slab / mediu / bun | 17,4 / 23,9 / 30,4 (40 / 55 / 70 %) | 22 900 / 31 500 / 40 100 | ×0,58 / 0,80 / 1,02 | 9 400 / 12 900 / 16 400 |

De ce `mm_train` ar fi ~12 %: fiecare FMA cere două citiri din memoria partajată (nicio refolosire în registre), deci frâna e memoria
partajată, nu calculul (DEDUS din codul NAK, `docs/C-160_P4_MATMUL.md`). E corect, nu rapid — așa a fost ales.

## 4. Ce măsurăm și în ce ordine (fiecare cu estimarea de mai sus scrisă dinainte)
1. **După ce pasul de aur mic trece pe placă** (C-160 p.6): un rând `kt` cu timpul `mm_train` la formele lui #513 pe o bucată de
   12 × 512 = 6 144 de rânduri (qkv înainte 6 144 × 768 → 2 304; fc 768 → 3 072; proj 3 072 → 768; dW-urile cu `TA`).
   **Așteptat: ~2–3 TFLOP/s.** Peste 5 = frâna nu e unde credem; sub 1 = altceva (QMD, grila, ceasul) — de lămurit înainte de orice.
2. **Programul nou `mm_tc`** (f16 × f16 → f32 pe circuitele speciale, plăci mari în memoria partajată, aceeași interfață ca `mm_train`:
   `TA`, `TB`, `BIAS`), întâi în SIMULARE cu oracolul și marginea scrisă pentru intrări rotunjite la f16 (P8 a trecut pe placă cu
   identitate 256/256 și 148 ‰ din toleranță, os-335). Pe placă doar ca probă separată, un singur lucru nou pe test.
   **Așteptat: 17–30 TFLOP/s** (40–70 %); sub ~17 = nu bate nici F32-ul bun → întâi plăcile, abia apoi pasul întreg.
3. Pasul întreg cu `mm_tc` pe lotul lui #513: proba de corectitudine nu mai e „aceleași numere” (f16 ≠ F32), ci **aceeași eroare pe
   cei 10 pași în toleranța scrisă atunci**, plus scalarea erorii (studiul, §3). Viteza în simboluri/s, comparată cu 39 300.

## 5. Ce rămâne nesigur (de spus cinstit)
- Ceasul plăcii sub sarcină lungă (plafoanele pot fi ±10 %; tabloul de bord îl poate citi).
- Cât din pas e „restul” (15 % e presupus; PyTorch are programe lipite pentru AdamW și LayerNorm). Se măsoară la pasul 3.
- Dacă NAK face din `mm_tc` cod care ține circuitele speciale ocupate (încărcările din memoria partajată, `ldmatrix`): se vede în
  listarea SASS înainte de placă.
- Comparația e cu bf16 la PyTorch; noi am avea f16 (alt domeniu, aceeași viteză pe circuite) — de aici scalarea erorii.

## 6. Adăugat după os-349 (sesiunea 159, 22 nov 2026): prima măsurătoare a unui program „înapoi” schimbă o presupunere
- `colsum` a trecut pe placă bit cu bit (`docs/C-157_P2_COMPARATII.md`, os-349). Timpul lui (MĂSURAT, buffer în RAM-ul procesorului):
  ~0,6–0,7 µs pe rând, pentru că un fir adună singur toate rândurile coloanei lui.
- Socoteala (DEDUS, `python3 tools/buget-antrenare/colsum_timp.py`): cele 73 de `colsum` ale unei bucăți #513 ar lua **~165–275 ms**, mai
  mult decât tot pasul PyTorch (156 ms). Cu „restul ~15 %” de la §3 nu se poate ajunge până nu împărțim `colsum` pe rânduri (podeaua
  ~6,6 ms = 4 %). **Presupunerea „15 %” rămâne, dar doar pentru un `colsum` împărțit**; la fel trebuie privite `ln_back`, `attn_back`
  și AdamW înainte de măsurătoarea pasului întreg (punctul 3 de la §4).

## Surse
- Formula 6·P și atenția: studiul `docs/C-160_MOTOR_STUDIU.md` §1 (Kaplan et al. 2020, „Scaling Laws”, numărarea operațiilor; llm.c MIT).
- Plafoanele: specificațiile publice NVIDIA pentru RTX 3070 Ti (48 SM, ceasul de boost) și raportul „NVIDIA Ampere GA102 GPU
  Architecture” (rata circuitelor speciale GA10x cu adunare F32 = jumătate din cea cu adunare f16). Nimic copiat; doar cifre.
- Viteza PyTorch: fișa lui Kronos (C-162), MĂSURAT de ei.
