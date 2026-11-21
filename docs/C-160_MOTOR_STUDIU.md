# C-160 punctul 2 — Studiul motorului de antrenare: calculele unui pas și memoria plăcii

Sesiunea 140, 21 nov 2026. Totul aici e **DEDUS** (socotit pe hârtie din arhitectură); nimic nu e măsurat pe placă.
Arhitectura și rețetele: `Cyber-hub/integrare/kronos/PAS_DE_AUR.md`, `MODEL.md`, `MODEL_651.md`. Cifrele de memorie:
`tools/buget-antrenare/buget.py` (Python simplu, fără dependențe; `python3 tools/buget-antrenare/buget.py`).

Notații: B = lotul (câte texte), T = contextul (câte simboluri pe text), C = n_emb, H = capete, d = C/H, V = vocabularul,
L = straturi, P = parametri. Mic: L 2, C 64, H 4, V 256, T ≤ 32. #513: L 12, C 768, H 12, V 16 384, T ≤ 512. #651: la fel, L 24.

## 1. Pasul, în ordine (cum îl face PyTorch la „pasul de aur”)

**(0) gradienții la zero** (toți tensorii).

**(1) Înainte** — pentru fiecare calcul notez ce **păstrăm** pentru drumul înapoi (★):
1. `x = emb_token[intrari] + emb_pozitie[0..T−1]` (citire după indice + adunare).
2. Pe fiecare strat ℓ = 0 … L−1:
   1. LN1: medie, varianță pe n (nu n−1), `x̂ = (x − μ)·r`, `r = 1/√(var + 1e-5)`, `y = x̂·γ + β` — ★ x (intrarea), μ, r; ★ y.
   2. qkv = y·W_inᵀ + b_in (matrice × matrice, [B·T, C] × [C, 3C]) — ★ qkv.
   3. Atenția, pe fiecare cap: `S = Q·Kᵀ/√d`, masca cauzală (j > i → −∞), `P = softmax(S)` pe rând, `O = P·V`; capetele
      lipite la loc — ★ P (B·H·T·T numere) sau, ca la FlashAttention, doar maximul și suma pe rând (P se recalculează înapoi); ★ O.
   4. `a = O·W_outᵀ + b_out`; `x = x + a` — ★ x (intrarea lui LN2).
   5. LN2 ca LN1 — ★ μ, r, y.
   6. `h = y·W_fcᵀ + b_fc` ([B·T, C] × [C, 4C]) — ★ h (înainte de GELU).
   7. `g = GELU(h)` exact, `0,5·h·(1 + erf(h/√2))` — ★ g.
   8. `m = g·W_projᵀ + b_proj` ([B·T, 4C] × [4C, C]); `x = x + m`.
3. LN final — ★ x, μ, r, y.
4. `logits = y·emb_tokenᵀ` ([B·T, C] × [C, V], fără deplasare; capul e legat de `emb_token`).

**(2) Eroarea:** pe fiecare rând, `p = softmax(logits)`, `pierdere = −ln p[tinta]`, media pe toate cele B·T poziții, în F32.
Gradientul ei, scris **peste** logit-uri (nu mai trebuie altă memorie): `dlogits = (p − unu_la_tinta) / (B·T)`.

**(3) Înapoi** (ordinea inversă; „dX” = gradientul lui X):
1. Capul legat: `dy_final = dlogits·emb_token` ([B·T, V] × [V, C]) și `d emb_token += dlogitsᵀ·y_final` ([V, B·T] × [B·T, C]).
2. LN final înapoi: `dγ += Σ dy·x̂`, `dβ += Σ dy`, `dx = r·(dŷ − medie(dŷ) − x̂·medie(dŷ·x̂))` cu `dŷ = dy·γ`.
3. Pe fiecare strat, de la L−1 la 0 (gradientul firului principal `dx` trece neschimbat prin adunări și primește în plus):
   1. proj: `dW_proj += dmᵀ·g`, `db_proj += Σ dm`, `dg = dm·W_proj`.
   2. GELU: `dh = dg·(Φ(h) + h·φ(h))` (Φ = funcția de repartiție normală, φ = densitatea) — ★ h folosit aici.
   3. fc: `dW_fc += dhᵀ·y₂`, `db_fc += Σ dh`, `dy₂ = dh·W_fc`.
   4. LN2 înapoi (ca la 3.2) → se adună la `dx`.
   5. out: `dW_out += daᵀ·O`, `db_out += Σ da`, `dO = da·W_out`.
   6. Atenția înapoi, pe cap: `dV = Pᵀ·dO`, `dP = dO·Vᵀ`, `dS = P ∘ (dP − Σ_rând(dP ∘ P))`, apoi `·1/√d`; `dQ = dS·K`,
      `dK = dSᵀ·Q` (masca: pe pozițiile tăiate P = 0, deci dS = 0 singur).
   7. qkv: `dW_in += dqkvᵀ·y₁`, `db_in += Σ dqkv`, `dy₁ = dqkv·W_in`.
   8. LN1 înapoi → se adună la `dx`.
4. Încorporările: `d emb_token[intrari[b,t]] += dx[b,t]` (adunare pe rânduri, același simbol poate apărea de mai multe ori)
   și `d emb_pozitie[t] += Σ_b dx[b,t]`. **La capul legat, `d emb_token` = suma celor două contribuții** (punctul 3.1 + aici).

**(4) Tăierea gradientului** (doar `adamw_513`): `N = √(Σ g²)` pe **toți** tensorii împreună, `c = min(1; 1/(N + 1e-6))`, `g ← g·c`.

**(5) Pasul optimizatorului**, pe fiecare element: SGD `w ← w − 0,01·g`; AdamW (lr 5e-5, β 0,9 / 0,999, eps 1e-8, wd 0,1, pe toți
tensorii): `w ← w·(1 − lr·wd)`; `m ← β₁m + (1−β₁)g`; `v ← β₂v + (1−β₂)g²`; `w ← w − (lr/(1−β₁ᵗ))·m/(√v/√(1−β₂ᵗ) + eps)`.

### Programele de placă care rezultă (pentru punctul 4)
| # | Program | Folosit la | Avem azi ceva asemănător? |
|---|---|---|---|
| 1 | matrice × matrice F32, cu A sau B transpusă (NN, NT, TN) + deplasare opțională | toate cele 6 înmulțiri pe strat, înainte și înapoi, capul legat | `matmul_smem`/`matmul_2d` (probe F32), nu în drumul generării; generarea folosește matrice × vector Q8_0 — **nu ajută la antrenare** (greutățile trebuie F32 și se schimbă la fiecare pas) |
| 2 | suma pe coloane (gradientul deplasărilor) | `db_*` | nu |
| 3 | LayerNorm înainte pe rânduri, care păstrează μ și r | LN1, LN2, LN final | `layernorm256` (un rând; de lărgit la multe rânduri + scrierea lui μ, r) |
| 4 | LayerNorm înapoi (dx, dγ, dβ) | idem | nu |
| 5 | GELU înainte (păstrează h) / înapoi | MLP | `gelu.comp` (înainte) |
| 6 | atenția cauzală pe T poziții, înainte și înapoi | atenția | `attention.comp` e pentru un singur simbol nou (generare) |
| 7 | softmax + entropia încrucișată + gradientul ei, pe rânduri de V | eroarea | `argmax` citește rânduri de V, atât |
| 8 | încorporarea înainte / adunarea înapoi pe rânduri (fără `atomic`, ca rezultatul să fie la fel la fiecare rulare) | încorporări | `embed.comp` (înainte, un simbol) |
| 9 | norma totală (sumă de pătrate pe toți tensorii) + înmulțirea cu c | tăierea | nu |
| 10 | AdamW / SGD pe elemente | optimizatorul | nu |

Ordinea propusă pentru punctul 4: 1 (fără el nu merge nimic și e ~90 % din calcul) → 7 → 3/4 → 5 → 6 → 8 → 2 → 9 → 10.
Fiecare cu oracolul din punctul 3 (Rust pe procesor) și cu fișierele `grad_pas1/` ale pasului de aur mic ca ultimă probă.

### Calculul unui pas (DEDUS)
≈ **6·P operații pe simbol** pentru înmulțiri (2 înainte + 4 înapoi), plus atenția ≈ 12·L·C·T pe simbol.
#513, T 512: ~0,59 + 0,06 ≈ **0,65 GFLOP pe simbol**; un lot de 12 × 512 = **~4 TFLOP**. Înmulțirile sunt deci aproape tot
timpul: viteza motorului = viteza programului 1 (de aici C-160 p.7, circuitele speciale).

## 2. Memoria plăcii (DEDUS, `tools/buget-antrenare/buget.py`)

Bugetul: **7 GiB** (regula lui Adrian pentru workstation, 12 oct 2026, ca la Kronos în Windows), din care 256 MiB rezervă pentru
programe, inele și calcule de moment (atenția înapoi lucrată câte un text pe rând, ca să nu țină dP pentru tot lotul).

**Starea fixă** = greutăți + gradienți + m + v, toate F32 = 16 octeți pe parametru:

| Model | P | Stare fixă |
|---|---|---|
| mic | 118 528 | 1,8 MiB |
| #513 | 98 032 128 | **1,46 GiB** |
| #651 | 183 086 592 | **2,73 GiB** |

**Activările păstrate** pe un text (★ de mai sus; pe strat și simbol: 16·C numere + H·T probabilități; plus logit-urile B·T·V),
și **lotul maxim** care încape în 7 GiB:

| Model · T | F32, cu P | F32, P recalculat | intrările înmulțirilor pe 16 biți, cu P | 16 biți, P recalculat |
|---|---|---|---|---|
| mic · 32 | 0,33 MiB → ~21 000 | — | — | — |
| #513 · 64 (pasul de aur: lot 1) | 42,6 MiB → 127 | 40,5 → 133 | 26,8 → 202 | 24,6 → 220 |
| #513 · 256 | 198 MiB → 27 | 162 → 33 | 134 → 40 | 98 → 55 |
| #513 · 512 | 467 MiB → **11** | 324 → **16** | 340 → 15 | 197 → **27** |
| #651 · 64 | 80,9 MiB → 50 | 76,5 → 53 | 49,3 → 83 | 45,0 → 91 |
| #651 · 256 | 378 MiB → 10 | 306 → 13 | 251 → 16 | 180 → 22 |
| #651 · 512 | 899 MiB → **4** | 612 → **6** | 646 → 6 | 360 → **11** |

Ce înseamnă:
- **Pasul de aur mic și pasul de aur #513 (1 × 64) încap fără nicio grijă**, totul în F32 — exact ce trebuie pentru „aceleași numere”.
- **#513 cu lotul lui adevărat** (12 × 512 pe bucată, 8 bucăți adunate pe pas — `reteta.json`): în F32 cu P păstrat **nu încape**
  (11 maxim); cu P recalculat la întoarcere (ca FlashAttention) încape (16). Deci atenția fără matricea P întreagă e necesară
  devreme, nu doar pentru viteză. Bucățile se adună în gradienți (deja numărați), deci cele 8 nu cer memorie în plus.
- **#651, 24 de straturi:** 4 × 512 în F32; 6 cu P recalculat; 11 cu 16 biți + P recalculat. Se poate antrena, dar cu loturi
  mici adunate (de ex. 6 × 512 × 2 bucăți = același număr de simboluri ca 12 × 512).
- Dacă nici asta nu ajunge: recalcularea întregului strat la întoarcere (păstrăm doar intrarea stratului: C numere pe simbol în
  loc de ~16·C + H·T) — costă ~⅓ calcul în plus. Notat, nu propus acum.
- Ce nu am socotit: memoria OS-ului pentru altceva (modelul de generare nu trebuie ținut încărcat în timpul antrenării) și cât
  ocupă firmware-ul GSP în VRAM (zonele de sus; de citit din `gpu-jurnal` la primul test). De aceea bugetul e 7, nu 8.

## 3. Ce rămâne în F32 și ce poate merge pe 16 biți

Pe GA104, prin NAK, circuitele speciale oferă doar **f16 × f16 → f32** (`C-147_PUNCTUL5_TENSOR.md`: bf16 și tf32 nu sunt oferite
de NVK pe Ampere). Asta contează: f16 are domeniu mic (cel mai mare număr 65 504, cel mai mic normal ~6·10⁻⁵), deci gradienții
mici dispar fără **„scalarea erorii”** (eroarea înmulțită cu un factor, de ex. 2¹⁰–2¹⁶, iar gradienții împărțiți la loc în F32
înainte de tăiere și AdamW; factorul scade dacă apare un infinit, iar pasul acela se sare).

| Rămâne **F32** mereu | Poate merge pe **f16** (cu adunarea în F32) |
|---|---|
| greutățile „de bază”, gradienții adunați, m și v ale AdamW | intrările înmulțirilor: copia f16 a greutăților (refăcută după fiecare pas) și activările y, qkv, O, g |
| LayerNorm (medii, varianțe, μ, r), softmax-ul atenției și al erorii, eroarea | Q·Kᵀ și P·V (cu softmax-ul în F32) |
| norma pentru tăiere, pasul AdamW, firul principal x (adunările reziduale) | gradienții de pe drum dintre înmulțiri (cu scalarea erorii) |

**Ordinea propusă:** întâi totul în F32 (punctele 3–6): toleranțele pasului de aur sunt măsurate în F32 și nu le putem verifica
altfel. Abia după ce pasul de aur trece pe placă, f16 ca **pas de viteză** (C-160 p.7), cu o probă nouă: aceeași eroare pe cei 10
pași în toleranța pe care o scriem atunci (16 biți nu pot da „aceleași numere” ca F32).

## 4. Întrebări deschise (pentru Kronos, C-162, nu pentru Adrian)
1. Cum a încăput 12 × 512 la #513 în Windows/PyTorch pe 8 GB: F32 cu atenția „SDPA” (fără P întreagă), 16 biți (`autocast`), sau
   altfel? Fișa de antrenare (C-162) ar trebui să spună, ca să comparăm viteza cu același fel de calcul.
2. Lotul și contextul cerute pentru o antrenare adevărată a lui #651 (dacă va fi): de ele depinde dacă f16 e nevoie sau doar viteză.

## 5. Surse (clean-room: formule și idei, codul îl scriem noi)
- `Cyber-hub/integrare/kronos/PAS_DE_AUR.md` — arhitectura exactă, ordinea pasului, formulele SGD/AdamW și tăierii (citit azi).
- PyTorch (BSD-3), `tools/autograd/derivatives.yaml` și documentația `torch.optim.AdamW`, `clip_grad_norm_` — formulele înapoi.
- llm.c (Karpathy, MIT), `train_gpt2.c`: aceeași rețea GPT-2 cu funcțiile înapoi scrise de mână (LayerNorm, GELU, atenție,
  softmax + entropie, AdamW) și lista activărilor păstrate — a doua sursă pentru fiecare formulă de mai sus.
- tinygrad (MIT): derivatele generate automat — a treia sursă, la nevoie.
- Korthikanti et al., „Reducing Activation Recomputation in Large Transformer Models” (2022, arXiv 2205.05198) — socoteala
  memoriei activărilor (≈ 34·T·B·C + 5·H·T²·B octeți pe strat în 16 biți; a noastră e varianta F32 / minimă).
- Dao et al., „FlashAttention” (2022, arXiv 2205.14135) — P recalculat la întoarcere din maximul și suma pe rând.
- Micikevicius et al., „Mixed Precision Training” (2018, arXiv 1710.03740) — copia F32 a greutăților și scalarea erorii la f16.
- Loshchilov & Hutter, „Decoupled Weight Decay Regularization” (2019, arXiv 1711.05101) — AdamW.

Sursele 2–8 sunt din studiul meu anterior, **nerecitite în această sesiune**; formulele nu se iau pe încredere: oracolul Rust
(punctul 3) trebuie să dea gradientul pasului 1 și greutățile după pașii 1 și 10 ale pasului de aur mic, în toleranțele din fișă.
