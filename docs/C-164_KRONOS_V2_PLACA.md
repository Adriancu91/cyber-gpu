# C-164 — Kronos croit pe placă: partea Cyber-GPU

Sesiunea 212 (24 nov 2026), în cloud, FĂRĂ placă. Doar proiect și SIMULARE (pe procesor). Nimic din ce e aici nu schimbă
`kronos_gpt`, `kronos/` sau pasul care rulează azi pe placă. Alegerea arhitecturii e a lui Kronos și a lui Adrian.

Codul: `gpu_compute::v2` (alegerile, numărătoarea lansărilor, parametrii, munca înmulțirilor, oracolele noi) + T218
(`nvidia-ga104/tests/c164_kronos_v2.rs`: numărătoarea pentru arhitectura de azi = listele planului de pe placă, la MIC și la #513).

## Întrebarea

Pentru fiecare candidat (ReLU² în loc de GELU, normă RMS în loc de LayerNorm, fără deplasări, lățimi ÷ 64, capul): ce programe
de placă dispar sau se simplifică și cât câștigă pasul.

## Predicții (scrise înainte de cod)

| # | Predicție | Ieșit |
|---|---|---|
| 1 | Pasul #513 are azi 31 de lansări pe strat + 13 = 385, din care 73 de `colsum` | **385 / 73** — numărat și pus față în față cu `Plan::step` (T218) ✔ |
| 2 | Fără deplasări dispar 4 `colsum` pe strat (48) | **337 lansări, 25 `colsum`** ✔ |
| 3 | Ideea centrului: „~5 feluri de programe în loc de 13” | **GREȘIT**: felurile rămân ~14 (atenția, încorporările, eroarea, optimizatorul nu se schimbă; lipirile sunt variante ale programelor de azi). Scad **lansările**, nu felurile. |
| 4 | dγ al normei RMS cere un program nou | **GREȘIT**: e exact formula `colsum` de azi (modul LayerNorm) cu media = 0 — abaterea ≤ 1e-6 pe 1 024 × 768 (testul `rms_dgamma_is_todays_colsum_formula_with_zero_mean`) |

## Ce face fiecare candidat (#513, lot 2 × 512, timpii de azi din os-410)

Timpii mici măsurați în os-410 (pasul 483 ms): `colsum` 24,3 ms / 73 = 332 µs pe lansare, `gelu_tr` 60 µs, `vadd` 17 µs,
`ln_fwd` 23 µs, `ln_back` 25 µs. Programele RMS / ReLU² nu există: le dau timpul celui pe care îl înlocuiesc (toate sunt treceri
prin memorie peste aceleași tablouri) — **DEDUS**.

| Candidat | Lansări / pas | `colsum` | Parametri | Ce dispare / se schimbă pe placă | Câștig azi (DEDUS) | După `colsum` împărțit (T216, ~2,5–5 ms) |
|---|---|---|---|---|---|---|
| azi (GELU, LayerNorm, deplasări) | 385 | 73 | 98 032 128 | — | — | — |
| ReLU² | 385 | 73 | la fel | `gelu_tr` → `relu2` (o comparație + o înmulțire în loc de erf); **aceiași biți ca oracolul** (fără toleranță); lipirea în `fc` devine trivială | ~0 ms (programul e oricum mic: 1,4 ms în total) | ~0 |
| fără deplasări | **337** | 25 | −102 144 | cele 4 sume de deplasare pe strat (48); adunarea deplasării din înmulțiri (steagul `BIAS`) | **−15,9 ms (−3,3 %)** | −1…−2 ms |
| RMS cu γ | 385 | 73 | −19 200 | `ln_fwd` / `ln_back` → variante RMS (fără medie: o reducere în minus); `colsum` rămâne (doar dγ) | <0,5 ms | ~0 |
| **toate** (ReLU², RMS fără γ, fără deplasări) | **312** | **0** | −121 344 | toate cele 73 de `colsum`; normele fără greutăți | **−24,3 ms (−5,0 %)** | −2,5…−5 ms |
| toate + cele 2 lipiri de azi (`ln_back_res`, activarea înapoi în `proj-dX`) | 276 | 0 | | + −24 `vadd`, −12 activări | −25,4 ms | |
| toate + activarea și înainte în `fc` (lipire nouă) | 264 | 0 | | + −12 activări | −26 ms | |

Ce înseamnă cinstit: **azi** câștigul vine aproape tot din dispariția sumelor pe coloane (`colsum`, 5 % din pas). Dar `colsum`
împărțit (gata în SIMULARE, T216, de legat de OS) le aduce oricum la ~2,5–5 ms — după el, candidații mai câștigă **1 %** sau mai puțin
din pasul de azi. Ponderea lor crește doar când înmulțirile se fac mult mai repede (tensor cores): la un pas de ~100 ms, cele
~28 ms de programe mici de azi ar fi ~1/4 din pas, iar „toate” le-ar tăia aproape de tot. Cifra „+10–20 %” din idee e posibilă
**doar după** pista A (înmulțirile rapide), nu acum.

## Lățimile și capul

- **#513 se potrivește deja pe plăci**: C 768, 3C 2 304, 4C 3 072, V 16 384, T 512, rânduri 1 024 — toate ÷ 64 și ÷ 128; singura
  lățime ne-÷128 e capul de atenție (d = 64), iar plăcile atenției sunt oricum de 64 (`widths_off_tile`). Pentru modelul nou:
  C multiplu de 128, d = 64, V multiplu de 128, lotul × contextul multiplu de 128.
- **Capul (legat de `emb_token`)** = 12,9 % din munca înmulțirilor (77,3 din 599,1 GFLOP pe pas). Un cap dezlegat nu schimbă
  calculul (aceleași înmulțiri), dar adaugă 12,6 M parametri (+~200 MB: greutăți, gradient, m și v ale AdamW). Un vocabular de 8 192 ar tăia jumătate
  din cap (−6,5 % din înmulțiri) — e decizia lui Kronos (tokenizatorul, calitatea), nu a motorului.

## Ce programe noi ar trebui (toate mici, toate variante ale celor de azi)

| Program | Din ce se face | Oracolul pe procesor |
|---|---|---|
| `rms_fwd` | `ln_fwd` fără medie | `v2::rms_rows` (γ sau fără γ) ✔ |
| `rms_back` / `rms_back_res` | `ln_back` / `ln_back_res` fără termenul `− mean(dx̂)` | `v2::rms_back_rows` ✔ (verificat cu diferențe finite) |
| dγ al RMS | **nimic nou**: `colsum` în modul LayerNorm cu media = 0 | `v2::dgamma_is_colsum_with_zero_mean` ✔ |
| `relu2` (înainte / înapoi) | `gelu_tr` cu altă formulă | `v2::relu2_rows`, `relu2_back_rows` ✔ (biți identici cu formula f32 a plăcii) |
| `mm_train_relu2` | `mm_train_gelu` cu altă formulă în scriere | `relu2_back_rows` după `linear_back` |

Nimic din acestea nu e scris încă pentru placă; le facem doar dacă Kronos + Adrian aleg candidații (C-164 merge mai departe
prin laboratorul C-163: calitatea se măsoară acolo, nu aici).

## Ce nu am evaluat

Pozițiile rotative și QK-norm (din idee, nu din lista C-164): ar scoate `emb_pozitie` și ar adăuga lucru în atenție — de socotit
dacă Kronos le propune.

## Rezultat (sesiunea 212, SIMULARE / numărare)

| Test | Ce verifică | Rezultat |
|---|---|---|
| T218 (2 teste) | numărătoarea pentru azi = `Plan::step` al planului `_3`, simplu și cu cele 2 lipiri, `adamw_513` și `sgd`, la MIC și #513 | PASS (pe procesor) |
| `v2::tests` (7 teste) | parametrii azi = tensori `kronos_gpt` (98 032 128); candidații; felurile; lățimile; 599,1 GFLOP; ReLU² biți + derivată; RMS diferențe finite (cu și fără γ); dγ = `colsum` cu media 0 | PASS (pe procesor) |
| câștigurile în ms | din timpii os-410 | DEDUS / NOT TESTED pe placă |
