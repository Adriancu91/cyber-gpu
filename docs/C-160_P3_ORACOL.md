# C-160 punctul 3 — oracolul pasului de antrenare (pe procesor)

Sesiunea 141, 21 noiembrie 2026. **Totul pe procesor (cloud), nimic pe placă.** E unealta cu care verificăm, pe rând, programele
de antrenare pentru placă (C-160 punctul 4); nu e un drum de antrenare pe procesor.

## Ce face

`crates/gpu-compute/src/train.rs` (modul nou, doar adăugare; interfața publică veche neschimbată):

| Funcție | Ce face |
|---|---|
| `Arch`, `MIC`, `M513`, `Arch::tensors()` | forma rețelei Kronos și lista tensorilor în ordinea PyTorch (28 la modelul mic, 148 la #513) |
| `loss_and_grad` | înainte (încorporări, 2 × [LN1 → atenția cauzală → LN2 → MLP cu GELU exact], LN final, capul legat) → eroarea (entropia încrucișată, media pe lot × context) → înapoi prin tot, cu gradientul `emb_token` = suma celor două folosiri |
| `grad_norm`, `clip_grad_norm` | norma totală și tăierea, ca `clip_grad_norm_` |
| `sgd_step`, `AdamW` | cele două rețete, în ordinea și rotunjirile F32 ale lui PyTorch 2.14 („single tensor”: scăderea greutăților întâi, `m` prin `lerp`) |
| `golden_run` | pasul de aur: de N ori zero → înainte → înapoi → (tăiere) → optimizator, pe același lot |

Tensorii sunt F32, ca în PyTorch; **toate sumele se fac în F64** și se rotunjesc o dată — oracolul stă cât mai aproape de valoarea
exactă, iar comparația cu PyTorch (F32, altă ordine a adunărilor) se judecă cu toleranțele din `PAS_DE_AUR.md`, scrise înainte.

## Rezultatul (T151, MĂSURAT pe procesor, față de fișierele PyTorch)

| Ce comparăm | `sgd` | `adamw_513` | Toleranța (din fișă) |
|---|---|---|---|
| eroarea pe cei 10 pași, relativ | ≤ 1,1e-7 | ≤ 8,8e-8 | ≤ 1e-5 |
| gradientul pasului 1, absolut | ≤ 3,6e-7 | ≤ 3,6e-7 | ≤ 1e-5 |
| gradientul pasului 1, ‖Δ‖ / ‖g‖ | ≤ 6,3e-7 | ≤ 6,3e-7 | ≤ 1e-3 |
| greutățile după pasul 1, absolut | ≤ 1,2e-7 | ≤ 1,9e-6 | 1e-6 / 1e-4 |
| greutățile după pasul 10, absolut | ≤ 2,4e-7 | ≤ 1,05e-5 | 1e-6 / 1e-4 |
| eroarea pasului 1 → 10 | 44,193806 → 33,886936 | 44,193806 → 43,21414 | (PyTorch: 44,193802 → 33,886936 / 43,214138) |

Abaterile sunt de același ordin cu ce a măsurat Kronos între 1 și 16 fire pe același procesor (`PAS_DE_AUR.md`). La AdamW după pasul 1
abaterea (1,9e-6) e mai mare decât la Kronos (1,8e-11), cum prevede fișa: unde gradientul e aproape zero, `m/√v` ≈ semnul lui, deci o
rotunjire diferită mută elementul cu până la 2·lr; rămâne de 50 de ori sub toleranță.

Verificări în plus:
- gradientul verificat **independent de PyTorch** cu diferențe finite pe un model minuscul (2 straturi, n_emb 8), la 3 elemente din
  fiecare tensor;
- greșelile plantate sunt prinse: un gradient mai mare cu 0,2 % pe un singur tensor, greutățile SGD comparate cu cele AdamW, țintele
  mutate cu o poziție;
- `erf`, `exp`, `sqrt` în F64 (fără libm) față de `std` și de tabelul A&S.

Rândurile `kt c160p3 …` (afișate de test cu `--nocapture`) dau abaterile maxime și tensorul în care apar.

## Ce urmează (C-160 punctul 4)

Programele de placă, câte unul, în SIMULARE, fiecare comparat cu bucata lui din acest fișier: întâi înmulțirea matrice × matrice
(și cu transpusa: `linear` / `linear_back` de aici), apoi eroarea cu gradientul ei, înapoi prin LayerNorm, GELU, atenție, adunarea la
tabela de simboluri, norma + tăierea, AdamW. Pentru #513 (`M513`) oracolul merge la fel, dar setul de date (2,7 GB) e doar pe
workstation; aici se verifică doar forma (148 de tensori, 98 032 128 de parametri).
