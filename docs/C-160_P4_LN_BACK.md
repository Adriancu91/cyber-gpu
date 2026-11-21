# C-160 punctul 4 — programele 4 și 2 ale antrenării: LayerNorm înapoi (`ln_back`) și sumele pe coloane (`colsum`)

Sesiunea 144, 21 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE**: nimic nu a rulat pe RTX 3070 Ti.

## Ce e
„Înapoi” prin LayerNorm are trei rezultate: dx (pe rânduri) și dγ, dβ (pe coloane, adunate peste toate rândurile lotului). Le-am
despărțit în **două programe**, ca niciun rând să nu adune în coloana altuia (fără `atomic`: aceiași biți la fiecare rulare — regula din studiu):

- **`ln_back`** (`tools/nak/ln_back.comp`, 2 736 octeți, 19 registre, 64 de octeți partajați, o barieră; SHA-256 `369afd8d…85cd7063`):
  un bloc de 256 de fire pe rând; cu μ și r = 1/√(var + eps) păstrate de „înainte”: x̂ = (x − μ)·r, g = dy·γ, m1 = Σg/C, m2 = Σg·x̂/C,
  **dx = r·(g − m1 − x̂·m2)**. Cele două sume merg împreună (fluture în warp, cele 8 sume de warp în ordine, ca la `layernorm256`).
- **`colsum`** (`tools/nak/colsum.comp`, 1 136 octeți, 19 registre, fără memorie partajată; SHA-256 `7a0fa2bd…a47fb10`): un fir pe
  coloană, rândurile adunate în ordine. Modul „deplasare”: Σ dy (**programul 2 din studiu**, gradientul oricărei deplasări). Modul
  „LayerNorm”: **dγ = Σ dy·x̂** și **dβ = Σ dy** (x̂ recalculat din μ, r — aceleași două operații ca în `ln_back`).

Fără unități MUFU, fără împărțiri (1/C îl dă apelantul): doar adunări, înmulțiri, fma. Am citit codul făcut de NAK (`*.sm86.txt`) și
modelul pe procesor (`gpu_compute::layer::ln_back_gpu256`, `colsum_gpu`) face aceleași operații în aceeași ordine — **cu o surpriză notată:**
în a doua trecere NAK a lipit `g − m1` într-un singur `ffma` (dy·γ − m1, fără rotunjirea lui g); modelul a fost corectat după listare.
Oracolul: `gpu_compute::train::layernorm_back_rows` = exact calculul din `loss_and_grad` (cel care a dat numerele PyTorch la pasul de aur
mic, T151), în f64. Adăugat și `train::layernorm_stats` (μ, r rotunjite la f32, cum le-ar păstra programul „înainte”). Nimic redenumit.

## Marginile, scrise înainte de măsurare
- dx (T154): ĥ greșește cu r·u·|μ| + 5u·|ĥ| (μ și r rotunjite, scăderea, produsul, plus x̂ păstrat în f32 de oracol); sumele cu
  (⌈C/256⌉ + 12) rotunjiri pe drum; m1, m2 cu 2u în plus; t = g − m1 − ĥ·m2 cu 2 rotunjiri; dx = r·t cu 3u·|dx| + FTZ; u = 2⁻²⁴.
- dγ, dβ (T155): R rânduri → R·u·Σ|termeni| + eroarea lui ĥ pe fiecare termen + rotunjirea oracolului + FTZ.

## Ce a ieșit (SIMULARE, modelul față de oracol)
| Caz | `ln_back`: max \|Δdx\| (cât din margine) |
|---|---|
| C = 64 / 255 / 256 / 257 | 1,9e-9 (11 %) / 9,3e-10 (12 %) / 1,9e-9 (12 %) / 9,3e-10 (18 %) |
| C = 768 (lățimea lui #513) / 1 000 / 2 048 | 1,9e-9 (12 %) / 3,0e-8 (12 %) / 2,8e-9 (15 %) |
| C = 768, media 100 (μ rotunjit contează) | 7,5e-9 (21 %) |
| γ-urile reale ale modelului mic după 10 pași `sgd` (cele 5 LayerNorm-uri), 64 × 64 | 1,9e-9 (≤ 19 %) |

| Caz (rânduri × C) | `colsum`: dγ / dβ (cât din margine) |
|---|---|
| 64 × 64 (lotul pasului de aur mic) / 64 × 256 | 1,8 % / 2,3 % — 3,5 % / 2,0 % |
| 3 × 257 / 300 × 100 / 512 × 768 / 2 048 × 64 | 24 % / 50 % — 0,6 % / 0,5 % — 0,7 % / 0,4 % — 0,09 % / 0,07 % |
| 64 × 768, media 100 | 12 % / 3,5 % |

Suma deplasării = dβ bit cu bit (aceleași adunări). Σ dx ≈ 0 pe fiecare rând (cum cere LayerNorm). Două rulări = aceiași biți. Refuzuri
(bloc greșit, fără memorie partajată / barieră: nimic scris), grila prea scurtă (coloanele neatinse rămân vizibile) și greșeli plantate
(modelul altui program; μ / r ale altui rând; dx fără termenul x̂·m2; dx fără r; dβ în locul lui dγ; un rând lipsă) — toate prinse.
Marginea pentru multe rânduri e o margine de cel mai rău caz (crește liniar cu R), de aceea fracțiile sunt mici la 2 048 de rânduri.

**Ce NU e dovedit:** că placa face exact aceste operații (fără MUFU ar trebui să fie bit cu bit — se vede doar pe placă); viteza.
`colsum` e prima versiune, corectă, nu rapidă: la C = 768 sunt doar 3 blocuri (placa are 48 de SM-uri) și fiecare fir adună toate rândurile
lotului (6 144 la #513) — mai târziu, rândurile împărțite pe mai multe blocuri + o a doua trecere. Cu 6 144 de rânduri adunate în f32 pe
rând, abaterea relativă poate crește spre 1e-4 în cel mai rău caz; pentru pașii de aur (64 de rânduri) e neglijabil.

## Ce e nou pentru placă (când va fi predat, C-160 punctul 6)
Două programe noi: `ln_back` (același fel de QMD ca `layernorm256` / `xent`: 256 de fire, 64 de octeți partajați, o barieră; bufferul
constant 0x60 octeți) și `colsum` (QMD simplu, fără memorie partajată; bufferul constant 0x64 octeți). Nepredate: punctul 6 cere întâi un
singur program „înapoi” pe placă — `ln_back` sau `colsum` sunt candidați buni (fără MUFU, deci comparabile bit cu bit cu modelul);
ordinea o hotărăște centrul de comandă.

## Ce urmează (punctul 4, mai departe)
Programul 3 (LayerNorm înainte pe multe rânduri, care scrie μ și r) — leagă `ln_back` de „înainte” — apoi 5 (GELU înainte / înapoi).
