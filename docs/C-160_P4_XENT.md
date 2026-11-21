# C-160 punctul 4 — programul 7 al antrenării: eroarea (softmax + entropia încrucișată) cu gradientul ei

Sesiunea 143, 21 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE**: nimic nu a rulat pe RTX 3070 Ti.

## Ce e
Programul de placă `xent` (`tools/nak/xent.comp` → Mesa NAK 25.0.7 → `tools/nak/out/xent.sm86.sass`, 3 680 octeți, 19 registre,
32 de octeți partajați, o barieră; SHA-256 `44cc3536…f420e339d`) face pasul dintre „înainte” și „înapoi” (studiul, `docs/C-160_MOTOR_STUDIU.md`,
programul #7): pentru fiecare rând de V logit-uri (un bloc de 256 de fire pe rând)

- eroarea rândului: `ln Σ e^(z−m) + m − z[țintă]` (m = maximul rândului; media pe rânduri o face apelantul — e doar de raportat);
- gradientul: `(softmax(z) − unu-la-țintă) · 1/N` — exact ce intră apoi în `mm_train` pentru capul legat.

Ordinea sumelor e cea a LayerNorm-ului nou (fiecare fir în ordine, fluture în warp, cele 8 sume de warp în ordine). Valorile `e^(z−m)` nu se
păstrează între treceri (V poate fi 16 384 la #513), se recalculează. Ținta în afara rândului (greșeala apelantului) → eroarea rândului NaN,
vizibil, nimic citit în afara rândului.

Ce a făcut NAK din el (listarea `xent.sm86.txt`): `exp` = `fadd.ftz` (z − m) → `fmul.ftz` cu log₂e → **`MUFU.EX2`**; `1/s` = **`MUFU.RCP`**
fără rafinare; `ln s + m` = **`MUFU.LG2`** urmat de un singur `ffma.ftz` cu ln 2; maximul = `fmnmx.ftz`. Unitățile MUFU sunt aproximative pe
placă, deci **nu există comparație bit cu bit**: modelul pe procesor (`gpu_compute::layer::xent_gpu256`, aceeași ordine a sumelor, `exp` / `ln` /
împărțirea corect rotunjite) și placa se judecă față de oracolul de antrenare în f64 (`gpu_compute::train::softmax_xent`, scos din
`loss_and_grad` fără nicio schimbare de calcul: pasul de aur mic dă aceleași numere, T151 trece neschimbat).

## Marginea, scrisă înainte de măsurare (T153, `crates/nvidia-ga104/tests/sim_xent.rs`)
Pentru un rând: δeₖ ≤ 2u·|zₖ−m| + 8u (scăderea, ·log₂e, MUFU.EX2 socotit generos la 2⁻²¹); δs ≤ (⌈V/256⌉ + 12)·u + δe_max; 1/s ≤ 5u;
|Δdₖ| ≤ scală·(pₖ·δpₖ + 3u·|pₖ − yₖ|); eroarea rândului ≤ δs + 2⁻²¹ + 3u·(|ln s| + |m| + |z_t|); u = 2⁻²⁴.
**Corectură cinstită:** la prima rulare marginea uitase FTZ (placa face 0 din orice rezultat sub 2⁻¹²⁶, oracolul f64 păstrează numerele
subnormale): am adăugat termenul absolut (scală + 1)·2⁻¹²⁶. Nimic altceva n-a fost lărgit.

## Ce a ieșit (SIMULARE, modelul față de oracol)
| Rânduri | max \|Δd\| (cât din margine) | max \|Δ eroare\| (cât din margine) |
|---|---|---|
| V = 255 / 256 / 257 (răspândire 4–8) | 1,5e-8 (41 %) / 9,3e-10 (7 %) / 9,3e-10 (3 %) | 3,2e-7 / 1,7e-7 / 8,4e-7 (≤ 12 %) |
| V = 1 000 (răspândire 20) | 1,5e-8 (23 %) | 9,0e-7 (6 %) |
| V = 16 384 (vocabularul lui #513) | 1,2e-10 (11 %) | 7,7e-7 (6 %) |
| V = 300, răspândire 200 (aproape totul sub 2⁻¹²⁶) | 7,5e-9 (50 %) | 1,1e-5 (11 %) |
| **logit-urile reale ale modelului mic**, lotul pasului de aur (64 × 256) | 9,3e-10 (75 % — doar un element subnormal făcut 0 de FTZ) | 2,2e-6 (13 %) |

Media erorilor celor 64 de rânduri: **44,193805307** față de `loss_and_grad` 44,193805695 (pierderea pasului 1 din PyTorch: 44,19380188),
|Δ| 3,9e-7. Gradientul fiecărui rând însumează ~0 (sub 1e-5). Refuzuri (bloc greșit, fără memorie partajată / barieră: nimic scris), ținta
greșită (NaN, fără unu-la-țintă) și greșeli plantate (modelul altui program; gradientul fără unu-la-țintă; logit-uri ~100) — prinse.

**Ce NU e dovedit:** cât greșesc de fapt `MUFU.EX2` / `LG2` / `RCP` pe RTX 3070 Ti — marginea le socotește generos; se vede doar pe placă.

## Ce e nou pentru placă (când va fi predat, C-160 punctul 6)
Programul `xent` (binar nou; același fel de QMD ca `layernorm256`: 256 de fire, 32 de octeți partajați, o barieră) și **bufferul constant
cu 6 argumente** (`nvidia_ga104::xent::Args::cbuf`, 0x50 octeți). Nu e predat: punctul 6 cere întâi un singur program „înapoi” pe placă,
ordinea o hotărăște centrul de comandă.

## Ce urmează (punctul 4, mai departe)
Următorul program din studiu: „înapoi” prin LayerNorm (sau gradientul deplasării = suma pe coloane, programul 2), față de `train`.
