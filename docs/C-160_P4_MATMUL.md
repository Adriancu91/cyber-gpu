# C-160 punctul 4 — programul 1 al antrenării: înmulțirea matrice × matrice F32 (și cu transpusa)

Sesiunea 142, 21 nov 2026, în cloud, **fără placă**. Totul de aici e **SIMULARE**: nimic nu a rulat pe RTX 3070 Ti.

## Ce e
Un singur program de placă, `mm_train` (`tools/nak/mm_train.comp` → Mesa NAK 25.0.7 → `tools/nak/out/mm_train.sm86.sass`,
4 128 octeți, 19 registre, 2 KiB partajați, o barieră; SHA-256 `69d54ef2…3fa350`), face **toate cele trei înmulțiri ale unui strat
`Linear`** de care are nevoie pasul de antrenare (studiul, `docs/C-160_MOTOR_STUDIU.md`, tabelul „Programele de placă”, #1):

| Produsul | A | B | opțiuni |
|---|---|---|---|
| înainte: Y = X·Wᵀ + b | X [rânduri, in] | W stocată [out, in] | `TB` \| `BIAS` |
| înapoi: dX = dY·W | dY [rânduri, out] | W [out, in] | — |
| înapoi: dW = dYᵀ·X | dY stocată [rânduri, out] | X [rânduri, in] | `TA` |

Cu el se fac qkv, out_proj, fc, proj și capul legat (logit-urile), înainte și înapoi. Gradientul deplasării (suma pe coloane) e programul 2,
separat. Aranjamentul e cel al treptei 3 (plăci 16×16 în memoria partajată), ales ca fiind cel mai simplu dintre cele verificate:
**întâi corectitudinea**; viteza (plăci mai mari, circuitele speciale) e C-160 punctul 7.

## Cum s-a verificat (T152, `crates/nvidia-ga104/tests/sim_mm_train.rs`, 7 teste)
1. **Bit cu bit** față de ce calculează codul compilat (`reference::matmul_t_fma_ftz`: lanțul de `ffma.ftz` de la 0, apoi `fadd.ftz`
   pentru deplasare — văzute în listarea NAK): 4 forme (cu plăci parțiale) × 6 combinații de opțiuni, prin procesorul de comenzi simulat,
   cu un model al programului care urmează plăcile lui. Și cazul −0,0 (ultima placă nu adună zerouri) pentru toate formele.
2. **Față de oracolul de antrenare** (`gpu_compute::train::linear` / `linear_back`, cel care a dat aceleași numere ca PyTorch la pasul de aur
   mic, T151) pe greutățile reale `in_proj` ale stratului 0 din modelul mic (W [192, 64], b [192]), 64 de rânduri:

| Produsul | max \|placă simulată − oracol\| | cât din marginea scrisă |
|---|---|---|
| Y = X·Wᵀ + b | 9,5e-7 | 4,6 % |
| dX = dY·W | 1,9e-8 | 1,2 % |
| dW = dYᵀ·X | 1,2e-7 | 4,4 % |

   Marginea, scrisă înainte de măsurare: \|d\| ≤ (K + 2)·2⁻²⁴·Σ\|aᵢbᵢ\| (+ \|b\|) — cât poate greși un lanț de K adunări în f32 (Higham).
   Pentru gradient, toleranța pasului de aur e 1e-5 absolut: suntem de peste 10 ori sub ea.
3. Refuzuri (opțiuni necunoscute, QMD fără memorie partajată sau fără barieră: nicio scriere) și **greșeli plantate prinse** (un model care
   ignoră opțiunile; dW calculat fără `TA`).

## Ce e nou pentru placă (când va fi predat, C-160 punctul 6)
Programul `mm_train` (alt binar decât `matmul_smem`, același fel de QMD: 2 KiB partajați + o barieră, deja trecut pe placă la probele
matmul) și **bufferul constant cu 8 argumente** (`nvidia_ga104::mm_train::Args::cbuf`, 0x58 octeți). Nu e predat încă: punctul 6 cere
întâi un program „înapoi” mic pe placă; ordinea predărilor o hotărăște centrul de comandă.

## Ce urmează (punctul 4, mai departe)
Programul 7 din studiu: eroarea (softmax + entropia încrucișată) cu gradientul ei, pe rânduri de V, față de `train::loss_and_grad`.
