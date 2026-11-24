# C-157 p.2y — `colsum` împărțit pe rânduri în pasul de antrenare (`Plan::with_colsum_split`)

Sesiunea 210 (24 nov 2026), în cloud, FĂRĂ placă. Planul și predicțiile sunt scrise **înainte** de cod.

## De ce

os-407 (pasul #513 pe atenția gen 3, `docs/C-157_P2_COMPARATII.md`): pasul 778 ms, `mm_train` 93 %, **`colsum` 24,3 ms = 3,1 %**
(73 de lansări, ~333 µs fiecare, 1 024 de rânduri adunate de un singur fir pe coloană, ~0,33 µs pe rând). Când `mm_train_tile`
(P6-188, `--mm-placi`) aduce pasul la ~175–295 ms (DEDUS), `colsum` devine **8–14 %** din pas.

Programul care împarte suma pe rânduri există din sesiunea 159 (`colsum_part`, T170, SIMULARE; NAK 25.0.7, SHA-256 `740f5d43…`):
trecerea 1 adună bucăți de `chunk` rânduri în sume parțiale `[G, c]`, trecerea 2 e `colsum` însuși (modul deplasare) peste cele G rânduri.
Nu a fost legat în pas și nu a rulat pe placă.

## Ce fac

1. `Prog::ColsumPart` (variantă nouă, la sfârșit), `PROGRAMS_ALL_SPLIT` (25 = `PROGRAMS_ALL_TILE` + `ColsumPart`, indicii vechi păstrați),
   `PROGRAMS_COLSUM_SPLIT_EXTRA`.
2. `Plan::with_colsum_split()` / `colsum_split()` / `colsum_split_programs()`: aceeași arenă **plus o zonă de lucru la capăt**
   (`Plan::colsum_scratch()`: p0 și p1, câte G × 4E f32; #513: G 32 × 3 072 → 2 × 384 KiB), `Plan::bytes()` o cuprinde.
   Fiecare lansare `colsum` din listă devine 2 (deplasare: `colsum_part` → `colsum` p0 → s0) sau 3 (LayerNorm: + `colsum` p1 → s1);
   `chunk` = `colsum_part::default_chunk(rânduri)` (1 024 → 32). Zona de lucru e refolosită de la o sumă la alta (lansările merg
   una după alta, ca toate celelalte din listă). Aplicată în `step`, `step_until_opt`, `step_mixed`; păstrată de `resident` / `with_recompute`.
   Oprită implicit; se combină cu `with_mm_tile` și restul.
3. `launches_per_step` numără lansările în plus (8 pe strat + 2: 4 deplasări × 1 + 2 LayerNorm × 2, plus LayerNorm-ul final).

## Predicții (scrise înainte)

| # | Ce | Predicție | Cum se verifică |
|---|---|---|---|
| 1 | Lansări #513 (2 × 512, `_3`, rezident) | 385 → **483** (73 de `colsum` → 73 `colsum_part` + 98 `colsum`), restul neschimbat | T216, numărat |
| 2 | Pasul mic (MIC, 2 × 32, 10 pași, `sgd` + `adamw_513`) cu opțiunea | **nu** bit cu bit (altă ordine a sumelor), dar = oracolul f64 în aceeași toleranță ca fără opțiune; pierderile față de fără opțiune ≤ 1e-5 | T216, SIMULARE |
| 3 | dγ, dβ, db ale pasului 1 față de oracolul f64 | abaterea **nu mai mare** decât fără opțiune (T170: la 6 144 × 768 de 10× mai mică) | T216 |
| 4 | `colsum` pe placă la #513 | 24,3 ms → **~2,5–5 ms** pe pas (171 de lansări, ~10–35 µs fiecare; podeaua ~5–10 µs pe lansare) | DEDUS; pe placă prin OS |
| 5 | Pasul #513 | −~19–22 ms (azi 778 → ~757; după `--mm-placi` −7…12 %) | DEDUS |

Ce nu știm: costul fix al unei lansări mici pe placă în lista lungă (os-407: `vadd` 16,7 µs, `ln_fwd` 23 µs, `emb_fwd` 27 µs);
dacă `colsum_part` merge pe placă cum merge `colsum` (aceleași operații; nou pentru placă: doar grila pe două dimensiuni și `chunk`).

## Siguranță

Nou pentru placă: programul `colsum_part` (121 de instrucțiuni, fără barieră, memorie partajată sau `atom`) și zona de lucru
din arenă. Niciun registru nou. Pasul rămâne judecat de oracolul procesorului (pașii 1–9 la 1e-3); la o diferență OS-ul se întoarce la
lista fără opțiune.

## Rezultat (sesiunea 210, SIMULARE, T216)

| # | Predicție | Ieșit | Rezultat |
|---|---|---|---|
| 1 | 385 → 483 lansări | **483** (48 de sume de deplasare + 25 LayerNorm; 312 lansări identice, în aceeași ordine); zona de lucru **768 KiB** după arenă; `step_until_opt` = pasul fără ultima lansare; pasul mixt are tot `launches_per_step_mixed` | PASS (pe procesor) |
| 2 | nu bit cu bit, = oracolul, pierderile ≤ 1e-5 față de fără opțiune | 6 / 6 (sgd, adamw_513 × `new`, `_3` + plăci, `_3` + lipiri + plăci): verdict față de PyTorch **TRECE**; pierderile **identice** pe 10 pași (abaterea 0); biții gradientului diferă | PASS (SIMULARE) |
| 3 | gradientul nu mai departe de oracolul f64 | **1,19e-7** cu împărțire față de **2,4–3,6e-7** fără (de 2–3 ori mai aproape, ca la T170); un strat #513 (2 × 128): aceleași cifre ca fără | PASS (SIMULARE) |
| — | greșeală plantată: a doua sumă a LayerNorm (dβ) lipsă | gradientul 9,5e-2 față de 1,2e-7 → prinsă | PASS |
| 4–5 | `colsum` 24 → ~2,5–5 ms, pasul −19–22 ms | — | DEDUS / NOT TESTED |
