# C-160 punctul 7 — `attn_p_tile`: doar P, pe plăci (varianta „doar P” a lui `attn_fwd_tile2`)

Sesiunea 176, 23 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte** de cod.
Corectitudinea se verifică doar în **SIMULARE** și prin compararea codului compilat; costul e **DEDUS** (nemăsurat).

## De ce
Sesiunea 168 a făcut `attn_p` = `attn_fwd` fără P·V (blocuri de 32, un fir pe rând), iar sesiunea 175 l-a legat în pas
(`Plan::new_p_only`). Pentru pasul #513 cu P recalculat (lot 12 × 512 încape sub 7 GiB doar așa, `docs/C-160_P7_P_RECALCULAT.md`)
programele care contează sunt cele **pe plăci** (os-358: corecte pe placă; `attn_fwd_tile2` vine cu os-364). Lipsea varianta
„doar P” pe plăci: refacerea lui P la întoarcere fără faza C (P·V).

## Planul (scris ÎNAINTE de cod)
Program nou `tools/nak/attn_p_tile.comp`, **lângă** cele vechi (nimic schimbat, nimic redenumit):
- fazele A (Q·Kᵀ pe plăci, scrie s = acc·scale, maximul pe rând) și B (x = exp(s − m), sumele parțiale pe `tx`) **copiate
  literal** din `attn_fwd_tile2.comp` — aceleași operații, în aceeași ordine;
- faza C redusă: o barieră (după 1/sum), apoi fiecare fir își normalizează cele 16 valori pe placă (`x · inv`, j ≤ i) și scrie
  +0 deasupra diagonalei, pe toate plăcile `jt < ⌈T/64⌉` — exact ce scrie faza C a lui `attn_fwd_tile2` în P; **fără** placa V,
  fără P·V, fără `att` (adresa de la 0x38 nu e citită);
- aceleași argumente (cbuf-ul lui `attn_fwd`), aceeași grilă [⌈T/64⌉, H, B], blocuri de 256, aceeași memorie partajată
  (37 888 B — plăcile Q / K ale fazei A rămân), o barieră.
- Rust: modulul `nvidia_ga104::attn_p_tile` (`Args = attn_fwd::Args`, `grid` / `shape_ok` ca `attn_fwd_tile`); modelul pe
  procesor `gpu_compute::layer::attn_p_tile_row_gpu` = partea P a lui `attn_fwd_tile_row_gpu` (acesta o va apela, deci e
  aceeași funcție); `gpu_sim::cmdproc::attn_p_tile_on_cpu`.
- **Nu** intră încă în `train_step` (programele pe plăci nu sunt încă în pas; nicio variantă nouă în `Prog`, deci OS-ul nu are
  nimic de schimbat).

## Predicții (scrise acum)
1. **Compilarea** (Mesa NAK 25.0.7): registre ≤ 117 (cât `attn_fwd_tile2`); mai puține instrucțiuni decât `attn_fwd_tile2`;
   37 888 B partajați, 1 barieră, nicio memorie locală, nicio `atom` / `red`.
2. **Codul compilat:** aceleași `mufu` ca `attn_fwd_tile2` (16 × `mufu.ex2`, 1 × `mufu.rcp`); fiecare fel de operație cu virgulă
   mobilă din `attn_p_tile` există și în `attn_fwd_tile2` (aceleași modificatoare `.ftz`); niciun acces la `c[0x0][0x38]` (`att`).
3. **SIMULARE:** pe cele 10 forme din T173 / T181, P scris de `attn_p_tile` = P scris de `attn_fwd_tile` și de `attn_fwd_tile2`
   **bit cu bit** (și = modelul), bufferul `att` (umplut cu santinele) rămâne neatins, nimic scris după P.
4. **Greșeli plantate prinse:** QMD cu memoria partajată a lui `attn_fwd_tile` (37 376 B, mai puțin decât cere programul) refuzat;
   grila cu un bloc lipsă lasă exact rândurile lipsă nescrise; blocuri ≠ 256 refuzate; d > 64 refuzat; scale greșit schimbă P.
5. **Costul (DEDUS, nemăsurat):** pe setul mare (4 × 512, 12 capete × 64) faza C a lui `attn_fwd_tile2` înseamnă urcarea plăcii V +
   64 × 16 `ffma` pe fir pe placă (cam cât faza A) plus citirea / scrierea lui P; `attn_p_tile` păstrează A, B și scrierea lui P.
   Estimare: **0,45–0,80** din timpul lui `attn_fwd_tile2` (intervalul larg: nu știm încă cât din timpul pe placă e memorie și cât
   e calcul — aflăm la os-364). Număr de instrucțiuni pe drumul plăcilor de sub diagonală: ≈ jumătate din `attn_fwd_tile2`.

## Ce NU face
Nu schimbă niciun program existent, nici `train_step` (`Prog`, `PROGRAMS*`, `Plan::*`), nici proba `attn_tile_probe`. Nimic nou
predat pe placă în această sesiune (vine după ce os-364 arată că `attn_fwd_tile2` dă biții lui `attn_fwd_tile` pe placă).

## Rezultatul (după cod)
(se completează după cod)
