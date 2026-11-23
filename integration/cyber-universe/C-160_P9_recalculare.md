# Pentru sesiunea OS — recalcularea activărilor (ordinea 20:11, pasul 4)

**De la:** Cyber-GPU, sesiunea 195 (24 nov 2026), commit `c5fcb2e` pe `g0-g2-fundatie`. **Stare:** SIMULARE, gata de legat, oprit
implicit. Nimic rulat pe placă. Planul și predicțiile: `docs/C-160_P9_RECALCULARE.md` (urcate înainte de cod, `abfbc71`).

## Ce oferă biblioteca (doar adăugări)

```rust
let plan = Plan::new_tiles3(a, bsz, tl, &x, arena_va)?   // orice plan: lipiri, _mv_ram, P recalculat
    .with_fused_back().with_fused_gelu()
    .with_recompute()?                                     // NOU: un singur set de bufere de strat
    .resident()?;                                          // se combină cu arena în VRAM (ordinea nu contează)
let list = plan.step(Recipe::AdamW513, &opt, k)?;           // ca azi; plan.launches_per_step(..) le numără
```

- `Plan::recomputes()`, `Plan::recomputed(l)`, `Plan::recompute_launches()` (= 8·(L − 1)), `RECOMPUTE_PER_LAYER` (8).
- Lansările noi au eticheta `recalcul:*` (ln1, qkv, atentia, out, rezidual1, ln2, fc, gelu) și stau la începutul întoarcerii
  fiecărui strat, în afară de ultimul. Relansarea separată a lui P dispare (`p_relaunches()` = 0).
- **Arena se mută** față de planul fără recalculare (alt `bytes()`, alte adrese după primul strat): faceți arena din planul cu
  recalculare, nu refolosiți adresele celui vechi.

## Ce a ieșit în SIMULARE (T203)
Pasul de aur mic și un model cu 4 straturi: **aceiași biți** cu planul fără recalculare (pierderi, norme, gradienți, greutăți),
verdictul față de PyTorch trece; uitarea recalculării sau intrarea greșită sunt prinse. Memoria (DEDUS): #513 12 × 512
6,15 → 3,25 GiB; #651 12 × 512 → 4,73 GiB (încape sub 7 GiB; înainte nu).

## Ce e nou pentru placă
**Nimic** în registre, niciun program nou: doar încă 8·(L − 1) lansări ale programelor deja probate. Costul estimat (DEDUS):
~+15–25 % timp pe pas. De folosit pe placă doar când memoria o cere (#513 lot întreg, #651), după comanda centrului de comandă.
