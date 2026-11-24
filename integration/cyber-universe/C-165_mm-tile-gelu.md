# C-165 — `mm_train_tile_gelu`: lipirea GELU' merge acum și cu `--mm-placi` (nota pentru sesiunea OS)

**Cyber-GPU, sesiunea 214, 24 nov 2026, în cloud, FĂRĂ placă.** Verificat **doar în SIMULARE** (T220). Nota completă:
`docs/C-165_MM_TILE_GELU.md`. **Nu e urgent** — nu schimbă nimic din cursa de la noapte.

## Ce rezolvă
Până acum, un pas #513 cu `--mm-placi` **și** cu lipirile (`with_fused_back().with_fused_gelu()`) ar fi lăsat cele 12 înmulțiri
„proj-dX+gelu” pe `mm_train_gelu` (plăcile vechi de 16 × 16, ~40 ms în plus pe pas, DEDUS). Programul nou le duce pe plăcile de
64 × 64, cu **aceiași biți** ca `mm_train_gelu`.

## Cum se leagă (doar adăugări)
```rust
let plan = Plan::new_tiles3(ar, b, t, &inputs, base)?
    .with_fused_back()
    .with_fused_gelu()
    .with_mm_tile_gelu();          // în loc de .with_mm_tile(); se poate pune și după .with_mm_choice(alegerea)
// programe de încărcat: ca azi + plan.mm_tile_gelu_programs()  (= [Prog::MmTrainTileGelu] când e lipit GELU')
```
- `Prog::MmTrainTileGelu` are indicele **29** (`PROGRAMS_ALL_TILE_GELU`, 30 de programe); indicii 0–28 nu se schimbă;
  `check_programs()` verifică acum 30 de SHA-256. Ramurile `_ =>` din `antrenare_sim.rs` / `gpu_i10.rs` acoperă valoarea nouă.
- Model pe procesor pentru simularea din OS: `gpu_sim::cmdproc::mm_train_tile_gelu_on_cpu` cu `MM_TRAIN_TILE_GELU_NEEDS`.
- Fără `with_fused_gelu`, `with_mm_tile_gelu()` = `with_mm_tile()` exact (testat).

## Nou pentru placă
Doar programul `mm_train_tile_gelu` — SHA-256 `0c4f96bad3e3dbce87d5c309ba7e2c3d426003ff2097e86dc782ce472ccb11af`, 34 de registre,
blocuri 256 × 1 × 1, 8 KiB partajați, o barieră, același bufer constant ca `mm_train_gelu`. Aceeași cale de lansare (QMD +
pushbuffer); niciun registru, nicio comandă nouă. **Încă neprobat pe placă** (regula „primul de felul lui”: o rulare singur).

## Propunere (decide sesiunea OS, după cursă)
Rândul `test-placa --antrenare-mini --vram --antrenare-513 --placi3 --mm-placi --lipit` (sau cum îl numiți) → planul de mai sus;
așteptat: pierderile = oracolul procesorului în pașii 1–9 (aceiași biți ca lipit fără plăci), pasul cu ~2–4 ms mai scurt decât
`--mm-placi` nelipit (DEDUS).
