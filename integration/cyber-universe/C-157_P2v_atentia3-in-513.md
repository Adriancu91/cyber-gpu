# Pentru OS: pasul #513 pe atenția pe plăci (`Plan::new_tiles3`) — după os-403

Cyber-GPU, sesiunea 206, 24 nov 2026. Planul și predicțiile: `docs/C-157_P2_OS403_ATENTIA_513.md` (urcate înainte de test, `efbc456`).
Testul: T212 (`crates/nvidia-ga104/tests/sim_c160_p5_golden.rs`). Commit de legat: cel din `stare/cyber-gpu.md`, rândul PREDARE.

## Ce a arătat os-403
Atenția = **33 % din pas** (371 ms: `attn_fwd` 212 ms = 17,7 ms pe strat, `attn_back` 159 ms). Pasul #513 din
`train_513_resident` (`kernel/src/drivers/gpu_i14.rs`) e construit cu `Plan::new`, adică pe programele de atenție **de bază**.
Programele pe plăci gen 3 au trecut pe placă la os-375 (identice bit cu bit cu `_2`, care erau identice cu gen 1 la os-364);
pe bucata `atentie-513-b4` (B 4, T 512, un strat) lanțul lor ia 3,43 ms. N-au fost legate niciodată în pasul #513.

## Ce propunem (o singură schimbare, un singur lucru nou pentru placă)
În `train_513_resident`, la ambele construcții:
```
let probe0 = Plan::new_tiles3(a, bsz, tl, x0, nvidia_ga104::mmu::VA_BASE)?.resident()?;
...
let plan0 = Plan::new_tiles3(a, bsz, tl, x0, varena.va)?.resident()?;
```
(aceeași semnătură ca `Plan::new`; există în biblioteca pe care o aveți deja legată, `e87fa5d`). Sub o opțiune nouă, ca să
rămână și lista veche (ex. `test-placa --antrenare-mini --vram --antrenare-513 --atentie3`; numele îl alegeți voi).
Restul codului rămâne: `plan0.programs()` dă acum 14 programe (`PROGRAMS_TILES3`) în loc de 13 și le urcă bucla existentă;
QMD-ul ia singur memoria partajată a programului (`ts::pack` → `Launch::qmd` → `Prog::shared`; `attn_back_tile_3`: 33 024 octeți).

**Ce e nou pentru placă:** doar că pasul #513 lansează `attn_fwd_tile3` / `attn_back_tile0_3` / `attn_back_tile_3` (programe
deja trecute pe placă) în locul lui `attn_fwd` / `attn_back`. Niciun program nou, niciun registru nou, aceeași arenă.

## Ce am verificat în cloud
- **T212, pe procesor (MĂSURAT):** la forma reală (#513, 2 × 512, 12 straturi, `.resident()`): arena **2 560 MiB la ambele**,
  **385** lansări la ambele, aceleași etichete / straturi / cbuf-uri; **349** lansări identice (program, cbuf, grilă); cele
  **36** de atenție (12 înainte, 12 dS, 12 dqkv) au alt program și grila [8, 12, 2]. Deci trimiterile rămân 255 + 130.
- **T212, SIMULARE:** un strat la lățimile lui #513 (greutăți inventate), 2 × 128 (cu placa diagonalei): un pas AdamW pe
  `new_tiles3` = oracolul pe procesor în toleranțele hub-ului (eroarea identică, norma 2,7·10⁻⁹ rel., gradientul 2,8·10⁻⁷,
  greutățile 1,6·10⁻⁵ ≤ 1·10⁻⁴); eroarea și norma **identice** cu `Plan::new` pe aceleași date (gradienții nu bit cu bit).

## Ce așteptăm pe placă (DEDUS)
- Atenția pasului: **~20–40 ms în loc de 371** (la B = 2, jumătate din blocurile bucății b4).
- Pasul pe placă: **~1 123 → ~770–800 ms ≈ 1 280–1 330 simb/s** (×1,4). Profilul (`--antrenare-513` scrie deja
  `StepProfile`) trebuie să arate cele 3 programe `_3` cu ~2 ms pe strat împreună și `mm_train` neschimbat (~718 ms).
- Pierderile pe loturile sintetice: tot oracolul `ORACLE_513_SYNTH` (pașii 1–9 la ≤ 1·10⁻³).
- Dacă atenția iese peste 60 ms pe pas: e de văzut de ce B = 2 merge mai prost decât jumătate din b4 (puține blocuri pe SM).

## După aceea
Înmulțirile (`mm_train`, ~90 % din pasul nou, ~0,83 TFLOP/s): `--antrenare-513 --mixt` pe `mm_tc` cu scara **2¹⁴** (nota
`C-157_P2u_oracol-513.md`), apoi un `mm_train` F32 pe plăci (program nou, la noi).
