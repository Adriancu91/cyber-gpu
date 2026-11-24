# C-157 p.2x — `mm_train_tile`: înmulțirea antrenării pe plăci de 64 × 64, aceiași biți (pentru sesiunea OS)

**De la:** Cyber-GPU, sesiunea 208 (24 nov 2026, cloud, fără placă). **Stare:** SIMULARE (T214, 6/6). **Încă nu e de legat
în pas** — urmează la noi `Plan::with_mm_tile` (aceeași listă, `mm_train` înlocuit), apoi o probă mică pentru placă.

## Ce e
- Program nou: `tools/nak/out/mm_train_tile.sm86.sass` (Mesa NAK 25.0.7, SM 8.6), **SHA-256
  `f8acce2165870280d06271c6b0a60ca96ebc418b23fc7abb8ff496e08bbf094d`**, 1 436 instrucțiuni (22 976 octeți), **34 de registre**,
  fără memorie locală; reconstruit de două ori în cloud → aceeași sumă (și `mm_train` reconstruit = `69d54ef2…` neschimbat).
- În Rust: `nvidia_ga104::mm_train_tile` (`MM_TRAIN_TILE`, `REGISTERS`, `BLOCK` = [256, 1, 1], `SHARED` = 8 KiB + o barieră,
  `grid(&mm_train::Args)`), modelul `gpu_sim::cmdproc::mm_train_tile_on_cpu` + `MM_TRAIN_TILE_NEEDS`. Doar adăugări.
- **Aceleași argumente ca `mm_train`** (bufferul constant 0 făcut de `mm_train::Args::cbuf`, 0x58 octeți). Diferă doar
  lansarea: blocuri de 256 de fire, grila ⌈N/64⌉ × ⌈M/64⌉, QMD cu 8 KiB partajați.
- **Aceiași biți ca `mm_train`**: fiecare element = lanțul de `ffma.ftz` p = 0..K−1 de la +0,0, apoi `fadd.ftz` cu bias.
  T214: bit cu bit egal cu `reference::matmul_t_fma_ftz` **și** cu `mm_train` pe toate cele 8 combinații TA / TB / BIAS,
  plăci parțiale (65 × 17 × 129, 130 × 33 × 70, 1 × 40 × 1 …) și cele 3 produse ale unui strat la formele pasului mic și
  o tăietură #513 (96 × 768 × 80); −0,0 păstrat, nicio umplere cu zero adunată.

## Ce e nou pentru placă
Un program nou (cel de mai sus) și un QMD cu 8 KiB de memorie partajată la 256 de fire. Nimic altceva. **Nimic de trimis
acum.**

## Ce așteptăm (DEDUS, NETESTAT)
`mm_train` pe placă ~0,83 TFLOP/s (os-403). `mm_train_tile` face 16 înmulțiri la 8 citiri din memoria partajată (față de 1 la 2):
estimat **×3–6 → ~2,5–5 TFLOP/s**; în pasul #513 (atenția `_3`): `mm_train` ~600 → ~120–250 ms, pasul ~770–800 → ~300–450 ms.

## Ce va cere OS-ul (în sesiunile următoare, nu acum)
1. După `Plan::with_mm_tile` la noi: aceeași comandă de antrenare ca la os-395 / os-403 cu o opțiune nouă — **aceeași amprentă
   a pasului** (biții nu se schimbă), timpul `mm_train` pe profil.
2. Întâi, separat: o probă mică (ca `mm_tc_probe`): seturile mici bit cu bit = `mm_train` pe aceeași placă + `qkv-513` cronometrat.
