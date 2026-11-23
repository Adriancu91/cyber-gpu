# Pentru sesiunea OS — atenția `_3` legată în pasul de antrenare (C-157 p.2, după os-375)

Cyber-GPU, sesiunea 187, 23 nov 2026. **SIMULARE** — nimic din ce e aici n-a rulat pe placă ca pas întreg.
Detalii, predicții și rezultat: `docs/C-157_P2_ATTN3_IN_PAS.md`.

## Ce e nou în `nvidia_ga104::train_step` (doar adăugări)
- `Prog::AttnFwdTile3`, `Prog::AttnBackTile0_3`, `Prog::AttnBackTile3` (la sfârșitul enum-ului).
- `PROGRAMS_ALL_3` (21 = `PROGRAMS_ALL` + cele 3), `PROGRAMS_TILES3` (14), `PROGRAMS_TILES3_P_ONLY` (15).
- `Plan::new_tiles3`, `Plan::new_p_only_tiles3`, `Plan::new_tiles3_mv_ram`, `Plan::new_p_only_tiles3_mv_ram`, `Plan::tiles3()`.
  Aceleași arene, aceleași lansări, cbuf-uri și grile ca planurile `_2`; doar programul celor 3 lansări de atenție pe strat
  e `_3`. Relansarea P rămâne `attn_p_tile`.

## Compatibilitate
- **`PROGRAMS_ALL` rămâne cu 18 programe** (aceiași indici), deci `shell/antrenare_sim.rs` (`SimBoard::new`, care refuză un
  program fără model) merge neschimbat la urcarea versiunii.
- Un `match` exhaustiv pe `train_step::Prog` fără `_ =>` nu mai compilează; în OS ambele locuri (`antrenare_sim.rs`,
  `gpu_i10.rs`) au deja brațul `_`.
- `Prog::index()` se ia acum din `PROGRAMS_ALL_3` (aceiași indici pentru cele 18; cele noi 18, 19, 20). Cine pune programele
  la `bază + 0x1_0000 · index` are nevoie de 21 de locuri dacă folosește un plan `_3`.

## Ca să legi un plan `_3` în OS (când vrei; nu e o cerere de test)
1. În `SimBoard::new`: treci prin `PROGRAMS_ALL_3` și înregistrează cele 3 `_3` cu aceleași modele ca primele pe plăci
   (`attn_fwd_tile_on_cpu`, `attn_back_tile0_on_cpu`, `attn_back_tile_on_cpu`) — exact cum face deja `gpu_i10.rs` (P6-160).
2. Planul: `Plan::new_tiles3` (sau `new_p_only_tiles3` / `_mv_ram`) în locul lui `new_tiles`; programele de încărcat:
   `plan.programs()`.

## Ce e nou pentru placă
Doar cele 3 programe `_3` **în pas** (au rulat deja pe placă în proba `--atentie3`, os-375: 6 / 6 identice bit cu bit cu
vechii). Pe bucata #513: atenția ~191 → ~123 ms pe pas (DEDUS). Pasul de antrenare pe placa reală rămâne o comandă separată.

## Testat
T195 (SIMULARE, commit-ul din `stare/cyber-gpu.md`): lista, pasul de aur mic 10 pași ambele rețete (`_3` = `_2` bit cu bit,
verdictul față de PyTorch trece), 4 straturi, amestecul `_2` / `_3`, greșeli plantate prinse, variantele `_mv_ram`.
