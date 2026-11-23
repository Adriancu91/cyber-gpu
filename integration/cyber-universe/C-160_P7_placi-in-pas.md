# Pentru OS: programele „pe plăci” legate în pasul de antrenare (C-160 punctul 7, sesiunea 177)

**Ce e nou pentru placă: nimic acum.** `Plan::new`, `new_p_recalc`, `new_p_only` trimit exact aceleași lansări; `PROGRAMS` (13) și
`PROGRAMS_P_ONLY` (14) neschimbate, indicii vechi la fel. Commit de legat: `1b9f79d` (ramura `g0-g2-fundatie`).

## Ce e nou în bibliotecă (doar adăugări)
- `train_step::Prog` are 4 variante noi, la sfârșit: `AttnFwdTile2`, `AttnBackTile0_2`, `AttnBackTile2`, `AttnPTile` (codul, SHA-256,
  registrele, blocul și memoria partajată ale modulelor `attn_fwd_tile2`, `attn_back_tile0_2`, `attn_back_tile_2`, `attn_p_tile`).
  `gpu_i10.rs` are brațul `_ => 0`, deci **compilează fără nicio schimbare**.
- `PROGRAMS_ALL` (18, indicii vechi aceiași), `PROGRAMS_TILES` (14), `PROGRAMS_TILES_P_ONLY` (15); `check_programs()` verifică 18.
- `Plan::new_tiles(…)` (P păstrat) și `Plan::new_p_only_tiles(…)` (un singur P, refăcut cu `attn_p_tile`): aceleași lansări, etichete
  și cbuf-uri ca `new` / `new_p_only`, doar atenția pe programele pe plăci (grila [⌈T/64⌉, H, B], QMD cu memoria partajată).
  Cer d ≤ 64. `Plan::tiles()`, `Plan::programs()` = lista de încărcat.

## Ce ar trebui în OS, doar când vreți planul nou (după ce os-364 arată `attn_*_2` pe placă)
1. Încărcarea programelor după `plan.programs()` (deja o faceți din P6-152).
2. În simulare (`gpu_i10.rs`), cele 3 programe `attn_*_2` sunt deja înregistrate; lipsește doar
   `cp.register_program(nvidia_ga104::attn_p_tile::ATTN_P_TILE, c::attn_p_tile_on_cpu)` pentru `new_p_only_tiles`.
3. Pe placă: o comandă nouă (pasul de aur pe plăci), nu acum.

## Dovada (SIMULARE)
T187: pasul de aur mic `sgd` / `adamw_513`, 10 pași: verdictul față de PyTorch trece; „înapoi” pe plăci = „înapoi” vechi bit cu bit;
P refăcut cu `attn_p_tile` = P păstrat bit cu bit; 4 straturi la fel; greșeli plantate prinse. Nu e bit cu bit cu `Plan::new`
(suma pe 16 parțiale din `attn_fwd_tile*`, cunoscută din T173). Nota completă: `docs/C-160_P7_PLACI_IN_PAS.md`.
