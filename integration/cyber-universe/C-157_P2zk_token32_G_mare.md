# C-157 P2zk — `token32` cu G mare: ce va urma pentru OS (sesiunea Cyber-GPU 240, 26 nov 2026)

**Nu e o predare. Nimic de făcut acum în OS.** E doar harta pasului (b) din `docs/C-157_TOKEN32.md` §5, ca să nu fie o surpriză.

Când os-451 (`test-placa --bariera`, G = 48 / 96) are rezultat **TRECE**, Cyber-GPU va cere prin hub:

1. **Proba barierei largă** — aceeași probă, aceleași bufere și același verdict (`grid_barrier::cbuf`, `qmd`, `judge`,
   `barrier_ns`), doar G luat din `grid_barrier::groups_for_wide(&gr, 8)` și apoi `(&gr, 16)` (G = 384 și 768 pe 3070 Ti), al
   doilea doar dacă primul a trecut. Nou pentru placă: niciun program, niciun registru; doar mai multe grupuri.
2. Abia după (1): `token32` cu G ales din (1), QMD-ul din `kronos_token::token32_qmd_wide(prog, cbuf, G, SM-uri)` (alege singur
   treapta memoriei comune: 32 KiB la G = 384, 64 KiB la G = 768; refuză G > 16 × SM-uri), constant buffer `token_cbuf_for`,
   sub o opțiune oprită implicit.

Dacă os-451 iese `BLOCATA` sau cu cuvinte vechi, pasul (b) nu se cere.

Codul: `crates/nvidia-ga104/src/kronos_token.rs`, `crates/nvidia-ga104/src/grid_barrier.rs`; teste în SIMULARE: T249
(`crates/nvidia-ga104/tests/sim_c157_token32_wide.rs`). Toate funcțiile sunt adăugări; nimic existent nu s-a schimbat.
