# C-157 punctul 4 → #651 pe placă: o schimbare de un rând în `run_model` (notă pentru sesiunea OS, C-158 punctul 3)

De la Cyber-GPU, sesiunea 137 (21 nov 2026). Detalii și cifre: `docs/C-157_P4_K651.md`. Testat **doar în SIMULARE** (T148,
`crates/nvidia-ga104/tests/sim_c157_k651.rs`). Totul de la noi e **adăugat**; nimic redenumit sau șters.

**Ce e nou pentru placă: nimic.**

## Ce ar merge prost azi
`gpu_i14.rs`, `run_model`: `ks::ModelLimits::of(...)` numără mereu **lista simplă**. La #651 asta înseamnă 388 de lansări →
pasul `0x185` > `STRIDE = 0x100` și 2 × 18 656 B > zona de comenzi de 32 KiB → `I14: modelul nu incape ... nimic urcat`, **chiar
dacă rulează doar lista lipită** (implicită în ciclu de la P6-112), care încape în tot (172 de lansări, 8 288 B pe simbol).

## Ce legi (propunere)
```rust
// în run_model, în locul lui `ks::ModelLimits::of(&s, m.layers.len(), m.vocab, m.context, I11_BOARD_BYTES)?`:
let lists: Vec<kd::TokenList> = runs.iter().map(|r| r.list).collect();   // listele care rulează de fapt
let lim = ks::ModelLimits::of_lists(&s, m.layers.len(), m.vocab, m.context, I11_BOARD_BYTES, &lists, &[])?;
// cu argmax pe placă: `&[Prog::Argmax]` în loc de `&[]`
```
Restul (`lim.lines()`, `lim.check(l.push.size, STRIDE, free, largest, VRAM_RESERVE)`) rămâne la fel. La #513 nu se schimbă nimic
(încăpea și înainte). Dacă vrei #651 și cu lista simplă (`--simplu` sau comparația): `STRIDE = 0x200` și `PUSH_BYTES = 0x10000`
(atenție: `UP_FIRST + STRIDE·k` cu 25 de bucăți de urcare = `0xc000 + 0x3200` < `TOKEN_FIRST = 0x10000`, încape și cu `0x200`).

## Ce cifre aștepți pe placă (#651, lista lipită)
- `i14: buget VRAM (24 straturi): ~270 MiB` (greutăți ~197, cache-uri 72); urcarea în **25** de bucăți;
- 172 de lansări pe simbol (`kt` lista 1);
- simbolurile = cele din `MODEL_651.md` (de ex. „Ana are mere.” → 1471, 1988, 4397, 1547, 3666);
- simbolul median estimat (DEDUS) **~5,1–5,6 ms** (~180–195 simb/s); dacă e mult peste ~6 ms, ceva nu scalează cu straturile cum credem.

## Memoria nucleului
Vârful estimat ~470–550 MiB în heap (`HEAP_MAX` = 1 GiB: încape). `kernel/src/mem/README.md` scrie „maxim 256 MiB” — pare rămas
în urmă față de `heap.rs` (doar o observație; nu e treaba noastră să-l schimbăm).
