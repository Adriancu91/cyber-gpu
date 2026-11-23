# Pentru OS: `attn_p` legat în pasul de antrenare (C-160 punctul 7, sesiunea 175)

**Ce e nou pentru placă: nimic acum.** Nimic din ce rulează azi nu se schimbă: `Plan::new` și `Plan::new_p_recalc` trimit exact
aceleași lansări, iar `train_step::PROGRAMS` rămâne cele 13 programe (`--pas-aur` încarcă ce încărca).

## Ce e nou în bibliotecă (doar adăugări, commit-ul de legat: vezi `stare/cyber-gpu.md`)
- `train_step::Prog::AttnP` (programul `attn_p`, SHA-256 `786ae995…148d386e`, 22 registre, blocuri de 32, fără memorie partajată).
- `train_step::PROGRAMS_P_ONLY` = cele 13 + `AttnP` la sfârșit (indicii celor vechi neschimbați).
- `Plan::new_p_only(arch, batch, len, inputs, base)` = arena lui `new_p_recalc` (un singur P), dar P-ul pierdut se reface cu
  `attn_p` (doar P, fără P·V; ≈ 0,51 din `attn_fwd`, DEDUS). `Plan::programs()` = lista de încărcat pentru planul dat;
  `Plan::p_program()` = programul relansării.
- `check_programs()` verifică acum și SHA-256 al lui `attn_p` (14).

## Ce ar trebui în OS, doar când vreți planul nou (nu e obligatoriu)
1. `kernel/src/drivers/gpu_i10.rs`, în `match prog`: `T::AttnP => cp.register_program(code, c::attn_p_on_cpu),`
   (brațul `_ => 0` poate rămâne sau poate fi scos).
2. `kernel/src/drivers/gpu_i14.rs`, la încărcarea programelor pasului: `for prog in plan.programs()` în loc de `ts::PROGRAMS`
   (pentru `Plan::new` dă aceleași 13).
3. Dacă în simulare puneți programele pe sloturi după `Prog::index()`: `AttnP` are indicele 13 — atenție să nu se suprapună cu
   alt program pus pe slotul 13 (la noi, `mm_tc` în `sim_c160_p7_mixed.rs` stă acolo, dar acel test nu folosește `new_p_only`).

## Dovada (SIMULARE)
T185: pasul de aur mic `sgd` / `adamw_513`, 10 pași, **bit cu bit** = P păstrat; 4 straturi bit cu bit cu 3 `attn_p`; relansarea
scoasă sau pe `qkv`-ul altui strat = prinse. Nota completă: `docs/C-160_P7_ATTN_P_IN_PAS.md`.
