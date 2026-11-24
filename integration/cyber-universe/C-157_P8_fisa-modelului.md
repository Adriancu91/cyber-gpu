# C-157 p.8 — fișa modelului alege pachetul de calcul (nota pentru OS)

**Ce e:** decizia lui Adrian din 20 nov 19:41 / 19:45 („o bază + pachete de calcul alese după fișa din KGR1; fără pachet
→ refuz clar”), primul pas de cod. Proiectarea: `cyber-gpu/docs/C-157_P8_FISA_MODELULUI.md`. Testat doar pe procesor
(T208); **nimic nou trimis la placă**.

**Ce oferă (doar adăugări, interfața veche neschimbată):**
- `gpu_compute::card::ModelCard::from_kgr1(&octeți)` — citește doar antetul și cheile KGR1 (primii octeți ai fișierului,
  fără tensori): arhitectura, `fel`, straturi, capete, n_emb, context, vocabular, `ln_eps`, `cap_legat`. Orice fișier
  stricat sau tăiat → `CardError` cu `line()` în română, fără panică.
- `nvidia_ga104::packs::select(&fișă)` →
  - `Ok(Choice)`: pachetul `kronos-gpt-q8-generare` v1 (lista `lipita-ln-32`) + dovada: **dovedit pe placă** pentru
    formele #513 (os-346) și #651 (os-355), altfel **PROBA NECESARĂ**; `Choice::limits(8 MiB)` = `ModelLimits` din
    numerele fișei (la #651 exact `ModelLimits::k651_lists(&[FusedLn32])`); `Choice::line()` pentru consolă;
  - `Err(Refusal)`: toate motivele (`Refusal::lines()`, primul rând „REFUZ: …; nimic nu ruleaza pe placa”).

**Cum s-ar lega în OS (propunere, decide sesiunea OS):** în `kronos --gpu` / `test-placa`, înainte de urcarea greutăților:
fișa din primii octeți de pe stick → `select` → la refuz se tipăresc rândurile și calculul pe placă nu pornește; la
`PROBA NECESARĂ` rulează întâi verificarea pe procesor care există azi; la „dovedit” totul merge ca acum. Cititorul KGR1
din `gpu_i14.rs` rămâne pentru tensori.

**Ce e nou pentru placă:** nimic.
