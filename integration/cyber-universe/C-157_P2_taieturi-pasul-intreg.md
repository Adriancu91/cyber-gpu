# Pentru OS: al doilea rând de tăieturi ale pasului de antrenare (C-157 p.2, după os-386)

**Ce e nou pentru placă: nimic.** Aceleași lansări, aceleași cuvinte în pushbuffer, aceleași înregistrări scrise de placă. Se
schimbă doar ce scrie / citește driverul prin fereastra VRAM. Proba: cyber-gpu T206 (SIMULARE), nota `docs/C-157_P2_TAIETURI_PAS.md`.

Se pune **după** remediul jurnalului (nota `C-157_P2_os-386_pasul-intreg.md`: rutina + `flush_live()` pe pas) — altfel cei
~350 ms ai stick-ului acoperă tot.

## Ce (în `train_mini_resident`, `gpu_i14.rs`; uneltele există deja și merg pe placă în generare)
1. **Umbra pushbuffer-ului:** înainte de buclă `let mut push_shadow = kf::WriteShadow::new(l.push.pa, l.push.size)?;` și în
   loc de `run_records(...)` → `gpu_i11::run_records_d2(..., Some(&mut push_shadow), d2)`. După `copy(...)` (arena sus / jos
   folosește același pushbuffer) sau după orice eroare: `push_shadow.forget()` (`run_records_d2` o face singur la eroare).
2. **Înregistrările pe zero o singură dată:** `put_bytes(..., ctx.rec.pa, &vec![0u8; 16 * (n_list + 1)])` doar la `k == 0`
   (valorile cresc cu `STRIDE` la fiecare trimitere; o înregistrare veche nu trece drept nouă — T206 K4, T141b).
3. **Doar prima + ultima înregistrare:** `let all = RecordPlan::DIET.reads_all(k);` →
   `Diet2Wait { plan: &wait_plan, all, polls: &mut polls }`; timpul plăcii: `if all { analyze_sequence(..).sum() } else {
   kronos_diet2::ends_span(&b[..16], &b[16 * n_list..16 * (n_list + 1)], n_list, first)? }`.
4. (opțional) **Așteptarea planificată:** `wait_plan.observe(busy / 1000)` după fiecare pas, `wait_plan.forget()` la eroare.

## Ce așteptăm pe placă (DEDUS, T206 K5)
Pe pas, de la pasul 2: scrise 397 de cuvinte (azi ~1 533), citite 73 (azi 369): **~1,2 ms → ~0,3 ms**; cu lotul în paralel
(P10) ~0,09 ms. Pasul întreg cu remediul jurnalului + tăieturile: **~4–4,5 ms** la `antrenare mini` (azi ~370 ms). La primii
2 pași umbra scrie tot (locul pushbuffer-ului se mută după urcarea arenei), apoi 76 de cuvinte pe pas.

Rândul de verificat în raport: același `pierderea … abaterea`, aceleași greutăți față de oracol (numerele nu se schimbă).
