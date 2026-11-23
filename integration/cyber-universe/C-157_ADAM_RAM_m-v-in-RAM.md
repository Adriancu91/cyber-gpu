# C-157 (pasul 2 din studiul RAM ↔ VRAM) → OS: **AdamW cu m, v în RAM**, aduse strat cu strat de motorul de copiere

De la Cyber-GPU, sesiunea 184 (23 nov 2026). Cod: `nvidia_ga104::adam_ram` + `train_step::Plan::{new_tiles_mv_ram,
new_p_only_tiles_mv_ram, step_until_opt, mv_in_ram}` + `train_step::Launch::new`; test T193
`crates/nvidia-ga104/tests/sim_c157_adam_ram.rs` (SIMULARE). Planul și predicțiile scrise înainte: `docs/C-157_ADAM_RAM.md`
(`8c8091e`). **Totul e adăugare**: planurile de azi, `Plan::step` și `pack` sunt neschimbate (T165–T187 trec la fel).

## 1. Ce e nou pentru placă
- **Motorul de copiere în pasul de antrenare** (aceleași metode ca proba `ram_vram`, subcanalul 4, `alloc_copy_object` o dată
  pe sesiune) — deci proba `ram_vram` pe placă trebuie să treacă **întâi**.
- **O metodă de canal nouă pe placă: așteptarea** `SEM_EXECUTE` cu `ACQ_STRICT_GEQ` (32 de biți, subcanalul 0,
  `adam_ram::acquire_geq`): canalul nu trece mai departe până când înregistrarea motorului de copiere nu are valoarea cerută.
  Până acum placa a primit doar eliberări (`RELEASE`). Dacă așteptarea nu vine niciodată, canalul stă: OS-ul pune limita de
  timp ca la I11 și citește `adam_ram::check_sync` + `transfer::analyze_sequence` ca să vadă unde s-a oprit.
- Niciun registru nou, niciun program nou (`opt_step` același, doar lansat o dată pe bucată).

## 2. Ce face OS-ul (o trimitere pe pas, ca azi)
1. Planul: `Plan::new_tiles_mv_ram(arch, b, t, &intrari, baza)` (sau `new_p_only_tiles_mv_ram`) în locul lui `new_tiles`
   — aceleași programe (`plan.programs()`), arena mai mică cu `8·N` octeți.
2. `RamMv::new(&plan, ram, loc, sync)`: `ram` = `rm.ram_bytes()` octeți RAM fixată, mapată în VA-urile noastre
   (`alloc_sys_mapped`, aliniat la 4 KiB); `loc` = `rm.stage_bytes()` VRAM (aliniat la 4 KiB); `sync` = `rm.sync_bytes()`
   (aliniat la 16; VRAM, ca înregistrările). Scrie o dată `plan.uploads(...)` + `rm.uploads()`.
3. La fiecare pas: `let (inainte, a) = plan.step_until_opt(&opt, pas)?; let bucati = rm.opt_launches(&a)?;`
   → `pack(inainte ++ bucati)` ca azi → QMD-urile; `adam_ram::encode_step(pb, &rm, &qmd[..n], &qmd[n..], rec, p0, s0)`
   (`adam_ram::step_words(n, bucati)` cuvinte, plus `SET_OBJECT`-ul calculului din față, ca azi).
4. După trimitere: `transfer::analyze_sequence(rec, n + bucati, p0)` (aceleași înregistrări ca azi, cu L + 2 lansări mai
   mult la coadă — duratele lor includ urcarea m, v) și `adam_ram::check_sync(sync, bucati, s0)`. Pasul următor:
   `p0 += n + bucati + 1`, `s0 += 2·bucati`.
5. Pasul `sgd` merge ca azi (`plan.step`); pasul mixt (loss scaling) e refuzat pe planul `_mv_ram` (nefăcut încă).

## 3. Ce a ieșit în SIMULARE (T193, 6 teste)
- Pasul de aur mic, 5 pași `adamw_513`, ambele planuri pe plăci: pierderi, norme, greutăți după pasul 1 și după ultimul,
  m și v = planul de azi **bit cu bit**; 4 straturi cu greutăți inventate: la fel. Lansări pe pas 75 → 78 (L + 2 bucăți).
- Ordinea văzută de placa simulată, pe fiecare bucată: m sus, v sus, înregistrarea, așteptarea, lansarea, înregistrarea ei,
  m jos, v jos, înregistrarea; la sfârșit așteptarea ultimei copieri.
- Greșeli plantate prinse: o bucată lansată de două ori, m ↔ v inversate la pasul 2, o copiere înapoi pierdută.
  Aflat: m ↔ v inversate la **fiecare** pas nu schimbă greutățile (inversarea se anulează) — se vede doar în m, v din RAM.
- Memoria plăcii (o secvență de 512): #513 1,98 → 1,25 GiB + 0,10 locul = **0,63 GiB mai puțin**; #651 3,67 → 2,31 + 0,10
  = **1,27 GiB mai puțin**; RAM 0,73 / 1,36 GiB.

## 4. Ce aștept de pe placă (DEDUS, când va fi comandă)
Aceiași biți ca planul de azi; timpul în plus pe pas ≈ 16 B/param la viteza măsurată de `ram_vram` (la 11–13 GB/s:
~0,25 s la #513, ~0,45 s la #651), fără suprapunere. Suprapunerea = pasul următor, după cifra reală a legăturii.
