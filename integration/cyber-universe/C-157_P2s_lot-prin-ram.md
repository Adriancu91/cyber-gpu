# Pentru OS: lotul și rezultatele pasului de antrenare prin RAM (C-157 p.2, după os-390)

**Ce e nou pentru placă: nimic nou ca program.** Programul de copiere `copy_b128` e deja încărcat în `train_mini_resident`
(`ctx.copy_va`) și a mers pe placă în os-390 în ambele sensuri (arena RAM → VRAM la început, VRAM → RAM la sfârșit, greutățile =
oracolul). Nou e doar locul: **7 copieri mici în lista pasului** (5 la început, 2 la coadă). Proba: cyber-gpu T209 (SIMULARE),
planul și predicțiile `docs/C-157_P2_LOT_PRIN_RAM.md`, modulul `nvidia_ga104::batch_ram` (doar adăugări).

## De ce (os-390, MĂSURAT)
Pasul întreg 4 150 µs, placa 3 625. Din cele 525 µs din afara plăcii: lotul (321 de cuvinte) scris prin fereastra VRAM, fiecare recitit
= ~245 µs la „golirea in VRAM” + 49 µs „lotul sus”; pierderile + norma citite prin fereastră = 70 µs.

## Ce (în `train_mini_resident`, `gpu_i14.rs`, doar pentru drumul `--vram`; cel vechi rămâne pentru comparație)
1. Zona de tranzit = **începutul lui `stage`** (RAM-ul fixat al arenei, nefolosit între urcarea și aducerea arenei). Pe pas:
   `let v = batch_ram::via_ram(&plan, x, y, stage.va)?;` apoi **`ram_write_at(&stage, 0, &v.staging)`** (2 048 B la `antrenare mini`)
   — nu `ram_write`, care pune pe zero toate paginile lui `stage` (~26 MiB). La pasul 0 la fel (lotul 0 e deja în arenă, dar așa toți
   pașii sunt la fel).
2. În locul buclei `for (va, bytes) in plan.batch_uploads(x, y)? { put_bytes(...) }` → nimic prin fereastră.
3. În locul lui `ts::pack(&list, prog_va, ctl.va)` → `batch_ram::pack_via_ram(&list, prog_va, ctx.copy_va, ctl.va, &v)?`
   (aceleași sloturi de `SLOT` octeți: 5 copieri, lista, 2 copieri). **`zone_bytes` și înregistrările pe zero:** `n_launch +
   batch_ram::extra_launches(&v)` (= 75 + 7 la `antrenare mini`); `n_list` = numărul de QMD-uri întors.
4. După așteptare, în locul celor două citiri `step_reads()` prin fereastră:
   `let (rows, norm) = batch_ram::results(&ram_read(&stage, v.staging.len() as u64), &v)?;`
5. Rândul `kt antrenare-mini lot prin RAM (1/0)` + aceleași bucăți ca acum (planul lotului, lotul sus, …) ca să se vadă unde s-a mutat timpul.

## Ce așteptăm pe placă (DEDUS, T209 R3/R5, cu costurile MĂSURATE în os-390)
- Aceleași numere: pierderile, normele, greutățile bit cu bit cu os-390 (T209 R1; arena identică, R2).
- Prin fereastră pe pas (de la pasul 3): scrise 397 → **83** cuvinte (doar valorile înregistrărilor), citite 73 → **8**;
  „golirea in VRAM” ~302 → ~60 µs, „lotul sus” 49 → ~2 µs, „pierderile + norma jos” 70 → ~3 µs; placa +7 lansări ≈ +20–30 µs.
- **Pasul întreg ~3,8 ms ≈ ~16 700 simb/s** (os-390: 4 150 µs, 15 421).
- Dacă o copiere de la coadă n-ar ajunge, procesorul citește pierderea 0 și norma 0 (zona de rezultate e scrisă pe zero cu lotul) —
  valori imposibile (pornirea e ~ln 16 384 = 9,7), deci prinse (T209 R4); verdictul față de oracol oricum pică.

## Atenție
- Cu lotul în paralel (P10) trebuie **două** zone de tranzit (câte una pe loc de lot), ca procesorul să nu scrie lotul următor peste
  cel pe care placa îl copiază încă.
