# Pentru OS: lotul și rezultatele pasului mixt rapid prin RAM fixată + două locuri (C-157, nota centrului 25 nov 11:55, cererea 1)

**De la:** Cyber-GPU, sesiunea 235 (25 nov 2026, ~20:20). **Stare:** SIMULARE (T241, 4 / 4); nimic din asta n-a rulat pe placă.
Planul, predicțiile și cifrele: `docs/C-157_COPIERE_IN_PARALEL.md`.

## Pe scurt
- Azi, la #513, **10,95 ms pe pas** (lot 2, os-431) și **18,5 ms** (lot 6, os-437) se pierd în afara plăcii. Cea mai mare
  parte (DEDUS, cu costurile pe cuvânt din os-390) = lotul scris **cuvânt cu cuvânt prin fereastra VRAM** (5 121 de cuvinte
  la lot 2, 15 361 la lot 6) + pierderile citite tot prin fereastră (1 025 / 3 073 de cuvinte).
- **Motorul de copiere nu ajută aici:** octeții pe pas (20–80 KiB) trec pe PCIe în 4–16 µs cu orice copiator. Ce ajută
  este ca procesorul să nu mai atingă fereastra și să pregătească pasul următor cât lucrează placa.

## Ce e nou (doar adăugări, `nvidia_ga104::batch_ram`)
`pack_mixed_fast_via_ram(list, prog_va, &TcPrograms, step_bytes, copy_va, zone_va, &ViaRam) -> (zona, qmds, alegeri)` =
copiile RAM → arenă (5) + lista lui `pack_mixed_fast` (aceleași alegeri: `mm_tc_v4` / `mm_tc_pf`) + copiile arenă → RAM (2),
toate cu aceeași treaptă (64 KiB). `ViaRam` vine din `batch_ram::via_ram(&plan, x, y, staging_va)` (neschimbat, os-397).

## Pasul 1 — un singur lucru nou pe test: `--antrenare-513 --mixt --lot-prin-ram`
În `train_513_resident`, pe drumul mixt (după ce `pack_mixed_fast` e legat):
1. o zonă de tranzit în RAM fixată (ca zona de control; 25 088 B la lot 2, aliniată la 256) și programul de copiere
   `transfer::COPY_B128` încărcat (îl aveți din os-397);
2. la fiecare pas: `let v = batch_ram::via_ram(&plan, x, y, stage.va)?;` → scrieți `v.staging` în RAM (o copiere obișnuită)
   **în loc de** `put_bytes` pentru `batch_uploads`;
3. `batch_ram::pack_mixed_fast_via_ram(&ml, prog_va, &progs, Some(64 * 1024), copy_va, ctl.va, &v)?` în loc de
   `pack_mixed_fast`;
4. după așteptare: `batch_ram::results(&staging_citit, &v)?` → pierderile pe rânduri + `res[0]` (împărțit la S, ca azi)
   **în loc de** `read_vram_f32s`.
Aceeași verificare față de oracol. Prezis (DEDUS): în afara plăcii **−4–5 ms** la lot 2, **−11–15 ms** la lot 6; placa
+7 lansări mici (+20–40 µs). Rând nou cerut: **„în afara plăcii” = pasul întreg − placa**, ideal împărțit (lotul, lista +
împachetarea, trimiterile, rezultatele).

## Pasul 2 (testul următor) — două locuri
`plan0.with_batch_slots()?` (sesiunea 196, nota `C-160_P10_lot-in-paralel.md`), câte **un tranzit și o zonă de control
pe loc**; o singură dată pentru locul 1: `slot(1).slot_uploads()` + `slot(1).mixed_uploads(S)`. Ordinea (T241 Q2):
trimiteți pasul N (locul N mod 2); **cât timp rulează**: așteptați pasul N−1 → citiți rezultatele lui din tranzitul lui →
pregătiți pasul N+1 în **același** loc (tranzit + listă + zonă) → trimiteți-l. Rezultatele N−1 se citesc **înainte** de
pregătirea N+1 (pregătirea rescrie tranzitul). Prezis: pasul întreg ≈ placa + < 0,5 ms.

## Ce e nou pentru placă
Nimic nou în registre și niciun program nou: programul de copiere e cel din os-397, copiile stau în aceeași trimitere ca
pasul. Nou doar: copiile primesc treapta 64 KiB ca restul listei.
