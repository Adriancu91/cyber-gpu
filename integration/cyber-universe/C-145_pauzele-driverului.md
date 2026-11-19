# C-145 → C-146: driverul mai rapid la `kronos --gpu` (notă pentru sesiunea OS)

De la Cyber-GPU, sesiunea 116 (19 nov 2026). Analiza și cifrele: `docs/C-145_PAUZELE_DRIVERULUI.md`. Totul e
**adăugat** (nimic redenumit sau șters; OS-ul fixat pe `3a23dc0` compilează neschimbat) și **oprit implicit**:
fără pașii de mai jos, `os-308` se comportă exact ca `os-307`. Testat doar în SIMULARE (`tests/sim_c145.rs`, T134).
Nu schimbă nimic din ce primește placa (aceleași comenzi, aceleași valori, aceeași ordine a acceselor).

## Unde se duce timpul azi (pe scurt)
294,8 ms pe simbol = ~246 ms jurnalul (15 rânduri × scriere + sincronizare stick) + ~41 ms rescrierea celor 196 de
sloturi de lansare prin fereastra VRAM (cu recitire) + 22,9 ms placa (suprapus) + restul. Cifra 0,24 simboluri/s
include verificarea cu motorul de referință (3,87 s pe poziție) între simboluri; 3,10 nu o include.

## Ce poate lega OS-ul (fiecare separat, în ordinea câștigului)

1. **Jurnalul ținut în memorie pentru rândurile de rutină** (`gpu_core::diag`):
   - `gpu_i6::journal(live)` → `Journal::with_live_sink(JOURNAL_CAP, LiveSink { line, flush })`, unde
     `line(e, Flush::Later)` doar adaugă rândul (cu ora lui) într-un tampon din RAM, iar `line(e, Flush::Now)` și
     `flush()` scriu tamponul + rândul **dintr-o singură** `fs::write` și apoi `fs::flush`. Ora din paranteză
     rămâne ora la care s-a petrecut accesul (se ia în `line`, nu la scriere).
   - înainte de bucla de generare (și, opțional, înaintea urcării greutăților):
     `j.set_routine(Some(nvidia_ga104::kronos_fast::gen_routine(&NOTE)))` cu
     `NOTE = ["C-55 gpu lant: fereastra cu Adrian (P6-42)", "soneria canalului de lucru (I10): scriere fara recitire"]`
     (constante `&'static str` — exact textele pe care le pune OS-ul);
   - după fiecare simbol: `j.flush_live()` (sau la 16 simboluri, dacă vreți viteza maximă; vezi tabelul din doc §4);
   - după buclă: `j.set_routine(None)` (golește tot); `j.live_stats()` dă câte rânduri au fost ținute / golirile.
   - Ce rămâne imediat: pașii, rândurile `kg`/`kv`/`kt`, citirile, configurația PCI, orice altă scriere.
2. **Umbra sloturilor de lansare** (`kronos_fast::WriteShadow`), în `BoardRunner::submit`: o umbră pe zona
   `self.launches` (`WriteShadow::new(launches.pa, launches.size)`), apoi pentru cbuf și QMD:
   `for (pa, r) in sh.changes(cb_pa, &la.cbuf)? { put_bytes(w, pa, &la.cbuf[r])? }` + `sh.remember(cb_pa, &la.cbuf)?`
   (la fel pentru QMD). La orice eroare: `sh.forget()`. Doar pentru sloturi — niciodată pentru înregistrări sau
   logit-uri. Opțional, la sfârșit: recitiți zona și `changes(pa, citit)` trebuie să fie gol (altfel jurnal + oprire).
3. **Verificarea pe procesor**: fie motorul optimizat al OS-ului în `CpuCheck` (14 ms în loc de 3,87 s; judecata
   rămâne `km::judge`, toleranța 1e-3), fie `Generation::run_check_after(&mut runner, &mut check)` în loc de
   `run(..., Some(check))` — aceeași judecată pentru fiecare poziție, după placă; memoria: `g.deferred_bytes()`.
4. **Raportul cerut de C-146** (`kronos_fast::TimeSplit`): pentru fiecare simbol
   `TokenSplit { pos, host_us, board_us, journal_us, screen_us, check_us }` — `host_us` = ce măsurați deja (P6-95),
   `board_us` = `(end_ns - start_ns)/1000` din `TokenOutput`, `journal_us` = timpul petrecut în `line`/`flush` în
   timpul simbolului (un contor global în `gpu_i6`), `screen_us` = timpul de afișare în timpul simbolului (0 dacă nu
   afișați nimic), `check_us` = din `CpuCheck::us`. La final `split.journal(&mut j)` (rândurile `kt …`) și
   `split.lines("placa")` în raport. Rândurile `kg`/`kv` rămân exact ca în `os-307`.

## Ce așteptăm de la `os-308` (estimare, nemăsurat)
Cu 1 + 2 și golire pe simbol: ~45 ms pe simbol (~20–22 simboluri/s); cu golire la 16 simboluri: ~30 ms (~30–34).
Plafonul (doar lansările): 43,9. Dacă verificarea de referință rămâne între simboluri, cifra „cu verificare”
rămâne ~0,25.
