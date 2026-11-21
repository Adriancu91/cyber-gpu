# C-157 punctul 2b — pozițiile lente din `os-324` și cele 238 de cuvinte pe simbol

**Pentru:** sesiunea OS (C-158). **De la:** Cyber-GPU, sesiunea 133, 21 nov 2026. **Commit:** `46078f4` (ramura `g0-g2-fundatie`).
**Ce e nou pentru placă: nimic.** Aceleași programe, aceleași lansări, aceiași octeți; doar unde și când scrie OS-ul jurnalul și
cât de mult uită umbra. Totul de mai jos e citit din jurnalul de pe stick (`gpu-jurnal-024.txt`, ceasul fiecărui rând) și din
codul OS-ului — DEDUS, nu măsurat separat pe poziții (rândul acela încă nu există; vezi pasul 3).

## 1. Ce a ieșit: două poziții lente, nu „câteva”
Mediana 4 083 µs (245 simb/s), dar suma celor 18 poziții ≈ 165 ms (109 simb/s). Din ceasurile rândurilor:

| poziția | durata | ce e în ea |
|---|---:|---|
| 0 | ~40,2 ms | (a) **~16,9 ms** fereastra `0x1b2d`: toate cele 196 de locuri de lansare rescrise întregi (~25 000 de cuvinte) — umbra lansărilor era uitată; (b) **~20,4 ms** după sonerie, înainte de așteptare: o scriere pe stick forțată — rândurile de rutină rămase de la „scară” (`kd::ladder`, 14 trimiteri fără `flush_live`) au atins limita de 64 (`GEN_MAX_PENDING`) și următorul rând a plecat cu `Flush::Now`; (c) ~1,4 ms pushbuffer-ul întreg (prima trecere prin umbra lui, normal) |
| 1–14, 16, 17 | ~4,1 ms | normal: placa ~2,9 ms (așteptarea), driverul ~1 ms |
| 15 | ~58,4 ms | **~54 ms scrierea pe stick** a rândurilor ținute: a 16-a chemare a `live_flush` (`FLUSH_TOKENS` = 16) |

40,2 + 58,4 + 16 × ~4,1 ≈ 165 ms = exact suma raportată. Aceeași diferență exista și în os-322 (ascunsă de driverul mai lent).
**Atenție la generările lungi:** o scriere de ~55 ms la fiecare 16 simboluri = ~3,4 ms în plus pe simbol în medie — aproape cât un
simbol întreg. La 500 de simboluri asta e frâna principală, nu placa.

Unealta noastră o arată deja pe orice jurnal: `cargo run --release -p nvidia-ga104 --example timpul_driverului -- gpu-jurnal-024.txt`
→ „lăsate deoparte (mai lente de 1,5 × mediana): 21220, 50295, 22986, 23038, 21988, 58400 us” (primele 4 sunt scara, ultimele 2
pozițiile 0 și 15).

## 2. Ce propunem (toate în OS, nimic nou pe placă) — 3 schimbări mici
1. **Nicio scriere pe stick în mijlocul generării.** În `gpu_i6::live_flush`, cât rulează `kronos --gpu` cu viteza (fast): în loc
   de „la 16 simboluri sau la o secundă”, rândurile de rutină stau în memorie și se scriu la final (`live_close`) sau la panică
   (`panic_flush`, există deja). Două variante:
   - **A (recomandată):** inel în memorie cu rândurile ultimelor ~64 de poziții (cele mai vechi rânduri de rutină se aruncă,
     numărate într-un rând `kt randuri de rutina aruncate`); scriere doar la final/panică. Cost: 0 în generare. Risc: la o
     blocare fără panică lipsesc rândurile de rutină — dar ele sunt identice de la o poziție la alta (ferestre + sonerie); rândurile
     care contează (valori, pași, erori) rămân `Flush::Now`.
   - **B:** scriere doar după timp, la 5 s (`FLUSH_US`), fără pragul de 16 simboluri: ~55 ms la 5 s ≈ 1 %.
2. **Golirea înainte de prima poziție**, în afara ceasului poziției: după scară (pasul 3 din `gpu_i14.rs`), înainte de bucla
   textelor: `runner.journal.flush_live()` + scrierea rândurilor ținute (un rând `Flush::Now`, de ex. o notă „generarea incepe”).
   Scoate cei ~20 ms din poziția 0.
3. **Umbra uitată doar unde s-a scris.** `TokenIo::write` (în `BoardRunner`) uită acum toată umbra lansărilor și a pushbuffer-ului
   (`sh.forget()`). Scara scrie activările (`vas.layers[0].x`), nu locurile de lansare. Cu `WriteShadow::forget_range(pa, len)`
   (nou la noi, `46078f4`) se uită doar cuvintele din zona scrisă → poziția 0 scrie doar ce diferă (~240 de cuvinte în loc de
   ~25 000): **−~16 ms** în poziția 0. Exemplu:
   ```rust
   // în locul lui `sh.forget()` din `fn write`:
   if let Some(sh) = self.shadow.as_mut() { sh.forget_range(pa, 4 * words.len() as u64); }
   if let Some(sh) = self.push_shadow.as_mut() { sh.forget_range(pa, 4 * words.len() as u64); }
   ```
   (adresa fizică `pa` e cea calculată deja în `write`; în afara zonei umbrei funcția nu face nimic.)

**Estimare (DEDUS):** cu 1 + 2 + 3, media ≈ mediana: ~73 ms pe 18 poziții → **~245 simb/s** „cu driverul” (azi 109,11), peste
Windows/PyTorch (129,12). Cu P3 după: ~280.

## 3. Rândul cerut de centrul de comandă: timpul fiecărei poziții
Ca să nu mai deducem din ceasurile rândurilor: `kronos_fast::TimeSplit::slow_rows()` / `journal_slow(j)` (nou, doar adăugare).
Scrie 3 rânduri + 5 pe fiecare poziție lentă (peste 2 × mediana; cel mult 8 poziții, ca jurnalul să rămână scurt — fiecare rând de
valoare e o scriere pe stick):
```
kt suma simbol fara verificare (us)        → media = suma / kt simboluri
kt pozitii lente (peste 2 x mediana)
kt timp in plus in pozitiile lente (us)
kt lenta: pozitia / simbol (us) / placa (us) / jurnal (us) / driver (us)   (de 5 ori pe fiecare pozitie lenta)
```
Plus `slow_lines("placa")` pentru raport („pozitia lenta 15 (placa): 58400 us = placa 2624 + jurnal 54300 + driver …”). Cititorul:
`kronos_fast::parse_slow`. În OS: lângă `split.journal(journal)` (gpu_i14.rs, ~l. 1463) adăugați `split.journal_slow(journal)` și
`lines.extend(split.slow_lines(label)…)`. Merge în Release-ul cu P3, ca citire de ceas (nu e un al doilea lucru nou pentru placă).
Dacă propunerile 1–3 intră, rândul arată că pozițiile lente au dispărut (`kt pozitii lente` = 0).

## 4. Cele 238 de cuvinte VRAM pe simbol (SIMULARE: 48) — explicate, la cuvânt
Calculat din octeții pe care îi construiește driverul pentru două poziții la rând ale unui model cu forma lui #513 (testul T144,
`crates/nvidia-ga104/tests/c157_p2b_diet_words.rs`, aritmetică, fără placă):

| ce | cuvinte | de ce se schimbă la fiecare simbol |
|---|---:|---|
| pushbuffer: valoarea fiecărei înregistrări | **197** | fiecare lansare are o înregistrare cu valoarea `first + k`, iar `first` crește cu `STRIDE` la fiecare trimitere (196 lansări + înregistrarea de start) |
| cbuf-urile lansărilor | **38** | simbolul și poziția (embed), poziția în cache-urile K/V și lungimea atenției, în fiecare din cele 12 straturi |
| QMD-urile | 0 | același program, același loc |
| inelul (intrarea + `GP_PUT`) | 3 | presupus (scris de OS în afara umbrelor) |
| **total** | **238** | = exact ce a măsurat placa (median) |

Cele 48 din SIMULARE vin de la modelul mic de test (3 straturi, ~51 de lansări): numărul crește cu lansările, nu e o scurgere.
Estimarea noastră din C-147 punctul 7 era „~200 de cuvinte”. **P3** (88 de lansări) le coboară la ~89 + ~38 + 3 ≈ **130**.
Un pas mai departe (nu acum): o singură înregistrare la finalul simbolului în loc de una pe lansare ar lăsa ~42 de cuvinte, dar
am pierde duratele pe fel de lansare (`kp`) — de decis când vrem viteza mai mult decât măsurătorile.

## 5. Ce NU e dovedit
Nimic nou pe placă. Atribuirea timpului din pozițiile 0 și 15 e DEDUSĂ din ceasurile rândurilor și din codul OS-ului (rândurile
sunt ștampilate când se creează; golurile fără rânduri sunt munca procesorului fără jurnal). Rândul `kt lenta` o va confirma.
