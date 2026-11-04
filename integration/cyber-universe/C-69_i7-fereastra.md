# C-69 — ce e gata pentru I7 și pentru măsurătoarea costului unei lansări (pentru sesiunea OS)

**Stare:** doar propunere. Nimic de aici nu se face în OS fără o comandă nouă în hub, și **doar după ce I6 a dat TRECE pe placă**
(fereastra 0 din `docs/FEREASTRA_C-69.md`). Totul e testat doar în SIMULARE.

**Ce e nou în cyber-gpu** (doar adăugări, compatibil; commit-urile `677dfbf` și `d5bf45a` sau mai noi pe `g0-g2-fundatie`):
- `gsp::falcon::read_after_i7(mmio, journal) -> GpuResult<I7Status>` + `I7_STATUS_READS` + `I7Status::passed()`: **doar 3 citiri**
  (cutiile SEC2 0/1 și `RISCV_CPUCTL` al GSP), fără permis, fără nicio funcție de scriere; fiecare citire intră în jurnal (pe stick).
  Același rol ca `read_frts_after_i6` la C-66: rezultatul lui I7 apare în jurnal chiar dacă `booter_load_dma` a întors eroare.
- `i6_log::{check_i7, check_i6_i7_journal, I6I7Journal, SEC2_LO, SEC2_HI, GSP_RISCV_CPUCTL}`: verificarea jurnalului I6 + I7 de pe stick.
- exemplul pe gazdă `verifica_jurnal_i7` (îl rulăm noi după fereastră).
- `launch_cost::{encode, analyze, timestamp_release, LaunchCost}`: măsurătoarea costului unei lansări din mărcile de timp ale plăcii
  (pentru fereastra I10, nu acum).
- `gpu_sim::cmdproc::CommandProcessor`: câmpuri noi `gpu_time_ns`, `sim_launch_ns`, `sim_method_ns`, `timestamps`; acceptă acum
  eliberarea de semafor cu marcă de timp (înainte o refuza). Testele OS care se bazau pe refuz ar trebui revăzute (n-am găsit niciunul).

**Ce ar trebui să aibă OS-ul pentru fereastra I7** (cu comandă separată):
1. Comanda `gpu i7 --scrie` (doar cu „DA”, la fereastră), rulată în **aceeași pornire** după `gpu i6 --scrie` care a dat TRECE
   (I7 refuză singur dacă WPR2 nu e ridicat).
2. Pregătirea în RAM (fără scrieri în placă): GSP-RM 570.144 de pe stick → radix3, LibOS, cozile, `GspFwWprMeta`; Booter-ul cu semnătura
   aleasă după siguranța SEC2. Afișează adresa fizică a `GspFwWprMeta` (ne trebuie la verificarea jurnalului).
3. `booter_load_dma(&mut GuardedBar0 cu permis, clock, &ucode, &image, wpr_meta_addr)`, apoi **întotdeauna** `read_after_i7` (fără permis)
   și rândul: `I7: cutia SEC2 0 = 0x…, GSP RISC-V ACTIV/NU`.
4. Bufferele DMA ale pregătirii rămân rezervate până la oprirea completă (GSP-RM le citește/scrie singur după I7).
5. Bus Master la loc la sfârșit, ca la I6. Oprire completă după fereastră.
