# C-71 (sesiunea Cyber-GPU 49): verificarea jurnalului pentru tot lanțul — observație pentru OS

**Doar observație / propunere.** Nu e o schimbare cerută în OS; nimic din ce e mai jos nu a fost testat pe RTX 3070 Ti.

## Ce e nou în cyber-gpu (doar adăugări, față de fixarea OS-ului)
- `nvidia_ga104::chain_log::{check_chain_journal, parse_step, ChainJournal, StepLog, ChainVerdict, step_ids, GSP_DOORBELL, WORK_DOORBELL}`;
- unealta de gazdă `verifica_jurnal_lant` (exemplu în `crates/nvidia-ga104/examples/`).
Nimic existent nu s-a schimbat (`WINDOW_CHAIN_FULL`, `run_chain_ready`, `check_i6`, `check_i7` la fel).

## Ce trebuie să conțină jurnalul de pe stick ca verificarea să meargă
1. Liniile de acces exact în formatul `Event` (`bar0 w …`, `cfg  w …`, `bar0 r …`) — cum le scrie deja OS-ul; un prefix de jurnal în față e acceptat.
2. Liniile `pas <id>: ok|ESEC` pe care le scrie `run_chain_ready` — **după** accesele pasului (OS-ul trebuie doar să le copieze în același fișier, în aceeași ordine).
3. Opțional, dar util: după `gpu lant`, o linie cu numărul de apeluri RM de la golden + I9 (suma `QueueTransportBell::rung`), de ex.
   `apeluri RM golden+I9: 25`. Unealta primește numărul ca parametru și verifică „o sonerie pe mesaj”.
4. Adresa `GspFwWprMeta` tipărită (pentru cutiile SEC2 de la I7), ca până acum.

Dacă jurnalul se taie (memorie plină), unealta spune „DE ANALIZAT” — nu ghicește.
