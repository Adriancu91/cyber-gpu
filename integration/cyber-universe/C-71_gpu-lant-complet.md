# C-71 — `gpu lant` complet: I6 → I7 → I8 → golden → I9 → I10 → costul lansării (pentru sesiunea OS)

**Stare:** propunere pentru partea OS din C-71. Tot ce e aici e testat doar în SIMULARE (cyber-gpu T78–T80, commit `e77d986` sau mai nou).

**Ce e nou în cyber-gpu** (doar adăugări):
- `chain::WINDOW_CHAIN_FULL` (13 pași, ID-urile: `pregatire-i6`, `bus-master-pornit`, `i6-fwsec-frts`, `i6-wpr2-citit`, `i7-booter-gsp`, `i7-citit`,
  `i8-gsp-init`, `i8e-golden`, `i9-canal-lucru`, `i10-add64`, `i10-cost-lansare`, `bus-master-oprit`, `tur-dupa`), `chain::READY_FOR_WINDOW`,
  `chain::run_chain_ready(steps, ready, writes_allowed, journal, run)`. `WINDOW_CHAIN` și `run_chain` sunt neschimbate.
- `gsp::golden::QueueTransportBell` — ca `QueueTransport`, plus `bell` (soneria GSP după fiecare mesaj) și `rung`. **Pe placă folosiți-l pe acesta**
  pentru golden image / I9: `QueueTransport` nu sună niciodată, iar GSP-RM real răspunde doar după sonerie (în simulare, fără sonerie: timeout, T79).

**Cum arată partea OS** (modelul complet e testul `gsp::user_space::tests::full_window`):
1. `gpu lant` (cu „DA”) → `run_chain_ready(WINDOW_CHAIN_FULL, READY_FOR_WINDOW, true, ...)`; fără „DA” → doar citire, ca acum.
2. Fiecare ID are ramura lui; ID-urile pe care OS-ul nu le are încă → `Err` (cum faceți deja: „pas fara implementare in OS”), deci lanțul se oprește
   cinstit acolo și pune Bus Master la loc. Se pot adăuga pe rând, fiecare testat în QEMU cu `gpu lant --sim`.
3. `i6-wpr2-citit` = `read_frts_after_i6`; `i7-citit` = `read_after_i7` (+ `passed()`); amândouă fără permis.
4. `i8-gsp-init`: `SetSystemInfo` + `SetRegistry` în coadă, `ring_doorbell`, `wait_init_done`.
5. `i8e-golden`, `i9-canal-lucru`: `init_golden_image`, `open_user_space`, `open_compute_group`, `open_work_channel_cleared`, `query_gr_info().check_sm86()`
   prin `QueueTransportBell` cu `bell = ring_doorbell` pe `GuardedBar0` cu permis.
6. `i10-add64`, `i10-cost-lansare`: `WorkQueue` + `launch_cost::encode` / `analyze`; rândul de pe ecran `LaunchCost::line()`.
   Soneria canalului de lucru (`0xBB0090`, DEDUS) nu e încă pe lista permisă — ea e ultimul lucru nou care trebuie aprobat înainte de I10 pe placă.
