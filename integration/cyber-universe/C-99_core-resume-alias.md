# C-99 — I8 „timeout după 2 s” la CORE_RESUME (10 nov 2026, os-277): cauza, reparația, așteptarea lungă

- **Commit cyber-gpu:** `3aed5eb` (ramura `g0-g2-fundatie`; reparația e în `3fe11f9`, așteptarea lungă în `3aed5eb`).
- **Ce s-a întâmplat pe placă** (`gpu-jurnal-005.txt`): toate cele 420 de comenzi ale secvențiatorului au mers până la ultima,
  `CORE_RESUME`. Acolo, după resetul GSP-ului în RISC-V și cutiile LibOS, driverul a scris pornirea SEC2 (Booter) în
  `0x840100` (`CPUCTL`): **`0x840100: 0x50 <- 0x2 => 0x50`** — registrul arăta `0x50` = HALTED | **ALIAS_EN** și scrierea
  nu s-a prins (citit înapoi neschimbat). SEC2 n-a pornit niciodată, deci „predarea etapei 3” (`0x1180f8` bit 26) n-a venit:
  timeout după 2 s. La I7 același registru arăta `0x10` (ALIAS_EN = 0) și pornirea a mers — după ce rulează Booter (program HS),
  `CPUCTL` rămâne în spatele aliasului. Driverul NVIDIA 570.144 (`kflcnStartCpu_TU102`) și nouveau (`nvkm_falcon_v1_start`)
  verifică bitul `ALIAS_EN` și scriu atunci în `CPUCTL_ALIAS` (`0x840130`); `CORE_START` al nostru făcea deja asta, `CORE_RESUME` nu.
- **Ce s-a schimbat (doar adăugări, interfața publică la fel):**
  1. `falcon::start_cpu(bar, motor)` (nou): `STARTCPU` prin `CPUCTL_ALIAS` când `ALIAS_EN` e pus; folosit de `run_to_halt`,
     `CORE_START` și acum `CORE_RESUME`. Pe placă, la CORE_RESUME, jurnalul va arăta `bar0 w 0x840130: ... <- 0x2`.
  2. **Ideea lui Adrian (aprobată 17:16):** așteptările I8 de după sonerie care depind de GSP-RM — `CORE_WAIT_FOR_HALT`,
     `CORE_RESUME` (SEC2 gata, RISC-V activ) și `GSP_INIT_DONE` — au acum limita `sequencer::LONG_WAIT_US` = **60 s**
     (`INIT_DONE_US` a crescut de la 10 s la 60 s — același nume, OS-ul îl dă neschimbat). Când răspunsul vine, jurnalul notează
     **după cât timp**: `asteptat CORE_RESUME: SEC2 (Booter) gata: raspuns dupa N us`, `asteptat CORE_RESUME: GSP RISC-V activ: …`,
     `asteptat GSP_INIT_DONE (de la sonerie): …` (eveniment nou `Event::Waited { what, waited_us }`; `Display` ca mai sus;
     OS-ul folosește `_ =>` la potrivire, deci compilează neschimbat). Dacă nu vine: `timeout dupa 60000000 us`, ca până acum.
  3. **Oprirea manuală:** `gpu_core::hal::Clock` are o metodă nouă cu implementare implicită, `abort_requested(&mut self) -> bool`
     (implicit `false`). Toate buclele de așteptare (`poll_until`, deci și `falcon::poll`, `sequencer::poll_reg`, `wait_init_done`)
     o întreabă la fiecare tură; `true` oprește așteptarea cu eroarea nouă `GpuError::Aborted { waited_us }` („oprit manual dupa N us”).
     Jurnalul numește comanda oprită, ca la orice eroare (`pas I8 secventiator: comanda de mai jos s-a oprit: ESEC`, `secv #419 CORE_RESUME`).
- **Ce are de făcut OS-ul (C-99 partea 2):**
  1. `kernel/Cargo.toml` pe `3aed5eb` (toate crate-urile). Fără asta compilează, dar nu are reparația.
  2. Pentru Esc și rândul de pe ecran (ideea lui Adrian): în `TscClock` (gpu.rs) implementează `abort_requested`: citește tastatura
     fără blocare și întoarce `true` la Esc; tot acolo (sau în `delay_us`) poate scrie pe ecran la fiecare ~2 s „aștept GSP… N s”
     (ceasul știe `now_us`). Metoda e apelată la fiecare ~10–100 µs de așteptare, deci citirea tastaturii trebuie să fie ieftină.
  3. Opțional, în același spirit: `STAT_HEADER_US` (gpu_i8.rs, 2 s — așteptarea antetului cozii de stare după sonerie) poate crește
     la `sequencer::LONG_WAIT_US` și bucla lui poate întreba `clock.abort_requested()`.
- **Testat:** doar SIMULARE. `gpu-sim` modelează placa: după Booter, `CPUCTL` al SEC2 citește `0x50` și scrierea în `CPUCTL` e
  ignorată (OBSERVAT); driverul vechi pe acest model dă exact `Timeout { waited_us: 2000000 }` (placa: 2000003 µs), cel nou pornește
  prin `0x840130` și ajunge la `GSP_INIT_DONE` (T111); așteptarea lungă + oprirea manuală + rândurile `asteptat` (T112).
  423 de teste PASS, build no_std PASS, CI verde pe `3fe11f9`.
- **Ce urmărim la următoarea rulare `gpu lant`:** la `secv #419 CORE_RESUME`: `bar0 w 0x840130`, apoi rândurile `asteptat …`
  (cât a durat), apoi `GSP_INIT_DONE` sau prima eroare după el (primele RPC-uri RM: `GSP_GET_GSP_STATIC_INFO`, golden/I9).
