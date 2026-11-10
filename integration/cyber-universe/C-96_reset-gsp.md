# C-96 — I8 „eroare hardware 0xbadf5620” (10 nov 2026, os-276): cauza și reparația

- **Commit cyber-gpu:** vezi `stare/cyber-gpu.md` din hub (ramura `g0-g2-fundatie`, după `24dc19c`).
- **Ce s-a întâmplat pe placă** (`gpu-jurnal-004.txt`): după sonerie, prima cerere a secvențiatorului GSP-RM a fost
  `WRITE 0x110040 ← 0`, apoi `CORE_RESET` = resetul falcon-ului GSP (`0x1103c0 ← 1`, `← 0`). **Prima citire a lui
  `HWCFG2` (0x1100f4) după eliberarea resetului a întors `0xbadf5620`** (cod ringstation `FECS_PRI_CLIENT_ERR`,
  `dev_pri_ringstation_sys.h`): falcon-ul care tocmai rula GSP-RM nu răspunde încă după reset. Driverul nostru trata orice
  `0xbadf…` ca defect și oprea I8 pe loc. Driverul NVIDIA 570.144 (`kflcnWaitForResetToFinish_GA102`) și nouveau citesc
  registrul în buclă până arată „scrubbing gata” — valorile `0xbadf…` din buclă nu opresc nimic. La I6/I7 același reset
  mersese pentru că GSP-ul era liber (nu rula GSP-RM).
- **Ce s-a schimbat (doar adăugări, interfața publică la fel):**
  1. `nvidia_ga104::gsp::falcon::poll` și polurile secvențiatorului (REG_POLL, CORE_WAIT_FOR_HALT, CORE_RESUME): un
     `0xbadf…` înseamnă „încă nu”, bucla continuă; dacă registrul rămâne inaccesibil până expiră timpul, eroarea
     raportează valoarea citită (ex. `0xbadf5620`), nu un simplu timeout. O citire simplă (nu în buclă) rămâne strictă.
  2. `falcon::reset`: pulsul de reset ca în RM (`kgspResetHw_TU102`: 10 citiri ale registrului ENGINE după fiecare scriere),
     așteptarea scrubbing-ului cu limita RM (4 s, se termină la prima citire „gata”); două note diferite în jurnal:
     „RESET_READY nu a venit” (RM: se întâmplă) vs „HWCFG2 inaccesibil (0xbadf....) inainte de reset”.
  3. `CORE_RESET` face și pasul RM `kflcnSwitchToFalcon_GA102` (nou: `falcon::switch_to_falcon`): dacă `BCR_CTRL` nu
     spune deja „falcon”, selectează nucleul falcon și așteaptă VALID (RM: asta eliberează și blocarea priv a RISC-V-ului).
  4. `CORE_RESUME` începe cu `gsp_reset_into_riscv` (RM `kflcnResetIntoRiscv` / nouveau `gsp->func->reset`) înainte de
     cutiile poștale LibOS — pasul lipsea; ar fi căzut la următorul test, după reparația de mai sus.
  5. **Jurnalul**: fiecare cerere a secvențiatorului se scrie întreagă înainte de execuție (`nota: I8 secventiator: cererea
     GSP-RM, comenzile urmeaza`, apoi `secv #0 WRITE 0x110040 0x0`, `secv #1 CORE_RESET`, …); dacă o comandă se oprește,
     după `pas I8 secventiator: comanda de mai jos s-a oprit: ESEC` vine din nou `secv #i …` cu comanda vinovată.
     Eveniment nou în `gpu_core::diag::Event`: `SeqCmd { index, op, n, args }` (OS-ul folosește `_ =>` la potrivire, deci
     compilează neschimbat; `Display` îl tipărește ca mai sus).
- **Ce are de făcut OS-ul (C-96 partea 2):** `kernel/Cargo.toml` pe noul commit; niciun apel nou, nicio schimbare de cod
  OS. Jurnalul OS va avea rândurile noi `secv #…` (prin `Display`, ca până acum).
- **Testat:** doar SIMULARE (`gpu-sim` răspunde `0xbadf5620` la primele citiri după resetul unui GSP care rula GSP-RM —
  OBSERVAT pe placă; durata reală e DEDUS). Driverul vechi, pe acest model, dă exact eroarea de pe placă; cel nou trece până
  la `GSP_INIT_DONE`. 420 de teste PASS, build no_std PASS.
- **Ce urmărim la următoarea rulare `gpu lant`:** rândurile `secv #…` (ce a cerut GSP-RM), nota de dinaintea resetului,
  apoi fie `GSP_INIT_DONE`, fie o eroare nouă la primul pas diferit de simulare (de ex. `CORE_WAIT_FOR_HALT` sau
  `CORE_RESUME`: cutia SEC2 `mailbox0 ≠ 0`, RISC-V inactiv).
