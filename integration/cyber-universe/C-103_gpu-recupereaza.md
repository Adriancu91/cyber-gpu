# C-103 — `gpu recupereaza`: placa readusă la starea de pornire fără repornirea calculatorului (partea GPU, SIMULARE)

Pentru sesiunea OS (cyber-universe). Commit cyber-gpu: `364ece2` pe `g0-g2-fundatie` (sau mai nou). **Doar adăugări.** Studiul complet: `cyber-gpu/docs/C-103_RESET_PLACA.md`.

## Pe scurt
Calea driverului NVIDIA (570.144, MIT) și a nouveau pentru a coborî WPR2 fără reset PCIe e **Booter Unload**: programul semnat `booter_unload-570.144.bin`
(**deja pe stick**, al 4-lea fișier I4) rulat pe SEC2 cu mailbox0/1 = `0xFF`/`0xFF`; după el `WPR2_ADDR_HI` trebuie să citească 0. Apoi I6 → I7 → I8 se pot relua
în aceeași pornire. Nimic testat pe placă.

## Ce există acum în `cyber-gpu`
- `nvidia_ga104::gsp::falcon::booter_unload_dma(bar, clock, &HsUcode, &DmaImage) -> GpuResult<Unloaded>` — ordinea RM: reset GSP (GSP-RM se oprește), reset SEC2,
  nucleul falcon, încărcarea DMA (același `hs_load::dma_load` ca la `booter_load_dma`), `run_to_halt` cu `BOOTER_UNLOAD_MAILBOX`, mailbox0 trebuie 0, WPR2 trebuie 0.
  `Unloaded::Wpr2NuEraRidicat` = WPR2 nu era ridicat, programul nu s-a rulat (ca RM); `Unloaded::Wpr2Coborat` = gata. Erori: `HardwareFault { code: mailbox0 }`,
  `InvalidArgument("… WPR2 e inca ridicat …")`. Jurnal: `valoare C-103 Booter Unload: WPR2 inainte …`, `pas C-103 Booter Unload: WPR2 coborat: ok`.
- Imaginea: `firmware::GspRole::BooterUnload` (există din I4) → `hs_load::HsUcode`/`DmaImage` exact ca pentru `BooterLoad` (aceeași funcție de pregătire; doar rolul diferă).
- `gpu_pci::caps::reset_capabilities(cfg) -> Option<ResetCapabilities>` + `.summary()` — **doar citiri**: FLR anunțat?, `No_Soft_Reset`, D-state, PME din D3cold.
- `gpu-sim`: `SimUcode::BooterUnload`, `SimFault::Wpr2Stuck` pentru testele voastre în QEMU.

## Ce are OS-ul de făcut (când ajunge la C-103; C-102 / os-280 are întâietate)
1. `gpu` / `raport`: rândul `reset_capabilities(...).summary()` (0 scrieri; poate intra în turul de citire C-54).
2. Comanda `gpu recupereaza` (doar cu „DA”, cu Adrian de față, ca orice scriere): Bus Master oprit dacă era pornit (ca la curățarea lanțului), apoi
   `booter_unload_dma` cu imaginea `BooterUnload` din fișierul de pe stick; pe ecran rezultatul (`Unloaded`) și valoarea WPR2 după.
3. Opțional, în `gpu lant`: dacă I6 pregătire spune „WPR2 deja ridicat” și Adrian a dat „DA” pentru recuperare, rulează `gpu recupereaza` și reia lanțul de la I6 —
   asta e „a doua încercare fără cablul scos”. Paginile din RAM mapate anterior pentru placă rămân în carantină până la repornire (regula din `recovery.rs`).
4. Ce se urmărește în jurnal la prima rulare pe placă: `WPR2 inainte`, scrierile `0x840040/0x840044 <- 0xff`, `bar0 w 0x840130`/`0x840100` (start), mailbox0 după oprire,
   apoi `bar0 r 0x1fa828` (WPR2 HI) = 0 → `pas C-103 Booter Unload: WPR2 coborat: ok`. Dacă mailbox0 ≠ 0: codul se notează, următoarea variantă e FWSEC-SB înainte (vezi studiul).

Nimic din FLR / Secondary Bus Reset nu e implementat ca scriere — rămân variante de rezervă, descrise în studiu, la decizia OS-ului și a lui Adrian.
