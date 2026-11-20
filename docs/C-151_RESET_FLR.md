# C-151 punctul 3 — resetul plăcii din interior (unealta 2): FLR la începutul lanțului, partea Cyber-GPU

*Sesiunea Cyber-GPU 122, 20 noiembrie 2026, în cloud, fără placă. Totul de aici e **SIMULARE**; pe RTX 3070 Ti: NOT TESTED.*
Pornire: `docs/C-142_K513_VITEZA_FLR.md` §4, varianta 1 (propusă de noi în C-142, aprobată de Adrian în C-151/C-152).
Nota pentru OS, cu pașii exacți: `integration/cyber-universe/C-151_reset-flr.md`.

## 1. Sursele, verificate încă o dată (20 nov 2026)

| Sursa (citită ca specificație; nimic copiat) | Ce face, în ordine |
|---|---|
| **driverul NVIDIA 570.144** (MIT; `src/nvidia/src/kernel/gpu/bif/arch/turing/kernel_bif_tu102.c`, `kbifDoFunctionLevelReset_TU102`, folosit și de GA10x prin `kbifDoFullChipReset_GA100` când placa anunță FLR) | salvează configurația PCI (+ măștile MSI-X) → oprește Bus Master → așteaptă „Transactions Pending” = 0 (la depășire: doar mesaj, merge mai departe) → **declanșează FLR** (Device Control bit 15) → **100 ms** → **rescrie configurația** → **așteaptă GFW boot** (VBIOS își reface inițializarea) → (șterge un contor intern XVE — registru NVIDIA pe care **noi nu-l scriem**) → verifică Vendor ID = NVIDIA |
| **tinygrad** (MIT, `tinygrad/runtime/support/nv/nvdev.py`, ramura principală la 20 nov 2026, `c0bb645`) | dacă `NV_PFB_PRI_MMU_WPR2_ADDR_HI ≠ 0`: Bus Master oprit → „WPR2 is up. Issuing a full reset.” → resetul funcției prin Linux (`/sys/bus/pci/devices/…/reset`) → 100 ms → Bus Master pornit → mai departe ca la pornire (falcon, FWSEC-FRTS din nou) |
| **PCIe Base Spec** (§ Function Level Reset; cunoaștere publică; Linux `pcie_flr` face la fel — GPL, nimic citit în cod) | bitul 28 din Device Capabilities = FLR anunțat; înainte, așteptarea „Transactions Pending”; după declanșare, **nicio cerere de configurare mai devreme de 100 ms**; funcția poate răspunde „încă nu” (Vendor ID `0x0001`, CRS) până e gata; registrele funcției revin la valorile de pornire (BAR-uri, Command, MSI) |

**Concluzia rămâne cea din C-142:** ordinea e aceeași în toate trei; că FLR coboară WPR2 pe GA104 e **DEDUS (~80 %)** — tinygrad se bazează pe asta
exact în cazul nostru, RM așteaptă VBIOS după FLR (deci placa se reinițializează), dar nicio sursă nu scrie „FLR coboară WPR2”. Se dovedește doar pe placă.
Diferență față de C-142: RM **rescrie configurația înainte** de așteptarea GFW (are nevoie de BAR0 ca s-o citească) — am urmat ordinea asta.

## 2. Ce s-a scris (cod nou, doar adăugări)

- `gpu_pci::flr` — `FlrPermit` (permis explicit), `save_config` (doar citiri: antetul 0x00–0x3c, registrele de control PCIe, MSI),
  `function_level_reset` (pașii de mai sus, fiecare scriere în jurnal cu valoarea dinainte și după; rescrie doar ce diferă, BAR-urile cu Command = 0,
  Command la urmă; recitește tot și compară). Nu scrie niciun registru BAR0. `#![forbid(unsafe_code)]` ca tot crate-ul.
- `nvidia_ga104::flr_reset` — `reset_if_wpr2_up`: WPR2 citit → dacă 0, nimic → altfel FLR → GFW boot (pasul nostru I2, doar citiri) → WPR2 recitit;
  `FlrVerdict` (8 rezultate, fiecare cu `meaning()` și `chain_may_run()`), rândul `kf` și cititorul `parse_row`. Trăsătura `FlrBoard` dă pe rând
  BAR0 și configurația PCI.
- `gpu-sim`: `SimConfig::with_flr()`, FLR-ul modelat în configurație (registrele la valorile de pornire, 3 citiri „încă nu”, defecte: nu mai răspunde,
  tranzacții blocate); `FakeGsp::sim_flr()` (WPR2 → 0, falcon-urile ca la pornire, GSP-RM dispărut, GFW în lucru 20 de citiri) și defectele
  `SimFault::Wpr2SurvivesFlr`, `SimFault::GfwNeverAfterFlr`.

## 3. Testele (T138, SIMULARE)

| Test | Ce arată | Rezultat |
|---|---|---|
| T138a | după I6 + I7 (WPR2 ridicat): FLR → GFW → WPR2 = 0 → **I6 + I7 trec a doua oară în aceeași „pornire”**; configurația identică cu cea salvată; BAR-urile nescrise cu decodarea pornită; WPR2 niciodată scris de noi; rândul `kf` dus-întors | PASS (SIMULARE) |
| T138b | WPR2 deja 0: **nicio scriere** (nici în configurație, nici în BAR0) | PASS (SIMULARE) |
| T138c | WPR2 rămâne după FLR: `wpr2-inca-ridicat`, un singur FLR, fără a doua încercare | PASS (SIMULARE) |
| T138d | FLR neanunțat (nimic scris), placa nu mai răspunde, BAR care refuză rescrierea, GFW care nu se termină, tranzacții blocate (doar notă) | PASS (SIMULARE) |

## 4. Ce rămâne

- Proba pe placă o face OS-ul (C-152 punctul 1, aprobată în ciclul automat). Până atunci „FLR coboară WPR2” rămâne DEDUS.
- MSI-X: placa noastră nu-l folosește în lanț (fără întreruperi); nu salvăm tabelul MSI-X (RM salvează doar măștile lui).
- Funcția audio a plăcii (`b3:00.1`): FLR resetează doar funcția 0; RM tratează la fel. Nimic de făcut.
