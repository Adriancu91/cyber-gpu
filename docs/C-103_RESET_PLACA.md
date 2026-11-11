# C-103 — Placa readusă la starea de pornire FĂRĂ repornirea calculatorului (studiu + ce s-a făcut în SIMULARE)

*Sesiunea Cyber-GPU 85, 11 noiembrie 2026, în cloud, fără placă. Pasul (1) al comenzii: studiu, doar citiri. Pasul (2): implementare în SIMULARE. Nimic testat pe RTX 3070 Ti.*

## 1. Problema, exact

După orice lanț `gpu lant` care a trecut de I6, placa rămâne cu **WPR2 ridicat** (`NV_PFB_PRI_MMU_WPR2_ADDR_HI ≠ 0`): regiunea protejată pe care FWSEC-FRTS a creat-o
pentru GSP-RM. Pasul nostru I6 refuză să ruleze FWSEC-FRTS a doua oară peste ea („I6: WPR2 deja ridicat (FWSEC a rulat dupa pornire): trebuie pornire rece”), la fel ca
driverul NVIDIA (`_kgspBootGspRm`: „unexpected WPR2 already up, cannot proceed with booting GSP — the GPU is likely in a bad state and may need to be reset”).
Pe workstation, Adrian scoate cablul 30 s: WPR2 coboară doar când placa pierde curentul. **O încercare = o pornire rece cu om la butoane** — asta vrea C-103 să elimine.

## 2. Căile publice de a coborî WPR2 — ce spun sursele

### 2a. Booter Unload (programul semnat NVIDIA) — calea driverului NVIDIA și a nouveau ✔ aleasă

- Driverul NVIDIA 570.144 (MIT, citit ca specificație) **coboară WPR2 fără niciun reset PCIe**, la descărcarea driverului (`kgspTeardown_TU102`): resetul falcon-ului GSP,
  FWSEC-SB („put back PreOsApps”), apoi **`kgspExecuteBooterUnloadIfNeeded_TU102`**: dacă WPR2 e ridicat → reset SEC2 → programul HS `booter_unload` pe SEC2 cu
  **mailbox0 = 0xFF, mailbox1 = 0xFF** (descărcare normală; la suspend primește adresa `GspFwSRMeta`, la GC6 `0xdeaddead`) → la oprire mailbox0 trebuie să fie 0 →
  **`WPR2_ADDR_HI` trebuie să citească 0** („WPR2 is still up” e eroare). Dacă WPR2 nu e ridicat, programul nu se rulează deloc.
- Înainte de teardown, driverul îi spune lui GSP-RM `UNLOADING_GUEST_DRIVER` și așteaptă ca procesorul lui să se suspende (GSP `MAILBOX0 == 0x80000000`,
  `kgspWaitForProcessorSuspend_TU102`). În cazul nostru GSP-RM a eșuat la inițializare, deci RPC-ul nu are rost: facem direct resetul falcon-ului GSP (cum face RM înainte de FWSEC-SB).
- nouveau (din cunoaștere generală, doar pentru comparație): la `fini` rulează același `booter_unload` și verifică WPR2; la pornire, dacă găsește WPR2 ridicat de la o
  încărcare anterioară, îl coboară tot cu Booter Unload înainte de FWSEC-FRTS.
- Fișierul e **deja pe stick-ul lui Adrian**: `nvidia/ga102/gsp/booter_unload-570.144.bin` (unul din cele 4 fișiere cerute de I4, `firmware.rs`, suma SHA-256 verificată).
  Se încarcă exact ca `booter_load` (același încărcător DMA `hs_load`, aceleași registre SEC2); singura diferență e programul și cele două cutii poștale.
- FWSEC-SB nu-l rulăm: pune la loc „aplicațiile pre-OS” (afișaj) de care nu depinde nimic de-al nostru; un program semnat în plus, fără nevoie, lărgește suprafața.
  Dacă pe placă Booter Unload refuză fără FWSEC-SB înainte (mailbox0 ≠ 0), îl adăugăm atunci — e același fișier FWSEC din VBIOS, cu altă comandă (`FWSECLIC_CMD_ID_SB`).

**Ce s-a implementat (SIMULARE, `364ece2`):** `nvidia_ga104::gsp::falcon::booter_unload[_dma]` — ordinea de mai sus, cu jurnal (WPR2 înainte, `pas C-103 Booter Unload:
WPR2 coborat: ok`, sau eroarea numită: mailbox0 ≠ 0 / „WPR2 inca ridicat dupa Booter Unload”). Doar ferestrele GSP/SEC2, prin politica de scriere (`GuardedBar0`);
registrele WPR2 nu se scriu niciodată de noi. Modelul `gpu-sim::fake_gsp` are `SimUcode::BooterUnload` (efectul vizibil: WPR2 → 0, GSP-RM dispărut) și `SimFault::Wpr2Stuck`.
Testele T118: după I6 + I7, unload → WPR2 = 0 → **I6 rulează din nou în aceeași „pornire”**; WPR2 jos → sărit, 0 scrieri; program greșit / WPR2 rămas → erori numite.

### 2b. Resetul PCIe al funcției (FLR) — de citit pe placă, probabil NU

- PCIe Base Spec: Device Capabilities bit 28 = „Function Level Reset Capability”; se declanșează cu Device Control bit 15. Plăcile GeForce de consum **nu anunță FLR**
  (cunoaștere generală: utilizatorii de virtualizare folosesc resetul de magistrală tocmai de aceea); pe RTX 3070 Ti nu știm — **se citește**.
- Implementat (doar citire): `gpu_pci::caps::reset_capabilities` → `flr_capable`, `flr_initiated`, `pm_no_soft_reset`, `power_state`, `pme_from_d3cold` + `summary()`.
  OS-ul îl poate pune în `gpu` / `raport` (T119).
- Chiar dacă ar exista: după FLR placa e „ca scoasă din priză” doar din punctul de vedere al funcției PCIe; dacă GFW (devinit din VBIOS) rulează din nou și dacă WPR2
  coboară nu scrie nicăieri public — ar trebui **observat** pe placă (`gpu gfw` + citirea WPR2 după). Risc: BAR-urile și Command se pierd → OS-ul trebuie să le rescrie.

### 2c. Secondary Bus Reset (SBR) pe puntea-părinte — posibil, mai dur, decizia OS-ului

- Bitul 6 din Bridge Control (offset `0x3E`) al **root-port-ului** din fața plăcii (placa e `b3:00.0`; root-port-urile Skylake-SP de pe `b2:00.0/02.0/03.0` —
  care din ele are bus secundar `b3` se citește din `0x19` al punții). Resetează tot ce e pe magistrală (placa + funcția ei audio `b3:00.1`), link-ul se reantrenează
  (~100 ms+), **tot spațiul de configurare al plăcii se pierde** (BAR-uri, Command, MSI) și trebuie salvat/rescris de OS. E ce fac Linux (`pci_reset_bus`) și vfio când nu există FLR.
- Dacă GFW rerulează după SBR și WPR2 coboară: de observat pe placă. Dacă nu coboară nici așa, rămâne doar pierderea curentului (2d).
- Scrieri în spațiul de configurare al **altui dispozitiv** (puntea) — în afara a tot ce a scris până acum proiectul; rămâne la OS, cu „DA”-ul lui Adrian, abia după ce 2a e verificată pe placă.

### 2d. Stări de alimentare

- **D3hot → D0**: resetează funcția numai dacă `No_Soft_Reset` (PMCSR bit 3) e 0. Plăcile NVIDIA au de regulă `No_Soft_Reset = 1` (păstrează starea) — **se citește** (2b).
- **D3cold** (fără curent pe slot): pe desktop nu există comandă de platformă pentru slotul PCIe (nu e laptop cu Optimus/`_PS3`). Varianta (b) a lui Adrian — oprire S5 +
  „Power On By RTC” din BIOS — e singura „pornire rece fără om” dacă 2a nu ajunge; de verificat pe workstation dacă în S5 placa pierde curentul ca la cablul scos
  (multe plăci de bază țin +3.3 V aux pe slot în S5 → WPR2 ar putea rămâne). Varianta (c), priza inteligentă, e decizia lui Adrian.

## 3. Pașii pentru fereastra pe workstation (doar cu Adrian, doar cu „DA”)

1. **Citire, fără scriere**: `reset_capabilities` → în `raport`: FLR da/nu, `No_Soft_Reset`, D-state. (Poate intra în turul de citire C-54: 0 scrieri.)
2. **Proba `gpu recupereaza`** după un lanț oprit la I8 (deci cu WPR2 ridicat și GSP-RM în stare necunoscută): reset GSP, reset SEC2, Booter Unload (0xFF, 0xFF),
   mailbox0, WPR2 citit → așteptat 0. Jurnalul spune fiecare pas. Dacă WPR2 = 0: **I6 → I7 → I8 se pot relua în aceeași pornire** (a doua încercare fără cablu scos).
3. Dacă Booter Unload dă eroare în mailbox0: se notează codul (antetele publice nu-l documentează; nu analizăm firmware-ul) și se încearcă varianta cu FWSEC-SB înainte.
4. Doar dacă 2 nu merge: FLR (dacă e anunțat) sau SBR, din OS, cu rescrierea configurației — comandă separată, alt „DA”.

## 4. Riscuri

| Risc | Măsură |
|---|---|
| Booter Unload rulat pe un GSP-RM încă viu | întâi resetul falcon-ului GSP (ca RM); GSP-RM nostru e oricum oprit la I8 |
| WPR2 coboară, dar placa ține în memoria video resturi (cozi, heap GSP-RM) | FWSEC-FRTS scrie din nou regiunea; GSP-RM pornește cu memorie proaspătă din WPR meta; nu depindem de conținutul vechi |
| Paginile din RAM-ul calculatorului mapate pentru placă | aceleași reguli ca `recovery.rs`: Bus Master oprit înainte, paginile vechi în carantină până la repornire |
| Programul semnat face altceva decât modelul SIM | modelul reproduce doar efectul vizibil (WPR2 → 0); diferențele se văd în jurnal (mailbox0, WPR2 după) și opresc lanțul |
