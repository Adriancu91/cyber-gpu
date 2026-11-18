# C-142 — #513 pe placă: ce cere (limite), cum măsurăm viteza, și resetul plăcii fără tăierea curentului (FLR)

*Sesiunea Cyber-GPU 113, 18 noiembrie 2026, ~19:25–20:00 (ora României), în cloud, FĂRĂ placă. Tot ce e aici e **SIMULARE** sau **citire de surse**.
Nimic nu a rulat pe RTX 3070 Ti în această sesiune.*

## 1. Forma lui #513 în `gpu-sim` (C-142 pasul 1)

Forma, din hub `integrare/kronos/MODEL.md`: 12 straturi, 12 capete, n_emb 768, context 512, vocabular 16 384, cap legat. Greutățile din test sunt
**generate** (nu ale lui #513: acelea stau pe stick, nu în repo). Testul: `crates/nvidia-ga104/tests/sim_k513.rs`.

| Ce s-a verificat (SIMULARE) | Rezultat |
|---|---|
| zona de comenzi 32 KiB | 4 poziții trimise, câte o sonerie de **196 de lansări** (195 ale simbolului + copierea logit-urilor în RAM) |
| felul de trimitere | `EachLaunch` = invalidare înaintea fiecărei lansări (cel care a trecut pe placă la `os-306`): **9 440 B** de comenzi pe simbol |
| urcarea greutăților | prin arena din RAM, **14 bucăți** (13 × 8 MiB + rest), câte o lansare `copy_b128` (I11) și o sonerie pe bucată; tabela de simboluri, tabela de poziții și ultimul strat recitite din VRAM = identice |
| calculul | 3 simboluri ale textului „Ana are mere.” + 2 generate, fiecare comparat cu procesorul (`step_cpu`): același simbol, abatere ≤ 1e-3 |
| contoarele de viteză | umplute ca în OS, scrise în jurnal (`kv …`) și citite înapoi |

## 2. Limitele reale (calculate din forme, `kronos_speed::ModelLimits::k513()`)

| Limita | #513 | Ce are OS-ul azi (`os-306`) | Concluzie |
|---|---|---|---|
| lansări pe simbol | 196 | — | — |
| comenzi pe simbol | 7 872 B (o invalidare), **9 440 B** (invalidare la fiecare lansare), 712 B (pe straturi) | zona: 16 KiB | **16 KiB nu ajung** (verificarea OS-ului cere două simboluri: 18 880 B) → **32 KiB** (64 KiB pentru 24 de straturi) — e deja în C-141 |
| pasul valorilor înregistrărilor pe sonerie | ≥ 197 (`0xc5`) | `STRIDE = 0x100` | ajunge |
| valorile urcării | `0xc000 + 0x100·k`, k < 14 | până la `TOKEN_FIRST = 0x10000` | ajunge (ar ajunge până la 64 de bucăți = 512 MiB) |
| VRAM | **142,2 MiB** = greutăți + lucru 105,6 + cache-uri 36,0 (12 straturi × 2 × 512 poziții × 768 × 4 B) + lansări 0,1 + tabele 0,4 | placa are 8 GiB; OS-ul verifică `vram_free()` înainte de urcare, cu 4 MiB rezervă | ajunge cu marjă mare |
| context | 512 poziții (`MAX_POS`) | — | textul + simbolurile generate ≤ 512 |
| bucăți de urcare | 14 × ≤ 8 MiB | arena I11 de 2 × 8 MiB | ajunge |

**Pe partea driverului nu mai trebuie nimic nou** pentru #513: același cod care a trecut I14 pe placă, cu zona de 32 KiB. Singurul lucru nou e ce
măsurăm (punctul 3).

### Timpul — ce știm și ce NU știm

- **Ceasul plăcii pentru #513: NECUNOSCUT.** Modelul din `gpu-sim` socotește 250 µs pe lansare — un număr de test, nu al plăcii. Jurnalele `os-305/306`
  nu au cifre de timp ale lansărilor (doar „ok”). De asta contoarele de la punctul 3.
- **Ce se vede din jurnalul `os-306`** (`gpu-jurnal-009.txt`, timpii rândurilor): în faza textelor, un simbol al modelului de control a luat **~0,27 s**
  de la o sonerie la următoarea, cu **15 rânduri de jurnal** pe simbol; un rând de jurnal costă **~17–23 ms** (scris pe stick). Deci la modelul de control
  timpul e aproape tot **scrierea jurnalului**, nu placa.
- **Estimare pentru #513 (DEDUS, nemăsurat):** tot o sonerie pe simbol ⇒ tot ~15 rânduri ⇒ **~0,3 s pe simbol** + timpul plăcii (necunoscut; greutățile,
  ~106 MiB, se citesc o dată pe simbol) + verificarea pe procesor a fiecărui simbol (OS-ul o face; pe procesor, ~85 milioane de înmulțiri pe simbol).
  Urcarea: 14 bucăți × ~16 rânduri ≈ **4–5 s**, plus citirea fișierului de 106 MiB de pe stick (necunoscut).
- Dacă Adrian vrea mai repede: jurnalul pe fiecare simbol poate fi scurtat (decizia OS-ului și a lui Adrian, regula D-013 cere scrierile în registre în
  jurnal; nu o schimbăm noi).

## 3. Contoarele de viteză pentru `kronos --gpu` (C-142 pasul 2, `kronos_speed::Speed`)

- **Ceasul plăcii** (marcajele de timp ale înregistrărilor, ce a făcut placa) e ținut **separat** de **ceasul procesorului** (cât așteaptă omul: driverul,
  jurnalul, ecranul). **Urcarea greutăților** e numărată separat de simboluri.
- Pe poziție: timpul lansărilor (suma), timpul de la prima la ultima înregistrare, timpul procesorului. La final: mediane, cel mai lent, simboluri/s
  (placă și procesor), MB/s la urcare (placă și procesor).
- Rânduri în jurnal: `valoare kv … = 0x…` (microsecunde, 32 de biți, `ffffffff` = fără ceas). Cu `per_token = false`: **14 rânduri** oricât de lung e textul
  (5 urcare + 9 rezumat); cu `true` încă 4 pe poziție. Cititorul: `kronos_speed::parse_journal`.
- O valoare lipsă nu se ghicește: fără ceasul procesorului rândul spune `ffffffff`, iar ecranul „?”.

## 4. Resetul plăcii fără tăierea curentului: FLR față de Booter Unload (C-142 pasul 3, regula „trei surse”)

**Faptul de pornire (K-2, 18 nov 18:04, workstation):** placa anunță **FLR da** (Function Level Reset), `No_Soft_Reset = 1` (D3hot → D0 nu resetează).
În studiul C-103 (11 nov) scrisesem „plăcile GeForce nu anunță FLR, probabil NU” — **greșit pentru RTX 3070 Ti**: o anunță.

### Ce fac cele trei surse când WPR2 e ridicat

| Sursa (citită ca specificație, nimic copiat) | La descărcare (după lucru) | La pornire, dacă găsește WPR2 ridicat |
|---|---|---|
| **driverul NVIDIA 570.144** (MIT) | reset GSP → FWSEC-SB → **Booter Unload** (`kgspExecuteBooterUnloadIfNeeded`); pe Linux, dacă RM cere (`PDB_PROP_GPU_TRIGGER_PCIE_FLR`), modulul face **și FLR** după oprirea plăcii (`nv.c`, `os_pci_trigger_flr` → `pcie_reset_flr`) | refuză: „unexpected WPR2 already up … may need to be reset” (`_kgspBootGspRm`) |
| **resetul complet al cipului în RM** (`kbifDoFullChipReset_GA100` → `kbifDoFunctionLevelReset_TU102`) | — | salvează configurația PCI, **declanșează FLR**, așteaptă 100 ms, rescrie configurația, **așteaptă ca VBIOS să termine iar inițializarea (GFW boot)**, verifică ID-ul NVIDIA. Deci după FLR placa își reface singură pornirea, ca la curent pus |
| **tinygrad** (MIT, ramura principală, `8ffd4c8`, 18 nov 2026; `runtime/support/nv/nvdev.py`) | `UNLOADING_GUEST_DRIVER` doar | **exact cazul nostru:** „WPR2 is up. Issuing a full reset.” → oprește Bus Master → **reset PCI** al funcției (prin Linux; pentru NVIDIA nu forțează resetul de magistrală, deci Linux alege FLR când e anunțat) → 100 ms → așteaptă GFW boot (`SECURE_SCRATCH_GROUP_05` & 0xff = 0xff) → FWSEC-FRTS din nou și merge mai departe. Pe plăci GA10x/AD10x |
| **nouveau** (GPL, doar citit, `nvkm/subdev/gsp/tu102.c`) | reset falcon → **FWSEC-SB** → **Booter Unload** → verifică WPR2 = 0 | — |
| **nova-core** (GPL, doar citit, `gsp/hal/tu102.rs`, `gsp/boot.rs`) | `UNLOADING_GUEST_DRIVER` → **FWSEC-SB** → **Booter Unload** (mailbox 0xFF/0xFF, mailbox0 = 0, WPR2 = 0) | refuză: „WPR2 region already exists - GPU needs to be reset to proceed” (resetul îl face sistemul) |

### Ce reiese

1. **FLR coboară WPR2 pe GA10x — DEDUS, nu scris explicit nicăieri.** tinygrad se bazează pe asta (reset PCI, apoi FWSEC-FRTS care ar pica dacă WPR2
   ar fi rămas); RM folosește FLR ca „reset complet al cipului” și după el așteaptă ca VBIOS să refacă inițializarea. Nicio sursă nu are textul „FLR
   coboară WPR2”. **Cât de sigur: ~80 %.** Se dovedește doar pe placă (proba de mai jos).
2. **Booter Unload** e calea „curată” a tuturor celor care pornesc GSP-RM (RM, nouveau, nova), dar: (a) cere un GSP-RM oprit frumos sau măcar
   falcon-ul resetat; (b) **toți trei rulează FWSEC-SB înainte** — noi, în C-103, am lăsat FWSEC-SB deoparte („doar dacă trebuie”): diferență de corectat
   dacă alegem calea asta; (c) rulează încă un program semnat NVIDIA pe SEC2; (d) n-a rulat niciodată pe placa noastră (`gpu recupereaza`, P6-74).
3. **FLR nu depinde de starea lui GSP-RM** (oricât de stricat ar fi lanțul) și nu rulează niciun program NVIDIA în plus: e un bit în configurația PCI a
   plăcii însăși (Device Control, bitul 15), standard PCIe. Costul: tot spațiul de configurare al plăcii (BAR-uri, Command, MSI) se pierde și **OS-ul trebuie
   să-l salveze înainte și să-l rescrie după**, ca RM și Linux. Dacă ecranul OS-ului e pe placa asta (GOP), ecranul se pierde — oricum îngheață după I8.

### Propunerea

**Varianta 1 (propusă): FLR la ÎNCEPUTUL lanțului, doar când WPR2 e ridicat** — ca tinygrad. Merge după orice s-a întâmplat înainte (lanț oprit la jumătate,
repornire caldă, Windows), nu depinde de GSP-RM. Pașii: Bus Master oprit → configurația salvată → FLR → 100 ms → configurația rescrisă → așteptare GFW boot
(pasul nostru I2, doar citiri) → **WPR2 recitit: trebuie 0** → abia apoi I3…I14. Dacă WPR2 tot e ridicat: oprire curată, „trebuie pornire rece (Adrian)”
(exact regula din C-143 punctul 3), fără nicio încercare în plus.

**Varianta 2 (rezervă): Booter Unload la sfârșitul lanțului**, cu FWSEC-SB înainte (ca RM, nouveau și nova). Mai multe programe semnate, depinde de starea lui
GSP-RM; o păstrăm pentru cazul în care FLR nu coboară WPR2.

**Proba cu Adrian (o singură fereastră, cu „DA”):** după un `gpu lant --i9 c135` reușit (WPR2 ridicat, ca la `os-306`): (1) `gpu identifica` + citirea WPR2
(doar citiri; așteptat: ridicat); (2) **FLR** (comandă nouă a OS-ului, cu „DA”); (3) citiri: ID-ul plăcii, GFW boot, **WPR2** (așteptat 0); (4) dacă e 0:
`gpu lant --i9 c135` din nou, **în aceeași pornire, fără cablu scos** — dacă trece și a doua oară, ciclul automat C-143 nu mai are nevoie de pornire rece.
Ce înseamnă fiecare rezultat: WPR2 = 0 și lanțul trece → varianta 1 intră în ciclu; WPR2 = 0 dar I6 pică → altceva rămâne din pornirea veche (se citește
jurnalul); WPR2 tot ridicat → varianta 2 (Booter Unload cu FWSEC-SB), apoi oprirea S5 / Auto On din BIOS.

Codul FLR stă în OS (scriere în configurația PCI) — **nu l-am scris**: C-142 cere doar analiză; îl facem (noi partea de citire și ordinea, OS-ul scrierea)
după „da”-ul lui Adrian pe această propunere.
