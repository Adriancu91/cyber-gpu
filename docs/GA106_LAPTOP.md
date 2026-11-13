# C-95 / C-116 — Laptopul ASUS TUF F15 (RTX 3060 Laptop, GA106) ca al doilea calculator de test

> **RELUAT de Adrian (C-116, 12 nov 20:57):** „ne mutăm cu testele aici (pe laptop) și vor fi mult mai rapide; apoi facem GPU pentru placa asta”.
> Istoric: parcat pe 10 nov 07:28 (C-95). De la sesiunea 95 (12 nov, noaptea) codul GA106 doar-citire e **înapoi în proiect** (`6623d3c`), plus
> „trezirea minimă” (`ccec8f9`). **Lanțul cu scrieri în registre GPU (I6+) rămâne doar pentru GA104** până la DA-ul separat al lui Adrian (C-116 pasul 3).

*Scris 10 nov 2026 (sesiunea 77), completat 12 nov 2026 (sesiunea 95) și 13 nov 2026 (sesiunea 98: primul jurnal de pe laptop, §0c), în cloud, FĂRĂ placă.*

## 0. Ce a arătat laptopul pe 12 nov 2026 (OBSERVAT de OS, os-295, `gpu identifica`; hub `rezultate/2026-11-11-laptop-asus-tuf/`)

| Ce | Valoare | Ce înseamnă pentru noi |
|---|---|---|
| Placa NVIDIA | **`10de:2520`** la 01:00.0, „RTX 3060 Laptop GPU”, audio HDMI `10de:228e` | ID în tabelul `GA106_MODELS`; cipul se confirmă din BOOT_0 (`0x176`), încă necitit |
| Stare de energie | **D0** | nu e adormită de laptop; nu avem de atins ACPI (`_PS0`) |
| **Command** | **`0x0000`** — Memory Space, I/O și Bus Master toate oprite de firmware | **BAR0 nu răspunde**: BOOT_0 / GFW / VBIOS nu se pot citi până nu pornim bitul Memory Space (§6). Pe workstation firmware-ul îl lasă pornit, de aceea n-a apărut acolo |
| BAR0 | 16 MiB @ `0x85000000` (mem32, alocat) | fereastra de registre, ca pe GA104 |
| BAR1 | **8 GiB** (Resizable BAR pornit de firmware) | pe workstation e 256 MiB; `fb_layout` citește mărimea VRAM din placă, nu din BAR1 |
| BAR3 | 32 MiB | ca pe GA104 |
| Driverul nostru (`49e6685`, în os-295) | „dispozitiv nesuportat 10de:2520”, registre necitite, **zero scrieri** | protecția a mers; de la `6623d3c` identificarea GA106 există, dar doar prin funcțiile `_ampere` |
| Altele | Intel UHD `8086:9a60` (ecranul e pe ea: hibrid), VMD `8086:9a0b` (discul NVMe ascuns), Wi-Fi `8086:43f0` (fără driver), i7-11800H, 64 GB | ecranul **nu** e pe placa NVIDIA → GSP-RM n-are ce afișaj să preia (ipoteza din os-295 §3) |

### 0b. Ce a raportat Windows (OBSERVAT de unealta UHL a lui Adrian, 12 nov 22:14 și 23:03; NVML/CUDA, doar citire, fără serii) — `gpu_sim::laptop`

| Ce | Valoare | Ce înseamnă pentru noi |
|---|---|---|
| Model laptop | ASUS TUF Gaming F15 **FX506HM**, BIOS FX506HM.318 (iun. 2025), Secure Boot oprit; **fără MUX** — OBSERVAT (raportul UHL de energie, 13 nov 07:11: funcțiile ASUS de MUX/eGPU/limite de putere „nesuportate”; dGPU activ), nu mai e dedus din anul modelului | ecranul rămâne pe Intel; placa NVIDIA e doar de calcul — exact ce ne trebuie |
| Placa | `10de:2520` **rev A1, subsistem 1043:16F2**, PCIe Gen4 **x8 activ** (x16 max), payload 256 B, **D3cold suportat**, ACPI `\_SB.PC00.PEG1.PEGP` | subsistemul și revizia intră în modelul `ga106_laptop_asus`; D3cold = laptopul poate tăia curentul plăcii (§3) — se vede în `pci.stare` |
| **VBIOS** | **`94.06.1E.00.23`** (imaginea nu e citibilă din Windows; UHL n-o citește — corect) | de la `835f24d` turul scrie **`vbios.versiune`** din jetonul BIT 'i' (§5): pe laptop trebuie să dea exact acest șir; pe workstation `94.04.65.00.3e` |
| Cip | GA106M, sm_86, **30 SM** (GA104: 48), L2 3 MiB, **6 GiB / 192 biți**, max 2100 MHz | aceleași programe SASS; `launch_cost`/`kronos_gen` au mai puține SM-uri la dispoziție; limita noastră pe laptop sub ~5,5 GB (pasul 4) |
| BAR1 | 8 GiB @ **`0x6000000000`** (Resizable BAR) | baza era SIM în sesiunea 95; acum e din raport |
| Putere | limită implicită 80 W, aplicată 95 W, repaus 20–26 W, 58–62 °C, P0; raportul de energie (13 nov 07:11): **25,6 W și 61 °C în repaus** (NVML), ventilatorul GPU 3100 RPM — condus de EC/firmware (ASUS DSTS), nu de zona ACPI; zona `\_TZ.THRM` ~79 °C (prag critic 100 °C) | doar informativ; driverul nu atinge nici limitele de putere, nici ventilatoarele (rămân la EC, fără scrieri) |

Ce rămâne SIM în model: baza BAR3, poziția capabilităților, I/O-ul, **conținutul ROM-ului** (nu copiem VBIOS-ul; versiunea din jetonul 'i' e doar șirul raportat de Windows).

### 0c. Ce a arătat PRIMUL TUR DE CITIRE pe laptop (OBSERVAT de driverul nostru, 13 nov 2026 ~07:35, `os-301` = cyber-gpu `835f24d`, `gpu tur --trezire`; hub `rezultate/2026-11-13-laptop-asus-os301/`: `gpu-jurnal-003.txt` + poza `gpu-tur-trezire.jpg`) — `gpu_sim::observed_laptop`

**Rezultat: TRECE, 45 de fapte, 2 scrieri în registrul Command al PCI (pornit, apoi repus la `0000`), 0 scrieri în registrele plăcii.** Trezirea minimă (§6) a mers exact cum era scrisă:
`cfg w 0x004: 0x00100000 <- 0x00000002 => 0x00100002`, turul, apoi `=> 0x00100000`. Placa era pe magistrală, în D0 (riscurile §3 pct. 1 și 2 nu s-au arătat).

| Ce | Pe laptop (OBSERVAT) | Pe workstation (GA104) | Ce înseamnă |
|---|---|---|---|
| BOOT_0 / BOOT_42 | `0xb76000a1` / `0x176a1000` → **GA106**, consecvent | `0xb74000a1` / `0x174a1000` | cipul confirmat din registru, nu din nume |
| GFW | progres `0xff` la prima rundă; `0x118128 = 0x8b8f` | la fel | firmware-ul de pornire al plăcii a terminat; aceleași registre la Ampere (nu mai e DEDUS) |
| VBIOS prin PROM | **valid**: fără IFR, **3 imagini** — PCI-AT `10de:2520` 65024 o la 0; FWSEC `10de:2200` 22016 o la 0x0fe00; FWSEC `10de:2200` 415744 o la 0x15400 (sfârșit 0x7ac00) | 4 imagini (are și **EFI** 83968 o; sfârșit 0x88000) | riscul §3 pct. 4 (VBIOS prin ACPI `_ROM`) **nu s-a arătat**; laptopul n-are imagine EFI (ecranul e pe Intel) |
| BIT | la 0x1b0, **17 jetoane, aceleași și în aceeași ordine** ca pe workstation (`0x32v1 … 0x69v2`) | la fel | același generator de VBIOS |
| **`vbios.versiune`** | **`94.06.1e.00.23`** — exact șirul din raportul UHL (Windows/NVML) | `94.04.65.00.3e` așteptat, necitit încă pe placă | **prima comparație automată Windows ↔ Cyber Universe a reușit** (C-117); ordinea octeților jetonului 'i' (§5) e confirmată pe placă |
| FWSEC | PMU la 0x7aafe, aceleași 16 aplicații (0x85 = FWSEC_PROD); descriptor v3 la 0x2c834, antet 0x4ac, **IMEM 0xe100**, DMEM 0x800, 3 semnături, mască 0x0007, motoare 0x0400, ucode_id 9 | descriptor la 0x41208, **IMEM 0xe200** | FWSEC-ul laptopului e altă construcție (puțin mai mică); formatul descriptorului e același, parserul nostru l-a citit fără schimbări |
| siguranța (I3c) | `0x8241e0 = 3` → versiunea 2, semnătura 2 | la fel | aceeași alegere a semnăturii FWSEC |
| VRAM | **6144 MiB** (`0x1183a4 = 0x1800`) | 8192 MiB | limita noastră pe laptop: sub ~5,5 GB (pasul 4) |
| WPR2 | coborât (`0x1fa828 = 0`), pornire rece → `i6.trece = da` | la fel la pornire rece | I6 ar putea porni **dacă** ar avea DA-ul (pasul 3) |
| BAR-uri / legătură | BAR0 `0x85000000`, BAR1 `0x6000000000` (8 GiB), **BAR3 `0x6200000000`**, **BAR5 I/O `0x4000`**, gen4 x8 (max x16) | — | BAR3 și BAR5 erau SIM în sesiunile 95–96; acum observate |
| Stick-ul (OS) | 60 de scrieri, 6,1 ms în medie, max 242 ms; sincronizări ~0 ms | ~1 s/rând pe Kingston (C-113) | pe laptop jurnalul nu încetinește |

**Modelul:** `gpu_sim::observed_laptop::{observed_laptop_config, observed_laptop_bar0, observed_laptop_rom}` ține aceste valori (ca `observed` pentru workstation: registre, poziții,
mărimi, ID-uri — nu conținutul VBIOS-ului, nu firmware). Testul `laptop_journal_003_agrees_with_the_observed_laptop_model` pune jurnalul 003 + faptele din poză față de turul pe model:
**36 la fel, 0 diferit, 1 diferit intenționat (cuvântul PROM), 0 doar pe placă.** Ce a rămas SIM: `observed_laptop::STILL_SIM` (poziția capabilităților, conținutul imaginilor, registrele
GSP/SEC2/FBIF — niciodată citite pe GA106).

## 6. „Trezirea minimă” (C-116 pasul 2) — ce face și ce NU face

- **Ce:** o singură scriere în registrul **Command** al funcției PCI (configurație, nu registre GPU): bitul 1 „Memory Space” pornit; Bus Master și I/O rămân cum erau (oprite).
  Citită înapoi; dacă nu se potrivește, valoarea veche e pusă la loc și eroarea e `VerifyFailed`. La **finalul turului** Command e **repus** la valoarea găsită (a doua scriere),
  și când turul pică — placa rămâne cum a lăsat-o firmware-ul. Cod: `gpu_pci::mem_space::{enable_memory_space, restore_memory_space, MemorySpacePermit}`,
  `nvidia_ga104::read_tour::run_tour_awake`.
- **Înainte de scriere, doar citiri:** placa răspunde (nu 0xffffffff), antet tip 0, **D0**, BAR0 alocat (adresă ≠ 0), fiecare BAR numit de Resizable BAR are adresă.
  Oricare pică → refuz, 0 scrieri. Dacă bitul e deja pornit (workstation-ul) → 0 scrieri.
- **De ce e sigur (trei surse):** e exact ce face Linux `pci_enable_device` pentru orice placă înainte ca nouveau / nova-core / tinygrad să atingă BAR0; PCI Local Bus 3.0 §6.2.2.
  Riscul real al pornirii decodării e un BAR nealocat (ar revendica adresa 0) — de aceea verificările de mai sus.
- **Ce NU face:** nu schimbă starea de energie, nu pornește Bus Master (DMA), nu scrie niciun registru GPU, nu atinge ACPI. Nu e „pornirea plăcii” — e doar deschiderea ferestrei de citire.
- **În OS (propunere, decizia e a sesiunii OS):** `gpu tur` pe laptop → după „DA” (ca la `gpu lant`), doar pe stick-ul de test (`gpu-scrie`, adaptorul de configurație care poate scrie),
  `run_tour_awake(...)`; jurnalul pe stick primește și cele două scrieri `cfg w 0x004: …`. Fără `gpu-scrie`, `run_tour_ampere` refuză curat cu
  `Memory Space oprit de firmware: BAR0 nu raspunde …` — nicio citire din BAR0.
- **Ce căutăm în primul jurnal de pe laptop:** `BOOT_0 = 0x176xxxxx` (GA106 confirmat), `BOOT_42` consecvent, GFW terminat (`GFW_BOOT`), imaginile VBIOS prin PROM (dacă sunt gunoi →
  §3 pct. 4, citirea prin ACPI `_ROM` devine sarcină separată), descriptorul FWSEC + semnăturile, siguranța, mărimea VRAM (~6 GiB), WPR2 (pornire rece: 0).

## Pe scurt (pentru Adrian)

- **Firmware-ul e același.** Pentru GA106, pachetul oficial `linux-firmware` are doar o scurtătură: `nvidia/ga106/gsp -> ../ga102/gsp`.
  Deci aceleași 4 fișiere 570.144, cu aceleași sume SHA-256 ca pe workstation (tabelul din `COMPATIBILITATE_FIRMWARE.md`).
  Pe stick nu se schimbă nimic.
- **Driverul NVIDIA tratează GA106 exact ca GA104** la toți pașii de pornire pe care îi avem (FWSEC, Booter, încărcarea prin DMA, siguranța
  „ucode version”, GSP-RM): în tabelele lui generate, GA102 | GA103 | GA104 | GA106 | GA107 folosesc aceleași funcții.
- **Ce diferă:** numărul cipului (`0x176`, nu `0x174`), ID-ul PCI, VBIOS-ul (al laptopului, cu FWSEC-ul lui), memoria (6 GB, nu 8),
  mai puține unități de calcul. Programele GPU (SASS sm_86) sunt aceleași.
- **Riscurile reale sunt de laptop, nu de cip:** placa poate fi oprită de Windows/ASUS („Eco mode”), alimentarea e controlată de ACPI,
  iar la unele laptopuri VBIOS-ul se citește altfel. Toate se văd la primul test **doar de citire**.
- **Cod (de la sesiunea 95):** recunoașterea GA106 **doar pentru citire** (`level0_ampere`, `run_tour_ampere`) și „trezirea minimă” (§6) — o scriere în
  registrul Command al PCI, ca BAR0 să răspundă, repusă la final. Lanțul `gpu lant` și tot ce scrie în registre GPU refuză în continuare laptopul.
- **De ce nu s-a putut citi nimic din placă pe 12 nov:** firmware-ul laptopului lasă funcția cu Memory Space oprit (Command 0) — nu placa e de vină, nici driverul; §0 și §6.

## 1. Identitatea cipului

| Ce | GA104 (workstation) | GA106 (laptop) | Sursa |
|---|---|---|---|
| `NV_PMC_BOOT_0`, câmpurile arhitectură/implementare | 0x17 / 0x4 → cip `0x174` | 0x17 / 0x6 → cip **`0x176`** | open-gpu-kernel-modules@570.144 `g_chips2halspec_nvoc.c` (MIT) |
| ID PCI „RTX 3060 Laptop GPU” | — | **`0x2520`, `0x2521`, `0x2560`** (GA106) | același repo, `README.md` (tabelul plăcilor) |
| Atenție | același nume „RTX 3060 Laptop GPU” apare și pe GA104 (`0x249c`, `0x24a0`, subsistem 1D05) și pe GA107 (`0x25a0`, `0x25a2`) | | idem |

Concluzie: numele plăcii nu dovedește nimic; **cipul se confirmă din BOOT_0**. Codul face exact asta: un ID PCI din tabelul GA106 cu BOOT_0
care nu spune `0x176` e refuzat (test `ga106_pci_id_with_other_chip_is_refused`).

## 2. Pașii I0–I14 pe GA106

Familia din driverul public NVIDIA 570.144 (`g_kernel_gsp_nvoc.c`, MIT, citit ca specificație):

| Funcția NVIDIA | Ce cipuri o folosesc | Pentru noi |
|---|---|---|
| `kgspGetBinArchiveGspRmBoot_GA102` | GA102, GA103, GA104, **GA106**, GA107 | același GSP-RM / bootloader |
| `kgspGetBinArchiveBooterLoadUcode_GA102` | GA102 … **GA106** … GA107 | același Booter (`booter_load`) |
| `kgspExecuteHsFalcon_GA102` | GA10x + AD10x | același încărcător DMA (I6/I7) |
| `kgspReadUcodeFuseVersion_GA100` | toate în afară de Turing | aceeași siguranță (I3c) |

| Pas | Pe GA106 | Stare |
|---|---|---|
| I0 identificare PCI | ID-uri noi (tabelul `GA106_MODELS`); **`10de:2520` văzut pe laptop** | **OBSERVAT pe laptop (13 nov, os-301)** |
| I0b Memory Space | **oprit de firmware** (Command 0) → trezirea minimă (§6) înainte de orice citire BAR0 | **OBSERVAT: a mers pe laptop** (2 scrieri în Command, repus) |
| I1 BOOT_0/BOOT_42 | cip `0x176` | **OBSERVAT: `0xb76000a1` / `0x176a1000`** |
| I2 GFW_BOOT | aceleași registre la Ampere | **OBSERVAT: `0xff`, 1 rundă** (nu mai e dedus) |
| I3 VBIOS prin PROM | riscul de laptop (§3 pct. 4) **nu s-a arătat** | **OBSERVAT: 3 imagini valide, BIT 17 jetoane, versiunea = Windows** |
| I3b FWSEC | FWSEC-ul laptopului: altă construcție (IMEM 0xe100), același format, același `ucode_id` 9 | **OBSERVAT** (§0c) |
| I3c siguranța | aceeași funcție NVIDIA (GA100) | **OBSERVAT: `0x8241e0 = 3`** (v2/s2, ca GA104) |
| I4 fișierele firmware | **identice** (scurtătura `ga106/gsp -> ga102/gsp`) | confirmat din `linux-firmware` `WHENCE` |
| I5a aranjamentul VRAM | aceeași aritmetică, mărimea citită din placă: **6144 MiB** (OBSERVAT) | aranjamentul: SIMULARE |
| I5b–I8 | același GSP-RM, aceleași mesaje | SIMULARE; scrieri = **pasul 3 C-116, doar cu DA-ul separat** (§7) |
| I8e golden image / I9 | mărimile bufferelor GR vin de la GSP; mai puține GPC/SM | SIMULARE |
| I10+ programe | sm_86 ca GA104 (NAK produce același cod) | SIMULARE |

## 3. Ce înseamnă „placă de laptop” (riscurile)

1. **Placa poate lipsi cu totul de pe magistrala PCI.** La laptopurile ASUS, modul „Eco” din Armoury Crate (sau „GPU mode”) poate opri placa
   NVIDIA din firmware; atunci nici Windows, nici OS-ul nostru nu o văd. **Înainte de test: în Armoury Crate, modul GPU = „Standard”
   (sau „Ultimate”), nu „Eco”.** (DEDUS din documentația ASUS / driverul Linux `asus-wmi` „dgpu_disable”; nu e verificat pe laptopul lui Adrian.)
2. **Alimentarea prin ACPI.** Pe laptopurile Optimus, placa NVIDIA e pornită la boot de firmware (UEFI) și abia sistemul de operare
   (Windows/Linux) o adoarme mai târziu (D3cold, prin metodele ACPI `_PR3`/`_PS3`). OS-ul nostru nu adoarme nimic, deci ne așteptăm ca
   placa să fie în D0. Codul verifică: dacă placa nu e în D0 → refuz, fără nicio citire din BAR0; dacă BOOT_0 = `0xffffffff` → „placa nu răspunde”.
   Nu încercăm `_PS0`/`_ON` (ar însemna scrieri și un interpretor ACPI). **Confirmat la primul test (13 nov): placa era în D0, pe magistrală.**
3. **Ecranul.** Fără comutator MUX (sau cu MUX pe „hibrid”), ecranul laptopului e ținut de placa Intel, deci placa NVIDIA nu are imaginea
   ecranului în memoria ei (mai simplu decât pe workstation). Cu MUX pe „doar NVIDIA” ecranul e pe placa NVIDIA, ca la workstation.
   TUF F15 din 2021 (FX506H*) nu are MUX; cel din 2022 (FX507Z*) are. Modelul exact îl spune eticheta de sub laptop sau `hw lista`.
4. **VBIOS-ul.** La plăcile Turing/Ampere pornirea (GFW) rulează din memoria flash a plăcii, deci VBIOS-ul ar trebui să se citească prin
   fereastra PROM, ca pe workstation. La unele laptopuri însă VBIOS-ul e dat și prin ACPI (`_ROM`); dacă `gpu vbios` citește gunoi pe
   laptop, aceasta e cauza probabilă și citirea prin ACPI devine o sarcină separată. **Nu s-a întâmplat (13 nov): VBIOS-ul prin PROM e valid (§0c); sarcina `_ROM` nu e necesară.**
5. **Memoria:** 6 GB (192 biți). Regula din proiect: sub ~5,5 GB pentru Kronos pe laptop.
6. **Laptopul e mai scump de reparat decât o placă desktop.** De aceea pasul 3 (scrieri) se discută separat, după jurnalul de citire.

## 4. Cod

Sesiunea 77 scrisese recunoașterea GA106 doar pentru citire (`256a1f2`), retrasă imediat la parcarea C-95 (`24dc19c`, istoria nu s-a rescris).
**Sesiunea 95 (C-116) a repus-o, adaptată** (`6623d3c`): `ids::{GA106_MODELS, Family, lookup_ampere}`, `boot::CHIP_GA106`, `probe::{probe_config_ampere,
probe_mmio_family, level0_ampere, MEMORY_SPACE_OFF}`, `read_tour::run_tour_ampere`, profilul `SimConfig::ga106_laptop_asus` / `SimMmio::ga106_bar0`;
plus trezirea minimă (`ccec8f9`): `gpu_pci::mem_space`, `read_tour::{run_tour_awake, Wake}`. Toate adăugări; `probe::level0`, `probe_config`,
`run_tour` și lanțul se poartă exact ca înainte (refuză laptopul). Nota pentru OS: `integration/cyber-universe/C-116_ga106-trezire.md`.

## 5. Ce ar trebui să scrie jurnalul de citire de pe laptop

`gpu tur` cu `run_tour_awake` (trezirea minimă + `run_tour_ampere`) ar da aproape tot: ID-ul PCI și subsistemul, starea D0/D3, BAR-urile și legătura PCIe, BOOT_0/BOOT_42,
starea GFW, imaginile VBIOS (tipuri, mărimi, ID-uri), jetoanele BIT, descriptorul FWSEC și semnăturile, siguranța, mărimea VRAM, WPR2.
În plus ar ajuta (partea OS): modelul laptopului din SMBIOS (`hw lista`), prezența plăcii Intel, și dacă există placa NVIDIA pe magistrală
(dacă lipsește → modul „Eco”).

**Versiunea VBIOS (de la `835f24d`, sesiunea 96):** turul scrie faptul **`vbios.versiune`** citit din jetonul BIT 'i' (0x69) — 5 octeți la offset-ul jetonului,
afișați `major.cip.minor.micro.patch` (primii 4 ca un întreg little-endian, al 5-lea „patch”), cum face nouveau `nvkm/subdev/bios/base.c` când tipărește
`version %02x.%02x.%02x.%02x.%02x` (citit ca specificație; din memorie — sursa n-a putut fi recitită din cloud). **Confirmat pe placă pe 13 nov (§0c): pe laptop a ieșit exact `94.06.1e.00.23`.** E același șir pe care îl arată Windows / NVML,
deci **prima comparație automată Windows ↔ Cyber Universe pe aceeași placă** (ideea din C-117): pe laptop trebuie să apară **`94.06.1e.00.23`**, pe workstation
**`94.04.65.00.3e`**. Jetonul 'i' există pe placa de pe workstation (`0x69v2` în lista observată, 29 oct); dacă lipsește sau e scurt, faptul spune „necitita …”, nu inventează.
`vbios::BiosVersion::matches_report("94.06.1E.00.23")` face comparația (majuscule/minuscule indiferent).

## 7. Pasul 3 din C-116 (I6–I8 pe GA106) — ce ar fi de făcut, scris ÎNAINTE de DA (pregătire, fără cod)

Pasul 3 cere **DA-ul separat al lui Adrian** (C-116; Adrian e plecat până luni seara, 16 nov) — nimic de aici nu e scris în cod. Lista, ca să fie pasul scurt când vine DA-ul:

1. **`Family::writes_allowed()`** azi = doar GA104. Pasul 3 = GA106 intră și el, **numai** când OS-ul rulează pe stick-ul de test cu `gpu-scrie` și după „DA” la ecran (ca pe workstation);
   în plus trezirea minimă (§6) trebuie ținută pornită pe toată durata lanțului (Memory Space + Bus Master pentru DMA), nu repusă după tur. Repunerea la final rămâne (bus-master-oprit + Command repus).
2. **Firmware-ul:** aceleași 4 fișiere 570.144 de pe stick (`ga106/gsp -> ga102/gsp`, `COMPATIBILITATE_FIRMWARE.md`); nimic de schimbat; aceleași sume SHA-256.
3. **I6 (FWSEC-FRTS):** descriptorul laptopului (0x2c834, IMEM 0xe100) e citit deja de același parser; WPR2 coborât la pornire rece → același drum ca pe GA104. Prima scriere reală în registrele GA106 ar fi aici.
4. **I7 (Booter) / I8 (GSP-RM):** aceeași secvență; varianta 1 (`rpc-inainte`) care a trecut I8 pe workstation (os-295). `SetSystemInfo`: mărimea VRAM 6144 MiB vine din placă (`fb_layout`), nu e constantă.
5. **Limita de memorie (pasul 4):** `laptop::VRAM_BUDGET_BYTES` (5,5 GB) — de legat la alocatorul VRAM când lanțul ajunge la I9+ pe laptop; până atunci nu se alocă nimic peste aranjamentul GSP.
6. **Ce NU știm încă pe GA106** (niciodată citit): registrele GSP/SEC2/FBIF, comportamentul Booter-ului pe laptop, dacă GSP-RM are nevoie de afișaj (nu există: ecranul e pe Intel — ipoteza din os-295 §3 se poate verifica aici).
7. **Test pe laptop, când vine DA-ul:** întâi `gpu i6` singur (primele scrieri de registre pe GA106: FWSEC-FRTS), jurnal pe stick, abia apoi `gpu lant` (I7, I8) — un pas pe test, nu tot lanțul dintr-o dată.

## Surse

- open-gpu-kernel-modules, tag `570.144` (MIT): `README.md`, `src/nvidia/generated/g_chips2halspec_nvoc.c`, `g_kernel_gsp_nvoc.c` — citite ca specificație.
- linux-firmware (gitlab.com/kernel-firmware/linux-firmware, `WHENCE`, commit `69e51bb`, 9 nov 2026): rândul `Link: nvidia/ga106/gsp -> ../ga102/gsp`.
  Niciun fișier firmware descărcat sau analizat.
- pci.ids (citat, nu copiat): atribuirea ID-urilor `0x25xx` sub `0x2580` la GA106.
- Hub `rezultate/2026-11-11-laptop-asus-tuf/REZULTAT.md` + pozele 4–7 (12 nov 2026): faptele din §0 (OBSERVATE de OS, os-295).
- Hub `rezultate/2026-11-13-laptop-asus-os301/` (13 nov 2026): `gpu-jurnal-003.txt` (24 de rânduri, cele 8 citiri BAR0 și cele 2 scrieri în Command) și poza `gpu-tur-trezire.jpg` (faptele din §0c, transcrise de noi; rândurile tăiate din poză nu sunt ghicite).
- PCI Local Bus Specification 3.0 §6.2.2 (Command, bitul 1); Linux `pci_enable_device` (GPL, citit ca specificație) — §6.
