# C-95 / C-116 — Laptopul ASUS TUF F15 (RTX 3060 Laptop, GA106) ca al doilea calculator de test

> **RELUAT de Adrian (C-116, 12 nov 20:57):** „ne mutăm cu testele aici (pe laptop) și vor fi mult mai rapide; apoi facem GPU pentru placa asta”.
> Istoric: parcat pe 10 nov 07:28 (C-95). De la sesiunea 95 (12 nov, noaptea) codul GA106 doar-citire e **înapoi în proiect** (`6623d3c`), plus
> „trezirea minimă” (`ccec8f9`). **Lanțul cu scrieri în registre GPU (I6+) rămâne doar pentru GA104** până la DA-ul separat al lui Adrian (C-116 pasul 3).

*Scris 10 nov 2026 (sesiunea 77), completat 12 nov 2026 (sesiunea 95), în cloud, FĂRĂ placă.*

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
| I0 identificare PCI | ID-uri noi (tabelul `GA106_MODELS`); **`10de:2520` văzut pe laptop** | cod: SIMULARE; ID-ul OBSERVAT |
| I0b Memory Space | **oprit de firmware** (Command 0) → trezirea minimă (§6) înainte de orice citire BAR0 | cod: SIMULARE (T126e–g) |
| I1 BOOT_0/BOOT_42 | cip `0x176` | cod: SIMULARE; de citit pe laptop după trezire |
| I2 GFW_BOOT | aceleași registre la Ampere (DEDUS: nu am găsit o variantă GA106 separată) | de citit pe laptop |
| I3 VBIOS prin PROM | **risc de laptop** (vezi §3) | de citit pe laptop |
| I3b FWSEC | VBIOS-ul laptopului are FWSEC-ul lui (alt `ucode_id`/semnături posibile) | de citit pe laptop |
| I3c siguranța | aceeași funcție NVIDIA (GA100) | de citit pe laptop |
| I4 fișierele firmware | **identice** (scurtătura `ga106/gsp -> ga102/gsp`) | confirmat din `linux-firmware` `WHENCE` |
| I5a aranjamentul VRAM | aceeași aritmetică, mărimea citită din placă (~6 GiB) | SIMULARE |
| I5b–I8 | același GSP-RM, aceleași mesaje | SIMULARE; scrieri = pasul 3 C-95 |
| I8e golden image / I9 | mărimile bufferelor GR vin de la GSP; mai puține GPC/SM | SIMULARE |
| I10+ programe | sm_86 ca GA104 (NAK produce același cod) | SIMULARE |

## 3. Ce înseamnă „placă de laptop” (riscurile)

1. **Placa poate lipsi cu totul de pe magistrala PCI.** La laptopurile ASUS, modul „Eco” din Armoury Crate (sau „GPU mode”) poate opri placa
   NVIDIA din firmware; atunci nici Windows, nici OS-ul nostru nu o văd. **Înainte de test: în Armoury Crate, modul GPU = „Standard”
   (sau „Ultimate”), nu „Eco”.** (DEDUS din documentația ASUS / driverul Linux `asus-wmi` „dgpu_disable”; nu e verificat pe laptopul lui Adrian.)
2. **Alimentarea prin ACPI.** Pe laptopurile Optimus, placa NVIDIA e pornită la boot de firmware (UEFI) și abia sistemul de operare
   (Windows/Linux) o adoarme mai târziu (D3cold, prin metodele ACPI `_PR3`/`_PS3`). OS-ul nostru nu adoarme nimic, deci ne așteptăm ca
   placa să fie în D0. Codul verifică: dacă placa nu e în D0 → refuz, fără nicio citire din BAR0; dacă BOOT_0 = `0xffffffff` → „placa nu răspunde”.
   Nu încercăm `_PS0`/`_ON` (ar însemna scrieri și un interpretor ACPI). DEDUS, de confirmat la primul test.
3. **Ecranul.** Fără comutator MUX (sau cu MUX pe „hibrid”), ecranul laptopului e ținut de placa Intel, deci placa NVIDIA nu are imaginea
   ecranului în memoria ei (mai simplu decât pe workstation). Cu MUX pe „doar NVIDIA” ecranul e pe placa NVIDIA, ca la workstation.
   TUF F15 din 2021 (FX506H*) nu are MUX; cel din 2022 (FX507Z*) are. Modelul exact îl spune eticheta de sub laptop sau `hw lista`.
4. **VBIOS-ul.** La plăcile Turing/Ampere pornirea (GFW) rulează din memoria flash a plăcii, deci VBIOS-ul ar trebui să se citească prin
   fereastra PROM, ca pe workstation. La unele laptopuri însă VBIOS-ul e dat și prin ACPI (`_ROM`); dacă `gpu vbios` citește gunoi pe
   laptop, aceasta e cauza probabilă și citirea prin ACPI devine o sarcină separată. Risc, nu certitudine.
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

## Surse

- open-gpu-kernel-modules, tag `570.144` (MIT): `README.md`, `src/nvidia/generated/g_chips2halspec_nvoc.c`, `g_kernel_gsp_nvoc.c` — citite ca specificație.
- linux-firmware (gitlab.com/kernel-firmware/linux-firmware, `WHENCE`, commit `69e51bb`, 9 nov 2026): rândul `Link: nvidia/ga106/gsp -> ../ga102/gsp`.
  Niciun fișier firmware descărcat sau analizat.
- pci.ids (citat, nu copiat): atribuirea ID-urilor `0x25xx` sub `0x2580` la GA106.
- Hub `rezultate/2026-11-11-laptop-asus-tuf/REZULTAT.md` + pozele 4–7 (12 nov 2026): faptele din §0 (OBSERVATE de OS, os-295).
- PCI Local Bus Specification 3.0 §6.2.2 (Command, bitul 1); Linux `pci_enable_device` (GPL, citit ca specificație) — §6.
