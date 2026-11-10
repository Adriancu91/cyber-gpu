# C-95 — Laptopul ASUS TUF F15 (RTX 3060 Laptop, GA106) ca al doilea calculator de test

> **PARCAT de Adrian (10 nov 07:28):** „rămânem pe workstation cu Cyber Universe, pentru că avem deja setări pentru placa aia video”.
> Documentul rămâne doar ca **notiță**; **niciun cod pentru GA106** în proiect. Workstation-ul (GA104) e singura țintă.

*Scris 10 nov 2026, sesiunea 77 (cloud, FĂRĂ placă). Nimic de aici nu e citit de pe laptop: e cercetare din surse publice.*

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
- **Cod:** niciunul în proiect (C-95 parcat). Driverul refuză în continuare orice placă în afară de GA104, încă de la identificare.

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
| I0 identificare PCI | ID-uri noi (tabelul `GA106_MODELS`) | cod: SIMULARE |
| I1 BOOT_0/BOOT_42 | cip `0x176` | cod: SIMULARE |
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

## 4. Cod (retras)

În aceeași sesiune s-a scris o recunoaștere GA106 doar pentru citire (`level0_ampere`, `run_tour_ampere`, profil `gpu-sim` GA106, 6 teste
SIMULARE, CI verde) — commit `256a1f2`. Parcarea lui Adrian a apărut în hub cu 3 minute înainte, nevăzută la timp; codul a fost
**retras imediat** printr-un commit de anulare (istoria nu se rescrie). Dacă C-95 se reia, punctul de plecare e `256a1f2`
(adăugări, nimic redenumit). Până atunci: `probe::level0`, `run_tour` și lanțul refuză laptopul, ca înainte.

## 5. Dacă se reia: ce ar trebui să scrie jurnalul de citire de pe laptop

`gpu tur` cu `run_tour_ampere` (din `256a1f2`) ar da aproape tot: ID-ul PCI și subsistemul, starea D0/D3, BAR-urile și legătura PCIe, BOOT_0/BOOT_42,
starea GFW, imaginile VBIOS (tipuri, mărimi, ID-uri), jetoanele BIT, descriptorul FWSEC și semnăturile, siguranța, mărimea VRAM, WPR2.
În plus ar ajuta (partea OS): modelul laptopului din SMBIOS (`hw lista`), prezența plăcii Intel, și dacă există placa NVIDIA pe magistrală
(dacă lipsește → modul „Eco”).

## Surse

- open-gpu-kernel-modules, tag `570.144` (MIT): `README.md`, `src/nvidia/generated/g_chips2halspec_nvoc.c`, `g_kernel_gsp_nvoc.c` — citite ca specificație.
- linux-firmware (gitlab.com/kernel-firmware/linux-firmware, `WHENCE`, commit `69e51bb`, 9 nov 2026): rândul `Link: nvidia/ga106/gsp -> ../ga102/gsp`.
  Niciun fișier firmware descărcat sau analizat.
- pci.ids (citat, nu copiat): atribuirea ID-urilor `0x25xx` sub `0x2580` la GA106.
