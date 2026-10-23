# Proveniență și licențe

Tot codul din `crates/` și `integration/` e **original**, MIT, scris după documentația de mai jos. Niciun fișier sursă străin
nu a fost copiat. Nicio dependență externă (doar `core`/`alloc`, și `std` în teste). Niciun firmware în repository.

Regula clean-room a OS-ului (`cyber-universe/docs/CLEAN_ROOM.md`): sursele deschise se citesc pentru idei, nu se copiază;
codul GPL nu se copiază niciodată.

| Sursă | Licență | Cum a fost folosită |
|---|---|---|
| PCI Local Bus Spec 3.0, PCI Express Base Spec | specificații | antet, BAR, capabilități, ReBAR (`gpu-pci`) |
| NVIDIA open-gpu-doc (`manuals/ampere/ga100/dev_boot.ref.txt`, `classes/host/clc56f.h`, `classes/compute/clc7c0*.h`) | MIT | câmpuri de registre, clase, format pushbuffer/GPFIFO — interfață documentată |
| NVIDIA open-gpu-kernel-modules (README, `clc361.h`) | MIT / dual MIT-GPLv2 | ID-uri, faptul că Ampere cere GSP, offset doorbell Volta |
| nouveau (`nvkm/...`) | MIT-style | ordinea pașilor GSP, calculul chipset-ului, adresa doorbell GA10x (dedusă) — citit, necopiat |
| nova-core (Linux, Rust) | GPL-2.0 | doar ordinea pașilor și discrepanța BOOT_42; procedura GFW (`gfw.rs`: ordinea mască → progres, limita de 4 s); **niciun cod** |
| Documentația oficială a kernelului Linux, „nova/core/vbios” (docs.kernel.org) | GPL-2.0 (documentație) | fereastra PROM 0x300000, antetul IFR/RFRD, tipurile de imagini, lanțul BIT → Falcon data → FWSEC — ca descriere de format (`vbios.rs`) |
| nouveau `devinit/tu102.c` (patch „wait for GFW_BOOT_PROGRESS”, citit pe code.opensuse.org) | MIT | adresele 0x118128/0x118234, măștile, valoarea 0xff (`gfw.rs`) |
| nouveau `bios/npde.c`, `image.c`, `bit.c` | MIT | câmpurile NPDE și ale tabelei BIT (DEDUS, de confirmat pe placă) |
| Patch-uri nova-core pe LKML: „firmware: add ucode descriptor used by FWSEC-FRTS” (v2 16/21, mai 2025) și „firmware: add comments to justify v3 header values” (v11 08/12, mar. 2026), citite pe lkml.iu.edu | GPL-2.0 | doar descrierea câmpurilor `FalconUCodeDescV3` (44 de octeți) și faptul că IMEM începe la 0, DMEM la `imem_load_size` (`fwsec.rs`); **niciun cod** |
| nouveau `bios/pmu.c`, `falcon/fw.c` (MIT) — din cunoaștere generală, necitit în sesiunea 3 | MIT | antetul și intrările tabelei PMU, alegerea semnăturii după siguranță (DEDUS, de confirmat pe placă) |
| PCI Firmware Specification 3.0 | specificație | antetul ROM, structura PCIR (`vbios.rs`) |
| Mesa NAK | MIT | identificat ca posibilă unealtă pe gazdă (TOOLCHAIN.md); necopiat |
| pci.ids | GPL-2.0+ / BSD-3 | date (ID-uri), citate |
| linux-firmware `LICENCE.nvidia` | proprietară, redistribuibilă condiționat | inventarul fișierelor; fișierele nu sunt incluse |
| Abramowitz & Stegun 7.1.26; metode clasice (Newton, reducerea intervalului) | domeniu public / manual | `gpu-compute::math` |
| Posturi forum cu `lspci` (bbs.archlinux.org, pciutils #151) | fără licență | doar ca fapt observat (BAR-uri) |

Componente binare externe de care va depinde calea GSP: firmware-urile NVIDIA semnate (`firmware.rs`) și VBIOS-ul plăcii.
