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
| nova-core (Linux, Rust) | GPL-2.0 | doar ordinea pașilor și discrepanța BOOT_42; **niciun cod** |
| Mesa NAK | MIT | identificat ca posibilă unealtă pe gazdă (TOOLCHAIN.md); necopiat |
| pci.ids | GPL-2.0+ / BSD-3 | date (ID-uri), citate |
| linux-firmware `LICENCE.nvidia` | proprietară, redistribuibilă condiționat | inventarul fișierelor; fișierele nu sunt incluse |
| Abramowitz & Stegun 7.1.26; metode clasice (Newton, reducerea intervalului) | domeniu public / manual | `gpu-compute::math` |
| Posturi forum cu `lspci` (bbs.archlinux.org, pciutils #151) | fără licență | doar ca fapt observat (BAR-uri) |

Componente binare externe de care va depinde calea GSP: firmware-urile NVIDIA semnate (`firmware.rs`) și VBIOS-ul plăcii.
