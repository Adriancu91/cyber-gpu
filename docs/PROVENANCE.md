# Proveniență și licențe

Tot codul din `crates/` și `integration/` e **original**, MIT, scris după documentația de mai jos. Niciun fișier sursă străin
nu a fost copiat. **Singura excepție (C-20 B, aprobată în hub):** `crates/nvidia-rm570/src/regs.rs` și `structs.rs` sunt **generate automat**
din antetele MIT ale NVIDIA (570.144), cu notificarea lor în `crates/nvidia-rm570/LICENSE-NVIDIA`. Nicio dependență externă (doar `core`/`alloc`, și `std` în teste). Unelte doar pe gazdă (nu intră în OS): Mesa NAK (`tools/nak/`), bindgen-cli 0.71.1 (`tools/gen-rm570/`). Niciun firmware în repository.

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
| Patch-uri nova-core pe LKML: „compute layout of the FRTS region” (v2 19/21, mai 2025) și „compute layout of more framebuffer regions required for GSP” (v9 01/15, nov. 2025), citite pe lkml.iu.edu | GPL-2.0 | doar ordinea regiunilor din VRAM, alinierile și constantele heap-ului LIBOS3 (`fb_layout.rs`); **niciun cod**. Neclar în surse: când se citește registrul zonei VGA (0x625f04) — lăsat ca parametru |
| Patch nova-core pe LKML „firmware: process and prepare the GSP firmware” (v4 07/11, sep. 2025), citit pe lkml.iu.edu | GPL-2.0 | doar formatul radix3 (pagini 4 KiB, intrări de 8 octeți, cele trei niveluri) (`radix3.rs`); **niciun cod** |
| Antetele NVIDIA open-gpu-kernel-modules, eticheta `570.144` (raw.githubusercontent.com): `gsp_fw_wpr_meta.h`, `msgq_priv.h`, `msgq.h`, `message_queue_priv.h`, `g_rpc-message-header.h`, `rpc_headers.h`, `rpc_global_enums.h`, `gsp_init_args.h`, `libos_init_args.h`, `g_rpc-structures.h` (doar `rpc_run_cpu_sequencer_v17_00`); `msgq.c`, `message_queue_cpu.c` citite pentru reguli (pointeri, SWAP_RX, mărimi, suma de control) | MIT | **interfața ABI** (offset-uri, constante, numere de funcții) rescrisă în `gsp/wpr_meta.rs`, `gsp/libos.rs`, `gsp/msgq.rs`, `gsp/rpc.rs`; **niciun cod copiat**; sesiunea 5, 26 oct 2026 |
| nouveau `gsp/r535.c` (MIT), din cunoaștere generală | MIT | DEDUS: ce regiune FB intră în ce câmp al `GspFwWprMeta`, cele 4 regiuni LibOS și funcția `id8`, numele secțiunilor ELF `.fwimage` / `.fwsignature_ga10x` — de confirmat pe placă |
| NVIDIA open-gpu-kernel-modules, tag `570.144`, commit `8ec351aeb96a` (github.com/NVIDIA/open-gpu-kernel-modules): `swref/published/ampere/ga102/{dev_boot,dev_falcon_v4(+_addendum),dev_falcon_second_pri,dev_fbif_v4,dev_gsp(+_addendum),dev_sec_pri(+_addendum),dev_riscv_pri,dev_gc6_island(+_addendum)}.h`, `turing/tu102/dev_fb.h`, `gsp_fw_wpr_meta.h`, `msgq_priv.h`, `gsp_init_args.h` | MIT (SPDX în fiecare antet; `COPYING`) | **generare automată** (C-20 B, sesiunea 6): `tools/gen-rm570/gen_regs.py` (scris de noi, doar biblioteca standard Python) pentru registre și **bindgen-cli 0.71.1** (unealtă pe gazdă, versiune fixată) pentru structuri → crate-ul `nvidia-rm570` (doar date), cu notificarea MIT NVIDIA. Folosit deocamdată doar în teste, ca verificare a offset-urilor scrise de mână |
| tinygrad (github.com/tinygrad/tinygrad), commit `9cf7829a6` (26 oct 2026): `runtime/support/nv/nvdev.py`, `ops_nv.py`, `support/system.py`, `helpers.py` (`fetch_fw`), `docs/runtime.md`, `docs/env_vars.md` | MIT (© 2024 the tiny corp) | **doar citit** (C-20 A/C, sesiunea 6): formatul liniei `wreg:` (parserul `gpu-sim::golden_log`, scris de noi), cum se alege interfața PCI și compilatorul NAK, de unde ia firmware-ul (pentru `tools/proba-tinygrad/proba.sh`); cele 2 intrări SET_REGISTRY. Niciun cod copiat |
| open-gpu-kernel-modules@570.144 `src/nvidia/src/kernel/vgpu/rpc.c` (`_issueRpcLarge`), `generated/g_os_nvoc.h` (`PACKED_REGISTRY_*`), `arch/nvalloc/unix/src/registry.c`, `osinit.c` | MIT | regula „continuation records” (funcția 71) în `gsp/rpc.rs` și formatul tabelului de registru în `gsp/registry.rs` — citite, rescrise (sesiunea 6) |
| NIST FIPS 180-4 (SHA-256) și vectorii de test NIST | standard public | `sha256.rs`, scris după specificație |
| Specificația ELF-64 (System V gABI) | publică | `gsp/elf.rs` (căutarea unei secțiuni după nume; conținutul rămâne opac) |
| linux-firmware (gitlab.com/kernel-firmware), commit `050f0a1e3a76`, tag `20261019` | `LICENCE.nvidia` | **doar** mărimea și SHA-256 ale celor 4 fișiere GSP 570.144 (calculate cu `sha256sum`, fișierele șterse imediat, conținutul neanalizat; s-a verificat doar că `gsp-570.144.bin` începe cu semnătura ELF, cum face orice încărcător) |
| PCI Firmware Specification 3.0 | specificație | antetul ROM, structura PCIR (`vbios.rs`) |
| Mesa 25.0.7 (NAK, NVK, drm-shim nouveau; archive.mesa3d.org, SHA-256 verificat) | MIT | **unealtă pe gazdă** (`tools/nak/`, C-18): compilează `add64.comp` pentru SM 8.6; nimic copiat în `crates/`; din sursă s-au citit doar `nak/api.rs` (formatul dump-ului), `nvk_shader.c` și `nouveau_noop.c` (cum se alege cipul) |
| pci.ids | GPL-2.0+ / BSD-3 | date (ID-uri), citate |
| linux-firmware `LICENCE.nvidia` | proprietară, redistribuibilă condiționat | inventarul fișierelor; fișierele nu sunt incluse |
| Abramowitz & Stegun 7.1.26; metode clasice (Newton, reducerea intervalului) | domeniu public / manual | `gpu-compute::math` |
| Posturi forum cu `lspci` (bbs.archlinux.org, pciutils #151) | fără licență | doar ca fapt observat (BAR-uri) |

Componente binare externe de care va depinde calea GSP: firmware-urile NVIDIA semnate (`firmware.rs`) și VBIOS-ul plăcii.
