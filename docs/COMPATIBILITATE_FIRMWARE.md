# Tabelul de compatibilitate cu firmware-ul (C-18)

*Creat 26 oct 2026 (sesiunea 5, cloud, fără placă). Un singur loc pentru tot ce depinde de placă și de firmware-ul NVIDIA.
Strânge ce era împrăștiat în `RESEARCH_I4_I8.md`, `GA104_INITIALIZATION.md` și `IMPLEMENTATION_STATUS.md` (acelea rămân ca istoric).*

**Versiunea fixată: GSP-RM 570.144** (decizia lui Adrian, 26 oct 17:10). Placa: RTX 3070 Ti, GA104, VBIOS 94.04.65.00.3e (K0).

**Stări:** **confirmat** = verificat pe placa reală · **parțial** = o parte e verificată (sursă oficială / fișierul real), restul nu ·
**necunoscut** = nu știm încă destul ca să scriem codul · **SIMULARE** = cod scris și testat doar pe modele software (`gpu-sim`, GSP simulat,
imagini sintetice). **Nimic din tabel nu e „confirmat” pe placă**: nicio linie de cod nu a rulat încă pe RTX 3070 Ti.

| # | Componentă | Sursa de referință (fișier + versiune) | Comportament observat | ABI / versiune | Dependențe | Fișierul Rust | Testul | Stare |
|---|---|---|---|---|---|---|---|---|
| I0 | Identificare PCI | pci.ids; README open-gpu-kernel-modules (MIT) | Windows/K0: `10de:2482`, subsistem `1043:8813`, magistrala B3, BAR0 16 MiB, BAR1 256 MiB, BAR3 32 MiB | PCI 3.0 | spațiul de configurare (OS) | `gpu-pci/*`, `nvidia-ga104/src/probe.rs`, `ids.rs` | `probe::tests::level0_on_simulated_workstation_config` | SIMULARE (OS-ul o rulează din os-189; nerulat pe workstation) |
| I1 | BOOT_0 / BOOT_42 | open-gpu-doc `dev_boot.ref.txt` (MIT); nouveau `device/base.c` | — | registre fixe Ampere | BAR0 primii 4 KiB, UC | `nvidia-ga104/src/boot.rs`, `regs.rs` | `boot::tests::*` | SIMULARE |
| I2 | Așteptarea GFW_BOOT | nouveau `devinit/tu102.c` (MIT); nova-core `gfw.rs` (doar ordinea) | — | registre `0x118128`, `0x118234` | BAR0 4 MiB UC | `nvidia-ga104/src/gfw.rs` | `gfw::tests::*` (5) | SIMULARE |
| I3 | VBIOS prin PROM | docs.kernel.org „nova/core/vbios”; PCI Firmware 3.0; nouveau `npde.c`, `bit.c` | — | fereastra `BAR0+0x300000`, BIT jeton `0x70` | I2 | `nvidia-ga104/src/vbios.rs` | `vbios::tests::*` (8) | SIMULARE |
| I3b | Descriptorul FWSEC_PROD | patch-uri nova-core LKML (`FalconUCodeDescV3`); nouveau `bios/pmu.c` | — | descriptor v3, 44 de octeți | I3 | `nvidia-ga104/src/fwsec.rs` | `fwsec::tests::*` (11) | SIMULARE |
| I4 | Fișierele firmware GSP pe stick | linux-firmware commit `050f0a1e3a76` (tag `20261019`), `nvidia/ga102/gsp/*-570.144.bin` | mărimile și SHA-256 calculate pe 26 oct din fișierele oficiale (conținutul neanalizat, fișierele șterse) | 570.144 | FAT pe stick (OS), `FirmwareSource` | `nvidia-ga104/src/firmware.rs`, `sha256.rs` | `firmware::tests::*` (5), `sha256::tests::*` (3, vectori NIST) | parțial: sumele = din fișierele reale; citirea de pe stick = SIMULARE |
| I4b | Secțiunile ELF ale GSP-RM (`.fwimage`, `.fwsignature_ga10x`) | specificația ELF-64; numele secțiunilor DEDUS din nouveau `gsp/r535.c` | — | ELF-64 LE | I4 | `nvidia-ga104/src/gsp/elf.rs` | `gsp::elf::tests::*` (2, ELF sintetic) | SIMULARE (pe fișierul real: nerulat) |
| I4c | Antetul fișierelor `bootloader` / `booter_load` (offset cod/date/manifest) | `rmRiscvUcode.h`@570.144 (MIT, generat: `RM_RISCV_UCODE_DESC` 84 B); nouveau `nvfw/fw.h`, `nvfw/hs.h` (MIT, DEDUS) | — | antete binare (0x10de) | I4 | `nvidia-ga104/src/fw_hdr.rs` | `fw_hdr::tests::*` (3, fișiere sintetice) | SIMULARE (pe fișierele reale: nerulat; conținutul rămâne opac) |
| I5a | Aranjamentul VRAM (VGA, FRTS, boot, ELF, heap WPR2, WPR2, heap) | patch-uri nova-core LKML „compute layout …” | — | LIBOS3 | mărimea VRAM (din `fb_size_from_scratch42`, registrul de citit pe placă) | `nvidia-ga104/src/fb_layout.rs` | `fb_layout::tests::*` (6) | SIMULARE (aritmetică) |
| I5b | `GspFwWprMeta` (256 B) | open-gpu-kernel-modules@570.144 `gsp_fw_wpr_meta.h` (MIT); câmpurile din nouveau `r535.c` | — | magic `0xdc3aae21371a60b3`, revizia 1 | I4b, I4c, I5a, buffere DMA | `nvidia-ga104/src/gsp/wpr_meta.rs` | `gsp::wpr_meta::tests::*` (3) | SIMULARE |
| I5c | Radix3 (imaginea GSP-RM în RAM) | patch nova-core LKML „process and prepare the GSP firmware” | — | pagini 4 KiB, 3 niveluri | I4b, buffere DMA | `nvidia-ga104/src/radix3.rs` | `radix3::tests::*` (3) | SIMULARE (aritmetică) |
| I5d | Argumentele LibOS (LOGINIT, LOGINTR, LOGRM, RMARGS) | `libos_init_args.h`@570.144 (MIT); cele 4 regiuni DEDUS din nouveau | — | 32 B / intrare, `id8` | buffere DMA | `nvidia-ga104/src/gsp/libos.rs` | `gsp::libos::tests::*` (3) | SIMULARE |
| I5e | `GSP_ARGUMENTS_CACHED` (72 B) | `gsp_init_args.h`@570.144 (MIT) | — | 570.144 (are `bDmemStack`, `profilerArgs`) | I5f | `nvidia-ga104/src/gsp/libos.rs` | `gsp::libos::tests::gsp_args_offsets` | SIMULARE |
| I5f | Memoria partajată + cozile (tabela de pagini, 2 × 0x40000) | `message_queue_priv.h`, `msgq_priv.h`, `msgq.c`, `message_queue_cpu.c`@570.144 (MIT) | — | `MSGQ_VERSION 0`, `SWAP_RX`, 63 elemente de 4 KiB | buffere DMA (OS), Bus Master | `nvidia-ga104/src/gsp/msgq.rs` | `gsp::msgq::tests::*` (3), `gsp::sim_peer::tests::*` (4) | SIMULARE |
| I5g | Bus Master + bufferele DMA | DECIZII D-013 (o scriere, valoarea înainte/după) | — | PCI | fereastra cu Adrian, VT-d oprit | — (va fi în glue-ul OS, sub `HardwareWritePermit`) | — | necunoscut (nescris, intenționat) |
| I6 | FWSEC-FRTS pe falcon-ul GSP | `dev_falcon_v4.h`, `dev_gsp.h`, `dev_riscv_pri.h`, `dev_fb.h`, `dev_bus.h`@570.144 (MIT, generate); ordinea: nouveau `falcon/ga102.c` + `gsp/r535.c`, tinygrad `ip.py` (MIT) | — | reset → nucleu falcon → încărcare → start → oprire → `SW_SCRATCH(0xe)` + WPR2 | I3b, I5a | `nvidia-ga104/src/gsp/falcon.rs` (`fwsec_frts`, `GuardedBar0`) | `gsp::falcon::tests::*` (5) pe `gpu-sim::fake_gsp` | SIMULARE (încărcarea ucode-ului prin DMA: nescrisă → `load_not_written` întoarce eroare) |
| I7 | Booter pe SEC2 + pornirea GSP RISC-V | aceleași antete + `dev_sec_pri.h`; ordinea: nouveau `r535.c`, tinygrad `ip.py` | — | mailbox 0/1 = adresa `GspFwWprMeta`; mailbox0 = 0 la oprire; `RISCV_CPUCTL.ACTIVE_STAT` | I4c, I5b–f, I6 | `nvidia-ga104/src/gsp/falcon.rs` (`booter_load`) | idem | SIMULARE (încărcarea Booter-ului: nescrisă) |
| I8a | Elementul de coadă + antetul RPC + suma de control | `message_queue_priv.h`, `g_rpc-message-header.h`, `rpc_headers.h`@570.144 (MIT) | — | antet v3.0, semnătura `VRPC` | I5f | `nvidia-ga104/src/gsp/rpc.rs` | `gsp::rpc::tests::*` (4), `sim_peer` | SIMULARE |
| I8b | Numerele funcțiilor și evenimentelor RPC | `rpc_global_enums.h`@570.144 (MIT) | — | `SET_SYSTEM_INFO`=72, `SET_REGISTRY`=73, `GSP_INIT_DONE`=0x1001, `RUN_CPU_SEQUENCER`=0x1002 | — | `nvidia-ga104/src/gsp/rpc.rs` (`function`, `event`) | `header_shape_matches_570_144` | parțial (copiate exact din antet; nefolosite pe placă) |
| I8c | Tabela registrului (SetRegistry) | `g_os_nvoc.h` (`PACKED_REGISTRY_*`), `registry.c`@570.144 (MIT); cele 2 intrări: tinygrad + `osinit.c` | — | intrare 16 B, nume după intrări | I8a | `nvidia-ga104/src/gsp/registry.rs` | `gsp::registry::tests::*` (3) | SIMULARE |
| I8c' | Conținutul `GspSystemInfo` (SetSystemInfo, 928 B) | `gsp_static_config.h`@570.144 (MIT, generat: `nvidia-rm570::sysinfo`); valorile: `rpc.c` `rpcGspSetSystemInfo_v17_00`, `gpu_mgr.c`, `osinit.c` (MIT) | — | 570.144; BAR0/1/3, `domain<<32 \| bus<<8 \| slot`, ID-urile PCI ca dword-uri citite, oglinda PCFG 0x88000/0x1000, pagina 4 KiB; ACPI/SR-IOV/hypervisor = 0 („absent”, DEDUS că GSP le acceptă) | I0 (BAR-urile, ID-urile), I8a | `nvidia-ga104/src/gsp/sysinfo.rs` | `gsp::sysinfo::tests::*` (3), `rm570_crosscheck::system_info_matches_the_header`; trimis în pornirea simulată (`boot_to_init_done…`) | SIMULARE |
| I8f | Mesaje RPC mai mari de 16 elemente („continuation records”, funcția 71) | `rpc.c` `_issueRpcLarge`@570.144 (MIT) | — | aceeași secvență RPC, cozi consecutive | I8a | `nvidia-ga104/src/gsp/rpc.rs` (`encode_split`, `join_split`) | `large_rpc_splits_into_continuation_records_and_joins_back` | SIMULARE |
| I8e | **„Golden image”** (pas nou între I8 și I9): client, dispozitiv, subdispozitiv, spațiu de adrese, tabela motoarelor, PDE-urile rezervate (512 MiB), canalul (32 intrări, GR0), bufferele de context GR (10, din care 9 promovate), obiectele compute + copiere | tinygrad `ip.py` `init_golden_image` (MIT, `9cf7829a6`); structurile: `nvos.h`, `cl0080.h`, `alloc_channel.h`, `ctrl2080fifo.h`, `ctrl2080gpu.h`, `ctrl2080internal.h`, `ctrl90f1.h`, `g_rpc-structures.h`@570.144 (MIT, generate: `nvidia-rm570::rmapi`) | — | `GSP_RM_ALLOC`=103 (antet 32 B), `GSP_RM_CONTROL`=76 (antet 24 B); canal 368 B, PROMOTE_CTX 560 B, tabela motoarelor 3 212 B | I8, memoria plăcii (I9: alocator VRAM + tabele de pagini) | `nvidia-ga104/src/gsp/rmapi.rs`, `gsp/golden.rs` | `gsp::rmapi::tests::*` (4), `gsp::golden::tests::*` (4): secvența cap-coadă prin cozile reale către `gpu-sim::fake_rm` (7 alocări + 4 comenzi acceptate); refuzurile RM întorc codul RM; GSP tăcut = limită de timp | SIMULARE (valorile „magice” ca `flags=0x200320` = DEDUS, dovedite doar de tinygrad; mărimile bufferelor GR vin de la placă, în model sunt SIM) |
| I8d | Secvențiatorul CPU (`GSP_RUN_CPU_SEQUENCER`) + așteptarea `GSP_INIT_DONE` | `g_rpc-structures.h`, `rmgspseq.h`@570.144 (MIT, generat); semnificația comenzilor: nouveau `r535.c` (DEDUS) | — | antet 40 B + comenzi de lungime variabilă (9 coduri) | I8a | `nvidia-ga104/src/gsp/sequencer.rs` | `gsp::sequencer::tests::*` (4): pornire simulată cap-coadă I6 → I7 → cozi → secvențiator → INIT_DONE; scrieri în afara GSP/SEC2 refuzate; GSP tăcut = limită de timp | SIMULARE |
| I9 | Canal GPFIFO + `AMPERE_COMPUTE_B` prin RM | open-gpu-doc `clc56f.h`, `clc7c0.h` (MIT, generate în `nvidia-rm570`) | — | clasele 0xC56F, 0xC7C0 | I8, I8e | `nvidia-ga104/src/push.rs` (codificarea) | `push::tests::*`, `rm570_crosscheck::pushbuffer_encoding_matches_the_header`, `sim_add64` (procesorul de comenzi simulat) | SIMULARE (codificarea = antetul; executată doar de modelul nostru) |
| I10 | Primul program (adunarea a 64 de f32) + QMD V03_00 | Mesa 25.0.7 NAK (MIT), `tools/nak/`; `clc6c0qmd.h` (open-gpu-doc, MIT, generat) | — | SASS SM86, 15 instrucțiuni / 240 B; argumentele în cbuf 0 la 0x28/0x30/0x38; 19 registre | I9, QMD v3 | `tools/nak/out/add64.sm86.sass` (date), `nvidia-ga104/src/qmd.rs` | `sh tools/nak/build.sh add64`; `qmd::tests::*`; `sim_add64` (lansat cap-coadă în modelul `gpu-sim::cmdproc`, rezultat identic bit cu bit cu referința) | parțial: compilat pe gazdă, lansat doar în SIMULARE (programul „rulează” pe procesor), **nerulat pe placă** |

## Fișierele pe care le pune Adrian pe stick (570.144)
Toate din pachetul oficial `linux-firmware` (sau https://gitlab.com/kernel-firmware/linux-firmware, folderul `nvidia/ga102/gsp/`).
Pe stick, exact la: `firmware/nvidia/ga102/gsp/<nume>` (driverul citește `/usb/firmware/nvidia/ga102/gsp/<nume>`).

| Fișier | Mărime (octeți) | SHA-256 |
|---|---|---|
| `booter_load-570.144.bin` | 61 304 | `4497e3eff7e95c774b8a569d17b27c08c9650158d10b229d2be81cdcad9a085b` |
| `booter_unload-570.144.bin` | 41 080 | `8e63db5b78d7d3e349f20a2d11099c3d7109081393cb09ffc0a28133324ae009` |
| `bootloader-570.144.bin` | 24 684 | `82428f532240727e95bb3083fbaaba9b2cc7b937314323f2d546ce7245f27fad` |
| `gsp-570.144.bin` | 63 571 696 | `a8c3ebeed280323aedb51c061f321e73379cce7a9ae643a33dd03915df027f7f` |

Plus licența `LICENCE.nvidia` din același pachet, pusă alături (condiția de redistribuire §2.1.2). Fișierele **nu** intră în niciun repo.
Verificarea pe Windows, înainte de copiere (PowerShell): `Get-FileHash .\gsp-570.144.bin -Algorithm SHA256`.

## Reguli pentru tabel
- O linie trece în „confirmat” doar după testul pe placă, cu poza/jurnalul notat în `TEST_MATRIX.md` și în hub `rezultate/`.
- Orice schimbare de versiune a firmware-ului = tabel nou (ABI diferit); codul refuză altă versiune (`firmware::verify`).

## De verificat — semnalate de studiul din hub (`studiu/2026-10-26_open-source/gpu.md`, C-20), după scrierea codului de azi
| Ce | Codul nostru (26 oct) | tinygrad / sursa studiului | Ce facem |
|---|---|---|---|
| Sumele SHA-256 ale fișierelor 570.144 | `firmware::GSP_570_144` | aceleași valori (ip.py, citit în studiu) | confirmare independentă a sumelor (nu a plăcii) |
| Heap-ul WPR2 pentru 8 GiB | 127 MiB (formula LIBOS3, nova) | fix `0x8100000` = 129 MiB | **diferență** → test de comparație + decizie la C-20 (C); pe placă decide GSP-RM (îl acceptă sau nu) |
| Semnătura FWSEC aleasă | după siguranțe (`fwsec.rs`, DEDUS nouveau) | ultima semnătură, fără siguranțe | la testul 5c se notează câte semnături sunt și ce index ar alege formula noastră |
| Imaginile semnate pentru falcon | presupus: DMA din RAM | puse în VRAM prin BAR1, DMA din memoria plăcii | de reverificat la I6 |
| Bitul `GP_ENTRY1_LEVEL` în GPFIFO | 0 (`push.rs`, verificat față de antet în sesiunea 6) | 1 la fiecare intrare | de aliniat la varianta dovedită pentru primul test real (I9) |
| Pas între I8 și I9 („golden image”) | **cod în sesiunea 9** (`gsp::golden`, SIMULARE); tot nu în `init.rs` (o variantă nouă în enumerarea publică ar putea strica OS-ul) | există | jurnalul probei P dă ordinea și valorile reale de comparat |
| `bIsPassthru` în `GspSystemInfo` | 0 (`gsp::sysinfo`, ca driverul NVIDIA pe o mașină fizică) | **1** (ip.py `rpc_set_gsp_system_info`) | diferență notată în sesiunea 9; nu schimbăm fără motiv: de văzut în jurnalul probei P dacă GSP-RM se poartă diferit |
| Handle-ul obiectului la alocarea clientului (`NV01_ROOT`) | ca tinygrad: `hClient`=0xc1e00004, `hObject`=primul handle generat | nouveau: `hObject` = `hClient` | DEDUS: GSP-RM folosește `hClient`; de confirmat pe placă |
| Registrele falcon GA10x (I6–I7) | **generate** în `nvidia-rm570` (sesiunea 6); folosite de `gsp/falcon.rs` (sesiunea 7) | publicate MIT în `swref/published/ampere/ga102/` | I6–I7 = SIMULARE pe GSP-ul fals; lipsește încărcătorul DMA |
| Câmpurile QMD pe care NAK le pune mereu (`API_VISIBLE_CALL_LIMIT=NO_CHECK`, `SAMPLER_INDEX=INDEPENDENTLY`, `SM_GLOBAL_CACHING_ENABLE=1`) | nesetate (`qmd.rs`); memoria partajată + barierele: da (sesiunea 8) | Mesa NAK `qmd.rs` (citit în sesiunea 8) | de aliniat cu jurnalul probei P înainte de prima lansare reală |
| Mărimea VRAM | `fb_size_from_scratch42` | registrul `0x1183a4` (`NV_USABLE_FB_SIZE_IN_MB`, antet NVIDIA MIT) | citire nouă, doar-citire, de adăugat pe lista permisă la C-20 |

**Sesiunea 6:** toate offset-urile scrise de mână în sesiunea 5 (`GspFwWprMeta`, cozile msgq, `GSP_ARGUMENTS_CACHED`, GFW_BOOT, codificarea pushbuffer/GPFIFO) au fost comparate automat cu valorile generate din antete: **toate egale** (`crates/nvidia-ga104/tests/rm570_crosscheck.rs`).

**Sesiunea 9 (27 oct):** I8e are cod (`gsp::rmapi` + `gsp::golden`, structurile generate din antete; secvența rulează în SIMULARE prin cozile reale către un RM fals scris independent); matmul treapta 4 (1D blocktiling, 8 rezultate pe fir) compilată cu NAK și lansată în model. Nimic nu a rulat pe placă.

**Sesiunea 8 (27 oct):** I8c' are cod (`gsp::sysinfo`, structura generată din antet, trimisă în pornirea simulată); matmul treapta 3 (memorie partajată) compilată cu NAK și lansată în model cu QMD-ul complet pentru memoria partajată. Nimic nu a rulat pe placă.

**Sesiunea 7 (27 oct):** I6, I7, I8d au cod (SIMULARE pe `gpu-sim::fake_gsp`), I4c are cod (SIMULARE pe fișiere sintetice); după I10: matmul treptele 1–2 compilate cu NAK (`ffma.ftz`) și lansate în modelul de comenzi, oracol `matmul_fma_ftz` (`tools/nak/README.md`). Nimic nu a rulat pe placă.
