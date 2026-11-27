# Proveniență și licențe

**Licența (C-33, 28 oct 2026):** codul nostru e **proprietar** („toate drepturile rezervate”, `LICENSE`) de la commit-ul de după `8313973`;
versiunile până la `8313973` inclusiv au fost publicate sub MIT și rămân MIT pentru cine le-a primit. Componentele străine își păstrează
licențele (fișierele generate din antetele NVIDIA: MIT NVIDIA, notificarea în `crates/nvidia-rm570/LICENSE-NVIDIA`). Firmware-ul NVIDIA
nu îl distribuim niciodată (nici în repo, nici în hub, nici în arhivele de stick): Adrian îl descarcă singur din `linux-firmware`.

Tot codul din `crates/` și `integration/` e **original**, scris de noi după documentația de mai jos (licența: vezi mai sus). Niciun fișier sursă străin
nu a fost copiat. **Singura excepție (C-20 B, aprobată în hub):** `crates/nvidia-rm570/src/regs.rs` și `structs.rs` sunt **generate automat**
din antetele MIT ale NVIDIA (570.144), cu notificarea lor în `crates/nvidia-rm570/LICENSE-NVIDIA`. Nicio dependență externă (doar `core`/`alloc`, și `std` în teste). Unelte doar pe gazdă (nu intră în OS): Mesa NAK (`tools/nak/`), bindgen-cli 0.71.1 (`tools/gen-rm570/`). Niciun firmware în repository.

Regula clean-room a OS-ului (`cyber-universe/docs/CLEAN_ROOM.md`): sursele deschise se citesc pentru idei, nu se copiază;
codul GPL nu se copiază niciodată.

| Sursă | Licență | Cum a fost folosită |
|---|---|---|
| PCI Local Bus Spec 3.0, PCI Express Base Spec | specificații | antet, BAR, capabilități, ReBAR (`gpu-pci`) |
| NVIDIA open-gpu-doc (`manuals/ampere/ga100/dev_boot.ref.txt`, `classes/host/clc56f.h`, `classes/compute/clc7c0*.h`) | MIT | câmpuri de registre, clase, format pushbuffer/GPFIFO — interfață documentată |
| NVIDIA open-gpu-kernel-modules (README, `clc361.h`) | MIT / dual MIT-GPLv2 | ID-uri, faptul că Ampere cere GSP, offset doorbell Volta |
| nouveau `nvkm/subdev/bios/image.c`, `pcir.c`, `npde.c` | MIT | semnăturile acceptate de imagine (`AA55`/`BB77`/`4E56`) și de PCIR („PCIR”/„RGIS”/„NPDS”), poziția NPDE (C-30, `vbios`, `vbios_diag`) — citit, necopiat |
| Linux nova-core `drivers/gpu/nova-core/vbios.rs` | GPL-2.0 | **doar citit ca specificație** (confirmarea semnăturilor `AA55`/`4E56` și „NPDS”); niciun rând copiat |
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
| NVIDIA open-gpu-kernel-modules, tag `570.144`, commit `8ec351aeb96a` (github.com/NVIDIA/open-gpu-kernel-modules): `swref/published/ampere/ga102/{dev_boot,dev_falcon_v4(+_addendum),dev_falcon_second_pri,dev_fbif_v4,dev_gsp(+_addendum),dev_sec_pri(+_addendum),dev_riscv_pri,dev_gc6_island(+_addendum)}.h`, `turing/tu102/dev_fb.h`, `gsp_fw_wpr_meta.h`, `msgq_priv.h`, `gsp_init_args.h`; plus **NVIDIA open-gpu-doc** commit `9fdf5c406200`: `classes/host/clc56f.h`, `classes/compute/clc7c0.h`, `classes/compute/clc6c0qmd.h` (doar QMD V03_00) | MIT (SPDX în fiecare antet; `COPYING`; open-gpu-doc `LICENSE`) | **generare automată** (C-20 B, sesiunea 6): `tools/gen-rm570/gen_regs.py` (scris de noi, doar biblioteca standard Python) pentru registre și **bindgen-cli 0.71.1** (unealtă pe gazdă, versiune fixată) pentru structuri → crate-ul `nvidia-rm570` (doar date), cu notificarea MIT NVIDIA. Folosit deocamdată doar în teste, ca verificare a offset-urilor scrise de mână |
| tinygrad (github.com/tinygrad/tinygrad), commit `9cf7829a6` (26 oct 2026): `runtime/support/nv/nvdev.py`, `ops_nv.py`, `support/system.py`, `helpers.py` (`fetch_fw`), `docs/runtime.md`, `docs/env_vars.md` | MIT (© 2024 the tiny corp) | **doar citit** (C-20 A/C, sesiunea 6): formatul liniei `wreg:` (parserul `gpu-sim::golden_log`, scris de noi), cum se alege interfața PCI și compilatorul NAK, de unde ia firmware-ul (pentru `tools/proba-tinygrad/proba.sh`); cele 2 intrări SET_REGISTRY. Niciun cod copiat |
| open-gpu-kernel-modules@570.144 `src/nvidia/src/kernel/vgpu/rpc.c` (`_issueRpcLarge`), `generated/g_os_nvoc.h` (`PACKED_REGISTRY_*`), `arch/nvalloc/unix/src/registry.c`, `osinit.c` | MIT | regula „continuation records” (funcția 71) în `gsp/rpc.rs` și formatul tabelului de registru în `gsp/registry.rs` — citite, rescrise (sesiunea 6) |
| open-gpu-kernel-modules@570.144 (același commit): `turing/tu102/dev_bus.h` (`NV_PBUS_SW_SCRATCH`), `nvalloc/common/inc/rmgspseq.h` (codurile secvențiatorului GSP, mărimile comenzilor), `rmRiscvUcode.h` (`RM_RISCV_UCODE_DESC`, 84 B) | MIT | **generate** (sesiunea 7) în `nvidia-rm570`; folosite de `gsp/sequencer.rs` și `fw_hdr.rs` |
| nouveau `nvkm/falcon/ga102.c`, `nvkm/subdev/gsp/r535.c` (`r535_gsp_msg_run_cpu_sequencer`, FWSEC-FRTS, Booter), `include/nvfw/fw.h`, `include/nvfw/hs.h` — din cunoaștere generală | MIT | DEDUS (sesiunea 7): ordinea resetării falcon-ului (RESET_READY → RESET → scrubbing), alegerea `CPUCTL_ALIAS`, codul de eroare FRTS în `NV_PBUS_SW_SCRATCH(0xe)[31:16]`, ce face fiecare comandă a secvențiatorului (inclusiv CORE_RESET/CORE_RESUME), antetele `nvfw_bin_hdr` și „HS v2” ale fișierelor Booter. Cod scris de noi în `gsp/falcon.rs`, `gsp/sequencer.rs`, `fw_hdr.rs`; de confirmat pe placă |
| tinygrad `test/mockgpu/am/amgpu.py` (MIT) | MIT | doar ideea unui „GPU fals” pe registre, ca mici automate → `gpu-sim::fake_gsp` (scris de noi, sesiunea 7) |
| musl libc `src/math/fmaf.c` (MIT) | MIT | doar ideea (produsul exact în f64, corecția cazului „exact la jumătate”) → `gpu-compute::math::fma_f32`, scris de noi și verificat bit cu bit față de `f32::mul_add` din `std` (sesiunea 7) |
| siboehm SGEMM_CUDA (MIT, © 2023 Simon Boehm) | MIT | doar treptele de matmul (nucleele 1 „naiv”, 2 „coalesced”, 3 „shared memory”, 4 „1D blocktiling”, 5 „2D blocktiling”, 6 „vectorize”) → `tools/nak/matmul_{naive,coalesced,smem,1d,2d,vec4}.comp`, scrise de noi în GLSL |
| Mesa 25.0.7 `src/nouveau/compiler/nak/qmd.rs` (MIT) | MIT | sesiunea 8: ce câmpuri QMD V03_00 se completează pentru memoria partajată și bariere (formula `gv100_sm_config_smem_size`), limita NVK 48 KiB → `qmd::encode_with_shared`, scris de noi |
| open-gpu-kernel-modules@570.144 (`8ec351ae`): `gsp_static_config.h` (generat → `nvidia-rm570::sysinfo`); citite ca specificație: `kernel/vgpu/rpc.c` (`rpcGspSetSystemInfo_v17_00`), `gpu_mgr.c`, `osinit.c` (`nv_encode_pci_info`), `g_gpu_nvoc.h` (`NVGPU_BUSDEVICE_*`, `OOR_ARCH_X86_64`=0), `kern_gpu_gm107.c`, `kernel_bif_gm107.c`, `dev_nv_xve.h` (`NV_PCFG`), `gpu_acpi_data.h`, `src/nvidia/Makefile` (opțiunile de compilare pentru bindgen) | MIT | sesiunea 8: structura generată automat (cu notificarea NVIDIA); ce valoare merge în fiecare câmp, scris de noi în `nvidia-ga104::gsp::sysinfo`; nouveau `gsp/r535.c` completează același subset (doar comparat, din cunoaștere generală) |
| open-gpu-kernel-modules@570.144 (`8ec351ae`): `nvos.h`, `class/cl0000.h`, `cl0080.h`, `cl2080.h`, `cl90f1.h`, `clc56f.h`, `clc7c0.h`, `clc7b5.h`, `alloc/alloc_channel.h`, `ctrl0080fifo.h`, `ctrl2080fifo.h`, `ctrl2080gpu.h`, `ctrl2080internal.h`, `ctrl90f1.h`, `g_rpc-structures.h` (generate → `nvidia-rm570::rmapi`); citite ca specificație: `kernel/gpu/fifo/engine_info.h` (indicii `engineData`), `nvstatuscodes.h` | MIT | sesiunea 9 (I8e): structurile RM_ALLOC/RM_CONTROL și parametrii lor generați automat (cu notificarea NVIDIA); învelișurile RPC au un tablou flexibil → poziția câmpurilor e calculată de clang și emisă ca constante (fără cod `unsafe`); codul care le folosește (`gsp::rmapi`, `gsp::golden`) scris de noi; modelul independent `gpu-sim::fake_rm` are pozițiile scrise de mână |
| tinygrad `runtime/support/nv/ip.py` (MIT, © the tiny corp), commit `9cf7829a6` | MIT | sesiunea 9: ordinea pasului „golden image” (`init_golden_image`, `rpc_rm_alloc`, `promote_ctx`) și valorile găsite empiric de ei (`flags=0x200320`, `internalFlags=0x1a`, `cid=3`, clientul `0xc1e00004`, +256 KiB la imaginea GR, alinierea 2 MiB a ATTRIBUTE_CB) → `gsp::golden`, `gsp::rmapi::golden_channel`, scrise de noi în Rust (nu traducere linie cu linie), marcate DEDUS |
| tinygrad `runtime/support/nv/ip.py` (MIT, © the tiny corp), `master` la 12 nov 2026 (`d415874`) | MIT | sesiunea 93 (os-295): **doar citit** ca specificație pentru `promote_ctx` al canalului de lucru (jumătatea fizică `bInitialize=1, bNonmapped=1, physAttr=0x4`, apoi cea virtuală cu `gpuVirtAddr` și restul 0; `hObject` = canalul, `engineType` 1, bufferele 0/1/2, imediat după alocarea obiectului compute) — comparat cu `gsp::user_space::open_channel` (identic ca formă). Niciun cod copiat, nicio linie schimbată din citire |
| (nicio sursă nouă) — sesiunea 94 (os-295, jurnalele 011/012) | — | **doar jurnalele plăcii** (`gpu-jurnal-011.txt`, `-012.txt`, hub) și codul nostru; limita de 5 s pentru „RISC-V activ” vine din nota deja existentă despre nova-core (`docs/GA104_INITIALIZATION.md`, pasul 6); sursele RM 570.144 (`kernel_gsp_ga102.c`, `kernel_gsp_tu102.c`) **n-au putut fi recitite din cloud** (acces refuzat) — afirmațiile despre ordinea RM din `docs/os-295_ANALIZA_011_012.md` sunt din citirile sesiunilor anterioare (C-103) și sunt marcate „de confirmat”. Codurile Booter-ului (0x88) nu sunt publicate în nicio sursă permisă |
| open-gpu-kernel-modules@570.144 (`8ec351ae`): `kernel-open/nvidia-uvm/hwref/turing/tu102/dev_mmu.h` (formatele PDE/PTE ale MMU v2; copia din `swref/published` e incompletă) → generat în `nvidia-rm570::regs::tu102_dev_mmu_ver2`; citit ca specificație: comentariul cu nivelurile din `kern_gmmu_fmt_gp10x.c` (PD3 [48:47] … PT [20:12]) și `kern_gmmu_fmt_ga10x.c` | MIT | sesiunea 10 (I9a): câmpurile generate automat (cu notificarea NVIDIA, adăugată în `LICENSE-NVIDIA`); codul care face tabelele (`nvidia-ga104::mmu`) scris de noi; cititorul independent `gpu-sim::vram` are pozițiile scrise de mână |
| tinygrad `runtime/support/nv/nvdev.py`, `runtime/support/memory.py` (MIT), commit `9cf7829a6` | MIT | sesiunea 10: doar alegerile lor, marcate DEDUS: `NO_ATS=1` în intrările de director, `KIND=6` în PTE, fereastra VA de la 64 GiB (2^44 B), mărimile nivelurilor trimise lui RM; cod scris de noi |
| open-gpu-kernel-modules@570.144 (`8ec351ae`): `swref/published/turing/tu102/dev_ram.h`, `maxwell/gm107/dev_bus.h` (doar `NV_PBUS_BAR0_WINDOW*`), `turing/tu102/dev_vm.h` (doar `FULL_PHYS_OFFSET` și `MMU_INVALIDATE*`) → generate în `nvidia-rm570::regs`; citite ca specificație: `g_kern_bus_nvoc.c`, `g_kern_gmmu_nvoc.c` (ce variantă folosește GA104), `kern_bus_gm107.c` (`kbusSetBAR0WindowVidOffset_GM107`), `kern_gmmu_tu102.c` (PDB, TRIGGER, așteptarea), `kern_gpu_tu102.c` (baza `0xb80000`) | MIT | sesiunea 11 (I9b): registrele generate automat; codul (`nvidia-ga104::vram_window`) scris de noi; modelul independent `gpu-sim::pramin` are pozițiile scrise de mână |
| open-gpu-kernel-modules@570.144 (`8ec351ae`): citite ca specificație `src/nvidia/src/kernel/gpu/gsp/arch/ampere/kernel_gsp_falcon_ga102.c` (`kgspExecuteHsFalcon_GA102`, `s_dmaTransfer_GA102`), `gpu/falcon/arch/turing/kernel_falcon_tu102.c` (`kflcnDisableCtxReq_TU102`), `gpu/gsp/kernel_gsp_fwsec.c` (descriptorul v3 → BOOT_FROM_HS), `gpu/gsp/kernel_gsp_booter.c` (`patchMeta`, `s_patchBooterUcodeSignature`), `gpu/gsp/arch/turing/kernel_gsp_frts_tu102.c` (`s_vbiosPatchInterfaceData`, `s_prepareForFwsec_TU102`: tabela de interfață, „DMEM mapper”, `FWSECLIC_FRTS_CMD` — interfața publicată de NVIDIA prin care driverul dă comanda FWSEC-ului; atingem doar tabela, `init_cmd` și bufferul de intrare, codul FWSEC rămâne opac), `inc/kernel/gpu/falcon/falcon_common.h` (`FLCN_BLK_ALIGNMENT`, `FLCN_DMEM_VA_INVALID`) | MIT | sesiunea 12 (I6/I7 DMA): ordinea și valorile registrelor, câmpurile descriptorilor și verificările lor → `nvidia-ga104::gsp::hs_load`, scris de noi în Rust (nu traducere linie cu linie); registrele erau deja generate (sesiunea 6); modelul `gpu-sim::fake_gsp` (motorul DMA, „boot ROM”-ul) scris de noi, comportament SIM |
| open-gpu-kernel-modules@570.144 (`8ec351ae`): `g_rpc-structures.h` (`rpc_set_page_directory_v1E_05`), `g_sdk-structures.h` (`NV0080_CTRL_DMA_SET_PAGE_DIRECTORY_PARAMS_v1E_05`), `ctrl0080dma.h` (comanda și câmpurile `FLAGS`) → generate în `nvidia-rm570::rmapi`; citite ca specificație: `src/nvidia/src/kernel/gpu/mem_mgr/dma.c` (`deviceCtrlCmdDmaSetPageDirectory_IMPL`), `mem_mgr/gpu_vaspace.c` (`gvaspaceExternalRootDirCommit_IMPL`: regulile de acceptare), `gpu/mem_mgr/vaspace_api.c` (traducerea `NV_VASPACE_ALLOCATION_FLAGS_*`), `nvos.h` (valorile `BIT(2)`, `BIT(3)`, `BIT(6)`, scrise de mână), `rpc_global_enums.h` (`SET_PAGE_DIRECTORY` = 54) | MIT | sesiunea 16 (I9c): structurile generate automat (cu notificarea NVIDIA); codul (`nvidia-ga104::gsp::user_space`, `rmapi::{user_vaspace_params, set_page_directory_rpc}`) scris de noi; modelul independent `gpu-sim::fake_rm` are pozițiile și regulile scrise de mână, după citirea noastră |
| tinygrad `runtime/ops_nv.py` (`PCIIface`, `NVDevice.__init__`), `runtime/support/nv/ip.py` (`rpc_rm_alloc`, `rpc_set_page_directory`), MIT, commit `9cf7829a6` | MIT | sesiunea 16: doar ordinea și valorile lor, marcate DEDUS: clientul `0xc1000000`, spațiul de adrese `vaBase=0x1000`, `vaSize=0x1fffffb000000`, `IS_EXTERNALLY_OWNED \| ENABLE_PAGE_FAULTING`, `flags=0x8`, `subDeviceId=1`, `pasid=0xffffffff`, o singură rădăcină pentru toți clienții; cod scris de noi |
| open-gpu-kernel-modules@570.144 (`8ec351ae`): `nvos.h` (`NV_CHANNEL_GROUP_ALLOCATION_PARAMETERS`, `NV_CTXSHARE_ALLOCATION_PARAMETERS`, `NV_CTXSHARE_ALLOCATION_FLAGS_SUBCONTEXT_*`), `class/cla06c.h`, `class/cl9067.h`, `class/cl2080_notification.h` (`NV2080_ENGINE_TYPE_GRAPHICS`) → generate în `nvidia-rm570::rmapi`; citite ca specificație: `src/nvidia/src/kernel/gpu/fifo/kernel_channel_group_api.c` (`kchangrpapiConstruct_IMPL`), `kernel_ctxshare.c` (`kctxshareapiConstruct_IMPL`, alegerea subcontextului), `mem_mgr/vaspace.c` (`vaspaceGetByHandleOrDeviceDefault_IMPL`), `nvstatuscodes.h` (`INSUFFICIENT_RESOURCES`) | MIT | sesiunea 17 (I9d): structurile generate automat (cu notificarea NVIDIA); codul (`nvidia-ga104::gsp::user_space::open_compute_group`, `rmapi::{channel_group_params, ctxshare_params, parse_ctxshare_reply}`) scris de noi; ordinea și valorile (GR0, `ASYNC`) după tinygrad `ops_nv.py` (MIT, `9cf7829a6`); modelul `gpu-sim::fake_rm` cu poziții și reguli scrise de mână |
| open-gpu-kernel-modules@570.144 (`8ec351ae`): `ctrl/ctrlc36f.h` (`NVC36F_CTRL_CMD_GPFIFO_GET_WORK_SUBMIT_TOKEN` + parametrii), `ctrl/ctrla06c.h` + `ctrl/ctrla06f/ctrla06fgpfifo.h` (`NVA06C_CTRL_CMD_GPFIFO_SCHEDULE`, `NVA06F_CTRL_GPFIFO_SCHEDULE_PARAMS`) → generate în `nvidia-rm570::rmapi`; citite ca specificație: `src/nvidia/src/kernel/gpu/fifo/kernel_channel.c` (`kchannelConstruct_IMPL`: regulile canalului din TSG, `_kchannelSendChannelAllocRpc`: ce primește GSP-RM, `kchannelCtrlCmdGpfifoGetWorkSubmitToken_IMPL`), `arch/ampere/kernel_fifo_ga100.c` (`kfifoGenerateWorkSubmitTokenHal_GA100`), `kernel_channel_group_api.c` (`kchangrpapiCtrlCmdGpFifoSchedule_IMPL`), `swref/published/ampere/ga100/dev_ctrl.h` (`NV_CTRL_VF_DOORBELL_VECTOR` 11:0, `_RUNLIST_ID` 22:16, scrise de mână), `nvstatuscodes.h` (`OBJECT_NOT_FOUND`) | MIT | sesiunea 19 (I9e): structurile generate automat (cu notificarea NVIDIA); codul (`nvidia-ga104::gsp::user_space::open_work_channel`, `rmapi::{work_channel_params, work_submit_token_request, parse_work_submit_token, doorbell_token, schedule_params}`) scris de noi; ordinea și valorile (canal în TSG, `engineType` 0, USERD 0x400 după inel, notificatorul de erori {0, 0xecc, 0}, handle-uri = cei 32 de biți de jos ai adresei fizice, promovarea MAIN/PM/PATCH în două jumătăți, jetonul `| runlist << 16`) după tinygrad `ops_nv.py` `_new_gpu_fifo` + `ip.py` `rpc_rm_alloc`/`promote_ctx` (MIT, `9cf7829a6`); modelul `gpu-sim::fake_rm` cu poziții și reguli scrise de mână |
| open-gpu-kernel-modules@570.144 (`8ec351ae`): `class/clc56f.h` (`Nvc56fControl_struct` / `AmpereAControlGPFifo` = USERD) → generat în `nvidia-rm570::rmapi`; citite ca specificație: `src/nvidia/src/kernel/gpu/fifo/kernel_channel.c` (când șterge RM USERD-ul), `arch/maxwell/kernel_fifo_gm107.c` (`kfifoSetupUserD_GM107`: ștergerea întregului USERD); open-gpu-doc `9fdf5c40` `classes/compute/clc7c0.h` (metodele `SET_SHADER_LOCAL_MEMORY_*`, `SET_SHADER_SHARED_MEMORY_WINDOW_*`, deja generate) | MIT | sesiunea 20 (I10a): structura generată automat (cu notificarea NVIDIA); codul (`nvidia-ga104::gsp::work_queue`, `user_space::open_work_channel_cleared`) scris de noi; modelul `gpu-sim::cmdproc` (metodele de configurare, inelul din USERD) scris de noi, pe câmpurile generate |
| tinygrad `runtime/ops_nv.py` (MIT), commit `f5b17be` (cel din studiul hub-ului) | MIT | sesiunea 20: doar ordinea și valorile, marcate DEDUS: `nv_submit` (~l. 96–101: intrarea = adresa + `len << 42` + bitul 41; slotul, apoi `GPPut`, apoi soneria), configurarea o dată pe canal (~l. 618–620: `SET_OBJECT` + ferestrele, subcanalul 1), adresele ferestrelor (~l. 602), `_ensure_has_local_memory` (~l. 680–689: rotunjirile 32 / 0x200 / 0x8000 / 0x20000, `MAX_SM_COUNT = 0xff`); cod scris de noi |
| tinygrad `runtime/ops_nv.py` (MIT), commit `f5b17be` | MIT | sesiunea 21 (I10b): doar ordinea și valorile, marcate DEDUS: `memory_barrier` (~l. 154–157: `INVALIDATE_SHADER_CACHES_NO_WFI` cu instrucțiuni, date globale, constante), `exec` pentru primul QMD (~l. 179–181: `SEND_PCAS_A` = adresa >> 8, `SEND_SIGNALING_PCAS2_B` = `PREFETCH_SCHEDULE`), `sem`/`release` (~l. 111–119: adresa, valoarea pe 64 de biți, `RELEASE` + `RELEASE_WFI_EN`); câmpurile luate din `nvidia-rm570` (`clc7c0.h`, `clc56f.h`, MIT NVIDIA); cod scris de noi (`gsp::work_queue`) |
| open-gpu-kernel-modules@570.144 (`8ec351ae`): `ctrl/ctrl2080/ctrl2080internal.h` (`NV2080_CTRL_CMD_INTERNAL_STATIC_KGR_GET_INFO`, `NV2080_CTRL_INTERNAL_STATIC_GR_GET_INFO_PARAMS`), `ctrl/ctrl0080/ctrl0080gr.h` (`NV0080_CTRL_GR_INFO_INDEX_*`, `NV0080_CTRL_GR_INFO_MAX_SIZE`) → generate în `nvidia-rm570::rmapi` (bindgen-cli 0.71.1); tinygrad `runtime/ops_nv.py` (MIT, commit `f5b17be`) `_query_gpu_info` (~l. 658–668) | MIT | sesiunea 22 (I10c): ce comandă se folosește pe calea GSP și ce 5 indici se citesc (DEDUS); codul Rust e scris de noi; valorile SIM din `gpu-sim::fake_rm::SIM_GR_INFO` alese de noi (forma unui GA104 complet), nu luate de pe placă |
| — (fără sursă nouă) | — | sesiunea 23 (I10d): refolosirea zonei de comenzi = un inel obișnuit (ideea generală, cunoscută); regula „eliberat doar după semafor” și codul (`gsp::work_queue::WorkQueue::retire`) scrise de noi; nicio sursă externă citită pentru acest pas |
| NIST FIPS 180-4 (SHA-256) și vectorii de test NIST | standard public | `sha256.rs`, scris după specificație |
| Specificația ELF-64 (System V gABI) | publică | `gsp/elf.rs` (căutarea unei secțiuni după nume; conținutul rămâne opac) |
| linux-firmware (gitlab.com/kernel-firmware), commit `050f0a1e3a76`, tag `20261019` | `LICENCE.nvidia` | **doar** mărimea și SHA-256 ale celor 4 fișiere GSP 570.144 (calculate cu `sha256sum`, fișierele șterse imediat, conținutul neanalizat; s-a verificat doar că `gsp-570.144.bin` începe cu semnătura ELF, cum face orice încărcător) |
| PCI Firmware Specification 3.0 | specificație | antetul ROM, structura PCIR (`vbios.rs`) |
| Mesa 25.0.7 (NAK, NVK, drm-shim nouveau; archive.mesa3d.org, SHA-256 verificat) | MIT | **unealtă pe gazdă** (`tools/nak/`, C-18): compilează `add64.comp` pentru SM 8.6; nimic copiat în `crates/`; din sursă s-au citit doar `nak/api.rs` (formatul dump-ului), `nvk_shader.c` și `nouveau_noop.c` (cum se alege cipul) |
| pci.ids | GPL-2.0+ / BSD-3 | date (ID-uri), citate |
| linux-firmware `LICENCE.nvidia` | proprietară, redistribuibilă condiționat | inventarul fișierelor; fișierele nu sunt incluse |
| Abramowitz & Stegun 7.1.26; metode clasice (Newton, reducerea intervalului) | domeniu public / manual | `gpu-compute::math` |
| Posturi forum cu `lspci` (bbs.archlinux.org, pciutils #151) | fără licență | doar ca fapt observat (BAR-uri) |

`fw_check.rs` (C-36) și `gpu-sim/src/fw_stick.rs` sunt scrise de noi; refolosesc citirile formatelor deja notate mai sus (`fw_hdr.rs`, `gsp/elf.rs`, `RM_RISCV_UCODE_DESC` generat). Fișierele din `fw_stick` sunt octeți inventați de noi, nu firmware NVIDIA.

Componente binare externe de care va depinde calea GSP: firmware-urile NVIDIA semnate (`firmware.rs`) și VBIOS-ul plăcii.

**C-39 (29 oct, sesiunea 18):** cuvintele de antet ale fișierelor oficiale `booter_load`, `booter_unload`, `bootloader` 570.144 (linux-firmware
`050f0a1e3a76`, descărcate temporar în sesiune, SHA-256 = cele oficiale, șterse imediat după citire) au fost citite **doar** ca să aflăm forma
containerului: antetul exterior (6 cuvinte), antetul „HS v2” (8 cuvinte) și cele 3 cuvinte la care arată, antetul de încărcare (7 cuvinte),
descriptorul `RM_RISCV_UCODE_DESC` (21 de cuvinte). Nu am citit codul, datele, semnăturile sau `meta_data`. Valorile acestor cuvinte (offset-uri și
mărimi) sunt în `gpu-sim/src/fw_stick.rs`, în jurul unor octeți de umplutură; nimic din firmware nu e în repo. Regula „pointer” pentru numărul de
semnături: nouveau `nvkm_falcon_fw_ctor_hs_v2` (MIT, din cunoaștere generală), confirmată de antetele reale.


**C-49 (1 nov, sesiunea 30):** `fuse.rs` și `i6_check.rs` sunt scrise de noi. Specificația citită (MIT, open-gpu-kernel-modules tag 570.144,
commit `8ec351ae`, clonă temporară în sesiune, nimic copiat în repo): `src/common/inc/swref/published/ampere/ga100/dev_fuse.h` (adresele
`NV_FUSE_OPT_FPF_{GSP,SEC2}_UCODE1_VERSION`), `kernel_gsp_ga100.c` / `kernel_sec2_ga100.c` (indexarea și „bitul cel mai înalt + 1”),
`g_kernel_gsp_nvoc.c` / `g_kernel_sec2_nvoc.c` (ce cipuri folosesc varianta GA100), `kernel_gsp_frts_tu102.c` și `kernel_gsp_booter.c`
(cum se folosește versiunea la alegerea semnăturii). Registrul mărimii VRAM (`0x1183A4`) era deja notat la `fb_layout.rs`.


**C-54 (1 nov, sesiunea 33):** `gpu_sim::observed`, `read_tour.rs`, `board_compare.rs`, `chain.rs` sunt scrise de noi. Datele din `observed` vin din
măsurătorile noastre (rapoartele Cyber Universe din hub `rezultate/`, 28 oct–1 nov): doar valori de registre, poziții, mărimi și id-uri; primul
cuvânt al VBIOS-ului și conținutul imaginilor **nu** sunt reproduse. Pentru temperatură am verificat antetele NVIDIA MIT 570.144 (clonă temporară,
nimic copiat): nu există `dev_therm.h` pentru Ampere → temperatura nu intră în tur până nu are o sursă confirmată.


**C-58 (3 nov, sesiunea 40):** `compare_frts` / `check_frts_detail` (`gsp/falcon.rs`), `wpr2_after_i6` (`i6_log.rs`) și opțiunea
`wpr2_hi_last_page` din `gpu-sim` sunt scrise de noi. Citit doar ca specificație, nimic copiat în repo (clone temporare în sesiune):
open-gpu-kernel-modules tag 570.144 (MIT) `src/nvidia/src/kernel/gpu/gsp/arch/turing/kernel_gsp_frts_tu102.c` (după FWSEC-FRTS: HI ≠ 0 și
LO == începutul FRTS; HI nu e comparat) și `kernel_gsp_tu102.c` (HI doar ≠ 0); Linux (GPL, **doar citit**, nimic preluat în cod):
`drivers/gpu/nova-core/fb.rs`, `fb/regs.rs`, `gsp/hal/tu102.rs` (aceeași verificare doar pe LO; comentariul numește HI „exclusiv”, dar nu-l verifică)
și nouveau `nvkm/subdev/gsp/fwsec.c` (doar afișează LO/HI). Concluzia că HI ține ultima pagină vine din măsurătoarea noastră pe placă (3 nov), nu din aceste surse.

**C-66 (4 nov, sesiunea 46):** `compare_frts_full`, `check_frts_full`, `Wpr2EndForm`, `StatusReadBar0`, `read_frts_after_i6` (`gsp/falcon.rs`),
`wpr2_after_i6_full` (`i6_log.rs`) și `FakeGsp::wpr2_hi_back` (`gpu-sim`) sunt scrise de noi. Forma „ultimul bloc de 128 KiB” vine din măsurătoarea
de pe placă (4 nov, os-222: HI `0x1ffee0`), nu dintr-o sursă; aceleași surse ca la C-58 (RM 570.144 MIT, nova-core GPL doar citit) verifică doar LO,
deci acceptă și ele această valoare. Nimic copiat.

**C-63 (3 nov, sesiunea 42):** `check_stick_journal` (`i6_log.rs`) și exemplul `verifica_jurnal_i6` sunt scrise de noi, doar procesare de text peste
funcțiile noastre existente (`parse_line`, `check_i6`, `wpr2_after_i6`); nicio sursă externă nouă.

**C-69 (4 nov, sesiunea 47):** `check_i7`, `check_i6_i7_journal` (`i6_log.rs`), `read_after_i7` / `I7Status` (`gsp/falcon.rs`), `launch_cost.rs`,
exemplul `verifica_jurnal_i7` și eliberarea cu marcă de timp din `gpu-sim::cmdproc` sunt scrise de noi. Ordinea scrierilor I7 = cea a lui I6 pe
SEC2 (RM 570.144 `kgspExecuteHsFalcon_GA102`, MIT, citit în sesiunile anterioare). Câmpurile `SEM_EXECUTE.RELEASE_TIMESTAMP` / `PAYLOAD_SIZE`
vin din `clc56f.h` (open-gpu-doc, MIT, generat în `nvidia-rm570`); poziția mărcii de timp la +8 în înregistrarea de 16 B e **DEDUS** (așa o citește
semnalul din tinygrad, MIT; necitit din nou în această sesiune) — de confirmat pe placă la prima măsurătoare. Nimic copiat.

**C-71 (5 nov, sesiunea 49):** `chain_log.rs` (verificarea pas cu pas a jurnalului `gpu lant`) și exemplul `verifica_jurnal_lant` sunt scrise de noi,
fără surse noi: regulile vin din codul nostru (`chain.rs`, `i6_log.rs`, `gsp/falcon.rs::WRITE_RANGES`, `vram_window::CONTROL_REGISTERS`).
Soneria GSP `NV_PGSP_QUEUE_HEAD(0)` = 0x110c00 e din `dev_gsp.h` (570.144, MIT, generat în `nvidia-rm570`); soneria canalului de lucru
0xbb0090 rămâne **DEDUS** (nouveau/tinygrad, citite ca specificație în sesiunile anterioare), doar numărată de unealtă. Nimic copiat.

**C-75 (5 nov, sesiunea 51):** `gsp_reset_into_riscv`, `write_libos_mailboxes`, `booter_load_dma_libos` (`gsp/falcon.rs`), `check_i7_libos`
(`i6_log.rs`) și modelul „cutii GSP blocate după FWSEC” din `gpu-sim::fake_gsp` sunt scrise de noi. Ordinea (resetarea GSP în modul RISC-V după
FWSEC, BCR_CTRL = 0x111, cutiile 0/1 = adresa LibOS înainte de Booter, `FALCON_OS` = 570144 după) e citită ca specificație din nouveau
(`r535_gsp_oneinit` / `ga102_gsp_reset`, MIT, citite în sesiunile anterioare) și RM 570.144 (MIT); câmpurile BCR_CTRL din `dev_riscv_pri.h`
(570.144, MIT, generat în `nvidia-rm570`). Efectul de pe placă (cutia scrisă `0x09c71000`, citită 0) e **OBSERVAT** în jurnalul din 5 nov;
cauza (FWSEC lasă GSP blocat până la resetare) e **DEDUS**. Nimic copiat.

**Sesiunea 52 (5 nov), soneria canalului de lucru:** `push::DOORBELL` (`crates/nvidia-ga104/src/push.rs`) și testul T90 sunt scrise de noi; valoarea
vine din constanta generată automat `tu102_dev_vm::NV_VIRTUAL_FUNCTION_DOORBELL` (open-gpu-kernel-modules 570.144, `tu102/dev_vm.h`, MIT, notificarea
în `crates/nvidia-rm570/LICENSE-NVIDIA`; generatorul `tools/gen-rm570/gen_regs.py` primește un prefix nou). Că GA104 o folosește așa e citit ca
specificație din RM 570.144 (MIT): `kernel_fifo_ga100.c` (`kfifoUpdateUsermodeDoorbell_GA100`), `kernel_fifo_tu102.c`
(`kfifoUpdateUsermodeDoorbell_TU102`), `kern_gpu_tu102.c` (`gpuGetVirtRegPhysOffset_TU102`), `g_kernel_fifo_nvoc.c` / `g_gpu_nvoc.h` (legarea HAL).
0xbb0090 trece din **DEDUS** în **DOCUMENTAT** (antet + cale RM); pe placă rămâne neobservat. Nimic copiat.

**Sesiunea 57 (6 nov), C-83 I11–I12:** `transfer.rs`, `q8_matvec.rs`, `gpu-compute/src/q8.rs`, adăugirile din `mmu.rs`, `gpu-sim` (`translate_any`,
`copy_b128_on_cpu`, `q8_matvec_on_cpu`) și programele GLSL `tools/nak/copy_b128.comp`, `tools/nak/q8_matvec.comp` sunt scrise de noi. Câmpurile PTE pentru RAM
(`APERTURE_SYSTEM_COHERENT_MEMORY`, `ADDRESS_SYS`) vin din constantele generate din `dev_mmu.h` (570.144, MIT); al doilea cititor din `gpu-sim` e scris de
mână din același antet. Alegerea „coerent + VOL” pentru RAM: DEDUS din felul în care RM 570.144 și tinygrad `nvdev.py` (MIT) mapează memoria de sistem
(citite ca specificație). Formatul Q8_0 e al lui Kronos (KGR1, `integrare/kronos/format_greutati.md` din hub); ordinea motorului OS e citită din
`cyber-universe/user/src/kgr.rs` (proiectul nostru). Binarele `.sm86.sass` sunt ieșirea Mesa NAK 25.0.7 (MIT) pe programele noastre. Nimic copiat.

**Sesiunea 58 (6 nov), C-83 I13:** `kronos_layer.rs`, `gpu-compute/src/layer.rs`, modelele din `gpu-sim` și programele `tools/nak/{vadd,layernorm,attention,gelu}.comp`
sunt scrise de noi. Ordinea stratului e cea din formatul KGR1 al lui Kronos (hub `integrare/kronos/format_greutati.md`); erf după Abramowitz & Stegun
7.1.26 (formula publică, ca `gpu_compute::math::erf`). I14: `kronos_model.rs`, `gpu-compute/src/model.rs`, `tools/nak/embed.comp`,
`WorkQueue::submit_fenced`, scrise de noi; ordinea simbolului din KGR1 (hub). Nimic copiat.

**Sesiunea 59 (6 nov), C-83 I14 — vectorii de control KGR1:** testele `kgr1_*` din `crates/nvidia-ga104/tests/sim_i14_token.rs`, cititorul KGR1 din
test (după `format_greutati.md` din hub) și `tools/vectori-kgr1/pregateste.py` sunt scrise de noi. Vectorii înșiși (hub-ul nostru privat,
`integrare/kronos/vectori/`, Kronos S39/C-24) **nu intră în acest repo public**: scriptul îi pregătește local, în `target/` (necomis), iar testele îi
citesc din `KGR1_VECTORI`. Nimic străin.

**Sesiunea 73 (9 nov), C-91 — câmpurile secvențiatorului:** `decode_sequencer` (`gsp/rpc.rs`) citește acum comenzile până la `cmdIndex`
(cuvintele umplute), iar `bufferSizeDWord` e doar capacitatea bufferului GSP. Sens citit (ca specificație, fără cod copiat) în nouveau
`r535_gsp_msg_run_cpu_sequencer` și tinygrad `run_cpu_seq` (MIT), plus diagnosticul sesiunii OS din hub
(`idei/2026-11-09_C-91-cauza-secventiator.md`, care citează driverul public NVIDIA 570.144 `kgspExecuteSequencerBuffer_IMPL`, MIT). Cod scris de noi.

**Sesiunea 77 (10 nov), C-95 — notiță GA106 (parcat, fără cod):** `docs/GA106_LAPTOP.md` citează open-gpu-kernel-modules@570.144
(`README.md`, `g_chips2halspec_nvoc.c`, `g_kernel_gsp_nvoc.c`, MIT, citite ca specificație), `WHENCE` din linux-firmware
(`nvidia/ga106/gsp -> ../ga102/gsp`; niciun fișier firmware descărcat) și pci.ids (citat). Codul GA106 scris în `256a1f2` a fost retras.

**Sesiunea 78 (10 nov), C-96 — resetul GSP în secvențiator:** open-gpu-kernel-modules@570.144 (`8ec351ae`, MIT), citite ca
specificație: `kernel/gpu/gsp/kernel_gsp.c` (`kgspExecuteSequencerBuffer_IMPL`: `GPU_REG_RD32` în bucle, fără oprire la `0xbadf…`),
`gsp/arch/ampere/kernel_gsp_ga102.c` (`kgspExecuteSequencerCommand_GA102`: CORE_RESUME = `kflcnResetIntoRiscv` → LibOS → SEC2),
`gsp/arch/turing/kernel_gsp_tu102.c` (`kgspResetHw_TU102`: citiri de propagare), `falcon/arch/turing/kernel_falcon_tu102.c`
(`kflcnReset_TU102`, `kflcnDisableCtxReq_TU102`), `falcon/arch/ampere/kernel_falcon_ga102.c` (`kflcnPreResetWait_GA102`,
`kflcnWaitForResetToFinish_GA102`, `kflcnSwitchToFalcon_GA102`, `kflcnIsRiscvSelected_GA102`: „BCR_CTRL poate fi blocat la citire”),
`inc/kernel/gpu/falcon/falcon_common.h` (`FLCN_RESET_PROPAGATION_DELAY_COUNT` = 10), `swref/published/maxwell/gm107/dev_pri_ringstation_sys.h`
(codurile `NV_PPRIV_SYS_PRI_ERROR_CODE_*`: `0xBADF50` = `FECS_PRI_CLIENT_ERR`). Nouveau (`r535_gsp_msg_run_cpu_sequencer`, `ga102_gsp_reset`)
din cunoaștere generală, doar pentru comparație. Cod scris de noi (`falcon::poll`, `engine_reset_pulse`, `switch_to_falcon`, `sequencer::poll_reg`,
`SeqCmd::event`); modelul `gpu-sim::fake_gsp` (`PRIV_ERROR_AFTER_RESET`) scris de noi din jurnalul plăcii (`gpu-jurnal-004.txt`).

**Sesiunea 80 (10 nov), C-99 — pornirea SEC2 prin `CPUCTL_ALIAS` la CORE_RESUME:** open-gpu-kernel-modules@570.144 (`8ec351ae`, MIT),
citite ca specificație: `falcon/arch/turing/kernel_falcon_tu102.c` (`kflcnStartCpu_TU102`: dacă `CPUCTL.ALIAS_EN` e pus, `STARTCPU` se scrie
în `NV_PFALCON_FALCON_CPUCTL_ALIAS`), `gsp/arch/ampere/kernel_gsp_ga102.c` (`kgspExecuteSequencerCommand_GA102`, CORE_RESUME: `kflcnStartCpu_HAL`
pe falcon-ul SEC2). Nouveau (`nvkm_falcon_v1_start`: același test al bitului 6) din cunoaștere generală, doar pentru comparație. Faptul de pe placă
(`gpu-jurnal-005.txt`: `0x840100: 0x50 <- 0x2 => 0x50` după Booter) e OBSERVAT; de ce programul HS lasă `ALIAS_EN` pus e DEDUS. Cod scris de noi
(`falcon::start_cpu`, `wait_riscv_active_for`, `sequencer::LONG_WAIT_US`, `Clock::abort_requested`, `GpuError::Aborted`, `Event::Waited`); modelul
`gpu-sim::fake_gsp` (`Falcon::alias_en`) scris de noi din jurnalul plăcii.

**C-102 (sesiunea 84, 11 noiembrie 2026).** Citite ca specificație din open-gpu-kernel-modules@570.144 (MIT): `gsp/arch/ampere/kernel_gsp_ga102.c`
(`kgspExecuteSequencerCommand_GA102`, CORE_RESUME: ordinea reset-în-RISC-V → argumentele LibOS → `kflcnStartCpu` SEC2 → `_kgspIsReloadCompleted`
(doar citirea `NV_PGC6_BSI_SECURE_SCRATCH_14`, bit `BOOT_STAGE_3_HANDOFF`) → SEC2 MAILBOX0 → `FALCON_OS` → RISC-V activ; fișierul nu scrie și nu șterge
SCRATCH_14 — de aceea nici noi), `nvstatuscodes.h` (`0x59` = `NV_ERR_OPERATING_SYSTEM`). Nouveau (`r535_gsp_msg_recv`: orice `rpc_result ≠ 0` e eroare,
`r535_gsp_msgq_wait`: pointerul de scriere nu e verificat) din cunoaștere generală, pentru comparație. Faptele de pe placă (`gpu-jurnal-006.txt`:
„2 µs” la predarea etapei 3 și la RISC-V activ, apoi `0x59` după ~150 ms) sunt OBSERVATE; cele două surse posibile ale lui `0x59` sunt DEDUSE din codul
nostru. Cod scris de noi (`Event::Value`, `falcon::{RESUME_STATE_READS, journal_resume_state}`, rândurile din `sequencer`); doar citiri, nicio scriere nouă.

**După os-279 (sesiunea 85, 11 noiembrie 2026).** Citite ca specificație din open-gpu-kernel-modules@570.144 (MIT): `kernel/gpu/gsp/kernel_gsp.c`
(`_kgspRpcGspPostNocatRecord`: payload-ul lui `GSP_POST_NOCAT_RECORD` citit ca `NV2080CtrlNocatJournalInsertRecord`; `_kgspRpcUcodeLibosPrint`:
`ucodeEngDesc`/`libosPrintBuf`, buffer PMU dat lui `kpmuLogBuf`; `kgspWaitForRmInitDone_IMPL`: `rpc_result` al lui `GSP_INIT_DONE` e un `NV_STATUS`),
`kernel/diagnostics/journal.c` (`_rcdbNocatReportAssert`: la ASSERT `source = "ASSERT"`, `errorCode` = adresa − `loadAddress` | linia << 32, `diagBuffer` =
`RM_NOCAT_ASSERT_DIAG_BUFFER`; `rcdbNocatInsertNocatError`: „RC Error”; `rcdbNocatInsertEngineError`/`TDRError`), `generated/g_journal_nvoc.h`
(`ASSERT_CALL_STACK_SIZE` = 10, `RM_NOCAT_ASSERT_DIAG_BUFFER`), `sdk/nvidia/inc/ctrl/ctrl2080/ctrl2080nvd.h` (`NV2080CtrlNocatJournalInsertRecord`,
`NV2080_NOCAT_JOURNAL_REC_TYPE_*`, `MAX_STR_LEN` 65, `MAX_DIAG_BUFFER` 1024), `generated/g_rpc-structures.h` (`rpc_gsp_post_nocat_record_v01_00`,
`rpc_ucode_libos_print_v1E_08`). **Generate** (C-20 B, unealtă nouă `tools/gen-rm570/gen_names.py`, doar biblioteca standard Python) din
`nvstatuscodes.h` și `inc/kernel/vgpu/rpc_global_enums.h` → `crates/nvidia-rm570/src/names.rs` (143 coduri `NV_STATUS` cu descrierea NVIDIA, 228 funcții
RPC, 36 evenimente; cu notificarea MIT NVIDIA). Offset-urile înregistrării NOCAT scrise de mână în `gsp/nocat.rs` au fost verificate cu `offsetof()` într-un
program C de probă (în afara repo-ului). Faptele de pe placă (`gpu-jurnal-007.txt`: 843 × `0x1020`, 2 × `0x100c`, `rpc_result 0x59`, SCRATCH_14 pus abia după
`0x840130 ← 2`) sunt OBSERVATE; că cele 843 sunt plângeri ale RM-ului de pe GSP în timpul inițializării e DEDUS din rolul mesajului în RM. Cod scris de noi
(`gsp::nocat`, `Event::Text`, `sequencer::{journal_skipped, journal_skipped_summary}`, `nvidia_rm570::{status_name, status_text, rpc_name}`); doar citiri
din mesaje primite, nicio scriere nouă. Conținutul `libosPrintBuf` și al `diagBuffer` (altul decât stiva asserturilor) nu se decodează: ar cere simbolurile
firmware-ului.

**C-103 (sesiunea 85, 11 noiembrie 2026) — WPR2 coborât fără repornire.** Citite ca specificație din open-gpu-kernel-modules@570.144 (MIT):
`kernel/gpu/gsp/arch/turing/kernel_gsp_booter_tu102.c` (`s_executeBooterUcode_TU102`, `kgspExecuteBooterLoad_TU102`, `kgspExecuteBooterUnloadIfNeeded_TU102`:
reset SEC2, mailbox (0xFF, 0xFF) la descărcarea normală / `GspFwSRMeta` la suspend / `0xdeaddead` la GC6, mailbox0 trebuie 0, apoi `kgspIsWpr2Up` trebuie fals),
`kernel_gsp_tu102.c` (`kgspTeardown_TU102`: reset GSP → FWSEC-SB → Booter Unload; `kgspIsWpr2Up_TU102`: câmpul VAL din `NV_PFB_PRI_MMU_WPR2_ADDR_HI`;
`kgspWaitForProcessorSuspend_TU102`: GSP MAILBOX0 == 0x80000000), `kernel_gsp.c` (`_kgspBootGspRm`: refuză pornirea cu WPR2 ridicat; `kgspUnloadRm`: ordinea
`UNLOADING_GUEST_DRIVER` → suspend → teardown). PCIe Base Specification (Device Capabilities/Control, PM `No_Soft_Reset`, Bridge Control) din cunoaștere generală a
standardului. Nouveau (`booter.unload` la `fini`/`init`) și comportamentul plăcilor GeForce față de FLR — cunoaștere generală, doar pentru comparație, marcate așa în
`docs/C-103_RESET_PLACA.md`. Cod scris de noi (`falcon::{booter_unload, booter_unload_dma, BOOTER_UNLOAD_MAILBOX, Unloaded}`, `gpu_pci::caps::{reset_capabilities,
ResetCapabilities}`); modelul `gpu-sim::fake_gsp` (`SimUcode::BooterUnload`, `SimFault::Wpr2Stuck`) scris de noi, reproduce doar efectul vizibil (WPR2 → 0), nu interiorul
programului semnat.

**După os-280 (sesiunea 87, 11 noiembrie 2026) — `bDmemStack`.** Citite ca specificație din open-gpu-kernel-modules@570.144 (MIT): `kernel/gpu/gsp/kernel_gsp.c`
(`kgspPopulateGspRmInitArgs_IMPL`: `bDmemStack` = adevărat când `RmGspStackPlacement` e implicit (0) sau DMEM (2); `kgspSetupLibosInitArgs_IMPL`: regiunile
LOG*/RMARGS; `_kgspCalculateFwHeapSize`), `interface/nvrm_registry.h` (`NV_REG_STR_RM_GSP_STACK_PLACEMENT`: 0 implicit, 1 stiva LibOS, 2 DMEM),
`libraries/utils/nvassert.c` (pe `NVOS_IS_LIBOS`: `nvAssertFailed` → `rcdbRmAssert(NV_RM_ASSERT_UNKNOWN_LINE_NUM, adresa de întoarcere)`; codul `NV_ASSERT_OK` merge
în `level`, nu în NOCAT), `sdk/nvidia/inc/rmcd.h` (`NV_RM_ASSERT_UNKNOWN_LINE_NUM` = 0), `kernel/diagnostics/journal.c` (`_rcdbRmAssert`, deduplicarea din
`_rcdbNocatReportAssert`), `kernel/vgpu/rpc.c` (`rpcGspSetSystemInfo_v17_00`, lista câmpurilor), `kernel_gsp_tu102.c` (`kgspCalculateFbLayout_TU102`),
`generated/g_kernel_gsp_nvoc.c` (`bBootGspRmWithBoostClocks` adevărat pe TU10x/GA10x/AD10x), `inc/kernel/gpu/gsp/gsp_fw_heap.h` (constantele heap-ului),
`arch/nvalloc/unix/src/osinit.c` (cheile de registru). Linux nouveau (MIT, `nvkm/subdev/gsp/rm/r570/gsp.c`: `r570_gsp_set_rmargs` → `bDmemStack = 1`,
`r570_gsp_set_system_info`, `r570_gsp_drop_post_nocat_record`; `rm/r570/rm.c`: parametrii heap-ului pe GA102; `tu102.c`: `tu102_gsp_wpr_meta_init`,
`tu102_gsp_oneinit`; `rm/r535/gsp.c`: cheile de registru; `ga102.c`: 570.144 = varianta 1) și tinygrad `runtime/support/nv/ip.py` (MIT: `GSP_ARGUMENTS_CACHED(bDmemStack=True)`,
regiunile LibOS, `init_wpr_meta`, `rpc_set_gsp_system_info` cu `bIsPassthru=1` și fără ACPI) și Linux nova-core (GPL, doar citit: `gsp/fw.rs` `GSP_ARGUMENTS_CACHED { bDmemStack: 1 }`,
`gsp/fw/commands.rs` `GspSetSystemInfo` fără ACPI, `bIsPrimary 0`; `gsp/commands.rs` cheile de registru; `gsp.rs` cele 4 regiuni LibOS; `fb.rs`) — **citite pentru comparație**, nimic copiat; tabelul comparativ e în
`docs/os-280_ANALIZA_0x59.md` §3. Faptele din `gpu-jurnal-008.txt` (844 ASSERT, toate înainte de secvențiator, adrese diferite crescătoare, linia 0, stiva goală,
`rpc_result 0x59`) sunt OBSERVATE; că `bDmemStack = 0` e cauza lui 0x59 e **DEDUS** (singura diferență față de toate cele patru drivere), de probat la os-282.
Schimbarea de cod: un octet în `gsp::libos::encode_gsp_args` (`B_DMEM_STACK_OFF` = 48, verificat cu bindgen în `tests/rm570_crosscheck.rs`).

**După os-283 (sesiunea 88, 11 noiembrie 2026) — ordinea RPC-urilor de pornire; variantele C-108.** Citite ca specificație din open-gpu-kernel-modules@570.144
(`8ec351a`, MIT, clonă parțială în sesiune): `kernel/gpu/gsp/kernel_gsp.c` (`kgspQueueAsyncInitRpcs_IMPL` + comentariul „process in early boot before OBJGPU is
created”; `kgspInitRm_IMPL`: mesajele puse înainte de `_kgspBootGspRm` → `kgspBootstrap_HAL`; `_kgspRpcSendMessage` → `kgspSetCmdQueueHead_HAL` la fiecare
trimitere; `kgspExecuteSequencerBuffer_IMPL`), `kernel_gsp_gh100.c` (`_kgspEstablishSpdmSession`: „GSP-RM may skip these RPCs if not sent in time”),
`arch/ampere/kernel_gsp_ga102.c` (`kgspExecuteSequencerCommand_GA102`, CORE_RESUME), `kernel/vgpu/rpc.c` (`rpcGspSetSystemInfo_v17_00`: `hostPageSize =
osGetPageSize()`, toate câmpurile), `arch/nvalloc/unix/src/osinit.c` (`nv_encode_pci_info` → `gpuEncodeDomainBusDevice`), `kernel_gsp_tu102.c`
(`GSP_FW_FLAGS_CLOCK_BOOST`, `bBootGspRmWithBoostClocks`), `gsp_fw_wpr_meta.h` (`GSP_FW_FLAGS_*`), `platform/chipset/chipset_pcie.c` și celelalte locuri cu
`NV_ERR_OPERATING_SYSTEM` (doar pentru a vedea că sunt căi de pe CPU), `gsp_static_config.h` (`GspSystemInfo`, `bIsPassthru` la 840 — verificat cu bindgen).
tinygrad `e51dc1d` (MIT, clonă parțială): `runtime/support/nv/ip.py` (`GSP.init_sw`: „Prefill cmd queue with info for gsp to start”; `rpc_set_gsp_system_info`:
`bIsPassthru=1`, fără `bIsPrimary`/`hostPageSize`; `rpc_set_registry_table`: 2 chei; `NVRpcQueue._send_rpc_record`: `NV_PGSP_QUEUE_HEAD[0].write(0)` la fiecare
trimitere; `Falcon.init_hw`: FRTS, Booter, `FALCON_OS = 0`). Linux `067c5c2` (GPL, **doar citite**, clonă parțială): nouveau `nvkm/subdev/gsp/rm/r535/gsp.c`
(`r535_gsp_oneinit`: `set_system_info` + `r535_gsp_rpc_set_registry` înainte de pornire; `r535_gsp_init`: `FALCON_OS`, poll `GSP_INIT_DONE`), `rm/r535/rpc.c`
(`0xc00 ← 0` la push), `rm/r570/gsp.c` (`r570_gsp_set_system_info`: fără `hostPageSize`, `bIsPrimary` după VGA), `tu102.c` (`tu102_gsp_init`: Booter apoi
`r535_gsp_init`); nova-core `gsp/boot.rs` (`Gsp::boot`: `SetSystemInfo`, `SetRegistry` înainte de `hal.boot`), `gsp/cmdq.rs` (`NV_PGSP_QUEUE_HEAD`),
`gsp/fw/commands.rs` (`bIsPrimary: 0`, fără `hostPageSize`), `gsp/commands.rs` (3 chei de registru). Nimic copiat; valorile și ordinea sunt în
`gsp/variants.rs` cu sursa lângă fiecare variantă; tabelul în `docs/os-283_ANALIZA_ORDINE_RPC.md`. Faptele din `gpu-jurnal-009.txt` (identic cu 008) sunt
OBSERVATE; că ordinea e cauza lui 0x59 e **DEDUS** (din comentariile RM), de probat joi. Modelul `SimGsp::start_early_boot` e scris de noi după acele comentarii.

**Sesiunea 95 (12 nov, noaptea), C-116 — GA106 doar-citire + „trezirea minimă”:** codul GA106 din `256a1f2` (sesiunea 77) e repus, adaptat (doar adăugări:
`ids::{GA106_MODELS, Family, lookup_ga106, lookup_ampere, LAPTOP_DEVICE}`, `boot::CHIP_GA106`, `probe::{probe_config_ampere, probe_mmio_family, level0_ampere,
MEMORY_SPACE_OFF}`, `read_tour::{run_tour_ampere, run_tour_awake, Wake, FACT_WAKE}`); sursele lui rămân cele din sesiunea 77 (open-gpu-kernel-modules@570.144
README + `g_chips2halspec_nvoc.c`, MIT; pci.ids citat; `linux-firmware` `WHENCE`). Faptele despre laptop (`10de:2520`, Command 0, BAR0 @0x85000000, BAR1 8 GiB,
BAR3 32 MiB, D0) sunt OBSERVATE de OS pe 12 nov (hub `rezultate/2026-11-11-laptop-asus-tuf/REZULTAT.md`); subsistemul, bazele BAR1/BAR3, legătura și I/O-ul din
`SimConfig::ga106_laptop_asus` sunt SIM. `gpu_pci::mem_space` (bitul Memory Space din Command): PCI Local Bus Specification 3.0 §6.2.2; Linux `pci_enable_device` /
`pci_enable_resources` (GPL, **citit ca specificație**: pune exact acest bit după ce verifică alocarea BAR-urilor) — nouveau, nova-core și tinygrad se bazează
pe el înainte de orice acces la BAR0. Cod original, după modelul `bus_master.rs` (D-013). Nimic rulat pe laptop.

**Sesiunea 96 (13 nov, ~01:30), C-116/C-117 — versiunea VBIOS din jetonul BIT 'i' + faptele UHL ale laptopului:** `vbios::{BiosVersion, BIT_TOKEN_BIOS_INFO,
read_bios_version}` (doar adăugări; câmp nou `VbiosLayout.version`). Formatul jetonului 'i' (0x69): 5 octeți la offset-ul jetonului, primii 4 un întreg little-endian
(octetul 3 = major, 2 = cip, 1 = minor, 0 = micro), al 5-lea = patch, tipărit `%02x.%02x.%02x.%02x.%02x` — după nouveau `nvkm/subdev/bios/base.c` (`nvkm_bios_new`, MIT),
**citit ca specificație, din memorie**: sursa n-a putut fi recitită din cloud în această sesiune (acces refuzat la fișier) → marcat „de confirmat” în `GA106_LAPTOP.md` §5;
confirmarea reală vine din primul jurnal cu `vbios.versiune` (trebuie să coincidă cu șirul raportat de Windows: `94.04.65.00.3e` pe workstation, `94.06.1E.00.23` pe laptop).
Că jetonul 'i' există pe placa noastră e OBSERVAT (`0x69v2` în `vbios.bit.jetoane`, 29 oct); lungimea și datele lui sunt SIM în `observed_rom` (5 octeți cu versiunea din
Windows, nu copiați dintr-un ROM). `gpu_sim::laptop`: toate constantele sunt din raportul UHL al lui Adrian (Windows, NVML/CUDA, doar citire; hub
`rezultate/2026-11-11-laptop-asus-tuf/REZULTAT.md`, 12 nov 22:14 și 23:03) și din `gpu identifica` (OS, os-295); nimic citit de driverul nostru de pe laptop.
UHL e unealta lui Adrian (repo privat `Adriancu91/uhl`; nu l-am citit; aici sunt doar cifrele publicate în hub, fără serii).

**Sesiunea 98 (13 nov, ~13:30), C-116 pasul 3 (partea de citire) — primul jurnal de pe laptop în model:** `gpu_sim::observed_laptop` (doar adăugări) ține valorile
OBSERVATE pe ASUS TUF F15 (GA106) la `gpu tur --trezire`, 13 nov ~07:35, `os-301` (cyber-gpu `835f24d`): cele 8 citiri BAR0 din `gpu-jurnal-003.txt` (hub
`rezultate/2026-11-13-laptop-asus-os301/`, fișier de 24 de rânduri, copiat întreg în testul `board_compare`) și cele 39 de fapte vizibile în poza `gpu-tur-trezire.jpg`
(transcrise de noi; primele rânduri ale ecranului, tăiate în poză, nu sunt ghicite). Ca la `observed` (workstation): se păstrează doar valori de registre, poziții, mărimi
și ID-uri — **conținutul VBIOS-ului nu e copiat** (ROM sintetic cu aceeași așezare; primul cuvânt PROM `0xeb7faa55`, jurnalizat de OS, nu e reprodus), niciun octet de
firmware NVIDIA. Ordinea octeților din jetonul BIT 'i' (`read_bios_version`, scrisă în sesiunea 96 „din memorie, de confirmat”) e acum **confirmată pe placă**:
`vbios.versiune = 94.06.1e.00.23` = șirul raportat de Windows/NVML pentru aceeași placă. `board_compare::line_facts` citește și rândurile de jurnal de pe stick
(`gpu: bar0 r …`), nu doar pe cele din rapoartele OS (`jurnal cyber-gpu:`). Nimic nou rulat pe laptop sau pe workstation în această sesiune.

**Sesiunea 103 (16 nov, ~19:30), C-125 (I9 V2 + V3, după `os-302`):** surse citite ca specificație, nimic copiat — NVIDIA open-gpu-kernel-modules **570.144**
(MIT): `ctrl2080gr.h` (`NV2080_CTRL_CMD_GR_GET_CTX_BUFFER_INFO`, structurile — **generate automat** cu bindgen în `nvidia-rm570/src/rmapi.rs`, ca restul
crate-ului, notificarea MIT păstrată), `g_subdevice_nvoc.c` (flagurile comenzilor: `0x8000` pentru `GR_GET_CTX_BUFFER_INFO`, `0x10244` pentru `PROMOTE_CTX`),
`control.h` (sensul flagurilor), `nv_gpu_ops.c` (`nvGpuOpsBindChannelResources`, cum trimite UVM jumătatea virtuală), `kernel_graphics_context.c`
(`kgrctxPreparePromoteCtxBuffer`, promovarea PM); `dev_mmu.h` (bitul `PRIVILEGE` = 5, deja generat). nouveau `r535/gr.c` (MIT): `priv = 1` pentru bufferele de
context — doar ideea; codul nostru (`mmu::encode_pte_privileged`, `user_space::open_work_channel_with`, `I9Report`) e scris de noi. Ideea și sursele au venit din
hub (`idei/2026-11-16_cercetare-i9-promote-ctx.md`, centrul de comandă). Valorile din `gpu_sim::fake_rm::SIM_CHANNEL_CTX_BUFFERS` sunt SIM. Niciun octet de
firmware NVIDIA citit sau copiat; adresele NOCAT din jurnal nu au fost căutate în firmware.


**Sesiunea 107 (17 nov, ~14:20), C-133 (I9 după `os-303`: promovarea într-un singur mesaj):** surse citite ca specificație, **nimic copiat** — NVIDIA
open-gpu-kernel-modules **570.144** (MIT): `kernel_graphics_object.c` (`kgrobjPromoteContext`, `kgrobjGetPromoteIds_FWCLIENT`: un mesaj cu PA + VA, lista
pentru compute), `kernel_graphics_context.c` (`kgrctxPrepareInitializeCtxBuffer`, `kgrctxPreparePromoteCtxBuffer`, `kgrctxGetCtxBuffers` — de unde vine
`NV_ERR_INVALID_OBJECT` la V3, `kgrctxFromKernelChannel`), `kernel_graphics.c` (`subdeviceCtrlCmdKGrGetCtxBufferInfo`), `kernel_channel.c`
(`kchannelConstruct`: tipurile notificatoarelor din `internalFlags`, regulile TSG / context share), `kernel_channel_group_api.c` (`hVASpace` în TSG, modul
legacy), `kernel_ctxshare.c`, `nv_gpu_ops.c` (`_nvGpuOpsRetainChannelResources`, `nvGpuOpsBindChannelResources`), `alloc_free.c` (constructorul înaintea
RPC-ului de alocare), `vaddr_list.c` (când întoarce `NV_ERR_OBJECT_NOT_FOUND`), `ctrl2080gpu.h` (descrierea `PROMOTE_CTX`), `nvos.h`, `g_kernel_channel_nvoc.h`,
`g_gpu_vaspace_nvoc.h`. **nouveau** (Linux, ramura principală la 17 nov 2026; fișierele citite poartă licența MIT): `nvkm/subdev/gsp/rm/r535/gr.c`
(`r535_gr_promote_ctx`: un mesaj, PA + VA, globalele doar cu VA pentru un canal care nu e cel golden, fără PM), `r535/fifo.c` + `r570/fifo.c` (alocarea
canalului), `r535/vmm.c` (spațiul de adrese), `r570/gr.c` (promovarea înaintea obiectului), `nvkm/subdev/gsp/ga102.c` (GA104 ↔ firmware 570.144).
**tinygrad** (MIT, commit `9ecf871`): `runtime/ops_nv.py`, `runtime/support/nv/ip.py`, `nvdev.py` (firmware 570.190). Codul nostru
(`user_space::open_work_channel_c133`, `C133Report`, `gpu_sim::fake_rm::PromoteRule`) e scris de noi; `PromoteRule` modelează **ipoteze**, nu placa. Niciun
octet de firmware NVIDIA citit sau copiat; adresele NOCAT din jurnale nu au fost căutate în firmware. Niciun cod GPL citit pentru această sesiune.

**Sesiunea 108 (17 nov, ~16:20), C-135 (I9 pasul 5 după `os-304`: jetonul refuzat cu `0x40`):** surse citite ca specificație, **nimic copiat** — NVIDIA
open-gpu-kernel-modules **570.144** (MIT): `kernel_channel.c` (`kchannelCtrlCmdGpfifoGetWorkSubmitToken`, `kchannelNotifyWorkSubmitToken`,
`kchannelUpdateNotifierMem` — verificarea `ADDR_FBMEM` de pe GSP; `kchannelConstruct`: contextul de erori, tipurile din `internalFlags`, runlist-ul la
alocare; `_kchannelSendChannelAllocRpc`: ChID-ul ales de RM-ul de pe CPU, `errorNotifierMem`; `kchannelCtrlCmdGpFifoSchedule`, `kchannelIsSchedulable`,
`kchannelCtrlCmdBind`), `kernel_channel_group_api.c` (`kchangrpapiCtrlCmdGpFifoSchedule`), `kernel_fifo.c` (`kfifoGenerateWorkSubmitToken`,
`kfifoChidMgrAllocChid`, `KFIFO_NUM_GSP_RESERVED_CHANNELS`), `arch/ampere/kernel_fifo_ga100.c` (`kfifoGenerateWorkSubmitTokenHal_GA100`),
`arch/maxwell/kernel_channel_gm107.c` (`kchannelAllocHwID`), `arch/maxwell/kernel_fifo_gm107.c` (`kfifoRunlistSetId`), `g_kernel_channel_nvoc.c`
(flagurile comenzilor: `0x8` pentru jeton, `0xc0` pentru varianta internă), `g_kernel_channel_nvoc.h` (`ErrorNotifierType`,
`NV_KERNELCHANNEL_ALLOC_INTERNALFLAGS_*`), `inc/kernel/rmapi/control.h` (sensul flagurilor), `alloc/alloc_channel.h` (`NVOS04_FLAGS_*`),
`ctrl/ctrlc36f.h`, `ctrl/ctrla06c.h`, `ctrl/ctrla06f/ctrla06fgpfifo.h`, `vgpu/rpc.c` (`rpcRmApiControl_GSP`: starea comenzii e în câmpul `status` al
răspunsului), `nvstatuscodes.h`. Pentru comparația de versiuni: aceleași fișiere de la eticheta **570.190** (canale, fifo, GR: identice cu 570.144).
**nouveau** (Linux, ramura principală la 17 nov 2026; fișiere cu licența MIT): `nvkm/subdev/gsp/rm/r535/fifo.c` + `r570/fifo.c` (`internalFlags` cu
notificatoare `NONE`, ChID prin `USERD_INDEX`, `BIND` + `GPFIFO_SCHEDULE` pe canal), `nvkm/engine/fifo/ga100.c` + `tu102.c` (jetonul soneriei calculat de
driver), `r535/ctrl.c` + `r535/alloc.c` (citesc `rpc->status`, pe lângă `rpc_result` din `r535/rpc.c`). **tinygrad** (MIT): `runtime/support/nv/ip.py` și `runtime/ops_nv.py` la commit-urile `9cf7829a6`
(26 oct, firmware 570.144) și `9ecf871` (17 nov, 570.190) — `read_resp` / `rpc_rm_alloc` / `rpc_rm_control`: se verifică doar rezultatul transportului
RPC, nu starea RM. Constantele `rmapi::internal_flags` și `rmapi::channel_flags` sunt scrise de mână din antetele MIT de mai sus (intervale de biți pe
care bindgen nu le poate evalua), cu sursa în comentariu. Codul nostru (`user_space::open_work_channel_c135`, `C135Report`,
`rmapi::work_channel_params_with`, `gpu_sim::fake_rm::TokenRule`) e scris de noi; `TokenRule` și regula ChID-urilor din model sunt **ipoteze / valori
SIM**, nu placa. Niciun octet de firmware NVIDIA citit sau copiat; adresele NOCAT din jurnale nu au fost căutate în firmware. Niciun cod GPL citit pentru
această sesiune.

**Sesiunea 112 (18 nov, ~12:30), C-139 (I14 după `os-305`: alt argmax pe placă):** surse citite ca specificație sau ca idee, **nimic copiat** —
**NVIDIA, documentația publică „Parallel Thread Execution ISA” 8.8** (pagina de arhivă CUDA 12.9.2), §9.7.9.1 „Cache Operators”, tabelele 31–32:
rândurile `.ca`, `.cv`, `.wb`, `.wt` și §9.7.9.9 (`ld.global.nc`) — de aici ideea că driverul trebuie să invalideze liniile L1 între lansări
dependente. **tinygrad** (MIT), `tinygrad/runtime/ops_nv.py` de pe ramura principală la 18 nov 2026: `NVComputeQueue.memory_barrier` (metoda și cele
trei câmpuri), `NVComputeQueue.exec` (prima lansare prin PCAS, următoarele înlănțuite cu `dependent_qmd0_*`), șablonul QMD din `NVProgramData`
(`sm_global_caching_enable`, biții `invalidate_*`, `cwd_membar_type`, `constant_buffer_invalidate_0`); locurile din care e chemat `memory_barrier`
sunt în alt fișier, necitit. **Mesa** (MIT), prin oglinda android.googlesource: commit `ad98b76a147` (NAK, MR 35217, „Set cache ops on global memory
access on Kepler” — `LdCacheOp` / `StCacheOp` și alegerea lor) și commit `c39bf7e7f98` (NVK, MR 35246 — invalidarea cache-ului de date de textură
pe Kepler; din el doar numele barierelor `NVK_BARRIER_INVALIDATE_TEX_DATA` / `_SHADER_DATA`). **mesa-dev**, 26 oct. 2016, „[PATCH 4/7] nvc0: be
smarter when invalidating shader caches” (driverul Gallium vechi, MIT): sensul biților 0 și 12 ai `MEM_BARRIER` și observația că driverul NVIDIA
trimite de obicei `0x1011`. **Forumul NVIDIA pentru dezvoltatori**, firul „cache data invalidation between kernel calls” (iunie 2012): un răspuns
care citează manualul PTX. Din sursele OS-ului (`cyber-universe`, doar citit): `kernel/src/drivers/gpu_i14.rs`, `gpu_i11.rs`, `gpu_i10.rs` la `os-305`
— ordinea verificărilor și ce trimite pasul. Metoda `INVALIDATE_SHADER_CACHES_NO_WFI` și valoarea ei erau deja în codul nostru din sesiunea 21
(I10b, `work_queue::invalidate_caches`, din antetele MIT generate); C-139 doar o trimite mai des. Codul nou (`transfer::encode_sequence_fresh`,
`kronos_diag`, `gpu_compute::{layer, model}::trace_gpu`, `kronos_layer::copy_launch`, uneltele `analiza_i14` / `verifica_jurnal_lant`) e scris de noi.
`gpu_sim::cmdproc::CommandProcessor::model_stale_reads` e o **ipoteză pusă în model** (un singur cache pentru toată placa), nu comportament
observat. Programele NAK nu au fost recompilate; din listările lor existente (`tools/nak/out/*.sm86.txt`) s-au citit doar numele instrucțiunilor.
Niciun octet de firmware NVIDIA citit sau copiat. Niciun cod GPL citit pentru această sesiune.

**Sesiunea 113 (18 nov, ~19:25), C-142 (#513 pe placă: limite, viteză, FLR):** codul nou (`kronos_speed`, testul `sim_k513`, extinderea `TokenSim`,
copierea pe pagini și traducerea ținută pe pagină din `gpu_sim::cmdproc`) e scris de noi, din codul nostru existent; forma lui #513 din hub
(`integrare/kronos/MODEL.md`). Pentru analiza FLR (doc `C-142_K513_VITEZA_FLR.md` §4), surse **citite ca specificație, nimic copiat**: **driverul NVIDIA
570.144** (MIT, `open-gpu-kernel-modules` eticheta `570.144`): `kernel_bif.c` (`kbifResetFromTimeoutFullChip`), `arch/ampere/kernel_bif_ga100.c`
(`kbifPrepareForFullChipReset_GA100`, `kbifDoFullChipReset_GA100`), `arch/turing/kernel_bif_tu102.c` (`kbifDoFunctionLevelReset_TU102`),
`gsp/kernel_gsp.c` (`_kgspBootGspRm`), `arch/nvalloc/unix/src/osinit.c` (`NV_FLAG_TRIGGER_FLR`), `kernel-open/nvidia/nv.c` și `os-pci.c`
(`os_pci_trigger_flr`). **tinygrad** (MIT, ramura principală `8ffd4c8`, 18 nov 2026): `runtime/support/nv/nvdev.py` (`_early_ip_init`: WPR2 ridicat →
reset PCI), `runtime/support/nv/ip.py` (`wait_for_reset`), `runtime/support/system.py` (`reset`, `reset_method`). **nouveau** (GPL, Linux, doar citit):
`drivers/gpu/drm/nouveau/nvkm/subdev/gsp/tu102.c` (`tu102_gsp_booter_unload`, `tu102_gsp_fini`). **nova-core** (GPL, Linux, doar citit):
`drivers/gpu/nova-core/gsp/boot.rs`, `gsp/hal/tu102.rs`, `fb.rs`, `fb/regs.rs`. Din sursele GPL doar ordinea pașilor și textele de eroare, pentru
comparație; **nicio linie GPL în cod** (nici nu s-a scris cod FLR în această sesiune). Faptul „FLR da” vine din răspunsul OS-ului K-2 de pe workstation.
Niciun octet de firmware NVIDIA citit sau copiat.

**Sesiunea 116 (19 nov, ~10:50), C-145 (pauzele driverului la `kronos --gpu`):** cod nou scris de noi (`gpu_core::diag`: `Flush`, `LiveSink`,
`Routine`, `LiveStats`, `Journal::{with_live_sink, set_routine, routine, flush_live, live_stats}`; modulul `nvidia_ga104::kronos_fast`;
`Generation::{run_check_after, deferred_bytes}`; testul `tests/sim_c145.rs`, umbra în `TokenSim`). Surse: **doar datele noastre** — jurnalele
`os-307` din hub (`rezultate/os-automat/2026-11-19/`) și codul OS `cyber-universe` (`gpu_i6.rs`, `gpu_i14.rs`, doar citit, nimic copiat). Nicio
sursă străină nouă, niciun registru nou (adresele de rutină sunt cele deja folosite: `vram_window::CONTROL_REGISTERS`, `chain_log::WORK_DOORBELL`).
Niciun octet de firmware NVIDIA citit sau copiat.

**Sesiunea 120 (20 nov, ~01:30), C-147 punctul 2 (etapa 3a):** programul `tools/nak/q8_fused.comp` e scris de noi, din programele noastre
`q8_matvec.comp` și `gelu.comp` (aceeași buclă, același arbore, aceeași formulă); compilat cu Mesa NAK 25.0.7 (MIT, sursa verificată cu SHA-256 în
`build.sh`). Codul Rust nou (`q8_fused`, `token_launches_fused`, `token_kinds_fused`, `q8_fused_on_cpu`, testul `sim_c147_fused`) e scris de noi.
Ideea „lipirii” vine din planul din hub (`idei/2026-11-19_plan-performanta-placa.md`, sursele lui: llama.cpp, documentația CUDA Graphs) — nimic citit
sau copiat din ele în această sesiune. Nicio sursă GPL. Niciun octet de firmware NVIDIA citit sau copiat.
| NVIDIA open-gpu-kernel-modules, tag `570.144` (commit `8ec351aeb96a`): `ctrl2080perf.h`, `ctrl2080internal.h` (`INTERNAL_PERF_BOOST_SET_2X`), `ctrl2080clk.h` / `ctrl2080thermal.h` (goale în 570.144), `src/nvidia/src/kernel/gpu/perf/kern_perf_boost.c`, `kern_perf_ctrl.c`, `src/nvidia/generated/g_subdevice_nvoc.c` (tabelul de export: steagurile comenzilor), `src/nvidia/inc/kernel/rmapi/control.h` (sensul steagurilor), `src/nvidia/src/kernel/vgpu/rpc.c` | MIT (SPDX în fiecare fișier) | sesiunea 119 (C-147): **doar citit** ca specificație: comenzile și formele `PERF_GET_CURRENT_PSTATE`, `PERF_BOOST`, `INTERNAL_PERF_BOOST_SET_2X`, biții steagurilor, durata maximă; faptul că `GET_CURRENT_PSTATE` e `ROUTE_TO_PHYSICAL` (trimis neschimbat la GSP) iar `PERF_BOOST` e tradus de driverul NVIDIA în `INTERNAL_PERF_BOOST_SET_2X` → `gsp::perf`, scris de noi (constante scrise de mână după antet, cu testele octet cu octet). Niciun cod copiat |
| tinygrad `runtime/ops_nv.py` (`NVDevice.__init__`), `runtime/support/nv/ip.py` (`rpc_rm_control`), MIT, `master` la 19 nov 2026 (`7755a04`) | MIT | sesiunea 119 (C-147 punctul 4): **doar citit**: ei trimit la GSP direct `PERF_BOOST` cu `CUDA_YES \| CUDA_PRIORITY_HIGH \| BOOST_TO_MAX` și durata `0xffffffff`, fără să verifice răspunsul → forma B din `gsp::perf` (oprită; implicit forma A, a driverului NVIDIA). Niciun cod copiat |
| Kronos (cyberspace) `config/conv_w2_mare_24zc.toml` | proprietar (proiectul lui Adrian) | sesiunea 119 (C-150): doar citit: forma lui #651 (24 straturi, 12 capete, E 768, context 512, `bpe_16384`) → `kronos_speed::K651_*` |

**Sesiunea 131 (21 nov), C-155 punctul 1 (P3 gata de legat):** `kronos_diag::{TokenList, send_token_list, submissions_of, push_bytes_of}`
și testul `sim_c155_p3` sunt scrise de noi, fără nicio sursă externă (doar codul nostru din `kronos_diag`, `kronos_model`, `kronos_layer`).

**Sesiunea 121 (20 nov, ~04:30), C-147 punctul 3 (etapa 4):** programul `tools/nak/q8_matvec32.comp` e scris de noi, din programul nostru
`q8_matvec.comp` (aceeași formulă, același arbore de adunare, altă împărțire a muncii); compilat cu Mesa NAK 25.0.7 (MIT, sursa verificată cu SHA-256
în `build.sh`). Codul Rust nou (`q8::row_gpu_order_w32`, `q8_matvec32_on_cpu`, modulele `q8_matvec32` și `q8_race`, testul `sim_c147_w32`) e scris de
noi; ordinea operațiilor din oracol e citită în listarea NAK a programului nostru. Ideile (toate firele la lucru, citiri de 128 de biți) vin din planul
din hub (`idei/2026-11-19_plan-performanta-placa.md`) — nicio sursă străină citită sau copiată în această sesiune. Nicio sursă GPL. Niciun octet de
firmware NVIDIA citit sau copiat.

**Sesiunea 122 (20 nov, ~07:30), C-151 punctul 3 (resetul FLR):** codul nou (`gpu_pci::flr`, `nvidia_ga104::flr_reset`, modelul FLR din
`gpu_sim::pci` și `FakeGsp::sim_flr`) e scris de noi. Citite ca specificație (nimic copiat sau tradus): driverul NVIDIA 570.144 (MIT,
`kernel_bif_tu102.c` `kbifDoFunctionLevelReset_TU102`, `kernel_bif_ga100.c` `kbifDoFullChipReset_GA100` / `kbifClearDownstreamReadCounter_GA100`,
`kernel_bif_gh100.c` — ordinea și registrele PCI standard) și tinygrad (MIT, `runtime/support/nv/nvdev.py` `_early_ip_init`, `system.py` `reset`,
commit `c0bb645`). PCIe Base Spec: cunoaștere publică. Nicio sursă GPL citită în cod. Niciun octet de firmware NVIDIA citit sau copiat.

**C-147 punctul 4 (sesiunea 123, 20 nov 2026):** `gsp::perf::{boost_start, boost_finish, parse_ks, BoostStart, BoostEnd, BoostVerdict}`,
`kronos_perf::PerfGain` și modelul `gpu_sim::fake_rm::{SimPerf, BoostRule}` sunt scrise de noi. Citite ca specificație (nimic copiat sau
tradus): driverul NVIDIA 570.144 (MIT): `kern_perf_boost.c` (`kperfBoostSet_IMPL`), `ctrl2080internal.h`, `ctrl2080perf.h`,
`g_subdevice_nvoc.c` (steagurile exportului), `control.c` (`RMCTRL_FLAGS_INTERNAL`), `vgpu/rpc.c` (`rpcRmApiControl_GSP`). Nicio sursă GPL.
Niciun octet de firmware NVIDIA citit sau copiat.

**C-151 punctul 1 (sesiunea 124, 20 nov 2026):** `gsp::rusd` (`rusd_start`, `rusd_stop`, `read_section`, `read_dashboard`,
`dashboard_at_end`, `parse_kd`, `RusdMem`, `Dashboard`) și modelul `gpu_sim::fake_rm::{SimRusd, RusdRule}` sunt scrise de noi. Citite ca
specificație (nimic copiat sau tradus): driverul NVIDIA 570.144 (MIT): `class/cl00de.h` (structura, măștile, regula de citire),
`ctrl/ctrl00de.h`, `ctrl2080internal.h` (cele două comenzi), `gpu_user_shared_data.c` (ordinea, memoria), `g_subdevice_nvoc.c` și
`rmapi/control.h` (steagurile), `nvrm_registry.h` (intervalul 500 ms, limitele 100–1000), `nvfixedtypes.h` (`NvTemp` = 24.8). Pozițiile
câmpurilor sunt tipărite cu `offsetof` de compilatorul C din antetul NVIDIA (programul de 20 de rânduri nu intră în repo).
nouveau, seria „GSP telemetry via RUSD” (Mohamed Ahmed, 14 iul 2026, GPL/MIT, nepreluată) — doar citită ca a treia sursă pentru ordine și
citire; **nicio linie preluată**. Nicio sursă GPL în cod. Niciun octet de firmware NVIDIA citit sau copiat.

**C-147 punctul 7 (sesiunea 125, 20 nov 2026):** `driver_split` (împărțirea timpului din ceasul rândurilor jurnalului), exemplul
`timpul_driverului`, `kronos_fast::ShadowedMemory` și testul T141 sunt scrise de noi. Surse: doar jurnalele noastre de pe placă (hub
`rezultate/os-automat/2026-11-20/`, rândurile din test sunt copiate de acolo) și codul OS-ului nostru (`cyber-universe` `7fad533`, citit).
Nicio sursă externă nouă, nimic GPL, niciun octet de firmware.

**C-147 punctul 5 (sesiunea 126, 20 nov 2026):** `gpu_compute::mma` (conversiile pe 16 biți, referința, ipoteza, toleranța, `judge`),
`nvidia_ga104::mma16`, modelul `gpu_sim::cmdproc::mma16_f16_on_cpu`, programele GLSL `mma16_f16.comp` / `mma16_s8.comp`, programul
gazdă `nak_dump_cm.c`, scriptul `build2.sh` și testul T142 sunt scrise de noi. Unealtă pe gazdă (niciodată în OS): Mesa 26.3.1 (MIT),
construită din arhiva oficială; ieșirea ei (`tools/nak/out2/*.sm86.sass`) e cod generat de compilator din programele noastre. Citite ca
specificație (nimic copiat): `nvk_physical_device.c` (formele „cooperative matrix”), extensiile Khronos `VK_KHR_cooperative_matrix` /
`GL_KHR_cooperative_matrix`; articolul Fasi, Higham, Mikaitis, Pranesh (PeerJ CS 2021) pentru ideea ipotezei de adunare. Nicio sursă
GPL, niciun octet de firmware NVIDIA.

**B1 (sesiunea 223, 25 nov 2026):** `gpu_compute::mma_s8` (oracolul în i64 și modelul modulo 2³²), `MMA16_S8` / `cbuf_s8` / `qmd_s8`
în `nvidia_ga104::mma16`, modelul `gpu_sim::cmdproc::mma16_s8_on_cpu` și testul T229 sunt scrise de noi. Programul `mma16_s8.sm86.sass` e
ieșirea Mesa NAK 26.3.1 (MIT) din `mma16_s8.comp` (al nostru, sesiunea 126). Citite ca specificație: PTX ISA (instrucțiunea `mma`, `.s8`/`.s32`,
`.satfinite`) și whitepaper-ul NVIDIA GA102 (Anexa A, vitezele). Nicio sursă GPL, niciun octet de firmware.

**B1 pasul 3 (sesiunea 225, 25 nov 2026):** `tools/nak/mm_s8.comp` (al nostru; așezarea după `mm_tc128.comp`, tot al nostru),
`gpu_compute::mma_s8::{ProductS8, mm_s8_ref, mm_s8_wrap}`, `nvidia_ga104::mm_s8`, `gpu_sim::cmdproc::mm_s8_on_cpu` și testul T231 sunt
scrise de noi. `mm_s8.sm86.sass` e ieșirea Mesa NAK 26.3.1 (MIT). Idei de așezare pe plăci: scrierile publice CUTLASS / „How to optimize
a CUDA matmul”; nimic copiat. Nicio sursă GPL, niciun octet de firmware.

**C-147 punctul 6 (sesiunea 127, 20 nov 2026) — doar studiu, fără cod:** `docs/C-147_PUNCTUL6_LANT.md`. Citite ca specificație (nimic
copiat, niciun cod scris): NVIDIA `open-gpu-doc` (MIT) commit `9fdf5c40` — `classes/compute/clc6c0qmd.h`, `clc7c0qmd.h`, `clc7c0.h`
(pozițiile câmpurilor QMD V03_00 și codurile acțiunilor); tinygrad (MIT) commit `5e8b5da9` — `tinygrad/runtime/ops_nv.py` (cum leagă
lansările cu `dependent_qmd0_*` și ce câmpuri pune în șablonul QMD); Mesa 26.3.1 (MIT) — `src/nouveau/compiler/nak/qmd.rs`,
`hw_tests.rs`, `hw_runner.rs`, `src/nouveau/vulkan/nvk_cmd_dispatch.c`. Nicio sursă GPL, niciun octet de firmware NVIDIA.

**C-157 punctul 3 (sesiunea 136, 21 nov 2026) — dieta 2:** `gpu_compute::model::{order_key, argmax_bits}`, `nvidia_ga104::kronos_diet2`,
`tests/sim_c157_diet2.rs`, `examples/dieta_doi.rs` — scrise de noi. Cheia de ordine din biții unui f32 e tehnica obișnuită (aceeași ca în
programul nostru `argmax.comp`, sesiunea 134). Cifrele: jurnalul os-328 din hub; codul OS `cyber-universe` doar citit. Nicio sursă externă
copiată, nimic GPL, niciun octet de firmware NVIDIA.

**C-157 punctul 4 (sesiunea 137, 21 nov 2026) — #651:** `kronos_speed::ModelLimits::{of_lists, k651_lists}`,
`kronos_gen::VramBudget::with_programs`, `tests/sim_c157_k651.rs` — scrise de noi. Cifrele modelului: hub `integrare/kronos/MODEL_651.md`;
constantele OS: codul `cyber-universe` doar citit. Nicio sursă externă copiată, nimic GPL, niciun octet de firmware NVIDIA.

**C-160 punctul 2 (sesiunea 140, 21 nov 2026) — studiul motorului de antrenare, fără cod de driver:** `docs/C-160_MOTOR_STUDIU.md`,
`tools/buget-antrenare/buget.py` (Python simplu, scris de noi; socoteala memoriei). Arhitectura și rețetele: hub
`integrare/kronos/PAS_DE_AUR.md`. Formulele înapoi și ideile (activările păstrate, P recalculat, scalarea erorii la f16, AdamW) din
studiul anterior al surselor deschise / publicate, notate în secțiunea 5 a documentului: PyTorch (BSD-3), llm.c (MIT), tinygrad (MIT),
articolele arXiv 2205.05198, 2205.14135, 1710.03740, 1711.05101. Nimic copiat, nimic GPL, niciun octet de firmware NVIDIA.

**C-160 punctul 3 (sesiunea 141, 21 nov 2026) — oracolul pasului de antrenare:** `crates/gpu-compute/src/train.rs` (Rust scris de noi:
înainte, entropia încrucișată, înapoi prin LayerNorm / GELU exact / atenția cauzală / încorporările legate, tăierea gradientului, SGD,
AdamW). Specificația: hub `integrare/kronos/PAS_DE_AUR.md` și documentația publică PyTorch (BSD-3) pentru formulele `LayerNorm`,
`GELU`, `MultiheadAttention`, `SGD`, `AdamW` (ordinea „single tensor”: `lerp` pentru m), `clip_grad_norm_`; formulele „înapoi” sunt cele
standard (și în `docs/C-160_MOTOR_STUDIU.md`). `erf` în f64: seria cu termeni pozitivi (Abramowitz & Stegun 7.1.6), scrisă de noi.
Datele de test `crates/gpu-compute/tests/date/pas_de_aur_mic/` = copia exactă a dosarului din hub (proiectul Kronos al lui Adrian,
greutăți din sămânța 1132, fără date personale); SHA-256 al fiecărui fișier verificat față de `reteta.json` la copiere (198/198),
`reteta.json` = `5e3c5824…e3a8`. Nimic copiat din PyTorch, nimic GPL, niciun octet de firmware NVIDIA.


### C-160 punctul 4 — `mm_train` (sesiunea 142)
`tools/nak/mm_train.comp`, `crates/nvidia-ga104/src/mm_train.rs`, `gpu_sim::cmdproc::mm_train_on_cpu`,
`gpu_compute::reference::matmul_t_fma_ftz`, testul `crates/nvidia-ga104/tests/sim_mm_train.rs`: scrise de noi, după propriul nostru
`matmul_smem.comp` (ideea plăcilor în memoria partajată: siboehm SGEMM_CUDA, MIT, doar ideea). Formele celor trei produse ale unui strat
`Linear` (Y = X·Wᵀ + b, dX = dY·W, dW = dYᵀ·X) sunt formulele standard (documentația PyTorch `Linear`, BSD). Marginea de eroare a unui lanț
de FMA în f32, γ_K ≈ K·2⁻²⁴·Σ|aᵢbᵢ|: Higham, „Accuracy and Stability of Numerical Algorithms”, cap. 3. Binarul: Mesa NAK 25.0.7 (MIT),
compilat pe gazdă. Nimic GPL, niciun octet de firmware NVIDIA.

### C-160 punctul 4 — `xent`, programul 7 (sesiunea 143)
`tools/nak/xent.comp`, `crates/nvidia-ga104/src/xent.rs`, `gpu_sim::cmdproc::xent_on_cpu`, `gpu_compute::layer::xent_gpu256`,
`gpu_compute::train::softmax_xent` / `logits` (scoase din `loss_and_grad`, același calcul), testul `crates/nvidia-ga104/tests/sim_xent.rs`:
scrise de noi. Formula (softmax cu scăderea maximului, gradientul (p − unu-la-țintă)/N): standard — documentația PyTorch `cross_entropy`
(BSD) și llm.c `crossentropy_softmax_backward` (MIT), doar ideea. Forma reducerilor: propriul nostru `layernorm256.comp`. Marginea: rotunjirile
f32 (Higham, cap. 3) + o margine generoasă pentru unitățile MUFU (fără cifre de la NVIDIA). Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă
(Mesa reconstruit în cloud; `mm_train` a ieșit identic, aceeași sumă). Nimic GPL, niciun octet de firmware NVIDIA.

### C-160 punctul 4 — `ln_back` (programul 4, dx) și `colsum` (programul 2 + dγ/dβ) (sesiunea 144)
`tools/nak/ln_back.comp`, `tools/nak/colsum.comp`, `crates/nvidia-ga104/src/ln_back.rs`, `crates/nvidia-ga104/src/colsum.rs`,
`gpu_sim::cmdproc::ln_back_on_cpu` / `colsum_on_cpu`, `gpu_compute::layer::ln_back_gpu256` / `colsum_gpu` / `LnStats`,
`gpu_compute::train::layernorm_stats` / `layernorm_back_rows` (adăugate; folosesc exact `layernorm` / `layernorm_back` din `loss_and_grad`),
testele `crates/nvidia-ga104/tests/sim_ln_back.rs` și `sim_colsum.rs`: scrise de noi. Formulele (dx = r·(g − media g − x̂·media g·x̂),
dγ = Σ dy·x̂, dβ = Σ dy): standard — documentația PyTorch `LayerNorm` (BSD) și llm.c `layernorm_backward` (MIT), doar ideea; despărțirea în
două programe (fără `atomic`) e alegerea noastră. Forma reducerii: propriul nostru `layernorm256.comp` / `xent.comp`. Marginile: rotunjirile
f32 (Higham, cap. 3). Binarele: Mesa NAK 25.0.7 (MIT), compilat pe gazdă (Mesa reconstruit în cloud; `xent` a ieșit identic). Nimic GPL,
niciun octet de firmware NVIDIA.
- **C-157 p.2 după os-335 (sesiunea 145):** `gpu_compute::mma::{HwAdder, GA104_OS335, mma16_hw}`, `mma16_probe::{fit_rows, …}` și testul `sim_c157_p2e_mma_fit.rs`: scrise de noi. Forma modelului (blocuri de produse adunate deodată, aliniere cu tăiere) e ideea din măsurătorile publicate pe V100/T4/A100 (Fasi, Higham, Mikaitis, Pranesh, „Numerical behavior of NVIDIA tensor cores”, PeerJ CS 2021) — doar ideea; parametrii (8 produse, exponentul nominal, 1 bit în plus) sunt potriviți de noi pe cifrele plăcii din os-335. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 4, programul 3 — `ln_fwd` (sesiunea 146):** `tools/nak/ln_fwd.comp`, `crates/nvidia-ga104/src/ln_fwd.rs`, `gpu_sim::cmdproc::ln_fwd_on_cpu`, `gpu_compute::layer::ln_fwd_gpu256`, `gpu_compute::train::layernorm_rows` (folosește exact `layernorm` din `loss_and_grad`) și testul `crates/nvidia-ga104/tests/sim_ln_fwd.rs`: scrise de noi. Formula LayerNorm (μ, varianța ÷ C, 1/√(var + ε), γ, β): standard — documentația PyTorch `LayerNorm` (BSD) și llm.c `layernorm_forward` (MIT, care păstrează și el μ și r pentru drumul înapoi), doar ideea. Forma reducerii: propriul nostru `layernorm256.comp`. Locul pentru `MUFU.RSQ` (2⁻²²): ordinul de mărime al erorii documentate public de NVIDIA pentru rădăcina inversă aproximativă (~2 ulp), nemăsurat pe placa noastră. Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă (Mesa reconstruit în cloud; `add64` și `ln_back` au ieșit identice). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 4, programul 5 — `gelu_tr` (sesiunea 147):** `tools/nak/gelu_tr.comp`, `crates/nvidia-ga104/src/gelu_tr.rs`, `gpu_sim::cmdproc::gelu_tr_on_cpu`, `gpu_compute::layer::gelu_tr_gpu`, `gpu_compute::train::{gelu_rows, gelu_back_rows}` (folosesc exact `gelu_and_grad` din `loss_and_grad`) și testul `crates/nvidia-ga104/tests/sim_gelu_tr.rs`: scrise de noi. GELU exact și derivata lui Φ + u·φ: formule standard (documentația PyTorch `GELU`, BSD — doar formula); erf prin Abramowitz & Stegun 7.1.26 (manual public), ca la `gelu.comp`. Locul pentru `MUFU.RCP` / `MUFU.EX2` (2⁻²² fiecare): ordinul de mărime documentat public de NVIDIA, nemăsurat pe placa noastră. Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă (Mesa reconstruit în cloud; `ln_fwd` a ieșit identic). Tot aici: `train::erf64(0)` nu se mai învârte la nesfârșit (întoarce ±0). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 4, programul 6 (înapoi) — `attn_back` (sesiunea 148):** `tools/nak/attn_back.comp`, `crates/nvidia-ga104/src/attn_back.rs`, `gpu_sim::cmdproc::attn_back_on_cpu`, `gpu_compute::layer::{AttnShape, attn_back_row_gpu, attn_back_col_gpu, attn_back_gpu}`, `gpu_compute::train::{attention_rows, attention_back_rows}` (folosesc exact `attention` / `attention_back` din `loss_and_grad`) și testul `crates/nvidia-ga104/tests/sim_attn_back.rs`: scrise de noi. Formulele atenției înapoi (dV = Pᵀ·dO, dP = dO·Vᵀ, dS = P∘(dP − Σ P∘dP), dQ = dS·K, dK = dSᵀ·Q): standard (documentația PyTorch, BSD, și llm.c, MIT — doar formulele, din studiul `docs/C-160_MOTOR_STUDIU.md`); împărțirea în două lansări fără `atomic` e alegerea noastră. Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă (Mesa reconstruit în cloud; `gelu_tr` a ieșit identic). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 4, programul 6 (înainte) — `attn_fwd` (sesiunea 149):** `tools/nak/attn_fwd.comp`, `crates/nvidia-ga104/src/attn_fwd.rs`, `gpu_sim::cmdproc::attn_fwd_on_cpu`, `gpu_compute::layer::{ATTN_FWD_LOG2E, attn_fwd_row_gpu, attn_fwd_gpu}` și testul `crates/nvidia-ga104/tests/sim_attn_fwd.rs`: scrise de noi. Formulele (softmax(QKᵀ/√d) cu masca cauzală, apoi ·V; scăderea maximului): standard (documentația PyTorch `scaled_dot_product_attention`, BSD, și llm.c `attention_forward`, MIT — doar formulele). Folosirea rândului lui P ca loc de lucru și un fir pe rând sunt alegerea noastră. Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă (Mesa reconstruit în cloud; `attn_back` a ieșit identic). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 4, programul 8 — `emb_fwd` + `emb_back` (sesiunea 150):** `tools/nak/emb_fwd.comp`, `tools/nak/emb_back.comp`, `crates/nvidia-ga104/src/{emb_fwd,emb_back}.rs`, `gpu_sim::cmdproc::{emb_fwd_on_cpu, emb_back_on_cpu}`, `gpu_compute::layer::{emb_fwd_gpu, EmbSegments, emb_back_seg_gpu, emb_back_gpu}`, `gpu_compute::train::{embed_rows, embed_back_rows, embed_grad_inputs}` și testul `crates/nvidia-ga104/tests/sim_emb.rs`: scrise de noi. Formulele (încorporarea = căutare în tabel + poziția; gradientul = adunarea rândurilor pe tabel, cu capul legat): standard (documentația PyTorch `nn.Embedding`, BSD, și llm.c `encoder_forward` / `encoder_backward`, MIT — doar ideile). Gruparea rândurilor în segmente pe procesor, ca să nu folosim `atomic` (deci aceiași biți la fiecare rulare), e alegerea noastră. Binarele: Mesa NAK 25.0.7 (MIT), compilate pe gazdă. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 4, programele 9 și 10 — `gnorm` + `opt_step` (sesiunea 152):** `tools/nak/gnorm.comp`, `tools/nak/opt_step.comp`, `crates/nvidia-ga104/src/{gnorm,opt_step}.rs`, `gpu_sim::cmdproc::{GNORM_NEEDS, gnorm_on_cpu, opt_step_on_cpu}`, `gpu_compute::layer::{gnorm_block_gpu, gnorm_finish_gpu, grad_norm_gpu, OptScalars, opt_step_gpu}`, `gpu_compute::train::{AdamW::scalars, AdamScalars}` și testul `crates/nvidia-ga104/tests/sim_c160_opt.rs`: scrise de noi. Formulele: documentația PyTorch `clip_grad_norm_` și `torch.optim.AdamW` (BSD), Loshchilov & Hutter 2019 (arXiv 1711.05101), llm.c `adamw_kernel` (MIT) — doar ideile. Sumele parțiale pe blocuri fără `atomic` (aceiași biți la fiecare rulare) și factorul de tăiere lăsat în memoria plăcii pentru pasul optimizatorului sunt alegerea noastră. Binarele: Mesa NAK 25.0.7 (MIT), compilate pe gazdă. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7, partea întâi — estimarea vitezei de antrenare (sesiunea 156):** `docs/C-160_P7_VITEZA_ESTIMARE.md`, `tools/buget-antrenare/viteza.py` (socoteală scrisă de noi). Cifre luate, nu cod: specificațiile publice NVIDIA RTX 3070 Ti (48 SM, ceasul de boost), raportul public „NVIDIA Ampere GA102 GPU Architecture” (rata circuitelor speciale cu adunare F32), formula 6·P (Kaplan et al. 2020), viteza PyTorch din `Cyber-hub/integrare/kronos/FISA_ANTRENARE.md` (MĂSURAT de Kronos). Nimic copiat.
- **C-160 punctul 7, partea a doua — `mm_tc`, înmulțirea antrenării pe circuitele speciale (sesiunea 157):** `tools/nak/mm_tc.comp`, `crates/nvidia-ga104/src/mm_tc.rs`, `gpu_sim::cmdproc::{MM_TC_NEEDS, mm_tc_on_cpu}`, `crates/gpu-compute/src/mm_tc.rs` și testul `crates/nvidia-ga104/tests/sim_mm_tc.rs`: scrise de noi. Ideile aranjamentului (bucăți 64×64 pe bloc, felii de K în memoria partajată, 2×2 acumulatoare pe grup): scrierile publice despre GEMM pe tensor cores (NVIDIA CUTLASS, BSD-3; siboehm „How to Optimize a CUDA Matmul Kernel”, MIT) — doar ideile, nimic copiat. `coopMatLoad` / `coopMatMulAdd`: specificația Khronos `GL_KHR_cooperative_matrix`. Marginea: derivată de noi din modelul `mma::GA104_OS335` (amprenta măsurată pe placă). Scalarea erorii (doar menționată): Micikevicius et al. 2018. Binarul: Mesa NAK 26.3.1 (MIT), a doua unealtă, compilat pe gazdă. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7 — `colsum` împărțit pe rânduri (sesiunea 159):** `tools/nak/colsum_part.comp`, `crates/nvidia-ga104/src/colsum_part.rs`, `gpu_sim::cmdproc::colsum_part_on_cpu`, `gpu_compute::layer::{colsum_part_gpu, colsum_split_gpu}`, `gpu_compute::train::colsum_split_margin`, testul `crates/nvidia-ga104/tests/sim_colsum_part.rs` și `tools/buget-antrenare/colsum_timp.py`: scrise de noi. Reducerea în două treceri (sume parțiale pe bucăți, apoi suma lor) e tehnica obișnuită a reducerilor paralele (Harris, „Optimizing Parallel Reduction in CUDA”, prezentare publică NVIDIA) — doar ideea. Marginea: rotunjirile f32 (Higham, cap. 3). Timpii: MĂSURAȚI pe placă de OS (os-349). Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă (Mesa reconstruit în cloud; `colsum` a ieșit identic, `7a0fa2bd…`). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7, partea a patra — proba `mm_tc` pentru placă (sesiunea 160):** `crates/nvidia-ga104/src/mm_tc_probe.rs`, testul `crates/nvidia-ga104/tests/sim_c160_p7_mm_tc_probe.rs`, `docs/C-160_P7_MM_TC_PROBA.md`, nota `integration/cyber-universe/C-160_P7d_mm_tc-proba.md`: scrise de noi, după forma probelor noastre `colsum_probe` / `mma16_probe`. Funcția de dispersie splitmix64 (Steele, Lea, Flood 2014, domeniu public ca idee). Forma qkv: dimensiunile lui #513 din `Cyber-hub/integrare/kronos/FISA_ANTRENARE.md`. Niciun binar nou (`mm_tc` din sesiunea 157). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7 — cine e limitat de un fir; atenția cu blocuri de 128 (sesiunea 162):** `tools/buget-antrenare/fir_limitat.py`, `docs/C-160_P7_FIR_LIMITAT.md`, `tools/nak/attn_fwd128.comp`, `tools/nak/attn_back128.comp` (copiile programelor noastre `attn_fwd` / `attn_back` cu blocuri de 128), `crates/nvidia-ga104/src/{attn_fwd128,attn_back128}.rs`, `gpu_sim::cmdproc::{attn_fwd128_on_cpu, attn_back128_on_cpu}` și testul `crates/nvidia-ga104/tests/sim_attn128.rs`: scrise de noi. Limitele SM 8.6 (16 blocuri, 48 de warp-uri pe SM): ghidul public CUDA; latențele: ordine de mărime publice din articole de microbenchmark Ampere — doar cifre, DEDUS. Binarele: Mesa NAK 25.0.7 (MIT), compilat pe gazdă (Mesa reconstruit în cloud; `attn_fwd` a ieșit identic). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7 — atenția înainte pe plăci (sesiunea 163):** `tools/nak/attn_fwd_tile.comp`, `docs/C-160_P7_ATENTIA_PE_PLACI.md`, `print_tiled` în `tools/buget-antrenare/fir_limitat.py`, `crates/nvidia-ga104/src/attn_fwd_tile.rs`, `gpu_compute::layer::{attn_fwd_tile_row_gpu, attn_fwd_tile_gpu, ATTN_TILE_MAX_D}`, `gpu_sim::cmdproc::attn_fwd_tile_on_cpu` și testul `crates/nvidia-ga104/tests/sim_attn_tile.rs` (marginea copiată din T160, a noastră): scrise de noi. Ideile: plăcile în memoria partajată cu 4 × 4 rezultate pe fir din pașii noștri `matmul_2d` (după siboehm SGEMM_CUDA, MIT, doar ideea); atenția pe plăci: articolul public FlashAttention și llm.c (MIT) — doar ideea. Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7 — atenția înapoi pe plăci, modul 1 (sesiunea 164):** `tools/nak/attn_back_tile.comp`, `docs/C-160_P7_ATENTIA_INAPOI_PE_PLACI.md`, `print_tiled_back` în `tools/buget-antrenare/fir_limitat.py`, `crates/nvidia-ga104/src/attn_back_tile.rs`, `gpu_sim::cmdproc::attn_back_tile_on_cpu` și testul `crates/nvidia-ga104/tests/sim_attn_back_tile.rs` (cu emularea literală a codului GLSL): scrise de noi. Ideile: plăcile din `matmul_2d` / `attn_fwd_tile` (după siboehm SGEMM_CUDA, MIT, doar ideea); drumul înapoi pe plăci: articolul public FlashAttention-2 și llm.c (MIT) — doar ideea. Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7 — atenția înapoi pe plăci, modul 0 (sesiunea 165):** `tools/nak/attn_back_tile0.comp`, `docs/C-160_P7_ATENTIA_INAPOI_MOD0_PE_PLACI.md`, `print_tiled_back0` în `tools/buget-antrenare/fir_limitat.py`, `crates/nvidia-ga104/src/attn_back_tile0.rs`, `gpu_sim::cmdproc::attn_back_tile0_on_cpu` și testul `crates/nvidia-ga104/tests/sim_attn_back_tile0.rs` (cu emularea literală a codului GLSL): scrise de noi. Ideile: plăcile din `matmul_2d` / `attn_*_tile` (după siboehm SGEMM_CUDA, MIT, doar ideea); formula dS = P ∘ (dP − rowsum(P ∘ dP)): articolul public FlashAttention-2 și llm.c (MIT) — doar formula; adresele rotite în memoria partajată: tehnică generală (ghidul public CUDA, conflictele de bănci). Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7 — proba atenției pe plăci pentru placă (sesiunea 166):** `crates/nvidia-ga104/src/attn_tile_probe.rs`, testul `crates/nvidia-ga104/tests/sim_c160_p7_attn_tile_probe.rs`, `docs/C-160_P7_ATENTIA_PROBA.md`, nota `integration/cyber-universe/C-160_P7e_atentia-pe-placi-proba.md`: scrise de noi, după forma probelor noastre `mm_tc_probe` / `colsum_probe`. Dispersia splitmix64 (Steele, Lea, Flood 2014, domeniu public ca idee). Precizia MUFU (EX2 / RCP aproximative): ghidul public CUDA, anexa funcțiilor matematice. Niciun binar nou (`attn_fwd_tile`, `attn_back_tile0`, `attn_back_tile` din sesiunile 163–165). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7 — `attn_p`, doar P fără P·V (sesiunea 168):** `tools/nak/attn_p.comp` (= `attn_fwd.comp` al nostru fără ultima buclă), `crates/nvidia-ga104/src/attn_p.rs`, `gpu_compute::layer::attn_p_row_gpu`, `gpu_sim::cmdproc::attn_p_on_cpu`, testul `crates/nvidia-ga104/tests/sim_attn_p.rs`, `docs/C-160_P7_ATTN_P.md`: scrise de noi. Ideea „recalculează în loc să păstrezi” (gradient checkpointing): articolul public Chen et al. 2016 și FlashAttention (doar ideea). Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 punctul 5 — ceasul termic (sesiunea 169):** `crates/nvidia-ga104/src/gsp/rusd_watch.rs`, testul `crates/nvidia-ga104/tests/sim_c157_p5_ceas_termic.rs`, `docs/C-157_P5_CEAS_TERMIC.md`, nota `integration/cyber-universe/C-157_P5_ceas-termic.md`: scrise de noi. Biții motivelor de încetinire = cei din `rusd::reason` (antetele NVIDIA 570.144, MIT). Pragul propus pornește de la pagina oficială NVIDIA GeForce RTX 3070 / 3070 Ti („Maximum GPU Temperature 93”) și de la forma pragurilor `nvidia-smi -q` pe Ampere (forumul NVIDIA pentru dezvoltatori: Max Operating 87 / Slowdown 102 / Shutdown 105 pe RTX 3060 Laptop) — doar cifre citate, niciun cod. Firmware-ul NVIDIA nu a fost deschis sau analizat.
- **C-157 punctul 7 — simbolul ales rămâne pe placă (sesiunea 170):** `tools/nak/embed_dev.comp` (= `embed.comp` al nostru cu simbolul citit din rezultatul argmax), `crates/nvidia-ga104/src/kronos_batch.rs`, `Generation::next_batch` / `accept_batch`, `gpu_sim::cmdproc::embed_dev_on_cpu`, testul `crates/nvidia-ga104/tests/sim_c157_p7_batch.rs`, `docs/C-157_P7_SIMBOL_PE_PLACA.md`, nota `integration/cyber-universe/C-157_P7_simbol-pe-placa.md`: scrise de noi. Ideea „simbolul rămâne pe placă, mai mulți pași pe o trimitere” e cea din planul de performanță al hub-ului (etapa 5, `idei/2026-11-19_plan-performanta-placa.md`); compilat cu Mesa NAK 25.0.7 (MIT), unealtă pe gazdă. Nimic copiat.
- **C-157 punctul 2 — comparația os-358, buclele NAK numărate (sesiunea 171):** `tools/buget-antrenare/nak_bucle.py`, secțiunea os-358 din `docs/C-157_P2_COMPARATII.md`, nota `integration/cyber-universe/C-157_P2l_os-358_atentie.md`: scrise de noi. Citesc doar listările noastre produse de Mesa NAK 25.0.7 (MIT). Băncile memoriei partajate (32 de bănci de 4 octeți), limitele SM 8.6 și ocuparea: ghidul public de programare CUDA; latențele: ordine de mărime publice din articole de microbenchmark Ampere — doar cifre, DEDUS. Nimic copiat, nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 punctul 2 după os-358 — `attn_fwd_tile2` (sesiunea 172):** `tools/nak/attn_fwd_tile2.comp` (din `attn_fwd_tile.comp`, al nostru), `crates/nvidia-ga104/src/attn_fwd_tile2.rs`, testul `crates/nvidia-ga104/tests/sim_attn_tile2.rs` (bucla naturală din `nak_bucle.py`, a noastră), `docs/C-157_P2_ATTN_TILE2.md`: scrise de noi. Ideile (rânduri cu un element în plus contra conflictelor de bancă, citiri emise deodată, bucle desfăcute): ghidul public CUDA și siboehm SGEMM_CUDA (MIT, doar ideea). Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă (`attn_fwd_tile` a ieșit identic). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 punctul 2 după os-358 — `attn_back_tile0_2` (sesiunea 173):** `tools/nak/attn_back_tile0_2.comp` (din `attn_back_tile0.comp`, al nostru), `crates/nvidia-ga104/src/attn_back_tile0_2.rs`, testul `crates/nvidia-ga104/tests/sim_attn_back_tile0_2.rs` (emularea literală a GLSL-ului, a noastră), `docs/C-157_P2_ATTN_TILE2.md`: scrise de noi. Aceleași idei și surse ca la `attn_fwd_tile2` (ghidul public CUDA, siboehm SGEMM_CUDA MIT, doar ideea). Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă (`attn_back_tile0` a ieșit identic). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 punctul 2 după os-358 — `attn_back_tile_2` (sesiunea 173):** `tools/nak/attn_back_tile_2.comp` (din `attn_back_tile.comp`, al nostru), `crates/nvidia-ga104/src/attn_back_tile_2.rs`, testul `crates/nvidia-ga104/tests/sim_attn_back_tile_2.rs` (emularea literală a GLSL-ului, a noastră), `docs/C-157_P2_ATTN_TILE2.md`: scrise de noi; aceleași idei și surse ca la `attn_fwd_tile2`. Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7 — `attn_p` legat în pas (sesiunea 175):** `train_step::Prog::AttnP`, `PROGRAMS_P_ONLY`, `Plan::new_p_only` / `programs` / `p_program`, testele T185 (`sim_c160_p5_golden.rs`, `sim_c160_p6_train_step.rs`), `docs/C-160_P7_ATTN_P_IN_PAS.md`: scrise de noi; niciun binar nou (`attn_p` din sesiunea 168). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7 — `attn_p_tile`, doar P pe plăci (sesiunea 176):** `tools/nak/attn_p_tile.comp` (= `attn_fwd_tile2.comp` al nostru fără P·V), `crates/nvidia-ga104/src/attn_p_tile.rs`, `gpu_compute::layer::attn_p_tile_row_gpu` / `attn_p_tile_gpu`, `gpu_sim::cmdproc::attn_p_tile_on_cpu`, testul `crates/nvidia-ga104/tests/sim_attn_p_tile.rs`, `docs/C-160_P7_ATTN_P_TILE.md`: scrise de noi; aceleași idei și surse ca la `attn_p` și `attn_fwd_tile2`. Binarul: Mesa NAK 25.0.7 (MIT), compilat pe gazdă (`attn_fwd_tile2` și `attn_p` au ieșit identice). Nimic GPL, niciun octet de firmware NVIDIA.
- **C-160 punctul 7 — programele pe plăci legate în pas (sesiunea 177):** adăugările din `crates/nvidia-ga104/src/train_step.rs` (`Prog::AttnFwdTile2` …, `Plan::new_tiles`, `new_p_only_tiles`), testele T187, `docs/C-160_P7_PLACI_IN_PAS.md`: scrise de noi; nicio sursă nouă. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 punctul 2 după os-364 — `attn_back_tile0_3`, citiri fără ramură (sesiunea 179):** `tools/nak/attn_back_tile0_3.comp` (din `attn_back_tile0_2.comp`, al nostru), `crates/nvidia-ga104/src/attn_back_tile0_3.rs`, testul `crates/nvidia-ga104/tests/sim_attn_back_tile0_3.rs` (emularea literală a GLSL-ului, a noastră), funcțiile `waits_fresh` / `s179` din `tools/buget-antrenare/nak_bucle.py`, `docs/C-157_P2_ATTN_TILE3.md`: scrise de noi. Ideea „citire la adresa strânsă + alegere în loc de ramură”: ghidul public CUDA (predicația); compilatorul Mesa NAK 25.0.7 (MIT) doar folosit pe gazdă, nimic copiat. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 punctul 2 după os-364 — `attn_back_tile_3`, citiri fără ramură (sesiunea 180):** `tools/nak/attn_back_tile_3.comp` (din `attn_back_tile_2.comp`, al nostru), `crates/nvidia-ga104/src/attn_back_tile_3.rs`, testul `crates/nvidia-ga104/tests/sim_attn_back_tile_3.rs` (emularea literală a GLSL-ului, a noastră), `PROGRAMS180` / `s180` din `tools/buget-antrenare/nak_bucle.py`, secțiunea din `docs/C-157_P2_ATTN_TILE3.md`: scrise de noi. Aceeași idee (ghidul public CUDA, predicația); Mesa NAK 25.0.7 (MIT) doar folosit pe gazdă, nimic copiat. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 punctul 2 după os-364 — `attn_fwd_tile3`, citiri fără ramură (sesiunea 181):** `tools/nak/attn_fwd_tile3.comp` (din `attn_fwd_tile2.comp`, al nostru), `crates/nvidia-ga104/src/attn_fwd_tile3.rs`, testul `crates/nvidia-ga104/tests/sim_attn_fwd_tile3.rs` (emularea literală a GLSL-ului, a noastră), `PROGRAMS181` / `s181` din `tools/buget-antrenare/nak_bucle.py`, secțiunea din `docs/C-157_P2_ATTN_TILE3.md`; `gpu_compute::layer::ex2_ftz` făcută publică (doar vizibilitatea, pentru emulare). Scrise de noi. Aceeași idee (ghidul public CUDA, predicația); Mesa NAK 25.0.7 (MIT) doar folosit pe gazdă, nimic copiat. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-157, punctul lui Adrian din 23 nov (a) — proba RAM ↔ VRAM cu motorul de copiere (sesiunea 183):** `classes/dma-copy/clc7b5.h` din NVIDIA open-gpu-doc (MIT, commit `9fdf5c40`, fixat în `tools/gen-rm570/generate.sh`) generat automat în `nvidia-rm570::regs::clc7b5` (cu notificarea NVIDIA); tinygrad `ops_nv.py` (MIT, commit `9cf7829a6`) citit ca specificație pentru copia într-o linie (`OFFSET_IN/OUT`, `LINE_LENGTH_IN`, `LAUNCH_DMA` = 0x182). Scrise de noi: `crates/nvidia-ga104/src/ram_vram.rs`, `gsp::user_space::alloc_copy_object`, modelul motorului de copiere din `crates/gpu-sim/src/cmdproc.rs`, testul T192.
- **C-157, pasul 2 din `docs/C-157_RAM_OFFLOAD.md` — AdamW cu m, v în RAM (sesiunea 184):** ideea mutării stărilor optimizatorului în RAM din ZeRO-Offload (Ren et al., 2021, arXiv 2101.06840, citit ca articol); `SEM_EXECUTE` `ACQ_STRICT_GEQ` din `clc56f.h` (NVIDIA, MIT, deja generat în `nvidia-rm570`); tinygrad `ops_nv.py` (MIT) citit ca specificație pentru așteptarea pe semafor. Scrise de noi: `crates/nvidia-ga104/src/adam_ram.rs`, adăugările din `train_step.rs`, testul T193.
- **C-157 punctul 7, partea a doua — lotul următor scris cât placa lucrează (sesiunea 185):** tehnica obișnuită a „dublei zone”
  (double buffering) a bufferelor de comenzi, fără o sursă anume; ordinea pe un canal și gardurile pe semafor ca în tinygrad
  `ops_nv.py` (MIT), citit ca specificație. Scrise de noi: `TwoZones`, `BatchIo`, `run_pipelined` în
  `crates/nvidia-ga104/src/kronos_batch.rs`, `Generation::batch_ahead`, testul T194.
- **C-157 punctul 2 după ciclul 15:57 — `attn_back_tile_4` (sesiunea 188):** `tools/nak/attn_back_tile_4.comp` (din
  `attn_back_tile_3.comp`, al nostru; o singură condiție schimbată), `crates/nvidia-ga104/src/attn_back_tile_4.rs`, testul
  `crates/nvidia-ga104/tests/sim_attn_back_tile_4.rs` (emularea literală, a noastră), `PROGRAMS188` / `s188` din
  `tools/buget-antrenare/nak_bucle.py`, `docs/C-157_P2_ATTN_TILE4.md`: scrise de noi, din observația proprie că urcarea scrie deja +0
  deasupra diagonalei. Mesa NAK 25.0.7 (MIT) doar folosit pe gazdă. Comparația os-376 (sesiunea 188): plafonul PCIe 3.0 x8 socotit din
  specificația publică PCIe (128b/130b, antetele pachetelor), sloturile Dell 5820 din fișa tehnică publică Dell. Nimic GPL, niciun octet
  de firmware NVIDIA.
- **C-157 punctul 2 — proba „vechi lângă nou” cu `attn_back_tile_4` (sesiunea 189):** adăugările din
  `crates/nvidia-ga104/src/attn_tile_probe.rs` (`Prog4`, `launch_bytes4`, `Probe4`, `parse_attn_tile4`), testul
  `crates/nvidia-ga104/tests/sim_c157_p2_attn_tile4_probe.rs`, nota `integration/cyber-universe/C-157_P2q_atentie4-vechi-langa-nou.md`:
  scrise de noi, după modelul probelor `_2` / `_3` (tot ale noastre). Nicio sursă externă nouă; nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 — diagnosticul legăturii PCIe (sesiunea 190):** `crates/gpu-pci/src/link.rs`, nota
  `integration/cyber-universe/C-157_P2r_legatura-port.md`, documentul `docs/C-157_PCIE_LEGATURA.md`: scrise de noi din
  specificația PCI Express (Link Capabilities / Link Status, tipul portului, magistrala secundară a punții) și din manualul public
  Dell Precision 5820 (tabelul sloturilor). Nicio sursă de cod externă; nimic GPL, niciun octet de firmware NVIDIA.
- **Ordinea 20:11 pasul 2 — pasul mixt pe forma lui `antrenare mini` (sesiunea 191):** testul
  `crates/nvidia-ga104/tests/sim_c160_p7_mixed_mini.rs`, documentul `docs/C-160_P7_PAS_MIXT_MINI.md`, nota
  `integration/cyber-universe/C-160_P7f_antrenare-mini-mixt.md`: scrise de noi. Forma modelului și legea greutăților de start citite din
  `cyber-universe/kernel/src/shell/antrenare_mini.rs` (repo-ul nostru public, doar citit); generatorul xorshift64* al nostru. Ideea scalării:
  Micikevicius et al., „Mixed Precision Training”, ICLR 2018; formatul f16: IEEE 754-2008. Nimic GPL, niciun octet de firmware NVIDIA.

- **Ordinea 20:11 pasul 3 — `ln_back_res` (sesiunea 192):** `tools/nak/ln_back_res.comp` (= `ln_back.comp` al nostru + o adunare),
  `crates/nvidia-ga104/src/ln_back_res.rs`, modelele din `gpu-compute::layer` / `gpu-sim::cmdproc`, testele T200, documentul
  `docs/C-160_P7_LIPIRE_INAPOI.md`, nota `integration/cyber-universe/C-160_P7g_ln-back-rezidual.md`: scrise de noi. Ideea lipirii (fusion)
  e generală (llm.c, PyTorch); comportamentul lui `precise` / `NoContraction` citit în specificațiile GLSL 4.60 și SPIR-V 1.6 (Khronos) și
  observat în ieșirea glslang 15.1. Codul compilat de Mesa NAK 25.0.7 (MIT). Nimic GPL, niciun octet de firmware NVIDIA.
- **Ordinea 20:11 pasul 3, partea a doua — `mm_train_gelu` (sesiunea 193):** `tools/nak/mm_train_gelu.comp` (= `mm_train.comp` al nostru
  + expresia lui `gelu_tr.comp` al nostru la scriere), `crates/nvidia-ga104/src/mm_train_gelu.rs`, modelele din `gpu-compute::layer` /
  `gpu-sim::cmdproc`, testele T201, documentul `docs/C-160_P7_LIPIRE_GELU.md`. Scrise de noi; ideea lipirii epilogului e generală
  (folosită în CUTLASS / tinygrad), fără cod copiat. Compilat cu Mesa NAK 25.0.7 (MIT), pe gazdă.
- **C-157 p.2 — `mm_tc_gelu` (sesiunea 199):** `tools/nak/mm_tc_gelu.comp` (= `mm_tc.comp` al nostru + expresia lui `gelu_tr.comp`
  al nostru la scriere), `crates/nvidia-ga104/src/mm_tc_gelu.rs`, `MixedLaunch::TcGelu` / `pack_mixed_gelu` în `train_step.rs`,
  modelul `gpu-sim::cmdproc::mm_tc_gelu_on_cpu`, testele T207, documentul `docs/C-160_P7_LIPIRE_GELU_TC.md`. Scrise de noi, fără cod
  copiat. Compilat cu a doua unealtă NAK (Mesa 26.3.1, MIT), pe gazdă. Nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 p.8 — fișa modelului și pachetele de calcul (sesiunea 201):** `crates/gpu-compute/src/card.rs`,
  `crates/nvidia-ga104/src/packs.rs`, testele T208 (`tests/sim_c157_p8_packs.rs`), documentul `docs/C-157_P8_FISA_MODELULUI.md`.
  Formatul KGR1 e al lui Kronos (hub `integrare/kronos/format_greutati.md`, citit ca specificație); cititorul e scris de noi.
  Nimic copiat, nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 p.2x — `mm_train_tile` (sesiunea 208):** `tools/nak/mm_train_tile.comp`, `crates/nvidia-ga104/src/mm_train_tile.rs`,
  modelul `gpu_sim::cmdproc::mm_train_tile_on_cpu`, testele T214 (`tests/sim_mm_train_tile.rs`), documentul
  `docs/C-157_P2x_MM_TRAIN_TILE.md`. Scrise de noi; ideea împărțirii pe blocuri 2D cu mai multe rezultate pe fir vine din articolul
  și depozitul siboehm SGEMM_CUDA (MIT, doar ideea) și din `matmul_2d.comp` al nostru. Compilat cu Mesa NAK 25.0.7 (MIT), pe gazdă.
  Nimic copiat, nimic GPL, niciun octet de firmware NVIDIA.
- **C-165 — variantele cursei pentru `mm_train_tile` (sesiunea 211):** `tools/nak/mm_train_tv.glsl` + `tools/nak/mm_tv_*.comp`,
  `crates/nvidia-ga104/src/mm_train_variants.rs`, modelul `gpu_sim::cmdproc::mm_train_tv_on_cpu`, testele T217
  (`tests/sim_mm_train_variants.rs`), documentul `docs/C-165_CURSA_MM.md`. Scrise de noi, pornind de la `mm_train_tile.comp` al nostru;
  ideea plăcilor 128 × 128 cu 8 × 8 pe fir: siboehm SGEMM_CUDA kernel 5 (MIT, doar ideea). Compilat cu Mesa NAK 25.0.7 (MIT), pe gazdă.
  Nimic copiat, nimic GPL, niciun octet de firmware NVIDIA.
- **C-165, pasul următor — `mm_train_tile_gelu` (sesiunea 214):** `tools/nak/mm_train_tile_gelu.comp`,
  `crates/nvidia-ga104/src/mm_train_tile_gelu.rs`, `Plan::with_mm_tile_gelu` în `train_step.rs`, modelul
  `gpu_sim::cmdproc::mm_train_tile_gelu_on_cpu`, unealta `tools/nak/compara_epilog.py`, testele T220, documentul
  `docs/C-165_MM_TILE_GELU.md`. Scrise de noi: corpul lui `mm_train_tile.comp` + finalul lui `mm_train_gelu.comp` (ambele ale noastre).
  Compilat cu Mesa NAK 25.0.7 (MIT), pe gazdă. Nimic copiat, nimic GPL, niciun octet de firmware NVIDIA.
- **C-157 p.2x pasul 2 — `Plan::with_mm_tile` (sesiunea 209):** `crates/nvidia-ga104/src/train_step.rs` (opțiunea, `Prog::MmTrainTile`,
  `PROGRAMS_ALL_TILE`), testele T215 (`tests/sim_c160_p5_golden.rs`). Cod al nostru, fără surse noi.
- **C-157 p.2y — `Plan::with_colsum_split` (sesiunea 210):** `crates/nvidia-ga104/src/train_step.rs` (opțiunea, `Prog::ColsumPart`,
  `PROGRAMS_ALL_SPLIT`, zona de lucru), testele T216 (`tests/sim_c160_p5_golden.rs`). Leagă în pas programul `colsum_part` al nostru
  (sesiunea 159, vezi mai sus). Cod al nostru, fără surse noi.
- **C-164 — Kronos croit pe placă (sesiunea 212):** `crates/gpu-compute/src/v2.rs` (numărătoarea lansărilor, oracolele RMS și ReLU²),
  testele T218 (`crates/nvidia-ga104/tests/c164_kronos_v2.rs`). Cod al nostru. Formule din documentația publică PyTorch (`RMSNorm`,
  `F.rms_norm`, BSD; eps implicit f32 = `finfo.eps`), Zhang & Sennrich 2019 (RMSNorm) și So et al. 2021 (Primer, ReLU²) — doar
  formulele, niciun cod copiat.
- **C-165 pasul 3 — `parse_race` (sesiunea 215):** cititorul raportului cursei, scris de noi. Formatul rândurilor e al OS-ului nostru
  (`cyber-universe`, P6-190, `race_row` din `gpu_i14.rs`, proiect propriu); în T221 șabloanele `format!` ale OS-ului sunt copiate ca să
  verificăm exact formatul real. Nicio sursă străină.
- **C-165 / os-413 — `qmd::program_map_bytes`, `RACE_HOLD` (sesiunea 216):** scrise de noi. Ideea paginii lăsate după codul
  programului: Mesa nouveau/nvc0, commit „nouveau/nvc0: increase overallocation on shader bo” (Dave Airlie, 2024, MIT; comentariul
  „Shader storage needs a 2K (from NVIDIA) overallocations at the end to avoid prefetch bugs”) și NVK (o pagină întreagă, după
  același commit). Citite ca idee; nicio linie copiată. Rândurile os-413 din T222 sunt copiate din raportul nostru (OS-ul propriu).
- **C-165 / pasul mixt pe forma #513 — `LOSS_SCALE_513`, `loss_scale_for`, T223 (sesiunea 217):** scrise de noi. Ideea scalării
  erorii (înmulțirea pierderii cu S înainte de pasul înapoi în f16, împărțirea gradientului la S înainte de optimizator) e tehnica
  publică „mixed precision training” (Micikevicius et al., 2017, arXiv:1710.03442); regula „cea mai mică putere a lui 2 cu
  S/(V·N) ≥ 2⁻¹⁰” e a noastră, din mărimea d(logits) la un softmax aproape uniform. Nicio linie de cod străin.

- **Sesiunea 218 (25 nov 2026), T224 / `train_profile::parse_profiles`:** cod scris de noi. Rândurile din `crates/nvidia-ga104/tests/rapoarte/os41{4,5}-profil.txt` sunt copiate din rapoartele automate ale OS-ului nostru (`Cyber-hub/rezultate/os-automat/2026-11-25/`); nicio sursă străină.
- **Sesiunea 219 (25 nov 2026), T225 / `MmChoice::all`, `RACE_WINNER_513`:** cod scris de noi. Rândurile din `crates/nvidia-ga104/tests/rapoarte/os416-cursa.txt` sunt copiate din raportul automat al OS-ului nostru (`Cyber-hub/rezultate/os-automat/2026-11-25/ciclu-auto-20261125-0135.txt`, rândurile 214–292); nicio sursă străină.
- **Sesiunea 220 (25 nov 2026), `tools/nak/mm_tc128.comp`, `mm_tc128.rs`, `mm_tc128_on_cpu`, T226:** cod scris de noi, pornind de la `tools/nak/mm_tc.comp` (al nostru); ideea plăcilor mari și a zonei mici pe subgrup pentru ieșire din materialele publice CUTLASS / „How to optimize a CUDA matmul” (doar idei, niciun cod copiat). Compilat cu Mesa NAK 26.3.1 (MIT) pe gazdă; binarul e ieșirea compilatorului pe codul nostru.
- **Sesiunea 224 (25 nov 2026), `occupancy.rs`, `qmd::encode_with_sm_config`, verificarea treptei în `gpu-sim`, T230:** cod scris de noi. Limitele unității pentru capacitatea 8.6 (fire, grupuri, blocuri, registre, alocarea pe grup de câte 256) și rezerva de 1 KiB pe bloc din ghidul public de programare CUDA și ghidul de optimizare Ampere (doar cifre și idei); codificarea treptei = cea deja notată din Mesa NAK `qmd.rs` (MIT). Ideea probei: studiul centrului (`Cyber-hub/idei/2026-11-25-centru-inmultirea-la-altii.md`, siboehm / ghidul CUDA, doar idei). Nicio linie de cod străin.
- **Sesiunea 229 (25 nov 2026), `tools/nak/mm_tc_pf.comp`, `mm_tc_pf.rs`, `mm_tc_pf_on_cpu`, T235, regula strictă din `citiri_seriale.py` / T234e:** cod scris de noi, pornind de la `tools/nak/mm_tc.comp` (al nostru) și de la ideea atenției gen 3 (a noastră: indice strâns, citiri în registre, zeroul după); ideea „citește felia următoare în registre cât lucrează calculul” e cunoscută public (materialele CUTLASS / „How to optimize a CUDA matmul”, doar idei). Despre numărătoarele de așteptare: ce se vede în listarea NAK (Mesa 26.3.1, MIT) a programelor noastre. Niciun cod copiat.
- **Sesiunea 235 (25 nov 2026), `batch_ram::pack_mixed_fast_via_ram`, T241, `docs/C-157_COPIERE_IN_PARALEL.md`:** cod scris de noi, compus
  din funcțiile noastre `batch_ram::pack_via_ram` (os-397) și `train_step::pack_mixed_fast` (sesiunea 234); cifrele din jurnalul OS-ului
  (os-431, os-437) și costurile pe cuvânt din os-390. Nicio sursă străină nouă, nimic din firmware-ul NVIDIA.
- **Sesiunea 236 (25 nov 2026), `persistent_study`, T242, `docs/C-157_PROGRAM_PERSISTENT.md`:** aritmetică scrisă de noi pe
  modelul valurilor (`wave_model`, al nostru) și pe timpii măsurați de OS (os-431, os-346, os-392). Cifre publice NVIDIA (fișa
  GA104 / ghidul CUDA pentru capacitatea 8.6): L2 4 MB, memoria partajată ≤ 100 KiB pe SM, 64 Ki registre pe SM, 608 GB/s. Nicio sursă
  străină de cod, nimic din firmware-ul NVIDIA.
- **Sesiunea 237 (25 nov 2026), `tools/nak/grid_barrier.comp`, `grid_barrier`, `grid_barrier_on_cpu`, T244–T245, `kronos_token::token_cbuf_for`:**
  cod scris de noi; bariera = aceeași funcție `grid_sync` ca în `token.comp` (al nostru, C-87). Așezarea `cs.group_count` la 0x0c în
  bufferul constant: citită în sursa Mesa 25.0.7 (MIT), `src/nouveau/vulkan/nvk_cmd_buffer.h` (`struct nvk_root_descriptor_table`),
  confirmată în listarea NAK (`c[0x0][0xc]`). Cifrele T244 = rândurile jurnalului OS-ului (os-443). Nimic din firmware-ul NVIDIA.
- **Sesiunea 239 (26 nov 2026), `tools/nak/token32.comp`, `kronos_token::{TOKEN32, work_list32, token32_qmd}`, `token32_on_cpu`, T248:**
  cod scris de noi; corpurile operațiilor = programele noastre separate (`layernorm256`, `q8_fused`, `q8_fused32`, `q8_matvec32`,
  `argmax`, `embed_dev`) și `token.comp` (C-87); limitele de ocupare = tabelul public al capabilității 8.6 (deja în `occupancy`).
  Compilat cu Mesa NAK 25.0.7 (MIT), unealta de pe gazdă. Nimic din firmware-ul NVIDIA.
- **C-157 — `attn_back_tile_tc` (sesiunea 243):** `tools/nak/attn_back_tile_tc.comp` (al nostru; cadrul din `attn_back_tile_3.comp`
  și `mm_tc.comp`, tot ale noastre; ideea plăcilor cu `coopMatMulAdd` din documentația publică CUTLASS / VK_KHR_cooperative_matrix,
  nicio linie copiată), `crates/gpu-compute/src/attn_tc.rs`, `crates/nvidia-ga104/src/attn_back_tile_tc.rs`, testul
  `crates/gpu-compute/tests/c157_attn_back_tc.rs` (ale noastre). Compilat cu Mesa NAK 26.3.1 (MIT), pe gazdă.
- **C-157 — `attn_back_tile_tc16`, înlocuitorii din `gpu-sim`, `attn_tc_list` / `pack_mixed_fast_attn` (sesiunea 244):**
  `tools/nak/attn_back_tile_tc16.comp` (al nostru, din `attn_back_tile_tc.comp`), `crates/nvidia-ga104/src/attn_back_tile_tc16.rs`,
  `crates/gpu-sim/src/cmdproc.rs` (`attn_back_tile_tc*_on_cpu`), `train_step.rs`, `batch_ram.rs`, testul
  `crates/nvidia-ga104/tests/sim_c157_attn_tc_mixed.rs` — scrise de noi. Locul argumentelor (0x700) citit din listarea NAK a
  programului nostru; cum socotește NAK pasul lui `coopMatLoad` dintr-un tablou de alt tip și ce înseamnă `f2fp.pack_ab` fără
  sufix: citit (ca specificație, nimic copiat) în sursa Mesa 26.3.1, MIT — `src/nouveau/compiler/nak_nir_lower_cmat.c` și
  `src/nouveau/compiler/nak/ir.rs`. Nimic din firmware-ul NVIDIA.
- **C-157 — `attn_back_tile_tc16_v4` / `attn_back_tile_tc16_v4pf` (sesiunea 248):** `tools/nak/attn_back_tile_tc16_v4.comp` și
  `attn_back_tile_tc16_v4pf.comp` (ale noastre, din `attn_back_tile_tc16.comp` și ideea citirilor din `mm_tc_v4.comp`, tot ale
  noastre), `crates/nvidia-ga104/src/attn_back_tile_tc16_v4{,pf}.rs`, `attn_back_tile_tc16_v4{,pf}_on_cpu` în
  `crates/gpu-sim/src/cmdproc.rs`, testul `crates/nvidia-ga104/tests/sim_c157_attn_tc16_v4.rs` — scrise de noi. Că NAK așteaptă
  toate citirile în zbor la o ramificare: aflat din listarea NAK a programului nostru (`tools/nak/citiri_seriale.py`).
  Compilat cu Mesa NAK 26.3.1 (MIT), pe gazdă. Nimic din firmware-ul NVIDIA.
- **C-157 — `attn_back_tile0_tc_v4` (sesiunea 251):** `tools/nak/attn_back_tile0_tc_v4.comp` (al nostru, din
  `attn_back_tile0_tc.comp` și ideea citirilor de 128 de biți din `attn_back_tile_tc16_v4.comp` / `mm_tc_v4.comp`, tot ale
  noastre), `crates/nvidia-ga104/src/attn_back_tile0_tc_v4.rs`, `attn_back_tile0_tc_v4_on_cpu` în `crates/gpu-sim/src/cmdproc.rs`,
  testul `crates/nvidia-ga104/tests/sim_c157_attn0_tc_v4.rs` — scrise de noi. Compilat cu Mesa NAK 26.3.1 (MIT), pe gazdă.
  Nimic din firmware-ul NVIDIA.
- **C-157 — os-467, reparația `attn_back_tile0_tc` / `_v4` și verificarea programelor (sesiunea 255):**
  `crates/nvidia-ga104/src/program_lint.rs`, testul `crates/nvidia-ga104/tests/lint_c157_os467.rs`, rândurile de 68 din
  `tools/nak/attn_back_tile0_tc{,_v4}.comp` — scrise de noi. Cauza citită în listarea NAK a programului nostru. Regula „pasul
  matricelor multiplu de 16 octeți”: idee din documentația publică CUDA (`wmma::load_matrix_sync` / `store_matrix_sync`, `ldm`),
  nimic copiat; că o adresă partajată nealiniată oprește firul: documentația publică `cuda-gdb` (lista excepțiilor). Compilat cu
  Mesa NAK 26.3.1 (MIT), pe gazdă. Nimic din firmware-ul NVIDIA.
- **Sesiunea 258 (26 nov 2026):** `tools/nak/mm_tc_v5.comp` (al nostru): bucla lui `mm_tc_v4.comp` neschimbată, finalul după
  `mm_tc128.comp` (o zonă 16 × 16 f32 pe grup de fire), tot ale noastre. Nicio sursă nouă din afară; compilat cu Mesa NAK 26.3.1 (MIT),
  doar unealta pe gazdă.
- **Sesiunea 260 (27 nov 2026):** `tools/nak/token32_v2.comp` (al nostru: `token32.comp` cu citirile prin buffere `readonly`),
  `kronos_token::{TOKEN32_V2, work_list_changes}`, testele T271 / T272 (ale noastre). Ce face NAK cu citirile globale și cu bariera
  (`.constant` / `.strong.sys`; `cctl` WBAll / IVAll în jurul `membar`) — citit în sursa Mesa 25.0.7
  `src/nouveau/compiler/nak/from_nir.rs` (MIT), ca specificație, nimic copiat. Compilat cu Mesa NAK 25.0.7 (MIT), pe gazdă. Nimic din
  firmware-ul NVIDIA.
- **Sesiunea 263 (27 nov 2026):** `tools/nak/mm_tc_v6.comp` (al nostru: `mm_tc_v5.comp` cu scrierile felie de câte 64 de biți
  și operanzii transpuși ținuți pe coloane), testul T275 (al nostru). Ideile (scrieri vectoriale în memoria comună, conflictele de
  bancă, regula jumătăților de warp la 64 de biți): ghidul public CUDA („shared memory”, capacitatea 8.6); ordinea `ColumnMajor` și
  tablourile de alt tip la `coopMatLoad`: specificația publică `GL_KHR_cooperative_matrix` / `SPV_KHR_cooperative_matrix`. Compilat
  cu Mesa NAK 26.3.1 (MIT), pe gazdă; `nvidia_ga104::mm_tc_v6`, `gpu_sim::cmdproc::mm_tc_v6_on_cpu` și T276 sunt ale noastre. Nimic
  copiat; nimic din firmware-ul NVIDIA.
- **Sesiunea 265 (27 nov 2026):** testul T279 și `docs/C-157_CITIRI_CONFLICTE.md` (ale noastre): adresele citirilor luate din listarea
  NAK a programului nostru `mm_tc_v6`; modelul băncilor din ghidul public CUDA (capacitatea 8.6) și forma lui `ldmatrix` din
  documentația publică PTX; pasul liniar al lui `coopMatLoad` din specificația publică `GL_KHR_cooperative_matrix`. Ideea umpluturii
  rândurilor e cunoscută public; nimic copiat; nimic din firmware-ul NVIDIA.
- **Sesiunea 266 (27 nov 2026):** `tools/nak/mm_tc_v7.comp` (al nostru: `mm_tc_v6.comp` cu 16 octeți de umplutură pe fiecare rând
  al feliilor), `nvidia_ga104::mm_tc_v7`, `gpu_sim::cmdproc::mm_tc_v7_on_cpu` și testul T280 (ale noastre). Aceleași surse publice ca
  la sesiunea 265 (ghidul CUDA, PTX `ldmatrix`, `GL_KHR_cooperative_matrix`). Compilat cu Mesa NAK 26.3.1 (MIT), pe gazdă. Nimic
  copiat; nimic din firmware-ul NVIDIA.
- **Sesiunea 269 (27 nov 2026):** `nvidia_ga104::gsp::perf::BoostPlan::covers_ms` și testul T283 (ale noastre): socoteală pe
  durata cererii de viteză (`duration_s`, câmpul deja descris din antetele NVIDIA 570.144, MIT) și pe jurnalul plăcii din os-485.
  Nicio sursă nouă; nimic copiat; nimic din firmware-ul NVIDIA.
- **Sesiunea 274 (27 nov 2026):** `tools/nak/token32_v2c.comp` (al nostru: `token32_v2.comp` cu citirea ceasului SM-ului în jurul
  fiecărei operații și scrieri într-o zonă separată), `nvidia_ga104::token_clock`, `gpu_sim::cmdproc::token32c_on_cpu`, testul T288
  (ale noastre). Ceasul: extensia publică `GL_ARB_shader_clock` (`clock2x32ARB`, specificația Khronos / `SPV_KHR_shader_clock`);
  ce instrucțiune iese (`cs2r sr[0x50]` / `sr[0x51]`) — citit în listarea NAK, nimic copiat. Compilat cu Mesa NAK 25.0.7 (MIT), pe
  gazdă. Nimic din firmware-ul NVIDIA.
