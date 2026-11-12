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
