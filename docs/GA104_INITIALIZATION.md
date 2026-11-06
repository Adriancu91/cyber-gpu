# GA104 — inițializare (G3)

> **Din 26 oct 2026:** starea fiecărei componente se ține într-un singur loc, `docs/COMPATIBILITATE_FIRMWARE.md` (C-18). Acest document rămâne ca istoric al cercetării.

*Surse verificate pe 23 oct 2026 (fișiere brute din repo-uri, API GitHub/GitLab). „Confirmat” = citit în sursă;
„dedus” = derivat din ce am citit. Liniile pot migra pe `master`.*

## 1. Identitate

| Fapt | Sursă (licență) | Stare |
|---|---|---|
| `10de:2482` = GeForce RTX 3070 Ti (GA104); celelalte ID-uri GA104 în `nvidia-ga104/src/ids.rs` | pci.ids, l. ~13578–13618 (GPL-2.0+/BSD-3); README open-gpu-kernel-modules l. 777–782 (MIT) | confirmat |
| `NV_PMC_BOOT_0` @0x0: MINOR 3:0, MAJOR 7:4, IMPLEMENTATION 23:20, ARCHITECTURE 28:24; Ampere = 0x17 | open-gpu-doc `manuals/ampere/ga100/dev_boot.ref.txt` (MIT) | confirmat |
| `NV_PMC_BOOT_42` @0xA00: CHIP_ID 28:20 etc. | același manual | confirmat |
| nova-core declară arhitectura BOOT_42 ca 29:24 (NVIDIA: 28:24) | `drivers/gpu/nova-core/regs.rs` (GPL-2.0, citit, necopiat) | **discrepanță**: raportăm bitul 29 separat |
| nouveau: `chipset = (boot0 & 0x1ff00000) >> 20`, 0x174 = GA104 | `nvkm/engine/device/base.c` l. 3190–3215, 3348 | confirmat |

## 2. BAR-uri (din `lspci` publicate, nu din documentație NVIDIA)
BAR0 32 biți neprefetchable 16 MiB; BAR1 64 biți prefetchable 256 MiB (8 GiB observat pe un laptop 3070 Ti, probabil ReBAR);
BAR3 64 biți prefetchable 32 MiB; I/O 128 B; ROM 512 KiB. Surse: bbs.archlinux.org (3060 GA104), pciutils issue #151.
**Stare: observat pe alte plăci.** Pe workstation (K0, din Windows, FAPT): BAR0 16 MiB @F9000000, BAR1 256 MiB @E0000000 (ReBAR oprit),
BAR3 32 MiB @F0000000, I/O 128 B — vezi [HARDWARE_WORKSTATION.md](HARDWARE_WORKSTATION.md). Încă necitite de codul nostru pe placă.

## 3. Secvența GSP-RM pe GA10x (confirmată în cod, neimplementată de noi)
Din nouveau `nvkm/subdev/gsp/tu102.c` (refolosit de `ga102.c`; MIT) și nova-core `gsp/boot.rs`, `gsp/hal/tu102.rs` (GPL-2.0, doar ordinea):

1. Așteaptă GFW_BOOT (devinit-ul e lansat de FWSEC pe PMU la pornire; nova `devinit.rst`). → **I2**
2. Citește VBIOS, găsește FWSEC. → **I3**
3. Pregătește WPR2 sub VBIOS: FRTS, boot binary, ELF, heap, `GspFwWprMeta`. → **I4/I5**
4. Rulează FWSEC-FRTS pe falcon-ul GSP; verifică SCRATCH_E; verifică începutul WPR2. → **I6**
5. Pune GSP în mod RISC-V, scrie adresa libos (falcon +0x040/+0x044); booter_load pe **SEC2** cu adresa `wpr_meta`. → **I7**
6. Așteaptă RISC-V activ (nova: 5 s), `SetSystemInfo`, `SetRegistry`, secvențiatorul, „init done”. → **I8**
7. Cozile cmdq/msgq (0x40000 fiecare) sunt **în memoria sistemului**, prin DMA coerent → **necesită Bus Master + IOMMU cunoscut**.

Pe Ampere se rulează numai firmware semnat de NVIDIA; GPU-ul verifică semnăturile. Firmware propriu = **imposibil** (Pista B închisă, plan v0.2 §6, confirmat).

## 3b. I2 și I3 — implementate (doar citire), testate doar în SIMULARE (sesiunea 2)

**I2 — GFW_BOOT** (`nvidia-ga104/src/gfw.rs`)
| Registru | Offset BAR0 | Ce înseamnă | Sursă |
|---|---|---|---|
| `NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_PRIV_LEVEL_MASK` | `0x118128` | bit 0 = FWSEC a coborât protecția; abia apoi se poate citi registrul următor | nouveau `devinit/tu102.c` `tu102_devinit_wait` (MIT) — confirmat; nova-core `gfw.rs` (doar procedura) |
| `NV_PGC6_AON_SECURE_SCRATCH_GROUP_05_0_GFW_BOOT` | `0x118234` | biții 7:0 = progres; `0xff` = terminat | idem |

Procedura: la fiecare ~1 ms citește masca; numai dacă bitul 0 e 1 citește progresul; terminat la `0xff`. Limită 4 s (ca nova-core;
nouveau: ~2050 × 1–2 ms). `0xffffffff` → placa nu răspunde; `0xbadfxxxx` (eroare de acces) → defect hardware, nu dată. Jurnalul păstrează
doar valorile finale (o așteptare poate avea mii de citiri).

**I3 — VBIOS prin PROM** (`nvidia-ga104/src/vbios.rs`)
- Fereastra: BAR0 + `0x300000` (nova-core, documentația oficială a kernelului „nova/core/vbios”; nouveau `shadowprom.c`); limita noastră: 1 MiB.
- Opțional antet IFR „NVGI” (versiunile 1–2: începutul imaginilor la `FIXED_DATA_SIZE + 4`; versiunea 3: director „RFRD”) — după documentația nova.
- Lanțul de imagini PCI: `0xAA55`, PCIR (Specificația PCI Firmware 3.0), extensia NVIDIA „NPDE” (mărime + „ultima imagine”) — DEDUS din nouveau `npde.c`.
- **Pe placa reală (28 oct, C-30):** la `0x24600` imaginea începe cu „NV” (`0x4E56`), nu cu `0xAA55` → parserul vechi s-a oprit. nouveau `image.c`
  (MIT) acceptă ca semnături de imagine `0xAA55`, `0xBB77`, `0x4E56`, iar `pcir.c` structurile „PCIR”, „RGIS”, „NPDS”; nova-core la fel (`AA55`/`4E56`,
  „PCIR”/„NPDS”). Parserul le acceptă acum (sesiunea 13); orice altă semnătură e tot refuzată. De confirmat la următoarea fereastră.
- **Diagnosticul I3** (`nvidia-ga104/src/vbios_diag.rs`, C-30): listează structura lanțului rând cu rând (offset, semnătură, PCIR, NPDE,
  rupturi cu primii 16 octeți, căutare înainte la pas de 512 B), doar citire, fără să copieze imaginile; în OS: `gpu vbios --detalii` (C-31).
- Tipuri: `0x00` PCI-AT, `0x03` EFI, `0x70` NBSI, `0xE0` FWSEC. Tabela BIT (`FF B8 'BIT'`) în imaginea PCI-AT; jetonul `0x70` = datele Falcon.
- **Criteriul I3:** PCI-AT cu tabelă BIT care are jetonul 0x70 + cel puțin o imagine FWSEC.
- Ce nu facem la I3: extragerea codului FWSEC (tabela PMU, aplicația 0x85) — vine la I6; interiorul firmware-ului rămâne blob opac.
- Limite: 2 s, maximum 300 000 de citiri, nicio citire în afara ferestrei.
- **Risc știut:** nouveau oprește „umbrirea ROM” înainte de citire (o scriere în configurație). Dacă pe placa noastră fereastra nu arată
  semnătura, I3 se oprește cu `Blocked` — nu încercăm scrierea fără aprobarea lui Adrian.

**Fereastra BAR0 necesară:** 4 MiB (`regs::INIT_WINDOW`), nu doar prima pagină. Kernelul trebuie să mapeze UC primii 4 MiB din BAR0
(observația pentru OS e în hub, `integrare/cyber-gpu/`).

## 4. Calea fără GSP (ACR/SEC2 + FECS/GPCCS semnate)
nouveau a adăugat GR pe GA102 în Linux 6.2 (`c4bdac754ca0`), cu firmware-uri din `linux-firmware/nvidia/ga102` și `ga104/gr`.
Și aceasta cere firmware semnat, DMA și cunoașterea protocolului ACR. Nu e mai simplă; e o alternativă dacă GSP-RM se dovedește prea opac.

## 5. Firmware și licență
- `nvidia/ga104/` conține fișiere proprii doar în `gr/`; restul (`acr`, `sec2`, `nvdec`, `gsp`) sunt legături spre `ga102/`.
  GSP: `booter_load`, `booter_unload`, `bootloader`, `gsp` pentru versiunile 535.113.01 și 570.144. Lista: `nvidia-ga104/src/firmware.rs`.
- `LICENSES/LICENCE.nvidia` §2.1.2 permite redistribuirea „solely for use on operating systems” cu licență aprobată OSI,
  fără modificarea fișierelor și cu licența inclusă. §2.1.3 interzice inginerie inversă/dezasamblare.
  **Cyber Universe OS e MIT (OSI)** → pare compatibil, dar **decizia e a lui Adrian** (NECUNOSCUT juridic, nu tehnic).
  Interdicția de dezasamblare înseamnă: tratăm fișierele ca blob-uri opace; nu le analizăm interiorul.

## 6. Canale și lansare (pentru I9–I10)
- Clase (open-gpu-doc, MIT): `AMPERE_CHANNEL_GPFIFO_A 0xC56F` (`classes/host/clc56f.h`), `AMPERE_COMPUTE_B 0xC7C0`
  (`classes/compute/clc7c0.h`), QMD V03_00 în `clc7c0qmd.h`.
- Antet de metodă și intrare GPFIFO: `clc56f.h` l. 265–332 → implementat în `push.rs` (doar codificare).
- Doorbell: pe Volta e la `0x810000+0x90`; pe GA10x nouveau folosește VFN `0xb80000` + user `0x030000` → `0xBB0090` (dedus la început).
  **Sesiunea 52 (5 nov): aceeași valoare e în antetul MIT 570.144** — `tu102/dev_vm.h`: `NV_VIRTUAL_FUNCTION_DOORBELL = 0x30090`, la baza
  `NV_VIRTUAL_FUNCTION_FULL_PHYS_OFFSET` = `0xB80000` (generată în `nvidia-rm570`, `push::DOORBELL`, test T90). Calea din RM 570.144 (citită ca
  specificație): `kfifoUpdateUsermodeDoorbell_GA100` → `_TU102` când RM rulează pe procesor → `GPU_VREG_WR32(NV_VIRTUAL_FUNCTION_DOORBELL, token)`;
  `gpuGetVirtRegPhysOffset_TU102` dă baza `0xB80000` (fără SR-IOV). Stare: **DOCUMENTAT, încă NEOBSERVAT pe placă**. Prima scriere = I10, în lanț,
  cu Adrian de față (C-72); nu se scrie în afara lanțului.
- Pe calea GSP, canalele se alocă **prin RPC către RM**, nu prin registre directe.

## 6b. I11–I12 (C-83) — scrise și testate doar în SIMULARE (sesiunea 57)
- **I11, copierea mare:** un program de calcul (`copy_b128`, 16 octeți pe fir, NAK) pe **același canal de lucru și același obiect** ca I10 — nicio
  clasă nouă, niciun obiect RM nou, niciun registru nou. Singurul lucru nou pentru placă: PTE-uri care arată spre **RAM** (aperture
  `SYSTEM_COHERENT_MEMORY`, `ADDRESS_SYS` = adresa de magistrală a paginii DMA a OS-ului, `VOL` = 1). DEDUS, neobservat. Motorul de copiere
  (`AMPERE_DMA_COPY_B`, al doilea canal ca la tinygrad) rămâne pentru mai târziu, dacă viteza nu ajunge.
  Ordinea într-o intrare GPFIFO: invalidare cache → marca 0 → copiere RAM→VRAM → marca 1 → copiere VRAM→RAM (alt buffer) → marca 2. Fiecare
  marcă are `RELEASE_WFI`, deci copierile merg una după alta; vitezele = octeți / (diferența mărcilor). Verificare: al doilea buffer RAM =
  modelul `pattern_word`, cuvânt cu cuvânt; 3 cuvinte din VRAM citite prin fereastră.
- **I12, y = W·x cu W în Q8_0:** `q8_matvec` (NAK): un grup de 32 de fire pe rând, fiecare fir blocurile t, t+32, …, apoi suma în arbore în memoria
  partajată (128 B). NAK a scos bariera hardware (un singur warp) → QMD cu 128 B partajată și 0 bariere. Operațiile citite în fișa NAK: `FMUL.FTZ`
  pentru primul termen, 31 `FFMA.FTZ`, `FFMA.FTZ` scara × suma, 5 `FADD.FTZ` în arbore → oracolul `gpu_compute::q8::matvec_gpu_order`.
  Verdictul (scris înainte): **identic** cu oracolul; altfel **în marjă** față de motorul OS (`2·k·2⁻²⁴·Σ|termeni|`) = trece, de analizat; altfel
  **nu trece**. Intrarea ferestrei: 3072 × 768 sintetic (forma `mlp.0` a lui Kronos), fără fișier.
- **I13, un strat Kronos (sesiunea 58):** 16 lansări pe poziție, în ordinea motorului procesorului: LayerNorm → qkv (`q8_matvec`) → +bias (`vadd`)
  → cheia și valoarea copiate în cache-uri (`copy_b128`) → atenția (un warp pe cap, softmax cu max/sumă în arbore) → proiecția + bias → rezidual
  → LayerNorm → `fc` + bias → GELU erf → `proj` + bias → rezidual înapoi în x. Programe noi: `vadd`, `layernorm`, `attention`, `gelu`. Toleranța
  (scrisă înainte): |placă − procesor| ≤ 1e-4 + 1e-4·|x| pe fiecare element. Intrarea ferestrei: strat sintetic 768/12, 4 poziții.

## 7. Necunoscute
1. Structurile RPC GSP-RM depind de versiune (r535 vs r570); nu au documentație oficială, doar codul nouveau/nova și headerele MIT ale NVIDIA.
2. Registrul exact și semnificația GFW_BOOT pe GA104 (nova `devinit.rst` descrie procedura; adresa trebuie extrasă și verificată).
3. ~~Starea IOMMU pe workstation~~ — **VT-d oprit** (poza BIOS, 23 oct): placa vede adresele fizice direct; poarta M2 rămâne obligatorie.
4. Dacă placa e GPU-ul principal (afișaj UEFI GOP): orice reset ar stinge ecranul; protocol de recuperare inexistent.

## 8. Criterii de verificare pe hardware (în ordine)
| Pas | Criteriu PASS |
|---|---|
| I0 | `gpu identifica`: `10de:2482`, clasa 03, BAR0 mem32 neprefetchable, D0; `0 scrieri in placa` în jurnal |
| I1 | `NV_PMC_BOOT_0` & 0x1ff00000 = 0x17400000; BOOT_42.chip_id = 0x174; bitul 29 notat |
| BAR | (opțional, `gpu dimensiuni da`) BAR0 16 MiB, BAR3 32 MiB, BAR1 = valoarea ReBAR; ecranul rămâne funcțional; restaurare verificată |
| I2 | `0x118128` bit 0 = 1 și `0x118234 & 0xff = 0xff` în < 4 s; 0 scrieri |
| I3 | semnătură `0xAA55` (sau „NVGI”) la BAR0+0x300000; lanț PCI-AT → … → FWSEC (0xE0); BIT cu jetonul 0x70; 0 scrieri |
