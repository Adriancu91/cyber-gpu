# Cercetare I4–I8: firmware, memorie, falcon, GSP, canale

*Sesiunea 2, 23 oct 2026. Doar documentare: **niciun cod din acest document nu scrie în placă**. Etichete: CONFIRMAT (citit într-o sursă),
DEDUS, NECUNOSCUT. Sursele sunt citite pentru idei, nimic copiat (clean-room, `PROVENANCE.md`).*

## 0. Ce s-a schimbat față de sesiunea 1
1. **GA104 nu folosește „bootloader”-ul FWSEC** (cel de pe Turing/GA100): nova-core are un HAL GA102 separat, cu
   `needs_fwsec_bootloader: false`, aplicat lui GA102+ și Ada; ambele folosesc însă **Booter pe SEC2** ca să pornească GSP-RM.
   (CONFIRMAT: patch nova-core „make use of FWSEC bootloader a property of the TU102 HAL”, v6, LKML iul. 2026.)
   → Pentru noi: un pas mai puțin la I6; FWSEC-FRTS se încarcă direct în falcon-ul GSP.
2. **Încărcarea codului semnat (HS) într-un falcon e DMA din memoria sistemului**: driverul pregătește în RAM imaginea, programează
   registrele BROM și DMA, iar falcon-ul o trage în IMEM/DMEM prin motorul FBDMA (FBIF alege RAM sau VRAM); Boot ROM-ul verifică semnătura.
   (CONFIRMAT: documentația oficială a kernelului „nova/core/falcon”.) → **I6 cere deja Bus Master + un buffer DMA**, nu doar I5.
3. **VT-d e oprit pe workstation** (K0, FAPT): adresele DMA văzute de placă = adresele fizice. Simplu, dar fără plasă de siguranță.

## 1. I4 — fișierele de firmware (de unde, cât, cum ajung în OS)
| Fișier (linux-firmware, `nvidia/ga102/…`, legat din `ga104/`) | Rol | Unde rulează |
|---|---|---|
| `gsp/booter_load-<ver>.bin` | Booter: autentifică și pornește GSP-RM | SEC2, HS |
| `gsp/booter_unload-<ver>.bin` | oprirea curată (la descărcare) | SEC2, HS |
| `gsp/bootloader-<ver>.bin` | încărcătorul RISC-V al GSP-RM | GSP |
| `gsp/gsp-<ver>.bin` | GSP-RM (ELF, zeci de MB) | GSP RISC-V, LS |
| FWSEC (în **VBIOS**, nu în linux-firmware) | devinit + FRTS | GSP, HS |

- Versiuni disponibile: 535.113.01 și 570.144 (inventar în `nvidia-ga104/src/firmware.rs`). nouveau folosește r535; nova-core r570.
  **Recomandare tehnică: o singură versiune, 570.144**, pentru că nova-core (cel mai activ, documentat) o folosește. (DEDUS)
- Transport în OS: fișierele nu pot sta în repo (regula proiectului). Variante: (a) pe stick, în `/usb/firmware/nvidia/`, copiate de Adrian
  din pachetul `linux-firmware`; (b) în initrd la construirea Release-ului (cere ca CI-ul OS să le descarce — redistribuire!). (a) e mai curată juridic.
- Verificare la încărcare: mărime + SHA-256 față de valori notate în `firmware.rs` (biblioteca `crypto/` a OS-ului are SHA-256).
- **Blocaj: decizia lui Adrian pe licență** (§5).

## 2. I5 — memoria pentru GSP (în RAM, prin DMA)
CONFIRMAT din ordinea pașilor (nouveau `gsp/r535.c`, nova-core `gsp/`): în RAM se pregătesc
- imaginea GSP-RM + tabela de pagini „radix3” care o descrie (GSP-ul o citește prin DMA);
- `GspFwWprMeta` (adresele și mărimile regiunilor din VRAM: WPR2, heap, FRTS, boot binary, ELF);
- cozile de mesaje cmdq/msgq (0x40000 octeți fiecare) + memoria partajată „libos”.
DEDUS pentru noi: totul încape în fereastra DMA a OS-ului (16 MiB – 4 GiB), cu excepția imaginii GSP-RM (~zeci de MB) care trebuie
copiată în pagini fizice (nu neapărat contigue — radix3 le descrie). Alocatorul OS nu golește paginile → le golim noi (observația 4).
Bus Master = **prima scriere în configurația plăcii** din tot proiectul → doar cu aprobarea lui Adrian, într-o fereastră pe workstation.

## 3. I6 — FWSEC-FRTS pe falcon-ul GSP
- Ce face: creează regiunea protejată WPR2 în VRAM și pune acolo FRTS; doar cod HS poate face asta. (CONFIRMAT: docs falcon)
- Ce trebuie: imaginea FWSEC găsită la I3 (tabela BIT → jetonul 0x70 → tabela PMU, aplicația 0x85 = FWSEC_PROD → descriptorul
  `FalconUCodeDescV3`), semnătura potrivită cu versiunea din siguranțele („fuse”) plăcii, un buffer DMA, resetarea falcon-ului GSP,
  registrele BROM/DMA, pornirea și așteptarea terminării (cu limită de timp). Apoi verificarea SCRATCH_E și a începutului WPR2.
- Riscul: e **prima scriere în registrele BAR0**. Pe GPU-ul care ține ecranul. Un falcon pornit greșit poate lăsa placa într-o stare din
  care doar o repornire o scoate. Mitigare: fereastră cu Adrian, Kronos oprit, totul în jurnal (valoare înainte/după, D-013).
- Ce lipsește ca să scriem codul: offset-urile exacte ale registrelor falcon pe GA10x (nouveau/nova le au; open-gpu-doc are
  `dev_falcon_v4.h` pentru Ampere — de verificat dacă e publicat sub MIT), algoritmul de alegere a semnăturii după fuse. NECUNOSCUT încă.

## 4. I7–I8 — Booter pe SEC2, GSP-RM, RPC
- I7: GSP în mod RISC-V, adresa libos în registrele de „mailbox”, Booter_load pe SEC2 cu adresa `wpr_meta`; așteptare (nova: 5 s).
- I8: GSP-RM trimite „init done” prin msgq după ce driverul răspunde la `SetSystemInfo`, `SetRegistry` și la cererile secvențiatorului.
  Structurile RPC **depind de versiune** (r535 ≠ r570) și nu au documentație oficială, doar antetele MIT din open-gpu-kernel-modules.
- Volum (ESTIMARE, după codul nouveau `r535.c` și nova-core `gsp/`): I6–I8 ≈ 6 000–10 000 de rânduri de Rust la noi, 2–4 luni de lucru
  efectiv, cu 5–15 sesiuni pe workstation. Este partea cea mai mare din drumul până la primul calcul.

## 5. Decizii care sunt ale lui Adrian — argumentele, nu decizia
### 5.1 Firmware-ul semnat NVIDIA (GSP-RM, Booter)
| | Pentru | Contra |
|---|---|---|
| Îl folosim | **singura cale tehnică** spre calcul pe RTX 3070 Ti (Ampere refuză firmware nesemnat; firmware propriu = imposibil); licența §2.1.2 permite redistribuirea pentru OS-uri cu licență OSI, iar OS-ul e MIT | zeci de MB de cod NVIDIA opac rulează în placă; nu respectă „90% propriu” pentru partea GPU; interdicția de inginerie inversă (§2.1.3) ne leagă mâinile la depanare; o versiune fixă (570.144) trebuie urmărită |
| Nu-l folosim | totul rămâne 100% propriu | **niciun calcul pe GPU-ul NVIDIA** din OS-ul nostru; GPU-ul rămâne doar pentru Windows/Kronos actual; K6-GPU se mută pe o placă documentată deschis (AMD, ex. iGPU-ul Renoir de pe laptop, unde firmware-ul e tot binar, dar protocolul e documentat) |
Juridic: noi nu suntem juriști; textul licenței pare compatibil, dar Adrian decide (eventual cu un avocat).

### 5.2 Compilatorul pentru programele GPU (SASS SM86)
| | Mesa NAK | NVIDIA `ptxas` |
|---|---|---|
| Licență | MIT; poate rula pe orice calculator | EULA CUDA: doar pe sisteme cu GPU NVIDIA; fără inginerie inversă a ieșirii |
| Unde rulează | pe gazdă (build), nu în OS | pe workstation (are RTX), nu în OS |
| Calitate cod | bună pentru shadere/compute Vulkan; fără tensor cores bf16 expuse ușor (DEDUS) | cea mai bună; tensor cores prin PTX `mma` |
| Risc | API intern Mesa, se schimbă; trebuie împachetat ca unealtă | juridic (parsarea cubin-ului), dependență de NVIDIA |
Recomandarea tehnică (nu decizia): **NAK pentru primul program** (adunarea a 64 de f32 — nu are nevoie de tensor cores), iar întrebarea
`ptxas` se pune doar dacă performanța matmul bf16 o cere, după K6-GPU.

## 6. Ce putem face fără placă (și ce nu)
- Putem: extragerea FWSEC din VBIOS (I3+, doar citire, pe imaginea sintetică); modelul de date `GspFwWprMeta` și calculul regiunilor WPR2
  (pur aritmetic, testabil); codificarea mesajelor RPC (după antetele MIT, versiunea fixată); tabela radix3 (pur aritmetic).
- Nu putem fără Adrian: decizia de licență (blochează I4 și tot ce urmează), orice scriere în placă (I5+), testul I0–I3 pe workstation.

## Surse citite în această sesiune
- Documentația kernelului Linux: „nova/core/falcon”, „nova/core/vbios” (docs.kernel.org) — descrieri, nu cod.
- Patch nova-core „gsp: make use of FWSEC bootloader a property of the TU102 HAL” v6 (lkml.iu.edu/2607.1/03099.html).
- Patch nouveau „devinit/tu102-: wait for GFW_BOOT_PROGRESS == COMPLETED” (code.opensuse.org, kernel-source).
- nova-core `gfw.rs` (kernelsources.org) — procedura, nu codul.
