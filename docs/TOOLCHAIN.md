# Toolchain și programe GPU (G6)

## Separarea
| Strat | Azi |
|---|---|
| Limbajul sursă al programelor | nedecis (opțiuni mai jos) |
| Compilator / assembler | **Mesa NAK pe gazdă** (`tools/nak/`, C-18); nimic în OS |
| Formatul binar | SASS SM86 (codificare **nedocumentată oficial**), ambalat de NVIDIA în cubin (ELF) |
| Loader | inexistent; `KernelRef::Binary` e refuzat de backend-ul GA104 |
| Protocolul de lansare | QMD V03_00 (`clc7c0qmd.h`, MIT) + metode `AMPERE_COMPUTE_B` — necodificat încă |
| Driverul | `nvidia-ga104` (blocat la I2/I4) |
| Runtime-ul tensorial | `gpu-compute` (referință CPU) |

## Decizia (26 oct 2026, 17:10, C-18): **Mesa NAK**, ca unealtă pe gazdă
NAK rulează **doar pe calculatorul de build** (gazda Linux / CI), niciodată în OS. Ieșirea lui (cod SASS SM86) e un fișier de date pe care
driverul îl încarcă. `ptxas` nu se folosește; întrebarea revine doar dacă, după K6-GPU, performanța matmul bf16 o cere.
Pregătirea uneltei: secțiunea „Unealta NAK pe gazdă” de mai jos.

## Opțiunile cântărite înainte de decizie
1. **Mesa NAK** (`src/nouveau/compiler/nak/sm70_encode.rs`, MIT, © Collabora): codifică SASS pentru SM ≥ 70, inclusiv Ampere
   (`is_ampere()` = SM 80–88). Folosit **ca unealtă pe gazdă** (generăm binare, nu copiem cod). Licența MIT permite și portarea
   cu atribuire, dar `docs/CLEAN_ROOM.md` din OS cere scriere proprie după specificație. **Recomandare:** unealtă pe gazdă la început.
2. **CUDA `ptxas`** pe gazdă → cubin. EULA CUDA: dezvoltare doar pe sisteme cu GPU NVIDIA (§2.1), fără inginerie inversă a SDK-ului
   sau a ieșirii lui pentru alte platforme (§1.2). Proprietatea asupra ieșirii nu e tratată explicit; compilatoarele nu sunt redistribuibile.
   Utilizabil pe workstation (are GPU NVIDIA), dar parsarea cubin-ului de către noi trebuie verificată juridic. **Risc de licență.**
3. **Assembler propriu** după specificație: nu există documentație oficială a codificării → ar trebui derivată din NAK (dependență intelectuală).

Nu inventăm un format binar. Niciun program nu e „executat pe GPU” până nu există toți pașii I0–I10 și un rezultat comparat cu
`gpu-compute::reference`.

## Primul program propus (pentru G5)
Adunarea a doi vectori f32 de 64 de elemente (`OpCode::Add`), un singur bloc de 64 de fire, rezultat comparat bit cu bit cu
`reference::add` (exact în f32). Avantaj: nicio problemă de rotunjire, deci orice diferență e un defect al lanțului.

## Unealta NAK pe gazdă (sesiunea 5, 26 oct 2026)
Pregătită și folosită o dată: `tools/nak/` (README, `build.sh`). Mesa 25.0.7 construit în cloud (NVK + NAK + drm-shim nouveau, fără placă),
`NOUVEAU_CHIPSET=174` (GA104, SM 8.6), `NAK_DEBUG=print`. Primul program, adunarea a 64 de f32 (`tools/nak/add64.comp`), compilat:
**15 instrucțiuni, 240 de octeți**, `tools/nak/out/add64.sm86.sass` (SHA-256 `a890b75c…98ce`). **Compilat, nerulat**: rularea vine după I8–I10, pe placă.
Constrângeri pentru lansator (I10): adresele a/b/c în bufferul constant 0 la 0x28/0x30/0x38, ≥ 19 registre, 64×1×1 fire.
Decizie de lucru: NAK rămâne unealtă pe gazdă (nu în CI: construirea Mesa durează ~15 min); `out/` se regenerează doar când se schimbă programul.
