# B1 — înmulțiri pe 8 biți pe tensor cores (IMMA), pasul 1–2

Pista B1 din `idei/2026-11-24-centru-viteza-peste-windows.md` (hub), aprobată de Adrian (24 nov 21:39):
întâi răspunsul „poate NAK int8 IMMA pe GA104?”, apoi programul în SIMULARE, apoi o probă mică pe placă
(un singur lucru nou), nu înaintea lui `mm_tc` / cursei de noapte.

## Pasul 1 — răspunsul: DA, NAK scoate IMMA pentru SM 8.6

- Programul de probă `tools/nak/mma16_s8.comp` (C-147 punctul 5, compilat atunci, nemodelat):
  `D[16,16] = A[16,32] · B[32,16] + C[16,16]`, A și B întregi cu semn pe 8 biți, C și D întregi pe 32 de biți,
  toate pe rânduri; un warp de 32 de fire.
- Compilat cu a doua unealtă NAK (Mesa 26.3.1, `tools/nak/build2.sh`): `tools/nak/out2/mma16_s8.sm86.sass`,
  2 368 octeți, SHA-256 `7f6bc59a…c84af`, 32 de registre, 0 memorie partajată, fără barieră.
- În listare (`tools/nak/out2/mma16_s8.sm86.txt`) apar **două `imma.m16n8k32.i8.i8`** (câte una pe jumătate de 8 coloane),
  adică exact instrucțiunea de tensor core pentru întregi pe 8 biți. A e citită cu cuvinte de 4 octeți, B octet cu octet
  (pentru ordinea pe care o cere instrucțiunea), C cu citiri de 8 octeți. Nicio instrucțiune `.sat` (fără saturare).
- Viteza posibilă (DEDUS, nemăsurat): whitepaper-ul NVIDIA GA102 (Anexa A, plăcile GeForce) dă pentru tensor cores
  int8 de **2 ori** viteza f16 cu acumulare f16 și de **4 ori** viteza f16 cu acumulare f32 (cea pe care o folosește `mm_tc`).
  Presupunem același raport pe GA104 (aceeași generație de SM); de verificat pe placă.

## Pasul 2 — programul în SIMULARE (sesiunea 223)

Diferența mare față de `mma16_f16`: la întregi **nu există rotunjire**. Fiecare produs a·b încape în 15 biți, suma a 32
de produse în 20 de biți; adunarea pe întregi modulo 2³² dă **același rezultat în orice ordine**. Deci aici putem cere
**aceiași biți** cu oracolul, nu o toleranță — singura ipoteză e ce se întâmplă la depășirea lui i32 (vezi mai jos).

Ce se adaugă (doar adăugări, nimic redenumit):
- `gpu_compute::mma_s8`: oracolul `mma16_s8_ref` (în i64, apoi verificat că încape în i32) și modelul `mma16_s8_wrap`
  (adunare modulo 2³², ipoteza pentru depășire);
- `nvidia_ga104::mma16`: `MMA16_S8` (octeții programului), `MMA16_S8_REGISTERS`, `cbuf_s8`, `qmd_s8`
  (aceeași așezare ca la `mma16_f16`: a, b, c, d la 0x700 / 0x708 / 0x710 / 0x718);
- `gpu_sim::cmdproc`: `mma16_s8_on_cpu` + `MMA16_S8_NEEDS`.

### Verificări (scrise înainte de cod, T229, SIMULARE)
- **T229a** octeții programului = fișierul NAK, SHA-256 `7f6bc59a…c84af`, 32 de registre în QMD, bloc 32×1×1.
- **T229b** 8 seturi aleatoare (A, B în −128…127, C în ±2²⁰) trimise prin modelul procesorului de comenzi
  (SET_OBJECT, SEND_PCAS_A, SEND_SIGNALING_PCAS2_B, semaforul gazdei): D = oracolul **bit cu bit**, 256 / 256 pe fiecare set;
  D pornește cu `0x7fff_fff7` peste tot, ca un element nescris să nu poată trece.
- **T229c** marginile: toate −128 × −128 (fiecare element = 32 · 16 384 + C = 524 288 + C), toate −128 × 127,
  A = identitatea pe primele 16 coloane (D = primele 16 rânduri ale lui B + C): oracolul trece, modelul = oracolul.
- **T229d** depășirea: C = i32::MAX și produse pozitive → oracolul spune „nu încape” (`None`), modelul dă valoarea
  modulo 2³² (IPOTEZĂ: PTX fără `.satfinite`; pe placă nu se trimit astfel de date până nu e măsurat).
- **T229e** lansările greșite sunt refuzate și D rămâne neatins: un octet schimbat în program, adresele la 0x28
  (așezarea primei unelte), o adresă nealiniată, bloc de 64.
- **T229f** programul f16 și cel s8 înregistrate în același procesor de comenzi: fiecare QMD rulează programul lui
  (nu se încurcă după SHA).

Predicție: toate 6 trec din prima; dacă T229b nu iese bit cu bit, greșeala e în așezarea A/B (rânduri vs coloane)
din model, nu în aritmetică.

## Surse
- Mesa NAK (MIT), `src/nouveau/compiler/nak/` — doar unealtă pe gazdă (vezi `docs/PROVENANCE.md`); nimic copiat.
- PTX ISA, instrucțiunea `mma` cu `.s8` / `.s32` și opțiunea `.satfinite` (documentație publică NVIDIA).
- NVIDIA Ampere GA102 GPU Architecture Whitepaper, Anexa A:
  https://images.nvidia.com/aem-dam/en-zz/Solutions/geforce/ampere/pdf/NVIDIA-ampere-GA102-GPU-Architecture-Whitepaper-V1.pdf
