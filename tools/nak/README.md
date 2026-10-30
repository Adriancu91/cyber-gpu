# Unealta NAK pe gazdă (C-18) — programele GPU pentru SM 8.6

**Ce e:** compilatorul Mesa NAK (MIT), folosit doar pe calculatorul de build (Linux), niciodată în OS. Compilează un program GLSL de
calcul pentru placa NVIDIA aleasă **fără placă**: driverul Vulkan NVK din Mesa crede că vorbește cu un GA104, prin „drm-shim”-ul nouveau
(un dispozitiv fals care nu execută nimic; `NOUVEAU_CHIPSET=174` = GA104 → SM 8.6), iar `NAK_DEBUG=print` tipărește codul codificat.

**Fișiere:** `add64.comp` (primul program: `c[i] = a[i] + b[i]`, 64 de f32, un grup de 64 de fire), `nak_dump.c` (programul gazdă
Vulkan care doar creează pipeline-ul), `build.sh` (construiește Mesa o dată, apoi compilează și extrage), `out/` (rezultatul, păstrat în repo).

## Comenzi
```sh
sh tools/nak/build.sh mesa    # o singura data: Mesa 25.0.7 (sursa verificata cu SHA-256) -> ~/.cache/cyber-gpu-nak
sh tools/nak/build.sh add64   # -> tools/nak/out/add64.sm86.sass (binar) + add64.sm86.txt (asamblare NAK + cuvintele)
```
Dependențele sunt în capul lui `build.sh`. Construirea Mesa: ~15 min pe 2 nuclee (cloud, 26 oct 2026).

## Rezultatul de azi (26 oct 2026, sesiunea 5, cloud) — COMPILAT, NU RULAT
| Ce | Valoare |
|---|---|
| Unealta | Mesa 25.0.7, NVK + NAK, drm-shim nouveau, cip `0x174` (SM 8.6 după `sm_for_chipset` din Mesa) |
| Program | `add64.comp` → SPIR-V (glslang 15.1) → NAK |
| Ieșire | 15 instrucțiuni × 16 octeți = **240 octeți**, 19 registre, 0 memorie locală |
| SHA-256 `add64.sm86.sass` | `a890b75c4207b216c2f959da3b59745dbb43f3cbd58490b37b7b4c1dd12698ce` (identic la două rulări) |
| Argumentele | cele 3 adrese pe 64 de biți sunt citite din **bufferul constant 0, la 0x28 / 0x30 / 0x38** (așa le pune NVK 25.0.7: „push constants” în tabela rădăcină) |
| Firul | indicele firului din registrul special `sr[0x21]` (TID.X); fiecare fir: 2 citiri globale, `fadd`, o scriere globală, `exit` |

**Ce NU e dovedit:** că acești 240 de octeți rulează corect pe RTX 3070 Ti. Asta vine abia după I8–I10 (canal, QMD, lansare), pe placă,
cu rezultatul comparat bit cu bit cu `gpu-compute::reference::add`.

## Treapta 1 matmul (27 oct 2026, sesiunea 7, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh matmul_naive` (același Mesa 25.0.7, reconstruit în cloud; `add64` a ieșit **identic**, aceeași sumă SHA-256).

| Ce | Valoare |
|---|---|
| Program | `matmul_naive.comp`: C[M,N] = A[M,K]·B[K,N], un fir pe element, blocuri 16×16, firul x → rândul, y → coloana (treapta „naivă” siboehm) |
| Ieșire | 51 de instrucțiuni, **816 octeți**, 19 registre, 0 memorie locală/partajată |
| SHA-256 `matmul_naive.sm86.sass` | `856a5d0f41f967e1f67a4ef9e3d40892fda00b156837f6e5a9c390730db336d6` |
| Argumente în bufferul constant 0 | a/b/c la 0x28/0x30/0x38 (ca la add64), **m/n/k la 0x40/0x44/0x48**; **baza grupurilor la 0x0/0x4** (NVK o adună la indicele blocului: lansatorul nostru trebuie s-o lase 0) |
| Aritmetica | **`ffma.ftz`**: înmulțire + adunare într-o singură rotunjire, cu numerele subnormale aduse la 0 (FTZ). Și `add64` folosește **`fadd.ftz`** |

**Treapta 2** (`sh tools/nak/build.sh matmul_coalesced`): aceeași, cu firul x → coloana (citiri/scrieri vecine pentru fire vecine). Tot 51 de
instrucțiuni, 816 octeți, 19 registre, `ffma.ftz`; SHA-256 `2895cb3b765a56ba46eb73554322c0bb5af9057e425663b4816caa2b2cf8f929`. În model dă **aceiași biți**
ca treapta 1 (doar ordinea firelor diferă); câștigul de viteză (×6 la siboehm pe A6000) se poate vedea **doar pe placă**.

**Decizia pentru comparație (C-20 E, „tratarea FMA”):** oracolul fiecărui program NAK e funcția pe care o calculează codul compilat:
`reference::add_ftz` pentru `add64`, `reference::matmul_fma_ftz` pentru matmul (`gpu-compute::math::fma_f32` = o singură rotunjire, verificată bit cu bit
față de `std` pe 2 milioane de cazuri). `reference::matmul` (două rotunjiri, fără FTZ) rămâne oracolul backend-ului CPU; diferențele dintre ele sunt
în ultimii biți (test: `sim_matmul.rs`). Dacă vrem vreodată numerele subnormale păstrate, se poate cere în SPIR-V (`DenormPreserve`) — nefăcut.
**NECUNOSCUT:** dacă placa decide „rezultat subnormal” înainte sau după rotunjire (contează doar la o unitate de 2⁻¹²⁶); noi aducem la 0 după.

## Treapta 3 matmul: plăci în memoria partajată (27 oct 2026, sesiunea 8, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh matmul_smem` (Mesa 25.0.7 reconstruit din nou în cloud; `add64` și `matmul_coalesced` au ieșit **identice**, aceleași sume).

| Ce | Valoare |
|---|---|
| Program | `matmul_smem.comp`: fiecare grup de 16×16 fire copiază o placă 16×16 din A și una din B în memoria partajată, așteaptă (`barrier`), apoi fiecare fir face cele 16 produse; firul x → coloana, y → rândul (ca treapta 2) |
| Ieșire | 218 instrucțiuni, **3 488 octeți**, 19 registre; memorie partajată **2 KiB** (`as_` la 0, `bs_` la 0x400), **o barieră** (`bar.sync`, de 2 ori) |
| SHA-256 `matmul_smem.sm86.sass` | `2d0cc12dc2053ede4235e3ff87bb15e1fa34b7c7fef7a1df5a456d026cecda7d` |
| Aritmetica | tot `ffma.ftz` (16 pe placă, bucla internă desfăcută). Bucla se oprește exact la K (fără zerouri adunate) → **același șir de FMA, în aceeași ordine**, ca treptele 1–2: oracolul rămâne `matmul_fma_ftz`, bit cu bit |
| QMD | în plus față de treptele 1–2: `SHARED_MEMORY_SIZE` = 0x800, configurarea L1/partajată (MIN/TARGET = 3, MAX = 17, formula NAK `gv100_sm_config_smem_size`), `BARRIER_COUNT` = 1 → `ComputeQmd::encode_with_shared` |

Test (SIMULARE, `sim_matmul.rs`, 3 teste noi): modelul pe procesor urmează algoritmul cu plăci (umple cele două plăci ca firele, apoi lanțul FMA),
identic bit cu bit cu oracolul pentru 4 forme (inclusiv plăci parțiale); un caz cu −0,0 dovedește că ultima placă nu adună zerouri (o variantă
„cu zerouri” a modelului pică testul); un QMD fără memorie partajată sau fără barieră e refuzat, fără nicio scriere în C.

## Treapta 4 matmul: „1D blocktiling”, 8 rezultate pe fir (27 oct 2026, sesiunea 9, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh matmul_1d` (Mesa 25.0.7 reconstruit în cloud; `add64`, `matmul_naive`, `matmul_smem` au ieșit **identice**, aceleași sume).

| Ce | Valoare |
|---|---|
| Program | `matmul_1d.comp`: un grup de **512 fire** calculează un bloc 64×64 din C; fiecare fir ține **8 rezultate** (o coloană, 8 rânduri) în registre. Pe fiecare placă de K = 8: fiecare fir aduce un element din placa 64×8 a lui A și unul din placa 8×64 a lui B în memoria partajată, barieră, apoi o valoare din B e refolosită pentru cele 8 sume |
| Ieșire | 478 instrucțiuni, **7 648 octeți**, **24 registre**; memorie partajată **4 KiB** (`as_` la 0, `bs_` la 0x800), o barieră |
| SHA-256 `matmul_1d.sm86.sass` | `2166ded6f1c288ea6b0fc648ecccf743c86ec309856468e828bf96bcdde1946c` (identic la două compilări) |
| Aritmetica | `ffma.ftz` (64 pe o placă plină, bucla desfăcută). Bucla se oprește exact la K → același șir de FMA, în aceeași ordine: oracolul rămâne `matmul_fma_ftz`, bit cu bit |
| Lansare | bloc 512×1×1, grilă x = coloane/64, y = rânduri/64; QMD cu 4 KiB partajată + 1 barieră (`encode_with_shared`), 24 registre |

Test (SIMULARE, `sim_matmul.rs`, 2 teste noi): modelul `matmul_1d_on_cpu` urmează algoritmul (cum încarcă cele 512 fire plăcile, apoi cele 8
sume pe fir), identic bit cu bit cu oracolul pentru 4 forme (inclusiv o grilă 2×3 și plăci parțiale); cazul −0,0 prinde o variantă „cu zerouri”,
iar o variantă care citește coloana greșită din B pică (verificat); lansările cu 2 KiB, cu bloc 16×16 sau fără barieră sunt refuzate, 0 scrieri în C.
Câștigul de viteză (siboehm: ×2–3 față de treapta 3) se poate vedea **doar pe placă**.

## Treapta 5 matmul: „2D blocktiling”, 8×8 rezultate pe fir (27 oct 2026, sesiunea 10, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh matmul_2d` (Mesa 25.0.7 reconstruit în cloud; `add64`, `matmul_smem`, `matmul_1d` au ieșit **identice**, aceleași sume).

| Ce | Valoare |
|---|---|
| Program | `matmul_2d.comp`: un grup de **64 de fire** calculează un bloc 64×64 din C; fiecare fir ține **64 de rezultate** (un pătrat 8×8) în registre. Pe fiecare placă de K = 8: fiecare fir aduce 8 elemente din A și 8 din B în memoria partajată, barieră, apoi pentru fiecare p: 8 valori din A și 8 din B în registre → 64 de produse |
| Ieșire | 1 953 instrucțiuni, **31 248 octeți**, **94 registre**, memorie locală **0**; memorie partajată **4 KiB**, o barieră |
| SHA-256 `matmul_2d.sm86.sass` | `468c978d66663c545ed9e04ba526ebb71bb5ddc492eb009cd9525d24f0af9154` |
| Constatare | fără `[[unroll]]` pe buclele de mărime fixă, NAK ține cele 64 de sume în **memoria locală a firului** (SLM 256 B, 1 996 instrucțiuni): lansatorul nostru nu pregătește memorie locală → buclele sunt desfăcute explicit |
| Aritmetica | `ffma.ftz` (64 pe pas p). Bucla se oprește exact la K → același șir de FMA, în aceeași ordine: oracolul rămâne `matmul_fma_ftz`, bit cu bit |
| Lansare | bloc 64×1×1, grilă x = coloane/64, y = rânduri/64; QMD cu 4 KiB partajată + 1 barieră, 94 registre |

Test (SIMULARE, `sim_matmul.rs`, 3 teste noi): modelul `matmul_2d_on_cpu` urmează algoritmul, identic bit cu bit cu oracolul pentru 5 forme
(inclusiv grile 2×3 și 3×2, plăci parțiale); cazul −0,0; lansările cu bloc 512, cu 2 KiB sau fără barieră sunt refuzate, 0 scrieri în C;
o greșeală plantată (două celule schimbate) e prinsă. Câștigul de viteză (siboehm: ~×2 față de treapta 4) se poate vedea **doar pe placă**.

## Treapta 6 matmul: „vectorize”, citiri și scrieri de 128 de biți (30 oct 2026, sesiunea 24, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh matmul_vec4` (Mesa 25.0.7 reconstruit în cloud; `add64` și `matmul_2d` au ieșit **identice**, aceleași sume; a doua compilare a
treptei 6 a dat aceeași sumă).

| Ce | Valoare |
|---|---|
| Program | `matmul_vec4.comp`: împărțirea treptei 5 (64 de fire, bloc 64×64, 8×8 rezultate pe fir, plăci K = 8), dar fiecare fir aduce **2 vec4 din A și 2 din B** pe placă; placa lui A stă **transpusă** în memoria partajată, ca cele 8 valori de care are nevoie un fir pentru un p să fie una lângă alta (2 citiri vec4 în loc de 8); C se scrie pe câte 4 |
| Ieșire | **821 instrucțiuni** (treapta 5: 1 953), **13 136 octeți**, **90 registre**, memorie locală **0**; memorie partajată **4 KiB** (`as_` transpus la 0, `bs_` la 0x800), o barieră |
| Accese de 128 de biți | 4 `ld.global` b128, 4 `ld.shared` b128, 16 `st.global` b128, 2 `st.shared` b128 (A transpus se scrie cu 8 scrieri de 32 de biți) |
| SHA-256 `matmul_vec4.sm86.sass` | `18efe1da2f168eb01dc85ed89f90f7cae8b2e2f4a3f95a27eb6246ca99d355f7` |
| Aritmetica | tot 64 de `ffma.ftz` pe pas p; bucla se oprește exact la K → același șir de FMA, în aceeași ordine: oracolul rămâne `matmul_fma_ftz`, bit cu bit (și identic cu treapta 5) |
| Condiții (programul **nu** le verifică) | **K și N multipli de 4**, adresele a/b/c **aliniate la 16 octeți** (M e liber). Pe placă, un acces de 128 de biți nealiniat dă eroare. Lansatorul trebuie să le verifice; modelul le refuză fără nicio scriere |
| Lansare | bloc 64×1×1, grilă x = coloane/64, y = rânduri/64; QMD cu 4 KiB partajată + 1 barieră, 90 registre |

Test (SIMULARE, `sim_matmul.rs`, 4 teste noi): modelul `matmul_vec4_on_cpu` urmează algoritmul (cum încarcă firele vectorii, transpunerea lui A, cele 64
de sume, scrierile pe câte 4), identic bit cu bit cu oracolul pentru 6 forme (inclusiv grile 2×3 și 3×2, plăci parțiale, M = 1) și cu treapta 5; cazul
−0,0; K = 10, N = 66 și C nealiniat refuzate, 0 scrieri; blocul 512, 2 KiB sau fără barieră refuzate. **8 greșeli plantate, toate prinse** (6 în model:
A netranspus, vec4 din B inversat, zerouri adunate, verificarea K/N slăbită, verificarea alinierii slăbită, jumătate din C nescrisă; 2 în test).
Câștigul de viteză (siboehm: ~×1,1 față de treapta 5 pe A6000) se poate vedea **doar pe placă**.

## Ce trebuie să respecte lansatorul nostru (I10), din dump — DEDUS, de confirmat pe placă
1. Bufferul constant 0 legat în QMD, cu adresele a, b, c la 0x28, 0x30, 0x38 (restul tabelei rădăcină NVK nu e citit de program).
2. Numărul de registre în QMD ≥ 19 (rotunjirea cerută de hardware: necunoscută încă).
3. Un singur bloc de 64×1×1 fire, fără memorie partajată sau locală.
4. Codul la o adresă GPU aliniată (alinierea cerută de SM 8.6: necunoscută încă; NVK aliniază programele — de citit în `nvk_shader.c`).

## Licențe
Mesa (NAK, NVK, drm-shim): MIT, folosit ca unealtă — nimic copiat în `crates/`. Binarul din `out/` e ieșirea compilatorului pentru
programele noastre (`add64.comp`, `matmul_*.comp`, cod propriu, `LICENSE`); nu conține cod NVIDIA. `ptxas` nu se folosește (C-18).
