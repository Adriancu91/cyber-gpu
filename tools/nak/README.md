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

## C-83 (6 nov 2026, sesiunea 57, cloud) — COMPILAT, NU RULAT
Mesa 25.0.7 reconstruit în cloud; `add64` a ieșit **identic** (aceeași sumă SHA-256).

| Program | Comandă | Ieșire | SHA-256 `.sm86.sass` |
|---|---|---|---|
| `copy_b128.comp` (I11): `dst[i] = src[i]`, `uvec4`, 256 fire pe bloc; argumente src/dst/n la cbuf0 0x28/0x30/0x38 | `sh tools/nak/build.sh kernel copy_b128 24` | 18 instrucțiuni, **288 octeți**, 19 registre; `ld.global...constant.b128`, `st.global.strong.sys.b128` | `ab8449d03cc3f521500f1239d1eabd77158c328e61708817bc7fd2a6826d7bdc` |
| `q8_matvec.comp` (I12): y = W·x, W în Q8_0; W/x/y la 0x28/0x30/0x38, rânduri 0x40, blocuri pe rând 0x44; 32 fire pe rând | `sh tools/nak/build.sh kernel q8_matvec 32` | 503 instrucțiuni, **8 048 octeți**, 19 registre; 128 B partajată, **fără `BAR.SYNC`** (un warp); 1 `FMUL.FTZ`, 32 `FFMA.FTZ`, 5 `FADD.FTZ` | `d8c118cc0683b98b40b0143ed76aec479274172b58b0b3f7ae608c21900bd028` |

Atenție pentru OS: `q8_matvec` are 8 KiB — bufferul programului trebuie să aibă mărimea programului, nu o pagină de 4 KiB (testul a prins asta).

## C-83 I13 (6 nov 2026, sesiunea 58, cloud) — COMPILAT, NU RULAT
Toate cu 19 registre; cele cu 32 de fire = un warp, NAK n-a pus `BAR.SYNC` (doar `MEMBAR`).

| Program | Comandă | Ieșire | Operații speciale | SHA-256 `.sm86.sass` |
|---|---|---|---|---|
| `vadd.comp`: c = a + b; a/b/c 0x28/0x30/0x38, n 0x40; 256 fire | `build.sh kernel vadd 32` | 22 instr., 352 B | 1 `FADD.FTZ` (bit cu bit) | `8757ddc1348f2ae034342c24a364639db3bdb85402e70b16a84410cb4e0487f5` |
| `layernorm.comp`: x/g/b/y 0x28–0x40, n 0x48, inv_n 0x4c, eps 0x50; 32 fire, 128 B partajată | `build.sh kernel layernorm 48` | 271 instr., 4 336 B | `MUFU.RSQ` (aproximativ); NAK contopește `x − mean` într-un FFMA | `fdbe075c7290feaca6842e39efdd009d598ab8fa8df3593eb15ffa88b4444f44` |
| `attention.comp`: q/kc/vc/o 0x28–0x40, n_pos 0x48, n_emb 0x4c, d 0x50, scale 0x54; un bloc de 32 pe cap, 2 176 B partajată | `build.sh kernel attention 48` | 301 instr., 4 816 B | `MUFU.EX2`, `MUFU.RCP` (aproximative) | `cda698124bd7db7d4287c18ede7c414152b196e67ab05380051d91c1daf51c36` |
| `gelu.comp`: x/y 0x28/0x30, n 0x38; 256 fire; erf A&S 7.1.26 | `build.sh kernel gelu 24` | 41 instr., 656 B | `MUFU.EX2`, `MUFU.RCP` | `9498b7f54c2d7d482c03e118eeb95dd2d4755749fbeb48091633358e2cadc08a` |

Unitățile MUFU sunt aproximative și nedocumentate bit cu bit: stratul se judecă în toleranță față de procesor, nu bit cu bit.

## C-83 I14 (6 nov 2026, sesiunea 58) — COMPILAT, NU RULAT
| Program | Comandă | Ieșire | Operații | SHA-256 `.sm86.sass` |
|---|---|---|---|---|
| `embed.comp`: x = rândul simbolului + rândul poziției (ambele Q8_0); rând simbol 0x28, rând poziție 0x30, x 0x38, n 0x40; 256 fire | `build.sh kernel embed 32` | 55 instr., 880 B, 19 registre | `FFMA.FTZ(scara_s, q_s, FMUL.FTZ(scara_p, q_p))` (model bit cu bit: `gpu_compute::model::embed_gpu`) | `b7c4252a457bf9fe3f3708aa13b8d61717d47f4916663c7cf3bc324e55cd3cb0` |

## C-87 pasul 2: `token.comp` — un singur program pe simbol (7 nov 2026, sesiunea 65, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh kernel token 20` (Mesa 25.0.7 reconstruit în cloud; `add64` a ieșit din nou identic, `a890b75c…`).

| Ce | Valoare |
|---|---|
| Program | rulează toată lista de operații a unui simbol (195 pentru #513) într-o singură lansare: G grupuri de 32 de fire își împart blocurile fiecărei operații; barieră pe toată placa între operații (contor atomic) |
| Ieșire | **43 232 octeți** (2 702 instrucțiuni), **35 de registre**, 0 memorie locală, fără `BAR.SYNC` (un singur warp pe grup) |
| SHA-256 `token.sm86.sass` | `ac1768191f52cb7cbdb6678005b86e68fabf1dd405893fabd2808cf077d8a79c` |
| Memorie | datele de lucru `coherent` → `ld/st.global.strong.sys` (ocolesc L1); greutățile prin `ld.global.constant`; bariera: `atom.add.strong.sys` + `membar.sc.gpu` |
| Argumente | listă 0x28, contorul barierei 0x30, număr de operații 0x38 |

**Ce NU e dovedit:** bariera pe toată placa (cere ca toate cele G grupuri să fie pe placă în același timp: G ≤ numărul de SM-uri, 48) și
coerența memoriei pe RTX 3070 Ti. Modelul din `gpu-sim` execută lista în ordine, deci nu poate arăta o blocare la barieră.

## C-147 punctul 2: `q8_fused` — calculele mărunte lipite (20 nov 2026, sesiunea 120, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh q8_fused` (Mesa 25.0.7 reconstruit în cloud; `q8_matvec` a ieșit **identic**, SHA-256 `d8c118cc…`).

| Ce | Valoare |
|---|---|
| Program | `q8_fused.comp`: `q8_matvec` + termenul liber + (rezidual \| GELU \| cheia și valoarea în cache), după steaguri în bufferul constant (0x68) |
| Ieșire | 578 instrucțiuni, **9 248 octeți**, 19 registre, 128 B memorie partajată, fără `BAR.SYNC` |
| SHA-256 `q8_fused.sm86.sass` | `d86e53e2f117e8769026a907fe53a9c4a33350f27807e5c0674c1dea3344dd08` |
| Argumente | W/x/b/rez/y/k/v la 0x28…0x58, rânduri 0x60, blocuri 0x64, steaguri 0x68, E 0x6c (72 octeți de „push constants”) |
| Aritmetica | produsul: aceleași instrucțiuni ca `q8_matvec`; `fadd.ftz` termen și rezidual (ordinea lui `vadd`); GELU: aceleași 21 de instrucțiuni ca `gelu` → oracolul = compunerea modelelor vechi, bit cu bit |

Detalii: `docs/C-147_ETAPA3A.md`.

## C-147 punctul 3: `q8_matvec32` — toate cele 32 de fire (20 nov 2026, sesiunea 121, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh q8_matvec32` (Mesa 25.0.7 reconstruit în cloud; `q8_matvec` și `q8_fused` au ieșit **identice**).

| Ce | Valoare |
|---|---|
| Program | `q8_matvec32.comp`: unitatea = un sfert de bloc (8 coloane); firul t ia sferturile t, t+32, …; x citit cu 2 × 128 de biți pe sfert |
| Ieșire | 199 instrucțiuni, **3 184 octeți**, 22 registre, 128 B memorie partajată, fără `BAR.SYNC` |
| SHA-256 `q8_matvec32.sm86.sass` | `86ae584bd079454150a585968c2ea63ef2b7dadec31054d21a6f59d6afc7c303` |
| Argumente | aceleași ca `q8_matvec` (32 de octeți de „push constants”); x aliniat la 16 |
| Aritmetica | pe sfert: `FFMA(q0, x0, FMUL(q1, x1))`, apoi 6 × `FFMA`, apoi `FFMA(scale, s, part)`; arborele lui `q8_matvec` → oracolul `q8::row_gpu_order_w32` |

Detalii: `docs/C-147_ETAPA4.md`.

## A doua unealtă: Mesa 26.3.1, circuitele speciale (C-147 punctul 5, 20 nov 2026, sesiunea 126) — COMPILAT, NU RULAT
`sh tools/nak/build2.sh mesa` (o dată, ~20 min, în `~/.cache/cyber-gpu-nak2`), apoi `sh tools/nak/build2.sh mma16_f16` / `mma16_s8`.
Ieșirea merge în `tools/nak/out2/`; **unealta veche (`build.sh`, `nak_dump.c`, `out/`) rămâne neatinsă**, ca sumele programelor de azi
să nu se schimbe. `nak_dump_cm.c` cere în plus `VK_KHR_cooperative_matrix` și scrie formele oferite în `out2/coopmat_shapes.sm86.txt`.

| Program | Ce emite NAK | Instr. | Registre | SHA-256 |
|---|---|---|---|---|
| `mma16_f16` (16×16×16, f16 → f32) | 2 × `hmma.m16n8k16.f32` | 78 | 32 | `f8c0db68b654372d94ed96c767c7775c32909298a1b77076ddda81a66afa1ef9` |
| `mma16_s8` (16×16×32, s8 → s32) | 2 × `imma.m16n8k32.i8.i8` | 148 | 32 | `7f6bc59aad0871f1164902ad0f19aad5f89d388003f6d3afd5a4551e9e0c84af` |

**Atenție:** NVK 26.3.1 pune argumentele la **cbuf 0 + 0x700** (unealta veche: 0x28). Detalii, oracol, toleranță: `docs/C-147_PUNCTUL5_TENSOR.md`.

## LayerNorm nou (21 nov 2026, sesiunea 132, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh layernorm256` (Mesa 25.0.7 reconstruit în cloud; `q8_matvec32` a ieșit **identic**, aceeași sumă).

| Ce | Valoare |
|---|---|
| Program | `layernorm256.comp`: 256 de fire, x/g/b citite o dată (12 citiri fără salt între ele: indicele de după capăt citește elementul n−1, nefolosit), `shfl.bfly` 16…1 în warp, 8 sume prin memoria partajată, `bar.sync` |
| Ieșire | 286 de instrucțiuni, **4 576 octeți**, **25 de registre**, memorie partajată 32 B, o barieră |
| SHA-256 `layernorm256.sm86.sass` | `a407dd68507f51d30a8f8ecd0853f8a7d95d9d4aa23a6cdd5644b6fe5b38334d` |
| Aritmetica (citită din cod) | prima valoare a firului intră fără rotunjire (`sel`), restul `fadd.ftz`; media nu se rotunjește niciodată: abaterea e un singur `ffma.ftz` d = (−sumă)·inv_n + x; d₀² cu `fmul.ftz`, apoi `ffma.ftz`; `mufu.rsq`; y = `ffma.ftz`(`fmul.ftz`(d, inv), g, b). Oracolul: `gpu_compute::layer::layernorm_gpu256` |

## Argmax pe placă (21 nov 2026, sesiunea 134, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh argmax` (Mesa 25.0.7 reconstruit în cloud; `layernorm256` a ieșit **identic**, aceeași sumă).

| Ce | Valoare |
|---|---|
| Program | `argmax.comp`: 256 de fire, runde de 8 citiri pornite una după alta (indicele de după capăt citește elementul n−1, nefolosit), cheie întreagă din biți (−0 → +0), `shfl.bfly` 16…1, 8 alegeri prin memoria partajată, `bar.sync`, firul 0 scrie 16 octeți |
| Ieșire | 493 de instrucțiuni, **7 888 octeți**, **25 de registre**, memorie partajată 96 B, o barieră; argumente: logit-uri 0x28, rezultat 0x30, n 0x38 |
| SHA-256 `argmax.sm86.sass` | `4d67cd3495e67cbb63a598e2ad4298925bed51d9660b65094c05586f0cb80572` |
| De ce chei întregi | prima variantă (comparații de numere) a ieșit cu `fsetp.*.ftz`: numerele subnormale ar fi fost luate drept 0, deci alt simbol decât procesorul în cazuri-limită. Cu chei: 0 `fsetp` în cod, ordinea exactă a procesorului. Oracolul: `gpu_compute::model::argmax_result` |

## C-155 punctul 4: `q8_fused32` (21 nov 2026, sesiunea 135, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh q8_fused32` (Mesa 25.0.7 reconstruit în cloud; `q8_matvec32` a ieșit **identic**, aceeași sumă).

| Ce | Valoare |
|---|---|
| Program | `q8_fused32.comp`: miezul lui `q8_matvec32` (un sfert de bloc pe fir, x cu `ld.global.constant.b128`) + coada lui `q8_fused` (bias, rezidual, GELU, K/V); push constants ale lui `q8_fused` (72 B) |
| Ieșire | **4 384 octeți**, 22 de registre, 128 B partajați, fără `bar.sync` (un singur grup de 32) |
| SHA-256 `q8_fused32.sm86.sass` | `f26faedd659916b8d0508831329319b754213c7637a6ba40d42f8ab02344bcb2` |
| Verificat în listare | bucla = `q8_matvec32` instrucțiune cu instrucțiune (doar `c[0x0][0x40/0x44]` → `0x60/0x64`); coada = `q8_fused` identic |

## C-160 punctul 4: `mm_train` — înmulțirea pentru antrenare (21 nov 2026, sesiunea 142, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh mm_train` (Mesa 25.0.7 reconstruit în cloud; `matmul_smem` a ieșit **identic**, aceeași sumă; `mm_train` identic la două compilări).

| Ce | Valoare |
|---|---|
| Program | `mm_train.comp`: C = op(A)·op(B) (+ deplasare), f32, plăci 16×16 în memoria partajată (aranjamentul treptei 3). Opțiuni: `TA` = 1 (A stocată [K,M]), `TB` = 2 (B stocată [N,K], ca `weight [out, in]`), `BIAS` = 4. Cele trei produse ale unui strat `Linear`: înainte `TB|BIAS`, dX `0`, dW `TA` |
| Ieșire | 258 de instrucțiuni, **4 128 octeți**, 19 registre; memorie partajată 2 KiB (`as_` la 0, `bs_` la 0x400), o barieră (`bar.sync` de 2 ori) |
| SHA-256 `mm_train.sm86.sass` | `69d54ef2334cca67d66fff11cbca838cfd366e33f12a5f9340196c16063fa350` |
| Argumente în bufferul constant 0 | a/b/c/bias la 0x28/0x30/0x38/0x40, **m/n/k/flags la 0x48/0x4c/0x50/0x54** (citite în listare); baza grupurilor la 0x0/0x4 = 0 |
| Aritmetica | 16 × `ffma.ftz` pe placă (bucla desfăcută), apoi un `fadd.ftz` pentru deplasare → oracolul bit cu bit `reference::matmul_t_fma_ftz` |

## C-160 punctul 4, programul 7: `xent` (21 nov 2026, sesiunea 143, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh xent` (Mesa 25.0.7 reconstruit în cloud; `mm_train` a ieșit **identic**, `69d54ef2…3fa350`).

| Ce | Valoare |
|---|---|
| Program | `xent.comp`: pe fiecare rând de V logit-uri (un bloc de 256 de fire), eroarea `ln Σ e^(z−m) + m − z[t]` și gradientul `(softmax − unu-la-țintă)·scală` |
| Ieșire | **3 680 octeți**, 19 registre, 32 de octeți partajați, o barieră |
| SHA-256 `xent.sm86.sass` | `44cc3536ec1dbd0a1cd2440472eaa58b1d8124578dd9defe0edc4e6f420e339d` |
| Argumente în bufferul constant 0 | logits / ținte / gradient / erori la 0x28 / 0x30 / 0x38 / 0x40, V la 0x48, scala la 0x4c; baza grupurilor la 0x0 (lăsată 0) |
| Aritmetica | `fadd.ftz`, `fmul.ftz` cu log₂e, **`mufu.exp2`**, **`mufu.rcp`**, **`mufu.log2`** + `ffma.ftz` cu ln 2, `fmnmx.ftz`, `shfl.bfly` |

Oracolul: `gpu_compute::train::softmax_xent` (f64), într-o margine scrisă — MUFU e aproximativ, deci nu bit cu bit. Testul T153.

## C-160 punctul 4, programele 4 și 2: `ln_back` și `colsum` (21 nov 2026, sesiunea 144, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh ln_back` / `colsum` (Mesa 25.0.7 reconstruit în cloud; `xent` a ieșit **identic**, `44cc3536…f420e339d`).

| Ce | `ln_back` | `colsum` |
|---|---|---|
| Program | LayerNorm înapoi pentru dx, un bloc de 256 de fire pe rând | sume pe coloane (un fir pe coloană): Σ dy (deplasare) sau Σ dy·x̂ și Σ dy (dγ, dβ) |
| Ieșire | **2 736 octeți**, 171 de instrucțiuni, 19 registre, 64 de octeți partajați, o barieră | **1 136 octeți**, 71 de instrucțiuni, 19 registre, fără memorie partajată |
| SHA-256 | `369afd8dd343b9a508d481998b75de899d417d33c991c11351ec21db85cd7063` | `7a0fa2bde1b464cff50af588371bb09adf0f03399e880a07d75021431a47fb10` |
| Bufferul constant 0 | x / dy / γ / μ / r / dx la 0x28…0x50, C la 0x58, 1/C la 0x5c | dy / x / μ / r / s0 / s1 la 0x28…0x50, rânduri 0x58, C 0x5c, modul 0x60 |
| Aritmetica | `fadd.ftz`, `fmul.ftz`, `ffma.ftz`, `shfl.bfly`; NAK a lipit `dy·γ − m1` într-un singur `ffma` | `fadd.ftz`, `fmul.ftz`, `ffma.ftz` |

Oracolul: `gpu_compute::train::layernorm_back_rows` (f64), într-o margine scrisă. Testele T154, T155.

## C-160 punctul 4, programul 3: `ln_fwd` (22 nov 2026, sesiunea 146, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh ln_fwd` (Mesa 25.0.7 reconstruit în cloud; `add64` și `ln_back` au ieșit **identice**).

| Ce | `ln_fwd` |
|---|---|
| Program | LayerNorm înainte pe multe rânduri (un bloc de 256 de fire pe rând, orice C, două treceri prin rând), scrie y, μ și r = 1/√(var + ε) |
| Ieșire | **3 456 octeți**, 216 instrucțiuni, 19 registre, 32 de octeți partajați, o barieră |
| SHA-256 | `895fc7988e387cd8ff279695996f5a78e3f326874fa8013aac5efea31e9f2204` |
| Bufferul constant 0 | x / γ / β / y / μ / r la 0x28…0x50, C la 0x58, 1/C la 0x5c, ε la 0x60 |
| Aritmetica | `fadd.ftz`, `fmul.ftz`, `ffma.ftz`, `shfl.bfly`, **un `mufu.rsq` pe rând**; μ rotunjit o dată (`precise`) |

Oracolul: `gpu_compute::train::layernorm_rows` + `layernorm_stats` (f64), într-o margine scrisă. Testul T157. Nota: `docs/C-160_P4_LN_FWD.md`.

## C-160 punctul 4, programul 5: `gelu_tr` (22 nov 2026, sesiunea 147, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh gelu_tr` (Mesa 25.0.7 reconstruit în cloud; `ln_fwd` a ieșit **identic**).

| Ce | `gelu_tr` |
|---|---|
| Program | GELU al antrenării, element cu element (256 de fire pe bloc): modul 0 = înainte, u·Φ(u); modul 1 = înapoi, g·(Φ(u) + u·φ(u)) |
| Ieșire | **912 octeți**, 57 de instrucțiuni, 19 registre, fără memorie partajată, fără barieră |
| SHA-256 | `52dd5f9a81fd88a864d77ebbeab5a3cc3ab537b89362f1dd36424e21c4ae54f1` |
| Bufferul constant 0 | u / g / y la 0x28 / 0x30 / 0x38, n la 0x40, modul la 0x44 |
| Aritmetica | `fmul.ftz`, `fadd.ftz`, `ffma.ftz`, **un `mufu.rcp` și un `mufu.exp2` pe element** (aceeași exponențială pentru Φ și φ) |

Oracolul: `gpu_compute::train::gelu_rows` / `gelu_back_rows` (f64), într-o margine scrisă. Testul T158. Nota: `docs/C-160_P4_GELU.md`.

## C-160 punctul 4, programul 6 (înapoi): `attn_back` (22 nov 2026, sesiunea 148, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh attn_back` (Mesa 25.0.7 reconstruit în cloud; `gelu_tr` a ieșit **identic**).

| Ce | `attn_back` |
|---|---|
| Program | atenția cauzală înapoi, două lansări: modul 0 = rândurile lui dS (grila [⌈T/32⌉, H, B]), modul 1 = dQ \| dK \| dV (grila [H, T, B]); blocuri de 32 |
| Ieșire | **5 104 octeți**, 319 instrucțiuni, 25 de registre, fără memorie partajată, fără barieră, fără MUFU |
| SHA-256 | `7092074774576fbe9d213b0d3adb4f02d6fa013771cde8315526879fb0079f54` |
| Bufferul constant 0 | qkv / P / dO / dS / dqkv la 0x28 / 0x30 / 0x38 / 0x40 / 0x48, T 0x50, C 0x54, H 0x58, d 0x5c, modul 0x60, scala 0x64; baza grupurilor 0x0 / 0x4 / 0x8 = 0 |
| Aritmetica | 5 lanțuri `ffma.ftz` (dP, dot, dQ, dK, dV), 2 `fadd.ftz`, 2 `fmul.ftz` |

Oracolul: `gpu_compute::train::attention_back_rows` (f64), într-o margine scrisă. Testul T159. Nota: `docs/C-160_P4_ATTN_BACK.md`.

## C-160 punctul 4, programul 6 (înainte): `attn_fwd` (22 nov 2026, sesiunea 149, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh attn_fwd` (Mesa 25.0.7 reconstruit în cloud; `attn_back` a ieșit **identic**).

| Ce | `attn_fwd` |
|---|---|
| Program | atenția cauzală înainte pe toate cele T poziții ale lotului, o lansare, grila [⌈T/32⌉, H, B], blocuri de 32; scrie P [B, H, T, T] (+0 deasupra diagonalei) și att [B·T, C] |
| Ieșire | **3 536 octeți**, 221 de instrucțiuni, 25 de registre, fără memorie partajată, fără barieră |
| SHA-256 | `fe97ab49c5b04303d0e2a2e2a9d7bc9c22d8bfbaaf2e8cbfa3e41104dbaa5596` |
| Bufferul constant 0 | qkv / P / att la 0x28 / 0x30 / 0x38, T 0x40, C 0x44, H 0x48, d 0x4c, scala 0x50; baza grupurilor 0x0 / 0x4 / 0x8 = 0 |
| Aritmetica | 2 lanțuri `ffma.ftz` (Q·K, P·V), `fmul.ftz` cu scala, `fmnmx.ftz` (maximul), `fadd.ftz`(s, −m), `fmul.ftz` cu log₂e (`0x3fb8aa3b`), **`mufu.exp2`**, lanțul `fadd.ftz` al sumei, **`mufu.rcp`** (fără pas Newton), `fmul.ftz`(x, 1/Σ) |

Oracolul: `gpu_compute::train::attention_rows` (f64), într-o margine scrisă (cu loc 2⁻²² pentru fiecare MUFU). Testul T160.
Nota: `docs/C-160_P4_ATTN_FWD.md`.

## C-160 punctul 4, programul 8: `emb_fwd` + `emb_back` (22 nov 2026, sesiunea 150, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh emb_fwd` / `emb_back` (Mesa 25.0.7 reconstruit în cloud; `attn_fwd` a ieșit **identic**).

| Ce | `emb_fwd` | `emb_back` |
|---|---|---|
| Program | x = emb[simbol] + poz[i], rândul b·T + i; blocuri de 64, grila [⌈C/64⌉, T, B] | gradientul tabelelor pe segmente (listă de pe procesor), fără `atomic`; blocuri de 64, grila [⌈C/64⌉, S, 1] |
| Ieșire | **672 octeți**, 42 de instrucțiuni, 19 registre | **1 280 octeți**, 80 de instrucțiuni, 19 registre |
| SHA-256 | `247e0e42871932822aa201cc6ee8e254ae9e657bec4ce6e30bbd3b691cd5bebb` | `fca7b007aebb0b8fbfbdbb209f6d4a7162e579212132b47a85da0524a12af0be` |
| Bufferul constant 0 | simboluri / emb / poz / x la 0x28 / 0x30 / 0x38 / 0x40, C 0x48, T 0x4c; baza 0x0 / 0x4 / 0x8 = 0 | dx / off / dst / rows / g la 0x28 … 0x48, C 0x50, modul 0x54; baza 0x0 / 0x4 = 0 |
| Aritmetica | un `fadd.ftz`, nimic altceva (fără împărțire: prima variantă cu r mod T folosea `mufu.rcp` și a fost scoasă) | lanțul `fadd.ftz` de la 0 + un `fadd.ftz` în modul „adună”; fără MUFU |

Oracolul: `gpu_compute::train::{embed_rows, embed_back_rows}`. Testul T161. Nota: `docs/C-160_P4_EMB.md`.

## C-160 punctul 7: `colsum_part` — prima trecere a sumei pe coloane împărțite pe rânduri (22 nov 2026, sesiunea 159, cloud) — COMPILAT, NU RULAT
`sh tools/nak/build.sh colsum_part` (Mesa 25.0.7 reconstruit în cloud; `colsum` a ieșit **identic**, `7a0fa2bd…`).

| Ce | Valoare |
|---|---|
| Program | `colsum_part.comp`: ca `colsum`, dar blocul (x, g) adună doar rândurile bucății g; scrie sume parțiale `[G, c]`; a doua trecere = `colsum` |
| Ieșire | 121 de instrucțiuni, 20 de registre, fără MUFU / memorie partajată / barieră / `atom` |
| SHA-256 `colsum_part.sm86.sass` | `740f5d439b357d37c06685384cdfd4fb7ba5cbda84c3023f49a77565d59725ac` |
| Argumente în bufferul constant 0 | ca la `colsum` (0x28…0x60) + **chunk la 0x64**; blocul y din `sr[0x26]` + **baza la 0x4** (lăsată 0) |
| Nota | `docs/C-160_P7_COLSUM_IMPARTIT.md` (T170, SIMULARE) |
