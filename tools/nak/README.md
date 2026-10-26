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

## Ce trebuie să respecte lansatorul nostru (I10), din dump — DEDUS, de confirmat pe placă
1. Bufferul constant 0 legat în QMD, cu adresele a, b, c la 0x28, 0x30, 0x38 (restul tabelei rădăcină NVK nu e citit de program).
2. Numărul de registre în QMD ≥ 19 (rotunjirea cerută de hardware: necunoscută încă).
3. Un singur bloc de 64×1×1 fire, fără memorie partajată sau locală.
4. Codul la o adresă GPU aliniată (alinierea cerută de SM 8.6: necunoscută încă; NVK aliniază programele — de citit în `nvk_shader.c`).

## Licențe
Mesa (NAK, NVK, drm-shim): MIT, folosit ca unealtă — nimic copiat în `crates/`. Binarul din `out/` e ieșirea compilatorului pentru
programul nostru (`add64.comp`, MIT); nu conține cod NVIDIA. `ptxas` nu se folosește (C-18).
