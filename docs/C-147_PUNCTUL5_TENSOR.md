# C-147 punctul 5 — circuitele speciale („tensor cores”), DOAR pregătire (T0–T1)

Sesiunea 126, 20 noiembrie 2026, ~19:25–20:00 (ora României), în cloud, **fără placă**. Nimic de aici nu a rulat pe RTX 3070 Ti.
Pe placă doar după **„da”-ul separat al lui Adrian** (cum scrie în C-147).

## Ce s-a cerut
1. o a doua unealtă NAK dintr-un Mesa curent, cea veche (Mesa 25.0.7) neatinsă, ca sumele programelor de azi să nu se schimbe;
2. verificat dacă NAK emite instrucțiunile circuitelor speciale pentru SM 8.6 și cu ce tipuri de numere;
3. un prim program mic (înmulțire de matrice 16×16, numere de 16 biți) compilat și modelat în `gpu-sim`, cu oracol pe procesor și toleranță scrisă.

## 1. A doua unealtă (FĂCUT)
| Ce | Valoare |
|---|---|
| Mesa | **26.3.1** (cea mai nouă versiune stabilă de pe archive.mesa3d.org pe 20 nov 2026), licență MIT |
| SHA-256 arhivă | `4e4e438eac7b840a9383988cbb7c08dad8041da544b9319fc26d319fa7348961` (calculată de noi la descărcare; de comparat cu anunțul versiunii) |
| Unde | `~/.cache/cyber-gpu-nak2` (separat de `~/.cache/cyber-gpu-nak` al uneltei vechi) |
| Scriptul | `tools/nak/build2.sh` (nou); programul gazdă `tools/nak/nak_dump_cm.c` (nou, cere și `VK_KHR_cooperative_matrix`, `shaderFloat16`, `shaderInt8`) |
| Ieșirea | `tools/nak/out2/` (nou). `build.sh`, `nak_dump.c` și `tools/nak/out/` **neatinse** (nicio sumă schimbată) |
| Cerințe noi | Meson ≥ 1.7 (folosim 1.8.3 într-un mediu Python separat), bindgen 0.71.1, cbindgen 0.29.0, Rust ≥ 1.85 |
| Timp | ~20 min pe 2 nuclee (cloud) |

## 2. Ce oferă NAK pentru GA104 / SM 8.6 (FĂCUT, citit din unealtă)
Formele pe care NVK 26.3.1 le anunță pentru cipul `0x174` (fișierul `tools/nak/out2/coopmat_shapes.sm86.txt`, scris de unealtă):

| Intrări A, B | Acumulare C, D | Forme (M×N×K) |
|---|---|---|
| f16 | f16 sau **f32** | 16×16×16, 16×8×16, 16×8×8 |
| s8 (cu sau fără saturare) | s32 | 16×16×32, 16×8×32, 16×8×16, 8×8×16 |
| u8 | u32 | aceleași 4 forme |

**bf16 și tf32 nu sunt oferite** de NVK pe Ampere (deși placa le are). Greutățile Kronos sunt Q8_0 (numere de 8 biți cu o scară pe
32) — forma **s8 × s8 → s32** e cea care ar putea, într-o zi, înmulți greutățile direct; pentru asta x ar trebui și el adus la 8 biți
(alt calcul decât azi, deci altă probă de exactitate). Notat, nefăcut.

Ce emite NAK, citit în listări:
| Program | Instrucțiunile circuitelor speciale | Instrucțiuni | Registre | Memorie partajată | SHA-256 |
|---|---|---|---|---|---|
| `mma16_f16` (16×16×16, f16 → f32) | 2 × `hmma.m16n8k16.f32` (+ 4 × `movm.16.m8n8.trans` pentru B) | 78 (1 248 octeți) | 32 | 0 | `f8c0db68…a1ef9` |
| `mma16_s8` (16×16×32, s8 → s32) | 2 × `imma.m16n8k32.i8.i8` | 148 | 32 | 0 | `7f6bc59a…c84af` |

Ambele ies **identice la două compilări**. `mma16_s8` e doar compilat (sonda pentru 8 biți), nemodelat.

**Diferență importantă față de unealta veche:** NVK 26.3.1 pune argumentele programului în bufferul constant 0 la **0x700**
(`ldc c[0x0][+0x700]`), nu la 0x28 ca NVK 25.0.7. Orice program din unealta nouă are nevoie de bufferul constant cu această așezare
(0x800 octeți); `nvidia_ga104::mma16::cbuf` o face, iar testul T142c arată că așezarea veche e prinsă (programul ar citi zerouri).

## 3. Primul program, oracolul și toleranța (FĂCUT în SIMULARE)
- **Programul:** `tools/nak/mma16_f16.comp` — un singur warp (32 de fire) face D = A·B + C, A și B 16×16 pe 16 biți, C și D pe 32 de
  biți, toate pe rânduri.
- **În driver:** `nvidia_ga104::mma16` (`MMA16_F16`, suma SHA-256 verificată în test, `cbuf` cu adresele la 0x700, `qmd`: un bloc de
  32, 32 de registre, fără memorie partajată).
- **Oracolul** (`gpu_compute::mma`): conversiile pe 16 biți (exacte, toate cele 65 536 de valori verificate dus-întors);
  `mma16_exact` = referința: fiecare produs de două numere pe 16 biți e exact, cei 17 termeni ai unui element se adună în f64 și se
  rotunjesc o dată.
- **Toleranța, scrisă înainte de orice rulare:** fiecare element al plăcii trebuie să fie la cel mult
  `17 · 2⁻²³ · (|c| + Σ|a·b|)` de referință. **Nu cerem identic bit cu bit**: NVIDIA nu documentează ordinea și rotunjirea adunărilor
  din circuitele speciale.
- **Modelul din `gpu-sim`** (`mma16_f16_on_cpu`) folosește o **IPOTEZĂ** (`mma16_block8_trunc`): produsele exacte, adunate câte 8
  odată la acumulator, rezultatul tăiat spre zero. Ideea vine din măsurătorile publicate pe V100/T4/A100 (vezi Surse); mărimea
  blocului pentru GA104 e **necunoscută** — am ales 8. Modelul nu e dovadă despre placă; servește ca testele să nu treacă doar pentru
  că modelul e „prea exact”.
- **Teste (T142, SIMULARE):** trimis ca de driver (pushbuffer, QMD, semafor), 8 seturi de numere, toate în toleranță; greșeli
  plantate (1 %, B transpus, C uitat, NaN, lungimi greșite) prinse; program schimbat, așezarea veche 0x28, 64 de fire, 2 blocuri —
  refuzate, D neatins. 604 → **614 teste** trecute.

## Ce NU e dovedit / ce urmează
| Ce | Stare |
|---|---|
| `hmma.m16n8k16.f32` rulează corect pe RTX 3070 Ti | NOT TESTED — doar după „da”-ul lui Adrian; proba: T142b pe placă (aceeași toleranță) |
| Cât adună circuitele odată și cum rotunjesc pe GA104 | NECUNOSCUT — se vede pe placă comparând cu referința (un test cu numere alese să arate tăierea) |
| Un câștig de viteză pentru Kronos | NEMĂSURAT. Azi, la un simbol, înmulțirile sunt matrice × **vector** (o singură coloană): circuitele speciale lucrează pe blocuri de 8–16 coloane, deci ajută la **loturi** (mai multe simboluri sau mai multe cereri odată, ori la citirea promptului), nu la generarea simbol cu simbol. Pentru #513 de azi, frâna rămâne driverul (punctul 7). |
| Forma s8 × s8 → s32 pentru greutățile Q8_0 | doar compilată (`mma16_s8`); ar cere x pe 8 biți — o decizie de exactitate separată |
| Programele de azi (`q8_matvec` etc.) | neschimbate: rămân pe unealta veche, sumele neatinse |

## Surse
- Mesa 26.3.1 (MIT): `src/nouveau/vulkan/nvk_physical_device.c` (formele anunțate, `nvk_GetPhysicalDeviceCooperativeMatrixPropertiesKHR`),
  `src/nouveau/compiler/nak/` (`hmma`/`imma` în `ir.rs`, `sm70_encode.rs`) — citite pentru înțelegere, rulate ca unealtă pe gazdă; nimic copiat.
- Khronos, `VK_KHR_cooperative_matrix` și `GL_KHR_cooperative_matrix` (specificațiile extensiilor).
- M. Fasi, N. J. Higham, M. Mikaitis, S. Pranesh, „Numerical behavior of NVIDIA tensor cores”, PeerJ Computer Science 7:e330 (2021) —
  ideea ipotezei (produse exacte, adunare pe blocuri, tăiere); valorile pentru GA104 nu sunt în articol.
