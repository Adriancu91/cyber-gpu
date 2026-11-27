# C-157 punctul 2 — conflictele la **citiri** (`coopMatLoad`) în bucla lui `mm_tc_v5` / `mm_tc_v6`, pe cele trei steaguri ale pasului #513, și planul rândurilor cu umplutură (`mm_tc_v7`)

Sesiunea Cyber-GPU 265, 27 nov 2026, ~06:10–07:40 (ora României), în cloud, **fără placă**. Totul de mai jos e **CALCUL** pe procesor
(testul T279, `crates/nvidia-ga104/tests/calc_c157_mm_tc_reads.rs`) și e scris **înainte** de orice cod nou de program GPU.
Nimic din ce e aici n-a rulat pe placă.

Steagurile pasului #513 (corectura din sesiunea 264): **înainte = `TB`**, **dX = fără steag**, **dW = `TA`**, câte 49 de înmulțiri
fiecare; proba qkv-513 e cazul `TB`.

## 1. De unde vin adresele

Din listarea NAK a lui `mm_tc_v6` (`tools/nak/out2/mm_tc_v6.sm86.txt`, blocurile buclei L361…L369; `r26` = numărul firului în warp,
`s2r sr[0x0]`). T279 verifică întâi că listarea are exact aceste citiri (32 `lds` de 32 de biți și 8 `ldsm.x4` cu deplasările lor),
apoi numără trecerile prin bănci după regulile publice CUDA pentru 8.6 (32 de bănci de 4 octeți; un `ldsm.x4` = 4 matrice 8 × 8,
fiecare servită separat, cu 8 rânduri de 16 octeți date de firele 8m…8m+7).

| Operandul și așezarea | Cum citește NAK | Adresa firului `l` (octeți) |
|---|---|---|
| op(A) pe rânduri (fără `TA`), rând 64 B | `ldsm.x4` (2 pe pas de K = 16) | `64·(l & 15) + (l & 16)` |
| op(A) pe coloane (`TA`), rând 128 B | 8 × `lds.b32` + `movm` | `128·(l >> 2) + 4·(l & 3)` |
| op(B) pe rânduri (fără `TB`), rând 128 B | 8 × `lds.b32` + `movm` | `128·(l >> 2) + 4·(l & 3)` |
| op(B) pe coloane (`TB`), rând 64 B | `ldsm.x4` (2 pe pas) | `64·((l & 7) │ ((l & 16) >> 1)) + 2·(l & 8)` |

`mm_tc_v5` are aceleași formule (același drum pe orice steag) = v6 fără steaguri.

## 2. Ce iese (CALCUL, exact)

- un `lds.b32` cu rânduri la 128 B: cele 8 rânduri cad pe **aceleași 4 bănci** → **8 la 1** (8 treceri în loc de 1);
- un `ldsm.x4` cu rânduri la 64 B: rândurile 0, 2, 4, 6 se suprapun → **4 la 1** pe matrice (16 treceri în loc de 4).

Treceri pe bloc (4 warp-uri) și felie de K = 32:

| Cazul (în pasul #513) | citiri v5 | scrieri v5 (T275) | **total v5** | citiri v6 | scrieri v6 | **total v6** |
|---|---|---|---|---|---|---|
| fără steag (dX) | 768 | 256 | **1 024** | 768 | 64 | **832** |
| `TB` (înainte, proba qkv-513) | 768 | 640 | **1 408** | 512 | 64 | **576** |
| `TA` (dW) | 768 | 1 152 | **1 920** | 1 024 | 64 | **1 088** |

Pentru comparație: o felie de bloc durează ~**2 108** tacți ai SM-ului la viteza os-476 (T275), iar circuitele speciale au nevoie de
cel puțin **1 024** (64 × `hmma.m16n8k16` pe felie, 128 de înmulțiri-adunări pe tact pe SM). La proba qkv-513, v5 ținea calea memoriei
comune ocupată **1 408 din 2 108** tacți (67 %), peste podeaua circuitelor.

Concluziile (DEDUS):
1. După v6, la dW **citirile** (1 024) sunt de 16 ori mai mari decât scrierile (64): acolo e următoarea tăietură.
2. La dX (fără steag) v6 a câștigat doar la scrieri (1 024 → 832); citirile au rămas 768.
3. Predicțiile V9 / V10 scrise în sesiunea 264 (dX ×1,00–1,10, dW ×1,15–1,70 față de v5) **nu se schimbă** (sunt scrise
   înainte); socoteala de azi e în acord cu ordinea lor (dW câștigă cel mai mult, dX cel mai puțin).

## 3. Planul: `mm_tc_v7` = `mm_tc_v6` cu **16 octeți de umplutură pe rând**

`coopMatLoad` primește un pas liniar, deci „amestecarea” adreselor (XOR, ca în CUTLASS) nu se poate scrie prin el; rămâne
umplutura. Pașii rândurilor devin:

| Așezarea | v6 | **v7** | pasul `coopMatLoad` (f16vec4) |
|---|---|---|---|
| op(A) pe rânduri | 64 B | **80 B** | 8 → **10** |
| op(A) pe coloane | 128 B | **144 B** | 16 → **18** |
| op(B) pe rânduri | 128 B | **144 B** | 16 → **18** |
| op(B) pe coloane | 64 B | **80 B** | 8 → **10** |

Ce dă (CALCUL, T279):
- toate citirile fără conflict: `lds` 1 trecere, `ldsm.x4` 4 → **128** treceri pe bloc și felie, pe orice steag;
- scrierile de 64 de biți: așezările de 128 B rămân fără conflict; cele de 64 B (A pe rânduri, B pe coloane) devin **2 la 1** (o
  jumătate de warp scrie două rânduri de 64 B, iar la 80 B al doilea se suprapune pe băncile primului). Doar un pas ≡ 64 (mod 128)
  le-ar ține libere, și tocmai acela face `ldsm` 4 la 1: schimbul e +32 treceri la scriere pentru −192 la citire, pe operand;
- total pe bloc și felie: fără steag **224**, `TB` **256**, `TA` **192** (față de 832 / 576 / 1 088 în v6);
- o umplutură mai mică nu ajunge: 4 B lasă `lds` 2 la 1, 8 B strică alinierea la 16 B a rândurilor `ldsm`;
- memoria comună: 5 120 + 5 120 + 4 096 = **14 336 B** (v6: 12 288); tot **4 blocuri pe SM** la pasul de 64 KiB (și cu 1 KiB rezervat);
  toți pașii sunt multipli de 16 B (regula os-467); fiecare element al feliei e scris o dată, coloanele de umplutură niciodată.

Restul **neschimbat** față de v6: aceleași citiri din memoria plăcii, aceeași ramură pe `flags`, aceleași conversii și scrieri de
64 de biți, același lanț `coopMatMulAdd`, același final, aceeași lansare și aceeași regulă `fits` → **aceiași biți ca `mm_tc`**.

Tăiat acum (de reluat după placă): schimbarea împărțirii vectorilor pe fire ca jumătățile de warp să scrie rândurile r și r + 4
(ar scoate și conflictul 2 la 1 la scriere, dar mută citirile din memoria plăcii); memoria comună dublă.

## 4. Predicțiile (scrise înainte de cod)

Două modele de margine, potrivite pe os-476 (v5, `TB`, 2 108 tacți, calea memoriei comune 1 408):
„suprapus” T = max(2 108, cale) → nicio tăietură nu câștigă; „în serie” T = max(1 024, 700 + cale) → v7 ar sta pe podeaua
circuitelor. Plafonul v7 / v6 în modelul „în serie”: fără steag ×1,50, `TB` ×1,25, `TA` ×1,75 (față de v5: ×1,68 / ×2,06 / ×2,56).

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| W1 | listarea NAK a lui v7 | aceleași instrucțiuni ca v6 (32 `lds` + 8 `ldsm`, aceleași `movm`), doar alte constante în adrese (pas 80 / 144) | 60 % |
| W2 | registre | ≤ 128, 0 memorie locală, 14 336 B | 65 % |
| W3 | SIMULARE | aceiași biți ca `mm_tc` pe toate produsele aliniate × toate opțiunile; nealiniatele refuzate | 90 % |
| W4 | pe placă: biții | aceiași ca `mm_tc` | 85 % |
| W5 | proba, setul `qkv-513` (`TB`), v7 / v6 | **×1,00–1,20** | 45 % |
| W6 | setul `qkv-513-dx` (fără steag), v7 / v6 | **×1,05–1,40** | 40 % |
| W7 | setul `qkv-513-dw` (`TA`), v7 / v6 | **×1,10–1,60** | 40 % |
| W8 | ordinea câștigurilor | dW > dX > înainte | 55 % |

Prag scris: dacă pe placă **v6 față de v5** la dW (V10) iese sub ×1,15, citirile conflictuale țin pe drumul critic cât au ținut
scrierile → v7 devine prioritar. Dacă v6 câștigă la dW, dar W7 iese sub ×1,10, calea memoriei comune nu mai e frâna → pasul următor
nu mai e la memoria comună (ci la lansare / la citirile din memoria plăcii).

## 5. Ce urmează (pașii, câte unul, fiecare cu test)

1. `tools/nak/mm_tc_v7.comp` (din `mm_tc_v6.comp`: pașii tablourilor și ai celor patru `coopMatLoad`, indicii scrierilor),
   compilat cu Mesa NAK 26.3.1 → W1–W2.
2. Rust (doar adăugări): `nvidia_ga104::mm_tc_v7`, modelul în `gpu_sim::cmdproc`, testul în SIMULARE (W3) cu o greșeală plantată
   (un rând citit fără umplutură).
3. Nota pentru OS: un rând în proba `mm_tc` după v6, pe cele trei seturi (W4–W7); în pas doar după ce proba v6 / v7 a rulat.

**Nou pentru placă:** nimic (doar socoteală). Când va fi: doar programul `mm_tc_v7`, aceeași lansare; nicio scriere în registre,
niciun firmware.

Surse (publice, citite ca specificație): ghidul CUDA („shared memory”, bănci, capacitatea 8.6), documentația PTX pentru `ldmatrix`
(o matrice 8 × 8 = 8 rânduri de 16 octeți date de 8 fire), specificația `GL_KHR_cooperative_matrix` (pasul liniar al lui
`coopMatLoad`). Codul și socoteala sunt ale noastre.
