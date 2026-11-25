# B1 — `mm_s8`: înmulțirea pe 8 biți pe plăci 128×128 (SIMULARE)

Pista B1 (aprobată de Adrian, 24 nov 21:39), pasul 3: după proba `mma16_s8` (sesiunea 223, `docs/B1_IMMA.md`),
prima înmulțire de mărime oarecare pe tensor cores int8. **Doar SIMULARE**; pe placă abia după ce proba `mma16_s8`
a trecut acolo (și după `mm_tc` / cursa cu 5 variante — ordinea o alege centrul).

## Ce face programul (`tools/nak/mm_s8.comp`)
`C[M,N] = op(A)[M,K] · op(B)[K,N] (+ bias[N])`, cu A și B **întregi cu semn pe 8 biți**, C și bias **întregi pe 32 de biți**.
- Aceleași constante ca `mm_tc` / `mm_tc128` (`mm_tc::Args`, la 0x700: a, b, c, bias, m, n, k, flags) și aceleași opțiuni:
  `TA` (A ținută [K,M]), `TB` (B ținută [N,K] — așezarea greutăților Q8_0: un rând pe ieșire), `BIAS`.
- Aceeași formă de lansare ca `mm_tc128` (câștigătoarea cursei os-416): blocuri de 256 de fire = 8 grupuri de 32,
  un bloc = o bucată 128×128 din C, fiecare grup 32×64 = 2 × 4 acumulatori 16×16 de tip i32.
- K în felii de 32 (exact forma `m16n8k32` a instrucțiunii `imma` și mărimea blocului Q8_0); în afara matricelor: zerouri.
- Memoria partajată: felia lui op(A) 128×32 + a lui op(B) 32×128 pe câte 1 octet (8 KiB) + câte o zonă 16×16 i32 pe grup
  pentru capăt (8 KiB) = **16 KiB**, o barieră.

## De ce verdictul e „aceiași biți” (nu o toleranță)
Ca la `mma16_s8`: produsele a·b sunt exacte (≤ 2¹⁴ în valoare absolută), iar adunarea întregilor modulo 2³² nu depinde
de ordine. Deci **orice** împărțire pe plăci, orice ordine a tensor core-ului și orice ordine a felilor dă aceiași biți
ca oracolul, cât timp rezultatul încape în i32. Singura ipoteză e depășirea lui i32 (instrucțiunea n-are `.sat`):
modelul adună modulo 2³² (IPOTEZĂ), oracolul refuză. Cu |a|, |b| ≤ 128, suma încape sigur cât timp K ≤ 2¹⁷ = 131 072
și bias-ul lasă loc (K · 16 384 + |bias| < 2³¹) — toate formele #513 au K ≤ 3 072.

## Predicții (scrise înainte de compilare și de cod)
- P1: NAK 26.3.1 compilează programul fără depășiri de registre (fără `ldl`/`stl`), **≤ 128 registre** (ca `mm_tc128`).
- P2: în listare apar **`imma`** de tip `m16n8k32.i8.i8` și **nicio** `hmma`; numărul lor = 2 × 4 acumulatori × 2 jumătăți
  × 1 pas de K (32) pe felie = **16 `imma` în bucla principală** (dacă NAK nu desface bucla; desfăcută: un multiplu de 16).
- P3: memoria partajată 16 KiB, o barieră.
- P4: în SIMULARE, modelul (`mm_s8_on_cpu`) = oracolul `mm_s8_ref` **bit cu bit** pe margini (toate opțiunile) și pe cele
  15 forme ale pasului #513 (rânduri și coloane tăiate, K întreg).
- Viteza (DEDUS, nemăsurat): whitepaper GA102 dă int8 ≈ ×4 față de f16 cu acumulare f32 (calea `mm_tc`). Cu aceeași
  eficiență ca `mm_tc` (încă nemăsurată pe placă!) ar fi ×2–4 față de `mm_tc128`; dar încărcarea octet cu octet în
  memoria partajată poate fi frâna — o vedem abia pe placă.

## Verificări T231 (scrise înainte de cod; `crates/nvidia-ga104/tests/sim_b1_mm_s8.rs`, SIMULARE)
- **T231a** binarul = fișierul NAK (SHA-256 scris în `nvidia_ga104::mm_s8`), instrucțiuni întregi de 16 octeți;
  listarea are `imma`, n-are `hmma`, n-are `ldl`/`stl`, are bariera și constantele la `c[0x0][0x700]`; registrele ≤ 128;
  memoria partajată din cod = cea a modelului = 16 KiB.
- **T231b** margini, toate cele 8 combinații de opțiuni: forme 128×32×128, 130×33×129, 5×1×3, 200×96×70, 64×200×257,
  257×48×16 — C = oracolul bit cu bit, nimic scris după C (paznic), niciun element nescris.
- **T231c** valorile extreme: toate −128 × −128 și −128 × 127 pe K = 3 072 (cel mai mare K din #513) + bias ±2²⁰:
  încape în i32, modelul = oracolul.
- **T231d** formele #513 (`mm_train_variants::forms`): K întreg, M și N tăiate la ≤ 160 (timpul procesorului) — bit cu bit.
- **T231e** refuzuri (C neatins): bloc de 128 în loc de 256, grilă care nu acoperă C, biți de opțiuni necunoscuți,
  BIAS fără adresă (cbuf refuzat), depășire (oracolul `None`; modelul = modulo 2³², IPOTEZĂ).
- **T231f** oracolul pe bucăți = `mma16_s8_ref` pe forma 16×32×16 (aceeași aritmetică ca proba).

## Surse
- Mesa NAK (MIT) — doar unealtă pe gazdă (`tools/nak/build2.sh`), nimic copiat.
- GLSL `GL_KHR_cooperative_matrix` + PTX ISA (`mma.m16n8k32` cu `.s8`/`.s32`), documentație publică.
- Ideile de așezare: `tools/nak/mm_tc128.comp` (al nostru), scrierile publice CUTLASS / „How to optimize a CUDA matmul”.
- NVIDIA Ampere GA102 Whitepaper, Anexa A (raportul int8 / f16).

## Rezultat (sesiunea 225, SIMULARE)
- Unealta NAK a doua refăcută în această sesiune reproduce exact binarul vechi `mma16_s8` (SHA-256 `7f6bc59a…c84af`).
- `mm_s8` compilat: SHA-256 `76ce12d5…a92c6`, 57 568 octeți, **104 registre** (P1 ✓, fără `ldl`/`stl`; 24 mutări
  între registre), **16 `imma.m16n8k32.i8.i8`**, 0 `hmma` (P2 ✓, exact cât am prezis), **16 KiB** partajați (P3 ✓);
  NAK scrie „Max warps/SM: 16” = 2 blocuri pe unitate.
- T231 **7/7 PASS** (P4 ✓): T231a–f ca mai sus, plus **T231g** (adăugat după compilare): codul GLSL transcris literal pe
  procesor dă oracolul bit cu bit pe 4 forme × 8 opțiuni; o greșeală de index plantată (rândul lui B la `TB`) e prinsă.
- **Nimic din acestea nu a rulat pe placă.** Viteza (×4 față de `mm_tc`) e DEDUSĂ, nemăsurată.
- Nota pentru OS: `integration/cyber-universe/B1_mm-s8.md` — pe placă abia după proba `mma16_s8`.
- Următorul pas B1: scara pe bloc Q8_0 (un f32 pe 32 de valori, înmulțit la fiecare felie) — schimbă aritmetica din
  întregi exacți în f32, deci verdictul trece la toleranță; întâi întrebarea pentru Kronos.
