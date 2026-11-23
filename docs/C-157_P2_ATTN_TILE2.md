# C-157 punctul 2, partea a doua pentru os-358 — `attn_*_tile2`: aceiași biți, bucle mai scurte

Sesiunea 172, 23 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise și urcate **înainte** de cod.
Totul despre viteză e **DEDUS**; corectitudinea se verifică doar în **SIMULARE**, iar „aceiași biți ca programul vechi” e o predicție
pentru placă (simularea rulează modelul pe procesor, nu codul NAK).

## De ce
os-358 (`docs/C-157_P2_COMPARATII.md`, prima secțiune): corect ca predicția, dar 2,49 / 1,81 / 3,99 ms în loc de ~0,53 / ~0,49 /
~0,42–0,68. Din listările NAK (`tools/buget-antrenare/nak_bucle.py`): bucle de 38–120 de instrucțiuni pentru 16 `ffma`, garda
`j <= i` pe fiecare rând la fiecare pas, conflicte de bancă 16 la 1 pe placa K, plăcile urcate câte o valoare.

## Planul (scris ÎNAINTE de cod)
Programe noi **lângă** cele vechi (cele vechi rămân neschimbate, OS-ul compilează neschimbat; nimic redenumit sau șters):
`attn_fwd_tile2`, apoi `attn_back_tile0_2` și `attn_back_tile_2` (în sesiuni următoare, câte unul). Aceleași argumente, aceeași grilă,
aceleași blocuri de 256 ca programul vechi. **Fiecare sumă în aceeași ordine** → același model pe procesor (`attn_fwd_tile_row_gpu`
etc.), aceiași biți pe placă (prezis). Patru schimbări, doar în felul în care se mișcă datele:
- **F1 — rânduri de 65** în memoria partajată (Q / P și K / V): `tile[r * 65 + e]`. Cele 16 fire `tx` care citesc 16 rânduri la același
  `e` cad în 16 bănci diferite (înainte: aceeași bancă, 16 la 1). Și cele două rânduri `ty` dintr-un warp nu se mai lovesc.
- **F2 — plăcile de sub diagonală fără gardă**: faza C (P·V) are două bucle — `jt < bx` fără `j <= i` (acolo e mereu adevărată) și
  placa diagonalei `jt == bx` cu garda veche. Aceleași `ffma`, în aceeași ordine (pe plăcile de sub diagonală garda nu sărea nimic).
- **F3 — plăcile urcate cu toate citirile emise deodată**: cele 16 citiri globale ale unui fir într-un tablou de registre, apoi cele 16
  scrieri în memoria partajată (înainte: citire → așteptare → scriere, de 16 ori).
- **F4 — bucla interioară desfăcută câte 4** (faza A peste `e`, faza C peste `jj`), cu restul (d nu e multiplu de 4) într-o buclă scurtă;
  fiecare acumulator primește aceleași `ffma` în aceeași ordine.
- Memoria partajată: 2 × 64 × 65 + 64 × 16 + 64 + 64 float = **37 888 de octeți** (+512 față de vechi) → tot 2 blocuri pe SM.
  Registrele pot crește (F3, F4) fără pierdere: cu 2 blocuri de 256 de fire, limita e 128 pe fir.

## Predicții pentru `attn_fwd_tile2` (scrise acum)
1. **Aceleași operații:** în listarea NAK tot 16 × `mufu.ex2` și 1 × `mufu.rcp` ca la vechi; nicio `atom`/`red`; nicio memorie locală
   (`ld.local` / `st.local`); 1 barieră (`bar.sync`).
2. **Registre ≤ 128** (dacă ies peste 96, notez de ce); memoria partajată 37 888 de octeți în QMD.
3. **Buclele (nak_bucle.py), pe drumul plăcilor de sub diagonală:** faza A ≤ 28 de instrucțiuni la 16 `ffma` (vechi 38 la 16) și **0
  zone `bssy`**; faza C fără gardă ≤ 30 la 16 `ffma` (vechi 68 la 16, 4 `bssy`) și 0 `bssy`; buclele de urcare a plăcilor: 16 `ldg`
  înaintea primului `sts` din aceeași iterație (sau bucla dispărută, desfăcută).
4. **SIMULARE:** pe cele 10 forme din T173, `attn_fwd_tile2` (același model pe procesor, QMD-ul cu 37 888 de octeți) = `attn_fwd_tile`
  **bit cu bit** (P și att), gărzile de după P / att neatinse; greșeli plantate prinse: QMD cu memoria partajată veche (37 376 < cât
  cere programul) refuzat de proba noastră, grila cu un bloc lipsă, blocuri ≠ 256, d > 64.
5. **Pe placă (când OS-ul îl pune lângă cel vechi, în aceeași rulare):** P și att **bit cu bit egale** cu `attn_fwd_tile` pe toate cele 6
  seturi; timpul pe setul mare (4 × 512, 12 × 64) **~0,6–1,2 ms** față de 2,49 (DEDUS: conflictele ~0,88 ms dispar, buclele A și C
  scurtate ~×2). Dacă iese tot peste ~1,8 ms, factorul NECUNOSCUT e altundeva decât în bucle (ceasul plăcii, latența memoriei).

## Ce NU face
Nimic pe placă în această sesiune. Nu schimbă `attn_fwd_tile` / `attn_back_tile*` / proba `attn_tile_probe` (proba „vechi lângă nou”
vine ca adăugare, după ce sunt gata toate trei, sau numai cu `fwd2` dacă OS-ul vrea mai devreme).

## Surse
Ideile (rânduri cu un element în plus contra conflictelor de bancă, citiri emise deodată, bucle desfăcute): ghidul public CUDA
(„shared memory bank conflicts”, capacitatea 8.6) și siboehm SGEMM_CUDA (MIT, doar ideea). Codul e al nostru, din `attn_fwd_tile.comp`.

## Rezultatul pentru `attn_fwd_tile2` (sesiunea 172, SIMULARE, T181 `crates/nvidia-ga104/tests/sim_attn_tile2.rs`)
- **Binarul** (Mesa NAK 25.0.7 reconstruit în cloud; `attn_fwd_tile` a ieșit **identic**, `4ef8dca3…`): `attn_fwd_tile2.sm86.sass`
  SHA-256 **`2e8cda57ead70edf8b72b7960f34acbe2f3e32fd4ed617eec53c5922c8a3dc08`**, 3 815 instrucțiuni, **117 registre**, 37 888 de
  octeți partajați.
- **Predicția 1: DA** — aceleași 16 `mufu.exp2` + 1 `mufu.rcp`, nicio atomică, nicio memorie locală, `bar.sync` cu o singură barieră.
- **Predicția 2: DA, cu notă** — 117 ≤ 128, dar peste 96: cele trei tablouri de câte 16 citiri (F3) și buclele desfăcute țin multe
  valori deodată. Ocuparea nu scade: 2 blocuri pe SM tot din cauza memoriei partajate (117 → 120 rotunjit × 32 × 16 warp-uri = 61 440
  ≤ 65 536 de registre pe SM).
- **Predicția 3: DA după o reparație.** Prima compilare: faza A 173 de instrucțiuni la 64 `ffma` (= 43 la 16, **peste** pragul 28),
  faza C 145 (36 la 16, peste 30) — NAK refăcea adresa fiecărei citiri partajate (`(ty + 16a)·65 + e`: înmulțire / deplasare +
  adunare la fiecare). Cu o bază pe operand și iterație și restul constant (`qs[qb + (1040a + u)]`, aritmetica neschimbată), constanta
  intră în adresa citirii: **faza A 109 la 64 = 27,3 la 16** (vechi 38), **faza C sub diagonală 106 = 26,5 la 16** (vechi 68), **0
  `bssy`** în ambele (vechi 4 la C). Întârzierile scrise de NAK pe iterație: A 178 / 4 = 44,5 la 16 `ffma` (vechi 107), C 263 / 4 = 66
  (vechi 217). Urcarea plăcilor Q / K / V: 16 citiri globale, apoi 16 scrieri partajate (vechi: citire–scriere de 16 ori). Placa P din
  faza C (P = x · inv, scris și în memoria partajată) a rămas citire–scriere pe element — candidat pentru pasul următor.
- **Predicția 4: DA** — pe cele 10 forme, P și att = `attn_fwd_tile` bit cu bit (același model pe procesor: arată doar partea
  driverului); QMD cu memoria partajată veche (37 376) refuzat, grila cu un bloc lipsă lasă exact rândurile 128–129 nescrise, bloc 128
  sau grilă peste ⌈T/64⌉ refuzate, d = 65 refuzat.
- **Predicția 5 (pe placă): NEMĂSURATĂ.** Rămâne ~0,6–1,2 ms pe setul mare față de 2,49 (DEDUS).
- `tools/check.sh`: **868 de teste** (+7 ignorate), format, clippy, build no_std — PASS.

**Următorul pas:** `attn_back_tile0_2` și `attn_back_tile_2` cu aceleași patru schimbări (plan scurt + predicții întâi), apoi proba
„vechi lângă nou” (adăugare în `attn_tile_probe`, fără să schimb lista `PROGS` de azi) și nota pentru OS.

## `attn_back_tile0_2` — planul și predicțiile (sesiunea 173, 23 nov 2026, scrise ÎNAINTE de cod)
Program nou **lângă** `attn_back_tile0` (cel vechi neschimbat). Aceleași argumente (cbuf-ul lui `attn_back`, modul 0), aceeași
grilă [⌈T/64⌉, H, B], blocuri de 256, același model pe procesor (`attn_back_row_gpu`); fiecare lanț în aceeași ordine. Cele patru
schimbări, adaptate (aici contează și **ocuparea**: vechiul încape 3 blocuri pe SM = 24 warp-uri; vreau să rămână 3):
- **F1** — placa V și placa P în `sb` cu **rânduri de 65** (în loc de „rotite”, `sk`), ca adresele să fie bază + constantă. Placa dO
  rămâne în `sa` cu rânduri de 64 (citirile ei sunt aceeași adresă pe 16 fire; rămâne conflictul de 2 la 1 dintre cele două `ty`
  ale unui warp, ca la vechi), placa dP din faza „dot” rămâne rotită în `sa` (citită pe rânduri de 64 de fire). Memoria partajată:
  64 × 64 + 64 × 65 float = **33 024 de octeți** (+256). Cu 1 KiB rezervat pe bloc (NECUNOSCUT dacă și la noi): 3 × 34 048 =
  102 144 ≤ 102 400 → tot 3 blocuri. Ambele plăci cu 65 (33 280) ar fi trecut peste → 2 blocuri; de aceea doar una.
- **F2** — lanțul `dot` (firele 0–63): plăcile de sub diagonală (`jt < bx`) fără garda `i < T && j <= i` (acolo `j < i0 ≤ i`; rândurile
  `i ≥ T` au dO = 0 și P = 0 în plăci → `dot` lor nu e folosit niciodată); placa diagonalei cu garda veche. La fel scrierea lui dP în
  dS sub diagonală doar cu `i < T`.
- **F3** — urcarea plăcilor dO, V și P: toate citirile globale ale unui fir emise întâi (în registre), apoi scrierile în `sa` / `sb`.
- **F4** — bucla lui dP peste `e` și bucla `dot` peste `jj` desfăcute câte 4 (restul lui d într-o buclă scurtă); fiecare
  acumulator primește aceleași `ffma` în aceeași ordine.
- Partea a doua (dS = P·(dP − dot)·scale, citiri / scrieri globale) rămâne neschimbată: e limitată de memorie, nu de bucle.

Predicții:
1. **Aceleași operații:** în listare 0 `mufu`, nicio atomică, nicio memorie locală, 1 barieră; nicio `fadd`/`fmul` nouă față de vechi
   (aceeași scădere, cele două înmulțiri, `ffma`).
2. **Registre ≤ 80** (ca să rămână 3 blocuri: 80 × 256 × 3 = 61 440 ≤ 65 536); dacă ies peste, notez și cobor citirile emise deodată.
3. **Buclele (nak_bucle.py):** bucla lui dP ≤ 28 de instrucțiuni la 16 `ffma` (vechi 53) și 0 `bssy`; bucla `dot` de sub diagonală
   fără `bssy` (vechi: garda în buclă); plăcile urcate cu toate citirile înaintea primei scrieri partajate.
4. **SIMULARE:** pe formele testului T175 (`sim_attn_back_tile0.rs`), dS = `attn_back_tile0` **bit cu bit** (același model), gărzile
   neatinse; greșeli plantate prinse: QMD cu memoria partajată veche (32 768 < 33 024), grilă cu un bloc lipsă, mod 1, d > 64.
5. **Pe placă (în proba „vechi lângă nou”):** dS bit cu bit = `attn_back_tile0` (100 % exacte față de model, ca în os-358); timpul pe
   setul mare **~0,7–1,1 ms** față de 1,81 (DEDUS: bucla dP ~×2 mai scurtă, `dot` fără gardă; factorul ~2 NECUNOSCUT din os-358 rămâne).

## Rezultatul pentru `attn_back_tile0_2` (sesiunea 173, SIMULARE, T182 `crates/nvidia-ga104/tests/sim_attn_back_tile0_2.rs`)
- **Binarul** (Mesa NAK 25.0.7 reconstruit în cloud; `attn_back_tile0` a ieșit **identic**, `7107ecc6…`): `attn_back_tile0_2.sm86.sass`
  SHA-256 **`9a8f97603cbcc7a89bec9430ed0d34ba940dc03077841064d85cb9962f4a00d2`**, **52 de registre**, 33 024 de octeți partajați.
- **Predicția 1: DA** — 0 `mufu`, nicio atomică, nicio memorie locală; aceleași 32 `fmul` + 32 `fadd` (partea a doua) și 6 `bar.sync`
  ca vechiul; `ffma` 88 (vechi 17) doar pentru că buclele sunt desfăcute (64 + 16 restul lui d + 4 + 4 în `dot`).
- **Predicția 2: DA după o reparație.** Prima compilare (toate cele 16 + 16 citiri ale plăcilor dO și V emise deodată, apoi cele 16 ale
  lui P): **135 de registre**, apoi 123 cu dO și V despărțite — peste 80, ar fi lăsat **1 bloc pe SM** (8 warp-uri în loc de 24).
  Cauza: placa P (16 citiri cu adrese pe 64 de biți). Cu citirile emise **câte 8** (8 citiri, apoi cele 8 scrieri partajate, de două
  ori pe placă): **52 de registre** → 3 blocuri pe SM (ca vechiul, 50). Deci F3 e pe jumătate: 8 așteptări de memorie pe placă → 2.
- **Predicția 3: DA** — bucla dP: **104 instrucțiuni la 64 `ffma` = 26 la 16** (vechi 53), întârzieri 143 / 4 ≈ 36 la 16 (vechi
  192), **0 `bssy`**; bucla `dot` de sub diagonală: 43 de instrucțiuni la 4 `ffma`, **0 `bssy`** (vechi 21 la 1 cu 1 `bssy` = 84 la 4);
  pe diagonală 79 la 4 cu garda (4 `bssy`). Urcarea plăcilor: 8 `ldg` înaintea celor 8 `sts`, de două ori.
- **Predicția 4: DA** — emularea literală a noului GLSL (placa dO de 64, plăcile V / P de 65, placa dP rotită, buclele desfăcute cu
  restul lui d, `dot` fără gardă sub diagonală) = `attn_back` mod 0 **bit cu bit** pe cele 10 forme din T175 (d = 1, 32, 40, 64;
  T = 5 … 200); lansarea = `attn_back_tile0` bit cu bit; NaN / 1e30 deasupra diagonalei lui P nu schimbă niciun bit; prinse: QMD cu
  memoria partajată veche (32 768), grila cu un bloc lipsă (exact rândurile 128–129 nescrise, = emularea), bloc 128, grilă peste ⌈T/64⌉,
  mod 1 (nimic scris), d = 65, garda scoasă pe placa diagonalei, restul lui d sărit.
- **Predicția 5 (pe placă): NEMĂSURATĂ.** Rămâne ~0,7–1,1 ms pe setul mare față de 1,81 (DEDUS).
- `tools/check.sh`: **872 de teste** (+7 ignorate), format, clippy, build no_std — PASS.

**Următorul pas:** `attn_back_tile_2` (modul 1: dQ, dK, dV; bucla dK/dV are 120 de instrucțiuni la 16 `ffma`) cu aceleași schimbări —
plan + predicții întâi, cu grija la registre de aici (citiri câte 8) —, apoi proba „vechi lângă nou” și nota pentru OS.

## `attn_back_tile_2` — planul și predicțiile (sesiunea 173, scrise ÎNAINTE de cod)
Program nou **lângă** `attn_back_tile` (modul 1: dQ, dK, dV; cel vechi neschimbat). Aceleași argumente, grilă [⌈T/64⌉, H, B], blocuri
de 256, model `attn_back_col_gpu`; fiecare lanț în aceeași ordine. Schimbările:
- **F1** — în trecerea dQ, placa dS `[t][j]` din `sa` cu **rânduri de 65** (citită pe rânduri `ty` / `ty+1` → acum bănci diferite);
  în trecerile dK / dV `sa` rămâne `[i][t]` cu rânduri de 64 (citirile acolo sunt deja pe bănci vecine). `sb` neschimbat (64).
  Memoria partajată: 64 × 65 + 64 × 64 float = **33 024 de octeți** → tot 3 blocuri pe SM (ca la `attn_back_tile0_2`).
- **F2** — dQ: plăcile `jt < bx` fără garda `j <= t` (acolo `j < t0 ≤ t`); dK / dV: plăcile `it > bx` care se termină înainte de T
  fără garda `i < T && t <= i` (acolo `t < it·64 ≤ i < T`). Garda lui `i < T` rămâne pe ultima placă: un termen `ffma(0, 0, acc)` în
  plus ar face din −0 un +0 (vezi T175, „bucla e până la d”).
- **F3** — urcarea plăcilor câte 8 citiri globale, apoi cele 8 scrieri partajate (lecția de la `attn_back_tile0_2`: câte 16 → peste 120
  de registre).
- **F4** — buclele peste `jj` (dQ) și `ii` (dK / dV) desfăcute câte 4 (64 e multiplu de 4: fără rest).

Predicții:
1. **Aceleași operații:** 0 `mufu`, nicio atomică, nicio memorie locală; aceleași `bar.sync` ca vechiul.
2. **Registre ≤ 80** (vechi 73) → 3 blocuri pe SM; 33 024 de octeți partajați în QMD.
3. **Buclele:** dQ sub diagonală ≤ 30 de instrucțiuni la 16 `ffma` (vechi 76, 4 `bssy`) și 0 `bssy`; dK / dV fără gardă ≤ 30 la 16
  (vechi 120, 8 `bssy`) și 0 `bssy`.
4. **SIMULARE:** emularea literală a noului GLSL = `attn_back` mod 1 (dqkv) **bit cu bit** pe formele lui T174; lansarea = `attn_back_tile`;
   prinse: QMD cu memoria partajată veche, grilă scurtă, mod 0, d > 64, garda `i < T` scoasă pe ultima placă (un −0 devine +0).
5. **Pe placă:** dqkv bit cu bit = `attn_back_tile`; timpul pe setul mare **~1,2–2,0 ms** față de 3,99 (DEDUS).
