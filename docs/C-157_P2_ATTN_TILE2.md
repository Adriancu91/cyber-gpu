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
