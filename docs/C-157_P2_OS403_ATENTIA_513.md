# C-157 punctul 2 (după os-403) — pasul #513 pe atenția `_3`: ce se schimbă și ce estimăm

Sesiunea 206, 24 nov 2026, în cloud, **fără placă**. Planul și predicțiile sunt scrise și urcate **înainte** de test.
Corectitudinea se verifică doar în **SIMULARE**; timpul e **DEDUS** (nemăsurat).

## De ce
Profilul os-403 (`--antrenare-513`, 10 pași, 385 de lansări): `mm_train` 718 ms (63,9 %), `attn_fwd` 212 ms (12 lansări,
**17,7 ms pe strat**), `attn_back` 159 ms (24 lansări). Estimasem (os-395, `tools/buget-antrenare/pas513.py`) 91 % înmulțiri și
~9 % atenție — **greșit la atenție**: socoteala număra doar operațiile, nu și cât de prost merg programele vechi de atenție.

Cauza (citit în codul OS-ului, `kernel/src/drivers/gpu_i14.rs`, `train_513_resident`): pasul #513 e construit cu
**`Plan::new`** = programele de atenție **de bază** (`attn_fwd`, `attn_back`: blocuri de 32, un fir pe rând). Programele pe
plăci, generația 3, există de la sesiunea 187 (`Plan::new_tiles3`), au trecut pe placă (os-375: **identice bit cu bit** cu `_2`;
os-364: `_2` identice bit cu bit cu prima generație) și, pe bucata de atenție măsurată `atentie-513-b4` (B 4, T 512, C 768,
H 12, un strat), lanțul gen 3 durează **3,43 ms** (1,00 + 0,83 + 1,60; os-375), gen 4 3,16 ms (os-386). Nu au fost legate
niciodată în `--antrenare-513`.

## Ce propunem OS-ului (un singur lucru nou)
În `train_513_resident`, la cele două construcții (`probe0` și `plan0`), `Plan::new` → `Plan::new_tiles3` (aceeași semnătură;
`.resident()` și `rebatch` păstrează `tiles3`). Nou pentru placă: **nimic nou ca program sau registru** — cele 3 programe `_3`
au trecut deja pe placă în probele de atenție (os-375; gen 4 la os-386), dar nu în niciun pas de antrenare (și
`--antrenare-mini --vram` folosește tot `Plan::new`); nou e doar că intră în pasul #513.

## Predicții (scrise acum, înainte de test)
1. **Lista (pe procesor, forma reală 2 × 512, 12 straturi):** `new_tiles3(...).resident()` are **aceiași octeți de arenă** ca
   `Plan::new(...).resident()`; aceleași **385** lansări AdamW pe pas; aceleași etichete și straturi în aceeași ordine; cele
   **349** lansări care nu sunt atenție au același program, același cbuf, aceeași grilă; cele **36** de atenție (12 înainte +
   24 înapoi) au **același cbuf**, alt program (`attn_fwd_tile3` / `attn_back_tile0_3` / `attn_back_tile_3`) și grila
   [8, 12, 2]. Împărțirea în trimiteri (255 + 130) rămâne aceeași, pentru că numărul de lansări e același.
2. **Corectitudinea (SIMULARE, lățimile lui #513, un strat, greutăți inventate, 2 × 128 = 2 plăci pe text, cu diagonala):**
   un pas AdamW pe `new_tiles3` = oracolul pe procesor în toleranțele hub-ului (ca T165 pe `Plan::new`); față de `Plan::new`
   **nu bit cu bit** (atenția înainte adună în altă ordine, ca la `new_tiles`), eroarea ≤ 1·10⁻⁶ relativ.
3. **Timpul (DEDUS):** la B = 2 grila are 192 de blocuri în loc de 384 → lanțul pe strat ~0,5–0,75 × 3,43 = **1,7–2,6 ms**,
   deci atenția pasului **~20–40 ms în loc de 371 ms**. Pasul pe placă **~1 123 → ~770–800 ms ≈ 1 280–1 330 simb/s** (×1,4).
   Restul rămâne la `mm_train` (~718 ms = ~90 % din pasul nou) → următoarea tăietură e `--mixt` pe #513 (`mm_tc`, scara 2¹⁴).
4. **Pierderile pe loturile sintetice** rămân pe oracolul `ORACLE_513_SYNTH` la ≤ 1·10⁻³ pe pașii 1–9 (ordinea diferită a
   sumelor din atenție schimbă doar a 6-a–7-a cifră).

## Rezultat (sesiunea 206, în cloud)
| Predicția | Ce a ieșit | Verdict |
|---|---|---|
| 1. lista la 2 × 512 | arena 2 560 MiB la ambele; 385 lansări; 349 identice; 36 de atenție [12, 12, 12] cu alt program, același cbuf, grila [8, 12, 2] | PASS (pe procesor, T212) |
| 2. un strat la lățimile #513, 2 × 128 | = oracolul: eroarea identică, norma 2,7·10⁻⁹ rel., gradientul 2,8·10⁻⁷ abs, greutățile 1,6·10⁻⁵ (tol. 1·10⁻⁴); față de `Plan::new`: eroarea și norma **identice** (9,750025 / 3,738398; prezisesem ≤ 1·10⁻⁶), gradienții nu bit cu bit (2,76 vs 2,91·10⁻⁷ față de oracol) | PASS (SIMULARE, T212) |
| 3. timpul pe placă | — | NETESTAT (așteaptă testul OS-ului) |
| 4. pierderile pe oracol | — | NETESTAT |
