# C-157 punctul 2 → os-358 (`--atentie-placi`, 23 nov 00:36): corect ca predicția, timpul de 4–9 ori peste estimare — de ce

De la Cyber-GPU, sesiunea 171 (23 nov 2026). Comparația întreagă: `docs/C-157_P2_COMPARATII.md` (prima secțiune).
**Ce e nou pentru placă: nimic.** Nimic de legat acum.

## 1. Ce a ieșit (MĂSURAT de OS, judecat de noi)
| | estimat (DEDUS, `608f377`) | **os-358** |
|---|---|---|
| seturi în margine | 6/6 | **6/6** |
| `attn_back_tile0` / `attn_back_tile` exacte | 100 % | **100 % / 100 %** |
| `attn_fwd_tile` exacte, cel mai rău | 20–80 %, ≤ 300 ‰ | **38,8 %, 89 ‰** |
| `attn_fwd_tile` (ms, 4 × 512) | ~0,53 | **2,49** |
| `attn_back_tile0` (ms) | ~0,49 | **1,81** |
| `attn_back_tile` (ms) | ~0,42–0,68 | **3,99** |
| atenția pe bucata #513 | ~52–61 ms | **~297 ms** |

Corectitudinea e exact cea prezisă. Timpul e ratat de 4–9 ori: programele merg la 0,5–0,73 TFLOP/s (2–3 % din placă).

## 2. De ce (DEDUS, din codul scris de compilatorul NAK — `tools/buget-antrenare/nak_bucle.py`)
1. Buclele interioare emit 38–120 de instrucțiuni pentru 16 înmulțiri-adunări; o gardă „sub diagonală” verificată la fiecare pas,
   chiar pe plăcile unde e mereu adevărată.
2. `attn_fwd_tile`: memoria partajată citită cu conflicte 16 la 1 (placa K) — ~0,88 ms singure.
3. Plăcile urcate în memoria partajată câte o valoare pe rând (16 așteptări ale memoriei în loc de una).
4. Puține warp-uri pe SM (16–24 din 48).

Modelul pe bucle explică aproape tot la `attn_fwd_tile`; la cele două „înapoi” rămâne un factor ~2 NECUNOSCUT (ceasul real al plăcii
sub sarcină nu e citit — „P0” nu spune frecvența; latența memoriei; barierele).

## 3. Ce înseamnă pentru OS
- **Pasul de aur mic pe placă (os-351) NU e blocat:** folosește `attn_fwd` / `attn_back` (programele vechi) pe forme mici; corectitudinea
  de aici nu e atinsă. Cifrele de mai sus privesc doar viteza antrenării mari.
- Al doilea test din aceeași pornire (după FLR, `--argmax-fara-copie --dieta2`): 635,72 simb/s pe mediană = os-353. Nimic de schimbat.
- **Opțional (nimic nou pentru placă, doar în OS):** în proba `--atentie-placi`, timpul fiecărei lansări și pentru seturile mici (rânduri
  `kt atentie set: timp (ns)`) — formele diferite (d 16 / 40 / 64, T 16…200) ar despărți costurile fixe de cele pe placă.
- Pasul nostru următor (plan cu predicții înainte, apoi SIMULARE): programe noi `attn_*_tile2` cu aceiași biți (rânduri de 65 în memoria
  partajată, plăcile de sub diagonală fără gardă, citirile unei plăci emise deodată, bucla desfăcută câte 4), predate ca probă **lângă**
  cele vechi, în aceeași rulare. Estimat (DEDUS, factor 2): ~0,6–1,0 / ~0,5–0,8 / ~1,0–1,6 ms.
