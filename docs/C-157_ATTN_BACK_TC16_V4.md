# C-157 — `attn_back_tile_tc16_v4`: atenția înapoi pe circuitele f16, cu citiri de 128 de biți și placa următoare citită înainte

Sesiunea Cyber-GPU 248, 26 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise **înainte** de cod
și de compilare (commit-ul care urcă doar această notă). De ce: comparația os-457 (`docs/C-157_P2_OS457_ATTN_TC.md` §3) —
`attn_back_tile_tc16` stă > 90 % din timp în așteptarea citirilor (32 de citiri scalare pe fir pe pas de placă, NAK ține
~5 în zbor → ~8 așteptări pe pas).

## 1. Ce schimbăm (și ce nu)

`tools/nak/attn_back_tile_tc16_v4.comp` = `attn_back_tile_tc16.comp` cu două schimbări, doar la citirea din memorie:

1. **Câte 4 numere vecine pe citire** (`vec4`, `ld.global.128`): fiecare placă 64 × 64 de sa și de sb = 1 024 de vectori,
   4 pe fir (256 de fire) → **4 + 4 citiri pe fir pe pas de placă**, în loc de 32. Firul `e = lid + 256·s` (s = 0..3) ia rândul
   `r = e / 16` și coloanele `x = 4·(e % 16) .. +3`; cele 4 numere ajung în aceleași două cuvinte ale memoriei partajate în care
   le punea tc16 (`r·32 + x/2` și `+1`, câte două f16, jumătatea de jos întâi).
2. **Placa următoare citită în registre înainte de produsele plăcii curente** (ca `mm_tc_v4`): după a doua barieră pleacă
   citirile pentru `kt + 1`, apoi lucrează circuitele pe `kt`. Fără ramificări: o citire „în afară” ia vectorul 0 al bufferului
   (mereu valid) și valoarea e înlocuită cu 0 la scrierea în memoria partajată.

Rămân neschimbate: aceleași valori f32 alese (aceleași condiții `keep` pe fiecare element, 0 în rest), aceeași rotunjire la f16
(`packFloat2x16`, RNE), aceleași locuri în memoria partajată, același lanț `coopMatMulAdd`, aceiași pași săriți pe diagonală,
aceeași scriere a rezultatului, același cbuf (argumentele la 0x700), aceeași grilă [⌈T/64⌉, H, B] și blocuri de 256.
Deci **aceiași biți ca tc16** — judecabil cuvânt cu cuvânt pe placă.

**Doar pentru forme aliniate** (`nvidia_ga104::attn_back_tile_tc16_v4::fits`): T și d = C / H multipli de 4, adresele qkv, p,
dout, dS multipli de 16 octeți. Atunci orice vector e aliniat și stă întreg în rândul lui. Altfel OS-ul rămâne pe tc16
(o citire de 128 de biți nealiniată e o eroare pe placă; modelul din `gpu-sim` refuză, fără scriere). #513 (T 512, C 768, H 12,
d 64) e aliniat; din probă, `t130-d64` nu e (T = 130).

**O diferență de citire (nu de valori):** tc16 nu citea niciodată deasupra diagonalei lui P / dS (indicele era tăiat la
diagonală). v4 citește vectorul întreg care atinge diagonala, deci **cel mult 3 numere deasupra ei, în același rând** (în
buffer: T e multiplu de 4); ele sunt aruncate de alegere (`keep ? v : 0`, nu înmulțire), deci un NaN acolo dă aceiași biți.

## 2. Predicții (scrise ÎNAINTE de cod și de compilare)

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| V1 | NAK folosește citiri de 128 de biți | **16 `ld.global….128`** în cod (8 înainte de bucla plăcilor + 8 în ea), 0 citiri globale scalare de f32 în afară de cbuf | 65 % |
| V2 | registre (NAK 26.3.1) | **72–96** (tc16 = 48, plus 32 de numere ținute pentru placa următoare); memorie locală 0, 0 vărsări; aceleași `hmma` / `ldsm` / `movm` ca tc16; 16 384 B partajată | 55 % |
| V3 | ocuparea la treapta 64 KiB | ≤ 80 registre → **3 blocuri pe SM** (cât tc16, limita e memoria partajată cu rezerva de 1 KiB); 88–96 → 2 | — (calcul) |
| V4 | SIMULARE: citirile GLSL scrise în Rust fir cu fir | **aceleași valori în memoria partajată ca tc16** pe fiecare formă aliniată, bloc, trecere și placă (și cu NaN deasupra diagonalei); fiecare vector aliniat și întreg în bufferul lui; cel mult 3 numere citite deasupra diagonalei pe rând | 90 % |
| V5 | SIMULARE prin procesorul de comenzi | aceiași biți ca tc16; formele nealiniate (T = 130, d = 6, o adresă + 4) refuzate fără scriere | 95 % |
| V6 | pe placă: aceleași cuvinte ca tc16 | da | 85 % |
| V7 | pe placă: `atentie-513-b4` | **0,35–0,55 ms pe lansare** (tc16 0,87), adică ×1,6–2,5; la lotul 2 în pas ~2,5–4 ms pe 12 lansări | 35 % |

Prag scris: dacă pe placă v4 nu e măcar **×1,3** mai rapid decât tc16 la `atentie-513-b4`, așteptarea citirilor nu era cauza
principală (atunci: L1 / cozile de memorie când stau 3 blocuri pe SM — următorul pas ar fi o placă de 128 de rânduri).

## 3. Ce e nou pentru placă

Doar programul `attn_back_tile_tc16_v4` (SHA-256 notat la compilare). Nicio scriere în registre, niciun firmware, aceeași
lansare ca tc16. Se lansează **numai** când `fits` spune da.

## 4. Rezultate

(se completează după cod și teste)
