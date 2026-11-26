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

## 4. Rezultate (sesiunea 248, după predicții)

**Compilarea a schimbat planul: au ieșit două programe, nu unul.** Prima compilare (citiri de 128 de biți + placa următoare citită
înainte, într-un singur program) a dat **104 registre** (V2 prezicea 72–96 — GREȘIT) și, mai rău, listarea arăta că NAK **aștepta
toate citirile în zbor la fiecare ramificare**: alegerea „trecerea 0 sau nu” pentru rândul / coloana plăcii, scrisă între citiri,
devenise o ramificare (`bra !up0`, cu `wt=` pe citirile plecate), deci tot ~2 citiri în zbor (`citiri_seriale.py`). Am mutat alegerea
o dată pe placă, înaintea citirilor (aceleași valori) — de aici **5 citiri în zbor** (regula strictă). Apoi am păstrat ambele variante,
ca placa să spună care e mai bună:

| Program | Ce face | Registre | Blocuri / SM (64 KiB) | `ld.global.128` | în zbor (strict) | SHA-256 |
|---|---|---|---|---|---|---|
| `attn_back_tile_tc16` (azi) | 32 citiri scalare pe fir pe placă | 48 | 3 | 0 (6 scalare în cod, buclă) | 2 | `8cbbab7e…` (recompilat identic) |
| **`attn_back_tile_tc16_v4`** | 4 + 4 citiri de 128 de biți, toate plecate înainte de prima folosire | **72** | **3** | 8 | **5** | `b19c8451…5a86` |
| **`attn_back_tile_tc16_v4pf`** | ca v4 + placa următoare citită înainte de produse | **104** | **2** | 16 | **5** | `3ba898a9…f853` |

Ambele: 16 384 B partajată, 0 memorie locală, 0 vărsări în memorie, aceleași `hmma` / `ldsm` / `movm` ca tc16, 0 `ffma`, argumentele
la 0x700. La v4, condițiile `keep` pe element se calculează la citire, ca biți într-un singur cuvânt (mai puține registre ținute).

| # | prezis | ieșit | verdict |
|---|---|---|---|
| V1 | 16 `ld.global….128`, nicio citire scalară | v4pf: 16, nicio altă citire globală; v4 (fără citirea înainte): 8 | CONFIRMATĂ (v4pf) |
| V2 | 72–96 registre, 0 locală / vărsări | v4pf **104** (GREȘIT); v4 72; ambele 0 locală, 0 vărsări în memorie | GREȘITĂ pentru varianta prezisă |
| V3 | ≤ 80 → 3 blocuri; 88–96 → 2 | v4 3 blocuri, v4pf 2 | (calcul) |
| V4 | aceleași valori în memoria partajată ca tc16, vectori aliniați și întregi, ≤ 3 numere peste diagonală | DA pe 6 forme mici (toate blocurile / trecerile / plăcile) și #513 (capetele 0 și 11), cu NaN peste diagonală; citirea în avans după ultima placă tot în buffer; exact 3 peste diagonală la #513; 2 greșeli plantate prinse | CONFIRMATĂ (SIMULARE, T258) |
| V5 | aceiași biți ca tc16 prin procesorul de comenzi; nealiniatele refuzate | v4 și v4pf = tc16 bit cu bit pe 6 forme; T 130, d 6, dS + 4 refuzate fără scriere; din probă iau v4 5 / 6 seturi (nu `t130-d64`) | CONFIRMATĂ (SIMULARE, T258) |
| V6 | pe placă aceleași cuvinte ca tc16 | — | NOT TESTED |
| V7 | `atentie-513-b4` 0,35–0,55 ms | — | NOT TESTED |

**Predicție nouă pentru timp (scrisă acum, înainte de placă, DEDUS):** v4 ≈ v4pf, ambele în banda V7 (0,35–0,55 ms); dacă diferă,
**v4 (3 blocuri) mai rapid decât v4pf (2 blocuri)** — încredere 55 %. Pragul de ×1,3 față de tc16 rămâne.

Verificările întregi: vezi `docs/IMPLEMENTATION_STATUS.md`, sesiunea 248.
