# C-157 p.2 după os-423 — `mm_tc` cu citirile în avans (`mm_tc_pf`)

Sesiunea Cyber-GPU 229, 25 nov 2026, în cloud, **fără placă**. Predicțiile de mai jos sunt scrise **înainte** de compilare și de cod.

## De ce
os-423: `mm_tc` corect bit cu bit pe placă, dar **1,03 TFLOP/s**. Listarea: 96 / 96 de citiri din memoria plăcii așteptate una câte
una (cel mult 1 în zbor, T234). Cauza în program: fiecare citire stă într-un `if (în matrice)` → o ramificare pe citire.

## Ce schimbăm (doar ordinea citirilor; aritmetica neschimbată)
- `tools/nak/mm_tc_pf.comp`: fiecare fir citește cele 16 valori din op(A) și 16 din op(B) ale feliei **în registre, fără ramificare**:
  în afara matricei indicele devine 0 (element valid când felia există), iar valoarea e înlocuită cu 0 după citire (ca atenția gen 3,
  trecută pe placă).
- **Felia următoare se citește în timp ce circuitele speciale lucrează pe felia curentă** (un pas înainte, „dublu tampon” în registre).
- La fel ca `mm_tc`: plăci 64 × 64, 128 de fire, aceeași rotunjire f16 a acelorași valori (0 în afară), același lanț `coopMatMulAdd`
  cu K în pași de 16 în ordine, aceeași scriere, aceeași adunare f32 a lui `bias[col]`, aceeași memorie comună (24 KiB), același cbuf.

## Predicții (scrise înainte)
| # | Ce | Prezis | Încredere |
|---|---|---|---|
| P1 | registre (NAK 26.3.1) | 80–112 (azi 64; +32 de valori ținute în avans), memorie locală 0 | 70 % |
| P2 | listarea (`citiri_seriale.py`, T234): cele mai multe citiri în zbor | **≥ 16** (ținta 32); citirile feliilor așteptate înainte de următoarea ≤ 4; doar citirile lui `bias` (la final, o dată) rămân una câte una | 75 % |
| P3 | SIMULARE (T235): aceiași biți ca `mm_tc` pe margini (TA/TB/BIAS, forme parțiale) și pe formele #513 tăiate | da (prin construcție: aceleași valori, aceeași ordine) | 95 % |
| P4 | pe placă: aceiași biți ca `mm_tc` / modelul (132 780 / 132 780 ca os-423) | da | 85 % |
| P5 | pe placă: qkv-513 | **~5–10 TFLOP/s** (os-423: 1,03); seturile mici ~5–10 µs pe felie „la rece” → ~3–6 µs | 50 % |

Prag scris: dacă P2 iese (≥ 16 în zbor) dar pe placă rămâne < 2 TFLOP/s, frâna nu (mai) e la citiri — atunci e la scrierea în memoria
comună / bariere, și o căutăm acolo, nu la ocupare.

## Ce e nou pentru placă
Doar programul `mm_tc_pf` (SHA-256 notat la compilare). Nicio scriere în registre, niciun firmware, aceeași lansare ca `mm_tc`.

## Rezultatul (sesiunea 229, în cloud) — ce a ieșit față de predicții
Unealta: Mesa 26.3.1 reconstruit în cloud; `mm_tc` recompilat a ieșit **identic** (SHA `f60cd944…`), deci unealta e aceeași.

| # | Prezis | Ieșit | Verdict |
|---|---|---|---|
| — | — | **trei compilări au fost nevoie**: (1) cu `?:` și `&&` la fiecare citire → NAK face ramificări, iar la fiecare ramificare **așteaptă toate citirile în zbor**: tot 1 în zbor; (2) totul cu selecții (`mix`), dar cele 16 + 16 „e înăuntru?” ca valori booleene → NAK are doar 7 registre de predicat și a ales zeroul imediat după fiecare citire; (3) „e înăuntru?” ca biți ai unui singur număr, zeroul ales abia la scrierea în memoria comună → asta e programul | fapt despre NAK 26.3.1 |
| P1 | 80–112 registre | **248** (0 memorie locală, 16 `hmma` ca `mm_tc`) | **GREȘIT**; 128 × 248 încape tot 2 blocuri pe unitate, cât dă și memoria comună la treapta de 64 KiB |
| P2 | ≥ 16 în zbor (regula T234) | regula T234: **32 în zbor**, 34 / 96 așteptate (32 = citirile lui `bias` la final + câte una pe felie) | PASS pe regula scrisă |
| P2′ | — | **regula strictă (nouă)**: un „tabel de așteptare” (scoreboard) e un numărător, iar NAK dă fiecărei citiri unul din cele 6 și îl așteaptă pe cel mai vechi când are nevoie de unul liber → **cel mult 5 citiri în zbor** (atenția gen 3, trecută pe placă: 6). Toate celelalte înmulțiri: 1. Felia următoare e și ea așteptată înainte de circuitele speciale (nu se suprapune cu ele) | MĂSURAT pe text; regula din sesiunea 228 supraestima |
| P3 | aceiași biți ca `mm_tc` în SIMULARE | T235: transcrierea în Rust a citirilor (fir cu fir, felie cu felie, toate opțiunile, margini) pune **aceleași valori** în memoria comună ca `mm_tc` și nu citește niciodată în afara lui A / B; prin procesorul de comenzi: aceiași biți | PASS (SIMULARE) |
| P4 | aceiași biți pe placă | — | NOT TESTED |
| P5 | qkv-513 ~5–10 TFLOP/s | **corectat acum (DEDUS):** cu ~5 în zbor în loc de 1, felia ≈ 32 / 5 ≈ 6–7 așteptări în loc de 32 → **~3–5 TFLOP/s** (×3–5 față de os-423) | NOT TESTED |

Ce urmează pentru mai mult (DEDUS): mai puține citiri, nu mai multe în zbor — citiri de 128 de biți (4 valori pe citire: 8 în loc de
32 pe felie) când rândurile sunt aliniate; cu aceeași limită de ~5 în zbor, felia ar ajunge la ~2 așteptări.
