# C-157 p.2 după os-423 — `mm_tc_pf` cu citiri de 128 de biți (`mm_tc_v4`)

Sesiunea Cyber-GPU 230, 25 nov 2026, în cloud, **fără placă**. Predicțiile de mai jos sunt scrise **înainte** de compilare și de cod.

## De ce
Sesiunea 229: NAK 26.3.1 ține cel mult ~5 citiri în zbor (6 „numărătoare de așteptare”). `mm_tc_pf` face 32 de citiri de câte un număr
pe felie → tot ~6–7 așteptări pe felie. Leacul mai mare: **mai puține citiri**, câte 4 numere vecine deodată (o citire de 128 de biți):
4 + 4 = **8 citiri pe felie** în loc de 32.

## Ce schimbăm
- `tools/nak/mm_tc_v4.comp`: ca `mm_tc_pf` (fără ramificări, felia următoare citită după barieră, zeroul ales la scrierea în memoria
  comună), dar fiecare fir citește 4 + 4 grupuri de câte 4 numere (`vec4`).
- **Doar pentru produse „aliniate”**: dimensiunea de-a lungul rândului fiecărei matrice stocate (k pentru A, m pentru Aᵀ, n pentru B,
  k pentru Bᵀ) e multiplu de 4, iar adresele lui A și B sunt multipli de 16 octeți. Altfel gazda lansează `mm_tc_pf`
  (`nvidia_ga104::mm_tc_v4::fits`). Formele #513 (768, 2304, 3072, 1024…) sunt toate aliniate.
- Aceleași valori rotunjite la f16 în aceleași felii din memoria comună, același lanț `coopMatMulAdd`, aceeași scriere, același
  `bias` → **aceiași biți ca `mm_tc`**.

## Predicții (scrise înainte)
| # | Ce | Prezis | Încredere |
|---|---|---|---|
| P1 | NAK folosește citiri de 128 de biți | 8 `ld.global` pe felie (+ cele din citirea dinaintea buclei), toate `.b128` / `.128` | 80 % |
| P2 | registre (NAK 26.3.1) | **≤ 248** (aceleași 32 de valori ținute, adrese mai puține); memorie locală 0 | 65 % |
| P3 | listarea, regula strictă (`citiri_seriale.py`) | **5–8 în zbor** (cele 8 ale unei felii pot sta pe 6 numărătoare) | 60 % |
| P4 | SIMULARE (T236): aceleași valori în memoria comună ca `mm_tc`, fără citiri în afara lui A / B, fără citiri nealiniate; aceiași biți prin procesorul de comenzi; produsele nealiniate refuzate | da | 90 % |
| P5 | pe placă: aceiași biți ca `mm_tc` (132 780 / 132 780) | da | 85 % |
| P6 | pe placă: qkv-513 | **~5–9 TFLOP/s** (os-423: 1,03; `mm_tc_pf` prezis 3–5) | 40 % |

Prag scris: dacă pe placă `mm_tc_v4` nu e măcar ×1,5 față de `mm_tc_pf`, frâna nu mai e numărul de citiri — atunci e la scrierea în
memoria comună (16 scrieri de 2 octeți pe vector) sau la bariere.

## Ce e nou pentru placă
Doar programul `mm_tc_v4` (SHA-256 notat la compilare). Nicio scriere în registre, niciun firmware, aceeași lansare ca `mm_tc`.
O citire de 128 de biți nealiniată ar fi o eroare pe placă → OS-ul îl lansează **numai** când `fits` spune da.

## Rezultatul (sesiunea 230, în cloud) — ce a ieșit față de predicții
Unealta: Mesa 26.3.1 reconstruit în cloud; `mm_tc_pf` recompilat a ieșit **identic** (SHA `48defc5d…`), deci unealta e aceeași.
Programul a ieșit din **prima** compilare: SHA-256 `488ca341fd5bf400cbaf41eebe65998c5f7ae167b8c5443828ef451e1967acf1`.

| # | Prezis | Ieșit | Verdict |
|---|---|---|---|
| P1 | 8 citiri de 128 de biți pe felie | **16 `ld.global…b128`** = 8 înainte de buclă + 8 în buclă; restul de 32 citiri sunt `bias` la final (ca la `mm_tc_pf`); 48 de citiri în total față de 96 | PASS |
| P2 | ≤ 248 registre, 0 memorie locală | **128 registre**, 0 memorie locală, 16 `hmma` ca `mm_tc` | PASS — și mai bine: 128 × 128 registre → 4 blocuri pe unitate după registre (față de 2), cât permite memoria comună la treapta maximă |
| P3 | 5–8 în zbor (regula strictă) | **5 în zbor** (marginea de jos); în buclă cele 8 citiri sunt așteptate în ~3 locuri, toate înainte de primele circuite speciale (nu se suprapun cu ele, ca la `mm_tc_pf`) | PASS (MĂSURAT pe text) |
| P4 | SIMULARE (T236) | transcrierea citirilor (fir cu fir, vector cu vector, toate opțiunile, margini): **aceleași valori** în memoria comună ca `mm_tc` pentru toate produsele aliniate; fiecare vector aliniat și întreg în buffer; formele nealiniate prinse; o greșeală plantată (ordinea benzilor la Aᵀ) prinsă; prin procesorul de comenzi: **aceiași biți** pe 50 de produse × opțiuni; produsele nealiniate (forme sau adrese) **refuzate** fără scriere | PASS (SIMULARE) |
| P5 | aceiași biți pe placă | — | NOT TESTED |
| P6 | qkv-513 ~5–9 TFLOP/s | DEDUS: așteptările pe felie ~6–7 → ~3, plus ocuparea dublă (registre) | NOT TESTED |

Ce ar urma pentru mai mult (DEDUS): citirile feliei următoare tot nu se suprapun cu circuitele speciale (NAK le așteaptă înainte de
primele `hmma`); ocuparea mai mare (4 blocuri) e cea care ar ascunde restul.
