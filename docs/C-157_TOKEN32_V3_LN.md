# C-157 p.2 — `token32_v3`: LayerNorm calculat în înmulțirea care urmează (plan + predicții, scrise ÎNAINTE de cod)

Sesiunea Cyber-GPU 277, 27 nov 2026, în cloud, **fără placă**. Nimic de aici n-a rulat pe placă. Pornește de la
`docs/C-157_TOKEN32_V2C_PLACA.md` §2–§4: la G = 768, cele 25 de `layernorm256` din `token32_v2` costă **459 µs** pe simbol
(un singur grup de 32 de fire face toată treaba, 18,6 µs fiecare; în listă ~4 µs), plus 25 de bariere de ~2,0 µs.

## 1. Ideea

Fiecare LayerNorm din lista de azi e urmat de o înmulțire care citește **doar** rezultatul lui:
`ln1 → qkv` (`q8_fused32`), `ln2 → fc` (`q8_fused32` cu GELU), `ln final → logit-uri` (`q8_matvec32`).
În `token32_v3` cele două devin **o singură operație** (operațiile noi 13 și 14 ale programului):

1. **Începutul (o dată pe grup, înainte de rândurile lui):** fiecare din cele G grupuri calculează singur media și abaterea pe
   cei 768 de x, cu **exact adunările lui `layernorm256`** (firul t joacă firul 32·w + t al warp-ului w, w = 0..7; suma lui pe
   k = 0..3 de la 0,0; același „fluture” de `shuffle xor` 16..1; sumele celor 8 warp-uri adunate în ordinea w = 0..7) → aceeași
   medie și aceeași abatere, bit cu bit.
2. **x normalizat ținut în registre:** în înmulțirile pe sferturi de bloc (`q8_*32`) firul t citește **mereu aceleași 24 de x**
   (indicii 256·m + 8·t + j, m = 0..2, j = 0..7), pentru orice rând. Deci firul își calculează o dată cele 24 de valori
   normalizate, cu aceeași expresie ca `layernorm256` (`(x − medie) · inv · g + b`), și le folosește la toate rândurile grupului
   (3 rânduri la qkv, 4 la fc, ~66 la logit-uri) — fără memorie partajată nouă (la 16 grupuri pe SM nu mai încape: 2 176 B azi,
   ~3 KiB cel mult) și fără să mai recitească x din memorie la fiecare rând.
3. **Ieșirea LayerNorm-ului** se scrie în continuare (grupul 0, aceleași valori) — nimeni n-o mai citește în program, dar
   memoria rămâne cea a listei (comparațiile bit cu bit pe tampoane rămân valabile).
4. **Restul neschimbat:** aceleași operații (copie, atenție, vadd, argmax, embed_dev, `layernorm256` și înmulțirile separate,
   dacă o pereche nu se poate lipi), aceeași barieră, aceeași citire prin L1 ca `token32_v2`.

Lista: **88 → 63 de operații** (25 de LayerNorm-uri lipite) → **87 → 62 de bariere**. Intrarea lipită = intrarea înmulțirii
(operația 13 sau 14, aceeași grilă, aceleași argumente de la 0x28) + argumentele LayerNorm-ului copiate la **0xa0** (x, g, b, y,
n, 1/n, eps). Driverul lipește o pereche **doar** dacă: LayerNorm-ul are grila 1, înmulțirea citește exact ieșirea lui
(adresa x a înmulțirii = y), n = 32 · blocuri și n ≤ 768 (24 de valori pe fir), octeții 0xa0..0x100 ai înmulțirii sunt 0.
Altfel perechea rămâne separată (operațiile 8 + 10/11, ca în v2).

Interfața publică: **doar adăugări** — modul nou `nvidia_ga104::token_ln` (programul, registrele, SHA-256, lista lipită, QMD-ul);
`Prog` nu primește variante noi (ar strica potrivirile complete din OS).

## 2. Predicțiile (scrise înainte; se judecă pe rularea OS-ului, lângă `token32_v2` din același lanț)

| # | Ce | Predicție | Încredere |
|---|---|---|---|
| L1 | biții | aceleași 16 simboluri ca `token32_v2`, amprenta `0x6c8501f0`; contorul barierei (63 − 1) · G | mare |
| L2 | listarea NAK | 0 memorie locală (cele 24 de valori stau în registre), **≤ 128 de registre** (16 grupuri pe SM la treapta de 64 KiB rămân), aceeași memorie partajată (2 176 B); începutul lipit are aceleași `fadd` / `shfl` / `mufu.rsq` ca `layernorm256` | medie |
| L3 | placa pe simbol, G = 768 | **1,18–1,38 ms** (v2 azi 1,666; tăiem ~0,46 ms de LayerNorm + 25 × 2,0 µs de barieră, adăugăm 25 × 1,5–4 µs de început în fiecare grup și puțin la înmulțiri) | medie |
| L4 | cu driverul (~0,61 ms) | **~505–560 simb/s** (lista obișnuită 484, v2 ~440) | medie |
| L5 | G = 384 față de G = 768 | G = 768 tot mai repede (cum a fost la v2) | medie |
| L6 | partea driverului | la trecerea de la un simbol la altul se schimbă tot puține cuvinte în listă (< 1/20 din ea), ca la v2 | mare |
| L7 | SIMULARE | un simbol al listei de azi + argmax la lățimea lui Kronos, G = 384 și 768 = lansările separate bit cu bit (logit-uri, argmax, ieșirile LayerNorm) | mare (aici) |

**Cum se citește rezultatul:** dacă L1 iese și L3 cade în bandă → `token32_v3` devine noua bază pentru „un program pe simbol” și
pasul următor e atenția pe mai multe grupuri (D2). Dacă L3 iese **peste** bandă (> 1,38 ms) → începutul lipit costă mai mult decât
cred (cele 768 de grupuri citesc aceiași 3 KiB deodată: L2 / L1) și măsurăm cu un `token32_v3c` cu ceas. Dacă L1 nu iese → `v3` nu
se folosește; diferența de biți se caută în listare (contractarea `fmul` + `fadd` → `ffma` altfel decât în `layernorm256`).

## 3. Pașii

1. `tools/nak/token32_v3.comp` (copia lui `token32_v2.comp` + operațiile 13 și 14) + compilarea NAK; în listare: L2.
2. Rust (doar adăugări): `nvidia_ga104::token_ln` — `TOKEN32_V3`, `TOKEN32_V3_REGISTERS`, `TOKEN32_V3_SHA256`, `work_list_ln`
   (lipirea, cu refuzurile de mai sus), `token32_v3_qmd_wide`.
3. `gpu-sim`: modelul `token32_v3` = operațiile lui `token32` + 13 / 14 = modelul `layernorm256` urmat de modelul înmulțirii
   (SIMULARE: arată lista și argumentele, nu biții SASS-ului — aceia sunt DEDUȘI din listare).
4. Testul L7 (SIMULARE) + nota pentru OS (PREDARE): o opțiune nouă, oprită implicit, `token32_v3` după `token32_v2`.

## 4. Ce a ieșit (sesiunea 277, după predicții)

- Compilat (Mesa NAK 25.0.7; `token32_v2` reconstruit identic `5f4d22a0…`): **10 548 de instrucțiuni, 97 de registre** (v2: 92),
  0 memorie locală, aceeași memorie partajată; SHA-256 `8ecf4fed…5b719`. **L2 PASS** (listare): ≤ 128 registre → 16 grupuri pe SM
  rămân; bariera aceeași (4 `cctl`, 2 adunări atomice); +80 `shfl.bfly` și +1 `mufu.rsq` = exact un LayerNorm în plus (începutul
  lipit apare o singură dată în program); toate scrierile `strong.sys`.
- **Prima compilare a dat 140 de registre** (16 grupuri pe SM n-ar mai fi încăput): cele 24 de valori erau o variabilă globală, deci
  NAK le ținea vii prin tot programul (și prin atenție). Mutate într-un tablou local al operației lipite (în `main`) → 97.
  Aritmetica nu s-a schimbat.
- **L7 PASS în SIMULARE** (T292): G = 384 și 768, trei simboluri, = lansările separate bit cu bit (logit-uri, argmax, ieșirile
  LayerNorm); toate LayerNorm-urile din lista de azi se lipesc; contorul (n − 1) · G. **L6 PASS** (lista lipită se schimbă în
  puține cuvinte între simboluri). Concluzia „aceiași biți pe placă” rămâne DEDUSĂ din sursă și listare; o decide L1.
- Nimic rulat pe placă. PREDARE: `integration/cyber-universe/C-157_P2zzt_token32-v3-ln-lipit.md`.

## 5. Pe placă (os-498, sesiunea 280)

L1, L5, L6 CONFIRMATE; **L3 NECONFIRMATĂ** (1,453 ms la G = 768, peste banda 1,18–1,38), **L4 NECONFIRMATĂ** (478 simb/s).
Comparația întreagă: `docs/C-157_TOKEN32_V3_PLACA.md` (T296).
