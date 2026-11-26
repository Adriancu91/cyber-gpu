# C-157 — `attn_back_tile0_tc_v4`: dS al atenției (dP pe circuitele f16) cu citiri și scrieri de 128 de biți

Sesiunea Cyber-GPU 251, 26 nov 2026, în cloud, **fără placă**. Planul și predicțiile de mai jos sunt scrise **înainte** de cod
și de compilare (commit-ul care urcă doar această notă). De ce: aceeași lecție ca la `attn_back_tile_tc16_v4` (sesiunea 248,
din os-457): programele atenției stau mai mult în așteptarea citirilor scalare decât în calcul. `attn_back_tile0_tc`
(sesiunea 245, legat în pas în sesiunea 250, PREDARE P2zs) citește tot scalar: 32 de citiri pe fir pentru plăcile dO / V,
16 pentru placa P, 32 pentru trecerea a doua (dS și P), plus 16 + 16 scrieri scalare în dS.

## 1. Ce schimbăm (și ce nu)

`tools/nak/attn_back_tile0_tc_v4.comp` = `attn_back_tile0_tc.comp` cu citirile și scrierile globale pe câte 4 numere vecine
(`vec4`, 128 de biți). Firul ia vectorii `e = lid + 256·s` (s = 0..3): rândul `r = e / 16`, coloanele `x = 4·(e % 16) .. +3`.

1. **Plăcile dO și V:** 4 + 4 citiri de 128 de biți pe fir (în loc de 16 + 16 scalare), toate plecate înainte de prima
   folosire; aceleași valori rotunjite la f16 (`packFloat2x16`, RNE) în aceleași cuvinte ale memoriei partajate
   (`r·32 + x/2` și `+1`).
2. **Placa P** (pentru lanțul `dot`): 4 citiri de 128 de biți (în loc de 16); aceleași valori f32 în aceleași locuri
   (`sk(r, x + m)`), 0 unde `i ≥ T` sau `j > i`.
3. **dP scris în dS** și **trecerea a doua** (dS = P·(dP − dot)·scale): pe vectori. Rândurile fiecărui fir rămân aceleași
   (`ty + 16·a`, deci `dr[a]` neschimbat); coloanele devin `4·tx .. +3` în loc de `tx + 16·k`. Aceleași operații pe fiecare
   element (`x = dS − dr`, `y = (P·x)·scale`), deci aceleași valori.
4. Lanțul `dot`, produsele pe circuite, grila [⌈T/64⌉, H, B], blocurile de 256, memoria partajată (33 024 B), cbuf-ul
   (argumentele la 0x700) — **neschimbate**.

Deci **aceiași biți în dS ca `attn_back_tile0_tc`** — judecabil cuvânt cu cuvânt pe placă.

**Doar pentru forme aliniate** (`nvidia_ga104::attn_back_tile0_tc_v4::fits`): T și d = C / H multipli de 4, adresele qkv, p,
dout, dS multipli de 16 octeți (o citire de 128 de biți nealiniată = eroare pe placă; modelul din `gpu-sim` refuză, fără
scriere). Altfel OS-ul rămâne pe `attn_back_tile0_tc`. #513 (T 512, d 64) e aliniat.

**Diferențe de citire / scriere (nu de valori), scrise dinainte:**
- pe placa de pe diagonală, un vector poate citi **cel mult 3 numere deasupra diagonalei** din P și dS, în același rând
  (aruncate de alegere `keep ? v : 0`, deci un NaN acolo dă aceiași biți);
- **dP pe placa de pe diagonală se scrie și deasupra diagonalei** (vectorul întreg, doar `j < T`), apoi trecerea a doua scrie
  acolo 0, ca azi; scrie același fir, în aceeași ordine, în rândurile blocului lui (niciun alt bloc nu le atinge). Rezultatul
  final: aceiași biți; ce e între cele două treceri diferă (doar în bufferul dS al blocului).
- un vector cu `j ≥ T` nu se citește și nu se scrie (T multiplu de 4: vectorul e întreg înăuntru sau întreg afară).

## 2. Predicții (scrise ÎNAINTE de cod și de compilare)

| # | Ce | Prezis | Încredere |
|---|---|---|---|
| U1 | citiri globale în cod | **toate de 128 de biți**: 8 (dO / V) + 4 (P) + 2 × 8 (trecerea a doua: plăcile din stânga și placa diagonală) = **28 `ld.global….b128`**, 0 citiri globale scalare; scrieri 4 + 3 × 4 = **16 `st.global….b128`**, 0 scalare | 55 % (numărul exact depinde de cum desface NAK buclele) |
| U2 | registre (NAK 26.3.1) | **64–88** (tc 56; 32 de numere ținute pentru dO / V); 0 memorie locală, 0 vărsări; aceleași 16 `hmma`, aceleași 8 `ffma.ftz` ale lanțului, 33 024 B partajată | 55 % |
| U3 | ocuparea la treapta 64 KiB | **1 bloc pe SM** (ca tc și `attn_back_tile0_3`: limita e memoria partajată, 33 024 B), oricâte registre ≤ 255 | — (calcul) |
| U4 | SIMULARE: citirile GLSL scrise în Rust fir cu fir | aceleași valori în memoria partajată ca tc (plăcile f16 dO / V și placa f32 P), pe fiecare formă aliniată, bloc și placă, și cu NaN deasupra diagonalei; fiecare vector aliniat și întreg în rândul lui; cel mult 3 numere peste diagonală pe vector | 90 % |
| U5 | SIMULARE: trecerile scrise în Rust (dP, apoi dS) | dS final = al lui tc bit cu bit pe fiecare formă aliniată (incl. NaN peste diagonală în P); nicio scriere în afara rândurilor blocului sau peste `j ≥ T` | 90 % |
| U6 | SIMULARE prin procesorul de comenzi | aceiași biți ca `attn_back_tile0_tc`; formele nealiniate (T = 130, d = 6, o adresă + 4) refuzate fără scriere | 95 % |
| U7 | pe placă: aceleași cuvinte ca `attn_back_tile0_tc` | da | 85 % |
| U8 | pe placă: `atentie-513-b4` | **×1,2–1,8** mai rapid decât `attn_back_tile0_tc` (o singură placă pe SM, deci așteptarea citirilor cântărește mult; dar lanțul `dot` pe 64 de fire rămâne serial) | 30 % |

Prag scris: dacă pe placă v4 nu e măcar **×1,15** mai rapid decât `attn_back_tile0_tc`, frâna e lanțul `dot` (64 de fire
din 256 lucrează, 64 de pași seriali pe placă) — pasul următor ar fi lanțul împărțit pe 4 fire pe rând (alt ordin al sumei,
deci alți biți, cu judecată nouă).

## 3. Ce e nou pentru placă

Doar programul `attn_back_tile0_tc_v4` (SHA-256 notat la compilare). Nicio scriere în registre GPU, niciun firmware, aceeași
lansare ca `attn_back_tile0_tc`. Se lansează **numai** când `fits` spune da.

## 4. Rezultate (sesiunea 251, după predicții)

Unealta NAK 2 (Mesa 26.3.1) construită din nou în cloud; `attn_back_tile0_tc` recompilat iese **identic** (`95fc392e…ef8d`),
deci comparația de mai jos e pe aceeași unealtă.

| Program | Registre | Partajată | Blocuri / SM (64 KiB) | Citiri globale | Scrieri globale | în zbor (`citiri_seriale.py`) | SHA-256 |
|---|---|---|---|---|---|---|---|
| `attn_back_tile0_tc` (azi) | 56 | 33 024 B | 1 | 88 scalare | 64 scalare | cel mult 32, strict 6 (pe 4 octeți) | `95fc392e…ef8d` |
| **`attn_back_tile0_tc_v4`** | **56** | 33 024 B | 1 | **28 de 128 de biți** | **16 de 128 de biți** | cel mult 8, strict 6 (pe 16 octeți) | `5d5eb75d…72d4` |

Ambele: 0 memorie locală, 0 vărsări, aceleași 16 `hmma.m16n8k16.f32`, aceleași 12 `ldsm`, aceleași 8 `ffma.ftz` ale lanțului
`dot`; rândurile cu `bra` în listare 115 → 57. Pe fir, pe placă: tc face 32 + 16 citiri scalare la produs și 32 în trecerea a doua;
v4 face 8 + 4 și 8 citiri de câte 16 octeți — la fel de multe citiri „strict în zbor” (6), dar fiecare aduce de 4 ori mai mult.

| # | prezis | ieșit | verdict |
|---|---|---|---|
| U1 | 28 `ld.global….b128`, 0 scalare; 16 `st.global….b128`, 0 scalare | exact 28 și 16, nicio altă citire / scriere globală | CONFIRMATĂ |
| U2 | 64–88 registre, 0 locală / vărsări, aceleași `hmma` / `ffma.ftz`, 33 024 B | **56** (cât tc: NAK refolosește registrele între cele două faze; GREȘIT, mai jos), restul exact | GREȘITĂ (în bine) |
| U3 | 1 bloc pe SM la 64 KiB | 1 (memoria partajată e limita) | (calcul) |
| U4 | aceleași valori în memoria partajată ca tc, vectori aliniați și întregi, ≤ 3 numere peste diagonală | DA pe 6 forme mici (T 16 / 64 / 100 / 132 / 200 / 128, 1–3 capete, 1–2 eșantioane; toate blocurile și plăcile) și #513 (capetele 0 și 11), cu NaN peste diagonala lui P; exact 3 peste diagonală; 2 greșeli plantate prinse | CONFIRMATĂ (SIMULARE, T261) |
| U5 | dS final = al lui tc bit cu bit; scrieri doar în rândurile blocului, `j < T` | DA pe cele 6 forme (orice dP, orice `dot`); greșeala plantată (vectorul diagonal fără alegere) prinsă | CONFIRMATĂ (SIMULARE, T261) |
| U6 | aceiași biți prin procesorul de comenzi; nealiniatele refuzate fără scriere | DA pe 6 forme; T 130, d 6, dS + 4 refuzate fără scriere; din proba atenției iau v4 5 / 6 seturi (nu `t130-d64`) | CONFIRMATĂ (SIMULARE, T261) |
| U7 | pe placă aceleași cuvinte ca tc | — | NOT TESTED |
| U8 | `atentie-513-b4` ×1,2–1,8 față de tc | — | NOT TESTED |

**Ce rămâne pentru placă:** U7 și U8, în proba „vechi lângă nou” a lui `attn_back_tile0_tc` (PREDARE P2zs pasul 1): v4 după tc,
pe seturile aliniate, cuvânt cu cuvânt + timpii. Pragul de ×1,15 rămâne. Legarea în pasul mixt (alegerea v4 când `fits`) o fac
după rezultatul plăcii — azi nu schimb ce e predat în P2zs.
