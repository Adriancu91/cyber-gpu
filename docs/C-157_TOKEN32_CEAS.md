# C-157 p.2 — unde stă „restul” lui `token32_v2`: programul cu ceas `token32_v2c` (plan + predicții, scrise ÎNAINTE de cod)

Sesiunea Cyber-GPU 274, 27 nov 2026, în cloud, **fără placă**. Nimic de aici n-a rulat pe placă. Pornește de la
`docs/C-157_TOKEN32_V2_PLACA.md` §3: la G = 768 placa face 1,668 ms pe simbol, iar **~0,67 ms nu sunt explicate** (multiplicatorul
muncii m = 1,94 față de lista obișnuită, după ce am scos pornirea și cele 87 de bariere de 3,11 µs).

## 1. Ce vrem să aflăm

Un singur lucru: **în ce operații stau cele ~0,67 ms** și dacă sunt muncă (grupurile lucrează mai încet decât lansările separate),
dezechilibru (unele grupuri termină mult după altele) sau barieră (după ce a terminat ultimul grup).

Bănuielile (niciuna verificată):
- **B1 — „ciocănitul” barierei.** În `grid_sync`, firul 0 al fiecărui grup care a terminat citește contorul **în buclă cu adunări
  atomice** pe aceeași adresă. La G = 768, sute de grupuri ciocănesc o singură felie a L2 cât timp celelalte încă lucrează; citirile
  greutăților care trec prin acea felie așteaptă → înmulțirile se lungesc. Proba barierei (os-459 / os-465) a măsurat bariera
  **fără** muncă în paralel, deci n-ar fi văzut asta.
- **B2 — operațiile făcute de un singur grup.** `layernorm256` (25 pe simbol) și `argmax` (1) rulează pe un singur grup de 32 de fire
  (în lansarea separată: 256 de fire); celelalte 767 de grupuri ciocănesc între timp.
- **B3 — împărțirea fixă a rândurilor.** Grupul g ia blocurile g, g + G, …; la lansările separate placa dă blocul următor primului
  warp liber. La rânduri ne-multiplu de 768 (logit-urile) sau la timpi inegali pe rând, ultimul grup ține bariera.

## 2. Programul cu ceas (`token32_v2c`)

`token32_v2` neschimbat ca aritmetică (aceleași operații în aceeași ordine), plus **numai citiri de ceas și scrieri într-o zonă
separată** (nicio scriere în registre ale plăcii, nicio memorie nouă citită):
- ceasul = numărătorul de cicluri al SM-ului (`clock2x32ARB`, `CS2R SR_CLOCKLO` — de verificat în listare); **o durată se măsoară
  doar pe același SM**, deci comparăm doar diferențe luate de același grup (fără ceas „comun” între SM-uri);
- adresa zonei: un câmp nou în bufferul constant, **0x40** (`CBUF_CLOCK`); cuvintele (u32):
  - `[k·G + g]` = ciclurile muncii grupului g la operația k (de la începutul buclei pe blocuri până la sfârșitul ei);
  - `[n·G + k]`, k = 0..n−1 = ceasul grupului 0 la începutul operației k (după barieră); `[n·G + n]` = la sfârșitul ultimei;
  - `[n·G + n + 1 + g]` = câte citiri ale contorului a făcut grupul g în toate barierele simbolului (ciocănitul, B1);
  - în total n·G + n + 1 + G cuvinte (n = 88, G = 768: **68 441 de cuvinte, ~267 KiB**).
- Fereastra operației k (văzută de grupul 0) = începutul lui k + 1 − începutul lui k = munca celui mai lent grup + bariera.
- Driverul (Rust, la noi): citește zona, o împarte pe feluri (ln1, qkv, atenție, out, ln2, fc, proj, ln final, logit, argmax,
  embed) și scrie rândurile `kt ceas …`; ciclurile se trec în µs cu frecvența dedusă (suma ferestrelor / timpul plăcii măsurat).

## 3. Predicțiile (scrise înainte; se judecă pe rularea OS-ului la G = 768, lângă `token32_v2` din același lanț)

| # | Ce | Predicție | Încredere |
|---|---|---|---|
| C1 | biții | aceleași 16 simboluri, amprenta `0x6c8501f0`, contorul (n − 1) · G, ca `token32_v2` | mare |
| C2 | costul ceasului | placa pe simbol cu ceas ≤ **+3 %** față de `token32_v2` din același lanț (~1,67 → ≤ 1,72 ms) | medie |
| C3 | ceasul e cel așteptat | suma ferestrelor (cicluri) / timpul plăcii = **1,4–2,0 GHz** (ceasul SM-ului sub cererea de viteză) | medie |
| C4 | bariera „curată” (fereastra − munca celui mai lent grup, însumată) | **0,14–0,40 ms** pe simbol (87 × 3,11 µs = 0,27 ms) | medie |
| C5 | unde e restul | **în munca înmulțirilor** (qkv, out, fc, proj, logit): munca celui mai lent grup, însumată pe aceste feluri, ≥ **0,40 ms** peste munca lor din lista obișnuită fără porniri (~0,51 ms: qkv 129, fc 178, proj 97, out 27, logit 83 µs; DEDUS din os-468) → B1 | mică |
| C6 | operațiile unui singur grup (B2) | `layernorm256` × 25 + `argmax`: împreună **0,05–0,20 ms** (nu tot restul) | mică |
| C7 | ciocănitul | media citirilor contorului pe grup și pe barieră: **≥ 20** (grupurile așteaptă mult mai mult decât un drum la L2) | mică |
| C8 | atenția | fereastra însumată a celor 12 atenții **≤ ×1,3** față de lista obișnuită fără porniri (~0,20 ms) | medie |

**Cum se citește rezultatul:** dacă C5 + C7 ies → pasul următor e o barieră care ciocăne mai rar (o pauză între citiri sau citiri
obișnuite în loc de adunări atomice), cu predicții noi. Dacă C6 iese mare → `layernorm256` / `argmax` împărțite pe mai multe grupuri.
Dacă munca e egală între grupuri dar fereastra mare → B3 / bariera. Orice ar ieși, e o **măsurătoare**, nu un câștig: `token32_v2c`
nu se folosește la generare.

## 4. Pașii

1. Programul `tools/nak/token32_v2c.comp` + compilarea NAK; în listare: aceleași operații aritmetice ca `token32_v2`, plus numai
   citiri de ceas și scrieri `st.global` în zona ceasului.
2. Rust (doar adăugări): modulul `nvidia_ga104::token_clock` (`TOKEN32_V2C`, `CBUF_CLOCK`, `token_cbuf_clock`, `clock_words`,
   `token32_v2c_qmd_wide`, împărțirea pe feluri + rândurile de jurnal), modelul în `gpu-sim` (SIMULARE: lista rulată ca la v2,
   ceasul umplut cu cifre **inventate, marcate SIMULARE**, doar ca să verificăm citirea zonei).
3. Nota pentru OS (PREDARE): o opțiune nouă, oprită implicit, care rulează `token32_v2c` la G = 768 după `token32_v2`.

## 5. Ce a ieșit (sesiunea 274, după predicții)

- Compilat (Mesa NAK 25.0.7; `token32_v2` reconstruit identic `5f4d22a0…`): **7 570 de instrucțiuni** (v2: 7 438), **99 de registre**
  (v2: 92), 0 memorie locală; SHA-256 `e899fa7c…6c18206`. Ceasul iese `cs2r sr[0x50]` / `sr[0x51]` (numărătorul de cicluri al SM-ului,
  citit sus–jos–sus pentru 64 de biți), o dată la începutul și o dată la sfârșitul operației.
- Listarea (T288): aceleași numere de `ffma`, `fadd`, `fmul`, `mufu`, `fsetp`, `fmnmx`, `i2f`, `f2i`, `shfl`, `fswzadd` ca v2; aceleași
  citiri (`.strong.sys`, `.constant`, b128, memoria partajată); bariera aceeași (4 `cctl`, 2 adunări atomice); 4 scrieri `st.global` în
  plus (zona ceasului). Concluzia „aceiași biți ca v2” rămâne DEDUSĂ; pe placă o decide C1.
- **Atenție la QMD:** 99 de registre ≠ 92 → `token_clock::token32_v2c_qmd_wide`, nu `kronos_token::token32_qmd_wide`. 16 blocuri pe SM
  încap în continuare (G = 768 pe 48 de SM).
- SIMULARE (T288): un simbol al listei de azi + argmax la lățimea lui Kronos, G = 768 = lansările separate bit cu bit; zona ceasului
  (cifre **inventate** de model) citită exact: ferestrele, munca cea mai lungă, bariera, citirile contorului, rândurile `kc`.
- Nimic rulat pe placă. PREDARE: `integration/cyber-universe/C-157_P2zzq_token32-v2c-ceas.md`.

## 6. Pe placă (sesiunea 276)

os-495 (ciclul 27 nov 18:06): C1, C2, C3, C4, C7 CONFIRMATE; C5, C6, C8 NECONFIRMATE — restul stă în LayerNorm făcut de un singur
grup (+0,36 ms) și în atenția pe 12 grupuri (+0,15 ms), nu în înmulțiri. Comparația: `docs/C-157_TOKEN32_V2C_PLACA.md` (T291).
