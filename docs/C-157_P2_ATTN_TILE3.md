# C-157 punctul 2, după os-364 — `attn_*_3`: aceiași biți, citiri fără ramură

Programele `_2` (os-364) au ieșit identice bit cu bit cu cele vechi și de ×1,17–×1,77 mai rapide, dar comparația
(`docs/C-157_P2_COMPARATII.md`, os-364) a arătat că schimbarea F3 („citirile emise deodată”) n-a câștigat nimic: NAK pune fiecare
citire globală păzită de un `if` în zona ei (`bssy` / `bsync`) și **așteaptă rezultatul la ieșirea din zonă**, deci citirile au rămas
una după alta, câte o latență întreagă a memoriei. Remediul (predicția 4 de acolo): **citiri fără ramură** — adresa strânsă în
buffer, citirea mereu, apoi alegerea 0 / valoare. Aceleași sume în aceeași ordine → aceiași biți.

Câte un program, cu plan și predicții întâi. Primul: `attn_back_tile0_3` (cel mai mare câștig prezis: trecerea a doua are 32 de
citiri în șir pe placă).

## `attn_back_tile0_3` — planul și predicțiile (sesiunea 179, 23 nov 2026, scrise ÎNAINTE de cod)
Program nou **lângă** `attn_back_tile0` și `attn_back_tile0_2` (neschimbate). Pornește de la `attn_back_tile0_2.comp`: aceleași
argumente (cbuf-ul lui `attn_back`, modul 0), aceeași grilă [⌈T/64⌉, H, B], blocuri de 256, aceeași memorie partajată (33 024 B),
același model pe procesor (`attn_back_row_gpu`), fiecare lanț în aceeași ordine. Schimbări:
- **G1 — trecerea a doua (dS = P·(dP − dot)·scale) împărțită pe feluri de plăci:**
  - plăcile de **sub diagonală** (`jt < bx`): acolo `j < i0 ≤ i`, deci garda `j <= i` e mereu adevărată; citirile lui dS și P
    **fără ramură**, cu rândul strâns `ic = min(i, T − 1)`; scrierea doar dacă `i < T` (o scriere nu se așteaptă);
  - placa **diagonalei** (`jt == bx`): citirile la `(ic, min(j, ic))` (mereu în buffer), apoi `y = (i < T && j <= i) ? valoarea : 0,0`;
    scrierea dacă `i < T && j < T` (ca înainte);
  - plăcile de **deasupra diagonalei** (`jt > bx`): acolo `j > i` mereu, vechiul scria 0,0 după o ramură; acum **scrie 0,0 fără nicio
    citire** (aceeași valoare, +0).
- **G2 — urcarea plăcilor dO, V și P fără ramură:** adresa strânsă (`min(i, T − 1)`, `min(x, d − 1)`, `min(j, ic)`), citirea mereu,
  apoi `cond ? valoarea : 0,0` (aceeași valoare scrisă în `sa` / `sb` ca la `_2`). Tot câte 8 citiri, apoi cele 8 scrieri partajate.
- Restul (bucla lui dP, lanțul `dot`, scrierea lui dP în dS) = `attn_back_tile0_2`, neschimbat.
- De ce aceiași biți: fiecare valoare care intră într-o sumă sau într-o scriere e aceeași (aceeași adresă când condiția e adevărată,
  0,0 când e falsă); valorile citite la adresele strânse și aruncate nu intră nicăieri (nici NaN-urile de deasupra diagonalei lui P).

Predicții:
1. **Aceleași operații:** în listare 0 `mufu`, nicio atomică, nicio memorie locală; felurile de operații aritmetice ⊂ cele din
   `attn_back_tile0_2` plus alegeri / minime pe întregi (`sel`, `imnmx` sau ce scrie NAK); `fmul` / `fadd` doar în trecerea a doua.
2. **Registre ≤ 80** (3 blocuri de 256 pe SM, ca `_2`). Dacă ies peste, cobor citirile emise deodată în trecerea a doua (de la 16
   elemente = 32 de citiri la 8 elemente) și notez.
3. **Buclele (`nak_bucle.py`, coloana `sir`):** fiecare iterație de urcare a unei plăci (8 citiri) **1** așteptare a memoriei (la `_2`:
   8); trecerea a doua, plăcile de sub diagonală: **≤ 2** așteptări pe iterație (la `_2`: 32 pe fiecare placă, și deasupra diagonalei);
   0 zone `bssy` în buclele de urcare și în trecerea a doua de sub diagonală.
4. **SIMULARE:** emularea literală a GLSL-ului nou = `attn_back` modul 0 **bit cu bit** pe cele 10 forme din T175 / T182; lansarea =
   `attn_back_tile0` bit cu bit; NaN / 1e30 deasupra diagonalei lui P și NaN în dO / V dincolo de `d` sau de `T` (unde citirile
   strânse ajung) nu schimbă niciun bit. Greșeli plantate prinse: alegerea scoasă la placa P (valoarea strânsă intră în sumă), alegerea
   scoasă pe diagonala trecerii a doua, plăcile de deasupra diagonalei sărite (fără scrierea lui 0), memoria partajată veche în QMD.
5. **Pe placă (într-o probă „vechi lângă nou” ca os-364, mai târziu):** dS bit cu bit = `attn_back_tile0` (~85 %); timpul pe setul
   mare **~0,5–0,9 ms** față de 1,54 la `_2` (DEDUS: modelul pe bucle cu o așteptare pe iterație 0,50 → 0,20 ms, înmulțit cu factorul
   măsurat ×3,12 → ~0,63 ms; factorul ×3 rămâne NECUNOSCUT, de aici intervalul).

## Ce NU face
- Nu schimbă `attn_back_tile0`, `attn_back_tile0_2`, `train_step` sau proba „vechi lângă nou” din os-364; programul `_3` se leagă
  în probă după ce sunt gata toate trei `_3` (sau doar el, dacă fwd / back1 întârzie).
- Nimic nu rulează pe placă din sesiunea asta. Nicio scriere în registre.

## Surse
- `docs/C-157_P2_COMPARATII.md` (os-364, predicția 4), `tools/buget-antrenare/nak_bucle.py` (`waits`, `model364`).
- Ideea „citire strânsă + alegere în loc de ramură”: ghidul public CUDA (divergența ramurilor, „predication”); nicio linie copiată.

## Rezultatul pentru `attn_back_tile0_3` (sesiunea 179, SIMULARE, T188 `crates/nvidia-ga104/tests/sim_attn_back_tile0_3.rs`)
- **Binarul** (Mesa NAK 25.0.7 reconstruit în cloud; `attn_back_tile0_2` a ieșit **identic**, `9a8f9760…`): `attn_back_tile0_3.sm86.sass`
  SHA-256 **`d8f4ec408707d6a12ae32b153962a3a55742adb05bc1ac21f1fddaa7c83b611f`**, **68 de registre**, 33 024 de octeți partajați.
- **Am aflat la prima compilare (deci nu e predicție):** NAK emite instrucțiunile **în ordinea din sursă**. Cu „citește, apoi alege”
  scrise unul după altul (`v = citire; lo = cond ? v : 0`), alegerea stătea imediat după citire și o aștepta → tot 8 așteptări pe 8
  citiri, chiar fără ramură. Reparat în GLSL: întâi cele 8 citiri (în registre), apoi alegerile și scrierile partajate; în trecerea a
  doua întâi cele 32 de citiri (dS și P), apoi aritmetica. Prima compilare: 55 de registre; cea finală: 68.
- **Predicția 1: DA** — 0 `mufu`, nicio atomică, nicio memorie locală, 6 `bar.sync` (ca `_2`); 88 `ffma` (ca `_2`); `fadd` / `fmul`
  doar în trecerea a doua, aceleași feluri (`fadd.ftz` dS − dot, `fmul.ftz` P·x, `fmul.ftz` ·scale), de două ori mai multe (32 / 64 față
  de 16 / 32) pentru că trecerea a doua e scrisă de două ori (sub diagonală și diagonala); nou: 40 `sel`, 104 `imnmx` (adresele strânse).
- **Predicția 2: DA** — 68 ≤ 80 → 3 blocuri de 256 pe SM (72 × 256 × 3 = 55 296 ≤ 65 536), ca `_2`.
- **Predicția 3: PE JUMĂTATE.** Numărate cu `nak_bucle.py` (coloana nouă `nou` = latențe proaspete: o așteptare contează doar dacă
  citirea așteptată a fost emisă după așteptarea precedentă; pe programele vechi dă exact aceleași cifre ca `sir`):

  | buclă | `_2` (`sir`) | `_3` (`nou`) | predicție |
  |---|---|---|---|
  | urcarea dO / V / P, 8 citiri | 8 / 8 / 8 | **2 / 2 / 2** | 1 — ratat |
  | trecerea a doua, o placă sub diagonală (32 de citiri) | 32 | **7** | ≤ 2 — ratat |
  | zone `bssy` în urcare | 8 / 8 / 8 | **0 / 0 / 0** | 0 — DA |
  | zone `bssy` în trecerea a doua sub diagonală | 32 | **17** | 0 — ratat (garda scrierii `i < T`) |

  De ce mai multe decât 1–2: GA104 are doar **6 „tabele de scor”** (semnalele care spun că o citire a sosit); NAK dă fiecărei citiri
  câte unul, deci după 5 citiri în zbor a 6-a trebuie să aștepte prima. 8 citiri = 2 latențe, 32 = 7. Zonele `bssy` rămase păzesc
  doar **scrierile** (`if (i < T)`): o scriere nu se așteaptă, deci nu adaugă latențe (doar instrucțiuni).
- **Ce dă modelul pe bucle (DEDUS, `python3 tools/buget-antrenare/nak_bucle.py`, funcția `s179`):** `attn_back_tile0_2` 0,50 ms (din
  care memoria 0,36) → `attn_back_tile0_3` **0,18 ms** (memoria 0,06), ×2,8; cu factorul măsurat la `_2` (×3,12) → **~0,55 ms** pe setul
  mare (os-364: 1,54). Placa diagonalei din trecerea a doua (cod drept, fără buclă) nu e în model (1 placă din 8 pe bloc).
- **Predicția 4: DA, cu o greșeală plantată care nu e greșeală.** Emularea literală a noului GLSL (citiri la adresele strânse — un
  indice în afara bufferului ar opri testul —, apoi alegerile; trecerea a doua pe trei feluri de plăci) = `attn_back` modul 0 **bit cu
  bit** pe cele 10 forme (d = 1, 32, 40, 64; T = 5 … 200); lansarea = `attn_back_tile0` bit cu bit; NaN / 1e30 deasupra diagonalei lui
  P nu schimbă niciun bit. Prinse: alegerea scoasă pe diagonala trecerii a doua, plăcile de deasupra diagonalei sărite, QMD cu memoria
  partajată a lui `attn_back_tile0`, grila cu un bloc lipsă, mod 1, d = 65. **Neprinsă (și pe drept):** alegerea scoasă la placa P nu
  schimbă niciun bit — valorile pe care le păzește (j > i, rândurile i ≥ T) **nu sunt citite niciodată**: lanțul `dot` își păstrează
  garda `j <= i` pe diagonală, iar `dot`-ul rândurilor i ≥ T nu e folosit. La fel și la dO / V (coloanele x ≥ d nu intră în bucla lui dP,
  rândurile ≥ T nu ajung în dS). Alegerile de la plăci rămân (planul): țin intrările nefolosite la 0, ca înainte. Testul scrie
  observația (NoSelectOnP = programul, pe două forme).
- **Predicția 5 (pe placă): NEMĂSURATĂ.** Rămâne ~0,5–0,9 ms pe setul mare față de 1,54 (DEDUS; modelul de azi: ~0,55).
- `tools/check.sh`: **896 de teste** (+7 ignorate), format, clippy, build no_std — PASS.

**Ce e nou pentru placă:** nimic încă (programul nu e legat în probă). **Următorul pas:** `attn_fwd_tile2` → `attn_fwd_tile3` și
`attn_back_tile_2` → `attn_back_tile_3` cu aceeași rețetă (citirile întâi, alegerile după; câte 5 în zbor e plafonul), plan + predicții
întâi; apoi proba „vechi lângă nou” cu `_3` (ca os-364) și nota pentru OS.

## `attn_back_tile_3` — planul și predicțiile (sesiunea 180, 23 nov 2026, scrise ÎNAINTE de cod)
Program nou **lângă** `attn_back_tile` și `attn_back_tile_2` (neschimbate). Pornește de la `attn_back_tile_2.comp`: aceleași
argumente (cbuf-ul lui `attn_back`, modul 1: dQ, dK, dV), aceeași grilă [⌈T/64⌉, H, B], blocuri de 256, aceeași memorie partajată
(33 024 B), același model pe procesor (`attn_back_col_gpu`), fiecare lanț în aceeași ordine. Aici singurele citiri globale din bucle
sunt **urcările plăcilor** (buclele interioare citesc doar memoria partajată), deci se schimbă doar ele:
- **H1 — trecerea dQ (placa dS și placa K), câte 2 + 2 citiri ca la `_2`:** dS la `(tc, min(j, tc))` cu `tc = min(t, T − 1)`, K la
  rândul `min(j, T − 1)` și coloana `min(x, d − 1)`; întâi toate citirile, apoi alegerile `cond ? v : 0,0` și scrierile partajate
  (rețeta aflată la `attn_back_tile0_3`: NAK emite în ordinea din sursă).
- **H2 — trecerile dK / dV (placa dS sau P, placa Q sau dO), câte 4 + 4 ca la `_2`:** dS / P la `(ic, min(t, ic))` cu
  `ic = min(i, T − 1)`, Q / dO la rândul `ic` și coloana `min(x, d − 1)`; alegerea între dS și P (și între Q și dO, cu pasul de rând
  3C sau C) se face **o dată pe trecere**, pe adresă (`uint64_t`), nu cu `if (pass == 0)` lângă fiecare citire.
- Restul (buclele interioare F2 / F4, scrierile rezultatelor cu garda lor) = `attn_back_tile_2`, neschimbat.
- De ce aceiași biți: fiecare valoare scrisă în `sa` / `sb` e aceeași ca la `_2` (aceeași adresă când condiția e adevărată, 0,0 când
  e falsă); valorile citite la adresele strânse și aruncate nu intră nicăieri.

Predicții:
1. **Aceleași operații:** 0 `mufu`, nicio atomică, nicio memorie locală, 6 `bar.sync` (ca `_2`); același număr de `ffma` ca `_2`
   (doar buclele interioare le au); nou doar `sel` / `imnmx` (și aritmetica pe 64 de biți a adresei alese).
2. **Registre ≤ 80** (`_2` are 79: risc real). Dacă ies peste 80, cobor urcarea dK / dV de la 4 + 4 la 2 + 2 și notez.
3. **Buclele (`nak_bucle.py`, coloana `nou`):** urcarea dQ (4 citiri) **1** latență pe iterație (`_2`: 4); urcarea dK / dV (8 citiri)
   **2** (`_2`: 8, plafonul celor 6 tabele de scor); **0** zone `bssy` în ambele bucle de urcare.
4. **SIMULARE:** emularea literală a GLSL-ului nou = `attn_back` modul 1 **bit cu bit** pe cele 10 forme din T183; lansarea =
   `attn_back_tile` bit cu bit; NaN deasupra diagonalei lui dS / P și NaN în Q / K / dO dincolo de `d` nu schimbă niciun bit. Greșeli
   plantate prinse: alegerea scoasă la placa dS a trecerii dQ (dacă intrările ei ajung în sume — altfel o notez ca la `_3` back0),
   placa P în loc de dS la trecerea dK, pasul de rând C în loc de 3C la Q, QMD cu memoria partajată greșită.
5. **Modelul pe bucle (DEDUS):** `attn_back_tile_2` → `_3` scade doar partea de memorie a urcărilor (la `_2` ~8 + 8 latențe pe placă
   → ~2 + 2); pe placă estimat **~1,6–2,0 ms** pe setul mare față de 2,25 (os-364) — câștig mai mic decât la back0, pentru că aici
   buclele interioare (din memoria partajată) cântăresc mai mult. Cifra exactă o scrie `nak_bucle.py` după compilare.

Nimic nu rulează pe placă din sesiunea asta; `attn_back_tile`, `attn_back_tile_2`, `train_step` și proba os-364 rămân neschimbate.

## Rezultatul pentru `attn_back_tile_3` (sesiunea 180, SIMULARE, T189 `crates/nvidia-ga104/tests/sim_attn_back_tile_3.rs`)
- **Binarul** (Mesa NAK 25.0.7 reconstruit în cloud; `attn_back_tile_2` a ieșit **identic**, `dc50f7d4…`):
  `attn_back_tile_3.sm86.sass` SHA-256 **`67c8d9a564f6b5e3fde212262c74b456918920a8664462865a256123ba085a9c`**, **80 de registre**,
  33 024 de octeți partajați.
- **Predicția 2: DA, după ramura de rezervă din plan.** Cu urcarea dK / dV câte 4 + 4 (ca `_2`) au ieșit **90** de registre (peste 80
  → doar 2 blocuri pe SM); coborâtă la 2 + 2, cum scria planul: **80** (80 × 256 × 3 = 61 440 ≤ 65 536 → 3 blocuri pe SM, ca `_2`).
- **Predicția 1: DA** — 0 `mufu`, nicio atomică, nicio memorie locală, aceleași bariere ca `_2`; **256 `ffma`** (= `_2`); 1 579 de
  operații față de 1 721 (cele două ramuri `pass == 0` / `pass == 1` de lângă fiecare citire au dispărut); nou: 14 `imnmx`, 99 `sel`
  (88 la `_2`).
- **Predicția 3: DA** (numărate cu `nak_bucle.py`, coloana `nou`):

  | buclă | `_2` | `_3` | predicție |
  |---|---|---|---|
  | urcarea dQ, 4 citiri pe iterație (8 iterații pe placă) | 4 latențe | **1** | 1 — DA |
  | urcarea dK / dV, pe placă | 4 × 8 = 32 latențe | **8 × 1 = 8** (2 + 2 citiri, 8 iterații) | 4 × 2 = 8 — DA (aceeași cifră pe placă) |
  | zone `bssy` în urcări | 4 / 8 | **0 / 0** | 0 — DA |

  Buclele interioare (din memoria partajată) au exact aceleași instrucțiuni ca la `_2` (105 / 107 / 261 / 444).
- **Predicția 5 (DEDUS, `python3 tools/buget-antrenare/nak_bucle.py`, funcția `s180`):** `attn_back_tile_2` model 0,60 ms (memoria
  0,33) → `attn_back_tile_3` **0,43 ms** (memoria 0,08; acum hotărăște emisia instrucțiunilor, nu memoria), ×1,40; cu factorul măsurat
  la `_2` (×3,76) → **~1,60 ms** pe setul mare față de 2,25 (os-364). În intervalul prezis (1,6–2,0).
- **Predicția 4: DA.** Emularea literală (fiecare citire la adresa ei strânsă — un indice în afara bufferului ar opri testul —, apoi
  alegerea) = `attn_back` modul 1 **bit cu bit** pe cele 10 forme; lansarea = `attn_back_tile` (și `_2`) bit cu bit; NaN / 1e30 deasupra
  diagonalei lui dS și P fără efect; −0 al ultimei plăci păstrat. Prinse: P în loc de dS la trecerea dK, Q citit cu pasul C, QMD cu
  memoria partajată a lui `attn_back_tile`, grila cu un bloc lipsă, blocul greșit, mod 0, d = 65. **Observație (ca la back0):** toate
  alegerile scoase dau **aceiași biți** — valorile pe care le păzesc nu sunt citite niciodată (plăcile diagonalei își păstrează
  gărzile, celelalte au doar intrări valide, acumulatorii rândurilor ≥ T sau coloanelor ≥ d nu se scriu). Alegerile rămân (planul).
- `tools/check.sh`: vezi `docs/IMPLEMENTATION_STATUS.md`, sesiunea 180.

**Ce e nou pentru placă:** nimic încă (programul nu e legat în probă). **Următorul pas:** `attn_fwd_tile2` → `attn_fwd_tile3` (plan +
predicții întâi), apoi proba „vechi lângă nou” cu cele trei `_3` (ca os-364) și nota pentru OS.

## `attn_fwd_tile3` — planul și predicțiile (sesiunea 181, 23 nov 2026, scrise ÎNAINTE de cod)
Program nou **lângă** `attn_fwd_tile` și `attn_fwd_tile2` (neschimbate). Pornește de la `attn_fwd_tile2.comp`: aceleași argumente
(cbuf-ul lui `attn_fwd`), aceeași grilă [⌈T/64⌉, H, B], blocuri de 256, aceeași memorie partajată (37 888 B), același model pe
procesor (`attn_fwd_tile_gpu`), fiecare sumă în aceeași ordine. Rețeta de la `_3` back (citirile întâi, alegerile după):
- **J1 — urcarea plăcilor Q, K și V:** rândul strâns `min(i, T − 1)` / `min(j, T − 1)`, coloana `min(e, d − 1)`; cele 16 citiri ale
  firului emise toate, apoi alegerile `cond ? v : 0,0` și cele 16 scrieri partajate (aceleași valori în `qs` / `ks` ca la `_2`).
- **J2 — faza B (exp, P recitit și rescris):** cele 16 citiri ale lui P pe placa `jt` la `(ic, min(j, ic))` cu `ic = min(i, T − 1)`,
  toate întâi; apoi, în ordinea de la `_2` (k afară, a înăuntru), `x = exp(v − rowm)` calculat mereu, `ps[a] = cond ? ps[a] + x :
  ps[a]` și scrierea lui `x` doar dacă `i < T && j <= i` (o scriere nu se așteaptă).
- **J3 — faza C, citirea lui P (P · inv, P rescris):** plăcile de **sub diagonală** (`jt < bx`: `j < i0 ≤ i` și `j < T` mereu): citirea
  la `(ic, j)` fără ramură, `pv = (i < T) ? v · inv : 0,0`, scrierea dacă `i < T`; placa **diagonalei**: citirea la `(ic, min(j, ic))`,
  `pv = (i < T && j <= i) ? v · inv : 0,0`, scrierea dacă `i < T && j < T` (ca înainte). Plăcile de deasupra diagonalei: neschimbate
  (scriu 0,0, nicio citire). Apoi placa V (J1).
- Restul (bucla fazei A, maximul pe rând, sumele din `red`, `rinv`, bucla fazei C sub / pe diagonală, scrierea lui `att`) = `_2`.
- De ce aceiași biți: fiecare valoare care intră într-o sumă, într-un maxim sau într-o scriere e aceeași ca la `_2` (aceeași adresă când
  condiția e adevărată, 0,0 sau suma neatinsă când e falsă); valorile citite la adresele strânse și aruncate (și `exp`-urile lor) nu
  intră nicăieri. Citirile strânse pot nimeri elementul altui fir (diagonala) în timp ce acela îl rescrie: valoarea e aruncată.

Predicții:
1. **Aceleași operații:** 16 `mufu.ex2` + 1 `mufu.rcp` (ca `_2`: exp-urile fazei B calculate mereu, tot 16 pe placă), nicio atomică,
   nicio memorie locală, aceleași bariere; același număr de `ffma` ca `_2`; nou doar `sel` / `imnmx` (și `fsel` / `fmnmx` dacă așa
   scrie NAK alegerile).
2. **Registre ≤ 128** (`_2`: 117; 2 blocuri de 256 pe SM). Dacă ies peste, cobor urcările de la 16 la 8 citiri deodată și notez.
3. **Buclele (`nak_bucle.py`, coloana `nou`; la `_2`: 16 / 16 / 32):** bucla fazei A (placa K + S scris) **≤ 4** latențe; faza B
   **≤ 4**; bucla exterioară a fazei C (P + placa V) **≤ 8**. Zonele `bssy` rămân doar în jurul scrierilor (nicio citire globală într-o
   zonă `bssy`).
4. **SIMULARE:** emularea literală a GLSL-ului nou (citiri la adresele strânse — un indice în afara bufferului oprește testul —, apoi
   alegerile) = `attn_fwd_tile_gpu` **bit cu bit** (P și att) pe cele 10 forme din T173 / T181; lansarea = `attn_fwd_tile2` bit cu bit;
   NaN în P deasupra diagonalei înainte de lansare (gunoiul unui pas vechi) nu schimbă niciun bit. Greșeli plantate prinse: alegerea
   scoasă la suma din faza B (dacă intrările ei ajung în sumă — altfel o notez, ca la back), alegerea scoasă pe diagonala fazei C,
   QMD cu memoria partajată greșită, grila cu un bloc lipsă.
5. **Pe placă (proba „vechi lângă nou”, mai târziu):** P și att bit cu bit = `attn_fwd_tile` (~85 %); timpul pe setul mare
   **~0,7–1,1 ms** față de 1,53 la `_2` (DEDUS: modelul cu o așteptare pe iterație dădea 0,54 → 0,27 ms, × factorul măsurat ×2,82 →
   ~0,76; cifra după compilare o scrie `nak_bucle.py`, funcția `s181`).

Nimic nu rulează pe placă din sesiunea asta; `attn_fwd_tile`, `attn_fwd_tile2`, `train_step` și proba os-364 rămân neschimbate.
