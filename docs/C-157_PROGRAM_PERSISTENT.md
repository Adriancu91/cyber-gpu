# C-157 — studiul „programului persistent” pe un strat (nota centrului 25 nov 11:55, cererea 2) — plan + predicții, scrise ÎNAINTE de calcule

**Data:** 25 noiembrie 2026, ~20:25 (ora României), sesiunea Cyber-GPU 236, în cloud, FĂRĂ placă. Totul de aici e
**DEDUS** (aritmetică pe timpii pe care i-a MĂSURAT OS-ul pe placă) sau **SIMULARE**; nimic nu rulează pe RTX 3070 Ti.

**Cererea centrului (11:55, studiul `idei/2026-11-25-centru-franele-altora.md`, punctul 2):** „studiu «program persistent» pe
un strat (fără limita TDR la noi) — după `mm_tc` cu ocupare”. Ideea din tabel: „un singur program care rămâne pe placă și
face un strat / tot pasul, cu datele ținute în memoria rapidă între etape (mai puține drumuri prin VRAM)”. Ordinea centrului
(ocuparea → `mm_tc` → copierea în paralel → persistent) e la ultimul punct: copierea s-a terminat în sesiunea 235.

## 1. Ce avem deja (citit în repo, nu presupus)

- **Un program persistent există deja, pentru generare:** `tools/nak/token.comp` (C-87, sesiunea 65, 7 nov) — toată lista
  unui simbol (195 de operații la #513) într-o singură lansare, G grupuri de 32 de fire, **barieră pe toată placa** între
  operații (`atom.add.strong.sys` + `membar.sc.gpu`), date de lucru `coherent` (ocolesc L1). Partea driverului:
  `nvidia_ga104::kronos_token` (lista de lucru, `groups_for`). În SIMULARE = cele 35 de lansări separate bit cu bit.
  **COMPILAT, NU RULAT** — bariera și coerența nu sunt dovedite pe placă (simulatorul execută lista în ordine).
- **Profilul pasului #513 pe placă** (os-431, `tests/rapoarte/os431-pas.txt`): 255,9 ms, 483 de lansări; înmulțirile
  `mm_tv_128x128` 87,2 %, atenția ~8 %, AdamW 1,9 %, restul (colsum, gelu, vadd, LayerNorm, xent, emb) ~2,7 %.
- **Timpii pe lansare la generare** (os-346, `lipita-ln-32`, MĂSURAT): fc 18,78 µs, qkv 14,69, proj 11,90, out 6,05,
  atenția 20,77, ln1/ln2 4,03/3,87, logit 88,0; placa 1 071 µs pe simbol.
- **Costul unei lansări** (MĂSURAT): os-392 −26 µs pentru 6 lansări lipite (~4,3 µs pe lansare, la `antrenare mini`);
  LayerNorm-ul nou (os-331) ~4 µs pe lansare.
- **Modelul valurilor** (`wave_model`, os-423 / os-431, ≤ 2,5 %): o lansare cu g blocuri pe 48 de SM-uri rulează în
  ⌈g / 48b⌉ valuri; formele 1024 × 768 au 48 de blocuri pe `128x128` → un singur val, jumătate din locuri goale la 2 blocuri pe SM.

## 2. Ce întreabă studiul (cinstit: ce ar câștiga, înainte de muncă grea)

Un program persistent câștigă din trei surse, și doar din ele:
1. **Golurile dintre lansări** (~4 µs fiecare), înlocuite cu bariere pe toată placa (și ele costă: un contor atomic + așteptare).
2. **Drumurile prin VRAM** ale rezultatelor intermediare — doar dacă rezultatele încap în memoria rapidă (L2 = 4 MiB pe GA104;
   memoria partajată ≤ 100 KiB × 48 SM ≈ 4,7 MiB; registrele 256 KiB × 48 = 12 MiB, dar doar pentru firul care le-a scris).
3. **Cozile valurilor**: când o înmulțire nu umple ultimul val, un planificator în program poate da locurile goale altei
   lucrări **independente** (la întoarcere: dX și dW ale aceluiași strat nu depind una de alta).

## 3. Predicțiile (scrise înainte de calcule)

| # | Predicție | Cum se verifică |
|---|---|---|
| P1 | La antrenare (lot 2 × 512), rezultatele intermediare ale unui strat (qkv 9 MiB, stratul ascuns 12 MiB) **nu încap** în L2 și nici în memoria partajată a plăcii → „datele ținute în memoria rapidă” merge doar pe lanțuri pe rânduri (deja „lipirile”) | aritmetică pe formele planului #513 |
| P2 | Golurile lansărilor la antrenare ≤ 1 % din pas (483 × ~4,3 µs ≈ 2 ms din 256) | profilul os-431 |
| P3 | Operațiile mărunte (nu înmulțiri, nu atenție, nu AdamW) ≤ 3 % din pas | profilul os-431 |
| P4 | Cea mai mare sursă la antrenare = **cozile valurilor** formelor dX cu 48 de blocuri; o lansare „dublă” dX + dW (fără program persistent) ar recupera **5–15 %** din pasul os-431 și **2–8 %** peste alegerea pe formă la 64 KiB | modelul valurilor |
| P5 | La generare golurile lansărilor = **30–40 %** din placă (~90 de lansări × 3,5–4 µs din 1 071 µs) → acolo persistentul (`token.comp`) are sens, nu la antrenare | timpii os-346 |
| P6 | Plafonul generării (greutățile Q8 citite o dată pe simbol la 608 GB/s) e de ~5–7 ori sub placa de azi | aritmetică |

## 4. Ce se face în sesiune

1. Modulul `nvidia_ga104::persistent_study` (doar adăugare, aritmetică pură, totul DEDUS): bugetul unui strat la antrenare
   (P1–P4) și la generare (P5, P6), din profilul os-431, planul #513 și timpii os-346.
2. Testul T242 (pe procesor) care fixează cifrele și verifică predicțiile.
3. Rezultatul în acest document (secțiunea 5) + recomandarea: ce urmează și ce NU merită.

Nimic nou pentru placă în această sesiune.

## 5. Rezultatul (sesiunea 236, ~20:43; T242, 6/6 pe procesor; cifrele = DEDUS din timpii MĂSURAȚI de OS)

Cod: `nvidia_ga104::persistent_study` (doar adăugare, aritmetică pură), testul `tests/c157_persistent.rs`.

| # | Predicție | Ieșit | Verdict |
|---|---|---|---|
| P1 | rezultatele unui strat nu încap în memoria rapidă | memoria rapidă ≈ 4,7 MiB (48 × 100 KiB partajată; L2 4 MiB); încap doar ieșirea LayerNorm și a atenției (3 MiB); **qkv 9 MiB, stratul ascuns 12 MiB nu încap**; la lotul 6 nu încape nimic | CORECTĂ |
| P2 | golurile lansărilor ≤ 1 % | 483 × 4,3 µs = **2,1 ms = 0,8 %** din 255,9 ms | CORECTĂ |
| P3 | operațiile mărunte ≤ 3 % | colsum + colsum_part + gelu_tr + vadd + LayerNorm + xent + emb + gnorm = **7,3 ms = 2,8 %** | CORECTĂ |
| P4a | lansarea „dublă” dX + dW, peste os-431 (`128x128` peste tot): 5–15 % | **19,1 %** (podeaua: −49 ms din 137 ms ai celor 48 de înmulțiri dX/dW); formele dX cu 48 de blocuri lasă 40–57 % din locuri goale; modelul reface cele 6 etichete tipărite de os-431 la 2,2 % (120,8 față de 118,2 ms) | **RATATĂ** (mai mult decât prezis) |
| P4b | peste alegerea pe formă la 64 KiB: 2–8 % | **3,1 %** (−8,1 ms): alegerea pe formă ia deja −43 din cele −49 ms | CORECTĂ |
| P5 | la generare golurile = 30–40 % din placă | 87 de lansări × ≤ 3,87 µs = **≤ 336 µs = 31 %** din 1 071 µs | CORECTĂ |
| P6 | plafonul generării de ~5–7 ori sub placa de azi | greutățile Q8 ale lui #513 citite o dată la 608 GB/s = **170 µs** → placa de azi = **×6,3** podeaua; pe strat 80,1 µs față de 12,4 µs | CORECTĂ |

### Ce înseamnă (pe scurt)

1. **La antrenare, un program persistent pe strat NU merită acum.** Ce poate câștiga singur (goluri + operații mărunte) e
   ≤ 3,6 % din pas, iar „datele ținute în memoria rapidă” nu se poate: rezultatele mari ale stratului (9–12 MiB) sunt de 2–3 ori
   mai mari decât toată memoria rapidă a plăcii. Lanțurile pe rânduri (LayerNorm → înmulțire, înmulțire → GELU / adunare) se
   câștigă mai ieftin prin **lipiri** (le avem deja: `ln_back_res`, `mm_train_gelu`, `mm_tc_gelu`).
2. **Partea bună a ideii (planificarea locurilor goale) se ia fără program persistent:** întâi **alegerea pe formă** (sesiunea
   232, nota `C-165_valuri-alegerea-64k.md`: −43 ms), apoi, opțional, o **lansare dublă** dX + dW (același program, blocurile
   împărțite între două înmulțiri independente) pentru încă ~3 % (−8 ms, podea). Același lucru se va repeta pe `mm_tc_v4`
   (pasul mixt), unde înmulțirile devin mai scurte și cozile cântăresc mai mult — de măsurat după os-438.
3. **La generare, persistentul are sens:** ~31 % din timpul plăcii e cost fix de lansare, iar placa e de 6,3 ori peste podeaua
   lățimii de bandă. Programul există deja (`token.comp`, C-87, compilat, nerulat). Ce lipsește e **dovada barierei pe toată
   placa** pe RTX 3070 Ti (toate grupurile pe placă deodată + coerența memoriei fără L1). Primul pas propus: o **probă mică a
   barierei** (un program care doar trece de N bariere și numără), gata de legat în SIMULARE, înainte de a pune `token.comp` pe placă.
   Estimare (DEDUS): dacă bariera costă ~1–2 µs, un simbol #513 ar câștiga ~150–250 µs pe placă (1 071 → ~820–920 µs).

### Ce NU e dovedit

- Podeaua lansării duble presupune blocuri împachetate perfect; un planificator real pierde ceva. E plafonul, nu predicția.
- Costul unei bariere pe toată placa pe GA104 nu e măsurat (nici de noi, nici citit într-o sursă oficială).
- Costul fix al unei lansări la generare e luat ca plafon (cea mai mică lansare măsurată, 3,87 µs, care face și ea puțin lucru).

**Ce e nou pentru placă:** nimic. **Nicio schimbare incompatibilă** (un modul nou și un test).
